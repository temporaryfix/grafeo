//! Shared exact DISTINCT ownership for pull and push execution.
//!
//! Keys use the grouping/index structural relation, encoded by the common
//! semantic-key codec. Original values and graph provenance remain witnesses.

use super::{OperatorError, OperatorResult};
use crate::execution::value_codec::{CodecLimits, CounterSortScratch, MeasuredSemanticKey};
use crate::execution::{DataChunk, QueryResourceContext, QueryResourceContextError, ValueVector};
use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryGrantError};
use grafeo_common::types::{LogicalType, Value};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn overflow() -> OperatorError {
    MemoryGrantError::ArithmeticOverflow {
        current_bytes: usize::MAX,
        additional_bytes: 1,
    }
    .into()
}
fn add(a: usize, b: usize) -> Result<usize, OperatorError> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, OperatorError> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn context_error(error: QueryResourceContextError) -> OperatorError {
    match error {
        QueryResourceContextError::Memory(error) => error.into(),
        other => OperatorError::Execution(other.to_string()),
    }
}
fn allocation(error: std::collections::TryReserveError) -> OperatorError {
    OperatorError::ResidentContainerAllocation {
        container: "exact DISTINCT",
        source: error,
    }
}
fn io(error: std::io::Error) -> OperatorError {
    if error.kind() == std::io::ErrorKind::OutOfMemory {
        return OperatorError::ResidentAllocation(error.to_string());
    }
    #[cfg(feature = "spill")]
    {
        OperatorError::from_spill_io_error(error)
    }
    #[cfg(not(feature = "spill"))]
    {
        OperatorError::Execution(error.to_string())
    }
}
fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<(), OperatorError> {
    if count <= values.capacity() {
        return Ok(());
    }
    values
        .try_reserve_exact(count - values.len())
        .map_err(allocation)?;
    if values.capacity() > count {
        return Err(OperatorError::ResidentContainerInvariant {
            container: "exact DISTINCT vector",
            message: "allocator exceeded admitted capacity",
        });
    }
    Ok(())
}

/// Canonical grouping-key bytes retained under their originating query grant.
///
/// The shared semantic-key codec preserves structural grouping equality, including
/// signed zero, maps and counters. Original values remain separate witnesses.
#[derive(Debug)]
pub struct AccountedSemanticKey {
    bytes: Vec<u8>,
    _grant: MemoryGrant,
}

type Key = AccountedSemanticKey;

impl AccountedSemanticKey {
    /// Borrows the canonical key without copying its allocation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the reservation retaining the encoded allocation.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self._grant.size()
    }

    /// Transfers the encoded allocation together with its reservation.
    ///
    /// The receiver must retain the grant until the bytes are dropped or their
    /// allocation is transferred into another admitted owner. Any simultaneous
    /// conversion allocation needs a separate reservation before allocation.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, MemoryGrant) {
        (self.bytes, self._grant)
    }
}

/// Encodes a structural grouping key through the existing exact DISTINCT codec.
///
/// Borrowed input remains owned by its caller. Encoding and counter-sort scratch
/// are admitted before allocation; scratch is released before the key returns.
///
/// # Errors
/// Returns cancellation, admission, allocation or bounded-codec errors before
/// publishing a key. No borrowed input is retained on success or failure.
pub fn encode_accounted_semantic_key(
    resources: &QueryResourceContext,
    values: &[Value],
) -> Result<AccountedSemanticKey, OperatorError> {
    resources.cancellation_token().check()?;
    let key = encode_key(resources, values)?;
    resources.cancellation_token().check()?;
    Ok(key)
}

fn encode_key(resources: &QueryResourceContext, values: &[Value]) -> Result<Key, OperatorError> {
    let limits = CodecLimits::format_max();
    let semantic_key = MeasuredSemanticKey::new(values, limits).map_err(io)?;
    let measured = semantic_key.measurement();
    let scratch_bytes = CounterSortScratch::requested_capacity_bytes(
        measured.counter_sort_entries,
        measured.counter_sort_key_bytes,
    )
    .map_err(io)?;
    let _scratch_grant = if scratch_bytes == 0 {
        None
    } else {
        Some(
            resources
                .try_allocate(scratch_bytes)
                .map_err(context_error)?,
        )
    };
    let grant = resources
        .try_allocate(measured.encoded_bytes)
        .map_err(context_error)?;
    let mut encoded = Vec::new();
    reserve(&mut encoded, measured.encoded_bytes)?;
    let mut scratch = CounterSortScratch::new();
    scratch
        .prepare(
            measured.counter_sort_entries,
            measured.counter_sort_key_bytes,
        )
        .map_err(io)?;
    if scratch.observed_capacity_bytes().map_err(io)? > scratch_bytes {
        return Err(OperatorError::ResidentContainerInvariant {
            container: "DISTINCT key scratch",
            message: "counter scratch exceeds admission",
        });
    }
    encoded.resize(measured.encoded_bytes, 0);
    semantic_key
        .serialize_with_prepared_scratch(&mut encoded.as_mut_slice(), &mut scratch)
        .map_err(io)?;
    Ok(Key {
        bytes: encoded,
        _grant: grant,
    })
}
impl Borrow<[u8]> for Key {
    fn borrow(&self) -> &[u8] {
        &self.bytes
    }
}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}
impl Eq for Key {}
struct Witness {
    // Ordinal, per-column graph provenance, then original values.
    values: Vec<Value>,
    _grant: MemoryGrant,
}

/// One exact, stable DISTINCT state; wrappers consume each output before next.
pub(crate) struct ExactDistinctState {
    columns: Option<Vec<usize>>,
    output_schema: Vec<LogicalType>,
    resources: Option<QueryResourceContext>,
    keys: HashMap<Key, usize>,
    witnesses: Vec<Option<Witness>>,
    container_grant: Option<MemoryGrant>,
    configuration_grant: Option<MemoryGrant>,
    output_grant: Option<MemoryGrant>,
    width: Option<usize>,
    ordinal: i64,
    unique: usize,
    position: usize,
    finished: bool,
    failed: bool,
    #[cfg(all(feature = "spill", test))]
    force_spill_rows: Option<usize>,
    #[cfg(feature = "spill")]
    spill: Option<DistinctSpill>,
}

impl ExactDistinctState {
    pub(crate) fn new(columns: Option<Vec<usize>>, output_schema: Vec<LogicalType>) -> Self {
        Self {
            columns,
            output_schema,
            resources: None,
            keys: HashMap::new(),
            witnesses: Vec::new(),
            container_grant: None,
            configuration_grant: None,
            output_grant: None,
            width: None,
            ordinal: 0,
            unique: 0,
            position: 0,
            finished: false,
            failed: false,
            #[cfg(all(feature = "spill", test))]
            force_spill_rows: None,
            #[cfg(feature = "spill")]
            spill: None,
        }
    }
    pub(crate) fn take_columns(&mut self) -> Option<Vec<usize>> {
        self.columns.take()
    }
    pub(crate) fn unique_count(&self) -> usize {
        self.unique
    }
    pub(crate) fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        let mut metadata = self
            .output_schema
            .capacity()
            .checked_mul(size_of::<LogicalType>())
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: self.output_schema.capacity(),
                additional_bytes: size_of::<LogicalType>(),
            })?;
        for ty in &self.output_schema {
            let bytes =
                logical_type_bytes(ty).map_err(|_| MemoryGrantError::ArithmeticOverflow {
                    current_bytes: metadata,
                    additional_bytes: 1,
                })?;
            metadata = metadata
                .checked_add(bytes)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: metadata,
                    additional_bytes: bytes,
                })?;
        }
        if let Some(columns) = &self.columns {
            let bytes = columns.capacity().checked_mul(size_of::<usize>()).ok_or(
                MemoryGrantError::ArithmeticOverflow {
                    current_bytes: columns.capacity(),
                    additional_bytes: size_of::<usize>(),
                },
            )?;
            metadata = metadata
                .checked_add(bytes)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: metadata,
                    additional_bytes: bytes,
                })?;
        }
        if self.configuration_grant.is_none()
            || self
                .resources
                .as_ref()
                .is_none_or(|old| old.query_id() != resources.query_id())
        {
            let configuration = resources.try_allocate(metadata)?;
            self.configuration_grant = Some(configuration);
        }
        self.resources = Some(resources.clone());
        self.container_grant = Some(resources.try_allocate(0)?);
        Ok(())
    }
    fn resources(&mut self) -> Result<QueryResourceContext, OperatorError> {
        if self.resources.is_none() {
            let resources = QueryResourceContext::new(BufferManager::with_budget(64 * 1024 * 1024))
                .map_err(context_error)?;
            self.install_resource_context(&resources)
                .map_err(context_error)?;
        }
        if self.container_grant.is_none() {
            let resources = self
                .resources
                .clone()
                .ok_or_else(|| OperatorError::Execution("DISTINCT resources missing".into()))?;
            self.install_resource_context(&resources)
                .map_err(context_error)?;
        }
        self.resources
            .clone()
            .ok_or_else(|| OperatorError::Execution("DISTINCT resources missing".into()))
    }
    fn copy_key(
        &self,
        chunk: &DataChunk,
        row: usize,
        resources: &QueryResourceContext,
    ) -> Result<Key, OperatorError> {
        let count = self.columns.as_ref().map_or(chunk.column_count(), Vec::len);
        let index = |n| self.columns.as_ref().map_or(n, |columns| columns[n]);
        let mut bytes = mul(count, size_of::<Value>())?;
        for n in 0..count {
            bytes = add(
                bytes,
                match chunk.column(index(n)) {
                    Some(column) => column.retained_value_bytes(row).ok_or_else(overflow)?,
                    None => size_of::<Value>(),
                },
            )?;
        }
        let _values_grant = resources.try_allocate(bytes).map_err(context_error)?;
        let mut values = Vec::new();
        reserve(&mut values, mul(count, 2)?)?;
        for n in 0..count {
            values.push(Value::Int64(provenance(
                chunk.column(index(n)).map(|column| column.data_type()),
            )));
            values.push(
                chunk
                    .column(index(n))
                    .and_then(|column| column.get_value(row))
                    .unwrap_or(Value::Null),
            );
        }
        encode_key(resources, &values)
    }

    fn witness(
        &self,
        chunk: &DataChunk,
        row: usize,
        resources: &QueryResourceContext,
    ) -> Result<Witness, OperatorError> {
        let width = chunk.column_count();
        let count = add(mul(width, 2)?, 1)?;
        let mut bytes = mul(add(width, 1)?, size_of::<Value>())?;
        for n in 0..width {
            bytes = add(
                bytes,
                chunk
                    .column(n)
                    .and_then(|column| column.retained_value_bytes(row))
                    .ok_or_else(overflow)?,
            )?;
        }
        let grant = resources.try_allocate(bytes).map_err(context_error)?;
        let mut values = Vec::new();
        reserve(&mut values, count)?;
        values.push(Value::Int64(self.ordinal));
        for n in 0..width {
            let ty = chunk.column(n).map(|column| column.data_type());
            let tag = match ty {
                Some(LogicalType::Node) => 1,
                Some(LogicalType::Edge) => 2,
                Some(LogicalType::List(item)) if **item == LogicalType::Edge => 3,
                _ => 0,
            };
            values.push(Value::Int64(tag));
        }
        for n in 0..width {
            values.push(
                chunk
                    .column(n)
                    .and_then(|column| column.get_value(row))
                    .unwrap_or(Value::Null),
            );
        }
        Ok(Witness {
            values,
            _grant: grant,
        })
    }
    fn admit_containers(&mut self) -> Result<(), OperatorError> {
        let count = add(self.witnesses.len(), 1)?;
        let grant = self
            .container_grant
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT container grant missing".into()))?;
        if self.keys.len() == self.keys.capacity() {
            let target = count
                .max(4)
                .checked_next_power_of_two()
                .ok_or_else(overflow)?;
            // Conservative pinned std HashMap bucket/control bound, covering
            // both old and replacement tables during reallocation.
            let bytes = add(
                mul(mul(target, 2)?, add(size_of::<(Key, usize)>(), 1)?)?,
                128,
            )?;
            grant.try_resize(add(grant.size(), bytes)?)?;
            self.keys
                .try_reserve(target - self.keys.len())
                .map_err(allocation)?;
            if self.keys.capacity() > target * 2 {
                return Err(overflow());
            }
        }
        if self.witnesses.len() == self.witnesses.capacity() {
            let target = count
                .max(4)
                .checked_next_power_of_two()
                .ok_or_else(overflow)?;
            grant.try_resize(add(
                grant.size(),
                mul(target, size_of::<Option<Witness>>())?,
            )?)?;
            reserve(&mut self.witnesses, target)?;
        }
        Ok(())
    }
    pub(crate) fn ingest(&mut self, chunk: &DataChunk, row: usize) -> Result<(), OperatorError> {
        match self.ingest_inner(chunk, row) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.fail(error)),
        }
    }
    pub(crate) fn fail(&mut self, primary: OperatorError) -> OperatorError {
        let result = self.reset_inner(Some(primary));
        self.failed = true;
        match result {
            Err(error) => error,
            Ok(()) => OperatorError::Execution("DISTINCT lost its primary failure".into()),
        }
    }
    fn ingest_inner(&mut self, chunk: &DataChunk, row: usize) -> Result<(), OperatorError> {
        if self.finished || self.failed {
            return Err(OperatorError::Execution(
                "terminal DISTINCT cannot accept rows".into(),
            ));
        }
        let resources = self.resources()?;
        resources.check_cancelled()?;
        if self
            .width
            .is_some_and(|width| width != chunk.column_count())
        {
            return Err(OperatorError::Execution(
                "DISTINCT input width changed".into(),
            ));
        }
        self.width = Some(chunk.column_count());
        self.ordinal = self.ordinal.checked_add(1).ok_or_else(overflow)?;
        #[cfg(feature = "spill")]
        if let Some(spill) = &mut self.spill {
            spill.retry_spent = false;
        }
        #[cfg(all(feature = "spill", test))]
        if self
            .force_spill_rows
            .is_some_and(|limit| self.witnesses.len() >= limit)
            && self.spill.is_none()
            && resources.has_spill_manager()
        {
            self.start_spilling(&resources, false)?;
        }
        #[cfg(feature = "spill")]
        if self.spill.is_none() && !self.witnesses.is_empty() && resources.has_spill_manager() {
            let stats = resources.query_stats();
            if stats.allocated_bytes > stats.growth_limit_bytes / 2 {
                self.start_spilling(&resources, false)?;
            }
        }
        let result = self.ingest_row(chunk, row, &resources);
        #[cfg(feature = "spill")]
        if result.as_ref().is_err_and(is_denial)
            && self.spill.is_none()
            && resources.has_spill_manager()
        {
            self.start_spilling(&resources, true)?;
            return self.ingest_row(chunk, row, &resources);
        }
        #[cfg(feature = "spill")]
        if result.as_ref().is_err_and(is_denial)
            && self.spill.as_ref().is_some_and(|spill| !spill.retry_spent)
        {
            drop(result);
            if let Some(spill) = &mut self.spill {
                spill.retry_spent = true;
            }
            self.flush_key_run(&resources)?;
            return self.ingest_row(chunk, row, &resources);
        }
        result
    }
    fn ingest_row(
        &mut self,
        chunk: &DataChunk,
        row: usize,
        resources: &QueryResourceContext,
    ) -> Result<(), OperatorError> {
        let key = self.copy_key(chunk, row, resources)?;
        #[cfg(feature = "spill")]
        if self.spill.is_some() {
            return self.ingest_spilled(key, chunk, row, resources);
        }
        if self.keys.contains_key(key.bytes.as_slice()) {
            return Ok(());
        }
        self.admit_containers()?;
        let witness = self.witness(chunk, row, resources)?;
        self.keys.insert(key, self.witnesses.len());
        self.witnesses.push(Some(witness));
        self.unique = add(self.unique, 1)?;
        Ok(())
    }
    pub(crate) fn finish_input(&mut self) -> Result<(), OperatorError> {
        self.finish_inner().map_err(|error| self.fail(error))
    }
    fn finish_inner(&mut self) -> Result<(), OperatorError> {
        if self.failed {
            return Err(OperatorError::Execution(
                "failed DISTINCT cannot finish".into(),
            ));
        }
        if self.finished {
            return Ok(());
        }
        let resources = self.resources()?;
        resources.check_cancelled()?;
        #[cfg(feature = "spill")]
        if self.spill.is_some() {
            self.finish_spilled(&resources)?;
        }
        self.keys = HashMap::new();
        self.finished = true;
        Ok(())
    }
    pub(crate) fn next_chunk(&mut self) -> OperatorResult {
        self.output_grant = None;
        if self.failed {
            return Ok(None);
        }
        let result = self.next_inner();
        result.map_err(|error| self.fail(error))
    }
    pub(crate) fn next_accounted_chunk(
        &mut self,
    ) -> Result<Option<crate::execution::AccountedDataChunk>, OperatorError> {
        let Some(chunk) = self.next_chunk()? else {
            return Ok(None);
        };
        let grant = self
            .output_grant
            .take()
            .ok_or_else(|| OperatorError::Execution("DISTINCT output authority missing".into()))?;
        crate::execution::accounted_chunk::try_accounted_distinct_output(chunk, grant)
            .map(Some)
            .map_err(|error| self.fail(error))
    }
    fn next_inner(&mut self) -> OperatorResult {
        if !self.finished {
            return Err(OperatorError::Execution(
                "DISTINCT output requested before finish".into(),
            ));
        }
        let resources = self.resources()?;
        resources.check_cancelled()?;
        #[cfg(feature = "spill")]
        if self.spill.is_some() {
            return self.next_spilled(&resources);
        }
        if self.position >= self.witnesses.len() {
            return Ok(None);
        }
        let width = self.width.unwrap_or(0);
        let first = self.witnesses[self.position]
            .as_ref()
            .ok_or_else(|| OperatorError::Execution("DISTINCT witness already consumed".into()))?;
        let mut rows: [&[Value]; 32] = [&[]; 32];
        let mut count = 0;
        for slot in self.witnesses[self.position..].iter().take(rows.len()) {
            let witness = slot.as_ref().ok_or_else(|| {
                OperatorError::Execution("DISTINCT witness already consumed".into())
            })?;
            if witness.values[1..][..width] != first.values[1..][..width] {
                break;
            }
            rows[count] = &witness.values;
            count += 1;
        }
        let (output, grant) =
            Self::build_output(&rows[..count], width, &self.output_schema, &resources)?;
        for slot in &mut self.witnesses[self.position..self.position + count] {
            *slot = None;
        }
        self.position += count;
        self.output_grant = Some(grant);
        Ok(Some(output))
    }
    #[cfg(feature = "spill")]
    fn output_from_values(
        &mut self,
        values: &[Value],
        resources: &QueryResourceContext,
    ) -> Result<DataChunk, OperatorError> {
        let (chunk, grant) = Self::build_output(
            &[values],
            self.width.unwrap_or(0),
            &self.output_schema,
            resources,
        )?;
        self.output_grant = Some(grant);
        Ok(chunk)
    }
    fn build_output(
        rows: &[&[Value]],
        width: usize,
        schema: &[LogicalType],
        resources: &QueryResourceContext,
    ) -> Result<(DataChunk, MemoryGrant), OperatorError> {
        let count = rows.len();
        let mut bytes = mul(
            width,
            add(
                size_of::<ValueVector>(),
                add(mul(count, size_of::<Value>())?, add(count, 16)?)?,
            )?,
        )?;
        let mut payload = 0;
        for values in rows {
            if values.len() != add(mul(width, 2)?, 1)? {
                return Err(OperatorError::Execution(
                    "invalid DISTINCT witness width".into(),
                ));
            }
            for value in &values[width + 1..] {
                payload = add(payload, value.retained_size_bytes().ok_or_else(overflow)?)?;
            }
        }
        bytes = add(bytes, payload)?;
        let mut schema_bytes = 0;
        for ty in schema {
            schema_bytes = add(schema_bytes, logical_type_bytes(ty)?)?;
        }
        bytes = add(bytes, schema_bytes)?;
        for n in 0..width {
            let edge_list = match schema.get(n) {
                Some(LogicalType::List(item)) if **item == LogicalType::Edge => true,
                Some(ty) if *ty != LogicalType::Any => false,
                _ => matches!(rows[0][n + 1], Value::Int64(3)),
            };
            if edge_list {
                bytes = add(bytes, size_of::<LogicalType>())?;
            }
        }
        let grant = resources.try_allocate(bytes).map_err(context_error)?;
        let mut columns = Vec::new();
        reserve(&mut columns, width)?;
        for n in 0..width {
            let ty = match schema.get(n) {
                Some(ty) if *ty != LogicalType::Any => ty.clone(),
                _ => match rows[0][n + 1] {
                    Value::Int64(1) => LogicalType::Node,
                    Value::Int64(2) => LogicalType::Edge,
                    Value::Int64(3) => LogicalType::List(Box::new(LogicalType::Edge)),
                    _ => LogicalType::Any,
                },
            };
            let mut vector = ValueVector::try_with_capacity(ty, count)
                .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
            vector
                .try_reserve_validity_capacity(count)
                .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
            for values in rows {
                vector.push_value(values[width + 1 + n].clone());
            }
            columns.push(vector);
        }
        let mut chunk = DataChunk::new(columns);
        chunk.set_count(count);
        let observed = chunk
            .observed_column_capacity_bytes()
            .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
        if add(add(observed, payload)?, schema_bytes)? > bytes {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "DISTINCT output",
                message: "observed chunk exceeds admission",
            });
        }
        Ok((chunk, grant))
    }
    pub(crate) fn reset(&mut self) -> Result<(), OperatorError> {
        self.reset_inner(None)
    }
    fn reset_inner(&mut self, primary: Option<OperatorError>) -> Result<(), OperatorError> {
        self.output_grant = None;
        self.keys = HashMap::new();
        self.witnesses = Vec::new();
        #[cfg(feature = "spill")]
        let result = self.cleanup_spill(primary);
        #[cfg(not(feature = "spill"))]
        let result = primary.map_or(Ok(()), Err);
        self.container_grant = None;
        self.width = None;
        self.ordinal = 0;
        self.unique = 0;
        self.position = 0;
        self.finished = false;
        self.failed = result.is_err();
        result
    }
}

fn provenance(ty: Option<&LogicalType>) -> i64 {
    match ty {
        Some(LogicalType::Node) => 1,
        Some(LogicalType::Edge) => 2,
        Some(LogicalType::List(item)) if **item == LogicalType::Edge => 3,
        _ => 0,
    }
}

fn logical_type_bytes(ty: &LogicalType) -> Result<usize, OperatorError> {
    let payload = match ty {
        LogicalType::List(item) => logical_type_bytes(item)?,
        LogicalType::Map { key, value } => {
            add(logical_type_bytes(key)?, logical_type_bytes(value)?)?
        }
        LogicalType::Struct(fields) => {
            let mut bytes = mul(fields.capacity(), size_of::<(String, LogicalType)>())?;
            for (name, ty) in fields {
                bytes = add(bytes, add(name.capacity(), logical_type_bytes(ty)?)?)?;
            }
            bytes
        }
        _ => 0,
    };
    add(size_of::<LogicalType>(), payload)
}

#[cfg(feature = "spill")]
use super::push::spill_state::{OperatorConsumerAdapter, OperatorSpillState};
#[cfg(feature = "spill")]
use crate::execution::spill::{
    ExternalSort, ExternalSortGrantObserver, ExternalSortOperationError, ExternalSortPrimary,
    OwnedExactSortCursor, SortKey,
};
#[cfg(feature = "spill")]
use grafeo_common::memory::buffer::{
    AccountedErrorPublisher, AccountedErrorPublisherBuildFailure, ConsumerRegistration,
};

#[cfg(feature = "spill")]
struct DistinctRunBuffer {
    rows: Vec<Vec<Value>>,
    _rows_grant: MemoryGrant,
    payload_grant: MemoryGrant,
}
#[cfg(feature = "spill")]
impl DistinctRunBuffer {
    fn new(resources: &QueryResourceContext) -> Result<Self, OperatorError> {
        let grant = resources
            .try_allocate(32 * size_of::<Vec<Value>>())
            .map_err(context_error)?;
        let mut rows = Vec::new();
        reserve(&mut rows, 32)?;
        Ok(Self {
            rows,
            _rows_grant: grant,
            payload_grant: resources.try_allocate(0).map_err(context_error)?,
        })
    }
    fn append(&mut self, key: Option<&[u8]>, values: &[Value]) -> Result<(), OperatorError> {
        if self.rows.len() == 32 {
            return Err(OperatorError::Execution(
                "DISTINCT run buffer must be drained before append".into(),
            ));
        }
        let mut bytes = 0;
        for value in values {
            bytes = add(bytes, value.retained_size_bytes().ok_or_else(overflow)?)?;
        }
        if let Some(key) = key {
            // Hex String and its ArcStr copy coexist during construction.
            bytes = add(
                bytes,
                add(
                    mul(key.len(), 4)?,
                    size_of::<Value>() + 2 * size_of::<usize>() + 7,
                )?,
            )?;
        }
        self.payload_grant
            .try_resize(add(self.payload_grant.size(), bytes)?)?;
        let mut row = Vec::new();
        reserve(&mut row, add(values.len(), usize::from(key.is_some()))?)?;
        if let Some(key) = key {
            let len = mul(key.len(), 2)?;
            let mut encoded = String::new();
            encoded.try_reserve_exact(len).map_err(allocation)?;
            if encoded.capacity() > len {
                return Err(OperatorError::ResidentContainerInvariant {
                    container: "DISTINCT semantic key",
                    message: "hex allocation exceeded admission",
                });
            }
            const DIGITS: &[u8; 16] = b"0123456789abcdef";
            for byte in key {
                encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
                encoded.push(char::from(DIGITS[usize::from(byte & 15)]));
            }
            row.push(Value::from(encoded));
        }
        row.extend_from_slice(values);
        self.rows.push(row);
        Ok(())
    }
    fn granted_bytes(&self) -> Result<usize, OperatorError> {
        add(self._rows_grant.size(), self.payload_grant.size())
    }
}
#[cfg(feature = "spill")]
struct DistinctSpill {
    key_sort: Option<ExternalSort>,
    key_cursor: Option<OwnedExactSortCursor>,
    ordinal_sort: Option<ExternalSort>,
    cursor: Option<OwnedExactSortCursor>,
    key_buffer: Option<DistinctRunBuffer>,
    ordinal_buffer: Option<DistinctRunBuffer>,
    key_state: std::sync::Arc<OperatorSpillState>,
    ordinal_state: std::sync::Arc<OperatorSpillState>,
    _key_registration: ConsumerRegistration,
    _ordinal_registration: ConsumerRegistration,
    _metadata_grant: MemoryGrant,
    failure_publisher: Option<AccountedErrorPublisher<ExternalSortOperationError>>,
    retry_spent: bool,
    terminal_publisher: Option<AccountedErrorPublisher<DistinctTerminalFailure>>,
    terminal_error: Option<OperatorError>,
}
#[cfg(feature = "spill")]
#[derive(Debug)]
enum DistinctCleanupError {
    Operator(OperatorError),
    Sort(ExternalSortOperationError),
}
#[cfg(feature = "spill")]
impl DistinctCleanupError {
    fn classification(&self) -> super::AccountedFailureClassification {
        match self {
            Self::Operator(error) => operator_classification(error),
            Self::Sort(error) => sort_classification(error),
        }
    }
}
#[cfg(feature = "spill")]
#[derive(Debug)]
struct DistinctTerminalFailure {
    primary: Option<OperatorError>,
    cleanup: [Option<DistinctCleanupError>; 4],
}
#[cfg(feature = "spill")]
impl std::fmt::Display for DistinctTerminalFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(primary) = &self.primary {
            write!(out, "{primary}; ")?;
        }
        out.write_str("DISTINCT cleanup failed")?;
        for error in self.cleanup.iter().flatten() {
            match error {
                DistinctCleanupError::Operator(error) => write!(out, "; {error}")?,
                DistinctCleanupError::Sort(error) => write!(out, "; {error}")?,
            }
        }
        Ok(())
    }
}
#[cfg(feature = "spill")]
impl std::error::Error for DistinctTerminalFailure {}
#[cfg(feature = "spill")]
fn io_classification(error: &std::io::Error) -> super::AccountedFailureClassification {
    use super::AccountedFailureClassification as C;
    match error.kind() {
        std::io::ErrorKind::OutOfMemory => C::ResidentAllocation,
        std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded => C::StorageFull,
        _ => C::Execution,
    }
}
#[cfg(feature = "spill")]
fn operator_classification(error: &OperatorError) -> super::AccountedFailureClassification {
    use super::AccountedFailureClassification as C;
    match error {
        OperatorError::ClassifiedAccountedFailure { classification, .. } => classification.clone(),
        OperatorError::Context { source, .. } => operator_classification(source),
        OperatorError::ResidentMemory(error) => C::ResidentMemory(error.clone()),
        OperatorError::QueryCancelled(error) => C::QueryCancelled(*error),
        OperatorError::StorageFull(_) => C::StorageFull,
        OperatorError::ResidentAllocation(_)
        | OperatorError::ResidentContainerAllocation { .. }
        | OperatorError::ResidentNativeMapAllocation { .. }
        | OperatorError::ResidentNativeMapAllocationWithRollback { .. } => C::ResidentAllocation,
        OperatorError::ResidentExactVectorAllocation(error) => {
            C::ResidentExactVectorAllocation(error.clone())
        }
        OperatorError::ResidentContainerInvariant { .. }
        | OperatorError::ResidentContainerInvariantWithRollback { .. } => C::ResidentInvariant,
        OperatorError::TypeMismatch { .. } => C::TypeMismatch,
        OperatorError::ColumnNotFound(_) => C::ColumnNotFound,
        OperatorError::ConstraintViolation(_) => C::ConstraintViolation,
        OperatorError::WriteConflict(_) => C::WriteConflict,
        OperatorError::UnsupportedAccountedTransport { .. } => C::UnsupportedAccountedTransport,
        _ => C::Execution,
    }
}
#[cfg(feature = "spill")]
fn new_failure_publisher<T: std::error::Error + Send + 'static>(
    resources: &QueryResourceContext,
) -> Result<AccountedErrorPublisher<T>, OperatorError> {
    AccountedErrorPublisher::try_new(resources.try_allocate(0).map_err(context_error)?).map_err(
        |error| match error.failure() {
            AccountedErrorPublisherBuildFailure::Admission(memory) => {
                OperatorError::ResidentMemory(memory.clone())
            }
            _ => OperatorError::ResidentAllocation(error.to_string()),
        },
    )
}
#[cfg(feature = "spill")]
fn publish_failure(
    publisher: AccountedErrorPublisher<ExternalSortOperationError>,
    error: ExternalSortOperationError,
) -> OperatorError {
    OperatorError::ClassifiedAccountedFailure {
        classification: sort_classification(&error),
        authority: publisher.publish(error),
    }
}
#[cfg(feature = "spill")]
fn sort_classification(
    error: &ExternalSortOperationError,
) -> super::AccountedFailureClassification {
    use super::AccountedFailureClassification as C;
    match error {
        ExternalSortOperationError::Cancelled(error)
        | ExternalSortOperationError::CancelledWithCleanup { error, .. } => {
            C::QueryCancelled(*error)
        }
        ExternalSortOperationError::Memory(error)
        | ExternalSortOperationError::MemoryWithCleanup { error, .. } => {
            C::ResidentMemory(error.clone())
        }
        ExternalSortOperationError::Allocation(_) => C::ResidentAllocation,
        ExternalSortOperationError::Io(error) => io_classification(error),
        ExternalSortOperationError::WithGrantRelease { primary, .. } => match primary {
            ExternalSortPrimary::Cancelled(error) => C::QueryCancelled(*error),
            ExternalSortPrimary::Memory(error) => C::ResidentMemory(error.clone()),
            ExternalSortPrimary::Allocation(_) => C::ResidentAllocation,
            ExternalSortPrimary::Io(error) => io_classification(error),
        },
    }
}
#[cfg(feature = "spill")]
fn is_denial(error: &OperatorError) -> bool {
    matches!(
        error,
        OperatorError::ResidentMemory(
            MemoryGrantError::LimitExceeded { .. } | MemoryGrantError::Denied { .. }
        )
    )
}

#[cfg(feature = "spill")]
impl ExactDistinctState {
    fn start_spilling(
        &mut self,
        resources: &QueryResourceContext,
        retry_spent: bool,
    ) -> Result<(), OperatorError> {
        let manager = resources
            .ensure_spill_manager()
            .map_err(context_error)?
            .cloned()
            .ok_or_else(|| OperatorError::Execution("DISTINCT spill manager missing".into()))?;
        let metadata = resources.try_allocate(1024).map_err(context_error)?;
        let key_state = std::sync::Arc::new(OperatorSpillState::new("DistinctKeySort".into()));
        let ordinal_state =
            std::sync::Arc::new(OperatorSpillState::new("DistinctOrdinalSort".into()));
        let key_registration = resources
            .register_consumer_scoped(std::sync::Arc::new(OperatorConsumerAdapter::new(
                key_state.clone(),
            )))
            .map_err(context_error)?;
        let ordinal_registration = resources
            .register_consumer_scoped(std::sync::Arc::new(OperatorConsumerAdapter::new(
                ordinal_state.clone(),
            )))
            .map_err(context_error)?;
        let width = add(mul(self.width.unwrap_or(0), 2)?, 1)?;
        let key_sort = ExternalSort::new_accounted_with_cancellation(
            manager,
            add(width, 1)?,
            vec![SortKey::ascending(0), SortKey::ascending(1)],
            resources.try_allocate(0).map_err(context_error)?,
            resources.cancellation_token().clone(),
        );
        self.spill = Some(DistinctSpill {
            key_sort: Some(key_sort),
            key_cursor: None,
            ordinal_sort: None,
            cursor: None,
            key_buffer: Some(DistinctRunBuffer::new(resources)?),
            ordinal_buffer: None,
            key_state,
            ordinal_state,
            _key_registration: key_registration,
            _ordinal_registration: ordinal_registration,
            _metadata_grant: metadata,
            failure_publisher: Some(new_failure_publisher(resources)?),
            retry_spent,
            terminal_publisher: Some(new_failure_publisher(resources)?),
            terminal_error: None,
        });
        // Grant accounting may mutate internally; Hash/Eq read only immutable key bytes.
        #[expect(
            clippy::mutable_key_type,
            reason = "grant ownership is excluded from key equality"
        )]
        let keys = std::mem::take(&mut self.keys);
        for (key, index) in keys {
            let witness = self.witnesses[index].take().ok_or_else(|| {
                OperatorError::Execution("DISTINCT migration lost witness".into())
            })?;
            self.append_key_record(&key.bytes, &witness.values, resources)?;
        }
        self.witnesses = Vec::new();
        self.container_grant = Some(resources.try_allocate(0).map_err(context_error)?);
        Ok(())
    }
    fn run_limit(&self) -> usize {
        #[cfg(test)]
        if self.force_spill_rows.is_some() {
            return 1;
        }
        32
    }
    fn append_key_record(
        &mut self,
        key: &[u8],
        values: &[Value],
        resources: &QueryResourceContext,
    ) -> Result<(), OperatorError> {
        let limit = self.run_limit();
        let spill = self
            .spill
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT spill state missing".into()))?;
        let attempt = spill
            .key_buffer
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT key buffer missing".into()))?
            .append(Some(key), values);
        if attempt.as_ref().is_err_and(is_denial) && !spill.retry_spent {
            spill.retry_spent = true;
            Self::flush_run(spill, true)?;
            spill
                .key_buffer
                .as_mut()
                .ok_or_else(|| OperatorError::Execution("DISTINCT key buffer missing".into()))?
                .append(Some(key), values)?;
        } else {
            attempt?;
        }
        let buffer = spill
            .key_buffer
            .as_ref()
            .ok_or_else(|| OperatorError::Execution("DISTINCT key buffer missing".into()))?;
        let sort_bytes = spill
            .key_sort
            .as_ref()
            .map(ExternalSort::checked_total_granted_bytes)
            .transpose()?
            .unwrap_or(0);
        spill
            .key_state
            .set_usage(add(buffer.granted_bytes()?, sort_bytes)?);
        if buffer.rows.len() >= limit || spill.key_state.take_eviction_request().is_some() {
            // A write/publication failure must never replay this accepted row.
            spill.retry_spent = true;
            self.flush_key_run(resources)?;
        }
        Ok(())
    }
    fn flush_run(spill: &mut DistinctSpill, keyed: bool) -> Result<(), OperatorError> {
        let (sort, buffer, state) = if keyed {
            (&mut spill.key_sort, &mut spill.key_buffer, &spill.key_state)
        } else {
            (
                &mut spill.ordinal_sort,
                &mut spill.ordinal_buffer,
                &spill.ordinal_state,
            )
        };
        let buffer = buffer
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT run buffer missing".into()))?;
        if buffer.rows.is_empty() {
            return Ok(());
        }
        for row in &buffer.rows {
            let valid = if keyed {
                matches!(
                    (row.first(), row.get(1)),
                    (Some(Value::String(_)), Some(Value::Int64(_)))
                )
            } else {
                matches!(row.first(), Some(Value::Int64(_)))
            };
            if !valid {
                return Err(OperatorError::Execution(
                    "DISTINCT sort record has an invalid key or ordinal".into(),
                ));
            }
        }
        if keyed {
            buffer
                .rows
                .sort_unstable_by(|a, b| match (&a[0], &b[0], &a[1], &b[1]) {
                    (Value::String(a), Value::String(b), Value::Int64(ai), Value::Int64(bi)) => {
                        a.cmp(b).then_with(|| ai.cmp(bi))
                    }
                    _ => std::cmp::Ordering::Equal,
                });
            buffer.rows.dedup_by(|a, b| a[0] == b[0]);
        } else {
            buffer.rows.sort_unstable_by(|a, b| match (&a[0], &b[0]) {
                (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
                _ => std::cmp::Ordering::Equal,
            });
        }
        let observer = ExternalSortGrantObserver::owned(state.clone());
        observer.publish_retained(buffer.granted_bytes()?)?;
        let sort = sort
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT sorter missing".into()))?;
        let publisher = spill.failure_publisher.take().ok_or_else(|| {
            OperatorError::Execution("DISTINCT error publisher already consumed".into())
        })?;
        match sort.spill_distinct_run_accounted(&buffer.rows, &observer) {
            Ok(()) => spill.failure_publisher = Some(publisher),
            Err(error) => return Err(publish_failure(publisher, error)),
        }
        buffer.rows.clear();
        buffer.payload_grant.try_resize(0)?;
        observer.publish_retained(buffer.granted_bytes()?)?;
        Ok(())
    }
    fn flush_key_run(&mut self, _resources: &QueryResourceContext) -> Result<(), OperatorError> {
        Self::flush_run(
            self.spill
                .as_mut()
                .ok_or_else(|| OperatorError::Execution("DISTINCT spill state missing".into()))?,
            true,
        )
    }
    fn ingest_spilled(
        &mut self,
        key: Key,
        chunk: &DataChunk,
        row: usize,
        resources: &QueryResourceContext,
    ) -> Result<(), OperatorError> {
        let witness = self.witness(chunk, row, resources)?;
        self.append_key_record(&key.bytes, &witness.values, resources)
    }
    fn finish_spilled(&mut self, resources: &QueryResourceContext) -> Result<(), OperatorError> {
        self.flush_key_run(resources)?;
        let limit = self.run_limit();
        let spill = self
            .spill
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT spill state missing".into()))?;
        spill.key_buffer = None;
        let observer = ExternalSortGrantObserver::owned(spill.key_state.clone());
        let sort = spill
            .key_sort
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT key sorter missing".into()))?;
        let publisher = spill.failure_publisher.take().ok_or_else(|| {
            OperatorError::Execution("DISTINCT error publisher already consumed".into())
        })?;
        match sort.finish_distinct_runs_accounted(&observer) {
            Ok(()) => spill.failure_publisher = Some(publisher),
            Err(error) => return Err(publish_failure(publisher, error)),
        }
        if !sort.try_enable_exact_owned_output(Some(&spill.key_state)) {
            return Err(MemoryGrantError::Denied {
                additional_bytes: 1,
            }
            .into());
        }
        let sort = spill.key_sort.take().ok_or_else(|| {
            OperatorError::Execution("DISTINCT key sorter already consumed".into())
        })?;
        spill.key_cursor = Some(
            sort.into_send_owned_disk_cursor(observer)
                .map_err(|error| error.into_operator_error())?,
        );
        let width = add(mul(self.width.unwrap_or(0), 2)?, 1)?;
        let manager = resources
            .ensure_spill_manager()
            .map_err(context_error)?
            .cloned()
            .ok_or_else(|| OperatorError::Execution("DISTINCT spill manager missing".into()))?;
        spill.ordinal_sort = Some(ExternalSort::new_accounted_with_cancellation(
            manager,
            width,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).map_err(context_error)?,
            resources.cancellation_token().clone(),
        ));
        spill.ordinal_buffer = Some(DistinctRunBuffer::new(resources)?);
        let mut previous: Option<(Value, MemoryGrant)> = None;
        self.unique = 0;
        loop {
            resources.check_cancelled()?;
            let cursor = spill
                .key_cursor
                .as_mut()
                .ok_or_else(|| OperatorError::Execution("DISTINCT key cursor missing".into()))?;
            let row = match cursor.next_owned_row() {
                Ok(Some(row)) => row,
                Ok(None) => break,
                Err(error) => {
                    return Err(cursor.finish_stream_failure(error, "DISTINCT key merge cleanup"));
                }
            };
            let emission = (|| {
                let key = row.values().first().ok_or_else(|| {
                    OperatorError::Execution("DISTINCT sorted key missing".into())
                })?;
                if !matches!(
                    (key, row.values().get(1)),
                    (Value::String(_), Some(Value::Int64(_)))
                ) {
                    return Err(OperatorError::Execution(
                        "DISTINCT sorted key or ordinal has an invalid type".into(),
                    ));
                }
                let fresh = previous.as_ref().is_none_or(|(old, _)| old != key);
                if fresh {
                    let append = |spill: &mut DistinctSpill| {
                        let grant = resources
                            .try_allocate(key.retained_size_bytes().ok_or_else(overflow)?)
                            .map_err(context_error)?;
                        let buffer = spill.ordinal_buffer.as_mut().ok_or_else(|| {
                            OperatorError::Execution("DISTINCT ordinal buffer missing".into())
                        })?;
                        buffer.append(None, &row.values()[1..])?;
                        Ok::<_, OperatorError>((key.clone(), grant))
                    };
                    let attempt = append(spill);
                    // The current reader row remains accounted while one
                    // existing ordinal run is flushed to admit its witness.
                    previous = Some(if attempt.as_ref().is_err_and(is_denial) {
                        Self::flush_run(spill, false)?;
                        append(spill)?
                    } else {
                        attempt?
                    });
                    self.unique = add(self.unique, 1)?;
                    let buffer = spill.ordinal_buffer.as_ref().ok_or_else(|| {
                        OperatorError::Execution("DISTINCT ordinal buffer missing".into())
                    })?;
                    let sort_bytes = spill
                        .ordinal_sort
                        .as_ref()
                        .map(ExternalSort::checked_total_granted_bytes)
                        .transpose()?
                        .unwrap_or(0);
                    spill
                        .ordinal_state
                        .set_usage(add(buffer.granted_bytes()?, sort_bytes)?);
                    if buffer.rows.len() >= limit
                        || spill.ordinal_state.take_eviction_request().is_some()
                    {
                        Self::flush_run(spill, false)?;
                    }
                }
                Ok(())
            })();
            drop(row.into_released_grant());
            let cursor = spill
                .key_cursor
                .as_mut()
                .ok_or_else(|| OperatorError::Execution("DISTINCT key cursor missing".into()))?;
            let release = cursor.release_transferred_retained();
            match emission {
                Ok(()) => {
                    release.map_err(|error| {
                        cursor.finish_stream_failure(error, "DISTINCT key transfer cleanup")
                    })?;
                }
                Err(error) => {
                    return Err(cursor.finish_operator_failure(
                        error,
                        release.err().map(|error| (error, "DISTINCT key transfer")),
                        "DISTINCT witness extraction cleanup",
                    ));
                }
            }
        }
        drop(previous);
        spill.key_cursor = None;
        Self::flush_run(spill, false)?;
        spill.ordinal_buffer = None;
        let observer = ExternalSortGrantObserver::owned(spill.ordinal_state.clone());
        let sort = spill
            .ordinal_sort
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("DISTINCT ordinal sorter missing".into()))?;
        let publisher = spill.failure_publisher.take().ok_or_else(|| {
            OperatorError::Execution("DISTINCT error publisher already consumed".into())
        })?;
        match sort.finish_distinct_runs_accounted(&observer) {
            Ok(()) => spill.failure_publisher = Some(publisher),
            Err(error) => return Err(publish_failure(publisher, error)),
        }
        if !sort.try_enable_exact_owned_output(Some(&spill.ordinal_state)) {
            return Err(MemoryGrantError::Denied {
                additional_bytes: 1,
            }
            .into());
        }
        let sort = spill.ordinal_sort.take().ok_or_else(|| {
            OperatorError::Execution("DISTINCT ordinal sorter already consumed".into())
        })?;
        spill.cursor = Some(
            sort.into_send_owned_disk_cursor(observer)
                .map_err(|error| error.into_operator_error())?,
        );
        Ok(())
    }
    fn next_spilled(&mut self, resources: &QueryResourceContext) -> OperatorResult {
        let cursor = self
            .spill
            .as_mut()
            .and_then(|spill| spill.cursor.as_mut())
            .ok_or_else(|| OperatorError::Execution("DISTINCT output cursor missing".into()))?;
        let row = match cursor.next_owned_row() {
            Ok(row) => row,
            Err(error) => return Err(cursor.finish_stream_failure(error, "DISTINCT merge cleanup")),
        };
        let Some(row) = row else {
            return Ok(None);
        };
        let output = self.output_from_values(row.values(), resources);
        drop(row.into_released_grant());
        let cursor = self
            .spill
            .as_mut()
            .and_then(|spill| spill.cursor.as_mut())
            .ok_or_else(|| OperatorError::Execution("DISTINCT output cursor missing".into()))?;
        let release = cursor.release_transferred_retained();
        match output {
            Ok(output) => {
                release.map_err(|error| {
                    cursor.finish_stream_failure(error, "DISTINCT transfer cleanup")
                })?;
                Ok(Some(output))
            }
            Err(error) => Err(cursor.finish_operator_failure(
                error,
                release
                    .err()
                    .map(|error| (error, "DISTINCT output release")),
                "DISTINCT output cleanup",
            )),
        }
    }
    fn cleanup_spill(&mut self, primary: Option<OperatorError>) -> Result<(), OperatorError> {
        let Some(spill) = &mut self.spill else {
            return primary.map_or(Ok(()), Err);
        };
        if let Some(error) = &spill.terminal_error {
            return Err(error.clone());
        }
        // Admission precedes cleanup: error publication never needs a new
        // allocation after a destructive or failed accounting transition.
        let publisher = spill.terminal_publisher.take().ok_or_else(|| {
            OperatorError::Execution("DISTINCT terminal publisher missing".into())
        })?;
        spill.key_buffer = None;
        spill.ordinal_buffer = None;
        let mut cleanup = [None, None, None, None];
        if let Some(cursor) = &mut spill.key_cursor {
            cleanup[0] = cursor
                .finish_early_stop()
                .err()
                .map(DistinctCleanupError::Operator);
        }
        if let Some(cursor) = &mut spill.cursor {
            cleanup[1] = cursor
                .finish_early_stop()
                .err()
                .map(DistinctCleanupError::Operator);
        }
        if let Some(sort) = &mut spill.key_sort {
            cleanup[2] = sort
                .cleanup_distinct_accounted()
                .err()
                .map(DistinctCleanupError::Sort);
        }
        if let Some(sort) = &mut spill.ordinal_sort {
            cleanup[3] = sort
                .cleanup_distinct_accounted()
                .err()
                .map(DistinctCleanupError::Sort);
        }
        if let Some(first) = cleanup.iter().flatten().next() {
            let classification = primary
                .as_ref()
                .map_or_else(|| first.classification(), operator_classification);
            let error = OperatorError::ClassifiedAccountedFailure {
                classification,
                authority: publisher.publish(DistinctTerminalFailure { primary, cleanup }),
            };
            spill.terminal_error = Some(error.clone());
            return Err(error);
        }
        spill.key_state.set_usage(0);
        spill.ordinal_state.set_usage(0);
        self.spill = None;
        primary.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::HashableValue;

    #[test]
    fn shared_semantic_key_scalar_rows_skip_empty_scratch_admissions() {
        use crate::execution::memory::take_query_admission_attempts;
        use crate::execution::value_codec::{deserialize_row, serialize_row};

        for count in [32, 64, 128] {
            let manager = BufferManager::with_budget(1 << 20);
            let resources = QueryResourceContext::new(manager.clone()).unwrap();
            take_query_admission_attempts();
            for index in 0..count {
                // The real DISTINCT neighbor key contains provenance plus a
                // scalar application ID. Neither needs counter-sort scratch.
                let row = [Value::Int64(0), Value::from(format!("node-{index}"))];
                let mut expected = Vec::new();
                serialize_row(&row, &mut expected).unwrap();
                let key = encode_accounted_semantic_key(&resources, &row).unwrap();
                assert_eq!(key.as_bytes(), expected.as_slice());
                assert_eq!(
                    deserialize_row(&mut key.as_bytes(), row.len()).unwrap(),
                    row
                );
                assert!(key.granted_bytes() > 0);
                assert_eq!(manager.allocated(), key.granted_bytes());
                drop(key);
                assert_eq!(manager.allocated(), 0);
            }
            let attempts = take_query_admission_attempts();
            assert_eq!(
                attempts.1, count,
                "each scalar key must retain its encoded-byte grant"
            );
            assert_eq!(
                attempts.0, 0,
                "N={count}: empty counter scratch must not enter admission"
            );
        }
    }

    #[test]
    fn shared_semantic_key_counter_scratch_keeps_positive_admission() {
        use crate::execution::memory::take_query_admission_attempts;
        use crate::execution::value_codec::{deserialize_row, serialize_row};

        let manager = BufferManager::with_budget(1 << 20);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        // A live zero-byte call proves that the scalar regression's zero
        // count cannot pass because its instrumentation is disconnected.
        take_query_admission_attempts();
        let empty = resources.try_allocate(0).unwrap();
        assert_eq!(take_query_admission_attempts(), (1, 0));
        drop(empty);
        assert_eq!(manager.allocated(), 0);

        let row = [Value::GCounter(std::sync::Arc::new(HashMap::from([
            ("z".to_string(), 7),
            ("a".to_string(), 3),
        ])))];
        let mut expected = Vec::new();
        serialize_row(&row, &mut expected).unwrap();
        let key = encode_accounted_semantic_key(&resources, &row).unwrap();
        assert_eq!(take_query_admission_attempts(), (0, 2));
        assert_eq!(key.as_bytes(), expected.as_slice());
        assert_eq!(
            deserialize_row(&mut key.as_bytes(), row.len()).unwrap(),
            row
        );
        // Workspace is released before publication; only encoded bytes remain.
        assert_eq!(manager.allocated(), key.granted_bytes());
        drop(key);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn shared_semantic_key_empty_row_keeps_encoded_grant_and_cancellation() {
        use crate::execution::memory::take_query_admission_attempts;
        use crate::execution::value_codec::serialize_row;

        let manager = BufferManager::with_budget(64);
        let control = crate::execution::QueryExecutionControl::new();
        let resources =
            QueryResourceContext::new_with_cancellation(manager.clone(), control.token()).unwrap();
        let mut expected = Vec::new();
        serialize_row(&[], &mut expected).unwrap();
        take_query_admission_attempts();
        let key = encode_accounted_semantic_key(&resources, &[]).unwrap();
        assert_eq!(key.as_bytes(), expected.as_slice());
        assert!(key.granted_bytes() > 0);
        assert_eq!(key.granted_bytes(), expected.len());
        assert_eq!(manager.allocated(), key.granted_bytes());
        // The row header retains its positive encoded-byte grant.
        assert_eq!(take_query_admission_attempts(), (0, 1));
        drop(key);
        assert_eq!(manager.allocated(), 0);
        control.cancellation_handle().cancel();
        assert!(matches!(
            encode_accounted_semantic_key(&resources, &[]),
            Err(OperatorError::QueryCancelled(_))
        ));
        assert_eq!(take_query_admission_attempts(), (0, 0));
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn shared_semantic_key_validates_and_measures_nested_values_once() {
        use crate::execution::value_codec::{
            deserialize_row, measure_serialized_row_with_limits, take_semantic_key_traversal_visits,
        };

        for width in [32, 64, 128] {
            let row = [Value::List(
                (0..width)
                    .map(|value| Value::List(vec![Value::Int64(value)].into()))
                    .collect::<Vec<_>>()
                    .into(),
            )];
            // Each real traversal touches the outer list, each inner list, and
            // its scalar. The standalone measurement is a live counter control.
            let expected_visits = 1 + 2 * usize::try_from(width).unwrap();
            take_semantic_key_traversal_visits();
            let measurement =
                measure_serialized_row_with_limits(&row, CodecLimits::format_max()).unwrap();
            assert_eq!(
                take_semantic_key_traversal_visits(),
                (expected_visits, expected_visits)
            );

            let manager = BufferManager::with_budget(1 << 20);
            let resources = QueryResourceContext::new(manager.clone()).unwrap();
            let key = encode_accounted_semantic_key(&resources, &row).unwrap();
            let actual_visits = take_semantic_key_traversal_visits();
            assert_eq!(key.as_bytes().len(), measurement.encoded_bytes);
            assert_eq!(
                deserialize_row(&mut key.as_bytes(), row.len()).unwrap(),
                row
            );
            assert!(key.granted_bytes() > 0);
            assert_eq!(manager.allocated(), key.granted_bytes());
            drop(key);
            assert_eq!(manager.allocated(), 0);
            assert_eq!(
                actual_visits,
                (expected_visits, expected_visits),
                "nested width {width}: accounted encoding must reuse its validated measurement"
            );
        }
    }

    #[test]
    fn shared_semantic_key_retains_only_encoded_bytes_and_preserves_identity() {
        let manager = BufferManager::with_budget(1 << 20);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let left = [
            Value::Float64(-0.0),
            Value::GCounter(std::sync::Arc::new(HashMap::from([
                ("b".to_string(), 2),
                ("a".to_string(), 1),
            ]))),
        ];
        let right = [
            Value::Float64(0.0),
            Value::GCounter(std::sync::Arc::new(HashMap::from([
                ("a".to_string(), 1),
                ("b".to_string(), 2),
            ]))),
        ];
        let a = encode_accounted_semantic_key(&resources, &left).unwrap();
        let b = encode_accounted_semantic_key(&resources, &right).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes());
        assert_eq!(manager.allocated(), a.granted_bytes() + b.granted_bytes());
        let different = encode_accounted_semantic_key(&resources, &[Value::Int64(0)]).unwrap();
        let float = encode_accounted_semantic_key(&resources, &[Value::Float64(0.0)]).unwrap();
        assert_ne!(different.as_bytes(), float.as_bytes());
        drop((b, different, float));
        let (bytes, grant) = a.into_parts();
        assert_eq!(bytes.capacity(), grant.size());
        drop(resources);
        assert_eq!(manager.allocated(), grant.size());
        drop(bytes);
        drop(grant);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn shared_semantic_key_denial_and_cancellation_publish_no_owner() {
        let manager = BufferManager::with_budget(64);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let result = encode_accounted_semantic_key(&resources, &[Value::from("x".repeat(1024))]);
        assert!(result.is_err());
        assert_eq!(manager.allocated(), 0);
        let control = crate::execution::QueryExecutionControl::new();
        let resources =
            QueryResourceContext::new_with_cancellation(manager.clone(), control.token()).unwrap();
        control.cancellation_handle().cancel();
        let result = encode_accounted_semantic_key(&resources, &[]);
        assert!(matches!(result, Err(OperatorError::QueryCancelled(_))));
        assert_eq!(manager.allocated(), 0);
    }

    fn chunk(rows: &[Vec<Value>]) -> DataChunk {
        let width = rows.first().map_or(0, Vec::len);
        let mut chunk = DataChunk::with_capacity(&vec![LogicalType::Any; width], rows.len());
        for row in rows {
            for (index, value) in row.iter().enumerate() {
                chunk.column_mut(index).unwrap().push_value(value.clone());
            }
        }
        chunk.set_count(rows.len());
        chunk
    }
    fn feed(state: &mut ExactDistinctState, rows: &[Vec<Value>]) {
        for rows in rows.chunks(3) {
            let chunk = chunk(rows);
            for row in 0..chunk.row_count() {
                state.ingest(&chunk, row).unwrap();
            }
        }
    }
    fn output(state: &mut ExactDistinctState) -> Vec<Vec<HashableValue>> {
        state.finish_input().unwrap();
        let mut rows = Vec::new();
        while let Some(chunk) = state.next_accounted_chunk().unwrap() {
            let chunk = chunk.chunk();
            for row in 0..chunk.row_count() {
                rows.push(
                    (0..chunk.column_count())
                        .map(|column| {
                            HashableValue(chunk.column(column).unwrap().get_value(row).unwrap())
                        })
                        .collect(),
                );
            }
        }
        rows
    }
    #[test]
    fn exact_keys_keep_typed_nested_values_and_first_signed_zero_witness() {
        let input = vec![
            vec![Value::Float64(-0.0)],
            vec![Value::Int64(0)],
            vec![Value::Float64(0.0)],
            vec![Value::List(vec![Value::Float64(-0.0)].into())],
            vec![Value::List(vec![Value::Float64(0.0)].into())],
            vec![Value::Vector(vec![-0.0_f32].into())],
            vec![Value::Vector(vec![0.0_f32].into())],
            vec![Value::Float64(f64::from_bits(0x7ff8_0000_0000_0001))],
            vec![Value::Float64(f64::from_bits(0x7ff8_0000_0000_0001))],
            vec![Value::Float64(f64::from_bits(0x7ff8_0000_0000_0002))],
            vec![Value::Null],
            vec![Value::Map(
                std::collections::BTreeMap::from([("key".into(), Value::Float64(-0.0))]).into(),
            )],
            vec![Value::Map(
                std::collections::BTreeMap::from([("key".into(), Value::Float64(0.0))]).into(),
            )],
        ];
        let mut state = ExactDistinctState::new(None, vec![LogicalType::Any]);
        feed(&mut state, &input);
        let result = output(&mut state);
        let expected: Vec<_> = [0, 1, 3, 5, 6, 7, 9, 10, 11]
            .into_iter()
            .map(|index| vec![HashableValue(input[index][0].clone())])
            .collect();
        assert_eq!(result, expected);
        let Value::Float64(value) = result[0][0].0 else {
            panic!("float witness lost");
        };
        assert_eq!(value.to_bits(), (-0.0_f64).to_bits());

        #[cfg(all(
            feature = "spill",
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            let directory = tempfile::tempdir().unwrap();
            let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
                .root()
                .unwrap();
            let resources = QueryResourceContext::with_spill_root(
                BufferManager::with_budget(64 << 20),
                &spill_root,
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
            let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
            let mut spilled = ExactDistinctState::new(None, vec![LogicalType::Any]);
            spilled.install_resource_context(&resources).unwrap();
            spilled.force_spill_rows = Some(1);
            feed(&mut spilled, &input);
            assert!(manager.disk_stats().published_live_bytes > 0);
            assert!(manager.disk_stats().peak_reserved_bytes > 0);
            assert_eq!(output(&mut spilled), expected);
        }
    }
    #[test]
    fn graph_identity_keys_and_witness_provenance_do_not_collapse_equal_numeric_ids() {
        let mut state = ExactDistinctState::new(None, vec![LogicalType::Any]);
        for ty in [
            LogicalType::Node,
            LogicalType::Edge,
            LogicalType::Int64,
            LogicalType::Node,
        ] {
            let mut chunk = DataChunk::with_capacity(&[ty], 1);
            chunk.column_mut(0).unwrap().push_value(Value::Int64(7));
            chunk.set_count(1);
            state.ingest(&chunk, 0).unwrap();
        }
        state.finish_input().unwrap();
        let mut types = Vec::new();
        while let Some(chunk) = state.next_accounted_chunk().unwrap() {
            types.push(chunk.chunk().column(0).unwrap().data_type().clone());
        }
        assert_eq!(
            types,
            [LogicalType::Node, LogicalType::Edge, LogicalType::Any]
        );
        assert_eq!(state.unique_count(), 3);

        #[cfg(all(
            feature = "spill",
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            let directory = tempfile::tempdir().unwrap();
            let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
                .root()
                .unwrap();
            let resources = QueryResourceContext::with_spill_root(
                BufferManager::with_budget(64 << 20),
                &spill_root,
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
            let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
            let mut spilled = ExactDistinctState::new(None, vec![LogicalType::Any]);
            spilled.install_resource_context(&resources).unwrap();
            spilled.force_spill_rows = Some(1);
            for ty in [
                LogicalType::Node,
                LogicalType::Edge,
                LogicalType::Int64,
                LogicalType::Node,
            ] {
                let mut chunk = DataChunk::with_capacity(&[ty], 1);
                chunk.column_mut(0).unwrap().push_value(Value::Int64(7));
                chunk.set_count(1);
                spilled.ingest(&chunk, 0).unwrap();
            }
            assert!(manager.disk_stats().published_live_bytes > 0);
            assert!(manager.disk_stats().peak_reserved_bytes > 0);
            spilled.finish_input().unwrap();
            let mut types = Vec::new();
            while let Some(chunk) = spilled.next_accounted_chunk().unwrap() {
                types.push(chunk.chunk().column(0).unwrap().data_type().clone());
            }
            assert_eq!(
                types,
                [LogicalType::Node, LogicalType::Edge, LogicalType::Any]
            );
        }
    }
    #[test]
    fn empty_key_and_zero_column_rows_have_one_witness() {
        let mut state = ExactDistinctState::new(None, vec![]);
        let mut input = DataChunk::new(vec![]);
        input.set_count(5);
        for row in 0..5 {
            state.ingest(&input, row).unwrap();
        }
        state.finish_input().unwrap();
        let result = state.next_accounted_chunk().unwrap().unwrap();
        assert_eq!(result.chunk().row_count(), 1);
        assert_eq!(result.chunk().column_count(), 0);
        assert!(state.next_accounted_chunk().unwrap().is_none());

        #[cfg(all(
            feature = "spill",
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            let directory = tempfile::tempdir().unwrap();
            let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
                .root()
                .unwrap();
            let resources = QueryResourceContext::with_spill_root(
                BufferManager::with_budget(64 << 20),
                &spill_root,
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
            let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
            let mut spilled = ExactDistinctState::new(None, vec![]);
            spilled.install_resource_context(&resources).unwrap();
            spilled.force_spill_rows = Some(1);
            let mut input = DataChunk::new(vec![]);
            input.set_count(5);
            for row in 0..5 {
                spilled.ingest(&input, row).unwrap();
            }
            assert!(manager.disk_stats().published_live_bytes > 0);
            assert!(manager.disk_stats().peak_reserved_bytes > 0);
            spilled.finish_input().unwrap();
            let result = spilled.next_accounted_chunk().unwrap().unwrap();
            assert_eq!(result.chunk().row_count(), 1);
            assert_eq!(result.chunk().column_count(), 0);
            assert!(spilled.next_accounted_chunk().unwrap().is_none());
        }
    }
    #[test]
    fn resident_denial_is_terminal_and_releases_partial_state() {
        let resources = QueryResourceContext::new(BufferManager::with_budget(8192)).unwrap();
        let mut state = ExactDistinctState::new(None, vec![LogicalType::Any]);
        state.install_resource_context(&resources).unwrap();
        feed(&mut state, &[vec![Value::Int64(1)]]);
        let oversized = chunk(&[vec![Value::from("x".repeat(32768))]]);
        assert!(matches!(
            state.ingest(&oversized, 0),
            Err(OperatorError::ResidentMemory(_))
        ));
        assert!(state.next_chunk().unwrap().is_none());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            state.configuration_grant.as_ref().unwrap().size()
        );
        drop(state);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
    #[test]
    fn accounted_batch_survives_state_reset_and_cancellation_cleans_remaining_state() {
        let control = crate::execution::QueryExecutionControl::new();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .unwrap();
        let mut state = ExactDistinctState::new(None, vec![LogicalType::Any]);
        state.install_resource_context(&resources).unwrap();
        let rows: Vec<_> = (0..40).map(|value| vec![Value::Int64(value)]).collect();
        feed(&mut state, &rows);
        state.finish_input().unwrap();
        let first = state.next_accounted_chunk().unwrap().unwrap();
        assert_eq!(first.chunk().row_count(), 32);
        let retained = first.granted_bytes();
        control.cancellation_handle().cancel();
        assert!(matches!(
            state.next_accounted_chunk(),
            Err(OperatorError::QueryCancelled(_))
        ));
        assert_eq!(
            resources.query_stats().allocated_bytes,
            retained + state.configuration_grant.as_ref().unwrap().size()
        );
        drop(state);
        assert_eq!(resources.query_stats().allocated_bytes, retained);
        assert_eq!(
            first.chunk().column(0).unwrap().get_value(31),
            Some(Value::Int64(31))
        );
        drop(first);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn forced_one_row_spill_keeps_first_witnesses_through_skew_and_ordinal_merge() {
        let directory = tempfile::tempdir().unwrap();
        let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
            .root()
            .unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(64 << 20),
            &spill_root,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
        let mut rows = vec![
            vec![Value::Int64(3), Value::from("first-three")],
            vec![Value::Int64(1), Value::from("first-one")],
        ];
        for key in 0..80 {
            rows.push(vec![
                Value::Int64(key),
                Value::from(format!("initial-{key}")),
            ]);
        }
        for _ in 0..32 {
            rows.push(vec![Value::Int64(3), Value::from("late-hot-key")]);
        }
        for key in (0..80).rev() {
            rows.push(vec![Value::Int64(key), Value::from("late-witness")]);
        }
        let mut resident = ExactDistinctState::new(Some(vec![0]), vec![LogicalType::Any; 2]);
        feed(&mut resident, &rows);
        let expected = output(&mut resident);
        let mut spilled = ExactDistinctState::new(Some(vec![0]), vec![LogicalType::Any; 2]);
        spilled.install_resource_context(&resources).unwrap();
        spilled.force_spill_rows = Some(1);
        feed(&mut spilled, &rows);
        assert!(
            manager.disk_stats().published_live_bytes > 0,
            "one-row control must create immutable keyed runs"
        );
        assert_eq!(output(&mut spilled), expected);
        assert_eq!(spilled.unique_count(), 80);
        assert!(manager.disk_stats().peak_reserved_bytes > 0);
        spilled.reset().unwrap();
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(
            resources.query_stats().allocated_bytes,
            spilled.configuration_grant.as_ref().unwrap().size()
        );
        drop(spilled);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn production_pressure_flushes_distinct_without_a_test_spill_threshold() {
        let directory = tempfile::tempdir().unwrap();
        let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
            .root()
            .unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(512 << 10),
            &spill_root,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
        let mut state = ExactDistinctState::new(Some(vec![0]), vec![LogicalType::Any; 2]);
        state.install_resource_context(&resources).unwrap();
        assert!(state.force_spill_rows.is_none());
        for key in 0..4096 {
            let row = chunk(&[vec![Value::Int64(key), Value::from("original")]]);
            state
                .ingest(&row, 0)
                .unwrap_or_else(|error| panic!("production pressure insertion {key}: {error}"));
        }
        assert!(
            manager.disk_stats().published_live_bytes > 0,
            "real budget pressure must publish DISTINCT runs before finalization"
        );
        state.finish_input().unwrap();
        let mut next = 0;
        while let Some(chunk) = state.next_accounted_chunk().unwrap() {
            for row in 0..chunk.chunk().row_count() {
                assert_eq!(
                    chunk.chunk().column(0).unwrap().get_value(row),
                    Some(Value::Int64(next))
                );
                assert_eq!(
                    chunk.chunk().column(1).unwrap().get_value(row),
                    Some(Value::from("original"))
                );
                next += 1;
            }
        }
        assert_eq!(next, 4096);
        state.reset().unwrap();
        drop(state);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    }
    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn failure_retains_primary_and_all_cleanup_payloads_without_formatting() {
        use crate::execution::spill::{
            CleartextSpillRecordProvider, SpillFrameLimits, SpillIo, SpillIoOperation,
        };
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        #[derive(Debug)]
        struct HostileCleanup {
            drops: Arc<AtomicUsize>,
            formats: Arc<AtomicUsize>,
        }
        impl std::fmt::Display for HostileCleanup {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.formats.fetch_add(1, Ordering::SeqCst);
                panic!("cleanup payload must not be formatted during failure");
            }
        }
        impl std::error::Error for HostileCleanup {}
        impl Drop for HostileCleanup {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct DeleteFault {
            armed: AtomicBool,
            attempts: AtomicUsize,
            drops: Arc<AtomicUsize>,
            formats: Arc<AtomicUsize>,
        }
        impl SpillIo for DeleteFault {
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                if operation == SpillIoOperation::Delete && self.armed.load(Ordering::SeqCst) {
                    self.attempts.fetch_add(1, Ordering::SeqCst);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        HostileCleanup {
                            drops: self.drops.clone(),
                            formats: self.formats.clone(),
                        },
                    ));
                }
                Ok(())
            }
        }
        let fault = Arc::new(DeleteFault {
            armed: AtomicBool::new(false),
            attempts: AtomicUsize::new(0),
            drops: Arc::new(AtomicUsize::new(0)),
            formats: Arc::new(AtomicUsize::new(0)),
        });
        let directory = tempfile::tempdir().unwrap();
        let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(fault.clone())
            .root()
            .unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(64 << 20),
            &spill_root,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
        let mut state = ExactDistinctState::new(None, vec![LogicalType::Int64]);
        state.install_resource_context(&resources).unwrap();
        state.force_spill_rows = Some(1);
        feed(&mut state, &[vec![Value::Int64(0)], vec![Value::Int64(1)]]);
        // Retain a second real owned sorter/run, as during the key-to-ordinal
        // handoff. Failure must attempt both independent cleanup owners.
        let spill = state.spill.as_mut().unwrap();
        let mut ordinal = ExternalSort::new_accounted_with_cancellation(
            manager.clone(),
            3,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).unwrap(),
            resources.cancellation_token().clone(),
        );
        ordinal
            .spill_distinct_run_accounted(
                &[vec![Value::Int64(0), Value::Int64(0), Value::Int64(0)]],
                &ExternalSortGrantObserver::owned(spill.ordinal_state.clone()),
            )
            .unwrap();
        spill.ordinal_sort = Some(ordinal);
        assert_eq!(manager.active_file_count(), 2);
        fault.armed.store(true, Ordering::SeqCst);
        let error = state.fail(OperatorError::QueryCancelled(
            crate::execution::QueryCancellationError::Cancelled,
        ));
        assert_eq!(fault.attempts.load(Ordering::SeqCst), 2);
        assert_eq!(fault.formats.load(Ordering::SeqCst), 0);
        assert_eq!(fault.drops.load(Ordering::SeqCst), 0);
        let OperatorError::ClassifiedAccountedFailure {
            classification,
            authority,
        } = &error
        else {
            panic!("primary and cleanup must retain accounted authority");
        };
        assert!(matches!(
            classification,
            super::super::AccountedFailureClassification::QueryCancelled(
                crate::execution::QueryCancellationError::Cancelled,
            )
        ));
        assert_eq!(
            authority.inspect::<DistinctTerminalFailure, _>(|failure| {
                assert!(matches!(
                    failure.primary,
                    Some(OperatorError::QueryCancelled(_))
                ));
                failure.cleanup.iter().flatten().count()
            }),
            Some(2)
        );
        let repeated = state.reset().unwrap_err();
        assert_eq!(fault.attempts.load(Ordering::SeqCst), 2);
        assert_eq!(fault.drops.load(Ordering::SeqCst), 0);
        let retained = error.clone();
        fault.armed.store(false, Ordering::SeqCst);
        drop(state);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(error);
        drop(repeated);
        assert_eq!(fault.drops.load(Ordering::SeqCst), 0);
        assert!(resources.query_stats().allocated_bytes > 0);
        drop(retained);
        assert_eq!(fault.drops.load(Ordering::SeqCst), 2);
        assert_eq!(fault.formats.load(Ordering::SeqCst), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
}
