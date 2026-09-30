//! Runtime enforcement and eager publication boundaries for catalog procedures.

use std::collections::VecDeque;

use grafeo_common::memory::buffer::MemoryGrant;
use grafeo_common::types::{LogicalType, PropertyKey, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::memory::{QueryResourceContext, QueryResourceContextError};
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult};

use crate::catalog::PropertyDataType;

/// Validates each body column against its declared `RETURNS` type before an
/// outer projection can rename/reorder it.
pub(crate) struct ProcedureOutputContractOperator {
    child: Box<dyn Operator>,
    procedure: String,
    columns: Vec<(String, PropertyDataType)>,
}

impl ProcedureOutputContractOperator {
    pub(crate) fn new(
        child: Box<dyn Operator>,
        procedure: String,
        columns: Vec<(String, PropertyDataType)>,
    ) -> Self {
        Self {
            child,
            procedure,
            columns,
        }
    }
}

impl Operator for ProcedureOutputContractOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.child.next()? else {
            return Ok(None);
        };
        if chunk.column_count() != self.columns.len() {
            return Err(OperatorError::Execution(format!(
                "procedure '{}' declared {} output columns but produced {}",
                self.procedure,
                self.columns.len(),
                chunk.column_count()
            )));
        }

        for (index, (name, expected)) in self.columns.iter().enumerate() {
            let column = chunk
                .column(index)
                .ok_or_else(|| OperatorError::ColumnNotFound(format!("procedure output {name}")))?;
            for row in chunk.selected_indices() {
                let value = column.get_value(row).unwrap_or(Value::Null);
                if !procedure_value_matches(expected, &value) {
                    return Err(OperatorError::TypeMismatch {
                        expected: expected.to_string(),
                        found: value.type_name().to_string(),
                    }
                    .with_context(format!(
                        "procedure '{}' output '{}' violates its declared return type",
                        self.procedure, name
                    )));
                }
            }
        }
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "ProcedureOutputContract"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &grafeo_core::execution::memory::QueryResourceContext,
    ) -> Result<(), grafeo_core::execution::memory::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }
}

/// Fully executes a write-capable procedure before exposing its first row.
///
/// A procedure invocation is a semantic boundary, not a macro expansion. If
/// an outer `LIMIT`, semi-join, or other short-circuiting consumer could stop
/// pulling the body, it could otherwise commit only a prefix of the declared
/// operation or hide a late return-contract failure. This operator drains and
/// validates the complete body on its first pull, then replays the buffered
/// output.
///
/// Retained chunks are charged to the query's resident-memory account. A
/// denied grant fails the statement (and therefore rolls it back) rather than
/// silently falling back to unaccounted memory. The current generic result
/// pipeline is resident-only; a future move-only result spool can replace the
/// buffer without changing this invocation contract.
pub(crate) struct EagerProcedureBoundaryOperator {
    child: Box<dyn Operator>,
    procedure: String,
    resources: Option<QueryResourceContext>,
    chunks: VecDeque<DataChunk>,
    grants: Vec<MemoryGrant>,
    materialized: bool,
}

impl EagerProcedureBoundaryOperator {
    pub(crate) fn new(child: Box<dyn Operator>, procedure: String) -> Self {
        Self {
            child,
            procedure,
            resources: None,
            chunks: VecDeque::new(),
            grants: Vec::new(),
            materialized: false,
        }
    }

    fn map_resource_error(error: QueryResourceContextError) -> OperatorError {
        match error {
            QueryResourceContextError::Memory(error) => OperatorError::ResidentMemory(error),
            other => OperatorError::Execution(other.to_string()),
        }
    }

    fn materialize(&mut self) -> Result<(), OperatorError> {
        let resources = self.resources.as_ref().ok_or_else(|| {
            OperatorError::Execution(format!(
                "write-capable procedure '{}' has no query resource context",
                self.procedure
            ))
        })?;

        loop {
            resources.check_cancelled()?;
            let Some(chunk) = self.child.next()? else {
                break;
            };

            let chunk_bytes = chunk
                .observed_column_capacity_bytes()
                .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
            let retained_bytes = chunk_bytes
                .checked_add(std::mem::size_of::<DataChunk>())
                .ok_or_else(|| {
                    OperatorError::ResidentAllocation(
                        "procedure result-buffer size overflow".to_string(),
                    )
                })?;
            let grant = resources
                .try_allocate(retained_bytes)
                .map_err(Self::map_resource_error)?;
            self.chunks.try_reserve(1).map_err(|error| {
                OperatorError::ResidentContainerAllocation {
                    container: "eager procedure result queue",
                    source: error,
                }
            })?;
            self.grants.try_reserve(1).map_err(|error| {
                OperatorError::ResidentContainerAllocation {
                    container: "eager procedure memory grants",
                    source: error,
                }
            })?;
            self.chunks.push_back(chunk);
            self.grants.push(grant);
        }
        resources.check_cancelled()?;
        self.materialized = true;
        Ok(())
    }
}

impl Operator for EagerProcedureBoundaryOperator {
    fn next(&mut self) -> OperatorResult {
        if !self.materialized {
            self.materialize()?;
        }
        Ok(self.chunks.pop_front())
    }

    fn reset(&mut self) {
        self.child.reset();
        self.chunks.clear();
        self.grants.clear();
        self.materialized = false;
    }

    fn name(&self) -> &'static str {
        "EagerProcedureBoundary"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        self.child.install_resource_context(resources)?;
        self.resources = Some(resources.clone());
        Ok(())
    }
}

pub(crate) fn procedure_value_matches(expected: &PropertyDataType, value: &Value) -> bool {
    match (expected, value) {
        (_, Value::Null) | (PropertyDataType::Any, _) => true,
        (PropertyDataType::Node, Value::Map(map)) => {
            map.contains_key(&PropertyKey::new("_id"))
                && map.contains_key(&PropertyKey::new("_labels"))
        }
        (PropertyDataType::Edge, Value::Map(map)) => {
            map.contains_key(&PropertyKey::new("_id"))
                && map.contains_key(&PropertyKey::new("_type"))
                && map.contains_key(&PropertyKey::new("_source"))
                && map.contains_key(&PropertyKey::new("_target"))
        }
        _ => expected.matches(value),
    }
}

/// Physical vector type used after a value has passed its procedure contract.
/// Graph elements are materialized maps at a RETURN boundary, so they retain a
/// generic vector while the runtime contract distinguishes NODE from EDGE.
pub(crate) fn procedure_logical_type(data_type: &PropertyDataType) -> LogicalType {
    match data_type {
        PropertyDataType::String => LogicalType::String,
        PropertyDataType::Int64 => LogicalType::Int64,
        PropertyDataType::Float64 => LogicalType::Float64,
        PropertyDataType::Bool => LogicalType::Bool,
        PropertyDataType::Date => LogicalType::Date,
        PropertyDataType::Time => LogicalType::Time,
        PropertyDataType::Timestamp => LogicalType::Timestamp,
        PropertyDataType::Duration => LogicalType::Duration,
        PropertyDataType::Bytes => LogicalType::Bytes,
        PropertyDataType::List
        | PropertyDataType::ListTyped(_)
        | PropertyDataType::Map
        | PropertyDataType::Node
        | PropertyDataType::Edge
        | PropertyDataType::Any => LogicalType::Any,
    }
}
