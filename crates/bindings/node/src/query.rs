//! Query results for the Node.js API.

use napi::bindgen_prelude::*;
use napi::sys;
use napi_derive::napi;

use grafeo_common::types::Value;
use grafeo_engine::database::OwnedRows;

use crate::error::native_to_js_error;
use crate::graph::{JsEdge, JsNode};
use crate::types;

/// Pure admission callback: executed by the engine before a statement commits.
pub(crate) fn admit_node_result(
    result: &grafeo_engine::database::QueryResult,
    limits: grafeo_engine::query::ResultLimits,
) -> grafeo_common::utils::error::Result<()> {
    preflight_node_result(result, limits.max_bytes)
}

pub(crate) fn preflight_node_result(
    result: &grafeo_engine::database::QueryResult,
    max_bytes: usize,
) -> grafeo_common::utils::error::Result<()> {
    let mut budget = types::CopyBudget::new(max_bytes);
    budget.columns(&result.columns)?;
    budget.repeated(
        result.column_types.capacity(),
        std::mem::size_of::<grafeo_common::LogicalType>(),
    )?;
    for column_type in &result.column_types {
        budget.logical_type(column_type, 0)?;
    }
    if let Some(status) = &result.status_message {
        budget.charge(status.capacity())?;
    }
    budget.list(result.row_count())?;
    // Empty entity vectors and their copied array headers remain present even
    // for status-only results, and must be covered before publication too.
    budget.list(0)?;
    budget.list(0)?;
    if result.is_int64_columnar() {
        // rows() would lazily allocate the dense result's row cache. This pass
        // only borrows columns and must precede every copied allocation.
        for _ in 0..result.row_count() {
            budget.dict(result.columns.len())?;
            for name in &result.columns {
                budget.string(name)?;
                budget.value(&Value::Int64(0))?;
            }
        }
    } else {
        for row in result.rows() {
            budget.row(&result.columns, row)?;
            // Entity extraction can additionally clone map/label containers and
            // retain one JavaScript wrapper for each top-level entity-shaped map.
            for value in row {
                if matches!(value, Value::Map(_)) {
                    budget.charge(512)?;
                    budget.value(value)?;
                }
            }
        }
    }
    Ok(())
}

/// Admit the retained entity wrappers and a copied nodes()/edges() result.
pub(crate) fn admit_entity_copies(
    budget: &mut types::CopyBudget,
    nodes: &[JsNode],
    edges: &[JsEdge],
) -> grafeo_common::utils::error::Result<()> {
    // Entity wrappers clone labels and property hash tables; nested Value
    // payloads are Arc-shared but charged again conservatively.
    budget.list(nodes.len())?;
    budget.list(edges.len())?;
    for node in nodes {
        budget.charge(256)?;
        budget.columns(&node.labels)?;
        budget.dict(node.properties.len())?;
        for (key, value) in &node.properties {
            budget.string(key.as_str())?;
            budget.value(value)?;
        }
    }
    for edge in edges {
        budget.charge(256)?;
        budget.string(&edge.edge_type)?;
        budget.dict(edge.properties.len())?;
        for (key, value) in &edge.properties {
            budget.string(key.as_str())?;
            budget.value(value)?;
        }
    }
    Ok(())
}

/// Results from a query - access rows, nodes, and edges.
#[napi]
pub struct QueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: OwnedRows,
    pub(crate) nodes: Vec<JsNode>,
    pub(crate) edges: Vec<JsEdge>,
    pub(crate) execution_time_ms: Option<f64>,
    pub(crate) rows_scanned: Option<u64>,
    conversion_limit: usize,
}

#[napi]
impl QueryResult {
    /// Get column names.
    #[napi(getter)]
    pub fn columns(&self, env: Env) -> Result<Vec<String>> {
        types::bounded_columns(&self.columns, self.conversion_limit)
            .map_err(|error| native_to_js_error(&env, error))
    }

    /// Get number of rows.
    #[napi(getter)]
    pub fn length(&self, env: Env) -> Result<u32> {
        u32::try_from(self.rows.len())
            .map_err(|_| native_to_js_error(&env, types::copy_limit_error()))
    }

    /// Query execution time in milliseconds (if available).
    #[napi(getter, js_name = "executionTimeMs")]
    pub fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Number of rows scanned during execution (if available).
    #[napi(getter, js_name = "rowsScanned")]
    pub fn rows_scanned(&self) -> Option<f64> {
        self.rows_scanned.map(|r| r as f64)
    }

    /// Get a single row by index as a plain object.
    #[napi]
    pub fn get(&self, env: Env, index: u32) -> Result<Object<'_>> {
        let idx = index as usize;
        if idx >= self.rows.len() {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "Row index out of range",
            ));
        }
        self.row_to_object(&env, idx)
    }

    /// Get all rows as an array of objects.
    #[napi(js_name = "toArray")]
    pub fn to_array(&self, env: Env) -> Result<Vec<Object<'_>>> {
        self.copy_budget()
            .map_err(|error| native_to_js_error(&env, error))?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.rows.len())
            .map_err(|_| native_to_js_error(&env, types::copy_limit_error()))?;
        for i in 0..self.rows.len() {
            result.push(self.row_to_object(&env, i)?);
        }
        Ok(result)
    }

    /// Get first column of first row (single value).
    #[napi]
    pub fn scalar(&self, env: Env) -> Result<Unknown<'_>> {
        if self.rows.is_empty() {
            return Err(napi::Error::new(
                napi::Status::GenericFailure,
                "No rows in result",
            ));
        }
        if self.columns.is_empty() {
            return Err(napi::Error::new(
                napi::Status::GenericFailure,
                "No columns in result",
            ));
        }
        let value = &self.rows[0][0];
        types::CopyBudget::new(self.conversion_limit)
            .value(value)
            .map_err(|error| native_to_js_error(&env, error))?;
        let raw = types::value_to_napi_admitted(env.raw(), value)?;
        // SAFETY: this is the same callback environment and the admitted
        // converter returned a live napi value without a borrowed native view.
        Ok(unsafe { Unknown::from_raw_unchecked(env.raw(), raw) })
    }

    /// Get nodes found in the result.
    #[napi]
    pub fn nodes(&self, env: Env) -> Result<Vec<JsNode>> {
        self.copy_budget()
            .map_err(|error| native_to_js_error(&env, error))?;
        Ok(self.nodes.clone())
    }

    /// Get edges found in the result.
    #[napi]
    pub fn edges(&self, env: Env) -> Result<Vec<JsEdge>> {
        self.copy_budget()
            .map_err(|error| native_to_js_error(&env, error))?;
        Ok(self.edges.clone())
    }

    /// Returns the result formatted as a Unicode table.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self, env: Env) -> Result<String> {
        let mut budget = self
            .copy_budget()
            .map_err(|error| native_to_js_error(&env, error))?;
        budget
            .charge(4096)
            .map_err(|error| native_to_js_error(&env, error))?;
        let lines = self
            .rows
            .len()
            .checked_add(4)
            .ok_or_else(|| native_to_js_error(&env, types::copy_limit_error()))?;
        for column in &self.columns {
            budget
                .repeated(column.len(), 32)
                .map_err(|error| native_to_js_error(&env, error))?;
            budget
                .repeated(lines, 1024)
                .map_err(|error| native_to_js_error(&env, error))?;
        }
        for row in &self.rows {
            budget
                .charge(128)
                .map_err(|error| native_to_js_error(&env, error))?;
            for value in row {
                let size = types::display_bytes(value, self.conversion_limit)
                    .map_err(|error| native_to_js_error(&env, error))?;
                budget
                    .repeated(size, 16)
                    .map_err(|error| native_to_js_error(&env, error))?;
                budget
                    .charge(128)
                    .map_err(|error| native_to_js_error(&env, error))?;
            }
        }
        Ok(grafeo_common::fmt::format_result_table(
            &self.columns,
            &self.rows,
            self.execution_time_ms,
            None,
        ))
    }

    /// Get all rows as an array of arrays (no column names).
    #[napi]
    pub fn rows(&self, env: Env) -> Result<Object<'_>> {
        self.copy_budget()
            .map_err(|error| native_to_js_error(&env, error))?;
        let env_raw = env.raw();
        let mut arr = std::ptr::null_mut();
        // SAFETY: env_raw is valid; napi_create_array_with_length writes to our out-pointer
        types::check_napi(unsafe {
            sys::napi_create_array_with_length(env_raw, self.rows.len(), &raw mut arr)
        })?;
        for (i, row) in self.rows.iter().enumerate() {
            let mut row_arr = std::ptr::null_mut();
            // SAFETY: env_raw is valid; napi_create_array_with_length writes to our out-pointer
            types::check_napi(unsafe {
                sys::napi_create_array_with_length(env_raw, row.len(), &raw mut row_arr)
            })?;
            for (j, val) in row.iter().enumerate() {
                let napi_val = types::value_to_napi_admitted(env_raw, val)?;
                // SAFETY: env_raw, row_arr, and napi_val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements
                #[allow(clippy::cast_possible_truncation)]
                types::check_napi(unsafe {
                    sys::napi_set_element(env_raw, row_arr, j as u32, napi_val)
                })?;
            }
            // SAFETY: env_raw, arr, and row_arr are valid napi values
            // reason: JS arrays are limited to 2^32-1 elements
            #[allow(clippy::cast_possible_truncation)]
            types::check_napi(unsafe { sys::napi_set_element(env_raw, arr, i as u32, row_arr) })?;
        }
        Ok(Object::from_raw(env_raw, arr))
    }
}

impl QueryResult {
    pub(crate) fn with_conversion_limit(mut self, max_bytes: usize) -> Self {
        self.conversion_limit = max_bytes;
        for node in &mut self.nodes {
            node.set_conversion_limit(max_bytes);
        }
        for edge in &mut self.edges {
            edge.set_conversion_limit(max_bytes);
        }
        self
    }

    fn copy_budget(&self) -> grafeo_common::Result<types::CopyBudget> {
        let mut budget = types::CopyBudget::new(self.conversion_limit);
        budget.columns(&self.columns)?;
        budget.list(self.rows.len())?;
        for row in &self.rows {
            budget.row(&self.columns, row)?;
        }
        admit_entity_copies(&mut budget, &self.nodes, &self.edges)?;
        Ok(budget)
    }

    /// Convert a row to a JS object with column names as keys.
    fn row_to_object(&self, env: &Env, idx: usize) -> Result<Object<'_>> {
        let row = &self.rows[idx];
        types::CopyBudget::new(self.conversion_limit)
            .row(&self.columns, row)
            .map_err(|error| native_to_js_error(env, error))?;
        let env = env.raw();
        let mut raw_obj = std::ptr::null_mut();
        // SAFETY: env is valid; napi_create_object writes to our out-pointer
        types::check_napi(unsafe { sys::napi_create_object(env, &raw mut raw_obj) })?;
        let mut obj = Object::from_raw(env, raw_obj);
        for (col, val) in self.columns.iter().zip(row.iter()) {
            let val_raw = types::value_to_napi_admitted(env, val)?;
            // SAFETY: env and val_raw are valid napi values produced by value_to_napi
            let val_unknown = unsafe { Unknown::from_raw_unchecked(env, val_raw) };
            obj.set_named_property(col, val_unknown)?;
        }
        Ok(obj)
    }

    pub fn new(
        columns: Vec<String>,
        rows: OwnedRows,
        nodes: Vec<JsNode>,
        edges: Vec<JsEdge>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            execution_time_ms: None,
            rows_scanned: None,
            conversion_limit: types::default_conversion_limit(),
        }
    }

    pub fn with_metrics(
        columns: Vec<String>,
        rows: OwnedRows,
        nodes: Vec<JsNode>,
        edges: Vec<JsEdge>,
        execution_time_ms: Option<f64>,
        rows_scanned: Option<u64>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            execution_time_ms,
            rows_scanned,
            conversion_limit: types::default_conversion_limit(),
        }
    }

    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: OwnedRows::empty(),
            nodes: Vec::new(),
            edges: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            conversion_limit: types::default_conversion_limit(),
        }
    }
}

#[cfg(feature = "arrow-export")]
#[napi]
impl QueryResult {
    /// Returns the result as Arrow IPC stream bytes (Buffer).
    ///
    /// Use with the `apache-arrow` npm package:
    /// ```js
    /// import { tableFromIPC } from 'apache-arrow';
    /// const table = tableFromIPC(result.toArrowIPC());
    /// ```
    #[napi(js_name = "toArrowIPC")]
    pub fn to_arrow_ipc(&self, env: Env) -> Result<napi::bindgen_prelude::Buffer> {
        let mut budget = self
            .copy_budget()
            .map_err(|error| native_to_js_error(&env, error))?;
        // Reference columns, builders, IPC buffers and the returned Buffer
        // coexist. Include initial builder capacities, alignment and growth.
        budget
            .charge(16384)
            .map_err(|error| native_to_js_error(&env, error))?;
        for column in &self.columns {
            budget
                .charge(4096)
                .map_err(|error| native_to_js_error(&env, error))?;
            budget
                .repeated(column.len(), 32)
                .map_err(|error| native_to_js_error(&env, error))?;
            budget
                .repeated(self.rows.len(), 1024)
                .map_err(|error| native_to_js_error(&env, error))?;
        }
        for row in &self.rows {
            for value in row {
                let size = types::display_bytes(value, self.conversion_limit)
                    .map_err(|error| native_to_js_error(&env, error))?;
                budget
                    .repeated(size, 32)
                    .map_err(|error| native_to_js_error(&env, error))?;
                match value {
                    Value::Vector(values) => {
                        budget
                            .repeated(values.len(), 32)
                            .map_err(|error| native_to_js_error(&env, error))?;
                    }
                    Value::Bytes(values) => {
                        budget
                            .repeated(values.len(), 16)
                            .map_err(|error| native_to_js_error(&env, error))?;
                    }
                    _ => {}
                }
            }
        }
        let col_types = vec![grafeo_common::LogicalType::Any; self.columns.len()];
        let batch = grafeo_engine::database::arrow::query_result_to_record_batch(
            &self.columns,
            &col_types,
            &self.rows,
        )
        .map_err(|e| {
            native_to_js_error(
                &env,
                grafeo_common::Error::Serialization(format!("Arrow export failed: {e}")),
            )
        })?;
        let ipc_bytes = grafeo_engine::database::arrow::record_batch_to_ipc_stream(&batch)
            .map_err(|e| {
                native_to_js_error(
                    &env,
                    grafeo_common::Error::Serialization(format!("Arrow IPC failed: {e}")),
                )
            })?;
        Ok(ipc_bytes.into())
    }
}
