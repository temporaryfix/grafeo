//! Rust-to-JavaScript value conversions for WASM bindings.

use grafeo_common::types::Value;
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Query keys are own data properties, including `__proto__`. A null prototype
/// also prevents ordinary query output from invoking inherited host setters.
pub(crate) fn result_object() -> Object {
    Object::create(JsValue::NULL.unchecked_ref::<Object>())
}

/// Converts a Grafeo [`Value`] to a JavaScript value.
pub fn value_to_js(value: &Value) -> JsValue {
    match value {
        Value::Null => JsValue::NULL,
        Value::Bool(b) => JsValue::from_bool(*b),
        Value::Int64(n) => {
            if *n > -(1i64 << 53) && *n < (1i64 << 53) {
                JsValue::from_f64(*n as f64)
            } else {
                js_sys::BigInt::from(*n).into()
            }
        }
        Value::Float64(f) => JsValue::from_f64(*f),
        Value::String(s) => JsValue::from_str(s),
        Value::Bytes(b) => {
            let arr = Uint8Array::new_with_length(b.len() as u32);
            arr.copy_from(b);
            arr.into()
        }
        Value::Timestamp(ts) => JsValue::from_str(&ts.to_string()),
        Value::Date(d) => JsValue::from_str(&d.to_string()),
        Value::Time(t) => JsValue::from_str(&t.to_string()),
        Value::Duration(d) => JsValue::from_str(&d.to_string()),
        Value::ZonedDatetime(zdt) => JsValue::from_str(&zdt.to_string()),
        Value::List(items) => {
            let arr = Array::new_with_length(items.len() as u32);
            for (i, item) in items.iter().enumerate() {
                arr.set(i as u32, value_to_js(item));
            }
            arr.into()
        }
        Value::Map(map) => {
            let obj = result_object();
            for (key, val) in map.iter() {
                let _ = Reflect::set(&obj, &JsValue::from_str(key.as_str()), &value_to_js(val));
            }
            obj.into()
        }
        Value::Vector(v) => {
            let arr = Float32Array::new_with_length(v.len() as u32);
            arr.copy_from(v);
            arr.into()
        }
        Value::Path { nodes, edges } => {
            let obj = result_object();
            let nodes_arr = Array::new_with_length(nodes.len() as u32);
            for (i, node) in nodes.iter().enumerate() {
                nodes_arr.set(i as u32, value_to_js(node));
            }
            let edges_arr = Array::new_with_length(edges.len() as u32);
            for (i, edge) in edges.iter().enumerate() {
                edges_arr.set(i as u32, value_to_js(edge));
            }
            let _ = Reflect::set(&obj, &JsValue::from_str("nodes"), &nodes_arr.into());
            let _ = Reflect::set(&obj, &JsValue::from_str("edges"), &edges_arr.into());
            let _ = Reflect::set(
                &obj,
                &JsValue::from_str("_type"),
                &JsValue::from_str("path"),
            );
            obj.into()
        }
        Value::GCounter(counts) => {
            let obj = result_object();
            for (replica, count) in counts.iter() {
                let _ = Reflect::set(
                    &obj,
                    &JsValue::from_str(replica),
                    &JsValue::from_f64(*count as f64),
                );
            }
            let wrapper = result_object();
            let _ = Reflect::set(&wrapper, &JsValue::from_str("$gcounter"), &obj.into());
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$value"),
                &JsValue::from_f64(counts.values().copied().map(|v| v as f64).sum()),
            );
            wrapper.into()
        }
        Value::OnCounter { pos, neg } => {
            let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
            let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
            let net = if pos_sum >= neg_sum {
                (pos_sum - neg_sum) as f64
            } else {
                -((neg_sum - pos_sum) as f64)
            };
            let wrapper = result_object();
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$pncounter"),
                &JsValue::from_str("pncounter"),
            );
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$value"),
                &JsValue::from_f64(net),
            );
            wrapper.into()
        }
        _ => JsValue::from_str(&value.to_string()),
    }
}

/// Converts a row of values to a JavaScript object with column names as keys.
pub fn row_to_js_object(columns: &[String], row: &[Value]) -> JsValue {
    let obj = result_object();
    for (col, val) in columns.iter().zip(row.iter()) {
        let _ = Reflect::set(&obj, &JsValue::from_str(col), &value_to_js(val));
    }
    obj.into()
}

type CopyResult<T> = grafeo_common::utils::error::Result<T>;

pub(crate) fn copy_limit_error() -> grafeo_common::utils::error::Error {
    use grafeo_common::utils::error::{Error, StorageError};
    Error::Storage(StorageError::Full)
        .with_context("WASM result conversion exceeds maxBytes or capacity")
}

/// Pure preallocation admission shared by eager rows, raw arrays, and streams.
/// Covers Rust row cache/retained values, JS handles, UTF-16 and container copies.
#[derive(Clone, Copy)]
pub(crate) struct CopyBudget {
    remaining: usize,
}

impl CopyBudget {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            remaining: max_bytes,
        }
    }

    pub(crate) fn charge(&mut self, bytes: usize) -> CopyResult<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(copy_limit_error)?;
        Ok(())
    }

    pub(crate) fn repeated(&mut self, count: usize, bytes: usize) -> CopyResult<()> {
        self.charge(count.checked_mul(bytes).ok_or_else(copy_limit_error)?)
    }

    pub(crate) fn string(&mut self, text: &str) -> CopyResult<()> {
        self.charge(128)?;
        self.repeated(text.len(), 8)
    }

    pub(crate) fn list(&mut self, count: usize) -> CopyResult<()> {
        u32::try_from(count).map_err(|_| copy_limit_error())?;
        self.charge(128)?;
        self.repeated(count, 64)
    }

    pub(crate) fn dict(&mut self, count: usize) -> CopyResult<()> {
        u32::try_from(count).map_err(|_| copy_limit_error())?;
        self.charge(1024)?;
        self.repeated(count, 384)
    }

    pub(crate) fn columns(&mut self, columns: &[String]) -> CopyResult<()> {
        self.list(columns.len())?;
        for name in columns {
            self.string(name)?;
        }
        Ok(())
    }

    pub(crate) fn row(&mut self, columns: &[String], values: &[Value]) -> CopyResult<()> {
        // Bound both the object-row and raw-array facades with one callback.
        self.dict(columns.len())?;
        self.list(values.len())?;
        for name in columns {
            self.string(name)?;
        }
        for value in values {
            self.value(value)?;
        }
        Ok(())
    }

    pub(crate) fn value(&mut self, value: &Value) -> CopyResult<()> {
        self.nested(value, 0)?;
        self.charge(value.retained_size_bytes().ok_or_else(copy_limit_error)?)
    }

    fn nested(&mut self, value: &Value, depth: usize) -> CopyResult<()> {
        if depth >= 128 {
            return Err(copy_limit_error());
        }
        self.charge(128)?;
        match value {
            Value::Null | Value::Bool(_) | Value::Int64(_) | Value::Float64(_) => Ok(()),
            Value::String(text) => self.string(text.as_str()),
            Value::Bytes(bytes) => {
                self.list(bytes.len())?;
                self.repeated(bytes.len(), 1)
            }
            Value::Vector(values) => {
                self.list(values.len())?;
                self.repeated(values.len(), 4)
            }
            Value::List(values) => {
                self.list(values.len())?;
                for value in values.iter() {
                    self.nested(value, depth + 1)?;
                }
                Ok(())
            }
            Value::Map(values) => {
                self.dict(values.len())?;
                for (key, value) in values.iter() {
                    self.string(key.as_str())?;
                    self.nested(value, depth + 1)?;
                }
                Ok(())
            }
            Value::Path { nodes, edges } => {
                self.dict(3)?;
                for text in ["nodes", "edges", "_type", "path"] {
                    self.string(text)?;
                }
                for values in [nodes, edges] {
                    self.list(values.len())?;
                    for value in values.iter() {
                        self.nested(value, depth + 1)?;
                    }
                }
                Ok(())
            }
            Value::GCounter(values) => {
                self.dict(2)?;
                self.string("$gcounter")?;
                self.string("$value")?;
                self.dict(values.len())?;
                for key in values.keys() {
                    self.string(key)?;
                    self.charge(128)?;
                }
                Ok(())
            }
            Value::OnCounter { .. } => {
                self.dict(2)?;
                for text in ["$pncounter", "$value", "pncounter"] {
                    self.string(text)?;
                }
                self.charge(128)
            }
            Value::Timestamp(_)
            | Value::Date(_)
            | Value::Time(_)
            | Value::Duration(_)
            | Value::ZonedDatetime(_)
            | Value::RdfLiteral { .. } => {
                let bytes = display_bytes(value, self.remaining / 8)?;
                self.charge(128)?;
                self.repeated(bytes, 8)
            }
            _ => Err(copy_limit_error()),
        }
    }
}

fn display_bytes(value: &Value, limit: usize) -> CopyResult<usize> {
    struct Counter {
        bytes: usize,
        limit: usize,
    }
    impl std::fmt::Write for Counter {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.bytes = self.bytes.checked_add(text.len()).ok_or(std::fmt::Error)?;
            if self.bytes > self.limit {
                return Err(std::fmt::Error);
            }
            Ok(())
        }
    }
    let mut counter = Counter { bytes: 0, limit };
    std::fmt::write(&mut counter, format_args!("{value}")).map_err(|_| copy_limit_error())?;
    Ok(counter.bytes)
}

pub(crate) fn preflight_result(
    result: &grafeo_engine::database::QueryResult,
    max_bytes: usize,
) -> CopyResult<()> {
    let mut budget = CopyBudget::new(max_bytes);
    budget.dict(3)?;
    for name in ["columns", "rows", "executionTimeMs"] {
        budget.string(name)?;
    }
    budget.columns(&result.columns)?;
    budget.list(result.row_count())?;
    if result.is_int64_columnar() {
        // Do not materialize rows()' lazy cache during admission.
        for _ in 0..result.row_count() {
            budget.dict(result.columns.len())?;
            budget.list(result.columns.len())?;
            for name in &result.columns {
                budget.string(name)?;
                budget.value(&Value::Int64(0))?;
            }
        }
    } else {
        for row in result.rows() {
            budget.row(&result.columns, row)?;
        }
    }
    Ok(())
}

/// Converts a fully admitted eager result without another fallible budget gate.
pub(crate) fn rows_to_js(result: &grafeo_engine::database::QueryResult) -> JsValue {
    let rows = Array::new();
    for row in result.rows() {
        rows.push(&row_to_js_object(&result.columns, row));
    }
    rows.into()
}

/// Raw columns/row arrays use the same precommit envelope as object rows.
pub(crate) fn raw_result_to_js(result: &grafeo_engine::database::QueryResult) -> JsValue {
    let object = result_object();
    let columns = Array::new();
    for name in &result.columns {
        columns.push(&JsValue::from_str(name));
    }
    let rows = Array::new();
    for row in result.rows() {
        let values = Array::new();
        for value in row {
            values.push(&value_to_js(value));
        }
        rows.push(&values);
    }
    let _ = Reflect::set(&object, &JsValue::from_str("columns"), &columns);
    let _ = Reflect::set(&object, &JsValue::from_str("rows"), &rows);
    if let Some(time) = result.execution_time_ms {
        let _ = Reflect::set(
            &object,
            &JsValue::from_str("executionTimeMs"),
            &JsValue::from_f64(time),
        );
    }
    object.into()
}
