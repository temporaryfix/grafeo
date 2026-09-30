//! Converts between Python and Grafeo value types automatically.
//!
//! | Python type | Grafeo type | Notes |
//! | ----------- | ----------- | ----- |
//! | `None` | `Null` | |
//! | `bool` | `Bool` | |
//! | `int` | `Int64` | |
//! | `float` | `Float64` | |
//! | `str` | `String` | |
//! | `list[float]` | `Vector` | Lists where every element is a Python float (not int) |
//! | `list` | `List` | All other lists converted recursively |
//! | `dict` | `Map` | Keys must be strings |
//! | `bytes` | `Bytes` | |
//! | `datetime` | `Timestamp` | Converted to/from UTC |

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDateTime, PyDict, PyFloat, PyList};

use grafeo_common::types::{PropertyKey, Timestamp, Value};

use crate::error::{PyGrafeoError, PyGrafeoResult};

/// Wraps a Grafeo value for explicit type handling.
///
/// Usually you don't need this - Python types convert automatically. Use this
/// when you need explicit control like `Value.null()` or type checking.
#[pyclass(name = "Value", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyValue {
    pub(crate) inner: Value,
}

#[pymethods]
impl PyValue {
    /// Create a null value.
    #[staticmethod]
    fn null() -> Self {
        Self { inner: Value::Null }
    }

    /// Create a boolean value.
    #[staticmethod]
    fn boolean(v: bool) -> Self {
        Self {
            inner: Value::Bool(v),
        }
    }

    /// Create an integer value.
    #[staticmethod]
    fn integer(v: i64) -> Self {
        Self {
            inner: Value::Int64(v),
        }
    }

    /// Create a float value.
    #[staticmethod]
    fn float(v: f64) -> Self {
        Self {
            inner: Value::Float64(v),
        }
    }

    /// Create a string value.
    #[staticmethod]
    fn string(v: String) -> Self {
        Self {
            inner: Value::String(v.into()),
        }
    }

    /// Check if value is null.
    fn is_null(&self) -> bool {
        matches!(self.inner, Value::Null)
    }

    /// Get boolean value.
    fn as_bool(&self) -> PyGrafeoResult<bool> {
        match &self.inner {
            Value::Bool(v) => Ok(*v),
            _ => Err(PyGrafeoError::Type("Value is not a boolean".into())),
        }
    }

    /// Get integer value.
    fn as_int(&self) -> PyGrafeoResult<i64> {
        match &self.inner {
            Value::Int64(v) => Ok(*v),
            _ => Err(PyGrafeoError::Type("Value is not an integer".into())),
        }
    }

    /// Get float value.
    fn as_float(&self) -> PyGrafeoResult<f64> {
        match &self.inner {
            Value::Float64(v) => Ok(*v),
            _ => Err(PyGrafeoError::Type("Value is not a float".into())),
        }
    }

    /// Get string value.
    fn as_str(&self) -> PyGrafeoResult<String> {
        match &self.inner {
            Value::String(v) => {
                CopyBudget::new(default_conversion_limit())
                    .string(v.as_str())
                    .map_err(PyGrafeoError::from)?;
                Ok(v.to_string())
            }
            _ => Err(PyGrafeoError::Type("Value is not a string".into())),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        self.admit_debug()?;
        Ok(format!("Value({:?})", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        self.admit_debug()?;
        Ok(format!("{:?}", self.inner))
    }
}

impl PyValue {
    fn admit_debug(&self) -> PyResult<()> {
        let limit = default_conversion_limit();
        let mut budget = CopyBudget::new(limit);
        budget.value(&self.inner).map_err(copy_error)?;
        let bytes = display_bytes(&format_args!("{:?}", self.inner), limit).map_err(copy_error)?;
        budget.charge(128).map_err(copy_error)?;
        budget.repeated(bytes, 8).map_err(copy_error)
    }

    /// Converts a Python object to a Grafeo Value.
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyGrafeoResult<Value> {
        if obj.is_none() {
            return Ok(Value::Null);
        }

        if let Ok(v) = obj.extract::<bool>() {
            return Ok(Value::Bool(v));
        }

        if let Ok(v) = obj.extract::<i64>() {
            return Ok(Value::Int64(v));
        }

        if let Ok(v) = obj.extract::<f64>() {
            return Ok(Value::Float64(v));
        }

        if let Ok(v) = obj.extract::<String>() {
            return Ok(Value::String(v.into()));
        }

        if let Ok(v) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
            // Only convert to Vector when ALL elements are Python floats (not ints
            // coerced to float). This prevents [1, 2, 3] from being stored as an
            // embedding vector instead of a general-purpose list.
            if !v.is_empty() && v.iter().all(|item| item.is_instance_of::<PyFloat>()) {
                let floats: Result<Vec<f32>, _> =
                    v.iter().map(|item| item.extract::<f32>()).collect();
                if let Ok(floats) = floats {
                    return Ok(Value::Vector(floats.into()));
                }
            }

            let mut items = Vec::new();
            for item in v {
                items.push(Self::from_py(&item)?);
            }
            return Ok(Value::List(items.into()));
        }

        if obj.is_instance_of::<PyDict>() {
            // SAFETY: We just checked it's a PyDict instance
            let dict: &Bound<'_, PyDict> = obj
                .cast()
                .map_err(|e| PyGrafeoError::Type(format!("Cannot cast to dict: {}", e)))?;
            let mut map = BTreeMap::new();
            for (key, value) in dict.iter() {
                let key_str: String = key
                    .extract()
                    .map_err(|e| PyGrafeoError::Type(format!("Dict key must be string: {}", e)))?;
                map.insert(PropertyKey::new(key_str), Self::from_py(&value)?);
            }
            return Ok(Value::Map(Arc::new(map)));
        }

        // Handle bytes
        if obj.is_instance_of::<PyBytes>() {
            let bytes: &Bound<'_, PyBytes> = obj
                .cast()
                .map_err(|e| PyGrafeoError::Type(format!("Cannot cast to bytes: {}", e)))?;
            let byte_slice: &[u8] = bytes.as_bytes();
            return Ok(Value::Bytes(byte_slice.into()));
        }

        // Handle datetime
        if obj.is_instance_of::<PyDateTime>() {
            // Extract timestamp as float (seconds since epoch)
            let timestamp: f64 = obj
                .call_method0("timestamp")
                .and_then(|ts| ts.extract())
                .map_err(|e| {
                    PyGrafeoError::Type(format!("Failed to get datetime timestamp: {}", e))
                })?;
            // reason: Convert to microseconds; Python timestamps are within i64 range
            #[allow(clippy::cast_possible_truncation)]
            let micros = (timestamp * 1_000_000.0) as i64;
            return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
        }

        let type_name = obj
            .get_type()
            .name()
            .map_or_else(|_| "<unknown>".to_string(), |s| s.to_string());
        Err(PyGrafeoError::Type(format!(
            "Unsupported Python type: {}",
            type_name
        )))
    }

    /// Converts a value after admitting its complete copied representation.
    pub fn to_py(value: &Value, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::to_py_bounded(value, py, default_conversion_limit())
    }

    pub(crate) fn to_py_bounded(
        value: &Value,
        py: Python<'_>,
        max_bytes: usize,
    ) -> PyResult<Py<PyAny>> {
        let mut budget = CopyBudget::new(max_bytes);
        budget.value(value).map_err(copy_error)?;
        Self::to_py_admitted(value, py)
    }

    /// Only called after a whole value/row/result has passed CopyBudget.
    pub(crate) fn to_py_admitted(value: &Value, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::conversion::IntoPyObjectExt;
        match value {
            Value::Null => Ok(py.None()),
            Value::Bool(v) => v.into_py_any(py),
            Value::Int64(v) => v.into_py_any(py),
            Value::Float64(v) => v.into_py_any(py),
            Value::String(v) => v.as_str().into_py_any(py),
            Value::List(items) => values_to_list_admitted(py, items),
            Value::Map(map) => {
                let dict = new_dict(py)?;
                for (key, value) in map.iter() {
                    dict.set_item(key.as_str(), Self::to_py_admitted(value, py)?)?;
                }
                Ok(dict.unbind().into_any())
            }
            Value::Bytes(bytes) => Ok(PyBytes::new_with(py, bytes.len(), |buffer| {
                buffer.copy_from_slice(bytes);
                Ok(())
            })?
            .unbind()
            .into_any()),
            Value::Timestamp(ts) => {
                let module = py.import("datetime")?;
                let utc = module.getattr("timezone")?.getattr("utc")?;
                Ok(module
                    .getattr("datetime")?
                    .call_method1("fromtimestamp", (ts.as_micros() as f64 / 1_000_000.0, utc))?
                    .unbind())
            }
            Value::Date(date) => Ok(py
                .import("datetime")?
                .getattr("date")?
                .call1((date.year(), date.month(), date.day()))?
                .unbind()),
            Value::Time(time) => Ok(py
                .import("datetime")?
                .getattr("time")?
                .call1((
                    time.hour(),
                    time.minute(),
                    time.second(),
                    time.nanosecond() / 1000,
                ))?
                .unbind()),
            Value::Duration(duration) => {
                let dict = new_dict(py)?;
                dict.set_item("months", duration.months())?;
                dict.set_item("days", duration.days())?;
                dict.set_item("nanos", duration.nanos())?;
                Ok(dict.unbind().into_any())
            }
            Value::ZonedDatetime(datetime) => {
                let module = py.import("datetime")?;
                let date = datetime.to_local_date();
                let time = datetime.to_local_time();
                let delta = module
                    .getattr("timedelta")?
                    .call1((0, datetime.offset_seconds()))?;
                let timezone = module.getattr("timezone")?.call1((delta,))?;
                Ok(module
                    .getattr("datetime")?
                    .call1((
                        date.year(),
                        date.month(),
                        date.day(),
                        time.hour(),
                        time.minute(),
                        time.second(),
                        time.nanosecond() / 1000,
                        timezone,
                    ))?
                    .unbind())
            }
            Value::Vector(values) => {
                let list = new_list(py)?;
                for value in values.iter() {
                    list.append(*value)?;
                }
                Ok(list.unbind().into_any())
            }
            Value::Path { nodes, edges } => {
                let dict = new_dict(py)?;
                dict.set_item("nodes", values_to_list_admitted(py, nodes)?)?;
                dict.set_item("edges", values_to_list_admitted(py, edges)?)?;
                Ok(dict.unbind().into_any())
            }
            Value::GCounter(counts) => {
                let dict = new_dict(py)?;
                let replicas = new_dict(py)?;
                let mut total = 0_u128;
                for (replica, count) in counts.iter() {
                    total += u128::from(*count);
                    replicas.set_item(replica.as_str(), *count)?;
                }
                dict.set_item("$gcounter", replicas)?;
                dict.set_item("$value", total)?;
                Ok(dict.unbind().into_any())
            }
            Value::OnCounter { pos, neg } => {
                let positive: u128 = pos.values().copied().map(u128::from).sum();
                let negative: u128 = neg.values().copied().map(u128::from).sum();
                let positive =
                    i128::try_from(positive).map_err(|_| copy_error(copy_limit_error()))?;
                let negative =
                    i128::try_from(negative).map_err(|_| copy_error(copy_limit_error()))?;
                let dict = new_dict(py)?;
                dict.set_item("$pncounter", true)?;
                dict.set_item("$value", positive - negative)?;
                Ok(dict.unbind().into_any())
            }
            _ => value.to_string().into_py_any(py),
        }
    }
}

/// The native default also applies to standalone Value conversions.
pub(crate) fn default_conversion_limit() -> usize {
    grafeo_engine::query::ResultLimits::default().max_bytes
}

type CopyResult<T> = grafeo_common::utils::error::Result<T>;

pub(crate) fn copy_limit_error() -> grafeo_common::utils::error::Error {
    use grafeo_common::utils::error::{Error, StorageError};
    Error::Storage(StorageError::Full).with_context("Python result conversion exceeds max_bytes")
}

pub(crate) fn copy_error(error: grafeo_common::utils::error::Error) -> PyErr {
    PyGrafeoError::from(error).into()
}

/// A no-allocation admission pass for binding-owned native and CPython copies.
///
/// Charges are conservative bounds for supported 64-bit CPython: 128 bytes per
/// scalar (native Value, Python object and argument/reference slots), 128 bytes
/// per dictionary entry (including resize overlap), and 32 bytes per list slot
/// (including native/reference slots and resize overlap). Strings charge their
/// native UTF-8 and the largest four-byte Python representation simultaneously.
/// The native retained-size bound is charged separately, including B-tree root
/// slack and counter hash-table capacity. Shared native allocations are
/// deliberately counted at every occurrence.
/// Imported modules and allocations inside third-party libraries are not owned
/// by this budget. No Python objects or formatted strings are made by this pass.
pub(crate) struct CopyBudget {
    remaining: usize,
}

#[cfg(test)]
std::thread_local! {
    static COPY_STRING_COST_EVALUATIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
pub(crate) fn take_copy_string_cost_evaluations() -> usize {
    COPY_STRING_COST_EVALUATIONS.with(|evaluations| evaluations.replace(0))
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
    pub(crate) fn string(&mut self, value: &str) -> CopyResult<()> {
        #[cfg(test)]
        COPY_STRING_COST_EVALUATIONS.with(|evaluations| evaluations.set(evaluations.get() + 1));
        self.charge(128)?;
        self.repeated(value.len(), 8)
    }
    pub(crate) fn list(&mut self, length: usize) -> CopyResult<()> {
        self.charge(128)?;
        self.repeated(length, 32)
    }
    pub(crate) fn dict(&mut self, length: usize) -> CopyResult<()> {
        self.charge(128)?;
        self.repeated(length, 128)
    }
    pub(crate) fn columns(&mut self, columns: &[String]) -> CopyResult<()> {
        self.list(columns.len())?;
        for column in columns {
            self.string(column)?;
        }
        Ok(())
    }
    /// Charge native nested schema storage retained alongside a copied result.
    pub(crate) fn logical_type(
        &mut self,
        logical_type: &grafeo_common::LogicalType,
        depth: usize,
    ) -> CopyResult<()> {
        use grafeo_common::LogicalType;
        if depth >= 256 {
            return Err(copy_limit_error());
        }
        match logical_type {
            LogicalType::List(item) => {
                self.charge(std::mem::size_of::<LogicalType>())?;
                self.logical_type(item, depth + 1)
            }
            LogicalType::Map { key, value } => {
                self.repeated(2, std::mem::size_of::<LogicalType>())?;
                self.logical_type(key, depth + 1)?;
                self.logical_type(value, depth + 1)
            }
            LogicalType::Struct(fields) => {
                self.repeated(
                    fields.capacity(),
                    std::mem::size_of::<(String, LogicalType)>(),
                )?;
                for (name, logical_type) in fields {
                    self.charge(name.capacity())?;
                    self.logical_type(logical_type, depth + 1)?;
                }
                Ok(())
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
            | LogicalType::Vector(_) => Ok(()),
            _ => Err(copy_limit_error()),
        }
    }
    pub(crate) fn row(&mut self, columns: &[String], values: &[Value]) -> CopyResult<()> {
        self.dict(columns.len())?;
        for column in columns {
            self.string(column)?;
        }
        for value in values {
            self.value(value)?;
        }
        Ok(())
    }
    /// Charges each copied row's identical dictionary and column names while
    /// measuring those names once. Values remain charged at every occurrence.
    pub(crate) fn repeated_row_headers(
        &mut self,
        columns: &[String],
        count: usize,
    ) -> CopyResult<()> {
        if count == 0 {
            return Ok(());
        }
        let mut header = Self::new(usize::MAX);
        header.row(columns, &[])?;
        self.repeated(count, usize::MAX - header.remaining)
    }
    pub(crate) fn value(&mut self, value: &Value) -> CopyResult<()> {
        self.nested_value(value, 0)?;
        self.charge(value.retained_size_bytes().ok_or_else(copy_limit_error)?)
    }
    fn nested_value(&mut self, value: &Value, depth: usize) -> CopyResult<()> {
        // Bound the conversion call stack as well as copied heap memory. This is
        // a resource failure, before recursion or Python allocation can overflow.
        if depth >= 256 {
            return Err(copy_limit_error());
        }
        self.charge(128)?;
        match value {
            Value::String(text) => self.string(text.as_str()),
            Value::Bytes(bytes) => self.repeated(bytes.len(), 2),
            Value::List(values) => {
                self.list(values.len())?;
                for value in values.iter() {
                    self.nested_value(value, depth + 1)?;
                }
                Ok(())
            }
            Value::Map(values) => {
                self.dict(values.len())?;
                for (key, value) in values.iter() {
                    self.string(key.as_str())?;
                    self.nested_value(value, depth + 1)?;
                }
                Ok(())
            }
            Value::Vector(values) => {
                self.list(values.len())?;
                self.repeated(values.len(), 64)
            }
            Value::Path { nodes, edges } => {
                self.dict(2)?;
                self.string("nodes")?;
                self.string("edges")?;
                for values in [nodes, edges] {
                    self.list(values.len())?;
                    for value in values.iter() {
                        self.nested_value(value, depth + 1)?;
                    }
                }
                Ok(())
            }
            Value::GCounter(values) => {
                self.dict(2)?;
                self.charge(512)?;
                self.dict(values.len())?;
                for key in values.keys() {
                    self.string(key.as_str())?;
                    self.charge(128)?;
                }
                Ok(())
            }
            Value::OnCounter { pos, neg } => {
                self.dict(2)?;
                self.charge(512)?;
                for values in [pos, neg] {
                    for key in values.keys() {
                        self.string(key.as_str())?;
                        self.charge(128)?;
                    }
                }
                Ok(())
            }
            Value::Timestamp(_)
            | Value::Date(_)
            | Value::Time(_)
            | Value::Duration(_)
            | Value::ZonedDatetime(_) => self.charge(4096),
            Value::Null | Value::Bool(_) | Value::Int64(_) | Value::Float64(_) => Ok(()),
            Value::RdfLiteral {
                lexical,
                language,
                datatype,
            } => {
                self.string(lexical.as_str())?;
                if let Some(language) = language {
                    self.string(language.as_str())?;
                }
                if let Some(datatype) = datatype {
                    self.string(datatype.as_str())?;
                }
                self.charge(128)
            }
            _ => Err(copy_limit_error()),
        }
    }
}

/// Count formatting output without creating the formatted string.
pub(crate) fn display_bytes(value: &impl std::fmt::Display, limit: usize) -> CopyResult<usize> {
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

pub(crate) fn new_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    Ok(py.get_type::<PyDict>().call0()?.cast_into::<PyDict>()?)
}
pub(crate) fn new_list(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    Ok(py.get_type::<PyList>().call0()?.cast_into::<PyList>()?)
}
fn values_to_list_admitted(py: Python<'_>, values: &[Value]) -> PyResult<Py<PyAny>> {
    let list = new_list(py)?;
    for value in values {
        list.append(PyValue::to_py_admitted(value, py)?)?;
    }
    Ok(list.unbind().into_any())
}
pub(crate) fn row_to_py_bounded(
    py: Python<'_>,
    columns: &[String],
    values: &[Value],
    max_bytes: usize,
) -> PyResult<Py<PyAny>> {
    CopyBudget::new(max_bytes)
        .row(columns, values)
        .map_err(copy_error)?;
    let dict = new_dict(py)?;
    for (column, value) in columns.iter().zip(values) {
        dict.set_item(column, PyValue::to_py_admitted(value, py)?)?;
    }
    Ok(dict.unbind().into_any())
}
pub(crate) fn columns_to_py_bounded(
    py: Python<'_>,
    columns: &[String],
    max_bytes: usize,
) -> PyResult<Py<PyAny>> {
    CopyBudget::new(max_bytes)
        .columns(columns)
        .map_err(copy_error)?;
    let list = new_list(py)?;
    for column in columns {
        list.append(column)?;
    }
    Ok(list.unbind().into_any())
}

impl From<Value> for PyValue {
    fn from(inner: Value) -> Self {
        Self { inner }
    }
}

impl From<PyValue> for Value {
    fn from(py_val: PyValue) -> Self {
        py_val.inner
    }
}

/// Creates a vector value from a list of floats.
///
/// Use this for explicit vector construction:
/// ```python
/// import grafeo
/// vec = grafeo.vector([0.1, 0.2, 0.3])
/// db.create_node(['Doc'], {'embedding': vec})
/// ```
///
/// Note: All-float Python lists are automatically converted to vectors,
/// so `[0.1, 0.2, 0.3]` works directly in most cases.
#[pyfunction]
pub fn vector(values: Vec<f32>) -> PyResult<Vec<f32>> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vector() requires at least one element",
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod copy_admission_tests {
    use super::*;

    #[test]
    fn repeated_row_headers_preserve_exact_charges_and_check_overflow() {
        let columns = vec!["id".into(), "😀label".into()];
        for count in [0, 1, 32, 64, 128] {
            let mut reference = CopyBudget::new(usize::MAX);
            for _ in 0..count {
                reference.row(&columns, &[]).unwrap();
            }
            let exact = usize::MAX - reference.remaining;
            let mut budget = CopyBudget::new(exact);
            budget.repeated_row_headers(&columns, count).unwrap();
            assert_eq!(budget.remaining, 0);
            if exact > 0 {
                assert!(
                    CopyBudget::new(exact - 1)
                        .repeated_row_headers(&columns, count)
                        .is_err()
                );
            }
        }
        take_copy_string_cost_evaluations();
        CopyBudget::new(0)
            .repeated_row_headers(&columns, 0)
            .unwrap();
        assert_eq!(take_copy_string_cost_evaluations(), 0);
        assert!(
            CopyBudget::new(usize::MAX)
                .repeated_row_headers(&columns, usize::MAX)
                .is_err()
        );
        assert!(
            CopyBudget::new(usize::MAX)
                .repeated_row_headers(&[], usize::MAX)
                .is_err()
        );
    }

    #[test]
    fn nested_copied_strings_are_admitted_before_conversion() {
        let value = Value::List(
            vec![Value::Map(Arc::new(BTreeMap::from([(
                PropertyKey::new("payload"),
                Value::from("😀".repeat(512)),
            )])))]
            .into(),
        );
        assert!(CopyBudget::new(4096).value(&value).is_err());
        assert!(
            CopyBudget::new(default_conversion_limit())
                .value(&value)
                .is_ok()
        );
    }

    #[test]
    fn repeated_row_keys_and_native_map_slack_are_charged() {
        let columns = vec!["column".repeat(256)];
        let values = [Value::Int64(1)];
        let mut budget = CopyBudget::new(32 * 1024);
        assert!(budget.row(&columns, &values).is_ok());
        assert!(budget.row(&columns, &values).is_ok());
        assert!(budget.row(&columns, &values).is_err());
        // Even an empty native B-tree may retain its root. The existing native
        // retained-size bound, not Python dict size alone, is authoritative.
        let value = Value::Map(Arc::new(BTreeMap::new()));
        assert!(CopyBudget::new(256).value(&value).is_err());
        assert!(CopyBudget::new(4096).value(&value).is_ok());
        let schema = grafeo_common::LogicalType::Struct(vec![(
            "field".repeat(1024),
            grafeo_common::LogicalType::Int64,
        )]);
        assert!(CopyBudget::new(4096).logical_type(&schema, 0).is_err());
    }
}
