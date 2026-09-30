//! Converts between JavaScript and Grafeo value types.
//!
//! | JavaScript type  | Grafeo type   | Notes                          |
//! | ---------------- | ------------- | ------------------------------ |
//! | `null/undefined` | `Null`        |                                |
//! | `boolean`        | `Bool`        |                                |
//! | `number`         | `Int64/Float64` | Integer if no fractional part |
//! | `string`         | `String`      |                                |
//! | `Array`          | `List`        | Elements converted recursively |
//! | `Object`         | `Map`         | Keys must be strings           |
//! | `Buffer`         | `Bytes`       |                                |
//! | `Date`           | `Timestamp`   | Millisecond precision          |
//! | `BigInt`         | `Int64`       |                                |
//! | `Float32Array`   | `Vector`      |                                |

#[cfg(any(
    test,
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use std::collections::BTreeMap;
use std::ffi::CString;
#[cfg(any(
    test,
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use std::sync::Arc;

use napi::bindgen_prelude::*;
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use napi::{JsDate, JsString, ValueType};
use napi::{JsValue, sys};

use grafeo_common::types::Value;
#[cfg(any(
    test,
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use grafeo_common::types::{PropertyKey, Timestamp};

/// Converts a JavaScript value to a Grafeo Value.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
pub fn js_to_value(env: &Env, val: Unknown<'_>) -> Result<Value> {
    #![allow(clippy::trivially_copy_pass_by_ref)] // Env refs are conventional in napi
    let value_type = val.get_type()?;
    match value_type {
        ValueType::Null | ValueType::Undefined => Ok(Value::Null),
        ValueType::Boolean => {
            let b = val.coerce_to_bool()?;
            Ok(Value::Bool(b))
        }
        ValueType::Number => {
            let n: f64 = val.coerce_to_number()?.get_double()?;
            // If the number is an integer within safe range, store as Int64
            if n.fract() == 0.0 && n.abs() < (1i64 << 53) as f64 {
                // reason: Value is a whole number in (-2^53, 2^53), fits in i64
                #[allow(clippy::cast_possible_truncation)]
                Ok(Value::Int64(n as i64))
            } else {
                Ok(Value::Float64(n))
            }
        }
        ValueType::String => {
            let s = val.coerce_to_string()?.into_utf8()?.into_owned()?;
            Ok(Value::String(s.into()))
        }
        ValueType::BigInt => {
            // SAFETY: type was checked as BigInt by the match arm, so cast is valid
            let bigint: BigInt = unsafe { val.cast()? };
            if bigint.words.len() > 1 {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    "BigInt value too large for i64",
                ));
            }
            let word = if bigint.words.is_empty() {
                0u64
            } else {
                bigint.words[0]
            };
            let signed = if bigint.sign_bit {
                if word == i64::MIN.unsigned_abs() {
                    i64::MIN
                } else if let Ok(v) = i64::try_from(word) {
                    -v
                } else {
                    return Err(napi::Error::new(
                        napi::Status::InvalidArg,
                        "BigInt value too large for i64",
                    ));
                }
            } else {
                i64::try_from(word).map_err(|_| {
                    napi::Error::new(napi::Status::InvalidArg, "BigInt value too large for i64")
                })?
            };
            Ok(Value::Int64(signed))
        }
        ValueType::Object => {
            // SAFETY: type was checked as Object by the match arm, so cast is valid
            let obj: Object<'_> = unsafe { val.cast()? };
            js_object_to_value(env, &obj)
        }
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Unsupported JavaScript type: {:?}", value_type),
        )),
    }
}

/// Converts a JavaScript object (Array, Buffer, Date, or plain object) to a Grafeo Value.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn js_object_to_value(env: &Env, obj: &Object<'_>) -> Result<Value> {
    if obj.is_array()? {
        let len = obj.get_array_length()?;
        let mut items = Vec::with_capacity(len as usize);
        for i in 0..len {
            let elem: Unknown<'_> = obj.get_element(i)?;
            items.push(js_to_value(env, elem)?);
        }
        return Ok(Value::List(items.into()));
    }

    if obj.is_buffer()? {
        // SAFETY: env and obj are valid napi values within this callback scope
        let unknown = unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()) };
        // SAFETY: obj.is_buffer() returned true, so casting to Buffer is valid
        let buf: Buffer = unsafe { unknown.cast()? };
        return Ok(Value::Bytes(buf.to_vec().into()));
    }

    if obj.is_date()? {
        // SAFETY: obj.is_date() returned true, and env/obj are valid in this scope
        let date: JsDate = unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
        let ms = date.value_of()?;
        // reason: JS Date.valueOf() returns ms since epoch; *1000 gives microseconds, fits in i64
        #[allow(clippy::cast_possible_truncation)]
        let micros = (ms * 1000.0) as i64;
        return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
    }

    // Check for TypedArray (Float32Array for vectors)
    if obj.is_typedarray()? {
        // SAFETY: obj.is_typedarray() returned true, and env/obj are valid in this scope
        let ta: TypedArray<'_> =
            unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
        if ta.typed_array_type == TypedArrayType::Float32 {
            // SAFETY: typed array type was verified as Float32, and env/obj are valid
            let f32arr: Float32Array =
                unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
            return Ok(Value::Vector(f32arr.to_vec().into()));
        }
    }

    // Plain object -> Map
    let keys = obj.get_property_names()?;
    let len = keys.get_array_length()?;
    let mut map = BTreeMap::new();
    for i in 0..len {
        let key: JsString = keys.get_element(i)?;
        let key_str = key.into_utf8()?.into_owned()?;
        let value: Unknown<'_> = obj.get_named_property(&key_str)?;
        map.insert(PropertyKey::new(key_str), js_to_value(env, value)?);
    }
    Ok(Value::Map(Arc::new(map)))
}

/// Helper to check napi_status and convert to Result.
pub(crate) fn check_napi(status: sys::napi_status) -> Result<()> {
    if status == sys::Status::napi_ok {
        Ok(())
    } else {
        Err(napi::Error::new(
            napi::Status::GenericFailure,
            format!("napi call failed with status: {status:?}"),
        ))
    }
}

/// The complete copied row/result must be admitted before entering this converter.
pub(crate) fn value_to_napi_admitted(env: sys::napi_env, value: &Value) -> Result<sys::napi_value> {
    match value {
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Null => unsafe { <Null as ToNapiValue>::to_napi_value(env, Null) },
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Bool(b) => unsafe { <bool as ToNapiValue>::to_napi_value(env, *b) },
        Value::Int64(i) => {
            // Use number for safe integer range, BigInt for larger values
            if *i > -(1i64 << 53) && *i < (1i64 << 53) {
                // SAFETY: env is a valid napi_env passed by the caller
                unsafe { <i64 as ToNapiValue>::to_napi_value(env, *i) }
            } else {
                // SAFETY: env is a valid napi_env passed by the caller
                unsafe {
                    <BigInt as ToNapiValue>::to_napi_value(
                        env,
                        BigInt {
                            sign_bit: *i < 0,
                            words: vec![i.unsigned_abs()],
                        },
                    )
                }
            }
        }
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Float64(f) => unsafe { <f64 as ToNapiValue>::to_napi_value(env, *f) },
        // SAFETY: env is a valid napi_env passed by the caller
        Value::String(s) => unsafe { <&str as ToNapiValue>::to_napi_value(env, s.as_ref()) },
        Value::List(items) => {
            let mut arr = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, items.len(), &raw mut arr)
            })?;
            for (i, item) in items.iter().enumerate() {
                let val = value_to_napi_admitted(env, item)?;
                // SAFETY: env, arr, and val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements, so index fits u32
                #[allow(clippy::cast_possible_truncation)]
                check_napi(unsafe { sys::napi_set_element(env, arr, i as u32, val) })?;
            }
            Ok(arr)
        }
        Value::Map(map) => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            for (key, val) in map.as_ref() {
                let key_cstr = CString::new(key.as_str())
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
                let napi_val = value_to_napi_admitted(env, val)?;
                // SAFETY: env, obj, key_cstr, and napi_val are all valid
                check_napi(unsafe {
                    sys::napi_set_named_property(env, obj, key_cstr.as_ptr(), napi_val)
                })?;
            }
            Ok(obj)
        }
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Bytes(bytes) => unsafe {
            <Buffer as ToNapiValue>::to_napi_value(env, Buffer::from(bytes.to_vec()))
        },
        Value::Timestamp(ts) => {
            let ms = ts.as_micros() as f64 / 1000.0;
            let env_wrapper = Env::from_raw(env);
            Ok(env_wrapper.create_date(ms)?.raw())
        }
        Value::Date(d) => {
            let s = d.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        Value::Time(t) => {
            let s = t.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        Value::Duration(d) => {
            let s = d.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        Value::ZonedDatetime(zdt) => {
            let s = zdt.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Vector(v) => unsafe {
            <Float32Array as ToNapiValue>::to_napi_value(env, Float32Array::new(v.to_vec()))
        },
        Value::Path { nodes, edges } => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;

            // Create nodes array
            let mut nodes_arr = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, nodes.len(), &raw mut nodes_arr)
            })?;
            for (i, node) in nodes.iter().enumerate() {
                let val = value_to_napi_admitted(env, node)?;
                // SAFETY: env, nodes_arr, and val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements
                #[allow(clippy::cast_possible_truncation)]
                check_napi(unsafe { sys::napi_set_element(env, nodes_arr, i as u32, val) })?;
            }

            // Create edges array
            let mut edges_arr = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, edges.len(), &raw mut edges_arr)
            })?;
            for (i, edge) in edges.iter().enumerate() {
                let val = value_to_napi_admitted(env, edge)?;
                // SAFETY: env, edges_arr, and val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements
                #[allow(clippy::cast_possible_truncation)]
                check_napi(unsafe { sys::napi_set_element(env, edges_arr, i as u32, val) })?;
            }

            let nodes_key = c"nodes";
            let edges_key = c"edges";
            // SAFETY: env, obj, and the key/value pointers are all valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, nodes_key.as_ptr(), nodes_arr)
            })?;
            // SAFETY: env, obj, and the key/value pointers are all valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, edges_key.as_ptr(), edges_arr)
            })?;

            Ok(obj)
        }
        Value::GCounter(counts) => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            let mut replicas = std::ptr::null_mut();
            check_napi(unsafe { sys::napi_create_object(env, &raw mut replicas) })?;
            // At most usize::MAX replicas each contribute u64::MAX; their
            // exact sum fits u128 before the single JavaScript Number conversion.
            let mut total: u128 = 0;
            for (replica, count) in counts.iter() {
                total += u128::from(*count);
                let mut val = std::ptr::null_mut();
                let count_f64 = *count as f64;
                // SAFETY: env is valid; napi_create_double writes to our out-pointer
                check_napi(unsafe { sys::napi_create_double(env, count_f64, &raw mut val) })?;
                let key = CString::new(replica.as_str())
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
                // SAFETY: env, replicas, key, and val are valid
                check_napi(unsafe {
                    sys::napi_set_named_property(env, replicas, key.as_ptr(), val)
                })?;
            }
            let gcounter_key = c"$gcounter";
            // SAFETY: env, obj, and replicas are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, gcounter_key.as_ptr(), replicas)
            })?;
            let mut total_val = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_double writes to our out-pointer
            check_napi(unsafe { sys::napi_create_double(env, total as f64, &raw mut total_val) })?;
            let value_key = c"$value";
            // SAFETY: env, obj, and total_val are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, value_key.as_ptr(), total_val)
            })?;
            Ok(obj)
        }
        Value::OnCounter { pos, neg } => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
            let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
            let net = if pos_sum >= neg_sum {
                (pos_sum - neg_sum) as f64
            } else {
                -((neg_sum - pos_sum) as f64)
            };
            let pncounter_key = c"$pncounter";
            let mut true_val = std::ptr::null_mut();
            // SAFETY: env is valid; napi_get_boolean writes to our out-pointer
            check_napi(unsafe { sys::napi_get_boolean(env, true, &raw mut true_val) })?;
            // SAFETY: env, obj, and true_val are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, pncounter_key.as_ptr(), true_val)
            })?;
            let mut net_val = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_double writes to our out-pointer
            check_napi(unsafe { sys::napi_create_double(env, net, &raw mut net_val) })?;
            let value_key = c"$value";
            // SAFETY: env, obj, and net_val are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, value_key.as_ptr(), net_val)
            })?;
            Ok(obj)
        }
        _ => {
            let s = value.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
    }
}

/// The native default bounds standalone JavaScript value conversions too.
pub(crate) fn default_conversion_limit() -> usize {
    grafeo_engine::query::ResultLimits::default().max_bytes
}

type CopyResult<T> = grafeo_common::utils::error::Result<T>;

pub(crate) fn copy_limit_error() -> grafeo_common::utils::error::Error {
    use grafeo_common::utils::error::{Error, StorageError};
    Error::Storage(StorageError::Full).with_context("Node result conversion exceeds max_bytes")
}

/// Allocation-free admission for native Values, copied serde_json storage and
/// V8 values. The bound covers both direct JS and existing tagged JSON routes.
/// Strings charge UTF-8, CString temporaries and two-byte V8 storage; arrays
/// charge JSON slots, napi handles and V8 backing including resize overlap.
/// Native retained-size accounting separately covers B-tree and hash capacity.
/// External libraries and V8 heap/GC policy are outside these binding-owned
/// copies; whole-process allocator qualification remains a separate gate.
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
    pub(crate) fn string(&mut self, value: &str) -> CopyResult<()> {
        self.charge(128)?;
        self.repeated(value.len(), 8)
    }
    pub(crate) fn list(&mut self, length: usize) -> CopyResult<()> {
        u32::try_from(length).map_err(|_| copy_limit_error())?;
        self.charge(128)?;
        self.repeated(length, 64)
    }
    pub(crate) fn dict(&mut self, length: usize) -> CopyResult<()> {
        u32::try_from(length).map_err(|_| copy_limit_error())?;
        // serde_json's B-tree root (11 slots), V8 property storage, and
        // property-name/entry temporaries while both representations coexist.
        self.charge(1024)?;
        self.repeated(length, 384)
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
    pub(crate) fn value(&mut self, value: &Value) -> CopyResult<()> {
        let before = self.remaining;
        self.nested_value(value, 0)?;
        // Tagged json! wrappers can copy their already converted descendants.
        // Admit both full representations without recursively revisiting them.
        self.charge(before - self.remaining)?;
        self.charge(value.retained_size_bytes().ok_or_else(copy_limit_error)?)
    }
    fn nested_value(&mut self, value: &Value, depth: usize) -> CopyResult<()> {
        // Bound the conversion call stack as well as copied heap memory. This is
        // a resource failure, before recursion or JavaScript allocation can overflow.
        if depth >= 256 {
            return Err(copy_limit_error());
        }
        self.charge(128)?;
        match value {
            Value::String(text) => self.string(text.as_str()),
            Value::Bytes(bytes) => {
                // Tagged JSON uses one Number value per byte; direct JS uses
                // Buffer. Admit the larger JSON array together with V8 elements.
                self.list(bytes.len())?;
                self.repeated(bytes.len(), 128)
            }
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
                self.dict(1)?;
                self.string("$path")?;
                self.dict(2)?;
                self.string("nodes")?;
                self.string("edges")?;
                for values in [nodes, edges] {
                    self.list(values.len())?;
                    for value in values.iter() {
                        // The existing tagged codec serializes temporary JSON
                        // arrays into the wrapper. Their copies coexist here.
                        self.nested_value(value, depth + 1)?;
                    }
                }
                Ok(())
            }
            Value::GCounter(values) => {
                self.dict(2)?;
                self.charge(4096)?;
                self.dict(values.len())?;
                for key in values.keys() {
                    self.string(key.as_str())?;
                    self.charge(128)?;
                }
                Ok(())
            }
            Value::OnCounter { pos, neg } => {
                self.dict(2)?;
                self.charge(4096)?;
                for values in [pos, neg] {
                    self.dict(values.len())?;
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

/// Preserve the shared JSON tag contract after admitting the complete copy.
#[cfg(any(test, feature = "gql"))]
pub(crate) fn bounded_row_to_json(
    columns: &[String],
    values: &[Value],
    max_bytes: usize,
) -> CopyResult<serde_json::Value> {
    let mut budget = CopyBudget::new(max_bytes);
    budget.row(columns, values)?;
    let mut object = serde_json::Map::new();
    for (column, value) in columns.iter().zip(values) {
        object.insert(
            column.clone(),
            grafeo_bindings_common::json::value_to_json(value),
        );
    }
    Ok(serde_json::Value::Object(object))
}

pub(crate) fn bounded_columns(columns: &[String], max_bytes: usize) -> CopyResult<Vec<String>> {
    CopyBudget::new(max_bytes).columns(columns)?;
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(columns.len())
        .map_err(|_| copy_limit_error())?;
    for column in columns {
        let mut name = String::new();
        name.try_reserve_exact(column.len())
            .map_err(|_| copy_limit_error())?;
        name.push_str(column);
        copied.push(name);
    }
    Ok(copied)
}

#[cfg(test)]
mod copy_admission_tests {
    use super::*;

    #[test]
    fn tagged_json_row_keeps_recursive_types_and_admits_byte_expansion() {
        let value = Value::Map(Arc::new(BTreeMap::from([
            (
                PropertyKey::new("bytes"),
                Value::Bytes(vec![255; 128].into()),
            ),
            (PropertyKey::new("text"), Value::from("雪😀")),
            (
                PropertyKey::new("timestamp"),
                Value::Timestamp(Timestamp::from_micros(42)),
            ),
            (
                PropertyKey::new("nested"),
                Value::List(vec![Value::Null, Value::List(Vec::new().into())].into()),
            ),
        ])));
        let columns = vec!["value".to_owned()];
        let values = [value];
        let error = bounded_row_to_json(&columns, &values, 8192).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        let copied = bounded_row_to_json(&columns, &values, 131072).unwrap();
        assert_eq!(copied["value"]["bytes"], serde_json::json!(vec![255; 128]));
        assert_eq!(copied["value"]["text"], "雪😀");
        assert_eq!(
            copied["value"]["timestamp"],
            serde_json::json!({"$timestamp_us": 42})
        );
        assert_eq!(copied["value"]["nested"], serde_json::json!([null, []]));
    }

    #[test]
    fn repeated_keys_and_json_path_wrapper_are_included_before_copy() {
        let columns = vec!["column".repeat(128)];
        let values = [Value::Int64(1)];
        let mut budget = CopyBudget::new(16384);
        assert!(budget.row(&columns, &values).is_ok());
        assert!(budget.row(&columns, &values).is_ok());
        assert!(budget.row(&columns, &values).is_err());
        let path = Value::Path {
            nodes: vec![Value::from("first"), Value::from("second")].into(),
            edges: vec![Value::from("edge")].into(),
        };
        let copied = bounded_row_to_json(&["path".to_owned()], &[path], 65536).unwrap();
        assert_eq!(
            copied,
            serde_json::json!({"path": {"$path": {
                "nodes": ["first", "second"], "edges": ["edge"],
            }}})
        );
    }
}
