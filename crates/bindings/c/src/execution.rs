//! Single-use native controls and pre-admitted C JSON output.

#[cfg(any(
    test,
    all(
        feature = "gql",
        any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native"
        )
    )
))]
use std::ffi::CString;
use std::ffi::{CStr, c_char};
use std::time::Duration;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_core::execution::{QueryCancellationHandle, QueryExecutionControl};
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::{ExecutionOptions, ResultLimits};
use parking_lot::Mutex;

use crate::error::{GrafeoStatus, set_error};

/// One execution owner. Free exactly once, after all uses of this allocation.
pub struct GrafeoQueryControl {
    owner: Mutex<Option<QueryExecutionControl>>,
    cancellation: QueryCancellationHandle,
}

/// Independent cancellation authority. Clone for independently freed owners.
pub struct GrafeoCancelHandle {
    cancellation: QueryCancellationHandle,
}

/// Explicit limits: zero is a real zero limit, not a default sentinel.
#[repr(C)]
pub struct GrafeoQueryOptions {
    pub control: *mut GrafeoQueryControl,
    pub max_rows: usize,
    pub max_bytes: usize,
    pub language: *const c_char,
}

fn invalid(message: &str) -> Error {
    Error::InvalidValue(message.into())
}

/// Creates a control; -1 means no deadline, otherwise milliseconds from now.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_query_control_create(timeout_ms: i64) -> *mut GrafeoQueryControl {
    let owner = if timeout_ms == -1 {
        Ok(QueryExecutionControl::new())
    } else {
        u64::try_from(timeout_ms)
            .map_err(|_| invalid("timeout_ms must be -1 or nonnegative"))
            .and_then(|milliseconds| {
                QueryExecutionControl::with_timeout(Duration::from_millis(milliseconds))
                    .map_err(|_| invalid("timeout_ms exceeds the native deadline range"))
            })
    };
    match owner {
        Ok(owner) => {
            let cancellation = owner.cancellation_handle();
            Box::into_raw(Box::new(GrafeoQueryControl {
                owner: Mutex::new(Some(owner)),
                cancellation,
            }))
        }
        Err(error) => {
            set_error(&error);
            std::ptr::null_mut()
        }
    }
}

/// Returns a separately owned cancellation handle, even after consumption.
///
/// # Safety
/// `control` must remain a live control allocation for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn grafeo_query_control_cancel_handle(
    control: *const GrafeoQueryControl,
) -> *mut GrafeoCancelHandle {
    let Some(control) = (unsafe { control.as_ref() }) else {
        set_error(&invalid("null query control"));
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(GrafeoCancelHandle {
        cancellation: control.cancellation.clone(),
    }))
}

/// Clones cancellation authority into a separately freed allocation.
///
/// # Safety
/// `handle` must remain live for this call; free each clone exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn grafeo_cancel_handle_clone(
    handle: *const GrafeoCancelHandle,
) -> *mut GrafeoCancelHandle {
    let Some(handle) = (unsafe { handle.as_ref() }) else {
        set_error(&invalid("null cancellation handle"));
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(GrafeoCancelHandle {
        cancellation: handle.cancellation.clone(),
    }))
}

/// Requests cancellation. Separate retained handles can cancel concurrently.
///
/// # Safety
/// `handle` must remain live throughout this call; concurrent free is invalid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn grafeo_cancel(handle: *const GrafeoCancelHandle) -> GrafeoStatus {
    let Some(handle) = (unsafe { handle.as_ref() }) else {
        return set_error(&invalid("null cancellation handle"));
    };
    handle.cancellation.cancel();
    GrafeoStatus::Ok
}

/// Frees a cancellation handle. Null is accepted.
///
/// # Safety
/// A non-null handle must be owned, live, and unused by concurrent calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn grafeo_cancel_handle_free(handle: *mut GrafeoCancelHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Frees a control. Already-created cancellation handles remain valid.
///
/// # Safety
/// A non-null control must be owned, live, and unused by concurrent calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn grafeo_query_control_free(control: *mut GrafeoQueryControl) {
    if !control.is_null() {
        drop(unsafe { Box::from_raw(control) });
    }
}

/// Validate borrowed options before consuming their control. No C pointer escapes.
///
/// # Safety
/// Non-null options/control/language pointers must remain valid during this call.
pub(crate) unsafe fn options_from_ptr(
    options: *const GrafeoQueryOptions,
) -> Result<ExecutionOptions> {
    let defaults = ResultLimits::default();
    let (control, limits, language) = match unsafe { options.as_ref() } {
        None => (std::ptr::null_mut(), defaults, None),
        Some(options) => {
            let language = if options.language.is_null() {
                None
            } else {
                let text = unsafe { CStr::from_ptr(options.language) }
                    .to_str()
                    .map_err(|_| invalid("query language is not valid UTF-8"))?;
                let mut owned = String::new();
                owned
                    .try_reserve_exact(text.len())
                    .map_err(|_| copy_limit_error())?;
                owned.push_str(text);
                Some(owned)
            };
            (
                options.control,
                ResultLimits {
                    max_rows: options.max_rows,
                    max_bytes: options.max_bytes,
                },
                language,
            )
        }
    };
    let owner = match unsafe { control.as_ref() } {
        None => QueryExecutionControl::new(),
        Some(control) => control
            .owner
            .lock()
            .take()
            .ok_or_else(|| invalid("query control has already been consumed"))?,
    };
    Ok(ExecutionOptions {
        control: owner,
        language,
        result_limits: Some(limits),
        result_admission: Some(admit_c_result),
    })
}

fn admit_c_result(result: &QueryResult, limits: ResultLimits) -> Result<()> {
    preflight_c_result(result, limits.max_bytes)
}

pub(crate) fn copy_limit_error() -> Error {
    Error::Storage(StorageError::Full)
        .with_context("C JSON output exceeds max_bytes or conversion capacity")
}

/// Conservative simultaneous native JSON tree, serialized/CString buffers and
/// entity-extraction envelope. Limits bound binding-owned copies, not C heaps.
pub(crate) struct CopyBudget {
    remaining: usize,
}
impl CopyBudget {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            remaining: max_bytes,
        }
    }
    pub(crate) fn charge(&mut self, bytes: usize) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(copy_limit_error)?;
        Ok(())
    }
    fn repeated(&mut self, count: usize, size: usize) -> Result<()> {
        self.charge(count.checked_mul(size).ok_or_else(copy_limit_error)?)
    }
    fn string(&mut self, text: &str) -> Result<()> {
        self.charge(128)?;
        // Worst-case six-byte JSON escaping, geometric writer growth, copy to
        // CString, and the retained native/JSON strings coexist.
        self.repeated(text.len(), 32)
    }
    fn list(&mut self, count: usize) -> Result<()> {
        self.charge(128)?;
        self.repeated(count, 128)
    }
    fn object(&mut self, count: usize) -> Result<()> {
        self.charge(1024)?;
        self.repeated(count, 384)
    }
    pub(crate) fn columns(&mut self, columns: &[String]) -> Result<()> {
        self.list(columns.len())?;
        for column in columns {
            self.string(column)?;
        }
        Ok(())
    }
    pub(crate) fn row(&mut self, columns: &[String], values: &[Value]) -> Result<()> {
        self.object(columns.len())?;
        for (column, value) in columns.iter().zip(values) {
            self.string(column)?;
            self.value(value)?;
        }
        Ok(())
    }
    pub(crate) fn value(&mut self, value: &Value) -> Result<()> {
        self.nested(value, 0)?;
        self.charge(value.retained_size_bytes().ok_or_else(copy_limit_error)?)
    }
    fn nested(&mut self, value: &Value, depth: usize) -> Result<()> {
        if depth >= 128 {
            return Err(copy_limit_error());
        }
        self.charge(128)?;
        match value {
            Value::Null | Value::Bool(_) | Value::Int64(_) | Value::Float64(_) => Ok(()),
            Value::String(text) => self.string(text.as_str()),
            Value::Bytes(bytes) => self.list(bytes.len()),
            Value::Vector(values) => self.list(values.len()),
            Value::List(values) => {
                self.list(values.len())?;
                for value in values.iter() {
                    self.nested(value, depth + 1)?;
                }
                Ok(())
            }
            Value::Map(values) => {
                self.object(values.len())?;
                for (key, value) in values.iter() {
                    self.string(key.as_str())?;
                    self.nested(value, depth + 1)?;
                }
                Ok(())
            }
            Value::Path { nodes, edges } => {
                self.charge(4096)?;
                for values in [nodes, edges] {
                    self.list(values.len())?;
                    for value in values.iter() {
                        self.nested(value, depth + 1)?;
                    }
                }
                Ok(())
            }
            Value::GCounter(values) => {
                self.charge(4096)?;
                self.object(values.len())?;
                for key in values.keys() {
                    self.string(key)?;
                    self.charge(128)?;
                }
                Ok(())
            }
            Value::OnCounter { pos, neg } => {
                self.charge(4096)?;
                for values in [pos, neg] {
                    self.object(values.len())?;
                    for key in values.keys() {
                        self.string(key)?;
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
                self.charge(512)
            }
            _ => Err(copy_limit_error()),
        }
    }
}

/// Fixed result/header/schema copies, independent of the number of rows.
pub(crate) fn c_result_base_bytes(columns: &[String]) -> Result<usize> {
    let mut budget = CopyBudget::new(usize::MAX);
    budget.charge(512)?;
    budget.columns(columns)?;
    budget.list(0)?;
    budget.list(0)?;
    budget.list(0)?;
    Ok(usize::MAX - budget.remaining)
}

/// One additive row reservation, including possible extracted entity copies.
pub(crate) fn c_result_row_bytes(columns: &[String], row: &[Value]) -> Result<usize> {
    let mut budget = CopyBudget::new(usize::MAX);
    budget.charge(128)?; // Outer result-array slot and retained row container.
    budget.row(columns, row)?;
    // Extraction scans every top-level value, including hidden columns.
    for value in row {
        if matches!(value, Value::Map(_)) {
            budget.charge(4096)?;
            budget.value(value)?;
            budget.value(value)?;
        }
    }
    Ok(usize::MAX - budget.remaining)
}

/// Precommit admission borrows dense columns; it never populates rows()' cache.
pub(crate) fn preflight_c_result(result: &QueryResult, max_bytes: usize) -> Result<()> {
    let mut budget = CopyBudget::new(max_bytes);
    budget.charge(c_result_base_bytes(&result.columns)?)?;
    if result.is_int64_columnar() {
        let mut row = CopyBudget::new(usize::MAX);
        row.charge(128)?;
        row.object(result.columns.len())?;
        for name in &result.columns {
            row.string(name)?;
            row.value(&Value::Int64(0))?;
        }
        budget.repeated(result.row_count(), usize::MAX - row.remaining)?;
    } else {
        for row in result.rows() {
            budget.charge(c_result_row_bytes(&result.columns, row)?)?;
        }
    }
    Ok(())
}

#[cfg(any(
    test,
    all(
        feature = "gql",
        any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native"
        )
    )
))]
pub(crate) fn bounded_row_json(
    columns: &[String],
    values: &[Value],
    max_bytes: usize,
) -> Result<CString> {
    let mut budget = CopyBudget::new(max_bytes);
    budget.charge(c_result_base_bytes(columns)?)?;
    budget.charge(c_result_row_bytes(columns, values)?)?;
    let row: serde_json::Map<String, serde_json::Value> = columns
        .iter()
        .zip(values)
        .map(|(column, value)| (column.clone(), crate::types::value_to_json(value)))
        .collect();
    let json = serde_json::to_vec(&row).map_err(|error| Error::Serialization(error.to_string()))?;
    CString::new(json).map_err(|error| Error::Serialization(error.to_string()))
}

#[cfg(any(
    test,
    all(
        feature = "gql",
        any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native"
        )
    )
))]
pub(crate) fn bounded_columns_json(columns: &[String], max_bytes: usize) -> Result<CString> {
    CopyBudget::new(max_bytes).columns(columns)?;
    let json =
        serde_json::to_vec(columns).map_err(|error| Error::Serialization(error.to_string()))?;
    CString::new(json).map_err(|error| Error::Serialization(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::execution::QueryCancellationError;

    #[test]
    fn c_control_is_consumed_once_and_cancel_handle_outlives_control() {
        unsafe {
            let control = grafeo_query_control_create(-1);
            assert!(!control.is_null());
            let handle = grafeo_query_control_cancel_handle(control);
            let clone = grafeo_cancel_handle_clone(handle);
            let options = GrafeoQueryOptions {
                control,
                max_rows: 17,
                max_bytes: 4096,
                language: c"gql".as_ptr(),
            };
            let execution = options_from_ptr(&raw const options).unwrap();
            assert_eq!(execution.language.as_deref(), Some("gql"));
            assert_eq!(execution.result_limits.unwrap().max_rows, 17);
            assert!(options_from_ptr(&raw const options).is_err());
            grafeo_query_control_free(control);
            grafeo_cancel_handle_free(handle);
            assert_eq!(grafeo_cancel(clone), GrafeoStatus::Ok);
            assert!(matches!(
                execution.control.check(),
                Err(QueryCancellationError::Cancelled)
            ));
            grafeo_cancel_handle_free(clone);
        }
    }

    #[test]
    fn c_control_precancel_deadline_and_invalid_options() {
        assert!(grafeo_query_control_create(-2).is_null());
        unsafe {
            for deadline in [-1, 0] {
                let control = grafeo_query_control_create(deadline);
                let handle = grafeo_query_control_cancel_handle(control);
                if deadline == -1 {
                    assert_eq!(grafeo_cancel(handle), GrafeoStatus::Ok);
                }
                let options = GrafeoQueryOptions {
                    control,
                    max_rows: 0,
                    max_bytes: 0,
                    language: std::ptr::null(),
                };
                let execution = options_from_ptr(&raw const options).unwrap();
                assert_eq!(execution.result_limits.unwrap().max_bytes, 0);
                match deadline {
                    -1 => assert!(matches!(
                        execution.control.check(),
                        Err(QueryCancellationError::Cancelled)
                    )),
                    _ => assert!(matches!(
                        execution.control.check(),
                        Err(QueryCancellationError::DeadlineExceeded { .. })
                    )),
                }
                grafeo_cancel_handle_free(handle);
                grafeo_query_control_free(control);
            }
            let control = grafeo_query_control_create(-1);
            let invalid_language = [255u8, 0];
            let mut options = GrafeoQueryOptions {
                control,
                max_rows: 1,
                max_bytes: 4096,
                language: invalid_language.as_ptr().cast(),
            };
            assert!(options_from_ptr(&raw const options).is_err());
            options.language = std::ptr::null();
            assert!(
                options_from_ptr(&raw const options).is_ok(),
                "validation failure must not consume control"
            );
            grafeo_query_control_free(control);
        }
    }

    #[test]
    fn c_json_admission_preserves_tags_and_rejects_nested_copy_before_conversion() {
        let columns = vec!["payload".to_owned()];
        let values = [Value::List(vec![Value::from("x".repeat(8192))].into())];
        let error = bounded_row_json(&columns, &values, 8192).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        let admitted = c_result_base_bytes(&columns)
            .unwrap()
            .checked_add(c_result_row_bytes(&columns, &values).unwrap())
            .unwrap();
        let encoded = bounded_row_json(&columns, &values, admitted).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(encoded.as_bytes()).unwrap();
        assert_eq!(decoded["payload"][0].as_str().unwrap().len(), 8192);
        assert!(bounded_row_json(&columns, &values, admitted - 1).is_err());
        let tagged = [Value::Path {
            nodes: vec![Value::Int64(1)].into(),
            edges: Vec::<Value>::new().into(),
        }];
        let encoded = bounded_row_json(&columns, &tagged, 65536).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(encoded.as_bytes()).unwrap();
        assert_eq!(decoded["payload"]["$path"]["nodes"][0], 1);
        assert_eq!(
            bounded_columns_json(&columns, 4096)
                .unwrap()
                .to_str()
                .unwrap(),
            "[\"payload\"]"
        );
        assert!(bounded_columns_json(&columns, 0).is_err());
    }
}
