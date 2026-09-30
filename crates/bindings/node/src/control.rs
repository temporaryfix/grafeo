//! Synchronous admission of one native query owner from JavaScript options.

use std::time::Duration;

use grafeo_core::execution::{QueryCancellationHandle, QueryExecutionControl};
use grafeo_engine::query::{ExecutionOptions, ResultLimits};
use napi::JsValue;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::Mutex;

/// Cancellation and optional deadline for a single query invocation.
#[napi]
pub struct QueryControl {
    owner: Mutex<Option<QueryExecutionControl>>,
    cancellation: QueryCancellationHandle,
}

#[napi]
impl QueryControl {
    /// Creates a control whose optional deadline starts immediately.
    #[napi(constructor)]
    pub fn new(timeout_ms: Option<f64>) -> Result<Self> {
        let owner = match timeout_ms {
            Some(value) => {
                let milliseconds = safe_integer(value, "timeoutMs")?;
                QueryExecutionControl::with_timeout(Duration::from_millis(milliseconds))
                    .map_err(|error| invalid(error.to_string()))?
            }
            None => QueryExecutionControl::new(),
        };
        let cancellation = owner.cancellation_handle();
        Ok(Self {
            owner: Mutex::new(Some(owner)),
            cancellation,
        })
    }

    /// Requests cancellation before or during execution.
    #[napi]
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Whether a query has consumed this control's single execution owner.
    #[napi(getter)]
    pub fn consumed(&self) -> bool {
        self.owner.lock().is_none()
    }
}

impl QueryControl {
    pub(crate) fn take_control(&self) -> Result<QueryExecutionControl> {
        self.owner
            .lock()
            .take()
            .ok_or_else(|| invalid("QueryControl has already been consumed"))
    }
}

fn invalid(message: impl Into<String>) -> napi::Error {
    napi::Error::new(napi::Status::InvalidArg, message.into())
}

fn safe_integer(value: f64, name: &str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 9_007_199_254_740_991.0
    {
        return Err(invalid(format!(
            "{name} must be a nonnegative JavaScript safe integer"
        )));
    }
    // The checked range is integral, nonnegative and below both u64::MAX and
    // JavaScript's exact integer boundary.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value as u64)
}

fn read_limit(options: &Object<'_>, name: &str) -> Result<Option<usize>> {
    options
        .get::<Option<f64>>(name)?
        .flatten()
        .map(|value| {
            let value = safe_integer(value, name)?;
            usize::try_from(value)
                .map_err(|_| invalid(format!("{name} exceeds this platform's capacity")))
        })
        .transpose()
}

/// Only owned native data may leave synchronous JavaScript argument admission.
pub(crate) struct PreparedOptions {
    pub native: ExecutionOptions,
    #[cfg(feature = "gql")]
    pub max_rows: Option<usize>,
    pub max_bytes: usize,
}

pub(crate) fn prepare_execution_options(options: Option<Object<'_>>) -> Result<PreparedOptions> {
    let defaults = ResultLimits::default();
    // Validate every limit before consuming the control, including getters
    // that throw. No JavaScript property is read after take_control().
    let (max_rows, max_bytes, control) = match options {
        Some(options) => {
            let max_rows = read_limit(&options, "maxRows")?;
            let max_bytes = read_limit(&options, "maxBytes")?.unwrap_or(defaults.max_bytes);
            let control = options.get::<Option<Unknown<'_>>>("control")?.flatten();
            let control = control
                .map(|control| {
                    let raw = control.value();
                    // SAFETY: the value is live in this synchronous callback.
                    // Validate the actual class before napi's unchecked unwrap;
                    // the class reference is consumed here and never escapes.
                    unsafe {
                        <ClassInstance<'_, QueryControl> as ValidateNapiValue>::validate(
                            raw.env, raw.value,
                        )?;
                        ClassInstance::<QueryControl>::from_napi_value(raw.env, raw.value)
                    }
                })
                .transpose()?;
            (max_rows, max_bytes, control)
        }
        None => (None, defaults.max_bytes, None),
    };
    let owner = control.map_or_else(
        || Ok(QueryExecutionControl::new()),
        |control| control.take_control(),
    )?;
    Ok(PreparedOptions {
        native: ExecutionOptions {
            control: owner,
            language: None,
            result_limits: Some(ResultLimits {
                max_rows: max_rows.unwrap_or(defaults.max_rows),
                max_bytes,
            }),
            result_admission: Some(crate::query::admit_node_result),
        },
        #[cfg(feature = "gql")]
        max_rows,
        max_bytes,
    })
}
