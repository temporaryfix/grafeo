//! Single-use native controls and synchronous JavaScript option admission.

use std::cell::RefCell;

use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
use grafeo_engine::query::{ExecutionOptions, ResultLimits};
use js_sys::Reflect;
use wasm_bindgen::prelude::*;

/// One native query owner. Cancellation remains valid after the owner is used.
#[wasm_bindgen]
pub struct QueryControl {
    owner: RefCell<Option<ExecutionOptions>>,
    cancellation: Box<dyn Fn()>,
}

#[wasm_bindgen]
impl QueryControl {
    /// Constructs a single-use owner. Deadlines require a qualified monotonic
    /// clock and currently return a structured unsupported error on this binding.
    ///
    /// # Errors
    /// Returns invalid-input for malformed timeout values, or unsupported for
    /// an explicit deadline. Omitting the timeout creates a cancellable owner.
    #[wasm_bindgen(constructor)]
    pub fn new(timeout_ms: Option<f64>) -> Result<QueryControl, JsValue> {
        if let Some(timeout) = timeout_ms {
            safe_integer(timeout, "timeoutMs")?;
            return Err(native_error(&Error::Query(QueryError::new(
                QueryErrorKind::Unsupported,
                "WASM query deadlines require a qualified monotonic clock",
            ))));
        }
        let owner = ExecutionOptions::default();
        let cancellation = owner.control.cancellation_handle();
        Ok(Self {
            owner: RefCell::new(Some(owner)),
            cancellation: Box::new(move || cancellation.cancel()),
        })
    }

    /// Requests cancellation of this owner's execution.
    pub fn cancel(&self) {
        (self.cancellation)();
    }

    /// Whether an invocation has reserved this control's single use.
    #[wasm_bindgen(getter)]
    pub fn consumed(&self) -> bool {
        self.owner.borrow().is_none()
    }
}

impl QueryControl {
    fn take(&self) -> Result<ExecutionOptions, JsValue> {
        self.owner
            .try_borrow_mut()
            .map_err(|_| invalid("QueryControl is already being consumed"))?
            .take()
            .ok_or_else(|| invalid("QueryControl has already been consumed"))
    }
}

/// Native error identity survives the JavaScript exception boundary.
pub(crate) fn native_error(error: &Error) -> JsValue {
    let exception = js_sys::Error::new(&error.to_string());
    let _ = Reflect::set(
        &exception,
        &JsValue::from_str("code"),
        &JsValue::from_str(error.error_code().as_str()),
    );
    exception.into()
}

pub(crate) fn invalid(message: &str) -> JsValue {
    native_error(&Error::InvalidValue(message.into()))
}

fn safe_integer(value: f64, name: &str) -> Result<u64, JsValue> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 9_007_199_254_740_991.0
    {
        return Err(invalid(&format!(
            "{name} must be a nonnegative JavaScript safe integer"
        )));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value as u64)
}

fn read_limit(options: &JsValue, name: &str) -> Result<Option<usize>, JsValue> {
    let value = Reflect::get(options, &JsValue::from_str(name))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let number = value
        .as_f64()
        .ok_or_else(|| invalid(&format!("{name} must be a number")))?;
    let value = safe_integer(number, name)?;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| invalid(&format!("{name} exceeds this platform's capacity")))
}

pub(crate) struct PreparedExecution {
    pub native: ExecutionOptions,
    #[cfg(all(
        feature = "gql",
        any(
            feature = "edge",
            feature = "lpg",
            feature = "native",
            feature = "compact-store"
        )
    ))]
    pub max_rows: Option<usize>,
    #[cfg(all(
        feature = "gql",
        any(
            feature = "edge",
            feature = "lpg",
            feature = "native",
            feature = "compact-store"
        )
    ))]
    pub max_bytes: usize,
}

/// All option getters and validation finish before control ownership moves.
pub(crate) fn parse_options(
    options: &JsValue,
    control: Option<&QueryControl>,
    streaming: bool,
) -> Result<PreparedExecution, JsValue> {
    let defaults = ResultLimits::default();
    let (max_rows, max_bytes, language) = if options.is_null() || options.is_undefined() {
        (None, defaults.max_bytes, None)
    } else {
        if !options.is_object() || js_sys::Array::is_array(options) {
            return Err(invalid("Execution options must be an object"));
        }
        let max_rows = read_limit(options, "maxRows")?;
        let max_bytes = read_limit(options, "maxBytes")?.unwrap_or(defaults.max_bytes);
        let language = Reflect::get(options, &JsValue::from_str("language"))?;
        let language = if language.is_null() || language.is_undefined() {
            None
        } else {
            Some(
                language
                    .as_string()
                    .ok_or_else(|| invalid("language must be a string"))?,
            )
        };
        (max_rows, max_bytes, language)
    };
    // The same effective ceiling applies before commit and during conversion.
    // It bounds supported JS string/typed-array capacities on wasm32 and hosts.
    let max_bytes = max_bytes.min(0x7fff_ffff);
    let mut native = match control {
        Some(control) => control.take()?,
        None => ExecutionOptions::default(),
    };
    native.language = language;
    native.result_limits = Some(ResultLimits {
        max_rows: max_rows.unwrap_or(if streaming {
            usize::MAX
        } else {
            defaults.max_rows
        }),
        max_bytes,
    });
    native.result_admission = if streaming { None } else { Some(admit_result) };
    Ok(PreparedExecution {
        native,
        #[cfg(all(
            feature = "gql",
            any(
                feature = "edge",
                feature = "lpg",
                feature = "native",
                feature = "compact-store"
            )
        ))]
        max_rows,
        #[cfg(all(
            feature = "gql",
            any(
                feature = "edge",
                feature = "lpg",
                feature = "native",
                feature = "compact-store"
            )
        ))]
        max_bytes,
    })
}

fn admit_result(
    result: &grafeo_engine::database::QueryResult,
    limits: ResultLimits,
) -> grafeo_common::utils::error::Result<()> {
    crate::types::preflight_result(result, limits.max_bytes)
}

#[wasm_bindgen(typescript_custom_section)]
const EXECUTION_OPTIONS_TYPESCRIPT: &str = r#"
export interface ExecutionOptions {
  maxRows?: number;
  maxBytes?: number;
  language?: string;
}
"#;
