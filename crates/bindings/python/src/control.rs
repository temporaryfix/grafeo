//! One-consumer query execution ownership with a shareable cancellation handle.

use std::time::Duration;

use grafeo_core::execution::{QueryCancellationHandle, QueryExecutionControl};
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Cancellation and optional deadline for one query invocation.
///
/// The deadline starts when this object is created. Passing the control to a
/// query consumes its execution owner; cancellation remains available from
/// other Python threads for the lifetime of the object.
#[pyclass(name = "QueryControl")]
pub struct PyQueryControl {
    owner: Mutex<Option<QueryExecutionControl>>,
    cancellation: QueryCancellationHandle,
}

impl PyQueryControl {
    pub(crate) fn take_control(&self) -> PyResult<QueryExecutionControl> {
        self.owner.lock().take().ok_or_else(|| {
            PyValueError::new_err("QueryControl has already been consumed by a query")
        })
    }
}

#[pymethods]
impl PyQueryControl {
    #[new]
    #[pyo3(signature = (timeout_ms=None))]
    fn new(timeout_ms: Option<u64>) -> PyResult<Self> {
        let owner = match timeout_ms {
            Some(milliseconds) => {
                QueryExecutionControl::with_timeout(Duration::from_millis(milliseconds))
                    .map_err(|error| PyValueError::new_err(error.to_string()))?
            }
            None => QueryExecutionControl::new(),
        };
        let cancellation = owner.cancellation_handle();
        Ok(Self {
            owner: Mutex::new(Some(owner)),
            cancellation,
        })
    }

    /// Requests cancellation, including before the control is consumed.
    fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Whether a query has taken the single execution owner.
    #[getter]
    fn consumed(&self) -> bool {
        self.owner.lock().is_none()
    }

    fn __repr__(&self) -> String {
        format!("QueryControl(consumed={})", self.consumed())
    }
}
