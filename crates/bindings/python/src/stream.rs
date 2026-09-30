//! Lazy Python rows with native terminal ownership and bounded conversion.

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use pyo3::prelude::*;

use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{OwnedResultStream, OwnedRowIterator};

use crate::error::PyGrafeoError;
use crate::types::{columns_to_py_bounded, row_to_py_bounded};

/// Iterator over lazy query rows, with explicit and context-managed cleanup.
///
/// Native pulls release the GIL. Each Python row is admitted against the
/// conversion byte cap before constructing its dictionary and nested values.
/// An explicit max_rows caps emitted rows; the default permits unlimited rows.
#[pyclass(name = "ResultStream")]
pub struct PyResultStream {
    iter: Mutex<OwnedRowIterator>,
    // Declared after the iterator so its keepalive drops after native cleanup.
    _database: Arc<RwLock<GrafeoDB>>,
    exhausted: bool,
    max_conversion_bytes: usize,
    max_rows: Option<usize>,
    emitted_rows: usize,
}

impl PyResultStream {
    pub(crate) fn new(
        database: Arc<RwLock<GrafeoDB>>,
        stream: OwnedResultStream,
        max_conversion_bytes: usize,
        max_rows: Option<usize>,
    ) -> Self {
        Self {
            _database: database,
            iter: Mutex::new(stream.into_row_iter()),
            exhausted: false,
            max_conversion_bytes,
            max_rows,
            emitted_rows: 0,
        }
    }
}

#[pymethods]
impl PyResultStream {
    /// Column names in the order they appear in each row dictionary.
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        columns_to_py_bounded(py, self.iter.lock().columns(), self.max_conversion_bytes)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if slf.exhausted {
            return Ok(None);
        }
        let limit = slf.max_conversion_bytes;
        let row_limit = slf.max_rows;
        let emitted_rows = slf.emitted_rows;
        let iter = slf.iter.get_mut();
        let row = py.detach(|| iter.next());
        match row {
            Some(Ok(values)) => {
                // Pull first: a stream ending exactly at the limit succeeds.
                // Reject a real excess row before constructing Python objects.
                let next_count = emitted_rows
                    .checked_add(1)
                    .filter(|count| row_limit.is_none_or(|limit| *count <= limit));
                let converted = match next_count {
                    Some(_) => row_to_py_bounded(py, iter.columns(), &values, limit),
                    None => Err(PyGrafeoError::from(
                        grafeo_common::utils::error::Error::Storage(
                            grafeo_common::utils::error::StorageError::Full,
                        )
                        .with_context("Python result stream exceeds max_rows"),
                    )
                    .into()),
                };
                match converted {
                    Ok(row) => {
                        if let Some(count) = next_count {
                            slf.emitted_rows = count;
                        }
                        Ok(Some(row))
                    }
                    Err(primary) => {
                        // Conversion is part of producing this row. A failed
                        // conversion closes the source, retaining its native
                        // cleanup resolution for an explicit subsequent close.
                        let cleanup = py.detach(|| iter.close());
                        slf.exhausted = true;
                        if let Err(cleanup) = cleanup {
                            let cleanup: PyErr = PyGrafeoError::from(cleanup).into();
                            let _ = primary
                                .value(py)
                                .setattr("cleanup_error", cleanup.value(py));
                        }
                        Err(primary)
                    }
                }
            }
            Some(Err(error)) => {
                slf.exhausted = true;
                Err(PyGrafeoError::from(error).into())
            }
            None => {
                slf.exhausted = true;
                Ok(None)
            }
        }
    }

    /// Stops iteration and releases native query resources immediately.
    ///
    /// Repeated calls retain the native close outcome, including cleanup errors.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        self.exhausted = true;
        let iter = self.iter.get_mut();
        py.detach(|| iter.close())
            .map_err(|error| PyGrafeoError::from(error).into())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if let Err(cleanup) = self.close(py) {
            if let Some(primary) = exc_value {
                // Preserve the exception from the with-body while exposing
                // cleanup failure; explicit close will report it again.
                let _ = primary.setattr("cleanup_error", cleanup.value(py));
            } else {
                return Err(cleanup);
            }
        }
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!("ResultStream(exhausted={})", self.exhausted)
    }
}
