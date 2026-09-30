//! Query results and builders for the Python API.

use std::collections::HashMap;
use std::fmt::Write as _;

use pyo3::prelude::*;

use grafeo_common::types::Value;
use grafeo_engine::database::{OwnedInt64Columns, OwnedRows};

use crate::graph::{PyEdge, PyNode};
use crate::types::{
    CopyBudget, PyValue, columns_to_py_bounded, copy_error, copy_limit_error,
    default_conversion_limit, display_bytes, new_dict, new_list, row_to_py_bounded,
};

/// Pure admission callback: executed by the engine before a statement commits.
pub(crate) fn admit_python_result(
    result: &grafeo_engine::database::QueryResult,
    limits: grafeo_engine::query::ResultLimits,
) -> grafeo_common::utils::error::Result<()> {
    preflight_query_result(result, limits.max_bytes)
}

pub(crate) fn preflight_query_result(
    result: &grafeo_engine::database::QueryResult,
    max_bytes: usize,
) -> grafeo_common::utils::error::Result<()> {
    let mut budget = CopyBudget::new(max_bytes);
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
    budget.dict(result.columns.len())?;
    for _ in &result.columns {
        budget.list(result.row_count())?;
    }
    budget.repeated_row_headers(&result.columns, result.row_count())?;
    if result.is_int64_columnar() {
        // rows() would lazily allocate the dense result's row cache. This pass
        // only borrows columns and must precede every copied allocation.
        let cells = result
            .row_count()
            .checked_mul(result.columns.len())
            .ok_or_else(copy_limit_error)?;
        budget.repeated(cells, 128)?;
    } else {
        for row in result.rows() {
            for value in row {
                budget.value(value)?;
            }
            // Entity extraction can additionally clone map/label containers and
            // retain one Python wrapper for each top-level entity-shaped map.
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
    budget: &mut CopyBudget,
    nodes: &[PyNode],
    edges: &[PyEdge],
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

/// Results from a GQL query - iterate rows or access nodes and edges directly.
///
/// Iterate with `for row in result:` where each row is a dict. Use
/// `result.column(0)` / `result.column("age")` for one column as a list
/// (no per-row dicts). `result.nodes()` and `result.edges()` return graph
/// elements. `result.scalar()` is the first column of the first row.
///
/// Query performance metrics are available via `execution_time_ms` and
/// `rows_scanned` properties when timing is enabled.
#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: OwnedRows,
    /// Dense Int64 columns from columnar execute (no per-row `Vec`s).
    pub(crate) int64_cols: Option<OwnedInt64Columns>,
    pub(crate) nodes: Vec<PyNode>,
    pub(crate) edges: Vec<PyEdge>,
    current_row: usize,
    conversion_limit: usize,
    /// Query execution time in milliseconds.
    pub(crate) execution_time_ms: Option<f64>,
    /// Number of rows scanned during execution.
    pub(crate) rows_scanned: Option<u64>,
}

#[pymethods]
impl PyQueryResult {
    /// Get column names.
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        columns_to_py_bounded(py, &self.columns, self.conversion_limit)
    }

    /// Get number of rows.
    fn __len__(&self) -> usize {
        if let Some(cols) = &self.int64_cols {
            return cols.first().map_or(0, Vec::len);
        }
        self.rows.len()
    }

    /// Get a row by index.
    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n = self.__len__();
        let idx = if idx < 0 {
            // reason: Python negative indexing
            #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
            let resolved = (n as isize + idx) as usize;
            resolved
        } else {
            // reason: non-negative isize fits usize
            #[allow(clippy::cast_sign_loss)]
            let resolved = idx as usize;
            resolved
        };
        if idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Row index out of range",
            ));
        }
        self.row_dict(idx, py)
    }

    /// Iterate over rows.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get next row.
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if slf.current_row >= slf.__len__() {
            return Ok(None);
        }
        let row = slf.row_dict(slf.current_row, py)?;
        slf.current_row += 1;
        Ok(Some(row))
    }

    /// One result column as a Python list (no per-row dicts).
    ///
    /// `key` is a 0-based index or a column name. `RETURN id(c)` of 128k
    /// dests is one `list[int]`, not 128k mappings.
    fn column(&self, key: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let idx = if let Ok(i) = key.extract::<isize>() {
            let n = isize::try_from(self.columns.len()).unwrap_or(isize::MAX);
            let resolved = if i < 0 { n.saturating_add(i) } else { i };
            usize::try_from(resolved)
                .ok()
                .filter(|&u| u < self.columns.len())
        } else if let Ok(name) = key.extract::<&str>() {
            self.columns.iter().position(|c| c == name)
        } else {
            None
        };
        let Some(idx) = idx else {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "column key out of range or unknown name",
            ));
        };
        self.check_copies()?;
        self.column_admitted(idx, py)
    }

    /// Get all nodes from the result.
    fn nodes(&self) -> PyResult<Vec<PyNode>> {
        self.check_copies()?;
        Ok(self.nodes.clone())
    }

    /// Get all edges from the result.
    fn edges(&self) -> PyResult<Vec<PyEdge>> {
        self.check_copies()?;
        Ok(self.edges.clone())
    }

    /// Convert to a list of dictionaries after admitting the entire copy.
    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_copies()?;
        let list = new_list(py)?;
        for i in 0..self.__len__() {
            list.append(self.row_dict(i, py)?)?;
        }
        Ok(list.unbind().into_any())
    }

    /// Get single value (first column of first row).
    fn scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.__len__() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("No rows in result"));
        }
        if self.columns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No columns in result",
            ));
        }
        if let Some(cols) = &self.int64_cols
            && let Some(&v) = cols.first().and_then(|c| c.first())
        {
            return PyValue::to_py_bounded(&Value::Int64(v), py, self.conversion_limit);
        }
        PyValue::to_py_bounded(&self.rows[0][0], py, self.conversion_limit)
    }

    /// Query execution time in milliseconds (if available).
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n")
    /// if result.execution_time_ms:
    ///     print(f"Query took {result.execution_time_ms:.2f}ms")
    /// ```
    #[getter]
    fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Number of rows scanned during query execution (if available).
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n")
    /// if result.rows_scanned:
    ///     print(f"Scanned {result.rows_scanned} rows")
    /// ```
    #[getter]
    fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    /// Convert to a pandas DataFrame.
    ///
    /// Requires pandas to be installed (`uv add pandas`). Each column in the
    /// query result becomes a DataFrame column, preserving types where possible.
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n.name, n.age")
    /// df = result.to_pandas()
    /// print(df.head())
    /// ```
    #[pyo3(signature = ())]
    fn to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_copies()?;
        let pd = py.import("pandas").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pandas is required for to_pandas(). Install it with: uv add pandas",
            )
        })?;

        let data = self.column_data(py)?;

        let df = pd.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    /// Convert to a polars DataFrame.
    ///
    /// Requires polars to be installed (`uv add polars`). Each column in the
    /// query result becomes a DataFrame column. Values are converted to native
    /// Python types first, then polars infers the best dtype.
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n.name, n.age")
    /// df = result.to_polars()
    /// print(df.head())
    /// ```
    #[pyo3(signature = ())]
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_copies()?;
        let pl = py.import("polars").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "polars is required for to_polars(). Install it with: uv add polars",
            )
        })?;

        let data = self.column_data(py)?;

        let df = pl.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    /// Convert to Arrow IPC bytes.
    ///
    /// Returns the query result as Arrow IPC stream format bytes. These can be
    /// read by any Arrow implementation:
    ///
    /// - `pyarrow.ipc.open_stream(buf).read_all()` for a PyArrow Table
    /// - `polars.read_ipc(buf)` for a Polars DataFrame
    ///
    /// Example:
    /// ```python
    /// ipc_bytes = result.to_arrow_ipc()
    /// import pyarrow as pa
    /// table = pa.ipc.open_stream(ipc_bytes).read_all()
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn to_arrow_ipc(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ipc_bytes = self.to_ipc_bytes()?;
        Ok(
            pyo3::types::PyBytes::new_with(py, ipc_bytes.len(), |buffer| {
                buffer.copy_from_slice(&ipc_bytes);
                Ok(())
            })?
            .unbind()
            .into_any(),
        )
    }

    /// Convert to a PyArrow Table.
    ///
    /// Requires pyarrow to be installed (`uv add pyarrow`). Returns an Arrow
    /// Table that can be used directly with DuckDB, Polars, pandas, or any
    /// other Arrow-compatible tool.
    ///
    /// Example:
    /// ```python
    /// table = result.to_arrow()
    /// # Convert to pandas: table.to_pandas()
    /// # Convert to polars: polars.from_arrow(table)
    /// # Use with DuckDB: duckdb.from_arrow(table)
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_copies()?;
        let pa = py.import("pyarrow").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pyarrow is required for to_arrow(). Install it with: uv add pyarrow",
            )
        })?;
        let ipc_mod = pa.getattr("ipc")?;

        let ipc_bytes = self.to_ipc_bytes()?;
        let py_bytes = pyo3::types::PyBytes::new_with(py, ipc_bytes.len(), |buffer| {
            buffer.copy_from_slice(&ipc_bytes);
            Ok(())
        })?;
        let reader = ipc_mod.call_method1("open_stream", (py_bytes,))?;
        let table = reader.call_method0("read_all")?;
        Ok(table.unbind())
    }

    /// Serialize CONSTRUCT-style results as N-Triples text.
    ///
    /// The result must have columns `["subject", "predicate", "object"]` (the
    /// standard shape returned by SPARQL CONSTRUCT queries). Each row is
    /// formatted as `<subject> <predicate> <object> .\n`.
    ///
    /// Returns a Python string. Raises `ValueError` if the columns do not
    /// match the expected triple pattern.
    ///
    /// Example:
    /// ```python
    /// result = db.execute_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
    /// print(result.to_ntriples())
    /// ```
    fn to_ntriples(&self) -> PyResult<String> {
        self.check_text_copies(false)?;
        self.validate_triple_columns()?;
        let (si, pi, oi) = self.triple_column_indices();
        let mut output = String::new();
        for row in &self.rows {
            let subj = Self::value_to_ntriples_term(&row[si]);
            let pred = Self::value_to_ntriples_term(&row[pi]);
            let obj = Self::value_to_ntriples_term(&row[oi]);
            let _ = writeln!(output, "{subj} {pred} {obj} .");
        }
        Ok(output)
    }

    /// Serialize CONSTRUCT-style results as Turtle text.
    ///
    /// Groups triples by subject and uses `;` to separate predicate-object
    /// pairs sharing the same subject. This is a convenience formatter, not a
    /// full Turtle serializer (no prefix declarations are emitted).
    ///
    /// Returns a Python string. Raises `ValueError` if the columns do not
    /// match the expected triple pattern.
    ///
    /// Example:
    /// ```python
    /// result = db.execute_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
    /// print(result.to_turtle())
    /// ```
    fn to_turtle(&self) -> PyResult<String> {
        self.check_text_copies(false)?;
        self.validate_triple_columns()?;
        let (si, pi, oi) = self.triple_column_indices();

        // Group by subject, preserving insertion order via Vec of (subject, predicates).
        let mut subjects: Vec<(String, Vec<(String, String)>)> = Vec::new();
        let mut subject_index: HashMap<String, usize> = HashMap::new();

        for row in &self.rows {
            let subj = Self::value_to_ntriples_term(&row[si]);
            let pred = Self::value_to_ntriples_term(&row[pi]);
            let obj = Self::value_to_ntriples_term(&row[oi]);

            if let Some(&idx) = subject_index.get(&subj) {
                subjects[idx].1.push((pred, obj));
            } else {
                let idx = subjects.len();
                subject_index.insert(subj.clone(), idx);
                subjects.push((subj, vec![(pred, obj)]));
            }
        }

        let mut output = String::new();
        for (subj, pairs) in &subjects {
            output.push_str(subj);
            for (i, (pred, obj)) in pairs.iter().enumerate() {
                if i == 0 {
                    let _ = write!(output, " {pred} {obj}");
                } else {
                    let _ = write!(output, " ;\n    {pred} {obj}");
                }
            }
            output.push_str(" .\n\n");
        }
        Ok(output)
    }

    fn __repr__(&self) -> PyResult<String> {
        self.check_text_copies(false)?;
        let time_str = self
            .execution_time_ms
            .map(|t| format!(", time={:.2}ms", t))
            .unwrap_or_default();
        Ok(format!(
            "QueryResult(columns={:?}, rows={}{})",
            self.columns,
            self.__len__(),
            time_str
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.check_text_copies(true)?;
        Ok(grafeo_common::fmt::format_result_table(
            &self.columns,
            &self.rows,
            self.execution_time_ms,
            None,
        ))
    }
}

impl PyQueryResult {
    /// Creates a new query result (used internally).
    pub fn new(
        columns: Vec<String>,
        rows: OwnedRows,
        nodes: Vec<PyNode>,
        edges: Vec<PyEdge>,
    ) -> Self {
        Self {
            columns,
            rows,
            int64_cols: None,
            nodes,
            edges,
            current_row: 0,
            conversion_limit: default_conversion_limit(),
            execution_time_ms: None,
            rows_scanned: None,
        }
    }

    /// Creates a new query result with execution metrics (used internally).
    pub fn with_metrics(
        columns: Vec<String>,
        rows: OwnedRows,
        nodes: Vec<PyNode>,
        edges: Vec<PyEdge>,
        execution_time_ms: Option<f64>,
        rows_scanned: Option<u64>,
    ) -> Self {
        Self {
            columns,
            rows,
            int64_cols: None,
            nodes,
            edges,
            current_row: 0,
            conversion_limit: default_conversion_limit(),
            execution_time_ms,
            rows_scanned,
        }
    }

    pub fn with_int64_cols(mut self, cols: Option<OwnedInt64Columns>) -> Self {
        self.int64_cols = cols;
        self
    }

    pub(crate) fn with_conversion_limit(mut self, max_bytes: usize) -> Self {
        self.conversion_limit = max_bytes;
        self
    }

    fn row_dict(&self, idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(cols) = &self.int64_cols {
            let mut budget = CopyBudget::new(self.conversion_limit);
            budget.dict(self.columns.len()).map_err(copy_error)?;
            for name in &self.columns {
                budget.string(name).map_err(copy_error)?;
                budget.charge(128).map_err(copy_error)?;
            }
            let dict = new_dict(py)?;
            for (name, col) in self.columns.iter().zip(cols.iter()) {
                let value = col.get(idx).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("Row index out of range")
                })?;
                dict.set_item(name, value)?;
            }
            return Ok(dict.unbind().into_any());
        }
        let row = self
            .rows
            .get(idx)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Row index out of range"))?;
        row_to_py_bounded(py, &self.columns, row, self.conversion_limit)
    }

    fn column_admitted(&self, idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = new_list(py)?;
        if let Some(cols) = &self.int64_cols {
            if let Some(column) = cols.get(idx) {
                for value in column {
                    list.append(value)?;
                }
            }
        } else {
            for row in &self.rows {
                let value = match row.get(idx) {
                    Some(value) => PyValue::to_py_admitted(value, py)?,
                    None => py.None(),
                };
                list.append(value)?;
            }
        }
        Ok(list.unbind().into_any())
    }

    fn column_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_copies()?;
        let dict = new_dict(py)?;
        for (idx, name) in self.columns.iter().enumerate() {
            dict.set_item(name, self.column_admitted(idx, py)?)?;
        }
        Ok(dict.unbind().into_any())
    }

    fn copy_budget(&self) -> grafeo_common::utils::error::Result<CopyBudget> {
        let mut budget = CopyBudget::new(self.conversion_limit);
        budget.columns(&self.columns)?;
        budget.list(self.__len__())?;
        budget.dict(self.columns.len())?;
        for _ in &self.columns {
            budget.list(self.__len__())?;
        }
        if let Some(columns) = &self.int64_cols {
            for idx in 0..self.__len__() {
                budget.dict(self.columns.len())?;
                for (name, column) in self.columns.iter().zip(columns.iter()) {
                    budget.string(name)?;
                    if column.get(idx).is_some() {
                        budget.charge(128)?;
                    }
                }
            }
        } else {
            for row in &self.rows {
                budget.row(&self.columns, row)?;
            }
        }
        admit_entity_copies(&mut budget, &self.nodes, &self.edges)?;
        Ok(budget)
    }

    fn check_copies(&self) -> PyResult<()> {
        self.copy_budget().map(|_| ()).map_err(copy_error)
    }

    /// Admit display strings, escaping intermediates, grouping tables and the
    /// Python Unicode copy before calling the existing serializers.
    fn check_text_copies(&self, table: bool) -> PyResult<()> {
        let mut budget = self.copy_budget().map_err(copy_error)?;
        budget.charge(4096).map_err(copy_error)?;
        for name in &self.columns {
            budget.repeated(name.len(), 64).map_err(copy_error)?;
        }
        for row in &self.rows {
            budget.charge(512).map_err(copy_error)?;
            for value in row {
                let size = display_bytes(value, self.conversion_limit).map_err(copy_error)?;
                // Escaping can double the input. The old formatter can retain
                // grouped keys, values, growing output and UCS4 output together.
                budget.repeated(size, 64).map_err(copy_error)?;
                budget.charge(1024).map_err(copy_error)?;
            }
        }
        if table {
            // format_result_table caps widths at 40 characters. UTF-8 borders,
            // padded/truncated cells and the final Unicode copy fit 1024/cell.
            let lines = self
                .__len__()
                .checked_add(4)
                .ok_or_else(|| copy_error(copy_limit_error()))?;
            for _ in &self.columns {
                budget.repeated(lines, 1024).map_err(copy_error)?;
            }
        }
        Ok(())
    }

    /// Serializes the query result to Arrow IPC stream bytes.
    #[cfg(feature = "arrow-export")]
    fn to_ipc_bytes(&self) -> PyResult<Vec<u8>> {
        let mut budget = self.copy_budget().map_err(copy_error)?;
        // Arrow conversion retains reference columns, builders, arrays, IPC
        // buffers (including growth) and the Python bytes copy simultaneously.
        budget.charge(16384).map_err(copy_error)?;
        for column in &self.columns {
            budget.charge(4096).map_err(copy_error)?;
            budget.repeated(column.len(), 32).map_err(copy_error)?;
            budget.repeated(self.__len__(), 1024).map_err(copy_error)?;
        }
        for row in &self.rows {
            for value in row {
                let size = display_bytes(value, self.conversion_limit).map_err(copy_error)?;
                budget.repeated(size, 32).map_err(copy_error)?;
                match value {
                    Value::Vector(values) => {
                        budget.repeated(values.len(), 32).map_err(copy_error)?;
                    }
                    Value::Bytes(values) => {
                        budget.repeated(values.len(), 16).map_err(copy_error)?;
                    }
                    _ => {}
                }
            }
        }
        let dense_rows;
        let rows = if let Some(columns) = &self.int64_cols {
            dense_rows = (0..self.__len__())
                .map(|idx| {
                    columns
                        .iter()
                        .map(|column| Value::Int64(column[idx]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            &dense_rows[..]
        } else {
            &self.rows[..]
        };
        let col_types = vec![grafeo_common::LogicalType::Any; self.columns.len()];
        let batch = grafeo_engine::database::arrow::query_result_to_record_batch(
            &self.columns,
            &col_types,
            rows,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow export failed: {e}"))
        })?;
        grafeo_engine::database::arrow::record_batch_to_ipc_stream(&batch).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow IPC failed: {e}"))
        })
    }

    /// Creates an empty result (used internally).
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: OwnedRows::empty(),
            int64_cols: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            current_row: 0,
            conversion_limit: default_conversion_limit(),
            execution_time_ms: None,
            rows_scanned: None,
        }
    }

    /// Validates that this result has triple-shaped columns (subject, predicate, object).
    fn validate_triple_columns(&self) -> PyResult<()> {
        let normalized = &self.columns;
        let has_s = normalized
            .iter()
            .any(|c| c.eq_ignore_ascii_case("subject") || c.eq_ignore_ascii_case("s"));
        let has_p = normalized
            .iter()
            .any(|c| c.eq_ignore_ascii_case("predicate") || c.eq_ignore_ascii_case("p"));
        let has_o = normalized
            .iter()
            .any(|c| c.eq_ignore_ascii_case("object") || c.eq_ignore_ascii_case("o"));

        if has_s && has_p && has_o {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "to_ntriples()/to_turtle() requires columns named [subject, predicate, object] (or [s, p, o])",
            ))
        }
    }

    /// Returns the column indices for (subject, predicate, object).
    /// Must be called after `validate_triple_columns`.
    fn triple_column_indices(&self) -> (usize, usize, usize) {
        let normalized = &self.columns;
        let si = normalized
            .iter()
            .position(|c| c.eq_ignore_ascii_case("subject") || c.eq_ignore_ascii_case("s"))
            .unwrap_or(0);
        let pi = normalized
            .iter()
            .position(|c| c.eq_ignore_ascii_case("predicate") || c.eq_ignore_ascii_case("p"))
            .unwrap_or(1);
        let oi = normalized
            .iter()
            .position(|c| c.eq_ignore_ascii_case("object") || c.eq_ignore_ascii_case("o"))
            .unwrap_or(2);
        (si, pi, oi)
    }

    /// Formats a `Value` as an N-Triples term.
    ///
    /// IRIs are wrapped in angle brackets, strings become quoted literals,
    /// and blank nodes are prefixed with `_:`.
    fn value_to_ntriples_term(val: &Value) -> String {
        match val {
            Value::String(s) => {
                let s_str: &str = s.as_ref();
                if s_str.starts_with("http://")
                    || s_str.starts_with("https://")
                    || s_str.starts_with("urn:")
                {
                    // Looks like an IRI
                    format!("<{s_str}>")
                } else if s_str.starts_with("_:") {
                    // Blank node
                    s_str.to_string()
                } else {
                    // Plain literal: escape special characters
                    let escaped = s_str
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t");
                    format!("\"{escaped}\"")
                }
            }
            Value::Int64(n) => format!("\"{n}\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            Value::Float64(f) => format!("\"{f}\"^^<http://www.w3.org/2001/XMLSchema#double>"),
            Value::Bool(b) => format!("\"{b}\"^^<http://www.w3.org/2001/XMLSchema#boolean>"),
            Value::Null => "\"\"".to_string(),
            other => {
                // Fallback: quote the display representation.
                format!("\"{}\"", other)
            }
        }
    }
}

/// Builds parameterized queries with a fluent API.
///
/// Add parameters with `.param("name", value)` to safely inject values
/// without string concatenation (prevents injection).
#[pyclass(name = "QueryBuilder")]
pub struct PyQueryBuilder {
    pub(crate) query: String,
    pub(crate) params: HashMap<String, Value>,
}

impl PyQueryBuilder {
    /// Creates a new query builder (Rust API).
    pub fn create(query: String) -> Self {
        Self {
            query,
            params: HashMap::new(),
        }
    }
}

#[pymethods]
impl PyQueryBuilder {
    /// Create a new query builder.
    #[new]
    fn new(query: String) -> Self {
        Self::create(query)
    }

    /// Set a parameter.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the value cannot be converted to a Grafeo type.
    fn param(&mut self, name: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = PyValue::from_py(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot convert parameter '{}' to a Grafeo value: {}",
                name, e
            ))
        })?;
        self.params.insert(name, v);
        Ok(())
    }

    /// Get the query string.
    #[getter]
    fn query(&self) -> &str {
        &self.query
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::*;
    use crate::types::take_copy_string_cost_evaluations;
    use grafeo_engine::database::QueryResult;

    // Retain the original row-by-row admission as an independent byte-threshold
    // oracle. In particular, maps retain their additional entity-copy charge.
    fn row_by_row_reference(
        result: &QueryResult,
        max_bytes: usize,
    ) -> grafeo_common::utils::error::Result<()> {
        let mut budget = CopyBudget::new(max_bytes);
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
        budget.dict(result.columns.len())?;
        for _ in &result.columns {
            budget.list(result.row_count())?;
        }
        if result.is_int64_columnar() {
            for _ in 0..result.row_count() {
                budget.dict(result.columns.len())?;
                for name in &result.columns {
                    budget.string(name)?;
                    budget.charge(128)?;
                }
            }
        } else {
            for row in result.rows() {
                budget.row(&result.columns, row)?;
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

    fn assert_exact_threshold(result: &QueryResult) {
        let mut denied = 0;
        let mut admitted = 1 << 24;
        assert!(row_by_row_reference(result, denied).is_err());
        assert!(row_by_row_reference(result, admitted).is_ok());
        while denied + 1 < admitted {
            let midpoint = denied + (admitted - denied) / 2;
            if row_by_row_reference(result, midpoint).is_ok() {
                admitted = midpoint;
            } else {
                denied = midpoint;
            }
        }
        for limit in [0, denied, admitted, admitted + 1, usize::MAX] {
            let expected = row_by_row_reference(result, limit);
            let actual = preflight_query_result(result, limit);
            assert_eq!(actual.is_ok(), expected.is_ok(), "limit {limit}");
            if let (Err(actual), Err(expected)) = (actual, expected) {
                assert_eq!(actual.to_string(), expected.to_string());
            }
        }
    }

    #[test]
    fn preflight_column_name_cost_is_independent_of_row_count() {
        for count in [32, 64, 128] {
            let result = QueryResult::from_rows(
                vec!["identity".into(), "visible".into()],
                vec![vec![Value::Int64(7), Value::Bool(true)]; count],
            );
            assert_exact_threshold(&result);
            take_copy_string_cost_evaluations();
            row_by_row_reference(&result, usize::MAX).unwrap();
            assert_eq!(take_copy_string_cost_evaluations(), 2 * (count + 1));
            preflight_query_result(&result, usize::MAX).unwrap();
            assert_eq!(
                take_copy_string_cost_evaluations(),
                4,
                "{count} rows: global columns and one repeated row-header measurement"
            );
        }
    }

    #[test]
    fn preflight_byte_threshold_preserves_nested_entities_schema_status_and_empty_rows() {
        use grafeo_common::types::{LogicalType, PropertyKey};
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let entity = Value::Map(Arc::new(BTreeMap::from([
            (PropertyKey::new("_id"), Value::Int64(1)),
            (
                PropertyKey::new("payload"),
                Value::List(vec![Value::from("😀".repeat(17)), Value::Null].into()),
            ),
        ])));
        for count in [0, 1, 8] {
            let mut result =
                QueryResult::from_rows(vec!["entity".repeat(7)], vec![vec![entity.clone()]; count]);
            result.column_types = vec![LogicalType::Struct(vec![(
                "nested".repeat(5),
                LogicalType::List(Box::new(LogicalType::String)),
            )])];
            result.status_message = Some("status".repeat(11));
            assert_exact_threshold(&result);
        }
        assert_exact_threshold(&QueryResult::empty());
        assert_exact_threshold(&QueryResult::from_rows(Vec::new(), vec![Vec::new(); 7]));
    }

    #[cfg(feature = "gql")]
    #[test]
    fn preflight_dense_int64_preserves_threshold_and_constant_column_work() {
        let db = grafeo_engine::GrafeoDB::new_in_memory();
        db.execute("UNWIND [1, 2, 3, 4] AS v INSERT (:CopyRow {value: v})")
            .unwrap();
        let result = db.execute("MATCH (n:CopyRow) RETURN id(n)").unwrap();
        assert!(result.is_int64_columnar());
        assert_eq!(result.row_count(), 4);
        let expected = result.int64_column(0).unwrap().to_vec();
        assert_exact_threshold(&result);
        take_copy_string_cost_evaluations();
        preflight_query_result(&result, usize::MAX).unwrap();
        assert_eq!(take_copy_string_cost_evaluations(), 2);
        assert!(result.is_int64_columnar());
        assert_eq!(result.int64_column(0).unwrap(), expected);
    }
}
