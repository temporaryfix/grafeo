//! Your main entry point for using Grafeo from Python.
//!
//! [`PyGrafeoDB`] wraps the Rust database engine and gives you a Pythonic API.
//! Start here - create a database, run queries, and manage transactions.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
#[cfg(feature = "gql")]
use pyo3_async_runtimes::tokio::future_into_py;

use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::types::{LogicalType, Value};
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
use grafeo_core::graph::Direction;
use grafeo_engine::config::{Config, GraphModel};
use grafeo_engine::database::{GrafeoDB, OwnedRows, QueryResult};

fn prepare_python_execution(
    language: &str,
    params: Option<&Bound<'_, pyo3::types::PyDict>>,
    control: Option<&crate::control::PyQueryControl>,
    max_rows: Option<usize>,
    max_bytes: Option<usize>,
) -> PyResult<(
    HashMap<String, Value>,
    grafeo_engine::query::ExecutionOptions,
)> {
    let mut values = HashMap::new();
    if let Some(params) = params {
        for (key, value) in params.iter() {
            values.insert(key.extract()?, PyValue::from_py(&value)?);
        }
    }
    let defaults = grafeo_engine::query::ResultLimits::default();
    let limits = grafeo_engine::query::ResultLimits {
        max_rows: max_rows.unwrap_or(defaults.max_rows),
        max_bytes: max_bytes.unwrap_or(defaults.max_bytes),
    };
    let control = match control {
        Some(control) => control.take_control()?,
        None => grafeo_core::execution::QueryExecutionControl::new(),
    };
    Ok((
        values,
        grafeo_engine::query::ExecutionOptions {
            control,
            language: Some(language.to_owned()),
            result_limits: Some(limits),
            result_admission: Some(crate::query::admit_python_result),
        },
    ))
}

#[cfg(feature = "triple-store")]
fn parse_python_rdf_term(s: &str) -> PyResult<grafeo_engine::Term> {
    let s = s.trim();
    grafeo_engine::Term::from_ntriples(s)
        .or_else(|| {
            if s.starts_with('"') || s.starts_with("_:") || s.starts_with('<') || s.is_empty() {
                None
            } else {
                Some(grafeo_engine::Term::iri(s))
            }
        })
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid RDF term '{s}': expected N-Triples or a bare IRI"
            ))
        })
}

#[cfg(feature = "triple-store")]
fn parse_python_rdf_quad(
    subject: &str,
    predicate: &str,
    object: &str,
    graph: Option<&str>,
) -> PyResult<grafeo_engine::Quad> {
    let subject_term = parse_python_rdf_term(subject)?;
    if !subject_term.is_iri() && !subject_term.is_blank_node() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "RDF subject must be an IRI or blank node",
        ));
    }
    let predicate_term = parse_python_rdf_term(predicate)?;
    if !predicate_term.is_iri() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "RDF predicate must be an IRI",
        ));
    }
    let triple =
        grafeo_engine::Triple::new(subject_term, predicate_term, parse_python_rdf_term(object)?);
    match graph {
        Some(g) if !g.is_empty() => {
            let iri = g
                .strip_prefix('<')
                .and_then(|inner| inner.strip_suffix('>'))
                .unwrap_or(g);
            Ok(grafeo_engine::Quad::named(triple, iri))
        }
        _ => Ok(grafeo_engine::Quad::new(triple)),
    }
}

#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
fn parse_asof_direction(direction: &str) -> PyResult<Direction> {
    match direction {
        "outgoing" | "out" => Ok(Direction::Outgoing),
        "incoming" | "in" => Ok(Direction::Incoming),
        "both" => Ok(Direction::Both),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown direction '{other}': expected 'outgoing', 'incoming', or 'both'"
        ))),
    }
}

/// Parses a section name string ("LpgStore", "VectorStore", etc.) into a
/// [`SectionType`]. Returns `Err` for unknown names.
fn parse_section_type(name: &str) -> Result<SectionType, PyErr> {
    match name {
        "Catalog" => Ok(SectionType::Catalog),
        "LpgStore" => Ok(SectionType::LpgStore),
        "RdfStore" => Ok(SectionType::RdfStore),
        "CompactStore" => Ok(SectionType::CompactStore),
        "VectorStore" => Ok(SectionType::VectorStore),
        "TextIndex" => Ok(SectionType::TextIndex),
        "RdfRing" => Ok(SectionType::RdfRing),
        "PropertyIndex" => Ok(SectionType::PropertyIndex),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown section type '{other}': expected one of \
             Catalog, LpgStore, RdfStore, CompactStore, VectorStore, \
             TextIndex, RdfRing, PropertyIndex"
        ))),
    }
}

/// Parses a tier name ("auto", "force_ram", "force_disk") into a
/// [`TierOverride`].
fn parse_tier_override(name: &str) -> Result<TierOverride, PyErr> {
    match name {
        "auto" | "Auto" => Ok(TierOverride::Auto),
        "force_ram" | "ForceRam" => Ok(TierOverride::ForceRam),
        "force_disk" | "ForceDisk" => Ok(TierOverride::ForceDisk),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown tier '{other}': expected 'auto', 'force_ram', or 'force_disk'"
        ))),
    }
}

/// Renders a [`SectionType`] back into the string used by the Python API.
fn section_type_to_str(section_type: SectionType) -> &'static str {
    match section_type {
        SectionType::Catalog => "Catalog",
        SectionType::LpgStore => "LpgStore",
        SectionType::RdfStore => "RdfStore",
        SectionType::CompactStore => "CompactStore",
        SectionType::OverlayDeletions => "OverlayDeletions",
        SectionType::VectorStore => "VectorStore",
        SectionType::TextIndex => "TextIndex",
        SectionType::RdfRing => "RdfRing",
        SectionType::PropertyIndex => "PropertyIndex",
        _ => "Unknown",
    }
}

/// Renders a [`grafeo_common::memory::buffer::StorageTier`] into the
/// lowercase string used by the Python API.
fn tier_to_str(tier: grafeo_common::memory::buffer::StorageTier) -> &'static str {
    use grafeo_common::memory::buffer::StorageTier;
    match tier {
        StorageTier::InMemory => "in_memory",
        StorageTier::OnDisk => "on_disk",
        StorageTier::Uninitialized => "uninitialized",
        _ => "unknown",
    }
}

#[cfg(feature = "algos")]
use crate::bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
use crate::error::PyGrafeoError;
use crate::graph::{PyEdge, PyNode};
use crate::query::{PyQueryBuilder, PyQueryResult};
use crate::types::PyValue;

/// Holds results from async query execution.
///
/// Works like [`PyQueryResult`], including [`nodes()`](Self::nodes) and
/// [`edges()`](Self::edges) extraction. Iterate directly or call
/// [`rows()`](Self::rows) to get all data.
#[pyclass(name = "AsyncQueryResult")]
pub struct AsyncQueryResult {
    columns: Vec<String>,
    rows: OwnedRows,
    #[allow(dead_code)] // Stored for future typed access; currently only raw rows exposed
    column_types: Vec<LogicalType>,
    nodes: Vec<PyNode>,
    edges: Vec<PyEdge>,
    conversion_limit: usize,
}

impl AsyncQueryResult {
    fn admit_conversion(&self) -> PyResult<()> {
        let mut budget = crate::types::CopyBudget::new(self.conversion_limit);
        budget
            .columns(&self.columns)
            .map_err(crate::types::copy_error)?;
        budget
            .list(self.rows.len())
            .map_err(crate::types::copy_error)?;
        for row in &self.rows {
            budget
                .row(&self.columns, row)
                .map_err(crate::types::copy_error)?;
        }
        crate::query::admit_entity_copies(&mut budget, &self.nodes, &self.edges)
            .map_err(crate::types::copy_error)?;
        Ok(())
    }
}

#[pymethods]
impl AsyncQueryResult {
    /// Get column names.
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::types::columns_to_py_bounded(py, &self.columns, self.conversion_limit)
    }

    /// Get all nodes from the result.
    fn nodes(&self) -> PyResult<Vec<PyNode>> {
        self.admit_conversion()?;
        Ok(self.nodes.clone())
    }

    /// Get all edges from the result.
    fn edges(&self) -> PyResult<Vec<PyEdge>> {
        self.admit_conversion()?;
        Ok(self.edges.clone())
    }

    /// Get all rows as a list of lists.
    fn rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.admit_conversion()?;
        let list = crate::types::new_list(py)?;
        for row in &self.rows {
            let py_row = crate::types::new_list(py)?;
            for val in row {
                let py_val = PyValue::to_py_admitted(val, py)?;
                py_row.append(py_val)?;
            }
            list.append(py_row)?;
        }
        Ok(list.into())
    }

    /// Get the number of rows.
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    /// Iterate over rows.
    fn __iter__(slf: PyRef<'_, Self>) -> AsyncQueryResultIter {
        AsyncQueryResultIter {
            owner: slf.into(),
            index: 0,
        }
    }

    /// Convert to a pandas DataFrame.
    ///
    /// Requires pandas to be installed (`uv add pandas`).
    #[pyo3(signature = ())]
    fn to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.admit_conversion()?;
        let pd = py.import("pandas").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pandas is required for to_pandas(). Install it with: uv add pandas",
            )
        })?;

        let data = crate::types::new_dict(py)?;
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let values = crate::types::new_list(py)?;
            for row in &self.rows {
                let val = row
                    .get(col_idx)
                    .map_or_else(|| Ok(py.None()), |v| PyValue::to_py_admitted(v, py))?;
                values.append(val)?;
            }
            data.set_item(col_name, values)?;
        }

        let df = pd.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    /// Convert to a polars DataFrame.
    ///
    /// Requires polars to be installed (`uv add polars`).
    #[pyo3(signature = ())]
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.admit_conversion()?;
        let pl = py.import("polars").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "polars is required for to_polars(). Install it with: uv add polars",
            )
        })?;

        let data = crate::types::new_dict(py)?;
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let values = crate::types::new_list(py)?;
            for row in &self.rows {
                let val = row
                    .get(col_idx)
                    .map_or_else(|| Ok(py.None()), |v| PyValue::to_py_admitted(v, py))?;
                values.append(val)?;
            }
            data.set_item(col_name, values)?;
        }

        let df = pl.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut budget = crate::types::CopyBudget::new(self.conversion_limit);
        budget.charge(512).map_err(crate::types::copy_error)?;
        for column in &self.columns {
            budget
                .repeated(column.len(), 64)
                .map_err(crate::types::copy_error)?;
        }
        Ok(format!(
            "AsyncQueryResult(columns={:?}, rows={})",
            self.columns,
            self.rows.len()
        ))
    }
}

/// Iterates through async query result rows one at a time.
#[pyclass]
pub struct AsyncQueryResultIter {
    owner: Py<AsyncQueryResult>,
    index: usize,
}

#[pymethods]
impl AsyncQueryResultIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let py_row = {
            let owner = slf.owner.bind(py).borrow();
            let Some(row) = owner.rows.get(slf.index) else {
                return Ok(None);
            };
            let mut budget = crate::types::CopyBudget::new(owner.conversion_limit);
            budget
                .row(&owner.columns, row)
                .map_err(crate::types::copy_error)?;
            let py_row = crate::types::new_list(py)?;
            for val in row {
                let py_val = PyValue::to_py_admitted(val, py)?;
                py_row.append(py_val)?;
            }
            py_row
        };
        slf.index += 1;
        Ok(Some(py_row.into()))
    }
}

/// Your connection to a Grafeo database.
///
/// Create one with `GrafeoDB()` for in-memory storage (fast, temporary) or
/// `GrafeoDB("path/to/db")` for persistent storage (survives restarts).
/// Then use [`execute()`](Self::execute) to run GQL queries.
///
/// Unlike the Rust API (which uses `db.session()` for query execution),
/// Python calls `db.execute()` directly. For transactions, use
/// `db.begin_transaction()` as a context manager:
///
/// ```python
/// with db.begin_transaction() as tx:
///     tx.execute("INSERT (:Person {name: 'Alix'})")
///     tx.commit()
/// ```
#[pyclass(name = "GrafeoDB")]
pub struct PyGrafeoDB {
    inner: Arc<RwLock<GrafeoDB>>,
}

impl PyGrafeoDB {
    /// Converts an optional Python dict of property filters to a Rust HashMap.
    #[cfg(any(feature = "vector-index", feature = "hybrid-search"))]
    fn convert_filters(
        filters: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Option<HashMap<String, Value>>> {
        let Some(dict) = filters else {
            return Ok(None);
        };
        let mut map = HashMap::new();
        for (key, value) in dict.iter() {
            let key_str: String = key.extract()?;
            let val = PyValue::from_py(&value)?;
            map.insert(key_str, val);
        }
        Ok(Some(map))
    }

    /// Builds Arrow IPC bytes for all nodes (Rust-only helper).
    #[cfg(feature = "arrow-export")]
    fn nodes_ipc_bytes(&self) -> PyResult<Vec<u8>> {
        let db = self.inner.read();
        let session = db.session();
        let _snapshot = session.snapshot().map_err(PyGrafeoError::from)?;
        let nodes: Vec<_> = db.iter_nodes().collect();
        grafeo_engine::database::arrow::nodes_to_ipc_stream(&nodes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow export failed: {e}"))
        })
    }

    /// Builds Arrow IPC bytes for all edges (Rust-only helper).
    #[cfg(feature = "arrow-export")]
    fn edges_ipc_bytes(&self) -> PyResult<Vec<u8>> {
        let db = self.inner.read();
        let session = db.session();
        let _snapshot = session.snapshot().map_err(PyGrafeoError::from)?;
        let edges: Vec<_> = db.iter_edges().collect();
        grafeo_engine::database::arrow::edges_to_ipc_stream(&edges).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow export failed: {e}"))
        })
    }

    /// Executes a query in the given language, converting Python params and
    /// extracting entities from the result.
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword options share one native execution owner"
    )]
    fn execute_language_impl(
        &self,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let (params, options) =
            prepare_python_execution(language, params, control, max_rows, max_bytes)?;
        let conversion_limit = options.result_limits.unwrap_or_default().max_bytes;
        let mut result = py
            .detach(|| {
                self.inner
                    .read()
                    .execute_with_options(query, params, options)
            })
            .map_err(PyGrafeoError::from)?;
        let db = self.inner.read();
        let (nodes, edges) = if result.is_int64_columnar() {
            (Vec::new(), Vec::new())
        } else {
            extract_entities(&result, &db)
        };
        let columns = std::mem::take(&mut result.columns);
        let exec_time = result.execution_time_ms;
        let scanned = result.rows_scanned;
        let int64_cols = result.take_int64_cols();
        let rows = result.into_rows().map_err(PyGrafeoError::from)?;
        Ok(
            PyQueryResult::with_metrics(columns, rows, nodes, edges, exec_time, scanned)
                .with_int64_cols(int64_cols)
                .with_conversion_limit(conversion_limit),
        )
    }
}

#[pymethods]
impl PyGrafeoDB {
    /// Creates a database. Pass a path for persistence, or omit for in-memory.
    ///
    /// Examples:
    ///     db = GrafeoDB()           # In-memory (fast, temporary)
    ///     db = GrafeoDB("./mydb")   # Persistent (survives restarts)
    ///
    ///     # Pin specific sections to a storage tier:
    ///     db = GrafeoDB("./mydb", section_tiers={
    ///         "VectorStore": "force_disk",   # spill at open
    ///         "LpgStore": "force_ram",       # never spill
    ///     })
    ///
    /// section_tiers keys: "Catalog", "LpgStore", "RdfStore", "CompactStore",
    /// "VectorStore", "TextIndex", "RdfRing", "PropertyIndex".
    /// Values: "auto" (default), "force_ram", "force_disk".
    #[new]
    #[pyo3(signature = (path=None, *, cdc=false, section_tiers=None, graph_model=None))]
    fn new(
        path: Option<String>,
        cdc: bool,
        section_tiers: Option<HashMap<String, String>>,
        graph_model: Option<String>,
    ) -> PyResult<Self> {
        let mut config = if let Some(p) = path {
            Config::persistent(p)
        } else {
            Config::in_memory()
        };
        if let Some(model) = graph_model {
            let parsed = GraphModel::from_name(&model).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown graph_model '{model}': expected 'lpg', 'rdf', or 'both'"
                ))
            })?;
            config = config.with_graph_model(parsed);
        }
        if cdc {
            config = config.with_cdc();
        }
        if let Some(tiers) = section_tiers {
            for (section_name, tier_name) in tiers {
                let section_type = parse_section_type(&section_name)?;
                let tier = parse_tier_override(&tier_name)?;
                config = config.with_section_config(
                    section_type,
                    SectionMemoryConfig {
                        max_ram: None,
                        tier,
                    },
                );
            }
        }

        let db = GrafeoDB::with_config(config).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Returns the current storage tier of every registered section.
    ///
    /// Result is a dict mapping section name to tier name. Tier values
    /// are "in_memory", "on_disk", or "uninitialized".
    ///
    /// Example:
    ///     >>> db.storage_tiers()
    ///     {'LpgStore': 'in_memory', 'VectorStore': 'on_disk'}
    fn storage_tiers(&self) -> HashMap<String, String> {
        let db = self.inner.read();
        db.storage_tiers()
            .into_iter()
            .map(|(section_type, tier)| {
                (
                    section_type_to_str(section_type).to_string(),
                    tier_to_str(tier).to_string(),
                )
            })
            .collect()
    }

    /// Reloads spilled section data back into RAM, up to `target_fraction`
    /// of the memory budget.
    ///
    /// Walks consumers currently OnDisk in priority order (highest first)
    /// and reloads each as long as projected memory stays below the target.
    /// Returns the number of consumers reloaded.
    ///
    /// `target_fraction` is clamped to [0.0, 1.0]. Use 0.7 (matching the
    /// default soft-limit) as a sane starting point.
    ///
    /// Example:
    ///     >>> db.reload_eligible(0.7)
    ///     2
    #[pyo3(signature = (target_fraction=0.7))]
    fn reload_eligible(&self, target_fraction: f64) -> usize {
        let db = self.inner.read();
        db.reload_eligible(target_fraction)
    }

    /// Open an existing database.
    #[staticmethod]
    fn open(path: String) -> PyResult<Self> {
        let config = Config::persistent(path);
        let db = GrafeoDB::with_config(config).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database in read-only mode.
    ///
    /// Uses a shared file lock, so multiple processes can read the same
    /// .grafeo file concurrently. Mutations will raise an error.
    ///
    /// Args:
    ///     path: Path to the .grafeo database file.
    ///
    /// Examples:
    ///     db = GrafeoDB.open_read_only("./my_graph.grafeo")
    ///     result = db.execute("MATCH (n) RETURN n LIMIT 10")
    #[staticmethod]
    fn open_read_only(path: String) -> PyResult<Self> {
        let config = Config::read_only(path);
        let db = GrafeoDB::with_config(config).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Runs a GQL query and returns the results.
    ///
    /// Use params for parameterized queries to avoid injection:
    ///     result = db.execute("MATCH (p:Person {name: $name}) RETURN p", {"name": "Alix"})
    ///
    /// Query performance metrics are available via `result.execution_time_ms`
    /// and `result.rows_scanned` properties.
    #[cfg(feature = "gql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("gql", query, params, control, max_rows, max_bytes, py)
    }

    /// Runs a read-only query with bounded native chunks and copied Python rows.
    /// Use a context manager or close() to release the publication snapshot.
    /// Parameters, DISTINCT and supported PROFILE queries share the eager
    /// execution control; unsupported native stream operators return an error.
    #[cfg(feature = "gql")]
    #[pyo3(signature = (query, params=None, *, control=None, max_rows=None, max_bytes=None))]
    fn execute_lazy(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<crate::stream::PyResultStream> {
        let (params, mut options) =
            prepare_python_execution("gql", params, control, max_rows, max_bytes)?;
        let conversion_limit = options.result_limits.unwrap_or_default().max_bytes;
        // Streaming rows are admitted by the binding on each pull. Eager
        // mutation-result admission is a different, pre-publication boundary.
        options.result_admission = None;
        let stream = py
            .detach(|| {
                self.inner
                    .read()
                    .stream_with_options(query, params, options)
            })
            .map_err(PyGrafeoError::from)?;
        Ok(crate::stream::PyResultStream::new(
            Arc::clone(&self.inner),
            stream,
            conversion_limit,
            max_rows,
        ))
    }

    /// Execute a GQL query at a specific historical epoch.
    ///
    /// Returns results as they would have appeared at the given epoch.
    /// This is a point-in-time query: all nodes, edges, and properties
    /// reflect the state at that epoch.
    ///
    /// Example:
    ///     result = db.execute_at_epoch("MATCH (n:Server) RETURN n.status", epoch=5)
    #[cfg(feature = "gql")]
    #[pyo3(signature = (query, epoch, params=None, *, control=None, max_rows=None, max_bytes=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword options share one native execution owner"
    )]
    fn execute_at_epoch(
        &self,
        query: &str,
        epoch: u64,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        let (params, options) =
            prepare_python_execution("gql", params, control, max_rows, max_bytes)?;
        let conversion_limit = options.result_limits.unwrap_or_default().max_bytes;
        let mut result = py
            .detach(|| {
                self.inner.read().session().execute_at_epoch_with_options(
                    query,
                    grafeo_common::types::EpochId::new(epoch),
                    params,
                    options,
                )
            })
            .map_err(PyGrafeoError::from)?;
        let (nodes, edges) = extract_entities(&result, &self.inner.read());
        let columns = std::mem::take(&mut result.columns);
        let exec_time = result.execution_time_ms;
        let scanned = result.rows_scanned;
        Ok(PyQueryResult::with_metrics(
            columns,
            result.into_rows().map_err(PyGrafeoError::from)?,
            nodes,
            edges,
            exec_time,
            scanned,
        )
        .with_conversion_limit(conversion_limit))
    }

    /// Execute a query and return a query builder.
    fn query(&self, query: String) -> PyQueryBuilder {
        PyQueryBuilder::create(query)
    }

    /// Execute a Cypher query.
    #[cfg(feature = "cypher")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_cypher(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("cypher", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    #[cfg(feature = "sql-pgq")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_sql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("sql", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a GQL query asynchronously.
    ///
    /// Returns a Python awaitable that can be used with ``asyncio``.
    /// Qualified read-only ORDER BY queries schedule input/output batches and
    /// sort finalization on the blocking pool, retaining resource and cancellation owners
    /// across awaits. Other query shapes execute on the blocking pool in one
    /// call. Both routes release the GIL so other coroutines can make progress.
    ///
    /// Example:
    /// ```python
    /// async def main():
    ///     db = GrafeoDB()
    ///     result = await db.execute_async("MATCH (n:Person) RETURN n")
    ///     for row in result:
    ///         print(row)
    ///
    /// asyncio.run(main())
    /// ```
    #[cfg(feature = "gql")]
    #[pyo3(signature = (query, params=None, *, control=None, max_rows=None, max_bytes=None))]
    fn execute_async<'py>(
        &self,
        py: Python<'py>,
        query: String,
        params: Option<&Bound<'py, pyo3::types::PyDict>>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (params, options) =
            prepare_python_execution("gql", params, control, max_rows, max_bytes)?;
        let conversion_limit = options.result_limits.unwrap_or_default().max_bytes;
        let db = self.inner.clone();
        future_into_py(py, async move {
            #[cfg(feature = "async-query")]
            let mut result = {
                use grafeo_engine::query::executor::AsyncSortDispatch;

                let dispatch = tokio::task::spawn_blocking(move || {
                    let dispatch = db
                        .read()
                        .execute_or_prepare_async_sort(&query, params, options)?;
                    Ok::<_, grafeo_common::utils::error::Error>(match dispatch {
                        AsyncSortDispatch::Completed(result) => {
                            AsyncSortDispatch::Completed(result)
                        }
                        AsyncSortDispatch::Prepared(prepared) => {
                            AsyncSortDispatch::Prepared(prepared.retain_database_owner(db))
                        }
                    })
                })
                .await
                .map_err(|error| PyGrafeoError::database(error.to_string()))?
                .map_err(PyGrafeoError::from)?;
                match dispatch {
                    AsyncSortDispatch::Completed(result) => result,
                    AsyncSortDispatch::Prepared(prepared) => {
                        prepared.execute().await.map_err(PyGrafeoError::from)?
                    }
                }
            };
            #[cfg(not(feature = "async-query"))]
            let mut result = tokio::task::spawn_blocking(move || {
                db.read().execute_with_options(&query, params, options)
            })
            .await
            .map_err(|error| PyGrafeoError::database(error.to_string()))?
            .map_err(PyGrafeoError::from)?;
            let (nodes, edges) = if result.is_int64_columnar() {
                (Vec::new(), Vec::new())
            } else {
                grafeo_bindings_common::entity::extract_and_map(
                    &result,
                    |n| PyNode::new(n.id, n.labels, n.properties),
                    |e| PyEdge::new(e.id, e.edge_type, e.source_id, e.target_id, e.properties),
                )
            };
            let columns = std::mem::take(&mut result.columns);
            let column_types = std::mem::take(&mut result.column_types);
            Ok(AsyncQueryResult {
                columns,
                rows: result.into_rows().map_err(PyGrafeoError::from)?,
                column_types,
                nodes,
                edges,
                conversion_limit,
            })
        })
    }

    /// Execute a Gremlin query.
    #[cfg(feature = "gremlin")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_gremlin(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("gremlin", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a GraphQL query.
    #[cfg(feature = "graphql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_graphql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("graphql", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a SPARQL query against the RDF triple store.
    ///
    /// SPARQL is the W3C standard query language for RDF data.
    ///
    /// Example:
    ///     result = db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    #[cfg(feature = "sparql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_sparql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("sparql", query, params, control, max_rows, max_bytes, py)
    }

    /// Return the physical execution plan for a SPARQL query without executing it.
    ///
    /// Equivalent to ``db.execute_sparql("EXPLAIN " + query)``.
    #[cfg(feature = "sparql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn explain_sparql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(
            "sparql",
            &format!("EXPLAIN {query}"),
            params,
            control,
            max_rows,
            max_bytes,
            py,
        )
    }

    /// Return the physical execution plan for a GQL query without executing it.
    ///
    /// Equivalent to ``db.execute("EXPLAIN " + query)``.
    #[cfg(feature = "gql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn explain(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(
            "gql",
            &format!("EXPLAIN {query}"),
            params,
            control,
            max_rows,
            max_bytes,
            py,
        )
    }

    /// Return the physical execution plan for a Cypher query without executing it.
    ///
    /// Equivalent to ``db.execute_cypher("EXPLAIN " + query)``.
    #[cfg(feature = "cypher")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn explain_cypher(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(
            "cypher",
            &format!("EXPLAIN {query}"),
            params,
            control,
            max_rows,
            max_bytes,
            py,
        )
    }

    /// Return the physical execution plan for a SQL/PGQ query without executing it.
    ///
    /// Equivalent to ``db.execute_sql("EXPLAIN " + query)``.
    #[cfg(feature = "sql-pgq")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn explain_sql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(
            "sql",
            &format!("EXPLAIN {query}"),
            params,
            control,
            max_rows,
            max_bytes,
            py,
        )
    }

    /// Return the physical execution plan for a Gremlin query without executing it.
    ///
    /// Equivalent to ``db.execute_gremlin("EXPLAIN " + query)``.
    #[cfg(feature = "gremlin")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn explain_gremlin(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(
            "gremlin",
            &format!("EXPLAIN {query}"),
            params,
            control,
            max_rows,
            max_bytes,
            py,
        )
    }

    /// Validate the default graph against SHACL shapes in a named graph.
    ///
    /// Returns a dict with ``conforms`` (bool), ``results`` (list), and ``results_text`` (str).
    ///
    /// Example:
    ///     report = db.validate_shacl("http://example.org/shapes")
    ///     if not report["conforms"]:
    ///         print(report["results_text"])
    #[cfg(feature = "shacl")]
    fn validate_shacl(
        &self,
        shapes_graph: &str,
        py: Python<'_>,
    ) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let db = self.inner.read();
        let session = db.session();
        let report = session
            .validate_shacl(shapes_graph)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("conforms", report.conforms)?;
        dict.set_item("results_text", format!("{report}"))?;

        let results_list = pyo3::types::PyList::empty(py);
        for r in &report.results {
            let rdict = pyo3::types::PyDict::new(py);
            rdict.set_item("focus_node", r.focus_node.to_string())?;
            rdict.set_item("severity", format!("{:?}", r.severity))?;
            rdict.set_item(
                "source_constraint_component",
                &r.source_constraint_component,
            )?;
            rdict.set_item("source_shape", r.source_shape.to_string())?;
            if let Some(ref v) = r.value {
                rdict.set_item("value", v.to_string())?;
            }
            if let Some(ref msg) = r.message {
                rdict.set_item("message", msg.as_str())?;
            }
            results_list.append(rdict)?;
        }
        dict.set_item("results", results_list)?;

        Ok(dict.into())
    }

    /// Execute a query in a named language (e.g. `"graphql-rdf"`).
    #[pyo3(signature = (language, query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword options share one native execution owner"
    )]
    fn execute_language(
        &self,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(language, query, params, control, max_rows, max_bytes, py)
    }

    /// Graph model this database was created with: `"lpg"`, `"rdf"`, or `"both"`.
    fn graph_model(&self) -> &'static str {
        self.inner.read().graph_model().as_name()
    }

    /// Insert one RDF quad. Terms are N-Triples or bare IRIs.
    ///
    /// Returns ``(inserted, epoch)``. ``graph=None`` is the default graph.
    #[cfg(feature = "triple-store")]
    #[pyo3(signature = (subject, predicate, object, graph=None))]
    fn insert_rdf_quad(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> PyResult<(usize, u64)> {
        let quad = parse_python_rdf_quad(subject, predicate, object, graph)?;
        let db = self.inner.read();
        let (n, epoch) = db.insert_rdf_quads([quad]).map_err(PyGrafeoError::from)?;
        Ok((n, epoch.as_u64()))
    }

    /// Bulk-insert RDF quads. Each item is ``(subject, predicate, object)`` or
    /// ``(subject, predicate, object, graph)``.
    ///
    /// Returns ``(inserted, epoch)``.
    #[cfg(feature = "triple-store")]
    fn insert_rdf_quads(
        &self,
        quads: Vec<(String, String, String, Option<String>)>,
    ) -> PyResult<(usize, u64)> {
        let parsed: Vec<grafeo_engine::Quad> = quads
            .iter()
            .map(|(s, p, o, g)| parse_python_rdf_quad(s, p, o, g.as_deref()))
            .collect::<PyResult<_>>()?;
        let db = self.inner.read();
        let (n, epoch) = db.insert_rdf_quads(parsed).map_err(PyGrafeoError::from)?;
        Ok((n, epoch.as_u64()))
    }

    /// Exact typed-quad membership (lexical form + datatype + graph).
    #[cfg(feature = "triple-store")]
    #[pyo3(signature = (subject, predicate, object, graph=None))]
    fn contains_rdf_quad(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> PyResult<bool> {
        let quad = parse_python_rdf_quad(subject, predicate, object, graph)?;
        Ok(self
            .inner
            .read()
            .try_contains_rdf_quad(&quad)
            .map_err(PyGrafeoError::from)?)
    }

    /// Create a node.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (labels, properties=None))]
    fn create_node(
        &self,
        mut labels: Vec<String>,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyNode> {
        let db = self.inner.read();
        let session = db.session();

        // Convert labels from Vec<String> to Vec<&str>
        labels.sort_unstable();
        labels.dedup();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let mut props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = Vec::new();
        if let Some(p) = properties {
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
        }
        let id = session
            .create_node_with_props(
                &label_refs,
                props
                    .iter()
                    .map(|(key, value)| (key.as_str(), value.clone())),
            )
            .map_err(PyGrafeoError::from)?;
        // Return the committed creation image without resolving the graph again.
        // Null removes a property in storage; do not return it as a retained key.
        props.retain(|(_, value)| !matches!(value, grafeo_common::types::Value::Null));
        Ok(PyNode::new(id, labels, props.into_iter().collect()))
    }

    /// Create an edge between two nodes.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (source_id, target_id, edge_type, properties=None))]
    fn create_edge(
        &self,
        source_id: u64,
        target_id: u64,
        edge_type: String,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyEdge> {
        let db = self.inner.read();
        let session = db.session();
        let src = NodeId(source_id);
        let dst = NodeId(target_id);

        let mut props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = Vec::new();
        if let Some(p) = properties {
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
        }
        let id = session
            .create_edge_with_props(
                src,
                dst,
                &edge_type,
                props
                    .iter()
                    .map(|(key, value)| (key.as_str(), value.clone())),
            )
            .map_err(PyGrafeoError::from)?;
        props.retain(|(_, value)| !matches!(value, grafeo_common::types::Value::Null));
        Ok(PyEdge::new(
            id,
            edge_type,
            src,
            dst,
            props.into_iter().collect(),
        ))
    }

    /// Get a node by ID.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_node(&self, id: u64) -> PyResult<Option<PyNode>> {
        let db = self.inner.read();
        let node_id = NodeId(id);

        if let Some(node) = db.get_node(node_id) {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            > = node.properties.into_iter().collect();
            Ok(Some(PyNode::new(node_id, labels, properties)))
        } else {
            Ok(None)
        }
    }

    /// Get an edge by ID.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_edge(&self, id: u64) -> PyResult<Option<PyEdge>> {
        let db = self.inner.read();
        let edge_id = EdgeId(id);

        if let Some(edge) = db.get_edge(edge_id) {
            let properties: HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            > = edge.properties.into_iter().collect();
            Ok(Some(PyEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )))
        } else {
            Ok(None)
        }
    }

    /// Get a node at a specific historical epoch.
    ///
    /// Returns the node as it existed at the given epoch, including
    /// properties and labels at that point in time.
    ///
    /// Returns None if the node didn't exist at that epoch.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_node_at_epoch(&self, id: u64, epoch: u64) -> PyResult<Option<PyNode>> {
        let db = self.inner.read();
        let node_id = NodeId(id);
        let epoch_id = grafeo_common::types::EpochId::new(epoch);

        if let Some(node) = db.get_node_at_epoch(node_id, epoch_id) {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            > = node.properties.into_iter().collect();
            Ok(Some(PyNode::new(node_id, labels, properties)))
        } else {
            Ok(None)
        }
    }

    /// Get an edge at a specific historical epoch.
    ///
    /// Returns the edge as it existed at the given epoch, including
    /// properties at that point in time.
    ///
    /// Returns None if the edge didn't exist at that epoch.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_edge_at_epoch(&self, id: u64, epoch: u64) -> PyResult<Option<PyEdge>> {
        let db = self.inner.read();
        let edge_id = EdgeId(id);
        let epoch_id = grafeo_common::types::EpochId::new(epoch);

        if let Some(edge) = db.get_edge_at_epoch(edge_id, epoch_id) {
            let properties: HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            > = edge.properties.into_iter().collect();
            Ok(Some(PyEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )))
        } else {
            Ok(None)
        }
    }

    /// Whole-state as-of scrub: node frames plus edge frames at ``epoch``.
    ///
    /// Returns
    /// ``{"nodes": [node_frame, ...], "edges": [edge_frame, ...]}``.
    ///
    /// Each node frame is
    /// ``{"label": str, "node_ids": list[int], "columns": {prop: [value | None, ...]}}``.
    /// Each edge frame is
    /// ``{"edge_type": str, "edge_ids": list[int], "src_ids": list[int],
    /// "dst_ids": list[int], "columns": {prop: [value | None, ...]}}``.
    /// Columns are aligned to the id list by index (``None`` means the
    /// property was absent at ``epoch``).
    ///
    /// ``epoch == 2**64 - 1`` (``EpochId::PENDING``) is the current snapshot
    /// (derived open CSR). Requires a compacted database (after
    /// :meth:`compact`); returns empty ``nodes`` / ``edges`` lists when
    /// nothing has been compacted yet.
    #[cfg(feature = "compact-store")]
    fn scrub_at_epoch(&self, py: Python<'_>, epoch: u64) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let epoch_id = grafeo_common::types::EpochId::new(epoch);
        let scrub = db.scrub_at_epoch(epoch_id);

        let nodes = pyo3::types::PyList::empty(py);
        for frame in &scrub.nodes {
            let frame_dict = pyo3::types::PyDict::new(py);
            frame_dict.set_item("label", frame.label.as_str())?;

            let node_ids: Vec<u64> = frame.node_ids.iter().map(|n| n.as_u64()).collect();
            frame_dict.set_item("node_ids", node_ids)?;

            let columns = pyo3::types::PyDict::new(py);
            for (key, values) in &frame.columns {
                let col = pyo3::types::PyList::empty(py);
                for v in values {
                    let py_val = v
                        .as_ref()
                        .map_or_else(|| Ok(py.None()), |val| PyValue::to_py(val, py))?;
                    col.append(py_val)?;
                }
                columns.set_item(key.as_str(), col)?;
            }
            frame_dict.set_item("columns", columns)?;
            nodes.append(frame_dict)?;
        }

        let edges = pyo3::types::PyList::empty(py);
        for frame in &scrub.edges {
            let frame_dict = pyo3::types::PyDict::new(py);
            frame_dict.set_item("edge_type", frame.edge_type.as_str())?;
            let edge_ids: Vec<u64> = frame.edge_ids.iter().map(|e| e.as_u64()).collect();
            let src_ids: Vec<u64> = frame.src_ids.iter().map(|n| n.as_u64()).collect();
            let dst_ids: Vec<u64> = frame.dst_ids.iter().map(|n| n.as_u64()).collect();
            frame_dict.set_item("edge_ids", edge_ids)?;
            frame_dict.set_item("src_ids", src_ids)?;
            frame_dict.set_item("dst_ids", dst_ids)?;

            let columns = pyo3::types::PyDict::new(py);
            for (key, values) in &frame.columns {
                let col = pyo3::types::PyList::empty(py);
                for v in values {
                    let py_val = v
                        .as_ref()
                        .map_or_else(|| Ok(py.None()), |val| PyValue::to_py(val, py))?;
                    col.append(py_val)?;
                }
                columns.set_item(key.as_str(), col)?;
            }
            frame_dict.set_item("columns", columns)?;
            edges.append(frame_dict)?;
        }

        let out = pyo3::types::PyDict::new(py);
        out.set_item("nodes", nodes)?;
        out.set_item("edges", edges)?;
        Ok(out.unbind().into_any())
    }

    /// Neighbors of ``node_id`` visible at ``epoch``.
    ///
    /// ``epoch is None`` (or ``2**64 - 1``) is the current 1-hop
    /// (``EpochId::PENDING`` == derived open CSR). ``direction`` is
    /// ``"outgoing"`` (default), ``"incoming"``, or ``"both"``.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (node_id, epoch=None, direction="outgoing"))]
    fn neighbors_at_epoch(
        &self,
        node_id: u64,
        epoch: Option<u64>,
        direction: &str,
    ) -> PyResult<Vec<u64>> {
        let dir = parse_asof_direction(direction)?;
        let epoch_id = epoch.map_or(grafeo_common::types::EpochId::PENDING, |e| {
            grafeo_common::types::EpochId::new(e)
        });
        let db = self.inner.read();
        Ok(db
            .neighbors_at_epoch(NodeId(node_id), dir, epoch_id)
            .into_iter()
            .map(|n| n.as_u64())
            .collect())
    }

    /// Every edge visible at ``epoch`` (``None`` / ``2**64 - 1`` = current).
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (epoch=None))]
    fn edges_at_epoch(&self, epoch: Option<u64>) -> PyResult<Vec<PyEdge>> {
        let epoch_id = epoch.map_or(grafeo_common::types::EpochId::PENDING, |e| {
            grafeo_common::types::EpochId::new(e)
        });
        let db = self.inner.read();
        Ok(db
            .edges_at_epoch(epoch_id)
            .into_iter()
            .map(|edge| {
                let properties: HashMap<
                    grafeo_common::types::PropertyKey,
                    grafeo_common::types::Value,
                > = edge.properties.into_iter().collect();
                PyEdge::new(
                    edge.id,
                    edge.edge_type.to_string(),
                    edge.src,
                    edge.dst,
                    properties,
                )
            })
            .collect())
    }

    /// Every node visible at ``epoch`` (``None`` / ``2**64 - 1`` = current).
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (epoch=None))]
    fn nodes_at_epoch(&self, epoch: Option<u64>) -> PyResult<Vec<PyNode>> {
        let epoch_id = epoch.map_or(grafeo_common::types::EpochId::PENDING, |e| {
            grafeo_common::types::EpochId::new(e)
        });
        let db = self.inner.read();
        Ok(db
            .nodes_at_epoch(epoch_id)
            .into_iter()
            .map(|node| {
                let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
                let properties: HashMap<
                    grafeo_common::types::PropertyKey,
                    grafeo_common::types::Value,
                > = node.properties.into_iter().collect();
                PyNode::new(node.id, labels, properties)
            })
            .collect())
    }

    /// Get the version history of a node.
    ///
    /// Returns a list of (created_epoch, deleted_epoch, node) tuples
    /// representing each version of the node. When the `temporal` feature
    /// is enabled, each version includes the correct properties at that epoch.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_node_history(&self, id: u64) -> PyResult<Vec<(u64, Option<u64>, PyNode)>> {
        let db = self.inner.read();
        let node_id = NodeId(id);

        let history = db.get_node_history(node_id);
        let mut result = Vec::with_capacity(history.len());
        for (created, deleted, node) in history {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            > = node.properties.into_iter().collect();
            result.push((
                created.as_u64(),
                deleted.map(|d| d.as_u64()),
                PyNode::new(node_id, labels, properties),
            ));
        }
        Ok(result)
    }

    /// Get the version history of an edge.
    ///
    /// Returns a list of (created_epoch, deleted_epoch, edge) tuples.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_edge_history(&self, id: u64) -> PyResult<Vec<(u64, Option<u64>, PyEdge)>> {
        let db = self.inner.read();
        let edge_id = EdgeId(id);

        let history = db.get_edge_history(edge_id);
        let mut result = Vec::with_capacity(history.len());
        for (created, deleted, edge) in history {
            let properties: HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            > = edge.properties.into_iter().collect();
            result.push((
                created.as_u64(),
                deleted.map(|d| d.as_u64()),
                PyEdge::new(
                    edge_id,
                    edge.edge_type.to_string(),
                    edge.src,
                    edge.dst,
                    properties,
                ),
            ));
        }
        Ok(result)
    }

    /// Get a property value at a specific historical epoch.
    ///
    /// Returns the property value as it existed at the given epoch,
    /// or None if the property didn't exist at that epoch.
    ///
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_node_property_at_epoch(
        &self,
        id: u64,
        key: &str,
        epoch: u64,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let db = self.inner.read();
        let node_id = NodeId(id);
        let epoch_id = grafeo_common::types::EpochId::new(epoch);
        db.get_node_property_at_epoch(node_id, key, epoch_id)
            .map(|v| PyValue::to_py(&v, py))
            .transpose()
    }

    /// Get the full version history for a specific property of a node.
    ///
    /// Returns a list of (epoch, value) tuples in ascending epoch order.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_node_property_history(
        &self,
        id: u64,
        key: &str,
        py: Python<'_>,
    ) -> PyResult<Vec<(u64, Py<PyAny>)>> {
        let db = self.inner.read();
        let node_id = NodeId(id);
        let history = db.get_node_property_history(node_id, key);
        history
            .into_iter()
            .map(|(epoch, value)| Ok((epoch.as_u64(), PyValue::to_py(&value, py)?)))
            .collect()
    }

    /// Get the full version history for ALL properties of a node.
    ///
    /// Returns a dict mapping property names to lists of (epoch, value) tuples.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_all_node_property_history(
        &self,
        id: u64,
        py: Python<'_>,
    ) -> PyResult<HashMap<String, Vec<(u64, Py<PyAny>)>>> {
        let db = self.inner.read();
        let node_id = NodeId(id);
        let history = db.get_all_node_property_history(node_id);
        let mut result = HashMap::new();
        for (key, entries) in history {
            let py_entries: Vec<(u64, Py<PyAny>)> = entries
                .into_iter()
                .map(|(epoch, value)| Ok((epoch.as_u64(), PyValue::to_py(&value, py)?)))
                .collect::<PyResult<_>>()?;
            result.insert(key.to_string(), py_entries);
        }
        Ok(result)
    }

    /// Returns the current epoch of the database.
    ///
    /// Transactions and standalone metadata publications, including catalog
    /// and projection declarations, share this ordered identity space. Failed
    /// durable publications may leave intentional gaps, so the value is not a
    /// transaction count.
    fn current_epoch(&self) -> u64 {
        self.inner.read().current_epoch().as_u64()
    }

    /// Get all nodes with a specific label and their properties.
    ///
    /// This is more efficient than calling `get_node()` in a loop because it
    /// batches the property lookups.
    ///
    /// Example:
    /// ```python
    /// # Get all Person nodes with properties
    /// people = db.get_nodes_by_label("Person", limit=100)
    /// for node_id, props in people:
    ///     print(f"Node {node_id}: {props}")
    ///
    /// # Pagination example
    /// page_size = 100
    /// for page in range(10):
    ///     nodes = db.get_nodes_by_label("Person", limit=page_size, offset=page * page_size)
    ///     for node_id, props in nodes:
    ///         process(node_id, props)
    /// ```
    ///
    /// Args:
    ///     label: The label to filter by
    ///     limit: Maximum number of nodes to return (None for all)
    ///     offset: Number of nodes to skip before returning results (default 0)
    ///
    /// Returns:
    ///     List of (node_id, properties_dict) tuples
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (label, limit=None, offset=0))]
    fn get_nodes_by_label(
        &self,
        py: Python<'_>,
        label: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> PyResult<Vec<(u64, Py<pyo3::types::PyDict>)>> {
        let db = self.inner.read();
        let session = db.session();
        let _snapshot = session.snapshot().map_err(PyGrafeoError::from)?;
        let store = db.graph_store();

        // Get node IDs by label
        let all_node_ids = store.nodes_by_label(label);

        // Apply offset
        let node_ids = if offset >= all_node_ids.len() {
            &[][..]
        } else {
            &all_node_ids[offset..]
        };

        // Apply limit
        let node_ids = match limit {
            Some(n) => &node_ids[..n.min(node_ids.len())],
            None => node_ids,
        };

        // Batch get all properties
        let props_batch = store.get_nodes_properties_batch(node_ids);

        // Convert to Python
        let mut results = Vec::with_capacity(node_ids.len());
        for (node_id, props) in node_ids.iter().zip(props_batch) {
            let py_dict = pyo3::types::PyDict::new(py);
            for (key, value) in props {
                py_dict.set_item(key.as_str(), PyValue::to_py(&value, py)?)?;
            }
            results.push((node_id.0, py_dict.into()));
        }

        Ok(results)
    }

    /// Get a specific property for multiple nodes at once.
    ///
    /// More efficient than calling `get_node()` in a loop when you only need
    /// one property.
    ///
    /// Example:
    /// ```python
    /// # Get ages for a list of node IDs
    /// node_ids = [1, 2, 3, 4, 5]
    /// ages = db.get_property_batch(node_ids, "age")
    /// for node_id, age in zip(node_ids, ages):
    ///     if age is not None:
    ///         print(f"Node {node_id} is {age} years old")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_property_batch(
        &self,
        py: Python<'_>,
        node_ids: Vec<u64>,
        property: &str,
    ) -> PyResult<Vec<Option<Py<pyo3::prelude::PyAny>>>> {
        let db = self.inner.read();
        let session = db.session();
        let _snapshot = session.snapshot().map_err(PyGrafeoError::from)?;
        let store = db.graph_store();
        let ids: Vec<NodeId> = node_ids.into_iter().map(NodeId).collect();
        let key = grafeo_common::types::PropertyKey::new(property);
        let values = store.get_node_property_batch(&ids, &key);

        values
            .into_iter()
            .map(|opt| opt.map(|v| PyValue::to_py(&v, py)).transpose())
            .collect()
    }

    /// Delete a node by ID.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn delete_node(&self, id: u64) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.delete_node(NodeId(id)))
    }

    /// Delete an edge by ID.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn delete_edge(&self, id: u64) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.delete_edge(EdgeId(id)))
    }

    /// Set a property on a node.
    /// Raises GrafeoError if the node is missing or the write is rejected.
    ///
    /// Example:
    /// ```python
    /// db.set_node_property(node_id, "name", "Alix")
    /// db.set_node_property(node_id, "age", 30)
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn set_node_property(
        &self,
        node_id: u64,
        key: &str,
        value: &Bound<'_, pyo3::prelude::PyAny>,
    ) -> PyResult<()> {
        let db = self.inner.read();
        let val = PyValue::from_py(value)?;
        db.set_node_property(NodeId(node_id), key, val)
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Add a label to an existing node.
    ///
    /// Returns True if the label was added, False if the node doesn't exist
    /// or already has the label.
    ///
    /// Example:
    /// ```python
    /// alix = db.create_node(["Person"], {"name": "Alix"})
    /// db.add_node_label(alix.id, "Employee")  # Now has Person and Employee
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn add_node_label(&self, node_id: u64, label: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.add_node_label(NodeId(node_id), label))
    }

    /// Remove a label from a node.
    ///
    /// Returns True if the label was removed, False if the node doesn't exist
    /// or doesn't have the label.
    ///
    /// Example:
    /// ```python
    /// db.remove_node_label(alix.id, "Contractor")  # Remove Contractor label
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn remove_node_label(&self, node_id: u64, label: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.remove_node_label(NodeId(node_id), label))
    }

    /// Get all labels for a node.
    ///
    /// Returns a list of label names, or None if the node doesn't exist.
    ///
    /// Example:
    /// ```python
    /// labels = db.get_node_labels(alix.id)
    /// if labels:
    ///     print(f"Alix has labels: {labels}")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn get_node_labels(&self, node_id: u64) -> PyResult<Option<Vec<String>>> {
        let db = self.inner.read();
        Ok(db.get_node_labels(NodeId(node_id)))
    }

    /// Set a property on an edge.
    /// Raises GrafeoError if the edge is missing or the write is rejected.
    ///
    /// Example:
    /// ```python
    /// db.set_edge_property(edge_id, "weight", 1.5)
    /// db.set_edge_property(edge_id, "since", "2024-01-01")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn set_edge_property(
        &self,
        edge_id: u64,
        key: &str,
        value: &Bound<'_, pyo3::prelude::PyAny>,
    ) -> PyResult<()> {
        let db = self.inner.read();
        let val = PyValue::from_py(value)?;
        db.set_edge_property(EdgeId(edge_id), key, val)
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Remove a property from a node.
    ///
    /// Returns True if the property existed and was removed, False otherwise.
    ///
    /// Example:
    /// ```python
    /// if db.remove_node_property(node_id, "deprecated_field"):
    ///     print("Property removed")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn remove_node_property(&self, node_id: u64, key: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.remove_node_property(NodeId(node_id), key))
    }

    /// Remove a property from an edge.
    ///
    /// Returns True if the property existed and was removed, False otherwise.
    ///
    /// Example:
    /// ```python
    /// if db.remove_edge_property(edge_id, "temporary"):
    ///     print("Property removed")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn remove_edge_property(&self, edge_id: u64, key: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.remove_edge_property(EdgeId(edge_id), key))
    }

    // =========================================================================
    // PROPERTY INDEX API
    // =========================================================================

    /// Create a graph-qualified index and return its committed owner ID.
    ///
    /// `kind` is "property", "btree", "text", or "vector". Text/vector
    /// require `label`; property/btree must omit it. `graph` is an array
    /// of path components: [] is root, [""] is an empty child, and ["a/b"]
    /// is distinct from ["a", "b"]. Omitted names receive a checked name.
    /// Vector-only keywords: dimensions, metric, m, ef, ef_construction,
    /// quantization. Unsupported features and invalid options raise errors.
    ///
    /// Example:
    ///     owner = db.create_index("embedding", kind="vector", label="Doc", dimensions=3)
    ///     db.rebuild_index(owner)
    ///     db.drop_index(owner)
    ///     text = db.create_index("body", kind="text", label="Doc", min_token_length=3)
    /// Text min_token_length defaults to 2; explicit zero is valid.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (property, *, kind="property", graph=None, name=None, label=None, **options))]
    fn create_index(
        &self,
        property: &str,
        kind: &str,
        graph: Option<Vec<String>>,
        name: Option<String>,
        label: Option<String>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<u32> {
        use grafeo_common::types::GraphPath;
        use grafeo_engine::{CreateIndexRequest, IndexCreateKind};
        use pyo3::exceptions::PyValueError;

        if let Some(options) = options {
            for (key, value) in options.iter() {
                let key: String = key.extract()?;
                if !matches!(
                    key.as_str(),
                    "dimensions"
                        | "metric"
                        | "m"
                        | "ef"
                        | "ef_construction"
                        | "quantization"
                        | "min_token_length"
                ) {
                    return Err(PyValueError::new_err(format!(
                        "unknown index option '{key}'"
                    )));
                }
                if !value.is_none() {
                    if key == "min_token_length" && kind != "text" {
                        return Err(PyValueError::new_err(
                            "min_token_length requires kind='text'",
                        ));
                    }
                    if key != "min_token_length" && kind != "vector" {
                        return Err(PyValueError::new_err(
                            "vector options require kind='vector'",
                        ));
                    }
                }
            }
        }
        let get = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            match options {
                Some(options) => Ok(options.get_item(key)?.filter(|value| !value.is_none())),
                None => Ok(None),
            }
        };
        let kind = match kind {
            "property" => IndexCreateKind::Property,
            "btree" => IndexCreateKind::BTree,
            "text" => IndexCreateKind::Text {
                min_token_length: get("min_token_length")?
                    .map(|value| {
                        if value.is_instance_of::<pyo3::types::PyBool>() {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "min_token_length must be a nonnegative integer, not bool",
                            ));
                        }
                        value.extract::<usize>()
                    })
                    .transpose()?,
            },
            "vector" => IndexCreateKind::Vector {
                dimensions: get("dimensions")?
                    .map(|value| value.extract::<usize>())
                    .transpose()?,
                metric: get("metric")?
                    .map(|value| value.extract::<String>())
                    .transpose()?,
                m: get("m")?
                    .map(|value| value.extract::<usize>())
                    .transpose()?,
                ef: get("ef")?
                    .map(|value| {
                        if value.is_instance_of::<pyo3::types::PyBool>() {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "ef must be a positive integer, not bool",
                            ));
                        }
                        let ef = value.extract::<usize>()?;
                        if ef == 0 {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "ef must be a positive integer",
                            ));
                        }
                        Ok(ef)
                    })
                    .transpose()?,
                ef_construction: get("ef_construction")?
                    .map(|value| value.extract::<usize>())
                    .transpose()?,
                quantization: get("quantization")?
                    .map(|value| value.extract::<String>())
                    .transpose()?,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown index kind '{other}'"
                )));
            }
        };
        let components: Vec<&str> = graph
            .as_deref()
            .map_or(&[][..], |path| path)
            .iter()
            .map(String::as_str)
            .collect();
        let graph = GraphPath::from_components(&components)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .read()
            .create_index(CreateIndexRequest {
                graph,
                name,
                label,
                property: property.to_owned(),
                kind,
            })
            .map(|owner| owner.as_u32())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Drop an index by owner ID; return False only when that owner is absent.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn drop_index(&self, owner: u32) -> PyResult<bool> {
        self.inner
            .read()
            .drop_index(grafeo_common::types::IndexId::new(owner))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Atomically rebuild an existing owner, preserving its ID and configuration.
    ///
    /// A missing owner raises RuntimeError; rebuild never creates a new index.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn rebuild_index(&self, owner: u32) -> PyResult<()> {
        self.inner
            .read()
            .rebuild_index(grafeo_common::types::IndexId::new(owner))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Uses the HNSW index created by create_index(kind="vector").
    ///
    /// Args:
    ///     label: Node label that was indexed
    ///     property: Property that was indexed
    ///     query: Query vector (list of floats)
    ///     k: Number of nearest neighbors to return
    ///     ef: Search beam width (higher = better recall, slower). Uses index default if None.
    ///
    /// Returns:
    ///     List of (node_id, distance) tuples, sorted by distance ascending
    ///     (lower distance = more similar). The distance scale depends on
    ///     the metric configured at index creation: cosine [0, 2],
    ///     euclidean [0, inf), dot_product (negated), manhattan [0, inf).
    ///
    /// Example:
    ///     results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=10, ef=200)
    ///     for node_id, distance in results:
    ///         print(f"Node {node_id}: distance={distance:.4f}")
    ///
    ///     # With property filters (only search among user_id=42 nodes):
    ///     results = db.vector_search("Doc", "embedding", query, k=10, filters={"user_id": 42})
    #[cfg(feature = "vector-index")]
    #[pyo3(signature = (label, property, query, k, ef=None, filters=None))]
    fn vector_search(
        &self,
        label: &str,
        property: &str,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
        filters: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let filter_map = Self::convert_filters(filters)?;
        let db = self.inner.read();
        let results = db
            .vector_search(label, property, &query, k, ef, filter_map.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|(id, dist)| (id.as_u64(), dist))
            .collect())
    }

    /// Bulk-insert nodes with vector properties.
    ///
    /// Creates N nodes all with the same label, each with a single vector
    /// property. Much faster than N individual create_node() calls.
    ///
    /// Args:
    ///     label: Node label for all nodes
    ///     property: Property name for the vectors
    ///     vectors: List of vectors (list of list of floats)
    ///
    /// Returns:
    ///     List of created node IDs.
    ///
    /// Example:
    ///     ids = db.batch_create_nodes("Doc", "embedding", [[1.0, 0.0], [0.0, 1.0]])
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (label, property, vectors))]
    fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> PyResult<Vec<u64>> {
        let db = self.inner.read();
        let ids = db.batch_create_nodes(label, property, vectors);
        Ok(ids.into_iter().map(|id| id.as_u64()).collect())
    }

    /// Batch-create nodes with full property maps.
    ///
    /// Each dict in `properties_list` is a complete set of properties for one
    /// node. Vector values (list of floats) are automatically inserted into
    /// matching vector indexes.
    ///
    /// Args:
    ///     label: Node label for all created nodes.
    ///     properties_list: List of property dicts, one per node.
    ///
    /// Returns:
    ///     List of created node IDs.
    ///
    /// Example:
    ///     ids = db.batch_create_nodes_with_props("Memory", [
    ///         {"text": "hello", "user_id": "u1", "embedding": [0.1, 0.2]},
    ///         {"text": "world", "user_id": "u1", "embedding": [0.3, 0.4]},
    ///     ])
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (label, properties_list))]
    fn batch_create_nodes_with_props(
        &self,
        label: &str,
        properties_list: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<u64>> {
        let db = self.inner.read();
        let mut props_vec = Vec::with_capacity(properties_list.len());
        for item in properties_list.iter() {
            let py_dict: &Bound<'_, pyo3::types::PyDict> = item.cast()?;
            let mut props = std::collections::HashMap::new();
            for (key, value) in py_dict.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value)?;
                props.insert(grafeo_common::types::PropertyKey::new(key_str), val);
            }
            props_vec.push(props);
        }
        let ids = db.batch_create_nodes_with_props(label, props_vec);
        Ok(ids.into_iter().map(|id| id.as_u64()).collect())
    }

    /// Batch search for nearest neighbors of multiple query vectors.
    ///
    /// Executes searches in parallel using all available CPU cores.
    ///
    /// Args:
    ///     label: Node label that was indexed
    ///     property: Property that was indexed
    ///     queries: List of query vectors
    ///     k: Number of nearest neighbors per query
    ///     ef: Search beam width (higher = better recall, slower). Uses index default if None.
    ///
    /// Returns:
    ///     List of results per query. Each result is a list of (node_id, distance) tuples.
    ///
    /// Example:
    ///     results = db.batch_vector_search("Doc", "embedding", [[1.0, 0.0], [0.0, 1.0]], k=5)
    #[cfg(feature = "vector-index")]
    #[pyo3(signature = (label, property, queries, k, ef=None, filters=None))]
    fn batch_vector_search(
        &self,
        label: &str,
        property: &str,
        queries: Vec<Vec<f32>>,
        k: usize,
        ef: Option<usize>,
        filters: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Vec<Vec<(u64, f32)>>> {
        let filter_map = Self::convert_filters(filters)?;
        let db = self.inner.read();
        let results = db
            .batch_vector_search(label, property, &queries, k, ef, filter_map.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|inner| {
                inner
                    .into_iter()
                    .map(|(id, dist)| (id.as_u64(), dist))
                    .collect()
            })
            .collect())
    }

    /// Search for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
    ///
    /// MMR balances relevance to the query with diversity among results,
    /// avoiding redundant results in RAG pipelines.
    ///
    /// Args:
    ///     label: Node label that was indexed
    ///     property: Property that was indexed
    ///     query: Query vector (list of floats)
    ///     k: Number of diverse results to return
    ///     fetch_k: Initial candidates from HNSW (default: 4*k)
    ///     lambda_mult: Relevance vs diversity (0=diverse, 1=relevant). Default: 0.5.
    ///     ef: Search beam width (higher = better recall, slower). Uses index default if None.
    ///
    /// Returns:
    ///     List of (node_id, distance) tuples in MMR selection order.
    ///     The distance values are identical to those returned by
    ///     vector_search() for the same nodes (lower = more similar).
    ///     The ordering reflects MMR's relevance-diversity balance,
    ///     not distance sorting.
    ///
    /// Example:
    ///     results = db.mmr_search("Doc", "embedding", [1.0, 0.0, 0.0], k=4, lambda_mult=0.5)
    ///     for node_id, distance in results:
    ///         print(f"Node {node_id}: distance={distance:.4f}")
    #[cfg(feature = "vector-index")]
    #[pyo3(signature = (label, property, query, k, fetch_k=None, lambda_mult=None, ef=None, filters=None))]
    #[allow(clippy::too_many_arguments)]
    fn mmr_search(
        &self,
        label: &str,
        property: &str,
        query: Vec<f32>,
        k: usize,
        fetch_k: Option<usize>,
        lambda_mult: Option<f32>,
        ef: Option<usize>,
        filters: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let filter_map = Self::convert_filters(filters)?;
        let db = self.inner.read();
        let results = db
            .mmr_search(
                label,
                property,
                &query,
                k,
                fetch_k,
                lambda_mult,
                ef,
                filter_map.as_ref(),
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|(id, dist)| (id.as_u64(), dist))
            .collect())
    }

    // ── Text Search ──────────────────────────────────────────────

    /// Search a text index using BM25 scoring.
    ///
    /// Returns up to k results as (node_id, score) tuples sorted by
    /// descending relevance (higher score = more relevant). BM25 scores
    /// are unbounded positive floats; compare them only within a single
    /// query's results.
    ///
    /// Args:
    ///     label: Node label that was indexed
    ///     property: Property that was indexed
    ///     query: Text query string
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     List of (node_id, score) tuples sorted by score descending.
    ///
    /// Example:
    ///     results = db.text_search("Article", "title", "graph database", k=10)
    ///     for node_id, score in results:
    ///         print(f"Node {node_id}: score={score:.4f}")
    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> PyResult<Vec<(u64, f64)>> {
        let db = self.inner.read();
        let results = db
            .text_search(label, property, query, k)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|(id, score)| (id.as_u64(), score))
            .collect())
    }

    /// Perform hybrid search combining text (BM25) and vector similarity.
    ///
    /// Runs both text and vector search, then fuses results using
    /// Reciprocal Rank Fusion (RRF) by default. Requires both a text
    /// index and a vector index, both created with create_index.
    /// If either index is missing, that source is silently omitted.
    ///
    /// Args:
    ///     label: Node label to search within
    ///     text_property: Property indexed for text search
    ///     vector_property: Property indexed for vector search
    ///     query_text: Text query for BM25 search
    ///     k: Number of results to return
    ///     query_vector: Vector query for similarity search (optional)
    ///     fusion: Fusion method - "rrf" (default) or "weighted"
    ///     weights: Weights for weighted fusion [text_weight, vector_weight]
    ///
    /// Returns:
    ///     List of (node_id, score) tuples sorted by fused score
    ///     descending (higher = more relevant). These are fusion scores,
    ///     NOT distances. Do not apply distance-based transformations
    ///     (like dividing by score) to these values.
    ///
    /// Example:
    ///     results = db.hybrid_search("Article", "title", "embedding",
    ///                                "graph databases", k=10,
    ///                                query_vector=[1.0, 0.0, 0.0])
    #[cfg(feature = "hybrid-search")]
    #[pyo3(signature = (label, text_property, vector_property, query_text, k, query_vector=None, fusion=None, weights=None, rrf_k=None))]
    #[allow(clippy::too_many_arguments)]
    fn hybrid_search(
        &self,
        label: &str,
        text_property: &str,
        vector_property: &str,
        query_text: &str,
        k: usize,
        query_vector: Option<Vec<f32>>,
        fusion: Option<&str>,
        weights: Option<Vec<f64>>,
        rrf_k: Option<usize>,
    ) -> PyResult<Vec<(u64, f64)>> {
        let fusion_method = match fusion {
            Some("weighted") => {
                let w = weights.unwrap_or_else(|| vec![0.5, 0.5]);
                Some(grafeo_core::index::text::FusionMethod::Weighted { weights: w })
            }
            Some("rrf") => Some(grafeo_core::index::text::FusionMethod::Rrf {
                k: rrf_k.unwrap_or(60),
            }),
            _ => rrf_k.map(|k_val| grafeo_core::index::text::FusionMethod::Rrf { k: k_val }),
        };

        let db = self.inner.read();
        let results = db
            .hybrid_search(
                label,
                text_property,
                vector_property,
                query_text,
                query_vector.as_deref(),
                k,
                fusion_method,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|(id, score)| (id.as_u64(), score))
            .collect())
    }

    // ── Embedding ─────────────────────────────────────────────────

    /// Register an ONNX embedding model for text-to-vector conversion.
    ///
    /// Once registered, use embed_text() and vector_search_text() with the model name.
    ///
    /// Args:
    ///     name: Model name for later reference (e.g., "minilm")
    ///     model_path: Path to the .onnx model file
    ///     tokenizer_path: Path to the tokenizer.json file
    ///     batch_size: Maximum batch size for embedding (default: 32)
    ///
    /// Example:
    ///     db.register_embedding_model("minilm", "model.onnx", "tokenizer.json")
    #[cfg(feature = "embed")]
    #[pyo3(signature = (name, model_path, tokenizer_path, batch_size=None))]
    fn register_embedding_model(
        &self,
        name: &str,
        model_path: &str,
        tokenizer_path: &str,
        batch_size: Option<usize>,
    ) -> PyResult<()> {
        let mut model = grafeo_engine::embedding::OnnxEmbeddingModel::from_files(
            name,
            model_path,
            tokenizer_path,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if let Some(bs) = batch_size {
            model = model.with_batch_size(bs);
        }
        let db = self.inner.read();
        db.register_embedding_model(name, std::sync::Arc::new(model));
        Ok(())
    }

    /// Generate embeddings for a list of texts using a registered model.
    ///
    /// Args:
    ///     model_name: Name of a previously registered model
    ///     texts: List of strings to embed
    ///
    /// Returns:
    ///     List of float vectors, one per input text.
    ///
    /// Example:
    ///     vectors = db.embed_text("minilm", ["hello world", "graph databases"])
    ///     assert len(vectors) == 2
    #[cfg(feature = "embed")]
    fn embed_text(&self, model_name: &str, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let db = self.inner.read();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        db.embed_text(model_name, &text_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Search a vector index using a text query, generating the embedding on-the-fly.
    ///
    /// Combines embed_text() + vector_search() in a single call.
    ///
    /// Args:
    ///     label: Node label to search within
    ///     property: Vector property name
    ///     model_name: Name of a registered embedding model
    ///     query_text: Text to embed and search for
    ///     k: Number of results to return
    ///     ef: Optional HNSW ef parameter for search quality
    ///
    /// Returns:
    ///     List of (node_id, distance) tuples.
    ///
    /// Example:
    ///     results = db.vector_search_text("Doc", "embedding", "minilm",
    ///                                     "hello world", k=10)
    #[cfg(all(feature = "embed", feature = "vector-index"))]
    #[pyo3(signature = (label, property, model_name, query_text, k, ef=None))]
    fn vector_search_text(
        &self,
        label: &str,
        property: &str,
        model_name: &str,
        query_text: &str,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let db = self.inner.read();
        let results = db
            .vector_search_text(label, property, model_name, query_text, k, ef)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|(id, dist)| (id.as_u64(), dist))
            .collect())
    }

    // ── Property Indexes ────────────────────────────────────────────

    /// Check if a property has an index.
    ///
    /// Example:
    /// ```python
    /// if not db.has_property_index("email"):
    ///     db.create_index("email")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn has_property_index(&self, property: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.has_property_index(property))
    }

    /// Find all nodes with a specific property value.
    ///
    /// If the property is indexed (via create_index), this is O(1).
    /// Otherwise it scans all nodes, which is O(n).
    ///
    /// Returns a list of node IDs.
    ///
    /// Example:
    /// ```python
    /// # Create index for fast lookups (optional but recommended)
    /// db.create_index("email")
    ///
    /// # Find nodes by property value
    /// alice_ids = db.find_nodes_by_property("email", "alix@example.com")
    /// for node_id in alice_ids:
    ///     node = db.get_node(node_id)
    ///     print(f"Found: {node}")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn find_nodes_by_property(
        &self,
        property: &str,
        value: &Bound<'_, pyo3::prelude::PyAny>,
    ) -> PyResult<Vec<u64>> {
        let db = self.inner.read();
        let val = PyValue::from_py(value)?;
        let nodes = db.find_nodes_by_property(property, &val);
        Ok(nodes.into_iter().map(|n| n.0).collect())
    }

    /// Begin a transaction.
    ///
    /// Returns a Transaction object that can be used as a context manager.
    /// The transaction provides snapshot isolation - all queries within the
    /// transaction see a consistent view of the database.
    ///
    /// Example:
    /// ```python
    /// with db.begin_transaction() as tx:
    ///     tx.execute("CREATE (n:Person {name: 'Alix'})")
    ///     tx.execute("CREATE (n:Person {name: 'Gus'})")
    ///     tx.commit()  # Both nodes created atomically
    ///
    /// # With explicit isolation level
    /// with db.begin_transaction("serializable") as tx:
    ///     tx.execute("MATCH (n:Counter) SET n.val = n.val + 1")
    ///     tx.commit()
    /// ```
    #[pyo3(signature = (isolation_level=None))]
    fn begin_transaction(
        &self,
        isolation_level: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTransaction> {
        let level_str = extract_isolation_level(isolation_level)?;
        PyTransaction::new(self.inner.clone(), level_str.as_deref(), None)
    }

    /// Begin a transaction with an explicit CDC override.
    ///
    /// When ``cdc_enabled`` is ``True``, mutations in this transaction are
    /// tracked regardless of the database default. ``False`` disables tracking
    /// for this transaction only.
    ///
    /// Example:
    /// ```python
    /// with db.begin_transaction_with_cdc(True) as tx:
    ///     tx.execute("INSERT (:Person {name: 'Alix'})")
    ///     tx.commit()
    /// # Read this transaction's changes with bounded node_history_after() pages
    /// ```
    #[cfg(feature = "cdc")]
    #[pyo3(signature = (cdc_enabled, isolation_level=None))]
    fn begin_transaction_with_cdc(
        &self,
        cdc_enabled: bool,
        isolation_level: Option<&str>,
    ) -> PyResult<PyTransaction> {
        PyTransaction::new(self.inner.clone(), isolation_level, Some(cdc_enabled))
    }

    /// Trigger manual garbage collection of old MVCC versions.
    ///
    /// Normally GC runs automatically after a configurable number of commits.
    /// Call this to force an immediate GC pass, freeing memory from old
    /// transaction snapshots that are no longer needed.
    fn gc(&self) -> PyResult<()> {
        let db = self.inner.read();
        db.gc().map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Get database statistics.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn stats(&self) -> PyResult<PyDatabaseStats> {
        let db = self.inner.read();
        Ok(PyDatabaseStats {
            node_count: db.node_count() as u64,
            edge_count: db.edge_count() as u64,
            label_count: db.label_count() as u64,
            property_count: db.property_key_count() as u64,
        })
    }

    // =========================================================================
    // ADMIN API
    // =========================================================================

    /// Returns high-level database information.
    ///
    /// Returns:
    ///     dict with keys: mode, node_count, edge_count, is_persistent, path,
    ///     wal_enabled, version
    ///
    /// Example:
    ///     info = db.info()
    ///     print(f"Nodes: {info['node_count']}, Edges: {info['edge_count']}")
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let info = db.info();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("mode", info.mode.to_string())?;
        dict.set_item("node_count", info.node_count)?;
        dict.set_item("edge_count", info.edge_count)?;
        dict.set_item("is_persistent", info.is_persistent)?;
        dict.set_item("path", info.path.map(|p| p.to_string_lossy().to_string()))?;
        dict.set_item("wal_enabled", info.wal_enabled)?;
        dict.set_item("version", info.version)?;
        dict.set_item("features", pyo3::types::PyList::new(py, &info.features)?)?;

        Ok(dict.into())
    }

    /// Returns detailed database statistics.
    ///
    /// Returns:
    ///     dict with keys: node_count, edge_count, label_count, edge_type_count,
    ///     property_key_count, index_count, memory_bytes, disk_bytes
    ///
    /// Example:
    ///     stats = db.detailed_stats()
    ///     print(f"Memory: {stats['memory_bytes']} bytes")
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn detailed_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let stats = db.detailed_stats();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("node_count", stats.node_count)?;
        dict.set_item("edge_count", stats.edge_count)?;
        dict.set_item("label_count", stats.label_count)?;
        dict.set_item("edge_type_count", stats.edge_type_count)?;
        dict.set_item("property_key_count", stats.property_key_count)?;
        dict.set_item("index_count", stats.index_count)?;
        dict.set_item("memory_bytes", stats.memory_bytes)?;
        dict.set_item("disk_bytes", stats.disk_bytes)?;

        Ok(dict.into())
    }

    /// Returns a hierarchical memory usage breakdown.
    ///
    /// Walks all internal structures (store, indexes, MVCC chains, caches,
    /// string pools, buffer manager) and returns estimated heap bytes.
    ///
    /// Returns:
    ///     dict with keys: total_bytes, store, indexes, mvcc, caches,
    ///     string_pool, buffer_manager (each a nested dict)
    ///
    /// Example:
    ///     usage = db.memory_usage()
    ///     print(f"Total: {usage['total_bytes']} bytes")
    ///     print(f"Store: {usage['store']['total_bytes']} bytes")
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn memory_usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let usage = db.memory_usage();

        let store = pyo3::types::PyDict::new(py);
        store.set_item("total_bytes", usage.store.total_bytes)?;
        store.set_item("nodes_bytes", usage.store.nodes_bytes)?;
        store.set_item("edges_bytes", usage.store.edges_bytes)?;
        store.set_item("node_properties_bytes", usage.store.node_properties_bytes)?;
        store.set_item("edge_properties_bytes", usage.store.edge_properties_bytes)?;
        store.set_item("property_column_count", usage.store.property_column_count)?;

        let indexes = pyo3::types::PyDict::new(py);
        indexes.set_item("total_bytes", usage.indexes.total_bytes)?;
        indexes.set_item(
            "forward_adjacency_bytes",
            usage.indexes.forward_adjacency_bytes,
        )?;
        indexes.set_item(
            "backward_adjacency_bytes",
            usage.indexes.backward_adjacency_bytes,
        )?;
        indexes.set_item("label_index_bytes", usage.indexes.label_index_bytes)?;
        indexes.set_item("node_labels_bytes", usage.indexes.node_labels_bytes)?;
        indexes.set_item("property_index_bytes", usage.indexes.property_index_bytes)?;

        let vec_idxs = pyo3::types::PyList::empty(py);
        for vi in &usage.indexes.vector_indexes {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("name", &vi.name)?;
            d.set_item("bytes", vi.bytes)?;
            d.set_item("item_count", vi.item_count)?;
            vec_idxs.append(d)?;
        }
        indexes.set_item("vector_indexes", vec_idxs)?;

        let txt_idxs = pyo3::types::PyList::empty(py);
        for ti in &usage.indexes.text_indexes {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("name", &ti.name)?;
            d.set_item("bytes", ti.bytes)?;
            d.set_item("item_count", ti.item_count)?;
            txt_idxs.append(d)?;
        }
        indexes.set_item("text_indexes", txt_idxs)?;

        let mvcc = pyo3::types::PyDict::new(py);
        mvcc.set_item("total_bytes", usage.mvcc.total_bytes)?;
        mvcc.set_item(
            "node_version_chains_bytes",
            usage.mvcc.node_version_chains_bytes,
        )?;
        mvcc.set_item(
            "edge_version_chains_bytes",
            usage.mvcc.edge_version_chains_bytes,
        )?;
        mvcc.set_item("average_chain_depth", usage.mvcc.average_chain_depth)?;
        mvcc.set_item("max_chain_depth", usage.mvcc.max_chain_depth)?;

        let caches = pyo3::types::PyDict::new(py);
        caches.set_item("total_bytes", usage.caches.total_bytes)?;
        caches.set_item(
            "parsed_plan_cache_bytes",
            usage.caches.parsed_plan_cache_bytes,
        )?;
        caches.set_item(
            "optimized_plan_cache_bytes",
            usage.caches.optimized_plan_cache_bytes,
        )?;
        caches.set_item("cached_plan_count", usage.caches.cached_plan_count)?;

        let string_pool = pyo3::types::PyDict::new(py);
        string_pool.set_item("total_bytes", usage.string_pool.total_bytes)?;
        string_pool.set_item(
            "label_registry_bytes",
            usage.string_pool.label_registry_bytes,
        )?;
        string_pool.set_item(
            "edge_type_registry_bytes",
            usage.string_pool.edge_type_registry_bytes,
        )?;
        string_pool.set_item("label_count", usage.string_pool.label_count)?;
        string_pool.set_item("edge_type_count", usage.string_pool.edge_type_count)?;

        let buffer_mgr = pyo3::types::PyDict::new(py);
        buffer_mgr.set_item("budget_bytes", usage.buffer_manager.budget_bytes)?;
        buffer_mgr.set_item("allocated_bytes", usage.buffer_manager.allocated_bytes)?;
        buffer_mgr.set_item(
            "graph_storage_bytes",
            usage.buffer_manager.graph_storage_bytes,
        )?;
        buffer_mgr.set_item(
            "index_buffers_bytes",
            usage.buffer_manager.index_buffers_bytes,
        )?;
        buffer_mgr.set_item(
            "execution_buffers_bytes",
            usage.buffer_manager.execution_buffers_bytes,
        )?;
        buffer_mgr.set_item(
            "spill_staging_bytes",
            usage.buffer_manager.spill_staging_bytes,
        )?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_bytes", usage.total_bytes)?;
        dict.set_item("store", store)?;
        dict.set_item("indexes", indexes)?;
        dict.set_item("mvcc", mvcc)?;
        dict.set_item("caches", caches)?;
        dict.set_item("string_pool", string_pool)?;
        dict.set_item("buffer_manager", buffer_mgr)?;

        Ok(dict.into())
    }

    /// Returns runtime metrics as a dict.
    ///
    /// Requires the `metrics` feature to be enabled. Returns counters for
    /// queries, transactions, sessions, cache, and GC.
    ///
    /// Returns:
    ///     dict with metric names as keys and numeric values
    ///
    /// Example:
    ///     m = db.metrics()
    ///     print(f"Queries: {m['query_count']}, Cache hits: {m['cache_hits']}")
    #[cfg(feature = "metrics")]
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let snap = db.metrics();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("query_count", snap.query_count)?;
        dict.set_item("query_errors", snap.query_errors)?;
        dict.set_item("query_timeouts", snap.query_timeouts)?;
        dict.set_item("rows_returned", snap.rows_returned)?;
        dict.set_item("rows_scanned", snap.rows_scanned)?;
        dict.set_item("tx_committed", snap.tx_committed)?;
        dict.set_item("tx_rolled_back", snap.tx_rolled_back)?;
        dict.set_item("tx_conflicts", snap.tx_conflicts)?;
        dict.set_item("tx_active", snap.tx_active)?;
        dict.set_item("session_created", snap.session_created)?;
        dict.set_item("session_active", snap.session_active)?;
        dict.set_item("gc_runs", snap.gc_runs)?;
        dict.set_item("cache_hits", snap.cache_hits)?;
        dict.set_item("cache_misses", snap.cache_misses)?;
        dict.set_item("cache_size", snap.cache_size)?;
        dict.set_item("cache_invalidations", snap.cache_invalidations)?;

        Ok(dict.into())
    }

    /// Returns runtime metrics in Prometheus text exposition format.
    ///
    /// Returns:
    ///     str: Prometheus-compatible text output
    ///
    /// Example:
    ///     print(db.metrics_prometheus())
    #[cfg(feature = "metrics")]
    fn metrics_prometheus(&self) -> String {
        let db = self.inner.read();
        db.metrics_prometheus()
    }

    /// Resets all metrics counters and histograms to zero.
    ///
    /// Example:
    ///     db.reset_metrics()
    #[cfg(feature = "metrics")]
    fn reset_metrics(&self) {
        let db = self.inner.read();
        db.reset_metrics();
    }

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// Returns:
    ///     dict with keys: labels (list of dicts), edge_types (list of dicts),
    ///     property_keys (list of strings)
    ///
    /// Example:
    ///     schema = db.schema()
    ///     for label in schema['labels']:
    ///         print(f"{label['name']}: {label['count']} nodes")
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let schema = db.schema();

        let dict = pyo3::types::PyDict::new(py);

        match schema {
            grafeo_engine::SchemaInfo::Lpg(lpg) => {
                dict.set_item("mode", "lpg")?;

                let labels = pyo3::types::PyList::empty(py);
                for label in lpg.labels {
                    let label_dict = pyo3::types::PyDict::new(py);
                    label_dict.set_item("name", label.name)?;
                    label_dict.set_item("count", label.count)?;
                    labels.append(label_dict)?;
                }
                dict.set_item("labels", labels)?;

                let edge_types = pyo3::types::PyList::empty(py);
                for et in lpg.edge_types {
                    let et_dict = pyo3::types::PyDict::new(py);
                    et_dict.set_item("name", et.name)?;
                    et_dict.set_item("count", et.count)?;
                    edge_types.append(et_dict)?;
                }
                dict.set_item("edge_types", edge_types)?;

                dict.set_item("property_keys", lpg.property_keys)?;
            }
            grafeo_engine::SchemaInfo::Rdf(rdf) => {
                dict.set_item("mode", "rdf")?;

                let predicates = pyo3::types::PyList::empty(py);
                for pred in rdf.predicates {
                    let pred_dict = pyo3::types::PyDict::new(py);
                    pred_dict.set_item("iri", pred.iri)?;
                    pred_dict.set_item("count", pred.count)?;
                    predicates.append(pred_dict)?;
                }
                dict.set_item("predicates", predicates)?;
                dict.set_item("named_graphs", rdf.named_graphs)?;
                dict.set_item("subject_count", rdf.subject_count)?;
                dict.set_item("object_count", rdf.object_count)?;
            }
            _ => {
                dict.set_item("mode", "unknown")?;
            }
        }

        Ok(dict.into())
    }

    /// Validates database integrity.
    ///
    /// Returns:
    ///     list of error dicts (empty = valid). Each error has keys:
    ///     code, message, context
    ///
    /// Example:
    ///     errors = db.validate()
    ///     if not errors:
    ///         print("Database is valid")
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn validate(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let result = db.validate();

        let errors = pyo3::types::PyList::empty(py);
        for error in result.errors {
            let error_dict = pyo3::types::PyDict::new(py);
            error_dict.set_item("code", error.code)?;
            error_dict.set_item("message", error.message)?;
            error_dict.set_item("context", error.context)?;
            errors.append(error_dict)?;
        }

        Ok(errors.into())
    }

    /// Returns WAL (Write-Ahead Log) status.
    ///
    /// Returns:
    ///     dict with keys: enabled, path, size_bytes, record_count,
    ///     last_checkpoint, current_epoch
    ///
    /// Example:
    ///     wal = db.wal_status()
    ///     print(f"WAL size: {wal['size_bytes']} bytes")
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn wal_status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let status = db.wal_status().map_err(PyGrafeoError::from)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("enabled", status.enabled)?;
        dict.set_item("path", status.path.map(|p| p.to_string_lossy().to_string()))?;
        dict.set_item("size_bytes", status.size_bytes)?;
        dict.set_item("record_count", status.record_count)?;
        dict.set_item("last_checkpoint", status.last_checkpoint)?;
        dict.set_item("current_epoch", status.current_epoch)?;

        Ok(dict.into())
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage.
    ///
    /// Example:
    ///     db.wal_checkpoint()
    fn wal_checkpoint(&self) -> PyResult<()> {
        let db = self.inner.read();
        db.wal_checkpoint().map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Saves the database to a file path.
    ///
    /// - If in-memory: creates a new persistent database at path
    /// - If file-backed: creates a copy at the new path
    ///
    /// The original database remains unchanged.
    ///
    /// Example:
    ///     db = GrafeoDB()  # in-memory
    ///     db.create_node(["Person"], {"name": "Alix"})
    ///     db.save("./mydb")  # save to file
    #[cfg(any(feature = "storage", feature = "embedded", feature = "native"))]
    fn save(&self, path: String) -> PyResult<()> {
        let db = self.inner.read();
        db.save(path).map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Creates a full backup of the database.
    ///
    /// Checkpoints the database, copies the container file to the backup directory,
    /// and creates a backup manifest.
    ///
    /// Example:
    ///     db = GrafeoDB("./mydb.grafeo")
    ///     db.backup_full("./backups/mydb")
    #[cfg(all(
        any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native"
        ),
        any(feature = "storage", feature = "embedded", feature = "native")
    ))]
    fn backup_full(&self, backup_dir: String) -> PyResult<()> {
        let db = self.inner.read();
        db.backup_full(std::path::Path::new(&backup_dir))
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Creates an incremental backup (WAL records since last backup).
    ///
    /// Requires a prior full backup in the backup directory.
    ///
    /// Example:
    ///     db.backup_incremental("./backups/mydb")
    #[cfg(all(
        any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native"
        ),
        any(feature = "storage", feature = "embedded", feature = "native")
    ))]
    fn backup_incremental(&self, backup_dir: String) -> PyResult<()> {
        let db = self.inner.read();
        db.backup_incremental(std::path::Path::new(&backup_dir))
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Restores a database to a specific epoch from a backup chain.
    ///
    /// Example:
    ///     GrafeoDB.restore_to_epoch("./backups/mydb", 500, "./restored.grafeo")
    #[cfg(any(feature = "storage", feature = "native", feature = "embedded"))]
    #[staticmethod]
    fn restore_to_epoch(backup_dir: String, epoch: u64, output_path: String) -> PyResult<()> {
        grafeo_engine::GrafeoDB::restore_to_epoch(
            std::path::Path::new(&backup_dir),
            grafeo_common::types::EpochId::new(epoch),
            std::path::Path::new(&output_path),
        )
        .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Creates an in-memory copy of this database.
    ///
    /// Returns a new database that is completely independent.
    /// Changes to the copy do not affect the original.
    ///
    /// Example:
    ///     file_db = GrafeoDB("./production.db")
    ///     test_db = file_db.to_memory()  # safe copy
    ///     test_db.create_node(...)  # doesn't affect production
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "triple-store"
    ))]
    fn to_memory(&self) -> PyResult<Self> {
        let db = self.inner.read();
        let new_db = db.to_memory().map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(new_db)),
        })
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// Example:
    ///     db = GrafeoDB.open_in_memory("./mydb")
    ///     db.create_node(...)  # doesn't affect file
    #[cfg(any(feature = "storage", feature = "embedded", feature = "native"))]
    #[staticmethod]
    fn open_in_memory(path: String) -> PyResult<Self> {
        let db = grafeo_engine::GrafeoDB::open_in_memory(path).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return False.
    #[getter]
    fn is_persistent(&self) -> bool {
        let db = self.inner.read();
        db.is_persistent()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[getter]
    fn path(&self) -> Option<String> {
        let db = self.inner.read();
        db.path().map(|p| p.to_string_lossy().to_string())
    }

    /// Clear all cached query plans.
    ///
    /// Forces re-parsing and re-optimization of all queries on next execution.
    /// Called automatically after DDL operations (CREATE INDEX, DROP TYPE, etc.),
    /// but can be invoked manually after external schema changes.
    ///
    /// Example:
    ///     db.clear_plan_cache()
    fn clear_plan_cache(&self) {
        self.inner.read().clear_plan_cache();
    }

    /// Folds retained committed LPG history into a columnar base with a writable overlay.
    ///
    /// Call again to fold later overlay writes. Requires no active transactions
    /// or live Sessions; closed or durability-poisoned databases are rejected.
    /// Compaction is not a durability checkpoint or a history-retention lease.
    ///
    /// Example:
    ///     db = GrafeoDB()
    ///     db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    ///     db.compact()
    ///     db.execute("MATCH (p:Person) SET p.age = 31")
    ///     db.compact()
    ///     result = db.execute("MATCH (p:Person) RETURN p.name")
    #[cfg(feature = "compact-store")]
    fn compact(&self) -> PyResult<()> {
        let mut db = self.inner.write();
        db.compact().map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Close the database.
    fn close(&self) -> PyResult<()> {
        let db = self.inner.read();
        db.close().map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Get the algorithms interface.
    ///
    /// Returns an Algorithms object providing access to all graph algorithms.
    ///
    /// Example:
    ///     pr = db.algorithms.pagerank()
    ///     path = db.algorithms.dijkstra(1, 5)
    #[cfg(feature = "algos")]
    #[getter]
    fn algorithms(&self) -> PyAlgorithms {
        PyAlgorithms::new(self.inner.clone())
    }

    /// Get a NetworkX-compatible view of the graph.
    ///
    /// Args:
    ///     directed: Whether to treat as directed (default: True)
    ///
    /// Returns:
    ///     NetworkXAdapter that can be used with NetworkX algorithms
    ///     or converted to a NetworkX graph with to_networkx().
    ///
    /// Example:
    ///     nx_adapter = db.as_networkx()
    ///     G = nx_adapter.to_networkx()  # Convert to NetworkX graph
    ///     pr = nx_adapter.pagerank()    # Use native Grafeo algorithms
    #[cfg(feature = "algos")]
    #[pyo3(signature = (directed=true))]
    fn as_networkx(&self, directed: bool) -> PyNetworkXAdapter {
        PyNetworkXAdapter::new(self.inner.clone(), directed)
    }

    /// Get a solvOR-compatible adapter for OR-style algorithms.
    ///
    /// Returns:
    ///     SolvORAdapter providing Operations Research style algorithms.
    ///
    /// Example:
    ///     solvor = db.as_solvor()
    ///     distance, path = solvor.shortest_path(1, 5)
    ///     result = solvor.max_flow(source=1, sink=10)
    #[cfg(feature = "algos")]
    fn as_solvor(&self) -> PySolvORAdapter {
        PySolvORAdapter::new(self.inner.clone())
    }

    /// Get number of nodes.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[getter]
    fn node_count(&self) -> usize {
        let db = self.inner.read();
        db.node_count()
    }

    /// Get number of edges.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[getter]
    fn edge_count(&self) -> usize {
        let db = self.inner.read();
        db.edge_count()
    }

    // =====================================================================
    // Arrow-based bulk export (nodes)
    // =====================================================================

    /// Export all nodes as a PyArrow Table.
    ///
    /// Schema: `id` (uint64), `labels` (list\<utf8\>), plus one column per
    /// unique property key. Returns an Arrow Table directly, no per-element
    /// PyO3 crossings.
    ///
    /// Requires pyarrow (`uv add pyarrow`).
    ///
    /// Example:
    /// ```python
    /// table = db.nodes_to_arrow()
    /// # Use with DuckDB: duckdb.from_arrow(table)
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn nodes_to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pa = py.import("pyarrow").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pyarrow is required for nodes_to_arrow(). Install it with: uv add pyarrow",
            )
        })?;
        let ipc_mod = pa.getattr("ipc")?;
        let ipc_bytes = self.nodes_ipc_bytes()?;
        let py_bytes = pyo3::types::PyBytes::new(py, &ipc_bytes);
        let reader = ipc_mod.call_method1("open_stream", (py_bytes,))?;
        let table = reader.call_method0("read_all")?;
        Ok(table.unbind())
    }

    /// Export all nodes as a Polars DataFrame.
    ///
    /// Requires polars (`uv add polars`). Does not require pyarrow.
    ///
    /// Example:
    /// ```python
    /// df = db.nodes_to_polars()
    /// print(df.filter(pl.col("labels").list.contains("Person")))
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn nodes_to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pl = py.import("polars").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "polars is required for nodes_to_polars(). Install it with: uv add polars",
            )
        })?;
        let io = py.import("io")?;
        let ipc_bytes = self.nodes_ipc_bytes()?;
        let py_bytes = pyo3::types::PyBytes::new(py, &ipc_bytes);
        let buf = io.call_method1("BytesIO", (py_bytes,))?;
        let df = pl.call_method1("read_ipc", (buf,))?;
        Ok(df.unbind())
    }

    /// Export all nodes as a pandas DataFrame (via Arrow).
    ///
    /// Requires pandas and pyarrow (`uv add pandas pyarrow`).
    /// This is the fast path: builds an Arrow RecordBatch in Rust, serializes
    /// to IPC, then converts to pandas via pyarrow. ~10-100x faster than the
    /// element-by-element `nodes_df()` fallback at scale.
    ///
    /// Example:
    /// ```python
    /// df = db.nodes_to_pandas()
    /// print(df[df["labels"].apply(lambda l: "Person" in l)])
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn nodes_to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.nodes_to_arrow(py)?;
        let df = table.call_method0(py, "to_pandas")?;
        Ok(df)
    }

    /// Export all nodes as a pandas DataFrame.
    ///
    /// Columns: `id` (int), `labels` (list[str]), plus one column per unique
    /// property key found across all nodes. Missing properties are `None`.
    ///
    /// Requires pandas (`uv add pandas`).
    ///
    /// Example:
    /// ```python
    /// df = db.nodes_df()
    /// print(df[df["labels"].apply(lambda l: "Person" in l)])
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = ())]
    fn nodes_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Fast path: use Arrow IPC when pyarrow is available
        #[cfg(feature = "arrow-export")]
        if py.import("pyarrow").is_ok() {
            return self.nodes_to_pandas(py);
        }

        // Slow fallback: element-by-element via PyO3
        let pd = py.import("pandas").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pandas is required for nodes_df(). Install it with: uv add pandas",
            )
        })?;

        let db = self.inner.read();
        let session = db.session();
        let _snapshot = session.snapshot().map_err(PyGrafeoError::from)?;

        // Collect all nodes and discover property keys.
        // Skip properties whose names collide with structural columns
        // to prevent silent overwrites (GrafeoDB/grafeo#254).
        const RESERVED_NODE_COLS: &[&str] = &["_id", "_labels"];
        let nodes: Vec<_> = db.iter_nodes().collect();
        let mut prop_keys: Vec<String> = Vec::new();
        let mut prop_key_set = std::collections::HashSet::new();
        for node in &nodes {
            for (key, _) in node.properties.iter() {
                let key_str = key.as_str().to_owned();
                if prop_key_set.insert(key_str.clone())
                    && !RESERVED_NODE_COLS.contains(&key_str.as_str())
                {
                    prop_keys.push(key_str);
                }
            }
        }

        // Build column-oriented data
        let ids = pyo3::types::PyList::empty(py);
        let labels = pyo3::types::PyList::empty(py);
        let prop_columns: Vec<_> = prop_keys
            .iter()
            .map(|_| pyo3::types::PyList::empty(py))
            .collect();

        for node in &nodes {
            ids.append(node.id.0)?;
            let node_labels: Vec<&str> = node.labels.iter().map(|l| l.as_ref()).collect();
            labels.append(
                pyo3::types::PyList::new(py, &node_labels)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            )?;
            for (i, key) in prop_keys.iter().enumerate() {
                let prop_key = grafeo_common::types::PropertyKey::new(key.clone());
                match node.properties.get(&prop_key) {
                    Some(v) => prop_columns[i].append(PyValue::to_py(v, py)?)?,
                    None => prop_columns[i].append(py.None())?,
                }
            }
        }

        let data = pyo3::types::PyDict::new(py);
        data.set_item("_id", ids)?;
        data.set_item("_labels", labels)?;
        for (key, col) in prop_keys.iter().zip(prop_columns.iter()) {
            data.set_item(key, col)?;
        }

        let df = pd.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    // =====================================================================
    // Arrow-based bulk export (edges)
    // =====================================================================

    /// Export all edges as a PyArrow Table.
    ///
    /// Schema: `id` (uint64), `type` (utf8), `source` (uint64), `target` (uint64),
    /// plus one column per unique property key.
    ///
    /// Requires pyarrow (`uv add pyarrow`).
    ///
    /// Example:
    /// ```python
    /// table = db.edges_to_arrow()
    /// # Use with DuckDB: duckdb.from_arrow(table)
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn edges_to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pa = py.import("pyarrow").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pyarrow is required for edges_to_arrow(). Install it with: uv add pyarrow",
            )
        })?;
        let ipc_mod = pa.getattr("ipc")?;
        let ipc_bytes = self.edges_ipc_bytes()?;
        let py_bytes = pyo3::types::PyBytes::new(py, &ipc_bytes);
        let reader = ipc_mod.call_method1("open_stream", (py_bytes,))?;
        let table = reader.call_method0("read_all")?;
        Ok(table.unbind())
    }

    /// Export all edges as a Polars DataFrame.
    ///
    /// Requires polars (`uv add polars`). Does not require pyarrow.
    ///
    /// Example:
    /// ```python
    /// df = db.edges_to_polars()
    /// print(df.filter(pl.col("_type") == "KNOWS"))
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn edges_to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pl = py.import("polars").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "polars is required for edges_to_polars(). Install it with: uv add polars",
            )
        })?;
        let io = py.import("io")?;
        let ipc_bytes = self.edges_ipc_bytes()?;
        let py_bytes = pyo3::types::PyBytes::new(py, &ipc_bytes);
        let buf = io.call_method1("BytesIO", (py_bytes,))?;
        let df = pl.call_method1("read_ipc", (buf,))?;
        Ok(df.unbind())
    }

    /// Export all edges as a pandas DataFrame (via Arrow).
    ///
    /// Requires pandas and pyarrow (`uv add pandas pyarrow`).
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn edges_to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.edges_to_arrow(py)?;
        let df = table.call_method0(py, "to_pandas")?;
        Ok(df)
    }

    /// Export all edges as a pandas DataFrame.
    ///
    /// Columns: `_id` (int), `_source` (int), `_target` (int), `_type` (str),
    /// plus one column per unique property key. Missing properties are `None`.
    ///
    /// Requires pandas (`uv add pandas`).
    ///
    /// Example:
    /// ```python
    /// df = db.edges_df()
    /// print(df[df["_type"] == "KNOWS"])
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = ())]
    fn edges_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Fast path: use Arrow IPC when pyarrow is available
        #[cfg(feature = "arrow-export")]
        if py.import("pyarrow").is_ok() {
            return self.edges_to_pandas(py);
        }

        // Slow fallback: element-by-element via PyO3
        let pd = py.import("pandas").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pandas is required for edges_df(). Install it with: uv add pandas",
            )
        })?;

        let db = self.inner.read();
        let session = db.session();
        let _snapshot = session.snapshot().map_err(PyGrafeoError::from)?;

        // Collect all edges and discover property keys.
        // Skip properties whose names collide with structural columns
        // to prevent silent overwrites (GrafeoDB/grafeo#254).
        const RESERVED_EDGE_COLS: &[&str] = &["_id", "_source", "_target", "_type"];
        let edges: Vec<_> = db.iter_edges().collect();
        let mut prop_keys: Vec<String> = Vec::new();
        let mut prop_key_set = std::collections::HashSet::new();
        for edge in &edges {
            for (key, _) in edge.properties.iter() {
                let key_str = key.as_str().to_owned();
                if prop_key_set.insert(key_str.clone())
                    && !RESERVED_EDGE_COLS.contains(&key_str.as_str())
                {
                    prop_keys.push(key_str);
                }
            }
        }

        // Build column-oriented data
        let ids = pyo3::types::PyList::empty(py);
        let sources = pyo3::types::PyList::empty(py);
        let targets = pyo3::types::PyList::empty(py);
        let types = pyo3::types::PyList::empty(py);
        let prop_columns: Vec<_> = prop_keys
            .iter()
            .map(|_| pyo3::types::PyList::empty(py))
            .collect();

        for edge in &edges {
            ids.append(edge.id.0)?;
            sources.append(edge.src.0)?;
            targets.append(edge.dst.0)?;
            let edge_type: &str = edge.edge_type.as_ref();
            types.append(edge_type)?;
            for (i, key) in prop_keys.iter().enumerate() {
                let prop_key = grafeo_common::types::PropertyKey::new(key.clone());
                match edge.properties.get(&prop_key) {
                    Some(v) => prop_columns[i].append(PyValue::to_py(v, py)?)?,
                    None => prop_columns[i].append(py.None())?,
                }
            }
        }

        let data = pyo3::types::PyDict::new(py);
        data.set_item("_id", ids)?;
        data.set_item("_source", sources)?;
        data.set_item("_target", targets)?;
        data.set_item("_type", types)?;
        for (key, col) in prop_keys.iter().zip(prop_columns.iter()) {
            data.set_item(key, col)?;
        }

        let df = pd.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    /// Import nodes or edges from a pandas or polars DataFrame.
    ///
    /// **Node import** (`mode='nodes'`): each row becomes a node. The `label`
    /// parameter sets the label(s). All DataFrame columns become properties.
    ///
    /// **Edge import** (`mode='edges'`): each row becomes an edge. The
    /// `source` and `target` columns must contain integer node IDs.
    /// Remaining columns become edge properties.
    ///
    /// Requires pandas or polars (`uv add pandas` or `uv add polars`).
    ///
    /// Example:
    /// ```python
    /// import pandas as pd
    ///
    /// # Import nodes
    /// people = pd.DataFrame({"name": ["Alix", "Gus"], "age": [30, 25]})
    /// db.import_df(people, mode="nodes", label="Person")
    ///
    /// # Import edges (source/target are node IDs)
    /// edges = pd.DataFrame({"source": [0, 1], "target": [1, 0], "since": [2020, 2021]})
    /// db.import_df(edges, mode="edges", edge_type="KNOWS")
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (df, mode, *, label=None, edge_type=None, source="source", target="target"))]
    fn import_df(
        &self,
        py: Python<'_>,
        df: &Bound<'_, PyAny>,
        mode: &str,
        label: Option<Py<PyAny>>,
        edge_type: Option<&str>,
        source: &str,
        target: &str,
    ) -> PyResult<u64> {
        // Extract columns and rows from pandas or polars DataFrame
        let (columns, rows) = extract_dataframe(py, df)?;

        let db = self.inner.read();
        let mut count: u64 = 0;

        match mode {
            "nodes" => {
                // Resolve label(s)
                let labels: Vec<String> = match label {
                    Some(ref obj) => {
                        let bound = obj.bind(py);
                        if let Ok(s) = bound.extract::<String>() {
                            vec![s]
                        } else if let Ok(list) = bound.extract::<Vec<String>>() {
                            list
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "label must be a string or list of strings",
                            ));
                        }
                    }
                    None => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "label is required for mode='nodes'",
                        ));
                    }
                };
                let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

                for row in &rows {
                    let props: Vec<(
                        grafeo_common::types::PropertyKey,
                        grafeo_common::types::Value,
                    )> = columns
                        .iter()
                        .zip(row.iter())
                        .filter(|(_, v)| !v.is_null())
                        .map(|(col, val)| {
                            (
                                grafeo_common::types::PropertyKey::new(col.clone()),
                                val.clone(),
                            )
                        })
                        .collect();

                    db.create_node_with_props(&label_refs, props);
                    count += 1;
                }
            }
            "edges" => {
                let edge_type_str = edge_type.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "edge_type is required for mode='edges'",
                    )
                })?;

                let source_idx = columns.iter().position(|c| c == source).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "source column '{source}' not found in DataFrame"
                    ))
                })?;
                let target_idx = columns.iter().position(|c| c == target).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "target column '{target}' not found in DataFrame"
                    ))
                })?;

                for row in &rows {
                    let src_id = value_to_node_id(&row[source_idx], source)?;
                    let dst_id = value_to_node_id(&row[target_idx], target)?;

                    let props: Vec<(
                        grafeo_common::types::PropertyKey,
                        grafeo_common::types::Value,
                    )> = columns
                        .iter()
                        .zip(row.iter())
                        .enumerate()
                        .filter(|(i, (_, val))| {
                            *i != source_idx && *i != target_idx && !val.is_null()
                        })
                        .map(|(_, (col, val))| {
                            (
                                grafeo_common::types::PropertyKey::new(col.clone()),
                                val.clone(),
                            )
                        })
                        .collect();

                    db.create_edge_with_props(src_id, dst_id, edge_type_str, props);
                    count += 1;
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "mode must be 'nodes' or 'edges'",
                ));
            }
        }

        Ok(count)
    }

    /// Import a CSV file as graph nodes.
    ///
    /// Each row becomes a node with the given label. Column headers are used
    /// as property names. Returns the number of nodes created.
    ///
    /// Example:
    ///     count = db.import_csv("people.csv", label="Person")
    #[pyo3(signature = (path, label="Row", headers=true))]
    fn import_csv(&self, path: &str, label: &str, headers: bool) -> PyResult<u64> {
        let abs_path = std::path::Path::new(path)
            .canonicalize()
            .map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(format!("{path}: {e}")))?;
        let path_str = escape_gql_string(&abs_path.to_string_lossy().replace('\\', "/"));
        let safe_label = sanitize_gql_identifier(label);

        let header_clause = if headers { " WITH HEADERS" } else { "" };

        // Read column headers to build property mapping
        let insert_clause = if headers {
            let columns = read_csv_headers(&abs_path, ',')?;
            if columns.is_empty() {
                format!("INSERT (:{safe_label} {{}})")
            } else {
                let props = columns
                    .iter()
                    .map(|col| {
                        let safe = sanitize_gql_identifier(col);
                        format!("{safe}: row.{safe}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("INSERT (:{safe_label} {{{props}}})")
            }
        } else {
            format!("INSERT (:{safe_label} {{}})")
        };

        let query =
            format!("LOAD DATA FROM '{path_str}' FORMAT CSV{header_clause} AS row {insert_clause}");

        let db = self.inner.read();
        let session = db.session();

        let before_count = count_nodes_with_label(&session, &safe_label);

        session.execute(&query).map_err(PyGrafeoError::from)?;

        let count = count_nodes_with_label(&session, &safe_label) - before_count;

        // reason: .max(0) guarantees non-negative, so cast to u64 is safe
        #[allow(clippy::cast_sign_loss)]
        Ok(count.max(0) as u64)
    }

    /// Import a JSON Lines file as graph nodes.
    ///
    /// Each line must be a valid JSON object. Object keys become property names.
    /// Returns the number of nodes created.
    ///
    /// Example:
    ///     count = db.import_jsonl("events.jsonl", label="Event")
    #[pyo3(signature = (path, label="Row"))]
    fn import_jsonl(&self, path: &str, label: &str) -> PyResult<u64> {
        let abs_path = std::path::Path::new(path)
            .canonicalize()
            .map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(format!("{path}: {e}")))?;
        let path_str = escape_gql_string(&abs_path.to_string_lossy().replace('\\', "/"));
        let safe_label = sanitize_gql_identifier(label);

        // Read first line to discover JSON keys
        let keys = read_jsonl_keys(&abs_path)?;

        let insert_clause = if keys.is_empty() {
            format!("INSERT (:{safe_label} {{}})")
        } else {
            let props = keys
                .iter()
                .map(|key| {
                    let safe = sanitize_gql_identifier(key);
                    format!("{safe}: row.{safe}")
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("INSERT (:{safe_label} {{{props}}})")
        };

        let query = format!("LOAD DATA FROM '{path_str}' FORMAT JSONL AS row {insert_clause}");

        let db = self.inner.read();
        let session = db.session();

        let before_count = count_nodes_with_label(&session, &safe_label);

        session.execute(&query).map_err(PyGrafeoError::from)?;

        let count = count_nodes_with_label(&session, &safe_label) - before_count;

        // reason: .max(0) guarantees non-negative, so cast to u64 is safe
        #[allow(clippy::cast_sign_loss)]
        Ok(count.max(0) as u64)
    }

    fn __repr__(&self) -> String {
        "GrafeoDB()".to_string()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    // ── Change Data Capture ─────────────────────────────────────────────

    /// Enable CDC for all future sessions.
    ///
    /// Existing sessions are not affected.
    #[cfg(feature = "cdc")]
    fn enable_cdc(&self) {
        self.inner.read().set_cdc_enabled(true);
    }

    /// Disable CDC for all future sessions.
    ///
    /// Existing sessions are not affected.
    #[cfg(feature = "cdc")]
    fn disable_cdc(&self) {
        self.inner.read().set_cdc_enabled(false);
    }

    /// Returns whether CDC is currently enabled for new sessions.
    #[cfg(feature = "cdc")]
    #[getter]
    fn cdc_enabled(&self) -> bool {
        self.inner.read().is_cdc_enabled()
    }

    /// Reads an owned bounded node history page, optionally from an inclusive epoch.
    // The foreign-function boundary keeps entity, cursor, both bounds and epoch explicit.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cdc")]
    #[pyo3(signature = (node_id, cursor, max_events, max_bytes, *, since_epoch=0))]
    fn node_history_after<'py>(
        &self,
        py: Python<'py>,
        node_id: u64,
        cursor: Option<&[u8]>,
        max_events: usize,
        max_bytes: usize,
        since_epoch: u64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let cursor = cursor
            .map(grafeo_common::types::DurableCursor::from_bytes)
            .transpose()
            .map_err(PyGrafeoError::from)?;
        let mut query =
            grafeo_engine::cdc::EntityHistoryQuery::new(grafeo_common::types::NodeId::new(node_id));
        query.since_epoch = grafeo_common::types::EpochId::new(since_epoch);
        let page = self
            .inner
            .read()
            .session()
            .history_after(&query, cursor.as_ref(), max_events, max_bytes)
            .map_err(PyGrafeoError::from)?;
        change_page_to_dict(py, page)
    }

    /// Reads an owned bounded edge history page, optionally from an inclusive epoch.
    // The foreign-function boundary keeps entity, cursor, both bounds and epoch explicit.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cdc")]
    #[pyo3(signature = (edge_id, cursor, max_events, max_bytes, *, since_epoch=0))]
    fn edge_history_after<'py>(
        &self,
        py: Python<'py>,
        edge_id: u64,
        cursor: Option<&[u8]>,
        max_events: usize,
        max_bytes: usize,
        since_epoch: u64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let cursor = cursor
            .map(grafeo_common::types::DurableCursor::from_bytes)
            .transpose()
            .map_err(PyGrafeoError::from)?;
        let mut query =
            grafeo_engine::cdc::EntityHistoryQuery::new(grafeo_common::types::EdgeId::new(edge_id));
        query.since_epoch = grafeo_common::types::EpochId::new(since_epoch);
        let page = self
            .inner
            .read()
            .session()
            .history_after(&query, cursor.as_ref(), max_events, max_bytes)
            .map_err(PyGrafeoError::from)?;
        change_page_to_dict(py, page)
    }

    /// Reads an owned bounded feed page. Resume with the returned canonical bytes.
    /// An unchanged cursor marks the end; an empty filtered page may advance.
    #[cfg(feature = "cdc")]
    #[pyo3(signature = (cursor, max_events, max_bytes))]
    fn changes_after<'py>(
        &self,
        py: Python<'py>,
        cursor: Option<&[u8]>,
        max_events: usize,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let cursor = cursor
            .map(grafeo_common::types::DurableCursor::from_bytes)
            .transpose()
            .map_err(PyGrafeoError::from)?;
        let page = self
            .inner
            .read()
            .session()
            .changes_after(cursor.as_ref(), max_events, max_bytes)
            .map_err(PyGrafeoError::from)?;
        change_page_to_dict(py, page)
    }

    // -----------------------------------------------------------------
    // Schema context
    // -----------------------------------------------------------------

    /// Sets the current schema for subsequent `execute()` calls.
    ///
    /// Equivalent to running `SESSION SET SCHEMA <name>` but persists across
    /// calls. Use `reset_schema()` to clear it.
    ///
    /// Example:
    ///     db.set_schema("reporting")
    ///     result = db.execute("SHOW GRAPH TYPES")  # only sees types in 'reporting'
    fn set_schema(&self, name: String) -> PyResult<()> {
        self.inner
            .read()
            .set_current_schema(Some(&name))
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Clears the current schema context.
    ///
    /// Subsequent `execute()` calls will use the default (no-schema) namespace.
    fn reset_schema(&self) {
        let _ = self.inner.read().set_current_schema(None);
    }

    /// Returns the current schema name, or `None` if no schema is set.
    ///
    /// Example:
    ///     db.set_schema("reporting")
    ///     assert db.current_schema() == "reporting"
    fn current_schema(&self) -> Option<String> {
        self.inner.read().current_schema()
    }

    // -----------------------------------------------------------------
    // Named graph management
    // -----------------------------------------------------------------

    /// Creates a named graph. Returns ``True`` if created, ``False`` if it
    /// already exists.
    ///
    /// Example:
    ///     db.create_graph("social")
    ///     db.set_graph("social")
    ///     db.execute("INSERT (:Person {name: 'Alix'})")
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn create_graph(&self, name: &str) -> PyResult<bool> {
        Ok(self
            .inner
            .read()
            .create_graph(name)
            .map_err(PyGrafeoError::from)?)
    }

    /// Drops a named graph. Returns ``True`` if dropped, ``False`` if it did
    /// not exist. Rejected lifecycle mutations raise an exception.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn drop_graph(&self, name: &str) -> PyResult<bool> {
        Ok(self
            .inner
            .read()
            .drop_graph(name)
            .map_err(PyGrafeoError::from)?)
    }

    /// Returns a list of all named graph names.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    fn list_graphs(&self) -> Vec<String> {
        self.inner.read().list_graphs()
    }

    // -----------------------------------------------------------------
    // Graph projections
    // -----------------------------------------------------------------

    /// Creates a named graph projection. Returns ``True`` if created, ``False``
    /// if a projection with that name already exists.
    ///
    /// A projection is a read-only, filtered view of the default graph. Only
    /// nodes with matching labels and edges with matching types are visible.
    ///
    /// Args:
    ///     name: Projection name.
    ///     node_labels: Node labels to include (empty means all).
    ///     edge_types: Edge types to include (empty means all).
    ///
    /// Example:
    ///     db.create_projection("social", node_labels=["Person"], edge_types=["KNOWS"])
    #[pyo3(signature = (name, node_labels=vec![], edge_types=vec![]))]
    fn create_projection(
        &self,
        name: &str,
        node_labels: Vec<String>,
        edge_types: Vec<String>,
    ) -> bool {
        use grafeo_core::graph::ProjectionSpec;

        let mut spec = ProjectionSpec::new();
        if !node_labels.is_empty() {
            spec = spec.with_node_labels(node_labels);
        }
        if !edge_types.is_empty() {
            spec = spec.with_edge_types(edge_types);
        }
        self.inner.read().create_projection(name, spec)
    }

    /// Drops a named graph projection. Returns ``True`` if it existed, ``False``
    /// otherwise.
    fn drop_projection(&self, name: &str) -> bool {
        self.inner.read().drop_projection(name)
    }

    /// Returns a list of all projection names.
    fn list_projections(&self) -> Vec<String> {
        self.inner.read().list_projections()
    }

    /// Sets the current graph for subsequent ``execute()`` calls.
    ///
    /// Equivalent to running ``USE GRAPH <name>`` but persists across calls.
    /// Use ``reset_graph()`` to clear it.
    ///
    /// Example:
    ///     db.set_graph("social")
    ///     result = db.execute("MATCH (n) RETURN n")  # queries 'social' graph
    fn set_graph(&self, name: &str) -> PyResult<()> {
        self.inner
            .read()
            .set_current_graph(Some(name))
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Clears the current graph context.
    ///
    /// Subsequent ``execute()`` calls will use the default graph.
    fn reset_graph(&self) {
        let _ = self.inner.read().set_current_graph(None);
    }

    /// Returns the current graph name, or ``None`` if no graph is set.
    ///
    /// Example:
    ///     db.set_graph("social")
    ///     assert db.current_graph() == "social"
    fn current_graph(&self) -> Option<String> {
        self.inner.read().current_graph()
    }
}

/// Extract isolation level from either a string or an `IsolationLevel` enum value.
fn extract_isolation_level(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    match value {
        None => Ok(None),
        Some(v) => {
            // Try enum first
            if let Ok(level) = v.extract::<PyIsolationLevel>() {
                return Ok(Some(level.as_str().to_string()));
            }
            // Fall back to string
            if let Ok(s) = v.extract::<String>() {
                return Ok(Some(s));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "isolation_level must be a string or IsolationLevel enum",
            ))
        }
    }
}

/// Isolation level for transactions.
///
/// Pass to ``begin_transaction()`` to control visibility of concurrent writes:
///
/// ```python
/// from grafeo import IsolationLevel
/// tx = db.begin_transaction(IsolationLevel.SERIALIZABLE)
/// ```
///
/// String values ``"read_committed"``, ``"snapshot"``, ``"serializable"`` are also accepted.
#[pyclass(
    name = "IsolationLevel",
    from_py_object,
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyIsolationLevel {
    /// Each statement sees committed writes from other transactions.
    ReadCommitted = 0,
    /// The transaction sees a consistent snapshot taken at begin time.
    Snapshot = 1,
    /// Full serializability with conflict detection.
    Serializable = 2,
}

impl PyIsolationLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReadCommitted => "read_committed",
            Self::Snapshot => "snapshot",
            Self::Serializable => "serializable",
        }
    }
}

#[pymethods]
impl PyIsolationLevel {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __repr__(&self) -> &'static str {
        match self {
            Self::ReadCommitted => "IsolationLevel.READ_COMMITTED",
            Self::Snapshot => "IsolationLevel.SNAPSHOT",
            Self::Serializable => "IsolationLevel.SERIALIZABLE",
        }
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __str__(&self) -> &'static str {
        self.as_str()
    }
}

/// Groups multiple operations into an atomic unit.
///
/// Use as a context manager - changes are isolated until you commit, and
/// automatically rolled back if an exception occurs:
///
/// ```python
/// with db.begin_transaction() as tx:
///     tx.execute("INSERT (:Person {name: 'Alix'})")
///     tx.execute("INSERT (:Person {name: 'Gus'})")
///     tx.commit()  # Both or neither
/// ```
///
/// Other connections see a consistent snapshot while you work.
#[pyclass(name = "Transaction")]
pub struct PyTransaction {
    db: Arc<RwLock<GrafeoDB>>,
    session: parking_lot::Mutex<Option<grafeo_engine::session::Session>>,
    committed: bool,
    rolled_back: bool,
    isolation_level_name: String,
}

impl PyTransaction {
    /// Executes a query in the given language within this transaction.
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword options share one native execution owner"
    )]
    fn execute_language_impl(
        &self,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot execute on completed transaction",
            ));
        }
        let (params, options) =
            prepare_python_execution(language, params, control, max_rows, max_bytes)?;
        let conversion_limit = options.result_limits.unwrap_or_default().max_bytes;
        let mut result = py
            .detach(|| {
                let mut guard = self.session.lock();
                let session = guard.as_mut().ok_or_else(|| {
                    grafeo_common::utils::error::Error::Internal(
                        "Transaction session not available".into(),
                    )
                })?;
                session.execute_with_options(query, params, options)
            })
            .map_err(PyGrafeoError::from)?;
        let db = self.db.read();
        let (nodes, edges) = extract_entities(&result, &db);
        let columns = std::mem::take(&mut result.columns);
        let exec_time = result.execution_time_ms;
        let scanned = result.rows_scanned;
        Ok(PyQueryResult::with_metrics(
            columns,
            result.into_rows().map_err(PyGrafeoError::from)?,
            nodes,
            edges,
            exec_time,
            scanned,
        )
        .with_conversion_limit(conversion_limit))
    }

    /// Create a new transaction with an optional isolation level and CDC override.
    fn new(
        db: Arc<RwLock<GrafeoDB>>,
        isolation_level: Option<&str>,
        _cdc_override: Option<bool>,
    ) -> PyResult<Self> {
        #[cfg(any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native",
            feature = "triple-store"
        ))]
        {
            // Parse isolation level string
            let (level, level_name) = match isolation_level {
                Some("read_committed") => (
                    Some(grafeo_engine::transaction::IsolationLevel::ReadCommitted),
                    "read_committed",
                ),
                Some("serializable") => (
                    Some(grafeo_engine::transaction::IsolationLevel::Serializable),
                    "serializable",
                ),
                Some("snapshot") | None => (None, "snapshot"),
                Some(other) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown isolation level '{}'. Use 'read_committed', 'snapshot', or 'serializable'",
                        other
                    )));
                }
            };

            // Create session from db, using CDC override when available
            let mut session = {
                let db_guard = db.read();
                #[cfg(feature = "cdc")]
                {
                    match _cdc_override {
                        Some(cdc) => db_guard.session_with_cdc(cdc),
                        None => db_guard.session(),
                    }
                }
                #[cfg(not(feature = "cdc"))]
                {
                    db_guard.session()
                }
            };

            // Begin the transaction with the specified isolation level.
            if let Some(level) = level {
                session
                    .begin_transaction_with_isolation(level)
                    .map_err(PyGrafeoError::from)?;
            } else {
                session.begin_transaction().map_err(PyGrafeoError::from)?;
            }

            Ok(Self {
                db,
                session: parking_lot::Mutex::new(Some(session)),
                committed: false,
                rolled_back: false,
                isolation_level_name: level_name.to_string(),
            })
        }
        #[cfg(not(any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native",
            feature = "triple-store"
        )))]
        {
            let _ = (db, isolation_level, _cdc_override);
            Err(pyo3::exceptions::PyValueError::new_err(
                "transactions require a native LPG or RDF store feature",
            ))
        }
    }
}

#[pymethods]
impl PyTransaction {
    /// The isolation level of this transaction.
    ///
    /// Returns one of: ``"read_committed"``, ``"snapshot"``, ``"serializable"``.
    #[getter]
    fn isolation_level(&self) -> &str {
        &self.isolation_level_name
    }

    /// Commit the transaction and return the assigned epoch.
    ///
    /// Makes all changes permanent. Raises an error if the transaction is
    /// already completed or if there's a write-write conflict.
    fn commit(&mut self) -> PyResult<u64> {
        #[cfg(any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native",
            feature = "triple-store"
        ))]
        {
            if self.committed || self.rolled_back {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Transaction already completed",
                ));
            }

            let mut session_guard = self.session.lock();
            let epoch = if let Some(ref mut session) = *session_guard {
                match session.commit() {
                    Ok(epoch) => epoch.as_u64(),
                    Err(error) => {
                        // Commit validation can end the native transaction before
                        // returning its error. Do not leave that session reachable
                        // through an active Python handle (and implicit autocommit).
                        // Earlier failures can retain an active transaction, which
                        // still needs explicit rollback or context cleanup.
                        if !session.in_transaction() {
                            *session_guard = None;
                            self.rolled_back = true;
                        }
                        return Err(PyGrafeoError::from(error).into());
                    }
                }
            } else {
                0
            };
            *session_guard = None; // Drop the session
            self.committed = true;
            Ok(epoch)
        }
        #[cfg(not(any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native",
            feature = "triple-store"
        )))]
        {
            Err(pyo3::exceptions::PyValueError::new_err(
                "transactions require a native LPG or RDF store feature",
            ))
        }
    }

    /// Rollback the transaction.
    ///
    /// Discards all changes made within this transaction.
    fn rollback(&mut self) -> PyResult<()> {
        #[cfg(any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native",
            feature = "triple-store"
        ))]
        {
            if self.committed || self.rolled_back {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Transaction already completed",
                ));
            }

            let mut session_guard = self.session.lock();
            if let Some(ref mut session) = *session_guard {
                session.rollback().map_err(PyGrafeoError::from)?;
            }
            *session_guard = None; // Drop the session
            self.rolled_back = true;
            Ok(())
        }
        #[cfg(not(any(
            feature = "lpg",
            feature = "compact-store",
            feature = "embedded",
            feature = "edge",
            feature = "native",
            feature = "triple-store"
        )))]
        {
            Err(pyo3::exceptions::PyValueError::new_err(
                "transactions require a native LPG or RDF store feature",
            ))
        }
    }

    /// Create a savepoint within this transaction.
    ///
    /// Savepoints let you partially roll back a transaction without
    /// aborting all of it.
    ///
    /// Example:
    /// ```python
    /// tx.savepoint("sp1")
    /// tx.execute("INSERT (:Temp {x: 1})")
    /// tx.rollback_to_savepoint("sp1")  # undo the insert
    /// tx.commit()  # commits without the Temp node
    /// ```
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "triple-store"
    ))]
    fn savepoint(&self, name: &str) -> PyResult<()> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        session.savepoint(name).map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Roll back to a named savepoint.
    ///
    /// Undoes all writes made after the savepoint was created.
    /// The savepoint remains active and can be rolled back to again.
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "triple-store"
    ))]
    fn rollback_to_savepoint(&self, name: &str) -> PyResult<()> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        session
            .rollback_to_savepoint(name)
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Release a savepoint without rolling back.
    ///
    /// Frees resources associated with the savepoint. Changes made after the
    /// savepoint become permanent within the transaction scope.
    fn release_savepoint(&self, name: &str) -> PyResult<()> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        session
            .release_savepoint(name)
            .map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Execute a query within this transaction.
    ///
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    #[cfg(feature = "gql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("gql", query, params, control, max_rows, max_bytes, py)
    }

    /// Insert one RDF quad in this transaction.
    #[cfg(feature = "triple-store")]
    #[pyo3(signature = (subject, predicate, object, graph=None))]
    fn insert_rdf_quad(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> PyResult<usize> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let quad = parse_python_rdf_quad(subject, predicate, object, graph)?;
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        Ok(session
            .insert_rdf_quads([quad])
            .map_err(PyGrafeoError::from)?)
    }

    /// Bulk-insert RDF quads in this transaction.
    ///
    /// Each item is ``(subject, predicate, object)`` or
    /// ``(subject, predicate, object, graph)``.
    #[cfg(feature = "triple-store")]
    fn insert_rdf_quads(
        &self,
        quads: Vec<(String, String, String, Option<String>)>,
    ) -> PyResult<usize> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let parsed: Vec<grafeo_engine::Quad> = quads
            .iter()
            .map(|(s, p, o, g)| parse_python_rdf_quad(s, p, o, g.as_deref()))
            .collect::<PyResult<_>>()?;
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        Ok(session
            .insert_rdf_quads(parsed)
            .map_err(PyGrafeoError::from)?)
    }

    /// Create a node inside this transaction (parser-free LPG mutation).
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    ))]
    #[pyo3(signature = (labels, properties=None))]
    fn create_node(
        &self,
        labels: Vec<String>,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyNode> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        let id = if let Some(p) = properties {
            let mut owned: Vec<(String, grafeo_common::types::Value)> = Vec::new();
            for (key, value) in p.iter() {
                owned.push((key.extract()?, PyValue::from_py(&value)?));
            }
            session
                .create_node_with_props(
                    &label_refs,
                    owned.iter().map(|(k, v)| (k.as_str(), v.clone())),
                )
                .map_err(PyGrafeoError::from)?
        } else {
            session.create_node(&label_refs)
        };
        if !id.is_valid() {
            return Err(PyGrafeoError::database("Failed to create node").into());
        }
        let node = session
            .get_node(id)
            .ok_or_else(|| PyGrafeoError::database("Failed to create node"))?;
        let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
        let properties: HashMap<grafeo_common::types::PropertyKey, grafeo_common::types::Value> =
            node.properties.into_iter().collect();
        Ok(PyNode::new(id, labels, properties))
    }

    /// Exact typed-quad membership in this transaction (includes pending writes).
    #[cfg(feature = "triple-store")]
    #[pyo3(signature = (subject, predicate, object, graph=None))]
    fn contains_rdf_quad(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> PyResult<bool> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }
        let quad = parse_python_rdf_quad(subject, predicate, object, graph)?;
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;
        Ok(session
            .try_contains_rdf_quad(&quad)
            .map_err(PyGrafeoError::from)?)
    }

    /// Execute a Cypher query within this transaction.
    #[cfg(feature = "cypher")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_cypher(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("cypher", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE) within this transaction.
    #[cfg(feature = "sql-pgq")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_sql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("sql", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a Gremlin query within this transaction.
    ///
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    #[cfg(feature = "gremlin")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_gremlin(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("gremlin", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a GraphQL query within this transaction.
    ///
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    #[cfg(feature = "graphql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_graphql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("graphql", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a SPARQL query within this transaction.
    ///
    /// SPARQL is the W3C standard query language for RDF data.
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    ///
    /// Example:
    ///     with db.begin_transaction() as tx:
    ///         tx.execute_sparql("INSERT DATA { <http://ex.org/s> <http://ex.org/p> 'value' }")
    ///         result = tx.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    ///         tx.commit()
    #[cfg(feature = "sparql")]
    #[pyo3(signature = (query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    fn execute_sparql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl("sparql", query, params, control, max_rows, max_bytes, py)
    }

    /// Execute a query in a named language (e.g. `"graphql-rdf"`).
    #[pyo3(signature = (language, query, params=None , *, control=None, max_rows=None, max_bytes=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python keyword options share one native execution owner"
    )]
    fn execute_language(
        &self,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
        control: Option<&crate::control::PyQueryControl>,
        max_rows: Option<usize>,
        max_bytes: Option<usize>,
    ) -> PyResult<PyQueryResult> {
        self.execute_language_impl(language, query, params, control, max_rows, max_bytes, py)
    }

    /// Check if transaction is active.
    #[getter]
    fn is_active(&self) -> bool {
        !self.committed && !self.rolled_back
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if !self.committed && !self.rolled_back {
            if exc_type.is_some() {
                self.rollback()?;
            } else {
                // Auto-commit on successful exit (no exception)
                self.commit()?;
            }
        }
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let status = if self.committed {
            "committed"
        } else if self.rolled_back {
            "rolled_back"
        } else {
            "active"
        };
        format!("Transaction(status={})", status)
    }
}

/// Quick stats about your database - node count, edge count, and more.
#[pyclass(name = "DatabaseStats")]
pub struct PyDatabaseStats {
    #[pyo3(get)]
    node_count: u64,
    #[pyo3(get)]
    edge_count: u64,
    #[pyo3(get)]
    label_count: u64,
    #[pyo3(get)]
    property_count: u64,
}

#[pymethods]
impl PyDatabaseStats {
    fn __repr__(&self) -> String {
        format!(
            "DbStats(nodes={}, edges={}, labels={}, properties={})",
            self.node_count, self.edge_count, self.label_count, self.property_count
        )
    }
}

/// Pulls nodes and edges out of query results so Python can work with them.
fn extract_entities(result: &QueryResult, _db: &GrafeoDB) -> (Vec<PyNode>, Vec<PyEdge>) {
    if result.is_int64_columnar() {
        return (Vec::new(), Vec::new());
    }
    grafeo_bindings_common::entity::extract_and_map(
        result,
        |n| PyNode::new(n.id, n.labels, n.properties),
        |e| PyEdge::new(e.id, e.edge_type, e.source_id, e.target_id, e.properties),
    )
}

/// Converts a CDC ChangeEvent to a Python dict-like HashMap.
#[cfg(feature = "cdc")]
fn change_event_to_dict(
    py: pyo3::Python<'_>,
    event: &grafeo_engine::cdc::ChangeEvent,
) -> PyResult<std::collections::HashMap<String, pyo3::Py<pyo3::PyAny>>> {
    use crate::types::PyValue;
    use pyo3::conversion::IntoPyObjectExt;

    let mut map = std::collections::HashMap::new();

    // entity_id and entity_type
    map.insert(
        "entity_id".to_string(),
        event.entity_id.as_u64().into_py_any(py)?,
    );
    let entity_type = if event.entity_id.is_node() {
        "node"
    } else if event.entity_id.is_triple() {
        "triple"
    } else {
        "edge"
    };
    map.insert("entity_type".to_string(), entity_type.into_py_any(py)?);

    // kind
    let kind = match event.kind {
        grafeo_engine::cdc::ChangeKind::Create => "create",
        grafeo_engine::cdc::ChangeKind::Update => "update",
        grafeo_engine::cdc::ChangeKind::Delete => "delete",
        _ => "unknown",
    };
    map.insert("kind".to_string(), kind.into_py_any(py)?);

    map.insert(
        "graph_incarnation".into(),
        event
            .graph_incarnation
            .map(|id| id.as_u64())
            .into_py_any(py)?,
    );

    // epoch and timestamp
    map.insert("epoch".to_string(), event.epoch.0.into_py_any(py)?);
    map.insert(
        "timestamp".to_string(),
        event.timestamp.as_u64().into_py_any(py)?,
    );

    map.insert("labels".into(), event.labels.clone().into_py_any(py)?);
    map.insert("edge_type".into(), event.edge_type.clone().into_py_any(py)?);
    map.insert("src_id".into(), event.src_id.into_py_any(py)?);
    map.insert("dst_id".into(), event.dst_id.into_py_any(py)?);

    // before (Option<HashMap<String, Value>> -> dict or None)
    let before_py = match &event.before {
        Some(props) => {
            let d: std::collections::HashMap<String, pyo3::Py<pyo3::PyAny>> = props
                .iter()
                .map(|(k, v)| Ok((k.clone(), PyValue::to_py(v, py)?)))
                .collect::<PyResult<_>>()?;
            d.into_py_any(py)?
        }
        None => py.None(),
    };
    map.insert("before".to_string(), before_py);

    // after (Option<HashMap<String, Value>> -> dict or None)
    let after_py = match &event.after {
        Some(props) => {
            let d: std::collections::HashMap<String, pyo3::Py<pyo3::PyAny>> = props
                .iter()
                .map(|(k, v)| Ok((k.clone(), PyValue::to_py(v, py)?)))
                .collect::<PyResult<_>>()?;
            d.into_py_any(py)?
        }
        None => py.None(),
    };
    map.insert("after".to_string(), after_py);

    map.insert(
        "lpg_graph".into(),
        event
            .graph_path()
            .map(|path| path.components().to_vec())
            .into_py_any(py)?,
    );
    map.insert(
        "triple_graph".into(),
        event.triple_graph.clone().into_py_any(py)?,
    );
    map.insert(
        "triple_subject".into(),
        event.triple_subject.clone().into_py_any(py)?,
    );
    map.insert(
        "triple_predicate".into(),
        event.triple_predicate.clone().into_py_any(py)?,
    );
    map.insert(
        "triple_object".into(),
        event.triple_object.clone().into_py_any(py)?,
    );
    Ok(map)
}

/// Extracts column names and row data from a pandas or polars DataFrame.
///
/// Returns `(column_names, rows)` where each row is a `Vec<Value>`.
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
fn extract_dataframe(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
) -> PyResult<(Vec<String>, Vec<Vec<Value>>)> {
    let columns_attr = df.getattr("columns")?;
    let columns: Vec<String> = columns_attr
        .extract()
        .or_else(|_| columns_attr.call_method0("tolist")?.extract())?;

    let num_rows: usize = df.call_method0("__len__")?.extract()?;
    let mut rows = Vec::with_capacity(num_rows);

    // Try iterrows (pandas) or iter_rows (polars)
    let is_polars = df
        .getattr("__class__")?
        .getattr("__module__")?
        .extract::<String>()?
        .starts_with("polars");

    if is_polars {
        // polars: iter_rows(named=False) returns tuples
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("named", false)?;
        let iter = df.call_method("iter_rows", (), Some(&kwargs))?;
        for row_result in iter.try_iter()? {
            let row_tuple = row_result?;
            let mut row_values = Vec::with_capacity(columns.len());
            for i in 0..columns.len() {
                let item = row_tuple.get_item(i)?;
                let val = PyValue::from_py(&item)?;
                row_values.push(val);
            }
            rows.push(row_values);
        }
    } else {
        // pandas: use .values.tolist() for efficient bulk extraction
        let values_list = df.getattr("values")?.call_method0("tolist")?;
        for row_result in values_list.try_iter()? {
            let row_list = row_result?;
            let mut row_values = Vec::with_capacity(columns.len());
            for i in 0..columns.len() {
                let item = row_list.get_item(i)?;
                // pandas NaN/NaT -> check for NaN explicitly
                let val = if is_pandas_na(py, &item) {
                    Value::Null
                } else {
                    PyValue::from_py(&item)?
                };
                row_values.push(val);
            }
            rows.push(row_values);
        }
    }

    Ok((columns, rows))
}

/// Check if a Python value is pandas NA / NaN / NaT.
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
fn is_pandas_na(py: Python<'_>, obj: &Bound<'_, PyAny>) -> bool {
    // float NaN
    if let Ok(f) = obj.extract::<f64>()
        && f.is_nan()
    {
        return true;
    }
    // pandas.isna()
    if let Ok(pd) = py.import("pandas")
        && let Ok(result) = pd.call_method1("isna", (obj,))
        && let Ok(b) = result.extract::<bool>()
    {
        return b;
    }
    false
}

/// Read CSV headers from the first line of a file.
/// Escape single quotes in a string for embedding in a GQL string literal.
fn escape_gql_string(s: &str) -> String {
    s.replace('\'', "\\'")
}

/// Sanitize a name for use as a GQL identifier.
fn sanitize_gql_identifier(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "_col".to_string()
    } else if sanitized.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{sanitized}")
    } else {
        sanitized
    }
}

/// Count nodes with a given label.
fn count_nodes_with_label(session: &grafeo_engine::Session, label: &str) -> i64 {
    session
        .execute(&format!("MATCH (n:{label}) RETURN count(n) AS c"))
        .ok()
        .and_then(|r| r.rows().first().cloned())
        .and_then(|row| row.first().cloned())
        .and_then(|v| match v {
            grafeo_common::types::Value::Int64(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0)
}

fn read_csv_headers(path: &std::path::Path, delimiter: char) -> PyResult<Vec<String>> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| {
        pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}: {e}", path.display()))
    })?;
    let mut reader = BufReader::new(f);
    let mut header_line = String::new();
    reader.read_line(&mut header_line).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read headers: {e}"))
    })?;
    Ok(header_line
        .trim()
        .split(delimiter)
        .map(|h| h.trim().trim_matches('"').to_string())
        .filter(|h| !h.is_empty())
        .collect())
}

/// Read JSON keys from the first non-empty line of a JSONL file.
fn read_jsonl_keys(path: &std::path::Path) -> PyResult<Vec<String>> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| {
        pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}: {e}", path.display()))
    })?;
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = line.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to read JSONL file: {e}"))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed)
        {
            return Ok(obj.keys().cloned().collect());
        }
        break;
    }
    Ok(Vec::new())
}

/// Convert a Value to a NodeId, validating that it's a valid integer.
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
fn value_to_node_id(value: &Value, col_name: &str) -> PyResult<NodeId> {
    match value {
        Value::Int64(i) => {
            if *i < 0 {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "negative node ID {i} in column '{col_name}'"
                )))
            } else {
                // reason: negative values rejected above, cast is safe
                #[allow(clippy::cast_sign_loss)]
                Ok(NodeId(*i as u64))
            }
        }
        Value::Float64(f) => {
            if *f < 0.0 || f.fract() != 0.0 {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid node ID {f} in column '{col_name}' (must be a non-negative integer)"
                )))
            } else if *f >= u64::MAX as f64 {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node ID {f} in column '{col_name}' exceeds u64 range"
                )))
            } else {
                // reason: negative, fractional, and out-of-range values rejected above
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Ok(NodeId(*f as u64))
            }
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "column '{col_name}' must contain integer node IDs, got {value:?}"
        ))),
    }
}

/// Converts one owned bounded native page without collecting the entire feed.
#[cfg(feature = "cdc")]
fn change_page_to_dict(
    py: Python<'_>,
    page: grafeo_engine::cdc::ChangePage,
) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let events = page
        .events
        .iter()
        .map(|event| change_event_to_dict(py, event))
        .collect::<PyResult<Vec<_>>>()?;
    let result = pyo3::types::PyDict::new(py);
    result.set_item("events", events)?;
    result.set_item("next", pyo3::types::PyBytes::new(py, &page.next.to_bytes()))?;
    Ok(result)
}
