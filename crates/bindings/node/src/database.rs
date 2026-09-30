//! Main entry point for using Grafeo from Node.js.
//!
//! [`JsGrafeoDB`] wraps the Rust database engine and gives you a JavaScript API.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use napi::JsString;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_common::types::Value;
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use grafeo_common::types::{EdgeId, NodeId};
#[cfg(feature = "compact-store")]
use grafeo_core::graph::Direction;
use grafeo_engine::config::{Config, GraphModel};
use grafeo_engine::database::{GrafeoDB, QueryResult as EngineQueryResult};

use crate::error::NodeGrafeoError;
use crate::graph::{JsEdge, JsNode};
use crate::query::QueryResult;
use crate::transaction::Transaction;
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
use crate::types;

/// A checked graph-qualified index creation request.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
#[napi(object, object_from_js = false)]
pub struct CreateIndexRequest {
    /// Property to index.
    pub property: String,
    /// "property" (default), "btree", "text", or "vector".
    #[napi(ts_type = "'property' | 'btree' | 'text' | 'vector'")]
    pub kind: Option<String>,
    /// Graph path components; [] is root and [""] is an empty child.
    pub graph: Option<Vec<String>>,
    /// Optional owner name; generated names use the reserved engine namespace.
    pub name: Option<String>,
    /// Required for text/vector; omitted for property/btree.
    pub label: Option<String>,
    /// Text-only minimum token length; omitted defaults to 2, and 0 is valid.
    pub min_token_length: Option<f64>,
    /// Vector dimensions; omitted to infer from data.
    pub dimensions: Option<f64>,
    /// Vector distance metric.
    pub metric: Option<String>,
    /// Vector HNSW links per node.
    pub m: Option<f64>,
    /// Vector HNSW construction beam width.
    pub ef_construction: Option<f64>,
    /// Vector quantization mode.
    pub quantization: Option<String>,
}

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
impl CreateIndexRequest {
    /// Capture JavaScript values on its own thread before any lossy UTF-8
    /// conversion or asynchronous engine work. Read only the validated keys.
    fn from_js(request: Unknown<'_>) -> Result<Self> {
        if request.get_type()? != napi::ValueType::Object {
            return Err(napi::Error::from_reason("index request must be an object"));
        }
        let object = request.coerce_to_object()?;
        if object.is_array()? {
            return Err(napi::Error::from_reason(
                "index request must not be an array",
            ));
        }
        let keys = object.get_all_property_names(
            KeyCollectionMode::OwnOnly,
            KeyFilter::AllProperties,
            KeyConversion::NumbersToStrings,
        )?;
        let mut names = Vec::new();
        for position in 0..keys.get_array_length()? {
            let key: JsString = keys.get_element(position)?;
            let name = checked_index_string(key, "request key")?;
            if !matches!(
                name.as_str(),
                "property"
                    | "kind"
                    | "graph"
                    | "name"
                    | "label"
                    | "minTokenLength"
                    | "dimensions"
                    | "metric"
                    | "m"
                    | "efConstruction"
                    | "quantization"
            ) {
                return Err(napi::Error::from_reason(format!(
                    "unknown index request field '{name}'"
                )));
            }
            names.push(name);
        }
        // Inherited enumerable options must not silently select a different
        // graph or configuration than the caller's request suggests.
        let inherited = object.get_all_property_names(
            KeyCollectionMode::IncludePrototypes,
            KeyFilter::Enumerable,
            KeyConversion::NumbersToStrings,
        )?;
        for position in 0..inherited.get_array_length()? {
            let key: JsString = inherited.get_element(position)?;
            if !object.has_own_property_js(key)? {
                return Err(napi::Error::from_reason(
                    "index request fields must be own properties",
                ));
            }
        }
        let mut decoded = Self {
            property: String::new(),
            kind: None,
            graph: None,
            name: None,
            label: None,
            min_token_length: None,
            dimensions: None,
            metric: None,
            m: None,
            ef_construction: None,
            quantization: None,
        };
        let mut has_property = false;
        for name in names {
            match name.as_str() {
                "property" => {
                    decoded.property =
                        checked_index_string(object.get_named_property("property")?, "property")?;
                    has_property = true;
                }
                "kind" => decoded.kind = optional_index_string(&object, "kind")?,
                "name" => decoded.name = optional_index_string(&object, "name")?,
                "label" => decoded.label = optional_index_string(&object, "label")?,
                "minTokenLength" => {
                    let value: Unknown<'_> = object.get_named_property("minTokenLength")?;
                    if value.get_type()? != napi::ValueType::Undefined {
                        if value.get_type()? != napi::ValueType::Number {
                            return Err(napi::Error::from_reason(
                                "minTokenLength must be a number",
                            ));
                        }
                        decoded.min_token_length = Some(value.coerce_to_number()?.get_double()?);
                    }
                }
                "metric" => decoded.metric = optional_index_string(&object, "metric")?,
                "quantization" => {
                    decoded.quantization = optional_index_string(&object, "quantization")?;
                }
                "dimensions" => decoded.dimensions = object.get_named_property("dimensions")?,
                "m" => decoded.m = object.get_named_property("m")?,
                "efConstruction" => {
                    decoded.ef_construction = object.get_named_property("efConstruction")?;
                }
                "graph" => {
                    let graph: Option<Object<'_>> = object.get_named_property("graph")?;
                    if let Some(graph) = graph {
                        if !graph.is_array()? {
                            return Err(napi::Error::from_reason(
                                "index graph must be an array of string components",
                            ));
                        }
                        let mut components = Vec::new();
                        for position in 0..graph.get_array_length()? {
                            let component: JsString = graph.get_element(position)?;
                            components.push(checked_index_string(component, "graph component")?);
                        }
                        decoded.graph = Some(components);
                    }
                }
                _ => return Err(napi::Error::from_reason("unvalidated index request field")),
            }
        }
        if !has_property {
            return Err(napi::Error::from_reason("index request requires property"));
        }
        Ok(decoded)
    }

    fn into_engine(self) -> Result<grafeo_engine::CreateIndexRequest> {
        use grafeo_common::types::GraphPath;
        use grafeo_engine::IndexCreateKind;

        let kind = self.kind.as_deref().map_or("property", |kind| kind);
        if kind != "text" && self.min_token_length.is_some() {
            return Err(NodeGrafeoError::InvalidArgument(
                "minTokenLength requires kind='text'".into(),
            )
            .into());
        }
        if kind != "vector"
            && (self.dimensions.is_some()
                || self.metric.is_some()
                || self.m.is_some()
                || self.ef_construction.is_some()
                || self.quantization.is_some())
        {
            return Err(NodeGrafeoError::InvalidArgument(
                "vector options require kind='vector'".into(),
            )
            .into());
        }
        let kind = match kind {
            "property" => IndexCreateKind::Property,
            "btree" => IndexCreateKind::BTree,
            "text" => IndexCreateKind::Text {
                min_token_length: checked_index_size(self.min_token_length, "minTokenLength")?,
            },
            "vector" => IndexCreateKind::Vector {
                dimensions: checked_index_size(self.dimensions, "dimensions")?,
                metric: self.metric,
                m: checked_index_size(self.m, "m")?,
                ef_construction: checked_index_size(self.ef_construction, "efConstruction")?,
                ef: None,
                quantization: self.quantization,
            },
            other => {
                return Err(NodeGrafeoError::InvalidArgument(format!(
                    "unknown index kind '{other}'"
                ))
                .into());
            }
        };
        let components: Vec<&str> = self
            .graph
            .as_deref()
            .map_or(&[][..], |path| path)
            .iter()
            .map(String::as_str)
            .collect();
        let graph = GraphPath::from_components(&components)
            .map_err(|error| NodeGrafeoError::InvalidArgument(error.to_string()))?;
        Ok(grafeo_engine::CreateIndexRequest {
            graph,
            name: self.name,
            label: self.label,
            property: self.property,
            kind,
        })
    }
}

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn checked_index_string(value: JsString<'_>, field: &str) -> Result<String> {
    // N-API's buffer includes one terminator; as_str excludes that terminator
    // while still validating UTF-16 and retaining caller-owned NUL characters.
    value
        .into_utf16()?
        .as_str()
        .map_err(|_| napi::Error::from_reason(format!("{field} contains invalid UTF-16")))
}

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn optional_index_string(object: &Object<'_>, field: &str) -> Result<Option<String>> {
    let value: Option<JsString<'_>> = object.get_named_property(field)?;
    value
        .map(|value| checked_index_string(value, field))
        .transpose()
}

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn checked_index_size(value: Option<f64>, field: &str) -> Result<Option<usize>> {
    value
        .map(|value| {
            if !(0.0..=9_007_199_254_740_991.0).contains(&value) || value.fract() != 0.0 {
                return Err(NodeGrafeoError::InvalidArgument(format!(
                    "{field} must be a non-negative safe integer"
                ))
                .into());
            }
            value.to_string().parse::<usize>().map_err(|_| {
                NodeGrafeoError::InvalidArgument(format!(
                    "{field} is outside the supported integer range"
                ))
                .into()
            })
        })
        .transpose()
}

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn checked_index_owner(owner: f64) -> Result<grafeo_common::types::IndexId> {
    owner
        .to_string()
        .parse::<u32>()
        .map(grafeo_common::types::IndexId::new)
        .map_err(|_| {
            NodeGrafeoError::InvalidArgument(
                "index owner must be an unsigned 32-bit integer".into(),
            )
            .into()
        })
}

/// Converts a serde_json filter map to a Grafeo filter map.
#[cfg(any(feature = "vector-index", feature = "hybrid-search"))]
fn convert_json_filters(
    filters: Option<HashMap<String, serde_json::Value>>,
) -> Result<Option<HashMap<String, Value>>> {
    let Some(map) = filters else {
        return Ok(None);
    };
    let mut result = HashMap::new();
    for (key, val) in &map {
        let grafeo_val = json_to_value(val)?;
        result.insert(key.clone(), grafeo_val);
    }
    Ok(Some(result))
}

/// Validate a JavaScript number as a safe node ID.
///
/// JavaScript numbers are f64, but entity IDs are u64. This rejects
/// negative values, NaN, Infinity, and values beyond `Number.MAX_SAFE_INTEGER`.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn validate_node_id(id: f64) -> Result<NodeId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&id) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid node ID: {id}")).into());
    }
    // reason: Range check above guarantees the value is in [0, 2^53-1], safe for u64
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(NodeId(id as u64))
}

/// Validate a JavaScript number as a safe edge ID.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
fn validate_edge_id(id: f64) -> Result<EdgeId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&id) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid edge ID: {id}")).into());
    }
    // reason: Range check above guarantees the value is in [0, 2^53-1], safe for u64
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(EdgeId(id as u64))
}

/// Validate a JavaScript number as a non-negative epoch ID.
///
/// Rejects negative values, NaN, Infinity, and values beyond
/// `Number.MAX_SAFE_INTEGER`. Epochs are unsigned 64-bit integers internally.
#[cfg(any(
    feature = "compact-store",
    feature = "storage",
    feature = "native",
    feature = "embedded"
))]
fn validate_epoch(epoch: f64) -> Result<grafeo_common::types::EpochId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&epoch) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid epoch: {epoch}")).into());
    }
    // reason: Range check above guarantees the value is in [0, 2^53-1], safe for u64
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(grafeo_common::types::EpochId::new(epoch as u64))
}

#[cfg(feature = "triple-store")]
pub(crate) fn parse_node_rdf_term(s: &str) -> Result<grafeo_engine::Term> {
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
            NodeGrafeoError::InvalidArgument(format!(
                "invalid RDF term '{s}': expected N-Triples or a bare IRI"
            ))
            .into()
        })
}

#[cfg(feature = "triple-store")]
pub(crate) fn parse_node_rdf_quad(
    subject: &str,
    predicate: &str,
    object: &str,
    graph: Option<&str>,
) -> Result<grafeo_engine::Quad> {
    let subject_term = parse_node_rdf_term(subject)?;
    if !subject_term.is_iri() && !subject_term.is_blank_node() {
        return Err(NodeGrafeoError::InvalidArgument(
            "RDF subject must be an IRI or blank node".into(),
        )
        .into());
    }
    let predicate_term = parse_node_rdf_term(predicate)?;
    if !predicate_term.is_iri() {
        return Err(NodeGrafeoError::InvalidArgument("RDF predicate must be an IRI".into()).into());
    }
    let triple =
        grafeo_engine::Triple::new(subject_term, predicate_term, parse_node_rdf_term(object)?);
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

#[cfg(feature = "triple-store")]
pub(crate) fn parse_node_quad_list(quads: &[Vec<String>]) -> Result<Vec<grafeo_engine::Quad>> {
    validate_rdf_input_count(quads.len())?;
    quads
        .iter()
        .map(|parts| {
            if parts.len() < 3 {
                return Err(NodeGrafeoError::InvalidArgument(
                    "RDF quad must be [subject, predicate, object] or [subject, predicate, object, graph]"
                        .into(),
                )
                .into());
            }
            parse_node_rdf_quad(
                &parts[0],
                &parts[1],
                &parts[2],
                parts.get(3).map(String::as_str),
            )
        })
        .collect()
}

#[cfg(feature = "triple-store")]
fn validate_rdf_input_count(count: usize) -> Result<()> {
    u32::try_from(count).map(|_| ()).map_err(|_| {
        NodeGrafeoError::InvalidArgument("RDF input exceeds the unsigned 32-bit count limit".into())
            .into()
    })
}

#[cfg(feature = "triple-store")]
pub(crate) fn rdf_insert_count(count: usize) -> Result<u32> {
    u32::try_from(count).map_err(|_| {
        NodeGrafeoError::Native(grafeo_common::utils::error::Error::Internal(
            "RDF inserted count exceeds the admitted input range".into(),
        ))
        .into()
    })
}

#[cfg(feature = "triple-store")]
fn rdf_insert_receipt(
    inserted: usize,
    epoch: grafeo_common::types::EpochId,
) -> Result<Vec<Either<u32, String>>> {
    Ok(vec![
        Either::A(rdf_insert_count(inserted)?),
        Either::B(epoch.as_u64().to_string()),
    ])
}

#[cfg(all(test, feature = "triple-store"))]
mod rdf_insert_receipt_tests {
    use super::*;

    #[test]
    fn receipt_preserves_unsigned_epochs_beyond_javascript_and_signed_ranges() {
        for (epoch, expected) in [
            (0, "0"),
            (9_007_199_254_740_991, "9007199254740991"),
            (9_007_199_254_740_993, "9007199254740993"),
            (9_223_372_036_854_775_808, "9223372036854775808"),
            (u64::MAX, "18446744073709551615"),
        ] {
            let receipt = rdf_insert_receipt(1, grafeo_common::types::EpochId::new(epoch))
                .expect("admitted count");
            assert!(
                matches!(receipt.as_slice(), [Either::A(1), Either::B(value)] if value == expected)
            );
        }
    }

    #[test]
    fn counts_preserve_the_full_unsigned_32_bit_range() {
        for count in [0, 1, u32::MAX] {
            let native =
                usize::try_from(count).expect("supported target has at least 32-bit usize");
            validate_rdf_input_count(native).expect("representable input");
            assert_eq!(
                rdf_insert_count(native).expect("representable result"),
                count
            );
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn oversized_input_and_impossible_results_fail_instead_of_truncating() {
        let too_large = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit target");
        assert_eq!(
            validate_rdf_input_count(too_large).unwrap_err().status,
            Status::InvalidArg
        );
        let error = rdf_insert_count(too_large).unwrap_err();
        assert_eq!(error.status, Status::GenericFailure);
        assert!(error.reason.contains("admitted input range"));
    }
}

#[cfg(feature = "compact-store")]
fn parse_asof_direction(direction: Option<&str>) -> Result<Direction> {
    match direction {
        None | Some("outgoing") | Some("out") => Ok(Direction::Outgoing),
        Some("incoming") | Some("in") => Ok(Direction::Incoming),
        Some("both") => Ok(Direction::Both),
        Some(other) => Err(NodeGrafeoError::InvalidArgument(format!(
            "unknown direction '{other}': expected 'outgoing', 'incoming', or 'both'"
        ))
        .into()),
    }
}

/// Your connection to a Grafeo database.
#[napi(js_name = "GrafeoDB")]
pub struct JsGrafeoDB {
    inner: Arc<RwLock<GrafeoDB>>,
}

#[napi]
impl JsGrafeoDB {
    /// Create a database. Pass a path for persistence, or omit for in-memory.
    /// `graphModel` is `"lpg"`, `"rdf"`, or `"both"`.
    #[napi(
        factory,
        ts_args_type = "path?: string | undefined | null, graphModel?: 'lpg' | 'rdf' | 'both'"
    )]
    pub fn create(path: Option<String>, graph_model: Option<String>) -> Result<Self> {
        let mut config = match path {
            Some(p) => Config::persistent(p),
            None => Config::in_memory(),
        };
        if let Some(model) = graph_model {
            let parsed = GraphModel::from_name(&model).ok_or_else(|| {
                NodeGrafeoError::InvalidArgument(format!(
                    "unknown graphModel '{model}': expected 'lpg', 'rdf', or 'both'"
                ))
            })?;
            config = config.with_graph_model(parsed);
        }
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Graph model this database was created with: `"lpg"`, `"rdf"`, or `"both"`.
    #[napi(js_name = "graphModel", ts_return_type = "'lpg' | 'rdf' | 'both'")]
    pub fn graph_model(&self) -> String {
        self.inner.read().graph_model().as_name().to_string()
    }

    /// Open an existing database at the given path.
    #[napi(factory)]
    pub fn open(path: String) -> Result<Self> {
        let config = Config::persistent(path);
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database in read-only mode.
    ///
    /// Uses a shared file lock, so multiple processes can read the same
    /// .grafeo file concurrently. Mutations will throw an error.
    #[napi(factory)]
    pub fn open_read_only(path: String) -> Result<Self> {
        let config = Config::read_only(path);
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Admit JavaScript arguments before scheduling owned native work.
    fn execute_language_impl<'env>(
        &self,
        env: &'env Env,
        language: &str,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        let (params, prepared) = prepare_node_query(language, params, options)?;
        let db = Arc::clone(&self.inner);
        crate::error::spawn_execution(env, async move {
            tokio::task::spawn_blocking(move || {
                let result = db
                    .read()
                    .execute_with_options(&query, params, prepared.native)
                    .map_err(NodeGrafeoError::from)?;
                finish_node_result(result, prepared.max_bytes)
            })
            .await
            .map_err(|error| NodeGrafeoError::Database(error.to_string()))?
        })
    }

    /// Execute a GQL query with optional cancellation and output limits.
    #[napi(
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "gql", query, params, options)
    }

    /// Begin a transaction with an optional isolation level.
    ///
    /// Isolation levels: "read_committed", "snapshot" (default), "serializable".
    #[napi(js_name = "beginTransaction")]
    pub fn begin_transaction(
        &self,
        env: Env,
        isolation_level: Option<String>,
    ) -> Result<Transaction> {
        Transaction::new(self.inner.clone(), isolation_level.as_deref()).map_err(
            |error| match error {
                NodeGrafeoError::Native(native) => crate::error::native_to_js_error(&env, native),
                other => other.into(),
            },
        )
    }

    /// Returns the Grafeo engine version string.
    #[napi]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Clear all cached query plans.
    ///
    /// Forces re-parsing and re-optimization on next execution.
    /// Called automatically after DDL operations, but can be invoked manually.
    #[napi(js_name = "clearPlanCache")]
    pub fn clear_plan_cache(&self) {
        self.inner.read().clear_plan_cache();
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage.
    #[napi(js_name = "walCheckpoint")]
    pub fn wal_checkpoint(&self) -> Result<()> {
        let db = self.inner.read();
        db.wal_checkpoint()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Close the database.
    #[napi]
    pub fn close(&self, env: Env) -> Result<()> {
        self.inner
            .read()
            .try_close()
            .map_err(|error| crate::error::native_to_js_error(&env, error))
    }

    // ── Schema context ───────────────────────────────────────────────────

    /// Sets the current schema for subsequent `execute()` calls.
    ///
    /// Equivalent to running `SESSION SET SCHEMA <name>` but persists across
    /// calls. Use `resetSchema()` to clear it.
    #[napi(js_name = "setSchema")]
    pub fn set_schema(&self, name: String) -> napi::Result<()> {
        self.inner
            .read()
            .set_current_schema(Some(&name))
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Clears the current schema context.
    ///
    /// Subsequent `execute()` calls will use the default (no-schema) namespace.
    #[napi(js_name = "resetSchema")]
    pub fn reset_schema(&self) {
        let _ = self.inner.read().set_current_schema(None);
    }

    /// Returns the current schema name, or `null` if no schema is set.
    #[napi(js_name = "currentSchema")]
    pub fn current_schema(&self) -> Option<String> {
        self.inner.read().current_schema()
    }

    // ── Graph projections ───────────────────────────────────────────────

    /// Creates a named graph projection. Returns `true` if created, `false`
    /// if a projection with that name already exists.
    ///
    /// A projection is a read-only, filtered view of the default graph.
    /// Only nodes with matching labels and edges with matching types are visible.
    #[napi(js_name = "createProjection")]
    pub fn create_projection(
        &self,
        name: String,
        node_labels: Option<Vec<String>>,
        edge_types: Option<Vec<String>>,
    ) -> bool {
        use grafeo_core::graph::ProjectionSpec;

        let mut spec = ProjectionSpec::new();
        if let Some(labels) = node_labels.filter(|l| !l.is_empty()) {
            spec = spec.with_node_labels(labels);
        }
        if let Some(types) = edge_types.filter(|t| !t.is_empty()) {
            spec = spec.with_edge_types(types);
        }
        self.inner.read().create_projection(name, spec)
    }

    /// Drops a named graph projection. Returns `true` if it existed.
    #[napi(js_name = "dropProjection")]
    pub fn drop_projection(&self, name: String) -> bool {
        self.inner.read().drop_projection(&name)
    }

    /// Returns the names of all graph projections.
    #[napi(js_name = "listProjections")]
    pub fn list_projections(&self) -> Vec<String> {
        self.inner.read().list_projections()
    }
}

// Canonical index owners are available in every profile enabling engine LPG.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
#[napi]
impl JsGrafeoDB {
    /// Create an index and return its committed owner ID.
    /// Graph paths are component arrays, never slash-delimited strings.
    #[napi(
        js_name = "createIndex",
        ts_args_type = "request: CreateIndexRequest",
        ts_return_type = "Promise<number>"
    )]
    pub fn create_index<'env>(
        &self,
        env: &'env Env,
        request: Unknown<'_>,
    ) -> Result<PromiseRaw<'env, u32>> {
        let request =
            CreateIndexRequest::from_js(request).and_then(CreateIndexRequest::into_engine);
        let db = Arc::clone(&self.inner);
        env.spawn_future(async move {
            let request = request?;
            tokio::task::spawn_blocking(move || {
                db.read()
                    .create_index(request)
                    .map(|owner| owner.as_u32())
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            })
            .await
            .map_err(|error| napi::Error::from_reason(error.to_string()))?
        })
    }

    /// Drop an owner; false means only that the owner was absent.
    #[napi(js_name = "dropIndex")]
    pub async fn drop_index(&self, owner: f64) -> Result<bool> {
        let owner = checked_index_owner(owner)?;
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || {
            db.read()
                .drop_index(owner)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|error| napi::Error::from_reason(error.to_string()))?
    }

    /// Atomically rebuild an existing owner, preserving its ID and configuration.
    #[napi(js_name = "rebuildIndex")]
    pub async fn rebuild_index(&self, owner: f64) -> Result<()> {
        let owner = checked_index_owner(owner)?;
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || {
            db.read()
                .rebuild_index(owner)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|error| napi::Error::from_reason(error.to_string()))?
    }
}

// LPG node/edge CRUD, admin, and backup. Separate impl because napi-rs
// generates callback registrations for every method inside a `#[napi]` impl,
// so a per-method `#[cfg]` does not work.
#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
#[napi]
impl JsGrafeoDB {
    /// Create a node with labels and optional properties.
    #[napi(js_name = "createNode")]
    pub fn create_node(
        &self,
        env: Env,
        mut labels: Vec<String>,
        properties: Option<Object<'_>>,
    ) -> Result<JsNode> {
        let db = self.inner.read();
        let session = db.session();
        labels.sort_unstable();
        labels.dedup();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let mut props = Vec::new();
        if let Some(props_obj) = properties {
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: Unknown<'_> = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
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
            .map_err(NodeGrafeoError::from)?;
        // Null removes a property in storage; response keys must match that image.
        props.retain(|(_, value)| !matches!(value, Value::Null));
        Ok(JsNode::new(id, labels, props.into_iter().collect()))
    }

    /// Create an edge between two nodes.
    #[napi(js_name = "createEdge")]
    pub fn create_edge(
        &self,
        env: Env,
        source_id: f64,
        target_id: f64,
        edge_type: String,
        properties: Option<Object<'_>>,
    ) -> Result<JsEdge> {
        let db = self.inner.read();
        let session = db.session();
        let src = validate_node_id(source_id)?;
        let dst = validate_node_id(target_id)?;

        let mut props = Vec::new();
        if let Some(props_obj) = properties {
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: Unknown<'_> = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
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
            .map_err(NodeGrafeoError::from)?;
        props.retain(|(_, value)| !matches!(value, Value::Null));
        Ok(JsEdge::new(
            id,
            edge_type,
            src,
            dst,
            props.into_iter().collect(),
        ))
    }

    /// Get a node by ID.
    #[napi(js_name = "getNode")]
    pub fn get_node(&self, id: f64) -> Result<Option<JsNode>> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.get_node(node_id).map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node.properties.into_iter().collect();
            JsNode::new(node_id, labels, properties)
        }))
    }

    /// Get an edge by ID.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(&self, id: f64) -> Result<Option<JsEdge>> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.get_edge(edge_id).map(|edge| {
            let properties = edge.properties.into_iter().collect();
            JsEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )
        }))
    }

    /// Delete a node by ID. Returns true if the node existed.
    #[napi(js_name = "deleteNode")]
    pub fn delete_node(&self, id: f64) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.delete_node(node_id))
    }

    /// Delete an edge by ID. Returns true if the edge existed.
    #[napi(js_name = "deleteEdge")]
    pub fn delete_edge(&self, id: f64) -> Result<bool> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.delete_edge(edge_id))
    }

    /// Set a property on a node.
    /// Throws if the node is missing or the write is rejected.
    #[napi(js_name = "setNodeProperty")]
    pub fn set_node_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_node_property(node_id, &key, val)
            .map_err(NodeGrafeoError::from)?;
        Ok(())
    }

    /// Set a property on an edge.
    /// Throws if the edge is missing or the write is rejected.
    #[napi(js_name = "setEdgeProperty")]
    pub fn set_edge_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_edge_property(edge_id, &key, val)
            .map_err(NodeGrafeoError::from)?;
        Ok(())
    }

    /// Get the number of nodes.
    #[napi(js_name = "nodeCount")]
    pub fn node_count(&self) -> u32 {
        // reason: WASM targets use 32-bit usize; on 64-bit, graphs with >4B nodes are unrealistic
        #[allow(clippy::cast_possible_truncation)]
        let count = self.inner.read().node_count() as u32;
        count
    }

    /// Get the number of edges.
    #[napi(js_name = "edgeCount")]
    pub fn edge_count(&self) -> u32 {
        // reason: WASM targets use 32-bit usize; on 64-bit, graphs with >4B edges are unrealistic
        #[allow(clippy::cast_possible_truncation)]
        let count = self.inner.read().edge_count() as u32;
        count
    }

    /// Bulk-insert nodes with vector properties.
    #[napi(js_name = "batchCreateNodes")]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    pub async fn batch_create_nodes(
        &self,
        label: String,
        property: String,
        vectors: Vec<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let vecs_f32: Vec<Vec<f32>> = vectors
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            let ids = db.batch_create_nodes(&label, &property, vecs_f32);
            Ok(ids
                .into_iter()
                .map(|id| id.as_u64() as f64)
                .collect::<Vec<f64>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Remove a property from a node. Returns true if the property existed.
    #[napi(js_name = "removeNodeProperty")]
    pub fn remove_node_property(&self, id: f64, key: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.remove_node_property(node_id, &key))
    }

    /// Remove a property from an edge. Returns true if the property existed.
    #[napi(js_name = "removeEdgeProperty")]
    pub fn remove_edge_property(&self, id: f64, key: String) -> Result<bool> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.remove_edge_property(edge_id, &key))
    }

    /// Add a label to an existing node. Returns true if the label was added.
    #[napi(js_name = "addNodeLabel")]
    pub fn add_node_label(&self, id: f64, label: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.add_node_label(node_id, &label))
    }

    /// Remove a label from a node. Returns true if the label was removed.
    #[napi(js_name = "removeNodeLabel")]
    pub fn remove_node_label(&self, id: f64, label: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.remove_node_label(node_id, &label))
    }

    /// Get all labels for a node. Returns null if the node doesn't exist.
    #[napi(js_name = "getNodeLabels")]
    pub fn get_node_labels(&self, id: f64) -> Result<Option<Vec<String>>> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.get_node_labels(node_id))
    }

    /// Returns high-level database information as a JSON object.
    #[napi]
    pub fn info(&self) -> Result<serde_json::Value> {
        let db = self.inner.read();
        let info = db.info();
        serde_json::to_value(&info).map_err(|e| NodeGrafeoError::Database(e.to_string()).into())
    }

    /// Returns schema information (labels, edge types, property keys) as a JSON object.
    #[napi]
    pub fn schema(&self) -> Result<serde_json::Value> {
        let db = self.inner.read();
        let schema = db.schema();
        serde_json::to_value(&schema).map_err(|e| NodeGrafeoError::Database(e.to_string()).into())
    }
}

// Saving is shared by LPG and RDF whenever this binding supplies WAL storage.
#[cfg(any(feature = "storage", feature = "embedded", feature = "native"))]
#[napi]
impl JsGrafeoDB {
    /// Saves the database to a file path.
    ///
    /// If in-memory, creates a new persistent database at the given path.
    /// If file-backed, creates a copy at the new path.
    /// The original database remains unchanged.
    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        let db = self.inner.read();
        db.save(path)
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }
}

// Backup additionally requires the compiled LPG plane.
#[cfg(all(
    any(
        feature = "lpg",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "compact-store"
    ),
    any(feature = "storage", feature = "embedded", feature = "native")
))]
#[napi]
impl JsGrafeoDB {
    /// Create a full backup of the database.
    #[napi]
    pub fn backup_full(&self, backup_dir: String) -> Result<()> {
        let db = self.inner.read();
        db.backup_full(std::path::Path::new(&backup_dir))
            .map(|_| ())
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Create an incremental backup (WAL records since last backup).
    #[napi]
    pub fn backup_incremental(&self, backup_dir: String) -> Result<()> {
        let db = self.inner.read();
        db.backup_incremental(std::path::Path::new(&backup_dir))
            .map(|_| ())
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }
}

// Vector-index methods live in a separate impl block so the entire block can
// be conditionally compiled.  napi-rs generates callback registrations for
// every method inside a `#[napi]` impl, so a per-method `#[cfg]` doesn't work.
#[cfg(feature = "vector-index")]
#[napi]
impl JsGrafeoDB {
    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Returns an array of [nodeId, distance] pairs sorted by distance
    /// ascending (lower = more similar). The distance scale depends on
    /// the metric configured at index creation.
    #[napi(js_name = "vectorSearch")]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    pub async fn vector_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f64>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
            let results = db
                .vector_search(
                    &label,
                    &property,
                    &query_f32,
                    k as usize,
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Batch search for nearest neighbors of multiple query vectors.
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    #[napi(js_name = "batchVectorSearch")]
    pub async fn batch_vector_search(
        &self,
        label: String,
        property: String,
        queries: Vec<Vec<f64>>,
        k: u32,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let queries_f32: Vec<Vec<f32>> = queries
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            let results = db
                .batch_vector_search(
                    &label,
                    &property,
                    &queries_f32,
                    k as usize,
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|inner| {
                    inner
                        .into_iter()
                        .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                        .collect::<Vec<Vec<f64>>>()
                })
                .collect::<Vec<Vec<Vec<f64>>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
    ///
    /// Returns an array of [nodeId, distance] pairs in MMR selection order.
    /// The distance values match vectorSearch() for the same nodes
    /// (lower = more similar). The ordering reflects relevance-diversity
    /// balance, not distance sorting.
    #[napi(js_name = "mmrSearch")]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_arguments)]
    pub async fn mmr_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        fetch_k: Option<u32>,
        lambda_mult: Option<f64>,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f64>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
            let results = db
                .mmr_search(
                    &label,
                    &property,
                    &query_f32,
                    k as usize,
                    fetch_k.map(|v| v as usize),
                    lambda_mult.map(|v| v as f32),
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Text-index methods live in a separate impl block for the same reason.
#[cfg(feature = "text-index")]
#[napi]
impl JsGrafeoDB {
    /// Search a text index using BM25 scoring.
    ///
    /// Returns an array of [nodeId, score] pairs sorted by descending
    /// relevance (higher score = more relevant). BM25 scores are
    /// unbounded positive floats.
    #[napi(js_name = "textSearch")]
    pub async fn text_search(
        &self,
        label: String,
        property: String,
        query: String,
        k: u32,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let results = db
                .text_search(&label, &property, &query, k as usize)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, score)| vec![id.as_u64() as f64, score])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Hybrid-search methods live in a separate impl block for the same reason.
#[cfg(feature = "hybrid-search")]
#[napi]
impl JsGrafeoDB {
    /// Perform hybrid search combining text (BM25) and vector similarity.
    ///
    /// Requires both a text index and a vector index
    /// (both created with createIndex). If either is missing, that source is silently
    /// omitted from fusion.
    ///
    /// Returns an array of [nodeId, score] pairs sorted by fused score
    /// descending (higher = more relevant). These are fusion scores,
    /// NOT distances.
    #[napi(js_name = "hybridSearch")]
    #[allow(clippy::too_many_arguments)]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    pub async fn hybrid_search(
        &self,
        label: String,
        text_property: String,
        vector_property: String,
        query_text: String,
        k: u32,
        query_vector: Option<Vec<f64>>,
        fusion: Option<String>,
        weights: Option<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let fusion_method = match fusion.as_deref() {
                Some("weighted") => {
                    let w = weights.unwrap_or_else(|| vec![0.5, 0.5]);
                    Some(grafeo_core::index::text::FusionMethod::Weighted { weights: w })
                }
                _ => None,
            };

            let query_vec_f32: Option<Vec<f32>> =
                query_vector.map(|v| v.iter().map(|&x| x as f32).collect());

            let db = db.read();
            let results = db
                .hybrid_search(
                    &label,
                    &text_property,
                    &vector_property,
                    &query_text,
                    query_vec_f32.as_deref(),
                    k as usize,
                    fusion_method,
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, score)| vec![id.as_u64() as f64, score])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Compact-store methods live in a separate impl block for the same reason.
#[cfg(feature = "compact-store")]
#[napi]
impl JsGrafeoDB {
    /// Folds retained committed LPG history into a columnar base with a writable overlay.
    ///
    /// Call again to fold later overlay writes. Finish asynchronous operations
    /// and drop live Sessions before maintenance; active transactions, closed
    /// or durability-poisoned databases are rejected. Throws on failure.
    /// This is not a durability checkpoint or a history-retention lease.
    #[napi]
    pub fn compact(&self) -> Result<()> {
        let mut db = self.inner.write();
        db.compact()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Whole-state as-of scrub: node frames plus edge frames at `epoch`.
    ///
    /// Returns `{ nodes: [...], edges: [...] }`.
    ///
    /// Each node frame is
    /// `{ label, nodeIds, columns }` and each edge frame is
    /// `{ edgeType, edgeIds, srcIds, dstIds, columns }`. Columns align to
    /// the id list (`null` = property absent at `epoch`).
    ///
    /// `PENDING` cannot be passed as a JS number; omit `epoch` on
    /// `neighborsAtEpoch` / `edgesAtEpoch` for the current snapshot, or
    /// pass `currentEpoch()`. Requires `compact()`; empty frames when
    /// nothing has been compacted yet.
    #[napi(js_name = "scrubAtEpoch")]
    pub fn scrub_at_epoch(&self, epoch: f64) -> Result<serde_json::Value> {
        let epoch_id = validate_epoch(epoch)?;
        let db = self.inner.read();
        let scrub = db.scrub_at_epoch(epoch_id);

        let mut nodes: Vec<serde_json::Value> = Vec::with_capacity(scrub.nodes.len());
        for frame in &scrub.nodes {
            let node_ids: Vec<serde_json::Value> = frame
                .node_ids
                .iter()
                .map(|n| serde_json::Value::from(n.as_u64()))
                .collect();
            let mut columns = serde_json::Map::new();
            for (key, values) in &frame.columns {
                let col: Vec<serde_json::Value> = values
                    .iter()
                    .map(|v| {
                        v.as_ref()
                            .map_or(serde_json::Value::Null, grafeo_value_to_json)
                    })
                    .collect();
                columns.insert(key.to_string(), serde_json::Value::Array(col));
            }
            let mut obj = serde_json::Map::new();
            obj.insert(
                "label".to_string(),
                serde_json::Value::String(frame.label.to_string()),
            );
            obj.insert("nodeIds".to_string(), serde_json::Value::Array(node_ids));
            obj.insert("columns".to_string(), serde_json::Value::Object(columns));
            nodes.push(serde_json::Value::Object(obj));
        }

        let mut edges: Vec<serde_json::Value> = Vec::with_capacity(scrub.edges.len());
        for frame in &scrub.edges {
            let edge_ids: Vec<serde_json::Value> = frame
                .edge_ids
                .iter()
                .map(|e| serde_json::Value::from(e.as_u64()))
                .collect();
            let src_ids: Vec<serde_json::Value> = frame
                .src_ids
                .iter()
                .map(|n| serde_json::Value::from(n.as_u64()))
                .collect();
            let dst_ids: Vec<serde_json::Value> = frame
                .dst_ids
                .iter()
                .map(|n| serde_json::Value::from(n.as_u64()))
                .collect();
            let mut columns = serde_json::Map::new();
            for (key, values) in &frame.columns {
                let col: Vec<serde_json::Value> = values
                    .iter()
                    .map(|v| {
                        v.as_ref()
                            .map_or(serde_json::Value::Null, grafeo_value_to_json)
                    })
                    .collect();
                columns.insert(key.to_string(), serde_json::Value::Array(col));
            }
            let mut obj = serde_json::Map::new();
            obj.insert(
                "edgeType".to_string(),
                serde_json::Value::String(frame.edge_type.to_string()),
            );
            obj.insert("edgeIds".to_string(), serde_json::Value::Array(edge_ids));
            obj.insert("srcIds".to_string(), serde_json::Value::Array(src_ids));
            obj.insert("dstIds".to_string(), serde_json::Value::Array(dst_ids));
            obj.insert("columns".to_string(), serde_json::Value::Object(columns));
            edges.push(serde_json::Value::Object(obj));
        }

        let mut out = serde_json::Map::new();
        out.insert("nodes".to_string(), serde_json::Value::Array(nodes));
        out.insert("edges".to_string(), serde_json::Value::Array(edges));
        Ok(serde_json::Value::Object(out))
    }

    /// Neighbors of `nodeId` visible at `epoch`.
    ///
    /// Omit `epoch` (or pass `null`) for `PENDING` — current 1-hop,
    /// the derived open CSR. `direction` is `"outgoing"` (default),
    /// `"incoming"`, or `"both"`.
    #[napi(js_name = "neighborsAtEpoch")]
    pub fn neighbors_at_epoch(
        &self,
        node_id: f64,
        epoch: Option<f64>,
        direction: Option<String>,
    ) -> Result<Vec<f64>> {
        let node = validate_node_id(node_id)?;
        let epoch_id = match epoch {
            None => grafeo_common::types::EpochId::PENDING,
            Some(e) => validate_epoch(e)?,
        };
        let dir = parse_asof_direction(direction.as_deref())?;
        let db = self.inner.read();
        Ok(db
            .neighbors_at_epoch(node, dir, epoch_id)
            .into_iter()
            .map(|n| {
                // reason: node ids returned to JS are already in the safe integer range
                #[allow(clippy::cast_precision_loss)]
                {
                    n.as_u64() as f64
                }
            })
            .collect())
    }

    /// Every edge visible at `epoch`. Omit `epoch` for current (`PENDING`).
    #[napi(js_name = "edgesAtEpoch")]
    pub fn edges_at_epoch(&self, epoch: Option<f64>) -> Result<Vec<JsEdge>> {
        let epoch_id = match epoch {
            None => grafeo_common::types::EpochId::PENDING,
            Some(e) => validate_epoch(e)?,
        };
        let db = self.inner.read();
        Ok(db
            .edges_at_epoch(epoch_id)
            .into_iter()
            .map(|edge| {
                let properties: HashMap<
                    grafeo_common::types::PropertyKey,
                    grafeo_common::types::Value,
                > = edge.properties.into_iter().collect();
                JsEdge::new(
                    edge.id,
                    edge.edge_type.to_string(),
                    edge.src,
                    edge.dst,
                    properties,
                )
            })
            .collect())
    }

    /// Every node visible at `epoch`. Omit `epoch` for current (`PENDING`).
    #[napi(js_name = "nodesAtEpoch")]
    pub fn nodes_at_epoch(&self, epoch: Option<f64>) -> Result<Vec<JsNode>> {
        let epoch_id = match epoch {
            None => grafeo_common::types::EpochId::PENDING,
            Some(e) => validate_epoch(e)?,
        };
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
                JsNode::new(node.id, labels, properties)
            })
            .collect())
    }
}

// CDC methods live in a separate impl block for the same reason.
#[cfg(feature = "cdc")]
#[napi]
impl JsGrafeoDB {
    /// Enable CDC for all future sessions.
    #[napi(js_name = "enableCdc")]
    pub fn enable_cdc(&self) {
        self.inner.read().set_cdc_enabled(true);
    }

    /// Disable CDC for all future sessions.
    #[napi(js_name = "disableCdc")]
    pub fn disable_cdc(&self) {
        self.inner.read().set_cdc_enabled(false);
    }

    /// Returns whether CDC is currently enabled for new sessions.
    #[napi(js_name = "isCdcEnabled", getter)]
    pub fn is_cdc_enabled(&self) -> bool {
        self.inner.read().is_cdc_enabled()
    }

    /// Reads bounded node history with exact decimal ID/epoch coordinates.
    // Keep the explicit foreign-function bounds and optional selector together.
    #[allow(clippy::too_many_arguments)]
    #[napi(js_name = "nodeHistoryAfter")]
    pub fn node_history_after<'env>(
        &self,
        env: &'env Env,
        node_id: String,
        cursor: Option<Buffer>,
        max_events: f64,
        max_bytes: f64,
        since_epoch: Option<String>,
    ) -> Result<PromiseRaw<'env, JsChangePage>> {
        self.cdc_page_impl(
            env,
            Some(entity_history_query(true, &node_id, since_epoch.as_deref())),
            cursor,
            max_events,
            max_bytes,
        )
    }

    /// Reads bounded edge history with exact decimal ID/epoch coordinates.
    // Keep the explicit foreign-function bounds and optional selector together.
    #[allow(clippy::too_many_arguments)]
    #[napi(js_name = "edgeHistoryAfter")]
    pub fn edge_history_after<'env>(
        &self,
        env: &'env Env,
        edge_id: String,
        cursor: Option<Buffer>,
        max_events: f64,
        max_bytes: f64,
        since_epoch: Option<String>,
    ) -> Result<PromiseRaw<'env, JsChangePage>> {
        self.cdc_page_impl(
            env,
            Some(entity_history_query(
                false,
                &edge_id,
                since_epoch.as_deref(),
            )),
            cursor,
            max_events,
            max_bytes,
        )
    }

    /// Reads an owned bounded feed page. An unchanged cursor marks the end.
    #[napi(js_name = "changesAfter")]
    pub fn changes_after<'env>(
        &self,
        env: &'env Env,
        cursor: Option<Buffer>,
        max_events: f64,
        max_bytes: f64,
    ) -> Result<PromiseRaw<'env, JsChangePage>> {
        self.cdc_page_impl(env, None, cursor, max_events, max_bytes)
    }

    fn cdc_page_impl<'env>(
        &self,
        env: &'env Env,
        query: Option<crate::error::NodeResult<grafeo_engine::cdc::EntityHistoryQuery>>,
        cursor: Option<Buffer>,
        max_events: f64,
        max_bytes: f64,
    ) -> Result<PromiseRaw<'env, JsChangePage>> {
        // Decode before scheduling: JavaScript can mutate its Buffer while
        // native work is running. Only a fixed-size owned cursor crosses here.
        let cursor = cursor
            .map(|bytes| grafeo_common::types::DurableCursor::from_bytes(&bytes))
            .transpose();
        let db = Arc::clone(&self.inner);
        crate::error::spawn_execution(env, async move {
            tokio::task::spawn_blocking(move || {
                let limit = |value: f64| {
                    if !value.is_finite()
                        || value < 1.0
                        || value.fract() != 0.0
                        || value > 9_007_199_254_740_991.0
                        || value >= usize::MAX as f64
                    {
                        return Err(NodeGrafeoError::from(
                            grafeo_common::utils::error::Error::InvalidValue(
                                "change page limits must be positive safe integers fitting usize"
                                    .into(),
                            ),
                        ));
                    }
                    // The finite, integral, positive platform bound above proves this cast.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let value = value as usize;
                    Ok(value)
                };
                let max_events = limit(max_events)?;
                let max_bytes = limit(max_bytes)?;
                let cursor = cursor.map_err(NodeGrafeoError::from)?;
                let query = query.transpose()?;
                let db = db.read();
                let session = db.session();
                let page = match query {
                    Some(query) => {
                        session.history_after(&query, cursor.as_ref(), max_events, max_bytes)
                    }
                    None => session.changes_after(cursor.as_ref(), max_events, max_bytes),
                }
                .map_err(NodeGrafeoError::from)?;
                Ok(JsChangePage {
                    events: page.events.iter().map(change_page_event_to_json).collect(),
                    next: page.next.to_bytes().to_vec().into(),
                })
            })
            .await
            .map_err(|error| NodeGrafeoError::Database(error.to_string()))?
        })
    }
}

// Embed methods live in a separate impl block so the entire block can be
// conditionally compiled.  napi-rs generates callback registrations for every
// method inside a `#[napi]` impl, so a per-method `#[cfg]` doesn't work.
#[cfg(feature = "embed")]
#[napi]
impl JsGrafeoDB {
    /// Register an ONNX embedding model for text-to-vector conversion.
    ///
    /// Once registered, use embedText() and vectorSearchText() with the model name.
    #[napi(js_name = "registerEmbeddingModel")]
    pub async fn register_embedding_model(
        &self,
        name: String,
        model_path: String,
        tokenizer_path: String,
        batch_size: Option<u32>,
    ) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = grafeo_engine::embedding::OnnxEmbeddingModel::from_files(
                &name,
                &model_path,
                &tokenizer_path,
            )
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)?;
            if let Some(bs) = batch_size {
                model = model.with_batch_size(bs as usize);
            }
            let db = db.read();
            db.register_embedding_model(&name, std::sync::Arc::new(model));
            Ok(())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Generate embeddings for a list of texts using a registered model.
    ///
    /// Returns an array of float arrays, one per input text.
    #[napi(js_name = "embedText")]
    pub async fn embed_text(
        &self,
        model_name: String,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let results = db
                .embed_text(&model_name, &text_refs)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            // Convert f32 → f64 for JavaScript number compatibility
            Ok(results
                .into_iter()
                .map(|v| v.into_iter().map(f64::from).collect())
                .collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// This method requires both embed and vector-index, so it gets its own block.
#[cfg(all(feature = "embed", feature = "vector-index"))]
#[napi]
impl JsGrafeoDB {
    /// Search a vector index using a text query, generating the embedding on-the-fly.
    ///
    /// Returns an array of [nodeId, distance] pairs.
    #[napi(js_name = "vectorSearchText")]
    pub async fn vector_search_text(
        &self,
        label: String,
        property: String,
        model_name: String,
        query_text: String,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let results = db
                .vector_search_text(
                    &label,
                    &property,
                    &model_name,
                    &query_text,
                    k as usize,
                    ef.map(|e| e as usize),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, f64::from(dist)])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

#[cfg(any(feature = "storage", feature = "native", feature = "embedded"))]
#[napi]
impl JsGrafeoDB {
    /// Restore a database to a specific epoch from a backup chain.
    #[napi]
    pub fn restore_to_epoch(backup_dir: String, epoch: f64, output_path: String) -> Result<()> {
        let epoch_id = validate_epoch(epoch)?;
        grafeo_engine::GrafeoDB::restore_to_epoch(
            std::path::Path::new(&backup_dir),
            epoch_id,
            std::path::Path::new(&output_path),
        )
        .map_err(NodeGrafeoError::from)
        .map_err(napi::Error::from)
    }
}

#[cfg(feature = "gql")]
#[napi]
impl JsGrafeoDB {
    /// Runs a read-only GQL query and returns an async cursor.
    #[napi(
        js_name = "executeStream",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_stream<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, crate::stream::JsResultStream>> {
        let (params, mut prepared) = prepare_node_query("gql", params, options)?;
        prepared.native.result_admission = None;
        let cancellation = prepared.native.control.cancellation_handle();
        let db = Arc::clone(&self.inner);
        crate::error::spawn_execution(env, async move {
            tokio::task::spawn_blocking(move || {
                let stream = db
                    .read()
                    .stream_with_options(&query, params, prepared.native)
                    .map_err(NodeGrafeoError::from)?;
                crate::stream::JsResultStream::new(
                    db,
                    stream,
                    prepared.max_bytes,
                    prepared.max_rows,
                    cancellation,
                )
            })
            .await
            .map_err(|error| NodeGrafeoError::Database(error.to_string()))?
        })
    }
}

#[cfg(feature = "triple-store")]
#[napi]
impl JsGrafeoDB {
    /// Insert one RDF quad. Terms are N-Triples or bare IRIs.
    /// Returns the number of newly inserted quads (0 or 1).
    #[napi(js_name = "insertRdfQuad")]
    pub fn insert_rdf_quad(
        &self,
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    ) -> Result<u32> {
        let quad = parse_node_rdf_quad(&subject, &predicate, &object, graph.as_deref())?;
        let (n, _) = self
            .inner
            .read()
            .insert_rdf_quads([quad])
            .map_err(NodeGrafeoError::from)?;
        rdf_insert_count(n)
    }

    /// Bulk-insert RDF quads. Each item is `[subject, predicate, object]` or
    /// `[subject, predicate, object, graph]`.
    /// Returns `[inserted, epoch]`, with the exact epoch as a decimal string.
    #[napi(
        js_name = "insertRdfQuads",
        ts_return_type = "[number, string]",
        ts_args_type = "quads: Array<[string, string, string] | [string, string, string, string]>"
    )]
    pub fn insert_rdf_quads(&self, quads: Vec<Vec<String>>) -> Result<Vec<Either<u32, String>>> {
        let parsed = parse_node_quad_list(&quads)?;
        let (n, epoch) = self
            .inner
            .read()
            .insert_rdf_quads(parsed)
            .map_err(NodeGrafeoError::from)?;
        rdf_insert_receipt(n, epoch)
    }

    /// Exact typed-quad membership.
    #[napi(js_name = "containsRdfQuad")]
    pub fn contains_rdf_quad(
        &self,
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    ) -> Result<bool> {
        let quad = parse_node_rdf_quad(&subject, &predicate, &object, graph.as_deref())?;
        Ok(self
            .inner
            .read()
            .try_contains_rdf_quad(&quad)
            .map_err(NodeGrafeoError::from)?)
    }
}

// Language-specific execute methods live in separate impl blocks so the
// `#[napi]` macro only generates C callback symbols when the feature is active.

#[cfg(feature = "cypher")]
#[napi]
impl JsGrafeoDB {
    /// Execute a Cypher query.
    #[napi(
        js_name = "executeCypher",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_cypher<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "cypher", query, params, options)
    }
}

#[cfg(feature = "sql-pgq")]
#[napi]
impl JsGrafeoDB {
    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    #[napi(
        js_name = "executeSql",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_sql<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "sql", query, params, options)
    }
}

#[cfg(feature = "gremlin")]
#[napi]
impl JsGrafeoDB {
    /// Execute a Gremlin query.
    #[napi(
        js_name = "executeGremlin",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_gremlin<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "gremlin", query, params, options)
    }
}

#[cfg(feature = "graphql")]
#[napi]
impl JsGrafeoDB {
    /// Execute a GraphQL query.
    #[napi(
        js_name = "executeGraphql",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_graphql<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "graphql", query, params, options)
    }
}

#[cfg(feature = "sparql")]
#[napi]
impl JsGrafeoDB {
    /// Execute a SPARQL query against the RDF triple store.
    #[napi(
        js_name = "executeSparql",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_sparql<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "sparql", query, params, options)
    }
}

#[napi]
impl JsGrafeoDB {
    /// Execute a query in a named language with the same execution owner.
    #[napi(
        js_name = "executeLanguage",
        ts_args_type = "language: string, query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_language<'env>(
        &self,
        env: &'env Env,
        language: String,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'env>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, &language, query, params, options)
    }
}

// -- File import methods --------------------------------------------------

#[napi]
impl JsGrafeoDB {
    /// Import a CSV file as graph nodes.
    ///
    /// Each row becomes a node with the given label. Column headers are used
    /// as property names. Returns the number of nodes created.
    #[napi(js_name = "importCsv")]
    pub async fn import_csv(&self, path: String, options: Option<CsvImportOptions>) -> Result<i64> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = options.unwrap_or_default();
            let label = sanitize_gql_identifier(&opts.label.unwrap_or_else(|| "Row".to_string()));
            let headers = opts.headers.unwrap_or(true);

            let abs_path = std::path::Path::new(&path)
                .canonicalize()
                .map_err(|e| NodeGrafeoError::Database(format!("{path}: {e}")))?;
            let path_str = escape_gql_string(&abs_path.to_string_lossy().replace('\\', "/"));

            let header_clause = if headers { " WITH HEADERS" } else { "" };

            let insert_clause = if headers {
                let columns = read_csv_headers(&abs_path, ',')
                    .map_err(|e| NodeGrafeoError::Database(e.clone()))?;
                if columns.is_empty() {
                    format!("INSERT (:{label} {{}})")
                } else {
                    let props = columns
                        .iter()
                        .map(|col| {
                            let safe = sanitize_gql_identifier(col);
                            format!("{safe}: row.{safe}")
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("INSERT (:{label} {{{props}}})")
                }
            } else {
                format!("INSERT (:{label} {{}})")
            };

            let query = format!(
                "LOAD DATA FROM '{path_str}' FORMAT CSV{header_clause} AS row {insert_clause}"
            );

            let db = db.read();
            let session = db.session();

            let before_count = count_nodes(&session, &label);

            session.execute(&query).map_err(NodeGrafeoError::from)?;

            let count = count_nodes(&session, &label) - before_count;

            Ok(count)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Import a JSON Lines file as graph nodes.
    ///
    /// Each line must be a valid JSON object. Object keys become property names.
    /// Returns the number of nodes created.
    #[napi(js_name = "importJsonl")]
    pub async fn import_jsonl(
        &self,
        path: String,
        options: Option<JsonlImportOptions>,
    ) -> Result<i64> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = options.unwrap_or_default();
            let label = sanitize_gql_identifier(&opts.label.unwrap_or_else(|| "Row".to_string()));

            let abs_path = std::path::Path::new(&path)
                .canonicalize()
                .map_err(|e| NodeGrafeoError::Database(format!("{path}: {e}")))?;
            let path_str = escape_gql_string(&abs_path.to_string_lossy().replace('\\', "/"));

            let keys =
                read_jsonl_keys(&abs_path).map_err(|e| NodeGrafeoError::Database(e.clone()))?;

            let insert_clause = if keys.is_empty() {
                format!("INSERT (:{label} {{}})")
            } else {
                let props = keys
                    .iter()
                    .map(|key| {
                        let safe = sanitize_gql_identifier(key);
                        format!("{safe}: row.{safe}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("INSERT (:{label} {{{props}}})")
            };

            let query = format!("LOAD DATA FROM '{path_str}' FORMAT JSONL AS row {insert_clause}");

            let db = db.read();
            let session = db.session();

            let before_count = count_nodes(&session, &label);

            session.execute(&query).map_err(NodeGrafeoError::from)?;

            let count = count_nodes(&session, &label) - before_count;

            Ok(count)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

/// Options for CSV import.
#[derive(Default)]
#[napi(object)]
pub struct CsvImportOptions {
    /// Label to assign to created nodes (default: "Row")
    pub label: Option<String>,
    /// Whether the first row contains headers (default: true)
    pub headers: Option<bool>,
}

/// Options for JSON Lines import.
#[derive(Default)]
#[napi(object)]
pub struct JsonlImportOptions {
    /// Label to assign to created nodes (default: "Row")
    pub label: Option<String>,
}

/// Read CSV headers from the first line of a file.
fn read_csv_headers(
    path: &std::path::Path,
    delimiter: char,
) -> std::result::Result<Vec<String>, String> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = BufReader::new(f);
    let mut header_line = String::new();
    reader
        .read_line(&mut header_line)
        .map_err(|e| format!("Failed to read headers: {e}"))?;
    Ok(header_line
        .trim()
        .split(delimiter)
        .map(|h| h.trim().trim_matches('"').to_string())
        .filter(|h| !h.is_empty())
        .collect())
}

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
fn count_nodes(session: &grafeo_engine::Session, label: &str) -> i64 {
    session
        .execute(&format!("MATCH (n:{label}) RETURN count(n) AS c"))
        .ok()
        .and_then(|r| r.rows().first().cloned())
        .and_then(|row| row.first().cloned())
        .and_then(|v| match v {
            Value::Int64(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0)
}

/// Read JSON keys from the first non-empty line of a JSONL file.
fn read_jsonl_keys(path: &std::path::Path) -> std::result::Result<Vec<String>, String> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read JSONL file: {e}"))?;
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

/// Validate parameters and consume the invocation's native control on JS entry.
pub(crate) fn prepare_node_query(
    language: &str,
    params: Option<serde_json::Value>,
    options: Option<Object<'_>>,
) -> Result<(HashMap<String, Value>, crate::control::PreparedOptions)> {
    let params = grafeo_bindings_common::json::json_params_to_map(params.as_ref())
        .map_err(|message| napi::Error::from(NodeGrafeoError::InvalidArgument(message)))?
        .unwrap_or_default();
    // Parameter validation precedes consumption of the single native owner.
    let mut prepared = crate::control::prepare_execution_options(options)?;
    prepared.native.language = Some(language.to_owned());
    Ok((params, prepared))
}

pub(crate) fn finish_node_result(
    mut result: EngineQueryResult,
    max_bytes: usize,
) -> crate::error::NodeResult<QueryResult> {
    // Dense columns have no entities; rows() would force an infallible cache.
    let (nodes, edges) = if result.is_int64_columnar() {
        (Vec::new(), Vec::new())
    } else {
        grafeo_bindings_common::entity::extract_and_map(
            &result,
            |node| JsNode::new(node.id, node.labels, node.properties),
            |edge| {
                JsEdge::new(
                    edge.id,
                    edge.edge_type,
                    edge.source_id,
                    edge.target_id,
                    edge.properties,
                )
            },
        )
    };
    let columns = std::mem::take(&mut result.columns);
    let time = result.execution_time_ms;
    let scanned = result.rows_scanned;
    Ok(QueryResult::with_metrics(
        columns,
        result.into_rows().map_err(NodeGrafeoError::from)?,
        nodes,
        edges,
        time,
        scanned,
    )
    .with_conversion_limit(max_bytes))
}

/// Convert a serde_json::Value to a Grafeo Value.
#[cfg(any(feature = "vector-index", feature = "hybrid-search"))]
pub(crate) fn json_to_value(v: &serde_json::Value) -> std::result::Result<Value, napi::Error> {
    Ok(grafeo_bindings_common::json::json_to_value(v))
}

/// Convert a Grafeo Value to serde_json::Value.
// Used by compact-store scrub readback; CDC uses the shared page converter.
#[cfg(feature = "compact-store")]
fn grafeo_value_to_json(v: &Value) -> serde_json::Value {
    grafeo_bindings_common::json::value_to_json(v)
}

/// Owned CDC page; its cursor is an opaque canonical byte string.
#[cfg(feature = "cdc")]
#[napi(object)]
pub struct JsChangePage {
    #[napi(ts_type = "Array<JsChangeEvent>")]
    pub events: Vec<serde_json::Value>,
    pub next: Buffer,
}

/// The bounded page transport preserves every bit of native coordinates.
#[cfg(feature = "cdc")]
fn change_page_event_to_json(event: &grafeo_engine::cdc::ChangeEvent) -> serde_json::Value {
    grafeo_bindings_common::cdc::change_event_to_json(event)
}

#[cfg(feature = "cdc")]
fn entity_history_query(
    node: bool,
    id: &str,
    since_epoch: Option<&str>,
) -> crate::error::NodeResult<grafeo_engine::cdc::EntityHistoryQuery> {
    let integer = |text: &str| {
        if text.is_empty() || text.len() > 20 || !text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(NodeGrafeoError::from(
                grafeo_common::utils::error::Error::InvalidValue(
                    "history coordinates must be unsigned decimal u64 strings".into(),
                ),
            ));
        }
        text.parse::<u64>().map_err(|_| {
            NodeGrafeoError::from(grafeo_common::utils::error::Error::InvalidValue(
                "history coordinate exceeds u64".into(),
            ))
        })
    };
    let id = integer(id)?;
    let entity = if node {
        grafeo_engine::cdc::EntityId::Node(grafeo_common::types::NodeId::new(id))
    } else {
        grafeo_engine::cdc::EntityId::Edge(grafeo_common::types::EdgeId::new(id))
    };
    let mut query = grafeo_engine::cdc::EntityHistoryQuery::new(entity);
    query.since_epoch = grafeo_common::types::EpochId::new(integer(since_epoch.unwrap_or("0"))?);
    Ok(query)
}
