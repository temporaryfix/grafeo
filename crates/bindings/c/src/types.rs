//! Opaque handle types and value conversion helpers for the C FFI layer.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use grafeo_common::types::Value;
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
use grafeo_common::types::{PropertyKey, PropertyMap};
use grafeo_engine::database::GrafeoDB;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque database handle. Created by `grafeo_open*`, freed by `grafeo_free_database`.
pub struct GrafeoDatabase {
    pub(crate) inner: Arc<RwLock<GrafeoDB>>,
}

/// Opaque transaction handle. Created by `grafeo_begin_transaction*`, freed by `grafeo_free_transaction`.
pub struct GrafeoTransaction {
    pub(crate) session: parking_lot::Mutex<Option<grafeo_engine::session::Session>>,
    pub(crate) committed: bool,
    pub(crate) rolled_back: bool,
}

impl Drop for GrafeoTransaction {
    fn drop(&mut self) {
        // Auto-rollback if not explicitly committed or rolled back.
        if !self.committed && !self.rolled_back {
            let mut guard = self.session.lock();
            if let Some(ref mut session) = *guard {
                let _ = rollback_transaction(session);
            }
        }
    }
}

pub(crate) fn begin_transaction(
    session: &mut grafeo_engine::session::Session,
) -> grafeo_common::Result<()> {
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "triple-store"
    ))]
    {
        session.begin_transaction()
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
        let _ = session;
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "Transactions require an enabled graph storage model",
            ),
        ))
    }
}

pub(crate) fn commit_transaction(
    session: &mut grafeo_engine::session::Session,
) -> grafeo_common::Result<grafeo_common::types::EpochId> {
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "triple-store"
    ))]
    {
        session.commit()
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
        let _ = session;
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "Transactions require an enabled graph storage model",
            ),
        ))
    }
}

pub(crate) fn rollback_transaction(
    session: &mut grafeo_engine::session::Session,
) -> grafeo_common::Result<()> {
    #[cfg(any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native",
        feature = "triple-store"
    ))]
    {
        session.rollback()
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
        let _ = session;
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "Transactions require an enabled graph storage model",
            ),
        ))
    }
}

/// Query result. Holds JSON-serialized rows and metadata.
pub struct GrafeoResult {
    pub(crate) json: CString,
    pub(crate) row_count: usize,
    pub(crate) execution_time_ms: f64,
    pub(crate) rows_scanned: u64,
    /// JSON array of extracted node objects (deduplicated, metadata stripped).
    pub(crate) nodes_json: CString,
    /// JSON array of extracted edge objects (deduplicated, metadata stripped).
    pub(crate) edges_json: CString,
}

/// Structured node returned by CRUD operations.
pub struct GrafeoNode {
    pub(crate) id: u64,
    pub(crate) labels_json: CString,
    pub(crate) properties_json: CString,
}

/// Structured edge returned by CRUD operations.
pub struct GrafeoEdge {
    pub(crate) id: u64,
    pub(crate) source_id: u64,
    pub(crate) target_id: u64,
    pub(crate) edge_type: CString,
    pub(crate) properties_json: CString,
}

// ---------------------------------------------------------------------------
// Value ↔ JSON conversion
// ---------------------------------------------------------------------------

/// Convert a Grafeo `Value` to a `serde_json::Value`.
pub fn value_to_json(v: &Value) -> serde_json::Value {
    grafeo_bindings_common::json::value_to_json(v)
}

/// Convert a `serde_json::Value` to a Grafeo `Value`.
pub fn json_to_value(v: &serde_json::Value) -> Value {
    grafeo_bindings_common::json::json_to_value(v)
}

/// Serialize a [`PropertyMap`] to a JSON `CString`.
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
pub fn properties_to_json(props: &PropertyMap) -> CString {
    let obj: serde_json::Map<std::string::String, serde_json::Value> = props
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), value_to_json(v)))
        .collect();
    let json_str = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default();
    CString::new(json_str).unwrap_or_default()
}

/// Parse a JSON C-string into a `Vec<(PropertyKey, Value)>` for node/edge creation.
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
pub fn parse_properties(json_ptr: *const c_char) -> Option<Vec<(PropertyKey, Value)>> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    let obj = parsed.as_object()?;
    let props: Vec<(PropertyKey, Value)> = obj
        .iter()
        .map(|(k, v)| (PropertyKey::new(k.clone()), json_to_value(v)))
        .collect();
    Some(props)
}

/// Parse a JSON C-string into a `Vec<String>` (for labels).
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
pub fn parse_labels(json_ptr: *const c_char) -> Option<Vec<String>> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    let arr = parsed.as_array()?;
    Some(
        arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
    )
}

/// Parse a JSON C-string into a single `Value`.
#[cfg(any(
    feature = "lpg",
    feature = "compact-store",
    feature = "embedded",
    feature = "edge",
    feature = "native"
))]
pub fn parse_value(json_ptr: *const c_char) -> Option<Value> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(json_to_value(&parsed))
}

/// Parse a JSON C-string into a `HashMap<String, Value>` for query params.
pub fn parse_params(
    json_ptr: *const c_char,
) -> grafeo_common::Result<std::collections::HashMap<String, Value>> {
    use grafeo_common::utils::error::Error;
    if json_ptr.is_null() {
        return Ok(std::collections::HashMap::new());
    }
    // SAFETY: Caller guarantees a valid null-terminated C string.
    let json = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .map_err(|_| Error::InvalidValue("Parameters must be valid UTF-8".into()))?;
    let parsed: serde_json::Value = serde_json::from_str(json)
        .map_err(|error| Error::InvalidValue(format!("Invalid query parameters: {error}")))?;
    let object = parsed
        .as_object()
        .ok_or_else(|| Error::InvalidValue("Query parameters must be a JSON object".into()))?;
    Ok(object
        .iter()
        .map(|(key, value)| (key.clone(), json_to_value(value)))
        .collect())
}
