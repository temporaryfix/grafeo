//! Owned bounded CDC pages at the native C boundary.

use std::ffi::CString;
use std::os::raw::c_char;

use grafeo_common::types::{DurableCursor, EdgeId, EpochId, NodeId};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_engine::cdc::EntityHistoryQuery;

use crate::error::{set_error, set_last_error};
use crate::types::GrafeoDatabase;

/// An owned page, independent of its database after the read completes.
pub struct GrafeoChangePage {
    events: CString,
    cursor: [u8; DurableCursor::LEN],
    event_count: usize,
}

fn decode_cursor(cursor: *const u8, length: usize) -> Result<Option<DurableCursor>> {
    if cursor.is_null() && length == 0 {
        return Ok(None);
    }
    if cursor.is_null() || length != DurableCursor::LEN {
        return Err(StorageError::CursorInvalid.into());
    }
    // SAFETY: The caller supplies length readable bytes for this call. Reject
    // every noncanonical length before forming or reading the fixed-size slice.
    let bytes = unsafe { std::slice::from_raw_parts(cursor, DurableCursor::LEN) };
    DurableCursor::from_bytes(bytes).map(Some)
}

fn read_page(
    db: *mut GrafeoDatabase,
    cursor: *const u8,
    cursor_len: usize,
    max_events: usize,
    max_bytes: usize,
    query: Option<EntityHistoryQuery>,
) -> *mut GrafeoChangePage {
    // SAFETY: The database allocation remains live throughout the call.
    let Some(db) = (unsafe { db.as_ref() }) else {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    };
    let result = (|| -> Result<GrafeoChangePage> {
        let cursor = decode_cursor(cursor, cursor_len)?;
        let page = {
            let db = db.inner.read();
            let session = db.session();
            match query {
                Some(query) => {
                    session.history_after(&query, cursor.as_ref(), max_events, max_bytes)
                }
                None => session.changes_after(cursor.as_ref(), max_events, max_bytes),
            }?
        };
        let events: Vec<_> = page
            .events
            .iter()
            .map(grafeo_bindings_common::cdc::change_event_to_json)
            .collect();
        let json = serde_json::to_string(&events)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        let events = CString::new(json).map_err(|error| Error::Serialization(error.to_string()))?;
        Ok(GrafeoChangePage {
            events,
            cursor: page.next.to_bytes(),
            event_count: page.events.len(),
        })
    })();
    match result {
        Ok(page) => Box::into_raw(Box::new(page)),
        Err(error) => {
            set_error(&error);
            std::ptr::null_mut()
        }
    }
}

/// Reads an owned whole-feed page, or returns null with a structured error.
///
/// `(NULL, 0)` starts at the retained floor; other cursors must contain exactly
/// 97 readable bytes. Both limits are positive; bytes count native event
/// encodings, excluding JSON/page envelopes. All pointers remain live during
/// this call. Free the returned page with `grafeo_free_change_page`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_changes_after(
    db: *mut GrafeoDatabase,
    cursor: *const u8,
    cursor_len: usize,
    max_events: usize,
    max_bytes: usize,
) -> *mut GrafeoChangePage {
    read_page(db, cursor, cursor_len, max_events, max_bytes, None)
}

/// Reads bounded node history with an inclusive minimum epoch.
/// Pointer, cursor, ownership and limit rules match `grafeo_changes_after`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_history_after(
    db: *mut GrafeoDatabase,
    node_id: u64,
    since_epoch: u64,
    cursor: *const u8,
    cursor_len: usize,
    max_events: usize,
    max_bytes: usize,
) -> *mut GrafeoChangePage {
    let mut query = EntityHistoryQuery::new(NodeId::new(node_id));
    query.since_epoch = EpochId::new(since_epoch);
    read_page(db, cursor, cursor_len, max_events, max_bytes, Some(query))
}

/// Reads bounded edge history with an inclusive minimum epoch.
/// Pointer, cursor, ownership and limit rules match `grafeo_changes_after`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_history_after(
    db: *mut GrafeoDatabase,
    edge_id: u64,
    since_epoch: u64,
    cursor: *const u8,
    cursor_len: usize,
    max_events: usize,
    max_bytes: usize,
) -> *mut GrafeoChangePage {
    let mut query = EntityHistoryQuery::new(EdgeId::new(edge_id));
    query.since_epoch = EpochId::new(since_epoch);
    read_page(db, cursor, cursor_len, max_events, max_bytes, Some(query))
}

/// Returns borrowed event JSON, valid until this page is freed; null on null.
/// Every native coordinate is an exact decimal string, including endpoints.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_change_page_events_json(page: *const GrafeoChangePage) -> *const c_char {
    // SAFETY: The caller retains this page throughout the accessor and its use.
    unsafe { page.as_ref() }.map_or(std::ptr::null(), |page| page.events.as_ptr())
}

/// Returns 97 borrowed cursor bytes, valid until this page is freed; null on null.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_change_page_cursor(page: *const GrafeoChangePage) -> *const u8 {
    // SAFETY: The caller retains this page throughout the accessor and its use.
    unsafe { page.as_ref() }.map_or(std::ptr::null(), |page| page.cursor.as_ptr())
}

/// Returns the number of owned events; zero on a null page.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_change_page_event_count(page: *const GrafeoChangePage) -> usize {
    // SAFETY: The caller retains the page throughout the call.
    unsafe { page.as_ref() }.map_or(0, |page| page.event_count)
}

/// Frees an owned page. Null is allowed; a non-null allocation is freed once.
/// No accessor or read of its borrowed JSON/cursor may race this call.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_change_page(page: *mut GrafeoChangePage) {
    if !page.is_null() {
        // SAFETY: The caller transfers its one owned page allocation back.
        unsafe { drop(Box::from_raw(page)) };
    }
}

#[cfg(all(
    test,
    any(
        feature = "lpg",
        feature = "compact-store",
        feature = "embedded",
        feature = "edge",
        feature = "native"
    )
))]
mod tests {
    use super::*;
    use crate::database::*;
    use crate::error::{GrafeoStatus, grafeo_last_error_code};
    use grafeo_common::utils::error::ErrorCode;
    use grafeo_engine::{Config, GrafeoDB};
    use parking_lot::RwLock;
    use std::ffi::CStr;
    use std::sync::Arc;

    fn database(config: Config) -> *mut GrafeoDatabase {
        Box::into_raw(Box::new(GrafeoDatabase {
            inner: Arc::new(RwLock::new(GrafeoDB::with_config(config).unwrap())),
        }))
    }

    fn close(db: *mut GrafeoDatabase) {
        assert_eq!(grafeo_close(db), GrafeoStatus::Ok);
        grafeo_free_database(db);
    }

    fn events(page: *const GrafeoChangePage) -> serde_json::Value {
        assert!(!page.is_null());
        // SAFETY: The test retains its page allocation throughout this read.
        let json = unsafe { CStr::from_ptr(grafeo_change_page_events_json(page)) };
        serde_json::from_slice(json.to_bytes()).unwrap()
    }

    fn cursor(page: *const GrafeoChangePage) -> [u8; DurableCursor::LEN] {
        assert!(!page.is_null());
        // SAFETY: A live page owns exactly DurableCursor::LEN readable bytes.
        unsafe { std::slice::from_raw_parts(grafeo_change_page_cursor(page), DurableCursor::LEN) }
            .try_into()
            .unwrap()
    }

    fn error_is(code: ErrorCode) {
        // SAFETY: The preceding failing C call installed a thread-local code.
        assert_eq!(
            unsafe { CStr::from_ptr(grafeo_last_error_code()) }
                .to_str()
                .unwrap(),
            code.as_str()
        );
    }

    fn node(db: *mut GrafeoDatabase) -> u64 {
        let id = grafeo_create_node(db, c"[\"N\"]".as_ptr(), std::ptr::null());
        assert_ne!(id, u64::MAX);
        id
    }

    #[test]
    fn cdc_pages_own_complete_creation_events_across_database_free() {
        let db = database(Config::in_memory().with_cdc());
        let a = node(db);
        let b = node(db);
        let edge = grafeo_create_edge(db, a, b, c"LINK".as_ptr(), std::ptr::null());
        assert_ne!(edge, u64::MAX);
        let mut next = None;
        let mut owned = Vec::new();
        for expected in [a, b, edge] {
            let (ptr, len) = next
                .as_ref()
                .map_or((std::ptr::null(), 0), |bytes: &[u8; 97]| {
                    (bytes.as_ptr(), bytes.len())
                });
            let page = grafeo_changes_after(db, ptr, len, 1, 4096);
            assert_eq!(grafeo_change_page_event_count(page), 1);
            let event = events(page);
            assert_eq!(event[0]["entity_id"], expected.to_string());
            assert!(event[0]["graph_incarnation"].is_string());
            next = Some(cursor(page));
            owned.push(page);
        }
        let next = next.unwrap();
        let eof = grafeo_changes_after(db, next.as_ptr(), next.len(), 1, 4096);
        assert_eq!(grafeo_change_page_event_count(eof), 0);
        assert_eq!(cursor(eof), next);
        grafeo_free_change_page(eof);
        close(db);
        assert_eq!(events(owned[0])[0]["labels"], serde_json::json!(["N"]));
        let event = events(owned[2]);
        assert_eq!(event[0]["edge_type"], "LINK");
        assert_eq!(event[0]["src_id"], a.to_string());
        assert_eq!(event[0]["dst_id"], b.to_string());
        assert_eq!(cursor(owned[2]), next);
        for page in owned {
            grafeo_free_change_page(page);
        }
        grafeo_free_change_page(std::ptr::null_mut());
        assert!(grafeo_change_page_cursor(std::ptr::null()).is_null());
        assert!(grafeo_change_page_events_json(std::ptr::null()).is_null());
        assert_eq!(grafeo_change_page_event_count(std::ptr::null()), 0);
    }

    #[test]
    fn cdc_errors_reject_invalid_pointers_lengths_limits_and_foreign_cursors() {
        let db = database(Config::in_memory().with_cdc());
        node(db);
        assert!(grafeo_changes_after(std::ptr::null_mut(), std::ptr::null(), 0, 1, 4096).is_null());
        for (ptr, len) in [
            (std::ptr::null(), 1),
            (std::ptr::dangling(), 0),
            (std::ptr::dangling(), usize::MAX),
        ] {
            assert!(grafeo_changes_after(db, ptr, len, 1, 4096).is_null());
            error_is(ErrorCode::CursorInvalid);
        }
        let malformed = [0_u8; DurableCursor::LEN];
        assert!(grafeo_changes_after(db, malformed.as_ptr(), malformed.len(), 1, 4096).is_null());
        error_is(ErrorCode::CursorInvalid);
        for (rows, bytes) in [(0, 4096), (1, 0)] {
            assert!(grafeo_changes_after(db, std::ptr::null(), 0, rows, bytes).is_null());
            error_is(ErrorCode::InvalidInput);
        }
        assert!(grafeo_changes_after(db, std::ptr::null(), 0, 1, 1).is_null());
        error_is(ErrorCode::StorageFull);
        let page = grafeo_changes_after(db, std::ptr::null(), 0, 1, 4096);
        let next = cursor(page);
        grafeo_free_change_page(page);
        let foreign = database(Config::in_memory().with_cdc());
        assert!(grafeo_changes_after(foreign, next.as_ptr(), next.len(), 1, 4096).is_null());
        error_is(ErrorCode::CursorForeign);
        close(foreign);
        assert_eq!(grafeo_close(db), GrafeoStatus::Ok);
        assert!(grafeo_changes_after(db, std::ptr::null(), 0, 1, 4096).is_null());
        error_is(ErrorCode::TransactionInvalidState);
        grafeo_free_database(db);
    }

    #[test]
    fn cdc_entity_selectors_and_eviction_keep_native_error_categories() {
        let mut config = Config::in_memory().with_cdc();
        config.cdc_retention.max_epochs = None;
        config.cdc_retention.max_events = Some(2);
        let db = database(config);
        let a = node(db);
        let page = grafeo_node_history_after(db, a, 0, std::ptr::null(), 0, 1, 4096);
        let next = cursor(page);
        let epoch = DurableCursor::from_bytes(&next).unwrap().epoch.as_u64();
        grafeo_free_change_page(page);
        let filtered = grafeo_node_history_after(db, a, epoch + 1, std::ptr::null(), 0, 1, 4096);
        assert_eq!(grafeo_change_page_event_count(filtered), 0);
        grafeo_free_change_page(filtered);
        let absent = grafeo_node_history_after(db, u64::MAX - 1, 0, std::ptr::null(), 0, 1, 4096);
        assert!(!absent.is_null());
        assert_eq!(grafeo_change_page_event_count(absent), 0);
        grafeo_free_change_page(absent);
        let b = node(db);
        let edge = grafeo_create_edge(db, a, b, c"LINK".as_ptr(), std::ptr::null());
        node(db);
        // Retention is an explicit maintenance transition, not a write side
        // effect. The live C fixture uses the existing native maintenance owner.
        unsafe { &*db }.inner.read().gc().unwrap();
        assert!(grafeo_changes_after(db, next.as_ptr(), next.len(), 1, 4096).is_null());
        error_is(ErrorCode::CursorEvicted);
        let edge_page = grafeo_edge_history_after(db, edge, 0, std::ptr::null(), 0, 1, 4096);
        assert_eq!(events(edge_page)[0]["entity_type"], "edge");
        assert_eq!(events(edge_page)[0]["src_id"], a.to_string());
        grafeo_free_change_page(edge_page);
        // Optional cross-language qualification witness: Go opens this exact
        // persisted cut and presents the cursor created before its floor moved.
        #[cfg(any(feature = "storage", feature = "embedded", feature = "native"))]
        if let Some(path) = std::env::var_os("GRAFEO_CDC_EVICTED_FIXTURE") {
            let path = std::path::PathBuf::from(path);
            unsafe { &*db }.inner.read().save(&path).unwrap();
            std::fs::write(path.with_extension("cursor"), next).unwrap();
        }
        close(db);
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn cdc_json_preserves_named_rdf_creation_terms() {
        let db = grafeo_open_memory_model(2);
        assert!(!db.is_null());
        grafeo_set_cdc_enabled(db, true);
        let result = grafeo_execute_sparql(
            db,
            c"INSERT DATA { GRAPH <urn:graph> { <urn:subject> <urn:predicate> \"value\" . } }"
                .as_ptr(),
        );
        assert!(!result.is_null());
        grafeo_free_result(result);
        let page = grafeo_changes_after(db, std::ptr::null(), 0, 1, 4096);
        let event = events(page);
        assert_eq!(event[0]["entity_type"], "triple");
        assert_eq!(event[0]["triple_graph"], "urn:graph");
        assert_eq!(event[0]["triple_subject"], "<urn:subject>");
        assert_eq!(event[0]["triple_predicate"], "<urn:predicate>");
        assert_eq!(event[0]["triple_object"], "\"value\"");
        assert!(event[0]["lpg_graph"].is_null());
        assert!(event[0]["labels"].is_null());
        assert!(event[0]["src_id"].is_null());
        grafeo_free_change_page(page);
        close(db);
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn cdc_cursor_resumes_one_event_across_two_directory_reopens() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("grafeo-c-cdc-{}-{unique}", std::process::id()));
        let name = CString::new(path.to_str().unwrap()).unwrap();
        let db = grafeo_open(name.as_ptr());
        assert!(!db.is_null());
        grafeo_set_cdc_enabled(db, true);
        let ids = [node(db), node(db), node(db)];
        let first = grafeo_changes_after(db, std::ptr::null(), 0, 1, 4096);
        assert_eq!(events(first)[0]["entity_id"], ids[0].to_string());
        let mut next = cursor(first);
        grafeo_free_change_page(first);
        close(db);
        for id in &ids[1..] {
            let db = grafeo_open(name.as_ptr());
            assert!(!db.is_null());
            let page = grafeo_changes_after(db, next.as_ptr(), next.len(), 1, 4096);
            assert_eq!(events(page)[0]["entity_id"], id.to_string());
            next = cursor(page);
            grafeo_free_change_page(page);
            close(db);
        }
        std::fs::remove_dir_all(path).unwrap();
    }
}
