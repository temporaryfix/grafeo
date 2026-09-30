//! Bounded C row/chunk delivery with one native cursor and terminal cleanup.

use std::os::raw::c_char;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::{Mutex, RwLock};

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_core::execution::QueryCancellationHandle;
use grafeo_engine::OwnedRowIterator;
use grafeo_engine::database::{GrafeoDB, QueryResult};

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::execution::{
    GrafeoQueryOptions, bounded_columns_json, bounded_row_json, c_result_base_bytes,
    c_result_row_bytes, options_from_ptr,
};
use crate::types::{GrafeoDatabase, GrafeoResult};

struct CursorState {
    iter: OwnedRowIterator,
    pending: Option<Vec<Value>>,
    emitted: usize,
    terminal: bool,
    failure: Option<Error>,
    // Native cursor must drop before its database keepalive.
    _database: Arc<RwLock<GrafeoDB>>,
}

impl CursorState {
    fn fail(&mut self, primary: Error) -> GrafeoStatus {
        self.terminal = true;
        self.pending = None;
        let error = match self.iter.close() {
            Ok(()) => primary,
            Err(cleanup) => {
                primary.with_context(format!("C stream cleanup also failed: {cleanup}"))
            }
        };
        let status = set_error(&error);
        self.failure = Some(error);
        status
    }

    fn terminal_status(&self) -> GrafeoStatus {
        self.failure.as_ref().map_or(GrafeoStatus::Ok, set_error)
    }

    fn next(&mut self, max_rows: Option<usize>) -> Result<Option<Vec<Value>>, Error> {
        let row = match self.pending.take() {
            Some(row) => Some(Ok(row)),
            None => self.iter.next(),
        };
        match row {
            Some(Ok(row)) => {
                if max_rows.is_some_and(|limit| self.emitted >= limit) {
                    return Err(limit_error("C stream exceeds max_rows"));
                }
                Ok(Some(row))
            }
            Some(Err(error)) => Err(error),
            None => {
                self.terminal = true;
                Ok(None)
            }
        }
    }
}

fn limit_error(message: &str) -> Error {
    Error::Storage(StorageError::Full).with_context(message)
}

/// Opaque owned cursor. Concurrent pulls serialize. Close may run concurrently;
/// freeing the allocation requires all calls using this pointer to have ended.
pub struct GrafeoStream {
    columns: Vec<String>,
    max_rows: Option<usize>,
    max_bytes: usize,
    cancellation: QueryCancellationHandle,
    closing: AtomicBool,
    active: AtomicUsize,
    state: Mutex<CursorState>,
}

struct ActivePull<'a>(&'a AtomicUsize);
impl Drop for ActivePull<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_open(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoStream {
    grafeo_stream_open_with_options(db, query, std::ptr::null(), std::ptr::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_open_with_options(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
    options: *const GrafeoQueryOptions,
) -> *mut GrafeoStream {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    let Ok(query) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = if params_json.is_null() {
        std::collections::HashMap::new()
    } else {
        let Ok(raw) = str_from_ptr(params_json) else {
            return std::ptr::null_mut();
        };
        match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(raw) {
            Ok(values) => values
                .into_iter()
                .map(|(key, value)| (key, crate::types::json_to_value(&value)))
                .collect(),
            Err(error) => {
                set_error(&Error::Serialization(error.to_string()));
                return std::ptr::null_mut();
            }
        }
    };
    // SAFETY: caller retains the options/control allocation throughout this call.
    let mut native = match unsafe { options_from_ptr(options) } {
        Ok(native) => native,
        Err(error) => {
            set_error(&error);
            return std::ptr::null_mut();
        }
    };
    let limits = native.result_limits.unwrap_or_default();
    let max_rows = (!options.is_null()).then_some(limits.max_rows);
    native.result_admission = None;
    let cancellation = native.control.cancellation_handle();
    // SAFETY: caller retains a live database allocation throughout this call.
    let database = unsafe { &*db };
    let mut stream = match database
        .inner
        .read()
        .stream_with_options(query, params, native)
    {
        Ok(stream) => stream,
        Err(error) => {
            set_error(&error);
            return std::ptr::null_mut();
        }
    };
    if let Err(error) = bounded_columns_json(stream.columns(), limits.max_bytes) {
        let error = match stream.close() {
            Ok(()) => error,
            Err(cleanup) => error.with_context(format!("C stream cleanup also failed: {cleanup}")),
        };
        set_error(&error);
        return std::ptr::null_mut();
    }
    let columns = stream.columns().to_vec();
    Box::into_raw(Box::new(GrafeoStream {
        columns,
        max_rows,
        max_bytes: limits.max_bytes,
        cancellation,
        closing: AtomicBool::new(false),
        active: AtomicUsize::new(0),
        state: Mutex::new(CursorState {
            iter: stream.into_row_iter(),
            pending: None,
            emitted: 0,
            terminal: false,
            failure: None,
            _database: Arc::clone(&database.inner),
        }),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_columns_json(stream: *const GrafeoStream) -> *mut c_char {
    if stream.is_null() {
        set_last_error("Null stream pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: caller retains a live stream for the duration of this call.
    let stream = unsafe { &*stream };
    match bounded_columns_json(&stream.columns, stream.max_bytes) {
        Ok(json) => json.into_raw(),
        Err(error) => {
            set_error(&error);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_next_row_json(
    stream: *mut GrafeoStream,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    if out_json.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: caller supplies a writable out-pointer.
    unsafe {
        *out_json = std::ptr::null_mut();
    }
    if stream.is_null() {
        set_last_error("Null stream pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: caller retains a live stream for the duration of this call.
    let stream = unsafe { &*stream };
    stream.active.fetch_add(1, Ordering::SeqCst);
    let _active = ActivePull(&stream.active);
    let mut state = stream.state.lock();
    if state.terminal {
        return state.terminal_status();
    }
    if stream.closing.load(Ordering::SeqCst) {
        return GrafeoStatus::Ok;
    }
    // Pending rows need an explicit cancellation check before delivery too.
    if let Err(error) = state.iter.check_execution() {
        return state.fail(error);
    }
    match state.next(stream.max_rows) {
        Ok(Some(row)) => match bounded_row_json(&stream.columns, &row, stream.max_bytes) {
            Ok(json) => {
                let Some(emitted) = state.emitted.checked_add(1) else {
                    return state.fail(limit_error("C stream row count overflow"));
                };
                state.emitted = emitted;
                // SAFETY: output was checked above and owns this fresh string.
                unsafe {
                    *out_json = json.into_raw();
                }
                GrafeoStatus::Ok
            }
            Err(error) => state.fail(error),
        },
        Ok(None) => GrafeoStatus::Ok,
        Err(error) => state.fail(error),
    }
}

/// Delivers at most max_rows (capped at 1024) and max_bytes of copied result.
/// Row and chunk calls share the same cursor. NULL output denotes clean EOF.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_next_chunk(
    stream: *mut GrafeoStream,
    max_rows: usize,
    out_result: *mut *mut GrafeoResult,
) -> GrafeoStatus {
    if out_result.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: caller supplies a writable out-pointer.
    unsafe {
        *out_result = std::ptr::null_mut();
    }
    if stream.is_null() {
        set_last_error("Null stream pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    if max_rows == 0 {
        set_last_error("Chunk max_rows must be positive");
        return GrafeoStatus::ErrorDatabase;
    }
    // SAFETY: caller retains a live stream for the duration of this call.
    let stream = unsafe { &*stream };
    stream.active.fetch_add(1, Ordering::SeqCst);
    let _active = ActivePull(&stream.active);
    let mut state = stream.state.lock();
    if state.terminal {
        return state.terminal_status();
    }
    if stream.closing.load(Ordering::SeqCst) {
        return GrafeoStatus::Ok;
    }
    let mut bytes = match c_result_base_bytes(&stream.columns) {
        Ok(bytes) => bytes,
        Err(error) => return state.fail(error),
    };
    if bytes > stream.max_bytes {
        return state.fail(limit_error("C chunk schema exceeds max_bytes"));
    }
    let minimum_row = match c_result_row_bytes(&stream.columns, &[]) {
        Ok(cost) => cost,
        Err(error) => return state.fail(error),
    };
    let Some(minimum_slot) = minimum_row.checked_add(std::mem::size_of::<Vec<Value>>()) else {
        return state.fail(limit_error("C chunk row size overflow"));
    };
    let capacity = max_rows
        .min(1024)
        .min((stream.max_bytes - bytes) / minimum_slot);
    if capacity == 0 {
        return state.fail(limit_error("C chunk row allocation exceeds max_bytes"));
    }
    bytes += capacity * std::mem::size_of::<Vec<Value>>();
    let mut rows = Vec::new();
    if rows.try_reserve_exact(capacity).is_err() {
        return state.fail(limit_error("C chunk row allocation failed"));
    }
    for _ in 0..capacity {
        if let Err(error) = state.iter.check_execution() {
            return state.fail(error);
        }
        let row = match state.next(stream.max_rows) {
            Ok(Some(row)) => row,
            Ok(None) => break,
            Err(error) => return state.fail(error),
        };
        let cost = match c_result_row_bytes(&stream.columns, &row) {
            Ok(cost) => cost,
            Err(error) => return state.fail(error),
        };
        let Some(next) = bytes.checked_add(cost) else {
            return state.fail(limit_error("C chunk size overflow"));
        };
        if next > stream.max_bytes {
            if rows.is_empty() {
                return state.fail(limit_error("C chunk row exceeds max_bytes"));
            }
            state.pending = Some(row);
            break;
        }
        bytes = next;
        let Some(emitted) = state.emitted.checked_add(1) else {
            return state.fail(limit_error("C stream row count overflow"));
        };
        state.emitted = emitted;
        rows.push(row);
    }
    if rows.is_empty() {
        return GrafeoStatus::Ok;
    }
    let result = QueryResult::from_rows(stream.columns.clone(), rows);
    match crate::database::build_result(&result, stream.max_bytes) {
        Ok(result) => {
            // SAFETY: output was checked above and owns this result allocation.
            unsafe {
                *out_result = Box::into_raw(Box::new(result));
            }
            GrafeoStatus::Ok
        }
        Err(error) => state.fail(error),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_close(stream: *mut GrafeoStream) -> GrafeoStatus {
    if stream.is_null() {
        return GrafeoStatus::Ok;
    }
    // SAFETY: caller retains a live stream until all concurrent calls complete.
    let stream = unsafe { &*stream };
    stream.closing.store(true, Ordering::SeqCst);
    if stream.active.load(Ordering::SeqCst) != 0 {
        stream.cancellation.cancel();
    }
    let mut state = stream.state.lock();
    if state.terminal {
        return state.terminal_status();
    }
    state.terminal = true;
    state.pending = None;
    match state.iter.close() {
        Ok(()) => GrafeoStatus::Ok,
        Err(error) => {
            let status = set_error(&error);
            state.failure = Some(error);
            status
        }
    }
}

/// Null-safe. The caller must have ended every concurrent use of this pointer.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_free(stream: *mut GrafeoStream) {
    if !stream.is_null() {
        let _ = grafeo_stream_close(stream);
        // SAFETY: caller transfers its one allocation exactly once.
        drop(unsafe { Box::from_raw(stream) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::{
        grafeo_close, grafeo_free_database, grafeo_free_result, grafeo_open_memory,
    };
    use crate::execution::{
        grafeo_cancel, grafeo_cancel_handle_free, grafeo_query_control_cancel_handle,
        grafeo_query_control_create, grafeo_query_control_free,
    };

    fn fixture(count: usize) -> *mut GrafeoDatabase {
        let db = grafeo_open_memory();
        // SAFETY: fresh allocation remains owned by this test.
        let native = unsafe { &*db };
        let guard = native.inner.read();
        for value in 0..count {
            let id = guard.create_node(&["Item"]);
            guard
                .set_node_property(
                    id,
                    "value",
                    Value::Int64(i64::try_from(value).expect("fixture value fits i64")),
                )
                .unwrap();
        }
        db
    }

    #[test]
    fn c_stream_chunks_rows_and_close_share_one_cursor() {
        let db = fixture(3075);
        let stream = grafeo_stream_open(db, c"MATCH (n:Item) RETURN n.value AS value".as_ptr());
        assert!(!stream.is_null());
        assert_ne!(
            grafeo_close(db),
            GrafeoStatus::Ok,
            "busy close must return without waiting"
        );
        let mut row = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_row_json(stream, &raw mut row),
            GrafeoStatus::Ok
        );
        assert!(!row.is_null());
        // SAFETY: returned string transfers ownership to this caller.
        drop(unsafe { std::ffi::CString::from_raw(row) });
        let mut values = std::collections::BTreeSet::new();
        let mut count = 1;
        loop {
            let mut result = std::ptr::null_mut();
            assert_eq!(
                grafeo_stream_next_chunk(stream, 257, &raw mut result),
                GrafeoStatus::Ok
            );
            if result.is_null() {
                break;
            }
            // SAFETY: fresh result remains owned until free below.
            let result_ref = unsafe { &*result };
            let rows: Vec<serde_json::Value> =
                serde_json::from_slice(result_ref.json.as_bytes()).unwrap();
            assert!(!rows.is_empty() && rows.len() <= 257);
            count += rows.len();
            for row in rows {
                assert!(values.insert(row["value"].as_i64().unwrap()));
            }
            grafeo_free_result(result);
        }
        assert_eq!(count, 3075);
        assert_eq!(grafeo_stream_close(stream), GrafeoStatus::Ok);
        assert_eq!(grafeo_stream_close(stream), GrafeoStatus::Ok);
        grafeo_stream_free(stream);
        assert_eq!(grafeo_close(db), GrafeoStatus::Ok);
        grafeo_free_database(db);
    }

    #[test]
    fn c_stream_cancel_buffered_rows_has_sticky_error_and_isolated_owner() {
        let db = fixture(1100);
        let control = grafeo_query_control_create(-1);
        // SAFETY: control is retained until opening has consumed its owner.
        let handle = unsafe { grafeo_query_control_cancel_handle(control) };
        let options = GrafeoQueryOptions {
            control,
            max_rows: usize::MAX,
            max_bytes: 8000,
            language: std::ptr::null(),
        };
        let stream = grafeo_stream_open_with_options(
            db,
            c"MATCH (n:Item) RETURN n.value AS value".as_ptr(),
            std::ptr::null(),
            &raw const options,
        );
        assert!(!stream.is_null());
        let other = grafeo_stream_open(db, c"RETURN 7 AS value".as_ptr());
        assert!(!other.is_null());
        let mut chunk = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_chunk(stream, 1024, &raw mut chunk),
            GrafeoStatus::Ok
        );
        assert!(!chunk.is_null());
        grafeo_free_result(chunk);
        // SAFETY: independent handles remain live for their individual calls.
        unsafe {
            grafeo_query_control_free(control);
            assert_eq!(grafeo_cancel(handle), GrafeoStatus::Ok);
        }
        let mut row = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_row_json(stream, &raw mut row),
            GrafeoStatus::ErrorCancelled
        );
        assert!(row.is_null());
        assert_eq!(
            grafeo_stream_next_chunk(stream, 1, &raw mut chunk),
            GrafeoStatus::ErrorCancelled
        );
        assert!(chunk.is_null());
        assert_eq!(grafeo_stream_close(stream), GrafeoStatus::ErrorCancelled);
        assert_eq!(grafeo_stream_close(stream), GrafeoStatus::ErrorCancelled);
        assert_eq!(
            grafeo_stream_next_row_json(other, &raw mut row),
            GrafeoStatus::Ok
        );
        assert!(!row.is_null());
        // SAFETY: own returned string and separate cancel handle exactly once.
        unsafe {
            drop(std::ffi::CString::from_raw(row));
            grafeo_cancel_handle_free(handle);
        }
        grafeo_stream_free(other);
        grafeo_stream_free(stream);
        grafeo_free_database(db);
    }

    #[test]
    fn c_stream_row_limit_errors_without_truncating_successfully() {
        let db = fixture(3);
        let options = GrafeoQueryOptions {
            control: std::ptr::null_mut(),
            max_rows: 2,
            max_bytes: 65536,
            language: std::ptr::null(),
        };
        let stream = grafeo_stream_open_with_options(
            db,
            c"MATCH (n:Item) RETURN n.value AS value".as_ptr(),
            std::ptr::null(),
            &raw const options,
        );
        assert!(!stream.is_null());
        for _ in 0..2 {
            let mut row = std::ptr::null_mut();
            assert_eq!(
                grafeo_stream_next_row_json(stream, &raw mut row),
                GrafeoStatus::Ok
            );
            // SAFETY: fresh returned string belongs to this test.
            drop(unsafe { std::ffi::CString::from_raw(row) });
        }
        let mut row = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_row_json(stream, &raw mut row),
            GrafeoStatus::ErrorResourceLimit
        );
        assert!(row.is_null());
        assert_eq!(
            grafeo_stream_close(stream),
            GrafeoStatus::ErrorResourceLimit
        );
        grafeo_stream_free(stream);
        grafeo_free_database(db);
    }

    #[test]
    fn c_stream_explicit_close_discards_buffer_and_allows_fresh_query() {
        let db = fixture(1100);
        let stream = grafeo_stream_open(db, c"MATCH (n:Item) RETURN n.value".as_ptr());
        assert!(!stream.is_null());
        let mut row = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_row_json(stream, &raw mut row),
            GrafeoStatus::Ok
        );
        // SAFETY: test owns returned string.
        drop(unsafe { std::ffi::CString::from_raw(row) });
        assert_eq!(grafeo_stream_close(stream), GrafeoStatus::Ok);
        assert_eq!(grafeo_stream_close(stream), GrafeoStatus::Ok);
        assert_eq!(
            grafeo_stream_next_row_json(stream, &raw mut row),
            GrafeoStatus::Ok
        );
        assert!(row.is_null());
        let next = grafeo_stream_open(db, c"RETURN 1".as_ptr());
        assert!(!next.is_null());
        grafeo_stream_free(next);
        grafeo_stream_free(stream);
        grafeo_free_database(db);
    }
    #[test]
    fn c_stream_active_cancel_and_close_join_before_free() {
        for close in [false, true] {
            let db = fixture(300);
            let control = grafeo_query_control_create(-1);
            // SAFETY: live control allocation; cancel handle is independently owned.
            let cancel = unsafe { grafeo_query_control_cancel_handle(control) };
            let options = GrafeoQueryOptions {
                control,
                max_rows: usize::MAX,
                max_bytes: 65536,
                language: std::ptr::null(),
            };
            let stream = grafeo_stream_open_with_options(db,
                c"MATCH (a:Item), (b:Item), (c:Item) WHERE a.value + b.value + c.value < 0 RETURN a.value".as_ptr(),
                std::ptr::null(), &raw const options);
            assert!(!stream.is_null());
            let worker_pointer = stream as usize;
            let worker = std::thread::spawn(move || {
                let mut row = std::ptr::null_mut();
                let status =
                    grafeo_stream_next_row_json(worker_pointer as *mut GrafeoStream, &raw mut row);
                assert!(row.is_null());
                status
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            // SAFETY: test retains the allocation until both calls have joined.
            let active = unsafe { &(*stream).active };
            while active.load(Ordering::SeqCst) == 0
                && !worker.is_finished()
                && std::time::Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            assert!(
                active.load(Ordering::SeqCst) > 0,
                "native pull must be active"
            );
            if close {
                assert_eq!(grafeo_stream_close(stream), GrafeoStatus::ErrorCancelled);
            } else {
                // SAFETY: cancel allocation is retained until cancellation returns.
                assert_eq!(unsafe { grafeo_cancel(cancel) }, GrafeoStatus::Ok);
            }
            assert_eq!(worker.join().unwrap(), GrafeoStatus::ErrorCancelled);
            // SAFETY: every call is joined before the independent handles are freed.
            unsafe {
                grafeo_cancel_handle_free(cancel);
                grafeo_query_control_free(control);
            }
            grafeo_stream_free(stream);
            grafeo_free_database(db);
        }
    }
    #[test]
    fn c_stream_deadline_rejects_pending_row_and_precancel_rejects_open() {
        let db = fixture(1100);
        let control = grafeo_query_control_create(500);
        let options = GrafeoQueryOptions {
            control,
            max_rows: usize::MAX,
            max_bytes: 8000,
            language: std::ptr::null(),
        };
        let stream = grafeo_stream_open_with_options(
            db,
            c"MATCH (n:Item) RETURN n.value AS value".as_ptr(),
            std::ptr::null(),
            &raw const options,
        );
        assert!(!stream.is_null());
        let mut chunk = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_chunk(stream, 1024, &raw mut chunk),
            GrafeoStatus::Ok
        );
        assert!(!chunk.is_null());
        grafeo_free_result(chunk);
        std::thread::sleep(std::time::Duration::from_millis(550));
        let mut row = std::ptr::null_mut();
        assert_eq!(
            grafeo_stream_next_row_json(stream, &raw mut row),
            GrafeoStatus::ErrorDeadline
        );
        assert!(row.is_null());
        grafeo_stream_free(stream);
        // SAFETY: opening consumed the owner; its allocation is still retained.
        unsafe {
            grafeo_query_control_free(control);
        }
        let control = grafeo_query_control_create(-1);
        // SAFETY: each independently owned allocation remains live until its free.
        unsafe {
            let cancel = grafeo_query_control_cancel_handle(control);
            assert_eq!(grafeo_cancel(cancel), GrafeoStatus::Ok);
            let options = GrafeoQueryOptions {
                control,
                max_rows: 10,
                max_bytes: 65536,
                language: std::ptr::null(),
            };
            assert!(
                grafeo_stream_open_with_options(
                    db,
                    c"RETURN 1".as_ptr(),
                    std::ptr::null(),
                    &raw const options
                )
                .is_null()
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(crate::error::grafeo_last_error_code()).to_bytes(),
                b"GRAFEO-Q007"
            );
            grafeo_cancel_handle_free(cancel);
            grafeo_query_control_free(control);
        }
        grafeo_free_database(db);
    }
}
