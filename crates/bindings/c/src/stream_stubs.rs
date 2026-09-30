//! Streaming symbols for builds without GQL plus LPG so the C header always links.

use std::os::raw::c_char;

use crate::error::{GrafeoStatus, set_last_error};
use crate::execution::GrafeoQueryOptions;
use crate::types::{GrafeoDatabase, GrafeoResult};

#[repr(C)]
pub struct GrafeoStream {
    _private: [u8; 0],
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_open(
    _db: *mut GrafeoDatabase,
    _query: *const c_char,
) -> *mut GrafeoStream {
    set_last_error("Streaming requires gql and LPG support");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_columns_json(_stream: *const GrafeoStream) -> *mut c_char {
    set_last_error("Streaming requires gql and LPG support");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_next_row_json(
    _stream: *mut GrafeoStream,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    if !out_json.is_null() {
        // SAFETY: caller-owned out pointer.
        unsafe { *out_json = std::ptr::null_mut() };
    }
    set_last_error("Streaming requires gql and LPG support");
    GrafeoStatus::ErrorQuery
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_free(_stream: *mut GrafeoStream) {}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_open_with_options(
    _db: *mut GrafeoDatabase,
    _query: *const c_char,
    _params_json: *const c_char,
    _options: *const GrafeoQueryOptions,
) -> *mut GrafeoStream {
    crate::error::set_error(&grafeo_common::utils::error::Error::Query(
        grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Unsupported,
            "Streaming requires gql and LPG support",
        ),
    ));
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_close(_stream: *mut GrafeoStream) -> GrafeoStatus {
    GrafeoStatus::Ok
}

#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_next_chunk(
    _stream: *mut GrafeoStream,
    _max_rows: usize,
    out_result: *mut *mut GrafeoResult,
) -> GrafeoStatus {
    if !out_result.is_null() {
        // SAFETY: caller provides a writable output pointer.
        unsafe {
            *out_result = std::ptr::null_mut();
        }
    }
    crate::error::set_error(&grafeo_common::utils::error::Error::Query(
        grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Unsupported,
            "Streaming requires gql and LPG support",
        ),
    ))
}
