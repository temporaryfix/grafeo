//! Converts Rust errors to JavaScript exceptions.
//!
//! Type errors and invalid arguments become `InvalidArg` status errors,
//! while database, query, and transaction errors become `GenericFailure`.

use napi::Status;
use napi::bindgen_prelude::{Env, PromiseRaw, ToNapiValue, Unknown};
use thiserror::Error;

/// Grafeo errors that translate to JavaScript Error instances.
#[derive(Error, Debug)]
pub enum NodeGrafeoError {
    /// Preserve the native taxonomy until the JavaScript rejection is constructed.
    #[error("{0}")]
    Native(grafeo_common::utils::error::Error),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Transaction error: {0}")]
    Transaction(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<NodeGrafeoError> for napi::Error {
    fn from(err: NodeGrafeoError) -> Self {
        match &err {
            NodeGrafeoError::Native(native) => napi::Error::new(
                Status::GenericFailure,
                format!("{}: {native}", native.error_code()),
            ),
            NodeGrafeoError::InvalidArgument(_) | NodeGrafeoError::Type(_) => {
                napi::Error::new(Status::InvalidArg, err.to_string())
            }
            NodeGrafeoError::Database(_)
            | NodeGrafeoError::Query(_)
            | NodeGrafeoError::Transaction(_) => {
                napi::Error::new(Status::GenericFailure, err.to_string())
            }
        }
    }
}

impl From<grafeo_common::utils::error::Error> for NodeGrafeoError {
    fn from(error: grafeo_common::utils::error::Error) -> Self {
        Self::Native(error)
    }
}

pub(crate) type NodeResult<T> = std::result::Result<T, NodeGrafeoError>;

/// Constructs the native code on the JavaScript thread, without parsing text.
pub(crate) fn native_to_js_error(
    env: &Env,
    error: grafeo_common::utils::error::Error,
) -> napi::Error {
    let code = error.error_code().to_string();
    let reason = format!("{code}: {error}");
    let error = napi::JsError::from(napi::Error::new(code, reason));
    // SAFETY: this function runs in the napi callback on its owning JS thread;
    // into_value returns a live Error and the reference remains on that thread.
    let raw = unsafe { error.into_value(env.raw()) };
    let value = unsafe { Unknown::from_raw_unchecked(env.raw(), raw) };
    napi::Error::from(value)
}

/// Only Rust-owned outcomes cross worker threads. Native failure is turned into
/// a JavaScript Error with its stable code when the promise settles on JS.
pub(crate) fn spawn_execution<T, F>(env: &Env, future: F) -> napi::Result<PromiseRaw<'_, T>>
where
    T: ToNapiValue + Send + 'static,
    F: std::future::Future<Output = NodeResult<T>> + Send + 'static,
{
    env.spawn_future_with_callback(async move { Ok(future.await) }, |env, outcome| {
        outcome.map_err(|error| match error {
            NodeGrafeoError::Native(native) => native_to_js_error(env, native),
            other => other.into(),
        })
    })
}
