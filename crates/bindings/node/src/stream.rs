//! Bounded Node.js row conversion with retained native cursor cleanup.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::{Mutex, RwLock};
use serde_json::Value as JsonValue;

use grafeo_common::utils::error::{Error as NativeError, StorageError};
use grafeo_core::execution::QueryCancellationHandle;
use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{OwnedResultStream, OwnedRowIterator};

use crate::error::{NodeGrafeoError, NodeResult, native_to_js_error, spawn_execution};
use crate::types::{bounded_columns, bounded_row_to_json};

struct CursorState {
    iterator: OwnedRowIterator,
    terminal: bool,
    emitted: usize,
    max_rows: Option<usize>,
    // Drop after the native iterator; in-flight tasks retain this same owner.
    _database: Arc<RwLock<GrafeoDB>>,
}

impl CursorState {
    fn fail(&mut self, primary: NativeError) -> NativeError {
        self.terminal = true;
        match self.iterator.close() {
            Ok(()) => primary,
            Err(cleanup) => {
                primary.with_context(format!("Node stream cleanup also failed: {cleanup}"))
            }
        }
    }

    fn next(&mut self, columns: &[String], max_bytes: usize) -> NodeResult<Option<JsonValue>> {
        if self.terminal {
            return Ok(None);
        }
        match self.iterator.next() {
            Some(Ok(values)) => {
                // Poll one row beyond the limit to distinguish exact EOF from
                // truncation. No JSON copy is made for the rejected row.
                if self.max_rows.is_some_and(|limit| self.emitted >= limit) {
                    let primary = NativeError::Storage(StorageError::Full)
                        .with_context("Node result stream exceeds maxRows");
                    return Err(NodeGrafeoError::from(self.fail(primary)));
                }
                let Some(next) = self.emitted.checked_add(1) else {
                    let primary = NativeError::Storage(StorageError::Full)
                        .with_context("Node result stream row count overflow");
                    return Err(NodeGrafeoError::from(self.fail(primary)));
                };
                match bounded_row_to_json(columns, &values, max_bytes) {
                    Ok(row) => {
                        self.emitted = next;
                        Ok(Some(row))
                    }
                    Err(primary) => Err(NodeGrafeoError::from(self.fail(primary))),
                }
            }
            Some(Err(primary)) => {
                self.terminal = true;
                Err(NodeGrafeoError::from(primary))
            }
            None => {
                self.terminal = true;
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> NodeResult<()> {
        self.terminal = true;
        self.iterator.close().map_err(NodeGrafeoError::from)
    }
}

// Separate from the cursor mutex so the JS thread can interrupt a native
// pull without waiting for that pull to finish.
struct PullControl {
    close_requested: AtomicBool,
    active: AtomicBool,
    cancellation: QueryCancellationHandle,
}

struct ActivePull<'a>(&'a PullControl);
impl Drop for ActivePull<'_> {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::SeqCst);
    }
}

/// Async cursor with explicit close and per-row copied-output admission.
#[napi(js_name = "ResultStream")]
pub struct JsResultStream {
    state: Arc<Mutex<CursorState>>,
    control: Arc<PullControl>,
    // Independent immutable schema makes columns access nonblocking while a
    // worker owns the iterator mutex. Both construction and getter copies are
    // admitted by the same conservative conversion bound.
    columns: Arc<Vec<String>>,
    max_bytes: usize,
}

impl JsResultStream {
    pub(crate) fn new(
        database: Arc<RwLock<GrafeoDB>>,
        mut stream: OwnedResultStream,
        max_bytes: usize,
        max_rows: Option<usize>,
        cancellation: QueryCancellationHandle,
    ) -> NodeResult<Self> {
        let columns = match bounded_columns(stream.columns(), max_bytes) {
            Ok(columns) => columns,
            Err(primary) => {
                let primary = match stream.close() {
                    Ok(()) => primary,
                    Err(cleanup) => {
                        primary.with_context(format!("Node stream cleanup also failed: {cleanup}"))
                    }
                };
                return Err(NodeGrafeoError::from(primary));
            }
        };
        Ok(Self {
            state: Arc::new(Mutex::new(CursorState {
                iterator: stream.into_row_iter(),
                terminal: false,
                emitted: 0,
                max_rows,
                _database: database,
            })),
            control: Arc::new(PullControl {
                close_requested: AtomicBool::new(false),
                active: AtomicBool::new(false),
                cancellation,
            }),
            columns: Arc::new(columns),
            max_bytes,
        })
    }
}

#[napi]
impl JsResultStream {
    /// Column names; never waits for an in-flight native pull.
    #[napi(getter)]
    pub fn columns(&self, env: Env) -> Result<Vec<String>> {
        bounded_columns(&self.columns, self.max_bytes)
            .map_err(|error| native_to_js_error(&env, error))
    }

    /// Resolves to the next row object, or null at EOF; terminal errors emit once.
    #[napi(ts_return_type = "Promise<JsonValue | null>")]
    pub fn next<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, Option<JsonValue>>> {
        let state = Arc::clone(&self.state);
        let control = Arc::clone(&self.control);
        let columns = Arc::clone(&self.columns);
        let max_bytes = self.max_bytes;
        spawn_execution(env, async move {
            tokio::task::spawn_blocking(move || {
                let mut state = state.lock();
                if state.terminal {
                    return Ok(None);
                }
                control.active.store(true, Ordering::SeqCst);
                let _active = ActivePull(&control);
                // close either observes active and cancels, or wins before
                // this check and prevents native work from starting.
                if control.close_requested.load(Ordering::SeqCst) {
                    return Ok(None);
                }
                state.next(&columns, max_bytes)
            })
            .await
            .map_err(|error| NodeGrafeoError::Database(error.to_string()))?
        })
    }

    /// Releases the native query. Repeated calls retain its cleanup resolution.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        self.control.close_requested.store(true, Ordering::SeqCst);
        if self.control.active.load(Ordering::SeqCst) {
            self.control.cancellation.cancel();
        }
        let state = Arc::clone(&self.state);
        spawn_execution(env, async move {
            tokio::task::spawn_blocking(move || state.lock().close())
                .await
                .map_err(|error| NodeGrafeoError::Database(error.to_string()))?
        })
    }
}
