//! Owned, bounded WASM cursors with sticky terminal errors.

use std::cell::Cell;
use std::rc::Rc;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result as NativeResult;
use js_sys::{Array, Reflect};
use wasm_bindgen::prelude::*;

use crate::execution::{invalid, native_error};
use crate::types::{self, CopyBudget};

trait Cursor: Iterator<Item = NativeResult<Vec<Value>>> {
    fn check_execution(&mut self) -> NativeResult<()>;
    fn close(&mut self) -> NativeResult<()>;
}

#[cfg(all(
    feature = "gql",
    any(
        feature = "edge",
        feature = "lpg",
        feature = "native",
        feature = "compact-store"
    )
))]
impl Cursor for grafeo_engine::query::executor::stream::OwnedRowIterator {
    fn check_execution(&mut self) -> NativeResult<()> {
        Self::check_execution(self)
    }
    fn close(&mut self) -> NativeResult<()> {
        Self::close(self)
    }
}

/// A lazy read cursor. Cancellation is observed between JavaScript calls;
/// synchronous native work cannot be interrupted by the same blocked event loop.
#[wasm_bindgen]
pub struct ResultStream {
    cursor: Option<Box<dyn Cursor>>,
    keepalive: Option<Rc<grafeo_engine::GrafeoDB>>,
    columns: Vec<String>,
    pending: Option<Vec<Value>>,
    streams: Rc<Cell<usize>>,
    max_rows: Option<usize>,
    max_bytes: usize,
    delivered: usize,
    closed: bool,
    failure: Option<JsValue>,
}

impl ResultStream {
    #[cfg(all(
        feature = "gql",
        any(
            feature = "edge",
            feature = "lpg",
            feature = "native",
            feature = "compact-store"
        )
    ))]
    pub(crate) fn new(
        mut cursor: grafeo_engine::query::executor::stream::OwnedRowIterator,
        max_rows: Option<usize>,
        max_bytes: usize,
        streams: Rc<Cell<usize>>,
        keepalive: Rc<grafeo_engine::GrafeoDB>,
    ) -> Result<Self, JsValue> {
        let mut budget = CopyBudget::new(max_bytes);
        if let Err(error) = budget
            .columns(cursor.columns())
            .and_then(|()| budget.list(0))
        {
            let primary = native_error(&error);
            if let Err(cleanup) = cursor.close() {
                attach_cleanup(&primary, &native_error(&cleanup));
            }
            return Err(primary);
        }
        let count = streams
            .get()
            .checked_add(1)
            .ok_or_else(|| invalid("Too many live streams"))?;
        let columns = cursor.columns().to_vec();
        let cursor = Box::new(cursor);
        streams.set(count);
        Ok(Self {
            cursor: Some(cursor),
            keepalive: Some(keepalive),
            columns,
            pending: None,
            streams,
            max_rows,
            max_bytes,
            delivered: 0,
            closed: false,
            failure: None,
        })
    }

    fn check(&self) -> Result<(), JsValue> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.closed {
            return Err(invalid("ResultStream is closed"));
        }
        Ok(())
    }

    fn budget(&self) -> Result<CopyBudget, JsValue> {
        let mut budget = CopyBudget::new(self.max_bytes);
        budget
            .columns(&self.columns)
            .map_err(|error| native_error(&error))?;
        budget.list(0).map_err(|error| native_error(&error))?;
        Ok(budget)
    }

    fn finish(&mut self, primary: Option<JsValue>) -> Result<(), JsValue> {
        self.pending = None;
        if let Some(mut cursor) = self.cursor.take() {
            let cleanup = cursor.close().err().map(|error| native_error(&error));
            self.streams.set(self.streams.get().saturating_sub(1));
            drop(cursor);
            self.keepalive.take();
            if let (Some(error), Some(cleanup)) = (&primary, &cleanup) {
                attach_cleanup(error, cleanup);
            }
            self.failure = primary.or(cleanup);
        } else if self.failure.is_none() {
            self.failure = primary;
        }
        match &self.failure {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn pull(&mut self) -> Result<Option<Vec<Value>>, JsValue> {
        self.check()?;
        let Some(cursor) = &mut self.cursor else {
            return Ok(None);
        };
        if let Err(error) = cursor.check_execution() {
            self.finish(Some(native_error(&error)))?;
            return Ok(None);
        }
        let next = self.pending.take().map(Ok).or_else(|| cursor.next());
        match next {
            None => {
                self.finish(None)?;
                Ok(None)
            }
            Some(Err(error)) => {
                self.finish(Some(native_error(&error)))?;
                Ok(None)
            }
            Some(Ok(row)) => {
                if self.max_rows.is_some_and(|limit| self.delivered >= limit) {
                    self.finish(Some(native_error(&types::copy_limit_error())))?;
                }
                Ok(Some(row))
            }
        }
    }

    fn record_delivery(&mut self) -> Result<(), JsValue> {
        match self.delivered.checked_add(1) {
            Some(count) => {
                self.delivered = count;
                Ok(())
            }
            None => self.finish(Some(native_error(&types::copy_limit_error()))),
        }
    }

    fn next_with_budget(&mut self, budget: &mut CopyBudget) -> Result<JsValue, JsValue> {
        let Some(row) = self.pull()? else {
            return Ok(JsValue::NULL);
        };
        if let Err(error) = budget.row(&self.columns, &row) {
            self.finish(Some(native_error(&error)))?;
        }
        self.record_delivery()?;
        Ok(types::row_to_js_object(&self.columns, &row))
    }
}

#[wasm_bindgen]
impl ResultStream {
    /// Copies the column names within this cursor's byte limit.
    ///
    /// # Errors
    /// Returns a structured error if column copies exceed the byte limit.
    #[wasm_bindgen(getter)]
    pub fn columns(&self) -> Result<JsValue, JsValue> {
        CopyBudget::new(self.max_bytes)
            .columns(&self.columns)
            .map_err(|error| native_error(&error))?;
        Ok(self
            .columns
            .iter()
            .map(|column| JsValue::from_str(column))
            .collect::<Array>()
            .into())
    }

    /// Pulls one bounded row, or null at clean exhaustion.
    ///
    /// # Errors
    /// Returns retained native, cancellation, cleanup or copy-limit errors.
    pub fn next(&mut self) -> Result<JsValue, JsValue> {
        self.check()?;
        let mut budget = self.budget()?;
        self.next_with_budget(&mut budget)
    }

    /// Pulls at most maxRows rows (maximum1024), or null at clean exhaustion.
    ///
    /// # Errors
    /// Returns retained execution/copy errors or invalid zero chunk size.
    #[wasm_bindgen(js_name = "nextChunk")]
    pub fn next_chunk(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "number")] max_rows: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.check()?;
        let max_rows = max_rows
            .as_f64()
            .filter(|value| {
                value.is_finite()
                    && *value >= 1.0
                    && value.fract() == 0.0
                    && *value <= f64::from(u32::MAX)
            })
            .ok_or_else(|| invalid("maxRows must be a positive integer within u32 capacity"))?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_rows = max_rows as u32;
        let mut budget = self.budget()?;
        let rows = Array::new();
        for _ in 0..max_rows.min(1024) {
            let Some(row) = self.pull()? else {
                break;
            };
            let mut candidate = budget;
            if let Err(error) = candidate
                .row(&self.columns, &row)
                .and_then(|()| candidate.charge(32))
            {
                if rows.length() == 0 {
                    self.finish(Some(native_error(&error)))?;
                }
                let mut fresh = self.budget()?;
                if let Err(error) = fresh
                    .row(&self.columns, &row)
                    .and_then(|()| fresh.charge(32))
                {
                    self.finish(Some(native_error(&error)))?;
                }
                self.pending = Some(row);
                break;
            }
            budget = candidate;
            rows.push(&types::row_to_js_object(&self.columns, &row));
            self.record_delivery()?;
        }
        Ok(if rows.length() == 0 {
            JsValue::NULL
        } else {
            rows.into()
        })
    }

    /// Collects the remaining rows within a total byte and row limit.
    ///
    /// # Errors
    /// Limit or execution failure returns no partial array and remains sticky.
    #[wasm_bindgen(js_name = "toArray")]
    pub fn to_array(&mut self) -> Result<JsValue, JsValue> {
        self.check()?;
        let mut budget = self.budget()?;
        let rows = Array::new();
        loop {
            let row = self.next_with_budget(&mut budget)?;
            if row.is_null() {
                break;
            }
            if rows.length() as usize >= self.max_rows.unwrap_or(1_000_000) {
                self.finish(Some(native_error(&types::copy_limit_error())))?;
            }
            // Includes grown JS backing capacity while previous capacity lives.
            if let Err(error) = budget.charge(32) {
                self.finish(Some(native_error(&error)))?;
            }
            rows.push(&row);
        }
        Ok(rows.into())
    }

    /// Releases native resources. Terminal errors remain observable on retry.
    ///
    /// # Errors
    /// Returns the primary execution error and retains any secondary cleanup.
    pub fn close(&mut self) -> Result<(), JsValue> {
        self.closed = true;
        self.finish(None)
    }
}

impl Drop for ResultStream {
    fn drop(&mut self) {
        if let Some(mut cursor) = self.cursor.take() {
            let _ = cursor.close();
            self.streams.set(self.streams.get().saturating_sub(1));
            drop(cursor);
            self.keepalive.take();
        }
    }
}

fn attach_cleanup(primary: &JsValue, cleanup: &JsValue) {
    let _ = Reflect::set(primary, &JsValue::from_str("cleanupError"), cleanup);
}
