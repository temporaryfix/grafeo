//! Owned bounded CDC transport; native admission precedes JS conversion.

use grafeo_common::types::{DurableCursor, EdgeId, EpochId, NodeId};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::cdc::EntityHistoryQuery;
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, prelude::*};

use crate::{Database, execution};

#[wasm_bindgen]
impl Database {
    /// Controls capture for future sessions; existing transactions retain their setting.
    ///
    /// # Errors
    /// Returns a structured error when the database is closed or busy.
    #[wasm_bindgen(js_name = "setCdcEnabled")]
    pub fn set_cdc_enabled(&self, enabled: bool) -> Result<(), JsValue> {
        let _operation = self.reserve_query()?;
        self.inner.set_cdc_enabled(enabled);
        Ok(())
    }

    /// Returns whether future sessions capture changes.
    ///
    /// # Errors
    /// Returns a structured error when the database is closed or busy.
    #[wasm_bindgen(js_name = "isCdcEnabled")]
    pub fn is_cdc_enabled(&self) -> Result<bool, JsValue> {
        let _operation = self.reserve_query()?;
        Ok(self.inner.is_cdc_enabled())
    }

    /// Reads an owned bounded page. Null/undefined starts at the retained floor.
    /// Limits count events/native encoded event bytes, excluding JS envelopes.
    ///
    /// # Errors
    /// Preserves native invalid/foreign/evicted/resource and transaction errors.
    #[wasm_bindgen(js_name = "changesAfter", unchecked_return_type = "ChangePage")]
    pub fn changes_after(
        &self,
        cursor: JsValue,
        max_events: f64,
        max_bytes: f64,
    ) -> Result<JsValue, JsValue> {
        self.read_change_page(cursor, max_events, max_bytes, None)
    }

    /// Reads node history with exact decimal u64 identity and inclusive minimum epoch.
    ///
    /// # Errors
    /// Invalid coordinates and page/native errors are structured exceptions.
    #[wasm_bindgen(js_name = "nodeHistoryAfter", unchecked_return_type = "ChangePage")]
    pub fn node_history_after(
        &self,
        id: &str,
        since_epoch: &str,
        cursor: JsValue,
        max_events: f64,
        max_bytes: f64,
    ) -> Result<JsValue, JsValue> {
        let mut query = EntityHistoryQuery::new(NodeId::new(coordinate(id)?));
        query.since_epoch = EpochId::new(coordinate(since_epoch)?);
        self.read_change_page(cursor, max_events, max_bytes, Some(query))
    }

    /// Reads edge history with the same bounds/coordinate rules as node history.
    ///
    /// # Errors
    /// Invalid coordinates and page/native errors are structured exceptions.
    #[wasm_bindgen(js_name = "edgeHistoryAfter", unchecked_return_type = "ChangePage")]
    pub fn edge_history_after(
        &self,
        id: &str,
        since_epoch: &str,
        cursor: JsValue,
        max_events: f64,
        max_bytes: f64,
    ) -> Result<JsValue, JsValue> {
        let mut query = EntityHistoryQuery::new(EdgeId::new(coordinate(id)?));
        query.since_epoch = EpochId::new(coordinate(since_epoch)?);
        self.read_change_page(cursor, max_events, max_bytes, Some(query))
    }
}

impl Database {
    fn read_change_page(
        &self,
        cursor: JsValue,
        max_events: f64,
        max_bytes: f64,
        query: Option<EntityHistoryQuery>,
    ) -> Result<JsValue, JsValue> {
        let _operation = self.reserve_query()?;
        let cursor = decode_cursor(&cursor)?;
        let max_events = limit(max_events)?;
        let max_bytes = limit(max_bytes)?;
        let read = |session: &grafeo_engine::session::Session| match &query {
            Some(query) => session.history_after(query, cursor.as_ref(), max_events, max_bytes),
            None => session.changes_after(cursor.as_ref(), max_events, max_bytes),
        };
        let tx = self
            .tx
            .try_borrow()
            .map_err(|_| execution::invalid("Database owner is busy"))?;
        let page = match tx.as_ref() {
            Some(session) => read(session),
            None => read(&self.inner.session()),
        }
        .map_err(|error| execution::native_error(&error))?;
        // Only admitted events cross the transport. Decimal coordinate strings
        // survive JS number limits; native cursor bytes are copied into JS memory.
        let events: Vec<_> = page
            .events
            .iter()
            .map(grafeo_bindings_common::cdc::change_event_to_json)
            .collect();
        let json = serde_json::to_string(&events)
            .map_err(|error| execution::native_error(&Error::Serialization(error.to_string())))?;
        let result = Object::new();
        Reflect::set(&result, &"events".into(), &js_sys::JSON::parse(&json)?)?;
        Reflect::set(
            &result,
            &"next".into(),
            &Uint8Array::from(page.next.to_bytes().as_slice()),
        )?;
        Ok(result.into())
    }
}

fn decode_cursor(value: &JsValue) -> Result<Option<DurableCursor>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let invalid = || execution::native_error(&StorageError::CursorInvalid.into());
    let bytes = value.dyn_ref::<Uint8Array>().ok_or_else(invalid)?;
    if bytes.length() as usize != DurableCursor::LEN {
        return Err(invalid());
    }
    let mut owned = [0; DurableCursor::LEN];
    bytes.copy_to(&mut owned);
    DurableCursor::from_bytes(&owned)
        .map(Some)
        .map_err(|error| execution::native_error(&error))
}

fn coordinate(value: &str) -> Result<u64, JsValue> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(execution::invalid(
            "CDC coordinates must be decimal u64 strings",
        ));
    }
    value
        .parse()
        .map_err(|_| execution::invalid("CDC coordinate exceeds u64"))
}

fn limit(value: f64) -> Result<usize, JsValue> {
    if !value.is_finite()
        || value < 1.0
        || value.fract() != 0.0
        || value > 9_007_199_254_740_991.0
        || value > usize::MAX as f64
    {
        return Err(execution::invalid(
            "CDC page limits must be positive safe integers fitting this platform",
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value as usize)
}

#[wasm_bindgen(typescript_custom_section)]
const CDC_TYPES: &str = r#"
/** Owned JS page; next is a 97-byte exclusive cursor. No native page needs freeing. */
export interface ChangePage {
  events: ChangeEvent[];
  next: Uint8Array;
}
/** Native coordinates are exact decimal strings. */
export interface ChangeEvent {
  entity_id: string; entity_type: string; kind: string;
  epoch: string; timestamp: string; graph_incarnation: string | null;
  before: Record<string, unknown> | null; after: Record<string, unknown> | null;
  labels: string[] | null; edge_type: string | null; src_id: string | null; dst_id: string | null;
  lpg_graph: string[] | null;
  triple_graph: string | null; triple_subject: string | null;
  triple_predicate: string | null; triple_object: string | null;
}
"#;
