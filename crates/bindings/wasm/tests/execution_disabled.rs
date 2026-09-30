//! Feature-disabled execution controls must reject without consuming owners.

#![cfg(all(
    target_arch = "wasm32",
    not(all(
        feature = "gql",
        any(
            feature = "edge",
            feature = "lpg",
            feature = "native",
            feature = "compact-store"
        )
    ))
))]

use grafeo_wasm::{Database, QueryControl};
use js_sys::Reflect;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn code(error: &JsValue) -> Option<String> {
    Reflect::get(error, &JsValue::from_str("code"))
        .ok()
        .and_then(|value| value.as_string())
}

#[cfg(any(
    feature = "rdf-model",
    feature = "lpg",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
#[wasm_bindgen_test]
fn unsupported_stream_rejects_without_consuming_control() -> Result<(), JsValue> {
    let db = Database::with_graph_model(if cfg!(feature = "rdf-model") {
        "rdf"
    } else {
        "lpg"
    })?;
    let control = QueryControl::new(None)?;
    let error = db
        .execute_stream_with_options("RETURN 1", &control, JsValue::UNDEFINED, JsValue::UNDEFINED)
        .err()
        .expect("feature-disabled stream must reject");
    assert_eq!(code(&error).as_deref(), Some("GRAFEO-Q004"));
    assert!(!control.consumed());
    control.cancel();
    db.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn explicit_deadline_is_structured_unsupported() {
    let error = QueryControl::new(Some(0.0))
        .err()
        .expect("deadline is unsupported");
    assert_eq!(code(&error).as_deref(), Some("GRAFEO-Q004"));
}

#[cfg(not(any(
    feature = "rdf-model",
    feature = "lpg",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
)))]
#[wasm_bindgen_test]
fn disabled_database_constructor_returns_error_instead_of_trapping() {
    let error = Database::new().err().expect("no LPG store compiled");
    assert!(code(&error).is_some(), "native error identity is preserved");
    let control = QueryControl::new(None).expect("cancellable owner needs no graph");
    control.cancel();
    assert!(!control.consumed());
}
