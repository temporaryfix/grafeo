//! Real WASM caller ownership, bounds, native errors and snapshot continuation.
#![cfg(all(target_arch = "wasm32", feature = "cdc", feature = "lpg"))]

use grafeo_wasm::Database;
use js_sys::{Array, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

fn get(value: &JsValue, key: &str) -> JsValue {
    Reflect::get(value, &key.into()).unwrap()
}
fn events(page: &JsValue) -> Array {
    get(page, "events").dyn_into().unwrap()
}
fn next(page: &JsValue) -> JsValue {
    get(page, "next")
}
fn bytes(value: &JsValue) -> Vec<u8> {
    value.dyn_ref::<Uint8Array>().unwrap().to_vec()
}
fn code(error: JsValue, expected: &str) {
    assert_eq!(get(&error, "code").as_string().as_deref(), Some(expected));
}
fn populated() -> Result<Database, JsValue> {
    let db = Database::new()?;
    db.set_cdc_enabled(true)?;
    assert!(db.is_cdc_enabled()?);
    db.execute("INSERT (:A), (:B)")?;
    db.execute("MATCH (a:A), (b:B) INSERT (a)-[:LINK]->(b)")?;
    Ok(db)
}

#[wasm_bindgen_test]
fn pages_own_creation_payloads_and_entity_history_after_close() -> Result<(), JsValue> {
    let db = populated()?;
    let mut cursor = JsValue::NULL;
    let mut pages = Vec::new();
    for kind in ["node", "node", "edge"] {
        let page = db.changes_after(cursor, 1.0, 4096.0)?;
        assert_eq!(events(&page).length(), 1);
        assert_eq!(get(&events(&page).get(0), "entity_type"), kind);
        cursor = next(&page);
        assert_eq!(bytes(&cursor).len(), 97);
        pages.push(page);
    }
    let eof = db.changes_after(cursor.clone(), 1.0, 4096.0)?;
    assert_eq!(events(&eof).length(), 0);
    assert_eq!(bytes(&next(&eof)), bytes(&cursor));
    let node = events(&pages[0]).get(0);
    let id = get(&node, "entity_id").as_string().unwrap();
    let epoch = get(&node, "epoch").as_string().unwrap();
    let history = db.node_history_after(&id, &epoch, JsValue::NULL, 1.0, 4096.0)?;
    assert_eq!(events(&history).length(), 1);
    assert_eq!(get(&events(&history).get(0), "epoch"), epoch);
    assert_eq!(
        events(&db.node_history_after(&id, "18446744073709551615", JsValue::NULL, 1.0, 4096.0)?)
            .length(),
        0
    );
    assert_eq!(
        events(&db.node_history_after("18446744073709551614", "0", JsValue::NULL, 1.0, 4096.0)?)
            .length(),
        0
    );
    let edge = events(&pages[2]).get(0);
    let edge_id = get(&edge, "entity_id").as_string().unwrap();
    assert_eq!(
        events(&db.edge_history_after(&edge_id, "0", JsValue::NULL, 1.0, 4096.0)?).length(),
        1
    );
    db.close()?;
    assert_eq!(Array::from(&get(&node, "labels")).get(0), "A");
    assert_eq!(get(&edge, "edge_type"), "LINK");
    assert_eq!(get(&edge, "src_id"), id);
    assert_eq!(
        get(&edge, "dst_id"),
        get(&events(&pages[1]).get(0), "entity_id")
    );
    assert!(get(&node, "graph_incarnation").is_string());
    code(
        db.changes_after(JsValue::NULL, 1.0, 4096.0).unwrap_err(),
        "GRAFEO-V001",
    );
    Ok(())
}

#[wasm_bindgen_test]
fn malformed_foreign_and_resource_inputs_keep_native_codes() -> Result<(), JsValue> {
    let db = populated()?;
    for length in [0, 96, 97, 98] {
        code(
            db.changes_after(Uint8Array::new_with_length(length).into(), 1.0, 4096.0)
                .unwrap_err(),
            "GRAFEO-S004",
        );
    }
    code(
        db.changes_after(Array::new().into(), 1.0, 4096.0)
            .unwrap_err(),
        "GRAFEO-S004",
    );
    for value in [0.0, -1.0, 1.5, f64::NAN, f64::INFINITY, 4_294_967_296.0] {
        code(
            db.changes_after(JsValue::NULL, value, 4096.0).unwrap_err(),
            "GRAFEO-V001",
        );
        code(
            db.changes_after(JsValue::NULL, 1.0, value).unwrap_err(),
            "GRAFEO-V001",
        );
    }
    for value in ["", "-1", "1.0", "18446744073709551616"] {
        code(
            db.node_history_after(value, "0", JsValue::NULL, 1.0, 4096.0)
                .unwrap_err(),
            "GRAFEO-V001",
        );
        code(
            db.edge_history_after("0", value, JsValue::NULL, 1.0, 4096.0)
                .unwrap_err(),
            "GRAFEO-V001",
        );
    }
    code(
        db.changes_after(JsValue::NULL, 1.0, 1.0).unwrap_err(),
        "GRAFEO-S001",
    );
    let other = populated()?;
    let foreign = next(&other.changes_after(JsValue::UNDEFINED, 1.0, 4096.0)?);
    code(
        db.changes_after(foreign, 1.0, 4096.0).unwrap_err(),
        "GRAFEO-S005",
    );
    db.close()?;
    other.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn cursor_continues_across_two_snapshot_imports() -> Result<(), JsValue> {
    let mut db = populated()?;
    let first = db.changes_after(JsValue::NULL, 1.0, 4096.0)?;
    let mut cursor = next(&first);
    for expected in ["node", "edge"] {
        let snapshot = db.export_snapshot()?;
        db.close()?;
        db = Database::import_snapshot(&snapshot)?;
        let page = db.changes_after(cursor, 1.0, 4096.0)?;
        assert_eq!(events(&page).length(), 1);
        assert_eq!(get(&events(&page).get(0), "entity_type"), expected);
        cursor = next(&page);
    }
    let eof = db.changes_after(cursor.clone(), 1.0, 4096.0)?;
    assert_eq!(events(&eof).length(), 0);
    assert_eq!(bytes(&next(&eof)), bytes(&cursor));
    db.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn active_transaction_does_not_publish_uncommitted_changes() -> Result<(), JsValue> {
    let db = Database::new()?;
    db.set_cdc_enabled(true)?;
    db.begin_transaction()?;
    db.execute("INSERT (:Kept)")?;
    assert_eq!(
        events(&db.changes_after(JsValue::NULL, 1.0, 4096.0)?).length(),
        0
    );
    db.commit_transaction()?;
    let page = db.changes_after(JsValue::NULL, 1.0, 4096.0)?;
    assert_eq!(events(&page).length(), 1);
    let cursor = next(&page);
    db.begin_transaction()?;
    db.execute("INSERT (:Discarded)")?;
    assert_eq!(
        events(&db.changes_after(cursor.clone(), 1.0, 4096.0)?).length(),
        0
    );
    db.rollback_transaction()?;
    let eof = db.changes_after(cursor.clone(), 1.0, 4096.0)?;
    assert_eq!(events(&eof).length(), 0);
    assert_eq!(bytes(&next(&eof)), bytes(&cursor));
    db.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn snapshot_retains_eviction_floor_and_stale_cursor_error() -> Result<(), JsValue> {
    let mut config = grafeo_engine::Config::in_memory().with_cdc();
    config.cdc_retention.max_epochs = None;
    config.cdc_retention.max_events = Some(1);
    let native = grafeo_engine::GrafeoDB::with_config(config).unwrap();
    native.execute("INSERT (:First)").unwrap();
    let stale = native.changes_after(None, 1, 4096).unwrap().next.to_bytes();
    native.execute("INSERT (:Second)").unwrap();
    native.execute("INSERT (:Third)").unwrap();
    native.gc().unwrap();
    let snapshot = native.export_snapshot().unwrap();
    let db = Database::import_snapshot(&snapshot)?;
    code(
        db.changes_after(Uint8Array::from(stale.as_slice()).into(), 1.0, 4096.0)
            .unwrap_err(),
        "GRAFEO-S006",
    );
    assert_eq!(
        events(&db.changes_after(JsValue::NULL, 1.0, 4096.0)?).length(),
        1
    );
    db.close()?;
    Ok(())
}

#[cfg(feature = "sparql")]
#[wasm_bindgen_test]
fn rdf_creation_payload_retains_named_graph_and_native_terms() -> Result<(), JsValue> {
    let db = Database::with_graph_model("both")?;
    db.set_cdc_enabled(true)?;
    db.execute_sparql("INSERT DATA { GRAPH <urn:g> { <urn:s> <urn:p> \"value\" . } }")?;
    let page = db.changes_after(JsValue::NULL, 1.0, 4096.0)?;
    assert_eq!(events(&page).length(), 1);
    let event = events(&page).get(0);
    assert_eq!(get(&event, "entity_type"), "triple");
    assert_eq!(get(&event, "triple_graph"), "urn:g");
    assert_eq!(get(&event, "triple_subject"), "<urn:s>");
    assert_eq!(get(&event, "triple_predicate"), "<urn:p>");
    assert_eq!(get(&event, "triple_object"), "\"value\"");
    assert!(get(&event, "labels").is_null());
    db.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn cdc_clock_uses_platform_wall_time_and_remains_monotone() {
    let before = js_sys::Date::now()
        .to_string()
        .parse::<u64>()
        .expect("exact timestamp milliseconds");
    let clock = grafeo_common::types::HlcClock::new();
    let first = clock.now();
    let second = clock.now();
    let after = js_sys::Date::now()
        .to_string()
        .parse::<u64>()
        .expect("exact timestamp milliseconds");
    assert!(first.physical_ms() >= before && first.physical_ms() <= after);
    assert!(second > first);
    let observed = grafeo_common::types::HlcTimestamp::new(after + 1000, 12);
    assert!(clock.update(observed) > observed);
}
