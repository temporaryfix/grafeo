//! Public WASM copy admission, cancellation, and exported owner-lifetime controls.
#![cfg(all(
    target_arch = "wasm32",
    feature = "gql",
    any(
        feature = "edge",
        feature = "lpg",
        feature = "native",
        feature = "compact-store"
    )
))]

use grafeo_wasm::{Database, QueryControl};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

fn options(max_rows: Option<u32>, max_bytes: Option<u32>, language: Option<&str>) -> JsValue {
    let value = Object::new();
    if let Some(rows) = max_rows {
        Reflect::set(
            &value,
            &JsValue::from_str("maxRows"),
            &JsValue::from_f64(rows as f64),
        )
        .unwrap();
    }
    if let Some(bytes) = max_bytes {
        Reflect::set(
            &value,
            &JsValue::from_str("maxBytes"),
            &JsValue::from_f64(bytes as f64),
        )
        .unwrap();
    }
    if let Some(language) = language {
        Reflect::set(
            &value,
            &JsValue::from_str("language"),
            &JsValue::from_str(language),
        )
        .unwrap();
    }
    value.into()
}

fn error_code(error: &JsValue) -> Option<String> {
    Reflect::get(error, &JsValue::from_str("code"))
        .ok()
        .and_then(|value| value.as_string())
}

fn rows(value: &JsValue) -> Array {
    value.clone().dyn_into().expect("result rows array")
}

#[wasm_bindgen_test]
fn cancelled_control_is_single_use_and_does_not_mutate() -> Result<(), JsValue> {
    let db = Database::new()?;
    let control = QueryControl::new(None)?;
    control.cancel();
    let error = db
        .execute_with_options(
            "INSERT (:Cancelled {i: 1}) RETURN 1",
            &control,
            JsValue::UNDEFINED,
            JsValue::UNDEFINED,
        )
        .expect_err("cancelled execution must fail");
    assert_eq!(error_code(&error).as_deref(), Some("GRAFEO-Q007"));
    assert!(control.consumed());
    let result = db.execute("MATCH (n:Cancelled) RETURN n")?;
    assert_eq!(rows(&result).length(), 0);
    assert!(
        db.execute_with_options("RETURN 1", &control, JsValue::UNDEFINED, JsValue::UNDEFINED,)
            .is_err()
    );
    Ok(())
}

#[wasm_bindgen_test]
fn unqualified_deadline_is_structured_unsupported() -> Result<(), JsValue> {
    let error = QueryControl::new(Some(0.0))
        .err()
        .expect("deadline construction must reject");
    assert_eq!(error_code(&error).as_deref(), Some("GRAFEO-Q004"));
    assert!(QueryControl::new(Some(-1.0)).is_err());
    Ok(())
}

#[wasm_bindgen_test]
fn eager_raw_params_and_language_honor_limits() -> Result<(), JsValue> {
    let db = Database::new()?;
    let control = QueryControl::new(None)?;
    let params = js_sys::JSON::parse(r#"{"value":17}"#)?;
    let result = db.execute_raw_with_options(
        "RETURN $value AS value",
        &control,
        options(Some(1), Some(16_384), Some("gql")),
        params,
    )?;
    let raw_rows = Reflect::get(&result, &JsValue::from_str("rows"))?;
    assert_eq!(rows(&raw_rows).length(), 1);
    Ok(())
}

#[cfg(any(
    feature = "lpg",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
#[wasm_bindgen_test]
fn limited_transaction_statement_preserves_prior_write() -> Result<(), JsValue> {
    let db = Database::new()?;
    db.begin_transaction()?;
    db.execute("INSERT (:Kept {i: 1})")?;
    let control = QueryControl::new(None)?;
    let error = db
        .execute_with_options(
            "INSERT (:Denied {payload: 'large'}) RETURN 1",
            &control,
            options(None, Some(1), None),
            JsValue::UNDEFINED,
        )
        .expect_err("byte cap must deny statement");
    assert_eq!(error_code(&error).as_deref(), Some("GRAFEO-S001"));
    db.commit_transaction()?;
    assert_eq!(db.node_count()?, 1);
    Ok(())
}

#[wasm_bindgen_test]
fn stream_chunks_are_exact_and_cancel_between_chunks() -> Result<(), JsValue> {
    let db = Database::new()?;
    let control = QueryControl::new(None)?;
    let mut stream = db.execute_stream_with_options(
        "UNWIND range(1, 2500) AS i RETURN i",
        &control,
        JsValue::UNDEFINED,
        JsValue::UNDEFINED,
    )?;
    let mut seen = std::collections::BTreeSet::new();
    let first = stream.next_chunk(JsValue::from_f64(257.0))?;
    let first_rows = rows(&first);
    assert!(first_rows.length() <= 257);
    for value in first_rows.iter() {
        let i = Reflect::get(&value, &JsValue::from_str("i"))?
            .as_f64()
            .unwrap()
            .to_string()
            .parse::<u32>()
            .expect("exact unsigned fixture integer");
        seen.insert(i);
    }
    control.cancel();
    let error = stream
        .next_chunk(JsValue::from_f64(257.0))
        .expect_err("cancel must be observed between chunks");
    assert_eq!(error_code(&error).as_deref(), Some("GRAFEO-Q007"));
    assert_eq!(
        error_code(&stream.close().expect_err("sticky close")).as_deref(),
        Some("GRAFEO-Q007")
    );
    assert!(stream.close().is_err());
    assert!(!seen.is_empty());
    Ok(())
}

#[wasm_bindgen_test]
fn independent_streams_and_early_close_are_isolated() -> Result<(), JsValue> {
    let db = Database::new()?;
    let first_control = QueryControl::new(None)?;
    let second_control = QueryControl::new(None)?;
    let mut first = db.execute_stream_with_options(
        "UNWIND range(1, 2500) AS i RETURN i",
        &first_control,
        JsValue::UNDEFINED,
        JsValue::UNDEFINED,
    )?;
    let mut second = db.execute_stream_with_options(
        "UNWIND range(1, 2500) AS i RETURN i",
        &second_control,
        JsValue::UNDEFINED,
        JsValue::UNDEFINED,
    )?;
    let _ = first.next()?;
    first.close()?;
    first.close()?;
    let collected = second.to_array()?;
    assert_eq!(rows(&collected).length(), 2500);
    second.close()?;
    Ok(())
}

fn mutation_rows(result: &JsValue, raw: bool) -> Result<Array, JsValue> {
    if raw {
        Ok(Reflect::get(result, &JsValue::from_str("rows"))?.dyn_into()?)
    } else {
        Ok(result.clone().dyn_into()?)
    }
}

#[wasm_bindgen_test]
fn eager_and_raw_mutation_limits_preserve_structured_values() -> Result<(), JsValue> {
    let cases = [
        ("string", r#"{"value":"plain"}"#, r#""plain""#),
        ("number", r#"{"value":17}"#, "17"),
        ("boolean", r#"{"value":true}"#, "true"),
        ("null", r#"{"value":null}"#, "null"),
        ("unicode", r#"{"value":"雪😀"}"#, r#""雪😀""#),
        (
            "nested",
            r#"{"value":[1,null,{"key":"v"}]}"#,
            r#"[1,null,{"key":"v"}]"#,
        ),
        (
            "proto",
            r#"{"value":{"__proto__":{"safe":true}}}"#,
            r#"{"__proto__":{"safe":true}}"#,
        ),
    ];
    for (index, (name, params_json, expected_json)) in cases.into_iter().enumerate() {
        for raw in [false, true] {
            let db = Database::new()?;
            let label = format!("Denied{index}_{name}_{raw}");
            let query = format!("INSERT (:{label} {{value: $value}}) RETURN $value AS value");
            let tiny = QueryControl::new(None)?;
            let error = if raw {
                db.execute_raw_with_options(
                    &query,
                    &tiny,
                    options(None, Some(1), None),
                    js_sys::JSON::parse(params_json)?,
                )
                .expect_err("tiny raw budget must deny mutation")
            } else {
                db.execute_with_options(
                    &query,
                    &tiny,
                    options(None, Some(1), None),
                    js_sys::JSON::parse(params_json)?,
                )
                .expect_err("tiny eager budget must deny mutation")
            };
            assert_eq!(error_code(&error).as_deref(), Some("GRAFEO-S001"));
            let denied = db.execute(&format!("MATCH (n:{label}) RETURN n"))?;
            assert_eq!(rows(&denied).length(), 0);

            let adequate = QueryControl::new(None)?;
            let result = if raw {
                db.execute_raw_with_options(
                    &query,
                    &adequate,
                    options(None, Some(131_072), None),
                    js_sys::JSON::parse(params_json)?,
                )?
            } else {
                db.execute_with_options(
                    &query,
                    &adequate,
                    options(None, Some(131_072), None),
                    js_sys::JSON::parse(params_json)?,
                )?
            };
            let returned = mutation_rows(&result, raw)?;
            assert_eq!(returned.length(), 1);
            let row = returned.get(0);
            let value = if raw {
                rows(&row).get(0)
            } else {
                Reflect::get(&row, &JsValue::from_str("value"))?
            };
            assert_eq!(
                js_sys::JSON::stringify(&value)?.as_string().as_deref(),
                Some(expected_json)
            );
            let committed = db.execute(&format!("MATCH (n:{label}) RETURN n"))?;
            assert_eq!(rows(&committed).length(), 1);
            db.close()?;
        }
    }
    Ok(())
}

#[wasm_bindgen_test]
fn exported_stream_survives_database_free() -> Result<(), JsValue> {
    let db = Database::new()?;
    let db_value: JsValue = db.into();
    let control: JsValue = QueryControl::new(None)?.into();
    let stream_method = Reflect::get(&db_value, &JsValue::from_str("executeStreamWithOptions"))?
        .dyn_into::<js_sys::Function>()?;
    let args = Array::new();
    args.push(&JsValue::from_str("UNWIND range(1, 2500) AS i RETURN i"));
    args.push(&control);
    args.push(&JsValue::UNDEFINED);
    args.push(&JsValue::UNDEFINED);
    let stream = stream_method.apply(&db_value, &args)?;
    Reflect::get(&control, &JsValue::from_str("free"))?
        .dyn_into::<js_sys::Function>()?
        .call0(&control)?;
    let free =
        Reflect::get(&db_value, &JsValue::from_str("free"))?.dyn_into::<js_sys::Function>()?;
    free.call0(&db_value)?;
    let to_array =
        Reflect::get(&stream, &JsValue::from_str("toArray"))?.dyn_into::<js_sys::Function>()?;
    let rows = to_array.call0(&stream)?.dyn_into::<Array>()?;
    assert_eq!(rows.length(), 2500);
    let close =
        Reflect::get(&stream, &JsValue::from_str("close"))?.dyn_into::<js_sys::Function>()?;
    close.call0(&stream)?;
    Ok(())
}

#[wasm_bindgen_test]
fn complete_chunks_preserve_exact_rows_and_validate_js_size() -> Result<(), JsValue> {
    let db = Database::new()?;
    let control = QueryControl::new(None)?;
    let mut stream = db.execute_stream_with_options(
        "UNWIND range(1, 2500) AS i RETURN i",
        &control,
        JsValue::UNDEFINED,
        JsValue::UNDEFINED,
    )?;
    for invalid in [
        JsValue::from_f64(-1.0),
        JsValue::from_f64(1.5),
        JsValue::from_str("2"),
        JsValue::from_f64(f64::NAN),
    ] {
        assert!(stream.next_chunk(invalid).is_err());
    }
    let mut seen = std::collections::BTreeSet::new();
    loop {
        let chunk = stream.next_chunk(JsValue::from_f64(257.0))?;
        if chunk.is_null() {
            break;
        }
        let array = rows(&chunk);
        assert!((1..=257).contains(&array.length()));
        for row in array.iter() {
            let value = Reflect::get(&row, &JsValue::from_str("i"))?
                .as_f64()
                .expect("integer");
            assert!(
                seen.insert(
                    value
                        .to_string()
                        .parse::<u32>()
                        .expect("exact unsigned fixture integer")
                )
            );
        }
    }
    assert_eq!(seen, (1..=2500).collect());
    stream.close()?;
    db.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn live_cursor_rejects_publication_writes_and_survives_database_drop() -> Result<(), JsValue> {
    let db = Database::new()?;
    let control = QueryControl::new(None)?;
    let mut stream = db.execute_stream_with_options(
        "UNWIND range(1, 2500) AS i RETURN i",
        &control,
        JsValue::UNDEFINED,
        JsValue::UNDEFINED,
    )?;
    assert!(db.execute("INSERT (:Blocked)").is_err());
    assert!(db.close().is_err());
    drop(db);
    assert_eq!(rows(&stream.to_array()?).length(), 2500);
    stream.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn collection_limit_is_sticky_and_releases_parent() -> Result<(), JsValue> {
    let db = Database::new()?;
    let control = QueryControl::new(None)?;
    let mut stream = db.execute_stream_with_options(
        "UNWIND range(1, 2500) AS i RETURN i",
        &control,
        options(None, Some(65536), None),
        JsValue::UNDEFINED,
    )?;
    let error = stream.to_array().expect_err("collection limit");
    assert_eq!(error_code(&error).as_deref(), Some("GRAFEO-S001"));
    assert!(stream.next().is_err());
    assert!(stream.close().is_err());
    db.close()?;
    Ok(())
}

#[wasm_bindgen_test]
fn options_getter_cannot_close_owner_and_proto_keys_remain_data() -> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    let control: JsValue = QueryControl::new(None)?.into();
    let factory = js_sys::Function::new_with_args(
        "db",
        "return {get maxRows(){try {db.close(); throw new Error('close succeeded')} catch(e){if(e.message==='close succeeded')throw e;} return 1;}}",
    );
    let options = factory.call1(&JsValue::UNDEFINED, &db)?;
    let execute = Reflect::get(&db, &JsValue::from_str("executeWithOptions"))?
        .dyn_into::<js_sys::Function>()?;
    let args = Array::new();
    args.push(&JsValue::from_str("INSERT (:Kept) RETURN $value AS value"));
    args.push(&control);
    args.push(&options);
    args.push(&js_sys::JSON::parse(r#"{"value":{"__proto__":17}}"#)?);
    let output = execute.apply(&db, &args)?;
    let value = Reflect::get(&rows(&output).get(0), &JsValue::from_str("value"))?;
    assert_eq!(
        Reflect::get(&value, &JsValue::from_str("__proto__"))?.as_f64(),
        Some(17.0)
    );
    Reflect::get(&db, &JsValue::from_str("close"))?
        .dyn_into::<js_sys::Function>()?
        .call0(&db)?;
    Ok(())
}
