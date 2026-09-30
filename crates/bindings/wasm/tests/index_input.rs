//! Actual JavaScript-boundary checks, including release-mode numeric coercion.
#![cfg(all(
    target_arch = "wasm32",
    any(
        feature = "lpg",
        feature = "edge",
        feature = "native",
        feature = "compact-store"
    )
))]

use grafeo_wasm::Database;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

fn call(db: &JsValue, method: &str, argument: &JsValue) -> Result<JsValue, JsValue> {
    let method =
        js_sys::Reflect::get(db, &JsValue::from_str(method))?.dyn_into::<js_sys::Function>()?;
    method.call1(db, argument)
}

#[wasm_bindgen_test]
fn index_input_rejects_malformed_utf16_without_aliasing_replacement_characters()
-> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    let seed = js_sys::eval(r#"({property: '\uFFFD'})"#)?;
    let owner = call(&db, "createIndex", &seed)?;
    for malformed in [r#"'\uD800'"#, r#"'\uDC00'"#, r#"'\uD800x'"#] {
        for field in [
            "property",
            "name",
            "kind",
            "label",
            "metric",
            "quantization",
        ] {
            let request =
                js_sys::eval(&format!("({{property:'unchanged', {field}:{malformed}}})"))?;
            let error =
                call(&db, "createIndex", &request).expect_err("malformed string must reject");
            let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))?;
            assert!(
                message
                    .as_string()
                    .is_some_and(|text| text.contains("UTF-16"))
            );
        }
        let request = js_sys::eval(&format!("({{property:'unchanged', graph:[{malformed}]}})"))?;
        let error = call(&db, "createIndex", &request).expect_err("malformed path must reject");
        let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))?;
        assert!(
            message
                .as_string()
                .is_some_and(|text| text.contains("UTF-16"))
        );
    }
    // Rejected requests did not create this property or consume an owner ID.
    let valid = js_sys::eval(r#"({property: 'unchanged'})"#)?;
    let next = call(&db, "createIndex", &valid)?;
    assert_eq!(next.as_f64(), owner.as_f64().map(|owner| owner + 1.0));
    call(&db, "rebuildIndex", &owner)?;
    assert_eq!(call(&db, "dropIndex", &owner)?.as_bool(), Some(true));

    // Legal surrogate pairs, NUL, slash and U+FFFD remain exact strings.
    let valid = js_sys::eval(r#"({property:'\uD83D\uDE00/\0\uFFFD'})"#)?;
    let valid_owner = call(&db, "createIndex", &valid)?;
    assert!(call(&db, "createIndex", &valid).is_err());
    assert_eq!(call(&db, "dropIndex", &valid_owner)?.as_bool(), Some(true));
    Ok(())
}

#[wasm_bindgen_test]
fn index_input_captures_getters_once_and_rejects_unknown_js_fields() -> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    for request in [
        r#"({property:'unchanged', graphPath:['child']})"#,
        r#"({property:'unchanged', ef_construction:2})"#,
    ] {
        assert!(call(&db, "createIndex", &js_sys::eval(request)?).is_err());
    }
    let request = js_sys::eval(
        r#"(() => {
        let reads = 0;
        return {get property() { return ++reads === 1 ? 'once' : '\uD800'; }};
    })()"#,
    )?;
    let owner = call(&db, "createIndex", &request)?;
    assert!(call(&db, "createIndex", &js_sys::eval("({property:'once'})")?).is_err());
    assert_eq!(call(&db, "dropIndex", &owner)?.as_bool(), Some(true));
    Ok(())
}

#[wasm_bindgen_test]
fn index_input_owner_rejects_coercion_and_preserves_live_owner() -> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    let owner = call(&db, "createIndex", &js_sys::eval("({property:'p'})")?)?;
    assert_eq!(owner.as_f64(), Some(0.0));
    let invalid: js_sys::Array = js_sys::eval(
        "[null, undefined, false, true, '0', [], [0], {valueOf(){return 0}}, new Number(0), 0n, -1, 0.5, 4294967296, NaN, Infinity]"
    )?.dyn_into()?;
    for invalid in invalid {
        assert!(call(&db, "dropIndex", &invalid).is_err());
        assert!(call(&db, "rebuildIndex", &invalid).is_err());
        call(&db, "rebuildIndex", &owner)?;
    }
    assert_eq!(call(&db, "dropIndex", &owner)?.as_bool(), Some(true));
    assert_eq!(call(&db, "dropIndex", &owner)?.as_bool(), Some(false));
    assert!(call(&db, "rebuildIndex", &owner).is_err());
    Ok(())
}

#[wasm_bindgen_test]
fn index_input_tokenizer_rejects_malformed_and_incompatible_values() -> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    let invalid: js_sys::Array = js_sys::eval(
        "[null, false, '3', 3n, [], [3], new Number(3), {valueOf(){return 3}}, -1, 1.5, NaN, Infinity, -Infinity, 4294967296, Number.MAX_SAFE_INTEGER + 1]",
    )?.dyn_into()?;
    for value in invalid {
        let request = js_sys::eval("({kind:'text',label:'Doc',property:'text'})")?;
        js_sys::Reflect::set(&request, &JsValue::from_str("minTokenLength"), &value)?;
        assert!(call(&db, "createIndex", &request).is_err());
    }
    for kind in ["undefined", "'property'", "'btree'", "'vector'"] {
        let request = js_sys::eval(&format!(
            "({{kind:{kind},property:'text',minTokenLength:3}})"
        ))?;
        let error = call(&db, "createIndex", &request).expect_err("non-Text option must reject");
        let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))?;
        assert!(
            message
                .as_string()
                .is_some_and(|text| text.contains("requires kind='text'"))
        );
    }
    for request in [
        "({kind:'text',label:'Doc',property:'text',min_token_length:3})",
        r#"({kind:'text',label:'Doc',property:'text',minTokenLength:3,graph:['\uD800']})"#,
    ] {
        assert!(call(&db, "createIndex", &js_sys::eval(request)?).is_err());
    }
    assert_eq!(
        call(
            &db,
            "createIndex",
            &js_sys::eval("({property:'unchanged'})")?
        )?
        .as_f64(),
        Some(0.0)
    );
    Ok(())
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
fn tokenizer_search_count(db: &JsValue, property: &str, query: &str) -> Result<u32, JsValue> {
    let method = js_sys::Reflect::get(db, &JsValue::from_str("textSearch"))?
        .dyn_into::<js_sys::Function>()?;
    let args = js_sys::Array::new();
    for value in [
        JsValue::from_str("Doc"),
        JsValue::from_str(property),
        JsValue::from_str(query),
        JsValue::from_f64(10.0),
    ] {
        args.push(&value);
    }
    Ok(method
        .apply(db, &args)?
        .dyn_into::<js_sys::Array>()?
        .length())
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
#[wasm_bindgen_test]
fn index_input_tokenizer_preserves_tokens_owners_rebuild_and_snapshot() -> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    call(
        &db,
        "execute",
        &JsValue::from_str(
            "INSERT (:Doc {strict: 'q ox cat', standard: 'q ox cat', zero: 'q ox cat', huge: 'q ox cat'})",
        ),
    )?;
    let strict_request = js_sys::eval(
        r#"(() => {
        let reads = 0;
        return {kind:'text',label:'Doc',property:'strict',graph:[],name:'strict-\uD83D\uDE80',
            get minTokenLength(){ return ++reads === 1 ? 3 : -1; }};
    })()"#,
    )?;
    let strict = call(&db, "createIndex", &strict_request)?;
    assert_eq!(strict.as_f64(), Some(0.0));
    assert!(
        call(
            &db,
            "createIndex",
            &js_sys::eval("({kind:'text',label:'Doc',property:'strict',minTokenLength:3})")?
        )
        .is_err()
    );
    let mut owners = vec![strict.clone()];
    for (property, option, expected) in [
        ("standard", "", 1.0),
        ("zero", ",minTokenLength:0", 2.0),
        ("huge", ",minTokenLength:4294967295", 3.0),
    ] {
        let request = js_sys::eval(&format!(
            "({{kind:'text',label:'Doc',property:'{property}'{option}}})"
        ))?;
        let owner = call(&db, "createIndex", &request)?;
        assert_eq!(owner.as_f64(), Some(expected));
        owners.push(owner);
    }
    let check_tokens = |db: &JsValue| -> Result<(), JsValue> {
        for (property, query, expected) in [
            ("strict", "ox", 0),
            ("strict", "cat", 1),
            ("standard", "q", 0),
            ("standard", "ox", 1),
            ("zero", "q", 1),
            ("huge", "cat", 0),
        ] {
            assert_eq!(tokenizer_search_count(db, property, query)?, expected);
        }
        Ok(())
    };
    check_tokens(&db)?;
    for owner in &owners {
        call(&db, "rebuildIndex", owner)?;
    }
    check_tokens(&db)?;
    let snapshot = call(&db, "exportSnapshot", &JsValue::UNDEFINED)?;
    let bytes = js_sys::Uint8Array::new(&snapshot).to_vec();
    let restored: JsValue = Database::import_snapshot(&bytes)?.into();
    check_tokens(&restored)?;
    for owner in &owners {
        call(&restored, "rebuildIndex", owner)?;
    }
    check_tokens(&restored)?;
    assert_eq!(call(&restored, "dropIndex", &strict)?.as_bool(), Some(true));
    assert_eq!(
        call(&restored, "dropIndex", &strict)?.as_bool(),
        Some(false)
    );
    assert!(call(&restored, "rebuildIndex", &strict).is_err());
    let replacement = call(
        &restored,
        "createIndex",
        &js_sys::eval("({kind:'text',label:'Doc',property:'strict',minTokenLength:3})")?,
    )?;
    assert_eq!(replacement.as_f64(), Some(4.0));
    Ok(())
}

#[cfg(not(feature = "text-index"))]
#[wasm_bindgen_test]
fn index_input_tokenizer_feature_disabled_preserves_owner_floor() -> Result<(), JsValue> {
    let db: JsValue = Database::new()?.into();
    for option in ["", ",minTokenLength:0", ",minTokenLength:3"] {
        let request = js_sys::eval(&format!(
            "({{kind:'text',label:'Doc',property:'text'{option}}})"
        ))?;
        let error = call(&db, "createIndex", &request).expect_err("disabled Text must reject");
        let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))?;
        assert!(
            message
                .as_string()
                .is_some_and(|text| text.to_lowercase().contains("text")
                    && (text.contains("feature") || text.contains("support")))
        );
    }
    assert_eq!(
        call(
            &db,
            "createIndex",
            &js_sys::eval("({property:'unchanged'})")?
        )?
        .as_f64(),
        Some(0.0)
    );
    Ok(())
}
