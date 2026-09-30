use super::{
    GrafeoIndexRequest, GrafeoStatus, GrafeoUtf8, GraphPath, IndexCreateKind, decode,
    grafeo_create_index, grafeo_drop_index, grafeo_rebuild_index,
};
use crate::database::{grafeo_free_database, grafeo_open_memory};

fn span(value: &str) -> GrafeoUtf8 {
    GrafeoUtf8 {
        data: value.as_ptr(),
        len: value.len(),
    }
}

#[test]
fn owner_lifecycle_and_missing_rebuild_transport_status() {
    let db = grafeo_open_memory();
    assert!(!db.is_null());
    let request = GrafeoIndexRequest {
        property: span("name"),
        ..Default::default()
    };
    let mut owner = u32::MAX;
    assert_eq!(
        grafeo_create_index(db, &raw const request, &raw mut owner),
        GrafeoStatus::Ok
    );
    assert_ne!(owner, u32::MAX);
    assert_eq!(grafeo_rebuild_index(db, owner), GrafeoStatus::Ok);
    let mut duplicate = u32::MAX;
    assert_ne!(
        grafeo_create_index(db, &raw const request, &raw mut duplicate),
        GrafeoStatus::Ok
    );
    assert_eq!(duplicate, u32::MAX);
    let mut dropped = -1;
    assert_eq!(
        grafeo_drop_index(db, owner, &raw mut dropped),
        GrafeoStatus::Ok
    );
    assert_eq!(dropped, 1);
    assert_eq!(
        grafeo_drop_index(db, owner, &raw mut dropped),
        GrafeoStatus::Ok
    );
    assert_eq!(dropped, 0);
    assert_ne!(grafeo_rebuild_index(db, owner), GrafeoStatus::Ok);
    grafeo_free_database(db);
}

#[test]
fn graph_components_preserve_empty_names_unicode_separators_and_nul() {
    let names = ["", "a/b", "λ\0graph"];
    let components: Vec<_> = names.iter().map(|name| span(name)).collect();
    let request = GrafeoIndexRequest {
        graph: components.as_ptr(),
        graph_count: components.len(),
        property: span("field\0name"),
        ..Default::default()
    };
    let decoded = decode(&request).unwrap();
    assert_eq!(decoded.graph, GraphPath::from_components(&names).unwrap());
    assert_eq!(decoded.property, "field\0name");
    assert_ne!(decoded.graph, GraphPath::root());
    let root = decode(&GrafeoIndexRequest {
        property: span("field"),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(root.graph, GraphPath::root());
}

#[test]
fn malformed_requests_and_null_outputs_fail_before_creation() {
    let db = grafeo_open_memory();
    let mut owner = 123;
    for request in [
        GrafeoIndexRequest {
            kind: 99,
            property: span("name"),
            ..Default::default()
        },
        GrafeoIndexRequest {
            options: 256,
            property: span("name"),
            ..Default::default()
        },
        GrafeoIndexRequest {
            options: 4,
            property: span("name"),
            ..Default::default()
        },
        GrafeoIndexRequest {
            graph_count: 1,
            property: span("name"),
            ..Default::default()
        },
        GrafeoIndexRequest {
            property: GrafeoUtf8 {
                data: std::ptr::null(),
                len: 1,
            },
            ..Default::default()
        },
        GrafeoIndexRequest {
            property: GrafeoUtf8 {
                data: b"\xff".as_ptr(),
                len: 1,
            },
            ..Default::default()
        },
    ] {
        assert_ne!(
            grafeo_create_index(db, &raw const request, &raw mut owner),
            GrafeoStatus::Ok
        );
        assert_eq!(owner, 123);
    }
    let request = GrafeoIndexRequest {
        property: span("name"),
        ..Default::default()
    };
    assert_eq!(
        grafeo_create_index(db, &raw const request, std::ptr::null_mut()),
        GrafeoStatus::ErrorNullPointer
    );
    assert_eq!(
        grafeo_create_index(db, &raw const request, &raw mut owner),
        GrafeoStatus::Ok
    );
    grafeo_free_database(db);
}

#[test]
fn text_minimum_preserves_presence_and_rejects_other_kinds() {
    for value in [0, 3, usize::MAX] {
        let request = GrafeoIndexRequest {
            kind: 2,
            options: 128,
            property: span("body"),
            min_token_length: value,
            ..Default::default()
        };
        assert!(matches!(decode(&request).unwrap().kind,
            IndexCreateKind::Text { min_token_length: Some(actual) } if actual == value));
        let absent = GrafeoIndexRequest {
            options: 0,
            ..request
        };
        assert!(matches!(
            decode(&absent).unwrap().kind,
            IndexCreateKind::Text {
                min_token_length: None
            }
        ));
        for kind in [0, 1, 3] {
            assert!(decode(&GrafeoIndexRequest { kind, ..request }).is_err());
        }
    }
}

#[cfg(feature = "text-index")]
#[test]
fn text_minimum_controls_real_owner_search_and_rebuild() {
    use crate::database::grafeo_create_node;
    for minimum in [None, Some(0), Some(3), Some(usize::MAX)] {
        let db = grafeo_open_memory();
        assert!(!db.is_null());
        let node = grafeo_create_node(
            db,
            c"[\"Doc\"]".as_ptr(),
            c"{\"body\":\"x ox fox\"}".as_ptr(),
        );
        assert_ne!(node, u64::MAX);
        let request = GrafeoIndexRequest {
            kind: 2,
            options: 2 | if minimum.is_some() { 128 } else { 0 },
            label: span("Doc"),
            property: span("body"),
            min_token_length: minimum.unwrap_or(0),
            ..Default::default()
        };
        let mut owner = u32::MAX;
        assert_eq!(
            grafeo_create_index(db, &raw const request, &raw mut owner),
            GrafeoStatus::Ok
        );
        assert_ne!(owner, u32::MAX);
        for _ in 0..2 {
            // SAFETY: this test owns a live handle until grafeo_free_database below.
            let engine = unsafe { &*db }.inner.read();
            for token in ["x", "ox", "fox"] {
                let expected = usize::from(token.len() >= minimum.unwrap_or(2));
                assert_eq!(
                    engine.text_search("Doc", "body", token, 10).unwrap().len(),
                    expected
                );
            }
            drop(engine);
            assert_eq!(grafeo_rebuild_index(db, owner), GrafeoStatus::Ok);
        }
        let mut untouched = u32::MAX;
        assert_ne!(
            grafeo_create_index(db, &raw const request, &raw mut untouched),
            GrafeoStatus::Ok
        );
        assert_eq!(untouched, u32::MAX);
        grafeo_free_database(db);
    }
}

#[cfg(not(feature = "text-index"))]
#[test]
fn disabled_text_minimum_returns_error_without_owner() {
    let db = grafeo_open_memory();
    let request = GrafeoIndexRequest {
        kind: 2,
        options: 2 | 128,
        label: span("Doc"),
        property: span("body"),
        min_token_length: 3,
        ..Default::default()
    };
    let mut untouched = 123;
    assert_ne!(
        grafeo_create_index(db, &raw const request, &raw mut untouched),
        GrafeoStatus::Ok
    );
    assert_eq!(untouched, 123);
    grafeo_free_database(db);
}

#[test]
fn explicit_empty_and_zero_options_are_not_defaults() {
    let request = GrafeoIndexRequest {
        kind: 3,
        options: 1 | 2 | 4 | 8 | 16 | 32 | 64,
        property: span("embedding"),
        ..Default::default()
    };
    let decoded = decode(&request).unwrap();
    assert_eq!(decoded.name.as_deref(), Some(""));
    assert_eq!(decoded.label.as_deref(), Some(""));
    let IndexCreateKind::Vector {
        dimensions,
        metric,
        m,
        ef_construction,
        ef,
        quantization,
    } = decoded.kind
    else {
        panic!("expected Vector request");
    };
    assert_eq!(dimensions, Some(0));
    assert_eq!(m, Some(0));
    assert_eq!(ef_construction, Some(0));
    assert_eq!(ef, None);
    assert_eq!(metric.as_deref(), Some(""));
    assert_eq!(quantization.as_deref(), Some(""));
}

#[cfg(feature = "gql")]
#[test]
fn graph_qualified_owners_do_not_alias_root_or_truncate_nul_components() {
    use crate::database::{grafeo_execute, grafeo_free_result};
    let db = grafeo_open_memory();
    let query = c"CREATE GRAPH scoped";
    let result = grafeo_execute(db, query.as_ptr());
    assert!(!result.is_null());
    grafeo_free_result(result);
    let request = GrafeoIndexRequest {
        property: span("name"),
        ..Default::default()
    };
    let mut root = u32::MAX;
    assert_eq!(
        grafeo_create_index(db, &raw const request, &raw mut root),
        GrafeoStatus::Ok
    );
    let components = [span("scoped")];
    let scoped = GrafeoIndexRequest {
        graph: components.as_ptr(),
        graph_count: 1,
        ..request
    };
    let mut owner = u32::MAX;
    assert_eq!(
        grafeo_create_index(db, &raw const scoped, &raw mut owner),
        GrafeoStatus::Ok
    );
    assert_ne!(root, owner);
    let mut dropped = -1;
    assert_eq!(
        grafeo_drop_index(db, owner, &raw mut dropped),
        GrafeoStatus::Ok
    );
    assert_eq!(dropped, 1);
    assert_eq!(grafeo_rebuild_index(db, root), GrafeoStatus::Ok);
    let missing_components = [span("scoped\0other")];
    let missing = GrafeoIndexRequest {
        graph: missing_components.as_ptr(),
        property: span("unique"),
        ..scoped
    };
    let mut untouched = u32::MAX;
    assert_ne!(
        grafeo_create_index(db, &raw const missing, &raw mut untouched),
        GrafeoStatus::Ok
    );
    assert_eq!(untouched, u32::MAX);
    assert!(!crate::error::grafeo_last_error().is_null());
    grafeo_free_database(db);
}
