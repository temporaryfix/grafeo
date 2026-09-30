//! Regression: `DISTINCT` and `GROUP BY` must treat `-0.0` and `+0.0` as equal
//! (IEEE-754 `-0.0 == 0.0` is true; every query language groups them together).
//!
//! Both `DistinctOperator` and `HashAggregateOperator` keyed floats by raw
//! `f64::to_bits()`, and `(-0.0).to_bits() != (0.0).to_bits()`, so the two zeros
//! were treated as distinct values / separate groups.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn setup() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["N"]);
    let b = db.create_node(&["N"]);
    // Set the two signed zeros precisely via the imperative API (a GQL literal
    // could be constant-folded).
    db.set_node_property(a, "val", Value::Float64(0.0))
        .expect("set node property");
    db.set_node_property(b, "val", Value::Float64(-0.0))
        .expect("set node property");
    db
}

#[test]
fn distinct_collapses_signed_zero() {
    let db = setup();
    let res = db
        .session()
        .execute("MATCH (n:N) RETURN DISTINCT n.val")
        .unwrap();
    assert_eq!(
        res.rows().len(),
        1,
        "+0.0 and -0.0 must be a single DISTINCT value, got {:?}",
        res.rows()
    );
}

#[test]
fn group_by_collapses_signed_zero() {
    let db = setup();
    let res = db
        .session()
        .execute("MATCH (n:N) RETURN n.val, count(*) AS c")
        .unwrap();
    assert_eq!(
        res.rows().len(),
        1,
        "+0.0 and -0.0 must form a single group, got {:?}",
        res.rows()
    );
}

#[test]
fn distinct_nested_lists_collapse_signed_zero_and_keep_first_bits() {
    use grafeo_engine::query::executor::{ExecutionOptions, ResultLimits};
    use std::collections::HashMap;

    let db = GrafeoDB::new_in_memory();
    let nested = |zero| {
        Value::List(vec![Value::Null, Value::List(vec![Value::Float64(zero)].into())].into())
    };
    let first = nested(-0.0);
    let params = HashMap::from([(
        "values".into(),
        Value::List(vec![first.clone(), nested(0.0), first.clone()].into()),
    )]);
    let query = "UNWIND $values AS value RETURN DISTINCT value";
    let eager = db.execute_with_params(query, params.clone()).unwrap();
    let streamed = db
        .stream_with_options(query, params, ExecutionOptions::default())
        .unwrap()
        .collect(ResultLimits::default())
        .unwrap();
    for result in [eager, streamed] {
        assert_eq!(result.rows(), &[vec![first.clone()]]);
        let Value::List(outer) = &result.rows()[0][0] else {
            panic!("list witness lost");
        };
        let Value::List(inner) = &outer[1] else {
            panic!("nested list witness lost");
        };
        let Value::Float64(zero) = inner[0] else {
            panic!("float witness lost");
        };
        assert_eq!(
            zero.to_bits(),
            (-0.0f64).to_bits(),
            "DISTINCT retains the original first operand"
        );
    }
}

#[test]
fn distinct_nested_maps_collapse_signed_zero_and_keep_first_bits() {
    use grafeo_common::types::PropertyKey;
    use grafeo_engine::query::executor::{ExecutionOptions, ResultLimits};
    use std::collections::{BTreeMap, HashMap};

    let db = GrafeoDB::new_in_memory();
    let nested = |zero| {
        Value::Map(
            BTreeMap::from([
                (
                    PropertyKey::new("value"),
                    Value::List(vec![Value::Float64(zero), Value::Null].into()),
                ),
                (
                    PropertyKey::new("empty"),
                    Value::Map(BTreeMap::new().into()),
                ),
            ])
            .into(),
        )
    };
    let first = nested(-0.0);
    let params = HashMap::from([(
        "values".into(),
        Value::List(vec![first.clone(), nested(0.0), first.clone()].into()),
    )]);
    let query = "UNWIND $values AS value RETURN DISTINCT value";
    let eager = db.execute_with_params(query, params.clone()).unwrap();
    let streamed = db
        .stream_with_options(query, params, ExecutionOptions::default())
        .unwrap()
        .collect(ResultLimits::default())
        .unwrap();
    for result in [eager, streamed] {
        assert_eq!(result.rows(), &[vec![first.clone()]]);
        let Value::Map(map) = &result.rows()[0][0] else {
            panic!("map witness lost");
        };
        let Value::List(values) = &map["value"] else {
            panic!("nested list witness lost");
        };
        let Value::Float64(zero) = values[0] else {
            panic!("float witness lost");
        };
        assert_eq!(
            zero.to_bits(),
            (-0.0f64).to_bits(),
            "DISTINCT retains the original first operand"
        );
    }
}
