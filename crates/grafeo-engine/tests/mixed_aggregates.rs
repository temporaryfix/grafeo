//! Regression: a RETURN mixing an aliased and an un-aliased aggregate must plan
//! and execute. The un-aliased aggregate's column was named `"sum(...)"` by the
//! planner while the post-return projection referenced the synthetic `"_agg_N"`,
//! so the query failed with `Undefined variable '_agg_1'`.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn mixed_aliased_and_unaliased_aggregates() {
    let db = GrafeoDB::new_in_memory();
    for age in [30_i64, 40] {
        let n = db.create_node(&["Person"]);
        db.set_node_property(n, "age", Value::Int64(age))
            .expect("set node property");
    }

    let res = db
        .session()
        .execute("MATCH (p:Person) RETURN count(p) AS cnt, sum(p.age)")
        .expect("mixed aliased + un-aliased aggregates must plan and run");

    assert_eq!(res.rows().len(), 1, "one aggregate row");
    let row = &res.rows()[0];
    assert_eq!(row.len(), 2, "two columns: cnt + sum");
    assert_eq!(row[0], Value::Int64(2), "count(p) = 2");
    // sum(30, 40) = 70 (Int or Float depending on accumulator).
    let sum_ok = matches!(row[1], Value::Int64(70))
        || matches!(row[1], Value::Float64(s) if (s - 70.0).abs() < 1e-9);
    assert!(sum_ok, "sum(p.age) = 70, got {:?}", row[1]);
}
