//! Tests for filter pushdown optimization.
//!
//! Verifies that the planner pushes equality predicates down to the store level
//! (bypassing DataChunk/expression overhead) and correctly handles:
//! - Index-based pushdown (existing behaviour)
//! - Label-first pushdown (no index, with label)
//! - Compound predicates with remaining non-equality parts
//! - Non-pushable expressions (kept as generic FilterOperator)

#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Builds a small social graph for filter tests.
///
/// 5 Person nodes (Alix/NYC, Gus/NYC, Harm/London, Dave/London, Eve/Paris)
/// 2 Company nodes (Acme, Globex)
fn setup() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "CREATE (a:Person {name: 'Alix', city: 'NYC', age: 30}),
                    (b:Person {name: 'Gus',   city: 'NYC', age: 25}),
                    (c:Person {name: 'Harm', city: 'London', age: 35}),
                    (d:Person {name: 'Dave',  city: 'London', age: 40}),
                    (e:Person {name: 'Eve',   city: 'Paris',  age: 28}),
                    (x:Company {name: 'Acme'}),
                    (y:Company {name: 'Globex'})",
        )
        .unwrap();

    db
}

// ── Equality with label, no index (new pushdown path) ──

#[test]
fn equality_filter_pushdown_without_index() {
    let db = setup();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name, n.city")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
    assert_eq!(result.rows()[0][1], Value::from("NYC"));
}

#[test]
fn compound_equality_pushdown_without_index() {
    let db = setup();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.city = 'NYC' AND n.age = 25 RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Gus"));
}

// ── Equality with property index (existing path still works) ──

#[test]
fn equality_filter_pushdown_with_index() {
    let db = setup();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "name".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Harm' RETURN n.city")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("London"));
}

// ── Compound predicate: equality + range (remaining predicate handling) ──

#[test]
fn mixed_equality_and_range_pushdown() {
    let db = setup();
    let session = db.session();

    // Equality on city pushed down, range on age kept as FilterOperator
    let result = session
        .execute("MATCH (n:Person) WHERE n.city = 'London' AND n.age > 36 RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Dave"));
}

#[test]
fn mixed_equality_and_range_no_match() {
    let db = setup();
    let session = db.session();

    // Equality matches Harm (35) and Dave (40), but range > 50 matches nobody
    let result = session
        .execute("MATCH (n:Person) WHERE n.city = 'London' AND n.age > 50 RETURN n.name")
        .unwrap();

    assert!(result.rows().is_empty());
}

// ── Range-only pushdown ──

#[test]
fn range_filter_pushdown() {
    let db = setup();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.age > 30 RETURN n.name")
        .unwrap();

    // Harm (35) and Dave (40) match
    assert_eq!(result.rows().len(), 2);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Harm")));
    assert!(names.contains(&&Value::from("Dave")));
}

#[test]
fn between_filter_pushdown() {
    let db = setup();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.age >= 28 AND n.age <= 35 RETURN n.name")
        .unwrap();

    // Alix (30), Harm (35), Eve (28) match
    assert_eq!(result.rows().len(), 3);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Alix")));
    assert!(names.contains(&&Value::from("Harm")));
    assert!(names.contains(&&Value::from("Eve")));
}

// ── Non-pushable expressions (stay as generic filter) ──

#[test]
fn non_pushable_expression_filter() {
    let db = setup();
    let session = db.session();

    // String function in predicate: not pushable, uses generic FilterOperator
    let result = session
        .execute("MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
}

// ── No label (no pushdown without label or index) ──

#[test]
fn equality_without_label_or_index_falls_through() {
    let db = setup();
    let session = db.session();

    // No label, no index: falls through to generic FilterOperator
    // Should still return correct results
    let result = session
        .execute("MATCH (n) WHERE n.name = 'Alix' RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
}

// ── Label selectivity: only label-matching nodes checked ──

#[test]
fn label_narrows_scan_correctly() {
    let db = setup();
    let session = db.session();

    // Company has name='Acme' too, but label restricts to Person
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Acme' RETURN n.name")
        .unwrap();

    assert!(result.rows().is_empty());
}

#[test]
fn label_filter_on_company() {
    let db = setup();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Company) WHERE n.name = 'Acme' RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Acme"));
}

// ── OR filter (zone map OR branch, logical OR evaluation) ──

#[test]
fn or_filter_matches_either_side() {
    let db = setup();
    let session = db.session();

    // OR filter: matches Alix (NYC) or Harm (London)
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Harm' RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Alix")));
    assert!(names.contains(&&Value::from("Harm")));
}

#[test]
fn or_filter_matches_no_side() {
    let db = setup();
    let session = db.session();

    // OR filter: neither side matches
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Nobody' OR n.name = 'Ghost' RETURN n.name")
        .unwrap();

    assert!(result.rows().is_empty());
}

#[test]
fn or_filter_matches_one_side() {
    let db = setup();
    let session = db.session();

    // OR filter: only left side matches
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Nobody' RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
}

// ── AND + OR combined (compound logic) ──

#[test]
fn and_or_combined_filter() {
    let db = setup();
    let session = db.session();

    // (city = NYC AND age > 28) OR name = Harm
    // Matches Alix (NYC, 30) and Harm (London, 35)
    let result = session
        .execute(
            "MATCH (n:Person) WHERE (n.city = 'NYC' AND n.age > 28) OR n.name = 'Harm' RETURN n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Alix")));
    assert!(names.contains(&&Value::from("Harm")));
}

// ── Reversed operands: literal on left side ──

#[test]
fn reversed_equality_literal_on_left() {
    let db = setup();
    let session = db.session();

    // Literal on left: 'Alix' = n.name
    let result = session
        .execute("MATCH (n:Person) WHERE 'Alix' = n.name RETURN n.city")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("NYC"));
}

#[test]
fn reversed_range_literal_on_left() {
    let db = setup();
    let session = db.session();

    // Literal on left: 30 < n.age means n.age > 30
    let result = session
        .execute("MATCH (n:Person) WHERE 30 < n.age RETURN n.name")
        .unwrap();

    // Harm (35) and Dave (40) match
    assert_eq!(result.rows().len(), 2);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Harm")));
    assert!(names.contains(&&Value::from("Dave")));
}

#[test]
fn reversed_range_ge_literal_on_left() {
    let db = setup();
    let session = db.session();

    // 35 <= n.age means n.age >= 35
    let result = session
        .execute("MATCH (n:Person) WHERE 35 <= n.age RETURN n.name")
        .unwrap();

    // Harm (35) and Dave (40) match
    assert_eq!(result.rows().len(), 2);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Harm")));
    assert!(names.contains(&&Value::from("Dave")));
}

// ── Property index with remaining predicate ──

#[test]
fn property_index_with_remaining_predicate() {
    let db = setup();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "city".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");
    let session = db.session();

    // Index pushes equality on city, remaining range predicate on age
    let result = session
        .execute("MATCH (n:Person) WHERE n.city = 'NYC' AND n.age > 28 RETURN n.name")
        .unwrap();

    // Only Alix (30) matches both conditions
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
}

// ── NOT/inequality filter (non-pushable, generic FilterOperator) ──

#[test]
fn not_equal_filter() {
    let db = setup();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.city <> 'NYC' RETURN n.name")
        .unwrap();

    // Harm (London), Dave (London), Eve (Paris) match
    assert_eq!(result.rows().len(), 3);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(!names.contains(&&Value::from("Alix")));
    assert!(!names.contains(&&Value::from("Gus")));
}

// ── BETWEEN variations (different boundary inclusivity) ──

#[test]
fn between_exclusive_both_sides() {
    let db = setup();
    let session = db.session();

    // Exclusive both sides: 25 < age < 35
    let result = session
        .execute("MATCH (n:Person) WHERE n.age > 25 AND n.age < 35 RETURN n.name")
        .unwrap();

    // Alix (30) and Eve (28) match
    assert_eq!(result.rows().len(), 2);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Alix")));
    assert!(names.contains(&&Value::from("Eve")));
}

#[test]
fn between_inclusive_lower_exclusive_upper() {
    let db = setup();
    let session = db.session();

    // Inclusive lower, exclusive upper: 25 <= age < 35
    let result = session
        .execute("MATCH (n:Person) WHERE n.age >= 25 AND n.age < 35 RETURN n.name")
        .unwrap();

    // Gus (25), Eve (28), Alix (30) match
    assert_eq!(result.rows().len(), 3);
    let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::from("Gus")));
    assert!(names.contains(&&Value::from("Eve")));
    assert!(names.contains(&&Value::from("Alix")));
}

// ── MVCC visibility: rolled-back nodes must not leak through pushdown ──

#[test]
fn rollback_hides_nodes_from_equality_pushdown() {
    let db = setup();
    let mut session = db.session();

    // Create a node inside a transaction, then roll back
    session.begin_transaction().unwrap();
    session
        .execute("CREATE (:Person {name: 'Ghost', city: 'Nowhere'})")
        .unwrap();
    session.rollback().unwrap();

    // Equality pushdown on label+property must NOT return the rolled-back node
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Ghost' RETURN n.name")
        .unwrap();
    assert!(
        result.rows().is_empty(),
        "rolled-back node leaked through equality pushdown"
    );

    // Original nodes still visible
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn rollback_hides_nodes_from_range_pushdown() {
    let db = setup();
    let mut session = db.session();

    // Create a node inside a transaction, then roll back
    session.begin_transaction().unwrap();
    session
        .execute("CREATE (:Person {name: 'Ghost', city: 'Nowhere', age: 99})")
        .unwrap();
    session.rollback().unwrap();

    // Range pushdown must NOT return the rolled-back node
    let result = session
        .execute("MATCH (n:Person) WHERE n.age > 90 RETURN n.name")
        .unwrap();
    assert!(
        result.rows().is_empty(),
        "rolled-back node leaked through range pushdown"
    );
}

#[test]
fn committed_tx_nodes_visible_in_pushdown() {
    let db = setup();
    let mut session = db.session();

    // Create a node inside a transaction and commit
    session.begin_transaction().unwrap();
    session
        .execute("CREATE (:Person {name: 'Frank', city: 'Berlin', age: 50})")
        .unwrap();
    session.commit().unwrap();

    // Equality pushdown should find the committed node
    let result = session
        .execute("MATCH (n:Person) WHERE n.name = 'Frank' RETURN n.city")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Berlin"));

    // Range pushdown should also find it
    let result = session
        .execute("MATCH (n:Person) WHERE n.age > 45 RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Frank"));
}

#[test]
fn conjuncts_split_and_anchor_on_their_own_scans() {
    // MATCH (a:Person),(b:City) WHERE a.name = 'Ann' AND b.name = 'Rome'
    // The conjunction must be split so each predicate anchors on its own scan,
    // instead of one combined AND filter sitting above a cartesian product.
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    s.execute("CREATE (:Person {name: 'Bob'})").unwrap();
    s.execute("CREATE (:City {name: 'Rome'})").unwrap();
    s.execute("CREATE (:City {name: 'Oslo'})").unwrap();

    // Correctness is unchanged: exactly one (Ann, Rome) row.
    let r = s
        .execute("MATCH (a:Person),(b:City) WHERE a.name = 'Ann' AND b.name = 'Rome' RETURN a.name, b.name")
        .unwrap();
    assert_eq!(r.row_count(), 1);

    // Structure: the combined "And" filter is gone; conjuncts are split.
    let plan = s
        .execute("EXPLAIN MATCH (a:Person),(b:City) WHERE a.name = 'Ann' AND b.name = 'Rome' RETURN a.name, b.name")
        .unwrap();
    let text = format!("{:?}", plan.rows());
    assert!(
        !text.contains(" And "),
        "conjuncts should be split into per-scan filters, not kept as one AND; plan:\n{text}"
    );
}

/// A LIMIT directly over a filter stack pushed its count into the range scan
/// under a residual filter, so the scan stopped before the residual found a
/// match: this query returned no row.
#[cfg(feature = "cypher")]
#[test]
fn limit_is_not_pushed_into_a_range_scan_under_a_residual_filter() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (:Node {id: 'n_0', age: 0, tag: 'early'}), \
                    (:Node {id: 'n_1', age: 1, tag: 'late'})",
        )
        .unwrap();

    for query in [
        "MATCH (n:Node) WHERE n.tag = 'late' AND n.age >= 0 WITH * LIMIT 1 RETURN n.id",
        "MATCH (n:Node) WHERE n.age >= 0 AND n.tag = 'late' WITH * LIMIT 1 RETURN n.id",
    ] {
        let result = session.execute_cypher(query).unwrap();
        assert_eq!(
            result.rows(),
            [[Value::from("n_1")]],
            "{query} must find the late node"
        );
    }
    // The pushdown still applies where the range scan answers the whole filter.
    let result = session
        .execute_cypher("MATCH (n:Node) WHERE n.age >= 0 WITH * LIMIT 1 RETURN n.id")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

// ── GQL range/LIMIT contracts ──

#[test]
fn gql_range_bounds_preserve_order_and_inclusivity() {
    let db = setup();
    let session = db.session();

    let cases = [
        (
            "MATCH (n:Person) WHERE n.age >= 25 AND n.age <= 35 \
             RETURN n.name ORDER BY n.name",
            vec![
                vec![Value::from("Alix")],
                vec![Value::from("Eve")],
                vec![Value::from("Gus")],
                vec![Value::from("Harm")],
            ],
        ),
        (
            "MATCH (n:Person) WHERE n.age <= 35 AND n.age >= 25 \
             RETURN n.name ORDER BY n.name",
            vec![
                vec![Value::from("Alix")],
                vec![Value::from("Eve")],
                vec![Value::from("Gus")],
                vec![Value::from("Harm")],
            ],
        ),
        (
            "MATCH (n:Person) WHERE n.age > 25 AND n.age < 35 \
             RETURN n.name ORDER BY n.name",
            vec![vec![Value::from("Alix")], vec![Value::from("Eve")]],
        ),
        (
            "MATCH (n:Person) WHERE n.age < 35 AND n.age > 25 \
             RETURN n.name ORDER BY n.name",
            vec![vec![Value::from("Alix")], vec![Value::from("Eve")]],
        ),
    ];

    for (query, expected) in cases {
        assert_eq!(session.execute(query).unwrap().rows(), expected, "{query}");
    }
}

#[test]
fn gql_return_limit_keeps_late_residual_match() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (:Node {id: 'n_0', age: 0, tag: 'early'}), \
                    (:Node {id: 'n_1', age: 1, tag: 'late'})",
        )
        .unwrap();

    let result = session
        .execute(
            "MATCH (n:Node) WHERE n.tag = 'late' AND n.age >= 0 \
             RETURN n.id LIMIT 1",
        )
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("n_1")]]);
}

#[test]
fn gql_order_by_blocks_range_limit_pushdown() {
    let db = setup();
    let result = db
        .session()
        .execute(
            "MATCH (n:Person) WHERE n.age >= 25 \
             RETURN n.name ORDER BY n.name LIMIT 2",
        )
        .unwrap();
    assert_eq!(
        result.rows(),
        [[Value::from("Alix")], [Value::from("Dave")]]
    );
}

#[test]
fn gql_distinct_blocks_range_limit_pushdown() {
    let db = setup();
    let result = db
        .session()
        .execute(
            "MATCH (n:Person) WHERE n.age >= 25 \
             RETURN DISTINCT n.city LIMIT 2",
        )
        .unwrap();
    assert_eq!(
        result.rows(),
        [[Value::from("NYC")], [Value::from("London")]]
    );
}

#[test]
fn gql_skip_blocks_range_limit_pushdown() {
    let db = setup();
    let result = db
        .session()
        .execute(
            "MATCH (n:Person) WHERE n.age >= 25 \
             RETURN n.name SKIP 1 LIMIT 2",
        )
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("Gus")], [Value::from("Harm")]]);
}

#[test]
fn gql_transaction_own_property_update_is_visible_to_range_filter() {
    let db = setup();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Person {name: 'Gus'}) SET n.age = 50")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person) WHERE n.age >= 50 RETURN n.name")
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("Gus")]]);
    session.rollback().unwrap();
}

// ── GQL label-scan LIMIT contracts ──

#[test]
fn gql_label_limit_does_not_truncate_count() {
    let db = setup();
    let result = db
        .session()
        .execute("MATCH (n:Person) RETURN count(n) LIMIT 1")
        .unwrap();
    assert_eq!(result.rows(), [[Value::Int64(5)]]);
}

#[test]
fn gql_label_limit_preserves_order_by() {
    let db = setup();
    let result = db
        .session()
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 2")
        .unwrap();
    assert_eq!(
        result.rows(),
        [[Value::from("Alix")], [Value::from("Dave")]]
    );
}

#[test]
fn gql_label_limit_preserves_distinct() {
    let db = setup();
    let result = db
        .session()
        .execute("MATCH (n:Person) RETURN DISTINCT n.city LIMIT 2")
        .unwrap();
    assert_eq!(
        result.rows(),
        [[Value::from("NYC")], [Value::from("London")]]
    );
}

#[test]
fn gql_label_limit_preserves_skip() {
    let db = setup();
    let result = db
        .session()
        .execute("MATCH (n:Person) RETURN n.name SKIP 1 LIMIT 2")
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("Gus")], [Value::from("Harm")]]);
}

#[test]
fn gql_label_limit_keeps_late_residual_match() {
    let db = setup();
    let result = db
        .session()
        .execute("MATCH (n:Person) WHERE n.city = 'Paris' RETURN n.name LIMIT 1")
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("Eve")]]);
}

#[test]
fn gql_label_scan_sees_own_create_and_delete() {
    let db = setup();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE (:Person {name: 'Frank', city: 'Berlin', age: 50})")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name LIMIT 6")
        .unwrap();
    assert_eq!(
        result.rows(),
        [
            [Value::from("Alix")],
            [Value::from("Gus")],
            [Value::from("Harm")],
            [Value::from("Dave")],
            [Value::from("Eve")],
            [Value::from("Frank")],
        ]
    );
    let result = session
        .execute("MATCH (n:Person) RETURN count(n) LIMIT 1")
        .unwrap();
    assert_eq!(result.rows(), [[Value::Int64(6)]]);

    session
        .execute("MATCH (n:Person) WHERE n.name = 'Frank' DELETE n")
        .unwrap();
    let result = session
        .execute("MATCH (n:Person) RETURN n.name LIMIT 6")
        .unwrap();
    assert_eq!(
        result.rows(),
        [
            [Value::from("Alix")],
            [Value::from("Gus")],
            [Value::from("Harm")],
            [Value::from("Dave")],
            [Value::from("Eve")],
        ]
    );
    let result = session
        .execute("MATCH (n:Person) RETURN count(n) LIMIT 1")
        .unwrap();
    assert_eq!(result.rows(), [[Value::Int64(5)]]);
    session.rollback().unwrap();
}

#[test]
fn gql_bare_label_limit_zero_is_empty() {
    let db = setup();
    let result = db
        .session()
        .execute("MATCH (n:Person) RETURN n.name LIMIT 0")
        .unwrap();
    assert!(result.rows().is_empty());
}
