//! Correlated equality must keep predicate and snapshot semantics.
#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{GraphPath, PropertyKey, Value};
use grafeo_engine::transaction::IsolationLevel;
use grafeo_engine::{Config, CreateIndexRequest, GrafeoDB, IndexCreateKind};
use std::collections::{BTreeMap, HashMap};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn database() -> Result<GrafeoDB, Box<dyn std::error::Error>> {
    Ok(GrafeoDB::with_config(
        Config::in_memory().with_gc_interval(0),
    )?)
}

fn add_index(db: &GrafeoDB) -> TestResult {
    db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("node_lookup".into()),
        label: None,
        property: "lookup".into(),
        kind: IndexCreateKind::Property,
    })?;
    Ok(())
}

fn row(entries: &[(&str, Value)]) -> Value {
    Value::Map(
        entries
            .iter()
            .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
            .collect::<BTreeMap<_, _>>()
            .into(),
    )
}

fn params(values: Vec<Value>) -> HashMap<String, Value> {
    HashMap::from([("rows".into(), Value::List(values.into()))])
}

#[test]
fn unwind_two_endpoint_lookup_preserves_duplicates_nulls_and_missing_keys() -> TestResult {
    let db = database()?;
    let session = db.session();
    for (lookup, tag) in [(1, "a"), (1, "b"), (2, "t")] {
        session.create_node_with_props(
            &["Node"],
            [("lookup", Value::Int64(lookup)), ("tag", Value::from(tag))],
        )?;
    }
    add_index(&db)?;
    let edge = row(&[("s", Value::Int64(1)), ("t", Value::Int64(2))]);
    session.execute_with_params(
        "UNWIND $rows AS e MATCH (s:Node {lookup:e.s}), (t:Node {lookup:e.t}) CREATE (s)-[:LINKS]->(t)",
        params(vec![edge.clone(), edge,
            row(&[("s", Value::Int64(99)), ("t", Value::Int64(2))]),
            row(&[("s", Value::Null), ("t", Value::Int64(2))])]),
    )?;
    let result =
        session.execute("MATCH (s:Node)-[:LINKS]->(t:Node) RETURN s.tag, t.tag ORDER BY s.tag")?;
    assert_eq!(
        result.rows(),
        &[
            vec![Value::from("a"), Value::from("t")],
            vec![Value::from("a"), Value::from("t")],
            vec![Value::from("b"), Value::from("t")],
            vec![Value::from("b"), Value::from("t")],
        ]
    );
    Ok(())
}

#[test]
fn unwind_lookup_matches_unoptimized_numeric_and_null_predicates() -> TestResult {
    let db = database()?;
    let session = db.session();
    let values = vec![
        Value::Int64(0),
        Value::Int64(1),
        Value::Float64(1.0),
        Value::Float64(f64::EPSILON / 2.0),
        Value::Float64(-0.0),
        Value::from("1"),
        Value::from("01"),
        Value::from("1.0"),
        Value::Bool(true),
        Value::Null,
        Value::List(vec![Value::Int64(1)].into()),
    ];
    for (ordinal, value) in values.iter().enumerate() {
        session.create_node_with_props(
            &["Node"],
            [
                ("lookup", value.clone()),
                ("tag", Value::Int64(i64::try_from(ordinal)?)),
            ],
        )?;
    }
    add_index(&db)?;
    let rows = values
        .into_iter()
        .enumerate()
        .map(|(ordinal, value)| {
            Ok(row(&[
                ("v", value),
                ("ordinal", Value::Int64(i64::try_from(ordinal)?)),
            ]))
        })
        .collect::<Result<Vec<_>, std::num::TryFromIntError>>()?;
    let optimized = session.execute_with_params(
        "UNWIND $rows AS e MATCH (n:Node {lookup:e.v}) RETURN e.ordinal, n.tag ORDER BY e.ordinal, n.tag",
        params(rows.clone()),
    )?;
    // OR FALSE is equivalent under three-valued logic, but cannot supply an
    // equality conjunct to the lookup planner. Keep the ordinary filter oracle.
    let ordinary = session.execute_with_params(
        "UNWIND $rows AS e MATCH (n:Node) WHERE (n.lookup = e.v) OR false RETURN e.ordinal, n.tag ORDER BY e.ordinal, n.tag",
        params(rows),
    )?;
    assert_eq!(optimized.rows(), ordinary.rows());
    assert!(
        session
            .execute_with_params(
                "UNWIND $rows AS e MATCH (n:Node {lookup:e.v}) RETURN n.tag",
                params(Vec::new()),
            )?
            .rows()
            .is_empty()
    );
    Ok(())
}

#[test]
fn unwind_lookup_sees_own_overlay_and_old_snapshot_without_current_index_false_negatives()
-> TestResult {
    let db = database()?;
    let writer = db.session();
    let changed = writer.create_node_with_props(&["Node"], [("lookup", Value::Int64(1))])?;
    let deleted = writer.create_node_with_props(&["Node"], [("lookup", Value::Int64(2))])?;
    add_index(&db)?;
    let mut old = db.session();
    old.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)?;
    writer.set_node_property(changed, "lookup", Value::Int64(10))?;
    let query = "UNWIND $rows AS e MATCH (n:Node {lookup:e.v}) RETURN n.lookup ORDER BY n.lookup";
    let old_result =
        old.execute_with_params(query, params(vec![row(&[("v", Value::Int64(1))])]))?;
    assert_eq!(old_result.rows(), &[vec![Value::Int64(1)]]);
    old.rollback()?;

    let mut own = db.session();
    own.begin_transaction()?;
    own.set_node_property(changed, "lookup", Value::Int64(20))?;
    own.create_node_with_props(&["Node"], [("lookup", Value::Int64(30))])?;
    assert!(own.delete_node(deleted));
    let result = own.execute_with_params(
        query,
        params(
            [2, 10, 20, 30]
                .into_iter()
                .map(|v| row(&[("v", Value::Int64(v))]))
                .collect(),
        ),
    )?;
    assert_eq!(
        result.rows(),
        &[vec![Value::Int64(20)], vec![Value::Int64(30)]]
    );
    own.rollback()?;
    let current = writer.execute("MATCH (n:Node) RETURN n.lookup ORDER BY n.lookup")?;
    assert_eq!(
        current.rows(),
        &[vec![Value::Int64(2)], vec![Value::Int64(10)]]
    );
    Ok(())
}

#[test]
fn unwind_range_after_match_preserves_constant_and_per_row_lists() -> TestResult {
    let db = database()?;
    let session = db.session();
    for (tag, steps) in [("a", 1_i64), ("b", 2_i64)] {
        session.create_node_with_props(
            &["Node"],
            [("tag", Value::from(tag)), ("steps", Value::Int64(steps))],
        )?;
    }

    let constant = "MATCH (n:Node) UNWIND range(0, 1) AS i \
                    RETURN n.tag, i ORDER BY n.tag, i";
    let constant_expected = [
        vec![Value::from("a"), Value::Int64(0)],
        vec![Value::from("a"), Value::Int64(1)],
        vec![Value::from("b"), Value::Int64(0)],
        vec![Value::from("b"), Value::Int64(1)],
    ];
    let per_row = "MATCH (n:Node) UNWIND range(0, n.steps) AS i \
                   RETURN n.tag, i ORDER BY n.tag, i";
    let per_row_expected = [
        vec![Value::from("a"), Value::Int64(0)],
        vec![Value::from("a"), Value::Int64(1)],
        vec![Value::from("b"), Value::Int64(0)],
        vec![Value::from("b"), Value::Int64(1)],
        vec![Value::from("b"), Value::Int64(2)],
    ];
    for _ in 0..2 {
        assert_eq!(session.execute(constant)?.rows(), constant_expected);
        assert_eq!(session.execute(per_row)?.rows(), per_row_expected);
    }
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "spill"))]
#[test]
fn cached_literal_lookup_retires_spill_scope_and_refuses_setup_failure() -> TestResult {
    let directory = tempfile::tempdir()?;
    let spill_root = directory.path().join("query-spill");
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_gc_interval(0)
            .with_spill_path(spill_root.clone()),
    )?;
    let session = db.session();
    for id in 0_i64..2 {
        session.create_node_with_props(&["Node"], [("lookup", Value::Int64(id))])?;
    }
    let query =
        "UNWIND [0, 1] AS key MATCH (n:Node {lookup:key}) RETURN n.lookup ORDER BY n.lookup";
    let expected = [vec![Value::Int64(0)], vec![Value::Int64(1)]];
    for _ in 0..2 {
        assert_eq!(session.execute(query)?.rows(), &expected);
        let namespace = spill_root.join(format!("grafeo-store-{}", db.store_id()));
        let entries = std::fs::read_dir(namespace)?.collect::<std::io::Result<Vec<_>>>()?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_name(), ".grafeo-spill-root");
    }
    #[cfg(feature = "cypher")]
    for _ in 0..2 {
        assert_eq!(session.execute_cypher(query)?.rows(), &expected);
        let namespace = spill_root.join(format!("grafeo-store-{}", db.store_id()));
        let entries = std::fs::read_dir(namespace)?.collect::<std::io::Result<Vec<_>>>()?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_name(), ".grafeo-spill-root");
    }

    // The cached plan must still qualify each execution's configured root.
    // Retain the authenticated namespace while substituting its ambient name.
    let retained_root = directory.path().join("retained-spill");
    std::fs::rename(&spill_root, &retained_root)?;
    std::fs::write(&spill_root, b"not a directory")?;
    assert!(matches!(
        session.execute(query),
        Err(grafeo_common::utils::error::Error::Io(_))
    ));
    std::fs::remove_file(&spill_root)?;
    std::fs::rename(&retained_root, &spill_root)?;
    assert_eq!(session.execute(query)?.rows(), &expected);
    let namespace = spill_root.join(format!("grafeo-store-{}", db.store_id()));
    let entries = std::fs::read_dir(namespace)?.collect::<std::io::Result<Vec<_>>>()?;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].file_name(), ".grafeo-spill-root");
    Ok(())
}
