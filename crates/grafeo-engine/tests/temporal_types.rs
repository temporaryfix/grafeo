//! Integration tests for temporal types: Date, Time, Duration.
//!
//! Tests temporal constructors, arithmetic, comparisons, and typed literals
//! across GQL, Cypher, and SPARQL query languages.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test temporal_types
//! ```

#[cfg(any(feature = "cypher", feature = "gql"))]
use grafeo_common::types::Value;
#[cfg(any(
    feature = "cypher",
    feature = "gql",
    all(feature = "sparql", feature = "triple-store")
))]
use grafeo_engine::GrafeoDB;

/// Creates 2 Person nodes (Alix birthday 1990-06-15, Gus birthday 2000-01-01).
#[cfg(any(feature = "cypher", feature = "gql"))]
fn create_test_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                (
                    "birthday",
                    Value::Date(grafeo_common::types::Date::from_ymd(1990, 6, 15).unwrap()),
                ),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                (
                    "birthday",
                    Value::Date(grafeo_common::types::Date::from_ymd(2000, 1, 1).unwrap()),
                ),
            ],
        )
        .unwrap();
    let _ = session.commit();
    db
}

// ============================================================================
// Cypher temporal constructors (standalone RETURN)
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute_cypher("RETURN date('2024-01-15') AS d").unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 1);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute_cypher("RETURN duration('P1Y') AS d").unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 12);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_time_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute_cypher("RETURN time('23:59:59') AS t").unwrap();
    let row = &result.rows()[0];
    let t = row[0].as_time().expect("expected Time value");
    assert_eq!(t.hour(), 23);
    assert_eq!(t.minute(), 59);
    assert_eq!(t.second(), 59);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_datetime_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN datetime('2024-03-15T14:30:00Z') AS dt")
        .unwrap();
    let row = &result.rows()[0];
    let ts = row[0].as_timestamp().expect("expected Timestamp value");
    let date = ts.to_date();
    assert_eq!(date.year(), 2024);
    assert_eq!(date.month(), 3);
    assert_eq!(date.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_with_components() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration('P1Y2M3D') AS d")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 14); // 1Y2M = 14 months
    assert_eq!(d.days(), 3);
}

// ============================================================================
// GQL typed literal syntax (needs MATCH context)
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATE '2024-03-15' AS d LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "gql")]
fn gql_time_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN TIME '14:30:00' AS t LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let t = row[0].as_time().expect("expected Time value");
    assert_eq!(t.hour(), 14);
    assert_eq!(t.minute(), 30);
    assert_eq!(t.second(), 0);
}

#[test]
#[cfg(feature = "gql")]
fn gql_duration_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DURATION 'P1Y2M3D' AS d LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 14);
    assert_eq!(d.days(), 3);
}

#[test]
#[cfg(feature = "gql")]
fn gql_datetime_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATETIME '2024-03-15T14:30:00Z' AS dt LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let ts = row[0].as_timestamp().expect("expected Timestamp value");
    let date = ts.to_date();
    assert_eq!(date.year(), 2024);
    assert_eq!(date.month(), 3);
    assert_eq!(date.day(), 15);
}

// ============================================================================
// Component extraction
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_year_month_day_extraction() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher(
            "RETURN year(date('2024-03-15')) AS y, month(date('2024-03-15')) AS m, day(date('2024-03-15')) AS d",
        )
        .unwrap();
    let row = &result.rows()[0];
    assert_eq!(row[0].as_int64(), Some(2024));
    assert_eq!(row[1].as_int64(), Some(3));
    assert_eq!(row[2].as_int64(), Some(15));
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_hour_minute_second_extraction() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher(
            "RETURN hour(time('14:30:45')) AS h, minute(time('14:30:45')) AS m, second(time('14:30:45')) AS s",
        )
        .unwrap();
    let row = &result.rows()[0];
    assert_eq!(row[0].as_int64(), Some(14));
    assert_eq!(row[1].as_int64(), Some(30));
    assert_eq!(row[2].as_int64(), Some(45));
}

// ============================================================================
// Temporal arithmetic
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_plus_duration() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN date('2024-01-15') + duration('P1M') AS next_month")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 2);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_minus_date() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN date('2024-03-15') - date('2024-01-01') AS diff")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    // 74 days between Jan 1 and Mar 15
    assert_eq!(d.days(), 74);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_arithmetic() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration('P1Y') + duration('P6M') AS combined")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 18); // 12 + 6
}

// ============================================================================
// GQL temporal arithmetic with typed literals (MATCH context)
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_plus_duration() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATE '2024-01-15' + DURATION 'P1M' AS next_month LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 2);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "gql")]
fn gql_date_minus_date() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATE '2024-03-15' - DATE '2024-01-01' AS diff LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.days(), 74);
}

// ============================================================================
// Temporal comparison in WHERE
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_comparison_where() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) WHERE p.birthday > DATE '1995-01-01' RETURN p.name AS name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Gus"));
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_comparison_where() {
    let db = create_test_db();
    let result = db
        .execute_cypher(
            "MATCH (p:Person) WHERE p.birthday > date('1995-01-01') RETURN p.name AS name",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Gus"));
}

// ============================================================================
// JSON parameter roundtrip
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_json_param_roundtrip() {
    let db = create_test_db();
    let mut params = std::collections::HashMap::new();
    params.insert(
        "d".to_string(),
        Value::Date(grafeo_common::types::Date::from_ymd(2024, 6, 15).unwrap()),
    );
    let result = db
        .execute_with_params("MATCH (p:Person) RETURN $d AS val LIMIT 1", params)
        .unwrap();
    let d = result.rows()[0][0]
        .as_date()
        .expect("expected Date value from param");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 6);
    assert_eq!(d.day(), 15);
}

// ============================================================================
// SPARQL xsd: typed literals
// ============================================================================

#[test]
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn sparql_xsd_date_literal() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::config::{Config, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))?;
    let result = db
        .execute_sparql(
            r#"SELECT ?d WHERE { BIND("2024-03-15"^^<http://www.w3.org/2001/XMLSchema#date> AS ?d) }"#,
        )
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.column_count(), 1);
    let d = result.rows()[0][0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
    Ok(())
}

#[test]
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn sparql_xsd_duration_literal() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::config::{Config, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))?;
    let result = db
        .execute_sparql(
            r#"SELECT ?d WHERE { BIND("P1Y6M"^^<http://www.w3.org/2001/XMLSchema#duration> AS ?d) }"#,
        )
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.column_count(), 1);
    let d = result.rows()[0][0]
        .as_duration()
        .expect("expected Duration value");
    assert_eq!(d.months(), 18);
    Ok(())
}

// ============================================================================
// Temporal map constructors: date({year:...}), time({hour:...}), etc.
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN date({year: 2024, month: 3, day: 15}) AS d")
        .unwrap();
    let d = result.rows()[0][0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_map_defaults() {
    let db = GrafeoDB::new_in_memory();
    // month and day should default to 1 when omitted
    let result = db.execute_cypher("RETURN date({year: 2024}) AS d").unwrap();
    let d = result.rows()[0][0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 1);
    assert_eq!(d.day(), 1);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_time_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN time({hour: 14, minute: 30, second: 45}) AS t")
        .unwrap();
    let t = result.rows()[0][0].as_time().expect("expected Time value");
    assert_eq!(t.hour(), 14);
    assert_eq!(t.minute(), 30);
    assert_eq!(t.second(), 45);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_datetime_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher(
            "RETURN datetime({year: 2024, month: 3, day: 15, hour: 14, minute: 30}) AS dt",
        )
        .unwrap();
    let ts = result.rows()[0][0]
        .as_timestamp()
        .expect("expected Timestamp value");
    let d = ts.to_date();
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
    let t = ts.to_time();
    assert_eq!(t.hour(), 14);
    assert_eq!(t.minute(), 30);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration({years: 1, months: 2, days: 3, hours: 4}) AS dur")
        .unwrap();
    let d = result.rows()[0][0]
        .as_duration()
        .expect("expected Duration value");
    assert_eq!(d.months(), 14); // 1 year + 2 months
    assert_eq!(d.days(), 3);
    // 4 hours in nanoseconds = 4 * 3_600_000_000_000
    assert_eq!(d.nanos(), 4 * 3_600_000_000_000);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_map_with_weeks() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration({weeks: 2, days: 3}) AS dur")
        .unwrap();
    let d = result.rows()[0][0]
        .as_duration()
        .expect("expected Duration value");
    assert_eq!(d.days(), 17); // 2 weeks + 3 days
}

/// Equality must reach the shared predicate evaluator, including the callers
/// that use it for membership, CASE, NULLIF and mutation selection.
#[cfg(feature = "gql")]
fn assert_temporal_predicate_equality(value: Value, equal: Value, different: Value) {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for (id, stored) in [(1, value), (2, different), (3, Value::Null)] {
        session
            .create_node_with_props(
                &["TemporalEquality"],
                [("id", Value::Int64(id)), ("value", stored)],
            )
            .unwrap();
    }
    session.commit().unwrap();
    for (predicate, expected) in [
        ("n.value = $needle", vec![1]),
        ("n.value <> $needle", vec![2]),
        ("n.value IN [null, $needle]", vec![1]),
        (
            "CASE n.value WHEN $needle THEN true ELSE false END",
            vec![1],
        ),
        (
            "n.value IS NOT NULL AND NULLIF(n.value, $needle) IS NULL",
            vec![1],
        ),
        ("n.value = null", vec![]),
        ("n.value = '2024-11-01'", vec![]),
    ] {
        let result = db
            .execute_with_params(
                &format!("MATCH (n:TemporalEquality) WHERE {predicate} RETURN n.id ORDER BY n.id"),
                std::collections::HashMap::from([("needle".to_owned(), equal.clone())]),
            )
            .unwrap();
        let expected: Vec<Vec<Value>> = expected
            .into_iter()
            .map(|id| vec![Value::Int64(id)])
            .collect();
        assert_eq!(result.rows(), expected, "{predicate}, needle={equal:?}");
    }
    db.execute_with_params(
        "MATCH (n:TemporalEquality) WHERE n.value = $needle SET n.selected = true",
        std::collections::HashMap::from([("needle".to_owned(), equal)]),
    )
    .unwrap();
    assert_eq!(
        db.execute("MATCH (n:TemporalEquality) WHERE n.selected = true RETURN n.id")
            .unwrap()
            .rows(),
        vec![vec![Value::Int64(1)]]
    );
}

#[test]
#[cfg(feature = "gql")]
fn temporal_predicate_date_equality() {
    use grafeo_common::types::Date;
    let value = Value::Date(Date::parse("2024-11-01").unwrap());
    assert_temporal_predicate_equality(
        value.clone(),
        value,
        Value::Date(Date::parse("2024-11-02").unwrap()),
    );
}

#[test]
#[cfg(feature = "gql")]
fn temporal_predicate_time_equality() {
    use grafeo_common::types::Time;
    let value = Value::Time(Time::parse("12:30:00.123456").unwrap());
    assert_temporal_predicate_equality(
        value.clone(),
        value,
        Value::Time(Time::parse("12:30:00.123457").unwrap()),
    );
}

#[test]
#[cfg(feature = "gql")]
fn temporal_predicate_timestamp_equality() {
    use grafeo_common::types::Timestamp;
    let value = Value::Timestamp(Timestamp::from_micros(1_234_567));
    assert_temporal_predicate_equality(
        value.clone(),
        value,
        Value::Timestamp(Timestamp::from_micros(1_234_568)),
    );
}

#[test]
#[cfg(feature = "gql")]
fn temporal_predicate_duration_equality() {
    use grafeo_common::types::Duration;
    let value = Value::Duration(Duration::parse("P1M").unwrap());
    assert_temporal_predicate_equality(
        value.clone(),
        value,
        Value::Duration(Duration::parse("P30D").unwrap()),
    );
}

#[test]
#[cfg(feature = "gql")]
fn temporal_predicate_zoned_instant_equality() {
    use grafeo_common::types::ZonedDatetime;
    assert_temporal_predicate_equality(
        Value::ZonedDatetime(ZonedDatetime::parse("2024-11-01T12:00:00+02:00").unwrap()),
        Value::ZonedDatetime(ZonedDatetime::parse("2024-11-01T10:00:00Z").unwrap()),
        Value::ZonedDatetime(ZonedDatetime::parse("2024-11-01T12:00:00Z").unwrap()),
    );
}

#[test]
#[cfg(feature = "gql")]
fn temporal_predicate_date_index_and_retained_snapshot() {
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:Event {id: 1, date: DATE '2024-11-01'})")
            .unwrap();
        db.execute("INSERT (:Event {id: 2, date: DATE '2024-11-02'})")
            .unwrap();
        if indexed {
            db.execute("CREATE INDEX event_date FOR (e:Event) ON (e.date)")
                .unwrap();
        }
        let old_epoch = db.current_epoch();
        let query = "MATCH (e:Event) WHERE e.date = DATE '2024-11-01' RETURN e.id";
        assert_eq!(
            db.execute(query).unwrap().rows(),
            vec![vec![Value::Int64(1)]]
        );
        db.execute(
            "MATCH (e:Event) WHERE e.date = DATE '2024-11-01' SET e.date = DATE '2024-11-03'",
        )
        .unwrap();
        assert!(db.execute(query).unwrap().rows().is_empty());
        assert_eq!(
            db.session()
                .execute_at_epoch(query, old_epoch)
                .unwrap()
                .rows(),
            vec![vec![Value::Int64(1)]],
            "indexed={indexed}"
        );
    }
}
