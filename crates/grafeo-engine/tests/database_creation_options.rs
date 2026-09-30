//! Integration tests for database creation options (0.4.3).
//!
//! Verifies GraphModel, DurabilityMode, Config::validate(), query routing,
//! schema_constraints, and inspection API.

use grafeo_engine::{Config, ConfigError, DurabilityMode, GrafeoDB, GraphModel};

// --- GraphModel routing tests ---

#[test]
fn lpg_database_executes_gql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[cfg(feature = "triple-store")]
#[test]
fn rdf_database_rejects_gql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    let result = session.execute("MATCH (p:Person) RETURN p.name");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("RDF database"),
        "Expected RDF error, got: {err_msg}"
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn rdf_database_executes_sparql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    // Simple SPARQL query - may return empty but should not error
    let result = session.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1");
    assert!(result.is_ok(), "SPARQL on RDF db should work: {result:?}");
}

#[cfg(feature = "triple-store")]
#[test]
fn database_info_reports_the_configured_native_model() {
    use grafeo_engine::admin::DatabaseMode;

    let rdf = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let triple = grafeo_core::graph::rdf::Triple::new(
        grafeo_core::graph::rdf::Term::iri("http://ex.org/s"),
        grafeo_core::graph::rdf::Term::iri("http://ex.org/p"),
        grafeo_core::graph::rdf::Term::literal("v"),
    );
    rdf.batch_insert_rdf([triple]).unwrap();
    let info = rdf.info();
    assert_eq!(info.mode, DatabaseMode::Rdf);
    assert_eq!(info.node_count, 1, "RDF node_count is distinct subjects");
    assert_eq!(info.edge_count, 1, "RDF edge_count is triples");
    assert_eq!(
        info.features.iter().any(|feature| feature == "gql"),
        cfg!(feature = "gql"),
        "inspection must report compiled features rather than assumed defaults"
    );

    #[cfg(feature = "lpg")]
    {
        let both =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        assert_eq!(both.info().mode, DatabaseMode::Both);
    }
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn lpg_database_rejects_explicit_sparql_at_every_entry_point() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let session = db.session();
    let result = session
        .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1")
        .unwrap_err();
    assert!(
        result.to_string().contains("LPG database"),
        "explicit Session SPARQL must honor GraphModel: {result}"
    );

    let with_params = session
        .execute_sparql_with_params(
            "SELECT ?s WHERE { ?s ?p ?o }",
            std::collections::HashMap::new(),
        )
        .unwrap_err();
    assert!(with_params.to_string().contains("LPG database"));

    let explain = db
        .execute_sparql("EXPLAIN SELECT ?s WHERE { ?s ?p ?o }")
        .unwrap_err();
    assert!(
        explain.to_string().contains("LPG database"),
        "GrafeoDB EXPLAIN must not bypass the Session boundary: {explain}"
    );
}

#[cfg(feature = "cypher")]
#[test]
fn lpg_database_executes_cypher() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let result = session.execute_cypher("MATCH (p:Person) RETURN p.name");
    assert!(result.is_ok());
}

#[cfg(all(feature = "cypher", feature = "triple-store"))]
#[test]
fn rdf_database_rejects_explicit_cypher() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    let result = session
        .execute_cypher("MATCH (p:Person) RETURN p.name")
        .unwrap_err();
    assert!(
        result.to_string().contains("RDF database"),
        "explicit Cypher must honor GraphModel: {result}"
    );
}

#[cfg(all(feature = "graphql", feature = "triple-store"))]
#[test]
fn explicit_graphql_entry_points_honor_graph_model() {
    let lpg = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let lpg_session = lpg.session();
    let rdf_query = lpg_session
        .execute_graphql_rdf("query { anything }")
        .unwrap_err();
    assert!(rdf_query.to_string().contains("LPG database"));
    let rdf_params = lpg_session
        .execute_graphql_rdf_with_params("query { anything }", std::collections::HashMap::new())
        .unwrap_err();
    assert!(rdf_params.to_string().contains("LPG database"));

    let rdf = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let lpg_query = rdf
        .session()
        .execute_graphql("query { anything }")
        .unwrap_err();
    assert!(lpg_query.to_string().contains("RDF database"));
}

// --- Config::validate() tests ---

#[test]
fn validate_rejects_zero_memory_limit() {
    let config = Config::in_memory().with_memory_limit(0);
    assert_eq!(config.validate(), Err(ConfigError::ZeroMemoryLimit));
}

#[test]
fn validate_rejects_zero_threads() {
    let config = Config::in_memory().with_threads(0);
    assert_eq!(config.validate(), Err(ConfigError::ZeroThreads));
}

#[test]
fn validate_rejects_zero_wal_flush_interval() {
    let mut config = Config::in_memory();
    config.wal_flush_interval_ms = 0;
    assert_eq!(config.validate(), Err(ConfigError::ZeroWalFlushInterval));
}

#[cfg(not(feature = "triple-store"))]
#[test]
fn validate_rejects_rdf_without_feature() {
    let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
    assert_eq!(config.validate(), Err(ConfigError::RdfFeatureRequired));
}

#[test]
fn with_config_rejects_invalid_config() {
    let config = Config::in_memory().with_threads(0);
    let result = GrafeoDB::with_config(config);
    assert!(result.is_err());
}

// --- Inspection API tests ---

#[test]
fn graph_model_accessor_returns_lpg() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(db.graph_model(), GraphModel::Lpg);
}

#[cfg(feature = "triple-store")]
#[test]
fn graph_model_accessor_returns_rdf() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    assert_eq!(db.graph_model(), GraphModel::Rdf);
}

#[test]
fn memory_limit_accessor_returns_none_by_default() {
    let db = GrafeoDB::new_in_memory();
    // Default in-memory config has no explicit memory limit
    // (it gets set during buffer manager init, not in config)
    assert!(db.memory_limit().is_none());
}

#[test]
fn memory_limit_accessor_returns_configured_value() {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_memory_limit(256 * 1024 * 1024)).unwrap();
    assert_eq!(db.memory_limit(), Some(256 * 1024 * 1024));
}

// --- DurabilityMode tests ---

#[test]
fn default_durability_is_strict_sync() {
    let config = Config::default();
    assert_eq!(config.wal_durability, DurabilityMode::Sync);
    assert_eq!(DurabilityMode::default(), DurabilityMode::Sync);
}

#[test]
fn config_with_sync_durability() {
    let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Sync);
    assert_eq!(config.wal_durability, DurabilityMode::Sync);
    assert!(config.validate().is_ok());
}

#[test]
fn config_with_nosync_durability() {
    let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::NoSync);
    assert_eq!(config.wal_durability, DurabilityMode::NoSync);
    assert!(config.validate().is_ok());
}

// --- schema_constraints tests ---

#[test]
fn schema_constraints_default_is_false() {
    let config = Config::default();
    assert!(!config.schema_constraints);
}

#[test]
fn schema_constraints_can_be_enabled() {
    let config = Config::in_memory().with_schema_constraints();
    assert!(config.schema_constraints);
}

// --- Session graph_model accessor ---

#[test]
fn session_reports_graph_model() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(session.graph_model(), GraphModel::Lpg);
}

#[cfg(feature = "triple-store")]
#[test]
fn session_reports_rdf_graph_model() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    assert_eq!(session.graph_model(), GraphModel::Rdf);
}

// --- SPARQL FILTER regression tests ---

/// Regression test: SPARQL FILTER equality must coerce types.
///
/// RDF stores all literal values as `Value::String`, but SPARQL FILTER
/// expressions parse numeric constants as `Value::Int64`.  Before the fix,
/// `Eq` used Rust's `PartialEq` (`==`) which never considered
/// `String("30") == Int64(30)`, causing FILTER(?age = 30) to return no rows.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_filter_equality_coerces_string_to_numeric() {
    use grafeo_common::types::Value;

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();

    // Insert triples: :alix :age "30" ; :gus :age "25"
    session
        .execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/age> "30" .
                <http://ex.org/gus>   <http://ex.org/age> "25" .
            }"#,
        )
        .unwrap();

    // FILTER(?age = 30) - the literal 30 is parsed as Int64, stored value is String "30"
    let result = session
        .execute_sparql(
            r#"SELECT ?s ?age WHERE {
                ?s <http://ex.org/age> ?age .
                FILTER(?age = 30)
            }"#,
        )
        .unwrap();

    assert_eq!(
        result.rows().len(),
        1,
        "FILTER(?age = 30) should match String '30' via type coercion"
    );

    // Verify the matched value
    let age = &result.rows()[0][1];
    assert!(
        matches!(age, Value::String(s) if s.as_str() == "30"),
        "Expected String '30', got {age:?}"
    );
}

/// Regression: FILTER inequality (!=) must also coerce types.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_filter_inequality_coerces_string_to_numeric() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();

    session
        .execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/age> "30" .
                <http://ex.org/gus>   <http://ex.org/age> "25" .
            }"#,
        )
        .unwrap();

    // FILTER(?age != 30) should return only gus (age "25")
    let result = session
        .execute_sparql(
            r#"SELECT ?s ?age WHERE {
                ?s <http://ex.org/age> ?age .
                FILTER(?age != 30)
            }"#,
        )
        .unwrap();

    assert_eq!(
        result.rows().len(),
        1,
        "FILTER(?age != 30) should exclude String '30'"
    );
}
