//! Persisted graph-model declarations must fail closed in feature-limited builds.

#[cfg(all(
    feature = "grafeo-file",
    not(all(feature = "lpg", feature = "triple-store"))
))]
use grafeo_common::utils::error::{Error, ErrorCode, StorageError};
#[cfg(all(
    feature = "grafeo-file",
    not(all(feature = "lpg", feature = "triple-store"))
))]
use grafeo_engine::Config;
#[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
use grafeo_engine::GraphModel;

#[cfg(all(
    feature = "grafeo-file",
    not(all(feature = "lpg", feature = "triple-store"))
))]
fn open_fixture(bytes: &[u8]) -> Error {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.grafeo");
    std::fs::write(&path, bytes).unwrap();
    let config = Config::read_only(&path);
    // Keep the request unpinned while selecting a model this binary can
    // represent. The persisted tag must then be adopted and revalidated.
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    let config = {
        let mut config = config;
        config.graph_model = GraphModel::Rdf;
        config
    };
    let error = grafeo_engine::GrafeoDB::with_config(config)
        .err()
        .expect("feature-limited open must reject an unsupported authoritative model");
    assert!(
        error
            .to_string()
            .contains("container declares unsupported graph model"),
        "unrelated corruption must not pass model containment: {error}"
    );
    error
}

#[cfg(all(
    feature = "grafeo-file",
    any(feature = "lpg", feature = "triple-store"),
    not(all(feature = "lpg", feature = "triple-store"))
))]
#[test]
fn feature_limited_open_accepts_its_current_model_fixture() -> Result<(), Box<dyn std::error::Error>>
{
    #[cfg(feature = "lpg")]
    let (bytes, model) = (
        include_bytes!("fixtures/graph_model_lpg.grafeo").as_slice(),
        grafeo_engine::GraphModel::Lpg,
    );
    #[cfg(feature = "triple-store")]
    let (bytes, model) = (
        include_bytes!("fixtures/graph_model_rdf.grafeo").as_slice(),
        grafeo_engine::GraphModel::Rdf,
    );
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("supported.grafeo");
    std::fs::write(&path, bytes)?;
    let db =
        grafeo_engine::GrafeoDB::with_config(Config::read_only(&path).with_graph_model(model))?;
    assert_eq!(db.graph_model(), model);
    drop(db);
    assert_eq!(std::fs::read(path)?, bytes);
    Ok(())
}

#[cfg(all(feature = "grafeo-file", not(feature = "lpg")))]
#[test]
fn current_lpg_container_is_rejected_without_lpg_support() {
    let error = open_fixture(include_bytes!("fixtures/graph_model_lpg.grafeo"));
    assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
    assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
}

#[cfg(all(feature = "grafeo-file", not(feature = "triple-store")))]
#[test]
fn current_rdf_container_is_rejected_without_rdf_support() {
    let error = open_fixture(include_bytes!("fixtures/graph_model_rdf.grafeo"));
    assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
    assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
}

#[cfg(all(
    feature = "grafeo-file",
    not(all(feature = "lpg", feature = "triple-store"))
))]
#[test]
fn current_both_container_is_rejected_without_both_model_planes() {
    let error = open_fixture(include_bytes!("fixtures/graph_model_both.grafeo"));
    assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
    assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
}

#[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
#[test]
fn rdf_only_new_in_memory_selects_the_available_authoritative_model() {
    let db = grafeo_engine::GrafeoDB::new_in_memory();
    assert_eq!(db.graph_model(), GraphModel::Rdf);
}

#[cfg(all(feature = "lpg", feature = "triple-store"))]
#[test]
fn fallible_rdf_membership_rejects_an_lpg_only_database() {
    use grafeo_core::graph::rdf::{Quad, Term, Triple};
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))
        .expect("construct LPG-only database in a dual-model binary");
    let quad = Quad::new(Triple::new(
        Term::iri("urn:grafeo:test:subject"),
        Term::iri("urn:grafeo:test:predicate"),
        Term::literal("object"),
    ));

    assert!(
        db.try_contains_rdf_quad(&quad).is_err(),
        "fallible membership must preserve model-containment errors"
    );
    assert!(
        !db.contains_rdf_quad(&quad),
        "the infallible compatibility shim may only fail closed"
    );
}
