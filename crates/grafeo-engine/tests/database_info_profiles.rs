//! Public inspection must describe the model available in each build profile.

#![cfg(any(feature = "lpg", feature = "triple-store"))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

fn assert_identity(db: &GrafeoDB, path: Option<&std::path::Path>) {
    let info = db.info();
    assert_eq!(db.path(), path);
    assert_eq!(db.is_persistent(), path.is_some());
    assert_eq!(info.path.as_deref(), path);
    assert_eq!(info.is_persistent, path.is_some());
    assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
    for (name, compiled) in [
        ("lpg", cfg!(feature = "lpg")),
        ("rdf", cfg!(feature = "triple-store")),
        ("gql", cfg!(feature = "gql")),
        ("sparql", cfg!(feature = "sparql")),
    ] {
        assert_eq!(
            info.features.iter().any(|feature| feature == name),
            compiled
        );
    }
}

#[cfg(feature = "triple-store")]
fn populate_rdf(db: &GrafeoDB) {
    use grafeo_core::graph::rdf::{Quad, Term, Triple};

    let triple =
        |subject, value| Triple::new(Term::iri(subject), Term::iri("urn:p"), Term::literal(value));
    db.insert_rdf_quads([
        Quad::new(triple("urn:a", "default-a")),
        Quad::new(triple("urn:b", "default-b")),
        Quad::named(triple("urn:a", "named-a"), "urn:g"),
        Quad::named(triple("urn:c", "named-c"), "urn:g"),
    ])
    .unwrap();
}

#[cfg(feature = "triple-store")]
#[test]
fn rdf_info_counts_distinct_subjects_across_native_graphs() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    populate_rdf(&db);
    let info = db.info();
    assert_eq!(info.mode, grafeo_engine::admin::DatabaseMode::Rdf);
    assert_eq!(info.node_count, 3);
    assert_eq!(info.edge_count, 4);
    assert_identity(&db, None);
}

#[cfg(all(feature = "triple-store", feature = "wal", feature = "grafeo-file"))]
#[test]
fn rdf_info_preserves_identity_and_counts_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rdf-info.grafeo");
    let db =
        GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Rdf)).unwrap();
    populate_rdf(&db);
    assert_identity(&db, Some(&path));
    assert!(db.info().wal_enabled);
    db.close().unwrap();
    let reopened = GrafeoDB::open_read_only(&path).unwrap();
    assert_identity(&reopened, Some(&path));
    let info = reopened.info();
    assert_eq!(info.mode, grafeo_engine::admin::DatabaseMode::Rdf);
    assert_eq!((info.node_count, info.edge_count), (3, 4));
}

#[cfg(feature = "lpg")]
#[test]
fn lpg_info_retains_native_node_and_edge_counts() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let a = db.create_node(&["Node"]);
    let b = db.create_node(&["Node"]);
    db.create_edge(a, b, "LINK");
    let info = db.info();
    assert_eq!(info.mode, grafeo_engine::admin::DatabaseMode::Lpg);
    assert_eq!((info.node_count, info.edge_count), (2, 1));
    assert_identity(&db, None);
}

#[cfg(all(feature = "lpg", feature = "triple-store"))]
#[test]
fn both_info_retains_its_lpg_count_contract() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
    db.create_node(&["Node"]);
    populate_rdf(&db);
    let info = db.info();
    assert_eq!(info.mode, grafeo_engine::admin::DatabaseMode::Both);
    assert_eq!((info.node_count, info.edge_count), (1, 0));
    assert_identity(&db, None);
}
