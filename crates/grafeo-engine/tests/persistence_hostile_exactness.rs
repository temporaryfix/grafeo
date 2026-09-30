//! Hostile persistence checks for state that descriptor-only recovery cannot
//! silently approximate.
//!
//! These downstream-style tests construct projection provenance and temporal
//! histories through public APIs; a rejected save must leave no destination.
//! Exotic index settings absent from public DDL have owning unit fixtures in
//! `database/save_exactness_tests.rs`, with matching catalog owners.

#![cfg(all(feature = "lpg", feature = "wal"))]

use grafeo_common::types::{EdgeId, EpochId, Value};
#[cfg(feature = "triple-store")]
use grafeo_engine::Config;
use grafeo_engine::GrafeoDB;

#[cfg(feature = "triple-store")]
fn assert_save_rejected_without_destination(db: &GrafeoDB, destination: &std::path::Path) {
    assert!(
        !destination.exists(),
        "test precondition: destination must not already exist"
    );
    let error = db
        .save(destination)
        .expect_err("lossy persistence must fail closed");
    assert!(
        !destination.exists(),
        "rejected save published destination {}: {error}",
        destination.display()
    );
}

#[cfg(feature = "triple-store")]
#[test]
fn exact_save_rejects_forged_projection_provenance_without_destination() {
    use grafeo_common::types::PropertyKey;
    use grafeo_core::graph::rdf::{
        RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY, Term, Triple,
    };
    use grafeo_engine::GraphModel;

    const PERSON: &str = "http://ex.org/Person";
    const ALIX: &str = "http://ex.org/alix";
    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("create dual-model database");

    // This row is ordinary user-authored LPG state. Its structural lifetime
    // deliberately predates any projection declaration or ownership marker.
    let ordinary = db.create_node(&["Person"]);
    let ordinary_created = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_node_history(ordinary)
        .first()
        .map(|(created, _, _)| *created)
        .expect("ordinary node lifetime");

    assert_eq!(
        db.batch_insert_rdf([Triple::new(
            Term::iri(ALIX),
            Term::iri(RDF_TYPE),
            Term::iri(PERSON),
        )])
        .expect("insert RDF projection source"),
        1
    );
    let projection_id = db
        .declare_rdf_lpg_projection(PERSON, "Person")
        .expect("declare projection");
    assert_eq!(
        db.rebuild_rdf_lpg_projection(projection_id)
            .expect("publish canonical generation"),
        1
    );
    let definition = db
        .rdf_lpg_projection(projection_id)
        .expect("published projection definition");
    let owner_marker = definition.owner_marker();
    assert_eq!(definition.row_count(), 1);

    let owner_key = PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY);
    let generated = grafeo_engine::database::testing::root_lpg_store(&db)
        .node_ids()
        .into_iter()
        .filter(|id| *id != ordinary)
        .find(|id| {
            grafeo_engine::database::testing::root_lpg_store(&db)
                .get_node(*id)
                .and_then(|node| node.properties.get(&owner_key).cloned())
                .and_then(|value| value.as_str().map(ToOwned::to_owned))
                .as_deref()
                == Some(owner_marker.as_str())
        })
        .expect("engine-generated projection row");

    // Advance beyond both lifetimes, then use the explicitly documented raw
    // in-memory store to model corrupt/legacy provenance that Session rejects:
    // retire the genuine row and graft its canonical current marker plane onto
    // the pre-existing ordinary row.
    let corruption_epoch = db.current_epoch().as_u64() + 1;
    advance_to_epoch(&db, corruption_epoch);
    assert!(grafeo_engine::database::testing::root_lpg_store(&db).delete_node(generated));
    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property(
        ordinary,
        RDF_LPG_PROJECTION_OWNER_PROPERTY,
        Value::from(owner_marker.clone()),
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property(
        ordinary,
        RDF_LPG_PROJECTION_IRI_PROPERTY,
        Value::from(ALIX),
    );

    let forged = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_node(ordinary)
        .expect("forged current row");
    assert_eq!(forged.labels.as_slice(), ["Person"]);
    assert_eq!(forged.properties.len(), 2);
    assert_eq!(
        forged.properties.get(&owner_key).and_then(Value::as_str),
        Some(owner_marker.as_str())
    );
    assert_eq!(
        forged
            .properties
            .get(&PropertyKey::new(RDF_LPG_PROJECTION_IRI_PROPERTY))
            .and_then(Value::as_str),
        Some(ALIX)
    );
    let owner_history = grafeo_engine::database::testing::root_lpg_store(&db)
        .node_property_history_for_key(ordinary, RDF_LPG_PROJECTION_OWNER_PROPERTY);
    assert_eq!(owner_history.len(), 1);
    assert!(
        owner_history[0].0 > ordinary_created,
        "ownership must have been grafted onto an older ordinary lifetime"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_node(generated)
            .is_none(),
        "the genuine generated row must be retired"
    );

    let temp = tempfile::tempdir().expect("create temporary directory");
    let destination = temp.path().join("forged-projection-provenance");
    assert_save_rejected_without_destination(&db, &destination);

    let export_error = db
        .export_snapshot()
        .expect_err("portable export must reject forged projection provenance");
    assert!(
        export_error
            .to_string()
            .contains("non-canonical projection provenance"),
        "unexpected portable export error: {export_error}"
    );
    let portable_destination = temp.path().join("forged-projection-provenance.grafeo");
    assert_save_rejected_without_destination(&db, &portable_destination);
}

fn advance_to_epoch(db: &GrafeoDB, target: u64) {
    while db.current_epoch().as_u64() < target {
        let mut session = db.session();
        session.begin_transaction().expect("begin epoch advance");
        session.commit().expect("commit epoch advance");
    }
}

fn edge_property_history_for_key(
    db: &GrafeoDB,
    edge: EdgeId,
    property: &str,
) -> Vec<(EpochId, Value)> {
    grafeo_engine::database::testing::root_lpg_store(db)
        .edge_property_history(edge)
        .into_iter()
        .find_map(|(key, history)| (key.as_str() == property).then_some(history))
        .unwrap_or_default()
}

#[test]
fn exact_save_preserves_raw_null_property_version_logs_exactly() {
    let source = GrafeoDB::new_in_memory();
    let subject = source.create_node(&["Subject"]);
    let peer = source.create_node(&["Peer"]);
    let edge = source.create_edge(subject, peer, "LINKS");
    assert!(subject.is_valid() && peer.is_valid() && edge.is_valid());

    let first_epoch = source.current_epoch().as_u64() + 1;
    let leading_null = EpochId::new(first_epoch);
    let live_value = EpochId::new(first_epoch + 1);
    let repeated_null = EpochId::new(first_epoch + 2);
    advance_to_epoch(&source, repeated_null.as_u64());

    let expected_node_history = vec![
        (leading_null, Value::Null),
        (live_value, Value::from("present")),
        (repeated_null, Value::Null),
        (repeated_null, Value::Null),
    ];
    for (epoch, value) in &expected_node_history {
        grafeo_engine::database::testing::root_lpg_store(&source).set_node_property_at_epoch(
            subject,
            "state",
            value.clone(),
            *epoch,
        );
    }

    let expected_edge_history = vec![
        (leading_null, Value::Null),
        (live_value, Value::Int64(17)),
        (repeated_null, Value::Null),
        (repeated_null, Value::Null),
    ];
    for (epoch, value) in &expected_edge_history {
        grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
            edge,
            "weight",
            value.clone(),
            *epoch,
        );
    }

    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&source)
            .node_property_history_for_key(subject, "state"),
        expected_node_history,
        "fixture must retain leading, transition, and duplicate Null entries"
    );
    assert_eq!(
        edge_property_history_for_key(&source, edge, "weight"),
        expected_edge_history,
        "fixture must retain the raw edge VersionLog"
    );

    let temp = tempfile::tempdir().expect("create temporary directory");
    let destination = temp.path().join("raw-null-version-logs");
    source
        .save(&destination)
        .expect("copy exact histories to container");

    let restored = GrafeoDB::open(&destination).expect("reopen exact container copy");
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&restored)
            .node_property_history_for_key(subject, "state"),
        expected_node_history,
        "node VersionLog changed during exact save/reopen"
    );
    assert_eq!(
        edge_property_history_for_key(&restored, edge, "weight"),
        expected_edge_history,
        "edge VersionLog changed during exact save/reopen"
    );
    assert_eq!(
        restored.export_snapshot().unwrap(),
        source.export_snapshot().unwrap()
    );
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().expect("close restored database");
}
