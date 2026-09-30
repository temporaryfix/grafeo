//! RDF→LPG projection target ownership and mutation-integrity regressions.

#![cfg(all(feature = "triple-store", feature = "sparql", feature = "lpg"))]

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::rdf::{RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PERSON: &str = "http://ex.org/Person";
const ALIX: &str = "http://ex.org/alix";

fn projected_db() -> (GrafeoDB, u64, NodeId, String) {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("create dual-model database");
    db.execute_sparql(&format!("INSERT DATA {{ <{ALIX}> a <{PERSON}> . }}"))
        .expect("insert source triple");
    let projection_id = db
        .declare_rdf_lpg_projection(PERSON, "Person")
        .expect("declare projection");
    db.rebuild_rdf_lpg_projection(projection_id)
        .expect("initial rebuild");
    let definition = db
        .rdf_lpg_projection(projection_id)
        .expect("projection definition");
    let node_id = grafeo_engine::database::testing::root_lpg_store(&db)
        .nodes_by_label("Person")
        .into_iter()
        .next()
        .expect("materialized node");
    (db, projection_id, node_id, definition.owner_marker())
}

fn assert_owned_row_unchanged(db: &GrafeoDB, node_id: NodeId, owner_marker: &str) {
    let node = db
        .session()
        .get_node(node_id)
        .expect("projection-owned row must remain visible");
    assert!(node.labels.iter().any(|label| label.as_str() == "Person"));
    assert_eq!(node.labels.len(), 1, "rejected label writes must roll back");
    assert_eq!(
        node.properties
            .get(&PropertyKey::new(RDF_LPG_PROJECTION_IRI_PROPERTY))
            .and_then(Value::as_str),
        Some(ALIX)
    );
    assert_eq!(
        node.properties
            .get(&PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY))
            .and_then(Value::as_str),
        Some(owner_marker)
    );
    assert_eq!(
        node.properties.len(),
        2,
        "rejected property writes must not leave residue"
    );
}

#[test]
fn public_queries_cannot_forge_or_mutate_projection_ownership() {
    let (db, _projection_id, node_id, owner_marker) = projected_db();
    let before_count = db.node_count();

    let forge = format!(
        "CREATE (:Fake {{{RDF_LPG_PROJECTION_OWNER_PROPERTY}: '{owner_marker}', iri: 'http://ex.org/forged'}})"
    );
    let error = db.session().execute(&forge).expect_err("forgery must fail");
    assert!(
        error.to_string().contains("reserved '__grafeo' namespace"),
        "unexpected forgery error: {error}"
    );
    assert_eq!(
        db.node_count(),
        before_count,
        "failed CREATE must roll back"
    );

    let mutations = [
        format!("MATCH (n) WHERE id(n) = {} SET n.iri = 'forged'", node_id.0),
        format!("MATCH (n) WHERE id(n) = {} SET n.extra = 1", node_id.0),
        format!(
            "MATCH (n) WHERE id(n) = {} SET n.{RDF_LPG_PROJECTION_OWNER_PROPERTY} = 'forged'",
            node_id.0
        ),
        format!("MATCH (n) WHERE id(n) = {} REMOVE n.iri", node_id.0),
        format!("MATCH (n) WHERE id(n) = {} SET n:Forged", node_id.0),
        format!("MATCH (n) WHERE id(n) = {} REMOVE n:Person", node_id.0),
        format!("MATCH (n) WHERE id(n) = {} DETACH DELETE n", node_id.0),
    ];
    for mutation in mutations {
        let error = db
            .session()
            .execute(&mutation)
            .expect_err("owned-row query mutation must fail");
        assert!(
            error.to_string().contains("projection") || error.to_string().contains("reserved"),
            "unexpected protected-row error for {mutation:?}: {error}"
        );
        assert_owned_row_unchanged(&db, node_id, &owner_marker);
    }
}

#[test]
fn direct_session_apis_cannot_forge_or_mutate_projection_ownership() {
    let (db, _projection_id, node_id, owner_marker) = projected_db();
    let before_count = db.node_count();
    let session = db.session();

    let forged = session.create_node_with_props(
        &["Fake"],
        [
            (
                RDF_LPG_PROJECTION_OWNER_PROPERTY,
                Value::from(owner_marker.clone()),
            ),
            (
                RDF_LPG_PROJECTION_IRI_PROPERTY,
                Value::from("http://ex.org/forged"),
            ),
        ],
    );
    assert!(forged.is_err(), "direct ownership forgery must fail");
    assert_eq!(db.node_count(), before_count);

    assert!(
        session
            .set_node_property(node_id, "extra", Value::Int64(1))
            .is_err(),
        "direct property mutation must fail"
    );
    assert!(
        session
            .set_node_property(
                node_id,
                RDF_LPG_PROJECTION_OWNER_PROPERTY,
                Value::from("forged"),
            )
            .is_err(),
        "direct owner-marker mutation must fail"
    );
    assert!(!session.remove_node_property(node_id, RDF_LPG_PROJECTION_IRI_PROPERTY));
    assert!(!session.add_node_label(node_id, "Forged"));
    assert!(!session.remove_node_label(node_id, "Person"));
    assert!(!session.delete_node(node_id));
    assert_owned_row_unchanged(&db, node_id, &owner_marker);

    let user_node = session
        .create_node_with_props(&["User"], [("name", Value::from("plain"))])
        .expect("ordinary direct create");
    assert!(
        session
            .set_node_property(user_node, "__grafeo_future_internal", Value::from("forged"))
            .is_err(),
        "the full reserved namespace must be protected"
    );
    assert!(
        session
            .get_node(user_node)
            .expect("ordinary node remains")
            .get_property("__grafeo_future_internal")
            .is_none()
    );
}

#[test]
fn rebuild_canonicalizes_duplicate_and_malformed_rows_before_success() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("create dual-model database");
    db.execute_sparql(&format!("INSERT DATA {{ <{ALIX}> a <{PERSON}> . }}"))
        .expect("insert source triple");
    let projection_id = db
        .declare_rdf_lpg_projection(PERSON, "Person")
        .expect("declare projection");
    let owner_marker = db
        .rdf_lpg_projection(projection_id)
        .expect("definition")
        .owner_marker();

    // The raw store is explicitly documented as uncoordinated. Use it here to
    // model legacy/corrupt target state that the public Session API now refuses
    // to manufacture: two owned rows for one desired IRI, with the retained
    // first row missing its required target label and carrying an extra
    // property, while the duplicate has an extra label.
    for (index, label) in ["Wrong", "Person"].into_iter().enumerate() {
        let node = grafeo_engine::database::testing::root_lpg_store(&db).create_node(&[label]);
        grafeo_engine::database::testing::root_lpg_store(&db).set_node_property(
            node,
            RDF_LPG_PROJECTION_OWNER_PROPERTY,
            Value::from(owner_marker.clone()),
        );
        grafeo_engine::database::testing::root_lpg_store(&db).set_node_property(
            node,
            RDF_LPG_PROJECTION_IRI_PROPERTY,
            Value::from(ALIX),
        );
        if index == 0 {
            grafeo_engine::database::testing::root_lpg_store(&db).set_node_property(
                node,
                "extra",
                Value::Int64(1),
            );
        } else {
            grafeo_engine::database::testing::root_lpg_store(&db).add_label(node, "Forged");
        }
    }

    assert_eq!(db.rebuild_rdf_lpg_projection(projection_id).unwrap(), 1);
    let owned: Vec<_> = grafeo_engine::database::testing::root_lpg_store(&db)
        .node_ids()
        .into_iter()
        .filter_map(|id| db.session().get_node(id))
        .filter(|node| {
            node.properties
                .get(&PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY))
                .and_then(Value::as_str)
                == Some(owner_marker.as_str())
        })
        .collect();
    assert_eq!(owned.len(), 1, "duplicate owned row must be deleted");
    assert_eq!(owned[0].labels.as_slice(), ["Person"]);
    assert_eq!(owned[0].properties.len(), 2);
    assert_eq!(
        owned[0]
            .properties
            .get(&PropertyKey::new(RDF_LPG_PROJECTION_IRI_PROPERTY))
            .and_then(Value::as_str),
        Some(ALIX)
    );
}
