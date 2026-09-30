//! Exact mixed-type property histories across compact persistence boundaries.

#![cfg(all(feature = "compact-store", feature = "lpg"))]

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

#[derive(Clone, Copy)]
struct Fixture {
    node: NodeId,
    edge: EdgeId,
    created: EpochId,
    changed: EpochId,
}

fn build_fixture(db: &GrafeoDB) -> Fixture {
    let mut session = db.session();
    session.begin_transaction().expect("begin create");
    let node = session
        .create_node_with_props(
            &["Subject"],
            [
                ("mixed", Value::Int64(7)),
                ("embedding", Value::Vector(vec![1.0, 2.0].into())),
            ],
        )
        .expect("create subject");
    let peer = session.create_node(&["Peer"]);
    let edge = session
        .create_edge_with_props(
            node,
            peer,
            "CHANGES",
            [
                ("mixed", Value::Int64(9)),
                ("embedding", Value::Vector(vec![3.0, 4.0].into())),
            ],
        )
        .expect("create edge");
    session.commit().expect("commit create");
    let created = db.current_epoch();

    session.begin_transaction().expect("begin type change");
    session
        .set_node_property(node, "mixed", Value::from("seven"))
        .expect("change node property type");
    session
        .set_node_property(node, "embedding", Value::Vector(vec![5.0, 6.0, 7.0].into()))
        .expect("change node vector dimension");
    session
        .set_edge_property(edge, "mixed", Value::from("nine"))
        .expect("change edge property type");
    session
        .set_edge_property(
            edge,
            "embedding",
            Value::Vector(vec![8.0, 9.0, 10.0].into()),
        )
        .expect("change edge vector dimension");
    session.commit().expect("commit type change");
    let changed = db.current_epoch();
    drop(session);

    Fixture {
        node,
        edge,
        created,
        changed,
    }
}

fn assert_fixture(db: &GrafeoDB, fixture: Fixture, stage: &str) {
    let created_node = db
        .get_node_at_epoch(fixture.node, fixture.created)
        .unwrap_or_else(|| panic!("node missing at creation ({stage})"));
    assert_eq!(
        created_node.get_property("mixed"),
        Some(&Value::Int64(7)),
        "node Int64 value ({stage})"
    );
    assert_eq!(
        created_node.get_property("embedding"),
        Some(&Value::Vector(vec![1.0, 2.0].into())),
        "node two-dimensional vector ({stage})"
    );

    let changed_node = db
        .get_node_at_epoch(fixture.node, fixture.changed)
        .unwrap_or_else(|| panic!("node missing at change ({stage})"));
    assert_eq!(
        changed_node.get_property("mixed"),
        Some(&Value::from("seven")),
        "node String value ({stage})"
    );
    assert_eq!(
        changed_node.get_property("embedding"),
        Some(&Value::Vector(vec![5.0, 6.0, 7.0].into())),
        "node three-dimensional vector ({stage})"
    );

    let created_edge = db
        .get_edge_at_epoch(fixture.edge, fixture.created)
        .unwrap_or_else(|| panic!("edge missing at creation ({stage})"));
    assert_eq!(
        created_edge.get_property("mixed"),
        Some(&Value::Int64(9)),
        "edge Int64 value ({stage})"
    );
    assert_eq!(
        created_edge.get_property("embedding"),
        Some(&Value::Vector(vec![3.0, 4.0].into())),
        "edge two-dimensional vector ({stage})"
    );

    let changed_edge = db
        .get_edge_at_epoch(fixture.edge, fixture.changed)
        .unwrap_or_else(|| panic!("edge missing at change ({stage})"));
    assert_eq!(
        changed_edge.get_property("mixed"),
        Some(&Value::from("nine")),
        "edge String value ({stage})"
    );
    assert_eq!(
        changed_edge.get_property("embedding"),
        Some(&Value::Vector(vec![8.0, 9.0, 10.0].into())),
        "edge three-dimensional vector ({stage})"
    );
}

#[test]
fn mixed_type_and_vector_shape_histories_survive_compact_recompact_and_portable_snapshot() {
    let mut source = GrafeoDB::new_in_memory();
    let fixture = build_fixture(&source);
    assert_fixture(&source, fixture, "before compact");

    source.compact().expect("compact mixed histories");
    assert_fixture(&source, fixture, "after compact");

    source.compact().expect("recompact mixed histories");
    assert_fixture(&source, fixture, "after recompact");

    let bytes = source
        .export_snapshot()
        .expect("export portable snapshot with mixed histories");
    let mut restored =
        GrafeoDB::import_snapshot(&bytes).expect("import portable snapshot with mixed histories");
    assert_fixture(&restored, fixture, "after portable import");

    restored.compact().expect("compact imported histories");
    assert_fixture(&restored, fixture, "after import and compact");
    restored
        .compact()
        .expect("recompact imported mixed histories");
    assert_fixture(&restored, fixture, "after import and recompact");
}
