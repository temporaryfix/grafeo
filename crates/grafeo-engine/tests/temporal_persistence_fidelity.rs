//! Persistence must preserve the LPG timeline, not merely its current image.
//!
//! These tests deliberately finish with a deleted node and edge. A persistence
//! implementation that enumerates only current IDs, recreates everything at
//! epoch zero, or serializes only current labels/properties cannot pass.

#![cfg(feature = "lpg")]

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_engine::GrafeoDB;

#[derive(Debug, Clone, Copy)]
struct Timeline {
    subject: NodeId,
    peer: NodeId,
    edge: EdgeId,
    before_create: EpochId,
    created: EpochId,
    updated: EpochId,
    label_removed: EpochId,
    edge_deleted: EpochId,
    node_deleted: EpochId,
}

fn build_closed_timeline(db: &GrafeoDB) -> Timeline {
    build_closed_timeline_in_graph(db, None)
}

fn graph_epoch(db: &GrafeoDB, graph: Option<&str>) -> EpochId {
    graph.map_or_else(
        || db.current_epoch(),
        |name| {
            grafeo_engine::database::testing::root_lpg_store(db)
                .graph(name)
                .unwrap_or_else(|| panic!("named graph {name:?} must exist"))
                .current_epoch()
        },
    )
}

fn build_closed_timeline_in_graph(db: &GrafeoDB, graph: Option<&str>) -> Timeline {
    let before_create = graph_epoch(db, graph);
    let mut session = db.session();
    if let Some(graph) = graph {
        session
            .use_graph_path(
                &grafeo_common::types::GraphPath::from_components(&[graph])
                    .expect("literal graph path"),
            )
            .expect("select existing graph");
    }

    session
        .begin_transaction()
        .expect("begin create transaction");
    let subject = session
        .create_node_with_props(
            &["Entity", "Transient"],
            [("state", Value::String("born".into()))],
        )
        .expect("create subject");
    let peer = session
        .create_node_with_props(
            &["Anchor"],
            [("name", Value::String("persistent-peer".into()))],
        )
        .expect("create peer");
    let edge = session
        .create_edge_with_props(subject, peer, "LINKS", [("weight", Value::Int64(1))])
        .expect("create edge");
    session.commit().expect("commit creation");
    let created = graph_epoch(db, graph);

    session
        .begin_transaction()
        .expect("begin update transaction");
    session
        .set_node_property(subject, "state", Value::String("updated".into()))
        .expect("update node property");
    session
        .set_edge_property(edge, "weight", Value::Int64(2))
        .expect("update edge property");
    assert!(
        session.add_node_label(subject, "Tracked"),
        "the update transaction must add Tracked"
    );
    session.commit().expect("commit update");
    let updated = graph_epoch(db, graph);

    session
        .begin_transaction()
        .expect("begin label removal transaction");
    assert!(
        session.remove_node_label(subject, "Transient"),
        "the label transaction must remove Transient"
    );
    session.commit().expect("commit label removal");
    let label_removed = graph_epoch(db, graph);

    session
        .begin_transaction()
        .expect("begin edge deletion transaction");
    assert!(session.delete_edge(edge), "the edge must be deleted");
    session.commit().expect("commit edge deletion");
    let edge_deleted = graph_epoch(db, graph);

    session
        .begin_transaction()
        .expect("begin node deletion transaction");
    assert!(session.delete_node(subject), "the subject must be deleted");
    session.commit().expect("commit node deletion");
    let node_deleted = graph_epoch(db, graph);

    drop(session);

    Timeline {
        subject,
        peer,
        edge,
        before_create,
        created,
        updated,
        label_removed,
        edge_deleted,
        node_deleted,
    }
}

fn assert_node_state(
    db: &GrafeoDB,
    timeline: Timeline,
    epoch: EpochId,
    expected_state: &str,
    expected_labels: &[&str],
    absent_labels: &[&str],
) {
    let node = db
        .get_node_at_epoch(timeline.subject, epoch)
        .unwrap_or_else(|| panic!("subject must be visible at epoch {epoch:?}"));

    assert_eq!(
        node.get_property("state").cloned(),
        Some(Value::String(expected_state.into())),
        "node property mismatch at epoch {epoch:?}"
    );
    assert_eq!(
        db.get_node_property_at_epoch(timeline.subject, "state", epoch),
        Some(Value::String(expected_state.into())),
        "single-property temporal read mismatch at epoch {epoch:?}"
    );
    for label in expected_labels {
        assert!(
            node.has_label(label),
            "label {label:?} must be present at epoch {epoch:?}"
        );
    }
    for label in absent_labels {
        assert!(
            !node.has_label(label),
            "label {label:?} must be absent at epoch {epoch:?}"
        );
    }
}

fn assert_edge_weight(db: &GrafeoDB, timeline: Timeline, epoch: EpochId, weight: i64) {
    let edge = db
        .get_edge_at_epoch(timeline.edge, epoch)
        .unwrap_or_else(|| panic!("edge must be visible at epoch {epoch:?}"));
    assert_eq!(edge.src, timeline.subject);
    assert_eq!(edge.dst, timeline.peer);
    assert_eq!(edge.edge_type.as_str(), "LINKS");
    assert_eq!(
        edge.get_property("weight").cloned(),
        Some(Value::Int64(weight)),
        "edge property mismatch at epoch {epoch:?}"
    );
}

fn assert_store_node_state(
    store: &LpgStore,
    timeline: Timeline,
    epoch: EpochId,
    expected_state: &str,
    expected_labels: &[&str],
    absent_labels: &[&str],
) {
    let node = store
        .get_node_at_epoch(timeline.subject, epoch)
        .unwrap_or_else(|| panic!("named-graph subject must be visible at epoch {epoch:?}"));
    assert_eq!(
        node.get_property("state").cloned(),
        Some(Value::String(expected_state.into()))
    );
    assert_eq!(
        store.get_node_property_at_epoch(timeline.subject, &PropertyKey::new("state"), epoch,),
        Some(Value::String(expected_state.into()))
    );
    for label in expected_labels {
        assert!(node.has_label(label), "missing {label:?} at {epoch:?}");
    }
    for label in absent_labels {
        assert!(!node.has_label(label), "unexpected {label:?} at {epoch:?}");
    }
}

fn assert_named_graph_timeline(db: &GrafeoDB, graph_name: &str, timeline: Timeline) {
    let store = grafeo_engine::database::testing::root_lpg_store(db)
        .graph(graph_name)
        .unwrap_or_else(|| panic!("named graph {graph_name:?} must exist"));

    assert!(store.get_node(timeline.subject).is_none());
    assert!(store.get_edge(timeline.edge).is_none());
    assert!(store.get_node(timeline.peer).is_some());
    assert!(
        store
            .get_node_at_epoch(timeline.subject, timeline.before_create)
            .is_none()
    );
    assert!(
        store
            .get_edge_at_epoch(timeline.edge, timeline.before_create)
            .is_none()
    );

    assert_store_node_state(
        &store,
        timeline,
        timeline.created,
        "born",
        &["Entity", "Transient"],
        &["Tracked"],
    );
    assert_eq!(
        store
            .get_edge_at_epoch(timeline.edge, timeline.created)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(1))
    );

    assert_store_node_state(
        &store,
        timeline,
        timeline.updated,
        "updated",
        &["Entity", "Transient", "Tracked"],
        &[],
    );
    assert_eq!(
        store
            .get_edge_at_epoch(timeline.edge, timeline.updated)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(2))
    );

    assert_store_node_state(
        &store,
        timeline,
        timeline.label_removed,
        "updated",
        &["Entity", "Tracked"],
        &["Transient"],
    );
    assert!(
        store
            .get_edge_at_epoch(timeline.edge, timeline.edge_deleted)
            .is_none()
    );
    assert!(
        store
            .get_node_at_epoch(timeline.subject, timeline.node_deleted)
            .is_none()
    );
}

fn assert_timeline(db: &GrafeoDB, timeline: Timeline) {
    assert!(
        db.get_node(timeline.subject).is_none(),
        "closed subject must be absent from the current graph"
    );
    assert!(
        db.get_edge(timeline.edge).is_none(),
        "closed edge must be absent from the current graph"
    );
    assert!(
        db.get_node(timeline.peer).is_some(),
        "the undeleted endpoint must remain current"
    );

    assert!(
        db.get_node_at_epoch(timeline.subject, timeline.before_create)
            .is_none(),
        "subject must not exist before its creation epoch"
    );
    assert!(
        db.get_edge_at_epoch(timeline.edge, timeline.before_create)
            .is_none(),
        "edge must not exist before its creation epoch"
    );

    assert_node_state(
        db,
        timeline,
        timeline.created,
        "born",
        &["Entity", "Transient"],
        &["Tracked"],
    );
    assert_edge_weight(db, timeline, timeline.created, 1);

    assert_node_state(
        db,
        timeline,
        timeline.updated,
        "updated",
        &["Entity", "Transient", "Tracked"],
        &[],
    );
    assert_edge_weight(db, timeline, timeline.updated, 2);

    assert_node_state(
        db,
        timeline,
        timeline.label_removed,
        "updated",
        &["Entity", "Tracked"],
        &["Transient"],
    );
    assert_edge_weight(db, timeline, timeline.label_removed, 2);

    assert_node_state(
        db,
        timeline,
        timeline.edge_deleted,
        "updated",
        &["Entity", "Tracked"],
        &["Transient"],
    );
    assert!(
        db.get_edge_at_epoch(timeline.edge, timeline.edge_deleted)
            .is_none(),
        "edge must be absent at its deletion epoch"
    );
    assert!(
        db.get_node_at_epoch(timeline.subject, timeline.node_deleted)
            .is_none(),
        "subject must be absent at its deletion epoch"
    );
}

#[derive(Debug, Clone, Copy)]
struct BoundaryDeletion {
    subject: NodeId,
    peer: NodeId,
    edge: EdgeId,
    created: EpochId,
    deleted: EpochId,
}

fn build_same_epoch_edge_node_deletion(
    db: &mut GrafeoDB,
    between_transactions: impl FnOnce(&mut GrafeoDB),
) -> BoundaryDeletion {
    let mut session = db.session();
    session.begin_transaction().expect("begin boundary fixture");
    let subject = session
        .create_node_with_props(
            &["Victim", "Old"],
            [("status", Value::String("alive".into()))],
        )
        .expect("create boundary subject");
    let peer = session.create_node(&["Peer"]);
    let edge = session
        .create_edge_with_props(subject, peer, "BOUND_TO", [("weight", Value::Int64(7))])
        .expect("create boundary edge");
    session.commit().expect("commit boundary fixture");
    let created = db.current_epoch();
    drop(session);
    between_transactions(db);

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin same-epoch deletion");
    assert!(session.add_node_label(subject, "Ephemeral"));
    assert!(session.remove_node_label(subject, "Old"));
    assert!(
        session.delete_edge(edge),
        "incident edge must be staged before its endpoint deletion"
    );
    assert!(session.delete_node(subject), "endpoint deletion must stage");
    session.commit().expect("commit same-epoch deletion");
    let deleted = db.current_epoch();
    drop(session);

    BoundaryDeletion {
        subject,
        peer,
        edge,
        created,
        deleted,
    }
}

fn string_label_history(db: &GrafeoDB, id: NodeId) -> Vec<(EpochId, Vec<String>)> {
    grafeo_engine::database::testing::root_lpg_store(db)
        .node_label_history(id)
        .into_iter()
        .map(|(epoch, labels)| {
            (
                epoch,
                labels.into_iter().map(|label| label.to_string()).collect(),
            )
        })
        .collect()
}

fn assert_same_epoch_edge_node_deletion(db: &GrafeoDB, fixture: BoundaryDeletion) {
    let at_create = db
        .get_node_at_epoch(fixture.subject, fixture.created)
        .expect("subject must exist at creation");
    assert!(at_create.has_label("Victim"));
    assert!(at_create.has_label("Old"));
    assert!(!at_create.has_label("Ephemeral"));
    assert_eq!(
        at_create.get_property("status"),
        Some(&Value::String("alive".into()))
    );
    assert_eq!(
        db.get_edge_at_epoch(fixture.edge, fixture.created)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(7))
    );

    assert!(
        db.get_node_at_epoch(fixture.subject, fixture.deleted)
            .is_none()
    );
    assert!(
        db.get_edge_at_epoch(fixture.edge, fixture.deleted)
            .is_none()
    );
    assert!(db.get_node(fixture.subject).is_none());
    assert!(db.get_edge(fixture.edge).is_none());
    assert!(db.get_node(fixture.peer).is_some());
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.edge_count(), 0);
    for label in ["Victim", "Old", "Ephemeral"] {
        assert!(
            grafeo_engine::database::testing::root_lpg_store(db)
                .nodes_by_label(label)
                .is_empty()
        );
    }

    let node_lifetimes: Vec<_> = db
        .get_node_history(fixture.subject)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect();
    assert_eq!(
        node_lifetimes,
        vec![(fixture.created, Some(fixture.deleted))]
    );
    let edge_lifetimes: Vec<_> = db
        .get_edge_history(fixture.edge)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect();
    assert_eq!(
        edge_lifetimes,
        vec![(fixture.created, Some(fixture.deleted))]
    );

    let label_history = string_label_history(db, fixture.subject);
    let deletion_states: Vec<_> = label_history
        .iter()
        .filter(|(epoch, _)| *epoch == fixture.deleted)
        .collect();
    assert!(
        !deletion_states.is_empty(),
        "label state at the structural delete boundary must be retained"
    );
    assert_eq!(
        deletion_states.last().expect("boundary state").1,
        vec![String::from("Ephemeral"), String::from("Victim")],
        "same-transaction label mutations must precede structural deletion"
    );
}

#[test]
fn portable_snapshot_preserves_closed_structural_property_and_label_history() {
    let source = GrafeoDB::new_in_memory();
    let timeline = build_closed_timeline(&source);
    assert_timeline(&source, timeline);

    let bytes = source.export_snapshot().expect("export portable snapshot");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import portable snapshot");

    assert_timeline(&restored, timeline);
}

#[test]
fn portable_snapshot_orders_same_epoch_labels_edge_delete_and_node_delete() {
    let mut source = GrafeoDB::new_in_memory();
    let fixture = build_same_epoch_edge_node_deletion(&mut source, |_| {});
    assert_same_epoch_edge_node_deletion(&source, fixture);

    let bytes = source.export_snapshot().expect("export boundary snapshot");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import boundary snapshot");

    assert_same_epoch_edge_node_deletion(&restored, fixture);
}

#[cfg(feature = "compact-store")]
#[test]
fn same_epoch_labels_on_layered_store_survive_repeated_compact_and_copy() {
    let mut source = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_gc_interval(0))
        .expect("open retained-history database");
    let fixture = build_same_epoch_edge_node_deletion(&mut source, |db| {
        db.compact()
            .expect("compact before label mutations and deletion");
    });
    assert_same_epoch_edge_node_deletion(&source, fixture);
    for _ in 0..2 {
        source.compact().expect("compact closed boundary history");
        assert!(source.get_node(fixture.subject).is_none());
        assert!(source.get_edge(fixture.edge).is_none());
        assert!(source.get_node(fixture.peer).is_some());
        // The raw store handle is overlay-only after folding. Check complete
        // label history through the existing exact portable-copy consumer.
        let copied = source.to_memory().expect("copy closed boundary history");
        assert_same_epoch_edge_node_deletion(&copied, fixture);
    }
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_closed_structural_property_and_label_history() {
    let source = GrafeoDB::new_in_memory();
    let timeline = build_closed_timeline(&source);
    assert_timeline(&source, timeline);

    let temp = tempfile::tempdir().expect("create temporary directory");
    let path = temp.path().join("temporal-copy");
    source.save(&path).expect("save exact container copy");

    let restored = GrafeoDB::open(&path).expect("reopen exact container copy");
    assert_timeline(&restored, timeline);
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().expect("close exact container copy");
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_same_epoch_labels_edge_delete_and_node_delete() {
    let mut source = GrafeoDB::new_in_memory();
    let fixture = build_same_epoch_edge_node_deletion(&mut source, |_| {});
    assert_same_epoch_edge_node_deletion(&source, fixture);

    let temp = tempfile::tempdir().expect("create temporary directory");
    let path = temp.path().join("same-epoch-delete-copy");
    source.save(&path).expect("save same-epoch exact copy");

    let restored = GrafeoDB::open(&path).expect("reopen same-epoch exact copy");
    assert_same_epoch_edge_node_deletion(&restored, fixture);
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().expect("close same-epoch exact copy");
}

#[test]
fn portable_snapshot_preserves_named_graph_temporal_history() {
    let source = GrafeoDB::new_in_memory();
    assert!(source.create_graph("archive").expect("create named graph"));
    let timeline = build_closed_timeline_in_graph(&source, Some("archive"));
    assert_named_graph_timeline(&source, "archive", timeline);

    let bytes = source
        .export_snapshot()
        .expect("export named-graph snapshot");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import named-graph snapshot");

    assert_named_graph_timeline(&restored, "archive", timeline);
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_named_graph_temporal_history() {
    let source = GrafeoDB::new_in_memory();
    assert!(source.create_graph("archive").expect("create named graph"));
    let timeline = build_closed_timeline_in_graph(&source, Some("archive"));
    assert_named_graph_timeline(&source, "archive", timeline);

    let temp = tempfile::tempdir().expect("create temporary directory");
    let path = temp.path().join("named-temporal-copy");
    source.save(&path).expect("save named-graph exact copy");

    let restored = GrafeoDB::open(&path).expect("reopen named-graph exact copy");
    assert_named_graph_timeline(&restored, "archive", timeline);
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().expect("close named-graph exact copy");
}

#[derive(Debug, Clone, Copy)]
struct PluralHistory {
    subject: NodeId,
    anchor: NodeId,
    edge: EdgeId,
}

fn advance_to_epoch(db: &GrafeoDB, target: u64) {
    while db.current_epoch().as_u64() < target {
        let mut session = db.session();
        session.begin_transaction().expect("begin epoch advance");
        session.commit().expect("commit epoch advance");
    }
}

fn build_plural_history_source() -> (GrafeoDB, PluralHistory) {
    let db = GrafeoDB::new_in_memory();
    advance_to_epoch(&db, 12);

    let subject = NodeId::new(40);
    let anchor = NodeId::new(41);
    let edge = EdgeId::new(30);
    let e1 = EpochId::new(1);
    let e2 = EpochId::new(2);
    let e3 = EpochId::new(3);
    let e5 = EpochId::new(5);
    let e6 = EpochId::new(6);
    let e7 = EpochId::new(7);
    let e8 = EpochId::new(8);
    let e9 = EpochId::new(9);
    let e10 = EpochId::new(10);

    grafeo_engine::database::testing::root_lpg_store(&db)
        .restore_node_history_exact(anchor, &[(e1, None)], &[(e1, vec![ArcStr::from("Anchor")])])
        .expect("restore anchor history");
    grafeo_engine::database::testing::root_lpg_store(&db)
        .restore_node_history_exact(
            subject,
            &[(e2, Some(e5)), (e6, Some(e10))],
            &[
                (e2, vec![ArcStr::from("FirstLife")]),
                (e5, vec![ArcStr::from("BoundaryOnly")]),
                (e6, vec![ArcStr::from("SecondLife")]),
                (
                    e8,
                    vec![ArcStr::from("SecondLife"), ArcStr::from("Tracked")],
                ),
                (e10, vec![ArcStr::from("ClosedFinal")]),
            ],
        )
        .expect("restore plural node history");
    grafeo_engine::database::testing::root_lpg_store(&db)
        .restore_edge_history_exact(
            edge,
            subject,
            anchor,
            "RETURNS_TO",
            &[(e3, Some(e5)), (e7, Some(e9))],
        )
        .expect("restore plural edge history");

    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property_at_epoch(
        subject,
        "phase",
        Value::String("first".into()),
        e2,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property_at_epoch(
        subject,
        "phase",
        Value::Null,
        e5,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property_at_epoch(
        subject,
        "phase",
        Value::String("second".into()),
        e6,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property_at_epoch(
        subject,
        "phase",
        Value::Null,
        e10,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_edge_property_at_epoch(
        edge,
        "weight",
        Value::Int64(1),
        e3,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_edge_property_at_epoch(
        edge,
        "weight",
        Value::Null,
        e5,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_edge_property_at_epoch(
        edge,
        "weight",
        Value::Int64(2),
        e7,
    );
    grafeo_engine::database::testing::root_lpg_store(&db).set_edge_property_at_epoch(
        edge,
        "weight",
        Value::Null,
        e9,
    );

    (
        db,
        PluralHistory {
            subject,
            anchor,
            edge,
        },
    )
}

fn assert_plural_history(db: &GrafeoDB, history: PluralHistory) {
    let first = db
        .get_node_at_epoch(history.subject, EpochId::new(2))
        .expect("first node lifetime");
    assert!(first.has_label("FirstLife"));
    assert_eq!(
        first.get_property("phase").cloned(),
        Some(Value::String("first".into()))
    );
    assert!(
        db.get_node_at_epoch(history.subject, EpochId::new(5))
            .is_none(),
        "node is structurally absent at the first delete boundary"
    );
    let second = db
        .get_node_at_epoch(history.subject, EpochId::new(8))
        .expect("second node lifetime");
    assert!(second.has_label("SecondLife"));
    assert!(second.has_label("Tracked"));
    assert_eq!(
        second.get_property("phase").cloned(),
        Some(Value::String("second".into()))
    );
    assert!(
        db.get_node_at_epoch(history.subject, EpochId::new(10))
            .is_none(),
        "node is structurally absent at the final delete boundary"
    );
    assert!(db.get_node(history.subject).is_none());
    assert!(db.get_node(history.anchor).is_some());

    assert_eq!(
        db.get_edge_at_epoch(history.edge, EpochId::new(3))
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(1))
    );
    assert!(
        db.get_edge_at_epoch(history.edge, EpochId::new(5))
            .is_none()
    );
    assert_eq!(
        db.get_edge_at_epoch(history.edge, EpochId::new(7))
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(2))
    );
    assert!(
        db.get_edge_at_epoch(history.edge, EpochId::new(9))
            .is_none()
    );

    assert_eq!(
        string_label_history(db, history.subject),
        vec![
            (EpochId::new(2), vec![String::from("FirstLife")]),
            (EpochId::new(5), vec![String::from("BoundaryOnly")]),
            (EpochId::new(6), vec![String::from("SecondLife")]),
            (
                EpochId::new(8),
                vec![String::from("SecondLife"), String::from("Tracked")],
            ),
            (EpochId::new(10), vec![String::from("ClosedFinal")]),
        ]
    );
}

#[test]
fn portable_snapshot_preserves_plural_node_and_edge_lifetimes_exactly() {
    let (source, history) = build_plural_history_source();
    assert_plural_history(&source, history);

    let bytes = source.export_snapshot().expect("export plural history");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import plural history");

    assert_plural_history(&restored, history);
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_plural_node_and_edge_lifetimes() {
    let (source, history) = build_plural_history_source();
    let temp = tempfile::tempdir().expect("create temporary directory");
    let path = temp.path().join("plural-node-copy");

    let expected = source.export_snapshot().expect("capture plural histories");
    source
        .save(&path)
        .expect("save plural node and edge histories");
    assert!(path.is_file());
    let restored = GrafeoDB::open(&path).expect("reopen plural histories");
    assert_plural_history(&restored, history);
    assert_eq!(restored.export_snapshot().unwrap(), expected);
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_plural_edge_lifetimes() {
    let source = GrafeoDB::new_in_memory();
    advance_to_epoch(&source, 12);
    let left = NodeId::new(50);
    let right = NodeId::new(51);
    let edge = EdgeId::new(40);
    grafeo_engine::database::testing::root_lpg_store(&source)
        .restore_node_history_exact(
            left,
            &[(EpochId::new(1), None)],
            &[(EpochId::new(1), vec![ArcStr::from("Left")])],
        )
        .expect("restore left endpoint");
    grafeo_engine::database::testing::root_lpg_store(&source)
        .restore_node_history_exact(
            right,
            &[(EpochId::new(1), None)],
            &[(EpochId::new(1), vec![ArcStr::from("Right")])],
        )
        .expect("restore right endpoint");
    grafeo_engine::database::testing::root_lpg_store(&source)
        .restore_edge_history_exact(
            edge,
            left,
            right,
            "REPEATS",
            &[
                (EpochId::new(2), Some(EpochId::new(4))),
                (EpochId::new(6), Some(EpochId::new(8))),
            ],
        )
        .expect("restore plural edge history");

    let temp = tempfile::tempdir().expect("create temporary directory");
    let path = temp.path().join("plural-edge-copy");
    let expected = source
        .export_snapshot()
        .expect("capture plural edge history");
    source.save(&path).expect("save plural edge history");
    assert!(path.is_file());
    let restored = GrafeoDB::open(&path).expect("reopen plural edge history");
    for epoch in [2, 3, 6, 7] {
        let row = restored
            .get_edge_at_epoch(edge, EpochId::new(epoch))
            .unwrap();
        assert_eq!((row.src, row.dst), (left, right));
        assert_eq!(row.edge_type.as_str(), "REPEATS");
    }
    for epoch in [0, 1, 4, 5, 8, 12] {
        assert!(
            restored
                .get_edge_at_epoch(edge, EpochId::new(epoch))
                .is_none()
        );
    }
    assert_eq!(restored.export_snapshot().unwrap(), expected);
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().unwrap();
}

#[cfg(feature = "compact-store")]
fn assert_live_multi_label_node(db: &GrafeoDB, node_id: NodeId, city: Option<&str>) {
    let node = db.get_node(node_id).expect("multi-label node must be live");
    let mut labels: Vec<_> = node.labels.iter().map(|label| label.to_string()).collect();
    labels.sort_unstable();
    assert_eq!(
        labels,
        vec![String::from("Person"), String::from("Researcher")],
        "materialization must retain two independent labels"
    );
    assert_eq!(
        node.get_property("name").cloned(),
        Some(Value::String("Alix".into()))
    );
    assert_eq!(
        node.get_property("city").cloned(),
        city.map(|value| Value::String(value.into()))
    );

    let store = db.graph_store();
    assert_eq!(
        store.nodes_by_label("Person"),
        vec![node_id],
        "the first label must independently resolve the compacted row"
    );
    assert_eq!(
        store.nodes_by_label("Researcher"),
        vec![node_id],
        "the second label must independently resolve the compacted row"
    );
    assert!(
        store.nodes_by_label("Person|Researcher").is_empty(),
        "a physical table key must never escape as a synthetic graph label"
    );
    let graph_labels = store.all_labels();
    assert!(graph_labels.iter().any(|label| label == "Person"));
    assert!(graph_labels.iter().any(|label| label == "Researcher"));
    assert!(
        graph_labels.iter().all(|label| !label.contains('|')),
        "logical label enumeration leaked a compound table key: {graph_labels:?}"
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_property_promotion_preserves_independent_multi_labels() {
    let mut source = GrafeoDB::new_in_memory();
    let mut session = source.session();
    session
        .begin_transaction()
        .expect("begin multi-label create");
    let node_id = session
        .create_node_with_props(
            &["Person", "Researcher"],
            [("name", Value::String("Alix".into()))],
        )
        .expect("create multi-label node");
    session.commit().expect("commit multi-label create");
    let created = source.current_epoch();
    drop(session);

    source.compact().expect("compact multi-label node");
    assert_live_multi_label_node(&source, node_id, None);

    let mut session = source.session();
    session
        .begin_transaction()
        .expect("begin property-only promotion");
    session
        .set_node_property(node_id, "city", Value::String("Amsterdam".into()))
        .expect("promote base node by property mutation");
    session.commit().expect("commit property-only promotion");
    drop(session);
    assert_live_multi_label_node(&source, node_id, Some("Amsterdam"));

    source.compact().expect("recompact promoted node");
    assert_live_multi_label_node(&source, node_id, Some("Amsterdam"));
    let at_create = source
        .get_node_at_epoch(node_id, created)
        .expect("multi-label node at creation epoch");
    assert!(at_create.has_label("Person"));
    assert!(at_create.has_label("Researcher"));
    assert!(!at_create.has_label("Person|Researcher"));
    assert!(at_create.get_property("city").is_none());

    let bytes = source
        .export_snapshot()
        .expect("export re-compacted multi-label node");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import multi-label node");
    assert_live_multi_label_node(&restored, node_id, Some("Amsterdam"));
    let restored_at_create = restored
        .get_node_at_epoch(node_id, created)
        .expect("restored multi-label node at creation");
    assert!(restored_at_create.has_label("Person"));
    assert!(restored_at_create.has_label("Researcher"));
    assert!(!restored_at_create.has_label("Person|Researcher"));
    assert!(restored_at_create.get_property("city").is_none());
}

#[cfg(feature = "compact-store")]
fn assert_two_lifetime_edge(db: &GrafeoDB, edge_id: EdgeId, stage: &str) {
    assert!(
        db.get_edge_at_epoch(edge_id, EpochId::new(1)).is_none(),
        "edge must be absent before its first lifetime ({stage})"
    );
    assert_eq!(
        db.get_edge_at_epoch(edge_id, EpochId::new(2))
            .and_then(|edge| edge.get_property("phase").cloned()),
        Some(Value::String("first".into())),
        "first lifetime property at creation ({stage})"
    );
    assert_eq!(
        db.get_edge_at_epoch(edge_id, EpochId::new(3))
            .and_then(|edge| edge.get_property("phase").cloned()),
        Some(Value::String("first".into())),
        "first lifetime property before deletion ({stage})"
    );
    assert!(
        db.get_edge_at_epoch(edge_id, EpochId::new(4)).is_none(),
        "edge must close exactly at the first delete boundary ({stage})"
    );
    assert!(
        db.get_edge_at_epoch(edge_id, EpochId::new(5)).is_none(),
        "the structural gap between edge lifetimes must remain empty ({stage})"
    );
    assert_eq!(
        db.get_edge_at_epoch(edge_id, EpochId::new(6))
            .and_then(|edge| edge.get_property("phase").cloned()),
        Some(Value::String("second".into())),
        "second lifetime property at creation ({stage})"
    );
    assert_eq!(
        db.get_edge_at_epoch(edge_id, EpochId::new(7))
            .and_then(|edge| edge.get_property("phase").cloned()),
        Some(Value::String("second".into())),
        "second lifetime property before deletion ({stage})"
    );
    assert!(
        db.get_edge_at_epoch(edge_id, EpochId::new(8)).is_none(),
        "edge must close exactly at its final delete boundary ({stage})"
    );
    assert!(db.get_edge(edge_id).is_none());
}

#[cfg(feature = "compact-store")]
fn restore_open_endpoint(db: &GrafeoDB, id: NodeId, label: &str) {
    grafeo_engine::database::testing::root_lpg_store(db)
        .restore_node_history_exact(
            id,
            &[(EpochId::new(1), None)],
            &[(EpochId::new(1), vec![ArcStr::from(label)])],
        )
        .unwrap_or_else(|error| panic!("restore endpoint {id}: {error}"));
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_roundtrip_preserves_two_edge_lifetimes_and_their_property_states() {
    let mut source = GrafeoDB::new_in_memory();
    advance_to_epoch(&source, 10);
    let left = NodeId::new(60);
    let right = NodeId::new(61);
    let edge_id = EdgeId::new(50);
    restore_open_endpoint(&source, left, "Left");
    restore_open_endpoint(&source, right, "Right");
    grafeo_engine::database::testing::root_lpg_store(&source)
        .restore_edge_history_exact(
            edge_id,
            left,
            right,
            "REAPPEARS",
            &[
                (EpochId::new(2), Some(EpochId::new(4))),
                (EpochId::new(6), Some(EpochId::new(8))),
            ],
        )
        .expect("restore exact edge lifetimes");
    grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
        edge_id,
        "phase",
        Value::String("first".into()),
        EpochId::new(2),
    );
    grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
        edge_id,
        "phase",
        Value::Null,
        EpochId::new(4),
    );
    grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
        edge_id,
        "phase",
        Value::String("second".into()),
        EpochId::new(6),
    );
    grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
        edge_id,
        "phase",
        Value::Null,
        EpochId::new(8),
    );

    assert_two_lifetime_edge(&source, edge_id, "source");
    source.compact().expect("compact exact edge history");
    assert_two_lifetime_edge(&source, edge_id, "after compact");
    source.compact().expect("recompact exact edge history");
    assert_two_lifetime_edge(&source, edge_id, "after recompact");

    let bytes = source
        .export_snapshot()
        .expect("export re-compacted edge history");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import edge history");
    assert_two_lifetime_edge(&restored, edge_id, "after portable roundtrip");
}

#[cfg(feature = "compact-store")]
fn assert_mixed_storage_edge_lifetimes(db: &GrafeoDB, edge_id: EdgeId, stage: &str) {
    for epoch in [2, 3] {
        let edge = db
            .get_edge_at_epoch(edge_id, EpochId::new(epoch))
            .unwrap_or_else(|| panic!("structure-only first life at epoch {epoch} ({stage})"));
        assert_eq!(edge.edge_type.as_str(), "MIXED_STORAGE");
        assert!(
            edge.get_property("phase").is_none(),
            "structure-only life acquired a property at epoch {epoch} ({stage})"
        );
    }
    for epoch in [4, 5] {
        assert!(
            db.get_edge_at_epoch(edge_id, EpochId::new(epoch)).is_none(),
            "edge gap must be empty at epoch {epoch} ({stage})"
        );
    }
    for epoch in [6, 7] {
        assert_eq!(
            db.get_edge_at_epoch(edge_id, EpochId::new(epoch))
                .and_then(|edge| edge.get_property("phase").cloned()),
            Some(Value::String("property-bearing".into())),
            "second-life property mismatch at epoch {epoch} ({stage})"
        );
    }
    assert!(
        db.get_edge_at_epoch(edge_id, EpochId::new(8)).is_none(),
        "edge must close at its final boundary ({stage})"
    );
    assert!(db.get_edge(edge_id).is_none());
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_roundtrip_preserves_mixed_structure_only_and_property_edge_lives() {
    let mut source = GrafeoDB::new_in_memory();
    advance_to_epoch(&source, 10);
    let left = NodeId::new(70);
    let right = NodeId::new(71);
    let edge_id = EdgeId::new(60);
    restore_open_endpoint(&source, left, "Left");
    restore_open_endpoint(&source, right, "Right");
    grafeo_engine::database::testing::root_lpg_store(&source)
        .restore_edge_history_exact(
            edge_id,
            left,
            right,
            "MIXED_STORAGE",
            &[
                (EpochId::new(2), Some(EpochId::new(4))),
                (EpochId::new(6), Some(EpochId::new(8))),
            ],
        )
        .expect("restore mixed-storage edge lifetimes");
    grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
        edge_id,
        "phase",
        Value::String("property-bearing".into()),
        EpochId::new(6),
    );
    grafeo_engine::database::testing::root_lpg_store(&source).set_edge_property_at_epoch(
        edge_id,
        "phase",
        Value::Null,
        EpochId::new(8),
    );

    assert_mixed_storage_edge_lifetimes(&source, edge_id, "source");
    source.compact().expect("compact mixed edge lives");
    assert_mixed_storage_edge_lifetimes(&source, edge_id, "after compact");
    source.compact().expect("recompact mixed edge lives");
    assert_mixed_storage_edge_lifetimes(&source, edge_id, "after recompact");

    let bytes = source.export_snapshot().expect("export mixed edge lives");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import mixed edge lives");
    assert_mixed_storage_edge_lifetimes(&restored, edge_id, "after portable roundtrip");
}

#[cfg(feature = "compact-store")]
fn compact_recompact_with_overlay_write(db: &mut GrafeoDB, timeline: Timeline) {
    db.compact().expect("compact temporal history");
    assert_timeline(db, timeline);

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin post-compact transaction");
    session
        .set_node_property(timeline.peer, "tier", Value::String("overlay".into()))
        .expect("write through mutable overlay");
    session.commit().expect("commit overlay write");
    drop(session);

    db.compact()
        .expect("fold overlay history into compact base");
    assert_timeline(db, timeline);
    assert_eq!(
        db.get_node(timeline.peer)
            .and_then(|node| node.get_property("tier").cloned()),
        Some(Value::String("overlay".into())),
        "the overlay write must survive recompaction"
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn portable_snapshot_after_recompact_preserves_the_complete_timeline() {
    let mut source = GrafeoDB::new_in_memory();
    let timeline = build_closed_timeline(&source);
    compact_recompact_with_overlay_write(&mut source, timeline);

    let bytes = source
        .export_snapshot()
        .expect("export re-compacted portable snapshot");
    let restored = GrafeoDB::import_snapshot(&bytes).expect("import re-compacted snapshot");

    assert_timeline(&restored, timeline);
    assert_eq!(
        restored
            .get_node(timeline.peer)
            .and_then(|node| node.get_property("tier").cloned()),
        Some(Value::String("overlay".into()))
    );
}

#[cfg(all(feature = "wal", feature = "compact-store"))]
#[test]
fn exact_save_after_recompact_preserves_the_complete_timeline() {
    let mut source = GrafeoDB::new_in_memory();
    let timeline = build_closed_timeline(&source);
    compact_recompact_with_overlay_write(&mut source, timeline);

    let temp = tempfile::tempdir().expect("create temporary directory");
    let path = temp.path().join("recompacted-temporal-copy");
    source
        .save(&path)
        .expect("save re-compacted exact container copy");

    let restored = GrafeoDB::open(&path).expect("reopen re-compacted exact container copy");
    assert_timeline(&restored, timeline);
    assert_eq!(
        restored
            .get_node(timeline.peer)
            .and_then(|node| node.get_property("tier").cloned()),
        Some(Value::String("overlay".into()))
    );
    assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
    restored.close().expect("close exact container copy");
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
mod projection_generations {
    use grafeo_common::types::Value;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    const PERSON: &str = "http://ex.org/Person";

    fn both_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
            .expect("create dual-model database")
    }

    fn projected_iris(db: &GrafeoDB) -> Vec<String> {
        let mut iris: Vec<_> = grafeo_engine::database::testing::root_lpg_store(db)
            .all_nodes()
            .filter(|node| node.has_label("Person"))
            .filter_map(|node| match node.get_property("iri") {
                Some(Value::String(iri)) => Some(iri.to_string()),
                _ => None,
            })
            .collect();
        iris.sort_unstable();
        iris
    }

    fn build_three_generations(db: &GrafeoDB) -> u64 {
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
            .expect("insert first RDF source row");
        let id = db
            .declare_rdf_lpg_projection(PERSON, "Person")
            .expect("declare projection");
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
        assert_eq!(db.rdf_lpg_projection(id).unwrap().generation(), 1);

        db.execute_sparql(r#"INSERT DATA { <http://ex.org/gus> a <http://ex.org/Person> . }"#)
            .expect("insert second RDF source row");
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 2);
        assert_eq!(db.rdf_lpg_projection(id).unwrap().generation(), 2);

        db.execute_sparql(r#"DELETE DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
            .expect("delete first RDF source row");
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
        assert_projection_state(db, id, 3);
        id
    }

    fn assert_projection_state(db: &GrafeoDB, id: u64, generation: u64) {
        let status = db
            .rdf_lpg_projection(id)
            .expect("projection definition and status");
        assert_eq!(status.type_iri(), PERSON);
        assert_eq!(status.node_label(), "Person");
        assert_eq!(status.generation(), generation);
        assert_eq!(status.row_count(), 1);
        assert_eq!(db.rdf_projection_lag(id), Some(0));
        assert_eq!(projected_iris(db), vec![String::from("http://ex.org/gus")]);
    }

    #[test]
    fn portable_snapshot_preserves_latest_of_multiple_projection_generations() {
        let source = both_db();
        let id = build_three_generations(&source);

        let bytes = source
            .export_snapshot()
            .expect("export multi-generation projection");
        let restored = GrafeoDB::import_snapshot(&bytes).expect("import projection snapshot");
        assert_projection_state(&restored, id, 3);

        assert_eq!(restored.rebuild_rdf_lpg_projection(id).unwrap(), 1);
        assert_projection_state(&restored, id, 4);
    }

    #[cfg(feature = "wal")]
    #[test]
    fn exact_save_preserves_latest_of_multiple_projection_generations() {
        let source = both_db();
        let id = build_three_generations(&source);
        let temp = tempfile::tempdir().expect("create temporary directory");
        let path = temp.path().join("projection-generations-copy");
        source.save(&path).expect("save projection generations");

        let restored =
            GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Both))
                .expect("reopen projection generations");
        assert_projection_state(&restored, id, 3);
        assert_eq!(restored.world_cut().unwrap(), source.world_cut().unwrap());
        assert_eq!(
            restored.export_snapshot().unwrap(),
            source.export_snapshot().unwrap()
        );

        assert_eq!(restored.rebuild_rdf_lpg_projection(id).unwrap(), 1);
        assert_projection_state(&restored, id, 4);
        restored.close().expect("close projection exact copy");
    }
}
