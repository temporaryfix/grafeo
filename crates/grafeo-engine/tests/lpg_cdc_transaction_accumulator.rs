//! Transaction-owned LPG CDC accumulator qualification.
//!
//! These witnesses intentionally qualify only the existing process-local
//! `ChangeEvent` / `CdcLog` projection. They make no persistence or restart
//! survival claim.

#![cfg(all(feature = "cdc", feature = "lpg", feature = "gql"))]
#![allow(missing_docs)]

use std::sync::Arc;
#[cfg(feature = "testing-statement-injection")]
use std::sync::mpsc;
#[cfg(feature = "testing-statement-injection")]
use std::time::Duration;

#[cfg(feature = "testing-statement-injection")]
use grafeo_common::types::NodeId;
use grafeo_common::types::{EpochId, GraphPath, Value};
use grafeo_engine::cdc::{ChangeEvent, ChangeKind, EntityId};
use grafeo_engine::transaction::IsolationLevel;
use grafeo_engine::{Config, GrafeoDB};

fn lpg_path(components: &[&str]) -> Result<GraphPath, grafeo_common::types::GraphPathError> {
    GraphPath::from_components(components)
}

fn is_lpg_graph(event: &ChangeEvent, components: &[&str]) -> bool {
    event.graph_path().is_some_and(|path| {
        path.components()
            .iter()
            .map(String::as_str)
            .eq(components.iter().copied())
    })
}

fn cdc_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_cdc()).expect("open CDC database")
}

fn all_changes(db: &GrafeoDB) -> Vec<ChangeEvent> {
    db.fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
        .expect("read complete process-local CDC projection")
}

fn has_label(event: &ChangeEvent, label: &str) -> bool {
    event
        .labels
        .as_ref()
        .is_some_and(|labels| labels.iter().any(|candidate| candidate == label))
}

fn create_events_with_label<'a>(
    events: &'a [ChangeEvent],
    label: &'a str,
) -> impl Iterator<Item = &'a ChangeEvent> {
    events
        .iter()
        .filter(move |event| event.kind == ChangeKind::Create && has_label(event, label))
}

fn populate_copy_source(db: &GrafeoDB, graph_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let session = db.session();
    session
        .execute(&format!("CREATE GRAPH {graph_name}"))
        .expect("create copy source graph");
    session.use_graph_path(&lpg_path(&[graph_name])?)?;
    let alix = session
        .create_node_with_props(
            &["Person", "Source"],
            [("name", Value::from("Alix")), ("rank", Value::Int64(7))],
        )
        .expect("create first source node");
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .expect("create second source node");
    session
        .create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(2024))])
        .expect("create source edge");
    Ok(())
}

#[test]
fn direct_query_batch_and_session_creates_share_the_commit_accumulator() {
    let db = cdc_db();

    let db_direct = db.create_node(&["DbDirect"]);
    assert!(db_direct.is_valid());

    let session_direct = db.session().create_node(&["SessionDirect"]);
    assert!(session_direct.is_valid());

    db.session()
        .execute("INSERT (:QueryPath)")
        .expect("query-path create");

    let batch = db.batch_create_nodes(
        "BatchPath",
        "embedding",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    );
    assert_eq!(batch.len(), 2);

    let changes = all_changes(&db);
    for (label, expected) in [
        ("DbDirect", 1),
        ("SessionDirect", 1),
        ("QueryPath", 1),
        ("BatchPath", 2),
    ] {
        let actual: Vec<_> = create_events_with_label(&changes, label).collect();
        assert_eq!(
            actual.len(),
            expected,
            "{label} must emit exactly one Create per committed node"
        );
        assert!(
            actual.iter().all(|event| event.epoch != EpochId::PENDING),
            "{label} events must be stamped only with the real commit epoch"
        );
    }

    let db_direct_history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(db_direct))
        .expect("DB-direct node history");
    assert_eq!(
        db_direct_history
            .iter()
            .filter(|event| event.kind == ChangeKind::Create)
            .count(),
        1,
        "moving DB CRUD into Session publication must not duplicate its event"
    );
    let session_direct_history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(session_direct))
        .expect("Session-direct node history");
    assert_eq!(
        session_direct_history
            .iter()
            .filter(|event| event.kind == ChangeKind::Create)
            .count(),
        1
    );
}

#[test]
fn named_lpg_events_use_exact_paths_separately_from_rdf_coordinates()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    assert!(db.create_graph("analytics").expect("create named graph"));

    let direct = db.session();
    direct.use_graph_path(&lpg_path(&["analytics"])?)?;
    let direct_id = direct.create_node(&["NamedDirect"]);
    assert!(direct_id.is_valid());

    let query = db.session();
    query.use_graph_path(&lpg_path(&["analytics"])?)?;
    query
        .execute("INSERT (:NamedQuery)")
        .expect("query mutation in named LPG graph");

    let changes = all_changes(&db);
    for label in ["NamedDirect", "NamedQuery"] {
        let event = create_events_with_label(&changes, label)
            .next()
            .unwrap_or_else(|| panic!("missing named graph event for {label}"));
        assert!(event.triple_graph.is_none());
        assert_eq!(event.graph_path(), Some(&lpg_path(&["analytics"])?));
    }

    let default_id = db.create_node(&["DefaultGraph"]);
    assert!(default_id.is_valid());
    assert_eq!(
        default_id, direct_id,
        "independent graph-local allocators deliberately reuse NodeId(0)"
    );
    let default_event = all_changes(&db)
        .into_iter()
        .find(|event| event.kind == ChangeKind::Create && has_label(event, "DefaultGraph"))
        .expect("default graph create event");
    assert_eq!(default_event.graph_path(), Some(&lpg_path(&[])?));

    let aggregate = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(default_id))
        .expect("aggregate history");
    assert_eq!(
        aggregate.len(),
        2,
        "entity-only lookup remains an aggregate across graph-local ID collisions"
    );
    let default_history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (default_id).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("default graph-qualified history");
    assert_eq!(default_history.len(), 1);
    assert!(has_label(&default_history[0], "DefaultGraph"));
    let named_history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (direct_id).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["analytics"])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("named graph-qualified history");
    assert_eq!(named_history.len(), 1);
    assert!(has_label(&named_history[0], "NamedDirect"));
    assert_eq!(
        named_history[0].graph_path(),
        Some(&lpg_path(&["analytics"])?)
    );

    let session_named_history = direct
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (direct_id).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["analytics"])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("Session graph-qualified history");
    assert_eq!(session_named_history.len(), 1);
    assert_eq!(
        session_named_history[0].entity_id,
        named_history[0].entity_id
    );
    assert_eq!(session_named_history[0].kind, named_history[0].kind);
    assert_eq!(session_named_history[0].epoch, named_history[0].epoch);
    assert_eq!(
        session_named_history[0].graph_path(),
        Some(&lpg_path(&["analytics"])?)
    );
    Ok(())
}

#[test]
fn create_graph_as_copy_of_stages_complete_create_snapshots_once()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    populate_copy_source(&db, "copy_source")?;

    let session = db.session();
    let pre_copy_epoch = db.current_epoch();
    session
        .execute("CREATE GRAPH copied AS COPY OF copy_source")
        .expect("commit copied graph");
    let commit_epoch = db.current_epoch();

    let copied: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| is_lpg_graph(event, &["copied"]) && event.epoch == commit_epoch)
        .collect();
    assert_eq!(copied.len(), 3, "two copied nodes and one copied edge");
    assert!(
        copied.iter().all(|event| event.kind == ChangeKind::Create),
        "properties belong to the entity Create snapshot, not duplicate Update events"
    );
    assert!(
        copied.iter().all(|event| event.epoch != EpochId::PENDING),
        "every copy event is stamped with the one durable commit epoch"
    );

    let alix = copied
        .iter()
        .find(|event| {
            event.entity_id.is_node()
                && event
                    .after
                    .as_ref()
                    .and_then(|properties| properties.get("name"))
                    == Some(&Value::from("Alix"))
        })
        .expect("copied Alix node Create");
    assert!(has_label(alix, "Person"));
    assert!(has_label(alix, "Source"));
    assert_eq!(
        alix.after
            .as_ref()
            .and_then(|properties| properties.get("rank")),
        Some(&Value::Int64(7))
    );

    let gus = copied
        .iter()
        .find(|event| {
            event.entity_id.is_node()
                && event
                    .after
                    .as_ref()
                    .and_then(|properties| properties.get("name"))
                    == Some(&Value::from("Gus"))
        })
        .expect("copied Gus node Create");
    let EntityId::Node(alix_id) = alix.entity_id else {
        panic!("Alix copy event must identify a node");
    };
    let EntityId::Node(gus_id) = gus.entity_id else {
        panic!("Gus copy event must identify a node");
    };

    let edge = copied
        .iter()
        .find(|event| matches!(event.entity_id, EntityId::Edge(_)))
        .expect("copied edge Create");
    assert_eq!(edge.edge_type.as_deref(), Some("KNOWS"));
    assert_eq!(edge.src_id, Some(alix_id.as_u64()));
    assert_eq!(edge.dst_id, Some(gus_id.as_u64()));
    assert_eq!(
        edge.after
            .as_ref()
            .and_then(|properties| properties.get("since")),
        Some(&Value::Int64(2024))
    );

    let destination = db.session();
    destination.use_graph_path(&lpg_path(&["copied"])?)?;
    for event in &copied {
        match event.entity_id {
            EntityId::Node(id) => {
                let node = destination
                    .get_node(id)
                    .unwrap_or_else(|| panic!("event node {id} must exist in copied state"));
                let properties: std::collections::HashMap<_, _> = node
                    .properties
                    .iter()
                    .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                    .collect();
                let expected = (!properties.is_empty()).then_some(properties);
                assert_eq!(
                    event.after.as_ref(),
                    expected.as_ref(),
                    "node event payload is the committed node post-image"
                );
                let mut state_labels: Vec<_> =
                    node.labels.iter().map(ToString::to_string).collect();
                state_labels.sort_unstable();
                let mut event_labels = event.labels.clone().unwrap_or_default();
                event_labels.sort_unstable();
                assert_eq!(event_labels, state_labels);
            }
            EntityId::Edge(id) => {
                let state_edge = destination
                    .get_edge(id)
                    .unwrap_or_else(|| panic!("event edge {id} must exist in copied state"));
                let properties: std::collections::HashMap<_, _> = state_edge
                    .properties
                    .iter()
                    .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                    .collect();
                let expected = (!properties.is_empty()).then_some(properties);
                assert_eq!(event.after.as_ref(), expected.as_ref());
                assert_eq!(event.src_id, Some(state_edge.src.as_u64()));
                assert_eq!(event.dst_id, Some(state_edge.dst.as_u64()));
                assert_eq!(
                    event.edge_type.as_deref(),
                    Some(state_edge.edge_type.as_str())
                );
            }
            EntityId::Triple(_) => panic!("LPG copy emitted an RDF event"),
            _ => panic!("LPG copy emitted an unsupported future entity kind"),
        }
    }
    assert_eq!(
        destination
            .execute("MATCH (n) RETURN n")
            .expect("read current copied nodes")
            .row_count(),
        2
    );
    for (query, expected_at_commit, entity_kind) in [
        ("MATCH (n) RETURN n", 2, "nodes"),
        ("MATCH ()-[e]->() RETURN e", 1, "edges"),
    ] {
        assert_eq!(
            destination
                .execute_at_epoch(query, pre_copy_epoch)
                .unwrap_or_else(|error| panic!("read copied {entity_kind} before commit: {error}"))
                .row_count(),
            0,
            "copied {entity_kind} must be absent at the exact cut before COPY"
        );
        assert_eq!(
            destination
                .execute_at_epoch(query, commit_epoch)
                .unwrap_or_else(|error| panic!("read copied {entity_kind} at commit: {error}"))
                .row_count(),
            expected_at_commit,
            "copied {entity_kind} must be born at the exact COPY commit epoch"
        );
    }
    Ok(())
}

#[test]
fn copy_of_inherits_the_source_graph_type_binding() -> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING)")
        .expect("create allowed node type");
    session
        .execute("CREATE NODE TYPE Animal (species STRING)")
        .expect("create forbidden node type");
    session
        .execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
        .expect("create graph type");
    session
        .execute("CREATE GRAPH typed_source TYPED PeopleOnly")
        .expect("create typed source");
    session.use_graph_path(&lpg_path(&["typed_source"])?)?;
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("populate typed source");
    session.reset_graph()?;
    session
        .execute("CREATE GRAPH typed_copy AS COPY OF typed_source")
        .expect("copy typed source");
    session.use_graph_path(&lpg_path(&["typed_copy"])?)?;

    assert!(
        session
            .execute("INSERT (:Animal {species: 'forbidden'})")
            .is_err(),
        "AS COPY OF includes the source schema binding, not data alone"
    );
    assert!(
        session.execute("INSERT (:Person {name: 'Gus'})").is_ok(),
        "the inherited binding must preserve allowed writes"
    );
    Ok(())
}

#[test]
fn copy_and_like_pin_one_coherent_source_incarnation() -> Result<(), Box<dyn std::error::Error>> {
    for (suffix, clause) in [("copy", "AS COPY OF"), ("like", "LIKE")] {
        let db = cdc_db();
        let setup = db.session();
        setup
            .execute("CREATE NODE TYPE CoherentNode (name STRING)")
            .expect("create source node type");
        setup
            .execute("CREATE GRAPH TYPE CoherentOnly (NODE TYPE CoherentNode)")
            .expect("create source graph type");
        setup
            .execute("CREATE GRAPH coherent_source TYPED CoherentOnly")
            .expect("create original source incarnation");
        setup.use_graph_path(&lpg_path(&["coherent_source"])?)?;
        setup
            .execute("INSERT (:CoherentNode {name: 'original'})")
            .expect("populate original source incarnation");

        let mut inheritor = db.session();
        inheritor
            .begin_transaction()
            .expect("begin inherited graph tx");
        inheritor
            .execute(&format!(
                "CREATE GRAPH coherent_{suffix} {clause} coherent_source"
            ))
            .expect("stage graph from coherent source bundle");

        let replacer = db.session();
        replacer
            .execute("DROP GRAPH coherent_source")
            .expect("drop original source incarnation");
        replacer
            .execute("CREATE GRAPH coherent_source")
            .expect("publish untyped replacement source");

        assert!(
            inheritor.commit().is_err(),
            "{clause} must validate the exact source Arc supplying inherited metadata/data"
        );
        assert!(
            !db.list_graphs()
                .iter()
                .any(|name| name == &format!("coherent_{suffix}")),
            "a target assembled from a stale source cut must not publish"
        );
    }
    Ok(())
}

#[test]
fn create_graph_if_not_exists_as_copy_of_is_a_true_no_op() -> Result<(), Box<dyn std::error::Error>>
{
    let db = cdc_db();
    populate_copy_source(&db, "noop_copy_source")?;
    let target = db.session();
    target
        .execute("CREATE GRAPH existing_target")
        .expect("create existing target");
    target.use_graph_path(&lpg_path(&["existing_target"])?)?;
    let retained = target.create_node(&["RetainedTarget"]);
    assert!(retained.is_valid());
    let target_events_before = all_changes(&db)
        .into_iter()
        .filter(|event| is_lpg_graph(event, &["existing_target"]))
        .count();

    target
        .execute("CREATE GRAPH IF NOT EXISTS existing_target AS COPY OF noop_copy_source")
        .expect("IF NOT EXISTS copy no-op");
    target
        .execute("CREATE GRAPH IF NOT EXISTS existing_target AS COPY OF missing_source")
        .expect("an unused missing COPY source cannot invalidate the no-op");

    assert_eq!(
        target
            .execute("MATCH (n) RETURN n")
            .expect("read unchanged existing target")
            .row_count(),
        1,
        "the existing target must not receive copied state"
    );
    assert!(target.get_node(retained).is_some());
    assert_eq!(
        all_changes(&db)
            .into_iter()
            .filter(|event| is_lpg_graph(event, &["existing_target"]))
            .count(),
        target_events_before,
        "the no-op must stage no copied entity events"
    );
    Ok(())
}

#[test]
fn graph_grants_cover_copy_sources_and_every_session_cdc_read_shape()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    for graph in ["secret", "visible", "allowed_existing"] {
        admin
            .execute(&format!("CREATE GRAPH {graph}"))
            .expect("create authorization fixture graph");
    }
    admin.use_graph_path(&lpg_path(&["secret"])?)?;
    let secret = admin.create_node(&["SecretOnly"]);
    admin.use_graph_path(&lpg_path(&["visible"])?)?;
    let visible = admin.create_node(&["VisibleOnly"]);
    admin.use_graph_path(&lpg_path(&["allowed_existing"])?)?;
    let retained = admin.create_node(&["RetainedAllowedTarget"]);
    assert_eq!(
        secret.as_u64(),
        visible.as_u64(),
        "the aggregate-history witness requires a graph-local ID collision"
    );

    let writer = db.session_with_identity(
        Identity::new("restricted-writer", [Role::ReadWrite]).with_grants([
            Grant::new(lpg_path(&["allowed_target"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["allowed_like"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["allowed_existing"])?, Role::ReadWrite),
        ]),
    );
    assert!(
        writer
            .execute("CREATE GRAPH allowed_target AS COPY OF secret")
            .is_err(),
        "COPY must not read a source outside the identity's graph grants"
    );
    assert!(
        writer
            .execute("CREATE GRAPH allowed_like LIKE secret")
            .is_err(),
        "LIKE must not inherit metadata from a denied source"
    );
    assert!(
        !db.list_graphs().iter().any(|name| name == "allowed_target")
            && !db.list_graphs().iter().any(|name| name == "allowed_like")
    );
    writer
        .execute("CREATE GRAPH IF NOT EXISTS allowed_existing AS COPY OF secret")
        .expect("an existing permitted target is a no-op before source authorization");
    let existing_reader = db.session();
    existing_reader.use_graph_path(&lpg_path(&["allowed_existing"])?)?;
    assert!(existing_reader.get_node(retained).is_some());

    let reader = db.session_with_identity(
        Identity::new("restricted-reader", [Role::ReadOnly])
            .with_grants([Grant::new(lpg_path(&["visible"])?, Role::ReadOnly)]),
    );
    assert!(
        reader
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (secret).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["secret"])?).clone()),
                since_epoch: EpochId::INITIAL
            })
            .is_err(),
        "qualified history must fail closed for a denied graph"
    );
    assert!(
        reader
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (secret).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["secret"])?).clone()),
                since_epoch: EpochId::INITIAL
            })
            .is_err()
    );
    let qualified = reader
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (visible).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["visible"])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("read granted qualified history");
    assert_eq!(qualified.len(), 1);
    assert!(has_label(&qualified[0], "VisibleOnly"));

    for aggregate in [
        reader
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(visible))
            .expect("read filtered legacy history"),
        reader
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (visible).into(),
                graph: grafeo_engine::cdc::HistoryGraph::All,
                since_epoch: EpochId::INITIAL,
            })
            .expect("read filtered legacy since-history"),
    ] {
        assert_eq!(aggregate.len(), 1);
        assert_eq!(aggregate[0].graph_path(), Some(&lpg_path(&["visible"])?));
        assert!(has_label(&aggregate[0], "VisibleOnly"));
    }
    let range = reader
        .fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
        .expect("read grant-filtered CDC range");
    assert!(!range.is_empty());
    assert!(
        range.iter().all(|event| is_lpg_graph(event, &["visible"])),
        "aggregate CDC APIs must never expose denied graph events"
    );
    Ok(())
}

#[test]
fn schema_graph_grants_bind_exact_storage_coordinates() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    admin
        .execute("CREATE GRAPH shared")
        .expect("create root graph with colliding local name");
    admin.use_graph_path(&lpg_path(&["shared"])?)?;
    let root_entity = admin.create_node(&["RootOnly"]);
    admin.reset_graph()?;
    admin
        .execute("CREATE SCHEMA vault")
        .expect("create authorization fixture schema");
    admin
        .execute("SESSION SET SCHEMA vault")
        .expect("select authorization fixture schema");
    admin
        .execute("CREATE NODE TYPE VaultNode (name STRING)")
        .expect("create schema-local node type");
    admin
        .execute("CREATE GRAPH TYPE VaultOnly (NODE TYPE VaultNode)")
        .expect("create schema-local graph type");
    for graph in ["shared", "source", "doomed"] {
        admin
            .execute(&format!("CREATE GRAPH {graph}"))
            .unwrap_or_else(|error| panic!("create vault/{graph}: {error}"));
    }
    admin.use_graph_path(&lpg_path(&["vault/source"])?)?;
    let vault_entity = admin.create_node(&["VaultOnly"]);
    admin.reset_graph()?;

    let local_only =
        db.session_with_identity(Identity::new("local-only", [Role::ReadWrite]).with_grants([
            Grant::new(lpg_path(&["shared"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["source"])?, Role::ReadOnly),
            Grant::new(lpg_path(&["new_copy"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["new_like"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["new_typed"])?, Role::ReadWrite),
        ]));
    local_only
        .execute("SESSION SET SCHEMA vault")
        .expect("select schema without changing graph grants");
    for command in [
        "CREATE GRAPH IF NOT EXISTS shared AS COPY OF missing_source",
        "CREATE GRAPH new_copy AS COPY OF source",
        "CREATE GRAPH new_like LIKE source",
        "CREATE GRAPH new_typed TYPED VaultOnly",
        "DROP GRAPH shared",
        "USE GRAPH shared",
        "SESSION SET GRAPH shared",
    ] {
        assert!(
            local_only.execute(command).is_err(),
            "a local-name grant must not authorize vault coordinates: {command}"
        );
    }
    assert!(
        local_only
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (vault_entity).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg(
                    (lpg_path(&["vault/source"])?).clone()
                ),
                since_epoch: EpochId::INITIAL
            })
            .is_err(),
        "qualified CDC history must require the exact schema coordinate"
    );
    let local_aggregate = local_only
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(vault_entity))
        .expect("aggregate history filters inaccessible coordinates");
    assert!(
        local_aggregate
            .iter()
            .all(|event| is_lpg_graph(event, &["shared"])),
        "aggregate CDC history may include the granted colliding root ID, but must not let that local-name grant roam into the current schema"
    );
    assert!(db.list_graphs().iter().any(|name| name == "vault/shared"));
    assert!(
        !db.list_graphs()
            .iter()
            .any(|name| name.starts_with("vault/new_")),
        "denied graph creation must leave no schema-local residue"
    );

    let target_only = db.session_with_identity(
        Identity::new("target-only", [Role::ReadWrite])
            .with_grants([Grant::new(lpg_path(&["vault/shared"])?, Role::ReadWrite)]),
    );
    target_only
        .execute("SESSION SET SCHEMA vault")
        .expect("select exact target schema");
    target_only
        .execute("CREATE GRAPH IF NOT EXISTS shared AS COPY OF source")
        .expect("authorized existing target is a no-op before unused source authorization");

    let exact = db.session_with_identity(
        Identity::new("exact-coordinate", [Role::ReadWrite]).with_grants([
            Grant::new(lpg_path(&["vault/source"])?, Role::ReadOnly),
            Grant::new(lpg_path(&["vault/new_copy"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["vault/new_like"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["vault/new_typed"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["vault/doomed"])?, Role::ReadWrite),
        ]),
    );
    exact
        .execute("SESSION SET SCHEMA vault")
        .expect("select exact-coordinate schema");
    exact
        .execute("USE GRAPH source")
        .expect("exact canonical read grant authorizes USE GRAPH");
    exact.reset_graph()?;
    assert_eq!(
        exact
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (vault_entity).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg(
                    (lpg_path(&["vault/source"])?).clone()
                ),
                since_epoch: EpochId::INITIAL
            })
            .expect("read exact-coordinate qualified history")
            .len(),
        1
    );
    let exact_aggregate = exact
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(root_entity))
        .expect("aggregate history is grant-filtered");
    assert!(
        exact_aggregate
            .iter()
            .all(|event| is_lpg_graph(event, &["vault/source"])),
        "the vault/source grant may expose its colliding local ID, but must not expose the root coordinate"
    );
    exact
        .execute("CREATE GRAPH new_copy AS COPY OF source")
        .expect("copy with exact source and target grants");
    exact
        .execute("CREATE GRAPH new_like LIKE source")
        .expect("like with exact source and target grants");
    exact
        .execute("CREATE GRAPH new_typed TYPED VaultOnly")
        .expect("typed create with an exact target grant");
    exact
        .execute("DROP GRAPH doomed")
        .expect("drop with an exact target grant");
    Ok(())
}

#[test]
fn graph_grants_guard_parser_free_query_and_direct_data_planes()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    admin
        .execute("CREATE SCHEMA vault")
        .expect("create data-plane fixture schema");
    admin
        .execute("SESSION SET SCHEMA vault")
        .expect("select data-plane fixture schema");
    let default_node = admin.create_node(&["VaultDefault"]);
    admin
        .execute("CREATE GRAPH secret")
        .expect("create denied named graph");
    admin.use_graph_path(&lpg_path(&["vault/secret"])?)?;
    let secret_node = admin.create_node(&["Secret"]);
    admin.reset_graph()?;

    let denied = db.session_with_identity(
        Identity::new("parser-free-denied", [Role::ReadWrite])
            .with_grants([Grant::new(lpg_path(&["allowed"])?, Role::ReadWrite)]),
    );
    denied.set_schema("vault")?;
    let denied_context = denied.current_graph_path();
    assert!(
        denied
            .use_graph_path(&lpg_path(&["vault/secret"])?)
            .is_err()
    );
    assert_eq!(denied.current_graph_path(), denied_context);
    assert!(
        denied.execute("MATCH (n) RETURN n").is_err(),
        "query reads must revalidate parser-free graph context"
    );
    assert!(
        denied.execute("INSERT (:QueryBypass)").is_err(),
        "query writes must revalidate parser-free graph context"
    );
    assert!(
        denied.get_node(secret_node).is_none()
            && denied.get_node_history(secret_node).is_empty()
            && !denied.node_exists(secret_node),
        "direct reads must disclose no data from a denied exact coordinate"
    );
    assert!(
        !denied.create_node(&["DirectBypass"]).is_valid(),
        "infallible direct writes must return their failure sentinel"
    );
    assert!(
        denied
            .create_node_with_props(&["DirectBypass"], [("value", Value::Int64(1))])
            .is_err(),
        "fallible direct writes must report authorization failure"
    );
    assert!(
        denied
            .set_node_property(secret_node, "leaked", Value::Bool(true))
            .is_err(),
        "direct updates must report authorization failure"
    );

    denied.reset_graph()?;
    assert!(
        denied.execute("MATCH (n) RETURN n").is_err(),
        "a schema selection implicitly targets its exact default partition"
    );
    assert!(denied.get_node(default_node).is_none());
    assert!(!denied.create_node(&["DefaultBypass"]).is_valid());

    let exact_named = db.session_with_identity(
        Identity::new("parser-free-exact-named", [Role::ReadWrite])
            .with_grants([Grant::new(lpg_path(&["vault/secret"])?, Role::ReadWrite)]),
    );
    exact_named.set_schema("vault")?;
    exact_named.use_graph_path(&lpg_path(&["vault/secret"])?)?;
    assert!(exact_named.get_node(secret_node).is_some());
    assert_eq!(
        exact_named
            .execute("MATCH (n) RETURN n")
            .expect("query read with exact named grant")
            .row_count(),
        1
    );
    assert!(exact_named.create_node(&["ExactDirect"]).is_valid());
    exact_named
        .execute("INSERT (:ExactQuery)")
        .expect("query write with exact named grant");

    let exact_default = db.session_with_identity(
        Identity::new("parser-free-exact-default", [Role::ReadWrite]).with_grants([Grant::new(
            lpg_path(&["vault/__default__"])?,
            Role::ReadWrite,
        )]),
    );
    exact_default.set_schema("vault")?;
    exact_default.reset_graph()?;
    assert!(exact_default.get_node(default_node).is_some());
    assert_eq!(
        exact_default
            .execute("MATCH (n) RETURN n")
            .expect("query read with exact schema-default grant")
            .row_count(),
        1
    );
    assert!(exact_default.create_node(&["ExactDefault"]).is_valid());
    Ok(())
}

#[test]
fn graph_grants_guard_index_ddl_and_metadata_surfaces() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    admin
        .execute("CREATE INDEX root_idx FOR (n:Doc) ON (n.root_value)")
        .expect("create root index fixture");
    for graph in ["secret", "visible"] {
        admin
            .execute(&format!("CREATE GRAPH {graph}"))
            .unwrap_or_else(|error| panic!("create {graph}: {error}"));
    }
    admin.use_graph_path(&lpg_path(&["secret"])?)?;
    admin
        .execute("CREATE INDEX secret_idx FOR (n:Doc) ON (n.secret_value)")
        .expect("create denied index fixture");
    admin.use_graph_path(&lpg_path(&["visible"])?)?;
    admin
        .execute("CREATE INDEX visible_idx FOR (n:Doc) ON (n.visible_value)")
        .expect("create allowed index fixture");
    admin.reset_graph()?;

    let restricted = db.session_with_identity(
        Identity::new("restricted-index-admin", [Role::Admin])
            .with_grants([Grant::new(lpg_path(&["visible"])?, Role::ReadWrite)]),
    );
    assert!(restricted.use_graph_path(&lpg_path(&["secret"])?).is_err());
    assert_eq!(restricted.current_graph_path(), GraphPath::root());
    assert!(
        restricted
            .execute("CREATE INDEX denied_new FOR (n:Doc) ON (n.denied_value)")
            .is_err(),
        "parser-free graph selection must not bypass exact index-write grants"
    );

    restricted.use_graph_path(&lpg_path(&["visible"])?)?;
    restricted
        .execute("CREATE INDEX allowed_new FOR (n:Doc) ON (n.allowed_value)")
        .expect("exact write grant authorizes index creation");
    for denied_drop in ["secret_idx", "root_idx"] {
        assert!(
            restricted
                .execute(&format!("DROP INDEX {denied_drop}"))
                .is_err(),
            "a global index name must not bypass its graph's exact write grant: {denied_drop}"
        );
    }

    let shown_indexes = restricted.execute("SHOW INDEXES").expect("show indexes");
    let shown_index_names: Vec<String> = shown_indexes
        .rows()
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::String(name)) => Some(name.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(shown_index_names, ["allowed_new", "visible_idx"]);

    let shown_graphs = restricted.execute("SHOW GRAPHS").expect("show graphs");
    let shown_graph_names: Vec<String> = shown_graphs
        .rows()
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::String(name)) => Some(name.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(shown_graph_names, ["visible"]);

    restricted
        .execute("DROP INDEX visible_idx")
        .expect("exact write grant authorizes index removal");
    Ok(())
}

#[test]
fn graph_grants_bind_virtual_projection_ownership_to_exact_coordinates()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    admin.execute("CREATE GRAPH shared").unwrap();
    admin.use_graph_path(&lpg_path(&["shared"])?)?;
    admin
        .execute("CREATE PROJECTION root_shared_projection LABELS (Person)")
        .expect("create root-coordinate projection");
    admin.reset_graph()?;
    admin.execute("CREATE SCHEMA vault").unwrap();
    admin.set_schema("vault")?;
    admin.execute("CREATE GRAPH shared").unwrap();
    admin.use_graph_path(&lpg_path(&["vault/shared"])?)?;
    admin
        .execute("CREATE PROJECTION vault_shared_projection LABELS (Person)")
        .expect("create schema-coordinate projection");

    let root_only = db.session_with_identity(
        Identity::new("root-projection-writer", [Role::ReadWrite])
            .with_grants([Grant::new(lpg_path(&["shared"])?, Role::ReadWrite)]),
    );
    root_only.set_schema("vault")?;
    let denied_context = root_only.current_graph_path();
    assert!(
        root_only
            .use_graph_path(&lpg_path(&["vault/shared"])?)
            .is_err()
    );
    assert_eq!(root_only.current_graph_path(), denied_context);
    assert!(
        root_only
            .execute("CREATE PROJECTION denied_projection LABELS (Person)")
            .is_err(),
        "a local-name grant must not create a projection over vault/shared"
    );
    assert!(
        root_only
            .execute("DROP PROJECTION vault_shared_projection")
            .is_err(),
        "a global projection name must not bypass its owner's exact grant"
    );
    let root_visible = root_only
        .execute("SHOW PROJECTIONS")
        .expect("show grant-filtered projections");
    assert_eq!(
        root_visible.rows(),
        &[vec![Value::from("root_shared_projection")]]
    );

    let vault_only = db.session_with_identity(
        Identity::new("vault-projection-writer", [Role::ReadWrite])
            .with_grants([Grant::new(lpg_path(&["vault/shared"])?, Role::ReadWrite)]),
    );
    vault_only.set_schema("vault")?;
    vault_only.use_graph_path(&lpg_path(&["vault/shared"])?)?;
    vault_only
        .execute("CREATE PROJECTION allowed_projection LABELS (Person)")
        .expect("exact coordinate grant authorizes projection creation");
    vault_only
        .execute("DROP PROJECTION vault_shared_projection")
        .expect("exact owner-coordinate grant authorizes projection removal");
    let vault_visible = vault_only.execute("SHOW PROJECTIONS").unwrap();
    assert_eq!(
        vault_visible.rows(),
        &[vec![Value::from("allowed_projection")]]
    );
    Ok(())
}

#[test]
fn graph_grants_bind_schema_lifecycle_to_its_exact_default_coordinate()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    for command in ["CREATE SCHEMA ``", "DROP SCHEMA ``"] {
        assert!(
            admin.execute(command).is_err(),
            "empty schema coordinates must be rejected: {command}"
        );
    }
    admin
        .execute("CREATE SCHEMA Vault")
        .expect("create canonical mixed-case schema fixture");

    let denied = db.session_with_identity(
        Identity::new("restricted-schema-admin", [Role::Admin]).with_grants([Grant::new(
            lpg_path(&["other/__default__"])?,
            Role::ReadWrite,
        )]),
    );
    for command in [
        "CREATE SCHEMA IF NOT EXISTS vAuLt",
        "DROP SCHEMA IF EXISTS vAuLt",
        "CREATE SCHEMA denied",
        "DROP SCHEMA IF EXISTS missing",
    ] {
        assert!(
            denied.execute(command).is_err(),
            "schema lifecycle must authorize the canonical default coordinate before existence or namespace checks: {command}"
        );
    }

    let exact = db.session_with_identity(
        Identity::new("exact-schema-admin", [Role::Admin]).with_grants([Grant::new(
            lpg_path(&["Vault/__default__"])?,
            Role::ReadWrite,
        )]),
    );
    exact
        .execute("CREATE SCHEMA IF NOT EXISTS vAuLt")
        .expect("canonical exact grant authorizes IF NOT EXISTS");
    exact
        .execute("DROP SCHEMA vAuLt")
        .expect("canonical exact grant authorizes schema removal");

    let create_exact = db.session_with_identity(
        Identity::new("new-schema-admin", [Role::Admin]).with_grants([Grant::new(
            lpg_path(&["allowed/__default__"])?,
            Role::ReadWrite,
        )]),
    );
    create_exact
        .execute("CREATE SCHEMA allowed")
        .expect("requested exact coordinate authorizes new schema creation");
    Ok(())
}

#[test]
fn graph_grants_require_complete_schema_scope_for_catalog_ddl_and_type_metadata()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    admin.execute("CREATE SCHEMA governed").unwrap();
    admin.set_schema("governed")?;
    admin
        .execute("CREATE NODE TYPE Person (email STRING)")
        .unwrap();
    admin
        .execute("CREATE GRAPH TYPE BoundGraph (NODE TYPE Person)")
        .unwrap();
    admin
        .execute("CREATE GRAPH secret TYPED BoundGraph")
        .unwrap();

    let partial = db.session_with_identity(
        Identity::new("partial-schema-admin", [Role::Admin]).with_grants([Grant::new(
            lpg_path(&["governed/__default__"])?,
            Role::ReadWrite,
        )]),
    );
    partial.set_schema("governed")?;
    assert!(
        partial
            .execute("CREATE CONSTRAINT denied_unique FOR (n:Person) ON (n.email) UNIQUE")
            .is_err(),
        "schema-wide constraint DDL requires grants for every graph it scans and governs"
    );
    let drop_error = partial
        .execute("DROP GRAPH TYPE BoundGraph")
        .expect_err("partial scope must not enumerate denied graph-type bindings");
    assert!(
        !drop_error.to_string().contains("governed/secret"),
        "authorization must fail before denied bound-graph names are disclosed"
    );
    let allowed_context = partial.current_graph_path();
    assert!(
        partial
            .use_graph_path(&lpg_path(&["governed/secret"])?)
            .is_err(),
        "native selection must not disclose a denied graph binding"
    );
    assert_eq!(partial.current_graph_path(), allowed_context);

    let complete = db.session_with_identity(
        Identity::new("complete-schema-admin", [Role::Admin]).with_grants([
            Grant::new(lpg_path(&["governed/__default__"])?, Role::ReadWrite),
            Grant::new(lpg_path(&["governed/secret"])?, Role::ReadWrite),
        ]),
    );
    complete.set_schema("governed")?;
    complete
        .execute("CREATE CONSTRAINT allowed_unique FOR (n:Person) ON (n.email) UNIQUE")
        .expect("complete schema scope authorizes constraint creation");
    complete
        .execute("DROP CONSTRAINT allowed_unique")
        .expect("complete schema scope authorizes constraint removal");
    complete.use_graph_path(&lpg_path(&["governed/secret"])?)?;
    let binding = complete
        .execute("SHOW CURRENT GRAPH TYPE")
        .expect("exact read grant authorizes graph-type metadata");
    assert_eq!(binding.row_count(), 1);
    Ok(())
}

#[cfg(all(feature = "algos", feature = "text-index"))]
#[test]
fn graph_grants_bind_indexed_search_to_the_exact_store() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = cdc_db();
    let admin = db.session();
    let root_secret = admin
        .create_node_with_props(
            &["SearchDoc"],
            [("body", Value::from("classified root payload"))],
        )
        .expect("create root search fixture");
    admin
        .execute("CREATE INDEX root_search FOR (n:SearchDoc) ON (n.body) USING TEXT")
        .expect("create root text index");
    admin
        .execute("CREATE GRAPH visible_search")
        .expect("create allowed search graph");
    admin.use_graph_path(&lpg_path(&["visible_search"])?)?;
    let visible = admin
        .create_node_with_props(
            &["SearchDoc"],
            [("body", Value::from("public graph bulletin"))],
        )
        .expect("create allowed search fixture");
    assert_eq!(
        root_secret, visible,
        "the disclosure witness requires a graph-local ID collision"
    );
    admin
        .execute("CREATE INDEX visible_search_idx FOR (n:SearchDoc) ON (n.body) USING TEXT")
        .expect("create named-graph text index");

    let mut restricted = db.session_with_identity(
        Identity::new("restricted-search-reader", [Role::ReadOnly])
            .with_grants([Grant::new(lpg_path(&["visible_search"])?, Role::ReadOnly)]),
    );
    restricted.use_graph_path(&lpg_path(&["visible_search"])?)?;
    let assert_exact_store = |session: &grafeo_engine::Session| {
        assert_eq!(
            session
                .execute("CALL grafeo.search.text('SearchDoc', 'body', 'classified', 10)")
                .expect("authorized named-graph search")
                .row_count(),
            0,
            "an allowed graph's procedure context must not consult the root index"
        );
        assert_eq!(
            session
                .execute("CALL grafeo.search.text('SearchDoc', 'body', 'public', 10)")
                .expect("search exact named-graph index")
                .row_count(),
            1
        );
    };
    assert_exact_store(&restricted);
    restricted
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin Serializable search witness");
    assert_exact_store(&restricted);
    restricted
        .commit()
        .expect("commit read-only Serializable search witness");
    Ok(())
}

#[test]
fn reserved_root_default_and_empty_named_graph_coordinates_are_rejected()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let session = db.session();
    let root_default = session.create_node(&["RootDefault"]);
    session
        .execute("CREATE GRAPH source")
        .expect("create reserved-coordinate copy source");
    session.use_graph_path(&lpg_path(&["source"])?)?;
    session.create_node(&["Source"]);
    session.reset_graph()?;

    for command in [
        "CREATE GRAPH default AS COPY OF source",
        "CREATE GRAPH DeFaUlT",
        "DROP GRAPH default",
        "CREATE GRAPH ``",
    ] {
        assert!(
            session.execute(command).is_err(),
            "reserved root coordinate must be rejected: {command}"
        );
    }
    assert!(
        db.create_graph("").is_err(),
        "the parser-free lifecycle API must reject an empty storage key"
    );
    assert!(db.drop_graph("").is_err());
    assert!(
        db.list_graphs()
            .iter()
            .all(|name| !name.is_empty() && !name.eq_ignore_ascii_case("default")),
        "no unreachable root named graph may enter the registry"
    );
    assert!(
        all_changes(&db).iter().all(|event| {
            event.graph_path().is_none_or(|path| {
                path.components()
                    .first()
                    .is_none_or(|name| !name.is_empty() && !name.eq_ignore_ascii_case("default"))
            })
        }),
        "rejected coordinates must publish no CDC event"
    );

    session
        .execute("CREATE GRAPH root_copy AS COPY OF default")
        .expect("the reserved root selector remains a valid COPY source");
    session
        .execute("CREATE GRAPH root_like LIKE default")
        .expect("the reserved root selector remains a valid LIKE source");
    session.use_graph_path(&lpg_path(&["root_copy"])?)?;
    assert!(session.get_node(root_default).is_some());
    session.reset_graph()?;

    session
        .execute("CREATE SCHEMA vault")
        .expect("create schema namespace control");
    session
        .execute("SESSION SET SCHEMA vault")
        .expect("select schema namespace control");
    let schema_default = session.create_node(&["SchemaDefault"]);
    assert!(
        session.execute("CREATE GRAPH default").is_err(),
        "the local default alias is reserved in every schema"
    );
    session
        .execute("CREATE GRAPH schema_copy AS COPY OF default")
        .expect("schema default remains a valid COPY source");
    session
        .execute("CREATE GRAPH schema_like LIKE default")
        .expect("schema default remains a valid LIKE source");
    session.use_graph_path(&lpg_path(&["vault/schema_copy"])?)?;
    assert!(session.get_node(schema_default).is_some());
    assert!(db.list_graphs().iter().all(|name| name != "vault/default"));
    Ok(())
}

#[test]
fn failed_typed_copy_leaves_no_committable_target_state_or_cdc()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let setup = db.session();
    setup
        .execute("CREATE NODE TYPE CopyNode (name STRING)")
        .expect("create COPY node type");
    setup
        .execute("CREATE GRAPH TYPE CopyType (NODE TYPE CopyNode)")
        .expect("create COPY graph type");

    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin typed COPY witness");
    writer
        .execute("CREATE GRAPH typed_source TYPED CopyType")
        .expect("create transaction-local typed source");
    writer.use_graph_path(&lpg_path(&["typed_source"])?)?;
    writer
        .create_node_with_props(&["CopyNode"], [("name", Value::from("retained"))])
        .expect("populate typed source");
    writer.reset_graph()?;

    assert!(writer.execute("DROP GRAPH TYPE CopyType").is_err());
    assert!(
        writer
            .execute("CREATE GRAPH typed_target TYPED MissingCopyType AS COPY OF typed_source")
            .is_err(),
        "the COPY statement must reject a type absent from its catalog view"
    );
    writer
        .execute("DROP GRAPH typed_source")
        .expect("cancel the source and its stale transaction-local binding");
    writer
        .commit()
        .expect("failed COPY must not leave a target that can commit later");

    assert!(
        db.list_graphs().iter().all(|name| name != "typed_target"),
        "a failed statement must leave no detached target graph"
    );
    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !is_lpg_graph(event, &["typed_target"])),
        "a failed statement must leave no target CDC"
    );
    Ok(())
}

#[test]
fn explicit_typed_copy_rejects_nonconforming_source_before_target_creation()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let setup = db.session();
    setup.execute("CREATE GRAPH untyped_source").unwrap();
    setup.use_graph_path(&lpg_path(&["untyped_source"])?)?;
    setup
        .execute("INSERT (:Person:Animal {name: 'chimera'})")
        .expect("populate an untyped source before declaring its contract");
    setup.reset_graph()?;
    setup
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    setup
        .execute("CREATE NODE TYPE Animal (name STRING NOT NULL)")
        .unwrap();
    setup
        .execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
        .unwrap();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    assert!(
        writer
            .execute("CREATE GRAPH invalid_typed_copy TYPED PeopleOnly AS COPY OF untyped_source",)
            .is_err(),
        "an explicit binding must validate every copied label before target creation"
    );
    writer
        .execute("INSERT (:UnrelatedControl)")
        .expect("the surrounding transaction remains usable after statement rejection");
    writer
        .commit()
        .expect("commit unrelated work without resurrecting the failed COPY");

    assert!(
        !db.list_graphs()
            .iter()
            .any(|graph| graph == "invalid_typed_copy")
    );
    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !is_lpg_graph(event, &["invalid_typed_copy"]))
    );
    db.session()
        .execute("CREATE GRAPH invalid_typed_copy")
        .expect("the rejected statement left neither lifecycle nor binding residue");
    Ok(())
}

#[test]
fn copy_of_uses_the_source_transactions_read_your_writes_post_image()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut session = db.session();
    session
        .execute("CREATE GRAPH live_source")
        .expect("create RYW source");
    session.use_graph_path(&lpg_path(&["live_source"])?)?;
    let retained = session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::from("Retained")),
                ("version", Value::Int64(1)),
            ],
        )
        .expect("create retained source node");
    let peer = session
        .create_node_with_props(&["Item"], [("name", Value::from("Peer"))])
        .expect("create source peer");
    let doomed = session
        .create_node_with_props(&["Item"], [("name", Value::from("Doomed"))])
        .expect("create source node to delete");
    let edge = session
        .create_edge_with_props(retained, peer, "LINK", [("weight", Value::Int64(1))])
        .expect("create source edge");

    session.begin_transaction().expect("begin source/copy tx");
    session
        .set_node_property(retained, "version", Value::Int64(2))
        .expect("buffer source node update");
    session
        .set_edge_property(edge, "weight", Value::Int64(2))
        .expect("buffer source edge update");
    assert!(session.add_node_label(retained, "Reviewed"));
    assert!(session.add_node_label(retained, "Transient"));
    assert!(session.remove_node_label(retained, "Transient"));
    let born = session
        .create_node_with_props(&["Item"], [("name", Value::from("BornInTransaction"))])
        .expect("create source node in transaction");
    assert!(born.is_valid());
    assert!(session.delete_node(doomed));
    session
        .execute("CREATE GRAPH ryw_copy AS COPY OF live_source")
        .expect("copy source transaction post-image");
    let commit_epoch = session.commit().expect("commit source and copy together");

    let source_nodes = session
        .execute("MATCH (n:Item) RETURN n.name, n.version ORDER BY n.name")
        .expect("read committed source nodes")
        .rows()
        .to_vec();
    let source_edges = session
        .execute("MATCH (a)-[e:LINK]->(b) RETURN a.name, e.weight, b.name")
        .expect("read committed source edges")
        .rows()
        .to_vec();
    session.use_graph_path(&lpg_path(&["ryw_copy"])?)?;
    assert_eq!(
        session
            .execute("MATCH (n:Item) RETURN n.name, n.version ORDER BY n.name")
            .expect("read copied nodes")
            .rows(),
        source_nodes,
        "clone must equal the source transaction's node post-image"
    );
    assert_eq!(
        session
            .execute("MATCH (a)-[e:LINK]->(b) RETURN a.name, e.weight, b.name")
            .expect("read copied edges")
            .rows(),
        source_edges,
        "clone must equal the source transaction's edge post-image"
    );

    let copy_events: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| is_lpg_graph(event, &["ryw_copy"]))
        .collect();
    assert_eq!(copy_events.len(), 4, "three live nodes and one live edge");
    assert!(copy_events.iter().all(|event| {
        event.kind == ChangeKind::Create
            && event.epoch == commit_epoch
            && event.epoch != EpochId::PENDING
    }));
    assert!(
        copy_events.iter().all(|event| {
            event
                .after
                .as_ref()
                .and_then(|properties| properties.get("name"))
                != Some(&Value::from("Doomed"))
        }),
        "same-transaction source deletes are absent from state and CDC"
    );
    let retained_copy = copy_events
        .iter()
        .find(|event| {
            event
                .after
                .as_ref()
                .and_then(|properties| properties.get("name"))
                == Some(&Value::from("Retained"))
        })
        .expect("updated retained node copied");
    assert_eq!(
        retained_copy
            .after
            .as_ref()
            .and_then(|properties| properties.get("version")),
        Some(&Value::Int64(2))
    );
    assert!(has_label(retained_copy, "Item"));
    assert!(has_label(retained_copy, "Reviewed"));
    assert!(!has_label(retained_copy, "Transient"));
    let copied_edge = copy_events
        .iter()
        .find(|event| matches!(event.entity_id, EntityId::Edge(_)))
        .expect("updated source edge copied");
    assert_eq!(
        copied_edge
            .after
            .as_ref()
            .and_then(|properties| properties.get("weight")),
        Some(&Value::Int64(2))
    );
    Ok(())
}

#[test]
fn serializable_copy_tracks_a_non_active_source_dataset_phantom()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session()
        .execute("CREATE GRAPH phantom_source")
        .expect("create empty phantom source");
    let seed_session = db.session();
    seed_session.use_graph_path(&lpg_path(&["phantom_source"])?)?;
    let retired_source_id = seed_session.create_node(&["RetiredSeed"]);
    assert!(seed_session.delete_node(retired_source_id));
    let shared = db.create_node_with_props(&["Shared"], [("version", Value::Int64(0))]);

    let mut copier = db.session();
    copier
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin serializable copy");
    assert_eq!(
        copier.current_graph_path(),
        GraphPath::root(),
        "source is deliberately non-active"
    );
    assert!(
        copier.delete_node(shared),
        "stage an immediately tracked structural write for the reverse dependency"
    );
    copier
        .execute("CREATE GRAPH phantom_copy AS COPY OF phantom_source")
        .expect("stage source-wide read after the shared write");

    let mut writer = db.session();
    writer
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin concurrent source writer");
    assert!(
        writer.get_node(shared).is_some(),
        "the writer records the opposite read dependency against the copier's pending delete"
    );
    writer.use_graph_path(&lpg_path(&["phantom_source"])?)?;
    let phantom = writer.create_node(&["ConcurrentPhantom"]);
    assert!(phantom.is_valid());
    assert_ne!(
        phantom.as_u64(),
        shared.as_u64(),
        "the SSI witness must not depend on graph-unqualified entity-ID collision"
    );
    writer.commit().expect("concurrent source insert commits");

    assert!(
        copier.commit().is_err(),
        "a source-wide Serializable read must conflict with a concurrent insert phantom"
    );
    assert!(
        !db.list_graphs().iter().any(|name| name == "phantom_copy"),
        "the losing detached destination must not publish"
    );
    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !is_lpg_graph(event, &["phantom_copy"]))
    );
    assert!(
        db.get_node(shared).is_some(),
        "the aborted copier's cycle-closing delete must roll back"
    );
    Ok(())
}

#[test]
fn read_committed_copy_uses_a_fresh_statement_cut() -> Result<(), Box<dyn std::error::Error>> {
    for (source, create_before_begin) in [
        ("late_created_source", false),
        ("post_begin_mutated_source", true),
    ] {
        let db = cdc_db();
        if create_before_begin {
            db.session()
                .execute(&format!("CREATE GRAPH {source}"))
                .unwrap();
        }

        let mut copier = db.session();
        copier
            .begin_transaction_with_isolation(IsolationLevel::ReadCommitted)
            .unwrap();

        let writer = db.session();
        if !create_before_begin {
            writer.execute(&format!("CREATE GRAPH {source}")).unwrap();
        }
        writer.use_graph_path(&lpg_path(&[source])?)?;
        writer
            .execute("INSERT (:VisibleAtStatement {version: 2})")
            .unwrap();

        let target = format!("{source}_copy");
        copier
            .execute(&format!("CREATE GRAPH {target} AS COPY OF {source}"))
            .expect("Read Committed COPY sees the statement-time committed source");
        let commit_epoch = copier.commit().unwrap();

        let reader = db.session();
        reader.use_graph_path(&lpg_path(&[&target])?)?;
        assert_eq!(
            reader
                .execute("MATCH (n:VisibleAtStatement) RETURN n.version")
                .unwrap()
                .rows(),
            &[vec![Value::Int64(2)]]
        );
        let events: Vec<_> = all_changes(&db)
            .into_iter()
            .filter(|event| is_lpg_graph(event, &[target.as_str()]))
            .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].epoch, commit_epoch);
    }
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn copy_of_compacted_default_reads_the_complete_tiered_snapshot()
-> Result<(), Box<dyn std::error::Error>> {
    let mut db = cdc_db();
    {
        let source = db.session();
        let left = source
            .create_node_with_props(&["Cold"], [("name", Value::from("left"))])
            .unwrap();
        let right = source
            .create_node_with_props(&["Cold"], [("name", Value::from("right"))])
            .unwrap();
        source
            .create_edge_with_props(left, right, "COLD_LINK", [("weight", Value::Int64(7))])
            .unwrap();
    }
    db.compact()
        .expect("move the default graph into the cold tier");

    let copier = db.session();
    copier
        .execute("CREATE GRAPH cold_copy AS COPY OF default")
        .expect("COPY reads both compact base and mutable overlay");
    let reader = db.session();
    reader.use_graph_path(&lpg_path(&["cold_copy"])?)?;
    assert_eq!(
        reader
            .execute("MATCH (n:Cold) RETURN n.name ORDER BY n.name")
            .unwrap()
            .rows(),
        &[vec![Value::from("left")], vec![Value::from("right")]]
    );
    assert_eq!(
        reader
            .execute("MATCH (a)-[e:COLD_LINK]->(b) RETURN e.weight")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(7)]]
    );
    assert_eq!(
        all_changes(&db)
            .iter()
            .filter(|event| is_lpg_graph(event, &["cold_copy"]))
            .count(),
        3,
        "the copied cold-tier post-image emits two node snapshots and one edge snapshot"
    );
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn snapshot_copy_retains_a_cold_row_deleted_after_begin() -> Result<(), Box<dyn std::error::Error>>
{
    let mut db = cdc_db();
    let retained = {
        let source = db.session();
        for ordinal in 0..8 {
            let retired = source
                .create_node_with_props(&["Retired"], [("ordinal", Value::Int64(ordinal))])
                .unwrap();
            assert!(source.delete_node(retired));
        }
        source
            .create_node_with_props(&["SnapshotRow"], [("name", Value::from("retained"))])
            .unwrap()
    };
    db.compact().unwrap();

    let mut copier = db.session();
    copier
        .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert!(
        db.session().delete_node(retained),
        "publish a base-tier tombstone after the COPY snapshot began"
    );
    copier
        .execute("CREATE GRAPH historical_copy AS COPY OF default")
        .expect("candidate enumeration retains identities visible to the old snapshot");
    copier.commit().unwrap();

    let reader = db.session();
    reader.use_graph_path(&lpg_path(&["historical_copy"])?)?;
    assert_eq!(
        reader
            .execute("MATCH (n:SnapshotRow) RETURN n.name")
            .unwrap()
            .rows(),
        &[vec![Value::from("retained")]]
    );
    Ok(())
}

fn explain_uses_property_index(session: &grafeo_engine::Session, property: &str) -> bool {
    let query = format!("EXPLAIN MATCH (n:Person) WHERE n.{property} = 'Alix' RETURN n");
    let result = session.execute(&query).expect("explain property lookup");
    result.rows()[0][0]
        .as_str()
        .is_some_and(|plan| plan.contains(&format!("[index: {property}]")))
}

#[test]
fn copy_inherits_the_source_transactions_property_index_post_image()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut session = db.session();
    session
        .execute("CREATE GRAPH indexed_source")
        .expect("create indexed source");
    session.use_graph_path(&lpg_path(&["indexed_source"])?)?;
    session
        .execute("INSERT (:Person {name: 'Alix', legacy: 'Alix'})")
        .expect("populate indexed source");
    session
        .execute("CREATE INDEX source_legacy FOR (n:Person) ON (n.legacy)")
        .expect("create committed source index");

    session
        .begin_transaction()
        .expect("begin index-copy transaction");
    session
        .execute("DROP INDEX source_legacy")
        .expect("drop source index in transaction");
    session
        .execute("CREATE INDEX source_name FOR (n:Person) ON (n.name)")
        .expect("create source index in transaction");
    session.reset_graph()?;
    session
        .execute("CREATE GRAPH indexed_copy AS COPY OF indexed_source")
        .expect("copy transaction-visible index post-image");
    session.commit().expect("commit source and copied indexes");

    session.use_graph_path(&lpg_path(&["indexed_copy"])?)?;
    assert!(explain_uses_property_index(&session, "name"));
    assert!(
        !explain_uses_property_index(&session, "legacy"),
        "a source index dropped earlier in the transaction must not be copied"
    );
    assert_eq!(
        session
            .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name")
            .expect("query copied property index")
            .row_count(),
        1
    );
    Ok(())
}

#[test]
fn copy_validates_the_exact_source_property_index_registry_cut()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session()
        .execute("CREATE GRAPH metadata_source")
        .expect("create metadata source");
    let source = db.session();
    source.use_graph_path(&lpg_path(&["metadata_source"])?)?;
    source
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("populate metadata source");
    let shared = db.create_node(&["MetadataCycleMarker"]);

    let mut copier = db.session();
    copier
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin serializable metadata copy");
    assert!(
        copier.delete_node(shared),
        "stage an immediately tracked reverse-edge write"
    );
    copier
        .execute("CREATE GRAPH metadata_copy AS COPY OF metadata_source")
        .expect("stage copy of source index absence");

    let mut index_writer = db.session();
    index_writer
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin concurrent source-index writer");
    assert!(
        index_writer.get_node(shared).is_some(),
        "record the reverse dependency against the copier's pending delete"
    );
    index_writer.use_graph_path(&lpg_path(&["metadata_source"])?)?;
    index_writer
        .execute("CREATE INDEX source_name FOR (n:Person) ON (n.name)")
        .expect("stage source property-index creation");
    index_writer
        .commit()
        .expect("concurrent source-index writer commits");

    assert!(
        copier.commit().is_err(),
        "COPY must reject a changed physical source property-index registry"
    );
    assert!(
        db.get_node(shared).is_some(),
        "the losing copier's reverse-edge write must roll back"
    );
    assert!(!db.list_graphs().iter().any(|name| name == "metadata_copy"));
    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !is_lpg_graph(event, &["metadata_copy"]))
    );

    db.session()
        .execute("CREATE GRAPH unrelated_index_source")
        .expect("create unrelated graph");
    let unrelated = db.session();
    unrelated.use_graph_path(&lpg_path(&["unrelated_index_source"])?)?;
    unrelated
        .execute("INSERT (:Person {other: 'value'})")
        .expect("populate unrelated graph");

    let mut disjoint_copy = db.session();
    disjoint_copy
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin disjoint-source copy");
    disjoint_copy
        .execute("CREATE GRAPH disjoint_metadata_copy AS COPY OF metadata_source")
        .expect("stage copy from unchanged exact source");
    unrelated
        .execute("CREATE INDEX unrelated_other FOR (n:Person) ON (n.other)")
        .expect("publish property index on a different source graph");
    disjoint_copy
        .commit()
        .expect("different-graph index DDL must not conflict with COPY");
    assert!(
        db.list_graphs()
            .iter()
            .any(|name| name == "disjoint_metadata_copy")
    );
    Ok(())
}

#[test]
fn copy_keeps_its_statement_time_snapshot_across_own_source_lifecycle()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();

    let mut detached = db.session();
    detached
        .begin_transaction()
        .expect("begin detached-source transaction");
    detached
        .execute("CREATE GRAPH ephemeral_source")
        .expect("create transaction-local source");
    detached.use_graph_path(&lpg_path(&["ephemeral_source"])?)?;
    detached
        .execute("INSERT (:Person {name: 'Ephemeral Alix'})")
        .expect("populate transaction-local source");
    detached
        .execute("CREATE INDEX ephemeral_name FOR (n:Person) ON (n.name)")
        .expect("stage transaction-local source index");
    detached.reset_graph()?;
    detached
        .execute("CREATE GRAPH retained_ephemeral_copy AS COPY OF ephemeral_source")
        .expect("materialize transaction-local source");
    detached
        .execute("DROP GRAPH ephemeral_source")
        .expect("drop source after COPY materialized it");
    detached
        .commit()
        .expect("commit target after cancelling its detached source");
    assert!(
        !db.list_graphs()
            .iter()
            .any(|name| name == "ephemeral_source")
    );
    let ephemeral_copy = db.session();
    ephemeral_copy.use_graph_path(&lpg_path(&["retained_ephemeral_copy"])?)?;
    assert_eq!(
        ephemeral_copy
            .execute("MATCH (n:Person) RETURN n")
            .expect("read retained detached-source copy")
            .row_count(),
        1
    );
    assert!(explain_uses_property_index(&ephemeral_copy, "name"));

    db.session()
        .execute("CREATE GRAPH replaced_source")
        .expect("create shared source");
    let old_source = db.session();
    old_source.use_graph_path(&lpg_path(&["replaced_source"])?)?;
    old_source
        .execute("INSERT (:Person {name: 'Old Alix'})")
        .expect("populate old source incarnation");
    old_source
        .execute("CREATE INDEX old_name FOR (n:Person) ON (n.name)")
        .expect("index old source incarnation");

    let mut replacement = db.session();
    replacement
        .begin_transaction()
        .expect("begin source-replacement transaction");
    replacement
        .execute("CREATE GRAPH retained_old_copy AS COPY OF replaced_source")
        .expect("materialize old source incarnation");
    replacement
        .execute("DROP GRAPH replaced_source")
        .expect("stage old source drop");
    replacement
        .execute("CREATE GRAPH replaced_source")
        .expect("stage empty replacement source");
    replacement
        .commit()
        .expect("commit target and source replacement atomically");

    let retained_old = db.session();
    retained_old.use_graph_path(&lpg_path(&["retained_old_copy"])?)?;
    assert_eq!(
        retained_old
            .execute("MATCH (n:Person) RETURN n")
            .expect("read retained old-incarnation copy")
            .row_count(),
        1
    );
    assert!(explain_uses_property_index(&retained_old, "name"));
    let new_source = db.session();
    new_source.use_graph_path(&lpg_path(&["replaced_source"])?)?;
    assert_eq!(
        new_source
            .execute("MATCH (n) RETURN n")
            .expect("read replacement source")
            .row_count(),
        0
    );
    Ok(())
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn copied_property_index_survives_wal_recovery() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::{Path, PathBuf};

    use grafeo_engine::{DurabilityMode, GraphModel};

    fn sidecar(path: &Path) -> PathBuf {
        let mut value = path.as_os_str().to_owned();
        value.push(".wal");
        PathBuf::from(value)
    }

    fn copy_live_database(source: &Path, destination: &Path) {
        std::fs::copy(source, destination).expect("copy live container");
        let source_wal = sidecar(source);
        if source_wal.exists() {
            let destination_wal = sidecar(destination);
            std::fs::create_dir_all(&destination_wal).expect("create copied WAL directory");
            for entry in std::fs::read_dir(source_wal).expect("read live WAL directory") {
                let entry = entry.expect("WAL directory entry");
                if entry.path().is_file() {
                    std::fs::copy(entry.path(), destination_wal.join(entry.file_name()))
                        .expect("copy WAL segment");
                }
            }
        }
    }

    let directory = tempfile::tempdir().expect("temporary copy-index directory");
    let live = directory.path().join("copy-index-live.grafeo");
    let crash_copy = directory.path().join("copy-index-recovered.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&live)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync)
            .with_cdc(),
    )
    .expect("open persistent copy-index database");
    let session = db.session();
    session
        .execute("CREATE GRAPH source")
        .expect("create source");
    session.use_graph_path(&lpg_path(&["source"])?)?;
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("populate source");
    session
        .execute("CREATE INDEX source_name FOR (n:Person) ON (n.name)")
        .expect("create source index");
    session.reset_graph()?;
    session
        .execute("CREATE GRAPH recovered_copy AS COPY OF source")
        .expect("copy indexed source");
    session.use_graph_path(&lpg_path(&["recovered_copy"])?)?;
    assert!(explain_uses_property_index(&session, "name"));
    db.wal().expect("configured WAL").sync().expect("sync WAL");
    copy_live_database(&live, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = GrafeoDB::open(&crash_copy).expect("recover copied graph and index");
    let reader = recovered.session();
    reader.use_graph_path(&lpg_path(&["recovered_copy"])?)?;
    assert!(explain_uses_property_index(&reader, "name"));
    assert_eq!(
        reader
            .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name")
            .expect("query recovered copied index")
            .row_count(),
        1
    );
    Ok(())
}

#[test]
fn copied_graph_dropped_in_the_same_transaction_publishes_no_entity_events()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    populate_copy_source(&db, "drop_copy_source")?;

    let mut session = db.session();
    session.begin_transaction().expect("begin copy/drop tx");
    session
        .execute("CREATE GRAPH discarded_copy AS COPY OF drop_copy_source")
        .expect("copy detached graph");
    session
        .execute("DROP GRAPH discarded_copy")
        .expect("drop detached copy");
    session.commit().expect("commit absent copy post-image");

    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !is_lpg_graph(event, &["discarded_copy"])),
        "the private destination incarnation is absent from the committed root"
    );
    Ok(())
}

#[test]
fn explicit_rollback_after_copy_publishes_no_entity_events()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    populate_copy_source(&db, "rollback_copy_source")?;

    let mut session = db.session();
    session.begin_transaction().expect("begin copy rollback tx");
    session
        .execute("CREATE GRAPH rolled_back_copy AS COPY OF rollback_copy_source")
        .expect("copy detached graph");
    session.rollback().expect("rollback copied graph");

    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !is_lpg_graph(event, &["rolled_back_copy"])),
        "rollback discards the whole copied-entity accumulator suffix"
    );
    Ok(())
}

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]
#[test]
fn copy_wal_failure_publishes_no_entity_events() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::testing::wal_failure::{
        disable_mutation_log_failure, enable_mutation_log_failure_once,
    };
    use grafeo_engine::{DurabilityMode, GraphModel};

    let directory = tempfile::tempdir().expect("temporary copy WAL directory");
    let path = directory.path().join("cdc-copy-wal-failure.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync)
            .with_cdc(),
    )
    .expect("open persistent CDC database");
    populate_copy_source(&db, "wal_copy_source")?;
    let events_before = db.memory_usage().cdc.event_count;

    enable_mutation_log_failure_once();
    let result = db
        .session()
        .execute("CREATE GRAPH failed_copy AS COPY OF wal_copy_source");
    disable_mutation_log_failure();

    assert!(result.is_err(), "copied entity WAL failure must escape");
    assert!(db.is_durability_poisoned());
    assert_eq!(
        db.memory_usage().cdc.event_count,
        events_before,
        "no copied entity event may precede successful WAL framing"
    );
    Ok(())
}

#[test]
fn create_mutate_drop_filters_the_detached_incarnation_at_commit()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut session = db.session();
    session.begin_transaction().expect("begin lifecycle tx");
    session
        .execute("CREATE GRAPH ephemeral")
        .expect("create detached graph");
    session.use_graph_path(&lpg_path(&["ephemeral"])?)?;
    let node = session.create_node(&["NeverPublished"]);
    assert!(node.is_valid());
    session
        .execute("DROP GRAPH ephemeral")
        .expect("drop detached graph");
    session.commit().expect("commit final absent post-image");

    assert_eq!(
        session
            .execute("SHOW GRAPHS")
            .expect("show graphs")
            .row_count(),
        0
    );
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["ephemeral"])?).clone()),
            since_epoch: EpochId::INITIAL
        })
        .expect("ephemeral graph history")
        .is_empty(),
        "a detached incarnation absent from the committed root must publish no event"
    );
    Ok(())
}

#[test]
fn drop_recreate_filters_old_incarnation_events_and_publishes_only_the_survivor()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut session = db.session();
    session
        .execute("CREATE GRAPH replaceable")
        .expect("create original graph");
    session.use_graph_path(&lpg_path(&["replaceable"])?)?;
    let old_node = session.create_node(&["OriginalIncarnation"]);
    assert!(old_node.is_valid());

    session.begin_transaction().expect("begin replacement tx");
    assert!(session.add_node_label(old_node, "OldMutation"));
    session.reset_graph()?;
    session
        .execute("DROP GRAPH replaceable")
        .expect("drop original graph");
    session
        .execute("CREATE GRAPH replaceable")
        .expect("create replacement graph");
    session.use_graph_path(&lpg_path(&["replaceable"])?)?;
    let new_node = session.create_node(&["ReplacementIncarnation"]);
    assert!(new_node.is_valid());
    session.commit().expect("commit replacement graph");

    let changes = all_changes(&db);
    assert!(
        !changes.iter().any(|event| has_label(event, "OldMutation")),
        "events owned by the dropped original incarnation must be filtered"
    );
    let replacements: Vec<_> =
        create_events_with_label(&changes, "ReplacementIncarnation").collect();
    assert_eq!(replacements.len(), 1);
    assert_eq!(
        replacements[0].graph_path(),
        Some(&lpg_path(&["replaceable"])?)
    );

    session.use_graph_path(&lpg_path(&["replaceable"])?)?;
    assert_eq!(
        session
            .execute("MATCH (n:ReplacementIncarnation) RETURN n")
            .expect("query replacement graph")
            .row_count(),
        1
    );
    assert_eq!(
        session
            .execute("MATCH (n:OriginalIncarnation) RETURN n")
            .expect("query replacement graph for original")
            .row_count(),
        0
    );
    Ok(())
}

#[test]
fn rollback_to_savepoint_restores_the_old_incarnation_and_its_staged_event()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut session = db.session();
    session
        .execute("CREATE GRAPH retained")
        .expect("create retained graph");
    session.use_graph_path(&lpg_path(&["retained"])?)?;
    let node = session.create_node(&["SavepointBase"]);

    session.begin_transaction().expect("begin savepoint tx");
    assert!(session.add_node_label(node, "BeforeDrop"));
    session
        .savepoint("before_drop")
        .expect("capture old incarnation");
    session.reset_graph()?;
    session
        .execute("DROP GRAPH retained")
        .expect("stage graph drop");
    session
        .rollback_to_savepoint("before_drop")
        .expect("restore old incarnation");
    let commit_epoch = session.commit().expect("commit restored prefix");

    session.use_graph_path(&lpg_path(&["retained"])?)?;
    let restored = session.get_node(node).expect("restored old node");
    assert!(
        restored
            .labels
            .iter()
            .any(|label| label.as_str() == "BeforeDrop")
    );
    let retained_events: Vec<_> = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["retained"])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("retained graph history")
        .into_iter()
        .filter(|event| event.epoch == commit_epoch)
        .collect();
    assert_eq!(retained_events.len(), 1);
    assert_eq!(retained_events[0].kind, ChangeKind::Update);
    assert!(has_label(&retained_events[0], "BeforeDrop"));
    Ok(())
}

#[test]
fn savepoint_rewinds_a_pending_created_graph_first_mutated_after_capture()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin detached graph tx");
    session
        .execute("CREATE GRAPH pending_at_savepoint")
        .expect("create detached graph before savepoint");
    session
        .savepoint("before_first_write")
        .expect("capture unwritten detached graph");
    session.use_graph_path(&lpg_path(&["pending_at_savepoint"])?)?;
    let discarded = session.create_node(&["PostSavepointA"]);
    assert!(discarded.is_valid());
    session
        .rollback_to_savepoint("before_first_write")
        .expect("rewind detached graph state");
    let retained = session.create_node(&["PostRollbackB"]);
    assert!(retained.is_valid());
    let commit_epoch = session.commit().expect("commit restored detached graph");

    let reader = db.session();
    reader.use_graph_path(&lpg_path(&["pending_at_savepoint"])?)?;
    assert_eq!(
        reader
            .execute("MATCH (n:PostSavepointA) RETURN n")
            .expect("query discarded detached write")
            .row_count(),
        0
    );
    assert_eq!(
        reader
            .execute("MATCH (n:PostRollbackB) RETURN n")
            .expect("query retained detached write")
            .row_count(),
        1
    );
    let events: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| is_lpg_graph(event, &["pending_at_savepoint"]))
        .collect();
    assert!(
        events
            .iter()
            .all(|event| !has_label(event, "PostSavepointA"))
    );
    let retained_events: Vec<_> = events
        .iter()
        .filter(|event| has_label(event, "PostRollbackB"))
        .collect();
    assert_eq!(retained_events.len(), 1);
    assert_eq!(retained_events[0].epoch, commit_epoch);
    assert_ne!(retained_events[0].epoch, EpochId::PENDING);
    Ok(())
}

#[test]
fn dropping_the_active_graph_then_mutating_default_exactly_touches_both_cuts()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session()
        .execute("CREATE GRAPH active_alpha")
        .expect("create active graph");
    let mut session = db.session();
    session.use_graph_path(&lpg_path(&["active_alpha"])?)?;
    session.begin_transaction().expect("begin active-drop tx");
    session
        .execute("DROP GRAPH active_alpha")
        .expect("drop active graph and reset selector");
    assert_eq!(session.current_graph_path(), GraphPath::root());
    let default_node = session.create_node(&["DefaultAfterActiveDrop"]);
    assert!(default_node.is_valid());
    let commit_epoch = session
        .commit()
        .expect("commit graph drop and default write");

    assert!(!db.list_graphs().iter().any(|name| name == "active_alpha"));
    assert!(db.get_node(default_node).is_some());
    let history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (default_node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("default event after active drop");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].epoch, commit_epoch);
    assert_ne!(history[0].epoch, EpochId::PENDING);
    Ok(())
}

#[test]
fn snapshot_transactions_never_upgrade_a_missing_graph_to_a_later_incarnation()
-> Result<(), Box<dyn std::error::Error>> {
    for (suffix, isolation) in [
        ("si", IsolationLevel::SnapshotIsolation),
        ("serializable", IsolationLevel::Serializable),
    ] {
        let db = cdc_db();
        let graph_name = format!("late_{suffix}");
        let mut pinned_missing = db.session();
        pinned_missing
            .begin_transaction_with_isolation(isolation)
            .expect("begin before first missing-graph selection");
        assert!(
            pinned_missing
                .use_graph_path(&lpg_path(&[&graph_name])?)
                .is_err()
        );
        assert_eq!(pinned_missing.current_graph_path(), GraphPath::root());

        db.session()
            .execute(&format!("CREATE GRAPH {graph_name}"))
            .expect("publish graph after the snapshot began");
        assert!(
            pinned_missing
                .use_graph_path(&lpg_path(&[&graph_name])?)
                .is_err(),
            "{isolation:?} must retain the absence recorded by the rejected selection"
        );
        assert_eq!(pinned_missing.current_graph_path(), GraphPath::root());
        assert!(
            pinned_missing.commit().is_err(),
            "{isolation:?} must validate its pinned absence at commit too"
        );

        let reader = db.session();
        reader.use_graph_path(&lpg_path(&[&graph_name])?)?;
        assert_eq!(
            reader
                .execute("MATCH (n) RETURN n")
                .expect("read later graph")
                .row_count(),
            0
        );
        assert!(
            all_changes(&db)
                .iter()
                .all(|event| !has_label(event, "MustRemainAbsent"))
        );
    }
    Ok(())
}

#[test]
fn read_committed_can_refresh_a_graph_that_was_missing_at_begin()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    let mut refreshing = db.session();
    refreshing
        .begin_transaction_with_isolation(IsolationLevel::ReadCommitted)
        .expect("begin Read Committed transaction");
    assert!(refreshing.use_graph_path(&lpg_path(&["late_rc"])?).is_err());
    assert_eq!(refreshing.current_graph_path(), GraphPath::root());
    db.session()
        .execute("CREATE GRAPH late_rc")
        .expect("publish graph after begin");

    refreshing.use_graph_path(&lpg_path(&["late_rc"])?)?;
    let node = refreshing.create_node(&["ReadCommittedRefresh"]);
    assert!(node.is_valid());
    let commit_epoch = refreshing.commit().expect("commit refreshed target");
    let history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["late_rc"])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("read refreshed graph event");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].epoch, commit_epoch);
    Ok(())
}

#[test]
fn rollback_to_savepoint_preserves_the_surviving_selector_snapshot_expectation()
-> Result<(), Box<dyn std::error::Error>> {
    for (suffix, isolation) in [
        ("si", IsolationLevel::SnapshotIsolation),
        ("serializable", IsolationLevel::Serializable),
    ] {
        let missing_db = cdc_db();
        let missing_name = format!("rollback_missing_{suffix}");
        let mut missing = missing_db.session();
        missing
            .begin_transaction_with_isolation(isolation)
            .expect("begin missing-selector transaction");
        assert!(
            missing
                .use_graph_path(&lpg_path(&[&missing_name])?)
                .is_err()
        );
        assert_eq!(missing.current_graph_path(), GraphPath::root());
        missing.savepoint("after_absence")?;
        missing_db
            .session()
            .execute(&format!("CREATE GRAPH {missing_name}"))
            .expect("publish graph after selector pinned absence");
        missing.rollback_to_savepoint("after_absence")?;
        assert!(
            missing
                .use_graph_path(&lpg_path(&[&missing_name])?)
                .is_err(),
            "{isolation:?} rollback must retain absence recorded before the savepoint"
        );
        assert_eq!(missing.current_graph_path(), GraphPath::root());
        assert!(missing.commit().is_err());
        assert!(
            all_changes(&missing_db)
                .iter()
                .all(|event| !has_label(event, "MustRemainMissingAfterRollback"))
        );

        let replaced_db = cdc_db();
        let replaced_name = format!("rollback_exact_{suffix}");
        replaced_db
            .session()
            .execute(&format!("CREATE GRAPH {replaced_name}"))
            .expect("create original exact graph");
        let mut exact = replaced_db.session();
        exact
            .begin_transaction_with_isolation(isolation)
            .expect("begin exact-selector transaction");
        exact.savepoint("before_selector").expect("savepoint");
        exact.use_graph_path(&lpg_path(&[&replaced_name])?)?;
        replaced_db
            .session()
            .execute(&format!("DROP GRAPH {replaced_name}"))
            .expect("drop original graph concurrently");
        replaced_db
            .session()
            .execute(&format!("CREATE GRAPH {replaced_name}"))
            .expect("publish replacement graph concurrently");
        exact
            .rollback_to_savepoint("before_selector")
            .expect("rollback while retaining exact selector");
        let old_incarnation_write = exact.create_node(&["MustStayOnOldIncarnation"]);
        assert!(old_incarnation_write.is_valid());
        assert!(
            exact.commit().is_err(),
            "{isolation:?} must reject the retained old Arc after replacement"
        );
        let replacement_reader = replaced_db.session();
        replacement_reader.use_graph_path(&lpg_path(&[&replaced_name])?)?;
        assert_eq!(
            replacement_reader
                .execute("MATCH (n:MustStayOnOldIncarnation) RETURN n")
                .expect("inspect replacement graph")
                .row_count(),
            0
        );
        assert!(
            all_changes(&replaced_db)
                .iter()
                .all(|event| !has_label(event, "MustStayOnOldIncarnation"))
        );

        let detached_db = cdc_db();
        let detached_name = format!("rollback_detached_{suffix}");
        let mut detached = detached_db.session();
        detached
            .begin_transaction_with_isolation(isolation)
            .expect("begin detached-selector transaction");
        detached.savepoint("before_create").expect("savepoint");
        detached
            .execute(&format!("CREATE GRAPH {detached_name}"))
            .expect("stage detached graph after savepoint");
        detached.use_graph_path(&lpg_path(&[&detached_name])?)?;
        detached_db
            .session()
            .execute(&format!("CREATE GRAPH {detached_name}"))
            .expect("publish same-named shared graph");
        detached
            .rollback_to_savepoint("before_create")
            .expect("remove detached incarnation while retaining selector");
        let rejected = detached.create_node(&["MustNotUpgradeDetachedSelector"]);
        assert!(
            !rejected.is_valid(),
            "{isolation:?} removed detached selector must degrade to Missing"
        );
        assert!(detached.commit().is_err());
        assert!(
            all_changes(&detached_db)
                .iter()
                .all(|event| !has_label(event, "MustNotUpgradeDetachedSelector"))
        );
    }
    Ok(())
}

#[test]
fn read_committed_refreshes_a_surviving_selector_after_savepoint_rollback()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session().execute("CREATE GRAPH rollback_late_rc")?;
    let mut refreshing = db.session();
    refreshing
        .begin_transaction_with_isolation(IsolationLevel::ReadCommitted)
        .expect("begin Read Committed transaction");
    refreshing.savepoint("before_selector").expect("savepoint");
    refreshing.use_graph_path(&lpg_path(&["rollback_late_rc"])?)?;
    db.session().execute("DROP GRAPH rollback_late_rc")?;
    db.session().execute("CREATE GRAPH rollback_late_rc")?;
    refreshing
        .rollback_to_savepoint("before_selector")
        .expect("rollback while retaining selector");

    let node = refreshing.create_node(&["ReadCommittedRollbackRefresh"]);
    assert!(node.is_valid());
    let commit_epoch = refreshing.commit().expect("commit refreshed target");
    let history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg(
                (lpg_path(&["rollback_late_rc"])?).clone(),
            ),
            since_epoch: EpochId::INITIAL,
        })
        .expect("read refreshed graph event");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].epoch, commit_epoch);
    Ok(())
}

#[test]
fn rollback_to_savepoint_retracks_the_surviving_graph_context()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session()
        .execute("CREATE GRAPH alpha")
        .expect("create named graph");

    let mut session = db.session();
    session.begin_transaction().expect("begin context tx");
    session.savepoint("before_graph").expect("savepoint");
    session.use_graph_path(&lpg_path(&["alpha"])?)?;
    let discarded = session.create_node(&["DiscardedAfterGraphSwitch"]);
    session
        .rollback_to_savepoint("before_graph")
        .expect("rollback graph mutation");
    assert_eq!(session.current_graph_path(), lpg_path(&["alpha"])?);
    let retained = session.create_node(&["RetainedAfterGraphRollback"]);
    let commit_epoch = session.commit().expect("commit current graph mutation");

    let reader = db.session();
    reader.use_graph_path(&lpg_path(&["alpha"])?)?;
    assert!(reader.get_node(discarded).is_none());
    assert!(reader.get_node(retained).is_some());
    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !has_label(event, "DiscardedAfterGraphSwitch"))
    );
    let history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (retained).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&["alpha"])?).clone()),
            since_epoch: EpochId::INITIAL,
        })
        .expect("retained graph event");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].epoch, commit_epoch);
    assert_ne!(history[0].epoch, EpochId::PENDING);
    Ok(())
}

#[test]
fn rollback_to_savepoint_retracks_the_surviving_schema_default_context()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session()
        .execute("CREATE SCHEMA qualified")
        .expect("create schema and its default graph");

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin schema context tx");
    session.savepoint("before_schema").expect("savepoint");
    session.set_schema("qualified")?;
    let discarded = session.create_node(&["DiscardedAfterSchemaSwitch"]);
    session
        .rollback_to_savepoint("before_schema")
        .expect("rollback schema mutation");
    assert_eq!(session.current_schema(), Some("qualified".to_string()));
    let retained = session.create_node(&["RetainedAfterSchemaRollback"]);
    let commit_epoch = session.commit().expect("commit schema-default mutation");

    let reader = db.session();
    reader.set_schema("qualified")?;
    reader.reset_graph()?;
    assert!(reader.get_node(discarded).is_none());
    assert!(reader.get_node(retained).is_some());
    assert!(
        all_changes(&db)
            .iter()
            .all(|event| !has_label(event, "DiscardedAfterSchemaSwitch"))
    );
    let history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (retained).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg(
                (lpg_path(&["qualified/__default__"])?).clone(),
            ),
            since_epoch: EpochId::INITIAL,
        })
        .expect("retained schema-default event");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].epoch, commit_epoch);
    assert_ne!(history[0].epoch, EpochId::PENDING);
    Ok(())
}

#[test]
fn concurrent_auto_commit_mutations_on_one_session_publish_matching_state_and_events()
-> Result<(), Box<dyn std::error::Error>> {
    const PER_WORKER: usize = 32;

    let db = cdc_db();
    let session = Arc::new(db.session());
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let mut workers = Vec::new();
    for label in ["ConcurrentA", "ConcurrentB"] {
        let session = Arc::clone(&session);
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            (0..PER_WORKER)
                .map(|_| session.create_node(&[label]))
                .collect::<Vec<_>>()
        }));
    }
    barrier.wait();

    let ids: Vec<_> = workers
        .into_iter()
        .flat_map(|worker| worker.join().expect("concurrent mutation worker"))
        .collect();
    assert_eq!(ids.len(), 2 * PER_WORKER);
    assert!(ids.iter().all(|id| id.is_valid()));
    let distinct: std::collections::HashSet<_> = ids.iter().copied().collect();
    assert_eq!(distinct.len(), ids.len());
    let mut commit_epochs = std::collections::HashSet::new();
    for id in ids {
        assert!(db.get_node(id).is_some(), "committed node must be visible");
        let history = db
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (id).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
                since_epoch: EpochId::INITIAL,
            })
            .expect("matching default-graph CDC history");
        assert_eq!(history.len(), 1, "each visible node has one create event");
        assert_eq!(history[0].kind, ChangeKind::Create);
        assert_ne!(history[0].epoch, EpochId::PENDING);
        assert!(
            commit_epochs.insert(history[0].epoch),
            "each auto-commit mutation must own a distinct transaction epoch"
        );
    }
    assert_eq!(commit_epochs.len(), 2 * PER_WORKER);
    Ok(())
}

#[test]
fn detach_cascade_events_keep_sorted_edges_before_the_node() {
    let db = cdc_db();
    let victim = db.create_node(&["CascadeVictim"]);
    let source = db.create_node(&["CascadeSource"]);
    let target = db.create_node(&["CascadeTarget"]);
    let edges = [
        db.create_edge(source, victim, "IN"),
        db.create_edge(victim, target, "OUT"),
        db.create_edge(victim, victim, "SELF"),
    ];
    assert!(edges.iter().all(|edge| edge.is_valid()));

    assert!(db.delete_node(victim));
    let delete_epoch = db.current_epoch();

    let deleted: Vec<_> = db
        .fixture_changes(delete_epoch..=delete_epoch)
        .expect("read cascade commit")
        .into_iter()
        .filter(|event| event.kind == ChangeKind::Delete)
        .map(|event| event.entity_id)
        .collect();

    let mut sorted_edges = edges;
    sorted_edges.sort_unstable();
    let mut expected: Vec<EntityId> = sorted_edges.into_iter().map(EntityId::Edge).collect();
    expected.push(EntityId::Node(victim));
    assert_eq!(
        deleted, expected,
        "cascade publication must retain staging order"
    );
}

#[test]
fn rollback_discards_and_savepoint_rollback_truncates_staged_creates() {
    let db = cdc_db();

    let rolled_back = {
        let mut session = db.session();
        session
            .begin_transaction()
            .expect("begin rollback transaction");
        let id = session.create_node(&["FullyRolledBack"]);
        assert!(id.is_valid());
        session.rollback().expect("rollback transaction");
        id
    };
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(rolled_back))
            .expect("rollback history")
            .is_empty()
    );

    let (retained, discarded, commit_epoch) = {
        let mut session = db.session();
        session
            .begin_transaction()
            .expect("begin savepoint transaction");
        let retained = session.create_node(&["BeforeSavepoint"]);
        session.savepoint("cdc_boundary").expect("create savepoint");
        let discarded = session.create_node(&["AfterSavepoint"]);
        session
            .rollback_to_savepoint("cdc_boundary")
            .expect("rollback to savepoint");
        let epoch = session.commit().expect("commit retained prefix");
        (retained, discarded, epoch)
    };

    let retained_history = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(retained))
        .expect("retained history");
    assert_eq!(retained_history.len(), 1);
    assert_eq!(retained_history[0].kind, ChangeKind::Create);
    assert_eq!(retained_history[0].epoch, commit_epoch);
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(discarded))
            .expect("discarded history")
            .is_empty()
    );
}

#[test]
fn serializable_private_graph_entities_do_not_conflict_with_unrelated_ids()
-> Result<(), Box<dyn std::error::Error>> {
    let db = cdc_db();
    db.session()
        .execute("CREATE GRAPH public_collision")
        .expect("create unrelated public graph");

    let mut public_writer = db.session();
    public_writer.use_graph_path(&lpg_path(&["public_collision"])?)?;
    public_writer
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin unrelated writer");
    public_writer
        .execute("INSERT (:PublicCollision)")
        .expect("hold the first graph-local node ID in an active transaction");

    let mut private_writer = db.session();
    private_writer
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin detached-graph writer");
    private_writer
        .execute("CREATE GRAPH private_collision")
        .expect("create detached transaction-private graph");
    private_writer.use_graph_path(&lpg_path(&["private_collision"])?)?;
    private_writer
        .execute("INSERT (:PrivateCollision)")
        .expect("equal graph-local IDs in unrelated graphs must not conflict");

    private_writer
        .commit()
        .expect("publish the complete private graph");
    public_writer.commit().expect("commit unrelated writer");

    for (graph, label) in [
        ("public_collision", "PublicCollision"),
        ("private_collision", "PrivateCollision"),
    ] {
        let reader = db.session();
        reader.use_graph_path(&lpg_path(&[graph])?)?;
        assert_eq!(
            reader
                .execute(&format!("MATCH (n:{label}) RETURN n"))
                .expect("read committed graph")
                .rows()
                .len(),
            1
        );
    }

    let private_events: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| {
            is_lpg_graph(event, &["private_collision"])
                && event.kind == ChangeKind::Create
                && has_label(event, "PrivateCollision")
        })
        .collect();
    assert_eq!(
        private_events.len(),
        1,
        "the successful private insert publishes exactly one create"
    );
    Ok(())
}

#[test]
fn serializable_conflict_publishes_no_loser_event() {
    let db = cdc_db();
    let account = db.create_node_with_props(&["ConflictAccount"], [("balance", Value::Int64(0))]);

    let mut winner = db.session();
    let mut loser = db.session();
    winner
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin winner");
    loser
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin loser");
    assert_eq!(
        winner.get_node_property(account, "balance"),
        Some(Value::Int64(0))
    );
    assert_eq!(
        loser.get_node_property(account, "balance"),
        Some(Value::Int64(0))
    );
    winner
        .set_node_property(account, "balance", Value::Int64(1))
        .expect("stage winner");
    loser
        .set_node_property(account, "balance", Value::Int64(2))
        .expect("stage loser");

    let winner_epoch = winner.commit().expect("winner commits");
    assert!(loser.commit().is_err(), "stale writer must conflict");

    let updates: Vec<_> = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(account))
        .expect("account history")
        .into_iter()
        .filter(|event| event.kind == ChangeKind::Update)
        .collect();
    assert_eq!(updates.len(), 1, "loser event must be discarded");
    assert_eq!(updates[0].epoch, winner_epoch);
    assert_eq!(
        updates[0]
            .after
            .as_ref()
            .and_then(|values| values.get("balance")),
        Some(&Value::Int64(1))
    );
}

#[cfg(feature = "testing-statement-injection")]
#[test]
fn injected_commit_failure_publishes_neither_state_nor_event() {
    use grafeo_common::testing::statement_failure::with_commit_failure;

    let db = cdc_db();
    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin injected transaction");
    let node = session.create_node(&["InjectedFailure"]);
    with_commit_failure(|| {
        assert!(session.commit().is_err(), "commit injection must fire");
    });

    assert!(db.get_node(node).is_none());
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(node))
            .expect("injected history")
            .is_empty()
    );
}

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]
#[test]
fn wal_mutation_failure_never_reaches_the_process_local_cdc_log()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::testing::wal_failure::{
        disable_mutation_log_failure, enable_mutation_log_failure_once,
    };
    use grafeo_engine::{DurabilityMode, GraphModel};

    let directory = tempfile::tempdir().expect("temporary WAL directory");
    let path = directory.path().join("cdc-wal-failure.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync)
            .with_cdc(),
    )
    .expect("open persistent CDC database");

    let session = db.session();
    enable_mutation_log_failure_once();
    let node = session.create_node(&["WalFailure"]);
    disable_mutation_log_failure();

    assert!(
        !node.is_valid(),
        "failed WAL mutation must not acknowledge an ID"
    );
    assert_eq!(
        db.memory_usage().cdc.event_count,
        0,
        "failed WAL mutation must emit zero process-local events"
    );
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(node))
            .is_err()
    );
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
            since_epoch: EpochId::INITIAL
        })
        .is_err()
    );
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::All,
            since_epoch: EpochId::INITIAL
        })
        .is_err(),
        "database CDC history must fail closed after poison"
    );
    assert!(
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
            entity_id: (node).into(),
            graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
            since_epoch: EpochId::INITIAL
        })
        .is_err(),
        "database graph-qualified CDC history must fail closed after poison"
    );
    assert!(
        db.fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
            .is_err(),
        "database CDC range reads must fail closed after poison"
    );
    assert!(
        session
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(node))
            .is_err()
    );
    assert!(
        session
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (node).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
                since_epoch: EpochId::INITIAL
            })
            .is_err()
    );
    assert!(
        session
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (node).into(),
                graph: grafeo_engine::cdc::HistoryGraph::All,
                since_epoch: EpochId::INITIAL
            })
            .is_err(),
        "session CDC history must fail closed after poison"
    );
    assert!(
        session
            .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery {
                entity_id: (node).into(),
                graph: grafeo_engine::cdc::HistoryGraph::Lpg((lpg_path(&[])?).clone()),
                since_epoch: EpochId::INITIAL
            })
            .is_err(),
        "session graph-qualified CDC history must fail closed after poison"
    );
    assert!(
        session
            .fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
            .is_err(),
        "session CDC range reads must fail closed after poison"
    );
    Ok(())
}

#[cfg(feature = "testing-statement-injection")]
#[test]
fn state_and_event_publication_share_one_observable_cut() {
    let db = Arc::new(cdc_db());
    let pause = db
        .testing_pause_cdc_before_publication()
        .expect("install database-local CDC publication pause");
    let (staged_tx, staged_rx) = mpsc::sync_channel::<NodeId>(0);

    let writer_db = Arc::clone(&db);
    let writer = std::thread::spawn(move || {
        let mut session = writer_db.session();
        session.begin_transaction().expect("begin writer");
        let node = session.create_node(&["AtomicCut"]);
        staged_tx.send(node).expect("report staged node");
        session.commit().expect("commit writer")
    });

    let node = staged_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("writer staged node");
    assert!(
        pause.wait_until_reached(Duration::from_secs(5)),
        "writer must stop after state finalization and before CDC insertion"
    );
    assert!(
        db.testing_publication_write_locked(),
        "the exact pause cut must retain the mixed-model publication write barrier"
    );

    let (state_tx, state_rx) = mpsc::sync_channel(0);
    let state_db = Arc::clone(&db);
    let state_reader = std::thread::spawn(move || {
        state_tx
            .send(state_db.get_node(node).is_some())
            .expect("report state cut");
    });

    let (cdc_tx, cdc_rx) = mpsc::sync_channel(0);
    let cdc_db = Arc::clone(&db);
    let cdc_reader = std::thread::spawn(move || {
        cdc_tx
            .send(
                cdc_db
                    .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(node))
                    .expect("read CDC cut"),
            )
            .expect("report CDC cut");
    });

    pause.release();
    let commit_epoch = writer.join().expect("writer thread");
    assert!(
        state_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("state reader completes after release")
    );
    let history = cdc_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("CDC reader completes after release");
    state_reader.join().expect("state reader thread");
    cdc_reader.join().expect("CDC reader thread");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].kind, ChangeKind::Create);
    assert_eq!(history[0].epoch, commit_epoch);
}

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-statement-injection"
))]
#[test]
fn cdc_preparation_failure_aborts_before_the_durable_commit_marker()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::DurabilityMode;
    use grafeo_storage::wal::{WalRecord, WalRecovery};

    let directory = tempfile::tempdir()?;
    let file = directory.path().join("prepare.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&file)
            .with_cdc()
            .with_wal_durability(DurabilityMode::Sync),
    )?;
    let mut session = db.session();
    let existing = session.create_node_with_props(&["Existing"], [("value", Value::Int64(1))])?;
    let before = serde_json::to_value(all_changes(&db))?;
    session.begin_transaction()?;
    let transaction = session
        .active_transaction_id()
        .ok_or("missing transaction")?;
    session.set_node_property(existing, "value", Value::Int64(2))?;
    session.create_node_with_props(&["Aborted"], [("value", Value::Int64(3))])?;
    db.testing_fail_next_cdc_preparation();
    assert!(session.commit().is_err());
    assert!(session.active_transaction_id().is_none());
    // Aborted preparation may consume a reserved epoch; state, events and the
    // durable marker below determine whether anything was published.
    assert_eq!(
        session.get_node_property(existing, "value"),
        Some(Value::Int64(1))
    );
    assert_eq!(serde_json::to_value(all_changes(&db))?, before);
    assert_eq!(
        session.execute("MATCH (n:Aborted) RETURN n")?.rows().len(),
        0
    );

    // Snapshot the Sync-acknowledged WAL before close can checkpoint it.
    let source = file.with_extension("grafeo.wal");
    let copy = directory.path().join("copied.wal");
    std::fs::create_dir(&copy)?;
    for entry in std::fs::read_dir(&source)? {
        let entry = entry?;
        assert!(entry.file_type()?.is_file());
        std::fs::copy(entry.path(), copy.join(entry.file_name()))?;
    }
    let records = WalRecovery::new(&copy)?.recover()?;
    assert!(!records.iter().any(|record| matches!(record,
        WalRecord::Committed { transaction_id, .. } if *transaction_id == transaction)));
    session.set_node_property(existing, "value", Value::Int64(4))?;
    assert_eq!(
        all_changes(&db).len(),
        before.as_array().ok_or("history array")?.len() + 1
    );
    db.close()?;
    Ok(())
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
