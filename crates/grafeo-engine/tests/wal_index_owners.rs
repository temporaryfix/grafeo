//! Canonical Property/BTree owners must survive the actual uncheckpointed WAL tail.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{GraphPath, IndexId, Value};
use grafeo_core::graph::{PropertyIndexPredicate, PropertyIndexRequest};
use grafeo_engine::{
    Config, CreateIndexRequest, DurabilityMode, GrafeoDB, GraphModel, IndexCreateKind,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn persistent(path: &Path) -> TestResult<GrafeoDB> {
    Ok(GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )?)
}

fn property(graph: GraphPath, name: Option<&str>, key: &str) -> CreateIndexRequest {
    CreateIndexRequest {
        graph,
        name: name.map(str::to_owned),
        label: None,
        property: key.into(),
        kind: IndexCreateKind::Property,
    }
}

fn sidecar(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(".wal");
    PathBuf::from(path)
}

fn show(db: &GrafeoDB) -> TestResult<Vec<Vec<Value>>> {
    Ok(db.session().execute("SHOW INDEXES")?.rows().to_vec())
}

fn row(name: &str, kind: &str, label: &str, property: &str) -> Vec<Value> {
    [name, kind, label, property]
        .into_iter()
        .map(Value::from)
        .collect()
}

fn recover_tail(db: GrafeoDB, source: &Path, copy: &Path, baseline: &[u8]) -> TestResult<GrafeoDB> {
    db.wal().ok_or("fixture needs a real WAL")?.sync()?;
    assert_eq!(
        std::fs::read(source)?,
        baseline,
        "the copied container must still precede every tested owner mutation"
    );
    std::fs::copy(source, copy)?;
    let destination = sidecar(copy);
    std::fs::create_dir_all(&destination)?;
    let mut segments = 0;
    for entry in std::fs::read_dir(sidecar(source))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
            segments += 1;
        }
    }
    assert!(
        segments > 0,
        "recovery must consume an actual copied WAL tail"
    );
    // The captured tail is independent now. Cleanly retire the real writer;
    // never leak a database or background worker to simulate a crash.
    db.close()?;
    drop(db);
    persistent(copy)
}

fn property_and_btree_tail(compact: bool) -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("owners.grafeo");
    let copy = temp.path().join("owners-tail.grafeo");
    let db = persistent(&source)?;
    assert!(db.create_graph("named")?);
    let root_node = db.session().create_node_with_props(
        &["Person"],
        [
            ("value", Value::from("before")),
            ("score", Value::from(7_i64)),
        ],
    )?;
    let deleted_node = db.session().create_node_with_props(
        &["Person"],
        [
            ("value", Value::from("before")),
            ("score", Value::from(8_i64)),
        ],
    )?;
    let named = db.session();
    named.use_graph_path(&GraphPath::from_components(&["named"])?)?;
    let named_node =
        named.create_node_with_props(&["Person"], [("value", Value::from("before"))])?;
    drop(named);
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    #[cfg(feature = "compact-store")]
    let db = {
        let mut db = db;
        if compact {
            db.compact()?;
        }
        db
    };
    #[cfg(not(feature = "compact-store"))]
    if compact {
        return Err("compact fixture requires compact-store".into());
    }

    let hash = db.create_index(property(GraphPath::root(), None, "value"))?;
    let mut ordered = property(GraphPath::root(), Some("ordered"), "score");
    ordered.kind = IndexCreateKind::BTree;
    let btree = db.create_index(ordered)?;
    assert_eq!(btree.as_u32(), hash.as_u32() + 1);
    let retained_epoch = db.current_epoch();
    let named = db.session();
    named.use_graph_path(&GraphPath::from_components(&["named"])?)?;
    named.execute("CREATE INDEX named_label FOR (n:Person) ON (n.value)")?;
    named.set_node_property(named_node, "value", Value::from("after"))?;
    drop(named);
    db.set_node_property(root_node, "value", Value::from("after"))?;
    db.set_node_property(root_node, "score", Value::from(9_i64))?;
    assert!(db.session().delete_node(deleted_node));
    let owners = show(&db)?;
    assert_eq!(
        owners,
        vec![
            row(
                &format!("@grafeo-index:{}", hash.as_u32()),
                "Hash",
                "",
                "value"
            ),
            row("named_label", "Hash", "Person", "value"),
            row("ordered", "BTree", "", "score"),
        ]
    );
    db.rebuild_index(btree)?;
    assert_eq!(show(&db)?, owners, "rebuild preserves the resolved owner");

    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(show(&recovered)?, owners);
    assert!(recovered.has_property_index("value"));
    assert!(recovered.has_property_index("score"));
    assert!(
        recovered
            .find_nodes_by_property("value", &Value::from("before"))
            .is_empty()
    );
    assert_eq!(
        recovered.find_nodes_by_property("value", &Value::from("after")),
        vec![root_node]
    );
    assert!(
        recovered
            .find_nodes_by_property("score", &Value::from(7_i64))
            .is_empty()
    );
    assert_eq!(
        recovered.find_nodes_by_property("score", &Value::from(9_i64)),
        vec![root_node]
    );
    let named = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .graph("named")
        .ok_or("missing named graph")?;
    assert!(named.has_property_index("value"));
    assert!(
        named
            .find_nodes_by_property("value", &Value::from("before"))
            .is_empty()
    );
    assert_eq!(
        named.find_nodes_by_property("value", &Value::from("after")),
        vec![named_node]
    );

    let current_epoch = recovered.current_epoch();
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered).retained_history_floor()
            <= retained_epoch
    );
    let old_score = Value::Int64(7);
    let old_score_eq = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .lookup_nodes_indexed(PropertyIndexRequest {
            property: "score",
            predicate: PropertyIndexPredicate::Equal(&old_score),
            epoch: retained_epoch,
            transaction_id: None,
        })?;
    assert_eq!(old_score_eq, Some(vec![root_node]));
    let deleted_score = Value::Int64(8);
    let old_scores = [old_score.clone(), deleted_score.clone()];
    let old_score_in = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .lookup_nodes_indexed(PropertyIndexRequest {
            property: "score",
            predicate: PropertyIndexPredicate::In(&old_scores),
            epoch: retained_epoch,
            transaction_id: None,
        })?;
    assert_eq!(old_score_in, Some(vec![root_node, deleted_node]));
    let old_score_range = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .lookup_nodes_indexed(PropertyIndexRequest {
            property: "score",
            predicate: PropertyIndexPredicate::Range {
                min: Some(&old_score),
                max: Some(&deleted_score),
                min_inclusive: true,
                max_inclusive: true,
            },
            epoch: retained_epoch,
            transaction_id: None,
        })?;
    assert_eq!(old_score_range, Some(vec![root_node, deleted_node]));

    let current_score = Value::Int64(9);
    let current_score_eq = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .lookup_nodes_indexed(PropertyIndexRequest {
            property: "score",
            predicate: PropertyIndexPredicate::Equal(&current_score),
            epoch: current_epoch,
            transaction_id: None,
        })?;
    assert_eq!(current_score_eq, Some(vec![root_node]));
    let current_scores = [current_score.clone(), deleted_score.clone()];
    let current_score_in = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .lookup_nodes_indexed(PropertyIndexRequest {
            property: "score",
            predicate: PropertyIndexPredicate::In(&current_scores),
            epoch: current_epoch,
            transaction_id: None,
        })?;
    assert_eq!(current_score_in, Some(vec![root_node]));
    let current_score_range = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .lookup_nodes_indexed(PropertyIndexRequest {
            property: "score",
            predicate: PropertyIndexPredicate::Range {
                min: Some(&current_score),
                max: Some(&current_score),
                min_inclusive: true,
                max_inclusive: true,
            },
            epoch: current_epoch,
            transaction_id: None,
        })?;
    assert_eq!(current_score_range, Some(vec![root_node]));
    recovered.rebuild_index(btree)?;
    assert_eq!(show(&recovered)?, owners);
    assert!(
        recovered.drop_index(hash)?,
        "the exact returned ID survives replay"
    );
    assert!(!recovered.has_property_index("value"));
    assert!(
        named.has_property_index("value"),
        "root drop must not hit named owner"
    );
    let next = recovered.create_index(property(GraphPath::root(), Some("next"), "next"))?;
    assert_eq!(
        next.as_u32(),
        btree.as_u32() + 2,
        "named creation consumes one ID; rebuild consumes none"
    );
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_owner_ids_names_labels_configuration_and_property_postimages_survive() -> TestResult {
    property_and_btree_tail(false)
}

#[cfg(feature = "compact-store")]
#[test]
fn wal_property_owners_survive_cold_row_edits() -> TestResult {
    property_and_btree_tail(true)
}

#[test]
fn wal_dropped_owner_floor_survives_without_aborted_or_savepoint_owners() -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("floor.grafeo");
    let copy = temp.path().join("floor-tail.grafeo");
    let db = persistent(&source)?;
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    let mut session = db.session();
    session.begin_transaction()?;
    session.execute("CREATE INDEX aborted FOR (n:Person) ON (n.aborted)")?;
    session.rollback()?;
    session.begin_transaction()?;
    session.savepoint("before_ddl")?;
    session.execute("CREATE INDEX rolled_back FOR (n:Person) ON (n.rolled_back)")?;
    session.rollback_to_savepoint("before_ddl")?;
    session.commit()?;
    drop(session);
    assert!(show(&db)?.is_empty());
    let mut request = property(GraphPath::root(), Some("removed"), "value");
    request.kind = IndexCreateKind::BTree;
    let removed = db.create_index(request)?;
    assert_eq!(
        removed,
        IndexId::new(0),
        "uncommitted DDL must not consume an ID"
    );
    db.rebuild_index(removed)?;
    assert!(db.drop_index(removed)?);
    assert!(show(&db)?.is_empty());

    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert!(show(&recovered)?.is_empty());
    for key in ["aborted", "rolled_back", "value"] {
        assert!(!recovered.has_property_index(key), "unexpected index {key}");
    }
    assert!(!recovered.drop_index(removed)?);
    assert!(recovered.rebuild_index(removed).is_err());
    let next = recovered.create_index(property(GraphPath::root(), None, "value"))?;
    assert_eq!(
        next.as_u32(),
        removed.as_u32() + 1,
        "an empty catalog retains its allocator floor"
    );
    assert_eq!(
        show(&recovered)?,
        vec![row(
            &format!("@grafeo-index:{}", next.as_u32()),
            "Hash",
            "",
            "value"
        )]
    );
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_drop_then_copy_same_graph_name_keeps_the_replacement_owner() -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("lifecycle.grafeo");
    let copy = temp.path().join("lifecycle-tail.grafeo");
    let db = persistent(&source)?;
    for (graph, value) in [("source", "copied"), ("replaceable", "old")] {
        assert!(db.create_graph(graph)?);
        let session = db.session();
        session.use_graph_path(&GraphPath::from_components(&[graph])?)?;
        session.create_node_with_props(&["Person"], [("value", Value::from(value))])?;
    }
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    let source_owner = db.create_index(property(
        GraphPath::from_components(&["source"])?,
        Some("source_owner"),
        "value",
    ))?;
    let old_owner = db.create_index(property(
        GraphPath::from_components(&["replaceable"])?,
        Some("old_owner"),
        "value",
    ))?;
    let mut session = db.session();
    session.begin_transaction()?;
    session.execute("DROP GRAPH replaceable")?;
    session.execute("CREATE GRAPH replaceable AS COPY OF source")?;
    session.commit()?;
    drop(session);
    let owners = show(&db)?;
    assert_eq!(owners.len(), 2);
    assert!(
        owners
            .iter()
            .all(|row| row[0].as_str() != Some("old_owner"))
    );
    assert!(
        !db.drop_index(old_owner)?,
        "obsolete ID must not remove replacement"
    );

    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(show(&recovered)?, owners);
    let replacement = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .graph("replaceable")
        .ok_or("replacement graph disappeared")?;
    assert_eq!(replacement.node_count(), 1);
    assert!(replacement.has_property_index("value"));
    assert_eq!(
        replacement
            .find_nodes_by_property("value", &Value::from("copied"))
            .len(),
        1
    );
    assert!(
        replacement
            .find_nodes_by_property("value", &Value::from("old"))
            .is_empty()
    );
    assert!(!recovered.drop_index(old_owner)?);
    assert!(replacement.has_property_index("value"));
    assert!(recovered.drop_index(source_owner)?);
    assert!(
        replacement.has_property_index("value"),
        "COPY has its own owner"
    );
    let copied_owner = IndexId::new(old_owner.as_u32() + 1);
    assert!(
        recovered.drop_index(copied_owner)?,
        "COPY owner identity survives exactly"
    );
    assert!(!replacement.has_property_index("value"));
    recovered.close()?;
    Ok(())
}

#[cfg(feature = "vector-index")]
#[test]
fn wal_vector_owners_allocate_once_and_reject_duplicate_requests() -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("guards.grafeo");
    let copy = temp.path().join("guards-tail.grafeo");
    let db = persistent(&source)?;
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    let first = db.create_index(property(GraphPath::root(), Some("first"), "first"))?;
    let request = CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("vector_owner".into()),
        label: Some("Doc".into()),
        property: "embedding".into(),
        kind: IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: Some(8),
            ef_construction: Some(32),
            ef: None,
            quantization: Some("scalar".into()),
        },
    };
    let vector = db.create_index(request.clone())?;
    assert_eq!(vector.as_u32(), first.as_u32() + 1);
    let owners = show(&db)?;
    let epoch = db.current_epoch();
    let sequence = db.wal().ok_or("missing WAL")?.record_count();
    assert!(db.create_index(request).is_err());
    assert!(db.session().execute("CREATE INDEX duplicate_vector FOR (n:Doc) ON (n.embedding) USING VECTOR {dimensions: 3}").is_err());
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!(show(&db)?, owners);
    // Both rejected auto-transactions append only their abort records.
    assert_eq!(db.wal().ok_or("missing WAL")?.record_count(), sequence + 2);
    db.rebuild_index(vector)?;
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "embedding")
            .is_some()
    );
    let next = db.create_index(property(GraphPath::root(), Some("next"), "next"))?;
    assert_eq!(next.as_u32(), vector.as_u32() + 1);
    let owners = show(&db)?;
    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(show(&recovered)?, owners);
    assert!(recovered.drop_index(first)?);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_vector_index("Doc", "embedding")
            .is_some()
    );
    assert!(recovered.drop_index(vector)?);
    assert!(!recovered.drop_index(vector)?);
    assert!(recovered.rebuild_index(vector).is_err());
    assert!(recovered.drop_index(next)?);
    recovered.close()?;
    Ok(())
}
