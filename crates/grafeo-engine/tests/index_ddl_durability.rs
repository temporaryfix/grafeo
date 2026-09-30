//! Transactional and crash-durable LPG index DDL.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "vector-index",
    feature = "text-index",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel, Session};

fn persistent(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent LPG database")
}

fn sidecar(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(".wal");
    PathBuf::from(value)
}

fn copy_live_database(src: &Path, dst: &Path) {
    std::fs::copy(src, dst).expect("copy live container");
    let src_wal = sidecar(src);
    if src_wal.exists() {
        let dst_wal = sidecar(dst);
        std::fs::create_dir_all(&dst_wal).expect("create copied WAL directory");
        for entry in std::fs::read_dir(src_wal).expect("read live WAL directory") {
            let entry = entry.expect("WAL directory entry");
            if entry.path().is_file() {
                std::fs::copy(entry.path(), dst_wal.join(entry.file_name()))
                    .expect("copy WAL segment");
            }
        }
    }
}

fn index_names(db: &GrafeoDB) -> Vec<String> {
    index_names_for_session(&db.session())
}

fn index_names_for_session(session: &Session) -> Vec<String> {
    let mut names: Vec<_> = session
        .execute("SHOW INDEXES")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| row[0].as_str().map(ToString::to_string))
        .collect();
    names.sort();
    names
}

#[test]
fn query_and_direct_indexes_survive_wal_only_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.grafeo");
    let copy = dir.path().join("crash-copy.grafeo");
    let db = persistent(&path);
    let person = db.create_node(&["Person"]);
    db.set_node_property(person, "name", Value::from("Alix"))
        .expect("set node property");
    db.set_node_property(person, "email", Value::from("a@example.com"))
        .expect("set node property");
    let doc = db.create_node(&["Doc"]);
    db.set_node_property(doc, "qemb", Value::Vector(Arc::from([1.0_f32, 0.0, 0.0])))
        .expect("set node property");
    db.set_node_property(doc, "demb", Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])))
        .expect("set node property");
    db.set_node_property(doc, "qbody", Value::from("transactional graph index"))
        .expect("set node property");
    db.set_node_property(doc, "dbody", Value::from("durable direct index"))
        .expect("set node property");
    db.wal_checkpoint().unwrap();

    let session = db.session();
    session
        .execute("CREATE INDEX idx_name FOR (n:Person) ON (n.name)")
        .unwrap();
    session
        .execute(
            "CREATE INDEX idx_qvec FOR (n:Doc) ON (n.qemb) USING VECTOR {dimensions: 3, metric: 'cosine'}",
        )
        .unwrap();
    session
        .execute("CREATE INDEX idx_qtext FOR (n:Doc) ON (n.qbody) USING TEXT")
        .unwrap();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "email".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "demb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("euclidean".into()),
            m: Some(8),
            ef_construction: Some(32),
            ef: None,
            quantization: Some("scalar".into()),
        },
    })
    .unwrap();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "dbody".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .unwrap();
    // Native index creation also has canonical catalog owners. Recovery must
    // preserve all six names, including the three generated native names.
    let expected_index_names = index_names(&db);
    assert_eq!(expected_index_names.len(), 6);
    for name in ["idx_name", "idx_qtext", "idx_qvec"] {
        assert!(expected_index_names.iter().any(|actual| actual == name));
    }
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(recovered.has_property_index("name"));
    assert!(recovered.has_property_index("email"));
    assert_eq!(index_names(&recovered), expected_index_names);
    assert!(
        recovered
            .vector_search("Doc", "qemb", &[1.0, 0.0, 0.0], 1, None, None)
            .is_ok()
    );
    assert!(
        recovered
            .vector_search("Doc", "demb", &[0.0, 1.0, 0.0], 1, None, None)
            .is_ok()
    );
    assert!(recovered.text_search("Doc", "qbody", "graph", 1).is_ok());
    assert!(recovered.text_search("Doc", "dbody", "direct", 1).is_ok());
}

#[test]
fn create_and_drop_obey_transaction_and_savepoint_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transactions.grafeo");
    let copy = dir.path().join("transactions-crash-copy.grafeo");
    let db = persistent(&path);
    db.wal_checkpoint().unwrap();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("CREATE INDEX idx_name FOR (n:Person) ON (n.name)")
        .unwrap();
    assert!(
        !db.has_property_index("name"),
        "DDL stays private until commit"
    );
    assert!(index_names(&db).is_empty(), "other sessions cannot see DDL");
    assert_eq!(index_names_for_session(&session), vec!["idx_name"]);
    session.rollback().unwrap();
    assert!(!db.has_property_index("name"));
    assert!(index_names(&db).is_empty());

    session.begin_transaction().unwrap();
    session.savepoint("before_index").unwrap();
    session
        .execute("CREATE INDEX idx_email FOR (n:Person) ON (n.email)")
        .unwrap();
    session.rollback_to_savepoint("before_index").unwrap();
    session.commit().unwrap();
    assert!(!db.has_property_index("email"));

    session
        .execute("CREATE INDEX idx_keep FOR (n:Person) ON (n.keep)")
        .unwrap();
    session.begin_transaction().unwrap();
    session.execute("DROP INDEX idx_keep").unwrap();
    assert!(
        db.has_property_index("keep"),
        "drop stays private until commit"
    );
    session.rollback().unwrap();
    assert!(db.has_property_index("keep"));
    assert_eq!(index_names(&db), vec!["idx_keep"]);

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(!recovered.has_property_index("name"));
    assert!(!recovered.has_property_index("email"));
    assert!(recovered.has_property_index("keep"));
    assert_eq!(index_names(&recovered), vec!["idx_keep"]);
}

#[test]
fn drop_index_then_graph_replays_as_one_final_post_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drop-graph-source.grafeo");
    let copy = dir.path().join("drop-graph-crash-copy.grafeo");
    let db = persistent(&path);
    db.wal_checkpoint().unwrap();

    let mut session = db.session();
    session.execute("CREATE GRAPH indexed_graph").unwrap();
    session.execute("SESSION SET GRAPH indexed_graph").unwrap();
    session
        .execute("CREATE INDEX idx_graph FOR (n:Person) ON (n.name)")
        .unwrap();

    session.begin_transaction().unwrap();
    session.execute("DROP INDEX idx_graph").unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();
    session.execute("DROP GRAPH indexed_graph").unwrap();
    session.commit().unwrap();

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(
        !recovered
            .list_graphs()
            .iter()
            .any(|graph| graph == "indexed_graph")
    );
    assert!(
        !index_names(&recovered)
            .iter()
            .any(|name| name == "idx_graph")
    );
}

#[test]
fn drop_recreate_drop_cascade_has_the_same_live_and_recovered_post_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drop-replacement-source.grafeo");
    let copy = dir.path().join("drop-replacement-crash-copy.grafeo");
    let db = persistent(&path);
    db.wal_checkpoint().unwrap();

    let mut session = db.session();
    session.execute("CREATE GRAPH replaceable").unwrap();
    session.execute("SESSION SET GRAPH replaceable").unwrap();
    session
        .execute("CREATE INDEX idx_original FOR (n:Person) ON (n.name)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();

    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH replaceable").unwrap();
    session.execute("CREATE GRAPH replaceable").unwrap();
    session.execute("SESSION SET GRAPH replaceable").unwrap();
    session
        .execute("CREATE INDEX idx_replacement FOR (n:Person) ON (n.email)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();
    session.execute("DROP GRAPH replaceable").unwrap();
    session
        .commit()
        .expect("cancelling the replacement must retain the original graph's index cascade");

    assert!(!db.list_graphs().iter().any(|graph| graph == "replaceable"));
    assert!(index_names(&db).is_empty());

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(
        !recovered
            .list_graphs()
            .iter()
            .any(|graph| graph == "replaceable")
    );
    assert!(index_names(&recovered).is_empty());
}

#[test]
fn recreated_graph_incarnation_does_not_replay_obsolete_vector_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recreated-graph-source.grafeo");
    let copy = dir.path().join("recreated-graph-crash-copy.grafeo");
    let db = persistent(&path);
    db.wal_checkpoint().unwrap();

    let session = db.session();
    session.execute("CREATE GRAPH indexed_graph").unwrap();
    session.execute("SESSION SET GRAPH indexed_graph").unwrap();
    session
        .execute("CREATE INDEX idx_graph FOR (n:Doc) ON (n.embedding) USING VECTOR {dimensions: 3}")
        .unwrap();

    session.execute("DROP INDEX idx_graph").unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();
    session.execute("DROP GRAPH indexed_graph").unwrap();
    session.execute("CREATE GRAPH indexed_graph").unwrap();
    session.execute("SESSION SET GRAPH indexed_graph").unwrap();
    session
        .execute("CREATE (:Doc {embedding: [1.0, 0.0, 0.0, 0.0]})")
        .unwrap();

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(
        recovered
            .list_graphs()
            .iter()
            .any(|graph| graph == "indexed_graph")
    );
    assert!(
        !index_names(&recovered)
            .iter()
            .any(|name| name == "idx_graph")
    );

    let recovered_session = recovered.session();
    recovered_session
        .execute("SESSION SET GRAPH indexed_graph")
        .unwrap();
    let rows = recovered_session
        .execute("MATCH (d:Doc) RETURN d.embedding")
        .unwrap();
    assert_eq!(rows.rows().len(), 1);
}

#[test]
fn drop_index_removes_the_correct_physical_registry() {
    let db = GrafeoDB::new_in_memory();
    let doc = db.create_node(&["Doc"]);
    db.set_node_property(doc, "emb", Value::Vector(Arc::from([1.0_f32, 0.0, 0.0])))
        .expect("set node property");
    db.set_node_property(doc, "body", Value::from("graph database"))
        .expect("set node property");
    let session = db.session();
    session
        .execute("CREATE INDEX vec_idx FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 3}")
        .unwrap();
    session
        .execute("CREATE INDEX text_idx FOR (n:Doc) ON (n.body) USING TEXT")
        .unwrap();
    session.execute("DROP INDEX vec_idx").unwrap();
    session.execute("DROP INDEX text_idx").unwrap();

    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .is_err()
    );
    assert!(db.text_search("Doc", "body", "graph", 1).is_err());
    assert!(index_names(&db).is_empty());
}

#[test]
fn logical_index_names_cannot_alias_one_physical_target() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE INDEX idx_name FOR (n:Person) ON (n.name)")
        .expect("create physical property index owner");

    let same_kind = session.execute("CREATE INDEX idx_alias FOR (n:Person) ON (n.name)");
    assert!(
        same_kind.is_err(),
        "a physical index cannot gain a second name"
    );

    // Property/B-tree declarations and labels collapse to the same current
    // property-wide registry, so both are aliases even though their logical
    // declarations differ.
    let collapsed_family =
        session.execute("CREATE INDEX idx_btree FOR (n:Other) ON (n.name) USING BTREE");
    assert!(
        collapsed_family.is_err(),
        "property and B-tree declarations must not alias one registry"
    );
    assert_eq!(index_names(&db), vec!["idx_name"]);
    assert!(db.has_property_index("name"));

    session
        .execute("DROP INDEX idx_name")
        .expect("drop sole owner");
    assert!(!db.has_property_index("name"));
    session
        .execute("CREATE INDEX idx_alias FOR (n:Person) ON (n.name)")
        .expect("target may be reused after its owner is dropped");
    assert_eq!(index_names(&db), vec!["idx_alias"]);
}

#[test]
fn catalog_names_survive_clean_checkpoint_and_wal_drop_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.grafeo");
    {
        let db = persistent(&path);
        db.session()
            .execute("CREATE INDEX idx_name FOR (n:Person) ON (n.name)")
            .unwrap();
        db.close().unwrap();
    }
    {
        let db = persistent(&path);
        assert_eq!(index_names(&db), vec!["idx_name"]);
        db.wal_checkpoint().unwrap();
        db.session().execute("DROP INDEX idx_name").unwrap();
        let copy = dir.path().join("drop-copy.grafeo");
        db.wal().unwrap().sync().unwrap();
        copy_live_database(&path, &copy);
        std::mem::forget(db);

        let recovered = persistent(&copy);
        assert!(!recovered.has_property_index("name"));
        assert!(index_names(&recovered).is_empty());
    }
}

/// Index preparation must validate the transaction's final row, not the
/// previously committed value that the transaction is replacing.
#[test]
fn mixed_ddl_vector_index_uses_valid_final_dimensions() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let id = session.create_node_with_props(
        &["Doc"],
        [("emb", Value::Vector(Arc::from([1.0_f32, 0.0, 0.0])))],
    )?;
    session.begin_transaction()?;
    session.execute("MATCH (n:Doc) SET n.emb = [0.0, 1.0]")?;
    session.execute(
        "CREATE INDEX idx_final FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}",
    )?;
    session.commit()?;

    assert_eq!(index_names(&db), vec!["idx_final"]);
    assert_eq!(
        db.vector_search("Doc", "emb", &[0.0, 1.0], 10, None, None)?,
        vec![(id, 0.0)],
    );
    Ok(())
}

/// A final dimension mismatch is an ordinary pre-publication validation
/// failure, never a post-commit index assertion or a partly installed index.
#[test]
fn mixed_ddl_vector_index_rejects_invalid_final_dimensions()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::utils::error::{Error, QueryErrorKind};

    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let original = Value::Vector(Arc::from([1.0_f32, 0.0]));
    let id = session.create_node_with_props(&["Doc"], [("emb", original.clone())])?;
    session.begin_transaction()?;
    session.execute("MATCH (n:Doc) SET n.emb = [0.0, 1.0, 0.0]")?;
    session.execute(
        "CREATE INDEX idx_invalid FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}",
    )?;
    let outcome = session.commit();
    let Err(error) = outcome else {
        return Err("a final 3D value was admitted to a 2D index".into());
    };
    assert!(matches!(
        error,
        Error::Query(ref details) if details.kind == QueryErrorKind::Semantic
            && details.message.contains("dimension mismatch")
    ));

    let node = db
        .get_node(id)
        .ok_or("validation failure removed the committed node")?;
    assert_eq!(node.get_property("emb"), Some(&original));
    assert_eq!(db.session().execute("SHOW INDEXES")?.row_count(), 0);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .is_none()
    );
    assert_eq!(
        session.execute("MATCH (n:Doc) RETURN n.emb")?.rows(),
        vec![vec![original]]
    );
    Ok(())
}

#[test]
fn mixed_ddl_vector_index_excludes_foreign_pending_create() -> Result<(), Box<dyn std::error::Error>>
{
    let db = GrafeoDB::new_in_memory();
    let valid = db.session().create_node_with_props(
        &["Doc"],
        [("emb", Value::Vector(Arc::from([1.0_f32, 0.0])))],
    )?;
    let mut foreign = db.session();
    foreign.begin_transaction()?;
    let pending = foreign.create_node_with_props(
        &["Doc"],
        [("emb", Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])))],
    )?;
    db.session().execute(
        "CREATE INDEX idx_committed FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}",
    )?;
    assert_eq!(
        db.vector_search("Doc", "emb", &[1.0, 0.0], 10, None, None)?,
        vec![(valid, 0.0)],
    );
    foreign.rollback()?;
    assert!(db.get_node(pending).is_none());
    assert_eq!(db.session().execute("SHOW INDEXES")?.row_count(), 1);
    assert_eq!(
        db.vector_search("Doc", "emb", &[1.0, 0.0], 10, None, None)?,
        vec![(valid, 0.0)],
    );
    Ok(())
}

/// A row leaving the final index membership cannot be sent through that
/// index's dimension check during intermediate overlay application.
#[test]
fn mixed_ddl_vector_index_uses_final_label_membership() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let id = session.create_node_with_props(
        &["Doc"],
        [("emb", Value::Vector(Arc::from([1.0_f32, 0.0])))],
    )?;
    session.begin_transaction()?;
    session.execute("MATCH (n:Doc) SET n.emb = [0.0, 1.0, 0.0]")?;
    session.execute("MATCH (n:Doc) REMOVE n:Doc")?;
    session.execute(
        "CREATE INDEX idx_final_label FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}",
    )?;
    session.commit()?;
    let node = db.get_node(id).ok_or("label removal deleted the node")?;
    assert!(!node.has_label("Doc"));
    assert_eq!(
        node.get_property("emb"),
        Some(&Value::from(vec![0.0_f64, 1.0, 0.0]))
    );
    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0], 10, None, None)?
            .is_empty()
    );
    assert_eq!(db.session().execute("SHOW INDEXES")?.row_count(), 1);
    Ok(())
}

#[test]
fn mixed_ddl_vector_index_excludes_final_node_delete() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let id = session.create_node_with_props(
        &["Doc"],
        [("emb", Value::Vector(Arc::from([1.0_f32, 0.0])))],
    )?;
    session.begin_transaction()?;
    session.execute("MATCH (n:Doc) SET n.emb = [0.0, 1.0, 0.0]")?;
    session.execute("MATCH (n:Doc) DELETE n")?;
    session.execute(
        "CREATE INDEX idx_final_delete FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}",
    )?;
    session.commit()?;
    assert!(db.get_node(id).is_none());
    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0], 10, None, None)?
            .is_empty()
    );
    assert_eq!(db.session().execute("SHOW INDEXES")?.row_count(), 1);
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn mixed_ddl_vector_index_retains_foreign_pending_cold_delete()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::utils::error::{Error, QueryErrorKind};

    let mut db = GrafeoDB::new_in_memory();
    let original = Value::Vector(Arc::from([1.0_f32, 0.0, 0.0]));
    let id = db
        .session()
        .create_node_with_props(&["Doc"], [("emb", original.clone())])?;
    db.compact()?;
    let mut foreign = db.session();
    foreign.begin_transaction()?;
    foreign.execute("MATCH (n:Doc) DELETE n")?;
    let result = db.session().execute(
        "CREATE INDEX idx_foreign_delete FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}",
    );
    assert!(matches!(
        result,
        Err(Error::Query(ref details)) if details.kind == QueryErrorKind::Semantic
            && details.message.contains("dimension mismatch")
    ));
    foreign.rollback()?;
    let node = db
        .get_node(id)
        .ok_or("foreign rollback lost the cold row")?;
    assert_eq!(node.get_property("emb"), Some(&original));
    assert_eq!(db.session().execute("SHOW INDEXES")?.row_count(), 0);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .is_none()
    );
    Ok(())
}
