//! `set_auto_commit(false)` must open an implicit transaction on the first
//! mutation; it must never degrade into an unframed SYSTEM write.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn persistent(path: &std::path::Path, model: GraphModel) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(model)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent database")
}

#[test]
fn disabled_auto_commit_frames_direct_lpg_crud_until_explicit_commit() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("direct.grafeo");
    let node_id;
    {
        let db = persistent(&path, GraphModel::Lpg);
        let mut writer = db.session();
        writer.set_auto_commit(false);

        node_id = writer.create_node(&["Pending"]);
        assert!(node_id.is_valid());
        assert!(
            writer.in_transaction(),
            "the first mutation must start an implicit transaction"
        );
        assert!(writer.get_node(node_id).is_some(), "writer gets RYW");
        assert!(
            db.session().get_node(node_id).is_none(),
            "an uncommitted direct mutation must stay private"
        );

        writer.commit().unwrap();
        assert!(db.session().get_node(node_id).is_some());
        db.close().unwrap();
    }

    let reopened = persistent(&path, GraphModel::Lpg);
    assert!(
        reopened.get_node(node_id).is_some(),
        "the explicit commit marker must make the write recoverable"
    );
}

#[test]
fn disabled_auto_commit_frames_gql_and_rollback_discards_it() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let mut writer = db.session();
    writer.set_auto_commit(false);

    writer.execute("INSERT (:Pending {value: 1})").unwrap();
    assert!(writer.in_transaction());
    let outsider = db
        .session()
        .execute("MATCH (n:Pending) RETURN count(n)")
        .unwrap();
    assert_eq!(outsider.rows()[0][0].as_int64(), Some(0));

    writer.rollback().unwrap();
    let after = db
        .session()
        .execute("MATCH (n:Pending) RETURN count(n)")
        .unwrap();
    assert_eq!(after.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn disabled_auto_commit_frames_rdf_until_explicit_commit() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf.grafeo");
    {
        let db = persistent(&path, GraphModel::Rdf);
        let mut writer = db.session();
        writer.set_auto_commit(false);
        writer
            .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "pending" . }"#)
            .unwrap();

        assert!(writer.in_transaction());
        assert_eq!(
            writer
                .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
                .unwrap()
                .row_count(),
            1,
            "writer gets RDF RYW"
        );
        assert_eq!(
            db.session()
                .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
                .unwrap()
                .row_count(),
            0,
            "another session must not see the implicit transaction"
        );

        writer.commit().unwrap();
        db.close().unwrap();
    }

    let reopened = persistent(&path, GraphModel::Rdf);
    assert_eq!(
        reopened
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1
    );
}
