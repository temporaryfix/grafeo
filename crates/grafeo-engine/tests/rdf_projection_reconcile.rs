//! RDF→LPG projection reconciliation and metadata persistence regressions.

#![cfg(all(feature = "triple-store", feature = "sparql", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_core::graph::rdf::{Term, Triple, ValidTimeInterval};
#[cfg(feature = "wal")]
use grafeo_engine::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PERSON: &str = "http://ex.org/Person";

fn both_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap()
}

#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    std::path::PathBuf::from(sidecar)
}

#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn copy_tree(source: &std::path::Path, target: &std::path::Path) {
    std::fs::create_dir_all(target).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let destination = target.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &destination);
        } else {
            std::fs::copy(entry.path(), destination).unwrap();
        }
    }
}

#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn copy_live_database(source: &std::path::Path, target: &std::path::Path) {
    std::fs::copy(source, target).unwrap();
    copy_tree(&sidecar_wal_dir(source), &sidecar_wal_dir(target));
}

fn projected_iris(db: &GrafeoDB) -> Vec<String> {
    db.session()
        .execute("MATCH (n:Person) RETURN n.iri ORDER BY n.iri")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(value) => Some(value.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn rebuild_is_idempotent_and_removes_rows_deleted_at_the_source() {
    let db = both_db();
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/alix> a <http://ex.org/Person> .
            <http://ex.org/gus> a <http://ex.org/Person> .
        }"#,
    )
    .unwrap();

    let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 2);
    assert_eq!(db.node_count(), 2);
    assert_eq!(
        projected_iris(&db),
        vec!["http://ex.org/alix", "http://ex.org/gus"]
    );
    let first = db.rdf_lpg_projection(id).unwrap();
    assert_eq!(first.generation(), 1);
    assert_eq!(first.row_count(), 2);
    assert_eq!(db.rdf_projection_lag(id), Some(0));

    // A full rebuild of the same source cut reuses both owned rows.
    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 2);
    assert_eq!(db.node_count(), 2, "repeat rebuild must not duplicate rows");
    assert_eq!(db.rdf_lpg_projection(id).unwrap().generation(), 2);

    db.execute_sparql(r#"DELETE DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    assert!(
        db.rdf_projection_lag(id).is_some_and(|lag| lag > 0),
        "source commit must make the published projection observably stale"
    );

    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(db.node_count(), 1, "stale owned row must be removed");
    assert_eq!(projected_iris(&db), vec!["http://ex.org/gus"]);
    let third = db.rdf_lpg_projection(id).unwrap();
    assert_eq!(third.generation(), 3);
    assert_eq!(third.row_count(), 1);
    assert_eq!(db.rdf_projection_lag(id), Some(0));
}

#[test]
fn concurrent_rebuilds_are_serialized_without_duplicate_rows_or_generations() {
    let db = std::sync::Arc::new(both_db());
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/alix> a <http://ex.org/Person> .
            <http://ex.org/gus> a <http://ex.org/Person> .
        }"#,
    )
    .unwrap();
    let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    let start = std::sync::Arc::new(std::sync::Barrier::new(3));

    let workers: Vec<_> = (0..2)
        .map(|_| {
            let db = std::sync::Arc::clone(&db);
            let start = std::sync::Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                db.rebuild_rdf_lpg_projection(id)
            })
        })
        .collect();
    start.wait();
    for worker in workers {
        assert_eq!(
            worker.join().expect("projection worker panicked").unwrap(),
            2
        );
    }

    assert_eq!(db.node_count(), 2);
    assert_eq!(
        projected_iris(&db),
        vec!["http://ex.org/alix", "http://ex.org/gus"]
    );
    let published = db.rdf_lpg_projection(id).unwrap();
    assert_eq!(published.generation(), 2);
    assert_eq!(published.row_count(), 2);
    assert_eq!(published.receipt().unwrap().generation(), 2);
}

#[test]
fn logical_memory_fork_keeps_mapping_but_drops_foreign_rows_and_receipt() {
    let source = both_db();
    source
        .session()
        .execute("INSERT (:Ordinary {name: 'kept'})")
        .unwrap();
    source
        .execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let id = source.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(source.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(source.node_count(), 2);

    let fork = source.to_memory().unwrap();
    assert_ne!(fork.rdf_store().store_id(), source.rdf_store().store_id());
    let pending = fork
        .rdf_lpg_projection(id)
        .expect("logical mapping survives");
    assert_eq!(pending.generation(), 0);
    assert_eq!(pending.last_source_epoch(), None);
    assert_eq!(pending.last_target_epoch(), None);
    assert_eq!(pending.row_count(), 0);
    assert_eq!(pending.receipt(), None);
    assert_eq!(fork.node_count(), 1, "foreign projection rows are omitted");
    assert_eq!(
        fork.session()
            .execute("MATCH (n:Ordinary) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );

    assert_eq!(fork.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    let published = fork.rdf_lpg_projection(id).unwrap();
    assert_eq!(published.generation(), 1);
    assert_eq!(
        published.receipt().unwrap().store_id(),
        fork.rdf_store().store_id()
    );
    assert_eq!(fork.node_count(), 2);
}

#[test]
fn named_source_receipt_pins_graph_incarnation_and_recreate_isolated() {
    const SOURCE: &str = "http://ex.org/claims";
    let db = both_db();
    db.execute_sparql(&format!("CREATE GRAPH <{SOURCE}>"))
        .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <http://ex.org/default-only> a <{PERSON}> .
            GRAPH <{SOURCE}> {{ <http://ex.org/named-old> a <{PERSON}> . }}
        }}"#
    ))
    .unwrap();

    let id = db
        .declare_named_rdf_lpg_projection(SOURCE, PERSON, "Person")
        .unwrap();
    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(projected_iris(&db), vec!["http://ex.org/named-old"]);
    let first = db.rdf_lpg_projection(id).unwrap();
    let first_receipt = first.receipt().expect("V3 rebuild publishes a receipt");
    assert_eq!(first_receipt.store_id(), db.rdf_store().store_id());
    assert_eq!(first_receipt.source_graph().name(), Some(SOURCE));
    assert!(first_receipt.source_graph().incarnation().as_u64() > 0);
    assert!(first_receipt.target_epoch() > first_receipt.source_epoch());
    let first_incarnation = first_receipt.source_graph().incarnation();

    db.execute_sparql(&format!("DROP GRAPH <{SOURCE}>"))
        .unwrap();
    assert_eq!(db.rdf_projection_lag(id), Some(1));
    db.execute_sparql(&format!("CREATE GRAPH <{SOURCE}>"))
        .unwrap();
    db.execute_sparql(&format!(
        "INSERT DATA {{ GRAPH <{SOURCE}> {{ <http://ex.org/named-new> a <{PERSON}> . }} }}"
    ))
    .unwrap();

    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(projected_iris(&db), vec!["http://ex.org/named-new"]);
    let second = db.rdf_lpg_projection(id).unwrap();
    let second_receipt = second.receipt().expect("second V3 receipt");
    assert_eq!(second_receipt.generation(), 2);
    assert_ne!(
        second_receipt.source_graph().incarnation(),
        first_incarnation
    );
    assert_eq!(db.rdf_projection_lag(id), Some(0));
}

#[cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "testing-crash-injection"
))]
#[test]
fn receipt_append_failure_rolls_back_rows_and_registry_generation() {
    use grafeo_common::testing::wal_failure::enable_projection_receipt_log_failure_once;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("projection-receipt-failure.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();

    enable_projection_receipt_log_failure_once();
    let error = db.rebuild_rdf_lpg_projection(id).unwrap_err();
    assert!(error.to_string().contains("projection receipt"));
    let status = db.rdf_lpg_projection(id).unwrap();
    assert_eq!(status.generation(), 0);
    assert_eq!(status.receipt(), None);
    assert_eq!(db.node_count(), 0);

    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(db.rdf_lpg_projection(id).unwrap().generation(), 1);
    assert_eq!(db.node_count(), 1);
    db.close().unwrap();
}

#[cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "testing-crash-injection"
))]
#[test]
fn commit_marker_failure_leaves_logged_receipt_and_rows_unpublished() {
    use grafeo_common::testing::wal_failure::{
        disable_commit_log_failure, enable_commit_log_failure_once,
    };

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("projection-commit-failure.grafeo");
    {
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Both)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
            .unwrap();
        let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();

        enable_commit_log_failure_once();
        let error = db.rebuild_rdf_lpg_projection(id).unwrap_err();
        disable_commit_log_failure();
        assert!(error.to_string().contains("outcome is unknown"), "{error}");
        assert!(db.is_durability_poisoned());
        assert_eq!(db.node_count(), 0);
        let pending = db.rdf_lpg_projection(id).unwrap();
        assert_eq!(pending.generation(), 0);
        assert_eq!(pending.receipt(), None);
    }

    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let id = grafeo_core::graph::rdf::RdfLpgProjectionDefinition::new(PERSON, "Person").id();
    let pending = reopened.rdf_lpg_projection(id).unwrap();
    assert_eq!(pending.generation(), 0);
    assert_eq!(pending.receipt(), None);
    assert_eq!(reopened.node_count(), 0);
    assert_eq!(reopened.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    reopened.close().unwrap();
}

#[cfg(feature = "grafeo-file")]
#[test]
fn projection_definition_status_and_owned_rows_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf-projection.grafeo");

    let id = {
        let db =
            GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Both))
                .unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
            .unwrap();
        let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
        assert_eq!(db.rdf_lpg_projection(id).unwrap().generation(), 1);
        db.close().unwrap();
        id
    };

    let db = GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Both))
        .unwrap();
    let restored = db
        .rdf_lpg_projection(id)
        .expect("projection definition/status restored");
    assert_eq!(restored.type_iri(), PERSON);
    assert_eq!(restored.node_label(), "Person");
    assert_eq!(restored.generation(), 1);
    assert_eq!(restored.row_count(), 1);
    assert_eq!(db.rdf_projection_lag(id), Some(0));
    assert_eq!(db.node_count(), 1);

    // Reconciliation after reopen finds the persisted owned row rather than
    // creating a second generation beside it.
    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.rdf_lpg_projection(id).unwrap().generation(), 2);
    db.close().unwrap();
}

#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn force_synced_projection_metadata_survives_a_live_file_crash_copy() {
    let dir = tempfile::TempDir::new().unwrap();
    let live = dir.path().join("live.grafeo");
    let crash_copy = dir.path().join("crash-copy.grafeo");
    let config = || {
        Config::persistent(&live)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync)
    };

    let db = GrafeoDB::with_config(config()).unwrap();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);

    // Deliberately do not close, checkpoint, or manually sync. Declaration and
    // published-status records themselves must be force-synced in Sync mode.
    copy_live_database(&live, &crash_copy);

    let recovered = GrafeoDB::with_config(
        Config::persistent(&crash_copy)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let status = recovered
        .rdf_lpg_projection(id)
        .expect("force-synced declaration must replay");
    assert_eq!(status.generation(), 1);
    assert_eq!(status.row_count(), 1);
    assert_eq!(recovered.rdf_projection_lag(id), Some(0));
    assert_eq!(recovered.node_count(), 1);
    assert_eq!(recovered.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(
        recovered.node_count(),
        1,
        "replay/rebuild must stay idempotent"
    );

    recovered.close().unwrap();
    db.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_projection_definition_and_status() {
    let source = both_db();
    source
        .execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let id = source.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    source.rebuild_rdf_lpg_projection(id).unwrap();
    let expected = source.rdf_lpg_projection(id).unwrap();
    let expected_lag = source.rdf_projection_lag(id);
    let expected_rows = projected_iris(&source);
    assert_eq!(expected.generation(), 1);
    assert_eq!(expected.row_count(), 1);
    assert!(expected.last_source_epoch().is_some());
    assert_eq!(expected_lag, Some(0));
    let expected_snapshot = source.export_snapshot().unwrap();
    let expected_cut = source.world_cut().unwrap();
    let target_epoch = expected.last_target_epoch().unwrap();

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal-copy");
    source.save(&path).unwrap();
    assert!(path.is_file());
    assert_eq!(source.export_snapshot().unwrap(), expected_snapshot);
    assert_eq!(source.world_cut().unwrap(), expected_cut);
    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let status = reopened
        .rdf_lpg_projection(id)
        .expect("saved declaration must reopen");
    assert_eq!(status.generation(), 1);
    assert_eq!(status.row_count(), 1);
    assert_eq!(status.last_source_epoch(), expected.last_source_epoch());
    assert_eq!(status.last_target_epoch(), expected.last_target_epoch());
    assert_eq!(reopened.export_snapshot().unwrap(), expected_snapshot);
    assert_eq!(reopened.world_cut().unwrap(), expected_cut);
    let before_target = grafeo_common::types::EpochId::new(target_epoch.as_u64() - 1);
    let owned = grafeo_engine::database::testing::root_lpg_store(&reopened).node_ids();
    assert_eq!(owned.len(), 1);
    for node in owned {
        assert!(reopened.get_node_at_epoch(node, before_target).is_none());
        assert!(reopened.get_node_at_epoch(node, target_epoch).is_some());
    }
    assert_eq!(reopened.rdf_projection_lag(id), expected_lag);
    assert_eq!(projected_iris(&reopened), expected_rows);
    assert_eq!(reopened.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(reopened.node_count(), 1);
    let rebuilt = reopened.rdf_lpg_projection(id).unwrap();
    assert_eq!(rebuilt.generation(), 2);
    assert_eq!(rebuilt.row_count(), 1);
    assert_eq!(reopened.rdf_projection_lag(id), Some(0));
    reopened.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_keeps_a_stale_projection_stale_and_owned() {
    let source = both_db();
    source
        .execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> a <http://ex.org/Person> .
                <http://ex.org/gus> a <http://ex.org/Person> .
            }"#,
        )
        .unwrap();
    let id = source.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(source.rebuild_rdf_lpg_projection(id).unwrap(), 2);
    source
        .execute_sparql(r#"DELETE DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();

    let expected = source.rdf_lpg_projection(id).unwrap();
    let expected_lag = source.rdf_projection_lag(id).unwrap();
    assert_eq!(expected.generation(), 1);
    assert_eq!(expected.row_count(), 2);
    assert!(expected_lag > 0);
    assert_eq!(
        projected_iris(&source),
        vec!["http://ex.org/alix", "http://ex.org/gus"]
    );

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("stale-wal-copy");
    source.save(&path).unwrap();

    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let status = reopened.rdf_lpg_projection(id).unwrap();
    assert_eq!(status.generation(), expected.generation());
    assert_eq!(status.row_count(), expected.row_count());
    assert_eq!(status.last_source_epoch(), expected.last_source_epoch());
    assert_eq!(status.last_target_epoch(), expected.last_target_epoch());
    assert_eq!(
        reopened.export_snapshot().unwrap(),
        source.export_snapshot().unwrap()
    );
    assert_eq!(reopened.world_cut().unwrap(), source.world_cut().unwrap());
    assert_eq!(reopened.rdf_projection_lag(id), Some(expected_lag));
    assert_eq!(
        projected_iris(&reopened),
        vec!["http://ex.org/alix", "http://ex.org/gus"],
        "a stale status must retain the last published owned generation"
    );

    assert_eq!(reopened.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(projected_iris(&reopened), vec!["http://ex.org/gus"]);
    let rebuilt = reopened.rdf_lpg_projection(id).unwrap();
    assert_eq!(rebuilt.generation(), 2);
    assert_eq!(rebuilt.row_count(), 1);
    assert_eq!(reopened.rdf_projection_lag(id), Some(0));
    reopened.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_rdf_tx_valid_history_and_named_graph_identity() {
    let source = both_db();
    let history = Triple::new(
        Term::iri("http://ex.org/history"),
        Term::iri("http://ex.org/p"),
        Term::literal("old"),
    );
    let valid = Triple::new(
        Term::iri("http://ex.org/valid"),
        Term::iri("http://ex.org/p"),
        Term::literal("bounded"),
    );
    let named = Triple::new(
        Term::iri("http://ex.org/named-subject"),
        Term::iri("http://ex.org/p"),
        Term::literal("named"),
    );

    source
        .execute_sparql(r#"INSERT DATA { <http://ex.org/history> <http://ex.org/p> "old" . }"#)
        .unwrap();
    let history_insert_epoch = source.rdf_store_commit_epoch();
    source
        .execute_sparql(r#"DELETE DATA { <http://ex.org/history> <http://ex.org/p> "old" . }"#)
        .unwrap();
    let history_delete_epoch = source.rdf_store_commit_epoch();
    source
        .insert_rdf_valid([valid.clone()], 1_000, 2_000)
        .unwrap();
    let valid_insert_epoch = source.rdf_store_commit_epoch();
    source
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/named> {
                <http://ex.org/named-subject> <http://ex.org/p> "named" .
            } }"#,
        )
        .unwrap();
    let named_insert_epoch = source.rdf_store_commit_epoch();
    let expected_snapshot = source.export_snapshot().unwrap();

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("history-wal-copy");
    source.save(&path).unwrap();
    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();

    assert_eq!(reopened.export_snapshot().unwrap(), expected_snapshot);
    assert_eq!(reopened.world_cut().unwrap(), source.world_cut().unwrap());

    let history_lives = reopened
        .rdf_store()
        .quad_history()
        .into_iter()
        .find_map(|(candidate, lives)| (candidate.as_ref() == &history).then_some(lives))
        .expect("closed default-graph history survives");
    assert_eq!(history_lives.len(), 1);
    assert_eq!(history_lives[0].tx.from(), history_insert_epoch);
    assert_eq!(history_lives[0].tx.to(), history_delete_epoch);
    assert_eq!(history_lives[0].valid, None);

    let valid_lives = reopened
        .rdf_store()
        .quad_history()
        .into_iter()
        .find_map(|(candidate, lives)| (candidate.as_ref() == &valid).then_some(lives))
        .expect("valid-time history survives");
    assert_eq!(valid_lives.len(), 1);
    assert_eq!(valid_lives[0].tx.from(), valid_insert_epoch);
    assert!(valid_lives[0].tx.is_open());
    assert_eq!(
        valid_lives[0].valid,
        Some(ValidTimeInterval::from_legacy_micros(1_000, 2_000).unwrap())
    );

    let named_graph = reopened
        .rdf_store()
        .graph("http://ex.org/named")
        .expect("named RDF graph survives");
    let named_lives = named_graph
        .quad_history()
        .into_iter()
        .find_map(|(candidate, lives)| (candidate.as_ref() == &named).then_some(lives))
        .expect("named-graph interval survives in its graph");
    assert_eq!(named_lives.len(), 1);
    assert_eq!(named_lives[0].tx.from(), named_insert_epoch);
    assert!(named_lives[0].tx.is_open());
    assert_eq!(named_lives[0].valid, None);
    reopened.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_refuses_to_mix_with_an_existing_directory_destination() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("existing");
    {
        let existing =
            GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Both))
                .unwrap();
        existing
            .session()
            .execute("INSERT (:Existing {name: 'keep-me'})")
            .unwrap();
        existing.close().unwrap();
    }

    let source = both_db();
    source
        .session()
        .execute("INSERT (:Source {name: 'must-not-mix'})")
        .unwrap();
    let error = source.save(&path).unwrap_err().to_string();
    assert!(error.contains("existing destination"), "{error}");

    let reopened =
        GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Both))
            .unwrap();
    assert_eq!(
        reopened
            .session()
            .execute("MATCH (n:Existing) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );
    assert_eq!(
        reopened
            .session()
            .execute("MATCH (n:Source) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(0)
    );
    reopened.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn exact_save_preserves_schema_and_named_index_semantics() {
    let source = both_db();
    let session = source.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL DEFAULT 'unknown')")
        .unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute("CREATE INDEX idx_person_name FOR (n:Person) ON (n.name)")
        .unwrap();
    drop(session);

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema-index-copy");
    source.save(&path).unwrap();
    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    assert!(reopened.has_property_index("name"));
    assert_eq!(
        reopened.export_snapshot().unwrap(),
        source.export_snapshot().unwrap()
    );
    assert_eq!(reopened.world_cut().unwrap(), source.world_cut().unwrap());
    let session = reopened.session();
    let node_types = session.execute("SHOW NODE TYPES").unwrap();
    assert!(
        node_types
            .rows()
            .iter()
            .any(|row| { matches!(&row[0], Value::String(name) if name.as_str() == "Person") })
    );
    let indexes = session.execute("SHOW INDEXES").unwrap();
    assert!(indexes.rows().iter().any(|row| {
        matches!(&row[0], Value::String(name) if name.as_str() == "idx_person_name")
    }));
    session.execute("DROP INDEX idx_person_name").unwrap();
    assert!(!reopened.has_property_index("name"));
    drop(session);
    reopened.close().unwrap();
}

#[test]
fn projection_declaration_rejects_empty_mapping_components() {
    let db = both_db();
    let empty_iri = db
        .declare_rdf_lpg_projection("", "Person")
        .unwrap_err()
        .to_string();
    assert!(empty_iri.contains("projection"), "{empty_iri}");
    let blank_label = db
        .declare_rdf_lpg_projection(PERSON, "   ")
        .unwrap_err()
        .to_string();
    assert!(blank_label.contains("projection"), "{blank_label}");
}

#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn projection_rebuild_is_rejected_on_a_read_only_database() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("projection-read-only.grafeo");
    let id = {
        let db =
            GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Both))
                .unwrap();
        let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
        db.close().unwrap();
        id
    };

    let read_only = GrafeoDB::open_read_only(&path).unwrap();
    let error = read_only
        .rebuild_rdf_lpg_projection(id)
        .unwrap_err()
        .to_string();
    assert!(error.to_lowercase().contains("read-only"), "{error}");
    read_only.close().unwrap();
}

#[test]
fn portable_snapshot_round_trips_projection_status() {
    let db = both_db();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    db.rebuild_rdf_lpg_projection(id).unwrap();

    let reopened = GrafeoDB::import_snapshot(&db.export_snapshot().unwrap()).unwrap();
    assert_eq!(reopened.rdf_lpg_projection(id).unwrap().generation(), 1);
    assert_eq!(reopened.rdf_projection_lag(id), Some(0));
    assert_eq!(reopened.rebuild_rdf_lpg_projection(id).unwrap(), 1);
    assert_eq!(reopened.node_count(), 1);
}
