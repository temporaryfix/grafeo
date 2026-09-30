//! Exact RDF delete framing and graph-incarnation recovery.

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
#![allow(missing_docs)]

use std::path::{Path, PathBuf};

use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::wal::{WalRecord, WalRecovery};

const PREDICATE: &str = "http://ex.org/p";
const SOURCE: &str = "http://ex.org/source";
const TARGET: &str = "http://ex.org/target";

fn persistent_rdf(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Rdf)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent RDF database")
}

fn sidecar_wal_dir(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(".wal");
    PathBuf::from(path)
}

fn copy_live_database(source: &Path, target: &Path) {
    std::fs::copy(source, target).expect("copy live container");
    let source_wal = sidecar_wal_dir(source);
    if !source_wal.exists() {
        return;
    }
    let target_wal = sidecar_wal_dir(target);
    std::fs::create_dir_all(&target_wal).expect("create copied WAL directory");
    for entry in std::fs::read_dir(source_wal).expect("read live WAL directory") {
        let entry = entry.expect("WAL directory entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), target_wal.join(entry.file_name()))
                .expect("copy live WAL segment");
        }
    }
}

fn recovered_records(path: &Path) -> Vec<WalRecord> {
    // Inspect a frozen fixture copy; live source ownership stays with the database.
    let temporary = tempfile::tempdir().unwrap();
    let inspection = temporary.path().join("wal");
    std::fs::create_dir(&inspection).unwrap();
    for entry in std::fs::read_dir(sidecar_wal_dir(path)).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inspection.join(entry.file_name())).unwrap();
    }
    WalRecovery::new(&inspection)
        .unwrap()
        .recover()
        .expect("recover committed WAL records")
}

fn triple(subject: &str, object: Term) -> Triple {
    Triple::new(Term::iri(subject), Term::iri(PREDICATE), object)
}

#[test]
fn explain_graph_mutations_without_transaction_preserves_graphs_and_wal() {
    fn wal_image(path: &Path) -> std::collections::BTreeMap<std::ffi::OsString, Vec<u8>> {
        std::fs::read_dir(path)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), std::fs::read(entry.path()).unwrap())
            })
            .collect()
    }

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("explain-graph-operations.grafeo");
    let db = persistent_rdf(&path);
    db.execute_sparql(
        r#"INSERT DATA {
            <urn:default> <urn:p> "default" .
            GRAPH <urn:source> { <urn:source> <urn:p> "source" . }
            GRAPH <urn:target> { <urn:target> <urn:p> "target" . }
        }"#,
    )
    .unwrap();
    db.execute_sparql("CREATE GRAPH <urn:empty>").unwrap();
    db.wal().unwrap().sync().unwrap();
    let before_wal = wal_image(&sidecar_wal_dir(&path));
    let before_epoch = db.current_epoch();
    let before = db.rdf_dataset_history().unwrap();
    let before_cut = db.rdf_history_cut(before_epoch).unwrap();
    let session = db.session();
    assert!(!session.in_transaction());

    for statement in [
        "CREATE GRAPH <urn:missing>",
        "CREATE SILENT GRAPH <urn:source>",
        "DROP DEFAULT",
        "DROP NAMED",
        "DROP ALL",
        "DROP GRAPH <urn:source>",
        "DROP SILENT GRAPH <urn:missing>",
        "COPY GRAPH <urn:empty> TO GRAPH <urn:missing>",
        "MOVE GRAPH <urn:source> TO GRAPH <urn:target>",
        "ADD GRAPH <urn:source> TO DEFAULT",
    ] {
        let result = session
            .execute_sparql(&format!("EXPLAIN {statement}"))
            .unwrap();
        assert!(!result.is_empty(), "{statement}");
        assert!(!session.in_transaction(), "{statement}");
        let after = db.rdf_dataset_history().unwrap();
        assert_eq!(db.current_epoch(), before_epoch);
        assert_eq!(after.store_id(), before.store_id());
        assert_eq!(after.completeness(), before.completeness());
        assert_eq!(
            after.next_graph_incarnation(),
            before.next_graph_incarnation()
        );
        assert_eq!(after.graph_lives(), before.graph_lives());
        assert_eq!(after.quad_versions(), before.quad_versions());
        assert_eq!(db.rdf_history_cut(before_epoch).unwrap(), before_cut);
        db.wal().unwrap().sync().unwrap();
        assert_eq!(
            wal_image(&sidecar_wal_dir(&path)),
            before_wal,
            "{statement}"
        );
    }
}

#[test]
fn delete_paths_are_exact_and_bound_terms_are_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("delete-paths.grafeo");
    let db = persistent_rdf(&path);

    db.execute_sparql(
        r#"DELETE DATA {
            GRAPH <http://ex.org/absent> {
                <http://ex.org/never> <http://ex.org/p> "never" .
            }
        }"#,
    )
    .unwrap();
    assert!(
        db.rdf_store().graph("http://ex.org/absent").is_none(),
        "DELETE against an absent named graph must not create a graph lifetime"
    );

    db.execute_sparql(
        r#"INSERT DATA {
            GRAPH <http://ex.org/source> {
                <did:example:subject> <http://ex.org/p> <tag:example:object> .
                <http://ex.org/date> <http://ex.org/p> "2026-08-28"^^<http://www.w3.org/2001/XMLSchema#date> .
                <http://ex.org/lang> <http://ex.org/p> "bonjour"@fr .
            }
        }"#,
    )
    .unwrap();

    db.execute_sparql(
        r#"INSERT { GRAPH <http://ex.org/target> { ?s <http://ex.org/p> ?o } }
           WHERE  { GRAPH <http://ex.org/source> { ?s <http://ex.org/p> ?o } }"#,
    )
    .unwrap();
    let target = db.rdf_store().graph(TARGET).expect("target graph");
    assert!(target.contains(&triple(
        "did:example:subject",
        Term::iri("tag:example:object")
    )));
    assert!(target.contains(&triple(
        "http://ex.org/date",
        Term::typed_literal("2026-08-28", "http://www.w3.org/2001/XMLSchema#date")
    )));
    assert!(target.contains(&triple(
        "http://ex.org/lang",
        Term::lang_literal("bonjour", "fr")
    )));
    let target_incarnation = target.graph_incarnation();

    db.execute_sparql(
        r#"DELETE WHERE { GRAPH <http://ex.org/target> { ?s <http://ex.org/p> ?o } }"#,
    )
    .unwrap();
    assert!(db.rdf_store().graph(TARGET).unwrap().is_empty());

    db.execute_sparql(
        r#"INSERT { GRAPH <http://ex.org/target> { ?s <http://ex.org/p> ?o } }
           WHERE  { GRAPH <http://ex.org/source> { ?s <http://ex.org/p> ?o } }"#,
    )
    .unwrap();
    db.execute_sparql(
        r#"DELETE { GRAPH <http://ex.org/target> { ?s <http://ex.org/p> ?o } }
           WHERE { GRAPH <http://ex.org/target> { ?s <http://ex.org/p> ?o } }"#,
    )
    .unwrap();
    assert!(db.rdf_store().graph(TARGET).unwrap().is_empty());

    db.execute_sparql(
        r#"INSERT DATA {
            GRAPH <http://ex.org/target> {
                <http://ex.org/direct> <http://ex.org/p> "direct" .
            }
        }"#,
    )
    .unwrap();
    db.execute_sparql(
        r#"DELETE DATA {
            GRAPH <http://ex.org/target> {
                <http://ex.org/direct> <http://ex.org/p> "direct" .
            }
        }"#,
    )
    .unwrap();

    db.wal().unwrap().sync().unwrap();
    let records = recovered_records(&path);
    let mut exact_deletes: Vec<_> = records
        .iter()
        .filter_map(|record| match record {
            WalRecord::DeleteRdfQuadV3 {
                subject,
                predicate,
                object,
                graph,
                graph_incarnation,
                ..
            } => Some((
                subject.as_str(),
                predicate.as_str(),
                object.as_str(),
                graph.as_deref(),
                *graph_incarnation,
            )),
            _ => None,
        })
        .collect();
    let mut expected_deletes: Vec<_> = [
        ("<did:example:subject>", "<tag:example:object>"),
        (
            "<http://ex.org/date>",
            "\"2026-08-28\"^^<http://www.w3.org/2001/XMLSchema#date>",
        ),
        ("<http://ex.org/lang>", "\"bonjour\"@fr"),
        ("<did:example:subject>", "<tag:example:object>"),
        (
            "<http://ex.org/date>",
            "\"2026-08-28\"^^<http://www.w3.org/2001/XMLSchema#date>",
        ),
        ("<http://ex.org/lang>", "\"bonjour\"@fr"),
        ("<http://ex.org/direct>", "\"direct\""),
    ]
    .into_iter()
    .map(|(subject, object)| {
        (
            subject,
            "<http://ex.org/p>",
            object,
            Some(TARGET),
            target_incarnation,
        )
    })
    .collect();
    exact_deletes.sort_unstable();
    expected_deletes.sort_unstable();
    assert_eq!(exact_deletes, expected_deletes);
}

#[test]
fn graph_variable_delete_where_replays_each_exact_named_graph_incarnation() {
    const FIRST: &str = "http://ex.org/variable-first";
    const SECOND: &str = "http://ex.org/variable-second";

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph-variable-delete.grafeo");
    let crash_copy = dir.path().join("graph-variable-delete-crash.grafeo");
    let db = persistent_rdf(&path);
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            GRAPH <{FIRST}> {{ <http://ex.org/a> <{PREDICATE}> "one" . }}
            GRAPH <{SECOND}> {{ <http://ex.org/b> <{PREDICATE}> "two" . }}
        }}"#
    ))
    .unwrap();
    let insert_epoch = db.rdf_store().commit_epoch();
    let first_incarnation = db
        .rdf_store()
        .graph(FIRST)
        .expect("first graph")
        .graph_incarnation();
    let second_incarnation = db
        .rdf_store()
        .graph(SECOND)
        .expect("second graph")
        .graph_incarnation();

    db.execute_sparql(&format!(
        r#"DELETE WHERE {{ GRAPH ?graph {{ ?subject <{PREDICATE}> ?object }} }}"#
    ))
    .unwrap();
    let delete_epoch = db.rdf_store().commit_epoch();
    assert!(db.rdf_store().graph(FIRST).unwrap().is_empty());
    assert!(db.rdf_store().graph(SECOND).unwrap().is_empty());

    db.wal().unwrap().sync().unwrap();
    let mut exact_deletes: Vec<_> = recovered_records(&path)
        .into_iter()
        .filter_map(|record| match record {
            WalRecord::DeleteRdfQuadV3 {
                graph: Some(graph),
                graph_incarnation,
                ..
            } if graph == FIRST || graph == SECOND => Some((graph, graph_incarnation)),
            _ => None,
        })
        .collect();
    exact_deletes.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        exact_deletes,
        vec![
            (FIRST.to_string(), first_incarnation),
            (SECOND.to_string(), second_incarnation),
        ]
    );

    copy_live_database(&path, &crash_copy);
    std::mem::forget(db);
    let recovered = persistent_rdf(&crash_copy);
    assert!(recovered.rdf_store().graph(FIRST).unwrap().is_empty());
    assert!(recovered.rdf_store().graph(SECOND).unwrap().is_empty());
    let diff = recovered
        .rdf_history_diff(insert_epoch, delete_epoch)
        .expect("recovered durable DELETE WHERE history");
    assert_eq!(diff.transitions.len(), 2);
    assert!(diff.transitions.iter().all(|transition| matches!(
        transition.kind,
        grafeo_engine::RdfHistoryTransitionKind::StatementRetracted { .. }
    )));
}

#[test]
fn modify_duplicate_insert_writes_one_exact_frame_and_replays_one_quad() {
    const DEDUP_GRAPH: &str = "http://ex.org/deduplicated-insert";

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deduplicated-insert.grafeo");
    let crash_copy = dir.path().join("deduplicated-insert-crash.grafeo");
    let db = persistent_rdf(&path);
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/source-a> <http://ex.org/matches> "one" .
            <http://ex.org/source-b> <http://ex.org/matches> "two" .
        }"#,
    )
    .unwrap();

    let before_insert = db.rdf_store().commit_epoch();
    let update = format!(
        r#"INSERT {{ GRAPH <{DEDUP_GRAPH}> {{
               <http://ex.org/result> <http://ex.org/p> "value"
           }} }}
           WHERE {{ ?source <http://ex.org/matches> ?value }}"#
    );
    db.execute_sparql(&update).unwrap();
    let insert_epoch = db.rdf_store().commit_epoch();
    db.execute_sparql(&update).unwrap();
    let noop_epoch = db.rdf_store().commit_epoch();

    assert_eq!(db.rdf_store().graph(DEDUP_GRAPH).unwrap().len(), 1);
    let first_diff = db
        .rdf_history_diff(before_insert, insert_epoch)
        .expect("first insert history");
    assert_eq!(
        first_diff
            .transitions
            .iter()
            .filter(|transition| matches!(
                transition.kind,
                grafeo_engine::RdfHistoryTransitionKind::StatementAsserted { .. }
            ))
            .count(),
        1
    );
    assert!(
        db.rdf_history_diff(insert_epoch, noop_epoch)
            .expect("no-op insert history")
            .transitions
            .is_empty()
    );

    db.wal().unwrap().sync().unwrap();
    let exact_inserts: Vec<_> = recovered_records(&path)
        .into_iter()
        .filter(|record| {
            matches!(
                record,
                WalRecord::InsertRdfQuadV3 { graph: Some(graph), .. }
                    if graph == DEDUP_GRAPH
            )
        })
        .collect();
    assert_eq!(exact_inserts.len(), 1);

    copy_live_database(&path, &crash_copy);
    std::mem::forget(db);
    let recovered = persistent_rdf(&crash_copy);
    assert_eq!(recovered.rdf_store().graph(DEDUP_GRAPH).unwrap().len(), 1);
}

#[derive(Clone, Copy, Debug)]
enum GraphOperation {
    Clear,
    Copy,
    Move,
}

#[test]
fn clear_copy_and_move_do_not_retarget_recreated_graphs_after_crash() {
    for operation in [
        GraphOperation::Clear,
        GraphOperation::Copy,
        GraphOperation::Move,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("graph-op-{operation:?}.grafeo"));
        let crash_copy = dir
            .path()
            .join(format!("graph-op-{operation:?}-crash.grafeo"));
        let db = persistent_rdf(&path);

        db.execute_sparql(&format!(
            r#"INSERT DATA {{
                GRAPH <{TARGET}> {{ <http://ex.org/old> <{PREDICATE}> "old" . }}
                GRAPH <{SOURCE}> {{ <http://ex.org/from-source> <{PREDICATE}> "source" . }}
            }}"#
        ))
        .unwrap();
        let old_incarnation = db
            .rdf_store()
            .graph(TARGET)
            .expect("old target graph")
            .graph_incarnation();
        let source_incarnation = db
            .rdf_store()
            .graph(SOURCE)
            .expect("source graph")
            .graph_incarnation();

        let statement = match operation {
            GraphOperation::Clear => format!("CLEAR GRAPH <{TARGET}>"),
            GraphOperation::Copy => format!("COPY GRAPH <{SOURCE}> TO GRAPH <{TARGET}>"),
            GraphOperation::Move => format!("MOVE GRAPH <{SOURCE}> TO GRAPH <{TARGET}>"),
        };
        db.execute_sparql(&statement).unwrap();
        db.execute_sparql(&format!("DROP GRAPH <{TARGET}>"))
            .unwrap();
        db.execute_sparql(&format!("CREATE GRAPH <{TARGET}>"))
            .unwrap();
        db.execute_sparql(&format!(
            r#"INSERT DATA {{ GRAPH <{TARGET}> {{
                <http://ex.org/replacement> <{PREDICATE}> "replacement" .
            }} }}"#
        ))
        .unwrap();
        let replacement_incarnation = db
            .rdf_store()
            .graph(TARGET)
            .expect("replacement target graph")
            .graph_incarnation();
        assert_ne!(replacement_incarnation, old_incarnation);

        db.wal().unwrap().sync().unwrap();
        let records = recovered_records(&path);
        let mut exact_deletes: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::DeleteRdfQuadV3 {
                    subject,
                    predicate,
                    object,
                    graph,
                    graph_incarnation,
                    ..
                } => Some((
                    subject.as_str(),
                    predicate.as_str(),
                    object.as_str(),
                    graph.as_deref(),
                    *graph_incarnation,
                )),
                _ => None,
            })
            .collect();
        let mut expected_deletes = vec![(
            "<http://ex.org/old>",
            "<http://ex.org/p>",
            "\"old\"",
            Some(TARGET),
            old_incarnation,
        )];
        if !matches!(operation, GraphOperation::Clear) {
            expected_deletes.push((
                "<http://ex.org/from-source>",
                "<http://ex.org/p>",
                "\"source\"",
                Some(TARGET),
                old_incarnation,
            ));
        }
        if matches!(operation, GraphOperation::Move) {
            expected_deletes.push((
                "<http://ex.org/from-source>",
                "<http://ex.org/p>",
                "\"source\"",
                Some(SOURCE),
                source_incarnation,
            ));
        }
        exact_deletes.sort_unstable();
        expected_deletes.sort_unstable();
        assert_eq!(exact_deletes, expected_deletes, "{operation:?}");

        copy_live_database(&path, &crash_copy);
        std::mem::forget(db);

        let recovered = persistent_rdf(&crash_copy);
        let active = recovered
            .rdf_store()
            .graph(TARGET)
            .expect("recovered replacement graph");
        assert_eq!(active.graph_incarnation(), replacement_incarnation);
        assert_eq!(active.len(), 1);
        assert!(active.contains(&triple(
            "http://ex.org/replacement",
            Term::literal("replacement")
        )));
        assert!(!active.contains(&triple("http://ex.org/old", Term::literal("old"))));

        let history = recovered.rdf_store().dataset_history().unwrap();
        let target_lives: Vec<_> = history
            .graph_lives()
            .iter()
            .filter(|life| life.graph().name() == Some(TARGET))
            .collect();
        assert_eq!(target_lives.len(), 2, "{operation:?}");
        assert_eq!(target_lives[0].graph().incarnation(), old_incarnation);
        assert_eq!(
            target_lives[1].graph().incarnation(),
            replacement_incarnation
        );
    }
}

#[test]
fn canonical_alias_wal_uses_chosen_spelling_and_reopens_one_lifetime() {
    for model in [GraphModel::Rdf, GraphModel::Both]
        .into_iter()
        .filter(|model| cfg!(feature = "lpg") || *model == GraphModel::Rdf)
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("canonical.grafeo");
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(model)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        db.execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "hello"@EN . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@EN } }"#).unwrap();
        let first_epoch = db.current_epoch();
        let first = db.rdf_history_cut(first_epoch).unwrap();
        db.execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "hello"@en . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@en } }"#).unwrap();
        db.execute_sparql(r#"DELETE DATA { <urn:s> <urn:p> "hello"@en . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@en } }"#).unwrap();
        db.execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "hello"@en . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@en } }"#).unwrap();
        let mut inserts = Vec::new();
        let mut deletes = Vec::new();
        for record in recovered_records(&path) {
            match record {
                WalRecord::InsertRdfQuadV3 { object, .. } => inserts.push(object),
                WalRecord::DeleteRdfQuadV3 { object, .. } => deletes.push(object),
                _ => {}
            }
        }
        assert_eq!(
            inserts,
            [
                r#""hello"@EN"#,
                r#""hello"@EN"#,
                r#""hello"@en"#,
                r#""hello"@en"#
            ]
        );
        assert_eq!(deletes, [r#""hello"@EN"#, r#""hello"@EN"#]);
        let expected = db.rdf_history_cut(db.current_epoch()).unwrap();
        let recovered_path = dir.path().join("acknowledged-copy.grafeo");
        copy_live_database(&path, &recovered_path);
        let recovered =
            GrafeoDB::with_config(Config::persistent(&recovered_path).with_graph_model(model))
                .unwrap();
        assert_eq!(recovered.rdf_history_cut(first_epoch).unwrap(), first);
        assert_eq!(
            recovered
                .rdf_history_cut(recovered.current_epoch())
                .unwrap(),
            expected
        );
        recovered.close().unwrap();
        db.close().unwrap();
        let reopened =
            GrafeoDB::with_config(Config::persistent(&path).with_graph_model(model)).unwrap();
        assert_eq!(reopened.rdf_history_cut(first_epoch).unwrap(), first);
        assert_eq!(
            reopened.rdf_history_cut(reopened.current_epoch()).unwrap(),
            expected
        );
        assert_eq!(reopened.rdf_store().len(), 1);
        assert_eq!(reopened.rdf_store().graph("urn:g").unwrap().len(), 1);
        reopened.close().unwrap();
    }
}

#[test]
fn add_graph_alias_is_a_wal_noop_and_preserves_destination_validity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("canonical-add.grafeo");
    let db = persistent_rdf(&path);
    let session = db.session();
    session.set_rdf_valid_time_tai_ns(10, 20).unwrap();
    session
        .execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "hello"@EN }"#)
        .unwrap();
    session.set_rdf_valid_time_tai_ns(30, 40).unwrap();
    session
        .execute_sparql(r#"INSERT DATA { GRAPH <urn:g> { <urn:s> <urn:p> "hello"@en } }"#)
        .unwrap();
    let before = db.rdf_dataset_history().unwrap();
    let inserts = |records: Vec<WalRecord>| {
        records
            .into_iter()
            .filter(|r| matches!(r, WalRecord::InsertRdfQuadV3 { .. }))
            .count()
    };
    assert_eq!(inserts(recovered_records(&path)), 2);
    session
        .execute_sparql("ADD GRAPH <urn:g> TO DEFAULT")
        .unwrap();
    assert_eq!(inserts(recovered_records(&path)), 2);
    assert_eq!(
        db.rdf_dataset_history().unwrap().quad_versions(),
        before.quad_versions()
    );
    db.close().unwrap();
    let reopened = persistent_rdf(&path);
    assert_eq!(
        reopened.rdf_dataset_history().unwrap().quad_versions(),
        before.quad_versions()
    );
    reopened.close().unwrap();
}
