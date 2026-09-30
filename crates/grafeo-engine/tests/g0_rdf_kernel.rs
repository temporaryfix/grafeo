//! G0 — RDF kernel black-box contract tests.
//!
//! Historical characterization tests are now positive regression gates.
//! G2 contracts (lossless terms, date/dateTime values, and native paths) run in CI.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file" \
//!   --test g0_rdf_kernel -- --test-threads=1
//! ```
//!
//! These tests exercise the public RDF transaction and persistence contracts.

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::fmt::Write as _;

use grafeo_common::types::Value;
use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .expect("open in-memory rdf db")
}

fn persistent_rdf(path: &std::path::Path) -> GrafeoDB {
    let config = Config::persistent(path).with_graph_model(GraphModel::Rdf);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn persistent_rdf_sync(path: &std::path::Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn sidecar_wal_path(path: &std::path::Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

fn wal_payload_bytes(path: &std::path::Path) -> u64 {
    let wal_dir = sidecar_wal_path(path);
    let mut n = 0u64;
    if let Ok(entries) = std::fs::read_dir(&wal_dir) {
        for e in entries.flatten() {
            if let Ok(meta) = e.metadata()
                && meta.is_file()
            {
                n = n.saturating_add(meta.len());
            }
        }
    }
    n
}

fn count_names(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .expect("select")
        .row_count()
}

// ---------------------------------------------------------------------------
// Historical characterization names retained as positive regression gates.
// ---------------------------------------------------------------------------

#[test]
fn characterization_sparql_lang_tag_survives_select() {
    let db = rdf_db();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/name> "Alix"@en . }"#)
        .unwrap();
    let rows = db
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    match &rows.rows()[0][0] {
        Value::RdfLiteral {
            lexical,
            language: Some(lang),
            ..
        } => {
            assert_eq!(lexical.as_str(), "Alix");
            assert_eq!(lang.as_str(), "en");
        }
        other => panic!("lang tag must survive SELECT, got {other:?}"),
    }
}

#[test]
fn characterization_typed_date_filter_matches() {
    let db = rdf_db();
    db.execute_sparql(
        r#"
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        INSERT DATA {
          <http://ex.org/r> <http://ex.org/start> "2010-01-01"^^xsd:date .
        }
        "#,
    )
    .unwrap();
    let hit = db
        .execute_sparql(
            r#"
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?s WHERE {
              ?s <http://ex.org/start> ?d .
              FILTER (?d <= "2015-06-01"^^xsd:date)
            }
            "#,
        )
        .unwrap();
    assert_eq!(
        hit.row_count(),
        1,
        "xsd:date FILTER compares in the date value space"
    );
}

#[test]
fn characterization_property_path_star_is_unbounded() {
    let db = rdf_db();
    // 51-edge chain :n0 -p-> :n1 ... -p-> :n51
    let mut insert = String::from("INSERT DATA {\n");
    for i in 0..51 {
        writeln!(
            insert,
            "  <http://ex.org/n{i}> <http://ex.org/p> <http://ex.org/n{}> .",
            i + 1
        )
        .unwrap();
    }
    insert.push_str("}\n");
    db.execute_sparql(&insert).unwrap();
    let far = db
        .execute_sparql("SELECT ?x WHERE { <http://ex.org/n0> <http://ex.org/p>* ?x }")
        .unwrap();
    let reached: Vec<_> = far.rows().iter().map(|r| format!("{}", r[0])).collect();
    assert!(
        reached.iter().any(|s| s.contains("n51")),
        "native path* must reach hop 51, got {reached:?}"
    );
}

#[test]
fn characterization_batch_insert_survives_close_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batch.grafeo");
    {
        let db = persistent_rdf(&path);
        let t = Triple::new(
            Term::iri("http://ex.org/s"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        );
        assert_eq!(db.batch_insert_rdf([t]).expect("batch insert"), 1);
        db.close().unwrap();
    }
    let db = persistent_rdf(&path);
    assert_eq!(
        count_names_pred(&db, "http://ex.org/p"),
        1,
        "batch_insert must survive close()/snapshot via the WAL chokepoint"
    );
}

#[test]
fn contract_batch_insert_logs_wal_before_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batch_wal.grafeo");
    let db = persistent_rdf_sync(&path);
    let before = wal_payload_bytes(&path);
    let t = Triple::new(
        Term::iri("http://ex.org/s"),
        Term::iri("http://ex.org/p"),
        Term::literal("v"),
    );
    assert_eq!(db.batch_insert_rdf([t]).expect("batch insert"), 1);
    let after = wal_payload_bytes(&path);
    assert!(
        after > before,
        "batch_insert_rdf must append WAL (before={before} after={after}); not RAM-only"
    );
    assert_eq!(count_names_pred(&db, "http://ex.org/p"), 1);
}

#[test]
fn contract_copy_move_add_log_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("copy_wal.grafeo");
    let db = persistent_rdf_sync(&path);
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/name> "n" . }"#)
        .unwrap();
    let before = wal_payload_bytes(&path);
    db.execute_sparql("COPY DEFAULT TO <http://ex.org/g>")
        .expect("COPY");
    let after_copy = wal_payload_bytes(&path);
    assert!(
        after_copy > before,
        "COPY must WAL-log (before={before} after={after_copy})"
    );
    db.execute_sparql("ADD <http://ex.org/g> TO <http://ex.org/g2>")
        .expect("ADD");
    let after_add = wal_payload_bytes(&path);
    assert!(after_add > after_copy, "ADD must WAL-log");
    db.execute_sparql("MOVE <http://ex.org/g2> TO <http://ex.org/g3>")
        .expect("MOVE");
    let after_move = wal_payload_bytes(&path);
    assert!(after_move > after_add, "MOVE must WAL-log");
}

#[test]
fn contract_load_returns_structured_error() {
    let db = rdf_db();
    let err = db
        .execute_sparql("LOAD <http://example.org/data.ttl>")
        .expect_err("LOAD must not succeed");
    let msg = err.to_string();
    assert!(
        msg.contains("LOAD") && msg.contains("not supported"),
        "structured LOAD error, got {msg}"
    );
}

fn count_names_pred(db: &GrafeoDB, pred: &str) -> usize {
    db.execute_sparql(&format!("SELECT ?o WHERE {{ ?s <{pred}> ?o }}"))
        .unwrap()
        .row_count()
}

#[test]
fn characterization_grafeodb_execute_sparql_has_no_session_tx() {
    // Autocommit SPARQL now goes through Session (one-statement tx). Persist
    // via close()/snapshot remains; crash-without-close is still a G1 kill -9 test.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oneshot.grafeo");
    {
        let db = persistent_rdf(&path);
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/name> "x" . }"#)
            .unwrap();
        db.close().unwrap();
    }
    let db = persistent_rdf(&path);
    assert_eq!(
        count_names(&db),
        1,
        "HEAD: GrafeoDB::execute_sparql INSERT survives close()/snapshot; crash-without-close is G1"
    );
}

// ---------------------------------------------------------------------------
// Contract (CI): G1 chokepoint + G2 term/date/path*.
// ---------------------------------------------------------------------------

#[test]
fn contract_lang_tag_survives_select() {
    let db = rdf_db();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/name> "Alix"@en . }"#)
        .unwrap();
    let rows = db
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    let v = &rows.rows()[0][0];
    match v {
        Value::String(_) => panic!("lang tag was stripped to Value::String: {v:?}"),
        other => {
            let s = format!("{other:?}");
            assert!(
                s.contains("en") || s.contains('@'),
                "expected a language-tagged RDF term, got {other:?}"
            );
        }
    }
}

#[test]
fn contract_xsd_date_filter_value_space() {
    let db = rdf_db();
    db.execute_sparql(
        r#"
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        INSERT DATA {
          <http://ex.org/r> <http://ex.org/start> "2010-01-01"^^xsd:date .
        }
        "#,
    )
    .unwrap();
    let hit = db
        .execute_sparql(
            r#"
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?s WHERE {
              ?s <http://ex.org/start> ?d .
              FILTER (?d <= "2015-06-01"^^xsd:date)
            }
            "#,
        )
        .unwrap();
    assert_eq!(hit.row_count(), 1);
}

#[test]
fn contract_property_path_star_is_unbounded() {
    let db = rdf_db();
    let mut insert = String::from("INSERT DATA {\n");
    for i in 0..51 {
        writeln!(
            insert,
            "  <http://ex.org/n{i}> <http://ex.org/p> <http://ex.org/n{}> .",
            i + 1
        )
        .unwrap();
    }
    insert.push_str("}\n");
    db.execute_sparql(&insert).unwrap();
    let far = db
        .execute_sparql("SELECT ?x WHERE { <http://ex.org/n0> <http://ex.org/p>* ?x }")
        .unwrap();
    let reached: Vec<_> = far.rows().iter().map(|r| format!("{}", r[0])).collect();
    assert!(
        reached.iter().any(|s| s.contains("n51")),
        "native path* must reach hop 51, got {reached:?}"
    );
}

#[test]
fn contract_session_ryw_and_isolation() {
    let db = rdf_db();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/name> "secret" . }"#)
        .unwrap();
    let mine = writer
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    assert_eq!(mine.row_count(), 1, "read-your-writes in the same Session");

    let reader = db.session();
    let theirs = reader
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    assert_eq!(
        theirs.row_count(),
        0,
        "uncommitted RDF must not be visible to another Session"
    );
    writer.commit().unwrap();
    let after = reader
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    assert_eq!(after.row_count(), 1);
}

#[test]
fn contract_named_graph_ryw_flush_and_isolation() {
    let db = rdf_db();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/claims> { <http://ex.org/s> <http://ex.org/name> "secret" . } }"#,
        )
        .unwrap();
    let mine = writer
        .execute_sparql(
            "SELECT ?n WHERE { GRAPH <http://ex.org/claims> { ?s <http://ex.org/name> ?n } }",
        )
        .unwrap();
    assert_eq!(
        mine.row_count(),
        1,
        "read-your-writes in a named GRAPH in the same Session"
    );
    let default = writer
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    assert_eq!(
        default.row_count(),
        0,
        "named-graph insert must not leak into the default graph"
    );

    let reader = db.session();
    let theirs = reader
        .execute_sparql(
            "SELECT ?n WHERE { GRAPH <http://ex.org/claims> { ?s <http://ex.org/name> ?n } }",
        )
        .unwrap();
    assert_eq!(
        theirs.row_count(),
        0,
        "uncommitted named-graph RDF must not be visible to another Session"
    );
    writer.commit().unwrap();
    let after = reader
        .execute_sparql(
            "SELECT ?n WHERE { GRAPH <http://ex.org/claims> { ?s <http://ex.org/name> ?n } }",
        )
        .unwrap();
    assert_eq!(
        after.row_count(),
        1,
        "named-graph tx buffers must flush on commit"
    );
}

// ---------------------------------------------------------------------------
// G2 remainder: nested paths, dateTime, typed integers, MODIFY GRAPH
// ---------------------------------------------------------------------------

#[test]
fn contract_nested_sequence_path_star_is_unbounded() {
    let db = rdf_db();
    let mut insert = String::from("INSERT DATA {\n");
    for i in 0..51 {
        writeln!(
            insert,
            "  <http://ex.org/n{i}> <http://ex.org/p> <http://ex.org/m{i}> ."
        )
        .unwrap();
        writeln!(
            insert,
            "  <http://ex.org/m{i}> <http://ex.org/q> <http://ex.org/n{}> .",
            i + 1
        )
        .unwrap();
    }
    insert.push_str("}\n");
    db.execute_sparql(&insert).unwrap();
    let far = db
        .execute_sparql(
            "SELECT ?x WHERE { <http://ex.org/n0> (<http://ex.org/p>/<http://ex.org/q>)* ?x }",
        )
        .unwrap();
    let reached: Vec<_> = far.rows().iter().map(|r| format!("{}", r[0])).collect();
    assert!(
        reached.iter().any(|s| s.contains("n51")),
        "nested (p/q)* must reach hop 51, got {reached:?}"
    );
}

#[test]
fn contract_nested_alternative_path_star_is_unbounded() {
    let db = rdf_db();
    let mut insert = String::from("INSERT DATA {\n");
    for i in 0..51 {
        let pred = if i % 2 == 0 {
            "http://ex.org/p"
        } else {
            "http://ex.org/q"
        };
        writeln!(
            insert,
            "  <http://ex.org/n{i}> <{pred}> <http://ex.org/n{}> .",
            i + 1
        )
        .unwrap();
    }
    insert.push_str("}\n");
    db.execute_sparql(&insert).unwrap();
    let far = db
        .execute_sparql(
            "SELECT ?x WHERE { <http://ex.org/n0> (<http://ex.org/p>|<http://ex.org/q>)* ?x }",
        )
        .unwrap();
    let reached: Vec<_> = far.rows().iter().map(|r| format!("{}", r[0])).collect();
    assert!(
        reached.iter().any(|s| s.contains("n51")),
        "nested (p|q)* must reach hop 51, got {reached:?}"
    );
}

#[test]
fn contract_xsd_datetime_filter_value_space() {
    let db = rdf_db();
    db.execute_sparql(
        r#"
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        INSERT DATA {
          <http://ex.org/r> <http://ex.org/at> "2010-01-01T00:00:00Z"^^xsd:dateTime .
        }
        "#,
    )
    .unwrap();
    let hit = db
        .execute_sparql(
            r#"
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?s WHERE {
              ?s <http://ex.org/at> ?d .
              FILTER (?d <= "2015-06-01T00:00:00Z"^^xsd:dateTime)
            }
            "#,
        )
        .unwrap();
    assert_eq!(
        hit.row_count(),
        1,
        "xsd:dateTime FILTER compares in the dateTime value space"
    );
}

#[test]
fn contract_typed_integer_survives_select() {
    let db = rdf_db();
    db.execute_sparql(
        r#"
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        INSERT DATA { <http://ex.org/a> <http://ex.org/n> "1"^^xsd:integer . }
        "#,
    )
    .unwrap();
    let rows = db
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/n> ?n }")
        .unwrap();
    match &rows.rows()[0][0] {
        Value::RdfLiteral {
            lexical,
            datatype: Some(dt),
            language: None,
        } => {
            assert_eq!(lexical.as_str(), "1");
            assert!(
                dt.ends_with("#integer"),
                "expected xsd:integer datatype, got {dt}"
            );
        }
        other => panic!("typed integer must keep datatype in SELECT, got {other:?}"),
    }
    let hit = db
        .execute_sparql(
            r#"
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?s WHERE { ?s <http://ex.org/n> ?n . FILTER (?n = 1) }
            "#,
        )
        .unwrap();
    assert_eq!(hit.row_count(), 1, "xsd:integer FILTER vs numeric 1");
}

#[test]
fn contract_modify_named_graph_and_tid() {
    let db = rdf_db();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> "v" . } }"#,
        )
        .unwrap();
    writer
        .execute_sparql(
            r#"DELETE { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> "v" . } }
               INSERT { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> "w" . } }
               WHERE { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> "v" . } }"#,
        )
        .unwrap();
    let mine = writer
        .execute_sparql(
            r#"SELECT ?o WHERE { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> ?o } }"#,
        )
        .unwrap();
    assert_eq!(mine.row_count(), 1, "MODIFY RYW in named GRAPH");
    assert!(
        format!("{:?}", mine.rows()[0][0]).contains('w'),
        "MODIFY must replace v with w in the named graph, got {:?}",
        mine.rows()[0][0]
    );
    let reader = db.session();
    let theirs = reader
        .execute_sparql(r#"SELECT ?o WHERE { GRAPH <http://ex.org/g> { ?s ?p ?o } }"#)
        .unwrap();
    assert_eq!(theirs.row_count(), 0, "uncommitted MODIFY must not leak");
    writer.commit().unwrap();
    let after = reader
        .execute_sparql(
            r#"SELECT ?o WHERE { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> ?o } }"#,
        )
        .unwrap();
    assert_eq!(after.row_count(), 1);
    assert!(format!("{:?}", after.rows()[0][0]).contains('w'));
}
