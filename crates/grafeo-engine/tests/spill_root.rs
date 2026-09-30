//! Public spill-root ownership witnesses for simultaneous query callers.

#![cfg(all(feature = "lpg", feature = "gql", feature = "spill"))]

use grafeo_common::utils::error::{Error, ErrorCode, QueryErrorKind};
use grafeo_core::execution::QueryExecutionControl;
use grafeo_engine::query::executor::ExecutionOptions;
use grafeo_engine::{Config, GrafeoDB, StreamChunk};
use std::collections::HashMap;

fn query_leaves(namespace: &std::path::Path) -> usize {
    std::fs::read_dir(namespace)
        .expect("read store namespace")
        .map(|entry| entry.expect("read namespace entry"))
        .filter(|entry| {
            entry.file_type().expect("query leaf type").is_dir()
                && entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("grafeo-query-")
        })
        .count()
}

fn assert_no_query_spill_leaves(root: &std::path::Path, database: &GrafeoDB) {
    let namespace = root.join(format!("grafeo-store-{}", database.store_id()));
    let mut names: Vec<_> = std::fs::read_dir(namespace)
        .expect("read store namespace")
        .map(|entry| entry.expect("read namespace entry").file_name())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            ".grafeo-spill-quota",
            ".grafeo-spill-quota.lock",
            ".grafeo-spill-root"
        ]
        .map(std::ffi::OsString::from)
    );
}

fn collect_chunk_values(chunk: &StreamChunk, values: &mut Vec<i64>) {
    for row in chunk.rows().expect("stream row view") {
        match row.first() {
            Some(grafeo_common::types::Value::Int64(value)) => values.push(*value),
            other => panic!("unexpected stream value: {other:?}"),
        }
    }
}

#[test]
fn simultaneous_resident_streams_keep_independent_owners_without_query_leaves() {
    let directory = tempfile::tempdir().expect("spill root");
    let config = Config::in_memory()
        .with_memory_limit(2 << 20)
        .with_spill_path(directory.path());
    let first_db = GrafeoDB::with_config(config.clone()).expect("first database");
    let second_db = GrafeoDB::with_config(config).expect("second database");
    let first_namespace = directory
        .path()
        .join(format!("grafeo-store-{}", first_db.store_id()));
    let second_namespace = directory
        .path()
        .join(format!("grafeo-store-{}", second_db.store_id()));
    assert_ne!(
        first_namespace, second_namespace,
        "stores need separate namespaces"
    );
    let first_control = QueryExecutionControl::new();
    let second_control = QueryExecutionControl::new();
    let third_control = QueryExecutionControl::new();
    let first_cancel = first_control.cancellation_handle();

    let mut first = first_db
        .stream_with_options(
            "UNWIND range(1, 2500) AS i RETURN i",
            HashMap::new(),
            ExecutionOptions {
                control: first_control,
                ..ExecutionOptions::default()
            },
        )
        .expect("first stream");
    let mut second = first_db
        .stream_with_options(
            "UNWIND range(1, 2500) AS i RETURN i",
            HashMap::new(),
            ExecutionOptions {
                control: second_control,
                ..ExecutionOptions::default()
            },
        )
        .expect("second stream");
    let mut third = second_db
        .stream_with_options(
            "UNWIND range(1, 2500) AS i RETURN i",
            HashMap::new(),
            ExecutionOptions {
                control: third_control,
                ..ExecutionOptions::default()
            },
        )
        .expect("third stream");

    let mut first_values = Vec::new();
    let mut second_values = Vec::new();
    let mut third_values = Vec::new();
    collect_chunk_values(
        &first
            .next_chunk()
            .expect("first chunk")
            .expect("first rows"),
        &mut first_values,
    );
    collect_chunk_values(
        &second
            .next_chunk()
            .expect("second chunk")
            .expect("second rows"),
        &mut second_values,
    );
    collect_chunk_values(
        &third
            .next_chunk()
            .expect("third chunk")
            .expect("third rows"),
        &mut third_values,
    );
    assert!(!first_values.is_empty());

    assert_eq!(query_leaves(&first_namespace), 0);
    assert_eq!(query_leaves(&second_namespace), 0);

    first_cancel.cancel();
    let first_error = first.next_chunk().expect_err("first owner cancellation");
    assert!(matches!(
        first_error,
        Error::Query(query) if query.kind == QueryErrorKind::Cancelled
    ));
    assert_eq!(query_leaves(&first_namespace), 0);
    while let Some(chunk) = second.next_chunk().expect("second next chunk") {
        collect_chunk_values(&chunk, &mut second_values);
    }
    while let Some(chunk) = third.next_chunk().expect("third next chunk") {
        collect_chunk_values(&chunk, &mut third_values);
    }
    let mut expected: Vec<_> = (1..=2500).collect();
    second_values.sort_unstable();
    third_values.sort_unstable();
    expected.sort_unstable();
    assert_eq!(second_values, expected);
    assert_eq!(third_values, expected);
    second.close().expect("second stream cleanup");
    third.close().expect("third stream cleanup");
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    feature = "lpg",
    feature = "gql"
))]
fn assert_session_external_sort_crosses_spill(mut config: Config, root_budget: bool) {
    use grafeo_common::types::Value;

    // The sort charges its doubled outer capacity and stable-sort scratch.
    // An explicit read transaction uses the owned planner route; cacheable
    // pull-sort accounting remains a P6 caller obligation. TopK structural
    // matching ignores aliases, so ORDER BY value cannot match the projected
    // arithmetic expression. LIMIT bounds output after the complete sort.
    let query = "UNWIND range(0, 24575) AS i \
                 RETURN 12287 - (i % 12288) AS value ORDER BY value LIMIT 768";
    config = config.with_memory_limit(2 << 20);

    let denied_directory = tempfile::tempdir().unwrap();
    let denied_root = denied_directory.path().join("spill");
    let denied_config = config.clone().with_spill_path(&denied_root);
    let denied_config = if root_budget {
        // Root metadata, one leaf and one small file fit; actual external-sort
        // data growth must consume the shared physical budget.
        denied_config
            .with_max_query_spill_bytes(64 << 20)
            .with_max_root_spill_bytes((1 << 20) + (256 << 10))
    } else {
        denied_config.with_max_query_spill_bytes(0)
    };
    let denied_database = GrafeoDB::with_config(denied_config).unwrap();
    let mut denied_session = denied_database.session();
    denied_session.begin_transaction().unwrap();
    let denied_execution = denied_session.execute(query);
    denied_session.rollback().unwrap();

    let directory = tempfile::tempdir().unwrap();
    let spill_root = directory.path().join("spill");
    let database = GrafeoDB::with_config(
        config
            .with_spill_path(&spill_root)
            .with_max_query_spill_bytes(64 << 20)
            .with_max_root_spill_bytes(64 << 20),
    )
    .unwrap();
    let mut session = database.session();
    session.begin_transaction().unwrap();
    let positive_execution = session.execute(query);
    session.rollback().unwrap();
    let Err(error) = denied_execution else {
        panic!(
            "zero disk quota unexpectedly completed the sort; positive: {:?}",
            positive_execution.as_ref().map(|result| result.row_count())
        );
    };
    assert_eq!(
        error.error_code(),
        ErrorCode::StorageFull,
        "zero disk quota must reject an actual sort spill: {error:?}; {error}; positive: {:?}",
        positive_execution.as_ref().map(|result| result.row_count())
    );
    assert!(
        error.to_string().contains("spill disk quota exceeded"),
        "the witness must reach disk admission: {error}"
    );
    assert_no_query_spill_leaves(&denied_root, &denied_database);

    let result = positive_execution.unwrap();
    assert_eq!(result.row_count(), 768);
    for (index, row) in result.rows().iter().enumerate() {
        assert_eq!(
            row,
            &vec![Value::Int64(i64::try_from(index / 2).unwrap())],
            "external sort must preserve order and both copies of every key"
        );
    }
    assert_no_query_spill_leaves(&spill_root, &database);
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    feature = "lpg",
    feature = "gql"
))]
#[test]
fn plaintext_session_external_sort_crosses_authenticated_spill_root() {
    assert_session_external_sort_crosses_spill(Config::in_memory(), false);
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    feature = "lpg",
    feature = "gql",
    feature = "encryption"
))]
#[test]
fn encrypted_session_external_sort_crosses_authenticated_spill_root() {
    let mut config = Config::in_memory();
    config.encryption = Some(grafeo_engine::config::EncryptionConfig {
        key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([0x6d; 32])),
    });
    assert_session_external_sort_crosses_spill(config, false);
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn forced_spill_distinct_route_preserves_exact_public_result() {
    use std::fmt::Write;

    let directory = tempfile::tempdir().expect("spill root");
    let config = Config::in_memory()
        .with_graph_model(grafeo_engine::GraphModel::Rdf)
        .with_memory_limit(2 << 20)
        .with_spill_path(directory.path());
    let mut query = String::from("SELECT DISTINCT (STR(?term) AS ?value) WHERE { VALUES ?term { ");
    for index in 0..4096 {
        write!(query, "<urn:pressure:{}> ", (index * 37) % 4096).unwrap();
    }
    query.push_str("} }");
    let denied = GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0))
        .expect("denial database")
        .execute_sparql(&query)
        .expect_err("zero spill quota must deny pressured DISTINCT");
    assert_eq!(denied.error_code(), ErrorCode::StorageFull);
    let result = GrafeoDB::with_config(config)
        .expect("spill database")
        .execute_sparql(&query)
        .expect("distinct spill route");
    assert_eq!(result.row_count(), 4096);
    let expected: Vec<_> = (0..4096)
        .map(|index| {
            vec![grafeo_common::types::Value::from(format!(
                "urn:pressure:{}",
                (index * 37) % 4096
            ))]
        })
        .collect();
    assert_eq!(result.rows(), expected.as_slice());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn idle_session_restore_roundtrip(config: Config, spill_parent: &std::path::Path) {
    let database = GrafeoDB::with_config(config).expect("restore target");
    database
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("seed first world");
    let first_id = database.store_id();
    let first_snapshot = database.export_snapshot().expect("capture first world");
    let first_namespace = spill_parent.join(format!("grafeo-store-{first_id}"));

    let other = GrafeoDB::new_in_memory();
    other
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("seed replacement world");
    let second_id = other.store_id();
    assert_ne!(first_id, second_id);
    let second_snapshot = other.export_snapshot().expect("capture replacement world");
    let second_namespace = spill_parent.join(format!("grafeo-store-{second_id}"));
    assert!(!second_namespace.exists());

    // Keep this exact public Session alive and idle across both replacements.
    let session = database.session();
    let query = "MATCH (p:Person) RETURN p.name";
    assert_eq!(
        session.execute(query).expect("first world query").rows(),
        &[vec![grafeo_common::types::Value::from("Alix")]],
    );
    let first_marker = std::fs::read(first_namespace.join(".grafeo-spill-root"))
        .expect("first world authenticated root");
    database
        .restore_snapshot(&second_snapshot)
        .expect("replace first world with second world");
    assert_eq!(database.store_id(), second_id);
    assert_eq!(
        session
            .execute(query)
            .expect("same session after restore")
            .rows(),
        &[vec![grafeo_common::types::Value::from("Gus")]],
    );
    assert!(second_namespace.join(".grafeo-spill-root").is_file());
    assert_eq!(query_leaves(&first_namespace), 0);
    assert_eq!(query_leaves(&second_namespace), 0);

    database
        .restore_snapshot(&first_snapshot)
        .expect("restore the original world");
    assert_eq!(database.store_id(), first_id);
    assert_eq!(
        session
            .execute(query)
            .expect("same session after roundtrip")
            .rows(),
        &[vec![grafeo_common::types::Value::from("Alix")]],
    );
    assert_eq!(
        std::fs::read(first_namespace.join(".grafeo-spill-root"))
            .expect("retained original authenticated root"),
        first_marker,
        "restoring A must authenticate its original root, not replace its seal",
    );
    assert_eq!(query_leaves(&first_namespace), 0);
    assert_eq!(query_leaves(&second_namespace), 0);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn idle_session_tracks_restored_store_and_reuses_in_memory_root_authority() {
    let directory = tempfile::tempdir().expect("restore spill parent");
    idle_session_restore_roundtrip(
        Config::in_memory().with_spill_path(directory.path()),
        directory.path(),
    );
}

#[cfg(all(any(target_os = "linux", target_os = "macos"), feature = "grafeo-file"))]
#[test]
fn idle_session_tracks_restored_store_in_wal_disabled_persistent_container() {
    let directory = tempfile::tempdir().expect("persistent restore directory");
    let path = directory.path().join("restore.grafeo");
    let spill = directory.path().join("spill");
    let mut config = Config::persistent(&path).with_spill_path(&spill);
    config.wal_enabled = false;
    idle_session_restore_roundtrip(config, &spill);
    assert!(
        path.is_file(),
        "the restore target must be a real container"
    );
    let authority_directory = directory.path().join("restore.grafeo.spill-auth");
    let records = std::fs::read_dir(authority_directory)
        .expect("persistent authorities")
        .map(|entry| entry.expect("authority entry"))
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("authority-")
        })
        .count();
    assert_eq!(records, 2, "both restored StoreIds retain their local keys");
}

#[cfg(all(any(target_os = "linux", target_os = "macos"), feature = "grafeo-file"))]
#[test]
fn persistent_plaintext_spill_authority_survives_close_and_reopen() {
    let directory = tempfile::tempdir().expect("persistent spill directory");
    let path = directory.path().join("restart.grafeo");
    let spill = directory.path().join("spill");
    let mut config = Config::persistent(&path).with_spill_path(&spill);
    config.wal_enabled = false;

    let database = GrafeoDB::with_config(config.clone()).expect("open persistent database");
    database
        .execute("INSERT (:Person {name: 'restart'})")
        .expect("seed persistent database");
    let store_id = database.store_id();
    assert_eq!(
        database
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .expect("first spill-backed query")
            .row_count(),
        1
    );
    let namespace = spill.join(format!("grafeo-store-{store_id}"));
    let marker_path = namespace.join(".grafeo-spill-root");
    let marker = std::fs::read(&marker_path).expect("persistent root marker");
    let authority_directory = path.with_extension("grafeo.spill-auth");
    let authority_entries: Vec<_> = std::fs::read_dir(&authority_directory)
        .expect("authority sidecar")
        .map(|entry| entry.expect("authority entry"))
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("authority-")
        })
        .collect();
    assert_eq!(authority_entries.len(), 1, "one StoreId authority record");
    let authority_path = authority_entries[0].path();
    let authority_bytes = std::fs::read(&authority_path).expect("authority bytes");
    assert_eq!(query_leaves(&namespace), 0);
    database
        .close()
        .expect("flush and close persistent database");
    drop(database);

    let reopened = GrafeoDB::with_config(config).expect("reopen persistent database");
    assert_eq!(reopened.store_id(), store_id);
    assert_eq!(
        reopened
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .expect("reopened spill-backed query")
            .row_count(),
        1
    );
    assert_eq!(
        std::fs::read(&marker_path).expect("reused root marker"),
        marker
    );
    assert_eq!(
        std::fs::read(&authority_path).expect("reused authority bytes"),
        authority_bytes
    );
    assert_eq!(query_leaves(&namespace), 0);
    reopened.close().expect("close reopened database");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn public_query_rejects_foreign_key_and_unknown_root_version_without_publication() {
    let directory = tempfile::tempdir().expect("authenticated root parent");
    let config = Config::in_memory().with_spill_path(directory.path());
    let first = GrafeoDB::with_config(config.clone()).expect("first authority owner");
    assert_eq!(
        first
            .execute("RETURN 1")
            .expect("establish root")
            .row_count(),
        1
    );
    let snapshot = first.export_snapshot().expect("capture exact StoreId");
    let namespace = directory
        .path()
        .join(format!("grafeo-store-{}", first.store_id()));
    let marker_path = namespace.join(".grafeo-spill-root");
    let original = std::fs::read(&marker_path).expect("authenticated marker");
    assert_eq!(query_leaves(&namespace), 0);

    let foreign = GrafeoDB::with_config(config).expect("independent authority owner");
    foreign
        .restore_snapshot(&snapshot)
        .expect("restore same logical StoreId");
    assert_eq!(foreign.store_id(), first.store_id());
    foreign
        .execute("RETURN 1")
        .expect_err("foreign key cannot adopt an existing root");
    assert_eq!(std::fs::read(&marker_path).unwrap(), original);
    assert_eq!(query_leaves(&namespace), 0);

    let mut unknown_version = original.clone();
    unknown_version[4] = 255;
    std::fs::write(&marker_path, &unknown_version).expect("inject unknown marker version");
    first
        .execute("RETURN 1")
        .expect_err("unknown root version must reject query admission");
    assert_eq!(std::fs::read(&marker_path).unwrap(), unknown_version);
    assert_eq!(query_leaves(&namespace), 0);

    std::fs::write(&marker_path, &original).expect("restore original authenticated marker");
    assert_eq!(
        first
            .execute("RETURN 1")
            .expect("original seal remains usable")
            .row_count(),
        1
    );
    assert_eq!(query_leaves(&namespace), 0);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn plaintext_external_sort_enforces_physical_root_budget() {
    assert_session_external_sort_crosses_spill(Config::in_memory(), true);
}

#[cfg(all(any(target_os = "linux", target_os = "macos"), feature = "encryption"))]
#[test]
fn encrypted_external_sort_enforces_physical_root_budget() {
    let mut config = Config::in_memory();
    config.encryption = Some(grafeo_engine::config::EncryptionConfig {
        key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([0x6d; 32])),
    });
    assert_session_external_sort_crosses_spill(config, true);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn configured_resident_queries_do_not_mutate_the_quota_ledger() {
    let directory = tempfile::tempdir().unwrap();
    let database = GrafeoDB::with_config(
        Config::in_memory()
            .with_spill_path(directory.path())
            .with_max_root_spill_bytes(1 << 20),
    )
    .unwrap();
    database.execute("RETURN 1").unwrap();
    let namespace = directory
        .path()
        .join(format!("grafeo-store-{}", database.store_id()));
    let ledger = namespace.join(".grafeo-spill-quota");
    let before = std::fs::read(&ledger).unwrap();
    for _ in 0..3 {
        assert_eq!(database.execute("RETURN 1").unwrap().row_count(), 1);
    }
    assert_eq!(std::fs::read(ledger).unwrap(), before);
    assert_eq!(query_leaves(&namespace), 0);
}

#[cfg(all(any(target_os = "linux", target_os = "macos"), feature = "encryption"))]
#[test]
fn profile_reports_terminal_physical_spill_history_plaintext_and_encrypted() {
    use grafeo_common::types::Value;
    for encrypted in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut config = Config::in_memory()
            .with_memory_limit(2 << 20)
            .with_spill_path(directory.path())
            .with_max_root_spill_bytes(64 << 20);
        if encrypted {
            config.encryption = Some(grafeo_engine::config::EncryptionConfig {
                key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new(
                    [0x52; 32],
                )),
            });
        }
        let database = GrafeoDB::with_config(config).unwrap();
        let result = database.execute("PROFILE UNWIND range(0, 24575) AS i RETURN 12287 - (i % 12288) AS value ORDER BY value LIMIT 768").unwrap();
        let Value::String(text) = &result.rows()[0][0] else {
            panic!("PROFILE text");
        };
        let physical = text
            .lines()
            .find(|line| line.contains("spill_physical_reserved_bytes="))
            .unwrap();
        let value = |name: &str| {
            physical
                .split_whitespace()
                .filter_map(|f| f.split_once('='))
                .find_map(|(k, v)| (k == name).then_some(v))
                .unwrap()
                .parse::<u64>()
                .unwrap()
        };
        assert_eq!(value("spill_physical_reserved_bytes"), 0, "{text}");
        assert_eq!(value("spill_cleanup_debt_bytes"), 0, "{text}");
        assert_eq!(
            value("spill_observed_file_bytes_at_publication"),
            0,
            "{text}"
        );
        assert!(value("spill_physical_peak_bytes") > 0, "{text}");
        assert!(value("spill_observed_file_peak_bytes") > 0, "{text}");
        assert!(physical.contains("spill_cleanup_failed=false"));
        assert!(physical.contains("spill_reservation_uncertain=false"));
        assert_no_query_spill_leaves(directory.path(), &database);
    }
}
