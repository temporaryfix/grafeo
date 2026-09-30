//! Crash coverage for committed backup manifest/cursor generations.

#![cfg(all(
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]

use std::path::Path;
use std::process::Command;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const FAILPOINTS: &[&str] = &[
    "backup:segment_write",
    "backup:segment_sync",
    "backup:segment_validated",
    "backup:manifest_installing_sync",
    "backup:manifest_rename",
    "backup:manifest_directory_sync",
    "backup:cursor_installing_sync",
    "backup:cursor_rename",
    "backup:cursor_directory_sync",
    "backup:commit_directory_sync",
];
const RESTORE_FAILPOINTS: &[&str] = &[
    "backup:restore_before_replace",
    "backup:restore_after_replace",
    "backup:restore_parent_sync",
];

fn populate(db: &GrafeoDB, value: i64) {
    db.execute(&format!("INSERT (:N {{n: {value}}})")).unwrap();
}

fn restored_values(
    backup_dir: &Path,
    epoch: grafeo_common::types::EpochId,
    output: &Path,
) -> Vec<Vec<Value>> {
    GrafeoDB::restore_to_epoch(backup_dir, epoch, output).unwrap();
    let db = GrafeoDB::open(output).unwrap();
    let rows = db
        .execute("MATCH (n:N) RETURN n.n ORDER BY n.n")
        .unwrap()
        .rows()
        .to_vec();
    db.close().unwrap();
    rows
}

fn chain_hex(chain_id: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(64);
    for byte in chain_id {
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}

#[test]
fn alternating_incrementals_keep_chain_namespaced_cursors() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    let db = GrafeoDB::open(&source_path).unwrap();
    populate(&db, 1);
    let first_dir = dir.path().join("first-backup");
    let second_dir = dir.path().join("second-backup");
    db.backup_full(&first_dir).unwrap();
    db.backup_full(&second_dir).unwrap();
    let first_chain = GrafeoDB::read_backup_manifest(&first_dir)
        .unwrap()
        .unwrap()
        .chain_id;
    let second_chain = GrafeoDB::read_backup_manifest(&second_dir)
        .unwrap()
        .unwrap()
        .chain_id;
    assert_ne!(first_chain, second_chain);

    populate(&db, 2);
    let first_increment = db.backup_incremental(&first_dir).unwrap();
    assert_eq!(
        restored_values(
            &first_dir,
            first_increment.end_epoch,
            &dir.path().join("first-restored.grafeo"),
        ),
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]]
    );
    populate(&db, 3);
    let second_increment = db.backup_incremental(&second_dir).unwrap();
    assert_eq!(
        restored_values(
            &second_dir,
            second_increment.end_epoch,
            &dir.path().join("second-restored.grafeo"),
        ),
        vec![
            vec![Value::Int64(1)],
            vec![Value::Int64(2)],
            vec![Value::Int64(3)]
        ]
    );
    let first = GrafeoDB::read_backup_manifest(&first_dir).unwrap().unwrap();
    let second = GrafeoDB::read_backup_manifest(&second_dir)
        .unwrap()
        .unwrap();
    assert_eq!(first.chain_id, first_chain);
    assert_eq!(second.chain_id, second_chain);
    db.close().unwrap();
}

#[test]
fn removing_advisory_manifest_recovers_immutable_pair() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    let backup_dir = dir.path().join("backup");
    let db = GrafeoDB::open(&source_path).unwrap();
    populate(&db, 1);
    let full = db.backup_full(&backup_dir).unwrap();
    std::fs::remove_file(backup_dir.join("backup_manifest.json")).unwrap();
    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .expect("immutable committed pair remains authoritative");
    assert_eq!(
        restored_values(
            &backup_dir,
            full.end_epoch,
            &dir.path().join("restored.grafeo"),
        ),
        vec![vec![Value::Int64(1)]]
    );
    assert_eq!(manifest.chain_id, full.chain_id);
    db.close().unwrap();
}

#[test]
fn installing_manifest_debris_refuses_increment_without_changing_previous_pair() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    let backup_dir = dir.path().join("backup");
    let db = GrafeoDB::open(&source_path).unwrap();
    populate(&db, 1);
    db.backup_full(&backup_dir).unwrap();
    populate(&db, 2);
    let manifest_before = std::fs::read(backup_dir.join("backup_manifest.json")).unwrap();
    let cursor_path =
        std::path::PathBuf::from(format!("{}.wal/backup_cursor.meta", source_path.display()));
    let cursor_before = std::fs::read(&cursor_path).unwrap();
    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    let next = manifest.generation.checked_add(1).unwrap();
    let installing = backup_dir.join(format!(
        "backup_manifest_{}_{}.meta.installing",
        chain_hex(&manifest.chain_id),
        format_args!("{next:020}")
    ));
    std::fs::create_dir_all(&installing).unwrap();
    assert!(db.backup_incremental(&backup_dir).is_err());
    assert_eq!(
        std::fs::read(backup_dir.join("backup_manifest.json")).unwrap(),
        manifest_before
    );
    assert_eq!(std::fs::read(&cursor_path).unwrap(), cursor_before);
    assert!(installing.is_dir(), "unowned installing debris must remain");
    assert_eq!(
        restored_values(
            &backup_dir,
            manifest.segments.last().unwrap().end_epoch,
            &dir.path().join("unchanged-restored.grafeo"),
        ),
        vec![vec![Value::Int64(1)]]
    );
    db.close().unwrap();
}

#[test]
fn publication_failures_preserve_previous_manifest_cursor_and_restore() {
    const FAILURES: &[(&str, bool)] = &[
        ("backup:segment_write_failure", true),
        ("backup:manifest_write", true),
        ("backup:manifest_rename_failure", false),
        ("backup:manifest_parent_sync", false),
        ("backup:cursor_write", true),
        ("backup:cursor_rename_failure", false),
        ("backup:cursor_parent_sync", false),
    ];
    for &(point, storage_full) in FAILURES {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.grafeo");
        let backup_dir = dir.path().join("backup");
        let db = GrafeoDB::open(&source_path).unwrap();
        populate(&db, 1);
        let full = db.backup_full(&backup_dir).unwrap();
        populate(&db, 2);
        let manifest_path = backup_dir.join("backup_manifest.json");
        let cursor_path =
            std::path::PathBuf::from(format!("{}.wal/backup_cursor.meta", source_path.display()));
        let manifest_before = std::fs::read(&manifest_path).unwrap();
        let cursor_before = std::fs::read(&cursor_path).unwrap();
        let result =
            grafeo_common::testing::wal_failure::with_backup_publication_failure(point, || {
                db.backup_incremental(&backup_dir)
            });
        let error = result.expect_err("injected backup publication failure must be returned");
        if storage_full {
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::StorageFull
            );
        } else {
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::IoError
            );
        }
        drop(db);

        let reopened = GrafeoDB::open(&source_path).unwrap();
        assert_eq!(std::fs::read(&manifest_path).unwrap(), manifest_before);
        assert_eq!(std::fs::read(&cursor_path).unwrap(), cursor_before);
        assert_eq!(
            restored_values(
                &backup_dir,
                full.end_epoch,
                &dir.path().join("failure-restored.grafeo"),
            ),
            vec![vec![Value::Int64(1)]]
        );
        reopened.close().unwrap();
    }
}

#[test]
fn backup_chain_crash_matrix() {
    for phase in ["full", "incremental"] {
        for point in FAILPOINTS {
            let dir = tempfile::tempdir().unwrap();
            let source_path = dir.path().join("source.grafeo");
            let backup_dir = dir.path().join("backup");
            let db = GrafeoDB::open(&source_path).unwrap();
            populate(&db, 1);
            let baseline = db.backup_full(&backup_dir).unwrap();
            db.close().unwrap();

            let status = Command::new(std::env::current_exe().unwrap())
                .args(["backup_chain_crash_child", "--exact", "--nocapture"])
                .env("GRAFEO_BACKUP_CRASH_ROLE", "child")
                .env("GRAFEO_BACKUP_CRASH_PATH", &source_path)
                .env("GRAFEO_BACKUP_CRASH_DIR", &backup_dir)
                .env("GRAFEO_BACKUP_CRASH_PHASE", phase)
                .env("GRAFEO_BACKUP_CRASH_POINT", point)
                .status()
                .unwrap();
            assert_eq!(
                status.code(),
                Some(86),
                "backup child did not crash at {phase}/{point}: {status}"
            );

            let source = GrafeoDB::open(&source_path).unwrap();
            let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
                .unwrap()
                .expect("a prior or committed next manifest must remain readable");
            let last = manifest.segments.last().unwrap();
            let baseline_rows = restored_values(
                &backup_dir,
                baseline.end_epoch,
                &dir.path().join("baseline-restored.grafeo"),
            );
            assert_eq!(baseline_rows, vec![vec![Value::Int64(1)]]);
            let expected_rows = if *point == "backup:commit_directory_sync" {
                vec![vec![Value::Int64(1)], vec![Value::Int64(2)]]
            } else {
                vec![vec![Value::Int64(1)]]
            };
            let rows = restored_values(
                &backup_dir,
                last.end_epoch,
                &dir.path().join("last-committed-restored.grafeo"),
            );
            assert_eq!(
                rows, expected_rows,
                "unexpected committed generation at {point}"
            );

            // A subsequent caller may either complete a new generation or
            // report structured incomplete-generation state, but must never
            // publish a cursor that disagrees with the manifest.
            populate(&source, 3);
            match source.backup_incremental(&backup_dir) {
                Ok(_) => {
                    let after = GrafeoDB::read_backup_manifest(&backup_dir)
                        .unwrap()
                        .unwrap();
                    let cursor = source.backup_cursor().unwrap().unwrap();
                    assert_eq!(cursor.chain_id, after.chain_id);
                    assert_eq!(cursor.generation, after.generation);
                    assert_eq!(
                        cursor.backed_up_epoch,
                        after.segments.last().unwrap().end_epoch
                    );
                }
                Err(error) => {
                    let text = error.to_string().to_ascii_lowercase();
                    assert!(
                        text.contains("generation")
                            || text.contains("cursor")
                            || text.contains("incomplete"),
                        "unexpected post-crash backup error: {error}"
                    );
                }
            }
            source.close().unwrap();
        }
    }
}

#[test]
fn restore_replacement_crash_matrix_leaves_old_or_complete_output() {
    for point in RESTORE_FAILPOINTS {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.grafeo");
        let backup_dir = dir.path().join("backup");
        let output = dir.path().join("output.grafeo");
        let db = GrafeoDB::open(&source_path).unwrap();
        populate(&db, 1);
        db.backup_full(&backup_dir).unwrap();
        populate(&db, 2);
        let increment = db.backup_incremental(&backup_dir).unwrap();
        db.close().unwrap();
        let old = GrafeoDB::open(&output).unwrap();
        populate(&old, 99);
        old.close().unwrap();

        let status = Command::new(std::env::current_exe().unwrap())
            .args(["backup_chain_crash_child", "--exact", "--nocapture"])
            .env("GRAFEO_BACKUP_CRASH_ROLE", "child")
            .env("GRAFEO_BACKUP_CRASH_PATH", &source_path)
            .env("GRAFEO_BACKUP_CRASH_DIR", &backup_dir)
            .env("GRAFEO_BACKUP_CRASH_OUTPUT", &output)
            .env(
                "GRAFEO_BACKUP_CRASH_TARGET",
                increment.end_epoch.as_u64().to_string(),
            )
            .env("GRAFEO_BACKUP_CRASH_PHASE", "restore")
            .env("GRAFEO_BACKUP_CRASH_POINT", point)
            .status()
            .unwrap();
        assert_eq!(
            status.code(),
            Some(86),
            "restore child did not crash at {point}"
        );

        let rows = restored_values_from_output(&output);
        let old_rows = vec![vec![Value::Int64(99)]];
        let new_rows = vec![vec![Value::Int64(1)], vec![Value::Int64(2)]];
        if *point == "backup:restore_before_replace" {
            assert_eq!(rows, old_rows);
        } else {
            assert_eq!(rows, new_rows);
        }
        assert!(!output.with_extension("grafeo.wal").exists());
    }
}

fn restored_values_from_output(output: &Path) -> Vec<Vec<Value>> {
    let db = GrafeoDB::open(output).unwrap();
    let rows = db
        .execute("MATCH (n:N) RETURN n.n ORDER BY n.n")
        .unwrap()
        .rows()
        .to_vec();
    db.close().unwrap();
    rows
}

#[test]
fn backup_chain_crash_child() {
    if std::env::var("GRAFEO_BACKUP_CRASH_ROLE").as_deref() != Ok("child") {
        return;
    }
    let source_path = std::env::var("GRAFEO_BACKUP_CRASH_PATH").unwrap();
    let backup_dir = std::env::var("GRAFEO_BACKUP_CRASH_DIR").unwrap();
    let phase = std::env::var("GRAFEO_BACKUP_CRASH_PHASE").unwrap();
    let point = std::env::var("GRAFEO_BACKUP_CRASH_POINT").unwrap();

    let expected_panic = format!("crash injection at: {point}");
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str));
        if payload == Some(expected_panic.as_str()) {
            std::process::exit(86);
        }
        eprintln!("unexpected backup crash child panic: {info}");
        std::process::exit(88);
    }));
    grafeo_common::testing::crash::enable_crash_named(Box::leak(point.into_boxed_str()));
    if phase == "restore" {
        let output = std::env::var("GRAFEO_BACKUP_CRASH_OUTPUT").unwrap();
        let target = std::env::var("GRAFEO_BACKUP_CRASH_TARGET")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        GrafeoDB::open(source_path).unwrap();
        GrafeoDB::restore_to_epoch(
            Path::new(&backup_dir),
            grafeo_common::types::EpochId::new(target),
            Path::new(&output),
        )
        .unwrap();
        std::process::exit(87);
    }
    let db = GrafeoDB::open(source_path).unwrap();
    populate(&db, 2);
    match phase.as_str() {
        "full" => {
            db.backup_full(Path::new(&backup_dir)).unwrap();
        }
        "incremental" => {
            db.backup_incremental(Path::new(&backup_dir)).unwrap();
        }
        _ => panic!("unknown backup crash phase {phase}"),
    }
    std::process::exit(87);
}
