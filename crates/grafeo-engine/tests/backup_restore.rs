//! Integration tests for incremental backup and point-in-time restore.
//!
//! Covers: full backup, incremental backup, restore to epoch, and the
//! backup chain model.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test backup_restore
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]

use grafeo_common::types::EpochId;
use grafeo_engine::GrafeoDB;

#[test]
fn independent_full_backups_mint_distinct_v2_chains() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let mut manifests = Vec::new();
    for name in ["a", "b"] {
        let backup_dir = dir.path().join(name);
        let full = db.backup_full(&backup_dir).unwrap();
        let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
            .unwrap()
            .unwrap();
        assert_eq!(
            manifest.version, 2,
            "full backups must write current v2 metadata"
        );
        manifests.push(manifest);
        let restored_path = dir.path().join(format!("restored-{name}.grafeo"));
        GrafeoDB::restore_to_epoch(&backup_dir, full.end_epoch, &restored_path).unwrap();
        let restored = GrafeoDB::open(&restored_path).unwrap();
        assert_eq!(restored.node_count(), 1);
        restored.close().unwrap();
    }
    assert_ne!(manifests[0].chain_id, manifests[1].chain_id);
    assert_eq!(manifests[0].store_id, manifests[1].store_id);
    db.close().unwrap();
}

#[test]
fn cursor_from_another_chain_cannot_advance_backup() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    db.backup_full(&a).unwrap();
    db.backup_full(&b).unwrap();
    // Corrupt A's authoritative namespace with B's independently valid cursor.
    // Merely selecting B most recently is now legitimate: locators are advisory.
    let cursor_path = |backup_dir: &std::path::Path| {
        let manifest = GrafeoDB::read_backup_manifest(backup_dir).unwrap().unwrap();
        use std::fmt::Write as _;
        let mut chain = String::with_capacity(64);
        for byte in manifest.chain_id {
            write!(chain, "{byte:02x}").unwrap();
        }
        dir.path().join("source.grafeo.wal").join(format!(
            "backup_cursor_{chain}_{:020}.meta",
            manifest.generation
        ))
    };
    std::fs::copy(cursor_path(&b), cursor_path(&a)).unwrap();
    db.execute("INSERT (:N {n: 2})").unwrap();
    let before = std::fs::read(a.join("backup_manifest.json")).unwrap();
    let error = db.backup_incremental(&a).unwrap_err();
    assert!(
        error.to_string().contains("cursor does not match"),
        "{error}"
    );
    assert_eq!(
        std::fs::read(a.join("backup_manifest.json")).unwrap(),
        before
    );
    assert!(!a.join("backup_incr_0001.wal").exists());
    db.backup_incremental(&b).unwrap();
    db.close().unwrap();
}

#[test]
fn incremental_metadata_binds_exact_frames_and_ending_world_cut() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let backup = dir.path().join("backup");
    let full = db.backup_full(&backup).unwrap();
    for n in 2..5 {
        db.execute(&format!("INSERT (:N {{n: {n}}})")).unwrap();
    }
    let increment = db.backup_incremental(&backup).unwrap();
    assert_eq!(increment.chain_id, full.chain_id);
    assert_eq!(increment.store_id, full.store_id);
    assert_eq!(increment.model, full.model);
    assert_eq!((full.sequence, increment.sequence), (0, 1));
    assert_eq!(increment.wal_start_sequence, full.wal_end_sequence + 1);
    assert!(increment.wal_end_sequence >= increment.wal_start_sequence);
    assert_ne!(increment.predecessor_digest, [0; 32]);
    assert_eq!(full.record_count, 0);
    assert!(increment.record_count > 3, "count frames, not WAL files");
    let bytes = std::fs::read(backup.join(&increment.filename)).unwrap();
    let (_, _, header_count) = grafeo_engine::database::backup::read_backup_header(&bytes).unwrap();
    assert_eq!(header_count, increment.record_count);
    assert_eq!(
        grafeo_storage::wal::count_wal_frames(&bytes[32..]).unwrap(),
        header_count
    );
    assert_eq!(
        increment.world_cut.as_ref().unwrap(),
        &db.world_cut().unwrap().encode().unwrap()
    );
    let manifest = GrafeoDB::read_backup_manifest(&backup).unwrap().unwrap();
    assert_eq!(manifest.world_cut, increment.world_cut);
    let restored = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup, increment.end_epoch, &restored).unwrap();
    let restored = GrafeoDB::open(&restored).unwrap();
    assert_eq!(restored.node_count(), 4);
    restored.close().unwrap();
    db.close().unwrap();
}

#[test]
fn incremental_capture_preserves_committed_groups_across_rotations() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 0})").unwrap();
    let backup = dir.path().join("backup");
    let full = db.backup_full(&backup).unwrap();
    for n in 1..5 {
        db.execute(&format!("INSERT (:N {{n: {n}}})")).unwrap();
        db.wal().unwrap().rotate().unwrap();
    }
    let end = db.wal().unwrap().current_sequence();
    let increment = db.backup_incremental(&backup).unwrap();
    assert_eq!(increment.wal_start_sequence, full.wal_end_sequence + 1);
    assert_eq!(increment.wal_end_sequence, end);
    let restored_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup, increment.end_epoch, &restored_path).unwrap();
    let restored = GrafeoDB::open(&restored_path).unwrap();
    let rows = restored
        .execute("MATCH (n:N) RETURN n.n ORDER BY n.n")
        .unwrap();
    let expected: Vec<Vec<grafeo_common::types::Value>> = (0..5)
        .map(|n| vec![grafeo_common::types::Value::Int64(n)])
        .collect();
    assert_eq!(rows.rows(), expected);
    restored.close().unwrap();
    db.close().unwrap();
}

#[test]
fn retired_interval_cannot_advance_backup_cursor_or_publish_segment() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 0})").unwrap();
    let backup = dir.path().join("backup");
    db.backup_full(&backup).unwrap();
    for n in 1..7 {
        db.execute(&format!("INSERT (:N {{n: {n}}})")).unwrap();
        db.wal_checkpoint().unwrap();
    }
    let cursor = db.backup_cursor().unwrap();
    let sequence = db.wal().unwrap().current_sequence();
    let manifest = std::fs::read(backup.join("backup_manifest.json")).unwrap();
    assert!(db.backup_incremental(&backup).is_err());
    assert_eq!(db.backup_cursor().unwrap(), cursor);
    assert_eq!(db.wal().unwrap().current_sequence(), sequence);
    assert_eq!(
        std::fs::read(backup.join("backup_manifest.json")).unwrap(),
        manifest
    );
    assert!(!backup.join("backup_incr_0001.wal").exists());
    db.close().unwrap();
}

#[test]
fn empty_increment_does_not_invent_an_epoch_or_advance_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N)").unwrap();
    let backup = dir.path().join("backup");
    db.backup_full(&backup).unwrap();
    db.wal().unwrap().rotate().unwrap();
    let cursor = db.backup_cursor().unwrap();
    let sequence = db.wal().unwrap().current_sequence();
    assert!(db.backup_incremental(&backup).is_err());
    assert_eq!(db.backup_cursor().unwrap(), cursor);
    assert_eq!(db.wal().unwrap().current_sequence(), sequence);
    db.close().unwrap();
}

#[test]
fn foreign_full_backup_image_cannot_mutate_restore_destination() {
    let dir = tempfile::tempdir().unwrap();
    let first = GrafeoDB::open(dir.path().join("first.grafeo")).unwrap();
    first.execute("INSERT (:N {n: 1})").unwrap();
    let first_dir = dir.path().join("first-backup");
    let full = first.backup_full(&first_dir).unwrap();
    let second = GrafeoDB::open(dir.path().join("second.grafeo")).unwrap();
    second.execute("INSERT (:N {n: 2}), (:N {n: 3})").unwrap();
    let second_dir = dir.path().join("second-backup");
    let other = second.backup_full(&second_dir).unwrap();
    std::fs::copy(
        second_dir.join(other.filename),
        first_dir.join(&full.filename),
    )
    .unwrap();

    let destination = dir.path().join("destination.grafeo");
    let sentinel = GrafeoDB::open(&destination).unwrap();
    sentinel.execute("INSERT (:Sentinel {keep: 7})").unwrap();
    sentinel.close().unwrap();
    let before = std::fs::read(&destination).unwrap();
    assert!(GrafeoDB::restore_to_epoch(&first_dir, full.end_epoch, &destination).is_err());
    assert_eq!(std::fs::read(&destination).unwrap(), before);

    // Matching checksums alone cannot substitute another store's image for
    // the cut and identity recorded by this chain.
    let mut forged = GrafeoDB::read_backup_manifest(&first_dir).unwrap().unwrap();
    forged.segments[0].checksum = other.checksum;
    forged.segments[0].content_digest = other.content_digest;
    forged.segments[0].size_bytes = other.size_bytes;
    grafeo_engine::database::backup::write_manifest(&first_dir, &forged).unwrap();
    assert!(GrafeoDB::restore_to_epoch(&first_dir, full.end_epoch, &destination).is_err());
    assert_eq!(std::fs::read(&destination).unwrap(), before);

    let reopened = GrafeoDB::open(&destination).unwrap();
    assert_eq!(
        reopened
            .execute("MATCH (n:Sentinel) RETURN n.keep")
            .unwrap()
            .rows(),
        [[grafeo_common::types::Value::Int64(7)]]
    );
    reopened.close().unwrap();
    first.close().unwrap();
    second.close().unwrap();
}

#[test]
fn missing_manifest_with_full_image_is_not_an_empty_backup() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let backup_dir = dir.path().join("backup");
    db.backup_full(&backup_dir).unwrap();
    // Removing only the advisory locator is recoverable from the committed pair.
    for entry in std::fs::read_dir(&backup_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "backup_manifest.json"
            || (name.starts_with("backup_manifest_") && name.ends_with(".meta"))
        {
            std::fs::remove_file(entry.path()).unwrap();
        }
    }
    assert!(GrafeoDB::read_backup_manifest(&backup_dir).is_err());
    let mut before = std::fs::read_dir(&backup_dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), std::fs::read(entry.path()).unwrap())
        })
        .collect::<Vec<_>>();
    before.sort();
    assert!(db.backup_full(&backup_dir).is_err());
    let mut after = std::fs::read_dir(&backup_dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), std::fs::read(entry.path()).unwrap())
        })
        .collect::<Vec<_>>();
    after.sort();
    assert_eq!(after, before);
    db.close().unwrap();
}

#[test]
fn exhausted_backup_generation_leaves_chain_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let backup_dir = dir.path().join("backup");
    db.backup_full(&backup_dir).unwrap();
    let mut manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    manifest.generation = u64::MAX;
    grafeo_engine::database::backup::write_manifest(&backup_dir, &manifest).unwrap();
    let image = || {
        let mut files = std::fs::read_dir(&backup_dir)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), std::fs::read(entry.path()).unwrap())
            })
            .collect::<Vec<_>>();
        files.sort();
        files
    };
    let before = image();
    assert!(db.backup_full(&backup_dir).is_err());
    assert_eq!(image(), before);
    db.close().unwrap();
}

#[test]
fn malformed_v2_metadata_rejects_before_creating_restore_destination() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let backup_dir = dir.path().join("backup");
    let full = db.backup_full(&backup_dir).unwrap();
    let manifest_path = backup_dir.join("backup_manifest.json");
    let valid = std::fs::read(&manifest_path).unwrap();
    let mut predecessor = valid[..17].to_vec();
    predecessor[4..8].copy_from_slice(&1_u32.to_le_bytes());
    let mut oversized = valid.clone();
    oversized[9..17].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut trailing = valid.clone();
    trailing.push(0);
    let cases = [
        ("predecessor", predecessor),
        ("oversized", oversized),
        ("trailing", trailing),
        ("truncated", valid[..3].to_vec()),
    ];
    for (name, bytes) in cases {
        std::fs::write(&manifest_path, bytes).unwrap();
        let error = GrafeoDB::read_backup_manifest(&backup_dir).unwrap_err();
        if name == "predecessor" {
            assert!(
                error
                    .to_string()
                    .contains("unsupported backup metadata version 1")
            );
        }
        let destination = dir.path().join(format!("{name}.grafeo"));
        assert!(GrafeoDB::restore_to_epoch(&backup_dir, full.end_epoch, &destination).is_err());
        assert!(
            !destination.exists(),
            "{name} must not create the destination"
        );
    }
    db.close().unwrap();
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn v2_full_backup_restores_rdf_and_both_models() {
    use grafeo_engine::{Config, GraphModel};

    let dir = tempfile::tempdir().unwrap();
    for (name, model) in [("rdf", GraphModel::Rdf), ("both", GraphModel::Both)] {
        let source = dir.path().join(format!("{name}.grafeo"));
        let db =
            GrafeoDB::with_config(Config::persistent(&source).with_graph_model(model)).unwrap();
        db.execute_sparql(
            "INSERT DATA { <http://ex/s> <http://ex/p> \"default\" . \
             GRAPH <http://ex/g> { <http://ex/s> <http://ex/p> \"named\" . } }",
        )
        .unwrap();
        if model == GraphModel::Both {
            db.execute("INSERT (:N {n: 1})").unwrap();
        }
        let queries = [
            "SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o }",
            "SELECT ?o WHERE { GRAPH <http://ex/g> { <http://ex/s> <http://ex/p> ?o } }",
        ];
        let expected = queries.map(|query| db.execute_sparql(query).unwrap().rows().to_vec());
        assert!(expected.iter().all(|rows| rows.len() == 1));
        let backup_dir = dir.path().join(format!("{name}-backup"));
        let full = db.backup_full(&backup_dir).unwrap();
        let destination = dir.path().join(format!("{name}-restored.grafeo"));
        GrafeoDB::restore_to_epoch(&backup_dir, full.end_epoch, &destination).unwrap();
        let restored = GrafeoDB::open(&destination).unwrap();
        assert_eq!(restored.graph_model(), model);
        for (query, rows) in queries.into_iter().zip(expected) {
            assert_eq!(restored.execute_sparql(query).unwrap().rows(), rows);
        }
        if model == GraphModel::Both {
            assert_eq!(restored.node_count(), 1);
        }
        restored.close().unwrap();
        db.close().unwrap();
    }
}

// ── Full backup roundtrip ─────────────────────────────────────────

#[test]
fn full_backup_and_restore_to_epoch() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("source.grafeo");
    let backup_dir = dir.path().join("backups");
    let restore_path = dir.path().join("restored.grafeo");

    // Create and populate
    {
        let db = GrafeoDB::open(&db_path).expect("open");
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', batch: 1})")
            .expect("insert");
        session
            .execute("INSERT (:Person {name: 'Gus', batch: 1})")
            .expect("insert");
        db.close().expect("close");
    }

    // Take a full backup
    let db = GrafeoDB::open(&db_path).expect("reopen");
    let segment = db.backup_full(&backup_dir).expect("full backup");
    assert_eq!(segment.start_epoch, EpochId::new(0));
    assert!(segment.size_bytes > 0);

    let current_epoch = segment.end_epoch;
    db.close().expect("close");

    // Restore to the full backup epoch
    GrafeoDB::restore_to_epoch(&backup_dir, current_epoch, &restore_path)
        .expect("restore to epoch");

    // Verify restored data
    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2, "restored should have 2 nodes");
    let session = restored.session();
    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    restored.close().expect("close");
}

// ── Full + incremental backup cycle ───────────────────────────────

#[test]
fn incremental_backup_captures_new_data() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("incr.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");

    // Initial data
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");

    // Full backup
    let full = db.backup_full(&backup_dir).expect("full backup");
    assert!(full.size_bytes > 0);

    // Add more data after the full backup
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect("insert");

    // Force a WAL rotation so incremental has new files to capture
    db.wal().expect("WAL").rotate().expect("rotate");
    session
        .execute("INSERT (:Person {name: 'Jules'})")
        .expect("insert");

    // Incremental backup
    let incr = db
        .backup_incremental(&backup_dir)
        .expect("incremental backup");
    assert!(incr.size_bytes > 0);
    assert!(incr.start_epoch > full.end_epoch);

    db.close().expect("close");

    // Verify manifest has both segments
    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .expect("read manifest")
        .expect("manifest exists");
    assert_eq!(manifest.segments.len(), 2);
}

// ── Backup cursor tracking ────────────────────────────────────────

#[test]
fn backup_cursor_updated_after_full_backup() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("cursor.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");

    // Use a session to advance the epoch beyond 0
    let session = db.session();
    session.execute("INSERT (:Test {val: 1})").expect("insert");

    assert!(
        db.backup_cursor().unwrap().is_none(),
        "no cursor before first backup"
    );

    db.backup_full(&backup_dir).expect("full backup");

    let cursor = db
        .backup_cursor()
        .expect("cursor read")
        .expect("cursor should exist after backup");
    assert!(
        cursor.backed_up_epoch.as_u64() > 0,
        "epoch should be > 0 after session commit"
    );

    db.close().expect("close");
}

// ── Backup manifest metadata ──────────────────────────────────────

#[test]
fn backup_manifest_tracks_segments() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("meta.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    db.create_node(&["Test"]);

    let segment = db.backup_full(&backup_dir).expect("full backup");

    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].filename, segment.filename);
    assert_eq!(manifest.epoch_range().unwrap().1, segment.end_epoch);

    db.close().expect("close");
}

// ── Error cases (all platforms) ───────────────────────────────────

#[test]
fn incremental_without_full_fails() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("nofull.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    db.create_node(&["Test"]);

    let result = db.backup_incremental(&backup_dir);
    assert!(result.is_err(), "incremental without full should fail");

    db.close().expect("close");
}

#[test]
fn restore_nonexistent_backup_fails() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let backup_dir = dir.path().join("empty_backups");
    let restore_path = dir.path().join("restored.grafeo");

    let result = GrafeoDB::restore_to_epoch(&backup_dir, EpochId::new(100), &restore_path);
    assert!(result.is_err(), "restore from empty dir should fail");
}

// ── Bug regression tests ─────────────────────────────────────────

/// Regression: backup_full() on a read-only database should succeed.
///
/// The on-disk `.grafeo` file is already a valid snapshot, so there is
/// nothing to flush. Previously, backup_full() unconditionally called
/// checkpoint_to_file() which rejects writes on read-only file managers.
#[test]
fn backup_full_on_read_only_database() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("readonly_backup.grafeo");
    let backup_dir = dir.path().join("backups");

    // Create and populate a database, then close it so the file is complete.
    {
        let db = GrafeoDB::open(&db_path).expect("open");
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})")
            .expect("insert");
        session
            .execute("INSERT (:Person {name: 'Gus'})")
            .expect("insert");
        db.close().expect("close");
    }

    // Re-open in read-only mode and take a full backup.
    let db = GrafeoDB::open_read_only(&db_path).expect("open read-only");
    let segment = db
        .backup_full(&backup_dir)
        .expect("backup_full on read-only should succeed");

    assert_eq!(segment.start_epoch, EpochId::new(0));
    assert!(segment.size_bytes > 0, "backup file should not be empty");

    // Verify the backup is a valid database by restoring and querying.
    let restore_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, segment.end_epoch, &restore_path)
        .expect("restore should succeed");

    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2, "restored should have 2 nodes");
    restored.close().expect("close");
}

/// Regression: backup_full() must work on Windows where the .grafeo file
/// is held open with an exclusive lock.
///
/// Previously, do_backup_full() used std::fs::copy() which tries to open
/// the source file with a new handle. On Windows, that fails because the
/// GrafeoFileManager already holds an exclusive lock.
#[test]
fn backup_full_works_while_database_is_open() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("open_backup.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");

    // This should succeed on ALL platforms, including Windows.
    let segment = db
        .backup_full(&backup_dir)
        .expect("backup_full on open database should work on all platforms");

    assert_eq!(segment.start_epoch, EpochId::new(0));
    assert!(segment.size_bytes > 0);

    db.close().expect("close");

    // Verify the backup is restorable.
    let restore_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, segment.end_epoch, &restore_path)
        .expect("restore should succeed");

    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2, "restored should have 2 nodes");
    restored.close().expect("close");
}

fn sentinel_rows(path: &std::path::Path) -> Vec<Vec<grafeo_common::types::Value>> {
    let db = GrafeoDB::open(path).unwrap();
    let rows = db
        .execute("MATCH (n) RETURN n.n ORDER BY n.n")
        .unwrap()
        .rows()
        .to_vec();
    db.close().unwrap();
    rows
}

#[test]
fn restore_validation_failure_preserves_populated_destination() {
    for missing in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.grafeo");
        let backup = dir.path().join("backup");
        let output = dir.path().join("output.grafeo");
        let db = GrafeoDB::open(&source).unwrap();
        db.execute("INSERT (:N {n: 1})").unwrap();
        let full = db.backup_full(&backup).unwrap();
        db.execute("INSERT (:N {n: 2})").unwrap();
        let increment = db.backup_incremental(&backup).unwrap();
        db.close().unwrap();
        let sentinel = GrafeoDB::open(&output).unwrap();
        sentinel.execute("INSERT (:Sentinel {n: 99})").unwrap();
        sentinel.close().unwrap();
        let before = sentinel_rows(&output);
        let before_bytes = std::fs::read(&output).unwrap();
        if missing {
            std::fs::remove_file(backup.join(&increment.filename)).unwrap();
        } else {
            let mut bytes = std::fs::read(backup.join(&increment.filename)).unwrap();
            let middle = bytes.len() / 2;
            bytes[middle] ^= 1;
            std::fs::write(backup.join(&increment.filename), bytes).unwrap();
        }
        assert!(GrafeoDB::restore_to_epoch(&backup, increment.end_epoch, &output).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), before_bytes);
        assert_eq!(sentinel_rows(&output), before);
        assert_eq!(full.record_count, 0);
    }
}

#[test]
fn restore_rejects_future_and_pending_targets_before_destination_work() {
    let dir = tempfile::tempdir().unwrap();
    let backup = dir.path().join("backup");
    let output = dir.path().join("output.grafeo");
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let full = db.backup_full(&backup).unwrap();
    db.close().unwrap();
    for target in [EpochId::new(full.end_epoch.as_u64() + 1), EpochId::PENDING] {
        assert!(GrafeoDB::restore_to_epoch(&backup, target, &output).is_err());
        assert!(!output.exists());
    }
}

#[test]
fn restore_partial_increment_replays_exact_target_group() {
    let dir = tempfile::tempdir().unwrap();
    let backup = dir.path().join("backup");
    let output = dir.path().join("output.grafeo");
    let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    db.backup_full(&backup).unwrap();
    db.execute("INSERT (:N {n: 2})").unwrap();
    let first = db.world_cut().unwrap().epoch();
    db.execute("INSERT (:N {n: 3})").unwrap();
    let increment = db.backup_incremental(&backup).unwrap();
    assert!(first < increment.end_epoch);
    GrafeoDB::restore_to_epoch(&backup, first, &output).unwrap();
    assert_eq!(sentinel_rows(&output).len(), 2);
    db.close().unwrap();
}

#[test]
fn restore_rejects_source_alias_and_held_open_output() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.grafeo");
    let backup = dir.path().join("backup");
    let db = GrafeoDB::open(&source).unwrap();
    db.execute("INSERT (:N {n: 1})").unwrap();
    let full = db.backup_full(&backup).unwrap();
    db.close().unwrap();
    let image = backup.join(&full.filename);
    let before = std::fs::read(&image).unwrap();
    assert!(GrafeoDB::restore_to_epoch(&backup, full.end_epoch, &image).is_err());
    assert_eq!(std::fs::read(&image).unwrap(), before);
    let held = GrafeoDB::open(&source).unwrap();
    assert!(GrafeoDB::restore_to_epoch(&backup, full.end_epoch, &source).is_err());
    held.close().unwrap();
}

#[cfg(unix)]
#[test]
fn restore_rejects_symlink_and_hardlink_backup_images() {
    use std::fs::hard_link;
    use std::os::unix::fs::symlink;
    for hard in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.grafeo");
        let backup = dir.path().join("backup");
        let foreign = dir.path().join("foreign.grafeo");
        let output = dir.path().join("output.grafeo");
        let db = GrafeoDB::open(&source).unwrap();
        db.execute("INSERT (:N {n: 1})").unwrap();
        let full = db.backup_full(&backup).unwrap();
        db.close().unwrap();
        std::fs::copy(&source, &foreign).unwrap();
        let image = backup.join(&full.filename);
        std::fs::remove_file(&image).unwrap();
        if hard {
            hard_link(&foreign, &image).unwrap();
        } else {
            symlink(&foreign, &image).unwrap();
        }
        assert!(GrafeoDB::restore_to_epoch(&backup, full.end_epoch, &output).is_err());
        assert!(!output.exists());
    }
}

/// Two full backups into the same directory produce two distinct segments
/// in the manifest. The second backup does not overwrite the first.
#[test]
fn backup_full_twice_produces_two_segments() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("double.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");

    let seg1 = db.backup_full(&backup_dir).expect("first backup");
    assert_eq!(seg1.filename, "backup_full_0000.grafeo");

    // Add more data between backups
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");

    let seg2 = db.backup_full(&backup_dir).expect("second backup");
    assert_eq!(seg2.filename, "backup_full_0001.grafeo");
    assert!(seg2.end_epoch >= seg1.end_epoch);

    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.segments.len(), 2);

    // Each full image retains its own cut, even after a later full backup.
    let older_path = dir.path().join("older.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, seg1.end_epoch, &older_path)
        .expect("restore older full backup");
    let older = GrafeoDB::open(&older_path).expect("open older backup");
    assert_eq!(older.node_count(), 1);
    older.close().expect("close older backup");

    // Restore from the second backup (latest state)
    let restore_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, seg2.end_epoch, &restore_path)
        .expect("restore should succeed");

    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2);
    restored.close().expect("close");

    db.close().expect("close");
}

/// Regression for GrafeoDB/grafeo#267: incremental backup must succeed
/// after a full backup without requiring a manual WAL rotation.
///
/// Previously, `do_backup_full` stored the active log file's sequence in the
/// cursor but did not rotate, so writes that landed in the same file were
/// invisible to incremental (which skips `seq <= cursor.log_sequence`).
#[test]
fn incremental_backup_works_without_manual_rotation() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("issue267.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();

    // Seed data and take a full backup
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    let full = db.backup_full(&backup_dir).expect("full backup");

    // Insert more data WITHOUT manually rotating the WAL
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect("insert");

    // Incremental must succeed (no manual wal.rotate() call)
    let incr = db
        .backup_incremental(&backup_dir)
        .expect("incremental backup should work without manual WAL rotation");
    assert!(incr.size_bytes > 0);
    assert!(incr.start_epoch > full.end_epoch);

    db.close().expect("close");
}

/// Regression for GrafeoDB/grafeo#267: two consecutive incremental backups
/// with writes between them must both succeed.
///
/// This tests the same boundary condition in `do_backup_incremental` itself:
/// the cursor it writes must not block the next incremental from seeing new
/// WAL data.
#[test]
fn multiple_incremental_backups_in_sequence() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("multi_incr.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();

    // Seed + full backup
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    db.backup_full(&backup_dir).expect("full backup");

    // First incremental
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    let incr1 = db
        .backup_incremental(&backup_dir)
        .expect("first incremental");

    // Second incremental (no manual rotation between them)
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect("insert");
    let incr2 = db
        .backup_incremental(&backup_dir)
        .expect("second incremental should also succeed");
    assert!(incr2.start_epoch > incr1.start_epoch);

    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    assert_eq!(
        manifest.segments.len(),
        3,
        "should have 1 full + 2 incremental segments"
    );

    db.close().expect("close");
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn restore_install_io_failures_leave_old_or_complete_image() {
    use grafeo_common::testing::wal_failure::with_backup_publication_failure;
    for point in [
        "backup:restore_before_replace",
        "backup:restore_after_replace",
        "backup:restore_parent_sync",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup");
        let source = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
        source.execute("INSERT (:N {n: 1})").unwrap();
        source.backup_full(&backup).unwrap();
        source.execute("INSERT (:N {n: 2})").unwrap();
        let increment = source.backup_incremental(&backup).unwrap();
        let expected = source.world_cut().unwrap();
        let output = dir.path().join("output.grafeo");
        let sentinel = GrafeoDB::open(&output).unwrap();
        sentinel.execute("INSERT (:Sentinel {n: 99})").unwrap();
        let old_cut = sentinel.world_cut().unwrap();
        sentinel.close().unwrap();
        let before = std::fs::read(&output).unwrap();
        let error = with_backup_publication_failure(point, || {
            GrafeoDB::restore_to_epoch(&backup, increment.end_epoch, &output)
        })
        .unwrap_err();
        let restored = GrafeoDB::open_read_only(&output).unwrap();
        if point == "backup:restore_before_replace" {
            assert_eq!(std::fs::read(&output).unwrap(), before);
            assert_eq!(restored.world_cut().unwrap(), old_cut);
        } else {
            assert!(error.to_string().contains("published"), "{point}: {error}");
            assert_eq!(restored.world_cut().unwrap(), expected);
        }
        assert!(!output.with_extension("grafeo.wal").exists());
        restored.close().unwrap();
        source.close().unwrap();
    }
}
