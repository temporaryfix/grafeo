//! Recovery authority must be established before replay or torn-tail repair.
#![cfg(all(
    feature = "wal",
    feature = "triple-store",
    feature = "sparql",
    feature = "grafeo-file"
))]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use grafeo_common::types::{
    EpochId, GraphIncarnationId, HistoryCompleteness, TransactionId, WorldIdentityMetadataV1,
};
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::wal::{LpgWal, WalRecord};

fn config(path: &Path, format: StorageFormat) -> Config {
    Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_storage_format(format)
        .with_wal_durability(DurabilityMode::Sync)
}

fn files(path: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn collect(root: &Path, path: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(path).unwrap(),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    collect(path, path, &mut out);
    out
}

fn append_torn_tail(wal_path: &Path) {
    let segment = std::fs::read_dir(wal_path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "log"))
        .max()
        .unwrap();
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(segment)
        .unwrap();
    file.write_all(&[0x80, 0x01]).unwrap();
    file.sync_all().unwrap();
}

fn mutation() -> WalRecord {
    WalRecord::InsertRdfQuadV3 {
        subject: "<http://example/s>".into(),
        predicate: "<http://example/p>".into(),
        object: "\"value\"".into(),
        graph: None,
        graph_incarnation: GraphIncarnationId::DEFAULT_GRAPH,
        valid_from_tai_ns: None,
        valid_to_tai_ns: None,
        transaction_id: TransactionId::new(7),
    }
}

fn assert_rejected_unchanged(root: &Path, database: &Path, format: StorageFormat) {
    let before = files(root);
    let result = GrafeoDB::with_config(config(database, format));
    assert!(
        result.is_err(),
        "unattributed or conflicting WAL must be rejected"
    );
    assert!(
        files(root) == before,
        "rejected recovery must not alter source files"
    );
}

#[test]
fn identityless_committed_wal_with_torn_suffix_is_rejected_without_repair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let wal_path = path.join("wal");
    let wal = LpgWal::open(&wal_path).unwrap();
    wal.log(&WalRecord::GraphModelMeta { model: 1 }).unwrap();
    wal.log(&mutation()).unwrap();
    wal.log(&WalRecord::Committed {
        transaction_id: TransactionId::new(7),
        epoch: EpochId::new(1),
    })
    .unwrap();
    wal.sync().unwrap();
    drop(wal);
    append_torn_tail(&wal_path);
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

#[test]
fn conflicting_identity_wal_with_torn_suffix_is_rejected_without_repair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let wal_path = path.join("wal");
    let wal = LpgWal::open(&wal_path).unwrap();
    for _ in 0..2 {
        let metadata = WorldIdentityMetadataV1::new(
            grafeo_common::types::StoreId::generate().unwrap(),
            HistoryCompleteness::Complete,
        )
        .unwrap();
        wal.log(&WalRecord::StoreIdentityMeta { metadata }).unwrap();
    }
    wal.log(&WalRecord::GraphModelMeta { model: 1 }).unwrap();
    wal.sync().unwrap();
    drop(wal);
    append_torn_tail(&wal_path);
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

#[test]
fn missing_model_wal_with_identity_and_torn_suffix_is_rejected_without_repair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let wal_path = path.join("wal");
    let wal = LpgWal::open(&wal_path).unwrap();
    let metadata = WorldIdentityMetadataV1::new(
        grafeo_common::types::StoreId::generate().unwrap(),
        HistoryCompleteness::Complete,
    )
    .unwrap();
    wal.log(&WalRecord::StoreIdentityMeta { metadata }).unwrap();
    wal.sync().unwrap();
    drop(wal);
    append_torn_tail(&wal_path);
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

fn checkpoint_with_retained_authority(
    path: &Path,
    prefix: WorldIdentityMetadataV1,
    prefix_model: u8,
    suffix: Option<WorldIdentityMetadataV1>,
) {
    let wal_path = path.join("wal");
    let wal = LpgWal::open(&wal_path).unwrap();
    wal.log(&WalRecord::StoreIdentityMeta { metadata: prefix })
        .unwrap();
    wal.log(&WalRecord::GraphModelMeta {
        model: prefix_model,
    })
    .unwrap();
    wal.checkpoint(TransactionId::new(7), EpochId::new(1))
        .unwrap();
    if let Some(metadata) = suffix {
        wal.log(&WalRecord::StoreIdentityMeta { metadata }).unwrap();
        wal.log(&WalRecord::GraphModelMeta { model: 1 }).unwrap();
    }
    wal.sync().unwrap();
    drop(wal);
    assert!(
        wal_path.join("wal_00000000.log").exists(),
        "test requires retained prefix"
    );
    assert!(
        wal_path.join("wal_00000001.log").exists(),
        "test requires checkpoint suffix"
    );
}

fn new_identity() -> WorldIdentityMetadataV1 {
    WorldIdentityMetadataV1::new(
        grafeo_common::types::StoreId::generate().unwrap(),
        HistoryCompleteness::Complete,
    )
    .unwrap()
}

#[test]
fn retained_prefix_identity_conflict_is_rejected_before_repair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    checkpoint_with_retained_authority(&path, new_identity(), 1, Some(new_identity()));
    append_torn_tail(&path.join("wal"));
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

#[test]
fn retained_prefix_model_conflict_is_rejected_before_repair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let identity = new_identity();
    checkpoint_with_retained_authority(&path, identity.clone(), 0, Some(identity));
    append_torn_tail(&path.join("wal"));
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

#[test]
fn retained_prefix_equal_declarations_are_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let identity = new_identity();
    checkpoint_with_retained_authority(&path, identity.clone(), 1, Some(identity.clone()));
    let db = GrafeoDB::with_config(config(&path, StorageFormat::WalDirectory)).unwrap();
    assert_eq!(db.world_identity(), identity);
}

#[test]
fn retained_prefix_cannot_supply_missing_suffix_authority() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    checkpoint_with_retained_authority(&path, new_identity(), 1, None);
    append_torn_tail(&path.join("wal"));
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

fn assert_unattributed_tail_rejected(kind: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let wal_path = path.join("wal");
    let wal = LpgWal::open(&wal_path).unwrap();
    if kind != "torn" {
        wal.log(&mutation()).unwrap();
    }
    if kind == "aborted" {
        wal.log(&WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(7),
        })
        .unwrap();
    }
    wal.sync().unwrap();
    drop(wal);
    if kind == "torn" {
        append_torn_tail(&wal_path);
    }
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

#[test]
fn uncommitted_only_wal_cannot_mint_fresh_identity() {
    assert_unattributed_tail_rejected("uncommitted");
}

#[test]
fn aborted_only_wal_cannot_mint_fresh_identity() {
    assert_unattributed_tail_rejected("aborted");
}

#[test]
fn torn_only_wal_cannot_mint_fresh_identity() {
    assert_unattributed_tail_rejected("torn");
}

#[test]
fn checkpoint_only_wal_directory_is_not_fresh_startup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let wal = LpgWal::open(path.join("wal")).unwrap();
    wal.checkpoint(TransactionId::new(7), EpochId::new(1))
        .unwrap();
    drop(wal);
    assert_rejected_unchanged(dir.path(), &path, StorageFormat::WalDirectory);
}

#[cfg(feature = "lpg")]
#[test]
fn container_rejects_same_identity_different_model_before_tail_repair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database.grafeo");
    let db = GrafeoDB::with_config(config(&path, StorageFormat::SingleFile)).unwrap();
    let identity = db.world_identity();
    db.close().unwrap();
    let wal_path = path.with_extension("grafeo.wal");
    let wal = LpgWal::open(&wal_path).unwrap();
    wal.log(&WalRecord::StoreIdentityMeta { metadata: identity })
        .unwrap();
    wal.log(&WalRecord::GraphModelMeta { model: 0 }).unwrap();
    wal.sync().unwrap();
    drop(wal);
    append_torn_tail(&wal_path);
    let before = files(dir.path());
    // Deliberately unpinned: the container, not a caller preference, is authority.
    assert!(GrafeoDB::open(&path).is_err());
    assert!(
        files(dir.path()) == before,
        "model mismatch must not repair the tail"
    );
}

#[test]
fn current_wal_only_preserves_identity_and_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database");
    let db = GrafeoDB::with_config(config(&path, StorageFormat::WalDirectory)).unwrap();
    let identity = db.world_identity();
    db.execute_sparql("INSERT DATA { <http://example/s> <http://example/p> \"value\" }")
        .unwrap();
    db.close().unwrap();
    let reopened = GrafeoDB::with_config(config(&path, StorageFormat::WalDirectory)).unwrap();
    assert_eq!(reopened.world_identity(), identity);
    assert_eq!(
        reopened
            .execute_sparql("SELECT * WHERE { ?s ?p ?o }")
            .unwrap()
            .row_count(),
        1
    );
}

#[test]
fn rdf_checkpoint_tail_inherits_verified_container_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.grafeo");
    let copied = dir.path().join("copy.grafeo");
    let db = GrafeoDB::with_config(config(&path, StorageFormat::SingleFile)).unwrap();
    let identity = db.world_identity();
    db.execute_sparql("INSERT DATA { <http://example/before> <http://example/p> \"value\" }")
        .unwrap();
    db.wal_checkpoint().unwrap();
    db.execute_sparql("INSERT DATA { <http://example/after> <http://example/p> \"value\" }")
        .unwrap();
    db.wal().unwrap().sync().unwrap();
    std::fs::copy(&path, &copied).unwrap();
    let copied_wal = copied.with_extension("grafeo.wal");
    std::fs::create_dir(&copied_wal).unwrap();
    for entry in std::fs::read_dir(path.with_extension("grafeo.wal")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), copied_wal.join(entry.file_name())).unwrap();
    }
    let reopened = GrafeoDB::with_config(config(&copied, StorageFormat::SingleFile)).unwrap();
    assert_eq!(reopened.world_identity(), identity);
    assert_eq!(
        reopened
            .execute_sparql("SELECT * WHERE { ?s ?p ?o }")
            .unwrap()
            .row_count(),
        2
    );
}
