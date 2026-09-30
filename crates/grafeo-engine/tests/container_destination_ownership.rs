//! Connected save, full-backup and restore destination exclusion.
#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_engine::GrafeoDB;
use grafeo_storage::file::GrafeoFileManager;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

// Dedicated probes assert an actual contention error and exit normally.
// Owner READY/spawn failure is never treated as a successful rejection probe.
struct WalPeer {
    child: Child,
    ready: mpsc::Receiver<()>,
}

impl WalPeer {
    fn start(path: &Path, mode: &str, settings: &[(&str, &std::ffi::OsStr)]) -> Self {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "wal_connected_child", "--nocapture"])
            .env("TASK3_PATH", path)
            .env("TASK3_MODE", mode)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for (key, value) in settings {
            command.env(key, value);
        }
        let mut child = command.spawn().unwrap();
        let output = child.stdout.take().unwrap();
        let (send, ready) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if matches!(line.as_deref(), Ok("READY")) {
                    let _ = send.send(());
                }
            }
        });
        let peer = Self { child, ready };
        peer.wait_ready();
        peer
    }
    fn wait_ready(&self) {
        self.ready
            .recv_timeout(Duration::from_secs(20))
            .expect("connected WAL child READY");
    }
    fn advance(&mut self) {
        writeln!(self.child.stdin.as_mut().unwrap(), "RELEASE").unwrap();
    }
    fn finish(mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "connected WAL child terminal status {status}"
                );
                return;
            }
            assert!(
                Instant::now() < deadline,
                "connected WAL child did not terminate"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for WalPeer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn wal_handshake() {
    println!("READY");
    std::io::stdout().flush().unwrap();
    let mut response = String::new();
    std::io::stdin().read_line(&mut response).unwrap();
    assert_eq!(response.trim(), "RELEASE");
}

#[cfg(feature = "testing-crash-injection")]
fn wal_probe(path: &Path, mode: &str) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "wal_connected_child", "--nocapture"])
        .env("TASK3_PATH", path)
        .env("TASK3_MODE", mode)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "probe {mode} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("ATTEMPTED {mode}")));
}

#[test]
fn wal_connected_child() {
    use grafeo_common::utils::error::Error;
    use grafeo_storage::wal::{WalManager, WalRecovery};
    let Some(path) = std::env::var_os("TASK3_PATH") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let mode = std::env::var("TASK3_MODE").unwrap();
    match mode.as_str() {
        "writer" => {
            let wal = WalManager::open(&path).unwrap();
            wal.checkpoint(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::EpochId::new(1),
            )
            .unwrap();
            wal_handshake();
            wal.close().unwrap();
        }
        "probe-writer" | "probe-recovery" => {
            let error = if mode == "probe-writer" {
                WalManager::open(&path).err()
            } else {
                WalRecovery::new(&path).err()
            }
            .expect("probe admitted a competing W");
            assert!(
                matches!(error, Error::Io(ref e) if e.kind() == std::io::ErrorKind::WouldBlock),
                "unexpected rejection: {error:?}"
            );
            println!("ATTEMPTED {mode}");
        }
        "probe-free" => {
            let _recovery = WalRecovery::new(&path).unwrap();
            println!("ATTEMPTED {mode}");
        }
        "probe-container-create" | "probe-container-open" | "probe-container-read-only" => {
            let result = match mode.as_str() {
                "probe-container-create" => GrafeoFileManager::create(&path),
                "probe-container-read-only" => GrafeoFileManager::open_read_only(&path),
                _ => GrafeoFileManager::open(&path),
            };
            let error = result
                .err()
                .expect("probe admitted a competing container owner");
            assert!(
                matches!(error, Error::Io(ref error) if error.kind() == std::io::ErrorKind::WouldBlock),
                "{error:?}"
            );
            println!("ATTEMPTED {mode}");
        }
        "save" => {
            let db = GrafeoDB::open(std::env::var_os("TASK3_SOURCE").unwrap()).unwrap();
            let result = db.save(&path);
            match std::env::var("TASK3_EXPECT").as_deref() {
                Ok("already-exists") => {
                    let error = result.unwrap_err();
                    assert!(
                        matches!(error, Error::Io(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists),
                        "{error:?}"
                    );
                }
                Ok("published") => {
                    let error = result.unwrap_err();
                    assert!(
                        matches!(
                            error,
                            Error::Transaction(
                                grafeo_common::utils::error::TransactionError::DurabilityFailure(_)
                            )
                        ),
                        "{error:?}"
                    );
                    assert!(error.to_string().contains("published"));
                }
                _ => result.unwrap(),
            }
            db.close().unwrap();
        }
        "restore" => {
            GrafeoDB::restore_to_epoch(
                Path::new(&std::env::var_os("TASK3_SOURCE").unwrap()),
                grafeo_common::types::EpochId::new(
                    std::env::var("TASK3_EPOCH").unwrap().parse().unwrap(),
                ),
                &path,
            )
            .unwrap();
        }
        "close-failure" => {
            let db = GrafeoDB::open(&path).unwrap();
            let session = db.session();
            session.execute("INSERT (:Retained {value: 8})").unwrap();
            let control = db.wal().unwrap();
            assert!(
                !control.is_poisoned(),
                "fixture requires healthy W before close"
            );
            let error = db.close().unwrap_err();
            let point = std::env::var("GRAFEO_OWNERSHIP_FAIL").unwrap();
            assert!(
                matches!(error, Error::Io(ref e) if e.kind() == std::io::ErrorKind::PermissionDenied),
                "{error:?}"
            );
            assert!(
                error
                    .to_string()
                    .contains(&format!("injected container ownership error at {point}")),
                "original C error lost: {error}"
            );
            assert!(control.flush().is_err());
            assert!(control.rotate().is_err());
            assert!(control.size_bytes().is_err());
            assert!(db.wal_status().is_err());
            assert!(db.backup_cursor().is_err());
            assert!(session.execute("INSERT (:AfterClose)").is_err());
            assert!(db.close().is_err());
            assert_eq!(
                control.is_poisoned(),
                point != "close-sync" && point != "sidecar-retire"
            );
            wal_handshake();
            drop(db);
            wal_handshake();
            drop(session);
        }
        _ => panic!("unknown connected WAL child mode {mode}"),
    }
}

fn wal_image(path: &Path) -> Vec<(std::ffi::OsString, Vec<u8>)> {
    if !path.exists() {
        return Vec::new();
    }
    let mut image = fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), fs::read(entry.path()).unwrap())
        })
        .collect::<Vec<_>>();
    image.sort();
    image
}

#[test]
fn full_only_restore_rejects_raw_final_w_owner_before_copy() {
    let dir = tempfile::tempdir().unwrap();
    let db = source(&dir.path().join("source.grafeo"));
    let backups = dir.path().join("backups");
    let full = db.backup_full(&backups).unwrap();
    let output = dir.path().join("output.grafeo");
    seed(&output);
    let wal_path = dir.path().join("output.grafeo.wal");
    let mut owner = WalPeer::start(&wal_path, "writer", &[]);
    let before = fs::read(&output).unwrap();
    let before_wal = wal_image(&wal_path);
    let error = GrafeoDB::restore_to_epoch(&backups, full.end_epoch, &output).unwrap_err();
    assert!(
        matches!(error, grafeo_common::utils::error::Error::Io(ref e) if e.kind() == std::io::ErrorKind::WouldBlock),
        "{error:?}"
    );
    assert_eq!(fs::read(&output).unwrap(), before);
    assert_eq!(wal_image(&wal_path), before_wal);
    owner.advance();
    owner.finish();
    let control = dir.path().join("control.grafeo");
    GrafeoDB::restore_to_epoch(&backups, full.end_epoch, &control).unwrap();
    assert!(!dir.path().join("control.grafeo.wal").exists());
    assert!(
        GrafeoFileManager::open_read_only(control)
            .unwrap()
            .read_section_directory()
            .unwrap()
            .is_some()
    );
}

#[test]
fn save_rejects_occupied_orphan_sidecar_without_mutating_either_lineage() {
    use grafeo_common::utils::error::Error;

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Source {value: 'must-not-mix'})")
        .unwrap();
    let source_snapshot = db.export_snapshot().unwrap();
    let source_cut = db.world_cut().unwrap();
    for name in ["saved", "saved.grafeo"] {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join(name);
        let sidecar = dir.path().join(format!("{name}.wal"));
        let mut owner = WalPeer::start(&sidecar, "writer", &[]);
        let held_image = wal_image(&sidecar);
        assert!(
            !held_image.is_empty(),
            "orphan sidecar contains a real checkpoint"
        );
        let error = db.save(&output).unwrap_err();
        assert!(
            matches!(error, Error::Io(ref error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "{error:?}"
        );
        assert!(fs::symlink_metadata(&output).is_err());
        assert_eq!(wal_image(&sidecar), held_image);
        owner.advance();
        owner.finish();

        let idle_image = wal_image(&sidecar);
        let error = db.save(&output).unwrap_err();
        assert!(
            matches!(error, Error::Io(ref error) if error.kind() == std::io::ErrorKind::AlreadyExists),
            "{error:?}"
        );
        assert!(
            error.to_string().contains("existing destination WAL"),
            "{error}"
        );
        assert!(fs::symlink_metadata(&output).is_err());
        assert_eq!(wal_image(&sidecar), idle_image);

        seed(&output);
        let primary_image = fs::read(&output).unwrap();
        assert!(db.save(&output).is_err());
        assert_eq!(fs::read(&output).unwrap(), primary_image);
        assert_eq!(wal_image(&sidecar), idle_image);
        assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".grafeo-file-save-")
        }));
        assert_eq!(db.export_snapshot().unwrap(), source_snapshot);
        assert_eq!(db.world_cut().unwrap(), source_cut);
    }
    db.close().unwrap();
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn parent_save_raw_owner_wins_native_no_replace_without_mutation() {
    use std::ffi::OsStr;
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    source(&source_path).close().unwrap();
    let output = dir.path().join("saved");
    let mut saver = WalPeer::start(
        &output,
        "save",
        &[
            ("TASK3_SOURCE", source_path.as_os_str()),
            ("TASK3_EXPECT", OsStr::new("already-exists")),
            (
                "GRAFEO_OWNERSHIP_RENDEZVOUS",
                OsStr::new("stage-before-install"),
            ),
        ],
    );
    let wal_path = output.join("wal");
    let mut owner = WalPeer::start(&wal_path, "writer", &[]);
    let before = wal_image(&wal_path);
    saver.advance();
    saver.finish();
    assert_eq!(wal_image(&wal_path), before);
    owner.advance();
    owner.finish();
}

#[cfg(all(feature = "testing-crash-injection", unix))]
#[test]
fn parent_save_install_retains_coordinate_and_excludes_processes_through_sync() {
    use std::ffi::OsStr;
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    source(&source_path).close().unwrap();
    let output = dir.path().join("saved");
    let mut saver = WalPeer::start(
        &output,
        "save",
        &[
            ("TASK3_SOURCE", source_path.as_os_str()),
            (
                "GRAFEO_OWNERSHIP_RENDEZVOUS",
                OsStr::new("stage-before-install,stage-after-install,stage-parent-sync"),
            ),
        ],
    );
    let stage = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .as_encoded_bytes()
                .starts_with(b".saved.grafeo-file-save-")
        })
        .unwrap();
    let coordinate = dir.path().join("saved.grafeo-owner");
    let old = fs::metadata(&coordinate).unwrap();
    let sidecar = dir.path().join("saved.wal");
    let wal_coordinate = dir.path().join("saved.wal.grafeo-owner");
    let old_wal = fs::metadata(&wal_coordinate).unwrap();
    assert!(!output.exists());
    wal_probe(&output, "probe-container-create");
    wal_probe(&sidecar, "probe-writer");
    wal_probe(&sidecar, "probe-recovery");
    assert!(!output.exists());
    assert!(!sidecar.exists());
    saver.advance();
    saver.wait_ready();
    assert!(
        stage.is_dir(),
        "stage ownership survives final publication until cleanup"
    );
    assert!(!stage.join("container.grafeo").exists());
    let retained = fs::metadata(&coordinate).unwrap();
    assert_eq!((old.dev(), old.ino()), (retained.dev(), retained.ino()));
    let retained_wal = fs::metadata(&wal_coordinate).unwrap();
    assert_eq!(
        (old_wal.dev(), old_wal.ino()),
        (retained_wal.dev(), retained_wal.ino())
    );
    let before = fs::read(&output).unwrap();
    wal_probe(&output, "probe-container-create");
    wal_probe(&output, "probe-container-open");
    wal_probe(&output, "probe-container-read-only");
    wal_probe(&sidecar, "probe-writer");
    wal_probe(&sidecar, "probe-recovery");
    assert_eq!(fs::read(&output).unwrap(), before);
    saver.advance();
    saver.wait_ready(); // Final parent sync is next; both authorities remain held.
    wal_probe(&output, "probe-container-open");
    wal_probe(&sidecar, "probe-writer");
    wal_probe(&sidecar, "probe-recovery");
    assert_eq!(fs::read(&output).unwrap(), before);
    assert!(!sidecar.exists());
    saver.advance();
    saver.finish();
    assert!(!stage.exists());
    wal_probe(&sidecar, "probe-free");
    let reopened = GrafeoDB::open(&output).unwrap();
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn published_parent_sync_error_preserves_final_bytes_and_no_old_stage() {
    use std::ffi::OsStr;
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    source(&source_path).close().unwrap();
    let output = dir.path().join("saved");
    let mut saver = WalPeer::start(
        &output,
        "save",
        &[
            ("TASK3_SOURCE", source_path.as_os_str()),
            ("TASK3_EXPECT", OsStr::new("published")),
            (
                "GRAFEO_OWNERSHIP_RENDEZVOUS",
                OsStr::new("stage-after-install"),
            ),
            ("GRAFEO_OWNERSHIP_FAIL", OsStr::new("stage-parent-sync")),
        ],
    );
    let before = fs::read(&output).unwrap();
    saver.advance();
    saver.finish();
    assert_eq!(fs::read(&output).unwrap(), before);
    assert!(!fs::read_dir(dir.path()).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .as_encoded_bytes()
            .starts_with(b".saved.grafeo-file-save-")
    }));
    let db = GrafeoDB::open(&output).unwrap();
    assert_eq!(db.node_count(), 1);
    db.close().unwrap();
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn failed_engine_close_drains_healthy_w_and_retains_pre_and_post_seal_authority() {
    use std::ffi::OsStr;
    for point in [
        "before-rename",
        "retirement-admission",
        "sidecar-retire",
        "close-sync",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("failed.grafeo");
        let wal_path = dir.path().join("failed.grafeo.wal");
        let mut settings = vec![("GRAFEO_OWNERSHIP_FAIL", OsStr::new(point))];
        if point == "close-sync" {
            settings.push(("GRAFEO_OWNERSHIP_RENDEZVOUS", OsStr::new("sidecar-retire")));
        }
        let mut owner = WalPeer::start(&path, "close-failure", &settings);
        if point == "close-sync" {
            // This READY is emitted only after the real seal is stored and
            // borrowed for retirement, before any retirement/C-close mutation.
            wal_probe(&wal_path, "probe-writer");
            owner.advance();
            owner.wait_ready();
        }
        let before = wal_image(&wal_path);
        wal_probe(&wal_path, "probe-writer");
        wal_probe(&wal_path, "probe-recovery");
        assert_eq!(wal_image(&wal_path), before);
        owner.advance();
        owner.wait_ready(); // DB dropped; inactive Session retained.
        wal_probe(
            &wal_path,
            if point == "close-sync" || point == "sidecar-retire" {
                "probe-free"
            } else {
                "probe-recovery"
            },
        );
        owner.advance();
        owner.finish();
        wal_probe(&wal_path, "probe-free");
    }
}

#[cfg(all(feature = "testing-crash-injection", unix))]
#[test]
fn restore_holds_permanent_final_coordinate_through_both_children_and_install() {
    use std::ffi::OsStr;
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let db = source(&dir.path().join("source.grafeo"));
    let backups = dir.path().join("backups");
    db.backup_full(&backups).unwrap();
    db.execute("INSERT (:AfterFull)").unwrap();
    let incremental = db.backup_incremental(&backups).unwrap();
    let output = dir.path().join("restored.grafeo");
    let epoch = incremental.end_epoch.as_u64().to_string();
    let mut restore = WalPeer::start(
        &output,
        "restore",
        &[
            ("TASK3_SOURCE", backups.as_os_str()),
            ("TASK3_EPOCH", OsStr::new(&epoch)),
            (
                "GRAFEO_OWNERSHIP_RENDEZVOUS",
                OsStr::new(
                    "restore-import,restore-before-install,restore-after-install,restore-parent-sync",
                ),
            ),
        ],
    );
    let coordinate = dir.path().join("restored.grafeo.wal.grafeo-owner");
    let first = fs::metadata(&coordinate).unwrap();
    for index in 0..4 {
        if index > 0 {
            restore.wait_ready();
        }
        let current = fs::metadata(&coordinate).unwrap();
        assert_eq!((first.dev(), first.ino()), (current.dev(), current.ino()));
        wal_probe(&output, "probe-container-create");
        wal_probe(&output, "probe-container-open");
        wal_probe(&output, "probe-container-read-only");
        wal_probe(&dir.path().join("restored.grafeo.wal"), "probe-writer");
        wal_probe(&dir.path().join("restored.grafeo.wal"), "probe-recovery");
        restore.advance();
    }
    restore.finish();
    let restored = GrafeoDB::open(output).unwrap();
    assert_eq!(restored.node_count(), 2);
    restored.close().unwrap();
}

struct Owner(Child);
impl Owner {
    fn start(path: &Path) -> Self {
        Self::start_mode(path, None)
    }
    fn start_mode(path: &Path, save_source: Option<&Path>) -> Self {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "destination_owner_child", "--nocapture"])
            .env("GRAFEO_DESTINATION_OWNER", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(source) = save_source {
            command
                .env("GRAFEO_DESTINATION_SAVE_SOURCE", source)
                .env("GRAFEO_OWNERSHIP_RENDEZVOUS", "stage-before-install");
        }
        let mut child = command.spawn().unwrap();
        let output = child.stdout.take().unwrap();
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if matches!(line.as_deref(), Ok("READY")) {
                    let _ = send.send(());
                }
            }
        });
        let owner = Self(child);
        receive
            .recv_timeout(Duration::from_secs(20))
            .expect("destination child READY");
        owner
    }
    fn release(mut self) {
        writeln!(self.0.stdin.as_mut().unwrap(), "RELEASE").unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                assert!(status.success(), "destination child exit {status}");
                return;
            }
            assert!(
                Instant::now() < deadline,
                "child did not exit after RELEASE"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn destination_owner_child() {
    let Some(path) = std::env::var_os("GRAFEO_DESTINATION_OWNER") else {
        return;
    };
    if let Some(source) = std::env::var_os("GRAFEO_DESTINATION_SAVE_SOURCE") {
        let db = GrafeoDB::open(source).unwrap();
        db.save(path).unwrap();
        db.close().unwrap();
        return;
    }
    let manager = GrafeoFileManager::open(path).unwrap();
    println!("READY");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "RELEASE");
    manager.close().unwrap();
}

fn seed(path: &Path) {
    let manager = GrafeoFileManager::create(path).unwrap();
    manager
        .write_snapshot(b"destination must survive contention", 1, 1, 0, 0)
        .unwrap();
    manager.close().unwrap();
}

fn source(path: &Path) -> GrafeoDB {
    let db = GrafeoDB::open(path).unwrap();
    db.execute("INSERT (:Owned {value: 7})").unwrap();
    db.close().unwrap();
    GrafeoDB::open(path).unwrap()
}

#[test]
fn destination_restore_rejects_process_owner_before_copy_or_sidecar_change() {
    let dir = tempfile::tempdir().unwrap();
    let db = source(&dir.path().join("source.grafeo"));
    let backups = dir.path().join("backups");
    let segment = db.backup_full(&backups).unwrap();
    let output = dir.path().join("output.grafeo");
    seed(&output);
    let sidecar = dir.path().join("output.grafeo.wal");
    fs::create_dir(&sidecar).unwrap();
    fs::write(sidecar.join("sentinel"), b"old sidecar").unwrap();
    let before = fs::read(&output).unwrap();
    let owner = Owner::start(&output);
    let result = GrafeoDB::restore_to_epoch(&backups, segment.end_epoch, &output);
    assert!(result.is_err(), "restore ignored destination owner");
    assert_eq!(fs::read(&output).unwrap(), before);
    assert_eq!(fs::read(sidecar.join("sentinel")).unwrap(), b"old sidecar");
    owner.release();
    let occupied = fs::read(&output).unwrap();
    assert!(GrafeoDB::restore_to_epoch(&backups, segment.end_epoch, &output).is_err());
    assert_eq!(fs::read(&output).unwrap(), occupied);
    assert_eq!(fs::read(sidecar.join("sentinel")).unwrap(), b"old sidecar");
    fs::remove_file(&output).unwrap();
    fs::remove_dir_all(&sidecar).unwrap();
    GrafeoDB::restore_to_epoch(&backups, segment.end_epoch, &output).unwrap();
    let restored = GrafeoFileManager::open_read_only(&output).unwrap();
    assert!(restored.read_section_directory().unwrap().is_some());
}

#[test]
fn destination_full_backup_rejects_process_owner_without_manifest_publication() {
    let dir = tempfile::tempdir().unwrap();
    let db = source(&dir.path().join("source.grafeo"));
    let backups = dir.path().join("backups");
    fs::create_dir(&backups).unwrap();
    let output = backups.join("backup_full_0000.grafeo");
    seed(&output);
    let before = fs::read(&output).unwrap();
    let owner = Owner::start(&output);
    assert!(db.backup_full(&backups).is_err());
    assert_eq!(fs::read(&output).unwrap(), before);
    assert!(!backups.join("backup_manifest.json").exists());
    owner.release();
    assert!(db.backup_full(&backups).is_err());
    assert_eq!(fs::read(&output).unwrap(), before);
    assert!(!backups.join("backup_manifest.json").exists());
    let clean = dir.path().join("clean-backups");
    let segment = db.backup_full(&clean).unwrap();
    assert_eq!(segment.filename, "backup_full_0000.grafeo");
}

#[test]
fn destination_save_rejects_process_owner_without_staging_or_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let db = source(&dir.path().join("source.grafeo"));
    let output = dir.path().join("output.grafeo");
    seed(&output);
    let before = fs::read(&output).unwrap();
    let owner = Owner::start(&output);
    assert!(db.save(&output).is_err());
    assert_eq!(fs::read(&output).unwrap(), before);
    assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".grafeo-file-save-")
    }));
    owner.release();
    // Save remains no-replace after exclusion ends.
    assert!(db.save(&output).is_err());
    assert_eq!(fs::read(&output).unwrap(), before);
}

#[test]
fn destination_owned_private_stage_installs_once_and_cleans_its_directory() {
    let dir = tempfile::tempdir().unwrap();
    let db = source(&dir.path().join("source.grafeo"));
    let output = dir.path().join("saved.grafeo");
    db.save(&output).unwrap();
    let saved = GrafeoDB::open(&output).unwrap();
    assert_eq!(saved.node_count(), 1);
    saved.close().unwrap();
    let before = fs::read(&output).unwrap();
    assert!(db.save(&output).is_err());
    assert_eq!(fs::read(&output).unwrap(), before);
    assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".grafeo-file-save-")
    }));
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn destination_staged_save_keeps_final_lease_after_private_manager_close() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    let db = source(&source_path);
    db.close().unwrap();
    drop(db);
    let output = dir.path().join("saved.grafeo");
    let owner = Owner::start_mode(&output, Some(&source_path));
    assert!(!output.exists());
    let result = GrafeoFileManager::create(&output);
    assert!(
        matches!(result, Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
    assert!(!output.exists());
    let stages: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .as_encoded_bytes()
                .windows(b".grafeo-file-save-".len())
                .any(|part| part == b".grafeo-file-save-")
        })
        .collect();
    assert_eq!(stages.len(), 1);
    let private = stages[0].join("container.grafeo");
    assert!(GrafeoFileManager::open(&private).is_err());
    // The stage's own coordination lock has actually retired, while the final
    // destination lease above still excludes publication by a different owner.
    let stage_coordinate = stages[0].join("container.grafeo.grafeo-owner");
    let stage_lock = fs::File::open(stage_coordinate).unwrap();
    stage_lock.try_lock().unwrap();
    stage_lock.unlock().unwrap();
    drop(stage_lock);
    owner.release();
    assert!(!stages[0].exists());
    let saved = GrafeoDB::open(&output).unwrap();
    assert_eq!(saved.node_count(), 1);
    saved.close().unwrap();
}
