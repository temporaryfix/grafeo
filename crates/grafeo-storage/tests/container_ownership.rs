//! Actual-process and retained-handle witnesses for container admission.
#![cfg(feature = "grafeo-file")]

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use grafeo_common::storage::SectionType;
use grafeo_storage::file::{ContainerDestination, GrafeoFileManager};

struct Owner(Child);

impl Owner {
    fn start(path: &Path, mode: &str) -> Self {
        Self::start_mode(path, mode, false)
    }

    fn start_mode(path: &Path, mode: &str, relative: bool) -> Self {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "ownership_child", "--nocapture"])
            .env("GRAFEO_OWNERSHIP_CHILD", mode)
            .env("GRAFEO_OWNERSHIP_PATH", path)
            .env(
                "GRAFEO_OWNERSHIP_RENDEZVOUS",
                mode.strip_prefix("replace:").unwrap_or(""),
            )
            .env(
                "GRAFEO_OWNERSHIP_FAIL",
                mode.strip_prefix("fail:").unwrap_or(""),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if relative {
            command
                .current_dir(path.parent().unwrap())
                .env("GRAFEO_OWNERSHIP_PATH", path.file_name().unwrap());
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
            .expect("child READY");
        owner
    }

    fn release(mut self) {
        writeln!(self.0.stdin.as_mut().unwrap(), "RELEASE").unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                assert!(status.success(), "child exit: {status}");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "child did not exit after RELEASE"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[cfg(all(unix, feature = "testing-crash-injection"))]
    fn kill(mut self) {
        self.0.kill().unwrap();
        // Child::wait closes its stdin before waiting. Keep the handshake pipe
        // alive until a terminal killed status so EOF cannot run child cleanup.
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    assert_eq!(
                        status.signal(),
                        Some(9),
                        "owner was not terminated by SIGKILL: {status}"
                    );
                }
                #[cfg(not(unix))]
                assert!(
                    !status.success(),
                    "kill unexpectedly ran normal child cleanup"
                );
                break;
            }
            assert!(
                Instant::now() < deadline,
                "killed child did not reach a terminal status"
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
fn ownership_child() {
    let Ok(mode) = std::env::var("GRAFEO_OWNERSHIP_CHILD") else {
        return;
    };
    let path = std::env::var_os("GRAFEO_OWNERSHIP_PATH").unwrap();
    let manager = match mode.as_str() {
        "reader" => GrafeoFileManager::open_read_only(&path).unwrap(),
        _ => GrafeoFileManager::open(&path).unwrap(),
    };
    if mode.starts_with("replace:") {
        manager
            .write_sections(
                &[(SectionType::LpgStore, b"replacement contents")],
                2,
                2,
                0,
                0,
            )
            .unwrap();
        manager.close().unwrap();
        return;
    }
    if let Some(point) = mode.strip_prefix("fail:") {
        let result = if point == "close-sync" {
            manager.close()
        } else {
            manager.write_sections(
                &[(SectionType::LpgStore, b"failed replacement contents")],
                2,
                2,
                0,
                0,
            )
        };
        assert!(
            matches!(result, Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied)
        );
        assert!(manager.read_snapshot().is_err());
        assert!(manager.sync().is_err());
        assert!(
            manager
                .write_snapshot(b"must not reach dummy handle", 3, 3, 0, 0)
                .is_err()
        );
        assert!(manager.remove_sidecar_wal().is_err());
    }
    println!("READY");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "RELEASE");
    manager.close().unwrap();
}

fn seed(path: &Path, data: &[u8]) {
    let manager = GrafeoFileManager::create(path).unwrap();
    manager.write_snapshot(data, 1, 1, 0, 0).unwrap();
    manager.close().unwrap();
}

#[test]
fn ownership_closed_manager_cannot_touch_reacquired_resource() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    let old = GrafeoFileManager::create(&path).unwrap();
    old.close().unwrap();
    old.close().unwrap();
    let new = GrafeoFileManager::open(&path).unwrap();
    new.write_snapshot(b"new owner's bytes", 1, 1, 0, 0)
        .unwrap();
    fs::create_dir(new.sidecar_wal_path()).unwrap();
    fs::write(new.sidecar_wal_path().join("sentinel"), b"retained WAL").unwrap();
    let before = fs::read(&path).unwrap();
    let entry = grafeo_common::storage::SectionDirectoryEntry {
        section_type: SectionType::LpgStore,
        version: 1,
        flags: grafeo_common::storage::SectionFlags {
            mmap_able: true,
            ..SectionType::LpgStore.default_flags()
        },
        offset: 12288,
        length: b"new owner's bytes".len() as u64,
        checksum: crc32fast::hash(b"new owner's bytes"),
    };
    assert!(
        matches!(old.read_section_data(&entry), Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotConnected)
    );
    assert!(
        matches!(old.mmap_section(&entry), Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotConnected)
    );
    let outcomes = [
        old.read_snapshot().is_err(),
        old.read_section_directory().is_err(),
        old.file_size().is_err(),
        old.sync().is_err(),
        old.copy_to(&dir.path().join("forbidden-copy.grafeo"))
            .is_err(),
        old.write_snapshot(b"old owner", 2, 2, 0, 0).is_err(),
        old.write_sections(&[(SectionType::LpgStore, b"old sections")], 2, 2, 0, 0)
            .is_err(),
        old.remove_sidecar_wal().is_err(),
    ];
    assert!(
        outcomes.into_iter().all(|rejected| rejected),
        "closed manager admitted I/O: {outcomes:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(
        fs::read(new.sidecar_wal_path().join("sentinel")).unwrap(),
        b"retained WAL"
    );
}

#[test]
fn ownership_copy_rejects_process_owned_destination_without_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.grafeo");
    let destination = dir.path().join("destination.grafeo");
    seed(&source_path, b"source contents");
    seed(&destination, b"destination contents");
    let before = fs::read(&destination).unwrap();
    let owner = Owner::start(&destination, "writer");
    let source = GrafeoFileManager::open(&source_path).unwrap();
    let result = source.copy_to(&destination);
    assert!(result.is_err(), "copy ignored destination owner");
    assert_eq!(fs::read(&destination).unwrap(), before);
    owner.release();
    source.copy_to(&destination).unwrap();
    assert_eq!(
        fs::read(&destination).unwrap(),
        fs::read(&source_path).unwrap()
    );
}

#[test]
fn ownership_public_reserved_names_reject_before_provisioning() {
    for name in [
        "other.grafeo.grafeo-owner",
        "other.grafeo.installing",
        "X.GRAFEO-OWNER",
        "other.wal/child.grafeo",
        "other.restore_wal/child.grafeo",
        "other.restore_wal.trimmed_wal/child.grafeo",
        "OTHER.WAL/child.grafeo",
        ".out.grafeo.grafeo-file-save-1-2.tmp/container.grafeo",
        ".out.grafeo.grafeo-save-1-2/container.grafeo",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let result = GrafeoFileManager::create(&path);
        assert!(result.is_err(), "admitted reserved resource {name}");
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            0,
            "provisioned reserved {name}"
        );
    }
}

#[cfg(unix)]
#[test]
fn ownership_hardlinks_reject_readers_and_writers_without_changing_either_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("primary.grafeo");
    let alias = dir.path().join("alias.grafeo");
    let source_path = dir.path().join("source.grafeo");
    seed(&path, b"hardlinked bytes");
    seed(&source_path, b"other bytes");
    fs::hard_link(&path, &alias).unwrap();
    let before = fs::read(&path).unwrap();
    assert!(
        GrafeoFileManager::open(&path).is_err(),
        "writable hardlink admitted"
    );
    let source = GrafeoFileManager::open(&source_path).unwrap();
    assert!(
        source.copy_to(&alias).is_err(),
        "hardlink destination admitted"
    );
    for name in [&path, &alias] {
        assert!(
            GrafeoFileManager::open_read_only(name).is_err(),
            "hardlink reader admitted through {}",
            name.display()
        );
    }
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read(&alias).unwrap(), before);
    fs::remove_file(&alias).unwrap();
    let reader = GrafeoFileManager::open_read_only(&path).unwrap();
    assert_eq!(reader.read_snapshot().unwrap(), b"hardlinked bytes");
}

#[test]
fn ownership_readers_provision_stable_coordination_and_exclude_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readers.grafeo");
    seed(&path, b"read-only bytes");
    let coordinate = dir.path().join("readers.grafeo.grafeo-owner");
    // This is isolated fixture provisioning, not runtime stale-lock cleanup.
    if coordinate.exists() {
        fs::remove_file(&coordinate).unwrap();
    }
    let before = fs::read(&path).unwrap();
    let first = Owner::start(&path, "reader");
    assert!(
        coordinate.is_file(),
        "reader did not provision stable ownership"
    );
    let second = GrafeoFileManager::open_read_only(&path).unwrap();
    assert!(GrafeoFileManager::open(&path).is_err());
    first.release();
    assert!(GrafeoFileManager::open(&path).is_err());
    second.close().unwrap();
    GrafeoFileManager::open(&path).unwrap().close().unwrap();
    assert!(coordinate.is_file());
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[cfg(feature = "testing-crash-injection")]
fn replacement_window(point: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replacement.grafeo");
    seed(&path, b"original contents");
    let owner = Owner::start(&path, &format!("replace:{point}"));
    let before = fs::read(&path).unwrap();
    let staging = dir.path().join("replacement.grafeo.installing");
    let staged_before = fs::read(&staging).ok();
    let writer_rejected = GrafeoFileManager::open(&path).is_err();
    let reader_rejected = GrafeoFileManager::open_read_only(&path).is_err();
    assert!(
        writer_rejected && reader_rejected,
        "replacement window {point}: writer rejected {writer_rejected}, reader rejected {reader_rejected}"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read(&staging).ok(), staged_before);
    owner.release();
    let reopened = GrafeoFileManager::open(&path).unwrap();
    let directory = reopened.read_section_directory().unwrap().unwrap();
    let entry = directory.find(SectionType::LpgStore).unwrap();
    assert_eq!(
        reopened.read_section_data(entry).unwrap(),
        b"replacement contents"
    );
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn ownership_replacement_excludes_contenders_after_old_primary_close() {
    replacement_window("after-close");
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn ownership_replacement_excludes_contenders_after_rename_before_reopen() {
    replacement_window("after-rename");
}

#[cfg(unix)]
#[test]
fn ownership_alias_into_reserved_tree_rejects_before_creating_parents() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let reserved = dir.path().join("out.grafeo.wal");
    fs::create_dir(&reserved).unwrap();
    let alias = dir.path().join("ordinary-alias");
    symlink(&reserved, &alias).unwrap();
    let result = GrafeoFileManager::create(alias.join("must-not-create").join("data.grafeo"));
    assert!(result.is_err());
    assert_eq!(
        fs::read_dir(&reserved).unwrap().count(),
        0,
        "reserved alias provisioned a child directory before rejection"
    );
}

#[cfg(unix)]
#[test]
fn ownership_aliases_converge_without_collapsing_symlink_parent_order() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let actual = dir.path().join("actual");
    fs::create_dir(&actual).unwrap();
    fs::create_dir(actual.join("nested")).unwrap();
    let path = actual.join("owned.grafeo");
    seed(&path, b"alias contents");
    let parent_alias = dir.path().join("parent-alias");
    symlink(actual.join("nested"), &parent_alias).unwrap();
    let final_alias = dir.path().join("final.grafeo");
    symlink(&path, &final_alias).unwrap();
    let owner = Owner::start(&path, "writer");
    for alias in [
        parent_alias.join("..").join("owned.grafeo"),
        final_alias.clone(),
        actual.join(".").join("owned.grafeo"),
    ] {
        assert!(GrafeoFileManager::open(&alias).is_err());
        assert!(GrafeoFileManager::open_read_only(&alias).is_err());
    }
    let separate = GrafeoFileManager::create(actual.join("different.grafeo")).unwrap();
    separate.close().unwrap();
    owner.release();
    let manager = GrafeoFileManager::open(&final_alias).unwrap();
    manager
        .write_sections(&[(SectionType::LpgStore, b"alias replacement")], 2, 2, 0, 0)
        .unwrap();
    manager.close().unwrap();
    assert!(
        fs::symlink_metadata(final_alias)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    let reopened = GrafeoFileManager::open(parent_alias.join("..").join("owned.grafeo")).unwrap();
    let directory = reopened.read_section_directory().unwrap().unwrap();
    assert_eq!(
        reopened
            .read_section_data(directory.find(SectionType::LpgStore).unwrap())
            .unwrap(),
        b"alias replacement"
    );
}

#[cfg(all(unix, feature = "testing-crash-injection"))]
#[test]
fn ownership_killed_replacement_owner_releases_same_coordination_inode() {
    use std::os::unix::fs::MetadataExt;
    for point in ["after-close", "after-rename"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("killed.grafeo");
        seed(&path, b"last good primary");
        let coordinate = dir.path().join("killed.grafeo.grafeo-owner");
        let metadata = fs::metadata(&coordinate).unwrap();
        let identity = (metadata.dev(), metadata.ino());
        let owner = Owner::start(&path, &format!("replace:{point}"));
        assert!(GrafeoFileManager::open(&path).is_err());
        owner.kill();
        let manager = GrafeoFileManager::open(&path).unwrap();
        if point == "after-close" {
            assert_eq!(manager.read_snapshot().unwrap(), b"last good primary");
        } else {
            let directory = manager.read_section_directory().unwrap().unwrap();
            assert_eq!(
                manager
                    .read_section_data(directory.find(SectionType::LpgStore).unwrap())
                    .unwrap(),
                b"replacement contents"
            );
        }
        manager.close().unwrap();
        let after = fs::metadata(&coordinate).unwrap();
        assert_eq!((after.dev(), after.ino()), identity);
    }
}

#[cfg(unix)]
#[test]
fn ownership_malformed_coordination_and_reserved_aliases_preserve_bytes() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.grafeo");
    seed(&path, b"actual contents");
    let coordinate = dir.path().join("data.grafeo.grafeo-owner");
    let before = fs::read(&path).unwrap();
    fs::remove_file(&coordinate).unwrap();
    symlink(&path, &coordinate).unwrap();
    assert!(GrafeoFileManager::open(&path).is_err());
    assert!(GrafeoFileManager::open_read_only(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_file(&coordinate).unwrap();
    fs::create_dir(&coordinate).unwrap();
    assert!(GrafeoFileManager::open(&path).is_err());
    fs::remove_dir(&coordinate).unwrap();
    for suffix in [".GRAFEO-OWNER", ".INSTALLING"] {
        let target = dir.path().join(format!("reserved{suffix}"));
        fs::write(&target, &before).unwrap();
        let alias = dir.path().join("alias.grafeo");
        symlink(&target, &alias).unwrap();
        assert!(GrafeoFileManager::open(&alias).is_err());
        assert!(GrafeoFileManager::open_read_only(&alias).is_err());
        assert_eq!(fs::read(&target).unwrap(), before);
        fs::remove_file(&alias).unwrap();
    }
    let dangling = dir.path().join("dangling.grafeo");
    symlink(dir.path().join("does-not-exist"), &dangling).unwrap();
    assert!(GrafeoFileManager::create(&dangling).is_err());
    assert!(
        fs::symlink_metadata(dangling)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read(path).unwrap(), before);
}

#[cfg(unix)]
#[test]
fn ownership_read_only_permission_denial_is_real_and_existing_metadata_is_readable() {
    use std::os::unix::fs::PermissionsExt;
    struct RestorePermissions(std::path::PathBuf, u32);
    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(self.1));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("permissions.grafeo");
    seed(&path, b"readable immutable contents");
    let coordinate = dir.path().join("permissions.grafeo.grafeo-owner");
    let _restore_data = RestorePermissions(path.clone(), 0o600);
    let _restore_coordinate = RestorePermissions(coordinate.clone(), 0o600);
    let _restore_parent = RestorePermissions(dir.path().to_owned(), 0o700);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).unwrap();
    fs::set_permissions(&coordinate, fs::Permissions::from_mode(0o400)).unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o500)).unwrap();
    let reader = GrafeoFileManager::open_read_only(&path).unwrap();
    assert_eq!(
        reader.read_snapshot().unwrap(),
        b"readable immutable contents"
    );
    reader.close().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::remove_file(&coordinate).unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o500)).unwrap();
    let control = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.path().join("permission-control"));
    assert!(
        matches!(control, Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied),
        "test requires actual parent permission denial, not root privileges"
    );
    let result = GrafeoFileManager::open_read_only(&path);
    assert!(
        matches!(result, Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied)
    );
    assert!(!coordinate.exists());
    assert_eq!(
        fs::read(&path).unwrap().len(),
        12288 + b"readable immutable contents".len()
    );
}

#[test]
fn ownership_private_stage_abandonment_and_install_failure_release_after_cleanup() {
    for failed_install in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.grafeo");
        let sidecar = dir.path().join("output.grafeo.wal");
        let stage = ContainerDestination::acquire(&path)
            .unwrap()
            .into_stage(0)
            .unwrap();
        let stage_path = stage.manager().path().to_owned();
        let stage_directory = stage_path.parent().unwrap().to_owned();
        stage
            .manager()
            .write_snapshot(b"private staged contents", 1, 1, 0, 0)
            .unwrap();
        assert!(GrafeoFileManager::create(&path).is_err());
        assert!(GrafeoFileManager::open(&stage_path).is_err());
        assert!(matches!(
            grafeo_storage::wal::WalDestination::acquire(&sidecar),
            Err(grafeo_common::utils::error::Error::Io(error))
                if error.kind() == std::io::ErrorKind::WouldBlock
        ));
        if failed_install {
            let error = stage
                .install(|_, _| {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "injected native install failure",
                    )
                    .into())
                })
                .unwrap_err();
            assert!(
                matches!(error, grafeo_common::utils::error::Error::Io(error) if error.kind() == std::io::ErrorKind::PermissionDenied)
            );
        } else {
            drop(stage);
        }
        assert!(!stage_directory.exists());
        assert!(!path.exists());
        assert!(!sidecar.exists());
        drop(grafeo_storage::wal::WalDestination::acquire(&sidecar).unwrap());
        assert!(dir.path().join("output.grafeo.grafeo-owner").is_file());
        GrafeoFileManager::create(&path).unwrap().close().unwrap();
    }
}

#[test]
fn ownership_copy_self_and_final_alias_preserve_all_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.grafeo");
    seed(&path, b"self copy survives");
    let before = fs::read(&path).unwrap();
    let manager = GrafeoFileManager::open(&path).unwrap();
    assert!(manager.copy_to(&path).is_err());
    #[cfg(unix)]
    {
        let alias = dir.path().join("alias.grafeo");
        std::os::unix::fs::symlink(&path, &alias).unwrap();
        assert!(manager.copy_to(&alias).is_err());
    }
    assert_eq!(fs::read(path).unwrap(), before);
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn ownership_failed_install_and_close_keep_exclusion_until_explicit_cleanup() {
    for point in [
        "before-rename",
        "before-parent-sync",
        "before-reopen",
        "close-sync",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("failed.grafeo");
        seed(&path, b"before failed install");
        let owner = Owner::start(&path, &format!("fail:{point}"));
        let before = fs::read(&path).unwrap();
        assert!(GrafeoFileManager::open(&path).is_err());
        assert!(GrafeoFileManager::open_read_only(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        owner.release();
        let manager = GrafeoFileManager::open(&path).unwrap();
        if point == "before-rename" || point == "close-sync" {
            assert_eq!(manager.read_snapshot().unwrap(), b"before failed install");
        } else {
            let directory = manager.read_section_directory().unwrap().unwrap();
            assert_eq!(
                manager
                    .read_section_data(directory.find(SectionType::LpgStore).unwrap())
                    .unwrap(),
                b"failed replacement contents"
            );
        }
        manager.close().unwrap();
    }
}

#[cfg(unix)]
#[test]
fn ownership_checkpoint_cleanup_error_preserves_staging_and_live_primary() {
    use std::os::unix::fs::PermissionsExt;
    struct RestoreParent(std::path::PathBuf);
    impl Drop for RestoreParent {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o700));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let _restore = RestoreParent(dir.path().to_owned());
    let path = dir.path().join("cleanup.grafeo");
    seed(&path, b"original primary");
    let manager = GrafeoFileManager::open(&path).unwrap();
    let staged = dir.path().join("cleanup.grafeo.installing");
    fs::write(&staged, b"staging evidence must remain unchanged").unwrap();
    let before = fs::read(&path).unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o500)).unwrap();
    let control = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.path().join("permission-control"));
    assert!(matches!(control, Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied));
    let result = manager.write_sections(&[(SectionType::LpgStore, b"replacement")], 2, 2, 0, 0);
    assert!(
        matches!(result, Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied)
    );
    assert_eq!(
        fs::read(&staged).unwrap(),
        b"staging evidence must remain unchanged"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(manager.read_snapshot().unwrap(), b"original primary");
}

#[cfg(unix)]
#[test]
fn ownership_stage_parent_sync_failure_reports_already_published_destination() {
    use std::os::unix::fs::PermissionsExt;
    struct RestoreParent(std::path::PathBuf);
    impl Drop for RestoreParent {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o700));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let _restore = RestoreParent(dir.path().to_owned());
    let destination = dir.path().join("published.grafeo");
    let stage = ContainerDestination::acquire(&destination)
        .unwrap()
        .into_stage(0)
        .unwrap();
    stage
        .manager()
        .write_snapshot(b"published before directory-sync denial", 1, 1, 0, 0)
        .unwrap();
    let error = stage
        .install(|source, destination| {
            fs::rename(source, destination)?;
            fs::set_permissions(
                destination.parent().unwrap(),
                fs::Permissions::from_mode(0o300),
            )?;
            Ok(())
        })
        .unwrap_err();
    assert!(
        matches!(
            error,
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(_)
            )
        ),
        "lost published-but-not-confirmed-durable error: {error}"
    );
    assert!(error.to_string().contains("published"));
    assert!(error.to_string().contains("published.grafeo"));
    assert!(destination.is_file());
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let reopened = GrafeoFileManager::open_read_only(&destination).unwrap();
    assert_eq!(
        reopened.read_snapshot().unwrap(),
        b"published before directory-sync denial"
    );
}

#[cfg(unix)]
#[test]
fn ownership_stage_cleanup_failure_identifies_published_output_and_private_residue() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let destination = dir.path().join("published.grafeo");
    let stage = ContainerDestination::acquire(&destination)
        .unwrap()
        .into_stage(0)
        .unwrap();
    let stage_directory = stage.manager().path().parent().unwrap().to_owned();
    stage
        .manager()
        .write_snapshot(b"published despite cleanup denial", 1, 1, 0, 0)
        .unwrap();
    let error = stage
        .install(|source, destination| {
            fs::rename(source, destination)?;
            fs::set_permissions(source.parent().unwrap(), fs::Permissions::from_mode(0o500))?;
            Ok(())
        })
        .unwrap_err();
    fs::set_permissions(&stage_directory, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        error
            .to_string()
            .contains("published and its parent sync completed")
    );
    assert!(error.to_string().contains("private stage cleanup failed"));
    assert_eq!(
        error.error_code(),
        grafeo_common::utils::error::ErrorCode::IoError
    );
    assert!(stage_directory.exists());
    let reopened = GrafeoFileManager::open_read_only(&destination).unwrap();
    assert_eq!(
        reopened.read_snapshot().unwrap(),
        b"published despite cleanup denial"
    );
}

#[test]
fn ownership_relative_child_path_converges_with_absolute_admission() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relative.grafeo");
    seed(&path, b"relative identity");
    let before = fs::read(&path).unwrap();
    let owner = Owner::start_mode(&path, "writer", true);
    assert!(GrafeoFileManager::open(&path).is_err());
    assert!(GrafeoFileManager::open_read_only(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    owner.release();
    assert_eq!(
        GrafeoFileManager::open(&path)
            .unwrap()
            .read_snapshot()
            .unwrap(),
        b"relative identity"
    );
}

#[cfg(unix)]
#[test]
fn ownership_non_utf8_names_preserve_native_admission_errors_or_exact_bytes() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir
        .path()
        .join(std::ffi::OsString::from_vec(b"source-\xff.grafeo".to_vec()));
    let copy_path = dir
        .path()
        .join(std::ffi::OsString::from_vec(b"copy-\xfe.grafeo".to_vec()));
    // Some filesystems reject these names before any container operation.
    // Prove native admission and preserve its error, never manufacture a lossy
    // replacement name or count rejection as a roundtrip witness.
    match fs::write(&source_path, b"native filesystem admission probe") {
        Ok(()) => fs::remove_file(&source_path).unwrap(),
        Err(native) => {
            let error = GrafeoFileManager::create(&source_path)
                .err()
                .expect("preserve native rejection");
            let grafeo_common::utils::error::Error::Io(error) = error else {
                panic!("native I/O error taxonomy changed")
            };
            assert_eq!(error.raw_os_error(), native.raw_os_error());
            assert_eq!(error.kind(), native.kind());
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
            return;
        }
    }
    seed(&source_path, b"lossless path contents");
    let source = GrafeoFileManager::open(&source_path).unwrap();
    assert!(
        source
            .sidecar_wal_path()
            .as_os_str()
            .as_bytes()
            .ends_with(b"source-\xff.grafeo.wal")
    );
    source.copy_to(&copy_path).unwrap();
    let copy = GrafeoFileManager::open(&copy_path).unwrap();
    assert_eq!(copy.read_snapshot().unwrap(), b"lossless path contents");
    assert!(
        dir.path()
            .join(std::ffi::OsString::from_vec(
                b"copy-\xfe.grafeo.grafeo-owner".to_vec()
            ))
            .is_file()
    );
}

#[test]
fn ownership_unicode_names_remain_exact_across_copy_and_sidecar_derivation() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source-雪🦀.grafeo");
    let copy_path = dir.path().join("copy-雪🦀.grafeo");
    seed(&source_path, b"Unicode path contents");
    let source = GrafeoFileManager::open(&source_path).unwrap();
    assert_eq!(
        source.sidecar_wal_path().file_name().unwrap(),
        "source-雪🦀.grafeo.wal"
    );
    source.copy_to(&copy_path).unwrap();
    let copy = GrafeoFileManager::open(&copy_path).unwrap();
    assert_eq!(copy.read_snapshot().unwrap(), b"Unicode path contents");
    assert!(dir.path().join("copy-雪🦀.grafeo.grafeo-owner").is_file());
}
