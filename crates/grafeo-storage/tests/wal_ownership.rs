//! Independent WAL owners must never share write authority.
#![cfg(feature = "wal")]

use grafeo_common::utils::error::Result;
use grafeo_storage::wal::WalManager;
use grafeo_storage::wal::{WalConfig, WalRecovery};
use std::io::{BufRead, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(feature = "grafeo-file")]
#[test]
fn standalone_container_removal_cannot_retire_an_owned_wal() -> Result<()> {
    use grafeo_storage::file::GrafeoFileManager;
    let temporary = tempfile::tempdir()?;
    let container = GrafeoFileManager::create(temporary.path().join("database.grafeo"))?;
    let owner = WalManager::open(container.sidecar_wal_path())?;
    let result = container.remove_sidecar_wal();
    assert!(
        result.is_err(),
        "container removed another owner's live WAL"
    );
    drop(owner);
    container.close()?;
    Ok(())
}

#[cfg(feature = "grafeo-file")]
#[test]
fn standalone_container_retirement_excludes_a_real_process_owner() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let container =
        grafeo_storage::file::GrafeoFileManager::create(temporary.path().join("database.grafeo"))?;
    let path = container.sidecar_wal_path();
    let owner = Owner::start(&path);
    let before = bytes(&path);
    let error = container.remove_sidecar_wal().unwrap_err();
    assert!(
        matches!(error, grafeo_common::utils::error::Error::Io(ref e) if e.kind() == std::io::ErrorKind::WouldBlock),
        "{error:?}"
    );
    assert_eq!(bytes(&path), before);
    owner.finish(false);
    container.remove_sidecar_wal()?;
    assert!(!path.exists());
    assert!(
        temporary
            .path()
            .join("database.grafeo.wal.grafeo-owner")
            .exists()
    );
    container.close()?;
    Ok(())
}

#[test]
fn capture_retains_opened_segments_and_rejects_foreign_descriptors() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let wal = WalManager::open(temporary.path().join("wal"))?;
    wal.checkpoint(
        grafeo_common::types::TransactionId::new(1),
        grafeo_common::types::EpochId::new(1),
    )?;
    let mut capture = wal.capture()?;
    assert!(capture.read_backup_cursor_bytes()?.is_none());
    let segments = capture.segments()?;
    let before = capture.read_segment(&segments[0])?;
    assert!(!before.is_empty());
    capture.rotate()?;
    assert_eq!(capture.read_segment(&segments[0])?, before);
    capture.write_backup_cursor_bytes(b"cursor-wire")?;
    assert_eq!(
        capture.read_backup_cursor_bytes()?,
        Some(b"cursor-wire".to_vec())
    );
    drop(capture);
    let mut next = wal.capture()?;
    assert!(next.read_segment(&segments[0]).is_err());
    drop(next);
    wal.close()?;
    assert!(wal.capture().is_err());
    Ok(())
}

#[test]
fn restore_stage_and_owned_trim_share_cleanup_in_both_drop_orders() -> Result<()> {
    use grafeo_storage::wal::WalDestination;
    for stage_first in [true, false] {
        let temporary = tempfile::tempdir()?;
        let final_path = temporary.path().join("output.wal");
        let mut stage = WalDestination::acquire(&final_path)?.into_restore_stage()?;
        let mut recovery = stage.trim_recovery()?;
        let wal_path = recovery.path().to_owned();
        let stage_path = wal_path.parent().unwrap().to_owned();
        recovery.recover()?;
        if stage_first {
            drop(stage);
            assert!(stage_path.exists());
            assert!(WalDestination::acquire(&final_path).is_err());
            let manager = recovery.into_wal(WalConfig::default())?;
            manager.checkpoint(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::EpochId::new(1),
            )?;
            let seal = manager.seal()?;
            drop(manager);
            assert!(wal_path.join("wal_00000000.log").exists());
            assert!(WalDestination::acquire(&final_path).is_err());
            drop(seal);
        } else {
            drop(recovery);
            assert!(stage_path.exists());
            assert!(WalDestination::acquire(&final_path).is_err());
            drop(stage);
        }
        assert!(!stage_path.exists());
        assert!(!final_path.exists());
        let _released = WalDestination::acquire(&final_path)?;
    }
    Ok(())
}

#[test]
fn restore_stage_drop_keeps_final_w_until_last_import_and_trim_owner() -> Result<()> {
    use grafeo_storage::wal::WalDestination;
    for import_first in [true, false] {
        let temporary = tempfile::tempdir()?;
        let final_path = temporary.path().join("output.wal");
        let mut stage = WalDestination::acquire(&final_path)?.into_restore_stage()?;
        let mut import = stage.import()?;
        import.write_segment(0, &[])?;
        let mut imported = import.into_recovery()?;
        let private_parent = imported.path().parent().unwrap().to_owned();
        imported.recover()?;
        let mut trim = stage.trim_recovery()?;
        trim.recover()?;
        let writer = trim.into_wal(WalConfig::default())?;
        drop(stage);
        assert!(private_parent.exists());
        assert!(WalDestination::acquire(&final_path).is_err());
        let imported = imported.seal()?;
        let trimmed = writer.seal()?;
        drop(writer);
        let (first, last) = if import_first {
            (imported, trimmed)
        } else {
            (trimmed, imported)
        };
        drop(first);
        assert!(private_parent.exists());
        assert!(WalDestination::acquire(&final_path).is_err());
        drop(last);
        assert!(!private_parent.exists());
        assert!(!final_path.exists());
        let _released = WalDestination::acquire(&final_path)?;
    }
    Ok(())
}

#[test]
fn restore_wrong_child_seals_fail_before_touching_final_bytes() -> Result<()> {
    use grafeo_storage::wal::WalDestination;
    let temporary = tempfile::tempdir()?;
    let final_path = temporary.path().join("output.wal");
    std::fs::create_dir(&final_path)?;
    std::fs::write(final_path.join("sentinel"), b"final bytes")?;
    let mut stage = WalDestination::acquire(&final_path)?.into_restore_stage()?;
    let mut import = stage.import()?.into_recovery()?;
    assert!(stage.import().is_err());
    import.recover()?;
    let mut trim = stage.trim_recovery()?;
    assert!(stage.trim_recovery().is_err());
    trim.recover()?;
    let error = stage
        .install(trim.seal()?, import.seal()?, |_, _| {
            panic!("foreign child reached rename")
        })
        .unwrap_err();
    assert!(matches!(
        error,
        grafeo_common::utils::error::Error::InvalidValue(_)
    ));
    assert_eq!(std::fs::read(final_path.join("sentinel"))?, b"final bytes");
    Ok(())
}

struct Owner(Child);

impl Owner {
    fn start(path: &std::path::Path) -> Self {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "wal_owner_child", "--nocapture"])
            .env("GRAFEO_WAL_OWNER_CHILD", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut owner = Self(child);
        let stdout = owner.0.stdout.take().unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                if matches!(line.as_deref(), Ok("READY")) {
                    let _ = send.send(());
                }
            }
        });
        receive
            .recv_timeout(Duration::from_secs(20))
            .expect("child READY");
        owner
    }

    fn finish(mut self, kill: bool) {
        if kill {
            self.0.kill().unwrap();
        } else {
            writeln!(self.0.stdin.as_mut().unwrap(), "RELEASE").unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                if kill {
                    #[cfg(unix)]
                    {
                        use std::os::unix::process::ExitStatusExt;
                        assert_eq!(status.signal(), Some(9));
                    }
                    #[cfg(not(unix))]
                    assert!(!status.success());
                } else {
                    assert!(status.success(), "child terminal status: {status}");
                }
                break;
            }
            assert!(Instant::now() < deadline, "child did not terminate");
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
fn wal_owner_child() {
    let Some(path) = std::env::var_os("GRAFEO_WAL_OWNER_CHILD") else {
        return;
    };
    let writer = WalManager::open(path).unwrap();
    writer
        .checkpoint(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::EpochId::new(1),
        )
        .unwrap();
    println!("READY");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "RELEASE");
    writer.close().unwrap();
}

fn bytes(path: &std::path::Path) -> Vec<(std::ffi::OsString, Vec<u8>)> {
    let mut image: Vec<_> = std::fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), std::fs::read(entry.path()).unwrap())
        })
        .collect();
    image.sort();
    image
}

#[test]
fn process_contenders_preserve_bytes_and_release_or_kill_reopens() -> Result<()> {
    for kill in [false, true] {
        let temporary = tempfile::tempdir()?;
        let path = temporary.path().join("wal");
        let owner = Owner::start(&path);
        let before = bytes(&path);
        assert!(WalManager::open(&path).is_err());
        assert!(WalRecovery::new(&path).is_err());
        assert_eq!(bytes(&path), before);
        owner.finish(kill);
        let mut recovery = WalRecovery::new(&path)?;
        recovery.recover()?;
    }
    Ok(())
}

#[test]
fn a_live_sync_writer_excludes_an_independent_writer() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let first = WalManager::open(&path)?;
    let second = WalManager::open(&path);
    assert!(
        second.is_err(),
        "one WAL admitted two independent writable managers"
    );
    drop(first);
    Ok(())
}

#[test]
fn recovery_excludes_writers_through_validation_and_consuming_handoff() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let mut recovery = WalRecovery::new(&path)?;
    assert!(
        !path.exists(),
        "recovery admission must not create an empty segment"
    );
    recovery.recover_validated(|_, _, _| {
        assert!(WalManager::open(&path).is_err());
        Ok(())
    })?;
    let writer = recovery.into_wal(WalConfig::default())?;
    assert!(WalRecovery::new(&path).is_err());
    writer.close()?;
    let _new = WalRecovery::new(&path)?;
    Ok(())
}

#[test]
fn failed_rescan_invalidates_consuming_handoff() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let mut recovery = WalRecovery::new(&path)?;
    recovery.recover()?;
    let result = recovery.recover_validated(|_, _, _| -> Result<()> {
        Err(std::io::Error::other("semantic rejection").into())
    });
    assert!(result.is_err());
    assert!(recovery.into_wal(WalConfig::default()).is_err());
    Ok(())
}

#[test]
fn close_is_terminal_and_seal_moves_authority_once() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let writer = WalManager::open(&path)?;
    let seal = writer.seal()?;
    assert!(writer.seal().is_err());
    writer.close()?;
    assert!(writer.flush().is_err());
    assert!(writer.size_bytes().is_err());
    assert!(WalManager::open(&path).is_err());
    drop(seal);
    let next = WalManager::open(&path)?;
    assert!(writer.rotate().is_err());
    next.close()?;
    next.close()?;
    assert!(temporary.path().join("wal.grafeo-owner").exists());
    Ok(())
}

#[test]
fn failed_drain_retires_files_but_retains_exclusion_until_drop() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let writer = WalManager::open(&path)?;
    writer.fail_and_drain()?;
    assert!(writer.is_poisoned());
    assert!(writer.sync().is_err());
    assert!(writer.close().is_err());
    assert!(WalManager::open(&path).is_err());
    drop(writer);
    let _next = WalManager::open(&path)?;
    Ok(())
}

#[test]
fn public_names_reject_reserved_ancestors_before_provisioning() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    for name in [
        "arbitrary",
        "WAL/missing/child.wal",
        "x.wal/missing/wal",
        "x.installing/missing/wal",
        ".x.grafeo-save-stage/missing/wal",
        ".x.grafeo-save-stage.wal",
    ] {
        assert!(
            WalManager::open(temporary.path().join(name)).is_err(),
            "accepted {name}"
        );
    }
    assert_eq!(std::fs::read_dir(temporary.path())?.count(), 0);
    for name in ["wal", "WAL", "graph.wAl"] {
        let writer = WalManager::open(temporary.path().join(name))?;
        writer.close()?;
    }
    Ok(())
}

#[cfg(feature = "grafeo-file")]
#[test]
fn containers_cannot_provision_beneath_a_bare_wal_component() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    assert!(
        grafeo_storage::file::ContainerDestination::acquire(
            temporary.path().join("WAL/missing/data.grafeo")
        )
        .is_err()
    );
    assert!(!temporary.path().join("WAL").exists());
    Ok(())
}

#[test]
fn adaptive_shutdown_joins_after_terminal_core_failure() -> Result<()> {
    use grafeo_storage::wal::AdaptiveFlusher;
    let temporary = tempfile::tempdir()?;
    let writer = std::sync::Arc::new(WalManager::open(temporary.path().join("wal"))?);
    let mut flusher = AdaptiveFlusher::new(std::sync::Arc::clone(&writer), 1)?;
    writer.close()?;
    assert!(flusher.shutdown().is_err());
    assert!(flusher.shutdown().is_err());
    assert!(writer.flush().is_err());
    Ok(())
}

#[cfg(unix)]
#[test]
fn aliases_resolve_before_parent_and_reserved_aliases_do_not_provision() -> Result<()> {
    use std::os::unix::fs::symlink;
    let temporary = tempfile::tempdir()?;
    let root = temporary.path();
    std::fs::create_dir_all(root.join("real/deep"))?;
    symlink(root.join("real/deep"), root.join("alias"))?;
    let writer = WalManager::open(root.join("alias/../wal"))?;
    assert_eq!(writer.dir(), std::fs::canonicalize(root.join("real/wal"))?);
    assert!(WalManager::open(root.join("real/wal")).is_err());
    symlink(root.join("real/wal"), root.join("reserved-alias"))?;
    assert!(WalManager::open(root.join("reserved-alias/missing/child.wal")).is_err());
    assert!(!root.join("real/wal/missing").exists());
    symlink(root.join("absent"), root.join("dangling.wal"))?;
    assert!(WalManager::open(root.join("dangling.wal")).is_err());
    assert!(!root.join("absent").exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn writable_artifacts_reject_symlinks_and_hardlinks_without_overwrite() -> Result<()> {
    use std::os::unix::fs::symlink;
    for artifact in [
        "wal_00000000.log",
        "checkpoint.meta",
        "checkpoint.meta.tmp",
        "wal_00000000.log.corrupt",
        "WAL_CORRUPT",
        "wal.grafeo-owner",
    ] {
        for kind in ["hardlink", "symlink", "directory"] {
            let temporary = tempfile::tempdir()?;
            let path = temporary.path().join("wal");
            std::fs::create_dir(&path)?;
            let sentinel = temporary.path().join("sentinel");
            std::fs::write(&sentinel, b"must remain exact")?;
            let artifact_path = if artifact == "wal.grafeo-owner" {
                temporary.path().join(artifact)
            } else {
                path.join(artifact)
            };
            match kind {
                "hardlink" => std::fs::hard_link(&sentinel, &artifact_path)?,
                "symlink" => symlink(&sentinel, &artifact_path)?,
                _ => std::fs::create_dir(&artifact_path)?,
            }
            if artifact.ends_with(".corrupt") || artifact == "WAL_CORRUPT" {
                let mut recovery = WalRecovery::new(&path)?;
                assert!(recovery.recover().is_err(), "accepted {artifact}");
            } else if artifact == "checkpoint.meta.tmp" {
                let writer = WalManager::open(&path)?;
                assert!(
                    writer
                        .checkpoint(
                            grafeo_common::types::TransactionId::new(1),
                            grafeo_common::types::EpochId::new(1)
                        )
                        .is_err()
                );
                assert!(writer.is_poisoned());
            } else {
                assert!(WalManager::open(&path).is_err(), "accepted {artifact}");
            }
            assert_eq!(std::fs::read(&sentinel)?, b"must remain exact");
        }
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn batch_failure_preserves_first_cause_once_and_retains_failed_owner() -> Result<()> {
    use grafeo_storage::wal::{DurabilityMode, WalRecord};
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let writer = WalManager::with_config(
        &path,
        WalConfig {
            durability: DurabilityMode::Batch {
                max_delay_ms: 50,
                max_records: u64::MAX,
            },
            ..WalConfig::default()
        },
    )?;
    writer.log(&WalRecord::EpochAdvance {
        epoch: grafeo_common::types::EpochId::new(1),
    })?;
    std::fs::hard_link(writer.path(), temporary.path().join("retained-link"))?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !writer.is_poisoned() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(writer.is_poisoned());
    assert!(
        matches!(writer.flush(), Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::InvalidInput)
    );
    assert!(
        matches!(writer.flush(), Err(grafeo_common::utils::error::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotConnected)
    );
    assert!(writer.close().is_err());
    assert!(WalRecovery::new(&path).is_err());
    drop(writer);
    let _recovery = WalRecovery::new(&path)?;
    Ok(())
}

#[cfg(feature = "encryption")]
#[test]
fn encrypted_handoff_preserves_key_configuration_and_nonce_offset() -> Result<()> {
    use grafeo_common::encryption::{KEY_SIZE, KeyChain};
    use grafeo_common::types::EpochId;
    use grafeo_storage::wal::{DurabilityMode, WalRecord};
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let chain = KeyChain::new([29; KEY_SIZE]);
    let first = WalManager::with_config_and_encryptor(
        &path,
        WalConfig::default(),
        chain.encryptor_for("grafeo-wal", &[0; 8]),
    )?;
    first.log(&WalRecord::EpochAdvance {
        epoch: EpochId::new(1),
    })?;
    first.close()?;
    let prefix = std::fs::read(first.path())?;
    let mut recovery =
        WalRecovery::with_encryptor(&path, chain.encryptor_for("grafeo-wal", &[0; 8]))?;
    assert_eq!(recovery.recover()?.len(), 1);
    let writer = recovery.into_wal(WalConfig {
        durability: DurabilityMode::NoSync,
        max_log_size: 1,
        compression: true,
    })?;
    assert_eq!(writer.durability_mode(), DurabilityMode::NoSync);
    assert!(writer.is_encrypted());
    writer.log(&WalRecord::EpochAdvance {
        epoch: EpochId::new(2),
    })?;
    assert_eq!(writer.current_sequence(), 1);
    writer.close()?;
    let continued = std::fs::read(first.path())?;
    assert!(continued.starts_with(&prefix));
    assert_ne!(
        &continued[4..16],
        &continued[prefix.len() + 4..prefix.len() + 16]
    );
    let mut recovery =
        WalRecovery::with_encryptor(&path, chain.encryptor_for("grafeo-wal", &[0; 8]))?;
    assert_eq!(recovery.recover()?.len(), 2);
    Ok(())
}

#[test]
fn unrelated_files_do_not_allocate_a_segment_identity() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    std::fs::create_dir(&path)?;
    std::fs::write(path.join("wal_123.backup"), b"unrelated")?;
    let writer = WalManager::open(&path)?;
    assert_eq!(writer.current_sequence(), 0);
    Ok(())
}

#[cfg(unix)]
#[test]
fn checkpoint_does_not_replace_a_dangling_metadata_entry() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let writer = WalManager::open(&path)?;
    let metadata = path.join("checkpoint.meta");
    std::os::unix::fs::symlink("missing-metadata", &metadata)?;
    assert!(
        writer
            .checkpoint(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::EpochId::new(1)
            )
            .is_err()
    );
    assert!(
        std::fs::symlink_metadata(metadata)?
            .file_type()
            .is_symlink()
    );
    Ok(())
}

#[cfg(feature = "encryption")]
#[test]
fn encrypted_checkpoint_nonce_exhaustion_is_non_poisoning() -> Result<()> {
    use grafeo_common::encryption::{KEY_SIZE, KeyChain};
    use grafeo_common::types::{EpochId, TransactionId};
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    std::fs::create_dir(&path)?;
    std::fs::write(path.join("wal_4294967296.log"), [])?;
    let chain = KeyChain::new([31; KEY_SIZE]);
    let writer = WalManager::with_config_and_encryptor(
        &path,
        WalConfig::default(),
        chain.encryptor_for("grafeo-wal", &[0; 8]),
    )?;
    let before = bytes(&path);
    assert!(
        writer
            .checkpoint(TransactionId::new(1), EpochId::new(1))
            .is_err()
    );
    assert!(!writer.is_poisoned());
    assert_eq!(bytes(&path), before);
    Ok(())
}
