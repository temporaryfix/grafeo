//! Closing an owner must not wait for an unrelated fork-to-exec window.
#![cfg(all(unix, feature = "wal"))]

use grafeo_common::utils::error::Result;
use grafeo_storage::wal::{WalDestination, WalManager};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;

// Pause only between fork and exec, without running database code in the child.
// Returning the callback result after releasing/reaping the child also keeps
// assertion failures from leaving a suspended subprocess behind.
#[allow(unsafe_code)]
fn during_fork<T>(operation: impl FnOnce() -> T) -> T {
    let (mut parent, mut child) = UnixStream::pair().unwrap();
    for stream in [&parent, &child] {
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(20)))
            .unwrap();
    }
    let mut command = Command::new("/usr/bin/true");
    // SAFETY: the child only performs read/write syscalls on already-open
    // streams. No allocation, locks, database access or destructors precede exec.
    unsafe {
        command.pre_exec(move || {
            child.write_all(&[1])?;
            let mut release = [0];
            child.read_exact(&mut release)?;
            Ok(())
        });
    }
    std::thread::scope(|scope| {
        let spawned = scope.spawn(move || command.spawn());
        let mut ready = [0];
        parent.read_exact(&mut ready).unwrap();
        assert_eq!(ready, [1]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        parent.write_all(&[1]).unwrap();
        let mut child = spawned.join().unwrap().unwrap();
        assert!(child.wait().unwrap().success());
        match result {
            Ok(value) => value,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

#[test]
fn wal_close_releases_ownership_during_unrelated_fork() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let owner = WalManager::open(&path)?;
    during_fork(|| {
        assert!(WalDestination::acquire(&path).is_err());
        owner.close()?;
        let _reopened = WalDestination::acquire(&path)?;
        Ok(())
    })
}

#[test]
fn wal_drop_releases_ownership_during_unrelated_fork() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let owner = WalManager::open(&path)?;
    during_fork(|| {
        drop(owner);
        let _reopened = WalManager::open(&path)?;
        Ok(())
    })
}

#[cfg(feature = "grafeo-file")]
#[test]
fn container_close_releases_ownership_during_unrelated_fork() -> Result<()> {
    use grafeo_storage::file::GrafeoFileManager;
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("database.grafeo");
    let owner = GrafeoFileManager::create(&path)?;
    during_fork(|| {
        assert!(GrafeoFileManager::open(&path).is_err());
        owner.close()?;
        let _reopened = GrafeoFileManager::open(&path)?;
        Ok(())
    })
}

#[cfg(feature = "grafeo-file")]
#[test]
fn container_drop_releases_ownership_during_unrelated_fork() -> Result<()> {
    use grafeo_storage::file::GrafeoFileManager;
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("database.grafeo");
    let owner = GrafeoFileManager::create(&path)?;
    during_fork(|| {
        drop(owner);
        let _reopened = GrafeoFileManager::open(&path)?;
        Ok(())
    })
}

#[test]
fn sealed_wal_keeps_ownership_until_retired_during_unrelated_fork() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let owner = WalManager::open(&path)?;
    during_fork(|| {
        let seal = owner.seal()?;
        drop(owner);
        assert!(WalDestination::acquire(&path).is_err());
        drop(seal);
        let _reopened = WalManager::open(&path)?;
        Ok(())
    })
}

#[cfg(feature = "grafeo-file")]
#[test]
fn shared_readers_retire_independently_during_unrelated_fork() -> Result<()> {
    use grafeo_storage::file::GrafeoFileManager;
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("database.grafeo");
    GrafeoFileManager::create(&path)?.close()?;
    let first = GrafeoFileManager::open_read_only(&path)?;
    let second = GrafeoFileManager::open_read_only(&path)?;
    during_fork(|| {
        first.close()?;
        assert!(GrafeoFileManager::open(&path).is_err());
        second.close()?;
        let _reopened = GrafeoFileManager::open(&path)?;
        Ok(())
    })
}
