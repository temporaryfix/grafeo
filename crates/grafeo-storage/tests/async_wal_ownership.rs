//! Async callers must share the synchronous physical WAL ownership boundary.
#![cfg(feature = "wal")]

use grafeo_common::utils::error::Result;
use grafeo_storage::wal::{AsyncWalManager, WalManager};
use std::io::{BufRead, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct AsyncOwner {
    child: Child,
    output: Option<std::thread::JoinHandle<()>>,
}

impl AsyncOwner {
    fn start(path: &std::path::Path) -> Self {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "async_owner_child", "--nocapture"])
            .env("GRAFEO_ASYNC_OWNER_CHILD", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut owner = Self {
            child,
            output: None,
        };
        let stdout = owner.child.stdout.take().unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        owner.output = Some(std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                if matches!(line.as_deref(), Ok("READY")) {
                    let _ = sender.send(());
                }
            }
        }));
        receiver
            .recv_timeout(Duration::from_secs(20))
            .expect("async child READY");
        owner
    }

    fn finish(mut self, kill: bool) {
        if kill {
            self.child.kill().unwrap();
        } else {
            writeln!(self.child.stdin.as_mut().unwrap(), "RELEASE").unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                if kill {
                    #[cfg(unix)]
                    {
                        use std::os::unix::process::ExitStatusExt;
                        assert_eq!(status.signal(), Some(9));
                    }
                    #[cfg(not(unix))]
                    assert!(!status.success());
                } else {
                    assert!(status.success(), "async child terminal status: {status}");
                }
                break;
            }
            assert!(Instant::now() < deadline, "async child did not terminate");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

impl Drop for AsyncOwner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(output) = self.output.take() {
            let _ = output.join();
        }
    }
}

#[tokio::test]
async fn async_owner_child() {
    let Some(path) = std::env::var_os("GRAFEO_ASYNC_OWNER_CHILD") else {
        return;
    };
    let wal = AsyncWalManager::open(path).await.unwrap();
    wal.log(&grafeo_storage::wal::WalRecord::EpochAdvance {
        epoch: grafeo_common::types::EpochId::new(29),
    })
    .await
    .unwrap();
    wal.sync().await.unwrap();
    println!("READY");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "RELEASE");
    wal.close().await.unwrap();
}

#[tokio::test]
async fn real_async_process_excludes_sync_recovery_and_async_until_release_or_kill() {
    for kill in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wal");
        let owner = AsyncOwner::start(&path);
        let segment = path.join("wal_00000000.log");
        let before = std::fs::read(&segment).unwrap();
        assert!(WalManager::open(&path).is_err());
        assert!(grafeo_storage::wal::WalRecovery::new(&path).is_err());
        assert!(AsyncWalManager::open(&path).await.is_err());
        assert_eq!(std::fs::read(&segment).unwrap(), before);
        owner.finish(kill);
        let records = grafeo_storage::wal::WalRecovery::new(&path)
            .unwrap()
            .recover()
            .unwrap();
        assert!(
            matches!(records.as_slice(), [grafeo_storage::wal::WalRecord::EpochAdvance { epoch }]
            if *epoch == grafeo_common::types::EpochId::new(29))
        );
        let reopened = AsyncWalManager::open(&path).await.unwrap();
        reopened.close().await.unwrap();
    }
}

#[tokio::test]
async fn async_owner_excludes_independent_sync_open() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    let first = AsyncWalManager::open(&path).await?;
    let contender = WalManager::open(&path);
    assert!(
        contender.is_err(),
        "async ownership admitted a second physical writer"
    );
    drop(first);
    Ok(())
}
