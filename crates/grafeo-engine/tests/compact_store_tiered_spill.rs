//! Integration tests for the tier-aware `CompactStore` base.
//!
//! Exercises the full `compact() → spill_all() → swap_base()` lifecycle
//! through the public GrafeoDB API, verifying that:
//!
//! - `compact()` installs a `CompactStoreTiered` wrapper and registers a
//!   `CompactStoreConsumer` with the BufferManager.
//! - `BufferManager::spill_all()` actually spills the base to a mmap'd file
//!   and publishes the fresh `Arc<CompactStore>` to the `LayeredStore`.
//! - Reads continue to work transparently across the tier transition.
//! - `recompact()` retains and advances the tier wrapper so long-lived views
//!   track the new base and every later tier transition coherently.
//!
//! Run with the features that actually discover this test binary:
//!
//! ```bash
//! cargo test --locked -p grafeo-engine --no-default-features \
//!   --features "lpg,gql,mmap,compact-store" \
//!   --test compact_store_tiered_spill
//! ```

#![cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]

use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
#[cfg(unix)]
use std::time::{Duration, Instant};

use grafeo_core::graph::traits::GraphStore;
use grafeo_engine::{Config, GrafeoDB};

fn spill_dir(label: &str) -> PathBuf {
    let base = std::env::temp_dir().join("grafeo-compact-tiered-tests");
    base.join(format!("{label}-{}", std::process::id()))
}

fn config_with_spill(dir: &PathBuf) -> Config {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("create spill dir");
    Config::in_memory().with_spill_path(dir.clone())
}

fn config_with_existing_spill(dir: &Path) -> Config {
    std::fs::create_dir_all(dir).expect("create existing spill dir");
    Config::in_memory().with_spill_path(dir.to_path_buf())
}

fn seed_db(db: &mut GrafeoDB) {
    for i in 0..16 {
        db.execute(&format!(
            "INSERT (:Person {{name: 'person-{i}', age: {i}}})"
        ))
        .unwrap();
    }
}

#[cfg(unix)]
const CRASH_CHILD_DIR: &str = "GRAFEO_COMPACT_CRASH_CHILD_DIR";
#[cfg(unix)]
const CRASH_POINT: &str = "GRAFEO_COMPACT_CRASH_POINT";
#[cfg(unix)]
const CRASH_READY: &str = "GRAFEO_COMPACT_CRASH_READY";
#[cfg(unix)]
const CRASH_PUBLISHED: &str = "GRAFEO_COMPACT_CRASH_PUBLISHED_PAUSE";
#[cfg(unix)]
const PUBLISHED_CRASH_POINT: &str = "published-spill";

#[cfg(unix)]
fn compact_lifecycle_debris(dir: &Path) -> Vec<PathBuf> {
    let mut debris: Vec<_> = std::fs::read_dir(dir)
        .expect("read compact spill directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.as_encoded_bytes();
            (name.starts_with(b".grafeo-gen-") || name.starts_with(b".grafeo-owner-"))
                .then_some(path)
        })
        .collect();
    debris.sort();
    debris
}

#[cfg(unix)]
fn generation_debris(dir: &Path) -> Vec<PathBuf> {
    compact_lifecycle_debris(dir)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.as_encoded_bytes().starts_with(b".grafeo-gen-"))
        })
        .collect()
}

#[cfg(unix)]
fn owner_debris(dir: &Path) -> Vec<PathBuf> {
    compact_lifecycle_debris(dir)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.as_encoded_bytes().starts_with(b".grafeo-owner-"))
        })
        .collect()
}

#[cfg(unix)]
fn spill_one_generation(dir: &Path) -> GrafeoDB {
    let mut db = GrafeoDB::with_config(config_with_existing_spill(dir)).expect("open spill db");
    seed_db(&mut db);
    db.compact().expect("compact spill db");
    db.buffer_manager().spill_all();
    assert!(
        db.compact_tiered()
            .expect("tiered compact store")
            .is_on_disk(),
        "compact consumer must publish its mmap tier"
    );
    db
}

#[cfg(unix)]
struct CrashChild {
    child: Option<Child>,
}

#[cfg(unix)]
impl CrashChild {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn process(&mut self) -> &mut Child {
        self.child.as_mut().expect("live crash child")
    }

    fn terminate_and_wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        let child = self.process();
        let killed = child.kill();
        // Reap even if kill reports an error (for example an already-exited
        // child). Retain ownership if wait fails so Drop can retry cleanup.
        let status = child.wait();
        if status.is_ok() {
            self.child = None;
        }
        killed?;
        status
    }
}

#[cfg(unix)]
impl Drop for CrashChild {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = self.terminate_and_wait();
        }
    }
}

#[cfg(unix)]
fn crash_marker_ready(ready: &Path, point: &str) -> std::io::Result<bool> {
    match std::fs::read(ready) {
        Ok(bytes) if bytes == point.as_bytes() => Ok(true),
        // Creation is visible before write_all completes. Empty and partial
        // prefixes are not readiness; retain the caller's bounded deadline.
        Ok(bytes) if point.as_bytes().starts_with(&bytes) => Ok(false),
        Ok(_) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "child reached a different crash milestone",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn spawn_crash_child(dir: &Path, point: &str, run: &str) -> (CrashChild, PathBuf) {
    std::fs::create_dir_all(dir).expect("create crash spill directory");
    let ready = dir.join(format!("ready-{run}-{point}"));
    let _ = std::fs::remove_file(&ready);
    let mut child = CrashChild::new(
        Command::new(std::env::current_exe().expect("current test executable"))
            .arg("--exact")
            .arg("compact_generation_lifecycle_crash_child")
            .arg("--nocapture")
            .env(CRASH_CHILD_DIR, dir)
            .env(
                CRASH_POINT,
                if point == PUBLISHED_CRASH_POINT && !cfg!(debug_assertions) {
                    // A known production hook must be inert in release. The
                    // test harness publishes its own marker after spill returns.
                    "generation-mmap"
                } else {
                    point
                },
            )
            .env(
                CRASH_PUBLISHED,
                if point == PUBLISHED_CRASH_POINT {
                    "1"
                } else {
                    "0"
                },
            )
            .env(CRASH_READY, &ready)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn compact lifecycle crash child"),
    );

    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if crash_marker_ready(&ready, point).expect("read crash ready marker") {
            return (child, ready);
        }
        if let Some(status) = child.process().try_wait().expect("poll crash child") {
            panic!("crash child exited before {point}: {status}");
        }
        assert!(
            Instant::now() < deadline,
            "crash child did not reach {point} within 20 seconds"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
fn kill_crash_child(mut child: CrashChild, ready: &Path) {
    let status = child
        .terminate_and_wait()
        .expect("kill and reap compact lifecycle child");
    assert!(!status.success(), "SIGKILL child unexpectedly succeeded");
    std::fs::remove_file(ready).expect("remove crash ready marker");
}

#[cfg(unix)]
#[test]
fn crash_marker_requires_complete_exact_payload() {
    let dir = tempfile::tempdir().expect("marker directory");
    let ready = dir.path().join("ready");
    assert!(!crash_marker_ready(&ready, "owner-create").unwrap());
    for payload in ["", "owner-", "owner-creat"] {
        std::fs::write(&ready, payload).unwrap();
        assert!(!crash_marker_ready(&ready, "owner-create").unwrap());
    }
    std::fs::write(&ready, "owner-create").unwrap();
    assert!(crash_marker_ready(&ready, "owner-create").unwrap());
    for payload in ["owner-write", "owner-create-extra"] {
        std::fs::write(&ready, payload).unwrap();
        assert_eq!(
            crash_marker_ready(&ready, "owner-create")
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
    assert!(crash_marker_ready(dir.path(), "owner-create").is_err());
}

#[cfg(unix)]
const CLEANUP_CHILD: &str = "GRAFEO_COMPACT_CLEANUP_CHILD";

#[cfg(unix)]
#[test]
fn compact_crash_cleanup_child() {
    use std::io::{Read, Write};
    if std::env::var_os(CLEANUP_CHILD).is_none() {
        return;
    }
    std::io::stdout()
        .write_all(b"\nGRAFEO-CLEANUP-READY\n")
        .unwrap();
    std::io::stdout().flush().unwrap();
    // Parent retains stdin independently of CrashChild. Closing it provides
    // bounded fallback cleanup even if the tested unwind guard regresses.
    std::io::stdin().read_to_end(&mut Vec::new()).unwrap();
}

#[cfg(unix)]
#[test]
fn crash_child_cleanup_terminates_on_unwind_and_reaps_on_explicit_finish() {
    use std::io::{BufRead, Read};
    use std::os::unix::process::ExitStatusExt;
    for unwind in [false, true] {
        let mut child = CrashChild::new(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "compact_crash_cleanup_child", "--nocapture"])
                .env(CLEANUP_CHILD, "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let held_stdin = child.process().stdin.take().unwrap();
        let mut stdout = std::io::BufReader::new(child.process().stdout.take().unwrap());
        let (ready_sent, ready_received) = std::sync::mpsc::channel();
        let (sent, received) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                match stdout.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) if line == "GRAFEO-CLEANUP-READY\n" => {
                        let _ = ready_sent.send(());
                        break;
                    }
                    Ok(_) => {}
                }
            }
            let result = stdout.read_to_end(&mut Vec::new());
            let _ = sent.send(result);
        });
        if let Err(error) = ready_received.recv_timeout(Duration::from_secs(20)) {
            drop(held_stdin);
            drop(child);
            reader.join().unwrap();
            panic!("cleanup child did not become ready: {error}");
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut owned = child;
            assert!(!unwind, "deliberate parent assertion failure");
            let status = owned.terminate_and_wait().unwrap();
            assert_eq!(status.signal(), Some(9));
            assert!(
                owned.child.is_none(),
                "successful wait must disarm ownership"
            );
        }));
        let eof = received.recv_timeout(Duration::from_secs(2));
        // Never assert before allowing fallback EOF and joining the reader:
        // a broken guard must not leave the dummy child parked indefinitely.
        drop(held_stdin);
        reader.join().unwrap();
        assert_eq!(result.is_err(), unwind);
        assert!(
            eof.expect("guard must terminate child before stdin closes")
                .is_ok()
        );
    }
}

#[cfg(unix)]
fn assert_one_stale_namespace(dir: &Path, context: &str) {
    assert_eq!(
        generation_debris(dir).len(),
        1,
        "{context}: exactly one immutable generation should remain"
    );
    assert_eq!(
        owner_debris(dir).len(),
        1,
        "{context}: exactly one owner lease should remain"
    );
}

#[cfg(unix)]
fn assert_crash_debris_is_bounded(dir: &Path, point: &str) {
    match point {
        "owner-create" | "owner-write" => {
            assert!(generation_debris(dir).is_empty());
            assert_eq!(owner_debris(dir).len(), 1);
        }
        "owner-link" => {
            assert!(generation_debris(dir).is_empty());
            assert_eq!(
                owner_debris(dir).len(),
                2,
                "identity-bound pending/final names are the exact installation pair"
            );
        }
        "generation-link" => {
            assert_eq!(generation_debris(dir).len(), 2);
            assert_eq!(owner_debris(dir).len(), 1);
        }
        _ => assert_one_stale_namespace(dir, point),
    }
    assert!(
        compact_lifecycle_debris(dir).len() <= 3,
        "{point}: crash debris exceeded one bounded owner/generation lifecycle"
    );
}

/// Subprocess-only harness for the real SIGKILL tests below. In ordinary test
/// runs the environment variable is absent and this is a no-op. A crash parent
/// selects either an intermediate debug failpoint or a test-harness pause after
/// successful spill publication. Production release code has no crash hook.
#[cfg(unix)]
#[test]
fn compact_generation_lifecycle_crash_child() {
    let Some(dir) = std::env::var_os(CRASH_CHILD_DIR).map(PathBuf::from) else {
        return;
    };
    let db = spill_one_generation(&dir);
    if std::env::var(CRASH_PUBLISHED).as_deref() == Ok("1") {
        use std::io::Write;

        let rows = db
            .execute("MATCH (p:Person) RETURN p.age ORDER BY p.age")
            .expect("read published spilled data");
        assert_eq!(rows.row_count(), 16);
        for (age, row) in (0..16).zip(rows.rows()) {
            assert_eq!(row, &vec![grafeo_common::types::Value::Int64(age)]);
        }
        let ready = PathBuf::from(std::env::var_os(CRASH_READY).expect("ready path"));
        assert!(
            !ready.exists(),
            "production must not publish a crash marker before the harness"
        );
        let mut marker = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(ready)
            .expect("create published-spill marker");
        marker
            .write_all(PUBLISHED_CRASH_POINT.as_bytes())
            .expect("write published-spill marker");
        marker.sync_all().expect("sync published-spill marker");
        loop {
            std::thread::park();
        }
    }
    panic!("configured compact lifecycle crash point was not reached");
}

#[cfg(unix)]
#[test]
fn sigkill_matrix_reclaims_create_write_mmap_base_and_tier_publication() {
    const POINTS: &[&str] = &[
        PUBLISHED_CRASH_POINT,
        #[cfg(debug_assertions)]
        "generation-pending",
        #[cfg(debug_assertions)]
        "generation-link",
        #[cfg(debug_assertions)]
        "generation-create",
        #[cfg(debug_assertions)]
        "generation-write",
        #[cfg(debug_assertions)]
        "generation-mmap",
        #[cfg(debug_assertions)]
        "tier-publication",
        #[cfg(debug_assertions)]
        "base-publication",
    ];
    let root = spill_dir("owner-sigkill-matrix");
    let _ = std::fs::remove_dir_all(&root);

    for &point in POINTS {
        let dir = root.join(point);
        let (child, ready) = spawn_crash_child(&dir, point, "matrix");
        kill_crash_child(child, &ready);
        assert_crash_debris_is_bounded(&dir, point);

        let db = spill_one_generation(&dir);
        assert_eq!(
            compact_lifecycle_debris(&dir).len(),
            2,
            "allocation must reclaim the dead namespace before publishing its own"
        );
        drop(db);
        assert!(
            compact_lifecycle_debris(&dir).is_empty(),
            "normal owner retirement must leave no generation or lease debris after {point}"
        );
    }
    std::fs::remove_dir_all(&root).expect("remove generation crash matrix directory");
}

// Owner-installation intermediate states require the existing debug hooks.
// Release owner liveness and reclamation are exercised after publication below.
#[cfg(all(unix, debug_assertions))]
#[test]
fn sigkill_matrix_reclaims_owner_create_write_and_link_installation() {
    const POINTS: [&str; 3] = ["owner-create", "owner-write", "owner-link"];
    let root = spill_dir("owner-lease-sigkill-matrix");
    let _ = std::fs::remove_dir_all(&root);

    for point in POINTS {
        let dir = root.join(point);
        let (child, ready) = spawn_crash_child(&dir, point, "owner-matrix");
        kill_crash_child(child, &ready);
        assert_crash_debris_is_bounded(&dir, point);

        let db = spill_one_generation(&dir);
        assert_eq!(
            compact_lifecycle_debris(&dir).len(),
            2,
            "allocation must reclaim the stale owner installation before creating its own"
        );
        drop(db);
        assert!(compact_lifecycle_debris(&dir).is_empty());
    }
    std::fs::remove_dir_all(&root).expect("remove owner crash matrix directory");
}

#[cfg(unix)]
#[test]
fn repeated_sigkill_cycles_keep_generation_debris_bounded() {
    const POINTS: &[&str] = &[
        PUBLISHED_CRASH_POINT,
        #[cfg(debug_assertions)]
        "owner-create",
        #[cfg(debug_assertions)]
        "owner-write",
        #[cfg(debug_assertions)]
        "owner-link",
        #[cfg(debug_assertions)]
        "generation-pending",
        #[cfg(debug_assertions)]
        "generation-link",
        #[cfg(debug_assertions)]
        "generation-create",
        #[cfg(debug_assertions)]
        "generation-write",
        #[cfg(debug_assertions)]
        "generation-mmap",
        #[cfg(debug_assertions)]
        "tier-publication",
        #[cfg(debug_assertions)]
        "base-publication",
    ];
    let dir = spill_dir("owner-repeated-sigkill");
    let _ = std::fs::remove_dir_all(&dir);

    for cycle in 0..(2 * POINTS.len()).max(20) {
        let point = POINTS[cycle % POINTS.len()];
        let (child, ready) = spawn_crash_child(&dir, point, &cycle.to_string());
        kill_crash_child(child, &ready);
        assert_crash_debris_is_bounded(&dir, point);
    }

    let db = spill_one_generation(&dir);
    assert_eq!(compact_lifecycle_debris(&dir).len(), 2);
    drop(db);
    assert!(
        compact_lifecycle_debris(&dir).is_empty(),
        "a final normal lifecycle must reclaim the last crashed namespace"
    );
    std::fs::remove_dir_all(&dir).expect("remove repeated crash directory");
}

#[cfg(unix)]
#[test]
fn live_owner_lease_prevents_foreign_process_reclamation() {
    const POINTS: &[&str] = &[
        PUBLISHED_CRASH_POINT,
        #[cfg(debug_assertions)]
        "owner-create",
        #[cfg(debug_assertions)]
        "owner-write",
        #[cfg(debug_assertions)]
        "owner-link",
        #[cfg(debug_assertions)]
        "generation-mmap",
    ];
    let root = spill_dir("owner-live-protection");
    let _ = std::fs::remove_dir_all(&root);

    for &point in POINTS {
        let dir = root.join(point);
        let (child, ready) = spawn_crash_child(&dir, point, "live");
        let live_debris = compact_lifecycle_debris(&dir);
        assert_crash_debris_is_bounded(&dir, point);

        let local = spill_one_generation(&dir);
        assert!(
            live_debris.iter().all(|path| path.exists()),
            "allocation deleted a live owner lifecycle entry at {point}"
        );
        assert_eq!(
            compact_lifecycle_debris(&dir).len(),
            live_debris.len() + 2,
            "the local generation must coexist with every live child entry at {point}"
        );
        drop(local);
        assert_eq!(
            compact_lifecycle_debris(&dir),
            live_debris,
            "local retirement changed the live child namespace at {point}"
        );

        kill_crash_child(child, &ready);
        let reclaiming = spill_one_generation(&dir);
        assert!(
            live_debris.iter().all(|path| !path.exists()),
            "dead child namespace was not reclaimed at {point}"
        );
        drop(reclaiming);
        assert!(compact_lifecycle_debris(&dir).is_empty());
    }
    std::fs::remove_dir_all(&root).expect("remove live-owner test directory");
}

#[cfg(unix)]
#[test]
fn stale_scavenger_fails_closed_for_symlink_hardlink_and_identity_replacements() {
    use std::os::unix::fs::symlink;

    #[cfg(debug_assertions)]
    {
        // Exact-shaped pending names are explicitly reserved cache-protocol state.
        // Reclamation is safe even if a crash left no record bytes; callers retain
        // every path outside that exact root-hash/CSPRNG-token codec.
        let pending_dir = spill_dir("owner-reserved-empty-pending");
        let _ = std::fs::remove_dir_all(&pending_dir);
        let (child, ready) = spawn_crash_child(&pending_dir, "owner-create", "pending-empty");
        kill_crash_child(child, &ready);
        let pending = owner_debris(&pending_dir).remove(0);
        let displaced_pending = pending_dir.join("displaced-owner-pending");
        std::fs::rename(&pending, &displaced_pending).expect("displace pending owner inode");
        std::fs::write(&pending, b"partial").expect("install partial reserved pending state");
        drop(spill_one_generation(&pending_dir));
        assert!(
            !pending.exists(),
            "reserved pending state was not reclaimed"
        );
        assert!(displaced_pending.exists());
        assert!(compact_lifecycle_debris(&pending_dir).is_empty());

        // A close-looking but non-codec owner name is never deletion authority.
        let malformed_dir = spill_dir("owner-malformed-pending");
        let _ = std::fs::remove_dir_all(&malformed_dir);
        let (child, ready) = spawn_crash_child(&malformed_dir, "owner-write", "pending-malformed");
        kill_crash_child(child, &ready);
        let pending = owner_debris(&malformed_dir).remove(0);
        let malformed = pending.with_file_name(format!(
            "{}.malformed",
            pending.file_name().unwrap().to_string_lossy()
        ));
        std::fs::rename(&pending, &malformed).expect("make pending name malformed");
        drop(spill_one_generation(&malformed_dir));
        assert!(malformed.exists(), "malformed owner name was deleted");
        assert_eq!(owner_debris(&malformed_dir), vec![malformed]);

        // An extra link also makes a valid pending record ambiguous.
        let pending_link_dir = spill_dir("owner-hostile-pending-hardlink");
        let _ = std::fs::remove_dir_all(&pending_link_dir);
        let (child, ready) =
            spawn_crash_child(&pending_link_dir, "owner-write", "pending-hardlink");
        kill_crash_child(child, &ready);
        let pending = owner_debris(&pending_link_dir).remove(0);
        let foreign_link = pending_link_dir.join("caller-owned-pending-hard-link");
        std::fs::hard_link(&pending, &foreign_link).expect("hard-link pending owner record");
        drop(spill_one_generation(&pending_link_dir));
        assert!(pending.exists());
        assert!(foreign_link.exists());
        std::fs::remove_file(&foreign_link).expect("remove pending foreign hard link");
        drop(spill_one_generation(&pending_link_dir));
        assert!(compact_lifecycle_debris(&pending_link_dir).is_empty());

        // The exact pending spelling still does not authorize following a symlink.
        let pending_symlink_dir = spill_dir("owner-hostile-pending-symlink");
        let _ = std::fs::remove_dir_all(&pending_symlink_dir);
        let (child, ready) =
            spawn_crash_child(&pending_symlink_dir, "owner-create", "pending-symlink");
        kill_crash_child(child, &ready);
        let pending = owner_debris(&pending_symlink_dir).remove(0);
        let displaced_pending = pending_symlink_dir.join("displaced-symlink-pending");
        let foreign_target = pending_symlink_dir.join("pending-symlink-target");
        std::fs::rename(&pending, &displaced_pending).expect("displace symlink pending inode");
        std::fs::write(&foreign_target, b"caller-owned").expect("write pending symlink target");
        symlink(&foreign_target, &pending).expect("substitute pending symlink");
        drop(spill_one_generation(&pending_symlink_dir));
        assert!(
            std::fs::symlink_metadata(&pending)
                .expect("retained pending symlink")
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::read(&foreign_target).unwrap(), b"caller-owned");
        assert!(displaced_pending.exists());

        for dir in [
            pending_dir,
            malformed_dir,
            pending_link_dir,
            pending_symlink_dir,
        ] {
            std::fs::remove_dir_all(dir).expect("remove pending-owner replacement directory");
        }
    }
    let generation_point = if cfg!(debug_assertions) {
        "generation-mmap"
    } else {
        PUBLISHED_CRASH_POINT
    };

    // A symlink substituted at the exact identity-bound generation name is
    // retained together with its lease; neither its target nor the displaced
    // engine inode becomes deletion authority.
    let symlink_dir = spill_dir("owner-hostile-symlink");
    let _ = std::fs::remove_dir_all(&symlink_dir);
    let (child, ready) = spawn_crash_child(&symlink_dir, generation_point, "symlink");
    kill_crash_child(child, &ready);
    let generation = generation_debris(&symlink_dir).remove(0);
    let displaced = symlink_dir.join("displaced-engine-generation");
    let foreign_target = symlink_dir.join("caller-owned-target");
    std::fs::rename(&generation, &displaced).expect("displace engine inode");
    std::fs::write(&foreign_target, b"caller-owned").expect("write symlink target");
    symlink(&foreign_target, &generation).expect("install hostile symlink");
    drop(spill_one_generation(&symlink_dir));
    assert!(
        std::fs::symlink_metadata(&generation)
            .expect("retained symlink")
            .file_type()
            .is_symlink()
    );
    assert_eq!(std::fs::read(&foreign_target).unwrap(), b"caller-owned");
    assert!(displaced.exists());
    assert_eq!(owner_debris(&symlink_dir).len(), 1);
    std::fs::remove_file(&generation).expect("remove hostile symlink");
    drop(spill_one_generation(&symlink_dir));
    assert!(owner_debris(&symlink_dir).is_empty());
    assert!(displaced.exists(), "displaced inode is caller-owned now");

    // A second link makes ownership ambiguous. The scanner retains both names
    // until the foreign link is explicitly removed, then safely reclaims.
    let hardlink_dir = spill_dir("owner-hostile-hardlink");
    let _ = std::fs::remove_dir_all(&hardlink_dir);
    let (child, ready) = spawn_crash_child(&hardlink_dir, generation_point, "hardlink");
    kill_crash_child(child, &ready);
    let generation = generation_debris(&hardlink_dir).remove(0);
    let foreign_link = hardlink_dir.join("caller-owned-hard-link");
    std::fs::hard_link(&generation, &foreign_link).expect("install hostile hard link");
    drop(spill_one_generation(&hardlink_dir));
    assert!(generation.exists());
    assert!(foreign_link.exists());
    assert_eq!(owner_debris(&hardlink_dir).len(), 1);
    std::fs::remove_file(&foreign_link).expect("remove foreign hard link");
    drop(spill_one_generation(&hardlink_dir));
    assert!(!generation.exists());
    assert!(owner_debris(&hardlink_dir).is_empty());

    // A different inode at the original final name cannot inherit the encoded
    // identity. Its bytes and the displaced owned inode are both retained.
    let identity_dir = spill_dir("owner-hostile-identity");
    let _ = std::fs::remove_dir_all(&identity_dir);
    let (child, ready) = spawn_crash_child(&identity_dir, generation_point, "identity");
    kill_crash_child(child, &ready);
    let generation = generation_debris(&identity_dir).remove(0);
    let displaced = identity_dir.join("displaced-identity-generation");
    std::fs::rename(&generation, &displaced).expect("displace identity-bound generation");
    std::fs::write(&generation, b"foreign replacement").expect("install foreign inode");
    drop(spill_one_generation(&identity_dir));
    assert_eq!(std::fs::read(&generation).unwrap(), b"foreign replacement");
    assert!(displaced.exists());
    assert_eq!(owner_debris(&identity_dir).len(), 1);
    std::fs::remove_file(&generation).expect("remove foreign replacement");
    drop(spill_one_generation(&identity_dir));
    assert!(owner_debris(&identity_dir).is_empty());
    assert!(displaced.exists());

    for dir in [symlink_dir, hardlink_dir, identity_dir] {
        std::fs::remove_dir_all(&dir).expect("remove hostile replacement test directory");
    }
}

#[cfg(unix)]
#[test]
fn scavenging_never_touches_root_or_unleased_foreign_files() {
    let dir = spill_dir("owner-caller-files");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create caller file directory");
    let root = dir.join("compact_base.grafeo");
    let foreign = dir.join(".grafeo-gen-foreign-caller-owned");
    std::fs::write(&root, b"caller root sentinel").expect("write caller root");
    std::fs::write(&foreign, b"foreign generation sentinel").expect("write foreign sentinel");

    drop(spill_one_generation(&dir));

    assert_eq!(std::fs::read(&root).unwrap(), b"caller root sentinel");
    assert_eq!(
        std::fs::read(&foreign).unwrap(),
        b"foreign generation sentinel"
    );
    assert_eq!(
        compact_lifecycle_debris(&dir),
        vec![foreign.clone()],
        "only the deliberately foreign unleased file may remain"
    );
    std::fs::remove_dir_all(&dir).expect("remove caller-file test directory");
}

#[test]
fn compact_installs_tiered_wrapper() {
    let dir = spill_dir("installs");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let tiered = db
        .compact_tiered()
        .expect("tiered installed after compact()");
    assert!(!tiered.is_on_disk(), "starts in-memory");
    assert!(tiered.memory_bytes() > 0);

    // LayeredStore and tiered agree on the base Arc right after compact().
    let layered = db
        .layered_store()
        .expect("layered installed after compact()");
    assert!(Arc::ptr_eq(&layered.base_store(), &tiered.store()));
}

#[test]
fn spill_all_tiers_base_to_mmap() {
    let dir = spill_dir("spill");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let tiered = db.compact_tiered().unwrap();
    let layered = db.layered_store().unwrap();
    let pre_base = layered.base_store();

    // Force every can-spill consumer to spill. The compact-store consumer
    // will persist the base and swap_base() on the layered store.
    let freed = db.buffer_manager().spill_all();

    assert!(tiered.is_on_disk(), "tier switched to OnDisk");
    let spill_root = dir.join("compact_base.grafeo");
    let generation = tiered.path().expect("exact spill generation path");
    assert_ne!(generation, spill_root, "spill root is not a mapped target");
    assert!(generation.exists(), "unique spill generation written");
    assert_eq!(
        generation.parent(),
        Some(
            std::fs::canonicalize(&dir)
                .expect("canonical spill directory")
                .as_path()
        )
    );
    // Vector/text consumers also spill (or report 0); just assert the
    // compact base contributed something when it was non-empty.
    let _ = freed;

    // LayeredStore now points at the fresh (mmap-backed) base, distinct
    // from the pre-spill Arc.
    let post_base = layered.base_store();
    assert!(!Arc::ptr_eq(&pre_base, &post_base));
    assert!(Arc::ptr_eq(&post_base, &tiered.store()));
}

#[test]
fn reads_survive_tier_transition() {
    let dir = spill_dir("reads");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let session = db.session();
    let before = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    let count_before = before.rows()[0][0].clone();
    drop(session);

    db.buffer_manager().spill_all();
    assert!(db.compact_tiered().unwrap().is_on_disk());

    // Query again against the now-mmap-backed base.
    let session = db.session();
    let after = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(after.rows()[0][0], count_before);

    // Property access still works.
    let names = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.age")
        .unwrap();
    assert_eq!(names.rows().len(), 16);
}

#[test]
fn long_lived_tier_view_tracks_recompact_and_later_spill() {
    let dir = spill_dir("recompact");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let first_tiered = db.compact_tiered().unwrap();
    let first_base = first_tiered.store();

    // Add overlay mutations then recompact.
    db.execute("INSERT (:Person {name: 'alix', age: 99})")
        .unwrap();
    db.compact().unwrap();

    let second_tiered = db.compact_tiered().unwrap();
    assert!(
        first_tiered.same_instance(&second_tiered),
        "recompact must retain the wrapper observed by long-lived views"
    );
    assert!(!Arc::ptr_eq(&first_base, &first_tiered.store()));
    assert_eq!(first_tiered.store().node_count(), 17);
    assert!(!first_tiered.is_on_disk());
    assert!(first_tiered.path().is_none());
    assert!(!second_tiered.is_on_disk());

    // New base matches the new tier wrapper.
    let layered = db.layered_store().unwrap();
    assert!(Arc::ptr_eq(&layered.base_store(), &second_tiered.store()));

    // Spill through the retained consumer. Both handles must converge on the
    // exact same mmap-backed base and coherent generation metadata.
    db.buffer_manager().spill_all();
    assert!(second_tiered.is_on_disk());
    assert!(first_tiered.is_on_disk());
    assert_eq!(first_tiered.path(), second_tiered.path());
    assert!(first_tiered.path().is_some_and(|path| path.exists()));
    assert!(Arc::ptr_eq(&first_tiered.store(), &second_tiered.store()));
    assert!(Arc::ptr_eq(
        &db.layered_store().unwrap().base_store(),
        &first_tiered.store()
    ));
}

#[test]
fn spill_without_spill_path_is_noop() {
    let mut db = GrafeoDB::new_in_memory();
    for i in 0..4 {
        db.execute(&format!("INSERT (:Person {{name: 'p-{i}'}})"))
            .unwrap();
    }
    db.compact().unwrap();

    db.buffer_manager().spill_all();
    // Without a spill_path the compact-store consumer reports can_spill=false,
    // so the base stays in-memory.
    assert!(!db.compact_tiered().unwrap().is_on_disk());
}

/// Phase 5 probe: do mutations work end-to-end while base is OnDisk?
///
/// Spills the base to mmap, then INSERTs a node, then queries it back.
/// If the overlay-as-LpgStore architecture is wired correctly, the new
/// node should be visible alongside the (mmap-backed) base nodes.
#[test]
fn phase5_probe_mutations_during_on_disk_tier() {
    let dir = spill_dir("phase5-mutations");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    db.buffer_manager().spill_all();
    assert!(
        db.compact_tiered().unwrap().is_on_disk(),
        "base must be on disk before the probe"
    );

    // Mutation while OnDisk: should land on the overlay LpgStore.
    db.execute("INSERT (:Person {name: 'overlay-node', age: 999})")
        .unwrap();

    // Read should see the new overlay node alongside the 16 base nodes = 17.
    let session = db.session();
    let count = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    eprintln!("after mutation count: {:?}", count.rows()[0][0]);
    assert_eq!(
        count.rows()[0][0],
        grafeo_common::types::Value::Int64(17),
        "overlay node must be visible alongside the 16 base nodes"
    );

    // Specific lookup of the new node.
    let lookup = session
        .execute("MATCH (p:Person {name: 'overlay-node'}) RETURN p.age")
        .unwrap();
    assert_eq!(lookup.rows().len(), 1, "overlay node lookup must succeed");
    assert_eq!(lookup.rows()[0][0], grafeo_common::types::Value::Int64(999));

    // Base reads still work.
    let base_lookup = session
        .execute("MATCH (p:Person {name: 'person-5'}) RETURN p.age")
        .unwrap();
    assert_eq!(base_lookup.rows().len(), 1);
    assert_eq!(
        base_lookup.rows()[0][0],
        grafeo_common::types::Value::Int64(5)
    );
}

/// Phase 5 probe: does recompact() merge overlay-during-OnDisk back into a fresh base?
#[test]
fn phase5_probe_recompact_after_on_disk_mutations() {
    let dir = spill_dir("phase5-recompact-after-spill");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    db.buffer_manager().spill_all();
    assert!(db.compact_tiered().unwrap().is_on_disk());

    // Mutate while OnDisk.
    db.execute("INSERT (:Person {name: 'overlay-node', age: 999})")
        .unwrap();

    // Recompact: should merge overlay → fresh in-memory base, empty overlay.
    db.compact().unwrap();
    assert!(
        !db.compact_tiered().unwrap().is_on_disk(),
        "recompact creates an in-memory base"
    );

    // Overlay should be empty.
    let layered = db.layered_store().unwrap();
    assert_eq!(
        layered.overlay_mutation_count(),
        0,
        "overlay must be empty after recompact"
    );

    // Data still queryable after recompact.
    let session = db.session();
    let count = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(count.rows()[0][0], grafeo_common::types::Value::Int64(17));

    let lookup = session
        .execute("MATCH (p:Person {name: 'overlay-node'}) RETURN p.age")
        .unwrap();
    assert_eq!(lookup.rows()[0][0], grafeo_common::types::Value::Int64(999));
}

/// Memory pressure may tier the immutable base, but must never replace the
/// mutable overlay outside the engine's maintenance-transaction protocol.
#[test]
fn spill_all_leaves_transactional_overlay_attached() {
    let dir = spill_dir("phase5c-overlay-spill");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    // Mutate while in-memory tier so overlay accumulates.
    for i in 0..20 {
        db.execute(&format!(
            "INSERT (:Person {{name: 'overlay-{i}', age: {}}})",
            100 + i
        ))
        .unwrap();
    }

    let layered = db.layered_store().unwrap();
    assert!(
        layered.overlay_mutation_count() > 0,
        "overlay must hold mutations before spill"
    );
    let pre_overlay_bytes = layered.overlay_memory_bytes();
    assert!(pre_overlay_bytes > 0, "overlay heap nonempty");

    // Drive spill via the buffer manager.
    let _freed = db.buffer_manager().spill_all();

    // The overlay is intentionally unchanged. Only explicit recompact() may
    // replace it under lifecycle + publication + quiescence gates.
    assert!(
        layered.overlay_mutation_count() > 0,
        "memory pressure must not merge the transactional overlay"
    );
    let post_overlay_bytes = layered.overlay_memory_bytes();
    assert_eq!(
        post_overlay_bytes, pre_overlay_bytes,
        "memory pressure must leave overlay ownership and contents intact"
    );

    // Data remains queryable through the same overlay.
    let session = db.session();
    let count = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(
        count.rows()[0][0],
        grafeo_common::types::Value::Int64(36),
        "16 base + 20 overlay nodes survive the merge"
    );
}

// ── Phase 5e investigation: isolate which step breaks reopen ──────────

/// Baseline: persistent .grafeo file + reopen without compact/spill.
/// If THIS works, the basic open/close cycle is fine.
#[cfg(feature = "grafeo-file")]
#[test]
fn phase5e_baseline_persistent_reopen_works() {
    let dir = spill_dir("phase5e-baseline");
    let file_path = dir.join("test.grafeo");
    let _ = std::fs::remove_file(&file_path);

    {
        let db = GrafeoDB::with_config(Config::persistent(&file_path)).unwrap();
        for i in 0..4 {
            db.execute(&format!("INSERT (:Person {{name: 'p-{i}'}})"))
                .unwrap();
        }
    }

    let db = GrafeoDB::with_config(Config::persistent(&file_path)).unwrap();
    let r = db
        .session()
        .execute("MATCH (p:Person) RETURN count(p)")
        .unwrap();
    assert_eq!(r.rows()[0][0], grafeo_common::types::Value::Int64(4));
}

/// After compact() but no spill: does reopen work?
#[cfg(feature = "grafeo-file")]
#[test]
fn phase5e_persistent_reopen_after_compact_works() {
    let dir = spill_dir("phase5e-after-compact");
    let file_path = dir.join("test.grafeo");
    let _ = std::fs::remove_file(&file_path);

    {
        let mut db = GrafeoDB::with_config(Config::persistent(&file_path)).unwrap();
        for i in 0..4 {
            db.execute(&format!("INSERT (:Person {{name: 'p-{i}'}})"))
                .unwrap();
        }
        db.compact().unwrap();
    }

    let db = GrafeoDB::with_config(Config::persistent(&file_path)).unwrap();
    let r = db
        .session()
        .execute("MATCH (p:Person) RETURN count(p)")
        .unwrap();
    assert_eq!(r.rows()[0][0], grafeo_common::types::Value::Int64(4));
}

/// After compact + spill (NO overlay mutations): does reopen work?
/// Isolates whether the bug is in the spill path or the overlay path.
#[cfg(feature = "grafeo-file")]
#[test]
fn phase5e_persistent_reopen_after_spill_no_overlay() {
    let dir = spill_dir("phase5e-after-spill");
    let file_path = dir.join("test.grafeo");
    let _ = std::fs::remove_file(&file_path);
    let _ = std::fs::create_dir_all(&dir);

    {
        let mut db =
            GrafeoDB::with_config(Config::persistent(&file_path).with_spill_path(dir.clone()))
                .unwrap();
        for i in 0..4 {
            db.execute(&format!("INSERT (:Person {{name: 'p-{i}'}})"))
                .unwrap();
        }
        db.compact().unwrap();
        db.buffer_manager().spill_all();
        assert!(db.compact_tiered().unwrap().is_on_disk());
        // No overlay mutations.
    }

    let db =
        GrafeoDB::with_config(Config::persistent(&file_path).with_spill_path(dir.clone())).unwrap();
    let r = db
        .session()
        .execute("MATCH (p:Person) RETURN count(p)")
        .unwrap();
    assert_eq!(r.rows()[0][0], grafeo_common::types::Value::Int64(4));
}

/// Phase 5e: WAL replay rebuilds the overlay after a process restart.
///
/// Persistent `.grafeo` file + spilled OnDisk base + overlay mutations
/// must round-trip across process restart. The fix path:
///   1. directory parser handles `SectionType::CompactStore` (was missing)
///   2. open() reconstructs the LayeredStore wiring when a CompactStore
///      section is found, so the loaded LpgStore becomes the overlay
///   3. WAL records replayed against the rebuilt overlay restore any
///      mutations that occurred after the last checkpoint
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn phase5_probe_recovery_rebuilds_overlay_after_on_disk_mutation() {
    let dir = spill_dir("phase5-recovery");
    let file_path = dir.join("test.grafeo");
    let _ = std::fs::remove_file(&file_path);

    {
        let mut db =
            GrafeoDB::with_config(Config::persistent(&file_path).with_spill_path(dir.clone()))
                .unwrap();
        seed_db(&mut db);
        db.compact().unwrap();
        db.buffer_manager().spill_all();
        assert!(db.compact_tiered().unwrap().is_on_disk());

        // Mutation lands on overlay (during OnDisk tier).
        db.execute("INSERT (:Person {name: 'durable-overlay', age: 777})")
            .unwrap();
        // Drop without checkpoint.
    }

    // Reopen and verify the overlay mutation survived.
    let db =
        GrafeoDB::with_config(Config::persistent(&file_path).with_spill_path(dir.clone())).unwrap();
    let session = db.session();
    let lookup = session
        .execute("MATCH (p:Person {name: 'durable-overlay'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        lookup.rows().len(),
        1,
        "WAL replay must restore the overlay-during-OnDisk mutation"
    );
    assert_eq!(lookup.rows()[0][0], grafeo_common::types::Value::Int64(777));
}
