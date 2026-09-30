//! Private, database-local authority for authenticating plaintext spill roots.
//!
//! This secret is deliberately outside portable database images and WAL. Losing
//! it cannot authorize adoption of an existing spill namespace.

use std::io;
use std::path::Path;
use std::sync::Arc;

use grafeo_common::encryption::KeyChain;
use grafeo_common::types::StoreId;
use zeroize::Zeroizing;

fn fresh_key_chain() -> io::Result<Arc<KeyChain>> {
    let mut key = Zeroizing::new([0u8; 32]);
    getrandom::fill(key.as_mut())
        .map_err(|_| io::Error::other("spill authority entropy failed"))?;
    Ok(Arc::new(KeyChain::new(*key)))
}

/// Loads the database-local spill key, or bootstraps only a fresh namespace.
///
/// In-memory keys live only as long as their owning database/provider. Persistent
/// keys are local operational authority and must not accompany portable exports.
///
/// # Errors
/// Rejects unknown, substituted, public, corrupt or foreign authority metadata,
/// missing authority for an existing namespace, and contended bootstrap locks.
/// Platforms without qualified private-directory enforcement are unsupported.
pub(crate) fn load_or_create_key_chain(
    database_path: Option<&Path>,
    store_id: StoreId,
    spill_parent: &Path,
) -> io::Result<Arc<KeyChain>> {
    let Some(database_path) = database_path else {
        return fresh_key_chain();
    };
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    {
        native::load_or_create(database_path, store_id, spill_parent)
    }
    #[cfg(not(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = (database_path, store_id, spill_parent);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "persistent spill authority requires qualified private-directory permissions",
        ))
    }
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
mod native {
    use super::{Arc, KeyChain, Path, StoreId, Zeroizing, io};
    use cap_fs_ext::{
        DirExt as _, FollowSymlinks, OpenOptionsFollowExt as _, OpenOptionsSyncExt as _,
    };
    use cap_std::fs::{
        Dir, DirBuilder, DirBuilderExt as _, MetadataExt as _, OpenOptions, OpenOptionsExt as _,
    };
    use std::ffi::OsString;
    use std::fs::{File, Metadata};
    use std::io::{Read as _, Write as _};
    use std::os::unix::fs::MetadataExt as _;

    const STAGING: &str = "authority.installing";
    const LOCK_FILE: &str = "bootstrap.lock";
    const MAGIC: &[u8; 8] = b"GRSPAUTH";
    const VERSION: [u8; 2] = 1u16.to_le_bytes();
    const STORE_OFFSET: usize = 10;
    const ID_OFFSET: usize = STORE_OFFSET + StoreId::LEN;
    const KEY_OFFSET: usize = ID_OFFSET + 16;
    const RECORD_BYTES: usize = KEY_OFFSET + 32;

    fn invalid(message: &'static str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, message)
    }

    fn sync_directory(directory: &Dir) -> io::Result<()> {
        // Capability descriptors may be O_PATH. Open the retained directory
        // itself for I/O without resolving its ambient pathname again.
        let descriptor = rustix::fs::openat(
            directory,
            ".",
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?;
        File::from(descriptor).sync_all()
    }

    fn private(metadata: &Metadata, directory: bool) -> io::Result<()> {
        let kind_matches = if directory {
            metadata.is_dir()
        } else {
            metadata.is_file()
        };
        if !kind_matches
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.mode() & 0o077 != 0
            || (!directory && metadata.nlink() != 1)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "spill authority must be private, owned, and free of file aliases",
            ));
        }
        Ok(())
    }

    fn identity_matches(metadata: &Metadata, other: &cap_std::fs::Metadata) -> bool {
        metadata.dev() == other.dev() && metadata.ino() == other.ino()
    }

    // Retain the opened parent; pathname checks alone never authorize writes.
    fn open_parent(path: &Path) -> io::Result<Dir> {
        let before = std::fs::symlink_metadata(path)?;
        if !before.is_dir() || before.file_type().is_symlink() {
            return Err(invalid("spill authority parent is not a real directory"));
        }
        let uid = rustix::process::geteuid().as_raw();
        if (before.uid() != uid && before.uid() != 0)
            || (before.mode() & 0o022 != 0 && before.mode() & 0o1000 == 0)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "spill authority parent lacks trusted ownership",
            ));
        }
        let parent = Dir::open_ambient_dir(path, cap_std::ambient_authority())?;
        if !identity_matches(&before, &parent.dir_metadata()?) {
            return Err(invalid("spill authority parent changed while opening"));
        }
        #[cfg(target_os = "macos")]
        grafeo_core::execution::spill::validate_no_acl_grants(&parent)?;
        Ok(parent)
    }

    fn sidecar_coordinates(database_path: &Path) -> io::Result<(Dir, OsString)> {
        let path = std::path::absolute(database_path)?;
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(invalid("spill authority database path is a symlink"));
        }
        if metadata.is_dir() {
            return Ok((open_parent(&path)?, OsString::from(".spill-auth")));
        }
        if !metadata.is_file() {
            return Err(invalid(
                "spill authority database path is not persistent storage",
            ));
        }
        let parent = path
            .parent()
            .ok_or_else(|| invalid("database has no parent"))?;
        let mut name = path
            .file_name()
            .ok_or_else(|| invalid("database has no name"))?
            .to_os_string();
        name.push(".spill-auth");
        Ok((open_parent(parent)?, name))
    }

    fn require_fresh_namespace(spill_parent: &Path, store_id: StoreId) -> io::Result<()> {
        let parent = match open_parent(spill_parent) {
            Ok(parent) => parent,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut namespace = String::from("grafeo-store-");
        for byte in store_id.as_bytes() {
            use std::fmt::Write as _;
            write!(&mut namespace, "{byte:02x}")
                .map_err(|_| invalid("spill namespace encoding failed"))?;
        }
        match parent.symlink_metadata(namespace) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
            Ok(_) => Err(invalid(
                "existing spill namespace has no trusted key authority",
            )),
        }
    }

    fn validate_directory(parent: &Dir, name: &std::ffi::OsStr, directory: &Dir) -> io::Result<()> {
        let opened = directory.try_clone()?.into_std_file().metadata()?;
        private(&opened, true)?;
        #[cfg(target_os = "macos")]
        grafeo_core::execution::spill::validate_no_acl_grants(directory)?;
        let named = parent.symlink_metadata(name)?;
        if !named.is_dir() || !identity_matches(&opened, &named) {
            return Err(invalid("spill authority directory was replaced"));
        }
        Ok(())
    }

    fn options(write: bool, create_new: bool) -> OpenOptions {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(write)
            .create_new(create_new)
            .mode(0o600)
            .follow(FollowSymlinks::No)
            .nonblock(true);
        options
    }

    fn validate_file(directory: &Dir, name: &str, file: &File, size: u64) -> io::Result<()> {
        let opened = file.metadata()?;
        private(&opened, false)?;
        #[cfg(target_os = "macos")]
        grafeo_core::execution::spill::validate_no_acl_grants(file)?;
        let named = directory.symlink_metadata(name)?;
        if !named.is_file() || !identity_matches(&opened, &named) || opened.len() != size {
            return Err(invalid("spill authority file changed identity or length"));
        }
        Ok(())
    }

    struct BootstrapLock {
        file: File,
        process_id: u32,
        locked: bool,
    }
    impl BootstrapLock {
        fn release(mut self) -> io::Result<()> {
            if self.process_id != std::process::id() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "spill bootstrap lock belongs to another process",
                ));
            }
            let result = self.file.unlock();
            self.locked = false;
            result
        }
    }
    impl Drop for BootstrapLock {
        fn drop(&mut self) {
            if self.locked && self.process_id == std::process::id() {
                let _ = self.file.unlock();
            }
        }
    }

    #[test]
    fn explicit_release_cannot_unlock_another_process_owner() {
        let temporary = tempfile::NamedTempFile::new().unwrap();
        let owner = temporary.reopen().unwrap();
        owner.try_lock().unwrap();
        let inherited = BootstrapLock {
            file: owner.try_clone().unwrap(),
            process_id: std::process::id().wrapping_add(1),
            locked: true,
        };
        assert_eq!(
            inherited.release().unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        let contender = temporary.reopen().unwrap();
        assert_eq!(
            io::Error::from(contender.try_lock().unwrap_err()).kind(),
            io::ErrorKind::WouldBlock
        );
        owner.unlock().unwrap();
        contender.try_lock().unwrap();
        contender.unlock().unwrap();
    }

    fn lock(directory: &Dir) -> io::Result<BootstrapLock> {
        let file = match directory.open_with(LOCK_FILE, &options(true, true)) {
            Ok(file) => file.into_std(),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => directory
                .open_with(LOCK_FILE, &options(true, false))?
                .into_std(),
            Err(error) => return Err(error),
        };
        validate_file(directory, LOCK_FILE, &file, 0)?;
        file.try_lock().map_err(io::Error::from)?;
        let lock = BootstrapLock {
            file,
            process_id: std::process::id(),
            locked: true,
        };
        validate_file(directory, LOCK_FILE, &lock.file, 0)?;
        Ok(lock)
    }

    fn read_key(directory: &Dir, store_id: StoreId) -> io::Result<Arc<KeyChain>> {
        let key_file = format!("authority-{store_id}");
        let mut file = directory
            .open_with(&key_file, &options(false, false))?
            .into_std();
        validate_file(directory, &key_file, &file, RECORD_BYTES as u64)?;
        let mut record = Zeroizing::new([0u8; RECORD_BYTES]);
        file.read_exact(record.as_mut())?;
        validate_file(directory, &key_file, &file, RECORD_BYTES as u64)?;
        if &record[..8] != MAGIC
            || record[8..10] != VERSION
            || &record[STORE_OFFSET..ID_OFFSET] != store_id.as_bytes()
        {
            return Err(invalid(
                "unknown, corrupt, or foreign spill authority record",
            ));
        }
        let mut key = Zeroizing::new([0u8; 32]);
        key.copy_from_slice(&record[KEY_OFFSET..]);
        Ok(Arc::new(KeyChain::new(*key)))
    }

    pub(super) fn load_or_create(
        database_path: &Path,
        store_id: StoreId,
        spill_parent: &Path,
    ) -> io::Result<Arc<KeyChain>> {
        let (parent, name) = sidecar_coordinates(database_path)?;
        let key_file = format!("authority-{store_id}");
        let directory = match parent.open_dir_nofollow(&name) {
            Ok(directory) => directory,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                require_fresh_namespace(spill_parent, store_id)?;
                let mut builder = DirBuilder::new();
                builder.mode(0o700);
                match parent.create_dir_with(&name, &builder) {
                    Ok(()) => sync_directory(&parent)?,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(error),
                }
                parent.open_dir_nofollow(&name)?
            }
            Err(error) => return Err(error),
        };
        validate_directory(&parent, &name, &directory)?;
        // Check absence before creating even the bootstrap file. A subsequent
        // disappearance during read/validation is an error, never bootstrap.
        if let Err(error) = directory.symlink_metadata(&key_file) {
            if error.kind() != io::ErrorKind::NotFound {
                return Err(error);
            }
            require_fresh_namespace(spill_parent, store_id)?;
        }
        let bootstrap = lock(&directory)?;
        let key = match directory.symlink_metadata(&key_file) {
            Ok(_) => read_key(&directory, store_id)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                require_fresh_namespace(spill_parent, store_id)?;
                let mut record = Zeroizing::new([0u8; RECORD_BYTES]);
                record[..8].copy_from_slice(MAGIC);
                record[8..10].copy_from_slice(&VERSION);
                record[STORE_OFFSET..ID_OFFSET].copy_from_slice(store_id.as_bytes());
                getrandom::fill(&mut record[ID_OFFSET..])
                    .map_err(|_| io::Error::other("spill authority entropy failed"))?;
                let mut stage = directory
                    .open_with(STAGING, &options(true, true))?
                    .into_std();
                validate_file(&directory, STAGING, &stage, 0)?;
                stage.write_all(record.as_ref())?;
                stage.sync_all()?;
                validate_file(&directory, STAGING, &stage, RECORD_BYTES as u64)?;
                validate_directory(&parent, &name, &directory)?;
                // link is atomic and refuses replacement. A crash before stage
                // removal leaves nlink=2, which reopening deliberately rejects.
                directory.hard_link(STAGING, &directory, &key_file)?;
                let published = directory.symlink_metadata(&key_file)?;
                let staged = stage.metadata()?;
                if staged.nlink() != 2
                    || !identity_matches(&staged, &published)
                    || !identity_matches(&staged, &directory.symlink_metadata(STAGING)?)
                {
                    return Err(invalid("spill authority publication identity changed"));
                }
                directory.remove_file(STAGING)?;
                sync_directory(&directory)?;
                read_key(&directory, store_id)?
            }
            Err(error) => return Err(error),
        };
        validate_directory(&parent, &name, &directory)?;
        validate_file(&directory, LOCK_FILE, &bootstrap.file, 0)?;
        bootstrap.release()?;
        Ok(key)
    }
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{PermissionsExt as _, symlink};
    use std::sync::Barrier;

    fn id(byte: u8) -> StoreId {
        StoreId::from_bytes([byte; StoreId::LEN]).unwrap()
    }
    fn namespace(store: StoreId) -> String {
        use std::fmt::Write as _;
        let mut namespace = String::from("grafeo-store-");
        for byte in store.as_bytes() {
            write!(namespace, "{byte:02x}").unwrap();
        }
        namespace
    }
    fn fixture() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        // macOS's system temp path can contain the /var symlink; resolve the
        // test fixture's trusted parent before exercising no-follow admission.
        let root = temp.path().canonicalize().unwrap();
        let db = root.join("store.grafeo");
        fs::write(&db, b"test database owner").unwrap();
        let spill = root.join("spill");
        (temp, db, spill)
    }
    fn fingerprint(key: &KeyChain) -> Zeroizing<[u8; 32]> {
        key.derive_dek("authority-test", b"fixed")
    }

    // Keep derived key bytes out of assertion failure diagnostics.
    fn same_key(first: &KeyChain, second: &KeyChain) -> bool {
        *fingerprint(first) == *fingerprint(second)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn acl_grants_cannot_bootstrap_or_reopen_private_authority() {
        use std::process::Command;
        for target in ["parent", "directory", "key", "lock"] {
            let (_temp, db, spill) = fixture();
            let sidecar = db.with_file_name("store.grafeo.spill-auth");
            if target != "parent" {
                load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
            }
            let path = match target {
                "parent" => db.parent().unwrap().to_owned(),
                "directory" => sidecar.clone(),
                "key" => sidecar.join(format!("authority-{}", id(1))),
                _ => sidecar.join("bootstrap.lock"),
            };
            assert!(
                Command::new("/bin/chmod")
                    .args([
                        "+a",
                        if path.is_dir() {
                            "everyone allow read,file_inherit,directory_inherit"
                        } else {
                            "everyone allow read"
                        }
                    ])
                    .arg(&path)
                    .status()
                    .unwrap()
                    .success()
            );
            let error = load_or_create_key_chain(Some(&db), id(1), &spill)
                .err()
                .unwrap();
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied, "{target}");
            if target == "parent" {
                assert!(
                    !sidecar.exists(),
                    "no secret may be written below inheritable grants"
                );
            }
            assert!(
                Command::new("/bin/chmod")
                    .arg("-N")
                    .arg(&path)
                    .status()
                    .unwrap()
                    .success()
            );
            load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        }
    }

    #[test]
    fn key_survives_reopen_and_rejects_corrupt_record() {
        let (_temp, db, spill) = fixture();
        let first = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        let second = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        assert!(same_key(&first, &second));
        let key_path = db
            .with_file_name("store.grafeo.spill-auth")
            .join(format!("authority-{}", id(1)));
        fs::write(key_path, b"invalid").unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(1), &spill).is_err());
    }

    #[test]
    fn store_identity_roundtrip_preserves_both_authorities() {
        let (_temp, db, spill) = fixture();
        let first = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        fs::create_dir_all(spill.join(namespace(id(1)))).unwrap();
        let second = load_or_create_key_chain(Some(&db), id(2), &spill).unwrap();
        fs::create_dir_all(spill.join(namespace(id(2)))).unwrap();
        assert!(!same_key(&first, &second));
        let first_again = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        let second_again = load_or_create_key_chain(Some(&db), id(2), &spill).unwrap();
        assert!(same_key(&first, &first_again));
        assert!(same_key(&second, &second_again));
        let sidecar = db.with_file_name("store.grafeo.spill-auth");
        assert!(sidecar.join(format!("authority-{}", id(1))).is_file());
        assert!(sidecar.join(format!("authority-{}", id(2))).is_file());
    }

    #[test]
    fn foreign_record_copied_to_another_store_filename_is_rejected() {
        let (_temp, db, spill) = fixture();
        let original = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        let sidecar = db.with_file_name("store.grafeo.spill-auth");
        fs::copy(
            sidecar.join(format!("authority-{}", id(1))),
            sidecar.join(format!("authority-{}", id(2))),
        )
        .unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(2), &spill).is_err());
        let reopened = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        assert!(same_key(&original, &reopened));
        assert!(!spill.exists());
    }

    #[test]
    fn missing_authority_never_blesses_existing_namespace() {
        let (_temp, db, spill) = fixture();
        fs::create_dir_all(spill.join(namespace(id(1)))).unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(1), &spill).is_err());
        assert!(!db.with_file_name("store.grafeo.spill-auth").exists());
    }

    #[test]
    fn removed_key_is_not_regenerated_for_existing_namespace() {
        let (_temp, db, spill) = fixture();
        load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        let authority = db
            .with_file_name("store.grafeo.spill-auth")
            .join(format!("authority-{}", id(1)));
        fs::remove_file(&authority).unwrap();
        fs::create_dir_all(spill.join(namespace(id(1)))).unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(1), &spill).is_err());
        assert!(!authority.exists());
    }

    #[test]
    fn directory_database_and_two_stores_share_parent_without_sharing_keys() {
        let (_temp, db, spill) = fixture();
        let other = db.with_file_name("directory-store");
        fs::create_dir(&other).unwrap();
        let first = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        fs::create_dir_all(spill.join(namespace(id(1)))).unwrap();
        let second = load_or_create_key_chain(Some(&other), id(2), &spill).unwrap();
        assert!(!same_key(&first, &second));
        let reopened = load_or_create_key_chain(Some(&other), id(2), &spill).unwrap();
        assert!(same_key(&second, &reopened));
        assert!(
            other
                .join(".spill-auth")
                .join(format!("authority-{}", id(2)))
                .is_file()
        );
    }

    #[test]
    fn authority_rejects_symlinks_hardlinks_and_public_permissions() {
        let (_temp, db, spill) = fixture();
        load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        let sidecar = db.with_file_name("store.grafeo.spill-auth");
        let key = sidecar.join(format!("authority-{}", id(1)));
        let saved = sidecar.join("saved");
        fs::rename(&key, &saved).unwrap();
        symlink(&saved, &key).unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(1), &spill).is_err());
        fs::remove_file(&key).unwrap();
        fs::hard_link(&saved, &key).unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(1), &spill).is_err());
        fs::remove_file(saved).unwrap();
        fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_or_create_key_chain(Some(&db), id(1), &spill).is_err());
    }

    #[test]
    fn bootstrap_contention_is_bounded_and_reopen_keeps_key() {
        let (_temp, db, spill) = fixture();
        let first = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(
                db.with_file_name("store.grafeo.spill-auth")
                    .join("bootstrap.lock"),
            )
            .unwrap();
        lock.try_lock().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);
        let worker_db = db.clone();
        let worker_spill = spill.clone();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            load_or_create_key_chain(Some(&worker_db), id(1), &worker_spill)
                .err()
                .unwrap()
                .kind()
        });
        barrier.wait();
        assert_eq!(worker.join().unwrap(), io::ErrorKind::WouldBlock);
        lock.unlock().unwrap();
        let reopened = load_or_create_key_chain(Some(&db), id(1), &spill).unwrap();
        assert!(same_key(&first, &reopened));
    }

    #[test]
    fn in_memory_authorities_are_independent() {
        let first = load_or_create_key_chain(None, id(1), Path::new("unused")).unwrap();
        let second = load_or_create_key_chain(None, id(1), Path::new("unused")).unwrap();
        assert!(!same_key(&first, &second));
    }
}
