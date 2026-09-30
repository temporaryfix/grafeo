//! Dead-leaf reclamation retains both leases and exact file identities.

use super::super::SpillScavengeReport;
use super::*;
use crate::execution::spill::SpillIoOperation;
use crate::execution::spill::file::SpillFileLifecycle;
use crate::execution::spill::manager::{
    is_query_leaf_name, is_spill_artifact_name, marker_matches_leaf,
};
use crate::execution::spill::quota::{ReservationKey, sync_directory, validate_private_directory};
use cap_std::fs::MetadataExt as _;

const MAX_ROOT_ENTRIES: usize = 4096;
const MAX_LEAF_FILES: usize = 256;

enum Outcome {
    Removed,
    Deferred,
    Invalid,
}

struct Artifact {
    name: PathBuf,
    file: File,
    lifecycle: SpillFileLifecycle,
}

struct DeadLeaf<'a> {
    root: &'a SpillRoot,
    name: &'a str,
    directory: Arc<Dir>,
    _lock: File,
    marker: RetainedMarker,
    bytes: [u8; OWNER_MARKER_BYTES],
    device: u64,
    inode: u64,
}

impl DeadLeaf<'_> {
    fn validate_directory(&self) -> io::Result<()> {
        self.root.native.validate(&self.root.path)?;
        validate_private_directory(&self.root.native.directory)?;
        validate_private_directory(&self.directory)?;
        let named = self.root.native.directory.symlink_metadata(self.name)?;
        if !named.is_dir() || named.dev() != self.device || named.ino() != self.inode {
            return Err(invalid("abandoned spill leaf was substituted"));
        }
        Ok(())
    }

    fn validate(&self) -> io::Result<()> {
        self.validate_directory()?;
        self.marker
            .validate(&self.directory, OWNER_MARKER, &self.bytes)
    }

    fn artifacts(&self) -> io::Result<Option<Vec<Artifact>>> {
        let mut files = Vec::new();
        for entry in self.directory.entries()? {
            let entry = entry?;
            let name = entry.file_name();
            if name == OWNER_MARKER {
                continue;
            }
            if files.len() == MAX_LEAF_FILES {
                return Ok(None);
            }
            if !name.to_str().is_some_and(is_spill_artifact_name) {
                return Err(invalid("unknown content in abandoned spill leaf"));
            }
            // Retain every admitted inode before any mutation. This bounds both
            // heap and descriptor use and rejects an oversized leaf intact.
            let file = open_marker(&self.directory, name.to_str().expect("checked name"), false)?;
            let name = PathBuf::from(name);
            let lifecycle = SpillFileLifecycle::capture_with_directory(
                &file,
                Arc::clone(&self.directory),
                name.clone(),
            )?;
            lifecycle.validate_entry(&name)?;
            files.push(Artifact {
                name,
                file,
                lifecycle,
            });
        }
        Ok(Some(files))
    }

    fn remove(self, files: Vec<Artifact>, key: ReservationKey, birth: u64) -> io::Result<()> {
        for artifact in files {
            self.root.io.check(SpillIoOperation::ScavengeDelete)?;
            self.validate()?;
            // The shared deletion receipt rejects replacement inodes. Check
            // fresh aliases/permissions as well before unlinking retained data.
            let metadata = artifact.file.metadata()?;
            if metadata.nlink() != 1
                || metadata.uid() != rustix::process::geteuid().as_raw()
                || metadata.mode() & 0o077 != 0
            {
                return Err(invalid("abandoned spill artifact acquired foreign aliases"));
            }
            #[cfg(target_os = "macos")]
            crate::execution::spill::validate_no_acl_grants(&artifact.file)?;
            artifact.lifecycle.validate_entry(&artifact.name)?;
            artifact.lifecycle.delete_path(&artifact.name, false)?;
            if artifact.file.metadata()?.nlink() != 0 {
                return Err(invalid("abandoned spill artifact deletion is unproved"));
            }
            // Close retained allocation handles before any quota credit.
            drop(artifact);
        }
        self.validate()?;
        // New unknown content must prevent marker removal as well as rmdir.
        for entry in self.directory.entries()? {
            if entry?.file_name() != OWNER_MARKER {
                return Err(invalid("abandoned spill leaf changed during cleanup"));
            }
        }
        self._lock.sync_all()?;
        self.directory.remove_file(OWNER_MARKER)?;
        let removed = (|| {
            self.root.io.check(SpillIoOperation::RemoveQueryDirectory)?;
            self.validate_directory()?;
            self.root.native.directory.remove_dir(self.name)
        })();
        if let Err(primary) = removed {
            // Recreate only under the still-locked exact directory; create_new
            // never overwrites a replacement seal. A failed restore retains debt.
            let primary = std::mem::ManuallyDrop::new(primary);
            let restored = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.root.io.check(SpillIoOperation::RestoreOwnerMarker)?;
                self.validate_directory()?;
                let mut marker = open_marker(&self.directory, OWNER_MARKER, true)?;
                marker.try_lock().map_err(io::Error::from)?;
                marker.write_all(&self.bytes)?;
                marker.sync_all()?;
                sync_directory(&self.directory)
            }));
            return match restored {
                Ok(Ok(())) => Err(std::mem::ManuallyDrop::into_inner(primary)),
                Ok(Err(secondary)) => Err(crate::execution::spill::combine_primary_and_cleanup(
                    std::mem::ManuallyDrop::into_inner(primary),
                    secondary,
                    "owner marker restore",
                )),
                Err(payload) => {
                    // Retain the original error on unwind rather than running a
                    // foreign destructor while propagating the restore panic.
                    std::panic::resume_unwind(payload)
                }
            };
        }
        self.root.io.check(SpillIoOperation::QuotaDeleteSync)?;
        self.root.native.validate(&self.root.path)?;
        validate_private_directory(&self.root.native.directory)?;
        if self.marker.file.metadata()?.nlink() != 0 {
            return Err(invalid("abandoned spill marker deletion is unproved"));
        }
        let metadata = self._lock.metadata()?;
        let unlinked = metadata.nlink() == 0;
        #[cfg(target_os = "macos")]
        let unlinked = unlinked
            || crate::execution::spill::quota::macos_directory_absent(&self._lock, metadata.ino())?;
        if !unlinked {
            return Err(invalid("abandoned spill directory deletion is unproved"));
        }
        sync_directory(&self.root.native.directory)?;
        // No allocation handles or inode leases survive the durable credit.
        let ledger = Arc::clone(&self.root.ledger);
        drop(self);
        ledger.release_after_delete(key, birth, &mut None)
    }
}

impl RootCapability {
    pub(in super::super) fn scavenge(&self, root: &SpillRoot) -> io::Result<SpillScavengeReport> {
        self.validate(&root.path)?;
        validate_private_directory(&self.directory)?;
        let mut report = SpillScavengeReport::default();
        for (index, entry) in self.directory.entries()?.enumerate() {
            if index == MAX_ROOT_ENTRIES {
                report.truncated = true;
                break;
            }
            let entry = entry?;
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some(
                    ".grafeo-spill-root"
                        | ".grafeo-spill-quota"
                        | ".grafeo-spill-quota.lock"
                        | ".grafeo-spill-quota.installing"
                )
            ) {
                continue;
            }
            report.inspected += 1;
            let Some(name) = name.to_str().filter(|name| is_query_leaf_name(name)) else {
                report.foreign += 1;
                continue;
            };
            let observed = match self.directory.symlink_metadata(name) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    report.deferred += 1;
                    continue;
                }
                Err(error) => return Err(error),
            };
            if !observed.is_dir() {
                report.invalid += 1;
                continue;
            }
            match self.scavenge_leaf(root, name, (observed.dev(), observed.ino()))? {
                Outcome::Removed => report.removed += 1,
                Outcome::Deferred => report.deferred += 1,
                Outcome::Invalid => report.invalid += 1,
            }
        }
        self.validate(&root.path)?;
        validate_private_directory(&self.directory)?;
        report.reserved_bytes = root.ledger.reserved_bytes()?;
        Ok(report)
    }

    fn scavenge_leaf(
        &self,
        root: &SpillRoot,
        name: &str,
        observed: (u64, u64),
    ) -> io::Result<Outcome> {
        root.io.check(SpillIoOperation::ScavengeOpen)?;
        self.validate(&root.path)?;
        validate_private_directory(&self.directory)?;
        let directory = match self.directory.open_dir_nofollow(name) {
            Ok(directory) => Arc::new(directory),
            Err(_) => return Ok(Outcome::Invalid),
        };
        if validate_private_directory(&directory).is_err() {
            return Ok(Outcome::Invalid);
        }
        let lock = directory_io_handle(&directory)?;
        let metadata = lock.metadata()?;
        if (metadata.dev(), metadata.ino()) != observed {
            return Ok(Outcome::Invalid);
        }
        match lock.try_lock().map_err(io::Error::from) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return Ok(Outcome::Deferred);
            }
            Err(error) => return Err(error),
        }
        let marker = match open_marker(&directory, OWNER_MARKER, false) {
            Ok(marker) => Arc::new(marker),
            Err(_) => return Ok(Outcome::Invalid),
        };
        match marker.try_lock().map_err(io::Error::from) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return Ok(Outcome::Deferred);
            }
            Err(error) => return Err(error),
        }
        let Ok(bytes) = read_exact_marker::<OWNER_MARKER_BYTES>(&marker) else {
            return Ok(Outcome::Invalid);
        };
        if &bytes[..4] != b"GRAQ" || bytes[4] != 3 || bytes[53..157] != self.bytes[5..109] {
            return Ok(Outcome::Invalid);
        }
        let identity =
            SpillQueryIdentity::from_bytes(bytes[5..21].try_into().expect("fixed identity"));
        if !marker_matches_leaf(name, identity) {
            return Ok(Outcome::Invalid);
        }
        let message = leaf_auth_message(&bytes, &lock)?;
        if !root
            .authority
            .verify_marker(&message, bytes[21..53].try_into().expect("fixed auth"))?
        {
            return Ok(Outcome::Invalid);
        }
        let leaf = DeadLeaf {
            root,
            name,
            directory,
            _lock: lock,
            marker: RetainedMarker::new(marker)?,
            bytes,
            device: observed.0,
            inode: observed.1,
        };
        root.io.check(SpillIoOperation::ScavengeValidate)?;
        leaf.validate()?;
        let key = ReservationKey::leaf(*identity.as_bytes());
        let Some((birth, _)) = root.ledger.recovery_reservation(key)? else {
            return Ok(Outcome::Invalid);
        };
        leaf.validate()?;
        let files = match leaf.artifacts() {
            Ok(Some(files)) => files,
            Ok(None) => return Ok(Outcome::Deferred),
            Err(_) => return Ok(Outcome::Invalid),
        };
        leaf.remove(files, key, birth)?;
        Ok(Outcome::Removed)
    }
}
