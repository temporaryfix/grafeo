//! Bounded durable accounting for an authenticated synchronous spill root.
//!
//! The ledger is streamed through fixed-size buffers. A stable lock inode spans
//! validation, installing-file sync, rename and directory sync. File data may
//! only consume a reservation after that transaction succeeds.

use super::manager::PhysicalDirectoryIdentity;
use super::{SpillIo, SpillIoOperation, SpillQuotaExceeded, SpillRootAuthority};
use crate::execution::QueryCancellationToken;
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt as _, OpenOptionsSyncExt as _};
use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt as _};
use parking_lot::{Mutex, MutexGuard};
use std::fs::File;
use std::io::{self, Read as _, Seek as _, Write as _};
use std::os::unix::fs::{FileExt as _, MetadataExt as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const LEDGER: &str = ".grafeo-spill-quota";
const INSTALLING: &str = ".grafeo-spill-quota.installing";
const LOCK: &str = ".grafeo-spill-quota.lock";
const HEADER_BYTES: usize = 144;
const ENTRY_BYTES: usize = 80;
const MAX_ENTRIES: u64 = 4096;
const MAGIC: &[u8; 8] = b"GRAQLED1";
const LOCK_ATTEMPTS: usize = 512;

// Reserve conservatively for allocation rounding, directory entries and both
// the published and installing ledger. Filesystems with larger allocation units
// must be rejected by root admission rather than silently under-accounted.
pub(super) const ALLOCATION_UNIT: u64 = 64 * 1024;
pub(super) const CONTROL_RESERVE: u64 = 16 * ALLOCATION_UNIT;
pub(super) const LEAF_RESERVE: u64 = 2 * ALLOCATION_UNIT;
pub(super) const FILE_RESERVE: u64 = ALLOCATION_UNIT;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ReservationKey([u8; 33]);

impl ReservationKey {
    pub(super) fn leaf(query: [u8; 16]) -> Self {
        let mut key = [0; 33];
        key[1..17].copy_from_slice(&query);
        Self(key)
    }

    pub(super) fn file(query: [u8; 16], file: [u8; 16]) -> Self {
        let mut key = Self::leaf(query).0;
        key[0] = 1;
        key[17..].copy_from_slice(&file);
        Self(key)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    key: ReservationKey,
    birth: u64,
    charged: u64,
    logical: u64,
}

impl Entry {
    fn encode(self) -> [u8; ENTRY_BYTES] {
        let mut bytes = [0; ENTRY_BYTES];
        bytes[..33].copy_from_slice(&self.key.0);
        bytes[33..41].copy_from_slice(&self.birth.to_le_bytes());
        bytes[41..49].copy_from_slice(&self.charged.to_le_bytes());
        bytes[49..57].copy_from_slice(&self.logical.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8; ENTRY_BYTES], generation: u64) -> io::Result<Self> {
        let result = Self {
            key: ReservationKey(bytes[..33].try_into().expect("fixed key width")),
            birth: number(&bytes[33..41]),
            charged: number(&bytes[41..49]),
            logical: number(&bytes[49..57]),
        };
        if bytes[57..].iter().any(|byte| *byte != 0)
            || result.birth == 0
            || result.birth > generation
            || result.key.0[0] > 1
            || (result.key.0[0] == 0
                && (result.key.0[17..].iter().any(|byte| *byte != 0)
                    || result.logical != 0
                    || result.charged < LEAF_RESERVE))
            || (result.key.0[0] == 1 && result.charged < file_charge(result.logical, 0)?)
        {
            return Err(invalid("invalid spill quota entry"));
        }
        Ok(result)
    }
}

fn number(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("fixed ledger integer width"))
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(super) fn file_charge(logical: u64, allocated: u64) -> io::Result<u64> {
    let rounded = logical
        .checked_add(ALLOCATION_UNIT - 1)
        .map(|value| value / ALLOCATION_UNIT * ALLOCATION_UNIT)
        .ok_or_else(|| invalid("spill allocation rounding overflow"))?;
    rounded
        .max(allocated)
        .checked_add(FILE_RESERVE)
        .ok_or_else(|| invalid("spill physical reservation overflow"))
}

fn open_nofollow(directory: &Dir, name: &str, create: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create_new(create)
        .mode(0o600)
        .follow(FollowSymlinks::No)
        .nonblock(true);
    Ok(directory.open_with(name, &options)?.into_std())
}

fn private_file(directory: &Dir, name: &str, create: bool) -> io::Result<File> {
    let file = open_nofollow(directory, name, create)?;
    validate_named_file(directory, name, &file)?;
    Ok(file)
}

// All ledger names are single components beneath a retained directory. Direct
// no-follow statat avoids an unnecessary capability path walk for each check.
// Return the length from the same fresh descriptor observation.
fn validate_named_file(directory: &Dir, name: &str, file: &File) -> io::Result<u64> {
    use rustix::fs::{AtFlags, FileType, fstat, statat};
    if name.is_empty() || name == "." || name == ".." || name.contains('/') {
        return Err(invalid("quota metadata name is not one component"));
    }
    let held = fstat(file)?;
    let named = statat(directory, name, AtFlags::SYMLINK_NOFOLLOW)?;
    if FileType::from_raw_mode(held.st_mode) != FileType::RegularFile
        || held.st_nlink != 1
        || held.st_uid != rustix::process::geteuid().as_raw()
        || held.st_mode & 0o077 != 0
        || FileType::from_raw_mode(named.st_mode) != FileType::RegularFile
        || named.st_dev != held.st_dev
        || named.st_ino != held.st_ino
    {
        return Err(invalid(
            "spill quota file identity or private ownership changed",
        ));
    }
    #[cfg(target_os = "macos")]
    super::validate_no_acl_grants(file)?;
    u64::try_from(held.st_size).map_err(|_| invalid("negative quota metadata length"))
}

pub(super) fn sync_directory(directory: &Dir) -> io::Result<()> {
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

struct HeldLock<'a> {
    file: &'a File,
    process: u32,
    locked: bool,
}

struct InstallingFile {
    file: File,
    directory: Arc<Dir>,
    armed: bool,
}

impl InstallingFile {
    fn cleanup(&mut self) -> io::Result<()> {
        if !self.armed {
            return Ok(());
        }
        match self.directory.symlink_metadata(INSTALLING) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if self.file.metadata()?.nlink() != 0 {
                    return Err(invalid("owned quota staging inode was moved"));
                }
            }
            Err(error) => return Err(error),
            Ok(_) => {
                validate_named_file(&self.directory, INSTALLING, &self.file)?;
                self.directory.remove_file(INSTALLING)?;
            }
        }
        sync_directory(&self.directory)?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for InstallingFile {
    fn drop(&mut self) {
        super::run_cleanup_backstop(|| self.cleanup());
    }
}

fn write_entry(
    file: &mut File,
    digest: &mut blake3::Hasher,
    count: &mut u64,
    used: &mut u64,
    entry: Entry,
) -> io::Result<()> {
    let bytes = entry.encode();
    file.write_all(&bytes)?;
    digest.update(&bytes);
    *count += 1;
    *used = used
        .checked_add(entry.charged)
        .ok_or_else(|| invalid("spill quota total overflow"))?;
    Ok(())
}

impl HeldLock<'_> {
    fn release(mut self) -> io::Result<()> {
        if self.process != std::process::id() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "spill quota lock belongs to another process",
            ));
        }
        let result = self.file.unlock();
        if result.is_ok() {
            self.locked = false;
        }
        result
    }
}

impl Drop for HeldLock<'_> {
    fn drop(&mut self) {
        // An inherited descriptor must not explicitly unlock its parent's OFD.
        if self.locked && self.process == std::process::id() {
            let _ = self.file.unlock();
        }
    }
}

#[derive(Clone, Copy)]
struct Header {
    bytes: [u8; HEADER_BYTES],
    generation: u64,
    limit: u64,
    used: u64,
    count: u64,
}

pub(super) struct RootLedger {
    directory: Arc<Dir>,
    path: PathBuf,
    identity: PhysicalDirectoryIdentity,
    binding: [u8; 32],
    authority: Arc<dyn SpillRootAuthority>,
    io: Arc<dyn SpillIo>,
    lock: File,
    root_marker: File,
    limit: u64,
    process: u32,
    // Also serializes threads using the same flock open-file description.
    observed_generation: Mutex<u64>,
}

impl RootLedger {
    pub(super) fn validate_limit(limit: u64) -> io::Result<()> {
        Self::admit(limit, 0, CONTROL_RESERVE)
    }

    pub(super) fn open(
        directory: Arc<Dir>,
        path: &Path,
        binding: [u8; 32],
        authority: Arc<dyn SpillRootAuthority>,
        io: Arc<dyn SpillIo>,
        limit: u64,
    ) -> io::Result<Arc<Self>> {
        Self::admit(limit, 0, CONTROL_RESERVE)?;
        let filesystem = rustix::fs::fstatvfs(directory.as_ref())?;
        // f_frsize is the allocation unit; f_bsize is the preferred I/O
        // request length (1 MiB on APFS despite its 4 KiB allocation unit).
        let unit = filesystem.f_frsize;
        if unit == 0 || unit > ALLOCATION_UNIT {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "spill filesystem allocation unit exceeds the qualified quota allowance",
            ));
        }
        let identity = PhysicalDirectoryIdentity::capture_capability(&directory, path)?;
        let root_marker = private_file(&directory, ".grafeo-spill-root", false)?;
        let lock = match private_file(&directory, LOCK, true) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                private_file(&directory, LOCK, false)?
            }
            Err(error) => return Err(error),
        };
        let result = Arc::new(Self {
            directory,
            path: path.to_owned(),
            identity,
            binding,
            authority,
            io,
            lock,
            root_marker,
            limit,
            process: std::process::id(),
            observed_generation: Mutex::new(0),
        });
        let mut observed = result.generation_guard(None)?;
        let held = result.acquire(None)?;
        match open_nofollow(&result.directory, LEDGER, false) {
            Ok(mut file) => {
                let (header, _) = result.read(&mut file, None)?;
                if header.limit != limit {
                    return Err(invalid("spill root budget differs from its durable policy"));
                }
                *observed = header.generation;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // Missing accounting never blesses known or unknown spill leaves.
                for entry in result.directory.entries()? {
                    let name = entry?.file_name();
                    if name != ".grafeo-spill-root" && name != LOCK {
                        return Err(invalid("spill root has content but no quota authority"));
                    }
                }
                let header = Header {
                    bytes: [0; HEADER_BYTES],
                    generation: 1,
                    limit,
                    used: CONTROL_RESERVE,
                    count: 0,
                };
                result.install(None, header, None, None, None)?;
                *observed = 1;
            }
            Err(error) => return Err(error),
        }
        held.release()?;
        drop(observed);
        Ok(result)
    }

    fn check_cancelled(cancellation: Option<&QueryCancellationToken>) -> io::Result<()> {
        if let Some(cancellation) = cancellation {
            cancellation
                .check()
                .map_err(|reason| io::Error::new(io::ErrorKind::Interrupted, reason))?;
        }
        Ok(())
    }

    fn generation_guard(
        &self,
        cancellation: Option<&QueryCancellationToken>,
    ) -> io::Result<MutexGuard<'_, u64>> {
        if self.process != std::process::id() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "spill quota authority belongs to another process",
            ));
        }
        for _ in 0..LOCK_ATTEMPTS {
            Self::check_cancelled(cancellation)?;
            if let Some(guard) = self
                .observed_generation
                .try_lock_for(std::time::Duration::from_millis(1))
            {
                return Ok(guard);
            }
        }
        Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "spill quota owner is busy",
        ))
    }

    fn validate_authority(&self) -> io::Result<()> {
        if self.process != std::process::id() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "spill quota authority belongs to another process",
            ));
        }
        self.identity.validate_path(&self.path)?;
        self.identity.validate_capability(&self.directory)?;
        validate_private_directory(&self.directory)?;
        let marker = validate_named_file(&self.directory, ".grafeo-spill-root", &self.root_marker)?;
        let length = usize::try_from(marker)
            .ok()
            .filter(|length| *length <= 512)
            .ok_or_else(|| invalid("spill root marker exceeds its authority bound"))?;
        let mut root_bytes = [0; 512];
        self.root_marker
            .read_exact_at(&mut root_bytes[..length], 0)?;
        if blake3::hash(&root_bytes[..length]).as_bytes() != &self.binding {
            return Err(invalid("spill root authority changed under its quota"));
        }
        if validate_named_file(&self.directory, LOCK, &self.lock)? != 0 {
            return Err(invalid("spill quota lock contains unknown data"));
        }
        Ok(())
    }

    fn acquire(&self, cancellation: Option<&QueryCancellationToken>) -> io::Result<HeldLock<'_>> {
        // Attest the stable lock inode before waiting. Every caller then runs
        // read/read_current, whose final authority check follows the provider
        // callback while this lock is held. Bootstrap instead validates in
        // install before creating its stage. No admission or mutation precedes
        // that full check, so duplicating it around flock adds no authority.
        if validate_named_file(&self.directory, LOCK, &self.lock)? != 0 {
            return Err(invalid("spill quota lock contains unknown data"));
        }
        for _ in 0..LOCK_ATTEMPTS {
            Self::check_cancelled(cancellation)?;
            match self.lock.try_lock().map_err(io::Error::from) {
                Ok(()) => {
                    let guard = HeldLock {
                        file: &self.lock,
                        process: self.process,
                        locked: true,
                    };
                    return Ok(guard);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "spill quota ledger is busy",
        ))
    }

    /// Creates one move-only reservation identity. Its birth generation prevents
    /// an old receipt releasing a later reservation that reused the same name.
    pub(super) fn reserve(
        &self,
        key: ReservationKey,
        cancellation: Option<&QueryCancellationToken>,
    ) -> io::Result<(u64, u64)> {
        if key.0[0] != 0 {
            return Err(invalid("only query capacity is durable"));
        }
        self.io.check(SpillIoOperation::QuotaReserve)?;
        let mut observed = self.generation_guard(cancellation)?;
        let held = self.acquire(cancellation)?;
        let mut file = open_nofollow(&self.directory, LEDGER, false)?;
        let (header, existing) = self.read_current(&mut file, key, *observed)?;
        if existing.is_some() {
            return Err(invalid("spill quota reservation identity already exists"));
        }
        if header.count == MAX_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::QuotaExceeded,
                "spill quota ledger entry envelope exhausted",
            ));
        }
        let charged = LEAF_RESERVE;
        Self::admit(header.limit, header.used, charged)?;
        let generation = header
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("spill quota generation exhausted"))?;
        let entry = Entry {
            key,
            birth: generation,
            charged,
            logical: 0,
        };
        let next = Header {
            generation,
            used: header.used + charged,
            count: header.count + 1,
            ..header
        };
        self.install(Some((&mut file, header)), next, None, Some(entry), None)?;
        *observed = generation;
        drop(file);
        held.release()?;
        Ok((generation, charged))
    }

    /// A query retains a durable high-water capacity until its leaf is deleted.
    /// Grow geometrically, capped at 8 MiB extra, with exact admission fallback.
    /// Measured allocation is debt even when admission fails.
    pub(super) fn ensure_capacity(
        &self,
        key: ReservationKey,
        birth: u64,
        required: u64,
        measured: bool,
        cancellation: Option<&QueryCancellationToken>,
    ) -> io::Result<(u64, io::Result<()>)> {
        if key.0[0] != 0 || required < LEAF_RESERVE {
            return Err(invalid("invalid query capacity request"));
        }
        let mut observed = self.generation_guard(cancellation)?;
        let held = self.acquire(cancellation)?;
        let mut file = open_nofollow(&self.directory, LEDGER, false)?;
        let (header, entry) = self.read_current(&mut file, key, *observed)?;
        let mut entry = entry
            .filter(|entry| entry.birth == birth)
            .ok_or_else(|| invalid("stale or foreign spill quota receipt"))?;
        let required = required.max(entry.charged);
        let admission = Self::admit(header.limit, header.used, required - entry.charged);
        if admission.is_err() && !measured {
            return admission.map(|()| (entry.charged, Ok(())));
        }
        if required == entry.charged {
            *observed = header.generation;
            drop(file);
            held.release()?;
            return Ok((entry.charged, admission));
        }
        self.io.check(SpillIoOperation::QuotaReserve)?;
        let preferred = required.max(entry.charged.saturating_add(entry.charged.min(8 << 20)));
        let charged = if Self::admit(header.limit, header.used, preferred - entry.charged).is_ok() {
            preferred
        } else {
            required
        };
        let generation = header
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("spill quota generation exhausted"))?;
        let next = Header {
            generation,
            used: header
                .used
                .checked_add(charged - entry.charged)
                .ok_or_else(|| invalid("spill physical debt overflow"))?,
            ..header
        };
        entry.charged = charged;
        self.install(
            Some((&mut file, header)),
            next,
            Some(key),
            Some(entry),
            None,
        )?;
        *observed = generation;
        drop(file);
        held.release()?;
        Ok((charged, admission))
    }

    fn validate_reservation(&self, key: ReservationKey, birth: u64) -> io::Result<()> {
        let mut observed = self.generation_guard(None)?;
        let held = self.acquire(None)?;
        let mut file = open_nofollow(&self.directory, LEDGER, false)?;
        let (header, entry) = self.read_current(&mut file, key, *observed)?;
        if entry.is_none_or(|entry| entry.birth != birth) {
            return Err(invalid("stale or foreign spill quota receipt"));
        }
        *observed = header.generation;
        drop(file);
        held.release()
    }

    /// Snapshot authenticated debt; only the root's exclusive dead-leaf owner
    /// may turn this birth receipt into a deletion credit.
    pub(super) fn recovery_reservation(
        &self,
        key: ReservationKey,
    ) -> io::Result<Option<(u64, u64)>> {
        let mut observed = self.generation_guard(None)?;
        let held = self.acquire(None)?;
        let mut file = open_nofollow(&self.directory, LEDGER, false)?;
        let (header, entry) = self.read_current(&mut file, key, *observed)?;
        *observed = header.generation;
        drop(file);
        held.release()?;
        Ok(entry.map(|entry| (entry.birth, entry.charged)))
    }

    pub(super) fn reserved_bytes(&self) -> io::Result<u64> {
        let mut observed = self.generation_guard(None)?;
        let held = self.acquire(None)?;
        let mut file = open_nofollow(&self.directory, LEDGER, false)?;
        let (header, _) = self.read_current(&mut file, ReservationKey::leaf([0; 16]), *observed)?;
        *observed = header.generation;
        drop(file);
        held.release()?;
        Ok(header.used)
    }

    /// Caller must own the creation receipt and have durably confirmed deletion.
    /// Drop alone never invokes this operation or reduces persistent debt.
    pub(super) fn release_after_delete(
        &self,
        key: ReservationKey,
        birth: u64,
        pending: &mut Option<u64>,
    ) -> io::Result<()> {
        self.io.check(SpillIoOperation::QuotaRelease)?;
        let cancellation = None;
        let mut observed = self.generation_guard(cancellation)?;
        let held = self.acquire(cancellation)?;
        let mut file = open_nofollow(&self.directory, LEDGER, false)?;
        let (header, entry) = self.read_current(&mut file, key, *observed)?;
        if let Some(generation) = *pending {
            if entry.is_some() || header.generation < generation {
                return Err(invalid("pending quota release was replaced"));
            }
            self.io.check(SpillIoOperation::QuotaSync)?;
            sync_directory(&self.directory)?;
            *observed = header.generation;
            drop(file);
            held.release()?;
            *pending = None;
            return Ok(());
        }
        let entry = entry
            .filter(|entry| entry.birth == birth)
            .ok_or_else(|| invalid("stale or foreign spill quota release"))?;
        let generation = header
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("spill quota generation exhausted"))?;
        let next = Header {
            generation,
            used: header.used - entry.charged,
            count: header.count - 1,
            ..header
        };
        self.install(
            Some((&mut file, header)),
            next,
            Some(key),
            None,
            Some(pending),
        )?;
        *observed = generation;
        drop(file);
        held.release()?;
        *pending = None;
        Ok(())
    }

    fn read_current(
        &self,
        file: &mut File,
        key: ReservationKey,
        observed: u64,
    ) -> io::Result<(Header, Option<Entry>)> {
        let result = self.read(file, Some(key))?;
        if result.0.generation < observed || result.0.limit != self.limit {
            return Err(invalid("spill quota generation or budget was replaced"));
        }
        Ok(result)
    }

    fn admit(limit: u64, used: u64, additional: u64) -> io::Result<()> {
        if used > limit || additional > limit - used {
            return Err(io::Error::new(
                io::ErrorKind::QuotaExceeded,
                SpillQuotaExceeded::from_usage(limit, used, additional),
            ));
        }
        Ok(())
    }

    fn read(
        &self,
        file: &mut File,
        wanted: Option<ReservationKey>,
    ) -> io::Result<(Header, Option<Entry>)> {
        self.read_named(file, LEDGER, wanted)
    }

    fn read_named(
        &self,
        file: &mut File,
        name: &str,
        wanted: Option<ReservationKey>,
    ) -> io::Result<(Header, Option<Entry>)> {
        self.read_named_expected(file, name, wanted, None)
    }

    // A header already authenticated in this locked transaction is a byte
    // commitment to its entries. Rechecking that exact commitment and digest
    // needs no second provider callback (and creates no new callback boundary).
    fn read_named_expected(
        &self,
        file: &mut File,
        name: &str,
        wanted: Option<ReservationKey>,
        expected: Option<&[u8; HEADER_BYTES]>,
    ) -> io::Result<(Header, Option<Entry>)> {
        validate_named_file(&self.directory, name, file)?;
        file.seek(io::SeekFrom::Start(0))?;
        let mut bytes = [0; HEADER_BYTES];
        file.read_exact(&mut bytes)?;
        if &bytes[..8] != MAGIC
            || bytes[8..16] != 1u64.to_le_bytes()
            || bytes[16..48] != self.binding
            || match expected {
                Some(expected) => &bytes != expected,
                None => !self.authority.verify_marker(
                    &bytes[..112],
                    bytes[112..].try_into().expect("fixed authenticator width"),
                )?,
            }
        {
            return Err(invalid("unknown or unauthenticated spill quota ledger"));
        }
        let header = Header {
            generation: number(&bytes[48..56]),
            limit: number(&bytes[56..64]),
            used: number(&bytes[64..72]),
            count: number(&bytes[72..80]),
            bytes,
        };
        if header.generation == 0 || header.count > MAX_ENTRIES {
            return Err(invalid("spill quota ledger exceeds its fixed envelope"));
        }
        let mut hasher = blake3::Hasher::new();
        let mut previous = None;
        let mut used = CONTROL_RESERVE;
        let mut found = None;
        for _ in 0..header.count {
            let mut encoded = [0; ENTRY_BYTES];
            file.read_exact(&mut encoded)?;
            hasher.update(&encoded);
            let entry = Entry::decode(&encoded, header.generation)?;
            if previous.is_some_and(|key| key >= entry.key) {
                return Err(invalid("spill quota entries are not unique and ordered"));
            }
            previous = Some(entry.key);
            used = used
                .checked_add(entry.charged)
                .ok_or_else(|| invalid("spill quota total overflow"))?;
            if wanted == Some(entry.key) {
                found = Some(entry);
            }
        }
        if hasher.finalize().as_bytes() != &header.bytes[80..112] || used != header.used {
            return Err(invalid("spill quota ledger digest or total is invalid"));
        }
        // The bounded entry count controls reads; verify exact length using
        // the final identity check, after any authority callback and reads.
        if validate_named_file(&self.directory, name, file)?
            != HEADER_BYTES as u64 + header.count * ENTRY_BYTES as u64
        {
            return Err(invalid(
                "spill quota ledger length changed during validation",
            ));
        }
        let mut current = [0; HEADER_BYTES];
        file.read_exact_at(&mut current, 0)?;
        if current != header.bytes {
            return Err(invalid("spill quota header changed during validation"));
        }
        if expected.is_none() {
            self.validate_authority()?;
        }
        Ok((header, found))
    }

    fn install(
        &self,
        old: Option<(&mut File, Header)>,
        next: Header,
        removed: Option<ReservationKey>,
        replacement: Option<Entry>,
        published: Option<&mut Option<u64>>,
    ) -> io::Result<()> {
        self.io.check(SpillIoOperation::QuotaWrite)?;
        self.validate_authority()?;
        let mut stage = InstallingFile {
            file: private_file(&self.directory, INSTALLING, true)?,
            directory: Arc::clone(&self.directory),
            armed: true,
        };
        let result = self.install_owned(&mut stage, old, next, removed, replacement, published);
        match result {
            Ok(()) => Ok(()),
            Err(primary) => match stage.cleanup() {
                Ok(()) => Err(primary),
                Err(cleanup) => Err(super::combine_primary_and_cleanup(
                    primary,
                    cleanup,
                    "quota staging cleanup",
                )),
            },
        }
    }

    fn install_owned(
        &self,
        stage: &mut InstallingFile,
        mut old: Option<(&mut File, Header)>,
        mut next: Header,
        removed: Option<ReservationKey>,
        mut replacement: Option<Entry>,
        published: Option<&mut Option<u64>>,
    ) -> io::Result<()> {
        stage.file.write_all(&[0; HEADER_BYTES])?;
        let mut digest = blake3::Hasher::new();
        let mut count = 0_u64;
        let mut used = CONTROL_RESERVE;
        if let Some((file, previous)) = old.as_mut() {
            file.seek(io::SeekFrom::Start(HEADER_BYTES as u64))?;
            let mut old_digest = blake3::Hasher::new();
            for _ in 0..previous.count {
                let mut bytes = [0; ENTRY_BYTES];
                file.read_exact(&mut bytes)?;
                old_digest.update(&bytes);
                let entry = Entry::decode(&bytes, previous.generation)?;
                if replacement.is_some_and(|item| item.key < entry.key) {
                    write_entry(
                        &mut stage.file,
                        &mut digest,
                        &mut count,
                        &mut used,
                        replacement.take().expect("checked replacement"),
                    )?;
                }
                if removed != Some(entry.key) {
                    write_entry(&mut stage.file, &mut digest, &mut count, &mut used, entry)?;
                }
            }
            if old_digest.finalize().as_bytes() != &previous.bytes[80..112] {
                return Err(invalid("spill quota ledger changed during replacement"));
            }
            let mut header = [0; HEADER_BYTES];
            file.read_exact_at(&mut header, 0)?;
            if header != previous.bytes {
                return Err(invalid("spill quota header changed during replacement"));
            }
            validate_named_file(&self.directory, LEDGER, file)?;
        }
        if let Some(entry) = replacement {
            write_entry(&mut stage.file, &mut digest, &mut count, &mut used, entry)?;
        }
        if count != next.count || used != next.used || count > MAX_ENTRIES {
            return Err(invalid("spill quota replacement does not match admission"));
        }
        next.bytes = [0; HEADER_BYTES];
        next.bytes[..8].copy_from_slice(MAGIC);
        next.bytes[8..16].copy_from_slice(&1_u64.to_le_bytes());
        next.bytes[16..48].copy_from_slice(&self.binding);
        next.bytes[48..56].copy_from_slice(&next.generation.to_le_bytes());
        next.bytes[56..64].copy_from_slice(&next.limit.to_le_bytes());
        next.bytes[64..72].copy_from_slice(&next.used.to_le_bytes());
        next.bytes[72..80].copy_from_slice(&next.count.to_le_bytes());
        next.bytes[80..112].copy_from_slice(digest.finalize().as_bytes());
        let auth = self.authority.authenticate_marker(&next.bytes[..112])?;
        next.bytes[112..].copy_from_slice(&auth);
        stage.file.write_all_at(&next.bytes, 0)?;
        self.io.check(SpillIoOperation::QuotaSync)?;
        stage.file.sync_all()?;
        self.io.check(SpillIoOperation::QuotaPublish)?;
        let (installed, _) =
            self.read_named_expected(&mut stage.file, INSTALLING, None, Some(&next.bytes))?;
        if installed.bytes != next.bytes {
            return Err(invalid("spill quota installing bytes were substituted"));
        }
        // The publication hook is an adversarial boundary. Recheck the exact
        // old entry after it, immediately before replacing that entry.
        if let Some((file, previous)) = old {
            let (current, _) =
                self.read_named_expected(file, LEDGER, None, Some(&previous.bytes))?;
            if current.bytes != previous.bytes {
                return Err(invalid("spill quota header changed before publication"));
            }
            validate_named_file(&self.directory, LEDGER, file)?;
        } else {
            match self.directory.symlink_metadata(LEDGER) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
                Ok(_) => return Err(invalid("spill quota bootstrap entry was substituted")),
            }
        }
        self.validate_authority()?;
        self.directory.rename(INSTALLING, &self.directory, LEDGER)?;
        stage.armed = false;
        if let Some(published) = published {
            *published = Some(next.generation);
        }
        self.io.check(SpillIoOperation::QuotaSync)?;
        sync_directory(&self.directory)
    }
}

/// Fresh descriptor checks also reject permission changes after admission.
pub(super) fn validate_private_directory(directory: &Dir) -> io::Result<()> {
    let metadata = rustix::fs::fstat(directory)?;
    if rustix::fs::FileType::from_raw_mode(metadata.st_mode) != rustix::fs::FileType::Directory
        || metadata.st_uid != rustix::process::geteuid().as_raw()
        || metadata.st_mode & 0o077 != 0
    {
        return Err(invalid("spill quota directory is no longer private"));
    }
    #[cfg(target_os = "macos")]
    super::validate_no_acl_grants(directory)?;
    Ok(())
}

/// An internal, non-cloneable creation receipt. Dropping a live or ambiguous
/// object preserves durable debt; only proved deletion permits reclamation.
pub(super) struct RootReservation {
    ledger: Arc<RootLedger>,
    key: ReservationKey,
    birth: u64,
    pool: Option<Arc<RootReservation>>,
    cancellation: QueryCancellationToken,
    state: Mutex<ReservationState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolAdmission {
    Open,
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReservationCertainty {
    Known,
    Uncertain,
}

struct ReservationState {
    charged: u64,
    peak_charged: u64,
    observed: u64,
    peak_observed: u64,
    certainty: ReservationCertainty,
    children: u64,
    child_charge: u64,
    admission: PoolAdmission,
    attempted: bool,
    object: Option<File>,
    parent: Option<Arc<Dir>>,
    pending_release: Option<u64>,
    released: bool,
    #[cfg(target_os = "macos")]
    directory_unlinked: bool,
}

impl std::fmt::Debug for RootReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootReservation")
            .field("birth", &self.birth)
            .finish_non_exhaustive()
    }
}

impl RootReservation {
    pub(super) fn physical_stats(&self) -> Option<super::SpillPhysicalStats> {
        let state = self.state.try_lock()?;
        Some(super::SpillPhysicalStats {
            reserved_bytes: if state.released {
                0
            } else {
                state
                    .charged
                    .max(LEAF_RESERVE.saturating_add(state.child_charge))
            },
            peak_reserved_bytes: state.peak_charged,
            observed_file_bytes: if state.released { 0 } else { state.observed },
            peak_observed_file_bytes: state.peak_observed,
            cleanup_debt_bytes: 0,
            cleanup_failed: false,
            reservation_uncertain: !state.released
                && (state.certainty == ReservationCertainty::Uncertain
                    || state.pending_release.is_some()),
        })
    }

    pub(super) fn new(
        ledger: Arc<RootLedger>,
        key: ReservationKey,
        cancellation: QueryCancellationToken,
    ) -> io::Result<Arc<Self>> {
        let (birth, charged) = ledger.reserve(key, Some(&cancellation))?;
        Ok(Arc::new(Self {
            ledger,
            key,
            birth,
            pool: None,
            cancellation,
            state: Mutex::new(ReservationState {
                charged,
                peak_charged: charged,
                observed: 0,
                peak_observed: 0,
                certainty: ReservationCertainty::Known,
                children: 0,
                child_charge: 0,
                admission: PoolAdmission::Open,
                attempted: false,
                object: None,
                parent: None,
                pending_release: None,
                released: false,
                #[cfg(target_os = "macos")]
                directory_unlinked: false,
            }),
        }))
    }

    /// Subdivide capacity already durably owned by this query. A file receipt
    /// retains its query owner, so crash/ambiguous deletion cannot discard debt.
    pub(super) fn new_file(pool: &Arc<Self>, key: ReservationKey) -> io::Result<Arc<Self>> {
        if pool.key.0[0] != 0 || key.0[0] != 1 || pool.key.0[1..17] != key.0[1..17] {
            return Err(invalid("foreign query capacity receipt"));
        }
        RootLedger::check_cancelled(Some(&pool.cancellation))?;
        let mut state = pool.pool_guard()?;
        if state.admission == PoolAdmission::Failed
            || state.released
            || state.pending_release.is_some()
            || state.object.is_none()
        {
            return Err(invalid("query capacity is not available"));
        }
        let charged = FILE_RESERVE + ALLOCATION_UNIT;
        let child_charge = state
            .child_charge
            .checked_add(charged)
            .ok_or_else(|| invalid("query capacity overflow"))?;
        let required = LEAF_RESERVE
            .checked_add(child_charge)
            .ok_or_else(|| invalid("query capacity overflow"))?;
        if required > state.charged {
            let (capacity, admission) = pool
                .ledger
                .ensure_capacity(
                    pool.key,
                    pool.birth,
                    required,
                    false,
                    Some(&pool.cancellation),
                )
                .inspect_err(|_| state.certainty = ReservationCertainty::Uncertain)?;
            state.certainty = ReservationCertainty::Known;
            state.charged = capacity;
            state.peak_charged = state.peak_charged.max(capacity);
            admission?;
        }
        state.children = state
            .children
            .checked_add(1)
            .ok_or_else(|| invalid("query file count overflow"))?;
        state.child_charge = child_charge;
        Ok(Arc::new(Self {
            ledger: Arc::clone(&pool.ledger),
            key,
            birth: pool.birth,
            pool: Some(Arc::clone(pool)),
            cancellation: pool.cancellation.clone(),
            state: Mutex::new(ReservationState {
                charged,
                peak_charged: charged,
                observed: 0,
                peak_observed: 0,
                certainty: ReservationCertainty::Known,
                children: 0,
                child_charge: 0,
                admission: PoolAdmission::Open,
                attempted: false,
                object: None,
                parent: None,
                pending_release: None,
                released: false,
                #[cfg(target_os = "macos")]
                directory_unlinked: false,
            }),
        }))
    }

    fn pool_guard(&self) -> io::Result<MutexGuard<'_, ReservationState>> {
        if self.ledger.process != std::process::id() {
            return Err(invalid("query capacity belongs to another process"));
        }
        self.state
            .try_lock_for(std::time::Duration::from_millis(512))
            .ok_or_else(|| io::Error::new(io::ErrorKind::WouldBlock, "query capacity is busy"))
    }

    // Only the identity-bound query construction/completion owner calls this,
    // after successful removal of its exact directory. Path absence alone is
    // never enough to create this proof.
    #[cfg(target_os = "macos")]
    pub(super) fn confirm_directory_unlink(&self) -> io::Result<()> {
        if self.key.0[0] != 0 {
            return Err(invalid("file receipt cannot prove a directory unlink"));
        }
        let mut state = self
            .state
            .try_lock_for(std::time::Duration::from_millis(512))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::WouldBlock, "spill quota receipt is busy")
            })?;
        state.directory_unlinked = true;
        Ok(())
    }

    pub(super) fn attempted(&self) {
        self.state.lock().attempted = true;
    }

    pub(super) fn bind(&self, object: File, parent: Arc<Dir>) -> io::Result<()> {
        let mut state = self.state.lock();
        if state.object.is_some() || state.released {
            return Err(invalid("spill creation receipt was already bound"));
        }
        state.attempted = true;
        state.object = Some(object);
        state.parent = Some(parent);
        Ok(())
    }

    pub(super) fn ensure_capacity(
        &self,
        logical: u64,
        allocated: u64,
        publication: bool,
    ) -> io::Result<()> {
        RootLedger::check_cancelled(Some(&self.cancellation))?;
        let mut state = self
            .state
            .try_lock_for(std::time::Duration::from_millis(512))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::WouldBlock, "spill quota receipt is busy")
            })?;
        if state.released || state.pending_release.is_some() || state.object.is_none() {
            return Err(invalid("spill quota receipt is not writable"));
        }
        let allocated = if publication {
            publication_allocation(state.object.as_ref().expect("checked receipt"), logical)?
        } else {
            allocated
        };
        // This receipt owns already-durable, non-revocable capacity. Bounded
        // frame progress inside it needs no ledger transaction. Growth and
        // publication always revalidate the authenticated shared ledger.
        if !publication && file_charge(logical, allocated)? <= state.charged {
            return Ok(());
        }
        if let Some(parent) = &state.parent {
            validate_private_directory(parent)?;
        }
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| invalid("leaf is not a file receipt"))?;
        let mut capacity = pool.pool_guard()?;
        if capacity.admission == PoolAdmission::Failed || capacity.released {
            return Err(invalid("query capacity is not writable"));
        }
        let charged = state.charged.max(file_charge(logical, allocated)?);
        let child_charge = capacity
            .child_charge
            .checked_add(charged - state.charged)
            .ok_or_else(|| invalid("query physical debt overflow"))?;
        let required = LEAF_RESERVE
            .checked_add(child_charge)
            .ok_or_else(|| invalid("query physical debt overflow"))?;
        // Observed allocation already exists, including on failed persistence.
        // Retain it locally and poison admission on error; cleanup can still
        // close it, while no other receipt can spend this apparent free space.
        if publication {
            // Observation only: admission continues to use the authenticated
            // reservation. Keep sampled allocation charged until proven deletion.
            capacity.observed = capacity
                .observed
                .saturating_sub(state.observed)
                .saturating_add(allocated);
            capacity.peak_observed = capacity.peak_observed.max(capacity.observed);
            state.observed = allocated;
            capacity.peak_charged = capacity.peak_charged.max(required);
            capacity.child_charge = child_charge;
            state.charged = charged;
        }
        if publication || required > capacity.charged {
            match pool.ledger.ensure_capacity(
                pool.key,
                pool.birth,
                required,
                publication,
                Some(&self.cancellation),
            ) {
                Ok((reserved, admission)) => {
                    capacity.certainty = ReservationCertainty::Known;
                    capacity.charged = reserved;
                    capacity.peak_charged = capacity.peak_charged.max(reserved);
                    if let Err(error) = admission {
                        capacity.admission = PoolAdmission::Failed;
                        return Err(error);
                    }
                }
                Err(error) => {
                    capacity.certainty = ReservationCertainty::Uncertain;
                    capacity.admission = PoolAdmission::Failed;
                    return Err(error);
                }
            }
        }
        capacity.child_charge = child_charge;
        state.charged = charged;
        drop(capacity);
        if publication
            && publication_allocation(state.object.as_ref().expect("checked receipt"), logical)?
                != allocated
        {
            return Err(invalid("spill allocation changed during quota publication"));
        }
        if let Some(parent) = &state.parent {
            validate_private_directory(parent)?;
        }
        RootLedger::check_cancelled(Some(&self.cancellation))?;
        Ok(())
    }

    /// Checks the retained inode, syncs its deletion, and closes the final owned
    /// allocation handle before returning capacity to the durable ledger.
    pub(super) fn release_deleted(&self) -> io::Result<()> {
        let mut state = self
            .state
            .try_lock_for(std::time::Duration::from_millis(512))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "spill quota receipt cleanup is already active",
                )
            })?;
        if state.released {
            return Ok(());
        }
        if state.children != 0 {
            return Err(invalid("query capacity still owns file receipts"));
        }
        if let Some(object) = &state.object {
            let metadata = object.metadata()?;
            let unlinked = metadata.nlink() == 0;
            #[cfg(target_os = "macos")]
            let unlinked = unlinked
                || (metadata.is_dir()
                    && state.directory_unlinked
                    && macos_directory_absent(object, metadata.ino())?);
            if !unlinked {
                return Err(invalid("spill quota object still has a directory link"));
            }
            let parent = state.parent.as_ref().expect("bound receipt parent");
            self.ledger.io.check(SpillIoOperation::QuotaDeleteSync)?;
            sync_directory(parent)?;
            drop(state.object.take());
            state.attempted = false;
        } else if state.attempted {
            return Err(invalid("spill creation outcome is not proved"));
        }
        if let Some(pool) = &self.pool {
            let mut capacity = pool.pool_guard()?;
            self.ledger.io.check(SpillIoOperation::QuotaRelease)?;
            self.ledger.validate_reservation(pool.key, pool.birth)?;
            capacity.child_charge = capacity
                .child_charge
                .checked_sub(state.charged)
                .ok_or_else(|| invalid("query capacity credit underflow"))?;
            capacity.observed = capacity.observed.saturating_sub(state.observed);
            capacity.children = capacity
                .children
                .checked_sub(1)
                .ok_or_else(|| invalid("query file credit underflow"))?;
        } else {
            self.ledger
                .release_after_delete(self.key, self.birth, &mut state.pending_release)?;
        }
        state.released = true;
        state.parent = None;
        Ok(())
    }
}

impl Drop for RootReservation {
    fn drop(&mut self) {
        // Live files/leaves deliberately survive as charged crash debt. This
        // backstop only handles a never-created or already-unlinked object.
        let state = self.state.get_mut();
        let reclaimable = !state.attempted
            || state.object.as_ref().is_some_and(|file| {
                file.metadata().is_ok_and(|m| {
                    let unlinked = m.nlink() == 0;
                    #[cfg(target_os = "macos")]
                    let unlinked = unlinked || (m.is_dir() && state.directory_unlinked);
                    unlinked
                })
            });
        if !state.released && reclaimable {
            super::run_cleanup_backstop(|| self.release_deleted());
        }
    }
}

fn publication_allocation(file: &File, logical: u64) -> io::Result<u64> {
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.len() != logical
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.mode() & 0o077 != 0
    {
        return Err(invalid(
            "spill physical file changed before quota publication",
        ));
    }
    #[cfg(target_os = "macos")]
    super::validate_no_acl_grants(file)?;
    metadata
        .blocks()
        .checked_mul(512)
        .ok_or_else(|| invalid("spill allocation overflow"))
}

// APFS retains the directory's synthetic link count after rmdir. Confirm the
// identity-bound owner's successful unlink against the filesystem catalog,
// rather than trusting either that count or a missing ambient pathname.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)] // Descriptor-derived fsid/inode; bounded libSystem output.
pub(super) fn macos_directory_absent(file: &File, inode: u64) -> io::Result<bool> {
    use std::ffi::{c_char, c_void};
    unsafe extern "C" {
        fn fsgetpath(buffer: *mut c_char, size: usize, fsid: *mut c_void, inode: u64) -> isize;
    }
    let mut filesystem = rustix::fs::fstatfs(file)?;
    let mut buffer = [0_u8; 1024];
    // SAFETY: Rustix returns Darwin's statfs ABI, including its correctly sized
    // and aligned fsid_t. Both pointers address live writable storage; the
    // buffer length is exact. No returned path is opened or used as authority.
    let result = unsafe {
        fsgetpath(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            (&raw mut filesystem.f_fsid).cast(),
            inode,
        )
    };
    if result >= 0 {
        return Ok(false);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::NotFound {
        Ok(true)
    } else {
        Err(error)
    }
}
