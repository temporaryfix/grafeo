//! Darwin ACL checks on retained descriptors, before spill authority is used.
//!
//! Mode bits do not constrain extended ACL grants on macOS. Deny-only ACLs
//! (including the usual home-directory delete denial) cannot widen access.
//! Reject every grant, including inherited and owner-only grants, rather than
//! attempting to duplicate Darwin's identity/group/inheritance evaluation.

#![allow(unsafe_code)] // Small, descriptor-only libSystem ACL boundary.

use std::ffi::{c_int, c_uint, c_void};
use std::io;
use std::os::fd::{AsFd, AsRawFd};
use std::ptr::NonNull;

// Darwin sys/acl.h: ACL_TYPE_EXTENDED, ACL_EXTENDED_DENY, ACL_MAX_ENTRIES.
const EXTENDED: c_uint = 0x100;
const DENY: c_uint = 2;
const MAX_ENTRIES: c_int = 128;

unsafe extern "C" {
    fn acl_get_fd_np(fd: c_int, kind: c_uint) -> *mut c_void;
    fn acl_valid(acl: *mut c_void) -> c_int;
    fn acl_get_entry(acl: *mut c_void, index: c_int, entry: *mut *mut c_void) -> c_int;
    fn acl_get_tag_type(entry: *mut c_void, tag: *mut c_uint) -> c_int;
    fn acl_free(acl: *mut c_void) -> c_int;
}

struct OwnedAcl(NonNull<c_void>);

impl Drop for OwnedAcl {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner of the allocation from acl_get_fd_np;
        // entry pointers are borrowed only within validate_no_acl_grants.
        unsafe { acl_free(self.0.as_ptr()) };
    }
}

/// Rejects macOS extended ACLs that can grant access beyond Unix mode bits.
///
/// This is shared with the engine's database-local authority implementation.
/// Callers must also validate ownership, mode, type and retained identity.
///
/// # Errors
/// Fails closed for grants, unknown entries, malformed ACLs and unsupported
/// filesystems. The only accepted query error is Darwin's absent-ACL ENOENT
/// on a live descriptor; no ambient pathname is queried.
pub fn validate_no_acl_grants(file: &impl AsFd) -> io::Result<()> {
    let fd = file.as_fd();
    // SAFETY: fd remains borrowed for this call; EXTENDED is the Darwin ABI
    // constant. A successful call transfers a separately allocated ACL copy.
    let raw = unsafe { acl_get_fd_np(fd.as_raw_fd(), EXTENDED) };
    let Some(acl) = NonNull::new(raw).map(OwnedAcl) else {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(2) {
            // Darwin filesec_get_property reports ENOENT when no ACL exists.
            // Ensure the descriptor itself still denotes a live filesystem node.
            rustix::fs::fstat(fd)?;
            return Ok(());
        }
        return Err(error);
    };
    // SAFETY: acl owns a valid allocation from libSystem and stays alive below.
    if unsafe { acl_valid(acl.0.as_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // Darwin supports absolute entry indices. Unlike POSIX/Linux, it returns
    // zero for an entry and -1/EINVAL at the end of this validated ACL copy.
    for index in 0..=MAX_ENTRIES {
        let mut entry = std::ptr::null_mut();
        // SAFETY: valid owned ACL and writable output pointer; no concurrent
        // mutation of this independent ACL copy is possible.
        if unsafe { acl_get_entry(acl.0.as_ptr(), index, &raw mut entry) } != 0 {
            let error = io::Error::last_os_error();
            return if error.raw_os_error() == Some(22) {
                Ok(())
            } else {
                Err(error)
            };
        }
        let mut tag = 0;
        if entry.is_null() || index == MAX_ENTRIES {
            return Err(io::Error::other("invalid or oversized spill ACL"));
        }
        // SAFETY: entry was returned by acl_get_entry and its owning ACL is
        // alive; tag is a writable ABI-sized integer, not a Rust enum.
        if unsafe { acl_get_tag_type(entry, &raw mut tag) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if tag != DENY {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "spill authority rejects extended ACL grants",
            ));
        }
    }
    Err(io::Error::other("spill ACL exceeds the Darwin entry limit"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
    use std::path::Path;
    use std::process::Command;

    fn acl(path: &Path, entry: &str) {
        assert!(
            Command::new("/bin/chmod")
                .arg("+a")
                .arg(entry)
                .arg(path)
                .status()
                .unwrap()
                .success()
        );
    }

    #[test]
    fn held_locks_exclude_another_process_until_the_last_owner_drops() {
        const PATH_ENV: &str = "GRAFEO_MAC_ACL_LOCK_PROBE_PATH";
        const HELD_ENV: &str = "GRAFEO_MAC_ACL_LOCK_PROBE_HELD";
        if let Some(path) = std::env::var_os(PATH_ENV) {
            let file = File::open(path).unwrap();
            if std::env::var_os(HELD_ENV).is_some() {
                assert!(matches!(file.try_lock(), Err(fs::TryLockError::WouldBlock)));
            } else {
                file.try_lock().unwrap();
                file.unlock().unwrap();
            }
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("file");
        File::create(&path).unwrap();
        for path in [temp.path(), path.as_path()] {
            let owner = File::open(path).unwrap();
            owner.try_lock().unwrap();
            let retained = owner.try_clone().unwrap();
            drop(owner);
            let probe = |held: bool| {
                let mut child = Command::new(std::env::current_exe().unwrap());
                child.args(["--exact", "execution::spill::macos_acl::tests::held_locks_exclude_another_process_until_the_last_owner_drops", "--nocapture"])
                    .env(PATH_ENV, path).env_remove(HELD_ENV);
                if held {
                    child.env(HELD_ENV, "1");
                }
                let result = child.output().unwrap();
                assert!(
                    result.status.success(),
                    "{}{}",
                    String::from_utf8_lossy(&result.stdout),
                    String::from_utf8_lossy(&result.stderr)
                );
                assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
            };
            probe(true);
            drop(retained);
            probe(false);
        }
    }

    #[test]
    fn absent_and_deny_only_acls_preserve_mode_authority() {
        let temp = tempfile::tempdir().unwrap();
        let file = File::create(temp.path().join("file")).unwrap();
        validate_no_acl_grants(&file).unwrap();
        let path = temp.path().join("file");
        acl(&path, "everyone deny delete");
        let validation = validate_no_acl_grants(&file);
        // The deliberate denial also prevents TempDir's best-effort unlink.
        // Clear it before asserting validation so failures do not leak fixtures.
        assert!(
            Command::new("/bin/chmod")
                .arg("-N")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        drop(file);
        let directory = temp.path().to_owned();
        temp.close().unwrap();
        assert!(!directory.exists());
        validation.unwrap();
    }

    #[test]
    fn private_mode_does_not_hide_direct_or_inherited_grants() {
        let temp = tempfile::tempdir().unwrap();
        acl(
            temp.path(),
            "everyone allow read,file_inherit,directory_inherit",
        );
        let path = temp.path().join("file");
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(
            validate_no_acl_grants(&file).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            validate_no_acl_grants(&File::open(temp.path()).unwrap())
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
    }

    #[test]
    fn acl_checks_follow_retained_inode_and_observe_later_mutation() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("file");
        let file = File::create(&path).unwrap();
        let retained = temp.path().join("retained");
        fs::rename(&path, &retained).unwrap();
        let replacement = File::create(&path).unwrap();
        acl(&path, "everyone allow read");
        validate_no_acl_grants(&file).unwrap();
        assert!(validate_no_acl_grants(&replacement).is_err());
        acl(&retained, "everyone allow read");
        assert!(validate_no_acl_grants(&file).is_err());
    }
}
