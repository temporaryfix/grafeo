//! One-shot WAL log failure injection for mutation, commit, and abort records.
//!
//! Production builds expose the same API as no-ops.

use std::fmt;

/// Error returned when an injected WAL log failure fires.
///
/// The abort hook returns this type. Arming it causes one failure when
/// `testing-crash-injection` is enabled; otherwise the hook remains a no-op.
///
/// ```
/// use grafeo_common::testing::wal_failure::{
///     InjectedWalLogFailure, disable_abort_log_failure,
///     enable_abort_log_failure_once, maybe_fail_abort_log,
/// };
///
/// disable_abort_log_failure();
/// let initial: Result<(), InjectedWalLogFailure> = maybe_fail_abort_log();
/// assert!(initial.is_ok());
/// enable_abort_log_failure_once();
/// let armed: Result<(), InjectedWalLogFailure> = maybe_fail_abort_log();
/// assert_eq!(armed.is_err(), cfg!(feature = "testing-crash-injection"));
/// assert!(maybe_fail_abort_log().is_ok());
/// disable_abort_log_failure();
/// ```
///
/// The former abort-specific error name is unavailable:
///
/// ```compile_fail,E0432
/// use grafeo_common::testing::wal_failure::InjectedAbortLogFailure;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InjectedWalLogFailure {
    operation: &'static str,
}

#[cfg(feature = "testing-crash-injection")]
impl InjectedWalLogFailure {
    const fn new(operation: &'static str) -> Self {
        Self { operation }
    }
}

impl fmt::Display for InjectedWalLogFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injected WAL {} log failure", self.operation)
    }
}

impl std::error::Error for InjectedWalLogFailure {}

#[cfg(feature = "testing-crash-injection")]
mod inner {
    use super::InjectedWalLogFailure;
    use std::cell::Cell;

    thread_local! {
        static FAIL_ABORT: Cell<bool> = const { Cell::new(false) };
        static FAIL_COMMIT: Cell<bool> = const { Cell::new(false) };
        static FAIL_COMMIT_ACK: Cell<bool> = const { Cell::new(false) };
        static FAIL_CATALOG_BATCH: Cell<bool> = const { Cell::new(false) };
        static FAIL_CATALOG_BATCH_ACK: Cell<bool> = const { Cell::new(false) };
        static FAIL_MUTATION: Cell<bool> = const { Cell::new(false) };
        static FAIL_PROJECTION_RECEIPT: Cell<bool> = const { Cell::new(false) };
        static FAIL_BACKUP_PUBLICATION: Cell<Option<&'static str>> = const { Cell::new(None) };
    }

    /// Runs an operation with one thread-local backup publication failure armed.
    /// Restores the previous hook on return or unwind, including nested scopes.
    pub fn with_backup_publication_failure<T>(point: &'static str, f: impl FnOnce() -> T) -> T {
        struct Reset(Option<&'static str>);
        impl Drop for Reset {
            fn drop(&mut self) {
                FAIL_BACKUP_PUBLICATION.with(|armed| armed.set(self.0));
            }
        }
        let _reset = Reset(FAIL_BACKUP_PUBLICATION.with(|armed| armed.replace(Some(point))));
        f()
    }

    /// Consumes the matching scoped backup publication failure, if armed.
    ///
    /// # Errors
    /// Write points return storage-full; rename and sync points return I/O errors.
    pub fn check_backup_publication_failure(
        point: &'static str,
    ) -> crate::utils::error::Result<()> {
        use crate::utils::error::{Error, StorageError};
        let fire = FAIL_BACKUP_PUBLICATION.with(|armed| {
            if armed.get() == Some(point) {
                armed.set(None);
                true
            } else {
                false
            }
        });
        if !fire {
            return Ok(());
        }
        if point.ends_with("_write") || point.ends_with("_write_failure") {
            Err(Error::Storage(StorageError::Full))
        } else {
            Err(
                std::io::Error::other(format!("injected backup publication failure: {point}"))
                    .into(),
            )
        }
    }

    /// Arm a one-shot failure for the next abort-record WAL log.
    pub fn enable_abort_log_failure_once() {
        FAIL_ABORT.with(|c| c.set(true));
    }

    /// Disable abort-log injection.
    pub fn disable_abort_log_failure() {
        FAIL_ABORT.with(|c| c.set(false));
    }

    /// Arm a one-shot failure for the next commit-record WAL log.
    pub fn enable_commit_log_failure_once() {
        FAIL_COMMIT.with(|c| c.set(true));
    }

    /// Disable commit-log injection.
    pub fn disable_commit_log_failure() {
        FAIL_COMMIT.with(|c| c.set(false));
    }

    /// Arm a one-shot outcome-ambiguous failure after the next commit frame
    /// has been written and synced but before success is acknowledged.
    pub fn enable_commit_ack_failure_once() {
        FAIL_COMMIT_ACK.with(|c| c.set(true));
    }

    /// Disable post-write commit acknowledgement failure injection.
    pub fn disable_commit_ack_failure() {
        FAIL_COMMIT_ACK.with(|c| c.set(false));
    }

    /// Arm a one-shot failure immediately before the next catalog-batch frame.
    pub fn enable_catalog_batch_log_failure_once() {
        FAIL_CATALOG_BATCH.with(|c| c.set(true));
    }

    /// Disable pre-append catalog-batch failure injection.
    pub fn disable_catalog_batch_log_failure() {
        FAIL_CATALOG_BATCH.with(|c| c.set(false));
    }

    /// Arm a one-shot failure after the next catalog-batch frame is durable
    /// but before its success is acknowledged to the publisher.
    pub fn enable_catalog_batch_ack_failure_once() {
        FAIL_CATALOG_BATCH_ACK.with(|c| c.set(true));
    }

    /// Disable post-sync catalog-batch acknowledgement failure injection.
    pub fn disable_catalog_batch_ack_failure() {
        FAIL_CATALOG_BATCH_ACK.with(|c| c.set(false));
    }

    /// Arm a one-shot failure for the next graph-data mutation WAL log.
    pub fn enable_mutation_log_failure_once() {
        FAIL_MUTATION.with(|c| c.set(true));
    }

    /// Disable mutation-log injection.
    pub fn disable_mutation_log_failure() {
        FAIL_MUTATION.with(|c| c.set(false));
    }

    /// Arm a one-shot failure immediately before the next RDF→LPG projection
    /// receipt WAL append.
    pub fn enable_projection_receipt_log_failure_once() {
        FAIL_PROJECTION_RECEIPT.with(|c| c.set(true));
    }

    /// Disable projection-receipt injection.
    pub fn disable_projection_receipt_log_failure() {
        FAIL_PROJECTION_RECEIPT.with(|c| c.set(false));
    }

    /// Returns [`InjectedWalLogFailure`] if projection-receipt injection is armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms before the receipt is appended.
    pub fn maybe_fail_projection_receipt_log() -> Result<(), InjectedWalLogFailure> {
        FAIL_PROJECTION_RECEIPT.with(|c| {
            if c.get() {
                c.set(false);
                Err(InjectedWalLogFailure::new("projection receipt"))
            } else {
                Ok(())
            }
        })
    }

    /// Returns [`InjectedWalLogFailure`] if mutation-log injection is armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms.
    pub fn maybe_fail_mutation_log() -> Result<(), InjectedWalLogFailure> {
        FAIL_MUTATION.with(|c| {
            if c.get() {
                c.set(false);
                Err(InjectedWalLogFailure::new("mutation"))
            } else {
                Ok(())
            }
        })
    }

    /// Returns [`InjectedWalLogFailure`] if commit-log injection is armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms.
    pub fn maybe_fail_commit_log() -> Result<(), InjectedWalLogFailure> {
        FAIL_COMMIT.with(|c| {
            if c.get() {
                c.set(false);
                Err(InjectedWalLogFailure::new("commit"))
            } else {
                Ok(())
            }
        })
    }

    /// Returns an error after a durable commit append when acknowledgement
    /// failure injection is armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms.
    pub fn maybe_fail_commit_ack() -> Result<(), InjectedWalLogFailure> {
        take_commit_ack_failure().map_or(Ok(()), Err)
    }

    /// Takes an armed commit-acknowledgement failure for deferred delivery.
    ///
    /// Async WAL writers consume this token before their first suspension so
    /// executor thread migration cannot strand a thread-local one-shot hook.
    pub fn take_commit_ack_failure() -> Option<InjectedWalLogFailure> {
        FAIL_COMMIT_ACK.with(|c| {
            c.replace(false)
                .then(|| InjectedWalLogFailure::new("commit acknowledgement"))
        })
    }

    /// Returns an error before a catalog-batch frame when injection is armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms before the frame is appended.
    pub fn maybe_fail_catalog_batch_log() -> Result<(), InjectedWalLogFailure> {
        FAIL_CATALOG_BATCH.with(|c| {
            if c.get() {
                c.set(false);
                Err(InjectedWalLogFailure::new("catalog batch"))
            } else {
                Ok(())
            }
        })
    }

    /// Returns an error after a durable catalog-batch append when armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms after the frame is durable.
    pub fn maybe_fail_catalog_batch_ack() -> Result<(), InjectedWalLogFailure> {
        take_catalog_batch_ack_failure().map_or(Ok(()), Err)
    }

    /// Takes an armed catalog-batch acknowledgement failure for deferred
    /// delivery after a successful durable append.
    ///
    /// Consuming the thread-local hook before an async suspension makes the
    /// resulting value safe to deliver even if the future resumes elsewhere.
    pub fn take_catalog_batch_ack_failure() -> Option<InjectedWalLogFailure> {
        FAIL_CATALOG_BATCH_ACK.with(|c| {
            c.replace(false)
                .then(|| InjectedWalLogFailure::new("catalog batch acknowledgement"))
        })
    }

    /// Returns [`InjectedWalLogFailure`] if abort-log injection is armed.
    ///
    /// # Errors
    ///
    /// Returns the injected failure and disarms.
    pub fn maybe_fail_abort_log() -> Result<(), InjectedWalLogFailure> {
        FAIL_ABORT.with(|c| {
            if c.get() {
                c.set(false);
                Err(InjectedWalLogFailure::new("abort"))
            } else {
                Ok(())
            }
        })
    }
}

#[cfg(not(feature = "testing-crash-injection"))]
mod inner {
    /// Runs the operation unchanged when failure injection is disabled.
    pub fn with_backup_publication_failure<T>(_point: &'static str, f: impl FnOnce() -> T) -> T {
        f()
    }

    /// No-op when failure injection is disabled.
    ///
    /// # Errors
    /// Never returns an error in this build.
    pub fn check_backup_publication_failure(
        _point: &'static str,
    ) -> crate::utils::error::Result<()> {
        Ok(())
    }

    use super::InjectedWalLogFailure;

    /// No-op when crash injection is disabled.
    pub fn enable_abort_log_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_abort_log_failure() {}
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_abort_log() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
    /// No-op when crash injection is disabled.
    pub fn enable_commit_log_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_commit_log_failure() {}
    /// No-op when crash injection is disabled.
    pub fn enable_commit_ack_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_commit_ack_failure() {}
    /// No-op when crash injection is disabled.
    pub fn enable_catalog_batch_log_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_catalog_batch_log_failure() {}
    /// No-op when crash injection is disabled.
    pub fn enable_catalog_batch_ack_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_catalog_batch_ack_failure() {}
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_commit_log() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_commit_ack() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
    /// No-op when crash injection is disabled.
    pub fn take_commit_ack_failure() -> Option<InjectedWalLogFailure> {
        None
    }
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_catalog_batch_log() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_catalog_batch_ack() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
    /// No-op when crash injection is disabled.
    pub fn take_catalog_batch_ack_failure() -> Option<InjectedWalLogFailure> {
        None
    }
    /// No-op when crash injection is disabled.
    pub fn enable_mutation_log_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_mutation_log_failure() {}
    /// No-op when crash injection is disabled.
    pub fn enable_projection_receipt_log_failure_once() {}
    /// No-op when crash injection is disabled.
    pub fn disable_projection_receipt_log_failure() {}
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_mutation_log() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
    /// No-op when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error in this build.
    pub fn maybe_fail_projection_receipt_log() -> Result<(), InjectedWalLogFailure> {
        Ok(())
    }
}

pub use inner::*;

#[cfg(all(test, feature = "testing-crash-injection"))]
mod tests {
    use super::*;

    #[test]
    fn backup_publication_failure_is_scoped_one_shot_and_unwind_safe() {
        use crate::utils::error::{Error, StorageError};
        with_backup_publication_failure("backup:cursor_write", || {
            assert!(check_backup_publication_failure("backup:manifest_write").is_ok());
            assert!(matches!(
                check_backup_publication_failure("backup:cursor_write"),
                Err(Error::Storage(StorageError::Full))
            ));
            assert!(check_backup_publication_failure("backup:cursor_write").is_ok());
        });
        let panicked = std::panic::catch_unwind(|| {
            with_backup_publication_failure("backup:cursor_parent_sync", || panic!("unwind hook"));
        });
        assert!(panicked.is_err());
        assert!(check_backup_publication_failure("backup:cursor_parent_sync").is_ok());
        with_backup_publication_failure("backup:cursor_parent_sync", || {
            with_backup_publication_failure("backup:segment_write_failure", || {
                assert!(matches!(
                    check_backup_publication_failure("backup:segment_write_failure"),
                    Err(Error::Storage(StorageError::Full))
                ));
            });
            assert!(matches!(
                check_backup_publication_failure("backup:cursor_parent_sync"),
                Err(Error::Io(_))
            ));
        });
    }

    #[test]
    fn deferred_ack_failures_can_cross_threads_without_stranding_hooks() {
        enable_commit_ack_failure_once();
        let commit = take_commit_ack_failure().expect("commit acknowledgement hook was armed");
        assert!(maybe_fail_commit_ack().is_ok(), "commit hook was consumed");

        enable_catalog_batch_ack_failure_once();
        let catalog =
            take_catalog_batch_ack_failure().expect("catalog-batch acknowledgement hook was armed");
        assert!(
            maybe_fail_catalog_batch_ack().is_ok(),
            "catalog hook was consumed"
        );

        let messages = std::thread::spawn(move || (commit.to_string(), catalog.to_string()))
            .join()
            .expect("deferred failures are Send across threads");
        assert!(messages.0.contains("commit acknowledgement"));
        assert!(messages.1.contains("catalog batch acknowledgement"));
    }
}
