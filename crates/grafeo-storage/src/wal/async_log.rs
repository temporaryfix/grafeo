//! Bounded async dispatch over the single owned synchronous WAL core.

use super::frame_buffer::{copy_frame, encode_frame};
use super::log::WalFrameIntent;
use super::record::WalEntry;
use super::{
    CheckpointMetadata, DurabilityMode, WalCapture, WalConfig, WalManager, WalRecord,
    WalRetentionLease,
};
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, ErrorCode, Result};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot};

const CALLER_LIMIT: usize = 16;

/// A shared terminal close failure retaining its original structured cause.
#[derive(Debug, Clone)]
pub struct WalCloseError(Arc<Error>);

impl WalCloseError {
    /// Owns a close failure without recreating or stringifying its cause.
    #[must_use]
    pub fn new(error: Error) -> Self {
        Self(Arc::new(error))
    }

    /// Returns the underlying structured error code.
    #[must_use]
    pub fn error_code(&self) -> ErrorCode {
        self.0.error_code()
    }
}

impl std::fmt::Display for WalCloseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for WalCloseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

type CloseResult = std::result::Result<(), WalCloseError>;

fn terminal_error() -> Error {
    std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        "async WAL admission is closed",
    )
    .into()
}

fn capacity_error() -> Error {
    std::io::Error::new(
        std::io::ErrorKind::WouldBlock,
        "async WAL caller capacity is exhausted",
    )
    .into()
}

fn interrupted_error() -> Error {
    std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "owned WAL completion was interrupted; outcome is not retryable",
    )
    .into()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Open,
    Closing,
    Closed,
}

struct Admission {
    phase: Phase,
    active: bool,
    close_result: Option<CloseResult>,
}

struct AsyncController {
    raw: WalManager,
    admission: Mutex<Admission>,
    callers: Arc<Semaphore>,
    dispatch: Arc<Semaphore>,
    completed: Notify,
}

impl AsyncController {
    fn new(raw: WalManager) -> Self {
        Self {
            raw,
            admission: Mutex::new(Admission {
                phase: Phase::Open,
                active: false,
                close_result: None,
            }),
            callers: Arc::new(Semaphore::new(CALLER_LIMIT)),
            dispatch: Arc::new(Semaphore::new(1)),
            completed: Notify::new(),
        }
    }

    fn caller(&self) -> Result<OwnedSemaphorePermit> {
        if self.admission.lock().phase != Phase::Open {
            return Err(terminal_error());
        }
        Arc::clone(&self.callers)
            .try_acquire_owned()
            .map_err(|_| capacity_error())
    }

    fn publish_close(&self, result: CloseResult) {
        {
            let mut state = self.admission.lock();
            state.close_result = Some(result);
            state.phase = Phase::Closed;
            state.active = false;
        }
        self.completed.notify_waiters();
    }

    fn close_result(&self) -> Option<CloseResult> {
        self.admission.lock().close_result.clone()
    }
}

/// The sole active physical dispatch/close-continuation obligation.
struct DispatchTicket {
    controller: Arc<AsyncController>,
    permit: Option<OwnedSemaphorePermit>,
    active: bool,
    interrupted: bool,
}

impl Drop for DispatchTicket {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if self.interrupted {
            // The raw operation guard already publishes physical unwind poison.
            // This drains terminal resources without retrying failed buffers.
            let _ = self.controller.raw.fail_and_drain();
        }
        let closing = {
            let mut state = self.controller.admission.lock();
            if state.phase == Phase::Closing {
                true // Keep active until this same ticket finishes close.
            } else {
                state.active = false;
                false
            }
        };
        if closing {
            let result = self.controller.raw.close().map_err(WalCloseError::new);
            self.controller.publish_close(result);
        }
        // Destruction occurs outside admission; no response retains this permit.
        self.permit.take();
    }
}

struct OwnedResponse<T> {
    result: Result<T>,
    _receipt: Option<OwnedSemaphorePermit>,
}

/// Field retirement is explicit: action/frame, ticket/physical permit, sender.
struct OwnedJob<T> {
    action: Option<Box<dyn FnOnce() -> Result<T> + Send>>,
    ticket: Option<DispatchTicket>,
    receipt: Option<OwnedSemaphorePermit>,
    sender: Option<oneshot::Sender<OwnedResponse<T>>>,
}

impl<T> OwnedJob<T> {
    fn run(mut self) {
        let result = match self.action.take() {
            Some(action) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)) {
                Ok(result) => result,
                Err(_) => {
                    if let Some(ticket) = self.ticket.as_mut() {
                        ticket.interrupted = true;
                    }
                    Err(interrupted_error())
                }
            },
            None => Err(interrupted_error()),
        };
        // FnOnce and all captured frame bytes have retired before ticket Drop.
        drop(self.ticket.take());
        let response = OwnedResponse {
            result,
            _receipt: self.receipt.take(),
        };
        if let Some(sender) = self.sender.take() {
            // Failed send returns ownership, including a constructor supervisor.
            if let Err(response) = sender.send(response) {
                drop(response);
            }
        }
    }
}

impl<T> Drop for OwnedJob<T> {
    fn drop(&mut self) {
        drop(self.action.take());
        drop(self.ticket.take());
        drop(self.receipt.take());
        // Sender cancellation cannot wake an observer ahead of owned cleanup.
        drop(self.sender.take());
    }
}

enum Envelope<T> {
    Pending(OwnedJob<T>),
    Taken,
    Abandoned(Option<OwnedSemaphorePermit>),
}

struct SubmissionGuard<T> {
    envelope: Option<Arc<Mutex<Envelope<T>>>>,
}

impl<T> SubmissionGuard<T> {
    fn reclaim(&mut self) -> bool {
        let Some(envelope) = self.envelope.take() else {
            return false;
        };
        let reclaimed = {
            let mut state = envelope.lock();
            match std::mem::replace(&mut *state, Envelope::Taken) {
                Envelope::Pending(mut job) => {
                    *state = Envelope::Abandoned(job.receipt.take());
                    Some(job)
                }
                other => {
                    *state = other;
                    None
                }
            }
        };
        let did_reclaim = reclaimed.is_some();
        drop(reclaimed);
        did_reclaim
    }
}

impl<T> Drop for SubmissionGuard<T> {
    fn drop(&mut self) {
        self.reclaim();
    }
}

/// Submit with a result channel independent of the possibly-unreturned handle.
fn submit_owned<T: Send + 'static>(
    handle: &Handle,
    action: Box<dyn FnOnce() -> Result<T> + Send>,
    ticket: Option<DispatchTicket>,
    receipt: Option<OwnedSemaphorePermit>,
) -> Result<oneshot::Receiver<OwnedResponse<T>>> {
    submit_using(action, ticket, receipt, |work| {
        drop(handle.spawn_blocking(work));
    })
}

fn submit_using<T: Send + 'static>(
    action: Box<dyn FnOnce() -> Result<T> + Send>,
    ticket: Option<DispatchTicket>,
    receipt: Option<OwnedSemaphorePermit>,
    spawn: impl FnOnce(Box<dyn FnOnce() + Send>),
) -> Result<oneshot::Receiver<OwnedResponse<T>>> {
    let (sender, receiver) = oneshot::channel();
    let envelope = Arc::new(Mutex::new(Envelope::Pending(OwnedJob {
        action: Some(action),
        ticket,
        receipt,
        sender: Some(sender),
    })));
    let mut guard = SubmissionGuard {
        envelope: Some(Arc::clone(&envelope)),
    };
    let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        spawn(Box::new(move || {
            let state = {
                let mut state = envelope.lock();
                std::mem::replace(&mut *state, Envelope::Taken)
            };
            match state {
                Envelope::Pending(job) => job.run(),
                Envelope::Abandoned(receipt) => drop(receipt),
                Envelope::Taken => {}
            }
        }));
    }));
    match submitted {
        Ok(()) => {
            drop(guard.envelope.take());
        }
        Err(_) if guard.reclaim() => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "WAL submission failed before its command started",
            )
            .into());
        }
        Err(_) => {} // Taken work owns the only authoritative response.
    }
    // No submitter envelope reference is retained across receive await.
    Ok(receiver)
}

async fn receive<T>(receiver: oneshot::Receiver<OwnedResponse<T>>) -> Result<T> {
    let response = receiver.await.map_err(|_| interrupted_error())?;
    response.result
}

/// A bounded asynchronous facade over one owned synchronous WAL.
///
/// Sixteen caller/response receipts and one preparation/physical permit bound
/// live transport. Dropping an observing future is not physical cancellation.
/// Final owner Drop may block while the synchronous supervisor drains.
pub struct AsyncWalManager {
    controller: Arc<AsyncController>,
}

impl AsyncWalManager {
    /// Opens a WAL using default configuration.
    ///
    /// # Errors
    /// Returns runtime, ownership or physical initialization failures.
    pub async fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(dir, WalConfig::default()).await
    }

    /// Opens a WAL in one owned blocking constructor.
    ///
    /// # Errors
    /// Returns runtime, ownership or physical initialization failures.
    pub async fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        let requested = dir.as_ref();
        let path = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            std::env::current_dir()?.join(requested)
        };
        let handle =
            Handle::try_current().map_err(|error| Error::Io(std::io::Error::other(error)))?;
        let raw = receive(submit_owned(
            &handle,
            Box::new(move || WalManager::with_config(path, config)),
            None,
            None,
        )?)
        .await?;
        Ok(Self {
            controller: Arc::new(AsyncController::new(raw)),
        })
    }

    async fn permit(&self) -> Result<OwnedSemaphorePermit> {
        Arc::clone(&self.controller.dispatch)
            .acquire_owned()
            .await
            .map_err(|_| terminal_error())
    }

    async fn dispatch<T: Send + 'static>(
        &self,
        receipt: OwnedSemaphorePermit,
        permit: OwnedSemaphorePermit,
        action: impl FnOnce(&WalManager) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        // Declaration order ensures captured frame retires before the permit on
        // admission/runtime rejection as well as successful job completion.
        struct Preparation<A> {
            action: A,
            permit: OwnedSemaphorePermit,
            receipt: OwnedSemaphorePermit,
        }
        let prepared = Preparation {
            action,
            permit,
            receipt,
        };
        let handle =
            Handle::try_current().map_err(|error| Error::Io(std::io::Error::other(error)))?;
        {
            let mut state = self.controller.admission.lock();
            if state.phase != Phase::Open {
                return Err(terminal_error());
            }
            state.active = true;
        }
        let Preparation {
            action,
            permit,
            receipt,
        } = prepared;
        let controller = Arc::clone(&self.controller);
        let ticket = DispatchTicket {
            controller: Arc::clone(&controller),
            permit: Some(permit),
            active: true,
            interrupted: false,
        };
        receive(submit_owned(
            &handle,
            Box::new(move || action(&controller.raw)),
            Some(ticket),
            Some(receipt),
        )?)
        .await
    }

    pub(super) async fn log_entry<R: WalEntry>(&self, record: &R) -> Result<()> {
        let receipt = self.controller.caller()?;
        record
            .validate_recovery()
            .map_err(|reason| Error::InvalidValue(format!("invalid WAL record: {reason}")))?;
        let intent = WalFrameIntent::capture(record);
        let permit = self.permit().await?;
        let data = encode_frame(record)?;
        self.dispatch(receipt, permit, move |raw| raw.write_frame(&data, intent))
            .await
    }

    /// Logs one semantically validated record.
    ///
    /// # Errors
    /// Returns capacity, validation, terminal or physical failures.
    pub async fn log(&self, record: &WalRecord) -> Result<()> {
        self.log_entry(record).await
    }

    /// Appends a current-generation payload produced by [`super::encode_record`].
    /// The envelope is validated; this path does not classify record semantics.
    ///
    /// # Errors
    /// Returns capacity, frame validation, terminal or physical failures.
    pub async fn write_frame(&self, data: &[u8], force_sync: bool) -> Result<()> {
        let receipt = self.controller.caller()?;
        super::validate_wal_frame_payload_len(data.len())?;
        let permit = self.permit().await?;
        let data = copy_frame(data)?;
        self.dispatch(receipt, permit, move |raw| {
            raw.write_frame(&data, WalFrameIntent::raw(force_sync))
        })
        .await
    }

    pub(super) async fn checkpoint_entry<R: WalEntry>(
        &self,
        transaction: TransactionId,
        epoch: EpochId,
    ) -> Result<()> {
        let receipt = self.controller.caller()?;
        if epoch == EpochId::PENDING || !transaction.is_valid() {
            return Err(Error::InvalidValue(
                "invalid WAL checkpoint identity".into(),
            ));
        }
        let permit = self.permit().await?;
        let record = R::make_checkpoint(transaction);
        record.validate_recovery().map_err(|reason| {
            Error::InvalidValue(format!("invalid checkpoint WAL record: {reason}"))
        })?;
        let data = encode_frame(&record)?;
        self.dispatch(receipt, permit, move |raw| {
            raw.write_checkpoint_frame(&data, transaction, epoch)
        })
        .await
    }

    /// Publishes an owned checkpoint marker and metadata.
    ///
    /// # Errors
    /// Returns capacity, validation, terminal or physical failures.
    pub async fn checkpoint(&self, transaction: TransactionId, epoch: EpochId) -> Result<()> {
        self.checkpoint_entry::<WalRecord>(transaction, epoch).await
    }

    async fn query<T: Send + 'static>(
        &self,
        action: impl FnOnce(&WalManager) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let receipt = self.controller.caller()?;
        let permit = self.permit().await?;
        self.dispatch(receipt, permit, action).await
    }

    /// Flushes accepted bytes.
    /// # Errors
    /// Returns admission or physical failures.
    pub async fn flush(&self) -> Result<()> {
        self.query(WalManager::flush).await
    }
    /// Synchronizes accepted bytes.
    /// # Errors
    /// Returns admission or physical failures.
    pub async fn sync(&self) -> Result<()> {
        self.query(WalManager::sync).await
    }
    /// Rotates the owned active segment.
    /// # Errors
    /// Returns admission, capacity or physical failures.
    pub async fn rotate(&self) -> Result<()> {
        self.query(WalManager::rotate).await
    }

    /// Retains the WAL generation containing `sequence` until the returned
    /// lease is consumed or dropped.
    ///
    /// # Errors
    /// Returns admission, ownership or retention errors.
    pub async fn retain_from(&self, sequence: u64) -> Result<WalRetentionLease> {
        self.query(move |raw| raw.retain_from(sequence)).await
    }

    /// Runs an operation against a lease-bound capture on the owned worker.
    /// The lease is moved into the dispatched action, so dropping the waiting
    /// future cannot release it while physical capture work is still running.
    ///
    /// # Errors
    /// Returns admission, retention, capture or action errors.
    pub async fn capture_with_lease<T, F>(&self, lease: WalRetentionLease, action: F) -> Result<T>
    where
        T: Send + 'static,
        F: for<'a> FnOnce(&mut WalCapture<'a>) -> Result<T> + Send + 'static,
    {
        self.query(move |raw| {
            let mut capture = raw.capture_with_lease(&lease)?;
            action(&mut capture)
        })
        .await
    }

    /// Enumerates owned physical segments.
    /// # Errors
    /// Returns admission or filesystem failures.
    pub async fn log_files(&self) -> Result<Vec<PathBuf>> {
        self.query(WalManager::log_files).await
    }
    /// Reads checked checkpoint metadata under raw admission.
    /// # Errors
    /// Returns admission, filesystem or corruption failures.
    pub async fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        self.query(WalManager::read_checkpoint_metadata).await
    }
    /// Reads the physical byte count.
    /// # Errors
    /// Returns admission, filesystem or capacity failures.
    pub async fn size_bytes(&self) -> Result<usize> {
        self.query(WalManager::size_bytes).await
    }
    /// Reads the durable checkpoint timestamp.
    /// # Errors
    /// Returns admission, filesystem or corruption failures.
    pub async fn last_checkpoint_timestamp(&self) -> Result<Option<u64>> {
        self.query(WalManager::last_checkpoint_timestamp).await
    }

    /// Returns the cached record count.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.controller.raw.record_count()
    }
    /// Returns the canonical resource locator.
    #[must_use]
    pub fn dir(&self) -> &Path {
        self.controller.raw.dir()
    }
    /// Returns the cached durability configuration.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.controller.raw.durability_mode()
    }
    /// Returns the cached physical sequence.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.controller.raw.current_sequence()
    }
    /// Returns the cached segment locator.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.controller.raw.path()
    }
    /// Returns raw outcome-ambiguous failure state.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.controller.raw.is_poisoned()
    }
    /// Returns the cached checkpoint epoch without filesystem dispatch.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        self.controller.raw.checkpoint_epoch()
    }

    fn request_close(&self) {
        let idle = {
            let mut state = self.controller.admission.lock();
            if state.phase != Phase::Open {
                return;
            }
            state.phase = Phase::Closing;
            if state.active {
                false
            } else {
                state.active = true;
                true
            }
        };
        // Mint the idle continuation before waking arbitrary semaphore waiters;
        // unwinding during wakeup must not leave Closing without an owner.
        let ticket = idle.then(|| DispatchTicket {
            controller: Arc::clone(&self.controller),
            permit: None,
            active: true,
            interrupted: false,
        });
        self.controller.dispatch.close();
        if let Some(ticket) = ticket {
            match Handle::try_current() {
                Ok(handle) => {
                    let _ = submit_owned(&handle, Box::new(|| Ok(())), Some(ticket), None);
                }
                Err(_) => drop(ticket), // Contained close; never reenter a missing runtime.
            }
        }
    }

    /// Irreversibly seals admission and observes the shared physical close.
    ///
    /// # Errors
    /// Returns observer capacity refusal or the original shared close failure.
    /// Cancelling an observer does not cancel a polled close.
    pub async fn close(&self) -> CloseResult {
        self.request_close();
        if let Some(result) = self.controller.close_result() {
            return result;
        }
        let receipt = Arc::clone(&self.controller.callers).try_acquire_owned();
        let Ok(_receipt) = receipt else {
            return self
                .controller
                .close_result()
                .unwrap_or_else(|| Err(WalCloseError::new(capacity_error())));
        };
        loop {
            let notified = self.controller.completed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self.controller.close_result() {
                return result;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::mpsc;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    async fn idle(wal: &AsyncWalManager) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while wal.controller.admission.lock().active
                || wal.controller.dispatch.available_permits() != 1
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("physical dispatch retired");
    }

    fn acquire_released_recovery(path: &Path) -> super::super::WalRecovery {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            match super::super::WalRecovery::new(path) {
                Ok(owner) => return owner,
                Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "physical W release did not complete"
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("unexpected recovery acquisition failure: {error}"),
            }
        }
    }

    fn ticket(wal: &AsyncWalManager) -> DispatchTicket {
        let permit = Arc::clone(&wal.controller.dispatch)
            .try_acquire_owned()
            .unwrap();
        wal.controller.admission.lock().active = true;
        DispatchTicket {
            controller: Arc::clone(&wal.controller),
            permit: Some(permit),
            active: true,
            interrupted: false,
        }
    }

    #[derive(Clone, Debug, serde::Deserialize)]
    struct ProbeRecord<'a> {
        #[serde(skip)]
        marker: std::marker::PhantomData<&'a ()>,
        #[serde(skip)]
        calls: Option<Arc<std::sync::atomic::AtomicUsize>>,
        #[serde(skip)]
        started: Option<mpsc::Sender<()>>,
        #[serde(skip)]
        release: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
        panic_during_encoding: bool,
    }

    impl serde::Serialize for ProbeRecord<'_> {
        fn serialize<S: serde::Serializer>(
            &self,
            serializer: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            if let Some(calls) = &self.calls {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(started) = &self.started {
                started.send(()).unwrap();
            }
            assert!(!self.panic_during_encoding, "injected preparation unwind");
            if let Some(release) = &self.release {
                release.lock().recv_timeout(Duration::from_secs(5)).unwrap();
            }
            serializer.serialize_bool(false)
        }
    }

    impl WalEntry for ProbeRecord<'_> {
        fn validate_recovery(&self) -> std::result::Result<(), String> {
            if let Some(calls) = &self.calls {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
        }
        fn requires_sync(&self) -> bool {
            false
        }
        fn is_commit(&self) -> bool {
            false
        }
        fn is_abort(&self) -> bool {
            false
        }
        fn is_checkpoint(&self) -> bool {
            false
        }
        fn make_checkpoint(_: TransactionId) -> Self {
            Self {
                marker: std::marker::PhantomData,
                calls: None,
                started: None,
                release: None,
                panic_during_encoding: false,
            }
        }
    }

    #[tokio::test]
    async fn unconsumed_responses_bound_callers_but_do_not_block_close() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let mut responses = Vec::new();
        for _ in 0..16 {
            let controller = Arc::clone(&wal.controller);
            let response = submit_owned(
                &Handle::try_current().unwrap(),
                Box::new(move || controller.raw.sync()),
                Some(ticket(&wal)),
                Some(wal.controller.caller().unwrap()),
            )
            .unwrap();
            idle(&wal).await;
            responses.push(response);
        }
        // A seventeenth caller must fail before payload validation/copy.
        let oversized = vec![0; super::super::MAX_WAL_FRAME_BYTES + 1];
        let error = wal.write_frame(&oversized, false).await.unwrap_err();
        assert!(
            matches!(error, Error::Io(ref error) if error.kind() == std::io::ErrorKind::WouldBlock)
        );
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probe = ProbeRecord {
            calls: Some(Arc::clone(&calls)),
            ..ProbeRecord::make_checkpoint(TransactionId::new(1))
        };
        assert!(wal.log_entry(&probe).await.is_err());
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "request capacity precedes both record validation and Serialize"
        );
        let close = wal.close().await;
        if let Err(error) = close {
            assert!(
                matches!(error.0.as_ref(), Error::Io(error) if error.kind() == std::io::ErrorKind::WouldBlock)
            );
        }
        idle(&wal).await;
        wal.close().await.unwrap();
        assert!(
            WalManager::open(dir.path()).is_ok(),
            "responses cannot retain W"
        );
        for response in responses {
            receive(response).await.unwrap();
        }
    }

    #[tokio::test]
    async fn frame_retirement_precedes_physical_permit_release_on_completion_and_discard() {
        struct Frame {
            bytes: Vec<u8>,
            controller: std::sync::Weak<AsyncController>,
            dropped: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl Drop for Frame {
            fn drop(&mut self) {
                let controller = self.controller.upgrade().unwrap();
                assert_eq!(
                    controller.dispatch.available_permits(),
                    0,
                    "a second preparation must not overlap the retiring frame"
                );
                self.dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        for discard in [false, true] {
            let dir = super::super::test_wal_dir().unwrap();
            let wal = AsyncWalManager::open(dir.path()).await.unwrap();
            let record = WalRecord::EpochAdvance {
                epoch: EpochId::new(3),
            };
            let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let frame = Frame {
                bytes: encode_frame(&record).unwrap(),
                controller: Arc::downgrade(&wal.controller),
                dropped: Arc::clone(&dropped),
            };
            let controller = Arc::clone(&wal.controller);
            let intent = WalFrameIntent::capture(&record);
            let receiver = submit_using(
                Box::new(move || controller.raw.write_frame(&frame.bytes, intent)),
                Some(ticket(&wal)),
                Some(wal.controller.caller().unwrap()),
                |work| {
                    if discard {
                        drop(work);
                    } else {
                        work();
                    }
                },
            )
            .unwrap();
            assert_eq!(dropped.load(std::sync::atomic::Ordering::Relaxed), 1);
            assert_eq!(wal.controller.dispatch.available_permits(), 1);
            assert_eq!(receive(receiver).await.is_err(), discard);
            assert_eq!(wal.record_count(), u64::from(!discard));
        }
    }

    #[tokio::test]
    async fn close_does_not_wait_for_unadmitted_serialization() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = Arc::new(AsyncWalManager::open(dir.path()).await.unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let record = ProbeRecord {
            started: Some(started_tx),
            release: Some(Arc::new(Mutex::new(release_rx))),
            ..ProbeRecord::make_checkpoint(TransactionId::new(1))
        };
        let owner = Arc::clone(&wal);
        let preparing = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(owner.log_entry(&record))
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(wal.controller.dispatch.available_permits(), 0);
        assert!(!wal.controller.admission.lock().active);
        tokio::time::timeout(Duration::from_secs(2), wal.close())
            .await
            .unwrap()
            .unwrap();
        assert!(WalManager::open(dir.path()).is_ok());
        release_tx.send(()).unwrap();
        assert!(preparing.join().unwrap().is_err());
        assert_eq!(wal.record_count(), 0);
    }

    #[tokio::test]
    async fn preparation_unwind_refunds_slots_without_poisoning_and_borrowed_types_need_no_static_bound()
     {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let record = ProbeRecord {
            panic_during_encoding: true,
            ..ProbeRecord::make_checkpoint(TransactionId::new(1))
        };
        let mut future = Box::pin(wal.log_entry(&record));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| poll_once(future.as_mut())))
                .is_err()
        );
        drop(future);
        assert_eq!(wal.controller.dispatch.available_permits(), 1);
        assert_eq!(wal.controller.callers.available_permits(), 16);
        assert!(!wal.is_poisoned());
        async fn borrowed<'a>(wal: &AsyncWalManager, _lifetime: &'a ()) {
            let record: ProbeRecord<'a> = ProbeRecord::make_checkpoint(TransactionId::new(1));
            wal.log_entry(&record).await.unwrap();
        }
        borrowed(&wal, &()).await;
        assert_eq!(wal.record_count(), 1);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[tokio::test]
    async fn cancelled_observer_cannot_skip_owned_acknowledgement_poison() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let record = WalRecord::Committed {
            transaction_id: TransactionId::new(4),
            epoch: EpochId::new(5),
        };
        grafeo_common::testing::wal_failure::enable_commit_ack_failure_once();
        let intent = WalFrameIntent::capture(&record);
        let data = encode_frame(&record).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let receipt = wal.controller.caller().unwrap();
        let permit = wal.permit().await.unwrap();
        let mut future = Box::pin(wal.dispatch(receipt, permit, move |raw| {
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            raw.write_frame(&data, intent)
        }));
        assert!(poll_once(future.as_mut()).is_pending());
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(future);
        release_tx.send(()).unwrap();
        idle(&wal).await;
        assert!(wal.is_poisoned());
        assert_eq!(wal.record_count(), 1);
        assert!(
            wal.log(&WalRecord::EpochAdvance {
                epoch: EpochId::new(6)
            })
            .await
            .is_err()
        );
        drop(wal);
        let records = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(
            matches!(records.as_slice(), [WalRecord::Committed { epoch, .. }] if *epoch == EpochId::new(5))
        );
    }

    #[tokio::test]
    async fn dropped_started_observer_keeps_actual_work_and_cancelled_close_continuation() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let receipt = wal.controller.caller().unwrap();
        let permit = wal.permit().await.unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let record = WalRecord::EpochAdvance {
            epoch: EpochId::new(17),
        };
        let data = encode_frame(&record).unwrap();
        let intent = WalFrameIntent::capture(&record);
        let mut append = Box::pin(wal.dispatch(receipt, permit, move |raw| {
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            raw.write_frame(&data, intent)
        }));
        assert!(poll_once(append.as_mut()).is_pending());
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(append);
        let mut close = Box::pin(wal.close());
        assert!(poll_once(close.as_mut()).is_pending());
        drop(close);
        assert!(WalManager::open(dir.path()).is_err());
        let raw_weak = Arc::downgrade(&wal.controller);
        drop(wal);
        assert!(raw_weak.upgrade().is_some(), "started job owns controller");
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while raw_weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let records = acquire_released_recovery(dir.path()).recover().unwrap();
        assert!(
            matches!(records.as_slice(), [WalRecord::EpochAdvance { epoch }] if *epoch == EpochId::new(17))
        );
    }

    #[tokio::test]
    async fn never_polled_close_does_not_seal_and_waiting_callers_reject_after_polled_close() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        drop(wal.close());
        wal.sync().await.unwrap();
        // Preparation permission is held but there is no admitted physical job.
        let preparing = wal.permit().await.unwrap();
        let mut waiting = Box::pin(wal.write_frame(&[1], false));
        assert!(poll_once(waiting.as_mut()).is_pending());
        wal.close().await.unwrap();
        assert!(waiting.await.is_err());
        // A preparation finishing after close cannot dispatch its bytes.
        let receipt = Arc::clone(&wal.controller.callers)
            .try_acquire_owned()
            .unwrap();
        assert!(
            wal.dispatch(receipt, preparing, |raw| raw
                .write_frame(&[1], WalFrameIntent::raw(false)))
                .await
                .is_err()
        );
        assert!(WalManager::open(dir.path()).is_ok());
    }

    #[tokio::test]
    async fn pending_submission_unwind_retains_stub_receipt_but_never_executes_command() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let runtime_stub = Mutex::new(None);
        let controller = Arc::clone(&wal.controller);
        let result = submit_using(
            Box::new(move || {
                controller.raw.log(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(9),
                })
            }),
            Some(ticket(&wal)),
            Some(wal.controller.caller().unwrap()),
            |work| {
                *runtime_stub.lock() = Some(work);
                panic!("injected queue-before-submit-error");
            },
        );
        assert!(result.is_err());
        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.controller.callers.available_permits(), 15);
        assert_eq!(wal.controller.dispatch.available_permits(), 1);
        let work = runtime_stub.lock().take().unwrap();
        work();
        assert_eq!(wal.controller.callers.available_permits(), 16);
        assert_eq!(wal.record_count(), 0);
        wal.sync().await.unwrap();
    }

    #[tokio::test]
    async fn taken_submission_unwind_observes_the_actual_success_or_failure() {
        for fail in [false, true] {
            let dir = super::super::test_wal_dir().unwrap();
            let wal = AsyncWalManager::open(dir.path()).await.unwrap();
            let controller = Arc::clone(&wal.controller);
            let (taken_tx, taken_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let worker = Mutex::new(None);
            let record = WalRecord::EpochAdvance {
                epoch: EpochId::new(23),
            };
            let data = encode_frame(&record).unwrap();
            let intent = WalFrameIntent::capture(&record);
            if fail {
                // Use an actual admitted I/O failure without private intent access.
                std::fs::hard_link(wal.path(), dir.path().join("duplicate-link")).unwrap();
            }
            let receiver = submit_using(
                Box::new(move || {
                    taken_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    controller.raw.write_frame(&data, intent)
                }),
                Some(ticket(&wal)),
                Some(wal.controller.caller().unwrap()),
                |work| {
                    *worker.lock() = Some(std::thread::spawn(work));
                    taken_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    panic!("injected taken-before-submit-error");
                },
            )
            .expect("Taken submission must preserve response channel");
            let mut response = Box::pin(receive(receiver));
            assert!(poll_once(response.as_mut()).is_pending());
            assert!(WalManager::open(dir.path()).is_err());
            release_tx.send(()).unwrap();
            let outcome = response.await;
            worker.lock().take().unwrap().join().unwrap();
            if fail {
                let error = outcome.unwrap_err();
                assert!(
                    !matches!(error, Error::Io(ref error) if error.kind() == std::io::ErrorKind::Interrupted)
                );
                assert!(wal.is_poisoned());
            } else {
                outcome.unwrap();
                assert!(!wal.is_poisoned());
                wal.close().await.unwrap();
                let records = super::super::WalRecovery::new(dir.path())
                    .unwrap()
                    .recover()
                    .unwrap();
                assert!(
                    matches!(records.as_slice(), [WalRecord::EpochAdvance { epoch }] if *epoch == EpochId::new(23))
                );
            }
        }
    }

    #[test]
    fn a_single_blocking_thread_drains_the_active_job_instead_of_queuing_a_waiter() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let dir = super::super::test_wal_dir().unwrap();
            let wal = AsyncWalManager::open(dir.path()).await.unwrap();
            let (blocked_tx, blocked_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                blocked_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            });
            blocked_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let record = WalRecord::EpochAdvance {
                epoch: EpochId::new(31),
            };
            let mut write = Box::pin(wal.log(&record));
            assert!(poll_once(write.as_mut()).is_pending());
            let mut close = Box::pin(wal.close());
            assert!(poll_once(close.as_mut()).is_pending());
            drop(close);
            release_tx.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(5), write)
                .await
                .unwrap()
                .unwrap();
            tokio::time::timeout(Duration::from_secs(5), wal.close())
                .await
                .unwrap()
                .unwrap();
            blocker.await.unwrap();
        });
    }

    #[tokio::test]
    async fn discarded_queued_write_carries_the_real_cancelled_close_continuation() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let queued = Mutex::new(None);
        let controller = Arc::clone(&wal.controller);
        let receiver = submit_using(
            Box::new(move || {
                controller.raw.log(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(53),
                })
            }),
            Some(ticket(&wal)),
            Some(wal.controller.caller().unwrap()),
            |work| {
                *queued.lock() = Some(work);
            },
        )
        .unwrap();
        let mut close = Box::pin(wal.close());
        assert!(poll_once(close.as_mut()).is_pending());
        drop(close);
        assert!(WalManager::open(dir.path()).is_err());
        // Model actual runtime closure discard, never under the envelope lock.
        let work = queued.lock().take().unwrap();
        drop(work);
        assert!(receive(receiver).await.is_err());
        wal.close().await.unwrap();
        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.controller.callers.available_permits(), 16);
        assert_eq!(wal.controller.dispatch.available_permits(), 1);
        let mut recovery = super::super::WalRecovery::new(dir.path()).unwrap();
        assert!(recovery.recover().unwrap().is_empty());
        drop(recovery);
        WalManager::open(dir.path()).unwrap().close().unwrap();
    }

    #[tokio::test]
    async fn constructor_taken_submission_unwind_delivers_or_drops_the_complete_owner() {
        for outcome in [0, 1, 2] {
            let dir = super::super::test_wal_dir().unwrap();
            let path = dir.path().to_path_buf();
            let (created_tx, created_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let worker = Mutex::new(None);
            let receiver = submit_using(
                Box::new(move || {
                    let raw = WalManager::with_config(
                        path,
                        WalConfig {
                            durability: DurabilityMode::Batch {
                                max_delay_ms: 10,
                                max_records: 1,
                            },
                            ..Default::default()
                        },
                    )?;
                    created_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    Ok(raw)
                }),
                None,
                None,
                |work| {
                    *worker.lock() = Some(std::thread::spawn(work));
                    created_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    panic!("injected constructor taken-before-submit-error");
                },
            )
            .unwrap();
            assert!(WalManager::open(dir.path()).is_err());
            if outcome == 1 {
                release_tx.send(()).unwrap();
                let raw = receive(receiver).await.unwrap();
                assert!(WalManager::open(dir.path()).is_err());
                raw.close().unwrap();
            } else if outcome == 0 {
                drop(receiver);
                assert!(WalManager::open(dir.path()).is_err());
                release_tx.send(()).unwrap();
            } else {
                release_tx.send(()).unwrap();
                worker.lock().take().unwrap().join().unwrap();
                assert!(
                    WalManager::open(dir.path()).is_err(),
                    "buffered constructor result owns its supervisor"
                );
                drop(receiver);
            }
            if let Some(worker) = worker.lock().take() {
                worker.join().unwrap();
            }
            assert!(WalManager::open(dir.path()).is_ok());
        }
    }

    #[test]
    fn missing_and_shut_down_runtimes_do_not_leave_a_pending_constructor_sender() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wal");
        let mut missing = Box::pin(AsyncWalManager::open(&path));
        assert!(matches!(poll_once(missing.as_mut()), Poll::Ready(Err(_))));
        assert!(!path.exists());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let handle = runtime.handle().clone();
        drop(runtime);
        let _entered = handle.enter();
        let mut closed = Box::pin(AsyncWalManager::open(&path));
        assert!(matches!(poll_once(closed.as_mut()), Poll::Ready(Err(_))));
        assert!(!path.exists());
    }

    #[test]
    fn relative_constructor_freezes_cwd_before_queued_execution() {
        if std::env::var_os("GRAFEO_ASYNC_CWD_CHILD").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "wal::async_log::ownership_tests::relative_constructor_freezes_cwd_before_queued_execution", "--test-threads=1"])
                .env("GRAFEO_ASYNC_CWD_CHILD", "1").status().unwrap();
            assert!(status.success(), "isolated CWD witness failed: {status}");
            return;
        }
        struct RestoreDirectory(PathBuf);
        impl Drop for RestoreDirectory {
            fn drop(&mut self) {
                std::env::set_current_dir(&self.0).unwrap();
            }
        }
        let original = tempfile::tempdir().unwrap();
        let changed = tempfile::tempdir().unwrap();
        let _restore = RestoreDirectory(std::env::current_dir().unwrap());
        std::env::set_current_dir(original.path()).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (started_tx, started_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            });
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let mut constructor = Box::pin(AsyncWalManager::open("wal"));
            assert!(poll_once(constructor.as_mut()).is_pending());
            std::env::set_current_dir(changed.path()).unwrap();
            release_tx.send(()).unwrap();
            let wal = constructor.await.unwrap();
            assert_eq!(
                wal.dir(),
                std::fs::canonicalize(original.path().join("wal")).unwrap()
            );
            assert!(!changed.path().join("wal").exists());
            wal.close().await.unwrap();
            blocker.await.unwrap();
        });
    }

    #[test]
    fn runtime_shutdown_keeps_a_queued_job_owned_until_its_actual_completion() {
        let dir = super::super::test_wal_dir().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        let (release_tx, release_rx) = mpsc::channel();
        let weak = runtime.block_on(async {
            let wal = AsyncWalManager::open(dir.path()).await.unwrap();
            let (blocked_tx, blocked_rx) = mpsc::channel();
            tokio::task::spawn_blocking(move || {
                blocked_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            });
            blocked_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let record = WalRecord::EpochAdvance {
                epoch: EpochId::new(41),
            };
            let mut queued = Box::pin(wal.log(&record));
            assert!(poll_once(queued.as_mut()).is_pending());
            drop(queued);
            let weak = Arc::downgrade(&wal.controller);
            drop(wal);
            weak
        });
        runtime.shutdown_background();
        assert!(WalManager::open(dir.path()).is_err());
        release_tx.send(()).unwrap();
        let mut recovery = acquire_released_recovery(dir.path());
        assert!(weak.upgrade().is_none());
        let records = recovery.recover().unwrap();
        // Pinned Tokio's busy worker drains this queue before checking shutdown.
        // Cancellation/shutdown is not a promise that queued work cannot run.
        assert!(
            matches!(records.as_slice(), [WalRecord::EpochAdvance { epoch }]
            if *epoch == EpochId::new(41))
        );
        drop(recovery);
        WalManager::open(dir.path()).unwrap().close().unwrap();
    }

    #[tokio::test]
    async fn constructor_failure_after_batch_start_joins_before_releasing_w() {
        let dir = super::super::test_wal_dir().unwrap();
        let path = dir.path().to_path_buf();
        let receiver = submit_owned(
            &Handle::try_current().unwrap(),
            Box::new(move || {
                let _raw = WalManager::with_config(
                    path,
                    WalConfig {
                        durability: DurabilityMode::Batch {
                            max_delay_ms: 1,
                            max_records: 1,
                        },
                        ..Default::default()
                    },
                )?;
                Err::<WalManager, Error>(std::io::Error::from_raw_os_error(5).into())
            }),
            None,
            None,
        )
        .unwrap();
        let result = receive(receiver).await;
        assert!(matches!(result, Err(Error::Io(error)) if error.raw_os_error() == Some(5)));
        assert!(WalManager::open(dir.path()).is_ok());
    }

    #[tokio::test]
    async fn failed_close_observers_share_the_original_cause_and_keep_ownership() {
        use std::error::Error as _;
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        std::fs::hard_link(wal.path(), dir.path().join("duplicate-link")).unwrap();
        let first = wal.close().await.unwrap_err();
        let second = wal.close().await.unwrap_err();
        assert!(Arc::ptr_eq(&first.0, &second.0));
        assert_eq!(first.error_code(), second.error_code());
        assert!(first.source().is_some());
        assert!(WalManager::open(dir.path()).is_err());
        assert!(wal.sync().await.is_err());
        std::fs::remove_file(dir.path().join("duplicate-link")).unwrap();
        drop(wal);
        assert!(WalManager::open(dir.path()).is_ok());
    }

    #[cfg(feature = "testing-crash-injection")]
    #[tokio::test]
    async fn physical_unwind_poison_and_failed_cleanup_do_not_retry_buffered_bytes() {
        let dir = super::super::test_wal_dir().unwrap();
        let wal = AsyncWalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Batch {
                    max_delay_ms: 1,
                    max_records: u64::MAX,
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let before = std::fs::read(wal.path()).unwrap();
        let receipt = wal.controller.caller().unwrap();
        let permit = wal.permit().await.unwrap();
        let result = wal
            .dispatch(receipt, permit, |raw| {
                grafeo_common::testing::crash::enable_crash_named("wal_after_write");
                struct Reset;
                impl Drop for Reset {
                    fn drop(&mut self) {
                        grafeo_common::testing::crash::disable_crash();
                    }
                }
                let _reset = Reset;
                raw.log(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(1),
                })
            })
            .await;
        assert!(result.is_err());
        assert!(wal.is_poisoned());
        assert!(wal.sync().await.is_err());
        assert!(wal.close().await.is_err());
        assert_eq!(std::fs::read(wal.path()).unwrap(), before);
        drop(wal);
        assert!(
            super::super::WalRecovery::new(dir.path())
                .unwrap()
                .recover()
                .unwrap()
                .is_empty()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_wal_dir as tempdir;
    use super::*;
    use grafeo_common::types::NodeId;
    use grafeo_common::utils::error::StorageError;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::fs;
    const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

    async fn read_checkpoint_metadata_from(path: &Path) -> Result<Option<CheckpointMetadata>> {
        let wal = AsyncWalManager::open(path).await?;
        let result = wal.read_checkpoint_metadata().await;
        // Dropping the async facade only requests close. These tests reopen
        // immediately, so await physical ownership release, including on a
        // rejected metadata image, rather than racing the close worker.
        wal.close()
            .await
            .map_err(|error| Error::Internal(error.to_string()))?;
        result
    }

    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()?
            .to_str()?
            .strip_prefix("wal_")?
            .parse()
            .ok()
    }

    async fn wal_file_image(wal: &AsyncWalManager) -> Vec<(PathBuf, Vec<u8>)> {
        let mut image = Vec::new();
        for path in wal.log_files().await.unwrap() {
            let bytes = fs::read(&path).await.unwrap();
            image.push((path, bytes));
        }
        image
    }

    #[tokio::test]
    async fn test_async_wal_write() {
        let dir = tempdir().unwrap();

        let wal = AsyncWalManager::open(dir.path()).await.unwrap();

        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
        );

        wal.log(&record).await.unwrap();
        wal.flush().await.unwrap();

        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn invalid_record_and_checkpoint_identity_are_rejected_before_mutation() {
        let dir = tempdir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let before = wal_file_image(&wal).await;

        let record_error = wal
            .log(&WalRecord::Committed {
                transaction_id: TransactionId::INVALID,
                epoch: EpochId::new(1),
            })
            .await
            .unwrap_err();
        assert!(matches!(record_error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal).await, before);
        assert_eq!(wal.record_count(), 0);

        let checkpoint_error = wal
            .checkpoint(TransactionId::INVALID, EpochId::new(1))
            .await
            .unwrap_err();
        assert!(matches!(checkpoint_error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal).await, before);
        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.current_sequence(), 0);

        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .expect("deterministic validation must leave the async WAL usable");
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn oversized_raw_frame_is_rejected_before_writer_state_changes() {
        let dir = tempdir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let before = wal_file_image(&wal).await;
        let count_before = wal.record_count();
        let payload = vec![0u8; super::super::MAX_WAL_FRAME_BYTES + 1];

        let error = wal.write_frame(&payload, false).await.unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal).await, before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.current_sequence(), 0);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn catalog_failpoints_bracket_direct_async_wal_manager_append() {
        let dir = tempdir().unwrap();
        let wal = AsyncWalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .await
        .unwrap();
        let batch = |epoch| WalRecord::CatalogBatchV3 {
            created_graph_incarnations: vec![],
            dropped_graph_incarnations: vec![],
            version: 2,
            epoch: EpochId::new(epoch),
            catalog_state: Vec::new(),
            created_graphs: Vec::new(),
            dropped_graphs: Vec::new(),
        };

        grafeo_common::testing::wal_failure::enable_catalog_batch_log_failure_once();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .expect("the catalog hook must ignore non-catalog records");
        assert!(wal.log(&batch(2)).await.is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_catalog_batch_log().is_ok(),
            "the pre-append hook is one-shot"
        );
        assert_eq!(wal.record_count(), 1);

        assert!(wal.is_poisoned());
        drop(wal);
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        grafeo_common::testing::wal_failure::enable_catalog_batch_ack_failure_once();
        assert!(wal.log(&batch(3)).await.is_err());
        assert_eq!(wal.record_count(), 1);

        drop(wal);
        let recovered = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(matches!(
            recovered.as_slice(),
            [WalRecord::EpochAdvance { epoch: first }, WalRecord::CatalogBatchV3 { epoch: second, .. }]
                if *first == EpochId::new(1) && *second == EpochId::new(3)
        ));
    }

    #[tokio::test]
    async fn test_async_wal_rotation() {
        let dir = tempdir().unwrap();

        // Small max size to force rotation
        let config = WalConfig {
            max_log_size: 100,
            ..Default::default()
        };

        let wal = AsyncWalManager::with_config(dir.path(), config)
            .await
            .unwrap();

        // Write enough records to trigger rotation
        for i in 0..10 {
            let record = WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Person".to_string()],
                },
            );
            wal.log(&record).await.unwrap();
        }

        wal.flush().await.unwrap();

        // Should have multiple log files
        let files = wal.log_files().await.unwrap();
        assert!(
            files.len() > 1,
            "Expected multiple log files after rotation"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_tiny_segments_keep_monotonic_active_sequence_and_recover_all() {
        use super::super::{LpgMutationOp, WalRecovery};

        const TASKS: u64 = 12;
        const TRANSACTIONS_PER_TASK: u64 = 20;

        let dir = tempdir().unwrap();
        let config = WalConfig {
            durability: DurabilityMode::NoSync,
            max_log_size: 128,
            ..Default::default()
        };
        let wal = Arc::new(
            AsyncWalManager::with_config(dir.path(), config)
                .await
                .unwrap(),
        );

        let mut tasks = Vec::new();
        for task in 0..TASKS {
            let wal = Arc::clone(&wal);
            tasks.push(tokio::spawn(async move {
                for offset in 0..TRANSACTIONS_PER_TASK {
                    let value = task * TRANSACTIONS_PER_TASK + offset + 1;
                    let transaction_id = TransactionId::new(value);
                    wal.log(&WalRecord::lpg(
                        transaction_id,
                        grafeo_common::types::GraphPath::root(),
                        LpgMutationOp::CreateNode {
                            id: NodeId::new(value),
                            labels: vec!["Concurrent".to_string()],
                        },
                    ))
                    .await?;
                    tokio::task::yield_now().await;
                    wal.log(&WalRecord::Committed {
                        transaction_id,
                        epoch: EpochId::new(value),
                    })
                    .await?;
                }
                Result::<()>::Ok(())
            }));
        }
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        wal.sync().await.unwrap();

        let files = wal.log_files().await.unwrap();
        let sequences: Vec<u64> = files
            .iter()
            .map(|path| sequence_from_path(path).unwrap())
            .collect();
        assert_eq!(
            sequences,
            (0..sequences.len() as u64).collect::<Vec<_>>(),
            "serialized rotation must not skip or reverse segment publication"
        );
        let active_sequence = sequence_from_path(&wal.path()).unwrap();
        assert_eq!(
            active_sequence,
            wal.current_sequence(),
            "the published atomic sequence must describe the installed writer"
        );
        assert_eq!(Some(&active_sequence), sequences.last());
        assert_eq!(wal.record_count(), TASKS * TRANSACTIONS_PER_TASK * 2);

        drop(wal);
        let records = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert_eq!(
            records.len() as u64,
            TASKS * TRANSACTIONS_PER_TASK * 2,
            "every concurrently committed frame must remain recovery-compatible"
        );
        let recovered_nodes: HashSet<NodeId> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::LpgMutation {
                    op: LpgMutationOp::CreateNode { id, .. },
                    ..
                } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(recovered_nodes.len() as u64, TASKS * TRANSACTIONS_PER_TASK);
    }

    #[tokio::test]
    async fn test_async_durability_modes() {
        let dir = tempdir().unwrap();

        // Test Sync mode
        let config = WalConfig {
            durability: DurabilityMode::Sync,
            ..Default::default()
        };
        let wal = AsyncWalManager::with_config(dir._temporary.path().join("sync.wal"), config)
            .await
            .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .await
        .unwrap();

        // Test NoSync mode
        let config = WalConfig {
            durability: DurabilityMode::NoSync,
            ..Default::default()
        };
        let wal = AsyncWalManager::with_config(dir._temporary.path().join("nosync.wal"), config)
            .await
            .unwrap();
        wal.log(&WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec![],
            },
        ))
        .await
        .unwrap();

        // Test Batch mode
        let config = WalConfig {
            durability: DurabilityMode::Batch {
                max_delay_ms: 10,
                max_records: 5,
            },
            ..Default::default()
        };
        let wal = AsyncWalManager::with_config(dir._temporary.path().join("batch.wal"), config)
            .await
            .unwrap();
        for i in 0..10 {
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(i),
                    labels: vec![],
                },
            ))
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn test_async_checkpoint() {
        let dir = tempdir().unwrap();

        let wal = AsyncWalManager::open(dir.path()).await.unwrap();

        // Write some records
        wal.log(&WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            },
        ))
        .await
        .unwrap();

        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .await
        .unwrap();

        // Create checkpoint
        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .await
            .unwrap();

        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
    }

    #[tokio::test]
    async fn pending_checkpoint_is_rejected_without_mutation_and_initial_round_trips() {
        let dir = tempdir().unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let files_before = wal_file_image(&wal).await;
        let count_before = wal.record_count();
        let sequence_before = wal.current_sequence();

        let error = wal
            .checkpoint(TransactionId::new(1), EpochId::PENDING)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal).await, files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.current_sequence(), sequence_before);
        assert_eq!(wal.checkpoint_epoch(), None);
        assert!(!dir.path().join(CHECKPOINT_METADATA_FILE).exists());

        wal.checkpoint(TransactionId::new(1), EpochId::INITIAL)
            .await
            .unwrap();
        let metadata = wal.read_checkpoint_metadata().await.unwrap().unwrap();
        assert_eq!(metadata.epoch, EpochId::INITIAL);
        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::INITIAL));
    }

    #[tokio::test]
    async fn hostile_pending_checkpoint_metadata_is_corruption() {
        let dir = tempdir().unwrap();
        let hostile = CheckpointMetadata {
            format_version: 5,
            retired_before: 0,
            epoch: EpochId::PENDING,
            log_sequence: 0,
            timestamp_ms: 0,
            transaction_id: TransactionId::new(1),
        };
        let data = bincode::serde::encode_to_vec(hostile, bincode::config::standard()).unwrap();
        fs::write(dir.path().join(CHECKPOINT_METADATA_FILE), data)
            .await
            .unwrap();

        let error = read_checkpoint_metadata_from(dir.path()).await.unwrap_err();
        assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
    }

    #[tokio::test]
    async fn checkpoint_metadata_reader_is_exact_bounded_and_validates_identity()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir().unwrap();
        let path = dir.path().join(CHECKPOINT_METADATA_FILE);

        assert!(
            read_checkpoint_metadata_from(dir.path())
                .await
                .unwrap()
                .is_none()
        );

        let valid = CheckpointMetadata {
            format_version: 5,
            retired_before: 3,
            epoch: EpochId::new(7),
            log_sequence: 3,
            timestamp_ms: 42,
            transaction_id: TransactionId::new(11),
        };
        let valid_bytes =
            bincode::serde::encode_to_vec(&valid, bincode::config::standard()).unwrap();
        // This helper opens a writer; its valid metadata needs a real boundary.
        fs::write(dir.path().join("wal_00000003.log"), []).await?;
        fs::write(&path, &valid_bytes).await.unwrap();
        let decoded = read_checkpoint_metadata_from(dir.path())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decoded.epoch, valid.epoch);
        assert_eq!(decoded.log_sequence, valid.log_sequence);
        assert_eq!(decoded.timestamp_ms, valid.timestamp_ms);
        assert_eq!(decoded.transaction_id, valid.transaction_id);

        let mut trailing = valid_bytes;
        trailing.push(0xa5);
        fs::write(&path, &trailing).await.unwrap();
        let trailing_error = read_checkpoint_metadata_from(dir.path()).await.unwrap_err();
        assert!(
            matches!(trailing_error, Error::Storage(StorageError::Corruption(_))),
            "{trailing_error:?}"
        );
        assert_eq!(fs::read(&path).await.unwrap(), trailing);

        let oversized = vec![0u8; super::super::log::MAX_CHECKPOINT_METADATA_BYTES + 1];
        fs::write(&path, &oversized).await.unwrap();
        let oversized_error = read_checkpoint_metadata_from(dir.path()).await.unwrap_err();
        assert!(
            matches!(oversized_error, Error::Storage(StorageError::Corruption(_))),
            "{oversized_error:?}"
        );
        assert_eq!(fs::read(&path).await.unwrap(), oversized);

        let invalid_transaction = CheckpointMetadata {
            transaction_id: TransactionId::INVALID,
            ..valid
        };
        let invalid_transaction_bytes =
            bincode::serde::encode_to_vec(invalid_transaction, bincode::config::standard())
                .unwrap();
        fs::write(&path, &invalid_transaction_bytes).await.unwrap();
        let invalid_transaction_error =
            read_checkpoint_metadata_from(dir.path()).await.unwrap_err();
        assert!(
            matches!(
                invalid_transaction_error,
                Error::Storage(StorageError::Corruption(_))
            ),
            "{invalid_transaction_error:?}"
        );
        assert_eq!(fs::read(&path).await.unwrap(), invalid_transaction_bytes);
        Ok(())
    }

    #[tokio::test]
    async fn exhausted_sequence_cannot_partially_append_a_checkpoint() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join(format!("wal_{}.log", u64::MAX)), [])
            .await
            .unwrap();
        let wal = AsyncWalManager::open(dir.path()).await.unwrap();
        let files_before = wal_file_image(&wal).await;
        let count_before = wal.record_count();

        let error = wal
            .checkpoint(TransactionId::new(1), EpochId::INITIAL)
            .await
            .unwrap_err();

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(
            error
                .to_string()
                .contains("WAL log sequence identity space is exhausted")
        );
        assert_eq!(wal_file_image(&wal).await, files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.checkpoint_epoch(), None);
        assert!(!dir.path().join(CHECKPOINT_METADATA_FILE).exists());
    }

    #[tokio::test]
    async fn threshold_crossing_at_exhausted_sequence_is_byte_and_count_identical() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join(format!("wal_{}.log", u64::MAX)), [])
            .await
            .unwrap();
        let wal = AsyncWalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::NoSync,
                // A complete current metadata frame crosses this threshold.
                max_log_size: 9,
                compression: false,
            },
        )
        .await
        .unwrap();
        let files_before = wal_file_image(&wal).await;
        let count_before = wal.record_count();

        let error = wal
            .write_frame(
                &encode_frame(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(1),
                })
                .unwrap(),
                false,
            )
            .await
            .unwrap_err();

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("rotation threshold"));
        assert_eq!(wal_file_image(&wal).await, files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.size_bytes().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn batch_background_sync_checks_an_idle_dirty_wal() {
        let dir = tempdir().unwrap();
        let wal = AsyncWalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Batch {
                    max_delay_ms: 100,
                    max_records: u64::MAX,
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .unwrap();
        // A second link makes the actual open handle invalid for further I/O.
        // Only the idle Batch worker runs after this point.
        fs::hard_link(wal.path(), dir.path().join("held-link"))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !wal.is_poisoned() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("idle Batch worker must inspect/sync without another request");
        assert!(wal.close().await.is_err());
    }
}
