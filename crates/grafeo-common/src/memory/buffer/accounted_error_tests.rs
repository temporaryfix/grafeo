use super::accounted_error::{
    InlineLock, refuse_next_test_allocation, reset_test_allocator_observation,
    test_allocation_attempts, test_deallocation_count,
};
use super::grant::GrantAccount;
use super::{
    AccountedErrorPublisher, AccountedErrorPublisherBuildFailure, BufferManager,
    BufferManagerConfig, MemoryGrant, MemoryGrantError, MemoryRegion,
};
use std::cell::Cell;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct TestFailure(&'static str);

impl fmt::Display for TestFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for TestFailure {}

#[derive(Debug)]
struct SendNotSyncFailure(Cell<usize>);

impl fmt::Display for SendNotSyncFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "send-only payload {}", self.0.get())
    }
}

impl std::error::Error for SendNotSyncFailure {}

struct StackFormatter {
    bytes: [u8; 128],
    len: usize,
}

impl StackFormatter {
    const fn new() -> Self {
        Self {
            bytes: [0; 128],
            len: 0,
        }
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("formatter stores UTF-8 input")
    }
}

impl fmt::Write for StackFormatter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let end = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
        let destination = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        destination.copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }
}

fn exact_manager(budget: usize) -> std::sync::Arc<BufferManager> {
    BufferManager::new(BufferManagerConfig {
        budget,
        hard_limit_fraction: 1.0,
        ..BufferManagerConfig::default()
    })
}

#[test]
fn inline_guard_unlocks_after_unwind_without_poison_or_parking_state() {
    let lock = InlineLock::new(0_usize);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut value = lock.lock();
        *value = 41;
        panic!("guard unwind");
    }));
    assert!(unwind.is_err());

    let mut value = lock.lock();
    assert_eq!(*value, 41);
    *value += 1;
    assert_eq!(*value, 42);
}

fn build_error<T>(grant: MemoryGrant) -> super::AccountedErrorPublisherBuildError
where
    T: std::error::Error + Send + 'static,
{
    match AccountedErrorPublisher::<T>::try_new(grant) {
        Ok(publisher) => {
            drop(publisher);
            panic!("publisher construction unexpectedly succeeded");
        }
        Err(error) => error,
    }
}

struct RollbackAccount {
    accounted: AtomicUsize,
    refuse_release: AtomicBool,
}

impl RollbackAccount {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            accounted: AtomicUsize::new(0),
            refuse_release: AtomicBool::new(false),
        })
    }
}

impl GrantAccount for RollbackAccount {
    fn release_accounted(
        &self,
        size: usize,
        _region: MemoryRegion,
    ) -> Result<(), MemoryGrantError> {
        if self.refuse_release.load(Ordering::Acquire) {
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "accounted-error-test",
            });
        }
        self.accounted
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(size)
            })
            .map(|_| ())
            .map_err(|current| MemoryGrantError::AccountingUnderflow {
                account: "accounted-error-test",
                accounted_bytes: current,
                release_bytes: size,
            })
    }

    fn try_reserve_growth(
        &self,
        size: usize,
        _region: MemoryRegion,
    ) -> Result<(), MemoryGrantError> {
        self.accounted
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(size)
            })
            .map(|_| ())
            .map_err(|current| MemoryGrantError::ArithmeticOverflow {
                current_bytes: current,
                additional_bytes: size,
            })
    }
}

#[test]
fn publisher_pre_admits_exact_shared_allocation() {
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");

    let publisher = AccountedErrorPublisher::try_new(grant).expect("exact admission");
    assert_eq!(publisher.granted_bytes(), required);
    assert_eq!(manager.allocated(), required);

    let error = publisher.publish(TestFailure("retained"));
    assert_eq!(error.granted_bytes(), required);
    assert_eq!(manager.allocated(), required);

    drop(error);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn one_byte_short_denies_before_allocating_and_retains_zero_grant() {
    reset_test_allocator_observation();
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required - 1);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");

    let error = build_error::<TestFailure>(grant);
    assert!(matches!(
        error.failure(),
        AccountedErrorPublisherBuildFailure::Admission(MemoryGrantError::LimitExceeded {
            requested_bytes,
            limit_bytes,
            ..
        }) if *requested_bytes == required && *limit_bytes == required - 1
    ));
    assert_eq!(error.grant().size(), 0);
    assert_eq!(test_allocation_attempts(), 0);
    assert_eq!(manager.allocated(), 0);
    drop(error);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn nonzero_grant_is_rejected_unchanged_before_allocating() {
    reset_test_allocator_observation();
    let manager = exact_manager(1);
    let grant = manager
        .try_allocate(1, MemoryRegion::ExecutionBuffers)
        .expect("one-byte grant");

    let error = build_error::<TestFailure>(grant);
    assert!(matches!(
        error.failure(),
        AccountedErrorPublisherBuildFailure::NonZeroGrant { bytes: 1 }
    ));
    assert_eq!(error.grant().size(), 1);
    assert_eq!(test_allocation_attempts(), 0);
    assert_eq!(manager.allocated(), 1);
    drop(error);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn allocator_refusal_rolls_back_to_zero_and_returns_retry_authority() {
    reset_test_allocator_observation();
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");

    let refusal = refuse_next_test_allocation();
    let error = build_error::<TestFailure>(grant);
    drop(refusal);
    assert!(matches!(
        error.failure(),
        AccountedErrorPublisherBuildFailure::Allocation
    ));
    assert_eq!(error.grant().size(), 0);
    assert_eq!(manager.allocated(), 0);
    assert_eq!(test_allocation_attempts(), 1);

    let publisher = AccountedErrorPublisher::<TestFailure>::try_new(error.into_grant())
        .expect("reconciled grant retries construction");
    assert_eq!(test_allocation_attempts(), 2);
    assert_eq!(manager.allocated(), required);
    drop(publisher);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn allocator_refusal_rollback_failure_returns_overcovered_retry_authority() {
    reset_test_allocator_observation();
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let account = RollbackAccount::new();
    account.refuse_release.store(true, Ordering::Release);
    let grant = MemoryGrant::new(
        Arc::clone(&account) as Arc<dyn GrantAccount>,
        0,
        MemoryRegion::ExecutionBuffers,
    );

    let refusal = refuse_next_test_allocation();
    let error = build_error::<TestFailure>(grant);
    drop(refusal);
    assert!(matches!(
        error.failure(),
        AccountedErrorPublisherBuildFailure::AllocationWithRollback(
            MemoryGrantError::AccountingPoisoned {
                account: "accounted-error-test"
            }
        )
    ));
    assert_eq!(error.grant().size(), required);
    assert_eq!(account.accounted.load(Ordering::Acquire), required);
    assert_eq!(test_allocation_attempts(), 1);

    account.refuse_release.store(false, Ordering::Release);
    let mut grant = error.into_grant();
    grant
        .try_resize(0)
        .expect("caller can explicitly reconcile retained authority");
    assert_eq!(account.accounted.load(Ordering::Acquire), 0);
    let publisher = AccountedErrorPublisher::<TestFailure>::try_new(grant)
        .expect("reconciled grant retries construction");
    assert_eq!(account.accounted.load(Ordering::Acquire), required);
    drop(publisher);
    assert_eq!(account.accounted.load(Ordering::Acquire), 0);
}

#[test]
fn unpublished_publisher_drop_deallocates_before_releasing_its_grant() {
    reset_test_allocator_observation();
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let publisher = AccountedErrorPublisher::<TestFailure>::try_new(grant).expect("publisher");
    assert_eq!(manager.allocated(), required);
    assert_eq!(test_deallocation_count(), 0);

    drop(publisher);
    assert_eq!(test_deallocation_count(), 1);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn unpublished_publisher_extraction_deallocates_before_returning_retry_authority() {
    reset_test_allocator_observation();
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let account = RollbackAccount::new();
    let grant = MemoryGrant::new(
        Arc::clone(&account) as Arc<dyn GrantAccount>,
        0,
        MemoryRegion::ExecutionBuffers,
    );
    let publisher = AccountedErrorPublisher::<TestFailure>::try_new(grant).expect("publisher");
    assert_eq!(test_allocation_attempts(), 1);
    assert_eq!(test_deallocation_count(), 0);
    assert_eq!(account.accounted.load(Ordering::Acquire), required);

    account.refuse_release.store(true, Ordering::Release);
    let mut grant = publisher.into_unpublished_grant();
    assert_eq!(test_deallocation_count(), 1);
    assert_eq!(grant.size(), required);
    assert_eq!(account.accounted.load(Ordering::Acquire), required);

    assert!(matches!(
        grant.try_resize(0),
        Err(MemoryGrantError::AccountingPoisoned {
            account: "accounted-error-test"
        })
    ));
    assert_eq!(grant.size(), required);
    assert_eq!(account.accounted.load(Ordering::Acquire), required);

    account.refuse_release.store(false, Ordering::Release);
    grant
        .try_resize(0)
        .expect("the returned live grant remains explicit retry authority");
    assert_eq!(grant.size(), 0);
    assert_eq!(account.accounted.load(Ordering::Acquire), 0);
    drop(grant);
    assert_eq!(test_deallocation_count(), 1);
}

#[test]
fn unpublished_publisher_cleans_up_during_an_existing_unwind() {
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let publisher = AccountedErrorPublisher::<TestFailure>::try_new(grant).expect("publisher");

    let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _publisher = publisher;
        panic!("outer publisher unwind");
    }));
    assert!(outer.is_err());
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn publish_clone_inspect_and_format_do_not_allocate_another_shared_block() {
    reset_test_allocator_observation();
    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let publisher = AccountedErrorPublisher::<TestFailure>::try_new(grant).expect("publisher");
    assert_eq!(test_allocation_attempts(), 1);

    let refusal = refuse_next_test_allocation();
    let error = publisher.publish(TestFailure("published"));
    let cloned = error.clone();
    assert_eq!(
        error.inspect::<TestFailure, _>(|payload| payload.0),
        Some("published")
    );
    let mut display = StackFormatter::new();
    fmt::write(&mut display, format_args!("{error}")).expect("stack display");
    assert_eq!(display.as_str(), "published");
    let mut debug = StackFormatter::new();
    fmt::write(&mut debug, format_args!("{error:?}")).expect("stack debug");
    assert!(debug.as_str().contains("published"));
    assert_eq!(test_allocation_attempts(), 1);
    drop(refusal);
    drop(error);
    drop(cloned);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn clone_shares_identity_without_changing_accounting() {
    fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
    assert_clone_send_sync::<super::AccountedError>();

    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let error = AccountedErrorPublisher::try_new(grant)
        .expect("publisher")
        .publish(TestFailure("shared"));

    let cloned = error.clone();
    assert!(error.ptr_eq(&cloned));
    assert_eq!(manager.allocated(), required);

    drop(error);
    assert_eq!(manager.allocated(), required);
    drop(cloned);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn inline_guard_makes_handle_send_and_sync_for_a_send_but_not_sync_payload() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<super::AccountedError>();
    assert_send_sync::<AccountedErrorPublisher<SendNotSyncFailure>>();

    let required = AccountedErrorPublisher::<SendNotSyncFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let error = AccountedErrorPublisher::try_new(grant)
        .expect("publisher accepts Send payload")
        .publish(SendNotSyncFailure(Cell::new(0)));

    assert_eq!(
        error.inspect::<SendNotSyncFailure, _>(|payload| {
            payload.0.set(1);
            payload.0.get()
        }),
        Some(1)
    );

    let threads = (0..8)
        .map(|_| {
            let cloned = error.clone();
            std::thread::spawn(move || {
                cloned.inspect::<SendNotSyncFailure, _>(|payload| {
                    let next = payload.0.get() + 1;
                    payload.0.set(next);
                    next
                })
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        assert!(thread.join().expect("guarded inspection thread").is_some());
    }
    assert_eq!(
        error.inspect::<SendNotSyncFailure, _>(|payload| payload.0.get()),
        Some(9)
    );

    let thread = std::thread::spawn(move || error.to_string());
    assert_eq!(
        thread.join().expect("accounted error crosses threads"),
        "send-only payload 9"
    );
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn concurrent_clone_drop_keeps_payload_until_the_last_owner() {
    #[derive(Debug)]
    struct DropWitness(Arc<AtomicBool>);

    impl fmt::Display for DropWitness {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("drop witness")
        }
    }

    impl std::error::Error for DropWitness {}

    impl Drop for DropWitness {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    let required = AccountedErrorPublisher::<DropWitness>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let dropped = Arc::new(AtomicBool::new(false));
    let error = AccountedErrorPublisher::try_new(grant)
        .expect("publisher")
        .publish(DropWitness(Arc::clone(&dropped)));

    let threads = (0..16)
        .map(|_| {
            let clone = error.clone();
            std::thread::spawn(move || drop(clone))
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().expect("clone drop thread");
    }

    assert!(!dropped.load(Ordering::Acquire));
    assert_eq!(manager.allocated(), required);
    drop(error);
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(manager.allocated(), 0);
}

struct GrantLiveDrop {
    manager: Arc<BufferManager>,
    expected_bytes: usize,
    observed: Arc<AtomicBool>,
}

impl fmt::Debug for GrantLiveDrop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("grant-live drop witness")
    }
}

impl fmt::Display for GrantLiveDrop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("grant-live drop witness")
    }
}

impl std::error::Error for GrantLiveDrop {}

impl Drop for GrantLiveDrop {
    fn drop(&mut self) {
        assert_eq!(self.manager.allocated(), self.expected_bytes);
        self.observed.store(true, Ordering::Release);
    }
}

#[test]
fn payload_drop_observes_live_grant_before_normal_last_owner_release() {
    let required = AccountedErrorPublisher::<GrantLiveDrop>::required_bytes();
    let manager = exact_manager(required);
    let observed = Arc::new(AtomicBool::new(false));
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let error = AccountedErrorPublisher::try_new(grant)
        .expect("publisher")
        .publish(GrantLiveDrop {
            manager: Arc::clone(&manager),
            expected_bytes: required,
            observed: Arc::clone(&observed),
        });

    drop(error);
    assert!(observed.load(Ordering::Acquire));
    assert_eq!(manager.allocated(), 0);
}

struct HostileDrop {
    manager: Arc<BufferManager>,
    expected_bytes: usize,
    observed: Arc<AtomicBool>,
}

impl fmt::Debug for HostileDrop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("hostile drop witness")
    }
}

impl fmt::Display for HostileDrop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("hostile drop witness")
    }
}

impl std::error::Error for HostileDrop {}

impl Drop for HostileDrop {
    fn drop(&mut self) {
        assert_eq!(self.manager.allocated(), self.expected_bytes);
        self.observed.store(true, Ordering::Release);
        std::panic::panic_any(vec![13_u8, 21, 34, 55]);
    }
}

fn hostile_error(
    manager: &Arc<BufferManager>,
    observed: &Arc<AtomicBool>,
) -> super::AccountedError {
    let required = AccountedErrorPublisher::<HostileDrop>::required_bytes();
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    AccountedErrorPublisher::try_new(grant)
        .expect("publisher")
        .publish(HostileDrop {
            manager: Arc::clone(manager),
            expected_bytes: required,
            observed: Arc::clone(observed),
        })
}

#[test]
fn ordinary_hostile_drop_resumes_heap_panic_and_retains_in_place_authority() {
    let required = AccountedErrorPublisher::<HostileDrop>::required_bytes();
    let manager = exact_manager(required);
    let observed = Arc::new(AtomicBool::new(false));
    let error = hostile_error(&manager, &observed);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(error)))
        .expect_err("ordinary payload-destructor panic must retain its owner");
    let payload = panic
        .downcast::<Vec<u8>>()
        .expect("exact heap-bearing panic payload is resumed");
    assert_eq!(payload.as_slice(), &[13_u8, 21, 34, 55]);
    assert_eq!(manager.allocated(), required);
    drop(payload);
    assert!(observed.load(Ordering::Acquire));
    assert_eq!(manager.allocated(), required);
}

#[test]
fn existing_unwind_skips_hostile_payload_drop_and_keeps_in_place_authority() {
    let required = AccountedErrorPublisher::<HostileDrop>::required_bytes();
    let manager = exact_manager(required);
    let observed = Arc::new(AtomicBool::new(false));

    let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _error = hostile_error(&manager, &observed);
        panic!("original outer panic");
    }))
    .expect_err("outer panic survives accounted cleanup");
    assert_eq!(
        outer.downcast_ref::<&'static str>(),
        Some(&"original outer panic")
    );
    assert!(
        !observed.load(Ordering::Acquire),
        "payload code must not run during an existing unwind"
    );
    assert_eq!(manager.allocated(), required);
}

#[test]
fn typed_inspection_recovers_after_unwind_without_extracting_the_payload() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<super::AccountedError>();

    let required = AccountedErrorPublisher::<TestFailure>::required_bytes();
    let manager = exact_manager(required);
    let grant = manager
        .try_allocate(0, MemoryRegion::ExecutionBuffers)
        .expect("dedicated zero grant");
    let error = AccountedErrorPublisher::try_new(grant)
        .expect("publisher")
        .publish(TestFailure("typed payload"));

    assert!(error.is::<TestFailure>());
    assert!(!error.is::<std::io::Error>());
    assert_eq!(
        error.inspect::<TestFailure, _>(|source| source.0),
        Some("typed payload")
    );
    assert_eq!(
        error.inspect::<std::io::Error, _>(|source| source.kind()),
        None
    );

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        error.inspect::<TestFailure, _>(|_| panic!("inspector unwind"));
    }));
    assert!(unwind.is_err());
    assert_eq!(
        error.inspect::<TestFailure, _>(|source| source.0),
        Some("typed payload")
    );
    assert_eq!(error.to_string(), "typed payload");
    assert!(format!("{error:?}").contains("typed payload"));

    drop(error);
    assert_eq!(manager.allocated(), 0);
}
