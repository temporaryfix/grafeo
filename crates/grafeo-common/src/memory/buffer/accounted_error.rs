//! Pre-admitted shared ownership for move-only error values.
//!
//! Accounting covers the shared control block itself. Allocations already
//! owned by the payload require their own authority.

#![allow(
    unsafe_code,
    reason = "the exact fallible allocation and intrusive reference count are the capability boundary"
)]

use super::{MemoryGrant, MemoryGrantError};
use std::alloc::{Layout, alloc, dealloc};
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};

#[cfg(test)]
std::thread_local! {
    static TEST_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_DEALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_REFUSE_NEXT_ALLOCATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Match `Arc`'s fail-fast refcount ceiling. Staying below `isize::MAX`
// prevents pointer-offset and counter wraparound even under hostile cloning.
const MAX_REFCOUNT: usize = isize::MAX as usize;
const INLINE_LOCK_SPINS: usize = 64;

/// Allocation-free mutual exclusion for one already-accounted inline value.
///
/// This lock has no poisoning, heap-backed waiter queue, or platform-mutex
/// initialization. Contention spins briefly and then yields the current OS
/// thread; the state and protected value remain wholly inline.
pub(super) struct InlineLock<T> {
    held: AtomicBool,
    value: UnsafeCell<T>,
}

impl<T> InlineLock<T> {
    /// Creates an unlocked guard around an inline value.
    pub(super) const fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Acquires the inline guard without allocating or parking a waiter.
    pub(super) fn lock(&self) -> InlineLockGuard<'_, T> {
        let mut spins = 0;
        loop {
            if self
                .held
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return InlineLockGuard {
                    lock: self,
                    _not_send_sync: PhantomData,
                };
            }

            if spins < INLINE_LOCK_SPINS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                spins = 0;
                std::thread::yield_now();
            }
        }
    }
}

// SAFETY: moving the lock moves its protected value, which is sound exactly
// when `T: Send`. No reference into `value` exists without a live guard.
unsafe impl<T: Send> Send for InlineLock<T> {}

// SAFETY: every shared access to `value` is serialized by the Acquire/Release
// `held` protocol. Moving a protected `T` between locking threads requires
// `T: Send`; `T: Sync` is unnecessary because no unguarded `&T` escapes.
unsafe impl<T: Send> Sync for InlineLock<T> {}

/// RAII access token that releases an [`InlineLock`] on every unwind path.
pub(super) struct InlineLockGuard<'a, T> {
    lock: &'a InlineLock<T>,
    // An acquired error-path guard has no reason to cross a thread boundary;
    // keeping it thread-local also prevents sharing its protected `&T`.
    _not_send_sync: PhantomData<std::rc::Rc<()>>,
}

impl<T> Deref for InlineLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: this guard exists only after changing `held` from false to
        // true with Acquire ordering. The atomic protocol permits exactly one
        // live guard, and the returned reference cannot outlive that guard.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for InlineLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: the unique mutable guard borrow plus the single-guard atomic
        // invariant excludes every other access to the protected value.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for InlineLockGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.held.store(false, Ordering::Release);
    }
}

struct PayloadSlot<T> {
    initialized: bool,
    value: MaybeUninit<T>,
}

impl<T> PayloadSlot<T> {
    const fn empty() -> Self {
        Self {
            initialized: false,
            value: MaybeUninit::uninit(),
        }
    }

    fn initialize(&mut self, value: T) {
        // The unique publisher reaches this once from an empty slot. Even if
        // that invariant were violated, `MaybeUninit::write` remains memory
        // safe (it would conservatively leak rather than double-drop the old
        // value); no unaccounted allocation occurs here.
        self.value.write(value);
        self.initialized = true;
    }

    fn get(&self) -> Option<&T> {
        if !self.initialized {
            return None;
        }
        // SAFETY: `initialized` is set only after `MaybeUninit::write` and is
        // cleared only after successful in-place destruction. A panicking
        // destructor leaves it true but immediately makes the entire block
        // unreachable, so no later safe call can observe a partial value.
        Some(unsafe { self.value.assume_init_ref() })
    }

    unsafe fn drop_in_place(&mut self) {
        // SAFETY: callers prove this reachable slot is initialized and uniquely
        // owned. If `T::drop` panics, the following flag clear is not reached
        // and the partially destroyed storage is never accessed again.
        unsafe { self.value.assume_init_drop() };
        self.initialized = false;
    }
}

fn allocate_shared_block(layout: Layout) -> *mut u8 {
    #[cfg(test)]
    {
        let _ = TEST_ALLOCATIONS.try_with(|attempts| {
            attempts.set(attempts.get().saturating_add(1));
        });
        let refuse = TEST_REFUSE_NEXT_ALLOCATION
            .try_with(|refuse| refuse.replace(false))
            .unwrap_or(false);
        if refuse {
            return std::ptr::null_mut();
        }
    }

    // SAFETY: callers provide a valid, non-zero `Layout`. Returning the raw
    // pointer transfers no initialized-value invariant; the caller checks for
    // null before writing and later deallocates with this same layout.
    unsafe { alloc(layout) }
}

unsafe fn deallocate_shared_block<T>(ptr: NonNull<SharedBlock<T>>) {
    #[cfg(test)]
    {
        let _ = TEST_DEALLOCATIONS.try_with(|deallocations| {
            deallocations.set(deallocations.get().saturating_add(1));
        });
    }

    // SAFETY: every caller proves that `ptr` came from `allocate_shared_block`
    // with this exact monomorphized layout and that no initialized field still
    // needs destruction.
    unsafe { dealloc(ptr.as_ptr().cast::<u8>(), Layout::new::<SharedBlock<T>>()) };
}

#[cfg(test)]
pub(super) struct TestAllocationFailureGuard {
    _not_send: PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl Drop for TestAllocationFailureGuard {
    fn drop(&mut self) {
        let _ = TEST_REFUSE_NEXT_ALLOCATION.try_with(|refuse| refuse.set(false));
    }
}

#[cfg(test)]
pub(super) fn reset_test_allocator_observation() {
    TEST_ALLOCATIONS.with(|attempts| attempts.set(0));
    TEST_DEALLOCATIONS.with(|deallocations| deallocations.set(0));
    TEST_REFUSE_NEXT_ALLOCATION.with(|refuse| refuse.set(false));
}

#[cfg(test)]
pub(super) fn test_allocation_attempts() -> usize {
    TEST_ALLOCATIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn test_deallocation_count() -> usize {
    TEST_DEALLOCATIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn refuse_next_test_allocation() -> TestAllocationFailureGuard {
    TEST_REFUSE_NEXT_ALLOCATION.with(|refuse| {
        assert!(
            !refuse.replace(true),
            "allocator-refusal hook is already armed"
        );
    });
    TestAllocationFailureGuard {
        _not_send: PhantomData,
    }
}

/// Why construction of an [`AccountedErrorPublisher`] failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AccountedErrorPublisherBuildFailure {
    /// The caller supplied a grant that was not the required dedicated zero child.
    #[error("accounted error publication requires a dedicated zero-byte grant, got {bytes} bytes")]
    NonZeroGrant {
        /// Bytes carried by the rejected grant.
        bytes: usize,
    },
    /// The grant's backing account denied exact control-block admission.
    #[error("accounted error control-block admission failed: {0}")]
    Admission(#[source] MemoryGrantError),
    /// The global allocator refused the already-admitted control block.
    #[error("allocator refused the accounted error control block")]
    Allocation,
    /// Allocation failed and the admitted grant could not be reconciled to zero.
    #[error("allocator refused the accounted error control block; grant rollback failed: {0}")]
    AllocationWithRollback(#[source] MemoryGrantError),
}

/// Failed publisher construction together with the undetached grant authority.
#[derive(Debug)]
#[must_use = "the returned grant remains the caller's retry or cleanup authority"]
pub struct AccountedErrorPublisherBuildError {
    failure: AccountedErrorPublisherBuildFailure,
    grant: MemoryGrant,
}

impl AccountedErrorPublisherBuildError {
    fn new(failure: AccountedErrorPublisherBuildFailure, grant: MemoryGrant) -> Self {
        Self { failure, grant }
    }

    /// Returns the structured construction failure.
    #[must_use]
    pub const fn failure(&self) -> &AccountedErrorPublisherBuildFailure {
        &self.failure
    }

    /// Returns the retained grant without detaching it from this failure.
    #[must_use]
    pub fn grant(&self) -> &MemoryGrant {
        &self.grant
    }

    /// Recovers the complete grant for explicit reconciliation or retry.
    #[must_use]
    pub fn into_grant(self) -> MemoryGrant {
        self.grant
    }
}

impl fmt::Display for AccountedErrorPublisherBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.failure, formatter)
    }
}

impl Error for AccountedErrorPublisherBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.failure.source()
    }
}

struct SharedBlock<T> {
    strong: AtomicUsize,
    value: InlineLock<PayloadSlot<T>>,
    grant: ManuallyDrop<MemoryGrant>,
}

struct AccountedErrorVTable {
    clone_strong: unsafe fn(NonNull<()>),
    drop_strong: unsafe fn(NonNull<()>),
    granted_bytes: unsafe fn(NonNull<()>) -> usize,
    type_id: fn() -> TypeId,
    display: unsafe fn(NonNull<()>, &mut fmt::Formatter<'_>) -> fmt::Result,
    debug: unsafe fn(NonNull<()>, &mut fmt::Formatter<'_>) -> fmt::Result,
}

struct AccountedErrorVTableFor<T>(PhantomData<fn() -> T>);

impl<T> AccountedErrorVTableFor<T>
where
    T: Error + Send + 'static,
{
    const VALUE: AccountedErrorVTable = AccountedErrorVTable {
        clone_strong: clone_strong::<T>,
        drop_strong: drop_strong::<T>,
        granted_bytes: granted_bytes::<T>,
        type_id: TypeId::of::<T>,
        display: display_value::<T>,
        debug: debug_value::<T>,
    };
}

/// A one-shot, pre-admitted publication slot for one move-only error.
///
/// Construction requires a dedicated zero-byte grant. It admits and allocates
/// the exact shared block before any caller-controlled operation can fail.
/// Publishing consumes the slot and performs no internal allocation. The
/// block uses an inline atomic guard rather than a heap-backed or lazily
/// allocated platform mutex; cloning, inspection, and formatting likewise add
/// no internal allocations. This grant covers the shared block itself; any
/// allocation already owned by `T` must arrive with its own authority and any
/// allocation performed by `T` or its formatting remains `T`'s responsibility.
///
/// The publisher is deliberately move-only:
///
/// ```compile_fail
/// use grafeo_common::memory::buffer::AccountedErrorPublisher;
///
/// fn duplicate<T>(publisher: AccountedErrorPublisher<T>)
/// where
///     T: std::error::Error + Send + 'static,
/// {
///     let _: AccountedErrorPublisher<T> = publisher.clone();
/// }
/// ```
pub struct AccountedErrorPublisher<T>
where
    T: Error + Send + 'static,
{
    ptr: NonNull<SharedBlock<T>>,
    marker: PhantomData<SharedBlock<T>>,
}

impl<T> AccountedErrorPublisher<T>
where
    T: Error + Send + 'static,
{
    /// Exact number of bytes required by this publisher's shared block.
    #[must_use]
    pub const fn required_bytes() -> usize {
        Layout::new::<SharedBlock<T>>().size()
    }

    /// Allocates one exact publication slot after admitting its complete layout.
    ///
    /// # Errors
    ///
    /// Returns the unchanged or reconciled grant when it is non-zero, its
    /// backing account denies admission, or the global allocator refuses the
    /// admitted block.
    pub fn try_new(mut grant: MemoryGrant) -> Result<Self, AccountedErrorPublisherBuildError> {
        if !grant.is_empty() {
            return Err(AccountedErrorPublisherBuildError::new(
                AccountedErrorPublisherBuildFailure::NonZeroGrant {
                    bytes: grant.size(),
                },
                grant,
            ));
        }

        let layout = Layout::new::<SharedBlock<T>>();
        if let Err(error) = grant.try_resize(layout.size()) {
            return Err(AccountedErrorPublisherBuildError::new(
                AccountedErrorPublisherBuildFailure::Admission(error),
                grant,
            ));
        }

        // `layout` is the exact non-zero layout of `SharedBlock<T>`. A null
        // result is handled without dereferencing it or invoking the
        // infallible allocation-error handler.
        let raw = allocate_shared_block(layout);
        let Some(ptr) = NonNull::new(raw.cast::<SharedBlock<T>>()) else {
            let failure = match grant.try_resize(0) {
                Ok(()) => AccountedErrorPublisherBuildFailure::Allocation,
                Err(rollback) => {
                    AccountedErrorPublisherBuildFailure::AllocationWithRollback(rollback)
                }
            };
            return Err(AccountedErrorPublisherBuildError::new(failure, grant));
        };

        // SAFETY: `ptr` came from `alloc(Layout::new::<SharedBlock<T>>())`, is
        // non-null and correctly aligned, and no initialized value currently
        // occupies the allocation. This write establishes its sole strong
        // owner and moves the admitted grant beside the physical block.
        unsafe {
            ptr.as_ptr().write(SharedBlock {
                strong: AtomicUsize::new(1),
                value: InlineLock::new(PayloadSlot::empty()),
                grant: ManuallyDrop::new(grant),
            });
        }

        Ok(Self {
            ptr,
            marker: PhantomData,
        })
    }

    /// Bytes currently held by the block's dedicated grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        // SAFETY: a live publisher owns the initial strong reference, so its
        // block is initialized and cannot enter last-owner destruction.
        unsafe { granted_bytes::<T>(self.ptr.cast()) }
    }

    /// Dismantles an unpublished slot and recovers its still-accounted grant.
    ///
    /// The empty shared block is physically deallocated before this method
    /// returns. The returned grant remains live at its admitted size: callers
    /// that need an observable completion result must explicitly resize,
    /// merge, or otherwise reconcile it instead of relying on [`Drop`].
    #[must_use = "the recovered grant remains the caller's release or transfer authority"]
    pub fn into_unpublished_grant(self) -> MemoryGrant {
        let this = ManuallyDrop::new(self);

        // SAFETY: consuming the unique publisher proves that the block is
        // initialized, unpublished, and has no erased handles or live guards.
        unsafe { take_unpublished_grant::<T>(this.ptr) }
    }

    /// Publishes `value` into the already allocated slot without allocating.
    #[must_use]
    pub fn publish(self, value: T) -> AccountedError {
        let this = ManuallyDrop::new(self);

        // SAFETY: consuming the unique publisher proves the block is live and
        // unpublished. No AccountedError handle can exist before this write,
        // so the slot is exclusively accessible here.
        let block = unsafe { this.ptr.as_ref() };
        let mut slot = block.value.lock();
        slot.initialize(value);
        drop(slot);

        AccountedError {
            ptr: this.ptr.cast(),
            vtable: &AccountedErrorVTableFor::<T>::VALUE,
        }
    }
}

impl<T> Drop for AccountedErrorPublisher<T>
where
    T: Error + Send + 'static,
{
    fn drop(&mut self) {
        // SAFETY: the unique publisher owns the initialized block and its sole
        // strong reference, and its payload slot is still empty. The consuming
        // `publish` path suppresses this destructor with `ManuallyDrop`.
        unsafe { drop_unpublished::<T>(self.ptr) };
    }
}

// SAFETY: a publisher uniquely owns its block; moving it between threads moves
// no published payload, and its admitted grant is itself safe to move. `T:
// Send` also makes a later consuming publication valid on that thread.
unsafe impl<T> Send for AccountedErrorPublisher<T> where T: Error + Send + 'static {}

// SAFETY: shared publisher access can only observe immutable grant size. The
// unique consuming `publish` operation cannot overlap a shared borrow, and the
// publisher is deliberately non-Clone.
unsafe impl<T> Sync for AccountedErrorPublisher<T> where T: Error + Send + 'static {}

/// Cloneable type-erased ownership of one pre-admitted error value.
///
/// This initial surface deliberately exposes neither raw parts nor extraction;
/// subsequent APIs only borrow a typed value while its internal lock is held.
/// Because that lock cannot outlive a method call, [`Error::source`] returns
/// `None`; use [`Self::is`] and [`Self::inspect`] for concrete-source access.
/// Inspection callbacks and payload `Display`/`Debug` implementations must not
/// re-enter or wait on another handle to the same owner: the inline guard is
/// deliberately non-reentrant.
/// Last-owner cleanup during an existing unwind invokes no payload code and
/// leaks the initialized block with its grant still in place. Outside an
/// unwind, one payload-destructor panic is resumed to its caller while the
/// partially destroyed block and its in-place grant remain unreachable and
/// charged. Normal cleanup destroys `T`, marks its slot empty, deallocates the
/// block, and only then releases its grant. Rust process aborts initiated
/// wholly inside `T`'s own nested destructor chain cannot be intercepted by an
/// outer `catch_unwind`.
pub struct AccountedError {
    ptr: NonNull<()>,
    vtable: &'static AccountedErrorVTable,
}

impl AccountedError {
    /// Bytes retained by the shared block's dedicated grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        // SAFETY: every AccountedError is created by a consuming publication
        // and owns one live strong reference governed by this exact vtable.
        unsafe { (self.vtable.granted_bytes)(self.ptr) }
    }

    /// Returns whether both handles share the same physical error owner.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }

    /// Returns whether the retained value has concrete type `T`.
    #[must_use]
    pub fn is<T>(&self) -> bool
    where
        T: Error + Send + 'static,
    {
        (self.vtable.type_id)() == TypeId::of::<T>()
    }

    /// Borrows the retained value as `T` for the duration of `inspect`.
    ///
    /// The payload remains behind its inline guard and cannot be extracted. The
    /// callback's return type is independent of the temporary borrow, so a
    /// reference into the payload cannot escape this method. A callback panic
    /// releases the guard during unwinding, so later inspection and formatting
    /// recover the protected value without poison state. The guard is
    /// deliberately non-reentrant: `inspect` callbacks must not inspect or
    /// format another handle to this same owner before returning.
    ///
    /// A reference into the protected value cannot escape:
    ///
    /// ```compile_fail
    /// use grafeo_common::memory::buffer::AccountedError;
    ///
    /// fn payload_reference(error: &AccountedError) -> Option<&std::io::Error> {
    ///     error.inspect::<std::io::Error, _>(|source| source)
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Propagates a panic from `inspect`. The inline guard unlocks during the
    /// unwind, so later operations can recover the still-authorized payload.
    pub fn inspect<T, R>(&self, inspect: impl FnOnce(&T) -> R) -> Option<R>
    where
        T: Error + Send + 'static,
    {
        if !self.is::<T>() {
            return None;
        }

        // SAFETY: equality of `TypeId` for `'static` types proves the vtable
        // and allocation were instantiated for this exact `T`. This live
        // handle keeps the block initialized for the lock's full lifetime.
        let block = unsafe { &*self.ptr.cast::<SharedBlock<T>>().as_ptr() };
        let slot = block.value.lock();
        Some(inspect(slot.get()?))
    }
}

impl fmt::Display for AccountedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: the publisher installed a vtable monomorphized for the
        // allocation's concrete payload type, and this live handle keeps that
        // allocation initialized while formatting holds its inline guard.
        unsafe { (self.vtable.display)(self.ptr, formatter) }
    }
}

impl fmt::Debug for AccountedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: the publisher installed a vtable monomorphized for the
        // allocation's concrete payload type, and this live handle keeps that
        // allocation initialized while formatting holds its inline guard.
        unsafe { (self.vtable.debug)(self.ptr, formatter) }
    }
}

impl Error for AccountedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl Clone for AccountedError {
    fn clone(&self) -> Self {
        // SAFETY: this live handle proves the block has at least one strong
        // reference. The installed vtable matches its allocation's concrete
        // type and performs only the atomic strong-count increment.
        unsafe { (self.vtable.clone_strong)(self.ptr) };
        Self {
            ptr: self.ptr,
            vtable: self.vtable,
        }
    }
}

impl Drop for AccountedError {
    fn drop(&mut self) {
        // SAFETY: this handle owns one strong reference paired with the
        // monomorphized vtable installed by its publisher.
        unsafe { (self.vtable.drop_strong)(self.ptr) };
    }
}

// SAFETY: an AccountedError can only be constructed from a publisher whose
// payload is `Send`. All access to that payload is serialized by the block's
// inline guard; reference counting and grant-size observation are atomic. The erased
// pointer is never exposed or cast except through its matching static vtable.
unsafe impl Send for AccountedError {}

// SAFETY: the same construction invariant as `Send` applies. Shared handles
// only mutate the atomic strong count or lock the payload guard, so concurrent
// observation and destruction cannot race with unprotected payload access.
unsafe impl Sync for AccountedError {}

unsafe fn clone_strong<T>(ptr: NonNull<()>)
where
    T: Error + Send + 'static,
{
    // SAFETY: callers use the vtable installed for the same `T`; their live
    // source handle keeps the initialized block alive throughout this atomic
    // increment.
    let block = unsafe { &*ptr.cast::<SharedBlock<T>>().as_ptr() };
    let previous = block.strong.fetch_add(1, Ordering::Relaxed);
    if previous > MAX_REFCOUNT {
        // This is the standard Arc-grade defense against hostile refcount
        // overflow. It is unrelated to allocator refusal, which remains a
        // normal fallible construction result and never reaches this path.
        std::process::abort();
    }
}

unsafe fn granted_bytes<T>(ptr: NonNull<()>) -> usize
where
    T: Error + Send + 'static,
{
    // SAFETY: callers use the vtable installed for the same `T`; a live strong
    // reference keeps the initialized block and its grant alive for this load.
    let block = unsafe { &*ptr.cast::<SharedBlock<T>>().as_ptr() };
    block.grant.size()
}

unsafe fn display_value<T>(ptr: NonNull<()>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
where
    T: Error + Send + 'static,
{
    // SAFETY: callers use the vtable installed for the same `T`; a live strong
    // reference keeps the initialized block alive while the inline guard lends
    // the payload exclusively to its formatter.
    let block = unsafe { &*ptr.cast::<SharedBlock<T>>().as_ptr() };
    let slot = block.value.lock();
    slot.get()
        .map_or(Err(fmt::Error), |value| fmt::Display::fmt(value, formatter))
}

unsafe fn debug_value<T>(ptr: NonNull<()>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
where
    T: Error + Send + 'static,
{
    // SAFETY: callers use the vtable installed for the same `T`; a live strong
    // reference keeps the initialized block alive while the inline guard lends
    // the payload exclusively to its formatter.
    let block = unsafe { &*ptr.cast::<SharedBlock<T>>().as_ptr() };
    let slot = block.value.lock();
    slot.get()
        .map_or(Err(fmt::Error), |value| fmt::Debug::fmt(value, formatter))
}

unsafe fn drop_unpublished<T>(ptr: NonNull<SharedBlock<T>>)
where
    T: Error + Send + 'static,
{
    // SAFETY: the unique unpublished publisher satisfies the extraction
    // invariant. Deallocation deliberately precedes even best-effort Drop
    // release during an unwind.
    let grant = unsafe { take_unpublished_grant::<T>(ptr) };
    drop(grant);
}

unsafe fn take_unpublished_grant<T>(ptr: NonNull<SharedBlock<T>>) -> MemoryGrant
where
    T: Error + Send + 'static,
{
    // SAFETY: only a never-published unique publisher reaches this function.
    // Its block is initialized, its slot is empty, and no erased handle or
    // guard can exist. Moving out the grant leaves no field requiring Drop.
    let grant = unsafe { ManuallyDrop::take(&mut (*ptr.as_ptr()).grant) };

    // SAFETY: `ptr` came from `allocate_shared_block` with this exact layout.
    // The empty payload slot, inline lock, and atomic counter require no
    // destruction, and the grant has already moved out.
    unsafe { deallocate_shared_block(ptr) };
    grant
}

unsafe fn drop_strong<T>(ptr: NonNull<()>)
where
    T: Error + Send + 'static,
{
    // SAFETY: callers use the vtable installed for the same `T`; the block is
    // initialized and this call consumes exactly one live strong reference.
    let block = unsafe { &*ptr.cast::<SharedBlock<T>>().as_ptr() };
    if block.strong.fetch_sub(1, Ordering::Release) != 1 {
        return;
    }
    fence(Ordering::Acquire);

    if std::thread::panicking() {
        // Calling arbitrary `T::drop` during an existing unwind could start a
        // second panic before this boundary can regain control. Fail closed
        // before touching any field: the initialized allocation becomes
        // unreachable with its grant still physically in place, so cleanup
        // cannot double-panic and accounting cannot be released early.
        return;
    }

    let raw = ptr.cast::<SharedBlock<T>>().as_ptr();

    // SAFETY: the final strong reference gives exclusive access, so no guard
    // can coexist with this mutable slot reference. The slot contains the one
    // payload installed before an AccountedError handle became visible.
    let slot = unsafe { &mut *(*raw).value.value.get() };
    if slot.initialized {
        let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the published slot owns one initialized `T`, and the
            // final strong reference makes this its sole destructor attempt.
            unsafe { slot.drop_in_place() };
        }));
        if let Err(panic) = dropped {
            // The partially destroyed payload, allocation, and grant all stay
            // in place and become unreachable. Resuming transfers ownership of
            // even a heap-bearing panic payload to the caller rather than
            // leaking unaccounted panic storage. A nested panic wholly inside
            // `T`'s destructor chain may still process-abort before this catch
            // regains control; generic Rust cleanup cannot intercept it.
            std::panic::resume_unwind(panic);
        }
    }
    // SAFETY: payload destruction completed and the slot is now empty. Moving
    // out the in-place grant leaves no block field requiring destruction.
    let grant = unsafe { ManuallyDrop::take(&mut (*raw).grant) };

    // SAFETY: the payload is destroyed, the slot is empty, the grant has moved
    // out, no strong references remain, and `ptr` was allocated with this
    // exact layout. Deallocation therefore precedes grant release.
    unsafe { deallocate_shared_block(ptr.cast::<SharedBlock<T>>()) };
    drop(grant);
}
