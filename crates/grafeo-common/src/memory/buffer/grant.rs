//! Memory grant RAII wrapper for automatic resource release.

use super::region::MemoryRegion;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

/// Scope whose resident-memory limit denied a grant operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryLimitScope {
    /// The database-wide buffer-manager hard limit.
    Global,
    /// The dynamic fair share assigned to one active query.
    Query,
}

/// Structured failure returned by fallible resident-memory grants.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryGrantError {
    /// Checked byte arithmetic overflowed before accounting or allocation.
    #[error(
        "resident-memory accounting overflow: {current_bytes} existing bytes plus {additional_bytes} requested bytes"
    )]
    ArithmeticOverflow {
        /// Bytes already accounted at the failing scope.
        current_bytes: usize,
        /// Additional bytes requested by the operation.
        additional_bytes: usize,
    },
    /// A global or per-query limit denied the requested total.
    #[error(
        "{scope:?} resident-memory limit exceeded: requested {requested_bytes} bytes, limit {limit_bytes} bytes"
    )]
    LimitExceeded {
        /// Limit that denied the request.
        scope: MemoryLimitScope,
        /// Total bytes the scope would own after success.
        requested_bytes: usize,
        /// Current limit for this scope.
        limit_bytes: usize,
    },
    /// An internal account implementation denied growth without exposing a
    /// public limit.
    #[error("resident-memory growth of {additional_bytes} bytes was denied by its account")]
    Denied {
        /// Additional bytes requested.
        additional_bytes: usize,
    },
    /// The monotonic count of live query pools cannot be incremented.
    #[error("active query-memory-pool count exhausted")]
    QueryPoolCountExhausted,
    /// A release exceeded the bytes held by an internal account.
    #[error(
        "resident-memory account {account} underflow: {accounted_bytes} bytes accounted, {release_bytes} bytes released"
    )]
    AccountingUnderflow {
        /// Internal account whose checked subtraction failed.
        account: &'static str,
        /// Bytes present before the rejected release.
        accounted_bytes: usize,
        /// Bytes the release attempted to subtract.
        release_bytes: usize,
    },
    /// An internal multi-account transition could not be rolled back safely.
    #[error("resident-memory account {account} is poisoned after an incomplete transition")]
    AccountingPoisoned {
        /// Internal account that must reject further transitions.
        account: &'static str,
    },
}

/// Private accounting capability carried only by RAII grants.
///
/// Keeping this trait crate-private prevents downstream callers from forging
/// tokenless allocation or release transitions against public managers.
pub(crate) trait GrantAccount: Send + Sync {
    /// Releases accounted memory through checked, ordered transitions.
    fn release_accounted(&self, size: usize, region: MemoryRegion) -> Result<(), MemoryGrantError>;

    /// Reserves additional accounted bytes for grant growth.
    ///
    /// # Errors
    ///
    /// Returns the structured allocation denial supplied by the account.
    fn try_reserve_growth(&self, size: usize, region: MemoryRegion)
    -> Result<(), MemoryGrantError>;

    /// Whether a grant may be reduced to an untracked byte count.
    ///
    /// Releasers with nested accounts should return `false`: dropping their
    /// last strong handle while bytes remain accounted would make those bytes
    /// impossible to release. Accounts must opt in explicitly when some other
    /// protocol can safely assume responsibility for the bytes.
    fn allows_untracked_consumption(&self) -> bool {
        false
    }
}

/// RAII wrapper for memory allocations.
///
/// Automatically releases memory back to the `BufferManager` when dropped.
/// Use [`Self::try_consume`] only when the crate-private backing account
/// explicitly permits transfer to another accounting protocol.
///
/// The infallible tokenless transfer method is unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_common::memory::buffer::MemoryGrant;
/// let _ = MemoryGrant::consume;
/// ```
pub struct MemoryGrant {
    /// Reference to the releaser (BufferManager).
    releaser: Arc<dyn GrantAccount>,
    /// Size of this grant in bytes.
    size: AtomicUsize,
    /// Memory region for this grant.
    region: MemoryRegion,
    /// Whether this grant has been consumed (transferred).
    consumed: bool,
}

/// RAII ownership token for accounted bytes detached from a [`MemoryGrant`].
///
/// The token retains the private account capability, size, and region. It can
/// be reattached without changing accounting, or released by explicit
/// [`Self::release`] or ordinary `Drop`. This is the safe replacement for
/// reducing a managed grant to a tokenless byte count.
#[must_use = "dropping a detached grant releases its accounted bytes"]
pub struct DetachedMemoryGrant {
    releaser: Arc<dyn GrantAccount>,
    size: usize,
    region: MemoryRegion,
    active: bool,
}

impl DetachedMemoryGrant {
    /// Returns the number of accounted bytes owned by this token.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the memory region owned by this token.
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        self.region
    }

    /// Reattaches this token to a resizable memory grant without changing
    /// either the byte count or its backing account.
    #[must_use]
    pub fn reattach(mut self) -> MemoryGrant {
        let grant = MemoryGrant::new(Arc::clone(&self.releaser), self.size, self.region);
        self.active = false;
        grant
    }

    /// Explicitly ends this token's lifetime, exactly like ordinary `Drop`.
    ///
    /// This is not a fallible completion signal: an already-poisoned internal
    /// account remains fail-closed and conservatively accounted because `Drop`
    /// cannot return an error. Call [`Self::reattach`] followed by
    /// [`MemoryGrant::try_resize`] to zero when the caller must observe an
    /// explicit reconciliation result.
    pub fn release(self) {
        drop(self);
    }

    fn try_into_untracked_bytes(mut self) -> Result<usize, Self> {
        if !self.releaser.allows_untracked_consumption() {
            return Err(self);
        }
        self.active = false;
        Ok(self.size)
    }
}

impl Drop for DetachedMemoryGrant {
    fn drop(&mut self) {
        if self.active && self.size != 0 {
            let _ = self.releaser.release_accounted(self.size, self.region);
        }
    }
}

impl std::fmt::Debug for DetachedMemoryGrant {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DetachedMemoryGrant")
            .field("size", &self.size)
            .field("region", &self.region)
            .field("active", &self.active)
            .finish()
    }
}

/// Opaque RAII transfer token for an entire [`CompositeGrant`].
///
/// Construction preflights the checked total before detaching any child. The
/// collection can be reattached or released as a unit, so callers never need
/// to reconstruct its private account tokens.
#[derive(Debug)]
#[must_use = "dropping a detached composite releases every accounted grant"]
pub struct DetachedCompositeGrant {
    grants: Vec<DetachedMemoryGrant>,
    total_size: usize,
}

impl DetachedCompositeGrant {
    /// Returns the preflighted total byte count.
    #[must_use]
    pub const fn total_size(&self) -> usize {
        self.total_size
    }

    /// Returns the number of detached child grants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Returns whether this token owns no child grants.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }

    /// Reattaches every child without changing any backing account.
    #[must_use]
    pub fn reattach(self) -> CompositeGrant {
        CompositeGrant {
            grants: self
                .grants
                .into_iter()
                .map(DetachedMemoryGrant::reattach)
                .collect(),
        }
    }

    /// Explicitly ends every child token's lifetime, exactly like ordinary
    /// `Drop`.
    ///
    /// This is best-effort reconciliation rather than a fallible completion
    /// signal. No atomic fallible composite-release operation is available;
    /// keep individual detached grants when each release must be handled
    /// separately.
    pub fn release(self) {
        drop(self);
    }
}

impl MemoryGrant {
    /// Creates a new memory grant.
    pub(crate) fn new(releaser: Arc<dyn GrantAccount>, size: usize, region: MemoryRegion) -> Self {
        Self {
            releaser,
            size: AtomicUsize::new(size),
            region,
            consumed: false,
        }
    }

    /// Returns the size of this grant in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Returns the memory region of this grant.
    #[must_use]
    pub fn region(&self) -> MemoryRegion {
        self.region
    }

    /// Attempts to resize the grant.
    ///
    /// Returns `true` if the resize succeeded, `false` if more memory
    /// could not be allocated.
    pub fn resize(&mut self, new_size: usize) -> bool {
        self.try_resize(new_size).is_ok()
    }

    /// Attempts to resize the grant with a structured denial reason.
    ///
    /// A failed growth leaves both the grant size and every backing account
    /// unchanged. Shrinking immediately releases capacity and fails closed if
    /// an internal accounting invariant was already poisoned.
    ///
    /// # Errors
    ///
    /// Returns the global, query-local, arithmetic, or internal-account failure
    /// that denied growth.
    pub fn try_resize(&mut self, new_size: usize) -> Result<(), MemoryGrantError> {
        let current = self.size.load(Ordering::Relaxed);

        match new_size.cmp(&current) {
            std::cmp::Ordering::Greater => {
                // Need more memory - try to allocate the difference
                let diff = new_size - current;
                self.releaser.try_reserve_growth(diff, self.region)?;
                self.size.store(new_size, Ordering::Release);
                Ok(())
            }
            std::cmp::Ordering::Less => {
                // Releasing memory
                let diff = current - new_size;
                self.releaser.release_accounted(diff, self.region)?;
                self.size.store(new_size, Ordering::Release);
                Ok(())
            }
            std::cmp::Ordering::Equal => Ok(()),
        }
    }

    /// Splits off a portion of this grant into a new grant.
    ///
    /// Returns `None` if the requested amount exceeds the current size.
    pub fn split(&mut self, amount: usize) -> Option<MemoryGrant> {
        let current = self.size.load(Ordering::Relaxed);
        if amount > current {
            return None;
        }

        self.size.store(current - amount, Ordering::Relaxed);
        Some(MemoryGrant {
            releaser: Arc::clone(&self.releaser),
            size: AtomicUsize::new(amount),
            region: self.region,
            consumed: false,
        })
    }

    /// Attempts to merge another grant into this one.
    ///
    /// Grants are transferable only when they cover the same region and are
    /// backed by the exact same releaser. This prevents moving query-local
    /// capacity into a grant whose drop would release a different account.
    /// A rejected merge returns the other grant unchanged.
    ///
    /// # Errors
    ///
    /// Returns the unchanged other grant when the region or releaser differs,
    /// or when the combined size cannot be represented by `usize`.
    pub fn try_merge(&mut self, mut other: MemoryGrant) -> Result<(), MemoryGrant> {
        if self.region != other.region || !Arc::ptr_eq(&self.releaser, &other.releaser) {
            return Err(other);
        }

        let current = self.size.load(Ordering::Relaxed);
        let other_size = other.size.load(Ordering::Relaxed);
        let Some(merged) = current.checked_add(other_size) else {
            return Err(other);
        };

        other.consumed = true;
        self.size.store(merged, Ordering::Relaxed);
        Ok(())
    }

    /// Merges another grant into this one.
    ///
    /// Both grants must cover the same region, be backed by the exact same
    /// releaser, and have a representable combined size.
    ///
    /// # Panics
    ///
    /// Panics if the grants are incompatible or their sizes overflow.
    pub fn merge(&mut self, other: MemoryGrant) {
        assert!(
            self.try_merge(other).is_ok(),
            "cannot merge grants from different regions or releasers, or with overflowing sizes"
        );
    }

    /// Attempts to consume this grant without releasing memory.
    ///
    /// A rejection returns the complete grant unchanged, retaining the only
    /// token that can reconcile its backing account.
    ///
    /// # Errors
    ///
    /// Returns the unchanged grant when its releaser requires the RAII token
    /// to remain attached to the accounted bytes.
    pub fn try_consume(self) -> Result<usize, Self> {
        match self.detach().try_into_untracked_bytes() {
            Ok(size) => Ok(size),
            Err(token) => Err(token.reattach()),
        }
    }

    /// Detaches this grant into an RAII token without releasing or changing
    /// its accounted bytes.
    pub fn detach(mut self) -> DetachedMemoryGrant {
        let token = DetachedMemoryGrant {
            releaser: Arc::clone(&self.releaser),
            size: self.size.load(Ordering::Relaxed),
            region: self.region,
            active: true,
        };
        self.consumed = true;
        token
    }

    /// Returns whether this grant has been consumed.
    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    /// Returns whether this grant is empty (size == 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.load(Ordering::Relaxed) == 0
    }
}

impl Drop for MemoryGrant {
    fn drop(&mut self) {
        if !self.consumed {
            let size = self.size.load(Ordering::Relaxed);
            if size > 0 {
                let _ = self.releaser.release_accounted(size, self.region);
            }
        }
    }
}

impl std::fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("size", &self.size.load(Ordering::Relaxed))
            .field("region", &self.region)
            .field("consumed", &self.consumed)
            .finish()
    }
}

/// A collection of memory grants that can be managed together.
///
/// The infallible tokenless transfer method is unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_common::memory::buffer::CompositeGrant;
/// let _ = CompositeGrant::consume_all;
/// ```
///
/// Total-size queries use checked arithmetic instead of an infallible wrapper:
///
/// ```compile_fail,E0599
/// use grafeo_common::memory::buffer::CompositeGrant;
/// let _ = CompositeGrant::total_size;
/// ```
///
/// Detaching a collection requires the fallible transfer method:
///
/// ```compile_fail,E0599
/// use grafeo_common::memory::buffer::CompositeGrant;
/// let _ = CompositeGrant::detach_all;
/// ```
#[derive(Debug, Default)]
pub struct CompositeGrant {
    grants: Vec<MemoryGrant>,
}

impl CompositeGrant {
    /// Creates a new empty composite grant.
    #[must_use]
    pub fn new() -> Self {
        Self { grants: Vec::new() }
    }

    /// Adds a grant to the collection.
    pub fn add(&mut self, grant: MemoryGrant) {
        self.grants.push(grant);
    }

    /// Returns the total size, or `None` when checked addition overflows.
    #[must_use]
    pub fn checked_total_size(&self) -> Option<usize> {
        self.grants
            .iter()
            .try_fold(0usize, |total, grant| total.checked_add(grant.size()))
    }

    /// Returns the number of grants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Returns whether the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }

    /// Attempts to reduce every grant to one untracked byte count.
    ///
    /// Total-size arithmetic and every releaser policy are preflighted before
    /// any grant is detached. Rejection returns the complete collection with
    /// all accounting tokens unchanged.
    ///
    /// # Errors
    ///
    /// Returns the unchanged composite when the total overflows or any grant
    /// requires its RAII accounting token to remain attached.
    pub fn try_consume_all(mut self) -> Result<usize, Self> {
        let Some(total) = self.checked_total_size() else {
            return Err(self);
        };
        if self
            .grants
            .iter()
            .any(|grant| !grant.releaser.allows_untracked_consumption())
        {
            return Err(self);
        }
        for grant in &mut self.grants {
            grant.consumed = true;
        }
        Ok(total)
    }

    /// Attempts to detach the collection into one opaque RAII token.
    ///
    /// # Errors
    ///
    /// Returns the unchanged composite when its checked total overflows.
    pub fn try_detach_all(self) -> Result<DetachedCompositeGrant, Self> {
        let Some(total_size) = self.checked_total_size() else {
            return Err(self);
        };
        Ok(DetachedCompositeGrant {
            grants: self.grants.into_iter().map(MemoryGrant::detach).collect(),
            total_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct MockReleaser {
        released: AtomicUsize,
        allocated: AtomicUsize,
    }

    impl MockReleaser {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                released: AtomicUsize::new(0),
                allocated: AtomicUsize::new(0),
            })
        }
    }

    impl GrantAccount for MockReleaser {
        fn release_accounted(
            &self,
            size: usize,
            _region: MemoryRegion,
        ) -> Result<(), MemoryGrantError> {
            self.released.fetch_add(size, Ordering::Relaxed);
            Ok(())
        }

        fn try_reserve_growth(
            &self,
            size: usize,
            _region: MemoryRegion,
        ) -> Result<(), MemoryGrantError> {
            self.allocated.fetch_add(size, Ordering::Relaxed);
            Ok(())
        }

        fn allows_untracked_consumption(&self) -> bool {
            true
        }
    }

    struct ManagedMockReleaser {
        released: AtomicUsize,
    }

    impl ManagedMockReleaser {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                released: AtomicUsize::new(0),
            })
        }
    }

    impl GrantAccount for ManagedMockReleaser {
        fn release_accounted(
            &self,
            size: usize,
            _region: MemoryRegion,
        ) -> Result<(), MemoryGrantError> {
            self.released.fetch_add(size, Ordering::Relaxed);
            Ok(())
        }

        fn try_reserve_growth(
            &self,
            _size: usize,
            _region: MemoryRegion,
        ) -> Result<(), MemoryGrantError> {
            Ok(())
        }
    }

    #[test]
    fn test_grant_drop_releases_memory() {
        let releaser = MockReleaser::new();

        {
            let _grant = MemoryGrant::new(
                Arc::clone(&releaser) as Arc<dyn GrantAccount>,
                1024,
                MemoryRegion::ExecutionBuffers,
            );
            assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
        }

        // After drop, memory should be released
        assert_eq!(releaser.released.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn test_grant_consume_no_release() {
        let releaser = MockReleaser::new();

        let grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            1024,
            MemoryRegion::ExecutionBuffers,
        );

        let size = grant.try_consume().expect("mock permits raw consumption");
        assert_eq!(size, 1024);

        // No release should happen
        assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_grant_resize_grow() {
        let releaser = MockReleaser::new();

        let mut grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            1024,
            MemoryRegion::ExecutionBuffers,
        );

        assert!(grant.resize(2048));
        assert_eq!(grant.size(), 2048);
        assert_eq!(releaser.allocated.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn test_grant_resize_shrink() {
        let releaser = MockReleaser::new();

        let mut grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            1024,
            MemoryRegion::ExecutionBuffers,
        );

        assert!(grant.resize(512));
        assert_eq!(grant.size(), 512);
        assert_eq!(releaser.released.load(Ordering::Relaxed), 512);
    }

    #[test]
    fn test_grant_split() {
        let releaser = MockReleaser::new();

        let mut grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            1000,
            MemoryRegion::ExecutionBuffers,
        );

        let split = grant.split(400).unwrap();
        assert_eq!(grant.size(), 600);
        assert_eq!(split.size(), 400);

        // Cannot split more than available
        assert!(grant.split(1000).is_none());
    }

    #[test]
    fn test_grant_merge() {
        let releaser = MockReleaser::new();

        let mut grant1 = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            600,
            MemoryRegion::ExecutionBuffers,
        );

        let grant2 = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            400,
            MemoryRegion::ExecutionBuffers,
        );

        grant1.merge(grant2);
        assert_eq!(grant1.size(), 1000);

        // grant2 was consumed during merge, no release
        assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_composite_grant() {
        let releaser = MockReleaser::new();

        let mut composite = CompositeGrant::new();
        assert!(composite.is_empty());

        composite.add(MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            100,
            MemoryRegion::ExecutionBuffers,
        ));
        composite.add(MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantAccount>,
            200,
            MemoryRegion::ExecutionBuffers,
        ));

        assert_eq!(composite.len(), 2);
        assert_eq!(composite.checked_total_size(), Some(300));

        let total = composite
            .try_consume_all()
            .expect("mock accounts permit raw consumption");
        assert_eq!(total, 300);

        // No release since all were consumed
        assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn mixed_releaser_composite_rejects_raw_consumption_before_detaching_any_grant() {
        let external = MockReleaser::new();
        let managed = ManagedMockReleaser::new();
        let mut composite = CompositeGrant::new();
        composite.add(MemoryGrant::new(
            Arc::clone(&external) as Arc<dyn GrantAccount>,
            10,
            MemoryRegion::ExecutionBuffers,
        ));
        composite.add(MemoryGrant::new(
            Arc::clone(&managed) as Arc<dyn GrantAccount>,
            20,
            MemoryRegion::ExecutionBuffers,
        ));

        let composite = composite
            .try_consume_all()
            .expect_err("managed account requires an RAII token");
        assert_eq!(external.released.load(Ordering::Acquire), 0);
        assert_eq!(managed.released.load(Ordering::Acquire), 0);
        drop(composite);
        assert_eq!(external.released.load(Ordering::Acquire), 10);
        assert_eq!(managed.released.load(Ordering::Acquire), 20);
    }

    #[test]
    fn detached_composite_is_one_opaque_reattachable_token() {
        let managed = ManagedMockReleaser::new();
        let mut composite = CompositeGrant::new();
        composite.add(MemoryGrant::new(
            Arc::clone(&managed) as Arc<dyn GrantAccount>,
            10,
            MemoryRegion::ExecutionBuffers,
        ));
        composite.add(MemoryGrant::new(
            Arc::clone(&managed) as Arc<dyn GrantAccount>,
            20,
            MemoryRegion::ExecutionBuffers,
        ));

        let detached = composite
            .try_detach_all()
            .expect("checked composite detaches");
        assert_eq!(detached.total_size(), 30);
        assert_eq!(detached.len(), 2);
        assert!(!detached.is_empty());
        assert_eq!(managed.released.load(Ordering::Acquire), 0);

        let composite = detached.reattach();
        assert_eq!(composite.checked_total_size(), Some(30));
        drop(composite);
        assert_eq!(managed.released.load(Ordering::Acquire), 30);
    }

    #[test]
    fn composite_checked_total_rejects_overflow_without_detaching_grants() {
        let first = MockReleaser::new();
        let second = MockReleaser::new();
        let mut composite = CompositeGrant::new();
        composite.add(MemoryGrant::new(
            Arc::clone(&first) as Arc<dyn GrantAccount>,
            usize::MAX,
            MemoryRegion::ExecutionBuffers,
        ));
        composite.add(MemoryGrant::new(
            Arc::clone(&second) as Arc<dyn GrantAccount>,
            1,
            MemoryRegion::ExecutionBuffers,
        ));

        assert_eq!(composite.checked_total_size(), None);
        let composite = composite
            .try_detach_all()
            .expect_err("overflow is rejected before opaque detachment");
        let composite = composite
            .try_consume_all()
            .expect_err("overflow is rejected during atomic preflight");
        assert_eq!(first.released.load(Ordering::Acquire), 0);
        assert_eq!(second.released.load(Ordering::Acquire), 0);
        drop(composite);
        assert_eq!(first.released.load(Ordering::Acquire), usize::MAX);
        assert_eq!(second.released.load(Ordering::Acquire), 1);
    }
}
