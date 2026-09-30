//! Shared immutable catalog cuts; only an actual edit copies a pinned cut.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::{Catalog, CatalogState};

pub(super) struct CatalogStateLock {
    pub(super) inner: RwLock<Arc<CatalogState>>,
}

impl CatalogStateLock {
    pub(super) fn new(state: CatalogState) -> Self {
        Self::shared(Arc::new(state))
    }

    pub(super) fn shared(state: Arc<CatalogState>) -> Self {
        Self {
            inner: RwLock::new(state),
        }
    }

    pub(super) fn read(&self) -> RwLockReadGuard<'_, Arc<CatalogState>> {
        self.inner.read()
    }

    pub(super) fn write(&self) -> CatalogWriteGuard<'_> {
        CatalogWriteGuard {
            inner: self.inner.write(),
        }
    }
}

pub(super) struct CatalogWriteGuard<'a> {
    inner: RwLockWriteGuard<'a, Arc<CatalogState>>,
}

impl Deref for CatalogWriteGuard<'_> {
    type Target = CatalogState;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for CatalogWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

impl Catalog {
    /// A private catalog wrapper sharing the current immutable payload.
    #[cfg(feature = "lpg")]
    pub(crate) fn snapshot(&self) -> Self {
        Self {
            state: CatalogStateLock::shared(Arc::clone(&self.state.read())),
        }
    }

    /// Exact cut identity, including dictionary allocator and index ownership.
    #[cfg(feature = "lpg")]
    pub(crate) fn same_cut(&self, other: &Self) -> bool {
        let left = Arc::clone(&self.state.read());
        let right = Arc::clone(&other.state.read());
        Arc::ptr_eq(&left, &right)
    }

    /// Data/index-only commits need stable schema semantics, not an unchanged
    /// unrelated owner registry. Exact owner/lifecycle CAS remains authoritative.
    #[cfg(feature = "lpg")]
    pub(crate) fn same_schema(&self, other: &Self) -> bool {
        let left = Arc::clone(&self.state.read());
        let right = Arc::clone(&other.state.read());
        Arc::ptr_eq(&left, &right) || left.schema == right.schema
    }
}
