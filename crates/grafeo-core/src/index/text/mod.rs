//! Full-text search with BM25 scoring and hybrid score fusion.
//!
//! This module provides text search capabilities for graph node properties,
//! enabling keyword-based retrieval alongside vector similarity search.
//!
//! # Components
//!
//! | Component | Feature | Description |
//! |-----------|---------|-------------|
//! | [`Tokenizer`] | `text-index` | Trait for text tokenization |
//! | [`SimpleTokenizer`] | `text-index` | Unicode-aware tokenizer with stop words |
//! | [`InvertedIndex`] | `text-index` | BM25-scored inverted index |
//! | [`FusionMethod`] | `hybrid-search` | Score fusion for combining search results |
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "text-index")]
//! # {
//! use grafeo_core::index::text::{InvertedIndex, BM25Config};
//! use grafeo_common::types::NodeId;
//!
//! let mut index = InvertedIndex::new(BM25Config::default());
//! index.insert(NodeId::new(1), "the quick brown fox");
//! index.insert(NodeId::new(2), "the lazy brown dog");
//!
//! let results = index.search("quick fox", 10);
//! assert_eq!(results[0].0, NodeId::new(1));
//! # }
//! ```

mod inverted_index;
#[cfg(all(feature = "text-index", feature = "lpg"))]
mod registry_fence;
pub mod section;
mod tokenizer;
pub mod tx_delta;
mod versioned;

pub use inverted_index::{BM25Config, InvertedIndex};
#[cfg(feature = "lpg")]
pub(crate) use inverted_index::{TextCommitScope, TextCommitWorkspace, TextScopeTransition};
#[cfg(all(feature = "text-index", feature = "lpg"))]
pub(crate) use registry_fence::{TextRegistryBatchFence, TextRegistryFenceWorkspace};
pub use section::TextIndexSection;
pub use tokenizer::{SimpleTokenizer, Tokenizer};
#[cfg(feature = "text-index")]
pub use tx_delta::TextIndexDelta;

#[cfg(feature = "hybrid-search")]
mod fusion;
#[cfg(feature = "hybrid-search")]
pub use fusion::{FusionMethod, fuse_results};

#[cfg(feature = "text-index")]
use std::sync::Arc;

#[cfg(feature = "text-index")]
use std::ops::{Deref, DerefMut};

#[cfg(feature = "text-index")]
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};

/// Shared guard for a text index's stable concrete image.
///
/// Registered indexes acquire the caller-retained outer gate first and the
/// private registry target second. Consequently a retained outer read or write
/// guard still linearizes with every store/view operation even though replacing
/// that gate's payload cannot replace the target.
#[cfg(feature = "text-index")]
pub struct TextIndexReadGuard {
    // Field order is intentional: release the target before the outer gate.
    target: Option<ArcRwLockReadGuard<RawRwLock, InvertedIndex>>,
    gate: ArcRwLockReadGuard<RawRwLock, InvertedIndex>,
}

#[cfg(feature = "text-index")]
impl TextIndexReadGuard {
    fn acquire(
        gate: Arc<RwLock<InvertedIndex>>,
        fixed_target: Option<Arc<RwLock<InvertedIndex>>>,
    ) -> Self {
        let gate_guard = gate.read_arc();
        let target = fixed_target.or_else(|| gate_guard.forwarding_target());
        let target = target.filter(|target| !Arc::ptr_eq(&gate, target));
        let target_guard = target.map(|target| target.read_arc());
        Self {
            target: target_guard,
            gate: gate_guard,
        }
    }
}

#[cfg(feature = "text-index")]
impl Deref for TextIndexReadGuard {
    type Target = InvertedIndex;

    fn deref(&self) -> &Self::Target {
        self.target.as_deref().unwrap_or(&self.gate)
    }
}

/// Exclusive gate→target guard used by trusted store and restore paths.
#[cfg(feature = "text-index")]
pub(crate) struct TextIndexWriteGuard {
    // Field order is intentional: release the target before the outer gate.
    target: Option<ArcRwLockWriteGuard<RawRwLock, InvertedIndex>>,
    gate: ArcRwLockWriteGuard<RawRwLock, InvertedIndex>,
}

#[cfg(feature = "text-index")]
impl TextIndexWriteGuard {
    fn acquire(
        gate: Arc<RwLock<InvertedIndex>>,
        fixed_target: Option<Arc<RwLock<InvertedIndex>>>,
    ) -> Self {
        let gate_guard = gate.write_arc();
        let target = fixed_target.or_else(|| gate_guard.forwarding_target());
        let target = target.filter(|target| !Arc::ptr_eq(&gate, target));
        let target_guard = target.map(|target| target.write_arc());
        Self {
            target: target_guard,
            gate: gate_guard,
        }
    }
}

#[cfg(feature = "text-index")]
impl Deref for TextIndexWriteGuard {
    type Target = InvertedIndex;

    fn deref(&self) -> &Self::Target {
        self.target.as_deref().unwrap_or(&self.gate)
    }
}

#[cfg(feature = "text-index")]
impl DerefMut for TextIndexWriteGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.target.as_deref_mut().unwrap_or(&mut self.gate)
    }
}

/// Store-owned registration retaining both synchronization identities.
///
/// `gate` is the exact outer Arc supplied by the caller. `target` is the
/// concrete authority-bound object that only the registry can replace.
#[cfg(all(feature = "text-index", feature = "lpg"))]
#[derive(Clone)]
pub(crate) struct RegisteredTextIndex {
    gate: Arc<RwLock<InvertedIndex>>,
    target: Arc<RwLock<InvertedIndex>>,
}

#[cfg(all(feature = "text-index", feature = "lpg"))]
impl RegisteredTextIndex {
    /// Retains concrete target identity, not its payload or replaceable caller shell.
    pub(crate) fn target_identity(&self) -> std::sync::Weak<RwLock<InvertedIndex>> {
        Arc::downgrade(&self.target)
    }

    pub(crate) fn new(
        gate: Arc<RwLock<InvertedIndex>>,
        target: Arc<RwLock<InvertedIndex>>,
    ) -> Self {
        Self { gate, target }
    }

    pub(crate) fn read(&self) -> TextIndexReadGuard {
        TextIndexReadGuard::acquire(Arc::clone(&self.gate), Some(Arc::clone(&self.target)))
    }

    pub(crate) fn write(&self) -> TextIndexWriteGuard {
        TextIndexWriteGuard::acquire(Arc::clone(&self.gate), Some(Arc::clone(&self.target)))
    }

    /// Preparation must not wait while retaining registry writers: aliases can
    /// already hold either lock while trying to enter a mutation scope.
    #[cfg(feature = "lpg")]
    pub(crate) fn try_write(&self) -> Option<TextIndexWriteGuard> {
        let gate = self.gate.try_write_arc()?;
        let target = if Arc::ptr_eq(&self.gate, &self.target) {
            None
        } else {
            Some(self.target.try_write_arc()?)
        };
        Some(TextIndexWriteGuard { target, gate })
    }

    /// Builds an exact, independently mutable-object-backed read snapshot.
    ///
    /// The registered caller gate belongs to the logical index and therefore
    /// moves to a same-incarnation successor. A retired physical store instead
    /// receives this private gate plus an exact concrete fork, so later writes
    /// through the successor cannot leak new identities into the old store's
    /// fresh registry reads.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    pub(crate) fn exact_runtime_fork_with_fence(&self) -> (Self, TextIndexReadGuard) {
        let fence = self.read();
        let frozen = Arc::new(RwLock::new(fence.exact_runtime_fork()));
        let private_gate = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        (Self::new(private_gate, frozen), fence)
    }

    fn view(&self) -> TextIndexView {
        TextIndexView {
            gate: Arc::clone(&self.gate),
            target: Some(Arc::clone(&self.target)),
        }
    }
}

/// Read-only capability for an LPG-owned full-text index.
///
/// The returned read guard exposes the normal search and inspection methods,
/// but no write-lock or mutable index access.
#[cfg(feature = "text-index")]
#[derive(Clone)]
pub struct TextIndexView {
    gate: Arc<RwLock<InvertedIndex>>,
    /// `Some` for an LPG registry view; `None` for a standalone/dynamic view
    /// constructed by `TextIndexSection::new`.
    target: Option<Arc<RwLock<InvertedIndex>>>,
}

#[cfg(feature = "text-index")]
impl TextIndexView {
    pub(crate) fn new(gate: Arc<RwLock<InvertedIndex>>) -> Self {
        Self { gate, target: None }
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn from_registered(index: &RegisteredTextIndex) -> Self {
        index.view()
    }

    /// Acquires one coherent shared guard for search and inspection.
    ///
    /// For an LPG-owned index this retains both the caller's outer gate and the
    /// registry's private concrete target for the guard's complete lifetime.
    pub fn read(&self) -> TextIndexReadGuard {
        TextIndexReadGuard::acquire(Arc::clone(&self.gate), self.target.clone())
    }
}

#[cfg(feature = "text-index")]
impl std::fmt::Debug for TextIndexView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let index = self.read();
        f.debug_struct("TextIndexView")
            .field("len", &index.len())
            .field("term_count", &index.term_count())
            .finish_non_exhaustive()
    }
}
