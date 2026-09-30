//! Vector similarity search support.
//!
//! This module provides infrastructure for storing and searching vector embeddings,
//! enabling AI/ML use cases like RAG, semantic search, and recommendations.
//!
//! # Distance Metrics
//!
//! Choose the metric based on your embedding type:
//!
//! | Metric | Best For | Range |
//! |--------|----------|-------|
//! | [`Cosine`](DistanceMetric::Cosine) | Normalized embeddings (text) | [0, 2] |
//! | [`Euclidean`](DistanceMetric::Euclidean) | Raw embeddings | [0, inf) |
//! | [`DotProduct`](DistanceMetric::DotProduct) | Max inner product search | (-inf, inf) |
//! | [`Manhattan`](DistanceMetric::Manhattan) | Outlier-resistant | [0, inf) |
//!
//! # Index Types
//!
//! | Index | Complexity | Use Case |
//! |-------|------------|----------|
//! | [`brute_force_knn`] | O(n) | Small datasets, exact results |
//! | [`HnswIndex`] | O(log n) | Large datasets, approximate results |
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::vector::{compute_distance, DistanceMetric, brute_force_knn};
//! use grafeo_common::types::NodeId;
//!
//! // Compute distance between two vectors
//! let query = [0.1f32, 0.2, 0.3];
//! let doc1 = [0.1f32, 0.2, 0.35];
//! let doc2 = [0.5f32, 0.6, 0.7];
//!
//! let dist1 = compute_distance(&query, &doc1, DistanceMetric::Cosine);
//! let dist2 = compute_distance(&query, &doc2, DistanceMetric::Cosine);
//!
//! // doc1 is more similar (smaller distance)
//! assert!(dist1 < dist2);
//!
//! // Brute-force k-NN search
//! let vectors = vec![
//!     (NodeId::new(1), doc1.as_slice()),
//!     (NodeId::new(2), doc2.as_slice()),
//! ];
//!
//! let results = brute_force_knn(vectors.into_iter(), &query, 1, DistanceMetric::Cosine);
//! assert_eq!(results[0].0, NodeId::new(1)); // doc1 is closest
//! ```
//!
//! # HNSW Index (requires `vector-index` feature)
//!
//! For larger datasets, use the HNSW approximate nearest neighbor index:
//!
//! ```no_run
//! # #[cfg(feature = "vector-index")]
//! # {
//! use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
//! use grafeo_common::types::NodeId;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let index = HnswIndex::new(config);
//!
//! // Build an accessor backed by a HashMap
//! let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
//! let embedding: Arc<[f32]> = vec![0.1f32; 384].into();
//! map.insert(NodeId::new(1), embedding.clone());
//! let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
//!
//! // Insert vectors (requires accessor for neighbor lookups)
//! index.insert(NodeId::new(1), &embedding, &accessor);
//!
//! // Search (O(log n))
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10, &accessor);
//! # }
//! ```

mod accessor;
mod distance;
mod mmr;
pub mod quantization;
pub mod rabitq;
mod simd;
pub mod storage;
pub mod zone_map;

#[cfg(feature = "vector-index")]
mod config;
#[cfg(feature = "vector-index")]
mod hnsw;
#[cfg(feature = "vector-index")]
pub mod paged_topology;
#[cfg(feature = "vector-index")]
mod quantized_hnsw;
#[cfg(feature = "vector-index")]
pub mod section;

pub use accessor::{
    PropertyVectorAccessor, SpillableVectorAccessor, VectorAccessor, VectorAccessorKind,
    value_to_vector,
};
pub use distance::{
    DistanceMetric, compute_distance, cosine_distance, cosine_similarity, dot_product,
    euclidean_distance, euclidean_distance_squared, l2_norm, manhattan_distance, normalize,
    simd_support,
};
pub use mmr::mmr_select;
pub use quantization::{BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer};
pub use rabitq::{
    RabitqCode, RabitqError, RabitqIndex, RabitqQuantizer, RabitqQuery, RabitqView,
    TwoStageVectorIndex,
};
#[cfg(feature = "mmap")]
pub use storage::MmapStorage;
pub use storage::{RamStorage, StorageBackend, VectorStorage};
pub use zone_map::VectorZoneMap;

#[cfg(feature = "vector-index")]
pub use config::HnswConfig;
#[cfg(all(feature = "vector-index", feature = "lpg"))]
pub(crate) mod maintenance;
#[cfg(feature = "vector-index")]
pub use hnsw::HnswIndex;
#[cfg(all(feature = "vector-index", feature = "lpg"))]
pub(crate) use hnsw::HnswScopeTransition;
#[cfg(feature = "vector-index")]
pub use quantized_hnsw::QuantizedHnswIndex;
#[cfg(feature = "vector-index")]
pub use section::VectorStoreSection;
// VectorIndexKind is defined below in this file (not in a sub-module).

use grafeo_common::types::NodeId;
#[cfg(feature = "vector-index")]
use hnsw::{HnswExactState, PreparedHnswExactState};
#[cfg(feature = "vector-index")]
use quantized_hnsw::{PreparedQuantizedExactState, QuantizedExactState};
#[cfg(feature = "vector-index")]
use std::collections::HashSet;
#[cfg(feature = "vector-index")]
use std::sync::Arc;

// ── VectorIndexKind ────────────────────────────────────────────────

/// Unified enum for vector indexes stored in the LPG store.
///
/// Wraps either a plain [`HnswIndex`] or a [`QuantizedHnswIndex`],
/// allowing the store and engine to handle both through a single type.
#[cfg(feature = "vector-index")]
pub enum VectorIndexKind {
    /// Standard full-precision HNSW index.
    Hnsw(HnswIndex),
    /// Quantized HNSW index (scalar, binary, or product quantization).
    Quantized(QuantizedHnswIndex),
}

/// Owned mutation exclusion retained across an exact runtime index handoff.
///
/// Every vector mutator enters the same reentrant publication gate. Holding
/// this fence after creating a frozen fork therefore keeps the fork equal to
/// the source until the caller publishes the registry move or abandons it.
#[cfg(all(feature = "vector-index", feature = "lpg", feature = "compact-store"))]
pub(crate) struct VectorIndexReadFence {
    _guard:
        parking_lot::ArcReentrantMutexGuard<parking_lot::RawMutex, parking_lot::RawThreadId, ()>,
}

#[cfg(feature = "vector-index")]
#[derive(Debug, Clone)]
enum VectorExactState {
    Hnsw(HnswExactState),
    Quantized(QuantizedExactState),
}

#[cfg(feature = "vector-index")]
enum PreparedVectorExactState {
    Hnsw(PreparedHnswExactState),
    Quantized(PreparedQuantizedExactState),
}

#[cfg(feature = "vector-index")]
struct VectorExactRestoreGuard<'index> {
    index: &'index VectorIndexKind,
    _mutation: hnsw::HnswMutationGuard<'index>,
}

/// A concrete replacement bound to its exact target and a retained mutation
/// guard. Binding checks every rejection condition before any state is moved;
/// applying the resulting capability cannot encounter a target/kind mismatch.
#[cfg(feature = "vector-index")]
#[must_use = "a ready exact replacement must be applied or explicitly abandoned"]
enum ReadyVectorExactRestore<'index, 'guard> {
    Hnsw {
        target: &'index HnswIndex,
        state: PreparedHnswExactState,
        _guard: &'guard VectorExactRestoreGuard<'index>,
    },
    Quantized {
        target: &'index QuantizedHnswIndex,
        state: PreparedQuantizedExactState,
        _guard: &'guard VectorExactRestoreGuard<'index>,
    },
}

#[cfg(feature = "vector-index")]
impl<'index, 'guard> ReadyVectorExactRestore<'index, 'guard> {
    fn bind(
        target: &'index VectorIndexKind,
        state: PreparedVectorExactState,
        guard: &'guard VectorExactRestoreGuard<'index>,
    ) -> std::result::Result<Self, String> {
        if !std::ptr::eq(target, guard.index) {
            return Err("vector restore guard belongs to a different target".to_string());
        }
        match (target, state) {
            (VectorIndexKind::Hnsw(target), PreparedVectorExactState::Hnsw(state)) => {
                Ok(Self::Hnsw {
                    target,
                    state,
                    _guard: guard,
                })
            }
            (VectorIndexKind::Quantized(target), PreparedVectorExactState::Quantized(state)) => {
                Ok(Self::Quantized {
                    target,
                    state,
                    _guard: guard,
                })
            }
            _ => Err("prepared vector state concrete kind does not match target index".to_string()),
        }
    }

    fn apply(self) {
        match self {
            Self::Hnsw { target, state, .. } => target.apply_prepared_exact_restore(state),
            Self::Quantized { target, state, .. } => target.apply_prepared_exact_restore(state),
        }
    }
}

/// Read-only capability for an LPG-owned vector index.
///
/// Cloning this value clones only the handle. Topology mutation remains owned
/// by the graph store and its transaction/publication protocol.
#[cfg(feature = "vector-index")]
#[derive(Clone)]
pub struct VectorIndexView {
    inner: Arc<VectorIndexKind>,
}

#[cfg(feature = "vector-index")]
impl VectorIndexView {
    pub(crate) fn new(inner: Arc<VectorIndexKind>) -> Self {
        Self { inner }
    }

    /// Internal borrowed target. Public maintenance additionally requires the
    /// concrete owning store and its continuously held write authority.
    #[cfg(feature = "lpg")]
    pub(crate) fn commit_target(&self) -> &VectorIndexKind {
        &self.inner
    }

    fn snapshot_exact(&self) -> std::result::Result<VectorExactState, String> {
        self.inner.snapshot_exact_state()
    }

    fn prepare_exact_restore(
        &self,
        state: VectorExactState,
    ) -> std::result::Result<PreparedVectorExactState, String> {
        self.inner.prepare_exact_state(state)
    }

    fn pin_exact_restore(&self) -> Option<VectorExactRestoreGuard<'_>> {
        let index = self.inner.as_ref();
        let mutation = index.pin_mutation()?;
        Some(VectorExactRestoreGuard {
            index,
            _mutation: mutation,
        })
    }

    fn exact_restore_identity(&self) -> *const VectorIndexKind {
        Arc::as_ptr(&self.inner)
    }

    fn bind_exact_restore<'index, 'guard>(
        &'index self,
        state: PreparedVectorExactState,
        guard: &'guard VectorExactRestoreGuard<'index>,
    ) -> std::result::Result<ReadyVectorExactRestore<'index, 'guard>, String> {
        ReadyVectorExactRestore::bind(self.inner.as_ref(), state, guard)
    }

    /// Returns the HNSW configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        self.inner.config()
    }

    /// Returns the number of live vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the index has no live vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns whether the index contains `id`.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.inner.contains(id)
    }

    /// Searches for the nearest neighbors.
    #[must_use]
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.inner.search(query, k, accessor)
    }

    /// Searches with a custom beam width.
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.inner.search_with_ef(query, k, ef, accessor)
    }

    /// Searches within an allowlist.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.inner.search_with_filter(query, k, allowlist, accessor)
    }

    /// Searches with a custom beam width within an allowlist.
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.inner
            .search_with_ef_and_filter(query, k, ef, allowlist, accessor)
    }

    /// Batch-searches query vectors.
    #[must_use]
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        self.inner.batch_search(queries, k, accessor)
    }

    /// Batch-searches with a custom beam width.
    #[must_use]
    pub fn batch_search_with_ef(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        self.inner.batch_search_with_ef(queries, k, ef, accessor)
    }

    /// Batch-searches within an allowlist.
    #[must_use]
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        self.inner
            .batch_search_with_filter(queries, k, allowlist, accessor)
    }

    /// Batch-searches with a custom beam width within an allowlist.
    #[must_use]
    pub fn batch_search_with_ef_and_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        self.inner
            .batch_search_with_ef_and_filter(queries, k, ef, allowlist, accessor)
    }

    /// Searches with snapshot visibility supplied by the caller.
    #[must_use]
    pub fn search_visible(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        is_visible: &dyn Fn(NodeId) -> bool,
        accessor: &dyn VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.inner
            .search_visible(query, k, ef, is_visible, accessor)
    }

    /// Snapshots topology for read-only persistence/export.
    #[must_use]
    pub fn snapshot_topology(&self) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        self.inner.snapshot_topology()
    }

    /// Returns estimated heap usage.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        self.inner.heap_memory_bytes()
    }

    /// Returns the quantization type, if any.
    #[must_use]
    pub fn quantization_type(&self) -> Option<QuantizationType> {
        self.inner.quantization_type()
    }

    /// Returns whether the current persistence descriptors can reconstruct
    /// this index exactly.
    #[must_use]
    pub fn has_persistence_representable_kind(&self) -> bool {
        self.inner.has_persistence_representable_kind()
    }
}

#[cfg(feature = "vector-index")]
impl std::fmt::Debug for VectorIndexView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndexView")
            .field("config", self.config())
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "vector-index")]
impl VectorIndexKind {
    fn snapshot_exact_state(&self) -> std::result::Result<VectorExactState, String> {
        match self {
            Self::Hnsw(index) => index.snapshot_exact().map(VectorExactState::Hnsw),
            Self::Quantized(index) => index.snapshot_exact().map(VectorExactState::Quantized),
        }
    }

    fn prepare_exact_state(
        &self,
        state: VectorExactState,
    ) -> std::result::Result<PreparedVectorExactState, String> {
        match (self, state) {
            (Self::Hnsw(index), VectorExactState::Hnsw(state)) => index
                .prepare_exact_restore(state)
                .map(PreparedVectorExactState::Hnsw),
            (Self::Quantized(index), VectorExactState::Quantized(state)) => index
                .prepare_exact_restore(state)
                .map(PreparedVectorExactState::Quantized),
            _ => Err("vector snapshot concrete kind does not match target index".to_string()),
        }
    }

    /// Creates a detached, permanently read-only clone of this index's exact
    /// logical state for a retained compact generation.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    pub(crate) fn fork_exact_read_snapshot(
        &self,
    ) -> std::result::Result<Arc<VectorIndexKind>, String> {
        let state = self.snapshot_exact_state()?;
        let fork = match &state {
            VectorExactState::Hnsw(state) => {
                Self::Hnsw(HnswIndex::with_seed(state.config.clone(), 0))
            }
            VectorExactState::Quantized(state) => Self::Quantized(QuantizedHnswIndex::with_seed(
                state.hnsw.config.clone(),
                state.quantization_type,
                0,
            )),
        };
        let prepared = fork.prepare_exact_state(state)?;
        let mutation = fork
            .pin_mutation()
            .ok_or_else(|| "fresh vector snapshot clone rejected exact restore".to_string())?;
        let guard = VectorExactRestoreGuard {
            index: &fork,
            _mutation: mutation,
        };
        ReadyVectorExactRestore::bind(&fork, prepared, &guard)?.apply();
        drop(guard);
        match &fork {
            Self::Hnsw(index) => index.freeze_exact_read_snapshot(),
            Self::Quantized(index) => index.freeze_exact_read_snapshot(),
        }
        Ok(Arc::new(fork))
    }

    /// Forks an exact frozen image while retaining exclusion against all
    /// subsequent source mutations.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    pub(crate) fn fork_exact_read_snapshot_with_fence(
        &self,
    ) -> std::result::Result<(Arc<VectorIndexKind>, VectorIndexReadFence), String> {
        let gate = match self {
            Self::Hnsw(index) => index.publication_gate(),
            Self::Quantized(index) => index.publication_gate(),
        };
        let guard = gate.lock_arc();
        let fork = self.fork_exact_read_snapshot()?;
        Ok((fork, VectorIndexReadFence { _guard: guard }))
    }

    /// Retains the global, rare index-ownership transition gate.
    ///
    /// LPG uses one proof across the preflight and apply phases for every
    /// index in a named-graph tree, preventing retained aliases from changing
    /// an index scope between those phases.
    #[cfg(feature = "lpg")]
    pub(crate) fn pin_scope_transition() -> hnsw::HnswScopeTransition {
        hnsw::HnswScopeTransition::acquire()
    }

    /// Compatibility entry point for the currently committed LPG installer.
    /// The staged store-authority follow-up retains one transition across its
    /// complete multi-index preflight and calls the `_under_transition` form
    /// directly; keeping this narrow wrapper lets the vector hardening commit
    /// remain independently buildable until that follow-up lands.
    #[cfg(feature = "lpg")]
    pub(crate) fn binding_is_compatible(
        &self,
        owner: u64,
        slot: u64,
        transition: &hnsw::HnswScopeTransition,
    ) -> bool {
        match self {
            Self::Hnsw(index) => index.binding_is_compatible(owner, slot, transition),
            Self::Quantized(index) => index.binding_is_compatible(owner, slot, transition),
        }
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn is_bound_to(
        &self,
        owner: u64,
        slot: u64,
        transition: &hnsw::HnswScopeTransition,
    ) -> bool {
        match self {
            Self::Hnsw(index) => index.is_bound_to(owner, slot, transition),
            Self::Quantized(index) => index.is_bound_to(owner, slot, transition),
        }
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn bind_under_transition(
        &self,
        owner: u64,
        slot: u64,
        transition: &hnsw::HnswScopeTransition,
    ) -> bool {
        match self {
            Self::Hnsw(index) => index.bind_under_transition(owner, slot, transition),
            Self::Quantized(index) => index.bind_under_transition(owner, slot, transition),
        }
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn scope_is_unsealed(&self, transition: &hnsw::HnswScopeTransition) -> bool {
        match self {
            Self::Hnsw(index) => index.scope_is_unsealed(transition),
            Self::Quantized(index) => index.scope_is_unsealed(transition),
        }
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn scope_is_compatible(
        &self,
        scope: u64,
        transition: &hnsw::HnswScopeTransition,
    ) -> bool {
        match self {
            Self::Hnsw(index) => index.scope_is_compatible(scope, transition),
            Self::Quantized(index) => index.scope_is_compatible(scope, transition),
        }
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn seal_with_scope_under_transition(
        &self,
        scope: u64,
        transition: &hnsw::HnswScopeTransition,
    ) -> bool {
        match self {
            Self::Hnsw(index) => index.seal_with_scope_under_transition(scope, transition),
            Self::Quantized(index) => index.seal_with_scope_under_transition(scope, transition),
        }
    }

    fn pin_mutation(&self) -> Option<hnsw::HnswMutationGuard<'_>> {
        match self {
            Self::Hnsw(index) => index.pin_mutation(),
            Self::Quantized(index) => index.pin_mutation(),
        }
    }

    /// Returns the HNSW configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        match self {
            Self::Hnsw(idx) => idx.config(),
            Self::Quantized(idx) => idx.config(),
        }
    }

    /// Returns the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.len(),
            Self::Quantized(idx) => idx.len(),
        }
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Hnsw(idx) => idx.is_empty(),
            Self::Quantized(idx) => idx.is_empty(),
        }
    }

    /// Returns true if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        match self {
            Self::Hnsw(idx) => idx.contains(id),
            Self::Quantized(idx) => idx.contains(id),
        }
    }

    /// Removes a vector from the index.
    pub fn remove(&self, id: NodeId) -> bool {
        match self {
            Self::Hnsw(idx) => idx.remove(id),
            Self::Quantized(idx) => idx.remove(id),
        }
    }

    /// Inserts a vector into the index.
    ///
    /// For `Hnsw`, the accessor is used for neighbor distance lookups.
    /// For `Quantized`, the vector is stored internally and the accessor is unused.
    pub fn insert(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        match self {
            Self::Hnsw(idx) => idx.insert(id, vector, accessor),
            Self::Quantized(idx) => idx.insert(id, vector),
        }
    }

    /// Searches for the k nearest neighbors.
    #[must_use]
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search(query, k, accessor),
            Self::Quantized(idx) => idx.search(query, k),
        }
    }

    /// Searches with a custom ef (beam width) parameter.
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_with_ef(query, k, ef, accessor),
            Self::Quantized(idx) => idx.search_with_ef(query, k, ef),
        }
    }

    /// Searches with an allowlist filter.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_with_filter(query, k, allowlist, accessor),
            Self::Quantized(idx) => idx.search_with_filter(query, k, allowlist),
        }
    }

    /// Searches with a custom ef and an allowlist filter.
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_with_ef_and_filter(query, k, ef, allowlist, accessor),
            Self::Quantized(idx) => idx.search_with_ef_and_filter(query, k, ef, allowlist),
        }
    }

    /// Batch search for multiple queries.
    #[must_use]
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => idx.batch_search(queries, k, accessor),
            Self::Quantized(idx) => idx.batch_search(queries, k),
        }
    }

    /// Batch search with custom ef for multiple queries.
    #[must_use]
    pub fn batch_search_with_ef(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => idx.batch_search_with_ef(queries, k, ef, accessor),
            Self::Quantized(idx) => idx.batch_search_with_ef(queries, k, ef),
        }
    }

    /// Batch search with an allowlist filter for multiple queries.
    #[must_use]
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => idx.batch_search_with_filter(queries, k, allowlist, accessor),
            Self::Quantized(idx) => idx.batch_search_with_filter(queries, k, allowlist),
        }
    }

    /// Batch search with custom ef and an allowlist filter.
    #[must_use]
    pub fn batch_search_with_ef_and_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => {
                idx.batch_search_with_ef_and_filter(queries, k, ef, allowlist, accessor)
            }
            Self::Quantized(idx) => idx.batch_search_with_ef_and_filter(queries, k, ef, allowlist),
        }
    }

    /// Snapshot-aware predicate-filtered search.
    ///
    /// Traverses the full graph for connectivity, returns only nodes that
    /// satisfy `is_visible`, and scores via the supplied `accessor`.
    /// The beam is widened by `VISIBLE_EF_FACTOR` (4×) so filtering still
    /// yields `k` results. Uses `&dyn` so the caller can pass trait objects.
    #[must_use]
    pub fn search_visible(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        is_visible: &dyn Fn(NodeId) -> bool,
        accessor: &dyn VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_visible(query, k, ef, is_visible, accessor),
            Self::Quantized(idx) => idx.search_visible(query, k, ef, is_visible, accessor),
        }
    }

    /// Snapshot the HNSW topology for serialization.
    #[must_use]
    pub fn snapshot_topology(&self) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        match self {
            Self::Hnsw(idx) => idx.snapshot_topology(),
            Self::Quantized(idx) => idx.snapshot_topology(),
        }
    }

    /// Restore topology from a snapshot.
    pub fn restore_topology(
        &self,
        entry_point: Option<NodeId>,
        max_level: usize,
        node_data: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    ) {
        match self {
            Self::Hnsw(idx) => idx.restore_topology(entry_point, max_level, node_data),
            Self::Quantized(idx) => idx.restore_topology(entry_point, max_level, node_data),
        }
    }

    /// Returns estimated heap memory in bytes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.heap_memory_bytes(),
            Self::Quantized(idx) => idx.heap_memory_bytes(),
        }
    }

    /// Garbage collects soft-deleted nodes below the GC horizon.
    ///
    /// Rebuilds the HNSW topology retaining only nodes where `is_live(id)`
    /// is `true`. For the plain HNSW variant, `accessor` is used to supply
    /// each live node's vector during the rebuild; for the quantized variant,
    /// vectors are read from the internal store and `accessor` is unused
    /// (callers may pass a no-op).
    ///
    /// # Errors
    /// Rejects denied authority or invalid retained vectors without replacing state.
    pub fn gc(
        &self,
        is_live: &dyn Fn(NodeId) -> bool,
        accessor: &dyn VectorAccessor,
    ) -> std::result::Result<(), String> {
        match self {
            Self::Hnsw(idx) => idx.gc(is_live, accessor),
            Self::Quantized(idx) => idx.gc(is_live),
        }
    }

    /// Returns the quantization type, if this is a quantized index.
    #[must_use]
    pub fn quantization_type(&self) -> Option<QuantizationType> {
        match self {
            Self::Hnsw(_) => None,
            Self::Quantized(idx) => Some(idx.quantization_type()),
        }
    }

    /// Whether current persistence descriptors can reconstruct the index kind
    /// and every quantized-specific option exactly.
    #[must_use]
    pub fn has_persistence_representable_kind(&self) -> bool {
        match self {
            Self::Hnsw(_) => true,
            Self::Quantized(index) => index.has_persistence_defaults(),
        }
    }

    /// Returns a reference to the inner `HnswIndex`, if this is a plain HNSW variant.
    #[must_use]
    pub fn as_hnsw(&self) -> Option<&HnswIndex> {
        match self {
            Self::Hnsw(idx) => Some(idx),
            Self::Quantized(_) => None,
        }
    }

    /// Returns a reference to the inner `QuantizedHnswIndex`, if quantized.
    #[must_use]
    pub fn as_quantized(&self) -> Option<&QuantizedHnswIndex> {
        match self {
            Self::Hnsw(_) => None,
            Self::Quantized(idx) => Some(idx),
        }
    }
}

#[cfg(feature = "vector-index")]
impl From<HnswIndex> for VectorIndexKind {
    fn from(idx: HnswIndex) -> Self {
        Self::Hnsw(idx)
    }
}

#[cfg(feature = "vector-index")]
impl From<QuantizedHnswIndex> for VectorIndexKind {
    fn from(idx: QuantizedHnswIndex) -> Self {
        Self::Quantized(idx)
    }
}

/// Configuration for vector search operations.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Expected vector dimensions (for validation).
    pub dimensions: usize,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
}

impl VectorConfig {
    /// Creates a new vector configuration.
    #[must_use]
    pub const fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        Self { dimensions, metric }
    }

    /// Creates a configuration for cosine similarity with the given dimensions.
    #[must_use]
    pub const fn cosine(dimensions: usize) -> Self {
        Self::new(dimensions, DistanceMetric::Cosine)
    }

    /// Creates a configuration for Euclidean distance with the given dimensions.
    #[must_use]
    pub const fn euclidean(dimensions: usize) -> Self {
        Self::new(dimensions, DistanceMetric::Euclidean)
    }
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 384, // Common embedding size (MiniLM, etc.)
            metric: DistanceMetric::default(),
        }
    }
}

/// Performs brute-force k-nearest neighbor search.
///
/// This is O(n) where n is the number of vectors. Use this for:
/// - Small datasets (< 10K vectors)
/// - Baseline comparisons
/// - Exact nearest neighbor search
///
/// For larger datasets, use an approximate index like HNSW.
///
/// # Arguments
///
/// * `vectors` - Iterator of (id, vector) pairs to search
/// * `query` - The query vector
/// * `k` - Number of nearest neighbors to return
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Vector of (id, distance) pairs sorted by distance (ascending).
///
/// # Example
///
/// ```
/// use grafeo_core::index::vector::{brute_force_knn, DistanceMetric};
/// use grafeo_common::types::NodeId;
///
/// let vectors = vec![
///     (NodeId::new(1), [0.1f32, 0.2, 0.3].as_slice()),
///     (NodeId::new(2), [0.4f32, 0.5, 0.6].as_slice()),
///     (NodeId::new(3), [0.7f32, 0.8, 0.9].as_slice()),
/// ];
///
/// let query = [0.15f32, 0.25, 0.35];
/// let results = brute_force_knn(vectors.into_iter(), &query, 2, DistanceMetric::Euclidean);
///
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].0, NodeId::new(1)); // Closest
/// ```
pub fn brute_force_knn<'a, I>(
    vectors: I,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(NodeId, f32)>
where
    I: Iterator<Item = (NodeId, &'a [f32])>,
{
    let mut results: Vec<(NodeId, f32)> = vectors
        .map(|(id, vec)| (id, compute_distance(query, vec, metric)))
        .collect();

    // Sort by distance (ascending)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to k
    results.truncate(k);
    results
}

/// Performs brute-force k-nearest neighbor search with a filter predicate.
///
/// Only considers vectors where the predicate returns true.
///
/// # Arguments
///
/// * `vectors` - Iterator of (id, vector) pairs to search
/// * `query` - The query vector
/// * `k` - Number of nearest neighbors to return
/// * `metric` - Distance metric to use
/// * `predicate` - Filter function; only vectors where this returns true are considered
///
/// # Returns
///
/// Vector of (id, distance) pairs sorted by distance (ascending).
pub fn brute_force_knn_filtered<'a, I, F>(
    vectors: I,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
    predicate: F,
) -> Vec<(NodeId, f32)>
where
    I: Iterator<Item = (NodeId, &'a [f32])>,
    F: Fn(NodeId) -> bool,
{
    let mut results: Vec<(NodeId, f32)> = vectors
        .filter(|(id, _)| predicate(*id))
        .map(|(id, vec)| (id, compute_distance(query, vec, metric)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Computes the distance between a query and multiple vectors in batch.
///
/// More efficient than computing distances one by one for large batches.
///
/// # Returns
///
/// Vector of (id, distance) pairs in the same order as input.
pub fn batch_distances<'a, I>(
    vectors: I,
    query: &[f32],
    metric: DistanceMetric,
) -> Vec<(NodeId, f32)>
where
    I: Iterator<Item = (NodeId, &'a [f32])>,
{
    vectors
        .map(|(id, vec)| (id, compute_distance(query, vec, metric)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_config_default() {
        let config = VectorConfig::default();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_vector_config_constructors() {
        let cosine = VectorConfig::cosine(768);
        assert_eq!(cosine.dimensions, 768);
        assert_eq!(cosine.metric, DistanceMetric::Cosine);

        let euclidean = VectorConfig::euclidean(1536);
        assert_eq!(euclidean.dimensions, 1536);
        assert_eq!(euclidean.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_brute_force_knn() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0, 0.0].as_slice()),
            (NodeId::new(2), [1.0f32, 0.0, 0.0].as_slice()),
            (NodeId::new(3), [2.0f32, 0.0, 0.0].as_slice()),
            (NodeId::new(4), [3.0f32, 0.0, 0.0].as_slice()),
        ];

        let query = [0.5f32, 0.0, 0.0];
        let results = brute_force_knn(vectors.into_iter(), &query, 2, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 2);
        // Closest should be node 1 (dist 0.5) or node 2 (dist 0.5)
        assert!(results[0].0 == NodeId::new(1) || results[0].0 == NodeId::new(2));
    }

    #[test]
    fn test_brute_force_knn_empty() {
        let vectors: Vec<(NodeId, &[f32])> = vec![];
        let query = [0.0f32, 0.0];
        let results = brute_force_knn(vectors.into_iter(), &query, 10, DistanceMetric::Cosine);
        assert!(results.is_empty());
    }

    #[test]
    fn test_brute_force_knn_k_larger_than_n() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0].as_slice()),
            (NodeId::new(2), [1.0f32, 0.0].as_slice()),
        ];

        let query = [0.0f32, 0.0];
        let results = brute_force_knn(vectors.into_iter(), &query, 10, DistanceMetric::Euclidean);

        // Should return all 2 vectors, not 10
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_brute_force_knn_filtered() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0].as_slice()),
            (NodeId::new(2), [1.0f32, 0.0].as_slice()),
            (NodeId::new(3), [2.0f32, 0.0].as_slice()),
        ];

        let query = [0.0f32, 0.0];

        // Only consider even IDs
        let results = brute_force_knn_filtered(
            vectors.into_iter(),
            &query,
            10,
            DistanceMetric::Euclidean,
            |id| id.as_u64() % 2 == 0,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(2));
    }

    #[test]
    fn test_batch_distances() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0].as_slice()),
            (NodeId::new(2), [3.0f32, 4.0].as_slice()),
        ];

        let query = [0.0f32, 0.0];
        let results = batch_distances(vectors.into_iter(), &query, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, NodeId::new(1));
        assert!((results[0].1 - 0.0).abs() < 0.001);
        assert_eq!(results[1].0, NodeId::new(2));
        assert!((results[1].1 - 5.0).abs() < 0.001); // 3-4-5 triangle
    }

    // ── VectorIndexKind Quantized dispatch ────────────────────────────

    #[cfg(feature = "vector-index")]
    mod vector_index_kind_tests {
        use super::super::*;
        use std::collections::HashSet;

        /// Minimal accessor that always returns None (quantized indexes
        /// store vectors internally so the accessor is unused).
        struct NoopAccessor;
        impl VectorAccessor for NoopAccessor {
            fn get_vector(&self, _id: NodeId) -> Option<std::sync::Arc<[f32]>> {
                None
            }
        }

        fn build_quantized_kind(n: usize) -> VectorIndexKind {
            let config = HnswConfig::new(4, DistanceMetric::Euclidean);
            let q = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
            for i in 0..n {
                let vec: Vec<f32> = (0..4)
                    .map(|j| ((i * 4 + j) as f32) / (n * 4) as f32)
                    .collect();
                q.insert(NodeId::new(i as u64 + 1), &vec);
            }
            VectorIndexKind::Quantized(q)
        }

        #[test]
        fn quantized_kind_basic_ops() {
            let kind = build_quantized_kind(20);
            assert_eq!(kind.len(), 20);
            assert!(!kind.is_empty());
            assert!(kind.contains(NodeId::new(1)));
            assert!(!kind.contains(NodeId::new(999)));
        }

        #[test]
        fn quantized_kind_insert_and_search() {
            let kind = build_quantized_kind(30);
            let query = vec![0.5, 0.5, 0.0, 0.0];
            let results = kind.search(&query, 3, &NoopAccessor);
            assert_eq!(results.len(), 3);
        }

        #[test]
        fn quantized_kind_search_with_ef() {
            let kind = build_quantized_kind(30);
            let query = vec![0.5, 0.5, 0.0, 0.0];
            let results = kind.search_with_ef(&query, 3, 50, &NoopAccessor);
            assert_eq!(results.len(), 3);
        }

        #[test]
        fn quantized_kind_search_with_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (1..=10).map(NodeId::new).collect();
            let query = vec![0.1, 0.1, 0.0, 0.0];
            let results = kind.search_with_filter(&query, 5, &allowlist, &NoopAccessor);
            assert!(!results.is_empty());
            for (id, _) in &results {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_search_with_ef_and_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (5..=15).map(NodeId::new).collect();
            let query = vec![0.3, 0.3, 0.0, 0.0];
            let results = kind.search_with_ef_and_filter(&query, 3, 50, &allowlist, &NoopAccessor);
            for (id, _) in &results {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_batch_search() {
            let kind = build_quantized_kind(30);
            let queries = vec![vec![0.1, 0.0, 0.0, 0.0], vec![0.9, 0.9, 0.0, 0.0]];
            let results = kind.batch_search(&queries, 2, &NoopAccessor);
            assert_eq!(results.len(), 2);
        }

        #[test]
        fn quantized_kind_batch_search_with_ef() {
            let kind = build_quantized_kind(30);
            let queries = vec![vec![0.1, 0.0, 0.0, 0.0]];
            let results = kind.batch_search_with_ef(&queries, 2, 50, &NoopAccessor);
            assert_eq!(results.len(), 1);
        }

        #[test]
        fn quantized_kind_batch_search_with_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (1..=10).map(NodeId::new).collect();
            let queries = vec![vec![0.1, 0.0, 0.0, 0.0]];
            let results = kind.batch_search_with_filter(&queries, 5, &allowlist, &NoopAccessor);
            assert_eq!(results.len(), 1);
            for (id, _) in &results[0] {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_batch_search_with_ef_and_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (1..=15).map(NodeId::new).collect();
            let queries = vec![vec![0.2, 0.0, 0.0, 0.0]];
            let results =
                kind.batch_search_with_ef_and_filter(&queries, 3, 50, &allowlist, &NoopAccessor);
            assert_eq!(results.len(), 1);
            for (id, _) in &results[0] {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_remove() {
            let kind = build_quantized_kind(5);
            assert!(kind.remove(NodeId::new(1)));
            assert_eq!(kind.len(), 4);
            assert!(!kind.contains(NodeId::new(1)));
        }

        #[test]
        // reason: test indices 0..10 are non-negative
        #[allow(clippy::cast_sign_loss)]
        fn quantized_kind_snapshot_restore() {
            let kind = build_quantized_kind(10);
            let (entry, level, nodes) = kind.snapshot_topology();

            let config = HnswConfig::new(4, DistanceMetric::Euclidean);
            let kind2 = VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                config,
                QuantizationType::Scalar,
            ));
            for i in 0..10 {
                let vec: Vec<f32> = (0..4).map(|j| ((i * 4 + j) as f32) / 40.0).collect();
                kind2.insert(NodeId::new(i as u64 + 1), &vec, &NoopAccessor);
            }
            kind2.restore_topology(entry, level, nodes);
            assert_eq!(kind2.len(), 10);
        }

        #[test]
        fn quantized_kind_heap_memory() {
            let kind = build_quantized_kind(10);
            assert!(kind.heap_memory_bytes() > 0);
        }

        #[test]
        fn quantized_kind_type_accessors() {
            let kind = build_quantized_kind(1);
            assert!(kind.as_quantized().is_some());
            assert!(kind.as_hnsw().is_none());
            assert_eq!(kind.quantization_type(), Some(QuantizationType::Scalar));
        }

        #[test]
        fn from_trait_impls() {
            let config = HnswConfig::new(4, DistanceMetric::Euclidean);

            let hnsw = HnswIndex::new(config.clone());
            let kind: VectorIndexKind = hnsw.into();
            assert!(kind.as_hnsw().is_some());

            let quantized = QuantizedHnswIndex::new(config, QuantizationType::Binary);
            let kind2: VectorIndexKind = quantized.into();
            assert!(kind2.as_quantized().is_some());
            assert_eq!(kind2.quantization_type(), Some(QuantizationType::Binary));
        }
    }
}
