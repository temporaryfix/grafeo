//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! HNSW is a graph-based approximate nearest neighbor algorithm that builds
//! a multi-layer navigable small world graph. It provides:
//!
//! - **O(log n)** search complexity (approximate)
//! - **>95%** recall at k=10 with default settings
//!
//! This index is **topology-only**: it stores only the neighbor graph
//! structure, not the vectors themselves. Vectors are read on-the-fly
//! through a [`VectorAccessor`], which typically reads from property
//! storage, the single source of truth, halving memory usage for
//! vector workloads.
//!
//! # Algorithm Overview
//!
//! 1. **Multi-layer graph**: Nodes exist at multiple layers, with decreasing
//!    probability at higher layers (exponential distribution).
//! 2. **Greedy search**: Starting from the entry point at the top layer,
//!    greedily traverse to find the nearest node, then descend.
//! 3. **Beam search**: At the bottom layer, maintain a candidate set of
//!    size `ef` to find the k nearest neighbors.
//!
//! # Example
//!
//! ```
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
//! let vec1: Arc<[f32]> = vec![0.1f32; 384].into();
//! map.insert(NodeId::new(1), vec1.clone());
//! let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
//!
//! // Insert vectors
//! index.insert(NodeId::new(1), &vec1, &accessor);
//!
//! // Search for nearest neighbors
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10, &accessor);
//! ```
//!
//! # References
//!
//! - Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs" (2018)

use super::VectorAccessor;
use super::compute_distance;
use super::paged_topology::{MmapTopology, NeighborsIter as MmapNeighborsIter};
use crate::index::vector::HnswConfig;
use grafeo_common::types::NodeId;
use ordered_float::OrderedFloat;
#[cfg(feature = "lpg")]
use parking_lot::{Mutex, MutexGuard};
use parking_lot::{ReentrantMutex, ReentrantMutexGuard, RwLock, RwLockReadGuard};
use rand::RngExt;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "lpg")]
pub(crate) mod maintenance;

/// Serializes the rare transition from standalone to store-owned indexes.
///
/// A graph can own several indexes and must be able to validate every scope
/// before publishing any of them. Keeping this gate process-wide lets that
/// preflight and publication share one stable transition proof even when a
/// caller retained an alias to one of the indexes.
#[cfg(feature = "lpg")]
static HNSW_SCOPE_TRANSITION_GATE: Mutex<()> = Mutex::new(());

/// Stable PRNG algorithm identifier carried by Vector Store v3 snapshots.
///
/// The algorithm is deliberately owned here rather than delegated to `rand`:
/// `SmallRng` is allowed to change implementation between dependency releases
/// and does not expose a portable serialized continuation. SplitMix64 has a
/// single `u64` state and a fixed transition, so a restored index makes
/// exactly the same future level choices as the source index.
pub(super) const HNSW_RNG_SPLITMIX64: u8 = 1;

/// Hard safety ceiling for a sampled HNSW level.
///
/// Sixty-four layers are already far beyond the useful hierarchy depth for
/// any addressable graph, while keeping malformed/extreme public
/// configurations from turning one insertion into an unbounded allocation.
const HNSW_MAX_SAMPLED_LEVEL: usize = 63;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HnswRng {
    state: u64,
}

impl HnswRng {
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    const fn from_state(state: u64) -> Self {
        Self { state }
    }

    fn from_entropy() -> Self {
        Self::from_state(rand::rng().random())
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // Every output is in (0, 1] and is exactly representable,
        // independent of host architecture. Excluding zero is essential:
        // `-ln(0)` would otherwise make level selection unbounded.
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        (((self.next_u64() >> 11) + 1) as f64) * SCALE
    }
}

/// Proof that no HNSW index can change ownership scope until this value drops.
///
/// The constructor is deliberately private to the vector module. LPG sealing
/// uses the proof to make a multi-index preflight and apply failure-atomic.
#[cfg(feature = "lpg")]
pub(crate) struct HnswScopeTransition {
    _guard: MutexGuard<'static, ()>,
}

#[cfg(feature = "lpg")]
impl HnswScopeTransition {
    pub(super) fn acquire() -> Self {
        Self {
            _guard: HNSW_SCOPE_TRANSITION_GATE.lock(),
        }
    }
}

/// A neighbor entry in the HNSW graph.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Neighbor {
    id: NodeId,
    distance: f32,
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller distance = higher priority
        OrderedFloat(other.distance).cmp(&OrderedFloat(self.distance))
    }
}

/// A candidate for the max-heap during search (furthest first).
#[derive(Debug, Clone, Copy, PartialEq)]
struct FurthestCandidate {
    id: NodeId,
    distance: f32,
}

impl Eq for FurthestCandidate {}

impl PartialOrd for FurthestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FurthestCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

/// Materializes every node's neighbor lists from an [`MmapTopology`]
/// into the heap representation expected by `snapshot_topology`.
///
/// Used only during checkpoint of an mmap-backed index.
fn snapshot_mmap_topology(topo: &MmapTopology) -> Vec<(NodeId, Vec<Vec<NodeId>>)> {
    let mut out = Vec::with_capacity(topo.len());
    for id in topo.iter_node_ids() {
        let mut layers: Vec<Vec<NodeId>> = Vec::new();
        let mut layer = 0usize;
        while let Some(iter) = topo.neighbors_at(id, layer) {
            layers.push(iter.collect());
            layer += 1;
        }
        out.push((id, layers));
    }
    out
}

/// Node data stored in the HNSW index (topology only, no vector data).
#[derive(Debug, Clone)]
struct HnswNode {
    /// Neighbors at each layer (layer 0 is the bottom).
    /// The node's max layer is `neighbors.len() - 1`.
    neighbors: Vec<Vec<NodeId>>,
}

/// Complete, portable HNSW state used by Vector Store v3.
///
/// The topology-only v1/v2 representation omitted soft deletes and the PRNG
/// continuation. Restoring it could therefore resurrect deleted results and
/// make the next insertion choose a different layer. This state is the exact
/// logical post-image needed to continue construction deterministically.
#[derive(Debug, Clone)]
pub(super) struct HnswExactState {
    pub(super) config: HnswConfig,
    pub(super) entry_point: Option<NodeId>,
    pub(super) max_level: usize,
    pub(super) nodes: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    pub(super) deleted: Vec<NodeId>,
    pub(super) rng_state: u64,
}

/// Fully validated and allocated replacement state.
///
/// Construction may fail, but publication is only a sequence of lock-guarded
/// moves. This is what gives a v3 restore its failure-before-mutation boundary.
pub(super) struct PreparedHnswExactState {
    entry_point: Option<NodeId>,
    max_level: usize,
    nodes: HashMap<NodeId, HnswNode>,
    deleted: HashSet<NodeId>,
    rng: HnswRng,
}

fn hnsw_configs_match(left: &HnswConfig, right: &HnswConfig) -> bool {
    left.dimensions == right.dimensions
        && left.metric == right.metric
        && left.m == right.m
        && left.m_max == right.m_max
        && left.ef_construction == right.ef_construction
        && left.ef == right.ef
        && left.ml.to_bits() == right.ml.to_bits()
        && left.alpha.to_bits() == right.alpha.to_bits()
        && left.max_elements == right.max_elements
}

fn validate_hnsw_exact_state(
    state: &HnswExactState,
    target_config: &HnswConfig,
) -> std::result::Result<(), String> {
    if !hnsw_configs_match(&state.config, target_config) {
        return Err("HNSW snapshot configuration does not match target index".to_string());
    }
    let mut node_ids = HashSet::with_capacity(state.nodes.len());
    let mut previous = None;
    let mut observed_max_level = 0;
    for (id, layers) in &state.nodes {
        if !id.is_valid() {
            return Err("HNSW snapshot contains an invalid node ID".to_string());
        }
        if previous.is_some_and(|prior| prior >= *id) {
            return Err("HNSW snapshot node IDs are duplicate or non-canonical".to_string());
        }
        previous = Some(*id);
        if layers.is_empty() {
            return Err(format!("HNSW node {id} has no topology layer"));
        }
        observed_max_level = observed_max_level.max(layers.len() - 1);
        node_ids.insert(*id);
    }

    match (state.nodes.is_empty(), state.entry_point) {
        (true, None) if state.max_level == 0 && state.deleted.is_empty() => return Ok(()),
        (true, _) => {
            return Err("empty HNSW snapshot has an entry point, level, or delete set".to_string());
        }
        (false, None) => return Err("non-empty HNSW snapshot has no entry point".to_string()),
        (false, Some(_)) => {}
    }

    if observed_max_level != state.max_level {
        return Err(format!(
            "HNSW max level mismatch: header {}, topology {observed_max_level}",
            state.max_level
        ));
    }
    let entry_point = state
        .entry_point
        .expect("non-empty snapshot entry point checked above");
    let Some((_, entry_layers)) = state.nodes.iter().find(|(id, _)| *id == entry_point) else {
        return Err("HNSW entry point is absent from topology".to_string());
    };
    if entry_layers.len() != state.max_level + 1 {
        return Err("HNSW entry point does not occupy the maximum layer".to_string());
    }

    let layer_counts: HashMap<NodeId, usize> = state
        .nodes
        .iter()
        .map(|(id, layers)| (*id, layers.len()))
        .collect();
    for (id, layers) in &state.nodes {
        for (layer, neighbors) in layers.iter().enumerate() {
            let mut unique = HashSet::with_capacity(neighbors.len());
            for neighbor in neighbors {
                if !neighbor.is_valid() {
                    return Err(format!("HNSW node {id} has an invalid neighbor"));
                }
                if *neighbor == *id {
                    return Err(format!("HNSW node {id} contains a self-neighbor"));
                }
                let Some(neighbor_layers) = layer_counts.get(neighbor) else {
                    return Err(format!(
                        "HNSW node {id} references missing neighbor {neighbor}"
                    ));
                };
                if *neighbor_layers <= layer {
                    return Err(format!(
                        "HNSW node {id} references neighbor {neighbor} above its maximum layer"
                    ));
                }
                if !unique.insert(*neighbor) {
                    return Err(format!(
                        "HNSW node {id} has duplicate neighbor {neighbor} at layer {layer}"
                    ));
                }
            }
        }
    }

    previous = None;
    for id in &state.deleted {
        if !id.is_valid() || !node_ids.contains(id) {
            return Err("HNSW delete set contains an invalid or absent node".to_string());
        }
        if previous.is_some_and(|prior| prior >= *id) {
            return Err("HNSW delete set is duplicate or non-canonical".to_string());
        }
        previous = Some(*id);
    }
    Ok(())
}

/// Exact inverse of one HNSW insertion.
///
/// Existing neighbor lists are moved back into place during rollback, so the
/// rollback path performs no allocation and does not need a vector accessor.
/// This matters for an unwind guard: allocating or rebuilding from `Drop`
/// could turn one caught panic into a process-aborting double panic.
struct HnswInsertionUndo {
    entry_point: Option<NodeId>,
    max_level: usize,
    rng: HnswRng,
    was_deleted: bool,
    old_node: Option<HnswNode>,
    neighbor_layers: Vec<(NodeId, usize, Vec<NodeId>)>,
}

/// Restores an insertion if any user-supplied vector accessor panics midway
/// through HNSW construction. Successful inserts discard the undo payload.
struct HnswInsertRollback<'index> {
    index: &'index HnswIndex,
    id: NodeId,
    undo: Option<HnswInsertionUndo>,
}

impl HnswInsertRollback<'_> {
    fn disarm(mut self) {
        self.undo = None;
    }
}

impl Drop for HnswInsertRollback<'_> {
    fn drop(&mut self) {
        if let Some(undo) = self.undo.take() {
            self.index.rollback_insertion(self.id, undo);
        }
    }
}

/// Holds the retained-alias serialization gate and the store write-scope pin
/// for one complete public topology mutation.
pub(super) struct HnswMutationGuard<'index> {
    _publication: ReentrantMutexGuard<'index, ()>,
    _scope: RwLockReadGuard<'index, ()>,
}

/// Topology storage backend for [`HnswIndex`].
///
/// Two variants: [`Heap`](Self::Heap) is the build/mutation-friendly
/// representation (HashMap of node neighbor lists); [`Mmap`](Self::Mmap)
/// keeps a zero-copy [`MmapTopology`] base plus sparse aggregate-maintenance
/// overrides. Reads are unified through [`Self::neighbors_at`]; ordinary
/// insertion still requires [`Self::Heap`].
enum TopologyBackend {
    /// Heap-resident, build-and-mutation friendly.
    Heap(HashMap<NodeId, HnswNode>),
    /// Immutable `Bytes`-backed base plus private prepared-write overrides.
    /// Ordinary insertion rejects this backend; aggregate maintenance can
    /// install a sparse override without materializing the complete base.
    Mmap {
        base: MmapTopology,
        overrides: HashMap<NodeId, HnswNode>,
        additional_nodes: usize,
    },
}

/// Borrowed topology reads shared by live search and private sparse staging.
trait TopologyRead {
    fn len(&self) -> usize;
    fn neighbors_at(&self, id: NodeId, layer: usize) -> Option<HnswNeighborsIter<'_>>;
}

impl TopologyRead for TopologyBackend {
    fn len(&self) -> usize {
        Self::len(self)
    }

    fn neighbors_at(&self, id: NodeId, layer: usize) -> Option<HnswNeighborsIter<'_>> {
        Self::neighbors_at(self, id, layer)
    }
}

impl TopologyBackend {
    fn new_heap() -> Self {
        Self::Heap(HashMap::new())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::Heap(HashMap::with_capacity(capacity))
    }

    fn len(&self) -> usize {
        match self {
            Self::Heap(map) => map.len(),
            Self::Mmap {
                base,
                additional_nodes,
                ..
            } => base.len() + additional_nodes,
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Heap(map) => map.is_empty(),
            Self::Mmap {
                base, overrides, ..
            } => base.is_empty() && overrides.is_empty(),
        }
    }

    fn contains(&self, id: NodeId) -> bool {
        match self {
            Self::Heap(map) => map.contains_key(&id),
            Self::Mmap {
                base, overrides, ..
            } => overrides.contains_key(&id) || base.contains(id),
        }
    }

    /// Returns an iterator over the neighbors of `id` at the given
    /// `layer`, or `None` if absent.
    fn neighbors_at(&self, id: NodeId, layer: usize) -> Option<HnswNeighborsIter<'_>> {
        match self {
            Self::Heap(map) => map.get(&id).and_then(|node| {
                if layer < node.neighbors.len() {
                    Some(HnswNeighborsIter::Heap(node.neighbors[layer].iter()))
                } else {
                    None
                }
            }),
            Self::Mmap {
                base, overrides, ..
            } => {
                if let Some(node) = overrides.get(&id) {
                    node.neighbors
                        .get(layer)
                        .map(|neighbors| HnswNeighborsIter::Heap(neighbors.iter()))
                } else {
                    base.neighbors_at(id, layer).map(HnswNeighborsIter::Mmap)
                }
            }
        }
    }

    /// Borrow the heap representation for mutation, panicking if the
    /// backend is in [`Self::Mmap`] mode.
    fn as_heap_mut(&mut self) -> &mut HashMap<NodeId, HnswNode> {
        match self {
            Self::Heap(map) => map,
            Self::Mmap { .. } => {
                panic!("HNSW topology is in mmap mode; cannot mutate. Reload to RAM first.")
            }
        }
    }

    fn snapshot_nodes(&self) -> Vec<(NodeId, Vec<Vec<NodeId>>)> {
        match self {
            Self::Heap(map) => map
                .iter()
                .map(|(id, node)| (*id, node.neighbors.clone()))
                .collect(),
            Self::Mmap {
                base, overrides, ..
            } => {
                let mut rows = snapshot_mmap_topology(base);
                rows.retain(|(id, _)| !overrides.contains_key(id));
                rows.extend(
                    overrides
                        .iter()
                        .map(|(id, node)| (*id, node.neighbors.clone())),
                );
                rows
            }
        }
    }
}

/// Unified iterator over neighbor IDs from either backend.
///
/// Yields one [`NodeId`] per neighbor; preserves source order.
pub enum HnswNeighborsIter<'a> {
    /// Iterating a heap-stored `Vec<NodeId>`.
    Heap(std::slice::Iter<'a, NodeId>),
    /// Iterating an mmap-backed packed neighbor list.
    Mmap(MmapNeighborsIter<'a>),
}

impl Iterator for HnswNeighborsIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        match self {
            Self::Heap(iter) => iter.next().copied(),
            Self::Mmap(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Heap(iter) => iter.size_hint(),
            Self::Mmap(iter) => iter.size_hint(),
        }
    }
}

/// HNSW (Hierarchical Navigable Small World) index.
///
/// Thread-safe approximate nearest neighbor index supporting concurrent
/// reads and exclusive writes. This index is topology-only: vectors are
/// read through a [`VectorAccessor`] rather than stored internally.
///
/// # Soft-delete (MVCC)
///
/// Deleted nodes are retained in the graph topology as routing hops for
/// snapshot isolation. `remove(id)` marks `id` in `deleted` rather than
/// erasing it. Search excludes deleted nodes from *results* but still
/// traverses them during beam/greedy search, so live nodes reachable only
/// through a deleted hop remain findable. A separate GC pass (not in this
/// struct) compacts the graph by rebuilding without dead nodes.
pub struct HnswIndex {
    /// Index configuration.
    config: HnswConfig,
    /// Node storage. May be a heap HashMap (build/mutate path) or a
    /// zero-copy [`MmapTopology`] view (post-Phase-7c).
    nodes: RwLock<TopologyBackend>,
    /// Entry point for search (node at the highest layer).
    entry_point: RwLock<Option<NodeId>>,
    /// Current maximum layer in the index.
    max_level: RwLock<usize>,
    /// Random number generator for level selection.
    rng: RwLock<HnswRng>,
    /// Soft-deleted node IDs. These nodes remain in `nodes` as routing
    /// hops but are excluded from search results and from `len`/`contains`.
    deleted: RwLock<HashSet<NodeId>>,
    /// Store-scoped authority required once this index belongs to a sealed
    /// WAL-backed graph (`0` means the standalone index remains mutable).
    mutation_scope: AtomicU64,
    /// Stable logical-store and label/property-slot binding, distinct from the
    /// database write scope. Binding occurs at index installation so one
    /// retained index cannot back two stores or two derived-index meanings.
    #[cfg(feature = "lpg")]
    owner_store: AtomicU64,
    #[cfg(feature = "lpg")]
    owner_slot: AtomicU64,
    /// Serializes complete topology mutations, including prepared commits and
    /// insertion rollback, against mutations through retained aliases.
    publication_gate: Arc<ReentrantMutex<()>>,
    /// Parallel public search admission. Aggregate maintenance excludes these
    /// readers before acquiring graph entity/property writers, since searches
    /// may call graph-backed accessors while holding topology readers.
    reader_admission: RwLock<()>,
    /// The alias mutex is reentrant. Reject same-thread ordinary mutations
    /// while a private aggregate postimage depends on unchanged live state.
    maintenance_active: std::sync::atomic::AtomicBool,
    /// Shared for complete logical mutations, exclusive while ownership scope
    /// is published. This closes the check-then-mutate race for retained aliases.
    mutation_scope_gate: RwLock<()>,
}

impl HnswIndex {
    /// Creates a new empty HNSW index with the given configuration.
    #[must_use]
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(TopologyBackend::new_heap()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(HnswRng::from_entropy()),
            deleted: RwLock::new(HashSet::new()),
            mutation_scope: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_store: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_slot: AtomicU64::new(0),
            publication_gate: Arc::new(ReentrantMutex::new(())),
            reader_admission: RwLock::new(()),
            maintenance_active: std::sync::atomic::AtomicBool::new(false),
            mutation_scope_gate: RwLock::new(()),
        }
    }

    /// Creates a new HNSW index with pre-allocated capacity.
    ///
    /// Use this when you know the approximate number of vectors upfront
    /// to avoid HashMap rehashing during bulk insertion.
    #[must_use]
    pub fn with_capacity(config: HnswConfig, capacity: usize) -> Self {
        Self {
            config,
            nodes: RwLock::new(TopologyBackend::with_capacity(capacity)),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(HnswRng::from_entropy()),
            deleted: RwLock::new(HashSet::new()),
            mutation_scope: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_store: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_slot: AtomicU64::new(0),
            publication_gate: Arc::new(ReentrantMutex::new(())),
            reader_admission: RwLock::new(()),
            maintenance_active: std::sync::atomic::AtomicBool::new(false),
            mutation_scope_gate: RwLock::new(()),
        }
    }

    /// Creates a new HNSW index with a fixed seed for reproducible results.
    #[must_use]
    pub fn with_seed(config: HnswConfig, seed: u64) -> Self {
        Self {
            config,
            nodes: RwLock::new(TopologyBackend::new_heap()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(HnswRng::from_state(seed)),
            deleted: RwLock::new(HashSet::new()),
            mutation_scope: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_store: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_slot: AtomicU64::new(0),
            publication_gate: Arc::new(ReentrantMutex::new(())),
            reader_admission: RwLock::new(()),
            maintenance_active: std::sync::atomic::AtomicBool::new(false),
            mutation_scope_gate: RwLock::new(()),
        }
    }

    #[cfg(all(test, feature = "lpg"))]
    fn seal_with_scope(&self, scope: u64) -> bool {
        let transition = HnswScopeTransition::acquire();
        self.seal_with_scope_under_transition(scope, &transition)
    }

    #[cfg(feature = "lpg")]
    pub(super) fn binding_is_compatible(
        &self,
        owner: u64,
        slot: u64,
        _transition: &HnswScopeTransition,
    ) -> bool {
        if owner == 0 || slot == 0 {
            return false;
        }
        let current_owner = self.owner_store.load(Ordering::Acquire);
        let current_slot = self.owner_slot.load(Ordering::Acquire);
        (current_owner == 0 && current_slot == 0)
            || (current_owner == owner && current_slot == slot)
    }

    #[cfg(feature = "lpg")]
    pub(super) fn is_bound_to(
        &self,
        owner: u64,
        slot: u64,
        _transition: &HnswScopeTransition,
    ) -> bool {
        owner != 0
            && slot != 0
            && self.owner_store.load(Ordering::Acquire) == owner
            && self.owner_slot.load(Ordering::Acquire) == slot
    }

    #[cfg(feature = "lpg")]
    pub(super) fn bind_under_transition(
        &self,
        owner: u64,
        slot: u64,
        _transition: &HnswScopeTransition,
    ) -> bool {
        if owner == 0 || slot == 0 {
            return false;
        }
        let current_owner = self.owner_store.load(Ordering::Acquire);
        let current_slot = self.owner_slot.load(Ordering::Acquire);
        match (current_owner, current_slot) {
            (0, 0) => {
                // Every binding transition retains the process-wide token, so
                // legitimate readers observe only stable pairs. Publish owner
                // last with Release so it also publishes the slot.
                self.owner_slot.store(slot, Ordering::Relaxed);
                self.owner_store.store(owner, Ordering::Release);
                true
            }
            (existing_owner, existing_slot) => existing_owner == owner && existing_slot == slot,
        }
    }

    #[cfg(feature = "lpg")]
    pub(super) fn scope_is_unsealed(&self, _transition: &HnswScopeTransition) -> bool {
        self.mutation_scope.load(Ordering::Acquire) == 0
    }

    #[cfg(feature = "lpg")]
    pub(super) fn scope_is_compatible(
        &self,
        scope: u64,
        _transition: &HnswScopeTransition,
    ) -> bool {
        let current = self.mutation_scope.load(Ordering::Acquire);
        scope != 0 && (current == 0 || current == scope)
    }

    #[cfg(feature = "lpg")]
    pub(super) fn seal_with_scope_under_transition(
        &self,
        scope: u64,
        _transition: &HnswScopeTransition,
    ) -> bool {
        if scope == 0 {
            return false;
        }
        let _active_mutations = self.mutation_scope_gate.write();
        match self
            .mutation_scope
            .compare_exchange(0, scope, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(existing) => existing == scope,
        }
    }

    pub(super) fn publication_gate(&self) -> Arc<ReentrantMutex<()>> {
        Arc::clone(&self.publication_gate)
    }

    fn rollback_insertion(&self, id: NodeId, undo: HnswInsertionUndo) {
        let mut nodes = self.nodes.write();
        let TopologyBackend::Heap(nodes) = &mut *nodes else {
            // A retained publication guard excludes topology replacement while
            // an undo is live, so this branch is unreachable without internal
            // corruption. Never allocate or panic from an unwind path.
            return;
        };

        if let Some(old_node) = undo.old_node {
            nodes.insert(id, old_node);
        } else {
            nodes.remove(&id);
        }
        for (neighbor, layer, old_neighbors) in undo.neighbor_layers.into_iter().rev() {
            if let Some(node) = nodes.get_mut(&neighbor)
                && let Some(neighbors) = node.neighbors.get_mut(layer)
            {
                *neighbors = old_neighbors;
            }
        }
        *self.entry_point.write() = undo.entry_point;
        *self.max_level.write() = undo.max_level;
        *self.rng.write() = undo.rng;
        let mut deleted = self.deleted.write();
        if undo.was_deleted {
            deleted.insert(id);
        } else {
            deleted.remove(&id);
        }
    }

    pub(super) fn pin_mutation(&self) -> Option<HnswMutationGuard<'_>> {
        let publication = self.publication_gate.lock();
        if self.maintenance_active.load(Ordering::Acquire) {
            return None;
        }
        // Mutators such as `gc` and `batch_insert` call other mutators. A
        // recursive shared acquisition keeps their entire public operation
        // linearized even when a scope transition is already waiting.
        let scope_guard = self.mutation_scope_gate.read_recursive();
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope == 0
            || std::num::NonZeroU64::new(scope).is_some_and(crate::graph::write_permit::is_held)
        {
            Some(HnswMutationGuard {
                _publication: publication,
                _scope: scope_guard,
            })
        } else {
            None
        }
    }

    /// Returns the index configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Quantized wrappers retain this same admission through auxiliary reads
    /// and graph callbacks, outside the recursive admission of inner searches.
    pub(super) fn admit_reader(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.reader_admission.read_recursive()
    }

    /// Returns the number of **live** (non-deleted) vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes
            .read()
            .len()
            .saturating_sub(self.deleted.read().len())
    }

    /// Returns true if there are no live (non-deleted) vectors in the index.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot the topology for serialization.
    ///
    /// Returns (entry_point, max_level, node_neighbors) where node_neighbors
    /// is a vec of (NodeId, neighbor_layers).
    ///
    /// Works on both heap and mmap backends; the mmap path materializes
    /// neighbor `Vec`s on the fly (used during checkpoint when the
    /// in-memory index is mmap-backed but needs to be re-serialized).
    #[must_use]
    pub fn snapshot_topology(&self) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        let nodes = self.nodes.read();
        let entry_point = *self.entry_point.read();
        let max_level = *self.max_level.read();

        let mut node_data = nodes.snapshot_nodes();
        node_data.sort_by_key(|(id, _)| *id);

        (entry_point, max_level, node_data)
    }

    /// Captures every state component that affects present or future HNSW
    /// behaviour.
    ///
    /// The publication gate makes the multi-lock read coherent.
    pub(super) fn snapshot_exact(&self) -> std::result::Result<HnswExactState, String> {
        let _publication = self.publication_gate.lock();
        let _scope = self.mutation_scope_gate.read_recursive();
        let nodes = self.nodes.read();
        let entry_point = *self.entry_point.read();
        let max_level = *self.max_level.read();
        let rng_state = self.rng.read().state;
        let mut deleted: Vec<NodeId> = self.deleted.read().iter().copied().collect();

        let mut node_data = nodes.snapshot_nodes();
        node_data.sort_by_key(|(id, _)| *id);
        deleted.sort_unstable();

        let state = HnswExactState {
            config: self.config.clone(),
            entry_point,
            max_level,
            nodes: node_data,
            deleted,
            rng_state,
        };
        validate_hnsw_exact_state(&state, &self.config)?;
        Ok(state)
    }

    /// Validates and allocates an exact replacement without changing this
    /// index. The target configuration is immutable, so it must match every
    /// persisted field bit-for-bit rather than being partially reconstructed.
    pub(super) fn prepare_exact_restore(
        &self,
        state: HnswExactState,
    ) -> std::result::Result<PreparedHnswExactState, String> {
        self.validate_exact_state(&state)?;

        let nodes = state
            .nodes
            .into_iter()
            .map(|(id, neighbors)| (id, HnswNode { neighbors }))
            .collect();
        let deleted = state.deleted.into_iter().collect();
        Ok(PreparedHnswExactState {
            entry_point: state.entry_point,
            max_level: state.max_level,
            nodes,
            deleted,
            rng: HnswRng::from_state(state.rng_state),
        })
    }

    pub(super) fn validate_exact_state(
        &self,
        state: &HnswExactState,
    ) -> std::result::Result<(), String> {
        validate_hnsw_exact_state(state, &self.config)
    }

    /// Permanently seals a detached historical clone against mutation.
    ///
    /// The fresh scope is intentionally discarded, making the clone a read
    /// capability only. This is used when a retained compact generation needs
    /// the exact pre-handoff index image for old readers.
    #[cfg(feature = "compact-store")]
    pub(super) fn freeze_exact_read_snapshot(&self) {
        let frozen_scope = crate::graph::write_permit::WriteAuthority::new()
            .scope()
            .get();
        let _active_mutations = self.mutation_scope_gate.write();
        let previous = self.mutation_scope.swap(frozen_scope, Ordering::AcqRel);
        debug_assert_eq!(previous, 0, "only a fresh exact clone may be frozen");
    }

    /// Publishes a prevalidated replacement while the caller retains this
    /// index's mutation guard.
    pub(super) fn apply_prepared_exact_restore(&self, state: PreparedHnswExactState) {
        // Readers acquire `nodes` first, so retaining all state write locks and
        // publishing the node map last on unlock prevents a mixed old/new read.
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();
        let mut rng = self.rng.write();
        let mut deleted = self.deleted.write();

        *nodes = TopologyBackend::Heap(state.nodes);
        *entry_point = state.entry_point;
        *max_level = state.max_level;
        *rng = state.rng;
        *deleted = state.deleted;
    }

    /// Restore topology from a snapshot. Replaces all current data.
    ///
    /// Always switches the backend to the heap representation; subsequent
    /// mutations work without needing a reload. Equivalent to constructing
    /// a fresh index and calling `insert` for each node, but skips the
    /// graph-build cost.
    ///
    /// Also clears the soft-deleted set, since the snapshot is a clean
    /// point-in-time image with no pending deletes.
    pub fn restore_topology(
        &self,
        entry_point: Option<NodeId>,
        max_level: usize,
        node_data: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let mut backend = self.nodes.write();
        let mut fresh: HashMap<NodeId, HnswNode> = HashMap::with_capacity(node_data.len());
        for (id, neighbors) in node_data {
            fresh.insert(id, HnswNode { neighbors });
        }
        *backend = TopologyBackend::Heap(fresh);
        *self.entry_point.write() = entry_point;
        *self.max_level.write() = max_level;
        self.deleted.write().clear();
    }

    /// Adopt a [`MmapTopology`] as the topology backend (Phase 7c).
    ///
    /// Replaces any existing topology with a zero-copy view of the
    /// given mmap-backed buffer. Reads through the backend will serve
    /// from the [`bytes::Bytes`] without rebuilding a `HashMap`.
    /// Ordinary [`Self::insert`] rejects this backend until it is reloaded
    /// into RAM via [`Self::restore_topology`]. Soft deletion and internal
    /// prepared aggregate maintenance do not require a full RAM copy.
    ///
    /// `entry_point` and `max_level` are taken from the topology header.
    pub fn adopt_mmap_topology(&self, topo: MmapTopology) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let entry_point = topo.entry_point();
        let max_level = topo.max_level();
        let mut backend = self.nodes.write();
        *backend = TopologyBackend::Mmap {
            base: topo,
            overrides: HashMap::new(),
            additional_nodes: 0,
        };
        *self.entry_point.write() = entry_point;
        *self.max_level.write() = max_level;
    }

    /// Returns true if the backend is currently mmap-backed.
    #[must_use]
    pub fn is_mmap_backed(&self) -> bool {
        matches!(*self.nodes.read(), TopologyBackend::Mmap { .. })
    }

    /// Returns estimated heap memory in bytes for the HNSW topology.
    ///
    /// In mmap mode, counts mutable override allocations only; the immutable
    /// base neighbor data remains in the mmap.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        let nodes = self.nodes.read();
        match &*nodes {
            TopologyBackend::Heap(map) | TopologyBackend::Mmap { overrides: map, .. } => {
                let map_overhead = map.capacity()
                    * (std::mem::size_of::<NodeId>() + std::mem::size_of::<HnswNode>() + 1);
                let mut node_bytes = 0usize;
                for node in map.values() {
                    node_bytes += node.neighbors.capacity() * std::mem::size_of::<Vec<NodeId>>();
                    for layer in &node.neighbors {
                        node_bytes += layer.capacity() * std::mem::size_of::<NodeId>();
                    }
                }
                map_overhead + node_bytes
            }
        }
    }

    /// Inserts a vector with the given ID into the index.
    ///
    /// The vector is used during insertion to find neighbors and build
    /// the graph topology, but is **not** stored in the index.
    ///
    /// # Panics
    ///
    /// Panics if the vector dimensions don't match the configuration.
    pub fn insert(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        assert_eq!(
            vector.len(),
            self.config.dimensions,
            "Vector dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            vector.len()
        );
        // Reject the immutable backend before advancing the insertion RNG.
        // The retained publication guard prevents a concurrent topology swap,
        // so a caught unsupported-mutation panic leaves retry state exact.
        assert!(
            !self.is_mmap_backed(),
            "HNSW topology is in mmap mode; cannot mutate. Reload to RAM first."
        );

        // Advancing the RNG is part of the insertion state: if a hostile
        // accessor unwinds later, retry must see the same topology decision.
        let rng_before = *self.rng.read();
        let sampled_level = self.random_level();
        let mut rollback = HnswInsertRollback {
            index: self,
            id,
            undo: None,
        };

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();

        // Insert path always operates on the heap backend; calling
        // `as_heap_mut` panics if the topology is mmap-backed. Reload
        // to RAM via `restore_topology` first if needed.

        // Capacity check + first-insertion path. Scoped so the mutable
        // borrow ends before the per-layer search loop reborrows `&nodes`.
        let level;
        {
            let nodes_map = nodes.as_heap_mut();
            let old_node = nodes_map.get(&id).cloned();
            // Re-embedding an existing identity keeps its established layer
            // height. In particular, replacing the entry point with a shorter
            // node would leave `max_level` naming a layer the entry point no
            // longer occupies. A genuinely new identity uses the next sampled
            // level as before.
            level = old_node
                .as_ref()
                .map_or(sampled_level, |node| node.neighbors.len().saturating_sub(1));

            rollback.undo = Some(HnswInsertionUndo {
                entry_point: *entry_point,
                max_level: *max_level,
                rng: rng_before,
                was_deleted: self.deleted.read().contains(&id),
                old_node: old_node.clone(),
                neighbor_layers: Vec::new(),
            });

            if let Some(max) = self.config.max_elements
                && !nodes_map.contains_key(&id)
            {
                let count = nodes_map.len();
                assert!(
                    count < max,
                    "HNSW index is full: max_elements={max}, current={count}"
                );
            }

            // Retain an existing identity's outgoing links long enough to seed
            // its reconnection search, including when it is the entry point.
            // Each rebuilt layer is replaced below. A new identity starts with
            // the usual empty topology.
            let node = old_node.unwrap_or_else(|| HnswNode {
                neighbors: vec![Vec::new(); level + 1],
            });

            // First insertion
            if entry_point.is_none() {
                self.deleted.write().remove(&id);
                nodes_map.insert(id, node);
                *entry_point = Some(id);
                *max_level = level;
                rollback.disarm();
                return;
            }

            nodes_map.insert(id, node);
        }

        let ep = entry_point.expect("entry_point confirmed Some above");
        let current_max_level = *max_level;

        // The exact prior node (normally absent) is already owned by
        // `rollback`; clear any prior soft delete before reconnecting it.
        self.deleted.write().remove(&id);

        // Search from top to the level above the new node's max layer.
        let mut current_ep = ep;
        for lc in (level + 1..=current_max_level).rev() {
            current_ep = self.search_layer_single(&*nodes, accessor, vector, current_ep, lc);
        }

        // For each layer from the new node's max layer down to 0
        for lc in (0..=level.min(current_max_level)).rev() {
            let m_max = if lc == 0 {
                self.config.m_max
            } else {
                self.config.m
            };

            // Find ef_construction nearest neighbors at this layer
            let neighbors: Vec<_> = self
                .search_layer(
                    &*nodes,
                    accessor,
                    vector,
                    current_ep,
                    self.config.ef_construction,
                    lc,
                )
                .into_iter()
                .filter(|neighbor| neighbor.id != id)
                .collect();

            // Select neighbors using diversity-aware heuristic
            let selected = self.select_neighbors_heuristic(accessor, &neighbors, m_max);

            // First pass: link new node + identify who needs pruning.
            // Scope the mutable borrow tightly.
            let mut needs_pruning: Vec<NodeId> = Vec::new();
            {
                let nodes_map = nodes.as_heap_mut();
                if let Some(new_node) = nodes_map.get_mut(&id) {
                    new_node.neighbors[lc].clone_from(&selected);
                }

                for &neighbor_id in &selected {
                    if let Some(neighbor) = nodes_map.get_mut(&neighbor_id)
                        && neighbor.neighbors.len() > lc
                        && !neighbor.neighbors[lc].contains(&id)
                    {
                        if let Some(undo) = rollback.undo.as_mut() {
                            // Clone and record before mutating. If allocation
                            // unwinds, the rollback guard still owns every
                            // earlier inverse and the current list is intact.
                            let old_neighbors = neighbor.neighbors[lc].clone();
                            undo.neighbor_layers.push((neighbor_id, lc, old_neighbors));
                        }
                        neighbor.neighbors[lc].push(id);

                        if neighbor.neighbors[lc].len() > m_max {
                            needs_pruning.push(neighbor_id);
                        }
                    }
                }
            }

            // Second pass: compute distances for pruning (immutable read).
            let mut prune_data: Vec<(NodeId, Vec<(NodeId, f32)>)> = Vec::new();
            {
                let nodes_map = nodes.as_heap_mut();
                for neighbor_id in &needs_pruning {
                    if let Some(neighbor) = nodes_map.get(neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        let Some(base_vec) = accessor.get_vector(*neighbor_id) else {
                            continue;
                        };
                        let distances: Vec<(NodeId, f32)> = neighbor.neighbors[lc]
                            .iter()
                            .map(|&nid| {
                                let dist = accessor
                                    .get_vector(nid)
                                    .map_or(f32::MAX, |v| self.vector_distance(&base_vec, &v));
                                (nid, dist)
                            })
                            .collect();
                        prune_data.push((*neighbor_id, distances));
                    }
                }
            }

            // Third pass: apply pruning (mutable borrow).
            {
                let nodes_map = nodes.as_heap_mut();
                for (neighbor_id, distances) in prune_data {
                    if let Some(neighbor) = nodes_map.get_mut(&neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        Self::prune_neighbors_with_distances(
                            &mut neighbor.neighbors[lc],
                            &distances,
                            m_max,
                        );
                    }
                }
            }

            // Update entry point for next layer
            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Update global entry point if needed
        if level > current_max_level {
            *entry_point = Some(id);
            *max_level = level;
        }
        rollback.disarm();
    }

    /// Garbage collects soft-deleted nodes that are no longer needed.
    ///
    /// Rebuilds the HNSW topology from its existing RNG continuation, retaining
    /// exactly nodes where `is_live(id)` returns `true`. Nodes that are
    /// soft-deleted and no longer retained (i.e., their
    /// delete epoch is at or below the GC horizon) are permanently removed
    /// from the topology; live soft-deleted nodes (above horizon) are
    /// retained as deleted routing nodes, not resurrected results.
    ///
    /// A full rebuild is used (HNSW has no cheap in-place hard-delete).
    /// GC is amortized and runs infrequently relative to inserts/deletes.
    ///
    /// # Errors
    ///
    /// Rejects denied mutation authority or an unavailable/invalid retained
    /// vector before publishing any replacement. Retained tombstones remain
    /// routing nodes, never live results. A no-removal collection is exact no-op.
    pub fn gc(
        &self,
        is_live: &dyn Fn(NodeId) -> bool,
        accessor: &dyn VectorAccessor,
    ) -> std::result::Result<(), String> {
        let _mutation = self
            .pin_mutation()
            .ok_or_else(|| "HNSW GC requires mutation authority".to_string())?;
        // Publication admission is reentrant: a caller's predicate/accessor
        // must not change the source while its private successor is prepared.
        // Reset before releasing the publication pin, including on unwind.
        struct Preparation<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Preparation<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        self.maintenance_active.store(true, Ordering::Release);
        let _preparation = Preparation(&self.maintenance_active);
        let mut retained_ids: Vec<NodeId> = {
            let nodes = self.nodes.read();
            match &*nodes {
                TopologyBackend::Heap(map) => map.keys().copied().collect(),
                TopologyBackend::Mmap {
                    base, overrides, ..
                } => base
                    .iter_node_ids()
                    .chain(overrides.keys().copied().filter(|id| !base.contains(*id)))
                    .collect(),
            }
        };
        // Record the caller's decision once, in canonical order. Rebuilding
        // from a hash-map iteration order or fresh entropy cannot be replayed
        // from an exact preimage and the same GC horizon.
        retained_ids.sort_unstable();
        let previous_count = retained_ids.len();
        retained_ids.retain(|id| is_live(*id));
        if retained_ids.len() == previous_count {
            return Ok(());
        }

        let mut vectors = HashMap::new();
        vectors
            .try_reserve(retained_ids.len())
            .map_err(|_| "cannot allocate HNSW GC vectors".to_string())?;
        for &id in &retained_ids {
            let vector = accessor
                .get_vector(id)
                .ok_or_else(|| format!("HNSW GC lacks retained vector for {id:?}"))?;
            if vector.len() != self.config.dimensions
                || vector.iter().any(|value| !value.is_finite())
            {
                return Err(format!("HNSW GC has invalid retained vector for {id:?}"));
            }
            vectors.insert(id, vector);
        }
        let retained_accessor = |id| vectors.get(&id).cloned();
        let fresh = HnswIndex::with_seed(self.config.clone(), 0);
        *fresh.rng.write() = *self.rng.read();
        for id in retained_ids {
            let vector = vectors
                .get(&id)
                .ok_or_else(|| "HNSW GC lost a prepared vector".to_string())?;
            fresh.insert(id, vector, &retained_accessor);
        }
        let mut deleted = self.deleted.read().clone();
        deleted.retain(|id| vectors.contains_key(id));
        let TopologyBackend::Heap(nodes) = fresh.nodes.into_inner() else {
            return Err("HNSW GC candidate is not writable topology".to_string());
        };
        // Reuse the exact installer's coherent multi-lock publication, including
        // the RNG continuation and surviving deletion set.
        self.apply_prepared_exact_restore(PreparedHnswExactState {
            nodes,
            entry_point: fresh.entry_point.into_inner(),
            max_level: fresh.max_level.into_inner(),
            rng: fresh.rng.into_inner(),
            deleted,
        });
        Ok(())
    }

    /// Searches for the k nearest neighbors to the query vector.
    ///
    /// Returns a vector of (NodeId, distance) pairs sorted by distance
    /// (closest first).
    ///
    /// # Panics
    ///
    /// Panics if the query vector dimensions don't match the configuration.
    #[must_use]
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.search_with_ef(query, k, self.config.ef, accessor)
    }

    /// Searches with a custom ef (beam width) parameter.
    ///
    /// Higher ef values give better recall at the cost of latency.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not match the configured `dimensions`.
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let _reader = self.reader_admission.read_recursive();
        assert_eq!(
            query.len(),
            self.config.dimensions,
            "Query dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            query.len()
        );

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.expect("entry_point confirmed Some above");

        // Greedy search from top layer to layer 1
        let mut current_ep = ep;
        for lc in (1..=max_level).rev() {
            current_ep = self.search_layer_single(&*nodes, accessor, query, current_ep, lc);
        }

        // Beam search at layer 0
        let ef_search = ef.max(k);
        let candidates = self.search_layer(&*nodes, accessor, query, current_ep, ef_search, 0);

        // Collect top-k, filtering soft-deleted nodes from results.
        // The beam still routed through deleted nodes above (connectivity),
        // but they must not appear in what is returned to the caller.
        let deleted = self.deleted.read();
        candidates
            .into_iter()
            .filter(|n| !deleted.contains(&n.id))
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect()
    }

    /// Searches for the k nearest neighbors with an allowlist filter.
    ///
    /// Only nodes in the `allowlist` can appear in results. The HNSW graph
    /// is still fully traversed for connectivity; the filter only restricts
    /// the result set. The search beam width (`ef`) is automatically scaled
    /// based on the allowlist selectivity to maintain recall.
    ///
    /// Returns an empty vector if the allowlist is empty.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        if allowlist.is_empty() {
            return Vec::new();
        }
        // Auto-scale ef based on selectivity ratio.
        // Use live count (excludes soft-deleted) so selectivity is accurate.
        let total = self.len();
        let selectivity = if total == 0 {
            1.0
        } else {
            (allowlist.len() as f64 / total as f64).max(0.01)
        };
        // reason: ef scaled by selectivity is non-negative and bounded by .min(total)
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ef_scaled = ((self.config.ef as f64 / selectivity).ceil() as usize)
            .min(total)
            .max(k);
        self.search_with_ef_and_filter(query, k, ef_scaled, allowlist, accessor)
    }

    /// Searches with a custom ef (beam width) and an allowlist filter.
    ///
    /// Only nodes in the `allowlist` can appear in results. Higher ef values
    /// give better recall at the cost of latency.
    ///
    /// Returns an empty vector if the allowlist is empty.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not match the configured `dimensions`.
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let _reader = self.reader_admission.read_recursive();
        if allowlist.is_empty() {
            return Vec::new();
        }

        assert_eq!(
            query.len(),
            self.config.dimensions,
            "Query dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            query.len()
        );

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.expect("entry_point confirmed Some above");

        // Greedy search from top layer to layer 1
        let mut current_ep = ep;
        for lc in (1..=max_level).rev() {
            current_ep = self.search_layer_single(&*nodes, accessor, query, current_ep, lc);
        }

        // Filtered beam search at layer 0
        let ef_search = ef.max(k);
        let candidates = self.search_layer_filtered(
            &*nodes, accessor, query, current_ep, ef_search, 0, allowlist,
        );

        // Collect top-k; also exclude soft-deleted nodes (beam traversed them,
        // but they must not appear in results).
        let deleted = self.deleted.read();
        candidates
            .into_iter()
            .filter(|n| !deleted.contains(&n.id))
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect()
    }

    /// Soft-deletes a vector from the index.
    ///
    /// The node is **retained** in the topology as a routing hop for
    /// snapshot isolation and graph connectivity. It is excluded from
    /// search results and from [`Self::len`] / [`Self::contains`].
    ///
    /// Returns `true` if the ID was present (and is now marked deleted).
    /// Returns `false` if the ID was not in the topology at all.
    ///
    /// Re-inserting a deleted ID via [`Self::insert`] un-deletes it.
    ///
    /// Works on both heap-backed and mmap-backed topologies (the topology
    /// is never mutated; only the in-memory `deleted` set is updated).
    pub fn remove(&self, id: NodeId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        // A node that is already soft-deleted is not "present" from the
        // caller's perspective — removing it again returns false.
        if self.deleted.read().contains(&id) {
            return false;
        }
        // Verify the node exists in the topology (works on both backends).
        if !self.nodes.read().contains(id) {
            return false;
        }
        // Mark as soft-deleted; topology and links are untouched.
        self.deleted.write().insert(id);
        true
    }

    /// Returns `true` if the index contains a **live** (non-deleted) vector
    /// with the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.read().contains(id) && !self.deleted.read().contains(&id)
    }

    /// Returns `true` if the topology contains `id`, regardless of whether
    /// it has been soft-deleted.
    ///
    /// Used by GC passes and tests to verify the node was retained as a
    /// routing hop after soft-deletion.
    #[must_use]
    pub fn contains_including_deleted(&self, id: NodeId) -> bool {
        self.nodes.read().contains(id)
    }

    #[cfg(all(test, feature = "lpg"))]
    pub(super) fn state_counts(&self) -> (usize, usize) {
        (self.nodes.read().len(), self.deleted.read().len())
    }

    /// Generates a random level for a new node.
    ///
    /// `libm` is pinned because the standard library does not promise that
    /// transcendental functions produce identical results on every target.
    /// Combined with the serialized SplitMix64 continuation, this makes the
    /// level stream portable rather than merely repeatable on one machine.
    fn random_level(&self) -> usize {
        let mut rng = self.rng.write();
        Self::sample_level(&mut rng, self.config.ml)
    }

    fn sample_level(rng: &mut HnswRng, ml: f64) -> usize {
        let r = rng.next_f64();
        let sampled = -libm::log(r) * ml;
        if sampled.is_nan() || sampled <= 0.0 {
            return 0;
        }
        if sampled >= HNSW_MAX_SAMPLED_LEVEL as f64 {
            return HNSW_MAX_SAMPLED_LEVEL;
        }
        // Even a publicly constructed extreme configuration cannot request an
        // allocation beyond the useful HNSW hierarchy depth.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            libm::floor(sampled) as usize
        }
    }

    /// Single-element greedy search at a layer.
    fn search_layer_single(
        &self,
        nodes: &impl TopologyRead,
        accessor: &impl VectorAccessor,
        query: &[f32],
        ep: NodeId,
        layer: usize,
    ) -> NodeId {
        let mut current = ep;
        let mut current_dist = self.node_distance(accessor, query, ep);

        loop {
            let mut changed = false;

            if let Some(neighbors) = nodes.neighbors_at(current, layer) {
                for neighbor in neighbors {
                    let dist = self.node_distance(accessor, query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Beam search at a layer, returning ef nearest neighbors.
    fn search_layer(
        &self,
        nodes: &impl TopologyRead,
        accessor: &impl VectorAccessor,
        query: &[f32],
        ep: NodeId,
        ef: usize,
        layer: usize,
    ) -> Vec<Neighbor> {
        let ep_dist = self.node_distance(accessor, query, ep);

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });

        // Max-heap of current best (furthest = top)
        let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        results.push(FurthestCandidate {
            id: ep,
            distance: ep_dist,
        });

        let mut visited: HashSet<NodeId> =
            HashSet::with_capacity(nodes.len().min(ef.saturating_mul(2)));
        visited.insert(ep);

        while let Some(current) = candidates.pop() {
            // If the closest candidate is further than the furthest result, stop
            if let Some(furthest) = results.peek()
                && current.distance > furthest.distance
                && results.len() >= ef
            {
                break;
            }

            // Explore neighbors
            if let Some(neighbors) = nodes.neighbors_at(current.id, layer) {
                for neighbor in neighbors {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let dist = self.node_distance(accessor, query, neighbor);

                    // Add to results if closer than furthest, or if we have room
                    let should_add =
                        results.len() < ef || results.peek().map_or(true, |f| dist < f.distance);

                    if should_add {
                        candidates.push(Neighbor {
                            id: neighbor,
                            distance: dist,
                        });
                        results.push(FurthestCandidate {
                            id: neighbor,
                            distance: dist,
                        });

                        // Keep only ef results
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<Neighbor> = results
            .into_iter()
            .map(|fc| Neighbor {
                id: fc.id,
                distance: fc.distance,
            })
            .collect();
        result_vec.sort_by_key(|a| OrderedFloat(a.distance));
        result_vec
    }

    /// Beam search at a layer with an allowlist filter on the result set.
    ///
    /// All nodes are visited for graph traversal (neighbor links followed),
    /// but only nodes in the `allowlist` can enter the result set. This
    /// preserves HNSW connectivity while restricting which nodes are returned.
    #[allow(clippy::too_many_arguments)]
    fn search_layer_filtered(
        &self,
        nodes: &impl TopologyRead,
        accessor: &impl VectorAccessor,
        query: &[f32],
        ep: NodeId,
        ef: usize,
        layer: usize,
        allowlist: &HashSet<NodeId>,
    ) -> Vec<Neighbor> {
        let ep_dist = self.node_distance(accessor, query, ep);

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });

        // best_seen tracks ALL visited candidates (for traversal termination)
        let mut best_seen: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        best_seen.push(FurthestCandidate {
            id: ep,
            distance: ep_dist,
        });

        // results only holds allowlisted nodes
        let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        if allowlist.contains(&ep) {
            results.push(FurthestCandidate {
                id: ep,
                distance: ep_dist,
            });
        }

        let mut visited: HashSet<NodeId> =
            HashSet::with_capacity(nodes.len().min(ef.saturating_mul(4)));
        visited.insert(ep);

        while let Some(current) = candidates.pop() {
            // Terminate when best candidate is worse than worst in best_seen
            if let Some(furthest) = best_seen.peek()
                && current.distance > furthest.distance
                && best_seen.len() >= ef
            {
                break;
            }

            // Explore neighbors
            if let Some(neighbors) = nodes.neighbors_at(current.id, layer) {
                for neighbor in neighbors {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let dist = self.node_distance(accessor, query, neighbor);

                    // Update best_seen for traversal guidance
                    let should_explore = best_seen.len() < ef
                        || best_seen.peek().map_or(true, |f| dist < f.distance);

                    if should_explore {
                        candidates.push(Neighbor {
                            id: neighbor,
                            distance: dist,
                        });
                        best_seen.push(FurthestCandidate {
                            id: neighbor,
                            distance: dist,
                        });
                        while best_seen.len() > ef {
                            best_seen.pop();
                        }
                    }

                    // Only add to results if in allowlist
                    if allowlist.contains(&neighbor) {
                        let should_add = results.len() < ef
                            || results.peek().map_or(true, |f| dist < f.distance);
                        if should_add {
                            results.push(FurthestCandidate {
                                id: neighbor,
                                distance: dist,
                            });
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<Neighbor> = results
            .into_iter()
            .map(|fc| Neighbor {
                id: fc.id,
                distance: fc.distance,
            })
            .collect();
        result_vec.sort_by_key(|a| OrderedFloat(a.distance));
        result_vec
    }

    /// Selects neighbors using diversity-aware heuristic (Vamana-style).
    ///
    /// Instead of simply taking the M closest candidates, this checks whether
    /// each candidate is "covered" by an already-selected neighbor. A candidate
    /// is covered if any selected neighbor is closer to it than
    /// `alpha * distance(candidate, query)`. This preserves graph navigability
    /// by ensuring neighbors point to diverse regions of the space.
    fn select_neighbors_heuristic(
        &self,
        accessor: &impl VectorAccessor,
        candidates: &[Neighbor],
        m: usize,
    ) -> Vec<NodeId> {
        let alpha = self.config.alpha;
        let mut selected: Vec<(NodeId, Arc<[f32]>)> = Vec::with_capacity(m);

        for candidate in candidates {
            if selected.len() >= m {
                break;
            }
            let Some(cv) = accessor.get_vector(candidate.id) else {
                continue;
            };
            let covered = selected
                .iter()
                .any(|(_, sv)| self.vector_distance(&cv, sv) < alpha * candidate.distance);
            if !covered {
                selected.push((candidate.id, cv));
            }
        }

        selected.into_iter().map(|(id, _)| id).collect()
    }

    /// Prunes a neighbor list using distance-based diversity heuristic.
    ///
    /// Similar to `select_neighbors_heuristic` but operates on `(NodeId, f32)`
    /// distance pairs instead of `Neighbor` structs. Used during post-insert
    /// pruning where distances have already been computed.
    fn prune_neighbors_with_distances(
        neighbors: &mut Vec<NodeId>,
        distances: &[(NodeId, f32)],
        m: usize,
    ) {
        if neighbors.len() <= m {
            return;
        }

        *neighbors = Self::pruned_neighbors_with_distances(distances, m);
    }

    fn pruned_neighbors_with_distances(distances: &[(NodeId, f32)], m: usize) -> Vec<NodeId> {
        // Sort by distance
        let mut sorted: Vec<_> = distances.to_vec();
        sorted.sort_by_key(|a| OrderedFloat(a.1));

        sorted.into_iter().take(m).map(|(id, _)| id).collect()
    }

    /// Computes distance between two raw vectors using the configured metric.
    #[inline]
    fn vector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        compute_distance(a, b, self.config.metric)
    }

    /// Computes the distance between a query vector and a stored node.
    fn node_distance(&self, accessor: &impl VectorAccessor, query: &[f32], id: NodeId) -> f32 {
        accessor
            .get_vector(id)
            // Soft-deleted routing hops can outlive their indexed property.
            // An incompatible replacement is unavailable routing payload.
            .filter(|vector| vector.len() == query.len())
            .map_or(f32::MAX, |v| self.vector_distance(query, &v))
    }

    // ========================================================================
    // Batch Operations
    // ========================================================================

    /// Inserts multiple vectors in batch.
    ///
    /// This method inserts vectors sequentially into the HNSW graph structure
    /// but with optimized internal operations. For truly parallel construction
    /// of very large indexes, consider using multiple indexes and merging.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Iterator of (NodeId, vector) pairs to insert
    /// * `accessor` - Vector accessor for reading vectors by ID
    ///
    /// # Panics
    ///
    /// Panics if any vector dimensions don't match the configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
    /// use grafeo_common::types::NodeId;
    /// use std::sync::Arc;
    /// use std::collections::HashMap;
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let index = HnswIndex::new(config);
    ///
    /// let vectors: Vec<(NodeId, Vec<f32>)> = (0..100)
    ///     .map(|i| (NodeId::new(i), vec![0.1f32; 384]))
    ///     .collect();
    ///
    /// // Build an accessor backed by a HashMap
    /// let map: HashMap<NodeId, Arc<[f32]>> = vectors
    ///     .iter()
    ///     .map(|(id, v)| (*id, Arc::from(v.as_slice())))
    ///     .collect();
    /// let accessor = move |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
    ///
    /// index.batch_insert(vectors.iter().map(|(id, v)| (*id, v.as_slice())), &accessor);
    /// ```
    pub fn batch_insert<'a, I>(&self, vectors: I, accessor: &impl VectorAccessor)
    where
        I: IntoIterator<Item = (NodeId, &'a [f32])>,
    {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        for (id, vector) in vectors {
            self.insert(id, vector, accessor);
        }
    }

    /// Searches for k nearest neighbors for multiple queries in parallel.
    ///
    /// This method runs multiple searches concurrently using rayon, providing
    /// significant speedup when you have many queries to execute.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query vectors (as `Vec<f32>` or similar)
    /// * `k` - Number of nearest neighbors to return for each query
    /// * `accessor` - Vector accessor for reading vectors by ID
    ///
    /// # Returns
    ///
    /// Vector of results, one per query. Each result is a vector of
    /// (NodeId, distance) pairs sorted by distance.
    ///
    /// # Panics
    ///
    /// Panics if any query vector dimensions don't match the configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
    /// use grafeo_common::types::NodeId;
    /// use std::sync::Arc;
    /// use std::collections::HashMap;
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let index = HnswIndex::new(config);
    ///
    /// // Build an accessor (empty for this example)
    /// let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
    /// let accessor = move |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
    ///
    /// let queries: Vec<Vec<f32>> = vec![
    ///     vec![0.1f32; 384],
    ///     vec![0.2f32; 384],
    ///     vec![0.3f32; 384],
    /// ];
    ///
    /// let all_results = index.batch_search(&queries, 10, &accessor);
    /// assert_eq!(all_results.len(), 3);
    /// ```
    #[must_use]
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
    }

    /// Searches for k nearest neighbors for multiple queries in parallel.
    ///
    /// This variant accepts query vectors as slices.
    #[must_use]
    pub fn batch_search_slices(
        &self,
        queries: &[&[f32]],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
    }

    /// Searches with custom ef parameter for multiple queries in parallel.
    ///
    /// Higher ef values give better recall at the cost of latency.
    #[must_use]
    pub fn batch_search_with_ef(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef(query, k, ef, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef(query, k, ef, accessor))
                .collect()
        }
    }

    /// Searches for k nearest neighbors for multiple queries with an allowlist filter.
    ///
    /// The beam width is automatically scaled based on allowlist selectivity.
    #[must_use]
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_filter(query, k, allowlist, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_filter(query, k, allowlist, accessor))
                .collect()
        }
    }

    /// Searches with custom ef for multiple queries with an allowlist filter.
    #[must_use]
    pub fn batch_search_with_ef_and_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef_and_filter(query, k, ef, allowlist, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef_and_filter(query, k, ef, allowlist, accessor))
                .collect()
        }
    }
}

// ── Visible-predicate search (snapshot-aware) ──────────────────────────────
//
// EF widening factor: the beam is run with ef * VISIBLE_EF_FACTOR so that
// filtering invisible nodes still leaves ≥ k visible candidates.
pub(super) const VISIBLE_EF_FACTOR: usize = 4;

impl HnswIndex {
    /// Snapshot-aware predicate-filtered search.
    ///
    /// Traverses **all** graph neighbors (deleted + invisible nodes remain
    /// routing hops and are never skipped during traversal), but only
    /// returns candidates that satisfy `is_visible(id)`.
    ///
    /// Distances are computed via the supplied `accessor`, not from any
    /// internal storage. This lets the caller wire in a snapshot-aware
    /// accessor (returning as-of-snapshot vectors) at a later stage.
    ///
    /// If `accessor.get_vector(id)` returns `None` or an incompatible dimension
    /// for a candidate, it is skipped (no usable vector at that snapshot).
    ///
    /// The beam width is widened to `ef.max(k * VISIBLE_EF_FACTOR)` so
    /// that filtering invisible nodes still leaves ≥ k results.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not match the configured `dimensions`.
    ///
    /// # Returns
    ///
    /// Up to `k` (id, distance) pairs sorted by distance (ascending).
    #[must_use]
    pub fn search_visible(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        is_visible: &dyn Fn(NodeId) -> bool,
        accessor: &dyn VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.search_visible_candidates(
            query,
            k,
            ef.max(k.saturating_mul(VISIBLE_EF_FACTOR)),
            is_visible,
            accessor,
            true,
        )
    }

    /// Visibility-aware candidates with an already calibrated beam budget.
    /// Quantized search has widened its coarse candidate count itself; widening
    /// again here would multiply its traversal cost without changing its limit.
    /// Its stable internal accessor also needs no duplicate distance pass;
    /// public snapshot search retains its original final accessor evaluation.
    pub(super) fn search_visible_candidates(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        is_visible: &dyn Fn(NodeId) -> bool,
        accessor: &dyn VectorAccessor,
        rescore_candidates: bool,
    ) -> Vec<(NodeId, f32)> {
        let _reader = self.reader_admission.read_recursive();
        assert_eq!(
            query.len(),
            self.config.dimensions,
            "Query dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            query.len()
        );

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.expect("entry_point confirmed Some above");

        // The private beam helpers are generic over `impl VectorAccessor`
        // (monomorphised, require `Sized`). To accept a `&dyn VectorAccessor`
        // we wrap it in a thin newtype that *is* `Sized`.
        struct DynAccessorRef<'a>(&'a dyn VectorAccessor);
        impl VectorAccessor for DynAccessorRef<'_> {
            fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
                self.0.get_vector(id)
            }
        }
        let wrapped = DynAccessorRef(accessor);

        // Greedy descent from top layer to layer 1 — traverse through all nodes.
        let mut current_ep = ep;
        for lc in (1..=max_level).rev() {
            current_ep = self.search_layer_single(&*nodes, &wrapped, query, current_ep, lc);
        }

        let ef_search = ef.max(k);

        // Full beam at layer 0 — all neighbors traversed, no pruning.
        let candidates = self.search_layer(&*nodes, &wrapped, query, current_ep, ef_search, 0);

        // Collect visible candidates, scored by the supplied accessor.
        // If the accessor returns None for a node (no vector at this snapshot),
        // skip it.
        let mut results: Vec<(NodeId, f32)> = candidates
            .into_iter()
            .filter(|n| is_visible(n.id))
            .filter_map(|n| {
                if !rescore_candidates {
                    return Some((n.id, n.distance));
                }
                // Re-score with the passed accessor in case it differs from the
                // one used during graph traversal (the beam already used it, so
                // this is just a consistency pass — same cost, always correct).
                accessor
                    .get_vector(n.id)
                    .filter(|vector| vector.len() == query.len())
                    .map(|v| (n.id, self.vector_distance(query, &v)))
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("config", &self.config)
            .field("len", &self.len())
            .field("max_level", &*self.max_level.read())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "lpg")]
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::vector::DistanceMetric;
    use crate::index::vector::VectorIndexKind;
    #[cfg(feature = "lpg")]
    use std::sync::mpsc;
    #[cfg(feature = "lpg")]
    use std::time::Duration;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32) / (n * dim) as f32)
                    .collect()
            })
            .collect()
    }

    /// Builds an accessor backed by a HashMap.
    fn make_accessor(map: &HashMap<NodeId, Arc<[f32]>>) -> impl VectorAccessor + '_ {
        move |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() }
    }

    #[test]
    fn level_sampling_excludes_zero_and_bounds_extreme_public_configurations() {
        // SplitMix64's mix of zero is zero. Advancing this predecessor state
        // therefore exercised ln(0) before the open-closed conversion.
        let zero_output_seed = 0_u64.wrapping_sub(HnswRng::GAMMA);
        let mut rng = HnswRng::from_state(zero_output_seed);
        let sample = rng.next_f64();
        assert!(sample > 0.0 && sample <= 1.0);

        // `with_m(1)` is publicly constructible and produces an infinite ml.
        // Persistence must preserve it, while insertion remains bounded and
        // cannot turn that configuration into an unbounded allocation.
        let index = HnswIndex::with_seed(
            HnswConfig::new(1, DistanceMetric::Euclidean).with_m(1),
            zero_output_seed,
        );
        let vector = Arc::<[f32]>::from([1.0]);
        let accessor = |_id: NodeId| Some(Arc::clone(&vector));
        index.insert(NodeId::new(1), &vector, &accessor);
        let (_, level, nodes) = index.snapshot_topology();
        assert!(level <= HNSW_MAX_SAMPLED_LEVEL);
        assert!(nodes[0].1.len() <= HNSW_MAX_SAMPLED_LEVEL + 1);
        let state = index
            .snapshot_exact()
            .expect("public m=1 configuration remains checkpointable");
        let target = HnswIndex::with_seed(index.config().clone(), 9);
        target
            .prepare_exact_restore(state)
            .expect("public m=1 configuration remains restorable");
    }

    #[test]
    fn level_sampling_has_a_portable_pinned_sequence() {
        let mut config = HnswConfig::new(1, DistanceMetric::Euclidean);
        config.ml = 1.0;
        let index = HnswIndex::with_seed(config, 0x0123_4567_89ab_cdef);
        let levels: Vec<_> = (0..24).map(|_| index.random_level()).collect();
        assert_eq!(
            levels,
            [
                2, 0, 1, 0, 5, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 7, 0, 0, 1, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn accessor_unwind_restores_exact_topology_and_rng_for_retry() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean)
            .with_m(2)
            .with_m_max(1);
        let actual = HnswIndex::with_seed(config.clone(), 73);
        let expected = HnswIndex::with_seed(config, 73);
        let mut vectors = HashMap::new();
        for raw in 1_u64..=33 {
            let value = raw as f32 / 34.0;
            vectors.insert(
                NodeId::new(raw),
                Arc::<[f32]>::from([value, value, value, value]),
            );
        }
        let accessor = make_accessor(&vectors);
        for raw in 1_u64..=32 {
            let id = NodeId::new(raw);
            let vector = vectors.get(&id).expect("fixture vector");
            actual.insert(id, vector, &accessor);
            expected.insert(id, vector, &accessor);
        }
        let before = actual.snapshot_topology();
        let new_id = NodeId::new(33);
        let new_vector = vectors.get(&new_id).expect("new vector");
        let hostile_accessor = |id: NodeId| -> Option<Arc<[f32]>> {
            assert_ne!(
                id, new_id,
                "hostile accessor unwind after an existing neighbor was linked"
            );
            vectors.get(&id).cloned()
        };

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actual.insert(new_id, new_vector, &hostile_accessor);
        }));
        assert!(unwind.is_err(), "fixture must reach the hostile accessor");
        assert_eq!(actual.snapshot_topology(), before);

        actual.insert(new_id, new_vector, &accessor);
        expected.insert(new_id, new_vector, &accessor);
        assert_eq!(
            actual.snapshot_topology(),
            expected.snapshot_topology(),
            "caught unwind must restore RNG state as well as visible topology"
        );
    }

    #[test]
    fn existing_identity_insertion_unwind_restores_topology_and_rng() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean)
            .with_m(4)
            .with_ef_construction(32);
        let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            config,
            0x005a_6e1d,
        )));
        let hnsw = index.as_hnsw().expect("plain HNSW fixture");
        let vectors = RwLock::new(HashMap::<NodeId, Arc<[f32]>>::new());
        for raw in 1_u64..=24 {
            let value = raw as f32 / 24.0;
            let id = NodeId::new(raw);
            vectors
                .write()
                .insert(id, Arc::from([value, value, value, value]));
            let vector = vectors.read().get(&id).cloned().expect("fixture vector");
            let accessor = |candidate| vectors.read().get(&candidate).cloned();
            index.insert(id, &vector, &accessor);
        }

        let entry_point = hnsw.snapshot_topology().0;
        let id = (1_u64..=24)
            .map(NodeId::new)
            .find(|candidate| Some(*candidate) != entry_point)
            .expect("fixture has a non-entry-point identity");
        let before = hnsw.snapshot_exact().expect("exact recovered state");
        let assert_exact = |actual: &HnswExactState, expected: &HnswExactState| {
            assert!(hnsw_configs_match(&actual.config, &expected.config));
            assert_eq!(actual.entry_point, expected.entry_point);
            assert_eq!(actual.max_level, expected.max_level);
            assert_eq!(actual.nodes, expected.nodes);
            assert_eq!(actual.deleted, expected.deleted);
            assert_eq!(actual.rng_state, expected.rng_state);
        };
        vectors
            .write()
            .insert(id, Arc::from([-100.0, -100.0, -100.0, -100.0]));
        assert!(index.remove(id));
        let before_hostile_insert = hnsw
            .snapshot_exact()
            .expect("ordinary re-embedding pre-insert state");
        let accessor_calls = std::sync::atomic::AtomicUsize::new(0);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let vector = vectors
                .read()
                .get(&id)
                .cloned()
                .expect("replacement vector");
            let hostile_accessor = |candidate| {
                assert_ne!(
                    accessor_calls.fetch_add(1, Ordering::SeqCst),
                    1,
                    "hostile accessor panic after insertion work began"
                );
                vectors.read().get(&candidate).cloned()
            };
            index.insert(id, &vector, &hostile_accessor);
        }));
        assert!(
            unwind.is_err(),
            "hostile accessor must unwind existing-ID insertion"
        );
        assert_exact(
            &hnsw.snapshot_exact().expect("post-hostile exact state"),
            &before_hostile_insert,
        );

        // A normal retry after the caught unwind must execute the logical update.
        let vector = vectors
            .read()
            .get(&id)
            .cloned()
            .expect("replacement vector");
        let accessor = |candidate| vectors.read().get(&candidate).cloned();
        index.insert(id, &vector, &accessor);

        assert!(index.contains(id));
        let logically_updated = hnsw
            .snapshot_exact()
            .expect("logical same-identity update remains canonical");
        assert_ne!(
            logically_updated.rng_state, before.rng_state,
            "the logical update must execute a real rebuild"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn derived_index_binding_is_exact_and_zero_fails_closed() {
        let index = HnswIndex::new(HnswConfig::new(2, DistanceMetric::Euclidean));
        let transition = HnswScopeTransition::acquire();

        assert!(!index.binding_is_compatible(0, 1, &transition));
        assert!(!index.binding_is_compatible(1, 0, &transition));
        assert!(index.binding_is_compatible(11, 17, &transition));
        assert!(index.bind_under_transition(11, 17, &transition));
        assert!(index.is_bound_to(11, 17, &transition));
        assert!(index.bind_under_transition(11, 17, &transition));
        assert!(!index.binding_is_compatible(12, 17, &transition));
        assert!(!index.binding_is_compatible(11, 18, &transition));
        assert!(!index.bind_under_transition(12, 17, &transition));
        assert!(!index.bind_under_transition(11, 18, &transition));

        assert!(index.scope_is_unsealed(&transition));
        assert!(!index.scope_is_compatible(0, &transition));
        assert!(!index.seal_with_scope_under_transition(0, &transition));
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn scope_seal_waits_for_inflight_mutation_and_retained_alias_fails_closed() {
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        let index = Arc::new(HnswIndex::with_seed(
            HnswConfig::new(2, DistanceMetric::Euclidean),
            7,
        ));

        // Model a public mutation that has passed its authority check but has
        // not completed. Scope publication must wait for the retained proof,
        // rather than changing authorization underneath the operation.
        let in_flight = index
            .pin_mutation()
            .expect("a standalone index accepts a mutation");
        let (started_tx, started_rx) = mpsc::channel();
        let (sealed_tx, sealed_rx) = mpsc::channel();
        let sealing_index = Arc::clone(&index);
        let owner_scope = owner.scope().get();
        let sealing_thread = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let sealed = sealing_index.seal_with_scope(owner_scope);
            sealed_tx.send(sealed).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(
            sealed_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "scope publication must block behind an in-flight mutation"
        );
        assert_eq!(index.mutation_scope.load(Ordering::Acquire), 0);
        drop(in_flight);
        assert!(
            sealed_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("scope publication completes once the mutation retires")
        );
        sealing_thread.join().unwrap();

        let no_vectors = |_id: NodeId| -> Option<Arc<[f32]>> { None };
        index.insert(NodeId::new(1), &[1.0, 0.0], &no_vectors);
        assert!(
            !index.contains(NodeId::new(1)),
            "a raw retained alias must fail closed after sealing"
        );
        with_authority(&foreign, || {
            index.insert(NodeId::new(1), &[1.0, 0.0], &no_vectors);
        });
        assert!(
            !index.contains(NodeId::new(1)),
            "foreign authority must not authorize the retained alias"
        );
        with_authority(&owner, || {
            index.insert(NodeId::new(1), &[1.0, 0.0], &no_vectors);
        });
        assert!(index.contains(NodeId::new(1)));

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || {
                let panicking_accessor =
                    |_id: NodeId| -> Option<Arc<[f32]>> { panic!("injected accessor panic") };
                index.insert(NodeId::new(2), &[0.0, 1.0], &panicking_accessor);
            });
        }));
        assert!(panic.is_err());
        index.insert(NodeId::new(3), &[0.5, 0.5], &no_vectors);
        assert!(
            !index.contains(NodeId::new(3)),
            "caught panic must release both the mutation proof and owner authority"
        );
    }

    #[test]
    fn test_hnsw_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(
            index
                .search(&[0.0, 0.0, 0.0, 0.0], 10, &accessor)
                .is_empty()
        );
    }

    #[test]
    fn test_hnsw_single_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let v: Arc<[f32]> = vec![0.1, 0.2, 0.3, 0.4].into();
        map.insert(NodeId::new(1), v.clone());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &v, &accessor);

        assert_eq!(index.len(), 1);
        assert!(index.contains(NodeId::new(1)));
        assert!(!index.contains(NodeId::new(2)));

        let results = index.search(&[0.1, 0.2, 0.3, 0.4], 1, &accessor);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));
        assert!(results[0].1 < 0.001); // Near-zero distance
    }

    #[test]
    fn test_hnsw_multiple_inserts() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        assert_eq!(index.len(), 100);

        // Search for nearest neighbors
        let query = &vectors[50];
        let results = index.search(query, 5, &accessor);

        assert_eq!(results.len(), 5);
        // The closest should be the vector itself
        assert_eq!(results[0].0, NodeId::new(51));
        assert!(results[0].1 < 0.001);
    }

    #[test]
    fn test_hnsw_search_returns_sorted() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let query = [0.5, 0.5, 0.5, 0.5];
        let results = index.search(&query, 10, &accessor);

        // Verify sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_hnsw_remove() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![0.1, 0.2, 0.3, 0.4].into());
        map.insert(NodeId::new(2), vec![0.5, 0.6, 0.7, 0.8].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4], &accessor);
        index.insert(NodeId::new(2), &[0.5, 0.6, 0.7, 0.8], &accessor);

        assert_eq!(index.len(), 2);

        assert!(index.remove(NodeId::new(1)));
        assert_eq!(index.len(), 1);
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains(NodeId::new(2)));

        // Removing again returns false
        assert!(!index.remove(NodeId::new(1)));
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![1.0, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(3), vec![0.707, 0.707, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        // Insert normalized vectors
        index.insert(NodeId::new(1), &[1.0, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(3), &[0.707, 0.707, 0.0, 0.0], &accessor);

        // Query similar to node 1
        let results = index.search(&[0.9, 0.1, 0.0, 0.0], 3, &accessor);

        // Node 1 should be closest (most similar direction)
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_hnsw_ef_parameter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let query = [0.5, 0.5, 0.5, 0.5];

        // Higher ef should give same or better results
        let results_low = index.search_with_ef(&query, 5, 10, &accessor);
        let results_high = index.search_with_ef(&query, 5, 100, &accessor);

        assert_eq!(results_low.len(), 5);
        assert_eq!(results_high.len(), 5);

        // High ef should find equal or better (smaller) distances
        assert!(results_high[0].1 <= results_low[0].1);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions mismatch")]
    fn test_hnsw_dimension_mismatch_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3], &accessor); // Wrong dimension
    }

    #[test]
    fn test_hnsw_max_elements_accepts_within_limit() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_max_elements(3);
        let index = HnswIndex::new(config);
        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), Arc::from([1.0f32, 0.0, 0.0].as_slice()));
        map.insert(NodeId::new(1), Arc::from([0.0f32, 1.0, 0.0].as_slice()));
        map.insert(NodeId::new(2), Arc::from([0.0f32, 0.0, 1.0].as_slice()));
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor);
        assert_eq!(index.len(), 3);
    }

    #[test]
    #[should_panic(expected = "HNSW index is full")]
    fn test_hnsw_rejects_above_max_elements() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_max_elements(2);
        let index = HnswIndex::new(config);
        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), Arc::from([1.0f32, 0.0, 0.0].as_slice()));
        map.insert(NodeId::new(1), Arc::from([0.0f32, 1.0, 0.0].as_slice()));
        map.insert(NodeId::new(2), Arc::from([0.0f32, 0.0, 1.0].as_slice()));
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor); // Should panic
    }

    #[test]
    #[should_panic(expected = "Query dimensions mismatch")]
    fn test_hnsw_dimension_mismatch_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![0.1, 0.2, 0.3, 0.4].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4], &accessor);
        let _ = index.search(&[0.1, 0.2, 0.3], 1, &accessor); // Wrong dimension
    }

    #[test]
    fn test_hnsw_batch_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        let pairs: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.as_slice()))
            .collect();

        index.batch_insert(pairs, &accessor);

        assert_eq!(index.len(), 100);

        // Verify search still works
        let results = index.search(&vectors[50], 5, &accessor);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, NodeId::new(51));
    }

    #[test]
    fn test_hnsw_batch_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Batch search with 5 queries
        let queries: Vec<Vec<f32>> = (0..5).map(|i| vectors[i * 20].clone()).collect();

        let all_results = index.batch_search(&queries, 3, &accessor);

        assert_eq!(all_results.len(), 5);
        for (i, results) in all_results.iter().enumerate() {
            assert_eq!(results.len(), 3);
            // First result should be the query vector itself
            assert_eq!(results[0].0, NodeId::new((i * 20 + 1) as u64));
            assert!(results[0].1 < 0.001);
        }
    }

    #[test]
    fn test_hnsw_batch_search_with_ef() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let queries: Vec<Vec<f32>> = vec![vectors[25].clone(), vectors[75].clone()];

        // Search with higher ef for better recall
        let results = index.batch_search_with_ef(&queries, 5, 100, &accessor);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 5);
        assert_eq!(results[1].len(), 5);
    }

    #[test]
    fn test_hnsw_batch_search_empty_index() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        let queries = vec![vec![0.0f32, 0.0, 0.0, 0.0]];
        let results = index.batch_search(&queries, 10, &accessor);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    /// Brute-force k-NN for recall verification.
    fn brute_force_knn(
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::index::vector::compute_distance(query, v, metric)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn test_hnsw_recall_euclidean() {
        // 1000 vectors, 20 dimensions, matches ann-benchmarks random-xs profile
        let n = 1000;
        let dim = 20;
        let k = 10;
        let num_queries = 100;

        // Deterministic pseudo-random vectors via linear congruential generator
        let mut seed: u64 = 12345;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec, &accessor);
        }

        // Measure recall over num_queries random queries
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut total_recall = 0.0f64;
        for query in &queries {
            let ground_truth = brute_force_knn(&vectors, query, k, DistanceMetric::Euclidean);
            let gt_set: std::collections::HashSet<u64> =
                ground_truth.iter().map(|&i| i as u64).collect();

            let results = index.search_with_ef(query, k, 50, &accessor);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| id.as_u64()).collect();

            let overlap = gt_set.intersection(&found).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Recall {avg_recall:.3} is below 0.90 threshold at M=16/ef=50"
        );
    }

    #[test]
    fn test_hnsw_recall_cosine() {
        let n = 500;
        let dim = 20;
        let k = 10;
        let num_queries = 50;

        let mut seed: u64 = 67890;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let config = HnswConfig::new(dim, DistanceMetric::Cosine).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec, &accessor);
        }

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut total_recall = 0.0f64;
        for query in &queries {
            let ground_truth = brute_force_knn(&vectors, query, k, DistanceMetric::Cosine);
            let gt_set: std::collections::HashSet<u64> =
                ground_truth.iter().map(|&i| i as u64).collect();

            let results = index.search_with_ef(query, k, 50, &accessor);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| id.as_u64()).collect();

            let overlap = gt_set.intersection(&found).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Cosine recall {avg_recall:.3} is below 0.90 threshold at M=16/ef=50"
        );
    }

    #[test]
    fn test_diversity_pruning_prevents_clustering() {
        // Verify that diversity pruning selects diverse neighbors, not just closest
        let dim = 4;
        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(4);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![0.0, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.01, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.02, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(3), vec![0.03, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(4), vec![0.04, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(5), vec![0.0, 1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        // Insert a cluster of very similar vectors and one outlier
        index.insert(NodeId::new(0), &[0.0, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.01, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.02, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(3), &[0.03, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(4), &[0.04, 0.0, 0.0, 0.0], &accessor);
        // Outlier in a different direction
        index.insert(NodeId::new(5), &[0.0, 1.0, 0.0, 0.0], &accessor);

        // Search for the outlier; it should be findable
        let results = index.search(&[0.0, 0.9, 0.0, 0.0], 1, &accessor);
        assert_eq!(results[0].0, NodeId::new(5));
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_single_vector() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);

        let results = index.search(&[1.0, 0.0, 0.0], 1, &accessor);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(0));
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);

        // k=10 but only 2 vectors
        let results = index.search(&[1.0, 0.0, 0.0], 10, &accessor);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_index_search() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        let results = index.search(&[1.0, 0.0, 0.0], 5, &accessor);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_and_search() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 0.0, 1.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor);

        index.remove(NodeId::new(1));
        let results = index.search(&[0.0, 1.0, 0.0], 3, &accessor);
        // Removed node should not appear
        assert!(results.iter().all(|(id, _)| *id != NodeId::new(1)));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_duplicate_insert() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);

        // Update the accessor with the new vector for node 0
        let mut map2: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map2.insert(NodeId::new(0), vec![0.0, 1.0, 0.0].into());
        let accessor2 = make_accessor(&map2);

        index.insert(NodeId::new(0), &[0.0, 1.0, 0.0], &accessor2); // Same ID, different vector

        assert_eq!(index.len(), 1);
        // Should use the latest vector
        let results = index.search(&[0.0, 1.0, 0.0], 1, &accessor2);
        assert_eq!(results[0].0, NodeId::new(0));
    }

    #[test]
    fn test_search_with_ef_zero() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);

        // ef=0 should still return results (search uses max(ef, k))
        let results = index.search_with_ef(&[1.0, 0.0, 0.0], 1, 0, &accessor);
        // Behavior may vary but should not panic
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_all_metrics_search() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let config = HnswConfig::new(3, metric);
            let index = HnswIndex::new(config);

            let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
            map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
            map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
            let accessor = make_accessor(&map);

            index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
            index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);

            let results = index.search(&[1.0, 0.0, 0.0], 2, &accessor);
            assert_eq!(results.len(), 2, "Failed for metric {metric:?}");
            assert_eq!(
                results[0].0,
                NodeId::new(0),
                "Closest not correct for metric {metric:?}"
            );
        }
    }

    #[test]
    fn test_batch_search_consistency() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 0.0, 1.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor);

        let queries: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let batch_results = index.batch_search(&queries, 1, &accessor);
        assert_eq!(batch_results.len(), 3);

        // Each query should find its exact match
        for (i, results) in batch_results.iter().enumerate() {
            assert_eq!(results[0].0, NodeId::new(i as u64));
        }
    }

    #[test]
    fn test_with_capacity_constructor() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::with_capacity(config, 100);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_high_m_value() {
        // M larger than number of nodes
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_m(64);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);

        let results = index.search(&[1.0, 0.0, 0.0], 2, &accessor);
        assert_eq!(results.len(), 2);
    }

    // ── Filtered search tests ─────────────────────────────────────

    #[test]
    fn test_filtered_search_returns_only_allowlisted() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Allowlist: only even-numbered nodes
        let allowlist: HashSet<NodeId> = (1..=50).filter(|i| i % 2 == 0).map(NodeId::new).collect();

        let results = index.search_with_filter(&vectors[25], 5, &allowlist, &accessor);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Every result must be in the allowlist
        for (id, _) in &results {
            assert!(allowlist.contains(id), "Result {id:?} not in allowlist");
        }
    }

    #[test]
    fn test_filtered_search_empty_allowlist() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(20, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let allowlist: HashSet<NodeId> = HashSet::new();
        let results = index.search_with_filter(&vectors[5], 5, &allowlist, &accessor);
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_full_allowlist_matches_unfiltered() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Allowlist contains all nodes
        let allowlist: HashSet<NodeId> = (1..=50).map(NodeId::new).collect();
        let query = &vectors[25];

        let unfiltered = index.search_with_ef(query, 5, 200, &accessor);
        let filtered = index.search_with_ef_and_filter(query, 5, 200, &allowlist, &accessor);

        // With full allowlist, results should match unfiltered (same ef)
        assert_eq!(unfiltered.len(), filtered.len());
        for (u, f) in unfiltered.iter().zip(filtered.iter()) {
            assert_eq!(u.0, f.0);
        }
    }

    #[test]
    fn test_filtered_search_single_allowlisted_node() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Only one node allowed
        let allowlist: HashSet<NodeId> = [NodeId::new(30)].into_iter().collect();
        let results = index.search_with_filter(&vectors[25], 5, &allowlist, &accessor);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(30));
    }

    #[test]
    fn test_filtered_search_sorted_by_distance() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let allowlist: HashSet<NodeId> =
            (1..=100).filter(|i| i % 3 == 0).map(NodeId::new).collect();

        let results = index.search_with_filter(&[0.5, 0.5, 0.5, 0.5], 10, &allowlist, &accessor);
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_batch_filtered_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Allowlist: nodes 1..=50
        let allowlist: HashSet<NodeId> = (1..=50).map(NodeId::new).collect();
        let queries: Vec<Vec<f32>> = vec![vectors[10].clone(), vectors[70].clone()];

        let all_results = index.batch_search_with_filter(&queries, 5, &allowlist, &accessor);
        assert_eq!(all_results.len(), 2);

        for results in &all_results {
            for (id, _) in results {
                assert!(allowlist.contains(id));
            }
        }
    }

    #[test]
    // reason: test indices are small known values
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn test_filtered_search_ef_scaling() {
        // Verify that auto-scaling ef produces reasonable recall
        let n = 500;
        let dim = 8;
        let k = 10;
        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        let mut seed: u64 = 99999;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec, &accessor);
        }

        // 20% allowlist, moderate selectivity
        let allowlist: HashSet<NodeId> = (0..n)
            .filter(|i| i % 5 == 0)
            .map(|i| NodeId::new(i as u64))
            .collect();

        let query: Vec<f32> = (0..dim).map(|_| rand_f32()).collect();

        // Brute-force ground truth (only among allowlisted nodes)
        let mut gt: Vec<(u64, f32)> = allowlist
            .iter()
            .map(|id| {
                let dist = crate::index::vector::compute_distance(
                    &query,
                    &vectors[id.as_u64() as usize],
                    DistanceMetric::Euclidean,
                );
                (id.as_u64(), dist)
            })
            .collect();
        gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt_set: std::collections::HashSet<u64> = gt.iter().take(k).map(|(id, _)| *id).collect();

        let results = index.search_with_filter(&query, k, &allowlist, &accessor);
        let found: std::collections::HashSet<u64> =
            results.iter().map(|(id, _)| id.as_u64()).collect();

        let overlap = gt_set.intersection(&found).count();
        let recall = overlap as f64 / k as f64;
        assert!(
            recall >= 0.60,
            "Filtered recall {recall:.3} is below 0.60 threshold (20% selectivity)"
        );
    }

    #[test]
    fn test_filtered_search_cosine() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![1.0, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(3), vec![0.707, 0.707, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[1.0, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(3), &[0.707, 0.707, 0.0, 0.0], &accessor);

        let allowlist: HashSet<NodeId> = [NodeId::new(2), NodeId::new(3)].into_iter().collect();
        let results = index.search_with_filter(&[0.9, 0.1, 0.0, 0.0], 2, &allowlist, &accessor);

        // Node 1 is closest overall but not in allowlist
        assert!(!results.is_empty());
        for (id, _) in &results {
            assert!(allowlist.contains(id));
        }
        // Node 3 should be closest among allowed
        assert_eq!(results[0].0, NodeId::new(3));
    }

    // ── Phase 7c-2: HnswIndex with mmap-backed topology ─────────────

    use crate::index::vector::paged_topology::{MmapTopology, serialize_topology};
    use bytes::Bytes;

    /// Build a small index in heap mode, snapshot its topology, swap
    /// the backend to mmap mode, and verify search returns identical
    /// results.
    #[test]
    fn alix_mmap_backed_search_matches_heap_search() {
        let config = HnswConfig::new(8, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 42);

        let map: HashMap<NodeId, Arc<[f32]>> = (1..=20u64)
            .map(|i| {
                let v: Arc<[f32]> = (0..8u64)
                    .map(|j| (i.wrapping_mul(31).wrapping_add(j) % 17) as f32 / 17.0)
                    .collect::<Vec<_>>()
                    .into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        // Reference: search results from heap-backed index.
        let query: Vec<f32> = vec![0.1, 0.4, 0.6, 0.2, 0.8, 0.5, 0.3, 0.7];
        let heap_results = heap_index.search(&query, 5, &accessor);
        assert!(!heap_results.is_empty());

        // Snapshot + serialize + load back as mmap topology.
        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);
        assert!(mmap_index.is_mmap_backed());

        let mmap_results = mmap_index.search(&query, 5, &accessor);

        // Same NodeIds in the same order, same distances.
        assert_eq!(mmap_results.len(), heap_results.len());
        for ((id_h, d_h), (id_m, d_m)) in heap_results.iter().zip(mmap_results.iter()) {
            assert_eq!(id_h, id_m);
            assert!((d_h - d_m).abs() < 1e-6);
        }
    }

    /// Mutating an mmap-backed index must panic with a clear message,
    /// not silently no-op, corrupt state, or consume retry RNG state.
    #[test]
    fn gus_mmap_backed_insert_panics_without_advancing_rng() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let nodes = vec![(NodeId::new(1), vec![vec![]])];
        let bytes = serialize_topology(Some(NodeId::new(1)), 0, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let index = HnswIndex::new(config);
        index.adopt_mmap_topology(topo);

        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
        let mut expected_rng = *index.rng.read();
        let expected_next = expected_rng.next_f64();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            index.insert(NodeId::new(2), &[0.0, 0.0, 0.0, 0.0], &accessor);
        }));
        assert!(caught.is_err(), "mmap insertion must fail closed");
        let mut actual_rng = *index.rng.read();
        let actual_next = actual_rng.next_f64();
        assert_eq!(actual_next.to_bits(), expected_next.to_bits());
    }

    /// Soft-delete on an mmap-backed index must work without panic:
    /// the node is marked deleted in the in-memory `deleted` set and
    /// excluded from search results, but the topology is untouched.
    #[test]
    fn vincent_mmap_backed_remove_soft_deletes() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let nodes_data = vec![(NodeId::new(1), vec![vec![]])];
        let bytes = serialize_topology(Some(NodeId::new(1)), 0, &nodes_data);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let index = HnswIndex::new(config);
        index.adopt_mmap_topology(topo);

        // Soft-delete must succeed on mmap-backed index
        assert!(index.remove(NodeId::new(1)));
        // Node is still in topology but excluded from public API
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains_including_deleted(NodeId::new(1)));
        assert_eq!(index.len(), 0);
    }

    /// `restore_topology` after `adopt_mmap_topology` must put the
    /// index back in heap mode and accept mutations again.
    #[test]
    fn jules_restore_topology_returns_to_heap_mode() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let nodes = vec![(NodeId::new(1), vec![vec![]])];
        let bytes = serialize_topology(Some(NodeId::new(1)), 0, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let index = HnswIndex::new(config);
        index.adopt_mmap_topology(topo);
        assert!(index.is_mmap_backed());

        index.restore_topology(
            Some(NodeId::new(1)),
            0,
            vec![(NodeId::new(1), vec![vec![]])],
        );
        assert!(!index.is_mmap_backed());

        // Now insert should work without panicking.
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
        index.insert(NodeId::new(2), &[0.0, 0.0, 0.0, 0.0], &accessor);
        assert_eq!(index.len(), 2);
    }

    /// Heap memory savings: an mmap-backed index reports nearly zero
    /// heap usage (just the small struct), while the heap-backed
    /// equivalent reports significant overhead.
    #[test]
    fn mia_mmap_backed_heap_overhead_is_tiny() {
        let config = HnswConfig::new(8, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 42);

        let map: HashMap<NodeId, Arc<[f32]>> = (1..=50u64)
            .map(|i| {
                let v: Arc<[f32]> = vec![0.1; 8].into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }
        let heap_bytes = heap_index.heap_memory_bytes();
        assert!(
            heap_bytes > 1000,
            "heap-mode should report > 1KB heap usage"
        );

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);
        let mmap_bytes = mmap_index.heap_memory_bytes();
        assert!(
            mmap_bytes < 256,
            "mmap-mode heap overhead should be < 256 bytes, got {mmap_bytes}"
        );
        assert!(
            mmap_bytes < heap_bytes / 10,
            "mmap-mode {mmap_bytes} should be far smaller than heap-mode {heap_bytes}"
        );
    }

    // ── Phase 7d: recall regression + variant coverage ──────────────

    /// Builds a 200-vector HNSW deterministically and runs the four
    /// search variants in both heap and mmap modes. Each variant must
    /// return identical (id, distance) sequences across modes.
    #[test]
    fn shosanna_all_search_variants_match_across_modes() {
        let config = HnswConfig::new(8, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 7);

        // Deterministic pseudo-random vectors.
        let map: HashMap<NodeId, Arc<[f32]>> = (1..=200u64)
            .map(|i| {
                let v: Arc<[f32]> = (0..8u64)
                    .map(|j| {
                        let s = i.wrapping_mul(37).wrapping_add(j.wrapping_mul(101));
                        ((s % 1000) as f32) / 1000.0
                    })
                    .collect::<Vec<_>>()
                    .into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        let query: Vec<f32> = vec![0.31, 0.42, 0.55, 0.18, 0.77, 0.91, 0.05, 0.62];

        // search()
        let h = heap_index.search(&query, 10, &accessor);
        let m = mmap_index.search(&query, 10, &accessor);
        assert_eq!(h.len(), m.len());
        for ((hid, hd), (mid, md)) in h.iter().zip(m.iter()) {
            assert_eq!(hid, mid);
            assert!((hd - md).abs() < 1e-6);
        }

        // search_with_ef()
        let h = heap_index.search_with_ef(&query, 10, 50, &accessor);
        let m = mmap_index.search_with_ef(&query, 10, 50, &accessor);
        assert_eq!(h, m);

        // search_with_filter()
        let allowlist: HashSet<NodeId> = (1..=100u64).map(NodeId::new).collect();
        let h = heap_index.search_with_filter(&query, 5, &allowlist, &accessor);
        let m = mmap_index.search_with_filter(&query, 5, &allowlist, &accessor);
        assert_eq!(h, m);

        // search_with_ef_and_filter()
        let h = heap_index.search_with_ef_and_filter(&query, 5, 80, &allowlist, &accessor);
        let m = mmap_index.search_with_ef_and_filter(&query, 5, 80, &allowlist, &accessor);
        assert_eq!(h, m);

        // batch_search()
        let queries = vec![query.clone(), vec![0.5; 8], vec![0.0; 8]];
        let h = heap_index.batch_search(&queries, 5, &accessor);
        let m = mmap_index.batch_search(&queries, 5, &accessor);
        assert_eq!(h, m);
    }

    /// Recall@k: search results from the heap-backed and mmap-backed
    /// indexes must be identical, so recall is trivially 100%. The
    /// test exists to fail loudly if a future refactor introduces any
    /// divergence (e.g. iterator order shift).
    #[test]
    fn butch_mmap_recall_at_10_is_100_percent() {
        let config = HnswConfig::new(16, DistanceMetric::Cosine);
        let heap_index = HnswIndex::with_seed(config.clone(), 1234);

        let map: HashMap<NodeId, Arc<[f32]>> = (1..=300u64)
            .map(|i| {
                let v: Arc<[f32]> = (0..16u64)
                    .map(|j| {
                        let s = i.wrapping_mul(53).wrapping_add(j.wrapping_mul(149));
                        ((s % 997) as f32) / 997.0
                    })
                    .collect::<Vec<_>>()
                    .into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        // Run 20 different queries; each must match exactly.
        for q in 0..20u64 {
            let query: Vec<f32> = (0..16u64)
                .map(|j| {
                    let s = q.wrapping_mul(71).wrapping_add(j.wrapping_mul(211));
                    ((s % 991) as f32) / 991.0
                })
                .collect();

            let heap_results: HashSet<NodeId> = heap_index
                .search(&query, 10, &accessor)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let mmap_results: HashSet<NodeId> = mmap_index
                .search(&query, 10, &accessor)
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            // Recall@10 = |heap ∩ mmap| / |heap| = 1.0 because results
            // must be identical (deterministic byte-format read).
            let intersection = heap_results.intersection(&mmap_results).count();
            assert_eq!(
                intersection,
                heap_results.len(),
                "query {q}: recall@10 must be 100% (heap={heap_results:?}, mmap={mmap_results:?})"
            );
        }
    }

    // ── Soft-delete (MVCC) tests ──────────────────────────────────────

    /// Soft-delete: node is removed from search results but the topology
    /// is preserved so other live nodes remain reachable.
    #[test]
    fn soft_delete_retains_topology() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![0.1, 0.1, 0.1, 0.1].into());
        map.insert(NodeId::new(2), vec![0.5, 0.5, 0.5, 0.5].into());
        map.insert(NodeId::new(3), vec![0.9, 0.9, 0.9, 0.9].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.1, 0.1, 0.1], &accessor);
        index.insert(NodeId::new(2), &[0.5, 0.5, 0.5, 0.5], &accessor);
        index.insert(NodeId::new(3), &[0.9, 0.9, 0.9, 0.9], &accessor);
        assert_eq!(index.len(), 3);

        // Soft-delete node 2
        assert!(index.remove(NodeId::new(2)));

        // Search must not return node 2
        let results = index.search(&[0.5, 0.5, 0.5, 0.5], 3, &accessor);
        assert!(
            results.iter().all(|(id, _)| *id != NodeId::new(2)),
            "soft-deleted node 2 must not appear in results: {results:?}"
        );

        // Public API: deleted node is not "live"
        assert!(!index.contains(NodeId::new(2)));
        // Topology still holds it (routing hop)
        assert!(index.contains_including_deleted(NodeId::new(2)));
        // len counts live nodes only
        assert_eq!(index.len(), 2);

        // Re-insert (un-delete): node 2 is live again
        index.insert(NodeId::new(2), &[0.5, 0.5, 0.5, 0.5], &accessor);
        assert!(index.contains(NodeId::new(2)));
        assert_eq!(index.len(), 3);

        // Removing non-existent node returns false
        assert!(!index.remove(NodeId::new(99)));
    }

    #[test]
    fn soft_deleted_routing_hop_with_changed_dimensions_is_unavailable() {
        let index = HnswIndex::with_seed(HnswConfig::new(2, DistanceMetric::Euclidean), 42);
        let removed = NodeId::new(1);
        let live = NodeId::new(2);
        let mut vectors: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        vectors.insert(removed, Arc::from([1.0, 0.0]));
        vectors.insert(live, Arc::from([0.0, 1.0]));
        index.insert(removed, &[1.0, 0.0], &make_accessor(&vectors));
        index.insert(live, &[0.0, 1.0], &make_accessor(&vectors));
        assert!(index.remove(removed));
        vectors.insert(removed, Arc::from([1.0, 0.0, 0.0]));
        let accessor = make_accessor(&vectors);
        assert_eq!(index.search(&[0.0, 1.0], 10, &accessor), vec![(live, 0.0)]);
        // A caller's visibility predicate must not re-admit an incompatible
        // routing payload during the separate candidate re-score.
        assert_eq!(
            index.search_visible(&[0.0, 1.0], 10, 40, &|_| true, &accessor),
            vec![(live, 0.0)]
        );
        assert!(index.contains_including_deleted(removed));
    }

    // ── GC tests ──────────────────────────────────────────────────────

    fn gc_fixture() -> (HnswIndex, HashMap<NodeId, Arc<[f32]>>) {
        let index = HnswIndex::with_seed(HnswConfig::new(4, DistanceMetric::Euclidean), 42);
        let vectors: HashMap<_, Arc<[f32]>> = (1..=8)
            .map(|id| (NodeId::new(id), vec![id as f32; 4].into()))
            .collect();
        let accessor = make_accessor(&vectors);
        for id in 1..=8 {
            index.insert(NodeId::new(id), &vectors[&NodeId::new(id)], &accessor);
        }
        drop(accessor);
        (index, vectors)
    }

    fn assert_gc_state_same(left: &HnswExactState, right: &HnswExactState) {
        assert_eq!(left.entry_point, right.entry_point);
        assert_eq!(left.max_level, right.max_level);
        assert_eq!(left.nodes, right.nodes);
        assert_eq!(left.deleted, right.deleted);
        assert_eq!(left.rng_state, right.rng_state);
    }

    #[test]
    fn gc_no_removal_preserves_exact_state_without_reading_vectors() -> Result<(), String> {
        let (index, _) = gc_fixture();
        index.remove(NodeId::new(3));
        let before = index.snapshot_exact()?;
        let reads = std::sync::atomic::AtomicUsize::new(0);
        index.gc(&|_| true, &|_| {
            reads.fetch_add(1, Ordering::Relaxed);
            None
        })?;
        assert_gc_state_same(&index.snapshot_exact()?, &before);
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn gc_rebuild_preserves_above_horizon_tombstones() -> Result<(), String> {
        let (index, vectors) = gc_fixture();
        index.remove(NodeId::new(2));
        index.remove(NodeId::new(3));
        let accessor = make_accessor(&vectors);
        index.gc(&|id| id != NodeId::new(2), &accessor)?;
        assert!(!index.contains_including_deleted(NodeId::new(2)));
        assert!(index.contains_including_deleted(NodeId::new(3)));
        assert!(!index.contains(NodeId::new(3)));
        assert_eq!(index.snapshot_exact()?.deleted, vec![NodeId::new(3)]);
        assert!(
            index
                .search(&[3.0; 4], 8, &accessor)
                .iter()
                .all(|(id, _)| *id != NodeId::new(3))
        );
        Ok(())
    }

    #[test]
    fn gc_exact_restore_rebuild_and_next_insert_are_deterministic() -> Result<(), String> {
        let (index, mut vectors) = gc_fixture();
        index.remove(NodeId::new(2));
        let restored = HnswIndex::with_seed(index.config.clone(), 99);
        restored
            .apply_prepared_exact_restore(restored.prepare_exact_restore(index.snapshot_exact()?)?);
        let accessor = make_accessor(&vectors);
        index.gc(&|id| id != NodeId::new(2), &accessor)?;
        restored.gc(&|id| id != NodeId::new(2), &accessor)?;
        assert_gc_state_same(&index.snapshot_exact()?, &restored.snapshot_exact()?);
        drop(accessor);
        vectors.insert(NodeId::new(9), vec![9.0; 4].into());
        let accessor = make_accessor(&vectors);
        index.insert(NodeId::new(9), &[9.0; 4], &accessor);
        restored.insert(NodeId::new(9), &[9.0; 4], &accessor);
        assert_gc_state_same(&index.snapshot_exact()?, &restored.snapshot_exact()?);
        Ok(())
    }

    #[test]
    fn gc_unavailable_retained_vector_leaves_exact_state_unchanged() -> Result<(), String> {
        let (index, mut vectors) = gc_fixture();
        index.remove(NodeId::new(2));
        let before = index.snapshot_exact()?;
        vectors.remove(&NodeId::new(3));
        assert!(
            index
                .gc(&|id| id != NodeId::new(2), &make_accessor(&vectors))
                .is_err()
        );
        assert_gc_state_same(&index.snapshot_exact()?, &before);
        Ok(())
    }

    #[test]
    fn gc_invalid_retained_vector_and_denied_authority_preserve_state() -> Result<(), String> {
        let (index, mut vectors) = gc_fixture();
        index.remove(NodeId::new(2));
        let before = index.snapshot_exact()?;
        for invalid in [vec![1.0; 3], vec![f32::NAN; 4]] {
            vectors.insert(NodeId::new(3), invalid.into());
            assert!(
                index
                    .gc(&|id| id != NodeId::new(2), &make_accessor(&vectors))
                    .is_err()
            );
            assert_gc_state_same(&index.snapshot_exact()?, &before);
        }
        #[cfg(feature = "compact-store")]
        {
            index.freeze_exact_read_snapshot();
            assert!(index.gc(&|_| true, &make_accessor(&vectors)).is_err());
            assert_gc_state_same(&index.snapshot_exact()?, &before);
        }
        Ok(())
    }

    #[test]
    fn gc_callbacks_cannot_mutate_source_and_unwind_releases_admission() -> Result<(), String> {
        let (index, vectors) = gc_fixture();
        index.remove(NodeId::new(2));
        let accessor = make_accessor(&vectors);
        index.gc(
            &|id| {
                assert!(!index.remove(NodeId::new(4)));
                assert!(index.gc(&|_| true, &accessor).is_err());
                id != NodeId::new(2)
            },
            &|id| {
                assert!(!index.remove(NodeId::new(4)));
                index.insert(NodeId::new(90), &[90.0; 4], &accessor);
                vectors.get(&id).cloned()
            },
        )?;
        assert!(index.contains(NodeId::new(4)));
        assert!(!index.contains_including_deleted(NodeId::new(90)));
        let before = index.snapshot_exact()?;
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = index.gc(&|_| panic!("GC predicate unwind"), &accessor);
        }));
        assert!(unwind.is_err());
        assert_gc_state_same(&index.snapshot_exact()?, &before);
        assert!(index.remove(NodeId::new(4)));
        Ok(())
    }

    /// GC drops nodes below the horizon and keeps nodes above it.
    /// A live-but-soft-deleted node (is_live=true despite being in the
    /// deleted set) is retained by the GC rebuild.
    #[test]
    fn gc_drops_deleted_below_horizon() -> Result<(), String> {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for i in 1u64..=5 {
            let v: f32 = i as f32 / 6.0;
            map.insert(NodeId::new(i), vec![v, v, v, v].into());
        }
        let accessor = make_accessor(&map);

        for i in 1u64..=5 {
            let v = map[&NodeId::new(i)].clone();
            index.insert(NodeId::new(i), &v, &accessor);
        }
        assert_eq!(index.len(), 5);

        // Soft-delete node 3.
        assert!(index.remove(NodeId::new(3)));
        assert_eq!(index.len(), 4);
        // Still in topology as a routing hop.
        assert!(index.contains_including_deleted(NodeId::new(3)));

        // GC with is_live = "not node 3" (simulates delete committed below horizon).
        index.gc(&|id| id != NodeId::new(3), &accessor)?;

        // Node 3 must be gone from the topology entirely.
        assert!(
            !index.contains_including_deleted(NodeId::new(3)),
            "GC must remove node 3 from topology"
        );
        // Remaining live nodes are all present and searchable.
        for i in [1u64, 2, 4, 5] {
            assert!(
                index.contains(NodeId::new(i)),
                "live node {i} must still be present after GC"
            );
        }
        let results = index.search(&[0.5, 0.5, 0.5, 0.5], 4, &accessor);
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        assert!(
            !ids.contains(&3),
            "node 3 must not appear in search after GC: {ids:?}"
        );
        assert_eq!(
            index.len(),
            4,
            "index must report 4 live nodes after GC; got {}",
            index.len()
        );

        // GC with is_live = all nodes — every node in the topology is kept.
        // (Node 3 is already gone from a previous GC; the others are kept.)
        index.gc(&|_id| true, &accessor)?;
        assert_eq!(
            index.len(),
            4,
            "all-live GC must retain the 4 remaining nodes"
        );
        for i in [1u64, 2, 4, 5] {
            assert!(
                index.contains(NodeId::new(i)),
                "node {i} must be retained by all-live GC"
            );
        }
        Ok(())
    }

    /// Connectivity preservation: after soft-deleting a "bridge" node,
    /// remaining live nodes are still findable via search.
    #[test]
    fn soft_delete_routes_through_deleted() {
        // Build a 7-node index; after deleting several middle nodes,
        // the cluster at the far end must still be reachable.
        let config = HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        // Linear chain: 1 → 2 → 3 → 4 → 5 → 6 → 7
        for i in 1u64..=7 {
            let v = (i as f32) / 8.0;
            map.insert(NodeId::new(i), vec![v, v, v, v].into());
        }
        let accessor = make_accessor(&map);
        for i in 1u64..=7 {
            let v = (i as f32) / 8.0;
            index.insert(NodeId::new(i), &[v, v, v, v], &accessor);
        }
        assert_eq!(index.len(), 7);

        // Delete nodes 3, 4, 5 (middle of the chain)
        index.remove(NodeId::new(3));
        index.remove(NodeId::new(4));
        index.remove(NodeId::new(5));
        assert_eq!(index.len(), 4);

        // Node 6 and 7 must still be findable
        let results = index.search(&[0.85, 0.85, 0.85, 0.85], 4, &accessor);
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        assert!(
            ids.contains(&6) || ids.contains(&7),
            "live nodes 6 or 7 must be reachable after deleting middle nodes; got {ids:?}"
        );
        // No deleted nodes in results
        for (id, _) in &results {
            assert!(
                !matches!(id.as_u64(), 3..=5),
                "deleted node {id:?} must not appear in results"
            );
        }
    }

    /// Empty-allowlist filter must short-circuit cleanly in mmap mode.
    #[test]
    fn django_mmap_empty_allowlist_returns_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 99);
        let map: HashMap<NodeId, Arc<[f32]>> = (1..=10u64)
            .map(|i| (NodeId::new(i), vec![0.1; 4].into()))
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        let allowlist: HashSet<NodeId> = HashSet::new();
        let results = mmap_index.search_with_filter(&[0.1; 4], 5, &allowlist, &accessor);
        assert!(results.is_empty());
    }

    /// Mmap-backed search on an empty index must not panic.
    #[test]
    fn beatrix_mmap_empty_topology_search_returns_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let bytes = serialize_topology(None, 0, &[]);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        let results = mmap_index.search(&[0.1; 4], 5, &accessor);
        assert!(results.is_empty());
        assert_eq!(mmap_index.len(), 0);
        assert!(mmap_index.is_empty());
    }

    // ── search_visible (snapshot-aware predicate filter) tests ──────────────

    /// search_visible returns only IDs that pass is_visible; invisible IDs
    /// (5, 7) must be absent from results. With 10 nodes and 2 invisible,
    /// k≤8 should return min(k, 8) results.
    #[test]
    fn search_visible_excludes_invisible() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        // Insert ids 1..=10 with distinct vectors.
        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for i in 1u64..=10 {
            let v: Vec<f32> = (0..4).map(|j| ((i - 1) * 4 + j) as f32 / 40.0).collect();
            map.insert(NodeId::new(i), Arc::from(v.as_slice()));
        }
        let accessor = make_accessor(&map);

        for i in 1u64..=10 {
            let v = map[&NodeId::new(i)].clone();
            index.insert(NodeId::new(i), &v, &accessor);
        }
        assert_eq!(index.len(), 10);

        let is_visible = |id: NodeId| id != NodeId::new(5) && id != NodeId::new(7);

        // Query near the centre; ask for up to 8 results.
        let query = [0.25f32, 0.25, 0.25, 0.25];
        let results = index.search_visible(&query, 8, 20, &is_visible, &accessor);

        // Must not contain 5 or 7.
        for (id, _) in &results {
            assert!(
                *id != NodeId::new(5) && *id != NodeId::new(7),
                "invisible node {id:?} appeared in search_visible results"
            );
        }
        // 10 total − 2 invisible = 8 visible; asking for k=8 → must return 8.
        assert_eq!(
            results.len(),
            8,
            "expected 8 visible results, got {}: {results:?}",
            results.len()
        );
        // Results must be sorted by distance (ascending).
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "results not sorted at index {i}: {:?}",
                results
            );
        }
    }

    /// With half the nodes invisible, search_visible(k=3) must still return 3
    /// visible results (the widened ef compensates for filtering losses).
    #[test]
    fn search_visible_widens_for_recall() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for i in 1u64..=20 {
            let v: Vec<f32> = (0..4).map(|j| ((i - 1) * 4 + j) as f32 / 80.0).collect();
            map.insert(NodeId::new(i), Arc::from(v.as_slice()));
        }
        let accessor = make_accessor(&map);
        for i in 1u64..=20 {
            let v = map[&NodeId::new(i)].clone();
            index.insert(NodeId::new(i), &v, &accessor);
        }

        // Odd IDs are invisible (10 invisible, 10 visible).
        let is_visible = |id: NodeId| id.as_u64().is_multiple_of(2);

        let query = [0.25f32, 0.25, 0.25, 0.25];
        let results = index.search_visible(&query, 3, 10, &is_visible, &accessor);

        // Must return exactly 3 (10 visible nodes exist, k=3).
        assert_eq!(
            results.len(),
            3,
            "expected 3 visible results from widened ef; got {results:?}"
        );
        for (id, _) in &results {
            assert!(
                id.as_u64() % 2 == 0,
                "odd (invisible) node {id:?} appeared in results"
            );
        }
    }
}
