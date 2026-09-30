//! Chunked adjacency lists - the core data structure for graph traversal.
//!
//! Every graph database needs fast neighbor lookups. This implementation uses
//! chunked storage with delta buffers, giving you:
//!
//! - **O(1) amortized inserts** - new edges go into a delta buffer
//! - **Cache-friendly scans** - chunks are sized for L1/L2 cache
//! - **Soft deletes** - deletions don't require recompaction
//! - **Concurrent reads** - RwLock allows many simultaneous traversals
//! - **Compression** - cold chunks can be compressed using DeltaBitPacked

use crate::codec::{BitPackedInts, DeltaBitPacked};
#[cfg(all(feature = "lpg", feature = "compact-store"))]
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
#[cfg(feature = "lpg")]
use parking_lot::RwLockWriteGuard;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(all(test, feature = "lpg"))]
std::thread_local! {
    static LAST_EDGE_BUFFER: std::cell::Cell<Option<(usize, usize, usize)>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(all(test, feature = "lpg"))]
pub(crate) fn take_edge_buffer_capture() -> Option<(usize, usize, usize)> {
    LAST_EDGE_BUFFER.with(|capture| capture.replace(None))
}

#[cfg(feature = "lpg")]
mod commit;
#[cfg(feature = "lpg")]
pub(crate) use commit::{AdjacencyCommitWorkspace, AdjacencyDataGuards};

/// Default chunk capacity (number of edges per chunk).
const DEFAULT_CHUNK_CAPACITY: usize = 64;

/// Threshold for delta buffer compaction.
///
/// Lower values reduce memory overhead and iteration cost for delta buffers,
/// but increase compaction frequency. 64 provides a good balance for typical workloads.
const DELTA_COMPACTION_THRESHOLD: usize = 64;

/// Threshold for cold chunk compression.
///
/// When the number of hot chunks exceeds this threshold, the oldest hot chunks
/// are compressed and moved to cold storage. This balances memory usage with
/// the cost of compression/decompression.
const COLD_COMPRESSION_THRESHOLD: usize = 4;

/// A chunk of adjacency entries.
#[derive(Debug, Clone)]
struct AdjacencyChunk {
    /// Destination node IDs.
    destinations: Vec<NodeId>,
    /// Edge IDs (parallel to destinations).
    edge_ids: Vec<EdgeId>,
    /// Capacity of this chunk.
    capacity: usize,
}

impl AdjacencyChunk {
    fn new(capacity: usize) -> Self {
        Self {
            destinations: Vec::with_capacity(capacity),
            edge_ids: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.destinations.len()
    }

    fn is_full(&self) -> bool {
        self.destinations.len() >= self.capacity
    }

    fn push(&mut self, dst: NodeId, edge_id: EdgeId) -> bool {
        if self.is_full() {
            return false;
        }
        self.destinations.push(dst);
        self.edge_ids.push(edge_id);
        true
    }

    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        self.destinations
            .iter()
            .copied()
            .zip(self.edge_ids.iter().copied())
    }

    /// Compresses this chunk into a `CompressedAdjacencyChunk`.
    ///
    /// The entries are sorted by destination node ID for better delta compression.
    /// Use this for cold chunks that won't be modified.
    fn compress(&self) -> CompressedAdjacencyChunk {
        // Sort entries by destination for better delta compression
        let mut entries: Vec<_> = self
            .destinations
            .iter()
            .copied()
            .zip(self.edge_ids.iter().copied())
            .collect();
        entries.sort_by_key(|(dst, _)| dst.as_u64());

        // Extract sorted destinations and corresponding edge IDs
        let sorted_dsts: Vec<u64> = entries.iter().map(|(d, _)| d.as_u64()).collect();
        let sorted_edges: Vec<u64> = entries.iter().map(|(_, e)| e.as_u64()).collect();

        let max_destination = sorted_dsts.last().copied().unwrap_or(0);

        CompressedAdjacencyChunk {
            destinations: DeltaBitPacked::encode(&sorted_dsts),
            edge_ids: BitPackedInts::pack(&sorted_edges),
            count: entries.len(),
            max_destination,
        }
    }
}

/// A compressed chunk of adjacency entries.
///
/// Uses DeltaBitPacked for destination node IDs (sorted for good compression)
/// and BitPackedInts for edge IDs. Typical compression ratio is 5-10x for
/// dense adjacency lists.
#[derive(Debug, Clone)]
struct CompressedAdjacencyChunk {
    /// Delta + bit-packed destination node IDs (sorted).
    destinations: DeltaBitPacked,
    /// Bit-packed edge IDs.
    edge_ids: BitPackedInts,
    /// Number of entries.
    count: usize,
    /// Maximum destination node ID (last element in sorted order).
    max_destination: u64,
}

impl CompressedAdjacencyChunk {
    /// Returns the number of entries in this chunk.
    fn len(&self) -> usize {
        self.count
    }

    /// Returns true if this chunk is empty.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Decompresses and iterates over all entries.
    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let dsts = self.destinations.decode();
        let edges = self.edge_ids.unpack();

        dsts.into_iter()
            .zip(edges)
            .map(|(d, e)| (NodeId::new(d), EdgeId::new(e)))
    }

    /// Returns the approximate memory size in bytes.
    fn memory_size(&self) -> usize {
        // DeltaBitPacked: 8 bytes base + packed deltas
        // BitPackedInts: packed data
        let dest_size = 8 + self.destinations.to_bytes().map_or(0, |b| b.len());
        let edge_size = self.edge_ids.data_bytes().len();
        dest_size + edge_size
    }

    /// Returns the minimum destination ID in this chunk.
    #[must_use]
    fn min_destination(&self) -> u64 {
        self.destinations.base()
    }

    /// Returns `(destination, edge_id)` pairs where destination is in `[min, max]`.
    ///
    /// Uses zone-map pruning to skip chunks entirely when the range doesn't
    /// overlap, then `partition_point` for efficient sub-range extraction.
    fn destinations_in_range(&self, min: u64, max: u64) -> Vec<(NodeId, EdgeId)> {
        if min > self.max_destination || max < self.destinations.base() {
            return Vec::new();
        }
        let destinations = self.destinations.decode();
        let edges = self.edge_ids.unpack();
        let start = destinations.partition_point(|&d| d < min);
        let end = destinations.partition_point(|&d| d <= max);
        destinations[start..end]
            .iter()
            .zip(&edges[start..end])
            .map(|(&d, &e)| (NodeId::new(d), EdgeId::new(e)))
            .collect()
    }

    /// Returns the compression ratio compared to uncompressed storage.
    #[cfg(test)]
    fn compression_ratio(&self) -> f64 {
        if self.count == 0 {
            return 1.0;
        }
        let uncompressed = self.count * 16; // 8 bytes each for NodeId and EdgeId
        let compressed = self.memory_size();
        if compressed == 0 {
            return f64::INFINITY;
        }
        uncompressed as f64 / compressed as f64
    }
}

/// Zone map entry for a single compressed cold chunk.
///
/// The skip index stores one entry per cold chunk, sorted by `min_destination`.
/// Binary search on this index identifies which chunks might contain a target
/// destination without decompressing any data.
#[derive(Debug, Clone, Copy)]
struct SkipIndexEntry {
    /// Minimum destination ID in the chunk (from `DeltaBitPacked::base()`).
    min_destination: u64,
    /// Maximum destination ID in the chunk.
    max_destination: u64,
    /// Index into `AdjacencyList::cold_chunks`.
    chunk_index: usize,
}

/// Adjacency list for a single node.
///
/// Uses a tiered storage model:
/// - **Hot chunks**: Recent data, uncompressed for fast modification
/// - **Cold chunks**: Older data, compressed for memory efficiency
/// - **Delta buffer**: Very recent insertions, not yet compacted
/// - **Skip index**: Zone map over cold chunks for O(log n) point lookups
#[derive(Debug)]
struct AdjacencyList {
    /// Hot chunks (mutable, uncompressed) - for recent data.
    hot_chunks: Vec<AdjacencyChunk>,
    /// Cold chunks (immutable, compressed) - for older data.
    cold_chunks: Vec<CompressedAdjacencyChunk>,
    /// Delta buffer for recent insertions.
    /// Plain Vec keeps struct small (~144 bytes) for better DashMap/FxHashMap
    /// cache density. Auto-compaction drains this buffer at 64 entries, so it
    /// stays short-lived; the one-per-node heap allocation on first edge is
    /// negligible with jemalloc/mimalloc thread-local arenas.
    delta_inserts: Vec<(NodeId, EdgeId)>,
    /// Set of deleted edge IDs.
    deleted: FxHashSet<EdgeId>,
    /// Zone map skip index over cold chunks, sorted by `min_destination`.
    /// Enables O(log n) point lookups and range queries without decompressing
    /// all cold chunks.
    skip_index: Vec<SkipIndexEntry>,
}

impl AdjacencyList {
    fn new() -> Self {
        Self {
            hot_chunks: Vec::new(),
            cold_chunks: Vec::new(),
            delta_inserts: Vec::new(),
            deleted: FxHashSet::default(),
            skip_index: Vec::new(),
        }
    }

    fn add_edge(&mut self, dst: NodeId, edge_id: EdgeId, chunk_capacity: usize) {
        // Try to add to the last hot chunk
        if let Some(last) = self.hot_chunks.last_mut()
            && last.push(dst, edge_id)
        {
            return;
        }

        // Add to delta buffer
        self.delta_inserts.push((dst, edge_id));

        // Auto-compact when delta buffer exceeds threshold
        if self.delta_inserts.len() >= DELTA_COMPACTION_THRESHOLD {
            self.compact(chunk_capacity);
        }
    }

    fn mark_deleted(&mut self, edge_id: EdgeId) {
        self.deleted.insert(edge_id);
    }

    /// Builds the complete replacement for a qualified physical purge without
    /// changing this list. Every allocation occurs here, before publication.
    #[cfg(feature = "lpg")]
    fn prepare_purge_edges(
        &self,
        edge_ids: &FxHashSet<EdgeId>,
        chunk_capacity: usize,
    ) -> Option<Self> {
        let mut retained = Vec::with_capacity(self.physical_count());
        let mut counts: FxHashMap<EdgeId, usize> = FxHashMap::default();
        for entry @ (_, candidate) in self.iter_including_deleted() {
            if edge_ids.contains(&candidate) {
                *counts.entry(candidate).or_default() += 1;
            } else {
                retained.push(entry);
            }
        }
        if edge_ids
            .iter()
            .any(|edge_id| !self.deleted.contains(edge_id) || counts.get(edge_id) != Some(&1))
        {
            return None;
        }

        let mut replacement = Self::new();
        let mut deleted = self.deleted.clone();
        deleted.retain(|edge_id| !edge_ids.contains(edge_id));
        for (dst, retained_id) in retained {
            replacement.add_edge(dst, retained_id, chunk_capacity);
        }
        replacement.deleted = deleted;
        Some(replacement)
    }

    fn compact(&mut self, chunk_capacity: usize) {
        if self.delta_inserts.is_empty() {
            return;
        }

        // Create new chunks from delta buffer
        // Check if last hot chunk has room, and if so, pop it to continue filling
        let last_has_room = self.hot_chunks.last().is_some_and(|c| !c.is_full());
        let mut current_chunk = if last_has_room {
            // Invariant: is_some_and() returned true, so hot_chunks is non-empty
            self.hot_chunks
                .pop()
                .expect("hot_chunks is non-empty: is_some_and() succeeded on previous line")
        } else {
            AdjacencyChunk::new(chunk_capacity)
        };

        for (dst, edge_id) in self.delta_inserts.drain(..) {
            if !current_chunk.push(dst, edge_id) {
                self.hot_chunks.push(current_chunk);
                current_chunk = AdjacencyChunk::new(chunk_capacity);
                current_chunk.push(dst, edge_id);
            }
        }

        if current_chunk.len() > 0 {
            self.hot_chunks.push(current_chunk);
        }

        // Check if we should compress some hot chunks to cold
        self.maybe_compress_to_cold();
    }

    /// Compresses oldest hot chunks to cold storage if threshold exceeded.
    ///
    /// Builds skip index entries for each new cold chunk, enabling O(log n)
    /// point lookups via zone-map pruning.
    fn maybe_compress_to_cold(&mut self) {
        // Keep at least COLD_COMPRESSION_THRESHOLD hot chunks for write performance
        while self.hot_chunks.len() > COLD_COMPRESSION_THRESHOLD {
            // Remove the oldest (first) hot chunk
            let oldest = self.hot_chunks.remove(0);

            // Skip empty chunks
            if oldest.len() == 0 {
                continue;
            }

            // Compress and add to cold storage
            let compressed = oldest.compress();
            let chunk_index = self.cold_chunks.len();
            self.skip_index.push(SkipIndexEntry {
                min_destination: compressed.min_destination(),
                max_destination: compressed.max_destination,
                chunk_index,
            });
            self.cold_chunks.push(compressed);
        }
        // Maintain sort order for binary search
        self.skip_index.sort_unstable_by_key(|e| e.min_destination);
    }

    /// Forces all hot chunks to be compressed to cold storage.
    ///
    /// Useful when memory pressure is high or the node is rarely accessed.
    /// Rebuilds the skip index to include all newly compressed chunks.
    fn freeze_all(&mut self) {
        for chunk in self.hot_chunks.drain(..) {
            if chunk.len() > 0 {
                let compressed = chunk.compress();
                let chunk_index = self.cold_chunks.len();
                self.skip_index.push(SkipIndexEntry {
                    min_destination: compressed.min_destination(),
                    max_destination: compressed.max_destination,
                    chunk_index,
                });
                self.cold_chunks.push(compressed);
            }
        }
        self.skip_index.sort_unstable_by_key(|e| e.min_destination);
    }

    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let deleted = &self.deleted;

        // Iterate cold chunks first (oldest data)
        let cold_iter = self.cold_chunks.iter().flat_map(|c| c.iter());

        // Then hot chunks
        let hot_iter = self.hot_chunks.iter().flat_map(|c| c.iter());

        // Finally delta buffer (newest data)
        let delta_iter = self.delta_inserts.iter().copied();

        cold_iter
            .chain(hot_iter)
            .chain(delta_iter)
            .filter(move |(_, edge_id)| !deleted.contains(edge_id))
    }

    /// Like `iter()` but includes soft-deleted entries.
    ///
    /// Used by snapshot-aware traversal: an edge deleted *after* a transaction's
    /// snapshot start must still be visible to that snapshot.  The soft-delete
    /// tombstone in `deleted` is applied at commit time, so simply skipping the
    /// filter lets the MVCC version chain (consulted by `is_edge_visible_versioned`)
    /// make the correct per-snapshot decision.
    fn iter_including_deleted(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let cold_iter = self.cold_chunks.iter().flat_map(|c| c.iter());
        let hot_iter = self.hot_chunks.iter().flat_map(|c| c.iter());
        let delta_iter = self.delta_inserts.iter().copied();
        cold_iter.chain(hot_iter).chain(delta_iter)
    }

    /// Checks whether a specific destination node exists in this list.
    ///
    /// Uses the skip index for O(log n) lookup over cold chunks (only
    /// decompresses chunks whose zone maps overlap the target). Then scans
    /// hot chunks and the delta buffer linearly. Respects soft-deleted edges.
    fn contains(&self, destination: NodeId) -> bool {
        let dst_raw = destination.as_u64();
        let deleted = &self.deleted;

        // Cold chunks: use skip index to find candidate chunks
        for entry in &self.skip_index {
            if dst_raw < entry.min_destination || dst_raw > entry.max_destination {
                continue;
            }
            let chunk = &self.cold_chunks[entry.chunk_index];
            let decoded_dsts = chunk.destinations.decode();
            let decoded_edges = chunk.edge_ids.unpack();
            if let Ok(pos) = decoded_dsts.binary_search(&dst_raw) {
                // Check this position and adjacent duplicates
                let mut i = pos;
                while i > 0 && decoded_dsts[i - 1] == dst_raw {
                    i -= 1;
                }
                for j in i..decoded_dsts.len() {
                    if decoded_dsts[j] != dst_raw {
                        break;
                    }
                    if !deleted.contains(&EdgeId::new(decoded_edges[j])) {
                        return true;
                    }
                }
            }
        }

        // Hot chunks: linear scan (small, unsorted)
        for chunk in &self.hot_chunks {
            for (dst, edge_id) in chunk.iter() {
                if dst == destination && !deleted.contains(&edge_id) {
                    return true;
                }
            }
        }

        // Delta buffer: linear scan
        for &(dst, edge_id) in &self.delta_inserts {
            if dst == destination && !deleted.contains(&edge_id) {
                return true;
            }
        }

        false
    }

    /// Returns edges whose destination falls in `[min, max]` (inclusive).
    ///
    /// Uses skip index zone-map pruning over cold chunks, then linear scan
    /// of hot chunks and delta buffer. Respects soft-deleted edges.
    fn destinations_in_range(&self, min: NodeId, max: NodeId) -> Vec<(NodeId, EdgeId)> {
        let min_raw = min.as_u64();
        let max_raw = max.as_u64();
        let deleted = &self.deleted;
        let mut results = Vec::new();

        // Cold chunks: skip index prunes non-overlapping chunks
        for entry in &self.skip_index {
            if entry.max_destination < min_raw || entry.min_destination > max_raw {
                continue;
            }
            let chunk = &self.cold_chunks[entry.chunk_index];
            results.extend(
                chunk
                    .destinations_in_range(min_raw, max_raw)
                    .into_iter()
                    .filter(|(_, eid)| !deleted.contains(eid)),
            );
        }

        // Hot chunks: linear scan
        for chunk in &self.hot_chunks {
            for (dst, edge_id) in chunk.iter() {
                if dst.as_u64() >= min_raw && dst.as_u64() <= max_raw && !deleted.contains(&edge_id)
                {
                    results.push((dst, edge_id));
                }
            }
        }

        // Delta buffer: linear scan
        for &(dst, edge_id) in &self.delta_inserts {
            if dst.as_u64() >= min_raw && dst.as_u64() <= max_raw && !deleted.contains(&edge_id) {
                results.push((dst, edge_id));
            }
        }

        results
    }

    fn neighbors(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.iter().map(|(dst, _)| dst)
    }

    fn degree(&self) -> usize {
        self.iter().count()
    }

    /// Returns the number of entries in hot storage.
    fn hot_count(&self) -> usize {
        self.hot_chunks.iter().map(|c| c.len()).sum::<usize>() + self.delta_inserts.len()
    }

    /// Returns the number of entries in cold storage.
    fn cold_count(&self) -> usize {
        self.cold_chunks.iter().map(|c| c.len()).sum()
    }

    /// Returns the number of physical entries across hot, delta, and cold
    /// storage. In particular, this must not treat a rebuilt low-degree list
    /// whose retained entries are still in the delta buffer as empty.
    #[cfg(feature = "lpg")]
    fn physical_count(&self) -> usize {
        self.hot_count() + self.cold_count()
    }

    /// Returns the approximate memory size in bytes.
    #[cfg(test)]
    fn memory_size(&self) -> usize {
        // Hot chunks: full uncompressed size
        let hot_size = self.hot_chunks.iter().map(|c| c.len() * 16).sum::<usize>();

        // Cold chunks: compressed size
        let cold_size = self
            .cold_chunks
            .iter()
            .map(|c| c.memory_size())
            .sum::<usize>();

        // Delta buffer
        let delta_size = self.delta_inserts.len() * 16;

        // Deleted set (rough estimate)
        let deleted_size = self.deleted.len() * 16;

        hot_size + cold_size + delta_size + deleted_size
    }
}

/// The main structure for traversing graph edges.
///
/// Given a node, this tells you all its neighbors and the edges connecting them.
/// Internally uses chunked storage (64 edges per chunk) with a delta buffer for
/// recent inserts. Deletions are soft (tombstones) until compaction.
///
/// # Example
///
/// ```
/// use grafeo_core::index::ChunkedAdjacency;
/// use grafeo_common::types::{NodeId, EdgeId};
///
/// let adj = ChunkedAdjacency::new();
///
/// // Build a star graph: node 0 connects to nodes 1, 2, 3
/// adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(100));
/// adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(101));
/// adj.add_edge(NodeId::new(0), NodeId::new(3), EdgeId::new(102));
///
/// // Fast neighbor lookup
/// let neighbors = adj.neighbors(NodeId::new(0));
/// assert_eq!(neighbors.len(), 3);
/// ```
pub struct ChunkedAdjacency {
    /// Adjacency lists indexed by source node.
    /// Lock order: 10 (nested, acquired via LpgStore::forward_adj/backward_adj)
    lists: RwLock<FxHashMap<NodeId, AdjacencyList>>,
    /// Chunk capacity for new chunks.
    chunk_capacity: usize,
    /// Total number of edges (including deleted).
    edge_count: AtomicUsize,
    /// Number of deleted edges.
    deleted_count: AtomicUsize,
    /// Test-only proof that transport batch qualification visits each affected
    /// adjacency list a constant number of times rather than once per edge.
    #[cfg(all(test, feature = "lpg"))]
    transport_list_scans: AtomicUsize,
    /// Test-only count of physical list rebuilds during transport purge.
    #[cfg(all(test, feature = "lpg"))]
    transport_list_rebuilds: AtomicUsize,
}

/// Paired complete adjacency image with retirement owned by the source.
#[cfg(feature = "lpg")]
pub(crate) struct PreparedAdjacencyRestore<'target, 'source> {
    target: &'target ChunkedAdjacency,
    lists: RwLockWriteGuard<'target, FxHashMap<NodeId, AdjacencyList>>,
    source: &'source mut ChunkedAdjacency,
}

#[cfg(feature = "lpg")]
impl PreparedAdjacencyRestore<'_, '_> {
    pub(crate) fn install(&mut self) {
        std::mem::swap(&mut *self.lists, self.source.lists.get_mut());
        let edges = self.source.edge_count.load(Ordering::Relaxed);
        let deleted = self.source.deleted_count.load(Ordering::Relaxed);
        let old_edges = self.target.edge_count.swap(edges, Ordering::AcqRel);
        let old_deleted = self.target.deleted_count.swap(deleted, Ordering::AcqRel);
        self.source.edge_count.store(old_edges, Ordering::Relaxed);
        self.source
            .deleted_count
            .store(old_deleted, Ordering::Relaxed);
    }
}

/// Unwind-atomic publication of one private compact-promotion edge.
///
/// The write guard prevents compaction or retained aliases from moving the
/// new entry while graph property replay remains private. All capacity is
/// reserved before publication. Drop removes the exact delta entry and
/// reverses its counter only if that counter step completed.
#[cfg(all(feature = "lpg", feature = "compact-store"))]
#[must_use = "private adjacency publication must be committed after graph routing"]
pub(crate) struct AdjacencyUnpublishedMutation<'a> {
    owner: &'a ChunkedAdjacency,
    lists: RwLockWriteGuard<'a, FxHashMap<NodeId, AdjacencyList>>,
    src: NodeId,
    edge_id: EdgeId,
    prepared_list: Option<AdjacencyList>,
    state: AdjacencyPublicationState,
}

#[cfg(all(feature = "lpg", feature = "compact-store"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum AdjacencyPublicationState {
    Prepared,
    PublishingExisting,
    PublishingNew,
    CountedExisting,
    CountedNew,
    Committed,
}

#[cfg(all(feature = "lpg", feature = "compact-store"))]
impl AdjacencyUnpublishedMutation<'_> {
    fn publish_inner(&mut self, dst: NodeId, after_physical: impl FnOnce()) {
        if self.state != AdjacencyPublicationState::Prepared {
            return;
        }
        // Arm physical rollback before the first infallible write so even an
        // injected post-write panic is covered.
        let publishes_new_list = self.prepared_list.is_some();
        self.state = if publishes_new_list {
            AdjacencyPublicationState::PublishingNew
        } else {
            AdjacencyPublicationState::PublishingExisting
        };
        if let Some(mut list) = self.prepared_list.take() {
            list.delta_inserts.push((dst, self.edge_id));
            self.lists.insert(self.src, list);
        } else if let Some(list) = self.lists.get_mut(&self.src) {
            list.delta_inserts.push((dst, self.edge_id));
        }
        after_physical();
        self.owner.edge_count.fetch_add(1, Ordering::Relaxed);
        self.state = if publishes_new_list {
            AdjacencyPublicationState::CountedNew
        } else {
            AdjacencyPublicationState::CountedExisting
        };
    }

    pub(crate) fn publish(&mut self, dst: NodeId) {
        self.publish_inner(dst, || {});
    }

    #[cfg(test)]
    fn publish_with_hook(&mut self, dst: NodeId, after_physical: impl FnOnce()) {
        self.publish_inner(dst, after_physical);
    }

    pub(crate) fn commit(mut self) {
        self.state = AdjacencyPublicationState::Committed;
    }
}

#[cfg(all(feature = "lpg", feature = "compact-store"))]
impl Drop for AdjacencyUnpublishedMutation<'_> {
    fn drop(&mut self) {
        let (inserted_new_list, counted) = match self.state {
            AdjacencyPublicationState::Prepared | AdjacencyPublicationState::Committed => return,
            AdjacencyPublicationState::PublishingExisting => (false, false),
            AdjacencyPublicationState::PublishingNew => (true, false),
            AdjacencyPublicationState::CountedExisting => (false, true),
            AdjacencyPublicationState::CountedNew => (true, true),
        };
        if let Some(list) = self.lists.get_mut(&self.src) {
            list.delta_inserts
                .retain(|(_, candidate)| *candidate != self.edge_id);
        }
        if inserted_new_list
            && self
                .lists
                .get(&self.src)
                .is_some_and(|list| list.physical_count() == 0)
        {
            self.lists.remove(&self.src);
        }
        if counted {
            self.owner.edge_count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Fully allocated, write-locked adjacency replacement for one purge batch.
/// Dropping it aborts without mutation; `commit` only swaps/removes entries and
/// updates counters, with no allocation or user-controlled code.
#[cfg(feature = "lpg")]
pub(crate) struct PreparedAdjacencyPurge<'a> {
    owner: &'a ChunkedAdjacency,
    lists: RwLockWriteGuard<'a, FxHashMap<NodeId, AdjacencyList>>,
    replacements: Vec<(NodeId, AdjacencyList)>,
    removed: usize,
}

#[cfg(feature = "lpg")]
impl PreparedAdjacencyPurge<'_> {
    pub(crate) fn commit(mut self) {
        for (src, replacement) in self.replacements.drain(..) {
            if replacement.physical_count() == 0 {
                self.lists.remove(&src);
            } else {
                // The key already exists under this retained write guard, so
                // replacement cannot grow/reallocate the hash table.
                drop(self.lists.insert(src, replacement));
            }
        }
        self.owner
            .edge_count
            .fetch_sub(self.removed, Ordering::Relaxed);
        self.owner
            .deleted_count
            .fetch_sub(self.removed, Ordering::Relaxed);
    }
}

impl ChunkedAdjacency {
    /// Acquires an empty target without waiting beneath another final writer.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_pristine_restore<'target, 'source>(
        &'target self,
        source: &'source mut Self,
    ) -> Option<PreparedAdjacencyRestore<'target, 'source>> {
        let prepared = self.prepare_replacement(source)?;
        if !prepared.lists.is_empty()
            || self.edge_count.load(Ordering::Acquire) != 0
            || self.deleted_count.load(Ordering::Acquire) != 0
        {
            return None;
        }
        Some(prepared)
    }

    /// Retains the populated directory while its complete backing is replaced.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_replacement<'target, 'source>(
        &'target self,
        source: &'source mut Self,
    ) -> Option<PreparedAdjacencyRestore<'target, 'source>> {
        let lists = self.lists.try_write()?;
        if self.chunk_capacity != source.chunk_capacity {
            return None;
        }
        Some(PreparedAdjacencyRestore {
            target: self,
            lists,
            source,
        })
    }

    /// Creates a new chunked adjacency structure.
    #[must_use]
    pub fn new() -> Self {
        Self::with_chunk_capacity(DEFAULT_CHUNK_CAPACITY)
    }

    /// Creates a new chunked adjacency with custom chunk capacity.
    #[must_use]
    pub fn with_chunk_capacity(capacity: usize) -> Self {
        Self {
            lists: RwLock::new(FxHashMap::default()),
            chunk_capacity: capacity,
            edge_count: AtomicUsize::new(0),
            deleted_count: AtomicUsize::new(0),
            #[cfg(all(test, feature = "lpg"))]
            transport_list_scans: AtomicUsize::new(0),
            #[cfg(all(test, feature = "lpg"))]
            transport_list_rebuilds: AtomicUsize::new(0),
        }
    }

    /// Prepares an allocation-complete, exact private edge publication.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    pub(crate) fn begin_unpublished_edge(
        &self,
        src: NodeId,
        edge_id: EdgeId,
    ) -> Result<Option<AdjacencyUnpublishedMutation<'_>>, AllocError> {
        let mut lists = self.lists.write();
        if let Some(list) = lists.get(&src)
            && list
                .iter_including_deleted()
                .any(|(_, candidate)| candidate == edge_id)
        {
            return Ok(None);
        }

        let prepared_list = if let Some(list) = lists.get_mut(&src) {
            list.delta_inserts
                .try_reserve(1)
                .map_err(|_| AllocError::InsufficientSpace)?;
            None
        } else {
            lists
                .try_reserve(1)
                .map_err(|_| AllocError::InsufficientSpace)?;
            let mut list = AdjacencyList::new();
            list.delta_inserts
                .try_reserve(1)
                .map_err(|_| AllocError::InsufficientSpace)?;
            Some(list)
        };
        Ok(Some(AdjacencyUnpublishedMutation {
            owner: self,
            lists,
            src,
            edge_id,
            prepared_list,
            state: AdjacencyPublicationState::Prepared,
        }))
    }

    /// Adds an edge from src to dst.
    pub fn add_edge(&self, src: NodeId, dst: NodeId, edge_id: EdgeId) {
        let mut lists = self.lists.write();
        lists
            .entry(src)
            .or_insert_with(AdjacencyList::new)
            .add_edge(dst, edge_id, self.chunk_capacity);
        self.edge_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds multiple edges in a single lock acquisition.
    ///
    /// Each tuple is `(src, dst, edge_id)`. Takes the write lock once and
    /// inserts all edges, then compacts any lists that exceed the delta
    /// threshold. Significantly faster than calling `add_edge()` in a loop
    /// for bulk imports.
    pub fn batch_add_edges(&self, edges: &[(NodeId, NodeId, EdgeId)]) {
        if edges.is_empty() {
            return;
        }
        let mut lists = self.lists.write();
        for &(src, dst, edge_id) in edges {
            lists
                .entry(src)
                .or_insert_with(AdjacencyList::new)
                .add_edge(dst, edge_id, self.chunk_capacity);
        }
        self.edge_count.fetch_add(edges.len(), Ordering::Relaxed);
    }

    /// Marks an edge as deleted.
    pub fn mark_deleted(&self, src: NodeId, edge_id: EdgeId) {
        let mut lists = self.lists.write();
        if let Some(list) = lists.get_mut(&src) {
            list.mark_deleted(edge_id);
            self.deleted_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Marks multiple edges as deleted under a single write lock.
    ///
    /// This is used by `delete_node_edges` to ensure concurrent readers of
    /// adjacency never observe a partially detached node.
    pub fn batch_mark_deleted(&self, edges: &[(NodeId, EdgeId)]) {
        let mut lists = self.lists.write();
        let mut count = 0usize;
        for &(src, edge_id) in edges {
            if let Some(list) = lists.get_mut(&src) {
                list.mark_deleted(edge_id);
                count += 1;
            }
        }
        if count > 0 {
            self.deleted_count.fetch_add(count, Ordering::Relaxed);
        }
    }

    /// Restores a previously deleted edge (removes it from the deleted set).
    ///
    /// Used during transaction rollback to undo a soft-delete.
    pub fn unmark_deleted(&self, src: NodeId, edge_id: EdgeId) {
        let mut lists = self.lists.write();
        if let Some(list) = lists.get_mut(&src)
            && list.deleted.remove(&edge_id)
        {
            self.deleted_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Allocates and validates a physical purge while retaining the adjacency
    /// write guard. No list or counter changes until [`PreparedAdjacencyPurge::commit`].
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_purge_edges(
        &self,
        edges: &[(NodeId, EdgeId)],
    ) -> Option<PreparedAdjacencyPurge<'_>> {
        if edges.is_empty() {
            return Some(PreparedAdjacencyPurge {
                owner: self,
                lists: self.lists.write(),
                replacements: Vec::new(),
                removed: 0,
            });
        }
        let mut grouped: FxHashMap<NodeId, FxHashSet<EdgeId>> = FxHashMap::default();
        for &(src, edge_id) in edges {
            if edge_id == EdgeId::INVALID || !grouped.entry(src).or_default().insert(edge_id) {
                return None;
            }
        }

        let lists = self.lists.write();
        let mut replacements = Vec::with_capacity(grouped.len());
        for (src, targets) in &grouped {
            let list = lists.get(src)?;
            #[cfg(all(test, feature = "lpg"))]
            self.transport_list_scans.fetch_add(1, Ordering::Relaxed);
            let replacement = list.prepare_purge_edges(targets, self.chunk_capacity)?;
            replacements.push((*src, replacement));
        }
        #[cfg(all(test, feature = "lpg"))]
        self.transport_list_rebuilds
            .fetch_add(grouped.len(), Ordering::Relaxed);
        Some(PreparedAdjacencyPurge {
            owner: self,
            lists,
            replacements,
            removed: edges.len(),
        })
    }

    /// Returns whether exactly one physical entry and its soft-delete tombstone
    /// exist for this identity.
    #[must_use]
    #[cfg(feature = "lpg")]
    pub(crate) fn is_exact_deleted_edge(&self, src: NodeId, edge_id: EdgeId) -> bool {
        self.lists.read().get(&src).is_some_and(|list| {
            list.deleted.contains(&edge_id)
                && list
                    .iter_including_deleted()
                    .filter(|(_, candidate)| *candidate == edge_id)
                    .count()
                    == 1
        })
    }

    /// Returns the subset of requested identities that have exactly one
    /// physical entry and one soft-delete tombstone. Each affected adjacency
    /// list is scanned once, so a star-shaped batch is linear in list size.
    #[must_use]
    #[cfg(feature = "lpg")]
    pub(crate) fn exact_deleted_edges(&self, edges: &[(NodeId, EdgeId)]) -> FxHashSet<EdgeId> {
        let mut grouped: FxHashMap<NodeId, FxHashSet<EdgeId>> = FxHashMap::default();
        for &(src, edge_id) in edges {
            if edge_id != EdgeId::INVALID {
                grouped.entry(src).or_default().insert(edge_id);
            }
        }
        let lists = self.lists.read();
        let mut qualified = FxHashSet::default();
        for (src, targets) in grouped {
            let Some(list) = lists.get(&src) else {
                continue;
            };
            #[cfg(all(test, feature = "lpg"))]
            self.transport_list_scans.fetch_add(1, Ordering::Relaxed);
            let mut counts: FxHashMap<EdgeId, usize> = FxHashMap::default();
            for (_, edge_id) in list.iter_including_deleted() {
                if targets.contains(&edge_id) {
                    *counts.entry(edge_id).or_default() += 1;
                }
            }
            qualified.extend(targets.into_iter().filter(|edge_id| {
                list.deleted.contains(edge_id) && counts.get(edge_id) == Some(&1)
            }));
        }
        qualified
    }

    /// Resets and snapshots the exact-list work counters used by transport
    /// scale regressions.
    #[cfg(all(test, feature = "lpg"))]
    pub(crate) fn reset_transport_work_for_test(&self) {
        self.transport_list_scans.store(0, Ordering::Relaxed);
        self.transport_list_rebuilds.store(0, Ordering::Relaxed);
    }

    #[cfg(all(test, feature = "lpg"))]
    pub(crate) fn transport_work_for_test(&self) -> (usize, usize) {
        (
            self.transport_list_scans.load(Ordering::Relaxed),
            self.transport_list_rebuilds.load(Ordering::Relaxed),
        )
    }

    /// Returns all neighbors of a node.
    ///
    /// Note: This allocates a Vec to collect neighbors while the internal lock
    /// is held, then returns the Vec. For traversal performance, consider using
    /// `edges_from` if you also need edge IDs, to avoid multiple lookups.
    #[must_use]
    pub fn neighbors(&self, src: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.fill_neighbors(src, &mut out);
        out
    }

    /// Append dests of `src` to `out` under the adjacency lock. No extra Vec.
    pub fn fill_neighbors(&self, src: NodeId, out: &mut Vec<NodeId>) {
        let lists = self.lists.read();
        if let Some(list) = lists.get(&src) {
            out.extend(list.neighbors());
        }
    }

    /// Dest lists for every source, one lock.
    #[must_use]
    pub fn snapshot_neighbors(&self) -> Vec<(NodeId, Vec<NodeId>)> {
        let lists = self.lists.read();
        let mut out = Vec::with_capacity(lists.len());
        for (src, list) in lists.iter() {
            let dests: Vec<NodeId> = list.neighbors().collect();
            if !dests.is_empty() {
                out.push((*src, dests));
            }
        }
        out
    }

    /// Returns all (neighbor, edge_id) pairs for outgoing edges from a node.
    ///
    /// Note: This allocates a Vec to collect edges while the internal lock
    /// is held, then returns the Vec. This is intentional to avoid holding
    /// the lock across iteration.
    #[must_use]
    pub fn edges_from(&self, src: NodeId) -> Vec<(NodeId, EdgeId)> {
        let lists = self.lists.read();
        let edges: Vec<(NodeId, EdgeId)> = lists
            .get(&src)
            .map(|list| list.iter().collect())
            .unwrap_or_default();
        #[cfg(all(test, feature = "lpg"))]
        LAST_EDGE_BUFFER.with(|capture| {
            capture.set(Some((
                edges.as_ptr() as usize,
                edges.len(),
                edges.capacity(),
            )));
        });
        edges
    }

    /// Returns all `(neighbor, edge_id)` pairs including soft-deleted entries.
    ///
    /// Unlike `edges_from`, this does **not** filter out edges in the `deleted`
    /// set.  Use this as the raw adjacency source for snapshot-aware traversal:
    /// an edge soft-deleted after a transaction's snapshot start is still present
    /// in the chunks and must be handed to the MVCC version-chain check
    /// (`is_edge_visible_versioned`) rather than being silently dropped here.
    #[must_use]
    pub fn edges_from_including_deleted(&self, src: NodeId) -> Vec<(NodeId, EdgeId)> {
        let lists = self.lists.read();
        lists
            .get(&src)
            .map(|list| list.iter_including_deleted().collect())
            .unwrap_or_default()
    }

    /// Returns the out-degree of a node (number of outgoing edges).
    ///
    /// For forward adjacency, this counts edges where `src` is the source.
    pub fn out_degree(&self, src: NodeId) -> usize {
        let lists = self.lists.read();
        lists.get(&src).map_or(0, |list| list.degree())
    }

    /// Returns the in-degree of a node (number of incoming edges).
    ///
    /// This is semantically equivalent to `out_degree` but named differently
    /// for use with backward adjacency where edges are stored in reverse.
    /// When called on `backward_adj`, this returns the count of edges
    /// where `node` is the destination.
    pub fn in_degree(&self, node: NodeId) -> usize {
        let lists = self.lists.read();
        lists.get(&node).map_or(0, |list| list.degree())
    }

    /// Compacts all adjacency lists.
    pub fn compact(&self) {
        let mut lists = self.lists.write();
        for list in lists.values_mut() {
            list.compact(self.chunk_capacity);
        }
    }

    /// Compacts delta buffers that exceed the threshold.
    pub fn compact_if_needed(&self) {
        let mut lists = self.lists.write();
        for list in lists.values_mut() {
            if list.delta_inserts.len() >= DELTA_COMPACTION_THRESHOLD {
                list.compact(self.chunk_capacity);
            }
        }
    }

    /// Returns the total number of edges (including deleted).
    pub fn total_edge_count(&self) -> usize {
        self.edge_count.load(Ordering::Relaxed)
    }

    /// Returns the number of active (non-deleted) edges.
    pub fn active_edge_count(&self) -> usize {
        self.edge_count.load(Ordering::Relaxed) - self.deleted_count.load(Ordering::Relaxed)
    }

    /// Returns the number of nodes with adjacency lists.
    pub fn node_count(&self) -> usize {
        self.lists.read().len()
    }

    /// Checks if an edge from `src` to `dst` exists (not deleted).
    ///
    /// Uses zone-map skip index over cold chunks for O(log n) lookup
    /// when most data is in cold storage. Hot chunks and the delta buffer
    /// are scanned linearly.
    #[must_use]
    pub fn contains_edge(&self, src: NodeId, dst: NodeId) -> bool {
        let lists = self.lists.read();
        lists.get(&src).is_some_and(|list| list.contains(dst))
    }

    /// Returns edges from `src` whose destination is in `[min_dst, max_dst]`.
    ///
    /// Only decompresses cold chunks whose zone maps overlap the requested range.
    #[must_use]
    pub fn edges_in_range(
        &self,
        src: NodeId,
        min_dst: NodeId,
        max_dst: NodeId,
    ) -> Vec<(NodeId, EdgeId)> {
        let lists = self.lists.read();
        lists
            .get(&src)
            .map(|list| list.destinations_in_range(min_dst, max_dst))
            .unwrap_or_default()
    }

    /// Clears all adjacency lists.
    pub fn clear(&self) {
        let mut lists = self.lists.write();
        lists.clear();
        self.edge_count.store(0, Ordering::Relaxed);
        self.deleted_count.store(0, Ordering::Relaxed);
    }

    /// Returns memory statistics for this adjacency structure.
    #[must_use]
    pub fn memory_stats(&self) -> AdjacencyMemoryStats {
        let lists = self.lists.read();

        let mut hot_entries = 0usize;
        let mut cold_entries = 0usize;
        let mut hot_bytes = 0usize;
        let mut cold_bytes = 0usize;

        for list in lists.values() {
            hot_entries += list.hot_count();
            cold_entries += list.cold_count();

            // Hot: uncompressed (16 bytes per entry)
            hot_bytes += list.hot_count() * 16;

            // Cold: compressed size from memory_size()
            for cold_chunk in &list.cold_chunks {
                cold_bytes += cold_chunk.memory_size();
            }
        }

        AdjacencyMemoryStats {
            hot_entries,
            cold_entries,
            hot_bytes,
            cold_bytes,
            node_count: lists.len(),
        }
    }

    /// Returns estimated heap memory in bytes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        let lists = self.lists.read();
        // Outer hash map overhead
        let map_overhead = lists.capacity()
            * (std::mem::size_of::<NodeId>() + std::mem::size_of::<AdjacencyList>() + 1);
        // Per-list memory: hot chunks + cold chunks + deltas + deleted set
        let mut list_bytes = 0usize;
        for list in lists.values() {
            // Hot chunks: Vec<AdjacencyChunk> capacity + each chunk's Vec capacity
            list_bytes += list.hot_chunks.capacity() * std::mem::size_of::<AdjacencyChunk>();
            for chunk in &list.hot_chunks {
                list_bytes += chunk.destinations.capacity() * std::mem::size_of::<NodeId>();
                list_bytes += chunk.edge_ids.capacity() * std::mem::size_of::<EdgeId>();
            }
            // Cold chunks: compressed data
            list_bytes +=
                list.cold_chunks.capacity() * std::mem::size_of::<CompressedAdjacencyChunk>();
            for cold in &list.cold_chunks {
                list_bytes += cold.memory_size();
            }
            // Delta buffer
            list_bytes += list.delta_inserts.capacity() * 16;
            // Deleted set
            list_bytes += list.deleted.capacity() * (std::mem::size_of::<EdgeId>() + 1);
            // Skip index
            list_bytes += list.skip_index.capacity() * std::mem::size_of::<SkipIndexEntry>();
        }
        map_overhead + list_bytes
    }

    /// Forces all hot chunks to be compressed for all adjacency lists.
    ///
    /// This is useful when memory pressure is high or during shutdown.
    pub fn freeze_all(&self) {
        let mut lists = self.lists.write();
        for list in lists.values_mut() {
            list.freeze_all();
        }
    }
}

/// Memory statistics for the adjacency structure.
#[derive(Debug, Clone)]
pub struct AdjacencyMemoryStats {
    /// Number of entries in hot (uncompressed) storage.
    pub hot_entries: usize,
    /// Number of entries in cold (compressed) storage.
    pub cold_entries: usize,
    /// Bytes used by hot storage.
    pub hot_bytes: usize,
    /// Bytes used by cold storage.
    pub cold_bytes: usize,
    /// Number of nodes with adjacency lists.
    pub node_count: usize,
}

impl AdjacencyMemoryStats {
    /// Returns the total number of entries.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.hot_entries + self.cold_entries
    }

    /// Returns the total memory used in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.hot_bytes + self.cold_bytes
    }

    /// Returns the compression ratio for cold storage.
    ///
    /// Values > 1.0 indicate actual compression.
    #[must_use]
    pub fn cold_compression_ratio(&self) -> f64 {
        if self.cold_entries == 0 || self.cold_bytes == 0 {
            return 1.0;
        }
        let uncompressed = self.cold_entries * 16;
        uncompressed as f64 / self.cold_bytes as f64
    }

    /// Returns the overall compression ratio.
    #[must_use]
    pub fn overall_compression_ratio(&self) -> f64 {
        let total_entries = self.total_entries();
        if total_entries == 0 || self.total_bytes() == 0 {
            return 1.0;
        }
        let uncompressed = total_entries * 16;
        uncompressed as f64 / self.total_bytes() as f64
    }
}

impl Default for ChunkedAdjacency {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_adjacency() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));
        adj.add_edge(NodeId::new(0), NodeId::new(3), EdgeId::new(2));

        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&NodeId::new(1)));
        assert!(neighbors.contains(&NodeId::new(2)));
        assert!(neighbors.contains(&NodeId::new(3)));
    }

    #[test]
    fn test_out_degree() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));

        assert_eq!(adj.out_degree(NodeId::new(0)), 2);
        assert_eq!(adj.out_degree(NodeId::new(1)), 0);
    }

    #[test]
    fn test_mark_deleted() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));

        adj.mark_deleted(NodeId::new(0), EdgeId::new(0));

        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_edges_from() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(10));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(20));

        let edges = adj.edges_from(NodeId::new(0));
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(NodeId::new(1), EdgeId::new(10))));
        assert!(edges.contains(&(NodeId::new(2), EdgeId::new(20))));
    }

    #[test]
    fn test_compaction() {
        let adj = ChunkedAdjacency::with_chunk_capacity(4);

        // Add more edges than chunk capacity
        for i in 0..10 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // All edges should still be accessible
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 10);
    }

    #[test]
    fn test_edge_counts() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));
        adj.add_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(2));

        assert_eq!(adj.total_edge_count(), 3);
        assert_eq!(adj.active_edge_count(), 3);

        adj.mark_deleted(NodeId::new(0), EdgeId::new(0));

        assert_eq!(adj.total_edge_count(), 3);
        assert_eq!(adj.active_edge_count(), 2);
    }

    #[test]
    fn test_clear() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));

        adj.clear();

        assert_eq!(adj.total_edge_count(), 0);
        assert_eq!(adj.node_count(), 0);
    }

    #[test]
    fn test_chunk_compression() {
        // Create a chunk with some edges
        let mut chunk = AdjacencyChunk::new(64);

        // Add edges with various destination IDs
        for i in 0..20 {
            chunk.push(NodeId::new(100 + i * 5), EdgeId::new(1000 + i));
        }

        // Compress the chunk
        let compressed = chunk.compress();

        // Verify all data is preserved
        assert_eq!(compressed.len(), 20);

        // Decompress and verify
        let entries: Vec<_> = compressed.iter().collect();
        assert_eq!(entries.len(), 20);

        // After compression, entries are sorted by destination
        // Verify destinations are sorted
        for window in entries.windows(2) {
            assert!(window[0].0.as_u64() <= window[1].0.as_u64());
        }

        // Verify all original destinations are present
        let original_dsts: std::collections::HashSet<_> =
            (0..20).map(|i| NodeId::new(100 + i * 5)).collect();
        let compressed_dsts: std::collections::HashSet<_> =
            entries.iter().map(|(d, _)| *d).collect();
        assert_eq!(original_dsts, compressed_dsts);

        // Check compression ratio (should be > 1.0 for sorted data)
        let ratio = compressed.compression_ratio();
        assert!(
            ratio > 1.0,
            "Expected compression ratio > 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_empty_chunk_compression() {
        let chunk = AdjacencyChunk::new(64);
        let compressed = chunk.compress();

        assert_eq!(compressed.len(), 0);
        assert!(compressed.is_empty());
        assert_eq!(compressed.iter().count(), 0);
    }

    #[test]
    fn test_hot_to_cold_migration() {
        // Use small chunk capacity to trigger cold compression faster
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add many edges to force multiple chunks and cold compression
        // With chunk_capacity=8 and COLD_COMPRESSION_THRESHOLD=4,
        // we need more than 4 * 8 = 32 edges to trigger cold compression
        for i in 0..100 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        // Force compaction to trigger hot→cold migration
        adj.compact();

        // All edges should still be accessible
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 100);

        // Check memory stats
        let stats = adj.memory_stats();
        assert_eq!(stats.total_entries(), 100);

        // With 100 edges and threshold of 4 hot chunks (32 edges),
        // we should have some cold entries
        assert!(
            stats.cold_entries > 0,
            "Expected some cold entries, got {}",
            stats.cold_entries
        );
    }

    #[test]
    fn test_memory_stats() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges
        for i in 0..20 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        let stats = adj.memory_stats();
        assert_eq!(stats.total_entries(), 20);
        assert_eq!(stats.node_count, 1);
        assert!(stats.total_bytes() > 0);
    }

    #[test]
    fn test_freeze_all() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add some edges
        for i in 0..30 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // Get initial stats
        let before = adj.memory_stats();

        // Freeze all hot chunks
        adj.freeze_all();

        // Get stats after freeze
        let after = adj.memory_stats();

        // All data should now be in cold storage
        assert_eq!(after.hot_entries, 0);
        assert_eq!(after.cold_entries, before.total_entries());

        // All edges should still be accessible
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 30);
    }

    #[test]
    fn test_cold_compression_ratio() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add many edges with sequential IDs (compresses well)
        for i in 0..200 {
            adj.add_edge(NodeId::new(0), NodeId::new(100 + i), EdgeId::new(i));
        }

        adj.compact();
        adj.freeze_all();

        let stats = adj.memory_stats();

        // Sequential data should compress well
        let ratio = stats.cold_compression_ratio();
        assert!(
            ratio > 1.5,
            "Expected cold compression ratio > 1.5, got {}",
            ratio
        );
    }

    #[test]
    fn test_deleted_edges_with_cold_storage() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges
        for i in 0..50 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // Delete some edges (some will be in cold storage after compaction)
        for i in (0..50).step_by(2) {
            adj.mark_deleted(NodeId::new(0), EdgeId::new(i));
        }

        // Should have half the edges
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 25);

        // Verify only odd-numbered destinations remain
        for neighbor in neighbors {
            assert_eq!(neighbor.as_u64() % 2, 0); // Original IDs were i+1, so even means odd i
        }
    }

    #[test]
    fn test_adjacency_list_memory_size() {
        let mut list = AdjacencyList::new();

        // Add edges (use large chunk_capacity to avoid auto-compaction here)
        for i in 0..50 {
            list.add_edge(NodeId::new(i + 1), EdgeId::new(i), 64);
        }

        // Compact with small chunk capacity to get multiple chunks
        list.compact(8);

        let size = list.memory_size();
        assert!(size > 0);

        // Size should be roughly proportional to entry count
        // Each entry is 16 bytes uncompressed
        assert!(size <= 50 * 16 + 200); // Allow some overhead
    }

    #[test]
    fn test_cold_iteration_order() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges in order
        for i in 0..50 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // Collect all edges
        let edges = adj.edges_from(NodeId::new(0));

        // All edges should be present
        assert_eq!(edges.len(), 50);

        // Verify edge IDs are present (may be reordered due to compression sorting)
        let edge_ids: std::collections::HashSet<_> = edges.iter().map(|(_, e)| *e).collect();
        for i in 0..50 {
            assert!(edge_ids.contains(&EdgeId::new(i)));
        }
    }

    #[test]
    fn test_in_degree() {
        let adj = ChunkedAdjacency::new();

        // Simulate backward adjacency: edges stored as (dst, src)
        // Edge 1->2: backward stores (2, 1)
        // Edge 3->2: backward stores (2, 3)
        adj.add_edge(NodeId::new(2), NodeId::new(1), EdgeId::new(0)); // 1->2 in backward
        adj.add_edge(NodeId::new(2), NodeId::new(3), EdgeId::new(1)); // 3->2 in backward

        // In-degree of node 2 is 2 (two edges point to it)
        assert_eq!(adj.in_degree(NodeId::new(2)), 2);

        // Node 1 has no incoming edges
        assert_eq!(adj.in_degree(NodeId::new(1)), 0);
    }

    #[test]
    fn test_bidirectional_edges() {
        let forward = ChunkedAdjacency::new();
        let backward = ChunkedAdjacency::new();

        // Add edge: 1 -> 2
        let edge_id = EdgeId::new(100);
        forward.add_edge(NodeId::new(1), NodeId::new(2), edge_id);
        backward.add_edge(NodeId::new(2), NodeId::new(1), edge_id); // Reverse for backward!

        // Forward: edges from node 1 → returns (dst=2, edge_id)
        let forward_edges = forward.edges_from(NodeId::new(1));
        assert_eq!(forward_edges.len(), 1);
        assert_eq!(forward_edges[0], (NodeId::new(2), edge_id));

        // Forward: node 2 has no outgoing edges
        assert_eq!(forward.edges_from(NodeId::new(2)).len(), 0);

        // Backward: edges to node 2 → stored as edges_from(2) → returns (src=1, edge_id)
        let backward_edges = backward.edges_from(NodeId::new(2));
        assert_eq!(backward_edges.len(), 1);
        assert_eq!(backward_edges[0], (NodeId::new(1), edge_id));

        // Backward: node 1 has no incoming edges
        assert_eq!(backward.edges_from(NodeId::new(1)).len(), 0);
    }

    #[test]
    fn test_bidirectional_chain() {
        // Test chain: A -> B -> C
        let forward = ChunkedAdjacency::new();
        let backward = ChunkedAdjacency::new();

        let a = NodeId::new(1);
        let b = NodeId::new(2);
        let c = NodeId::new(3);

        // Edge A -> B
        let edge_ab = EdgeId::new(10);
        forward.add_edge(a, b, edge_ab);
        backward.add_edge(b, a, edge_ab);

        // Edge B -> C
        let edge_bc = EdgeId::new(20);
        forward.add_edge(b, c, edge_bc);
        backward.add_edge(c, b, edge_bc);

        // Forward traversal from A: should reach B
        let from_a = forward.edges_from(a);
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].0, b);

        // Forward traversal from B: should reach C
        let from_b = forward.edges_from(b);
        assert_eq!(from_b.len(), 1);
        assert_eq!(from_b[0].0, c);

        // Backward traversal to C: should find B
        let to_c = backward.edges_from(c);
        assert_eq!(to_c.len(), 1);
        assert_eq!(to_c[0].0, b);

        // Backward traversal to B: should find A
        let to_b = backward.edges_from(b);
        assert_eq!(to_b.len(), 1);
        assert_eq!(to_b[0].0, a);

        // Node A has no incoming edges
        assert_eq!(backward.edges_from(a).len(), 0);

        // Node C has no outgoing edges
        assert_eq!(forward.edges_from(c).len(), 0);
    }

    // === Skip Index Tests ===

    #[test]
    fn test_contains_edge_basic() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));
        adj.add_edge(NodeId::new(0), NodeId::new(3), EdgeId::new(2));

        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(1)));
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(2)));
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(3)));
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(4)));
        assert!(!adj.contains_edge(NodeId::new(1), NodeId::new(0)));
    }

    #[test]
    fn test_contains_edge_after_delete() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(10));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(20));

        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(1)));

        adj.mark_deleted(NodeId::new(0), EdgeId::new(10));

        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(1)));
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(2)));
    }

    #[test]
    fn test_contains_edge_in_cold_storage() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add enough edges to trigger hot→cold compression
        for i in 0..100 {
            adj.add_edge(NodeId::new(0), NodeId::new(100 + i), EdgeId::new(i));
        }

        adj.compact();
        adj.freeze_all();

        // All edges should be findable via skip index
        for i in 0..100 {
            assert!(
                adj.contains_edge(NodeId::new(0), NodeId::new(100 + i)),
                "Should find destination {} in cold storage",
                100 + i
            );
        }

        // Non-existent destinations should not be found
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(0)));
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(99)));
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(200)));
    }

    #[test]
    fn test_contains_edge_in_delta_only() {
        let adj = ChunkedAdjacency::new();

        // Add just a few edges (stays in delta buffer)
        adj.add_edge(NodeId::new(0), NodeId::new(5), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(10), EdgeId::new(1));

        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(5)));
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(10)));
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(7)));
    }

    #[test]
    fn test_edges_in_range() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges with destinations 100..200
        for i in 0..100 {
            adj.add_edge(NodeId::new(0), NodeId::new(100 + i), EdgeId::new(i));
        }

        adj.compact();

        // Range [130, 140] should return 11 edges (130..=140)
        let results = adj.edges_in_range(NodeId::new(0), NodeId::new(130), NodeId::new(140));
        assert_eq!(
            results.len(),
            11,
            "Expected 11 edges in range [130, 140], got {}",
            results.len()
        );

        // Verify all results are in range
        for (dst, _) in &results {
            assert!((130..=140).contains(&dst.as_u64()));
        }

        // Out-of-range query should return empty
        let empty = adj.edges_in_range(NodeId::new(0), NodeId::new(200), NodeId::new(300));
        assert!(empty.is_empty());
    }

    #[test]
    fn test_skip_index_prunes_chunks() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges in distinct ranges to create separate cold chunks
        // Range A: destinations 100-107, Range B: 200-207, Range C: 300-307
        for i in 0..8 {
            adj.add_edge(NodeId::new(0), NodeId::new(100 + i), EdgeId::new(i));
        }
        adj.compact();

        for i in 0..8 {
            adj.add_edge(NodeId::new(0), NodeId::new(200 + i), EdgeId::new(100 + i));
        }
        adj.compact();

        for i in 0..8 {
            adj.add_edge(NodeId::new(0), NodeId::new(300 + i), EdgeId::new(200 + i));
        }
        adj.compact();
        adj.freeze_all();

        // contains_edge for each range
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(103)));
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(205)));
        assert!(adj.contains_edge(NodeId::new(0), NodeId::new(307)));

        // Values between ranges should not exist
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(150)));
        assert!(!adj.contains_edge(NodeId::new(0), NodeId::new(250)));

        // Range query spanning only one chunk range
        let range_b = adj.edges_in_range(NodeId::new(0), NodeId::new(200), NodeId::new(207));
        assert_eq!(range_b.len(), 8);
    }

    #[test]
    fn test_contains_after_freeze_all() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        for i in 0..50 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();
        adj.freeze_all();

        // Verify all edges are in cold storage
        let stats = adj.memory_stats();
        assert_eq!(stats.hot_entries, 0);
        assert_eq!(stats.cold_entries, 50);

        // All edges should still be findable
        for i in 0..50 {
            assert!(
                adj.contains_edge(NodeId::new(0), NodeId::new(i + 1)),
                "Should find destination {} after freeze_all",
                i + 1
            );
        }
    }

    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    #[test]
    fn unpublished_edge_receipt_rolls_back_post_physical_pre_counter_panic() {
        for adjacency in [ChunkedAdjacency::new(), ChunkedAdjacency::new()] {
            let src = NodeId::new(7);
            let dst = NodeId::new(8);
            let edge = EdgeId::new(9);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut publication = adjacency
                    .begin_unpublished_edge(src, edge)
                    .expect("capacity")
                    .expect("vacant identity");
                publication.publish_with_hook(dst, || panic!("injected adjacency panic"));
            }));
            assert!(caught.is_err());
            assert_eq!(adjacency.total_edge_count(), 0);
            assert!(adjacency.neighbors(src).is_empty());

            let mut retry = adjacency
                .begin_unpublished_edge(src, edge)
                .expect("capacity")
                .expect("rollback restored exact vacancy");
            retry.publish(dst);
            retry.commit();
            assert_eq!(adjacency.total_edge_count(), 1);
            assert_eq!(adjacency.neighbors(src), vec![dst]);
        }
    }
}
