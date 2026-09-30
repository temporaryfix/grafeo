//! Compressed Sparse Row adjacency representation.
//!
//! For node i, its neighbors are `targets[offsets[i]..offsets[i+1]]`.
//! Uses u32 for both offsets and targets (max ~4B nodes/edges per table).
//!
//! Task 4 / Option A: [`crate::graph::compact::csr::PackedOpenAdjacency`] is the interval-annotated fat
//! CSR (open rows packed as a prefix of each source run). Current 1-hop is
//! that prefix slice or a [`crate::graph::compact::csr::build_current_csr_from_open_edges`] derived
//! [`crate::graph::compact::csr::CsrAdjacency`] — never `filter(is_open)` on the fat run.

use grafeo_common::types::{EdgeId, EpochId, EpochInterval, EpochZoneMap};

/// Compressed Sparse Row adjacency structure.
///
/// Stores a directed graph in two flat arrays: `offsets` (one per node + 1
/// sentinel) and `targets` (concatenated neighbor lists). This layout is
/// cache-friendly for forward traversal and has O(1) neighbor access.
#[derive(Debug, Clone)]
pub struct CsrAdjacency {
    /// One entry per node plus a trailing sentinel.
    /// `offsets[i]..offsets[i+1]` is the range in `targets` for node `i`.
    offsets: Vec<u32>,
    /// Concatenated target node offsets, grouped by source.
    targets: Vec<u32>,
    /// Optional per-edge auxiliary data, parallel to `targets`.
    /// For backward CSRs, stores the corresponding forward CSR position.
    edge_data: Option<Vec<u32>>,
}

impl CsrAdjacency {
    /// Empty CSR: `num_nodes` sources, no edges.
    #[must_use]
    pub fn empty(num_nodes: usize) -> Self {
        Self {
            offsets: vec![0u32; num_nodes + 1],
            targets: Vec::new(),
            edge_data: None,
        }
    }

    /// Builds a CSR from pre-sorted `(src, dst)` pairs.
    ///
    /// The input **must** be sorted by `src`. Each source run is dest-sorted.
    /// `num_nodes` is the total number of source nodes, nodes beyond the
    /// highest `src` in `edges` are treated as having zero out-degree.
    /// Grown if any `src` is `>= num_nodes` (backward CSRs are dest-indexed;
    /// Source→Entity tables have `|dst| > |src|`).
    ///
    /// # Panics
    ///
    /// Panics if `edges` is not sorted by source.
    #[must_use]
    pub fn from_sorted_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        assert!(
            edges.windows(2).all(|w| w[0].0 <= w[1].0),
            "edges must be sorted by source"
        );

        let max_src = edges
            .iter()
            .map(|&(s, _)| s as usize + 1)
            .max()
            .unwrap_or(0);
        let num_nodes = num_nodes.max(max_src);
        let mut offsets = vec![0u32; num_nodes + 1];

        // Count edges per source.
        for &(src, _) in edges {
            offsets[src as usize + 1] += 1;
        }

        // Prefix sum.
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let mut targets: Vec<u32> = edges.iter().map(|&(_, dst)| dst).collect();
        // Dest-sort each source run so leapfrog can intersect slices.
        for i in 0..num_nodes {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            targets[start..end].sort_unstable();
        }

        Self {
            offsets,
            targets,
            edge_data: None,
        }
    }

    /// Sets optional per-edge auxiliary data parallel to `targets`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `self.targets.len()`.
    pub fn set_edge_data(&mut self, data: Vec<u32>) {
        assert_eq!(
            data.len(),
            self.targets.len(),
            "edge_data length must equal targets length"
        );
        self.edge_data = Some(data);
    }

    /// Returns `true` if per-edge auxiliary data has been set.
    #[must_use]
    pub fn has_edge_data(&self) -> bool {
        self.edge_data.is_some()
    }

    /// Returns the auxiliary data for the edge at the given CSR position.
    ///
    /// Returns `None` if no edge data has been set, or if the position is
    /// out of bounds.
    #[must_use]
    pub fn edge_data_at(&self, position: usize) -> Option<u32> {
        self.edge_data.as_ref()?.get(position).copied()
    }

    /// Returns the number of nodes in this CSR.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        // offsets has num_nodes + 1 entries.
        self.offsets.len().saturating_sub(1)
    }

    /// Returns the total number of edges in this CSR.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.targets.len()
    }

    /// Returns the neighbors (target offsets) of the given node.
    ///
    /// Returns an empty slice if `node_offset` is out of range.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, node_offset: u32) -> &[u32] {
        let i = node_offset as usize;
        if i + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.targets[start..end]
    }

    /// Returns the out-degree of the given node.
    ///
    /// Returns 0 if `node_offset` is out of range.
    #[inline]
    #[must_use]
    pub fn degree(&self, node_offset: u32) -> usize {
        self.neighbors(node_offset).len()
    }

    /// Finds the source node for a given CSR position via binary search.
    ///
    /// The CSR position is an index into `targets`. This method returns the
    /// node offset `i` such that `offsets[i] <= position < offsets[i+1]`.
    /// Returns `None` if `position` is out of range.
    #[must_use]
    pub fn source_for_position(&self, position: u32) -> Option<u32> {
        if position as usize >= self.targets.len() {
            return None;
        }

        // Binary search: find the last offset <= position.
        // offsets is monotonically non-decreasing with len = num_nodes + 1.
        let num_nodes = self.num_nodes();
        let mut lo = 0usize;
        let mut hi = num_nodes;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.offsets[mid + 1] <= position {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // reason: CSR position index fits u32
        #[allow(clippy::cast_possible_truncation)]
        Some(lo as u32)
    }

    /// Returns the starting CSR position (index into `targets`) for the given node.
    ///
    /// This is `offsets[node_offset]`, the index at which this node's
    /// neighbor list begins in the targets array.
    ///
    /// Returns 0 if `node_offset` is out of range.
    #[inline]
    #[must_use]
    pub fn offset_of(&self, node_offset: u32) -> u32 {
        let i = node_offset as usize;
        if i >= self.offsets.len() {
            return 0;
        }
        self.offsets[i]
    }

    /// Reconstructs from pre-built raw parts.
    ///
    /// Used by section deserialization.
    #[must_use]
    pub fn from_raw_parts(
        offsets: Vec<u32>,
        targets: Vec<u32>,
        edge_data: Option<Vec<u32>>,
    ) -> Self {
        Self {
            offsets,
            targets,
            edge_data,
        }
    }

    /// Returns the raw offsets array.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Returns the raw targets array.
    #[must_use]
    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    /// Returns the raw edge_data array, if set.
    #[must_use]
    pub fn edge_data(&self) -> Option<&[u32]> {
        self.edge_data.as_deref()
    }

    /// Serializes this CSR to a byte buffer.
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        // offsets
        write_usize_as_u32(buf, self.offsets.len());
        for &o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        // targets
        write_usize_as_u32(buf, self.targets.len());
        for &t in &self.targets {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        // edge_data
        match &self.edge_data {
            Some(ed) => {
                buf.push(1);
                write_usize_as_u32(buf, ed.len());
                for &d in ed {
                    buf.extend_from_slice(&d.to_le_bytes());
                }
            }
            None => buf.push(0),
        }
    }

    /// Deserializes a CSR from a byte buffer at the given offset.
    ///
    /// # Errors
    ///
    /// Returns an error string if data is truncated.
    pub fn read_from(data: &[u8], pos: &mut usize) -> Result<Self, &'static str> {
        let offsets_len = read_u32_le(data, pos)? as usize;
        let mut offsets = Vec::with_capacity(offsets_len);
        for _ in 0..offsets_len {
            offsets.push(read_u32_le(data, pos)?);
        }
        let targets_len = read_u32_le(data, pos)? as usize;
        let mut targets = Vec::with_capacity(targets_len);
        for _ in 0..targets_len {
            targets.push(read_u32_le(data, pos)?);
        }
        let has_edge_data = *data.get(*pos).ok_or("truncated edge_data flag")?;
        *pos += 1;
        let edge_data = if has_edge_data == 1 {
            let ed_len = read_u32_le(data, pos)? as usize;
            let mut ed = Vec::with_capacity(ed_len);
            for _ in 0..ed_len {
                ed.push(read_u32_le(data, pos)?);
            }
            Some(ed)
        } else {
            None
        };
        Ok(Self::from_raw_parts(offsets, targets, edge_data))
    }

    /// Returns the approximate heap memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<u32>()
            + self.targets.len() * std::mem::size_of::<u32>()
            + self
                .edge_data
                .as_ref()
                .map_or(0, |d| d.len() * std::mem::size_of::<u32>())
    }
}

/// One interval-annotated edge version used to pack Option A adjacency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalEdgeRow {
    /// Source node offset (table-local).
    pub src: u32,
    /// Destination node offset (table-local).
    pub dst: u32,
    /// Structural validity of this version.
    pub validity: EpochInterval,
    /// Original edge id (parallel to the fat row).
    pub edge_id: EdgeId,
}

/// `EpochId::PENDING` stored as `u32::MAX`; other epochs must fit in `u32`.
const EPOCH_U32_PENDING: u32 = u32::MAX;

pub(crate) fn pack_epoch(e: EpochId) -> u32 {
    if e == EpochId::PENDING {
        EPOCH_U32_PENDING
    } else {
        u32::try_from(e.as_u64()).unwrap_or(EPOCH_U32_PENDING - 1)
    }
}

pub(crate) fn unpack_epoch(v: u32) -> EpochId {
    if v == EPOCH_U32_PENDING {
        EpochId::PENDING
    } else {
        EpochId::new(u64::from(v))
    }
}

fn pack_interval(iv: EpochInterval) -> (u32, u32) {
    (pack_epoch(iv.from()), pack_epoch(iv.to()))
}

pub(crate) fn unpack_interval(from: u32, to: u32) -> EpochInterval {
    EpochInterval::closed(unpack_epoch(from), unpack_epoch(to))
}

/// Original [`EdgeId`]s stored in a narrow fixed-width absolute or delta
/// encoding. A high-valued but narrow id range uses a `u64` base plus `u16`
/// offsets.
#[derive(Debug, Clone, Default)]
pub(crate) enum CompactEdgeIds {
    #[default]
    Empty,
    U16(Vec<u16>),
    DeltaU16 {
        base: u64,
        offsets: Vec<u16>,
    },
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl CompactEdgeIds {
    pub(crate) fn from_ids(ids: Vec<EdgeId>) -> Self {
        if ids.is_empty() {
            return Self::Empty;
        }
        if ids.iter().all(|id| u16::try_from(id.as_u64()).is_ok()) {
            return Self::U16(
                ids.into_iter()
                    .map(|id| {
                        u16::try_from(id.as_u64()).expect("all edge ids were checked to fit u16")
                    })
                    .collect(),
            );
        }
        let base = ids
            .iter()
            .map(|id| id.as_u64())
            .min()
            .expect("non-empty edge id list has a minimum");
        if ids.iter().all(|id| {
            id.as_u64()
                .checked_sub(base)
                .is_some_and(|offset| u16::try_from(offset).is_ok())
        }) {
            return Self::DeltaU16 {
                base,
                offsets: ids
                    .into_iter()
                    .map(|id| {
                        let offset = id
                            .as_u64()
                            .checked_sub(base)
                            .expect("minimum edge id cannot exceed an id in the same set");
                        u16::try_from(offset).expect("all edge id offsets were checked to fit u16")
                    })
                    .collect(),
            };
        }
        if ids.iter().all(|id| u32::try_from(id.as_u64()).is_ok()) {
            Self::U32(
                ids.into_iter()
                    .map(|id| {
                        u32::try_from(id.as_u64()).expect("all edge ids were checked to fit u32")
                    })
                    .collect(),
            )
        } else {
            Self::U64(ids.into_iter().map(|id| id.as_u64()).collect())
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::U16(v) => v.is_empty(),
            Self::DeltaU16 { offsets, .. } => offsets.is_empty(),
            Self::U32(v) => v.is_empty(),
            Self::U64(v) => v.is_empty(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::U16(v) => v.len(),
            Self::DeltaU16 { offsets, .. } => offsets.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
        }
    }

    pub(crate) fn get(&self, i: usize) -> Option<EdgeId> {
        match self {
            Self::Empty => None,
            Self::U16(v) => v.get(i).copied().map(|id| EdgeId::new(u64::from(id))),
            Self::DeltaU16 { base, offsets } => offsets
                .get(i)
                .copied()
                .map(|offset| EdgeId::new(*base + u64::from(offset))),
            Self::U32(v) => v.get(i).copied().map(|id| EdgeId::new(u64::from(id))),
            Self::U64(v) => v.get(i).copied().map(EdgeId::new),
        }
    }

    pub(crate) fn iter(&self) -> CompactEdgeIdIter<'_> {
        CompactEdgeIdIter { ids: self, i: 0 }
    }

    pub(crate) fn to_vec(&self) -> Vec<EdgeId> {
        self.iter().collect()
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::U16(v) => v.len() * std::mem::size_of::<u16>(),
            Self::DeltaU16 { offsets, .. } => offsets.len() * std::mem::size_of::<u16>(),
            Self::U32(v) => v.len() * std::mem::size_of::<u32>(),
            Self::U64(v) => v.len() * std::mem::size_of::<u64>(),
        }
    }
}

pub(crate) struct CompactEdgeIdIter<'a> {
    ids: &'a CompactEdgeIds,
    i: usize,
}

impl Iterator for CompactEdgeIdIter<'_> {
    type Item = EdgeId;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.ids.get(self.i)?;
        self.i += 1;
        Some(id)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.ids.len().saturating_sub(self.i);
        (n, Some(n))
    }
}

/// Per-row packed intervals, collapsed when every row shares one window.
#[derive(Debug, Clone)]
enum PackedValidity {
    Uniform { from: u32, to: u32 },
    Rows { from: Vec<u32>, to: Vec<u32> },
}

impl PackedValidity {
    fn from_pairs(from: Vec<u32>, to: Vec<u32>) -> Self {
        debug_assert_eq!(from.len(), to.len());
        if let (Some(&f0), Some(&t0)) = (from.first(), to.first())
            && from.iter().all(|&f| f == f0)
            && to.iter().all(|&t| t == t0)
        {
            return Self::Uniform { from: f0, to: t0 };
        }
        Self::Rows { from, to }
    }

    fn get(&self, i: usize) -> Option<(u32, u32)> {
        match self {
            Self::Uniform { from, to } => Some((*from, *to)),
            Self::Rows { from, to } => Some((*from.get(i)?, *to.get(i)?)),
        }
    }

    fn extend_into(&self, start: usize, end: usize, from: &mut Vec<u32>, to: &mut Vec<u32>) {
        match self {
            Self::Uniform { from: f, to: t } => {
                let n = end.saturating_sub(start);
                from.extend(std::iter::repeat_n(*f, n));
                to.extend(std::iter::repeat_n(*t, n));
            }
            Self::Rows { from: fs, to: ts } => {
                from.extend_from_slice(&fs[start..end]);
                to.extend_from_slice(&ts[start..end]);
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Uniform { .. } => 0,
            Self::Rows { from, to } => from.len() * 4 + to.len() * 4,
        }
    }
}

/// Interval-annotated CSR of every edge version (Option A).
///
/// Open (still-current) rows are packed at the front of each source run so
/// current 1-hop is `targets[offsets[i]..open_ends[i]]` — the same slice walk
/// as [`CsrAdjacency::neighbors`]. Closed tails sit after the prefix for as-of.
/// Current reads must not scan the fat run and `filter(is_open)`.
///
/// Validity is stored as `u32` epoch pairs in RAM (8 B/row). `PENDING` is
/// `u32::MAX`. v5 persist still writes 16 B [`EpochInterval`]s.
#[derive(Debug, Clone)]
pub struct PackedOpenAdjacency {
    offsets: Vec<u32>,
    /// Exclusive open-prefix ends. Empty means closed-only (`open_end(i) == offsets[i]`).
    open_ends: Vec<u32>,
    targets: Vec<u32>,
    validity: PackedValidity,
    edge_ids: Vec<EdgeId>,
    /// Per-source epoch coverage (rebuilt on load; not persisted).
    src_zone: Vec<EpochZoneMap>,
}

impl PackedOpenAdjacency {
    /// Packs `rows` with open versions first in each source run.
    ///
    /// `num_nodes` is the source-table size; grown if a row's `src` is larger.
    ///
    /// # Panics
    ///
    /// Panics if the packed target count exceeds `u32::MAX`.
    #[must_use]
    pub fn from_rows(num_nodes: usize, rows: &[TemporalEdgeRow]) -> Self {
        let max_src = rows.iter().map(|r| r.src as usize + 1).max().unwrap_or(0);
        let num_nodes = num_nodes.max(max_src);

        let mut open: Vec<Vec<(u32, EpochInterval, EdgeId)>> = vec![Vec::new(); num_nodes];
        let mut closed: Vec<Vec<(u32, EpochInterval, EdgeId)>> = vec![Vec::new(); num_nodes];
        for row in rows {
            let bucket = if row.validity.is_open() {
                &mut open[row.src as usize]
            } else {
                &mut closed[row.src as usize]
            };
            bucket.push((row.dst, row.validity, row.edge_id));
        }
        for bucket in open.iter_mut().chain(closed.iter_mut()) {
            bucket.sort_unstable_by_key(|&(dst, _, _)| dst);
        }

        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut open_ends = Vec::with_capacity(num_nodes);
        let mut targets = Vec::with_capacity(rows.len());
        let mut from_e = Vec::with_capacity(rows.len());
        let mut to_e = Vec::with_capacity(rows.len());
        let mut edge_ids = Vec::with_capacity(rows.len());
        offsets.push(0);
        for i in 0..num_nodes {
            for &(dst, iv, id) in &open[i] {
                targets.push(dst);
                let (f, t) = pack_interval(iv);
                from_e.push(f);
                to_e.push(t);
                edge_ids.push(id);
            }
            open_ends.push(u32::try_from(targets.len()).expect("csr targets fit u32"));
            for &(dst, iv, id) in &closed[i] {
                debug_assert!(!iv.is_open(), "closed tail must not contain open rows");
                targets.push(dst);
                let (f, t) = pack_interval(iv);
                from_e.push(f);
                to_e.push(t);
                edge_ids.push(id);
            }
            offsets.push(u32::try_from(targets.len()).expect("csr targets fit u32"));
        }
        debug_assert_eq!(open_ends.len(), num_nodes);
        debug_assert!(open_ends.iter().zip(offsets.iter()).all(|(e, o)| e >= o));

        let validity = PackedValidity::from_pairs(from_e, to_e);
        let src_zone = src_zones_from_packed(&offsets, &validity);
        Self {
            offsets,
            open_ends,
            targets,
            validity,
            edge_ids,
            src_zone,
        }
    }

    fn open_end(&self, i: usize) -> u32 {
        self.open_ends.get(i).copied().unwrap_or(self.offsets[i])
    }

    /// Interval of fat-run row `i`.
    #[must_use]
    pub fn interval(&self, i: usize) -> Option<EpochInterval> {
        let (from, to) = self.validity.get(i)?;
        Some(unpack_interval(from, to))
    }

    /// Number of source nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total versions (open + closed).
    #[must_use]
    pub fn num_versions(&self) -> usize {
        self.targets.len()
    }

    /// Open (current) row count — sum of prefix lengths.
    #[must_use]
    pub fn num_open(&self) -> usize {
        self.open_ends
            .iter()
            .enumerate()
            .map(|(i, &end)| end.saturating_sub(self.offsets[i]) as usize)
            .sum()
    }

    /// Open `(dst, edge_id)` pairs for current 1-hop.
    #[must_use]
    pub fn current_edges(&self, src: u32) -> Vec<(u32, EdgeId)> {
        let i = src as usize;
        if i >= self.num_nodes() {
            return Vec::new();
        }
        let start = self.offsets[i] as usize;
        let end = self.open_end(i) as usize;
        self.targets[start..end]
            .iter()
            .zip(self.edge_ids[start..end].iter())
            .map(|(&d, &e)| (d, e))
            .collect()
    }

    /// Current 1-hop: the open prefix slice. Does not inspect `validity`.
    #[inline]
    #[must_use]
    pub fn current_neighbors(&self, src: u32) -> &[u32] {
        let i = src as usize;
        if i >= self.num_nodes() {
            return &[];
        }
        let start = self.offsets[i] as usize;
        let end = self.open_end(i) as usize;
        &self.targets[start..end]
    }

    /// Epoch coverage of source `src` (empty zone if out of range).
    #[inline]
    #[must_use]
    pub fn src_zone(&self, src: u32) -> EpochZoneMap {
        self.src_zone
            .get(src as usize)
            .copied()
            .unwrap_or(EpochZoneMap::EMPTY)
    }

    /// Appends neighbors visible at `epoch` onto `out` (does not clear).
    ///
    /// `PENDING` copies the open prefix. Other epochs skip the source when
    /// its [`Self::src_zone`] cannot contain `epoch`, else filter `validity.contains`.
    pub fn extend_neighbors_at_epoch(&self, src: u32, epoch: EpochId, out: &mut Vec<u32>) {
        let i = src as usize;
        if i + 1 >= self.offsets.len() {
            return;
        }
        if epoch == EpochId::PENDING {
            out.extend_from_slice(self.current_neighbors(src));
            return;
        }
        if !self.src_zone.is_empty() && !self.src_zone(src).may_contain(epoch) {
            return;
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        for e in start..end {
            if self.interval(e).is_some_and(|iv| iv.contains(epoch)) {
                out.push(self.targets[e]);
            }
        }
    }

    /// Fills `out` with neighbors visible at `epoch` (clears `out` first).
    pub fn fill_neighbors_at_epoch(&self, src: u32, epoch: EpochId, out: &mut Vec<u32>) {
        out.clear();
        self.extend_neighbors_at_epoch(src, epoch, out);
    }

    /// Neighbors visible at `epoch` (see [`Self::fill_neighbors_at_epoch`]).
    #[must_use]
    pub fn neighbors_at_epoch(&self, src: u32, epoch: EpochId) -> Vec<u32> {
        let mut out = Vec::new();
        self.fill_neighbors_at_epoch(src, epoch, &mut out);
        out
    }

    /// Appends `(target, edge_id)` pairs visible at `epoch` (does not clear).
    pub fn extend_edges_at_epoch(&self, src: u32, epoch: EpochId, out: &mut Vec<(u32, EdgeId)>) {
        let i = src as usize;
        if i + 1 >= self.offsets.len() {
            return;
        }
        if epoch == EpochId::PENDING {
            let start = self.offsets[i] as usize;
            let end = self.open_end(i) as usize;
            out.extend(
                self.targets[start..end]
                    .iter()
                    .zip(self.edge_ids[start..end].iter())
                    .map(|(&dst, &eid)| (dst, eid)),
            );
            return;
        }
        if !self.src_zone.is_empty() && !self.src_zone(src).may_contain(epoch) {
            return;
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        for e in start..end {
            if self.interval(e).is_some_and(|iv| iv.contains(epoch)) {
                out.push((self.targets[e], self.edge_ids[e]));
            }
        }
    }

    /// `(target, edge_id)` pairs visible at `epoch` (packed fat run).
    ///
    /// `PENDING` is the open prefix. Other epochs filter `validity.contains`.
    #[must_use]
    pub fn edges_at_epoch(&self, src: u32, epoch: EpochId) -> Vec<(u32, EdgeId)> {
        let mut out = Vec::new();
        self.extend_edges_at_epoch(src, epoch, &mut out);
        out
    }

    /// Incoming `(src, edge_id)` pairs visible at `epoch` (scans the fat run).
    ///
    /// Used when the fat backward twin is not kept in RAM. Current incoming
    /// should prefer a tight reverse CSR instead of this scan.
    #[must_use]
    pub fn incoming_at_epoch(&self, dst: u32, epoch: EpochId) -> Vec<(u32, EdgeId)> {
        let mut out = Vec::new();
        for src in 0..self.num_nodes() {
            let src_u = u32::try_from(src).unwrap_or(u32::MAX);
            if epoch == EpochId::PENDING {
                for (i, &target) in self.current_neighbors(src_u).iter().enumerate() {
                    if target == dst {
                        let start = self.offsets[src] as usize;
                        out.push((src_u, self.edge_ids[start + i]));
                    }
                }
                continue;
            }
            if !self.src_zone.is_empty() && !self.src_zone(src_u).may_contain(epoch) {
                continue;
            }
            let start = self.offsets[src] as usize;
            let end = self.offsets[src + 1] as usize;
            for i in start..end {
                if self.targets[i] == dst && self.interval(i).is_some_and(|iv| iv.contains(epoch)) {
                    out.push((src_u, self.edge_ids[i]));
                }
            }
        }
        out
    }

    /// Tight reverse CSR of the open prefix (`edge_data` = fat-run position).
    #[must_use]
    pub fn derive_current_bwd(&self, num_dst: usize) -> CsrAdjacency {
        let mut triples: Vec<(u32, u32, u32)> = Vec::with_capacity(self.num_open());
        for src in 0..self.num_nodes() {
            let src_u = u32::try_from(src).unwrap_or(u32::MAX);
            let start = self.offsets[src];
            let end = self.open_end(src);
            for pos in start..end {
                let dst = self.targets[pos as usize];
                triples.push((dst, src_u, pos));
            }
        }
        triples.sort_unstable_by_key(|&(dst, src, _)| (dst, src));
        let max_dst = triples.last().map_or(0, |&(d, _, _)| d as usize + 1);
        let n = num_dst.max(max_dst);
        let edges: Vec<(u32, u32)> = triples.iter().map(|&(d, s, _)| (d, s)).collect();
        let data: Vec<u32> = triples.iter().map(|&(_, _, p)| p).collect();
        let mut csr = CsrAdjacency::from_sorted_edges(n, &edges);
        if !data.is_empty() {
            csr.set_edge_data(data);
        }
        csr
    }

    /// Create epochs of the open prefix, in derived-current CSR row order.
    #[must_use]
    pub fn open_from_epochs(&self) -> Vec<EpochId> {
        let mut out = Vec::with_capacity(self.num_open());
        for i in 0..self.num_nodes() {
            let start = self.offsets[i] as usize;
            let end = self.open_end(i) as usize;
            for pos in start..end {
                let Some((from, _)) = self.validity.get(pos) else {
                    continue;
                };
                out.push(unpack_epoch(from));
            }
        }
        out
    }

    /// Drops the open prefix from each source run. Current 1-hop must then
    /// come from a derived [`CsrAdjacency`]; this structure keeps closed
    /// tails only (as-of). `num_open` becomes 0.
    ///
    /// # Panics
    ///
    /// Panics if this already-valid CSR contains more than `u32::MAX` target
    /// rows. Such a value cannot be constructed through the checked builders.
    #[must_use]
    pub fn into_closed_only(self) -> Self {
        if self.num_open() == 0 {
            return self;
        }
        let n = self.num_nodes();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut targets = Vec::with_capacity(self.targets.len().saturating_sub(self.num_open()));
        let mut from_e = Vec::with_capacity(targets.capacity());
        let mut to_e = Vec::with_capacity(targets.capacity());
        let mut edge_ids = Vec::with_capacity(targets.capacity());
        offsets.push(0);
        for i in 0..n {
            let c0 = self.open_end(i) as usize;
            let c1 = self.offsets[i + 1] as usize;
            targets.extend_from_slice(&self.targets[c0..c1]);
            self.validity.extend_into(c0, c1, &mut from_e, &mut to_e);
            edge_ids.extend_from_slice(&self.edge_ids[c0..c1]);
            offsets.push(u32::try_from(targets.len()).expect("csr targets fit u32"));
        }
        // Closed-only: no open_ends array (open_end(i) == offsets[i]).
        Self {
            offsets,
            open_ends: Vec::new(),
            targets,
            validity: PackedValidity::from_pairs(from_e, to_e),
            edge_ids,
            src_zone: Vec::new(),
        }
    }

    /// Tight current CSR from the open prefixes (Task 4 derived current).
    #[must_use]
    pub fn derive_current_csr(&self) -> CsrAdjacency {
        let mut edges = Vec::with_capacity(self.num_open());
        for src in 0..self.num_nodes() {
            // reason: source index is a CSR node offset
            #[allow(clippy::cast_possible_truncation)]
            let src_u = src as u32;
            for &dst in self.current_neighbors(src_u) {
                edges.push((src_u, dst));
            }
        }
        CsrAdjacency::from_sorted_edges(self.num_nodes(), &edges)
    }

    /// Validity of the `open_idx`-th current (open-prefix) row.
    ///
    /// `open_idx` is a derived-current CSR position (concatenation of open
    /// prefixes), not a fat-run index.
    #[must_use]
    pub fn open_interval_at(&self, open_idx: usize) -> Option<EpochInterval> {
        let mut acc = 0usize;
        for i in 0..self.num_nodes() {
            let start = self.offsets[i] as usize;
            let end = self.open_end(i) as usize;
            let len = end.saturating_sub(start);
            if open_idx < acc + len {
                return self.interval(start + (open_idx - acc));
            }
            acc += len;
        }
        None
    }

    /// Structural intervals of the open prefix, in derived-CSR row order.
    #[must_use]
    pub fn open_validity(&self) -> Vec<EpochInterval> {
        let mut out = Vec::with_capacity(self.num_open());
        for i in 0..self.num_nodes() {
            let start = self.offsets[i] as usize;
            let end = self.open_end(i) as usize;
            for pos in start..end {
                if let Some(iv) = self.interval(pos) {
                    out.push(iv);
                }
            }
        }
        out
    }

    /// Original edge ids of the open prefix, in derived-CSR row order.
    #[must_use]
    pub fn open_edge_ids(&self) -> Vec<EdgeId> {
        let mut out = Vec::with_capacity(self.num_open());
        for i in 0..self.num_nodes() {
            let start = self.offsets[i] as usize;
            let end = self.open_end(i) as usize;
            out.extend_from_slice(&self.edge_ids[start..end]);
        }
        out
    }

    /// Exclusive open-prefix ends (`current = targets[offsets[i]..open_ends[i]]`).
    #[must_use]
    pub fn open_ends(&self) -> &[u32] {
        &self.open_ends
    }

    /// Fat-run offsets (all versions).
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Fat target list.
    #[must_use]
    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    /// Source-table offset whose fat run contains `fat_pos`.
    #[must_use]
    pub fn src_of(&self, fat_pos: usize) -> Option<u32> {
        if fat_pos >= self.targets.len() {
            return None;
        }
        let next = self.offsets.partition_point(|&o| (o as usize) <= fat_pos);
        u32::try_from(next.saturating_sub(1)).ok()
    }

    /// Per-version structural validity, parallel to [`Self::targets`].
    #[must_use]
    pub fn validity(&self) -> Vec<EpochInterval> {
        (0..self.targets.len())
            .filter_map(|i| self.interval(i))
            .collect()
    }

    /// Original edge ids, parallel to [`Self::targets`].
    #[must_use]
    pub fn edge_ids(&self) -> &[EdgeId] {
        &self.edge_ids
    }

    /// Rebuilds from serialized parts. `offsets.len() == open_ends.len() + 1`
    /// and `targets`, `validity`, `edge_ids` share one length.
    ///
    /// # Errors
    ///
    /// Returns an error if the arrays are inconsistent.
    pub fn from_raw_parts(
        offsets: Vec<u32>,
        open_ends: Vec<u32>,
        targets: Vec<u32>,
        validity: Vec<EpochInterval>,
        edge_ids: Vec<EdgeId>,
    ) -> Result<Self, &'static str> {
        if offsets.is_empty() {
            return Err("packed adjacency offsets empty");
        }
        let n_nodes = offsets.len() - 1;
        if !open_ends.is_empty() && open_ends.len() != n_nodes {
            return Err("packed adjacency offsets/open_ends length mismatch");
        }
        if targets.len() != validity.len() || targets.len() != edge_ids.len() {
            return Err("packed adjacency target/validity/edge_id length mismatch");
        }
        if offsets.last().copied() != Some(u32::try_from(targets.len()).unwrap_or(u32::MAX)) {
            return Err("packed adjacency offset sentinel != target count");
        }
        let mut from_e = Vec::with_capacity(validity.len());
        let mut to_e = Vec::with_capacity(validity.len());
        for iv in &validity {
            let (f, t) = pack_interval(*iv);
            from_e.push(f);
            to_e.push(t);
        }
        let packed_validity = PackedValidity::from_pairs(from_e, to_e);
        let src_zone = src_zones_from_packed(&offsets, &packed_validity);
        Ok(Self {
            offsets,
            open_ends,
            targets,
            validity: packed_validity,
            edge_ids,
            src_zone,
        })
    }

    /// Serializes the fat packed adjacency (v5 rel addendum).
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        write_usize_as_u32(buf, self.offsets.len());
        for &o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        write_usize_as_u32(buf, self.open_ends.len());
        for &e in &self.open_ends {
            buf.extend_from_slice(&e.to_le_bytes());
        }
        write_usize_as_u32(buf, self.targets.len());
        for &t in &self.targets {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        for i in 0..self.targets.len() {
            let iv = self
                .interval(i)
                .unwrap_or_else(|| EpochInterval::open(EpochId::INITIAL));
            buf.extend_from_slice(&iv.from().as_u64().to_le_bytes());
            buf.extend_from_slice(&iv.to().as_u64().to_le_bytes());
        }
        for id in &self.edge_ids {
            buf.extend_from_slice(&id.as_u64().to_le_bytes());
        }
    }

    /// Deserializes a packed adjacency written by [`Self::write_to`].
    ///
    /// # Errors
    ///
    /// Returns an error string if data is truncated or inconsistent.
    pub fn read_from(data: &[u8], pos: &mut usize) -> Result<Self, &'static str> {
        let offsets_len = read_u32_le(data, pos)? as usize;
        let mut offsets = Vec::with_capacity(offsets_len);
        for _ in 0..offsets_len {
            offsets.push(read_u32_le(data, pos)?);
        }
        let ends_len = read_u32_le(data, pos)? as usize;
        let mut open_ends = Vec::with_capacity(ends_len);
        for _ in 0..ends_len {
            open_ends.push(read_u32_le(data, pos)?);
        }
        let n = read_u32_le(data, pos)? as usize;
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            targets.push(read_u32_le(data, pos)?);
        }
        let mut validity = Vec::with_capacity(n);
        for _ in 0..n {
            let from = EpochId::new(read_u64_le(data, pos)?);
            let to = EpochId::new(read_u64_le(data, pos)?);
            validity.push(EpochInterval::closed(from, to));
        }
        let mut edge_ids = Vec::with_capacity(n);
        for _ in 0..n {
            edge_ids.push(EdgeId::new(read_u64_le(data, pos)?));
        }
        Self::from_raw_parts(offsets, open_ends, targets, validity, edge_ids)
    }

    /// Approximate heap usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.offsets.len() * 4
            + self.open_ends.len() * 4
            + self.targets.len() * 4
            + self.validity.heap_bytes()
            + self.edge_ids.len() * std::mem::size_of::<EdgeId>()
            + self.src_zone.len() * std::mem::size_of::<EpochZoneMap>()
    }
}

/// Builds a tight [`CsrAdjacency`] containing only rows open at `PENDING`.
///
/// Packs an Option A prefix first, then derives the current CSR from that
/// prefix — never a `filter(is_open)` walk of a fat neighbor list at read time.
#[must_use]
pub fn build_current_csr_from_open_edges(
    num_nodes: usize,
    rows: &[TemporalEdgeRow],
) -> CsrAdjacency {
    PackedOpenAdjacency::from_rows(num_nodes, rows).derive_current_csr()
}

/// Packs temporal rows into Option A adjacency (open prefix + closed tails).
#[must_use]
pub fn pack_open_prefix(num_nodes: usize, rows: &[TemporalEdgeRow]) -> PackedOpenAdjacency {
    PackedOpenAdjacency::from_rows(num_nodes, rows)
}

fn src_zones_from_packed(offsets: &[u32], validity: &PackedValidity) -> Vec<EpochZoneMap> {
    if offsets.len() < 2 {
        return Vec::new();
    }
    let n = offsets.len() - 1;
    let mut zones = Vec::with_capacity(n);
    for i in 0..n {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let mut z = EpochZoneMap::EMPTY;
        for j in start..end {
            if let Some((from, to)) = validity.get(j) {
                z.include(unpack_interval(from, to));
            }
        }
        zones.push(z);
    }
    zones
}

fn write_usize_as_u32(buf: &mut Vec<u8>, v: usize) {
    let n = u32::try_from(v).expect("value exceeds u32::MAX in CSR serialization");
    buf.extend_from_slice(&n.to_le_bytes());
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated u32");
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64, &'static str> {
    if *pos + 8 > data.len() {
        return Err("truncated u64");
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_edge_ids_use_narrow_absolute_and_delta_encodings() {
        let absolute = CompactEdgeIds::from_ids(vec![EdgeId::new(0), EdgeId::new(65_535)]);
        assert!(matches!(&absolute, CompactEdgeIds::U16(_)));
        assert_eq!(absolute.to_vec(), vec![EdgeId::new(0), EdgeId::new(65_535)]);
        assert_eq!(absolute.heap_bytes(), 4);

        let next_width = CompactEdgeIds::from_ids(vec![EdgeId::new(0), EdgeId::new(65_536)]);
        assert!(matches!(&next_width, CompactEdgeIds::U32(_)));
        assert_eq!(
            next_width.to_vec(),
            vec![EdgeId::new(0), EdgeId::new(65_536)]
        );

        let base = u64::from(u32::MAX) + 10_000;
        let delta = CompactEdgeIds::from_ids(vec![EdgeId::new(base), EdgeId::new(base + 65_000)]);
        assert!(matches!(&delta, CompactEdgeIds::DeltaU16 { .. }));
        assert_eq!(
            delta.to_vec(),
            vec![EdgeId::new(base), EdgeId::new(base + 65_000)]
        );
        assert_eq!(delta.heap_bytes(), 4);

        let near_max = u64::MAX - 100_000;
        let near_max_delta =
            CompactEdgeIds::from_ids(vec![EdgeId::new(near_max), EdgeId::new(near_max + 65_000)]);
        assert!(matches!(&near_max_delta, CompactEdgeIds::DeltaU16 { .. }));
        assert_eq!(
            near_max_delta.to_vec(),
            vec![EdgeId::new(near_max), EdgeId::new(near_max + 65_000)]
        );

        let wide_high = CompactEdgeIds::from_ids(vec![
            EdgeId::new(1_u64 << 40),
            EdgeId::new((1_u64 << 40) + u64::from(u16::MAX) + 1),
        ]);
        assert!(matches!(&wide_high, CompactEdgeIds::U64(_)));
        assert_eq!(
            wide_high.to_vec(),
            vec![
                EdgeId::new(1_u64 << 40),
                EdgeId::new((1_u64 << 40) + u64::from(u16::MAX) + 1)
            ]
        );
    }

    #[test]
    fn test_basic_csr() {
        // 3 nodes, edges: 0->1, 0->2, 1->2
        let edges = vec![(0u32, 1u32), (0, 2), (1, 2)];
        let csr = CsrAdjacency::from_sorted_edges(3, &edges);

        assert_eq!(csr.num_nodes(), 3);
        assert_eq!(csr.num_edges(), 3);

        // Node 0: neighbors [1, 2]
        assert_eq!(csr.neighbors(0), &[1, 2]);
        assert_eq!(csr.degree(0), 2);

        // Node 1: neighbors [2]
        assert_eq!(csr.neighbors(1), &[2]);
        assert_eq!(csr.degree(1), 1);

        // Node 2: no neighbors
        assert_eq!(csr.neighbors(2), &[] as &[u32]);
        assert_eq!(csr.degree(2), 0);
    }

    #[test]
    fn from_sorted_edges_dest_sorts_each_run() {
        let edges = vec![(0u32, 7), (0, 1), (0, 3), (1, 4), (1, 2)];
        let csr = CsrAdjacency::from_sorted_edges(2, &edges);
        assert_eq!(csr.neighbors(0), &[1, 3, 7]);
        assert_eq!(csr.neighbors(1), &[2, 4]);
    }

    #[test]
    fn test_source_for_position() {
        // 3 nodes, edges: 0->1, 0->2, 1->2
        // CSR targets: [1, 2, 2]
        // offsets:      [0, 2, 3, 3]
        // position 0 -> source 0 (0->1)
        // position 1 -> source 0 (0->2)
        // position 2 -> source 1 (1->2)
        let edges = vec![(0u32, 1u32), (0, 2), (1, 2)];
        let csr = CsrAdjacency::from_sorted_edges(3, &edges);

        assert_eq!(csr.source_for_position(0), Some(0));
        assert_eq!(csr.source_for_position(1), Some(0));
        assert_eq!(csr.source_for_position(2), Some(1));

        // Out of range.
        assert_eq!(csr.source_for_position(3), None);
        assert_eq!(csr.source_for_position(100), None);
    }

    #[test]
    fn from_sorted_edges_grows_to_max_src() {
        // Backward CSRs are dest-indexed; a Source→Entity table may pass
        // dest offsets larger than the source-table count.
        let csr = CsrAdjacency::from_sorted_edges(2, &[(0, 1), (7, 0)]);
        assert_eq!(csr.num_nodes(), 8);
        assert_eq!(csr.neighbors(7), &[0]);
        assert!(csr.neighbors(2).is_empty());
    }

    #[test]
    fn test_empty_graph() {
        // 0 nodes, 0 edges.
        let csr = CsrAdjacency::from_sorted_edges(0, &[]);
        assert_eq!(csr.num_nodes(), 0);
        assert_eq!(csr.num_edges(), 0);
        assert_eq!(csr.source_for_position(0), None);
        assert_eq!(csr.memory_bytes(), 4); // 1 offset entry (sentinel)
    }

    fn row(src: u32, dst: u32, iv: EpochInterval, id: u64) -> TemporalEdgeRow {
        TemporalEdgeRow {
            src,
            dst,
            validity: iv,
            edge_id: EdgeId::new(id),
        }
    }

    /// Only open edges appear in the derived current CSR; closed lives stay
    /// addressable on the packed fat run for as-of.
    #[test]
    fn current_csr_from_open_rows_excludes_closed() {
        let e10 = EpochId::new(10);
        let e20 = EpochId::new(20);
        let rows = [
            row(0, 1, EpochInterval::open(e10), 1),
            row(0, 2, EpochInterval::closed(e10, e20), 2),
            row(1, 2, EpochInterval::open(e10), 3),
        ];

        let csr = build_current_csr_from_open_edges(3, &rows);
        assert_eq!(csr.num_edges(), 2);
        assert_eq!(csr.neighbors(0), &[1]);
        assert_eq!(csr.neighbors(1), &[2]);
        assert!(csr.neighbors(2).is_empty());

        let packed = pack_open_prefix(3, &rows);
        assert_eq!(packed.num_versions(), 3);
        assert_eq!(packed.num_open(), 2);
        // Current 1-hop is the prefix slice, not a filter over the fat run.
        assert_eq!(packed.current_neighbors(0), &[1]);
        let start = packed.offsets()[0] as usize;
        let end = packed.open_ends()[0] as usize;
        assert!(
            std::ptr::eq(
                packed.current_neighbors(0).as_ptr(),
                packed.targets()[start..end].as_ptr()
            ),
            "current 1-hop must be the open prefix slice"
        );
        assert_eq!(&packed.targets()[end..packed.offsets()[1] as usize], &[2]);

        assert_eq!(
            packed.neighbors_at_epoch(0, EpochId::PENDING),
            vec![1],
            "PENDING must copy the prefix, not filter is_open"
        );
        let mut at15 = packed.neighbors_at_epoch(0, EpochId::new(15));
        at15.sort_unstable();
        assert_eq!(at15, vec![1, 2], "as-of inside the closed life sees both");
        assert_eq!(
            packed.neighbors_at_epoch(0, EpochId::new(25)),
            vec![1],
            "as-of after close sees only the open life"
        );
        assert!(
            packed.edge_ids().iter().any(|id| *id == EdgeId::new(2)),
            "deleted edge remains addressable in the fat run"
        );
        assert_eq!(packed.src_of(0), Some(0));
        let closed_pos = packed
            .edge_ids()
            .iter()
            .position(|id| *id == EdgeId::new(2))
            .expect("closed id on fat run");
        assert_eq!(packed.src_of(closed_pos), Some(0));
        assert_eq!(packed.src_of(packed.num_versions()), None);

        let closed = packed.into_closed_only();
        assert_eq!(closed.num_open(), 0);
        assert_eq!(closed.num_versions(), 1);
        assert_eq!(closed.targets(), &[2]);
        assert_eq!(closed.edge_ids(), &[EdgeId::new(2)]);
    }

    #[test]
    fn closed_only_uniform_intervals_store_one_window() {
        let e10 = EpochId::new(10);
        let e20 = EpochId::new(20);
        let rows = [
            row(0, 1, EpochInterval::closed(e10, e20), 1),
            row(1, 2, EpochInterval::closed(e10, e20), 2),
            row(2, 0, EpochInterval::closed(e10, e20), 3),
        ];
        let packed = pack_open_prefix(3, &rows).into_closed_only();
        assert_eq!(packed.num_versions(), 3);
        assert_eq!(packed.interval(0), Some(EpochInterval::closed(e10, e20)));
        assert_eq!(packed.interval(2), Some(EpochInterval::closed(e10, e20)));
        let mixed = pack_open_prefix(
            3,
            &[
                row(0, 1, EpochInterval::closed(e10, e20), 1),
                row(1, 2, EpochInterval::closed(e10, EpochId::new(30)), 2),
            ],
        );
        assert!(
            packed.memory_bytes() < mixed.memory_bytes(),
            "uniform closed tails must not store per-row interval pairs"
        );
    }

    #[test]
    fn packed_prefix_agrees_with_derived_current_csr() {
        let rows = [
            row(0, 1, EpochInterval::open(EpochId::new(1)), 1),
            row(
                0,
                3,
                EpochInterval::closed(EpochId::new(1), EpochId::new(2)),
                2,
            ),
            row(2, 0, EpochInterval::open(EpochId::new(1)), 3),
        ];
        let packed = pack_open_prefix(4, &rows);
        let derived = packed.derive_current_csr();
        for src in 0..4u32 {
            assert_eq!(
                packed.current_neighbors(src),
                derived.neighbors(src),
                "derived current CSR must match the open prefix at src={src}"
            );
        }
    }

    /// Task 4.4 unit-level current vs as-of: prefix walk equals derived CSR;
    /// as-of still sees closed lives. Full 320k Criterion numbers stay in
    /// `asof_adjacency_spike` (Task 0: A packed +3.9% vs CSR; A-naive +13.6%).
    #[test]
    fn current_prefix_vs_asof_closed_tail() {
        let mut rows = Vec::new();
        for src in 0..64u32 {
            rows.push(row(
                src,
                (src + 1) % 64,
                EpochInterval::open(EpochId::new(10)),
                u64::from(src),
            ));
            rows.push(row(
                src,
                (src + 2) % 64,
                EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
                1000 + u64::from(src),
            ));
        }
        let packed = pack_open_prefix(64, &rows);
        let derived = packed.derive_current_csr();
        let mut acc_prefix = 0u64;
        let mut acc_derived = 0u64;
        let mut acc_asof = 0u64;
        for src in 0..64u32 {
            acc_prefix += packed.current_neighbors(src).len() as u64;
            acc_derived += derived.neighbors(src).len() as u64;
            acc_asof += packed.neighbors_at_epoch(src, EpochId::new(15)).len() as u64;
        }
        assert_eq!(acc_prefix, acc_derived);
        assert_eq!(acc_prefix, 64);
        assert_eq!(acc_asof, 128, "as-of 15 must include the closed tail");
        // Note (before/after): current path is prefix/derived (Task 0 A packed
        // +3.9%); banned naive is_open filter was +13.6% pin / +57% criterion.
    }

    #[test]
    fn packed_adjacency_write_read_round_trip() {
        let rows = vec![
            row(0, 1, EpochInterval::open(EpochId::new(10)), 1),
            row(
                0,
                2,
                EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
                2,
            ),
        ];
        let packed = pack_open_prefix(2, &rows);
        let mut buf = Vec::new();
        packed.write_to(&mut buf);
        let mut pos = 0;
        let restored = PackedOpenAdjacency::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(restored.num_nodes(), packed.num_nodes());
        assert_eq!(restored.num_versions(), packed.num_versions());
        assert_eq!(
            restored.neighbors_at_epoch(0, EpochId::new(15)),
            packed.neighbors_at_epoch(0, EpochId::new(15))
        );
        assert_eq!(restored.current_neighbors(0), packed.current_neighbors(0));
    }

    #[test]
    fn extend_neighbors_does_not_clear() {
        let rows = vec![
            row(0, 1, EpochInterval::open(EpochId::new(10)), 1),
            row(
                0,
                2,
                EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
                2,
            ),
        ];
        let packed = pack_open_prefix(2, &rows);
        let mut out = vec![99];
        packed.extend_neighbors_at_epoch(0, EpochId::new(15), &mut out);
        assert_eq!(out[0], 99);
        assert!(out.contains(&1) && out.contains(&2));
    }

    #[test]
    fn src_zone_skips_epoch_before_create() {
        let rows = vec![row(0, 1, EpochInterval::open(EpochId::new(10)), 1)];
        let packed = pack_open_prefix(1, &rows);
        assert!(!packed.src_zone(0).may_contain(EpochId::new(5)));
        let mut out = Vec::new();
        packed.extend_neighbors_at_epoch(0, EpochId::new(5), &mut out);
        assert!(out.is_empty());
    }
}
