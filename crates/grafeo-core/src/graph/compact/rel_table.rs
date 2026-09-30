//! Relationship table: double-indexed CSR for a single edge type.
//!
//! Stores all edges of one type with optional forward and backward CSR.
//! Edge properties are columnar, parallel to the forward CSR targets.

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, EpochInterval, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::csr::{CompactEdgeIds, CsrAdjacency, PackedOpenAdjacency};
use super::id::{encode_edge_id, encode_node_id};
use super::schema::EdgeSchema;
use super::temporal_column::TemporalColumn;
use crate::codec::BitPackedInts;

/// A relationship table holding all edges of a single type.
///
/// Edges are stored in a forward CSR indexed by source node offset, with an
/// optional backward CSR indexed by target node offset. Edge properties are
/// stored in columnar format, parallel to the forward CSR targets array
/// (i.e. the property at index `i` corresponds to the edge at CSR position `i`).
///
/// v5 persist writes the derived current `fwd` CSR, property value codecs,
/// plus an explicit rel addendum when [`PackedOpenAdjacency`] is present.
/// v4 and older loads reconstruct as all-open (no packed tails).
#[derive(Debug)]
pub struct RelTable {
    /// Schema describing the edge type and connected node labels.
    schema: EdgeSchema,
    /// Forward CSR, indexed by source node offset.
    fwd: CsrAdjacency,
    /// Backward CSR, indexed by target node offset. `None` means backward
    /// traversal falls back to a full scan of the forward CSR.
    /// When present, its `edge_data` stores the corresponding forward CSR
    /// position for each backward edge.
    bwd: Option<CsrAdjacency>,
    /// Edge properties, keyed by property name, parallel to forward CSR targets.
    /// Fresh bases wrap each value codec as all-open [`TemporalColumn`]s
    /// (see [`TemporalColumn::all_open`]); as-of reads go through
    /// [`Self::get_property_at_epoch`].
    properties: FxHashMap<PropertyKey, TemporalColumn>,
    /// Table ID of the source node table.
    src_table_id: u16,
    /// Table ID of the destination node table.
    dst_table_id: u16,
    /// Per-CSR-row structural validity. Empty means all-open
    /// `[INITIAL, PENDING)` (fresh builder / deserialized v4-or-older base).
    /// Parallel to the **derived current** `fwd` (open rows only).
    /// Persisted in the v5 rel addendum when packed adjacency is present.
    validity: Vec<EpochInterval>,
    /// Option A packed fat adjacency. After a no-property slim this is
    /// **closed tails only**; current 1-hop is [`Self::fwd`].
    /// Persisted reconstructed (open prefix + tails) in the v5 addendum.
    packed_fwd: Option<PackedOpenAdjacency>,
    /// Backward twin of [`Self::packed_fwd`] (open prefix of in-edges).
    packed_bwd: Option<PackedOpenAdjacency>,
    /// Create epoch of each current (`fwd`) row. Empty = all-open
    /// `[INITIAL, PENDING)`. Parallel to `fwd` targets.
    open_from: OpenEpochs,
    /// Original [`EdgeId`] of each current (`fwd`) row. Empty when current
    /// 1-hop still uses compact `(rel, csr_pos)` ids (property-bearing).
    /// Stored at the narrowest lossless absolute or delta width.
    open_edge_ids: CompactEdgeIds,
    /// Fat positions of closed packed rows, sorted by `packed.edge_ids[pos]`.
    closed_id_ord: ClosedIdOrder,
}

/// Create epochs for the tight current CSR. Uniform epochs collapse to one
/// scalar; low-cardinality mixed epochs use a dictionary with bit-packed codes;
/// high-cardinality input stays as direct packed-`u32` rows.
#[derive(Debug, Clone, Default)]
enum OpenEpochs {
    #[default]
    Empty,
    Uniform {
        epoch: u32,
        len: usize,
    },
    Dictionary {
        epochs: Vec<u32>,
        codes: BitPackedInts,
    },
    Rows(Vec<u32>),
}

impl OpenEpochs {
    fn from_epochs(epochs: Vec<u32>) -> Self {
        let Some(&epoch) = epochs.first() else {
            return Self::Empty;
        };
        if epochs.iter().all(|&candidate| candidate == epoch) {
            Self::Uniform {
                epoch,
                len: epochs.len(),
            }
        } else {
            let mut dictionary = FxHashMap::default();
            let mut distinct = Vec::new();
            let mut raw_codes = Vec::with_capacity(epochs.len());
            for &candidate in &epochs {
                let next_code =
                    u64::try_from(distinct.len()).expect("epoch dictionary cardinality fits u64");
                let code = *dictionary.entry(candidate).or_insert_with(|| {
                    distinct.push(candidate);
                    next_code
                });
                raw_codes.push(code);
            }
            let codes = BitPackedInts::pack(&raw_codes);
            let dictionary_bytes = distinct.len() * std::mem::size_of::<u32>()
                + codes.word_count() * std::mem::size_of::<u64>();
            let row_bytes = epochs.len() * std::mem::size_of::<u32>();
            if dictionary_bytes < row_bytes {
                Self::Dictionary {
                    epochs: distinct,
                    codes,
                }
            } else {
                Self::Rows(epochs)
            }
        }
    }

    fn get(&self, index: usize) -> Option<u32> {
        match self {
            Self::Empty => None,
            Self::Uniform { epoch, len } => (index < *len).then_some(*epoch),
            Self::Dictionary { epochs, codes } => {
                let code = usize::try_from(codes.get(index)?).ok()?;
                epochs.get(code).copied()
            }
            Self::Rows(epochs) => epochs.get(index).copied(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Uniform { len, .. } => *len == 0,
            Self::Dictionary { codes, .. } => codes.is_empty(),
            Self::Rows(epochs) => epochs.is_empty(),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Empty | Self::Uniform { .. } => 0,
            Self::Dictionary { epochs, codes } => {
                epochs.len() * std::mem::size_of::<u32>()
                    + codes.word_count() * std::mem::size_of::<u64>()
            }
            Self::Rows(epochs) => epochs.len() * std::mem::size_of::<u32>(),
        }
    }
}

/// Closed packed-row positions ordered by stable edge id. When packing already
/// produced that order, retain only the length instead of an identity vector.
#[derive(Debug, Clone, Default)]
enum ClosedIdOrder {
    #[default]
    Empty,
    Identity(u32),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl ClosedIdOrder {
    fn from_positions(positions: Vec<u32>) -> Self {
        if positions.is_empty() {
            return Self::Empty;
        }
        if positions
            .iter()
            .enumerate()
            .all(|(index, &position)| u32::try_from(index) == Ok(position))
        {
            Self::Identity(
                u32::try_from(positions.len()).expect("packed adjacency positions fit u32"),
            )
        } else if positions
            .iter()
            .all(|&position| u16::try_from(position).is_ok())
        {
            Self::U16(
                positions
                    .into_iter()
                    .map(|position| {
                        u16::try_from(position)
                            .expect("all closed positions were checked to fit u16")
                    })
                    .collect(),
            )
        } else {
            Self::U32(positions)
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Identity(len) => {
                usize::try_from(*len).expect("u32 packed-row count fits the host usize")
            }
            Self::U16(positions) => positions.len(),
            Self::U32(positions) => positions.len(),
        }
    }

    fn get(&self, index: usize) -> Option<u32> {
        match self {
            Self::Empty => None,
            Self::Identity(len) => {
                let index = u32::try_from(index).ok()?;
                (index < *len).then_some(index)
            }
            Self::U16(positions) => positions.get(index).copied().map(u32::from),
            Self::U32(positions) => positions.get(index).copied(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len()).filter_map(|index| self.get(index))
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Empty | Self::Identity(_) => 0,
            Self::U16(positions) => positions.len() * std::mem::size_of::<u16>(),
            Self::U32(positions) => positions.len() * std::mem::size_of::<u32>(),
        }
    }
}

impl RelTable {
    /// Creates a new relationship table.
    ///
    /// # Panics
    ///
    /// Panics if `bwd` is a non-empty CSR that has no edge data populated.
    #[must_use]
    pub fn new(
        schema: EdgeSchema,
        fwd: CsrAdjacency,
        bwd: Option<CsrAdjacency>,
        properties: FxHashMap<PropertyKey, TemporalColumn>,
        src_table_id: u16,
        dst_table_id: u16,
    ) -> Self {
        if let Some(ref b) = bwd {
            assert!(
                b.has_edge_data() || b.num_edges() == 0,
                "backward CSR must have edge_data populated"
            );
        }
        Self {
            schema,
            fwd,
            bwd,
            properties,
            src_table_id,
            dst_table_id,
            validity: Vec::new(),
            packed_fwd: None,
            packed_bwd: None,
            open_from: OpenEpochs::Empty,
            open_edge_ids: CompactEdgeIds::Empty,
            closed_id_ord: ClosedIdOrder::Empty,
        }
    }

    /// Returns the edge type name (e.g. `"KNOWS"`).
    #[must_use]
    pub fn edge_type(&self) -> &ArcStr {
        &self.schema.edge_type
    }

    /// Returns the relationship table ID (encoded into [`EdgeId`] values).
    #[must_use]
    pub fn rel_table_id(&self) -> u16 {
        self.schema.rel_table_id
    }

    /// Returns the table ID of the source node table.
    #[must_use]
    pub fn src_table_id(&self) -> u16 {
        self.src_table_id
    }

    /// Returns the table ID of the destination node table.
    #[must_use]
    pub fn dst_table_id(&self) -> u16 {
        self.dst_table_id
    }

    /// Returns the total number of edges in this table.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        if self.fwd.num_edges() > 0 {
            return self.fwd.num_edges();
        }
        if let Some(packed) = &self.packed_fwd
            && self.properties.is_empty()
        {
            return packed.num_open();
        }
        self.fwd.num_edges()
    }

    /// Returns `true` if a backward CSR is available.
    #[must_use]
    pub fn has_backward(&self) -> bool {
        self.bwd.as_ref().is_some_and(|b| b.num_edges() > 0) || self.packed_bwd.is_some()
    }

    /// Current 1-hop is the packed open prefix (`fwd` empty).
    ///
    /// After the no-property slim, current lives on [`Self::fwd`] and this
    /// is false. Property-bearing tables keep `fwd` aligned with columns.
    #[inline]
    #[must_use]
    pub fn current_from_packed(&self) -> bool {
        self.packed_fwd.is_some() && self.properties.is_empty() && self.fwd.num_edges() == 0
    }

    /// Current 1-hop already carries original [`EdgeId`]s (do not remap).
    #[inline]
    #[must_use]
    pub fn current_uses_original_ids(&self) -> bool {
        !self.open_edge_ids.is_empty() || self.current_from_packed()
    }

    /// Original ids of current (`fwd`) rows after the no-property slim.
    #[must_use]
    pub fn open_edge_ids(&self) -> Vec<EdgeId> {
        self.open_edge_ids.to_vec()
    }

    /// Whether current rows have no original-id sidecar (compact CSR ids).
    #[must_use]
    pub fn open_edge_ids_empty(&self) -> bool {
        self.open_edge_ids.is_empty()
    }

    /// Original [`EdgeId`] of the current-CSR row at `pos`, if recorded.
    #[must_use]
    pub(crate) fn open_edge_id_at(&self, pos: usize) -> Option<EdgeId> {
        self.open_edge_ids.get(pos)
    }

    /// Returns all **current** edges originating from the given source node.
    ///
    /// Walks the packed open prefix when [`Self::current_from_packed`], else
    /// the derived current `fwd` CSR. Never scans closed tails.
    ///
    /// Packed rows carry original [`EdgeId`]s. The derived-CSR path encodes
    /// compact `(rel_table, csr_position)` ids (caller translates).
    #[must_use]
    pub fn edges_from_source(&self, src_offset: u32) -> Vec<(NodeId, EdgeId)> {
        if !self.open_edge_ids.is_empty() && self.fwd.num_edges() > 0 {
            let start = self.fwd.offset_of(src_offset) as usize;
            return self
                .fwd
                .neighbors(src_offset)
                .iter()
                .enumerate()
                .filter_map(|(i, &dst)| {
                    let eid = self.open_edge_ids.get(start + i)?;
                    Some((encode_node_id(self.dst_table_id, u64::from(dst)), eid))
                })
                .collect();
        }
        if let Some(packed) = &self.packed_fwd
            && self.properties.is_empty()
            && self.fwd.num_edges() == 0
        {
            return packed
                .current_edges(src_offset)
                .into_iter()
                .map(|(dst, eid)| (encode_node_id(self.dst_table_id, u64::from(dst)), eid))
                .collect();
        }
        let neighbors = self.fwd.neighbors(src_offset);
        let start_pos = u64::from(self.fwd.offset_of(src_offset));
        let rel_id = self.schema.rel_table_id;

        neighbors
            .iter()
            .enumerate()
            .map(|(i, &target_offset)| {
                let node_id = encode_node_id(self.dst_table_id, u64::from(target_offset));
                let edge_id = encode_edge_id(rel_id, start_pos + i as u64);
                (node_id, edge_id)
            })
            .collect()
    }

    /// Returns all edges pointing to the given target node.
    ///
    /// Returns `None` if no backward CSR is available. Each result is a
    /// `(source_NodeId, EdgeId)` pair. The `EdgeId` is derived from the
    /// *forward* CSR position for stability.
    #[must_use]
    pub fn edges_to_target(&self, dst_offset: u32) -> Option<Vec<(NodeId, EdgeId)>> {
        if !self.open_edge_ids.is_empty()
            && let Some(bwd) = &self.bwd
        {
            let bwd_start = bwd.offset_of(dst_offset) as usize;
            let results = bwd
                .neighbors(dst_offset)
                .iter()
                .enumerate()
                .filter_map(|(i, &src_offset)| {
                    let fwd_pos = bwd.edge_data_at(bwd_start + i)? as usize;
                    let eid = self.open_edge_ids.get(fwd_pos)?;
                    Some((
                        encode_node_id(self.src_table_id, u64::from(src_offset)),
                        eid,
                    ))
                })
                .collect();
            return Some(results);
        }
        if self.current_from_packed()
            && let Some(packed) = &self.packed_fwd
            && let Some(bwd) = &self.bwd
        {
            let bwd_start = bwd.offset_of(dst_offset) as usize;
            let results = bwd
                .neighbors(dst_offset)
                .iter()
                .enumerate()
                .filter_map(|(i, &src_offset)| {
                    let fat = bwd.edge_data_at(bwd_start + i)?;
                    let eid = packed.edge_ids().get(fat as usize).copied()?;
                    Some((
                        encode_node_id(self.src_table_id, u64::from(src_offset)),
                        eid,
                    ))
                })
                .collect();
            return Some(results);
        }
        if let Some(packed) = &self.packed_bwd
            && self.properties.is_empty()
        {
            return Some(
                packed
                    .current_edges(dst_offset)
                    .into_iter()
                    .map(|(src, eid)| (encode_node_id(self.src_table_id, u64::from(src)), eid))
                    .collect(),
            );
        }
        let bwd = self.bwd.as_ref()?;
        let bwd_start = bwd.offset_of(dst_offset) as usize;
        let source_offsets = bwd.neighbors(dst_offset);
        let rel_id = self.schema.rel_table_id;

        let results = source_offsets
            .iter()
            .enumerate()
            .filter_map(|(i, &src_offset)| {
                // O(1) lookup via edge_data stored on the backward CSR.
                // Returns None if edge_data was not populated on backward CSR.
                let fwd_pos = bwd.edge_data_at(bwd_start + i)?;
                let node_id = encode_node_id(self.src_table_id, u64::from(src_offset));
                let edge_id = encode_edge_id(rel_id, u64::from(fwd_pos));
                Some((node_id, edge_id))
            })
            .collect();

        Some(results)
    }

    /// Returns the current property value for a specific edge (by CSR position) and key.
    ///
    /// Routes through the current-value projection (all-open identity: CSR
    /// position == physical row). Closed-interval rows are still readable here
    /// via the value codec; use [`Self::get_property_at_epoch`] for as-of gating.
    #[must_use]
    pub fn get_edge_property(&self, csr_position: usize, key: &PropertyKey) -> Option<Value> {
        self.properties.get(key)?.values().get(csr_position)
    }

    /// Returns the edge property value valid at `epoch` (as-of read).
    ///
    /// Identity layout: `edge_offset` is the forward CSR position / physical
    /// row. Structural validity (when populated) gates the row first, then
    /// the property column's interval. `epoch == PENDING` is the current view.
    #[must_use]
    pub fn get_property_at_epoch(
        &self,
        edge_offset: usize,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        if !self.row_contains(edge_offset, epoch) {
            return None;
        }
        self.properties.get(key)?.value_as_of(edge_offset, epoch)
    }

    /// Structural interval of the CSR row, or all-open when `validity` is empty.
    ///
    /// Packed-current tables (`fwd` dropped) treat `csr_position` as a fat-run
    /// index. Property-bearing tables keep `fwd` and use the packed open
    /// prefix (derived-CSR order) so create epochs survive clearing the
    /// duplicate `validity` vec.
    #[must_use]
    pub fn interval_at(&self, csr_position: usize) -> EpochInterval {
        if let Some(from) = self.open_from.get(csr_position) {
            return EpochInterval::open(super::csr::unpack_epoch(from));
        }
        if let Some(packed) = &self.packed_fwd {
            if self.fwd.num_edges() == 0 {
                return packed
                    .interval(csr_position)
                    .unwrap_or_else(|| EpochInterval::open(EpochId::INITIAL));
            }
            if let Some(iv) = packed.open_interval_at(csr_position) {
                return iv;
            }
        }
        self.validity
            .get(csr_position)
            .copied()
            .unwrap_or_else(|| EpochInterval::open(EpochId::INITIAL))
    }

    /// Whether CSR row `csr_position` is structurally valid at `epoch`.
    ///
    /// Empty `validity` is the fresh-base all-open identity. `PENDING` matches
    /// still-open intervals (current view). With packed adjacency, consults
    /// fat-run or open-prefix validity instead of the dropped duplicate vec.
    #[must_use]
    pub fn row_contains(&self, csr_position: usize, epoch: EpochId) -> bool {
        if let Some(from) = self.open_from.get(csr_position) {
            let iv = EpochInterval::open(super::csr::unpack_epoch(from));
            return if epoch == EpochId::PENDING {
                iv.is_open()
            } else {
                iv.contains(epoch)
            };
        }
        if let Some(packed) = &self.packed_fwd {
            let iv = if self.fwd.num_edges() == 0 {
                packed.interval(csr_position)
            } else {
                packed.open_interval_at(csr_position)
            };
            let Some(iv) = iv else {
                return false;
            };
            return if epoch == EpochId::PENDING {
                iv.is_open()
            } else {
                iv.contains(epoch)
            };
        }
        if self.validity.is_empty() {
            return csr_position < self.fwd.num_edges();
        }
        let Some(iv) = self.validity.get(csr_position) else {
            return false;
        };
        if epoch == EpochId::PENDING {
            iv.is_open()
        } else {
            iv.contains(epoch)
        }
    }

    /// Per-row structural validity (empty = all-open identity).
    #[must_use]
    pub fn validity(&self) -> &[EpochInterval] {
        &self.validity
    }

    /// True when packed fat adjacency is installed (v5 explicit rel addendum).
    #[must_use]
    pub fn is_explicit_temporal(&self) -> bool {
        self.packed_fwd.is_some()
    }

    /// Restores packed adjacency after v5 deserialize. Does not replace `fwd`.
    pub(crate) fn restore_temporal_addendum(
        &mut self,
        packed_fwd: PackedOpenAdjacency,
        packed_bwd: Option<PackedOpenAdjacency>,
        validity: Vec<EpochInterval>,
    ) {
        let _ = validity;
        self.validity.clear();
        if self.properties.is_empty() {
            self.slim_no_prop_packed(packed_fwd, None, packed_bwd);
            return;
        }
        self.set_packed_fwd(packed_fwd);
        self.packed_bwd = packed_bwd;
    }

    /// Installs in-memory structural validity after a temporal fold.
    ///
    /// `intervals` must be empty or one per forward CSR position. Reconstructed
    /// from packed adjacency on v5 load.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_validity(&mut self, intervals: Vec<EpochInterval>) {
        debug_assert!(
            intervals.is_empty() || intervals.len() == self.fwd.num_edges(),
            "structural validity must be empty or parallel to fwd CSR"
        );
        self.validity = intervals;
    }

    /// Installs Option A packed adjacency and the derived current CSR.
    ///
    /// `fwd` stays persist-safe: replaced with `derived_fwd` only when the
    /// open prefix matches the existing current snapshot (same offsets/targets)
    /// or this table has no property columns (no remap needed). v5 also
    /// writes the packed arrays in the rel addendum.
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn install_packed_open(
        &mut self,
        packed_fwd: PackedOpenAdjacency,
        packed_bwd: Option<PackedOpenAdjacency>,
        derived_fwd: CsrAdjacency,
    ) {
        // Packed already holds per-version intervals. Copying the open
        // prefix here is 16 B × open edges with no read-path consumer.
        self.validity.clear();
        if self.properties.is_empty() {
            self.slim_no_prop_packed(packed_fwd, Some(derived_fwd), packed_bwd);
            return;
        }
        let same_current = self.fwd.targets() == derived_fwd.targets()
            && self.fwd.offsets() == derived_fwd.offsets();
        if same_current {
            let _ = derived_fwd;
        } else {
            debug_assert!(
                same_current,
                "derived current CSR diverged from snapshot RelTable.fwd; \
                 keeping persist-safe current CSR (property columns stay aligned)"
            );
            let _ = derived_fwd;
        }
        self.set_packed_fwd(packed_fwd);
        self.packed_bwd = packed_bwd;
    }

    /// Current 1-hop on a tight CSR; packed keeps closed tails only.
    fn slim_no_prop_packed(
        &mut self,
        packed_fwd: PackedOpenAdjacency,
        derived_fwd: Option<CsrAdjacency>,
        packed_bwd: Option<PackedOpenAdjacency>,
    ) {
        let open_from = OpenEpochs::from_epochs(
            packed_fwd
                .open_from_epochs()
                .into_iter()
                .map(super::csr::pack_epoch)
                .collect(),
        );
        let open_edge_ids = CompactEdgeIds::from_ids(packed_fwd.open_edge_ids());
        let derived = derived_fwd.unwrap_or_else(|| packed_fwd.derive_current_csr());
        // Backward CSR is dest-table indexed. Source→Entity (provenance
        // OBSERVED_BY) has |src| << |dst|; the forward source count OOBs
        // as soon as dest_offset >= |src|.
        let max_edge_dst = derived
            .targets()
            .iter()
            .map(|&d| d as usize + 1)
            .max()
            .unwrap_or(0);
        let num_dst = self
            .bwd
            .as_ref()
            .map_or(0, CsrAdjacency::num_nodes)
            .max(
                packed_bwd
                    .as_ref()
                    .map_or(0, PackedOpenAdjacency::num_nodes),
            )
            .max(max_edge_dst);
        let bwd = backward_from_fwd(&derived, num_dst);
        self.fwd = derived;
        self.bwd = Some(bwd);
        self.open_from = open_from;
        self.open_edge_ids = open_edge_ids;
        self.set_packed_fwd(packed_fwd.into_closed_only());
        self.packed_bwd = None;
        self.validity.clear();
    }

    fn set_packed_fwd(&mut self, packed: PackedOpenAdjacency) {
        self.packed_fwd = Some(packed);
        self.rebuild_closed_id_ord();
    }

    fn rebuild_closed_id_ord(&mut self) {
        self.closed_id_ord = ClosedIdOrder::Empty;
        let Some(packed) = &self.packed_fwd else {
            return;
        };
        let mut positions = Vec::new();
        for (pos, _) in packed.edge_ids().iter().enumerate() {
            if packed.interval(pos).is_some_and(|iv| !iv.is_open()) {
                let Ok(p) = u32::try_from(pos) else {
                    continue;
                };
                positions.push(p);
            }
        }
        let ids = packed.edge_ids();
        positions.sort_unstable_by_key(|&p| ids[p as usize]);
        self.closed_id_ord = ClosedIdOrder::from_positions(positions);
    }

    /// Fat position of a structure-only closed packed row, if `id` is one.
    #[must_use]
    pub(crate) fn packed_closed_pos(&self, id: EdgeId) -> Option<u32> {
        let packed = self.packed_fwd.as_ref()?;
        let ids = packed.edge_ids();
        let mut low = 0usize;
        let mut high = self.closed_id_ord.len();
        while low < high {
            let mid = low + (high - low) / 2;
            let position = self.closed_id_ord.get(mid)?;
            match ids[position as usize].cmp(&id) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(position),
            }
        }
        None
    }

    /// Closed packed fat positions (sorted by original [`EdgeId`]).
    pub(crate) fn closed_id_ord(&self) -> impl Iterator<Item = u32> + '_ {
        self.closed_id_ord.iter()
    }

    /// Reconstructs Option A (open prefix + closed tails) for v5 persist.
    #[must_use]
    pub(crate) fn packed_for_persist(&self) -> Option<PackedOpenAdjacency> {
        let packed = self.packed_fwd.as_ref()?;
        if packed.num_open() > 0 || self.fwd.num_edges() == 0 {
            return Some(packed.clone());
        }
        let n = self.fwd.num_nodes().max(packed.num_nodes());
        let mut rows = Vec::with_capacity(self.fwd.num_edges() + packed.num_versions());
        for src in 0..self.fwd.num_nodes() {
            let src_u = u32::try_from(src).unwrap_or(u32::MAX);
            let start = self.fwd.offset_of(src_u) as usize;
            for (i, &dst) in self.fwd.neighbors(src_u).iter().enumerate() {
                let from = self
                    .open_from
                    .get(start + i)
                    .map_or(EpochId::INITIAL, super::csr::unpack_epoch);
                let eid = self.open_edge_ids.get(start + i).unwrap_or_else(|| {
                    encode_edge_id(self.schema.rel_table_id, (start + i) as u64)
                });
                rows.push(super::csr::TemporalEdgeRow {
                    src: src_u,
                    dst,
                    validity: EpochInterval::open(from),
                    edge_id: eid,
                });
            }
        }
        for i in 0..packed.num_nodes() {
            let src_u = u32::try_from(i).unwrap_or(u32::MAX);
            let start = packed.offsets()[i] as usize;
            let end = packed.offsets()[i + 1] as usize;
            for pos in start..end {
                rows.push(super::csr::TemporalEdgeRow {
                    src: src_u,
                    dst: packed.targets()[pos],
                    validity: packed
                        .interval(pos)
                        .unwrap_or_else(|| EpochInterval::open(EpochId::INITIAL)),
                    edge_id: packed.edge_ids()[pos],
                });
            }
        }
        Some(PackedOpenAdjacency::from_rows(n, &rows))
    }

    /// Packed fat forward adjacency (open prefix + closed tails), if installed.
    #[must_use]
    pub fn packed_fwd(&self) -> Option<&PackedOpenAdjacency> {
        self.packed_fwd.as_ref()
    }

    /// Packed fat backward adjacency, if installed.
    #[must_use]
    pub fn packed_bwd(&self) -> Option<&PackedOpenAdjacency> {
        self.packed_bwd.as_ref()
    }

    /// Fat-run index of `id` (hint is tried first; else closed-id binary search).
    pub(crate) fn fat_pos_for_edge(&self, id: EdgeId, hint: u64) -> Option<usize> {
        let packed = self.packed_fwd.as_ref()?;
        if let Ok(h) = usize::try_from(hint)
            && packed.edge_ids().get(h) == Some(&id)
        {
            return Some(h);
        }
        self.packed_closed_pos(id)
            .map(|p| p as usize)
            .or_else(|| packed.edge_ids().iter().position(|&e| e == id))
    }

    /// Current 1-hop targets: derived `fwd` CSR (open prefix).
    #[inline]
    #[must_use]
    pub fn current_targets(&self, src_offset: u32) -> &[u32] {
        if self.fwd.num_edges() > 0 {
            return self.fwd.neighbors(src_offset);
        }
        if let Some(packed) = &self.packed_fwd {
            return packed.current_neighbors(src_offset);
        }
        self.fwd.neighbors(src_offset)
    }

    /// Appends neighbors visible at `epoch` onto `out` (does not clear).
    pub fn extend_neighbors_at_epoch(&self, src_offset: u32, epoch: EpochId, out: &mut Vec<u32>) {
        if epoch == EpochId::PENDING {
            out.extend_from_slice(self.current_targets(src_offset));
            return;
        }
        // Full packed tables already carry their open rows. Only an unpacked
        // table or the slim representation's explicit open sidecar uses CSR.
        if self.fwd.num_edges() > 0 && (self.packed_fwd.is_none() || !self.open_from.is_empty()) {
            let start = self.fwd.offset_of(src_offset) as usize;
            for (i, &dst) in self.fwd.neighbors(src_offset).iter().enumerate() {
                if self.row_contains(start + i, epoch) {
                    out.push(dst);
                }
            }
        }
        if let Some(packed) = &self.packed_fwd {
            packed.extend_neighbors_at_epoch(src_offset, epoch, out);
        }
    }

    /// Neighbors visible at `epoch`. `PENDING` is the derived current CSR
    /// (open prefix); other epochs scan packed validity.
    #[must_use]
    pub fn neighbors_at_epoch(&self, src_offset: u32, epoch: EpochId) -> Vec<u32> {
        let mut out = Vec::new();
        self.extend_neighbors_at_epoch(src_offset, epoch, &mut out);
        out
    }

    /// Appends `(target, edge_id)` pairs visible at `epoch` (does not clear).
    ///
    /// Packed rows carry original [`EdgeId`]s. The all-open / `PENDING` path
    /// encodes compact `(rel_table, csr_position)` ids (caller translates).
    pub fn extend_edges_from_at_epoch(
        &self,
        src_offset: u32,
        epoch: EpochId,
        out: &mut Vec<(NodeId, EdgeId)>,
    ) {
        if epoch == EpochId::PENDING {
            out.extend(self.edges_from_source(src_offset));
            return;
        }
        // Do not mix compact-position IDs from the derived CSR with original
        // IDs from full packed rows. Slim tables keep original IDs alongside
        // their CSR open rows and only closed history in packed storage.
        if self.fwd.num_edges() > 0 && (self.packed_fwd.is_none() || !self.open_from.is_empty()) {
            let start = self.fwd.offset_of(src_offset) as usize;
            let rel_id = self.schema.rel_table_id;
            for (i, &dst) in self.fwd.neighbors(src_offset).iter().enumerate() {
                if self.row_contains(start + i, epoch) {
                    let eid = self
                        .open_edge_ids
                        .get(start + i)
                        .unwrap_or_else(|| encode_edge_id(rel_id, (start + i) as u64));
                    out.push((encode_node_id(self.dst_table_id, u64::from(dst)), eid));
                }
            }
        }
        if let Some(packed) = &self.packed_fwd {
            let mut scratch = Vec::new();
            packed.extend_edges_at_epoch(src_offset, epoch, &mut scratch);
            out.reserve(scratch.len());
            for (dst_off, eid) in scratch {
                out.push((encode_node_id(self.dst_table_id, u64::from(dst_off)), eid));
            }
        }
    }

    /// `(target, edge_id)` pairs visible at `epoch`.
    #[must_use]
    pub fn edges_from_at_epoch(&self, src_offset: u32, epoch: EpochId) -> Vec<(NodeId, EdgeId)> {
        let mut out = Vec::new();
        self.extend_edges_from_at_epoch(src_offset, epoch, &mut out);
        out
    }

    /// Incoming `(source, edge_id)` pairs visible at `epoch`.
    ///
    /// `None` when no backward packed / current CSR is available.
    #[must_use]
    pub fn incoming_edges_at_epoch(
        &self,
        dst_offset: u32,
        epoch: EpochId,
    ) -> Option<Vec<(NodeId, EdgeId)>> {
        if epoch == EpochId::PENDING {
            return self.edges_to_target(dst_offset);
        }
        if let Some(packed) = &self.packed_bwd {
            return Some(
                packed
                    .edges_at_epoch(dst_offset, epoch)
                    .into_iter()
                    .map(|(src_off, eid)| {
                        (encode_node_id(self.src_table_id, u64::from(src_off)), eid)
                    })
                    .collect(),
            );
        }
        let mut out = Vec::new();
        // Unpacked CSR rows are all-open without a sidecar. Packed open rows
        // must come from exactly one representation, not both CSR and packed.
        if let Some(bwd) = &self.bwd
            && (self.packed_fwd.is_none() || !self.open_from.is_empty())
        {
            let bwd_start = bwd.offset_of(dst_offset) as usize;
            for (i, &src_off) in bwd.neighbors(dst_offset).iter().enumerate() {
                if let Some(fwd_pos) = bwd.edge_data_at(bwd_start + i) {
                    let pos = fwd_pos as usize;
                    if self.row_contains(pos, epoch) {
                        let eid = self.open_edge_ids.get(pos).unwrap_or_else(|| {
                            encode_edge_id(self.schema.rel_table_id, u64::from(fwd_pos))
                        });
                        out.push((encode_node_id(self.src_table_id, u64::from(src_off)), eid));
                    }
                }
            }
        }
        if self.packed_bwd.is_none()
            && let Some(packed) = &self.packed_fwd
        {
            out.extend(packed.incoming_at_epoch(dst_offset, epoch).into_iter().map(
                |(src_off, eid)| (encode_node_id(self.src_table_id, u64::from(src_off)), eid),
            ));
        }
        if out.is_empty() && self.packed_fwd.is_none() && self.bwd.is_none() {
            return None;
        }
        Some(out)
    }

    /// Incoming neighbors visible at `epoch` (`PENDING` = packed open prefix
    /// or derived current bwd).
    #[must_use]
    pub fn incoming_at_epoch(&self, dst_offset: u32, epoch: EpochId) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        if !self.extend_incoming_at_epoch(dst_offset, epoch, &mut out) {
            return None;
        }
        Some(out)
    }

    /// Appends incoming source offsets visible at `epoch` (does not clear).
    ///
    /// Dest-only: no [`EdgeId`]s. Prefers packed-bwd / current bwd. Scans
    /// packed-fwd only when there is no reverse CSR (a "who observed H"
    /// is a single dest — still O(degree) with bwd).
    ///
    /// Returns `false` when this table has no incoming adjacency at all.
    pub fn extend_incoming_at_epoch(
        &self,
        dst_offset: u32,
        epoch: EpochId,
        out: &mut Vec<u32>,
    ) -> bool {
        if epoch == EpochId::PENDING {
            if let Some(packed) = &self.packed_bwd {
                out.extend_from_slice(packed.current_neighbors(dst_offset));
                return true;
            }
            if let Some(bwd) = &self.bwd {
                out.extend_from_slice(bwd.neighbors(dst_offset));
                return true;
            }
            return false;
        }
        if let Some(packed) = &self.packed_bwd {
            packed.extend_neighbors_at_epoch(dst_offset, epoch, out);
            return true;
        }
        // Match the pair-producing reader: an empty unpacked sidecar is
        // all-open, while packed open rows must not be appended twice.
        if let Some(bwd) = &self.bwd
            && (self.packed_fwd.is_none() || !self.open_from.is_empty())
        {
            let bwd_start = bwd.offset_of(dst_offset) as usize;
            for (i, &src_off) in bwd.neighbors(dst_offset).iter().enumerate() {
                if let Some(fwd_pos) = bwd.edge_data_at(bwd_start + i)
                    && self.row_contains(fwd_pos as usize, epoch)
                {
                    out.push(src_off);
                }
            }
        }
        if self.packed_bwd.is_none()
            && let Some(packed) = &self.packed_fwd
        {
            // Closed tails live on packed-fwd when packed-bwd was slimmed.
            for (src_off, _) in packed.incoming_at_epoch(dst_offset, epoch) {
                out.push(src_off);
            }
        }
        self.packed_fwd.is_some() || self.bwd.is_some()
    }

    /// Source/dest table-local offsets for a current-CSR or packed-fat position.
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn fwd_src_dst(&self, pos: usize) -> Option<(u32, u32)> {
        if self.current_from_packed()
            && let Some(packed) = &self.packed_fwd
        {
            let src = packed.src_of(pos)?;
            let dst = *packed.targets().get(pos)?;
            return Some((src, dst));
        }
        let pos_u = u32::try_from(pos).ok()?;
        let src = self.fwd.source_for_position(pos_u)?;
        let start = self.fwd.offset_of(src) as usize;
        let local = pos.checked_sub(start)?;
        let dst = *self.fwd.neighbors(src).get(local)?;
        Some((src, dst))
    }

    /// Returns all current properties for the edge at the given forward CSR position.
    ///
    /// Same current-value / closed-row semantics as [`Self::get_edge_property`].
    #[must_use]
    pub fn get_all_edge_properties(&self, csr_position: usize) -> FxHashMap<PropertyKey, Value> {
        let mut props = FxHashMap::default();
        for (key, col) in &self.properties {
            if let Some(value) = col.values().get(csr_position) {
                props.insert(key.clone(), value);
            }
        }
        props
    }

    /// Returns all property keys present in this relationship table.
    #[must_use]
    pub fn property_keys(&self) -> Vec<PropertyKey> {
        self.properties.keys().cloned().collect()
    }

    /// Returns the source [`NodeId`] for the edge at the given forward CSR position.
    #[must_use]
    pub fn source_node_id(&self, csr_position: u32) -> Option<NodeId> {
        if self.current_from_packed()
            && let Some(packed) = &self.packed_fwd
        {
            let src_offset = packed.src_of(csr_position as usize)?;
            return Some(encode_node_id(self.src_table_id, u64::from(src_offset)));
        }
        let src_offset = self.fwd.source_for_position(csr_position)?;
        Some(encode_node_id(self.src_table_id, u64::from(src_offset)))
    }

    /// Returns the destination [`NodeId`] for the edge at the given forward CSR position.
    #[must_use]
    pub fn dest_node_id(&self, csr_position: u32) -> Option<NodeId> {
        if self.current_from_packed()
            && let Some(packed) = &self.packed_fwd
        {
            let target_offset = *packed.targets().get(csr_position as usize)?;
            return Some(encode_node_id(self.dst_table_id, u64::from(target_offset)));
        }
        let src = self.fwd.source_for_position(csr_position)?;
        let start = self.fwd.offset_of(src);
        let local_idx = (csr_position - start) as usize;
        let target_offset = *self.fwd.neighbors(src).get(local_idx)?;
        Some(encode_node_id(self.dst_table_id, u64::from(target_offset)))
    }

    /// Returns the out-degree of a source node.
    #[must_use]
    pub fn out_degree(&self, src_offset: u32) -> usize {
        if self.fwd.num_edges() > 0 {
            return self.fwd.degree(src_offset);
        }
        if let Some(packed) = &self.packed_fwd {
            return packed.current_neighbors(src_offset).len();
        }
        self.fwd.degree(src_offset)
    }

    /// Returns the in-degree of a target node, or `None` if no backward CSR.
    #[must_use]
    pub fn in_degree(&self, dst_offset: u32) -> Option<usize> {
        if let Some(packed) = &self.packed_bwd {
            return Some(packed.current_neighbors(dst_offset).len());
        }
        self.bwd.as_ref().map(|b| b.degree(dst_offset))
    }

    /// Returns the forward CSR (for serialization).
    #[must_use]
    pub fn fwd(&self) -> &CsrAdjacency {
        &self.fwd
    }

    /// Returns the backward CSR, if present (for serialization).
    #[must_use]
    pub fn bwd(&self) -> Option<&CsrAdjacency> {
        self.bwd.as_ref()
    }

    /// Returns edge property columns (for serialization).
    ///
    /// Persist writes [`TemporalColumn::values`] (current projection). Packed
    /// adjacency and closed-edge property history live in the v5 addendum.
    #[must_use]
    pub fn properties(&self) -> &FxHashMap<PropertyKey, TemporalColumn> {
        &self.properties
    }

    /// Returns an estimate of heap memory used by the CSR structures and
    /// edge property columns in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let fwd_bytes = self.fwd.memory_bytes();
        let bwd_bytes = self.bwd.as_ref().map_or(0, |b| b.memory_bytes());
        let prop_bytes: usize = self.properties.values().map(|c| c.heap_bytes()).sum();
        let validity_bytes = self.validity.len() * std::mem::size_of::<EpochInterval>();
        let open_from_bytes = self.open_from.heap_bytes();
        let open_id_bytes = self.open_edge_ids.heap_bytes();
        let closed_ord_bytes = self.closed_id_ord.heap_bytes();
        let packed_bytes = self
            .packed_fwd
            .as_ref()
            .map_or(0, PackedOpenAdjacency::memory_bytes)
            + self
                .packed_bwd
                .as_ref()
                .map_or(0, PackedOpenAdjacency::memory_bytes);
        fwd_bytes
            + bwd_bytes
            + prop_bytes
            + validity_bytes
            + open_from_bytes
            + open_id_bytes
            + closed_ord_bytes
            + packed_bytes
    }

    /// Heap bytes of `(open epochs, open ids, closed-id order)` used by the
    /// temporal density regression's component diagnostic.
    #[cfg(all(test, feature = "lpg"))]
    pub(crate) fn temporal_sidecar_memory_bytes(&self) -> (usize, usize, usize) {
        (
            self.open_from.heap_bytes(),
            self.open_edge_ids.heap_bytes(),
            self.closed_id_ord.heap_bytes(),
        )
    }
}

/// Rebuilds a backward CSR (with `edge_data` → forward positions) from `fwd`.
pub(super) fn backward_from_fwd(fwd: &CsrAdjacency, dst_count: usize) -> CsrAdjacency {
    let mut bwd_edges: Vec<(u32, u32, u32)> = Vec::new();
    for src in 0..fwd.num_nodes() {
        // reason: source index is a CSR node offset
        #[allow(clippy::cast_possible_truncation)]
        let src_u = src as u32;
        for (local, &dst) in fwd.neighbors(src_u).iter().enumerate() {
            // reason: CSR position fits the existing u32 edge-data space
            #[allow(clippy::cast_possible_truncation)]
            let fwd_pos = fwd.offset_of(src_u) + local as u32;
            bwd_edges.push((dst, src_u, fwd_pos));
        }
    }
    bwd_edges.sort_by_key(|&(dst, src, _)| (dst, src));
    let max_dst = bwd_edges
        .iter()
        .map(|&(d, _, _)| d as usize + 1)
        .max()
        .unwrap_or(0);
    let bwd_pairs: Vec<(u32, u32)> = bwd_edges.iter().map(|&(dst, src, _)| (dst, src)).collect();
    let mut bwd = CsrAdjacency::from_sorted_edges(dst_count.max(max_dst), &bwd_pairs);
    let mapping = bwd_edges.into_iter().map(|(_, _, pos)| pos).collect();
    bwd.set_edge_data(mapping);
    bwd
}

#[cfg(test)]
mod tests {
    use super::super::column::ColumnCodec;
    use super::super::id::decode_node_id;
    use super::*;

    #[test]
    fn uniform_open_epochs_and_identity_closed_order_have_no_heap_sidecar() {
        let epochs = OpenEpochs::from_epochs(vec![10; 4]);
        assert_eq!(epochs.get(0), Some(10));
        assert_eq!(epochs.get(3), Some(10));
        assert_eq!(epochs.get(4), None);
        assert_eq!(epochs.heap_bytes(), 0);

        let alternating = OpenEpochs::from_epochs(vec![10, 20, 10, 20, 10, 20, 10, 20]);
        assert_eq!(
            (0..8)
                .map(|index| alternating.get(index).expect("encoded epoch"))
                .collect::<Vec<_>>(),
            vec![10, 20, 10, 20, 10, 20, 10, 20]
        );
        assert_eq!(alternating.get(8), None);
        assert!(alternating.heap_bytes() < 8 * std::mem::size_of::<u32>());

        let identity = ClosedIdOrder::from_positions(vec![0, 1, 2, 3]);
        assert_eq!(identity.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(identity.heap_bytes(), 0);

        let narrow = ClosedIdOrder::from_positions(vec![4, 1, 3]);
        assert_eq!(narrow.iter().collect::<Vec<_>>(), vec![4, 1, 3]);
        assert_eq!(narrow.heap_bytes(), 3 * std::mem::size_of::<u16>());
    }

    /// Helper to create a simple test scenario.
    ///
    /// 3 source nodes (table 0), 2 target nodes (table 1), 5 edges:
    ///   src0 -> dst0, src0 -> dst1
    ///   src1 -> dst0, src1 -> dst1
    ///   src2 -> dst0
    fn make_test_rel_table() -> RelTable {
        // Forward CSR: 3 source nodes.
        let fwd_edges = vec![(0u32, 0u32), (0, 1), (1, 0), (1, 1), (2, 0)];
        let fwd = CsrAdjacency::from_sorted_edges(3, &fwd_edges);

        // Backward CSR: 2 target nodes.
        // dst0 <- src0, src1, src2
        // dst1 <- src0, src1
        let bwd_edges = vec![(0u32, 0u32), (0, 1), (0, 2), (1, 0), (1, 1)];
        let mut bwd = CsrAdjacency::from_sorted_edges(2, &bwd_edges);

        // Pre-compute bwd-to-fwd position mapping and store as edge_data.
        let mut mapping = Vec::with_capacity(bwd_edges.len());
        for &(dst, src) in &bwd_edges {
            let fwd_start = fwd.offset_of(src);
            let local_idx = fwd.neighbors(src).iter().position(|&t| t == dst).unwrap();
            // CSR uses u32 offsets, so position within a neighbor list fits u32
            // reason: CSR uses u32 offsets, neighbor position fits u32
            #[allow(clippy::cast_possible_truncation)]
            mapping.push(fwd_start + local_idx as u32);
        }
        bwd.set_edge_data(mapping);

        let schema = EdgeSchema::new("LIKES", 5, "Person", "Movie", vec![]);

        RelTable::new(
            schema,
            fwd,
            Some(bwd),
            FxHashMap::default(),
            0, // src_table_id
            1, // dst_table_id
        )
    }

    #[test]
    fn backward_from_fwd_maps_interleaved_parallel_rows_exactly() {
        let fwd =
            CsrAdjacency::from_sorted_edges(2, &[(0, 2), (0, 1), (0, 1), (1, 2), (1, 0), (1, 2)]);
        let bwd = backward_from_fwd(&fwd, 3);
        let mut seen = Vec::new();
        for dst in 0..3u32 {
            let start = bwd.offset_of(dst) as usize;
            for (i, &src) in bwd.neighbors(dst).iter().enumerate() {
                let pos = bwd.edge_data_at(start + i).expect("backward mapping");
                assert_eq!(fwd.source_for_position(pos), Some(src));
                assert_eq!(fwd.neighbors(src)[(pos - fwd.offset_of(src)) as usize], dst);
                seen.push(pos);
            }
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..u32::try_from(fwd.num_edges()).unwrap()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_forward_traversal() {
        let rt = make_test_rel_table();

        assert_eq!(rt.edge_type().as_str(), "LIKES");
        assert_eq!(rt.rel_table_id(), 5);
        assert_eq!(rt.num_edges(), 5);
        assert!(rt.has_backward());

        // Source 0 -> targets [0, 1]
        let edges_0 = rt.edges_from_source(0);
        assert_eq!(edges_0.len(), 2);
        let (node_id_0, _edge_id_0) = edges_0[0];
        let (table, offset) = decode_node_id(node_id_0);
        assert_eq!(table, 1); // dst_table_id
        assert_eq!(offset, 0); // target offset 0

        let (node_id_1, _edge_id_1) = edges_0[1];
        let (table, offset) = decode_node_id(node_id_1);
        assert_eq!(table, 1);
        assert_eq!(offset, 1);

        // Source 1 -> targets [0, 1]
        let edges_1 = rt.edges_from_source(1);
        assert_eq!(edges_1.len(), 2);

        // Source 2 -> targets [0]
        let edges_2 = rt.edges_from_source(2);
        assert_eq!(edges_2.len(), 1);
        let (nid, _) = edges_2[0];
        let (table, offset) = decode_node_id(nid);
        assert_eq!(table, 1);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_backward_traversal() {
        let rt = make_test_rel_table();

        // Target 0 <- sources [0, 1, 2]
        let edges_to_0 = rt.edges_to_target(0).expect("backward CSR is present");
        assert_eq!(edges_to_0.len(), 3);
        let source_offsets: Vec<u64> = edges_to_0
            .iter()
            .map(|(nid, _)| decode_node_id(*nid).1)
            .collect();
        assert_eq!(source_offsets, vec![0, 1, 2]);

        // Target 1 <- sources [0, 1]
        let edges_to_1 = rt.edges_to_target(1).expect("backward CSR is present");
        assert_eq!(edges_to_1.len(), 2);
        let source_offsets: Vec<u64> = edges_to_1
            .iter()
            .map(|(nid, _)| decode_node_id(*nid).1)
            .collect();
        assert_eq!(source_offsets, vec![0, 1]);
    }

    #[test]
    fn incoming_all_open_pairs_at_finite_epochs() {
        let rt = make_test_rel_table();
        let expected = vec![
            (NodeId::new(0), encode_edge_id(5, 0)),
            (NodeId::new(1), encode_edge_id(5, 2)),
            (NodeId::new(2), encode_edge_id(5, 4)),
        ];
        // Evaluate both finite epochs before asserting, including epoch zero
        // used by an external CompactStore's transaction manager.
        let actual =
            [EpochId::new(0), EpochId::new(37)].map(|epoch| rt.incoming_edges_at_epoch(0, epoch));
        assert_eq!(actual, [Some(expected.clone()), Some(expected.clone())]);
        assert_eq!(rt.edges_to_target(0), Some(expected.clone()));
        assert_eq!(
            rt.incoming_edges_at_epoch(0, EpochId::PENDING),
            Some(expected)
        );
    }

    #[test]
    fn incoming_all_open_offsets_at_finite_epochs_append() {
        let rt = make_test_rel_table();
        let actual = [EpochId::new(0), EpochId::new(37)].map(|epoch| {
            let mut out = vec![99];
            let present = rt.extend_incoming_at_epoch(0, epoch, &mut out);
            (present, out)
        });
        assert_eq!(
            actual,
            [(true, vec![99, 0, 1, 2]), (true, vec![99, 0, 1, 2])]
        );
        assert_eq!(
            rt.incoming_at_epoch(0, EpochId::PENDING),
            Some(vec![0, 1, 2])
        );
    }

    #[test]
    fn incoming_unpacked_explicit_validity_keeps_interval_boundaries() {
        let mut rt = make_test_rel_table();
        rt.set_validity(vec![
            EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
            EpochInterval::open(EpochId::new(30)),
            EpochInterval::open(EpochId::new(10)),
            EpochInterval::closed(EpochId::new(5), EpochId::new(10)),
            EpochInterval::open(EpochId::new(20)),
        ]);
        for (epoch, expected_pairs, expected_offsets) in [
            (0, vec![], vec![99]),
            (9, vec![], vec![99]),
            (
                10,
                vec![
                    (NodeId::new(0), encode_edge_id(5, 0)),
                    (NodeId::new(1), encode_edge_id(5, 2)),
                ],
                vec![99, 0, 1],
            ),
            (
                19,
                vec![
                    (NodeId::new(0), encode_edge_id(5, 0)),
                    (NodeId::new(1), encode_edge_id(5, 2)),
                ],
                vec![99, 0, 1],
            ),
            (
                20,
                vec![
                    (NodeId::new(1), encode_edge_id(5, 2)),
                    (NodeId::new(2), encode_edge_id(5, 4)),
                ],
                vec![99, 1, 2],
            ),
            (
                21,
                vec![
                    (NodeId::new(1), encode_edge_id(5, 2)),
                    (NodeId::new(2), encode_edge_id(5, 4)),
                ],
                vec![99, 1, 2],
            ),
        ] {
            let epoch = EpochId::new(epoch);
            assert_eq!(rt.incoming_edges_at_epoch(0, epoch), Some(expected_pairs));
            let mut out = vec![99];
            assert!(rt.extend_incoming_at_epoch(0, epoch, &mut out));
            assert_eq!(out, expected_offsets);
        }
        // PENDING retains the existing current-CSR route, not the finite-epoch
        // validity filter introduced by these tests.
        assert_eq!(
            rt.incoming_edges_at_epoch(0, EpochId::PENDING),
            Some(vec![
                (NodeId::new(0), encode_edge_id(5, 0)),
                (NodeId::new(1), encode_edge_id(5, 2)),
                (NodeId::new(2), encode_edge_id(5, 4)),
            ])
        );
        assert_eq!(
            rt.incoming_at_epoch(0, EpochId::PENDING),
            Some(vec![0, 1, 2])
        );
    }

    #[test]
    fn incoming_packed_history_keeps_exact_multiplicity_with_and_without_properties() {
        use super::super::csr::{
            TemporalEdgeRow, build_current_csr_from_open_edges, pack_open_prefix,
        };

        let rows = [
            TemporalEdgeRow {
                src: 0,
                dst: 1,
                validity: EpochInterval::open(EpochId::new(10)),
                edge_id: EdgeId::new(101),
            },
            TemporalEdgeRow {
                src: 1,
                dst: 1,
                validity: EpochInterval::closed(EpochId::new(5), EpochId::new(20)),
                edge_id: EdgeId::new(202),
            },
            TemporalEdgeRow {
                src: 2,
                dst: 1,
                validity: EpochInterval::open(EpochId::new(20)),
                edge_id: EdgeId::new(303),
            },
        ];
        for with_properties in [false, true] {
            let derived = build_current_csr_from_open_edges(3, &rows);
            let bwd = backward_from_fwd(&derived, 2);
            let mut properties = FxHashMap::default();
            if with_properties {
                properties.insert(
                    PropertyKey::new("weight"),
                    TemporalColumn::all_open(ColumnCodec::raw_i64(vec![7, 9])),
                );
            }
            let schema = EdgeSchema::new("LIKES", 5, "Person", "Movie", vec![]);
            let mut rt = RelTable::new(schema, derived.clone(), Some(bwd), properties, 0, 1);
            rt.install_packed_open(pack_open_prefix(3, &rows), None, derived);
            assert!(rt.packed_bwd().is_none());
            assert_eq!(
                rt.packed_fwd().map(PackedOpenAdjacency::num_open),
                Some(if with_properties { 2 } else { 0 })
            );

            for (epoch, expected_pairs, expected_offsets) in [
                (0, vec![], vec![99]),
                (5, vec![(NodeId::new(1), EdgeId::new(202))], vec![99, 1]),
                (
                    10,
                    vec![
                        (NodeId::new(0), EdgeId::new(101)),
                        (NodeId::new(1), EdgeId::new(202)),
                    ],
                    vec![99, 0, 1],
                ),
                (
                    19,
                    vec![
                        (NodeId::new(0), EdgeId::new(101)),
                        (NodeId::new(1), EdgeId::new(202)),
                    ],
                    vec![99, 0, 1],
                ),
                (
                    20,
                    vec![
                        (NodeId::new(0), EdgeId::new(101)),
                        (NodeId::new(2), EdgeId::new(303)),
                    ],
                    vec![99, 0, 2],
                ),
                (
                    25,
                    vec![
                        (NodeId::new(0), EdgeId::new(101)),
                        (NodeId::new(2), EdgeId::new(303)),
                    ],
                    vec![99, 0, 2],
                ),
            ] {
                let epoch = EpochId::new(epoch);
                assert_eq!(rt.incoming_edges_at_epoch(1, epoch), Some(expected_pairs));
                let mut out = vec![99];
                assert!(rt.extend_incoming_at_epoch(1, epoch, &mut out));
                assert_eq!(out, expected_offsets);
            }
            let pending_pairs = if with_properties {
                vec![
                    (NodeId::new(0), encode_edge_id(5, 0)),
                    (NodeId::new(2), encode_edge_id(5, 1)),
                ]
            } else {
                vec![
                    (NodeId::new(0), EdgeId::new(101)),
                    (NodeId::new(2), EdgeId::new(303)),
                ]
            };
            assert_eq!(
                rt.incoming_edges_at_epoch(1, EpochId::PENDING),
                Some(pending_pairs)
            );
            let mut out = vec![99];
            assert!(rt.extend_incoming_at_epoch(1, EpochId::PENDING, &mut out));
            assert_eq!(out, vec![99, 0, 2]);
        }
    }

    #[test]
    fn outgoing_packed_history_keeps_parallel_identity_with_and_without_properties() {
        use super::super::csr::{
            TemporalEdgeRow, build_current_csr_from_open_edges, pack_open_prefix,
        };

        let rows = [
            TemporalEdgeRow {
                src: 0,
                dst: 1,
                validity: EpochInterval::open(EpochId::new(10)),
                edge_id: EdgeId::new(101),
            },
            TemporalEdgeRow {
                src: 0,
                dst: 1,
                validity: EpochInterval::open(EpochId::new(20)),
                edge_id: EdgeId::new(303),
            },
            TemporalEdgeRow {
                src: 0,
                dst: 2,
                validity: EpochInterval::closed(EpochId::new(5), EpochId::new(20)),
                edge_id: EdgeId::new(202),
            },
            TemporalEdgeRow {
                src: 1,
                dst: 0,
                validity: EpochInterval::open(EpochId::new(10)),
                edge_id: EdgeId::new(404),
            },
        ];
        for with_properties in [false, true] {
            let derived = build_current_csr_from_open_edges(2, &rows);
            let mut properties = FxHashMap::default();
            if with_properties {
                properties.insert(
                    PropertyKey::new("weight"),
                    TemporalColumn::all_open(ColumnCodec::raw_i64(vec![7, 9, 11])),
                );
            }
            let schema = EdgeSchema::new("LINK", 5, "Source", "Target", vec![]);
            let mut rt = RelTable::new(schema, derived.clone(), None, properties, 0, 1);
            rt.install_packed_open(pack_open_prefix(2, &rows), None, derived);
            for epoch in [0, 4, 5, 9, 10, 19, 20, 21] {
                let epoch = EpochId::new(epoch);
                for source in [0, 1, 2] {
                    let mut expected: Vec<_> = rows
                        .iter()
                        .filter(|row| row.src == source && row.validity.contains(epoch))
                        .map(|row| (encode_node_id(1, u64::from(row.dst)), row.edge_id))
                        .collect();
                    expected.sort_unstable();
                    let sentinel = (NodeId::INVALID, EdgeId::INVALID);
                    let mut actual = vec![sentinel];
                    rt.extend_edges_from_at_epoch(source, epoch, &mut actual);
                    assert_eq!(actual.remove(0), sentinel);
                    actual.sort_unstable();
                    assert_eq!(
                        actual, expected,
                        "properties={with_properties}; epoch={epoch:?}; source={source}"
                    );
                    let mut destinations = vec![99];
                    rt.extend_neighbors_at_epoch(source, epoch, &mut destinations);
                    assert_eq!(destinations.remove(0), 99);
                    destinations.sort_unstable();
                    let mut expected_destinations: Vec<_> = rows
                        .iter()
                        .filter(|row| row.src == source && row.validity.contains(epoch))
                        .map(|row| row.dst)
                        .collect();
                    expected_destinations.sort_unstable();
                    assert_eq!(destinations, expected_destinations);
                }
            }
        }
    }

    #[test]
    fn incoming_absent_and_present_empty_adjacency_stay_distinct() {
        let schema = EdgeSchema::new("LIKES", 5, "Person", "Movie", vec![]);
        let absent = RelTable::new(
            schema.clone(),
            CsrAdjacency::from_sorted_edges(3, &[(0, 1)]),
            None,
            FxHashMap::default(),
            0,
            1,
        );
        let empty = RelTable::new(
            schema,
            CsrAdjacency::from_sorted_edges(3, &[]),
            Some(CsrAdjacency::from_sorted_edges(2, &[])),
            FxHashMap::default(),
            0,
            1,
        );
        for epoch in [EpochId::new(0), EpochId::new(37), EpochId::PENDING] {
            assert_eq!(absent.incoming_edges_at_epoch(1, epoch), None);
            let mut out = vec![99];
            assert!(!absent.extend_incoming_at_epoch(1, epoch, &mut out));
            assert_eq!(out, vec![99]);
            assert_eq!(empty.incoming_edges_at_epoch(1, epoch), Some(vec![]));
            assert!(empty.extend_incoming_at_epoch(1, epoch, &mut out));
            assert_eq!(out, vec![99]);
        }
    }

    #[test]
    fn test_degree() {
        let rt = make_test_rel_table();

        assert_eq!(rt.out_degree(0), 2);
        assert_eq!(rt.out_degree(1), 2);
        assert_eq!(rt.out_degree(2), 1);

        assert_eq!(rt.in_degree(0), Some(3));
        assert_eq!(rt.in_degree(1), Some(2));
    }

    #[test]
    fn test_no_backward_csr() {
        let fwd_edges = vec![(0u32, 1u32)];
        let fwd = CsrAdjacency::from_sorted_edges(2, &fwd_edges);
        let schema = EdgeSchema::new("FOLLOWS", 10, "User", "User", vec![]);

        let rt = RelTable::new(schema, fwd, None, FxHashMap::default(), 0, 0);

        assert!(!rt.has_backward());
        assert_eq!(rt.edges_to_target(0), None);
        assert_eq!(rt.in_degree(0), None);
    }

    #[test]
    fn test_get_property_at_epoch_pending_vs_past() {
        use grafeo_common::types::{EpochId, EpochInterval};

        // One edge, two property columns: all-open `since` (fresh-base wrap)
        // and closed-interval `weight` (PENDING is latest-only; past epoch gates).
        let fwd = CsrAdjacency::from_sorted_edges(2, &[(0u32, 1u32)]);
        let schema = EdgeSchema::new("KNOWS", 1, "Person", "Person", vec![]);
        let since = PropertyKey::new("since");
        let weight = PropertyKey::new("weight");

        let mut properties = FxHashMap::default();
        properties.insert(
            since.clone(),
            TemporalColumn::all_open(ColumnCodec::raw_i64(vec![1999])),
        );
        properties.insert(
            weight.clone(),
            TemporalColumn::new(
                ColumnCodec::raw_i64(vec![10]),
                vec![EpochInterval::closed(EpochId::new(10), EpochId::new(20))],
            ),
        );

        let rt = RelTable::new(schema, fwd, None, properties, 0, 0);

        // All-open: PENDING (latest) and any real past epoch both see the value.
        assert_eq!(
            rt.get_property_at_epoch(0, &since, EpochId::PENDING),
            Some(Value::Int64(1999))
        );
        assert_eq!(
            rt.get_property_at_epoch(0, &since, EpochId::new(1)),
            Some(Value::Int64(1999))
        );

        // Closed [10, 20): PENDING requires an open interval; past-in-range is visible.
        assert_eq!(rt.get_property_at_epoch(0, &weight, EpochId::PENDING), None);
        assert_eq!(
            rt.get_property_at_epoch(0, &weight, EpochId::new(15)),
            Some(Value::Int64(10))
        );
        assert_eq!(rt.get_property_at_epoch(0, &weight, EpochId::new(5)), None);
        assert_eq!(
            rt.get_property_at_epoch(0, &PropertyKey::new("missing"), EpochId::PENDING),
            None
        );
    }

    #[test]
    fn test_structural_validity_gates_as_of_property() {
        let fwd = CsrAdjacency::from_sorted_edges(2, &[(0u32, 1u32)]);
        let schema = EdgeSchema::new("KNOWS", 1, "Person", "Person", vec![]);
        let since = PropertyKey::new("since");
        let mut properties = FxHashMap::default();
        properties.insert(
            since.clone(),
            TemporalColumn::all_open(ColumnCodec::raw_i64(vec![1999])),
        );
        let mut rt = RelTable::new(schema, fwd, None, properties, 0, 0);
        rt.set_validity(vec![EpochInterval::open(EpochId::new(10))]);

        assert_eq!(
            rt.get_property_at_epoch(0, &since, EpochId::new(15)),
            Some(Value::Int64(1999))
        );
        assert_eq!(rt.get_property_at_epoch(0, &since, EpochId::new(5)), None);
        assert_eq!(
            rt.get_property_at_epoch(0, &since, EpochId::PENDING),
            Some(Value::Int64(1999))
        );
    }

    #[test]
    fn install_packed_open_keeps_current_csr_open_only() {
        use super::super::csr::{
            TemporalEdgeRow, build_current_csr_from_open_edges, pack_open_prefix,
        };

        let rows = [
            TemporalEdgeRow {
                src: 0,
                dst: 1,
                validity: EpochInterval::open(EpochId::new(10)),
                edge_id: EdgeId::new(1),
            },
            TemporalEdgeRow {
                src: 0,
                dst: 2,
                validity: EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
                edge_id: EdgeId::new(2),
            },
        ];
        let derived = build_current_csr_from_open_edges(3, &rows);
        let packed = pack_open_prefix(3, &rows);
        let schema = EdgeSchema::new("KNOWS", 1, "Person", "Person", vec![]);
        // Start from an open-only snapshot CSR (what from_graph_store emits).
        let mut rt = RelTable::new(schema, derived.clone(), None, FxHashMap::default(), 0, 0);
        rt.install_packed_open(packed, None, derived);

        assert_eq!(rt.num_edges(), 1);
        assert_eq!(rt.current_targets(0), &[1]);
        assert_eq!(rt.neighbors_at_epoch(0, EpochId::PENDING), vec![1]);
        let mut at15 = rt.neighbors_at_epoch(0, EpochId::new(15));
        at15.sort_unstable();
        assert_eq!(at15, vec![1, 2]);
        assert_eq!(rt.neighbors_at_epoch(0, EpochId::new(25)), vec![1]);
        assert_eq!(
            rt.packed_fwd().map(PackedOpenAdjacency::num_versions),
            Some(1),
            "RAM packed keeps closed tails only"
        );
        assert_eq!(rt.packed_fwd().map(PackedOpenAdjacency::num_open), Some(0));
        assert!(
            rt.validity().is_empty(),
            "open-prefix validity is not duplicated when packed is installed"
        );
        assert_eq!(
            rt.fwd().num_edges(),
            1,
            "current 1-hop is the tight derived CSR"
        );
        assert!(!rt.current_from_packed());
        assert!(rt.packed_bwd().is_none());
        assert_eq!(rt.out_degree(0), 1);
        assert_eq!(rt.edges_from_source(0).len(), 1);
        assert_eq!(rt.in_degree(1), Some(1));
        assert_eq!(rt.packed_closed_pos(EdgeId::new(2)), Some(0));
        assert_eq!(rt.packed_closed_pos(EdgeId::new(1)), None);
    }

    #[test]
    fn slim_no_prop_bwd_uses_dest_offsets_not_src_count() {
        use super::super::csr::{
            TemporalEdgeRow, build_current_csr_from_open_edges, pack_open_prefix,
        };

        // 2 sources, dest offsets 0 and 7 (OBSERVED_BY Source→Entity).
        let rows = [
            TemporalEdgeRow {
                src: 0,
                dst: 0,
                validity: EpochInterval::open(EpochId::new(10)),
                edge_id: EdgeId::new(1),
            },
            TemporalEdgeRow {
                src: 1,
                dst: 7,
                validity: EpochInterval::open(EpochId::new(10)),
                edge_id: EdgeId::new(2),
            },
        ];
        let derived = build_current_csr_from_open_edges(2, &rows);
        let packed = pack_open_prefix(2, &rows);
        let schema = EdgeSchema::new("OBSERVED_BY", 1, "Source", "Entity", vec![]);
        let mut rt = RelTable::new(schema, derived.clone(), None, FxHashMap::default(), 0, 1);
        rt.install_packed_open(packed, None, derived);

        assert_eq!(rt.fwd().num_nodes(), 2);
        assert!(
            rt.bwd().is_some_and(|b| b.num_nodes() >= 8),
            "backward CSR must cover dest offset 7, not the 2-source table"
        );
        assert_eq!(rt.in_degree(0), Some(1));
        assert_eq!(rt.in_degree(7), Some(1));
        assert_eq!(
            rt.incoming_at_epoch(7, EpochId::PENDING).as_deref(),
            Some(&[1][..])
        );
    }
}
