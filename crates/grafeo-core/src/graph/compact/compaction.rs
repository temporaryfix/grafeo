//! Compaction: folding hot VersionLog history into the cold temporal columns.
//!
//! [`crate::graph::compact::compaction::fold_history`] turns a slot's epoch-ordered version history — `get_history`'s
//! `(EpochId, Value)` pairs — into `(Value, EpochInterval)` rows: consecutive
//! versions define half-open `[from, to)` windows, the last version stays open, a
//! same-epoch rewrite collapses to the later value, and a **tombstone** (a
//! `Value::Null` from a property remove) closes the prior window and leaves a gap
//! (no row covers it — `as_of` in the gap returns nothing). The value column is
//! therefore **null-free**. The same fold applies to node properties, edge
//! properties, and (later) fact/statement cells — it is not node-specific.
//! [`crate::graph::compact::compaction::fold_to_temporal_column`] encodes those rows into a [`crate::graph::compact::temporal_column::TemporalColumn`];
//! [`crate::graph::compact::compaction::fold_edge_properties`] maps a per-key history bag through that path;
//! [`crate::graph::compact::compaction::fold_edge_validity`] turns overlay create/delete stamps into structural
//! `[created, deleted)` intervals. [`crate::graph::compact::compaction::compact_all_history`] does a whole entity's
//! properties; [`crate::graph::compact::compaction::compact_entity_from_store`] reads that history live from a
//! [`crate::graph::lpg::PropertyStorage`] (an immutable read — no hot-path mutation).

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, EpochInterval, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::column::ColumnCodec;
use super::csr::{PackedOpenAdjacency, TemporalEdgeRow};
use super::temporal_column::TemporalColumn;
use crate::codec::{BitPackedInts, BitVector, DictionaryBuilder};
use crate::graph::lpg::PropertyStorage;

pub use super::csr::{build_current_csr_from_open_edges, pack_open_prefix};

/// Folds an epoch-ordered version history into `(value, validity-interval)` rows.
///
/// `history` must be sorted by epoch ascending (as `get_history` returns it). Each
/// non-null version is valid from its epoch until the next version's epoch; the
/// final version's interval is open. Same-epoch rewrites collapse to the later
/// value, and tombstones (`Value::Null`, i.e. removes) close the prior window and
/// emit no row — so removed ranges are gaps and the value column carries no nulls.
#[must_use]
pub fn fold_history(history: &[(EpochId, Value)]) -> Vec<(Value, EpochInterval)> {
    let mut rows: Vec<(Value, EpochInterval)> = Vec::with_capacity(history.len());
    for (i, (epoch, value)) in history.iter().enumerate() {
        let to = history
            .get(i + 1)
            .map_or(EpochId::PENDING, |(next, _)| *next);
        if *epoch == to {
            continue; // zero-width window (same-epoch rewrite) — the next row wins
        }
        if value.is_null() {
            continue; // tombstone (remove): the prior window already closes at this
            // epoch; the gap stays uncovered (as_of returns nothing)
        }
        rows.push((value.clone(), EpochInterval::closed(*epoch, to)));
    }
    rows
}

/// Builds a raw [`ColumnCodec`] from a homogeneous, non-null value slice.
///
/// The folded values carry no nulls (removes are gaps — see [`fold_history`]), so
/// this only handles non-null homogeneous values: non-negative ints
/// (`BitPacked`), signed ints (`RawI64`), floats, strings (`Dict`), bools
/// (`Bitmap`), and float vectors (`Float32Vector`, fixed dimension). Returns
/// `None` for an empty slice, mixed types, vectors of differing (or zero / >u16)
/// dimensions, or a type not supported here.
fn codec_from_values(values: &[Value]) -> Option<ColumnCodec> {
    match values.first()? {
        Value::Int64(_) => {
            let ints: Vec<i64> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Int64(n) => Some(*n),
                    _ => None,
                })
                .collect();
            if ints.len() != values.len() {
                return None;
            }
            if ints.iter().all(|&value| value >= 0) {
                let unsigned = ints
                    .into_iter()
                    .map(|value| {
                        u64::try_from(value).expect("non-negative i64 values always fit u64")
                    })
                    .collect::<Vec<_>>();
                Some(ColumnCodec::BitPacked(BitPackedInts::pack(&unsigned)))
            } else {
                Some(ColumnCodec::raw_i64(ints))
            }
        }
        Value::Float64(_) => {
            let floats: Vec<f64> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Float64(f) => Some(*f),
                    _ => None,
                })
                .collect();
            (floats.len() == values.len()).then(|| ColumnCodec::float64(floats))
        }
        Value::String(_) => {
            let strings: Vec<&str> = values
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            (strings.len() == values.len()).then(|| {
                let mut builder = DictionaryBuilder::new();
                for s in &strings {
                    builder.add(s);
                }
                ColumnCodec::Dict(builder.build())
            })
        }
        Value::Bool(_) => {
            let bools: Vec<bool> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect();
            (bools.len() == values.len())
                .then(|| ColumnCodec::Bitmap(BitVector::from_bools(&bools)))
        }
        Value::Vector(first) => {
            // Vectors fold into a fixed-stride `Float32Vector` column. A single
            // column has one stride, so every row must share `first`'s dimension
            // (and it must fit in u16 and be non-zero — a 0-dim stride can't be
            // decoded). Differing dims / a too-large dim / a non-vector row makes
            // this not a homogeneous foldable column, so we return None and the
            // column stays all-open (current value only).
            //
            // Density note: unlike ints/floats/strings/bools, folding a vector
            // column stores *every* historical embedding verbatim (no
            // cross-version compression). Vectors are large, so this trades space
            // for as-of correctness — we keep the full history so a read at a past
            // epoch returns that epoch's embedding rather than only the latest.
            let dims = u16::try_from(first.len()).ok().filter(|&d| d > 0)?;
            let dim = first.len();
            let mut flat: Vec<f32> = Vec::with_capacity(values.len() * dim);
            for v in values {
                match v {
                    Value::Vector(c) if c.len() == dim => flat.extend_from_slice(c),
                    _ => return None,
                }
            }
            Some(ColumnCodec::float32_vector(flat, dims))
        }
        _ => None,
    }
}

/// Folds a slot's version history into a [`TemporalColumn`] — the compaction output
/// for one property column.
///
/// Returns `None` if the (non-null) values' type isn't yet supported by the cold
/// codec (extended in later slices); an empty or all-tombstone history also yields
/// `None` (nothing to encode).
#[must_use]
pub fn fold_to_temporal_column(history: &[(EpochId, Value)]) -> Option<TemporalColumn> {
    let rows = fold_history(history);
    let (values, validity): (Vec<Value>, Vec<EpochInterval>) = rows.into_iter().unzip();
    let codec = codec_from_values(&values)?;
    Some(TemporalColumn::new(codec, validity))
}

/// Folds one property's per-node history into a single temporal column plus
/// per-node `(row_start, row_count)` ranges — the per-property building block of
/// SP2 compaction.
///
/// `node_property_histories[i]` is node offset `i`'s epoch-ascending history for
/// the property (empty if the node lacks it — it then contributes zero rows).
/// Each node's history folds via [`fold_history`]; the runs concatenate in node
/// order, so the column is sorted by `(node, epoch)`. Returns `None` if the
/// (non-null) values' type isn't a supported cold codec (strings/bools/vectors
/// arrive in later slices), matching [`fold_to_temporal_column`].
#[must_use]
pub fn fold_property_across_nodes(
    node_property_histories: &[Vec<(EpochId, Value)>],
) -> Option<(TemporalColumn, Vec<(u32, u32)>)> {
    let mut values: Vec<Value> = Vec::new();
    let mut validity: Vec<EpochInterval> = Vec::new();
    let mut ranges: Vec<(u32, u32)> = Vec::with_capacity(node_property_histories.len());
    for history in node_property_histories {
        let start = values.len();
        for (value, interval) in fold_history(history) {
            values.push(value);
            validity.push(interval);
        }
        // reason: per-node row counts are bounded by u32::MAX (section format).
        #[allow(clippy::cast_possible_truncation)]
        ranges.push((start as u32, (values.len() - start) as u32));
    }
    let codec = codec_from_values(&values)?;
    Some((TemporalColumn::new(codec, validity), ranges))
}

/// Compacts a whole entity's property history (as from `get_all_history`) into
/// per-property [`TemporalColumn`]s, skipping properties whose type the cold codec
/// doesn't yet support. Generic over the key so it stays decoupled from the store.
#[must_use]
pub fn compact_all_history<K: Clone>(
    all_history: &[(K, Vec<(EpochId, Value)>)],
) -> Vec<(K, TemporalColumn)> {
    all_history
        .iter()
        .filter_map(|(key, history)| fold_to_temporal_column(history).map(|tc| (key.clone(), tc)))
        .collect()
}

/// Compacts one node's live property history — read from a [`PropertyStorage`] via
/// `get_all_history` — into per-property [`TemporalColumn`]s.
///
/// This is an *immutable read* of the store (no hot-path mutation). The compaction
/// trigger (when/how to run, and how to install the result into the cold base) is a
/// later SP2 slice; this is its read + transform core. Node-typed for now; edges
/// follow once `EntityId` is exposed.
#[must_use]
pub fn compact_entity_from_store(
    store: &PropertyStorage,
    id: NodeId,
) -> Vec<(PropertyKey, TemporalColumn)> {
    compact_all_history(&store.get_all_history(id))
}

/// One committed structural lifetime of an edge: created at `created`, and
/// optionally deleted at `deleted` (`None` / `PENDING` = still open).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeLifetime {
    /// Commit epoch of the create (or `PENDING` while uncommitted).
    pub created: EpochId,
    /// Commit epoch of the delete, if any.
    pub deleted: Option<EpochId>,
}

impl EdgeLifetime {
    /// Builds a lifetime from overlay `get_edge_history` stamps.
    #[must_use]
    pub const fn new(created: EpochId, deleted: Option<EpochId>) -> Self {
        Self { created, deleted }
    }
}

/// Overlay (or reconstructed base) history for one `EdgeId`, used at temporal
/// merge to fold properties and derive structural validity.
#[derive(Debug, Clone, Default)]
pub struct EdgeFullHistory {
    /// Source endpoint.
    pub src: NodeId,
    /// Destination endpoint.
    pub dst: NodeId,
    /// Relationship type name.
    pub edge_type: ArcStr,
    /// Structural create/delete versions, oldest first.
    pub lifetimes: Vec<EdgeLifetime>,
    /// Per-property epoch-ascending version log (includes `Null` tombstones).
    pub properties: FxHashMap<PropertyKey, Vec<(EpochId, Value)>>,
}

impl EdgeFullHistory {
    /// Whether this history has no committed structural lifetime.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lifetimes.is_empty()
    }
}

/// One folded structural edge row. Persisted in the v5 closed-edge sidecar.
#[derive(Debug, Clone)]
pub struct FoldedEdgeRow {
    /// Original edge id.
    pub id: EdgeId,
    /// Source endpoint.
    pub src: NodeId,
    /// Destination endpoint.
    pub dst: NodeId,
    /// Relationship type name.
    pub edge_type: ArcStr,
    /// Structural validity of this row.
    pub validity: EpochInterval,
    /// Folded property columns for this lifetime (may have multiple runs).
    pub properties: FxHashMap<PropertyKey, TemporalColumn>,
    /// Exact fallback columns for histories that cannot be represented by one
    /// homogeneous cold codec (for example `Int64` -> `String`, or vectors
    /// whose dimensions change). Mutually exclusive with `properties` per key.
    pub raw_properties: FxHashMap<PropertyKey, RawTemporalColumn>,
}

/// Exact temporal property rows used when a homogeneous [`ColumnCodec`] cannot
/// represent a history without losing values.
///
/// This is intentionally a sidecar fallback, not the default physical layout:
/// homogeneous histories keep the compressed columnar fast path while mixed or
/// otherwise unsupported values retain their precise half-open validity runs.
#[derive(Debug, Clone)]
pub struct RawTemporalColumn {
    history: Vec<(EpochId, Value)>,
    rows: Vec<(Value, EpochInterval)>,
}

impl RawTemporalColumn {
    /// Builds an exact fallback column from an epoch-ordered version log.
    /// Same-epoch entries remain ordered in `history`; `rows` is the derived
    /// observable state used by as-of reads.
    #[must_use]
    pub fn new(history: Vec<(EpochId, Value)>) -> Self {
        let rows = fold_history(&history);
        Self { history, rows }
    }

    /// Number of exact version-log entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the fallback contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Exact value visible at `epoch`; `PENDING` means the current open run.
    #[must_use]
    pub fn value_as_of(&self, epoch: EpochId) -> Option<Value> {
        self.rows.iter().find_map(|(value, validity)| {
            let visible = if epoch == EpochId::PENDING {
                validity.is_open()
            } else {
                validity.contains(epoch)
            };
            visible.then(|| value.clone())
        })
    }

    /// Returns the exact epoch-ordered history, including ordered same-epoch
    /// writes and tombstones at a structural delete boundary.
    #[must_use]
    pub fn runs_as_history(&self) -> Vec<(EpochId, Value)> {
        self.history.clone()
    }

    /// Exact version-log entries, used by the versioned compact-section codec.
    #[must_use]
    pub fn history(&self) -> &[(EpochId, Value)] {
        &self.history
    }

    /// Approximate heap bytes retained by the fallback rows. Dynamic payloads
    /// are included using [`Value::estimated_size_bytes`].
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<(Value, EpochInterval)>()
            + self
                .history
                .iter()
                .map(|(_, value)| value.estimated_size_bytes())
                .sum::<usize>()
            + self.history.capacity() * std::mem::size_of::<(EpochId, Value)>()
    }
}

/// Overlay (or reconstructed base) history for one node. Structural lifetime
/// is kept independently of properties so label-only nodes have exact as-of
/// visibility and fully deleted nodes can be retained outside current tables.
#[derive(Debug, Clone, Default)]
pub struct NodeFullHistory {
    /// Labels carried by the node. Compact current tables project one primary
    /// label; this remains the latest complete set for compatibility.
    pub labels: Vec<ArcStr>,
    /// Epoch-ascending complete label-set versions. Entries at the same epoch
    /// retain insertion order and the last entry is authoritative. This is
    /// separate from structural lifetimes because labels may change while a
    /// node remains structurally alive.
    pub label_versions: Vec<(EpochId, Vec<ArcStr>)>,
    /// Structural create/delete versions, oldest first.
    pub lifetimes: Vec<EdgeLifetime>,
    /// Per-property epoch-ascending version log (includes `Null` tombstones).
    pub properties: FxHashMap<PropertyKey, Vec<(EpochId, Value)>>,
}

impl NodeFullHistory {
    /// Whether this history has no committed structural lifetime.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lifetimes.is_empty()
    }
}

/// One folded node lifetime persisted in CompactStore's temporal-node sidecar.
#[derive(Debug, Clone)]
pub struct FoldedNodeRow {
    /// Original node id.
    pub id: NodeId,
    /// Complete label set for historical reconstruction.
    pub labels: Vec<ArcStr>,
    /// Complete label-set versions clipped to this structural lifetime,
    /// ordered by ascending epoch.
    pub label_versions: Vec<(EpochId, Vec<ArcStr>)>,
    /// Structural validity of the node.
    pub validity: EpochInterval,
    /// Folded property columns for this lifetime.
    pub properties: FxHashMap<PropertyKey, TemporalColumn>,
    /// Exact fallback columns for non-homogeneous or otherwise unencodable
    /// histories. Mutually exclusive with `properties` per key.
    pub raw_properties: FxHashMap<PropertyKey, RawTemporalColumn>,
}

/// Merges cold-base structural lives with overlay create/delete stamps.
///
/// A property-write promote copies the edge into the overlay at
/// `current_epoch()`, restamping create. When the base already has validity,
/// keep the earlier `from` on the continued life and retain any base lives
/// that closed before the overlay chain starts.
#[must_use]
pub fn merge_edge_lifetimes(base: &[EdgeLifetime], overlay: &[EdgeLifetime]) -> Vec<EdgeLifetime> {
    if overlay.is_empty() {
        return base.to_vec();
    }
    if base.is_empty() {
        return overlay.to_vec();
    }

    // Exact same-incarnation promotion copies the complete base history.
    // Recognize that ordered prefix before the legacy continued-life merge:
    // a zero-width life at overlay_start otherwise gets appended twice.
    // Compare positionally, preserving repeated zero-width lives rather than
    // deduplicating intervals by value. Only the final open base life may
    // gain a deletion stamp in the promoted overlay.
    if base.len() <= overlay.len()
        && base
            .iter()
            .zip(overlay)
            .enumerate()
            .all(|(position, (old, new))| {
                old.created == new.created
                    && (old.deleted == new.deleted
                        || (position + 1 == base.len() && old.deleted.is_none()))
            })
    {
        return overlay.to_vec();
    }

    let overlay_start = overlay[0].created;
    let mut out = Vec::with_capacity(base.len() + overlay.len());
    let mut continuing = None;
    for &lt in base {
        let closed_before = matches!(
            lt.deleted,
            Some(d) if d != EpochId::PENDING && d <= overlay_start
        );
        if closed_before {
            out.push(lt);
        } else {
            continuing = Some(lt);
        }
    }

    let mut merged = overlay.to_vec();
    if let Some(base_life) = continuing
        && merged[0].created != EpochId::PENDING
        && base_life.created < merged[0].created
    {
        merged[0].created = base_life.created;
    }
    out.extend(merged);
    out
}

/// Folds overlay create/delete stamps into half-open structural intervals.
///
/// Uncommitted creates (`created == PENDING`) are dropped. An uncommitted
/// delete (`deleted == PENDING` or `None`) leaves the interval open. A
/// committed delete at `D` yields `[C, D)` and the row is retained. Two
/// lifetimes on the same endpoints (delete-then-recreate) become two
/// disjoint intervals — the gap must not be covered (no resurrection).
/// A zero-width committed lifetime is retained for exact audit history but
/// contains no epoch and never contributes to the current open prefix.
#[must_use]
pub fn fold_edge_validity(versions: &[(EpochId, Option<EpochId>)]) -> Vec<EpochInterval> {
    versions
        .iter()
        .filter_map(|(created, deleted)| {
            if *created == EpochId::PENDING {
                return None;
            }
            match *deleted {
                None => Some(EpochInterval::open(*created)),
                Some(d) if d == EpochId::PENDING => Some(EpochInterval::open(*created)),
                Some(d) => Some(EpochInterval::closed(*created, d)),
            }
        })
        .collect()
}

/// Folds each property's epoch-ordered history into a [`TemporalColumn`].
///
/// Empty or non-homogeneous histories are omitted (same rule as
/// [`fold_to_temporal_column`]). [`fold_history`] is the per-key transform —
/// including remove/tombstone gaps — so edge property logs use the same
/// null-free interval encoding as node properties.
#[must_use]
pub fn fold_edge_properties(
    histories: FxHashMap<PropertyKey, Vec<(EpochId, Value)>>,
) -> FxHashMap<PropertyKey, TemporalColumn> {
    histories
        .into_iter()
        .filter_map(|(k, h)| fold_to_temporal_column(&h).map(|tc| (k, tc)))
        .collect()
}

/// Folds every non-empty property history without loss.
///
/// Histories representable by one homogeneous cold codec are returned in the
/// first map. Mixed types, changing vector dimensions, and value kinds without
/// a cold codec are returned as exact [`RawTemporalColumn`]s in the second map.
/// A key is present in at most one map.
#[must_use]
pub fn fold_properties_exact(
    histories: FxHashMap<PropertyKey, Vec<(EpochId, Value)>>,
) -> (
    FxHashMap<PropertyKey, TemporalColumn>,
    FxHashMap<PropertyKey, RawTemporalColumn>,
) {
    let mut encoded = FxHashMap::default();
    let mut raw = FxHashMap::default();
    for (key, history) in histories {
        if history.is_empty() {
            continue;
        }
        let rows = fold_history(&history);
        let (values, validity): (Vec<Value>, Vec<EpochInterval>) = rows.iter().cloned().unzip();
        let has_same_epoch_entries = history.windows(2).any(|pair| pair[0].0 == pair[1].0);
        if !has_same_epoch_entries && let Some(codec) = codec_from_values(&values) {
            encoded.insert(key, TemporalColumn::new(codec, validity));
        } else {
            raw.insert(key, RawTemporalColumn::new(history));
        }
    }
    (encoded, raw)
}

/// Drops versions outside `validity` without inventing property mutations.
///
/// Structural validity already gates every folded node/edge row, so a closed
/// entity lifetime does not need a synthetic `Null` property version at its
/// delete epoch. A `Null` present in the source log is retained exactly; one
/// absent from that log must remain absent from exported/reconstructed history.
#[must_use]
pub fn clip_history_to_interval(
    history: &[(EpochId, Value)],
    validity: EpochInterval,
) -> Vec<(EpochId, Value)> {
    history
        .iter()
        .filter(|(epoch, _)| {
            *epoch >= validity.from() && (validity.is_open() || *epoch <= validity.to())
        })
        .cloned()
        .collect()
}

/// Clips complete label-set versions to one structural lifetime.
///
/// The state visible at `validity.from()` is emitted as the first entry even
/// when its source version predates the lifetime, followed by every change
/// through the structural delete boundary. Delete-boundary entries are retained
/// because exact persistence/recovery uses them to reproduce committed label
/// logs even though the node itself is no longer visible at that epoch. Names
/// within each label set are canonicalized; explicit image entries are never
/// coalesced, even when their sets and epochs are equal. A synthetic baseline
/// is needed only when no explicit image exists at the interval's start.
#[must_use]
pub fn clip_label_history_to_interval(
    history: &[(EpochId, Vec<ArcStr>)],
    fallback: &[ArcStr],
    validity: EpochInterval,
) -> Vec<(EpochId, Vec<ArcStr>)> {
    let from = validity.from();
    let mut out = Vec::new();
    if !history.iter().any(|(epoch, _)| *epoch == from) {
        let mut baseline = history
            .iter()
            .rev()
            .find(|(epoch, _)| *epoch < from)
            .map_or_else(|| fallback.to_vec(), |(_, labels)| labels.clone());
        baseline.sort_unstable();
        baseline.dedup();
        out.push((from, baseline));
    }
    for (epoch, labels) in history
        .iter()
        .filter(|(epoch, _)| *epoch >= from && (validity.is_open() || *epoch <= validity.to()))
    {
        let mut labels = labels.clone();
        labels.sort_unstable();
        labels.dedup();
        out.push((*epoch, labels));
    }
    out
}

/// Keeps only lifetimes and property versions at or before `boundary`.
/// Uncommitted (`PENDING`) creates are dropped; uncommitted deletes stay open.
#[must_use]
pub fn committed_edge_history(mut history: EdgeFullHistory, boundary: EpochId) -> EdgeFullHistory {
    history
        .lifetimes
        .retain(|lt| lt.created != EpochId::PENDING && lt.created <= boundary);
    for lt in &mut history.lifetimes {
        if let Some(d) = lt.deleted
            && (d == EpochId::PENDING || d > boundary)
        {
            lt.deleted = None;
        }
    }
    history.properties = history
        .properties
        .into_iter()
        .filter_map(|(key, versions)| {
            let kept: Vec<(EpochId, Value)> = versions
                .into_iter()
                .filter(|(epoch, _)| *epoch <= boundary)
                .collect();
            (!kept.is_empty()).then_some((key, kept))
        })
        .collect();
    history
}

/// Keeps only node lifetimes and property versions committed at `boundary`.
#[must_use]
pub fn committed_node_history(mut history: NodeFullHistory, boundary: EpochId) -> NodeFullHistory {
    history
        .lifetimes
        .retain(|lt| lt.created != EpochId::PENDING && lt.created <= boundary);
    for lt in &mut history.lifetimes {
        if let Some(d) = lt.deleted
            && (d == EpochId::PENDING || d > boundary)
        {
            lt.deleted = None;
        }
    }
    history.properties = history
        .properties
        .into_iter()
        .filter_map(|(key, versions)| {
            let kept: Vec<(EpochId, Value)> = versions
                .into_iter()
                .filter(|(epoch, _)| *epoch <= boundary)
                .collect();
            (!kept.is_empty()).then_some((key, kept))
        })
        .collect();
    history
        .label_versions
        .retain(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= boundary);
    if let Some((_, labels)) = history.label_versions.last() {
        history.labels.clone_from(labels);
    }
    history
}

/// Folds one node's committed history into structural rows.
#[must_use]
pub fn fold_node_rows(id: NodeId, history: &NodeFullHistory) -> Vec<FoldedNodeRow> {
    let stamps: Vec<(EpochId, Option<EpochId>)> = history
        .lifetimes
        .iter()
        .map(|lt| (lt.created, lt.deleted))
        .collect();
    let validities = fold_edge_validity(&stamps);
    validities
        .iter()
        .copied()
        .enumerate()
        .map(|(position, validity)| {
            let clipped: FxHashMap<PropertyKey, Vec<(EpochId, Value)>> = history
                .properties
                .iter()
                .map(|(key, versions)| (key.clone(), clip_history_to_interval(versions, validity)))
                .collect();
            let mut label_versions =
                clip_label_history_to_interval(&history.label_versions, &history.labels, validity);
            // Exact restore assigns a shared boundary's last image to the
            // new lifetime and preceding images to the closing lifetime.
            // Do not duplicate that physical boundary across folded rows or
            // coalesce genuinely separate equal images from the two lives.
            if position
                .checked_sub(1)
                .and_then(|prior| validities.get(prior))
                .is_some_and(|prior| !prior.is_open() && prior.to() == validity.from())
            {
                let after_birth =
                    label_versions.partition_point(|(epoch, _)| *epoch <= validity.from());
                if after_birth > 1 {
                    label_versions.drain(..after_birth - 1);
                }
            }
            if !validity.is_open()
                && validities
                    .get(position + 1)
                    .is_some_and(|next| next.from() == validity.to())
                && label_versions
                    .last()
                    .is_some_and(|(epoch, _)| *epoch == validity.to())
            {
                label_versions.pop();
            }
            let labels = label_versions
                .last()
                .map_or_else(|| history.labels.clone(), |(_, labels)| labels.clone());
            let (properties, raw_properties) = fold_properties_exact(clipped);
            FoldedNodeRow {
                id,
                labels,
                label_versions,
                validity,
                properties,
                raw_properties,
            }
        })
        .collect()
}

/// Folds one edge's committed history into structural rows (usually one).
///
/// Property histories are clipped to each lifetime so a delete closes property
/// runs at the same epoch as the edge. Uncommitted lifetimes yield no row.
#[must_use]
pub fn fold_edge_row(id: EdgeId, history: &EdgeFullHistory) -> Vec<FoldedEdgeRow> {
    let stamps: Vec<(EpochId, Option<EpochId>)> = history
        .lifetimes
        .iter()
        .map(|lt| (lt.created, lt.deleted))
        .collect();
    fold_edge_validity(&stamps)
        .into_iter()
        .map(|validity| {
            let clipped: FxHashMap<PropertyKey, Vec<(EpochId, Value)>> = history
                .properties
                .iter()
                .map(|(k, h)| (k.clone(), clip_history_to_interval(h, validity)))
                .collect();
            let (properties, raw_properties) = fold_properties_exact(clipped);
            FoldedEdgeRow {
                id,
                src: history.src,
                dst: history.dst,
                edge_type: history.edge_type.clone(),
                validity,
                properties,
                raw_properties,
            }
        })
        .collect()
}

/// Packs temporal edge rows (Option A) and derives the current CSR from the
/// open prefix. Current 1-hop is the derived CSR / prefix slice — never a
/// `filter(is_open)` walk of the fat run.
#[must_use]
pub fn pack_and_derive_current(
    num_nodes: usize,
    rows: &[TemporalEdgeRow],
) -> (PackedOpenAdjacency, super::csr::CsrAdjacency) {
    let packed = pack_open_prefix(num_nodes, rows);
    let current = packed.derive_current_csr();
    (packed, current)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(n: u64) -> EpochId {
        EpochId::new(n)
    }

    #[test]
    fn test_fold_builds_consecutive_intervals_last_open() {
        let history = vec![
            (e(10), Value::Int64(100)),
            (e(20), Value::Int64(200)),
            (e(30), Value::Int64(300)),
        ];
        let rows = fold_history(&history);
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[0],
            (Value::Int64(100), EpochInterval::closed(e(10), e(20)))
        );
        assert_eq!(
            rows[1],
            (Value::Int64(200), EpochInterval::closed(e(20), e(30)))
        );
        assert_eq!(rows[2].0, Value::Int64(300));
        assert!(rows[2].1.is_open());
    }

    #[test]
    fn test_fold_empty() {
        assert!(fold_history(&[]).is_empty());
    }

    #[test]
    fn test_fold_single_is_open() {
        let rows = fold_history(&[(e(5), Value::Int64(42))]);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].1.is_open());
        assert_eq!(rows[0].1.from(), e(5));
    }

    #[test]
    fn test_fold_drops_same_epoch_rewrite() {
        let history = vec![
            (e(5), Value::Int64(1)),
            (e(5), Value::Int64(2)),
            (e(10), Value::Int64(3)),
        ];
        let rows = fold_history(&history);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            (Value::Int64(2), EpochInterval::closed(e(5), e(10)))
        );
        assert!(rows[1].1.is_open());
    }

    #[test]
    fn test_fold_tombstone_leaves_a_gap() {
        let history = vec![
            (e(10), Value::Int64(1)),
            (e(20), Value::Null),
            (e(30), Value::Int64(2)),
        ];
        let rows = fold_history(&history);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            (Value::Int64(1), EpochInterval::closed(e(10), e(20)))
        );
        assert_eq!(rows[1].0, Value::Int64(2));
        assert_eq!(rows[1].1.from(), e(30));
        assert!(rows[1].1.is_open());
    }

    #[test]
    fn test_fold_to_temporal_column_int64() {
        let history = vec![(e(10), Value::Int64(100)), (e(20), Value::Int64(i64::MAX))];
        let tc = fold_to_temporal_column(&history).expect("int64 column");
        assert!(matches!(tc.values(), ColumnCodec::BitPacked(_)));
        assert_eq!(tc.len(), 2);
        assert_eq!(tc.value_as_of(0, e(15)), Some(Value::Int64(100)));
        assert_eq!(tc.value_as_of(1, e(25)), Some(Value::Int64(i64::MAX)));
        assert!(!tc.may_contain(e(5)));
    }

    #[test]
    fn test_fold_to_temporal_column_negative_int64_stays_signed() {
        let history = vec![(e(10), Value::Int64(-1)), (e(20), Value::Int64(2))];
        let tc = fold_to_temporal_column(&history).expect("signed int64 column");
        assert!(matches!(tc.values(), ColumnCodec::RawI64(_)));
        assert_eq!(tc.value_as_of(0, e(15)), Some(Value::Int64(-1)));
        assert_eq!(tc.value_as_of(1, e(25)), Some(Value::Int64(2)));
    }

    #[test]
    fn test_fold_to_temporal_column_tombstone_gap_is_uncovered() {
        let history = vec![(e(10), Value::Int64(1)), (e(20), Value::Null)];
        let tc = fold_to_temporal_column(&history).expect("int64 column");
        assert_eq!(tc.len(), 1);
        assert!(tc.value_as_of(0, e(15)).is_some());
        assert!(tc.rows_as_of(e(25)).is_empty());
    }

    #[test]
    fn test_fold_to_temporal_column_mixed_type_is_none() {
        // A history mixing types can't form one homogeneous column.
        let history = vec![(e(10), Value::Int64(1)), (e(20), Value::from("x"))];
        assert!(fold_to_temporal_column(&history).is_none());
    }

    #[test]
    fn test_fold_properties_exact_retains_mixed_types_and_vector_dimensions() {
        let mut histories = FxHashMap::default();
        histories.insert(
            PropertyKey::new("mixed"),
            vec![(e(10), Value::Int64(1)), (e(20), Value::from("one"))],
        );
        histories.insert(
            PropertyKey::new("embedding"),
            vec![
                (e(10), Value::Vector(vec![1.0, 2.0].into())),
                (e(20), Value::Vector(vec![3.0, 4.0, 5.0].into())),
            ],
        );
        histories.insert(
            PropertyKey::new("encoded"),
            vec![(e(10), Value::Int64(7)), (e(20), Value::Int64(8))],
        );

        let (encoded, raw) = fold_properties_exact(histories);
        assert_eq!(encoded.len(), 1);
        assert!(encoded.contains_key(&PropertyKey::new("encoded")));
        assert_eq!(raw.len(), 2);
        assert_eq!(
            raw[&PropertyKey::new("mixed")].value_as_of(e(15)),
            Some(Value::Int64(1))
        );
        assert_eq!(
            raw[&PropertyKey::new("mixed")].value_as_of(e(25)),
            Some(Value::from("one"))
        );
        assert_eq!(
            raw[&PropertyKey::new("embedding")].value_as_of(e(15)),
            Some(Value::Vector(vec![1.0, 2.0].into()))
        );
        assert_eq!(
            raw[&PropertyKey::new("embedding")].value_as_of(e(25)),
            Some(Value::Vector(vec![3.0, 4.0, 5.0].into()))
        );
    }

    #[test]
    fn test_fold_to_temporal_column_empty_is_none() {
        assert!(fold_to_temporal_column(&[]).is_none());
    }

    #[test]
    fn test_compact_all_history_per_property_folds_supported_skips_unsupported() {
        let all_history = vec![
            // int, with a remove -> folds (one row, gap after the tombstone)
            (
                "score",
                vec![(e(10), Value::Int64(1)), (e(20), Value::Null)],
            ),
            // bool -> now folds
            ("flag", vec![(e(10), Value::Bool(true))]),
            // mixed types within a property -> not a homogeneous column, skipped
            (
                "mixed",
                vec![(e(10), Value::Int64(1)), (e(20), Value::from("x"))],
            ),
        ];
        let cols = compact_all_history(&all_history);
        let keys: Vec<&str> = cols.iter().map(|(k, _)| *k).collect();
        assert_eq!(cols.len(), 2);
        assert!(keys.contains(&"score"));
        assert!(keys.contains(&"flag"));
        assert!(!keys.contains(&"mixed"));
        // "score"'s remove leaves a gap: as-of after it reads nothing.
        let score = &cols.iter().find(|(k, _)| *k == "score").unwrap().1;
        assert_eq!(score.len(), 1);
        assert!(score.rows_as_of(e(25)).is_empty());
    }

    #[test]
    fn test_fold_property_across_nodes_preserves_as_of() {
        // Node 0: 100@[10,20), 200@[20,PENDING). Node 1: 50@[5,PENDING). Node 2: absent.
        let histories = vec![
            vec![(e(10), Value::Int64(100)), (e(20), Value::Int64(200))],
            vec![(e(5), Value::Int64(50))],
            vec![],
        ];
        let (col, ranges) = fold_property_across_nodes(&histories).expect("int column");
        // node -> (row_start, row_count); node 2 (absent) contributes zero rows.
        assert_eq!(ranges, vec![(0, 2), (2, 1), (3, 0)]);
        // Invariant #1 (read-preserving): as-of via the compacted column equals
        // as-of via the original folded history, at every node and epoch.
        for (node, hist) in histories.iter().enumerate() {
            let (start, count) = (ranges[node].0 as usize, ranges[node].1 as usize);
            for ep in [e(0), e(7), e(15), e(25), e(100)] {
                let via_col = col.value_in_range_as_of(start, count, ep);
                let via_hist = fold_history(hist)
                    .iter()
                    .find(|(_, iv)| iv.contains(ep))
                    .map(|(v, _)| v.clone());
                assert_eq!(via_col, via_hist, "node {node} epoch {ep:?}");
            }
        }
    }

    #[test]
    fn test_fold_property_across_nodes_mixed_type_is_none() {
        // A property whose history mixes types can't form one homogeneous column.
        let histories = vec![vec![(e(10), Value::Int64(1)), (e(20), Value::from("oops"))]];
        assert!(fold_property_across_nodes(&histories).is_none());
    }

    #[test]
    fn test_fold_property_across_nodes_string_and_bool() {
        // String history folds into a Dict-backed temporal column.
        let strings = vec![vec![(e(10), Value::from("a")), (e(20), Value::from("b"))]];
        let (col, ranges) = fold_property_across_nodes(&strings).expect("string column");
        assert_eq!(ranges, vec![(0, 2)]);
        assert_eq!(
            col.value_in_range_as_of(0, 2, e(15)),
            Some(Value::from("a"))
        );
        assert_eq!(
            col.value_in_range_as_of(0, 2, e(25)),
            Some(Value::from("b"))
        );

        // Bool history folds into a Bitmap-backed temporal column.
        let bools = vec![vec![(e(5), Value::Bool(true)), (e(10), Value::Bool(false))]];
        let (bcol, _) = fold_property_across_nodes(&bools).expect("bool column");
        assert_eq!(
            bcol.value_in_range_as_of(0, 2, e(7)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            bcol.value_in_range_as_of(0, 2, e(12)),
            Some(Value::Bool(false))
        );
    }

    fn vec_value(components: &[f32]) -> Value {
        Value::Vector(std::sync::Arc::from(components))
    }

    #[test]
    fn test_fold_property_across_nodes_vector() {
        // Vector history folds into a Float32Vector-backed temporal column:
        // every historical embedding is retained and as-of reads the right one.
        let v_old = vec![0.1f32, 0.2, 0.3];
        let v_new = vec![0.4f32, 0.5, 0.6];
        let histories = vec![vec![(e(10), vec_value(&v_old)), (e(20), vec_value(&v_new))]];
        let (col, ranges) = fold_property_across_nodes(&histories).expect("vector column");
        assert_eq!(ranges, vec![(0, 2)]);
        assert_eq!(
            col.value_in_range_as_of(0, 2, e(15)),
            Some(vec_value(&v_old))
        );
        assert_eq!(
            col.value_in_range_as_of(0, 2, e(25)),
            Some(vec_value(&v_new))
        );
    }

    #[test]
    fn test_fold_property_across_nodes_vector_mixed_dims_is_none() {
        // Vectors of differing dimensions can't share one fixed-stride column.
        let histories = vec![vec![
            (e(10), vec_value(&[0.1, 0.2])),
            (e(20), vec_value(&[0.3, 0.4, 0.5])),
        ]];
        assert!(fold_property_across_nodes(&histories).is_none());
    }

    /// Invariant #1 (read-preserving) as a property over many generated
    /// write/remove-at-epoch histories: for every node and epoch, as-of via the
    /// compacted column equals as-of via the original folded history. Seeded
    /// xorshift so it is deterministic without a proptest dependency.
    #[test]
    fn test_fold_property_across_nodes_read_preserving_generated() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _trial in 0..300 {
            // One node's ascending-epoch history of sets and removes.
            let n_ops = usize::try_from(next() % 9).unwrap();
            let mut history: Vec<(EpochId, Value)> = Vec::new();
            let mut epoch = 1u64;
            for _ in 0..n_ops {
                epoch += 1 + next() % 5;
                if next() % 4 == 0 {
                    history.push((e(epoch), Value::Null)); // remove
                } else {
                    let v = i64::try_from(next() % 1000).unwrap();
                    history.push((e(epoch), Value::Int64(v)));
                }
            }
            let histories = vec![history.clone()];
            let Some((col, ranges)) = fold_property_across_nodes(&histories) else {
                continue; // all-tombstone or empty -> no column; nothing to read
            };
            let (start, count) = (ranges[0].0 as usize, ranges[0].1 as usize);
            let folded = fold_history(&history);
            for q in 0..=(epoch + 3) {
                let via_col = col.value_in_range_as_of(start, count, e(q));
                let via_hist = folded
                    .iter()
                    .find(|(_, iv)| iv.contains(e(q)))
                    .map(|(v, _)| v.clone());
                assert_eq!(via_col, via_hist, "epoch {q}, history {history:?}");
            }
        }
    }

    #[test]
    fn test_compact_entity_from_store_reads_live_history() {
        // A real PropertyStorage: set, set, remove (remove appends a Null tombstone).
        let store = PropertyStorage::<NodeId>::new();
        let id = NodeId::new(1);
        let key = PropertyKey::new("score");
        store.set(id, key.clone(), Value::Int64(100), e(10));
        store.set(id, key.clone(), Value::Int64(200), e(20));
        store.remove(id, &key, e(30));
        let cols = compact_entity_from_store(&store, id);
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].0, key);
        let tc = &cols[0].1;
        // 100 valid [10,20), 200 valid [20,30), gap after the remove at 30
        assert_eq!(tc.len(), 2);
        assert!(tc.value_as_of(0, e(15)).is_some());
        assert!(tc.value_as_of(1, e(25)).is_some());
        assert!(tc.rows_as_of(e(35)).is_empty());
    }

    /// Delete-then-recreate on the same endpoints must produce two disjoint
    /// intervals with an uncovered gap — the resurrection regression.
    #[test]
    fn test_merge_edge_lifetimes_exact_promotion_preserves_zero_width_multiplicity() {
        let zero = EdgeLifetime::new(e(10), Some(e(10)));
        let base = [zero, zero, EdgeLifetime::new(e(10), None)];
        assert_eq!(merge_edge_lifetimes(&base, &base), base);
        let closed = [zero, zero, EdgeLifetime::new(e(10), Some(e(20)))];
        assert_eq!(merge_edge_lifetimes(&base, &closed), closed);
        let recreated = [
            zero,
            zero,
            EdgeLifetime::new(e(10), Some(e(20))),
            EdgeLifetime::new(e(30), None),
        ];
        assert_eq!(merge_edge_lifetimes(&base, &recreated), recreated);
    }

    #[test]
    fn test_merge_edge_lifetimes_keeps_earlier_from_on_promote() {
        // Base [10, PENDING) + overlay restamp [20, PENDING) → [10, PENDING).
        let merged = merge_edge_lifetimes(
            &[EdgeLifetime::new(e(10), None)],
            &[EdgeLifetime::new(e(20), None)],
        );
        assert_eq!(merged, vec![EdgeLifetime::new(e(10), None)]);
    }

    #[test]
    fn test_merge_edge_lifetimes_promote_then_delete_keeps_create() {
        let merged = merge_edge_lifetimes(
            &[EdgeLifetime::new(e(10), None)],
            &[EdgeLifetime::new(e(20), Some(e(30)))],
        );
        assert_eq!(merged, vec![EdgeLifetime::new(e(10), Some(e(30)))]);
    }

    #[test]
    fn test_merge_edge_lifetimes_keeps_pre_promote_closed_lives() {
        let merged = merge_edge_lifetimes(
            &[EdgeLifetime::new(e(10), Some(e(15)))],
            &[EdgeLifetime::new(e(20), None)],
        );
        assert_eq!(
            merged,
            vec![
                EdgeLifetime::new(e(10), Some(e(15))),
                EdgeLifetime::new(e(20), None),
            ]
        );
    }

    #[test]
    fn test_fold_edge_validity_delete_then_recreate_no_resurrection() {
        let versions = [(e(10), Some(e(20))), (e(30), None)];
        let ivs = fold_edge_validity(&versions);
        assert_eq!(ivs.len(), 2);
        assert_eq!(ivs[0], EpochInterval::closed(e(10), e(20)));
        assert_eq!(ivs[1], EpochInterval::open(e(30)));
        assert!(ivs[0].contains(e(15)));
        assert!(ivs[1].contains(e(35)));
        assert!(
            !ivs.iter().any(|iv| iv.contains(e(25))),
            "gap between delete and recreate must not resurrect the edge"
        );
        assert!(
            !ivs[1].contains(e(15)),
            "recreated row must not cover the first life"
        );
    }

    #[test]
    fn test_fold_edge_property_like_history_with_tombstone() {
        // Edge `weight` set, removed, set again — same fold as node properties.
        let history = vec![
            (e(10), Value::Int64(7)),
            (e(20), Value::Null),
            (e(30), Value::Int64(9)),
        ];
        let rows = fold_history(&history);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            (Value::Int64(7), EpochInterval::closed(e(10), e(20)))
        );
        assert_eq!(rows[1].0, Value::Int64(9));
        assert_eq!(rows[1].1.from(), e(30));
        assert!(rows[1].1.is_open());
        assert!(!rows.iter().any(|(_, iv)| iv.contains(e(25))));
    }

    #[test]
    fn test_fold_edge_properties_maps_histories_to_columns() {
        let mut histories = FxHashMap::default();
        histories.insert(
            PropertyKey::new("weight"),
            vec![(e(10), Value::Int64(1)), (e(20), Value::Null)],
        );
        histories.insert(
            PropertyKey::new("since"),
            vec![(e(10), Value::from("2020"))],
        );
        histories.insert(
            PropertyKey::new("mixed"),
            vec![(e(10), Value::Int64(1)), (e(20), Value::from("x"))],
        );
        let cols = fold_edge_properties(histories);
        assert!(cols.contains_key(&PropertyKey::new("weight")));
        assert!(cols.contains_key(&PropertyKey::new("since")));
        assert!(!cols.contains_key(&PropertyKey::new("mixed")));
        let weight = &cols[&PropertyKey::new("weight")];
        assert_eq!(weight.len(), 1);
        assert!(weight.rows_as_of(e(25)).is_empty());
        assert_eq!(weight.value_as_of(0, e(15)), Some(Value::Int64(1)));
    }

    #[test]
    fn test_fold_edge_validity_create_only_is_open() {
        let ivs = fold_edge_validity(&[(e(5), None)]);
        assert_eq!(ivs.len(), 1);
        assert!(ivs[0].is_open());
        assert_eq!(ivs[0].from(), e(5));
        assert!(ivs[0].contains(e(100)));
        assert!(!ivs[0].contains(e(4)));
    }

    #[test]
    fn test_fold_edge_validity_create_delete_retains_closed_row() {
        let ivs = fold_edge_validity(&[(e(10), Some(e(40)))]);
        assert_eq!(ivs.len(), 1);
        assert_eq!(ivs[0], EpochInterval::closed(e(10), e(40)));
        assert!(ivs[0].contains(e(10)));
        assert!(ivs[0].contains(e(39)));
        assert!(!ivs[0].contains(e(40)));
    }

    #[test]
    fn zero_width_rows_and_exact_logs_survive_current_compact_section() {
        use std::sync::Arc;

        use grafeo_common::storage::section::Section;

        use crate::graph::compact::CompactStoreBuilder;
        use crate::graph::compact::section::CompactStoreSection;
        use crate::graph::traits::GraphStore;

        let node = NodeId::new(7);
        let edge = EdgeId::new(9);
        let commit = e(12);
        let property = PropertyKey::new("audit");
        let versions = vec![(commit, Value::Int64(42)), (commit, Value::Null)];
        let labels = vec![
            (commit, vec![ArcStr::from("Draft")]),
            (commit, vec![ArcStr::from("Reviewed")]),
        ];
        let node_history = NodeFullHistory {
            labels: vec![ArcStr::from("Reviewed")],
            label_versions: labels.clone(),
            lifetimes: vec![EdgeLifetime::new(commit, Some(commit))],
            properties: [(property.clone(), versions.clone())].into_iter().collect(),
        };
        let edge_history = EdgeFullHistory {
            src: node,
            dst: node,
            edge_type: ArcStr::from("AUDIT"),
            lifetimes: vec![EdgeLifetime::new(commit, Some(commit))],
            properties: [(property.clone(), versions.clone())].into_iter().collect(),
        };
        assert_eq!(fold_node_rows(node, &node_history).len(), 1);
        assert_eq!(fold_edge_row(edge, &edge_history).len(), 1);
        // The interval-only property fold remains empty: retaining audit rows
        // must never turn an unobservable property value into a visible one.
        assert!(fold_history(&versions).is_empty());

        let compact = CompactStoreBuilder::new()
            .build()
            .unwrap()
            .install_temporal_nodes(|_| node_history.clone(), [node])
            .unwrap()
            .upgrade_rels_temporal(|_| edge_history.clone(), [edge]);
        let original = CompactStoreSection::new(Arc::new(compact));
        let bytes = original.serialize().unwrap();
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();

        for section in [&original, &restored] {
            let store = section.store().unwrap();
            let history = store.temporal_node_history(node).unwrap();
            assert_eq!(history.label_versions, labels);
            assert_eq!(history.properties.get(&property), Some(&versions));
            assert_eq!(history.lifetimes.len(), 1);
            assert_eq!(history.lifetimes[0].created, commit);
            assert_eq!(history.lifetimes[0].deleted, Some(commit));
            assert_eq!(
                store.closed_edge_property_history(edge),
                vec![(property.clone(), versions.clone())]
            );
            assert_eq!(store.node_count(), 0);
            assert_eq!(store.edge_count(), 0);
            assert!(store.get_node(node).is_none());
            assert!(store.get_edge(edge).is_none());
            for epoch in [e(11), commit, e(13), EpochId::PENDING] {
                assert!(store.temporal_node_row_at(node, epoch).is_none());
                assert!(store.retained_edge_row_at(edge, epoch).is_none());
            }
        }
    }

    #[test]
    fn explicit_equal_images_and_adjacent_lifetime_births_survive_compact_section() {
        use std::sync::Arc;

        use grafeo_common::storage::section::Section;

        use crate::graph::compact::CompactStoreBuilder;
        use crate::graph::compact::section::CompactStoreSection;

        let node = NodeId::new(7);
        let initial = vec![ArcStr::from("Initial")];
        let shared = vec![ArcStr::from("Shared")];
        let final_labels = vec![ArcStr::from("Final")];
        let labels = vec![
            (e(2), initial.clone()),
            (e(2), initial),
            (e(4), shared.clone()),
            (e(4), shared),
            (e(6), final_labels.clone()),
        ];
        let source = NodeFullHistory {
            labels: final_labels,
            label_versions: labels.clone(),
            lifetimes: vec![
                EdgeLifetime::new(e(2), Some(e(4))),
                EdgeLifetime::new(e(4), None),
            ],
            properties: FxHashMap::default(),
        };
        let rows = fold_node_rows(node, &source);
        assert_eq!(rows[0].label_versions, labels[..3]);
        assert_eq!(rows[1].label_versions, labels[3..]);
        let compact = CompactStoreBuilder::new()
            .build()
            .unwrap()
            .install_temporal_nodes(|_| source.clone(), [node])
            .unwrap();
        let original = CompactStoreSection::new(Arc::new(compact));
        let bytes = original.serialize().unwrap();
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();
        for section in [&original, &restored] {
            let history = section
                .store()
                .unwrap()
                .temporal_node_history(node)
                .unwrap();
            assert_eq!(history.label_versions, labels);
            assert_eq!(history.lifetimes, source.lifetimes);
        }
    }

    #[test]
    fn zero_width_label_clip_retains_only_ordered_boundary_states() {
        let labels = |names: &[&str]| -> Vec<ArcStr> {
            names.iter().map(|name| ArcStr::from(*name)).collect()
        };
        let history = vec![
            (e(3), labels(&["Before"])),
            (e(5), labels(&["B", "A", "A"])),
            (e(5), labels(&["A", "B"])),
            (e(5), labels(&["C"])),
            (e(7), labels(&["After"])),
        ];
        assert_eq!(
            clip_label_history_to_interval(&history, &[], EpochInterval::closed(e(5), e(5))),
            vec![
                (e(5), labels(&["A", "B"])),
                (e(5), labels(&["A", "B"])),
                (e(5), labels(&["C"]))
            ]
        );
        // Positive-width clipping retains the same explicit birth sequence.
        assert_eq!(
            clip_label_history_to_interval(&history, &[], EpochInterval::closed(e(5), e(6))),
            vec![
                (e(5), labels(&["A", "B"])),
                (e(5), labels(&["A", "B"])),
                (e(5), labels(&["C"]))
            ]
        );
        assert_eq!(
            clip_label_history_to_interval(
                &[],
                &labels(&["Only"]),
                EpochInterval::closed(e(5), e(5))
            ),
            vec![(e(5), labels(&["Only"]))]
        );
    }

    #[test]
    fn test_fold_edge_validity_excludes_uncommitted_create() {
        assert!(fold_edge_validity(&[(EpochId::PENDING, None)]).is_empty());
        let ivs = fold_edge_validity(&[(e(10), Some(EpochId::PENDING))]);
        assert_eq!(ivs.len(), 1);
        assert!(ivs[0].is_open());
    }

    #[test]
    fn test_clip_history_does_not_manufacture_structural_tombstone() {
        let history = vec![(e(10), Value::Int64(1)), (e(20), Value::Int64(2))];
        let clipped = clip_history_to_interval(&history, EpochInterval::closed(e(10), e(30)));
        assert_eq!(clipped, history);
        let rows = fold_history(&clipped);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].0, Value::Int64(2));
        assert!(
            rows[1].1.is_open(),
            "the enclosing structural row, not a fabricated property event, closes visibility"
        );
    }

    #[test]
    fn test_fold_edge_row_property_change_then_delete() {
        let mut history = EdgeFullHistory {
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: ArcStr::from("KNOWS"),
            lifetimes: vec![EdgeLifetime::new(e(10), Some(e(30)))],
            properties: FxHashMap::default(),
        };
        history.properties.insert(
            PropertyKey::new("w"),
            vec![(e(10), Value::Int64(1)), (e(20), Value::Int64(2))],
        );
        let rows = fold_edge_row(EdgeId::new(7), &history);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].validity, EpochInterval::closed(e(10), e(30)));
        let col = &rows[0].properties[&PropertyKey::new("w")];
        assert_eq!(
            col.value_in_range_as_of(0, col.len(), e(15)),
            Some(Value::Int64(1))
        );
        assert_eq!(
            col.value_in_range_as_of(0, col.len(), e(25)),
            Some(Value::Int64(2))
        );
        assert!(
            !rows[0].validity.contains(e(35)),
            "the enclosing structural row excludes reads after edge deletion"
        );
    }

    #[test]
    fn test_committed_edge_history_drops_pending() {
        let mut history = EdgeFullHistory {
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: ArcStr::from("KNOWS"),
            lifetimes: vec![
                EdgeLifetime::new(e(10), None),
                EdgeLifetime::new(EpochId::PENDING, None),
            ],
            properties: FxHashMap::default(),
        };
        history.properties.insert(
            PropertyKey::new("w"),
            vec![
                (e(10), Value::Int64(1)),
                (EpochId::PENDING, Value::Int64(99)),
            ],
        );
        let committed = committed_edge_history(history, e(10));
        assert_eq!(committed.lifetimes.len(), 1);
        assert_eq!(committed.lifetimes[0].created, e(10));
        assert_eq!(
            committed.properties[&PropertyKey::new("w")],
            vec![(e(10), Value::Int64(1))]
        );
    }

    #[test]
    fn test_build_current_csr_from_open_edges_drops_closed() {
        let rows = [
            TemporalEdgeRow {
                src: 0,
                dst: 1,
                validity: EpochInterval::open(e(10)),
                edge_id: EdgeId::new(1),
            },
            TemporalEdgeRow {
                src: 0,
                dst: 2,
                validity: EpochInterval::closed(e(10), e(20)),
                edge_id: EdgeId::new(2),
            },
        ];
        let (packed, current) = pack_and_derive_current(3, &rows);
        assert_eq!(current.neighbors(0), &[1]);
        assert_eq!(current.num_edges(), 1);
        assert_eq!(packed.num_versions(), 2);
        assert_eq!(
            packed.neighbors_at_epoch(0, e(15)),
            vec![1, 2],
            "closed life remains addressable for as-of"
        );
        assert_eq!(
            build_current_csr_from_open_edges(3, &rows).neighbors(0),
            &[1]
        );
    }
}
