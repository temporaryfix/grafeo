//! Temporal columnar block: values paired with per-row epoch validity.
//!
//! A [`crate::graph::compact::temporal_column::TemporalColumn`] pairs a value [`crate::graph::compact::column::ColumnCodec`] (the existing codec'd value
//! storage) with a parallel column of [`EpochInterval`]s — one per row — and an
//! [`grafeo_common::types::EpochZoneMap`] summarising the block's epoch coverage. As-of reads are
//! block-pruned (skip blocks the zone-map excludes) then gated per row by the
//! interval, with no delta replay. This is the SP1 cold-base block's in-memory
//! shape; on-disk serialization + the physical Merkle hash arrive in SP1-5.

use grafeo_common::types::{EpochId, EpochInterval, EpochZoneMap, Value};

use super::column::ColumnCodec;

/// Per-row intervals, collapsed when every row shares one window.
#[derive(Debug, Clone)]
enum ValidityStore {
    /// `[INITIAL, PENDING)` on every row (fresh all-open column).
    AllOpen,
    /// The same interval on every row.
    Uniform(EpochInterval),
    /// Packed `u32` epoch pairs (`PENDING` = `u32::MAX`).
    Rows { from: Vec<u32>, to: Vec<u32> },
}

impl ValidityStore {
    fn from_intervals(ivs: &[EpochInterval]) -> Self {
        if ivs.is_empty()
            || ivs
                .iter()
                .all(|iv| iv.is_open() && iv.from() == EpochId::INITIAL)
        {
            return Self::AllOpen;
        }
        if ivs.windows(2).all(|w| w[0] == w[1]) {
            return Self::Uniform(ivs[0]);
        }
        let mut from = Vec::with_capacity(ivs.len());
        let mut to = Vec::with_capacity(ivs.len());
        for iv in ivs {
            from.push(super::csr::pack_epoch(iv.from()));
            to.push(super::csr::pack_epoch(iv.to()));
        }
        Self::Rows { from, to }
    }

    fn get(&self, row: usize, n: usize) -> Option<EpochInterval> {
        if row >= n {
            return None;
        }
        Some(match self {
            Self::AllOpen => EpochInterval::open(EpochId::INITIAL),
            Self::Uniform(iv) => *iv,
            Self::Rows { from, to } => super::csr::unpack_interval(*from.get(row)?, *to.get(row)?),
        })
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::AllOpen => 0,
            Self::Uniform(_) => std::mem::size_of::<EpochInterval>(),
            Self::Rows { from, to } => from.len() * 4 + to.len() * 4,
        }
    }

    fn expand(&self, n: usize) -> Vec<EpochInterval> {
        match self {
            Self::AllOpen => vec![EpochInterval::open(EpochId::INITIAL); n],
            Self::Uniform(iv) => vec![*iv; n],
            Self::Rows { from, to } => from
                .iter()
                .zip(to.iter())
                .map(|(&f, &t)| super::csr::unpack_interval(f, t))
                .collect(),
        }
    }
}

/// A block of values with per-row epoch validity, for the temporal cold base.
#[derive(Debug, Clone)]
pub struct TemporalColumn {
    values: ColumnCodec,
    validity: ValidityStore,
    epoch_zone: EpochZoneMap,
}

impl TemporalColumn {
    /// Builds a temporal column from a value codec and a parallel validity column.
    ///
    /// `validity` must have exactly one interval per value row.
    #[must_use]
    pub fn new(values: ColumnCodec, validity: Vec<EpochInterval>) -> Self {
        debug_assert_eq!(
            validity.len(),
            values.len(),
            "validity column must have one interval per value row"
        );
        let epoch_zone = EpochZoneMap::from_intervals(validity.iter().copied());
        Self {
            values,
            validity: ValidityStore::from_intervals(&validity),
            epoch_zone,
        }
    }

    /// Number of rows (value + interval pairs).
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the block has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The block-level epoch coverage summary.
    #[must_use]
    pub fn epoch_zone(&self) -> EpochZoneMap {
        self.epoch_zone
    }

    /// Whether the block may hold any value valid at `epoch` (block-level prune;
    /// never a false negative).
    #[must_use]
    pub fn may_contain(&self, epoch: EpochId) -> bool {
        self.epoch_zone.may_contain(epoch)
    }

    /// The value at `row` if its validity interval contains `epoch`, else `None`.
    ///
    /// `epoch == PENDING` is the "latest" sentinel: a half-open interval can never
    /// *contain* `PENDING` (`to` is exclusive and an open interval's `to` IS
    /// `PENDING`), so at the latest epoch a row is valid iff its interval is still
    /// open (current). Mirrors the as-of node accessors.
    #[must_use]
    pub fn value_as_of(&self, row: usize, epoch: EpochId) -> Option<Value> {
        let iv = self.validity.get(row, self.values.len())?;
        let valid = if epoch == EpochId::PENDING {
            iv.is_open()
        } else {
            iv.contains(epoch)
        };
        if valid { self.values.get(row) } else { None }
    }

    /// All `(row, value)` pairs valid at `epoch` — block-pruned, then gated per row.
    #[must_use]
    pub fn rows_as_of(&self, epoch: EpochId) -> Vec<(usize, Value)> {
        // PENDING = "latest": match still-open intervals (the zone map's
        // `may_contain` excludes the PENDING sentinel, so don't gate on it).
        let n = self.values.len();
        if epoch == EpochId::PENDING {
            return (0..n)
                .filter(|&r| self.validity.get(r, n).is_some_and(|iv| iv.is_open()))
                .filter_map(|r| self.values.get(r).map(|v| (r, v)))
                .collect();
        }
        if !self.may_contain(epoch) {
            return Vec::new();
        }
        (0..n)
            .filter(|&r| self.validity.get(r, n).is_some_and(|iv| iv.contains(epoch)))
            .filter_map(|r| self.values.get(r).map(|v| (r, v)))
            .collect()
    }

    /// Builds an all-open temporal column from current values: every row gets
    /// the open interval `[INITIAL, PENDING)`.
    ///
    /// This is the SP1-5 shape for a freshly-built cold base with no recorded
    /// history — `as_of(any epoch)` returns the current value, preserving
    /// current-read semantics (all-open == current).
    #[must_use]
    pub fn all_open(values: ColumnCodec) -> Self {
        Self {
            values,
            validity: ValidityStore::AllOpen,
            epoch_zone: EpochZoneMap::from_intervals([EpochInterval::open(EpochId::INITIAL)]),
        }
    }

    /// The underlying value codec (the current-value projection).
    ///
    /// Used by the current-read scan path: in an all-open base, physical rows
    /// are 1:1 with nodes, so scans over this codec return node offsets directly.
    #[must_use]
    pub fn values(&self) -> &ColumnCodec {
        &self.values
    }

    /// The physical row of the open (still-current) interval within the
    /// half-open physical row range `[start, start + count)`, if any.
    ///
    /// A node's versions are contiguous and at most one is open (the current
    /// value); a fully-retracted node has none.
    #[must_use]
    pub(super) fn open_row_in(&self, start: usize, count: usize) -> Option<usize> {
        (start..start.saturating_add(count)).find(|&r| {
            self.validity
                .get(r, self.values.len())
                .is_some_and(|iv| iv.is_open())
        })
    }

    /// The current value of the node occupying physical rows
    /// `[start, start + count)` — the value of its open interval — or `None`
    /// if the node is fully retracted (no open row) or the range is out of bounds.
    #[must_use]
    pub fn current_value(&self, start: usize, count: usize) -> Option<Value> {
        self.values.get(self.open_row_in(start, count)?)
    }

    /// Raw `u64` of the node's current (open) row for a bit-packed column;
    /// `None` for non-bit-packed columns, a retracted node, or an out-of-bounds
    /// range. Mirrors [`current_value`](Self::current_value) for FK columns.
    #[must_use]
    pub fn current_raw_u64(&self, start: usize, count: usize) -> Option<u64> {
        self.values.get_raw_u64(self.open_row_in(start, count)?)
    }

    /// The value valid at `epoch` for the node occupying physical rows
    /// `[start, start + count)` — the value of the version whose interval
    /// contains `epoch` — or `None` if no version covers it (a gap, before the
    /// node's first version, or fully retracted) or the range is out of bounds.
    ///
    /// The block-level [`EpochZoneMap`]
    /// prunes the whole column first (never a false negative); the per-row
    /// validity then gates the scan over the node's contiguous run.
    #[must_use]
    pub fn value_in_range_as_of(
        &self,
        start: usize,
        count: usize,
        epoch: EpochId,
    ) -> Option<Value> {
        // PENDING = "latest": the node's open (still-current) version. Bypass the
        // zone-map prune, which excludes the PENDING sentinel.
        if epoch == EpochId::PENDING {
            return self.current_value(start, count);
        }
        if !self.may_contain(epoch) {
            return None;
        }
        (start..start.saturating_add(count)).find_map(|r| self.value_as_of(r, epoch))
    }

    /// Estimated heap bytes: the value codec plus the per-row validity column.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.values.heap_bytes() + self.validity.heap_bytes()
    }

    /// Whether every row is exactly the all-open interval `[INITIAL, PENDING)` —
    /// i.e. this column was built from current values with no recorded history
    /// and can be reconstructed by [`all_open`](Self::all_open). Drives the
    /// compact `all-open` serialization path (no validity bytes stored).
    #[must_use]
    pub fn is_all_open(&self) -> bool {
        matches!(self.validity, ValidityStore::AllOpen)
    }

    /// The per-row epoch validity intervals (for serialization).
    #[must_use]
    pub fn validity(&self) -> Vec<EpochInterval> {
        self.validity.expand(self.values.len())
    }

    /// Reconstructs the `(epoch, value)` history of the node occupying physical
    /// rows `[start, start + count)` — the inverse of folding it. Each run emits
    /// `(from, value)`; a gap between consecutive runs (a removal) emits a `Null`
    /// tombstone at the prior run's end, as does a trailing closed run. Re-folding
    /// the result via `fold_history` yields the same runs. Used by the SP2 merge
    /// to pull a base node's history back out of the cold columns.
    #[must_use]
    pub fn runs_as_history(&self, start: usize, count: usize) -> Vec<(EpochId, Value)> {
        let n = self.values.len();
        let end = start.saturating_add(count).min(n);
        let mut history = Vec::new();
        let mut prev_to: Option<EpochId> = None;
        for r in start..end {
            let Some(iv) = self.validity.get(r, n) else {
                continue;
            };
            let Some(value) = self.values.get(r) else {
                continue;
            };
            if let Some(pt) = prev_to
                && pt < iv.from()
            {
                history.push((pt, Value::Null));
            }
            history.push((iv.from(), value));
            prev_to = if iv.is_open() { None } else { Some(iv.to()) };
        }
        if let Some(pt) = prev_to {
            history.push((pt, Value::Null));
        }
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(n: u64) -> EpochId {
        EpochId::new(n)
    }

    fn sample() -> TemporalColumn {
        // three rows, three validity windows
        let values = ColumnCodec::raw_i64(vec![100, 200, 300]);
        let validity = vec![
            EpochInterval::closed(e(10), e(20)),
            EpochInterval::closed(e(20), e(30)),
            EpochInterval::open(e(30)),
        ];
        TemporalColumn::new(values, validity)
    }

    #[test]
    fn test_value_as_of_gated_by_interval() {
        let tc = sample();
        assert!(tc.value_as_of(0, e(15)).is_some()); // [10,20) contains 15
        assert!(tc.value_as_of(0, e(20)).is_none()); // exclusive upper bound
        assert!(tc.value_as_of(0, e(25)).is_none()); // outside row 0's window
        assert!(tc.value_as_of(2, e(1_000)).is_some()); // open tail
        assert!(tc.value_as_of(9, e(15)).is_none()); // out-of-range row
    }

    #[test]
    fn test_rows_as_of_block_pruned_then_gated() {
        let tc = sample();
        let rows: Vec<usize> = tc.rows_as_of(e(25)).into_iter().map(|(r, _)| r).collect();
        assert_eq!(rows, vec![1]); // only row 1's [20,30) contains 25
        assert!(tc.rows_as_of(e(5)).is_empty()); // before all windows (block pruned)
    }

    #[test]
    fn test_block_pruning() {
        let tc = sample();
        assert!(!tc.may_contain(e(5))); // below min_from=10
        assert!(tc.may_contain(e(15)));
        assert!(tc.may_contain(e(1_000_000))); // open tail -> unbounded above
        assert_eq!(tc.epoch_zone().min_from(), e(10));
        assert_eq!(tc.epoch_zone().max_to(), EpochId::PENDING);
    }

    #[test]
    fn test_len() {
        assert_eq!(sample().len(), 3);
        assert!(!sample().is_empty());
    }

    #[test]
    fn test_all_open_from_current_values_is_current_everywhere() {
        // A base built from current values has every row open [INITIAL, PENDING):
        // as_of at any epoch yields the current value (current-read preserved),
        // and values() exposes the underlying codec for the scan path.
        let tc = TemporalColumn::all_open(ColumnCodec::raw_i64(vec![10, 20, 30]));
        assert_eq!(tc.len(), 3);
        for r in 0..3 {
            assert!(tc.value_as_of(r, e(0)).is_some(), "row {r} open at epoch 0");
            assert!(
                tc.value_as_of(r, e(1_000_000)).is_some(),
                "row {r} open at large epoch"
            );
        }
        assert_eq!(tc.values().get(1), Some(Value::Int64(20)));
        assert_eq!(tc.values().len(), 3);
    }

    #[test]
    fn test_current_value_reads_open_row_in_range() {
        let tc = TemporalColumn::all_open(ColumnCodec::raw_i64(vec![10, 20, 30]));
        // All-open identity layout: node i occupies physical row i (count 1).
        assert_eq!(tc.current_value(0, 1), Some(Value::Int64(10)));
        assert_eq!(tc.current_value(2, 1), Some(Value::Int64(30)));
        assert_eq!(tc.current_value(3, 1), None); // out of range
    }

    #[test]
    fn test_value_in_range_as_of_picks_covering_row() {
        // A node occupying physical rows [0,3): 100@[10,20), 200@[20,30),
        // 300@[30,PENDING). As-of scans the range for the covering interval.
        let values = ColumnCodec::raw_i64(vec![100, 200, 300]);
        let validity = vec![
            EpochInterval::closed(e(10), e(20)),
            EpochInterval::closed(e(20), e(30)),
            EpochInterval::open(e(30)),
        ];
        let tc = TemporalColumn::new(values, validity);
        assert_eq!(
            tc.value_in_range_as_of(0, 3, e(15)),
            Some(Value::Int64(100))
        );
        assert_eq!(
            tc.value_in_range_as_of(0, 3, e(25)),
            Some(Value::Int64(200))
        );
        assert_eq!(
            tc.value_in_range_as_of(0, 3, e(999)),
            Some(Value::Int64(300))
        );
        assert_eq!(tc.value_in_range_as_of(0, 3, e(5)), None); // before creation
    }

    #[test]
    fn test_value_in_range_as_of_all_open_is_current_everywhere() {
        let tc = TemporalColumn::all_open(ColumnCodec::raw_i64(vec![10, 20, 30]));
        // All-open identity: node i -> row i; as-of at any real epoch == current.
        for ep in [e(0), e(1), e(1_000_000)] {
            assert_eq!(tc.value_in_range_as_of(1, 1, ep), Some(Value::Int64(20)));
        }
    }

    #[test]
    fn test_runs_as_history_round_trips_through_fold() {
        use super::super::compaction::fold_history;
        // node 0 spans rows [0,3): 100@[10,20), 200@[20,25) then removed, 300@[30,PENDING).
        let values = ColumnCodec::raw_i64(vec![100, 200, 300]);
        let validity = vec![
            EpochInterval::closed(e(10), e(20)),
            EpochInterval::closed(e(20), e(25)),
            EpochInterval::open(e(30)),
        ];
        let tc = TemporalColumn::new(values, validity);
        let hist = tc.runs_as_history(0, 3);
        assert_eq!(
            hist,
            vec![
                (e(10), Value::Int64(100)),
                (e(20), Value::Int64(200)),
                (e(25), Value::Null), // removal tombstone (gap [25,30))
                (e(30), Value::Int64(300)),
            ]
        );
        // Re-folding the reconstructed history reproduces the original runs.
        let refolded = fold_history(&hist);
        assert_eq!(refolded.len(), 3);
        assert_eq!(refolded[0].1, EpochInterval::closed(e(10), e(20)));
        assert_eq!(refolded[2].1, EpochInterval::open(e(30)));
    }

    #[test]
    fn test_runs_as_history_all_open_single_version() {
        let tc = TemporalColumn::all_open(ColumnCodec::raw_i64(vec![42, 99]));
        // All-open identity: node i -> row i, a single [INITIAL, PENDING) version.
        assert_eq!(
            tc.runs_as_history(0, 1),
            vec![(EpochId::INITIAL, Value::Int64(42))]
        );
        assert_eq!(
            tc.runs_as_history(1, 1),
            vec![(EpochId::INITIAL, Value::Int64(99))]
        );
    }

    #[test]
    fn test_is_all_open_and_validity_accessor() {
        // Built from current values: every interval is exactly open(INITIAL).
        let open = TemporalColumn::all_open(ColumnCodec::raw_i64(vec![1, 2]));
        assert!(open.is_all_open());
        assert_eq!(open.validity().len(), 2);
        assert!(
            open.validity()
                .iter()
                .all(|iv| iv.is_open() && iv.from() == EpochId::INITIAL)
        );

        // A real closed interval (or an open one that doesn't start at INITIAL)
        // is NOT all-open and must serialize via the explicit v4 path.
        let temporal = TemporalColumn::new(
            ColumnCodec::raw_i64(vec![10, 20]),
            vec![EpochInterval::closed(e(5), e(9)), EpochInterval::open(e(9))],
        );
        assert!(!temporal.is_all_open());
    }

    #[test]
    fn test_validity_collapses_uniform_and_all_open() {
        let open = TemporalColumn::all_open(ColumnCodec::raw_i64(vec![1, 2, 3, 4]));
        assert_eq!(
            open.heap_bytes(),
            ColumnCodec::raw_i64(vec![1, 2, 3, 4]).heap_bytes()
        );

        let uniform = TemporalColumn::new(
            ColumnCodec::raw_i64(vec![1, 2, 3, 4]),
            vec![EpochInterval::open(e(10)); 4],
        );
        assert!(!uniform.is_all_open());
        assert_eq!(uniform.validity().len(), 4);
        assert!(
            uniform.heap_bytes()
                < TemporalColumn::new(
                    ColumnCodec::raw_i64(vec![1, 2, 3, 4]),
                    vec![
                        EpochInterval::closed(e(10), e(20)),
                        EpochInterval::closed(e(20), e(30)),
                        EpochInterval::closed(e(30), e(40)),
                        EpochInterval::open(e(40)),
                    ],
                )
                .heap_bytes()
        );
    }
}
