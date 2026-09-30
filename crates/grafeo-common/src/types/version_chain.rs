//! Epoch-interval version chains for the temporal cold base (SP1).
//!
//! [`EpochInterval`] is the half-open `[from, to)` epoch validity window the cold
//! base stamps on each value (the SP1 time-encoding). [`VersionChain`] is an
//! epoch-ordered sequence of values, each holding the interval over which it was
//! current, supporting `as_of(epoch)` point reads without delta replay.
//!
//! Epochs here are MVCC commit epochs ([`EpochId`]) — the *version* time axis used
//! for as-of-epoch scrub. (Distinct from [`ValidityTs`](crate::types::ValidityTs),
//! the reverse-ordered disk-key wall-clock timestamp, and from application time.)
//!
//! `VersionChain::as_of` is an O(n) scan at this layer (SP1-1); the columnar block
//! + epoch zone-map (SP1-2/3) add block-level pruning for scale.

use serde::{Deserialize, Serialize};

use crate::types::EpochId;

/// A half-open epoch validity window `[from, to)`.
///
/// `to == EpochId::PENDING` means the interval is still open — the value is
/// current. The window is half-open: `from` is included, `to` is excluded.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct EpochInterval {
    from: EpochId,
    to: EpochId,
}

impl EpochInterval {
    /// An open interval starting at `from` (current; `to = PENDING`).
    #[must_use]
    pub const fn open(from: EpochId) -> Self {
        Self {
            from,
            to: EpochId::PENDING,
        }
    }

    /// A closed interval `[from, to)`.
    #[must_use]
    pub const fn closed(from: EpochId, to: EpochId) -> Self {
        Self { from, to }
    }

    /// The start epoch (inclusive).
    #[must_use]
    pub const fn from(&self) -> EpochId {
        self.from
    }

    /// The end epoch (exclusive); `EpochId::PENDING` if still open.
    #[must_use]
    pub const fn to(&self) -> EpochId {
        self.to
    }

    /// Whether the interval is still open (the value is current).
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.to == EpochId::PENDING
    }

    /// Whether `epoch` falls within `[from, to)`.
    #[must_use]
    pub fn contains(&self, epoch: EpochId) -> bool {
        self.from <= epoch && epoch < self.to
    }
}

/// An epoch-ordered chain of values, each valid over an [`EpochInterval`].
///
/// [`append`](VersionChain::append)-ing a new value closes the previous open
/// interval at the new epoch and opens a fresh one. [`as_of`](VersionChain::as_of)
/// returns the value whose interval contains an epoch.
#[derive(Clone, Debug, Default)]
pub struct VersionChain<V> {
    versions: Vec<(V, EpochInterval)>,
}

impl<V> VersionChain<V> {
    /// Creates an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Appends a new current value effective at `epoch`, closing the prior open
    /// interval at `epoch`. Versions must be appended in non-decreasing epoch order.
    pub fn append(&mut self, value: V, epoch: EpochId) {
        if let Some((_, last)) = self.versions.last_mut()
            && last.is_open()
        {
            debug_assert!(
                last.from <= epoch,
                "versions must be appended in non-decreasing epoch order"
            );
            *last = EpochInterval::closed(last.from, epoch);
        }
        self.versions.push((value, EpochInterval::open(epoch)));
    }

    /// Closes the current open interval at `epoch` without opening a new one — a
    /// retraction. Nothing is current at or after `epoch` until the next `append`.
    pub fn close_current(&mut self, epoch: EpochId) {
        if let Some((_, last)) = self.versions.last_mut()
            && last.is_open()
        {
            debug_assert!(last.from <= epoch);
            *last = EpochInterval::closed(last.from, epoch);
        }
    }

    /// Returns the value whose interval contains `epoch`, if any.
    #[must_use]
    pub fn as_of(&self, epoch: EpochId) -> Option<&V> {
        self.versions
            .iter()
            .find(|(_, iv)| iv.contains(epoch))
            .map(|(v, _)| v)
    }

    /// Returns the current value (the latest version whose interval is still open),
    /// or `None` if the chain is empty or the last value was retracted.
    #[must_use]
    pub fn latest(&self) -> Option<&V> {
        self.versions
            .last()
            .filter(|(_, iv)| iv.is_open())
            .map(|(v, _)| v)
    }

    /// Number of versions in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Whether the chain has no versions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

/// Per-block epoch coverage summary for as-of pruning.
///
/// Tracks the smallest `from` and largest `to` across a block's [`EpochInterval`]s.
/// A conservative filter: [`may_contain`](EpochZoneMap::may_contain) returning
/// `false` guarantees no value in the block is valid at that epoch (safe to skip);
/// `true` means an in-block scan is needed. The epoch-axis analogue of the value
/// zone-maps the columnar store already keeps.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct EpochZoneMap {
    min_from: EpochId,
    max_to: EpochId,
}

impl EpochZoneMap {
    /// An empty zone-map that matches no epoch; fold intervals in via
    /// [`include`](EpochZoneMap::include).
    pub const EMPTY: Self = Self {
        min_from: EpochId::PENDING,
        max_to: EpochId::INITIAL,
    };

    /// Builds a zone-map covering all the given intervals.
    #[must_use]
    pub fn from_intervals(intervals: impl IntoIterator<Item = EpochInterval>) -> Self {
        let mut z = Self::EMPTY;
        for iv in intervals {
            z.include(iv);
        }
        z
    }

    /// Widens the zone-map to cover `interval`.
    pub fn include(&mut self, interval: EpochInterval) {
        self.min_from = self.min_from.min(interval.from());
        self.max_to = self.max_to.max(interval.to());
    }

    /// Whether the block *may* hold a value valid at `epoch` (conservative — never a
    /// false negative).
    #[must_use]
    pub fn may_contain(&self, epoch: EpochId) -> bool {
        self.min_from <= epoch && epoch < self.max_to
    }

    /// Whether the block's coverage overlaps the half-open epoch range `[from, to)`.
    #[must_use]
    pub fn overlaps_range(&self, from: EpochId, to: EpochId) -> bool {
        self.min_from < to && from < self.max_to
    }

    /// The smallest interval start covered.
    #[must_use]
    pub const fn min_from(&self) -> EpochId {
        self.min_from
    }

    /// The largest interval end covered (`EpochId::PENDING` if any interval is open).
    #[must_use]
    pub const fn max_to(&self) -> EpochId {
        self.max_to
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(n: u64) -> EpochId {
        EpochId::new(n)
    }

    #[test]
    fn test_epoch_interval_contains_half_open() {
        let iv = EpochInterval::closed(e(10), e(20));
        assert!(!iv.contains(e(9)));
        assert!(iv.contains(e(10))); // from inclusive
        assert!(iv.contains(e(19)));
        assert!(!iv.contains(e(20))); // to exclusive
        assert!(!iv.is_open());
        assert_eq!(iv.from(), e(10));
        assert_eq!(iv.to(), e(20));
    }

    #[test]
    fn test_epoch_interval_open_contains_from_onward() {
        let iv = EpochInterval::open(e(5));
        assert!(iv.is_open());
        assert!(!iv.contains(e(4)));
        assert!(iv.contains(e(5)));
        assert!(iv.contains(e(1_000_000)));
    }

    #[test]
    fn test_version_chain_as_of_picks_interval() {
        let mut vc = VersionChain::new();
        vc.append("a", e(10));
        vc.append("b", e(20));
        vc.append("c", e(30));
        assert_eq!(vc.as_of(e(9)), None); // before first
        assert_eq!(vc.as_of(e(10)), Some(&"a"));
        assert_eq!(vc.as_of(e(19)), Some(&"a"));
        assert_eq!(vc.as_of(e(20)), Some(&"b")); // boundary -> new version
        assert_eq!(vc.as_of(e(25)), Some(&"b"));
        assert_eq!(vc.as_of(e(30)), Some(&"c"));
        assert_eq!(vc.as_of(e(999)), Some(&"c")); // open tail
        assert_eq!(vc.latest(), Some(&"c"));
        assert_eq!(vc.len(), 3);
    }

    #[test]
    fn test_version_chain_retract_closes_current() {
        let mut vc = VersionChain::new();
        vc.append("x", e(10));
        vc.close_current(e(25)); // retract at 25
        assert_eq!(vc.as_of(e(10)), Some(&"x"));
        assert_eq!(vc.as_of(e(24)), Some(&"x"));
        assert_eq!(vc.as_of(e(25)), None); // retracted
        assert_eq!(vc.as_of(e(100)), None);
        assert_eq!(vc.latest(), None); // nothing current
    }

    #[test]
    fn test_version_chain_empty() {
        let vc: VersionChain<i32> = VersionChain::new();
        assert!(vc.is_empty());
        assert_eq!(vc.as_of(e(1)), None);
        assert_eq!(vc.latest(), None);
    }

    #[test]
    fn test_epoch_zone_map_empty_matches_nothing() {
        let z = EpochZoneMap::EMPTY;
        assert!(!z.may_contain(e(0)));
        assert!(!z.may_contain(e(100)));
    }

    #[test]
    fn test_epoch_zone_map_closed_coverage() {
        let z = EpochZoneMap::from_intervals([
            EpochInterval::closed(e(10), e(20)),
            EpochInterval::closed(e(20), e(30)),
        ]);
        assert_eq!(z.min_from(), e(10));
        assert_eq!(z.max_to(), e(30));
        assert!(!z.may_contain(e(5)));
        assert!(z.may_contain(e(10)));
        assert!(z.may_contain(e(29)));
        assert!(!z.may_contain(e(30))); // exclusive upper bound
    }

    #[test]
    fn test_epoch_zone_map_open_is_unbounded_above() {
        let z = EpochZoneMap::from_intervals([
            EpochInterval::closed(e(10), e(20)),
            EpochInterval::open(e(20)),
        ]);
        assert_eq!(z.max_to(), EpochId::PENDING);
        assert!(z.may_contain(e(10)));
        assert!(z.may_contain(e(1_000_000)));
        assert!(!z.may_contain(e(5)));
    }

    #[test]
    fn test_epoch_zone_map_overlaps_range() {
        let z = EpochZoneMap::from_intervals([EpochInterval::closed(e(10), e(30))]);
        assert!(z.overlaps_range(e(25), e(40)));
        assert!(z.overlaps_range(e(0), e(15)));
        assert!(!z.overlaps_range(e(0), e(10))); // adjacent below
        assert!(!z.overlaps_range(e(30), e(40))); // adjacent above
    }
}
