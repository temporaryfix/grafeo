//! Counted work, for tests that assert cost instead of results.
//!
//! Every fixture in this repository is small, so a query path that does work
//! proportional to the database instead of to the result passes every
//! result-equality test. That is how an indexed point lookup came to materialise
//! its label's entire id set per query — 30.6 ms at a million labelled nodes —
//! while 4,700 tests stayed green.
//!
//! These counters make that visible at fixture size. A test runs one query
//! shape against two database sizes and asserts the counted work is **equal**:
//! deterministic, machine-independent, and unaffected by CI load, which a
//! wall-clock threshold is not.
//!
//! **Why not PROFILE.** Enabling profiling changes the plan — it disables
//! factorized filter, project, top-k and join, and expand-chain fusion — so
//! anything measured under it is not the production plan's cost. PROFILE stays a
//! human diagnostic; these counters are the machine-checkable one.
//!
//! **What is counted, and what deliberately is not.** The original counters
//! covered only scan primitives: calls that build a `Vec` of ids whose length
//! is a property of the data rather than of the result. Property-index probes
//! now expose their bounded internal work separately: ordered routing keys,
//! posting identities, posting intervals, and detached rebuild source rows.
//! Node materialization and direct visibility probes expose hidden work in ID seeks.
//! These are diagnostic relaxed adds, not query admission or synchronization.
//!
//! Counters live on the store instance, not in a process global, so tests cannot
//! interfere with each other under any test runner.

use std::sync::atomic::{AtomicU64, Ordering};

/// The two scan primitives, kept apart because a point lookup must perform
/// neither, and "no label scan" is a different assertion from "not much work".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkSnapshot {
    /// Calls to `nodes_by_label` / `nodes_by_label_visible`.
    pub label_scan_calls: u64,
    /// Ids those calls returned.
    pub label_scan_ids: u64,
    /// Calls to `all_node_ids`.
    pub full_scan_calls: u64,
    /// Ids those calls returned.
    pub full_scan_ids: u64,
    /// Ordered routing keys inspected by property-index probes.
    pub property_index_route_keys: u64,
    /// Posting identities visited by property-index probes.
    pub property_index_posting_ids: u64,
    /// Historical posting intervals inspected by property-index probes.
    pub property_index_posting_intervals: u64,
    /// Whole-node label/property maps materialized by store accessors.
    pub node_materializations: u64,
    /// Direct epoch visibility probes that load no labels or properties.
    pub node_visibility_checks: u64,
    /// Per-identity rows read while rebuilding a property index.
    pub property_index_rebuild_rows: u64,
}

impl WorkSnapshot {
    /// Work performed between two snapshots.
    ///
    /// Saturating, so a counter that wrapped cannot produce a nonsense budget.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            label_scan_calls: self
                .label_scan_calls
                .saturating_sub(earlier.label_scan_calls),
            label_scan_ids: self.label_scan_ids.saturating_sub(earlier.label_scan_ids),
            full_scan_calls: self.full_scan_calls.saturating_sub(earlier.full_scan_calls),
            full_scan_ids: self.full_scan_ids.saturating_sub(earlier.full_scan_ids),
            property_index_route_keys: self
                .property_index_route_keys
                .saturating_sub(earlier.property_index_route_keys),
            property_index_posting_ids: self
                .property_index_posting_ids
                .saturating_sub(earlier.property_index_posting_ids),
            property_index_posting_intervals: self
                .property_index_posting_intervals
                .saturating_sub(earlier.property_index_posting_intervals),
            node_materializations: self
                .node_materializations
                .saturating_sub(earlier.node_materializations),
            node_visibility_checks: self
                .node_visibility_checks
                .saturating_sub(earlier.node_visibility_checks),
            property_index_rebuild_rows: self
                .property_index_rebuild_rows
                .saturating_sub(earlier.property_index_rebuild_rows),
        }
    }

    /// Whether any id was materialised by a data-sized scan.
    ///
    /// The budget a point lookup must meet: it may probe an index, which is
    /// result-sized and uncounted, but it may not scan.
    #[must_use]
    pub fn scanned_any(self) -> bool {
        self.label_scan_ids > 0 || self.full_scan_ids > 0
    }
}

/// Per-store work counters.
///
/// `Relaxed` throughout: these are diagnostic totals, never a synchronisation
/// mechanism, and a test reads them after the query it measured has returned.
#[derive(Debug, Default)]
pub struct WorkCounters {
    label_scan_calls: AtomicU64,
    label_scan_ids: AtomicU64,
    full_scan_calls: AtomicU64,
    full_scan_ids: AtomicU64,
    property_index_route_keys: AtomicU64,
    property_index_posting_ids: AtomicU64,
    property_index_posting_intervals: AtomicU64,
    property_index_rebuild_rows: AtomicU64,
    node_materializations: AtomicU64,
    node_visibility_checks: AtomicU64,
}

impl WorkCounters {
    /// Records one label scan returning `ids` ids.
    pub fn record_label_scan(&self, ids: usize) {
        self.label_scan_calls.fetch_add(1, Ordering::Relaxed);
        self.label_scan_ids.fetch_add(ids as u64, Ordering::Relaxed);
    }

    /// Records one full node scan returning `ids` ids.
    pub fn record_full_scan(&self, ids: usize) {
        self.full_scan_calls.fetch_add(1, Ordering::Relaxed);
        self.full_scan_ids.fetch_add(ids as u64, Ordering::Relaxed);
    }

    /// Records ordered directory keys inspected by a property-index probe.
    pub fn record_property_index_route_keys(&self, keys: usize) {
        self.property_index_route_keys
            .fetch_add(keys as u64, Ordering::Relaxed);
    }

    /// Records posting identities and historical intervals inspected by a
    /// property-index probe.
    pub fn record_property_index_postings(&self, identities: usize, intervals: usize) {
        self.property_index_posting_ids
            .fetch_add(identities as u64, Ordering::Relaxed);
        self.property_index_posting_intervals
            .fetch_add(intervals as u64, Ordering::Relaxed);
    }

    /// Records source rows visited while rebuilding a property index.
    pub fn record_property_index_rebuild_rows(&self, rows: usize) {
        self.property_index_rebuild_rows
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Records construction of a whole node with labels and properties.
    pub fn record_node_materialization(&self) {
        self.node_materializations.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a direct epoch visibility probe.
    pub fn record_node_visibility_check(&self) {
        self.node_visibility_checks.fetch_add(1, Ordering::Relaxed);
    }

    /// Reads every counter.
    #[must_use]
    pub fn snapshot(&self) -> WorkSnapshot {
        WorkSnapshot {
            label_scan_calls: self.label_scan_calls.load(Ordering::Relaxed),
            label_scan_ids: self.label_scan_ids.load(Ordering::Relaxed),
            full_scan_calls: self.full_scan_calls.load(Ordering::Relaxed),
            full_scan_ids: self.full_scan_ids.load(Ordering::Relaxed),
            property_index_route_keys: self.property_index_route_keys.load(Ordering::Relaxed),
            property_index_posting_ids: self.property_index_posting_ids.load(Ordering::Relaxed),
            property_index_posting_intervals: self
                .property_index_posting_intervals
                .load(Ordering::Relaxed),
            property_index_rebuild_rows: self.property_index_rebuild_rows.load(Ordering::Relaxed),
            node_materializations: self.node_materializations.load(Ordering::Relaxed),
            node_visibility_checks: self.node_visibility_checks.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkCounters, WorkSnapshot};

    #[test]
    fn counts_each_primitive_separately() {
        let counters = WorkCounters::default();
        counters.record_label_scan(100);
        counters.record_label_scan(4);
        counters.record_full_scan(7);
        counters.record_property_index_route_keys(3);
        counters.record_property_index_postings(5, 8);
        counters.record_property_index_rebuild_rows(11);
        counters.record_node_materialization();
        counters.record_node_visibility_check();

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.label_scan_calls, 2);
        assert_eq!(snapshot.label_scan_ids, 104);
        assert_eq!(snapshot.full_scan_calls, 1);
        assert_eq!(snapshot.full_scan_ids, 7);
        assert_eq!(snapshot.property_index_route_keys, 3);
        assert_eq!(snapshot.property_index_posting_ids, 5);
        assert_eq!(snapshot.property_index_posting_intervals, 8);
        assert_eq!(snapshot.property_index_rebuild_rows, 11);
        assert_eq!(snapshot.node_materializations, 1);
        assert_eq!(snapshot.node_visibility_checks, 1);
        assert!(snapshot.scanned_any());
    }

    #[test]
    fn an_untouched_snapshot_has_scanned_nothing() {
        assert!(!WorkCounters::default().snapshot().scanned_any());
    }

    #[test]
    fn since_reports_the_work_between_two_snapshots() {
        let counters = WorkCounters::default();
        counters.record_label_scan(10);
        let before = counters.snapshot();
        counters.record_label_scan(3);
        counters.record_full_scan(2);
        counters.record_property_index_route_keys(4);
        counters.record_property_index_postings(5, 6);
        counters.record_property_index_rebuild_rows(7);
        counters.record_node_materialization();
        counters.record_node_visibility_check();

        let delta = counters.snapshot().since(before);
        assert_eq!(delta.label_scan_calls, 1);
        assert_eq!(delta.label_scan_ids, 3);
        assert_eq!(delta.full_scan_ids, 2);
        assert_eq!(delta.full_scan_calls, 1);
        assert_eq!(delta.property_index_route_keys, 4);
        assert_eq!(delta.property_index_posting_ids, 5);
        assert_eq!(delta.property_index_posting_intervals, 6);
        assert_eq!(delta.property_index_rebuild_rows, 7);
        assert_eq!(delta.node_materializations, 1);
        assert_eq!(delta.node_visibility_checks, 1);
    }

    #[test]
    fn since_never_underflows() {
        let later = WorkSnapshot::default();
        let earlier = WorkSnapshot {
            label_scan_ids: 5,
            ..WorkSnapshot::default()
        };
        assert_eq!(later.since(earlier).label_scan_ids, 0);
    }
}
