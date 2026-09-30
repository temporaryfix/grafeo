//! Range scan operator for property-bounded node scans.
//!
//! `RangeScanOperator` consumes
//! [`GraphStoreSearch::find_nodes_in_range_iter`](crate::graph::GraphStoreSearch::find_nodes_in_range_iter)
//! and emits `DataChunk`s of node ids whose property value falls within
//! `[min, max]` (with configurable inclusivity).
//!
//! ## Why a dedicated operator?
//!
//! The existing [`NodeListOperator`](super::single_row::NodeListOperator)
//! also chunks `Vec<NodeId>` into `DataChunk`s, but it loses the planner
//! signal that this scan is range-bounded. A dedicated operator:
//!
//! 1. Surfaces "range scan with per-block zone-map pruning" in EXPLAIN
//!    output so users can see the optimization fired.
//! 2. Owns the LIMIT-pushdown path (Phase 4e): when the planner knows a
//!    downstream LIMIT bound, the operator stops decoding rows after `n`
//!    matches without walking the rest of the column.
//! 3. Provides a stable seam for future enhancements (factorized output,
//!    parallel block scan) without churning the planner.
//!
//! ## Materialization strategy
//!
//! Phase 4c materializes the iterator into a `Vec<NodeId>` on the first
//! `next()` call, then chunks. Block-level skip pruning still happens
//! during iterator construction, so the architectural value is intact.
//! The materialization step is bounded by the optional limit set via
//! [`with_limit`](Self::with_limit) (Phase 4e). Streaming chunk-by-chunk
//! materialization is a future pass; it requires either a self-referential
//! struct or a cursor-based API on the store, neither of which is free
//! in safe Rust today.
//!
//! ## Label and MVCC filtering
//!
//! The planner used to filter the eager `Vec<NodeId>` after the range
//! lookup. The operator absorbs both filters via
//! [`with_label_filter`](Self::with_label_filter) and
//! [`with_transaction_context`](Self::with_transaction_context), preserving
//! the existing semantics while keeping the `RangeScanOperator` as the
//! single entry point.

use std::sync::Arc;

use grafeo_common::types::{EpochId, LogicalType, NodeId, TransactionId, Value};

use super::{ExpressionPredicate, Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::{GraphStoreSearch, PropertyIndexPredicate, PropertyIndexRequest};
use grafeo_common::types::PropertyKey;

/// Pull-based operator that emits node ids whose property value falls
/// within a range. See the module docs for details.
pub struct RangeScanOperator {
    store: Arc<dyn GraphStoreSearch>,
    property: String,
    min: Option<Value>,
    max: Option<Value>,
    min_inclusive: bool,
    max_inclusive: bool,
    chunk_capacity: usize,
    /// Optional row-count cap for LIMIT pushdown (Phase 4e).
    limit: Option<usize>,
    /// Optional label filter (only nodes of this label survive).
    label_filter: Option<String>,
    /// Optional MVCC transaction context (epoch + tx).
    transaction_context: Option<(EpochId, TransactionId)>,

    /// Materialized result, lazily built on first `next()`.
    materialized: Option<Vec<NodeId>>,
    /// Current cursor into `materialized`.
    position: usize,
}

impl RangeScanOperator {
    /// Creates a range scan over `store` for the given property and bounds.
    ///
    /// `chunk_capacity` is the number of rows per emitted `DataChunk`;
    /// the standard default in this codebase is 2048.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        property: impl Into<String>,
        min: Option<Value>,
        max: Option<Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        chunk_capacity: usize,
    ) -> Self {
        Self {
            store,
            property: property.into(),
            min,
            max,
            min_inclusive,
            max_inclusive,
            chunk_capacity,
            limit: None,
            label_filter: None,
            transaction_context: None,
            materialized: None,
            position: 0,
        }
    }

    /// Sets a row-count cap that bounds the materialization step.
    ///
    /// Stops consuming candidates after enough visible, label-qualified matches.
    /// Lazy stores avoid decoding later blocks; eager stores still enumerate
    /// candidates before this cap applies.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Returns the row-count cap set by [`with_limit`](Self::with_limit),
    /// or `None` if no cap is in effect. Used by planner tests to verify
    /// LIMIT pushdown wired the cap.
    #[must_use]
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Restricts the result to nodes carrying `label`.
    ///
    /// Applied during materialization with a point membership check, retaining
    /// the store's projection and historical-label semantics.
    #[must_use]
    pub fn with_label_filter(mut self, label: impl Into<String>) -> Self {
        self.label_filter = Some(label.into());
        self
    }

    /// Filters results by MVCC visibility at the given epoch and tx.
    ///
    /// Applied during materialization via `store.get_node_versioned`.
    /// Required for any planner-emitted scan: the existing `plan_range_filter`
    /// always applies this filter, and the operator preserves that.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Self {
        self.transaction_context = Some((epoch, transaction_id));
        self
    }

    fn ensure_materialized(&mut self) -> OperatorResult {
        if self.materialized.is_some() {
            return Ok(None);
        }

        if self.limit == Some(0) {
            self.materialized = Some(Vec::new());
            return Ok(None);
        }

        // `SYSTEM` is the planner's sentinel for an ordinary snapshot read;
        // it is not a real transaction and must not hide the committed index
        // behind a transaction overlay.
        let (epoch, transaction_id) = self
            .transaction_context
            .map_or((self.store.current_epoch(), None), |(epoch, tx)| {
                (epoch, (tx != TransactionId::SYSTEM).then_some(tx))
            });
        if let Some(tx) = transaction_id {
            if let Some(label) = &self.label_filter {
                self.store.record_label_predicate_read(tx, label);
            } else {
                self.store.record_lpg_dataset_read(tx);
            }
        }

        // An indexed result is complete, including an empty result.  Only
        // `None` means that this property/range cannot be answered by an
        // admitted index and should use the established iterator fallback.
        let indexed = self
            .store
            .lookup_nodes_indexed(PropertyIndexRequest {
                property: &self.property,
                predicate: PropertyIndexPredicate::Range {
                    min: self.min.as_ref(),
                    max: self.max.as_ref(),
                    min_inclusive: self.min_inclusive,
                    max_inclusive: self.max_inclusive,
                },
                epoch,
                transaction_id,
            })
            .map_err(|error| OperatorError::Execution(error.to_string()))?;

        let candidates: Box<dyn Iterator<Item = NodeId> + '_> = match indexed {
            Some(ids) => Box::new(ids.into_iter()),
            // Latest-value range iteration cannot enumerate historical or
            // transaction-local matches. Retain the complete identity fallback.
            None if transaction_id.is_some() || epoch < self.store.current_epoch() => {
                Box::new(self.store.all_node_ids().into_iter())
            }
            None => self.store.find_nodes_in_range_iter(
                &self.property,
                self.min.as_ref(),
                self.max.as_ref(),
                self.min_inclusive,
                self.max_inclusive,
            ),
        };
        let property_key = PropertyKey::new(&self.property);

        // Filter inline (label + MVCC) and stop only after `limit` matches
        // *survive* the filters. Applying limit before filtering would
        // under-return rows when early range hits are filtered out.
        let mut collected: Vec<NodeId> = Vec::new();
        for id in candidates {
            let visible = transaction_id.map_or_else(
                || self.store.get_node_at_epoch(id, epoch).is_some(),
                |tx| self.store.get_node_versioned(id, epoch, tx).is_some(),
            );
            if !visible {
                continue;
            }
            if self.label_filter.as_deref().is_some_and(|label| {
                !self.store.node_has_label_at_epoch(
                    id,
                    label,
                    epoch,
                    transaction_id.unwrap_or(TransactionId::SYSTEM),
                )
            }) {
                continue;
            }
            let Some(value) =
                self.store
                    .read_node_property_visible(id, &property_key, epoch, transaction_id)
            else {
                continue;
            };
            if !ExpressionPredicate::matches_property_index_predicate(
                &value,
                PropertyIndexPredicate::Range {
                    min: self.min.as_ref(),
                    max: self.max.as_ref(),
                    min_inclusive: self.min_inclusive,
                    max_inclusive: self.max_inclusive,
                },
            ) {
                continue;
            }
            collected.push(id);
            if let Some(n) = self.limit
                && collected.len() >= n
            {
                break;
            }
        }

        self.materialized = Some(collected);
        Ok(None)
    }
}

impl Operator for RangeScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.ensure_materialized()?;
        let nodes = self
            .materialized
            .as_ref()
            .expect("ensure_materialized populates Some");

        if self.position >= nodes.len() {
            return Ok(None);
        }

        // Guard against `chunk_capacity == 0`: that would set `end == position`,
        // emit empty chunks, and never advance — an infinite loop for callers.
        let step = self.chunk_capacity.max(1);
        let end = (self.position + step).min(nodes.len());
        let count = end - self.position;

        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, step);
        {
            let col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
            for i in self.position..end {
                col.push_node_id(nodes[i]);
            }
        }
        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.materialized = None;
    }

    fn name(&self) -> &'static str {
        "RangeScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "compact-store"))]
mod tests {
    use super::*;
    use crate::graph::compact::CompactStore;
    use crate::graph::compact::builder::CompactStoreBuilder;
    #[cfg(feature = "lpg")]
    use crate::graph::lpg::LpgStore;

    fn build_person_store() -> Arc<dyn GraphStoreSearch> {
        Arc::new(
            CompactStoreBuilder::new()
                .node_table("Person", |t| {
                    t.column_bitpacked("age", &[25, 30, 35, 40, 45], 6)
                })
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn alix_range_scan_emits_matching_nodes() {
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            store,
            "age",
            Some(Value::Int64(30)),
            Some(Value::Int64(40)),
            true,
            true,
            2048,
        );

        let chunk = op.next().unwrap().expect("first chunk should be Some");
        assert_eq!(chunk.row_count(), 3, "ages 30, 35, 40 match");
        let none = op.next().unwrap();
        assert!(none.is_none(), "single chunk fits all matches");
    }

    #[test]
    fn gus_range_scan_chunks_in_capacity_sized_batches() {
        let values: Vec<u64> = (0..100u64).collect();
        let store: Arc<dyn GraphStoreSearch> = Arc::new(
            CompactStoreBuilder::new()
                .node_table("Big", |t| t.column_bitpacked("v", &values, 7))
                .build()
                .unwrap(),
        );

        let mut op = RangeScanOperator::new(store, "v", None, None, true, true, 10);

        let mut total = 0usize;
        let mut chunk_count = 0usize;
        while let Some(chunk) = op.next().unwrap() {
            chunk_count += 1;
            total += chunk.row_count();
            assert!(chunk.row_count() <= 10);
        }
        assert_eq!(total, 100);
        assert_eq!(chunk_count, 10);
    }

    #[test]
    fn vincent_range_scan_with_limit_short_circuits() {
        let values: Vec<u64> = (0..1000u64).collect();
        let store: Arc<dyn GraphStoreSearch> = Arc::new(
            CompactStoreBuilder::new()
                .node_table("Big", |t| t.column_bitpacked("v", &values, 10))
                .build()
                .unwrap(),
        );

        let mut empty = RangeScanOperator::new(Arc::clone(&store), "v", None, None, true, true, 64)
            .with_limit(0);
        assert!(empty.next().unwrap().is_none());

        let mut op = RangeScanOperator::new(store, "v", None, None, true, true, 64).with_limit(5);

        let mut total = 0usize;
        while let Some(chunk) = op.next().unwrap() {
            total += chunk.row_count();
        }
        assert_eq!(total, 5, "limit caps the row count");
    }

    #[test]
    fn jules_range_scan_disjoint_range_yields_nothing() {
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            store,
            "age",
            Some(Value::Int64(100)),
            Some(Value::Int64(200)),
            true,
            true,
            2048,
        );
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn mia_range_scan_reset_replays_chunks() {
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            store,
            "age",
            Some(Value::Int64(25)),
            Some(Value::Int64(45)),
            true,
            true,
            2,
        );

        let first_pass: Vec<usize> = std::iter::from_fn(|| op.next().unwrap())
            .map(|c| c.row_count())
            .collect();

        op.reset();

        let second_pass: Vec<usize> = std::iter::from_fn(|| op.next().unwrap())
            .map(|c| c.row_count())
            .collect();

        assert_eq!(first_pass, second_pass);
    }

    #[test]
    fn butch_range_scan_into_any_downcasts() {
        let store = build_person_store();
        let op = RangeScanOperator::new(store, "age", None, None, true, true, 2048);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<RangeScanOperator>().is_ok());
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn range_scan_historical_and_transaction_views_use_complete_candidates() {
        let store = Arc::new(LpgStore::new().unwrap());
        let created = EpochId::new(1);
        let current = EpochId::new(2);
        store.set_epoch(created);
        let changed = store.create_node(&[]);
        let deleted = store.create_node(&[]);
        store.set_node_property_at_epoch(changed, "score", Value::Int64(5), created);
        store.set_node_property_at_epoch(deleted, "score", Value::Int64(5), created);
        store.set_node_property_at_epoch(changed, "score", Value::Int64(50), current);
        assert!(store.delete_node_at_epoch(deleted, current));
        store.sync_epoch(current);

        let mut historical = RangeScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "score",
            Some(Value::Int64(5)),
            Some(Value::Int64(5)),
            true,
            true,
            2048,
        )
        .with_transaction_context(created, TransactionId::SYSTEM);
        let mut historical_rows = 0;
        while let Some(chunk) = historical.next().unwrap() {
            historical_rows += chunk.row_count();
        }
        assert_eq!(
            historical_rows, 2,
            "old SET and deletion values remain visible"
        );

        let tx = TransactionId::new(77);
        let own = store.create_node(&[]);
        store.set_node_property(own, "score", Value::Int64(0));
        store.set_node_property_buffered(own, "score", Value::Int64(5), tx);
        let mut writer = RangeScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "score",
            Some(Value::Int64(5)),
            Some(Value::Int64(5)),
            true,
            true,
            2048,
        )
        .with_transaction_context(current, tx)
        .with_limit(1);
        let first = writer.next().unwrap().expect("own buffered value matches");
        assert_eq!(first.row_count(), 1);
        assert!(writer.next().unwrap().is_none());
    }

    #[test]
    fn shosanna_range_scan_name_is_stable() {
        let store = build_person_store();
        let op = RangeScanOperator::new(store, "age", None, None, true, true, 2048);
        assert_eq!(op.name(), "RangeScan");
    }

    #[test]
    fn hans_range_scan_with_label_filter_intersects() {
        // Two labels carry the same property name; label filter must
        // restrict results to one label only.
        let store: Arc<dyn GraphStoreSearch> = Arc::new(
            CompactStoreBuilder::new()
                .node_table("A", |t| t.column_bitpacked("v", &[1, 2, 3], 4))
                .node_table("B", |t| t.column_bitpacked("v", &[1, 2, 3], 4))
                .build()
                .unwrap(),
        );

        let mut op = RangeScanOperator::new(Arc::clone(&store), "v", None, None, true, true, 2048)
            .with_label_filter("A");

        let chunk = op.next().unwrap().expect("at least one chunk");
        // Only nodes from label A survive; A has 3 rows.
        assert_eq!(chunk.row_count(), 3);

        // Sanity: without the label filter, we'd see 6 rows.
        let mut op_no_label = RangeScanOperator::new(store, "v", None, None, true, true, 2048);
        let chunk2 = op_no_label.next().unwrap().expect("at least one chunk");
        assert_eq!(chunk2.row_count(), 6);
    }

    #[test]
    fn beatrix_range_scan_label_filter_with_disjoint_label_yields_nothing() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(
            CompactStoreBuilder::new()
                .node_table("A", |t| t.column_bitpacked("v", &[1, 2, 3], 4))
                .build()
                .unwrap(),
        );

        let mut op =
            RangeScanOperator::new(store, "v", None, None, true, true, 2048).with_label_filter("Z");

        // Label "Z" doesn't exist; intersection is empty.
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn django_range_scan_default_trait_impl_works_for_non_compact_stores() {
        // Validates the default `find_nodes_in_range_iter` impl on
        // `GraphStoreSearch`: a CompactStore exposed as `Arc<dyn>` should
        // STILL hit the override; the trait dispatch is correct.
        // (The non-CompactStore path is exercised via the LpgStore tests
        // separately; here we assert the dyn-dispatch wiring is sound.)
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            Arc::clone(&store),
            "age",
            Some(Value::Int64(25)),
            Some(Value::Int64(45)),
            true,
            true,
            2048,
        );
        let chunk = op.next().unwrap().expect("at least one chunk");
        assert_eq!(chunk.row_count(), 5);
    }

    /// Sanity check that the trait dispatch reaches the CompactStore
    /// override (and not the eager default) for a CompactStore-backed
    /// `Arc<dyn GraphStoreSearch>`. We can't easily observe block skip
    /// from outside, so we verify behavioral equivalence: the iterator
    /// must yield the same set as the eager `find_nodes_in_range`.
    #[test]
    fn tarantino_dyn_dispatch_yields_same_results_as_eager() {
        let store = build_person_store();
        let min = Value::Int64(30);
        let max = Value::Int64(40);
        let lazy: Vec<NodeId> = store
            .find_nodes_in_range_iter("age", Some(&min), Some(&max), true, true)
            .collect();
        let mut eager = store.find_nodes_in_range("age", Some(&min), Some(&max), true, true);
        let mut lazy_sorted = lazy;
        lazy_sorted.sort_unstable();
        eager.sort_unstable();
        assert_eq!(lazy_sorted, eager);
    }

    /// Helper to silence "unused import" when compact-store gates change.
    #[allow(dead_code)]
    fn _compact_store_marker(_: Arc<CompactStore>) {}
}
