//! Streaming bounded-heap top-K operator.
//!
//! Subsumes `Limit` over `Sort` for the `LIMIT k ORDER BY ...` pattern.
//! Instead of materializing every input row, sorting all of them, and
//! discarding all but the first k, this operator maintains a max-heap of
//! size k keyed by the user's sort tuple, with the comparator inverted so
//! `peek()` returns the worst row by user order. For input cardinality N,
//! memory is O(k) and comparisons are O(N log k). The heap drains in
//! user-requested order via `BinaryHeap::into_sorted_vec`, so no separate
//! sort step is needed.
//!
//! Stability matches `slice::sort`'s stable guarantee: rows tied on every
//! sort key are output in input order, achieved with a monotonic
//! insertion-id tiebreaker.
//!
//! See `plan_limit` in `grafeo-engine` for the dispatch point that builds
//! this operator. PROFILE-mode plans bypass the rewrite for entry-count
//! parity with the logical tree, and `LimitOperator` over `SortOperator`
//! runs instead.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};

use super::sort::{SortKey, compare_sort_values};
use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// Streaming bounded top-K operator.
pub struct TopKOperator {
    child: Box<dyn Operator>,
    /// Shared with every `HeapEntry` via `Arc` so `HeapEntry::Ord` can
    /// compare without raw pointers (`unsafe_code = "deny"` workspace-wide).
    /// Allocated once in `new` and refcount-bumped per heap insertion. The
    /// marginal cost is negligible at k=50, N=1M.
    sort_keys: Arc<Vec<SortKey>>,
    limit: usize,
    output_schema: Vec<LogicalType>,
    state: TopKState,
    #[cfg(test)]
    materialized_rows: std::sync::atomic::AtomicUsize,
}

enum TopKState {
    Building {
        heap: BinaryHeap<HeapEntry>,
        next_insertion_id: u64,
    },
    Draining {
        rows: Vec<HeapEntry>,
        position: usize,
    },
    Done,
}

struct HeapEntry {
    sort_values: Vec<Option<Value>>,
    row_values: Vec<Option<Value>>,
    /// Effective output schema for this retained row. This is bounded by k
    /// and preserves typed edge-list provenance after the child is drained.
    row_schema: Arc<Vec<LogicalType>>,
    insertion_id: u64,
    /// Shared with the owning operator. Refcount-bumped per insertion.
    sort_keys: Arc<Vec<SortKey>>,
}

impl TopKOperator {
    /// Constructs a streaming bounded top-k operator that yields the first
    /// `limit` rows of `child` in `sort_keys` order, using O(limit) memory
    /// regardless of `child`'s cardinality.
    ///
    /// Equivalent in output to `LimitOperator(SortOperator(child, sort_keys), limit)`,
    /// including stability on ties.
    ///
    /// `output_schema` must have the same width as `child`'s output; the
    /// operator asserts this on first pull (`debug_assert`) to catch planner
    /// bugs that would silently truncate or null-pad rows.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::execution::DataChunk;
    /// use grafeo_core::execution::chunk::DataChunkBuilder;
    /// use grafeo_core::execution::operators::{Operator, OperatorResult, SortKey, TopKOperator};
    /// use grafeo_common::types::LogicalType;
    ///
    /// struct Source { chunk: Option<DataChunk> }
    /// impl Operator for Source {
    ///     fn next(&mut self) -> OperatorResult { Ok(self.chunk.take()) }
    ///     fn reset(&mut self) {}
    ///     fn name(&self) -> &'static str { "Source" }
    ///     fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> { self }
    /// }
    ///
    /// let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
    /// for v in [19i64, 88, 33, 8, 319] {
    ///     b.column_mut(0).unwrap().push_int64(v);
    ///     b.advance_row();
    /// }
    /// let source = Source { chunk: Some(b.finish()) };
    ///
    /// let mut top_k = TopKOperator::new(
    ///     Box::new(source),
    ///     vec![SortKey::descending(0)],
    ///     3,
    ///     vec![LogicalType::Int64],
    /// );
    ///
    /// let chunk = top_k.next().unwrap().unwrap();
    /// let mut out = vec![];
    /// for row in chunk.selected_indices() {
    ///     out.push(chunk.column(0).unwrap().get_int64(row).unwrap());
    /// }
    /// assert_eq!(out, vec![319, 88, 33]);
    /// ```
    #[must_use]
    pub fn new(
        child: Box<dyn Operator>,
        sort_keys: Vec<SortKey>,
        limit: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            sort_keys: Arc::new(sort_keys),
            limit,
            output_schema,
            state: TopKState::Building {
                heap: BinaryHeap::new(),
                next_insertion_id: 0,
            },
            #[cfg(test)]
            materialized_rows: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Decomposes this operator into its child, sort keys, and limit.
    ///
    /// Mirrors `SortOperator::into_parts` and `LimitOperator::into_parts`
    /// so a future `TopKPushOperator` can drop in via `pipeline_convert.rs`
    /// without an API break. `Arc::try_unwrap` succeeds before the operator
    /// is first pulled or once it has reached `TopKState::Done`. Mid-drain
    /// the rows `Vec` still holds `Arc` clones, so the fallback clones the
    /// keys.
    #[must_use]
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<SortKey>, usize) {
        let sort_keys = Arc::try_unwrap(self.sort_keys).unwrap_or_else(|arc| (*arc).clone());
        (self.child, sort_keys, self.limit)
    }

    /// Resolves the schema carried by rows from one source chunk. An open
    /// `Any` output preserves a typed edge-list source while ordinary values
    /// remain open, so integer lists cannot acquire edge provenance.
    fn effective_schema(&self, source: &DataChunk) -> Vec<LogicalType> {
        self.output_schema
            .iter()
            .enumerate()
            .map(|(index, configured)| {
                if !matches!(configured, LogicalType::Any) {
                    return configured.clone();
                }
                let Some(source_type) = source.column(index).map(|column| column.data_type())
                else {
                    return configured.clone();
                };
                if matches!(source_type, LogicalType::Node | LogicalType::Edge)
                    || matches!(
                        source_type,
                        LogicalType::List(item) if item.as_ref() == &LogicalType::Edge
                    )
                {
                    source_type.clone()
                } else {
                    configured.clone()
                }
            })
            .collect()
    }
}

impl Operator for TopKOperator {
    fn next(&mut self) -> OperatorResult {
        if matches!(self.state, TopKState::Building { .. }) {
            let TopKState::Building {
                mut heap,
                mut next_insertion_id,
            } = std::mem::replace(&mut self.state, TopKState::Done)
            else {
                unreachable!("matches! guard above")
            };

            let mut schema_checked = false;
            while let Some(chunk) = self.child.next()? {
                let mut chunk_schema = None;
                if !schema_checked {
                    debug_assert_eq!(
                        chunk.column_count(),
                        self.output_schema.len(),
                        "TopKOperator output_schema width must match child schema width",
                    );
                    schema_checked = true;
                }

                for row_idx in chunk.selected_indices() {
                    let new_sort_values =
                        extract_sort_values(&chunk, row_idx, self.sort_keys.as_slice());

                    let should_push = if heap.len() < self.limit {
                        true
                    } else if let Some(top) = heap.peek() {
                        row_beats_heap_top(&new_sort_values, top, self.sort_keys.as_slice())
                    } else {
                        // limit == 0: heap stays empty, never push.
                        false
                    };

                    if !should_push {
                        continue;
                    }

                    let row_values = extract_row_values(&chunk, row_idx, self.output_schema.len());
                    let row_schema = Arc::clone(
                        chunk_schema.get_or_insert_with(|| Arc::new(self.effective_schema(&chunk))),
                    );
                    #[cfg(test)]
                    self.materialized_rows
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let entry = HeapEntry {
                        sort_values: new_sort_values,
                        row_values,
                        row_schema,
                        insertion_id: next_insertion_id,
                        sort_keys: Arc::clone(&self.sort_keys),
                    };
                    next_insertion_id += 1;
                    if heap.len() < self.limit {
                        heap.push(entry);
                    } else {
                        // Heap is full and the new entry beat the worst: replace
                        // the heap's max in place. One sift-down vs push+pop's
                        // two reheapifies, significant for large N.
                        let mut top = heap.peek_mut().expect("heap.len() == limit > 0");
                        *top = entry;
                    }
                }
            }

            let rows = heap.into_sorted_vec();
            self.state = TopKState::Draining { rows, position: 0 };
        }

        if let TopKState::Draining { rows, position } = &mut self.state {
            if *position < rows.len() {
                let effective_schema = Arc::clone(&rows[*position].row_schema);
                let mut builder = DataChunkBuilder::with_capacity(effective_schema.as_ref(), 2048);
                while *position < rows.len() && !builder.is_full() {
                    let entry = &rows[*position];
                    if entry.row_schema.as_ref() != effective_schema.as_ref() {
                        break;
                    }
                    for col_idx in 0..self.output_schema.len() {
                        if let Some(dst_col) = builder.column_mut(col_idx) {
                            let val = entry.row_values[col_idx].clone().unwrap_or(Value::Null);
                            dst_col.push_value(val);
                        }
                    }
                    builder.advance_row();
                    *position += 1;
                }
                if builder.row_count() > 0 {
                    return Ok(Some(builder.finish()));
                }
            }
            self.state = TopKState::Done;
        }

        Ok(None)
    }

    fn reset(&mut self) {
        self.child.reset();
        self.state = TopKState::Building {
            heap: BinaryHeap::new(),
            next_insertion_id: 0,
        };
        #[cfg(test)]
        self.materialized_rows
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn name(&self) -> &'static str {
        "TopK"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
impl TopKOperator {
    pub(crate) fn materialized_rows(&self) -> usize {
        self.materialized_rows
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn extract_sort_values(
    chunk: &DataChunk,
    row_idx: usize,
    sort_keys: &[SortKey],
) -> Vec<Option<Value>> {
    sort_keys
        .iter()
        .map(|k| chunk.column(k.column).and_then(|c| c.get_value(row_idx)))
        .collect()
}

fn extract_row_values(chunk: &DataChunk, row_idx: usize, n_cols: usize) -> Vec<Option<Value>> {
    (0..n_cols)
        .map(|i| chunk.column(i).and_then(|c| c.get_value(row_idx)))
        .collect()
}

/// Strict better-than test: does `new` beat the current heap top per
/// user-requested order?
///
/// Inserting a new row that ties on every key must NOT displace the existing
/// top. The existing top arrived first and wins ties (stability).
fn row_beats_heap_top(new: &[Option<Value>], top: &HeapEntry, keys: &[SortKey]) -> bool {
    for (i, key) in keys.iter().enumerate() {
        let user_cmp =
            compare_sort_values(&new[i], &top.sort_values[i], key.direction, key.null_order);
        match user_cmp {
            Ordering::Less => return true,
            Ordering::Greater => return false,
            Ordering::Equal => continue,
        }
    }
    false
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.insertion_id == other.insertion_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Both entries share the same Arc<Vec<SortKey>> (one per
        // TopKOperator); use self's view.
        //
        // Goal: BinaryHeap is a max-heap. peek() must return the
        // worst-by-user-order so we can evict it on overflow. The final user
        // comparison already makes that worst row `Greater` for either
        // direction while preserving independently requested null placement.
        for (i, key) in self.sort_keys.iter().enumerate() {
            let heap_cmp = compare_sort_values(
                &self.sort_values[i],
                &other.sort_values[i],
                key.direction,
                key.null_order,
            );
            if heap_cmp != Ordering::Equal {
                return heap_cmp;
            }
        }
        // Tiebreak: larger insertion_id is "greater" so newer ties bubble to
        // peek and pop() evicts them first. into_sorted_vec then yields
        // older-first = input order, preserving stability.
        self.insertion_id.cmp(&other.insertion_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::selection::SelectionVector;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        original_chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                original_chunks: chunks.clone(),
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.chunks.clone_from(&self.original_chunks);
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn chunk_int64(values: &[i64]) -> DataChunk {
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for &v in values {
            b.column_mut(0).unwrap().push_int64(v);
            b.advance_row();
        }
        b.finish()
    }

    fn collect_int64_col(op: &mut dyn Operator) -> Vec<i64> {
        let mut out = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }
        out
    }

    fn chunk_values(values: &[Value]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for value in values {
            builder.column_mut(0).unwrap().push_value(value.clone());
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_values(op: &mut dyn Operator) -> Vec<Value> {
        let mut values = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                values.push(
                    chunk
                        .column(0)
                        .unwrap()
                        .get_value(row)
                        .expect("test row has one value"),
                );
            }
        }
        values
    }

    #[test]
    fn top_k_returns_top_k_descending() {
        let mock = MockOperator::new(vec![chunk_int64(&[19, 88, 33, 8, 319])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            3,
            vec![LogicalType::Int64],
        );
        let out = collect_int64_col(&mut top_k);
        assert_eq!(out, vec![319, 88, 33]);
    }

    #[test]
    fn top_k_respects_final_null_placement_for_every_direction() {
        use super::super::sort::{NullOrder, SortDirection};

        let input = [
            Value::Int64(2),
            Value::Null,
            Value::Int64(1),
            Value::Null,
            Value::Int64(3),
        ];
        let cases = [
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Ascending,
                    null_order: NullOrder::NullsFirst,
                },
                vec![Value::Null, Value::Null, Value::Int64(1)],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Ascending,
                    null_order: NullOrder::NullsLast,
                },
                vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Descending,
                    null_order: NullOrder::NullsFirst,
                },
                vec![Value::Null, Value::Null, Value::Int64(3)],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Descending,
                    null_order: NullOrder::NullsLast,
                },
                vec![Value::Int64(3), Value::Int64(2), Value::Int64(1)],
            ),
        ];

        for (key, expected) in cases {
            let mock = MockOperator::new(vec![chunk_values(&input)]);
            let mut top_k =
                TopKOperator::new(Box::new(mock), vec![key], 3, vec![LogicalType::Int64]);
            assert_eq!(collect_values(&mut top_k), expected);
        }
    }

    fn chunk_int_str(rows: &[(i64, &str)]) -> DataChunk {
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
        for (n, s) in rows {
            b.column_mut(0).unwrap().push_int64(*n);
            b.column_mut(1).unwrap().push_string(*s);
            b.advance_row();
        }
        b.finish()
    }

    fn collect_int_str(op: &mut dyn Operator) -> Vec<(i64, String)> {
        let mut out = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                let n = chunk.column(0).unwrap().get_int64(row).unwrap();
                let s = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                out.push((n, s));
            }
        }
        out
    }

    #[test]
    fn top_k_is_stable_on_ties_descending() {
        // Tied on key=88 across two rows; stability says the first arrival wins.
        let mock = MockOperator::new(vec![chunk_int_str(&[
            (3, "Vincent"),
            (88, "Jules"),
            (3, "Mia"),
            (88, "Butch"),
        ])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            2,
            vec![LogicalType::Int64, LogicalType::String],
        );
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(88, "Jules".into()), (88, "Butch".into())]);
    }

    #[test]
    fn top_k_is_stable_on_ties_ascending() {
        let mock = MockOperator::new(vec![chunk_int_str(&[
            (88, "Vincent"),
            (3, "Jules"),
            (88, "Mia"),
            (3, "Butch"),
        ])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            2,
            vec![LogicalType::Int64, LogicalType::String],
        );
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(3, "Jules".into()), (3, "Butch".into())]);
    }

    #[test]
    fn top_k_skips_materialization_for_losers() {
        // 1000 inputs forming a permutation of 0..1000 via i*31 mod 1000
        // (gcd(31, 1000) = 1, so this is a true permutation, no duplicates).
        // k=5 ASC: after the heap fills with the first 5 inputs, each
        // subsequent winner causes one peek_mut replace (1 materialization).
        // Total materializations should be far below 1000.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let values: Vec<i64> = (0..1000_i64).map(|i| (i * 31 + 7) % 1000).collect();
        let mock = MockOperator::new(vec![chunk_int64(&values)]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            5,
            vec![LogicalType::Int64],
        );

        let out = collect_int64_col(&mut top_k);
        assert_eq!(out.len(), 5);

        // Pessimistic upper bound: every distinct minimum seen along the way
        // could be a materialization. For an unbiased permutation, the
        // expected number of new minima in 1000 draws is H(1000) ~ 7.5;
        // allow 50 for slack against the specific permutation above.
        let materialized = top_k.materialized_rows();
        assert!(
            materialized < 50,
            "expected < 50 materializations for k=5 over 1000 inputs, got {materialized}"
        );
    }

    #[test]
    fn top_k_multi_key_mixed_directions() {
        // ORDER BY x DESC, y ASC. With k=2, the top 2 by (x DESC, y ASC):
        // input (88, "5"), (88, "3"), (19, "8"), (88, "5b") gives top 2 of
        // (88, "3") then (88, "5"). The second (88, "5b") is dropped, strictly
        // worse than (88, "5") on ASC string order.
        let mock = MockOperator::new(vec![chunk_int_str(&[
            (88, "5"),
            (88, "3"),
            (19, "8"),
            (88, "5b"),
        ])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0), SortKey::ascending(1)],
            2,
            vec![LogicalType::Int64, LogicalType::String],
        );
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(88, "3".into()), (88, "5".into())]);
    }

    #[test]
    fn top_k_handles_nulls_first_ascending() {
        use super::super::sort::NullOrder;
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [Some(19_i64), None, Some(88), None, Some(3)] {
            match v {
                Some(n) => b.column_mut(0).unwrap().push_int64(n),
                None => b.column_mut(0).unwrap().push_value(Value::Null),
            }
            b.advance_row();
        }
        let chunk = b.finish();
        let mock = MockOperator::new(vec![chunk]);

        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0).with_null_order(NullOrder::NullsFirst)],
            3,
            vec![LogicalType::Int64],
        );

        // ORDER BY x ASC NULLS FIRST gives [Null, Null, 3, 19, 88]; LIMIT 3 = [Null, Null, 3].
        let mut out = Vec::new();
        while let Some(chunk) = top_k.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_value(row));
            }
        }
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0], Some(Value::Null)));
        assert!(matches!(out[1], Some(Value::Null)));
        assert_eq!(out[2], Some(Value::Int64(3)));
    }

    #[test]
    fn top_k_handles_nulls_last_ascending() {
        use super::super::sort::NullOrder;
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [Some(19_i64), None, Some(88), None, Some(3)] {
            match v {
                Some(n) => b.column_mut(0).unwrap().push_int64(n),
                None => b.column_mut(0).unwrap().push_value(Value::Null),
            }
            b.advance_row();
        }
        let chunk = b.finish();
        let mock = MockOperator::new(vec![chunk]);

        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0).with_null_order(NullOrder::NullsLast)],
            3,
            vec![LogicalType::Int64],
        );

        // ORDER BY x ASC NULLS LAST gives [3, 19, 88, Null, Null]; LIMIT 3 = [3, 19, 88].
        let mut out = Vec::new();
        while let Some(chunk) = top_k.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_value(row));
            }
        }
        assert_eq!(
            out,
            vec![
                Some(Value::Int64(3)),
                Some(Value::Int64(19)),
                Some(Value::Int64(88))
            ]
        );
    }

    #[test]
    fn top_k_empty_input() {
        let mock = MockOperator::new(vec![]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), Vec::<i64>::new());
    }

    #[test]
    fn top_k_k_zero_returns_no_rows() {
        let mock = MockOperator::new(vec![chunk_int64(&[3, 19, 88])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            0,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), Vec::<i64>::new());
    }

    #[test]
    fn top_k_k_greater_than_n() {
        let mock = MockOperator::new(vec![chunk_int64(&[19, 88, 3])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            10,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), vec![88, 19, 3]);
    }

    #[test]
    fn top_k_returns_top_k_ascending() {
        let mock = MockOperator::new(vec![chunk_int64(&[19, 88, 33, 8, 319])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            3,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), vec![8, 19, 33]);
    }

    #[test]
    fn top_k_spans_multiple_input_chunks() {
        let mock = MockOperator::new(vec![
            chunk_int64(&[19, 88]),
            chunk_int64(&[33, 8]),
            chunk_int64(&[40, 319]),
        ]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            3,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), vec![319, 88, 40]);
    }

    #[test]
    fn top_k_into_parts_round_trip() {
        let mock = MockOperator::new(vec![chunk_int64(&[3, 19, 88])]);
        let top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        );
        let (mut child, sort_keys, limit) = top_k.into_parts();
        assert_eq!(sort_keys.len(), 1);
        assert_eq!(limit, 5);
        let chunk = child.next().unwrap().expect("mock yields one chunk");
        assert_eq!(chunk.row_count(), 3);
    }

    #[test]
    fn top_k_name() {
        let mock = MockOperator::new(vec![]);
        let top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        );
        assert_eq!(top_k.name(), "TopK");
    }

    #[test]
    fn top_k_into_any_downcasts() {
        let mock = MockOperator::new(vec![]);
        let op: Box<dyn Operator> = Box::new(TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        ));
        let any = op.into_any();
        assert!(any.downcast::<TopKOperator>().is_ok());
    }

    fn typed_edge_list_chunk(rows: &[(i64, i64)]) -> DataChunk {
        let edge_list = LogicalType::List(Box::new(LogicalType::Edge));
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, edge_list]);
        for &(key, edge_id) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::List(vec![Value::Int64(edge_id)].into()));
            builder.advance_row();
        }
        builder.finish()
    }

    fn ordinary_list_chunk(rows: &[(i64, i64)]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[
            LogicalType::Int64,
            LogicalType::List(Box::new(LogicalType::Int64)),
        ]);
        for &(key, value) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::List(vec![Value::Int64(value)].into()));
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_mixed_list_rows(operator: &mut dyn Operator) -> Vec<(LogicalType, i64, Value)> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            let schema = chunk.column(1).unwrap().data_type().clone();
            for row in chunk.selected_indices() {
                rows.push((
                    schema.clone(),
                    chunk.column(0).unwrap().get_int64(row).unwrap(),
                    chunk.column(1).unwrap().get_value(row).unwrap(),
                ));
            }
        }
        rows
    }

    fn any_list_chunk(rows: &[(i64, i64)]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Any]);
        for &(key, value) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::List(vec![Value::Int64(value)].into()));
            builder.advance_row();
        }
        builder.finish()
    }

    fn typed_entity_chunk(entity_type: LogicalType, rows: &[(i64, i64)]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, entity_type]);
        for &(key, entity_id) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::Int64(entity_id));
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_typed_entity_rows(operator: &mut dyn Operator) -> Vec<(LogicalType, i64, i64)> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            let schema = chunk.column(1).unwrap().data_type().clone();
            for row in chunk.selected_indices() {
                rows.push((
                    schema.clone(),
                    chunk.column(0).unwrap().get_int64(row).unwrap(),
                    chunk
                        .column(1)
                        .unwrap()
                        .get_value(row)
                        .and_then(|value| value.as_int64())
                        .unwrap(),
                ));
            }
        }
        rows
    }

    #[test]
    fn top_k_preserves_mixed_row_provenance_for_selected_rows_and_reset() {
        let mut edge = typed_edge_list_chunk(&[(2, 200), (1, 100), (99, 9900)]);
        edge.set_selection(SelectionVector::from_predicate(3, |row| row != 2));
        let ordinary = ordinary_list_chunk(&[(3, 300), (0, 0)]);
        let expected = vec![
            (
                LogicalType::Any,
                0,
                Value::List(vec![Value::Int64(0)].into()),
            ),
            (
                LogicalType::List(Box::new(LogicalType::Edge)),
                1,
                Value::List(vec![Value::Int64(100)].into()),
            ),
            (
                LogicalType::List(Box::new(LogicalType::Edge)),
                2,
                Value::List(vec![Value::Int64(200)].into()),
            ),
        ];
        let mut top_k = TopKOperator::new(
            Box::new(MockOperator::new(vec![edge, ordinary])),
            vec![SortKey::ascending(0)],
            3,
            vec![LogicalType::Int64, LogicalType::Any],
        );

        assert_eq!(collect_mixed_list_rows(&mut top_k), expected);
        top_k.reset();
        assert_eq!(collect_mixed_list_rows(&mut top_k), expected);
    }

    #[test]
    fn top_k_keeps_tied_edge_and_ordinary_lists_stable_and_uninferred() {
        let edge = typed_edge_list_chunk(&[(7, 700)]);
        let ordinary = any_list_chunk(&[(7, 7)]);
        let mut top_k = TopKOperator::new(
            Box::new(MockOperator::new(vec![edge, ordinary])),
            vec![SortKey::ascending(0)],
            2,
            vec![LogicalType::Int64, LogicalType::Any],
        );

        let rows = collect_mixed_list_rows(&mut top_k);
        assert_eq!(rows[0].0, LogicalType::List(Box::new(LogicalType::Edge)));
        assert_eq!(rows[0].2, Value::List(vec![Value::Int64(700)].into()));
        assert_eq!(rows[1].0, LogicalType::Any);
        assert_eq!(rows[1].2, Value::List(vec![Value::Int64(7)].into()));
    }

    #[test]
    fn top_k_preserves_node_and_edge_provenance_for_selected_same_ids() {
        let mut node = typed_entity_chunk(LogicalType::Node, &[(2, 42), (9, 42)]);
        node.set_selection(SelectionVector::from_predicate(2, |row| row == 0));
        let edge = typed_entity_chunk(LogicalType::Edge, &[(1, 42)]);
        let mut top_k = TopKOperator::new(
            Box::new(MockOperator::new(vec![node, edge])),
            vec![SortKey::ascending(0)],
            2,
            vec![LogicalType::Int64, LogicalType::Any],
        );

        assert_eq!(
            collect_typed_entity_rows(&mut top_k),
            vec![(LogicalType::Edge, 1, 42), (LogicalType::Node, 2, 42)]
        );
    }
}
