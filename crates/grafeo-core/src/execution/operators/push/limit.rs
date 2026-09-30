//! Push-based limit operator.

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::selection::SelectionVector;

/// Push-based limit operator.
///
/// Passes through rows until the limit is reached, then signals early termination.
pub struct LimitPushOperator {
    /// Maximum number of rows to pass through.
    limit: usize,
    /// Number of rows passed through so far.
    passed: usize,
}

impl LimitPushOperator {
    /// Create a new limit operator.
    pub fn new(limit: usize) -> Self {
        Self { limit, passed: 0 }
    }

    /// Get the limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Get the number of rows passed through.
    pub fn passed(&self) -> usize {
        self.passed
    }

    /// Check if limit has been reached.
    pub fn is_exhausted(&self) -> bool {
        self.passed >= self.limit
    }
}

impl PushOperator for LimitPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if self.passed >= self.limit {
            // Already at limit, signal termination
            return Ok(false);
        }

        let chunk_len = chunk.len();
        let remaining = self.limit - self.passed;

        if chunk_len <= remaining {
            // Pass entire chunk
            self.passed += chunk_len;
            let should_continue = sink.consume(chunk)?;
            Ok(should_continue && self.passed < self.limit)
        } else {
            // Need to truncate chunk
            self.passed += remaining;

            // Select the first `remaining` physical rows from the active
            // selection, rather than treating selection positions as rows.
            let mut selection = SelectionVector::with_capacity(remaining);
            for row in chunk.selected_indices().take(remaining) {
                selection.push(row);
            }
            let truncated = chunk.filter(&selection);

            sink.consume(truncated)?;
            Ok(false) // Limit reached
        }
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        // Nothing to finalize
        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        // If limit is small, use small chunks to avoid processing extra data
        if self.limit < 256 {
            ChunkSizeHint::AtMost(self.limit)
        } else if self.limit < 1000 {
            ChunkSizeHint::Small
        } else {
            ChunkSizeHint::Default
        }
    }

    fn name(&self) -> &'static str {
        "LimitPush"
    }
}

/// Push-based skip operator.
///
/// Skips the first N rows, then passes through the rest.
pub struct SkipPushOperator {
    /// Number of rows to skip.
    skip: usize,
    /// Number of rows skipped so far.
    skipped: usize,
}

impl SkipPushOperator {
    /// Create a new skip operator.
    pub fn new(skip: usize) -> Self {
        Self { skip, skipped: 0 }
    }

    /// Get the skip count.
    pub fn skip(&self) -> usize {
        self.skip
    }

    /// Check if skip phase is complete.
    pub fn skip_complete(&self) -> bool {
        self.skipped >= self.skip
    }
}

impl PushOperator for SkipPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if self.skipped >= self.skip {
            // Skip phase complete, pass everything through
            return sink.consume(chunk);
        }

        let chunk_len = chunk.len();
        let remaining_to_skip = self.skip - self.skipped;

        if chunk_len <= remaining_to_skip {
            // Skip entire chunk
            self.skipped += chunk_len;
            Ok(true)
        } else {
            // Skip first `remaining_to_skip` rows, pass the rest
            self.skipped = self.skip;

            let mut selection = SelectionVector::with_capacity(chunk_len - remaining_to_skip);
            for row in chunk.selected_indices().skip(remaining_to_skip) {
                selection.push(row);
            }
            let passed = chunk.filter(&selection);

            sink.consume(passed)
        }
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SkipPush"
    }
}

/// Combined skip and limit operator.
///
/// Skips the first N rows, then passes through at most M rows.
pub struct SkipLimitPushOperator {
    skip: SkipPushOperator,
    limit: LimitPushOperator,
}

impl SkipLimitPushOperator {
    /// Create a new skip+limit operator.
    pub fn new(skip: usize, limit: usize) -> Self {
        Self {
            skip: SkipPushOperator::new(skip),
            limit: LimitPushOperator::new(limit),
        }
    }
}

impl PushOperator for SkipLimitPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if self.limit.is_exhausted() {
            return Ok(false);
        }

        if !self.skip.skip_complete() {
            // Still in skip phase
            // We need to handle partial skip
            let chunk_len = chunk.len();
            let remaining_to_skip = self.skip.skip - self.skip.skipped;

            if chunk_len <= remaining_to_skip {
                // Skip entire chunk
                self.skip.skipped += chunk_len;
                return Ok(true);
            }

            // Partial skip
            self.skip.skipped = self.skip.skip;
            let mut selection = SelectionVector::with_capacity(chunk_len - remaining_to_skip);
            for row in chunk.selected_indices().skip(remaining_to_skip) {
                selection.push(row);
            }
            let passed = chunk.filter(&selection);

            return self.limit.push(passed, sink);
        }

        // Skip complete, apply limit
        self.limit.push(chunk, sink)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        self.skip.finalize(sink)?;
        self.limit.finalize(sink)
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        self.limit.preferred_chunk_size()
    }

    fn name(&self) -> &'static str {
        "SkipLimitPush"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::selection::SelectionVector;
    use crate::execution::sink::CollectorSink;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::{LogicalType, Value};

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    fn create_typed_edge_list_chunk(rows: &[&[i64]]) -> DataChunk {
        let edge_list_type = LogicalType::List(Box::new(LogicalType::Edge));
        let mut column = ValueVector::with_type(edge_list_type);
        for ids in rows {
            let values = ids.iter().map(|&id| Value::Int64(id)).collect::<Vec<_>>();
            column.push_value(Value::List(values.into()));
        }
        DataChunk::new(vec![column])
    }

    #[test]
    fn limit_push_preserves_typed_edge_lists_when_truncating() {
        let mut limit = LimitPushOperator::new(2);
        let mut sink = CollectorSink::new();

        limit
            .push(
                create_typed_edge_list_chunk(&[&[11], &[22], &[33]]),
                &mut sink,
            )
            .unwrap();

        let chunks = sink.chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(11)].into()))
        );
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(22)].into()))
        );
    }

    #[test]
    fn limit_push_preserves_preselected_physical_rows_when_truncating() {
        let mut chunk = create_typed_edge_list_chunk(&[&[10], &[11], &[12], &[13]]);
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));

        let mut limit = LimitPushOperator::new(1);
        let mut sink = CollectorSink::new();
        limit.push(chunk, &mut sink).unwrap();

        assert_eq!(sink.row_count(), 1);
        assert_eq!(
            sink.chunks()[0].column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(11)].into()))
        );
    }

    #[test]
    fn skip_limit_push_preserves_typed_edge_lists_across_partial_boundaries() {
        let mut op = SkipLimitPushOperator::new(1, 2);
        let mut sink = CollectorSink::new();

        op.push(
            create_typed_edge_list_chunk(&[&[11], &[22], &[33], &[44]]),
            &mut sink,
        )
        .unwrap();

        let chunks = sink.chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(22)].into()))
        );
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(33)].into()))
        );
    }

    #[test]
    fn skip_push_preserves_preselected_physical_rows() {
        let mut chunk = create_typed_edge_list_chunk(&[&[10], &[11], &[12], &[13]]);
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));

        let mut skip = SkipPushOperator::new(1);
        let mut sink = CollectorSink::new();
        skip.push(chunk, &mut sink).unwrap();

        assert_eq!(sink.row_count(), 1);
        assert_eq!(
            sink.chunks()[0].column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(13)].into()))
        );
    }

    #[test]
    fn skip_limit_push_preserves_preselected_physical_rows() {
        let mut chunk = create_typed_edge_list_chunk(&[&[10], &[11], &[12], &[13]]);
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));

        let mut op = SkipLimitPushOperator::new(1, 1);
        let mut sink = CollectorSink::new();
        op.push(chunk, &mut sink).unwrap();

        assert_eq!(sink.row_count(), 1);
        assert_eq!(
            sink.chunks()[0].column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(13)].into()))
        );
    }

    #[test]
    fn test_limit_under_limit() {
        let mut limit = LimitPushOperator::new(10);
        let mut sink = CollectorSink::new();

        limit
            .push(create_test_chunk(&[1, 2, 3]), &mut sink)
            .unwrap();
        limit.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3);
        assert!(!limit.is_exhausted());
    }

    #[test]
    fn test_limit_exact_limit() {
        let mut limit = LimitPushOperator::new(5);
        let mut sink = CollectorSink::new();

        let should_continue = limit
            .push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        limit.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5);
        assert!(!should_continue);
        assert!(limit.is_exhausted());
    }

    #[test]
    fn test_limit_over_limit() {
        let mut limit = LimitPushOperator::new(3);
        let mut sink = CollectorSink::new();

        let should_continue = limit
            .push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        limit.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3);
        assert!(!should_continue);
    }

    #[test]
    fn test_limit_multiple_chunks() {
        let mut limit = LimitPushOperator::new(5);
        let mut sink = CollectorSink::new();

        limit.push(create_test_chunk(&[1, 2]), &mut sink).unwrap();
        limit.push(create_test_chunk(&[3, 4]), &mut sink).unwrap();
        let should_continue = limit
            .push(create_test_chunk(&[5, 6, 7]), &mut sink)
            .unwrap();
        limit.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5);
        assert!(!should_continue);
    }

    #[test]
    fn test_skip_under_skip() {
        let mut skip = SkipPushOperator::new(10);
        let mut sink = CollectorSink::new();

        skip.push(create_test_chunk(&[1, 2, 3]), &mut sink).unwrap();
        skip.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 0);
        assert!(!skip.skip_complete());
    }

    #[test]
    fn test_skip_exact_skip() {
        let mut skip = SkipPushOperator::new(3);
        let mut sink = CollectorSink::new();

        skip.push(create_test_chunk(&[1, 2, 3]), &mut sink).unwrap();
        skip.push(create_test_chunk(&[4, 5, 6]), &mut sink).unwrap();
        skip.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3); // 4, 5, 6
        assert!(skip.skip_complete());
    }

    #[test]
    fn test_skip_partial() {
        let mut skip = SkipPushOperator::new(2);
        let mut sink = CollectorSink::new();

        skip.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        skip.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3); // 3, 4, 5
    }

    #[test]
    fn test_skip_limit_combined() {
        let mut op = SkipLimitPushOperator::new(2, 3);
        let mut sink = CollectorSink::new();

        op.push(create_test_chunk(&[1, 2, 3, 4, 5, 6, 7]), &mut sink)
            .unwrap();
        op.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3); // 3, 4, 5 (skip 1,2; limit 3)
    }
}
