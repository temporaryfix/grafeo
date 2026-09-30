//! Limit and Skip operators for result pagination.
//!
//! This module provides:
//! - `LimitOperator`: Limits the number of output rows
//! - `SkipOperator`: Skips a number of input rows
//! - `LimitSkipOperator`: Combined LIMIT and OFFSET/SKIP

use grafeo_common::types::{LogicalType, Value};

use super::{Operator, OperatorPipelineDecomposition, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

fn schema_for_chunk(output_schema: &[LogicalType], chunk: &DataChunk) -> Vec<LogicalType> {
    output_schema
        .iter()
        .enumerate()
        .map(|(column, configured)| {
            if *configured == LogicalType::Any {
                chunk.column(column).map_or(LogicalType::Any, |source| {
                    if matches!(source.data_type(), LogicalType::Node | LogicalType::Edge)
                        || matches!(
                            source.data_type(),
                            LogicalType::List(element) if element.as_ref() == &LogicalType::Edge
                        )
                    {
                        source.data_type().clone()
                    } else {
                        LogicalType::Any
                    }
                })
            } else {
                configured.clone()
            }
        })
        .collect()
}

/// Limit operator.
///
/// Returns at most `limit` rows from the input.
pub struct LimitOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Maximum number of rows to return.
    limit: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Number of rows returned so far.
    returned: usize,
    /// Whether the child must run to completion even after the output limit.
    exhaustive: bool,
    /// Whether the child has reported terminal exhaustion.
    child_exhausted: bool,
    /// Query identity and cancellation inherited from the execution owner.
    resources: Option<crate::execution::QueryResourceContext>,
}

impl LimitOperator {
    /// Creates a new limit operator.
    pub fn new(child: Box<dyn Operator>, limit: usize, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            limit,
            output_schema,
            returned: 0,
            exhaustive: false,
            child_exhausted: false,
            resources: None,
        }
    }

    /// Creates a limit that always exhausts its child.
    ///
    /// This variant is used when the input contains effects or validation that
    /// must complete even when the visible limit is zero or is reached before
    /// the input ends. It deliberately remains a pull-pipeline boundary: an
    /// ordinary push limit is allowed to terminate its source early.
    pub fn new_exhaustive(
        child: Box<dyn Operator>,
        limit: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            limit,
            output_schema,
            returned: 0,
            exhaustive: true,
            child_exhausted: false,
            resources: None,
        }
    }

    /// Decomposes this operator for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, usize) {
        (self.child, self.limit)
    }

    fn check_cancelled(&self) -> Result<(), super::OperatorError> {
        self.resources
            .as_ref()
            .map_or(
                Ok(()),
                crate::execution::QueryResourceContext::check_cancelled,
            )
            .map_err(Into::into)
    }

    fn exhaust_child(&mut self) -> Result<(), super::OperatorError> {
        while !self.child_exhausted {
            self.check_cancelled()?;
            if self.child.next()?.is_none() {
                self.child_exhausted = true;
            }
        }
        self.check_cancelled()
    }
}

impl Operator for LimitOperator {
    fn next(&mut self) -> OperatorResult {
        if self.child_exhausted {
            return Ok(None);
        }
        if self.returned >= self.limit {
            if self.exhaustive {
                self.exhaust_child()?;
            }
            return Ok(None);
        }

        let remaining = self.limit - self.returned;

        loop {
            self.check_cancelled()?;
            let Some(chunk) = self.child.next()? else {
                self.child_exhausted = true;
                self.check_cancelled()?;
                return Ok(None);
            };

            let row_count = chunk.row_count();
            if row_count == 0 {
                continue;
            }

            if row_count <= remaining {
                // Return entire chunk
                self.returned += row_count;
                if self.exhaustive && self.returned >= self.limit {
                    self.exhaust_child()?;
                }
                return Ok(Some(chunk));
            }

            // Return partial chunk
            let output_schema = schema_for_chunk(&self.output_schema, &chunk);
            let mut builder = DataChunkBuilder::with_capacity(&output_schema, remaining);

            let mut count = 0;
            for row in chunk.selected_indices() {
                if count >= remaining {
                    break;
                }

                for col_idx in 0..chunk.column_count() {
                    if let (Some(src_col), Some(dst_col)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(value) = src_col.get_value(row) {
                            dst_col.push_value(value);
                        } else {
                            dst_col.push_value(Value::Null);
                        }
                    }
                }
                builder.advance_row();
                count += 1;
            }

            self.returned += count;
            if self.exhaustive {
                self.exhaust_child()?;
            }
            return Ok(Some(builder.finish()));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.returned = 0;
        self.child_exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Limit"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)?;
        if self.exhaustive {
            self.resources = Some(resources.clone());
        }
        Ok(())
    }

    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        _resources: &crate::execution::QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, crate::execution::QueryResourceContextError> {
        if self.exhaustive {
            return Ok(OperatorPipelineDecomposition::boundary(self));
        }
        let (child, count) = (*self).into_parts();
        Ok(OperatorPipelineDecomposition::unary(
            child,
            Box::new(super::push::LimitPushOperator::new(count)),
        ))
    }
}

/// Terminal operator that exhausts its child while discarding every row.
///
/// Unlike `LimitOperator` with a zero limit, a drain must pull the child to
/// completion so side effects performed by mutation operators are executed.
pub struct DrainOperator {
    child: Box<dyn Operator>,
    drained: bool,
}

impl DrainOperator {
    /// Creates a terminal drain over `child`.
    pub fn new(child: Box<dyn Operator>) -> Self {
        Self {
            child,
            drained: false,
        }
    }
}

impl Operator for DrainOperator {
    fn next(&mut self) -> OperatorResult {
        if self.drained {
            return Ok(None);
        }

        while self.child.next()?.is_some() {}
        self.drained = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.child.reset();
        self.drained = false;
    }

    fn name(&self) -> &'static str {
        "Drain"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }
}

/// Skip operator.
///
/// Skips the first `skip` rows from the input.
pub struct SkipOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Number of rows to skip.
    skip: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Number of rows skipped so far.
    skipped: usize,
}

impl SkipOperator {
    /// Creates a new skip operator.
    pub fn new(child: Box<dyn Operator>, skip: usize, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            skip,
            output_schema,
            skipped: 0,
        }
    }
}

impl Operator for SkipOperator {
    fn next(&mut self) -> OperatorResult {
        // Skip rows until we've skipped enough
        while self.skipped < self.skip {
            let Some(chunk) = self.child.next()? else {
                return Ok(None);
            };

            let row_count = chunk.row_count();
            let to_skip = (self.skip - self.skipped).min(row_count);

            if to_skip >= row_count {
                // Skip entire chunk
                self.skipped += row_count;
                continue;
            }

            // Skip partial chunk
            self.skipped = self.skip;

            let output_schema = schema_for_chunk(&self.output_schema, &chunk);
            let mut builder = DataChunkBuilder::with_capacity(&output_schema, row_count - to_skip);

            let rows: Vec<usize> = chunk.selected_indices().collect();
            for &row in rows.iter().skip(to_skip) {
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src_col), Some(dst_col)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(value) = src_col.get_value(row) {
                            dst_col.push_value(value);
                        } else {
                            dst_col.push_value(Value::Null);
                        }
                    }
                }
                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }

        // After skipping, just pass through
        self.child.next()
    }

    fn reset(&mut self) {
        self.child.reset();
        self.skipped = 0;
    }

    fn name(&self) -> &'static str {
        "Skip"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }
}

/// Combined Limit and Skip operator.
///
/// Equivalent to OFFSET skip LIMIT limit.
pub struct LimitSkipOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Number of rows to skip.
    skip: usize,
    /// Maximum number of rows to return.
    limit: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Number of rows skipped so far.
    skipped: usize,
    /// Number of rows returned so far.
    returned: usize,
}

impl LimitSkipOperator {
    /// Creates a new limit/skip operator.
    pub fn new(
        child: Box<dyn Operator>,
        skip: usize,
        limit: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            skip,
            limit,
            output_schema,
            skipped: 0,
            returned: 0,
        }
    }
}

impl Operator for LimitSkipOperator {
    fn next(&mut self) -> OperatorResult {
        // Check if we've returned enough
        if self.returned >= self.limit {
            return Ok(None);
        }

        loop {
            let Some(chunk) = self.child.next()? else {
                return Ok(None);
            };

            let row_count = chunk.row_count();
            if row_count == 0 {
                continue;
            }

            let rows: Vec<usize> = chunk.selected_indices().collect();
            let mut start_idx = 0;

            // Skip rows if needed
            if self.skipped < self.skip {
                let to_skip = (self.skip - self.skipped).min(row_count);
                if to_skip >= row_count {
                    self.skipped += row_count;
                    continue;
                }
                self.skipped = self.skip;
                start_idx = to_skip;
            }

            // Calculate how many rows to return
            let remaining_in_chunk = row_count - start_idx;
            let remaining_to_return = self.limit - self.returned;
            let to_return = remaining_in_chunk.min(remaining_to_return);

            if to_return == 0 {
                return Ok(None);
            }

            let output_schema = schema_for_chunk(&self.output_schema, &chunk);
            let mut builder = DataChunkBuilder::with_capacity(&output_schema, to_return);

            for &row in rows.iter().skip(start_idx).take(to_return) {
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src_col), Some(dst_col)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(value) = src_col.get_value(row) {
                            dst_col.push_value(value);
                        } else {
                            dst_col.push_value(Value::Null);
                        }
                    }
                }
                builder.advance_row();
            }

            self.returned += to_return;
            return Ok(Some(builder.finish()));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.skipped = 0;
        self.returned = 0;
    }

    fn name(&self) -> &'static str {
        "LimitSkip"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::operators::OperatorError;
    use crate::execution::selection::SelectionVector;
    use crate::execution::vector::ValueVector;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
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
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn create_numbered_chunk(values: &[i64]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for &v in values {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        builder.finish()
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

    fn create_typed_node_edge_chunk() -> DataChunk {
        let mut nodes = ValueVector::with_type(LogicalType::Node);
        let mut edges = ValueVector::with_type(LogicalType::Edge);
        for id in 10..14 {
            nodes.push_value(Value::Int64(id));
            edges.push_value(Value::Int64(id + 10));
        }
        let mut chunk = DataChunk::new(vec![nodes, edges]);
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));
        chunk
    }

    struct ReplayOperator {
        chunk: DataChunk,
        served: bool,
    }

    impl Operator for ReplayOperator {
        fn next(&mut self) -> OperatorResult {
            if self.served {
                Ok(None)
            } else {
                self.served = true;
                Ok(Some(self.chunk.clone()))
            }
        }

        fn reset(&mut self) {
            self.served = false;
        }

        fn name(&self) -> &'static str {
            "Replay"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn limit_preserves_typed_edge_lists_for_partial_output_and_reset() {
        let chunk = create_typed_edge_list_chunk(&[&[11], &[22], &[33]]);
        let mut limit = LimitOperator::new(
            Box::new(ReplayOperator {
                chunk,
                served: false,
            }),
            2,
            vec![LogicalType::Any],
        );

        let result = limit.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(11)].into()))
        );
        assert_eq!(
            result.column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(22)].into()))
        );

        limit.reset();
        let repeated = limit.next().unwrap().unwrap();
        assert_eq!(
            repeated.column(0).unwrap().data_type(),
            result.column(0).unwrap().data_type()
        );
        assert_eq!(
            repeated.column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(22)].into()))
        );
    }

    #[test]
    fn skip_preserves_typed_edge_lists_for_partial_output() {
        let chunk = create_typed_edge_list_chunk(&[&[11], &[22], &[33]]);
        let mut skip = SkipOperator::new(
            Box::new(MockOperator::new(vec![chunk])),
            1,
            vec![LogicalType::Any],
        );

        let result = skip.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(22)].into()))
        );
        assert_eq!(
            result.column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(33)].into()))
        );
    }

    #[test]
    fn limit_skip_preserves_typed_edge_lists_for_partial_output() {
        let chunk = create_typed_edge_list_chunk(&[&[11], &[22], &[33], &[44]]);
        let mut op = LimitSkipOperator::new(
            Box::new(MockOperator::new(vec![chunk])),
            1,
            2,
            vec![LogicalType::Any],
        );

        let result = op.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(22)].into()))
        );
        assert_eq!(
            result.column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(33)].into()))
        );
    }

    #[test]
    fn partial_entity_modifiers_preserve_node_edge_schema_for_selection_and_reset() {
        let mut limit = LimitOperator::new(
            Box::new(ReplayOperator {
                chunk: create_typed_node_edge_chunk(),
                served: false,
            }),
            1,
            vec![LogicalType::Any, LogicalType::Any],
        );
        let result = limit.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column(0).unwrap().data_type(), &LogicalType::Node);
        assert_eq!(result.column(1).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Int64(11))
        );
        assert_eq!(
            result.column(1).unwrap().get_value(0),
            Some(Value::Int64(21))
        );
        limit.reset();
        let replay = limit.next().unwrap().unwrap();
        assert_eq!(replay.column(0).unwrap().data_type(), &LogicalType::Node);
        assert_eq!(replay.column(1).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(
            replay.column(0).unwrap().get_value(0),
            Some(Value::Int64(11))
        );
        assert_eq!(
            replay.column(1).unwrap().get_value(0),
            Some(Value::Int64(21))
        );

        let mut skip = SkipOperator::new(
            Box::new(MockOperator::new(vec![create_typed_node_edge_chunk()])),
            1,
            vec![LogicalType::Any, LogicalType::Any],
        );
        let skipped = skip.next().unwrap().unwrap();
        assert_eq!(skipped.column(0).unwrap().data_type(), &LogicalType::Node);
        assert_eq!(skipped.column(1).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(
            skipped.column(0).unwrap().get_value(0),
            Some(Value::Int64(13))
        );
        assert_eq!(
            skipped.column(1).unwrap().get_value(0),
            Some(Value::Int64(23))
        );

        let mut combined = LimitSkipOperator::new(
            Box::new(MockOperator::new(vec![create_typed_node_edge_chunk()])),
            1,
            1,
            vec![LogicalType::Any, LogicalType::Any],
        );
        let combined = combined.next().unwrap().unwrap();
        assert_eq!(combined.column(0).unwrap().data_type(), &LogicalType::Node);
        assert_eq!(combined.column(1).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(
            combined.column(0).unwrap().get_value(0),
            Some(Value::Int64(13))
        );
        assert_eq!(
            combined.column(1).unwrap().get_value(0),
            Some(Value::Int64(23))
        );
    }

    #[test]
    fn test_limit() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3, 4, 5])]);

        let mut limit = LimitOperator::new(Box::new(mock), 3, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = limit.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_limit_larger_than_input() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3])]);

        let mut limit = LimitOperator::new(Box::new(mock), 10, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = limit.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn exhaustive_limit_zero_drains_while_ordinary_limit_zero_does_not() {
        let ordinary_calls = Arc::new(AtomicUsize::new(0));
        let ordinary = DrainProbe {
            chunks: 3,
            position: 0,
            next_calls: Arc::clone(&ordinary_calls),
            reset_calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut ordinary = LimitOperator::new(Box::new(ordinary), 0, Vec::new());
        assert!(ordinary.next().unwrap().is_none());
        assert_eq!(ordinary_calls.load(Ordering::SeqCst), 0);

        let exhaustive_calls = Arc::new(AtomicUsize::new(0));
        let exhaustive = DrainProbe {
            chunks: 3,
            position: 0,
            next_calls: Arc::clone(&exhaustive_calls),
            reset_calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut exhaustive = LimitOperator::new_exhaustive(Box::new(exhaustive), 0, Vec::new());
        assert!(exhaustive.next().unwrap().is_none());
        assert_eq!(exhaustive_calls.load(Ordering::SeqCst), 4);

        assert!(exhaustive.next().unwrap().is_none());
        assert_eq!(exhaustive_calls.load(Ordering::SeqCst), 4);
    }

    struct LateErrorProbe {
        position: usize,
    }

    impl Operator for LateErrorProbe {
        fn next(&mut self) -> OperatorResult {
            self.position += 1;
            match self.position {
                1 => Ok(Some(create_numbered_chunk(&[1]))),
                _ => Err(OperatorError::Execution(
                    "late exhaustive-limit failure".to_string(),
                )),
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "LateErrorProbe"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn exhaustive_limit_surfaces_late_failure_before_its_terminal_row() {
        let mut limit = LimitOperator::new_exhaustive(
            Box::new(LateErrorProbe { position: 0 }),
            1,
            vec![LogicalType::Int64],
        );

        assert!(matches!(
            limit.next(),
            Err(OperatorError::Execution(message))
                if message == "late exhaustive-limit failure"
        ));
    }

    struct DrainProbe {
        chunks: usize,
        position: usize,
        next_calls: Arc<AtomicUsize>,
        reset_calls: Arc<AtomicUsize>,
    }

    impl Operator for DrainProbe {
        fn next(&mut self) -> OperatorResult {
            self.next_calls.fetch_add(1, Ordering::SeqCst);
            if self.position == self.chunks {
                return Ok(None);
            }
            self.position += 1;
            Ok(Some(DataChunk::empty()))
        }

        fn reset(&mut self) {
            self.position = 0;
            self.reset_calls.fetch_add(1, Ordering::SeqCst);
        }

        fn name(&self) -> &'static str {
            "DrainProbe"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn test_drain_exhausts_child_once_and_reset_rearms_it() {
        let next_calls = Arc::new(AtomicUsize::new(0));
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let probe = DrainProbe {
            chunks: 3,
            position: 0,
            next_calls: Arc::clone(&next_calls),
            reset_calls: Arc::clone(&reset_calls),
        };
        let mut drain = DrainOperator::new(Box::new(probe));

        assert!(drain.next().unwrap().is_none());
        assert_eq!(next_calls.load(Ordering::SeqCst), 4);

        // A completed drain is terminal and must not poll its child again.
        assert!(drain.next().unwrap().is_none());
        assert_eq!(next_calls.load(Ordering::SeqCst), 4);

        drain.reset();
        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);
        assert!(drain.next().unwrap().is_none());
        assert_eq!(next_calls.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_skip() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3, 4, 5])]);

        let mut skip = SkipOperator::new(Box::new(mock), 2, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = skip.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![3, 4, 5]);
    }

    #[test]
    fn test_skip_all() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3])]);

        let mut skip = SkipOperator::new(Box::new(mock), 5, vec![LogicalType::Int64]);

        let result = skip.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_limit_skip_combined() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        ])]);

        let mut op = LimitSkipOperator::new(
            Box::new(mock),
            3, // Skip first 3
            4, // Take next 4
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_limit_across_chunks() {
        let mock = MockOperator::new(vec![
            create_numbered_chunk(&[1, 2]),
            create_numbered_chunk(&[3, 4]),
            create_numbered_chunk(&[5, 6]),
        ]);

        let mut limit = LimitOperator::new(Box::new(mock), 5, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = limit.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_skip_across_chunks() {
        let mock = MockOperator::new(vec![
            create_numbered_chunk(&[1, 2]),
            create_numbered_chunk(&[3, 4]),
            create_numbered_chunk(&[5, 6]),
        ]);

        let mut skip = SkipOperator::new(Box::new(mock), 3, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = skip.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![4, 5, 6]);
    }

    #[test]
    fn test_limit_into_parts() {
        let child = Box::new(MockOperator::new(vec![]));
        let limit = LimitOperator::new(child, 42, vec![LogicalType::Int64]);
        let (_, limit_value) = limit.into_parts();
        assert_eq!(limit_value, 42);
    }

    #[test]
    fn test_limit_into_any() {
        let child = Box::new(MockOperator::new(vec![]));
        let limit: Box<dyn Operator> = Box::new(LimitOperator::new(child, 10, vec![]));
        let any = limit.into_any();
        assert!(any.downcast::<LimitOperator>().is_ok());
    }
}
