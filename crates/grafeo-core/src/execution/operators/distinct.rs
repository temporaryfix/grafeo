//! Exact ordinary DISTINCT with shared pull/push state.
use super::distinct_state::ExactDistinctState;
use super::{Operator, OperatorError, OperatorPipelineDecomposition, OperatorResult};
#[cfg(test)]
use crate::execution::DataChunk;
use crate::execution::{QueryResourceContext, QueryResourceContextError};
use grafeo_common::types::LogicalType;
#[cfg(test)]
use grafeo_common::types::Value;

/// Removes duplicate keys and retains their earliest original witness.
pub struct DistinctOperator {
    child: Box<dyn Operator>,
    state: parking_lot::Mutex<ExactDistinctState>,
    consumed: bool,
    reset_error: Option<OperatorError>,
}
impl DistinctOperator {
    /// Creates DISTINCT over every input column.
    pub fn new(child: Box<dyn Operator>, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            state: parking_lot::Mutex::new(ExactDistinctState::new(None, output_schema)),
            consumed: false,
            reset_error: None,
        }
    }
    /// Creates DISTINCT over selected key columns, preserving complete rows.
    pub fn on_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            state: parking_lot::Mutex::new(ExactDistinctState::new(Some(columns), output_schema)),
            consumed: false,
            reset_error: None,
        }
    }
    /// Decomposes an unexecuted wrapper for native push conversion.
    pub fn into_parts(mut self) -> (Box<dyn Operator>, Option<Vec<usize>>) {
        (self.child, self.state.get_mut().take_columns())
    }
}
impl Operator for DistinctOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(error) = self.reset_error.take() {
            return Err(error);
        }
        if !self.consumed {
            self.consumed = true;
            loop {
                let chunk = match self.child.next() {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break,
                    Err(error) => return Err(self.state.get_mut().fail(error)),
                };
                for position in 0..chunk.row_count() {
                    if let Some(row) = chunk
                        .selection()
                        .map_or(Some(position), |selection| selection.get(position))
                    {
                        self.state.get_mut().ingest(&chunk, row)?;
                    }
                }
            }
            self.state.get_mut().finish_input()?;
        }
        self.state.get_mut().next_chunk()
    }
    fn reset(&mut self) {
        self.child.reset();
        self.reset_error = self.state.get_mut().reset().err();
        self.consumed = false;
    }
    fn name(&self) -> &'static str {
        "Distinct"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        self.child.install_resource_context(resources)?;
        self.state.get_mut().install_resource_context(resources)
    }
    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        resources: &QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, QueryResourceContextError> {
        let (child, columns) = (*self).into_parts();
        let push =
            super::push::DistinctPushOperator::with_resource_context(columns, resources.clone())?;
        Ok(OperatorPipelineDecomposition::unary(child, Box::new(push)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::selection::SelectionVector;
    use crate::execution::vector::ValueVector;

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

    fn create_chunk_with_duplicates() -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);

        let data = [
            (1i64, "a"),
            (2, "b"),
            (1, "a"), // Duplicate
            (3, "c"),
            (2, "b"), // Duplicate
            (1, "a"), // Duplicate
        ];

        for (num, text) in data {
            builder.column_mut(0).unwrap().push_int64(num);
            builder.column_mut(1).unwrap().push_string(text);
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

    fn create_typed_entity_chunk(entity_type: LogicalType, rows: &[(i64, i64)]) -> DataChunk {
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
    fn test_distinct_preserves_typed_edge_lists_for_selected_rows_when_output_is_any() {
        let mut chunk = create_typed_edge_list_chunk(&[&[11], &[99], &[22]]);
        chunk.set_selection(SelectionVector::from_predicate(3, |row| row != 1));

        let mut distinct = DistinctOperator::new(
            Box::new(MockOperator::new(vec![chunk])),
            vec![LogicalType::Any],
        );
        let result = distinct.next().unwrap().unwrap();

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
    }

    #[test]
    fn test_distinct_preserves_node_and_edge_provenance_for_selected_mixed_rows() {
        let mut node = create_typed_entity_chunk(LogicalType::Node, &[(1, 42), (9, 42)]);
        node.set_selection(SelectionVector::from_predicate(2, |row| row == 0));
        let edge = create_typed_entity_chunk(LogicalType::Edge, &[(2, 42)]);

        let mut distinct = DistinctOperator::new(
            Box::new(MockOperator::new(vec![node, edge])),
            vec![LogicalType::Int64, LogicalType::Any],
        );

        assert_eq!(
            collect_typed_entity_rows(&mut distinct),
            vec![(LogicalType::Node, 1, 42), (LogicalType::Edge, 2, 42)]
        );
    }

    #[test]
    fn test_distinct_keeps_all_rows_from_one_oversized_chunk_and_reset() {
        let physical_row_count = 5000;
        let mut builder =
            DataChunkBuilder::with_capacity(&[LogicalType::Int64], physical_row_count);
        for value in 0..physical_row_count {
            builder
                .column_mut(0)
                .unwrap()
                .push_int64(i64::try_from(value).unwrap());
            builder.advance_row();
        }
        let mut chunk = builder.finish();
        chunk.set_selection(SelectionVector::from_predicate(physical_row_count, |row| {
            row % 2 == 1
        }));
        let mut distinct = DistinctOperator::new(
            Box::new(ReplayOperator {
                chunk,
                served: false,
            }),
            vec![LogicalType::Int64],
        );

        let collect_rows = |operator: &mut DistinctOperator| {
            let mut values = Vec::new();
            while let Some(chunk) = operator.next().unwrap() {
                for row in chunk.selected_indices() {
                    values.push(chunk.column(0).unwrap().get_int64(row).unwrap());
                }
            }
            values
        };
        let first = collect_rows(&mut distinct);
        assert_eq!(
            first,
            (0..physical_row_count)
                .filter(|row| row % 2 == 1)
                .map(|row| i64::try_from(row).unwrap())
                .collect::<Vec<_>>()
        );

        distinct.reset();
        assert_eq!(collect_rows(&mut distinct), first);
    }

    #[test]
    fn test_distinct_all_columns() {
        let mock = MockOperator::new(vec![create_chunk_with_duplicates()]);

        let mut distinct = DistinctOperator::new(
            Box::new(mock),
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                let text = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                results.push((num, text));
            }
        }

        // Should have 3 unique rows
        assert_eq!(results.len(), 3);

        // Sort for consistent comparison
        results.sort();
        assert_eq!(
            results,
            vec![
                (1, "a".to_string()),
                (2, "b".to_string()),
                (3, "c".to_string()),
            ]
        );
    }

    #[test]
    fn test_distinct_single_column() {
        let mock = MockOperator::new(vec![create_chunk_with_duplicates()]);

        let mut distinct = DistinctOperator::on_columns(
            Box::new(mock),
            vec![0], // Only consider first column
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        // Should have 3 unique values in column 0
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_distinct_across_chunks() {
        // Create two chunks with overlapping values
        let mut builder1 = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in [1, 2, 3] {
            builder1.column_mut(0).unwrap().push_int64(i);
            builder1.advance_row();
        }

        let mut builder2 = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in [2, 3, 4] {
            builder2.column_mut(0).unwrap().push_int64(i);
            builder2.advance_row();
        }

        let mock = MockOperator::new(vec![builder1.finish(), builder2.finish()]);

        let mut distinct = DistinctOperator::new(Box::new(mock), vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        // Should have 4 unique values: 1, 2, 3, 4
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_distinct_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = DistinctOperator::new(Box::new(mock), vec![LogicalType::Int64]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<DistinctOperator>().is_ok());
    }

    #[test]
    fn test_distinct_into_parts() {
        let mock = MockOperator::new(vec![]);
        let op = DistinctOperator::on_columns(
            Box::new(mock),
            vec![0, 2],
            vec![LogicalType::Int64, LogicalType::String, LogicalType::Int64],
        );
        let (mut child, distinct_columns) = op.into_parts();
        assert_eq!(distinct_columns, Some(vec![0, 2]));
        assert!(child.next().unwrap().is_none());
    }

    #[test]
    fn test_distinct_into_parts_all_columns() {
        let mock = MockOperator::new(vec![]);
        let op = DistinctOperator::new(Box::new(mock), vec![LogicalType::Int64]);
        let (_child, distinct_columns) = op.into_parts();
        assert!(distinct_columns.is_none());
    }
}
