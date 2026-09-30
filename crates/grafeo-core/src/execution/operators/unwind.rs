//! Unwind operator for expanding lists into individual rows.

use super::{Operator, OperatorResult};
use crate::execution::chunk::{DataChunk, DataChunkBuilder};
use grafeo_common::types::{LogicalType, Value};

/// Unwind operator that expands a list column into individual rows.
///
/// For each input row, if the list column contains N elements, this operator
/// produces N output rows, each with one element from the list.
pub struct UnwindOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Index of the column containing the list to unwind.
    list_col_idx: usize,
    /// Name of the new variable for the unwound elements.
    variable_name: String,
    /// Schema of output columns (inherited from input plus the new column).
    output_schema: Vec<LogicalType>,
    /// Whether to emit a 1-based ORDINALITY column.
    emit_ordinality: bool,
    /// Whether to emit a 0-based OFFSET column.
    emit_offset: bool,
    /// Current input chunk being processed.
    current_chunk: Option<DataChunk>,
    /// Current row index within the chunk.
    current_row: usize,
    /// Current index within the list being unwound.
    current_list_idx: usize,
    /// Current list being unwound.
    current_list: Option<Vec<Value>>,
}

impl UnwindOperator {
    /// Creates a new unwind operator.
    ///
    /// # Arguments
    /// * `child` - The input operator
    /// * `list_col_idx` - The column index containing the list to unwind
    /// * `variable_name` - The name of the new variable
    /// * `output_schema` - An inherited input-column prefix, then the element,
    ///   then any ordinality/offset columns. An empty prefix projects only the
    ///   element; retaining all input columns also retains the original list.
    /// * `emit_ordinality` - Whether to emit a 1-based index column
    /// * `emit_offset` - Whether to emit a 0-based index column
    pub fn new(
        child: Box<dyn Operator>,
        list_col_idx: usize,
        variable_name: String,
        output_schema: Vec<LogicalType>,
        emit_ordinality: bool,
        emit_offset: bool,
    ) -> Self {
        Self {
            child,
            list_col_idx,
            variable_name,
            output_schema,
            emit_ordinality,
            emit_offset,
            current_chunk: None,
            current_row: 0,
            current_list_idx: 0,
            current_list: None,
        }
    }

    /// Returns the variable name for the unwound elements.
    #[must_use]
    pub fn variable_name(&self) -> &str {
        &self.variable_name
    }

    /// Advances to the next row or fetches the next chunk.
    fn advance(&mut self) -> OperatorResult {
        loop {
            // If we have a current list, try to get the next element
            if let Some(list) = &self.current_list
                && self.current_list_idx < list.len()
            {
                // Still have elements in the current list
                return Ok(Some(self.emit_row()?));
            }

            // Need to move to the next row
            self.current_list_idx = 0;
            self.current_list = None;

            // Get the next chunk if needed
            if self.current_chunk.is_none() {
                self.current_chunk = self.child.next()?;
                self.current_row = 0;
                if self.current_chunk.is_none() {
                    return Ok(None); // No more data
                }
            }

            let chunk = self
                .current_chunk
                .as_ref()
                .expect("current_chunk is Some: checked above");

            // Find the next row with a list value
            while self.current_row < chunk.row_count() {
                if let Some(col) = chunk.column(self.list_col_idx)
                    && let Some(value) = col.get_value(self.current_row)
                {
                    // Extract list elements from either Value::List or Value::Vector
                    let list = match value {
                        Value::List(list_arc) => list_arc.iter().cloned().collect::<Vec<Value>>(),
                        Value::Vector(vec_arc) => {
                            vec_arc.iter().map(|f| Value::Float64(*f as f64)).collect()
                        }
                        _ => {
                            self.current_row += 1;
                            continue;
                        }
                    };
                    if !list.is_empty() {
                        self.current_list = Some(list);
                        return Ok(Some(self.emit_row()?));
                    }
                }
                self.current_row += 1;
            }

            // Exhausted current chunk, get next one
            self.current_chunk = None;
        }
    }

    /// Emits a single row with the current list element.
    fn emit_row(&mut self) -> Result<DataChunk, super::OperatorError> {
        let chunk = self.current_chunk.as_ref().ok_or_else(|| {
            super::OperatorError::Execution("UNWIND input chunk is absent".into())
        })?;
        let list = self.current_list.as_ref().ok_or_else(|| {
            super::OperatorError::Execution("UNWIND active list is absent".into())
        })?;
        let element = list.get(self.current_list_idx).cloned().ok_or_else(|| {
            super::OperatorError::Execution("UNWIND active list position is absent".into())
        })?;

        // Build output row: copy all columns from input + add the unwound element
        let mut builder = DataChunkBuilder::new(&self.output_schema);

        // The output schema declares its inherited input prefix followed by
        // the element and optional ordinality/offset. Preserve every declared
        // input value, including the original list: the planner appends the
        // element, it does not replace the list column with a sparse hole.
        let extra_cols = usize::from(self.emit_ordinality) + usize::from(self.emit_offset);
        let element_col_idx = self
            .output_schema
            .len()
            .checked_sub(1 + extra_cols)
            .ok_or_else(|| {
                super::OperatorError::Execution(
                    "UNWIND output schema lacks its element column".into(),
                )
            })?;
        for col_idx in 0..element_col_idx {
            let value = chunk
                .column(col_idx)
                .and_then(|col| col.get_value(self.current_row))
                .ok_or_else(|| {
                    super::OperatorError::ColumnNotFound(format!("UNWIND input column {col_idx}"))
                })?;
            builder
                .column_mut(col_idx)
                .ok_or_else(|| {
                    super::OperatorError::ColumnNotFound(format!("UNWIND output column {col_idx}"))
                })?
                .push_value(value);
        }

        // Add the unwound element after the inherited prefix.
        builder
            .column_mut(element_col_idx)
            .ok_or_else(|| super::OperatorError::ColumnNotFound("UNWIND output element".into()))?
            .push_value(element);

        // Add ORDINALITY (1-based) if requested
        let mut next_col = element_col_idx + 1;
        if self.emit_ordinality {
            if let Some(out_col) = builder.column_mut(next_col) {
                // reason: list index fits i64 for practical sizes
                #[allow(clippy::cast_possible_wrap)]
                out_col.push_value(Value::Int64((self.current_list_idx + 1) as i64));
            }
            next_col += 1;
        }

        // Add OFFSET (0-based) if requested
        if self.emit_offset
            && let Some(out_col) = builder.column_mut(next_col)
        {
            // reason: list index fits i64 for practical sizes
            #[allow(clippy::cast_possible_wrap)]
            out_col.push_value(Value::Int64(self.current_list_idx as i64));
        }

        builder.advance_row();
        self.current_list_idx += 1;

        // If we've exhausted this list, move to the next row
        if self.current_list_idx >= list.len() {
            self.current_row += 1;
        }

        Ok(builder.finish())
    }
}

impl Operator for UnwindOperator {
    fn next(&mut self) -> OperatorResult {
        self.advance()
    }

    fn reset(&mut self) {
        self.child.reset();
        self.current_chunk = None;
        self.current_row = 0;
        self.current_list_idx = 0;
        self.current_list = None;
    }

    fn name(&self) -> &'static str {
        "Unwind"
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use std::sync::Arc;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
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
            "MockOperator"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn unwind_preserves_declared_list_and_null_input_columns()
    -> Result<(), Box<dyn std::error::Error>> {
        let list = Value::List(vec![Value::Int64(4), Value::Null].into());
        let mut input = DataChunkBuilder::new(&[LogicalType::Any, LogicalType::Any]);
        input
            .column_mut(0)
            .ok_or("missing fixture list column")?
            .push_value(list.clone());
        input
            .column_mut(1)
            .ok_or("missing fixture null column")?
            .push_value(Value::Null);
        input.advance_row();
        let mut unwind = UnwindOperator::new(
            Box::new(MockOperator {
                chunks: vec![input.finish()],
                position: 0,
            }),
            0,
            "element".into(),
            vec![LogicalType::Any, LogicalType::Any, LogicalType::Any],
            false,
            false,
        );
        for expected in [Value::Int64(4), Value::Null] {
            let output = unwind.next()?.ok_or("missing UNWIND row")?;
            assert_eq!(output.row_count(), 1);
            assert_eq!(
                output.column(0).and_then(|column| column.get_value(0)),
                Some(list.clone())
            );
            assert_eq!(
                output.column(1).and_then(|column| column.get_value(0)),
                Some(Value::Null)
            );
            assert_eq!(
                output.column(2).and_then(|column| column.get_value(0)),
                Some(expected)
            );
            assert!(output.columns().iter().all(|column| column.len() == 1));
        }
        assert!(unwind.next()?.is_none());
        Ok(())
    }

    #[test]
    fn test_unwind_basic() {
        // Create a chunk with a list column [1, 2, 3]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]); // Any for list
        let list = Value::List(Arc::new([
            Value::Int64(1),
            Value::Int64(2),
            Value::Int64(3),
        ]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Create unwind operator
        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64], // Output is just the unwound element
            false,
            false,
        );

        // Should produce 3 rows
        let mut results = Vec::new();
        while let Ok(Some(chunk)) = unwind.next() {
            results.push(chunk);
        }

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_unwind_empty_list() {
        // A list with zero elements should produce no output rows
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64],
            false,
            false,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = unwind.next() {
            results.push(chunk);
        }

        assert_eq!(results.len(), 0, "Empty list should produce no rows");
    }

    #[test]
    fn test_unwind_empty_input() {
        // No chunks at all
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64],
            false,
            false,
        );

        assert!(unwind.next().unwrap().is_none());
    }

    #[test]
    fn test_unwind_multiple_rows() {
        // Two rows with lists of different sizes
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);

        let list1 = Value::List(Arc::new([Value::Int64(10), Value::Int64(20)]));
        builder.column_mut(0).unwrap().push_value(list1);
        builder.advance_row();

        let list2 = Value::List(Arc::new([Value::Int64(30)]));
        builder.column_mut(0).unwrap().push_value(list2);
        builder.advance_row();

        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64],
            false,
            false,
        );

        let mut count = 0;
        while let Ok(Some(_chunk)) = unwind.next() {
            count += 1;
        }

        // 2 from first list + 1 from second list = 3 rows
        assert_eq!(count, 3);
    }

    #[test]
    fn test_unwind_single_element_list() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([Value::String("hello".into())]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "item".to_string(),
            vec![LogicalType::String],
            false,
            false,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = unwind.next() {
            results.push(chunk);
        }

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_unwind_variable_name() {
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };

        let unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "my_var".to_string(),
            vec![LogicalType::Any],
            false,
            false,
        );

        assert_eq!(unwind.variable_name(), "my_var");
    }

    #[test]
    fn test_unwind_name() {
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };

        let unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Any],
            false,
            false,
        );

        assert_eq!(unwind.name(), "Unwind");
    }

    #[test]
    fn test_unwind_with_ordinality() {
        // Create a chunk with a list column [10, 20, 30]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([
            Value::Int64(10),
            Value::Int64(20),
            Value::Int64(30),
        ]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Output schema: [element (Any), ordinality (Int64)]
        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Int64],
            true,  // emit_ordinality
            false, // emit_offset
        );

        let mut ordinalities = Vec::new();
        while let Ok(Some(chunk)) = unwind.next() {
            if let Some(col) = chunk.column(1)
                && let Some(Value::Int64(v)) = col.get_value(0)
            {
                ordinalities.push(v);
            }
        }

        // ORDINALITY is 1-based
        assert_eq!(ordinalities, vec![1, 2, 3]);
    }

    #[test]
    fn test_unwind_with_offset() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([
            Value::Int64(10),
            Value::Int64(20),
            Value::Int64(30),
        ]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Output schema: [element (Any), offset (Int64)]
        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Int64],
            false, // emit_ordinality
            true,  // emit_offset
        );

        let mut offsets = Vec::new();
        while let Ok(Some(chunk)) = unwind.next() {
            if let Some(col) = chunk.column(1)
                && let Some(Value::Int64(v)) = col.get_value(0)
            {
                offsets.push(v);
            }
        }

        // OFFSET is 0-based
        assert_eq!(offsets, vec![0, 1, 2]);
    }

    #[test]
    fn test_unwind_into_any() {
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };
        let op = UnwindOperator::new(
            Box::new(mock),
            0,
            "items".to_string(),
            vec![LogicalType::Any, LogicalType::Any],
            false,
            false,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<UnwindOperator>().is_ok());
    }
}
