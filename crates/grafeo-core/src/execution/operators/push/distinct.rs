//! Push adapters for the shared exact DISTINCT state.

use crate::execution::operators::{OperatorError, distinct_state::ExactDistinctState};
use crate::execution::pipeline::{
    AccountedPushPermit, AccountedSinkPermit, ChunkTransport, PushOperator, Sink,
    qualified_accounted_transport::QualifiedPushOperator,
};
use crate::execution::{
    AccountedDataChunk, DataChunk, QueryResourceContext, QueryResourceContextError,
};
use parking_lot::Mutex;

/// Removes duplicate rows using the same admitted state as pull execution.
///
/// Input is consumed once. Finalization emits earliest witnesses in input order,
/// including when retained keys and witnesses spill to disk.
pub struct DistinctPushOperator {
    // The execution cursor is exclusively accessed while pushing/finalizing.
    // Mutex supplies Sync without requiring internal cursor cells to be shared.
    state: Mutex<ExactDistinctState>,
    finalized: bool,
    qualified: bool,
}

impl DistinctPushOperator {
    /// Creates DISTINCT on every input column.
    pub fn new() -> Self {
        Self::from_columns(None)
    }

    /// Creates DISTINCT on the specified key columns, retaining full witnesses.
    pub fn on_columns(columns: Vec<usize>) -> Self {
        Self::from_columns(Some(columns))
    }

    fn from_columns(columns: Option<Vec<usize>>) -> Self {
        Self {
            state: Mutex::new(ExactDistinctState::new(columns, Vec::new())),
            finalized: false,
            qualified: false,
        }
    }

    /// Creates DISTINCT under the actual query resource and cancellation owner.
    ///
    /// # Errors
    /// Returns an error if the query cannot admit the operator resource owner.
    pub fn with_resource_context(
        columns: Option<Vec<usize>>,
        resources: QueryResourceContext,
    ) -> Result<Self, QueryResourceContextError> {
        let mut operator = Self::from_columns(columns);
        operator.qualified = true;
        operator
            .state
            .get_mut()
            .install_resource_context(&resources)?;
        Ok(operator)
    }

    /// Returns the number of unique witnesses identified so far.
    /// Spill execution determines the complete count during finalization.
    pub fn unique_count(&self) -> usize {
        self.state.lock().unique_count()
    }

    fn ingest_chunk(&mut self, chunk: &DataChunk) -> Result<bool, OperatorError> {
        if self.finalized {
            return Ok(false);
        }
        let state = self.state.get_mut();
        for row in chunk.selected_indices() {
            state.ingest(chunk, row)?;
        }
        Ok(true)
    }
}

impl Default for DistinctPushOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOperator for DistinctPushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        self.ingest_chunk(&chunk)
    }

    fn push_accounted(
        &mut self,
        chunk: AccountedDataChunk,
        _sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        // Keep the incoming envelope alive until all admitted copies complete.
        self.qualified = true;
        self.ingest_chunk(chunk.chunk())
    }

    fn __accounted_push_permit(&mut self) -> Option<AccountedPushPermit<'_>> {
        Some(AccountedPushPermit::new(self))
    }

    fn admit_chunk_transport(
        &self,
        input: ChunkTransport,
    ) -> Result<ChunkTransport, OperatorError> {
        Ok(if self.qualified {
            ChunkTransport::MayBeAccounted
        } else {
            input
        })
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        let state = self.state.get_mut();
        state.finish_input()?;
        if self.qualified {
            let consumer = sink.name();
            let mut permit = sink
                .__accounted_sink_permit()
                .ok_or(OperatorError::UnsupportedAccountedTransport { consumer })
                .map_err(|error| state.fail(error))?;
            while let Some(chunk) = state.next_accounted_chunk()? {
                if !permit.consume(chunk).map_err(|error| state.fail(error))? {
                    state.reset()?;
                    break;
                }
            }
        } else {
            // Standalone constructors have no installed query owner. Their
            // existing plain-output contract makes the caller own each chunk.
            while let Some(chunk) = state.next_chunk()? {
                if !sink.consume(chunk).map_err(|error| state.fail(error))? {
                    state.reset()?;
                    break;
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DistinctPush"
    }
}

impl QualifiedPushOperator for DistinctPushOperator {
    fn push_accounted_qualified(
        &mut self,
        chunk: AccountedDataChunk,
        _sink: &mut AccountedSinkPermit<'_>,
    ) -> Result<bool, OperatorError> {
        self.qualified = true;
        self.ingest_chunk(chunk.chunk())
    }
}

/// A materializing DISTINCT adapter using the shared exact state.
pub struct DistinctMaterializingOperator {
    inner: DistinctPushOperator,
}

impl DistinctMaterializingOperator {
    /// Creates DISTINCT on every input column.
    pub fn new() -> Self {
        Self {
            inner: DistinctPushOperator::new(),
        }
    }

    /// Creates DISTINCT on specified columns while retaining complete rows.
    pub fn on_columns(columns: Vec<usize>) -> Self {
        Self {
            inner: DistinctPushOperator::on_columns(columns),
        }
    }

    /// Creates DISTINCT under the query's resource and cancellation owner.
    ///
    /// # Errors
    /// Returns an error if operator resource admission fails.
    pub fn with_resource_context(
        columns: Option<Vec<usize>>,
        resources: QueryResourceContext,
    ) -> Result<Self, QueryResourceContextError> {
        Ok(Self {
            inner: DistinctPushOperator::with_resource_context(columns, resources)?,
        })
    }
}

impl Default for DistinctMaterializingOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOperator for DistinctMaterializingOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        self.inner.push(chunk, sink)
    }

    fn push_accounted(
        &mut self,
        chunk: AccountedDataChunk,
        sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        self.inner.push_accounted(chunk, sink)
    }

    fn __accounted_push_permit(&mut self) -> Option<AccountedPushPermit<'_>> {
        Some(AccountedPushPermit::new(self))
    }

    fn admit_chunk_transport(
        &self,
        input: ChunkTransport,
    ) -> Result<ChunkTransport, OperatorError> {
        self.inner.admit_chunk_transport(input)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        self.inner.finalize(sink)
    }

    fn name(&self) -> &'static str {
        "DistinctMaterializing"
    }
}

impl QualifiedPushOperator for DistinctMaterializingOperator {
    fn push_accounted_qualified(
        &mut self,
        chunk: AccountedDataChunk,
        sink: &mut AccountedSinkPermit<'_>,
    ) -> Result<bool, OperatorError> {
        self.inner.push_accounted_qualified(chunk, sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::OperatorError;
    use crate::execution::pipeline::Sink;
    use crate::execution::selection::SelectionVector;
    use crate::execution::sink::CollectorSink;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::{LogicalType, Value};

    #[derive(Default)]
    struct RetainingAccountedSink {
        chunks: Vec<AccountedDataChunk>,
    }

    impl Sink for RetainingAccountedSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            Err(OperatorError::Execution(
                "qualified output lost its envelope".into(),
            ))
        }
        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            "RetainingAccountedDistinctTest"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
        fn admit_chunk_transport(&self, input: ChunkTransport) -> Result<(), OperatorError> {
            let _ = input;
            Ok(())
        }
        fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
            Some(AccountedSinkPermit::new(self))
        }
    }

    impl crate::execution::pipeline::qualified_accounted_transport::QualifiedSink
        for RetainingAccountedSink
    {
        fn consume_accounted_qualified(
            &mut self,
            chunk: AccountedDataChunk,
        ) -> Result<bool, OperatorError> {
            self.chunks.push(chunk);
            Ok(true)
        }
    }

    #[test]
    fn qualified_distinct_output_keeps_its_grant_after_operator_drop() {
        use grafeo_common::memory::buffer::BufferManager;
        let resources = QueryResourceContext::new(BufferManager::with_budget(1024 * 1024)).unwrap();
        let mut operator =
            DistinctPushOperator::with_resource_context(None, resources.clone()).unwrap();
        assert_eq!(
            operator
                .admit_chunk_transport(ChunkTransport::PlainOnly)
                .unwrap(),
            ChunkTransport::MayBeAccounted
        );
        let mut sink = RetainingAccountedSink::default();
        operator
            .push(create_test_chunk(&[3, 1, 3, 2]), &mut sink)
            .unwrap();
        operator.finalize(&mut sink).unwrap();
        let rows = sink
            .chunks
            .iter()
            .map(|chunk| chunk.chunk().row_count())
            .sum::<usize>();
        assert_eq!(rows, 3);
        operator.finalize(&mut sink).unwrap();
        assert_eq!(
            sink.chunks
                .iter()
                .map(|chunk| chunk.chunk().row_count())
                .sum::<usize>(),
            3
        );
        drop(operator);
        assert!(resources.query_stats().allocated_bytes > 0);
        assert_eq!(
            sink.chunks[0].chunk().column(0).unwrap().get_value(0),
            Some(Value::Int64(3))
        );
        drop(sink);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn qualified_distinct_rejects_unqualified_retaining_sink_without_raw_delivery() {
        use grafeo_common::memory::buffer::BufferManager;
        let resources = QueryResourceContext::new(BufferManager::with_budget(1024 * 1024)).unwrap();
        let mut operator =
            DistinctPushOperator::with_resource_context(None, resources.clone()).unwrap();
        let mut sink = CollectorSink::new();
        operator
            .push(create_test_chunk(&[1, 1, 2]), &mut sink)
            .unwrap();
        assert!(matches!(
            operator.finalize(&mut sink),
            Err(OperatorError::UnsupportedAccountedTransport { .. })
        ));
        assert_eq!(sink.row_count(), 0);
        drop(operator);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn qualified_sink_error_cleans_spill_before_drop_and_retains_cleanup_authority() {
        use crate::execution::operators::AccountedFailureClassification;
        use crate::execution::spill::{
            CleartextSpillRecordProvider, SpillFrameLimits, SpillIo, SpillIoOperation,
        };
        use grafeo_common::memory::buffer::BufferManager;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        #[derive(Debug)]
        struct CleanupPayload {
            drops: Arc<AtomicUsize>,
            formats: Arc<AtomicUsize>,
        }
        impl std::fmt::Display for CleanupPayload {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.formats.fetch_add(1, Ordering::SeqCst);
                panic!("sink failure cleanup must retain its payload without formatting");
            }
        }
        impl std::error::Error for CleanupPayload {}
        impl Drop for CleanupPayload {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct DeleteFault {
            armed: AtomicBool,
            attempts: AtomicUsize,
            drops: Arc<AtomicUsize>,
            formats: Arc<AtomicUsize>,
        }
        impl SpillIo for DeleteFault {
            fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
                // Reader hooks never allocate or fail; only deletion is armed.
                Some(0)
            }
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                if operation == SpillIoOperation::Delete && self.armed.load(Ordering::SeqCst) {
                    self.attempts.fetch_add(1, Ordering::SeqCst);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        CleanupPayload {
                            drops: self.drops.clone(),
                            formats: self.formats.clone(),
                        },
                    ));
                }
                Ok(())
            }
        }
        struct FailingSink {
            retained: RetainingAccountedSink,
            fault: Arc<DeleteFault>,
            primary: Option<OperatorError>,
        }
        impl Sink for FailingSink {
            fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
                self.retained.consume(chunk)
            }
            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }
            fn name(&self) -> &'static str {
                "FailingAccountedDistinctTest"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
            fn admit_chunk_transport(&self, _: ChunkTransport) -> Result<(), OperatorError> {
                Ok(())
            }
            fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
                Some(AccountedSinkPermit::new(self))
            }
        }
        impl crate::execution::pipeline::qualified_accounted_transport::QualifiedSink for FailingSink {
            fn consume_accounted_qualified(
                &mut self,
                chunk: AccountedDataChunk,
            ) -> Result<bool, OperatorError> {
                self.retained.chunks.push(chunk);
                // Arm only after real sort finalization and the first transfer;
                // the fault therefore belongs to sink-error cleanup, not setup.
                self.fault.armed.store(true, Ordering::SeqCst);
                Err(self.primary.take().expect("failing sink is consumed once"))
            }
        }

        let fault = Arc::new(DeleteFault {
            armed: AtomicBool::new(false),
            attempts: AtomicUsize::new(0),
            drops: Arc::new(AtomicUsize::new(0)),
            formats: Arc::new(AtomicUsize::new(0)),
        });
        let directory = tempfile::tempdir().unwrap();
        let spill_root = crate::execution::spill::RootedSpillFixture::new(directory.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(fault.clone())
            .root()
            .unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(512 << 10),
            &spill_root,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = resources.ensure_spill_manager().unwrap().unwrap().clone();
        let mut operator =
            DistinctPushOperator::with_resource_context(None, resources.clone()).unwrap();
        let mut sink = FailingSink {
            retained: RetainingAccountedSink::default(),
            fault: fault.clone(),
            primary: Some(OperatorError::ColumnNotFound("sink_primary".into())),
        };
        for first in (0..4096_i64).step_by(32) {
            let mut input =
                DataChunk::with_capacity(&[LogicalType::Int64, LogicalType::String], 32);
            for key in first..first + 32 {
                input.column_mut(0).unwrap().push_int64(key);
                input.column_mut(1).unwrap().push_string("original");
            }
            input.set_count(32);
            operator.push(input, &mut sink).unwrap();
        }
        assert!(manager.disk_stats().published_live_bytes > 0);
        let error = operator.finalize(&mut sink).unwrap_err();
        assert_eq!(sink.retained.chunks.len(), 1);
        assert!(
            fault.attempts.load(Ordering::SeqCst) > 0,
            "sink error must attempt spill cleanup before operator Drop"
        );
        assert_eq!(fault.formats.load(Ordering::SeqCst), 0);
        assert_eq!(fault.drops.load(Ordering::SeqCst), 0);
        assert!(matches!(
            &error,
            OperatorError::ClassifiedAccountedFailure {
                classification: AccountedFailureClassification::ColumnNotFound,
                ..
            }
        ));
        let retained_error = error.clone();
        fault.armed.store(false, Ordering::SeqCst);
        drop(operator);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(
            sink.retained.chunks[0]
                .chunk()
                .column(0)
                .unwrap()
                .get_value(0),
            Some(Value::Int64(0))
        );
        let output_bytes = sink.retained.chunks[0].granted_bytes();
        assert!(resources.query_stats().allocated_bytes > output_bytes);
        drop(error);
        assert_eq!(fault.drops.load(Ordering::SeqCst), 0);
        drop(retained_error);
        assert!(fault.drops.load(Ordering::SeqCst) > 0);
        assert_eq!(fault.formats.load(Ordering::SeqCst), 0);
        assert_eq!(resources.query_stats().allocated_bytes, output_bytes);
        drop(sink);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    struct StopAfterFirstSink {
        consumes: usize,
    }

    impl Sink for StopAfterFirstSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            self.consumes += 1;
            Ok(false)
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "StopAfterFirst"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

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

    fn create_typed_entity_chunk(entity_type: LogicalType, rows: &[(i64, i64)]) -> DataChunk {
        let mut key = ValueVector::with_type(LogicalType::Int64);
        let mut entity = ValueVector::with_type(entity_type);
        for &(row_key, entity_id) in rows {
            key.push_value(Value::Int64(row_key));
            entity.push_value(Value::Int64(entity_id));
        }
        DataChunk::new(vec![key, entity])
    }

    #[test]
    fn distinct_push_preserves_typed_edge_lists_after_row_filtering() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(
                create_typed_edge_list_chunk(&[&[11], &[11], &[22]]),
                &mut sink,
            )
            .unwrap();

        distinct.finalize(&mut sink).unwrap();
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
    fn distinct_push_uses_physical_indices_for_preselected_rows() {
        let mut chunk = create_typed_edge_list_chunk(&[&[10], &[11], &[12], &[13]]);
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));

        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 2);
        assert_eq!(
            sink.chunks()[0].column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(11)].into()))
        );
        assert_eq!(
            sink.chunks()[0].column(0).unwrap().get_value(1),
            Some(Value::List(vec![Value::Int64(13)].into()))
        );
    }

    #[test]
    fn distinct_materializing_preserves_typed_edge_lists_across_chunks() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_typed_edge_list_chunk(&[&[11], &[22]]), &mut sink)
            .unwrap();
        distinct
            .push(create_typed_edge_list_chunk(&[&[22], &[33]]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        let chunks = sink.chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(2),
            Some(Value::List(vec![Value::Int64(33)].into()))
        );
    }

    #[test]
    fn distinct_materializing_emits_homogeneous_schema_runs() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_typed_edge_list_chunk(&[&[11]]), &mut sink)
            .unwrap();
        distinct.push(create_test_chunk(&[22]), &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.chunks().len(), 2);
        assert_eq!(
            sink.chunks()[0].column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            sink.chunks()[1].column(0).unwrap().data_type(),
            &LogicalType::Any
        );
    }

    #[test]
    fn distinct_materializing_preserves_node_and_edge_provenance_for_same_ids() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(
                create_typed_entity_chunk(LogicalType::Node, &[(1, 42)]),
                &mut sink,
            )
            .unwrap();
        distinct
            .push(
                create_typed_entity_chunk(LogicalType::Edge, &[(2, 42)]),
                &mut sink,
            )
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.chunks().len(), 2);
        assert_eq!(
            sink.chunks()[0].column(1).unwrap().data_type(),
            &LogicalType::Node
        );
        assert_eq!(
            sink.chunks()[1].column(1).unwrap().data_type(),
            &LogicalType::Edge
        );
        assert_eq!(
            sink.chunks()[0].column(1).unwrap().get_value(0),
            Some(Value::Int64(42))
        );
        assert_eq!(
            sink.chunks()[1].column(1).unwrap().get_value(0),
            Some(Value::Int64(42))
        );
    }

    #[test]
    fn distinct_materializing_stops_after_sink_requests_termination() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut collector = CollectorSink::new();
        distinct
            .push(create_typed_edge_list_chunk(&[&[11]]), &mut collector)
            .unwrap();
        distinct
            .push(create_test_chunk(&[22]), &mut collector)
            .unwrap();

        let mut sink = StopAfterFirstSink { consumes: 0 };
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(sink.consumes, 1);
    }

    #[test]
    fn test_distinct_all_unique() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5);
        assert_eq!(distinct.unique_count(), 5);
    }

    #[test]
    fn test_distinct_with_duplicates() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 1, 3, 2, 1, 4]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 4); // 1, 2, 3, 4
        assert_eq!(distinct.unique_count(), 4);
    }

    #[test]
    fn test_distinct_all_same() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[5, 5, 5, 5, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 1);
        assert_eq!(distinct.unique_count(), 1);
    }

    #[test]
    fn test_distinct_multiple_chunks() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 3]), &mut sink)
            .unwrap();
        distinct
            .push(create_test_chunk(&[2, 3, 4]), &mut sink)
            .unwrap();
        distinct
            .push(create_test_chunk(&[3, 4, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5); // 1, 2, 3, 4, 5
    }

    #[test]
    fn test_distinct_materializing() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        // All output comes in finalize
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 7); // 7 unique values
    }

    fn create_mixed_chunk(values: &[Value]) -> DataChunk {
        let vector = ValueVector::from_values(values);
        DataChunk::new(vec![vector])
    }

    #[test]
    fn test_distinct_null_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[Value::Null, Value::Null, Value::Int64(1)]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2); // Null + 1
    }

    #[test]
    fn test_distinct_bool_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[Value::Bool(true), Value::Bool(false), Value::Bool(true)]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_float_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Float64(1.0),
            Value::Float64(2.0),
            Value::Float64(1.0),
            Value::Float64(f64::NAN),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 3); // 1.0, 2.0, NaN
    }

    #[test]
    fn test_distinct_string_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk =
            create_mixed_chunk(&[Value::from("Alix"), Value::from("Gus"), Value::from("Alix")]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_bytes_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Bytes(vec![1u8, 2, 3].into()),
            Value::Bytes(vec![4u8, 5, 6].into()),
            Value::Bytes(vec![1u8, 2, 3].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_list_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
            Value::List(vec![Value::Int64(3), Value::Int64(4)].into()),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_map_values() {
        use std::collections::BTreeMap;

        let mut map1 = BTreeMap::new();
        map1.insert("a".into(), Value::Int64(1));
        let mut map2 = BTreeMap::new();
        map2.insert("b".into(), Value::Int64(2));

        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Map(map1.clone().into()),
            Value::Map(map2.into()),
            Value::Map(map1.into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_vector_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Value::Vector(vec![3.0_f32, 4.0].into()),
            Value::Vector(vec![1.0_f32, 2.0].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_path_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let path1 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::Int64(10)].into(),
        };
        let path2 = Value::Path {
            nodes: vec![Value::Int64(3), Value::Int64(4)].into(),
            edges: vec![Value::Int64(20)].into(),
        };

        let chunk = create_mixed_chunk(&[path1.clone(), path2, path1]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_mixed_types_are_distinct() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        // Different types with "similar" content should be distinct
        let chunk = create_mixed_chunk(&[
            Value::Int64(1),
            Value::Float64(1.0),
            Value::from("1"),
            Value::Bool(true),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 4);
    }
}
