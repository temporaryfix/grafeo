//! Converts pull-based operator trees into push-based pipelines.
//!
//! The converter walks the operator tree top-down through each operator's
//! object-safe decomposition hook. Source operators (scan, expand, join) stay
//! pull-based and get wrapped in [`OperatorSource`](super::source::OperatorSource).
//!
//! This enables the documented push-based execution model without modifying
//! the planner, which continues to emit pull-based operator trees.
//!
//! The unqualified converter is unavailable:
//!
//! ```compile_fail,E0432
//! use grafeo_core::execution::pipeline_convert::convert_to_pipeline;
//! ```

#![cfg_attr(
    feature = "spill",
    doc = r#"
The former optional-memory-context entry point is unavailable:

```compile_fail,E0432
use grafeo_core::execution::pipeline_convert::convert_to_pipeline_with_memory;
```
"#
)]

#[cfg(test)]
use super::chunk::DataChunk;
#[cfg(test)]
use super::operators::push::FilterPredicate;
#[cfg(test)]
use super::operators::{
    DistinctOperator, FilterOperator, HashAggregateOperator, LimitOperator, Predicate,
    ProjectOperator, SortOperator,
};
use super::operators::{Operator, OperatorPipelineDecomposition};
use super::pipeline::PushOperator;

// -------------------------------------------------------------------------
// Type adapters (bridge pull types to push types)
// -------------------------------------------------------------------------

pub use super::operators::PredicateAdapter;

#[cfg(test)]
fn convert_sort_key(pull: &super::operators::SortKey) -> super::operators::push::SortKey {
    use super::operators::{NullOrder, SortDirection};
    super::operators::push::SortKey {
        column: pull.column,
        direction: match pull.direction {
            SortDirection::Ascending => super::operators::push::SortDirection::Ascending,
            SortDirection::Descending => super::operators::push::SortDirection::Descending,
        },
        null_order: match pull.null_order {
            NullOrder::NullsFirst => super::operators::push::NullOrder::First,
            NullOrder::NullsLast => super::operators::push::NullOrder::Last,
        },
    }
}

// NOTE: ProjectExprAdapter and is_simple_project are intentionally omitted.
// ProjectOperator carries store references, transaction context, and session
// context that cannot be transferred to push operators. Project stays pull-based.
// When a dedicated PushProjectOperator with store access is added, revisit this.

// -------------------------------------------------------------------------
// Pipeline converter
// -------------------------------------------------------------------------

/// Converts a pull-based tree using one query's always-available resources.
///
/// Returns the deepest non-convertible source and push operators in source-first
/// pipeline order. Unknown operators remain explicit pull/source boundaries.
///
/// Sort always receives the context: resident execution charges its owned row
/// capacities and stable-sort scratch envelope, while a configured spill
/// manager selects the spill-capable variant. Aggregate grant enforcement is a
/// subsequent stage, but both Aggregate variants receive the shared immutable
/// cancellation token; its configured spill variant also retains the context
/// and scoped registration.
///
/// ```
/// use grafeo_common::memory::buffer::BufferManager;
/// use grafeo_core::execution::{
///     QueryResourceContext,
///     operators::single_row::SingleRowOperator,
///     pipeline_convert::convert_to_pipeline_with_resources,
/// };
///
/// let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
/// let (source, push) = convert_to_pipeline_with_resources(
///     Box::new(SingleRowOperator::new()), &resources,
/// ).unwrap();
/// assert_eq!(source.name(), "SingleRow");
/// assert!(push.is_empty());
/// ```
///
/// # Errors
///
/// Returns a structured resource-context error if a resident breaker cannot
/// create its initial grant or a spill-capable operator cannot acquire its
/// unique scoped consumer registration.
pub fn convert_to_pipeline_with_resources(
    mut root: Box<dyn Operator>,
    resources: &super::memory::QueryResourceContext,
) -> Result<(Box<dyn Operator>, Vec<Box<dyn PushOperator>>), super::memory::QueryResourceContextError>
{
    // Context propagation is independent of conversion reachability. A pull
    // wrapper may remain the source boundary while a bounded breaker deeper in
    // its subtree still retains this exact query context.
    root.install_resource_context(resources)?;
    let mut push_ops: Vec<Box<dyn PushOperator>> = Vec::new();
    let source = decompose_recursive_resources(root, &mut push_ops, resources)?;
    push_ops.reverse();
    Ok((source, push_ops))
}

/// Recursively decomposes operators with one execution's resources.
fn decompose_recursive_resources(
    op: Box<dyn Operator>,
    push_ops: &mut Vec<Box<dyn PushOperator>>,
    ctx: &super::memory::QueryResourceContext,
) -> Result<Box<dyn Operator>, super::memory::QueryResourceContextError> {
    // DISTINCT has both a qualified pull cursor and an accounted push output.
    // Keep its pull cursor when an outer push stage cannot preserve the output
    // envelope. This is a conservative conversion choice, never authority to
    // invoke an unqualified sink. The pull boundary retains its bounded current
    // chunk while the established caller consumes it.
    if op.name() == "Distinct"
        && push_ops.iter_mut().any(|outer| {
            outer
                .admit_chunk_transport(super::pipeline::ChunkTransport::MayBeAccounted)
                .is_err()
                || outer.__accounted_push_permit().is_none()
        })
    {
        return Ok(op);
    }
    match op.decompose_pipeline_with_resources(ctx)? {
        OperatorPipelineDecomposition::Boundary(source) => Ok(source),
        OperatorPipelineDecomposition::Unary { child, push } => {
            push_ops.push(push);
            decompose_recursive_resources(child, push_ops, ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::QueryResourceContext;
    use crate::execution::operators::{OperatorResult, SortKey};
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::LogicalType;

    /// A trivial predicate that always returns true (for testing decomposition only).
    struct AlwaysTruePredicate;

    impl Predicate for AlwaysTruePredicate {
        fn evaluate(
            &self,
            _chunk: &DataChunk,
            _row: usize,
        ) -> Result<bool, crate::execution::operators::OperatorError> {
            Ok(true)
        }
    }

    /// A minimal test operator that produces one chunk.
    struct TestScanOperator {
        emitted: bool,
    }

    impl TestScanOperator {
        fn new() -> Self {
            Self { emitted: false }
        }
    }

    impl Operator for TestScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.emitted {
                return Ok(None);
            }
            self.emitted = true;
            let mut col = crate::execution::vector::ValueVector::with_type(LogicalType::Int64);
            col.push_int64(1);
            col.push_int64(2);
            col.push_int64(3);
            Ok(Some(DataChunk::new(vec![col])))
        }

        fn reset(&mut self) {
            self.emitted = false;
        }

        fn name(&self) -> &'static str {
            "TestScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn convert_bare_scan_produces_empty_pipeline() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(scan, &resources).unwrap();
        assert!(push_ops.is_empty());
        assert_eq!(source.name(), "TestScan");
    }

    #[test]
    fn convert_filter_scan_produces_one_push_op() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(filter, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert_eq!(push_ops.len(), 1);
        // Push operators have their own naming convention
        assert!(
            push_ops[0].name().contains("Filter"),
            "expected filter push op, got {}",
            push_ops[0].name()
        );
    }

    #[test]
    fn convert_limit_filter_scan_produces_two_push_ops() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));
        let limit: Box<dyn Operator> =
            Box::new(LimitOperator::new(filter, 10, vec![LogicalType::Int64]));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(limit, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 2);
        // Pipeline order: filter first, then limit
        assert!(push_ops[0].name().contains("Filter"));
        assert!(push_ops[1].name().contains("Limit"));
    }

    #[test]
    fn convert_sort_scan_produces_one_push_op() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let keys = vec![SortKey::ascending(0)];
        let sort: Box<dyn Operator> =
            Box::new(SortOperator::new(scan, keys, vec![LogicalType::Int64]));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(sort, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Sort"));
    }

    #[test]
    fn convert_aggregate_scan_produces_one_push_op() {
        use crate::execution::operators::{AggregateExpr, AggregateFunction};

        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let aggregates = vec![AggregateExpr {
            function: AggregateFunction::Count,
            column: None,
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }];
        let agg: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            scan,
            vec![],
            aggregates,
            vec![LogicalType::Int64],
        ));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(agg, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Aggregate"));
    }

    #[test]
    fn convert_distinct_scan_produces_one_push_op() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let distinct: Box<dyn Operator> =
            Box::new(DistinctOperator::new(scan, vec![LogicalType::Int64]));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(distinct, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Distinct"));
    }

    #[test]
    fn convert_distinct_on_columns_scan() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let distinct: Box<dyn Operator> = Box::new(DistinctOperator::on_columns(
            scan,
            vec![0],
            vec![LogicalType::Int64],
        ));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(distinct, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Distinct"));
    }

    #[test]
    fn convert_deep_pipeline_sort_filter_limit() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));
        let keys = vec![SortKey::ascending(0)];
        let sort: Box<dyn Operator> =
            Box::new(SortOperator::new(filter, keys, vec![LogicalType::Int64]));
        let limit: Box<dyn Operator> =
            Box::new(LimitOperator::new(sort, 5, vec![LogicalType::Int64]));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(limit, &resources).unwrap();
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 3);
        // Pipeline order: filter, sort, limit (source-first)
        assert!(push_ops[0].name().contains("Filter"));
        assert!(push_ops[1].name().contains("Sort"));
        assert!(push_ops[2].name().contains("Limit"));
    }

    #[test]
    fn pipeline_roundtrip_produces_correct_results() {
        use crate::execution::pipeline::Pipeline;
        use crate::execution::sink::CollectorSink;
        use crate::execution::source::OperatorSource;

        // Build: Scan -> Filter(always true) -> Sort(col 0 ASC)
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));
        let keys = vec![SortKey::ascending(0)];
        let sort: Box<dyn Operator> =
            Box::new(SortOperator::new(filter, keys, vec![LogicalType::Int64]));

        // Convert to pipeline
        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(sort, &resources).unwrap();
        assert_eq!(push_ops.len(), 2); // Filter + Sort

        // Execute the pipeline
        let source = Box::new(OperatorSource::new(source));
        let collector = CollectorSink::new();
        let mut pipeline = Pipeline::new(source, push_ops, Box::new(collector));
        pipeline.execute().unwrap();

        // Extract results
        let sink_box = pipeline.into_sink();
        let any_sink: Box<dyn std::any::Any> = sink_box.into_any();
        let collector = any_sink.downcast::<CollectorSink>().unwrap();
        assert_eq!(collector.row_count(), 3);
    }

    #[test]
    fn predicate_adapter_delegates_correctly() -> Result<(), Box<dyn std::error::Error>> {
        let mut col = crate::execution::vector::ValueVector::with_type(LogicalType::Int64);
        col.push_int64(42);
        let chunk = DataChunk::new(vec![col]);

        let adapter = PredicateAdapter(Box::new(AlwaysTruePredicate));
        assert!(adapter.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn convert_sort_key_maps_directions() {
        use crate::execution::operators::{NullOrder, SortDirection};

        use crate::execution::operators::push::{
            NullOrder as PushNullOrder, SortDirection as PushSortDirection,
        };

        let asc = super::convert_sort_key(&SortKey {
            column: 3,
            direction: SortDirection::Ascending,
            null_order: NullOrder::NullsFirst,
        });
        assert_eq!(asc.column, 3);
        assert_eq!(asc.direction, PushSortDirection::Ascending);
        assert_eq!(asc.null_order, PushNullOrder::First);

        let desc = super::convert_sort_key(&SortKey {
            column: 7,
            direction: SortDirection::Descending,
            null_order: NullOrder::NullsLast,
        });
        assert_eq!(desc.column, 7);
        assert_eq!(desc.direction, PushSortDirection::Descending);
        assert_eq!(desc.null_order, PushNullOrder::Last);
    }

    #[test]
    fn test_distinct_on_columns_pipeline_execution() {
        use crate::execution::pipeline::Pipeline;
        use crate::execution::sink::CountingSink;

        // Build: Scan -> Distinct(on column 0)
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let distinct: Box<dyn Operator> = Box::new(DistinctOperator::on_columns(
            scan,
            vec![0],
            vec![LogicalType::Int64],
        ));

        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(distinct, &resources).unwrap();
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Distinct"));

        // Execute the pipeline and verify results
        let source = Box::new(crate::execution::source::OperatorSource::new(source));
        let counter = CountingSink::new();
        let mut pipeline = Pipeline::new(source, push_ops, Box::new(counter));
        pipeline.execute().unwrap();

        let sink_box = pipeline.into_sink();
        let any_sink: Box<dyn std::any::Any> = sink_box.into_any();
        let counter = any_sink.downcast::<CountingSink>().unwrap();
        // TestScan produces [1, 2, 3], all distinct, so 3 rows
        assert_eq!(counter.count(), 3);
    }

    #[test]
    fn test_unrecognized_operator_stays_as_source() {
        /// A custom operator with an unrecognized name.
        struct CustomJoinOperator;

        impl Operator for CustomJoinOperator {
            fn next(&mut self) -> OperatorResult {
                Ok(None)
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "CustomNestedLoopJoin"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        let join: Box<dyn Operator> = Box::new(CustomJoinOperator);
        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let (source, push_ops) = convert_to_pipeline_with_resources(join, &resources).unwrap();
        assert_eq!(source.name(), "CustomNestedLoopJoin");
        assert!(
            push_ops.is_empty(),
            "unrecognized operator should produce no push ops"
        );
    }

    #[test]
    fn pull_project_boundaries_forward_one_context_through_both_breakers() {
        use std::sync::{Arc, Mutex};

        struct ContextProbe(Arc<Mutex<Vec<u64>>>);

        impl Operator for ContextProbe {
            fn next(&mut self) -> OperatorResult {
                Ok(None)
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "ContextProbe"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }

            fn install_resource_context(
                &mut self,
                resources: &crate::execution::QueryResourceContext,
            ) -> Result<(), crate::execution::QueryResourceContextError> {
                self.0.lock().unwrap().push(resources.query_id().get());
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let aggregate: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            Box::new(ContextProbe(Arc::clone(&seen))),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        let inner_project: Box<dyn Operator> =
            Box::new(ProjectOperator::new(aggregate, Vec::new(), Vec::new()));
        let order: Box<dyn Operator> =
            Box::new(SortOperator::new(inner_project, Vec::new(), Vec::new()));
        let outer_project: Box<dyn Operator> =
            Box::new(ProjectOperator::new(order, Vec::new(), Vec::new()));
        let resources = crate::execution::QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20),
        )
        .unwrap();

        let (source, push) = convert_to_pipeline_with_resources(outer_project, &resources).unwrap();

        assert_eq!(source.name(), "Project");
        assert!(push.is_empty(), "project remains an explicit pull boundary");
        assert_eq!(*seen.lock().unwrap(), vec![resources.query_id().get()]);
    }

    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    fn spill_resource_context(
        root: &std::path::Path,
    ) -> (
        std::sync::Arc<grafeo_common::memory::buffer::BufferManager>,
        crate::execution::QueryResourceContext,
    ) {
        let manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let spill_root = crate::execution::spill::RootedSpillFixture::new(root)
            .root()
            .unwrap();
        let resources = crate::execution::QueryResourceContext::with_spill_root(
            std::sync::Arc::clone(&manager),
            &spill_root,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        (manager, resources)
    }

    #[test]
    fn resource_converter_without_spill_manager_charges_resident_sort() {
        let manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(std::sync::Arc::clone(&manager)).unwrap();
        let sort: Box<dyn Operator> = Box::new(SortOperator::new(
            Box::new(TestScanOperator::new()),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        ));

        let (source, push_ops) = convert_to_pipeline_with_resources(sort, &resources).unwrap();

        assert_eq!(push_ops.len(), 1);
        assert_eq!(push_ops[0].name(), "SortPush");
        assert_eq!(manager.stats().consumer_count, 0);
        assert_eq!(manager.allocated(), 0);

        let source = Box::new(crate::execution::source::OperatorSource::new(source));
        let mut pipeline = crate::execution::pipeline::Pipeline::new(
            source,
            push_ops,
            Box::new(crate::execution::sink::CollectorSink::new()),
        );
        pipeline.execute().unwrap();
        assert!(
            manager.allocated() > 0,
            "the qualified resident route must retain a real grant through finalize"
        );

        drop(pipeline);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resource_converter_propagates_cancellation_to_resident_aggregate() {
        use crate::execution::operators::AggregateExpr;

        let manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let resources =
            crate::execution::QueryResourceContext::new_with_cancellation(manager, control.token())
                .unwrap();
        let aggregate: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            Box::new(TestScanOperator::new()),
            Vec::new(),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        ));
        let (source, push_ops) = convert_to_pipeline_with_resources(aggregate, &resources).unwrap();
        assert_eq!(push_ops.len(), 1);
        assert_eq!(push_ops[0].name(), "AggregatePush");

        control.cancellation_handle().cancel();
        let source = Box::new(crate::execution::source::OperatorSource::new(source));
        let mut pipeline = crate::execution::pipeline::Pipeline::new(
            source,
            push_ops,
            Box::new(crate::execution::sink::CollectorSink::new()),
        );
        let error = pipeline.execute().unwrap_err();

        assert!(matches!(
            error,
            crate::execution::operators::OperatorError::QueryCancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        let sink = pipeline.into_sink().into_any();
        let collector = sink
            .downcast::<crate::execution::sink::CollectorSink>()
            .unwrap();
        assert_eq!(collector.row_count(), 0);
    }

    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn two_same_kind_spill_breakers_hold_independent_scoped_registrations() {
        let root = tempfile::tempdir().unwrap();
        let (manager, resources) = spill_resource_context(root.path());
        let inner: Box<dyn Operator> = Box::new(SortOperator::new(
            Box::new(TestScanOperator::new()),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        ));
        let outer: Box<dyn Operator> = Box::new(SortOperator::new(
            inner,
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        ));

        let (_source, mut push_ops) =
            convert_to_pipeline_with_resources(outer, &resources).unwrap();
        assert_eq!(manager.stats().consumer_count, 2);

        drop(push_ops.remove(0));
        assert_eq!(manager.stats().consumer_count, 1);
        drop(push_ops);
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn spill_breaker_registrations_cleanup_during_unwind() {
        let root = tempfile::tempdir().unwrap();
        let (manager, resources) = spill_resource_context(root.path());

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let sort: Box<dyn Operator> = Box::new(SortOperator::new(
                Box::new(TestScanOperator::new()),
                vec![SortKey::ascending(0)],
                vec![LogicalType::Int64],
            ));
            let (_source, _push_ops) =
                convert_to_pipeline_with_resources(sort, &resources).unwrap();
            assert_eq!(manager.stats().consumer_count, 1);
            panic!("exercise operator unwind cleanup");
        }));

        assert!(unwind.is_err());
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn configured_resources_create_spillable_sort_and_aggregate_registrations() {
        use crate::execution::operators::{AggregateExpr, AggregateFunction};

        let root = tempfile::tempdir().unwrap();
        let (manager, resources) = spill_resource_context(root.path());
        let aggregate: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            Box::new(TestScanOperator::new()),
            vec![0],
            vec![AggregateExpr {
                function: AggregateFunction::Count,
                column: None,
                column2: None,
                distinct_key_column: None,
                distinct: false,
                alias: None,
                percentile: None,
                separator: None,
            }],
            vec![LogicalType::Int64, LogicalType::Int64],
        ));
        let sort: Box<dyn Operator> = Box::new(SortOperator::new(
            aggregate,
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::Int64],
        ));

        let (_source, push_ops) = convert_to_pipeline_with_resources(sort, &resources).unwrap();

        assert_eq!(push_ops.len(), 2);
        assert!(
            push_ops
                .iter()
                .all(|operator| operator.name().contains("Spillable"))
        );
        assert_eq!(manager.stats().consumer_count, 2);
        drop(push_ops);
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[cfg(all(
        feature = "spill",
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn spill_resource_converter_propagates_cancellation_without_pipeline_checkpoint() {
        use crate::execution::operators::AggregateExpr;

        let root = tempfile::tempdir().unwrap();
        let manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let spill_root = crate::execution::spill::RootedSpillFixture::new(root.path())
            .root()
            .unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::with_spill_root(
            std::sync::Arc::clone(&manager),
            &spill_root,
            control.token(),
        )
        .unwrap();
        let aggregate: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            Box::new(TestScanOperator::new()),
            Vec::new(),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        ));
        let (source, push_ops) = convert_to_pipeline_with_resources(aggregate, &resources).unwrap();
        assert_eq!(push_ops.len(), 1);
        assert_eq!(push_ops[0].name(), "SpillableAggregatePush");
        assert_eq!(manager.stats().consumer_count, 1);

        control.cancellation_handle().cancel();
        let source = Box::new(crate::execution::source::OperatorSource::new(source));
        let mut pipeline = crate::execution::pipeline::Pipeline::new(
            source,
            push_ops,
            Box::new(crate::execution::sink::CollectorSink::new()),
        );
        let error = pipeline.execute().unwrap_err();

        assert!(matches!(
            error,
            crate::execution::operators::OperatorError::QueryCancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        let sink = pipeline.into_sink().into_any();
        assert_eq!(manager.stats().consumer_count, 0);
        let collector = sink
            .downcast::<crate::execution::sink::CollectorSink>()
            .unwrap();
        assert_eq!(collector.row_count(), 0);
    }
}
