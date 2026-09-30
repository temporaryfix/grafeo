//! Real aggregate state boundaries, with bounded generated input chunks.

use super::*;
use grafeo_common::memory::buffer::BufferManager;
use grafeo_core::execution::spill::{
    CleartextSpillRecordProvider, SpillFrameLimits, SpillIo, SpillIoOperation,
};
use grafeo_core::execution::{QueryCancellationHandle, QueryExecutionControl};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

const GROUPS: usize = 512;

fn tagged(lexical: &str, datatype: &str) -> Value {
    super::super::tagged_rdf_term(
        Value::RdfLiteral {
            lexical: lexical.into(),
            language: None,
            datatype: Some(datatype.into()),
        },
        Term::typed_literal(lexical, datatype),
    )
}

struct PhasedInput {
    next: usize,
    observed: Arc<AtomicUsize>,
}

impl Operator for PhasedInput {
    fn next(&mut self) -> OperatorResult {
        if self.next == GROUPS * 3 {
            return Ok(None);
        }
        let mut chunk = DataChunkBuilder::with_capacity(&[const { LogicalType::Any }; 4], 16);
        for index in self.next..(self.next + 16).min(GROUPS * 3) {
            let phase = index / GROUPS;
            let values = [
                Value::Int64(i64::try_from(index % GROUPS).unwrap()),
                tagged(["1e16", "1", "-1e16"][phase], Literal::XSD_DOUBLE),
                tagged(&phase.to_string(), Literal::XSD_INTEGER),
                tagged(&phase.to_string(), Literal::XSD_STRING),
            ];
            for (column, value) in values.into_iter().enumerate() {
                chunk.column_mut(column).unwrap().push_value(value);
            }
            chunk.advance_row();
            self.next += 1;
        }
        self.observed.store(self.next, AtomicOrdering::Relaxed);
        Ok(Some(chunk.finish()))
    }
    fn reset(&mut self) {
        self.next = 0;
    }
    fn name(&self) -> &'static str {
        "PhasedAggregateInput"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct ObservedRuns {
    input: Arc<AtomicUsize>,
    creates: AtomicUsize,
    first_create_input: AtomicUsize,
}

impl SpillIo for ObservedRuns {
    fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
        if operation == SpillIoOperation::Create
            && self.creates.fetch_add(1, AtomicOrdering::Relaxed) == 0
        {
            self.first_create_input.store(
                self.input.load(AtomicOrdering::Relaxed),
                AtomicOrdering::Relaxed,
            );
        }
        Ok(())
    }
    fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        // Observations/cancellation are atomic; returned ErrorKind has no heap.
        Some(0)
    }
    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }
}

fn aggregate_operator(observed: Arc<AtomicUsize>) -> RdfAggregateOperator {
    let mut sample = AggregateExpr::count(2);
    sample.function = AggregateFunction::Sample;
    let mut concat = AggregateExpr::count(3);
    concat.function = AggregateFunction::GroupConcat;
    concat.separator = Some("|".to_string());
    RdfAggregateOperator::new(
        Box::new(PhasedInput { next: 0, observed }),
        vec![0],
        vec![
            AggregateExpr::sum(1),
            AggregateExpr::avg(1),
            sample,
            concat,
            AggregateExpr::count_star(),
        ],
        vec![LogicalType::Any; 6],
        None,
    )
}

fn execute(resources: &QueryResourceContext, observed: Arc<AtomicUsize>) -> Vec<Vec<Value>> {
    let mut operator = aggregate_operator(observed);
    operator.install_resource_context(resources).unwrap();
    let mut rows = Vec::new();
    while let Some(chunk) = operator.next().unwrap() {
        assert_eq!(chunk.column_count(), 6);
        for row in chunk.selected_indices() {
            rows.push(
                (0..6)
                    .map(|column| chunk.column(column).unwrap().get_value(row).unwrap())
                    .collect(),
            );
        }
    }
    assert!(operator.next().unwrap().is_none());
    operator.reset();
    let mut repeated = Vec::new();
    while let Some(chunk) = operator.next().unwrap() {
        for row in chunk.selected_indices() {
            repeated.push(
                (0..6)
                    .map(|column| chunk.column(column).unwrap().get_value(row).unwrap())
                    .collect::<Vec<_>>(),
            );
        }
    }
    assert_eq!(repeated.len(), rows.len(), "reset row count");
    for (index, (repeated, original)) in repeated.iter().zip(&rows).enumerate() {
        assert_eq!(repeated, original, "reset group {index}");
    }
    drop(operator);
    rows
}

#[test]
fn exact_aggregate_spill_replays_partial_groups_in_original_ordinal_order() {
    let expected: Vec<_> = (0..GROUPS)
        .map(|group| {
            vec![
                Value::Int64(i64::try_from(group).unwrap()),
                Value::Float64(0.0),
                Value::Float64(0.0),
                tagged("0", Literal::XSD_INTEGER),
                Value::from("0|1|2"),
                Value::Int64(3),
            ]
        })
        .collect();
    let resident = BufferManager::with_budget(64 << 20);
    let resources = QueryResourceContext::new(resident.clone()).unwrap();
    let baseline = execute(&resources, Arc::new(AtomicUsize::new(0)));
    assert_eq!(baseline.len(), expected.len());
    for (index, (actual, expected)) in baseline.iter().zip(&expected).enumerate() {
        assert_eq!(actual, expected, "resident group {index}");
    }
    assert_eq!(resident.allocated(), 0);

    let input = Arc::new(AtomicUsize::new(0));
    let io = Arc::new(ObservedRuns {
        input: input.clone(),
        creates: AtomicUsize::new(0),
        first_create_input: AtomicUsize::new(usize::MAX),
    });
    let directory = tempfile::tempdir().unwrap();
    let memory = BufferManager::with_budget(4 << 20);
    let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
        directory.path(),
        memory.clone(),
        QueryExecutionControl::new().token(),
        Arc::new(CleartextSpillRecordProvider),
        SpillFrameLimits::format_max(),
        io.clone(),
        grafeo_core::execution::SpillDiskQuota::new(u64::MAX),
    );
    let actual = execute(&resources, input);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual, expected, "group {index}");
    }
    assert!(
        io.creates.load(AtomicOrdering::Relaxed) > 0,
        "fixture must create real runs"
    );
    assert!(
        io.first_create_input.load(AtomicOrdering::Relaxed) < GROUPS * 2,
        "spill must precede the cancellation-sensitive final operand phase"
    );
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    assert_eq!(memory.allocated(), 0);
}

#[test]
fn non_distinct_aggregate_spill_avoids_membership_sort_publication() {
    let input = Arc::new(AtomicUsize::new(0));
    let io = Arc::new(ObservedRuns {
        input: input.clone(),
        creates: AtomicUsize::new(0),
        first_create_input: AtomicUsize::new(usize::MAX),
    });
    let directory = tempfile::tempdir().unwrap();
    let memory = BufferManager::with_budget(4 << 20);
    let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
        directory.path(),
        memory.clone(),
        QueryExecutionControl::new().token(),
        Arc::new(CleartextSpillRecordProvider),
        SpillFrameLimits::format_max(),
        io.clone(),
        grafeo_core::execution::SpillDiskQuota::new(u64::MAX),
    );
    let mut operator = aggregate_operator(input.clone());
    operator.install_resource_context(&resources).unwrap();
    let mut rows = Vec::new();
    while let Some(chunk) = operator.next().unwrap() {
        assert_eq!(chunk.column_count(), 6);
        for row in chunk.selected_indices() {
            rows.push(
                (0..6)
                    .map(|column| chunk.column(column).unwrap().get_value(row).unwrap())
                    .collect::<Vec<_>>(),
            );
        }
    }
    assert!(operator.next().unwrap().is_none());
    assert_eq!(input.load(AtomicOrdering::Relaxed), GROUPS * 3);
    assert_eq!(rows.len(), GROUPS);
    for (group, row) in rows.iter().enumerate() {
        assert_eq!(
            row,
            &vec![
                Value::Int64(i64::try_from(group).unwrap()),
                Value::Float64(0.0),
                Value::Float64(0.0),
                tagged("0", Literal::XSD_INTEGER),
                Value::from("0|1|2"),
                Value::Int64(3),
            ],
            "group {group}"
        );
    }
    drop(operator);
    drop(rows);
    assert!(io.first_create_input.load(AtomicOrdering::Relaxed) < GROUPS * 2);
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    assert_eq!(memory.allocated(), 0);
    let (published_bytes, runs, partitions, _) = manager.profile_totals();
    println!(
        "non-DISTINCT aggregate publication: input_rows={} groups={GROUPS} bytes={published_bytes} runs={runs} partitions={partitions}",
        GROUPS * 3,
    );
    assert!(published_bytes > 0, "must publish actual spill records");
    assert!(runs > 0, "must execute external sort");
    assert_eq!(partitions, 0);
    // Measured redundant-pass baseline: 3,476,880 published bytes / 18 runs.
    // A 3 MiB budget requires at least 9.5% less publication while allowing
    // the necessary retained sorts. This fixed-fixture regression budget
    // includes intermediate merges; it makes no general complexity claim.
    const PUBLISHED_BYTE_CAP: u64 = 3 << 20;
    assert!(
        published_bytes <= PUBLISHED_BYTE_CAP,
        "non-DISTINCT aggregation published {published_bytes} bytes in {runs} runs; cap {PUBLISHED_BYTE_CAP} requires eliminating the unnecessary membership pass"
    );
}

struct AggregateFault {
    operation: SpillIoOperation,
    armed: AtomicBool,
    fired: AtomicBool,
    cancellation: Option<QueryCancellationHandle>,
    memory: Arc<BufferManager>,
    charged_at_fault: AtomicUsize,
}

impl SpillIo for AggregateFault {
    fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
        if operation == self.operation && self.armed.swap(false, AtomicOrdering::SeqCst) {
            self.fired.store(true, AtomicOrdering::SeqCst);
            self.charged_at_fault
                .store(self.memory.allocated(), AtomicOrdering::SeqCst);
            if let Some(cancellation) = &self.cancellation {
                cancellation.cancel();
            } else {
                // ErrorKind stores no heap payload; both reader hook bounds
                // below include every successful/error outcome without growth.
                return Err(std::io::ErrorKind::PermissionDenied.into());
            }
        }
        Ok(())
    }
    fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        // Observations/cancellation are atomic; returned ErrorKind has no heap.
        Some(0)
    }
    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }
}

#[test]
fn aggregate_spill_faults_and_cancellation_are_terminal_and_release_owners() {
    for operation in [
        SpillIoOperation::WritePayload,
        SpillIoOperation::ReadPayload,
        SpillIoOperation::Sync,
    ] {
        for cancel in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let memory = BufferManager::with_budget(4 << 20);
            let control = QueryExecutionControl::new();
            let io = Arc::new(AggregateFault {
                operation,
                armed: AtomicBool::new(true),
                fired: AtomicBool::new(false),
                cancellation: cancel.then(|| control.cancellation_handle()),
                memory: memory.clone(),
                charged_at_fault: AtomicUsize::new(0),
            });
            let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
                directory.path(),
                memory.clone(),
                control.token(),
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
                io.clone(),
                grafeo_core::execution::SpillDiskQuota::new(u64::MAX),
            );
            let mut operator = aggregate_operator(Arc::new(AtomicUsize::new(0)));
            operator.install_resource_context(&resources).unwrap();
            let error = operator
                .next()
                .expect_err("fault must prevent aggregate output publication");
            assert!(
                io.fired.load(AtomicOrdering::SeqCst),
                "{operation:?}: failure must cross selected real spill boundary"
            );
            assert!(
                io.charged_at_fault.load(AtomicOrdering::SeqCst) > 0,
                "spill callback must retain its query admission"
            );
            if cancel {
                assert!(
                    error.to_string().to_ascii_lowercase().contains("cancel"),
                    "{operation:?}: {error}"
                );
            } else {
                assert!(
                    error
                        .to_string()
                        .to_ascii_lowercase()
                        .contains("permission"),
                    "{operation:?}: {error}"
                );
            }
            assert!(
                operator.next().is_err(),
                "failed aggregate must not publish a later successful prefix"
            );
            drop(error);
            assert_eq!(
                manager.active_file_count(),
                0,
                "{operation:?}: active files after terminal cleanup"
            );
            assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
            assert_eq!(
                memory.allocated(),
                0,
                "{operation:?}: grants after dropping primary failure"
            );
            // Reset reuses the input child, but cancellation authority belongs
            // to the old admitted leaf. A new execution must admit a fresh leaf.
            manager.finish_query().unwrap();
            operator.reset();
            let (fresh, fresh_manager) = crate::spill_crypto::admitted_spill_test_resources(
                directory.path(),
                memory.clone(),
                QueryExecutionControl::new().token(),
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
                io.clone(),
                grafeo_core::execution::SpillDiskQuota::new(u64::MAX),
            );
            assert_ne!(fresh.query_id(), resources.query_id());
            assert_ne!(fresh_manager.spill_dir(), manager.spill_dir());
            operator.install_resource_context(&fresh).unwrap();
            let mut rows = 0;
            while let Some(chunk) = operator.next().unwrap() {
                rows += chunk.row_count();
            }
            assert_eq!(
                rows, GROUPS,
                "{operation:?}: reset must execute the retained child"
            );
            drop(operator);
            assert_eq!(fresh_manager.active_file_count(), 0);
            assert_eq!(fresh_manager.disk_stats().reserved_live_bytes, 0);
            fresh_manager.finish_query().unwrap();
            assert_eq!(memory.allocated(), 0);
        }
    }
}

struct FinalConcatInput {
    next: usize,
    observed: Arc<AtomicUsize>,
}

struct HotConcatInput {
    next: usize,
    hot_rows: usize,
    observed: Arc<AtomicUsize>,
}

impl Operator for HotConcatInput {
    fn next(&mut self) -> OperatorResult {
        let end = GROUPS + self.hot_rows;
        if self.next == end {
            return Ok(None);
        }
        let mut chunk = DataChunkBuilder::with_capacity(&[const { LogicalType::Any }; 2], 16);
        for index in self.next..(self.next + 16).min(end) {
            let group = index.min(GROUPS);
            let text = if index < GROUPS {
                "cold".to_string()
            } else {
                format!("{:04}", index - GROUPS)
            };
            chunk
                .column_mut(0)
                .unwrap()
                .push_value(Value::Int64(i64::try_from(group).unwrap()));
            chunk
                .column_mut(1)
                .unwrap()
                .push_value(tagged(&text, Literal::XSD_STRING));
            chunk.advance_row();
            self.next += 1;
        }
        self.observed.store(self.next, AtomicOrdering::Relaxed);
        Ok(Some(chunk.finish()))
    }
    fn reset(&mut self) {
        self.next = 0;
    }
    fn name(&self) -> &'static str {
        "HotConcatPhasedInput"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

fn concat_expression() -> AggregateExpr {
    let mut expression = AggregateExpr::count(1);
    expression.function = AggregateFunction::GroupConcat;
    expression.separator = Some("|".to_string());
    expression
}

#[test]
fn concat_visit_counter_detects_repeated_growing_state_scans() {
    let mut accumulator = RdfAccumulator::new(&concat_expression());
    bounded::take_concat_operand_visits();
    for ordinal in 0..32 {
        accumulator
            .update_at(Some(tagged("x", Literal::XSD_STRING)), None, None, ordinal)
            .unwrap();
        bounded::accumulator_retained_bytes(&accumulator).unwrap();
    }
    assert_eq!(bounded::take_concat_operand_visits(), 32 * 33 / 2);
}

#[test]
fn hot_group_concat_spill_fold_has_linear_operand_work() {
    for hot_rows in [1024, 2048, 4096] {
        let input = Arc::new(AtomicUsize::new(0));
        let io = Arc::new(ObservedRuns {
            input: input.clone(),
            creates: AtomicUsize::new(0),
            first_create_input: AtomicUsize::new(usize::MAX),
        });
        let directory = tempfile::tempdir().unwrap();
        let memory = BufferManager::with_budget(2 << 20);
        let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
            directory.path(),
            memory.clone(),
            QueryExecutionControl::new().token(),
            Arc::new(CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
            io.clone(),
            grafeo_core::execution::SpillDiskQuota::new(u64::MAX),
        );
        let mut operator = RdfAggregateOperator::new(
            Box::new(HotConcatInput {
                next: 0,
                hot_rows,
                observed: input,
            }),
            vec![0],
            vec![concat_expression()],
            vec![LogicalType::Any; 2],
            None,
        );
        operator.install_resource_context(&resources).unwrap();
        bounded::take_concat_operand_visits();
        let mut actual = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                actual.push(vec![
                    chunk.column(0).unwrap().get_value(row).unwrap(),
                    chunk.column(1).unwrap().get_value(row).unwrap(),
                ]);
            }
        }
        let visits = bounded::take_concat_operand_visits();
        let mut expected: Vec<_> = (0..GROUPS)
            .map(|group| {
                vec![
                    Value::Int64(i64::try_from(group).unwrap()),
                    Value::from("cold"),
                ]
            })
            .collect();
        expected.push(vec![
            Value::Int64(i64::try_from(GROUPS).unwrap()),
            Value::from(
                (0..hot_rows)
                    .map(|ordinal| format!("{ordinal:04}"))
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
        ]);
        assert_eq!(actual, expected, "hot operands={hot_rows}");
        assert!(io.creates.load(AtomicOrdering::Relaxed) > 0, "must spill");
        assert!(
            io.first_create_input.load(AtomicOrdering::Relaxed) < GROUPS + hot_rows,
            "spill must begin before the hot group is fully ingested"
        );
        assert!(visits >= hot_rows, "actual operand loops must be counted");
        assert!(
            visits <= 8 * (GROUPS + hot_rows),
            "concat fold rescans growing state: hot operands={hot_rows}, visits={visits}"
        );
        drop(operator);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(memory.allocated(), 0);
    }
}

impl Operator for FinalConcatInput {
    fn next(&mut self) -> OperatorResult {
        const ROWS: usize = 16_384;
        if self.next == ROWS {
            return Ok(None);
        }
        let mut chunk = DataChunkBuilder::with_capacity(&[LogicalType::Any], 16);
        for _ in self.next..(self.next + 16).min(ROWS) {
            chunk.column_mut(0).unwrap().push_value(tagged(
                "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                Literal::XSD_STRING,
            ));
            chunk.advance_row();
            self.next += 1;
        }
        self.observed.store(self.next, AtomicOrdering::Relaxed);
        Ok(Some(chunk.finish()))
    }
    fn reset(&mut self) {
        self.next = 0;
    }
    fn name(&self) -> &'static str {
        "AggregateFinalConcatInput"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[test]
fn oversized_final_concat_is_denied_after_input_before_result_materialization() {
    let observed = Arc::new(AtomicUsize::new(0));
    let mut expression = AggregateExpr::count(0);
    expression.function = AggregateFunction::GroupConcat;
    expression.separator = Some(String::new());
    let mut operator = RdfAggregateOperator::new(
        Box::new(FinalConcatInput {
            next: 0,
            observed: observed.clone(),
        }),
        Vec::new(),
        vec![expression],
        vec![LogicalType::Any],
        None,
    );
    // Stored 32-byte operands plus their vector fit. The final reference
    // vector, joined string and conversion overlap exceed the remaining grant.
    let memory = BufferManager::with_budget(2 << 20);
    let resources = QueryResourceContext::new(memory.clone()).unwrap();
    operator.install_resource_context(&resources).unwrap();
    let error = operator
        .next()
        .expect_err("oversized final copy must be denied");
    assert_eq!(
        observed.load(AtomicOrdering::Relaxed),
        16_384,
        "witness must reach finalization, not fail while reading an operand"
    );
    assert!(matches!(error, OperatorError::ResidentMemory(_)), "{error}");
    assert!(operator.next().is_err());
    drop(error);
    assert_eq!(memory.allocated(), 0);
    drop(operator);
    assert_eq!(memory.allocated(), 0);
}
