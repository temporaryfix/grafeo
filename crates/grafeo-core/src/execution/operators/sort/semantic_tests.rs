//! Terminal behavior of the real resident semantic sort caller.

use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use grafeo_common::memory::buffer::BufferManager;
use grafeo_common::types::{LogicalType, Value};

use super::{AccountedValueComparator, SemanticComparisonError, SortKey, SortOperator};
use crate::execution::operators::{Operator, OperatorError, OperatorResult};
use crate::execution::{
    DataChunk, QueryCancellationHandle, QueryExecutionControl, QueryResourceContext,
};

struct Input(Option<DataChunk>);

impl Operator for Input {
    fn next(&mut self) -> OperatorResult {
        Ok(self.0.take())
    }

    fn reset(&mut self) {
        self.0 = None;
    }

    fn name(&self) -> &'static str {
        "SemanticSortTestInput"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct TerminalComparator {
    calls: AtomicUsize,
    memory: Arc<BufferManager>,
    cancel: Option<QueryCancellationHandle>,
}

impl AccountedValueComparator for TerminalComparator {
    fn scratch_bytes(
        &self,
        _left: Option<&Value>,
        _right: Option<&Value>,
    ) -> Result<usize, SemanticComparisonError> {
        Ok(4096)
    }

    fn compare(
        &self,
        _left: Option<&Value>,
        _right: Option<&Value>,
    ) -> Result<Ordering, SemanticComparisonError> {
        self.calls.fetch_add(1, AtomicOrdering::Relaxed);
        assert!(
            self.memory.allocated() >= 4096,
            "comparison scratch was not admitted"
        );
        if let Some(cancel) = &self.cancel {
            cancel.cancel();
            Ok(Ordering::Equal)
        } else {
            Err(SemanticComparisonError::Invalid(
                "injected semantic failure",
            ))
        }
    }
}

fn sort(comparator: Arc<TerminalComparator>, resources: &QueryResourceContext) -> SortOperator {
    let mut input = DataChunk::with_capacity(&[LogicalType::Int64], 16);
    for value in (0..16).rev() {
        input.column_mut(0).unwrap().push_int64(value);
    }
    input.set_count(16);
    let mut sort = SortOperator::new(
        Box::new(Input(Some(input))),
        vec![SortKey::ascending(0)],
        vec![LogicalType::Int64],
    )
    .with_semantic_comparator(comparator);
    sort.install_resource_context(resources).unwrap();
    sort
}

#[test]
fn provider_failure_is_structured_sticky_and_releases_grants() {
    let memory = BufferManager::with_budget(1 << 20);
    let resources = QueryResourceContext::new(memory.clone()).unwrap();
    let comparator = Arc::new(TerminalComparator {
        calls: AtomicUsize::new(0),
        memory: memory.clone(),
        cancel: None,
    });
    let mut sort = sort(comparator.clone(), &resources);
    for _ in 0..2 {
        assert!(matches!(
            sort.next(),
            Err(OperatorError::ResidentContainerInvariant {
                container: "semantic comparison",
                message: "injected semantic failure",
            })
        ));
        assert_eq!(memory.allocated(), 0);
    }
    assert_eq!(comparator.calls.load(AtomicOrdering::Relaxed), 1);
    sort.reset();
    assert_eq!(memory.allocated(), 0);
}

#[test]
fn cancellation_between_semantic_comparisons_stops_and_releases_grants() {
    let memory = BufferManager::with_budget(1 << 20);
    let control = QueryExecutionControl::new();
    let resources =
        QueryResourceContext::new_with_cancellation(memory.clone(), control.token()).unwrap();
    let comparator = Arc::new(TerminalComparator {
        calls: AtomicUsize::new(0),
        memory: memory.clone(),
        cancel: Some(control.cancellation_handle()),
    });
    let mut sort = sort(comparator.clone(), &resources);
    for _ in 0..2 {
        assert!(matches!(sort.next(), Err(OperatorError::QueryCancelled(_))));
        assert_eq!(memory.allocated(), 0);
    }
    assert_eq!(
        comparator.calls.load(AtomicOrdering::Relaxed),
        1,
        "a second semantic comparison ran after cancellation"
    );
    sort.reset();
    assert_eq!(memory.allocated(), 0);
}
