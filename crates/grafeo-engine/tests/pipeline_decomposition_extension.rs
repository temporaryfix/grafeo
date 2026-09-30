//! Cross-crate proof that engine-local operators can extend core pipeline decomposition.

use grafeo_common::memory::buffer::BufferManager;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::memory::{QueryResourceContext, QueryResourceContextError};
use grafeo_core::execution::operators::{
    Operator, OperatorError, OperatorPipelineDecomposition, OperatorResult,
};
use grafeo_core::execution::pipeline::{PushOperator, Sink};
use grafeo_core::execution::pipeline_convert::convert_to_pipeline_with_resources;
use std::sync::{Arc, Mutex};

struct ExternalSource;

impl Operator for ExternalSource {
    fn next(&mut self) -> OperatorResult {
        Ok(None)
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "ExternalSource"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct ExternalPushBreaker;

impl PushOperator for ExternalPushBreaker {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        sink.consume(chunk)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        sink.finalize()
    }

    fn name(&self) -> &'static str {
        "ExternalPushBreaker"
    }
}

struct ExternalPullBreaker {
    child: Box<dyn Operator>,
    installed_query_ids: Arc<Mutex<Vec<u64>>>,
}

impl Operator for ExternalPullBreaker {
    fn next(&mut self) -> OperatorResult {
        self.child.next()
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        // Deliberately collide with a core operator. The old converter would
        // trust this debug label and panic while downcasting our external type.
        "Sort"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        self.installed_query_ids
            .lock()
            .unwrap()
            .push(resources.query_id().get());
        self.child.install_resource_context(resources)
    }

    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        resources: &QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, QueryResourceContextError> {
        self.installed_query_ids
            .lock()
            .unwrap()
            .push(resources.query_id().get());
        Ok(OperatorPipelineDecomposition::unary(
            self.child,
            Box::new(ExternalPushBreaker),
        ))
    }
}

#[test]
fn engine_local_operator_decomposes_without_core_name_matching_or_downcast() {
    let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
    let installed_query_ids = Arc::new(Mutex::new(Vec::new()));
    let root: Box<dyn Operator> = Box::new(ExternalPullBreaker {
        child: Box::new(ExternalSource),
        installed_query_ids: Arc::clone(&installed_query_ids),
    });

    let (source, push) = convert_to_pipeline_with_resources(root, &resources).unwrap();

    assert_eq!(source.name(), "ExternalSource");
    assert_eq!(push.len(), 1);
    assert_eq!(push[0].name(), "ExternalPushBreaker");
    assert_eq!(
        *installed_query_ids.lock().unwrap(),
        vec![resources.query_id().get(), resources.query_id().get()]
    );
}
