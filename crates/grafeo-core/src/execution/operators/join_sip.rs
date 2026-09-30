//! Join-driven SIP: collect build-side node ids, then expand with [`SipTarget::TargetSet`].
//!
//! Honours `design-proposals-factorization-wcoj` proposal B
//! ("Join-driven TargetSet from early hash join bindings") without
//! materializing a hash-join Cartesian of the expand side.

use grafeo_common::utils::hash::FxHashSet;

use super::{LazyFactorizedChainOperator, Operator, OperatorResult, SipTarget};

/// Build-side node list ⋈ expand chain, SIP on `hop`.
pub struct JoinSipExpandOperator {
    build: Option<Box<dyn Operator>>,
    build_column: usize,
    expand: Option<LazyFactorizedChainOperator>,
    hop: usize,
    emitting: Option<LazyFactorizedChainOperator>,
}

impl JoinSipExpandOperator {
    /// `build` yields node ids in `build_column`; those become the allow-list
    /// for expand hop `hop`.
    pub fn new(
        build: Box<dyn Operator>,
        build_column: usize,
        expand: LazyFactorizedChainOperator,
        hop: usize,
    ) -> Self {
        Self {
            build: Some(build),
            build_column,
            expand: Some(expand),
            hop,
            emitting: None,
        }
    }

    fn collect_and_bind(&mut self) -> Result<(), super::OperatorError> {
        let Some(mut build) = self.build.take() else {
            return Ok(());
        };
        let Some(expand) = self.expand.take() else {
            return Ok(());
        };
        let mut allow = FxHashSet::default();
        while let Some(chunk) = build.next()? {
            let Some(col) = chunk.column(self.build_column) else {
                continue;
            };
            for i in 0..chunk.row_count() {
                if let Some(id) = col.get_node_id(i) {
                    allow.insert(id);
                }
            }
        }
        self.emitting = Some(expand.with_sip(SipTarget::TargetSet {
            hop: self.hop,
            allow,
        }));
        Ok(())
    }
}

impl Operator for JoinSipExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.emitting.is_none() {
            self.collect_and_bind()?;
        }
        match self.emitting.as_mut() {
            Some(exp) => exp.next(),
            None => Ok(None),
        }
    }

    fn reset(&mut self) {
        self.emitting = None;
    }

    fn name(&self) -> &'static str {
        "JoinSipExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::operators::Operator;
    use crate::execution::operators::factorized_expand::ExpandStep;
    use crate::graph::Direction;
    use crate::graph::GraphStoreSearch;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::LogicalType;
    use std::sync::Arc;

    struct OneChunk(Option<DataChunk>);
    impl Operator for OneChunk {
        fn next(&mut self) -> OperatorResult {
            Ok(self.0.take())
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "OneChunk"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn join_sip_keeps_only_build_ids() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let keep = store.create_node(&["C"]);
        let drop = store.create_node(&["C"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, keep, "R");
        store.create_edge(b, drop, "R");

        let mut seed = DataChunk::with_capacity(&[LogicalType::Node], 1);
        seed.column_mut(0).unwrap().push_node_id(a);
        seed.set_count(1);
        let expand = LazyFactorizedChainOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            Box::new(OneChunk(Some(seed))),
            vec![
                ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".into()],
                    sip: None,
                    need_edge: true,
                },
                ExpandStep {
                    source_column: 1,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".into()],
                    sip: None,
                    need_edge: true,
                },
            ],
        );

        let mut build = DataChunk::with_capacity(&[LogicalType::Node], 1);
        build.column_mut(0).unwrap().push_node_id(keep);
        build.set_count(1);

        let mut op = JoinSipExpandOperator::new(Box::new(OneChunk(Some(build))), 0, expand, 1);
        let mut rows = 0usize;
        while let Some(c) = op.next().unwrap() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }
}
