//! Regression controls for the logical output-column contract.

#![cfg(feature = "lpg")]

use super::super::Planner;
use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, ExpandDirection, ExpandOp, LogicalExpression,
    LogicalOperator, LogicalPlan, NodeScanOp, PathMode, PathSearch, ProjectOp, Projection,
};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_core::graph::lpg::LpgStore;
use std::sync::Arc;

fn planner() -> Planner {
    let store = Arc::new(LpgStore::new().unwrap());
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    store.create_edge(a, b, "REL");
    Planner::new(store as Arc<dyn GraphStoreSearch>)
}

fn shortest_path(path_alias: Option<&str>) -> LogicalOperator {
    LogicalOperator::Expand(ExpandOp {
        from_variable: "a".into(),
        to_variable: "b".into(),
        edge_variable: None,
        direction: ExpandDirection::Outgoing,
        edge_types: vec!["REL".into()],
        min_hops: 1,
        max_hops: Some(3),
        input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
            variable: "a".into(),
            label: Some("Node".into()),
            input: None,
        })),
        path_alias: path_alias.map(str::to_owned),
        path_mode: PathMode::Trail,
        path_search: PathSearch::Shortest {
            k: 1,
            groups: false,
        },
        edge_predicate: None,
        path_predicate: None,
    })
}

#[test]
fn shortest_named_path_requires_alias_and_each_auxiliary_binding() {
    let planner = planner();
    let op = shortest_path(Some("p"));
    let required = [
        "a",
        "b",
        "p",
        "_path_length_p",
        "_path_nodes_p",
        "_path_edges_p",
    ];
    for missing in required {
        let columns = required
            .iter()
            .filter(|column| **column != missing)
            .map(|column| (*column).to_owned())
            .collect::<Vec<_>>();
        assert!(
            planner.validate_output_bindings(&op, &columns).is_err(),
            "omitting {missing} must invalidate a named shortest-path output"
        );
    }
}

#[test]
fn projection_and_aggregation_may_drop_path_bindings() {
    let planner = planner();
    let input = shortest_path(Some("p"));
    let project = LogicalOperator::Project(ProjectOp {
        projections: vec![
            Projection {
                expression: LogicalExpression::Variable("a".into()),
                alias: None,
            },
            Projection {
                expression: LogicalExpression::Variable("b".into()),
                alias: None,
            },
        ],
        input: Box::new(input.clone()),
        pass_through_input: false,
    });
    assert!(
        planner
            .validate_output_bindings(&project, &["a".into(), "b".into()])
            .is_ok()
    );

    let aggregate = LogicalOperator::Aggregate(AggregateOp {
        group_by: vec![],
        aggregates: vec![AggregateExpr {
            function: AggregateFunction::Count,
            expression: None,
            expression2: None,
            distinct_key: None,
            distinct: false,
            alias: Some("n".into()),
            percentile: None,
            separator: None,
        }],
        input: Box::new(input),
        having: None,
    });
    assert!(
        planner
            .validate_output_bindings(&aggregate, &["n".into()])
            .is_ok()
    );
}

#[test]
fn plan_entrypoints_accept_a_real_named_shortest_path() {
    for mode in [0_u8, 1, 2] {
        let planner = planner();
        let logical = LogicalPlan::new(shortest_path(Some("p")));
        match mode {
            0 => assert!(planner.plan(&logical).is_ok()),
            1 => assert!(planner.plan_profiled(&logical).is_ok()),
            _ => assert!(planner.plan_adaptive(&logical).is_ok()),
        }
    }
}

#[test]
fn fused_expand_rejects_missing_intermediate_binding() {
    let planner = planner();
    let inner = shortest_path(Some("p1"));
    let outer = LogicalOperator::Expand(ExpandOp {
        from_variable: "b".into(),
        to_variable: "c".into(),
        edge_variable: None,
        direction: ExpandDirection::Outgoing,
        edge_types: vec!["REL".into()],
        min_hops: 1,
        max_hops: Some(1),
        input: Box::new(inner),
        path_alias: Some("p2".into()),
        path_mode: PathMode::Trail,
        path_search: PathSearch::All,
        edge_predicate: None,
        path_predicate: None,
    });
    let columns = [
        "a",
        "c",
        "p1",
        "_path_length_p1",
        "_path_nodes_p1",
        "_path_edges_p1",
        "p2",
        "_path_length_p2",
        "_path_nodes_p2",
        "_path_edges_p2",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    assert!(planner.validate_output_bindings(&outer, &columns).is_err());
}

#[test]
fn finish_drops_bindings_but_regular_skip_preserves_them() {
    use crate::query::plan::{CountExpr, SkipOp};
    let planner = planner();
    for count in [0, 1, usize::MAX] {
        let op = LogicalOperator::Skip(SkipOp {
            count: CountExpr::Literal(count),
            input: Box::new(shortest_path(Some("p"))),
        });
        assert_eq!(
            planner.validate_output_bindings(&op, &[]).is_ok(),
            count == usize::MAX,
            "only FINISH declares all consumed child bindings absent"
        );
    }
}
