//! Acceptance: what a bounded path pattern is allowed to return.
//!
//! The unified path contract is exercised through public queries. Every test
//! initially left an endpoint free, because pinning both is what hid the
//! original defect: the operator's source-by-target cross product collapses to a
//! single row and its habit of emitting unreachable targets cannot show.
//!
//! Expected answers come from `support::reachable_within`, which walks the
//! fixture's own edge table rather than asking the engine — so a passing test
//! means the engine agrees with the fixture, not with another code path that
//! might be wrong in the same way.

#![cfg(feature = "lpg")]

mod support;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use support::{adversarial_path_graph, reachable_within, sorted_ids, walk_count};

#[test]
fn fixture_reference_sets_are_what_the_fixture_says() {
    // Guards the guard: if this drifts, every expectation below is wrong.
    // s --REL--> a, b;  a --> d, h;  b --> d, g;  d --> e;  e <-> f;  g --> g;
    // h --> d;  s --OTHER--> x;  u --REL--> v is a separate component.
    assert_eq!(reachable_within("s", 1, 1, "REL"), set(&["a", "b"]));
    assert_eq!(
        reachable_within("s", 1, 2, "REL"),
        set(&["a", "b", "d", "g", "h"])
    );
    assert_eq!(
        reachable_within("s", 1, 3, "REL"),
        set(&["a", "b", "d", "e", "g", "h"])
    );

    // No route leads back to s, so s appearing as its own target is always a
    // defect on this fixture and never a legitimate cycle.
    assert!(!reachable_within("s", 1, 6, "REL").contains("s"));
    // The disconnected component and the OTHER-typed target are never reachable.
    for unreachable in ["u", "v", "x"] {
        assert!(!reachable_within("s", 1, 6, "REL").contains(unreachable));
    }
    // Walks outnumber reachable nodes, which is what pruning must exploit.
    assert_eq!(walk_count("s", 1, 3, "REL"), 10);
    assert_eq!(reachable_within("s", 1, 3, "REL").len(), 6);
}

#[test]
fn all_anonymous_hops_preserve_parallel_edge_multiplicity() {
    let db = adversarial_path_graph();
    let session = db.session();
    for query in [
        "MATCH (a:Node {id: 'a'})<-[:REL]-(b)-[]->(c) RETURN c.id",
        "MATCH (a:Node {id: 'a'})<-[:REL]-(b)-[edge]->(c) RETURN c.id, id(edge)",
    ] {
        let result = session.execute(query).unwrap();
        assert_eq!(sorted_ids(&result), ["a", "b", "x", "x"]);
        if result.columns.len() == 2 {
            let edges: std::collections::BTreeSet<_> = result
                .rows()
                .iter()
                .map(|row| row[1].as_int64().unwrap())
                .collect();
            assert_eq!(
                edges.len(),
                4,
                "each duplicate destination has its own raw edge"
            );
        }
    }
}

#[test]
fn bounded_any_shortest_returns_only_reachable_targets() {
    let db = adversarial_path_graph();
    let session = db.session();

    let result = session
        .execute("MATCH ANY SHORTEST (a:Node {id: 's'})-[:REL*1..2]->(b) RETURN b.id")
        .expect("query must succeed");

    let expected: Vec<String> = reachable_within("s", 1, 2, "REL").into_iter().collect();
    assert_eq!(
        sorted_ids(&result),
        expected,
        "a bounded shortest-path pattern must return exactly the nodes reachable \
         within its bound: no unreachable component, no start node, nothing past the bound"
    );
}

#[test]
fn bounded_any_shortest_honours_its_hop_bound() {
    let db = adversarial_path_graph();
    let session = db.session();

    let result = session
        .execute("MATCH ANY SHORTEST (a:Node {id: 's'})-[:REL*1..1]->(b) RETURN b.id")
        .expect("query must succeed");

    assert_eq!(
        sorted_ids(&result),
        vec!["a".to_string(), "b".to_string()],
        "a one-hop bound must exclude everything two hops away"
    );
}

#[test]
fn shortest_path_binds_its_path_variable() {
    let db = adversarial_path_graph();
    let session = db.session();

    // d sits at the far corner of the diamond: two distinct shortest routes,
    // s-a-d and s-b-d, each of length 2.
    let result = session
        .execute(
            "MATCH p = ANY SHORTEST (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) \
             RETURN [n IN nodes(p) | n.id]",
        )
        .expect("nodes(p) must be available on a shortest-path pattern");

    assert_eq!(result.rows().len(), 1, "ANY SHORTEST returns one path");
    let grafeo_common::types::Value::List(ids) = &result.rows()[0][0] else {
        panic!(
            "nodes(p) must project a list, got {:?}",
            result.rows()[0][0]
        );
    };
    let ids: Vec<String> = ids
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("node id must be a string: {value:?}"))
        })
        .map(str::to_owned)
        .collect();
    assert_eq!(ids.len(), 3, "a length-2 path visits three nodes: {ids:?}");
    assert_eq!(ids.first().map(String::as_str), Some("s"));
    assert_eq!(ids.last().map(String::as_str), Some("d"));
}

#[test]
fn shortest_path_reports_its_length() {
    let db = adversarial_path_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH p = ANY SHORTEST (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) \
             RETURN length(p)",
        )
        .expect("length(p) must be available");

    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::from(2i64),
        "the shortest route to d is two hops, not the three-hop way round through h"
    );
}

#[test]
fn all_shortest_returns_every_minimal_path() {
    let db = adversarial_path_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH p = ALL SHORTEST (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) \
             RETURN [n IN nodes(p) | n.id]",
        )
        .expect("query must succeed");

    assert_eq!(
        result.rows().len(),
        2,
        "the diamond gives two shortest routes to d, s-a-d and s-b-d"
    );
    let mut routes: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            grafeo_common::types::Value::List(ids) => ids
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .unwrap_or_else(|| panic!("node id must be a string: {value:?}"))
                })
                .map(str::to_owned)
                .collect::<Vec<_>>()
                .join("-"),
            other => panic!("nodes(p) must project a list, got {other:?}"),
        })
        .collect();
    routes.sort();
    assert_eq!(routes, vec!["s-a-d".to_string(), "s-b-d".to_string()]);
}

#[test]
fn shortest_prefix_k_groups_and_keep_are_public() {
    let db = adversarial_path_graph();
    let session = db.session();

    let one = session
        .execute("MATCH SHORTEST 1 (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) RETURN b.id")
        .expect("SHORTEST 1 must execute");
    assert_eq!(
        one.rows().len(),
        1,
        "k=1 keeps one route to the fixed target"
    );

    let two = session
        .execute("MATCH SHORTEST 2 (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) RETURN b.id")
        .expect("SHORTEST 2");
    assert_eq!(two.rows().len(), 2);

    let grouped = session
        .execute("MATCH SHORTEST 2 GROUPS (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) RETURN b.id")
        .expect("SHORTEST k GROUPS must execute");
    assert_eq!(
        grouped.rows().len(),
        3,
        "two admitted lengths keep both length-2 routes and the length-3 route"
    );

    let trail = session
        .execute(
            "MATCH p = ALL SHORTEST (a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'}) \
             KEEP DIFFERENT EDGES RETURN [n IN nodes(p) | n.id]",
        )
        .expect("KEEP DIFFERENT EDGES must execute");
    assert_eq!(
        trail.rows().len(),
        2,
        "trail shortest keeps the diamond routes"
    );
    for (keep, expected) in [
        ("KEEP REPEATABLE ELEMENTS", vec![3i64, 4]),
        ("KEEP DIFFERENT EDGES", vec![3i64]),
    ] {
        let result = session.execute(&format!("MATCH p = SHORTEST 2 GROUPS (a:Node {{id: 's'}})-[:REL*3..4]->(b:Node {{id: 'g'}}) {keep} RETURN length(p)")).expect("KEEP cycle witness");
        let mut lengths: Vec<_> = result
            .rows()
            .iter()
            .map(|row| row[0].as_int64().expect("length"))
            .collect();
        lengths.sort_unstable();
        assert_eq!(
            lengths, expected,
            "{keep} must govern repeated self-loop admission"
        );
    }
}

#[test]
fn shortest_edge_predicates_filter_shortcuts_and_every_interior_edge() {
    let db = weighted_path_graph();
    let session = db.session();

    // The direct shortcut and the s-x edge are forbidden. Only s-a-d has all
    // edges at cost 1, so checking the returned nodes proves the predicate is
    // applied at every transition, rather than only at the final edge.
    for query in [
        "MATCH p = ALL SHORTEST (a:S {id: 's'})-[:R*1..3 {cost: 1}]->(b:T {id: 'd'}) RETURN [n IN nodes(p) | n.id]",
        "MATCH p = ALL SHORTEST (a:S {id: 's'})-[r:R*1..3 WHERE r.cost <= 1]->(b:T {id: 'd'}) RETURN [n IN nodes(p) | n.id]",
    ] {
        let result = session
            .execute(query)
            .expect("intrinsic edge predicate must execute");
        assert_eq!(result.rows().len(), 1);
        assert_eq!(
            string_list(&result.rows()[0][0]),
            vec!["s", "a", "d"],
            "forbidden shortcut/interior edges must be removed before shortest selection"
        );
    }
}

#[test]
fn uncertified_intrinsic_predicate_is_structured_unsupported_but_all_is_accepted() {
    let db = weighted_path_graph();
    let session = db.session();
    for predicate in ["[x IN [r.cost] | x] = [1]", "coalesce(rand(), 0) <= 1"] {
        let pruned = session.execute(&format!(
        "MATCH p = ANY SHORTEST (a:S {{id: 's'}})-[r:R*1..3 WHERE {predicate}]->(b:T {{id: 'd'}}) RETURN p"
    ));
        assert!(matches!(
            pruned,
            Err(grafeo_common::utils::error::Error::Query(error))
                if error.kind == grafeo_common::utils::error::QueryErrorKind::Unsupported
        ));

        let ordinary = session.execute(&format!(
        "MATCH p = (a:S {{id: 's'}})-[r:R*1..3 WHERE {predicate}]->(b:T {{id: 'd'}}) RETURN length(p)"
    ));
        let ordinary = ordinary.expect("ordinary ALL traversal must retain its evaluator");
        assert_eq!(
            ordinary.rows().len(),
            if predicate.contains("rand") { 3 } else { 1 }
        );
    }
}

#[test]
fn correlated_shortest_predicate_matches_separate_match_binding() {
    let db = weighted_path_graph();
    let session = db.session();
    let queries = [
        "MATCH (a:S), (b:T), p = ANY SHORTEST (a)-[r:R*1..3 WHERE r.cost <= b.budget]->(b) RETURN length(p)",
        "MATCH (a:S), (b:T) MATCH p = ANY SHORTEST (a)-[r:R*1..3 WHERE r.cost <= b.budget]->(b) RETURN length(p)",
    ];

    let mut lengths = Vec::new();
    for query in queries {
        let result = session
            .execute(query)
            .expect("correlated shortest must execute");
        assert_eq!(result.rows().len(), 1);
        lengths.push(result.rows()[0][0].clone());
    }
    assert_eq!(lengths, vec![Value::from(2i64), Value::from(2i64)]);
}

#[test]
fn path_edge_comprehensions_preserve_nested_and_outer_provenance() {
    let db = weighted_path_graph();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:S {id: 's'})-[r:R*1..3 WHERE r.cost <= 1]->(b:T {id: 'd'}) \
             RETURN [e IN edges(p) | coalesce(e.cost, 0)] AS costs, \
                    [e IN edges(p) | [x IN [1] | e.cost + x]] AS nested_costs, \
                    all(e IN edges(p) WHERE coalesce(e.cost, 0) > 0) AS positive, \
                    coalesce(edges(p), []) AS coalesced_edges, \
                    [e IN edges(p)[0..1] | e.cost] AS sliced_costs, \
                    [a IN [1] | a + 1] AS shadow, a.adjustment AS adjustment",
        )
        .expect("nested path edge expressions must execute");

    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    assert_eq!(int_list(&row[0]), vec![1, 1]);
    assert_eq!(nested_int_list(&row[1]), vec![vec![2], vec![2]]);
    assert_eq!(row[2], Value::Bool(true));
    assert_eq!(
        list_len(&row[3]),
        2,
        "coalesce must preserve raw edge values"
    );
    assert_eq!(
        int_list(&row[4]),
        vec![1],
        "slice must retain edge provenance"
    );
    assert_eq!(
        int_list(&row[5]),
        vec![2],
        "inner scalar must shadow only its own scope"
    );
    assert_eq!(
        row[6],
        Value::from(7i64),
        "outer node binding must be restored"
    );
}

#[test]
fn path_edge_comprehensions_keep_scopes_nulls_and_evaluated_aliases() {
    let db = weighted_path_graph();
    let session = db.session();
    let result = session.execute("MATCH p = (a:S {id: 's'})-[r:R*1..3 WHERE r.cost <= 1]->(b:T) RETURN [e IN coalesce(edges(p), []) | e.cost], [e IN edges(p) | e.cost + a.adjustment], [e IN edges(p) | [[e IN [1] | e + 1], e.cost]], [e IN edges(p) | e.absent], [e IN [0, 1] | coalesce(e.cost, 99)], [e IN edges(p) | reduce(total = 0, x IN [1, 2] | coalesce(total, 0) + e.cost + x)]").expect("lexical scopes and source provenance");
    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    assert_eq!(int_list(&row[0]), vec![1, 1]);
    assert_eq!(int_list(&row[1]), vec![8, 8]);
    let scalar_list = Value::List(vec![Value::from(2i64)].into());
    let per_edge = Value::List(vec![scalar_list, Value::from(1i64)].into());
    assert_eq!(row[2], Value::List(vec![per_edge.clone(), per_edge].into()));
    assert_eq!(row[3], Value::List(vec![Value::Null, Value::Null].into()));
    assert_eq!(
        int_list(&row[4]),
        vec![99, 99],
        "ordinary integers have no edge provenance"
    );

    assert_eq!(
        int_list(&row[5]),
        vec![5, 5],
        "reduce shares accumulator, item and outer edge scope"
    );

    let mixed = session.execute("MATCH p = (a:S {id: 's'})-[r:R*1..3 WHERE r.cost <= 1]->(b:T) UNWIND [true, false, true] AS choose WITH choose, CASE WHEN choose THEN edges(p) ELSE [0, 1] END AS chosen RETURN choose, [e IN chosen | coalesce(e.cost, 99)]").expect("evaluated WITH alias provenance");
    assert_eq!(mixed.rows().len(), 3);
    for (row, choose) in mixed.rows().iter().zip([true, false, true]) {
        assert_eq!(row[0], Value::Bool(choose));
        assert_eq!(
            int_list(&row[1]),
            if choose { vec![1, 1] } else { vec![99, 99] }
        );
    }
}

#[cfg(feature = "cypher")]
#[test]
fn legacy_shortest_match_where_selects_eligible_longer_paths() {
    let db = legacy_prefilter_graph();
    let session = db.session();
    for (function, expected_count) in [("shortestPath", 1), ("allShortestPaths", 2)] {
        for bound in ["*1..3", "*"] {
            let query = format!(
                "MATCH p = {function}((a:PrefilterStart)-[:PREFILTER{bound}]->(b:PrefilterEnd)) \
                 WHERE length(p) = 2 AND all(e IN relationships(p) WHERE e.ok) \
                 RETURN length(p), [e IN relationships(p) | e.cost]"
            );
            let result = session
                .execute_cypher(&query)
                .unwrap_or_else(|error| panic!("{query}: {error}"));
            assert_eq!(result.rows().len(), expected_count, "{query}");
            for row in result.rows() {
                assert_eq!(row[0], Value::Int64(2));
                assert_eq!(int_list(&row[1]), vec![1, 1]);
            }
        }
    }
}

#[cfg(feature = "cypher")]
#[test]
fn legacy_shortest_prefilter_substitutes_parameters_in_nested_predicates() {
    let db = legacy_prefilter_graph();
    let query = "MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)) \
                 WHERE length(p) = $hops AND all(e IN relationships(p) WHERE e.cost <= $budget) \
                 RETURN length(p)";
    let params = std::collections::HashMap::from([
        ("hops".to_string(), Value::Int64(2)),
        ("budget".to_string(), Value::Int64(1)),
    ]);
    let result = db.execute_cypher_with_params(query, params).unwrap();
    assert_eq!(result.rows(), &[vec![Value::Int64(2)]]);
    let second = std::collections::HashMap::from([
        ("hops".to_string(), Value::Int64(1)),
        ("budget".to_string(), Value::Int64(9)),
    ]);
    assert_eq!(
        db.execute_cypher_with_params(query, second).unwrap().rows(),
        &[vec![Value::Int64(1)]]
    );
    let missing = std::collections::HashMap::from([("hops".to_string(), Value::Int64(2))]);
    assert!(
        db.execute_cypher_with_params(query, missing)
            .unwrap_err()
            .to_string()
            .contains("Missing parameter: $budget")
    );
}

#[test]
fn shortest_intrinsic_predicate_substitutes_parameters() {
    let db = legacy_prefilter_graph();
    let params = std::collections::HashMap::from([("budget".to_string(), Value::Int64(1))]);
    let result = db.execute_with_params(
        "MATCH p = ANY SHORTEST (a:PrefilterStart)-[e:PREFILTER*1..3 WHERE e.cost <= $budget]->(b:PrefilterEnd) RETURN length(p)",
        params,
    ).unwrap();
    assert_eq!(result.rows(), &[vec![Value::Int64(2)]]);
}

#[test]
fn legacy_shortest_with_where_and_modern_where_remain_postfilters() {
    let db = legacy_prefilter_graph();
    let session = db.session();
    #[cfg(feature = "cypher")]
    let legacy = session
        .execute_cypher(
            "MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)) \
         WITH p WHERE length(p) = 2 RETURN length(p)",
        )
        .expect("WITH separates postfilter from shortest selection");
    #[cfg(feature = "cypher")]
    assert!(legacy.rows().is_empty());
    #[cfg(feature = "cypher")]
    let retained = session
        .execute_cypher(
            "MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)) \
         WITH p RETURN length(p)",
        )
        .expect("first-class path length after WITH");
    #[cfg(feature = "cypher")]
    assert_eq!(retained.rows(), &[vec![Value::Int64(1)]]);
    let modern = session
        .execute(
            "MATCH p = ANY SHORTEST (a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd) \
         WHERE length(p) = 2 RETURN length(p)",
        )
        .expect("modern selector WHERE remains postfilter");
    assert!(modern.rows().is_empty());
}

#[cfg(feature = "cypher")]
#[test]
fn legacy_shortest_prefilter_retains_outer_bindings_and_nested_call_scope() {
    let db = legacy_prefilter_graph();
    let session = db.session();
    for query in [
        "WITH 2 AS hopBudget MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)) WHERE all(e IN relationships(p) WHERE e.ok AND length(p) = hopBudget) RETURN length(p)",
        "MATCH (a:PrefilterStart), (b:PrefilterEnd) MATCH p = shortestPath((a)-[:PREFILTER*1..3]->(b)) WHERE length(p) = a.budget AND all(e IN relationships(p) WHERE e.ok) RETURN length(p)",
        "CALL { MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)) WHERE length(p) = 2 AND all(e IN relationships(p) WHERE e.ok) RETURN length(p) AS n } RETURN n",
    ] {
        let result = session
            .execute_cypher(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        assert_eq!(result.rows(), &[vec![Value::Int64(2)]], "{query}");
    }
}

#[cfg(feature = "cypher")]
#[test]
fn legacy_shortest_prefilter_handles_independent_patterns_in_one_match() {
    let db = legacy_prefilter_graph();
    let session = db.session();
    session.execute("CREATE (:PrefilterSibling)").unwrap();
    let sibling = session
        .execute_cypher(
            "MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)), \
         (x:PrefilterSibling) WHERE length(p) = 2 RETURN length(p)",
        )
        .expect("independent sibling must not move WHERE after quota");
    assert_eq!(sibling.rows(), &[vec![Value::Int64(2)]]);
    for (function, count) in [("shortestPath", 1), ("allShortestPaths", 4)] {
        let query = format!(
            "MATCH p = {function}((a:PrefilterStart)-[:PREFILTER*1..3]->(b:PrefilterEnd)), \
             q = {function}((c:PrefilterStart)-[:PREFILTER*1..3]->(d:PrefilterEnd)) \
             WHERE length(p) = 2 AND length(q) = 2 RETURN length(p), length(q)"
        );
        let result = session
            .execute_cypher(&query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        assert_eq!(result.rows().len(), count, "{query}");
        for row in result.rows() {
            assert_eq!(row, &[Value::Int64(2), Value::Int64(2)]);
        }
    }
}

#[cfg(feature = "cypher")]
#[test]
fn legacy_shortest_prefilter_rejects_repeated_relationships() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (a:PrefilterStart), (b:PrefilterEnd)")
        .unwrap();
    session.execute("MATCH (a:PrefilterStart), (b:PrefilterEnd) CREATE (a)-[:PREFILTER]->(b), (b)-[:PREFILTER]->(a)").unwrap();
    for bound in ["*1..3", "*"] {
        let result = session
            .execute_cypher(&format!(
                "MATCH p = shortestPath((a:PrefilterStart)-[:PREFILTER{bound}]->(b:PrefilterEnd)) \
             WHERE length(p) = 3 RETURN length(p)"
            ))
            .expect("legacy shortest uses finite trails");
        assert!(
            result.rows().is_empty(),
            "a length3 result would reuse the outgoing relationship"
        );
    }
}

fn legacy_prefilter_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (a:PrefilterStart {budget: 2}), (x:PrefilterMiddle), (y:PrefilterMiddle), (b:PrefilterEnd)").unwrap();
    session.execute("MATCH (a:PrefilterStart), (b:PrefilterEnd) CREATE (a)-[:PREFILTER {ok: false, cost: 9}]->(b)").unwrap();
    session.execute("MATCH (a:PrefilterStart), (m:PrefilterMiddle), (b:PrefilterEnd) CREATE (a)-[:PREFILTER {ok: true, cost: 1}]->(m), (m)-[:PREFILTER {ok: true, cost: 1}]->(b)").unwrap();
    db
}

#[test]
fn path_list_wrappers_retain_actual_edge_properties() {
    let db = scalar_path_alias_graph();
    let result = db
        .session()
        .execute(
            "MATCH p = (a:AliasStart)-[:ALIAS*2..2]->(b:AliasEnd) \
         RETURN [e IN tail(edges(p)) | e.cost], \
                [e IN reverse(edges(p)) | e.cost], \
                [e IN [head(edges(p))] | e.cost], \
                [e IN [last(edges(p))] | e.cost], \
                [e IN reverse([0, 1]) | coalesce(e.cost, 99)]",
        )
        .expect("list wrapper property callers");
    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    assert_eq!(int_list(&row[0]), vec![22]);
    assert_eq!(int_list(&row[1]), vec![22, 11]);
    assert_eq!(int_list(&row[2]), vec![11]);
    assert_eq!(int_list(&row[3]), vec![22]);
    assert_eq!(int_list(&row[4]), vec![99, 99]);
}

#[test]
fn scalar_path_aliases_distinguish_colliding_node_and_edge_ids() {
    let db = scalar_path_alias_graph();
    let session = db.session();
    let collision = session
        .execute("MATCH (a:AliasStart)-[e:ALIAS]->(b) RETURN id(a), id(e)")
        .expect("collision precondition");
    assert_eq!(collision.rows().len(), 1);
    assert_eq!(collision.rows()[0][0], collision.rows()[0][1]);
    for expression in [
        "head(edges(p))",
        "coalesce(head(edges(p)), a)",
        "edges(p)[0]",
    ] {
        let result = session
            .execute(&format!(
                "MATCH p = (a:AliasStart)-[:ALIAS*2..2]->(b:AliasEnd) \
             WITH {expression} AS e WITH e AS alias RETURN alias.cost, type(alias), id(alias)"
            ))
            .unwrap_or_else(|error| panic!("{expression}: {error}"));
        assert_eq!(result.rows().len(), 1, "{expression}");
        assert_eq!(result.rows()[0][0], Value::from(11i64), "{expression}");
        assert_eq!(result.rows()[0][1].as_str(), Some("ALIAS"), "{expression}");
        assert_eq!(result.rows()[0][2], collision.rows()[0][1], "{expression}");
    }
}

#[test]
fn scalar_path_aliases_preserve_mixed_types_through_modifiers() {
    let db = scalar_path_alias_graph();
    let session = db.session();
    for (distinct, modifier, expected) in [
        ("", "", vec![(3, 11), (1, 101), (2, 11)]),
        ("", "ORDER BY ordinal", vec![(1, 101), (2, 11), (3, 11)]),
        (
            "DISTINCT ",
            "ORDER BY ordinal",
            vec![(1, 101), (2, 11), (3, 11)],
        ),
        ("", "ORDER BY ordinal LIMIT 2", vec![(1, 101), (2, 11)]),
        (
            "DISTINCT ",
            "ORDER BY ordinal SKIP 1 LIMIT 2",
            vec![(2, 11), (3, 11)],
        ),
    ] {
        let query = format!(
            "MATCH p = (a:AliasStart)-[:ALIAS*2..2]->(b:AliasEnd) \
             UNWIND [3, 1, 2] AS ordinal \
             WITH {distinct}ordinal, CASE WHEN ordinal = 1 THEN a ELSE head(edges(p)) END AS entity \
             {modifier} RETURN ordinal, entity.cost"
        );
        let result = if modifier.is_empty() {
            session.execute(&query)
        } else {
            #[cfg(feature = "cypher")]
            {
                session.execute_cypher(&query)
            }
            #[cfg(not(feature = "cypher"))]
            {
                continue;
            }
        }
        .unwrap_or_else(|error| panic!("{query}: {error}"));
        let actual: Vec<_> = result
            .rows()
            .iter()
            .map(|row| {
                (
                    row[0].as_int64().expect("ordinal"),
                    row[1].as_int64().expect("entity cost"),
                )
            })
            .collect();
        assert_eq!(actual, expected, "{query}");
    }
}

fn scalar_path_alias_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE (a:AliasStart {cost: 101}), (b:AliasMiddle {cost: 202}), (c:AliasEnd {cost: 303})").expect("alias collision nodes");
    session
        .execute("MATCH (a:AliasStart), (b:AliasMiddle) CREATE (a)-[:ALIAS {cost: 11}]->(b)")
        .expect("first alias edge");
    session
        .execute("MATCH (b:AliasMiddle), (c:AliasEnd) CREATE (b)-[:ALIAS {cost: 22}]->(c)")
        .expect("last alias edge");
    db
}

#[test]
fn nested_reduce_uses_one_binding_for_dot_and_bracket_properties() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .session()
        .execute("RETURN reduce(x = {v: 1}, x IN [{v: 2}] | [x.v, x['v']])")
        .expect("consistent reducer lexical scope");
    assert_eq!(result.rows().len(), 1);
    // Accumulator precedence already governs bare variable access; dot access
    // must resolve the same binding rather than switching to the item by syntax.
    assert_eq!(int_list(&result.rows()[0][0]), vec![1, 1]);
}

#[test]
fn path_edge_comprehension_id_returns_real_edge_identity() {
    let db = weighted_path_graph();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:S {id: 's'})-[r:R*1..3 WHERE r.cost <= 1]->(b:T {id: 'd'}) \
             RETURN [e IN edges(p) | id(e)]",
        )
        .expect("id(e) over edges(p) must execute");
    let ids = int_list(&result.rows()[0][0]);
    assert_eq!(ids.len(), 2);
    for id in ids {
        assert!(
            db.get_edge(grafeo_common::types::EdgeId::new(
                u64::try_from(id).expect("edge identity must be nonnegative")
            ))
            .is_some(),
            "id(e) must remain a resolvable edge identity"
        );
    }
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_shortest_paths_cover_free_targets_and_parallel_edges() {
    fn assert_path(db: &GrafeoDB, target: &str, nodes: &Value, edges: &Value) -> Vec<i64> {
        let nodes = string_list(nodes);
        let Value::List(edges) = edges else {
            panic!("edges(p) must return a list: {edges:?}");
        };
        assert_eq!(nodes.first().map(String::as_str), Some("s"));
        assert_eq!(nodes.last().map(String::as_str), Some(target));
        let expected_length = usize::from(!matches!(target, "a" | "b")) + 1;
        assert_eq!(
            edges.len(),
            expected_length,
            "wrong shortest length for {target}"
        );
        assert_eq!(nodes.len(), edges.len() + 1);

        edges
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                let raw = edge.as_int64().expect("raw edge id");
                let id = u64::try_from(raw).expect("edge id must be nonnegative");
                let edge = db
                    .get_edge(grafeo_common::types::EdgeId::new(id))
                    .expect("raw edge id must resolve");
                assert_eq!(edge.edge_type.as_str(), "REL");
                let endpoint = |node| {
                    db.get_node(node)
                        .and_then(|node| node.get_property("id").cloned())
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .expect("path endpoint id")
                };
                assert_eq!(endpoint(edge.src), nodes[index]);
                assert_eq!(endpoint(edge.dst), nodes[index + 1]);
                raw
            })
            .collect()
    }

    let db = adversarial_path_graph();
    let session = db.session();

    let any = session
        .execute_cypher(
            "MATCH p = shortestPath((a:Node {id: 's'})-[:REL*1..2]->(b)) \
             RETURN b.id, [n IN nodes(p) | n.id], edges(p)",
        )
        .expect("Cypher shortestPath with a free endpoint must execute");
    assert_eq!(sorted_ids(&any), ["a", "b", "d", "g", "h"]);
    for row in any.rows() {
        assert_path(&db, row[0].as_str().expect("target id"), &row[1], &row[2]);
    }

    session
        .execute(
            "MATCH (s:Node {id: 's'}), (a:Node {id: 'a'}) \
             CREATE (s)-[:REL]->(a)",
        )
        .expect("parallel shortest edge");
    let all = session
        .execute_cypher(
            "MATCH p = allShortestPaths((a:Node {id: 's'})-[:REL*1..2]->(b)) \
             RETURN b.id, [n IN nodes(p) | n.id], edges(p)",
        )
        .expect("Cypher allShortestPaths with a free endpoint must execute");
    let mut paths =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<Vec<i64>>>::new();
    for row in all.rows() {
        let target = row[0].as_str().expect("target id");
        let edge_ids = assert_path(&db, target, &row[1], &row[2]);
        assert!(
            paths.entry(target.to_owned()).or_default().insert(edge_ids),
            "allShortestPaths duplicated an identical path to {target}"
        );
    }
    assert_eq!(
        paths
            .into_iter()
            .map(|(target, paths)| (target, paths.len()))
            .collect::<std::collections::BTreeMap<_, _>>(),
        std::collections::BTreeMap::from([
            ("a".to_owned(), 2),
            ("b".to_owned(), 1),
            ("d".to_owned(), 3),
            ("g".to_owned(), 1),
            ("h".to_owned(), 2),
        ]),
        "free-endpoint allShortestPaths must retain every parallel-edge path and no unreachable row"
    );
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_shortest_path_exposes_raw_edge_values() {
    let db = adversarial_path_graph();
    let session = db.session();
    let result = session
        .execute_cypher(
            "MATCH p = shortestPath((a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'})) \
             RETURN edges(p)",
        )
        .expect("Cypher shortestPath must execute");

    assert_eq!(result.rows().len(), 1);
    let Value::List(edges) = &result.rows()[0][0] else {
        panic!("edges(p) must return a list: {:?}", result.rows()[0][0]);
    };
    assert_eq!(edges.len(), 2, "the shortest path has two raw edges");
    assert!(edges.iter().all(|edge| matches!(edge, Value::Int64(_))));
    let resolved: Vec<_> = edges
        .iter()
        .map(|edge| {
            let id =
                u64::try_from(edge.as_int64().expect("raw edge ID")).expect("nonnegative edge ID");
            db.get_edge(grafeo_common::types::EdgeId::new(id))
                .expect("raw ID resolves to a real edge")
        })
        .collect();
    assert_eq!(resolved[0].dst, resolved[1].src);
    assert_eq!(
        db.get_node(resolved[0].src).unwrap().get_property("id"),
        Some(&Value::from("s"))
    );
    assert_eq!(
        db.get_node(resolved[1].dst).unwrap().get_property("id"),
        Some(&Value::from("d"))
    );
    assert!(resolved.iter().all(|edge| edge.edge_type.as_str() == "REL"));

    session
        .execute("MATCH (s:Node {id: 's'}), (a:Node {id: 'a'}) CREATE (s)-[:REL {_id: 999}]->(a)")
        .expect("parallel shortest edge");
    let parallel = session.execute_cypher("MATCH p = allShortestPaths((a:Node {id: 's'})-[:REL*1..3]->(b:Node {id: 'd'})) RETURN edges(p)").expect("parallel edge identity");
    assert_eq!(parallel.rows().len(), 3);
    let paths: std::collections::BTreeSet<Vec<i64>> = parallel
        .rows()
        .iter()
        .map(|row| {
            let Value::List(edges) = &row[0] else {
                panic!("path edges")
            };
            assert_eq!(edges.len(), 2);
            edges
                .iter()
                .map(|edge| {
                    let id = edge.as_int64().expect("raw edge");
                    assert!(
                        db.get_edge(grafeo_common::types::EdgeId::new(
                            u64::try_from(id).unwrap()
                        ))
                        .is_some()
                    );
                    id
                })
                .collect()
        })
        .collect();
    assert_eq!(
        paths.len(),
        3,
        "parallel shortest paths retain distinct real edge identities"
    );

    let weighted = weighted_path_graph();
    let session = weighted.session();
    for relationship in [":R*1..3 {cost: 1}", "r:R*1..3 WHERE r.cost <= 1"] {
        let result = session.execute_cypher(&format!("MATCH p = allShortestPaths((a:S)-[{relationship}]->(b:T)) RETURN [n IN nodes(p) | n.id]")).expect("Cypher intrinsic edge predicate");
        assert_eq!(result.rows().len(), 1);
        assert_eq!(string_list(&result.rows()[0][0]), vec!["s", "a", "d"]);
    }
}

#[test]
fn variable_length_distinct_returns_the_reachable_set() {
    let db = adversarial_path_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (a:Node {id: 's'})-[:REL*1..3]->(b) RETURN DISTINCT b.id")
        .expect("query must succeed");

    let expected: Vec<String> = reachable_within("s", 1, 3, "REL").into_iter().collect();
    assert_eq!(
        sorted_ids(&result),
        expected,
        "DISTINCT over a variable-length pattern is the reachable set"
    );
}

#[test]
fn untyped_variable_length_traverses_every_edge_type() {
    // The control for the typed assertions above: x hangs off s by an OTHER edge
    // and is unreachable over REL, so an untyped pattern must include it and a
    // typed one must not. Without this pair, a pattern that silently ignored the
    // edge type would satisfy the untyped expectation.
    let db = adversarial_path_graph();
    let session = db.session();

    let untyped = session
        .execute("MATCH (a:Node {id: 's'})-[*1..1]->(b) RETURN DISTINCT b.id")
        .expect("query must succeed");
    assert_eq!(
        sorted_ids(&untyped),
        vec!["a".to_string(), "b".to_string(), "x".to_string()],
        "an untyped one-hop pattern must include the OTHER-typed neighbour"
    );

    let typed = session
        .execute("MATCH (a:Node {id: 's'})-[:REL*1..1]->(b) RETURN DISTINCT b.id")
        .expect("query must succeed");
    assert_eq!(
        sorted_ids(&typed),
        vec!["a".to_string(), "b".to_string()],
        "a :REL-typed pattern must exclude the OTHER-typed neighbour"
    );
}

fn set(ids: &[&str]) -> std::collections::BTreeSet<String> {
    ids.iter().map(|id| (*id).to_string()).collect()
}

fn string_list(value: &Value) -> Vec<String> {
    let Value::List(values) = value else {
        panic!("expected string list, got {value:?}");
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("expected string item, got {value:?}"))
                .to_owned()
        })
        .collect()
}

fn weighted_path_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (s:S {id: 's', adjustment: 7}), (a:Node {id: 'a'}), (x:Node {id: 'x'}), \
                    (d:T {id: 'd', budget: 1})",
        )
        .expect("weighted path nodes");
    for query in [
        "MATCH (s:S {id: 's'}), (a:Node {id: 'a'}) CREATE (s)-[:R {cost: 1}]->(a)",
        "MATCH (a:Node {id: 'a'}), (d:T {id: 'd'}) CREATE (a)-[:R {cost: 1}]->(d)",
        "MATCH (s:S {id: 's'}), (x:Node {id: 'x'}) CREATE (s)-[:R {cost: 9}]->(x)",
        "MATCH (x:Node {id: 'x'}), (d:T {id: 'd'}) CREATE (x)-[:R {cost: 1}]->(d)",
        "MATCH (s:S {id: 's'}), (d:T {id: 'd'}) CREATE (s)-[:R {cost: 9}]->(d)",
    ] {
        session.execute(query).expect("weighted path edge");
    }
    db
}

fn int_list(value: &Value) -> Vec<i64> {
    let Value::List(values) = value else {
        panic!("expected integer list, got {value:?}");
    };
    values
        .iter()
        .map(|value| {
            value
                .as_int64()
                .unwrap_or_else(|| panic!("expected integer: {value:?}"))
        })
        .collect()
}

fn nested_int_list(value: &Value) -> Vec<Vec<i64>> {
    let Value::List(values) = value else {
        panic!("expected nested integer list, got {value:?}");
    };
    values.iter().map(int_list).collect()
}

fn list_len(value: &Value) -> usize {
    let Value::List(values) = value else {
        panic!("expected list, got {value:?}");
    };
    values.len()
}

#[test]
fn specialized_triangle_public_rows_and_count_preserve_parallel_edges() {
    let db = support::adversarial_triangle_graph();
    let session = db.session();
    for pattern in [
        "(a:Triangle)-[ab:TRI]->(b)-[bc:TRI]->(c)-[ca:TRI]->(a)",
        "(a:Triangle)-[ab:TRI]->(b)-[bc:TRI]->(c), (c)-[ca:TRI]->(a)",
    ] {
        let rows = session
            .execute(&format!(
                "MATCH {pattern} RETURN a.id, id(ab), id(bc), id(ca)"
            ))
            .unwrap();
        assert_eq!(rows.row_count(), 72, "{pattern}");
        let mut by_start = std::collections::BTreeMap::new();
        for row in rows.rows() {
            let start = row[0].as_str().unwrap().to_string();
            let edge_tuple = (
                row[1].as_int64().unwrap(),
                row[2].as_int64().unwrap(),
                row[3].as_int64().unwrap(),
            );
            assert!(
                by_start
                    .entry(start)
                    .or_insert_with(std::collections::BTreeSet::new)
                    .insert(edge_tuple),
                "duplicate edge tuple"
            );
        }
        assert_eq!(by_start.len(), 3);
        assert!(by_start.values().all(|paths| paths.len() == 24));
        for aggregate in ["count(*)", "count(a)", "count(ab)"] {
            let count = session
                .execute(&format!("MATCH {pattern} RETURN {aggregate}"))
                .unwrap();
            assert_eq!(
                count.rows(),
                &[vec![Value::Int64(72)]],
                "{pattern}: {aggregate}"
            );
        }
    }
}

#[cfg(feature = "compact-store")]
#[test]
fn specialized_triangle_public_compact_type_filter_is_not_elided() {
    let mut db = support::adversarial_triangle_graph();
    db.compact().unwrap();
    let rows = db.execute("MATCH (a:Triangle)-[ab:TRI]->(b)-[bc:TRI]->(c), (c)-[ca:TRI]->(a) RETURN a.id, id(ab), id(bc), id(ca)").unwrap();
    let direct = db
        .execute("MATCH (a:Triangle)-[e:TRI]->(b) RETURN a.id, b.id, id(e)")
        .unwrap();
    assert_eq!(rows.row_count(), 72);
    let edge_tuples: std::collections::BTreeSet<_> = rows
        .rows()
        .iter()
        .map(|row| {
            (
                row[1].as_int64().unwrap(),
                row[2].as_int64().unwrap(),
                row[3].as_int64().unwrap(),
            )
        })
        .collect();
    assert_eq!(edge_tuples.len(), 72);
    assert_eq!(direct.row_count(), 9);
    let edge_ids: std::collections::BTreeSet<_> = direct
        .rows()
        .iter()
        .map(|row| row[2].as_int64().unwrap())
        .collect();
    assert_eq!(edge_ids.len(), 9);
    for (edge_type, expected) in [("TRI", 72), ("ABSENT", 0)] {
        let query = format!(
            "MATCH (a)-[:{edge_type}]->(b)-[:{edge_type}]->(c), (c)-[:{edge_type}]->(a) RETURN count(*)"
        );
        assert_eq!(
            db.execute(&query).unwrap().rows(),
            &[vec![Value::Int64(expected)]],
            "{query}"
        );
    }
}

#[test]
fn specialized_triangle_public_snapshot_and_own_writes() {
    let db = support::adversarial_triangle_graph();
    let query = "MATCH (a:Triangle)-[:TRI]->(b)-[:TRI]->(c), (c)-[:TRI]->(a) RETURN count(*)";
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(
        reader.execute(query).unwrap().rows(),
        &[vec![Value::Int64(72)]]
    );
    db.execute("MATCH (a:Triangle {id:'ta'}), (b:Triangle {id:'tb'}) CREATE (a)-[:TRI]->(b)")
        .unwrap();
    assert_eq!(
        reader.execute(query).unwrap().rows(),
        &[vec![Value::Int64(72)]],
        "snapshot excludes committed later edge"
    );
    reader
        .execute("MATCH (a:Triangle {id:'ta'}), (b:Triangle {id:'tb'}) CREATE (a)-[:TRI]->(b)")
        .unwrap();
    assert_eq!(
        reader.execute(query).unwrap().rows(),
        &[vec![Value::Int64(108)]],
        "own extra edge is visible at fixed snapshot"
    );
    reader
        .execute("MATCH (a:Triangle {id:'tb'})-[e:TRI]->(b:Triangle {id:'tc'}) DELETE e")
        .unwrap();
    assert_eq!(
        reader.execute(query).unwrap().rows(),
        &[vec![Value::Int64(0)]],
        "own deletes remove closing walks"
    );
    reader.rollback().unwrap();
    assert_eq!(
        db.execute(query).unwrap().rows(),
        &[vec![Value::Int64(108)]]
    );
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql",
    feature = "cypher"
))]
#[test]
fn cached_sort_spill_preserves_free_endpoint_path_edge_provenance() {
    use grafeo_common::types::EdgeId;
    use grafeo_common::utils::error::ErrorCode;
    use std::collections::{BTreeMap, BTreeSet};

    let fixture = adversarial_path_graph();
    let expected_result = fixture
        .session()
        .execute("MATCH (a:Node {id: 's'})-[e]->(b) RETURN b.id, type(e), id(e)")
        .unwrap();
    let expected: BTreeSet<_> = expected_result
        .rows()
        .iter()
        .map(|row| {
            (
                row[0].as_str().unwrap().to_owned(),
                row[1].as_str().unwrap().to_owned(),
                row[2].as_int64().unwrap(),
            )
        })
        .collect();
    assert_eq!(expected.len(), 4);
    let mut endpoints = BTreeMap::new();
    for (endpoint, _, _) in &expected {
        *endpoints.entry(endpoint.as_str()).or_insert(0) += 1;
    }
    assert_eq!(endpoints, BTreeMap::from([("a", 1), ("b", 1), ("x", 2)]));
    let snapshot = fixture.export_snapshot().unwrap();
    // The arithmetic alias prevents TopK; edges(p) must remain typed through
    // sorting and limiting before UNWIND resolves type(e) and id(e).
    // Keep all 16,384 inputs without exceeding the parser's statement-link
    // safety limit. This remains a repeated Session query with no parameters.
    let query = "MATCH p = (a:Node {id: 's'})-[*1..1]->(b) \
                 UNWIND range(0, 16383) AS i \
                 WITH 16383 - i AS ordinal, edges(p) AS path_edges, b.id AS endpoint \
                 ORDER BY ordinal, endpoint LIMIT 256 \
                 UNWIND path_edges AS e RETURN ordinal, endpoint, type(e), id(e)";
    let short_control = "MATCH p = (a:Node {id: 's'})-[*1..1]->(b) \
                         UNWIND [0, 1] AS i RETURN b.id, i, edges(p)";
    let source_session = fixture.session();
    assert_eq!(
        source_session
            .execute_cypher(short_control)
            .unwrap()
            .row_count(),
        8
    );
    assert_eq!(
        source_session.execute_cypher(query).unwrap().row_count(),
        256,
        "full source query before restore"
    );
    for quota in [0, 64 << 20] {
        let directory = tempfile::tempdir().unwrap();
        let database = GrafeoDB::with_config(
            grafeo_engine::Config::in_memory()
                .with_memory_limit(2 << 20)
                .with_spill_path(directory.path())
                .with_max_query_spill_bytes(quota),
        )
        .unwrap();
        database.restore_snapshot(&snapshot).unwrap();
        let session = database.session();
        assert_eq!(
            session.execute_cypher(short_control).unwrap().row_count(),
            8,
            "restored free-endpoint repetition control"
        );
        for _ in 0..if quota == 0 { 1 } else { 2 } {
            let outcome = session.execute_cypher(query);
            if quota == 0 {
                let error = match outcome {
                    Err(error) => error,
                    Ok(result) => panic!(
                        "path sort bypassed disk admission: {} rows",
                        result.row_count()
                    ),
                };
                assert_eq!(
                    error.error_code(),
                    ErrorCode::StorageFull,
                    "{error:?}; {error}"
                );
                assert!(
                    error.to_string().contains("spill disk quota exceeded"),
                    "{error}"
                );
            } else {
                let result = outcome.unwrap();
                assert_eq!(result.row_count(), 256);
                let mut groups: BTreeMap<i64, BTreeSet<(String, String, i64)>> = BTreeMap::new();
                for (row_index, row) in result.rows().iter().enumerate() {
                    let ordinal = row[0].as_int64().unwrap();
                    let endpoint = row[1].as_str().unwrap().to_owned();
                    assert_eq!(ordinal, i64::try_from(row_index / 4).unwrap());
                    assert_eq!(endpoint, ["a", "b", "x", "x"][row_index % 4]);
                    let edge_type = row[2].as_str().unwrap().to_owned();
                    let edge_id = row[3].as_int64().unwrap();
                    let edge = database
                        .get_edge(EdgeId::new(u64::try_from(edge_id).unwrap()))
                        .unwrap();
                    assert_eq!(edge.edge_type.as_str(), edge_type);
                    assert_eq!(
                        database.get_node(edge.src).unwrap().get_property("id"),
                        Some(&Value::from("s"))
                    );
                    assert_eq!(
                        database.get_node(edge.dst).unwrap().get_property("id"),
                        Some(&Value::from(endpoint.as_str()))
                    );
                    assert!(
                        groups
                            .entry(ordinal)
                            .or_default()
                            .insert((endpoint, edge_type, edge_id)),
                        "duplicate edge within ordinal"
                    );
                }
                assert_eq!(
                    groups.keys().copied().collect::<Vec<_>>(),
                    (0..64).collect::<Vec<_>>()
                );
                for actual in groups.values() {
                    assert_eq!(
                        actual, &expected,
                        "both parallel OTHER edge identities must survive"
                    );
                }
            }
            let namespace = directory
                .path()
                .join(format!("grafeo-store-{}", database.store_id()));
            let entries: Vec<_> = std::fs::read_dir(namespace)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            assert_eq!(
                entries,
                vec![std::ffi::OsString::from(".grafeo-spill-root")]
            );
        }
    }
}
