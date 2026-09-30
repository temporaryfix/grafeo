//! Plan contracts on the benchmark workload.
//!
//! Earlier silent de-optimizations — the index dropped for writers,
//! the correlated lookup dropped for a whole statement, no fast path for
//! `id()` — passed every result-equality test, because only rows were
//! asserted. These contracts assert the plan instead.
//!
//! Plans are observed through [`plan_trace`], which records the operators
//! `Planner::plan` builds on the real `Session` path without wrapping them or
//! disabling any profiling-gated rewrite. `PROFILE` is not used: it changes
//! the plan. Each contract is an exact match on the operator names and the
//! access paths, so a contract cannot pass without reaching the node it
//! names.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_core::graph::WorkSnapshot;

use super::{AccessPath, Decline, PlanTrace, Planner, plan_trace};
use crate::GrafeoDB;
use crate::session::Session;

/// The plan one query shape must build.
struct Contract {
    query: &'static str,
    params: fn(&Fixture) -> Vec<(&'static str, Value)>,
    operators: &'static [&'static str],
    access_paths: &'static [AccessPath],
    /// Optimizations the planner must decline, and why; empty for most.
    declines: &'static [Decline],
}

/// Internal ids the parameterised shapes need.
struct Fixture {
    node_id: i64,
    vector_node_id: i64,
}

/// The benchmark's data shape at fixture size: `Node`, `Person` and
/// `VectorNode` rows, a `T` edge, and the benchmark's only index, on `id`.
fn benchmark_db(indexed: bool) -> (GrafeoDB, Fixture) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (:Node {id: 'n_0'}), (:Node {id: 'n_1'}), \
             (:Person {id: 'p_0', version: 0}), (:Person {id: 'p_1', version: 1}), \
             (:VectorNode {id: 'v_0'})",
        )
        .expect("fixture nodes");
    session
        .execute("MATCH (a:Node {id: 'n_0'}), (b:Node {id: 'n_1'}) CREATE (a)-[:T]->(b)")
        .expect("fixture edge");
    if indexed {
        session
            .execute("CREATE INDEX gb_id FOR (n:Node) ON (n.id)")
            .expect("fixture index");
    }
    let id_of = |query: &str| match session.execute(query).expect("fixture id").rows()[0][0] {
        Value::Int64(id) => id,
        ref other => panic!("id(n) must be Int64, got {other:?}"),
    };
    let fixture = Fixture {
        node_id: id_of("MATCH (n:Node {id: 'n_1'}) RETURN id(n)"),
        vector_node_id: id_of("MATCH (n:VectorNode {id: 'v_0'}) RETURN id(n)"),
    };
    drop(session);
    (db, fixture)
}

/// Executes `query` and returns the one plan it built.
///
/// Fails if the query built no plan (a physical-plan cache hit would
/// otherwise pass vacuously) or more than one.
fn traced(session: &Session, query: &str, params: Vec<(&str, Value)>) -> PlanTrace {
    let params: HashMap<String, Value> = params
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect();
    let (result, mut plans) = plan_trace::capture(|| session.execute_with_params(query, params));
    result.unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_eq!(
        plans.len(),
        1,
        "{query} must build exactly one plan: {plans:?}"
    );
    let trace = plans.remove(0);
    // Every lookup that builds a `NodeList` records how it found its nodes.
    // A new lookup that forgot to would otherwise look like any other.
    let node_lists = trace
        .operators
        .iter()
        .filter(|name| *name == "NodeList")
        .count();
    let node_list_paths = trace
        .access_paths
        .iter()
        .filter(|path| {
            matches!(
                path,
                AccessPath::InternalId | AccessPath::PropertyIndex | AccessPath::LabelFirst
            )
        })
        .count();
    assert_eq!(
        node_lists, node_list_paths,
        "{query}: each NodeList needs one recorded access path: {trace:?}"
    );
    trace
}

fn satisfies(trace: &PlanTrace, contract: &Contract) -> bool {
    trace.operators == contract.operators
        && trace.access_paths == contract.access_paths
        && trace.declines == contract.declines
}

fn mismatch(contract: &Contract, trace: &PlanTrace) -> String {
    format!(
        "{}\n  expected {:?} {:?} {:?}\n  built    {:?} {:?} {:?}",
        contract.query,
        contract.operators,
        contract.access_paths,
        contract.declines,
        trace.operators,
        trace.access_paths,
        trace.declines
    )
}

/// Executes `query` like [`traced`] and also returns the store work it did.
fn traced_with_work(
    db: &GrafeoDB,
    session: &Session,
    query: &str,
    params: Vec<(&str, Value)>,
) -> (PlanTrace, WorkSnapshot) {
    let before = crate::database::testing::root_lpg_store(db).work_snapshot();
    let trace = traced(session, query, params);
    (
        trace,
        crate::database::testing::root_lpg_store(db)
            .work_snapshot()
            .since(before),
    )
}

/// A lookup may probe an index but may not scan, not even an empty label.
fn scanned(work: WorkSnapshot) -> bool {
    work.scanned_any() || work.label_scan_calls > 0 || work.full_scan_calls > 0
}

/// Whether `contract` promises an index, which the store must then have read.
fn uses_index(contract: &Contract) -> bool {
    contract.access_paths.iter().any(|path| {
        matches!(
            path,
            AccessPath::PropertyIndex | AccessPath::CorrelatedIndex
        )
    })
}

/// Contracts from `contracts` that `session` breaks: by plan, by scanning, or
/// by promising an index the store never read.
fn broken_lookups(
    db: &GrafeoDB,
    session: &Session,
    contracts: &[Contract],
    fixture: &Fixture,
) -> Vec<String> {
    contracts
        .iter()
        .filter_map(|contract| {
            let (trace, work) =
                traced_with_work(db, session, contract.query, (contract.params)(fixture));
            if !satisfies(&trace, contract) {
                Some(mismatch(contract, &trace))
            } else if scanned(work) {
                Some(format!("{}\n  scanned {work:?}", contract.query))
            } else if uses_index(contract) && work.property_index_posting_ids == 0 {
                Some(format!(
                    "{}\n  read no index postings {work:?}",
                    contract.query
                ))
            } else {
                None
            }
        })
        .collect()
}

const POINT_LOOKUP: &[&str] = &["NodeScan", "NodeList", "Project"];
const POINT_SET: &[&str] = &["NodeScan", "NodeList", "SetProperty", "Project"];

fn no_params(_: &Fixture) -> Vec<(&'static str, Value)> {
    Vec::new()
}

fn node_app_id(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![("id", Value::from("n_1"))]
}

fn person_app_id(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![("id", Value::from("p_1"))]
}

const LABELLED_POINT_LOOKUP: Contract = Contract {
    query: "MATCH (p:Person {id: $id}) RETURN p.version AS version",
    params: person_app_id,
    operators: POINT_LOOKUP,
    access_paths: &[AccessPath::PropertyIndex],
    declines: &[],
};

// Positive control: the full id map is a scan by definition, so a trace that
// never recorded `Scan` could not satisfy it.
const FULL_ID_MAP: Contract = Contract {
    query: "MATCH (n) RETURN id(n) AS nid, n.id AS id",
    params: no_params,
    operators: &["Scan", "Project"],
    access_paths: &[],
    declines: &[],
};

/// Reader shapes: the adapter's point lookups, the internal-id mapping every
/// native algorithm result goes through, and the full id map.
const READER_CONTRACTS: &[Contract] = &[
    LABELLED_POINT_LOOKUP,
    Contract {
        query: "MATCH (n:Node {id: $id}) RETURN id(n) AS nid",
        params: node_app_id,
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n:Node {id: $id}) RETURN n",
        params: node_app_id,
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n {id: $id}) RETURN n",
        params: node_app_id,
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n:Node) WHERE n.id IN $ids RETURN id(n) AS nid",
        params: |_| {
            vec![(
                "ids",
                Value::List(vec![Value::from("n_0"), Value::from("n_1")].into()),
            )]
        },
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n:Node) WHERE id(n) IN $ids RETURN id(n) AS nid, n.id AS id",
        params: |fixture| {
            vec![(
                "ids",
                Value::List(vec![Value::Int64(fixture.node_id)].into()),
            )]
        },
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::InternalId],
        declines: &[],
    },
    Contract {
        query: "MATCH (n:VectorNode) WHERE id(n) IN $ids RETURN id(n) AS nid, n.id AS id",
        params: |fixture| {
            vec![(
                "ids",
                Value::List(vec![Value::Int64(fixture.vector_node_id)].into()),
            )]
        },
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::InternalId],
        declines: &[],
    },
    Contract {
        query: "MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS nid, n.id AS id",
        params: |fixture| {
            vec![(
                "ids",
                Value::List(vec![Value::Int64(fixture.node_id)].into()),
            )]
        },
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::InternalId],
        declines: &[],
    },
    FULL_ID_MAP,
];

#[test]
fn reader_shapes_build_their_contracted_plans() {
    let (db, fixture) = benchmark_db(true);
    let session = db.session();
    let failures: Vec<String> = READER_CONTRACTS
        .iter()
        .filter_map(|contract| {
            let trace = traced(&session, contract.query, (contract.params)(&fixture));
            (!satisfies(&trace, contract)).then(|| mismatch(contract, &trace))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "plan contracts broken:\n{}",
        failures.join("\n")
    );
}

#[test]
fn unindexed_point_lookup_is_rejected_by_the_indexed_contract() {
    // Control: without the index the labelled lookup still builds `NodeList`,
    // by checking every labelled node. Operator names alone would accept it.
    let (db, fixture) = benchmark_db(false);
    let session = db.session();
    let contract = &LABELLED_POINT_LOOKUP;
    let trace = traced(&session, contract.query, (contract.params)(&fixture));
    assert_eq!(trace.operators, POINT_LOOKUP);
    assert_eq!(trace.access_paths, [AccessPath::LabelFirst]);
    assert!(!satisfies(&trace, contract));
}

#[test]
fn cached_plans_are_not_mistaken_for_contracts() {
    // Control: a repeated parameterless read is served from the physical-plan
    // cache and builds nothing, which `traced` must refuse rather than pass.
    let (db, fixture) = benchmark_db(true);
    let session = db.session();
    let contract = &FULL_ID_MAP;
    traced(&session, contract.query, (contract.params)(&fixture));
    let ((), plans) = plan_trace::capture(|| {
        session.execute(contract.query).expect("cached read");
    });
    assert!(
        plans.is_empty(),
        "second execution must hit the cache: {plans:?}"
    );
}

#[test]
fn trace_names_the_production_plan_not_the_profiled_one() {
    // Parity control. Expand-chain fusion is disabled under profiling, so a
    // trace taken through `plan_profiled` would show separate expands. The
    // trace must match `plan_profiled` where no profiling gate applies and
    // differ where one does.
    use crate::query::binder::Binder;
    use crate::query::optimizer::Optimizer;
    use crate::query::plan::LogicalPlan;
    use crate::query::translators::gql;
    use grafeo_core::graph::GraphStoreSearch;

    let (db, _) = benchmark_db(true);
    let store: Arc<dyn GraphStoreSearch> =
        Arc::clone(crate::database::testing::root_lpg_store(&db)) as _;
    let plan_both = |query: &str| {
        let logical: LogicalPlan = gql::translate(query).expect("translate");
        Binder::new().bind(&logical).expect("bind");
        let logical = Optimizer::from_graph_store(store.as_ref())
            .optimize(logical)
            .expect("optimize");
        // A fresh planner for each call: planning mutates per-planner state.
        let planner = || Planner::new(Arc::clone(&store)).with_factorized_execution(true);
        let (_, trace) = planner().plan_traced(&logical).expect("traced");
        let (_, entries) = planner().plan_profiled(&logical).expect("profiled");
        let profiled: Vec<String> = entries.into_iter().map(|entry| entry.name).collect();
        (trace.operators, profiled)
    };

    let (traced_names, profiled_names) = plan_both("MATCH (n:Node {id: 'n_1'}) RETURN n");
    assert_eq!(traced_names, profiled_names);

    let (traced_names, profiled_names) =
        plan_both("MATCH (a:Node)-[:T]->(b)-[:T]->(c) RETURN c.id");
    assert_eq!(
        profiled_names
            .iter()
            .filter(|name| *name == "Expand")
            .count(),
        2,
        "profiled plan keeps both expands: {profiled_names:?}"
    );
    assert!(
        !traced_names.iter().any(|name| name == "Expand"),
        "production plan fuses the expand chain: {traced_names:?}"
    );
}

fn person_version_update(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![("id", Value::from("p_1")), ("v", Value::Int64(7))]
}

fn node_score_update(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![("id", Value::from("n_0")), ("s", Value::Int64(3))]
}

const PERSON_VERSION_SET: Contract = Contract {
    query: "MATCH (p:Person {id: $id}) SET p.version = $v",
    params: person_version_update,
    operators: POINT_SET,
    access_paths: &[AccessPath::PropertyIndex],
    declines: &[],
};

/// Shapes the benchmark's writer transactions run. Each must keep the index
/// inside a transaction, where the committed index alone cannot see the
/// transaction's own writes and the index was once dropped for that reason.
/// The writes target `p_1` and `n_0`; the reads target `n_1`.
const WRITER_CONTRACTS: &[Contract] = &[
    LABELLED_POINT_LOOKUP,
    PERSON_VERSION_SET,
    Contract {
        query: "MATCH (n:Node {id: $id}) RETURN n",
        params: node_app_id,
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n:Node {id: $id}) RETURN id(n) AS nid",
        params: node_app_id,
        operators: POINT_LOOKUP,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n {id: $id}) SET n.score = $s",
        params: node_score_update,
        operators: POINT_SET,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n {id: $id}) SET n.score = $s RETURN n",
        params: node_score_update,
        operators: POINT_SET,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
];

/// The one internal id `MATCH (n:Node {id: $id}) RETURN id(n)` finds, if any.
fn node_id_by_app_id(session: &Session, app_id: &str) -> Option<i64> {
    let result = session
        .execute_with_params(
            "MATCH (n:Node {id: $id}) RETURN id(n) AS nid",
            HashMap::from([("id".to_string(), Value::from(app_id))]),
        )
        .expect("id lookup");
    match result.rows() {
        [] => None,
        [row] => match row[0] {
            Value::Int64(id) => Some(id),
            ref other => panic!("id(n) must be Int64, got {other:?}"),
        },
        rows => panic!("{app_id} must name at most one node, got {rows:?}"),
    }
}

#[test]
fn writer_shapes_keep_their_index_inside_a_transaction() {
    let (db, fixture) = benchmark_db(true);
    let mut session = db.session();
    session.begin_transaction().expect("begin");

    let failures = broken_lookups(&db, &session, WRITER_CONTRACTS, &fixture);
    assert!(
        failures.is_empty(),
        "writer plan contracts broken:\n{}",
        failures.join("\n")
    );

    // The transaction's own index write: a renamed node must be found under
    // its new id through the index, before anything is committed.
    session
        .execute("MATCH (n:Node {id: 'n_1'}) SET n.id = 'n_1_renamed'")
        .expect("rename");
    // The index answers from the transaction's buffered write, so no
    // committed posting is read; the rows below prove the node was found.
    let (trace, work) = traced_with_work(
        &db,
        &session,
        "MATCH (n:Node {id: $id}) RETURN id(n) AS nid",
        vec![("id", Value::from("n_1_renamed"))],
    );
    assert_eq!(trace.operators, POINT_LOOKUP);
    assert_eq!(trace.access_paths, [AccessPath::PropertyIndex]);
    assert!(!scanned(work), "own-write lookup scanned: {work:?}");
    assert_eq!(
        node_id_by_app_id(&session, "n_1_renamed"),
        Some(fixture.node_id)
    );
    assert_eq!(node_id_by_app_id(&session, "n_1"), None);

    // Positive control: the counters see a scan on this path, so the
    // no-scan half of each contract above is not vacuous.
    let (trace, work) = traced_with_work(
        &db,
        &session,
        FULL_ID_MAP.query,
        (FULL_ID_MAP.params)(&fixture),
    );
    assert!(
        satisfies(&trace, &FULL_ID_MAP),
        "{}",
        mismatch(&FULL_ID_MAP, &trace)
    );
    assert!(scanned(work), "a full scan must be counted: {work:?}");

    session.rollback().expect("rollback");
}

#[test]
fn snapshot_behind_a_concurrent_commit_keeps_its_index() {
    let (db, fixture) = benchmark_db(true);
    let mut session = db.session();
    session.begin_transaction().expect("begin");

    // A concurrent commit renames the node this transaction reads, so the
    // current index no longer holds the id the snapshot must still find.
    let concurrent = db.session();
    concurrent
        .execute("MATCH (n:Node {id: 'n_1'}) SET n.id = 'n_1_moved'")
        .expect("concurrent rename");
    concurrent
        .execute("CREATE (:Node {id: 'n_2'})")
        .expect("concurrent create");
    assert_eq!(node_id_by_app_id(&concurrent, "n_1"), None);
    assert_eq!(
        node_id_by_app_id(&session, "n_1"),
        Some(fixture.node_id),
        "the transaction's snapshot must be behind the concurrent commit"
    );
    assert_eq!(node_id_by_app_id(&session, "n_1_moved"), None);

    let failures = broken_lookups(&db, &session, WRITER_CONTRACTS, &fixture);
    assert!(
        failures.is_empty(),
        "snapshot-behind plan contracts broken:\n{}",
        failures.join("\n")
    );
    session.rollback().expect("rollback");
}

#[test]
fn unindexed_writer_lookup_fails_its_plan_and_work_contract() {
    // Control: without the index a writer's labelled lookup checks every
    // labelled node. Both halves of the contract must reject it.
    let (db, fixture) = benchmark_db(false);
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let contract = &PERSON_VERSION_SET;
    let (trace, work) =
        traced_with_work(&db, &session, contract.query, (contract.params)(&fixture));
    assert_eq!(trace.operators, POINT_SET);
    assert_eq!(trace.access_paths, [AccessPath::LabelFirst]);
    assert!(!satisfies(&trace, contract));
    assert!(
        scanned(work),
        "a label-first lookup must be counted: {work:?}"
    );
    session.rollback().expect("rollback");
}

fn map(pairs: &[(&str, Value)]) -> Value {
    Value::Map(
        pairs
            .iter()
            .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into(),
    )
}

/// The adapter's edge batch: endpoints by application id, one property
/// column `p0`, and a property map `p` for the `SET r = e.p` form.
fn edge_batch(_: &Fixture) -> Vec<(&'static str, Value)> {
    let edge = |s: &str, t: &str, w: i64| {
        map(&[
            ("s", Value::from(s)),
            ("t", Value::from(t)),
            ("p0", Value::Int64(w)),
            ("p", map(&[("w", Value::Int64(w))])),
        ])
    };
    vec![(
        "es",
        Value::List(vec![edge("n_0", "n_1", 1), edge("n_1", "n_0", 2)].into()),
    )]
}

/// The adapter's node batch; a key missing on some rows is passed as null.
fn node_batch(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![(
        "rows",
        Value::List(
            vec![
                map(&[("id", Value::from("x_0")), ("v", Value::Int64(0))]),
                map(&[("id", Value::from("x_1")), ("v", Value::Null)]),
            ]
            .into(),
        ),
    )]
}

const BATCH_INSERT: &[&str] = &["Empty", "Unwind", "CreateNode", "Project"];
const BOTH_ENDPOINTS_INDEXED: &[AccessPath] =
    &[AccessPath::CorrelatedIndex, AccessPath::CorrelatedIndex];
/// The candidate `Scan` under each correlated lookup is built in both modes
/// but only iterated without an index, so the operators cannot tell the modes
/// apart: the access paths and the work counters do.
const BATCH_EDGE: &[&str] = &[
    "Empty",
    "Unwind",
    "Scan",
    "Filter",
    "Scan",
    "Filter",
    "CreateEdge",
    "Project",
];

const BARE_EDGE_BATCH: Contract = Contract {
    query: "UNWIND $es AS e MATCH (s:Node {id: e.s}), (t:Node {id: e.t}) CREATE (s)-[:T]->(t)",
    params: edge_batch,
    operators: BATCH_EDGE,
    access_paths: BOTH_ENDPOINTS_INDEXED,
    declines: &[],
};

/// The adapter's batched loads and neighbour reads. Nodes load before the
/// index exists, so the insert rows hold with or without it.
const BATCH_CONTRACTS: &[Contract] = &[
    Contract {
        query: "UNWIND $rows AS r INSERT (:Node {id: r.id, v: r.v})",
        params: node_batch,
        operators: BATCH_INSERT,
        access_paths: &[],
        declines: &[],
    },
    Contract {
        query: "UNWIND $rows AS r INSERT (:Person:Node {id: r.id, v: r.v})",
        params: node_batch,
        operators: BATCH_INSERT,
        access_paths: &[],
        declines: &[],
    },
    BARE_EDGE_BATCH,
    Contract {
        query: "UNWIND $es AS e MATCH (s:Node {id: e.s}), (t:Node {id: e.t}) \
                CREATE (s)-[:T {w: e.p0}]->(t)",
        params: edge_batch,
        operators: BATCH_EDGE,
        access_paths: BOTH_ENDPOINTS_INDEXED,
        declines: &[],
    },
    // Trailing clauses once cost ~1.7 ms per edge by disabling the lookup
    // for the whole statement.
    Contract {
        query: "UNWIND $es AS e MATCH (s:Node {id: e.s}), (t:Node {id: e.t}) \
                CREATE (s)-[r:T]->(t) SET r = e.p",
        params: edge_batch,
        operators: &[
            "Empty",
            "Unwind",
            "Scan",
            "Filter",
            "Scan",
            "Filter",
            "CreateEdge",
            "SetProperty",
            "Project",
        ],
        access_paths: BOTH_ENDPOINTS_INDEXED,
        declines: &[],
    },
    Contract {
        query: "UNWIND $es AS e MATCH (s:Node {id: e.s}), (t:Node {id: e.t}) \
                CREATE (s)-[r:T]->(t) SET r.k = e.p0",
        params: edge_batch,
        operators: &[
            "Empty",
            "Unwind",
            "Scan",
            "Filter",
            "Scan",
            "Filter",
            "CreateEdge",
            "SetProperty",
            "Project",
        ],
        access_paths: BOTH_ENDPOINTS_INDEXED,
        declines: &[],
    },
    Contract {
        query: "UNWIND $es AS e MATCH (s:Node {id: e.s}), (t:Node {id: e.t}) \
                CREATE (s)-[r:T]->(t) RETURN count(r) AS c",
        params: edge_batch,
        operators: &[
            "Empty",
            "Unwind",
            "Scan",
            "Filter",
            "Scan",
            "Filter",
            "CreateEdge",
            "SimpleAggregate",
            "SimpleAggregate",
        ],
        access_paths: BOTH_ENDPOINTS_INDEXED,
        declines: &[],
    },
    Contract {
        query: "MATCH (n:Node {id: $id})-[:T]->(m) RETURN DISTINCT m.id AS id",
        params: node_app_id,
        operators: &["NodeScan", "NodeList", "Expand", "Distinct"],
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (n:Node {id: $id})-[]->(m) RETURN DISTINCT m.id AS id",
        params: node_app_id,
        operators: &["NodeScan", "NodeList", "Expand", "Distinct"],
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
];

#[test]
fn batch_shapes_build_their_contracted_plans() {
    let (db, fixture) = benchmark_db(true);
    let mut session = db.session();
    let failures = broken_lookups(&db, &session, BATCH_CONTRACTS, &fixture);
    assert!(
        failures.is_empty(),
        "batch plan contracts broken:\n{}",
        failures.join("\n")
    );

    session.begin_transaction().expect("begin");
    let failures = broken_lookups(&db, &session, BATCH_CONTRACTS, &fixture);
    assert!(
        failures.is_empty(),
        "batch plan contracts broken in a transaction:\n{}",
        failures.join("\n")
    );
    session.rollback().expect("rollback");

    // The batches did their work: two edges per edge batch, in autocommit.
    let edges = session
        .execute("MATCH (:Node)-[r:T]->(:Node) RETURN count(r) AS c")
        .expect("edge count");
    assert_eq!(edges.rows(), [[Value::Int64(1 + 5 * 2)]]);
}

#[test]
fn unindexed_edge_batch_is_rejected_by_the_indexed_contract() {
    // Control: without the index the same operators are built, and only the
    // access paths and the scanned work tell the batch apart.
    let (db, fixture) = benchmark_db(false);
    let session = db.session();
    let contract = &BARE_EDGE_BATCH;
    let (trace, work) =
        traced_with_work(&db, &session, contract.query, (contract.params)(&fixture));
    assert_eq!(trace.operators, BATCH_EDGE);
    assert_eq!(
        trace.access_paths,
        [AccessPath::CorrelatedScan, AccessPath::CorrelatedScan]
    );
    assert!(!satisfies(&trace, contract));
    assert!(
        scanned(work),
        "the candidate scans must be counted: {work:?}"
    );
    assert!(!broken_lookups(&db, &session, &[BARE_EDGE_BATCH], &fixture).is_empty());
}

fn person_ids_after_version(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![
        (
            "ids",
            Value::List(vec![Value::from("p_0"), Value::from("p_1")].into()),
        ),
        ("t", Value::Int64(0)),
    ]
}

fn age_over(_: &Fixture) -> Vec<(&'static str, Value)> {
    vec![("a", Value::Int64(20))]
}

const IN_LIST_WITH_RESIDUAL: &[&str] = &["NodeScan", "NodeList", "Filter", "Project"];

/// An indexed `IN` list with an unindexed residual, in both conjunct orders.
/// Filter pushdown leaves the last conjunct innermost, so the order once
/// decided whether the index was used at all.
const IN_LIST_CONTRACTS: &[Contract] = &[
    Contract {
        query: "MATCH (p:Person) WHERE p.id IN $ids AND p.version > $t RETURN p.id AS id",
        params: person_ids_after_version,
        operators: IN_LIST_WITH_RESIDUAL,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
    Contract {
        query: "MATCH (p:Person) WHERE p.version > $t AND p.id IN $ids RETURN p.id AS id",
        params: person_ids_after_version,
        operators: IN_LIST_WITH_RESIDUAL,
        access_paths: &[AccessPath::PropertyIndex],
        declines: &[],
    },
];

#[test]
fn indexed_in_list_is_used_whatever_the_conjunct_order() {
    let (db, fixture) = benchmark_db(true);
    let mut session = db.session();
    for transaction in [false, true] {
        if transaction {
            session.begin_transaction().expect("begin");
        }
        let failures = broken_lookups(&db, &session, IN_LIST_CONTRACTS, &fixture);
        assert!(
            failures.is_empty(),
            "IN-list contracts broken (transaction: {transaction}):\n{}",
            failures.join("\n")
        );
        for contract in IN_LIST_CONTRACTS {
            let rows = session
                .execute_with_params(
                    contract.query,
                    (contract.params)(&fixture)
                        .into_iter()
                        .map(|(name, value)| (name.to_string(), value))
                        .collect(),
                )
                .expect("IN-list read");
            assert_eq!(rows.rows(), [[Value::from("p_1")]], "{}", contract.query);
        }
    }
    session.rollback().expect("rollback");
}

const UNINDEXED_RANGE: &str = "MATCH (n:Node) WHERE n.age > $a RETURN n.id AS id";

/// Deliberate declines, each checked by the reason the planner records. A
/// regression that loses the optimization without the reason, or declines it
/// for a new reason, breaks the contract.
const DECLINE_CONTRACTS: &[Contract] = &[
    // The committed-latest range scan cannot see a transaction's writes.
    Contract {
        query: UNINDEXED_RANGE,
        params: age_over,
        operators: &["Scan", "Filter", "Project"],
        access_paths: &[],
        declines: &[Decline::UnindexedRangeView],
    },
    // The index cannot answer a list-valued equality.
    Contract {
        query: "MATCH (n:Node) WHERE n.id = $v RETURN n.id AS id",
        params: |_| vec![("v", Value::List(vec![Value::Int64(1)].into()))],
        operators: &["Scan", "Filter", "Project"],
        access_paths: &[],
        declines: &[Decline::StoreDeclined],
    },
    Contract {
        query: "MATCH (n:Node) WHERE n.id IN $v RETURN n.id AS id",
        params: |_| {
            vec![(
                "v",
                Value::List(vec![Value::List(vec![Value::Int64(1)].into())].into()),
            )]
        },
        operators: &["Scan", "Filter", "Project"],
        access_paths: &[],
        declines: &[Decline::StoreDeclined],
    },
    // The statement rewrites the lookup key, so a per-row lookup is unsafe.
    Contract {
        query: "UNWIND $es AS e MATCH (n:Node {id: e}) SET n.id = e",
        params: |_| vec![("es", Value::List(vec![Value::from("n_0")].into()))],
        operators: &[
            "Empty",
            "Unwind",
            "NestedLoopJoin",
            "Filter",
            "SetProperty",
            "Project",
        ],
        access_paths: &[],
        declines: &[Decline::CorrelatedAdmission],
    },
    // Known gap (Task 13): a trailing delete disables the lookup for the
    // whole statement. Pinned so the decline stays visible, not endorsed.
    Contract {
        query: "UNWIND $es AS e MATCH (n:Node {id: e}) DETACH DELETE n",
        params: |_| vec![("es", Value::List(vec![Value::from("absent")].into()))],
        operators: &[
            "Empty",
            "Unwind",
            "NestedLoopJoin",
            "Filter",
            "DeleteNode",
            "Project",
        ],
        access_paths: &[],
        declines: &[Decline::CorrelatedAdmission],
    },
];

#[test]
fn declined_optimizations_record_their_reason() {
    let (db, fixture) = benchmark_db(true);
    set_fixture_ages(&db);
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let failures: Vec<String> = DECLINE_CONTRACTS
        .iter()
        .filter_map(|contract| {
            let trace = traced(&session, contract.query, (contract.params)(&fixture));
            (!satisfies(&trace, contract)).then(|| mismatch(contract, &trace))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "decline contracts broken:\n{}",
        failures.join("\n")
    );
    session.rollback().expect("rollback");

    // Control: outside a transaction the same range is a range scan and
    // declines nothing, so the decline above belongs to the view.
    let reader = Contract {
        query: UNINDEXED_RANGE,
        params: age_over,
        operators: &["NodeScan", "RangeScan", "Project"],
        access_paths: &[],
        declines: &[],
    };
    let trace = traced(&db.session(), reader.query, (reader.params)(&fixture));
    assert!(satisfies(&trace, &reader), "{}", mismatch(&reader, &trace));
}

fn set_fixture_ages(db: &GrafeoDB) {
    db.session()
        .execute("MATCH (n:Node) SET n.age = 30")
        .expect("fixture ages");
}

#[test]
fn range_limit_bounds_materialization_and_counts_native_scan() {
    for size in [128, 512] {
        let db = GrafeoDB::new_in_memory();
        db.session()
            .execute(&format!(
                "UNWIND range(0, {}) AS age CREATE (:RangePerson {{age: age}})",
                size - 1
            ))
            .unwrap();
        for predicate in [
            "n.age >= 0",
            "n.age >= 0 AND n.age <= 1000",
            "n.age <= 1000 AND n.age >= 0",
        ] {
            let query = format!("MATCH (n:RangePerson) WHERE {predicate} RETURN n.age LIMIT 5");
            let store = crate::database::testing::root_lpg_store(&db);
            let before = store.work_snapshot();
            let (result, plans) = plan_trace::capture(|| db.session().execute(&query));
            let result = result.unwrap();
            assert_eq!(result.rows().len(), 5, "{query}");
            assert!(
                result
                    .rows()
                    .iter()
                    .all(|row| row[0].as_int64().is_some_and(|age| age >= 0 && age < size))
            );
            let work = store.work_snapshot().since(before);
            assert!(work.node_materializations <= 20, "{query}: {work:?}");
            assert_eq!(
                work.full_scan_ids,
                u64::try_from(size).unwrap(),
                "native scan must be counted: {query}: {work:?}"
            );
            assert_eq!(plans.len(), 1);
            assert_eq!(
                plans[0]
                    .operators
                    .iter()
                    .filter(|op| op.as_str() == "RangeScan")
                    .count(),
                1
            );
            assert!(
                !plans[0].operators.iter().any(|op| op == "Filter"),
                "both bounds must reach one range: {:?}",
                plans[0]
            );
        }
    }
}

#[test]
fn label_limit_bounds_entity_materialization() {
    for size in [128, 512] {
        let db = GrafeoDB::new_in_memory();
        db.session()
            .execute(&format!(
                "UNWIND range(0, {}) AS id CREATE (:LimitedPerson {{id: id}})",
                size - 1
            ))
            .unwrap();
        for projection in ["n", "n.id"] {
            let query = format!("MATCH (n:LimitedPerson) RETURN {projection} LIMIT 5");
            let store = crate::database::testing::root_lpg_store(&db);
            let before = store.work_snapshot();
            let result = db.session().execute(&query).unwrap();
            assert_eq!(result.rows().len(), 5, "{query}");
            let work = store.work_snapshot().since(before);
            assert!(work.node_materializations <= 10, "{query}: {work:?}");
            assert_eq!(
                work.label_scan_ids,
                u64::try_from(size).unwrap(),
                "identity enumeration is still unbounded"
            );
        }
    }
}
