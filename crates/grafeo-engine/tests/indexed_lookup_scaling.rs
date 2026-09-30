//! Acceptance: an indexed point lookup must not cost the label's cardinality.
//!
//! Guards against materialising an entire label set for each point lookup.
//! `MATCH (n:Node {id: $id})` measured 197 µs at 10,000 labelled nodes, 2,358 µs
//! at 100,000 and 30,641 µs at 1,000,000, against a flat ~10 µs for the
//! unlabelled `MATCH (n {id: $id})`. The index was selected in both cases; the
//! planner then materialised and sorted the label's entire id set to intersect
//! it with the single node the index had already returned.
//!
//! The unlabelled form is the control. The labelled form does the same index
//! hit plus a label check on the nodes the index returned, so anything beyond a
//! small constant factor means a per-lookup label set is still being built.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

use grafeo_common::types::Value;
use grafeo_core::graph::WorkSnapshot;
use grafeo_engine::{GrafeoDB, Session};

/// Nodes carrying `:Node`. At the defect's measured ~20 ns per labelled node
/// this is ~400 µs of intersection per lookup against a ~10 µs index hit.
const LABEL_CARDINALITY: usize = 20_000;
const ITERATIONS: usize = 15;
const MAX_LABELLED_RATIO: f64 = 8.0;

fn populated_db(label_cardinality: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for chunk_start in (0..label_cardinality).step_by(500) {
        let chunk_end = (chunk_start + 500).min(label_cardinality);
        let mut stmt = String::from("CREATE ");
        for (offset, i) in (chunk_start..chunk_end).enumerate() {
            if offset > 0 {
                stmt.push(',');
            }
            write!(stmt, "(:Node {{id: 'n_{i}', v: 0}})").expect("string write");
        }
        session.execute(&stmt).unwrap();
    }
    session
        .execute("CREATE INDEX gb_id FOR (n:Node) ON (n.id)")
        .unwrap();

    db
}

/// Median wall time of `query` over `ITERATIONS` runs, in microseconds.
///
/// Median, not mean: the first run pays plan-cache population and any lazy
/// index warm-up, which is not the per-lookup cost under test.
fn median_micros(session: &Session, query: &str, target: &str) -> f64 {
    let mut params = HashMap::new();
    params.insert("id".to_string(), Value::from(target));

    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let result = session
            .execute_with_params(query, params.clone())
            .expect("lookup must succeed");
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        assert_eq!(result.rows().len(), 1, "lookup must find exactly one node");
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
fn labelled_indexed_lookup_is_flat_in_label_cardinality() {
    let db = populated_db(LABEL_CARDINALITY);
    let session = db.session();
    let target = format!("n_{}", LABEL_CARDINALITY / 2);

    let unlabelled = median_micros(&session, "MATCH (n {id: $id}) RETURN n.id", &target);
    let labelled = median_micros(&session, "MATCH (n:Node {id: $id}) RETURN n.id", &target);
    let ratio = labelled / unlabelled.max(f64::MIN_POSITIVE);

    println!(
        "label cardinality {LABEL_CARDINALITY}: unlabelled {unlabelled:.1} µs, \
         labelled {labelled:.1} µs, ratio {ratio:.1}x"
    );
    assert!(
        ratio < MAX_LABELLED_RATIO,
        "labelled indexed lookup cost {labelled:.1} µs against {unlabelled:.1} µs unlabelled \
         ({ratio:.1}x) at {LABEL_CARDINALITY} labelled nodes: the label's id set is still being \
         built per lookup"
    );
}

#[test]
fn labelled_indexed_lookup_still_filters_by_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (:Node {id: 'shared'}), (:Other {id: 'shared'}),
                    (:Node:Extra {id: 'both'})",
        )
        .unwrap();
    session
        .execute("CREATE INDEX gb_id FOR (n:Node) ON (n.id)")
        .unwrap();

    // Two nodes carry id 'shared'; the label must select exactly one of them.
    let node = session
        .execute("MATCH (n:Node {id: 'shared'}) RETURN count(n)")
        .unwrap();
    assert_eq!(node.rows()[0][0], Value::from(1i64));

    let other = session
        .execute("MATCH (n:Other {id: 'shared'}) RETURN count(n)")
        .unwrap();
    assert_eq!(other.rows()[0][0], Value::from(1i64));

    // A label the node does not carry must match nothing.
    let absent = session
        .execute("MATCH (n:Missing {id: 'shared'}) RETURN count(n)")
        .unwrap();
    assert_eq!(absent.rows()[0][0], Value::from(0i64));

    // Either label of a multi-label node must match it.
    for label in ["Node", "Extra"] {
        let result = session
            .execute(&format!("MATCH (n:{label} {{id: 'both'}}) RETURN count(n)"))
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::from(1i64),
            "multi-label node must match :{label}"
        );
    }
}

#[test]
fn labelled_in_list_lookup_still_filters_by_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE (:Node {id: 'a'}), (:Node {id: 'b'}),
                    (:Other {id: 'c'}), (:Node:Extra {id: 'd'})",
        )
        .unwrap();
    session
        .execute("CREATE INDEX gb_id FOR (n:Node) ON (n.id)")
        .unwrap();

    let labelled = session
        .execute("MATCH (n:Node) WHERE n.id IN ['a', 'c', 'd'] RETURN n.id ORDER BY n.id")
        .unwrap();
    let ids: Vec<&Value> = labelled.rows().iter().map(|row| &row[0]).collect();
    assert_eq!(
        ids,
        vec![&Value::from("a"), &Value::from("d")],
        "the :Other node must not survive the :Node constraint"
    );

    let unlabelled = session
        .execute("MATCH (n) WHERE n.id IN ['a', 'c', 'd'] RETURN count(n)")
        .unwrap();
    assert_eq!(unlabelled.rows()[0][0], Value::from(3i64));
}

/// Internal node ids, as `Value::Int64`, for the first `limit` `:Node` rows.
fn internal_ids(session: &Session, limit: usize) -> Vec<Value> {
    let result = session
        .execute(&format!("MATCH (n:Node) RETURN id(n) LIMIT {limit}"))
        .expect("id() projection must succeed");
    let ids: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(ids.len(), limit, "fixture must supply {limit} internal ids");
    assert!(
        ids.iter().all(|v| matches!(v, Value::Int64(_))),
        "id(n) must project Int64"
    );
    ids
}

fn count_of(session: &Session, query: &str, ids: &[Value]) -> i64 {
    let mut params = HashMap::new();
    params.insert("ids".to_string(), Value::List(ids.to_vec().into()));
    let result = session
        .execute_with_params(query, params)
        .expect("id() lookup must succeed");
    match result.rows()[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("count(n) must be Int64, got {other:?}"),
    }
}

fn internal_id_of(session: &Session, application_id: &str) -> i64 {
    let result = session
        .execute(&format!(
            "MATCH (n:Node {{id: '{application_id}'}}) RETURN id(n)"
        ))
        .expect("lookup must succeed");
    assert_eq!(result.rows().len(), 1, "'{application_id}' must be unique");
    match result.rows()[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("id(n) must be Int64, got {other:?}"),
    }
}

fn retained_index_fixture(nodes: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (:Node {id: 'target', k: 10})")
        .expect("target must be created");
    for chunk_start in (0..nodes.saturating_sub(1)).step_by(500) {
        let chunk_end = (chunk_start + 500).min(nodes.saturating_sub(1));
        let mut stmt = String::from("CREATE ");
        for (offset, i) in (chunk_start..chunk_end).enumerate() {
            if offset > 0 {
                stmt.push(',');
            }
            write!(stmt, "(:Node {{id: 'n_{i}', k: {}}})", 100 + i).expect("string write");
        }
        if chunk_start < chunk_end {
            session.execute(&stmt).expect("fixture nodes must load");
        }
    }
    db
}

fn create_k_index(db: &GrafeoDB) {
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "k".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("property index must build");
}

fn count_query(session: &Session, query: &str) -> i64 {
    match session
        .execute(query)
        .expect("count query must succeed")
        .rows()[0][0]
    {
        Value::Int64(count) => count,
        ref value => panic!("count must be Int64, got {value:?}"),
    }
}

fn retained_index_work(nodes: usize, query: &str) -> (WorkSnapshot, i64) {
    let db = retained_index_fixture(nodes);
    let mut reader = db.session();
    reader
        .begin_transaction()
        .expect("reader transaction must begin");

    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("writer transaction must begin");
    writer
        .execute("MATCH (n:Node {id: 'target'}) SET n.k = 20")
        .expect("writer update must succeed");
    writer.commit().expect("writer commit must succeed");
    create_k_index(&db);

    let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
    let count = count_query(&reader, query);
    let work = grafeo_engine::database::testing::root_lpg_store(&db)
        .work_snapshot()
        .since(before);
    (work, count)
}

/// `MATCH (n) WHERE id(n) IN $ids` must be a direct fetch per id, not a scan.
///
/// Defect D4: `id(n)` only existed at runtime as a per-row column read, so the
/// planner had no plan-time representation of an internal id and every such
/// predicate ran a full `NodeScan` — 32.8 ms at 100k nodes. Every native
/// algorithm result pays this once per call when mapping internal ids back to
/// application ids.
#[test]
fn internal_id_lookup_is_flat_in_node_count() {
    let db = populated_db(LABEL_CARDINALITY);
    let session = db.session();

    // Collect real internal ids, then look them up by id() — the mapping every
    // native algorithm result has to do.
    let id_list = internal_ids(&session, 50);
    let mut params = HashMap::new();
    params.insert("ids".to_string(), Value::List(id_list.clone().into()));

    let expected = i64::try_from(id_list.len()).expect("id count fits in i64");

    let control = median_micros(&session, "MATCH (n {id: $id}) RETURN n.id", "n_1");
    // One point lookup per id is the budget: anything at or above it is still
    // paying for the node count rather than for the ids asked about.
    let budget = control * id_list.len() as f64;
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let result = session
            .execute_with_params(
                "MATCH (n) WHERE id(n) IN $ids RETURN count(n)",
                params.clone(),
            )
            .expect("id() lookup must succeed");
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        assert_eq!(result.rows()[0][0], Value::Int64(expected));
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];

    println!(
        "id() IN list of {}: {median:.1} µs, point-lookup control {control:.1} µs",
        id_list.len()
    );
    assert!(
        median < budget,
        "id() IN list cost {median:.1} µs for {} ids against a {control:.1} µs single point \
         lookup: still scanning",
        id_list.len()
    );
}

/// The direct fetch must answer exactly what the scan answered: the single-id
/// equality form, an id no node carries, a label the node does not have, and an
/// id whose node has since been deleted.
#[test]
fn internal_id_lookup_returns_the_same_rows_as_a_scan() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE (:Node {id: 'a'}), (:Node {id: 'b'}), (:Other {id: 'c'})")
        .unwrap();

    let a = internal_id_of(&session, "a");
    let b = internal_id_of(&session, "b");
    let ids = vec![Value::Int64(a), Value::Int64(b)];

    // Single-literal equality.
    let single = session
        .execute(&format!("MATCH (n) WHERE id(n) = {a} RETURN n.id"))
        .unwrap();
    assert_eq!(
        single.rows().len(),
        1,
        "id(n) = <literal> must match one node"
    );
    assert_eq!(single.rows()[0][0], Value::from("a"));

    // An id no node carries returns zero rows rather than erroring.
    let absent_id = a.max(b) + 10_000;
    let absent = session
        .execute(&format!("MATCH (n) WHERE id(n) = {absent_id} RETURN n.id"))
        .unwrap();
    assert!(
        absent.rows().is_empty(),
        "a nonexistent internal id must return no rows"
    );

    // The list form over the scan's own label.
    assert_eq!(
        count_of(
            &session,
            "MATCH (n:Node) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        2,
        ":Node must admit both ids"
    );

    // A label the nodes do not carry must match nothing, even though the ids
    // are valid and visible.
    assert_eq!(
        count_of(
            &session,
            "MATCH (n:Other) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        0,
        ":Other must reject ids belonging to :Node nodes"
    );

    // Unlabelled, both.
    assert_eq!(
        count_of(
            &session,
            "MATCH (n) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        2
    );

    // A deleted node's id must stop matching.
    session
        .execute("MATCH (n:Node {id: 'b'}) DELETE n")
        .unwrap();
    assert_eq!(
        count_of(
            &session,
            "MATCH (n) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        1,
        "a deleted node must not be fetched by its internal id"
    );
}

#[test]
fn internal_id_scalar_mapping_skips_unrelated_property_materialization() {
    for width in [32, 64, 128] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let names: Vec<_> = (0..width).map(|index| format!("unused_{index}")).collect();
        let targets: Vec<_> = ["alpha", "beta"]
            .into_iter()
            .map(|application_id| {
                session
                    .create_node_with_props(
                        &["Node"],
                        std::iter::once(("id", Value::from(application_id))).chain(
                            names
                                .iter()
                                .map(|name| (name.as_str(), Value::from("unused payload"))),
                        ),
                    )
                    .unwrap()
            })
            .collect();
        let wrong_label = session
            .create_node_with_props(&["Other"], [("id", Value::from("other"))])
            .unwrap();
        let deleted = session
            .create_node_with_props(&["Node"], [("id", Value::from("deleted"))])
            .unwrap();
        session
            .execute("MATCH (n:Node {id: 'deleted'}) DELETE n")
            .unwrap();
        let integer = |id: grafeo_common::types::NodeId| i64::try_from(id.as_u64()).unwrap();
        let mut ids = vec![
            Value::Int64(integer(targets[0])),
            Value::Int64(integer(targets[1])),
            Value::Int64(integer(targets[0])),
            Value::Int64(integer(wrong_label)),
            Value::Int64(integer(deleted)),
            Value::Int64(i64::MAX),
        ];
        let params = HashMap::from([(
            "ids".to_owned(),
            Value::List(std::mem::take(&mut ids).into()),
        )]);
        let expected = vec![
            vec![Value::Int64(integer(targets[0])), Value::from("alpha")],
            vec![Value::Int64(integer(targets[1])), Value::from("beta")],
        ];
        let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
        // Positive control: the counter must observe actual whole-node loading.
        let materialized = grafeo_engine::database::testing::root_lpg_store(&db)
            .get_node_at_epoch(
                targets[0],
                grafeo_engine::database::testing::root_lpg_store(&db).current_epoch(),
            )
            .unwrap();
        assert_eq!(materialized.properties.len(), width + 1);
        assert!(
            grafeo_engine::database::testing::root_lpg_store(&db)
                .work_snapshot()
                .since(before)
                .node_materializations
                > 0
        );
        drop(materialized);
        for iteration in 0..2 {
            let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
            let result = session
                .execute_with_params(
                    "MATCH (n:Node) WHERE id(n) IN $ids RETURN id(n) AS nid, n.id AS id",
                    params.clone(),
                )
                .unwrap();
            assert_eq!(result.rows(), expected.as_slice());
            let work = grafeo_engine::database::testing::root_lpg_store(&db)
                .work_snapshot()
                .since(before);
            assert!(!work.scanned_any(), "fixed-ID mapping scanned: {work:?}");
            assert_eq!(
                work.node_materializations, 0,
                "unrelated property width {width}: {work:?}"
            );
            if iteration == 0 {
                assert_eq!(
                    work.node_visibility_checks, 5,
                    "one check per unique candidate: {work:?}"
                );
            } else {
                assert!(
                    matches!(work.node_visibility_checks, 0 | 5),
                    "repeat may reuse an epoch-qualified plan: {work:?}"
                );
            }
        }
    }
}

#[test]
fn internal_id_lookup_preserves_historical_label_membership() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let old = session
        .create_node_with_props(&["Stable", "Node"], [("id", Value::from("old"))])
        .unwrap();
    let new = session
        .create_node_with_props(&["Stable"], [("id", Value::from("new"))])
        .unwrap();
    let old = i64::try_from(old.as_u64()).unwrap();
    let new = i64::try_from(new.as_u64()).unwrap();
    let epoch = grafeo_engine::database::testing::root_lpg_store(&db).current_epoch();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (n:Stable {id: 'old'}) REMOVE n:Node")
        .unwrap();
    writer
        .execute("MATCH (n:Stable {id: 'new'}) SET n:Node")
        .unwrap();
    writer.commit().unwrap();
    let ids = Value::List(
        vec![
            Value::Int64(old),
            Value::Int64(new),
            Value::Int64(old),
            Value::Int64(i64::MAX),
        ]
        .into(),
    );
    let params = HashMap::from([("ids".to_owned(), ids)]);
    let query = "MATCH (n:Node) WHERE id(n) IN $ids RETURN id(n) AS nid, n.id AS id";
    let previous = vec![vec![Value::Int64(old), Value::from("old")]];
    let current = vec![vec![Value::Int64(new), Value::from("new")]];
    assert_eq!(
        reader
            .execute_with_params(query, params.clone())
            .unwrap()
            .rows(),
        previous.as_slice()
    );
    assert_eq!(
        session
            .execute_at_epoch_with_params(query, epoch, Some(params.clone()))
            .unwrap()
            .rows(),
        previous.as_slice()
    );
    assert_eq!(
        session.execute_with_params(query, params).unwrap().rows(),
        current.as_slice()
    );
    for (id, old_count, new_count) in [(old, 1, 0), (new, 0, 1)] {
        let query = format!("MATCH (n:Node) WHERE id(n) = {id} RETURN count(n)");
        assert_eq!(count_query(&reader, &query), old_count);
        assert_eq!(
            session.execute_at_epoch(&query, epoch).unwrap().rows()[0][0],
            Value::Int64(old_count)
        );
        assert_eq!(count_query(&session, &query), new_count);
    }
    reader.rollback().unwrap();
}

#[test]
fn internal_id_lookup_respects_projected_hidden_labels() {
    use std::sync::Arc;

    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_core::graph::{GraphProjection, GraphStoreSearch, ProjectionSpec};
    use grafeo_engine::query::{Executor, Planner, translators::gql};

    let store = Arc::new(LpgStore::new().unwrap());
    let visible = store.create_node(&["Node", "Hidden"]);
    let hidden = store.create_node(&["Hidden"]);
    store.set_node_property(visible, "id", Value::from("visible"));
    store.set_node_property(hidden, "id", Value::from("hidden"));
    let projection = Arc::new(GraphProjection::new(
        store as Arc<dyn GraphStoreSearch>,
        ProjectionSpec::new().with_node_labels(["Node"]),
    ));
    for label in ["Node", "Hidden"] {
        for predicate in [
            format!("id(n) = {}", visible.as_u64()),
            format!(
                "id(n) IN [{}, {}, {}]",
                visible.as_u64(),
                hidden.as_u64(),
                visible.as_u64()
            ),
        ] {
            let logical = gql::translate(&format!(
                "MATCH (n:{label}) WHERE {predicate} RETURN id(n) AS nid, n.id AS id"
            ))
            .unwrap();
            let planner = Planner::new(Arc::clone(&projection) as Arc<dyn GraphStoreSearch>);
            let mut plan = planner.plan(&logical).unwrap();
            let result = Executor::with_columns(plan.columns.clone())
                .execute(plan.operator.as_mut())
                .unwrap();
            if label == "Node" {
                assert_eq!(
                    result.rows(),
                    &[vec![
                        Value::Int64(i64::try_from(visible.as_u64()).unwrap()),
                        Value::from("visible")
                    ]]
                );
            } else {
                assert!(
                    result.rows().is_empty(),
                    "projected-out label must not leak through ID seek"
                );
            }
        }
    }
}

/// The rewrite must stay enabled inside a write transaction: `get_node_versioned`
/// is delta-aware, so unlike the property index it sees the writer's own
/// buffered inserts and deletes.
#[test]
fn internal_id_lookup_sees_buffered_writes_in_a_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("CREATE (:Node {id: 'committed'})").unwrap();
    let committed = internal_id_of(&session, "committed");

    session.begin_transaction().expect("begin");
    session.execute("CREATE (:Node {id: 'buffered'})").unwrap();
    let buffered = internal_id_of(&session, "buffered");
    let ids = vec![Value::Int64(committed), Value::Int64(buffered)];

    assert_eq!(
        count_of(
            &session,
            "MATCH (n) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        2,
        "the writer must see its own buffered node through id()"
    );

    session
        .execute("MATCH (n:Node {id: 'committed'}) DELETE n")
        .unwrap();
    assert_eq!(
        count_of(
            &session,
            "MATCH (n) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        1,
        "a node deleted inside the transaction must not be fetched by id"
    );

    session.rollback().expect("rollback");

    assert_eq!(
        count_of(
            &session,
            "MATCH (n) WHERE id(n) IN $ids RETURN count(n)",
            &ids
        ),
        1,
        "after rollback only the committed node exists"
    );
    let survivor = session.execute("MATCH (n) RETURN n.id").unwrap();
    assert_eq!(survivor.rows().len(), 1);
    assert_eq!(survivor.rows()[0][0], Value::from("committed"));
}

#[test]
fn indexed_lookup_in_a_write_transaction_reads_its_own_buffered_writes() {
    // The sharpest case for serving a writer from the committed index: a SET that
    // moves an indexed value. The index still maps 'a' to the node and knows
    // nothing of 'c', so both halves of the rewrite are exercised at once — the
    // index hit for 'a' must be rejected, and the node must be found under 'c'
    // even though no index entry exists for it.
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session
        .execute("CREATE (:Node {id: 'a', v: 1}), (:Node {id: 'b', v: 2})")
        .unwrap();
    session
        .execute("CREATE INDEX gb_id FOR (n:Node) ON (n.id)")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Node {id: 'a'}) SET n.id = 'c'")
        .unwrap();

    let moved = session
        .execute("MATCH (n:Node {id: 'c'}) RETURN count(n)")
        .unwrap();
    assert_eq!(
        moved.rows()[0][0],
        Value::from(1i64),
        "the writer must see its own new value, which the committed index cannot know"
    );
    let stale = session
        .execute("MATCH (n:Node {id: 'a'}) RETURN count(n)")
        .unwrap();
    assert_eq!(
        stale.rows()[0][0],
        Value::from(0i64),
        "the writer must not see its own old value, which the committed index still reports"
    );
    // The IN-list path takes the same route.
    let in_list = session
        .execute("MATCH (n:Node) WHERE n.id IN ['a', 'c'] RETURN count(n)")
        .unwrap();
    assert_eq!(in_list.rows()[0][0], Value::from(1i64));

    session.rollback().unwrap();

    let after = session
        .execute("MATCH (n:Node {id: 'a'}) RETURN count(n)")
        .unwrap();
    assert_eq!(
        after.rows()[0][0],
        Value::from(1i64),
        "rollback must restore the committed value"
    );
    let gone = session
        .execute("MATCH (n:Node {id: 'c'}) RETURN count(n)")
        .unwrap();
    assert_eq!(gone.rows()[0][0], Value::from(0i64));
}

fn labelled_property_lookup_reads_own_label_changes_case(indexed: bool) {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {id: 'a'})").unwrap();
    if indexed {
        writer
            .execute("CREATE INDEX gb_id FOR (n:Person) ON (n.id)")
            .unwrap();
    }
    writer.begin_transaction().unwrap();
    writer.execute("MATCH (n:Person) SET n:Secret").unwrap();
    for predicate in ["n.id = 'a'", "n.id IN ['a']"] {
        let result = writer
            .execute(&format!(
                "MATCH (n:Secret) WHERE {predicate} RETURN count(n)"
            ))
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(1),
            "own added label must match: indexed={indexed}, predicate={predicate}"
        );
    }
    writer.execute("MATCH (n:Person) REMOVE n:Person").unwrap();
    for predicate in ["n.id = 'a'", "n.id IN ['a']"] {
        let result = writer
            .execute(&format!(
                "MATCH (n:Person) WHERE {predicate} RETURN count(n)"
            ))
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(0),
            "own removed label must not match: indexed={indexed}, predicate={predicate}"
        );
    }
    writer.rollback().unwrap();
}

#[test]
fn labelled_property_lookup_reads_own_label_changes_without_index() {
    labelled_property_lookup_reads_own_label_changes_case(false);
}

#[test]
fn labelled_property_lookup_reads_own_label_changes_with_index() {
    labelled_property_lookup_reads_own_label_changes_case(true);
}

#[test]
fn labelled_property_lookup_records_empty_predicate_for_serializable() {
    use grafeo_engine::transaction::IsolationLevel;

    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE (:A {k: 0}), (:B {k: 0})")
        .unwrap();
    let mut first = db.session();
    let mut second = db.session();
    first
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    second
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    // Neither read returns a node. The absorbed label scan must still record
    // its predicate, so inserting into each other's gaps cannot both commit.
    for (session, label) in [(&first, "A"), (&second, "B")] {
        let result = session
            .execute(&format!("MATCH (n:{label} {{k: 1}}) RETURN count(n)"))
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(0));
    }
    first.execute("CREATE (:B {k: 1})").unwrap();
    second.execute("CREATE (:A {k: 1})").unwrap();
    let commits = [first.commit(), second.commit()];
    let errors: Vec<_> = commits
        .iter()
        .filter_map(|result| result.as_ref().err())
        .collect();
    assert!(
        !errors.is_empty(),
        "both sides of a predicate cycle committed"
    );
    for error in errors {
        assert!(
            error.to_string().contains("Serialization failure"),
            "expected an SSI rejection, got {error}"
        );
    }
}

fn empty_indexed_property_predicate_conflicts_with_existing_node_updates(predicate: &str) {
    use grafeo_engine::transaction::{ConflictGranularity, IsolationLevel};

    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE (:A {k: 0}), (:B {k: 0})")
        .unwrap();
    create_k_index(&db);
    let mut first = db.session();
    let mut second = db.session();
    for session in [&mut first, &mut second] {
        session.set_conflict_granularity(ConflictGranularity::Property);
        session
            .begin_transaction_with_isolation(IsolationLevel::Serializable)
            .unwrap();
    }

    let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
    for (session, label) in [(&first, "A"), (&second, "B")] {
        assert_eq!(
            count_query(
                session,
                &format!("MATCH (n:{label}) WHERE {predicate} RETURN count(n)"),
            ),
            0,
            "the indexed predicate must initially be empty",
        );
    }
    let work = grafeo_engine::database::testing::root_lpg_store(&db)
        .work_snapshot()
        .since(before);
    assert!(!work.scanned_any(), "empty indexed reads scanned: {work:?}");

    // No structural mutation can supply a wildcard conflict accidentally:
    // both gaps are filled by changing an existing node's property only.
    first.execute("MATCH (n:B) SET n.k = 1").unwrap();
    second.execute("MATCH (n:A) SET n.k = 1").unwrap();
    let commits = [first.commit(), second.commit()];
    let errors: Vec<_> = commits
        .iter()
        .filter_map(|result| result.as_ref().err())
        .collect();
    assert!(
        !errors.is_empty(),
        "both sides of the empty indexed {predicate} SSI cycle committed at property granularity",
    );
    for error in errors {
        assert!(
            error.to_string().contains("Serialization failure"),
            "expected an SSI rejection, got {error}",
        );
    }
}

#[test]
fn empty_indexed_equality_conflicts_with_existing_node_updates() {
    empty_indexed_property_predicate_conflicts_with_existing_node_updates("n.k = 1");
}

#[test]
fn empty_indexed_in_conflicts_with_existing_node_updates() {
    empty_indexed_property_predicate_conflicts_with_existing_node_updates("n.k IN [1]");
}

#[test]
fn empty_indexed_range_conflicts_with_existing_node_updates() {
    empty_indexed_property_predicate_conflicts_with_existing_node_updates("n.k >= 1 AND n.k <= 1");
}

fn property_index_ssi_independent_writers(read_index: bool) {
    use grafeo_engine::transaction::{ConflictGranularity, IsolationLevel};

    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE (:A {k: 1, other: 0}), (:B {k: 2, other: 0})")
        .unwrap();
    create_k_index(&db);
    let first_id = grafeo_engine::database::testing::root_lpg_store(&db)
        .find_nodes_by_property("k", &Value::Int64(1))[0];
    let second_id = grafeo_engine::database::testing::root_lpg_store(&db)
        .find_nodes_by_property("k", &Value::Int64(2))[0];
    let mut first = db.session();
    let mut second = db.session();
    for session in [&mut first, &mut second] {
        session.set_conflict_granularity(ConflictGranularity::Property);
        session
            .begin_transaction_with_isolation(IsolationLevel::Serializable)
            .unwrap();
    }
    if read_index {
        assert_eq!(
            count_query(&first, "MATCH (n:A) WHERE n.k = 1 RETURN count(n)"),
            1
        );
        assert_eq!(
            count_query(&second, "MATCH (n:B) WHERE n.k = 2 RETURN count(n)"),
            1
        );
    }
    // Direct writes avoid introducing an indexed predicate through their MATCH.
    // In the read-free case both write the SAME indexed property on distinct
    // nodes; in the other case index reads of k cannot conflict with other.
    let property = if read_index { "other" } else { "k" };
    first
        .set_node_property(first_id, property, Value::Int64(11))
        .unwrap();
    second
        .set_node_property(second_id, property, Value::Int64(22))
        .unwrap();
    first
        .commit()
        .expect("first independent property writer must commit");
    second
        .commit()
        .expect("coarse property predicate must not impose a write-write lock");
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_node_property(first_id, &property.into()),
        Some(Value::Int64(11))
    );
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_node_property(second_id, &property.into()),
        Some(Value::Int64(22))
    );
}

#[test]
fn indexed_property_predicates_preserve_disjoint_property_writes() {
    property_index_ssi_independent_writers(true);
}

#[test]
fn indexed_property_predicates_allow_read_free_same_property_writers() {
    property_index_ssi_independent_writers(false);
}

#[test]
fn indexed_property_predicates_preserve_same_node_write_write_conflicts() {
    use grafeo_common::utils::error::{Error, TransactionError};
    use grafeo_engine::transaction::{ConflictGranularity, IsolationLevel};

    let db = GrafeoDB::new_in_memory();
    db.session().execute("CREATE (:Node {k: 1})").unwrap();
    create_k_index(&db);
    let id = grafeo_engine::database::testing::root_lpg_store(&db)
        .find_nodes_by_property("k", &Value::Int64(1))[0];
    let mut first = db.session();
    let mut second = db.session();
    for session in [&mut first, &mut second] {
        session.set_conflict_granularity(ConflictGranularity::Property);
        session
            .begin_transaction_with_isolation(IsolationLevel::Serializable)
            .unwrap();
    }
    first.set_node_property(id, "k", Value::Int64(11)).unwrap();
    let second_write = second.set_node_property(id, "k", Value::Int64(22));
    first.commit().unwrap();
    let error = second_write
        .and_then(|()| second.commit())
        .expect_err("same-node property writers must still conflict");
    assert!(
        matches!(
            error,
            Error::Transaction(TransactionError::WriteConflict(_))
        ),
        "expected a real node write-write conflict, got {error}"
    );
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db).get_node_property(id, &"k".into()),
        Some(Value::Int64(11))
    );
}

#[test]
fn internal_id_lookup_reads_own_label_changes_and_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("CREATE (:Node:Person {id: 'a'})").unwrap();
    let node_id = internal_id_of(&session, "a");
    let ids = vec![Value::Int64(node_id)];

    let assert_label_count = |session: &Session, label: &str, expected: i64, phase: &str| {
        let equality = session
            .execute(&format!(
                "MATCH (n:{label}) WHERE id(n) = {node_id} RETURN count(n)"
            ))
            .unwrap();
        assert_eq!(
            equality.rows()[0][0],
            Value::Int64(expected),
            "{phase}: labeled internal-id equality must honor :{label}"
        );

        let in_list = count_of(
            session,
            &format!("MATCH (n:{label}) WHERE id(n) IN $ids RETURN count(n)"),
            &ids,
        );
        assert_eq!(
            in_list, expected,
            "{phase}: labeled internal-id IN must honor :{label}"
        );
    };

    session.begin_transaction().unwrap();
    session.execute("MATCH (n:Person) SET n:Secret").unwrap();
    assert_label_count(&session, "Secret", 1, "own label add");

    session.execute("MATCH (n:Person) REMOVE n:Person").unwrap();
    assert_label_count(&session, "Person", 0, "own label removal");

    session.rollback().unwrap();
    assert_label_count(&session, "Person", 1, "after rollback");
    assert_label_count(&session, "Secret", 0, "after rollback");
}

#[test]
fn retained_index_snapshot_equality_in_range_and_own_overlay() {
    let db = retained_index_fixture(4);
    let mut reader = db.session();
    reader
        .begin_transaction()
        .expect("reader transaction must begin");

    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("writer transaction must begin");
    writer
        .execute("MATCH (n:Node {id: 'target'}) SET n.k = 20")
        .expect("writer update must succeed");
    writer.commit().expect("writer commit must succeed");
    // Build the latest-only index after reader has retained the k=10 snapshot.
    create_k_index(&db);

    for query in [
        "MATCH (n:Node) WHERE n.k = 10 RETURN count(n)",
        "MATCH (n:Node) WHERE n.k IN [10] RETURN count(n)",
        "MATCH (n:Node) WHERE n.k >= 10 AND n.k <= 10 RETURN count(n)",
    ] {
        assert_eq!(
            count_query(&reader, query),
            1,
            "reader must see k=10: {query}"
        );
    }
    for query in [
        "MATCH (n:Node) WHERE n.k = 20 RETURN count(n)",
        "MATCH (n:Node) WHERE n.k IN [20] RETURN count(n)",
        "MATCH (n:Node) WHERE n.k >= 20 AND n.k <= 20 RETURN count(n)",
    ] {
        assert_eq!(
            count_query(&reader, query),
            0,
            "reader must hide k=20: {query}"
        );
    }

    reader
        .execute("MATCH (n:Node {id: 'target'}) SET n.k = 30")
        .expect("reader own property update must succeed");
    reader
        .execute("MATCH (n:Node {id: 'target'}) SET n:Own")
        .expect("reader own label update must succeed");
    for query in [
        "MATCH (n:Own) WHERE n.k = 30 RETURN count(n)",
        "MATCH (n:Own) WHERE n.k IN [30] RETURN count(n)",
        "MATCH (n:Own) WHERE n.k >= 30 AND n.k <= 30 RETURN count(n)",
    ] {
        assert_eq!(
            count_query(&reader, query),
            1,
            "reader must see own k=30: {query}"
        );
    }
    reader.rollback().expect("reader rollback must succeed");

    let after = db.session();
    assert_eq!(
        count_query(&after, "MATCH (n:Node) WHERE n.k = 20 RETURN count(n)"),
        1,
        "reader rollback must preserve writer's k=20"
    );
}

#[test]
fn retained_index_property_remove_readd_a_b_a() {
    let db = retained_index_fixture(4);
    create_k_index(&db);
    let mut session = db.session();
    session.begin_transaction().expect("transaction must begin");
    session
        .execute("MATCH (n:Node {id: 'target'}) SET n.k = 20")
        .expect("property update must succeed");
    session
        .execute("MATCH (n:Node {id: 'target'}) REMOVE n.k")
        .expect("property removal must succeed");
    session
        .execute("MATCH (n:Node {id: 'target'}) SET n.k = 10")
        .expect("property re-add must succeed");

    for query in [
        "MATCH (n:Node) WHERE n.k = 10 RETURN count(n)",
        "MATCH (n:Node) WHERE n.k IN [10] RETURN count(n)",
        "MATCH (n:Node) WHERE n.k >= 10 AND n.k <= 10 RETURN count(n)",
    ] {
        assert_eq!(
            count_query(&session, query),
            1,
            "A-B-A must restore k=10: {query}"
        );
    }
    assert_eq!(
        count_query(&session, "MATCH (n:Node) WHERE n.k = 20 RETURN count(n)"),
        0,
        "A-B-A must hide stale k=20"
    );
    session.rollback().expect("rollback must succeed");
    assert_eq!(
        count_query(&session, "MATCH (n:Node) WHERE n.k = 10 RETURN count(n)"),
        1,
        "rollback must retain the original k=10"
    );
}

#[test]
fn retained_index_work_is_flat_for_equality_in_and_range() {
    // This is a fixed result-sized budget, shared by both fixture sizes.  It
    // covers the requested value hits, typed alias routes, retained history
    // intervals, and a bounded AVL boundary path; index cardinality must not
    // enter the probe budget.
    const MAX_RETAINED_PROBE_WORK: u64 = 1_000;
    const MAX_ROUTING_HEIGHT_DELTA: u64 = 64;
    let queries = [
        "MATCH (n:Node) WHERE n.k = 10 RETURN count(n)",
        "MATCH (n:Node) WHERE n.k IN [10] RETURN count(n)",
        "MATCH (n:Node) WHERE n.k >= 10 AND n.k <= 10 RETURN count(n)",
    ];
    for query in queries {
        let (small, small_count) = retained_index_work(1_000, query);
        let (large, large_count) = retained_index_work(4_000, query);
        assert_eq!(small_count, 1, "retained reader must see k=10: {query}");
        assert_eq!(large_count, 1, "retained reader must see k=10: {query}");
        assert!(
            !small.scanned_any(),
            "retained indexed lookup scanned: {small:?}"
        );
        assert!(
            !large.scanned_any(),
            "retained indexed lookup scanned: {large:?}"
        );
        let small_probe_work = small.property_index_route_keys
            + small.property_index_posting_ids
            + small.property_index_posting_intervals;
        let large_probe_work = large.property_index_route_keys
            + large.property_index_posting_ids
            + large.property_index_posting_intervals;
        assert!(
            small_probe_work <= MAX_RETAINED_PROBE_WORK,
            "retained indexed probe inspected {small_probe_work} ordered/posting items at 1,000 nodes: {small:?}"
        );
        assert!(
            large_probe_work <= MAX_RETAINED_PROBE_WORK,
            "retained indexed probe inspected {large_probe_work} ordered/posting items at 4,000 nodes: {large:?}"
        );
        assert!(
            large_probe_work <= small_probe_work + MAX_ROUTING_HEIGHT_DELTA,
            "retained indexed probe added {growth} items at 4,000 nodes (small={small_probe_work}, large={large_probe_work}): {large:?}",
            growth = large_probe_work.saturating_sub(small_probe_work),
        );
        assert_eq!(
            (
                small.property_index_posting_ids,
                small.property_index_posting_intervals
            ),
            (
                large.property_index_posting_ids,
                large.property_index_posting_intervals
            ),
            "retained matching-posting work grew with unrelated values for {query}"
        );
    }
}

fn boundary_fixture(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(
            &["Node"],
            [
                ("id", Value::from("low")),
                ("k", Value::Int64(9_007_199_254_740_992)),
            ],
        )
        .expect("low boundary node must be created");
    session
        .create_node_with_props(
            &["Node"],
            [
                ("id", Value::from("high")),
                ("k", Value::Int64(9_007_199_254_740_993)),
            ],
        )
        .expect("high boundary node must be created");
    if indexed {
        create_k_index(&db);
    }
    db
}

#[test]
fn retained_index_numeric_boundary_matches_unindexed_oracle() {
    let indexed = boundary_fixture(true);
    let oracle = boundary_fixture(false);
    for query in [
        "MATCH (n:Node) WHERE n.k = 9007199254740992 RETURN n.id ORDER BY n.id",
        "MATCH (n:Node) WHERE n.k = 9007199254740993 RETURN n.id ORDER BY n.id",
        "MATCH (n:Node) WHERE n.k IN [9007199254740992, 9007199254740993] RETURN n.id ORDER BY n.id",
        "MATCH (n:Node) WHERE n.k >= 9007199254740992 AND n.k <= 9007199254740993 RETURN n.id ORDER BY n.id",
    ] {
        let indexed_rows = indexed.session().execute(query).unwrap();
        let oracle_rows = oracle.session().execute(query).unwrap();
        assert_eq!(
            indexed_rows.rows(),
            oracle_rows.rows(),
            "indexed result diverged from unindexed oracle: {query}"
        );
    }
    // ExpressionPredicate's current Int64/Float64 coercion makes both adjacent
    // integers equal to the 2^53 Float64 value. Keep the unindexed oracle
    // explicit so an indexed path cannot silently use lossy key equality.
    for query in [
        "MATCH (n:Node) WHERE n.k = 9007199254740992.0 RETURN n.id ORDER BY n.id",
        "MATCH (n:Node) WHERE n.k IN [9007199254740992.0] RETURN n.id ORDER BY n.id",
        "MATCH (n:Node) WHERE n.k >= 9007199254740992.0 AND n.k <= 9007199254740992.0 RETURN n.id ORDER BY n.id",
    ] {
        let indexed_rows = indexed.session().execute(query).unwrap();
        // Remove the label from the oracle so the no-index planner reaches the
        // generic ExpressionPredicate instead of its label-first shortcut.
        let oracle_query = query.replacen("MATCH (n:Node)", "MATCH (n)", 1);
        let oracle_rows = oracle.session().execute(&oracle_query).unwrap();
        assert_eq!(
            oracle_rows.rows().len(),
            2,
            "Float64 2^53 coercion oracle must match both adjacent integers: {query}"
        );
        assert_eq!(
            indexed_rows.rows(),
            oracle_rows.rows(),
            "indexed Float64 result diverged from unindexed oracle: {query}"
        );
    }
}

// Task 3: a real mutating statement straddles candidate admission and commit.
mod concurrent_index_capture {
    use std::sync::Arc;

    use arcstr::ArcStr;
    use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
    use grafeo_common::utils::hash::FxHashMap;
    use grafeo_core::graph::lpg::{CompareOp, Edge, LpgStore, Node};
    use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut, GraphStoreSearch};
    use grafeo_core::statistics::Statistics;

    use super::count_query;
    use grafeo_common::utils::error::Result;
    use grafeo_core::execution::operators::{SharedReadTracker, SharedWriteTracker};
    use grafeo_core::graph::lpg::TxDelta;
    use grafeo_core::graph::traits::LpgCommitTarget;
    use grafeo_core::graph::{PropertyIndexRequest, TxStructuralSnapshot};
    use grafeo_engine::{Config, GrafeoDB};
    use std::sync::{Mutex, mpsc};
    use std::time::Duration;

    // A single-use pause at the existing indexed-lookup capability boundary.
    // All storage and native publication remain on the real LPG store.
    struct PausedLookup(
        Arc<LpgStore>,
        Mutex<Option<(mpsc::Sender<EpochId>, mpsc::Receiver<()>)>>,
    );

    macro_rules! forward {
    ($source:ident; $(fn $method:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)+) => {
        $(fn $method(&self, $($arg: $ty),*) $(-> $ret)? {
            $source::$method(self.0.as_ref(), $($arg),*)
        })+
    };
}

    impl GraphStore for PausedLookup {
        forward! { GraphStore;
            fn lpg_commit_target() -> Result<LpgCommitTarget<'_>>;
            fn has_property_index(property: &str) -> bool;
            fn pending_node_creates(tx: TransactionId) -> Vec<NodeId>;
            fn pending_edge_creates(tx: TransactionId) -> Vec<EdgeId>;
            fn pending_node_deletes_peek(tx: TransactionId) -> Vec<NodeId>;
            fn pending_edge_deletes_peek(tx: TransactionId) -> Vec<EdgeId>;
            fn overlay_touched_entities(tx: TransactionId) -> (Vec<NodeId>, Vec<EdgeId>);
            fn register_read_tracker(tx: TransactionId, tracker: SharedReadTracker);
            fn unregister_read_tracker(tx: TransactionId);
            fn register_write_tracker(tx: TransactionId, tracker: SharedWriteTracker);
            fn unregister_write_tracker(tx: TransactionId);
            fn record_label_predicate_read(tx: TransactionId, label: &str);
            fn record_lpg_dataset_read(tx: TransactionId);
            fn read_node_property_visible(id: NodeId, key: &PropertyKey, epoch: EpochId, tx: Option<TransactionId>) -> Option<Value>;
            fn read_node_properties_visible(id: NodeId, epoch: EpochId, tx: Option<TransactionId>) -> FxHashMap<PropertyKey, Value>;
            fn prepare_index_node_rows(epoch: EpochId, tx: Option<TransactionId>) -> Result<Vec<Node>>;
            fn prepare_index_node_rows_by_id(epoch: EpochId, tx: Option<TransactionId>, ids: &[NodeId]) -> Result<Vec<Node>>;
            fn nodes_with_buffered_property(tx: TransactionId, key: &PropertyKey) -> Option<Vec<NodeId>>;
            fn node_has_label_visible(id: NodeId, label: &str, tx: Option<TransactionId>) -> bool;
            fn node_has_label_at_epoch(id: NodeId, label: &str, epoch: EpochId, tx: TransactionId) -> bool;
            fn nodes_by_label_visible(label: &str, tx: Option<TransactionId>) -> Vec<NodeId>;

            fn get_node(id: NodeId) -> Option<Node>;
            fn get_edge(id: EdgeId) -> Option<Edge>;
            fn get_node_versioned(id: NodeId, epoch: EpochId, tx: TransactionId) -> Option<Node>;
            fn get_edge_versioned(id: EdgeId, epoch: EpochId, tx: TransactionId) -> Option<Edge>;
            fn get_node_at_epoch(id: NodeId, epoch: EpochId) -> Option<Node>;
            fn get_edge_at_epoch(id: EdgeId, epoch: EpochId) -> Option<Edge>;
            fn get_node_property(id: NodeId, key: &PropertyKey) -> Option<Value>;
            fn get_edge_property(id: EdgeId, key: &PropertyKey) -> Option<Value>;
            fn get_node_property_batch(ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>>;
            fn get_nodes_properties_batch(ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>>;
            fn get_nodes_properties_selective_batch(ids: &[NodeId], keys: &[PropertyKey]) -> Vec<FxHashMap<PropertyKey, Value>>;
            fn get_edges_properties_selective_batch(ids: &[EdgeId], keys: &[PropertyKey]) -> Vec<FxHashMap<PropertyKey, Value>>;
            fn neighbors(id: NodeId, direction: Direction) -> Vec<NodeId>;
            fn edges_from(id: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)>;
            fn out_degree(id: NodeId) -> usize;
            fn in_degree(id: NodeId) -> usize;
            fn has_backward_adjacency() -> bool;
            fn node_ids() -> Vec<NodeId>;
            fn nodes_by_label(label: &str) -> Vec<NodeId>;
            fn node_count() -> usize;
            fn edge_count() -> usize;
            fn edge_type(id: EdgeId) -> Option<ArcStr>;
            fn find_nodes_by_property(key: &str, value: &Value) -> Vec<NodeId>;
            fn find_nodes_by_properties(conditions: &[(&str, Value)]) -> Vec<NodeId>;
            fn find_nodes_in_range(key: &str, min: Option<&Value>, max: Option<&Value>, min_inclusive: bool, max_inclusive: bool) -> Vec<NodeId>;
            fn node_property_might_match(key: &PropertyKey, op: CompareOp, value: &Value) -> bool;
            fn edge_property_might_match(key: &PropertyKey, op: CompareOp, value: &Value) -> bool;
            fn statistics() -> Arc<Statistics>;
            fn estimate_label_cardinality(label: &str) -> f64;
            fn estimate_avg_degree(edge_type: &str, outgoing: bool) -> f64;
            fn current_epoch() -> EpochId;
        }
    }

    impl GraphStoreSearch for PausedLookup {
        fn lookup_nodes_indexed(
            &self,
            request: PropertyIndexRequest<'_>,
        ) -> Result<Option<Vec<NodeId>>> {
            let pause = self.1.lock().unwrap().take();
            if let Some((entered, resume)) = pause {
                assert_eq!(request.property, "k");
                assert!(
                    request.transaction_id.is_some(),
                    "pause must belong to the mutating reader"
                );
                entered.send(request.epoch).unwrap();
                resume
                    .recv_timeout(Duration::from_secs(30))
                    .expect("publication must release the indexed reader");
            }
            GraphStoreSearch::lookup_nodes_indexed(self.0.as_ref(), request)
        }
    }

    impl GraphStoreMut for PausedLookup {
        fn lpg_commit_store(self: Arc<Self>) -> Option<Arc<LpgStore>> {
            Some(Arc::clone(&self.0))
        }
        forward! { GraphStoreMut;
            fn set_node_property_buffered(id: NodeId, key: &str, value: Value, tx: TransactionId);
            fn remove_node_property_buffered(id: NodeId, key: &str, tx: TransactionId);
            fn add_label_buffered(id: NodeId, label: &str, tx: TransactionId);
            fn remove_label_buffered(id: NodeId, label: &str, tx: TransactionId);
            fn drop_tx_overlay(tx: TransactionId);
            fn tx_overlay_snapshot(tx: TransactionId) -> TxDelta;
            fn tx_overlay_restore(tx: TransactionId, snapshot: TxDelta);
            fn tx_structural_snapshot(tx: TransactionId) -> TxStructuralSnapshot;
            fn tx_structural_restore(tx: TransactionId, snapshot: TxStructuralSnapshot) -> std::result::Result<(), String>;

            fn create_node(labels: &[&str]) -> NodeId;
            fn create_node_versioned(labels: &[&str], epoch: EpochId, tx: TransactionId) -> NodeId;
            fn create_edge(src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId;
            fn create_edge_versioned(src: NodeId, dst: NodeId, edge_type: &str, epoch: EpochId, tx: TransactionId) -> EdgeId;
            fn batch_create_edges(edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId>;
            fn delete_node(id: NodeId) -> bool;
            fn delete_node_versioned(id: NodeId, epoch: EpochId, tx: TransactionId) -> bool;
            fn delete_node_edges(id: NodeId);
            fn delete_edge(id: EdgeId) -> bool;
            fn delete_edge_versioned(id: EdgeId, epoch: EpochId, tx: TransactionId) -> bool;
            fn set_node_property(id: NodeId, key: &str, value: Value);
            fn set_edge_property(id: EdgeId, key: &str, value: Value);
            fn remove_node_property(id: NodeId, key: &str) -> Option<Value>;
            fn remove_edge_property(id: EdgeId, key: &str) -> Option<Value>;
            fn add_label(id: NodeId, label: &str) -> bool;
            fn remove_label(id: NodeId, label: &str) -> bool;
        }
    }

    #[test]
    fn mutating_index_capture_retains_old_value_across_concurrent_publication() {
        for predicate in ["n.k = 10", "n.k IN [10]", "n.k >= 10 AND n.k <= 10"] {
            let inner = Arc::new(LpgStore::new().unwrap());
            // Register on the exact native store before engine ownership. Even
            // the fixture population is then published through real maintenance.
            inner.create_property_index("k");
            assert!(inner.has_property_index("k"));
            let source = Arc::new(PausedLookup(Arc::clone(&inner), Mutex::new(None)));
            let db = GrafeoDB::with_store(
                Arc::clone(&source) as Arc<dyn GraphStoreMut>,
                Config::in_memory(),
            )
            .unwrap();
            db.session()
                .execute("CREATE (:Node {k: 10}), (:Scratch {k: -1})")
                .unwrap();
            let target = inner.find_nodes_by_property("k", &Value::Int64(10))[0];
            let mut reader = db.session();
            reader.begin_transaction().unwrap();
            reader
                .execute("MATCH (n:Scratch) WHERE n.k = -1 SET n:Own")
                .unwrap();
            reader
                .execute("MATCH (n:Scratch) WHERE n.k = -1 REMOVE n:Scratch")
                .unwrap();
            let mut writer = db.session();
            writer.begin_transaction().unwrap();
            writer
                .execute("MATCH (n:Node) WHERE n.k = 10 SET n.k = 20")
                .unwrap();

            let (entered_tx, entered_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            *source.1.lock().unwrap() = Some((entered_tx, resume_rx));
            let before = inner.work_snapshot();
            // Prior own-label writes affect the sentinel; the matching target
            // is first mutated only after the paused lookup resumes.
            let query = format!("MATCH (n:Node) WHERE {predicate} SET n.seen = 1 RETURN n.k");
            let worker = std::thread::spawn(move || {
                let result = reader.execute(&query);
                let own = count_query(&reader, "MATCH (n:Own) WHERE n.k = -1 RETURN count(n)");
                let removed =
                    count_query(&reader, "MATCH (n:Scratch) WHERE n.k = -1 RETURN count(n)");
                reader.rollback().unwrap();
                (result, own, removed)
            });
            let old_epoch = entered_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("mutating statement must reach the registered lookup");
            let (committed_tx, committed_rx) = mpsc::channel();
            let publisher = std::thread::spawn(move || {
                let result = writer.commit();
                committed_tx.send(result).unwrap();
            });
            // Release the reader even on failure before inspecting assertions: a
            // broken publication lock must not strand either test thread.
            let committed = committed_rx.recv_timeout(Duration::from_secs(10));
            let newer_epoch = inner.current_epoch();
            resume_tx.send(()).unwrap();
            publisher.join().unwrap();
            let (result, own, removed) = worker.join().unwrap();
            committed
                .expect("writer must publish while the old indexed capture is paused")
                .unwrap();
            assert!(newer_epoch > old_epoch);
            assert_eq!(
                result.unwrap().rows(),
                &[vec![Value::Int64(10)]],
                "old snapshot lost A after B published: {predicate}"
            );
            assert_eq!(
                (own, removed),
                (1, 0),
                "own added/removed labels must survive the interleaving"
            );
            let work = inner.work_snapshot().since(before);
            assert!(
                !work.scanned_any(),
                "indexed interleaving scanned: {work:?}"
            );
            assert_eq!(
                work.property_index_rebuild_rows, 0,
                "ordinary commit rebuilt its index"
            );
            assert!(
                work.property_index_posting_intervals > 0,
                "real retained postings must be probed"
            );
            let current = db.session();
            assert_eq!(
                count_query(&current, "MATCH (n:Node) WHERE n.k = 20 RETURN count(n)"),
                1
            );
            assert_eq!(
                count_query(&current, "MATCH (n:Node) WHERE n.k = 10 RETURN count(n)"),
                0
            );
            assert_eq!(
                count_query(&current, "MATCH (n:Own) WHERE n.k = 20 RETURN count(n)"),
                0
            );
            assert_eq!(
                inner.get_node_property(target, &PropertyKey::new("seen")),
                None,
                "rollback must discard the mutating reader's property delta"
            );
        }
    }
}
