//! Acceptance: counted work must not track the database size.
//!
//! Query work is measured through the public execution path.
//! An indexed point lookup that materialised its label's entire id
//! set cost 30.6 ms at a million labelled nodes and stayed invisible to 4,700
//! result-equality tests, because every fixture in this repository is small
//! enough that the linear term is microseconds.
//!
//! These tests compare *counted work* at two database sizes instead of timing
//! anything, so they are deterministic, unaffected by machine load, and run in
//! milliseconds on fixture-sized data.
//!
//! Note what is deliberately included: `label_scan_is_counted` is a positive
//! control. Without it, every budget here could pass because the counters never
//! increment at all — the same way a plan-shape walker with a silent wildcard
//! arm can pass without reaching the node it claims to check.

#![cfg(feature = "lpg")]

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_core::graph::WorkSnapshot;
use grafeo_engine::GrafeoDB;

/// Two sizes, four times apart: enough that any per-node term in a lookup shows
/// up as a difference, small enough that both fixtures build in milliseconds.
const SMALL: usize = 1_000;
const LARGE: usize = 4_000;

fn populated_db(nodes: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for chunk_start in (0..nodes).step_by(500) {
        let chunk_end = (chunk_start + 500).min(nodes);
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

/// Counted work performed by one query.
fn work_of(db: &GrafeoDB, query: &str) -> WorkSnapshot {
    let session = db.session();
    let before = grafeo_engine::database::testing::root_lpg_store(db).work_snapshot();
    session.execute(query).expect("query must succeed");
    grafeo_engine::database::testing::root_lpg_store(db)
        .work_snapshot()
        .since(before)
}

/// The work one query shape performs at both fixture sizes.
fn work_at_both_sizes(query: &str) -> (WorkSnapshot, WorkSnapshot) {
    (
        work_of(&populated_db(SMALL), query),
        work_of(&populated_db(LARGE), query),
    )
}

fn internal_id_work(nodes: usize) -> [WorkSnapshot; 4] {
    let db = populated_db(nodes);
    let session = db.session();
    let target = session
        .execute("MATCH (n:Node {id: 'n_1'}) RETURN id(n)")
        .expect("target lookup must succeed");
    let Value::Int64(target) = target.rows()[0][0] else {
        panic!("id(n) must be Int64");
    };

    let queries = [
        format!("MATCH (n) WHERE id(n) = {target} RETURN count(n)"),
        format!("MATCH (n) WHERE id(n) IN [{target}] RETURN count(n)"),
        format!("MATCH (n:Node) WHERE id(n) = {target} RETURN count(n)"),
        format!("MATCH (n:Node) WHERE id(n) IN [{target}] RETURN count(n)"),
    ];
    queries.map(|query| {
        let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
        let result = session
            .execute(&query)
            .expect("internal-id lookup must succeed");
        assert_eq!(result.rows()[0][0], Value::Int64(1));
        let work = grafeo_engine::database::testing::root_lpg_store(&db)
            .work_snapshot()
            .since(before);
        assert!(!work.scanned_any(), "internal-id lookup scanned: {work:?}");
        work
    })
}

const TRAILING_BATCH_ROWS: i64 = 16;

fn trailing_batch_fixture(nodes: usize) -> (GrafeoDB, HashMap<String, Value>) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for chunk_start in (0..nodes).step_by(500) {
        let chunk_end = (chunk_start + 500).min(nodes);
        let mut stmt = String::from("CREATE ");
        for (offset, i) in (chunk_start..chunk_end).enumerate() {
            if offset > 0 {
                stmt.push(',');
            }
            write!(stmt, "(:Node {{id: {i}}})").expect("string write");
        }
        session.execute(&stmt).expect("node fixture must load");
    }
    session
        .execute("CREATE INDEX gb_id FOR (n:Node) ON (n.id)")
        .expect("index must build");

    let rows = (0..TRAILING_BATCH_ROWS)
        .map(|i| {
            Value::Map(
                BTreeMap::from([
                    (
                        PropertyKey::new("s"),
                        Value::Int64(i % i64::try_from(nodes).unwrap()),
                    ),
                    (
                        PropertyKey::new("t"),
                        Value::Int64((i + 1) % i64::try_from(nodes).unwrap()),
                    ),
                    (
                        PropertyKey::new("p"),
                        Value::Map(
                            BTreeMap::from([(PropertyKey::new("w"), Value::Int64(i))]).into(),
                        ),
                    ),
                ])
                .into(),
            )
        })
        .collect::<Vec<_>>();
    let params = HashMap::from([("es".to_string(), Value::List(rows.into()))]);
    (db, params)
}

fn trailing_batch_work(
    nodes: usize,
    trailing_clause: &str,
    expected_result: Option<i64>,
) -> (GrafeoDB, WorkSnapshot) {
    let (db, params) = trailing_batch_fixture(nodes);
    let session = db.session();
    let query = format!(
        "UNWIND $es AS e MATCH (s:Node {{id:e.s}}), (t:Node {{id:e.t}}) \
         CREATE (s)-[r:REL]->(t) {trailing_clause}"
    );
    let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
    let result = session
        .execute_with_params(&query, params)
        .expect("batched trailing query must succeed");
    let work = grafeo_engine::database::testing::root_lpg_store(&db)
        .work_snapshot()
        .since(before);
    if let Some(expected) = expected_result {
        assert_eq!(result.rows()[0][0], Value::Int64(expected));
    }
    (db, work)
}

fn indexed_merge_fixture(
    nodes: usize,
    existing: &[i64],
    merge_values: &[i64],
) -> (GrafeoDB, HashMap<String, Value>) {
    let (db, _) = trailing_batch_fixture(nodes);
    let session = db.session();
    if !existing.is_empty() {
        let mut stmt = String::from("CREATE ");
        for (offset, value) in existing.iter().enumerate() {
            if offset > 0 {
                stmt.push(',');
            }
            write!(stmt, "(:Other {{n: {value}}})").expect("string write");
        }
        session.execute(&stmt).expect("merge hit fixture must load");
    }
    session
        .execute("CREATE INDEX other_n FOR (n:Other) ON (n.n)")
        .expect("merge property index must build");
    let rows = merge_values
        .iter()
        .map(|value| {
            Value::Map(
                BTreeMap::from([
                    (PropertyKey::new("s"), Value::Int64(*value)),
                    (
                        PropertyKey::new("t"),
                        Value::Int64((*value + 1) % i64::try_from(nodes).unwrap()),
                    ),
                    (
                        PropertyKey::new("p"),
                        Value::Map(
                            BTreeMap::from([(PropertyKey::new("w"), Value::Int64(*value))]).into(),
                        ),
                    ),
                ])
                .into(),
            )
        })
        .collect::<Vec<_>>();
    (
        db,
        HashMap::from([("es".to_string(), Value::List(rows.into()))]),
    )
}

fn indexed_merge_work(
    nodes: usize,
    existing: &[i64],
    merge_values: &[i64],
) -> (GrafeoDB, WorkSnapshot) {
    let (db, params) = indexed_merge_fixture(nodes, existing, merge_values);
    let session = db.session();
    let query = "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) \
                 CREATE (s)-[r:REL]->(t) MERGE (extra:Other {n:e.s}) RETURN count(extra)";
    let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
    let result = session
        .execute_with_params(query, params)
        .expect("indexed MERGE query must succeed");
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(i64::try_from(merge_values.len()).unwrap())
    );
    let work = grafeo_engine::database::testing::root_lpg_store(&db)
        .work_snapshot()
        .since(before);
    (db, work)
}

fn edge_property_count(db: &GrafeoDB, predicate: &str) -> i64 {
    let result = db
        .session()
        .execute(&format!(
            "MATCH ()-[r:REL]->() WHERE {predicate} RETURN count(r)"
        ))
        .expect("edge verification must succeed");
    match result.rows()[0][0] {
        Value::Int64(count) => count,
        ref value => panic!("edge count must be Int64, got {value:?}"),
    }
}

#[test]
fn labelled_point_lookup_scans_nothing_and_is_flat_in_node_count() {
    let (small, large) = work_at_both_sizes("MATCH (n:Node {id: 'n_1'}) RETURN n.id");

    println!("labelled point lookup: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert!(
        !small.scanned_any(),
        "a labelled point lookup must scan neither the label nor the node table, got {small:?}"
    );
    assert_eq!(
        small.label_scan_calls, 0,
        "the label's id set must not be built for a point lookup"
    );
    assert_eq!(
        small, large,
        "counted work grew with the database: {SMALL} nodes {small:?} against {LARGE} nodes {large:?}"
    );
}

#[test]
fn unlabelled_point_lookup_scans_nothing_and_is_flat_in_node_count() {
    let (small, large) = work_at_both_sizes("MATCH (n {id: 'n_1'}) RETURN n.id");

    println!("unlabelled point lookup: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert!(!small.scanned_any(), "got {small:?}");
    assert_eq!(small, large, "{small:?} against {large:?}");
}

#[test]
fn label_scan_is_counted() {
    // Positive control: a query that genuinely scans a label must be *seen* to
    // scan it, and must scale with the label. Without this, every budget above
    // would pass if the counters silently never incremented.
    let small = work_of(&populated_db(SMALL), "MATCH (n:Node) RETURN count(n)");
    let large = work_of(&populated_db(LARGE), "MATCH (n:Node) RETURN count(n)");

    println!("label scan: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert!(
        small.scanned_any(),
        "a whole-label scan must register as scanned work, got {small:?}"
    );
    let scanned = |w: WorkSnapshot| w.label_scan_ids + w.full_scan_ids;
    assert!(
        scanned(large) > scanned(small),
        "scanned work must grow with the label: {SMALL} nodes {small:?} against \
         {LARGE} nodes {large:?}"
    );
}

#[test]
fn write_path_lookup_scans_nothing_and_is_flat_in_node_count() {
    // A writing transaction used to abandon the property index entirely, because
    // the committed index cannot see its buffered writes, so every SET ran a full
    // scan: 24.6 ms at 100,000 nodes and 440 ms labelled at a million. The fixed
    // 300-operation benchmark rows cost 898.6 ms at sf1 against 22.4 ms for a
    // competitor that stayed flat.
    let work = |nodes: usize| {
        let db = populated_db(nodes);
        let mut session = db.session();
        let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
        session.begin_transaction().expect("begin");
        session
            .execute("MATCH (n:Node {id: 'n_1'}) SET n.v = 1")
            .expect("indexed set");
        session.commit().expect("commit");
        grafeo_engine::database::testing::root_lpg_store(&db)
            .work_snapshot()
            .since(before)
    };

    let small = work(SMALL);
    let large = work(LARGE);
    println!("write-path lookup: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert!(
        !small.scanned_any(),
        "an indexed SET must not scan: got {small:?}"
    );
    assert_eq!(
        small, large,
        "write-path work grew with the database: {SMALL} nodes {small:?} against \
         {LARGE} nodes {large:?}"
    );
}

#[test]
fn counting_costs_a_negligible_fraction_of_the_scan_it_instruments() {
    // The counters are always on, so their cost has to be unmeasurable rather
    // than merely small. Each counted call performs two relaxed adds; this
    // measures them against the scan that carries them.
    //
    // This bound is why property-index probes are not counted. The same two adds
    // measured 12.3 ns against a 285 ns index probe — 4.3%, paid again per value
    // on an `IN`-list path. Against a scan that collects and sorts a thousand
    // ids they are a rounding error, which is the whole argument for leaving the
    // scan counters always on.
    const ITERATIONS: u32 = 2_000;
    const TRIALS: usize = 15;
    const MAX_FRACTION_OF_A_SCAN: f64 = 0.01;
    const MAX_ABSOLUTE_NANOS: f64 = 30.0;

    // Seconds per iteration of `body`, from the fastest of several trials.
    // Preemption and machine load only ever add time, so the fastest trial is
    // the cost estimate; a genuinely slow add is slow in every trial.
    fn fastest_per_iteration(mut body: impl FnMut()) -> f64 {
        (0..TRIALS)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    body();
                }
                start.elapsed().as_secs_f64() / f64::from(ITERATIONS)
            })
            .fold(f64::INFINITY, f64::min)
    }

    let db = populated_db(SMALL);
    let store = grafeo_engine::database::testing::root_lpg_store(&db);

    for _ in 0..50 {
        assert_eq!(store.nodes_by_label("Node").len(), SMALL);
    }
    let per_scan = fastest_per_iteration(|| {
        assert_eq!(store.nodes_by_label("Node").len(), SMALL);
    });

    let counter = AtomicU64::new(0);
    let per_count = fastest_per_iteration(|| {
        counter.fetch_add(1, Ordering::Relaxed);
        counter.fetch_add(7, Ordering::Relaxed);
    });
    assert!(counter.load(Ordering::Relaxed) > 0, "counter must be used");

    let fraction = per_count / per_scan;
    println!(
        "scan of {SMALL} ids {:.0} ns, two relaxed adds {:.1} ns, overhead {:.4}% of the scan",
        per_scan * 1e9,
        per_count * 1e9,
        fraction * 100.0
    );
    assert!(
        per_count * 1e9 < MAX_ABSOLUTE_NANOS,
        "two relaxed adds cost {:.1} ns, which is not the uncontended add this design assumes",
        per_count * 1e9
    );
    assert!(
        fraction < MAX_FRACTION_OF_A_SCAN,
        "work counting costs {:.3}% of a {SMALL}-id scan ({:.1} ns against {:.0} ns): \
         gate the counters behind a feature",
        fraction * 100.0,
        per_count * 1e9,
        per_scan * 1e9
    );
}

#[test]
fn internal_id_lookup_work_is_flat_for_equality_and_in_list() {
    let small = internal_id_work(SMALL);
    let large = internal_id_work(LARGE);
    println!("internal-id work: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert_eq!(
        small, large,
        "internal-id work grew with the database: {small:?} against {large:?}"
    );
}

// Two indexed endpoints per row. Matching postings stay fixed; the AVL
// routing boundary may grow logarithmically as unrelated value keys are added.
fn assert_bounded_endpoint_work(small: WorkSnapshot, large: WorkSnapshot) {
    let probes = 2 * TRAILING_BATCH_ROWS as u64;
    for (nodes, work) in [(SMALL, small), (LARGE, large)] {
        assert!(!work.scanned_any(), "indexed endpoints scanned: {work:?}");
        assert_eq!(work.property_index_rebuild_rows, 0);
        assert_eq!(work.property_index_posting_ids, probes);
        assert_eq!(work.property_index_posting_intervals, probes);

        // The fixture stores one Integer routing key per id. Integer equality
        // hash-probes its native bucket and seeks two additional lanes (Float
        // and StringAsInt), both empty here. Each empty seek follows at most
        // one AVL root-to-leaf path, regardless of insertion/tree shape.
        // Minimum nodes at height h: n(0)=0, n(1)=1,
        // n(h)=1+n(h-1)+n(h-2). Invert that recurrence for the exact maximum
        // permitted height, rather than comparing two independent tree shapes.
        let (mut previous, mut minimum, mut max_height) = (0, 1, 0_u64);
        while minimum <= nodes {
            max_height += 1;
            (previous, minimum) = (minimum, 1 + previous + minimum);
        }
        // Each 500-node CREATE contributes at most one value-root epoch;
        // index creation and these edge writes add no id-property events.
        // visit_at also counts partition_point's <= ceil(log2 E)+1 probes.
        let epochs = nodes.div_ceil(500);
        let epoch_probes = u64::from(epochs.next_power_of_two().ilog2()) + 1;
        let additional_lanes = 2;
        let route_bound = probes * additional_lanes * (max_height + epoch_probes);
        assert!(
            work.property_index_route_keys <= route_bound,
            "{nodes} nodes exceeded AVL/epoch routing bound {route_bound}: {work:?}"
        );
    }
}

#[test]
fn trailing_edge_map_set_work_is_flat_in_node_count() {
    let (small_db, small) = trailing_batch_work(SMALL, "SET r = e.p", None);
    let (large_db, large) = trailing_batch_work(LARGE, "SET r = e.p", None);
    assert_eq!(
        edge_property_count(&small_db, "r.w >= 0"),
        TRAILING_BATCH_ROWS
    );
    assert_eq!(
        edge_property_count(&large_db, "r.w >= 0"),
        TRAILING_BATCH_ROWS
    );
    println!("trailing edge map SET work: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_edge_scalar_set_work_is_flat_in_node_count() {
    let (small_db, small) = trailing_batch_work(SMALL, "SET r.k = e.s", None);
    let (large_db, large) = trailing_batch_work(LARGE, "SET r.k = e.s", None);
    assert_eq!(
        edge_property_count(&small_db, "r.k >= 0"),
        TRAILING_BATCH_ROWS
    );
    assert_eq!(
        edge_property_count(&large_db, "r.k >= 0"),
        TRAILING_BATCH_ROWS
    );
    println!("trailing edge scalar SET work: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_aggregate_work_is_flat_in_node_count() {
    let (_, small) = trailing_batch_work(SMALL, "RETURN count(r) AS n", Some(TRAILING_BATCH_ROWS));
    let (_, large) = trailing_batch_work(LARGE, "RETURN count(r) AS n", Some(TRAILING_BATCH_ROWS));
    println!("trailing aggregate work: {SMALL} nodes {small:?}, {LARGE} nodes {large:?}");
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_node_property_set_keeps_indexed_endpoint_work_flat() {
    let (small_db, small) = trailing_batch_work(
        SMALL,
        "SET s.unrelated = e.p RETURN count(r)",
        Some(TRAILING_BATCH_ROWS),
    );
    let (large_db, large) = trailing_batch_work(
        LARGE,
        "SET s.unrelated = e.p RETURN count(r)",
        Some(TRAILING_BATCH_ROWS),
    );
    for db in [&small_db, &large_db] {
        let result = db
            .session()
            .execute("MATCH (n:Node) WHERE n.unrelated IS NOT NULL RETURN count(n)")
            .expect("node property verification must succeed");
        assert_eq!(result.rows()[0][0], Value::Int64(TRAILING_BATCH_ROWS));
    }
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_node_label_set_keeps_indexed_endpoint_work_flat() {
    let (small_db, small) = trailing_batch_work(
        SMALL,
        "SET s:TrailingLabel RETURN count(r)",
        Some(TRAILING_BATCH_ROWS),
    );
    let (large_db, large) = trailing_batch_work(
        LARGE,
        "SET s:TrailingLabel RETURN count(r)",
        Some(TRAILING_BATCH_ROWS),
    );
    for db in [&small_db, &large_db] {
        let result = db
            .session()
            .execute("MATCH (n:TrailingLabel) RETURN count(n)")
            .expect("label verification must succeed");
        assert_eq!(result.rows()[0][0], Value::Int64(TRAILING_BATCH_ROWS));
    }
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_node_label_add_remove_keeps_indexed_endpoint_work_flat() {
    let (small_db, small) = trailing_batch_work(
        SMALL,
        "SET s:TrailingLabel REMOVE s:TrailingLabel RETURN count(r)",
        Some(TRAILING_BATCH_ROWS),
    );
    let (large_db, large) = trailing_batch_work(
        LARGE,
        "SET s:TrailingLabel REMOVE s:TrailingLabel RETURN count(r)",
        Some(TRAILING_BATCH_ROWS),
    );
    for db in [&small_db, &large_db] {
        let result = db
            .session()
            .execute("MATCH (n:TrailingLabel) RETURN count(n)")
            .expect("removed label verification must succeed");
        assert_eq!(result.rows()[0][0], Value::Int64(0));
        assert_eq!(edge_property_count(db, "true"), TRAILING_BATCH_ROWS);
    }
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_disjoint_node_create_keeps_indexed_endpoint_work_flat() {
    let (small_db, small) = trailing_batch_work(
        SMALL,
        "CREATE (extra:Other {n:e.s}) RETURN count(extra)",
        Some(TRAILING_BATCH_ROWS),
    );
    let (large_db, large) = trailing_batch_work(
        LARGE,
        "CREATE (extra:Other {n:e.s}) RETURN count(extra)",
        Some(TRAILING_BATCH_ROWS),
    );
    for db in [&small_db, &large_db] {
        let result = db
            .session()
            .execute("MATCH (n:Other) RETURN count(n)")
            .expect("disjoint created-node verification must succeed");
        assert_eq!(result.rows()[0][0], Value::Int64(TRAILING_BATCH_ROWS));
        assert_eq!(edge_property_count(db, "true"), TRAILING_BATCH_ROWS);
    }
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_disjoint_node_merge_keeps_endpoint_lookups_indexed() {
    let (small_db, small) = trailing_batch_work(
        SMALL,
        "MERGE (extra:Other {n:e.s}) RETURN count(extra)",
        Some(TRAILING_BATCH_ROWS),
    );
    let (large_db, large) = trailing_batch_work(
        LARGE,
        "MERGE (extra:Other {n:e.s}) RETURN count(extra)",
        Some(TRAILING_BATCH_ROWS),
    );
    for (db, work) in [(&small_db, small), (&large_db, large)] {
        let result = db
            .session()
            .execute("MATCH (n:Other) RETURN count(n)")
            .expect("disjoint merged-node verification must succeed");
        assert_eq!(result.rows()[0][0], Value::Int64(TRAILING_BATCH_ROWS));
        assert_eq!(edge_property_count(db, "true"), TRAILING_BATCH_ROWS);
        assert_eq!(
            work.label_scan_calls, 0,
            "MERGE used a label scan: {work:?}"
        );
        assert_eq!(work.label_scan_ids, 0, "MERGE returned label ids: {work:?}");
        assert_eq!(work.full_scan_calls, TRAILING_BATCH_ROWS as u64);
        assert_eq!(work.property_index_rebuild_rows, 0);
        assert_eq!(
            work.property_index_posting_ids,
            2 * TRAILING_BATCH_ROWS as u64
        );
        assert_eq!(
            work.property_index_posting_intervals,
            2 * TRAILING_BATCH_ROWS as u64
        );
    }
}

#[test]
fn trailing_edge_delete_keeps_indexed_endpoint_work_flat() {
    let (small_db, small) = trailing_batch_work(SMALL, "DELETE r", None);
    let (large_db, large) = trailing_batch_work(LARGE, "DELETE r", None);
    assert_eq!(edge_property_count(&small_db, "true"), 0);
    assert_eq!(edge_property_count(&large_db, "true"), 0);
    assert_bounded_endpoint_work(small, large);
}

#[test]
fn trailing_relationship_merge_admission_keeps_indexed_endpoint_work_flat() {
    let (small_db, small) = trailing_batch_work(
        SMALL,
        "MERGE (s)-[m:REL]->(t) RETURN count(m)",
        Some(TRAILING_BATCH_ROWS),
    );
    let (large_db, large) = trailing_batch_work(
        LARGE,
        "MERGE (s)-[m:REL]->(t) RETURN count(m)",
        Some(TRAILING_BATCH_ROWS),
    );
    assert_eq!(edge_property_count(&small_db, "true"), TRAILING_BATCH_ROWS);
    assert_eq!(edge_property_count(&large_db, "true"), TRAILING_BATCH_ROWS);
    assert_bounded_endpoint_work(small, large);
}

fn assert_indexed_merge_work(work: WorkSnapshot, merge_postings: u64) {
    assert_eq!(
        work.label_scan_calls, 0,
        "indexed MERGE used a label scan: {work:?}"
    );
    assert_eq!(
        work.label_scan_ids, 0,
        "indexed MERGE returned label ids: {work:?}"
    );
    assert_eq!(
        work.full_scan_calls, 0,
        "indexed MERGE used a full scan: {work:?}"
    );
    assert_eq!(work.full_scan_ids, 0);
    assert_eq!(work.property_index_rebuild_rows, 0);
    assert_eq!(work.property_index_posting_ids, 32 + merge_postings);
    assert_eq!(work.property_index_posting_intervals, 32 + merge_postings);
}

#[test]
fn indexed_merge_hits_stay_flat_at_both_fixture_sizes() {
    let values: Vec<_> = (0..TRAILING_BATCH_ROWS).collect();
    let (small_db, small) = indexed_merge_work(SMALL, &values, &values);
    let (large_db, large) = indexed_merge_work(LARGE, &values, &values);
    for db in [&small_db, &large_db] {
        assert_eq!(
            db.session()
                .execute("MATCH (n:Other) RETURN count(n)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(TRAILING_BATCH_ROWS)
        );
        assert_eq!(edge_property_count(db, "true"), TRAILING_BATCH_ROWS);
    }
    assert_indexed_merge_work(small, TRAILING_BATCH_ROWS as u64);
    assert_indexed_merge_work(large, TRAILING_BATCH_ROWS as u64);
}

#[test]
fn indexed_merge_misses_create_rows_without_scanning() {
    let values: Vec<_> = (0..TRAILING_BATCH_ROWS).collect();
    let existing: Vec<_> = (TRAILING_BATCH_ROWS..(2 * TRAILING_BATCH_ROWS)).collect();
    let (small_db, small) = indexed_merge_work(SMALL, &existing, &values);
    let (large_db, large) = indexed_merge_work(LARGE, &existing, &values);
    for db in [&small_db, &large_db] {
        assert_eq!(
            db.session()
                .execute("MATCH (n:Other) RETURN count(n)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(2 * TRAILING_BATCH_ROWS)
        );
        assert_eq!(edge_property_count(db, "true"), TRAILING_BATCH_ROWS);
    }
    assert_indexed_merge_work(small, 0);
    assert_indexed_merge_work(large, 0);
}

#[test]
fn indexed_merge_reads_its_own_buffered_nodes() {
    let values: Vec<_> = (0..TRAILING_BATCH_ROWS / 2)
        .flat_map(|value| [value, value])
        .collect();
    let (small_db, small) = indexed_merge_work(SMALL, &[], &values);
    let (large_db, large) = indexed_merge_work(LARGE, &[], &values);
    for db in [&small_db, &large_db] {
        assert_eq!(
            db.session()
                .execute("MATCH (n:Other) RETURN count(n)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(TRAILING_BATCH_ROWS / 2)
        );
        assert_eq!(edge_property_count(db, "true"), TRAILING_BATCH_ROWS);
    }
    assert_indexed_merge_work(small, 0);
    assert_indexed_merge_work(large, 0);
}

#[test]
fn aliased_indexed_node_property_write_preserves_same_node_reference() {
    let db = populated_db(SMALL);
    let session = db.session();
    let before = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
    let result = session
        .execute("UNWIND ['n_1'] AS key MATCH (a:Node {id: key}), (b:Node {id: key}) SET a.id = 'changed' RETURN id(a), id(b)")
        .expect("aliased indexed-node write must succeed");
    let work = grafeo_engine::database::testing::root_lpg_store(&db)
        .work_snapshot()
        .since(before);
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], result.rows()[0][1]);
    assert!(
        work.scanned_any(),
        "aliased indexed-key mutation must take the guarded fallback path: {work:?}"
    );

    let changed = db
        .session()
        .execute("MATCH (n:Node {id: 'changed'}) RETURN count(n)")
        .expect("changed indexed property must remain queryable");
    assert_eq!(changed.rows()[0][0], Value::Int64(1));
}

mod support;

fn distinct_budget_fixture(nodes: usize) -> GrafeoDB {
    let db = support::adversarial_path_graph();
    let session = db.session();
    // Same label as the reachable nodes, but no id property: any accidental
    // label scan grows while the anchored index posting remains unchanged.
    for first in (support::node_ids().len()..nodes).step_by(500) {
        let count = (nodes - first).min(500);
        let statement = format!("CREATE {}", vec!["(:Node)"; count].join(","));
        session
            .execute(&statement)
            .expect("unreachable padding nodes");
    }
    db
}

fn distinct_expand_work(db: &GrafeoDB, depth: usize, edge_type: &str) -> (WorkSnapshot, usize) {
    let session = db.session();
    let pattern =
        format!("MATCH (source:Node {{id: 's'}})-[:{edge_type}*1..{depth}]->(target:Node)");
    let expected: Vec<_> = support::reachable_within("s", 1, depth, edge_type)
        .into_iter()
        .collect();
    let before = grafeo_engine::database::testing::root_lpg_store(db).work_snapshot();
    let result = session
        .execute(&format!("{pattern} RETURN DISTINCT target.id"))
        .unwrap();
    let work = grafeo_engine::database::testing::root_lpg_store(db)
        .work_snapshot()
        .since(before);
    assert_eq!(support::sorted_ids(&result), expected);
    assert_eq!(
        result.row_count(),
        expected.len(),
        "one row per reachable target"
    );

    let before = grafeo_engine::database::testing::root_lpg_store(db).work_snapshot();
    let result = session
        .execute(&format!("PROFILE {pattern} RETURN DISTINCT target.id"))
        .unwrap();
    let profile_work = grafeo_engine::database::testing::root_lpg_store(db)
        .work_snapshot()
        .since(before);
    assert_eq!(
        work, profile_work,
        "PROFILE must preserve the store-work budget"
    );
    let Value::String(profile) = &result.rows()[0][0] else {
        panic!("PROFILE must return text");
    };
    let expand = profile
        .lines()
        .find(|line| line.trim_start().starts_with("VariableLengthExpand ("))
        .expect("PROFILE must observe the real variable-length expand");
    let rows = expand
        .split("  rows=")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<usize>()
        .unwrap();
    assert_eq!(
        rows,
        expected.len(),
        "expand must prune walks before DISTINCT: {profile}"
    );
    assert!(
        !work.scanned_any(),
        "indexed anchor must not scan padding nodes: {work:?}"
    );
    assert_eq!(work.property_index_posting_ids, 1);
    assert_eq!(work.property_index_rebuild_rows, 0);

    // Independent walk oracle counts parallel physical edges separately. A
    // non-DISTINCT consumer must preserve this multiplicity on the same graph.
    let walks = session
        .execute(&format!("{pattern} RETURN count(*)"))
        .unwrap();
    let expected_walks = support::walk_count("s", 1, depth, edge_type);
    assert_eq!(
        walks.rows()[0][0],
        Value::Int64(i64::try_from(expected_walks).unwrap())
    );
    assert!(
        expected_walks > expected.len(),
        "positive multiplicity control must distinguish walks from targets"
    );
    (work, rows)
}

#[test]
fn variable_distinct_work_is_flat_in_database_size_and_walk_depth() {
    let small = distinct_budget_fixture(SMALL);
    let large = distinct_budget_fixture(LARGE);
    for edge_type in ["REL", "OTHER"] {
        let mut shallow_rows = None;
        for depth in [4, 32] {
            let small_work = distinct_expand_work(&small, depth, edge_type);
            let large_work = distinct_expand_work(&large, depth, edge_type);
            assert_eq!(
                small_work, large_work,
                "DISTINCT work must not track unrelated database size: {edge_type} depth={depth}"
            );
            if let Some(rows) = shallow_rows {
                assert_eq!(
                    small_work.1, rows,
                    "deeper cyclic walks must not grow expand output"
                );
            } else {
                shallow_rows = Some(small_work.1);
            }
        }
    }
}
