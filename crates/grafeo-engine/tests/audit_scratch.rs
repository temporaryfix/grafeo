//! Temporary audit probes — verifying suspected correctness bugs.
#![cfg(feature = "lpg")]

use grafeo_engine::GrafeoDB;

/// Probe 1: join reorder must not drop WHERE predicates on cross-MATCH joins.
#[test]
fn join_reorder_keeps_filters() {
    let db = GrafeoDB::new_in_memory();
    let mut s = db.session();
    s.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    s.execute("CREATE (:Person {name: 'Bob'})").unwrap();
    s.execute("CREATE (:Person {name: 'Cyd'})").unwrap();
    s.execute("CREATE (:City {name: 'Rome'})").unwrap();
    s.execute("CREATE (:City {name: 'Oslo'})").unwrap();

    let r = s
        .execute("MATCH (a:Person), (b:City) WHERE a.name = 'Ann' AND b.name = 'Rome' RETURN a.name, b.name")
        .unwrap();
    assert_eq!(r.row_count(), 1, "expected exactly 1 row (Ann, Rome)");
}

/// Probe 1b: three-relation cyclic-ish join with filters.
#[test]
fn join_reorder_three_relations_keeps_filters() {
    let db = GrafeoDB::new_in_memory();
    let mut s = db.session();
    for i in 0..5 {
        s.execute(&format!("CREATE (:A {{x: {i}}})")).unwrap();
        s.execute(&format!("CREATE (:B {{x: {i}}})")).unwrap();
        s.execute(&format!("CREATE (:C {{x: {i}}})")).unwrap();
    }
    let r = s
        .execute(
            "MATCH (a:A), (b:B), (c:C) WHERE a.x = 1 AND b.x = 2 AND c.x = 3 RETURN a.x, b.x, c.x",
        )
        .unwrap();
    assert_eq!(r.row_count(), 1, "expected exactly 1 row (1,2,3)");
}

/// Probe 2: an uncommitted DELETE in one session must not be visible to another.
///
/// KNOWN BUG (confirmed 2026-06-09): `delete_*_transactional` sets
/// `deleted_epoch` to the current epoch immediately (not PENDING) and marks
/// adjacency tombstones eagerly, so the delete is globally visible before
/// commit. Remove the #[ignore] once deletes are isolated.
#[test]
fn uncommitted_delete_invisible_to_others() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p")
        .unwrap();

    // Other session: Ann must still exist (delete not committed).
    let mut reader = db.session();
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name")
        .unwrap();
    let visible_during = r.row_count();

    writer.rollback().unwrap();

    let r2 = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name")
        .unwrap();
    let visible_after_rollback = r2.row_count();

    assert_eq!(
        visible_after_rollback, 1,
        "node must exist after rollback of delete"
    );
    assert_eq!(
        visible_during, 1,
        "uncommitted delete must not be visible to other sessions (dirty delete)"
    );
}

/// Probe 3: an uncommitted property SET must not be visible to another session.
///
/// KNOWN BUG (confirmed 2026-06-09): in non-temporal builds properties are a
/// single global column (no MVCC); transactional SET mutates it in place and
/// relies on the undo log for rollback, so other sessions dirty-read the
/// uncommitted value. Remove the #[ignore] once property writes are isolated.
#[test]
fn uncommitted_property_write_invisible_to_others() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .unwrap();

    let reader = db.session();
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    // Assert on the actual cell value, not the Debug of the whole QueryResult
    // (the previous version matched "30" inside the execution_time_ms field —
    // a false pass that masked the live dirty-read bug).
    let age = r.rows()[0][0].clone();
    writer.rollback().unwrap();
    assert_eq!(
        age,
        grafeo_common::types::Value::Int64(30),
        "uncommitted SET must not be visible to other sessions (dirty read)"
    );
}

/// Probe 4: rollback of a created node must fully clean up (no phantom label matches).
#[test]
fn rolled_back_create_leaves_no_phantoms() {
    let db = GrafeoDB::new_in_memory();
    let mut s = db.session();
    s.begin_transaction().unwrap();
    s.execute("CREATE (:Ghost {name: 'Boo'})").unwrap();
    s.rollback().unwrap();

    let r = s.execute("MATCH (g:Ghost) RETURN g").unwrap();
    assert_eq!(
        r.row_count(),
        0,
        "rolled-back node must not match label scan"
    );

    let r2 = s.execute("MATCH (g:Ghost {name: 'Boo'}) RETURN g").unwrap();
    assert_eq!(
        r2.row_count(),
        0,
        "rolled-back node must not match property scan"
    );
}

/// Probe 5: commit conflict must not leave the transaction active / leak versions.
#[test]
fn failed_commit_does_not_pin_gc() {
    let db = GrafeoDB::new_in_memory();

    // Simulate: two sessions write the same node -> second record_write conflicts
    // at write time (first-writer-wins), which surfaces as an execute error.
    let mut s1 = db.session();
    s1.execute("CREATE (:Acct {id: 1, bal: 100})").unwrap();

    let mut s2 = db.session();

    s1.begin_transaction().unwrap();
    s2.begin_transaction().unwrap();

    s1.execute("MATCH (a:Acct {id: 1}) SET a.bal = 50").unwrap();
    // Second writer should fail (write-write conflict)
    let r2 = s2.execute("MATCH (a:Acct {id: 1}) SET a.bal = 60");
    let _ = r2; // may fail at execute or commit; both acceptable

    let c2 = s2.commit();
    let c1 = s1.commit();
    // At least one of them must succeed
    assert!(
        c1.is_ok() || c2.is_ok(),
        "both commits failed: {c1:?} {c2:?}"
    );

    // Whatever happened, a fresh read must see exactly one consistent value
    let mut s3 = db.session();
    let r = s3.execute("MATCH (a:Acct {id: 1}) RETURN a.bal").unwrap();
    assert_eq!(r.row_count(), 1);
}

/// Probe 6: inspect plan shape for cross-MATCH joins.
#[test]
fn explain_cross_match() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    s.execute("CREATE (:City {name: 'Rome'})").unwrap();
    let r = s
        .execute("EXPLAIN MATCH (a:Person), (b:City) WHERE a.name = 'Ann' AND b.name = 'Rome' RETURN a.name, b.name")
        .unwrap();
    println!("PLAN: {r:#?}");
}

/// Probe 7: variable-length expand on a cyclic graph must not blow up.
#[test]
fn var_length_expand_cycle_walk_blowup() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE (a:N {id: 1})-[:R]->(b:N {id: 2})")
        .unwrap();
    s.execute("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (b)-[:R]->(a)")
        .unwrap();

    let t = std::time::Instant::now();
    let r = s
        .execute("MATCH (a:N {id: 1})-[:R*1..20]->(b) RETURN count(b) AS c")
        .unwrap();
    println!("cycle expand took {:?}, result {:?}", t.elapsed(), r);
}

/// Probe 7b: branching cyclic graph — walk count explodes (Walk mode, no dedup).
///
/// KNOWN BUG (confirmed 2026-06-09): a 2-node graph with 2 parallel edges in
/// each direction and `[:R*1..24]` runs for minutes (2^24 walks enumerated by
/// the BFS in `VariableLengthExpandOperator::process_input_row`, which has no
/// visited-set and materializes all results). Unbounded `[*]` maps to
/// `min_hops + 100`, which is astronomically worse. Remove the #[ignore]
/// once walk enumeration is bounded (trail semantics / dedup / streaming).
#[test]
#[ignore = "known bug: exponential walk enumeration on cyclic graphs (hangs)"]
fn var_length_expand_branching_cycle() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE (a:M {id: 1})").unwrap();
    s.execute("CREATE (b:M {id: 2})").unwrap();
    // two parallel edges each way -> branching factor 2 at every step
    s.execute("MATCH (a:M {id:1}), (b:M {id:2}) CREATE (a)-[:R]->(b), (a)-[:R]->(b), (b)-[:R]->(a), (b)-[:R]->(a)").unwrap();

    let t = std::time::Instant::now();
    let r = s
        .execute("MATCH (a:M {id: 1})-[:R*1..24]->(b) RETURN count(b) AS c")
        .unwrap();
    println!(
        "branching cycle expand took {:?}, result {:?}",
        t.elapsed(),
        r.rows()
    );
}
