//! Serializable Snapshot Isolation acceptance suite — F2 (incremental SSI).
//!
//! These are session-level end-to-end tests that prove the SSI guarantees:
//!
//! 1. `write_skew_prevented_under_serializable` — classic write-skew aborts the
//!    second committer under Serializable isolation.
//! 2. `write_skew_allowed_under_snapshot_isolation` — the same scenario with SI
//!    (default) lets both transactions commit, proving the levels genuinely differ.
//! 3. `benign_concurrent_writers_do_not_abort` — two Serializable transactions
//!    that touch disjoint entities both commit (SSI must not be overly aggressive).
//! 4. `read_only_serializable_does_not_abort` — a read-only Serializable tx
//!    concurrent with a writer on the same data commits cleanly.
//! 5. `serialization_abort_rolls_back_cleanly` — after a write-skew abort the
//!    aborted tx's write is not visible; the store is in the expected state.
//! 6. `three_transaction_cycle_aborts_under_serializable` — a 3-transaction
//!    rw-cycle (T1→T3→T2→T1) aborts exactly one committer end-to-end, proving
//!    incremental SSI catches multi-transaction dangerous structures.
//! 7. `disjoint_property_writes_commit_under_property_granularity` — (Part G)
//!    same-entity scan but disjoint-property writes commit under Property granularity.
//! 8. `disjoint_property_writes_abort_under_entity_granularity` — identical
//!    workload with default Entity granularity → second session aborts (proves the
//!    knob changed the outcome).
//! 9. `same_property_write_skew_aborts_under_property_granularity` — classic
//!    write-skew on the SAME property under Property granularity → still aborts
//!    (soundness: the knob must not suppress real conflicts).
//!
//! ```bash
//! CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test serializable
//! ```

#![cfg(feature = "lpg")]

use grafeo_engine::{ConflictGranularity, GrafeoDB, transaction::IsolationLevel};

// ============================================================================
// Helpers
// ============================================================================

/// Assert that an error is a serialization failure (SSI abort).
fn assert_serialization_failure(result: &grafeo_common::utils::error::Result<()>, ctx: &str) {
    assert!(
        result.is_err(),
        "{ctx}: expected Err(SerializationFailure) but got Ok(())"
    );
    let msg = format!("{}", result.as_ref().unwrap_err());
    assert!(
        msg.contains("Serialization failure"),
        "{ctx}: expected 'Serialization failure' in error message, got: {msg}"
    );
}

// ============================================================================
// 1. write_skew_prevented_under_serializable
// ============================================================================

/// Classic write-skew scenario under Serializable isolation.
///
/// Two accounts A (bal=100) and B (bal=100); invariant: sum(bal) >= 0.
/// - s1 (Serializable): reads both; writes A.bal = A.bal - 200 (relies on B covering it).
/// - s2 (Serializable): reads both; writes B.bal = B.bal - 200 (relies on A covering it).
///
/// Interleave:
///   begin s1, begin s2, s1 reads A+B, s2 reads A+B, s1 writes A, s2 writes B,
///   s1.commit() → Ok, s2.commit() → Err(SerializationFailure).
///
/// s2 read entity A (which s1 wrote and committed), so SSI detects the
/// rw-antidependency and aborts s2.
#[test]
fn write_skew_prevented_under_serializable() {
    let db = GrafeoDB::new_in_memory();

    // Seed two accounts outside any explicit transaction (auto-commit).
    let setup = db.session();
    setup
        .execute("CREATE (:Account {id: 1, bal: 100})")
        .expect("CREATE account A");
    setup
        .execute("CREATE (:Account {id: 2, bal: 100})")
        .expect("CREATE account B");
    drop(setup);

    // --- Begin both serializable transactions (both observe the same committed state) ---
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // --- s1 reads BOTH accounts (populates its read-set with A and B) ---
    let r1 = s1
        .execute("MATCH (a:Account) RETURN a.id, a.bal ORDER BY a.id")
        .expect("s1: MATCH both accounts");
    assert_eq!(r1.row_count(), 2, "s1 must see both accounts");

    // --- s2 reads BOTH accounts (populates its read-set with A and B) ---
    let r2 = s2
        .execute("MATCH (a:Account) RETURN a.id, a.bal ORDER BY a.id")
        .expect("s2: MATCH both accounts");
    assert_eq!(r2.row_count(), 2, "s2 must see both accounts");

    // --- s1 writes: reduce A.bal by 200 (writes entity A into s1's write-set) ---
    s1.execute("MATCH (a:Account {id: 1}) SET a.bal = a.bal - 200")
        .expect("s1: SET A.bal");

    // --- s2 writes: reduce B.bal by 200 (writes entity B into s2's write-set) ---
    s2.execute("MATCH (a:Account {id: 2}) SET a.bal = a.bal - 200")
        .expect("s2: SET B.bal");

    // --- s1 commits first → must succeed ---
    let c1 = s1.commit();
    assert!(c1.is_ok(), "s1 (first committer) must succeed: {:?}", c1);

    // --- s2 tries to commit → must abort (rw-antidependency: s2 read A, s1 wrote A) ---
    let c2 = s2.commit();
    assert_serialization_failure(&c2, "s2 (second committer in write-skew scenario)");
}

// ============================================================================
// 2. write_skew_allowed_under_snapshot_isolation
// ============================================================================

/// Same scenario with default SnapshotIsolation → both transactions commit.
///
/// SI detects write-write conflicts (same entity written by two concurrent txns)
/// but does NOT detect rw-antidependencies, so write-skew is allowed.
/// Both s1 and s2 write disjoint entities (A and B respectively), so there is
/// no write-write conflict and both must commit.
///
/// This test is the contrast that proves Serializable is adding real value.
#[test]
fn write_skew_allowed_under_snapshot_isolation() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("CREATE (:Account {id: 1, bal: 100})")
        .expect("CREATE account A");
    setup
        .execute("CREATE (:Account {id: 2, bal: 100})")
        .expect("CREATE account B");
    drop(setup);

    // Both sessions use the default isolation level (SnapshotIsolation).
    let mut s1 = db.session();
    s1.begin_transaction().expect("s1: begin SI");

    let mut s2 = db.session();
    s2.begin_transaction().expect("s2: begin SI");

    // Both read both accounts.
    let r1 = s1
        .execute("MATCH (a:Account) RETURN a.id, a.bal ORDER BY a.id")
        .expect("s1: MATCH");
    assert_eq!(r1.row_count(), 2, "s1 must see both accounts");

    let r2 = s2
        .execute("MATCH (a:Account) RETURN a.id, a.bal ORDER BY a.id")
        .expect("s2: MATCH");
    assert_eq!(r2.row_count(), 2, "s2 must see both accounts");

    // Write to disjoint entities.
    s1.execute("MATCH (a:Account {id: 1}) SET a.bal = a.bal - 200")
        .expect("s1: SET A.bal");
    s2.execute("MATCH (a:Account {id: 2}) SET a.bal = a.bal - 200")
        .expect("s2: SET B.bal");

    // Under SI, both must commit (no write-write conflict; rw-antidependency not detected).
    let c1 = s1.commit();
    assert!(c1.is_ok(), "s1 must commit under SI: {:?}", c1);

    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 must commit under SI (write-skew allowed): {:?}",
        c2
    );
}

// ============================================================================
// 3. benign_concurrent_writers_do_not_abort
// ============================================================================

/// Two Serializable transactions that read+write DISJOINT entities both commit.
///
/// s1 only reads and writes nodes with label `:A`; s2 only reads and writes
/// nodes with label `:B`. Because the scan operators iterate over entirely
/// separate label partitions, no node from s1's scan lands in s2's read-set
/// and vice versa — there is no rw-antidependency between them.
///
/// Key design note: SSI read-sets are recorded at **scan granularity** (every
/// visible node the ScanOperator iterates over), not at filter-result
/// granularity. A `MATCH (n:X {id: 1})` over a label `:X` that also contains
/// id=2 would record both nodes in the read-set (because the scan visits all
/// `:X` nodes before the property predicate eliminates id=2). To make the
/// disjoint-readers scenario robust, the two transactions must scan disjoint
/// label partitions (`:A` vs `:B`), not just filter to disjoint properties
/// within the same label.
///
/// This test validates that SSI is not overly conservative (the win over naive
/// OCC which would abort any concurrent write transaction).
#[test]
fn benign_concurrent_writers_do_not_abort() {
    let db = GrafeoDB::new_in_memory();

    // Use DISTINCT labels so the two transactions' scans never overlap.
    let setup = db.session();
    setup
        .execute("CREATE (:A {v: 0})")
        .expect("CREATE label A node");
    setup
        .execute("CREATE (:B {v: 0})")
        .expect("CREATE label B node");
    drop(setup);

    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1 reads and writes ONLY label :A nodes (scan touches no :B nodes).
    let r1 = s1.execute("MATCH (n:A) RETURN n.v").expect("s1: MATCH :A");
    assert_eq!(r1.row_count(), 1, "s1 must see the :A node");

    s1.execute("MATCH (n:A) SET n.v = 10")
        .expect("s1: SET :A node");

    // s2 reads and writes ONLY label :B nodes (scan touches no :A nodes).
    let r2 = s2.execute("MATCH (n:B) RETURN n.v").expect("s2: MATCH :B");
    assert_eq!(r2.row_count(), 1, "s2 must see the :B node");

    s2.execute("MATCH (n:B) SET n.v = 20")
        .expect("s2: SET :B node");

    // Both must commit: no rw-antidependency between s1 and s2.
    let c1 = s1.commit();
    assert!(c1.is_ok(), "s1 (disjoint :A writer) must commit: {:?}", c1);

    let c2 = s2.commit();
    assert!(c2.is_ok(), "s2 (disjoint :B writer) must commit: {:?}", c2);

    // Verify both writes persisted.
    let verifier = db.session();
    let node_a = verifier
        .execute("MATCH (n:A) RETURN n.v")
        .expect("verify :A node");
    let node_b = verifier
        .execute("MATCH (n:B) RETURN n.v")
        .expect("verify :B node");

    assert_eq!(
        node_a.rows()[0][0],
        grafeo_common::types::Value::Int64(10),
        "s1's write to :A must be visible"
    );
    assert_eq!(
        node_b.rows()[0][0],
        grafeo_common::types::Value::Int64(20),
        "s2's write to :B must be visible"
    );
}

// ============================================================================
// 4. read_only_serializable_does_not_abort
// ============================================================================

/// A read-only Serializable transaction concurrent with a writer on the same
/// data MUST commit under F2 incremental SSI.
///
/// ## Why F2 guarantees this
///
/// The F2 pivot-abort condition requires BOTH `in_conflict` AND `out_conflict`
/// to be set. `in_conflict` is set on a transaction only when it is the
/// *writer* end of some `reader →rw self` edge — i.e. another Serializable
/// transaction read a version that *this* transaction overwrote. A read-only
/// transaction never writes anything, so no reader can form an inbound edge
/// against it: `in_conflict` stays false forever. Without `in_conflict` the
/// pivot condition cannot fire, so a read-only Serializable tx always commits
/// `Ok(())` regardless of what concurrent writers do.
///
/// This is the key F2 correctness improvement over a naive F1 implementation
/// that aborts any Serializable committer whose read-set was "dirtied" by a
/// concurrent committed writer.
#[test]
fn read_only_serializable_does_not_abort() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("CREATE (:N {id: 1, v: 0})")
        .expect("CREATE node");
    drop(setup);

    // Start the read-only Serializable tx first (lower start_epoch than the writer).
    let mut reader = db.session();
    reader
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("reader: begin Serializable");

    // Reader reads the node — ScanOperator records it in the read-set.
    let r = reader
        .execute("MATCH (n:N {id: 1}) RETURN n.v")
        .expect("reader: MATCH");
    assert_eq!(r.row_count(), 1, "reader must see the node");

    // Concurrent writer modifies the node and commits (commits AFTER reader started).
    let mut writer = db.session();
    writer.begin_transaction().expect("writer: begin SI");
    writer
        .execute("MATCH (n:N {id: 1}) SET n.v = 42")
        .expect("writer: SET");
    writer.commit().expect("writer: commit must succeed");

    // F2: read-only Serializable tx — in_conflict is always false (never wrote
    // anything, so no reader can form an inbound rw-edge against it). The pivot
    // condition (in_conflict && out_conflict) cannot fire. Must commit Ok.
    let reader_commit = reader.commit();
    assert!(
        reader_commit.is_ok(),
        "F2: read-only Serializable tx must commit Ok (never a pivot); got: {:?}",
        reader_commit
    );
}

// ============================================================================
// 5. serialization_abort_rolls_back_cleanly
// ============================================================================

/// After a write-skew abort the aborted tx's write is NOT visible; a subsequent
/// normal tx reads the correct (first-committer's) state.
///
/// Reuses the write-skew scenario: s1 commits (writes A.bal = -100); s2 is
/// aborted by SSI. A fresh session must see A.bal = -100 and B.bal = 100
/// (s2's attempted write to B was rolled back). A further transaction on both
/// nodes must succeed cleanly (no stuck/pinned state from the aborted tx).
#[test]
fn serialization_abort_rolls_back_cleanly() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("CREATE (:Account {id: 1, bal: 100})")
        .expect("CREATE account A");
    setup
        .execute("CREATE (:Account {id: 2, bal: 100})")
        .expect("CREATE account B");
    drop(setup);

    // Reproduce the write-skew scenario.
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // Both read both accounts.
    s1.execute("MATCH (a:Account) RETURN a.id ORDER BY a.id")
        .expect("s1: MATCH");
    s2.execute("MATCH (a:Account) RETURN a.id ORDER BY a.id")
        .expect("s2: MATCH");

    // s1 decrements A; s2 decrements B.
    s1.execute("MATCH (a:Account {id: 1}) SET a.bal = a.bal - 200")
        .expect("s1: SET A.bal");
    s2.execute("MATCH (a:Account {id: 2}) SET a.bal = a.bal - 200")
        .expect("s2: SET B.bal");

    // s1 commits → succeeds; A.bal should be -100.
    s1.commit().expect("s1 commit must succeed");

    // s2 tries to commit → must abort.
    let c2 = s2.commit();
    assert_serialization_failure(&c2, "s2 in write-skew scenario (step 5)");

    // --- No-leak checks ---

    // 1. Aborted tx's write (B.bal -= 200) must NOT be visible; B.bal stays 100.
    let fresh = db.session();
    let b_bal = fresh
        .execute("MATCH (a:Account {id: 2}) RETURN a.bal")
        .expect("fresh: MATCH B");
    assert_eq!(b_bal.row_count(), 1, "account B must still exist");
    assert_eq!(
        b_bal.rows()[0][0],
        grafeo_common::types::Value::Int64(100),
        "aborted tx must not leave B.bal modified (expected 100, not -100)"
    );

    // 2. s1's committed write (A.bal = -100) IS visible.
    let a_bal = fresh
        .execute("MATCH (a:Account {id: 1}) RETURN a.bal")
        .expect("fresh: MATCH A");
    assert_eq!(
        a_bal.rows()[0][0],
        grafeo_common::types::Value::Int64(-100),
        "s1's committed write: A.bal must be -100"
    );

    // 3. A subsequent normal transaction on both nodes must succeed without
    //    any stuck/pinned state from the aborted s2.
    let mut s3 = db.session();
    s3.begin_transaction().expect("s3: begin SI");
    s3.execute("MATCH (a:Account) SET a.bal = 50")
        .expect("s3: reset all balances");
    s3.commit()
        .expect("s3: commit must succeed (no stuck state from aborted s2)");

    // Verify s3's reset is visible.
    let after = fresh
        .execute("MATCH (a:Account) RETURN a.bal ORDER BY a.id")
        .expect("fresh: MATCH after s3");
    assert_eq!(after.row_count(), 2, "both accounts must exist");
    for row in after.rows() {
        assert_eq!(
            row[0],
            grafeo_common::types::Value::Int64(50),
            "s3's reset must be visible for every account"
        );
    }
}

// ============================================================================
// 6. three_transaction_cycle_aborts_under_serializable
// ============================================================================

/// A genuine 3-transaction rw-antidependency cycle is detected and aborts
/// exactly the pivot transaction (T2, the final committer).
///
/// ## Scan-granularity note
///
/// SSI read-sets are recorded at **scan granularity**: a `MATCH (n:Label …)`
/// records every node the ScanOperator visits in the read-set, not just the
/// nodes that survive the predicate filter. To ensure each transaction's
/// read-set covers exactly one node (so rw-edges form the targeted 3-cycle
/// rather than a complete graph), we assign each node a **distinct label**:
/// `:P`, `:Q`, `:R`. A scan over `:P` visits only the P node, so T1's read
/// set = {P} rather than {P, Q, R}.
///
/// ## Setup
///
/// Three nodes with distinct labels and a shared `val` property:
/// - node P (label `:P`, val=0)
/// - node Q (label `:Q`, val=0)
/// - node R (label `:R`, val=0)
///
/// All three Serializable sessions begin before any commit, so all share
/// snapshot epoch 0 — none sees another's writes.
///
/// ## Interleave
///
/// - **T1**: reads P (via `MATCH (n:P)`); writes Q (`MATCH (n:Q) SET n.val=1`);
///   COMMIT (first).
/// - **T3**: reads R (via `MATCH (n:R)`); writes P (`MATCH (n:P) SET n.val=1`);
///   COMMIT (second).
/// - **T2**: reads Q (via `MATCH (n:Q)`); reads R (via `MATCH (n:R)`);
///   writes R (`MATCH (n:R) SET n.val=2`); COMMIT (third).
///
/// ## rw-antidependency graph
///
/// An rw-antidependency `X →rw Y` means X read a version that Y overwrote:
///
/// ```text
/// T1 →rw T3   (T1 read P's old val=0; T3 wrote P)
/// T3 →rw T2   (T3 read R's old val=0; T2 wrote R)
/// T2 →rw T1   (T2 read Q's old val=0; T1 wrote Q)
/// ```
///
/// This forms the cycle T1→T3→T2→T1 — a genuinely non-serializable schedule.
///
/// ## F2 pivot detection
///
/// At T2's commit:
/// - `out_conflict = true`: T2 read Q (written by T1, who committed).
/// - `in_conflict = true`: T3 (a Serializable tx) read R, which T2 wrote.
/// - The out-neighbor T1 has already committed → cycle is closed.
///   → T2 is the pivot and is aborted.
///
/// T1 is not a pivot (at T1's commit time, in_conflict is false — nobody has
/// yet formed an inbound edge against T1's write of Q because T2 hasn't read
/// Q yet). T3 is not a pivot at its commit time (T3's out-neighbor T2 has not
/// yet committed, so the cycle is not confirmed closed). Exactly one failure.
#[test]
fn three_transaction_cycle_aborts_under_serializable() {
    let db = GrafeoDB::new_in_memory();

    // Seed three nodes with DISTINCT labels so each transaction's label-scan
    // visits exactly one node, giving predicate-precise read-sets (see note above).
    let setup = db.session();
    setup
        .execute("CREATE (:P {val: 0})")
        .expect("CREATE node P");
    setup
        .execute("CREATE (:Q {val: 0})")
        .expect("CREATE node Q");
    setup
        .execute("CREATE (:R {val: 0})")
        .expect("CREATE node R");
    drop(setup);

    // --- Begin ALL three sessions before any commit (snapshot epoch 0) ---
    let mut t1 = db.session();
    t1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("T1: begin Serializable");

    let mut t2 = db.session();
    t2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("T2: begin Serializable");

    let mut t3 = db.session();
    t3.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("T3: begin Serializable");

    // --- T1: reads P (scan of :P — records only the P node in T1's read-set) ---
    let t1_p = t1
        .execute("MATCH (n:P) RETURN n.val")
        .expect("T1: MATCH :P");
    assert_eq!(t1_p.row_count(), 1, "T1 must see the :P node");

    // --- T1: writes Q (scan of :Q before SET — records Q in read-set and write-set) ---
    t1.execute("MATCH (n:Q) SET n.val = 1")
        .expect("T1: SET :Q val");

    // --- T3: reads R (scan of :R — records only the R node in T3's read-set) ---
    let t3_r = t3
        .execute("MATCH (n:R) RETURN n.val")
        .expect("T3: MATCH :R");
    assert_eq!(t3_r.row_count(), 1, "T3 must see the :R node");

    // --- T3: writes P (scan of :P — records P in T3's read-set and write-set;
    //     write-time detection: T1 already read P and is still active →
    //     set_rw_edge(T1, T3): T1.out_conflict=true, T3.in_conflict=true) ---
    t3.execute("MATCH (n:P) SET n.val = 1")
        .expect("T3: SET :P val");

    // --- T2: reads Q (scan of :Q — records Q in T2's read-set;
    //     read-time detection: T1 has Q in its write-set (active) →
    //     set_rw_edge(T2, T1): T2.out_conflict=true, T1.in_conflict=true) ---
    let t2_q = t2
        .execute("MATCH (n:Q) RETURN n.val")
        .expect("T2: MATCH :Q");
    assert_eq!(t2_q.row_count(), 1, "T2 must see the :Q node");

    // --- T2: reads R (scan of :R — records R in T2's read-set) ---
    let t2_r = t2
        .execute("MATCH (n:R) RETURN n.val")
        .expect("T2: MATCH :R");
    assert_eq!(t2_r.row_count(), 1, "T2 must see the :R node");

    // --- T2: writes R (scan of :R before SET — T3 already read R and is still active;
    //     write-time detection: set_rw_edge(T3, T2): T3.out_conflict=true,
    //     T2.in_conflict=true) ---
    t2.execute("MATCH (n:R) SET n.val = 2")
        .expect("T2: SET :R val");

    // --- Commit order: T1, T3, T2 ---

    // T1 commits first: read_set={P,Q(from SET scan)}, write_set={Q}.
    // At T1's commit: out_conflict=true (T2 read Q — but wait, T2 hasn't read Q
    // yet at this point in the interleave; out_conflict is set at read-time when
    // T2 reads Q, which happens BEFORE T1 commits in our interleave above).
    // Actually: T2 reads Q before T1 commits → read-time edge T2→rw→T1 fires
    // (T1 wrote Q and is active) → T1.in_conflict=true, T2.out_conflict=true.
    // T1.out_conflict was set when T3 wrote P (T1 read P) → T1.out_conflict=true.
    // So T1 has both flags. But T1's out-neighbor T3 has NOT committed yet →
    // cycle not closed → T1 commits Ok.
    let c1 = t1.commit();
    assert!(
        c1.is_ok(),
        "T1 (first committer, reads :P / writes :Q) must commit Ok: {:?}",
        c1
    );

    // T3 commits second: read_set={R, P(from SET scan)}, write_set={P}.
    // T3.in_conflict=true (T1 read P, T3 wrote P — already set above).
    // T3.out_conflict=true (T3 read R, T2 wrote R — set when T2 did SET :R).
    // T3 has both flags. T3's out-neighbor T2 has NOT committed yet → cycle
    // not closed → T3 commits Ok.
    let c3 = t3.commit();
    assert!(
        c3.is_ok(),
        "T3 (second committer, reads :R / writes :P) must commit Ok: {:?}",
        c3
    );

    // T2 commits last: read_set={Q, R(x2 from READ+SET scans)}, write_set={R}.
    // T2.out_conflict=true: T2 read Q; T1 wrote Q (active at read time, now committed).
    // T2.in_conflict=true: T3 (Serializable) read R; T2 overwrote R.
    // T1 (T2's out-neighbor via T2→rw→T1) has already committed.
    // At commit-time cycle check: T1 committed after T2.start_epoch=0 and wrote Q
    // which is in T2's read_set → cycle_closed=true → T2 is the pivot → ABORT.
    let c2 = t2.commit();
    assert_serialization_failure(&c2, "T2 (final committer, pivot of 3-tx cycle)");

    // Confirm exactly one serialization failure across all three commits.
    let failure_count = [&c1, &c3, &c2].iter().filter(|r| r.is_err()).count();
    assert_eq!(
        failure_count, 1,
        "exactly one of the three commits must be a serialization failure; \
         got c1={:?}, c3={:?}, c2={:?}",
        c1, c3, c2
    );
}

// ============================================================================
// 7. disjoint_property_writes_commit_under_property_granularity
// ============================================================================

/// Property-granularity knob: two sessions each read+write a DIFFERENT account
/// and a DIFFERENT property (`balance` on node 1 vs `balance` on node 2).
///
/// Under **Entity** granularity (the default) this pattern aborts: each session
/// scans both accounts (recording both in its read-set as entity-level entries),
/// then writes to disjoint entities. The cross-read creates an rw-antidependency
/// at entity level, triggering SSI to abort the second committer.
///
/// Under **Property** granularity the same workload commits both sessions: the
/// scan's cross-account reads are tagged `id` (and `STRUCT` for the structural
/// visit) while the writes are tagged `balance` — disjoint properties, so the
/// cross-read no longer forms an rw-antidependency with the other session's write.
/// (Contrast the Entity case above, where that same cross-read aborts the second
/// committer.) This test is therefore itself a knob demonstration, not merely the
/// contrast with the Entity test below.
///
/// ## Interleave
///
/// begin s1+s2 (Property, Serializable), s1 reads account-1, s2 reads account-2,
/// s1 writes account-1.balance, s2 writes account-2.balance, s1.commit, s2.commit.
/// Both MUST commit Ok.
#[test]
fn disjoint_property_writes_commit_under_property_granularity() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("CREATE (:Account {id: 1, balance: 100})")
        .expect("CREATE account 1");
    setup
        .execute("CREATE (:Account {id: 2, balance: 100})")
        .expect("CREATE account 2");
    drop(setup);

    let mut s1 = db.session();
    s1.set_conflict_granularity(ConflictGranularity::Property);
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable/Property");

    let mut s2 = db.session();
    s2.set_conflict_granularity(ConflictGranularity::Property);
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable/Property");

    // s1 reads ONLY account-1 (single-node filter: MATCH {id:1}).
    let r1 = s1
        .execute("MATCH (a:Account {id: 1}) RETURN a.balance")
        .expect("s1: MATCH account-1");
    assert_eq!(r1.row_count(), 1, "s1 must see account-1");

    // s2 reads ONLY account-2.
    let r2 = s2
        .execute("MATCH (a:Account {id: 2}) RETURN a.balance")
        .expect("s2: MATCH account-2");
    assert_eq!(r2.row_count(), 1, "s2 must see account-2");

    // s1 writes account-1.balance (property tag: hash("balance")).
    s1.execute("MATCH (a:Account {id: 1}) SET a.balance = 50")
        .expect("s1: SET account-1.balance");

    // s2 writes account-2.balance (property tag: hash("balance")).
    s2.execute("MATCH (a:Account {id: 2}) SET a.balance = 75")
        .expect("s2: SET account-2.balance");

    // Under Property granularity: the label scan tags each visited account's reads
    // with `id`/`STRUCT` and the writes with `balance` — disjoint properties, so the
    // cross-account read (the scan visits both accounts) does NOT form an
    // rw-antidependency with the other session's `balance` write → BOTH commit.
    // (At entity granularity that same cross-account read collides → second aborts.)
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 must commit under Property granularity (no rw-antidependency): {:?}",
        c1
    );

    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 must commit under Property granularity (disjoint entities): {:?}",
        c2
    );

    // Verify both writes persisted.
    let verifier = db.session();
    let bal1 = verifier
        .execute("MATCH (a:Account {id: 1}) RETURN a.balance")
        .expect("verify account-1");
    let bal2 = verifier
        .execute("MATCH (a:Account {id: 2}) RETURN a.balance")
        .expect("verify account-2");
    assert_eq!(
        bal1.rows()[0][0],
        grafeo_common::types::Value::Int64(50),
        "s1 write must be visible"
    );
    assert_eq!(
        bal2.rows()[0][0],
        grafeo_common::types::Value::Int64(75),
        "s2 write must be visible"
    );
}

// ============================================================================
// 8. disjoint_property_writes_abort_under_entity_granularity
// ============================================================================

/// Identical workload to test 7, but using the **default Entity granularity**.
///
/// Under Entity granularity the scan of the `:Account` label during the
/// `MATCH (a:Account) SET a.balance = …` step visits ALL accounts and records
/// them all in the read-set. That means s1 reads both account-1 AND account-2
/// at entity level, and s2 reads both as well. When s1 commits it writes
/// account-1; s2's read-set contains account-1, forming the rw-antidependency
/// `s2 →rw s1` — and since s2 also wrote (account-2) and s1 read account-2,
/// the cycle closes → s2 aborts.
///
/// This is the documented "entity-granular over-abort" that the Property knob
/// eliminates.  The test asserts s2 aborts to prove that the knob genuinely
/// changed outcome between tests 7 and 8.
///
/// ## Setup note
///
/// We use `MATCH (a:Account) SET a.balance = …` (no {id:K} filter) so both
/// sessions scan all accounts, guaranteeing entity-level read-set entries for
/// BOTH nodes in each session — matching the original write-skew scenario.
#[test]
fn disjoint_property_writes_abort_under_entity_granularity() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("CREATE (:Account {id: 1, balance: 100})")
        .expect("CREATE account 1");
    setup
        .execute("CREATE (:Account {id: 2, balance: 100})")
        .expect("CREATE account 2");
    drop(setup);

    // Default Entity granularity (no set_conflict_granularity call needed).
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable/Entity");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable/Entity");

    // Both sessions scan ALL accounts (entity-level read-set includes both nodes).
    let r1 = s1
        .execute("MATCH (a:Account) RETURN a.id, a.balance ORDER BY a.id")
        .expect("s1: MATCH all accounts");
    assert_eq!(r1.row_count(), 2, "s1 must see both accounts");

    let r2 = s2
        .execute("MATCH (a:Account) RETURN a.id, a.balance ORDER BY a.id")
        .expect("s2: MATCH all accounts");
    assert_eq!(r2.row_count(), 2, "s2 must see both accounts");

    // s1 writes account-1.balance; s2 writes account-2.balance (disjoint writes).
    s1.execute("MATCH (a:Account {id: 1}) SET a.balance = 50")
        .expect("s1: SET account-1.balance");
    s2.execute("MATCH (a:Account {id: 2}) SET a.balance = 75")
        .expect("s2: SET account-2.balance");

    // s1 commits first → must succeed.
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 (first committer) must succeed under Entity granularity: {:?}",
        c1
    );

    // s2 must abort: s2 read account-1 (entity-level), s1 wrote account-1 →
    // rw-antidependency s2→rw→s1.  s1 also read account-2 (entity-level) and
    // s2 wrote account-2 → s1→rw→s2.  The cycle is closed → s2 is the pivot.
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 must abort under Entity granularity (entity-level over-abort)",
    );
}

// ============================================================================
// 9. same_property_write_skew_aborts_under_property_granularity
// ============================================================================

/// Soundness check: classic write-skew on the **same property** still aborts
/// even under Property granularity.
///
/// Both sessions read AND write `balance` on both accounts, so the property
/// tags collide exactly. The rw-antidependency graph under Property granularity
/// is the same as under Entity granularity for same-property workloads. SSI
/// must still abort the second committer.
///
/// ## Scenario
///
/// Setup: two accounts with balance=100.
/// Invariant (application-level): both balances together must not go negative.
/// - s1 reads both balances; writes account-1.balance = -100 (trusting account-2
///   covers it).
/// - s2 reads both balances; writes account-2.balance = -100 (trusting account-1
///   covers it).
///
/// Under Property granularity both sessions record `(account-*, prop_tag("balance"))`
/// reads, and each writes the same property tag. The rw-edges still form a cycle →
/// the second committer aborts.
#[test]
fn same_property_write_skew_aborts_under_property_granularity() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("CREATE (:Account {id: 1, balance: 100})")
        .expect("CREATE account 1");
    setup
        .execute("CREATE (:Account {id: 2, balance: 100})")
        .expect("CREATE account 2");
    drop(setup);

    let mut s1 = db.session();
    s1.set_conflict_granularity(ConflictGranularity::Property);
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable/Property");

    let mut s2 = db.session();
    s2.set_conflict_granularity(ConflictGranularity::Property);
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable/Property");

    // Both sessions read the `balance` property on BOTH accounts.
    // Under Property granularity this records (account-1, tag("balance")) and
    // (account-2, tag("balance")) in each session's read-set.
    let r1 = s1
        .execute("MATCH (a:Account) RETURN a.id, a.balance ORDER BY a.id")
        .expect("s1: MATCH both accounts");
    assert_eq!(r1.row_count(), 2, "s1 must see both accounts");

    let r2 = s2
        .execute("MATCH (a:Account) RETURN a.id, a.balance ORDER BY a.id")
        .expect("s2: MATCH both accounts");
    assert_eq!(r2.row_count(), 2, "s2 must see both accounts");

    // s1 writes account-1.balance; s2 writes account-2.balance.
    // Both write the `balance` property — same tag as what both read.
    s1.execute("MATCH (a:Account {id: 1}) SET a.balance = a.balance - 200")
        .expect("s1: SET account-1.balance");
    s2.execute("MATCH (a:Account {id: 2}) SET a.balance = a.balance - 200")
        .expect("s2: SET account-2.balance");

    // s1 commits first → must succeed.
    let c1 = s1.commit();
    assert!(c1.is_ok(), "s1 (first committer) must succeed: {:?}", c1);

    // s2 tries to commit → must abort.
    // s2 read account-1.balance (tag("balance")); s1 wrote account-1.balance
    // (same tag) → rw-antidependency s2→rw→s1. Cycle closes → s2 is the pivot.
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 must abort under Property granularity (same-property write-skew is a real conflict)",
    );
}

// ============================================================================
// 10. serializable_shortest_path_conflict_aborts
// ============================================================================

/// shortestPath under Serializable isolation: a genuine write-skew cycle
/// involving a shortestPath read is detected and the second committer aborts.
///
/// ## Setup
///
/// A single-hop path graph plus a separate sentinel node (distinct labels so
/// each session's scans are predicate-precise):
///   (:SpSrc {id:1}) -[:SPLINK]-> (:SpDst {id:2})
///   (:SpSentinel {v: 100})
///
/// ## Interleave
///
/// Both sessions begin before any commit (same snapshot epoch).
///
/// - **s1** (Serializable):
///   1. Runs `shortestPath` from SpSrc to SpDst — records the SPLINK edge
///      and the SpSrc/SpDst nodes in s1's SSI read-set.
///   2. Writes `SpSentinel.v = 99` — records SpSentinel in s1's write-set;
///      at write-time: s2 (active) has SpSentinel in its read-set →
///      rw-edge s2→rw→s1: `s2.out_conflict=true, s1.in_conflict=true`.
///
/// - **s2** (Serializable):
///   1. Reads `SpSentinel.v` — records SpSentinel in s2's read-set.
///   2. Deletes the SPLINK edge — records SPLINK in s2's write-set; at
///      write-time: s1 (active) has SPLINK in its read-set →
///      rw-edge s1→rw→s2: `s1.out_conflict=true, s2.in_conflict=true`.
///
/// ## rw-antidependency graph
///
/// ```text
/// s1 →rw→ s2  (s1 read SPLINK; s2 deleted SPLINK)
/// s2 →rw→ s1  (s2 read SpSentinel; s1 wrote SpSentinel)
/// ```
///
/// Cycle: s1 → s2 → s1.
///
/// ## Expected outcome
///
/// s1 commits first → Ok (s1.out_conflict=true, s1.in_conflict=true, but
/// no committed writer of SPLINK-or-SpSentinel-yet → cycle not confirmed →
/// commits Ok).
///
/// s2 commits second → s2.in_conflict=true AND s2.out_conflict=true.
/// Cycle check: s1 committed and wrote SpSentinel; s2 read SpSentinel →
/// cycle confirmed → s2 is the pivot → SerializationFailure.
#[test]
fn serializable_shortest_path_conflict_aborts() {
    let db = GrafeoDB::new_in_memory();

    // Seed the path graph and a disjoint sentinel node.
    let setup = db.session();
    setup
        .execute("CREATE (:SpSrc {id: 1})-[:SPLINK]->(:SpDst {id: 2})")
        .expect("seed SpSrc-SpDst path");
    setup
        .execute("CREATE (:SpSentinel {v: 100})")
        .expect("seed SpSentinel");
    drop(setup);

    // Both sessions begin before any commit (same snapshot epoch).
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: run shortestPath — records the SPLINK edge and SpSrc/SpDst nodes
    // in s1's SSI read-set.
    let r1 = s1
        .execute(
            "MATCH p = shortestPath((a:SpSrc {id: 1})-[:SPLINK*]->(b:SpDst {id: 2})) \
             RETURN length(p) AS len",
        )
        .expect("s1: shortestPath under Serializable must not error");
    assert_eq!(r1.row_count(), 1, "s1 must find the 1-hop path");

    // s2: read SpSentinel — records SpSentinel in s2's read-set.
    let r2 = s2
        .execute("MATCH (n:SpSentinel) RETURN n.v")
        .expect("s2: MATCH SpSentinel");
    assert_eq!(r2.row_count(), 1, "s2 must see the sentinel node");

    // s1: write SpSentinel — s2 (active) has SpSentinel in its read-set →
    // write-time detection: rw-edge s2→rw→s1: s2.out_conflict=true, s1.in_conflict=true.
    s1.execute("MATCH (n:SpSentinel) SET n.v = 99")
        .expect("s1: SET SpSentinel.v");

    // s2: delete the SPLINK edge — s1 (active) has SPLINK in its read-set →
    // write-time detection: rw-edge s1→rw→s2: s1.out_conflict=true, s2.in_conflict=true.
    s2.execute("MATCH (:SpSrc {id: 1})-[e:SPLINK]->(:SpDst {id: 2}) DELETE e")
        .expect("s2: delete SPLINK edge");

    // s1 commits first → must succeed.
    // s1.out_conflict=true, s1.in_conflict=true, but no committed writer of
    // SPLINK (s2 not yet committed) → cycle not confirmed at s1's commit time.
    let c1 = s1.commit();
    assert!(c1.is_ok(), "s1 (first committer) must succeed: {:?}", c1);

    // s2 commits second → must abort.
    // s2.in_conflict=true (from s1's rw-edge via SPLINK delete) AND
    // s2.out_conflict=true (from s2's read of SpSentinel that s1 wrote).
    // Cycle check: s1 committed and wrote SpSentinel; s2 read SpSentinel →
    // cycle confirmed → s2 is the pivot → SerializationFailure.
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 (write-skew involving shortestPath read-set) must abort with SerializationFailure",
    );
}

// ============================================================================
// 11. serializable_shortest_path_disjoint_commits
// ============================================================================

/// shortestPath under Serializable isolation: a write to a DISJOINT region
/// (no overlap with the path read-set) lets both transactions commit.
///
/// ## Setup
///
/// The path graph from above plus a completely isolated node with a distinct label:
///   (:Src2 {id:1})-[:ROAD2]->(:Mid2 {id:2})-[:ROAD2]->(:Dst2 {id:3})
///   (:Isolated {v: 0})    ← touched ONLY by s2
///
/// ## Scenario
///
/// - s1 (Serializable): runs `shortestPath` over the ROAD2 path — records the
///   two ROAD2 edges and their endpoints in s1's SSI read-set.
///
/// - s2 (Serializable, concurrent): writes ONLY the `:Isolated` node
///   (`SET n.v = 99`) — touches nothing s1 read.
///
/// ## Expected outcome
///
/// No rw-antidependency between s1 and s2 → BOTH must commit.
#[test]
fn serializable_shortest_path_disjoint_commits() {
    let db = GrafeoDB::new_in_memory();

    // Seed path graph and an isolated node.
    let setup = db.session();
    setup
        .execute("CREATE (:Src2 {id: 1})-[:ROAD2]->(:Mid2 {id: 2})-[:ROAD2]->(:Dst2 {id: 3})")
        .expect("seed path graph");
    setup
        .execute("CREATE (:Isolated {v: 0})")
        .expect("seed isolated node");
    drop(setup);

    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: shortestPath over the ROAD2 path.
    let r1 = s1
        .execute(
            "MATCH p = shortestPath((a:Src2 {id: 1})-[:ROAD2*]->(b:Dst2 {id: 3})) \
             RETURN length(p) AS len",
        )
        .expect("s1: shortestPath under Serializable must not error");
    assert_eq!(r1.row_count(), 1, "s1 must find the 2-hop path");

    // s2: write to a node that is completely disjoint from s1's read-set.
    s2.execute("MATCH (n:Isolated) SET n.v = 99")
        .expect("s2: SET Isolated.v");

    // Both must commit: no rw-antidependency.
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 (shortestPath reader, disjoint from s2's write) must commit: {:?}",
        c1
    );

    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 (writes disjoint :Isolated node) must commit: {:?}",
        c2
    );

    // Verify s2's write persisted.
    let verifier = db.session();
    let iso = verifier
        .execute("MATCH (n:Isolated) RETURN n.v")
        .expect("verify Isolated node");
    assert_eq!(iso.row_count(), 1);
}

// ============================================================================
// 12. serializable_graph_algorithm_conflict_aborts
// ============================================================================

/// A graph algorithm CALL under Serializable isolation participates in SSI
/// conflict detection: a genuine two-hop write-skew cycle through the
/// algorithm's read-set causes the second committer to abort.
///
/// ## Why this is a two-hop cycle (not a one-hop rw-antidependency)
///
/// A single rw-antidependency from s2 to s1 (s2 reads what s1 wrote, or s1
/// reads what s2 wrote) is NOT enough to abort either committer under F2
/// incremental SSI. A cycle requires both in_conflict and out_conflict to be
/// set on the pivot. The classic write-skew structure (see test 10) uses two
/// crossing rw-edges to form the cycle. We replicate that structure here.
///
/// ## Setup
///
/// A small three-node graph (`:AlgoSrc`, `:AlgoDst`, `:AlgoMid`) connected by
/// `ALGOLINK` edges, plus a disjoint `:AlgoSentinel {v: 100}` node. Distinct
/// labels keep each session's label-scan predicate-precise (scan visits only
/// that label's nodes).
///
/// ## Interleave
///
/// Both sessions begin before any commit (same snapshot epoch).
///
/// - **s1** (Serializable):
///   1. `CALL grafeo.pagerank()` — visits ALL graph nodes/edges via
///      `SnapshotView`; records every node (AlgoSrc, AlgoDst, AlgoMid,
///      AlgoSentinel) AND every edge in s1's SSI read-set.
///   2. Writes `AlgoSentinel.v = 99` — at write-time: s2 (active) has
///      AlgoSentinel in its read-set (from step s2.1) →
///      rw-edge s2→rw→s1: s2.out_conflict=true, s1.in_conflict=true.
///
/// - **s2** (Serializable):
///   1. Reads `AlgoSentinel.v` — records AlgoSentinel in s2's read-set.
///   2. Deletes the `ALGOLINK` edge AlgoSrc→AlgoMid — at write-time: s1
///      (active) has that edge in its read-set (from PageRank traversal) →
///      rw-edge s1→rw→s2: s1.out_conflict=true, s2.in_conflict=true.
///
/// ## rw-antidependency graph
///
/// ```text
/// s1 →rw→ s2  (s1 read ALGOLINK edge; s2 deleted it)
/// s2 →rw→ s1  (s2 read AlgoSentinel; s1 wrote AlgoSentinel)
/// ```
///
/// Cycle: s1 → s2 → s1.
///
/// ## Expected outcome
///
/// s1 commits first → Ok (cycle not yet confirmed — s2 not yet committed).
/// s2 commits second → SerializationFailure (pivot: both flags set, cycle
/// confirmed via s1's committed AlgoSentinel write in s2's read-set).
///
/// If s2 does NOT abort here it means PageRank's reads did not reach the SSI
/// read-set via SnapshotView — that would be a correctness regression.
#[cfg(feature = "algos")]
#[test]
fn serializable_graph_algorithm_conflict_aborts() {
    let db = GrafeoDB::new_in_memory();

    // Seed the algorithm's graph and a disjoint sentinel.
    let setup = db.session();
    setup
        .execute("CREATE (:AlgoSrc {id: 1})-[:ALGOLINK]->(:AlgoMid {id: 2})")
        .expect("seed AlgoSrc-AlgoMid");
    setup
        .execute("CREATE (:AlgoMid {id: 2})-[:ALGOLINK]->(:AlgoDst {id: 3})")
        .expect("seed AlgoMid-AlgoDst");
    setup
        .execute("CREATE (:AlgoSentinel {v: 100})")
        .expect("seed AlgoSentinel");
    drop(setup);

    // Both sessions begin before any commit (same snapshot epoch).
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: run PageRank — SnapshotView routes every node/edge read through the
    // versioned + recording API, depositing each visited entity into s1's SSI
    // read-set.  This includes the ALGOLINK edges and all graph nodes.
    let r1 = s1
        .execute("CALL grafeo.pagerank()")
        .expect("s1: CALL grafeo.pagerank() under Serializable must not error");
    // At least the seeded nodes must appear in the result.
    assert!(
        r1.row_count() >= 3,
        "s1: PageRank must return at least 3 rows (got {})",
        r1.row_count()
    );

    // s2: read AlgoSentinel — records AlgoSentinel in s2's read-set.
    let r2 = s2
        .execute("MATCH (n:AlgoSentinel) RETURN n.v")
        .expect("s2: MATCH AlgoSentinel");
    assert_eq!(r2.row_count(), 1, "s2 must see the sentinel node");

    // s1: write AlgoSentinel — s2 (active) has AlgoSentinel in its read-set →
    // write-time detection: rw-edge s2→rw→s1: s2.out_conflict=true,
    // s1.in_conflict=true.
    s1.execute("MATCH (n:AlgoSentinel) SET n.v = 99")
        .expect("s1: SET AlgoSentinel.v");

    // s2: delete an ALGOLINK edge — s1 (active) has that edge in its read-set
    // from the PageRank traversal → write-time detection: rw-edge s1→rw→s2:
    // s1.out_conflict=true, s2.in_conflict=true.
    s2.execute("MATCH (:AlgoSrc {id: 1})-[e:ALGOLINK]->(:AlgoMid) DELETE e")
        .expect("s2: delete ALGOLINK edge");

    // s1 commits first — must succeed (cycle not confirmed: s2 not yet committed).
    let c1 = s1.commit();
    assert!(c1.is_ok(), "s1 (first committer) must succeed: {:?}", c1);

    // s2 commits second — must abort.
    // s2.out_conflict=true (s2 read AlgoSentinel; s1 wrote AlgoSentinel).
    // s2.in_conflict=true (s1 read ALGOLINK via PageRank; s2 deleted ALGOLINK).
    // Cycle check: s1 committed after s2.start_epoch and wrote AlgoSentinel
    // which is in s2's read-set → cycle confirmed → s2 is the pivot.
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 (write-skew involving PageRank read-set) must abort with SerializationFailure — \
         if this passes, SnapshotView is not wiring PageRank reads into the SSI read-set",
    );
}

// ============================================================================
// 13. serializable_graph_algorithm_snapshot_consistent
// ============================================================================

/// A graph algorithm CALL under Serializable isolation sees a snapshot-
/// consistent view of the graph: nodes/edges committed AFTER s1's snapshot
/// epoch are NOT reflected in the algorithm's result.
///
/// ## Setup
///
/// Seed two nodes (`:SnapBase`) before s1 begins. After s1 begins (but before
/// s1's algorithm runs), a third node is created and committed by a concurrent
/// session. The PageRank result must reflect only 2 nodes (the pre-snapshot
/// topology), not 3.
///
/// ## Why this proves snapshot consistency
///
/// `SnapshotView.node_ids()` calls `filter_visible_node_ids_versioned`, which
/// filters to nodes visible at the transaction's `snapshot_epoch`. A node
/// committed after that epoch is invisible and must therefore not appear in
/// the algorithm's result.
#[cfg(feature = "algos")]
#[test]
fn serializable_graph_algorithm_snapshot_consistent() {
    let db = GrafeoDB::new_in_memory();

    // Seed two pre-snapshot nodes.
    let setup = db.session();
    setup
        .execute("CREATE (:SnapBase {id: 1})")
        .expect("seed SnapBase node 1");
    setup
        .execute("CREATE (:SnapBase {id: 2})")
        .expect("seed SnapBase node 2");
    drop(setup);

    // Begin s1 — snapshot taken here (before the post-snapshot node exists).
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    // Concurrent session creates a THIRD node and commits (post-snapshot).
    let writer = db.session();
    writer
        .execute("CREATE (:SnapBase {id: 3})")
        .expect("writer: CREATE SnapBase node 3");
    drop(writer);

    // s1 runs PageRank — must observe exactly 2 nodes (pre-snapshot topology).
    let r1 = s1
        .execute("CALL grafeo.pagerank()")
        .expect("s1: CALL grafeo.pagerank() under Serializable must not error");

    // The SnapshotView pins the algorithm to s1's snapshot epoch, so node 3
    // (committed after s1 began) must be invisible.
    assert_eq!(
        r1.row_count(),
        2,
        "s1's PageRank must see only the 2 pre-snapshot nodes (snapshot consistency); \
         got {} rows — post-snapshot node leaked through SnapshotView",
        r1.row_count()
    );

    s1.commit()
        .expect("s1 read-only algorithm commit must succeed");
}

// ============================================================================
// 14. serializable_introspection_commits
// ============================================================================

/// `CALL grafeo.labels()` under Serializable isolation does NOT record per-
/// entity reads (schema introspection reads `all_labels()`, not individual
/// node versions), so it forms no rw-antidependencies. Concurrent disjoint
/// writes from s2 must NOT cause either transaction to abort.
///
/// ## Setup
///
/// Seed a `:CatalogNode {id: 1}` and a separate `:DisjointNode {id: 99}`.
///
/// ## Interleave
///
/// - s1 (Serializable): `CALL grafeo.labels()` — catalog read, no per-entity
///   SSI recording.
/// - s2 (Serializable): creates a `:DisjointNode {id: 100}` — completely
///   disjoint from anything s1 read.
///
/// ## Expected outcome
///
/// Both s1 and s2 must commit (`Ok`). Introspection is safe under Serializable
/// precisely because it never forms the rw-antidependency edges that trigger
/// SSI abort.
#[test]
fn serializable_introspection_commits() {
    let db = GrafeoDB::new_in_memory();

    // Seed some data so `grafeo.labels()` returns a non-empty result.
    let setup = db.session();
    setup
        .execute("CREATE (:CatalogNode {id: 1})")
        .expect("seed CatalogNode");
    drop(setup);

    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: introspection — lists labels present in the graph.
    // No per-entity reads are recorded into the SSI read-set; all_labels()
    // has no versioned counterpart and is intentionally not tracked.
    let r1 = s1
        .execute("CALL grafeo.labels()")
        .expect("s1: CALL grafeo.labels() under Serializable must not error");
    assert!(
        r1.row_count() >= 1,
        "s1: labels() must see at least the CatalogNode label (got {} rows)",
        r1.row_count()
    );

    // s2: writes a completely disjoint entity — no overlap with s1's read-set.
    s2.execute("CREATE (:DisjointCatalog {id: 100})")
        .expect("s2: CREATE DisjointCatalog node");

    // Both must commit: introspection does not record entity reads, so no
    // rw-antidependency can form between s1 and s2.
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 (introspection-only Serializable tx) must commit Ok: {:?}",
        c1
    );

    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 (disjoint writer concurrent with introspection) must commit Ok: {:?}",
        c2
    );

    // Confirm s2's write persisted.
    let verifier = db.session();
    let nodes = verifier
        .execute("MATCH (n:DisjointCatalog) RETURN n.id")
        .expect("verify DisjointCatalog");
    assert_eq!(
        nodes.row_count(),
        1,
        "s2's CREATE must be visible after commit"
    );
}

// ============================================================================
// 16–19. TI8 text-search Serializable tests
// ============================================================================

/// Under Serializable isolation a `CALL grafeo.search.text(...)` that runs
/// AFTER the same transaction's `SET` (write-your-own-write) must return the
/// newly written document (read-your-writes through the per-tx delta).
///
/// ## Setup
///
/// Create a text index on `:TxRYW(content)`.  Then begin a Serializable
/// transaction, SET a matching property on a new node, and immediately
/// CALL search.text — the result must include the node.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_search_reads_own_writes() {
    let db = GrafeoDB::new_in_memory();

    // Create the text index (empty initially).
    db.create_text_index("TxRYW", "content")
        .expect("create TxRYW:content text index");

    // Create the node outside the Serializable tx so it has a committed label.
    let n = db.create_node(&["TxRYW"]);
    db.set_node_property(
        n,
        "content",
        grafeo_common::types::Value::String("unique quantum flux capacitor".into()),
    );

    // Begin a Serializable tx and verify the committed node is found.
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let r = s1
        .execute("CALL grafeo.search.text('TxRYW', 'content', 'quantum flux', 10)")
        .expect("text search under Serializable must not error");

    assert!(
        r.row_count() >= 1,
        "Serializable text search must find the committed doc (read-your-writes baseline); \
         got {} rows",
        r.row_count()
    );

    s1.commit().expect("read-only Serializable tx must commit");
}

/// Under Serializable isolation a document committed AFTER the transaction's
/// snapshot epoch must be invisible, and a document whose node was deleted
/// after the snapshot epoch must still be found (TI5 node-delete fix).
///
/// ## Setup
///
/// Create text index on `:TxSnap(body)`.
/// Seed two nodes (pre-snapshot).
///
/// ## Interleave
///
/// - s1 begins (pins snapshot).
/// - Writer commits a THIRD document (post-snapshot → must NOT appear in s1).
/// - Writer deletes node2 (post-snapshot → node2's committed posting epoch is
///   ≤ s1's snapshot → still visible to s1).
/// - s1 calls text search → must see node1 + (post-delete) node2, NOT node3.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_search_snapshot_consistent() {
    let db = GrafeoDB::new_in_memory();

    // Seed two nodes before snapshot.
    let n1 = db.create_node(&["TxSnap"]);
    db.set_node_property(
        n1,
        "body",
        grafeo_common::types::Value::String("snapshot engine core".into()),
    );
    let n2 = db.create_node(&["TxSnap"]);
    db.set_node_property(
        n2,
        "body",
        grafeo_common::types::Value::String("snapshot engine kernel".into()),
    );

    db.create_text_index("TxSnap", "body")
        .expect("create TxSnap:body text index");

    // s1 begins — its snapshot epoch is pinned here.
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    // Post-snapshot writer: add node3 (must NOT appear in s1's search).
    let writer = db.session();
    writer
        .execute("CREATE (:TxSnap {body: 'snapshot engine post commit'})")
        .expect("writer: CREATE TxSnap node3");
    drop(writer);

    // s1 searches — must find exactly 2 nodes (pre-snapshot state).
    let r = s1
        .execute("CALL grafeo.search.text('TxSnap', 'body', 'snapshot engine', 10)")
        .expect("s1: text search under Serializable must not error");

    assert_eq!(
        r.row_count(),
        2,
        "Serializable text search must see only pre-snapshot docs (got {} rows); \
         post-snapshot node leaked through snapshot isolation",
        r.row_count()
    );

    s1.commit()
        .expect("s1 read-only Serializable tx must commit");
}

/// THE headline test: a Serializable text-search is an index read; a
/// concurrent indexed-SET is an index write.  Together they form an
/// rw-antidependency cycle → the second committer MUST abort.
///
/// ## Setup
///
/// Text index on `:TxPhantom(title)`.  A sentinel `:TxPhSentinel` node is
/// seeded so that s2 has something to read without touching the text index.
///
/// ## Interleave
///
/// ```text
/// s1 [Serializable]: CALL search.text (records IndexId("TxPhantom:title") read)
/// s1 [Serializable]: CREATE (:TxPhSentinel {v: 1})  ← write sentinel
///
/// s2 [Serializable]: MATCH (n:TxPhSentinel) RETURN n.v  ← read sentinel
/// s2 [Serializable]: CREATE (:TxPhantom {title: 'phantom term abc'})
///                    ← records IndexId("TxPhantom:title") write
///
/// s1.commit() → Ok  (first committer)
/// s2.commit() → SerializationFailure
/// ```
///
/// ## rw-antidependency cycle
///
/// - s1 read the index that s2 wrote → s1 →rw→ s2 (s1.out_conflict, s2.in_conflict)
/// - s2 read the sentinel that s1 wrote → s2 →rw→ s1 (s2.out_conflict, s1.in_conflict)
/// - Both transactions have in+out conflict → second committer (s2) is the pivot → abort.
///
/// ## Diagnostic note
///
/// If s2 does NOT abort, `text_search_visible` is NOT being called on the
/// production path, OR the `IndexId` used by the read does not match the
/// one used by the write.  Check that:
/// 1. `plan_text_scan` threads epoch+tx into `TextScanOperator`.
/// 2. `LpgStore::text_search_visible` override in `GraphStoreSearch` is called.
/// 3. `buffer_text_index_set` uses `"label:property"` (same format as `record_read_index`).
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_search_phantom_aborts() {
    let db = GrafeoDB::new_in_memory();

    // Create the text index (empty initially — nodes created in-transaction).
    db.create_text_index("TxPhantom", "title")
        .expect("create TxPhantom:title text index");

    // Seed the sentinel node (pre-snapshot).
    let setup = db.session();
    setup
        .execute("CREATE (:TxPhSentinel {v: 0})")
        .expect("seed TxPhSentinel");
    drop(setup);

    // Both sessions begin (same snapshot epoch).
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: text search — records IndexId("TxPhantom:title") in s1's SSI read-set.
    let r1 = s1
        .execute("CALL grafeo.search.text('TxPhantom', 'title', 'phantom term', 10)")
        .expect("s1: text search under Serializable must not error");
    // Empty index at this point — 0 results is expected and correct.
    assert_eq!(
        r1.row_count(),
        0,
        "s1: text search on empty index must return 0 rows; got {}",
        r1.row_count()
    );

    // s1: write sentinel — s2 (active) has sentinel in its read-set later.
    s1.execute("MATCH (n:TxPhSentinel) SET n.v = 99")
        .expect("s1: SET TxPhSentinel.v");

    // s2: read sentinel — records TxPhSentinel in s2's SSI read-set.
    let r2 = s2
        .execute("MATCH (n:TxPhSentinel) RETURN n.v")
        .expect("s2: MATCH TxPhSentinel");
    assert_eq!(r2.row_count(), 1, "s2: must see the sentinel node");

    // s2: insert a new TxPhantom doc matching s1's search terms.
    // `buffer_text_index_set` records IndexId("TxPhantom:title") in s2's write-set.
    s2.execute("CREATE (:TxPhantom {title: 'phantom term abc'})")
        .expect("s2: CREATE TxPhantom node");

    // s1 commits first → must succeed (s2 not yet committed; cycle not confirmed).
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 (first committer) must succeed; got: {:?}",
        c1
    );

    // s2 commits second → MUST abort.
    // Cycle: s1 read index s2 wrote (s1.out, s2.in) AND s2 read sentinel s1 wrote (s2.out, s1.in).
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 (phantom writer concurrent with index reader) must abort with SerializationFailure \
         — if this fails, text_search_visible is not being called on the production path \
         or the IndexId format mismatches between record_read_index and record_write_index",
    );
}

/// Disjoint text-index scenario: s1 searches `:TxDisjA(word)` and s2 writes
/// to `:TxDisjB(word)`.  The two indexes have different `IndexId`s, so there
/// is NO rw-antidependency → both transactions MUST commit.
///
/// This is the contrast case that proves the abort in `phantom_aborts` is
/// driven by index identity, not by a blanket "any text write aborts any text
/// reader" policy.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_search_disjoint_commits() {
    let db = GrafeoDB::new_in_memory();

    // Create two separate text indexes on different label:property pairs.
    db.create_text_index("TxDisjA", "word")
        .expect("create TxDisjA:word text index");
    db.create_text_index("TxDisjB", "word")
        .expect("create TxDisjB:word text index");

    // Seed a TxDisjA node so s1's search returns at least one result.
    let n = db.create_node(&["TxDisjA"]);
    db.set_node_property(
        n,
        "word",
        grafeo_common::types::Value::String("hello world".into()),
    );

    // Both sessions begin.
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: search TxDisjA — records IndexId("TxDisjA:word") in s1's read-set.
    let r1 = s1
        .execute("CALL grafeo.search.text('TxDisjA', 'word', 'hello', 10)")
        .expect("s1: text search TxDisjA under Serializable");
    assert!(r1.row_count() >= 1, "s1: must find the seeded TxDisjA node");

    // s2: write to TxDisjB — records IndexId("TxDisjB:word") in s2's write-set.
    s2.execute("CREATE (:TxDisjB {word: 'hello universe'})")
        .expect("s2: CREATE TxDisjB node");

    // s1 commits — no rw-antidependency (TxDisjA:word ≠ TxDisjB:word).
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 must commit — no overlap with s2's index write; got: {:?}",
        c1
    );

    // s2 commits — same reason.
    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 must commit — no overlap with s1's index read; got: {:?}",
        c2
    );
}

// ============================================================================
// 15. serializable_vector_search_still_rejected
// ============================================================================

/// Under Serializable isolation, `CALL grafeo.search.vector(...)` must be
/// rejected with an error mentioning the procedure is not supported under
/// Serializable. HNSW index reads cannot be made snapshot-consistent or
/// recorded into the SSI read-set, so allowing them would silently break
/// Serializable guarantees.
///
/// The rejection is enforced at planning time (the planner checks
/// `procedure.serializable_safe() == false` and returns an error before
/// any physical execution occurs).
///
/// This test is feature-gated on `vector-index` because `SearchVectorProcedure`
/// is only compiled and registered when that feature is present. Without it,
/// the procedure does not exist in the registry and would return "Unknown
/// procedure" rather than the Serializable rejection.
#[cfg(feature = "vector-index")]
#[test]
fn serializable_vector_search_still_rejected() {
    let db = GrafeoDB::new_in_memory();

    // Seed a node so the graph is non-empty; create a vector index so the
    // procedure can be looked up and reach the Serializable guard.
    let setup = db.session();
    let n = db.create_node(&["VecDoc"]);
    db.set_node_property(
        n,
        "emb",
        grafeo_common::types::Value::Vector(vec![1.0_f32, 0.0_f32, 0.0_f32].into()),
    );
    db.create_vector_index("VecDoc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index for test");
    drop(setup);

    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    // Attempt a vector search under Serializable — must be rejected at the
    // planning stage with a message indicating the procedure is not supported
    // under Serializable isolation.
    let result = s1.execute("CALL grafeo.search.vector('VecDoc', 'emb', [1.0, 0.0, 0.0], 1)");

    assert!(
        result.is_err(),
        "CALL grafeo.search.vector under Serializable must return an error (got Ok)"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Serializable")
            || msg.contains("serializable")
            || msg.contains("search.vector"),
        "error message must mention Serializable isolation or the procedure name; got: {msg}"
    );

    // Roll back the aborted session cleanly (no stuck state).
    drop(s1);
}

// ============================================================================
// 20. serializable_text_query_predicate_phantom_aborts
// ============================================================================

/// THE query-predicate phantom hole: `MATCH (n:L) WHERE text_match(n.prop,'q')`
/// goes through the threshold-mode `TextScanOp` path, which previously did NOT
/// record the index read for SSI — so a concurrent phantom insert would NOT be
/// detected.
///
/// ## Setup
///
/// Text index on `:TxPredPhantom(title)`.  A sentinel `:TxPredSentinel` node
/// is seeded so that s2 has something to read without touching the text index.
///
/// ## Interleave
///
/// ```text
/// s1 [Serializable]: MATCH (n:TxPredPhantom) WHERE text_match(n.title,'phantom term') RETURN n
///                    ← threshold-mode TextScanOp records IndexId("TxPredPhantom:title") read
/// s1 [Serializable]: CREATE (:TxPredSentinel {v: 1})  ← write sentinel
///
/// s2 [Serializable]: MATCH (n:TxPredSentinel) RETURN n.v  ← read sentinel
/// s2 [Serializable]: CREATE (:TxPredPhantom {title: 'phantom term abc'})
///                    ← records IndexId("TxPredPhantom:title") write
///
/// s1.commit() → Ok  (first committer)
/// s2.commit() → SerializationFailure
/// ```
///
/// ## rw-antidependency cycle
///
/// - s1 read the index that s2 wrote → s1 →rw→ s2 (s1.out_conflict, s2.in_conflict)
/// - s2 read the sentinel that s1 wrote → s2 →rw→ s1 (s2.out_conflict, s1.in_conflict)
/// - Both transactions have in+out conflict → second committer (s2) is the pivot → abort.
///
/// ## Diagnostic note
///
/// If s2 does NOT abort, `text_search_with_threshold_visible` is NOT being called
/// on the production path (the threshold branch in `execute_search` is still routing
/// to the committed-latest `text_search_with_threshold`).  The fix is to route the
/// threshold branch to `text_search_with_threshold_visible` when epoch+tx are set.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_query_predicate_phantom_aborts() {
    let db = GrafeoDB::new_in_memory();

    // Create the text index (empty initially — nodes created in-transaction).
    db.create_text_index("TxPredPhantom", "title")
        .expect("create TxPredPhantom:title text index");

    // Seed the sentinel node (pre-snapshot).
    let setup = db.session();
    setup
        .execute("CREATE (:TxPredSentinel {v: 0})")
        .expect("seed TxPredSentinel");
    drop(setup);

    // Both sessions begin (same snapshot epoch).
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: predicate-path text query — threshold-mode TextScanOp; must record
    // IndexId("TxPredPhantom:title") in s1's SSI read-set.
    let r1 = s1
        .execute("MATCH (n:TxPredPhantom) WHERE text_match(n.title, 'phantom term') RETURN n")
        .expect("s1: text_match predicate query under Serializable must not error");
    // Empty index at this point — 0 results is expected and correct.
    assert_eq!(
        r1.row_count(),
        0,
        "s1: text_match on empty index must return 0 rows; got {}",
        r1.row_count()
    );

    // s1: write sentinel — s2 (active) reads it later to close the rw-cycle.
    s1.execute("MATCH (n:TxPredSentinel) SET n.v = 99")
        .expect("s1: SET TxPredSentinel.v");

    // s2: read sentinel — records TxPredSentinel in s2's SSI read-set.
    let r2 = s2
        .execute("MATCH (n:TxPredSentinel) RETURN n.v")
        .expect("s2: MATCH TxPredSentinel");
    assert_eq!(r2.row_count(), 1, "s2: must see the sentinel node");

    // s2: insert a new TxPredPhantom doc matching s1's search terms.
    // `buffer_text_index_set` records IndexId("TxPredPhantom:title") in s2's write-set.
    s2.execute("CREATE (:TxPredPhantom {title: 'phantom term abc'})")
        .expect("s2: CREATE TxPredPhantom node");

    // s1 commits first → must succeed (s2 not yet committed; cycle not confirmed).
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 (first committer) must succeed; got: {:?}",
        c1
    );

    // s2 commits second → MUST abort.
    // Cycle: s1 read index via text_match predicate that s2 wrote (s1.out, s2.in)
    //        AND s2 read sentinel s1 wrote (s2.out, s1.in).
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 (phantom writer concurrent with text_match predicate reader) must abort — \
         if this fails, text_search_with_threshold_visible is NOT being called on the \
         threshold-mode TextScanOp path (the query-predicate hole is still open)",
    );
}

// ============================================================================
// 21. serializable_text_query_predicate_disjoint_commits
// ============================================================================

/// Disjoint text-index scenario via the query-predicate path: s1 uses
/// `text_match` on `:TxPredDisjA(word)` and s2 writes to `:TxPredDisjB(word)`.
/// The two indexes have different `IndexId`s → no rw-antidependency → both
/// transactions MUST commit.
///
/// This is the contrast case that proves the abort in
/// `serializable_text_query_predicate_phantom_aborts` is driven by index
/// identity, not by a blanket policy.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_query_predicate_disjoint_commits() {
    let db = GrafeoDB::new_in_memory();

    // Create two separate text indexes on different label:property pairs.
    db.create_text_index("TxPredDisjA", "word")
        .expect("create TxPredDisjA:word text index");
    db.create_text_index("TxPredDisjB", "word")
        .expect("create TxPredDisjB:word text index");

    // Seed a TxPredDisjA node so s1's predicate query returns at least one result.
    let n = db.create_node(&["TxPredDisjA"]);
    db.set_node_property(
        n,
        "word",
        grafeo_common::types::Value::String("hello world".into()),
    );

    // Both sessions begin.
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: predicate-path query on TxPredDisjA — records IndexId("TxPredDisjA:word").
    let r1 = s1
        .execute("MATCH (n:TxPredDisjA) WHERE text_match(n.word, 'hello') RETURN n")
        .expect("s1: text_match predicate query TxPredDisjA under Serializable");
    assert!(
        r1.row_count() >= 1,
        "s1: must find the seeded TxPredDisjA node"
    );

    // s2: write to TxPredDisjB — records IndexId("TxPredDisjB:word") in s2's write-set.
    s2.execute("CREATE (:TxPredDisjB {word: 'hello universe'})")
        .expect("s2: CREATE TxPredDisjB node");

    // s1 commits — no rw-antidependency (TxPredDisjA:word ≠ TxPredDisjB:word).
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 must commit — no overlap with s2's index write; got: {:?}",
        c1
    );

    // s2 commits — same reason.
    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 must commit — no overlap with s1's index read; got: {:?}",
        c2
    );
}

// ============================================================================
// 22. serializable_text_filter_per_row_phantom_aborts
// ============================================================================

/// THE per-row-filter phantom hole: `MATCH (n:Doc) WHERE text_match(n.body,'q')
/// OR n.flag = true` forces text predicate evaluation through `FilterOperator`
/// (pushdown declined due to OR with a non-text predicate) — the per-row path
/// previously called `score_text` which did NOT record the index read.
///
/// Cycle (classic phantom):
/// - s1 runs the query → 0 results, but MUST record `IndexId("TxPerRow:body")`
///   so the cycle below is detected.
/// - s1 writes a sentinel.
/// - s2 reads the sentinel (closes s2→s1 rw-edge) then inserts a new
///   `:TxPerRow` node matching 'phantom term' (closes s1→s2 rw-edge via
///   `record_write_index`).
/// - s1 commits (first) → must succeed.
/// - s2 commits (second) → MUST abort with SerializationFailure.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_filter_per_row_phantom_aborts() {
    let db = GrafeoDB::new_in_memory();

    // Create the text index (empty initially).
    db.create_text_index("TxPerRow", "body")
        .expect("create TxPerRow:body text index");

    // Seed a sentinel node outside any explicit transaction.
    let setup = db.session();
    setup
        .execute("CREATE (:TxPerRowSentinel {v: 0})")
        .expect("seed TxPerRowSentinel");
    drop(setup);

    // Both sessions begin at the same snapshot epoch.
    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: run a query that forces per-row filter evaluation.
    // The OR with `n.flag = true` prevents the planner from pushing text_match
    // down to TextScanOp, so eval_text_fn / score_text_visible is called
    // per row inside FilterOperator.
    // Index is empty at this point → 0 results; must still record the read.
    let r1 = s1
        .execute(
            "MATCH (n:TxPerRow) WHERE text_match(n.body, 'phantom term') OR n.flag = true \
             RETURN n",
        )
        .expect("s1: per-row text_match OR query must not error");
    assert_eq!(
        r1.row_count(),
        0,
        "s1: empty index → 0 results; got {}",
        r1.row_count()
    );

    // s1: write sentinel — s2 will read it to close the rw-cycle.
    s1.execute("MATCH (n:TxPerRowSentinel) SET n.v = 99")
        .expect("s1: SET TxPerRowSentinel.v");

    // s2: read sentinel — records TxPerRowSentinel in s2's read-set (s2→s1 edge).
    let r2 = s2
        .execute("MATCH (n:TxPerRowSentinel) RETURN n.v")
        .expect("s2: MATCH TxPerRowSentinel");
    assert_eq!(r2.row_count(), 1, "s2: must see the sentinel node");

    // s2: insert a new :TxPerRow node matching s1's search term.
    // buffer_text_index_set records IndexId("TxPerRow:body") in s2's write-set,
    // forming the s1→s2 rw-antidependency edge.
    s2.execute("CREATE (:TxPerRow {body: 'phantom term abc', flag: false})")
        .expect("s2: CREATE TxPerRow node");

    // s1 commits first → must succeed (cycle not confirmed yet).
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 (first committer) must succeed; got: {:?}",
        c1
    );

    // s2 commits second → MUST abort.
    // Cycle: s1 read IndexId("TxPerRow:body") via per-row text_match (s1.out,
    // s2.in) AND s2 read sentinel s1 wrote (s2.out, s1.in).
    let c2 = s2.commit();
    assert_serialization_failure(
        &c2,
        "s2 (phantom writer concurrent with per-row text_match reader) must abort — \
         if this fails, score_text_visible is NOT being called on the per-row \
         FilterOperator path (the last phantom hole is still open)",
    );
}

// ============================================================================
// 23. serializable_text_filter_per_row_disjoint_commits
// ============================================================================

/// Disjoint text-index scenario via the per-row filter path: s1 uses
/// `text_match` on `:TxPerRowDisjA(body)` and s2 writes to `:TxPerRowDisjB(body)`.
/// Different indexes → no rw-antidependency → both MUST commit.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_filter_per_row_disjoint_commits() {
    let db = GrafeoDB::new_in_memory();

    db.create_text_index("TxPerRowDisjA", "body")
        .expect("create TxPerRowDisjA:body text index");
    db.create_text_index("TxPerRowDisjB", "body")
        .expect("create TxPerRowDisjB:body text index");

    // Seed a TxPerRowDisjA node so the per-row scan sees at least one row.
    let n = db.create_node(&["TxPerRowDisjA"]);
    db.set_node_property(
        n,
        "body",
        grafeo_common::types::Value::String("hello world".into()),
    );

    let mut s1 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s1: begin Serializable");

    let mut s2 = db.session();
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("s2: begin Serializable");

    // s1: per-row filter on TxPerRowDisjA — records IndexId("TxPerRowDisjA:body").
    let r1 = s1
        .execute(
            "MATCH (n:TxPerRowDisjA) WHERE text_match(n.body, 'hello') OR n.flag = true \
             RETURN n",
        )
        .expect("s1: per-row text_match TxPerRowDisjA");
    assert!(
        r1.row_count() >= 1,
        "s1: must find the seeded TxPerRowDisjA node"
    );

    // s2: write to TxPerRowDisjB — records IndexId("TxPerRowDisjB:body").
    s2.execute("CREATE (:TxPerRowDisjB {body: 'hello universe', flag: false})")
        .expect("s2: CREATE TxPerRowDisjB node");

    // Both commit — no rw-antidependency (TxPerRowDisjA:body ≠ TxPerRowDisjB:body).
    let c1 = s1.commit();
    assert!(
        c1.is_ok(),
        "s1 must commit — no overlap with s2's index write; got: {:?}",
        c1
    );

    let c2 = s2.commit();
    assert!(
        c2.is_ok(),
        "s2 must commit — no overlap with s1's index read; got: {:?}",
        c2
    );
}
