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
//!
//! ```bash
//! CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test serializable
//! ```

#![cfg(feature = "lpg")]

use grafeo_engine::{GrafeoDB, transaction::IsolationLevel};

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
