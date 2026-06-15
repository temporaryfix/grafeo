# Audit Wave 1 — Query/Transaction Quick-Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land six independent correctness/perf fixes from the engine audit (commit-conflict leak, NULL join keys, dropped join-tree filters, conjunction splitting, reject-Serializable, real `copy_graph`) as one TDD commit each.

**Architecture:** Each finding is a self-contained fix in `grafeo-core` (execution operators, LPG store) or `grafeo-engine` (session, optimizer). Every fix is gated by a failing test written first. No upstream-shared-primitive behavior changes beyond what each fix requires; fork-local branch `fix/audit-wave1-query-tx-correctness` off `integration`.

**Tech Stack:** Rust 2024, `cargo test`, `parking_lot`, the existing `Operator`/`LogicalOperator`/`LpgStore` abstractions.

**Spec:** `docs/superpowers/specs/2026-06-15-audit-wave1-query-tx-correctness-design.md`

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `crates/grafeo-engine/src/session/mod.rs` | Commit conflict branch: discard versions + abort tx; F1 unit test | F1 |
| `crates/grafeo-core/src/execution/operators/join.rs` | Never insert NULL hash keys; F2 unit tests | F2 |
| `crates/grafeo-engine/src/query/optimizer/mod.rs` | `collect_join_tree` keeps filtered relations / declines reorder; F3 unit test | F3 |
| `crates/grafeo-engine/src/query/optimizer/mod.rs` | `push_filters_down` splits top-level AND conjuncts | F4 |
| `crates/grafeo-engine/tests/filter_pushdown.rs` | F4 EXPLAIN/results test | F4 |
| `crates/grafeo-engine/src/session/mod.rs` | Reject `Serializable` at begin | F5 |
| `crates/grafeo-engine/tests/coverage_session.rs` | F5 test (update existing) | F5 |
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` | Real `copy_graph` deep copy | F6 |
| `crates/grafeo-core/src/graph/lpg/store/tests.rs` | F6 unit test | F6 |

Each task produces one commit. Run order F1 → F2 → F3 → F4 → F5 → F6, then Task 7 (wave verification).

---

## Task F1: Commit-conflict leak — abort + discard

**Files:**
- Modify: `crates/grafeo-engine/src/session/mod.rs` (commit conflict branch, ~lines 4014-4017, and the `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test** — add to the `#[cfg(test)] mod tests` block at the bottom of `crates/grafeo-engine/src/session/mod.rs`:

```rust
#[test]
fn failed_commit_aborts_tx_and_does_not_pin_gc() {
    // Two transactions begin at the same epoch. T1 writes+commits. T2 then
    // writes the same entity (admitted, since T1 is no longer Active) and
    // commits -> commit-time write-write conflict. The failed commit MUST
    // abort the transaction; leaving it Active pins min_active_epoch and
    // stalls MVCC GC forever.
    let db = GrafeoDB::new_in_memory();
    let mut s1 = db.session();
    s1.execute("CREATE (:Acct {id: 1, bal: 100})").unwrap();

    let mut s2 = db.session();
    s1.begin_transaction().unwrap();
    s2.begin_transaction().unwrap();

    s1.execute("MATCH (a:Acct {id: 1}) SET a.bal = 50").unwrap();
    s1.commit().unwrap();

    s2.execute("MATCH (a:Acct {id: 1}) SET a.bal = 60").unwrap();
    let r = s2.commit();
    assert!(r.is_err(), "expected a commit-time write-write conflict");

    // White-box: no zombie Active transaction left behind.
    assert_eq!(
        s2.transaction_manager.active_count(),
        0,
        "failed commit left a zombie Active transaction (GC-pinning leak)"
    );

    // Data is consistent: T1's committed value is visible exactly once.
    let s3 = db.session();
    let q = s3.execute("MATCH (a:Acct {id: 1}) RETURN a.bal").unwrap();
    assert_eq!(q.row_count(), 1);
    assert_eq!(q.rows()[0][0], grafeo_common::types::Value::Int64(50));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --lib session::tests::failed_commit_aborts_tx_and_does_not_pin_gc`
Expected: FAIL on the `active_count` assertion (`left: 1, right: 0`) — the conflict branch never aborts the tx.

- [ ] **Step 3: Apply the fix** — in the commit conflict branch, replace the property-only rollback loop and add the abort. Change:

```rust
                // Conflict detected: rollback the data changes
                for graph_name in &touched {
                    let store = self.resolve_store(graph_name);
                    store.rollback_transaction_properties(transaction_id);
                }
```

to:

```rust
                // Conflict detected: discard the transaction's uncommitted
                // (PENDING) versions and replay its property undo log, then mark
                // it aborted. rollback_transaction_properties alone leaks the
                // PENDING node/edge versions; skipping abort leaves the tx Active
                // forever, pinning min_active_epoch and stalling MVCC GC.
                for graph_name in &touched {
                    let store = self.resolve_store(graph_name);
                    store.discard_uncommitted_versions(transaction_id);
                }
                let _ = self.transaction_manager.abort(transaction_id);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --lib session::tests::failed_commit_aborts_tx_and_does_not_pin_gc`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-engine/src/session/mod.rs
git commit -m "fix(session): abort + discard uncommitted versions on commit conflict

A failed commit() only replayed the property undo log: it never discarded
the transaction's PENDING node/edge versions and never marked the tx
aborted. The zombie Active tx pinned min_active_epoch and stalled MVCC GC.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task F2: NULL keys never join (outer joins)

**Files:**
- Modify: `crates/grafeo-core/src/execution/operators/join.rs` (`build_hash_table`, ~lines 263-272, and the `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test** — add to the `#[cfg(test)] mod tests` in `join.rs`. First add two chunk helpers next to `create_int_chunk`, then the test:

```rust
fn create_nullable_int_chunk(values: &[Option<i64>]) -> DataChunk {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
    for v in values {
        match v {
            Some(x) => builder.column_mut(0).unwrap().push_int64(*x),
            None => builder.column_mut(0).unwrap().push_value(Value::Null),
        }
        builder.advance_row();
    }
    builder.finish()
}

fn create_nullable_int_chunk_2col(rows: &[(Option<i64>, Option<i64>)]) -> DataChunk {
    let mut b = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);
    for (k, p) in rows {
        match k {
            Some(x) => b.column_mut(0).unwrap().push_int64(*x),
            None => b.column_mut(0).unwrap().push_value(Value::Null),
        }
        match p {
            Some(x) => b.column_mut(1).unwrap().push_int64(*x),
            None => b.column_mut(1).unwrap().push_value(Value::Null),
        }
        b.advance_row();
    }
    b.finish()
}

#[test]
fn test_hash_join_left_outer_null_key_is_null_padded_not_matched() {
    // Left keys: [1, NULL]; Right rows: [(key=1, payload=100), (key=NULL, payload=999)].
    // LEFT join must emit (1 -> matched 100) and (NULL -> null-padded). The
    // distinguishing payload column proves the NULL left row is null-padded
    // (payload NULL), NOT matched to the right's NULL key (payload 999).
    let left = MockOperator::new(vec![create_nullable_int_chunk(&[Some(1), None])]);
    let right = MockOperator::new(vec![create_nullable_int_chunk_2col(&[
        (Some(1), Some(100)),
        (None, Some(999)),
    ])]);
    let output_schema = vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64];
    let mut join = HashJoinOperator::new(
        Box::new(left),
        Box::new(right),
        vec![0],
        vec![0],
        JoinType::Left,
        output_schema,
    );

    let mut total = 0;
    let mut payload_for_null_left: Option<Option<Value>> = None;
    while let Some(chunk) = join.next().unwrap() {
        for row in chunk.selected_indices() {
            total += 1;
            let lk = chunk.column(0).unwrap().get_value(row);
            if matches!(lk, None | Some(Value::Null)) {
                payload_for_null_left = Some(chunk.column(2).unwrap().get_value(row));
            }
        }
    }
    assert_eq!(total, 2, "LEFT join must emit each left row exactly once");
    let p = payload_for_null_left.expect("null-key left row must be present");
    assert!(
        matches!(p, None | Some(Value::Null)),
        "NULL left key must be null-padded, not matched to a NULL right key; got {p:?}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --lib execution::operators::join::tests::test_hash_join_left_outer_null_key_is_null_padded_not_matched`
Expected: FAIL — current code inserts NULL build keys for outer joins, so the NULL left row matches the NULL right row and `payload` is `999`.

- [ ] **Step 3: Apply the fix** — in `build_hash_table`, replace:

```rust
                // Skip null keys for inner/semi/anti joins
                if matches!(key, HashKey::Null)
                    && !matches!(
                        self.join_type,
                        JoinType::Left | JoinType::Right | JoinType::Full
                    )
                {
                    continue;
                }
```

with:

```rust
                // NULL never equals NULL in a join key (three-valued logic), so
                // NULL keys are never inserted into the hash table for any join
                // type. Outer-join unmatched rows are still emitted via the
                // build_matched/probe_matched tracking and the no-match null-pad
                // path, neither of which depends on NULL being in the table.
                if matches!(key, HashKey::Null) {
                    continue;
                }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --lib execution::operators::join::tests`
Expected: PASS (new test plus all existing join tests — confirm `test_hash_join_left_outer`, `_right_outer`, `_full_outer`, `_semi`, `_anti` still pass).

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/execution/operators/join.rs
git commit -m "fix(join): NULL keys never match in outer joins

build_hash_table inserted NULL build keys for Left/Right/Full joins, so a
NULL probe key matched them and NULL = NULL joined rows, violating
three-valued logic. NULL keys are now never inserted; outer-join unmatched
emission is unaffected (it runs off match-tracking, not the table).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task F3: `collect_join_tree` must not drop Filter predicates

**Files:**
- Modify: `crates/grafeo-engine/src/query/optimizer/mod.rs` (`collect_join_tree` Filter arm, ~lines 675-678, and the `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test** — add to the optimizer `#[cfg(test)] mod tests`:

```rust
#[test]
fn test_reorder_preserves_filter_on_relation() {
    // Join(Filter(a.age > 30, NodeScan a:Person), NodeScan b:Person) ON a.id = b.id.
    // Join reorder must NOT drop the a.age predicate when flattening the tree.
    let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
        items: vec![ReturnItem {
            expression: LogicalExpression::Variable("a".to_string()),
            alias: None,
        }],
        distinct: false,
        input: Box::new(LogicalOperator::Join(JoinOp {
            left: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "a".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                pushdown_hint: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "b".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Property {
                    variable: "a".to_string(),
                    property: "id".to_string(),
                },
                right: LogicalExpression::Property {
                    variable: "b".to_string(),
                    property: "id".to_string(),
                },
            }],
        })),
    }));

    let optimized = Optimizer::new().optimize(plan).unwrap();
    let tree = optimized.root.explain_tree();
    assert!(
        tree.contains("age"),
        "join reorder dropped the a.age filter predicate; plan was:\n{tree}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --lib query::optimizer::tests::test_reorder_preserves_filter_on_relation`
Expected: FAIL — `collect_join_tree` recurses into the Filter's input and discards `a.age > 30`, so `explain_tree()` no longer mentions `age`.

- [ ] **Step 3: Apply the fix** — in `collect_join_tree`, replace the Filter arm:

```rust
            LogicalOperator::Filter(filter) => {
                // A filter on a base relation is still part of the join tree
                self.collect_join_tree(&filter.input, relations, conditions)
            }
```

with:

```rust
            LogicalOperator::Filter(filter) => {
                // A filter wrapping a single base relation rides along with that
                // relation through reordering: record the whole Filter(relation)
                // as the relation entry so its predicate is never lost. If the
                // filter sits above a join (spans relations), decline to flatten
                // (return false) so reordering is skipped and the original,
                // predicate-bearing plan is kept intact.
                match filter.input.as_ref() {
                    LogicalOperator::NodeScan(scan) => {
                        relations.push((scan.variable.clone(), op.clone()));
                        true
                    }
                    LogicalOperator::EdgeScan(scan) => {
                        relations.push((scan.variable.clone(), op.clone()));
                        true
                    }
                    LogicalOperator::Expand(expand) => {
                        relations.push((expand.to_variable.clone(), op.clone()));
                        true
                    }
                    #[cfg(feature = "triple-store")]
                    LogicalOperator::TripleScan(_) => {
                        self.collect_join_tree(&filter.input, relations, conditions)
                    }
                    _ => false,
                }
            }
```

Note: `op` is the `&LogicalOperator` for this Filter; `op.clone()` keeps the filter wrapped around its scan. This mirrors the existing `NodeScan`/`EdgeScan`/`Expand` arms that push `(var, op.clone())`.

- [ ] **Step 4: Run test to verify it passes**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --lib query::optimizer::tests`
Expected: PASS (new test plus all existing optimizer tests).

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-engine/src/query/optimizer/mod.rs
git commit -m "fix(optimizer): join reorder keeps filtered relations' predicates

collect_join_tree's Filter arm recursed into the child and dropped the
predicate, so reordering a Join whose relation carried a filter silently
lost it. Now a filter on a single base relation travels with that relation;
a filter spanning a join declines the flatten so reorder is skipped.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task F4: Conjunction splitting in filter pushdown

**Files:**
- Modify: `crates/grafeo-engine/src/query/optimizer/mod.rs` (`push_filters_down` Filter arm, ~lines 890-893)
- Test: `crates/grafeo-engine/tests/filter_pushdown.rs`

- [ ] **Step 1: Write the failing test** — append to `crates/grafeo-engine/tests/filter_pushdown.rs`:

```rust
#[test]
fn conjuncts_split_and_anchor_on_their_own_scans() {
    // MATCH (a:Person),(b:City) WHERE a.name = 'Ann' AND b.name = 'Rome'
    // The conjunction must be split so each predicate anchors on its own scan,
    // instead of one combined AND filter sitting above a cartesian product.
    let db = grafeo_engine::GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    s.execute("CREATE (:Person {name: 'Bob'})").unwrap();
    s.execute("CREATE (:City {name: 'Rome'})").unwrap();
    s.execute("CREATE (:City {name: 'Oslo'})").unwrap();

    // Correctness is unchanged: exactly one (Ann, Rome) row.
    let r = s
        .execute("MATCH (a:Person),(b:City) WHERE a.name = 'Ann' AND b.name = 'Rome' RETURN a.name, b.name")
        .unwrap();
    assert_eq!(r.row_count(), 1);

    // Structure: the combined "And" filter is gone; conjuncts are split.
    let plan = s
        .execute("EXPLAIN MATCH (a:Person),(b:City) WHERE a.name = 'Ann' AND b.name = 'Rome' RETURN a.name, b.name")
        .unwrap();
    let text = format!("{:?}", plan.rows());
    assert!(
        !text.contains(" And "),
        "conjuncts should be split into per-scan filters, not kept as one AND; plan:\n{text}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --test filter_pushdown conjuncts_split_and_anchor_on_their_own_scans`
Expected: FAIL on the `" And "` assertion — the predicate is kept as one combined `a.name Eq "Ann" And b.name Eq "Rome"` filter above both scans.

- [ ] **Step 3: Apply the fix** — in `push_filters_down`, replace the Filter arm:

```rust
            // For Filter operators, try to push the predicate into the child
            LogicalOperator::Filter(filter) => {
                let optimized_input = self.push_filters_down(*filter.input);
                self.try_push_filter_into(filter.predicate, optimized_input)
            }
```

with:

```rust
            // For Filter operators, split the top-level AND chain into individual
            // conjuncts and push each independently. A conjunct that can anchor on
            // one relation no longer rides above a cartesian product just because
            // it shares a Filter with a conjunct on another relation. Conjuncts
            // that cannot be pushed re-stack as filters via try_push_filter_into.
            LogicalOperator::Filter(filter) => {
                let mut current = self.push_filters_down(*filter.input);
                for conjunct in split_conjuncts(filter.predicate) {
                    current = self.try_push_filter_into(conjunct, current);
                }
                current
            }
```

Add a module-private `split_conjuncts` helper near the bottom of `optimizer/mod.rs` (mirrors `query/translators/common.rs`; kept local to avoid a visibility change):

```rust
/// Splits a top-level conjunctive (AND-chain) predicate into individual conjuncts.
fn split_conjuncts(expr: LogicalExpression) -> Vec<LogicalExpression> {
    fn go(expr: LogicalExpression, out: &mut Vec<LogicalExpression>) {
        if let LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } = expr
        {
            go(*left, out);
            go(*right, out);
        } else {
            out.push(expr);
        }
    }
    let mut out = Vec::new();
    go(expr, &mut out);
    out
}
```

Ensure `BinaryOp` and `LogicalExpression` are in scope (they are already imported in this module).

- [ ] **Step 4: Run test to verify it passes**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --test filter_pushdown conjuncts_split_and_anchor_on_their_own_scans`
Expected: PASS.

- [ ] **Step 5: Reconcile plan-shape fallout (broadest blast radius)**

Run the full optimizer + query suites; this rewrite reshapes many plans:
`CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --lib query::optimizer 2>&1 | tail -20`
`CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --test filter_pushdown --test planner_coverage --test query_correctness 2>&1 | tail -20`
Expected: PASS. If any EXPLAIN/plan-snapshot test asserts the old combined-AND shape, update it to the split shape (results must be unchanged — never relax a result assertion to make it pass).

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-engine/src/query/optimizer/mod.rs crates/grafeo-engine/tests/filter_pushdown.rs
git commit -m "perf(optimizer): split AND conjuncts before pushdown

Filter(A AND B) was pushed as one unit, stranding a conjunct that could
anchor on one relation above a cartesian product. Split the top-level AND
into conjuncts and push each independently; each single-predicate filter
then also gets its own index/range hint in annotate_pushdown_hints.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task F5: Reject Serializable at begin

**Files:**
- Modify: `crates/grafeo-engine/src/session/mod.rs` (`begin_transaction_inner`, ~lines 3925-3929)
- Modify: `crates/grafeo-engine/tests/coverage_session.rs` (the existing Serializable begin test, ~line 454)

- [ ] **Step 1: Update the existing test to expect rejection** — in `coverage_session.rs`, find the test that calls `begin_transaction_with_isolation(... IsolationLevel::Serializable)` and expects success (~line 454) and change it to assert an error. Add a dedicated test as well:

```rust
#[test]
fn serializable_isolation_is_rejected_until_real_ssi() {
    // Serializable currently behaves as Snapshot Isolation (record_read is never
    // called, so SSI validation never fires). Until real SSI lands (Wave 2),
    // begin must reject it rather than silently downgrade.
    let db = grafeo_engine::GrafeoDB::new_in_memory();
    let mut session = db.session();
    let result = session
        .begin_transaction_with_isolation(grafeo_engine::transaction::IsolationLevel::Serializable);
    assert!(
        result.is_err(),
        "Serializable must be rejected until real SSI is implemented"
    );

    // ReadCommitted and SnapshotIsolation are unaffected.
    let mut s2 = db.session();
    assert!(
        s2.begin_transaction_with_isolation(
            grafeo_engine::transaction::IsolationLevel::SnapshotIsolation
        )
        .is_ok()
    );
}
```

For the pre-existing `begin_transaction_with_isolation(... Serializable)` assertion(s) around line 454 and the nested variant near line 466, change any `.unwrap()`/`is_ok()` expectation on the Serializable path to expect `is_err()` (or switch those cases to `SnapshotIsolation` if the test's intent is "a transaction begins", not "Serializable specifically").

- [ ] **Step 2: Run test to verify it fails**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --test coverage_session serializable_isolation_is_rejected_until_real_ssi`
Expected: FAIL — begin currently accepts Serializable and returns `Ok`.

- [ ] **Step 3: Apply the fix** — in `begin_transaction_inner`, after the nested-transaction early return and before `begin_with_isolation`, reject Serializable. Replace:

```rust
        let transaction_id = if let Some(level) = isolation_level {
            self.transaction_manager.begin_with_isolation(level)
        } else {
            self.transaction_manager.begin()
        };
```

with:

```rust
        let transaction_id = if let Some(level) = isolation_level {
            if level == crate::transaction::IsolationLevel::Serializable {
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(
                        "Serializable isolation is not yet supported; use SnapshotIsolation \
                         (real SSI is tracked for the Wave 2 isolation rework)"
                            .to_string(),
                    ),
                ));
            }
            self.transaction_manager.begin_with_isolation(level)
        } else {
            self.transaction_manager.begin()
        };
```

This sits after the nested-transaction handling, so nested begins (which ignore isolation and create savepoints) are unaffected, and it covers both the direct API and the GQL mapping (both funnel through `begin_transaction_inner`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-engine --test coverage_session`
Expected: PASS (new test + the updated existing ones).

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-engine/src/session/mod.rs crates/grafeo-engine/tests/coverage_session.rs
git commit -m "fix(session): reject Serializable isolation until real SSI exists

record_read is never called from the live path, so Serializable silently
behaved as Snapshot Isolation. begin now returns a clear error instead of
shipping a guarantee it does not deliver; real SSI is tracked for Wave 2.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task F6: Real `copy_graph` deep copy

**Files:**
- Modify: `crates/grafeo-core/src/graph/lpg/store/mod.rs` (`copy_graph`, ~lines 755-765; add `Edge`/`Node` import)
- Test: `crates/grafeo-core/src/graph/lpg/store/tests.rs`

- [ ] **Step 1: Write the failing test** — append to the `#[cfg(test)] mod` in `crates/grafeo-core/src/graph/lpg/store/tests.rs` (it already exercises `LpgStore`):

```rust
#[test]
fn copy_graph_deep_copies_nodes_edges_props_labels_and_index() {
    use grafeo_common::types::{PropertyKey, Value};

    let store = LpgStore::new().expect("arena");
    let src = store.graph_or_create("src").expect("src graph");
    let a = src.create_node_with_props(
        &["Person"],
        [("name", Value::from("Ann")), ("age", Value::from(30i64))],
    );
    let b = src.create_node_with_props(&["Person"], [("name", Value::from("Bob"))]);
    src.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);
    src.create_property_index("name");

    store.copy_graph(Some("src"), Some("dst")).expect("copy");
    let dst = store.graph("dst").expect("dst graph created");

    // Counts, labels, props.
    assert_eq!(dst.node_count(), 2);
    assert_eq!(dst.edge_count(), 1);
    let ann_ids = dst.find_nodes_by_property("name", &Value::from("Ann"));
    assert_eq!(ann_ids.len(), 1, "property index must work on the copy");
    let ann = dst.get_node(ann_ids[0]).unwrap();
    assert!(ann.labels.iter().any(|l| l.as_str() == "Person"));
    assert_eq!(
        ann.properties.get(&PropertyKey::new("age")),
        Some(&Value::Int64(30))
    );

    // Edge: type, remapped endpoints, and property carried.
    let edges: Vec<_> = dst.all_edges().collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_type.as_str(), "KNOWS");
    assert_eq!(
        edges[0].properties.get(&PropertyKey::new("since")),
        Some(&Value::Int64(2020))
    );

    // Deep copy: mutating the copy does not affect the source.
    dst.set_node_property(ann_ids[0], "age", Value::from(99i64));
    let src_ann = src.find_nodes_by_property("name", &Value::from("Ann"));
    assert_eq!(
        src.get_node(src_ann[0]).unwrap().properties.get(&PropertyKey::new("age")),
        Some(&Value::Int64(30)),
        "source must be unchanged by mutations to the copy"
    );
}

#[test]
fn copy_graph_self_copy_is_a_noop() {
    let store = LpgStore::new().expect("arena");
    let g = store.graph_or_create("g").expect("g");
    g.create_node(&["X"]);
    store.copy_graph(Some("g"), Some("g")).expect("self-copy ok");
    assert_eq!(store.graph("g").unwrap().node_count(), 1, "self-copy must not duplicate");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --lib graph::lpg::store::tests::copy_graph_deep_copies_nodes_edges_props_labels_and_index`
Expected: FAIL — `copy_graph` is a no-op, so `dst.node_count()` is `0`.

- [ ] **Step 3: Apply the fix** — add the import at the top of `store/mod.rs` (near the other `crate::graph::lpg` imports):

```rust
use crate::graph::lpg::{Edge, Node};
```

Replace the `copy_graph` body:

```rust
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) -> Result<(), AllocError> {
        let _src = match source {
            Some(n) => self.graph(n),
            None => None, // default graph
        };
        let _dest_graph = dest.map(|n| self.graph_or_create(n)).transpose()?;
        // Full graph copy is complex (requires iterating all entities).
        // For now, this creates the destination graph structure.
        // Full entity-level copy will be implemented when needed.
        Ok(())
    }
```

with:

```rust
    /// Deep-copies all data from the source graph into the destination graph.
    ///
    /// Copies nodes (labels + properties), edges (type + properties, with
    /// remapped endpoints), and property indexes. Vector and text indexes are
    /// NOT carried by the copy (they would require re-embedding / re-tokenizing
    /// every value); recreate them on the destination if needed. The copy is a
    /// true deep copy: mutating the destination never affects the source.
    /// Copying a graph onto itself is a no-op.
    ///
    /// `None` refers to this (default) store; `Some(name)` to a named graph.
    /// A missing source is treated as empty (no-op). The destination is created
    /// if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the destination store cannot be allocated.
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) -> Result<(), AllocError> {
        // Self-copy guard: copying a store onto itself would iterate the
        // destination while mutating it.
        let same = match (source, dest) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        if same {
            return Ok(());
        }

        // Resolve the source store (None = this default store). A missing named
        // source has nothing to copy.
        let src_arc;
        let src: &LpgStore = match source {
            Some(name) => match self.graph(name) {
                Some(g) => {
                    src_arc = g;
                    &src_arc
                }
                None => return Ok(()),
            },
            None => self,
        };

        // Snapshot source data into owned vectors so the source is not borrowed
        // while the destination is written.
        let nodes: Vec<Node> = src.all_nodes().collect();
        let edges: Vec<Edge> = src.all_edges().collect();
        let index_keys = src.property_index_keys();

        // Resolve-or-create the destination store (None = this default store).
        let dst_arc;
        let dst: &LpgStore = match dest {
            Some(name) => {
                dst_arc = self.graph_or_create(name)?;
                &dst_arc
            }
            None => self,
        };

        // Copy nodes, recording an old -> new id mapping for edge endpoints.
        let mut id_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        for node in nodes {
            let labels: Vec<&str> = node.labels.iter().map(arcstr::ArcStr::as_str).collect();
            let new_id = dst.create_node_with_props(&labels, node.properties);
            id_map.insert(node.id, new_id);
        }

        // Copy edges with remapped endpoints.
        for edge in edges {
            let (Some(&new_src), Some(&new_dst)) =
                (id_map.get(&edge.src), id_map.get(&edge.dst))
            else {
                continue; // endpoint not copied (should not happen for live edges)
            };
            dst.create_edge_with_props(
                new_src,
                new_dst,
                edge.edge_type.as_str(),
                edge.properties,
            );
        }

        // Re-create property indexes on the destination.
        for key in index_keys {
            dst.create_property_index(&key);
        }

        Ok(())
    }
```

Note: `node.labels.iter().map(arcstr::ArcStr::as_str)` — if `arcstr` is not a direct dependency path in this crate, use `|l| l.as_str()` instead. `FxHashMap`, `NodeId`, and `AllocError` are already imported in this module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --lib graph::lpg::store::tests::copy_graph_deep_copies_nodes_edges_props_labels_and_index graph::lpg::store::tests::copy_graph_self_copy_is_a_noop`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/graph/lpg/store/mod.rs crates/grafeo-core/src/graph/lpg/store/tests.rs
git commit -m "fix(lpg): copy_graph performs a real deep copy

LpgStore::copy_graph was a silent no-op, so CREATE GRAPH x AS COPY OF y
produced an empty graph and reported success. It now deep-copies nodes
(labels+props), edges (type+props with remapped endpoints), and property
indexes; self-copy is a no-op. Vector/text indexes are not carried (doc'd).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Wave verification + probe cleanup

**Files:**
- Modify: `crates/grafeo-engine/tests/audit_scratch.rs` (remove Wave-1 probes that are now homed; keep Wave 2/4 probes)

- [ ] **Step 1: Retire the Wave-1 probes now covered by homed tests**

In `audit_scratch.rs`, delete `failed_commit_does_not_pin_gc` (replaced by F1's `failed_commit_aborts_tx_and_does_not_pin_gc`), `join_reorder_keeps_filters`, `join_reorder_three_relations_keeps_filters`, and `explain_cross_match` (F3/F4 now have homed coverage). **Keep** `uncommitted_delete_invisible_to_others`, `uncommitted_property_write_invisible_to_others` (Wave 2) and `var_length_expand_*` (Wave 4).

- [ ] **Step 2: Full-suite verification (the wave bar)**

```bash
CARGO_INCREMENTAL=0 cargo test --all-features 2>&1 | tail -30
CARGO_INCREMENTAL=0 cargo build -p grafeo-core --no-default-features --features lpg 2>&1 | tail -5
CARGO_INCREMENTAL=0 cargo build -p grafeo-core --no-default-features --features analytics 2>&1 | tail -5
CARGO_INCREMENTAL=0 cargo build -p grafeo-bindings-wasm 2>&1 | tail -5
```
Expected: all green / compile clean. (Exact wasm crate name: confirm via `cargo metadata`; adjust if it differs.)

- [ ] **Step 3: Clippy — zero new warnings**

```bash
CARGO_INCREMENTAL=0 cargo clippy --all-features 2>&1 | grep -c "warning:"
```
Expected: no new warnings introduced by the wave (compare against a pre-wave baseline if needed).

- [ ] **Step 4: Commit cleanup + update CHANGELOG**

Add a `Fixed`/`Performance` block to `CHANGELOG.md` summarizing F1–F6, then:

```bash
git add crates/grafeo-engine/tests/audit_scratch.rs CHANGELOG.md
git commit -m "test+docs: retire homed Wave-1 probes; changelog for audit Wave 1

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push branch + open PR (fork-local) — only on user go-ahead**

OPSEC-scan the diff for any private downstream schema/business logic before pushing. Then push `fix/audit-wave1-query-tx-correctness` and open a PR against `integration` (fork-local), summarizing the six findings.

---

## Self-Review (completed by plan author)

- **Spec coverage:** F1=1d, F2=1e, F3=2e, F4=conjunction-split, F5=1f, F6=1g — all six Wave-1 findings have a task. Edge-type case (dropped) and compressed-column (Wave 5) are correctly absent. ✓
- **Placeholder scan:** every code step shows the actual code/edit; no TBD/TODO. The two adaptation notes (`arcstr::ArcStr::as_str` fallback; wasm crate name) are explicit fallbacks, not placeholders. ✓
- **Type consistency:** `discard_uncommitted_versions`, `active_count`, `create_node_with_props`/`create_edge_with_props`, `all_nodes`/`all_edges`, `property_index_keys`, `JoinOp`/`JoinCondition`/`FilterOp`/`NodeScanOp` fields, `explain_tree()`, `split_conjuncts`/`BinaryOp::And` all match the verified sources. ✓
