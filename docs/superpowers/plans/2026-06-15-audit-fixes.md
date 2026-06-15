# Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate the 10 findings from the 2026-06-13 branch-diff audit with root-cause fixes, one commit per finding, all on `fix/audit-followups`.

**Architecture:** Five clusters. A (layered-store tier invariant) lands first and unblocks the extract_subgraph fix. B–E are independent. Every fix is TDD: red test → minimal change → green → commit.

**Tech Stack:** Rust workspace (`grafeo-core`, `grafeo-engine`, `crates/bindings/wasm`). Tests via `cargo test`. Spec: `docs/superpowers/specs/2026-06-14-audit-fixes-design.md`.

**Preconditions:** On branch `fix/audit-followups` (already created; spec committed as `2759aa79`). The pre-existing untracked `crates/grafeo-engine/tests/audit_scratch.rs` is unrelated — do not stage or delete it.

---

## File Structure

| File | Responsibility | Cluster |
|---|---|---|
| `crates/grafeo-core/src/graph/compact/layered.rs` | Tier-merge invariant: `merged_edges` helper, `neighbors` delegation, complete deletes | A1, A2 |
| `crates/grafeo-engine/src/database/mod.rs` | `read_graph_view` accessor; A4 accessor audit | A3, A4 |
| `crates/grafeo-engine/src/database/persistence.rs` | `extract_subgraph`/`remove_orphan_edges` via merged view; `union_index_metadata` conflict rejection; open_multi docs | A3, C1, C2 |
| `crates/grafeo-engine/src/query/planner/lpg/project.rs` | TopK structural sort-key match | B |
| `crates/grafeo-core/src/codec/bitstream.rs` | zigzag via `delta::zigzag_encode/decode` | D1 |
| `crates/bindings/wasm/src/codecs.rs` | f64 id surface (RaBitQ + WebGraph) | D2 |
| `crates/grafeo-core/src/codec/fsst.rs` | first-byte index + borrowing `train` | E1 |
| `crates/grafeo-core/src/index/vector/rabitq.rs` | scratch-buffer reuse + selection instead of full sort | E2 |
| Test files (new/edited) | per-task, listed inline | all |

---

## Task A1: neighbors() respects edge tombstones (fixes #2)

**Files:**
- Modify: `crates/grafeo-core/src/graph/compact/layered.rs` (`neighbors` at 577-612; add helper)
- Test: same file, `mod tests` (fixture `build_test_layered` at 1503)

- [ ] **Step 1: Write the failing test**

Add to the `mod tests` block in `layered.rs` (e.g. after the §R traversal tests near line 3500):

```rust
#[test]
fn neighbors_excludes_target_of_deleted_base_edge() {
    let layered = build_test_layered();
    let person = layered.nodes_by_label("Person")[0];
    let (target, eid) = layered.edges_from(person, Direction::Outgoing)[0];

    // Delete the (un-promoted) base edge. edges_from already drops it...
    assert!(layered.delete_edge(eid));
    assert!(layered.edges_from(person, Direction::Outgoing).is_empty());

    // ...but neighbors() must agree: no target reachable only via a deleted edge.
    assert!(
        !layered.neighbors(person, Direction::Outgoing).contains(&target),
        "neighbors() reported a target whose only edge was deleted"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-core --all-features neighbors_excludes_target_of_deleted_base_edge`
Expected: FAIL — `neighbors()` still returns `target` (it never consults `deleted_from_base_edges`).

- [ ] **Step 3: Rewrite `neighbors` to delegate to `edges_from`**

Replace the entire `neighbors` body (layered.rs:577-612) with:

```rust
fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
    // Single source of truth for tier-merged adjacency: derive neighbors
    // from edges_from so the node AND edge tombstone filters (and overlay
    // promotion) are applied in exactly one place.
    let mut targets: Vec<NodeId> = self
        .edges_from(node, direction)
        .into_iter()
        .map(|(target, _eid)| target)
        .collect();
    targets.sort_unstable();
    targets.dedup();
    targets
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p grafeo-core --all-features neighbors_excludes_target_of_deleted_base_edge`
Expected: PASS

- [ ] **Step 5: Run the full layered suite for regressions**

Run: `cargo test -p grafeo-core --all-features --lib graph::compact::layered`
Expected: PASS (all existing `neighbors`/traversal tests still green)

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/graph/compact/layered.rs
git commit -m "fix(compact): neighbors() respects per-edge base tombstones

neighbors() hand-rolled its own base+overlay merge that filtered
deleted_from_base_nodes but never deleted_from_base_edges, so a
target reachable only via a deleted base edge was still reported.
Delegate to edges_from() so the tombstone filters live in one place.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task A2: deletes tombstone the base copy of promoted entities (fixes #1)

**Files:**
- Modify: `crates/grafeo-core/src/graph/compact/layered.rs` (`delete_edge` 1270-1282, `delete_edge_versioned` 1284-1304, `delete_node` 1213-1226, `delete_node_versioned` 1228-1248)
- Test: same file, `mod tests`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn delete_promoted_edge_tombstones_base() {
    let layered = build_test_layered();
    let person = layered.nodes_by_label("Person")[0];
    let (target, eid) = layered.edges_from(person, Direction::Outgoing)[0];

    // Promote the edge into the overlay (copies it; base copy remains).
    layered.set_edge_property(eid, "weight", Value::Int64(5));
    assert!(layered.is_edge_dirty(eid));

    // Delete it. Every read path must agree it is gone.
    assert!(layered.delete_edge(eid));
    assert!(layered.get_edge(eid).is_none(), "deleted promoted edge still resolves");
    assert!(layered.edges_from(person, Direction::Outgoing).is_empty());
    assert!(!layered.neighbors(person, Direction::Outgoing).contains(&target));
    // Idempotent: nothing left to delete.
    assert!(!layered.delete_edge(eid));
}

#[test]
fn delete_promoted_node_tombstones_base() {
    let layered = build_test_layered();
    let person = layered.nodes_by_label("Person")[0];

    // Promote the node (labels/properties copied to overlay; base adjacency stays).
    layered.set_node_property(person, "nick", Value::from("x"));
    assert!(layered.is_node_dirty(person));

    assert!(layered.delete_node(person));
    assert!(layered.get_node(person).is_none(), "deleted promoted node still resolves");
    // Its base edges must not resurface through the deleted node.
    assert!(layered.edges_from(person, Direction::Outgoing).is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p grafeo-core --all-features delete_promoted`
Expected: FAIL — `delete_edge`/`delete_node` take the `is_*_dirty` branch and only delete the overlay copy; the base copy is never tombstoned, so `get_edge`/`edges_from` still surface it.

- [ ] **Step 3: Make `delete_edge` delete overlay AND tombstone base**

Replace `delete_edge` (layered.rs:1270-1282) with:

```rust
fn delete_edge(&self, id: EdgeId) -> bool {
    let _guard = self.merge_guard.read();
    // Delete the overlay copy if present, and independently tombstone the
    // base copy if present. A promoted edge lives in both tiers, so both
    // must happen; a fresh overlay-only edge has no base copy.
    let overlay_removed = self.overlay.load().delete_edge(id);
    let base_tombstoned = self.base.load().get_edge(id).is_some()
        && self.deleted_from_base_edges.write().insert(id);
    if base_tombstoned {
        self.deletions_dirty.store(true, Ordering::Release);
    }
    overlay_removed || base_tombstoned
}
```

- [ ] **Step 4: Apply the same shape to `delete_edge_versioned`**

Replace `delete_edge_versioned` (layered.rs:1284-1304) with:

```rust
fn delete_edge_versioned(
    &self,
    id: EdgeId,
    epoch: EpochId,
    transaction_id: TransactionId,
) -> bool {
    let _guard = self.merge_guard.read();
    let overlay_removed = self
        .overlay
        .load()
        .delete_edge_versioned(id, epoch, transaction_id);
    let base_tombstoned = self.base.load().get_edge(id).is_some()
        && self.deleted_from_base_edges.write().insert(id);
    if base_tombstoned {
        self.deletions_dirty.store(true, Ordering::Release);
    }
    overlay_removed || base_tombstoned
}
```

- [ ] **Step 5: Apply the same shape to `delete_node` and `delete_node_versioned`**

Replace `delete_node` (layered.rs:1213-1226) with:

```rust
fn delete_node(&self, id: NodeId) -> bool {
    let _guard = self.merge_guard.read();
    let overlay_removed = self.overlay.load().delete_node(id);
    let base_tombstoned = self.base.load().get_node(id).is_some()
        && self.deleted_from_base_nodes.write().insert(id);
    if base_tombstoned {
        self.deletions_dirty.store(true, Ordering::Release);
    }
    overlay_removed || base_tombstoned
}
```

Replace `delete_node_versioned` (layered.rs:1228-1248) with:

```rust
fn delete_node_versioned(
    &self,
    id: NodeId,
    epoch: EpochId,
    transaction_id: TransactionId,
) -> bool {
    let _guard = self.merge_guard.read();
    let overlay_removed = self
        .overlay
        .load()
        .delete_node_versioned(id, epoch, transaction_id);
    let base_tombstoned = self.base.load().get_node(id).is_some()
        && self.deleted_from_base_nodes.write().insert(id);
    if base_tombstoned {
        self.deletions_dirty.store(true, Ordering::Release);
    }
    overlay_removed || base_tombstoned
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p grafeo-core --all-features delete_promoted`
Expected: PASS (both tests)

- [ ] **Step 7: Run the full layered + compact suite for regressions**

Run: `cargo test -p grafeo-core --all-features --lib graph::compact`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add crates/grafeo-core/src/graph/compact/layered.rs
git commit -m "fix(compact): deletes tombstone the base copy of promoted entities

delete_edge/delete_node (and _versioned) branched either/or on
is_*_dirty: a promoted entity only had its overlay copy removed, so
its base copy resurfaced in get_*/edges_from/neighbors and a second
delete returned false. Delete the overlay copy and independently
tombstone the base copy when present.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task A3: extract_subgraph / remove_orphan_edges read the merged view (fixes #3)

**Files:**
- Modify: `crates/grafeo-engine/src/database/mod.rs` (add `read_graph_view` near `graph_store_ref` at 228)
- Modify: `crates/grafeo-engine/src/database/persistence.rs` (`extract_subgraph` 1407-1537, `remove_orphan_edges` 1554-end)
- Test: new `crates/grafeo-engine/tests/extract_subgraph_post_compact.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/grafeo-engine/tests/extract_subgraph_post_compact.rs`:

```rust
//! Regression: extract_subgraph/remove_orphan_edges must read through the
//! LayeredStore after compact(), not the overlay-only LpgStore.
#![cfg(all(feature = "lpg", feature = "compact-store"))]

use grafeo_engine::GrafeoDB;

#[test]
fn extract_subgraph_sees_base_tier_after_compact() {
    let db = GrafeoDB::new_in_memory();
    {
        let session = db.session();
        // Single-statement node+edge insert (confirmed GQL pattern); edges
        // between *existing* nodes would instead use `MATCH ... CREATE`.
        session
            .execute("INSERT (:A {k: 1})-[:T]->(:B {k: 2})")
            .unwrap();
    }
    db.compact().unwrap();

    // The 'A' node now lives in the compacted base tier.
    let layered = db.layered_store().expect("compacted DB has a layered store");
    let a_nodes = layered.nodes_by_label("A");
    assert_eq!(a_nodes.len(), 1);

    // Pre-fix: extract_subgraph reads lpg_store() (overlay only) → base node
    // "does not exist" → Err; its outgoing edge is silently dropped.
    let extract = db
        .extract_subgraph(&a_nodes)
        .expect("base-tier node must be extractable after compact");
    assert_eq!(
        extract.edge_count(),
        1,
        "the base-tier node's outgoing edge must survive the extract"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-engine --all-features extract_subgraph_sees_base_tier_after_compact`
Expected: FAIL — `extract_subgraph` returns `Err("... does not exist in source database")`.

- [ ] **Step 3: Add the `read_graph_view` accessor**

In `crates/grafeo-engine/src/database/mod.rs`, add immediately after `lpg_store` (after line 211):

```rust
    /// Returns the active **read** graph view, tier-merged.
    ///
    /// After [`compact()`](Self::compact) this is the `LayeredStore`
    /// (columnar base + overlay); otherwise the built-in `LpgStore`.
    /// Use this for whole-graph reads that must see both tiers
    /// (`extract_subgraph`, `remove_orphan_edges`); `lpg_store()` alone
    /// is overlay-only post-compact.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    fn read_graph_view(&self) -> &dyn grafeo_core::graph::GraphStore {
        if let Some(ref layered) = self.layered_store {
            &***layered
        } else {
            &**self.lpg_store()
        }
    }

    /// Non-compact builds: the read view is always the built-in store.
    #[cfg(all(not(feature = "compact-store"), feature = "lpg"))]
    fn read_graph_view(&self) -> &dyn grafeo_core::graph::GraphStore {
        &**self.lpg_store()
    }
```

- [ ] **Step 4: Point `extract_subgraph` and `remove_orphan_edges` at it**

In `crates/grafeo-engine/src/database/persistence.rs`:

`extract_subgraph` — change line 1408 from:
```rust
        let store = self.lpg_store();
```
to:
```rust
        let store = self.read_graph_view();
```

`remove_orphan_edges` — change line 1555 from:
```rust
        let store = self.lpg_store();
```
to:
```rust
        let store = self.read_graph_view();
```

Both functions only *read* through `store` (`get_node`, `edges_from`, `get_edge`) and write to a separate `target_store` / via `self.delete_edge`, so widening `store` to `&dyn GraphStore` is sufficient. If the compiler reports a method on `store` that `GraphStore` does not expose, that call is also tier-sensitive — route it through `GraphStore` rather than reverting to `lpg_store()`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p grafeo-engine --all-features extract_subgraph_sees_base_tier_after_compact`
Expected: PASS

- [ ] **Step 6: Run the existing extract/persistence suites**

Run: `cargo test -p grafeo-engine --all-features extract_subgraph remove_orphan_edges`
Expected: PASS (existing `crates/grafeo-engine/tests/extract_subgraph.rs` and the in-module `remove_orphan_edges_*` tests stay green)

- [ ] **Step 7: Commit**

```bash
git add crates/grafeo-engine/src/database/mod.rs crates/grafeo-engine/src/database/persistence.rs crates/grafeo-engine/tests/extract_subgraph_post_compact.rs
git commit -m "fix(persistence): extract_subgraph reads the merged tier view

extract_subgraph and remove_orphan_edges read self.lpg_store(), which
post-compact() is the overlay tier only: base-tier nodes errored as
'does not exist' and promoted nodes' base edges were silently dropped.
Add a read_graph_view() accessor (LayeredStore when compacted) and
route both functions through it.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task A4: audit the other store accessors (investigate, fix if reachable)

**Files:**
- Investigate: `crates/grafeo-engine/src/database/mod.rs` (`graph_store_ref` 228, `graph_store` 2122, `graph_store_mut` 2140)
- Possibly modify: same file
- Possibly test: new test if a reachable post-compact path is found

- [ ] **Step 1: Trace callers**

```bash
cd /Users/chris/TMP/grafeo
grep -rn "graph_store_ref()\|\.graph_store()\|graph_store_mut()" crates/grafeo-engine/src --include=*.rs | grep -v "fn graph_store"
```
For each caller, determine whether it can run **after `compact()`** (i.e. when `self.layered_store` is `Some`). The session/query path overrides stores with the `LayeredStore` (mod.rs:1789) and does **not** use these accessors — focus on vector/text/embed search entry points that call `graph_store_ref()`.

- [ ] **Step 2a: If a reachable post-compact path exists — write a failing test**

Create `crates/grafeo-engine/tests/graph_store_post_compact.rs` exercising that path after `compact()` and asserting a base-tier node is returned (mirror the A3 test shape against the specific search API found). Run it; expect FAIL (overlay-only result misses the base node).

- [ ] **Step 2b: If NO reachable post-compact path exists — document and stop**

Add a doc line to `graph_store_ref` (and `graph_store`/`graph_store_mut`) noting they intentionally return the overlay store and that compacted reads must use `read_graph_view`/the session override, then skip to Step 4.

- [ ] **Step 3: Fix the reachable accessor(s)**

Route the read accessors through the same merged view. For `graph_store_ref`:

```rust
    fn graph_store_ref(&self) -> &dyn grafeo_core::graph::GraphStore {
        if let Some(ref ext_read) = self.external_read_store {
            ext_read.as_ref()
        } else {
            #[cfg(all(feature = "compact-store", feature = "lpg"))]
            {
                return self.read_graph_view();
            }
            #[cfg(all(not(feature = "compact-store"), feature = "lpg"))]
            {
                &**self.lpg_store()
            }
            #[cfg(not(feature = "lpg"))]
            unreachable!("no graph store available: enable the `lpg` feature or use with_store()")
        }
    }
```
Re-run the Step 2a test; expect PASS. (For `graph_store()`/`graph_store_mut()`, which return `Arc<dyn ...>`, clone the `layered_store` Arc when present — mirror the session path at mod.rs:1789-1798.)

- [ ] **Step 4: Verify and commit**

Run: `cargo test -p grafeo-engine --all-features` (subset touching search + compact)
Expected: PASS

```bash
git add crates/grafeo-engine/src/database/mod.rs
# include the new test file if Step 2a was taken
git commit -m "fix(database): route post-compact read accessors through the merged view

Audit of graph_store_ref/graph_store/graph_store_mut for the same
overlay-only-post-compact defect as extract_subgraph. <Reachable: routed
through read_graph_view + regression test | Not reachable: documented
that these return the overlay tier by design>.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task B: TopK sort-key resolution matches expressions, not name strings (fixes #4)

**Files:**
- Modify: `crates/grafeo-engine/src/query/planner/lpg/project.rs` (`try_heap_topk_rewrite` 1249-1306; add matcher + structural resolver; retire name-based path)
- Test: `crates/grafeo-engine/tests/topk_rewrite.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/grafeo-engine/tests/topk_rewrite.rs`:

```rust
#[test]
fn topk_alias_collision_orders_by_real_sort_key() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // foo descends as age ascends, so a wrong-column sort is detectable.
    for i in 0..10i64 {
        session
            .execute(&format!("INSERT (:P {{foo: {}, age: {}}})", 100 - i, i))
            .unwrap();
    }
    // Alias "n_age" collides with resolved_column_name(n.age) = "n_age".
    let result = session
        .execute("MATCH (n:P) RETURN n.foo AS n_age ORDER BY n.age LIMIT 5")
        .unwrap();
    let got: Vec<i64> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::Int64(v) => *v,
            other => panic!("expected Int64, got {other:?}"),
        })
        .collect();
    // ORDER BY n.age ASC LIMIT 5 → ages 0..4 → their foo values 100..96.
    assert_eq!(
        got,
        vec![100, 99, 98, 97, 96],
        "rows must be ordered by n.age, not by the colliding alias column"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-engine --all-features topk_alias_collision_orders_by_real_sort_key`
Expected: FAIL — got `[91, 92, 93, 94, 95]` (sorted by `n.foo`).

- [ ] **Step 3: Add the conservative structural matcher + resolver**

In `crates/grafeo-engine/src/query/planner/lpg/project.rs`, add these free functions next to `resolve_logical_to_physical_keys` (near line 1381):

```rust
/// Conservative structural equality for the two expression shapes the TopK
/// rewrite can fuse on. Anything else returns false → the rewrite bails to
/// the (correct) unfused sort. Deliberately does NOT compare aliases or
/// formatted names, which is what let a user alias collide with a synthetic
/// `{var}_{prop}` column name.
fn sort_key_matches_projection(key: &LogicalExpression, projected: &LogicalExpression) -> bool {
    match (key, projected) {
        (LogicalExpression::Variable(a), LogicalExpression::Variable(b)) => a == b,
        (
            LogicalExpression::Property { variable: kv, property: kp },
            LogicalExpression::Property { variable: pv, property: pp },
        ) => kv == pv && kp == pp,
        _ => false,
    }
}

/// Resolves sort keys to physical column indices by structural match against
/// the projected expressions. Returns `None` (→ skip the rewrite) if any key
/// is not structurally present among the projected columns.
fn resolve_sort_keys_structurally(
    keys: &[crate::query::plan::SortKey],
    projected: &[(LogicalExpression, usize)],
) -> Option<Vec<grafeo_core::execution::operators::SortKey>> {
    use crate::query::plan::{NullsOrdering, SortOrder};
    use grafeo_core::execution::operators::{NullOrder, SortDirection, SortKey as PhysSortKey};

    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let column = projected
            .iter()
            .find(|(expr, _)| sort_key_matches_projection(&key.expression, expr))
            .map(|(_, idx)| *idx)?;
        let direction = match key.order {
            SortOrder::Ascending => SortDirection::Ascending,
            SortOrder::Descending => SortDirection::Descending,
        };
        let null_order = match key.nulls {
            Some(NullsOrdering::First) => NullOrder::NullsFirst,
            Some(NullsOrdering::Last) | None => NullOrder::NullsLast,
        };
        out.push(PhysSortKey { column, direction, null_order });
    }
    Some(out)
}
```

- [ ] **Step 4: Rewrite the resolution in `try_heap_topk_rewrite`**

In `try_heap_topk_rewrite`, replace the block from `let Some(predicted_columns) = predict_subtree_columns(...)` through the `let Ok(physical_keys) = resolve_logical_to_physical_keys(...)` guard (project.rs:1262-1288) with:

```rust
        // Only a bare Return is predicted/fused. Other shapes fall through.
        let LogicalOperator::Return(ret) = sort.input.as_ref() else {
            return Ok(None);
        };
        // RETURN * expands from input columns inside plan_return_projection;
        // its output count is unknown without planning the input — skip.
        if ret.items.len() == 1
            && matches!(&ret.items[0].expression, LogicalExpression::Variable(n) if n == "*")
        {
            return Ok(None);
        }

        // Match each sort key to a projected column by EXPRESSION identity,
        // not formatted name — a user alias must never satisfy a sort key it
        // does not actually project.
        let projected: Vec<(LogicalExpression, usize)> = ret
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| (item.expression.clone(), i))
            .collect();
        let Some(physical_keys) = resolve_sort_keys_structurally(&sort.keys, &projected) else {
            return Ok(None);
        };

        // Predicted output column names (for the operator schema + drift check).
        let predicted_columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| output_column_name(item.alias.as_deref(), &item.expression))
            .collect();
```

The remainder of the function (the `plan_operator` call, `debug_assert_eq!`, schema derivation, and `TopKOperator::new`) is unchanged.

- [ ] **Step 5: Remove now-dead helpers if unused**

```bash
cargo build -p grafeo-engine --all-features 2>&1 | grep -E "never used|dead_code" || echo "no dead-code warnings"
```
If `predict_subtree_columns` and/or `register_return_property_sort_aliases` are now reported unused, delete them (and the `resolve_logical_to_physical_keys` fn if it too is now unused). If still used elsewhere, leave them.

- [ ] **Step 6: Run the new test + the full TopK suite**

Run: `cargo test -p grafeo-engine --all-features --test topk_rewrite`
Expected: PASS (new collision test + all existing fire/fall-through cases, including `RETURN n.title ORDER BY n.title LIMIT k`)

- [ ] **Step 7: Commit**

```bash
git add crates/grafeo-engine/src/query/planner/lpg/project.rs crates/grafeo-engine/tests/topk_rewrite.rs
git commit -m "fix(planner): TopK matches sort keys by expression, not name string

The heap TopK rewrite resolved sort keys to projected columns by
formatted name (resolved_column_name), so a user alias colliding with a
synthetic {var}_{prop} name (RETURN n.foo AS n_age ORDER BY n.age) fused
a sort on the wrong column. Match the sort key's LogicalExpression
structurally against the projected expressions; unmatched keys bail to
the correct unfused sort.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task C1: open_multi rejects conflicting index configs (fixes #6)

**Files:**
- Modify: `crates/grafeo-engine/src/database/persistence.rs` (`union_index_metadata` 994-1050 → returns `Result`; call site 1920; `restore_indexes_from_snapshot` 883-917 + caller 1725 for the secondary hardening)
- Test: `crates/grafeo-engine/tests/snapshot_multi.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/grafeo-engine/tests/snapshot_multi.rs` (uses the `TestSnapshot*` bincode mirror already in that file; follow the existing test pattern there for constructing two blobs that each declare a vector index on the same `(label, property)` with different `dimensions`). Skeleton:

```rust
#[test]
#[cfg(feature = "vector-index")]
fn open_multi_rejects_conflicting_vector_index_dimensions() {
    // snap_a: vector index :Doc(embedding) dimensions=4
    // snap_b: vector index :Doc(embedding) dimensions=8
    let snap_a = build_snapshot_with_vector_index("Doc", "embedding", 4);
    let snap_b = build_snapshot_with_vector_index("Doc", "embedding", 8);

    let err = GrafeoDB::open_multi([snap_a, snap_b])
        .expect_err("conflicting index dimensions must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("conflicting") && msg.contains("embedding"),
        "error must name the conflicting index, got: {msg}"
    );
}
```

Add a `build_snapshot_with_vector_index(label, property, dims)` helper alongside the existing `TestSnapshot` builders in the file (construct a `TestSnapshot` whose `indexes.vector_indexes` holds one descriptor, bincode-encode with the leading version byte, return `Vec<u8>`). If the existing helpers already expose a vector-index path, reuse it instead of adding a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-engine --all-features open_multi_rejects_conflicting_vector_index_dimensions`
Expected: FAIL — `open_multi` returns `Ok` (first-wins keeps dims=4 silently).

- [ ] **Step 3: Make `union_index_metadata` fallible and conflict-checking**

In `persistence.rs`, change the signature (line 994) to:
```rust
fn union_index_metadata(snapshots: &[Snapshot]) -> Result<SnapshotIndexes> {
```
Replace the vector-index block (1005-1025) with order-preserving, conflict-checking logic:

```rust
    #[cfg(feature = "vector-index")]
    let mut vector_seen: HashMap<(String, String), usize> = HashMap::new();
    #[cfg(feature = "vector-index")]
    let mut vector_indexes: Vec<SnapshotVectorIndex> = Vec::new();
    #[cfg(feature = "vector-index")]
    for (snap_idx, snap) in snapshots.iter().enumerate() {
        for vi in &snap.indexes.vector_indexes {
            let key = (vi.label.clone(), vi.property.clone());
            if let Some(&existing_idx) = vector_seen.get(&key) {
                let existing = &vector_indexes[existing_idx];
                if existing.dimensions != vi.dimensions
                    || existing.metric.name() != vi.metric.name()
                    || existing.m != vi.m
                    || existing.ef_construction != vi.ef_construction
                {
                    return Err(Error::Internal(format!(
                        "open_multi: vector index :{}({}) declared with conflicting configuration \
                         (existing dims={}/metric={}/m={}/ef={}; snapshot[{}] dims={}/metric={}/m={}/ef={})",
                        vi.label, vi.property,
                        existing.dimensions, existing.metric.name(), existing.m, existing.ef_construction,
                        snap_idx, vi.dimensions, vi.metric.name(), vi.m, vi.ef_construction,
                    )));
                }
            } else {
                vector_seen.insert(key, vector_indexes.len());
                vector_indexes.push(SnapshotVectorIndex {
                    label: vi.label.clone(),
                    property: vi.property.clone(),
                    dimensions: vi.dimensions,
                    metric: vi.metric,
                    m: vi.m,
                    ef_construction: vi.ef_construction,
                });
            }
        }
    }
    #[cfg(not(feature = "vector-index"))]
    let vector_indexes = Vec::new();
```
Leave the property-index and text-index blocks as-is (no extra config → no conflict possible), and change the final return to:
```rust
    Ok(SnapshotIndexes {
        property_indexes,
        vector_indexes,
        text_indexes,
    })
```

- [ ] **Step 4: Propagate at the call site**

In `open_multi_with`, change line 1920 from:
```rust
            restore_indexes_from_snapshot(&db, &union_index_metadata(&decoded));
```
to:
```rust
            restore_indexes_from_snapshot(&db, &union_index_metadata(&decoded)?);
```

- [ ] **Step 5: Secondary hardening — surface rebuild failures on the multi path**

Change `restore_indexes_from_snapshot` (883) to return `Result<()>`, replacing each `grafeo_warn!` swallow with an early `return Err(Error::Internal(...))` carrying the same message. Then:
- multi path (line 1920, now inside `open_multi_with`): add `?` → `restore_indexes_from_snapshot(&db, &union_index_metadata(&decoded)?)?;`
- single-snapshot path (line 1725, inside `import_snapshot`/`restore_snapshot`): preserve current leniency:
```rust
        if let Err(e) = restore_indexes_from_snapshot(&db, &snapshot.indexes) {
            grafeo_warn!("index restore: {e}");
        }
```

- [ ] **Step 6: Run the new test + the snapshot suites**

Run: `cargo test -p grafeo-engine --all-features open_multi_rejects_conflicting_vector_index_dimensions`
Expected: PASS
Run: `cargo test -p grafeo-engine --all-features --test snapshot_multi --test snapshot_multi_diagnostics`
Expected: PASS (identical-config sibling extracts still merge)

- [ ] **Step 7: Commit**

```bash
git add crates/grafeo-engine/src/database/persistence.rs crates/grafeo-engine/tests/snapshot_multi.rs
git commit -m "fix(persistence): open_multi rejects conflicting index configs

union_index_metadata deduped vector indexes by (label, property)
first-wins, silently dropping a later snapshot's differing config; the
merged DB then failed to rebuild the whole index (warn-only) and
returned Ok with no index. Reject any differing config for the same
key at merge time, and surface rebuild failures on the multi path.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task C2: correct the open_multi rustdoc (fixes #10)

**Files:**
- Modify: `crates/grafeo-engine/src/database/persistence.rs` (doc comments 1736-1787)

- [ ] **Step 1: Replace the stale enforcement bullets**

In the `open_multi` doc comment, replace the five bullets at 1736-1752 with their now-true forms (drop every "(not yet enforced; see Task N…)" parenthetical):

```rust
    /// - A NodeId (or EdgeId) appearing in two snapshots is rejected as a
    ///   producer bug; the caller must emit disjoint subsets.
    /// - Every edge endpoint must exist somewhere in the union; a chunk may
    ///   carry edges whose endpoints belong to a different chunk.
    /// - Schema catalogs are reconciled per `OpenMultiOptions::schema_policy`
    ///   (default `UnionWithConflictCheck`): same-named types must match;
    ///   incompatible definitions are rejected.
    /// - Indexes are unioned across snapshots; a conflicting configuration for
    ///   the same `(label, property)` is rejected. The epoch is the maximum
    ///   across inputs.
    /// - At most one snapshot may carry named graphs or RDF triples.
```

- [ ] **Step 2: Fix the `open_multi_with` Panics section**

Replace the `# Panics` section at 1784-1787 with:
```rust
    /// # Errors
    ///
    /// Returns an error if `snapshots` is empty, in addition to the
    /// conditions listed on [`open_multi`](Self::open_multi).
```
(Remove the `# Panics` heading entirely — the function returns `Err`, it does not panic.)

- [ ] **Step 3: Verify docs build**

Run: `cargo doc -p grafeo-engine --all-features --no-deps 2>&1 | tail -5`
Expected: builds with no new warnings.
Run: `cargo test -p grafeo-engine --all-features --doc`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-engine/src/database/persistence.rs
git commit -m "docs(persistence): correct stale open_multi rustdoc

Remove five '(not yet enforced; see Task N)' parentheticals for
validation that is now implemented, describe the actual index-union +
conflict-rejection behavior, and replace open_multi_with's bogus
'# Panics if empty' with the real Err return.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task D1: zigzag-gamma uses the canonical fold (fixes #7)

**Files:**
- Modify: `crates/grafeo-core/src/codec/bitstream.rs` (`write_zigzag_gamma` 85-95, `read_zigzag_gamma` 179-192)
- Test: same file, `mod tests`

- [ ] **Step 1: Write the failing tests**

In the `mod tests` block of `bitstream.rs`, add:

```rust
#[test]
fn zigzag_gamma_round_trips_full_i64_domain() {
    let cases = [
        0i64, 1, -1, 2, -2, 1000, -1000,
        i64::MAX, i64::MIN + 1, i32::MIN as i64, i32::MAX as i64,
    ];
    for &n in &cases {
        let mut w = BitWriter::new();
        w.write_zigzag_gamma(n);
        let (bytes, bits) = w.into_bytes();
        let mut r = BitReader::new(&bytes, bits);
        assert_eq!(r.read_zigzag_gamma(), Some(n), "round-trip failed for {n}");
    }
}

#[test]
#[should_panic(expected = "zigzag-gamma overflow")]
fn zigzag_gamma_i64_min_panics_clearly() {
    // i64::MIN cannot fit gamma's [1, 2^64-1] domain; it requires a graph
    // of > 2^63 nodes (unreachable in memory). Defined panic, not silent
    // stream corruption.
    let mut w = BitWriter::new();
    w.write_zigzag_gamma(i64::MIN);
}
```

- [ ] **Step 2: Run tests to verify behavior**

Run: `cargo test -p grafeo-core --all-features zigzag_gamma_round_trips_full_i64_domain`
Expected: FAIL — current code overflows for `i64::MAX` (the `2n+1` arm: `i64::MAX as u64 * 2 + 1` is fine, but `i64::MIN+1`/large negatives stress the branchy `-2n` arm) and produces a non-clear panic for `i64::MIN`.

- [ ] **Step 3: Reimplement both functions via `delta`**

Add an import near the top of `bitstream.rs` (with the other `use` lines):
```rust
use crate::codec::delta::{zigzag_decode, zigzag_encode};
```
Replace `write_zigzag_gamma` (85-95) with:
```rust
    /// Appends `zigzag_gamma(n)`: folds `n` to non-negative via the canonical
    /// zig-zag map, then gamma-codes `folded + 1` (gamma needs `>= 1`).
    ///
    /// Supported domain: any `i64` except `i64::MIN`, which would require a
    /// gap of magnitude `2^63` — unreachable for an in-memory graph
    /// (`< 2^63` nodes). `i64::MIN` panics with a clear message rather than
    /// silently corrupting the stream.
    pub(crate) fn write_zigzag_gamma(&mut self, n: i64) {
        let folded = zigzag_encode(n)
            .checked_add(1)
            .expect("zigzag-gamma overflow: gap == i64::MIN is unreachable for in-memory graphs");
        self.write_gamma(folded);
    }
```
Replace `read_zigzag_gamma` (179-192) with:
```rust
    /// Reads one zigzag-gamma-encoded signed integer.
    pub(crate) fn read_zigzag_gamma(&mut self) -> Option<i64> {
        // read_gamma returns >= 1, so folded - 1 >= 0 is a valid zig-zag code.
        let folded = self.read_gamma()?;
        Some(zigzag_decode(folded - 1))
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p grafeo-core --all-features zigzag_gamma`
Expected: PASS (round-trip across the full domain; `i64::MIN` panics with "zigzag-gamma overflow")

- [ ] **Step 5: Run the WebGraph codec round-trip suite (the only caller)**

Run: `cargo test -p grafeo-core --all-features --test webgraph_round_trip`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/codec/bitstream.rs
git commit -m "fix(codec): zigzag-gamma uses the canonical delta fold

write_zigzag_gamma hand-rolled a branchy signed fold that overflowed
for i64::MIN (silently wrapping to write_gamma(0) in release). Reuse
delta::zigzag_encode/decode and gamma-code folded+1; the single
unrepresentable value (i64::MIN, unreachable for in-memory graphs)
now panics with a clear message instead of corrupting the stream.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task D2: WASM codec ids cross the boundary as f64 (fixes #5)

**Files:**
- Modify: `crates/bindings/wasm/src/codecs.rs` (`RabitqCodec::search` 71-81; `WebGraphCodec::successors` 222-231, `out_degree` 234-242, `num_nodes` 245-253, and the `node: u32` args)
- Test: `crates/grafeo-core/tests/rabitq_recall.rs` (value-preservation guard, native) + `crates/bindings/wasm/tests/web.rs` (binding guard, wasm)

- [ ] **Step 1: Write the native value-preservation guard**

Append to `crates/grafeo-core/tests/rabitq_recall.rs`:

```rust
#[test]
fn search_preserves_node_ids_above_u32_max() {
    use grafeo_common::types::NodeId;
    use grafeo_core::index::vector::TwoStageVectorIndex;

    let big = (u32::MAX as u64) + 5; // 4_294_967_300
    let vectors = vec![
        (NodeId::new(big), vec![1.0f32, 0.0, 0.0, 0.0]),
        (NodeId::new(big + 1), vec![0.0f32, 1.0, 0.0, 0.0]),
    ];
    let index = TwoStageVectorIndex::build(&vectors, 4, 42);
    let hits = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 8);
    assert_eq!(hits[0].0, NodeId::new(big), "core must preserve the full u64 id");
    // f64 represents this id exactly (< 2^53), so the JS surface is lossless.
    assert_eq!(hits[0].0.as_u64() as f64 as u64, big);
}
```

- [ ] **Step 2: Run it (guards that the truncation is the binding's fault)**

Run: `cargo test -p grafeo-core --all-features search_preserves_node_ids_above_u32_max`
Expected: PASS (core already returns the full `u64`; the bug is the binding cast).

- [ ] **Step 3: Change `RabitqCodec::search` to return f64 ids**

In `crates/bindings/wasm/src/codecs.rs`, replace `search` (71-81) with:

```rust
    /// Searches for the `k` nearest neighbours of `query`. Returns node ids
    /// nearest-first as a `Float64Array`. Node ids are exact for any value
    /// below 2^53 (every realistic node count). `rerank_factor` controls the
    /// recall/latency trade-off (8–16 is typical).
    #[wasm_bindgen(js_name = "search")]
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize, rerank_factor: usize) -> Vec<f64> {
        // reason: f64 holds any node id below 2^53 exactly — lossless for the JS surface
        #[allow(clippy::cast_precision_loss)]
        self.inner
            .search(query, k, rerank_factor)
            .into_iter()
            .map(|(id, _)| id.as_u64() as f64)
            .collect()
    }
```

- [ ] **Step 4: Widen the WebGraph binding the same way**

Replace `successors` (222-231):
```rust
    /// Returns the successors of `node` as a `Float64Array` (ids exact below 2^53).
    #[wasm_bindgen(js_name = "successors")]
    #[must_use]
    pub fn successors(&self, node: f64) -> Vec<f64> {
        // reason: f64 holds any node id below 2^53 exactly
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        self.inner
            .successors(node as u64)
            .map(|d| d as f64)
            .collect()
    }
```
Replace `out_degree` (234-242):
```rust
    /// Out-degree of `node`.
    #[wasm_bindgen(js_name = "outDegree")]
    #[must_use]
    pub fn out_degree(&self, node: f64) -> f64 {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            self.inner.out_degree(node as u64) as f64
        }
    }
```
Replace `num_nodes` (245-253):
```rust
    /// Number of nodes.
    #[wasm_bindgen(js_name = "numNodes")]
    #[must_use]
    pub fn num_nodes(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.inner.num_nodes() as f64
        }
    }
```
Leave `num_edges` (256+) returning `u32` unless the same `as u32` truncation appears there — if it does, apply the identical `f64` treatment.

- [ ] **Step 5: Update the wasm-bindgen binding test**

In `crates/bindings/wasm/tests/web.rs`, add (or extend) a `#[wasm_bindgen_test]` that encodes a blob and asserts `search` returns the expected ids as `f64`. Mirror the existing tests in that file for blob construction.

- [ ] **Step 6: Verify the binding builds + (if toolchain present) run the wasm test**

Run: `cargo build -p grafeo-wasm --all-features` (crate name per `crates/bindings/wasm/Cargo.toml`; substitute if different)
Expected: compiles.
If `wasm-pack` is installed:
Run: `wasm-pack test --node crates/bindings/wasm -- --all-features`
Expected: PASS. (If `wasm-pack` is absent, the Step 2 native guard plus a clean build are sufficient evidence; note this in the commit.)

- [ ] **Step 7: Commit**

```bash
git add crates/bindings/wasm/src/codecs.rs crates/grafeo-core/tests/rabitq_recall.rs crates/bindings/wasm/tests/web.rs
git commit -m "fix(wasm): codec node ids cross to JS as f64, not truncated u32

RabitqCodec.search cast NodeId u64 -> u32, silently corrupting ids
>= 2^32 from natively-built blobs; the WebGraph binding truncated
identically. Return f64 (exact below 2^53) for ids and the WebGraph
successors/out_degree/num_nodes surface.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task E1: FSST first-byte index + borrowing train (fixes #9)

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs` (`SymbolTable` struct 73-90; `longest_match` 134-161; `train` 239+)
- Test: same file, `mod tests`

- [ ] **Step 1: Write the failing equivalence test**

In the `mod tests` block of `fsst.rs`, add:

```rust
#[test]
fn longest_match_index_matches_scan() {
    // Train a table on a representative sample, then assert the index-backed
    // longest_match equals the exhaustive scan for every suffix of the sample.
    let sample: Vec<&[u8]> = vec![b"banana", b"band", b"can", b"candy", b"a"];
    let table = SymbolTable::train(&sample);
    for s in &sample {
        for i in 0..s.len() {
            assert_eq!(
                table.longest_match(&s[i..]),
                table.longest_match_scan(&s[i..]),
                "index and scan disagree at suffix {:?}",
                &s[i..]
            );
        }
    }
}
```

- [ ] **Step 2: Run it to verify it fails (no `longest_match_scan` yet)**

Run: `cargo test -p grafeo-core --all-features longest_match_index_matches_scan`
Expected: FAIL — compile error: `no method named longest_match_scan`.

- [ ] **Step 3: Add the first-byte index to the struct**

Replace the `SymbolTable` struct + derives (fsst.rs:73-90) with:

```rust
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Symbol length per code (0 = absent). Index 0 is unused (escape).
    lengths: [u8; 256],
    /// Symbol bodies, 8 bytes per slot (right-padded with zeros). Index 0
    /// is unused.
    bodies: [[u8; MAX_SYMBOL_LEN]; 256],
    /// Codes bucketed by their first byte, each bucket ordered longest-first
    /// so `longest_match` returns the first full match. Derived state, rebuilt
    /// by `rebuild_first_byte_index`; excluded from `PartialEq`.
    first_byte_index: [Vec<u8>; 256],
    /// True once `first_byte_index` reflects `lengths`/`bodies`. When false
    /// (raw `set` without a finalizing constructor), `longest_match` falls
    /// back to the exhaustive scan.
    index_built: bool,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            lengths: [0u8; 256],
            bodies: [[0u8; MAX_SYMBOL_LEN]; 256],
            first_byte_index: std::array::from_fn(|_| Vec::new()),
            index_built: false,
        }
    }
}

// Semantic equality is the symbol set only; the derived index/flag are
// excluded so tables built via different paths compare equal.
impl PartialEq for SymbolTable {
    fn eq(&self, other: &Self) -> bool {
        self.lengths == other.lengths && self.bodies == other.bodies
    }
}
```

- [ ] **Step 4: Add the rebuild method, rename the scan, add the indexed `longest_match`**

Inside `impl SymbolTable`, add the rebuild method and replace `longest_match` (134-161) with the scan (renamed) plus the index-backed entry point:

```rust
    /// Rebuilds `first_byte_index` from `lengths`/`bodies`. Call after a
    /// table is fully assembled (train/build/from_bytes).
    fn rebuild_first_byte_index(&mut self) {
        for bucket in &mut self.first_byte_index {
            bucket.clear();
        }
        for code in 1u8..=255 {
            let len = self.lengths[code as usize] as usize;
            if len == 0 {
                continue;
            }
            self.first_byte_index[self.bodies[code as usize][0] as usize].push(code);
        }
        // Order each bucket longest-first (ties: smaller code first, matching
        // the scan's tie-break) so the first full match is the longest.
        for bucket in &mut self.first_byte_index {
            bucket.sort_unstable_by(|&a, &b| {
                self.lengths[b as usize]
                    .cmp(&self.lengths[a as usize])
                    .then(a.cmp(&b))
            });
        }
        self.index_built = true;
    }

    /// Exhaustive O(255 × MAX_SYMBOL_LEN) longest-prefix scan. Correct
    /// fallback when the first-byte index has not been built.
    #[must_use]
    pub fn longest_match_scan(&self, input: &[u8]) -> Option<(u8, usize)> {
        if input.is_empty() {
            return None;
        }
        let max_check = input.len().min(MAX_SYMBOL_LEN);
        let mut best: Option<(u8, usize)> = None;
        for code in 1u8..=255 {
            let len = self.lengths[code as usize] as usize;
            if len == 0 || len > max_check {
                continue;
            }
            if self.bodies[code as usize][..len] == input[..len] {
                match best {
                    None => best = Some((code, len)),
                    Some((_, blen)) if len > blen => best = Some((code, len)),
                    _ => {}
                }
            }
        }
        best
    }

    /// Returns `(code, length)` of the longest symbol that prefixes `input`.
    /// Uses the per-first-byte index; falls back to the scan if it was not
    /// built. Ties on length break by the smaller code.
    #[must_use]
    pub fn longest_match(&self, input: &[u8]) -> Option<(u8, usize)> {
        if input.is_empty() {
            return None;
        }
        if !self.index_built {
            return self.longest_match_scan(input);
        }
        let max_check = input.len().min(MAX_SYMBOL_LEN);
        for &code in &self.first_byte_index[input[0] as usize] {
            let len = self.lengths[code as usize] as usize;
            if len > max_check {
                continue; // longer symbol can't fit; shorter ones follow in-bucket
            }
            if self.bodies[code as usize][..len] == input[..len] {
                return Some((code, len)); // bucket is longest-first
            }
        }
        None
    }
```

- [ ] **Step 5: Finalize the index in `train` and borrow during counting**

In `train` (fsst.rs:239+), change the counts map to borrow from the sample, and rebuild the index before returning. Replace the counting block:
```rust
        let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for s in sample {
            for start in 0..s.len() {
                let max_end = (start + MAX_SYMBOL_LEN).min(s.len());
                for end in (start + 1)..=max_end {
                    let sub = &s[start..end];
                    *counts.entry(sub.to_vec()).or_insert(0) += 1;
                }
            }
        }
```
with:
```rust
        let mut counts: HashMap<&[u8], u64> = HashMap::new();
        for s in sample {
            for start in 0..s.len() {
                let max_end = (start + MAX_SYMBOL_LEN).min(s.len());
                for end in (start + 1)..=max_end {
                    *counts.entry(&s[start..end]).or_insert(0) += 1;
                }
            }
        }
```
Update the `scored` type that consumes `counts` from `Vec<(Vec<u8>, u64)>` to `Vec<(&[u8], u64)>` (the downstream `set(code, sub)` call already takes `&[u8]`, so `sub` flows through unchanged). At the end of `train`, immediately before `Self`/the table is returned, call `table.rebuild_first_byte_index();` on the mutable table value (introduce a `let mut table = ...; table.rebuild_first_byte_index(); table` binding if the function currently returns an expression).

- [ ] **Step 6: Finalize the index in `from_bytes`**

Find the `FsstCodec::from_bytes`/`SymbolTable` reconstruction (around fsst.rs:487-621 builds the table via `set`). After the table's `lengths`/`bodies` are fully populated and before it is returned/used, call `table.rebuild_first_byte_index();`. (Search for where the decoded `SymbolTable` is finalized; add the one call there.)

- [ ] **Step 7: Run the equivalence test + the FSST suites**

Run: `cargo test -p grafeo-core --all-features longest_match_index_matches_scan`
Expected: PASS
Run: `cargo test -p grafeo-core --all-features --test fsst_round_trip && cargo test -p grafeo-core --all-features --lib codec::fsst`
Expected: PASS (round-trip + all in-module FSST tests, including `train_*`)

- [ ] **Step 8: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs
git commit -m "perf(codec): FSST first-byte index + borrowing train

longest_match scanned all 255 codes at every input position. Add a
per-first-byte bucket index (longest-first) built once at table
construction, with the exhaustive scan kept as a fallback. train()
now counts substrings by borrowing the sample instead of a Vec<u8>
allocation per occurrence.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task E2: RaBitQ query path — reuse scratch, select instead of full-sort (fixes #8)

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs` (`coarse_search` 410-421; `RabitqView` `read_code_bits` 1028-1037 + `search` 1059-1090)
- Test: same file, `mod tests`

- [ ] **Step 1: Write the ordering-preservation test**

In the `mod tests` block of `rabitq.rs`, add:

```rust
#[test]
fn coarse_search_returns_n_smallest_in_order() {
    // Build a small index; coarse_search must return exactly the n nearest
    // by estimate, ascending — identical to a full sort + truncate.
    let mut rng = SplitMix64::new(123);
    let dim = 8;
    let vectors: Vec<(NodeId, Vec<f32>)> = (0..50u64)
        .map(|i| (NodeId::new(i), (0..dim).map(|_| rng.next_gaussian()).collect()))
        .collect();
    let index = TwoStageVectorIndex::build(&vectors, dim, 7);
    let query: Vec<f32> = (0..dim).map(|_| rng.next_gaussian()).collect();

    let got = index.coarse_search(&query, 10);
    // Reference: score everything, full-sort, truncate.
    let q = index.quantizer().encode_query(&query); // see note below
    // The simplest reference is to call coarse_search with n == len and take
    // the first 10 — it must equal the n==10 call.
    let full = index.coarse_search(&query, vectors.len());
    assert_eq!(got.len(), 10);
    assert_eq!(got, full[..10].to_vec(), "selection must match full-sort prefix");
    // ascending by estimate
    for w in got.windows(2) {
        assert!(w[0].1 <= w[1].1, "coarse_search not ascending");
    }
    let _ = q; // (drop the encode_query line above if `quantizer()` isn't public)
}
```
If `quantizer()` is not accessible from the test, delete the `let q = ...` and `let _ = q;` lines — the `full[..10]` reference is the authoritative check.

- [ ] **Step 2: Run it (passes today; it is the regression guard)**

Run: `cargo test -p grafeo-core --all-features coarse_search_returns_n_smallest_in_order`
Expected: PASS (guards the behavior the perf change must preserve).

- [ ] **Step 3: Replace the full sort in `coarse_search` with selection**

Replace `coarse_search`'s sort/truncate (rabitq.rs:418-419) — change:
```rust
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
```
to:
```rust
        let cmp = |a: &(NodeId, f32), b: &(NodeId, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        if n < scored.len() {
            scored.select_nth_unstable_by(n - 1, cmp);
            scored.truncate(n);
        }
        scored.sort_unstable_by(cmp);
        scored
```
(`n` here is `>= 1`; callers pass `k * rerank_factor` clamped to `>= 1`.)

- [ ] **Step 4: Add a scratch-filling reader and reuse one code buffer in `RabitqView::search`**

Add next to `read_code_bits` (rabitq.rs:1028):
```rust
    /// Fills `buf` with the code words at `row`, reusing its allocation.
    fn read_code_bits_into(&self, row: usize, buf: &mut Vec<u64>) {
        buf.clear();
        let start = self.codes_offset + row * self.code_stride;
        let bytes = self.blob.as_ref();
        for w in 0..self.words {
            let pos = start + w * 8;
            buf.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().expect("8 bytes")));
        }
    }
```
In `RabitqView::search` (1059-1090), replace the coarse pass (1067-1077) with a single reused `RabitqCode` and selection:
```rust
        // Coarse pass: one reused code buffer, no per-row allocation.
        let q = self.rotation_quantizer.encode_query(query);
        let mut code = RabitqCode {
            bits: Vec::with_capacity(self.words),
            dot_oo: 0.0,
            norm: 0.0,
        };
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.count);
        for row in 0..self.count {
            self.read_code_bits_into(row, &mut code.bits);
            let (dot_oo, norm) = self.read_code_factors(row);
            code.dot_oo = dot_oo;
            code.norm = norm;
            let est = self.rotation_quantizer.estimate_distance(&q, &code);
            scored.push((row, est));
        }
        let cmp = |a: &(usize, f32), b: &(usize, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        if candidate_n < scored.len() {
            scored.select_nth_unstable_by(candidate_n - 1, cmp);
            scored.truncate(candidate_n);
        }
        scored.sort_unstable_by(cmp);
```
Then apply the same selection shape to the rerank step (1087-1088): replace `reranked.sort_by(...); reranked.truncate(k);` with the `select_nth_unstable_by(k - 1)` + `truncate(k)` + `sort_unstable_by` pattern guarded by `if k < reranked.len()`.

- [ ] **Step 5: Run the view parity + recall suites**

Run: `cargo test -p grafeo-core --all-features coarse_search_returns_n_smallest_in_order`
Expected: PASS
Run: `cargo test -p grafeo-core --all-features --test rabitq_recall --test codec_view_parity`
Expected: PASS (the view↔owned parity proptest confirms identical results after the change)

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs
git commit -m "perf(vector): RaBitQ reuses scratch + selects instead of full-sort

RabitqView::search allocated a Vec<u64> per stored vector per query and
both coarse paths fully sorted all N candidates to keep the top n. Reuse
one code buffer across the scan and select_nth_unstable + sort the kept
prefix (O(N + n log n)).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] **Full affected-crate test pass**

Run: `cargo test -p grafeo-core -p grafeo-engine --all-features`
Expected: PASS

- [ ] **Clippy clean on the touched crates**

Run: `cargo clippy -p grafeo-core -p grafeo-engine --all-features -- -D warnings`
Expected: no warnings.

- [ ] **CI-representative feature profiles compile** (the matrix from `.github/workflows/ci.yml`)

Run: `cargo test -p grafeo-engine --features lpg,gql,cypher,gremlin,sql-pgq,wal,spill,mmap,regex --no-run`
Run: `cargo test -p grafeo-engine --features algos,vector-index,text-index,hybrid-search --no-run`
Expected: both compile (catches a feature-gated break the `--all-features` run would hide).

- [ ] **Update CHANGELOG.md** with the audit-fix entries, then commit.

---

## Self-Review

**Spec coverage:** A1↔#2, A2↔#1, A3↔#3, A4↔accessor-audit decision, B↔#4, C1↔#6, C2↔#10, D1↔#7, D2↔#5, E1↔#9, E2↔#8 — all 10 findings + the A4 decision have a task.

**Type consistency:** `read_graph_view(&self) -> &dyn GraphStore` used identically in A3/A4; `sort_key_matches_projection`/`resolve_sort_keys_structurally` defined and called in B; `union_index_metadata -> Result<SnapshotIndexes>` matched by the `?` at its call site in C1; `rebuild_first_byte_index`/`longest_match_scan`/`index_built` consistent across E1 steps; `read_code_bits_into` + reused `RabitqCode` consistent in E2.

**Placeholder scan:** No TBD/TODO. A4 is conditional by nature (investigate → fix-or-document) with both branches spelled out — not a placeholder. The E1 `from_bytes` finalize (Step 6) names the exact call to add and where to locate it.

**Open knowns the executor must confirm at the file (not guess):** the wasm crate's package name (D2 Step 6); the `SnapshotVectorIndex.metric` accessor (`.name()` used, avoiding a `PartialEq` assumption); the exact `from_bytes` table-finalization point (E1 Step 6); `quantizer()`/`SplitMix64` visibility from the rabitq test module (E2 Step 1, with a documented fallback).
