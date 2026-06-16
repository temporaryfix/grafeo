# Increment 2c — MERGE read-your-writes completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MERGE` see the writing transaction's own uncommitted work — a same-tx-created node is a match candidate (no duplicate), and a relationship-MERGE matching on a property `SET` earlier in the tx matches — completing the read-your-writes isolation model for MERGE and closing Part C (the last piece of "Plan 1": A labels + B edge-deletes + C MERGE).

**Architecture:** Two targeted, independent fixes in `execution/operators/merge.rs`, both mirroring patterns already used in `find_matching_node`'s per-node filter. (1) **Node candidate set:** `find_matching_node`'s candidate set comes from the committed property index (`find_nodes_by_properties`) / committed label scan, so a node CREATEd in this same tx (PENDING, properties only in the overlay) is never a candidate → MERGE duplicates it. Fix: when in a transaction, **union** the candidate set with the tx's same-tx-created node ids (a new non-draining `pending_node_creates` accessor over `pending_tx_creates`); the existing per-node `read_node_property_visible` + `read_node_labels_visible` filter loop already handles the rest correctly. (2) **Edge property read:** `find_matching_edge`'s property comparison reads committed properties via `edge.get_property`, not the per-tx delta — route it through the existing `read_edge_property_visible` (delta-aware), mirroring `find_matching_node`'s `read_node_property_visible` routing.

**Tech Stack:** Rust, `cargo test`. Operator in `grafeo-core` (`execution/operators/merge.rs`); store accessor in `graph/lpg/store/{mod.rs,graph_store_impl.rs}` + trait in `graph/traits.rs`; wrapper delegation in `graph/compact/layered.rs`, `database/{cdc_store,wal_store}.rs`. `CARGO_INCREMENTAL=0`.

---

## Orientation

Part C of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§5). Parts A (labels) and B (edge-deletes) are merged to `integration`; 2b's holistic review already landed the *existence* half of `find_matching_edge` (C1: resolve candidates via `get_edge_versioned`) but explicitly left the *property-read* TODO — this plan finishes it.

- **`find_matching_node`** (`merge.rs:252`). The per-node loop (`merge.rs:286-333`) is ALREADY delta-aware: it resolves via `get_node_versioned` when a tx is attached, checks labels via `read_node_labels_visible`, and checks properties via `read_node_property_visible(node_id, &prop_key, epoch, Some(tid))`. The ONLY gap is the **candidate set** (`merge.rs:266-277`): `find_nodes_by_properties` (committed index) / `nodes_by_label` / `node_ids` do not include same-tx-created PENDING nodes. The code already documents this gap in a comment at `merge.rs:259-265`.
- **`find_matching_edge`** (`merge.rs:815`). Candidates come from `edges_from(src, Outgoing)` (same-tx-CREATED edges ARE present — CREATE does not defer adjacency, unlike delete) and are resolved via `get_edge_versioned` (the 2b C1 fix, `merge.rs:834-840`). The gap is ONLY the property comparison (`merge.rs:847-854`): `edge.get_property(key)` reads the committed snapshot, not the tx delta. The TODO is at `merge.rs:847-851`.
- **`read_edge_property_visible`** (`property_ops.rs:1106`, trait `traits.rs:106`): `(&self, id: EdgeId, key: &PropertyKey, epoch: EpochId, transaction_id: Option<TransactionId>) -> Option<Value>`. Reads the per-tx `tx_property_overlay` `edge_props` delta first (Set→Some, Remove→None), else committed. The node twin `read_node_property_visible` is at `property_ops.rs:1073` / `traits.rs:94` and is already called from `find_matching_node:318`.
- **`pending_tx_creates`** (`mod.rs:520`): `RwLock<FxHashMap<TransactionId, (Vec<NodeId>, Vec<EdgeId>)>>` (created nodes, created edges) per tx. Populated by `record_pending_node`/`record_pending_edge` (`mod.rs:968/981`). There is a **draining** `take_pending_creates(tx)` (`mod.rs:996`) used by commit/rollback — do NOT use it here (it removes the entries). This plan adds a **non-draining** read accessor.
- **`self.store`** in the merge operators is the trait object that already exposes `read_node_property_visible` / `find_nodes_by_properties` / `get_node_versioned`. Add the new accessor to the SAME trait (`graph/traits.rs`, alongside `read_node_property_visible` at `traits.rs:94`), with a default returning an empty `Vec`, overridden on `LpgStore`. Confirm the exact trait name when implementing (it is the trait whose method `read_node_property_visible` the operator calls).

**Verification gate:** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` AND `--features full -p grafeo-engine` for the isolation integration tests. Keep `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean. **Hygiene:** format only changed files (`rustfmt <file>`, not whole-crate `cargo fmt`); confirm `git status` shows only intended files before committing. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. **OPSEC:** scan the diff for the private project's name/schema (use generic labels like `:Thing`/`:N` in tests).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/tests/mvcc_isolation.rs` | Acceptance probes | Modify (3 MERGE read-your-writes probes) |
| `crates/grafeo-core/src/graph/traits.rs` | store trait surface | Modify (add `pending_node_creates` default-empty) |
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` | inherent accessor | Modify (add non-draining `pending_node_creates`) |
| `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` | trait impl | Modify (override `pending_node_creates` → inherent) |
| `crates/grafeo-core/src/graph/compact/layered.rs`, `database/cdc_store.rs`, `database/wal_store.rs` | wrappers | Modify (delegate `pending_node_creates` to inner/overlay) |
| `crates/grafeo-core/src/execution/operators/merge.rs` | MERGE matching | Modify (`find_matching_node` candidate union; `find_matching_edge` delta-aware property read) |

---

## Task 0: TDD baseline — MERGE read-your-writes probes, confirm RED

**Files:** Modify `crates/grafeo-engine/tests/mvcc_isolation.rs`

- [ ] **Step 1: Add three probes.** (Confirm `MERGE`/`CREATE`/`SET` syntax + the `:Thing`/`:N` label and relationship forms against existing engine MERGE tests — grep the test suite for `MERGE` — and adjust the query strings to the supported dialect, keeping the assertions/semantics intact.)
```rust
#[test]
fn merge_twice_in_tx_creates_one_node() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.begin_transaction().unwrap();
    w.execute("MERGE (n:Thing {k: 1})").unwrap();
    w.execute("MERGE (n:Thing {k: 1})").unwrap();
    let own = w.execute("MATCH (n:Thing {k: 1}) RETURN n").unwrap();
    assert_eq!(own.row_count(), 1, "two MERGEs of the same key in one tx must create exactly one node");
    w.commit().unwrap();
    let reader = db.session();
    let after = reader.execute("MATCH (n:Thing {k: 1}) RETURN n").unwrap();
    assert_eq!(after.row_count(), 1, "exactly one node after commit");
}

#[test]
fn merge_matches_same_tx_created_node() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.begin_transaction().unwrap();
    w.execute("CREATE (n:Thing {k: 1})").unwrap();
    w.execute("MERGE (m:Thing {k: 1})").unwrap();
    let own = w.execute("MATCH (n:Thing {k: 1}) RETURN n").unwrap();
    assert_eq!(own.row_count(), 1, "MERGE must match the same-tx CREATEd node, not duplicate it");
    w.commit().unwrap();
}

#[test]
fn merge_edge_matches_same_tx_set_property() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (a:N {id: 1})-[:R {w: 1}]->(b:N {id: 2})").unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) SET r.w = 5").unwrap();
    // MERGE on the tx-visible value (5) must MATCH the existing edge, not create a second.
    w.execute("MERGE (a:N {id: 1})-[r:R {w: 5}]->(b:N {id: 2})").unwrap();
    let own = w.execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) RETURN r").unwrap();
    assert_eq!(own.row_count(), 1, "MERGE must match the edge whose prop was SET earlier in the tx, not create a duplicate");
    w.commit().unwrap();
}
```
- [ ] **Step 2: Run; confirm RED.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation merge_`. Expected: all three FAIL with `row_count == 2` (a duplicate node/edge was created because MERGE could not see the tx's own work). If any passes unexpectedly, investigate the query shape (it may not be hitting `find_matching_node`/`find_matching_edge`) and report before proceeding. Don't touch `src/` to fix yet.
- [ ] **Step 3: Commit** the failing baseline (`test(mvcc): failing MERGE read-your-writes probes`).

---

## Task 1: `find_matching_node` candidate-set union with same-tx creates

**Files:** `graph/traits.rs` (trait method), `graph/lpg/store/mod.rs` (inherent accessor), `graph/lpg/store/graph_store_impl.rs` (override), `graph/compact/layered.rs` + `database/cdc_store.rs` + `database/wal_store.rs` (delegation), `execution/operators/merge.rs` (`find_matching_node`)

- [ ] **Step 1: Inherent non-draining accessor** in `mod.rs`, next to `take_pending_creates` (`mod.rs:996`):
```rust
/// Non-draining snapshot of the node ids this transaction has created with a
/// PENDING version (from `pending_tx_creates`). Used by MERGE to treat
/// same-tx-created nodes as match candidates (read-your-writes). Returns empty
/// for the system transaction (its creates are immediately visible / committed).
pub fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
    self.pending_tx_creates
        .read()
        .get(&transaction_id)
        .map(|(nodes, _edges)| nodes.clone())
        .unwrap_or_default()
}
```
- [ ] **Step 2: Trait method** in `graph/traits.rs`, alongside `read_node_property_visible` (`traits.rs:94`), default empty so non-LpgStore impls and wrappers compile:
```rust
/// Node ids created (PENDING) by `transaction_id` in this tx, for MERGE
/// read-your-writes candidate discovery. Default: none.
fn pending_node_creates(&self, _transaction_id: TransactionId) -> Vec<NodeId> {
    Vec::new()
}
```
- [ ] **Step 3: Override on `LpgStore`** in `graph_store_impl.rs`, mirroring the `read_node_property_visible` override (delegate to inherent):
```rust
fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
    LpgStore::pending_node_creates(self, transaction_id)
}
```
- [ ] **Step 4: Wrapper delegation** — add `pending_node_creates` overrides delegating to inner/overlay in `LayeredStore` (`layered.rs`, → `self.overlay.load()`), `CdcGraphStore` (`cdc_store.rs`, → `self.inner`), `WalGraphStore` (`wal_store.rs`, → `self.inner`), exactly mirroring how each delegates `read_node_property_visible` / `take_pending_edge_deletes`.
- [ ] **Step 5: Union in `find_matching_node`** (`merge.rs:266-277`). After the existing `let candidates: Vec<NodeId> = ...;` block, before the `for node_id in candidates` loop, append the tx's same-tx-created nodes (deduped) when a transaction is attached, and update the stale comment at `merge.rs:259-265` to say the gap is now closed:
```rust
// Read-your-writes: a node CREATEd in this same transaction is PENDING and absent
// from the committed candidate sources above, so union it in. The per-node filter
// below (read_node_labels_visible + read_node_property_visible) then matches it
// correctly. (Perf: O(tx-created) per MERGE — acceptable; tightened in Part G.)
let mut candidates = candidates;
if let Some(tid) = self.transaction_id {
    let existing: std::collections::HashSet<NodeId> = candidates.iter().copied().collect();
    for nid in self.store.pending_node_creates(tid) {
        if !existing.contains(&nid) {
            candidates.push(nid);
        }
    }
}
```
- [ ] **Step 6: Run GREEN.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation merge_twice_in_tx_creates_one_node merge_matches_same_tx_created_node` → both PASS (the edge probe still RED until Task 2). `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green; clippy clean. Commit (`feat(mvcc): MERGE node candidate set includes same-tx creates`).

---

## Task 2: `find_matching_edge` delta-aware property read

**Files:** `execution/operators/merge.rs` (`find_matching_edge`)

- [ ] **Step 1: Route the property comparison through `read_edge_property_visible`** (`merge.rs:847-854`), mirroring `find_matching_node:315-326`, and delete the `TODO(unified-mvcc)` comment at `merge.rs:847-851`:
```rust
let has_all_props = resolved_match_props.iter().all(|(key, expected)| {
    // Delta-aware: a property SET earlier in this tx is visible here (read-your-writes),
    // mirroring find_matching_node's read_node_property_visible routing. Absent a tx
    // context, fall back to the committed snapshot already resolved above.
    let prop_key = PropertyKey::new(key.as_str());
    let actual = match (self.viewing_epoch, self.transaction_id) {
        (Some(epoch), Some(tid)) => {
            self.store.read_edge_property_visible(edge_id, &prop_key, epoch, Some(tid))
        }
        _ => edge.get_property(key).cloned(),
    };
    actual.as_ref().is_some_and(|v| v == expected)
});
```
(Confirm `PropertyKey` is in scope in `merge.rs` — `find_matching_node` already uses it; add the `use` if needed. Confirm `read_edge_property_visible` is on the trait `self.store` implements — it is, per `traits.rs:106`.)
- [ ] **Step 2: Run GREEN.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation merge_` → all THREE pass. Commit (`fix(mvcc): MERGE edge match reads tx property delta (read-your-writes)`).

---

## Task 3: Full verification + OPSEC

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green; `--features full -p grafeo-engine --test mvcc_isolation` green (incl. the 3 new probes; watch for any existing MERGE test whose behavior changed — if one breaks, determine whether it asserted the old buggy duplicate-on-merge behavior and report, don't blindly edit).
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` (`wasm32-unknown-unknown`) compile.
- [ ] `git status` clean (only intended files; revert collateral fmt). OPSEC: scan the diff for the private project's name/schema.

---

## Acceptance
- Within one transaction: `MERGE (n {k:1})` twice creates exactly one node; `CREATE (n {k:1}) … MERGE (m {k:1})` matches `n` (no duplicate); a relationship `MERGE` whose match-property was `SET` earlier in the tx matches the existing edge.
- Full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; tree clean; OPSEC-clean.
- MERGE read-your-writes is complete → **Part C done, completing Plan 1 (A+B+C — the isolation model)**. Next: Plan 2 (Parts D+E — read routing + record_read + read-registry + write-set).

## Risks
- **MERGE query shape may not hit the fast path.** If a probe doesn't go RED, the query may route through a different operator than `find_matching_node`/`find_matching_edge` (e.g. a full MATCH-then-create plan). Confirm the operator path before assuming the fix site; if MERGE compiles to something else, the candidate/property fix must target the actual matching site. Report rather than forcing a probe green.
- **Perf of the node union.** Each `find_matching_node` now scans the tx's created-node list (O(tx-created)); a big `UNWIND … MERGE` is O(n²). Accepted for correctness; Part G (property-keyed/registry path) is where this is optimized. Do not add speculative indexing here (YAGNI).
- **`transaction_id` always paired with `viewing_epoch`.** The per-node/-edge filters assume both are `Some` together in production planner paths (existing comment at `merge.rs:321-322`); the union (Task 1) keys only off `transaction_id`, which is consistent. Don't introduce a path that has a tx id but no epoch.
- **Wrapper parity.** As in 2b Task 4, the session transactional path resolves to the concrete `Arc<LpgStore>`, so wrapper delegation is for trait completeness; mirror the existing `read_node_property_visible` delegation exactly and don't rework the wrappers.
