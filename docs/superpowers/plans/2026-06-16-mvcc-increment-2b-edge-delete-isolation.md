# Increment 2b — Edge-delete isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make transactional edge deletion (including `DETACH DELETE` of a node with edges) snapshot-isolated — an uncommitted edge delete is invisible to other sessions (via `neighbors`/`edges_from`/edge-property reads) but gone for the writing transaction; commit applies, rollback restores — closing the last dirty-*write* hole and completing the isolation model uniformly across existence, properties, labels, node-deletes, and edge-deletes.

**Architecture:** Mirror increment 1's node-delete isolation exactly. Transactional edge deletion stamps the edge version chain `deleted_epoch = EpochId::PENDING` (invisible to the deleter via `deleted_by`, still visible to others since PENDING > any real epoch), and **defers the adjacency tombstone** to commit by recording `(src, edge_id, dst)` in a store-level `pending_tx_edge_deletes`. Commit finalizes (`deleted_epoch` PENDING→commit_epoch + applies the deferred `forward_adj`/`backward_adj` tombstones); rollback/conflict `unmark_deleted_by` + drops the deferred set (adjacency was never touched, so nothing to restore). Transactional `DETACH` routes through this path instead of the eager `TransactionId::SYSTEM` one.

**Tech Stack:** Rust, `cargo test`. Storage in `grafeo-core` (`graph/lpg/store/edge_ops.rs`, `node_ops.rs`, `mod.rs`), adjacency in `graph-core/src/index/adjacency.rs`, transaction wiring in `grafeo-engine/src/session/mod.rs`. `CARGO_INCREMENTAL=0`.

---

## Orientation

This is increment 2, Part B of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§4). Increment 1 did exactly this for **node** deletes; edges are the same shape, and most of the machinery exists as a template:

- **The bug.** `LpgStore::delete_edge_transactional` (`edge_ops.rs:471` non-temporal, `:545` temporal) does `chain.mark_deleted(epoch, transaction_id)` with the **real** epoch (so the delete is visible at that epoch to everyone) AND eagerly tombstones adjacency: `self.forward_adj.mark_deleted(src, id)` + `backward.mark_deleted(dst, id)` (`edge_ops.rs:509-511` / `587-589`) — globally visible before commit. `delete_node_edges` (the `DETACH` path, `node_ops.rs`) is worse: `TransactionId::SYSTEM` + eager `batch_mark_deleted`.
- **The node-delete template to mirror** (all in `node_ops.rs` / `mod.rs`):
  - `pending_tx_deletes: RwLock<FxHashMap<TransactionId, Vec<NodeId>>>` (`mod.rs:534`).
  - `delete_node_transactional` (`node_ops.rs:621`): `chain.mark_deleted(EpochId::PENDING, transaction_id)` + push to `pending_tx_deletes` + deferred index removal.
  - `finalize_deletes_by_id(tx, commit_epoch, &[NodeId])` (`node_ops.rs:1131`): `chain.finalize_deleted_epochs(tx, commit_epoch)` per id + apply the deferred removal.
  - `rollback_pending_deletes(tx, &[NodeId])` (`node_ops.rs:1246`): `chain.unmark_deleted_by(tx)` per id.
  - `take_pending_deletes(tx) -> Vec<NodeId>` (`node_ops.rs:1276`).
  - `VersionChain::finalize_deleted_epochs` + `OptionalEpochId::PENDING` sentinel (grafeo-common, from increment 1) cover both temporal and tiered.
- **Adjacency API** (`index/adjacency.rs`): `forward_adj.mark_deleted(src, edge_id)` (`:591`), `batch_mark_deleted(&[(NodeId, EdgeId)])` (`:603`), `unmark_deleted(src, edge_id)` (`:620`).
- **Reads to confirm honor PENDING.** `neighbors()`/`edges_from()`/edge resolution consult the edge version chain's `visible_to(epoch, tx)` and the adjacency tombstones. With the adjacency tombstone deferred, the writer sees the edge gone via the chain's `deleted_by == tx`; others see it alive (PENDING `deleted_epoch` + no adjacency tombstone yet).
- **The acceptance probe** is already written and `#[ignore]`d: `crates/grafeo-engine/tests/mvcc_isolation.rs:237` `uncommitted_detach_delete_edges_invisible_to_other_sessions` — un-ignore it.
- **Commit/rollback wiring** lives in `session/mod.rs` `commit_inner` (the success loop already calls `finalize_deletes_by_id` + `take_pending_deletes`; conflict branch + `rollback_inner` call `rollback_pending_deletes`). Add the edge-delete equivalents alongside.

**Verification gate:** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` AND `--features full -p grafeo-engine` for the isolation integration tests. **Run `--all-features -p grafeo-engine`, not just grafeo-core** (the 2a temporal MERGE-dedup regression slipped a grafeo-core-only gate). Keep `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean. **Hygiene:** format only changed files (`rustfmt <file>`, not whole-crate `cargo fmt`); confirm `git status` shows only intended files before committing. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/tests/mvcc_isolation.rs` | Acceptance tests | Modify (un-ignore the DETACH-edges probe; add direct edge-delete isolation tests) |
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` | store fields | Modify (add `pending_tx_edge_deletes`) |
| `crates/grafeo-core/src/graph/lpg/store/edge_ops.rs` | edge delete + finalize/rollback | Modify (`delete_edge_transactional` → PENDING + deferred; `finalize_edge_deletes_by_id`, `take_pending_edge_deletes`, `rollback_pending_edge_deletes`) |
| `crates/grafeo-core/src/graph/lpg/store/node_ops.rs` | DETACH path | Modify (`delete_node_edges` transactional variant routes through deferred edge-delete) |
| `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` + `graph/traits.rs` | trait surface | Modify (`finalize_edge_deletes_by_id` / `take_pending_edge_deletes` GraphStoreMut methods) |
| `crates/grafeo-engine/src/session/mod.rs` | commit/rollback wiring | Modify (finalize edge-deletes on commit; rollback on rollback/conflict) |
| `crates/grafeo-core/src/graph/compact/layered.rs`, `database/cdc_store.rs`, `wal_store.rs` | wrappers | Modify (delegate the new methods, mirroring increment-1 Task 6) |

---

## Task 0: TDD baseline — un-ignore probe + direct edge-delete tests, confirm RED

**Files:** Modify `crates/grafeo-engine/tests/mvcc_isolation.rs`

- [ ] **Step 1: Un-ignore the DETACH-edges probe** (remove the `#[ignore = "..."]` line at ~237). Add two direct edge-delete tests:
```rust
#[test]
fn uncommitted_edge_delete_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (a:N {id: 1})-[:R {w: 5}]->(b:N {id: 2})").unwrap();

    writer.begin_transaction().unwrap();
    writer.execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) DELETE r").unwrap();

    // Writer no longer traverses the edge (read-your-writes).
    let own = writer.execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id").unwrap();
    assert_eq!(own.row_count(), 0, "writer must not see its own deleted edge");

    // Other session must still traverse a->b.
    let reader = db.session();
    let during = reader.execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id").unwrap();
    let visible_during = during.row_count();

    writer.rollback().unwrap();
    let after = reader.execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id").unwrap();

    assert_eq!(visible_during, 1, "uncommitted edge delete must not be visible to other sessions");
    assert_eq!(after.row_count(), 1, "edge must exist after rollback of delete");
}

#[test]
fn committed_edge_delete_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (a:N {id: 1})-[:R]->(b:N {id: 2})").unwrap();
    writer.begin_transaction().unwrap();
    writer.execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) DELETE r").unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader.execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id").unwrap();
    assert_eq!(r.row_count(), 0, "committed edge delete must be visible");
}
```
(Confirm the `CREATE (a)-[:R]->(b)` + `MATCH ()-[r]->() DELETE r` syntax against existing engine tests; adjust to the supported form if needed, keeping the assertions.)

- [ ] **Step 2: Run; confirm RED.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation uncommitted_edge_delete committed_edge_delete uncommitted_detach_delete_edges`. Expected: `uncommitted_edge_delete_*` and `uncommitted_detach_delete_edges_*` FAIL (the reader sees the edge gone during the uncommitted delete — dirty write); `committed_edge_delete_*` PASS. Don't touch `src/` to fix yet.
- [ ] **Step 3: Commit** the failing baseline (`test(mvcc): failing edge-delete isolation probes`).

---

## Task 1: PENDING `deleted_epoch` + deferred adjacency in `delete_edge_transactional`

**Files:** `mod.rs` (field), `edge_ops.rs` (delete + finalize/rollback/take); Test `store/tests.rs`

- [ ] **Step 1: Failing store-level test** in `store/tests.rs` (`edge_delete_pending_isolates`): create a->b edge; `delete_edge_transactional(eid, real_epoch_ignored, tx)`; assert the edge is invisible to `tx` (writer) via `edges_from(a, Outgoing)` / `get_edge_versioned(eid, epoch, tx)` but VISIBLE to another tx (`get_edge_versioned(eid, epoch, other_tx)` is `Some`, `edges_from` for another reader still has it); assert `forward_adj`/`backward_adj` were NOT eagerly tombstoned (the edge still in the raw adjacency until commit); then `finalize_edge_deletes_by_id(tx, commit_epoch, &[...])` makes it gone for everyone at `commit_epoch`. Run → fails to compile.
- [ ] **Step 2: Add `pending_tx_edge_deletes`** to `LpgStore` in `mod.rs` next to `pending_tx_deletes` (line 534): `pub(crate) pending_tx_edge_deletes: RwLock<FxHashMap<TransactionId, Vec<(NodeId, EdgeId, NodeId)>>>` (src, edge, dst), initialized empty in the constructor.
- [ ] **Step 3: Rework `delete_edge_transactional`** (both cfg variants, `edge_ops.rs:471`/`545`): change `chain.mark_deleted(epoch, transaction_id)` → `chain.mark_deleted(EpochId::PENDING, transaction_id)`; **remove the eager `forward_adj.mark_deleted` / `backward.mark_deleted`**; instead push `(src, id, dst)` to `self.pending_tx_edge_deletes`. Keep capturing whatever the undo path needs (mirror how `delete_node_transactional` was reworked in increment 1 — it dropped the eager removal and the heavy undo entry in favor of `unmark_deleted_by` on rollback).
- [ ] **Step 4: Add `finalize_edge_deletes_by_id` / `take_pending_edge_deletes` / `rollback_pending_edge_deletes`** in `edge_ops.rs`, mirroring the node versions (`node_ops.rs:1131`/`1246`/`1276`):
  - `finalize_edge_deletes_by_id(tx, commit_epoch, edges: &[(NodeId, EdgeId, NodeId)])`: per edge `chain.finalize_deleted_epochs(tx, commit_epoch)` (the edge version chain — use the same method node-deletes use) AND apply the deferred adjacency: `self.forward_adj.mark_deleted(src, eid)` + `backward.mark_deleted(dst, eid)`; decrement live-edge count / edge-type count consistently (check what the eager path did).
  - `take_pending_edge_deletes(tx) -> Vec<(NodeId, EdgeId, NodeId)>`.
  - `rollback_pending_edge_deletes(tx, edges)`: `chain.unmark_deleted_by(tx)` per edge (adjacency untouched → nothing to restore).
- [ ] **Step 5: Run the store test GREEN**; `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --features lpg edge_delete_pending_isolates`. Commit.

---

## Task 2: Route transactional `DETACH` through the deferred edge-delete path

**Files:** `node_ops.rs` (`delete_node_edges`)

- [ ] **Step 1.** `delete_node_edges` (the DETACH path) currently uses `TransactionId::SYSTEM` + eager `batch_mark_deleted`. Add/condition a transactional variant: when called within a transaction (thread `transaction_id` in, as `delete_node_transactional` already is for the node), delete each incident edge via `delete_edge_transactional(eid, _, transaction_id)` (PENDING + deferred adjacency) instead of the eager SYSTEM path. Find the call site that invokes `delete_node_edges` during a transactional `DETACH DELETE` (in the node-delete operator / session) and ensure it passes the transaction context. Mirror how `delete_node_transactional` gets its `transaction_id`.
- [ ] **Step 2: Gate.** The `uncommitted_detach_delete_edges_invisible_to_other_sessions` probe needs both this AND Task 1. After Task 2, run it: `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation uncommitted_detach_delete_edges` → should pass once commit/rollback wiring (Task 3) lands; if it needs Task 3 first, note that and proceed. Commit.

---

## Task 3: Wire commit/rollback/conflict for edge-deletes + trait methods

**Files:** `graph/traits.rs` + `graph_store_impl.rs` (trait methods), `session/mod.rs` (wiring)

- [ ] **Step 1: Trait methods.** Add `finalize_edge_deletes_by_id(&self, tx, commit_epoch, edges: &[(NodeId, EdgeId, NodeId)])` and `take_pending_edge_deletes(&self, tx) -> Vec<(NodeId, EdgeId, NodeId)>` to `GraphStoreMut` (defaults: no-op / empty), overridden on `LpgStore` (delegate to inherent). Add `rollback_pending_edge_deletes` similarly (or call it directly on the resolved store like increment-1 did for `rollback_pending_deletes`).
- [ ] **Step 2: Commit wiring.** In `session/mod.rs` `commit_inner` success loop, next to `take_pending_deletes`/`finalize_deletes_by_id`, add `let ed = store.take_pending_edge_deletes(transaction_id); store.finalize_edge_deletes_by_id(transaction_id, commit_epoch, &ed);`.
- [ ] **Step 3: Rollback/conflict wiring.** In `rollback_inner` and the commit-conflict branch, next to `rollback_pending_deletes`, add `let ed = store.take_pending_edge_deletes(transaction_id); store.rollback_pending_edge_deletes(transaction_id, &ed);`.
- [ ] **Step 4: Gate.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation uncommitted_edge_delete committed_edge_delete uncommitted_detach_delete_edges` → ALL GREEN. Full `--all-features -p grafeo-core -p grafeo-engine` green. Commit.

---

## Task 4: Wrapper delegation

**Files:** `compact/layered.rs`, `database/cdc_store.rs`, `wal_store.rs`

- [ ] Delegate `finalize_edge_deletes_by_id` / `take_pending_edge_deletes` (and `rollback_pending_edge_deletes` if on the trait) through `LayeredStore` (to inner overlay), `CdcStore`/`WalStore` (to inner) — mirroring increment-1 Task 6 and 2a Task 5. The delete path on `LayeredStore` must defer adjacency on the overlay consistently. Gate: full `--all-features` + `--features full` green; clippy clean. Commit.

---

## Task 5: Full verification

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green; `--features full -p grafeo-engine --test mvcc_isolation --test audit_scratch --test savepoint_undo --test regression_external` green (incl. the un-ignored DETACH-edges probe + the MERGE-dedup tests).
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` (`wasm32-unknown-unknown`) compile.
- [ ] `git status` clean (only intended files; revert collateral fmt). OPSEC: scan the diff for the private project's name/schema.

---

## Acceptance
- Uncommitted edge delete (direct `DELETE r` and `DETACH DELETE` of a node with edges) is invisible to other sessions' `neighbors`/`edges_from`/edge-property reads but gone for the writer; commit applies the adjacency tombstone at the commit epoch; rollback restores the edge. The increment-1 `#[ignore]`d DETACH-edges probe passes un-ignored.
- Full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; tree clean; OPSEC-clean.
- The isolation model is now uniform across existence, properties, labels, node-deletes, AND edge-deletes on `LpgStore` + `LayeredStore`.

## Risks
- **Adjacency visibility for the writer.** With the tombstone deferred, the writer must still see the edge as gone — verify it does so via the edge version chain's `deleted_by == tx` in `edges_from`/`neighbors` (read the read path; if those consult only adjacency, not the chain, the writer would still see the edge — in that case the deferred model needs the read path to also check chain `visible_to`, mirroring the node case). If the read path is adjacency-only, escalate before hacking.
- **live_edge_count / edge-type counts.** The eager path decremented at delete; move to finalize (or document approximation) — mirror the node-delete `live_node_count` handling.
- **DETACH threading.** `delete_node_edges` must receive the transaction context; if the call chain doesn't currently thread it, that's the main integration work — keep it minimal.
- **Savepoints.** Edge-delete pending set is tx-granular like node-deletes; savepoint partial rollback of edge-deletes is deferred unless a test forces it (mirror increment-1's savepoint note).
