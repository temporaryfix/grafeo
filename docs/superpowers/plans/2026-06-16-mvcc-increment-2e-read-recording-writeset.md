# Increment 2e — Read-recording + store-derived write-set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the *feed* for the existing commit-time serializable validation (`manager.rs:334-377`): (1) populate a Serializable transaction's **read-set** by recording every entity it reads at the scan/expand producers — via a new `ReadTracker` trait that mirrors the existing `WriteTracker`; and (2) populate its **complete write-set** by deriving it from the store-level commit chokepoints (complete by construction). Both are **inert for non-Serializable transactions** and remain end-to-end dormant until Plan 3 (Part F) removes the session-level Serializable rejection — 2e builds and unit/component-tests the tracking, it does **not** enable Serializable.

**Architecture:**
- **Read recording (Part D tracking half, Decision A):** a `ReadTracker` trait in `grafeo-core` (`record_node_read`/`record_edge_read`, mirroring `WriteTracker`); a `TransactionReadTracker` bridge in `grafeo-engine` forwarding to `TransactionManager::record_read`. The planner creates the read-tracker **only when the transaction's isolation is Serializable** (so SI/RC never allocate or call it — zero fast-path cost) and threads it (`with_read_tracker`) into the entity-producing operators (scan + expand families), which call it as they materialize visible ids.
- **Write-set (Part E, Decision B):** at commit, **before** the manager's validation runs, union the store chokepoints — `pending_tx_creates ∪ pending_tx_deletes ∪ pending_tx_edge_deletes ∪ entities touched in tx_property_overlay ∪ entities touched in the label delta` — into `TransactionInfo.write_set` (complete by construction). The eager first-writer-wins `record_write` stays as-is for its write-time conflict behavior; this only *completes* the set for validation.
- **Deferred to 2f** (per scope decision): the sharded **read-registry** + GC. Its only consumer is the F2 SSI write-time rw-edge detection (Plan 3); building it now ships unconsumed infra (YAGNI). `record_read` in 2e populates the read-set only.

**Tech Stack:** Rust, `cargo test`. Trait in `grafeo-core` (`execution/operators/mod.rs`); bridge in `grafeo-engine` (`transaction/read_tracker.rs`); planner threading (`query/planner/lpg/{mod,scan,expand,...}.rs`); operators (`execution/operators/{scan,range_scan,parameter_scan,expand,variable_length_expand,factorized_expand}.rs`); write-set in `transaction/manager.rs` + `session/mod.rs` commit path. `CARGO_INCREMENTAL=0`.

---

## Orientation

This is **Part D's tracking half + Part E** of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§6 record_read, §7 write-set, §10 Decisions A/B). Increment **2d** (merged to `integration` @ `13a09d93`) completed Part D's read *routing* — every read site now carries `(viewing_epoch, transaction_id)`. 2e instruments those producers to *record* the reads and completes the write-set. The sharded read-registry (also in spec §6) is **deferred to 2f** (its consumer is F2/Plan 3).

**What already exists (build on, do not rebuild):**
- **The validation is already there, just unfed.** `TransactionInfo` has `read_set` + `write_set` (`manager.rs:97-99`); `TransactionManager::record_read` (`manager.rs:261`) inserts into `read_set`; `record_write` (`:180`, first-writer-wins) and `record_entity` (`:233`, no-conflict) insert into `write_set`; `commit` (`:300`) already runs write-write validation (`:334-347`) and, for Serializable, read-write validation (`:358-377`). `record_read` is currently called **only from tests** — the production query path never records reads. There is `write_set(tx)` getter (`:433`) and `set_write_set` (`:452`).
- **`WriteTracker` pattern to mirror exactly:** trait `WriteTracker` (`execution/operators/mod.rs:132`) with `record_node_write(tx, NodeId)` / `record_edge_write(tx, EdgeId)`; `SharedWriteTracker = Arc<dyn WriteTracker>` (`:157`). Bridge `TransactionWriteTracker` (`grafeo-engine/src/transaction/write_tracker.rs`) forwards to `manager.record_write`, mapping errors to `OperatorError::WriteConflict`. Planner holds `write_tracker: Option<SharedWriteTracker>` (`planner/lpg/mod.rs:203`), creates it when `transaction_id.is_some()` (`:268-275`), and threads it via `op.with_write_tracker(Arc::clone(tracker))` (only in `planner/lpg/mutation.rs`). Re-exported via `pub use write_tracker::TransactionWriteTracker` (`transaction/mod.rs:207`).
- **`isolation_level(tx)`** on the manager (`manager.rs:163`) → the planner can gate read-tracker creation to Serializable-only.
- **Scan/expand producers already carry the snapshot** (`viewing_epoch`/`transaction_id` + `with_transaction_context`), from prior increments — `read_tracker` threads in the same way. `ScanOperator` (`scan.rs`) materializes visible node ids into `self.batch` after `filter_visible_node_ids_versioned` (~`scan.rs:102-110`) then pushes them to the output column (`col.push_node_id(self.batch[i])`, ~`:140`) — the entity-granularity `record_read` point.
- **Commit chokepoints** (the write-set source) are drained in `session/mod.rs` commit (~`:4085-4105`): `take_pending_creates`, `apply_tx_overlay`, `take_pending_edge_deletes`, `take_pending_deletes` + `finalize_*`. The `TxDelta` (`graph/lpg/store/mod.rs:154`) holds `node_props`/`edge_props`/`node_labels` (the overlay+label entities). A **non-draining** `pending_node_creates(tx)` accessor already exists (added in 2c, `store/mod.rs`); 2e adds sibling non-draining peeks for the other chokepoints.
- **Serializable is rejected at the session layer** (`session/mod.rs:3936-3939`). **Leave it** — Plan 3 removes it. The **manager accepts** `begin_with_isolation(Serializable)` directly (existing manager tests use it), so 2e's tests run at the manager/operator-internal level.

**Testing strategy (2e is inert end-to-end).** Because Serializable is session-rejected, test at the component/manager level, not via session SQL:
- **Read recording:** operator-level — build a `ScanOperator`/`ExpandOperator` with `.with_read_tracker(tracker)` and a Serializable tx, run `next()`, assert the tracker recorded exactly the visible ids (use a test-double `ReadTracker` that collects calls, or the real bridge against a manager with a `begin_with_isolation(Serializable)` tx then assert `manager.write_set`/read-set via a getter). Also assert **no** recording when the tx is SI/RC (tracker is `None` / not created).
- **Write-set:** manager/session-level — after a tx that creates/deletes/sets-props/sets-labels, assert `manager.write_set(tx)` equals the chokepoint union (every touched entity present). If a current mutation path already covers it via eager `record_write`, the test still asserts completeness (defense-in-construction); if a path is missing today, the test is RED until the derivation lands.

**Verification gate (every task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine`. Keep `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean. **Hygiene:** `rustfmt` only changed files; `git status` shows only intended files before commit. Commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. **OPSEC:** generic labels (`:Thing`/`:N`) only. **Stay in lane:** no read-registry, no GC, no enabling Serializable (those are 2f / Plan 3).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-core/src/execution/operators/mod.rs` | tracker traits | Modify (add `ReadTracker` + `SharedReadTracker` next to `WriteTracker`) |
| `crates/grafeo-engine/src/transaction/read_tracker.rs` | bridge | Create (mirror `write_tracker.rs`; Serializable-gated forward to `record_read`) |
| `crates/grafeo-engine/src/transaction/mod.rs` | module | Modify (`mod read_tracker; pub use read_tracker::TransactionReadTracker`) |
| `crates/grafeo-engine/src/query/planner/lpg/mod.rs` | planner state | Modify (add `read_tracker` field; create when Serializable) |
| `crates/grafeo-engine/src/query/planner/lpg/{scan,expand}.rs` | planner wiring | Modify (thread `with_read_tracker` to producers) |
| `crates/grafeo-core/src/execution/operators/scan.rs`, `range_scan.rs`, `parameter_scan.rs` | node producers | Modify (`read_tracker` field + builder + record at materialization) |
| `crates/grafeo-core/src/execution/operators/expand.rs`, `variable_length_expand.rs`, `factorized_expand.rs` | edge/neighbor producers | Modify (same) |
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` (+ `graph/traits.rs`, wrappers) | non-draining chokepoint peeks | Modify (add peeks mirroring `pending_node_creates`) |
| `crates/grafeo-engine/src/transaction/manager.rs` | write-set completion | Modify (bulk `extend_write_set` helper if cleaner than N× `record_entity`) |
| `crates/grafeo-engine/src/session/mod.rs` | commit path | Modify (derive+set write-set from chokepoints before validation) |
| `crates/grafeo-engine/tests/serializable_tracking.rs` | acceptance | Create (component/manager-level read-set + write-set tests) |

---

## Task 1: `ReadTracker` trait (grafeo-core)

**Files:** `crates/grafeo-core/src/execution/operators/mod.rs`

- [ ] **Step 1: Write the failing test** (in `mod.rs`'s test module or a focused unit test): a trivial in-memory `ReadTracker` impl collects `(tx, NodeId)`/`(tx, EdgeId)` calls; assert the trait is object-safe (`let _: Arc<dyn ReadTracker> = ...`). Expected FAIL: `ReadTracker` not defined.
- [ ] **Step 2: Add the trait**, mirroring `WriteTracker` (`mod.rs:132-157`) exactly:
```rust
/// Trait for recording read operations during query execution (Serializable only).
///
/// Bridges `grafeo-core` read operators with `grafeo-engine`'s `TransactionManager`
/// (which tracks read-sets for serializable conflict detection). Lives in `grafeo-core`
/// to avoid a circular dependency, mirroring [`WriteTracker`]. Only attached for
/// Serializable transactions, so SI/ReadCommitted pay nothing.
pub trait ReadTracker: Send + Sync {
    /// Records that `transaction_id` read node `node_id` (at its snapshot).
    fn record_node_read(&self, transaction_id: TransactionId, node_id: NodeId);
    /// Records that `transaction_id` read edge `edge_id` (at its snapshot).
    fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId);
}

/// Type alias for a shared read tracker.
pub type SharedReadTracker = Arc<dyn ReadTracker>;
```
(Note: unlike `WriteTracker`, these return `()` — a recorded read never fails/conflicts; recording is best-effort/over-approximating. Confirm `Arc`, `TransactionId`, `NodeId`, `EdgeId` are imported in `mod.rs` — `WriteTracker` already uses them.)
- [ ] **Step 3: Run** the test → PASS. `cargo test -p grafeo-core` green; clippy clean.
- [ ] **Step 4: Commit** (`feat(mvcc): ReadTracker trait mirroring WriteTracker`).

---

## Task 2: `TransactionReadTracker` bridge (grafeo-engine), Serializable-gated

**Files:** Create `crates/grafeo-engine/src/transaction/read_tracker.rs`; modify `transaction/mod.rs`

- [ ] **Step 1: Write the failing test** (in `read_tracker.rs`): build a `TransactionManager`, `begin_with_isolation(Serializable)` → tx_s; `begin()` (SI) → tx_si. A `TransactionReadTracker::new(Arc::clone(&mgr))`; call `record_node_read(tx_s, n)` then assert the manager's read-set for tx_s contains `n` (add/confirm a test-only read-set getter on the manager, mirroring `write_set(tx)` at `manager.rs:433`); call `record_node_read(tx_si, n2)` and assert tx_si's read-set is **empty** (the bridge skips non-Serializable). Expected FAIL: `TransactionReadTracker` not defined.
- [ ] **Step 2: Add a read-set getter** on `TransactionManager` (mirror `write_set`, `manager.rs:433`), so tests (and the bridge test) can assert:
```rust
/// Returns a copy of the read-set of a transaction (for tests / serializable validation introspection).
pub fn read_set(&self, transaction_id: TransactionId) -> HashSet<EntityId> {
    self.transactions.read().get(&transaction_id).map(|i| i.read_set.clone()).unwrap_or_default()
}
```
- [ ] **Step 3: Add the bridge**, mirroring `TransactionWriteTracker` (`write_tracker.rs`) but **gated on Serializable** and infallible:
```rust
//! Bridge between read operators and the transaction manager's read tracking.
use std::sync::Arc;
use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_core::execution::operators::ReadTracker;
use super::{IsolationLevel, TransactionManager};

/// Implements [`ReadTracker`] by forwarding to [`TransactionManager::record_read`].
/// No-op unless the transaction is Serializable (one isolation check).
pub struct TransactionReadTracker {
    manager: Arc<TransactionManager>,
}
impl TransactionReadTracker {
    pub fn new(manager: Arc<TransactionManager>) -> Self { Self { manager } }
}
impl ReadTracker for TransactionReadTracker {
    fn record_node_read(&self, transaction_id: TransactionId, node_id: NodeId) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let _ = self.manager.record_read(transaction_id, node_id);
        }
    }
    fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let _ = self.manager.record_read(transaction_id, edge_id);
        }
    }
}
```
(The planner will also gate *creation* on Serializable — Task 3 — so in production this `isolation_level` check is belt-and-suspenders; keep it so the bridge is correct in isolation. `record_read` returns `Result`; a non-active tx errors — ignore via `let _ =`, recording is best-effort.)
- [ ] **Step 4: Wire the module** — `transaction/mod.rs`: `mod read_tracker;` + `pub use read_tracker::TransactionReadTracker;` (mirror the `write_tracker` lines at `:207`).
- [ ] **Step 5: Run** the test → PASS (tx_s read-set has `n`; tx_si empty). Full `-p grafeo-engine` green; clippy clean.
- [ ] **Step 6: Commit** (`feat(mvcc): TransactionReadTracker bridge (Serializable-gated record_read)`).

---

## Task 3: Planner — create the read-tracker (Serializable only) + plumbing

**Files:** `crates/grafeo-engine/src/query/planner/lpg/mod.rs`

- [ ] **Step 1: Write the failing test:** a planner built `with_context(... Serializable tx ...)` exposes a `Some(read_tracker)`; built with an SI tx exposes `None`. (If `read_tracker` is private, assert indirectly via a scan plan recording reads — but a direct field check in a `#[cfg(test)]` within the planner module is simplest.) Expected FAIL: no `read_tracker` field.
- [ ] **Step 2: Add the field + creation.** In `Planner` add `read_tracker: Option<grafeo_core::execution::operators::SharedReadTracker>` (next to `write_tracker`, `mod.rs:203`); set `None` in `new` (`:249`); in `with_context` (`:258-295`) create it **only for Serializable**:
```rust
use crate::transaction::TransactionReadTracker;
let read_tracker: Option<SharedReadTracker> = match transaction_id {
    Some(tid) if transaction_manager.isolation_level(tid) == Some(IsolationLevel::Serializable) =>
        Some(Arc::new(TransactionReadTracker::new(Arc::clone(&transaction_manager)))),
    _ => None,
};
```
and add `read_tracker,` to the struct literal (`:295`). (Confirm `IsolationLevel`/`SharedReadTracker` imports.)
- [ ] **Step 3: Run** → PASS. Full `-p grafeo-engine` green; clippy clean.
- [ ] **Step 4: Commit** (`feat(mvcc): planner creates read-tracker for Serializable txns`).

---

## Task 4: Record reads at the node producers (scan family)

**Files:** `execution/operators/{scan,range_scan,parameter_scan}.rs`; planner `query/planner/lpg/scan.rs`

- [ ] **Step 1: Write the failing test** (operator-level, in `scan.rs` tests): a test-double `ReadTracker` (collects node ids); build a `ScanOperator` over a store with nodes, `.with_transaction_context(epoch, Some(tx))`, `.with_read_tracker(Arc::new(double))`; drain `next()`; assert the double collected exactly the visible node ids. Expected FAIL: no `with_read_tracker`.
- [ ] **Step 2: Add `read_tracker` field + builder** to `ScanOperator` (mirror `with_transaction_context`, `scan.rs:67`): field `read_tracker: Option<SharedReadTracker>` (default `None`); `pub fn with_read_tracker(mut self, t: SharedReadTracker) -> Self { self.read_tracker = Some(t); self }`.
- [ ] **Step 3: Record at materialization.** Where the visible batch is finalized (after `filter_visible_node_ids_versioned`, ~`scan.rs:102-110`), record each id when a tx + tracker are present:
```rust
if let (Some(tracker), Some(tid)) = (&self.read_tracker, self.transaction_id) {
    for id in &self.batch {
        tracker.record_node_read(tid, *id);
    }
}
```
(Record the **visible** batch — the ids the tx actually observes. Place it once where `self.batch` is set, not in the per-row `push` loop, to avoid double-recording across `next()` chunks. Confirm `self.batch` is set once per scan, not per chunk; if chunked, record on first materialization only.)
- [ ] **Step 4: Apply the same field+builder+record pattern** to `range_scan.rs` and `parameter_scan.rs` (their visible-id materialization points — find where each emits node ids under a tx). If a scan variant has no clear single materialization point, record at the per-id emit guarded against duplicates, and note it.
- [ ] **Step 5: Plumb from the planner** (`query/planner/lpg/scan.rs:23/53` — where scan ops get `with_transaction_context`): after that, `if let Some(t) = &self.read_tracker { op = op.with_read_tracker(Arc::clone(t)); }`. Do this at each scan construction site.
- [ ] **Step 6: Run** → the scan test PASSes; also assert an SI-context scan records nothing (tracker `None`). Full `--all-features -p grafeo-core -p grafeo-engine` green; clippy clean.
- [ ] **Step 7: Commit** (`feat(mvcc): record_read at scan-family node producers`).

---

## Task 5: Record reads at the edge/neighbor producers (expand family)

**Files:** `execution/operators/{expand,variable_length_expand,factorized_expand}.rs`; planner `query/planner/lpg/expand.rs`

- [ ] **Step 1: Write the failing test** (operator-level): a test-double `ReadTracker`; build an `ExpandOperator` over `(a)-[r]->(b)` with a tx + tracker; run; assert it recorded the traversed **edge** ids (and the neighbor **node** ids it materializes, if the operator emits them). Expected FAIL: no `with_read_tracker`.
- [ ] **Step 2: Add `read_tracker` field + `with_read_tracker` builder** to `ExpandOperator` (mirror its `with_transaction_context`).
- [ ] **Step 3: Record at traversal.** Where expand resolves visible neighbors/edges (it already calls `is_edge_visible_versioned`/`neighbors` under the snapshot — see the expand visibility checks), record each **edge id** it visits and each **neighbor node id** it emits, guarded by `(Some(tracker), Some(tid))`. Record only **visible** edges/neighbors (after the visibility filter), mirroring the scan pattern.
- [ ] **Step 4: Apply the same pattern** to `variable_length_expand.rs` and `factorized_expand.rs` at their visible-edge/neighbor emit points. (Note: `factorized_expand` may, like `factorized_filter` in 2d, be only partially reachable — if you find it has no production construction site, add the field+builder+record for completeness and verify by inspection; record the finding.)
- [ ] **Step 5: Plumb from the planner** (`query/planner/lpg/expand.rs:73/93/211-214`): after each `with_transaction_context`, add `if let Some(t) = &self.read_tracker { op = op.with_read_tracker(Arc::clone(t)); }` (and the lazy/variable-length builder sites).
- [ ] **Step 6: Run** → expand test PASSes. Full `--all-features` green; clippy clean.
- [ ] **Step 7: Commit** (`feat(mvcc): record_read at expand-family edge/neighbor producers`).

---

## Task 6: Store-derived complete write-set (Part E)

**Files:** `graph/lpg/store/mod.rs` (+ `graph/traits.rs` + wrappers `layered.rs`/`cdc_store.rs`/`wal_store.rs`), `transaction/manager.rs`, `session/mod.rs`, `crates/grafeo-engine/tests/serializable_tracking.rs`

- [ ] **Step 1: Write the failing/asserting test** (manager/session-level): run a Serializable tx (`begin_with_isolation(Serializable)` via the manager, with session mutators) that (a) creates a node, (b) sets a property on a pre-existing node, (c) sets a label on another, (d) deletes another, then — **before commit finalizes** — assert `manager.write_set(tx)` contains **all four** entities. Identify which (if any) are missing today (the gap Part E closes): a pure label-set or a path that doesn't call `record_write` is the likely gap. If all four are already present via eager `record_write`, keep the test as a completeness guarantee (it must stay green as a structural invariant). Expected: RED if a gap exists; otherwise a green invariant the derivation must preserve.
- [ ] **Step 2: Add non-draining chokepoint peeks** on `LpgStore` (mirror the existing non-draining `pending_node_creates`): `pending_edge_creates(tx)`, `pending_node_deletes_peek(tx)`, `pending_edge_deletes_peek(tx)`, `overlay_touched_entities(tx) -> (Vec<NodeId>, Vec<EdgeId>)` (keys of `tx_property_overlay`'s `node_props`/`edge_props` + `node_labels` node ids). Add to the store trait (`graph/traits.rs`) with default-empty, override on `LpgStore`, delegate in `layered.rs`/`cdc_store.rs`/`wal_store.rs` (mirror how `pending_node_creates` is delegated). Keep them **non-draining** — commit's existing `take_*`/`finalize_*` still consume.
- [ ] **Step 3: Derive + set the write-set at commit, before validation.** In `session/mod.rs` commit, **before** the `manager.commit(tx)` call (confirm the ordering — validation happens inside `manager.commit`, so the write-set must be complete first; the chokepoints are still present pre-`take_*`), gather the union from the peeks across the touched stores and add them to the write-set. Add a manager helper to extend without conflict (mirror `record_entity` but bulk):
```rust
// manager.rs
pub fn extend_write_set(&self, transaction_id: TransactionId, entities: impl IntoIterator<Item = EntityId>) {
    if let Some(info) = self.transactions.write().get_mut(&transaction_id) {
        if info.state == TransactionState::Active { info.write_set.extend(entities); }
    }
}
```
Then in session commit: build `Vec<EntityId>` from `pending_node_creates ∪ pending_edge_creates ∪ pending_node_deletes_peek ∪ pending_edge_deletes_peek ∪ overlay_touched_entities` for each touched store and call `manager.extend_write_set(tx, union)` **before** `manager.commit(tx)`. (Confirm the exact pre-commit hook point; if `manager.commit` is called from a place without store access, gather in the session method that has both and set the write-set just prior.)
- [ ] **Step 4: Run** → the write-set test PASSes (all touched entities present). Confirm no existing write-conflict test regressed (the eager `record_write` path is unchanged). Full `--all-features` green; clippy clean.
- [ ] **Step 5: Commit** (`feat(mvcc): store-derived complete write-set from commit chokepoints`).

---

## Task 7: Full verification + OPSEC

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green (incl. the new `serializable_tracking` tests). `--features full -p grafeo-engine --test mvcc_isolation` still green (2d behavior unchanged).
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` (wasm32) compile.
- [ ] **Inertness audit:** confirm SI/ReadCommitted transactions allocate **no** read-tracker and record nothing (planner creates it only for Serializable); confirm the read-set/write-set additions are behind the Serializable gate or are inert (write-set derivation runs for all txns but only *matters* for Serializable validation — confirm it doesn't change non-Serializable commit behavior). Confirm Serializable is **still rejected at the session** (`session/mod.rs:3936`) — 2e does not enable it.
- [ ] `git status` clean (only intended files; revert collateral fmt). OPSEC: generic labels only.

---

## Acceptance
- A Serializable transaction's reads at scan/expand producers are recorded into its read-set (component tests); SI/RC record nothing (no tracker created).
- The write-set is complete by construction from the commit chokepoints (creates ∪ deletes ∪ edge-deletes ∪ property-overlay ∪ label-delta), verified by the tracking test; eager first-writer-wins behavior unchanged.
- Everything inert end-to-end: Serializable stays session-rejected; SI/RC fast path unchanged; full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; OPSEC-clean.
- **The OCC/SSI validation feed (read-set + write-set) is complete** → next: **2f** (the sharded read-registry + GC), then **Plan 3** (F1: remove the session rejection so the existing `manager.rs:358-377` validation goes live → *sound serializable*; F2: incremental SSI over the registry + G performance).

## Risks
- **Missed producer = incomplete read-set.** Sound only once Serializable is enabled (Plan 3) — but 2e should still record the common scan/expand paths. The comprehensive read-site sweep + the "every execution read hits the tracker" audit is part of the road to F1; note any producer you couldn't instrument (e.g., vector/text scans that bypass scan/expand) for the 2f sweep. Over-approximation is sound; under is not.
- **Double-recording across chunked `next()`.** Producers emit in chunks; record at the once-per-scan visible-id materialization, not per emitted row, or dedupe. The read-set is a `HashSet` so duplicates are harmless for correctness, but avoid O(rows) tracker calls on hot paths — record the batch once.
- **Write-set timing.** The union must be set **before** `manager.commit`'s validation and read from the chokepoints **before** they're drained by `take_*`/`finalize_*`. Confirm the exact ordering in `session/mod.rs` commit; the peeks are non-draining so they can run just prior.
- **Layering.** The store (grafeo-core) cannot call the manager (grafeo-engine); that's why read recording is operator-level (`ReadTracker`, Decision A) and the write-set is gathered at the session (which holds both), not in the store. Don't try to populate the manager from the store.
- **Reachability (expand/factorized variants).** As in 2d, some producers may be unreachable from production plans; thread + record for completeness, verify by inspection, and report — don't invent wiring.
- **Don't enable Serializable, don't build the registry.** Those are Plan 3 and 2f. Keep 2e to "populate read-set + write-set, inert."

---

## STATUS: COMPLETE

All 7 tasks landed on `feat/mvcc-increment-2e` (commits `83cf9ec3` → `835100d9`, branched from `integration` @ `13a09d93`), via subagent-driven-development (fresh implementer per task + spec review) and a final holistic review.

**Final verification:**
- `--all-features -p grafeo-core -p grafeo-engine` = **7422 passed / 0 failed** (122 binaries; 7411 after 2d + 11 new 2e tests).
- Established clippy gate (`--all-features -p grafeo-core -p grafeo-engine -- -D warnings`) **clean**.
- Profiles: **default / lpg / temporal / tiered-storage** all compile; **`grafeo-wasm` (wasm32-unknown-unknown)** compiles.
- OPSEC-clean (generic `:ExtraLabel` etc.); **no new `TODO(unified-mvcc)`**; no registry/F1/F2/GC code (scope-disciplined).
- Final holistic review (opus): **READY TO MERGE** — no Critical/Important.

**What landed:**
- **Read recording (inert for non-Serializable):** `ReadTracker` trait (grafeo-core) + `TransactionReadTracker` bridge (Serializable-gated) + planner creates the tracker only for Serializable txns + `record_read` at the scan family (`ScanOperator`/`RangeScanOperator`; `ParameterScanOperator` correctly excluded) and the expand family (`ExpandOperator`/`VariableLengthExpandOperator`/`FactorizedExpand*`), recording visible ids once per scan, post-visibility-filter.
- **Write-set completion (active for all txns):** non-draining chokepoint peeks (`pending_edge_creates`/`pending_node_deletes_peek`/`pending_edge_deletes_peek`/`overlay_touched_entities`) across store/trait/wrappers + `manager.extend_write_set`; unioned into the write-set in `session::commit` **before** validation.

**Key findings:**
- **Task 5 spec review caught a missed planner site** — the factorized-aggregate fast-path (`planner/lpg/aggregate.rs:399`) also constructs a `LazyFactorizedChainOperator` and was missing `with_read_tracker`; fixed (all 6 producer-attach sites now covered). A missed site = a silent read-set gap when Serializable lands.
- **Part E fixes a latent SI lost-update bug (verified sound by an opus review).** Session-direct CRUD (`create_node`/`delete_node`/`set_node_property`) **never called `record_write`**, so those entities were absent from the write-set — meaning two concurrent SI transactions modifying the same pre-existing entity via session-CRUD both committed (lost update). Completing the write-set from chokepoints feeds the commit-time write-write check (`manager.rs:334-347`, all isolation levels), so they now correctly conflict (first-updater-wins). No false conflicts (creates get unique monotonic ids), no over-abort (gated on concurrent committers), consistent with the already-conflicting operator/GQL path; 8 concurrency/isolation test files re-run, no outcome flips.
- **Inert vs active:** read recording is fully inert (Serializable still session-rejected at `session/mod.rs:3939`; SI/RC allocate no tracker). The write-set completion is **active** for all txns (the bug-fix above) — so 2e is not *entirely* inert, but the active part is a sound correctness improvement.

**Non-blocking follow-ups (a read-site sweep before Plan 3 enables Serializable):**
- Vector/text scans (`scan_vector.rs`/`scan_text.rs`) are uninstrumented producers (and not yet MVCC-snapshot-aware) — absent from the read-set when Serializable lands.
- `VariableLengthExpandOperator` records only the final hop of each emitted path; interior hops below `min_depth` are under-recorded for `min_depth > 1`.
- Pre-existing `clippy::large_stack_arrays` under `--all-targets` in untouched `graph/compact/column.rs:3334` (noted since 2d).

**Next: increment 2f** — the sharded read-registry + GC (Part D's remaining piece; its consumer is F2 in Plan 3). Then **Plan 3**: F1 (remove the session Serializable rejection so the existing `manager.rs:358-377` validation goes live → *sound serializable*), F2 (incremental SSI over the registry), G (performance). The read-site sweep above should land before F1.
