# Store-Level Read-Recording (Sound SSI Read-Set Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-architect Serializable read-recording from **per-operator** (increment 2e's Decision A — fragile, a denylist that can never be confidently complete) to the **store's visible-read API chokepoint** — so a Serializable transaction's read-set captures *every* snapshot-consistent read **by construction**, regardless of which operator issued it. This makes serializability soundness an auditable property of a small, bounded API surface instead of a per-operator audit that must be re-run forever. The operators that *bypass* the visible-read API (vector/text scans, `shortestPath`, algos) become a distinct, finite **integrate-or-guard** class.

**Why:** Executing increment 2f surfaced that per-operator recording is fragile — the audit found `ShortestPath`, `MERGE`, `MergeRelationship`, and `CallProcedure` as reachable reads the 2e scan/expand instrumentation missed. The project already learned this for *writes*: the spec's Decision B derived the write-set from store chokepoints precisely because re-threading `record_write` through every operator was "the fragility Wave 2a fled." Reads have the same shape; this applies the same fix.

**Architecture:**
- The store (`LpgStore`) holds a per-transaction read-tracker registry `read_trackers: RwLock<FxHashMap<TransactionId, SharedReadTracker>>` (mirroring `tx_property_overlay`). The engine **registers** the existing `TransactionReadTracker` (2e's bridge) when a Serializable transaction begins and **unregisters** it at commit/rollback.
- Every **visible-read accessor** (the bounded chokepoint surface, below) records the visible entity into the registered tracker. Because every snapshot-consistent read physically flows through one of these, the read-set is complete by construction.
- The now-redundant **operator-level `record_read`** (2e's scan/expand wiring + 2f's parked var-length recording) is **removed** — the store records instead.
- The **non-MVCC-aware operators** (which read raw adjacency/indexes, bypassing the visible-read API — `shortestPath`, vector/text scans, algos `CallProcedure`) are **guarded** under Serializable (sound: reject what we can't track). Each is later MVCC-integrated to remove its guard.
- **Reuses all of 2e's infrastructure unchanged** — the `ReadTracker` trait, the `TransactionReadTracker` bridge, the manager read-set + commit-time validation. Only the *call site* moves (store accessors, not operators).

**Tech Stack:** Rust, `cargo test`. Store registry in grafeo-core (`graph/lpg/store/mod.rs` + `graph/traits.rs` + wrappers); accessor instrumentation (`graph/lpg/store/{property_ops,node_ops,edge_ops,…}.rs`); engine registration (`session/mod.rs` + transaction lifecycle); operator-recording removal (`execution/operators/{scan,range_scan,expand,factorized_expand,variable_length_expand}.rs` + `query/planner/lpg/*`); guards (`query/planner/lpg/{mutation,mod}.rs`). `CARGO_INCREMENTAL=0`.

---

## Scope decision (read first)

This **supersedes** increment 2e's operator-level read-recording (Decision A) with store-level recording (the spec's own Decision-B pattern, applied to reads). It does **not** enable Serializable — that re-lands as a follow-up (the parked `feat/mvcc-increment-2f` enable + acceptance suite, now sound by construction). It does **not** build the read-registry (2f-track) or F2/G. Inert end-to-end (Serializable stays session-rejected); tested at the component/manager level.

**The soundness invariant this establishes:** *every snapshot-consistent read goes through the visible-read API; recording there is complete; an operator that bypasses the API is — by definition — neither snapshot-consistent nor recorded, which is a pre-existing MVCC deficiency to integrate-or-guard.* The audit question becomes the single checkable "does this operator read through the visible-read API?" — not "did we instrument every operator?".

---

## Orientation

Builds on `integration` @ `c8a20726` (2e complete; `docs/research/` gitignore carried over). Design basis: `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` §6/§10 (revises Decision A → store-level) and the parked 2f plan's STATUS note.

**The chokepoint surface (the completeness-critical list — every snapshot read flows through these):**
| Accessor | `graph/traits.rs` | Records |
|---|---|---|
| `get_node_versioned(id, epoch, tx)` | :56 | the node, when visible (`Some`) |
| `get_edge_versioned(id, epoch, tx)` | :64 | the edge, when visible (`Some`) |
| `read_node_property_visible(id, key, epoch, tx)` | :129 | the node |
| `read_node_properties_visible(id, epoch, tx)` | :158 | the node |
| `read_edge_property_visible(id, key, epoch, tx)` | :141 | the edge |
| `read_edge_properties_visible(id, epoch, tx)` | :171 | the edge |
| `read_node_labels_visible(id, epoch, tx)` | :202 | the node |
| `nodes_by_label_visible(label, …, tx)` | :286 | each returned node |
| `filter_visible_node_ids_versioned(ids, epoch, tx)` | :461 | each returned (visible) node |
| `is_node_visible_versioned(id, epoch, tx)` | :420 | the node, when visible (`true`) |
| `is_edge_visible_versioned(id, epoch, tx)` | :440 | the edge, when visible (`true`) |

(Record only the **visible** result — `Some`/`true`/the returned set — i.e. entities the tx actually observed; recording over candidate sets would over-approximate. Phantom/predicate reads — recording observed *non-existence* — are a later refinement, noted but not in scope.)

**Coverage proof (why this is complete for MVCC-aware operators):**
- `ScanOperator` (`scan.rs:108/119`) reads via `nodes_by_label_visible` + `filter_visible_node_ids_versioned` → recorded. `RangeScanOperator` likewise via the versioned filter.
- Expand family resolves visible edges/neighbors via `is_edge_visible_versioned` / `get_*_versioned` → recorded.
- `MERGE`/`MergeRelationship` matching uses `read_node_property_visible` / `get_node_versioned` per candidate (increment 2c) → **auto-recorded** (MERGE is MVCC-aware; the earlier "MERGE gap" was a per-operator artifact that store-level recording closes for free).
- `Filter` property introspection, `SET` source reads, aggregates → all use the visible accessors (2d) → recorded.

**The integrate-or-guard class (bypass the visible-read API → NOT snapshot-consistent, NOT recorded):**
- `shortestPath`/`allShortestPaths` (`shortest_path.rs:179` uses raw `edges_from`, no visibility) — **guard**.
- Vector/text scans (`scan_vector.rs`/`scan_text.rs`, not MVCC-snapshot-aware) — **guard** (2f already had these guards; re-land).
- algos `CallProcedure` (reads the raw graph) — **guard**.

**Per-tx store state to mirror:** `tx_property_overlay: RwLock<FxHashMap<TransactionId, TxDelta>>` (`store/mod.rs:527`), cleared per-tx. The `read_trackers` map mirrors its lifecycle.

**Reused (unchanged):** `ReadTracker` trait (`execution/operators/mod.rs`), `TransactionReadTracker` bridge (`transaction/read_tracker.rs`), manager read-set + validation. The planner's per-operator `read_tracker` creation/plumbing (2e Task 3/4/5) is **removed** in favor of store registration.

**Verification gate:** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine`; clippy `--all-features … -- -D warnings` clean; `default`/`lpg`/`temporal`/`tiered-storage` + `grafeo-wasm` compile. Hygiene: `rustfmt` **changed files only** (NOT `cargo fmt`); `git status` shows only intended files; `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC: generic labels.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` | tracker registry | Modify (`read_trackers` map + `register_read_tracker`/`unregister_read_tracker` + a private `record_node_read`/`record_edge_read` helper + clear on reset) |
| `crates/grafeo-core/src/graph/traits.rs` | trait surface | Modify (register/unregister default no-ops) |
| `crates/grafeo-core/src/graph/lpg/store/{property_ops,node_ops,edge_ops,schema}.rs` | accessor instrumentation | Modify (record at each chokepoint accessor) |
| `crates/grafeo-core/src/graph/compact/layered.rs`, `database/{cdc_store,wal_store}.rs` | wrappers | Modify (delegate register/unregister) |
| `crates/grafeo-engine/src/session/mod.rs` (+ tx lifecycle) | engine registration | Modify (register bridge at Serializable begin; unregister at commit/rollback) |
| `crates/grafeo-core/src/execution/operators/{scan,range_scan,expand,factorized_expand,variable_length_expand}.rs` + `query/planner/lpg/*` | remove operator-level recording | Modify (delete `read_tracker` fields/builders/calls + planner plumbing) |
| `crates/grafeo-engine/src/query/planner/lpg/{mutation,mod}.rs` | non-MVCC guards | Modify (guard `plan_shortest_path`, `plan_vector_scan`, `plan_text_scan`, algos `CallProcedure` under Serializable) |
| `crates/grafeo-engine/tests/serializable_readset.rs` | completeness tests | Create (component/manager-level read-set completeness) |

---

## Task 1: Store read-tracker registry

**Files:** `graph/lpg/store/mod.rs`, `graph/traits.rs`, wrappers (`layered.rs`, `cdc_store.rs`, `wal_store.rs`)

- [ ] **Step 1: Add the registry** to `LpgStore` (next to `tx_property_overlay`, `mod.rs:527`):
```rust
/// Per-transaction read trackers (Serializable only). Set by the engine at
/// Serializable tx begin, dropped at commit/rollback. When present for a tx,
/// the visible-read accessors record observed entities into it (the SSI read-set,
/// complete by construction). Empty for SI/ReadCommitted (zero cost).
read_trackers: RwLock<FxHashMap<TransactionId, SharedReadTracker>>,
```
Initialize `RwLock::new(FxHashMap::default())` in the constructor; `.write().clear()` wherever `tx_property_overlay` is cleared (`mod.rs:725`).
- [ ] **Step 2: register/unregister + record helpers** (inherent on `LpgStore`):
```rust
pub fn register_read_tracker(&self, tx: TransactionId, tracker: SharedReadTracker) {
    self.read_trackers.write().insert(tx, tracker);
}
pub fn unregister_read_tracker(&self, tx: TransactionId) {
    self.read_trackers.write().remove(&tx);
}
#[inline]
fn record_read_node(&self, tx: TransactionId, id: NodeId) {
    if let Some(t) = self.read_trackers.read().get(&tx) { t.record_node_read(tx, id); }
}
#[inline]
fn record_read_edge(&self, tx: TransactionId, id: EdgeId) {
    if let Some(t) = self.read_trackers.read().get(&tx) { t.record_edge_read(tx, id); }
}
```
(The `read()`-then-maybe-call is the SI/RC fast path: empty map → one uncontended read-lock + miss. If this shows on a profile, add an `AtomicBool any_trackers` short-circuit; YAGNI for now.)
- [ ] **Step 3: trait register/unregister** (`graph/traits.rs`, default no-ops so non-LpgStore impls compile); override on `LpgStore` (`graph_store_impl.rs`); delegate in `LayeredStore`/`CdcGraphStore`/`WalGraphStore` (mirror `pending_node_creates` delegation). The `record_read_*` helpers stay private to `LpgStore` (called only by its own accessors).
- [ ] **Step 4: unit test** — register a test-double tracker for a tx; call `store.record_read_node(tx, n)` (via a visible accessor in Task 3, or a `#[cfg(test)]` shim); assert recorded; unregister; assert no-op. Run; commit (`feat(mvcc): store read-tracker registry`).

---

## Task 2: Engine registers the tracker at Serializable tx begin/commit/rollback

**Files:** `crates/grafeo-engine/src/session/mod.rs` (tx begin/commit/rollback)

- [ ] **Step 1: Write the failing test** (component-level): a `GrafeoDB`; via the manager-level Serializable path (manager `begin_with_isolation(Serializable)` — the session rejects Serializable, so the test registers/asserts at the store/manager level, OR a `#[cfg(test)]` helper that begins a Serializable tx bypassing the session guard), run a `MATCH (n) RETURN n` and assert the manager's read-set is populated. Expected FAIL: no registration yet (read-set empty).
- [ ] **Step 2: Register/unregister.** At the point a Serializable transaction begins (`begin_transaction_inner`, after `begin_with_isolation`), if the isolation is Serializable, build the `TransactionReadTracker` (the 2e bridge: `Arc::new(TransactionReadTracker::new(Arc::clone(&self.transaction_manager)))`) and register it on the active store: `store.register_read_tracker(tx, tracker)`. At **every** tx exit — commit success, conflict/serialization rollback, and explicit rollback — call `store.unregister_read_tracker(tx)`. (Grep the commit/rollback paths — `session/mod.rs` `commit`/`rollback_inner`/the conflict branch at `:4031-4051` — and unregister in each. Missing an unregister leaks the tracker entry; bound it by also clearing on store reset.)
- [ ] **Step 3:** Run → GREEN (read-set populated for the Serializable MATCH). Full gate green; clippy clean. Commit (`feat(mvcc): register read-tracker on store for Serializable txns`).

NOTE: this task depends on Task 3 (accessors must record for the read-set to populate). Implement Task 3 first or together; the test in Step 1 goes green once both land.

---

## Task 3: Instrument the visible-read chokepoint surface

**Files:** `graph/lpg/store/{property_ops,node_ops,edge_ops,schema}.rs` (wherever each accessor lives)

For **each** accessor in the Orientation table, when called with `transaction_id = Some(tx)`, record the **visible** entity via `self.record_read_node(tx, id)` / `self.record_read_edge(tx, id)`. Do it where visibility is confirmed (the entity is actually observed), not over the candidate set.

- [ ] **Step 1: Per-entity accessors.** In `get_node_versioned`/`get_edge_versioned` record on the `Some(_)` return; in `read_node_property_visible`/`read_node_properties_visible`/`read_node_labels_visible` (+ edge twins) record the entity unconditionally when `tx` is `Some` (the call *is* the read); in `is_node_visible_versioned`/`is_edge_visible_versioned` record on the `true` return. Find each accessor (`graph/traits.rs` lists the trait sites; the `LpgStore` impls are in `property_ops.rs`/`node_ops.rs`/`edge_ops.rs`/`schema.rs`).
- [ ] **Step 2: Batch accessors.** In `filter_visible_node_ids_versioned` record each id in the returned (visible) `Vec`; in `nodes_by_label_visible` record each returned node.
- [ ] **Step 3: Write the failing/asserting test** alongside Task 2's: a Serializable tx runs `MATCH (n:Thing) RETURN n.v` (scan + property read) and `MATCH (a)-[r]->(b) RETURN r` (expand) → assert the read-set contains the scanned nodes + traversed edges/neighbors. Also: a `MERGE (n:Thing {k:1})` under Serializable records the matched/candidate node (proving MERGE is auto-covered). Run → GREEN.
- [ ] **Step 4:** Full gate green; clippy clean. Commit (`feat(mvcc): record reads at the store visible-read chokepoints`).

---

## Task 4: Remove the now-redundant operator-level recording

**Files:** `execution/operators/{scan,range_scan,expand,factorized_expand,variable_length_expand}.rs`; `query/planner/lpg/{scan,filter,expand,aggregate,mod}.rs`

The store now records all reads; the operator-level `record_read` (2e + parked 2f var-length) is redundant double-work. Remove it to keep the architecture single-sourced and confident.

- [ ] **Step 1:** Delete the `read_tracker: Option<SharedReadTracker>` field + `with_read_tracker` builder + the `record_node_read`/`record_edge_read` call sites from `ScanOperator`, `RangeScanOperator`, `ExpandOperator`, the factorized-expand operators, and `VariableLengthExpandOperator`. (The var-length recording is on the parked 2f branch, NOT on `integration`/this branch — so for var-length there is nothing to remove here; the 2e operator recording IS on integration and must be removed.)
- [ ] **Step 2:** Delete the planner plumbing: the `if let Some(t) = &self.read_tracker { op = op.with_read_tracker(...) }` blocks in `planner/lpg/{scan.rs, filter.rs, expand.rs, aggregate.rs}`. Decide the planner's `read_tracker` field: it's no longer threaded to operators — either remove it (registration now lives in the session, Task 2) or keep it unused-and-`#[allow(dead_code)]` if the session reuses the planner's creation. Prefer removing it; the session builds + registers the bridge directly.
- [ ] **Step 3:** Run the FULL gate → green. The read-set is still populated (now via the store, Task 3) — confirm the Task 3 completeness tests still pass after removal (this proves store-level recording fully replaced operator-level). Clippy clean. Commit (`refactor(mvcc): remove operator-level read-recording (superseded by store-level)`).

---

## Task 5: Guard the non-MVCC-aware operators under Serializable

**Files:** `query/planner/lpg/mutation.rs` (`plan_shortest_path`), `query/planner/lpg/mod.rs` (`plan_vector_scan`, `plan_text_scan`, algos `CallProcedure`)

These operators bypass the visible-read API (raw adjacency/index reads), so they are neither snapshot-consistent nor recorded. Reject Serializable rather than be silently unsound. The signal: a Serializable tx has a registered read-tracker — but at *plan* time the cleanest signal is the planner's knowledge of the isolation level. **Confirm the planner can tell it's a Serializable tx at plan time** (it knows `transaction_id` + can query `transaction_manager.isolation_level(tid)`; in 2e it created a `read_tracker` iff Serializable — if that field is removed in Task 4, add a small `self.is_serializable()` helper that checks `isolation_level`). Use that signal.

- [ ] **Step 1:** Add `fn is_serializable(&self) -> bool` on the planner (checks `transaction_id` + `transaction_manager.isolation_level(...) == Some(Serializable)`), or reuse the existing read-tracker signal if kept.
- [ ] **Step 2:** Guard `plan_shortest_path` (top), `plan_vector_scan` (top), `plan_text_scan` (top), and the algos `CallProcedure` planner: `if self.is_serializable() { return Err(Error::Internal("Serializable isolation is not yet supported with <feature>; use SnapshotIsolation".into())); }`. Use a per-feature message (shortestPath/allShortestPaths; vector/text search; graph algorithms).
- [ ] **Step 3: Soundness sweep.** Enumerate EVERY arm of `plan_operator` (`mod.rs` ~`:728-840`); classify each: records-via-store (uses visible accessors), guarded (this task), unreachable Err, or no-store-read (joins/params/set-ops/mutations-on-already-read-entities). **List the full classification in the commit message / a comment.** If any arm reads the store outside the visible-read API and isn't guarded, guard it.
- [ ] **Step 4:** Tests (component-level, Serializable tx): `shortestPath`, a vector search, a text search, and an algos call each return the guard error; a normal `MATCH`/`MERGE` does not. Full gate green; clippy clean. Commit (`feat(mvcc): guard non-MVCC-aware operators under Serializable`).

---

## Task 6: Full verification + soundness audit

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green (incl. `serializable_readset`); `--features full -p grafeo-engine --test mvcc_isolation` unchanged.
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` + `grafeo-wasm` (wasm32) compile.
- [ ] **Soundness audit (the deliverable invariant):** confirm (a) every visible-read accessor in the Orientation table records; (b) every `plan_operator` arm is records-via-store / guarded / unreachable / no-store-read; (c) Serializable is still session-rejected (this increment does NOT enable it); (d) SI/RC allocate no tracker and record nothing (empty `read_trackers`). Document the arm classification.
- [ ] `git status` clean (only intended files; no `cargo fmt` churn). OPSEC: generic labels.

---

## Acceptance
- A Serializable transaction's read-set is populated **at the store**, capturing every snapshot-consistent read (scan/expand/MERGE/filter/aggregate) by construction; operator-level recording is gone; the completeness tests pass.
- Operators that bypass the visible-read API (`shortestPath`, vector/text, algos) are guarded under Serializable; the `plan_operator` arms are fully classified (records/guard/unreachable/no-read).
- Inert end-to-end (Serializable still session-rejected); SI/RC fast path unchanged; full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; OPSEC-clean.
- **The SSI read-set is now sound by construction** → next: **re-land the F1 enable** (remove the session rejection + the acceptance suite from parked `feat/mvcc-increment-2f`) — now sound, not audited. Then the integrate-or-guard follow-ups (MVCC-integrate shortestPath/vector/text/algos to remove their guards), then F2 (incremental SSI + read-registry) + G (granularity/perf).

## Risks
- **A missed chokepoint accessor = incomplete read-set.** The Orientation table IS the auditable surface — Task 6 verifies every entry records. If a new visible-read accessor is added later, it must record; document this as the convention (a `// records into read-tracker` contract on the visible-read API). This is the whole point: a bounded, auditable surface instead of unbounded operators.
- **An operator reading the store OUTSIDE the visible-read API** (raw `node_ids`/`edges_from`/index without a versioned/visible call) is an unrecorded read AND non-snapshot-consistent. Task 5's sweep must find them all; guard each. (These are the same pre-existing MVCC deficiencies — fixing them = MVCC-integrate, removing the guard.)
- **Unregister leaks.** Every tx-exit path (commit, serialization-abort, explicit rollback, error) must unregister; bound by clearing on store reset. A leaked tracker entry over-records a later tx reusing the id (ids are monotonic, so low risk) — still, unregister everywhere.
- **Lock contention on `read_trackers`.** The SI/RC fast path takes a read-lock on an empty map per accessor call. If it profiles hot, gate with an `AtomicBool`/`active_count`; YAGNI until measured.
- **Over-recording.** Store-level records every visible read incl. read-modify-write reads (entity also in write-set — harmless) and repeated reads (HashSet dedup). Sound; precision (granularity) is Part G.
- **Don't enable Serializable, don't build the registry, don't MVCC-integrate the guarded operators.** Those are the follow-ups. Keep this increment to "read-set complete by construction + guard the bypassers."

---

## STATUS: COMPLETE

Landed on `feat/mvcc-store-level-read-recording` (commits `014eb1cf` → `acd4d838`, from `integration` @ `c8a20726`), via subagent-driven-development + spec/holistic review.

**Final verification:** `--all-features -p grafeo-core -p grafeo-engine` = **7450 passed / 0 failed**; clippy `--all-features` clean; `default`/`temporal`/`tiered-storage` + `grafeo-wasm` compile (these don't build `layered.rs` without `compact-store`; `--all-features` does, and is green); inert (Serializable still session-rejected at `session/mod.rs:3936`).

**What landed:**
- **Store read-tracker registry** (`LpgStore.read_trackers` + register/unregister + `record_read_node/edge`; trait + cdc/wal/layered delegation).
- **Chokepoint instrumentation** — all 12 visible-read accessors on `LpgStore` record the visible entity (the surface table above + `edge_type_versioned`); engine registers the bridge at Serializable begin, unregisters at all 3 tx-exits.
- **Operator-level recording removed** (2e's per-operator `record_read` superseded; trait/bridge/manager kept).
- **Non-MVCC operators guarded** (`shortestPath`/vector/text/algos) via `is_serializable()`; full `plan_operator` arm classification (records-via-store / guarded / unreachable / no-store-read) as a doc-comment; catch-all `_ => Err` fails new raw-access ops closed.
- **LayeredStore base-resident reads recorded** (C1 fix) — the 10 overridden layered accessors record into the overlay's tracker; `filter_visible_node_ids_versioned` covered transitively.

**Key findings (review caught, fixed):**
- The opus Task-2 review enumerated the store read API and found **`edge_type_versioned` (LpgStore)** was the one un-instrumented tx-visible read → fixed. This is the payoff of "complete by construction": the surface is *enumerable*, so the gap was *findable* (vs. the unbounded per-operator audit).
- The opus holistic review found **C1: `LayeredStore` served base-resident reads from a tracker-less base layer** → fixed with 10 regression tests.
- `MERGE`/`MergeRelationship` are **auto-covered** (their matching uses the visible accessors) — the per-operator "MERGE gap" closed for free.

**Residual (documented, latent, non-blocking):**
- `edge_type_versioned` is a trait method `LayeredStore` does NOT override (uses the trait default → not recorded by construction on layered). Latent/caller-dependent: the edge reaches `type(r)`/expand-predicate via EXPAND, which records it through layered's now-instrumented `is_edge_visible_versioned`. Address by an explicit `LayeredStore::edge_type_versioned` override (or MVCC-integration) when the F1 enable hardens.
- Over-recording (`nodes_by_label_visible`/Merge candidate examination) and absence/phantom reads not recorded — sound, deferred to Part G granularity.

**Next: re-land the F1 enable** — the parked `feat/mvcc-increment-2f` (remove session rejection + `serializable.rs` acceptance suite), now **sound by construction**. Then integrate-or-guard follow-ups (MVCC-integrate shortestPath/vector/text/algos to remove their guards), then F2 (incremental SSI + read-registry) + G (granularity/perf).
