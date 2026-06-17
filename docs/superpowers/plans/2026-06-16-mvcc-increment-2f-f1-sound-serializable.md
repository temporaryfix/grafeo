# Increment 2f — F1: sound Serializable (OCC rung) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Serializable **actually work** (F1, the verified OCC rung): bring the existing commit-time read-write validation (`manager.rs:358-377`, now fed by increment 2e's read-set + write-set) live by removing the session-level Serializable rejection — **soundly**. The acceptance milestone: **write-skew is prevented under Serializable**, while the same scenario commits under SnapshotIsolation; benign concurrent writers and read-only Serializable transactions do **not** abort; rw-conflict aborts roll back cleanly with no leaked versions/epoch.

**Architecture:** Serializable already has all the machinery — it was only gated off at the session boundary and starved of a read-set (now fixed by 2e). F1 = (1) complete the last instrumented-producer read-recording gap (`VariableLengthExpandOperator` interior hops); (2) **guard** the still-uninstrumented read producers so Serializable only accepts queries whose reads it fully records — the planner sets `read_tracker = Some` iff the tx is Serializable (2e Task 3), so `self.read_tracker.is_some()` is a clean Serializable signal; in the vector/text scan planning sites, reject Serializable with a clear error (keeps Serializable **sound** — it never runs a query whose reads it can't track); (3) remove the session Serializable rejection so `begin_with_isolation(Serializable)` flows and the existing OCC validation goes live — `SerializationFailure` already routes through the generic commit-error rollback (`session/mod.rs:4031-4051`). **Deferred:** full vector/text MVCC+SSI integration (removes the guard); F2 (incremental SSI + the sharded read-registry); G (performance). This is the OCC backward-validation rung — F2 later replaces the commit-time scan with incremental detection.

**Tech Stack:** Rust, `cargo test`. `VariableLengthExpandOperator` (grafeo-core `execution/operators/variable_length_expand.rs`); the vector/text scan planners (grafeo-engine `query/planner/lpg/mod.rs`); the session begin path (`session/mod.rs`); a new acceptance suite (`tests/serializable.rs`). `CARGO_INCREMENTAL=0`.

---

## Scope decision (read first)

The spec §8 F1 says "just remove the begin-time rejection; the existing backward validation becomes sound serializable." That assumed Part D recorded **every** read site. 2d+2e completed the scan/expand producers, but two read paths remain unrecorded (per the 2e holistic review): **(a)** `VariableLengthExpandOperator` records only the final hop of each emitted path (interior hops for `min_depth > 1` are traversed but not recorded), and **(b)** vector/text scans (`scan_vector.rs`/`scan_text.rs`) are not MVCC-snapshot-aware at all (a pre-existing gap) and record nothing. A missed read = unsound SSI (a silently non-serializable result).

**This plan's call:** **fix (a)** (var-length is already instrumented, just incompletely — completing it is clean and var-length paths are common) and **guard (b)** (vector/text need a foundational MVCC integration that is out of F1 scope; rejecting Serializable for those queries keeps Serializable sound for everything it accepts). The full vector/text MVCC+SSI integration that removes the guard is a documented follow-up. *(If you'd prefer document-only or a vector/text-sweep-first ordering, this is the seam to change.)*

---

## Orientation

F1 of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§8). Builds on 2e (`integration` @ `5ee89c89`): the read-set (`record_read` at scan/expand) + write-set (store-derived) now feed the commit-time validation.

**What already exists (this is why F1 is small):**
- **The OCC validation is live-on-arrival.** `TransactionManager::commit` (`manager.rs:300`) runs, for Serializable, the read-write check (`:358-377`): for each tx that committed after our start, if it wrote any entity in **our read-set** → `SerializationFailure`. With 2e populating the read-set + write-set, this is *sound serializable* the moment Serializable is allowed. The write-write check (`:334-347`) already runs for all levels.
- **The abort path is generic.** In `session::commit`, `self.transaction_manager.commit(transaction_id)` (`session/mod.rs:4029`) returning `Err(_)` (including `SerializationFailure`) hits the rollback branch (`:4031-4051`): discard PENDING node/edge versions, replay the property undo log, clear pending deletes, `abort`. `SerializationFailure` (`manager.rs:386`; `error.rs:278` → `ErrorCode::TransactionSerialization`; `gqlstatus.rs:219` → `TX_ROLLBACK`) flows through it. So serialization-abort cleanup is already correct — confirm with a no-leak test.
- **Isolation plumbing is complete.** GQL `... ISOLATION LEVEL SERIALIZABLE` → `IsolationLevel::Serializable` (`session/mod.rs:1105-1106`); `begin_transaction(level)` → `begin_transaction_inner` (`:3916`) → **the rejection at `:3935-3944`** → otherwise `transaction_manager.begin_with_isolation(level)` (`:3945`). The manager already accepts Serializable.
- **The read-set is complete for scan/expand** (2e), incl. the factorized-aggregate path (2e Task 5 fix). `read_tracker.is_some()` on the planner ⟺ Serializable tx (2e Task 3 — the tracker is created only for Serializable).
- **Mirror test:** `manager.rs:1065 test_write_skew_prevented_by_ssi` is the manager-level write-skew test (uses `record_read`/`record_write` directly). F1's acceptance is the **session-level** end-to-end equivalent: real Serializable queries whose reads are recorded by 2e's wiring, asserting the abort.

**Verification gate:** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine`; `--features full -p grafeo-engine --test mvcc_isolation` (unchanged) + the new `serializable` suite. Clippy `--all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean. Hygiene: `rustfmt` changed files only; `git status` clean before commit; `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC: generic labels.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-core/src/execution/operators/variable_length_expand.rs` | var-length producer | Modify (record interior-hop edges/nodes, not just the final hop) |
| `crates/grafeo-engine/src/query/planner/lpg/mod.rs` | vector/text scan planning | Modify (reject Serializable for uninstrumented producers) |
| `crates/grafeo-engine/src/session/mod.rs` | tx begin | Modify (remove the Serializable rejection) |
| `crates/grafeo-engine/tests/serializable.rs` | acceptance | Create (write-skew, anomalies, benign-no-abort, clean rollback, guard) |

---

## Task 1: Complete `VariableLengthExpandOperator` interior-hop recording

**Files:** `execution/operators/variable_length_expand.rs`

The operator records only the final `(edge_id, target_id)` of each emitted path row (the 2e wiring, ~`:567-576`); interior hops (below `min_depth`, or intermediate steps of a longer path) are traversed via `get_edges` in the BFS but never recorded → a Serializable tx running a `*min..max` path with `min > 1` (or any multi-hop path) under-records its reads.

- [ ] **Step 1: Write the failing test** (operator-level, mirror the 2e expand test): a test-double `ReadTracker`; build a `VariableLengthExpandOperator` over `(a)-[r1]->(b)-[r2]->(c)` with a 2-hop pattern (`*2..2`) + a tx + `.with_read_tracker(double)`; run; assert it recorded **all** traversed edges (`r1`, `r2`) and intermediate node `b`, not just the final `(r2, c)`. Expected FAIL: interior `r1`/`b` missing.
- [ ] **Step 2: Record during traversal.** In the BFS where visible interior edges/neighbors are resolved (`process_input_row` / the `get_edges` step — find where each visible hop is taken, guarded by the existing snapshot visibility), record each visible interior edge id + neighbor node id via `(Some(tracker), Some(tid))` — mirroring the per-visible-hop pattern but at the traversal step, not only the emit step. Record only **post-visibility-filter** hops; dedupe is free (read-set is a `HashSet`), but record at the visible-hop resolution to cover interior steps.
- [ ] **Step 3: Run GREEN.** The probe passes (all hops recorded). `--all-features -p grafeo-core -p grafeo-engine` green; clippy clean. Commit (`fix(mvcc): record variable-length interior-hop reads`).

---

## Task 2: Enable Serializable soundly (remove rejection + guard uninstrumented reads)

**Files:** `session/mod.rs` (remove rejection), `query/planner/lpg/mod.rs` (guard)

Do these **together** so Serializable is never enabled without the guard (no unsound intermediate state).

- [ ] **Step 1: Add the guard** at the vector/text scan planning sites. Find `plan_text_scan` (`mod.rs:919`) and the vector-scan planner (the `VectorScan`/`scan_vector` planning site near `mod.rs:880`). At the top of each, reject Serializable:
```rust
// Serializable requires recording every read; vector/text scans are not yet
// MVCC-snapshot-aware (no read tracking), so a Serializable tx using them could
// miss conflicts. Reject rather than be silently unsound. (read_tracker is Some
// iff the tx is Serializable — see planner read-tracker creation.)
if self.read_tracker.is_some() {
    return Err(Error::Internal(
        "Serializable isolation is not yet supported with vector/text search; \
         use SnapshotIsolation for these queries".to_string(),
    ));
}
```
(Confirm the exact error variant/type used elsewhere in these planners; match it. Confirm both the text and vector scan planning entry points are covered — grep `plan_text_scan`/`plan_vector_scan`/`VectorScan`.)
- [ ] **Step 2: Remove the session rejection.** Delete the `if level == IsolationLevel::Serializable { return Err(... "not yet supported" ...) }` block at `session/mod.rs:3935-3944`, so the `else` falls through to `self.transaction_manager.begin_with_isolation(level)` for Serializable too. (Keep the surrounding structure; just drop the rejection branch.)
- [ ] **Step 3: Smoke test.** A Serializable session tx can now `begin`, run a simple `MATCH ... RETURN`, and `commit` (no error). A Serializable query using vector/text search returns the guard error. Write these two probes; run → both behave as asserted. `--all-features` green; clippy clean. Commit (`feat(mvcc): enable sound Serializable (F1 OCC rung) with uninstrumented-read guard`).

---

## Task 3: Serializable acceptance suite

**Files:** Create `crates/grafeo-engine/tests/serializable.rs`

Session-level, end-to-end. Two concurrent `db.session()`s share the transaction manager, so cross-session SSI validation applies. Confirm the GQL dialect for `BEGIN ... ISOLATION LEVEL SERIALIZABLE` (or the session API `begin_transaction(IsolationLevel::Serializable)`) against existing tests; use whichever the engine supports.

- [ ] **Step 1: Write-skew prevented (the milestone).** Two Serializable txns, classic write-skew: both read rows A and B; T1 writes A, T2 writes B; both commit. Assert the **second committer aborts** with a serialization failure. Then the **same scenario under SnapshotIsolation commits both** (levels genuinely differ).
```rust
#[test]
fn write_skew_prevented_under_serializable_allowed_under_si() {
    // setup: two nodes A,B with a balance/flag; classic write-skew (each reads both, writes one)
    // Serializable: second commit -> Err(serialization failure). SI: both Ok.
    // (Adjust query shape to the supported dialect; assert via commit() result.)
}
```
- [ ] **Step 2: Benign concurrent writers do NOT abort.** Two Serializable txns that write **disjoint** entities (and read only what they write) both commit — the SSI-over-OCC win; Serializable must not abort non-conflicting writers.
- [ ] **Step 3: Read-only Serializable does not abort.** A read-only Serializable tx concurrent with a writer commits without abort.
- [ ] **Step 4: Clean abort / no leaks.** After a serialization abort, assert: the aborted tx's writes are not visible; a fresh tx sees the pre-abort state; the version chain isn't leaking PENDING versions and `min_active_epoch` isn't pinned (mirror how existing conflict-rollback tests assert no-leak — e.g. a subsequent successful commit + read, and MVCC GC not stalled). Reuse the existing write-conflict rollback assertions as a template.
- [ ] **Step 5: Run.** All acceptance tests green; `--all-features` + `--features full` green; clippy clean. Commit (`test(mvcc): Serializable acceptance suite (write-skew, benign, read-only, no-leak)`).

---

## Task 4: Full verification + OPSEC + soundness audit

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green (incl. `serializable` + the var-length probe). `--features full -p grafeo-engine --test mvcc_isolation` unchanged.
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` (wasm32) compile.
- [ ] **Soundness audit:** every read producer reachable by a Serializable query either records reads (scan/expand/var-length) or is guarded (vector/text). Confirm no third uninstrumented producer is reachable under Serializable (grep the planner's `LogicalOperator` dispatch for scan/seek/scan-like ops; verify each is covered or guarded). Document the guarded gap in the plan's STATUS.
- [ ] `git status` clean. OPSEC: generic labels.

---

## Acceptance
- A Serializable transaction can begin/run/commit; write-skew is prevented under Serializable and allowed under SI; benign concurrent writers and read-only Serializable transactions commit without aborting; serialization aborts roll back cleanly with no leaked versions/pinned epoch.
- Serializable is **sound for what it accepts**: every reachable read producer records reads or is guarded (vector/text rejected). Variable-length interior reads are now recorded.
- Full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; OPSEC-clean.
- **Serializable now works (the F1 milestone).** Next: the **vector/text MVCC+SSI sweep** (remove the guard → full soundness), then **F2** (incremental SSI: per-tx in/out rw-flags + dangerous-structure pivot abort over the sharded read-registry, replacing the commit-time backward scan) + **G** (performance: sharded registry, conflict-granularity knob, no global commit scan).

## Risks
- **A missed reachable read producer = unsound Serializable.** The Task 4 audit is load-bearing: enumerate every `LogicalOperator` that reads entities and confirm record-or-guard. If a producer is found that neither records nor is guarded, guard it (conservative) and note it. Over-reject is safe; under-record is not.
- **Guard placement.** The guard must fire for *every* path that builds a vector/text scan under Serializable (confirm there's a single planning entry per scan type; if vector search is also reachable via a hybrid/index operator, guard there too).
- **Var-length interior recording cost.** Recording every interior hop is O(traversed) tracker calls; the read-set is a `HashSet` (dedup) and this is Serializable-only, so acceptable. Don't record pre-visibility-filter hops.
- **Existing SI/RC behavior must not change.** Removing the rejection only affects Serializable; the guard only fires when `read_tracker.is_some()` (Serializable). Confirm no SI/RC test changes.
- **Test dialect.** Confirm the supported way to start a Serializable tx (GQL clause vs session API) before writing the suite; mirror existing isolation tests.

---

## STATUS: PARKED — enable held pending store-level read-recording

Implemented on branch `feat/mvcc-increment-2f` (commits `1cad020c` var-length recording, `a0ae8502` enable+guard, `ac3a7909` acceptance suite, `613cd167` gitignore). The milestone works — `write_skew_prevented_under_serializable` passes; the acceptance suite is genuine. **But this branch is NOT merged**, because executing it surfaced a deeper architectural finding:

**Per-operator read-recording (increment 2e's Decision A) is fragile — a denylist that can never be confidently complete.** The final holistic review found `ShortestPath` is a reachable, unrecorded, un-guarded entity-reading producer under Serializable (verified with a repro: it silently fails to abort a write-skew the recorded traversal correctly aborts). A follow-on audit found the same for `MERGE`/`MergeRelationship` (unrecorded match-reads) and `CallProcedure` (algos). Guarding each as-found is whack-a-mole; "Serializable that silently misses conflicts" is the worst outcome for a soundness feature.

**Decision (with the user):** re-architect read-recording to the **store's visible-read API chokepoint** — complete by construction (mirroring the write-set's Decision B), reusing 2e's `ReadTracker`/bridge/manager/planner infrastructure but relocating the *call site* from operators to the store accessors. The non-MVCC-aware operators (vector/text/shortestPath/algos) become a distinct, finite **integrate-or-guard** class. *Then* enable Serializable — sound by construction, not by audit. See the store-level read-recording plan.

**What re-lands on the store-level foundation:** the enable (remove session rejection), the non-MVCC guards, and this acceptance suite (its proof). **What's superseded:** the operator-level `record_read` in 2e + this branch's var-length operator recording (`1cad020c`) — the store accessors will record instead. The `docs/research/` gitignore protection (`613cd167`) is independent and is carried to `integration` directly.
