# F2 — Incremental SSI (read-registry + dangerous-structure) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace F1's commit-time OCC backward-validation (a global scan that aborts *any* Serializable committer whose read-set was touched by a concurrent committer — over-aggressive: read-only txns abort, benign same-item readers abort) with **incremental Serializable Snapshot Isolation** (Cahill et al.): track read-write antidependencies as they form via a shared **read-registry**, flag each transaction with `in_conflict`/`out_conflict`, and abort **only a pivot in a potential dependency cycle** (a tx with both an inbound and an outbound rw-edge). Result: read-only Serializable transactions don't abort, benign concurrent writers don't abort, and commit becomes O(1) flag checks instead of a global scan — while write-skew and the standard SSI anomalies stay prevented.

**Why now:** F1 (sound serializable) is merged and works, but its OCC rung is conservative (the acceptance suite documents read-only-aborts + same-label false-aborts). F2 is the spec's "target" (§8 F2) that makes Serializable *practically* usable. The read-set (store-level recording) + write-set (store-derived) it needs are already complete and sound by construction.

**Architecture (Cahill SSI):**
- **Read-registry** — a sharded `EntityId → set<reader TransactionId>` of *active Serializable readers* (the SIREAD locks). Fed when a Serializable tx records a read; entries GC'd when the reader commits/aborts (bounded by the active-tx set). Sharded by entity hash so concurrent readers/writers rarely contend.
- **Two flags per active Serializable tx** on `TransactionInfo`: `in_conflict` (some `T' →rw self` — an rw-edge *into* it) and `out_conflict` (`self →rw T'` — an rw-edge *out* of it). An rw-antidependency `T_reader →rw T_writer` (reader read an item the writer overwrites with a newer version) sets `T_reader.out_conflict = true` **and** `T_writer.in_conflict = true`.
- **Detect rw-edges incrementally, at both ends (the symmetry matters):**
  - **Write-time:** when `T_writer` writes entity `E`, consult the registry for *concurrent* Serializable readers of `E`; for each `T_reader`, set the edge `T_reader →rw T_writer`.
  - **Read-time:** when `T_reader` reads `E`, if a *concurrent* transaction `T_writer` has already written a newer version of `E` (visible via the version chain / `T_writer ∈ committed-after-our-start` or active with `E` in its write-set), set the edge `T_reader →rw T_writer`.
- **Dangerous-structure abort:** a tx with **both** `in_conflict` and `out_conflict` is a pivot that can anchor a non-serializable cycle. Abort it. Checked at commit (O(1) flag read) — and the standard refinement (abort the pivot whose out-neighbor has already committed, to guarantee progress) is applied. Aborts surface as the existing `SerializationFailure` with the existing clean rollback.

**Tech Stack:** Rust, `cargo test`. All in `grafeo-engine` `transaction/` (`manager.rs` + a new `read_registry.rs`); the read-registry is fed via `TransactionManager::record_read` (already called by the store-level read-tracker bridge) and consulted via `record_write`. `CARGO_INCREMENTAL=0`.

---

## Scope decision (read first)

F2 changes **only the conflict-detection logic in the transaction manager** — it does NOT touch the read-recording (store-level, complete) or the guards (non-MVCC ops stay guarded). It is gated to `IsolationLevel::Serializable` (SI/RC unchanged). The **read-registry** (deferred from increment 2e as premature) is built here, now that F2 (its only consumer) exists. **Granularity stays entity-level** (the spec's sound default); the property-level knob to further cut false antidependencies is **Part G** (a follow-up), not F2.

**Behavior change vs F1 (the wins, each an explicit test):**
- read-only Serializable tx concurrent with a writer of what it read → **commits** (F1: aborted). The `read_only_serializable_does_not_abort` acceptance test (currently accepts both) is tightened to assert **Ok**.
- benign concurrent writers (disjoint, or non-pivot) → **commit** (F1: same-item readers could abort).
- write-skew (a true pivot) → still **aborts**; the standard anomalies (read-only anomaly, batch-processing) → prevented.

---

## Orientation

F2 of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§8 F2, §9 G, §10-C). On `integration` (F1 merged): Serializable is enabled and sound; the read-set (store-level) + write-set feed the **current** validation.

**Current validation to REPLACE** (`transaction/manager.rs`):
- `commit` (`:320`) holds `transactions.write()` + `committed_epochs.write()`, then does:
  - write-write check (`:354-367`): for each `other` committed after our start that wrote an entity in **our write-set** → `WriteConflict`. **KEEP this** (first-committer-wins; orthogonal to SSI).
  - **SSI read-write backward scan** (`:378-394`): for each `other` committed after our start that wrote an entity in **our read-set** → `SerializationFailure`. **REPLACE** with the F2 pivot check.
- `record_read` (`:281`) inserts into `read_set`. F2: also register in the read-registry + run read-time detection.
- `record_write` (`:180`) first-writer-wins + inserts into `write_set`. F2: also run write-time detection (consult the registry).
- `record_entity`/`extend_write_set` (`:233`/`:264`) populate write-set without conflict logic. The store-derived write-set completion (`extend_write_set` at session commit) happens *before* `commit` — F2's write-time detection should also see those entities (run detection over the completed write-set at commit-time as a safety net, see Task 5).
- `TransactionInfo` (`:89`): `state`, `isolation_level`, `start_epoch`, `write_set`, `read_set`. F2 adds `in_conflict: bool`, `out_conflict: bool`.
- `committed_epochs` (`:127`): `Tx → commit EpochId`. `active_count` (`:122`). No `min_active_epoch` getter on the manager — GC the registry on reader commit/abort (Task 2) rather than by epoch horizon.

**Concurrency model (verified):** `begin` never serializes writers; multiple Serializable txns are concurrent on the shared manager. The registry must be safe under concurrent record_read (readers) + record_write (writers). Shard by `EntityId` hash; each shard a `RwLock<FxHashMap<EntityId, SmallVec<[TransactionId; _]>>>` (or `FxHashSet`). The flags live on `TransactionInfo` (under `transactions.write()`).

**"Concurrent" definition:** `T_a` and `T_b` are concurrent iff neither committed before the other started — i.e. overlapping `[start, commit]`. In practice: a reader/writer is concurrent with an *active* tx, or with a tx that committed after our start (`committed_epoch > our_start_epoch`). The existing checks already use `commit_epoch.as_u64() > our_start_epoch.as_u64()` for the committed case; reuse it.

**Verification gate:** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full -p grafeo-engine --test serializable`. Clippy `--all-features … -D warnings` clean; profiles + wasm compile. Hygiene: `rustfmt --edition 2024` changed files only (NOT `cargo fmt`); `git status` clean; `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC: generic labels.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/src/transaction/read_registry.rs` | sharded SIREAD-lock registry | Create (`ReadRegistry`: shards, `record_reader`/`readers_of`/`remove_reader`) |
| `crates/grafeo-engine/src/transaction/manager.rs` | flags + detection + pivot abort | Modify (`TransactionInfo` flags; registry field; `record_read`/`record_write` detection; `commit` pivot check replaces the backward scan; GC on commit/abort) |
| `crates/grafeo-engine/src/transaction/mod.rs` | module | Modify (`mod read_registry;`) |
| `crates/grafeo-engine/tests/serializable.rs` | acceptance | Modify (tighten read-only→Ok; add benign-readers→Ok; add SSI anomalies; keep write-skew→abort) |

---

## Task 1: rw-conflict flags on `TransactionInfo`

**Files:** `transaction/manager.rs`

- [ ] **Step 1: Write the failing test** (manager-level): begin two Serializable txns; assert both have `in_conflict == false && out_conflict == false` initially (via a `#[cfg(test)]` getter or direct field read in the manager's test module). Expected FAIL: no fields.
- [ ] **Step 2: Add fields** `in_conflict: bool` + `out_conflict: bool` to `TransactionInfo` (`:89`), both `false` in `new` (`:104`). Add a private helper `fn set_rw_edge(&self, reader: TransactionId, writer: TransactionId)` on `TransactionManager` that, under `transactions.write()`, sets `reader.out_conflict = true` and `writer.in_conflict = true` (guard: both must be Active + Serializable; ignore if either is gone). Add a `#[cfg(test)] fn conflict_flags(&self, tx) -> (bool, bool)` getter.
- [ ] **Step 3:** Run → PASS. Commit (`feat(mvcc): rw-conflict flags on TransactionInfo`).

---

## Task 2: the sharded read-registry

**Files:** Create `transaction/read_registry.rs`; `transaction/mod.rs`

- [ ] **Step 1: Write the failing test** (in `read_registry.rs`): a `ReadRegistry`; `record_reader(E, t1)`, `record_reader(E, t2)`; `readers_of(E)` returns `{t1, t2}`; `remove_reader(t1)` (all entities); `readers_of(E)` returns `{t2}`. Expected FAIL: no `ReadRegistry`.
- [ ] **Step 2: Implement** `ReadRegistry`:
```rust
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use super::EntityId;
use grafeo_common::types::TransactionId;

const SHARDS: usize = 64; // power of two; entity-hash sharded

pub struct ReadRegistry {
    shards: Vec<RwLock<FxHashMap<EntityId, FxHashSet<TransactionId>>>>,
    // reverse index for O(reads) GC of a finished reader:
    by_tx: RwLock<FxHashMap<TransactionId, Vec<EntityId>>>,
}
impl ReadRegistry {
    pub fn new() -> Self { /* SHARDS empty maps */ }
    fn shard(&self, e: &EntityId) -> &RwLock<FxHashMap<EntityId, FxHashSet<TransactionId>>> { /* hash % SHARDS */ }
    /// Register that `tx` (a Serializable reader) read `entity`.
    pub fn record_reader(&self, entity: EntityId, tx: TransactionId) {
        self.shard(&entity).write().entry(entity).or_default().insert(tx);
        self.by_tx.write().entry(tx).or_default().push(entity);
    }
    /// Active Serializable readers of `entity` (for write-time detection).
    pub fn readers_of(&self, entity: EntityId) -> Vec<TransactionId> {
        self.shard(&entity).read().get(&entity).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }
    /// GC all entries for a finished (committed/aborted) reader.
    pub fn remove_reader(&self, tx: TransactionId) {
        if let Some(entities) = self.by_tx.write().remove(&tx) {
            for e in entities {
                let mut sh = self.shard(&e).write();
                if let Some(set) = sh.get_mut(&e) { set.remove(&tx); if set.is_empty() { sh.remove(&e); } }
            }
        }
    }
}
```
(Confirm `EntityId: Hash + Eq` — it derives `Hash`/`Eq` at `manager.rs:67`. Use a stable hash for sharding, e.g. `FxHasher`.) Wire `mod read_registry;` in `transaction/mod.rs`.
- [ ] **Step 3:** Run → PASS. Commit (`feat(mvcc): sharded read-registry (SIREAD locks)`).

---

## Task 3: feed the registry + read-time detection in `record_read`

**Files:** `transaction/manager.rs`

- [ ] **Step 1: Write the failing test:** t_w (Serializable) writes E and commits; t_r (Serializable, started before t_w committed) then reads E → assert `t_r.out_conflict == true` and (t_w being committed) the edge recorded. Also: t_r reads E that no one wrote → no flags. Expected FAIL.
- [ ] **Step 2:** Add a `read_registry: ReadRegistry` field to `TransactionManager`. In `record_read` (`:281`), after inserting into `read_set`, **only for Serializable** `tx`:
  - `self.read_registry.record_reader(entity, tx);`
  - **read-time detection:** find any tx `T_w` that wrote `entity` and is concurrent (active with `entity` in its write-set, OR committed with `commit_epoch > our_start_epoch`); for each, `self.set_rw_edge(tx, T_w)`. (Scan `transactions` for active writers of `entity`; scan `committed_epochs` for committers-after-our-start whose write-set contains `entity`. Reuse the commit-path concurrency test.)
- [ ] **Step 3:** Run → PASS. Full gate green; clippy clean. Commit (`feat(mvcc): read-registry feed + read-time rw-edge detection`).

---

## Task 4: write-time detection in `record_write`

**Files:** `transaction/manager.rs`

- [ ] **Step 1: Write the failing test:** t_r (Serializable) reads E; t_w (Serializable, concurrent) writes E → assert `t_r.out_conflict == true && t_w.in_conflict == true`. Expected FAIL.
- [ ] **Step 2:** In `record_write` (`:180`) — and also for the store-derived completion path — after the first-writer-wins check, **only for Serializable** `tx`: `for t_r in self.read_registry.readers_of(entity) { if t_r != tx { self.set_rw_edge(t_r, tx); } }`. (A concurrent reader of E now has `t_r →rw t_w`.) Note: `record_write` holds `transactions.write()`; `set_rw_edge` also takes it — refactor so the lock is held once (compute the edges, then apply) to avoid re-entrant locking. Mirror for the `extend_write_set`/commit-time completion so store-derived writes also detect (Task 5 ties this off).
- [ ] **Step 3:** Run → PASS. Full gate green; clippy clean. Commit (`feat(mvcc): write-time rw-edge detection via read-registry`).

---

## Task 5: dangerous-structure pivot abort at commit (replace the backward scan)

**Files:** `transaction/manager.rs`

- [ ] **Step 1: Write the failing tests** (the F2 wins + the preserved guarantee):
  - `write_skew_pivot_aborts`: the classic write-skew (each reads both, writes one) → the pivot (both flags) aborts. (Move/adapt from the acceptance suite.)
  - `read_only_does_not_abort`: read-only Serializable + concurrent writer of what it read → the reader has `out_conflict` only (no `in_conflict`, it wrote nothing) → **commits Ok**.
  - `benign_non_pivot_commits`: a writer with only `in_conflict` (a reader read what it wrote) but no `out_conflict` → commits Ok.
- [ ] **Step 2:** In `commit` (`:320`), **before** advancing the epoch, replace the SSI backward scan (`:378-394`) with: gather the store-derived write-set completion's edges first (run write-time detection over `our_write_set` against the registry — the safety net so store-direct writes that bypassed `record_write` are covered), then **pivot check**: if `our_isolation == Serializable && our_in_conflict && our_out_conflict` → `SerializationFailure` (with the existing clean rollback path). Keep the write-write check (`:354-367`) unchanged. Apply the progress refinement: only abort if an out-neighbor has committed (a fully-concurrent pivot where no out-neighbor has committed yet need not abort *this* one — but the simplest sound rule is "pivot ⇒ abort"; implement "pivot ⇒ abort" first, verified, then refine to the commit-order rule if a benign case false-aborts — document the choice).
- [ ] **Step 3: GC on finish.** In `commit` (success) and `abort`, call `self.read_registry.remove_reader(tx)` (the tx is no longer an active reader). Also clear its flags implicitly (the tx leaves Active). Confirm no registry entry outlives the active-tx set.
- [ ] **Step 4:** Run → the F2 win tests pass AND `write_skew_prevented_under_serializable` (acceptance suite) still passes. Full gate green; clippy clean. Commit (`feat(mvcc): dangerous-structure pivot abort (incremental SSI commit)`).

---

## Task 6: tighten the acceptance suite + SSI anomalies

**Files:** `crates/grafeo-engine/tests/serializable.rs`

- [ ] **Step 1:** Tighten `read_only_serializable_does_not_abort` (currently accepts Ok or Err) to assert **Ok** (F2 must not abort it) — and rename it to drop the "conservative" framing. Update `benign_concurrent_writers_do_not_abort` to also cover the SAME-label case that F1 false-aborted (entity-granular still over-records the label scan, so this may STILL abort under F2's entity granularity — if so, keep the disjoint-label form and note that same-label is the Part-G property-granularity win, NOT F2; confirm empirically and document which it is).
- [ ] **Step 2: Add the standard SSI anomalies** as end-to-end tests: the **read-only anomaly** (a read-only tx exposes a non-serializable schedule among two writers — SSI must abort one) and the **batch-processing anomaly**, per the Cahill paper / PostgreSQL SSI test cases. Assert the serialization failure occurs (and the same scenarios commit under SI).
- [ ] **Step 3:** Run → all pass. Full gate green. Commit (`test(mvcc): F2 acceptance — read-only commits, anomalies prevented`).

---

## Task 7: full verification + soundness/perf audit

- [ ] `--all-features -p grafeo-core -p grafeo-engine` green; `--features full --test serializable` green (write-skew aborts; read-only commits; anomalies prevented; benign commit).
- [ ] clippy `--all-features … -D warnings` clean; `default`/`lpg`/`temporal`/`tiered-storage` + `grafeo-wasm` compile.
- [ ] **Soundness audit:** (a) every rw-edge sets BOTH endpoints' flags (reader.out, writer.in); (b) the pivot rule aborts a true write-skew and the anomalies; (c) read-only + benign-non-pivot do NOT abort; (d) the registry is fed for every Serializable read (it rides on the store-level `record_read` → manager `record_read`) and GC'd on every tx-exit (no leak); (e) SI/RC never touch the registry/flags. (f) Commit no longer does the O(committed) backward scan (O(1) flag check + the write-set completion detection).
- [ ] `git status` clean; OPSEC.

---

## Acceptance
- Write-skew and the standard SSI anomalies (read-only anomaly, batch-processing) are prevented under Serializable; the same scenarios commit under SI.
- **Read-only Serializable transactions and benign (non-pivot) concurrent writers do NOT abort** — the SSI win over OCC; explicit tests.
- rw-antidependencies tracked incrementally via the sharded read-registry; commit is an O(1) pivot-flag check (no global backward scan); the registry is GC'd on every tx-exit.
- Full `--all-features` + `--features full` green; clippy clean; profiles + wasm; OPSEC-clean.
- **Serializable is now SSI, not OCC** → next: **G** (property-level conflict granularity to cut the same-label false aborts + benchmarks: concurrent disjoint-write throughput, abort-rate under contention), and the integrate-or-guard follow-ups (MVCC-integrate shortestPath/vector/text/algos to remove their Serializable guards).

## Risks
- **SSI correctness is subtle** (the spec's headline risk). The flag directions (`reader.out`, `writer.in`), the read/write-time symmetry (both must fire or an edge is missed), and the pivot/abort rule are easy to get subtly wrong. Mitigation: TDD each edge direction in isolation (Tasks 3/4 test the flags directly), then the end-to-end anomalies (Task 6) against known SSI test cases; the existing `write_skew_prevented` acceptance test is the regression anchor — it MUST keep passing through every task.
- **Missing an rw-edge = unsound (a missed abort).** Read-time AND write-time detection must both be wired (an edge can form from either side depending on interleaving). The store-derived write-set completion means some writes bypass `record_write` — the commit-time safety net (Task 5 Step 2) covers them; verify a store-direct-write write-skew still aborts.
- **Over-abort (false positive) regression.** The whole point is fewer aborts; if F2 aborts a read-only or benign tx, the flag logic is wrong (likely setting `in_conflict` where it shouldn't). The read-only/benign tests are the guard.
- **Registry leak / unbounded growth.** GC on every tx-exit (commit + abort + the conflict-rollback path); the `by_tx` reverse index makes GC O(reads). Verify no entry outlives its reader.
- **Lock re-entrancy.** `record_write`/`record_read` hold `transactions.write()`; `set_rw_edge` needs it too — compute edges, then apply under one lock acquisition (Task 4 note). Don't deadlock.
- **Entity vs property granularity.** F2 stays entity-level; same-label false aborts persist (they're the Part-G win). Don't conflate — document which acceptance behaviors are F2 vs G.

---

## STATUS: NOT STARTED

Plan written against `integration` (F1 merged — Serializable enabled + sound). Scope: **incremental SSI** (read-registry + rw-edge flags + dangerous-structure pivot abort) replacing F1's commit-time OCC backward scan; entity-granularity (property-granularity is Part G). Execute via subagent-driven-development + final holistic review. The Cahill SSI algorithm is subtle — recommend executing with fresh context and reviewing the flag-direction/detection design first. On completion: **Part G** (property granularity + benchmarks) and the integrate-or-guard follow-ups.
