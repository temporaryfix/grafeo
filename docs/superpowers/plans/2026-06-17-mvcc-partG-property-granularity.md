# Part G — Property-Level Conflict Granularity (knob) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an **opt-in property-level conflict-granularity policy** to the incremental-SSI engine so that read-write antidependencies are tracked per `(entity, property)` instead of per `entity` — eliminating the dominant F2 false-abort class (a label scan reads the `id` column while a concurrent transaction writes the `balance` column of the same node; entity-granularity sees a conflict, property-granularity does not). Entity-level stays the **sound default**; property-level is a per-session/per-database policy that trades higher tracking cost for fewer false aborts (the spec's "performance knob", §9).

**Why now:** F2 (incremental SSI) is sound and merged; its one practical residual is entity-granular over-abort on wide-node / shared-label workloads. The spec makes this knob **benchmark-driven** — so this plan's acceptance includes a **deterministic demonstration**: the exact disjoint-property workload that entity-level aborts and property-level commits (stronger, non-flaky evidence than a throughput micro-benchmark).

**Architecture:**
- **`PropTag = Option<u64>`** attached to every conflict key. `None` = a structural/whole-entity read or write (existence, labels, delete, or any read under entity-level policy). `Some(h)` = a specific property (a stable 64-bit hash of the `PropertyKey`; hash collisions only ever *merge* keys → a safe over-approximation, never a missed conflict). The read-set, the read-registry, and `retired_readers` carry `PropTag`; the **rw-antidependency overlap** becomes `same entity && prop_compatible(a, b)` where `prop_compatible(a,b) = a.is_none() || b.is_none() || a == b`.
- **Granularity is policy-gated.** A `ConflictGranularity { Entity, Property }` (default `Entity`) on the database/session. Under `Entity`, recording always emits `PropTag::None` → the overlap is `same entity` → **byte-for-byte the current behavior** (this is the safety net: the default path is provably unchanged). Under `Property`, property accessors emit `Some(hash)`.
- **Only the SSI rw-detection goes property-level.** The first-committer-wins **W-W check stays entity-level** (two writes to the same node can lose-update under whole-node versioning regardless of which property — keeping it entity-granular is the sound choice; it is also already non-conflicting for the disjoint-*entity* scan case).
- **The read-recording trait gains a property-aware path** but stays *complete by construction*: every visible read still records (the property is additional information, never a gate). Structural accessors record `None`; property accessors record the property.

**Tech Stack:** Rust, `cargo test`. Core: `grafeo-core` `ReadTracker` trait + the store's record-read chokepoints. Engine: `grafeo-engine` `transaction/manager.rs` + `read_registry.rs` + the `TransactionReadTracker` bridge + the session/db policy. `CARGO_INCREMENTAL=0`.

---

## Scope decision (read first)

This is an **additive, opt-in** change. The entity-level default path must remain behavior-identical (the F2 acceptance suite + the 7483-test gate stay green unchanged). Property-level is exercised only when a session/db opts in. Staging is deliberately **behavior-preserving first, enable last**:
- Tasks 1-2 generalize the *internal* key representation to carry `PropTag`, with every producer emitting `None` → no behavior change.
- Task 3 adds the trait/recording property path, still emitting `None` under the default policy → no behavior change.
- Task 4 adds the policy + flips property accessors to emit the real property under `Property` policy → the new behavior, behind the opt-in.

This ordering means a regression in Tasks 1-3 shows up immediately as a *changed* default-path result, isolating risk away from the sound foundation. **Do not** make the W-W check property-level. **Do not** change `EntityId`'s own definition — `PropTag` rides *alongside* it.

**Out of scope:** scan/predicate-granularity (a label scan still visits and records every matched node — property-granularity narrows *which property* of each, not *which nodes*; narrowing the node set is a separate, harder predicate-pushdown concern). Throughput benchmarks (separate, optional follow-up). Interned property-id reuse (we hash the `PropertyKey`; switching to the store's interned id later is a perf refinement).

---

## Orientation

Builds on `integration` @ `eecf0359` (F2 merged). Key current state:

- **`ReadTracker` trait** (`grafeo-core`): `record_node_read(&self, tx: TransactionId, id: NodeId)` + `record_edge_read(&self, tx, id: EdgeId)` — **no property argument today**. Implemented by `TransactionReadTracker` (engine bridge → `manager.record_read`) + the layered store's overlay + test doubles (`store/tests.rs`, `compact/layered.rs:4585`).
- **Store record-read chokepoints** (`grafeo-core/src/graph/lpg/store/`): `record_read_node(tx, id)` (`mod.rs:1118`), `record_read_edge` (`mod.rs:1127`) call the registered tracker. Called from the 12 visible-read accessors across `node_ops.rs`/`edge_ops.rs`/`property_ops.rs`/`schema.rs`. **Property accessors** (`read_node_property_visible` `property_ops.rs:1073` takes `key: &PropertyKey`; `read_edge_property_visible` `:1108`) have the property in scope. **Structural accessors** (existence, labels, get/scan) do not.
- **Manager** (`grafeo-engine/src/transaction/manager.rs`): `read_set: HashSet<EntityId>`, the registry `read_registry: ReadRegistry` keyed by `EntityId`, `retired_readers`, `record_read(tx, entity: impl Into<EntityId>)` (`:281`-ish, called by the bridge), the F2 read-time/write-time detection + `set_rw_edge` + the pivot abort. `write_set: HashSet<EntityId>` (W-W check — stays entity-level).
- **`PropertyKey`**: a string-like key (the accessor takes `&PropertyKey`). We hash it to `u64` for the tag.

**Verification gate (every task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full -p grafeo-engine --test serializable`. Clippy `--all-features … -D warnings` clean. Profiles (`default`/`lpg`/`lpg,temporal` engine; `grafeo-core` `lpg,tiered-storage`) + `grafeo-wasm` compile. Hygiene: `rustfmt --edition 2024` changed files only (NOT `cargo fmt`); `git status` clean (a pre-existing untracked `ce/` exists — never `git add` it); `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC: generic labels in tests.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/src/transaction/conflict_key.rs` | `PropTag`, `prop_compatible`, prop hashing | Create |
| `crates/grafeo-engine/src/transaction/read_registry.rs` | registry stores `(TxId, PropTag)` per entity; compatibility-filtered `readers_of` | Modify |
| `crates/grafeo-engine/src/transaction/manager.rs` | read-set carries `PropTag`; rw-detection uses `prop_compatible`; W-W unchanged; policy field | Modify |
| `crates/grafeo-core/src/graph/lpg/store/read_tracker`/trait + accessors | `ReadTracker` gains a property-aware record path; property accessors pass the prop, structural pass `None` | Modify (`grafeo-core` trait + `mod.rs` chokepoints + the property accessors) |
| `crates/grafeo-engine/src/transaction/read_tracker.rs` (bridge) | forward the prop to `manager.record_read` | Modify |
| engine session/db | `ConflictGranularity` policy (default `Entity`), plumbed to the bridge | Modify |
| `crates/grafeo-engine/tests/serializable.rs` | the deterministic disjoint-property demonstration (entity aborts, property commits) | Modify |

---

## Task 1: `PropTag` + compatibility + prop hashing

**Files:** Create `crates/grafeo-engine/src/transaction/conflict_key.rs`; wire `mod conflict_key;`.

- [ ] **Step 1 (test):** `prop_compatible(None, None)`, `(None, Some(1))`, `(Some(1), None)` are all `true`; `(Some(1), Some(1))` true; `(Some(1), Some(2))` false. `prop_tag("balance") == prop_tag("balance")` and `!= prop_tag("id")` (stable, non-trivial hash). FAIL (no module).
- [ ] **Step 2:** Implement:
```rust
//! Conflict-key granularity: an optional property tag riding alongside EntityId.
pub type PropTag = Option<u64>; // None = structural/whole-entity; Some(hash) = a specific property

/// Two rw-conflict participants on the SAME entity actually conflict iff their
/// property tags are compatible: a structural (None) read/write touches the whole
/// entity, so it conflicts with anything; two property tags conflict iff equal.
#[inline]
pub fn prop_compatible(a: PropTag, b: PropTag) -> bool {
    a.is_none() || b.is_none() || a == b
}

/// Stable 64-bit tag for a property key. Collisions only MERGE keys (safe
/// over-approximation — a false conflict at worst, never a missed one).
#[inline]
pub fn prop_tag(key: &str) -> u64 {
    grafeo_common::utils::hash::hash_one(key) // reuse the registry's hasher (foldhash)
}
```
(Confirm `hash_one` is the same helper `read_registry.rs` uses; if not, use the available stable hasher.)
- [ ] **Step 3:** PASS. Commit (`feat(mvcc): PropTag + prop_compatible for conflict granularity`).

---

## Task 2: read-set + registry carry `PropTag` (entity-level default — behavior-preserving)

**Files:** `read_registry.rs`, `manager.rs`. **This is the big mechanical generalization; it MUST NOT change behavior (every caller passes `None` until Task 4).**

- [ ] **Step 1 (test):** registry: `record_reader(N, t1, None)` then `readers_of_compatible(N, None)` == {t1}; with `record_reader(N, t2, Some(7))`: `readers_of_compatible(N, Some(7))` == {t1,t2} (t1 None matches anything), `readers_of_compatible(N, Some(9))` == {t1} (t2's Some(7) incompatible with Some(9); t1's None matches). Manager: a read recorded under default policy still aborts the write-skew (existing test) — i.e. all-None behaves exactly as today. FAIL.
- [ ] **Step 2:**
  - `ReadRegistry`: value type `FxHashMap<EntityId, Vec<(TransactionId, PropTag)>>` (or `FxHashSet<(TransactionId, PropTag)>`); `record_reader(entity, tx, tag)`; `readers_of_compatible(entity, write_tag) -> Vec<TransactionId>` returning readers whose stored tag is `prop_compatible` with `write_tag` (dedup tx). `remove_reader(tx)` unchanged (drops all of tx's entries). Keep `by_tx` GC index.
  - `manager.rs`: `read_set: HashSet<(EntityId, PropTag)>`; `record_read` gains a `tag: PropTag` param (callers in this task pass `None`). Read-time/write-time detection use `readers_of_compatible` / compatibility against the writer's recorded tags. `retired_readers` unaffected by tag (it maps tx→commit_epoch). **W-W check + `write_set: HashSet<EntityId>` UNCHANGED** (entity-level).
  - **Crucially:** under all-`None`, `readers_of_compatible(e, None)` == old `readers_of(e)` and `(e, None)` set membership == old `e` membership. Verify the F2 suite is green with zero behavior change.
- [ ] **Step 3:** PASS; full gate green; serializable suite green (unchanged). Commit (`refactor(mvcc): conflict keys carry PropTag (entity-level no-op default)`).

---

## Task 3: `ReadTracker` property path + store recording (still `None` by default)

**Files:** `grafeo-core` `ReadTracker` trait + `store/mod.rs` chokepoints + property accessors + layered/test impls; engine bridge `read_tracker.rs`.

- [ ] **Step 1 (test):** a test `ReadTracker` double records `(id, PropTag)`; calling `read_node_property_visible` records the entity with the property's tag *when the store is in property mode*, `None` otherwise; a structural accessor (e.g. existence) records `None`. FAIL.
- [ ] **Step 2:**
  - `ReadTracker` trait: add `record_node_property_read(&self, tx, id: NodeId, tag: PropTag)` + `record_edge_property_read(&self, tx, id: EdgeId, tag: PropTag)` with **default methods** delegating to the existing `record_node_read`/`record_edge_read` (so existing impls keep compiling and behave entity-level). `TransactionReadTracker` (bridge) overrides them to forward `tag` to `manager.record_read`.
  - Store: `record_read_node`/`record_read_edge` gain an internal `tag` path; the property accessors (`read_node_property_visible`/`read_edge_property_visible` + the tiered twins) compute `prop_tag(key)` and pass it **only when the store's granularity is Property** (a store-level flag set from the policy, default Entity → pass `None`). Structural accessors keep recording `None`. (The store learns its granularity the same way it learns the active tracker — via the registration in Task 4.)
  - Bridge: `record_read(tx, entity, tag)`.
- [ ] **Step 3:** PASS; full gate green (default policy → all `None` → unchanged). Commit (`feat(mvcc): property-aware read-recording path (inert under entity policy)`).

---

## Task 4: the `ConflictGranularity` policy + enable + demonstration

**Files:** engine session/db (policy), the store registration, `tests/serializable.rs`.

- [ ] **Step 1 (the headline demonstration test):** `disjoint_property_writes_do_not_abort_under_property_granularity`. Seed `(:Account {id:1, balance:100})`, `(:Account {id:2, balance:100})`. Two Serializable sessions under **Property** granularity, both `MATCH (a:Account {id:K}) SET a.balance = ...` for disjoint K (1 and 2). Interleave like the write-skew test (both read via the scan, both write, commit in order). **Assert BOTH commit Ok** (the scan reads the `id` column; the writes touch `balance` — different properties, no rw-antidependency). Add the contrast `..._abort_under_entity_granularity` (same workload, **Entity** policy) asserting the second **aborts** (the documented entity-granular over-abort) — proving the knob is what changes the outcome. Write first → the property-granularity test FAILS today (entity-level aborts it).
- [ ] **Step 2:**
  - Add `ConflictGranularity { Entity, Property }` (default `Entity`) as a database option + a session/transaction selector (mirror how `IsolationLevel` is threaded). Plumb it: when a Serializable tx begins under `Property`, the store/bridge records property tags (set the store-level granularity flag from Task 3 + the manager records real tags); under `Entity`, everything stays `None`.
  - Confirm the policy is **per-transaction/session** (so the default-`Entity` global gate stays green and only the opted-in test sees property tags).
- [ ] **Step 3:** the demonstration passes (property → both commit; entity → second aborts); write-skew on the SAME property still aborts under Property granularity (a real conflict — add `same_property_write_skew_still_aborts_under_property_granularity`); the full F2 suite (default Entity) unchanged. Full gate green. Commit (`feat(mvcc): property-level conflict-granularity policy (Part G knob)`).

---

## Task 5: verification + audit + docs

- [ ] `--all-features -p grafeo-core -p grafeo-engine` green; `--features full --test serializable` green (default-Entity suite unchanged + the 3 new property-granularity tests).
- [ ] clippy `--all-features … -D warnings` clean; profiles + `grafeo-wasm` compile.
- [ ] **Soundness audit:** (a) entity-level default path is byte-unchanged (all `None`; `prop_compatible(None,*)` always true → old behavior); (b) property-level NEVER misses a real conflict — same-property write-skew still aborts, structural (None) reads/writes still conflict with everything (delete/label changes); (c) hash collision is a *merge* (over-approx), never a split; (d) the W-W check stayed entity-level; (e) the read-recording stayed complete-by-construction (every visible read still records — the property is extra info, never a gate); (f) policy is Serializable-only and opt-in.
- [ ] Update the F2 plan's "Residuals" note: same-label/wide-node false aborts now have an opt-in mitigation (Property granularity); scan/predicate-granularity remains the residual. `git status` clean; OPSEC.

---

## Acceptance
- An opt-in `ConflictGranularity::Property` policy tracks rw-antidependencies per `(entity, property)`; the **deterministic demonstration** shows the disjoint-property scan workload commits under Property and aborts under Entity.
- The **entity-level default is provably unchanged** (the full F2 suite + 7483-gate green with zero behavior delta); property-level is sound (same-property conflicts + structural conflicts still abort; collisions over-approximate).
- W-W lost-update prevention stays entity-level; read-recording stays complete-by-construction.
- Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC-clean.
- **Next:** optional throughput benchmarks (disjoint-write scaling, abort-rate under contention); the integrate-or-guard follow-ups (MVCC-integrate shortestPath/vector/text/algos to drop their Serializable guards); scan/predicate-granularity (narrow which nodes a filtered scan records — the deeper residual).

## Risks
- **Touching the sound read-recording foundation.** The trait gains a property path; the risk is a recording site that stops recording. Mitigation: default methods delegate to the existing entity-level record (no site can silently drop a read); Tasks 1-3 are all-`None` behavior-preserving, so any regression shows immediately in the default-path gate before property-level is enabled.
- **A missed conflict = unsound.** `prop_compatible` must treat `None` as a wildcard on BOTH sides (a structural read must conflict with a property write and vice-versa). The same-property-still-aborts + structural-conflict tests are the guard. Hash collisions merge (safe); never split.
- **Behavior drift on the default path.** If any Task-1-3 caller passes a non-`None` tag prematurely, the default changes. Mitigation: grep that only the property accessors (Task 3/4) ever produce `Some`; everything else `None`.
- **Policy plumbing breadth.** Threading `ConflictGranularity` mirrors `IsolationLevel`; follow that exact path. Keep it per-session so the global default-Entity gate is untouched.
- **Property identity via hash.** Acceptable (collisions over-approximate); a later refinement can use the store's interned property id for exactness + cheaper keys.

---

## STATUS: NOT STARTED

Plan written against `integration` @ `eecf0359` (F2 merged). Scope: **opt-in property-level conflict-granularity knob** (the spec's Part-G performance knob), behavior-preserving entity-level default, demonstrated deterministically. **Invasive** (a `grafeo-core` `ReadTracker` trait change + store recording + manager key generalization + a session/db policy) and it touches the sound read-recording foundation — recommend executing with fresh context, reviewing the trait change + the all-`None` behavior-preservation (Tasks 1-3) carefully, and gating each task on the unchanged default-path suite. On completion: throughput benchmarks + the integrate-or-guard follow-ups.
