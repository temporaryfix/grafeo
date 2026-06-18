# Guarded-Operator Integration (snapshot-aware traversal + index assessment) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Serializable **guards** on the non-MVCC operators by making them snapshot-aware and SSI-read-recording — starting with the tractable, highest-value one (**shortestPath**), generalizing to **graph algorithms**, and **assessing** (not forcing) the index-based operators (**vector / text**), which need index versioning. Net result: Serializable supports graph traversal/algorithms; the index operators stay soundly guarded with a documented integration path.

**Why this is its own plan (not part of F1/F2/G):** the Serializable arc (F1 sound enable → F2 incremental SSI → Part G granularity) is complete and merged. These operators were *guarded* (rejected under Serializable) because they **bypass the MVCC visible-read API**: shortestPath/algos read raw `edges_from` adjacency (no epoch/tx visibility, no read recording); vector/text search global HNSW/BM25 indexes that reflect committed state, not the transaction's snapshot. The guards are the **sound, correct** interim answer. Integrating requires *new* MVCC infrastructure — chiefly a **versioned traversal API** — which touches the store's adjacency/visibility core and deserves its own careful cycle.

**Architecture (the key reuse):** A versioned traversal is **base adjacency filtered by the existing per-edge visibility chokepoint**. `is_edge_visible_versioned(edge, epoch, tx)` (store-level increment) already (a) decides snapshot visibility — created ≤ epoch or own-write, not deleted ≤ epoch or own-delete — and (b) **records the read into the SSI read-set**. So `edges_from_versioned(node, dir, epoch, tx) = edges_from(node, dir).filter(|(_, e)| is_edge_visible_versioned(e, epoch, tx))` gives a snapshot-consistent, SSI-recording traversal *for free* on the existing infra — no parallel versioning machinery. ShortestPath/algos then thread `(epoch, tx)` and traverse through it.

**Tech Stack:** Rust, `cargo test`. `grafeo-core` (store traversal + the operators), `grafeo-engine` (planner: thread `(epoch, tx)`, drop the guard). `CARGO_INCREMENTAL=0`.

---

## Orientation

On `integration` @ `208f9811` (Part G merged). The guards (store-level increment, Task 4):
- `plan_shortest_path` (`query/planner/lpg/mutation.rs:667`) — rejects Serializable; comment: *"reads raw edges_from adjacency with no MVCC visibility and cannot record reads for SSI."*
- `plan_call_procedure` / graph algorithms (`mutation.rs:741`).
- `plan_text_scan` (`mod.rs:1003`), `plan_vector_scan` (`mod.rs:1071`).

Building blocks (verified present):
- **Base traversal** `LpgStore::edges_from(node, direction) -> impl Iterator<(NodeId, EdgeId)>` (`store/traversal.rs:44`) + `neighbors` (`:15`) — base `forward_adj`/`backward_adj`, no visibility.
- **Edge visibility chokepoint** `is_edge_visible_versioned(edge, epoch, tx)` (`store/edge_ops.rs:1032`, tiered twin `:1053`) — records the read (SSI) + decides snapshot visibility. Node twin: `is_node_visible_versioned`.
- **The operator** `ShortestPathOperator` (`execution/operators/shortest_path.rs`) holds `store: Arc<dyn GraphStoreSearch>`, traverses via `get_neighbors_directed` → `self.store.edges_from(node, dir)` + `self.store.edge_type(edge_id)` — **no `(epoch, tx)` today** (constructed without them).
- **Read-tracker plumbing:** the store records reads to the registered `ReadTracker` (per-tx); a Serializable operator just needs to route through the `*_versioned` accessors and the recording happens at the chokepoint.

**⚠️ Verify-first unknowns (Task 0):** (1) Does the base `forward_adj`/`backward_adj` **retain tombstoned (deleted) edges** so an old snapshot can still see an edge deleted after its start? (The audit fix "neighbors()/edges_from() no longer resurrect promoted-then-deleted entities" suggests tombstone awareness — confirm the direction: a *committed-after-my-start* delete must STILL be visible to me.) If deletes are physically removed from the adjacency, `edges_from_versioned` can't reconstruct them and needs the version chain — a bigger change. (2) Are the tx's **own uncommitted created edges** present in the base adjacency (so `is_edge_visible_versioned` can surface them as own-writes)? Confirm before building Task 1.

**Verification gate (each task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full --test serializable`. Clippy `--all-features … -D warnings`. Profiles + wasm. Hygiene: `rustfmt --edition 2024` changed files only (NOT `cargo fmt`); `git status` clean (untracked `ce/` exists — never add it); `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC: generic labels.

---

## Task 0: Confirm the adjacency-tombstone + own-create invariants

**Files:** read-only investigation; write findings into this plan.

- [ ] **Step 1:** Write a store-level test: tx A (epoch 0) sees edge E; tx B deletes E and commits (epoch 1); tx A (still at snapshot 0) calls `edges_from(src, dir)` — is E still returned by the base adjacency (to be filtered *in* by `is_edge_visible_versioned` for A)? And: tx A creates edge F (PENDING) — does `edges_from` return F for A?
- [ ] **Step 2:** If both hold (base adjacency retains tombstoned edges + own pending creates), proceed to Task 1 as written. If deletes are physically pruned from adjacency, STOP and escalate: `edges_from_versioned` must instead walk the edge version chain (a larger change) — re-plan Task 1.

---

## Task 1: `edges_from_versioned` / `neighbors_versioned` (snapshot-aware traversal)

**Files:** `crates/grafeo-core/src/graph/lpg/store/traversal.rs`; the `GraphStore`/`GraphStoreSearch` trait (`graph/traits.rs`); `compact/layered.rs` (override for base-resident edges).

- [ ] **Step 1 (test):** seed a small graph; tx at epoch E. `edges_from_versioned(n, dir, E, Some(tx))` returns exactly the base `edges_from` edges that pass `is_edge_visible_versioned(e, E, tx)`; a concurrently-committed-after-E edge is excluded; the tx's own pending edge is included; **and** the read-set now contains those edges (recording happened). FAIL.
- [ ] **Step 2:** Implement on `LpgStore`:
```rust
pub fn edges_from_versioned(&self, node: NodeId, direction: Direction, epoch: EpochId, tx: Option<TransactionId>)
    -> Vec<(NodeId, EdgeId)> {
    self.edges_from(node, direction)
        .filter(|(_, e)| self.is_edge_visible_versioned(*e, epoch, tx)) // records the read for SSI
        .collect()
}
// neighbors_versioned likewise, deriving targets from the visible edges (so a node reached
// only via an invisible edge is not yielded). Record the node too via is_node_visible_versioned
// where the operator needs node-level reads.
```
Add to the trait with an entity-level default (delegating to `edges_from` + no recording) so non-Lpg stores compile; `LayeredStore` overrides to filter base-resident edges through the overlay tracker (mirror its existing `*_versioned` overrides). **Note:** the filter calls `is_edge_visible_versioned`, which is the SSI recording chokepoint — so traversal reads are recorded by construction, exactly the property the guard comment said was missing.
- [ ] **Step 3:** Run → PASS. Full gate. Commit (`feat(mvcc): edges_from_versioned snapshot-aware traversal (records reads)`).

---

## Task 2: thread `(epoch, tx)` into `ShortestPathOperator` and traverse versioned

**Files:** `execution/operators/shortest_path.rs`; `query/planner/lpg/mutation.rs` (construction).

- [ ] **Step 1 (test):** a Serializable tx runs a shortest-path over a seeded graph and gets the correct path; a concurrent committed edge insertion on the path is NOT seen by the tx's snapshot (snapshot consistency); the traversed edges/nodes are in the read-set.
- [ ] **Step 2:** Give `ShortestPathOperator` an `epoch: EpochId` + `tx: Option<TransactionId>` (set at plan time from the active transaction — mirror how scan operators receive the snapshot). Change `get_neighbors_directed` to call `self.store.edges_from_versioned(node, dir, self.epoch, self.tx)` instead of `edges_from`; use the visible edge set for `edge_type` filtering. Under SI the same path now also becomes snapshot-consistent (a latent SI correctness improvement — note it, keep behavior tested).
- [ ] **Step 3:** Run → PASS. Full gate. Commit (`feat(mvcc): shortestPath traverses the tx snapshot + records reads`).

---

## Task 3: drop the shortestPath guard + Serializable acceptance

**Files:** `query/planner/lpg/mutation.rs`; `tests/serializable.rs`; the guard-assertion test at `mod.rs:4070`.

- [ ] **Step 1 (test):** in `serializable.rs`, a Serializable shortest-path that reads a path, concurrent with another tx that writes (deletes/adds) an edge on that path → the SSI rw-antidependency is detected (the second committer aborts, or the path tx aborts per the dangerous-structure rule). And a benign shortest-path (disjoint graph region) commits.
- [ ] **Step 2:** Remove the `is_serializable()` rejection in `plan_shortest_path` (`:667`). Update the planner test that asserts the rejection (`mod.rs:4070` `!msg.contains("...shortestPath")`) — flip it to assert shortestPath now plans under Serializable. Keep `vector`/`text`/`algorithms` guards (still rejected) — confirm the `_ => Err` catch-all + the other guards remain.
- [ ] **Step 3:** Run → PASS (the new acceptance + the flipped guard test). Full gate. Commit (`feat(mvcc): enable Serializable for shortestPath (guard removed)`).

---

## Task 4: graph algorithms (CALL) — integrate or keep guarded, per algorithm

**Files:** investigation + `query/planner/lpg/mutation.rs:741`; the algorithm operators.

- [ ] **Step 1:** Enumerate the CALL graph algorithms and how each reads the store. Those that traverse via `edges_from`/`neighbors` → route through `edges_from_versioned` + thread `(epoch, tx)` (same as Task 2) and drop their share of the guard. Those that need global structure (e.g. precomputed indexes, whole-graph matrices) or are read-only-analytical with no snapshot contract → keep guarded, with a per-algorithm note.
- [ ] **Step 2:** For the traversal-based ones, mirror Tasks 2-3 (versioned traversal + record + acceptance test + drop guard). For the rest, leave the guard and document why. Commit per integrated algorithm.

---

## Task 5: vector / text search — index-versioning assessment (NOT forced)

**Files:** this plan (findings); optionally a documented partial path.

- [ ] **Step 1:** Document the core problem: HNSW (`scan_vector.rs`) and BM25 (`scan_text.rs`) search **global indexes that reflect committed state**, not the tx snapshot — so under Serializable they (a) can't record per-result reads soundly, (b) miss the tx's own uncommitted inserts (read-your-writes), (c) may return committed-after-start results. The index has no per-epoch versioning.
- [ ] **Step 2:** Assess the options, recommend, do NOT build blind:
  - **Snapshot post-filter (partial, possibly tractable):** run the index search (committed view), then filter the K results through `is_node_visible_versioned(epoch, tx)` (records the reads). Gives visibility-correctness + SSI recording for returned results, but NOT completeness (misses own uncommitted inserts) — so it is **not** true snapshot isolation; only acceptable if documented as "approximate, committed-index-based". Probably still warrants keeping the guard.
  - **Versioned index (full, research-grade):** per-epoch index snapshots / MVCC-aware HNSW+BM25 — a large separate project.
  - **Recommendation:** keep vector/text **guarded** until a versioned-index design exists; record the post-filter as a documented option, not a default.
- [ ] **Step 3:** Update the guard comments to point at this plan's findings (so the "not yet supported" message is backed by a rationale, not a TODO). No code behavior change.

---

## Acceptance
- A `edges_from_versioned`/`neighbors_versioned` snapshot-aware traversal exists and **records reads via the existing visibility chokepoint** (SSI-sound by construction).
- **shortestPath works under Serializable** (guard removed): correct snapshot-consistent paths, rw-antidependencies detected, benign paths commit. Traversal-based graph algorithms similarly integrated; the rest documented-and-guarded.
- **vector/text remain soundly guarded** with a documented integration path (post-filter vs versioned index); the guards reference the rationale.
- Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC-clean. The `_ => Err` catch-all + remaining guards still fail new non-MVCC ops closed.

## Risks
- **Adjacency tombstone semantics (Task 0 gate).** If deletes are physically pruned from `forward_adj`, `edges_from_versioned` can't reconstruct old-snapshot-visible edges by filtering — it needs the edge version chain. Confirm BEFORE Task 1; re-plan if so.
- **Over/under-recording on traversal.** `edges_from_versioned` records every *visible* edge it filters — good for soundness, but a deep traversal records a large read-set (false-abort pressure). Entity-granular; Part-G property granularity doesn't help structural edge reads. Acceptable (sound > precise); note it.
- **Snapshot consistency vs the SI status quo.** ShortestPath today reads the live committed graph even under SI (a latent inconsistency). Routing through `*_versioned` *fixes* SI too — verify no test depended on the old live-read behavior.
- **Vector/text temptation.** The post-filter looks like an integration but is NOT snapshot-complete (misses own writes). Do not silently enable it as "Serializable support" — that would be unsound-by-omission. Keep guarded unless explicitly accepted as approximate.
- **Operator `(epoch, tx)` plumbing.** ShortestPath/algos are constructed in the planner without the snapshot; thread it the same way scans receive it. Don't bypass the planner's snapshot source.

---

## STATUS: shortestPath INTEGRATED + merged to `integration` @ `ae0b8728` — algos/vector/text remain guarded

**Done + merged (5 commits `dec609e7..ae0b8728`):**
- **Task 0** (tombstone/own-create gate): resolved — base adjacency soft-deletes (edge retained in chunks), so `edges_from_versioned` iterates a RAW adjacency (`iter_including_deleted`, skipping the `deleted` filter) + filters by `is_edge_visible_versioned` (the version-chain authority, which records the read).
- **Task 1** — `edges_from_versioned`/`neighbors_versioned` snapshot-aware traversal (records reads via the visibility chokepoint). LpgStore verified correct by opus review.
- **Task 1b + completion (DISCOVERED PREREQUISITE, not in the original plan):** the Task-1 opus review **probe-confirmed a PRE-EXISTING LayeredStore bug** — base-edge AND base-node deletes were epoch-blind (`deleted_from_base_{edges,nodes}` plain sets) → a base entity deleted-after-my-snapshot was hidden from EVERY snapshot (not snapshot-isolated, post-`compact()`). FIXED: epoch+tx-stamped both tombstones (`BaseEdgeDelete`/`BaseNodeDelete{epoch, deleter}`; snapshot-aware `_at` predicates for the 5+6 versioned accessors; latest-view `contains_key` preserved for the non-versioned; `pending_base_*_deletes` driving commit-finalize + rollback; persistence keys-only, no on-disk format change). 3 opus reviews (edge fix; completion incl. 2 missed edge property accessors; node mirror = faithful). See [[grafeo-audit-findings]].
- **Task 2** — `ShortestPathOperator` threads `(epoch, tx)`, traverses `edges_from_versioned` + records node reads (`is_node_visible_versioned`); `None` fallback = unchanged non-tx behavior.
- **Task 3** — shortestPath Serializable guard REMOVED. Acceptance: a write-skew cycle through the shortestPath read-set aborts the second committer (proving the path reads reach the SSI read-set); disjoint paths commit; vector/text/algorithms still guarded.

**Verification:** `--all-features` 7514/0; serializable suite 11/11; clippy clean; profiles + wasm; OPSEC (upstream un-pushed).

**Remaining (still SOUNDLY guarded):**
- **Task 4 — graph algorithms (CALL):** the traversal-based ones can now integrate via the SAME `edges_from_versioned` + `(epoch, tx)` plumbing (the Task-2 pattern); whole-graph/index-based ones stay guarded. NOT done.
- **Task 5 — vector/text:** still guarded; need index versioning (HNSW/BM25 as-of-snapshot) — research-grade; the post-filter is partial (not snapshot-complete). Documented §5.
