# Graph-Algorithm (CALL) Serializable Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the **graph-algorithm CALL procedures** (PageRank, connected components, Dijkstra/SSSP/Bellman-Ford, Louvain, label propagation, MST, k-core, articulation points, bridges, …) and the **catalog-introspection procedures** (`labels`/`relationshipTypes`/`propertyKeys`) run under **Serializable** — snapshot-consistent and SSI-read-recording — by routing their store access through a single **snapshot-recording view**, then replacing the coarse `is_serializable()` rejection in `plan_call_procedure` with a **per-procedure** policy. Index-based procedures (`search.vector`/`search.text`) stay **guarded** (they need index versioning — out of scope, see §Subsequent steps).

**Why this is the next step:** the Serializable arc (F1 → F2 → Part G) is complete and shortestPath is integrated. The CALL guard (`query/planner/lpg/mutation.rs:741`) currently rejects **every** procedure under Serializable with one coarse check — over-rejecting the read-only introspection + traversal-based algorithms that the existing MVCC infra can already serve. The `edges_from_versioned`/`is_*_visible_versioned` accessors built for shortestPath (and the now-snapshot-correct LayeredStore base-delete model) are exactly what's needed.

**Architecture (the elegant core — one wrapper, ~14 algorithms for free):** the algorithms (`grafeo-adapters/src/plugins/algorithms/*`) all consume `&dyn GraphStore` and read via `store.node_ids()` (whole-graph) + `store.edges_from(node, dir)` (adjacency). Rather than thread `(epoch, tx)` through every algorithm, introduce a **`SnapshotView<'a>`** that wraps `(store, epoch, tx)` and implements `GraphStore`, overriding the **read** methods to route through the **snapshot-visible, read-recording** accessors:
- `node_ids()` → `filter_visible_node_ids_versioned(epoch, tx)` (snapshot-visible + records each via `is_node_visible_versioned`),
- `edges_from(n, dir)` → `edges_from_versioned(n, dir, epoch, tx)` (records edge reads),
- `neighbors`/`get_node`/property/label reads → their `*_versioned`/`*_visible` counterparts (which record at the store chokepoint).

Pass each algorithm a `SnapshotView` under Serializable ⇒ every algorithm becomes snapshot-consistent **and** populates the SSI read-set, complete by construction, with **zero per-algorithm changes**. The read-set is whole-graph (the algorithm genuinely reads the whole graph) → it correctly conflicts with any concurrent graph mutation; that coarseness is inherent to graph analytics under serializability and is documented, not a defect.

**Tech Stack:** Rust, `cargo test`. `grafeo-engine` (`procedures.rs` + the planner `plan_call_procedure`); the `SnapshotView` lives wherever it can see both the `GraphStore` trait and the versioned accessors (likely `grafeo-core` `graph/`, re-exported). `CARGO_INCREMENTAL=0`.

---

## Orientation

On `integration` @ `854a6a43` (shortestPath integrated; LayeredStore base-edge/node deletes epoch-versioned + snapshot-correct).

- **The guard:** `plan_call_procedure` (`grafeo-engine/src/query/planner/lpg/mutation.rs:741`) — `if self.is_serializable() { return Err("...graph algorithms...") }` BEFORE resolving the procedure name → rejects **all** CALLs under Serializable.
- **Procedures** (`grafeo-engine/src/procedures.rs`, 1495 lines): `Procedure` trait (`execute(&self, ctx: &ProcedureContext, params)`); `ProcedureContext { store: &dyn GraphStoreSearch, .. }` (`:76`). Kinds:
  - `GraphAlgorithmProcedure` (`:114`) wraps `Arc<dyn GraphAlgorithm>`; `execute` calls `self.inner.execute(ctx.store, params)`. ~14 algorithms registered (`:1021`-`:1154`).
  - Introspection: `LabelsProcedure` (`:157` `ctx.store.all_labels()`), `RelationshipTypesProcedure` (`:186`), `PropertyKeysProcedure` (`:215`) — catalog reads.
  - `SearchVectorProcedure` (`:301`, `#[cfg(vector-index)]`) + a text twin — HNSW/BM25 index reads.
- **The algorithms** (`grafeo-adapters/src/plugins/algorithms/`): `GraphAlgorithm` trait (`traits.rs:206`) `execute(&self, store: &dyn GraphStore, params)`. Read via `store.node_ids()` + `store.edges_from(node, dir)` (e.g. `components.rs:98,118`). Base (non-versioned, non-recording).
- **Available versioned/recording accessors** (built for shortestPath + the store-level increment): `edges_from_versioned`/`neighbors_versioned` (`graph/lpg/store/traversal.rs`, records via `is_edge_visible_versioned`); `is_node_visible_versioned` (records node read); `filter_visible_node_ids_versioned`; `get_node_versioned`; `read_node_property_visible`/`read_node_labels_visible`; all on `GraphStore`/`LpgStore`/`LayeredStore` (LayeredStore now snapshot-correct for base deletes).

**Verification gate (each task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full -p grafeo-engine --test serializable`. Clippy `--all-features … -D warnings`; profiles + `grafeo-wasm`. Hygiene: `rustfmt --edition 2024` changed files only (NOT `cargo fmt`); `git status` clean (untracked `ce/` — never add it); `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC: generic labels.

---

## Task 0: enumerate the `GraphStore` read surface the algorithms touch

**Files:** read-only; record findings in this plan.

- [ ] **Step 1:** grep `grafeo-adapters/src/plugins/algorithms/*` for every `store.` call. Build the set of `GraphStore` methods the algorithms actually use (expected: `node_ids`, `edges_from`, `neighbors`, maybe `get_node`, `node_count`, property/label reads). For each, confirm a **snapshot-visible, read-recording** counterpart exists (`*_versioned`/`*_visible` or `filter_visible_*`). 
- [ ] **Step 2:** If any used method has NO recording counterpart (e.g. a count or a whole-node-set read that bypasses the chokepoints), note it — the `SnapshotView` override for it must synthesize the recording (iterate + `is_node_visible_versioned`) or the algorithm reading it would have an **unrecorded read = unsound SSI**. List the exact methods `SnapshotView` must override vs can delegate. This is the completeness gate for Task 1.

---

## Task 1: `SnapshotView` — a snapshot-recording `GraphStore` wrapper

**Files:** create the wrapper (e.g. `grafeo-core/src/graph/snapshot_view.rs`); export it.

- [ ] **Step 1 (test):** build an `LpgStore` graph; a tx at epoch E with a registered read-tracker. `let v = SnapshotView::new(&store, E, tx);` — `v.node_ids()` returns only snapshot-visible nodes AND the read-set now contains them; `v.edges_from(n, dir)` equals `store.edges_from_versioned(n, dir, E, tx)` AND records; a concurrently-committed-after-E node/edge is excluded. FAIL (no `SnapshotView`).
- [ ] **Step 2:** Implement
```rust
pub struct SnapshotView<'a> { inner: &'a dyn GraphStore, epoch: EpochId, tx: TransactionId }
impl<'a> SnapshotView<'a> { pub fn new(inner: &'a dyn GraphStore, epoch: EpochId, tx: TransactionId) -> Self {...} }
impl<'a> GraphStore for SnapshotView<'a> {
    fn node_ids(&self) -> Vec<NodeId> { self.inner.filter_visible_node_ids_versioned(self.epoch, self.tx) } // records
    fn edges_from(&self, n: NodeId, d: Direction) -> Vec<(NodeId, EdgeId)> { self.inner.edges_from_versioned(n, d, self.epoch, self.tx) } // records
    fn neighbors(&self, n: NodeId, d: Direction) -> Vec<NodeId> { self.inner.neighbors_versioned(n, d, self.epoch, self.tx) }
    // ... every read method from Task 0 → its versioned/visible+recording counterpart;
    // delegate the rest (and write methods, which read-only algorithms never call — leave default/delegate).
}
```
   Cover EXACTLY the methods Task 0 enumerated (over-cover with recording is safe; under-cover = a missed read). For any read method whose counterpart only filters-without-recording, wrap it to also record (iterate + `is_node_visible_versioned`/`is_edge_visible_versioned`). Confirm `GraphStore`'s default methods don't silently provide a NON-recording path the algorithms hit.
- [ ] **Step 3:** PASS; full gate. Commit (`feat(mvcc): SnapshotView — snapshot-recording GraphStore wrapper`).

---

## Task 2: per-procedure guard + wrap the store under Serializable

**Files:** `grafeo-engine/src/procedures.rs` (ProcedureContext + the procedure classification), `grafeo-engine/src/query/planner/lpg/mutation.rs` (`plan_call_procedure`).

- [ ] **Step 1 (test):** planner-level — under Serializable, a `CALL` to a graph algorithm (e.g. `pagerank`/`connectedComponents`) and to `labels`/`relationshipTypes`/`propertyKeys` **plans OK**; a `CALL search.vector`/`search.text` still **rejects** with the guard message. FAIL (coarse guard rejects all).
- [ ] **Step 2:**
   - Give each `Procedure` a classification: `serializable_safe(&self) -> bool` (default per-kind) — `true` for `GraphAlgorithmProcedure` (reads only, will get a `SnapshotView`) + the introspection procedures (catalog reads; record a coarse read or are read-only-safe); `false` for `SearchVectorProcedure`/text (index, not snapshot-aware).
   - In `plan_call_procedure`: **remove the blanket `is_serializable()` reject**; resolve the procedure first, then `if self.is_serializable() && !proc.serializable_safe() { return Err("...<that procedure> not yet supported with Serializable; use SnapshotIsolation") }`.
   - Thread `(epoch, tx)` to the procedure execution and, under Serializable, build the `ProcedureContext` with the store wrapped in `SnapshotView(epoch, tx)` (so `GraphAlgorithmProcedure::execute(ctx.store, ..)` traverses the snapshot + records). Confirm `ProcedureContext.store` can be a `&dyn GraphStoreSearch` over a `SnapshotView` — if `SnapshotView` must also impl `GraphStoreSearch` (the search supertrait) for the context type, impl the search methods to delegate (algorithms don't call them; the introspection procedures call `all_labels()`-style reads — route those through visible/recording variants or accept catalog-level reads).
- [ ] **Step 3:** PASS; full gate. Commit (`feat(mvcc): per-procedure Serializable policy + SnapshotView for graph algorithms`).

---

## Task 3: acceptance — algorithms sound under Serializable

**Files:** `grafeo-engine/tests/serializable.rs`.

- [ ] **Step 1:** add:
   - `serializable_graph_algorithm_conflict_aborts`: T1 (Serializable) `CALL` a graph algorithm over the graph (reads whole graph → records it); T2 (Serializable, concurrent) writes an edge/node the algorithm read; commit order so the rw-antidependency triggers → the second committer / pivot gets `SerializationFailure`. (Whole-graph read-set ⇒ any concurrent graph write conflicts — that's the expected coarse-but-correct behavior; assert it.)
   - `serializable_introspection_commits`: T1 `CALL labels`/`propertyKeys` under Serializable concurrent with a disjoint write → commits.
   - `serializable_vector_search_still_rejected`: `CALL search.vector` under Serializable still errors.
   - A snapshot-consistency check: an algorithm under Serializable does NOT see a concurrently-committed-after-its-snapshot edge.
- [ ] **Step 2:** Run → pass. If the conflict case doesn't abort, trace whether `SnapshotView` actually recorded the algorithm's reads (register a tracker, inspect the read-set) — do NOT weaken. Commit (`test(mvcc): Serializable graph-algorithm acceptance`).

---

## Task 4: verification + drop the now-dead guard arms

- [ ] `--all-features -p grafeo-core -p grafeo-engine` green; `--features full --test serializable` green (algorithms commit/abort correctly; introspection commits; vector/text rejected).
- [ ] clippy clean; profiles + wasm. Update the guard-assertion planner tests (the ones asserting "graph algorithms" rejection) to the per-procedure reality (algorithms/introspection plan; vector/text reject). Confirm the `_ => Err` catch-all + the shortestPath/vector/text guards' integrity.
- [ ] **Soundness audit:** every algorithm read goes through `SnapshotView` (no algorithm holds a `&dyn GraphStore` to the *unwrapped* store under Serializable); the read-set is populated (whole-graph) so concurrent mutations conflict; SI/RC unaffected (no `SnapshotView`, no recording — the algorithms run on the raw store as before). `git status` clean; OPSEC.

---

## Acceptance
- Graph-algorithm + introspection CALLs run under Serializable, snapshot-consistent, with their reads in the SSI read-set (via `SnapshotView`); a concurrent graph mutation correctly aborts one; benign/disjoint commits.
- `search.vector`/`search.text` remain soundly **guarded** (per-procedure), with the catch-all intact.
- Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC-clean.
- SI/ReadCommitted CALL behavior **unchanged** (no `SnapshotView`, no recording overhead).

## Subsequent steps (NOT this plan)
- **Vector/text search** under Serializable — needs index versioning (HNSW/BM25 as-of-snapshot); research-grade; keep guarded.
- **Refinements:** concurrent-throughput benchmarks (Part G's other half); multi-property-`SET` property tagging; the `edge_type_versioned`-on-LayeredStore residual.

## Risks
- **Unrecorded read = unsound SSI.** If an algorithm calls a `GraphStore` method `SnapshotView` does NOT override (falls through to the raw store with no recording), that read escapes the read-set → a missed conflict. Task 0 enumerates the surface; Task 1 must cover all of it; the acceptance conflict test + a read-set inspection are the guard. Over-record rather than miss.
- **`GraphStore` vs `GraphStoreSearch` typing.** `ProcedureContext.store` is `&dyn GraphStoreSearch`; the algorithms take `&dyn GraphStore`. `SnapshotView` must satisfy whichever the context needs — if `GraphStoreSearch`, impl the search methods (delegate; algorithms don't use them, introspection uses catalog reads). Don't let a search-method call bypass recording for a procedure that's marked serializable_safe.
- **Whole-graph read-set / abort-thrash.** A Serializable graph algorithm reads the whole graph → conflicts with any concurrent writer. This is correct (the analysis depends on the whole graph) but means algorithms + concurrent writers abort-heavily under Serializable. Document it; it's inherent, not a bug. (Users wanting a consistent snapshot without abort risk use SnapshotIsolation.)
- **Introspection granularity.** `labels`/`propertyKeys` read the catalog, not per-entity. Decide: record a coarse catalog read (so a concurrent label/type addition conflicts) vs treat as read-only-safe. Simplest sound choice: a coarse read marker; document.
- **Parallel execution path.** `GraphAlgorithm::execute_parallel` (traits.rs:242) — if any algorithm uses it, confirm it also goes through `SnapshotView` (or is not taken under Serializable). Flag in Task 0.

---

## STATUS: COMPLETE — merged to `integration` @ `65730e89`

**Done + merged (Tasks 0-4):**
- **Task 0** (read-surface gate): algorithms are read-only; their entire `GraphStore` read surface is 6 methods (`node_ids`, `edges_from`, `neighbors`, `get_node`, `get_edge`, `find_nodes_by_property`), each with a recording counterpart (`find_nodes_by_property` synthesized via `is_node_visible_versioned`); `execute_parallel` uses the same methods.
- **Task 1** — `SnapshotView` (`grafeo-core/src/graph/snapshot_view.rs`): a `GraphStoreSearch` wrapper over `(store, epoch, tx)` whose 6 reads route through the snapshot-visible, SSI-recording accessors; everything else delegates. **Opus-reviewed COMPLETE** (no algorithm read escapes recording; `find_nodes_by_property`'s committed-index residual is sound — returned set == recorded set). Carries a maintainer completeness-invariant doc.
- **Task 2** — per-procedure `serializable_safe()` (GraphAlgorithm + introspection → true; `search.vector`/`search.text` → false, default false); the blanket CALL guard replaced with a post-resolution per-procedure check; the executor wraps the store in `SnapshotView` under Serializable (epoch+tx from the planner). Introspection catalog reads are NOT recorded (documented residual — benign).
- **Task 3** — acceptance: `serializable_graph_algorithm_conflict_aborts` (a write-skew cycle through PageRank's whole-graph read-set aborts the second committer — proving the algorithm reads reach the SSI read-set via `SnapshotView`); snapshot-consistency (post-snapshot node invisible); introspection commits; `search.vector` still rejected.
- **Task 4** — verification + the `SnapshotView` invariant doc + a Part-G test doc-lint fix.

**Verification:** `--all-features` 7543/0; serializable suite 15/15; clippy clean (incl. `--test serializable`); profiles (`default`/`lpg`/`lpg,temporal` engine, `grafeo-core` tiered) + wasm; OPSEC (upstream un-pushed).

**Residuals (documented):** whole-graph read-set ⇒ a Serializable algorithm conflicts with any concurrent graph mutation (correct, but abort-prone — use SI for snapshot-without-abort); introspection catalog reads unrecorded; `find_nodes_by_property` committed-index view (uncommitted-match incompleteness); a future algorithm calling a non-overridden `GraphStore` read must add a `SnapshotView` override (per its doc).

**Still guarded:** `search.vector`/`search.text` — need index versioning (research-grade; §Subsequent steps).
