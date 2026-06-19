# Snapshot-Versioned HNSW Vector Search under Serializable — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HNSW kNN search snapshot-consistent (as-of-epoch visible nodes, scored with as-of-epoch vectors, + read-your-writes, recall preserved via widened `ef`) and SSI-sound (coarse `EntityId::Index` recording), then remove the **last** Serializable guard.

**Architecture:** The HNSW is a snapshot-filtered **candidate generator** over the node/property MVCC chains — NOT an index-internal version history. Visibility from `is_node_visible_versioned`; as-of-E vector values from a snapshot-aware `VectorAccessor` reading `read_node_property_visible`; topology retains deleted nodes (soft-delete) for connectivity + GC; per-tx read-your-writes from `tx_property_overlay`; coarse `EntityId::Index` recording at the operator chokepoint (reused from text).

**Tech Stack:** Rust, `cargo test`. `grafeo-core` (`index/vector/`, `graph/lpg/store/`, `execution/operators/scan_vector.rs`, `vector_join.rs`), `grafeo-engine` (`transaction/`, `procedures.rs`, planner). `CARGO_INCREMENTAL=0`.

**Spec:** `docs/superpowers/specs/2026-06-18-mvcc-vector-index-snapshot-versioning-design.md`. Read it first, especially §3 (why B‴) and §3a-rejected (A).

---

## Orientation & reuse templates

On `integration` @ `fb6ffd04`+. **Reuses the merged text cycle:**
- `EntityId::Index(IndexId)` + `IndexId::for_text_index` (`transaction/manager.rs`) — generalize to `for_index`; the read/write-tracker bridges + `record_read_index`/`record_write_index` (store) + the session registration are all in place.
- The **recording-completeness discipline** (execution-time at the operator, complete coverage of every entry point) — text commits `73164b9c`/`a7730a8a`; memory [[project_serializable_guard_recording]].
- GC-at-horizon (`database/mod.rs gc()` + `min_active_epoch`) — text `7e61f78a`.
- `read_node_property_visible(id, key, epoch, tx)` (`store/property_ops.rs:1089`) — the as-of-E property oracle (also honours the tx's `tx_property_overlay`).
- `is_node_visible_versioned` — the node-chain visibility predicate (SnapshotView / graph-algos).

**Current vector code:**
- `HnswIndex` (`index/vector/hnsw.rs`): `HnswNode { neighbors }` (topology only, **no vectors**); `insert(id, vec, accessor)`, `remove(id)` (**hard delete** `:745`), `search`/`search_with_ef`/`search_with_filter` (naive post-filter), `search_layer` (beam, `:491/638`). `VectorAccessor::get_vector(id) -> Option<Arc<[f32]>>` (`accessor.rs:33`).
- `QuantizedHnswIndex` (`quantized_hnsw.rs`): `vectors: HashMap<NodeId, Arc<[f32]>>` (full-precision, for rescore) + quantized maps; coarse-search-then-rescore.
- Maintained at `crud.rs` (`index.insert`/`remove` on node create/delete/set-vector). Guarded: `plan_vector_scan` (`planner/lpg/mod.rs:1078`), `SearchVectorProcedure.serializable_safe()==false` (`procedures.rs:382`), `VectorScanOperator` (`scan_vector.rs:59`, `execute_search :129`).

**Gate (each task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full --test serializable`. Clippy `-D warnings`; profiles + wasm. Hygiene: `rustfmt --edition 2024` per file (NOT `cargo fmt`); `git status` clean except untracked `ce/` (never add it); `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC (un-pushed).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `index/vector/hnsw.rs` | retain-deleted (`deleted` set + soft `remove`); `search_visible` (filtered beam) | Modify |
| `index/vector/quantized_hnsw.rs` | same retain-deleted + `search_visible` (coarse latest, rescore via accessor) | Modify |
| `index/vector/mod.rs` | `VectorIndexKind::search_visible` dispatch | Modify |
| `graph/lpg/store/vector_accessor.rs` (new) | `SnapshotVectorAccessor { store, property, epoch, tx }` → `read_node_property_visible` | Create |
| `graph/lpg/store/index.rs` + `graph_store_impl.rs` + `traits.rs` | `vector_search_visible` (accessor + is_visible + tx-overlay brute-force) + trait method + wrapper delegation | Modify |
| `transaction/manager.rs` | `IndexId::for_index` (generalize `for_text_index`) | Modify |
| `execution/operators/scan_vector.rs`, `vector_join.rs` | route through `vector_search_visible` (epoch/tx) + record `EntityId::Index` execution-time | Modify |
| `procedures.rs` | `SearchVectorProcedure.serializable_safe=true` + route | Modify |
| `query/planner/lpg/mod.rs` | drop `plan_vector_scan` guard; thread epoch/tx; flip tests | Modify |
| `database/mod.rs` + `store/index.rs` | vector GC (rebuild, drop deleted-below-horizon) | Modify |
| `tests/serializable.rs` | acceptance | Modify |

---

## Task 1: Retain-deleted topology (soft-delete) — behavior-preserving

**Files:** `index/vector/hnsw.rs`, `quantized_hnsw.rs`.

- [ ] **Step 1 (failing test):** in `hnsw.rs` tests, insert ids 1,2,3; `remove(2)`; assert (a) `search(query, 3)` does NOT return 2 (committed-latest excludes deleted); (b) node 2 is **still in the topology** (a new `pub(crate) fn contains_including_deleted(2) == true` / the node count incl. deleted is unchanged). FAIL.
- [ ] **Step 2:** Add `deleted: RwLock<HashSet<NodeId>>` to `HnswIndex` (and `QuantizedHnswIndex`). Change `remove(id)` to a **soft-delete**: insert `id` into `deleted`, **do NOT** remove from `nodes`/topology and **do NOT** re-link (keep it as a routing hop). `contains(id)` returns `!deleted.contains(id) && topology has id`. The committed-latest `search`/`search_with_ef` filter out `deleted` members from results (add `&& !deleted.contains(id)` at result collection). `len()`/`is_empty()` count live (non-deleted). Re-insert of an existing id removes it from `deleted` + updates topology (overwrite).
- [ ] **Step 3:** Run → PASS. Full gate: existing vector tests stay green (committed-latest results unchanged — deleted nodes excluded as before; they're just physically retained now). Commit (`feat(vec-mvcc): retain-deleted HNSW topology (soft-delete) — behavior-preserving`).

---

## Task 2: Predicate-filtered, accessor-scored `search_visible`

**Files:** `index/vector/hnsw.rs`, `quantized_hnsw.rs`, `mod.rs`.

- [ ] **Step 1 (test):** `HnswIndex::search_visible(query, k, ef, is_visible: &dyn Fn(NodeId)->bool, accessor) -> Vec<(NodeId,f32)>`: with nodes 1..=10 inserted, `is_visible = |id| id != 5 && id != 7`, assert the result excludes 5 and 7 but still finds k visible neighbors (the beam traverses through 5/7 but doesn't return them), and is non-empty for k≤8. FAIL.
- [ ] **Step 2:** Implement `search_visible`: run the layer-0 beam (`search_layer` with `ef_search = ef.max(k * F)`, F e.g. 4) — traverse **through all** neighbors for connectivity — but when building the returned top-k heap, **include a candidate only if `is_visible(id)`** (and not in `deleted` for the committed sense — but `is_visible` is the authority under a snapshot). Score with the passed `accessor`. Return top-k visible by distance. For `QuantizedHnswIndex::search_visible`: coarse-search the quantized codes (latest) for candidates, then **rescore with the passed accessor** (as-of-E full-precision), filter by `is_visible`. Add `VectorIndexKind::search_visible` dispatch (`mod.rs`).
- [ ] **Step 3:** PASS; gate. Commit (`feat(vec-mvcc): predicate-filtered search_visible (traverse-all, collect-visible, widen ef)`).

---

## Task 3: `SnapshotVectorAccessor` (as-of-E values)

**Files:** Create `graph/lpg/store/vector_accessor.rs`; wire `mod.rs`.

- [ ] **Step 1 (test):** a node with `SET n.embedding=[1,0,0]` committed at C1, then `[0,1,0]` at C2. `SnapshotVectorAccessor { store, property:"embedding", epoch: C1, tx: INVALID }.get_vector(n)` returns `[1,0,0]` (as-of-C1); at `epoch: C2` returns `[0,1,0]`. FAIL.
- [ ] **Step 2:** `pub(crate) struct SnapshotVectorAccessor<'a> { store: &'a LpgStore, property: PropertyKey, epoch: EpochId, tx: TransactionId }` impl `VectorAccessor`: `get_vector(id)` = `self.store.read_node_property_visible(id, &self.property, self.epoch, self.tx)` → if `Some(Value::Vector(v))`/list-of-f32, return `Some(Arc::from(v.as_slice()))`, else `None`. (Match the existing `Value` → vector conversion used by `crud.rs` insert.)
- [ ] **Step 3:** PASS; gate. Commit (`feat(vec-mvcc): snapshot-aware VectorAccessor (read_node_property_visible)`).

---

## Task 4: Store `vector_search_visible` (visibility + read-your-writes)

**Files:** `graph/lpg/store/index.rs`, `graph_store_impl.rs`, `traits.rs` (+ wrapper delegations).

- [ ] **Step 1 (test):** store-level. Build an index on `(:Doc, embedding)`; a Serializable tx `SET`s a new `:Doc` with a vector (uncommitted); `vector_search_visible(index_key, query, k, epoch, tx)` returns it (read-your-writes) while the committed-latest `vector_search` does not. A node committed-after-`epoch` is excluded; a node deleted-after-`epoch` is included. FAIL.
- [ ] **Step 2:** `LpgStore::vector_search_visible(index_key, query, k, epoch, tx)`: look up the committed `VectorIndexKind`; build `SnapshotVectorAccessor` + `is_visible = |id| self.is_node_visible_versioned(id, epoch, tx)`; call `idx.search_visible(query, k, ef, &is_visible, &accessor)`; then **brute-force the tx's uncommitted vectors** for this `(label,property)` from `tx_property_overlay` (iterate the tx's buffered `SET`s on `property` where the node has the label, compute distance, merge into top-k; a tx-deleted node is already excluded by `is_visible`). Add `GraphStoreSearch::vector_search_visible` (default delegates to committed-latest `vector_search`); override on `LpgStore`; **delegate on `WalGraphStore`/`CdcGraphStore`/`LayeredStore`/`SnapshotView`** (mirror text's `text_search_visible` delegation — `active_store()` is a `WalGraphStore`).
- [ ] **Step 3:** PASS; gate. Commit (`feat(vec-mvcc): vector_search_visible (snapshot visibility + read-your-writes)`).

---

## Task 5: `EntityId::Index` recording for vector

**Files:** `transaction/manager.rs`, `graph/lpg/store/index.rs`, the vector operators.

- [ ] **Step 1 (test):** manager-level — generalize `IndexId::for_text_index(label,prop)` to `IndexId::for_index(label,prop)` (keep `for_text_index` as an alias). `record_read(tx, EntityId::Index(IndexId::for_index("Doc","embedding")), None)` + a concurrent `record_write(tx2, same)` form an rw-edge. FAIL only if the alias/rename breaks; otherwise add a vector-keyed test.
- [ ] **Step 2:** **Write side** — wherever the vector index is maintained (`crud.rs`/the store path doing `index.insert`/`index.remove` for a vector property under a tx), call `record_write_index(tx, index_key)` (reuse the text write-record chokepoint; the key is `"label:property"`). **Read side** — `vector_search_visible` records `record_read_index(tx, index_key)` as its FIRST statement (mirror `search_text_visible`).
- [ ] **Step 3:** PASS; gate (existing SSI tests green — additive). Commit (`feat(vec-mvcc): EntityId::Index recording for vector search + writes`).

---

## Task 6: GC (rebuild, drop deleted-below-horizon)

**Files:** `index/vector/hnsw.rs` + `quantized_hnsw.rs`, `store/index.rs`, `database/mod.rs`.

- [ ] **Step 1 (test):** insert 1..=5, soft-`remove(3)`; `gc(live_predicate)` where `live_predicate(3)=false` (deleted-below-horizon) and true for others → rebuilds without 3; `search` unaffected for live nodes; node 3 gone from topology. FAIL.
- [ ] **Step 2:** `VectorIndexKind::gc(&self, is_live: &dyn Fn(NodeId)->bool)`: rebuild the HNSW from nodes where `is_live(id)` (drop the rest + clear them from `deleted`). `LpgStore::gc_vector_indexes(horizon)`: for each vector index, `idx.gc(&|id| !node_deleted_at_or_below(id, horizon))` (a node is live unless its version chain shows `deleted_epoch <= horizon`). Wire into `database/mod.rs gc()` next to `gc_text_indexes` (`#[cfg(feature="vector-index")]`).
- [ ] **Step 3:** PASS; gate. Commit (`feat(vec-mvcc): GC vector index (rebuild, drop deleted-below-horizon)`).

---

## Task 7: Integration, guard removal, acceptance, verification

**Files:** `scan_vector.rs`, `vector_join.rs`, `procedures.rs`, `query/planner/lpg/mod.rs`, `tests/serializable.rs`.

- [ ] **Step 1 (acceptance tests, `tests/serializable.rs`):**
  - `serializable_vector_search_reads_own_writes`; `serializable_vector_search_snapshot_consistent` (committed-after-E excluded, deleted-after-E included);
  - `serializable_vector_reembed_scores_as_of_epoch` (the soundness crux — re-embed across epochs; an earlier snapshot ranks by the as-of-E vector; oracle `read_node_property_visible`);
  - `serializable_vector_search_phantom_aborts` (search + concurrent closer-vector insert → second committer `SerializationFailure`; disjoint → commit) — drive it via **both** `CALL grafeo.search.vector` **and** the vector-scan operator path (the recording-completeness sweep);
  - the previously-guarded queries now plan+run under Serializable.
- [ ] **Step 2:** Thread `(epoch, tx)` into `VectorScanOperator` + `VectorJoinOperator` (mirror `TextScanOperator.with_transaction_context`); under Serializable route `execute_search` through `store.vector_search_visible(.., epoch, tx)` + record `EntityId::Index` **execution-time, once per execution** (mirror the text FilterOperator/scan record). Flip `SearchVectorProcedure.serializable_safe()` → `true` + route through `vector_search_visible`. **Delete** the `plan_vector_scan` Serializable guard (`mod.rs:1078`); flip `test_plan_vector_scan_rejected_under_serializable` → allowed + `test_plan_call_search_vector_rejected_under_serializable`.
- [ ] **Step 3:** Full verification: `--all-features` green; `--features full --test serializable` (vector + text) green; clippy; profiles (`default`/`lpg`/`lpg,vector-index`/`lpg,text-index,vector-index`) + wasm. **Soundness audit:** the recording-completeness sweep over every vector entry point (Scan / Join / CALL / any hybrid `filter_hybrid` vector path) — grep all `vector_search`/`.search(` callers reachable under Serializable, confirm each routes through `vector_search_visible` or is SI/RC-only. **Confirm ZERO Serializable guards remain in the planner** (grep `not yet supported with .* SnapshotIsolation`). `git status` clean; OPSEC. Commit (`feat(mvcc): enable Serializable for vector search — last guard removed`).

---

## Acceptance
- HNSW kNN under Serializable: read-your-writes, snapshot-consistent (committed-after-E excluded, deleted-after-E included), **as-of-E scoring under re-embed** (the crux), recall via widened `ef`; the phantom is caught at every entry point (search + concurrent closer insert aborts; disjoint commits).
- SI/Read-Committed byte-unchanged (retain+filter-at-current = same results; committed accessor; recording Serializable-only).
- **Zero Serializable guards remain in the planner** — the isolation surface is complete.
- Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC. Quantized rescore honours as-of-E.

## Risks (from spec §8)
- **Recall under re-embed/churn** — candidate gen on the latest topology, scoring as-of-E; widen `ef`; documented.
- **Soft-delete connectivity** — Task-0: beam routes through retained-deleted nodes; entry-point on a deleted node still functions.
- **Quantized rescore** — must use the snapshot-aware accessor (its internal `vectors` is latest-only).
- **Recording completeness** — apply the text sweep; a vector path that doesn't record = a phantom hole (the text cycle took 4 rounds; do the sweep up front here).
- **Topology completeness** — every visible-vector node must be a candidate (the `crud.rs` maintenance + retain-on-delete guarantee it).

---

## STATUS: COMPLETE — merged to `integration` @ `853b2fc2`

Snapshot-versioned HNSW vector search under Serializable is **done and merged** (23 files, +3252/−129). **The last Serializable guard is removed — the entire isolation surface is complete.** B‴ shape: the HNSW is a snapshot-filtered candidate generator over the node/property MVCC chains, NOT an index-internal version history.

**Tasks (all landed):**
- **VI1** retain-deleted topology (hard-delete → soft-delete; keep nodes+links for connectivity; `contains_including_deleted`) — behavior-preserving.
- **VI2** `search_visible` (traverse-all, collect-visible, widen `ef` ×4, accessor-scored; quantized rescore via the *passed* accessor).
- **VI3** `SnapshotVectorAccessor` (as-of-E via `read_node_property_visible`, RYW-honouring).
- **VI4** `vector_search_visible` (node-chain visibility + as-of-E accessor + tx-overlay read-your-writes) + trait method + 4 wrapper delegations.
- **VI5** `EntityId::Index` recording (generalized `IndexId::for_index`; read on search, write on indexed-vector SET/REMOVE — reuses the text bridges).
- **VI6** GC (rebuild dropping deleted-below-horizon) + the tiered `node_deleted_at_or_below` fix (`version_history()` delete-epoch, not visibility — don't GC created-after-horizon live nodes).
- **VI7** integration + guard removal (Scan/Join/Procedure routed + recorded execution-time; `plan_vector_scan` guard deleted) + acceptance.
- **VI8** the **re-embed-as-of-E** soundness test + **un-guarded MMR** → all search procedures `serializable_safe=true`.
- A build fix: a concurrent commit (`0abbf55b`) had dropped a feature-conditional `mut` in `filter.rs`, breaking `--all-features`; restored via `cfg_attr`.

**Verification:** `--all-features` 7647/0; serializable **33/33** (full+temporal — reads-own-writes, snapshot-consistent, re-embed-as-of-E, phantom-aborts on CALL+scan, multi-label, MMR); clippy; profiles (`default`/`lpg`/`lpg,vector-index`/`lpg,text-index,vector-index`) + wasm; **effectively zero planner guards** (the one `mutation.rs` per-procedure message is dormant — every procedure passes). **Opus final review: "FULLY and ROBUSTLY sound — safe to merge"** (3 probes: quantized as-of-E rescore, GC-keeps-live-reembed, GC-drops-deleted; no residual hole). The recording-completeness sweep was done up-front (the text lesson) → **one round, no whack-a-mole**.

**Residuals (documented, §8):** candidate-gen recall on re-embedded regions (best-effort, `ef`-tunable); GC = full rebuild; as-of-E *scoring* requires `temporal` (engine-consistent; anti-phantom fires regardless).

**The Serializable arc is finished:** F1 → F2 → Part G → shortestPath → graph algorithms → text index → **vector index**. Zero guards remain.
