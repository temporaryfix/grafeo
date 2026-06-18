# Snapshot-Versioned HNSW Vector Search under Serializable — Design

**Status:** Design (approved in brainstorming 2026-06-18). Next: implementation plan via writing-plans. This removes the **last** Serializable guard.

## 1. Goal

Make HNSW kNN vector search **snapshot-consistent** (kNN over the as-of-epoch *visible* vectors + read-your-writes, recall preserved by widening `ef`) and **SSI-sound** (a coarse `EntityId::Index` predicate-read so a Serializable vector search aborts against a concurrent indexed write — the kNN phantom), then remove the vector-search guard. This is the **second instance** of the MVCC-secondary-index pattern the text cycle established; vector reuses the pattern and adds HNSW-specific filtered traversal.

## 2. Context (current state)

- The vector index is `VectorIndexKind = Hnsw(HnswIndex) | Quantized(QuantizedHnswIndex)` (`crates/grafeo-core/src/index/vector/`), keyed `"label:property"` in `LpgStore::vector_indexes`. `insert(id, vec, accessor)` / `remove(id)` / `search*`.
- **Maintained at the engine `crud.rs` layer**, not the store: node create/delete/set-vector-property call `index.insert`/`index.remove` directly (`crates/grafeo-engine/src/database/crud.rs`), committed-latest. (The text index, by contrast, lived in the store.)
- **`remove()` is a HARD delete** (`hnsw.rs:745`): it removes the node from the graph and re-links neighbors — there is no soft-delete/tombstone today.
- `search_with_filter` is a **naive post-filter**: `search(k.max(allowlist.len()))` then `.filter(allowlist.contains)` — O(N) when the visible set is large (the common case), and structurally cannot do read-your-writes or see deleted-after-snapshot vectors.
- No epoch awareness, no read recording → guarded under Serializable (`plan_vector_scan`, `query/planner/lpg/mod.rs:1078`; `CALL grafeo.search.vector` rejected; `SearchVectorProcedure.serializable_safe()==false`).

## 3. Reused vs new

**Reused from the text cycle (merged `ac1c5420`) — unchanged:** the `EntityId::Index(IndexId)` conflict key + read/write-tracker bridges; the per-tx-delta + commit-promote + GC-at-horizon shape; the single-source discipline (index epoch == property commit epoch); the **execution-time, operator-chokepoint, property-keyed recording** lesson ([[project_serializable_guard_recording]]); the per-procedure `serializable_safe` guard.

**New (HNSW-specific):**
1. **Versioned entries on a hard-delete index** — add `created_epoch`/`deleted_epoch` and convert `remove()` from hard-delete to **soft-delete-with-epoch** (retain the node + its links for connectivity; GC hard-rebuilds below the horizon).
2. **Predicate-filtered traversal** — traverse the full graph for connectivity, collect only visible candidates, widen `ef` for recall.
3. **Relocating index maintenance from the engine `crud.rs` layer into the store** so the single-source invariant holds.

## 4. Decisions (from brainstorming)

- **Completeness bar:** full snapshot set + best-effort recall — the result *set* is the as-of-E visible vectors + read-your-writes; ANN ranking stays approximate (inherent); `ef` widened to compensate for filtered-out invisibles.
- **HNSW approach:** predicate-filtered traversal (not per-epoch snapshots — memory-prohibitive; not the naive post-filter — O(N) + incomplete).
- **Single-source locus:** move vector-index maintenance into the **store** (consistent with text; clean single-source), accepting the refactor.
- **Recording:** coarse `EntityId::Index` (index-level is the natural grain for kNN — there's no term-level analog), execution-time at the operator chokepoint.

## 5. Architecture

### 5a. Versioned HNSW entries (hard-delete → soft-delete + epochs)

Each entry gains `created_epoch: EpochId`, `created_by: Option<TransactionId>`, `deleted_epoch: Option<EpochId>`, `deleted_by: Option<TransactionId>`. Visibility-at-(E,tx) is identical to `posting_visible`/`VersionInfo::is_visible_to`. **`remove(id, epoch, tx)` becomes a soft-delete:** stamp `deleted_epoch`, **keep the node and its bidirectional links** (so it remains a routing hop for connectivity and is still visible to snapshots `< epoch`). `insert(id, vec, epoch, tx, accessor)` stamps `created_epoch`. (Applies to both `HnswIndex` and `QuantizedHnswIndex`.) **Task 0 verification:** confirm the beam-search (`search_layer`) can route *through* a soft-deleted node while excluding it from results.

### 5b. Predicate-filtered search

`search_visible(query, k, ef, is_visible: impl Fn(NodeId) -> bool) -> Vec<(NodeId, f32)>`: run the existing beam search, but at **result collection** skip any candidate where `!is_visible(id)` — **while still traversing through it** (its neighbors are explored) so connectivity is preserved. Widen the effective `ef` (e.g. `ef.max(k * F)` for a small factor F, tunable) so the result heap fills with `k` visible despite skipped invisibles. `is_visible(id)` = entry-visible-at-(E,tx) AND not tx-removed.

### 5c. Per-transaction vector delta

The transaction's uncommitted vector inserts/removes for the index (`Vec<(NodeId, Vec<f32>)>` + a removed set). Searched by **brute force** (k is small, the uncommitted set is small): compute distances to the query, merge into the top-k from 5b; a delta-removed node is excluded. Read-your-writes. Mirrors `TextIndexDelta`; lives in a `vector_index_overlay` next to `text_index_overlay`.

### 5d. Single-source — store-maintained, stamped at the property commit

Relocate the `crud.rs` vector insert/remove into the store as `update_vector_index_on_set`/`_on_remove` (mirroring `update_text_index_on_set`). Under a transaction they buffer into `vector_index_overlay[tx]`; the non-transactional/auto-commit path mutates the committed index directly at `current_epoch()`. **Commit-promote** in `apply_tx_overlay` stamps each promoted entry's epoch = the commit epoch `C` — **the same `C` that finalizes the vector property version** (single-source). The store supplies a `VectorAccessor` over its node-properties for HNSW neighbor-distance lookups during insert. Rollback/abort drops `vector_index_overlay[tx]`.

> **INVARIANT (acceptance):** for every node `N` and epoch `E`, an entry for `N` is visible in the index at E iff `N`'s indexed vector property is visible (and equals that value) at E — the index is a consistent-by-construction projection of the vector property's version chain. One commit stamps both.

### 5e. GC

Soft-deleted entries with `deleted_epoch <= min_active_epoch` are reclaimed by an **HNSW rebuild** (the graph is rebuilt from the live entries) — HNSW has no cheap in-place hard delete. Triggered from the same horizon-driven GC path as the store/text GC; bounded by the active-tx horizon.

### 5f. SSI recording (coarse `EntityId::Index`)

`IndexId::for_text_index` generalizes to any `(label, property)` index (rename/alias to `for_index`). A Serializable vector search records `read(EntityId::Index(idx))` **execution-time at the operator chokepoint** — `VectorScanOperator`, `VectorJoinOperator`, and `SearchVectorProcedure` (record once, regardless of rows, via the established once-per-execution mechanism). A vector-property write (the store's `update_vector_index_on_*`) records `write(EntityId::Index(idx))`. The kNN phantom — a concurrent insert of a vector closer than the current k-th — is caught (it wrote the index; the search read it). Coarse (any indexed-vector write conflicts); no term-level analog.

## 6. Integration + guard removal

- Route `VectorScanOperator`/`VectorJoinOperator` + `SearchVectorProcedure` through `search_visible(epoch, tx)` (+ the per-tx delta + the `EntityId::Index` read) under Serializable; committed-latest `search` under SI/RC (byte-unchanged).
- Flip `SearchVectorProcedure.serializable_safe()` → `true`; drop the `plan_vector_scan` guard (`mod.rs:1078`); update the guard-assertion tests (`test_plan_vector_scan_rejected_under_serializable` → allowed; `test_plan_call_search_vector_rejected_under_serializable`). **No guard remains** after this.

## 7. Acceptance criteria

1. **Completeness:** under Serializable, vector search (a) returns the transaction's own uncommitted matching vectors (read-your-writes); (b) excludes vectors committed after the snapshot; (c) includes vectors whose node was deleted after the snapshot; (d) preserves recall via widened `ef` (a recall test: with a fraction of entries soft-deleted-but-visible, recall stays within tolerance of the unfiltered baseline).
2. **Single-source invariant (§5d) — tested:** mutate a node's vector across epochs (insert / re-embed / delete-node) and assert the index's as-of-E entry set/values equal the vector property's as-of-E version chain.
3. **SSI soundness (anti-phantom):** a Serializable vector search concurrent with a transaction inserting a closer vector → a serialization failure; a disjoint write (different indexed property) → both commit. Recorded at every vector entry point (Scan / Join / CALL) — apply the text recording-completeness sweep.
4. **SI / Read-Committed unchanged:** committed-latest search + results identical to today; versioning additive; delta + recording Serializable-only.
5. **Guard removed:** `CALL grafeo.search.vector` + vector scan/join plan & run under Serializable. **Zero Serializable guards remain in the planner.**
6. Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC (un-pushed).

## 8. Residuals / risks

- **HNSW soft-delete + connectivity (the #1 risk):** routing through soft-deleted nodes while excluding them from results must be correct in the beam search; a re-link bug or an entry-point on a deleted node degrades recall or correctness. Mitigation: Task-0 verification + the recall test.
- **GC = full rebuild** — O(index) cost when entries fall below the horizon; amortized/triggered like the store GC. Acceptable; note it.
- **Recall under heavy churn** — many invisible entries widen the effective search; `ef` factor is tunable; documented.
- **Single-source relocation** — moving maintenance from `crud.rs` to the store touches the create/delete/set-vector paths; the existing vector tests are the regression guard; stamp at the same `C` as the property (the text commit-flow trace is the template).
- **Quantized vs full HNSW** — both variants need the versioned-entry + filtered-search treatment; the quantized path stores vectors internally (no accessor) — verify the soft-delete + filter apply there too.
- **`EntityId::Index` already exists** (text) — vector reuses it; only the recording call-sites (vector operators) are new.
