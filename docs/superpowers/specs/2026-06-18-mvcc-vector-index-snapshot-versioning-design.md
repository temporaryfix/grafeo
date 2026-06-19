# Snapshot-Versioned HNSW Vector Search under Serializable — Design

**Status:** Design (approved in brainstorming 2026-06-18, revised to B‴ after grounding 2026-06-19). Next: implementation plan. This removes the **last** Serializable guard.

## 1. Goal

Make HNSW kNN vector search **snapshot-consistent** (kNN over the as-of-epoch *visible* nodes, scored with their as-of-epoch vector values, + read-your-writes, recall preserved by widening `ef`) and **SSI-sound** (a coarse `EntityId::Index` predicate-read so a Serializable vector search aborts against a concurrent indexed write — the kNN phantom), then remove the vector-search guard. Reuses the text cycle's `EntityId::Index` recording + per-tx delta + GC-at-horizon, and the node/property MVCC the store already has.

## 2. Context (current state)

- `VectorIndexKind = Hnsw(HnswIndex) | Quantized(QuantizedHnswIndex)` (`crates/grafeo-core/src/index/vector/`), keyed `"label:property"` in `LpgStore::vector_indexes`.
- **The full-precision `HnswIndex` stores NO vectors** — `HnswNode { neighbors: Vec<Vec<NodeId>> }` is *topology only* (`hnsw.rs:129`); `insert(id, vec, accessor)` builds topology and uses the `VectorAccessor` for distances; vectors live in node-properties. `VectorAccessor::get_vector(id) -> Option<Arc<[f32]>>` (`accessor.rs:33`). Full-precision (`VectorIndexKind::Hnsw`) is the **default**.
- **`QuantizedHnswIndex` DOES store vectors internally** (`vectors: HashMap<NodeId, Arc<[f32]>>` for rescoring + `scalar_vectors`/`binary_vectors`) — coarse quantized search then full-precision **rescore**.
- **`remove()` is a HARD delete** (`hnsw.rs:745`) — removes the node + re-links neighbors; no soft-delete today.
- Maintained at the engine `crud.rs` layer (committed-latest, `index.insert`/`remove` on node create/delete/set-vector). `search_with_filter` is a naive O(N) post-filter. No epoch awareness, no recording → guarded (`plan_vector_scan`, `query/planner/lpg/mod.rs:1078`; `CALL grafeo.search.vector` rejected; `SearchVectorProcedure.serializable_safe()==false`).

## 3. The load-bearing insight (why B‴, not text's index-internal versioning)

Text versioned its postings because **text stores the indexed data in the index** (the postings), and a node's term-membership changes. **Vector does not store the indexed data in the (full-precision) index** — the vectors are in the node-property MVCC version chain, *already snapshot-versioned*. So the snapshot-correct vector value is `read_node_property_visible(N, prop, E, tx)` — **existing infra** — not something to re-version inside the HNSW.

Therefore the index is a **snapshot-filtered candidate generator over the node/property version chains**, not a parallel version history:
- **Visibility** (which nodes) comes from the node version chain — `is_node_visible_versioned(N, E, tx)` (the SnapshotView predicate the graph-algorithm cycle already uses).
- **Vector values** (for distance/score) come from the property version chain — a **snapshot-aware `VectorAccessor`** reading `read_node_property_visible`. This is sound under **re-embedding**: a node re-embedded `V0→V1` is scored with its as-of-E value (`V0` for a snapshot before the change), never a future value. (The unsound shortcut — score with the index's/committed-latest value — would read the future and void SSI's SI-read foundation.)
- **The index itself carries no per-entry epochs and no value copies.** Its only MVCC change is **retaining deleted nodes** (soft-delete instead of hard-delete) so connectivity holds and deleted-after-snapshot nodes resolve; GC prunes them below the horizon.

**Rejected:** (A) index-internal entry-value-versioning — for full-precision there is nothing to version (it stores no vectors); doing it means *adding* a versioned vector store that duplicates the property chain. The earlier draft chose A on the wrong assumption that the index stores vectors; the grounding (topology-only) reverses it.

## 4. Decisions (from brainstorming)

- **Completeness bar:** full snapshot set + best-effort recall — result *set* is the as-of-E visible nodes; scoring uses as-of-E vectors (sound); ANN ranking inherently approximate; `ef` widened to compensate for filtered-out invisibles.
- **HNSW approach:** predicate-filtered traversal (not per-epoch snapshots; not naive post-filter).
- **Visibility + values from the node/property version chains** (B‴), not an index-internal history.
- **Recording:** coarse `EntityId::Index` (index-level is the natural kNN grain), execution-time at the operator chokepoint.

## 5. Architecture

### 5a. Retain-deleted topology (the only HNSW structural change)

`remove(id)` becomes a **soft-delete**: keep the node and its bidirectional links (so it remains a routing hop and is still reachable for snapshots that should see it) instead of hard-removing + re-linking. Re-embedding (`SET n.vec = …`) re-inserts `id`, overwriting its topology position toward the new vector (committed-latest topology). Applies to `HnswIndex` and `QuantizedHnswIndex`. No per-entry epochs. **Task 0 verification:** confirm the beam search routes *through* a retained node while the filter (5b) excludes it from results; confirm re-insert of an existing id updates rather than duplicates.

### 5b. Predicate-filtered, as-of-E-scored search

`search_visible(query, k, ef, is_visible: impl Fn(NodeId)->bool, accessor: &impl VectorAccessor) -> Vec<(NodeId, f32)>`: run the beam search over the (latest) topology — traversing *through* all nodes for connectivity — but **collect into results only** candidates where `is_visible(id)`; widen the effective `ef` (e.g. `ef.max(k * F)`, F tunable) to keep the result heap filled despite skipped invisibles. **Distances/scores use the supplied accessor**, which the caller makes snapshot-aware (5c). For `Quantized`, the coarse phase uses internal codes (latest) for candidate gen, and the **rescore** phase uses the snapshot-aware accessor for as-of-E full-precision scores.

### 5c. Snapshot-aware accessor (as-of-E values, the soundness mechanism)

A `SnapshotVectorAccessor { store, property, epoch, tx }` impl of `VectorAccessor`: `get_vector(id)` = `read_node_property_visible(id, property, epoch, tx)` converted to `&[f32]` (the as-of-E committed vector, or the tx's own uncommitted value via the overlay path `read_node_property_visible` already honours). Under SI/RC the operator passes the existing committed-latest accessor (byte-unchanged). Under Serializable it passes the snapshot-aware one.

### 5d. Per-transaction read-your-writes

The transaction's uncommitted vectors come from the existing `tx_property_overlay` (its buffered `SET n.vec`). Two parts: (i) a node the tx **re-embedded/created** is scored with the tx's value — already handled because `read_node_property_visible(.., tx)` returns the tx's buffered value; (ii) a node the tx **created** that isn't in the committed topology yet is found by **brute-forcing the tx's buffered vectors for this index** against the query and merging into top-k; a node the tx **deleted** is filtered by `is_visible` (= `is_node_visible_versioned(.., tx)` returns not-visible). No separate `vector_index_overlay` — derive from `tx_property_overlay`.

### 5e. GC

Soft-deleted nodes whose node version chain shows `deleted_epoch <= min_active_epoch` are reclaimed by an **HNSW rebuild** from the still-live nodes (HNSW has no cheap in-place hard delete). Triggered from the same horizon-driven GC path as the store/text GC.

### 5f. SSI recording (coarse `EntityId::Index`)

`IndexId::for_text_index` generalizes to `IndexId::for_index(label, property)` (text keeps working). A Serializable vector search records `read(EntityId::Index(idx))` **execution-time at the operator chokepoint** — `VectorScanOperator`, `VectorJoinOperator`, and `SearchVectorProcedure` (record once per execution, regardless of rows, via the established mechanism). A vector-property write (the `crud.rs`/store path that does `index.insert`/`remove`) records `write(EntityId::Index(idx))`. The kNN phantom (a concurrent insert of a closer vector) is caught. Apply the **text recording-completeness sweep** ([[project_serializable_guard_recording]]) across every vector entry point.

## 6. Integration + guard removal

- Route `VectorScanOperator`/`VectorJoinOperator` + `SearchVectorProcedure` through `search_visible` with the snapshot-aware accessor + the `is_node_visible_versioned` predicate + the tx-overlay brute-force + the `EntityId::Index` read, under Serializable; committed-latest path under SI/RC (byte-unchanged).
- Flip `SearchVectorProcedure.serializable_safe()` → `true`; drop the `plan_vector_scan` guard (`mod.rs:1078`); update the guard-assertion tests. **No Serializable guard remains in the planner after this.**

## 7. Acceptance criteria

1. **Completeness:** under Serializable, vector search (a) returns the transaction's own uncommitted vectors (read-your-writes); (b) excludes nodes committed after the snapshot; (c) includes nodes deleted after the snapshot; (d) preserves recall via widened `ef` (recall within tolerance of the unfiltered baseline with a fraction of nodes soft-deleted-but-visible).
2. **As-of-E scoring under re-embed — tested (the soundness crux):** re-embed a node's vector across epochs; a Serializable search at an earlier snapshot ranks it by its **as-of-E** vector (via the snapshot-aware accessor), never the latest. Oracle = `read_node_property_visible`.
3. **SSI soundness (anti-phantom):** a Serializable vector search concurrent with a transaction inserting a closer vector → serialization failure; a disjoint write → both commit. Recorded at every entry point (Scan / Join / CALL) — apply the recording-completeness sweep.
4. **SI / Read-Committed unchanged:** committed-latest search results identical to today (retain + filter-at-current produce the same set; the committed accessor is used; recording Serializable-only).
5. **Guard removed:** `CALL grafeo.search.vector` + vector scan/join plan & run under Serializable. **Zero Serializable guards remain.**
6. Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC (un-pushed).

## 8. Residuals / risks

- **Recall under re-embed / heavy churn (the #1 caveat):** candidate generation runs on the *latest* topology while scoring as-of-E, so re-embedded regions (and many soft-deleted nodes) can lower recall; `ef` factor is tunable; documented, accepted under "best-effort recall."
- **HNSW soft-delete + connectivity:** routing through retained-deleted nodes while excluding them from results must be correct; an entry-point landing on a logically-deleted node must still function. Task-0 verification + the recall test.
- **GC = full rebuild** — O(index) when nodes fall below the horizon; amortized/triggered like the store GC.
- **Accessor cost:** `read_node_property_visible` per scored candidate (a version-chain lookup) — negligible beside the distance computations, but real; the committed-latest SI/RC path keeps the cheap accessor.
- **Quantized rescore path** must use the snapshot-aware accessor for the full-precision rescore (its internal `vectors` map is latest-only); verify the coarse→rescore split honours as-of-E.
- **Topology completeness invariant:** every node with a committed (visible) vector must be in the topology (until GC) or it's a false-negative candidate — the existing `crud.rs` maintenance guarantees this; retain-on-delete preserves it.
