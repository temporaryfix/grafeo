# Snapshot-Versioned BM25 Text Index under Serializable — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BM25 text search snapshot-consistent (exact as-of-epoch incl. read-your-writes) and SSI-sound under Serializable, then remove its guard — by epoch-versioning the inverted index (MVCC delta-over-base) and recording a coarse index-level predicate read.

**Architecture:** Version the committed inverted index (epoch-stamped postings + as-of-E aggregate stats) retained until GC; buffer a transaction's uncommitted text changes in a per-tx index delta; `search_visible` merges committed-visible-at-(E,tx) ⊕ delta with as-of-E `avgdl`; commit promotes the delta into the base **stamping the same epoch that finalizes the property** (single-source invariant); a new `EntityId::Index` conflict-key records the search's predicate read and the indexed-write so the existing rw-detection catches phantoms. Built as the reusable secondary-index pattern HNSW will inherit.

**Tech Stack:** Rust, `cargo test`. `grafeo-core` (`index/text/inverted_index.rs`, `graph/lpg/store/`), `grafeo-engine` (`transaction/`, `procedures.rs`, the text-search operator/executor). `CARGO_INCREMENTAL=0`.

**Spec:** `docs/superpowers/specs/2026-06-18-mvcc-text-index-snapshot-versioning-design.md`. Read it first.

---

## Orientation & reuse templates

On `integration` @ `d1a4998b` (the spec commit). This increment heavily **reuses recently-merged patterns** — study these as concrete templates:
- **`EntityId::Index` threading** mirrors the Part-G `PropTag` generalization (commits `5c453745` + the conflict-key plumbing): adding a variant/field through `read_set`/`write_set`/`read_registry`/`retired_readers`. `EntityId` is in `transaction/manager.rs` (~12 `EntityId::Node|Edge` sites + `read_registry.rs` + the bridges).
- **Per-tx delta + commit-promote + rollback** mirrors `tx_property_overlay: Map<TxId, TxDelta>` (`store/mod.rs:155,528`) with `apply_tx_overlay`/`drop_tx_overlay` (commit/rollback) and `finalize_*` (epoch stamping). The text-index delta is the same shape.
- **GC-at-horizon** mirrors the store's `min_active_epoch`-driven compaction.
- **Snapshot-recording read path** mirrors the store visible-read chokepoints (`record_read_*`) + the read-tracker bridge.
- **Guard removal + per-procedure policy** mirrors the just-merged graph-algorithm work (`procedures.rs` `serializable_safe()`, commit `d3c3a43a`).

**Current text index** (`index/text/inverted_index.rs`): `Posting{node_id, term_freq}`, `PostingList{postings: Vec<Posting>}`, `InvertedIndex{postings: HashMap<String,PostingList>, doc_lengths: HashMap<NodeId,u32>, total_length: u64, tokenizer, config}`. BM25 via `bm25_term_score(df, tf, dl, n, avg_dl)`; `search(query,k)`/`score_document`/`search_with_threshold`. Store holds `text_indexes: Map<"label:prop", RwLock<InvertedIndex>>` (`store/mod.rs:469`); `update_text_index_on_set`/`_on_remove` (`store/index.rs:232`) mutate it on the committed property write.

**Verification gate (each task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full --test serializable` + `--test text_index_mutations`. Clippy `--all-features … -D warnings`; profiles + wasm. Hygiene: `rustfmt --edition 2024` changed files only (NOT `cargo fmt`); `git status` clean (untracked `ce/` — never add); `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `index/text/versioned.rs` (new) | `VersionedPosting`, visibility, the versioned `InvertedIndex` internals, `search_visible`, as-of-E stats, GC | Create (or grow `inverted_index.rs` — keep the file focused; new file preferred) |
| `index/text/inverted_index.rs` | keep `search`/`score_document` as the committed-latest path (delegating to versioned with E=latest) | Modify |
| `index/text/tx_delta.rs` (new) | per-tx text-index delta (buffered inserts/removes) | Create |
| `graph/lpg/store/index.rs` + `mod.rs` | route `update_text_index_*` into the tx delta; commit-promote + rollback + GC hooks; `search_text_visible` | Modify |
| `graph/lpg/store/graph_store_impl.rs` | `text_search` routes through `search_text_visible(epoch, tx)` | Modify |
| `transaction/manager.rs` + `read_registry.rs` | `EntityId::Index(IndexId)` variant + threading | Modify |
| `transaction/read_tracker.rs` / `write_tracker.rs` (or store recording) | record `Index` read on search / `Index` write on indexed SET | Modify |
| `procedures.rs` + the text-search planner/executor | route through versioned search + record; `serializable_safe=true`; drop guard | Modify |
| `tests/serializable.rs`, `tests/text_index_mutations.rs` | acceptance incl. the single-source invariant test | Modify |

---

## Task 1: Epoch-versioned postings + visibility (committed index)

**Files:** Create `index/text/versioned.rs`; modify `inverted_index.rs`.

- [ ] **Step 1 (failing test):** in `versioned.rs`, a `VersionedPosting{node_id, term_freq, created_epoch: EpochId, deleted_epoch: EpochId}` and `posting_visible(p, epoch, tx) -> bool` mirroring `VersionInfo::is_visible_to` (visible iff `created<=epoch || created==PENDING&&created_by==tx`-style; not deleted-at-epoch / own-delete). Test the matrix: created-after-E excluded; deleted-after-E included; own-PENDING-create visible to tx; own-PENDING-delete hidden to tx. Use `EpochId::PENDING` (`u64::MAX`) as the uncommitted sentinel (same as the store + the LayeredStore base-delete fix). FAIL.
- [ ] **Step 2:** Implement `VersionedPosting` + `posting_visible`. (Model the boundary `deleted_epoch.as_u64() <= epoch.as_u64()` on the LayeredStore `is_edge_deleted_from_base_at` fix — consistent with the overlay.) Track `created_by`/`deleted_by: Option<TransactionId>` on the posting for own-tx visibility (mirror `BaseEdgeDelete`).
- [ ] **Step 3:** Change `InvertedIndex.postings` value to `Vec<VersionedPosting>`; `insert(id, text, epoch, tx)` stamps `created_epoch` (PENDING for uncommitted, real for committed-direct); `remove(id, epoch, tx)` stamps `deleted_epoch` on the live posting (retain it). Keep `doc_lengths` keyed by id but version it in Task 2. The legacy `insert(id, text)`/`remove(id)` become thin wrappers passing a "committed-now" epoch + `None` tx (preserves the current direct-mutation call sites until they're rerouted in Task 3).
- [ ] **Step 4:** Run → PASS. Full gate (existing `search`/`text_index_mutations` still green — the committed-latest path is unchanged because all current postings are visible at latest). Commit (`feat(text-mvcc): epoch-versioned postings + visibility`).

---

## Task 2: As-of-E aggregate stats (BM25 `avgdl@E`)

**Files:** `versioned.rs`.

- [ ] **Step 1 (test):** insert docs at epochs E1<E2; `doc_count_at(E1)` / `total_length_at(E1)` reflect only docs visible at E1; `avgdl_at(E1) == total_length_at(E1)/doc_count_at(E1)`. A doc deleted after E1 still counts at E1. FAIL.
- [ ] **Step 2:** Version `doc_lengths` as `Vec<(NodeId, u32, created_epoch, deleted_epoch)>` (or reuse the posting-visibility helper over a per-doc record). Maintain a small **epoch-stamped aggregate log** of `(epoch, delta_total_length, delta_doc_count)` so `total_length_at(E)`/`doc_count_at(E)` are a prefix-sum ≤ E (O(log) with a sorted vec, or O(commits-since) — bounded by the active horizon). `avgdl_at(E)` from those.
- [ ] **Step 3:** PASS; gate. Commit (`feat(text-mvcc): as-of-epoch aggregate stats for avgdl`).

---

## Task 3: Per-transaction text-index delta + reroute the write path

**Files:** Create `index/text/tx_delta.rs`; modify `store/index.rs` + `store/mod.rs`.

- [ ] **Step 1 (test):** a `TextIndexDelta` holding a tx's buffered `(index_key, node, Option<text>)` changes; `delta.search_terms(index_key, terms)` returns the tx's own matching postings. A store-level test: under a tx, `SET n.title='...'` (indexed) buffers into the delta and does NOT mutate the committed index. FAIL.
- [ ] **Step 2:** Add `TextIndexDelta` (per-tx, keyed by `index_key`, holding buffered inserts/removes tokenized at `PENDING`). Add `text_index_overlay: RwLock<FxHashMap<TransactionId, TextIndexDelta>>` to the store (mirror `tx_property_overlay`). Reroute `update_text_index_on_set`/`_on_remove` so that **when called under a transaction** they buffer into `text_index_overlay[tx]` instead of mutating the committed index; the non-transactional (auto-commit/SYSTEM) path keeps the direct committed mutation (at `current_epoch`). (This is the single-source wiring start: the index change is now bound to the tx and will be stamped at the tx's commit in Task 5.)
- [ ] **Step 3:** PASS; gate (committed-latest auto-commit path unchanged). Commit (`feat(text-mvcc): per-tx text-index delta + buffered write path`).

---

## Task 4: `search_visible` — snapshot search merging committed + delta

**Files:** `versioned.rs`, `store/index.rs`, `graph_store_impl.rs`.

- [ ] **Step 1 (test):** `InvertedIndex::search_visible(query, k, epoch, tx, delta) -> Vec<(NodeId,f64)>`: returns committed postings visible-at-(epoch,tx) ⊕ the tx delta's matching docs, BM25-scored with `avgdl_at(epoch)` (delta docs included in the as-of view). Tests: (a) a doc committed after E is excluded; (b) a doc deleted after E is included; (c) the tx's own uncommitted insert is returned (read-your-writes); (d) scores use `avgdl@E`. FAIL.
- [ ] **Step 2:** Implement `search_visible` (filter posting lists by `posting_visible`, union the delta postings, score with `df`/`n`/`avgdl` taken as-of-E incl. delta). Add `LpgStore::search_text_visible(index_key, query, k, epoch, tx)` that looks up the committed `InvertedIndex` + the tx's `text_index_overlay` delta and calls `search_visible`. Route `GraphStoreSearch::text_search` (`graph_store_impl.rs:392`) to `search_text_visible` when an `(epoch, tx)` snapshot is in scope; the no-tx path uses the committed-latest `search`.
- [ ] **Step 3:** PASS; gate. Commit (`feat(text-mvcc): search_visible snapshot search (committed + delta)`).

---

## Task 5: Commit-promote + rollback (the single-source invariant)

**Files:** `store/index.rs`, `store/mod.rs` (the commit/rollback hooks), `versioned.rs`.

- [ ] **Step 1 (the headline invariant test):** the single-source acceptance (spec §9.2): a tx SETs `n.title` across epochs (insert term, change term, remove); for each committed epoch E, assert `search_text_visible(...)` term-membership for `n` equals `tokenize(the committed title as-of E)` — i.e. the index's as-of-E view is exactly the property's as-of-E tokenization. Plus: after commit, the promoted postings carry `created/deleted_epoch == the commit epoch`. FAIL.
- [ ] **Step 2:** In the store's commit path (where `apply_tx_overlay` finalizes the property delta — find it, it's the same place `tx_property_overlay` promotes), **also promote `text_index_overlay[tx]` into the committed `InvertedIndex`(es), stamping `created_epoch`/`deleted_epoch = the commit epoch C`** (the SAME C the property finalize uses — pass it through). In rollback/`drop_tx_overlay`, drop `text_index_overlay[tx]`. This binds the index epoch to the property commit: one C stamps both ⇒ the index is a consistent-by-construction projection.
- [ ] **Step 3:** PASS — especially the invariant test. Full gate. Commit (`feat(text-mvcc): commit-promote text-index delta at the property commit epoch (single-source)`).

---

## Task 6: GC

**Files:** `versioned.rs`, `store/index.rs`.

- [ ] **Step 1 (test):** `InvertedIndex::gc(horizon)` drops postings with `deleted_epoch <= horizon` (and prunes empty lists + stale aggregate-log entries); a posting deleted at E is retained while `horizon < E`, dropped once `horizon >= E`. FAIL.
- [ ] **Step 2:** Implement `gc(horizon)`; call it from the store's existing GC/recompact path against `min_active_epoch` (mirror how the store GCs version chains). No behavior change for live/visible postings.
- [ ] **Step 3:** PASS; gate. Commit (`feat(text-mvcc): GC versioned postings below the active horizon`).

---

## Task 7: `EntityId::Index` + SSI predicate recording

**Files:** `transaction/manager.rs`, `read_registry.rs`, the read/write bridges, `store/index.rs`.

- [ ] **Step 1 (test):** manager-level — `record_read(tx, EntityId::Index(idx), None)` and a concurrent `record_write(tx2, EntityId::Index(idx), None)` form an rw-edge (reuse the existing edge-direction tests). A `(label,prop)` maps to a stable `IndexId`. FAIL.
- [ ] **Step 2:** Add `Index(IndexId)` to `EntityId` (where `IndexId` is a cheap stable id for `(label, property)` — e.g. a `u64` hash like `prop_tag`, or an interned id). **Thread it through exactly like Part-G `PropTag`/the existing `EntityId` variants** (read_set/write_set tuples, registry, retired_readers, the W-W + rw checks) — the variant is additive; Node/Edge paths unchanged. Most `match EntityId` sites just need the new arm (often `Index` behaves like a Node for keying).
- [ ] **Step 3:** Record sites: a Serializable `search_text_visible` records `read(tx, EntityId::Index(idx))` (via the read-tracker, like the store chokepoints); `update_text_index_on_set/_on_remove` under a tx records `write(tx, EntityId::Index(idx))` (in addition to the node write). `idx = IndexId::of(label, prop)`.
- [ ] **Step 4:** Run → PASS; full gate (additive — no SI/RC change). Commit (`feat(text-mvcc): EntityId::Index predicate-read recording (anti-phantom)`).

---

## Task 8: Integration, guard removal, acceptance, verification

**Files:** `procedures.rs`, the text-search planner/executor, `tests/serializable.rs`, `tests/text_index_mutations.rs`.

- [ ] **Step 1 (acceptance tests):**
  - `serializable_text_search_reads_own_writes`: a Serializable tx inserts a matching doc, then `CALL text_search` sees it.
  - `serializable_text_search_snapshot_consistent`: a doc committed after the tx's snapshot is not returned; one deleted after is.
  - `serializable_text_search_phantom_aborts`: a Serializable text search + a concurrent tx inserting a matching doc → second committer `SerializationFailure` (anti-phantom via `EntityId::Index`); a disjoint-index write → both commit.
  - The single-source invariant test from Task 5 lives in `text_index_mutations.rs`.
- [ ] **Step 2:** Flip `SearchTextProcedure` (+ MMR/hybrid text variants) `serializable_safe()` → `true` (mirror `d3c3a43a`); route the text-search operator/procedure through `search_text_visible(epoch, tx)` + the `Index` read recording; drop the text guard arm (`planner/lpg/mod.rs:1003`). Update the guard-assertion test (text now plans under Serializable; vector still rejects).
- [ ] **Step 3:** Full verification: `--all-features -p grafeo-core -p grafeo-engine` green; `--features full --test serializable` + `--test text_index_mutations` green; clippy `--all-features … -D warnings` clean; profiles (`default`/`lpg`/`lpg,temporal` + `text-index`; `grafeo-core` tiered) + `grafeo-wasm`. **Soundness audit:** the single-source invariant holds (the index = property projection); the phantom is caught; SI/RC behavior byte-unchanged; vector still guarded. `git status` clean; OPSEC. Commit (`feat(mvcc): enable Serializable for text search — guard removed (text-mvcc)`).

---

## Acceptance
- BM25 text search under Serializable: read-your-writes, snapshot-consistent (committed-after-E excluded, deleted-after-E included), exact `avgdl@E`; the phantom is caught (search + concurrent matching insert aborts; disjoint commits).
- **Single-source invariant holds + is tested:** the index's as-of-E term-membership equals the property's as-of-E tokenization (one commit stamps both).
- The versioned-index / tx-delta / commit-promote / GC / `EntityId::Index` machinery is factored so the HNSW (vector) cycle inherits it (a short "what vector reuses" note).
- SI/Read-Committed text search byte-unchanged. Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC. Vector still guarded.

## Risks (from the spec §11)
- **Single-source drift (#1):** any `update_text_index_*` path that stamps a different epoch than the property commit, or a tokenization mismatch, makes the index lie. Mitigation: one C stamps both (Task 5); the invariant test (Task 5/8); audit every index-update call site routes through the delta + same-commit promote.
- **`EntityId::Index` blast radius:** additive variant through the SSI machinery; gate on the arm; mirror the Part-G discipline; the manager edge-direction + write-skew tests are the regression anchor.
- **Predicate-read coarseness:** a Serializable text search over-conflicts (any indexed write). Documented; term-level is the follow-on.
- **`avgdl@E` cost / correctness:** the aggregate log must be exact as-of-E; test against re-derivation; fall back to latest-`avgdl` (doc set stays exact) only if cost dominates.
- **Behavior preservation:** Tasks 1-6 keep the committed-latest auto-commit path unchanged (versioning additive; delta/recording Serializable-only) — the existing text-index suite is the guard at every task.

---

## STATUS: COMPLETE — merged to `integration` @ `ac1c5420`

Snapshot-versioned BM25 text search under Serializable is **done and merged** (29 files, +5224/−164). The committed-latest index became an MVCC delta-over-base structure: epoch-versioned postings + as-of-E aggregate stats (`avgdl@E`) + per-tx delta + `search_visible` merge + commit-promote (single-source) + GC; a coarse `EntityId::Index` predicate-read makes a Serializable text search abort against a concurrent indexed write (anti-phantom); guard removed; **vector still guarded** (its cycle).

**Tasks (all landed):**
- **TI1-2** versioned postings + visibility (`is_visible_to` mirror) + as-of-E aggregate log — behavior-preserving foundation.
- **TI3-4** per-tx `text_index_overlay` (mirrors `tx_property_overlay`) + `search_visible` (committed-visible ⊕ delta, read-your-writes).
- **TI5** commit-promote — **single-source invariant** (index posting epoch == property version epoch, stamped at the same commit; opus-reviewed, invariant test with the property version chain as oracle). Follow-up: node-delete path (`remove_from_all_text_indexes`) stamped epoch-0 → fixed to `current_epoch()` (review-found).
- **TI6** GC postings + agg-log below `min_active_epoch`.
- **TI7** `EntityId::Index(IndexId)` + the read/write-tracker bridges + a new store-level WriteTracker registration (opus-reviewed).
- **TI8** integration + guard removal + acceptance (`text_search_visible`; per-procedure `serializable_safe`).

**The anti-phantom find-and-fix tail (4 review-found phantom paths — the recording had to reach every entry point):**
- **TI9** index-pushdown operator path (top-k + threshold) + the 4 wrapper-store delegations (`active_store()` is a `WalGraphStore`).
- **TI10** per-row filter path (`text_match`/`text_score` when pushdown declined) via `score_text_visible`.
- **TI11** moved recording to **execution-time** (FilterOperator first-poll, robust to plan caching) + **complete predicate walk** (`coalesce`/`CASE`/comprehensions/…); retired the fragile plan-time side-effect.
- **TI12** **property-driven** recording (every text index on the predicate's property, not just the static scan label) — closes the multi-label 0-row corner. Enumerator hardened to exact `":{property}"` suffix-match (final-review finding, `ac1c5420`).

**Verification:** `--all-features` 7610/0; serializable suite **27/27** (CALL / operator / per-row / nested / 0-row / multi-label phantom-aborts + disjoint-commits + read-your-writes + snapshot-consistent + single-source + node-delete); clippy; profiles (`default`/`lpg`/`lpg,temporal`/`lpg,text-index`) + wasm; SI/RC byte-unchanged; OPSEC (un-pushed). Read/write `IndexId` keys symmetric; tx-context uniform across nested/subquery/UNION filters (single `plan_filter` chokepoint).

**Documented residuals (sound, not holes):** coarse predicate recording over-conflicts (term-level is the follow-on); projection-only `RETURN text_score` over 0 rows is unrecorded but benign (row set governed by MATCH, not the index); the synthetic-node (`NodeId::INVALID`) recording trigger relies on `record_read_index` running first (it does, in all impls) — a noted latent coupling; concurrent CREATE TEXT INDEX (DDL) not modelled.

**Next:** term-level predicate recording (refinement) + the **vector/HNSW cycle** (same MVCC-secondary-index pattern + a versioned HNSW entry structure + traverse-all-return-visible search + per-tx delta).
