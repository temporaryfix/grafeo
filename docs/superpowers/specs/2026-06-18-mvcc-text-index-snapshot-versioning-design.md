# Snapshot-Versioned BM25 Text Index under Serializable — Design

**Status:** Design (approved in brainstorming 2026-06-18). Next: implementation plan via writing-plans.

## 1. Goal

Make BM25 full-text search **snapshot-consistent** (exact as-of-epoch results, including read-your-writes) and **SSI-sound** (predicate-read recording that prevents phantoms) under Serializable, then remove the text-search guard. Apply the LpgStore's own **MVCC-delta-over-base** model to the inverted index, structured as the **first instance of a reusable secondary-index MVCC pattern** that HNSW (vector) and, later, the property/label indexes can inherit.

This is the first of two cycles (decided in brainstorming): **text (BM25) first, vector (HNSW) next** — text forces the pattern out cleanly before HNSW's ANN-connectivity wrinkle.

## 2. Context (current state)

- The BM25 index (`crates/grafeo-core/src/index/text/inverted_index.rs`) is a classic inverted index: `postings: HashMap<String, Vec<Posting{node_id, term_freq}>>`, `doc_lengths: HashMap<NodeId, u32>`, `total_length: u64` (for `avgdl`). **No epoch awareness.**
- It reflects **committed-latest** state: `update_text_index_on_set`/`_on_remove` (`store/index.rs`) run on the committed property write; buffered (uncommitted) writes do NOT touch it (the increment-1 dirty-read fix). So a transaction's own uncommitted text edits are invisible to search, and an old snapshot sees the latest committed index, not its own epoch.
- `text_search` (`store/graph_store_impl.rs:392`) queries that committed index with **no visibility filter and no read recording** — which is exactly why it's guarded under Serializable (per-procedure policy from the graph-algorithm work; `SearchTextProcedure.serializable_safe() == false`).

## 3. The load-bearing insight (why version the index at all)

The unit of versioning is **term-membership** — "does node N contain term T at epoch E" — which is **finer than node existence**. A *living* node's text can change (a `SET` removes a word) without the node being created or deleted. Therefore the cheap pattern other indexes use (the label index stores `label→node-id` and filters results by *node* visibility at read) is **insufficient** for text: it would return a stale `(T, N)` posting for a still-visible node whose text no longer contains `T`. The postings need their **own** MVCC history.

The alternative considered and rejected — keep the index committed-latest and reconcile against the property version log at search time — collapses on the search path: per-posting reconciliation requires re-deriving each candidate doc's as-of-E term set, which is exactly the history you'd otherwise store. So **versioning the index is necessary**, and delta-over-base is the right form.

## 4. The two sharpenings (what makes this the strongest long-term shape)

### 4a. Single source of truth — the load-bearing invariant (first-class, tested)

The index's posting epochs are stamped **at, and derived from, the property's commit**. The index is a **consistent-by-construction projection** of the indexed property's MVCC version chain — never an independent history that can drift.

> **INVARIANT (acceptance criterion):** for every node `N` and epoch `E`, the index's as-of-E term set for `N` equals `tokenize(committed text of N's indexed property as-of E)`. The index never asserts a term-membership the property version chain doesn't.

Mechanically: a `SET N.<indexed-prop>` under a transaction buffers into **both** the property delta **and** the text-index delta (Section 5b); **commit finalizes both at the same epoch C**. There is no path that updates the index at a different time/epoch than the property it derives from. This is the same "one accessor, one source" discipline that fixed the dirty-read root cause earlier in the MVCC arc, applied to a derived structure. It is the place long-term bugs would otherwise live, so it is promoted to an explicit, tested requirement (Section 9).

### 4b. Reusable secondary-index MVCC pattern (not a text one-off)

Factor the machinery into separable concerns so HNSW (and later property/label indexes) inherit it rather than re-deriving:

- **versioned entries** (entry + `created_epoch` + `deleted_epoch`, retained until GC),
- **per-transaction delta** (uncommitted entry changes),
- **commit-promote** (delta → base, stamped at the commit epoch),
- **GC-at-horizon** (drop entries deleted ≤ `min_active_epoch`),
- **an index conflict-key for SSI** (Section 6).

The text index is the first concrete instance. The pattern is **extracted from a clean first instance, not designed speculatively upfront** (YAGNI): we build text with these concerns separated; the abstraction (a trait or a documented contract) crystallizes when vector adopts it. Vector adds only what is genuinely different (a versioned HNSW entry structure + a *traverse-all, return-visible* search to preserve ANN connectivity).

## 5. Architecture

### 5a. Versioned committed inverted index

`VersionedInvertedIndex` (evolves `InvertedIndex`):
- `Posting { node_id, term_freq, created_epoch, deleted_epoch }` (`deleted_epoch = NONE` while live, the commit epoch once superseded/removed; `created_epoch = PENDING` is never stored in the *committed* index — see 5b). A posting is **visible at (E, tx)** iff `created_epoch ≤ E` (or own-tx) and not (`deleted_epoch ≤ E` (or own-tx delete)). Mirrors `VersionInfo::is_visible_to`.
- `doc_lengths` versioned per-doc (a node's length can change with its text), reconstructable as-of-E.
- **As-of-E aggregate stats** for BM25 `avgdl`: maintain a small **versioned aggregate** (`total_length` + `doc_count` as an epoch-stamped log) so `avgdl@E = total_length@E / doc_count@E` is exact (consistent with "full completeness"). (Documented alternative if cost bites: latest `avgdl` — keeps the doc *set* exact, only ranking drifts; we choose exact.)
- Superseded/removed postings retained until GC.

### 5b. Per-transaction text-index delta

A per-tx overlay capturing the transaction's *uncommitted* text changes (its own `SET`/`REMOVE` on the indexed property, tokenized into posting inserts/removes at `PENDING`). Lives alongside `tx_property_overlay` in the store. The **only** writer of `PENDING`-epoch postings; the committed index holds no `PENDING`.

### 5c. Snapshot search — `search_visible(terms, E, tx)`

For each searched term: take the committed posting list **filtered to visible-at-(E, tx)** ⊕ the tx-delta postings for that term; BM25-score using the as-of-E aggregate stats; return the exact as-of-E top-K. This yields full completeness:
- own uncommitted inserts → from the delta (read-your-writes),
- committed-after-E → filtered out (`created_epoch > E`),
- deleted-after-E → retained + visible (`deleted_epoch > E`),
- term changes within a living node → handled by per-posting epochs.

### 5d. Commit / rollback

- **Commit (epoch C):** promote the tx text-index delta into the committed index, stamping `created_epoch`/`deleted_epoch = C` — **the same C that finalizes the property version** (Section 4a). Reuse the store's commit-finalize path (the SET already routes through the property commit; the index promote is wired into the same step).
- **Rollback / conflict-abort:** drop the tx text-index delta (alongside dropping the property delta).

### 5e. GC

Compact the committed index against `min_active_epoch` (the existing horizon driving store GC): drop postings whose `deleted_epoch ≤ horizon` and prune empty posting lists. Reuse the recompact trigger.

## 6. SSI recording — coarse index-level predicate (anti-phantom)

A text search reads a *predicate* ("docs matching these terms"); a concurrent transaction inserting a new matching doc is a **phantom**. To make the search serializable:

- Introduce an **`EntityId::Index(IndexId)`** conflict key, where `IndexId` identifies a `(label, property)` text index.
- A Serializable text search records a **read** of `EntityId::Index(idx)` (via the existing read-tracker → manager read-set/registry).
- A `SET`/`REMOVE` on the indexed `(label, property)` records a **write** of `EntityId::Index(idx)` (in addition to the node write).
- They conflict through the **existing** rw-detection — so a concurrent matching (indeed *any* indexed) insert/update aborts one of the pair. The phantom cannot slip through.

This is the standard table-/predicate-level lock: **sound, coarse** (a search conflicts with *any* write to that indexed property, even non-matching terms). **Term-level** predicate recording (record only the searched terms' posting-keys; write-side diffs the changed terms) is the documented follow-on — the same entity→property refinement Part G made for scans.

`EntityId` gains a third variant threaded through the SSI read-set/write-set/registry/Part-G-`PropTag` keys (clean reuse of rw-detection) — the chosen shape over a separate predicate read-set.

## 7. Integration + guard removal

- `text_search` / `SearchTextProcedure`: route through `search_visible(E, tx)` + record the `EntityId::Index` read (Serializable only; SI/RC use the committed-latest path unchanged).
- Flip `SearchTextProcedure.serializable_safe()` (and the MMR/hybrid text variants) to `true`; the per-procedure guard then admits them. (Vector stays `false` until its cycle.)
- The write path (`update_text_index_on_set`/`_on_remove`, and the SET operator under a tx) records the `EntityId::Index` write + buffers the tx text-index delta.

## 8. Data flow (summary)

| Event (Serializable) | Index action | SSI |
|---|---|---|
| `CALL text_search(label, prop, query)` at (E, tx) | `search_visible` (committed-visible-at-E ⊕ tx delta, avgdl@E) | record **read** `Index(idx)` |
| `SET N.prop = …` (indexed) in tx | buffer tokens into tx text-index delta (PENDING) | record **write** `Index(idx)` (+ node write) |
| commit @ C | promote tx delta → committed index, stamp C (== property commit) | — |
| rollback / abort | drop tx delta | — |
| gc @ horizon H | drop postings `deleted_epoch ≤ H` | — |

SI / Read-Committed: no tx delta merge, no recording, committed-latest search — **byte-unchanged**.

## 9. Acceptance criteria (done when)

1. **Full completeness:** under Serializable, a text search (a) returns the transaction's own uncommitted matching inserts; (b) excludes docs committed after the transaction's snapshot; (c) includes docs deleted after the snapshot; (d) scores with as-of-E `avgdl` (exact). Each an explicit test.
2. **Single-source invariant (Section 4a) — explicitly tested:** mutate a node's indexed text across several epochs (insert/change/remove terms, with concurrent committed snapshots), and assert that for every (node, epoch) the index's as-of-E term set equals `tokenize(as-of-E committed property)`. The index never returns a term the property chain doesn't have at E, and never misses one it does.
3. **SSI soundness (anti-phantom):** a Serializable text search concurrent with a transaction that inserts/updates a matching doc → a serialization failure (the phantom is caught); a search disjoint from a concurrent write (different indexed property) → both commit.
4. **SI / Read-Committed unchanged:** the committed-latest search path + results are identical to today (versioning is additive; delta + recording are Serializable-only). Full existing text-index test suite green.
5. **Guard removed:** `CALL text_search`/MMR/hybrid plan + run under Serializable; vector still rejects.
6. **Reusable shape:** the versioned-entries / tx-delta / commit-promote / GC / index-conflict-key concerns are separated such that the vector cycle can adopt them without rewriting them (a short "what vector reuses" note in the code/docs).
7. Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC.

## 10. Residuals / future (documented, not in this cycle)

- **Coarse predicate recording over-conflicts** — a Serializable text search aborts against any write to the indexed property. Term-level predicate recording is the follow-on refinement.
- **Vector / HNSW** — the next cycle: same MVCC-secondary-index pattern + a versioned HNSW entry structure + *traverse-all, return-visible* search (ANN connectivity) + per-tx delta merge.
- **`avgdl` exactness cost** — the versioned aggregate adds bookkeeping; if it dominates, fall back to latest-`avgdl` (doc set stays exact).
- **Memory** — versioned postings retained until GC; bounded by the active-tx horizon (like the store).

## 11. Risks

- **Single-source drift (the #1 risk).** Any path that updates the index at a different epoch than the property it derives from — or a tokenization mismatch between the index-update and a re-derivation — makes the index lie. Mitigation: one commit stamps both; the invariant test (Section 9.2); audit every `update_text_index_*` call site routes through the tx delta + same-commit promote.
- **`EntityId::Index` blast radius.** A third `EntityId` variant threads through read-set/write-set/registry/`retired_readers`/`PropTag` keys. Mitigation: it's additive (Node/Edge paths unchanged); gate behavior on the variant; reuse the Part-G generalization discipline.
- **Predicate-read coarseness** — documented over-abort; term-level deferred.
- **Aggregate-stats correctness** — `avgdl@E` must use the as-of-E doc set; an off-by-one in the versioned aggregate skews scores (not soundness, but correctness). Test against re-derivation.
