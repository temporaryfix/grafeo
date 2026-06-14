# Audit-fix design — integration branch remediation

**Date:** 2026-06-14
**Branch:** `integration` → `fix/audit-followups`
**Scope:** Remediate the 10 findings from the 2026-06-13 branch-diff audit (Plan 2 codecs, `open_multi`/`extract_subgraph` persistence, planner TopK rewrite). All changes are confined to the project's own compact-store / Plan-2 / engine code — no upstream shared primitives are modified, so no RFC is required.

## Context

The audit compared `integration` against `upstream/main` and produced 7 test-confirmed correctness bugs plus 3 performance/doc findings. Five of the correctness bugs were reproduced with failing tests during the audit. This spec defines best-in-class fixes, grouped into five clusters by subsystem.

### Design decisions (locked with the user)

| Decision | Choice |
|---|---|
| Layered-store tier merge | **Centralized invariant** — one visibility rule, shared helpers, all read + delete paths routed through it |
| TopK alias collision | **Structural expression match** — compare `LogicalExpression`s, not formatted name strings; conservative local matcher, no `PartialEq` on the shared enum |
| WASM id surface | **f64 / JS number** — exact below 2^53, idiomatic |
| open_multi index-config conflict | **Reject any difference** — error at open time, consistent with schema-conflict handling |
| Extra graph accessors (`graph_store_ref` etc.) | **Investigate, fix if reachable** — trace real post-compact read paths; fix via shared helper if reachable, comment if provably dead |
| Branch strategy | **One branch (`fix/audit-followups`), one commit per finding, one PR** |

## Cluster A — Layered-store tier invariant (findings #1, #2, #3)

### Root cause
Tier-visibility rules are hand-rolled independently in each `GraphStore` method on `LayeredStore`, and the delete paths and read paths disagree about the invariant. Promotion (`ensure_in_overlay` / `ensure_edge_in_overlay`) copies an entity's labels/properties into the overlay but **leaves its adjacency in the base tier**. A prior fix (commit 70c9b92a) removed an `is_node_dirty` guard so the base tier is always consulted for adjacency — correct — but the delete paths never tombstoned the base copy of a promoted entity, so deletes silently fail to take effect.

### The invariant (stated once)
An entity is **live** iff `overlay-live OR (base-live AND NOT tombstoned)`, where "tombstoned" means membership in `deleted_from_base_nodes` / `deleted_from_base_edges`.

### Fixes

**A1 — Centralize adjacency merge (fixes #2, ghost neighbors).**
Extract the current `edges_from` body into one private helper `merged_edges(node, dir) -> Vec<(NodeId, EdgeId)>` that applies the node + edge tombstone filters and dedups promoted edges by `EdgeId`. Rewrite `neighbors()` to delegate: `merged_edges(node, dir).into_iter().map(|(t, _)| t)`, then `sort_unstable` + `dedup`. This removes `neighbors`' separately-hand-rolled merge that only filtered `deleted_from_base_nodes` and never consulted `deleted_from_base_edges`. `out_degree`/`in_degree` already delegate to `edges_from`, so they inherit the fix.

**A2 — Make deletes complete (fixes #1, resurrection).**
`delete_edge`, `delete_node`, and their `_versioned` variants currently branch either/or: `if dirty { overlay.delete } else if base.contains { tombstone }`. Replace with independent both-sides handling:
- delete the overlay copy if the overlay contains the entity, capturing whether it removed anything;
- if the base tier contains the entity, insert its id into the tombstone set;
- return `true` if either side removed/tombstoned.

A promoted entity exists in both tiers → both branches run. A fresh overlay-only entity has no base copy → only the overlay delete runs. `delete_node_edges` already tombstones base edges + deletes overlay edges (both sides) and is left as-is, but verified against the invariant.

**A3 — Route extract_subgraph / remove_orphan_edges through the merged view (fixes #3).**
Both functions currently read `self.lpg_store()` — the overlay-only tier — so post-`compact()` they (a) error validating base-tier nodes as "does not exist" and (b) silently drop promoted nodes' base edges. The session/query path already solves this by overriding stores with the `LayeredStore` (mod.rs:1789). Add a private accessor:

```rust
fn read_graph_view(&self) -> &dyn grafeo_core::graph::GraphStore {
    // LayeredStore when compacted, else the overlay LpgStore.
}
```

Route `extract_subgraph` and `remove_orphan_edges` through it for all reads (`get_node`, `edges_from`, `get_edge`). Writes to the *target* extract DB still use its own `lpg_store()` (the target is never compacted mid-extract). The deliberate, documented "source-side ownership / dangling-dst edge" design and the `remove_orphan_edges` utility are intentional and **retained** — only the source-side store accessor is corrected.

**A4 — Audit the other accessors (investigate, fix if reachable).**
`graph_store_ref()` (mod.rs:228; used by vector/text/embed search), `graph_store()`, and `graph_store_mut()` read `lpg_store()` directly and would return overlay-only results post-compact. Trace whether any real read path reaches them after `compact()` (search may always go through the session override). If reachable, route through `read_graph_view` (read accessors) / the layered store (mut). If provably dead post-compact, leave a code comment documenting why. No silent landmine either way.

### Tests
- `compact → set_edge_property(e) → delete_edge(e)`: assert `edges_from`, `neighbors`, `get_edge`, and a second `delete_edge` all agree the edge is gone.
- `compact → set_node_property(n) → delete_node(n)`: assert adjacency + `get_node` agree.
- The `_versioned` delete paths under the same scenario.
- `compact → extract_subgraph([base_node])`: succeeds; a promoted node's pre-compact base edges survive into the extract (the audit's failing test: expected 1 edge, got 0).
- Single-edge base deletion: `edges_from` and `neighbors` both omit the target.

## Cluster B — TopK alias collision (finding #4)

### Bug
`try_heap_topk_rewrite` resolves sort keys to projected columns by formatted name string via `resolved_column_name` → `variable_columns: HashMap<String, usize>`. A user alias can collide with the synthetic `{var}_{prop}` name: `RETURN n.foo AS n_age ORDER BY n.age LIMIT 5` predicts column `"n_age"` (the alias), and `resolved_column_name(Property{n,age})` is also `"n_age"`, so the sort key resolves to the alias's column and the query sorts by `n.foo`.

### Fix (structural match)
Decide fusion by comparing the sort key's `LogicalExpression` against each projected Return item's **source expression**, not their output names. `LogicalExpression` is `#[non_exhaustive]` and derives only `Debug, Clone`, so add a small conservative matcher local to `project.rs`:

```rust
fn sort_key_matches_projection(key: &LogicalExpression, item: &LogicalExpression) -> bool {
    match (key, item) {
        (Variable(a), Variable(b)) => a == b,
        (Property { variable: v1, property: p1 }, Property { variable: v2, property: p2 })
            => v1 == v2 && p1 == p2,
        _ => false, // conservative: unknown → no match → bail to unfused sort
    }
}
```

In the rewrite, when `sort.input` is a `Return`, build the list of `(item.expression, column_index)` and resolve each sort key by structural match against it. No match for any key → return `Ok(None)` → fall through to the correct unfused `plan_sort` path (which augments the projection with the real sort column). This:
- eliminates the entire name-collision class;
- lets `register_return_property_sort_aliases` be retired for the TopK path (structural match handles `RETURN v.p ORDER BY v.p` directly — both expressions are `Property{v,p}`);
- keeps the `debug_assert_eq!` drift check on the *name* prediction (still used to build the operator's output columns) as a real guard.

We deliberately do **not** add `PartialEq`/`Eq` to the shared `LogicalExpression` enum (float-NaN semantics + broad blast radius); the local matcher with a fail-safe default is sufficient and contained.

### Tests
- The collision case asserts rows ordered by `n.age` (audit's confirmed failing case).
- Cases that must still fuse: `RETURN n.title ORDER BY n.title LIMIT k`; `RETURN n.a AS x, n.b AS y ORDER BY n.a LIMIT k` (sorts by `n.a`).
- Case that must still bail: `RETURN n ORDER BY n.p LIMIT k` (property of a projected entity, not itself projected) → unfused, correct.

## Cluster C — open_multi index merge + docs (findings #6, #10)

**C1 — Reject conflicting index configs (fixes #6).**
`union_index_metadata` dedups vector/text indexes by `(label, property)` first-wins, silently dropping a later snapshot's differing config; the merged DB then fails to rebuild the whole index (dimension mismatch on the first foreign-dimension vector), swallowed as a `grafeo_warn!`, and `open_multi` returns `Ok` with no index.

- Make `union_index_metadata` return `Result<SnapshotIndexes>`. When the same `(label, property)` recurs with **any** differing config field (`dimensions`, `metric`, `m`, `ef_construction`), return `Err` naming the conflicting `(label, property)`, the differing field(s), and the snapshot indices — mirroring `merge_snapshot_schemas`' existing conflict-report style. Identical configs merge unchanged.
- `open_multi` / `open_multi_with` propagate the `Err` before populating any data.
- Secondary hardening: make `restore_indexes_from_snapshot` propagate rebuild failures (return `Result`) so `open_multi` cannot return `Ok` over a broken index. Verify the single-snapshot callers (`import_snapshot`, `restore_snapshot`): if any deliberately tolerate a missing-feature/unbuildable index, keep that path lenient and apply strictness only on the multi path.

**C2 — Fix stale docs (fixes #10).**
Rewrite `open_multi`/`open_multi_with` rustdoc:
- delete the five "(not yet enforced; see Task N…)" parentheticals — the validation they disclaim is now implemented;
- correct the index description to "union across snapshots; conflicting configs for the same `(label, property)` are rejected";
- fix `open_multi_with`'s "# Panics … if `snapshots` is empty" → documents the `Err` return.

### Tests
- Two snapshots declaring `(:Doc, embedding)` at 384 vs 768 dims → `open_multi` returns `Err` naming the conflict; no partial DB.
- Identical-config sibling extracts → merge cleanly, index present and queryable.
- Empty `snapshots` slice → `Err` (matches corrected docs).

## Cluster D — codec correctness + WASM ids (findings #7, #5)

**D1 — zigzag i64::MIN (fixes #7).**
`write_zigzag_gamma`/`read_zigzag_gamma` hand-roll a branchy signed fold that overflows for `i64::MIN` (`-2n` = 2^64; `+2` wraps to 0 in release → `write_gamma(0)` corrupts the stream; debug panics).

- Replace the fold with the canonical `delta::zigzag_encode` / `zigzag_decode` (reuse; branchless; correct for `i64::MIN` → `u64::MAX`).
- Gamma requires `n ≥ 1`: encode `delta::zigzag_encode(n).checked_add(1).expect("zigzag-gamma overflow: |gap| == i64::MIN, unreachable for in-memory graphs")`; decode `delta::zigzag_decode(read_gamma()? - 1)`.
- The single unrepresentable input (`i64::MIN`, requiring a graph with > 2^63 nodes) now produces a **consistent panic in debug and release** instead of silent stream corruption. Document the supported domain on the function.

**D2 — WASM id truncation → f64 (fixes #5).**
`RabitqCodec.search` returns `Vec<u32>` via `id.as_u64() as u32`, silently corrupting ids ≥ 2^32 from natively-built blobs.

- Change the return to `Vec<f64>` (→ `Float64Array` in JS): exact for any id < 2^53, idiomatic as JS array indices/keys, no BigInt ergonomics cost. Document the 2^53 bound.
- Same defect, same file: `WebGraphCodec.successors(node: u32)`, `out_degree`, and `num_nodes` truncate identically. Widen the node argument and these returns to `f64` for consistency (in-scope: same bug class, same binding file).

### Tests
- Rust `wasm` test: build a blob containing a `NodeId ≥ 2^32`, open it, `search`, assert the id round-trips exactly through the `f64` surface.
- zigzag round-trip extended across the full i64 range up to the documented boundary (the existing test stops at `i32::MIN`).

## Cluster E — codec performance (findings #8, #9)

**E1 — FSST encode (fixes #9).**
`SymbolTable::longest_match` scans all 255 codes × up to 8 bytes at every input position (its own doc comment names the fix).

- Build a 256-entry first-byte bucket index at `SymbolTable` construction: for each possible first byte, the list of symbol codes starting with that byte, ordered longest-first so greedy match returns immediately. `longest_match` consults only `input[pos]`'s bucket.
- Fix `train()`'s per-occurrence `counts.entry(sub.to_vec())` heap allocation with a borrowing `HashMap<&[u8], u64>` keyed on slices of the sample.

**E2 — RaBitQ query path (fixes #8).**
`RabitqView::search` allocates a `Vec<u64>` per stored vector per query (`read_code_bits`), and both `coarse_search` and `RabitqView::search` fully sort all N scored candidates to keep the top `n`.

- Reuse a single `words`-sized scratch buffer across the per-vector scan (or read LE code words directly from the blob slice into the hamming kernel — no `Vec`).
- Replace `sort_by + truncate(n)` with `select_nth_unstable_by(n)` then sort only the kept `n`-prefix — O(N + n log n) vs O(N log N) — on both the owned (`coarse_search`, ~line 418) and view (~line 1076) paths.

### Tests
These are performance fixes under existing correctness contracts; behavior must not change. Coverage:
- existing `fsst_round_trip`, `rabitq_recall`, and codec parity tests must continue to pass unchanged;
- confirm `rabitq_recall` numbers hold after the selection-sort change (selection + partial sort yields the same top-n set/order);
- optionally assert ordering equivalence between `coarse_search` (full sort) and the new partial-selection path in a unit test.

## Sequencing

1. **Cluster A** first — it establishes the tier invariant and unblocks the `extract_subgraph` test. A4 (accessor investigation) rides with A.
2. **Clusters B, C, D, E** are mutually independent and can land in any order after A.
3. Each finding is its own commit on `fix/audit-followups`. Each fix is **TDD**: write/restore the failing test first (five repros already exist from the audit), confirm red, then implement to green.
4. Final `cargo test` + `cargo clippy` across affected crates and feature flags (`lpg`, `compact-store`, `vector-index`, `text-index`, `wasm`) before PR.

## Out of scope

- The earlier (2026-06-09) engine-isolation findings (uncommitted DELETE/SET visibility, variable-length expand blowup, commit-conflict abort leak) — tracked separately in `audit_scratch.rs`, not part of this branch's diff.
- Codec abstraction de-duplication (the triplicated blob header parsers, SplitMix64 copies, snapshot mirror structs) — real cleanups but pure churn for this remediation; deferred unless the user wants them folded in.
- Any change to upstream shared primitives or the `LogicalExpression` enum's derives.
