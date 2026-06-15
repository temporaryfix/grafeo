# Audit Wave 1 — query-operator & transaction quick-correctness

## Context

The 2026-06-09 engine audit produced a set of engine-core findings (transaction
isolation, query-operator correctness, traversal semantics, value semantics,
hygiene). They were deliberately **out of scope** for the branch-diff remediation
(`fix/audit-followups`, merged to `integration` as `ae067b88..a9aa6830`) and
tracked separately in `crates/grafeo-engine/tests/audit_scratch.rs`.

On 2026-06-15 the engine-core findings were re-verified against current
`integration`: all remain live and unchanged by the branch-diff work. (One probe,
`uncommitted_property_write_invisible_to_others`, was a *false pass* — it matched
`"30"` inside the non-deterministic `execution_time_ms` field of the `Debug`
output rather than the cell value; corrected to assert on the actual value, it now
correctly fails on the live dirty-read.)

The engine-core track is too large for a single spec. It is decomposed into five
waves, each its own spec → plan → PR:

| Wave | Subsystem | Findings |
|---|---|---|
| **1 (this spec)** | Query-operator + cheap correctness | NULL join keys (1e), join-tree filter drop (2e), conjunction splitting (perf), commit-conflict leak (1d), Serializable-is-SI (1f), `copy_graph` no-op (1g) |
| 2 | Transaction isolation + O(N) commit | write-set-scoped commit/rollback (2a), uncommitted DELETE visibility (1a), uncommitted SET/label/index visibility (1b), real SSI |
| 3 | Unified value semantics | one canonical equality/ordering, migrate the ~6 divergent sites (2b) |
| 4 | Traversal semantics | variable-length expand exponential blowup (1c) |
| 5 | Perf/hygiene tail | adjacency tombstone purge (2c), epoch dual-source-of-truth (2f), `record_write` O(all-txns) scan, dormant compressed-column data loss |

Wave 1 is the **quick-correctness wave**: independent, low-risk, individually
shippable fixes that also validate the spec → plan → TDD loop before the deeper
isolation rework in Wave 2.

### Design decisions (locked with the user)

| Decision | Choice |
|---|---|
| Sequencing | Quick-correctness wave first; then isolation (Wave 2), value semantics (3), traversal (4), hygiene tail (5) |
| Serializable isolation (1f) | **Reject at begin now** — `begin` with `Serializable` returns a clear "not yet supported" error. Real SSI (read-set tracking through scans) is deferred to Wave 2, where the read-path work lives |
| `copy_graph` (1g) | **Implement the real deep copy now** — `CREATE GRAPH x AS COPY OF y` must actually copy data |
| Edge-type case-insensitivity | **Not a bug — dropped.** Deliberate, test-covered behavior (`test_expand_edge_type_case_insensitive`, `test_edge_type_filter_case_insensitive`), consistent across all six traversal operators. Strict case-sensitivity would be a separate, explicit behavior change |
| Compressed-column data loss | **Deferred to Wave 5.** Dormant — zero engine callers reach the public compression API; defensive-only, out of place in a correctness wave |
| `collect_join_tree` filter drop (2e) | **Fix properly** (preserve predicate or decline reorder), not gate — `Join` is produced by the binder + every translator and join-reorder is on by default, so it is reachable, not merely latent |
| Branch strategy | One branch (`fix/audit-wave1-query-tx-correctness`) off `integration`, **fork-local**, one commit per finding, one PR. Fork-local is the standing scoped exception for engine-core remediation; the RFC-first rule still applies generally. OPSEC: no private downstream schema/business logic in tests or commits |

## Findings

Each finding is one TDD commit: failing test first, then the fix, then green.
Listed in commit order (cheapest + most independent first, largest last).

### F1 — Commit-conflict leak (audit 1d)

**Root cause.** In `session/mod.rs`, the commit path's conflict branch (when
`transaction_manager.commit(tx)` returns `Err`) only calls
`store.rollback_transaction_properties(tx)` per touched graph. It never:
- calls `store.discard_uncommitted_versions(tx)`, so PENDING node/edge versions
  created by the transaction leak permanently into the version chains; and
- calls `transaction_manager.abort(tx)`, so the transaction stays `Active`
  forever. A permanently-`Active` transaction pins `min_active_epoch`, which
  stops MVCC version GC entirely, and leaves a stale write-set that
  `record_write` re-scans on every future write.

**Fix.** In the conflict branch, per touched graph call
`discard_uncommitted_versions(tx)` (which internally replays the property undo
log and flags stats for recompute), then call `transaction_manager.abort(tx)`.
This mirrors the existing `rollback_inner` cleanup. Existing post-conflict state
resets (read-only flag, savepoints, touched graphs, CDC pending) are preserved.

**Contract.** After `session.commit()` returns `Err` (conflict), the transaction
is over (aborted); the caller must `begin_transaction()` again to retry.

**Test.** Force a *commit-time* write-write conflict reachable from the session
API: T1 and T2 begin at the same epoch; T1 writes entity E and commits; T2 then
writes E (allowed — T1 is no longer `active`, so first-writer-wins does not fire)
and commits, which returns `Err`. Assert: T2 is `Aborted` (not `Active`); a
subsequent `transaction_manager.gc()` reclaims; a fresh session reads T1's
committed value exactly once.

### F2 — NULL keys join in outer joins (audit 1e)

**Root cause.** `HashJoinOperator::build_hash_table` inserts NULL build keys into
the hash table for `Left | Right | Full` joins (it only skips NULL keys for
inner/semi/anti). A probe row whose key is NULL then looks up the NULL bucket and
matches, so `NULL = NULL` joins rows — violating three-valued logic (filters get
this right; the join does not).

**Fix.** Never insert NULL keys into the hash table, for any join type. Outer-join
correctness is preserved because unmatched-row emission does not depend on the
table:
- LEFT/FULL: a probe row with a NULL key finds no bucket → the existing no-match
  path emits it null-padded.
- RIGHT/FULL: build rows are emitted as unmatched via the `build_matched`
  tracking array (independent of the hash table), so a NULL-key build row is
  still emitted unmatched.

**Test.** Outer join on a key that is NULL on both sides → no spurious match;
LEFT emits the left row null-padded; RIGHT/FULL emit the unmatched build row;
semi/anti unchanged. (Float/cross-type key consistency is Wave 3, not here.)

### F3 — `collect_join_tree` drops Filter predicates (audit 2e)

**Root cause.** In the optimizer's join-reorder pass, `collect_join_tree`'s
`Filter` arm recurses into `filter.input` and discards `filter.predicate`. When a
filtered relation participates in a `Join` tree and reordering rebuilds the tree
from the collected relations + conditions, the predicate is silently lost.
Reachable: `Join` is constructed by the binder and every translator, and
`enable_join_reorder` defaults to `true`.

**Fix.** In the `Filter` arm of `collect_join_tree`:
- if the filter's child is a single base relation (`NodeScan` / `EdgeScan` /
  `TripleScan` / `Expand`), record the *entire* `Filter(child)` as that
  relation's entry, so the predicate travels with its relation through
  reordering; otherwise
- (filter spans a join / complex subtree) return `false`, declining to flatten —
  `extract_join_tree` then returns `None` and reordering is skipped, leaving the
  original (predicate-bearing) plan intact.

Either way the predicate is never dropped. Declining to reorder is a performance
concession in rare shapes, never a correctness change.

**Test.** Optimizer unit test: `Join(Filter(p, NodeScan a), NodeScan b)` through
`reorder_joins` still contains `p` (structural check on the rebuilt plan). End-to-
end: a multi-relation query whose filtered relation would be reordered returns
the correctly filtered result.

### F4 — Conjunction splitting in filter pushdown (performance)

**Root cause.** `push_filters_down` treats `Filter(A AND B)` as one unit. A
conjunct that could anchor on relation X is stranded above a cartesian product
(e.g. `MATCH (a),(b) WHERE a.x = 1 AND b.y = 2` keeps the whole `AND` above the
two scans). Results are already correct; this is the performance cliff (and the
broadest-blast-radius item in the wave).

**Fix.** At the head of `push_filters_down`'s `Filter` arm, split the top-level
`AND` chain into individual conjuncts and push each independently via
`try_push_filter_into`. Conjuncts that cannot be pushed re-stack as filters (the
existing fall-through behavior). Because `optimize()` runs before
`annotate_pushdown_hints`, each resulting single-predicate filter then receives
its own index/range pushdown hint — this also retires the "inspect only the first
conjunct" limitation in `infer_pushdown`. Reuse or mirror the existing
`split_conjuncts` helper from `query/translators/common.rs`.

**Test.** EXPLAIN shows each predicate anchored on its own scan rather than one
combined filter above a product; results are unchanged. **Risk control:** this
reshapes many plans — run the full engine suite + spec-compliance suites and
reconcile any plan-snapshot/EXPLAIN tests as part of this commit.

### F5 — Reject Serializable at begin (audit 1f)

**Root cause.** `Serializable` is reachable from the public API
(`begin_transaction_with_isolation`) and GQL (`TransactionIsolationLevel::
Serializable`), but `record_read` is never called from the live execution path,
so the commit-time SSI validation never fires — `Serializable` silently behaves as
Snapshot Isolation.

**Fix.** In `begin_transaction_inner` (after the nested-transaction early return,
so it covers both the direct API and the GQL mapping, and so nested begins that
ignore isolation are unaffected): if the requested level is `Serializable`, return
a clear error — "Serializable isolation is not yet supported; use
SnapshotIsolation (real SSI is tracked for Wave 2)". Manager-level SSI unit tests
remain valid (they exercise `TransactionManager` directly, which is correct).

**Test.** `begin_transaction_with_isolation(Serializable)` and the GQL form return
`Err`; `ReadCommitted` and `SnapshotIsolation` are unaffected. Update
`crates/grafeo-engine/tests/coverage_session.rs` (the `Serializable` begin and its
nested variant) to expect the rejection.

### F6 — Real `copy_graph` deep copy (audit 1g)

**Root cause.** `CREATE GRAPH x AS COPY OF y` routes to `LpgStore::copy_graph`,
which is a silent no-op returning `Ok(())` — it produces an empty graph and
reports success (silent data loss on a reachable, advertised feature).

**Fix.** Implement a deep copy in `LpgStore::copy_graph(source, dest)`:
1. Resolve the source partition (`None` = default/self; `Some(name)` = named
   graph) and resolve-or-create the destination partition. **Reject copy-to-self**
   (source and dest resolve to the same store) with a clear error.
2. Iterate the source's live nodes; for each, read its labels + properties and
   create the node in dest, building an `old_id → new_id` map. (IDs are remapped
   rather than preserved: dest is an independent partition with its own ID
   allocator; remapping avoids any assumption about ID reuse.)
3. Iterate the source's live edges; for each, read its type + properties and
   create the edge in dest with endpoints translated through the ID map.
4. Re-create each of the source's property indexes on dest.

**Scope (v1).** Copies node/edge data, labels, properties, and **property
indexes**. Vector and text indexes are **not** carried by the copy in v1 (they
require re-embedding / re-tokenizing all values, which is non-trivial); this is
documented on the method and flagged for a follow-up if full index copy is wanted.
The copy is a true deep copy (no `Arc` sharing): mutating the copy never affects
the source.

**Test.** Build a source graph with nodes, edges, properties, labels, and a
property index; `CREATE GRAPH dest AS COPY OF source`; assert node/edge counts,
properties, labels, and index lookups match; assert the source is unchanged; and
assert mutating the copy does not affect the source.

## Sequencing

Commit order: **F1 → F2 → F3 → F4 → F5 → F6**. F1–F3 and F5 are small,
independent, and low-risk; F4 has the broadest plan-shape blast radius and is
sequenced after the small correctness fixes so any plan-snapshot churn is isolated
to its own commit; F6 is the largest (new code) and lands last. Each finding is a
single commit with its test(s) first (TDD). Wave-1-relevant probes are promoted
into properly-homed tests (the existing `failed_commit_does_not_pin_gc` probe is
strengthened into a transaction integration test that forces a true commit-time
conflict; a NULL-join unit test is added where no probe exists). The Wave 2/4
probes (`uncommitted_delete_invisible_to_others`,
`uncommitted_property_write_invisible_to_others`, the variable-length-expand
blowup) **remain** in `audit_scratch.rs` for their waves; the scratch file is
retired only when the last wave that depends on it lands.

## Verification bar

Matching the prior remediation's bar, checked before opening the PR:
- `cargo test --all-features` green;
- the `lpg` and `analytics` feature profiles and the `wasm` crate compile;
- `clippy` adds **zero** new warnings.

## Out of scope

- Real SSI / read-set tracking (Wave 2), uncommitted DELETE/SET/label/index
  isolation (Wave 2), O(N) → write-set-scoped commit/rollback (Wave 2).
- Unified value semantics / the ~6 divergent comparison sites (Wave 3); float and
  cross-type join-key consistency (Wave 3).
- Variable-length expand exponential blowup (Wave 4).
- Adjacency tombstone purge, epoch dual-source-of-truth, `record_write`
  O(all-txns) scan, dormant compressed-column point-read data loss (Wave 5).
- Strict edge-type case-sensitivity — deliberate, test-covered current behavior;
  revisit only on explicit request.
- Any change to upstream shared primitives beyond what these fixes require, and
  any change to the `LogicalExpression` enum's derives.
