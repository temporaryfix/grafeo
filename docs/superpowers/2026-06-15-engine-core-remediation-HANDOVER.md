# Handover — engine-core remediation (2026-06-15)

You are picking up a multi-wave remediation of the grafeo engine's core
(correctness, architecture, performance). Read your auto-memory
`project_grafeo_audit.md` too — it has the full finding list and history. This
doc is the "continue here" pointer.

## TL;DR

- **Shipped & public** (`origin/integration`, fast-forward to `532e190d`):
  **Wave 1** (6 query/tx correctness fixes) + **Wave 2a** (write-set-scoped
  commit/rollback). Verified `--all-features` green, clippy clean, profiles+wasm
  compile.
- **In progress** (local branch `design/unified-mvcc-isolation`, NOT pushed):
  the **unified MVCC isolation design** (pre-RFC) + its **step-1 foundation**
  (additive, tested, nothing wired yet).
- **Next concrete task:** wire the live read/write paths through the new
  snapshot-aware accessor so the two dirty-read bugs are actually fixed. Details
  below.

## The strategic picture (don't re-litigate this)

The engine had three partial MVCC mechanisms (version-chain existence [correct],
non-temporal write-through+undo, unused temporal `VersionLog`) plus the
`LayeredStore` overlay — all morally "small mutable recent layer over big
compressed committed base," none unified. The read path honors MVCC for
*existence* but reads property/label *data* at latest (ignores the snapshot) —
**that is the dirty-read root cause.**

The agreed target (design doc:
`docs/superpowers/specs/2026-06-15-unified-mvcc-isolation-design.md`):
**one snapshot-consistent read accessor, over one MVCC delta, over one columnar
committed base.** Committed data stays columnar/compressed (analytics + WASM
memory); uncommitted/recent lives in a small short-lived MVCC delta; every read
goes through one accessor (the load-bearing invariant). Migration leads with the
accessor (fixes the bugs, establishes the invariant), then converges the delta,
then adds compaction, then retires the `temporal` flag.

Two earlier approaches were tried and rejected for good reasons — **do not
revive them**: (a) collapsing to always-versioned storage (loses compression,
spill, and zone maps — bad for the WASM budget); (b) a per-transaction
write-buffer as a standalone mechanism (it's a 4th mechanism, not convergence).
The superseded write-buffer spec lives on branch
`fix/audit-wave2b-property-isolation` — ignore it except for its capability-loss
finding (already folded into the unified design).

## Exactly where to continue (step 1, the next increment)

The **foundation is built** on `design/unified-mvcc-isolation` (commit
`b6ec2035`), additive — it changes no existing behavior yet:
- `LpgStore.tx_property_overlay` — the per-transaction delta (`TxDelta` /
  `PropOp` in `store/mod.rs`).
- `read_node_property_visible(id, key, epoch, Option<tx>)` /
  `read_edge_property_visible(...)` — the snapshot-aware accessor
  (`store/property_ops.rs`). Writer sees own delta (read-your-writes); everyone
  else reads committed (no dirty read); committed column untouched.
- `set_node_property_buffered` / `remove_node_property_buffered` /
  `set_edge_property_buffered` — buffer into the delta.
- `apply_tx_overlay(tx)` (commit) / `drop_tx_overlay(tx)` (rollback).
- Unit test: `tx_property_overlay_isolates_uncommitted_writes` (passing).

**To make step 1 functional, wire the live paths (TDD, one commit per piece):**

1. **Write path.** In `crates/grafeo-engine/src/session/mod.rs`,
   `set_node_property` / `set_edge_property` (and the operator-driven SET path in
   `crates/grafeo-core/src/execution/operators/mutation.rs` `SetPropertyOperator`)
   should, **when a transaction is active**, call the `*_buffered` methods instead
   of write-through (`set_*_property_versioned`). Non-transactional (SYSTEM) writes
   stay write-through.
2. **Read path (the broad, subtle one — be exhaustive).** Every property read in
   execution and in node/edge *materialization* must go through
   `read_*_property_visible`, passing the operator's `transaction_id`. Start by
   enumerating call sites: `grep -rn "get_node_property\|get_edge_property\|get_all\|get_nodes_properties_batch\|get_selective_batch" crates/grafeo-core/src/execution`. The
   node/edge materialization path (building `Node`/`Edge` objects with all their
   properties — used by RETURN/projection) is the most important; find where
   `build_node`/`get_all`/property batches feed query results and route them
   through the accessor with the tx. **Completeness is the whole game: one missed
   site = a remaining dirty read the probes may not catch.**
3. **Commit/rollback.** Session commit calls `store.apply_tx_overlay(tx)` per
   touched graph (after the Wave 2a version finalize); rollback + conflict call
   `store.drop_tx_overlay(tx)`. Then the property undo log is no longer needed for
   the buffered path (keep it only for SYSTEM/savepoint write-through).
4. **Deletes** (for the second probe): transactional delete should set
   `deleted_epoch = PENDING` (extend `VersionInfo`/`mark_deleted` to take a PENDING
   epoch and `finalize_epochs` to promote it), and defer adjacency tombstones to
   commit. See the unified design §2.2 and the superseded 2b spec §"Deletes".

**Acceptance tests:** the two `#[ignore]`d probes in
`crates/grafeo-engine/tests/audit_scratch.rs`
(`uncommitted_property_write_invisible_to_others`,
`uncommitted_delete_invisible_to_others`) must pass — remove their `#[ignore]`.
Also: read-your-writes within a tx; rollback restores; other-session isolation.

## Roadmap after step 1

Step 2 (converge the delta, retire `VersionLog`) → step 3 (compaction folds
delta→columnar base, retire the `temporal` flag) → **2c** (real SSI: wire
`record_read` through scans, flip the Wave-1 Serializable rejection) → **Wave 3**
(unify the ~6 value-comparison sites) → **Wave 4** (variable-length expand
exponential blowup — Trail default + frontier budget; probe
`var_length_expand_branching_cycle`) → **Wave 5** (adjacency tombstone purge,
epoch dual-source-of-truth, `record_write` O(all-txns) scan, dormant
compressed-column data loss). Full list in memory.

## Critical gotchas / conventions (learned the hard way)

- **Run `cargo test --all-features -p grafeo-core -p grafeo-engine`, not just
  targeted tests.** It caught that MERGE/LOAD DATA bypass operator-level write
  tracking (the Wave 2a write-set fragility) — targeted tests missed it. It's the
  real gate.
- **Feature-gating:** `execute_cypher` is `#[cfg(feature="cypher")]`,
  `execute_sql` is `#[cfg(feature="sql-pgq")]`. Integration tests that use them
  need `--features full` (or the right feature) to even compile — a bare
  `cargo test --test X` fails with "no method", which is **pre-existing**, not your
  regression.
- **Write-set completeness:** scope commit/rollback via the **store-level**
  `pending_tx_creates` recorded at `create_*_versioned` (the chokepoint), NOT the
  transaction-manager write-set (operators like MERGE/LOAD DATA bypass
  `record_write`). `TransactionManager::record_entity` exists but is currently
  unused (kept for the session-direct conflict-detection gap — a separate noted
  bug: session-direct mutators don't conflict-detect).
- **OPSEC (hard rule):** never put the private downstream project's name/schema/
  business terms in any committed/pushed artifact. (It's already publicly leaked
  in pre-existing `plan-2*.md` + `codec_size_report.rs` on the public fork — the
  user accepted that as a one-off; **do not re-flag or remediate it**, just keep
  YOUR outputs clean and OPSEC-scan diffs before any push.)
- **Network is sandbox-blocked** in this environment; `git push` needs
  `dangerouslyDisableSandbox: true`. Local git works normally.
- **Commits:** end messages with `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`. Fork-local fixes are the agreed mode for this
  remediation (a scoped exception to the RFC-first rule), but the unified MVCC
  model itself IS RFC-worthy (it's the shared primitive) — design doc written.
- **Disk** was tight earlier (cleaned `target/debug/incremental`); use
  `CARGO_INCREMENTAL=0`.
- **Verification bar** before merging a wave: `--all-features` green, clippy zero
  new warnings in changed files, `grafeo` `lpg`+`analytics` profiles and
  `grafeo-wasm` compile.

## Process

The user works brainstorm → spec → plan → TDD (one commit per finding), merges
waves to `integration` locally, pushes when ready. They value the *long-term
shape* over tactical patches (they course-corrected the isolation approach
twice) — bring architectural judgment, surface tradeoffs, don't just patch
symptoms.
