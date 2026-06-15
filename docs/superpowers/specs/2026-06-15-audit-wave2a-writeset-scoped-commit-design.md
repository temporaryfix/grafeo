# Audit Wave 2a — Write-set-scoped commit/rollback

## Context

Wave 2 (transaction isolation + O(N) commit) is the engine-core remediation's
headline rework. Two gating decisions are locked with the user:

1. **Isolation model:** extend the existing MVCC (version chains, `visible_to`,
   PENDING, finalize) to properties and labels via snapshot-aware reads, using
   the `epoch`+`transaction_id` the operators already carry.
2. **Storage:** a single always-versioned property/label representation
   (`VersionLog`-based, PENDING for uncommitted), collapsing the non-temporal
   `#[cfg]` fork (also retires audit finding 2d).

Wave 2 is delivered as three spec → plan → PR sub-units:

- **2a (this spec):** write-set-scoped commit/rollback — foundational,
  behavior-preserving, independently shippable. Do first.
- **2b:** property/label/delete isolation (the storage collapse + snapshot reads
  + PENDING deletes).
- **2c:** real SSI (wire `record_read`, flip the Wave 1 Serializable rejection).

## Problem

`session.commit()` and `rollback()` call, per touched graph,
`finalize_version_epochs(tx, commit_epoch)` / `discard_uncommitted_versions(tx)`,
each of which iterates **every** node and edge version chain in the store under a
write lock. On a large graph this makes committing a one-row transaction an
O(all entities) operation that blocks all writers. The transaction manager
already tracks a per-transaction write-set, and savepoint rollback already scopes
its work via `discard_entities_by_id`; commit and full rollback should do the
same.

### Critical finding (gates the scoping)

The write-set is **incomplete**. Mutation *operators* (the `session.execute`
path) record every touched entity via `WriteTracker::record_node_write` /
`record_edge_write`. But the **session-direct mutation APIs**
(`Session::create_node`, `create_node_with_props`, `create_edge`,
`create_edge_with_props`, `set_node_property`, `set_edge_property`, `add_label`,
`remove_label`, `delete_node`, `delete_edge`, and the `database/crud.rs`
equivalents) call the store with the transaction id but never record into the
write-set. The current full-scan commit works *because* it scans all chains and
does not rely on the write-set. Naive write-set scoping would silently leave
these entities' PENDING versions unfinalized — committed but invisible — a data
corruption bug. **The write-set must be made complete before scoping.**

A related but separate latent gap: because session-direct mutations skip
`record_write`, they also bypass write-write conflict detection. **Closing that
conflict-detection gap is out of scope for 2a** (2a is behavior-preserving for
conflict detection — see below); it is tracked for a later sub-unit.

## Design

### Make the write-set complete (behavior-preserving)

Add `TransactionManager::record_entity(tx, entity)` — a conflict-free insert into
the transaction's write-set (no first-writer-wins scan, no error return). Call it
from every session-direct mutator so the write-set captures every entity the
transaction touches, regardless of path.

Using a conflict-free insert (rather than the conflict-checking `record_write`)
is deliberate: it makes the write-set complete for **scoping** without changing
conflict-detection behavior. Session-direct creates allocate fresh ids and cannot
write-write-conflict; session-direct modifies keep their current
(non-conflict-detecting) behavior. Closing the conflict-detection gap is a
separate, later change. Several session-direct mutators (`create_node -> NodeId`,
`delete_node -> bool`) cannot propagate a conflict `Result` without an API break,
which reinforces keeping conflict detection out of 2a.

### Scope commit

Add `LpgStore::finalize_entities_by_id(tx, commit_epoch, node_ids, edge_ids)`
(and the `tiered-storage` variant), mirroring the existing
`discard_entities_by_id`. It finalizes only the named entities' version chains
(`chain.finalize_epochs`), then performs the existing bulk temporal
property/label `finalize_pending` (unchanged — per-entity property finalize
belongs to 2b, which restructures property storage), then `sync_epoch`.

In `session.commit()`, replace the per-graph `finalize_version_epochs(tx,
commit_epoch)` with: fetch the transaction's write-set from the manager, split
into node ids and edge ids, and call `finalize_entities_by_id(tx, commit_epoch,
&node_ids, &edge_ids)` per touched graph (a store ignores ids it does not hold).

### Scope rollback and the commit-conflict path

Replace `discard_uncommitted_versions(tx)` in `rollback_inner` and in the
commit-conflict branch (Wave 1 F1) with: `discard_entities_by_id(tx, &node_ids,
&edge_ids)` (the existing scoped chain discard) followed by
`rollback_transaction_properties(tx)` (the property undo replay, already
transaction-scoped via the undo log). The full-scan `discard_uncommitted_versions`
is retained only for callers that have no write-set (none remain on the hot path
after this change).

### Behavior preservation

The set of entities finalized/discarded is identical to today's full scan,
provided the write-set is complete (which the first design point ensures and the
tests below verify). The temporal property/label finalize stays bulk, so temporal
behavior is unchanged. No conflict-detection behavior changes.

## Sequencing (one commit per step, TDD)

1. `TransactionManager::record_entity` + unit test (entity appears in write-set,
   no conflict raised even when another active tx holds the same entity).
2. `LpgStore::finalize_entities_by_id` (+ tiered variant) + unit test (finalizes
   only the named entities; PENDING of a named entity becomes visible, an
   un-named PENDING entity does not).
3. Wire session-direct mutators (and `crud.rs`) to `record_entity` + test (after
   a session-direct create/set/delete in a transaction, the entity is in the
   write-set).
4. Scope commit, rollback, and the conflict path to the write-set + a
   comprehensive test: a transaction performing every mutation type **via both
   `session.execute` and the session-direct APIs**, committed, is fully visible
   to a fresh session; the same transaction rolled back leaves no trace. (A gap
   in write-set completeness surfaces here as committed-but-invisible data.)

## Verification bar

- `cargo test --all-features` (core + engine) green;
- `grafeo` `lpg` and `analytics` profiles and `grafeo-wasm` compile;
- `clippy` adds zero new warnings.

## Out of scope

- Property/label snapshot reads and the storage collapse (2b); PENDING deletes
  (2b); real SSI (2c).
- Closing the session-direct conflict-detection gap (session-direct modifies do
  not participate in write-write conflict detection) — noted, deferred.
- Per-entity temporal property finalize (2b restructures property storage).
