# Design — Unified MVCC increment 2: complete isolation + Serializable (OCC)

**Status:** Design / pre-RFC. Builds on increment 1 (`2026-06-15-unified-mvcc-isolation-design.md` + `…-increment-1.md`, landed on `integration` @ `3c993082`). Foundational: extends the shared MVCC/visibility primitive and turns on the Serializable isolation level.

## 1. Goal & frame

Increment 1 made **property** writes and **node** deletes snapshot-isolated through one read accessor, and verified the dirty-read probes pass. This increment lands **the most correct, complete, cohesive version of the isolation + serializability model** on the primary store paths (`LpgStore` + `LayeredStore`):

- Finish the isolation model **uniformly** — labels and edge-deletes get the same snapshot treatment properties and node-deletes already have, so the engine no longer half-honors MVCC for "data".
- Complete read-set and write-set tracking, and **turn on Serializable** by feeding the conflict-validation engine that already exists in `manager.rs` (sound OCC read-set validation, currently unfed and rejected at `begin`).

**Why these specific pieces and not others:** the dirty-read root cause was "reads honor existence-MVCC but read property **and label** data at latest." Increment 1 fixed properties; labels are the symmetric half of the *same* bug. Sound Serializable requires a *complete* read-set (every read tracked → all reads routed) and a *complete* write-set (every write tracked → including edge-deletes and labels). So "finish isolation" and "land serializable" are the same body of work approached from two directions; doing them together is what makes it cohesive rather than a sequence of patches.

### In scope (this increment)
- **A. Label isolation** — transactional `SET`/`REMOVE` label buffered into a per-tx delta; label reads (`has_label`, `labels(n)`, label scans) snapshot-consistent; commit applies, rollback drops.
- **B. Edge-delete / DETACH adjacency isolation** — transactional edge deletes use `PENDING` adjacency tombstones deferred to commit (mirrors node-delete), and are tracked.
- **C. MERGE read-your-writes completeness** — `find_matching_edge` routes through the accessor; the MERGE candidate set sees same-tx creates.
- **D. Complete read routing + `record_read`** — thread `(epoch, tx)` through every remaining `tx=None` read site; record the read-set (entity-granular, Serializable-only) via a `ReadTracker`.
- **E. Complete write-set** — derived from the store-level per-tx chokepoints (complete by construction).
- **F. Flip Serializable on** — remove the `begin`-time rejection; the existing commit validation does the rest; write-skew tests.

### Out of scope (separate, established tracks — not isolation/serializability-correctness)
- **Storage-engine convergence** — unify the delta (retire the temporal `VersionLog` + non-temporal write-through+undo onto one delta), retire the `temporal` feature flag, tier/compaction. Physical-layout optimization; orthogonal to correctness; can land later without touching the model.
- **Full SSI** (Cahill dangerous-structure detection) — the agreed follow-on optimization over OCC (fewer aborts, more code).
- **CDC/WAL wrapper isolation** — those deployment wrappers stay on write-through (preserving change-event/WAL recording, per increment 1's Option B). Serializable + isolation are scoped to `LpgStore` + `LayeredStore`; CDC/WAL isolation is a separate completeness item (it needs buffered-write event/WAL recording, a different subsystem). **Because their writes remain write-through (uncommitted data observable), Serializable is gated to the isolated store paths and stays rejected on CDC/WAL-wrapped stores** until their isolation lands — enabling it there would be unsound (claiming serializable while permitting dirty reads of uncommitted writes). The conflict-*validation* is store-agnostic, but the read-time *isolation* it presupposes is not yet there for those wrappers.

---

## 2. The model (the invariant we're completing)

> Every read of existence/labels/properties resolves at the transaction's snapshot through one accessor; every transactional write (create / set-prop / remove-prop / add-label / remove-label / node-delete / edge-delete) lands in a per-transaction delta invisible to others until commit; for a **Serializable** transaction, every read is recorded and validated at commit against the write-sets of transactions that committed after it started.

Increment 1 established this for existence + properties + node-deletes. This increment extends it to **labels + edge-deletes** and adds the **read-set/validation** layer. The same primitives are reused throughout: the per-tx delta (`tx_property_overlay` → generalize to also hold labels), the store-level pending chokepoints (`pending_tx_creates`/`pending_tx_deletes` → add edge-deletes), the snapshot accessor, and the commit/rollback `apply`/`drop` wiring.

---

## 3. Part A — Label isolation

**Current state.** `node_labels: FxHashMap<NodeId, FxHashSet<u32>>` (non-temporal) / `VersionLog<FxHashSet<u32>>` (temporal); `label_index: Vec<FxHashMap<NodeId,()>>` (label_id → nodes, for scans). Transactional writes go through `add_label_versioned`/`remove_label_versioned` (write-through + undo log). Reads — `node.has_label(l)` (filter/merge), `labels(n)` (filter), and label scans over `label_index` — all read committed/latest. Same dirty-read shape as properties had.

**Design (mirror properties).**
- Extend the per-tx delta to carry **label ops**, node-scoped: the delta struct grows `node_labels: HashMap<(NodeId, label_id), LabelOp>` with `LabelOp::Add` / `LabelOp::Remove` (the exact map shape is an implementation detail; the design decision is that label ops live in the *same* per-tx delta object as property ops, so commit/rollback/savepoint stay single-sourced).
- Add the snapshot label accessor: `read_node_labels_visible(id, epoch, Option<tx>) -> FxHashSet<u32>` — committed label set merged with the writing tx's buffered label ops. Add a `node_has_label_visible(id, label_id, epoch, tx)` convenience.
- **Route every label read** through it: `has_label`/`labels(n)` in `filter.rs`/`merge.rs` and the whole-entity `NodeResolve` label materialization (which increment 1 takes from the resolved node — must now take from the accessor for the writer).
- **Label scans** (`MATCH (:Foo)`) read `label_index`, which stays committed-only (uncommitted label-adds are NOT inserted into `label_index`, so other sessions never scan them). The **writing tx's** own label scan must reflect its buffered adds/removes: apply the same **writer-only bypass** increment 1 used for property index/zone-map — when `transaction_id.is_some()`, a label scan falls back to a committed scan merged with the tx's label delta. (Never insert uncommitted labels into `label_index` → no dirty read via scan.)
- Transactional `add_label`/`remove_label` buffer into the delta instead of write-through; commit applies (writes through `add_label`/`remove_label`, updating `label_index`); rollback drops.

**Acceptance:** an uncommitted `SET n:Secret` / `REMOVE n:Public` is invisible to other sessions (via `has_label`, `labels(n)`, and `MATCH (:Secret)` scans) but visible to the writer; commit makes it visible; rollback restores. (Symmetric tracked tests to increment 1's property probes.)

---

## 4. Part B — Edge-delete / DETACH adjacency isolation

**Current state.** Node deletes are isolated (increment 1: `PENDING deleted_epoch` + `pending_tx_deletes` + deferred label-index removal). But transactional `DETACH DELETE` of a node with edges, and direct edge deletes, go through `delete_node_edges`/`delete_edge` using `TransactionId::SYSTEM` + **eager** `batch_mark_deleted` on adjacency — so an uncommitted edge delete is globally visible (dirty write) and its tombstones aren't rolled back transactionally.

**Design (mirror node-delete).**
- Make transactional edge deletion stamp the edge version chain `deleted_epoch = PENDING` (it already has version chains) and **defer the adjacency tombstone** to commit: record `(src, edge_id, dst)` in a store-level `pending_tx_edge_deletes` keyed by tx (mirroring `pending_tx_deletes` for nodes).
- Add `finalize_edge_deletes_by_id(tx, commit_epoch, …)` (stamp commit epoch + apply the deferred `forward_adj`/`backward_adj` tombstones) and `rollback_edge_deletes(tx, …)` (`unmark_deleted_by` on the edge chain; nothing to restore since adjacency was never touched).
- `neighbors()`/`edges_from()` already consult version-chain visibility for edges — confirm they honor `PENDING` for the writer vs others (same `visible_to` semantics the node path validated; the `OptionalEpochId::PENDING` sentinel from increment 1 covers tiered).
- Wire into commit/rollback/conflict alongside the node-delete finalize.

**Acceptance:** an uncommitted edge delete (incl. `DETACH DELETE` of a node with edges) is invisible to other sessions' `neighbors`/`edges_from`/edge-property reads but gone for the writer; commit applies; rollback restores. The increment-1 `#[ignore]`d `uncommitted_detach_delete_edges_invisible_to_other_sessions` probe is un-ignored and passes.

---

## 5. Part C — MERGE read-your-writes completeness

**Current state (documented TODOs from increment 1).** `MergeRelationshipOperator::find_matching_edge` reads committed edge properties (not the delta); `MergeOperator::find_matching_node`'s candidate set (`find_nodes_by_properties`) is the committed property index, so a node created earlier in the same tx is not a match candidate → MERGE can create a duplicate.

**Design.**
- Route `find_matching_edge`'s property comparison through `read_edge_property_visible` (thread `epoch`/`tid` there, mirroring the node-side fix already in `find_matching_node`).
- For the candidate set: when `transaction_id.is_some()`, union the committed-index candidates with the tx's same-tx-created nodes (available from `pending_tx_creates`) before the per-node `read_node_property_visible` filter — so a same-tx create with matching properties is found. (This is the label/property delta-merge applied to MERGE matching.)

**Acceptance:** within one transaction, `MERGE (n {k:1})` twice creates one node; `CREATE (n {k:1}) … MERGE (m {k:1})` matches `n` rather than creating a duplicate; MERGE on an edge whose match-property was `SET` earlier in the tx matches.

---

## 6. Part D — Complete read routing + `record_read`

**The read-set foundation.** Sound OCC serializability requires that **every** entity a Serializable transaction's result depended on is in its read-set. Over-approximating (recording extra) is sound; missing a read is **unsound**. So:

- **Finish read routing:** thread `(epoch, tx)` through every remaining read site that still passes `tx=None` or reads committed directly — `factorized_filter`, `horizontal_aggregate`, `vector_join` (add the snapshot fields these operators lack), `keys()`/`properties()`/`property_values()`/`property_exists()` and label reads in `filter.rs`, and `PropertySource::PropertyAccess` in `mutation.rs`. These are the increment-1-documented `TODO(unified-mvcc)` sites.
- **`record_read` — Decision A (layering).** The accessor is in `grafeo-core`; `TransactionManager` is in `grafeo-engine`; core cannot call the manager. Mirror the existing `WriteTracker` pattern (`transaction/write_tracker.rs`, operators hold `Arc<dyn WriteTracker>` and call it to `record_write`): add a **`ReadTracker`** trait in core, operators call `record_read(entity, epoch)` at each read site; the engine impl forwards to `TransactionManager::record_read`, which inserts into `TransactionInfo.read_set`. The tracker **no-ops for non-Serializable transactions** (one isolation-level lookup) so SI/ReadCommitted pay ~nothing and operators don't thread isolation level.
- **Granularity:** entity-level (`EntityId::Node(id)` / `Edge(id)`) — the standard sound granularity. (Per-property would abort less but isn't needed for soundness and complicates the read-set.) Scans record each emitted entity; filters/projects/expands record each entity whose data they read.

**Why entity-granular over-approximation is acceptable:** it can cause extra aborts (a tx that read node N's `name` and another that wrote N's unrelated `age` would conflict), but never a missed conflict. Tightening to property-granular is a future optimization alongside full SSI.

---

## 7. Part E — Complete write-set (Decision B)

The validation checks `other.write_set.contains(entity)` for each entity in our read-set. If a committed transaction's write-set is **incomplete**, the validation misses a conflict → unsound. Rather than re-thread `record_write` through every operator (the exact fragility Wave 2a fled — MERGE/LOAD DATA/session-direct mutators bypass it), **derive the write-set from the store-level per-tx chokepoints that already exist and are complete by construction:**

```
write_set(tx) = pending_tx_creates(tx)            // nodes + edges created
              ∪ pending_tx_deletes(tx)            // nodes deleted (incr 1)
              ∪ pending_tx_edge_deletes(tx)       // edges deleted (Part B)
              ∪ entities touched in tx_property_overlay(tx)   // prop writes (incr 1)
              ∪ entities touched in the label delta(tx)       // label writes (Part A)
```

Each of those is recorded at the single store chokepoint every such mutation passes through, so the union is the complete set of entities the transaction wrote. Populate `TransactionInfo.write_set` from this union at commit (just before the validation), keeping the existing validation code unchanged. (This also subsumes/aligns the partial `record_write` path; first-writer-wins write-write conflict detection stays as-is for its eager behavior.)

---

## 8. Part F — Turn Serializable on

- Remove the `begin`-time Serializable rejection (the `cf2401c5`/GQL-path guards). Serializable transactions begin normally and carry `IsolationLevel::Serializable`.
- At commit, the **existing** validation (`manager.rs:349-377`) runs: for each tx that committed after our start epoch, if it wrote any entity in our read-set → `SerializationFailure` (abort + the increment-1 conflict-rollback path: discard pending creates/deletes, drop overlays, abort the tx, release GC). Confirm the abort path for a serialization failure reuses the same cleanup as a write-write conflict (no zombie tx / leaked versions — the Wave 1 F1 fix).
- **Tests (the proof):**
  - **Write skew prevented:** two Serializable txns each read entities X and Y, then write disjointly (one writes X, the other Y), both attempt commit → exactly one commits, the other gets `SerializationFailure`. The classic on-call/ balance invariant.
  - **SI still allows write skew:** the same scenario under `SnapshotIsolation` → both commit (proving the levels genuinely differ, and that we didn't accidentally make SI serializable).
  - **Read-only Serializable** doesn't spuriously abort; a Serializable tx whose read-set is untouched by concurrent committers commits.
  - **rw-conflict caught:** Serializable tx reads X; a concurrent tx writes X and commits first; the reader's commit aborts.
  - Increment-1 isolation probes + the new label/edge-delete tests stay green.

---

## 9. Architecture decisions (the two that matter)

- **Decision A — `ReadTracker` trait (core) mirroring `WriteTracker`.** Keeps the `grafeo-core` → `grafeo-engine` layering intact; operators (which already carry `(epoch, tx)` and a write-tracker) gain a read-tracker; the engine forwards to the manager. No-op for non-Serializable. This is the same proven shape as write tracking, so it's low-risk and consistent.
- **Decision B — derive the write-set from store-level chokepoints, not `record_write` coverage.** Reuses Wave 2a's "the store chokepoint is complete by construction" insight. Avoids the operator-bypass fragility entirely and makes write-set completeness an invariant of where mutations are recorded, not of remembering to call `record_write` at every site.

Both decisions deliberately reuse patterns already validated in this codebase rather than introducing new mechanism.

---

## 10. Ordering & decomposition

The parts are ordered so each is independently landable and the riskiest correctness (read-set completeness for soundness) builds on completed isolation:

1. **A (labels)** and **B (edge-deletes)** — complete the isolation model; they also populate the label/edge-delete write-set inputs E needs. Independent of each other; either order.
2. **C (MERGE)** — small, depends on A/B's deltas being present.
3. **D (read routing + `record_read`)** — finish routing first (behavior-preserving, like increment 1's read-routing), then add the `ReadTracker` + `record_read` wiring (delta/read-set empty for non-Serializable ⇒ no behavior change).
4. **E (write-set derivation)** — depends on A/B (their pending sets exist).
5. **F (flip Serializable + tests)** — last; everything it needs (complete read-set, complete write-set) is in place.

This likely warrants **2–3 implementation plans** rather than one (A+B+C isolation-completeness; D+E tracking; F enablement), each its own plan→execute cycle, but designed as one cohesive arc with F as the capstone. Each part keeps the full `--all-features` + `--features full` gates green; F adds the write-skew suite.

---

## 11. Risks & mitigations
- **Missed read site = unsound serializable** (not just a dirty read now — a missed *conflict*). Mitigation: the read-routing completeness sweep is the same checklist as increment 1's, plus a holistic "every execution read records into the read-set" audit; over-approximate freely.
- **Read-set memory / abort rate.** Entity-granular read-sets on large scans can be big and cause many aborts under contention. Mitigation: only for Serializable txns (opt-in); document the OCC abort characteristic; full SSI (follow-on) reduces it. Consider a read-set size cap → escalate to a coarser conflict (table/label-level) if it ever matters (deferred).
- **Label/edge-delete deltas interacting with the property delta at commit/rollback/savepoint.** Mitigation: extend the existing `apply_tx_overlay`/`drop_tx_overlay`/savepoint snapshot to cover labels + edge-deletes uniformly (one delta object), so the commit/rollback/savepoint paths stay single-sourced.
- **Serialization-failure abort cleanup.** Must reuse the Wave 1 F1 conflict-rollback (no zombie tx, GC released). Mitigation: explicit test that a `SerializationFailure` leaves no leaked versions / pinned epoch.

## 12. Acceptance (the increment is done when)
- Labels and edge-deletes are snapshot-isolated (new tracked tests + the un-ignored DETACH-edges probe pass); MERGE read-your-writes holds.
- A `Serializable` transaction prevents write skew; the same scenario under SI does not; read-only Serializable doesn't spuriously abort; rw-conflicts abort with clean cleanup.
- Full `--all-features -p grafeo-core -p grafeo-engine` + `--features full` integration suites green; clippy `-D warnings` clean; `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` compile; OPSEC-clean.
- The isolation/serializability story is uniform across existence, properties, labels, node-deletes, and edge-deletes on `LpgStore` + `LayeredStore`, with storage convergence, full SSI, and CDC/WAL isolation cleanly documented as the remaining separate tracks.
