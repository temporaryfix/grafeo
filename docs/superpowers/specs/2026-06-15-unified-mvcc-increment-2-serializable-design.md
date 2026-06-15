# Design — Unified MVCC increment 2: complete isolation + Serializable (SSI)

**Status:** Design / pre-RFC. Builds on increment 1 (`2026-06-15-unified-mvcc-isolation-design.md` + `…-increment-1.md`, landed on `integration` @ `3c993082`). Foundational: extends the shared MVCC/visibility primitive and lands **concurrent-writer Serializable Snapshot Isolation**.

## 1. Goal & frame

grafeo is a **concurrent-writer** MVCC engine (verified: `begin` never serializes writers; `record_write` first-writer-wins only fires when `active_count > 1`; concurrent-write conflict is tested). This increment makes that concurrency **correct, complete, and fast** by landing the most powerful version of the isolation + serializability model on the primary store paths (`LpgStore` + `LayeredStore`):

- Finish the isolation model **uniformly** — labels and edge-deletes get the same snapshot treatment properties and node-deletes already have.
- Land **Serializable Snapshot Isolation (SSI)** — sound serializability that aborts only genuinely non-serializable transactions, so concurrent writers run concurrently instead of thrashing. The OCC read-set validation already in `manager.rs` is used as a verified intermediate rung, then refined to full SSI.
- Make the concurrent-writer path **scale** — incremental conflict detection (no global per-commit scan), a sharded read-registry, and a conflict-granularity knob.

**Why SSI, not OCC:** OCC aborts on any read-write conflict; under concurrent-writer contention that defeats the point of concurrent writers. SSI (Cahill et al., "Serializable Snapshot Isolation in PostgreSQL") aborts only on a **dangerous structure** (a pivot transaction with both an inbound and an outbound rw-antidependency where the outbound target commits first), and detects conflicts incrementally at read/write time rather than scanning all committers under a global lock at commit. So SSI is simultaneously the correctness target and the performance answer.

### In scope (this increment)
- **A. Label isolation** — transactional `SET`/`REMOVE` label buffered into the per-tx delta; label reads (`has_label`, `labels(n)`, label scans) snapshot-consistent; commit applies, rollback drops.
- **B. Edge-delete / DETACH adjacency isolation** — transactional edge deletes use `PENDING` adjacency tombstones deferred to commit (mirrors node-delete), and are tracked.
- **C. MERGE read-your-writes completeness** — `find_matching_edge` through the accessor; the MERGE candidate set sees same-tx creates.
- **D. Complete read routing + `record_read` + read-registry** — thread `(epoch, tx)` through every remaining `tx=None` read site; record reads into the tx read-set *and* a shared entity-keyed read-registry (so a writer can find concurrent readers of the version it overwrites).
- **E. Complete write-set** — derived from the store-level per-tx chokepoints (complete by construction).
- **F. Serializable, staged** — F1: feed the existing OCC read-set validation → *sound* serializable (verified milestone); F2: refine to full SSI (in/out conflict flags + dangerous-structure pivot abort).
- **G. Performance** — incremental detection (no global commit scan), sharded read-registry, conflict-granularity knob (entity-level default; property-level to cut false aborts), read-registry GC bounded by the active-tx horizon.

### Out of scope (separate, established tracks)
- **Storage-engine convergence** — unify the delta (retire the temporal `VersionLog` + non-temporal write-through+undo onto one delta), retire the `temporal` flag, tier/compaction. Analytical/memory performance; orthogonal to the concurrency model; lands later without touching correctness.
- **CDC/WAL wrapper isolation** — those wrappers stay write-through (preserving change-event/WAL recording). **Because their writes remain write-through (uncommitted data observable), Serializable is gated to the isolated store paths (`LpgStore` + `LayeredStore`) and stays rejected on CDC/WAL-wrapped stores** until their isolation lands — enabling it there would be unsound. The conflict-detection is store-agnostic, but the read-time isolation it presupposes is not yet there for those wrappers.

---

## 2. The model

> Every read of existence/labels/properties resolves at the transaction's snapshot through one accessor; every transactional write lands in a per-transaction delta invisible to others until commit; for a **Serializable** transaction, every read is recorded (in its read-set and a shared read-registry) and every read-write antidependency between concurrent transactions is tracked, so a transaction is aborted **iff** it participates in a dangerous structure that could produce a non-serializable schedule.

Increment 1 established the snapshot/delta/accessor for existence + properties + node-deletes. This increment extends it to **labels + edge-deletes**, adds the **read-set + registry + SSI conflict tracking**, and turns Serializable on. The same primitives are reused: the per-tx delta (generalized to hold labels), the store-level pending chokepoints (add edge-deletes), the snapshot accessor, and the commit/rollback/savepoint `apply`/`drop`/snapshot wiring.

---

## 3. Part A — Label isolation

**Current state.** `node_labels: FxHashMap<NodeId, FxHashSet<u32>>` (non-temporal) / `VersionLog<FxHashSet<u32>>` (temporal); `label_index: Vec<FxHashMap<NodeId,()>>` for scans. Transactional writes go through `add_label_versioned`/`remove_label_versioned` (write-through + undo log). Reads — `has_label`, `labels(n)`, label scans — read committed/latest (same dirty-read shape properties had).

**Design (mirror properties).**
- Extend the per-tx delta to carry **label ops**, node-scoped: `node_labels: HashMap<(NodeId, label_id), LabelOp>` with `LabelOp::Add` / `LabelOp::Remove`, in the *same* delta object as property ops (so commit/rollback/savepoint stay single-sourced).
- Add `read_node_labels_visible(id, epoch, Option<tx>) -> FxHashSet<u32>` (committed label set merged with the writing tx's buffered ops) + a `node_has_label_visible` convenience. Route every label read through it (`has_label`/`labels(n)` in `filter.rs`/`merge.rs`, and `NodeResolve` label materialization).
- **Label scans** (`MATCH (:Foo)`) keep reading the committed-only `label_index`; uncommitted adds are never inserted there → no dirty read via scan. The **writing tx's** own label scan applies the increment-1 **writer-only bypass** (`transaction_id.is_some()` → committed scan merged with the tx's label delta).
- Transactional `add_label`/`remove_label` buffer into the delta; commit applies (write-through + `label_index` update); rollback drops.

**Acceptance:** uncommitted `SET n:Secret` / `REMOVE n:Public` invisible to other sessions (via `has_label`, `labels(n)`, `MATCH (:Secret)`), visible to the writer; commit makes it visible; rollback restores.

---

## 4. Part B — Edge-delete / DETACH adjacency isolation

**Current state.** Node deletes are isolated (increment 1). Transactional `DETACH DELETE` of a node with edges and direct edge deletes go through `delete_node_edges`/`delete_edge` using `TransactionId::SYSTEM` + **eager** `batch_mark_deleted` — uncommitted edge deletes are globally visible (dirty write) and not transactionally rolled back.

**Design (mirror node-delete).**
- Transactional edge deletion stamps the edge version chain `deleted_epoch = PENDING` and **defers the adjacency tombstone** to commit: record `(src, edge_id, dst)` in a store-level `pending_tx_edge_deletes` keyed by tx (mirroring `pending_tx_deletes`).
- Add `finalize_edge_deletes_by_id(tx, commit_epoch, …)` (stamp commit epoch + apply deferred `forward_adj`/`backward_adj` tombstones) and `rollback_edge_deletes(tx, …)` (`unmark_deleted_by`; adjacency untouched so nothing to restore). Reuse the `OptionalEpochId::PENDING` sentinel from increment 1 for tiered.
- Confirm `neighbors()`/`edges_from()` honor edge-chain `PENDING` visibility (writer-vs-others) — same `visible_to` semantics the node path validated.
- Wire into commit/rollback/conflict alongside node-delete finalize.

**Acceptance:** uncommitted edge delete (incl. `DETACH DELETE` of a node with edges) invisible to other sessions' `neighbors`/`edges_from`/edge-property reads, gone for the writer; commit applies; rollback restores. The increment-1 `#[ignore]`d `uncommitted_detach_delete_edges_invisible_to_other_sessions` probe un-ignored and passing.

---

## 5. Part C — MERGE read-your-writes completeness

**Current state (increment-1 TODOs).** `find_matching_edge` reads committed edge properties (not the delta); `find_matching_node`'s candidate set (`find_nodes_by_properties`) is the committed property index, so a same-tx-created node is not a match candidate → MERGE can duplicate.

**Design.** Route `find_matching_edge`'s property comparison through `read_edge_property_visible` (thread `epoch`/`tid`, mirroring the node-side fix). For the candidate set: when `transaction_id.is_some()`, union committed-index candidates with the tx's same-tx-created nodes (from `pending_tx_creates`) before the per-node `read_node_property_visible` filter.

**Acceptance:** within one tx, `MERGE (n {k:1})` twice creates one node; `CREATE (n {k:1}) … MERGE (m {k:1})` matches `n`; MERGE on an edge whose match-property was `SET` earlier in the tx matches.

---

## 6. Part D — Complete read routing + `record_read` + read-registry

Sound SSI requires that **every** entity a Serializable transaction read is observable two ways: in the tx's own read-set (for validation), and in a **shared read-registry keyed by entity** (so that when another tx *writes* that entity, it can discover the concurrent reader and record the rw-antidependency `reader →rw writer`). Over-approximating reads is sound; missing one is unsound.

- **Finish read routing:** thread `(epoch, tx)` through every remaining read site still passing `tx=None` / reading committed directly — `factorized_filter`, `horizontal_aggregate`, `vector_join` (add the snapshot fields these lack), `keys()`/`properties()`/`property_values()`/`property_exists()` + label reads in `filter.rs`, `PropertySource::PropertyAccess` in `mutation.rs` (the increment-1-documented `TODO(unified-mvcc)` sites).
- **`record_read` — Decision A (layering).** Mirror the `WriteTracker` pattern: a `ReadTracker` trait in `grafeo-core`; operators call `record_read(entity, epoch)` at each read site; the `grafeo-engine` impl forwards to `TransactionManager::record_read`, which (a) inserts into `TransactionInfo.read_set` and (b) registers the read in the **shared read-registry**. No-op for non-Serializable txns (one isolation lookup) so SI/RC pay ~nothing.
- **Read-registry (SIREAD locks).** A concurrency-friendly structure mapping `EntityId → set of (reader tx)` for *active* Serializable readers, **sharded** to avoid a global lock. Entries are GC'd when the reader commits/aborts and no longer needed (below the active-tx horizon). This is what makes write-time rw-edge detection cheap and keeps the commit path free of a global scan.
- **Granularity:** entity-level by default (sound). A property-level mode (registry keyed by `(EntityId, PropertyKey)`) is the **performance knob** in Part G to cut false antidependencies for wide-node workloads.

---

## 7. Part E — Complete write-set (Decision B)

SSI needs each transaction's complete write-set to (a) detect `reader →rw writer` (a write must find concurrent readers of the entity) and (b) for the OCC fallback (F1). Rather than re-thread `record_write` through every operator (the fragility Wave 2a fled), **derive the write-set from the store-level chokepoints, complete by construction:**

```
write_set(tx) = pending_tx_creates(tx) ∪ pending_tx_deletes(tx) ∪ pending_tx_edge_deletes(tx)
              ∪ entities touched in tx_property_overlay(tx) ∪ entities touched in the label delta(tx)
```

Populate `TransactionInfo.write_set` from this union; first-writer-wins write-write conflict stays as-is for its eager behavior.

---

## 8. Part F — Serializable, staged (OCC rung → full SSI)

**F1 — sound serializable via the existing OCC validation (verified milestone).** Feed `manager.rs:349-377`: with read-set + write-set now populated, remove the `begin`-time Serializable rejection; the existing backward validation (for each post-start committer, if it wrote anything in our read-set → `SerializationFailure`) becomes live. This is *sound* serializable immediately — a checkpoint we test (write-skew prevented) before optimizing. Confirm the abort reuses the Wave-1-F1 conflict-rollback (no zombie tx / leaked versions / pinned epoch).

**F2 — full SSI (the target).** Replace the backward global scan with incremental **rw-antidependency tracking**:
- Per active Serializable tx, two flags: `in_conflict` (a concurrent tx has an rw-edge *into* it) and `out_conflict` (it has an rw-edge *out* to a concurrent tx).
- **Detect `T_reader →rw T_writer`** at *write* time: when `T_writer` writes entity E, consult the read-registry for concurrent readers of E; for each such `T_reader`, set `T_reader.out_conflict` and `T_writer.in_conflict`.
- **Detect `T_reader →rw T_writer`** at *read* time too: when `T_reader` reads E and a concurrent `T_writer` has already written a newer version (visible in the version chain / write-set), set `T_reader.out_conflict` and `T_writer.in_conflict`.
- **Dangerous structure → abort:** when a transaction has both `in_conflict` and `out_conflict` (it is a pivot), it can anchor a non-serializable cycle; abort it (or the appropriate participant) per the standard SSI rule (abort the pivot, preferring the one whose out-neighbor has committed). Aborts surface as `SerializationFailure` with the same clean rollback.
- Commit becomes O(1) flag checks — no global committer scan.

**Acceptance (F):** write-skew prevented under Serializable; the same scenario under SI commits both (levels genuinely differ); **read-only and non-conflicting concurrent writers do not abort** (the SSI win over OCC — explicit tests that benign concurrent writers all commit); rw-conflict aborts with clean cleanup; no leaked versions/epoch on a serialization abort.

---

## 9. Part G — Performance (make concurrent writers actually scale)

- **No global commit serialization:** SSI's incremental detection means commit checks two flags, not an O(committed) scan under a global lock — the main concurrent-writer throughput win.
- **Sharded read-registry:** shard by entity hash so concurrent readers/writers rarely contend; size/GC the registry against the active-tx horizon (reuse `min_active_epoch`).
- **Conflict-granularity knob:** entity-level (default, sound, cheaper tracking) vs property-level (registry keyed by `(EntityId, PropertyKey)` — fewer false antidependencies for wide nodes with disjoint hot columns, at higher tracking cost). Make it a per-database/session policy; default entity-level.
- **Read-set/registry cost is Serializable-only:** SI/ReadCommitted transactions never touch the registry or read-set — the fast path is unchanged.
- Benchmark: concurrent disjoint-write throughput (should scale ~linearly, unlike OCC), and serialization-abort rate under a contended write-skew workload (should approach the theoretical minimum — only true dangerous structures).

---

## 10. Architecture decisions
- **A — `ReadTracker` trait (core) mirroring `WriteTracker`.** Keeps the core→engine layering; operators (already carrying `(epoch, tx)` + a write-tracker) gain a read-tracker; engine forwards to the manager + registry; no-op for non-Serializable.
- **B — store-derived write-set** from the chokepoints, not `record_write` coverage. Reuses Wave 2a's complete-by-construction insight.
- **C — incremental SSI over a sharded read-registry**, not a global commit-time validation scan — the decision that makes concurrent-writer serializable *fast*, not just correct.

## 11. Ordering & decomposition (one cohesive arc, ~3 plans)
1. **A (labels) + B (edge-deletes) + C (MERGE)** — complete the isolation model; populate the label/edge-delete write-set inputs. (Plan 1.)
2. **D (read routing + `record_read` + read-registry) + E (write-set)** — the tracking foundation; routing is behavior-preserving, registry/read-set inert for non-Serializable. (Plan 2.)
3. **F1 (OCC rung) → F2 (SSI) + G (performance)** — enable + refine + harden; F2 is the capstone. (Plan 3.)

Each part keeps the full `--all-features` + `--features full` gates green; F adds the write-skew + concurrent-no-false-abort suites; G adds the concurrency benchmarks.

## 12. Risks & mitigations
- **Missed read site = unsound SSI** (a missed antidependency, not just a dirty read). Mitigation: the read-routing completeness sweep + a holistic "every execution read hits the registry" audit; over-approximate.
- **SSI complexity / correctness.** Cahill SSI is subtle (pivot detection, abort choice, the read-time vs write-time edge symmetry). Mitigation: land F1 (sound OCC) first as a verified rung; build F2 behind the same `SerializationFailure` contract with a property-test/model-check of the dangerous-structure rule against known SSI test cases (write skew, the "batch processing" anomaly, read-only-anomaly).
- **Read-registry memory & GC.** Bounded by active Serializable txns × their read-sets; GC against the horizon. A read-set/registry cap → escalate to coarser conflict (label/table-level) is a documented safety valve.
- **False-abort rate.** Entity-granular may over-abort wide-node disjoint writes; the property-level knob (G) is the mitigation; benchmark-driven.
- **Serialization-abort cleanup.** Reuse Wave-1-F1 conflict-rollback; explicit no-leak test.

## 13. Acceptance (done when)
- Labels and edge-deletes are snapshot-isolated (new tracked tests + the un-ignored DETACH-edges probe); MERGE read-your-writes holds.
- **Serializable is SSI:** write-skew and the standard SSI anomalies (read-only anomaly, batch-processing) are prevented; the same scenarios under SI are not; **benign concurrent writers and read-only Serializable transactions do not abort**; rw-conflicts abort with clean cleanup and no leaks.
- Concurrent disjoint-write throughput scales (benchmark); abort rate approaches the dangerous-structure minimum.
- Full `--all-features -p grafeo-core -p grafeo-engine` + `--features full` integration green; clippy `-D warnings` clean; `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` compile; OPSEC-clean.
- The isolation/serializability story is uniform across existence, properties, labels, node-deletes, edge-deletes on `LpgStore` + `LayeredStore`, with storage convergence and CDC/WAL isolation cleanly documented as the remaining separate tracks.
