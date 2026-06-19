# Multi-Granularity SIREAD Escalation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound the SSI read-recording cost + read-set memory by promoting per-row read-set entries to per-label/per-type when a Serializable transaction reads many rows — PostgreSQL-style multi-granularity SIREAD with promotion. Default-on, one threshold `T`.

**Architecture:** Fine recording stays at the F1 store chokepoint (unchanged); the store's *label/type-scan accessors* tag their reads with the scan predicate (label/type); the manager's read-set buckets fine entries per predicate and promotes `Node→Label` / `Edge→RelType` at `T`, dropping the fine entries; writers check the coarse key (and new-node/label-add records the coarse write — the phantom). Sound by construction: coarsening only adds false conflicts.

**Tech Stack:** Rust. `grafeo-engine` (`transaction/manager.rs`, `read_registry.rs`, `read_tracker.rs`, session config), `grafeo-core` (`graph/lpg/store/` chokepoints + label/type-scan accessors, `execution/operators/mod.rs` `ReadTracker`/`WriteTracker`). `CARGO_INCREMENTAL=0`.

**Spec:** `docs/superpowers/specs/2026-06-19-ssi-read-granularity-escalation-design.md`. Read it first — especially §4 (architecture), §4e (the phantom), §6 (why it can't break serializability).

---

## Orientation & reuse templates

On `integration` @ `8f4765cb`+. **Reuses the merged `EntityId::Index` / Part-G machinery verbatim:**
- `EntityId` enum (`transaction/manager.rs:115`, `#[non_exhaustive]`, variants Node/Edge/Index); `IndexId::for_index` (hash) — `EntityId::Label`/`RelType` thread identically (commit `2d18e903` is the template).
- `TransactionInfo { read_set: HashSet<(EntityId, PropTag)>, write_set: HashSet<(EntityId, PropTag)>, … }` (`manager.rs:180-212`).
- `record_read(tx, entity, tag)` (`manager.rs:484`): inserts `(entity, tag)` into `read_set`, gathers concurrent writers (active + committed-after-start) whose `write_set` has a `prop_compatible` entry for `entity`, registers in `read_registry`, sets rw-edges.
- `record_write(tx, entity, tag)` (`manager.rs:309`): `readers_of_compatible(entity, tag)` → rw-edges.
- `read_registry.record_reader(entity, tx, tag)` / `readers_of_compatible(entity, tag)` (`read_registry.rs`).
- Store chokepoint `record_read_node(tx, id)` (`graph/lpg/store/mod.rs:1187`) → `ReadTracker::record_read_node` (`execution/operators/mod.rs:212`) → `manager.record_read(tx, id, tag)` (`read_tracker.rs:57`). Label scans loop `record_read_node` in `schema.rs:380/428/733`.
- `ConflictGranularity` policy (Part-G, `manager.rs`) — the precedent for a per-session, begin-time-latched knob; `T` rides alongside it.

**Gate (each task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full --test serializable` + `--test serializable_tracking`. Clippy `-D warnings`; profiles + wasm. Hygiene: `rustfmt --edition 2024` per file (NOT `cargo fmt`); `git status` clean except untracked `ce/` (never add); `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. OPSEC.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `transaction/manager.rs` | `EntityId::Label`/`RelType` + `LabelId`/`RelTypeId`; the read-set buckets + promotion; writer multi-granularity check; `T` field | Modify |
| `transaction/read_registry.rs` | (additive — `EntityId` is opaque key; verify Label/RelType participate) | Verify/Modify |
| `transaction/read_tracker.rs` / `write_tracker.rs` | predicate-carrying read bridge; coarse-write bridge | Modify |
| `execution/operators/mod.rs` | `ReadTracker::record_read_node_in_label` (+ edge) default methods | Modify |
| `graph/lpg/store/mod.rs` | store chokepoint `record_read_node_in_label`; `record_write` label/type fan-out | Modify |
| `graph/lpg/store/schema.rs` + scan accessors | label/type-scan loops call the predicate-tagged chokepoint | Modify |
| `graph/lpg/store/node_ops.rs` / `edge_ops.rs` | create-node/add-label/new-edge record the coarse write (phantom) | Modify |
| session config | the `T` setter (default 256) | Modify |
| `tests/serializable.rs` | acceptance | Modify |
| `benches/serializable_overhead.rs` | re-measure | (run) |

---

## Task 1: `EntityId::Label` + `EntityId::RelType` conflict keys

**Files:** `transaction/manager.rs` (+ `read_registry.rs` if any exhaustive match).

- [ ] **Step 1 (failing test):** manager test — `record_read(tx1, EntityId::Label(LabelId::new(7)), None)` + concurrent `record_write(tx2, EntityId::Label(LabelId::new(7)), None)` form an rw-edge (reuse the `EntityId::Index` edge-direction test shape); different `LabelId`s don't conflict; `EntityId::RelType` symmetric. FAIL (variant missing).
- [ ] **Step 2:** Add `Label(LabelId)` and `RelType(RelTypeId)` to `EntityId` (`#[non_exhaustive]`). Define `LabelId(u32)` / `RelTypeId(u32)` newtypes wrapping the label/relationship-type registry id (use the existing `LabelId` type from the label registry if one exists — grep `label_registry` / `LabelId`; otherwise a `u32` newtype + `From`). Thread through every internal `match`/use of `EntityId` exactly like `Index` (most sites treat it as an opaque `Hash + Eq` key — add arms only where a `match` is exhaustive). `PropTag` for a coarse key is always `None`.
- [ ] **Step 3:** Run → PASS. Full gate (additive — Node/Edge/Index paths unchanged; nothing records Label/RelType yet). Commit (`feat(ssi): EntityId::Label + EntityId::RelType conflict keys`).

---

## Task 2: Writer multi-granularity check + phantom coarse-write

**Files:** `transaction/manager.rs`, `graph/lpg/store/{mod.rs,node_ops.rs,edge_ops.rs}`, `write_tracker.rs`.

This must land **before** promotion (Task 3) so that when promotion drops a fine `Node(n)` entry, a writer of `n` still finds the reader via the coarse `Label(L)` key.

- [ ] **Step 1 (test):** manager-level — an escalated-style reader records `EntityId::Label(L)` read; a writer that records `EntityId::Node(n)` **where `n` has label `L`** must form the rw-edge with that reader (i.e. `record_write` for a node fans out to check `Label(L)` readers). Drive it by calling a new `manager.record_node_write(tx, n, labels: &[LabelId], tag)` that records `Node(n)` **and** checks/records `Label(L)` for each label. FAIL.
- [ ] **Step 2:** Add `manager.record_node_write(tx, node, labels, tag)`: do the existing `record_write(Node(node), tag)` AND, for each `label` in `labels`, `record_write(Label(label), None)` (records into `write_set` + `readers_of_compatible(Label(label), None)` → rw-edges with escalated readers). Symmetric `record_edge_write(tx, edge, rel_type, tag)` → `record_write(Edge(edge), tag)` + `record_write(RelType(rel_type), None)`. **Route the store's write recording** (`record_write_node`/the write-tracker bridge) through these so a `SET`/`REMOVE`/label-change on node `n` supplies `n`'s labels. **Phantom:** node creation (`node_ops.rs` create) and `add_label` must also call `record_node_write` with the (new) label so a concurrent escalated `:L` reader conflicts; new-edge creation records `RelType(T)`. (Look up the node's labels at the write site — the store has them.)
- [ ] **Step 3:** PASS; full gate (behavior-preserving: no `Label` *readers* exist yet, so the new coarse-write checks find nothing; the extra `write_set` entries are inert). Commit (`feat(ssi): writer multi-granularity check + phantom coarse-write (Label/RelType)`).

---

## Task 3: Read-set promotion (the escalation core) + threshold `T`

**Files:** `transaction/manager.rs`.

- [ ] **Step 1 (test):** manager — with `T = 4`, call `record_read_in_label(tx, Node(i), None, LabelId(9))` for `i in 0..10`. Assert: after the 5th, the tx's `read_set` contains `EntityId::Label(LabelId(9))` and **no** `EntityId::Node(i)` for label 9 (promoted + fine dropped); `read_set` size is bounded (1 coarse key, not 10). A read under a *different* label stays fine. `T = usize::MAX` → never promotes (all 10 fine, no coarse). FAIL.
- [ ] **Step 2:** Add to `TransactionInfo`: `scan_buckets: FxHashMap<EntityId /*Label(L)/RelType(T)*/, Vec<(EntityId, PropTag)>>` (the fine entries recorded under each predicate) + `escalated: FxHashSet<EntityId>` (promoted predicates). Add `manager.escalation_threshold: usize` (default 256). New `record_read_in_label(tx, entity, tag, predicate: EntityId)` and `record_read_in_rel_type(...)`:
  - If `predicate ∈ info.escalated` → ensure the coarse key is in `read_set`/registry (it is); **do not** insert the fine entry. Return.
  - Else: do the normal `record_read(entity, tag)` (fine entry + rw-detection + registry) AND push `(entity, tag)` into `scan_buckets[predicate]`.
  - If `scan_buckets[predicate].len() > T`: **promote** — (a) `record_read(predicate, None)` (records the coarse read WITH its own rw-detection against concurrent `Label(L)`/`RelType(T)` writers); (b) for each `(e, t)` in `scan_buckets[predicate]`, remove it from `info.read_set` and `read_registry.remove_reader(e, tx, t)` (add `remove_reader` to the registry if absent); (c) `info.escalated.insert(predicate)`, clear the bucket.
  - (Lock discipline: mirror `record_read` — gather under the `transactions` write lock, do registry ops after release.)
- [ ] **Step 3:** PASS; full gate (still behavior-preserving end-to-end — nothing *calls* `record_read_in_label` yet; Task 4 wires it). Commit (`feat(ssi): read-set Node→Label / Edge→RelType promotion at threshold T`).

---

## Task 4: Wire the predicate hint from the store's scan accessors

**Files:** `execution/operators/mod.rs` (`ReadTracker`), `read_tracker.rs`, `graph/lpg/store/mod.rs`, `schema.rs` + the label/type-scan accessors.

- [ ] **Step 1 (test):** engine/store-level — a Serializable tx runs `MATCH (n:L) RETURN n` over `> T` `:L` nodes; afterward the tx's `read_set` (inspect via `manager.read_set_tagged(tx)`) contains **one** `EntityId::Label(L)` and zero `EntityId::Node` for those rows. A `MATCH (n:L {id: X})` indexed point read (1 row) stays fine (`EntityId::Node`). FAIL.
- [ ] **Step 2:** Add `ReadTracker::record_read_node_in_label(&self, tx, node_id, label_id)` (default no-op) + `record_read_edge_in_rel_type` (default no-op); `TransactionReadTracker` forwards to `manager.record_read_in_label(tx, Node(node_id), tag, EntityId::Label(label_id))`. Add store chokepoint `record_read_node_in_label(tx, id, label_id)` forwarding to the tracker. In the **label-scan accessors** (`schema.rs:380/428/733` and the `filter_visible_node_ids_versioned`/`nodes_by_label` paths — the loops that call `record_read_node` after computing a label's visible set), call `record_read_node_in_label(tx, id, <scanned label_id>)` instead. Edge/type-scan accessors → `record_read_edge_in_rel_type`. **Leave point/property/index reads on the plain `record_read_node`** (no predicate → fine, never escalated). Resolve the `label_id` from the label registry at the scan site (it already has the label name/id to do the scan).
- [ ] **Step 3:** PASS; full gate. Commit (`feat(ssi): label/type-scan accessors supply the escalation predicate`).

---

## Task 5: The `T` knob + default-on

**Files:** session/manager config (where `ConflictGranularity` is set).

- [ ] **Step 1 (test):** a session with `set_escalation_threshold(usize::MAX)` runs the `> T` `:L` scan → read_set has the N fine `EntityId::Node` entries, no coarse key (full precision restored). Default session (`T=256`) over `> 256` rows → escalates. FAIL (no setter).
- [ ] **Step 2:** Expose `Session::set_escalation_threshold(t: usize)` (or a field on the granularity config struct) that sets `manager.escalation_threshold`, latched at `begin_transaction_with_isolation` like `ConflictGranularity` (document the begin-time latch). Default `256`. `usize::MAX` = off. Serializable-only (SI/RC never build a read-set, so it's inert there — assert).
- [ ] **Step 3:** PASS; gate. Commit (`feat(ssi): configurable escalation threshold T (default 256, MAX = full precision)`).

---

## Task 6: Acceptance, soundness, and re-measure

**Files:** `tests/serializable.rs`; run `benches/serializable_overhead.rs`.

- [ ] **Step 1 (acceptance tests):**
  - `serializable_escalated_write_skew_still_aborts`: two Serializable txns each scan `> T` `:Account` rows (forcing escalation to `Label(Account)`) then each write a different row; the classic write-skew → second committer `SerializationFailure` (proves promotion preserved the conflict).
  - `serializable_escalated_scan_phantom_aborts`: T1 scans `> T` `:L` rows (escalates), T2 `CREATE (:L …)` (or `add_label` making a node `:L`) + reads a sentinel T1 wrote → rw-cycle → abort; a disjoint label → both commit.
  - `serializable_escalation_bounds_read_set`: a `> T` `:L` scan ⇒ `read_set_tagged(tx)` holds exactly one `EntityId::Label(L)` for those rows (asserted count). Edge-scan symmetric (`RelType`).
  - `serializable_escalation_threshold_max_is_full_precision`: `T=MAX` ⇒ the read_set is the N fine entries; behavior identical to pre-refinement.
- [ ] **Step 2:** Full verification — `--all-features` green; `--test serializable` + `--test serializable_tracking` green (**the whole SSI suite is the by-construction soundness anchor — every pre-existing write-skew / 3-tx-cycle / Part-G / text+vector phantom test must still pass with escalation default-on**); clippy; profiles + wasm; OPSEC. **Re-run `cargo bench --bench serializable_overhead`** and record the new `read_scan/serializable` numbers (expect the 5–8× to collapse toward ~1.x) in the commit message / a `bench` note.
- [ ] **Step 3:** Commit (`feat(ssi): multi-granularity escalation acceptance + bench (read_scan 5-8x -> ~1.x)`).

---

## Acceptance
- Serializable read-set promotes `Node→Label` / `Edge→RelType` at `T`; read-set memory bounded; the measured `read_scan` overhead collapses from 5–8× toward ~1.x; SI/RC byte-unchanged; `T=MAX` reproduces today.
- **Soundness preserved by construction:** the entire `serializable`/`serializable_tracking` suite passes with escalation default-on (escalated write-skew still aborts; escalated scan + concurrent `CREATE :L`/`add_label` aborts — the phantom).
- Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC.

## Risks (from spec §6/§8)
- **Dropping fine entries on promotion** must be matched by the writer checking the coarse key (Task 2 before Task 3) and the phantom coarse-write on create/add_label — else a conflict is missed. The escalated-write-skew + phantom acceptance tests are the load-bearing checks; the full SSI suite is the regression anchor.
- **Lock discipline** in promotion (registry ops after releasing the `transactions` lock) mirrors `record_read`; a deadlock/ordering bug surfaces in the tracking suite.
- **Predicate-hint coverage** is perf-only: a scan accessor missing the hint ⇒ fine recording ⇒ sound-but-unescalated (never a correctness bug).
- **Escalation subsumes Part-G precision** above `T` (coarse key is structural `PropTag=None`) — documented; precision retained below `T`.

---

## STATUS: NOT STARTED

Plan written against `integration` @ `8f4765cb`+ (spec committed). Scope: multi-granularity SIREAD escalation (`EntityId::Label`/`RelType`) — PostgreSQL-style read-set promotion, default-on threshold `T`. Reuses the `EntityId::Index`/Part-G additive-variant machinery. Execute subagent-driven, one task behind its gate; **Task 2 must precede Task 3** (writer coarse-check before promotion drops fine entries); the escalated-write-skew + phantom acceptance (Task 6) and the full SSI regression suite are the load-bearing soundness checks. Primary payoff: read-set memory ceiling + the measured scan-overhead collapse.
