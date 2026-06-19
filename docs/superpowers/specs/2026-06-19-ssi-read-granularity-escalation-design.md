# Multi-Granularity SIREAD Escalation — Design

**Status:** Design (approved in brainstorming 2026-06-19). Next: implementation plan via writing-plans.

## 1. Goal

Bound the Serializable (SSI) **read-recording cost and read-set memory** by promoting per-row read-set entries to per-label / per-type when a transaction reads many rows of a label/type — the PostgreSQL multi-granularity SIREAD-with-promotion technique (Ports & Grittner, VLDB 2012), applied to Grafeo's `EntityId` conflict-key framework. Default-on with one tunable threshold `T`. Sound by construction: coarsening a read-set entry only ever adds **false** conflicts (extra aborts), never misses a real one.

**Motivation (measured):** the `serializable_overhead` bench (committed `7d456411`) shows Serializable vs SnapshotIsolation overhead is read-recording-bound — ~0.8 µs per recorded read, **5–8× on full-label scans** (and the corpus's open "MVCC overhead on multi-hop perf, 50–120%" is the edge-read analog during traversal). The fixed per-tx cost is negligible (+41 ns). So the lever is the *per-read* recording: record coarse when a transaction reads a whole label/type.

## 2. Background: why coarsening is always sound

PostgreSQL records SIREAD locks at tuple/page/relation granularity and **promotes** finer to coarser (all tuples in a page → page lock; sequential scan → relation lock immediately) to bound memory. The safety argument: a coarse lock covers a **superset** of the rows actually read, so every write that would conflict with a fine entry still conflicts with the coarse one (plus some that wouldn't have) → only false positives, never false negatives. Grafeo's `ConflictGranularity` doc already states the identical frame for Part-G ("hash collisions can only produce false conflicts… never missed conflicts. The knob cannot introduce anomalies"). This refinement is the **upward** sibling of Part-G's downward (entity→property) refinement, on the same `EntityId` axis.

## 3. The decisions (from brainstorming)

- **Scope:** full multi-granularity with escalation (not just immediate seq-scan coarsening).
- **Activation:** default-on, one configurable threshold `T` (per label/type per tx); `T = usize::MAX` reproduces today's exact full-precision behavior.
- **Axes:** nodes (`EntityId::Label`) **and** edges (`EntityId::RelType`).
- **Locus (the protected choice):** the store chokepoints keep recording fine (F1 complete-by-construction, **unchanged**); the **read-set in the manager owns the promotion mechanism**; the **scan/expand operator provides the scan predicate** (label/type) as *optional context* to the chokepoint. A missing predicate ⇒ sound-but-unescalated. This arrangement is structurally incapable of breaking serializability (see §6).

## 4. Architecture

### 4a. New conflict keys
`EntityId::Label(LabelId)` and `EntityId::RelType(RelTypeId)` — additive variants threaded through `read_set`, `write_set`, `read_registry`, `retired_readers`, and the rw/W-W checks exactly like `EntityId::Index` (commit `2d18e903`). `LabelId`/`RelTypeId` are the interned ids from the label/relationship-type registry (or a stable hash, mirroring `IndexId::for_index`). A coarse read carries `PropTag = None` (structural/wildcard).

### 4b. Recording: fine at the chokepoint + an optional predicate hint
The store visible-read chokepoints (`record_read_node`/`record_read_edge`) still record fine `EntityId::Node`/`Edge` per read — **F1 unchanged, completeness preserved**. They gain an optional predicate parameter (the scan label / edge type), supplied by the operator that knows what it is iterating:
- `NodeScan(:L)` → tags its reads with `Label(L)`.
- `Expand`/traversal over `[:T]` → tags its edge reads with `RelType(T)`.
- Point/indexed/property reads with no scan predicate → no tag → fine, never escalated.

The predicate is the **scan's** label/type (the set the read belongs to), not the node's incidental labels — a `MATCH (n:A)` over an `:A:B` node read it *because it is `:A`*; escalating to `A` is sound and tight, to `B` would over-abort. Only the operator has this; the node does not.

### 4c. Promotion (the read-set mechanism, in the manager)
The read-set buckets fine entries by `(predicate, tx)` and counts them. When a bucket for `Label(L)` (or `RelType(T)`) exceeds `T`:
1. Insert `EntityId::Label(L)` into the read-set + registry (`PropTag = None`).
2. **Demote**: remove that label's fine `EntityId::Node` entries from the read-set + registry (subsumed by the coarse key); subsequent reads tagged `Label(L)` short-circuit to "already covered" without inserting a fine entry.
Result: read-set memory bounded at ≈ `T` per label/type + one coarse key. Promotion is one-way within a transaction.

### 4d. Writer multi-granularity check (the conflict side)
`record_write(EntityId::Node(n), …)` checks `readers_of_compatible` for **`EntityId::Node(n)` AND `EntityId::Label(L)` for each label `L` of `n`** — so a write conflicts with both fine readers (read this exact node) and escalated readers (read the whole label). One extra registry lookup per label of `n` (labels are few; the write path has headroom at ~1.2×). Symmetric for edges: an edge write checks `RelType(T)`.

### 4e. The phantom (the load-bearing predicate semantics)
An escalated reader recorded "I depend on the `:L` set." A concurrent **`CREATE` of a new `:L` node**, or an `add_label` that makes a node `:L`, is a phantom that changes that set — it must record an `EntityId::Label(L)` **write** so it conflicts with the escalated read. So the write-side label/type recording fires not only on writes to existing `:L` nodes but on **node creation / label addition** (and new-edge creation for `RelType`). This is what makes the coarse key a true predicate lock rather than an aggregate of existing rows.

## 5. The threshold knob
`T: usize`, default `256`, settable per session alongside `ConflictGranularity` (the read-granularity policy). `T = usize::MAX` ⇒ never escalate ⇒ byte-identical to today's behavior. Lower `T` ⇒ smaller/cheaper read-sets, more false aborts on big scans. Serializable-only (SI/RC build no read-set, so escalation is inert there).

## 6. Soundness (why this can't break serializability)

1. **Coarsening ⇒ only false positives.** `EntityId::Label(L)` covers a superset of the `:L` rows the txn read; any write conflicting with a dropped fine entry still conflicts with the coarse key (§4d), and the phantom is caught (§4e). No rw-edge that existed pre-escalation disappears.
2. **F1 completeness is independent of the hint.** Every read is recorded fine at the chokepoint regardless of the predicate hint; promotion is a pure read-set transformation *after* recording. A missing/incorrect hint changes *which granularity* is retained, never *whether* the read is in the read-set. So no read can be lost.
3. **Benign failure mode.** A scan operator that omits its predicate ⇒ that path stays fine-grained ⇒ sound, just unoptimized. Unlike the text-recording arc, a missed hint is a perf gap, not a phantom hole — so coverage can grow incrementally with zero anomaly risk.
4. **Regression anchor:** the full SSI acceptance suite (write-skew, 3-transaction cycle, Part-G disjoint-property, the text/vector phantom-aborts) must pass unchanged with escalation default-on. Those are the by-construction proof that promotion preserved every real conflict.

## 7. Acceptance criteria

1. **Soundness preserved:** the entire `serializable` + `serializable_tracking` suite passes with escalation default-on (`T=256`); a write-skew whose readers escalate (each side scans > T `:Account` rows) **still** aborts the second committer.
2. **Phantom under escalation — tested:** a Serializable txn scans `:L` (escalates), a concurrent txn `CREATE (:L …)` (or `add_label`) → serialization failure; a disjoint label → both commit.
3. **Bounded read-set — tested:** a transaction scanning `> T` `:L` rows ends with **one** `EntityId::Label(L)` read-set entry, not N `EntityId::Node` entries (assert read-set size). Edge-scan symmetric.
4. **`T = usize::MAX` ⇒ identical to today:** no escalation, the pre-refinement read-set, all behavior byte-unchanged.
5. **Measured win:** re-run `serializable_overhead`; `read_scan/serializable` collapses from 5–8× toward ~1.x (and read-set memory is bounded). Capture before/after numbers in the bench/plan.
6. Full `--all-features` + `--features full` green; clippy; profiles + wasm; OPSEC (un-pushed).

## 8. Residuals / future

- **False aborts on escalated big scans** — the trade; `T` is the dial; the bench is the instrument to tune it on the abort-rate-vs-cost curve.
- **Escalation subsumes Part-G property precision** — once a label is escalated to the structural coarse key, disjoint-property concurrency for those rows is lost (the coarse read conflicts with any `:L` property write). Acceptable: precision is retained below `T` (the common OLTP case); escalation only triggers on large scans where precision was already expensive.
- **Range-predicate locks** (record `[lo,hi]` on an indexed property; conflict on inserts into the range) — a finer tool for selective range scans than label-level. Deferred to a later increment; label/type is the 80/20.
- **Predicate-hint coverage** — heavy producers (NodeScan, Expand) first; other read-producing operators added as bench data justifies (each is a pure perf gain, never a correctness fix).
