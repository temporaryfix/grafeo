# Design — Unified MVCC isolation model

**Status:** Design / pre-RFC. Foundational change to the engine's core
visibility + storage primitive. Supersedes the tactical
`2026-06-15-audit-wave2b-property-isolation-design.md` (per-transaction
write-buffer), which solved the dirty-read symptom but added a *fourth*
isolation mechanism rather than converging the existing three.

**Why an RFC, not a wave:** this touches the shared storage/visibility primitive
that everything else builds on. It should be designed and agreed before code,
and landed in shippable increments. Wave 2a (write-set-scoped commit) already
fits underneath it unchanged.

---

## 1. The problem: three half-built deltas, and reads that ignore the snapshot

The engine has **three different, partial mechanisms for one concept — MVCC
visibility:**

| # | Subsystem | Mechanism | Snapshot-aware reads? |
|---|---|---|---|
| 1 | Node/edge **existence** | `VersionChain<Record>` + `visible_to(epoch, tx)` + PENDING + finalize | **Yes** — this part is correct |
| 2 | Properties/labels (non-temporal, default) | single unversioned column + write-through + undo log | **No** — `get()` returns latest |
| 3 | Properties/labels (temporal feature) | per-entity `VersionLog<Value>` with epoch tags + PENDING | **No** — read path still calls latest `get()` |

Two consequences:

- **The read path honors MVCC for *existence* but reads the *data* at "latest".**
  A node is visible-or-not at your snapshot, but its property and label values are
  whatever is newest. That mismatch *is* the dirty-read bug (confirmed for both
  `SET` and `DELETE`), in **both** build profiles — temporal stores versions but
  the execution read path never asks for them at a snapshot.
- **There are already three different "delta over base" structures** that are
  morally the same thing and don't know about each other:
  - the **temporal `VersionLog`** (uncommitted/recent property versions over the
    committed value),
  - the **`LayeredStore` overlay** (mutable overlay over the compacted columnar
    base),
  - the proposed **per-transaction write-buffer** (would have been a third).

  Plus the same hot-over-cold shape recurs in `ChunkedAdjacency` (hot chunks +
  delta buffer + cold compressed) and inside `PropertyColumn` (hot `HashMap` +
  compressed data).

The engine keeps reinventing "small mutable recent layer over large immutable
compressed base." The right shape is to make that **one** thing and put MVCC
visibility on it.

## 2. Target shape: one base, one delta, one read accessor

> **One snapshot-consistent read accessor, over one MVCC-versioned delta, over one
> columnar committed base.**

Four principles:

### 2.1 Snapshot-consistent reads through a single accessor (the linchpin)

Every property/label/existence read in execution and in node/edge
materialization goes through **one** accessor:

```
read_node(id, snapshot) -> Option<NodeView>      // existence + labels + props
read_node_property(id, key, snapshot) -> Option<Value>
read_edge(id, snapshot) -> Option<EdgeView>
// snapshot = (viewing_epoch, Option<transaction_id>) — what operators ALREADY carry
```

The accessor resolves visibility and merges tiers; callers never touch storage
directly. This single invariant is load-bearing: once every read goes through it,
**isolation is correct by construction**, and the storage underneath (§2.3) can
change without re-threading operators. It also kills the "scattered latest
`get()`" problem permanently.

**Consistency comes from the snapshot, not from physical co-location.** A read at
snapshot `S` resolves *every* property of a node at the same `S`, so the
reconstructed node is a consistent cut even if different properties were last
written at different epochs. (This is why per-property storage is *semantically*
fine — the defect was never the granularity, it was that reads ignored `S`.)

### 2.2 The MVCC delta is one mechanism

Uncommitted and recent changes live in **one** delta tier, MVCC-versioned with
`(created_epoch, deleted_epoch, created_by, deleted_by)` — the same
`VersionInfo` that existence already uses. The delta subsumes the temporal
`VersionLog`, the `LayeredStore` overlay, and the would-be write-buffer.

- **Writes** (transactional) append a new delta version (PENDING epoch); they do
  **not** touch the committed base.
- **Visibility:** `visible_to(epoch, tx)` on the delta entry — uncommitted
  versions are visible only to their writing transaction; everyone else sees the
  base (or an older committed delta).
- **Deletes** set `deleted_epoch = PENDING` (mirroring PENDING creates), finalized
  on commit; adjacency tombstones are part of the delta and applied at commit.

**Recommended delta granularity: per node/edge** (a delta entry carries that
version's labels + changed properties, copy-on-write carry-forward), because it
matches the existing `VersionChain`, gives the cleanest GC, and keeps a node's
version atomic. Per-property granularity is the space-efficient alternative and is
*also* snapshot-consistent; it is an internal storage choice, deferrable, and does
not change the accessor contract. Either way the delta is **short-lived** (§2.4),
so its memory cost is bounded by the GC horizon, not by total data size.

### 2.3 Tiers: hot MVCC delta over cold columnar base

- **Cold/committed base:** the existing columnar `PropertyStorage` (and the
  compact store) — compressed, zone-mapped, spillable, vectorized-scan-friendly.
  Represents committed data at/below a base epoch. **Untouched by writes**, so it
  keeps every analytical optimization (this is exactly the capability the
  write-buffer detour was trying to protect, made first-class).
- **Hot delta:** §2.2, small, row-oriented, MVCC.
- The read accessor (§2.1) merges: delta-version-visible-at-`S` if present, else
  the base.

This is the `LayeredStore` overlay/base relationship **generalized into the MVCC
mechanism** — same pattern the engine already runs for adjacency hot→cold.

### 2.4 Commit promotes; compaction folds

- **Commit:** stamp the transaction's delta versions PENDING→commit_epoch and
  apply deferred index/adjacency changes. O(entities written) — Wave 2a already
  does this for existence; it generalizes to the whole delta.
- **Rollback:** drop the transaction's delta versions. Trivial — the base was
  never touched (no undo replay).
- **Compaction (background / threshold):** delta versions whose visibility is
  below the GC horizon (visible to *all* active transactions) are folded into a
  new immutable columnar base, and the old base + folded delta are discarded.
  This is the existing `recompact()` / adjacency hot→cold migration, generalized.

## 3. How current components map onto the target

| Component | Fate |
|---|---|
| `VersionChain` + `VersionInfo` + `visible_to` (existence) | **Keep & generalize** — becomes the delta's visibility core for labels/props too |
| Columnar `PropertyStorage` / `PropertyColumn` (compression, zone maps, spill) | **Keep** as the cold committed base |
| `LayeredStore` overlay-over-compact | **Generalize** — it *is* the hot-delta/cold-base split; make it the one MVCC tiering |
| Temporal `VersionLog<Value>` | **Subsume** into the delta (right idea, wrong layer — it was a per-property delta nobody read at a snapshot) |
| Non-temporal write-through + property undo log | **Replace** — uncommitted writes become delta versions; rollback drops them, no undo replay |
| Write-set-scoped commit (Wave 2a) | **Reuse as-is** — already node-granular; generalizes from "finalize existence" to "promote delta" |
| `ChunkedAdjacency` hot/cold + delta | **Align** — same pattern; ideally the same tiering vocabulary |
| The `temporal` feature flag | **Retire** — versioned reads stop being optional; the base/delta split replaces it |

## 4. Migration plan (shippable increments)

The order is chosen so each step is independently correct and the riskiest,
highest-value invariant (the single read accessor) lands first.

1. **One read accessor over current storage.** Introduce
   `read_node`/`read_node_property`/edge variants taking `(epoch, tx)`. Back them
   initially with today's storage + a minimal uncommitted delta (even just the
   existing version chains for existence + a small per-tx property/label delta).
   Route **every** operator + materialization read through it. *This fixes both
   dirty-read bugs and establishes the load-bearing invariant.* Most of the
   delicate work, done once, in the right direction.
2. **Unify the delta.** Replace write-through + undo log with delta versions for
   transactional property/label writes; deletes use PENDING `deleted_epoch` +
   deferred adjacency. Commit promotes (reuse Wave 2a scoping); rollback drops.
   Retire the temporal `VersionLog` path in favor of the one delta.
3. **Tier + compaction.** Make the columnar `PropertyStorage`/compact store the
   formal cold base; make compaction fold below-horizon delta into it
   (generalize `recompact()`); retire the `temporal` flag.
4. **Converge adjacency** onto the same tiering vocabulary (optional, cleanup).

After step 1 the system is correct and isolated; steps 2–4 are
performance/architecture convergence with no further user-visible semantic
change.

## 5. First increment (concrete scope)

**Goal:** every property/label read is snapshot-consistent; both dirty-read
probes pass; no analytical capability lost.

In scope:
- The `read_*` accessor surface (existence already MVCC; add label + property
  snapshot reads) on `LpgStore` / the graph-store traits.
- A minimal per-transaction delta for uncommitted property/label writes + PENDING
  deletes (this *is* the write-buffer, but framed as "the delta of the unified
  model, read through the one accessor" — so it's a stepping stone, not a 4th
  mechanism).
- Threading `(epoch, tx)` into the enumerated read sites (operators +
  materialization).
- Commit applies the delta (reuse Wave 2a write-set scoping); rollback drops it.

Out of scope for the first increment: folding the delta into the columnar base
(compaction stays on the existing committed column for now), retiring the
`temporal` flag, per-property delta storage, adjacency convergence.

## 6. Open questions / risks / tradeoffs

- **Read-path cost.** The accessor adds a delta check per read. Mitigation: the
  common case (no active transaction, or entity not in the delta) is a cheap
  "delta empty / miss → base" fast path; the delta is small and short-lived. Needs
  benchmarking against the current direct `get()`.
- **Delta memory.** Bounded by the GC horizon, but a long-running transaction or a
  stalled GC horizon grows it. Need a horizon/spill policy (the overlay-spill
  question deferred from the write-buffer spec lives here).
- **Indexed reads inside a writing transaction.** Property indexes describe
  committed data; an uncommitted write isn't reflected. First increment: own-tx
  indexed reads fall back to a scan-with-delta-merge (correct, slower, only for the
  writing session, only while its tx is open). Long-term: delta-aware index probing.
- **Delta granularity (node vs property).** Node-granular carry-forward is simplest
  and recommended; per-property is space-efficient for wide nodes with frequent
  small writes. Deferrable; doesn't change the accessor.
- **Compaction trigger & cost.** Folding delta→base rewrites columnar blocks;
  reuse the existing `recompact()`/freeze cadence and horizon. Needs a trigger
  policy (delta size / age).
- **Cross-graph + RDF.** Named graphs each own an `LpgStore` (their own
  base+delta); the RDF triple store has its own structures and needs the same
  accessor discipline if it is to share the model (or stays separate, documented).

## 7. Out of scope

- Real SSI / `record_read` (separate; the read accessor makes read-set tracking
  natural to add later).
- Unified *value semantics* (the ~6 comparison sites — a different audit finding).
- The session-direct conflict-detection gap (noted in Wave 2a).
