# Increment 2a — Label isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make transactional `SET`/`REMOVE` label snapshot-isolated — an uncommitted label change is invisible to other sessions (via `has_label`, `labels(n)`, and `MATCH (:Label)` scans) but visible to the writing transaction — completing the symmetric other half of the increment-1 dirty-read fix (which covered properties).

**Architecture:** Mirror the increment-1 property mechanism exactly. Extend the per-transaction `TxDelta` to carry node label ops; buffer transactional label writes into the delta instead of write-through; route every label read through a snapshot-aware accessor that merges the delta for the writing transaction and returns committed-only for everyone else; commit promotes the delta (existing `apply_tx_overlay` extended), rollback/savepoint drop/snapshot it (existing paths extended). The committed `label_index` is never populated with uncommitted labels (no dirty read via label scans); the writing transaction's own label scan uses the same writer-only bypass increment 1 used for property indexes.

**Tech Stack:** Rust, `cargo test`. Storage in `grafeo-core` (`graph/lpg/store/`), execution operators in `grafeo-core/src/execution/operators/`, transaction wiring in `grafeo-engine/src/session/mod.rs`. `CARGO_INCREMENTAL=0`.

---

## Orientation

This is increment 2, Part A of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§3). Increment 1 (on `integration` @ `3c993082`, carried onto this branch `feat/mvcc-increment-2`) did exactly this for **properties**; labels are the same shape:

- **The delta:** `TxDelta` (`crates/grafeo-core/src/graph/lpg/store/mod.rs:144`) holds `node_props: FxHashMap<(NodeId, PropertyKey), PropOp>` + `edge_props`. `PropOp` (mod.rs:127) is `Set(Value) | Remove`. The per-tx map is `tx_property_overlay: RwLock<FxHashMap<TransactionId, TxDelta>>` (mod.rs:515). Increment 1's methods `set_node_property_buffered` / `read_node_property_visible` / `apply_tx_overlay` / `drop_tx_overlay` / `tx_overlay_snapshot` / `tx_overlay_restore` live in `store/property_ops.rs`.
- **Label storage:** `node_labels: FxHashMap<NodeId, FxHashSet<u32>>` (non-temporal) / `VersionLog<FxHashSet<u32>>` (temporal) (mod.rs:430/434); `label_index: Vec<FxHashMap<NodeId,()>>` (label_id → nodes, for scans). Label ids are `u32`. The registry mapping name↔id is `label_registry` / `label_to_id`.
- **Label writes:** `add_label_versioned(node_id, label, tid)` / `remove_label_versioned(...)` (`schema.rs:413+`, trait at `graph_store_impl.rs:636-645`), called from the `Some(tid)` branch of `AddLabelOperator` (`mutation.rs:1111`) and `RemoveLabelOperator` (`mutation.rs:1256`). Non-transactional uses `add_label`/`remove_label`.
- **Label reads (route these):** `node.has_label(l)` — `filter.rs:578` (label predicate `MATCH (n:Foo)`), `merge.rs:292`; `node.labels` — `filter.rs:1857` (`labels(n)` function), `filter.rs:876`/`1749` (other label-function uses), `project.rs:446` (`NodeResolve` whole-node materialization). All read the committed/resolved node's labels → the dirty read.
- **Label scans** (`MATCH (:Foo)`) read `label_index` (committed-only). Keep it committed-only; the writing tx's own scan gets a delta-merge bypass.

**Why this is safe to do incrementally:** like increment 1, routing reads while writes still write-through is a no-op (delta empty), then flipping writes to buffered activates isolation. Commit/rollback/savepoint already call `apply_tx_overlay`/`drop_tx_overlay`/`tx_overlay_snapshot`/`tx_overlay_restore`; extending those to also handle labels means the session wiring needs **no change**.

**Verification gate:** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` plus `--features full` for the isolation integration tests. Keep `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean. **Hygiene:** format only changed files (`rustfmt <file>`, not whole-crate `cargo fmt`); confirm `git status` shows only intended files before committing. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/tests/mvcc_isolation.rs` | Tracked acceptance tests | Modify (add label isolation tests) |
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` | `TxDelta`, `LabelOp` | Modify (add `node_labels` to `TxDelta`, `LabelOp` enum) |
| `crates/grafeo-core/src/graph/lpg/store/property_ops.rs` | inherent delta methods | Modify (label buffered writers + accessor; extend apply/drop/snapshot/restore) |
| `crates/grafeo-core/src/graph/lpg/store/schema.rs` | label read accessor | Modify (`read_node_labels_visible`) |
| `crates/grafeo-core/src/graph/traits.rs` | trait surface | Modify (defaulted `read_node_labels_visible` + `add/remove_label_buffered`) |
| `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` | `LpgStore` overrides | Modify |
| `crates/grafeo-core/src/execution/operators/mutation.rs` | AddLabel/RemoveLabel ops | Modify (route to buffered) |
| `crates/grafeo-core/src/execution/operators/filter.rs`, `project.rs`, `merge.rs` | label reads | Modify (route through accessor) |
| `crates/grafeo-engine/src/query/planner/lpg/filter.rs` | label-scan planning | Modify (writer-only bypass for `MATCH (:Foo)`) |
| `crates/grafeo-core/src/graph/compact/layered.rs`, `database/cdc_store.rs`, `wal_store.rs` | wrappers | Modify (delegate the new methods, mirroring increment-1 Task 6) |

---

## Task 0: TDD baseline — label isolation tests + confirm RED

**Files:** Modify `crates/grafeo-engine/tests/mvcc_isolation.rs`

- [ ] **Step 1: Add the failing tests**

Append to `mvcc_isolation.rs`:
```rust
#[test]
fn uncommitted_label_add_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();

    writer.begin_transaction().unwrap();
    writer.execute("MATCH (p:Person {name: 'Ann'}) SET p:Secret").unwrap();

    // Writer sees its own label (read-your-writes), via has-label and via scan.
    let own = writer.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    assert_eq!(own.row_count(), 1, "writer must see its own uncommitted label");

    // Other session must NOT see the :Secret label.
    let reader = db.session();
    let scan = reader.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    let lbls = reader.execute("MATCH (p:Person {name: 'Ann'}) RETURN labels(p) AS l").unwrap();
    let seen_scan = scan.row_count();
    let seen_labels = format!("{:?}", lbls.rows()[0][0]);

    writer.rollback().unwrap();
    assert_eq!(seen_scan, 0, "uncommitted label must not be visible via scan to other sessions");
    assert!(!seen_labels.contains("Secret"), "uncommitted label must not appear in labels(p) for other sessions: {seen_labels}");

    // After rollback the writer's tx label is gone everywhere.
    let after = reader.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    assert_eq!(after.row_count(), 0, "rolled-back label must not exist");
}

#[test]
fn committed_label_add_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    writer.begin_transaction().unwrap();
    writer.execute("MATCH (p:Person {name: 'Ann'}) SET p:Secret").unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    assert_eq!(r.row_count(), 1, "committed label must be visible to other sessions");
}

#[test]
fn uncommitted_label_remove_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person:Vip {name: 'Ann'})").unwrap();

    writer.begin_transaction().unwrap();
    writer.execute("MATCH (p:Person {name: 'Ann'}) REMOVE p:Vip").unwrap();

    // Other session must still see :Vip (remove not committed).
    let reader = db.session();
    let during = reader.execute("MATCH (p:Vip) RETURN p.name").unwrap();
    let seen_during = during.row_count();

    writer.rollback().unwrap();
    let after = reader.execute("MATCH (p:Vip) RETURN p.name").unwrap();

    assert_eq!(seen_during, 1, "uncommitted label-remove must not be visible to other sessions");
    assert_eq!(after.row_count(), 1, ":Vip must be restored after rollback");
}
```

- [ ] **Step 2: Run; confirm RED**

Run: `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation uncommitted_label committed_label`
Expected: `uncommitted_label_add_*` and `uncommitted_label_remove_*` FAIL on the assertions (the reader sees the uncommitted label / a scan finds it). `committed_label_add_*` PASSES. (If a test fails to *compile* — e.g. `SET p:Secret` syntax — check how the engine's tests express label set/remove and adjust; `set_node_label`/the Cypher `SET n:Label` form should parse.) Do NOT touch `src/` to make them pass yet.

- [ ] **Step 3: Commit the failing baseline**
```bash
git add crates/grafeo-engine/tests/mvcc_isolation.rs
git commit -m "test(mvcc): failing label isolation probes (uncommitted SET/REMOVE label)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 1: Extend the delta with label ops + inherent accessor/writers

**Files:** Modify `store/mod.rs` (`TxDelta`, `LabelOp`), `store/property_ops.rs` (buffered writers, apply/drop/snapshot extension), `store/schema.rs` (`read_node_labels_visible`); Test `store/tests.rs`

- [ ] **Step 1: Failing inherent-method test**

Append to `crates/grafeo-core/src/graph/lpg/store/tests.rs`:
```rust
#[test]
fn label_delta_isolates_buffered_label_ops() {
    use grafeo_common::types::TransactionId;
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    let tx = TransactionId::new(7);
    let person_id = store.label_id("Person").unwrap();

    // Buffer add :Secret and remove :Person for tx.
    store.add_label_buffered(n, "Secret", tx);
    store.remove_label_buffered(n, "Person", tx);

    let secret_id = store.label_id("Secret").unwrap();
    let writer_view = store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), Some(tx));
    assert!(writer_view.contains(&secret_id), "writer sees buffered add");
    assert!(!writer_view.contains(&person_id), "writer sees buffered remove");

    // Other readers (None) see committed labels unchanged.
    let committed_view = store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), None);
    assert!(committed_view.contains(&person_id), "others see committed :Person");
    assert!(!committed_view.contains(&secret_id), "others do NOT see uncommitted :Secret");

    // Apply promotes.
    store.apply_tx_overlay(tx);
    let after = store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), None);
    assert!(after.contains(&secret_id) && !after.contains(&person_id), "commit applied label ops");
}
```
(Use the store's actual label-name→id helper — confirm the method name; it may be `label_id(name) -> Option<u32>` or via `label_to_id`. If absent, the buffered writers below resolve the id internally and you can assert on `read_node_labels_visible` returning ids that match `store.label_id(...)`.)

- [ ] **Step 2: Run; confirm it fails to compile (methods missing)**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --features lpg label_delta_isolates_buffered_label_ops`
Expected: compile error — `add_label_buffered`/`read_node_labels_visible` not found.

- [ ] **Step 3: Add `LabelOp` + extend `TxDelta`**

In `store/mod.rs`, next to `PropOp` (line 127) add:
```rust
/// A buffered label change for a transaction's delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LabelOp {
    Add,
    Remove,
}
```
And in `TxDelta` (line 144) add a field:
```rust
    /// Uncommitted node label changes, keyed by (node, label_id).
    pub(super) node_labels: FxHashMap<(NodeId, u32), LabelOp>,
```
(Confirm `TxDelta` derives `Default` + `Clone` — it must, since `tx_property_overlay` uses `.or_default()` and Task 5b cloned it; the new field is `FxHashMap`, both derive cleanly.)

- [ ] **Step 4: Add the inherent buffered writers + accessor**

In `property_ops.rs`, near `set_node_property_buffered`, add (resolving the label name to an id via the store's intern path — use the same call `add_label`/the registry uses to get-or-create a label id; check `schema.rs` for `intern_label`/`label_to_id` and reuse it):
```rust
    /// Buffers an uncommitted label add into the transaction's delta.
    #[doc(hidden)]
    pub fn add_label_buffered(&self, id: NodeId, label: &str, transaction_id: TransactionId) {
        let label_id = self.intern_label_id(label); // get-or-create the u32 id (same path add_label uses)
        self.tx_property_overlay.write().entry(transaction_id).or_default()
            .node_labels.insert((id, label_id), super::LabelOp::Add);
    }

    /// Buffers an uncommitted label remove into the transaction's delta.
    #[doc(hidden)]
    pub fn remove_label_buffered(&self, id: NodeId, label: &str, transaction_id: TransactionId) {
        if let Some(label_id) = self.label_id(label) {
            self.tx_property_overlay.write().entry(transaction_id).or_default()
                .node_labels.insert((id, label_id), super::LabelOp::Remove);
        }
    }
```
(If the existing label-intern helper has a different name, use it; the design requirement is "resolve the same `u32` id `add_label` would use" so the merged set matches `label_index`/`node_labels`.)

Add the accessor in `schema.rs` (where label reads live):
```rust
    /// Snapshot-consistent node label set: committed labels merged with the
    /// writing transaction's buffered label ops.
    #[doc(hidden)]
    #[must_use]
    pub fn read_node_labels_visible(&self, id: NodeId, epoch: grafeo_common::types::EpochId, transaction_id: Option<TransactionId>) -> FxHashSet<u32> {
        // committed base
        #[cfg(not(feature = "temporal"))]
        let mut labels: FxHashSet<u32> = self.node_labels.read().get(&id).cloned().unwrap_or_default();
        #[cfg(feature = "temporal")]
        let mut labels: FxHashSet<u32> = self.node_labels.read().get(&id)
            .and_then(|log| log.visible_at(epoch).cloned()).unwrap_or_default();
        #[cfg(not(feature = "temporal"))] let _ = epoch;
        if let Some(tx) = transaction_id {
            if let Some(delta) = self.tx_property_overlay.read().get(&tx) {
                for ((nid, label_id), op) in &delta.node_labels {
                    if *nid == id {
                        match op { super::LabelOp::Add => { labels.insert(*label_id); }
                                   super::LabelOp::Remove => { labels.remove(label_id); } }
                    }
                }
            }
        }
        labels
    }
```
(Confirm the temporal `VersionLog` accessor name — `visible_at(epoch)` per `grafeo-common::mvcc`; adjust if `node_labels` uses a different temporal read. Confirm `FxHashSet` is imported in `schema.rs`.)

- [ ] **Step 5: Extend `apply_tx_overlay` / `drop_tx_overlay` to handle labels**

In `property_ops.rs` `apply_tx_overlay`, after applying `node_props`/`edge_props`, add:
```rust
            for ((id, label_id), op) in delta.node_labels {
                match op {
                    super::LabelOp::Add => { self.add_label_by_id(id, label_id); }
                    super::LabelOp::Remove => { self.remove_label_by_id(id, label_id); }
                }
            }
```
(If there is no `add_label_by_id`/`remove_label_by_id`, resolve the id→name via the registry and call `add_label`/`remove_label`, OR add thin by-id helpers in `schema.rs`. The requirement: commit writes the label through the normal path so `node_labels` + `label_index` both update.) `drop_tx_overlay` already removes the whole `TxDelta` entry, so labels are dropped for free — no change needed. `tx_overlay_snapshot`/`tx_overlay_restore` clone/replace the whole `TxDelta`, so labels are covered for savepoints for free — confirm by reading those methods.

- [ ] **Step 6: Run; confirm GREEN**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --features lpg label_delta_isolates_buffered_label_ops`
Expected: PASS.

- [ ] **Step 7: Commit**
```bash
git add crates/grafeo-core/src/graph/lpg/store/mod.rs crates/grafeo-core/src/graph/lpg/store/property_ops.rs crates/grafeo-core/src/graph/lpg/store/schema.rs crates/grafeo-core/src/graph/lpg/store/tests.rs
git commit -m "feat(mvcc): per-transaction label delta + snapshot-aware label accessor

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Trait surface for label accessor + buffered writers

**Files:** Modify `graph/traits.rs`, `store/graph_store_impl.rs`; Test `store/tests.rs`

- [ ] **Step 1: Failing trait-level test** — append to `tests.rs` a copy of `label_delta_isolates_buffered_label_ops` but through `&dyn GraphStoreMut` (named `trait_label_accessor_isolates`), calling the methods on the trait object. Run it; confirm compile error (methods not on trait).

- [ ] **Step 2: Add defaulted trait methods.** In `traits.rs`, on `GraphStore` add `read_node_labels_visible(&self, id, epoch, Option<tx>) -> FxHashSet<u32>` with a default that returns the committed label set (`self.node_labels_of(id)` or the existing committed-label getter — find it; e.g. `get_node(id).labels` mapped to ids, or a direct `node_label_ids(id)`). On `GraphStoreMut` add `add_label_buffered`/`remove_label_buffered` with defaults falling back to `add_label_versioned`/`remove_label_versioned`.

- [ ] **Step 3: Override on `LpgStore`** in `graph_store_impl.rs`, delegating to the inherent methods (mirror the increment-1 property overrides exactly).

- [ ] **Step 4: Run the trait test GREEN + regression gate** `CARGO_INCREMENTAL=0 cargo check --all-features -p grafeo-core -p grafeo-engine` (other trait impls still compile via defaults) and the trait test passes.

- [ ] **Step 5: Commit** (`feat(mvcc): expose label accessor + buffered label writers on store traits`).

---

## Task 3: Route label reads through the accessor (behavior-preserving)

**Files:** Modify `execution/operators/filter.rs`, `project.rs`, `merge.rs`

- [ ] **Step 1: Enumerate label read sites**
```bash
grep -rn "\.has_label(\|\.labels\b\|node_label" crates/grafeo-core/src/execution --include="*.rs" | grep -v test
```
- [ ] **Step 2: Route each** — replace `node.has_label(l)` / `node.labels` materialization with `store.read_node_labels_visible(node_id, snap_epoch, tx_id)` (then membership / id→name as needed), sourcing `snap_epoch = viewing_epoch.unwrap_or_else(|| store.current_epoch())` and `tx_id = transaction_id` from the operator fields. The canonical sites: `filter.rs:578` (label predicate), `filter.rs:1857` + `876`/`1749` (`labels()` function), `project.rs:446` (`NodeResolve`), `merge.rs:292`. For label-name comparisons, resolve ids via the store registry (`store.label_id(name)`), comparing id sets. Preserve exact truth/Null semantics. Where an operator lacks `transaction_id`, pass `None` + a `// TODO(unified-mvcc): thread snapshot` (non-probe paths).
- [ ] **Step 3: Gate** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core` GREEN (delta empty ⇒ identical behavior; a `temporal` shift toward at-snapshot label reads is the intended improvement — report DONE_WITH_CONCERNS if a `temporal` test moves, don't force it).
- [ ] **Step 4: Commit** (`refactor(mvcc): route execution label reads through the snapshot accessor`).

---

## Task 4: Buffer transactional label writes + label-scan writer bypass (flips the probe)

**Files:** Modify `execution/operators/mutation.rs` (AddLabel/RemoveLabel), `query/planner/lpg/filter.rs` (label-scan planning)

- [ ] **Step 1: Route the writes.** In `mutation.rs` `AddLabelOperator` (`Some(tid)` branch, ~1111) flip `add_label_versioned` → `add_label_buffered`; `RemoveLabelOperator` (~1256) flip `remove_label_versioned` → `remove_label_buffered`. Leave the non-`Some(tid)` (auto-commit) branches write-through.
- [ ] **Step 2: Label-scan writer bypass.** A `MATCH (:Foo)` label scan reads the committed `label_index`. For a **writing** transaction it must reflect its buffered label adds/removes; for others it must NOT (committed-only). Find the label-scan planner/operator (`query/planner/lpg/filter.rs` plans `LabelScan`; the operator reads `label_index`). When `transaction_id.is_some()`, the label scan must additionally include nodes the tx buffered `:Foo` onto and exclude nodes it buffered `:Foo` off of — apply the same shape as increment-1's property-index writer bypass: when `transaction_id.is_some()`, do a committed label scan then merge the tx's `node_labels` delta for that label id (add buffered-adds, drop buffered-removes). **Never insert uncommitted labels into `label_index`** (that would dirty-read other sessions' scans).
- [ ] **Step 3: Run the probes** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation uncommitted_label committed_label` → all GREEN (writer sees own label via `has_label` + scan; readers see committed only; commit/rollback correct).
- [ ] **Step 4: Full gate** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` + `--features full` integration → green; handle any within-tx label-index regression with the writer-bypass (mirror increment-1's index handling). Commit (`feat(mvcc): buffer transactional label writes into the per-tx delta + label-scan writer bypass`).

---

## Task 5: Wrapper delegation + completeness

**Files:** Modify `graph/compact/layered.rs`, `database/cdc_store.rs`, `database/wal_store.rs`

- [ ] **Step 1: Delegate** the new trait methods (`read_node_labels_visible`, `add_label_buffered`, `remove_label_buffered`) through `LayeredStore` (full delegation to inner, like increment-1 Task 6) and `CdcStore`/`WalStore` (delegate the read accessor; for the buffered writers keep the Option-B write-through default OR delegate consistently with how increment 1 handled `*_buffered` — match that decision exactly, with the same `TODO(unified-mvcc)` rationale).
- [ ] **Step 2: Gate** full `--all-features` + `--features full` green; clippy clean. Commit (`feat(mvcc): delegate label accessor + delta through store wrappers`).

---

## Task 6: Full verification

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green; the label probes + increment-1 probes + savepoint suite green under `--features full`.
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` (`wasm32-unknown-unknown`) compile.
- [ ] `git status` clean (only intended files across the task commits; revert any collateral `cargo fmt` churn).
- [ ] OPSEC: scan the diff for the private project's name/schema before any push.

---

## Acceptance
- Uncommitted `SET`/`REMOVE` label is invisible to other sessions via `has_label`, `labels(n)`, AND `MATCH (:Label)` scans, but visible to the writer; commit makes it visible; rollback restores; savepoints scope it (via the existing snapshot/restore covering the whole `TxDelta`).
- Full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; tree clean; OPSEC-clean.
- Isolation is now uniform across existence + properties + **labels** + node-deletes on `LpgStore` + `LayeredStore` (edge-deletes follow in increment 2b).

## Risks
- **Label id resolution mismatch:** the buffered op must use the *same* `u32` id `add_label`/`label_index` use, or the merged set won't match scans. Mitigation: reuse the exact intern path; the store-level test asserts id agreement with `store.label_id(name)`.
- **Label-scan writer bypass completeness:** mirror increment-1's property-index bypass precisely; the probe's scan assertion is the guard.
- **Temporal label reads:** `read_node_labels_visible` reads at-snapshot under `temporal` (an improvement over committed/latest) — a `temporal` test asserting old behavior may shift; report rather than force.
