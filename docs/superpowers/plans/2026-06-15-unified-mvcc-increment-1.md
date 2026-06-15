# Unified MVCC — Increment 1: snapshot-consistent property & delete reads

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every transactional property write and node delete snapshot-isolated so the two `#[ignore]`d dirty-read probes in `crates/grafeo-engine/tests/audit_scratch.rs` (Probe 3 = uncommitted `SET`, Probe 2 = uncommitted `DELETE`) pass, with no analytical capability lost and the full `--all-features` suite green.

**Architecture:** Route every property *read* in execution through the foundation's snapshot-aware accessor (`read_node_property_visible` / `read_edge_property_visible`), and every *transactional* property *write* into the foundation's per-transaction delta (`set_node_property_buffered` etc.) instead of writing through to the committed columnar store. The committed columnar store is untouched by uncommitted writes (compression/spill/zone-maps preserved); the writing transaction reads its own delta (read-your-writes); every other session reads only committed data (no dirty reads). Commit promotes the delta (`apply_tx_overlay`, reusing Wave 2a write-set scoping); rollback drops it (`drop_tx_overlay`). Deletes are stamped with `EpochId::PENDING` `deleted_epoch` (mirroring PENDING creates) with deferred label-index/adjacency, finalized at commit. This is migration step 1 of `docs/superpowers/specs/2026-06-15-unified-mvcc-isolation-design.md` — the single-read-accessor invariant.

**Tech Stack:** Rust, `cargo test`, `grafeo-core` (storage + execution operators) and `grafeo-engine` (session/transaction). MVCC primitives in `grafeo-common/src/mvcc.rs`. Disk-conserving builds: `CARGO_INCREMENTAL=0`.

---

## Orientation (read before starting)

**The dirty read, exactly.** `crates/grafeo-core/src/execution/operators/project.rs:217-225` already threads `(viewing_epoch, transaction_id)` and resolves *existence* correctly via `store.get_node_versioned(node_id, ep, tx)` — but then calls `node.get_property("age")`, which reads the property at **latest**, ignoring the snapshot. That is the root cause for both `SET` and `DELETE` (existence is MVCC-correct; *data* is read latest-only).

**The foundation (already committed, additive, `crates/grafeo-core/src/graph/lpg/store/property_ops.rs:989-1148`):**
- Delta writers: `set_node_property_buffered(id, key, value, tx)`, `remove_node_property_buffered(id, key, tx)`, `set_edge_property_buffered(id, key, value, tx)`. **Gap: there is no `remove_edge_property_buffered` — Task 1 adds it.**
- Accessor: `read_node_property_visible(id, &PropertyKey, epoch, Option<tx>)`, `read_edge_property_visible(...)`. Writing tx sees the delta (`Set`→value, `Remove`→None); everyone else (`tx == None`) reads the committed column.
- Commit/rollback: `apply_tx_overlay(tx)` (writes delta into the committed column then drops it), `drop_tx_overlay(tx)`.
- These are **inherent `LpgStore` methods** (`#[doc(hidden)]`), **not** on the `GraphStore` traits yet. Operators hold trait objects, so Task 1 exposes them on the traits.

**Trait layering** (`crates/grafeo-core/src/graph/traits.rs`):
- `GraphStore` (line 44) — base reads. `project.rs` holds `Arc<dyn GraphStoreSearch>` (`GraphStoreSearch: GraphStore`, line 360), so the read accessor goes on **`GraphStore`**.
- `GraphStoreMut: GraphStoreSearch` (line 497) — `mutation.rs`/`merge.rs` hold `Arc<dyn GraphStoreMut>`, so the buffered writers + `apply`/`drop` go on **`GraphStoreMut`**.
- The traits already use default-method bodies, so new methods get **default impls** (fall back to today's committed-read / write-through) and are **overridden only on `LpgStore`** (`crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs`). Wrappers (CDC/WAL/Layered) inherit the safe default until Task 6 upgrades them.

**The snapshot is already plumbed.** Read operators carry `viewing_epoch: Option<EpochId>` + `transaction_id: Option<TransactionId>` (scan/project/filter/expand). Mutation operators compute `let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM)` and only call the `*_versioned` write path inside `if let Some(tid) = self.transaction_id` — so non-transactional (auto-commit) writes already bypass the versioned path and must keep writing through.

**MVCC delete semantics** (`grafeo-common/src/mvcc.rs:42-99`): `is_visible_to(epoch, tx)` returns `false` if `deleted_by == Some(tx)` (writer sees its own delete), else `is_visible_at(epoch)` which keeps a version visible while `deleted_epoch > viewing_epoch`. Since `EpochId::PENDING` compares greater than any real epoch, `mark_deleted(EpochId::PENDING, tx)` makes a delete invisible to the writer but visible to everyone else — exactly the mirror of PENDING creates. The *current* bug: `delete_node_transactional` (`node_ops.rs:638`) stamps the **real** epoch AND eagerly strips the node from `label_index` (`node_ops.rs:675-688`), so a label scan loses it immediately.

**The probes** (`crates/grafeo-engine/tests/audit_scratch.rs`): Probe 3 (`uncommitted_property_write_invisible_to_others`, line 90) and Probe 2 (`uncommitted_delete_invisible_to_others`, line 48). Both use a second `db.session()` reader (auto-commit, `tx == None`). The reader filters on `name` (unmodified) and returns `age` / `name`. Acceptance = both pass after removing `#[ignore]`.

**Verification gate (run from repo root):**
```bash
CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine
```
This is the real gate — it caught MERGE/LOAD DATA bypassing write-tracking in Wave 2a when targeted tests did not. `execute_cypher`/`execute_sql` are feature-gated; integration tests need `--features full`. A bare `cargo test` failing with "no method execute_cypher" is pre-existing, not a regression.

**Scope guardrails.**
- IN: transactional node/edge property `SET`/`REMOVE` isolation; node `DELETE` isolation; commit/rollback wiring; routing the enumerated execution read sites.
- OUT (explicit, deferred to later increments — note in code, do not silently skip): label snapshot reads (`RETURN labels(n)` of an uncommitted label change — the probes do not exercise it), whole-node `RETURN n` materialization in engine-side result formatting, per-property delta storage, folding the delta into the columnar base, retiring the `temporal` flag, real SSI. Edge *delete* isolation is handled to the extent the transactional path defers adjacency (Task 5), but the probes only exercise an edgeless node.
- OPSEC: keep the private project's name/schema out of all code, tests, and commit messages. Commit messages end with the `Co-Authored-By: Claude Fable 5` line.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/tests/mvcc_isolation.rs` | Tracked acceptance tests mirroring the probes + read-your-writes | **Create** |
| `crates/grafeo-engine/tests/audit_scratch.rs` | Scratch probes | Modify (un-ignore Probe 2 + 3) |
| `crates/grafeo-common/src/mvcc.rs` | MVCC visibility + finalize | Modify (`finalize_deleted_epochs`, delete-visibility unit test) |
| `crates/grafeo-core/src/graph/traits.rs` | `GraphStore` / `GraphStoreMut` surface | Modify (defaulted accessor + buffered + apply/drop methods) |
| `crates/grafeo-core/src/graph/lpg/store/property_ops.rs` | Inherent delta methods | Modify (add `remove_edge_property_buffered`) |
| `crates/grafeo-core/src/graph/lpg/store/mod.rs` | Store fields | Modify (add `pending_tx_deletes`) |
| `crates/grafeo-core/src/graph/lpg/store/node_ops.rs` | Node delete | Modify (`delete_node_transactional` → PENDING + deferred index; `finalize_deletes_by_id`) |
| `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` | `LpgStore` trait impls | Modify (override new trait methods; `delete`-finalize trait method) |
| `crates/grafeo-core/src/execution/operators/project.rs` | `RETURN p.age` read | Modify (route through accessor) |
| `crates/grafeo-core/src/execution/operators/filter.rs` | property predicates | Modify (route) |
| `crates/grafeo-core/src/execution/operators/factorized_filter.rs`, `horizontal_aggregate.rs`, `vector_join.rs`, `range_scan.rs` | other property reads | Modify (route) |
| `crates/grafeo-core/src/execution/operators/mutation.rs`, `merge.rs` | transactional writes | Modify (route to buffered) |
| `crates/grafeo-engine/src/session/mod.rs` | commit/rollback | Modify (`apply`/`drop` overlay; delete finalize) |
| `crates/grafeo-engine/src/database/cdc_store.rs`, `wal_store.rs`, `async_wal_store.rs`, `crates/grafeo-core/src/graph/compact/layered.rs` | store wrappers | Modify (delegate new methods — Task 6) |

---

## Task 0: TDD baseline — tracked acceptance test + confirm RED

**Files:**
- Create: `crates/grafeo-engine/tests/mvcc_isolation.rs`
- Modify: `crates/grafeo-engine/tests/audit_scratch.rs`

- [ ] **Step 1: Write the tracked acceptance test (failing)**

Create `crates/grafeo-engine/tests/mvcc_isolation.rs`:
```rust
//! Snapshot isolation for transactional property writes and deletes
//! (unified-MVCC increment 1).
#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn uncommitted_set_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann', age: 30})").unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .unwrap();

    // Read-your-writes: the writer sees 99.
    let own = writer
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    assert_eq!(own.rows()[0][0].clone(), Value::Int64(99), "writer must see its own write");

    // Another session must still see the committed value (30).
    let reader = db.session();
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    let seen = r.rows()[0][0].clone();

    writer.rollback().unwrap();
    assert_eq!(seen, Value::Int64(30), "uncommitted SET must not be visible to other sessions");

    // After rollback the committed value is unchanged.
    let after = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    assert_eq!(after.rows()[0][0].clone(), Value::Int64(30), "rollback restores committed value");
}

#[test]
fn committed_set_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann', age: 30})").unwrap();
    writer.begin_transaction().unwrap();
    writer.execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99").unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader.execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age").unwrap();
    assert_eq!(r.rows()[0][0].clone(), Value::Int64(99), "committed SET must be visible");
}

#[test]
fn uncommitted_delete_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();

    writer.begin_transaction().unwrap();
    writer.execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p").unwrap();

    // Writer no longer sees Ann (read-your-writes for delete).
    let own = writer.execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name").unwrap();
    assert_eq!(own.row_count(), 0, "writer must not see its own deleted node");

    // Another session must still see Ann.
    let reader = db.session();
    let during = reader.execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name").unwrap();
    let visible_during = during.row_count();

    writer.rollback().unwrap();
    let after = reader.execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name").unwrap();

    assert_eq!(visible_during, 1, "uncommitted delete must not be visible to other sessions");
    assert_eq!(after.row_count(), 1, "node must exist after rollback of delete");
}

#[test]
fn committed_delete_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    writer.begin_transaction().unwrap();
    writer.execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p").unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader.execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name").unwrap();
    assert_eq!(r.row_count(), 0, "committed delete must be visible");
}
```

- [ ] **Step 2: Un-ignore the scratch probes**

In `crates/grafeo-engine/tests/audit_scratch.rs`, delete the `#[ignore = "..."]` line above `fn uncommitted_delete_invisible_to_others` (line 48) and above `fn uncommitted_property_write_invisible_to_others` (line 90). Leave Probe 7b (`var_length_expand_branching_cycle`) ignored — that is Wave 4.

- [ ] **Step 3: Run; confirm RED**

Run:
```bash
CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation --test audit_scratch
```
Expected: `uncommitted_set_is_invisible_to_other_sessions` FAILS (reader sees 99), `uncommitted_delete_is_invisible_to_other_sessions` FAILS (reader sees 0 during), the two un-ignored probes FAIL. `committed_*` tests PASS. This is the baseline.

- [ ] **Step 4: Commit the failing baseline**
```bash
git add crates/grafeo-engine/tests/mvcc_isolation.rs crates/grafeo-engine/tests/audit_scratch.rs
git commit -m "test(mvcc): failing isolation probes for uncommitted SET/DELETE

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 1: Expose the delta accessor + buffered writers on the store traits

**Files:**
- Modify: `crates/grafeo-core/src/graph/lpg/store/property_ops.rs` (add `remove_edge_property_buffered`)
- Modify: `crates/grafeo-core/src/graph/traits.rs` (defaulted trait methods)
- Modify: `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` (override on `LpgStore`)
- Test: `crates/grafeo-core/src/graph/lpg/store/tests.rs`

- [ ] **Step 1: Write the failing trait-level test**

Append to `crates/grafeo-core/src/graph/lpg/store/tests.rs`:
```rust
#[test]
fn trait_accessor_isolates_buffered_writes() {
    use crate::graph::traits::{GraphStore, GraphStoreMut};
    use grafeo_common::types::{PropertyKey, TransactionId, EpochId};

    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    store.set_node_property(n, "age", Value::Int64(30));

    let tx = TransactionId::new(7);
    let key = PropertyKey::new("age");

    // Buffer an uncommitted write via the trait object.
    let s: &dyn GraphStoreMut = &store;
    s.set_node_property_buffered(n, "age", Value::Int64(99), tx);

    // Writer (Some(tx)) sees its own write; everyone else (None) sees committed.
    assert_eq!(s.read_node_property_visible(n, &key, EpochId::new(0), Some(tx)), Some(Value::Int64(99)));
    assert_eq!(s.read_node_property_visible(n, &key, EpochId::new(0), None), Some(Value::Int64(30)));

    // Apply promotes; drop discards.
    s.apply_tx_overlay(tx);
    assert_eq!(s.read_node_property_visible(n, &key, EpochId::new(0), None), Some(Value::Int64(99)));
}
```

- [ ] **Step 2: Run; confirm it fails to compile (methods not on trait)**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --features lpg trait_accessor_isolates_buffered_writes`
Expected: compile error — `no method named set_node_property_buffered`/`read_node_property_visible` for `dyn GraphStoreMut`.

- [ ] **Step 3: Add the missing inherent `remove_edge_property_buffered`**

In `property_ops.rs`, directly after `set_edge_property_buffered` (line 1046), add:
```rust
    /// Buffers an uncommitted edge property removal (tombstone) into the delta.
    #[doc(hidden)]
    pub fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        self.tx_property_overlay
            .write()
            .entry(transaction_id)
            .or_default()
            .edge_props
            .insert((id, PropertyKey::new(key)), super::PropOp::Remove);
    }
```

- [ ] **Step 4: Add defaulted methods to the `GraphStore` read trait**

In `traits.rs`, inside `pub trait GraphStore` (after `get_edge_property`, near line 82), add:
```rust
    /// Snapshot-consistent node property read (unified-MVCC accessor).
    ///
    /// Default: ignores isolation and returns the committed value — safe for
    /// stores without a per-transaction delta. `LpgStore` overrides this to
    /// merge its delta for the writing transaction.
    fn read_node_property_visible(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        let _ = (epoch, transaction_id);
        self.get_node_property(id, key)
    }

    /// Snapshot-consistent edge property read. See [`read_node_property_visible`](Self::read_node_property_visible).
    fn read_edge_property_visible(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        let _ = (epoch, transaction_id);
        self.get_edge_property(id, key)
    }
```
Ensure `EpochId` is imported in `traits.rs` (it uses `TransactionId` already in the `*_versioned` signatures; add `EpochId` to the existing `grafeo_common::types` import if missing).

- [ ] **Step 5: Add defaulted methods to the `GraphStoreMut` write trait**

In `traits.rs`, inside `pub trait GraphStoreMut` (near the other `*_versioned` methods), add:
```rust
    /// Buffers an uncommitted node property write into the transaction's delta.
    /// Default: falls back to write-through (`set_node_property_versioned`).
    fn set_node_property_buffered(&self, id: NodeId, key: &str, value: Value, transaction_id: TransactionId) {
        self.set_node_property_versioned(id, key, value, transaction_id);
    }
    /// Buffers an uncommitted node property removal. Default: write-through.
    fn remove_node_property_buffered(&self, id: NodeId, key: &str, transaction_id: TransactionId) {
        self.remove_node_property_versioned(id, key, transaction_id);
    }
    /// Buffers an uncommitted edge property write. Default: write-through.
    fn set_edge_property_buffered(&self, id: EdgeId, key: &str, value: Value, transaction_id: TransactionId) {
        self.set_edge_property_versioned(id, key, value, transaction_id);
    }
    /// Buffers an uncommitted edge property removal. Default: write-through.
    fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        self.remove_edge_property_versioned(id, key, transaction_id);
    }
    /// Promotes a transaction's buffered property delta to the committed store
    /// (commit). Default: no-op (write-through stores have nothing buffered).
    fn apply_tx_overlay(&self, transaction_id: TransactionId) {
        let _ = transaction_id;
    }
    /// Drops a transaction's buffered property delta (rollback). Default: no-op.
    fn drop_tx_overlay(&self, transaction_id: TransactionId) {
        let _ = transaction_id;
    }
```

- [ ] **Step 6: Override on `LpgStore`**

In `graph_store_impl.rs`, inside the `impl GraphStore for LpgStore` block add (delegating to the inherent methods):
```rust
    fn read_node_property_visible(&self, id: NodeId, key: &PropertyKey, epoch: EpochId, transaction_id: Option<TransactionId>) -> Option<Value> {
        LpgStore::read_node_property_visible(self, id, key, epoch, transaction_id)
    }
    fn read_edge_property_visible(&self, id: EdgeId, key: &PropertyKey, epoch: EpochId, transaction_id: Option<TransactionId>) -> Option<Value> {
        LpgStore::read_edge_property_visible(self, id, key, epoch, transaction_id)
    }
```
And inside `impl GraphStoreMut for LpgStore`:
```rust
    fn set_node_property_buffered(&self, id: NodeId, key: &str, value: Value, transaction_id: TransactionId) {
        LpgStore::set_node_property_buffered(self, id, key, value, transaction_id);
    }
    fn remove_node_property_buffered(&self, id: NodeId, key: &str, transaction_id: TransactionId) {
        LpgStore::remove_node_property_buffered(self, id, key, transaction_id);
    }
    fn set_edge_property_buffered(&self, id: EdgeId, key: &str, value: Value, transaction_id: TransactionId) {
        LpgStore::set_edge_property_buffered(self, id, key, value, transaction_id);
    }
    fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        LpgStore::remove_edge_property_buffered(self, id, key, transaction_id);
    }
    fn apply_tx_overlay(&self, transaction_id: TransactionId) {
        LpgStore::apply_tx_overlay(self, transaction_id);
    }
    fn drop_tx_overlay(&self, transaction_id: TransactionId) {
        LpgStore::drop_tx_overlay(self, transaction_id);
    }
```
Confirm `EpochId` is imported in `graph_store_impl.rs`.

- [ ] **Step 7: Run; confirm GREEN**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-core --features lpg trait_accessor_isolates_buffered_writes`
Expected: PASS.

- [ ] **Step 8: Commit**
```bash
git add crates/grafeo-core/src/graph/traits.rs crates/grafeo-core/src/graph/lpg/store/property_ops.rs crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs crates/grafeo-core/src/graph/lpg/store/tests.rs
git commit -m "feat(mvcc): expose snapshot-aware property accessor + delta writers on store traits

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Route execution property reads through the accessor (delta empty ⇒ behavior-preserving)

Routing reads while writes still write-through is a no-op (the delta is always empty), so this establishes the load-bearing invariant safely. Do **not** change write paths in this task.

**Files:**
- Modify: `crates/grafeo-core/src/execution/operators/project.rs`
- Modify: `crates/grafeo-core/src/execution/operators/filter.rs`
- Modify: `crates/grafeo-core/src/execution/operators/factorized_filter.rs`
- Modify: `crates/grafeo-core/src/execution/operators/horizontal_aggregate.rs`
- Modify: `crates/grafeo-core/src/execution/operators/vector_join.rs`
- Modify: `crates/grafeo-core/src/execution/operators/range_scan.rs`

- [ ] **Step 1: Enumerate every property read site**

Run:
```bash
grep -rn "\.get_property(\|get_node_property\|get_edge_property\|get_nodes_properties\|get_edges_properties" \
  crates/grafeo-core/src/execution --include="*.rs" | grep -v test
```
Every hit that materializes a property *for query output or predicates* (not a write-path "read old value for index/undo") must be routed. The write-path reads inside `mutation.rs`/`merge.rs` are handled in Task 4; leave them for now.

- [ ] **Step 2: Route the canonical site — `project.rs` `PropertyAccess`**

In `project.rs` (the block at lines 207-240), the node branch currently does:
```rust
let node = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
    store.get_node_versioned(node_id, ep, tx)
} else if let Some(ep) = epoch {
    store.get_node_at_epoch(node_id, ep)
} else {
    store.get_node(node_id)
};
if let Some(prop) = node.and_then(|n| n.get_property(property).cloned()) {
    prop
} else if let Some(edge_id) = input_col.get_edge_id(row) {
    ...
```
Replace the node-property fetch with the accessor (existence is still resolved by the column type; the accessor returns the snapshot-correct *value*):
```rust
let value = if let Some(node_id) = input_col.get_node_id(row) {
    let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
    if let Some(prop) = store.read_node_property_visible(node_id, &prop_key, snap_epoch, tx_id) {
        prop
    } else if let Some(edge_id) = input_col.get_edge_id(row) {
        let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
        store
            .read_edge_property_visible(edge_id, &prop_key, snap_epoch, tx_id)
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    }
} else if let Some(edge_id) = input_col.get_edge_id(row) {
    let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
    store.read_edge_property_visible(edge_id, &prop_key, snap_epoch, tx_id).unwrap_or(Value::Null)
} else {
    Value::Null
};
```
Keep the surrounding `for row in input.selected_indices()` loop and `output_col.push_value(value)`. Preserve any existing null/unset semantics (match what `get_property` returned: missing property → `Value::Null`).

- [ ] **Step 3: Route the remaining sites the same way**

For each remaining hit from Step 1 in `filter.rs`, `factorized_filter.rs`, `horizontal_aggregate.rs`, `vector_join.rs`, `range_scan.rs`: replace `store.get_node_property(id, key)` with `store.read_node_property_visible(id, key, epoch, tx)` and `store.get_edge_property(id, key)` with `store.read_edge_property_visible(id, key, epoch, tx)`, sourcing `epoch`/`tx` from the operator's existing `viewing_epoch`/`transaction_id` fields (use `epoch.unwrap_or_else(|| store.current_epoch())`, pass `transaction_id` straight through as `Option`). If an operator lacks those fields, pass `store.current_epoch()` and `None` (committed read) and add a `// TODO(unified-mvcc): thread snapshot` note — these are non-probe paths; do not expand scope to add plumbing here.

- [ ] **Step 4: Run the full suite; confirm still GREEN (no behavior change)**

Run: `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core`
Expected: PASS (delta is empty, so every `read_*_visible` returns exactly what `get_*_property` returned). If anything fails, a routed call passed the wrong epoch/tx or changed null semantics — fix before proceeding.

- [ ] **Step 5: Commit**
```bash
git add crates/grafeo-core/src/execution/operators/
git commit -m "refactor(mvcc): route execution property reads through the snapshot accessor

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Wire commit/rollback to apply/drop the delta

Wire commit/rollback **before** flipping writes to buffered, so that when Task 4 starts buffering, commits already persist and rollbacks already discard.

**Files:**
- Modify: `crates/grafeo-engine/src/session/mod.rs`

- [ ] **Step 1: Apply the delta on successful commit**

In `commit_inner`, the success-path loop at lines 4069-4078 finalizes PENDING creates. Immediately after `store.finalize_entities_by_id(...)` inside that `for graph_name in &touched` loop, add:
```rust
            store.apply_tx_overlay(transaction_id);
```
(So each touched store both finalizes existence and promotes its buffered property delta at the commit epoch.)

- [ ] **Step 2: Drop the delta on conflict rollback**

In `commit_inner`, the conflict branch (lines 4028-4034) loops over touched graphs calling `discard_entities_by_id` + `rollback_transaction_properties`. Add, inside that loop:
```rust
                    store.drop_tx_overlay(transaction_id);
```

- [ ] **Step 3: Drop the delta on explicit rollback**

In `rollback_inner`, alongside the existing `store.rollback_transaction_properties(transaction_id)` (line 4239), add for each resolved store:
```rust
            store.drop_tx_overlay(transaction_id);
```
Match the existing per-store loop structure in `rollback_inner`.

- [ ] **Step 4: Savepoint rollback note**

`rollback_to_savepoint` (line 4342) uses `rollback_transaction_properties_to` for partial undo. The buffered delta is whole-transaction granular in this increment, so a partial savepoint rollback cannot drop only post-savepoint buffered writes. Add a code comment at the savepoint path: `// TODO(unified-mvcc): buffered property delta is tx-granular; savepoint partial-rollback of buffered writes is deferred (delta keys would need savepoint stamping).` Do not implement savepoint granularity now — no probe exercises it; verify in Task 7 that no existing savepoint test regresses (savepoint tests that SET then roll back to a savepoint within a tx may need attention; if one fails, the minimal fix is to also stamp delta entries with a savepoint position, but only if a test forces it).

- [ ] **Step 5: Build; confirm compiles, suite still green**

Run: `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-engine`
Expected: PASS (writes still write-through, so `apply_tx_overlay`/`drop_tx_overlay` operate on an empty delta — no behavior change yet).

- [ ] **Step 6: Commit**
```bash
git add crates/grafeo-engine/src/session/mod.rs
git commit -m "feat(mvcc): apply property delta on commit, drop on rollback/conflict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4a: Make whole-node/edge materialization (`RETURN n`) delta-aware

**Why this exists (discovered during execution, not in the original plan):** `Session::with_auto_commit` wraps **every** mutation statement in an implicit transaction (`begin_transaction_inner` → run operators → `commit_inner`), so operators run with `Some(transaction_id)` even in auto-commit. Task 2 routed single-property reads (`RETURN n.k` via `PropertyAccess`) but deliberately deferred whole-entity materialization. `project.rs`'s `ProjectExpr::NodeResolve` (lines ~307-336) and `EdgeResolve` (~338-367) materialize `RETURN n` / `RETURN e` by resolving existence with `get_node_versioned(node_id, ep, tx)` and then building a map from the node's **committed** properties (`node_to_map(&n)`). Once Task 4b buffers writes, that map would NOT reflect the transaction's own buffered writes — breaking read-your-writes for the ubiquitous `CREATE (n {..}) RETURN n` and `MATCH ... SET ... RETURN n`. So whole-entity materialization must become delta-aware first. Behavior-preserving here (the delta is still empty until 4b), exactly like Task 2.

**Files:**
- Modify: `crates/grafeo-core/src/graph/lpg/store/property_ops.rs` (whole-entity accessor)
- Modify: `crates/grafeo-core/src/graph/traits.rs` (defaulted trait methods)
- Modify: `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` (`LpgStore` override)
- Modify: `crates/grafeo-core/src/execution/operators/project.rs` (`NodeResolve`/`EdgeResolve`)
- Test: `crates/grafeo-core/src/graph/lpg/store/tests.rs`

- [ ] **Step 1: Add the whole-entity accessor (inherent on `LpgStore`)**

Add `read_node_properties_visible(&self, id: NodeId, epoch: EpochId, transaction_id: Option<TransactionId>) -> FxHashMap<PropertyKey, Value>`: start from the committed property map — non-temporal `self.node_properties.get_all(id)`, temporal the at-epoch equivalent — then if `transaction_id` is `Some(tx)` and `self.tx_property_overlay` has an entry for `tx`, apply that delta's `node_props` ops for this `id` (`PropOp::Set(v)` → `insert(key, v)`, `PropOp::Remove` → `remove(key)`). Add the edge twin `read_edge_properties_visible`. (This is the whole-entity form of the per-property accessor; it is what the design's `read_node(id, snapshot) -> NodeView` calls for. Keep `#[doc(hidden)]`.)

- [ ] **Step 2: Add defaulted trait methods + `LpgStore` override**

On `GraphStore`: `read_node_properties_visible` / `read_edge_properties_visible` with defaults that ignore isolation and return the committed whole-property map (`self.get_nodes_properties_batch(&[id]).pop().unwrap_or_default()` or the existing whole-property read — match what `node_to_map` consumes). Override both on `LpgStore` in `graph_store_impl.rs` delegating to the inherent methods.

- [ ] **Step 3: Route `NodeResolve` / `EdgeResolve`**

In `project.rs`, keep the `get_node_versioned`/`get_node_at_epoch`/`get_node` call for **existence + labels**, but build the materialized map's **properties** from `store.read_node_properties_visible(node_id, snap_epoch, tx_id)` (with `snap_epoch = epoch.unwrap_or_else(|| store.current_epoch())`) rather than from the node's own committed property map. Inspect `node_to_map`/`edge_to_map`; either add a variant that takes an explicit property map, or overlay the merged properties onto the resolved node before mapping. Preserve labels/id/type exactly (labels stay from the resolved node — uncommitted label changes are out of scope for this increment). Do the same for `EdgeResolve`.

- [ ] **Step 4: Sweep for any other whole-entity materialization**

```bash
grep -rn "node_to_map\|edge_to_map\|get_all(\|get_nodes_properties_batch\|get_edges_properties" crates/grafeo-core/src/execution --include="*.rs" | grep -v test
```
Route any other site that builds a returned node/edge's full property set through the new accessor. List any you leave and why.

- [ ] **Step 5: Test + gate**

Add a store-level test in `tests.rs`: buffer a `Set` and a `Remove` for a node via `set_node_property_buffered`/`remove_node_property_buffered`, then assert `read_node_properties_visible(id, epoch, Some(tx))` reflects both (changed key present with new value, removed key absent) while `read_node_properties_visible(id, epoch, None)` returns the committed map unchanged.
Gate (behavior unchanged, delta empty for query paths): `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` — green. The isolation probes stay RED (writes not buffered yet). Commit (message ends with the `Co-Authored-By: Claude Fable 5` line).

---

## Task 4b: Route transactional property writes into the delta (flips Probe 3)

**Depends on Task 4a:** with both single-property (`PropertyAccess`, Task 2) and whole-entity (`NodeResolve`/`EdgeResolve`, Task 4a) reads now delta-aware, flipping writes to buffered preserves read-your-writes within the writing transaction while isolating other sessions.

**Files:**
- Modify: `crates/grafeo-core/src/execution/operators/mutation.rs`
- Modify: `crates/grafeo-core/src/execution/operators/merge.rs`
- Modify: `crates/grafeo-engine/src/session/mod.rs`

- [ ] **Step 1: Enumerate transactional write sites**

Run:
```bash
grep -rn "set_node_property_versioned\|set_edge_property_versioned\|remove_node_property_versioned\|remove_edge_property_versioned" \
  crates/grafeo-core/src/execution/operators/mutation.rs \
  crates/grafeo-core/src/execution/operators/merge.rs \
  crates/grafeo-engine/src/session/mod.rs
```
Each of these is reached only inside an `if let Some(tid) = self.transaction_id` (or equivalent `Some(tid)`) guard — i.e., a real transaction. Auto-commit writes use the plain `set_node_property` path and must stay write-through.

- [ ] **Step 2: Flip each `*_versioned` call to its `*_buffered` counterpart**

For every site from Step 1, rename the method (same arguments): `set_node_property_versioned` → `set_node_property_buffered`, `set_edge_property_versioned` → `set_edge_property_buffered`, `remove_node_property_versioned` → `remove_node_property_buffered`, `remove_edge_property_versioned` → `remove_edge_property_buffered`. Example, `mutation.rs:323-326`:
```rust
// before
if let Some(tid) = self.transaction_id {
    self.store.set_node_property_versioned(node_id, name, value.clone(), tid);
}
// after
if let Some(tid) = self.transaction_id {
    self.store.set_node_property_buffered(node_id, name, value.clone(), tid);
}
```
Do **not** touch the non-`Some(tid)` / auto-commit branches.

- [ ] **Step 3: Run Probe 3 + the tracked SET tests; confirm GREEN**

Run:
```bash
CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine \
  --test mvcc_isolation uncommitted_set_is_invisible_to_other_sessions committed_set_is_visible_to_other_sessions \
  --test audit_scratch uncommitted_property_write_invisible_to_others
```
Expected: all PASS. The writer's own `RETURN p.age` now reads 99 from the delta (Task 2 routing), the reader reads 30 from the committed column, commit persists 99 (Task 3), rollback drops the delta.

- [ ] **Step 4: Full `grafeo-engine` + `grafeo-core` suite; address index-staleness regressions**

Run: `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine`
Buffered writes do **not** update the property/text indexes (those update at commit via `apply_tx_overlay` → `set_node_property`). So a *within-transaction* query that SETs an indexed property then matches on it via an index scan can miss its own write. If such a test regresses:
  - Preferred minimal fix: make the writing transaction's indexed property scans fall back to a non-indexed scan merged with the delta **only when `transaction_id.is_some()`** (committed/other-session scans keep using the index). Add a focused test reproducing the case first (TDD), then implement the fallback at the scan operator that consults the property index.
  - Do **not** update the index on the buffered write — that would re-expose uncommitted values to other sessions through index scans (a dirty read through the index).
If nothing regresses, add a guard test anyway:
```rust
#[test]
fn writer_sees_own_set_via_filter_not_just_projection() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (:Person {name: 'Ann', age: 30})").unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99").unwrap();
    let r = w.execute("MATCH (p:Person) WHERE p.age = 99 RETURN p.name").unwrap();
    assert_eq!(r.row_count(), 1, "writer must match its own uncommitted SET in WHERE");
    w.rollback().unwrap();
}
```
Place it in `mvcc_isolation.rs`. If it fails, it is the index-staleness case above — fix per the preferred path.

- [ ] **Step 5: Commit**
```bash
git add crates/grafeo-core/src/execution/operators/mutation.rs crates/grafeo-core/src/execution/operators/merge.rs crates/grafeo-engine/src/session/mod.rs crates/grafeo-engine/tests/mvcc_isolation.rs
git commit -m "feat(mvcc): buffer transactional property writes into the per-tx delta

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Isolate transactional node deletes (flips Probe 2)

**Files:**
- Modify: `crates/grafeo-common/src/mvcc.rs` (`finalize_deleted_epochs` + unit test)
- Modify: `crates/grafeo-core/src/graph/lpg/store/mod.rs` (`pending_tx_deletes` field)
- Modify: `crates/grafeo-core/src/graph/lpg/store/node_ops.rs` (PENDING delete + deferred index + finalize)
- Modify: `crates/grafeo-core/src/graph/lpg/store/graph_store_impl.rs` + `traits.rs` (finalize-deletes trait method)
- Modify: `crates/grafeo-engine/src/session/mod.rs` (call finalize on commit)

- [ ] **Step 1: Unit-test the PENDING-delete visibility assumption (mvcc.rs)**

Append to the `tests` module in `crates/grafeo-common/src/mvcc.rs`:
```rust
#[test]
fn pending_delete_is_invisible_to_deleter_visible_to_others() {
    let mut v = VersionInfo::new(EpochId::new(1), TransactionId::new(1));
    let deleter = TransactionId::new(7);
    v.mark_deleted(EpochId::PENDING, deleter);

    // Deleter sees its own delete (node gone for it).
    assert!(!v.is_visible_to(EpochId::new(5), deleter));
    // Everyone else still sees the node (delete not committed).
    assert!(v.is_visible_to(EpochId::new(5), TransactionId::new(8)));
    assert!(v.is_visible_at(EpochId::new(5)));
}

#[test]
fn finalize_deleted_epochs_makes_delete_visible_at_commit() {
    let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));
    let deleter = TransactionId::new(7);
    chain.mark_deleted(EpochId::PENDING, deleter);
    chain.finalize_deleted_epochs(deleter, EpochId::new(10));
    // After finalize at epoch 10: visible before 10, gone at/after 10.
    assert_eq!(chain.visible_at(EpochId::new(9)), Some(&"v1"));
    assert_eq!(chain.visible_at(EpochId::new(10)), None);
}
```

- [ ] **Step 2: Run; confirm `pending_delete_*` passes and `finalize_deleted_epochs` fails to compile**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-common pending_delete finalize_deleted`
Expected: `pending_delete_*` PASS (validates the visibility assumption today), `finalize_deleted_epochs` FAILS to compile (method missing).

- [ ] **Step 3: Add `finalize_deleted_epochs` to `VersionChain` (and `VersionIndex` under `tiered-storage`)**

In `mvcc.rs`, in `impl<T> VersionChain<T>` next to `finalize_epochs` (line 233):
```rust
    /// Finalizes PENDING `deleted_epoch`s for versions deleted by the given
    /// transaction. Called at commit to make a delete visible at the real
    /// commit epoch instead of `EpochId::PENDING`.
    pub fn finalize_deleted_epochs(&mut self, transaction_id: TransactionId, commit_epoch: EpochId) {
        for version in &mut self.versions {
            if version.info.deleted_by == Some(transaction_id)
                && version.info.deleted_epoch == Some(EpochId::PENDING)
            {
                version.info.deleted_epoch = Some(commit_epoch);
            }
        }
    }
```
Add the analogous method to `impl VersionIndex` (cfg `tiered-storage`), iterating `self.hot`/`self.cold` and setting `deleted_epoch = OptionalEpochId::some(commit_epoch)` where `deleted_by == Some(tx) && deleted_epoch.get() == Some(EpochId::PENDING)`, then `recalculate_latest_epoch()`.

- [ ] **Step 4: Run mvcc tests; confirm GREEN**

Run: `CARGO_INCREMENTAL=0 cargo test -p grafeo-common --features tiered-storage finalize_deleted pending_delete`
Then also without the feature: `CARGO_INCREMENTAL=0 cargo test -p grafeo-common finalize_deleted pending_delete`
Expected: PASS in both.

- [ ] **Step 5: Add a `pending_tx_deletes` index to the store**

In `crates/grafeo-core/src/graph/lpg/store/mod.rs`, next to `pending_tx_creates` (around line 497), add a field and its init (default empty), mirroring the creates index:
```rust
    /// Per-transaction lists of node IDs deleted with a PENDING `deleted_epoch`,
    /// recorded at `delete_node_transactional` — the chokepoint that defers
    /// label-index/adjacency removal until commit. Finalized by
    /// `finalize_deletes_by_id`, dropped on rollback.
    pub(crate) pending_tx_deletes: RwLock<FxHashMap<TransactionId, Vec<NodeId>>>,
```
Match the exact lock/map types and constructor style already used for `pending_tx_creates` in this file.

- [ ] **Step 6: Make `delete_node_transactional` defer and stamp PENDING**

In `node_ops.rs:621-...` (non-tiered) — and the tiered twin at 723:
  - Change `chain.mark_deleted(epoch, transaction_id);` (line 638) to `chain.mark_deleted(EpochId::PENDING, transaction_id);`.
  - **Remove the eager label-index strip** (lines 675-688: the `node_labels_w.remove(&id)` + `index.get_mut(...).remove(&id)` block). Instead, record the node for deferred finalize:
```rust
            self.pending_tx_deletes
                .write()
                .entry(transaction_id)
                .or_default()
                .push(id);
```
  - Keep capturing labels/properties for the existing `NodeDeleted` undo entry (rollback restores via `restore_deleted_node`, which calls `unmark_deleted_by` — that already clears the PENDING `deleted_epoch`). Keep `live_node_count` semantics consistent: the existing path decrements on delete; since the label-index removal is now deferred, ensure the count is only decremented at finalize (commit) — move the `live_node_count` decrement out of `delete_node_transactional` into `finalize_deletes_by_id` (Step 7), or leave it and accept that `live_node_count` is an approximate stat (check how it is consumed; if only used for `count(*)`-style stats that the probes do not assert, a code comment noting the approximation is acceptable for this increment).
  - For `DETACH`, `delete_node_edges` (node_ops.rs:835/919) currently uses `TransactionId::SYSTEM` + eager `batch_mark_deleted`. For the writing transaction, defer: when called within a transaction, stamp adjacency deletions as part of the pending set OR skip eager adjacency and let `finalize_deletes_by_id` apply it. Probe 2's node has no edges, so the minimal correct change is to ensure the transactional DETACH path does not eagerly tombstone adjacency for *other* readers; if threading `transaction_id` into `delete_node_edges` is more than a small change, scope this task to node-version isolation (sufficient for Probe 2) and leave a `// TODO(unified-mvcc): defer adjacency tombstones for transactional DETACH` with a tracked follow-up test asserting an *edge* delete is isolated (mark `#[ignore]` with a clear reason). Decide based on the actual call shape; prefer the complete fix if it is localized.

- [ ] **Step 7: Add `finalize_deletes_by_id` on the store + trait method**

In `node_ops.rs`, add:
```rust
    /// Finalizes PENDING deletes for a committed transaction: stamps each
    /// deleted version's `deleted_epoch` PENDING→commit_epoch and applies the
    /// deferred label-index/adjacency removal.
    pub(crate) fn finalize_deletes_by_id(&self, transaction_id: TransactionId, commit_epoch: EpochId, node_ids: &[NodeId]) {
        let mut nodes = self.nodes.write();
        for &id in node_ids {
            if let Some(chain) = nodes.get_mut(&id) {
                chain.finalize_deleted_epochs(transaction_id, commit_epoch);
            }
        }
        drop(nodes);
        // Apply the deferred label-index removal now that the delete is committed.
        let mut node_labels_w = self.node_labels.write();
        let mut index = self.label_index.write();
        for &id in node_ids {
            if let Some(removed) = node_labels_w.remove(&id) {
                #[cfg(not(feature = "temporal"))]
                let label_ids = removed;
                #[cfg(feature = "temporal")]
                let label_ids = removed.latest().cloned().unwrap_or_default();
                for label_id in label_ids {
                    if let Some(set) = index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }
            self.live_node_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    /// Takes (removes and returns) the pending delete list for a transaction.
    pub(crate) fn take_pending_deletes(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.pending_tx_deletes.write().remove(&transaction_id).unwrap_or_default()
    }
```
Expose a trait method `finalize_deletes_by_id(&self, tx, commit_epoch, node_ids: &[NodeId])` and `take_pending_deletes(&self, tx) -> Vec<NodeId>` on `GraphStoreMut` (defaults: no-op / empty) and override on `LpgStore` in `graph_store_impl.rs`, mirroring `finalize_entities_by_id` / `take_pending_creates`.

- [ ] **Step 8: Call finalize on commit, drop pending-deletes on rollback**

In `session/mod.rs` `commit_inner` success loop (next to `finalize_entities_by_id` + `apply_tx_overlay` from Task 3):
```rust
            let pending_deletes = store.take_pending_deletes(transaction_id);
            store.finalize_deletes_by_id(transaction_id, commit_epoch, &pending_deletes);
```
In `rollback_inner` and the commit conflict branch: the existing `unmark_deleted_by`/undo-log path restores the node; add `let _ = store.take_pending_deletes(transaction_id);` to clear the deferred set so it is not finalized later.

- [ ] **Step 9: Run Probe 2 + tracked delete tests; confirm GREEN**

Run:
```bash
CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine \
  --test mvcc_isolation uncommitted_delete_is_invisible_to_other_sessions committed_delete_is_visible_to_other_sessions \
  --test audit_scratch uncommitted_delete_invisible_to_others
```
Expected: all PASS.

- [ ] **Step 10: Commit**
```bash
git add crates/grafeo-common/src/mvcc.rs crates/grafeo-core/src/graph/lpg/store/ crates/grafeo-core/src/graph/traits.rs crates/grafeo-engine/src/session/mod.rs
git commit -m "feat(mvcc): isolate transactional node deletes via PENDING deleted_epoch + deferred finalize

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Wrapper delegation + completeness sweep

**Files:**
- Modify: `crates/grafeo-engine/src/database/cdc_store.rs`
- Modify: `crates/grafeo-engine/src/database/wal_store.rs`
- Modify: `crates/grafeo-engine/src/database/async_wal_store.rs`
- Modify: `crates/grafeo-core/src/graph/compact/layered.rs`

- [ ] **Step 1: Delegate the new trait methods through each wrapper**

`CdcStore`, `WalStore`, `AsyncWalStore` wrap an inner `GraphStore`/`GraphStoreMut`. For each, override `read_node_property_visible`, `read_edge_property_visible`, the four `*_buffered`, `apply_tx_overlay`, `drop_tx_overlay`, `finalize_deletes_by_id`, `take_pending_deletes` to delegate to `self.inner.<method>(...)` (matching how they already delegate `*_versioned` and `get_*_property`). For `LayeredStore` (`compact/layered.rs`), delegate the same to its inner `LpgStore` so compacted/overlay builds also get isolation. The buffered write + accessor both operate on the inner `LpgStore`'s delta, so delegation is sufficient — no overlay-specific merge needed for this increment.

- [ ] **Step 2: Run the full gate**

Run: `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine`
Expected: PASS, including both probes and all `mvcc_isolation` tests.

- [ ] **Step 3: Commit**
```bash
git add crates/grafeo-engine/src/database/ crates/grafeo-core/src/graph/compact/layered.rs
git commit -m "feat(mvcc): delegate snapshot accessor + delta through store wrappers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Full verification, lint, profiles, docs

- [ ] **Step 1: The real gate**

Run: `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine`
Expected: PASS. Confirm in output that `audit_scratch::uncommitted_delete_invisible_to_others`, `audit_scratch::uncommitted_property_write_invisible_to_others`, and all `mvcc_isolation::*` pass.

- [ ] **Step 2: Clippy adds zero new warnings**

Run: `CARGO_INCREMENTAL=0 cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings`
Expected: clean. Fix any new warnings (e.g., unused `epoch` params in default trait bodies — already `let _ = ...`).

- [ ] **Step 3: Feature-profile + WASM compile checks**

Run:
```bash
CARGO_INCREMENTAL=0 cargo check -p grafeo-core --no-default-features --features lpg
CARGO_INCREMENTAL=0 cargo check -p grafeo-core --no-default-features --features "lpg,analytics"
CARGO_INCREMENTAL=0 cargo check -p grafeo-core --features temporal
CARGO_INCREMENTAL=0 cargo check -p grafeo-core --features tiered-storage
# WASM crate (adjust path/target to the repo's wasm crate)
CARGO_INCREMENTAL=0 cargo check -p grafeo-wasm --target wasm32-unknown-unknown
```
Expected: all compile. The `temporal` and `tiered-storage` cfgs touch `read_*_property_visible` (temporal uses `get_at`) and `finalize_deleted_epochs`/`VersionIndex` — verify both build.

- [ ] **Step 4: Update the changelog + audit memory**

Add a changelog entry (neutral wording, no private project name) describing snapshot-isolated transactional property writes and node deletes. Update `project_grafeo_audit.md` memory: mark Wave 2b first-increment as implemented (probes 2 + 3 green), note any deferred items (savepoint-granular delta, transactional DETACH adjacency, label snapshot reads, whole-node `RETURN n` engine-side materialization, writer-own index-scan fallback if not needed).

- [ ] **Step 5: OPSEC scan before any push**

Run: `git log --oneline origin/integration..HEAD` and `git diff origin/integration..HEAD` — scan diffs and commit messages for the private project's name/schema/business logic. Do not push in-session unless the user asks; if pushing, network needs `dangerouslyDisableSandbox`.

- [ ] **Step 6: Final commit**
```bash
git add -A
git commit -m "docs(mvcc): changelog + audit notes for unified-MVCC increment 1

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Acceptance

- `audit_scratch::uncommitted_property_write_invisible_to_others` and `audit_scratch::uncommitted_delete_invisible_to_others` pass with `#[ignore]` removed.
- All `mvcc_isolation::*` tests pass (read-your-writes for SET and DELETE; committed changes visible to others; uncommitted invisible; rollback restores).
- `cargo test --all-features -p grafeo-core -p grafeo-engine` green; clippy adds no new warnings; lpg / analytics / temporal / tiered-storage profiles and the wasm crate compile.
- The committed columnar store is never mutated by an uncommitted write (no analytical capability lost).

## Risks & mitigations

- **Missed read site = residual dirty read.** Completeness is the whole game. Task 2's grep enumeration is the checklist; Task 7's full suite is the backstop. Whole-node `RETURN n` engine-side materialization is explicitly deferred — note it so it is not mistaken for done.
- **Index staleness within a writing transaction** (Task 4 Step 4): buffered writes do not update property/text indexes, so the writer's own indexed scans can miss its writes. Mitigation: writing-tx scans fall back to non-indexed scan+delta-merge; never update the index on a buffered write.
- **Delete completeness for edges**: this increment guarantees node-version isolation (Probe 2). Transactional DETACH adjacency deferral may be split to a follow-up with an `#[ignore]`d edge-delete probe if not localized — make that explicit, do not silently leave eager adjacency that other readers can observe.
- **`live_node_count` accuracy** across deferred deletes: move the decrement to finalize, or document the approximation if the stat is not asserted by tests.
- **Savepoint partial rollback** of buffered writes is tx-granular in this increment (Task 3 Step 4) — deferred unless an existing test forces savepoint stamping.
