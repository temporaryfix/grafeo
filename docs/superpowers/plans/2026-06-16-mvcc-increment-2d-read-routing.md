# Increment 2d — Complete read routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread the transaction snapshot `(viewing_epoch, transaction_id)` through the final remaining `TODO(unified-mvcc)` read sites — property introspection (`keys`/`properties`/`property_values`/`property_exists`), the mutation source-property read, factorized filters, horizontal aggregation, and vector joins — so the writing transaction sees its own buffered `SET`/`REMOVE` (read-your-writes), closing the documented gaps and completing the read-routing surface that increment **2e** (the SSI tracking foundation) will instrument with `record_read`.

**Architecture:** Two fix shapes. **(1) Snapshot-aware sites** — `FilterOperator`'s property-introspection builtins (`filter.rs`) and `PropertySource::PropertyAccess` (`mutation.rs`) — already carry (or can reach) `(viewing_epoch, transaction_id)`; reroute their committed reads to the existing delta-merged accessors (`read_node_properties_visible`/`read_edge_properties_visible` for whole-set; `read_node_property_visible`/`read_edge_property_visible` for single-prop). **(2) Field-less operators** — `HorizontalAggregateOperator`, `VectorJoinOperator`, and the factorized-filter `PropertyPredicate` path — gain `viewing_epoch`/`transaction_id` fields + a `with_transaction_context` builder, plumbed from the planner exactly as `expand`/`aggregate`/`scan`/`filter` already do, then read through the visible accessors.

**Tech Stack:** Rust, `cargo test`. Operators in `grafeo-core` (`execution/operators/{filter,mutation,factorized_filter,horizontal_aggregate,vector_join}.rs`); planner plumbing in `grafeo-engine` (`query/planner/lpg/{mod,…}.rs`). The visible accessors already exist on the store trait (`graph/traits.rs`) and `LpgStore` (`graph/lpg/store/property_ops.rs`). `CARGO_INCREMENTAL=0`.

---

## Orientation

This is **Part D's behavior-preserving half** of `docs/superpowers/specs/2026-06-15-unified-mvcc-increment-2-serializable-design.md` (§6). Parts A (labels), B (edge-deletes), C (MERGE) are merged to `integration` (Plan 1 complete, tip `78201cb7`). §11 splits the remaining work along an explicit seam — *"routing is behavior-preserving, registry/read-set inert for non-Serializable"* — and this plan (2d) takes the routing; **2e** takes the inert tracking (`record_read` + read-registry + store-derived write-set). Do **not** add any `record_read`, read-registry, or write-set work here — 2d is reads-see-the-snapshot only.

**What already exists (do not rebuild):**
- **Delta-merged visible accessors** on the store trait `graph/traits.rs` and `LpgStore`:
  - `read_node_property_visible(id, &PropertyKey, epoch, Option<tx>) -> Option<Value>` (single prop; `traits.rs:100`, used by `merge.rs`/`filter.rs::resolve_node`).
  - `read_node_properties_visible(id, epoch, Option<tx>)` + `read_edge_properties_visible(id, epoch, Option<tx>)` — **whole-set, committed-merged-with-`TxDelta`** (`traits.rs:129/142`; `LpgStore` impl `property_ops.rs:1256/1297`; layered merge test `layered.rs:4323`). Confirm the exact return type when implementing (a `Vec<(PropertyKey, Value)>`-shaped whole set) and adapt iteration to it.
  - Edge twins `read_edge_property_visible` (`traits.rs:106`).
- **Canonical planner→operator plumbing:** `operator.with_transaction_context(self.viewing_epoch, self.transaction_id)` — see `expand.rs:73`, `aggregate.rs:166/305`, `scan.rs:23/53`, `filter_hybrid.rs:205`. The planner holds `viewing_epoch: EpochId` and `transaction_id: Option<TransactionId>` (`planner/lpg/mod.rs`).
- **`FilterOperator` already has the snapshot:** `viewing_epoch: Option<EpochId>` + `transaction_id: Option<TransactionId>` (`filter.rs:227-229`), `with_transaction_context` (`filter.rs:523`), and `resolve_node` (`filter.rs:540`) already branches on `(self.viewing_epoch, self.transaction_id)`. So the introspection reroute (Task 1) needs **no** new plumbing — just the visible accessor calls.
- **Label reads in `filter.rs` are already routed (done in 2a):** `labels` (`filter.rs:1773`), `haslabel` (`filter.rs:1879`), and the label sites at `:598/:909/:3349/:3619` already call `read_node_labels_visible(node_id, snap_epoch, self.transaction_id)`. Spec §6 lists "label reads in filter.rs" under Part D, but increment 2a (Part A) completed them — **no 2d task is needed for labels.** Do not re-touch them.

**The 9 target sites (all marked `// TODO(unified-mvcc)`):**
| Site | File:line | Shape | Task |
|---|---|---|---|
| `property_exists` | `filter.rs:1857` | snapshot-aware reroute (whole-set) | 1 |
| `keys` / `properties` / `property_values` | `filter.rs:2163/2198/2228` | snapshot-aware reroute (whole-set) | 1 |
| `PropertySource::PropertyAccess` | `mutation.rs:214` | reroute (single-prop) + thread snapshot into helper | 2 |
| factorized `PropertyPredicate` | `factorized_filter.rs:358/409` | add fields + plumb + reroute | 3 |
| `HorizontalAggregateOperator` | `horizontal_aggregate.rs:78` | add fields + plumb + reroute | 4 |
| `VectorJoinOperator` | `vector_join.rs:238/284` | add fields + plumb + reroute | 5 |

**Out of scope (explicit):** the CDC/WAL `TODO(unified-mvcc)` sites (`database/wal_store.rs:395/708`, `database/cdc_store.rs:465/1053`) — those wrappers stay write-through by design (spec §1 "Out of scope"). Touch only the `LpgStore`/operator paths.

**Test tiers.** The clearly-observable sites (filter introspection, mutation source, horizontal aggregate) get a session-level RED→GREEN read-your-writes probe in `crates/grafeo-engine/tests/mvcc_isolation.rs` (18 tests today; `GrafeoDB::new_in_memory()` / `session()` / `begin_transaction()` / `execute()` / `row_count()`). The deep/feature-gated operators (factorized filter, vector join) may not be reachable by a transactional session query that isolates them; for those, do the **mechanical** snapshot-threading and verify **no regression** + confirm-by-inspection that the read now passes `self.transaction_id` (not `None`). If a probe you expected to go RED stays green, the query routed through a different operator — **investigate and report, do not force it**.

**Verification gate (every task):** `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` AND `--features full -p grafeo-engine --test mvcc_isolation`. Keep `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean. **Hygiene:** format only changed files (`rustfmt <file>`, not whole-crate `cargo fmt`); confirm `git status` shows only intended files before committing. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. **OPSEC:** generic labels only (`:Thing`/`:N`); never the private project's schema/names.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `crates/grafeo-engine/tests/mvcc_isolation.rs` | acceptance probes | Modify (read-your-writes probes) |
| `crates/grafeo-core/src/execution/operators/filter.rs` | property introspection | Modify (reroute 4 introspection builtins to whole-set visible accessor) |
| `crates/grafeo-core/src/execution/operators/mutation.rs` | `PropertySource::PropertyAccess` | Modify (thread snapshot into `resolve`; single-prop visible read) |
| `crates/grafeo-core/src/execution/operators/factorized_filter.rs` | `PropertyPredicate` | Modify (add snapshot fields + builder + reroute) |
| `crates/grafeo-core/src/execution/operators/horizontal_aggregate.rs` | horizontal aggregation | Modify (add snapshot fields + builder + reroute) |
| `crates/grafeo-core/src/execution/operators/vector_join.rs` | vector join | Modify (add snapshot fields + builder + reroute) |
| `crates/grafeo-engine/src/query/planner/lpg/*.rs` | planner plumbing | Modify (append `.with_transaction_context(self.viewing_epoch, self.transaction_id)` at the 3 field-less operators' construction sites) |

---

## Task 0: TDD baseline — read-your-writes probes, confirm RED

**Files:** Modify `crates/grafeo-engine/tests/mvcc_isolation.rs`

Add the probes below. **First** grep the existing suite for the function/clause forms (`property_exists`, `keys(`, list/aggregate syntax) and confirm the supported dialect, adjusting query strings to match while keeping the assertions/semantics intact (mirror the 2c "confirm syntax" discipline).

- [ ] **Step 1: Add the observable-site probes.**
```rust
#[test]
fn property_exists_reflects_same_tx_set() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (n:Thing {a: 1})").unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (n:Thing) SET n.b = 2").unwrap();
    // Writer must see its own buffered SET via property_exists (read-your-writes).
    let r = w
        .execute("MATCH (n:Thing) WHERE property_exists(n, 'b') RETURN n")
        .unwrap();
    assert_eq!(r.row_count(), 1, "writer sees its own buffered SET via property_exists");
    w.commit().unwrap();
}

#[test]
fn mutation_source_property_reflects_same_tx_set() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (n:Thing {a: 1})").unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (n:Thing) SET n.a = 5").unwrap();
    // A later SET sourced from n.a must read the tx's own buffered value (5), not committed (1).
    w.execute("MATCH (n:Thing) SET n.b = n.a").unwrap();
    let r = w
        .execute("MATCH (n:Thing) WHERE n.b = 5 RETURN n")
        .unwrap();
    assert_eq!(r.row_count(), 1, "mutation source read sees the tx's own buffered SET");
    w.commit().unwrap();
}

#[test]
fn horizontal_aggregate_reflects_same_tx_set() {
    // NOTE: confirm the query shape that compiles to HorizontalAggregateOperator
    // (variable-length-path / group-list aggregation, planner GE09). If no
    // transactional session query isolates this operator, demote this to a
    // no-regression assertion and rely on Task 4's by-inspection verification.
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (a:N {id: 1, v: 1})-[:R]->(b:N {id: 2, v: 1})").unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (n:N) SET n.v = 10").unwrap();
    // A horizontal aggregate over n.v must read the tx's buffered value (10).
    // Assertion shape TBD against the dialect's aggregate-over-list form.
    w.commit().unwrap();
}
```
- [ ] **Step 2: Run; confirm RED** for the first two (they exercise observable gaps): `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation property_exists_reflects mutation_source_property`. Expected: both FAIL (`row_count == 0` — the writer's buffered value was invisible). If either passes, the query didn't hit the target operator — investigate the plan and report before proceeding. The `horizontal_aggregate` probe is a placeholder to finalize in Task 4.
- [ ] **Step 3: Commit** the failing baseline (`test(mvcc): failing read-routing read-your-writes probes`).

---

## Task 1: `filter.rs` property introspection → whole-set visible accessor

**Files:** `execution/operators/filter.rs` (the 4 introspection builtins at `:1857`, `:2163`, `:2198`, `:2228`)

`FilterOperator` already carries `(viewing_epoch, transaction_id)` and `resolve_node`/`resolve_edge` already use them. The introspection builtins instead read `node.properties` / `edge.properties` off the **committed** resolved entity. Reroute them to the delta-merged whole-set accessor.

- [ ] **Step 1: Add private helpers** on `FilterOperator` (near `resolve_node`, `filter.rs:540`), mirroring its `(viewing_epoch, transaction_id)` branch. Confirm the return type of `read_node_properties_visible` (`traits.rs:129`) and adapt:
```rust
/// Snapshot-consistent whole property set for a node: the writing tx's buffered
/// SET/REMOVE merged over the committed set (read-your-writes), else committed.
fn visible_node_properties(&self, id: NodeId) -> Vec<(PropertyKey, Value)> {
    let epoch = self.viewing_epoch.unwrap_or_else(|| self.store.current_epoch());
    self.store
        .read_node_properties_visible(id, epoch, self.transaction_id)
}

fn visible_edge_properties(&self, id: EdgeId) -> Vec<(PropertyKey, Value)> {
    let epoch = self.viewing_epoch.unwrap_or_else(|| self.store.current_epoch());
    self.store
        .read_edge_properties_visible(id, epoch, self.transaction_id)
}
```
- [ ] **Step 2: Reroute `property_exists`** (`filter.rs:1857`), deleting the `TODO(unified-mvcc)` comment:
```rust
if let Some(nid) = col.get_node_id(row) {
    let exists = self
        .visible_node_properties(nid)
        .iter()
        .any(|(k, _)| k.as_str() == key.as_str());
    return Some(Value::Bool(exists));
}
if let Some(eid) = col.get_edge_id(row) {
    let exists = self
        .visible_edge_properties(eid)
        .iter()
        .any(|(k, _)| k.as_str() == key.as_str());
    return Some(Value::Bool(exists));
}
```
- [ ] **Step 3: Reroute the sibling builtins** at `filter.rs:2163`, `:2198`, `:2228` (`keys` / `properties` / `property_values` — confirm which function sits at each line). Replace each committed `node.properties` / `edge.properties` read with `self.visible_node_properties(nid)` / `self.visible_edge_properties(eid)`, preserving each function's output shape (key list / map / value list). Delete each `TODO(unified-mvcc)` comment.
- [ ] **Step 4: Run GREEN.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation property_exists_reflects` → PASS. `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green; clippy clean. Watch for any existing introspection test asserting the old committed-read behavior — if one breaks, determine whether it asserted the buggy pre-routing behavior and **report**, don't blindly edit. Commit (`fix(mvcc): property introspection reads tx delta (read-your-writes)`).

---

## Task 2: `mutation.rs` `PropertySource::PropertyAccess` → single-prop visible read

**Files:** `execution/operators/mutation.rs` (`PropertySource::resolve`, `:200-230`)

`PropertySource::resolve` takes `store: &dyn GraphStore` but no snapshot, so `PropertyAccess` reads committed `get_node().get_property()`. Thread the snapshot from the calling operator (mutation operators already carry `viewing_epoch`/`transaction_id` — see the planner's `with_transaction_context` calls in `planner/lpg/mutation.rs`).

- [ ] **Step 1: Add snapshot params to `resolve`.** Change the signature to accept `epoch: Option<EpochId>` and `transaction_id: Option<TransactionId>` (or a small `Snapshot` newtype if cleaner), and update the call site(s) to pass the operator's `self.viewing_epoch`/`self.transaction_id`. Grep the call sites of `PropertySource::resolve` and thread the operator's fields through.
- [ ] **Step 2: Reroute the node/edge arms** (`mutation.rs:214`), deleting the `TODO(unified-mvcc)` comment. **Confirm the store trait** passed to `resolve` exposes `read_node_property_visible` (it is on `GraphStoreSearch`/the visible-read trait; if `resolve` is handed a narrower `&dyn GraphStore`, widen that parameter to the trait that carries the visible accessors — this is the main friction in this task; if widening ripples further than this operator, **stop and report** rather than restructuring broadly):
```rust
PropertySource::PropertyAccess { column, property } => {
    let Some(col) = chunk.column(*column) else { return Value::Null; };
    let prop_key = PropertyKey::new(property);
    if let Some(node_id) = col.get_node_id(row) {
        match (epoch, transaction_id) {
            (Some(ep), Some(tx)) => store
                .read_node_property_visible(node_id, &prop_key, ep, Some(tx))
                .unwrap_or(Value::Null),
            _ => store
                .get_node(node_id)
                .and_then(|node| node.get_property(property).cloned())
                .unwrap_or(Value::Null),
        }
    } else if let Some(edge_id) = col.get_edge_id(row) {
        match (epoch, transaction_id) {
            (Some(ep), Some(tx)) => store
                .read_edge_property_visible(edge_id, &prop_key, ep, Some(tx))
                .unwrap_or(Value::Null),
            _ => store
                .get_edge(edge_id)
                .and_then(|edge| edge.get_property(property).cloned())
                .unwrap_or(Value::Null),
        }
    } else if let Some(Value::Map(map)) = col.get_value(row) {
        let key = PropertyKey::new(property);
        map.get(&key).cloned().unwrap_or(Value::Null)
    } else {
        Value::Null
    }
}
```
- [ ] **Step 3: Run GREEN.** `CARGO_INCREMENTAL=0 cargo test --features full -p grafeo-engine --test mvcc_isolation mutation_source_property` → PASS. Full `--all-features` green; clippy clean. Commit (`fix(mvcc): mutation source property reads tx delta (read-your-writes)`).

---

## Task 3: `factorized_filter.rs` `PropertyPredicate` → thread snapshot

**Files:** `execution/operators/factorized_filter.rs` (`PropertyPredicate`, reads at `:358/:409`); planner construction site (locate)

`PropertyPredicate::new(col, …, prop, op, value, store)` carries no snapshot, so its property reads (`:358/:409`) use `current_epoch()` + `None`.

- [ ] **Step 1: Add fields + builder.** Add `viewing_epoch: Option<EpochId>` and `transaction_id: Option<TransactionId>` to `PropertyPredicate` (default `None` in `new`), plus a `with_transaction_context(mut self, epoch: EpochId, transaction_id: Option<TransactionId>) -> Self` setting `viewing_epoch = Some(epoch)` (mirror `FilterOperator::with_transaction_context`, `filter.rs:523`).
- [ ] **Step 2: Reroute the reads** at `:358/:409`, deleting the `TODO(unified-mvcc)` comments: use `self.viewing_epoch.unwrap_or_else(|| self.store.current_epoch())` for the epoch and `self.transaction_id` for the tx in the `read_node_property_visible`/`read_edge_property_visible` call (replace the `None`).
- [ ] **Step 3: Plumb from the planner.** Locate where `PropertyPredicate` is constructed in production (grep `PropertyPredicate::new` outside `#[cfg(test)]` — likely under `query/planner/lpg/`; if the factorized-filter predicate is built in a helper that already has `self.viewing_epoch`/`self.transaction_id`, append `.with_transaction_context(self.viewing_epoch, self.transaction_id)`). If **no** production construction site exists (predicate only built in tests / behind a disabled path), record that finding, apply the field+builder change for completeness, and verify by inspection — **do not invent a wiring path**.
- [ ] **Step 4: Run.** Full `--all-features -p grafeo-core -p grafeo-engine` green; `--features full` mvcc_isolation green; clippy clean. If you added a session-reachable RED probe for factorized filtering, confirm it goes GREEN; otherwise verify no regression. Commit (`feat(mvcc): factorized filter predicate reads tx snapshot`).

---

## Task 4: `horizontal_aggregate.rs` → snapshot fields + plumb

**Files:** `execution/operators/horizontal_aggregate.rs` (`:40-103`); `query/planner/lpg/mod.rs:849` (construction)

- [ ] **Step 1: Add fields + builder.** Add `viewing_epoch: Option<EpochId>` and `transaction_id: Option<TransactionId>` to `HorizontalAggregateOperator` (default `None` in `new`, `:55`), plus `with_transaction_context(mut self, epoch: EpochId, transaction_id: Option<TransactionId>) -> Self` (mirror `filter.rs:523`).
- [ ] **Step 2: Reroute `get_property_value`** (`:76-103`), deleting the `TODO(unified-mvcc)` comment:
```rust
fn get_property_value(&self, entity_value: &Value) -> Option<Value> {
    let prop_key = PropertyKey::new(&self.property);
    let snap_epoch = self.viewing_epoch.unwrap_or_else(|| self.store.current_epoch());
    match self.entity_kind {
        EntityKind::Edge => {
            let id = match entity_value {
                #[allow(clippy::cast_sign_loss)]
                Value::Int64(i) => EdgeId(*i as u64),
                _ => return None,
            };
            self.store
                .read_edge_property_visible(id, &prop_key, snap_epoch, self.transaction_id)
        }
        EntityKind::Node => {
            let id = match entity_value {
                #[allow(clippy::cast_sign_loss)]
                Value::Int64(i) => NodeId(*i as u64),
                _ => return None,
            };
            self.store
                .read_node_property_visible(id, &prop_key, snap_epoch, self.transaction_id)
        }
    }
}
```
- [ ] **Step 3: Plumb from the planner** at `mod.rs:849` — append to the `HorizontalAggregateOperator::new(…)` builder chain:
```rust
.with_transaction_context(self.viewing_epoch, self.transaction_id)
```
(confirm `EpochId`/`TransactionId` are imported in `horizontal_aggregate.rs`; they are used as `EdgeId`/`NodeId` neighbors already).
- [ ] **Step 4: Finalize the Task 0 `horizontal_aggregate` probe.** Settle the aggregate-over-list query shape + assertion; if it goes RED→GREEN, keep it; if the operator isn't reachable from a transactional session query, demote to a no-regression check and note the by-inspection verification (the read now passes `self.transaction_id`). Full `--all-features` green; clippy clean. Commit (`feat(mvcc): horizontal aggregate reads tx snapshot`).

---

## Task 5: `vector_join.rs` → snapshot fields + plumb

**Files:** `execution/operators/vector_join.rs` (reads at `:238/:284`); planner construction site (locate)

- [ ] **Step 1: Add fields + builder.** Add `viewing_epoch: Option<EpochId>` + `transaction_id: Option<TransactionId>` (default `None`) and `with_transaction_context` (mirror `filter.rs:523`). `VectorJoinOperator` has multiple constructors (`with_static_query`, `entity_to_entity`) — set the fields to `None` in each and rely on the builder.
- [ ] **Step 2: Reroute the reads** at `:238/:284`, deleting the `TODO(unified-mvcc)` comments — epoch `self.viewing_epoch.unwrap_or_else(|| self.store.current_epoch())`, tx `self.transaction_id`.
- [ ] **Step 3: Plumb from the planner.** Locate the production `VectorJoinOperator` construction (grep outside `#[cfg(test)]`; vector join is feature-gated — it may only build under the vector feature / a vector-index plan). Append `.with_transaction_context(self.viewing_epoch, self.transaction_id)`. If only test constructions exist, record that, apply the field+builder change, and verify by inspection — do not invent wiring.
- [ ] **Step 4: Run.** Full `--all-features -p grafeo-core -p grafeo-engine` green; `--features full` mvcc_isolation green; clippy clean. Commit (`feat(mvcc): vector join reads tx snapshot`).

---

## Task 6: Full verification + OPSEC

- [ ] `CARGO_INCREMENTAL=0 cargo test --all-features -p grafeo-core -p grafeo-engine` green; `--features full -p grafeo-engine --test mvcc_isolation` green (incl. the new probes). If any pre-existing test broke, determine whether it asserted the old committed-read behavior and **report** (don't blindly edit).
- [ ] `cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings` clean.
- [ ] `default`/`lpg`/`temporal`/`tiered-storage` profiles + `grafeo-wasm` (`wasm32-unknown-unknown`) compile.
- [ ] `git status` clean (only intended files; revert collateral fmt). OPSEC: scan the diff for the private project's name/schema (generic `:Thing`/`:N` only).
- [ ] Confirm **no** `record_read` / read-registry / write-set code crept in (that is 2e). Confirm zero remaining `TODO(unified-mvcc)` read sites in the five operator files (`grep -rn "TODO(unified-mvcc)" execution/operators/{filter,mutation,factorized_filter,horizontal_aggregate,vector_join}.rs` → empty; the CDC/WAL TODOs remain, by design).

---

## Acceptance
- The writing transaction sees its own buffered `SET`/`REMOVE` through property introspection (`property_exists`, and `keys`/`properties`/`property_values`), the mutation source-property read, and (by test or by-inspection) factorized filters, horizontal aggregation, and vector joins.
- No cross-session dirty read is introduced (these were never cross-session leaks — only writer-own read-your-writes gaps).
- All five operator files are free of `TODO(unified-mvcc)` read markers; the CDC/WAL sites remain (out of scope).
- Full `--all-features` + `--features full` green; clippy clean; profiles + wasm compile; tree clean; OPSEC-clean.
- **Part D's read-routing surface is complete** → next: **2e** — `ReadTracker`/`record_read` wiring at these (now-complete) read sites + the sharded read-registry + the store-derived write-set (Part D's inert half + Part E). Then Plan 3 (F1 OCC rung → F2 SSI + G performance).

## Risks
- **Deep operators may not be session-reachable.** Factorized filter and vector join are selected only under specific plans/features; a transactional session query may not isolate them. The threading is still required (so 2e's `record_read` sees a `transaction_id` there, and read-your-writes holds whenever they *are* hit). Verify by no-regression + inspection; **report** if a RED probe can't be constructed — don't force a contrived one or fake a wiring path.
- **`mutation.rs` trait-surface friction.** `PropertySource::resolve` takes `&dyn GraphStore`, which may not expose the visible accessors. Widening to the visible-read trait is the intended fix, but if it ripples beyond this operator, stop and report (don't broadly restructure — see `feedback_no_gratuitous_refactors`).
- **Whole-set accessor return type.** `read_node_properties_visible`'s exact type (Vec vs map) determines the introspection iteration; confirm at `traits.rs:129` and adapt the Task 1 helpers.
- **Behavior-change vs behavior-preserving.** Routing closes read-your-writes gaps (a writer-visible change), but must not alter committed/cross-session results. If an existing test changes, confirm it asserted the pre-routing committed behavior before touching it.
- **Stay in lane.** No `record_read`, no registry, no write-set work — those are 2e. Keep this increment purely "reads resolve at the snapshot."

---

## STATUS: COMPLETE

All 6 tasks landed on `feat/mvcc-increment-2d` (commits `c4f4ac3f` → `1e82999d`, branched from `integration` @ `78201cb7`), via subagent-driven-development (fresh implementer per task + spec/quality review) and a final holistic review.

**Final verification:**
- `--all-features -p grafeo-core -p grafeo-engine` = **7411 passed / 0 failed** (121 binaries; baseline 7408 + 3 new probes).
- `mvcc_isolation` = **21/21** (18 prior + `property_exists_reflects_same_tx_set` GREEN, `mutation_source_property_reflects_same_tx_set` GREEN; `horizontal_aggregate_reflects_same_tx_set` is a no-regression baseline).
- Established clippy gate (`cargo clippy --all-features -p grafeo-core -p grafeo-engine -- -D warnings`) **clean**.
- Profile compiles: **default / lpg / temporal / tiered-storage** all green; **`grafeo-wasm` (wasm32-unknown-unknown)** compiles.
- OPSEC-clean (generic `:Thing`/`:N`); **zero `TODO(unified-mvcc)`** in the 5 operator files; **no `record_read`/registry/write-set** code (that is 2e).
- Final holistic review (opus): **READY TO MERGE** — no Critical/Important.

**What landed:**
- **Observable read-your-writes fixes (RED→GREEN):** Task 1 — `filter.rs` property introspection (`property_exists`/`keys`/`properties`/`property_values`) reads the delta-merged whole-set *behind the `resolve_node`/`resolve_edge` visibility gate*; Task 2 — `mutation.rs` `PropertySource::resolve` source-property read (single-prop visible accessor), with the shared `resolve` signature threaded through `merge.rs`.
- **Mechanical completeness (inert; by-inspection):** Tasks 3–5 — `factorized_filter`, `horizontal_aggregate`, `vector_join` gained `viewing_epoch`/`transaction_id` + `with_transaction_context`; reads now pass `self.transaction_id`.

**Key findings:**
- **Mid-development bug caught by the Task 1 spec review and fixed:** the first introspection reroute dropped the `resolve_node`/`resolve_edge` snapshot-visibility gate at all 4 sites — which would have let a same-tx-**deleted** entity report stale properties (`read_*_properties_visible` does no entity-visibility check). Restored the gate (delta-merged read *behind* the gate); verified sound at the store source — `get_node_versioned` returns `None` for not-visible-or-PENDING-deleted, while same-tx creates stay visible (so read-your-writes holds in both directions).
- **`factorized_filter`, `horizontal_aggregate`, `vector_join` are unreachable from the production planner today** (the first two have no construction site; the third is built at `planner/lpg/mod.rs:849` only via a never-populated `group_list_variables` path). Their threading is correctness-by-construction (inert — fields default `None`), so 2e's `record_read` and any future wiring already carry the tx. Spec §6 lists them as routing targets → mandated completeness, not invented wiring.
- **Pre-existing lint (not a 2d regression):** `clippy::large_stack_arrays` fires under `--all-targets --all-features` in untouched compact-store test code (`graph/compact/column.rs:3334`) — that file is byte-identical to `integration`. Out of scope here.

**Non-blocking follow-ups:**
- Delete-direction coverage: the gate's exclusion of same-tx-deleted entities in introspection is sound-by-construction and exercised via prior increments' expand/EXISTS paths, but has no dedicated probe (a session probe can't isolate it — upstream scan filtering masks the path; a core-level `FilterOperator` unit test could lock it). Optional.
- The pre-existing `column.rs` `--all-targets` clippy lint is a separate cleanup.

**Next: increment 2e** (Plan 2's second half — Part D's inert tracking + Part E): `ReadTracker`/`record_read` wiring at these now-complete read sites + a sharded read-registry + the store-derived write-set. Then Plan 3 (F1 OCC rung → F2 SSI + G performance).
