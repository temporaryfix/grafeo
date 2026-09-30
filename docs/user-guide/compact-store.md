---
title: Compact Store
description: Explicit, repeatable compaction of retained LPG history into a columnar base with a writable overlay.
tags:
  - performance
  - storage
  - compact-store
  - temporal
  - wasm
---

# Compact Store

`compact()` changes the physical layout of Grafeo's managed property graph,
not its role as a writable temporal store. It folds retained committed data
into a columnar base and leaves a mutable overlay for subsequent writes.
Call the same `compact()` operation again to fold later changes into the base.

This page describes the **unreleased 0.0.1 candidate**. Compaction is not a
release-readiness, durability or performance guarantee. LPG and RDF remain
native models in the same engine; this operation organizes the LPG storage
layout, not the RDF dataset into an LPG projection.

## Compact, write, compact again

For the local `grafeo` dependency, use
`default-features = false, features = ["native", "compact-store"]`.
This example needs no query parser. Automatic garbage collection is disabled
only to keep the demonstrated revision available.

```rust
use grafeo::{Config, GrafeoDB, Result, Value};

fn main() -> Result<()> {
    let mut db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let (id, recorded) = {
        let mut session = db.session();
        session.begin_transaction()?;
        let id = session.create_node_with_props(
            &["Person", "Researcher"],
            [("name", Value::from("Alix")), ("age", Value::Int64(30))],
        )?;
        let recorded = session.commit()?;
        (id, recorded)
    }; // Drop the Session before maintenance.

    db.compact()?;
    db.set_node_property(id, "age", Value::Int64(31))?;
    db.compact()?;

    assert_eq!(
        db.get_node_property_at_epoch(id, "age", recorded),
        Some(Value::Int64(30)),
    );
    assert_eq!(
        db.get_node_property_at_epoch(id, "age", db.current_epoch()),
        Some(Value::Int64(31)),
    );
    Ok(())
}
```

Both calls preserve the managed graph's retained node/edge lifetimes, label
history and property values. Node and edge IDs stay stable. Multiple labels
remain independent logical labels: a `Person`/`Researcher` node does not
require querying an internal compound table name.

Use the database or Session read APIs to see the merged base and overlay.
Raw overlay access is not a whole-graph read interface.

## When maintenance may run

Explicit `compact()` requires an open, non-poisoned database, no active
transactions and **no live Session handles**, including idle or historical
Sessions. Commit or roll back transactions, finish queries and streams, then
drop their Sessions before calling it. In Rust it also requires `&mut GrafeoDB`.
A violation returns an error; maintenance does not discard an active
transaction to make progress.

Compaction is an explicit whole-store maintenance operation, not a per-commit
background task. Preparation may need memory for both the old generation and
its successor. Retained low-level views can keep an old generation alive
after a successful transfer, so do not expect immediate reclamation of every
old allocation.

For managed data, repeated calls keep the same Layered store owner while
replacing its base/overlay generation. Named graph topology, property-index
definitions and existing Text/Vector index objects are transferred through
the managed representation boundary; compaction does not create a second
index registry or require callers to rebuild those indexes. Search and query
capabilities still depend on the features and operations supported by the
selected build.

## Optional threshold policy

Rust exposes `compact_if_needed() -> Result<bool>` as an application-invoked
maintenance checkpoint. Configure it with
`Config::compaction_overlay_threshold: Option<usize>`, or
`with_compaction_overlay_threshold(threshold)`. The default `None` disables
the policy; it does not disable explicit `compact()`.

A call considers the managed native store before initial conversion, or the
hot overlay once layered. Its reported node count plus edge count must
**strictly exceed** the threshold. This is not a byte limit, number of commits,
or number of property revisions; repeatedly editing one node need not increase
that count.

The helper returns `Ok(false)` when disabled, at/below threshold, while a
transaction is active, or when there is no eligible managed LPG source.
It does not automatically convert an externally supplied read store.
When eligible, it invokes `compact()` and returns `Ok(true)` after success.
An idle live Session still makes an attempted compaction return an error.

```rust
use grafeo::{Config, GrafeoDB, Result, Value};

fn main() -> Result<()> {
    let config = Config::in_memory().with_compaction_overlay_threshold(1);
    let mut db = GrafeoDB::with_config(config)?;

    let first = db
        .session()
        .create_node_with_props(&["Item"], [("value", Value::Int64(1))])?;
    assert!(!db.compact_if_needed()?); // One node equals the threshold.

    let second = db
        .session()
        .create_node_with_props(&["Item"], [("value", Value::Int64(2))])?;
    assert!(db.compact_if_needed()?); // Two native nodes: first conversion.

    db.set_node_property(first, "value", Value::Int64(3))?;
    assert!(!db.compact_if_needed()?); // One promoted node in the overlay.
    db.set_node_property(second, "value", Value::Int64(4))?;
    assert!(db.compact_if_needed()?); // Fold the two overlay nodes.
    Ok(())
}
```

Schedule this checkpoint at an application maintenance boundary. Setting the
threshold alone does not start a timer or attach a commit hook.

## Layout and value semantics

The base uses per-label tables, forward/backward compressed sparse row (CSR)
adjacency, and column/epoch zone maps where applicable. Temporal rows retain
validity intervals; compaction does not eliminate history to obtain a smaller
current-state snapshot.

Homogeneous histories can use specialized value codecs: bit-packed
non-negative integers, native signed integers and floats, boolean bitmaps,
dictionary strings and fixed-dimension float vectors. Histories that cannot
fit one such codec use exact typed history rows. Mixed types, changing vector
dimensions and complex values are not a request to turn the managed graph's
values into strings. Historical vector values still consume space.

A columnar layout can benefit read-heavy workloads, but the balance depends
on graph shape, retained history, indexes and mutation rate. Measure memory,
query latency, write latency and compaction cost for your workload; this guide
does not claim a fixed memory reduction or traversal speedup.

## Bindings and features

These existing binding operations call the same engine maintenance boundary:

| Binding | Call | Failure channel |
|---|---|---|
| Python | `db.compact()` | Python exception |
| Node.js | `db.compact()` | Synchronous JavaScript exception |
| WASM | `db.compact()` | JavaScript exception |
| C | `grafeo_compact(db)` | Non-`GRAFEO_OK` status; inspect `grafeo_last_error()` |
| C# | `db.Compact()` | `GrafeoException` through the C binding |

The calls are repeatable and leave the database writable. Finish outstanding
asynchronous Node.js operations before maintenance. Do not invent a separate
binding method for later folds or assume the Rust threshold policy is exposed
in each binding.

The engine method requires both `lpg` and `compact-store`. Neither the
default engine nor the default Rust facade build includes `compact-store`;
enable it explicitly. Current Python, Node.js and C default feature closures
include it. The WASM default resolves to its `edge` profile, which includes
LPG, GQL and compact storage. Custom feature builds can differ. Add query
languages, Text/Vector search, algorithms or other capabilities explicitly
when the chosen profile does not include them.

With the separate `mmap` feature, the engine can register the compact base
with its memory manager for disk-backed spill. Spilling a base is not an
implicit overlay compaction, and an mmap spill file is not a database backup.

## History, persistence and limits

Compaction preserves **retained** committed history. It cannot recover versions
already collected by GC and does not grant an epoch-retention lease. See
[temporal graphs and retained history](temporal.md) for historical reads,
retention limits and the distinction between publication epochs and RDF valid
time.

Compaction is not a WAL sync, checkpoint or backup. After successful initial
conversion, the flat-only background checkpoint timer is stopped because it
cannot represent the layered graph. For persistent databases, use the
topology-aware `wal_checkpoint()` or `close()` at their required quiescent
boundaries, handle errors, and retain the configured durability policy.
The browser build does not acquire crash durability merely by compacting.

Native container and portable snapshot routes have different exactness limits.
In particular, portable export and `to_memory()` currently reject ordinary
nonempty commit-born Text indexes that they cannot represent exactly; compacting
does not remove that restriction. Recursive format and release qualification
remain unfinished. See the [persistence guide](persistence/index.md) and
[temporal persistence boundaries](temporal.md#snapshots-persistence-and-evidence).

The retained-history guarantees above apply to engine-managed native/layered
data. Explicit conversion from an externally supplied read store copies its
current snapshot; it cannot reconstruct a transaction history that the external
interface does not supply. Low-level compact builders and external snapshot
conversion are not substitutes for the managed temporal path.

An available compaction API does not by itself establish release readiness.
