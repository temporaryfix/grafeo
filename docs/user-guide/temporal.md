---
title: Temporal Graphs & Retained History
description: Committed epochs, RDF valid time, historical reads and their retention and persistence limits.
tags:
  - temporal
  - time-travel
  - versioning
---

# Temporal Graphs & Retained History

Grafeo is the embedded store for native property graphs
and RDF datasets. Applications can ask what the retained graph said at a
committed publication, not just what it says now. This page describes the
**unreleased 0.0.1 candidate**, not a claim that every release gate has passed.

Start with a database, use a Session transaction to group changes, and retain
the epoch returned by `commit()` alongside the database's `StoreId`.
Historical reads, durable storage and application valid time are distinct
contracts. There is no `temporal` Cargo feature to enable.

## Three different meanings of time

| Coordinate | Meaning | What it is not |
|---|---|---|
| `EpochId` | Ordered publication coordinate within one store; used for LPG as-of reads and RDF transaction-time cuts | A timestamp, a transaction count, or a coordinate portable between unrelated stores |
| `TaiNanoseconds` | Signed, lossless `i128` application valid-time coordinate for RDF statements | A commit epoch or an automatically converted UTC timestamp |
| Date/time property values | Values stored and queried as application data | A request to make a query historical |

Committed transactions and standalone metadata publications share the epoch
space. Failed durable publications can leave gaps. Capture your transaction's
returned epoch; reading `current_epoch()` afterward can observe another
publisher. `EpochId::PENDING` is a reserved current-view sentinel on APIs that
explicitly support it, not an epoch to record as a committed cut.

## First historical read

Use the local checkout (or an explicitly selected source revision), with
`default-features = false, features = ["native"]` on the `grafeo` dependency.
`native` supplies parser-free LPG/RDF and persistence support; this example
uses the default LPG database model.

The example disables automatic garbage collection so these two revisions
remain available for the demonstration. This is **not** a durable retention
policy or protection against explicit garbage collection.

```rust
use grafeo::{Config, GrafeoDB, Result, Value};

fn main() -> Result<()> {
    let db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let mut session = db.session();

    session.begin_transaction()?;
    let id = session.create_node_with_props(&["Asset"], [("status", Value::from("ready"))])?;
    let recorded = session.commit()?;

    session.begin_transaction()?;
    session.set_node_property(id, "status", Value::from("running"))?;
    session.commit()?;

    assert_eq!(
        db.get_node_property_at_epoch(id, "status", recorded),
        Some(Value::from("ready")),
    );
    Ok(())
}
```

This is in-memory history, not crash durability. For durable deployments, use
the supported persistent configuration and read the
[persistence guide](persistence/index.md).

### Mutation results are part of the contract

`GrafeoDB::set_node_property` and `set_edge_property` return `Result<()>`:
success means their automatic transaction committed under the configured
durability policy. The same Session setters stage writes in an active explicit
transaction; its `commit()` still has to succeed. Missing targets are errors,
not implicit entity creation. READ ONLY transactions and historical views
reject setters. A target deleted since BEGIN produces a write conflict.

Propagate these results. After a durability failure, stop writing and reopen
to recover; an error is not proof that nothing reached storage. Other direct
mutation families still have inherited sentinel/boolean return contracts.
Any upstream API changes require a separate compatibility decision.

## Reading a committed LPG cut

The typed host methods `get_node_at_epoch`, `get_edge_at_epoch`,
`get_node_property_at_epoch` and the epoch-qualified neighbor methods read
retained structural, label and property state. The managed database routes
supported reads across the compacted base and its mutable overlay.
`compact()` preserves retained history; it does not recover versions already
removed by garbage collection.

With the `gql` feature, `session.execute_at_epoch(query, epoch)` executes a
historical **GQL** read. It rejects an active transaction rather than replacing
that transaction's BEGIN snapshot. Other frontends' ordinary execute methods
do not become historical merely because the engine has a time axis.

For repeated LPG reads, `set_viewing_epoch(epoch)` selects a Session override;
`clear_viewing_epoch()` removes it. LPG mutations are rejected while the
override applies. An existing transaction keeps its BEGIN snapshot, and the
override takes effect after it ends. The override does **not** pin retention.

`scrub_at_epoch` returns columnar node/edge frames for a managed compacted
database. Before compaction it returns an empty scrub, not a general-purpose
snapshot. Its shape suits timeline rendering; no particular frame rate is
qualified here.

### History is subject to retention

Automatic LPG garbage collection is enabled by default, every 100 commits.
It uses the transaction manager's minimum active epoch to prune obsolete
versions. Keeping an epoch number, a viewing override or a `WorldCut` is not
a lease that keeps those versions alive.

Do not treat a missing historical value as an audit proof that the entity
never existed: the convenience `Option` accessors do not provide a general
expired-cut error contract. Explicit retained-history policy and historical
reader retention authority remain release/API work. Budget retained history;
disabling automatic collection trades bounded reclamation for growth.

Text indexes have a narrower checked contract: `retained_from()` reports their
earliest available epoch, and epoch-qualified search, scoring and corpus
statistics return errors below that floor, even for empty queries or zero
limits. Explicit Rust collection is fallible (`db.gc()?`); it respects active
transaction epochs but is not an atomic database-wide collection. Text5 images
preserve this monotonically advancing floor through containers and Snapshot12.
WAL commits also preserve exact Text births and sparse committed changes across
recovery, without rebuilding retained history from current strings. WAL-backed
databases with Text or Vector indexes (including named-graph indexes) refuse
`gc()` before any pruning: an independently durable GC transition is not yet
implemented. Normal Vector commits preserve exact topology, RNG continuation,
deleted membership and quantizer state without replay search or retraining.
In-memory collection remains available. This is not a general historical-reader
lease.

### Structural lifetimes are not a property-change feed

`get_node_history` and `get_edge_history` return retained structural
lifetimes, newest first, as `(created_epoch, deleted_epoch, entity)`.
Each entity is materialized at its lifetime's creation epoch.

Updating a property three times does **not** produce three structural lifetime
entries. Use property/as-of reads at recorded commit epochs to compare values.
Do not present these lifetime APIs as a complete revision-by-revision audit
log.

## RDF: transaction time plus application valid time

Use `GraphModel::Rdf` or `GraphModel::Both` when configuring an RDF database;
the default constructor selects LPG. The `triple-store` capability supplies
the typed RDF kernel; `sparql` adds its query frontend. See the
[RDF host example](native-host.md#rdf).

The graph-qualified APIs are the front door for historical RDF:

- `rdf_history_cut(epoch)` selects a retained dataset at a publication epoch.
- `rdf_history_cut_at(epoch, Some(valid_at))` also filters by application
  valid time, across default and named graphs. `None` applies no valid-time
  filter; statements without a valid-time interval are always valid.
- `rdf_history_diff(from, through)` returns ordered lifecycle and statement
  transitions in `(from, through]`.
- `rdf_cdc_page` pages persisted RDF history with store-bound cursors. It is
  distinct from the optional, currently process-local LPG CDC surface.

These cuts retain named-graph incarnations and stable statement handles.
The default-graph `rdf_triples_at_valid_time` convenience method is not a
replacement for a historical, graph-qualified dataset cut.

Set `session.set_rdf_valid_time_tai_ns(from, to)` before statement creation.
Intervals are non-empty and half-open, `[from, to)`, in signed TAI nanoseconds.
Empty or inverted intervals fail before mutation. UTC/calendar/leap-second
conversion belongs at ingestion and presentation boundaries.

Direct inserts and statement-creating SPARQL operations capture the current
Session interval per queued mutation. Commit, rollback and savepoint rollback
do not rewind the Session setting. `COPY`, `MOVE` and `ADD` preserve each
source statement's interval. The LPG viewing-epoch override is not an RDF
time-selection API, and arbitrary historical SPARQL is not promised here.

The candidate's RDF container grammar is v6 only; predecessor RDF container
sections are rejected. Inherited portable microsecond adapters still await
removal and are not a supported 0.0.1 migration contract.

## Snapshots, persistence and evidence

A historical view selects data. A persistence format must also represent all
required data and index history exactly. These are not interchangeable APIs:

| Route | Current boundary |
|---|---|
| Native `.grafeo` save/open | Preserves recursive LPG histories and current catalog, exact Text and Vector images, including the Text retention floor |
| Portable `export_snapshot` / `SnapshotArtifact` | Snapshot12 carries recursive LPG/RDF histories and exact catalog/Text/Vector images; unsupported state fails closed |
| `to_memory()` | Creates an independent writable lineage through the portable path; inherits its unsupported-state restrictions |
| Historical Session / typed as-of read | Reads retained history; does not itself create a backup, authenticate provenance or reserve retention |

Portable export preserves Text posting/document/aggregate history without
rebuilding it from current rows. Indexed live restore remains guarded; exact
import into a new database does not make live replacement atomic.
A successfully persisted snapshot cannot restore versions collected before
capture. Backup-chain v2 and durable LPG CDC are also unfinished; do not infer
their readiness from the existence of old method names.

A `MixedSnapshot` holds a thread-affine publication read guard for a coherent
LPG/RDF view. It is not an owned, transferable historical snapshot to retain
across arbitrary async work.

A `WorldCut` binds a store identity and epoch to exact representation metadata
and bytes. Unkeyed BLAKE3 verification establishes internal consistency, **not
the producer's identity**. Authenticate transport or application signatures
separately. Container and portable representations can have different digests
at the same store/epoch. Capturing a cut can serialize O(database) state;
it is not a cheap retention token.

Exact replicas retain `StoreId`; do not independently write to two such copies.
An intentional writable fork uses the explicit fork path and a new identity.
See [identity, snapshots and world cuts](native-host.md#identity-snapshots-and-world-cuts).

Treat a poisoned durability state as a failure requiring recovery, not a
successful commit with a warning. Check `is_durability_poisoned()` and handle
structured transaction/storage errors. These APIs and focused crash tests do
not by themselves qualify the entire candidate for sole-copy deployment.

## Temporal property values

Date, time, datetime, zoned time/datetime and duration are ordinary values;
storing a datetime property does not select a commit epoch or assign RDF valid
time. For GQL constructors, literals, arithmetic and component extraction,
see [Temporal Functions](gql/functions-temporal.md). Binding and frontend
representations are separate API contracts; do not assume lossless epoch
transport through JavaScript `Number`.

## Qualification and next steps

The executable boundaries live in `temporal_host`, `temporal_properties`,
`time_travel`, `temporal_persistence_fidelity`, `g3_rdf_time`,
`world_cut_snapshot`, `world_cut_container` and the current Text/container
commit tests. Listing a target here does not claim it was rerun for every
platform or binding. See the review guide for validation scope.

Start with the [native host guide](native-host.md) and
[transactions](transactions.md). Release qualification, OPSEC and explicit
publication permission remain separate gates.
