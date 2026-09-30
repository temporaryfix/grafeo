---
title: Persistent Storage
description: Using Grafeo with durable storage, portable snapshots, and integrity-sealed world cuts.
tags:
  - persistence
  - storage
---

# Persistent Storage

Persistent mode stores committed data durably on disk. Grafeo supports a
WAL-directory database and a single-file container. A new `.grafeo` path selects
the container automatically; an existing saved container is recognized even
without that suffix.

## Creating a Persistent Database

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB(path="my_graph.db")
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let db = GrafeoDB::open("my_graph.db")?;
    ```

A directory path owns an implementation-managed WAL directory and recovery
metadata. Do not copy individual files from a live database. Use `save`, a
coordinated backup, or a portable snapshot so every model is captured at one
committed publication cut.

## Durability Guarantees

- **Write-ahead logging (WAL):** committed mutations are framed before they
  become public.
- **Checkpointing:** one coherent LPG/RDF/catalog cut is consolidated into a
  durable image.
- **Crash recovery:** complete committed transactions after the checkpoint are
  replayed; incomplete tails are ignored.

Persistent databases default to strict `Sync` acknowledgement durability. The
guarantee changes only when another durability mode is selected explicitly. See
[WAL Recovery](wal.md) for the exact `Sync`, `Batch`, `Adaptive`, and `NoSync`
contracts.

## Configuration

```python
# The Python constructor accepts path and cdc.
# Durability mode configuration is available through the Rust Config builder.
db = grafeo.GrafeoDB(path="my_graph.db")
```

## Single-File Format (`.grafeo`)

A container stores one checksummed section image and uses a sidecar WAL
directory named `<path>.wal` for crash recovery. The `.grafeo` suffix is
conventional, not required when reopening an existing container.

=== "Python"

    ```python
    db = grafeo.GrafeoDB(path="my_graph.grafeo")
    ```

=== "Rust"

    ```rust
    let db = GrafeoDB::open("my_graph.grafeo")?;
    ```

Every current section image is complete and model-appropriate. An LPG or Both
database includes its LPG section; an RDF or Both database includes its RDF
section even when that model is empty. Each authoritative section carries its
exact serializer version. `WORLD_METADATA` version 1 then binds the complete
image to a self-verifying `WorldCut`, including:

- the logical `StoreId`, committed epoch, and graph model;
- exact catalog, LPG, RDF, history, compact-base, and deletion format versions
  that apply to that image;
- the canonical schema digest and complete verified projection receipts;
- RDF history completeness; and
- full BLAKE3 state and manifest digests over the authoritative plaintext
  sections.

Derived indexes are checksummed but deliberately excluded from the state
digest because they can be rebuilt. Grafeo verifies all section checksums,
versions, the world manifest, and decoded identity/schema/projection
coordinates before installing any state. A corrupt or contradictory current
container fails closed.

See the [container format specification](../../architecture/storage/container-format.md)
for the wire-level layout and checkpoint protocol.

## Read-Only Mode

Open a database in read-only mode to allow multiple processes to read the same
`.grafeo` file concurrently. Mutations are rejected at the session boundary.

=== "Python"

    ```python
    db = grafeo.GrafeoDB.open_read_only("my_graph.grafeo")
    ```

=== "Rust"

    ```rust
    let db = GrafeoDB::open_read_only("my_graph.grafeo")?;
    ```

Read-only mode uses a shared file lock; a writer requires the exclusive lock.

## Saving a Coherent Copy

`save` captures the source once and leaves it unchanged. Every destination,
including an extensionless path, receives the same complete versioned section
container with `WORLD_METADATA`. This is an exact logical replica at the captured
epoch: `StoreId`, world cut, recursive graph history, index owners and allocator
floors, exact Text/Vector images, and projection receipts are preserved without
synthetic destination transactions or shifted epochs. Reopening the saved file
recognizes the container; subsequent mutations use its `<path>.wal` sidecar.

Operational WAL-directory databases remain supported. Creating or opening one
does not change its layout, but `save` always produces an exact container copy.

=== "Python"

    ```python
    db.save("backup.grafeo")
    ```

=== "Rust"

    ```rust
    db.save("backup.grafeo")?;
    ```

The destination must not already exist. Keep only one writable lineage for an
exact replica: take the old primary offline (or keep it read-only) before
writing to the restored copy. Two independently writable databases with the
same `StoreId` would create divergent histories in one statement-handle and
CDC-cursor namespace.

## Portable Snapshots and `WorldCut`

The Rust API exposes portable snapshot v12 in two forms:

- `export_snapshot` returns the exact snapshot bytes;
- `export_snapshot_artifact` returns a `SnapshotArtifact` that binds those
  bytes to a self-verifying `WorldCut`.

Use the integrity-sealed form whenever snapshots cross a storage boundary:

```rust
use grafeo::{GrafeoDB, SnapshotArtifact, WorldCut};

let artifact = db.export_snapshot_artifact()?;
artifact.verify()?;

// Persist these two values together. WorldCut has a bounded, exact decoder.
let snapshot_bytes = artifact.bytes().to_vec();
let cut_bytes = artifact.cut().encode()?;

let cut = WorldCut::decode(&cut_bytes)?;
let received = SnapshotArtifact::from_parts(snapshot_bytes, cut)?;
let replica = GrafeoDB::import_snapshot_artifact(&received)?;
```

`SnapshotArtifact::verify` checks the BLAKE3 binding of every snapshot byte and
the canonical manifest. Import additionally recomputes the descriptor from the decoded
payload, so a valid digest paired with false store, epoch, model, schema,
history, format, or projection coordinates is rejected.

The digest is unkeyed: it detects accidental damage and inconsistent or
partially rewritten artifacts, but it does not authenticate who produced an
artifact when an adversary can replace both bytes and manifest. Authenticate
the transport or attach an application signature/MAC when origin or
adversarial tamper resistance matters.

`artifact.cut()` uses the `SnapshotBytes` digest grammar and includes the
portable-snapshot format in its descriptor. `db.world_cut()?` is the companion
container-oriented API: it serializes the current model-appropriate
authoritative sections in memory and seals them with the
`AuthoritativeComponents` grammar used by `.grafeo` `WORLD_METADATA`. It does
not write a file or mark source sections clean, but it is O(database), not a
cheap live counter.

The two cuts can describe the same StoreId and committed epoch while carrying
different format sets, state digests, and manifest digests. Compare their
logical coordinates when correlating them; do not expect their representation
digests to be equal. Use `export_snapshot_artifact` when the portable bytes are
also required.

The raw-byte path is:

```rust
let bytes = db.export_snapshot()?;
let replica = GrafeoDB::import_snapshot(&bytes)?;
```

Both import paths create an exact replica and preserve the snapshot `StoreId`,
RDF graph incarnations, statement handles, allocator high-water marks,
transaction/valid-time history, schema, indexes, and projection provenance.
`restore_snapshot` is a separate, non-atomic live replacement operation with
additional restrictions described below; prefer importing a new database.

## Replica or Fork?

Choose by lineage, not by storage location:

| Operation | `StoreId` | RDF handles/history | Projection materialization | Intended use |
|-----------|-----------|---------------------|----------------------------|--------------|
| `import_snapshot` / `import_snapshot_artifact` | Preserved | Preserved exactly | Verified provenance preserved | Move or restore the same logical store |
| `save` / reopen (any suffix) | Preserved | Preserved exactly | Verified provenance preserved | Exact durable copy of the same logical store |
| `import_snapshot_as_fork` | New | History is re-keyed to the new identity | Owned LPG rows removed; mappings reset to pending | Independent branch from snapshot bytes |
| `to_memory` | New | Copied RDF state is re-keyed to the new identity | Owned LPG rows removed; mappings reset to pending | Independent in-memory work |
| `open_multi` | New | At most one RDF-bearing input; history re-keyed | Owned LPG rows removed; mappings reset to pending | Union producer-coordinated snapshot parts |

For an independently writable database, fork explicitly:

```rust
let bytes = db.export_snapshot()?;
let branch = GrafeoDB::import_snapshot_as_fork(&bytes)?;

// Or fork the current database directly into memory.
let scratch = db.to_memory()?;
```

Forking changes every StoreId-derived RDF statement handle. It also prevents a
verified RDF→LPG receipt from being misrepresented as belonging to the new
store: projection-owned LPG rows are removed and the logical mappings are
retained in pending state for an explicit rebuild.

The Python binding exposes the direct in-memory fork:

```python
scratch = db.to_memory()
```

`open_multi` preserves producer-assigned IDs and rejects conflicting graph
ownership, duplicate entity IDs in one graph namespace, and unresolved endpoints.
Independent index owners need disjoint owner IDs. Shared Text/Vector owners must
have identical exact images, even when their surrounding index sets differ:
`{A, B}` and `{A, C}` can produce `{A, B, C}` when A is byte-identical.
Equal configuration alone is insufficient: posting history, retained epochs,
topology, quantizer state and RNG continuation are part of the index state.
Conflicts are rejected without rebuilding indexes or changing either input.

## Current Snapshot Format

Portable Snapshot12 includes structural LPG lifetimes, property and label
timelines, recursively graph-qualified rows and allocator floors, typed-Quad
transaction history, signed TAI-nanosecond RDF valid time, named-graph
lifecycle/incarnation history, stable statement handles, logical store identity,
and projection state. Graph paths are component sequences: a literal name
`a/b` is distinct from graph `b` nested inside graph `a`.

The snapshot embeds the current Catalog7 state, including exact index owner
IDs and the allocator floor after dropped indexes. Text5 and Vector4 carry
exact BM25 and HNSW/quantization images and resolved configuration; import does
not invent owner IDs or reconstruct those images from current rows.
Text5 also preserves the earliest retained Text epoch. Older Text4 sections are
rejected; there is no migration reader.
`open_multi` rejects conflicting owner IDs and incompatible shared images
rather than renumbering owners or combining HNSW payloads.

Only Snapshot12 is accepted. All earlier snapshot versions are rejected; there is no
legacy reader, upgrade path, or migration support. Container import additionally
validates per-section versions and sealed world metadata.

Exact snapshot import constructs an unpublished database. Live
`restore_snapshot` atomically replaces a quiescent in-memory database, including
its exact indexes, allocator history, catalog, RDF and world identity.
WAL-backed or compacted targets and active transactions remain refused.
Subgraph extraction has separate restrictions for index owners or allocator
history. Exact container save preserves those coordinates and recursive LPG
paths; it does not reconstruct them from a stream of mutations.

## Durable RDF CDC

RDF change capture is derived from the persisted typed-Quad interval history;
it is not the optional process-local LPG CDC log. `rdf_cdc_page` is
graph-qualified, includes named-graph lifecycle transitions, survives WAL
recovery, checkpoint, exact snapshot restore, and reopen, and uses
StoreId-bound resumable cursors. See [Change Data Capture](../cdc.md#durable-rdf-dataset-cdc).
