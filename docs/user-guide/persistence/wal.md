---
title: WAL Recovery
description: Understanding write-ahead logging and crash recovery.
tags:
  - persistence
  - wal
  - recovery
---

# WAL Recovery

Grafeo uses a checksummed write-ahead log (WAL) to provide crash recovery for
persistent databases. The exact acknowledgement guarantee depends on the
configured durability mode.

## How WAL Works

1. **Prepare** - Mutations remain private until their commit boundary
2. **Log** - Data records and their commit marker are appended to the WAL
3. **Publish** - The committed state becomes visible at one database epoch
4. **Checkpoint** - A consistent publication cut is written to the container
5. **Retire** - WAL already covered by that durable checkpoint can be removed

Standalone catalog DDL uses one epoch-framed complete catalog post-image, plus
its named-graph create/drop set. A statement that changes several catalog
objects therefore cannot recover as a partially applied sequence of deltas.

## Crash Recovery

Only the current WAL generation is accepted; there is no legacy reader or
migration path. Every record carries `GRAFOWAL` followed by generation 4 as a
little-endian `u16`, its group role and transaction coordinate, and the serialized
record, inside the existing checksum or authenticated-encryption boundary.
Commit and abort markers carry an ordered record count and BLAKE3 group seal that
also binds the marker's identity and epoch. The complete envelope and seal count
toward the 64 MiB frame limit. Unsupported or
missing generations reject without rewriting the files, including retained
segments before the checkpoint.

When opening a database after a crash:

1. Grafeo validates the generation-4 checkpoint and retained-group boundary,
   rejects reserved `PENDING` coordinates, and checks WAL framing, checksums,
   group seals, and storage-decoded records' local epoch invariants
2. Records are grouped and released according to decoded commit and catalog
   markers
3. Incomplete transaction tails are ignored
4. Selected records are applied after the checkpoint epoch
5. The shared LPG/RDF/catalog epoch clock is advanced to the recovered cut

Record-local validation runs before a decoded frame can change transaction
grouping, transaction-ID high-water tracking, or the last-good recovery
boundary. Group seals validate retained complete groups before checkpoint replay
filtering or application callbacks. Files below the checkpoint's declared group
retirement floor may be residual fragments; their framing and record schema still
validate, but they no longer provide complete-group authority.

Within storage-decoded WAL commit/publication coordinates and the
checkpoint epoch, `EpochId::PENDING` is not a legal commit, live
catalog/projection-declaration, projection source/target, or checkpoint
coordinate. Data structures may still use `PENDING` where their schema
explicitly defines an open or uncommitted interval. Opaque engine-owned record
payloads are validated by their engine decoder rather than this storage pass.

Epoch zero remains readable as the initial cut, a synthetic exact-history
`Committed(0)` boundary, a frozen `EpochAdvance(0)`, and a compatible
projection source coordinate. Newly allocated live publications and projection
targets are positive. These checks deliberately do not claim global WAL epoch
monotonicity, unique epoch ownership, or repeated-transaction-marker
consistency; whole-stream enforcement remains qualification work. A
checksum-valid built-in record that fails local semantic validation stops
recovery with `GRAFEO-S002` and remains byte-identical in place for diagnosis
or manual repair. Recovery neither silently truncates nor quarantines that
frame.

Standalone WAL writers validate the physical stream before reopening it for
append. They reject torn tails, quarantine markers, ambiguous segment identities
and invalid checkpoint boundaries without repairing them. Use explicit
`WalRecovery` and its consuming `into_wal` handoff to recover a torn tail;
database open performs its managed recovery. Custom typed WALs retain their own
record codec; direct writer admission checks the shared envelope and group seals,
without imposing the LPG record codec.
Async backend batches accept payloads from `grafeo_storage::wal::encode_record`,
not bare bincode bytes. A failed batch can retain earlier accepted records.

One WAL owner admits at most 65,536 unfinished transaction groups. Exceeding that
limit rejects the next new group before append; finishing or aborting a group
frees capacity. Raw WAL checkpoints refuse unfinished live groups. After recovery,
a checkpoint can retire orphaned groups that the recovered state has abandoned.

```python
# Recovery happens automatically on open
db = grafeo.GrafeoDB(path="my_graph.db")
# The currently supported recovery pass completed successfully
```

## Checkpointing

Checkpoints merge WAL changes into the main data files:

```python
# Manual checkpoint
db.checkpoint()

# Automatic checkpointing can be configured for flat LPG persistent stores
```

Epoch zero is a valid checkpoint cut. The reserved `PENDING` value is rejected
before a checkpoint marker, counter, or metadata file can change. If publishing
the required fresh WAL segment would exhaust its sequence identity space, the
checkpoint returns `GRAFEO-S001` before appending its marker. A persisted
checkpoint carrying `PENDING` is instead corruption (`GRAFEO-S002`) and cannot
select a recovery boundary. Checkpoint decoding is exact and bounded; its claimed
segment must exist and the replay suffix must be contiguous.

`close()` performs the final topology-aware checkpoint. In compact/layered
mode, use an explicit checkpoint (or `close()`); the flat-store periodic timer
is intentionally disabled because it cannot independently serialize the cold
base, overlay, and deletion state as one generation.
The restriction is temporary; topology-neutral scheduling remains a product
target, while explicit checkpoints retain the full durability guarantee.

## Coherent Exact Copies

`save(path)` captures one quiescent publication cut across LPG, RDF, catalog,
recursive named graphs, indexes, and RDF→LPG projections. It always writes the
exact versioned section container, including for extensionless destinations.
It preserves the `WorldCut`, `StoreId`, structural/property/label histories,
RDF transaction/valid-time intervals, index owners and allocator floors, exact
Text/Vector images, and verified projection receipts. No synthetic catalog,
index, or projection transactions consume epochs in the saved copy.

The saved path is a file, not a WAL-directory reconstruction. On reopen, the
existing container is detected regardless of suffix and later mutations use
its `<path>.wal` sidecar. Operational WAL-directory databases remain supported
for normal creation, mutation, rotation, and recovery; saving one produces an
exact container without converting the source directory.

Incremental-backup range admission remains pre-mutation: a transaction manager
exposing `PENDING` and a persisted
`PENDING` cursor are corruption, while a cursor at the final real epoch returns
`GRAFEO-S001` because no committed successor exists. A caller-supplied
`PENDING` point-in-time restore target is invalid input.

Current RDF quad WAL records carry the exact graph incarnation and store
validated, half-open valid-time intervals as signed `i128` TAI nanoseconds.
Predecessor triple, name-only graph and graph-clear record forms
are rejected during recovery, including both earlier valid-time forms. Their
reserved tags cannot be written. Empty, inverted, or half-present current
intervals also fail closed during recovery.

The container is built and synced in a private sibling staging directory.
Publication moves only the completed container file using the operating
system's atomic no-replace rename, then syncs the destination parent directory.
Container and sidecar WAL ownership remain held through publication, parent
sync and cleanup. An existing sidecar is refused, not adopted or erased. If any
file, directory, or symlink already owns the requested name,
the save fails and retains that object unchanged. A platform or filesystem
without a safe no-replace primitive fails closed instead of falling back to a
check-then-replacing rename.

## Durability Levels

| Mode | A successful commit means |
|------|---------------------------|
| `Sync` | The commit has been forced to stable storage before success is returned |
| `Batch` | The commit is in the WAL and will be forced at the configured record/time threshold |
| `Adaptive` | The commit is in the WAL and a background flusher chooses the sync cadence |
| `NoSync` | The commit is in OS buffers; recent work can be lost on power/OS failure |

All modes preserve transaction framing during ordinary process-crash recovery.
Only `Sync` promises that a successful commit has already crossed the storage
flush boundary.

## Best Practices

1. **Keep the default `Sync` mode for strict acknowledgement durability** -
   select `Batch`, `Adaptive`, or `NoSync` only as an explicit throughput trade
2. **Checkpoint regularly** - This bounds recovery work and WAL growth
3. **Reserve disk headroom** - The WAL grows between durable checkpoints
4. **Call `close()` when possible** - It performs a final coordinated
   checkpoint; crash recovery remains the fallback, not the shutdown protocol
