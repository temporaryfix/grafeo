# `.grafeo` Container Format Specification

> Development-format notes: the numbered versions below include historical
> design descriptions. The current decoders and golden fixtures in this review
> source define its accepted bytes. In particular, compact sections accept only
> version 9 and overlay-deletion sections only version 2. This snapshot is not
> an upstream-compatible migration or release; adoption requires an explicit
> format and compatibility review.

The `.grafeo` file is the single-file persistence format for Grafeo databases.
It stores data in typed **sections**, each independently addressable, checksummed,
and (for index sections) memory-mappable.

## File Layout

```
Offset    Size     Contents
────────────────────────────────────────────────────
0x0000    4 KiB    FileHeader (magic, version, page size)
0x1000    4 KiB    DbHeader H1 (iteration, checksum, metadata)
0x2000    4 KiB    DbHeader H2 (alternating crash-safe copy)
0x3000    4 KiB    Section Directory (type/offset/length/CRC entries)
0x4000+   varies   Section data (page-aligned per section)
```

Total header overhead: 16 KiB. All regions are page-aligned (4 KiB boundaries).

---

## FileHeader (0x0000, 4 KiB)

Written once at database creation. Never modified afterwards.

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `[u8; 4]` | `magic` | `0x47524146` ("GRAF") |
| 4 | 4 | `u32 LE` | `format_version` | `1` (current) |
| 8 | 4 | `u32 LE` | `page_size` | Always `4096` |
| 12 | 8 | `u64 LE` | `creation_timestamp_ms` | Unix epoch milliseconds |
| 20 | 32 | `[u8; 32]` | `creator_version` | UTF-8 Grafeo version, zero-padded |
| 52 | 1 | `u8` | `graph_model` | Immutable model tag: `0` LPG, `1` RDF, `2` Both; absent in old files and decoded as `0` |
| 53 | 4043 | - | (reserved) | Zero-filled |

The header is serialized with bincode and zero-padded to 4 KiB.

**Validation on open:**

- `magic` must equal `b"GRAF"` (reject otherwise)
- `format_version` must be `<= FORMAT_VERSION` (reject unknown future versions)

---

## DbHeader H1/H2 (0x1000 and 0x2000, 4 KiB each)

Two alternating header slots provide crash safety. On each checkpoint, the
**inactive** slot is overwritten with the new state, then fsynced. If the process
crashes mid-write, the other slot still contains valid metadata.

| Field | Size | Type | Description |
|-------|------|------|-------------|
| `iteration` | 8 | `u64 LE` | Monotonic counter, higher = current |
| `checksum` | 4 | `u32 LE` | CRC-32 of section directory (v2) or snapshot (v1) |
| `snapshot_length` | 8 | `u64 LE` | `0` for v2 section format, `>0` for v1 blob format |
| `epoch` | 8 | `u64 LE` | MVCC epoch at checkpoint |
| `transaction_id` | 8 | `u64 LE` | Transaction-ID high-water mark; a WAL replay floor only when coordinate-sealed metadata authenticates it |
| `node_count` | 8 | `u64 LE` | LPG node count |
| `edge_count` | 8 | `u64 LE` | LPG edge count |
| `timestamp_ms` | 8 | `u64 LE` | Checkpoint timestamp (Unix epoch ms) |
| (reserved) | ~3940 | - | Zero-filled to 4 KiB |

**Active header selection:** On open, read both H1 and H2. The header with
the higher `iteration` value is the active state. If both are empty
(`iteration == 0`), the database has never been checkpointed.

**v1/v2 detection:** If the active header has `snapshot_length > 0`, the file
uses the v1 blob format (a single bincode snapshot starting at `DATA_OFFSET`).
If `snapshot_length == 0` and `iteration > 0`, the file uses the v2 section
format with a section directory at `0x3000`.

---

## Section Directory (0x3000, 4 KiB)

A fixed-size page containing an array of section entries. Each entry is 32 bytes.
Maximum capacity: 127 sections (`(4096 - 8) / 32`).

### Directory Header

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 4 | `u32 LE` | `entry_count` |
| 4 | 4 | `u32 LE` | `reserved` (zero) |

### Directory Entry (32 bytes each, starting at offset 8)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `u32 LE` | `section_type` | Section type ID (see table below) |
| 4 | 1 | `u8` | `version` | Per-section format version |
| 5 | 1 | `u8` | `flags` | Bit 0: required, Bit 1: mmap-able |
| 6 | 2 | `u16 LE` | `reserved` | Zero |
| 8 | 8 | `u64 LE` | `offset` | Byte offset from file start |
| 16 | 8 | `u64 LE` | `length` | Byte length of section data |
| 24 | 4 | `u32 LE` | `checksum` | CRC-32 of section data |
| 28 | 4 | `u32 LE` | `reserved` | Zero |

Remaining bytes after the last entry are zero-filled to 4 KiB.

---

## Section Types

| Value | Name | Required | Mmap-able | Description |
|-------|------|----------|-----------|-------------|
| 1 | `CATALOG` | yes | no | Schema defs, named constraints, index metadata, epoch, config |
| 2 | `LPG_STORE` | yes | no | Nodes, edges, properties, named graphs |
| 3 | `RDF_STORE` | no | no | RDF triples, named graphs |
| 4 | `COMPACT_STORE` | yes | yes | Columnar layered-store base, including temporal history |
| 5 | `OVERLAY_DELETIONS` | no | no | Authoritative layered-base tombstones and their delete epochs |
| 6 | `WORLD_METADATA` | yes | no | Store identity and integrity-sealed committed world-cut manifest |
| 10 | `VECTOR_STORE` | no | yes | Embeddings + HNSW topology |
| 11 | `TEXT_INDEX` | no | yes | BM25 postings + term dictionary |
| 12 | `RDF_RING` | no | yes | Wavelet trees + dictionary |
| 20 | `PROPERTY_INDEX` | no | yes | Property hash/btree indexes |

**Type ranges:**

- 1-9: Data sections (authoritative, cannot be rebuilt)
- 10-19: Index/acceleration sections. Recovery semantics are version-specific;
  the current vector/text images are installed exactly.
- 20+: Reserved for acceleration structures

**Flags:**

- **Bit 0 (required):** Records whether the section is logically required or
  rebuildable. The current v2 directory reader deliberately fails closed on
  every unknown section type and on every known-but-unsupported section
  version, regardless of this bit. A future directory format may use the bit
  for forward-compatible skipping, but current readers do not claim that
  guarantee.
- **Bit 1 (mmap-able):** If set, the section uses a fixed binary layout
  suitable for zero-copy memory-mapped access. If clear, the section must
  be deserialized into RAM (bincode format).

Empty acceleration sections are omitted when the catalog declares no matching
index. Authoritative model sections are not: an LPG or Both database always
carries `LPG_STORE`, and an RDF or Both database always carries `RDF_STORE`,
even when the dataset is empty. This preserves the RDF StoreId, history
provenance, graph-incarnation allocator, and statement handle namespace.
RDF-only containers carry a model-independent catalog payload and do not
fabricate an empty LPG section.

---

## Section Data (0x4000+)

Sections are written sequentially after the directory, each starting at a
page-aligned (4 KiB) offset. The next section starts at the first 4 KiB
boundary after the previous section ends.

```
0x4000  [CATALOG data ................] pad
0x5000  [LPG_STORE data ..............] pad
0xA000  [VECTOR_STORE data ...........] pad
...
```

### Data Section Encoding (Catalog, LPG, RDF)

Section payloads use versioned, section-specific schemas: bounded **bincode**
inside checked envelopes for some sections, custom packed encodings for others.
The directory's `version` byte identifies each section's grammar independently.

For LPG-bearing containers, the current `CATALOG` payload is version 6. It
persists user-visible named constraint ownership in addition to type
definitions and physical/logical index metadata, including HNSW, quantization,
and BM25 scoring configuration. Version 6 is also the explicit
graph-exact semantic boundary: every vector and text descriptor, in the
default or a named graph, is an empty recovery target whose contents must come
from the matching exact auxiliary-index section. Version 1–5 descriptor-rebuild
decoders still exist as historical compatibility paths requiring upstream review; they are not part of the
0.0.1 compatibility contract. These remaining paths lack complete persisted
ownership and cannot establish the planned single-index-owner guarantee.
RDF-only containers continue to use the model-independent Catalog state
payload version 1 rather than fabricating LPG index metadata.

The current `RDF_STORE` payload is version 6. It stores the canonical,
graph-qualified typed-Quad interval history: StoreId and history completeness,
named-graph create/drop/incarnation lifetimes, allocator high-water, stable
statement handles, transaction intervals, and optional valid-time intervals as
validated signed `i128` TAI nanoseconds. Its checksummed V3 addendum persists
bounded projection mappings and complete verified receipts. Only version 6 is
accepted. Versions 1–5 reject at the outer version boundary; their readers,
current-state imports and microsecond-promotion paths have been removed.

RDF→LPG projection metadata also has epoch-framed WAL V3 records. A declaration
binds the full mapping digest and format version. Publication is part of the
same LPG transaction as its rows and carries a bounded receipt containing the
StoreId, exact source graph incarnation/cut, target commit epoch, generation,
row count, reconciliation state, and full receipt digest. Replay requires the
receipt target epoch to equal its owning commit marker and rejects foreign,
conflicting, malformed, or trailing data. Older WAL status-record handling is
historical compatibility paths requiring upstream review, not a migration promise; it must never silently
promote unverified state to verified receipts.

Projection metadata records request the WAL durability barrier. In `Sync` mode
that request performs an immediate fsync before the API acknowledges the
metadata. `Batch`, `Adaptive`, and `NoSync` retain their configured weaker timing;
the phrase "requires sync" on a WAL record does not override the selected
durability mode.

`COMPACT_STORE` uses its own checksummed columnar encoding rather than bincode.
The current payload is version 8: version 4 introduced temporal node rows and
property validity, version 5 added relationship validity and closed-edge
history, version 6 added store-level structural node history so fully deleted
nodes survive save/load, version 7 added epoch-ordered node-label timelines,
and version 8 adds exact raw fallback histories for heterogeneous property
types and changing vector dimensions. Remaining version 1–7 readers are interim
historical compatibility paths requiring upstream review, not a supported 0.0.1 predecessor contract.

### World metadata

Every newly published section container includes `WORLD_METADATA` version 2.
It contains a portable `WorldCut`: logical StoreId, committed global epoch,
graph model, exact authoritative serializer versions, canonical catalog/schema
digest, verified projection receipts, RDF history-completeness boundary, and a
full BLAKE3 manifest digest. The WorldCut's domain-separated state digest binds
those logical coordinates to the exact plaintext bytes of every authoritative
model component.

Catalog, LPG, RDF (including its embedded dataset history), compact base, and
overlay deletions are authoritative model state. Vector, text, Ring, and
property indexes remain acceleration state and are deliberately excluded from
the logical WorldCut, so a physical rebuild does not invent a different world
at the same StoreId and epoch. Version 2 separately carries a domain-separated
recovery-image digest. The current V2 recovery profile covers the complete
canonical non-metadata section inventory—stable section type, serializer
version, length, and exact bytes—and a reserved canonical component for the
immutable `FileHeader` graph model plus the active `DbHeader` epoch,
transaction-ID replay floor, node count, and edge count. This prevents either a
CRC-valid index image or separately valid foreign recovery coordinates from
being paired with the local model while preserving reproducible logical cuts.
The frozen first V2 profile covered the same exact section inventory but
predates the synthetic recovery-coordinate component; it remains readable and
authoritative for exact auxiliary sections, while its active-`DbHeader`
transaction ID and cardinalities are not treated as authenticated.
Remaining version 1 metadata and section-only V2 handling are historical compatibility paths requiring upstream
review. An exact v3 vector/text image is never installed from an unsealed
v1 recovery image; neither predecessor profile authorizes a nonempty WAL tail.
Vector and Text persistence have no predecessor or unsealed-cache fallback:
non-v3 index sections are rejected before decoding, and exact v3 sections
require an authenticated recovery image. Unsealed Vector section payloads are
rejected rather than used as rebuild hints. Other predecessor Catalog and
WorldMetadata paths, including descriptor-only index rebuilding when no exact
section is present, require a separate upstream compatibility decision.
`WORLD_METADATA` is not self-hashed.

The separate portable graph snapshot format is currently version 9. Its v9
envelope retains v8's exact catalog, graph-scoped index metadata, temporal LPG
content, and signed `i128` TAI-nanosecond RDF valid time. It additionally stores
the logical `StoreId`, an explicit history-completeness boundary, the canonical
dataset-wide typed-Quad interval history, exact named-graph
create/drop/incarnation lifetimes, stable 256-bit statement handles, and the
graph-incarnation allocator high-water mark. Remaining v4–v8 DTOs and portable
current-state/microsecond-promotion paths are historical compatibility paths requiring upstream review.
This development format is not an upstream migration promise.

`OVERLAY_DELETIONS` is logically authoritative even though its directory
`required` bit is clear for compatibility with older readers. Its current
payload is version 2 and stores exact `(entity_id, delete_epoch)` records for
base nodes and edges. The remaining version 1 ID-only reader is historical compatibility paths requiring upstream
review, not a 0.0.1 format promise.

### Index Section Encoding (Vector, Text, Ring, Property)

Vector sections accept only graph-qualified exact version 3 packed
envelopes (`GVST`), including complete HNSW and quantization state. Version 1
and 2 readers, migration DTOs and topology-only hydration are deleted.
Text indexes accept only their
graph-qualified exact version 3 payload, including the committed temporal image
and exact tokenizer descriptor. A v6 catalog declaration makes its matching v3
vector/text section required, and the v2 recovery-image seal binds those bytes
to the complete container image. Ring sections accept only the version 2
packed envelope (`GRFR`); their predecessor bincode reader is deleted. There
is no persistent `PROPERTY_INDEX` decoder; logical property-index definitions
live in the catalog and are rebuilt. A reader rejects an advertised section
version or framing it cannot decode. The `mmap_able` bit describes the intended
access layout; it is not permission to ignore an unsupported payload.

---

## Checkpoint Flow

```
Checkpoint(reason):
  1. If periodic and every section is clean, skip publication.
  2. Otherwise capture one quiescent global commit cut and serialize the
     complete model-appropriate section set exactly once.
  3. Seal authoritative model bytes into the logical WorldCut and the complete
     non-metadata section inventory into the separate recovery-image digest.
  4. Compute per-section CRC-32 and write the complete v2 container to `{path}.installing`
     (file header, empty db headers, section data, directory, new header)
  5. fsync the temp file
  6. Close the live handle, rename `{path}.installing` over `{path}`, fsync the parent directory
  7. Reopen and exclusive-lock the installed primary
  8. (Engine) only then retire WAL (`checkpoint.meta` / truncate)
```

**Dirty tracking:** Each section has an `is_dirty()` flag. Dirty state decides
whether a periodic poll publishes at all; it never selects a partial container
image. The file manager atomically replaces the whole container, so once one
section requires publication every model and acceleration section in that cut
is included. All wrappers become clean only after installation succeeds.

**Crash safety:** Dual-slot headers protect header publication for readers, but
they do **not** protect section payloads. `write_sections` therefore never
overwrites the live `.grafeo` in place. A crash during steps 3–4 leaves the
previous primary intact and a leftover `{path}.installing`, which `open()`
deletes. A crash after rename but before WAL retirement keeps both the new
snapshot and the sidecar WAL. Dual-header alternation is not the commit point
for v2 data.

**Pre-fix files:** An in-place overwrite that already tore the primary cannot
be repaired if the previous checkpoint had retired the WAL. Grafeo therefore
fails closed on an unreadable or inconsistent active container; it does not
guess that a sidecar contains a complete prefix.

---

## Memory-Mapped Section Access

After a checkpoint, index sections with `flags.mmap_able = true` can be
memory-mapped for zero-copy read access. This is the foundation for tiered
storage: when RAM is scarce, index sections are flushed to the container
and served via mmap instead of keeping the full data in heap memory.

**Lifecycle:**

1. Engine flushes dirty sections via checkpoint
2. Engine calls `mmap_section()` for index sections
3. CRC-32 is verified against the mmap'd bytes (also warms page cache)
4. Engine drops in-memory copy of the section data
5. Reads go through the mmap (OS page cache manages eviction)
6. Before next checkpoint: drop all mmaps, then write

**Platform note:** On Windows, the OS rejects writes to a file with active
memory mappings (error 1224). All mmap handles must be dropped before
`write_sections()`. On Linux/macOS, writes succeed with active mappings
but the drop-before-write lifecycle is used on all platforms for consistency.

---

## Recovery

```
Open database:
  1. Read FileHeader at 0x0000, validate magic and format_version
  2. Read both DbHeaders (H1 at 0x1000, H2 at 0x2000)
  3. Select active header (highest iteration)
  4. Detect format:
     - snapshot_length > 0: v1 blob format (read snapshot at DATA_OFFSET)
     - snapshot_length == 0 && iteration > 0: v2 section format
  5. For v2: read section directory at 0x3000
  6. Read every directory entry and verify every CRC before model installation
  7. If WORLD_METADATA is present, verify its component digest and exact
     versions before deserializing any model. Current V2 images also verify the
     sealed immutable-`FileHeader` model and active-`DbHeader`
     epoch/replay-floor/cardinality coordinates; frozen section-only V2 images
     never contribute a trusted transaction-ID replay floor.
  8. Deserialize into detached constructor-owned stores, then cross-check the
     decoded StoreId/history, canonical schema, and verified projection receipts
  9. If sidecar WAL exists: scan without repair, then resolve immutable
     identity/model agreement using the fully decoded RecoverySealedV2
     checkpoint and retained WAL declarations. A loaded unverified or
     predecessor checkpoint cannot authorize a nonempty tail. Compare
     retained pre-checkpoint identity/model declarations without replaying
     their data or allowing them to supply missing authority. Only after
     validation, filter committed groups using the authenticated checkpoint
     epoch and transaction floor, then repair any incomplete final frame.
 10. Database is ready
```

WAL-directory startup has no authenticated container proof: prior durable
input must supply both its own identity and model in the recovery suffix.
An empty committed list does not establish fresh startup; checkpoint metadata
and aborted-only, orphaned-only or torn-only bytes are prior state too.
Authority rejection leaves the source container and WAL bytes unchanged.
This does not extend to the file manager's existing `.installing` cleanup.
The validated scanner retains the scanned descriptor for deferred repair and
checks its length, but still requires exclusive source access. Single-file
startup holds its container lock; WAL-directory cross-process writer exclusion
is not yet enforced and remains a release blocker. Read-only snapshot opens
continue to bypass WAL replay.

Before a frame can affect transaction grouping, high-water tracking, or the
last-good recovery boundary, recovery validates the record-local semantics of
storage-decoded durable epoch fields. The reserved `PENDING` sentinel is never
a valid commit, live catalog/projection-declaration, v2 projection
source/target, or checkpoint-epoch coordinate; it remains legal where a data
schema explicitly defines an open or uncommitted interval. Opaque engine-owned
payloads are validated by their engine decoder. Live catalog/projection
publication epochs and projection target epochs must be positive real epochs;
synthetic exact-history `Committed(0)` boundaries and legacy/source epoch zero
remain readable for compatibility. This record-local check does not prove
global epoch monotonicity, unique epoch ownership, repeated-transaction-marker
consistency or exact checkpoint decoding; those need their separate gates.
The physical scanner independently checks that the checkpoint's exact WAL
segment exists and that the replay suffix is contiguous. A checksum-valid but
semantically invalid storage-decoded frame stops recovery with `GRAFEO-S002`
(`InvalidWalEntry`), remains byte-identical in place for diagnosis or manual
repair, and is not silently truncated or moved to corruption quarantine.

The current WAL payload is `GRAFOWAL` + little-endian `u16(1)` + one exact
bincode record. Length/CRC or AEAD framing covers that complete payload, with
the existing 64 MiB bound including the envelope and encryption overhead.
Readers authenticate before interpreting the generation. Missing/unsupported
generations fail as invalid entries, never repairable torn tails. Direct writer
open and recovery both validate retained pre-checkpoint segments; no predecessor
reader or migration is provided.

Metadata-free container handling remains historical compatibility paths requiring upstream review, subject
to every other section-admission check. It cannot authorize a nonempty WAL
tail. Its existing rewrite-on-checkpoint behavior is not a supported 0.0.1
upgrade or migration contract.

---

## Periodic Checkpoints

When `Config::checkpoint_interval` is set, a background thread periodically
flushes sections to the container. This bounds WAL size and recovery work;
acknowledged-commit durability remains governed by the WAL mode and defaults
to strict `Sync`.

Periodic checkpoints currently run only for the flat LPG topology. Entering
compact/layered mode stops the timer, and reopening a layered container does
not start it: a timer bound to one mutable overlay cannot serialize the cold
base, overlay-deletion log, and future overlay replacements as one generation.
Use `wal_checkpoint()` (or `close()`) for topology-aware layered checkpoints.
This is a temporary automatic-scheduling qualification boundary, not a reduced
storage contract: a topology-neutral periodic checkpoint worker remains the
target, and explicit checkpoints already preserve the complete layered state.

The timer polls a shutdown flag every 100 ms. On database close, the timer
is stopped before the final checkpoint to prevent races.

---

## File Locking

- **Exclusive lock** on create/open (read-write mode): prevents concurrent
  writers on the same file.
- **Shared lock** on open (read-only mode): allows multiple concurrent
  readers.
- Locks are released on close or drop.

---

## Size Estimates

| Component | Size |
|-----------|------|
| Fixed overhead (headers + directory) | 16 KiB |
| Empty database (headers + empty catalog + LPG) | ~20 KiB |
| Per-section overhead | 32 bytes (directory entry) + page alignment padding |
| Typical 10K-node LPG | ~1-5 MB |
| 1M-vector HNSW index (384-dim, f32) | ~1.5 GB |

---

## Version History

| Version | Format | Notes |
|---------|--------|-------|
| v1 (0.5.0-0.5.34) | Monolithic blob at `DATA_OFFSET` | Single bincode snapshot |
| v2 (0.5.35+) | Section-based with directory at `0x3000` | Independent sections, mmap support |
