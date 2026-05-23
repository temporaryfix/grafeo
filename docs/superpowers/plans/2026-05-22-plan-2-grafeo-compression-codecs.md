# Plan 2 — Grafeo Compression Codecs (Overview)

> **For agentic workers:** This is the *overview* document for Plan 2 of the Nota
> data-efficiency program. It scopes and sequences four sub-plans. Each sub-plan
> is its own executable document under `docs/superpowers/plans/`. Execute them
> in the order given. Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` per sub-plan.

**Goal:** Give the Grafeo fork four independently-usable, independently-tested
compression codecs — vector quantization, static graph adjacency, string
compression, and zero-copy chunk loading — exposed through the WASM bindings so
Nota's TypeScript build pipeline (Plan 3) can encode and query each component.

**Architecture:** Each codec is a self-contained module that can (a) encode a
data component to a compressed byte blob and (b) open and query that blob. The
deliverable is *component-wise* export — three separate blobs (vectors,
adjacency, strings) — not a monolithic snapshot. Plan 1 wraps each blob in a
content-addressed chunk; Plan 3 calls these codecs to produce the blobs.

**Tech stack:** Rust 2024, `grafeo-core` for the codecs, `crates/bindings/wasm`
for the JS surface. No new third-party crates except where a sub-plan explicitly
spikes one and confirms it builds for `wasm32-unknown-unknown`.

---

## Constraints carried into every sub-plan

- **Upstreamable.** All codecs are generic algorithms (RaBitQ, WebGraph BV,
  FSST). Keep them free of Nota-specific schema, entity types, or business
  logic so the work can be a clean upstream PR to mainline Grafeo. Open an RFC
  before touching any shared storage primitive; codecs own their own types.
- **WASM budget.** Everything must compile to `wasm32-unknown-unknown` and fit
  the 248 MB WASM heap budget. Prefer in-tree, dependency-free implementations.
- **Regression discipline.** The fork guards `compact()` with a property test
  (`crates/grafeo-engine/tests/compact_roundtrip_proptest.rs`). Each codec adds
  an equivalent: a *recall-floor* proptest for the vector codec, *round-trip
  equivalence* proptests for adjacency and strings.
- **No gratuitous refactors.** Extend existing modules (`codec/`, `index/vector/`,
  `graph/compact/section.rs`); do not restructure working code.

---

## Sub-plan sequence

### 2a — RaBitQ + int8 two-stage vector codec  *(priority 1, unblocks vectors)*

Document: `2026-05-22-plan-2a-rabitq-vector-codec.md` — **written in full.**

1-bit-per-dimension RaBitQ binary quantization with a random orthogonal
rotation and per-vector correction factor, SIMD popcount/XOR distance, and a
two-stage search (RaBitQ coarse pass → rerank top-K against int8-quantized
vectors). int8 is the existing `ScalarQuantizer` — confirmed, reused. PQ
(`ProductQuantizer`, `quantization.rs:541`) stays as the documented fallback;
a criterion benchmark compares the two so the fallback decision is evidence-based.

### 2c — FSST string compression  *(priority 2, small and independent)*

Document: `2026-05-22-plan-2c-fsst-string-codec.md` — **to be written.**

In-tree Fast Static Symbol Table codec: train a 256-entry symbol table from a
string sample, compress with single-byte codes, decode any individual string in
O(1) without touching its neighbours. New module `codec/fsst.rs`. Integrates as
a new `ColumnCodec` variant for compact-store string columns
(`graph/compact/column.rs:227`) and is also usable standalone for the string
blob. Round-trip equivalence proptest (every input string decodes bit-identical;
random access matches sequential decode).

### 2b — WebGraph adjacency format  *(priority 3, carries a crate-evaluation risk)*

Document: `2026-05-22-plan-2b-webgraph-adjacency-codec.md` — **to be written.**

Static compressed adjacency: gap coding + referentiation + zeta-3 codes
(~3–8 bits/edge). Snapshots are immutable after build, so the static-sorted-graph
assumption holds. **First task is an explicit spike:** attempt to build the
`webgraph` crate for `wasm32-unknown-unknown`; if it pulls in `mmap`/`rayon`/
`epserde` and fails, fall back to an in-tree BV implementation in
`codec/webgraph.rs`. Either way, traversal must run against the compressed form
without full decompression (streaming successor iteration). New static sibling
to `index/adjacency.rs`'s `ChunkedAdjacency`; do not modify the mutable one.
Round-trip equivalence proptest (compressed adjacency yields the same
neighbour sets as the source graph).

### 2d — Zero-copy chunk loading  *(priority 4, implemented last, designed first)*

Document: `2026-05-22-plan-2d-zero-copy-chunk-loading.md` — **to be written.**

A chunk memory layout the engine can `mmap` and query without a deserialize
pass. Extends the existing zero-copy path in `graph/compact/section.rs`
(`deserialize_from_bytes(Bytes)`, which already borrows from mmap regions via
`Bytes::from_owner`). Query-hot structures (RaBitQ codes, compressed adjacency)
are queryable in place; only cold data may need a decompress step. Preserves the
~13 ms client init. Honours Plan 1's per-chunk open-string codec field.

**Designed early:** the zero-copy layout rules below are a hard contract that
sub-plans 2a/2b/2c must satisfy *now*, even though 2d implements the loader last.

---

## Cross-cutting zero-copy contract (applies to 2a, 2b, 2c)

Every codec's `to_bytes()` output MUST be openable by reference, not by copy:

1. **Fixed header.** Each blob starts with a 4-byte magic, 1-byte version,
   and a fixed-layout header of `u32`/`u64` fields. No length-prefixed
   variable header before the first array.
2. **Natural alignment.** Arrays of `u64`/`f32` start at offsets that are
   multiples of their element size. Pad with zero bytes; record padding in the
   header so the reader does not guess.
3. **No internal pointers.** Offsets are stored relative to the start of the
   blob, as `u64`. A blob is position-independent and relocatable.
4. **Borrowable arrays.** `from_bytes` accepts `bytes::Bytes`; large arrays are
   constructed via `Bytes::slice(range)` (refcount bump, no copy), matching
   `CompactStoreSection::deserialize_from_bytes`. The `&[u8]` entry point may
   incur one boundary copy — that is acceptable for cold data only.
5. **Self-describing.** The header records element counts and dimensions so a
   reader can bounds-check every slice without a separate schema.

Sub-plan 2a Task 9 ("Blob serialize/deserialize") is the first place this
contract is exercised; review it as the reference implementation for 2b/2c.

---

## Definition of done for Plan 2

- Four codec modules, each with `to_bytes`/`from_bytes` honouring the zero-copy
  contract, each query-able in place.
- WASM bindings for each, callable from TypeScript.
- One recall-floor proptest (vectors) and three round-trip proptests
  (adjacency, strings, chunk layout) — all green.
- A criterion benchmark comparing RaBitQ against PQ.
- `CHANGELOG.md` entries; no modifications to shared storage primitives without
  an RFC.
