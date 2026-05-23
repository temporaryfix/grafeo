# Sub-plan 2b — WebGraph Adjacency Codec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static compressed adjacency codec to the Grafeo fork that compresses sorted successor lists with gap coding + gamma codes and supports streaming successor iteration without full decompression, exposed through the WASM bindings.

**Architecture:** A new in-tree, dependency-free module `codec/webgraph.rs`. For each node `u`, encode its sorted successor list `[v1, v2, ...]` as gap-coded gammas: emit the out-degree `k` in gamma, then the first gap `v1 − u` as a zigzag-gamma (signed) to handle dst < src, then subsequent gaps `vi − v(i-1)` in plain gamma. A bit-offset index `offsets[u]` gives O(1) seek to node `u`'s adjacency list, enabling per-node successor iteration without decompressing any other lists. Blob `to_bytes`/`from_bytes` honours Plan 2's zero-copy contract (fixed header, blob-relative `u64` offsets, `bytes::Bytes` overload, CRC32). New static sibling to `index/adjacency.rs`'s mutable `ChunkedAdjacency`; the latter is not touched.

**Tech Stack:** Rust 2024, `grafeo-core` (`codec/`), `crates/bindings/wasm`, `proptest` (round-trip equivalence). No new runtime dependencies — the `webgraph` crate was eliminated by a `wasm32` build spike (fails on `getrandom`, `mmap-rs`, `rayon-core`, `sysctl`).

**Why gamma, not zeta-3 or referentiation:** The WebGraph paper's ~3–8 bits/edge figure assumes both referentiation (intra-list redundancy of similar adjacency lists, a web-graph property) and zeta-3 (tuned for power-law gap distributions). Nota's social-graph-style adjacency has neither property strongly. Gap + gamma + per-node offset index is a tractable single-pass codec that:
- Compresses dense neighborhoods to ~`log₂(N) + small` bits per gap (vs. 64 bits/edge raw)
- Supports streaming successor iteration with no decompression buffer
- Keeps the bit-I/O surface small enough to fit one focused file
- Leaves zeta-3 and referentiation as later wins behind a clean `WebGraphCodec` API (see "Follow-ups" at the end)

---

## File Structure

- **Create** `crates/grafeo-core/src/codec/bitstream.rs` — bit-level reader/writer with gamma codec (private to the crate; future codecs may reuse).
- **Create** `crates/grafeo-core/src/codec/webgraph.rs` — the codec: `WebGraphError`, `WebGraphBuilder`, `WebGraphCodec`, `SuccessorIter`, blob serialization, unit tests.
- **Modify** `crates/grafeo-core/src/codec/mod.rs` — register `pub mod webgraph;` (and `mod bitstream;` privately) and re-export the public types.
- **Create** `crates/grafeo-core/tests/webgraph_round_trip.rs` — proptest + fixed-seed regression cases.
- **Create** `crates/bindings/wasm/src/codecs.rs` extension — add `#[wasm_bindgen] struct WebGraphCodec` next to the existing `RabitqCodec` and `FsstCodec`.
- **Modify** `crates/bindings/wasm/Cargo.toml` — add `webgraph-codec = ["dep:grafeo-core"]` feature.
- **Modify** `crates/bindings/wasm/src/lib.rs` — extend the codec-module gate to include the new feature.
- **Modify** `crates/bindings/wasm/tests/web.rs` — wasm round-trip test gated on `webgraph-codec`.
- **Modify** `CHANGELOG.md` — note the new codec.

Reference facts from the existing code (read before starting):
- `crates/grafeo-core/src/index/adjacency.rs` — the mutable `ChunkedAdjacency`. **Do not modify.** WebGraph is the immutable sibling.
- `crates/grafeo-core/src/codec/bitvec.rs` — existing `BitVector` is a sequential bit-array, not a bit-stream codec. WebGraph needs bit-level write/read with gamma/zeta codes — a separate `bitstream.rs` is the right unit.
- `crates/grafeo-core/src/codec/bitpack.rs` — `BitPackedInts` is fixed-width bit-packed integers. Different shape from variable-length codes; cannot be reused directly.
- `crates/grafeo-core/src/index/vector/rabitq.rs` (sub-plan 2a) and `crates/grafeo-core/src/codec/fsst.rs` (sub-plan 2c) — the blob layout pattern (fixed header with `u64` offsets, `from_bytes_shared(Bytes)`, CRC trailer) is established. Mirror it.

---

## Task 1: Bit-stream codec — `BitWriter` and `BitReader` with gamma codes

**Files:**
- Create: `crates/grafeo-core/src/codec/bitstream.rs`
- Modify: `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/grafeo-core/src/codec/bitstream.rs` with this content:

```rust
//! Bit-level reader/writer with Elias gamma codes.
//!
//! Universal variable-length integer codes used by static graph and
//! sequence codecs. `gamma(n)` for `n >= 1` is `floor(log2(n))` zeros
//! followed by the binary of `n` (a unary prefix + the bits below the
//! top one). `gamma(1) = "1"`, `gamma(2) = "010"`, `gamma(4) = "00100"`.
//!
//! `zigzag_gamma(n)` for any `i64` maps `n` to `n >= 0 ? 2n+1 : -2n` and
//! gamma-encodes the result, giving variable-length encoding for signed
//! values (used for the first gap in an adjacency list, which may be
//! negative when `dst < src`).

/// Maximum integer encodable in `gamma` without overflow on the decoder
/// side: `2^63 - 1`. The encoded length for the maximum is 127 bits
/// (63 zeros + 64-bit binary).
pub(crate) const GAMMA_MAX: u64 = u64::MAX >> 1;

/// A growable bit-packed write buffer.
///
/// Bits are appended MSB-first within each byte; a `BitReader` over the
/// same buffer reads them in the same order. The total bit length is
/// tracked separately from the byte length so trailing zero-padding in
/// the last byte is not interpreted as part of the stream.
#[derive(Debug, Clone, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Number of bits actually written. Always `<= bytes.len() * 8`.
    bit_len: u64,
}

impl BitWriter {
    /// Creates an empty writer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the number of bits written so far.
    pub(crate) fn bit_len(&self) -> u64 {
        self.bit_len
    }

    /// Appends a single bit (LSB of `bit`).
    pub(crate) fn write_bit(&mut self, bit: u8) {
        // reason: bit fits in u8::from(bool)
        #[allow(clippy::cast_possible_truncation)]
        let byte_idx = (self.bit_len / 8) as usize;
        let bit_in_byte = 7 - (self.bit_len % 8) as u8;
        if byte_idx == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit & 1 != 0 {
            self.bytes[byte_idx] |= 1u8 << bit_in_byte;
        }
        self.bit_len += 1;
    }

    /// Appends the low `nbits` of `value`, most-significant bit first.
    pub(crate) fn write_bits(&mut self, value: u64, nbits: u8) {
        debug_assert!(nbits <= 64, "nbits must be <= 64");
        for i in (0..nbits).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Appends `gamma(n)` for `n >= 1`.
    ///
    /// # Panics
    /// Panics if `n == 0` or `n > GAMMA_MAX`.
    pub(crate) fn write_gamma(&mut self, n: u64) {
        assert!(n >= 1, "gamma encodes n >= 1, got 0");
        assert!(n <= GAMMA_MAX, "gamma overflow: n={n} > GAMMA_MAX");
        let bits_needed = 64 - n.leading_zeros() as u8; // 1..=64
        // unary prefix: (bits_needed - 1) zeros + a terminating 1.
        for _ in 0..(bits_needed - 1) {
            self.write_bit(0);
        }
        // The remaining body is the binary of n in `bits_needed` bits,
        // MSB first — but the MSB is the terminating 1 of the unary
        // prefix, so we write the full `bits_needed`-bit value here.
        self.write_bits(n, bits_needed);
    }

    /// Appends `zigzag_gamma(n)`: maps `n` to non-negative, then gamma.
    pub(crate) fn write_zigzag_gamma(&mut self, n: i64) {
        // n >= 0 -> 2n + 1; n < 0 -> -2n. Always yields a positive u64.
        let folded: u64 = if n >= 0 {
            (n as u64).checked_mul(2).expect("zigzag overflow")
                + 1
        } else {
            // reason: n is strictly negative, abs fits in u64
            #[allow(clippy::cast_sign_loss)]
            {
                ((-(n + 1)) as u64)
                    .checked_mul(2)
                    .expect("zigzag overflow")
                    + 2
            }
        };
        self.write_gamma(folded);
    }

    /// Consumes the writer and returns the underlying bytes plus the
    /// exact bit length.
    pub(crate) fn into_bytes(self) -> (Vec<u8>, u64) {
        (self.bytes, self.bit_len)
    }
}

/// A bit-level reader over a borrowed byte slice.
///
/// `bit_pos` tracks the current bit offset from the start of the slice;
/// callers can save and restore it to implement random access (the
/// WebGraph codec uses this for per-node seek via the offsets index).
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: u64,
    bit_len: u64,
}

impl<'a> BitReader<'a> {
    /// Creates a reader over `bytes` with the given total `bit_len`.
    pub(crate) fn new(bytes: &'a [u8], bit_len: u64) -> Self {
        Self {
            bytes,
            bit_pos: 0,
            bit_len,
        }
    }

    /// Seeks to `bit_pos` from the start of the stream.
    pub(crate) fn seek(&mut self, bit_pos: u64) {
        self.bit_pos = bit_pos;
    }

    /// Current bit position from the start of the stream.
    pub(crate) fn bit_pos(&self) -> u64 {
        self.bit_pos
    }

    /// Reads one bit (returns 0 or 1).
    ///
    /// # Errors
    /// Returns `None` if past the end of the stream.
    pub(crate) fn read_bit(&mut self) -> Option<u8> {
        if self.bit_pos >= self.bit_len {
            return None;
        }
        // reason: bit_pos < bit_len <= bytes.len() * 8, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let byte_idx = (self.bit_pos / 8) as usize;
        let bit_in_byte = 7 - (self.bit_pos % 8) as u8;
        self.bit_pos += 1;
        Some((self.bytes[byte_idx] >> bit_in_byte) & 1)
    }

    /// Reads `nbits` bits MSB-first into a `u64`.
    pub(crate) fn read_bits(&mut self, nbits: u8) -> Option<u64> {
        debug_assert!(nbits <= 64);
        let mut acc = 0u64;
        for _ in 0..nbits {
            acc = (acc << 1) | u64::from(self.read_bit()?);
        }
        Some(acc)
    }

    /// Reads one gamma-encoded integer.
    pub(crate) fn read_gamma(&mut self) -> Option<u64> {
        // Count leading zeros (unary prefix length).
        let mut zeros: u8 = 0;
        loop {
            match self.read_bit()? {
                0 => zeros += 1,
                _ => break,
            }
            if zeros > 63 {
                return None; // Malformed: prefix would overflow.
            }
        }
        // We consumed `zeros + 1` bits; the leading 1 is the high bit of
        // the value. Read the remaining `zeros` low bits.
        if zeros == 0 {
            return Some(1);
        }
        let low = self.read_bits(zeros)?;
        Some((1u64 << zeros) | low)
    }

    /// Reads one zigzag-gamma-encoded signed integer.
    pub(crate) fn read_zigzag_gamma(&mut self) -> Option<i64> {
        let folded = self.read_gamma()?;
        // odd -> non-negative, even -> negative.
        Some(if folded & 1 == 1 {
            // reason: folded - 1 is even, half fits i64 in practice
            #[allow(clippy::cast_possible_wrap)]
            (((folded - 1) >> 1) as i64)
        } else {
            // reason: folded is even and >= 2, half fits i64
            #[allow(clippy::cast_possible_wrap)]
            (-((folded >> 1) as i64))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_round_trip_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0b1011_0100, 8);
        w.write_bits(0b11, 2);
        let (bytes, bits) = w.into_bytes();
        assert_eq!(bits, 10);
        let mut r = BitReader::new(&bytes, bits);
        assert_eq!(r.read_bits(8), Some(0b1011_0100));
        assert_eq!(r.read_bits(2), Some(0b11));
        assert_eq!(r.read_bit(), None);
    }

    #[test]
    fn gamma_known_values() {
        // gamma(1) = "1" (1 bit), gamma(2) = "010" (3 bits),
        // gamma(3) = "011" (3 bits), gamma(4) = "00100" (5 bits),
        // gamma(7) = "00111" (5 bits), gamma(8) = "0001000" (7 bits).
        for n in [1u64, 2, 3, 4, 7, 8, 100, 12345] {
            let mut w = BitWriter::new();
            w.write_gamma(n);
            let (bytes, bits) = w.into_bytes();
            let mut r = BitReader::new(&bytes, bits);
            assert_eq!(r.read_gamma(), Some(n), "round-trip failed for {n}");
        }
    }

    #[test]
    fn zigzag_gamma_round_trip() {
        for n in [
            0i64,
            1,
            -1,
            2,
            -2,
            42,
            -42,
            1000,
            -1000,
            i32::MAX as i64,
            i32::MIN as i64,
        ] {
            let mut w = BitWriter::new();
            w.write_zigzag_gamma(n);
            let (bytes, bits) = w.into_bytes();
            let mut r = BitReader::new(&bytes, bits);
            assert_eq!(r.read_zigzag_gamma(), Some(n), "round-trip failed for {n}");
        }
    }

    #[test]
    fn gamma_then_bits_then_gamma() {
        // Multiple codes interleaved must round-trip without bit drift.
        let mut w = BitWriter::new();
        w.write_gamma(13);
        w.write_bits(0b101, 3);
        w.write_gamma(1);
        w.write_zigzag_gamma(-5);
        let (bytes, bits) = w.into_bytes();
        let mut r = BitReader::new(&bytes, bits);
        assert_eq!(r.read_gamma(), Some(13));
        assert_eq!(r.read_bits(3), Some(0b101));
        assert_eq!(r.read_gamma(), Some(1));
        assert_eq!(r.read_zigzag_gamma(), Some(-5));
    }

    #[test]
    fn bitwriter_records_exact_bit_length() {
        let mut w = BitWriter::new();
        for _ in 0..10 {
            w.write_bit(1);
        }
        assert_eq!(w.bit_len(), 10);
        let (bytes, bits) = w.into_bytes();
        // 10 bits → 2 bytes; the second byte has 2 high bits set.
        assert_eq!(bytes.len(), 2);
        assert_eq!(bits, 10);
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1] & 0b1100_0000, 0b1100_0000);
    }

    #[test]
    fn seek_enables_random_access() {
        let mut w = BitWriter::new();
        w.write_gamma(1);
        let pos_after_first = w.bit_len();
        w.write_gamma(2);
        let pos_after_second = w.bit_len();
        w.write_gamma(3);
        let (bytes, bits) = w.into_bytes();

        let mut r = BitReader::new(&bytes, bits);
        r.seek(pos_after_second);
        assert_eq!(r.read_gamma(), Some(3));

        r.seek(pos_after_first);
        assert_eq!(r.read_gamma(), Some(2));

        r.seek(0);
        assert_eq!(r.read_gamma(), Some(1));
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/grafeo-core/src/codec/mod.rs`, after the line `pub mod bitpack;` add:

```rust
pub(crate) mod bitstream;
```

(Lowercase `mod` only — the bit-stream types are crate-private helpers used by `webgraph.rs`; they are not part of the codec's public surface.)

- [ ] **Step 3: Run the tests**

Run: `cargo test -p grafeo-core --lib codec::bitstream`
Expected: PASS — 6 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/src/codec/bitstream.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(codec): bit-stream reader/writer with gamma codes"
```

---

## Task 2: `WebGraphError` and `WebGraphBuilder` scaffold

**Files:**
- Create: `crates/grafeo-core/src/codec/webgraph.rs`
- Modify: `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/grafeo-core/src/codec/webgraph.rs` with this content:

```rust
//! WebGraph-style static compressed adjacency.
//!
//! Each node's sorted successor list is encoded as gap-coded gammas: the
//! out-degree `k` is emitted in gamma, then the first gap `v1 − u` as
//! zigzag-gamma (signed, since `dst < src` is allowed), then subsequent
//! gaps `vi − v(i-1)` in plain gamma (always positive — successors are
//! sorted and de-duplicated). A bit-offset index `offsets[u]` gives O(1)
//! seek to any node's adjacency without decompressing other lists.
//!
//! Snapshots are immutable after build, so the static-sorted-graph
//! assumption holds. For mutable graphs see
//! [`crate::index::adjacency::ChunkedAdjacency`].

use super::bitstream::{BitReader, BitWriter};

/// Errors returned when opening or building a WebGraph blob.
#[derive(Debug, thiserror::Error)]
pub enum WebGraphError {
    /// The blob is shorter than the bytes a field needs.
    #[error("webgraph blob truncated: need {need} bytes, have {have}")]
    Truncated {
        /// Minimum required buffer length (end offset of the failed read).
        need: usize,
        /// Bytes available.
        have: usize,
    },
    /// The leading magic bytes are not `GWBG`.
    #[error("webgraph blob: bad magic (expected GWBG)")]
    BadMagic,
    /// The version byte is not supported.
    #[error("webgraph blob: unsupported version {0}")]
    BadVersion(u8),
    /// The trailing CRC32 does not match the body.
    #[error("webgraph blob: crc mismatch (stored {stored:#010x}, computed {computed:#010x})")]
    CrcMismatch {
        /// CRC read from the trailer.
        stored: u32,
        /// CRC computed over the body.
        computed: u32,
    },
    /// An edge references a node outside `[0, num_nodes)`.
    #[error("webgraph: edge ({src}, {dst}) out of range — num_nodes is {num_nodes}")]
    EdgeOutOfRange {
        /// Source node id.
        src: u64,
        /// Destination node id.
        dst: u64,
        /// Number of nodes declared at builder construction.
        num_nodes: u64,
    },
    /// The bit-offset for a node is malformed (decreasing or past end).
    #[error("webgraph blob: offset {offset} for node {node} exceeds bit stream length {bit_len}")]
    BadOffset {
        /// Node whose offset is invalid.
        node: u64,
        /// The out-of-range offset (bit position).
        offset: u64,
        /// Total bit-stream length.
        bit_len: u64,
    },
    /// Reached the end of the stream mid-record (malformed gamma).
    #[error("webgraph decode: unexpected end of stream at bit {0}")]
    UnexpectedEnd(u64),
}

/// Builds a [`WebGraphCodec`] from `(src, dst)` edges.
///
/// Edges may be added in any order; `build()` sorts them per source and
/// de-duplicates parallel edges.
#[derive(Debug, Clone)]
pub struct WebGraphBuilder {
    num_nodes: u64,
    edges: Vec<(u64, u64)>,
}

impl WebGraphBuilder {
    /// Creates an empty builder for a graph with `num_nodes` nodes. Node
    /// ids are `0..num_nodes`.
    #[must_use]
    pub fn new(num_nodes: u64) -> Self {
        Self {
            num_nodes,
            edges: Vec::new(),
        }
    }

    /// Adds an edge `src -> dst`.
    ///
    /// # Errors
    /// Returns `EdgeOutOfRange` if `src` or `dst` is `>= num_nodes`.
    pub fn add_edge(&mut self, src: u64, dst: u64) -> Result<(), WebGraphError> {
        if src >= self.num_nodes || dst >= self.num_nodes {
            return Err(WebGraphError::EdgeOutOfRange {
                src,
                dst,
                num_nodes: self.num_nodes,
            });
        }
        self.edges.push((src, dst));
        Ok(())
    }

    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges added so far (before de-duplication).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_records_added_edges() {
        let mut b = WebGraphBuilder::new(10);
        b.add_edge(0, 5).unwrap();
        b.add_edge(0, 3).unwrap();
        b.add_edge(2, 7).unwrap();
        assert_eq!(b.num_nodes(), 10);
        assert_eq!(b.edge_count(), 3);
    }

    #[test]
    fn builder_rejects_out_of_range_edges() {
        let mut b = WebGraphBuilder::new(5);
        assert!(matches!(
            b.add_edge(5, 0),
            Err(WebGraphError::EdgeOutOfRange { src: 5, .. })
        ));
        assert!(matches!(
            b.add_edge(0, 5),
            Err(WebGraphError::EdgeOutOfRange { dst: 5, .. })
        ));
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/grafeo-core/src/codec/mod.rs`, after the `pub(crate) mod bitstream;` line from Task 1, add:

```rust
pub mod webgraph;
```

(Do NOT add a `pub use webgraph::{...}` re-export yet — types are added in later tasks; the re-export goes in Task 4 Step 5.)

- [ ] **Step 3: Run the tests**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: PASS — 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/src/codec/webgraph.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(webgraph): module scaffold with WebGraphError and WebGraphBuilder"
```

---

## Task 3: `WebGraphBuilder::build` — encode all adjacency lists

**Files:**
- Modify: `crates/grafeo-core/src/codec/webgraph.rs`

- [ ] **Step 1: Write the failing test**

Add to the existing `tests` module:

```rust
#[test]
fn build_empty_graph_has_no_edges() {
    let codec = WebGraphBuilder::new(0).build();
    assert_eq!(codec.num_nodes(), 0);
    assert_eq!(codec.num_edges(), 0);
}

#[test]
fn build_isolated_nodes_have_zero_degree() {
    let codec = WebGraphBuilder::new(5).build();
    assert_eq!(codec.num_nodes(), 5);
    assert_eq!(codec.num_edges(), 0);
    for u in 0u64..5 {
        assert_eq!(codec.out_degree(u), 0);
    }
}

#[test]
fn build_records_edge_count_after_deduplication() {
    let mut b = WebGraphBuilder::new(4);
    b.add_edge(0, 1).unwrap();
    b.add_edge(0, 2).unwrap();
    b.add_edge(0, 1).unwrap(); // duplicate
    b.add_edge(1, 2).unwrap();
    let codec = b.build();
    assert_eq!(codec.num_edges(), 3); // duplicate removed
    assert_eq!(codec.out_degree(0), 2);
    assert_eq!(codec.out_degree(1), 1);
    assert_eq!(codec.out_degree(2), 0);
    assert_eq!(codec.out_degree(3), 0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: FAIL — `no method named 'build'`.

- [ ] **Step 3: Write the implementation**

Add to `webgraph.rs`, after the `WebGraphBuilder` impl (and before the `#[cfg(test)] mod tests`):

```rust
impl WebGraphBuilder {
    /// Sorts and de-duplicates the added edges, then encodes each node's
    /// adjacency list with gap+gamma.
    ///
    /// Returns a [`WebGraphCodec`] holding the bit-packed stream and a
    /// per-node bit-offset index.
    #[must_use]
    pub fn build(mut self) -> WebGraphCodec {
        // Sort by (src, dst) then de-duplicate parallel edges.
        self.edges.sort_unstable();
        self.edges.dedup();

        let mut writer = BitWriter::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(self.num_nodes as usize + 1);

        // Walk edges in source order; each node's successors are a
        // contiguous sub-slice.
        let mut idx: usize = 0;
        for u in 0..self.num_nodes {
            offsets.push(writer.bit_len());

            // Find the end of node u's successor block.
            let start = idx;
            while idx < self.edges.len() && self.edges[idx].0 == u {
                idx += 1;
            }
            let successors = &self.edges[start..idx];
            let k = successors.len() as u64;

            // Encode out-degree + 1 in gamma (so degree=0 encodes as gamma(1)).
            writer.write_gamma(k + 1);
            if k == 0 {
                continue;
            }

            // First gap is signed: v1 - u, encoded as zigzag-gamma.
            // reason: edges validated as u64 < num_nodes in add_edge
            #[allow(clippy::cast_possible_wrap)]
            let first_gap = successors[0].1 as i64 - u as i64;
            writer.write_zigzag_gamma(first_gap);

            // Subsequent gaps are strictly positive (sorted, deduped).
            for w in successors.windows(2) {
                let gap = w[1].1 - w[0].1; // > 0
                writer.write_gamma(gap);
            }
        }

        // Terminal offset = total bit length, lets `successors` know the
        // end of the last node's adjacency.
        offsets.push(writer.bit_len());

        let (bytes, bit_len) = writer.into_bytes();
        let num_edges = self.edges.len() as u64;
        WebGraphCodec {
            num_nodes: self.num_nodes,
            num_edges,
            offsets,
            bits: bytes,
            bit_len,
        }
    }
}

/// A static compressed adjacency in WebGraph-style gap+gamma encoding.
///
/// Built via [`WebGraphBuilder::build`]; loaded via [`Self::from_bytes`].
/// Per-node successor iteration is supported in-place without
/// decompressing other nodes via the bit-offset index.
#[derive(Debug, Clone)]
pub struct WebGraphCodec {
    pub(crate) num_nodes: u64,
    pub(crate) num_edges: u64,
    /// Bit offset of each node's adjacency block in `bits`. Length
    /// `num_nodes + 1`; the trailing entry is the total bit length.
    pub(crate) offsets: Vec<u64>,
    /// Bit-packed adjacency stream.
    pub(crate) bits: Vec<u8>,
    /// Number of bits actually written (`<= bits.len() * 8`).
    pub(crate) bit_len: u64,
}

impl WebGraphCodec {
    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges (post-deduplication).
    #[must_use]
    pub fn num_edges(&self) -> u64 {
        self.num_edges
    }

    /// Out-degree of `node`. Reads only the first gamma of the node's
    /// adjacency block — O(log degree).
    #[must_use]
    pub fn out_degree(&self, node: u64) -> u64 {
        if node >= self.num_nodes {
            return 0;
        }
        let start = self.offsets[node as usize];
        let mut reader = BitReader::new(&self.bits, self.bit_len);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        reader.read_gamma().unwrap_or(1) - 1
    }
}
```

Note: this task does not yet implement `successors` (the iterator) or the blob `to_bytes`/`from_bytes`. Those are Tasks 4 and 6. The three tests added here exercise only `num_nodes`, `num_edges`, and `out_degree`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/codec/webgraph.rs
git commit -m "feat(webgraph): WebGraphBuilder::build with gap+gamma encoding"
```

---

## Task 4: `successors` — streaming iteration without full decompression

**Files:**
- Modify: `crates/grafeo-core/src/codec/webgraph.rs`, `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to the existing `tests` module:

```rust
#[test]
fn successors_match_input_for_simple_graph() {
    let mut b = WebGraphBuilder::new(6);
    // Node 0 -> {1, 3, 5}, node 2 -> {0, 4}, node 5 -> {5} (self-loop).
    for (s, d) in [(0u64, 1), (0, 3), (0, 5), (2, 0), (2, 4), (5, 5)] {
        b.add_edge(s, d).unwrap();
    }
    let codec = b.build();
    assert_eq!(
        codec.successors(0).collect::<Vec<_>>(),
        vec![1, 3, 5]
    );
    assert_eq!(codec.successors(1).collect::<Vec<_>>(), Vec::<u64>::new());
    assert_eq!(codec.successors(2).collect::<Vec<_>>(), vec![0, 4]);
    assert_eq!(codec.successors(5).collect::<Vec<_>>(), vec![5]);
}

#[test]
fn successors_handles_dst_less_than_src() {
    // First gap is signed; verify dst < src round-trips correctly.
    let mut b = WebGraphBuilder::new(10);
    b.add_edge(7, 0).unwrap();
    b.add_edge(7, 1).unwrap();
    b.add_edge(7, 9).unwrap();
    let codec = b.build();
    assert_eq!(codec.successors(7).collect::<Vec<_>>(), vec![0, 1, 9]);
}

#[test]
fn successors_for_out_of_range_node_is_empty() {
    let codec = WebGraphBuilder::new(3).build();
    assert_eq!(codec.successors(99).collect::<Vec<_>>(), Vec::<u64>::new());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: FAIL — `no method named 'successors'`.

- [ ] **Step 3: Write the implementation**

Add to `webgraph.rs`, inside the existing `impl WebGraphCodec` block:

```rust
    /// Iterator over the successors of `node`, in ascending dst order.
    ///
    /// Reads one node's adjacency block in-place from the bit stream; no
    /// other node's data is touched.
    pub fn successors(&self, node: u64) -> SuccessorIter<'_> {
        if node >= self.num_nodes {
            return SuccessorIter::empty();
        }
        let start = self.offsets[node as usize];
        let mut reader = BitReader::new(&self.bits, self.bit_len);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        let degree_plus_one = reader.read_gamma().unwrap_or(1);
        let degree = degree_plus_one - 1;
        SuccessorIter {
            reader,
            node,
            remaining: degree,
            last_dst: None,
        }
    }
```

And after the `impl WebGraphCodec` block (still before the `mod tests`), add the iterator type:

```rust
/// Streaming iterator over a single node's successors.
pub struct SuccessorIter<'a> {
    reader: BitReader<'a>,
    node: u64,
    remaining: u64,
    /// `None` before the first successor, `Some(prev)` after.
    last_dst: Option<u64>,
}

impl<'a> SuccessorIter<'a> {
    fn empty() -> Self {
        Self {
            reader: BitReader::new(&[], 0),
            node: 0,
            remaining: 0,
            last_dst: None,
        }
    }
}

impl<'a> Iterator for SuccessorIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let dst = match self.last_dst {
            None => {
                let first_gap = self.reader.read_zigzag_gamma()?;
                // reason: encoder ensures (node as i64 + first_gap) >= 0 by
                // construction (dst was a valid u64 node id).
                #[allow(clippy::cast_sign_loss)]
                {
                    (self.node as i64 + first_gap) as u64
                }
            }
            Some(prev) => prev + self.reader.read_gamma()?,
        };
        self.last_dst = Some(dst);
        Some(dst)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: PASS — 8 tests.

- [ ] **Step 5: Add the public re-export to `codec/mod.rs`**

In `crates/grafeo-core/src/codec/mod.rs`, after the existing `pub use fsst::{...};` line (added by sub-plan 2c), add:

```rust
pub use webgraph::{SuccessorIter, WebGraphBuilder, WebGraphCodec, WebGraphError};
```

Run: `cargo build -p grafeo-core` — expected clean.

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/codec/webgraph.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(webgraph): SuccessorIter for in-place adjacency iteration"
```

---

## Task 5: Blob `to_bytes` / `from_bytes` — zero-copy layout contract

This mirrors sub-plans 2a Task 7 and 2c Task 6 exactly in structure: fixed header, blob-relative `u64` section offsets, naturally-aligned arrays, `from_bytes_shared(bytes::Bytes)` overload, CRC32 trailer.

**Files:**
- Modify: `crates/grafeo-core/src/codec/webgraph.rs`

Wire format (all little-endian):

```text
offset 0   "GWBG"               magic (4 bytes)
       4   version = 1          u8
       5   flags = 0            u8
       6   padding              u16
       8   num_nodes            u64
      16   num_edges            u64
      24   bit_len              u64   (bits actually used in the stream)
      32   offsets_offset       u64
      40   bits_offset          u64
      48   reserved             u64   (zero; aligns next section to 8)
      56   reserved             u64   (zero)
      64   offsets: (num_nodes + 1) × u64  + pad to 8
           bits: ceil(bit_len / 8) bytes  + pad to 4
           crc32                u32
```

- [ ] **Step 1: Write the failing test**

Add to the existing `tests` module:

```rust
#[test]
fn blob_round_trip_preserves_adjacency() {
    let mut b = WebGraphBuilder::new(20);
    // A varied small graph.
    let edges = [
        (0u64, 1), (0, 5), (0, 17),
        (1, 0), (1, 2),
        (3, 3),
        (5, 18), (5, 19),
        (10, 0), (10, 5), (10, 11), (10, 12),
        (19, 0),
    ];
    for &(s, d) in &edges {
        b.add_edge(s, d).unwrap();
    }
    let codec = b.build();
    let blob = codec.to_bytes();

    assert_eq!(&blob[0..4], b"GWBG");
    assert_eq!(blob[4], 1);

    let reopened = WebGraphCodec::from_bytes(&blob).expect("from_bytes");
    assert_eq!(reopened.num_nodes(), codec.num_nodes());
    assert_eq!(reopened.num_edges(), codec.num_edges());
    for u in 0..codec.num_nodes() {
        let original: Vec<u64> = codec.successors(u).collect();
        let reopened_succ: Vec<u64> = reopened.successors(u).collect();
        assert_eq!(reopened_succ, original, "successors of {u} mismatched");
    }
}

#[test]
fn blob_rejects_bad_magic_and_crc() {
    let mut b = WebGraphBuilder::new(3);
    b.add_edge(0, 1).unwrap();
    b.add_edge(1, 2).unwrap();
    let mut blob = b.build().to_bytes();

    let mut bad_magic = blob.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        WebGraphCodec::from_bytes(&bad_magic),
        Err(WebGraphError::BadMagic)
    ));

    let mid = blob.len() / 2;
    blob[mid] ^= 0xFF;
    assert!(matches!(
        WebGraphCodec::from_bytes(&blob),
        Err(WebGraphError::CrcMismatch { .. })
    ));
}

#[test]
fn blob_from_bytes_shared_round_trip() {
    let mut b = WebGraphBuilder::new(3);
    b.add_edge(0, 2).unwrap();
    let blob = bytes::Bytes::from(b.build().to_bytes());
    let reopened = WebGraphCodec::from_bytes_shared(blob).expect("from_bytes_shared");
    assert_eq!(reopened.successors(0).collect::<Vec<_>>(), vec![2]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: FAIL — `no method named 'to_bytes'`.

- [ ] **Step 3: Write the implementation**

Add to `webgraph.rs`, at module level (just before the `#[cfg(test)] mod tests` block):

```rust
/// Current WebGraph blob format version.
const BLOB_VERSION: u8 = 1;

/// Appends zero bytes until `buf.len()` is a multiple of `align`.
fn pad_to(buf: &mut Vec<u8>, align: usize) {
    while !buf.len().is_multiple_of(align) {
        buf.push(0);
    }
}

/// Reads a little-endian `u32` at `*pos`, advancing `*pos`.
fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, WebGraphError> {
    let end = *pos + 4;
    let slice = buf.get(*pos..end).ok_or(WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, WebGraphError> {
    let end = *pos + 8;
    let slice = buf.get(*pos..end).ok_or(WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

impl WebGraphCodec {
    /// Serializes to a self-describing, position-independent blob.
    ///
    /// Honours the Plan 2 zero-copy contract: a fixed 64-byte header with
    /// blob-relative `u64` section offsets, naturally-aligned arrays, and
    /// a trailing CRC32. See [`BLOB_VERSION`] for the version.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GWBG");
        buf.push(BLOB_VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // padding
        buf.extend_from_slice(&self.num_nodes.to_le_bytes());
        buf.extend_from_slice(&self.num_edges.to_le_bytes());
        buf.extend_from_slice(&self.bit_len.to_le_bytes());

        // Four u64s: two offsets + two reserved (zero), patched after assembly.
        let offsets_pos = buf.len(); // = 32
        buf.extend_from_slice(&[0u8; 32]);

        // Offsets array (num_nodes + 1 entries).
        // reason: section offsets fit u64
        #[allow(clippy::cast_possible_truncation)]
        let offsets_offset = buf.len() as u64;
        for &o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        #[allow(clippy::cast_possible_truncation)]
        let bits_offset = buf.len() as u64;
        buf.extend_from_slice(&self.bits);
        pad_to(&mut buf, 4);

        // Patch the two offsets.
        buf[offsets_pos..offsets_pos + 8]
            .copy_from_slice(&offsets_offset.to_le_bytes());
        buf[offsets_pos + 8..offsets_pos + 16]
            .copy_from_slice(&bits_offset.to_le_bytes());
        // The third and fourth u64s stay zero (reserved).

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on bad magic, unsupported version,
    /// truncation, CRC mismatch, or a malformed offsets array.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, WebGraphError> {
        if buf.len() < 8 {
            return Err(WebGraphError::Truncated {
                need: 8,
                have: buf.len(),
            });
        }
        if &buf[0..4] != b"GWBG" {
            return Err(WebGraphError::BadMagic);
        }
        if buf[4] != BLOB_VERSION {
            return Err(WebGraphError::BadVersion(buf[4]));
        }

        let body_end = buf.len() - 4;
        let stored = u32::from_le_bytes(buf[body_end..].try_into().expect("4 bytes"));
        let computed = crc32fast::hash(&buf[..body_end]);
        if stored != computed {
            return Err(WebGraphError::CrcMismatch { stored, computed });
        }

        let mut pos = 8;
        let num_nodes = read_u64(buf, &mut pos)?;
        let num_edges = read_u64(buf, &mut pos)?;
        let bit_len = read_u64(buf, &mut pos)?;
        let offsets_offset = read_u64(buf, &mut pos)? as usize;
        let bits_offset = read_u64(buf, &mut pos)? as usize;
        let _reserved1 = read_u64(buf, &mut pos)?;
        let _reserved2 = read_u64(buf, &mut pos)?;

        debug_assert_eq!(pos, 64, "header end at offset 64");

        // Offsets array.
        let n_offsets = num_nodes as usize + 1;
        let offsets_byte_end = offsets_offset + 8 * n_offsets;
        let offsets_bytes = buf
            .get(offsets_offset..offsets_byte_end)
            .ok_or(WebGraphError::Truncated {
                need: offsets_byte_end,
                have: buf.len(),
            })?;
        let mut offsets = Vec::with_capacity(n_offsets);
        for chunk in offsets_bytes.chunks_exact(8) {
            offsets.push(u64::from_le_bytes(chunk.try_into().expect("8 bytes")));
        }

        // Bit stream.
        let bytes_needed = bit_len.div_ceil(8) as usize;
        let bits_end = bits_offset + bytes_needed;
        let bits = buf
            .get(bits_offset..bits_end)
            .ok_or(WebGraphError::Truncated {
                need: bits_end,
                have: buf.len(),
            })?
            .to_vec();

        // Validate offsets: monotonic, [0] == 0, last == bit_len.
        if let Some(&first) = offsets.first() {
            if first != 0 {
                return Err(WebGraphError::BadOffset {
                    node: 0,
                    offset: first,
                    bit_len,
                });
            }
        }
        for (i, w) in offsets.windows(2).enumerate() {
            if w[1] < w[0] || w[1] > bit_len {
                return Err(WebGraphError::BadOffset {
                    node: i as u64,
                    offset: w[1],
                    bit_len,
                });
            }
        }
        if let Some(&last) = offsets.last() {
            if last != bit_len {
                return Err(WebGraphError::BadOffset {
                    node: num_nodes,
                    offset: last,
                    bit_len,
                });
            }
        }

        Ok(Self {
            num_nodes,
            num_edges,
            offsets,
            bits,
            bit_len,
        })
    }

    /// Opens a blob shared via [`bytes::Bytes`].
    ///
    /// Provided so callers can pass an mmap-backed `Bytes` region without
    /// copying it through `&[u8]` first. The current implementation parses
    /// into owned `Vec`s (one copy); sub-plan 2d adds a borrowing
    /// `WebGraphView` over the same wire format.
    ///
    /// # Errors
    /// Same as [`Self::from_bytes`].
    pub fn from_bytes_shared(blob: bytes::Bytes) -> Result<Self, WebGraphError> {
        Self::from_bytes(&blob)
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: PASS — 11 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/codec/webgraph.rs
git commit -m "feat(webgraph): zero-copy-layout blob serialization with crc"
```

---

## Task 6: Round-trip equivalence proptest

**Files:**
- Create: `crates/grafeo-core/tests/webgraph_round_trip.rs`

- [ ] **Step 1: Write the test file**

Create `crates/grafeo-core/tests/webgraph_round_trip.rs`:

```rust
//! Round-trip equivalence regression test for the WebGraph adjacency codec.
//!
//! The invariants:
//! 1. After build, every input edge `(src, dst)` (de-duplicated) appears
//!    in `codec.successors(src)`.
//! 2. After a blob round-trip, the same property holds.
//! 3. `out_degree(u)` equals `successors(u).count()` for every `u`.
//!
//! ```bash
//! cargo test -p grafeo-core --test webgraph_round_trip
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test webgraph_round_trip
//! ```

use grafeo_core::codec::{WebGraphBuilder, WebGraphCodec};
use proptest::prelude::*;
use std::collections::BTreeSet;

fn edge_strategy(num_nodes: u64) -> impl Strategy<Value = (u64, u64)> {
    (0u64..num_nodes, 0u64..num_nodes)
}

fn graph_strategy() -> impl Strategy<Value = (u64, Vec<(u64, u64)>)> {
    (1u64..=24).prop_flat_map(|n| {
        let edges = proptest::collection::vec(edge_strategy(n), 0..=80);
        (Just(n), edges)
    })
}

fn check_round_trip(num_nodes: u64, edges: &[(u64, u64)]) {
    // Build expected adjacency: per source, the set of distinct destinations
    // in ascending order.
    let mut expected: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); num_nodes as usize];
    for &(s, d) in edges {
        expected[s as usize].insert(d);
    }

    let mut builder = WebGraphBuilder::new(num_nodes);
    for &(s, d) in edges {
        builder.add_edge(s, d).unwrap();
    }
    let codec = builder.build();

    // Invariant: each node's successors match the de-duped sorted set.
    for u in 0..num_nodes {
        let got: Vec<u64> = codec.successors(u).collect();
        let want: Vec<u64> = expected[u as usize].iter().copied().collect();
        assert_eq!(got, want, "successors of {u} mismatched");
        assert_eq!(codec.out_degree(u), got.len() as u64);
    }

    // Total edges = sum of distinct successors.
    let total: u64 = expected.iter().map(|s| s.len() as u64).sum();
    assert_eq!(codec.num_edges(), total);

    // Blob round-trip.
    let blob = codec.to_bytes();
    let reopened = WebGraphCodec::from_bytes(&blob).expect("from_bytes");
    for u in 0..num_nodes {
        let got: Vec<u64> = reopened.successors(u).collect();
        let want: Vec<u64> = expected[u as usize].iter().copied().collect();
        assert_eq!(got, want, "reopened successors of {u} mismatched");
    }
    assert_eq!(reopened.num_edges(), total);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every input edge survives encoding and blob round-trip; per-node
    /// successor lists match the de-duped, sorted input.
    #[test]
    fn webgraph_round_trip_arbitrary_small_graphs(
        (num_nodes, edges) in graph_strategy()
    ) {
        check_round_trip(num_nodes, &edges);
    }
}

// ── Fixed regression seeds ───────────────────────────────────────

#[test]
fn webgraph_round_trip_empty_graph() {
    check_round_trip(0, &[]);
}

#[test]
fn webgraph_round_trip_isolated_nodes() {
    check_round_trip(7, &[]);
}

#[test]
fn webgraph_round_trip_self_loops() {
    let edges = [(0u64, 0), (3, 3), (5, 5)];
    check_round_trip(6, &edges);
}

#[test]
fn webgraph_round_trip_dst_below_src() {
    // First-gap is signed; exercise dst < src.
    let edges = [(9u64, 0), (9, 1), (9, 9)];
    check_round_trip(10, &edges);
}

#[test]
fn webgraph_round_trip_dense_node() {
    // One node with edges to every other.
    let edges: Vec<(u64, u64)> = (0u64..20).map(|d| (0u64, d)).collect();
    check_round_trip(20, &edges);
}

#[test]
fn webgraph_round_trip_duplicated_edges() {
    let edges = [(0u64, 1), (0, 1), (0, 1), (0, 2), (1, 0), (1, 0)];
    check_round_trip(3, &edges);
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p grafeo-core --test webgraph_round_trip`
Expected: PASS — 1 proptest with 128 cases + 6 fixed-seed tests = 7 PASS.

Also run `PROPTEST_CASES=512 cargo test -p grafeo-core --test webgraph_round_trip` and confirm. Round-trip equivalence is an absolute property — any failure indicates a real bug. STOP and report if it fails.

- [ ] **Step 3: Commit**

```bash
git add crates/grafeo-core/tests/webgraph_round_trip.rs
git commit -m "test(webgraph): round-trip equivalence proptest"
```

---

## Task 7: WASM bindings — `WebGraphCodec`

**Files:**
- Modify: `crates/bindings/wasm/Cargo.toml`
- Modify: `crates/bindings/wasm/src/lib.rs`
- Modify: `crates/bindings/wasm/src/codecs.rs`
- Modify: `crates/bindings/wasm/tests/web.rs`

- [ ] **Step 1: Add the `webgraph-codec` feature**

In `crates/bindings/wasm/Cargo.toml`, after the existing `fsst-codec = ["dep:grafeo-core"]` line, add:

```toml
webgraph-codec = ["dep:grafeo-core"]
```

- [ ] **Step 2: Extend the module gate**

In `crates/bindings/wasm/src/lib.rs`, find the existing line:

```rust
#[cfg(any(feature = "rabitq-codec", feature = "fsst-codec"))]
pub mod codecs;
```

Change it to:

```rust
#[cfg(any(feature = "rabitq-codec", feature = "fsst-codec", feature = "webgraph-codec"))]
pub mod codecs;
```

- [ ] **Step 3: Write the failing test**

Append to `crates/bindings/wasm/tests/web.rs`:

```rust
#[cfg(feature = "webgraph-codec")]
#[wasm_bindgen_test]
fn webgraph_codec_encode_open_successors_round_trip() {
    use grafeo_wasm::codecs::WebGraphCodec;

    // Star graph: node 0 connects to 1, 2, 3; node 5 has a self-loop.
    let srcs: Vec<u32> = vec![0, 0, 0, 5];
    let dsts: Vec<u32> = vec![1, 2, 3, 5];
    let blob = WebGraphCodec::encode(6, &srcs, &dsts).expect("encode");
    assert_eq!(&blob[0..4], b"GWBG");

    let codec = WebGraphCodec::open(&blob).expect("open");
    assert_eq!(codec.num_nodes(), 6);
    assert_eq!(codec.num_edges(), 4);
    assert_eq!(codec.successors(0), vec![1u32, 2, 3]);
    assert_eq!(codec.successors(1), Vec::<u32>::new());
    assert_eq!(codec.successors(5), vec![5u32]);
}
```

- [ ] **Step 4: Run to verify it fails**

```bash
wasm-pack test --node crates/bindings/wasm --features webgraph-codec -- webgraph
```
Expected: FAIL — `cannot find type 'WebGraphCodec'` in `grafeo_wasm::codecs`.

If wasm-pack is unavailable: `cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features webgraph-codec` should fail.

- [ ] **Step 5: Write the implementation**

Append to `crates/bindings/wasm/src/codecs.rs`:

```rust
/// A JS-facing handle to a WebGraph adjacency codec.
#[cfg(feature = "webgraph-codec")]
#[wasm_bindgen]
pub struct WebGraphCodec {
    inner: grafeo_core::codec::WebGraphCodec,
}

#[cfg(feature = "webgraph-codec")]
#[wasm_bindgen]
impl WebGraphCodec {
    /// Encodes an edge list into a compressed adjacency blob.
    ///
    /// `srcs` and `dsts` are parallel arrays; entry `i` is the edge
    /// `srcs[i] -> dsts[i]`. All ids must be `< num_nodes`. Duplicates are
    /// de-duplicated by the underlying codec. Returns the blob bytes.
    ///
    /// # Errors
    /// Returns a `JsError` if `srcs.length != dsts.length` or if any id
    /// is `>= num_nodes`.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode(num_nodes: u32, srcs: &[u32], dsts: &[u32]) -> Result<Vec<u8>, JsError> {
        if srcs.len() != dsts.len() {
            return Err(JsError::new("srcs.length must equal dsts.length"));
        }
        let mut builder = grafeo_core::codec::WebGraphBuilder::new(u64::from(num_nodes));
        for (&s, &d) in srcs.iter().zip(dsts) {
            builder
                .add_edge(u64::from(s), u64::from(d))
                .map_err(|e| JsError::new(&e.to_string()))?;
        }
        Ok(builder.build().to_bytes())
    }

    /// Opens a blob produced by [`WebGraphCodec::encode`] for querying.
    ///
    /// # Errors
    /// Returns a `JsError` if the blob is malformed (bad magic, version,
    /// truncation, or CRC mismatch).
    #[wasm_bindgen(js_name = "open")]
    pub fn open(blob: &[u8]) -> Result<WebGraphCodec, JsError> {
        let inner = grafeo_core::codec::WebGraphCodec::from_bytes(blob)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the successors of `node` as a `Uint32Array`.
    #[wasm_bindgen(js_name = "successors")]
    #[must_use]
    pub fn successors(&self, node: u32) -> Vec<u32> {
        // reason: snapshot node ids fit u32 for the JS surface
        #[allow(clippy::cast_possible_truncation)]
        self.inner
            .successors(u64::from(node))
            .map(|d| d as u32)
            .collect()
    }

    /// Out-degree of `node`.
    #[wasm_bindgen(js_name = "outDegree")]
    #[must_use]
    pub fn out_degree(&self, node: u32) -> u32 {
        // reason: degree fits u32 for any practical snapshot
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.out_degree(u64::from(node)) as u32
        }
    }

    /// Number of nodes.
    #[wasm_bindgen(js_name = "numNodes")]
    #[must_use]
    pub fn num_nodes(&self) -> u32 {
        // reason: snapshot node count fits u32
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.num_nodes() as u32
        }
    }

    /// Number of edges.
    #[wasm_bindgen(js_name = "numEdges")]
    #[must_use]
    pub fn num_edges(&self) -> u32 {
        // reason: snapshot edge count fits u32
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.num_edges() as u32
        }
    }
}
```

- [ ] **Step 6: Run to verify it passes**

```bash
wasm-pack test --node crates/bindings/wasm --features webgraph-codec -- webgraph
```
Expected: PASS.

Confirm other codecs still work with combined features:
```bash
wasm-pack test --node crates/bindings/wasm --features "rabitq-codec fsst-codec webgraph-codec"
```

If `wasm-pack` is unavailable, fall back to:
```bash
cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features webgraph-codec
cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features "rabitq-codec fsst-codec webgraph-codec"
```

- [ ] **Step 7: Commit**

```bash
git add crates/bindings/wasm/Cargo.toml crates/bindings/wasm/src/lib.rs crates/bindings/wasm/src/codecs.rs crates/bindings/wasm/tests/web.rs
git commit -m "feat(wasm): WebGraphCodec binding for encode/open/successors"
```

---

## Task 8: CHANGELOG and full verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the CHANGELOG entry**

In `CHANGELOG.md`, under the same `## [Unreleased] / ### Added` section sub-plans 2a and 2c populated, append:

```markdown
- `codec::webgraph` — Static compressed adjacency codec (gap coding + Elias gamma codes + per-node bit-offset index) with streaming successor iteration in-place, and a `WebGraphCodec` WASM binding (behind the `webgraph-codec` feature on the wasm crate). The `webgraph` crate from `crates.io` was evaluated and rejected for `wasm32-unknown-unknown` due to incompatible transitive dependencies (`mmap-rs`, `rayon-core`, `getrandom v0.3`).
```

- [ ] **Step 2: Run the full verification suite**

Run each, confirming output before claiming completion (`superpowers:verification-before-completion`):

```bash
cargo test -p grafeo-core --lib codec::bitstream
cargo test -p grafeo-core --lib codec::webgraph
cargo test -p grafeo-core --test webgraph_round_trip
PROPTEST_CASES=512 cargo test -p grafeo-core --test webgraph_round_trip
cargo clippy -p grafeo-core --all-targets -- -D warnings
cargo build -p grafeo-core --target wasm32-unknown-unknown
cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features "rabitq-codec fsst-codec webgraph-codec"
```

If `wasm-pack` is in PATH, also:
```bash
wasm-pack test --node crates/bindings/wasm --features "rabitq-codec fsst-codec webgraph-codec"
```
(The 13 pre-existing failures noted in sub-plans 2a/2c remain pre-existing; do not try to fix them.)

If clippy finds fixable nits in our task's files (`bitstream.rs`, `webgraph.rs`, `codecs.rs` additions, `webgraph_round_trip.rs`), fix them in the same commit. Do NOT touch other crates.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note WebGraph adjacency codec"
```

(If clippy fixes were needed: `docs(changelog): note WebGraph adjacency codec + clippy nits`.)

---

## Self-Review (completed by the plan author)

**Spec coverage** — every Plan 2b requirement maps to a task:
- Static compressed adjacency via gap coding → Task 3 (`WebGraphBuilder::build`).
- Gamma codes (zeta-1, the universal-code substitute for zeta-3) → Task 1 (`BitWriter::write_gamma`, `BitReader::read_gamma`). Zeta-3 is documented as a follow-up.
- Per-node bit-offset index for O(1) seek → Task 3 (`WebGraphCodec::offsets`).
- Streaming traversal without full decompression → Task 4 (`SuccessorIter` reads one node's block from the bit stream).
- Round-trip equivalence proptest → Task 6.
- WASM bindings → Task 7.
- Sibling to (not modification of) `ChunkedAdjacency` → new module `codec/webgraph.rs`; `index/adjacency.rs` untouched.
- Zero-copy contract (fixed header, `u64` offsets, `from_bytes_shared`, CRC) → Task 5.
- WebGraph crate spike → done inline before plan was written; result documented in Tech Stack and CHANGELOG.

**Type consistency** — `BitWriter`/`BitReader` (`new`, `write_bit`, `write_bits`, `write_gamma`, `write_zigzag_gamma`, `into_bytes`, `bit_len`, `seek`, `bit_pos`, `read_bit`, `read_bits`, `read_gamma`, `read_zigzag_gamma`), `WebGraphBuilder` (`new`, `add_edge`, `build`, `num_nodes`, `edge_count`), `WebGraphCodec` (`num_nodes`, `num_edges`, `out_degree`, `successors`, `to_bytes`, `from_bytes`, `from_bytes_shared`; `pub(crate)` fields `num_nodes`, `num_edges`, `offsets`, `bits`, `bit_len`), `SuccessorIter` (`empty`, `next`, `size_hint`), `WebGraphError` variants (`Truncated`, `BadMagic`, `BadVersion`, `CrcMismatch`, `EdgeOutOfRange`, `BadOffset`, `UnexpectedEnd`) — all consistent. The `GAMMA_MAX` and `BLOB_VERSION` constants are referenced where defined.

**Placeholder scan** — every step shows complete code; no TBD/TODO; no "similar to Task N" placeholders. The "WebGraph crate spike" decision was made inline (build failed on getrandom/mmap-rs); the plan documents the outcome rather than re-running the spike.

**Known follow-ups (out of scope for 2b):**
- **Referentiation.** Encode a node's adjacency as a bitmask over a nearby reference node's list plus extras. The WebGraph paper's biggest single contribution to compression on web-like graphs; less impactful on social graphs. Add only if the integration benchmark shows a meaningful win for Nota's data.
- **Zeta-3 codes.** ~10% smaller than gamma on power-law gap distributions. Behind the same `BitWriter`/`BitReader` interface — a drop-in replacement for `write_gamma`/`read_gamma` if needed.
- **In-place borrowing view.** Sub-plan 2d adds a `WebGraphView` over `bytes::Bytes` that doesn't copy the bit stream. The `from_bytes_shared` stub is the API hook.
- **Backward adjacency.** This codec encodes outgoing edges only. A second pass over `(dst, src)` builds a reverse-edge `WebGraphCodec` for incoming traversal. Independent — can be added later.
- **`ColumnCodec` integration.** Like the FSST `ColumnCodec::Fsst` addition in 2c, a `RelTableSection` could hold a `WebGraphCodec` for compact-store edge tables. Belongs in 2d when the section wire format is extended.
