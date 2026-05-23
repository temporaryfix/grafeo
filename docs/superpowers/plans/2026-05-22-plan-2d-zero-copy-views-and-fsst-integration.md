# Sub-plan 2d — Zero-Copy Views and FSST Column Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real borrowing implementations of the `from_bytes_shared(bytes::Bytes)` contract for all three Plan 2 codecs (RaBitQ, FSST, WebGraph), complete the FSST `ColumnCodec` serialization that sub-plan 2c deferred, and benchmark the View vs Owned query latency to inform whether the WASM bindings should be migrated.

**Architecture:** Each codec gets a new `*View` type that holds a single `bytes::Bytes` plus parsed header offsets and a small set of owned auxiliary data (a `ScalarQuantizer` for RaBitQ, a `SymbolTable` for FSST, nothing for WebGraph). Query methods slice the `Bytes` on demand for codes/int8 (RaBitQ), compressed stream/offsets (FSST), and bit stream (WebGraph), avoiding the per-element heap allocation that the owned `from_bytes` does. Parity proptests confirm every query returns identical results from `View` and owned codecs. Separately, the FSST `ColumnCodec::Fsst` serialization is completed: a new `ColumnType::FsstString` variant, full `write_to`/`write_to_v2`/`write_to_v3` paths in `column.rs`, and corresponding `read_from*` cases. A criterion benchmark compares `View` vs owned query latency.

**Tech Stack:** Rust 2024, `grafeo-core` (`codec/`, `graph/compact/`), `bytes`, `criterion`, `proptest`. No new runtime dependencies.

---

## File Structure

- **Modify** `crates/grafeo-core/src/index/vector/rabitq.rs` — add `RabitqView` type + `from_bytes_shared` real implementation.
- **Modify** `crates/grafeo-core/src/codec/fsst.rs` — add `FsstView` type + real `from_bytes_shared`.
- **Modify** `crates/grafeo-core/src/codec/webgraph.rs` — add `WebGraphView` type + real `from_bytes_shared`.
- **Create** `crates/grafeo-core/tests/codec_view_parity.rs` — proptest comparing View vs Owned results.
- **Modify** `crates/grafeo-core/src/graph/compact/schema.rs` — add `ColumnType::FsstString`.
- **Modify** `crates/grafeo-core/src/graph/compact/builder.rs` — replace the `FIXME(plan-2d)` stub with `ColumnType::FsstString`.
- **Modify** `crates/grafeo-core/src/graph/compact/section.rs` — replace the `FIXME(plan-2d)` stub; add the read_from cases for FSST.
- **Modify** `crates/grafeo-core/src/graph/compact/column.rs` — replace `unimplemented!()` in `emit_blocked_codec`; add `Self::Fsst` arms to `write_to`/`write_to_v2`/`read_from*`.
- **Create** `crates/grafeo-core/benches/codec_views.rs` — criterion bench, View vs Owned for the three codecs.
- **Modify** `crates/grafeo-core/Cargo.toml` — register the new bench.
- **Modify** `CHANGELOG.md` — note both deliverables.

---

## Task 1: `RabitqView` — borrowing two-stage vector reader

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`
- Modify: `crates/grafeo-core/src/index/vector/mod.rs`

- [ ] **Step 1: Write the failing test**

Append to the existing `tests` module in `rabitq.rs`:

```rust
#[test]
fn view_search_matches_owned_search() {
    use grafeo_common::types::NodeId;
    let dim = 32;
    let mut rng = SplitMix64::new(2025);
    let vectors: Vec<(NodeId, Vec<f32>)> = (0..40)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_gaussian()).collect();
            (NodeId::new(i + 1), v)
        })
        .collect();
    let owned = TwoStageVectorIndex::build(&vectors, dim, 7);
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = RabitqView::open(blob).expect("open");

    assert_eq!(view.len(), owned.len());

    let query = vectors[3].1.clone();
    let owned_hits = owned.search(&query, 10, 8);
    let view_hits = view.search(&query, 10, 8);
    assert_eq!(view_hits, owned_hits);
}

#[test]
fn view_rejects_bad_magic() {
    use grafeo_common::types::NodeId;
    let owned = TwoStageVectorIndex::build(
        &[(NodeId::new(1), vec![1.0f32; 8])],
        8,
        1,
    );
    let mut bad = owned.to_bytes();
    bad[0] = b'X';
    assert!(matches!(
        RabitqView::open(bytes::Bytes::from(bad)),
        Err(RabitqError::BadMagic)
    ));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `cannot find type 'RabitqView'`.

- [ ] **Step 3: Write the implementation**

Add to `rabitq.rs`, after the `impl TwoStageVectorIndex` block that contains `to_bytes`/`from_bytes` (and before the `#[cfg(test)] mod tests` block):

```rust
/// A borrowing reader over a [`TwoStageVectorIndex`] blob.
///
/// Holds a single `bytes::Bytes` and a parsed header; the int8 rerank
/// codes and the RaBitQ bit codes are sliced from the held bytes on
/// demand for each query, avoiding the per-vector heap allocations that
/// [`TwoStageVectorIndex::from_bytes`] does.
///
/// Search results are bit-identical to the owned path; see the
/// `view_search_matches_owned_search` test and the
/// `codec_view_parity` integration proptest.
#[derive(Debug, Clone)]
pub struct RabitqView {
    blob: bytes::Bytes,
    dim: usize,
    count: usize,
    words: usize,
    /// Decoded once at open time (small).
    scalar: ScalarQuantizer,
    /// Decoded once at open time (`dim*dim` f32s — typically ≤ 256 KB).
    rotation: Rotation,
    /// Owned to avoid alignment concerns on `[u64]` borrowed from Bytes.
    /// Decoded once at open time.
    rotation_quantizer: RabitqQuantizer,
    /// Owned `Vec<NodeId>` decoded at open time (8 bytes per id; small
    /// relative to int8/codes which stay in the borrowed `blob`).
    ids: Vec<NodeId>,
    /// Byte offset into `blob` where the codes section starts.
    codes_offset: usize,
    /// Byte stride for one code: `words*8 + 8` bytes (bits + dot_oo + norm).
    code_stride: usize,
    /// Byte offset into `blob` where the int8 section starts.
    int8_offset: usize,
}

impl RabitqView {
    /// Opens a blob produced by [`TwoStageVectorIndex::to_bytes`].
    ///
    /// The blob's `bytes::Bytes` is held; per-query reads slice it
    /// directly. The scalar quantizer, rotation matrix, and id list are
    /// decoded once into owned storage (each is small).
    ///
    /// # Errors
    /// Returns [`RabitqError`] on a malformed blob — same conditions as
    /// [`TwoStageVectorIndex::from_bytes`].
    pub fn open(blob: bytes::Bytes) -> Result<Self, RabitqError> {
        let buf = blob.as_ref();
        if buf.len() < 8 {
            return Err(RabitqError::Truncated {
                need: 8,
                have: buf.len(),
            });
        }
        if &buf[0..4] != b"GRBQ" {
            return Err(RabitqError::BadMagic);
        }
        if buf[4] != BLOB_VERSION {
            return Err(RabitqError::BadVersion(buf[4]));
        }
        let body_end = buf.len() - 4;
        let stored = u32::from_le_bytes(buf[body_end..].try_into().expect("4 bytes"));
        let computed = crc32fast::hash(&buf[..body_end]);
        if stored != computed {
            return Err(RabitqError::CrcMismatch { stored, computed });
        }

        let mut pos = 8;
        let dim = read_u32(buf, &mut pos)? as usize;
        let count = read_u32(buf, &mut pos)? as usize;
        let seed = read_u64(buf, &mut pos)?;
        let words = read_u32(buf, &mut pos)? as usize;
        let quant_len = read_u32(buf, &mut pos)? as usize;
        // Skip the four offset slots written by `to_bytes`.
        let rotation_offset = read_u64(buf, &mut pos)? as usize;
        let ids_offset = read_u64(buf, &mut pos)? as usize;
        let codes_offset = read_u64(buf, &mut pos)? as usize;
        let int8_offset = read_u64(buf, &mut pos)? as usize;

        // ScalarQuantizer — decoded once into owned storage.
        let quant_slice = buf.get(pos..pos + quant_len).ok_or(RabitqError::Truncated {
            need: pos + quant_len,
            have: buf.len(),
        })?;
        let (scalar, _): (ScalarQuantizer, usize) =
            bincode::serde::decode_from_slice(quant_slice, bincode::config::standard())
                .map_err(|e| RabitqError::Quantizer(e.to_string()))?;

        // Rotation matrix — decoded once into owned storage.
        let mut matrix = Vec::with_capacity(dim * dim);
        let mut rpos = rotation_offset;
        for _ in 0..dim * dim {
            matrix.push(read_f32(buf, &mut rpos)?);
        }
        let rotation = Rotation::from_matrix(dim, matrix);
        let rotation_quantizer =
            RabitqQuantizer::from_parts(dim, seed, rotation.clone());

        // ids — decoded once.
        let mut ids = Vec::with_capacity(count);
        let mut ipos = ids_offset;
        for _ in 0..count {
            ids.push(NodeId::new(read_u64(buf, &mut ipos)?));
        }

        let code_stride = words * 8 + 8;
        // Validate codes/int8 sections fit in the blob.
        let codes_end = codes_offset
            .checked_add(count.saturating_mul(code_stride))
            .ok_or(RabitqError::Truncated {
                need: usize::MAX,
                have: buf.len(),
            })?;
        if codes_end > buf.len() {
            return Err(RabitqError::Truncated {
                need: codes_end,
                have: buf.len(),
            });
        }
        let int8_end = int8_offset
            .checked_add(count.saturating_mul(dim))
            .ok_or(RabitqError::Truncated {
                need: usize::MAX,
                have: buf.len(),
            })?;
        if int8_end > buf.len() {
            return Err(RabitqError::Truncated {
                need: int8_end,
                have: buf.len(),
            });
        }

        Ok(Self {
            blob,
            dim,
            count,
            words,
            scalar,
            rotation,
            rotation_quantizer,
            ids,
            codes_offset,
            code_stride,
            int8_offset,
        })
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reads the bits of code at row `row` directly from `self.blob`,
    /// returning a per-call `Vec<u64>` (one allocation of `words * 8`
    /// bytes per coarse-search candidate).
    fn read_code_bits(&self, row: usize) -> Vec<u64> {
        let start = self.codes_offset + row * self.code_stride;
        let buf = self.blob.as_ref();
        (0..self.words)
            .map(|w| {
                let pos = start + w * 8;
                u64::from_le_bytes(buf[pos..pos + 8].try_into().expect("8 bytes"))
            })
            .collect()
    }

    /// Reads the `dot_oo` and `norm` factors for the code at row `row`.
    fn read_code_factors(&self, row: usize) -> (f32, f32) {
        let start = self.codes_offset + row * self.code_stride + self.words * 8;
        let buf = self.blob.as_ref();
        let dot_oo = f32::from_le_bytes(buf[start..start + 4].try_into().expect("4 bytes"));
        let norm = f32::from_le_bytes(buf[start + 4..start + 8].try_into().expect("4 bytes"));
        (dot_oo, norm)
    }

    /// Returns the int8 code slice for row `row` (borrowed from `self.blob`).
    fn int8_row(&self, row: usize) -> &[u8] {
        let start = self.int8_offset + row * self.dim;
        &self.blob.as_ref()[start..start + self.dim]
    }

    /// Searches for the `k` nearest neighbours of `query`.
    ///
    /// Identical results to [`TwoStageVectorIndex::search`] (see the
    /// parity proptest).
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize, rerank_factor: usize) -> Vec<(NodeId, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let candidate_n = k
            .saturating_mul(rerank_factor.max(1))
            .min(self.count);

        // Coarse pass: encode the query, then iterate stored codes.
        let q = self.rotation_quantizer.encode_query(query);
        let mut scored: Vec<(usize, f32)> = (0..self.count)
            .map(|row| {
                let bits = self.read_code_bits(row);
                let (dot_oo, norm) = self.read_code_factors(row);
                let code = RabitqCode { bits, dot_oo, norm };
                let est = self.rotation_quantizer.estimate_distance(&q, &code);
                (row, est)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(candidate_n);

        // Rerank by int8.
        let mut reranked: Vec<(NodeId, f32)> = scored
            .iter()
            .map(|&(row, _)| {
                let dist = self.scalar.asymmetric_distance(query, self.int8_row(row));
                (self.ids[row], dist)
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }
}
```

The implementation uses the existing `RabitqCode` struct as a stack-friendly scratch type per candidate. The `read_code_bits` allocation is `words * 8` bytes per candidate (typically 32 B for 256-dim) — much smaller than the owned path's full `Vec<RabitqCode>`. Future optimization could lift this allocation by lifting `estimate_distance` to operate on a raw bit slice.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 17 tests (15 from before + 2 new).

- [ ] **Step 5: Add the public re-export**

In `crates/grafeo-core/src/index/vector/mod.rs`, extend the existing `pub use rabitq::{...};` line to include `RabitqView`:

```rust
pub use rabitq::{
    RabitqCode, RabitqError, RabitqIndex, RabitqQuantizer, RabitqQuery, RabitqView,
    TwoStageVectorIndex,
};
```

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs crates/grafeo-core/src/index/vector/mod.rs
git commit -m "feat(rabitq): RabitqView for zero-copy borrowing search"
```

---

## Task 2: `FsstView` — borrowing string codec reader

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs`
- Modify: `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Append to the existing `tests` module in `fsst.rs`:

```rust
#[test]
fn view_get_matches_owned_get() {
    let strings: Vec<&[u8]> = vec![
        b"alpha",
        b"beta gamma",
        b"",
        b"the quick brown fox",
    ];
    let owned = FsstCodec::build(&strings);
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = FsstView::open(blob).expect("open");

    assert_eq!(view.len(), owned.len());
    for i in 0..strings.len() {
        let owned_s = owned.get(i).expect("owned").expect("decode");
        let view_s = view.get(i).expect("view").expect("decode");
        assert_eq!(view_s, owned_s, "string {i} mismatched");
    }
    assert!(view.get(strings.len()).expect("view").is_none());
}

#[test]
fn view_rejects_bad_magic() {
    let owned = FsstCodec::build(&[b"hello"]);
    let mut bad = owned.to_bytes();
    bad[0] = b'X';
    assert!(matches!(
        FsstView::open(bytes::Bytes::from(bad)),
        Err(FsstError::BadMagic)
    ));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: FAIL — `cannot find type 'FsstView'`.

- [ ] **Step 3: Write the implementation**

Add to `fsst.rs`, after the `impl FsstCodec` block containing `to_bytes`/`from_bytes`/`from_bytes_shared` (and before the `#[cfg(test)] mod tests`):

```rust
/// A borrowing reader over an [`FsstCodec`] blob.
///
/// Holds a single `bytes::Bytes` and the parsed offsets; per-string
/// decode slices the compressed stream directly from the blob, avoiding
/// the heap-copy that [`FsstCodec::from_bytes`] does on the `compressed`
/// vector. The symbol table is decoded once at open time into an owned
/// [`SymbolTable`] (~2 KB).
#[derive(Debug, Clone)]
pub struct FsstView {
    blob: bytes::Bytes,
    /// Decoded once at open time (~2 KB).
    table: SymbolTable,
    /// Number of strings.
    count: usize,
    /// Byte offset into `blob` where the offsets array starts.
    offsets_offset: usize,
    /// Byte offset into `blob` where the compressed stream starts.
    compressed_offset: usize,
    /// Length of the compressed stream in bytes.
    compressed_len: usize,
}

impl FsstView {
    /// Opens a blob produced by [`FsstCodec::to_bytes`].
    ///
    /// # Errors
    /// Returns [`FsstError`] on a malformed blob — same conditions as
    /// [`FsstCodec::from_bytes`].
    pub fn open(blob: bytes::Bytes) -> Result<Self, FsstError> {
        let buf = blob.as_ref();
        if buf.len() < 8 {
            return Err(FsstError::Truncated {
                need: 8,
                have: buf.len(),
            });
        }
        if &buf[0..4] != b"GFST" {
            return Err(FsstError::BadMagic);
        }
        if buf[4] != BLOB_VERSION {
            return Err(FsstError::BadVersion(buf[4]));
        }
        let body_end = buf.len() - 4;
        let stored = u32::from_le_bytes(buf[body_end..].try_into().expect("4 bytes"));
        let computed = crc32fast::hash(&buf[..body_end]);
        if stored != computed {
            return Err(FsstError::CrcMismatch { stored, computed });
        }

        let mut pos = 8;
        let count = read_u32(buf, &mut pos)? as usize;
        let compressed_len = read_u32(buf, &mut pos)? as usize;
        let table_offset = read_u64(buf, &mut pos)? as usize;
        let offsets_offset = read_u64(buf, &mut pos)? as usize;
        let compressed_offset = read_u64(buf, &mut pos)? as usize;
        let _reserved = read_u64(buf, &mut pos)?;

        // Symbol table — decoded once into owned SymbolTable.
        let lengths_end = table_offset + 256;
        let lengths_slice = buf.get(table_offset..lengths_end).ok_or(FsstError::Truncated {
            need: lengths_end,
            have: buf.len(),
        })?;
        let bodies_end = lengths_end + 256 * MAX_SYMBOL_LEN;
        let bodies_slice = buf.get(lengths_end..bodies_end).ok_or(FsstError::Truncated {
            need: bodies_end,
            have: buf.len(),
        })?;
        let mut table = SymbolTable::default();
        for (code, &len_byte) in lengths_slice.iter().enumerate() {
            // reason: code 0..=255 fits u8 by construction
            #[allow(clippy::cast_possible_truncation)]
            let code_u8 = code as u8;
            if code_u8 == ESCAPE {
                if len_byte != 0 {
                    return Err(FsstError::BadSymbolLength {
                        code: code_u8,
                        length: len_byte,
                    });
                }
                continue;
            }
            if len_byte == 0 {
                continue;
            }
            if (len_byte as usize) > MAX_SYMBOL_LEN {
                return Err(FsstError::BadSymbolLength {
                    code: code_u8,
                    length: len_byte,
                });
            }
            let body_start = code * MAX_SYMBOL_LEN;
            let sym = &bodies_slice[body_start..body_start + len_byte as usize];
            table.set(code_u8, sym);
        }

        // Validate the offsets and compressed sections fit.
        let offsets_byte_end = offsets_offset + 4 * (count + 1);
        if offsets_byte_end > buf.len() {
            return Err(FsstError::Truncated {
                need: offsets_byte_end,
                have: buf.len(),
            });
        }
        let compressed_end = compressed_offset + compressed_len;
        if compressed_end > buf.len() {
            return Err(FsstError::Truncated {
                need: compressed_end,
                have: buf.len(),
            });
        }
        // Validate offsets[0] == 0.
        if count > 0 {
            let first = u32::from_le_bytes(
                buf[offsets_offset..offsets_offset + 4]
                    .try_into()
                    .expect("4 bytes"),
            );
            if first != 0 {
                return Err(FsstError::BadOffset {
                    index: 0,
                    offset: u64::from(first),
                    len: u64::from(compressed_len as u32),
                });
            }
        }

        Ok(Self {
            blob,
            table,
            count,
            offsets_offset,
            compressed_offset,
            compressed_len,
        })
    }

    /// Number of strings stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reads `offsets[index]` from the blob.
    fn read_offset(&self, index: usize) -> u32 {
        let pos = self.offsets_offset + index * 4;
        u32::from_le_bytes(
            self.blob.as_ref()[pos..pos + 4]
                .try_into()
                .expect("4 bytes"),
        )
    }

    /// Decodes string `index`, or `Ok(None)` if out of bounds.
    ///
    /// # Errors
    /// Returns [`FsstError`] if the stored bytes are malformed.
    pub fn get(&self, index: usize) -> Result<Option<Vec<u8>>, FsstError> {
        if index >= self.count {
            return Ok(None);
        }
        let start = self.read_offset(index) as usize;
        let end = self.read_offset(index + 1) as usize;
        if end > self.compressed_len || start > end {
            return Err(FsstError::BadOffset {
                index,
                offset: u64::from(self.read_offset(index)),
                len: u64::from(self.compressed_len as u32),
            });
        }
        let compressed = &self.blob.as_ref()
            [self.compressed_offset + start..self.compressed_offset + end];
        Ok(Some(self.table.decode(compressed)?))
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — 26 tests (24 from before + 2 new).

- [ ] **Step 5: Add the public re-export**

In `crates/grafeo-core/src/codec/mod.rs`, extend the existing `pub use fsst::{...};` line:

```rust
pub use fsst::{FsstCodec, FsstError, FsstView, SymbolTable};
```

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(fsst): FsstView for zero-copy borrowing decode"
```

---

## Task 3: `WebGraphView` — borrowing adjacency reader

**Files:**
- Modify: `crates/grafeo-core/src/codec/webgraph.rs`
- Modify: `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Append to the existing `tests` module in `webgraph.rs`:

```rust
#[test]
fn view_successors_matches_owned() {
    let mut b = WebGraphBuilder::new(15);
    let edges = [
        (0u64, 1), (0, 5), (0, 14),
        (3, 0), (3, 3),
        (10, 7), (10, 8), (10, 9),
        (14, 0),
    ];
    for &(s, d) in &edges {
        b.add_edge(s, d).unwrap();
    }
    let owned = b.build();
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = WebGraphView::open(blob).expect("open");

    assert_eq!(view.num_nodes(), owned.num_nodes());
    assert_eq!(view.num_edges(), owned.num_edges());
    for u in 0..owned.num_nodes() {
        let owned_succ: Vec<u64> = owned.successors(u).collect();
        let view_succ: Vec<u64> = view.successors(u).collect();
        assert_eq!(view_succ, owned_succ, "successors of {u} mismatched");
    }
}

#[test]
fn view_rejects_bad_magic() {
    let owned = WebGraphBuilder::new(3).build();
    let mut bad = owned.to_bytes();
    bad[0] = b'X';
    assert!(matches!(
        WebGraphView::open(bytes::Bytes::from(bad)),
        Err(WebGraphError::BadMagic)
    ));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: FAIL — `cannot find type 'WebGraphView'`.

- [ ] **Step 3: Write the implementation**

Add to `webgraph.rs`, after the `impl WebGraphCodec` block containing `to_bytes`/`from_bytes`/`from_bytes_shared` (and before the `#[cfg(test)] mod tests`):

```rust
/// A borrowing reader over a [`WebGraphCodec`] blob.
///
/// Holds a single `bytes::Bytes` and parsed header offsets; the
/// per-node bit-offset index and the bit stream are sliced from the
/// held bytes on demand. The owned `Vec<u64>` of offsets is decoded
/// once at open time (typically 8 bytes per node — small relative to
/// the bit stream).
#[derive(Debug, Clone)]
pub struct WebGraphView {
    blob: bytes::Bytes,
    num_nodes: u64,
    num_edges: u64,
    bit_len: u64,
    /// Decoded once at open time.
    offsets: Vec<u64>,
    /// Byte offset of the bit stream within `blob`.
    bits_offset: usize,
    /// Number of bytes in the bit stream (`bit_len.div_ceil(8)`).
    bits_byte_len: usize,
}

impl WebGraphView {
    /// Opens a blob produced by [`WebGraphCodec::to_bytes`].
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on a malformed blob — same conditions
    /// as [`WebGraphCodec::from_bytes`].
    pub fn open(blob: bytes::Bytes) -> Result<Self, WebGraphError> {
        let buf = blob.as_ref();
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

        // Decode offsets array.
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
        // Validate offsets.
        if let Some(&first) = offsets.first()
            && first != 0
        {
            return Err(WebGraphError::BadOffset {
                node: 0,
                offset: first,
                bit_len,
            });
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
        if let Some(&last) = offsets.last()
            && last != bit_len
        {
            return Err(WebGraphError::BadOffset {
                node: num_nodes,
                offset: last,
                bit_len,
            });
        }

        // Validate the bits section fits.
        // reason: bit_len/8 fits usize for any practical snapshot
        #[allow(clippy::cast_possible_truncation)]
        let bits_byte_len = bit_len.div_ceil(8) as usize;
        let bits_end = bits_offset + bits_byte_len;
        if bits_end > buf.len() {
            return Err(WebGraphError::Truncated {
                need: bits_end,
                have: buf.len(),
            });
        }

        Ok(Self {
            blob,
            num_nodes,
            num_edges,
            bit_len,
            offsets,
            bits_offset,
            bits_byte_len,
        })
    }

    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges.
    #[must_use]
    pub fn num_edges(&self) -> u64 {
        self.num_edges
    }

    /// Out-degree of `node`.
    #[must_use]
    pub fn out_degree(&self, node: u64) -> u64 {
        if node >= self.num_nodes {
            return 0;
        }
        let start = self.offsets[node as usize];
        let bits = &self.blob.as_ref()[self.bits_offset..self.bits_offset + self.bits_byte_len];
        let mut reader = BitReader::new(bits, self.bit_len);
        reader.seek(start);
        reader.read_gamma().unwrap_or(1) - 1
    }

    /// Iterator over the successors of `node`, in ascending dst order.
    pub fn successors(&self, node: u64) -> SuccessorIter<'_> {
        if node >= self.num_nodes {
            return SuccessorIter::empty();
        }
        let start = self.offsets[node as usize];
        let bits = &self.blob.as_ref()[self.bits_offset..self.bits_offset + self.bits_byte_len];
        let mut reader = BitReader::new(bits, self.bit_len);
        reader.seek(start);
        let degree_plus_one = reader.read_gamma().unwrap_or(1);
        let degree = degree_plus_one - 1;
        SuccessorIter::new(reader, node, degree)
    }
}
```

The `SuccessorIter::empty` and `SuccessorIter::new` helpers exist; the iterator type is the same one `WebGraphCodec::successors` returns. Add a small constructor `new` to `SuccessorIter` if the existing private constructor signature doesn't match — check the existing impl and either reuse or add:

```rust
impl<'a> SuccessorIter<'a> {
    pub(crate) fn new(reader: BitReader<'a>, node: u64, remaining: u64) -> Self {
        Self {
            reader,
            node,
            remaining,
            last_dst: None,
        }
    }
}
```

If this constructor already exists with the same signature, skip adding it.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::webgraph`
Expected: PASS — 13 tests (11 from before + 2 new).

- [ ] **Step 5: Add the public re-export**

In `crates/grafeo-core/src/codec/mod.rs`, extend the existing `pub use webgraph::{...};` line:

```rust
pub use webgraph::{
    SuccessorIter, WebGraphBuilder, WebGraphCodec, WebGraphError, WebGraphView,
};
```

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/codec/webgraph.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(webgraph): WebGraphView for zero-copy borrowing successors"
```

---

## Task 4: Cross-codec parity proptest

**Files:**
- Create: `crates/grafeo-core/tests/codec_view_parity.rs`

- [ ] **Step 1: Write the test file**

Create `crates/grafeo-core/tests/codec_view_parity.rs`:

```rust
//! Parity regression test: `*View` borrowing readers produce identical
//! results to the owned codecs across arbitrary inputs.
//!
//! ```bash
//! cargo test -p grafeo-core --test codec_view_parity
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test codec_view_parity
//! ```

use grafeo_common::types::NodeId;
use grafeo_core::codec::{FsstCodec, FsstView, WebGraphBuilder, WebGraphCodec, WebGraphView};
use grafeo_core::index::vector::{RabitqView, TwoStageVectorIndex};
use proptest::prelude::*;

// ── FSST parity ──────────────────────────────────────────────────

fn string_set() -> impl Strategy<Value = Vec<Vec<u8>>> {
    proptest::collection::vec(
        proptest::collection::vec(any::<u8>(), 0..=32),
        0..=16,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// FsstView::get matches FsstCodec::get for every string index.
    #[test]
    fn fsst_view_matches_owned(strings in string_set()) {
        let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
        let owned = FsstCodec::build(&refs);
        let blob = bytes::Bytes::from(owned.to_bytes());
        let view = FsstView::open(blob).expect("open");

        prop_assert_eq!(view.len(), owned.len());
        for i in 0..strings.len() {
            let owned_s = owned.get(i).expect("owned").expect("decode");
            let view_s = view.get(i).expect("view").expect("decode");
            prop_assert_eq!(&view_s, &owned_s);
        }
    }
}

// ── WebGraph parity ──────────────────────────────────────────────

fn webgraph_input() -> impl Strategy<Value = (u64, Vec<(u64, u64)>)> {
    (1u64..=20).prop_flat_map(|n| {
        let edges = proptest::collection::vec((0u64..n, 0u64..n), 0..=60);
        (Just(n), edges)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// WebGraphView::successors matches WebGraphCodec::successors for every node.
    #[test]
    fn webgraph_view_matches_owned((num_nodes, edges) in webgraph_input()) {
        let mut b = WebGraphBuilder::new(num_nodes);
        for &(s, d) in &edges {
            b.add_edge(s, d).unwrap();
        }
        let owned = b.build();
        let blob = bytes::Bytes::from(owned.to_bytes());
        let view = WebGraphView::open(blob).expect("open");

        prop_assert_eq!(view.num_nodes(), owned.num_nodes());
        prop_assert_eq!(view.num_edges(), owned.num_edges());
        for u in 0..owned.num_nodes() {
            let owned_succ: Vec<u64> = owned.successors(u).collect();
            let view_succ: Vec<u64> = view.successors(u).collect();
            prop_assert_eq!(view_succ, owned_succ);
        }
    }
}

// ── RaBitQ parity ────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.f32().max(f32::MIN_POSITIVE);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RabitqView::search matches TwoStageVectorIndex::search.
    #[test]
    fn rabitq_view_matches_owned(
        seed in any::<u64>(),
        count in 5usize..=40,
    ) {
        let dim = 32;
        let mut rng = Rng(seed);
        let vectors: Vec<(NodeId, Vec<f32>)> = (0..count)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
                (NodeId::new(i as u64 + 1), v)
            })
            .collect();
        let owned = TwoStageVectorIndex::build(&vectors, dim, seed ^ 0xAA);
        let blob = bytes::Bytes::from(owned.to_bytes());
        let view = RabitqView::open(blob).expect("open");

        prop_assert_eq!(view.len(), owned.len());

        let query = vectors[(seed as usize) % vectors.len()].1.clone();
        let k = 5;
        prop_assert_eq!(view.search(&query, k, 8), owned.search(&query, k, 8));
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p grafeo-core --test codec_view_parity`
Expected: PASS — 3 proptests (128 + 128 + 64 cases). Also run at 512: `PROPTEST_CASES=512 cargo test -p grafeo-core --test codec_view_parity`.

- [ ] **Step 3: Commit**

```bash
git add crates/grafeo-core/tests/codec_view_parity.rs
git commit -m "test(codec): View vs Owned parity proptest for all three codecs"
```

---

## Task 5: `ColumnType::FsstString` schema variant

**Files:**
- Modify: `crates/grafeo-core/src/graph/compact/schema.rs`
- Modify: `crates/grafeo-core/src/graph/compact/builder.rs`
- Modify: `crates/grafeo-core/src/graph/compact/section.rs`

- [ ] **Step 1: Survey the existing ColumnType enum**

```bash
cd /Users/chris/TMP/grafeo
grep -n "enum ColumnType\|ColumnType::" crates/grafeo-core/src/graph/compact/schema.rs | head -20
grep -n "FIXME(plan-2d)" crates/grafeo-core/src/graph/compact/builder.rs crates/grafeo-core/src/graph/compact/section.rs
```

Locate the `pub enum ColumnType` definition and the two `FIXME(plan-2d)` markers.

- [ ] **Step 2: Add the variant**

In `crates/grafeo-core/src/graph/compact/schema.rs`, find the `pub enum ColumnType` (likely `#[non_exhaustive]`). Add a new variant after `DictString`:

```rust
    /// FSST-compressed UTF-8 strings.
    FsstString,
```

If the enum has a Display impl or other downstream uses (find with `grep -n 'ColumnType::' crates/grafeo-core/src/graph/compact/`), add `Self::FsstString => "fsst_string"` arms wherever the compiler complains about a non-exhaustive match.

- [ ] **Step 3: Update the FIXME stubs**

In `crates/grafeo-core/src/graph/compact/builder.rs`, find the `FIXME(plan-2d)` block and replace `ColumnCodec::Fsst(_) => ColumnType::DictString,` with:

```rust
ColumnCodec::Fsst(_) => ColumnType::FsstString,
```

Remove the surrounding `FIXME(plan-2d):` comment block — the FIXME is resolved.

Repeat in `crates/grafeo-core/src/graph/compact/section.rs`.

- [ ] **Step 4: Run the tests**

Run: `cargo build -p grafeo-core --features compact-store`
Expected: clean build. The compiler will surface any missing match arm on `ColumnType` — fix those as you go.

Run: `cargo test -p grafeo-core --features compact-store --lib graph::compact`
Expected: all pre-existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/graph/compact/schema.rs crates/grafeo-core/src/graph/compact/builder.rs crates/grafeo-core/src/graph/compact/section.rs
git commit -m "feat(compact): ColumnType::FsstString schema variant"
```

---

## Task 6: `ColumnCodec::Fsst` serialization wire format

**Files:**
- Modify: `crates/grafeo-core/src/graph/compact/column.rs`

- [ ] **Step 1: Survey existing codec serialization**

```bash
cd /Users/chris/TMP/grafeo
sed -n '850,920p' crates/grafeo-core/src/graph/compact/column.rs
sed -n '1070,1180p' crates/grafeo-core/src/graph/compact/column.rs
```

Look at how `Self::Dict(dict) => ...` is handled in `write_to`, `write_to_v2`, `write_to_v3` (via `emit_blocked_codec`), and the corresponding `read_from*` functions. Note the tag byte used to discriminate codec variants on the wire (likely 0/1/2/etc. — find with `grep -n 'CODEC_TAG\|write_u8\|tag.*=' crates/grafeo-core/src/graph/compact/column.rs | head -20`).

- [ ] **Step 2: Assign a tag byte for `Self::Fsst`**

Pick the next unused codec tag. If existing tags are 0–6 (BitPacked=0, Dict=1, Bitmap=2, Int8Vector=3, Float64=4, Float32Vector=5, RawI64=6), use `7` for `Fsst`. Search to confirm:

```bash
grep -n "0u8\|1u8\|2u8\|3u8\|4u8\|5u8\|6u8\|7u8" crates/grafeo-core/src/graph/compact/column.rs | head -30
```

Add a `const FSST_TAG: u8 = 7;` (or whatever the next unused value is) at the top of the file alongside any existing codec tags.

- [ ] **Step 3: Write the `Self::Fsst` arms**

For each of `write_to`, `write_to_v2`, and `emit_blocked_codec` (which feeds `write_to_v3`), replace the `unimplemented!()` panic with a proper serialization:

```rust
Self::Fsst(fsst) => {
    buf.push(FSST_TAG);
    let body = fsst.to_bytes();
    buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
    buf.extend_from_slice(&body);
}
```

(The exact framing — tag + length prefix + body — should match the surrounding pattern. If `Dict` uses a different framing, mirror that instead.)

For the `read_from*` functions, add a corresponding match case on `FSST_TAG`:

```rust
FSST_TAG => {
    let len = u32::from_le_bytes(/* ... read 4 bytes ... */) as usize;
    let body = /* ... read len bytes ... */;
    let fsst = FsstCodec::from_bytes(body).map_err(|e| /* ... wrap into the codec's error type ... */)?;
    Ok(Self::Fsst(fsst))
}
```

The exact form depends on the existing `read_from*` signatures and the error type they return. Find a comparable arm (likely the `Dict` arm) and mirror it.

- [ ] **Step 4: Write a test that exercises the full round-trip**

Append to the existing `#[cfg(test)] mod tests` in `column.rs`:

```rust
    #[test]
    fn column_codec_fsst_section_round_trip() {
        use crate::codec::FsstCodec;
        let strings: Vec<&[u8]> = vec![b"alpha", b"beta", b"alpha"];
        let codec = FsstCodec::build(&strings);
        let col = ColumnCodec::Fsst(codec);

        // Serialize via write_to_v3 (the current production path).
        let mut buf = Vec::new();
        col.write_to_v3(&mut buf, None);

        // Deserialize via read_from_v3.
        let mut pos = 0;
        let bytes = bytes::Bytes::from(buf);
        let (decoded, _stats) = ColumnCodec::read_from_v3(&bytes, &mut pos).expect("decode");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded.get(0), Some(Value::String(ArcStr::from("alpha"))));
        assert_eq!(decoded.get(1), Some(Value::String(ArcStr::from("beta"))));
        assert_eq!(decoded.get(2), Some(Value::String(ArcStr::from("alpha"))));
    }
```

The exact signature of `read_from_v3` may differ — check the existing test for `Dict` or another codec and mirror.

- [ ] **Step 5: Run to verify it passes**

```bash
cargo test -p grafeo-core --features compact-store --lib graph::compact::column::tests::column_codec_fsst_section_round_trip
cargo test -p grafeo-core --features compact-store --lib graph::compact
cargo test -p grafeo-engine --features "compact-store lpg gql" --test compact_roundtrip_proptest
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/graph/compact/column.rs
git commit -m "feat(compact): ColumnCodec::Fsst wire format (resolves plan-2d FIXME)"
```

---

## Task 7: Benchmark — View vs Owned query latency

**Files:**
- Create: `crates/grafeo-core/benches/codec_views.rs`
- Modify: `crates/grafeo-core/Cargo.toml`

- [ ] **Step 1: Register the bench**

In `crates/grafeo-core/Cargo.toml`, after the existing bench entries, add:

```toml
[[bench]]
name = "codec_views"
harness = false
```

- [ ] **Step 2: Write the benchmark**

Create `crates/grafeo-core/benches/codec_views.rs`:

```rust
//! View vs Owned query latency for the three Plan 2 codecs.
//!
//! Run: `cargo bench -p grafeo-core --bench codec_views`
//!
//! The output informs whether the WASM bindings should be migrated to
//! the View types. A meaningful latency penalty (e.g., > 1.5×) means the
//! current owned-codec WASM bindings are the right default; a comparable
//! or smaller latency means the views are a Pareto improvement.

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_core::codec::{FsstCodec, FsstView, WebGraphBuilder, WebGraphCodec, WebGraphView};
use grafeo_core::index::vector::{RabitqView, TwoStageVectorIndex};
use std::hint::black_box;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.f32().max(f32::MIN_POSITIVE);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn bench_rabitq(c: &mut Criterion) {
    const DIM: usize = 128;
    const COUNT: usize = 2_000;
    let mut rng = Rng(42);
    let vectors: Vec<(NodeId, Vec<f32>)> = (0..COUNT)
        .map(|i| {
            let v: Vec<f32> = (0..DIM).map(|_| rng.gaussian()).collect();
            (NodeId::new(i as u64 + 1), v)
        })
        .collect();
    let owned = TwoStageVectorIndex::build(&vectors, DIM, 1);
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = RabitqView::open(blob).expect("open");
    let query = vectors[0].1.clone();

    let mut group = c.benchmark_group("rabitq_search_2k_128d");
    group.bench_function("owned", |b| {
        b.iter(|| black_box(owned.search(black_box(&query), 10, 8)));
    });
    group.bench_function("view", |b| {
        b.iter(|| black_box(view.search(black_box(&query), 10, 8)));
    });
    group.finish();
}

fn bench_fsst(c: &mut Criterion) {
    let mut rng = Rng(99);
    let strings: Vec<Vec<u8>> = (0..1000)
        .map(|_| {
            let len = (rng.u64() % 32) as usize + 4;
            (0..len).map(|_| (rng.u64() & 0x7F) as u8 + 32).collect()
        })
        .collect();
    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let owned = FsstCodec::build(&refs);
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = FsstView::open(blob).expect("open");

    let mut group = c.benchmark_group("fsst_get_random_1k");
    group.bench_function("owned", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 31) % 1000;
            black_box(owned.get(black_box(i)).unwrap().unwrap())
        });
    });
    group.bench_function("view", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 31) % 1000;
            black_box(view.get(black_box(i)).unwrap().unwrap())
        });
    });
    group.finish();
}

fn bench_webgraph(c: &mut Criterion) {
    let mut rng = Rng(7);
    let n: u64 = 1000;
    let mut b = WebGraphBuilder::new(n);
    for _ in 0..15_000 {
        let s = rng.u64() % n;
        let d = rng.u64() % n;
        b.add_edge(s, d).unwrap();
    }
    let owned = b.build();
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = WebGraphView::open(blob).expect("open");

    let mut group = c.benchmark_group("webgraph_successors_random_1k");
    group.bench_function("owned", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 17) % 1000;
            black_box(owned.successors(black_box(i)).collect::<Vec<_>>())
        });
    });
    group.bench_function("view", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 17) % 1000;
            black_box(view.successors(black_box(i)).collect::<Vec<_>>())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_rabitq, bench_fsst, bench_webgraph);
criterion_main!(benches);
```

- [ ] **Step 3: Run the benchmark**

Run: `cargo bench -p grafeo-core --bench codec_views`
Expected: compiles and runs. Record the latency numbers in the commit message.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/benches/codec_views.rs crates/grafeo-core/Cargo.toml
git commit -m "bench(codec): View vs Owned query latency for all three codecs"
```

---

## Task 8: CHANGELOG and full verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the CHANGELOG entry**

In `CHANGELOG.md`, under the existing `## [Unreleased] / ### Added` section, append:

```markdown
- `RabitqView`, `FsstView`, `WebGraphView` — borrowing readers over each codec's blob that hold `bytes::Bytes` and slice the query-hot data on demand. Search/get/successors produce bit-identical results to the owned codecs (see `codec_view_parity` proptest) without the per-vector heap allocations the owned `from_bytes` path materializes. Plan 3's mmap-backed build pipeline targets these.
- `ColumnType::FsstString` and `ColumnCodec::Fsst` serialization — completes the deferred sub-plan 2c integration. FSST columns now round-trip through the compact-store section format and are correctly classified by the schema inference paths.
```

- [ ] **Step 2: Run the full verification suite**

```bash
cargo test -p grafeo-core --lib codec::bitstream
cargo test -p grafeo-core --lib codec::fsst
cargo test -p grafeo-core --lib codec::webgraph
cargo test -p grafeo-core --lib index::vector::rabitq
cargo test -p grafeo-core --test codec_view_parity
PROPTEST_CASES=512 cargo test -p grafeo-core --test codec_view_parity
cargo test -p grafeo-core --test rabitq_recall
cargo test -p grafeo-core --test fsst_round_trip
cargo test -p grafeo-core --test webgraph_round_trip
cargo test -p grafeo-core --features compact-store --lib graph::compact
cargo test -p grafeo-engine --features "compact-store lpg gql" --test compact_roundtrip_proptest
cargo clippy -p grafeo-core --all-targets -- -D warnings
cargo build -p grafeo-core --target wasm32-unknown-unknown
cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features "rabitq-codec fsst-codec webgraph-codec"
```

If `wasm-pack` is in PATH:
```bash
wasm-pack test --node crates/bindings/wasm --features "rabitq-codec fsst-codec webgraph-codec"
```

Expected: all green (pre-existing wasm test failures from sub-plans 2a/2c remain pre-existing; do not try to fix them).

If clippy finds nits in our task files, fix them in the same commit. Do NOT touch other crates.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note zero-copy views and FSST column integration"
```

---

## Self-Review (completed by the plan author)

**Spec coverage** — every Plan 2d requirement maps to a task:
- Borrowing `from_bytes_shared(Bytes)` for all three codecs → Tasks 1, 2, 3.
- Parity proptest → Task 4.
- `ColumnType::FsstString` → Task 5.
- `ColumnCodec::Fsst` serialization → Task 6.
- Benchmark to inform WASM-binding decision → Task 7.

**Type consistency** — `RabitqView` (`open`, `len`, `is_empty`, `search`), `FsstView` (`open`, `len`, `is_empty`, `get`), `WebGraphView` (`open`, `num_nodes`, `num_edges`, `out_degree`, `successors`). The `SuccessorIter::new` constructor is added if not present. The `FSST_TAG` byte is assigned at the next unused codec tag.

**Placeholder scan** — Task 5 and Task 6 contain "find via grep" steps because the existing `column.rs` and `schema.rs` are too large to dictate exact line numbers in advance; the engineer can locate the FIXME markers and codec tag site in seconds. All code blocks are complete; the read_from* arm template explicitly says "mirror the Dict arm" because the exact signature varies between v1/v2/v3.

**Known follow-ups (out of scope for 2d):**
- WASM bindings: the Task 7 benchmark informs whether to migrate. Either keep the owned path (cheap), add View-based bindings alongside (more API surface), or migrate internally (transparent). Defer until the bench numbers are in.
- `ColumnCodec::Webgraph` integration: WebGraph blobs are not currently a compact-store column codec. A future sub-plan could add a `RelTableSection` variant holding a `WebGraphCodec` for compressed edge tables. Independent of this work.
- Referentiation and zeta-3 for WebGraph: noted in sub-plan 2b's follow-ups; no change here.
