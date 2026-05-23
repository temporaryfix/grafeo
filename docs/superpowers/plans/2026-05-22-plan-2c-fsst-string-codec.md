# Sub-plan 2c — FSST String Codec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Fast Static Symbol Table string compression codec to the Grafeo fork, with O(1) random-access decode of any individual string, exposed through the WASM bindings.

**Architecture:** A new in-tree, dependency-free module `codec/fsst.rs`. Build a 256-entry symbol table from a sample of strings — code 0 is reserved as the escape marker, codes 1..=255 hold 1–8-byte symbols selected greedily by `(length−1) × frequency` (the saved-bytes-per-occurrence score). Encoding scans each string left-to-right finding the longest matching symbol; unmatched bytes use the escape mechanism. An `FsstCodec` holds the symbol table plus a per-string offsets array, giving random-access decode by slicing the compressed stream. A blob `to_bytes`/`from_bytes` round-trip honours Plan 2's zero-copy contract (fixed header, naturally-aligned arrays, blob-relative `u64` offsets, `bytes::Bytes` overload, CRC32). Integrates as a new `ColumnCodec::Fsst` variant for compact-store string columns, and is also usable standalone for the string blob in Plan 3's pipeline.

**Tech Stack:** Rust 2024, `grafeo-core` (`codec/`, `graph/compact/`), `crates/bindings/wasm`, `proptest` (round-trip equivalence). No new runtime dependencies.

---

## File Structure

- **Create** `crates/grafeo-core/src/codec/fsst.rs` — the codec: `FsstError`, `SymbolTable`, `FsstCodec`, blob serialization, unit tests.
- **Modify** `crates/grafeo-core/src/codec/mod.rs` — register `pub mod fsst;` and re-export the public types.
- **Modify** `crates/grafeo-core/src/graph/compact/column.rs` — add a `ColumnCodec::Fsst` variant for string columns (additive to the `#[non_exhaustive]` enum); plumb `get`, `len`, `find_eq`, and `memory_bytes` paths.
- **Create** `crates/grafeo-core/tests/fsst_round_trip.rs` — proptest plus fixed-seed regression cases.
- **Modify** `crates/grafeo-core/Cargo.toml` — no new deps (proptest dev-dep was added by sub-plan 2a).
- **Create** `crates/bindings/wasm/src/codecs.rs` extension — add `#[wasm_bindgen] struct FsstCodec` next to the existing `RabitqCodec`.
- **Modify** `crates/bindings/wasm/Cargo.toml` — extend or reuse the `rabitq-codec` feature, OR add a sibling `fsst-codec` feature. The plan adds `fsst-codec` to keep the two codecs independently selectable.
- **Modify** `crates/bindings/wasm/tests/web.rs` — wasm round-trip test gated on `fsst-codec`.
- **Modify** `CHANGELOG.md` — note the new codec.

Reference facts from the existing code (read before starting):
- `crates/grafeo-core/src/codec/mod.rs` — codec module registry; existing public exports include `BitPackedInts`, `DeltaBitPacked`, `BitVector`, `DictionaryEncoding`. FSST joins this set.
- `crates/grafeo-core/src/codec/dictionary.rs:147` — `DictionaryEncoding`; the codec FSST competes with for short repeated strings. Reuse none of its internals — FSST is a different shape (per-string compression with random access, not a per-value code).
- `crates/grafeo-core/src/graph/compact/column.rs:227` — `pub enum ColumnCodec` is `#[non_exhaustive]`, so adding a variant is safe for downstream callers. `Dict(DictionaryEncoding)` is today's string variant; `Fsst(FsstCodec)` is the new sibling.
- `crates/grafeo-core/src/index/vector/rabitq.rs` — sub-plan 2a's reference implementation. The blob layout (fixed header with `u64` offsets, `from_bytes_shared(Bytes)` overload, CRC trailer) is the contract template; mirror it.

---

## Task 1: Module scaffold — `FsstError` and `SymbolTable` type

**Files:**
- Create: `crates/grafeo-core/src/codec/fsst.rs`
- Modify: `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/grafeo-core/src/codec/fsst.rs` with this content:

```rust
//! FSST — Fast Static Symbol Table string compression (Boncz, Neumann, Leis,
//! PVLDB 2020) with O(1) random-access decode of individual strings.
//!
//! A symbol table holds up to 255 user symbols (1–8 bytes each); code 0 is
//! reserved as the escape marker, signalling that the next byte in the
//! compressed stream is a literal. Compression scans each input string
//! left-to-right, emitting the code of the longest matching symbol at each
//! position, or `[0, byte]` when no symbol matches. Decompression is a single
//! pass of table lookups.
//!
//! Each string is stored independently with a recorded byte offset into the
//! shared compressed stream, so any string can be decoded in isolation
//! without touching its neighbours — the property `DictionaryEncoding`
//! provides per-value but FSST extends to actual short-string compression
//! (typical 2–3× over dictionary encoding alone for name-like columns).

/// Reserved code marking that the next compressed byte is a literal.
pub(crate) const ESCAPE: u8 = 0;

/// Maximum symbol length in bytes (FSST paper convention).
pub const MAX_SYMBOL_LEN: usize = 8;

/// Errors returned when opening or decoding an FSST blob.
#[derive(Debug, thiserror::Error)]
pub enum FsstError {
    /// The blob is shorter than the bytes a field needs.
    #[error("fsst blob truncated: need {need} bytes, have {have}")]
    Truncated {
        /// Minimum required buffer length (end offset of the read that failed).
        need: usize,
        /// Bytes available.
        have: usize,
    },
    /// The leading magic bytes are not `GFST`.
    #[error("fsst blob: bad magic (expected GFST)")]
    BadMagic,
    /// The version byte is not supported by this build.
    #[error("fsst blob: unsupported version {0}")]
    BadVersion(u8),
    /// The trailing CRC32 does not match the body.
    #[error("fsst blob: crc mismatch (stored {stored:#010x}, computed {computed:#010x})")]
    CrcMismatch {
        /// CRC read from the trailer.
        stored: u32,
        /// CRC computed over the body.
        computed: u32,
    },
    /// A stored symbol's `length` byte is out of range.
    #[error("fsst blob: symbol {code} has invalid length {length} (must be 1..={MAX_SYMBOL_LEN})")]
    BadSymbolLength {
        /// Symbol code whose length is invalid.
        code: u8,
        /// The out-of-range length byte read from the blob.
        length: u8,
    },
    /// A per-string offset extends past the compressed stream's end.
    #[error("fsst blob: offset {offset} for string {index} exceeds compressed length {len}")]
    BadOffset {
        /// String index whose offset is out of range.
        index: usize,
        /// The out-of-range offset.
        offset: u64,
        /// Length of the compressed stream.
        len: u64,
    },
    /// A compressed sequence contained an escape byte with no following literal.
    #[error("fsst decode: truncated escape sequence at position {0}")]
    TruncatedEscape(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_marker_is_zero() {
        assert_eq!(ESCAPE, 0);
    }

    #[test]
    fn max_symbol_len_is_eight() {
        assert_eq!(MAX_SYMBOL_LEN, 8);
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/grafeo-core/src/codec/mod.rs`, after the line `pub mod dictionary;` add:

```rust
pub mod fsst;
```

(Do NOT yet add a `pub use fsst::{...}` re-export — the types it would name are added in later tasks. The re-export is added in Task 5 Step 5.)

- [ ] **Step 3: Run the tests**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — `escape_marker_is_zero` and `max_symbol_len_is_eight`, 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(fsst): module scaffold with FsstError and constants"
```

---

## Task 2: `SymbolTable` — entries and lookups

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn symbol_table_default_has_no_symbols() {
    let t = SymbolTable::default();
    for code in 1u8..=255 {
        assert!(t.symbol(code).is_none(), "code {code} should have no symbol");
    }
}

#[test]
fn symbol_table_set_and_lookup() {
    let mut t = SymbolTable::default();
    t.set(1, b"abc");
    t.set(2, b"xy");
    assert_eq!(t.symbol(1), Some(b"abc" as &[u8]));
    assert_eq!(t.symbol(2), Some(b"xy" as &[u8]));
    assert_eq!(t.symbol(3), None);
}

#[test]
#[should_panic(expected = "symbol length must be 1..=8")]
fn symbol_table_rejects_oversize_symbol() {
    let mut t = SymbolTable::default();
    t.set(1, b"ninebytes");
}

#[test]
#[should_panic(expected = "cannot assign code 0 (escape)")]
fn symbol_table_rejects_escape_code() {
    let mut t = SymbolTable::default();
    t.set(0, b"x");
}

#[test]
fn longest_match_finds_the_longest_prefix() {
    let mut t = SymbolTable::default();
    t.set(1, b"a");
    t.set(2, b"ab");
    t.set(3, b"abc");
    // Longest match for "abcdef" is "abc" (code 3, length 3).
    assert_eq!(t.longest_match(b"abcdef"), Some((3, 3)));
    // Longest match for "abxyz" is "ab" (code 2, length 2).
    assert_eq!(t.longest_match(b"abxyz"), Some((2, 2)));
    // No symbol starts with 'z'.
    assert_eq!(t.longest_match(b"zzz"), None);
    // Empty input: no match.
    assert_eq!(t.longest_match(b""), None);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: FAIL — `cannot find type/value 'SymbolTable'`.

- [ ] **Step 3: Write the implementation**

Add to `fsst.rs`, after the `FsstError` enum (and before the `#[cfg(test)] mod tests`):

```rust
/// A 256-code symbol table. Code 0 is the escape marker; codes 1..=255 hold
/// 1–8-byte symbols. Empty slots (length = 0) are absent symbols, looked up
/// as [`None`] by [`Self::symbol`].
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolTable {
    /// Symbol length per code (0 = absent). Index 0 is unused (escape).
    lengths: [u8; 256],
    /// Symbol bodies, 8 bytes per slot (right-padded with zeros). Index 0
    /// is unused.
    bodies: [[u8; MAX_SYMBOL_LEN]; 256],
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            lengths: [0u8; 256],
            bodies: [[0u8; MAX_SYMBOL_LEN]; 256],
        }
    }
}

impl SymbolTable {
    /// Returns the symbol bytes for `code`, or `None` if the slot is empty
    /// or the code is the escape marker.
    #[must_use]
    pub fn symbol(&self, code: u8) -> Option<&[u8]> {
        if code == ESCAPE {
            return None;
        }
        let len = self.lengths[code as usize] as usize;
        if len == 0 {
            None
        } else {
            Some(&self.bodies[code as usize][..len])
        }
    }

    /// Assigns `symbol` to `code`.
    ///
    /// # Panics
    /// Panics if `code` is the escape marker (0), if `symbol.is_empty()`,
    /// or if `symbol.len() > MAX_SYMBOL_LEN`.
    pub fn set(&mut self, code: u8, symbol: &[u8]) {
        assert!(code != ESCAPE, "cannot assign code 0 (escape)");
        assert!(
            (1..=MAX_SYMBOL_LEN).contains(&symbol.len()),
            "symbol length must be 1..=8, got {}",
            symbol.len()
        );
        let idx = code as usize;
        // reason: symbol.len() validated 1..=8 above
        #[allow(clippy::cast_possible_truncation)]
        {
            self.lengths[idx] = symbol.len() as u8;
        }
        self.bodies[idx][..symbol.len()].copy_from_slice(symbol);
        // Zero-fill the remainder so PartialEq / serialization don't see
        // stale bytes from a previous symbol.
        for b in &mut self.bodies[idx][symbol.len()..] {
            *b = 0;
        }
    }

    /// Returns `(code, length)` of the longest symbol whose bytes are a
    /// prefix of `input`. Ties on length break by the smaller code.
    ///
    /// O(255 × MAX_SYMBOL_LEN) worst case; for hot paths build a per-first-
    /// byte index, but this scalar lookup is adequate for the codec's
    /// expected string sizes.
    #[must_use]
    pub fn longest_match(&self, input: &[u8]) -> Option<(u8, usize)> {
        if input.is_empty() {
            return None;
        }
        let max_check = input.len().min(MAX_SYMBOL_LEN);
        let mut best: Option<(u8, usize)> = None;
        for code in 1u8..=255 {
            let len = self.lengths[code as usize] as usize;
            if len == 0 || len > max_check {
                continue;
            }
            if &self.bodies[code as usize][..len] == &input[..len] {
                match best {
                    None => best = Some((code, len)),
                    Some((_, blen)) if len > blen => best = Some((code, len)),
                    _ => {}
                }
            }
        }
        best
    }

    /// Returns the number of assigned symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lengths.iter().filter(|&&l| l > 0).count()
    }

    /// True if no symbols are assigned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lengths.iter().all(|&l| l == 0)
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs
git commit -m "feat(fsst): SymbolTable with longest-prefix match"
```

---

## Task 3: Training — greedy symbol selection from a sample

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn train_empty_sample_returns_empty_table() {
    let table = SymbolTable::train(&[]);
    assert!(table.is_empty());
}

#[test]
fn train_picks_frequent_substrings() {
    // Sample where "the " appears many times.
    let strings: Vec<&[u8]> = vec![
        b"the cat",
        b"the dog",
        b"the bird",
        b"the rat",
        b"the fox",
    ];
    let table = SymbolTable::train(&strings);
    // The table should contain "the " (or a prefix of it) as a multi-byte symbol.
    let has_the = (1u8..=255).any(|c| {
        table.symbol(c).is_some_and(|s| s.starts_with(b"the"))
    });
    assert!(has_the, "expected a symbol covering 'the'");
}

#[test]
fn train_always_yields_a_table_that_can_encode_sample_bytes() {
    // Any byte in the sample is either matched by a symbol or escape-encoded —
    // we test the latter by checking the symbol table is buildable.
    let strings: Vec<&[u8]> = vec![b"hello", b"world", b""];
    let _ = SymbolTable::train(&strings);  // does not panic on empty strings
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: FAIL — `no function or associated item named 'train'`.

- [ ] **Step 3: Write the implementation**

Add a `train` associated function inside `impl SymbolTable`:

```rust
    /// Builds a symbol table from a sample of strings by greedy substring
    /// selection.
    ///
    /// All substrings of length 1..=MAX_SYMBOL_LEN are scored by
    /// `(length − 1) × frequency`, the bytes-saved-per-occurrence
    /// heuristic versus a literal escape-encoding. The top 255 by score
    /// are assigned to codes 1..=255 in descending score order. The
    /// resulting table can encode any input — bytes with no matching
    /// symbol use the escape mechanism.
    #[must_use]
    pub fn train(sample: &[&[u8]]) -> Self {
        use std::collections::HashMap;

        if sample.iter().all(|s| s.is_empty()) {
            return Self::default();
        }

        // Count every substring of length 1..=MAX_SYMBOL_LEN in the sample.
        let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for s in sample {
            for start in 0..s.len() {
                let max_end = (start + MAX_SYMBOL_LEN).min(s.len());
                for end in (start + 1)..=max_end {
                    let sub = &s[start..end];
                    *counts.entry(sub.to_vec()).or_insert(0) += 1;
                }
            }
        }

        // Score by (length − 1) × frequency. Length-1 substrings score 0
        // but are kept so the most-common bytes still get codes.
        let mut scored: Vec<(Vec<u8>, u64)> = counts
            .into_iter()
            .map(|(sub, freq)| {
                let score = (sub.len() as u64 - 1).max(0) * freq + freq;
                // The `+ freq` term lets length-1 substrings out-rank
                // zero-frequency multi-byte ones, preserving byte coverage.
                (sub, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())));

        let mut table = Self::default();
        let mut next_code: u8 = 1;
        for (sub, _) in scored.into_iter().take(255) {
            table.set(next_code, &sub);
            if next_code == 255 {
                break;
            }
            next_code += 1;
        }
        table
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs
git commit -m "feat(fsst): greedy symbol-table training from sample strings"
```

---

## Task 4: Encode and decode a single string

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn encode_decode_round_trip_simple() {
    let mut table = SymbolTable::default();
    table.set(1, b"the ");
    table.set(2, b"cat");
    table.set(3, b"dog");

    let input = b"the cat";
    let compressed = table.encode(input);
    let decoded = table.decode(&compressed).expect("decode");
    assert_eq!(decoded, input);

    // 'the ' (1) + 'cat' (1) = 2 bytes compressed vs 7 input.
    assert_eq!(compressed.len(), 2);
}

#[test]
fn encode_decode_round_trip_with_escapes() {
    let mut table = SymbolTable::default();
    table.set(1, b"hello");
    // 'world' is NOT in the table; every byte should be escape-encoded.

    let input = b"helloworld";
    let compressed = table.encode(input);
    let decoded = table.decode(&compressed).expect("decode");
    assert_eq!(decoded, input);

    // 1 byte for 'hello' + 5 × 2 bytes for 'world' escapes = 11 bytes.
    assert_eq!(compressed.len(), 11);
}

#[test]
fn encode_empty_string() {
    let table = SymbolTable::default();
    assert!(table.encode(b"").is_empty());
    assert_eq!(table.decode(&[]).expect("decode"), b"");
}

#[test]
fn encode_with_no_symbols_uses_only_escapes() {
    let table = SymbolTable::default();
    let input = b"abc";
    let compressed = table.encode(input);
    // 3 input bytes × 2 (escape + literal) = 6 bytes.
    assert_eq!(compressed, vec![0, b'a', 0, b'b', 0, b'c']);
    assert_eq!(table.decode(&compressed).expect("decode"), input);
}

#[test]
fn decode_rejects_truncated_escape() {
    let table = SymbolTable::default();
    // Trailing escape byte with no literal following.
    assert!(matches!(
        table.decode(&[0]),
        Err(FsstError::TruncatedEscape(0))
    ));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: FAIL — `no method named 'encode'`.

- [ ] **Step 3: Write the implementation**

Add to `impl SymbolTable`:

```rust
    /// Encodes `input` to a byte stream using this symbol table.
    ///
    /// At each position the longest matching symbol is emitted as a single
    /// code byte; bytes with no matching symbol are encoded as `[ESCAPE, byte]`.
    #[must_use]
    pub fn encode(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0usize;
        while i < input.len() {
            match self.longest_match(&input[i..]) {
                Some((code, len)) => {
                    out.push(code);
                    i += len;
                }
                None => {
                    out.push(ESCAPE);
                    out.push(input[i]);
                    i += 1;
                }
            }
        }
        out
    }

    /// Decodes a compressed byte stream produced by [`Self::encode`].
    ///
    /// # Errors
    /// Returns [`FsstError::TruncatedEscape`] if the stream ends mid-escape
    /// (a trailing `ESCAPE` byte with no following literal).
    pub fn decode(&self, compressed: &[u8]) -> Result<Vec<u8>, FsstError> {
        let mut out = Vec::with_capacity(compressed.len());
        let mut i = 0usize;
        while i < compressed.len() {
            let c = compressed[i];
            if c == ESCAPE {
                let literal = *compressed
                    .get(i + 1)
                    .ok_or(FsstError::TruncatedEscape(i))?;
                out.push(literal);
                i += 2;
            } else if let Some(sym) = self.symbol(c) {
                out.extend_from_slice(sym);
                i += 1;
            } else {
                // Absent symbol — treat as a literal of the code byte itself.
                // This branch is unreachable for streams produced by
                // `encode` on this same table, but a corrupt or mismatched
                // table could otherwise crash the decoder.
                out.push(c);
                i += 1;
            }
        }
        Ok(out)
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — 15 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs
git commit -m "feat(fsst): encode/decode with escape mechanism"
```

---

## Task 5: `FsstCodec` — build a set of strings with random-access decode

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs`, `crates/grafeo-core/src/codec/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn fsst_codec_round_trip_random_access() {
    let strings = vec![
        b"the quick brown fox".to_vec(),
        b"jumps over the lazy dog".to_vec(),
        b"".to_vec(),
        b"the".to_vec(),
        b"the cat sat on the mat".to_vec(),
    ];
    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let codec = FsstCodec::build(&refs);

    assert_eq!(codec.len(), 5);
    for (i, s) in strings.iter().enumerate() {
        let decoded = codec.get(i).expect("get").expect("decode");
        assert_eq!(&decoded, s, "string {i} mismatched");
    }
    // Out-of-bounds returns Ok(None).
    assert!(codec.get(5).expect("get").is_none());
}

#[test]
fn fsst_codec_handles_all_empty() {
    let refs: Vec<&[u8]> = vec![b"", b"", b""];
    let codec = FsstCodec::build(&refs);
    assert_eq!(codec.len(), 3);
    for i in 0..3 {
        assert_eq!(codec.get(i).expect("get").expect("decode"), Vec::<u8>::new());
    }
}

#[test]
fn fsst_codec_handles_no_strings() {
    let refs: Vec<&[u8]> = vec![];
    let codec = FsstCodec::build(&refs);
    assert_eq!(codec.len(), 0);
    assert!(codec.is_empty());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: FAIL — `cannot find type 'FsstCodec'`.

- [ ] **Step 3: Write the implementation**

Add to `fsst.rs`, after `impl SymbolTable`:

```rust
/// A compressed set of strings with O(1) random-access decode.
///
/// Internally stores a [`SymbolTable`] trained on the input, a flat
/// `compressed` byte stream of all strings concatenated, and an
/// `offsets` array where `offsets[k]..offsets[k+1]` is the byte range
/// of string `k` in `compressed`.
#[derive(Debug, Clone, PartialEq)]
pub struct FsstCodec {
    table: SymbolTable,
    /// Concatenated compressed strings.
    compressed: Vec<u8>,
    /// One entry per string + one trailing total-length entry.
    /// `offsets[k]..offsets[k+1]` is string `k`'s byte range.
    offsets: Vec<u32>,
}

impl FsstCodec {
    /// Builds a codec from a slice of byte strings.
    ///
    /// Trains a [`SymbolTable`] on the input and compresses each string in
    /// turn, recording its start offset for random access.
    #[must_use]
    pub fn build(strings: &[&[u8]]) -> Self {
        let table = SymbolTable::train(strings);
        let mut compressed = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(strings.len() + 1);
        for s in strings {
            // reason: compressed size is bounded by 2 × total input bytes; for
            // any practical workload this stays well under u32::MAX (4 GiB).
            #[allow(clippy::cast_possible_truncation)]
            offsets.push(compressed.len() as u32);
            compressed.extend(table.encode(s));
        }
        // reason: same bound as above
        #[allow(clippy::cast_possible_truncation)]
        offsets.push(compressed.len() as u32);
        Self {
            table,
            compressed,
            offsets,
        }
    }

    /// Reconstructs a codec from its parts (blob deserialization).
    ///
    /// # Panics
    /// Panics if `offsets` is empty or the last offset exceeds
    /// `compressed.len()`.
    #[must_use]
    pub(crate) fn from_parts(
        table: SymbolTable,
        compressed: Vec<u8>,
        offsets: Vec<u32>,
    ) -> Self {
        assert!(
            !offsets.is_empty(),
            "offsets must contain at least the terminal length"
        );
        assert!(
            *offsets.last().unwrap() as usize <= compressed.len(),
            "terminal offset must not exceed compressed length"
        );
        Self {
            table,
            compressed,
            offsets,
        }
    }

    /// Number of strings stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// True if no strings are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decodes string `index`, or `Ok(None)` if `index` is out of bounds.
    ///
    /// # Errors
    /// Returns [`FsstError`] if the stored bytes are malformed (e.g., a
    /// trailing escape with no literal).
    pub fn get(&self, index: usize) -> Result<Option<Vec<u8>>, FsstError> {
        if index >= self.len() {
            return Ok(None);
        }
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        Ok(Some(self.table.decode(&self.compressed[start..end])?))
    }

    /// Accessors used by blob serialization.
    #[must_use]
    pub(crate) fn parts(&self) -> (&SymbolTable, &[u8], &[u32]) {
        (&self.table, &self.compressed, &self.offsets)
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — 18 tests.

- [ ] **Step 5: Add public re-exports to `codec/mod.rs`**

In `crates/grafeo-core/src/codec/mod.rs`, after the existing `pub use dictionary::{DictionaryBuilder, DictionaryEncoding};`, add:

```rust
pub use fsst::{FsstCodec, FsstError, SymbolTable};
```

Run: `cargo build -p grafeo-core` — expected clean (the `pub(crate) from_parts`/`parts` produce `dead_code` warnings until Task 6 consumes them).

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs crates/grafeo-core/src/codec/mod.rs
git commit -m "feat(fsst): FsstCodec with random-access decode"
```

---

## Task 6: Blob `to_bytes` / `from_bytes` — zero-copy layout contract

This honours the Plan 2 cross-cutting zero-copy contract — same shape as sub-plan 2a Task 7. Fixed header with `u64` offsets, naturally-aligned arrays, CRC32 trailer, and a `from_bytes_shared(bytes::Bytes)` overload.

**Files:**
- Modify: `crates/grafeo-core/src/codec/fsst.rs`

Wire format (all little-endian):

```text
offset 0   "GFST"               magic (4 bytes)
       4   version = 1          u8
       5   flags = 0            u8
       6   padding              u16
       8   count                u32   (number of strings; 0 allowed)
      12   compressed_len       u32   (bytes in compressed stream)
      16   table_offset         u64   (NEW — Plan 2 rule 3)
      24   offsets_offset       u64
      32   compressed_offset    u64
      40   reserved             u64   (zero; aligns next section to 8)
      48   symbol table: 256 × (length u8) + 256 × [u8; 8] = 256 + 2048 = 2304 bytes
           ... pad to 8
           offsets: (count + 1) × u32 + pad to 8
           compressed: compressed_len bytes + pad to 4
           crc32                u32
```

The symbol table is stored as two parallel arrays (256 length bytes, then 256×8 body bytes) so the body region is naturally u8-aligned; reading is sequential.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn fsst_blob_round_trip_preserves_strings() {
    let strings = vec![
        b"alpha".to_vec(),
        b"beta gamma delta".to_vec(),
        b"".to_vec(),
        b"alpha beta gamma".to_vec(),
        b"the quick brown fox jumps over the lazy dog".to_vec(),
    ];
    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let codec = FsstCodec::build(&refs);
    let blob = codec.to_bytes();

    assert_eq!(&blob[0..4], b"GFST");
    assert_eq!(blob[4], 1);

    let reopened = FsstCodec::from_bytes(&blob).expect("from_bytes");
    assert_eq!(reopened.len(), codec.len());
    for (i, s) in strings.iter().enumerate() {
        assert_eq!(&reopened.get(i).expect("get").expect("decode"), s);
    }
}

#[test]
fn fsst_blob_rejects_bad_magic_and_crc() {
    let codec = FsstCodec::build(&[b"hi", b"bye"]);
    let mut blob = codec.to_bytes();

    let mut bad_magic = blob.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        FsstCodec::from_bytes(&bad_magic),
        Err(FsstError::BadMagic)
    ));

    let mid = blob.len() / 2;
    blob[mid] ^= 0xFF;
    assert!(matches!(
        FsstCodec::from_bytes(&blob),
        Err(FsstError::CrcMismatch { .. })
    ));
}

#[test]
fn fsst_blob_from_bytes_shared_round_trip() {
    let codec = FsstCodec::build(&[b"hello world"]);
    let blob = bytes::Bytes::from(codec.to_bytes());
    let reopened = FsstCodec::from_bytes_shared(blob).expect("from_bytes_shared");
    assert_eq!(reopened.get(0).expect("get").expect("decode"), b"hello world");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: FAIL — `no method named 'to_bytes'`.

- [ ] **Step 3: Write the implementation**

Add to `fsst.rs`, at module level (just before the `#[cfg(test)] mod tests` block):

```rust
/// Current FSST blob format version.
const BLOB_VERSION: u8 = 1;

/// Appends zero bytes until `buf.len()` is a multiple of `align`.
fn pad_to(buf: &mut Vec<u8>, align: usize) {
    while !buf.len().is_multiple_of(align) {
        buf.push(0);
    }
}

/// Reads a little-endian `u32` at `*pos`, advancing `*pos`.
fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, FsstError> {
    let end = *pos + 4;
    let slice = buf.get(*pos..end).ok_or(FsstError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, FsstError> {
    let end = *pos + 8;
    let slice = buf.get(*pos..end).ok_or(FsstError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

impl FsstCodec {
    /// Serializes the codec to a self-describing, position-independent blob.
    ///
    /// The layout honours the Plan 2 zero-copy contract: a fixed 48-byte
    /// header with blob-relative `u64` section offsets, naturally-aligned
    /// arrays, and a trailing CRC32. See [`BLOB_VERSION`] for the version.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let (table, compressed, offsets) = self.parts();
        // reason: counts/sizes bounded well below u32::MAX in practice
        #[allow(clippy::cast_possible_truncation)]
        let count = (offsets.len().saturating_sub(1)) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let compressed_len = compressed.len() as u32;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GFST");
        buf.push(BLOB_VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // padding
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&compressed_len.to_le_bytes());

        // Four u64s: three offsets + one reserved (zero), patched after assembly.
        let offsets_pos = buf.len(); // = 16
        buf.extend_from_slice(&[0u8; 32]);

        // Symbol table: 256 length bytes, then 256 × 8 body bytes.
        let table_offset = buf.len() as u64;
        for code in 0u8..=255 {
            buf.push(table.lengths[code as usize]);
        }
        for code in 0u8..=255 {
            buf.extend_from_slice(&table.bodies[code as usize]);
        }
        pad_to(&mut buf, 8);

        let offsets_section_offset = buf.len() as u64;
        for &o in offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        let compressed_section_offset = buf.len() as u64;
        buf.extend_from_slice(compressed);
        pad_to(&mut buf, 4);

        // Patch the three offsets.
        buf[offsets_pos..offsets_pos + 8]
            .copy_from_slice(&table_offset.to_le_bytes());
        buf[offsets_pos + 8..offsets_pos + 16]
            .copy_from_slice(&offsets_section_offset.to_le_bytes());
        buf[offsets_pos + 16..offsets_pos + 24]
            .copy_from_slice(&compressed_section_offset.to_le_bytes());
        // The fourth u64 stays zero (reserved for future format additions).

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`FsstError`] on a bad magic, unsupported version, truncation,
    /// CRC mismatch, or a malformed symbol table.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, FsstError> {
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

        debug_assert_eq!(pos, 48, "header end at offset 48");

        // Symbol table.
        let lengths_slice = buf
            .get(table_offset..table_offset + 256)
            .ok_or(FsstError::Truncated {
                need: table_offset + 256,
                have: buf.len(),
            })?;
        let bodies_slice = buf
            .get(table_offset + 256..table_offset + 256 + 256 * MAX_SYMBOL_LEN)
            .ok_or(FsstError::Truncated {
                need: table_offset + 256 + 256 * MAX_SYMBOL_LEN,
                have: buf.len(),
            })?;
        let mut table = SymbolTable::default();
        for (code, &len_byte) in lengths_slice.iter().enumerate() {
            // reason: code 0..=255 fits u8 by construction
            #[allow(clippy::cast_possible_truncation)]
            let code_u8 = code as u8;
            if code_u8 == ESCAPE {
                // Skip the escape slot — its length must be 0.
                if len_byte != 0 {
                    return Err(FsstError::BadSymbolLength {
                        code: code_u8,
                        length: len_byte,
                    });
                }
                continue;
            }
            if len_byte == 0 {
                continue; // Empty slot.
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

        // Offsets.
        let offsets_byte_end = offsets_offset + 4 * (count + 1);
        let offsets_bytes = buf
            .get(offsets_offset..offsets_byte_end)
            .ok_or(FsstError::Truncated {
                need: offsets_byte_end,
                have: buf.len(),
            })?;
        let mut offsets = Vec::with_capacity(count + 1);
        for chunk in offsets_bytes.chunks_exact(4) {
            offsets.push(u32::from_le_bytes(chunk.try_into().expect("4 bytes")));
        }

        // Compressed stream.
        let compressed_end = compressed_offset + compressed_len;
        let compressed = buf
            .get(compressed_offset..compressed_end)
            .ok_or(FsstError::Truncated {
                need: compressed_end,
                have: buf.len(),
            })?
            .to_vec();

        // Validate offsets monotonically increase and stay within compressed.
        for (i, w) in offsets.windows(2).enumerate() {
            if w[1] < w[0] || (w[1] as u64) > compressed_len as u64 {
                return Err(FsstError::BadOffset {
                    index: i,
                    offset: w[1] as u64,
                    len: compressed_len as u64,
                });
            }
        }

        Ok(Self::from_parts(table, compressed, offsets))
    }

    /// Opens a blob shared via [`bytes::Bytes`].
    ///
    /// Provided so callers can pass an mmap-backed `Bytes` region without
    /// copying it through `&[u8]` first. The current implementation parses
    /// into owned `Vec`s (one copy); sub-plan 2d adds a borrowing
    /// `FsstView` over the same wire format that holds `Bytes` slices of
    /// the compressed stream and offsets array directly.
    ///
    /// # Errors
    /// Same as [`Self::from_bytes`].
    pub fn from_bytes_shared(blob: bytes::Bytes) -> Result<Self, FsstError> {
        Self::from_bytes(&blob)
    }
}
```

Note: the `from_bytes` path accesses `table.lengths` and `table.bodies` directly — both fields are crate-private. Since this code lives in the same module, direct access works.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib codec::fsst`
Expected: PASS — 21 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/codec/fsst.rs
git commit -m "feat(fsst): zero-copy-layout blob serialization with crc"
```

---

## Task 7: Round-trip equivalence proptest

**Files:**
- Create: `crates/grafeo-core/tests/fsst_round_trip.rs`

- [ ] **Step 1: Write the test**

Create `crates/grafeo-core/tests/fsst_round_trip.rs`:

```rust
//! Round-trip equivalence regression test for the FSST string codec.
//!
//! Mirrors the fork's property-test discipline (see
//! `grafeo-engine/tests/compact_roundtrip_proptest.rs`): a `proptest!`
//! block generating arbitrary string sets, plus fixed-seed regression
//! cases. The invariants:
//!
//! 1. Every input string decodes bit-identical to its original.
//! 2. Random access (`codec.get(i)`) equals sequential decode.
//! 3. The blob round-trips: `from_bytes(codec.to_bytes())` yields a codec
//!    with the same `get` results.
//!
//! ```bash
//! cargo test -p grafeo-core --test fsst_round_trip
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test fsst_round_trip
//! ```

use grafeo_core::codec::FsstCodec;
use proptest::prelude::*;

fn string_strategy() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..=64)
}

fn string_set_strategy() -> impl Strategy<Value = Vec<Vec<u8>>> {
    proptest::collection::vec(string_strategy(), 0..=24)
}

fn check_round_trip(strings: &[Vec<u8>]) {
    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let codec = FsstCodec::build(&refs);

    // 1. Every string decodes bit-identical to its original.
    assert_eq!(codec.len(), strings.len());
    for (i, s) in strings.iter().enumerate() {
        let decoded = codec.get(i).expect("get").expect("decode");
        assert_eq!(&decoded, s, "string {i} decoded mismatched");
    }

    // 2. Blob round-trip preserves all results.
    let blob = codec.to_bytes();
    let reopened = FsstCodec::from_bytes(&blob).expect("from_bytes");
    assert_eq!(reopened.len(), strings.len());
    for (i, s) in strings.iter().enumerate() {
        let decoded = reopened.get(i).expect("get").expect("decode");
        assert_eq!(&decoded, s, "blob-reopened string {i} mismatched");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every input string decodes bit-identical, before and after a blob
    /// round-trip, for arbitrary byte strings.
    #[test]
    fn fsst_round_trip_arbitrary_strings(strings in string_set_strategy()) {
        check_round_trip(&strings);
    }
}

// ── Fixed regression seeds ───────────────────────────────────────

#[test]
fn fsst_round_trip_empty_set() {
    check_round_trip(&[]);
}

#[test]
fn fsst_round_trip_all_empty_strings() {
    check_round_trip(&[vec![], vec![], vec![]]);
}

#[test]
fn fsst_round_trip_all_bytes_0_to_255() {
    let s: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
    check_round_trip(&[s]);
}

#[test]
fn fsst_round_trip_realistic_names() {
    let names = [
        "Vincent Vega",
        "Mia Wallace",
        "Butch Coolidge",
        "Jules Winnfield",
        "Marsellus Wallace",
        "Vincent Vega",  // duplicate
        "Honey Bunny",
        "Pumpkin",
    ];
    let strings: Vec<Vec<u8>> = names.iter().map(|s| s.as_bytes().to_vec()).collect();
    check_round_trip(&strings);
}

#[test]
fn fsst_round_trip_long_string_with_repeats() {
    let mut s = Vec::new();
    for _ in 0..64 {
        s.extend_from_slice(b"the ");
    }
    s.extend_from_slice(b"END");
    check_round_trip(&[s]);
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p grafeo-core --test fsst_round_trip`
Expected: PASS — 1 proptest with 128 cases + 5 fixed-seed tests = 6 PASS.

Also run `PROPTEST_CASES=512 cargo test -p grafeo-core --test fsst_round_trip` and confirm — round-trip equivalence is an absolute property (no statistical floor), so any failure indicates a real codec bug.

- [ ] **Step 3: Commit**

```bash
git add crates/grafeo-core/tests/fsst_round_trip.rs
git commit -m "test(fsst): round-trip equivalence proptest"
```

---

## Task 8: `ColumnCodec::Fsst` integration

This adds FSST as a new string-column variant alongside `Dict(DictionaryEncoding)`. The `ColumnCodec` enum is `#[non_exhaustive]`, so adding a variant is a non-breaking change. We wire up `get`, `len`, and the memory accounting path; other paths (`find_eq`, `find_in_range`, `compare_values`) fall back to the per-row decode via `get`.

**Files:**
- Modify: `crates/grafeo-core/src/graph/compact/column.rs`

- [ ] **Step 1: Write the failing test**

Add a new test inside the existing `#[cfg(test)] mod tests` block at the bottom of `column.rs`:

```rust
    #[test]
    fn column_codec_fsst_round_trip() {
        use crate::codec::FsstCodec;

        let strings: Vec<&[u8]> = vec![
            b"Vincent",
            b"Mia",
            b"Vincent",
            b"Butch",
        ];
        let codec = FsstCodec::build(&strings);
        let col = ColumnCodec::Fsst(codec);

        assert_eq!(col.len(), 4);
        assert_eq!(col.get(0), Some(Value::String(ArcStr::from("Vincent"))));
        assert_eq!(col.get(1), Some(Value::String(ArcStr::from("Mia"))));
        assert_eq!(col.get(2), Some(Value::String(ArcStr::from("Vincent"))));
        assert_eq!(col.get(3), Some(Value::String(ArcStr::from("Butch"))));
        assert_eq!(col.get(4), None);

        // find_eq falls back to scanning via get; verify it still finds duplicates.
        let hits = col.find_eq(&Value::String("Vincent".into()));
        assert_eq!(hits, vec![0, 2]);
    }
```

(The exact module-path import of `FsstCodec` depends on the existing import block at the top of the test module. If `use crate::codec::*;` is in scope, drop the explicit `use` line.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib graph::compact::column`
Expected: FAIL — `no variant 'Fsst'` (or `cannot find FsstCodec`).

- [ ] **Step 3: Add the variant and wire up the necessary paths**

In `crates/grafeo-core/src/graph/compact/column.rs`:

1. At the top of the file, alongside the existing `use crate::codec::{...}` line at line 13, add `FsstCodec`:
   ```rust
   use crate::codec::{BitPackedInts, BitVector, BlockEntry, DictionaryEncoding, FsstCodec};
   ```

2. In the `pub enum ColumnCodec` definition (line 227), add a new variant right after `Dict(DictionaryEncoding)`:
   ```rust
       /// FSST-compressed strings with O(1) random-access decode.
       /// See [`crate::codec::FsstCodec`]. Sits alongside [`Self::Dict`]; pick
       /// FSST for high-cardinality string columns where dictionary encoding
       /// loses to per-string compression.
       Fsst(FsstCodec),
   ```

3. In the `pub fn get(&self, index: usize) -> Option<Value>` method (around line 379), add a new match arm right after the existing `Self::Dict(dict) => ...` arm:
   ```rust
           Self::Fsst(fsst) => fsst
               .get(index)
               .ok()
               .flatten()
               .and_then(|bytes| std::str::from_utf8(&bytes).ok().map(|s| Value::String(ArcStr::from(s)))),
   ```

4. Locate the `pub fn len(&self) -> usize` method (find it via `grep -n 'pub fn len' crates/grafeo-core/src/graph/compact/column.rs`). Add a new match arm for `Self::Fsst(fsst) => fsst.len(),` in the same style as the existing arms.

5. Locate the `pub fn memory_bytes(&self) -> usize` method (find via `grep -n memory_bytes crates/grafeo-core/src/graph/compact/column.rs`). Add a new match arm for `Self::Fsst(fsst) => { let (_, c, o) = fsst.parts(); c.len() + o.len() * 4 + std::mem::size_of::<crate::codec::SymbolTable>() },` — adjusted to match the existing helper's signature (sum the compressed bytes, the offsets array, and the symbol table size).

6. Locate `find_eq`, `find_in_range`, and any other match-on-`ColumnCodec` site that needs to handle `Fsst`. If the existing `_ => ...` catch-all delegates to `self.get(index)` for fallback, the new variant is covered automatically. For `find_eq` in particular, check whether the existing match returns early for `Self::Dict(...)` (fast-path via `DictionaryEncoding::encode`) — if so, the `Fsst` arm should fall through to a linear scan using `self.get(i)` over `0..self.len()`. Use the existing pattern from `BitPacked` or `RawI64` fall-through as a template.

   If the catch-all `_ => ...` already exists, no change is needed for `find_eq` — the test asserts the fallback works. If the existing match is exhaustive (no `_ =>`), add an `Self::Fsst(_) => { ... }` arm that scans by index. The test in Step 1 will tell you whether the fallback works or you need to add the arm.

   Note: if Rust complains about a non-exhaustive match, the existing `_ =>` is missing for some method — add the arm explicitly to avoid silently widening unrelated codepaths.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib graph::compact::column::tests::column_codec_fsst_round_trip`
Expected: PASS.

Then run the broader compact-store tests to confirm no regressions:
```bash
cargo test -p grafeo-core --features compact-store --lib graph::compact
cargo test -p grafeo-engine --features "compact-store lpg gql" --test compact_roundtrip_proptest
```
Expected: both still pass.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/graph/compact/column.rs
git commit -m "feat(compact): ColumnCodec::Fsst variant for FSST-compressed string columns"
```

---

## Task 9: WASM bindings — `FsstCodec`

**Files:**
- Modify: `crates/bindings/wasm/src/codecs.rs`
- Modify: `crates/bindings/wasm/Cargo.toml`
- Modify: `crates/bindings/wasm/tests/web.rs`

- [ ] **Step 1: Add the `fsst-codec` feature**

In `crates/bindings/wasm/Cargo.toml`, after the existing `rabitq-codec = ["dep:grafeo-core"]` line, add a sibling feature:

```toml
fsst-codec = ["dep:grafeo-core"]
```

This keeps the two codecs independently selectable, matching the `rabitq-codec` precedent.

In `crates/bindings/wasm/src/lib.rs`, change the existing module declaration `#[cfg(feature = "rabitq-codec")] pub mod codecs;` to:

```rust
#[cfg(any(feature = "rabitq-codec", feature = "fsst-codec"))]
pub mod codecs;
```

This way `codecs.rs` compiles whenever EITHER feature is enabled; individual types inside `codecs.rs` gate themselves on the specific feature.

In `crates/bindings/wasm/src/codecs.rs`, wrap the existing `RabitqCodec` struct and its impl block in `#[cfg(feature = "rabitq-codec")]` if they are not already so gated. Then append the FSST binding (Step 3).

- [ ] **Step 2: Write the failing test**

Append to `crates/bindings/wasm/tests/web.rs`:

```rust
#[cfg(feature = "fsst-codec")]
#[wasm_bindgen_test]
fn fsst_codec_encode_open_get_round_trip() {
    use grafeo_wasm::codecs::FsstCodec;

    let strings = ["alpha", "beta gamma", "alpha", "the quick brown fox"];
    let lengths: Vec<u32> = strings.iter().map(|s| s.len() as u32).collect();
    let mut flat: Vec<u8> = Vec::new();
    for s in &strings {
        flat.extend_from_slice(s.as_bytes());
    }

    let blob = FsstCodec::encode(&flat, &lengths).expect("encode");
    assert_eq!(&blob[0..4], b"GFST");

    let codec = FsstCodec::open(&blob).expect("open");
    assert_eq!(codec.len() as usize, strings.len());
    for (i, expected) in strings.iter().enumerate() {
        let decoded_bytes = codec.get(i as u32).expect("get");
        let decoded_str = std::str::from_utf8(&decoded_bytes).expect("utf-8");
        assert_eq!(decoded_str, *expected, "string {i} mismatched");
    }
}
```

- [ ] **Step 3: Run to verify it fails**

```bash
wasm-pack test --node crates/bindings/wasm --features fsst-codec -- fsst
```
Expected: FAIL — `cannot find type 'FsstCodec'` in `grafeo_wasm::codecs`.

If `wasm-pack` is unavailable, fall back to `cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features fsst-codec` — should fail to compile.

- [ ] **Step 4: Write the implementation**

Append to `crates/bindings/wasm/src/codecs.rs`:

```rust
/// A JS-facing handle to an FSST string codec.
#[cfg(feature = "fsst-codec")]
#[wasm_bindgen]
pub struct FsstCodec {
    inner: grafeo_core::codec::FsstCodec,
}

#[cfg(feature = "fsst-codec")]
#[wasm_bindgen]
impl FsstCodec {
    /// Encodes a batch of strings into a compressed blob.
    ///
    /// `flat` is the concatenation of all string bodies; `lengths` gives
    /// the length of each string in bytes. Strings are NOT separated by
    /// any delimiter — they are reconstructed from `lengths`. The empty
    /// string is permitted (length 0). Returns the blob bytes.
    ///
    /// # Errors
    /// Returns a `JsError` if `sum(lengths) != flat.length`.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode(flat: &[u8], lengths: &[u32]) -> Result<Vec<u8>, JsError> {
        let total: u64 = lengths.iter().map(|&l| u64::from(l)).sum();
        if total != flat.len() as u64 {
            return Err(JsError::new(
                "sum of lengths must equal flat.length",
            ));
        }
        let mut strings: Vec<&[u8]> = Vec::with_capacity(lengths.len());
        let mut cursor = 0usize;
        for &len in lengths {
            let len = len as usize;
            strings.push(&flat[cursor..cursor + len]);
            cursor += len;
        }
        Ok(grafeo_core::codec::FsstCodec::build(&strings).to_bytes())
    }

    /// Opens a blob produced by [`FsstCodec::encode`] for querying.
    ///
    /// # Errors
    /// Returns a `JsError` if the blob is malformed.
    #[wasm_bindgen(js_name = "open")]
    pub fn open(blob: &[u8]) -> Result<FsstCodec, JsError> {
        let inner = grafeo_core::codec::FsstCodec::from_bytes(blob)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Decodes string `index`. Returns an empty `Vec` for an empty string;
    /// returns an empty `Vec` and the caller must check `len()` to distinguish
    /// out-of-bounds — JS callers should check `index < codec.len()` first.
    #[wasm_bindgen(js_name = "get")]
    #[must_use]
    pub fn get(&self, index: u32) -> Vec<u8> {
        self.inner
            .get(index as usize)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// Number of stored strings.
    #[wasm_bindgen(js_name = "len")]
    #[must_use]
    pub fn len(&self) -> u32 {
        // reason: count of strings fits u32 for any practical snapshot
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.len() as u32
        }
    }

    /// True if the codec holds no strings.
    #[wasm_bindgen(js_name = "isEmpty")]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
```

- [ ] **Step 5: Run to verify it passes**

```bash
wasm-pack test --node crates/bindings/wasm --features fsst-codec -- fsst
```
Expected: PASS — `fsst_codec_encode_open_get_round_trip`.

Also confirm the existing rabitq test still passes with both features:
```bash
wasm-pack test --node crates/bindings/wasm --features rabitq-codec -- rabitq
wasm-pack test --node crates/bindings/wasm --features "rabitq-codec fsst-codec"
```

- [ ] **Step 6: Commit**

```bash
git add crates/bindings/wasm/src/codecs.rs crates/bindings/wasm/src/lib.rs crates/bindings/wasm/Cargo.toml crates/bindings/wasm/tests/web.rs
git commit -m "feat(wasm): FsstCodec binding for encode/open/get"
```

---

## Task 10: CHANGELOG and full verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the CHANGELOG entry**

In `CHANGELOG.md`, under the same `## [Unreleased] / ### Added` section that sub-plan 2a added to, append:

```markdown
- `codec::fsst` — Fast Static Symbol Table string compression codec with O(1) random-access decode, a new `ColumnCodec::Fsst` variant for compact-store string columns, and an `FsstCodec` WASM binding (behind the `fsst-codec` feature on the wasm crate).
```

- [ ] **Step 2: Run the full verification suite**

Run each, confirming output before claiming completion (use `superpowers:verification-before-completion`):

```bash
cargo test -p grafeo-core --lib codec::fsst
cargo test -p grafeo-core --test fsst_round_trip
cargo test -p grafeo-core --features compact-store --lib graph::compact::column
cargo test -p grafeo-engine --features "compact-store lpg gql" --test compact_roundtrip_proptest
cargo clippy -p grafeo-core --all-targets -- -D warnings
cargo build -p grafeo-core --target wasm32-unknown-unknown
wasm-pack test --node crates/bindings/wasm --features "rabitq-codec fsst-codec"
```

Expected: all green. If `wasm-pack` is unavailable, the wasm32 build above plus `cargo build -p grafeo-wasm --target wasm32-unknown-unknown --features "rabitq-codec fsst-codec"` is the fallback.

If any step fails, STOP and report. Do not paper over.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note FSST string codec"
```

---

## Self-Review (completed by the plan author)

**Spec coverage** — every Plan 2c requirement maps to a task:
- 256-entry symbol table with escape mechanism → Tasks 1, 2.
- Training from a sample → Task 3.
- Random-access decode of individual strings → Tasks 4, 5 (`FsstCodec::get`).
- Zero-copy blob layout (fixed header, `u64` offsets, `bytes::Bytes` overload, CRC) → Task 6 (matches sub-plan 2a Task 7's pattern).
- Round-trip equivalence proptest (every input decodes bit-identical, random access matches sequential) → Task 7.
- Integration as `ColumnCodec::Fsst` variant → Task 8.
- WASM bindings → Task 9.
- CHANGELOG + full verification → Task 10.

**Type consistency** — `SymbolTable` (`set`, `symbol`, `longest_match`, `train`, `len`, `is_empty`, `lengths`/`bodies` fields), `FsstCodec` (`build`, `from_parts`, `parts`, `len`, `is_empty`, `get`, `to_bytes`, `from_bytes`, `from_bytes_shared`), `FsstError` variants (`Truncated`, `BadMagic`, `BadVersion`, `CrcMismatch`, `BadSymbolLength`, `BadOffset`, `TruncatedEscape`), constants `ESCAPE` and `MAX_SYMBOL_LEN` — all consistent across tasks. Pub vs `pub(crate)` decisions match: `from_parts`/`parts` are `pub(crate)` (used only by Task 6 serialization).

**Placeholder scan** — every code step shows complete code; the only "find via grep" steps are in Task 8 where the file is too large (3813 lines) to dictate exact line numbers reliably, and the engineer can locate the methods by name in seconds.

**Known follow-ups (out of scope for 2c, tracked in the Plan 2 overview):**
- A borrowing `FsstView` over the Task 6 blob layout — sub-plan 2d.
- SIMD-accelerated decode loop (the FSST paper's main perf trick) — a follow-up optimization, not in scope for the codec's correctness.
- More sophisticated training (multi-pass with bigram refinement per the original paper) — the current greedy single-pass training is correct (always produces a usable table) but suboptimal; revisit if the integration benchmark in a later phase shows it matters.
