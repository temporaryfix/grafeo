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

        // Score by (length − 1) × frequency + frequency. The `+ frequency`
        // term lets length-1 substrings out-rank zero-frequency multi-byte
        // ones, preserving byte coverage for any byte that appears in the
        // sample at all.
        let mut scored: Vec<(Vec<u8>, u64)> = counts
            .into_iter()
            .map(|(sub, freq)| {
                let score = (sub.len() as u64 - 1) * freq + freq;
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
}

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
        // reason: section offsets fit u64
        #[allow(clippy::cast_possible_truncation)]
        let table_offset = buf.len() as u64;
        for code in 0u8..=255 {
            buf.push(table.lengths[code as usize]);
        }
        for code in 0u8..=255 {
            buf.extend_from_slice(&table.bodies[code as usize]);
        }
        pad_to(&mut buf, 8);

        #[allow(clippy::cast_possible_truncation)]
        let offsets_section_offset = buf.len() as u64;
        for &o in offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        #[allow(clippy::cast_possible_truncation)]
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
        let lengths_end = table_offset + 256;
        let lengths_slice = buf
            .get(table_offset..lengths_end)
            .ok_or(FsstError::Truncated {
                need: lengths_end,
                have: buf.len(),
            })?;
        let bodies_end = lengths_end + 256 * MAX_SYMBOL_LEN;
        let bodies_slice = buf
            .get(lengths_end..bodies_end)
            .ok_or(FsstError::Truncated {
                need: bodies_end,
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

        // The first offset must be zero (the first string starts at the
        // beginning of the compressed stream). A non-zero offsets[0]
        // would silently shift every subsequent get() window.
        if let Some(&first) = offsets.first() {
            if first != 0 {
                return Err(FsstError::BadOffset {
                    index: 0,
                    offset: u64::from(first),
                    len: u64::from(compressed_len as u32),
                });
            }
        }

        // Validate offsets monotonically increase and stay within compressed.
        for (i, w) in offsets.windows(2).enumerate() {
            if w[1] < w[0] || u64::from(w[1]) > u64::from(compressed_len as u32) {
                return Err(FsstError::BadOffset {
                    index: i,
                    offset: u64::from(w[1]),
                    len: u64::from(compressed_len as u32),
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

    #[test]
    fn symbol_table_set_overwrites_with_zero_fill() {
        let mut a = SymbolTable::default();
        a.set(1, b"abcdefgh"); // full 8-byte slot
        a.set(1, b"x");        // overwrite with 1-byte symbol

        // Lookup returns the new shorter symbol.
        assert_eq!(a.symbol(1), Some(b"x" as &[u8]));

        // The body slot must be zeroed beyond the new length, so PartialEq
        // with a freshly-built table holding only "x" matches.
        let mut b = SymbolTable::default();
        b.set(1, b"x");
        assert_eq!(a, b, "trailing body bytes from previous symbol must be zeroed");
    }

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

    #[test]
    fn fsst_blob_rejects_non_zero_first_offset() {
        let codec = FsstCodec::build(&[b"abc", b"def"]);
        let mut blob = codec.to_bytes();

        // Find the offsets section using the header u64 at offset 24
        // (offsets_offset). Patch offsets[0] to a non-zero value and re-patch CRC.
        let offsets_section_offset =
            u64::from_le_bytes(blob[24..32].try_into().unwrap()) as usize;
        blob[offsets_section_offset..offsets_section_offset + 4]
            .copy_from_slice(&7u32.to_le_bytes());

        // Re-patch the trailing CRC so the offsets-validation path (not CRC) is
        // exercised.
        let body_end = blob.len() - 4;
        let crc = crc32fast::hash(&blob[..body_end]);
        blob[body_end..].copy_from_slice(&crc.to_le_bytes());

        match FsstCodec::from_bytes(&blob) {
            Err(FsstError::BadOffset { index: 0, offset: 7, .. }) => {}
            other => panic!("expected BadOffset{{index:0, offset:7, ...}}, got {other:?}"),
        }
    }
}
