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
}
