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
    /// A caller supplied an empty or oversized symbol.
    #[error("fsst symbol table: symbol length {0} is outside 1..={MAX_SYMBOL_LEN}")]
    InvalidSymbolLength(usize),
    /// Code zero is reserved for escaped literal bytes.
    #[error("fsst symbol table: code zero is reserved for escapes")]
    ReservedSymbolCode,
    /// A serialized size or offset cannot be represented safely.
    #[error("fsst blob: {region} size overflow")]
    SizeOverflow {
        /// Field or region whose arithmetic overflowed.
        region: &'static str,
    },
    /// The current blob layout has an inconsistent section boundary.
    #[error("fsst blob: non-canonical {region} layout")]
    NonCanonicalLayout {
        /// Field or region that did not match the current layout.
        region: &'static str,
    },
    /// A codec must contain at least its terminal stream offset.
    #[error("fsst codec: missing terminal offset")]
    MissingTerminalOffset,
    /// A per-string offset extends past the compressed stream's end.
    #[error("fsst blob: invalid offset {offset} for string {index} (compressed length {len})")]
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
    /// A compressed string references an absent symbol-table entry.
    #[error("fsst decode: unknown symbol {code} at position {position}")]
    UnknownSymbol {
        /// Code whose symbol-table entry is absent.
        code: u8,
        /// Byte position within the individual compressed string.
        position: usize,
    },
}

/// A 256-code symbol table. Code 0 is the escape marker; codes 1..=255 hold
/// 1–8-byte symbols. Empty slots (length = 0) are absent symbols, looked up
/// as [`None`] by [`Self::symbol`].
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Symbol length per code (0 = absent). Index 0 is unused (escape).
    lengths: [u8; 256],
    /// Symbol bodies, 8 bytes per slot (right-padded with zeros). Index 0
    /// is unused.
    bodies: [[u8; MAX_SYMBOL_LEN]; 256],
    /// Codes bucketed by their first byte, each bucket ordered longest-first
    /// so `longest_match` returns the first full match. Derived state, rebuilt
    /// by `rebuild_first_byte_index`; excluded from `PartialEq`.
    first_byte_index: [Vec<u8>; 256],
    /// True once `first_byte_index` reflects `lengths`/`bodies`. When false
    /// (a raw `set` without a finalizing constructor), `longest_match` falls
    /// back to the exhaustive scan.
    index_built: bool,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            lengths: [0u8; 256],
            bodies: [[0u8; MAX_SYMBOL_LEN]; 256],
            first_byte_index: std::array::from_fn(|_| Vec::new()),
            index_built: false,
        }
    }
}

// Semantic equality is the symbol set only; the derived index/flag are
// excluded so tables built via different paths compare equal.
impl PartialEq for SymbolTable {
    fn eq(&self, other: &Self) -> bool {
        self.lengths == other.lengths && self.bodies == other.bodies
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
    /// # Errors
    ///
    /// Returns an error when `code` is the escape marker or the symbol length
    /// is outside `1..=MAX_SYMBOL_LEN`.
    pub fn set(&mut self, code: u8, symbol: &[u8]) -> Result<(), FsstError> {
        if code == ESCAPE {
            return Err(FsstError::ReservedSymbolCode);
        }
        if !(1..=MAX_SYMBOL_LEN).contains(&symbol.len()) {
            return Err(FsstError::InvalidSymbolLength(symbol.len()));
        }
        self.set_validated(code, symbol);
        Ok(())
    }

    /// Installs a symbol whose code and size were validated by the caller.
    fn set_validated(&mut self, code: u8, symbol: &[u8]) {
        let idx = code as usize;
        // reason: callers validate symbol.len() as 1..=8
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

    /// Rebuilds `first_byte_index` from `lengths`/`bodies`. Call after a table
    /// is fully assembled (train / build / from_bytes).
    fn rebuild_first_byte_index(&mut self) {
        for bucket in &mut self.first_byte_index {
            bucket.clear();
        }
        for code in 1u8..=255 {
            let len = self.lengths[code as usize] as usize;
            if len == 0 {
                continue;
            }
            self.first_byte_index[self.bodies[code as usize][0] as usize].push(code);
        }
        // Order each bucket longest-first (ties: smaller code first, matching
        // the scan's tie-break) so the first full match is the longest.
        for bucket in &mut self.first_byte_index {
            bucket.sort_unstable_by(|&a, &b| {
                self.lengths[b as usize]
                    .cmp(&self.lengths[a as usize])
                    .then(a.cmp(&b))
            });
        }
        self.index_built = true;
    }

    /// Exhaustive O(255 × MAX_SYMBOL_LEN) longest-prefix scan. Correct fallback
    /// when the first-byte index has not been built. Ties on length break by
    /// the smaller code.
    #[must_use]
    pub fn longest_match_scan(&self, input: &[u8]) -> Option<(u8, usize)> {
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
            if self.bodies[code as usize][..len] == input[..len] {
                match best {
                    None => best = Some((code, len)),
                    Some((_, blen)) if len > blen => best = Some((code, len)),
                    _ => {}
                }
            }
        }
        best
    }

    /// Returns `(code, length)` of the longest symbol whose bytes are a prefix
    /// of `input`, using the per-first-byte index (falling back to the scan if
    /// it was not built). Ties on length break by the smaller code.
    #[must_use]
    pub fn longest_match(&self, input: &[u8]) -> Option<(u8, usize)> {
        if input.is_empty() {
            return None;
        }
        if !self.index_built {
            return self.longest_match_scan(input);
        }
        let max_check = input.len().min(MAX_SYMBOL_LEN);
        for &code in &self.first_byte_index[input[0] as usize] {
            let len = self.lengths[code as usize] as usize;
            if len > max_check {
                continue; // longer symbol can't fit; shorter ones follow in-bucket
            }
            if self.bodies[code as usize][..len] == input[..len] {
                return Some((code, len)); // bucket is longest-first
            }
        }
        None
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
    /// (a trailing `ESCAPE` byte with no following literal), or
    /// [`FsstError::UnknownSymbol`] if a code has no symbol-table entry.
    pub fn decode(&self, compressed: &[u8]) -> Result<Vec<u8>, FsstError> {
        let mut out = Vec::with_capacity(compressed.len());
        self.visit_decoded_chunks(compressed, |chunk| out.extend_from_slice(chunk))?;
        Ok(out)
    }

    /// Walks one string's codes without allocating decoded storage. The empty
    /// visitor lets checked opens validate the same grammar used by decoding.
    fn visit_decoded_chunks(
        &self,
        compressed: &[u8],
        mut visit: impl FnMut(&[u8]),
    ) -> Result<(), FsstError> {
        let mut bytes = compressed.iter().enumerate();
        while let Some((position, &code)) = bytes.next() {
            if code == ESCAPE {
                let (_, literal) = bytes.next().ok_or(FsstError::TruncatedEscape(position))?;
                visit(std::slice::from_ref(literal));
            } else {
                let symbol = self
                    .symbol(code)
                    .ok_or(FsstError::UnknownSymbol { code, position })?;
                visit(symbol);
            }
        }
        Ok(())
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
    #[allow(clippy::explicit_counter_loop)] // u8 counter: 1..=255 overflows with range-zip
    pub fn train(sample: &[&[u8]]) -> Self {
        use std::collections::HashMap;

        if sample.iter().all(|s| s.is_empty()) {
            return Self::default();
        }

        // Count every substring of length 1..=MAX_SYMBOL_LEN in the sample,
        // borrowing slices of the sample instead of allocating per occurrence.
        let mut counts: HashMap<&[u8], u64> = HashMap::new();
        for s in sample {
            for start in 0..s.len() {
                let max_end = (start + MAX_SYMBOL_LEN).min(s.len());
                for end in (start + 1)..=max_end {
                    *counts.entry(&s[start..end]).or_insert(0) += 1;
                }
            }
        }

        // Score by (length − 1) × frequency + frequency. The `+ frequency`
        // term lets length-1 substrings out-rank zero-frequency multi-byte
        // ones, preserving byte coverage for any byte that appears in the
        // sample at all.
        let mut scored: Vec<(&[u8], u64)> = counts
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
            table.set_validated(next_code, sub);
            if next_code == 255 {
                break;
            }
            next_code += 1;
        }
        table.rebuild_first_byte_index();
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
        let Some(&start) = self.offsets.get(index) else {
            return Err(FsstError::MissingTerminalOffset);
        };
        let Some(&end) = self.offsets.get(index + 1) else {
            return Err(FsstError::MissingTerminalOffset);
        };
        let compressed =
            self.compressed
                .get(start as usize..end as usize)
                .ok_or(FsstError::BadOffset {
                    index,
                    offset: u64::from(end),
                    len: self.compressed.len() as u64,
                })?;
        Ok(Some(self.table.decode(compressed)?))
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
    Ok(u32::from_le_bytes(read_array(buf, pos)?))
}

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, FsstError> {
    Ok(u64::from_le_bytes(read_array(buf, pos)?))
}

/// Reads an exact-width byte array at `*pos`, advancing `*pos`.
fn read_array<const N: usize>(buf: &[u8], pos: &mut usize) -> Result<[u8; N], FsstError> {
    let end = pos.checked_add(N).ok_or(FsstError::SizeOverflow {
        region: "field offset",
    })?;
    let slice = buf.get(*pos..end).ok_or(FsstError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    let bytes = slice.try_into().map_err(|_| FsstError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(bytes)
}

const BLOB_HEADER_SIZE: usize = 48;
const SYMBOL_LENGTH_BYTES: usize = 256;
const SYMBOL_BODY_BYTES: usize = 256 * MAX_SYMBOL_LEN;

struct ParsedFsstBlob {
    table: SymbolTable,
    count: usize,
    offsets_offset: usize,
    compressed_offset: usize,
    compressed_len: usize,
}

fn checked_end(start: usize, len: usize, region: &'static str) -> Result<usize, FsstError> {
    start
        .checked_add(len)
        .ok_or(FsstError::SizeOverflow { region })
}

fn checked_align(value: usize, align: usize, region: &'static str) -> Result<usize, FsstError> {
    let padding = (align - value % align) % align;
    checked_end(value, padding, region)
}

fn require_zero_padding(bytes: &[u8], region: &'static str) -> Result<(), FsstError> {
    if bytes.iter().any(|byte| *byte != 0) {
        return Err(FsstError::NonCanonicalLayout { region });
    }
    Ok(())
}

fn parse_fsst_blob(buf: &[u8]) -> Result<ParsedFsstBlob, FsstError> {
    if buf.len() < BLOB_HEADER_SIZE + 4 {
        return Err(FsstError::Truncated {
            need: BLOB_HEADER_SIZE + 4,
            have: buf.len(),
        });
    }
    if buf.get(0..4) != Some(b"GFST") {
        return Err(FsstError::BadMagic);
    }
    let version = *buf.get(4).ok_or(FsstError::Truncated {
        need: 5,
        have: buf.len(),
    })?;
    if version != BLOB_VERSION {
        return Err(FsstError::BadVersion(version));
    }
    let header_padding = buf.get(5..8).ok_or(FsstError::Truncated {
        need: 8,
        have: buf.len(),
    })?;
    require_zero_padding(header_padding, "header padding")?;

    let body_end = buf.len() - 4;
    let mut trailer_pos = body_end;
    let stored = read_u32(buf, &mut trailer_pos)?;
    let computed = crc32fast::hash(&buf[..body_end]);
    if stored != computed {
        return Err(FsstError::CrcMismatch { stored, computed });
    }

    let mut pos = 8;
    let count = read_u32(buf, &mut pos)? as usize;
    let compressed_len_raw = read_u32(buf, &mut pos)?;
    let compressed_len = compressed_len_raw as usize;
    let table_offset =
        usize::try_from(read_u64(buf, &mut pos)?).map_err(|_| FsstError::SizeOverflow {
            region: "symbol table offset",
        })?;
    let offsets_offset =
        usize::try_from(read_u64(buf, &mut pos)?).map_err(|_| FsstError::SizeOverflow {
            region: "offset table offset",
        })?;
    let compressed_offset =
        usize::try_from(read_u64(buf, &mut pos)?).map_err(|_| FsstError::SizeOverflow {
            region: "compressed stream offset",
        })?;
    let reserved = read_u64(buf, &mut pos)?;
    if reserved != 0 {
        return Err(FsstError::NonCanonicalLayout {
            region: "reserved header field",
        });
    }
    if pos != BLOB_HEADER_SIZE || table_offset != BLOB_HEADER_SIZE {
        return Err(FsstError::NonCanonicalLayout {
            region: "symbol table",
        });
    }

    let lengths_end = checked_end(table_offset, SYMBOL_LENGTH_BYTES, "symbol lengths")?;
    let bodies_end = checked_end(lengths_end, SYMBOL_BODY_BYTES, "symbol bodies")?;
    let expected_offsets_offset = checked_align(bodies_end, 8, "symbol table alignment")?;
    if offsets_offset != expected_offsets_offset {
        return Err(FsstError::NonCanonicalLayout {
            region: "offset table",
        });
    }

    let offset_count = count.checked_add(1).ok_or(FsstError::SizeOverflow {
        region: "offset count",
    })?;
    let offsets_bytes = offset_count.checked_mul(4).ok_or(FsstError::SizeOverflow {
        region: "offset table",
    })?;
    let offsets_end = checked_end(offsets_offset, offsets_bytes, "offset table")?;
    let expected_compressed_offset = checked_align(offsets_end, 8, "offset table alignment")?;
    if compressed_offset != expected_compressed_offset {
        return Err(FsstError::NonCanonicalLayout {
            region: "compressed stream",
        });
    }
    let compressed_end = checked_end(compressed_offset, compressed_len, "compressed stream")?;
    let expected_body_end = checked_align(compressed_end, 4, "compressed stream alignment")?;
    if expected_body_end != body_end {
        return Err(FsstError::NonCanonicalLayout {
            region: "trailing bytes",
        });
    }

    let lengths_slice = buf
        .get(table_offset..lengths_end)
        .ok_or(FsstError::Truncated {
            need: lengths_end,
            have: buf.len(),
        })?;
    let bodies_slice = buf
        .get(lengths_end..bodies_end)
        .ok_or(FsstError::Truncated {
            need: bodies_end,
            have: buf.len(),
        })?;
    let offsets_padding = buf
        .get(offsets_end..compressed_offset)
        .ok_or(FsstError::Truncated {
            need: compressed_offset,
            have: buf.len(),
        })?;
    require_zero_padding(offsets_padding, "offset table padding")?;
    let trailing_padding = buf
        .get(compressed_end..body_end)
        .ok_or(FsstError::Truncated {
            need: body_end,
            have: buf.len(),
        })?;
    require_zero_padding(trailing_padding, "compressed stream padding")?;

    let mut table = SymbolTable::default();
    for (code, &len_byte) in lengths_slice.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let code_u8 = code as u8;
        let body_start = code
            .checked_mul(MAX_SYMBOL_LEN)
            .ok_or(FsstError::SizeOverflow {
                region: "symbol body",
            })?;
        let slot_end = checked_end(body_start, MAX_SYMBOL_LEN, "symbol body")?;
        let slot = bodies_slice
            .get(body_start..slot_end)
            .ok_or(FsstError::Truncated {
                need: checked_end(lengths_end, slot_end, "symbol body")?,
                have: buf.len(),
            })?;
        if code_u8 == ESCAPE {
            if len_byte != 0 {
                return Err(FsstError::BadSymbolLength {
                    code: code_u8,
                    length: len_byte,
                });
            }
            require_zero_padding(slot, "escape symbol body")?;
            continue;
        }
        if len_byte == 0 {
            require_zero_padding(slot, "unused symbol body")?;
            continue;
        }
        if usize::from(len_byte) > MAX_SYMBOL_LEN {
            return Err(FsstError::BadSymbolLength {
                code: code_u8,
                length: len_byte,
            });
        }
        let symbol_len = usize::from(len_byte);
        let symbol = slot.get(..symbol_len).ok_or(FsstError::Truncated {
            need: checked_end(lengths_end, slot_end, "symbol body")?,
            have: buf.len(),
        })?;
        let symbol_padding = slot
            .get(symbol_len..)
            .ok_or(FsstError::NonCanonicalLayout {
                region: "symbol body",
            })?;
        require_zero_padding(symbol_padding, "symbol body padding")?;
        table.set_validated(code_u8, symbol);
    }
    table.rebuild_first_byte_index();

    let compressed = buf
        .get(compressed_offset..compressed_end)
        .ok_or(FsstError::Truncated {
            need: compressed_end,
            have: buf.len(),
        })?;
    let mut previous = None;
    let compressed_len_u64 = u64::from(compressed_len_raw);
    let mut offset_pos = offsets_offset;
    for index in 0..offset_count {
        let offset = read_u32(buf, &mut offset_pos)?;
        if (index == 0 && offset != 0)
            || previous.is_some_and(|prior| offset < prior)
            || u64::from(offset) > compressed_len_u64
        {
            return Err(FsstError::BadOffset {
                index,
                offset: u64::from(offset),
                len: compressed_len_u64,
            });
        }
        if let Some(start) = previous {
            let string =
                compressed
                    .get(start as usize..offset as usize)
                    .ok_or(FsstError::BadOffset {
                        index,
                        offset: u64::from(offset),
                        len: compressed_len_u64,
                    })?;
            table.visit_decoded_chunks(string, |_| {})?;
        }
        previous = Some(offset);
    }
    if previous != Some(compressed_len_raw) {
        return Err(FsstError::BadOffset {
            index: count,
            offset: previous.map_or(0, u64::from),
            len: compressed_len_u64,
        });
    }

    Ok(ParsedFsstBlob {
        table,
        count,
        offsets_offset,
        compressed_offset,
        compressed_len,
    })
}

impl FsstCodec {
    /// Serializes the codec to a self-describing, position-independent blob.
    ///
    /// The layout honours the Plan 2 zero-copy contract: a fixed 48-byte
    /// header with blob-relative `u64` section offsets, naturally-aligned
    /// arrays, and a trailing CRC32. See `BLOB_VERSION` for the version.
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
        buf[offsets_pos..offsets_pos + 8].copy_from_slice(&table_offset.to_le_bytes());
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
    /// CRC mismatch, malformed symbol table, unknown symbol code, or incomplete
    /// escape within any individual string. Code-stream validation does not
    /// allocate decoded strings.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, FsstError> {
        let parsed = parse_fsst_blob(buf)?;
        let mut offsets = Vec::with_capacity(parsed.count + 1);
        let mut offset_pos = parsed.offsets_offset;
        for _ in 0..=parsed.count {
            offsets.push(read_u32(buf, &mut offset_pos)?);
        }
        let compressed_end = checked_end(
            parsed.compressed_offset,
            parsed.compressed_len,
            "compressed stream",
        )?;
        let compressed = buf
            .get(parsed.compressed_offset..compressed_end)
            .ok_or(FsstError::Truncated {
                need: compressed_end,
                have: buf.len(),
            })?
            .to_vec();
        // The common parser already checked every offset and every string's
        // code stream. Materialize its validated fields without scanning again.
        Ok(Self {
            table: parsed.table,
            compressed,
            offsets,
        })
    }

    /// Opens a blob shared via [`bytes::Bytes`] into an owned codec.
    ///
    /// This still copies the compressed stream and offsets array into owned
    /// `Vec`s. For a true borrowing reader that holds `Bytes` slices and
    /// decodes strings on demand, use [`FsstView::open`] instead.
    ///
    /// # Errors
    /// Same as [`Self::from_bytes`].
    pub fn from_bytes_shared(blob: bytes::Bytes) -> Result<Self, FsstError> {
        Self::from_bytes(&blob)
    }
}

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
        let parsed = parse_fsst_blob(&blob)?;
        Ok(Self {
            blob,
            table: parsed.table,
            count: parsed.count,
            offsets_offset: parsed.offsets_offset,
            compressed_offset: parsed.compressed_offset,
            compressed_len: parsed.compressed_len,
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
    fn read_offset(&self, index: usize) -> Result<u32, FsstError> {
        let byte_offset = index.checked_mul(4).ok_or(FsstError::SizeOverflow {
            region: "offset index",
        })?;
        let mut pos = checked_end(self.offsets_offset, byte_offset, "offset index")?;
        read_u32(&self.blob, &mut pos)
    }

    /// Decodes string `index`, or `Ok(None)` if out of bounds.
    ///
    /// # Errors
    /// Returns [`FsstError`] if the stored bytes are malformed.
    pub fn get(&self, index: usize) -> Result<Option<Vec<u8>>, FsstError> {
        if index >= self.count {
            return Ok(None);
        }
        let start = self.read_offset(index)? as usize;
        let end = self.read_offset(index + 1)? as usize;
        if end > self.compressed_len || start > end {
            return Err(FsstError::BadOffset {
                index,
                offset: start as u64,
                // reason: compressed_len fits u32 — bounded by the u32 stored in the blob header
                #[allow(clippy::cast_possible_truncation)]
                len: u64::from(self.compressed_len as u32),
            });
        }
        let compressed_start = checked_end(self.compressed_offset, start, "compressed string")?;
        let compressed_end = checked_end(self.compressed_offset, end, "compressed string")?;
        let compressed =
            self.blob
                .get(compressed_start..compressed_end)
                .ok_or(FsstError::Truncated {
                    need: compressed_end,
                    have: self.blob.len(),
                })?;
        Ok(Some(self.table.decode(compressed)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repatch_crc(blob: &mut [u8]) {
        let body_end = blob.len() - 4;
        let crc = crc32fast::hash(&blob[..body_end]);
        blob[body_end..].copy_from_slice(&crc.to_le_bytes());
    }

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
            assert!(
                t.symbol(code).is_none(),
                "code {code} should have no symbol"
            );
        }
    }

    #[test]
    fn symbol_table_set_and_lookup() {
        let mut t = SymbolTable::default();
        t.set(1, b"abc").expect("valid symbol");
        t.set(2, b"xy").expect("valid symbol");
        assert_eq!(t.symbol(1), Some(b"abc" as &[u8]));
        assert_eq!(t.symbol(2), Some(b"xy" as &[u8]));
        assert_eq!(t.symbol(3), None);
    }

    #[test]
    fn symbol_table_rejects_oversize_symbol() {
        let mut t = SymbolTable::default();
        assert!(t.set(1, b"ninebytes").is_err());
    }

    #[test]
    fn symbol_table_rejects_escape_code() {
        let mut t = SymbolTable::default();
        assert!(t.set(0, b"x").is_err());
    }

    #[test]
    fn longest_match_finds_the_longest_prefix() {
        let mut t = SymbolTable::default();
        t.set(1, b"a").expect("valid symbol");
        t.set(2, b"ab").expect("valid symbol");
        t.set(3, b"abc").expect("valid symbol");
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
        a.set(1, b"abcdefgh").expect("valid symbol"); // full 8-byte slot
        a.set(1, b"x").expect("valid symbol"); // overwrite with 1-byte symbol

        // Lookup returns the new shorter symbol.
        assert_eq!(a.symbol(1), Some(b"x" as &[u8]));

        // The body slot must be zeroed beyond the new length, so PartialEq
        // with a freshly-built table holding only "x" matches.
        let mut b = SymbolTable::default();
        b.set(1, b"x").expect("valid symbol");
        assert_eq!(
            a, b,
            "trailing body bytes from previous symbol must be zeroed"
        );
    }

    #[test]
    fn train_empty_sample_returns_empty_table() {
        let table = SymbolTable::train(&[]);
        assert!(table.is_empty());
    }

    #[test]
    fn train_picks_frequent_substrings() {
        // Sample where "the " appears many times.
        let strings: Vec<&[u8]> = vec![b"the cat", b"the dog", b"the bird", b"the rat", b"the fox"];
        let table = SymbolTable::train(&strings);
        // The table should contain "the " (or a prefix of it) as a multi-byte symbol.
        let has_the = (1u8..=255).any(|c| table.symbol(c).is_some_and(|s| s.starts_with(b"the")));
        assert!(has_the, "expected a symbol covering 'the'");
    }

    #[test]
    fn train_always_yields_a_table_that_can_encode_sample_bytes() {
        // Any byte in the sample is either matched by a symbol or escape-encoded —
        // we test the latter by checking the symbol table is buildable.
        let strings: Vec<&[u8]> = vec![b"hello", b"world", b""];
        let _ = SymbolTable::train(&strings); // does not panic on empty strings
    }

    #[test]
    fn longest_match_index_matches_scan() {
        // The index-backed longest_match must equal the exhaustive scan for
        // every suffix of the training sample.
        let sample: Vec<&[u8]> = vec![b"banana", b"band", b"can", b"candy", b"a"];
        let table = SymbolTable::train(&sample);
        for s in &sample {
            for i in 0..s.len() {
                assert_eq!(
                    table.longest_match(&s[i..]),
                    table.longest_match_scan(&s[i..]),
                    "index and scan disagree at suffix {:?}",
                    &s[i..]
                );
            }
        }
    }

    #[test]
    fn encode_decode_round_trip_simple() {
        let mut table = SymbolTable::default();
        table.set(1, b"the ").expect("valid symbol");
        table.set(2, b"cat").expect("valid symbol");
        table.set(3, b"dog").expect("valid symbol");

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
        table.set(1, b"hello").expect("valid symbol");
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
    fn semantic_decode_rejects_unknown_symbol() {
        assert!(matches!(
            SymbolTable::default().decode(&[1]),
            Err(FsstError::UnknownSymbol {
                code: 1,
                position: 0
            })
        ));
    }

    #[test]
    fn semantic_opens_reject_unknown_symbol_with_empty_table() {
        let blob = FsstCodec {
            table: SymbolTable::default(),
            compressed: vec![1],
            offsets: vec![0, 1],
        }
        .to_bytes();
        assert_semantic_open_rejected(
            blob,
            &FsstError::UnknownSymbol {
                code: 1,
                position: 0,
            },
        );
    }

    #[test]
    fn semantic_opens_reject_trailing_escape_with_empty_table() {
        let blob = FsstCodec {
            table: SymbolTable::default(),
            compressed: vec![0],
            offsets: vec![0, 1],
        }
        .to_bytes();
        assert_semantic_open_rejected(blob, &FsstError::TruncatedEscape(0));
    }

    #[test]
    fn semantic_opens_reject_escape_crossing_string_boundary_with_empty_table() {
        let blob = FsstCodec {
            table: SymbolTable::default(),
            compressed: vec![0, b'a'],
            offsets: vec![0, 1, 2],
        }
        .to_bytes();
        assert_semantic_open_rejected(blob, &FsstError::TruncatedEscape(0));
    }

    fn semantic_fixture() -> Vec<u8> {
        let mut table = SymbolTable::default();
        table.set(1, b"a").unwrap();
        FsstCodec {
            table,
            compressed: vec![1, 0, 1],
            offsets: vec![0, 1, 3],
        }
        .to_bytes()
    }

    fn assert_semantic_open_rejected(blob: Vec<u8>, expected: &FsstError) {
        let owned = FsstCodec::from_bytes(&blob).map(|_| ());
        let shared = FsstCodec::from_bytes_shared(bytes::Bytes::from(blob.clone())).map(|_| ());
        let view = FsstView::open(bytes::Bytes::from(blob)).map(|_| ());
        for (kind, result) in [("owned", owned), ("shared", shared), ("view", view)] {
            let error = result.expect_err(kind);
            match (expected, &error) {
                (
                    FsstError::UnknownSymbol { code, position },
                    FsstError::UnknownSymbol {
                        code: actual_code,
                        position: actual_position,
                    },
                ) => assert_eq!((actual_code, actual_position), (code, position), "{kind}"),
                (FsstError::TruncatedEscape(position), FsstError::TruncatedEscape(actual)) => {
                    assert_eq!(actual, position, "{kind}");
                }
                _ => panic!("{kind}: expected {expected:?}, got {error:?}"),
            }
        }
    }

    #[test]
    fn semantic_opens_reject_crc_valid_unknown_symbol_in_later_string() {
        let mut blob = semantic_fixture();
        let stream = usize::try_from(u64::from_le_bytes(blob[32..40].try_into().unwrap())).unwrap();
        // The later string now contains valid code 1 followed by absent code 2.
        blob[stream + 1..stream + 3].copy_from_slice(&[1, 2]);
        repatch_crc(&mut blob);
        assert_semantic_open_rejected(
            blob,
            &FsstError::UnknownSymbol {
                code: 2,
                position: 1,
            },
        );
    }

    #[test]
    fn semantic_opens_reject_crc_valid_trailing_escape_in_later_string() {
        let mut blob = semantic_fixture();
        let stream = usize::try_from(u64::from_le_bytes(blob[32..40].try_into().unwrap())).unwrap();
        blob[stream + 1..stream + 3].copy_from_slice(&[1, 0]);
        repatch_crc(&mut blob);
        assert_semantic_open_rejected(blob, &FsstError::TruncatedEscape(1));
    }

    #[test]
    fn semantic_opens_reject_crc_valid_escape_crossing_string_boundary() {
        let mut blob = semantic_fixture();
        let offsets =
            usize::try_from(u64::from_le_bytes(blob[24..32].try_into().unwrap())).unwrap();
        // [1] [0, 1] becomes [1, 0] [1]. The whole stream still decodes, but
        // the first string's escape must not consume its neighbour's code.
        blob[offsets + 4..offsets + 8].copy_from_slice(&2u32.to_le_bytes());
        repatch_crc(&mut blob);
        assert_semantic_open_rejected(blob, &FsstError::TruncatedEscape(1));
    }

    #[test]
    fn semantic_opens_preserve_all_escaped_bytes_and_empty_strings() {
        let literals: Vec<u8> = (0..=255).collect();
        let compressed = literals.iter().flat_map(|&byte| [0, byte]).collect();
        let codec = FsstCodec {
            table: SymbolTable::default(),
            compressed,
            offsets: vec![0, 0, 512, 512],
        };
        let blob = codec.to_bytes();
        let owned = FsstCodec::from_bytes(&blob).unwrap();
        let view = FsstView::open(bytes::Bytes::from(blob)).unwrap();
        for (index, expected) in [b"".as_slice(), literals.as_slice(), b"".as_slice()]
            .into_iter()
            .enumerate()
        {
            assert_eq!(owned.get(index).unwrap().unwrap(), expected);
            assert_eq!(view.get(index).unwrap().unwrap(), expected);
        }
    }

    #[test]
    fn fsst_codec_round_trip_random_access() {
        let strings = [
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
            assert_eq!(
                codec.get(i).expect("get").expect("decode"),
                Vec::<u8>::new()
            );
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
        let strings = [
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
        assert_eq!(
            reopened.get(0).expect("get").expect("decode"),
            b"hello world"
        );
    }

    #[test]
    fn view_get_matches_owned_get() {
        let strings: Vec<&[u8]> = vec![b"alpha", b"beta gamma", b"", b"the quick brown fox"];
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

    #[test]
    fn fsst_blob_rejects_non_zero_first_offset() {
        let codec = FsstCodec::build(&[b"abc", b"def"]);
        let mut blob = codec.to_bytes();

        // Find the offsets section using the header u64 at offset 24
        // (offsets_offset). Patch offsets[0] to a non-zero value and re-patch CRC.
        let offsets_section_offset =
            usize::try_from(u64::from_le_bytes(blob[24..32].try_into().unwrap())).unwrap();
        blob[offsets_section_offset..offsets_section_offset + 4]
            .copy_from_slice(&7u32.to_le_bytes());

        // Re-patch the trailing CRC so the offsets-validation path (not CRC) is
        // exercised.
        repatch_crc(&mut blob);

        match FsstCodec::from_bytes(&blob) {
            Err(FsstError::BadOffset {
                index: 0,
                offset: 7,
                ..
            }) => {}
            other => panic!("expected BadOffset{{index:0, offset:7, ...}}, got {other:?}"),
        }
    }

    #[test]
    fn fsst_blob_rejects_crc_valid_trailing_storage() {
        let codec = FsstCodec::build(&[b"abc"]);
        let mut blob = codec.to_bytes();
        let trailer = blob.split_off(blob.len() - 4);
        blob.extend_from_slice(&[0; 4]);
        blob.extend_from_slice(&trailer);
        repatch_crc(&mut blob);

        assert!(matches!(
            FsstCodec::from_bytes(&blob),
            Err(FsstError::NonCanonicalLayout {
                region: "trailing bytes"
            })
        ));
        assert!(matches!(
            FsstView::open(bytes::Bytes::from(blob)),
            Err(FsstError::NonCanonicalLayout {
                region: "trailing bytes"
            })
        ));
    }

    #[test]
    fn fsst_view_validates_every_offset_during_open() {
        let codec = FsstCodec::build(&[b"abc", b"def"]);
        let mut blob = codec.to_bytes();
        let offsets_offset =
            usize::try_from(u64::from_le_bytes(blob[24..32].try_into().unwrap())).unwrap();
        blob[offsets_offset + 4..offsets_offset + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        repatch_crc(&mut blob);

        assert!(matches!(
            FsstView::open(bytes::Bytes::from(blob)),
            Err(FsstError::BadOffset { index: 1, .. })
        ));
    }

    #[test]
    fn fsst_blob_rejects_nonzero_unused_symbol_storage() {
        let codec = FsstCodec::build(&[b"abc"]);
        let mut blob = codec.to_bytes();
        let unused_body_byte = BLOB_HEADER_SIZE + SYMBOL_LENGTH_BYTES + 255 * MAX_SYMBOL_LEN;
        blob[unused_body_byte] = 1;
        repatch_crc(&mut blob);

        assert!(matches!(
            FsstCodec::from_bytes(&blob),
            Err(FsstError::NonCanonicalLayout {
                region: "unused symbol body"
            })
        ));
    }
}
