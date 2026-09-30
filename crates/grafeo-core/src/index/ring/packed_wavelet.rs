//! Packed wavelet tree for the canonical Ring on-disk format.
//!
//! The in-memory [`WaveletTree`] stores `height` `SuccinctBitVector`
//! levels alongside rank/select sampling caches. Those caches are O(n)
//! rebuildable from the bits alone, so the wire image keeps only the bit
//! data as little-endian `u64` words and reconstructs the caches on load.
//!
//! ## Layout
//!
//! ```text
//! Header (40 bytes):
//!     0..4    magic "WTRE"
//!     4       version u8 = 1
//!     5..8    reserved (3 bytes, zero)
//!     8..12   height u32 LE
//!     12..16  padding (4 bytes, zero) — aligns u64 fields to 8-byte boundary
//!     16..24  sigma u64 LE                 // alphabet size
//!     24..32  len u64 LE                   // sequence length (== bits per level)
//!     32..40  symbol_count u64 LE          // length of the symbols region in elements
//!
//! symbols region: symbol_count * 8 bytes (u64 LE, sorted)
//! per-level region (height entries):
//!     bit_count: u64 LE
//!     word_count: u64 LE
//!     word_count * 8 bytes of LE u64 BitVector data
//! ```

use bytes::Bytes;

use crate::codec::BitVector;
use crate::codec::succinct::{SuccinctBitVector, WaveletTree};

const MAGIC: &[u8; 4] = b"WTRE";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 40;

/// Errors returned when parsing a packed wavelet tree from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedWaveletError {
    /// Buffer is too short to contain even the fixed-size header.
    TruncatedHeader,
    /// First 4 bytes don't match "WTRE".
    BadMagic,
    /// Version byte not recognized.
    UnsupportedVersion(u8),
    /// Reserved or padding header bytes are non-zero.
    NonZeroReserved,
    /// Recorded sizes overflow the input buffer.
    Truncated {
        /// Region we were trying to read.
        region: &'static str,
    },
    /// A field overflows the platform-native usize.
    SizeOverflow,
    /// Header fields cannot describe a canonical wavelet tree.
    InconsistentMetadata,
    /// The in-memory source violates its structural invariants.
    InvalidSource(String),
    /// Per-level bit count doesn't match the declared `len` field.
    BitCountMismatch {
        /// Level index where the mismatch was observed.
        level: usize,
        /// Bit count declared in the header.
        expected: u64,
        /// Bit count observed in the level.
        actual: u64,
    },
    /// Word count is not exactly `ceil(bit_count / 64)`.
    WordCountMismatch {
        /// Level index where the mismatch was observed.
        level: usize,
        /// Exact word count required by the bit count.
        expected: u64,
        /// Word count carried in the level header.
        actual: u64,
    },
    /// Padding bits above the declared bit count must be zero.
    NonZeroPaddingBits {
        /// Level index containing non-canonical padding bits.
        level: usize,
    },
    /// Level bits encode a code outside the declared alphabet.
    InvalidCode {
        /// Encoded code carrying at least one occurrence.
        code: usize,
    },
    /// Bytes remain after the declared levels.
    TrailingBytes {
        /// Exact canonical length implied by the metadata.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// Reconstructed parts violated a structural [`WaveletTree`]
    /// invariant — caught here rather than letting the tree return
    /// inconsistent answers from `access`/`rank`.
    InvariantViolation(crate::codec::succinct::WaveletInvariantError),
}

impl std::fmt::Display for PackedWaveletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "packed wavelet header truncated"),
            Self::BadMagic => write!(f, "packed wavelet bad magic (expected 'WTRE')"),
            Self::UnsupportedVersion(v) => write!(f, "packed wavelet unsupported version {v}"),
            Self::NonZeroReserved => write!(f, "packed wavelet reserved bytes must be zero"),
            Self::Truncated { region } => write!(f, "packed wavelet truncated in {region}"),
            Self::SizeOverflow => write!(f, "packed wavelet size field overflows usize"),
            Self::InconsistentMetadata => {
                write!(f, "packed wavelet metadata is not canonical")
            }
            Self::InvalidSource(error) => write!(f, "wavelet source is invalid: {error}"),
            Self::BitCountMismatch {
                level,
                expected,
                actual,
            } => write!(
                f,
                "packed wavelet bit count mismatch at level {level}: expected {expected}, got {actual}"
            ),
            Self::WordCountMismatch {
                level,
                expected,
                actual,
            } => write!(
                f,
                "packed wavelet word count mismatch at level {level}: expected {expected}, got {actual}"
            ),
            Self::NonZeroPaddingBits { level } => {
                write!(f, "packed wavelet level {level} has non-zero padding bits")
            }
            Self::InvalidCode { code } => write!(
                f,
                "packed wavelet code {code} lies outside the declared alphabet"
            ),
            Self::TrailingBytes { expected, actual } => write!(
                f,
                "packed wavelet has trailing bytes: expected {expected}, got {actual}"
            ),
            Self::InvariantViolation(e) => write!(f, "packed wavelet invariant violation: {e}"),
        }
    }
}

impl std::error::Error for PackedWaveletError {}

/// Serializes a [`WaveletTree`] to the canonical packed format.
///
/// # Errors
///
/// Returns an error if any size cannot be represented by the wire grammar.
pub fn serialize_wavelet_tree(tree: &WaveletTree) -> Result<Vec<u8>, PackedWaveletError> {
    validate_source(tree)?;
    let symbols = tree.symbols_slice();
    let height = tree.height();
    let levels = tree.levels_slice();
    let sigma = tree.sigma();
    let len = u64::try_from(tree.len()).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let height = u32::try_from(height).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let symbol_count =
        u64::try_from(symbols.len()).map_err(|_| PackedWaveletError::SizeOverflow)?;
    validate_padding_bits(levels, tree.len())?;

    // Estimate total size to pre-allocate.
    let symbols_bytes = symbols
        .len()
        .checked_mul(8)
        .ok_or(PackedWaveletError::SizeOverflow)?;
    let level_bytes = levels.iter().try_fold(0usize, |total, sbv| {
        total
            .checked_add(16)
            .and_then(|value| value.checked_add(sbv.inner().data_bytes().len()))
            .ok_or(PackedWaveletError::SizeOverflow)
    })?;
    let total = HEADER_SIZE
        .checked_add(symbols_bytes)
        .and_then(|value| value.checked_add(level_bytes))
        .ok_or(PackedWaveletError::SizeOverflow)?;

    let mut buf = Vec::with_capacity(total);
    // Header (40 bytes total — see module-top layout doc):
    buf.extend_from_slice(MAGIC); // 0..4
    buf.push(VERSION); // 4
    buf.extend_from_slice(&[0u8; 3]); // 5..8 reserved
    buf.extend_from_slice(&height.to_le_bytes()); // 8..12
    buf.extend_from_slice(&[0u8; 4]); // 12..16 padding to align sigma
    buf.extend_from_slice(&sigma.to_le_bytes()); // 16..24
    buf.extend_from_slice(&len.to_le_bytes()); // 24..32
    buf.extend_from_slice(&symbol_count.to_le_bytes()); // 32..40 symbol_count

    // Symbols.
    for &sym in symbols {
        buf.extend_from_slice(&sym.to_le_bytes());
    }

    // Levels.
    for sbv in levels {
        let bv = sbv.inner();
        let bit_count = u64::try_from(bv.len()).map_err(|_| PackedWaveletError::SizeOverflow)?;
        let word_data = bv.data_bytes();
        let word_count =
            u64::try_from(word_data.len() / 8).map_err(|_| PackedWaveletError::SizeOverflow)?;
        buf.extend_from_slice(&bit_count.to_le_bytes());
        buf.extend_from_slice(&word_count.to_le_bytes());
        buf.extend_from_slice(word_data);
    }

    Ok(buf)
}

/// Parses a [`WaveletTree`] from the canonical packed format. Rebuilds rank/select
/// caches per level via [`SuccinctBitVector::from_bitvec`].
///
/// `data` is consumed via `Bytes::slice` so the underlying allocation is
/// shared with the caller. Per-level `BitVector`s adopt their slices via
/// [`BitVector::from_bytes_storage`], so a mmap-backed buffer never copies.
///
/// # Errors
///
/// Returns a [`PackedWaveletError`] on truncation, magic/version
/// mismatch, or per-level bit-count inconsistency.
///
pub fn deserialize_wavelet_tree(data: Bytes) -> Result<WaveletTree, PackedWaveletError> {
    if data.len() < HEADER_SIZE {
        return Err(PackedWaveletError::TruncatedHeader);
    }
    if &data[0..4] != MAGIC {
        return Err(PackedWaveletError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(PackedWaveletError::UnsupportedVersion(version));
    }
    if data[5..8] != [0; 3] || data[12..16] != [0; 4] {
        return Err(PackedWaveletError::NonZeroReserved);
    }
    // Header offsets (per module-top layout doc):
    let height_raw = read_u32(&data, 8).ok_or(PackedWaveletError::TruncatedHeader)?;
    let sigma = read_u64(&data, 16).ok_or(PackedWaveletError::TruncatedHeader)?;
    let len_raw = read_u64(&data, 24).ok_or(PackedWaveletError::TruncatedHeader)?;
    let symbol_count_raw = read_u64(&data, 32).ok_or(PackedWaveletError::TruncatedHeader)?;

    let height = usize::try_from(height_raw).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let len_usize = usize::try_from(len_raw).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let symbol_count =
        usize::try_from(symbol_count_raw).map_err(|_| PackedWaveletError::SizeOverflow)?;

    let expected_height = if sigma == 0 {
        0
    } else if sigma == 1 {
        1
    } else {
        usize::try_from(64 - (sigma - 1).leading_zeros())
            .map_err(|_| PackedWaveletError::SizeOverflow)?
    };
    if (len_usize == 0 && (height != 0 || sigma != 0 || symbol_count != 0))
        || (len_usize > 0
            && (symbol_count == 0 || symbol_count_raw != sigma || height != expected_height))
    {
        return Err(PackedWaveletError::InconsistentMetadata);
    }

    let mut cursor = HEADER_SIZE;

    // Symbols region.
    let symbols_bytes = symbol_count
        .checked_mul(8)
        .ok_or(PackedWaveletError::SizeOverflow)?;
    let symbols_end = cursor
        .checked_add(symbols_bytes)
        .ok_or(PackedWaveletError::SizeOverflow)?;
    if symbols_end > data.len() {
        return Err(PackedWaveletError::Truncated { region: "symbols" });
    }
    let mut symbols: Vec<u64> = Vec::with_capacity(symbol_count);
    for i in 0..symbol_count {
        let off = i
            .checked_mul(8)
            .and_then(|value| cursor.checked_add(value))
            .ok_or(PackedWaveletError::SizeOverflow)?;
        symbols
            .push(read_u64(&data, off).ok_or(PackedWaveletError::Truncated { region: "symbols" })?);
    }
    cursor = symbols_end;

    // Levels region.
    let minimum_levels_end = height
        .checked_mul(16)
        .and_then(|value| cursor.checked_add(value))
        .ok_or(PackedWaveletError::SizeOverflow)?;
    if minimum_levels_end > data.len() {
        return Err(PackedWaveletError::Truncated {
            region: "level headers",
        });
    }
    let mut levels: Vec<SuccinctBitVector> = Vec::with_capacity(height);
    for level_idx in 0..height {
        let level_header_end = cursor
            .checked_add(16)
            .ok_or(PackedWaveletError::SizeOverflow)?;
        if level_header_end > data.len() {
            return Err(PackedWaveletError::Truncated {
                region: "level header",
            });
        }
        let bit_count = read_u64(&data, cursor).ok_or(PackedWaveletError::Truncated {
            region: "level header",
        })?;
        let word_count = read_u64(&data, cursor + 8).ok_or(PackedWaveletError::Truncated {
            region: "level header",
        })?;
        cursor = level_header_end;

        if bit_count != len_raw {
            return Err(PackedWaveletError::BitCountMismatch {
                level: level_idx,
                expected: len_raw,
                actual: bit_count,
            });
        }

        let expected_word_count = len_raw.div_ceil(64);
        if word_count != expected_word_count {
            return Err(PackedWaveletError::WordCountMismatch {
                level: level_idx,
                expected: expected_word_count,
                actual: word_count,
            });
        }

        let level_bytes = usize::try_from(
            word_count
                .checked_mul(8)
                .ok_or(PackedWaveletError::SizeOverflow)?,
        )
        .map_err(|_| PackedWaveletError::SizeOverflow)?;
        let level_data_end = cursor
            .checked_add(level_bytes)
            .ok_or(PackedWaveletError::SizeOverflow)?;
        if level_data_end > data.len() {
            return Err(PackedWaveletError::Truncated {
                region: "level data",
            });
        }
        let level_slice = data.slice(cursor..level_data_end);
        cursor = level_data_end;

        let remainder = len_usize % 64;
        if remainder != 0 {
            let last_word_offset =
                level_slice
                    .len()
                    .checked_sub(8)
                    .ok_or(PackedWaveletError::Truncated {
                        region: "level bits",
                    })?;
            let last_word =
                read_u64(&level_slice, last_word_offset).ok_or(PackedWaveletError::Truncated {
                    region: "level bits",
                })?;
            let padding_mask = !((1u64 << remainder) - 1);
            if last_word & padding_mask != 0 {
                return Err(PackedWaveletError::NonZeroPaddingBits { level: level_idx });
            }
        }

        let bv = BitVector::from_bytes_storage(level_slice, len_usize).map_err(|_| {
            PackedWaveletError::Truncated {
                region: "level bits",
            }
        })?;
        levels.push(SuccinctBitVector::from_bitvec(bv));
    }

    if cursor != data.len() {
        return Err(PackedWaveletError::TrailingBytes {
            expected: cursor,
            actual: data.len(),
        });
    }

    validate_codes(&levels, height, len_usize, symbols.len())?;
    WaveletTree::from_packed_parts(levels, height, sigma, len_usize, symbols)
        .map_err(PackedWaveletError::InvariantViolation)
}

fn validate_source(tree: &WaveletTree) -> Result<(), PackedWaveletError> {
    let height = tree.height();
    let sigma = tree.sigma();
    let len = tree.len();
    let levels = tree.levels_slice();
    let symbols = tree.symbols_slice();
    if len == 0 {
        if height != 0 || sigma != 0 || !levels.is_empty() || !symbols.is_empty() {
            return Err(PackedWaveletError::InvalidSource(
                "empty tree carries non-empty metadata".to_owned(),
            ));
        }
        return Ok(());
    }
    let symbol_count = u64::try_from(symbols.len())
        .map_err(|_| PackedWaveletError::InvalidSource("symbol count overflows u64".to_owned()))?;
    let expected_height = if sigma <= 1 {
        1
    } else {
        usize::try_from(64 - (sigma - 1).leading_zeros())
            .map_err(|_| PackedWaveletError::InvalidSource("height overflows usize".to_owned()))?
    };
    if levels.len() != height
        || symbol_count != sigma
        || height != expected_height
        || levels.iter().any(|level| level.len() != len)
        || symbols.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PackedWaveletError::InvalidSource(
            "tree metadata is inconsistent".to_owned(),
        ));
    }
    Ok(())
}

fn validate_codes(
    levels: &[SuccinctBitVector],
    height: usize,
    len: usize,
    symbol_count: usize,
) -> Result<(), PackedWaveletError> {
    let height = u32::try_from(height).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let code_space = 1usize
        .checked_shl(height)
        .ok_or(PackedWaveletError::SizeOverflow)?;
    for code in symbol_count..code_space {
        let mut lower = 0usize;
        let mut upper = len;
        for (level, bit_vector) in levels.iter().enumerate() {
            let level = u32::try_from(level).map_err(|_| PackedWaveletError::SizeOverflow)?;
            let bit_position = height
                .checked_sub(level + 1)
                .ok_or(PackedWaveletError::InconsistentMetadata)?;
            if (code >> bit_position) & 1 == 0 {
                lower = bit_vector.rank0(lower);
                upper = bit_vector.rank0(upper);
            } else {
                let zero_count = bit_vector.count_zeros();
                lower = zero_count
                    .checked_add(bit_vector.rank1(lower))
                    .ok_or(PackedWaveletError::SizeOverflow)?;
                upper = zero_count
                    .checked_add(bit_vector.rank1(upper))
                    .ok_or(PackedWaveletError::SizeOverflow)?;
            }
        }
        if upper > lower {
            return Err(PackedWaveletError::InvalidCode { code });
        }
    }
    Ok(())
}

fn validate_padding_bits(
    levels: &[SuccinctBitVector],
    len: usize,
) -> Result<(), PackedWaveletError> {
    let remainder = len % 64;
    if remainder == 0 {
        return Ok(());
    }
    for (level, bit_vector) in levels.iter().enumerate() {
        let bytes = bit_vector.inner().data_bytes();
        let last_word_offset = bytes
            .len()
            .checked_sub(8)
            .ok_or_else(|| PackedWaveletError::InvalidSource("missing final word".to_owned()))?;
        let last_word = read_u64(bytes, last_word_offset)
            .ok_or_else(|| PackedWaveletError::InvalidSource("truncated final word".to_owned()))?;
        let padding_mask = !((1u64 << remainder) - 1);
        if last_word & padding_mask != 0 {
            return Err(PackedWaveletError::NonZeroPaddingBits { level });
        }
    }
    Ok(())
}

fn read_u64(bytes: &[u8], start: usize) -> Option<u64> {
    let end = start.checked_add(8)?;
    let chunk: [u8; 8] = bytes.get(start..end)?.try_into().ok()?;
    Some(u64::from_le_bytes(chunk))
}

fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    let end = start.checked_add(4)?;
    let chunk: [u8; 4] = bytes.get(start..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(chunk))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tree(seq: &[u64]) -> WaveletTree {
        WaveletTree::new(seq)
    }

    fn encode(tree: &WaveletTree) -> Vec<u8> {
        serialize_wavelet_tree(tree).expect("serialize wavelet tree")
    }

    fn assert_trees_equal(orig: &WaveletTree, restored: &WaveletTree) {
        assert_eq!(orig.len(), restored.len());
        assert_eq!(orig.sigma(), restored.sigma());
        for i in 0..orig.len() {
            assert_eq!(
                orig.access(i),
                restored.access(i),
                "access mismatch at position {i}"
            );
        }
    }

    #[test]
    fn alix_packed_wavelet_roundtrip_small() {
        let seq = vec![1u64, 3, 2, 1, 2, 3, 1, 2];
        let tree = build_tree(&seq);
        let bytes = encode(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_trees_equal(&tree, &restored);
    }

    #[test]
    fn gus_packed_wavelet_roundtrip_large() {
        // 1024 symbols drawn from an alphabet of 16 — exercises multi-level
        // wavelet structure at non-trivial size.
        let seq: Vec<u64> = (0..1024u64).map(|i| (i * 7) % 16).collect();
        let tree = build_tree(&seq);
        let bytes = encode(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_trees_equal(&tree, &restored);
    }

    #[test]
    fn vincent_packed_wavelet_empty() {
        let tree = WaveletTree::new(&[]);
        let bytes = encode(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn jules_packed_wavelet_single_symbol() {
        // sigma = 1; height ends up = 1 per `WaveletTree::new`.
        let seq = vec![42u64; 16];
        let tree = build_tree(&seq);
        let bytes = encode(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_trees_equal(&tree, &restored);
    }

    #[test]
    fn mia_packed_wavelet_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; HEADER_SIZE]);
        assert_eq!(
            deserialize_wavelet_tree(bad).unwrap_err(),
            PackedWaveletError::BadMagic
        );
    }

    #[test]
    fn shosanna_packed_wavelet_truncated_header_rejected() {
        let short = Bytes::from(vec![b'W', b'T', b'R', b'E']);
        assert_eq!(
            deserialize_wavelet_tree(short).unwrap_err(),
            PackedWaveletError::TruncatedHeader
        );
    }

    #[test]
    fn beatrix_packed_wavelet_unsupported_version_rejected() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[..4].copy_from_slice(MAGIC);
        buf[4] = 99;
        assert_eq!(
            deserialize_wavelet_tree(Bytes::from(buf)).unwrap_err(),
            PackedWaveletError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn hans_packed_wavelet_size_matches_grammar() {
        let seq: Vec<u64> = (0..2048u64).map(|i| (i * 11) % 64).collect();
        let tree = build_tree(&seq);
        let bytes = encode(&tree);
        let expected = HEADER_SIZE
            + tree.symbols_slice().len() * 8
            + tree
                .levels_slice()
                .iter()
                .map(|level| 16 + level.inner().data_bytes().len())
                .sum::<usize>();
        assert_eq!(bytes.len(), expected);
    }

    #[test]
    fn django_packed_wavelet_zero_copy_per_level() {
        // Round-trip and confirm restored levels' inner BitVector data
        // shares the underlying source allocation (zero-copy mmap path).
        let seq = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let tree = build_tree(&seq);
        let bytes = encode(&tree);
        let source = Bytes::from(bytes);
        let source_ptr = source.as_ptr();
        let source_len = source.len();

        let restored = deserialize_wavelet_tree(source).expect("deserialize");
        for (idx, level) in restored.levels_slice().iter().enumerate() {
            let inner_ptr = level.inner().data_bytes().as_ptr();
            let offset = inner_ptr as usize - source_ptr as usize;
            assert!(
                offset < source_len,
                "level {idx}: inner BitVector should be inside source allocation; offset={offset}"
            );
        }
    }

    #[test]
    fn reserved_bytes_are_rejected() {
        let mut bytes = encode(&build_tree(&[1]));
        bytes[12] = 1;
        assert_eq!(
            deserialize_wavelet_tree(Bytes::from(bytes)).unwrap_err(),
            PackedWaveletError::NonZeroReserved
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = encode(&build_tree(&[1]));
        let expected = bytes.len();
        bytes.push(0);
        assert_eq!(
            deserialize_wavelet_tree(Bytes::from(bytes)).unwrap_err(),
            PackedWaveletError::TrailingBytes {
                expected,
                actual: expected + 1,
            }
        );
    }

    #[test]
    fn non_zero_padding_bits_are_rejected() {
        let mut bytes = encode(&build_tree(&[1]));
        let first_level_word = HEADER_SIZE + 8 + 16;
        bytes[first_level_word + 7] = 0x80;
        assert_eq!(
            deserialize_wavelet_tree(Bytes::from(bytes)).unwrap_err(),
            PackedWaveletError::NonZeroPaddingBits { level: 0 }
        );
    }

    #[test]
    fn bit_pattern_outside_alphabet_is_rejected() {
        let mut bytes = encode(&build_tree(&[10, 20, 30]));
        let symbols_end = HEADER_SIZE + 3 * 8;
        let first_word = symbols_end + 16;
        let second_word = first_word + 8 + 16;
        bytes[first_word] = 0b111;
        bytes[second_word] = 0b111;
        assert!(matches!(
            deserialize_wavelet_tree(Bytes::from(bytes)).unwrap_err(),
            PackedWaveletError::InvalidCode { .. }
        ));
    }
}
