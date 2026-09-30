//! Packed succinct permutation for the canonical Ring on-disk format.
//!
//! [`SuccinctPermutation`] stores both the forward and inverse mappings on
//! the heap (O(2n) space). The wire format stores only the forward mapping
//! as a packed `u32` LE array.
//! Deserialization rebuilds the inverse in a single linear pass, halving
//! the persisted mapping footprint.
//!
//! ## Layout
//!
//! ```text
//! Header (16 bytes):
//!     0..4    magic "PERM"
//!     4       version u8 = 1
//!     5..8    reserved (3 bytes, zero)
//!     8..16   n u64 LE
//!
//! forward region: n * 4 bytes (u32 LE)
//! ```

use bytes::Bytes;

use crate::index::ring::SuccinctPermutation;

const MAGIC: &[u8; 4] = b"PERM";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 16;

/// Errors returned when parsing a packed permutation from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedPermutationError {
    /// Buffer is too short to contain even the fixed-size header.
    TruncatedHeader,
    /// First 4 bytes don't match "PERM".
    BadMagic,
    /// Version byte not recognized.
    UnsupportedVersion(u8),
    /// Reserved header bytes must be zero in the canonical encoding.
    NonZeroReserved,
    /// Forward array is shorter than `n` declares.
    TruncatedForward {
        /// Bytes the forward array should contain.
        expected: usize,
        /// Bytes available in the input.
        actual: usize,
    },
    /// `n` field overflows the platform-native usize.
    SizeOverflow,
    /// The input contains bytes beyond its declared forward array.
    TrailingBytes {
        /// Exact canonical size implied by the header.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
    /// An in-memory permutation omitted an in-range forward target.
    MissingTarget(usize),
    /// A `forward\[i\]` entry references an index >= n (not a valid
    /// permutation).
    InvalidPermutation {
        /// Position that contained the bad value.
        index: usize,
        /// The bad value.
        value: u32,
    },
    /// A target index appears twice in the forward mapping (not a
    /// bijection).
    DuplicateTarget {
        /// Position whose target collided with an earlier one.
        index: usize,
        /// The duplicated target value.
        value: u32,
    },
}

impl std::fmt::Display for PackedPermutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "packed permutation header truncated"),
            Self::BadMagic => write!(f, "packed permutation bad magic (expected 'PERM')"),
            Self::UnsupportedVersion(v) => {
                write!(f, "packed permutation unsupported version {v}")
            }
            Self::NonZeroReserved => {
                write!(f, "packed permutation reserved bytes must be zero")
            }
            Self::TruncatedForward { expected, actual } => write!(
                f,
                "packed permutation forward truncated: expected {expected} bytes, got {actual}"
            ),
            Self::SizeOverflow => write!(f, "packed permutation size field overflows usize"),
            Self::TrailingBytes { expected, actual } => write!(
                f,
                "packed permutation has trailing bytes: expected {expected}, got {actual}"
            ),
            Self::MissingTarget(index) => {
                write!(f, "permutation is missing in-range target at index {index}")
            }
            Self::InvalidPermutation { index, value } => write!(
                f,
                "packed permutation forward[{index}] = {value} is out of range"
            ),
            Self::DuplicateTarget { index, value } => write!(
                f,
                "packed permutation forward[{index}] = {value} duplicates an earlier entry"
            ),
        }
    }
}

impl std::error::Error for PackedPermutationError {}

/// Serializes a [`SuccinctPermutation`] to the canonical packed format.
///
/// # Errors
///
/// Returns an error if the permutation cannot be represented by the wire
/// grammar or its in-memory forward map is incomplete.
pub fn serialize_permutation(
    perm: &SuccinctPermutation,
) -> Result<Vec<u8>, PackedPermutationError> {
    let n = perm.len();
    let n_u64 = u64::try_from(n).map_err(|_| PackedPermutationError::SizeOverflow)?;
    let n_u32 = u32::try_from(n).map_err(|_| PackedPermutationError::SizeOverflow)?;
    let total = n
        .checked_mul(4)
        .and_then(|value| HEADER_SIZE.checked_add(value))
        .ok_or(PackedPermutationError::SizeOverflow)?;
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(MAGIC); // 0..4
    buf.push(VERSION); // 4
    buf.extend_from_slice(&[0u8; 3]); // 5..8 reserved
    buf.extend_from_slice(&n_u64.to_le_bytes()); // 8..16
    let mut seen = vec![false; n];
    for i in 0..n {
        let target = perm
            .apply(i)
            .ok_or(PackedPermutationError::MissingTarget(i))?;
        let target = u32::try_from(target).map_err(|_| PackedPermutationError::SizeOverflow)?;
        if target >= n_u32 {
            return Err(PackedPermutationError::InvalidPermutation {
                index: i,
                value: target,
            });
        }
        if seen[target as usize] {
            return Err(PackedPermutationError::DuplicateTarget {
                index: i,
                value: target,
            });
        }
        seen[target as usize] = true;
        buf.extend_from_slice(&target.to_le_bytes());
    }
    Ok(buf)
}

/// Parses a [`SuccinctPermutation`] from the canonical packed format. Rebuilds
/// the inverse mapping in a single linear pass.
///
/// # Errors
///
/// Returns a [`PackedPermutationError`] on truncation, magic/version
/// mismatch, out-of-range entries, or duplicate targets (the input is
/// not a valid permutation).
///
pub fn deserialize_permutation(data: Bytes) -> Result<SuccinctPermutation, PackedPermutationError> {
    if data.len() < HEADER_SIZE {
        return Err(PackedPermutationError::TruncatedHeader);
    }
    if &data[0..4] != MAGIC {
        return Err(PackedPermutationError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(PackedPermutationError::UnsupportedVersion(version));
    }
    if data[5..8] != [0; 3] {
        return Err(PackedPermutationError::NonZeroReserved);
    }
    let n_raw = read_u64(&data, 8).ok_or(PackedPermutationError::TruncatedHeader)?;
    let n = usize::try_from(n_raw).map_err(|_| PackedPermutationError::SizeOverflow)?;

    let forward_bytes = n
        .checked_mul(4)
        .ok_or(PackedPermutationError::SizeOverflow)?;
    let total = HEADER_SIZE
        .checked_add(forward_bytes)
        .ok_or(PackedPermutationError::SizeOverflow)?;
    if total > data.len() {
        return Err(PackedPermutationError::TruncatedForward {
            expected: forward_bytes,
            actual: data.len() - HEADER_SIZE,
        });
    }
    if total != data.len() {
        return Err(PackedPermutationError::TrailingBytes {
            expected: total,
            actual: data.len(),
        });
    }

    // Validate + collect forward array as usize for SuccinctPermutation::new.
    let n_u32 = u32::try_from(n).map_err(|_| PackedPermutationError::SizeOverflow)?;
    let mut forward: Vec<usize> = Vec::with_capacity(n);
    let mut seen: Vec<bool> = vec![false; n];
    for i in 0..n {
        let off = i
            .checked_mul(4)
            .and_then(|value| HEADER_SIZE.checked_add(value))
            .ok_or(PackedPermutationError::SizeOverflow)?;
        let value = read_u32(&data, off).ok_or(PackedPermutationError::TruncatedForward {
            expected: forward_bytes,
            actual: data.len() - HEADER_SIZE,
        })?;
        if value >= n_u32 {
            return Err(PackedPermutationError::InvalidPermutation { index: i, value });
        }
        if seen[value as usize] {
            return Err(PackedPermutationError::DuplicateTarget { index: i, value });
        }
        seen[value as usize] = true;
        forward.push(value as usize);
    }

    Ok(SuccinctPermutation::new(&forward))
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

    fn build_perm(forward: &[usize]) -> SuccinctPermutation {
        SuccinctPermutation::new(forward)
    }

    fn encode(perm: &SuccinctPermutation) -> Vec<u8> {
        serialize_permutation(perm).expect("serialize permutation")
    }

    #[test]
    fn alix_packed_perm_roundtrip_small() {
        let forward = vec![3usize, 0, 4, 1, 2];
        let perm = build_perm(&forward);
        let bytes = encode(&perm);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("deserialize");
        assert_eq!(restored.len(), perm.len());
        for i in 0..perm.len() {
            assert_eq!(restored.apply(i), perm.apply(i));
            assert_eq!(restored.apply_inverse(i), perm.apply_inverse(i));
        }
    }

    #[test]
    fn gus_packed_perm_roundtrip_identity() {
        let forward: Vec<usize> = (0..256).collect();
        let perm = build_perm(&forward);
        let bytes = encode(&perm);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("deserialize");
        for i in 0..256 {
            assert_eq!(restored.apply(i), Some(i));
        }
    }

    #[test]
    fn vincent_packed_perm_roundtrip_reverse() {
        let forward: Vec<usize> = (0..128).rev().collect();
        let perm = build_perm(&forward);
        let bytes = encode(&perm);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("deserialize");
        for i in 0..128 {
            assert_eq!(restored.apply(i), Some(127 - i));
            // Inverse of reverse is also reverse.
            assert_eq!(restored.apply_inverse(i), Some(127 - i));
        }
    }

    #[test]
    fn jules_packed_perm_empty() {
        let perm = build_perm(&[]);
        let bytes = encode(&perm);
        assert_eq!(bytes.len(), HEADER_SIZE);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("empty");
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn mia_packed_perm_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; HEADER_SIZE]);
        assert_eq!(
            deserialize_permutation(bad).unwrap_err(),
            PackedPermutationError::BadMagic
        );
    }

    #[test]
    fn shosanna_packed_perm_truncated_header_rejected() {
        let short = Bytes::from(vec![b'P', b'E', b'R', b'M']);
        assert_eq!(
            deserialize_permutation(short).unwrap_err(),
            PackedPermutationError::TruncatedHeader
        );
    }

    #[test]
    fn beatrix_packed_perm_unsupported_version_rejected() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[..4].copy_from_slice(MAGIC);
        buf[4] = 99;
        assert_eq!(
            deserialize_permutation(Bytes::from(buf)).unwrap_err(),
            PackedPermutationError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn hans_packed_perm_invalid_target_rejected() {
        // n=2 but forward[0] = 5 (out of range)
        let mut buf = Vec::with_capacity(HEADER_SIZE + 8);
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes()); // bad
        buf.extend_from_slice(&0u32.to_le_bytes());
        let result = deserialize_permutation(Bytes::from(buf));
        assert!(matches!(
            result.unwrap_err(),
            PackedPermutationError::InvalidPermutation { index: 0, value: 5 }
        ));
    }

    #[test]
    fn django_packed_perm_duplicate_target_rejected() {
        // n=3, forward = [0, 0, 2] — 0 appears twice.
        let mut buf = Vec::with_capacity(HEADER_SIZE + 12);
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // dup
        buf.extend_from_slice(&2u32.to_le_bytes());
        let result = deserialize_permutation(Bytes::from(buf));
        assert!(matches!(
            result.unwrap_err(),
            PackedPermutationError::DuplicateTarget { index: 1, value: 0 }
        ));
    }

    #[test]
    fn tarantino_packed_perm_has_fixed_width_size() {
        let forward: Vec<usize> = (0..512).map(|i| (i * 17) % 512).collect();
        let perm = build_perm(&forward);
        let bytes = encode(&perm);
        assert_eq!(bytes.len(), HEADER_SIZE + forward.len() * 4);
    }

    #[test]
    fn non_zero_reserved_bytes_are_rejected() {
        let mut bytes = encode(&build_perm(&[0]));
        bytes[5] = 1;
        assert_eq!(
            deserialize_permutation(Bytes::from(bytes)).unwrap_err(),
            PackedPermutationError::NonZeroReserved
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = encode(&build_perm(&[0]));
        let expected = bytes.len();
        bytes.push(0);
        assert_eq!(
            deserialize_permutation(Bytes::from(bytes)).unwrap_err(),
            PackedPermutationError::TrailingBytes {
                expected,
                actual: expected + 1,
            }
        );
    }
}
