//! Canonical packed on-disk format for the Ring index.
//!
//! Composes the packed sub-formats into a single mmap-friendly byte buffer:
//!
//! - [`PackedTermDictionary`]
//! - `PackedWaveletTree` (see [`super::packed_wavelet`]) for subjects,
//!   predicates and objects
//! - `PackedPermutation` (see [`super::packed_permutation`]) for spo→pos
//!   and spo→osp
//!
//! ## Layout
//!
//! ```text
//! Header (64 bytes):
//!     0..4    magic "GRFR"
//!     4       version u8 = 2
//!     5..8    reserved (3 bytes, zero)
//!     8..16   num_triples u64 LE
//!     16..24  dict_offset u64 LE
//!     24..32  subjects_offset u64 LE
//!     32..40  predicates_offset u64 LE
//!     40..48  objects_offset u64 LE
//!     48..56  spo_to_pos_offset u64 LE
//!     56..64  spo_to_osp_offset u64 LE
//!
//! sub-sections (laid out at their declared offsets, each carries its own
//! magic + header):
//!     PackedTermDictionary (PDCT)
//!     PackedWaveletTree subjects (WTRE)
//!     PackedWaveletTree predicates (WTRE)
//!     PackedWaveletTree objects (WTRE)
//!     PackedPermutation spo_to_pos (PERM)
//!     PackedPermutation spo_to_osp (PERM)
//!
//! trailer (4 bytes):
//!     CRC32 LE of bytes [0..end-4]
//! ```
//!
//! Explicit offsets (rather than sequential parsing) let an mmap reader
//! `Bytes::slice` directly to any sub-section without first walking the
//! preceding ones.

use bytes::Bytes;

use crate::index::ring::{
    PackedDictError, PackedPermutationError, PackedTermDictionary, PackedWaveletError,
    SuccinctPermutation, TripleRing, deserialize_permutation, deserialize_wavelet_tree,
    serialize_permutation, serialize_wavelet_tree,
};

const MAGIC: &[u8; 4] = b"GRFR";
const VERSION: u8 = 2;
const HEADER_SIZE: usize = 64;
const HEADER_SIZE_U64: u64 = 64;
const TRAILER_SIZE: usize = 4;

/// Errors returned when encoding or parsing a packed Ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedRingError {
    /// Buffer is too small to even read the header.
    TruncatedHeader,
    /// First 4 bytes don't match "GRFR".
    BadMagic,
    /// Version byte not recognized (only `2` is accepted).
    UnsupportedVersion(u8),
    /// Reserved header bytes must be zero in the canonical encoding.
    NonZeroReserved,
    /// A size cannot be represented safely by the wire grammar.
    SizeOverflow {
        /// Component whose size overflowed.
        section: &'static str,
    },
    /// The first sub-section must begin immediately after the header.
    NonCanonicalLayout,
    /// A declared sub-section offset points outside the buffer.
    OffsetOutOfBounds {
        /// Sub-section that had the bad offset.
        section: &'static str,
        /// The declared offset.
        offset: u64,
    },
    /// The file's CRC32 trailer doesn't match the computed CRC.
    ChecksumMismatch {
        /// CRC32 the trailer claims.
        expected: u32,
        /// CRC32 computed from the bytes.
        actual: u32,
    },
    /// Embedded `PackedTermDictionary` failed to parse.
    Dict(PackedDictError),
    /// Embedded `PackedWaveletTree` failed to parse.
    Wavelet(PackedWaveletError),
    /// Embedded `PackedPermutation` failed to parse.
    Permutation(PackedPermutationError),
    /// `num_triples` declared in the header doesn't match the rebuilt
    /// permutations / wavelets.
    NumTriplesMismatch {
        /// Value declared in the header.
        declared: usize,
        /// Value implied by the parsed sub-sections.
        observed: usize,
    },
    /// Reconstructed sub-components violated a [`TripleRing`]
    /// structural invariant — caught here rather than letting the ring
    /// panic on a later query.
    RingInvariantViolation(super::triple_ring::TripleRingInvariantError),
}

impl std::fmt::Display for PackedRingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "packed ring header truncated"),
            Self::BadMagic => write!(f, "packed ring bad magic (expected 'GRFR')"),
            Self::UnsupportedVersion(v) => write!(f, "packed ring unsupported version {v}"),
            Self::NonZeroReserved => write!(f, "packed ring reserved bytes must be zero"),
            Self::SizeOverflow { section } => {
                write!(f, "packed ring {section} size overflows the wire grammar")
            }
            Self::NonCanonicalLayout => {
                write!(f, "packed ring sub-sections are not canonically contiguous")
            }
            Self::OffsetOutOfBounds { section, offset } => write!(
                f,
                "packed ring offset out of bounds: section '{section}' at {offset}"
            ),
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "packed ring CRC mismatch: expected {expected:#010X}, got {actual:#010X}"
            ),
            Self::Dict(e) => write!(f, "packed ring dictionary error: {e}"),
            Self::Wavelet(e) => write!(f, "packed ring wavelet error: {e}"),
            Self::Permutation(e) => write!(f, "packed ring permutation error: {e}"),
            Self::NumTriplesMismatch { declared, observed } => write!(
                f,
                "packed ring num_triples mismatch: declared {declared}, observed {observed}"
            ),
            Self::RingInvariantViolation(e) => {
                write!(f, "packed ring invariant violation: {e}")
            }
        }
    }
}

impl std::error::Error for PackedRingError {}

impl From<PackedDictError> for PackedRingError {
    fn from(e: PackedDictError) -> Self {
        Self::Dict(e)
    }
}

impl From<PackedWaveletError> for PackedRingError {
    fn from(e: PackedWaveletError) -> Self {
        Self::Wavelet(e)
    }
}

impl From<PackedPermutationError> for PackedRingError {
    fn from(e: PackedPermutationError) -> Self {
        Self::Permutation(e)
    }
}

impl From<super::triple_ring::TripleRingInvariantError> for PackedRingError {
    fn from(e: super::triple_ring::TripleRingInvariantError) -> Self {
        Self::RingInvariantViolation(e)
    }
}

/// Serializes a [`TripleRing`] to the canonical packed format.
///
/// The output buffer is laid out per the module-top documentation: a
/// 64-byte header with explicit sub-section offsets, the six packed
/// sub-sections in order, then a 4-byte CRC32 trailer.
///
/// # Errors
///
/// Returns an error if a component or aggregate size cannot be represented,
/// or if an in-memory component is structurally incomplete.
pub fn serialize_triple_ring(ring: &TripleRing) -> Result<Vec<u8>, PackedRingError> {
    // Serialize each sub-section first so we know their sizes for the
    // offset table.
    let dict_bytes = PackedTermDictionary::from_term_dict(ring.dictionary())?.to_bytes()?;
    let subj_bytes = serialize_wavelet_tree(ring.subjects_wt())?;
    let pred_bytes = serialize_wavelet_tree(ring.predicates_wt())?;
    let obj_bytes = serialize_wavelet_tree(ring.objects_wt())?;
    let pos_bytes = serialize_permutation(ring.spo_to_pos_perm())?;
    let osp_bytes = serialize_permutation(ring.spo_to_osp_perm())?;

    let dict_offset = HEADER_SIZE;
    let subj_offset =
        dict_offset
            .checked_add(dict_bytes.len())
            .ok_or(PackedRingError::SizeOverflow {
                section: "dictionary",
            })?;
    let pred_offset =
        subj_offset
            .checked_add(subj_bytes.len())
            .ok_or(PackedRingError::SizeOverflow {
                section: "subjects",
            })?;
    let obj_offset =
        pred_offset
            .checked_add(pred_bytes.len())
            .ok_or(PackedRingError::SizeOverflow {
                section: "predicates",
            })?;
    let pos_offset = obj_offset
        .checked_add(obj_bytes.len())
        .ok_or(PackedRingError::SizeOverflow { section: "objects" })?;
    let osp_offset =
        pos_offset
            .checked_add(pos_bytes.len())
            .ok_or(PackedRingError::SizeOverflow {
                section: "spo_to_pos",
            })?;
    let body_end =
        osp_offset
            .checked_add(osp_bytes.len())
            .ok_or(PackedRingError::SizeOverflow {
                section: "spo_to_osp",
            })?;
    let total = body_end
        .checked_add(TRAILER_SIZE)
        .ok_or(PackedRingError::SizeOverflow { section: "image" })?;
    let to_wire_offset = |offset: usize| {
        u64::try_from(offset).map_err(|_| PackedRingError::SizeOverflow { section: "offset" })
    };
    let dict_offset = to_wire_offset(dict_offset)?;
    let subj_offset = to_wire_offset(subj_offset)?;
    let pred_offset = to_wire_offset(pred_offset)?;
    let obj_offset = to_wire_offset(obj_offset)?;
    let pos_offset = to_wire_offset(pos_offset)?;
    let osp_offset = to_wire_offset(osp_offset)?;
    let num_triples = u64::try_from(ring.len())
        .map_err(|_| PackedRingError::SizeOverflow { section: "triples" })?;
    let mut buf = Vec::with_capacity(total);

    // Header.
    buf.extend_from_slice(MAGIC); // 0..4
    buf.push(VERSION); // 4
    buf.extend_from_slice(&[0u8; 3]); // 5..8 reserved
    buf.extend_from_slice(&num_triples.to_le_bytes()); // 8..16 num_triples
    buf.extend_from_slice(&dict_offset.to_le_bytes()); // 16..24
    buf.extend_from_slice(&subj_offset.to_le_bytes()); // 24..32
    buf.extend_from_slice(&pred_offset.to_le_bytes()); // 32..40
    buf.extend_from_slice(&obj_offset.to_le_bytes()); // 40..48
    buf.extend_from_slice(&pos_offset.to_le_bytes()); // 48..56
    buf.extend_from_slice(&osp_offset.to_le_bytes()); // 56..64

    // Sub-sections at their declared offsets.
    buf.extend_from_slice(&dict_bytes);
    buf.extend_from_slice(&subj_bytes);
    buf.extend_from_slice(&pred_bytes);
    buf.extend_from_slice(&obj_bytes);
    buf.extend_from_slice(&pos_bytes);
    buf.extend_from_slice(&osp_bytes);

    // Trailer: CRC32 of everything above.
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    Ok(buf)
}

/// Parses a [`TripleRing`] from the canonical packed format.
///
/// `data` is consumed via [`Bytes::slice`] so each sub-section's
/// allocation is shared with the caller. Mmap-backed buffers stay
/// zero-copy through term-dict + per-level-bitvector reconstruction.
///
/// # Errors
///
/// Returns a [`PackedRingError`] on any of: truncation, magic/version
/// mismatch, out-of-bounds sub-section offsets, CRC trailer mismatch,
/// or any of the embedded sub-format errors propagating up.
///
pub fn deserialize_triple_ring(data: Bytes) -> Result<TripleRing, PackedRingError> {
    if data.len() < HEADER_SIZE + TRAILER_SIZE {
        return Err(PackedRingError::TruncatedHeader);
    }
    if &data[0..4] != MAGIC {
        return Err(PackedRingError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(PackedRingError::UnsupportedVersion(version));
    }
    if data[5..8] != [0; 3] {
        return Err(PackedRingError::NonZeroReserved);
    }

    // CRC trailer first — fail fast on corruption.
    let body_end = data.len() - TRAILER_SIZE;
    let expected_crc = read_u32(&data, body_end).ok_or(PackedRingError::TruncatedHeader)?;
    let actual_crc = crc32fast::hash(&data[..body_end]);
    if actual_crc != expected_crc {
        return Err(PackedRingError::ChecksumMismatch {
            expected: expected_crc,
            actual: actual_crc,
        });
    }

    let num_triples_raw = read_u64(&data, 8).ok_or(PackedRingError::TruncatedHeader)?;
    let num_triples =
        usize::try_from(num_triples_raw).map_err(|_| PackedRingError::OffsetOutOfBounds {
            section: "num_triples",
            offset: num_triples_raw,
        })?;
    let dict_offset = read_u64(&data, 16).ok_or(PackedRingError::TruncatedHeader)?;
    let subj_offset = read_u64(&data, 24).ok_or(PackedRingError::TruncatedHeader)?;
    let pred_offset = read_u64(&data, 32).ok_or(PackedRingError::TruncatedHeader)?;
    let obj_offset = read_u64(&data, 40).ok_or(PackedRingError::TruncatedHeader)?;
    let pos_offset = read_u64(&data, 48).ok_or(PackedRingError::TruncatedHeader)?;
    let osp_offset = read_u64(&data, 56).ok_or(PackedRingError::TruncatedHeader)?;

    let body_end_u64 =
        u64::try_from(body_end).map_err(|_| PackedRingError::SizeOverflow { section: "body" })?;
    for (section, offset) in [
        ("dict", dict_offset),
        ("subjects", subj_offset),
        ("predicates", pred_offset),
        ("objects", obj_offset),
        ("spo_to_pos", pos_offset),
        ("spo_to_osp", osp_offset),
    ] {
        if offset > body_end_u64 {
            return Err(PackedRingError::OffsetOutOfBounds { section, offset });
        }
    }
    // Offsets must be strictly increasing.
    let offsets = [
        dict_offset,
        subj_offset,
        pred_offset,
        obj_offset,
        pos_offset,
        osp_offset,
    ];
    if dict_offset != HEADER_SIZE_U64 {
        return Err(PackedRingError::NonCanonicalLayout);
    }
    for window in offsets.windows(2) {
        if window[0] >= window[1] {
            return Err(PackedRingError::OffsetOutOfBounds {
                section: "(non-monotonic offsets)",
                offset: window[1],
            });
        }
    }

    // Slice each region. Lengths are: end-of-this to start-of-next, with
    // the last one going to body_end.
    let to_usize = |o: u64| -> Result<usize, PackedRingError> {
        usize::try_from(o).map_err(|_| PackedRingError::OffsetOutOfBounds {
            section: "(offset overflow)",
            offset: o,
        })
    };
    let dict_slice = data.slice(to_usize(dict_offset)?..to_usize(subj_offset)?);
    let subj_slice = data.slice(to_usize(subj_offset)?..to_usize(pred_offset)?);
    let pred_slice = data.slice(to_usize(pred_offset)?..to_usize(obj_offset)?);
    let obj_slice = data.slice(to_usize(obj_offset)?..to_usize(pos_offset)?);
    let pos_slice = data.slice(to_usize(pos_offset)?..to_usize(osp_offset)?);
    let osp_slice = data.slice(to_usize(osp_offset)?..body_end);

    let dict = PackedTermDictionary::from_ring_bytes(dict_slice)?;
    let subjects = deserialize_wavelet_tree(subj_slice)?;
    let predicates = deserialize_wavelet_tree(pred_slice)?;
    let objects = deserialize_wavelet_tree(obj_slice)?;
    let spo_to_pos: SuccinctPermutation = deserialize_permutation(pos_slice)?;
    let spo_to_osp: SuccinctPermutation = deserialize_permutation(osp_slice)?;

    if subjects.len() != num_triples {
        return Err(PackedRingError::NumTriplesMismatch {
            declared: num_triples,
            observed: subjects.len(),
        });
    }

    TripleRing::from_packed_parts(
        dict,
        num_triples,
        subjects,
        predicates,
        objects,
        spo_to_pos,
        spo_to_osp,
    )
    .map_err(PackedRingError::RingInvariantViolation)
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
    use crate::graph::rdf::{Term, Triple, TriplePattern};

    fn build_test_ring() -> TripleRing {
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://ex.org/gus"),
            ),
        ];
        TripleRing::from_triples(triples.into_iter())
    }

    fn encode(ring: &TripleRing) -> Vec<u8> {
        serialize_triple_ring(ring).expect("serialize packed ring")
    }

    fn rewrite_crc(bytes: &mut [u8]) {
        let body_end = bytes.len() - TRAILER_SIZE;
        let crc = crc32fast::hash(&bytes[..body_end]);
        bytes[body_end..].copy_from_slice(&crc.to_le_bytes());
    }

    #[test]
    fn alix_packed_ring_roundtrip() {
        let ring = build_test_ring();
        let bytes = encode(&ring);
        let restored = deserialize_triple_ring(Bytes::from(bytes)).expect("deserialize");

        assert_eq!(restored.len(), ring.len());
        assert_eq!(restored.num_terms(), ring.num_terms());

        // Query equivalence: foaf:name predicate should match 2 triples.
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(restored.count(&pattern), ring.count(&pattern));
        assert_eq!(restored.count(&pattern), 2);
    }

    #[test]
    fn gus_packed_ring_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let bytes = encode(&ring);
        let restored = deserialize_triple_ring(Bytes::from(bytes)).expect("deserialize empty");
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn vincent_packed_ring_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; HEADER_SIZE + TRAILER_SIZE]);
        assert_eq!(
            deserialize_triple_ring(bad).unwrap_err(),
            PackedRingError::BadMagic
        );
    }

    #[test]
    fn jules_packed_ring_truncated_header_rejected() {
        let short = Bytes::from(vec![b'G', b'R', b'F', b'R']);
        assert_eq!(
            deserialize_triple_ring(short).unwrap_err(),
            PackedRingError::TruncatedHeader
        );
    }

    #[test]
    fn mia_packed_ring_unsupported_version_rejected() {
        // Build a header that has the right magic but a wrong version
        // byte. The trailer doesn't matter for this assertion path
        // because version is checked before CRC.
        let mut buf = vec![0u8; HEADER_SIZE + TRAILER_SIZE];
        buf[..4].copy_from_slice(MAGIC);
        buf[4] = 99;
        rewrite_crc(&mut buf);
        assert_eq!(
            deserialize_triple_ring(Bytes::from(buf)).unwrap_err(),
            PackedRingError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn shosanna_packed_ring_corrupted_byte_caught_by_crc() {
        let ring = build_test_ring();
        let mut bytes = encode(&ring);
        // Flip a byte deep inside the body.
        let len = bytes.len();
        bytes[len / 2] ^= 0x01;
        let result = deserialize_triple_ring(Bytes::from(bytes));
        assert!(matches!(
            result.unwrap_err(),
            PackedRingError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn beatrix_packed_ring_query_correctness() {
        // Build a richer graph and verify several query patterns
        // round-trip correctly.
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/a"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/x"),
            ),
            Triple::new(
                Term::iri("http://ex.org/a"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/y"),
            ),
            Triple::new(
                Term::iri("http://ex.org/b"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/x"),
            ),
            Triple::new(
                Term::iri("http://ex.org/b"),
                Term::iri("http://ex.org/q"),
                Term::iri("http://ex.org/y"),
            ),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let bytes = encode(&ring);
        let restored = deserialize_triple_ring(Bytes::from(bytes)).expect("deserialize");

        // All-? pattern: count every triple.
        let all = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        assert_eq!(restored.count(&all), 4);

        // ?, p, ?: 3 matches.
        let p_only = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://ex.org/p")),
            object: None,
        };
        assert_eq!(restored.count(&p_only), 3);
        assert_eq!(restored.count(&p_only), ring.count(&p_only));

        // ?, ?, x: 2 matches.
        let x_only = TriplePattern {
            subject: None,
            predicate: None,
            object: Some(Term::iri("http://ex.org/x")),
        };
        assert_eq!(restored.count(&x_only), 2);

        // a, p, ?: 2 matches.
        let a_p = TriplePattern {
            subject: Some(Term::iri("http://ex.org/a")),
            predicate: Some(Term::iri("http://ex.org/p")),
            object: None,
        };
        assert_eq!(restored.count(&a_p), 2);
    }

    #[test]
    fn hans_packed_ring_stays_within_size_budget() {
        let triples: Vec<Triple> = (0..100u32)
            .flat_map(|i| {
                let s = format!("http://ex.org/s-{i}");
                (0..5u32).map(move |j| {
                    Triple::new(
                        Term::iri(s.clone()),
                        Term::iri(format!("http://ex.org/p-{}", j)),
                        Term::literal(format!("value-{}-{}", i, j)),
                    )
                })
            })
            .collect();
        let ring = TripleRing::from_triples(triples.into_iter());
        let bytes = encode(&ring);
        assert!(
            bytes.len() < 50_000,
            "500-triple packed Ring exceeded its 50 KiB budget: {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn reserved_header_bytes_are_rejected_with_valid_crc() {
        let mut bytes = encode(&build_test_ring());
        bytes[5] = 1;
        rewrite_crc(&mut bytes);
        assert_eq!(
            deserialize_triple_ring(Bytes::from(bytes)).unwrap_err(),
            PackedRingError::NonZeroReserved
        );
    }

    #[test]
    fn non_canonical_first_offset_is_rejected_with_valid_crc() {
        let mut bytes = encode(&build_test_ring());
        bytes[16..24].copy_from_slice(&65u64.to_le_bytes());
        rewrite_crc(&mut bytes);
        assert_eq!(
            deserialize_triple_ring(Bytes::from(bytes)).unwrap_err(),
            PackedRingError::NonCanonicalLayout
        );
    }

    #[test]
    fn crc_valid_noncanonical_term_is_rejected_before_reconstruction() {
        let term = Term::iri("http://example.org/only");
        let ring =
            TripleRing::from_triples([Triple::new(term.clone(), term.clone(), term)].into_iter());
        let mut bytes = encode(&ring);
        bytes[HEADER_SIZE + 24] = b'x';
        rewrite_crc(&mut bytes);
        assert_eq!(
            deserialize_triple_ring(Bytes::from(bytes)).unwrap_err(),
            PackedRingError::RingInvariantViolation(
                super::super::triple_ring::TripleRingInvariantError::DictionaryNonCanonicalTerm {
                    id: 0,
                }
            )
        );
    }
}
