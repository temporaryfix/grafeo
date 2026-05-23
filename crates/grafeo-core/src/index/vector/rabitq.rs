//! RaBitQ: 1-bit-per-dimension binary vector quantization (Gao & Long,
//! SIGMOD 2024) with a two-stage search.
//!
//! Each vector is unit-normalised, rotated by a fixed random orthogonal
//! matrix, then sign-quantized to one bit per dimension. A per-vector
//! correction factor unbiases a popcount-based distance estimator. The
//! coarse RaBitQ pass is reranked against int8-quantized vectors
//! ([`super::quantization::ScalarQuantizer`]) for an accurate top-K.
//!
//! A 256-dim `f32` vector (1024 B) yields a 32 B sign-bit code, plus 8 B
//! of scalar correction factors stored alongside it.

use super::quantization::{ScalarQuantizer, hamming_distance_simd};
use grafeo_common::types::NodeId;
use grafeo_common::utils::hash::FxHashMap;

/// A tiny deterministic PRNG (SplitMix64). In-tree so the codec stays
/// dependency-free and `wasm32`-friendly; seeding makes the rotation
/// reproducible across processes and platforms that store the matrix.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)` using the top 24 bits.
    fn next_f32(&mut self) -> f32 {
        // reason: 24-bit value always fits f32 mantissa exactly
        #[allow(clippy::cast_precision_loss)]
        {
            const SCALE: f32 = 1.0 / (1u32 << 24) as f32;
            (self.next_u64() >> 40) as f32 * SCALE
        }
    }

    /// Standard-normal `f32` via the Box-Muller transform.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Errors returned when opening a RaBitQ blob.
#[derive(Debug, thiserror::Error)]
pub enum RabitqError {
    /// The blob is shorter than the bytes a field needs.
    #[error("rabitq blob truncated: need {need} bytes, have {have}")]
    Truncated {
        /// Bytes required.
        need: usize,
        /// Bytes available.
        have: usize,
    },
    /// The leading magic bytes are not `GRBQ`.
    #[error("rabitq blob: bad magic (expected GRBQ)")]
    BadMagic,
    /// The version byte is not supported by this build.
    #[error("rabitq blob: unsupported version {0}")]
    BadVersion(u8),
    /// The trailing CRC32 does not match the body.
    #[error("rabitq blob: crc mismatch (stored {stored:#010x}, computed {computed:#010x})")]
    CrcMismatch {
        /// CRC read from the trailer.
        stored: u32,
        /// CRC computed over the body.
        computed: u32,
    },
    /// The embedded scalar-quantizer sub-blob failed to decode.
    #[error("rabitq blob: scalar quantizer decode failed: {0}")]
    Quantizer(String),
}

/// A fixed random orthogonal `D × D` rotation matrix.
///
/// RaBitQ rotates every data and query vector by the same matrix before
/// sign-quantizing. The rotation decorrelates the coordinates, which is
/// what gives the method its error bound over plain sign quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotation {
    dim: usize,
    /// Row-major `D × D` orthonormal matrix.
    matrix: Vec<f32>,
}

impl Rotation {
    /// Builds a random orthogonal matrix from `seed` by orthonormalising
    /// Gaussian random rows with modified Gram-Schmidt.
    ///
    /// # Panics
    /// Panics if `dim` is zero.
    #[must_use]
    pub fn new_seeded(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "rotation dimension must be > 0");
        let mut rng = SplitMix64::new(seed);
        let mut rows: Vec<Vec<f32>> = (0..dim)
            .map(|_| (0..dim).map(|_| rng.next_gaussian()).collect())
            .collect();

        for i in 0..dim {
            for j in 0..i {
                let dot: f32 = (0..dim).map(|k| rows[i][k] * rows[j][k]).sum();
                for k in 0..dim {
                    rows[i][k] -= dot * rows[j][k];
                }
            }
            let norm: f32 = rows[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                norm > f32::EPSILON,
                "rabitq rotation: row {i} collapsed to near-zero norm \
                 (PRNG produced linearly dependent rows; this should be unreachable)"
            );
            let inv = 1.0 / norm;
            for x in &mut rows[i] {
                *x *= inv;
            }
        }

        Self {
            dim,
            matrix: rows.into_iter().flatten().collect(),
        }
    }

    /// Reconstructs a rotation from an already-computed matrix (used by
    /// blob deserialization).
    ///
    /// # Panics
    /// Panics if `matrix.len() != dim * dim`.
    #[must_use]
    pub(crate) fn from_matrix(dim: usize, matrix: Vec<f32>) -> Self {
        assert_eq!(matrix.len(), dim * dim, "matrix must be dim*dim");
        Self { dim, matrix }
    }

    /// Returns the rotated vector `M · v`.
    ///
    /// # Panics
    /// Panics if `v.len() != self.dim()`.
    #[must_use]
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        (0..self.dim)
            .map(|i| {
                let row = &self.matrix[i * self.dim..(i + 1) * self.dim];
                row.iter().zip(v).map(|(&m, &x)| m * x).sum()
            })
            .collect()
    }

    /// Number of dimensions.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The raw row-major matrix (used by blob serialization).
    #[must_use]
    pub(crate) fn matrix(&self) -> &[f32] {
        &self.matrix
    }
}

/// One quantized vector: a sign bit per dimension, plus two scalar
/// correction factors. The packed bit array is `ceil(D/64)` `u64` words
/// (32 bytes for 256 dims); the factors add 8 bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct RabitqCode {
    /// Sign bits of the rotated unit vector, 64 dimensions per word.
    bits: Vec<u64>,
    /// `⟨o, ō⟩` — dot product of the rotated unit vector with its own
    /// quantized form. Unbiases the popcount estimator. Lies in `(0, 1]`.
    dot_oo: f32,
    /// Original L2 norm of the input, so Euclidean magnitude is
    /// recoverable from the stored unit-vector code. Zero for a zero input.
    norm: f32,
}

impl RabitqCode {
    /// Size of the packed sign-bit array in bytes (excludes the 8-byte
    /// correction factors).
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// The `⟨o, ō⟩` correction factor.
    #[must_use]
    pub fn dot_oo(&self) -> f32 {
        self.dot_oo
    }

    /// The original L2 norm of the encoded vector.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.norm
    }
}

/// Encodes `f32` vectors to [`RabitqCode`]s and estimates distances.
#[derive(Debug, Clone)]
pub struct RabitqQuantizer {
    dim: usize,
    seed: u64,
    rotation: Rotation,
}

impl RabitqQuantizer {
    /// Creates a quantizer for `dim`-dimensional vectors. `seed` fixes the
    /// rotation so encoding is reproducible.
    ///
    /// # Panics
    /// Panics if `dim` is zero.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim,
            seed,
            rotation: Rotation::new_seeded(dim, seed),
        }
    }

    /// Reconstructs a quantizer from a stored rotation matrix.
    #[must_use]
    pub(crate) fn from_parts(dim: usize, seed: u64, rotation: Rotation) -> Self {
        Self {
            dim,
            seed,
            rotation,
        }
    }

    /// Number of dimensions.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The rotation seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The rotation matrix (used by blob serialization).
    #[must_use]
    pub(crate) fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// Number of `u64` words in a code's bit array.
    #[must_use]
    pub fn words(&self) -> usize {
        self.dim.div_ceil(64)
    }

    /// Rotates and unit-normalises `vector`, returning `(rotated_unit, norm)`.
    fn rotate_unit(&self, vector: &[f32]) -> (Vec<f32>, f32) {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > f32::EPSILON { 1.0 / norm } else { 0.0 };
        let unit: Vec<f32> = vector.iter().map(|&x| x * inv).collect();
        (self.rotation.apply(&unit), norm)
    }

    /// Packs sign bits of `rotated` into `u64` words (bit `i` set iff
    /// `rotated[i] >= 0`). Padding bits past `dim` stay zero.
    fn sign_bits(&self, rotated: &[f32]) -> Vec<u64> {
        let mut bits = vec![0u64; self.words()];
        for (i, &x) in rotated.iter().enumerate() {
            if x >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        bits
    }

    /// Encodes a data vector to a [`RabitqCode`].
    ///
    /// # Panics
    /// Panics if `vector.len() != self.dim()`.
    #[must_use]
    pub fn encode(&self, vector: &[f32]) -> RabitqCode {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        let (rotated, norm) = self.rotate_unit(vector);
        let bits = self.sign_bits(&rotated);
        // ō_i = ±1/√D, so ⟨o, ō⟩ = (1/√D) · Σ|o_i|.
        let abs_sum: f32 = rotated.iter().map(|x| x.abs()).sum();
        // reason: dim is small and positive, cast is exact
        #[allow(clippy::cast_precision_loss)]
        let dot_oo = abs_sum / (self.dim as f32).sqrt();
        RabitqCode { bits, dot_oo, norm }
    }
}

/// A sign-quantized query vector. See [`RabitqQuantizer::encode_query`].
#[derive(Debug, Clone)]
pub struct RabitqQuery {
    bits: Vec<u64>,
    norm: f32,
}

impl RabitqQuantizer {
    /// Encodes a query vector. The query is sign-quantized the same way as
    /// data vectors, so distance estimation is a popcount over two bit
    /// arrays.
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim()`.
    #[must_use]
    pub fn encode_query(&self, query: &[f32]) -> RabitqQuery {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        let (rotated, norm) = self.rotate_unit(query);
        RabitqQuery {
            bits: self.sign_bits(&rotated),
            norm,
        }
    }

    /// Estimates the Euclidean distance between `query` and the vector
    /// behind `code`. Lower is closer.
    ///
    /// This is the coarse-stage score; [`TwoStageVectorIndex`] reranks the
    /// top candidates against int8 vectors for an accurate ordering.
    #[must_use]
    pub fn estimate_distance(&self, query: &RabitqQuery, code: &RabitqCode) -> f32 {
        // reason: dim and hamming are small non-negative integers
        #[allow(clippy::cast_precision_loss)]
        {
            let hamming = hamming_distance_simd(&query.bits, &code.bits);
            // ⟨q̄, ō⟩ for ±1/√D codebooks = (D − 2·hamming) / D.
            let ip_quant = (self.dim as f32 - 2.0 * hamming as f32) / self.dim as f32;
            // Unbias by the data-side quantization loss.
            let cos_est = (ip_quant / code.dot_oo.max(f32::MIN_POSITIVE)).clamp(-1.0, 1.0);
            // d² = |a|² + |b|² − 2|a||b|cosθ.
            let d2 = code.norm.mul_add(code.norm, query.norm * query.norm)
                - 2.0 * code.norm * query.norm * cos_est;
            d2.max(0.0).sqrt()
        }
    }
}

/// An in-memory set of RaBitQ codes supporting a coarse nearest-neighbour
/// scan. Used standalone, or as the first stage of [`TwoStageVectorIndex`].
#[derive(Debug, Clone)]
pub struct RabitqIndex {
    quantizer: RabitqQuantizer,
    ids: Vec<NodeId>,
    codes: Vec<RabitqCode>,
}

impl RabitqIndex {
    /// Creates an empty index for `dim`-dimensional vectors.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            quantizer: RabitqQuantizer::new(dim, seed),
            ids: Vec::new(),
            codes: Vec::new(),
        }
    }

    /// Creates an empty index sharing an existing quantizer.
    #[must_use]
    pub(crate) fn with_quantizer(quantizer: RabitqQuantizer) -> Self {
        Self {
            quantizer,
            ids: Vec::new(),
            codes: Vec::new(),
        }
    }

    /// Number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The underlying quantizer.
    #[must_use]
    pub fn quantizer(&self) -> &RabitqQuantizer {
        &self.quantizer
    }

    /// Encodes and stores one vector.
    pub fn insert(&mut self, id: NodeId, vector: &[f32]) {
        self.codes.push(self.quantizer.encode(vector));
        self.ids.push(id);
    }

    /// Returns the `n` nearest candidates to `query` by RaBitQ distance
    /// estimate, sorted ascending (closest first).
    #[must_use]
    pub fn coarse_search(&self, query: &[f32], n: usize) -> Vec<(NodeId, f32)> {
        let q = self.quantizer.encode_query(query);
        let mut scored: Vec<(NodeId, f32)> = self
            .ids
            .iter()
            .zip(&self.codes)
            .map(|(&id, code)| (id, self.quantizer.estimate_distance(&q, code)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Stored ids, parallel to [`Self::codes`].
    #[must_use]
    pub(crate) fn ids(&self) -> &[NodeId] {
        &self.ids
    }

    /// Stored codes, parallel to [`Self::ids`].
    #[must_use]
    pub(crate) fn codes(&self) -> &[RabitqCode] {
        &self.codes
    }

    /// Replaces the index contents with pre-decoded ids and codes
    /// (blob deserialization).
    ///
    /// # Panics
    /// Panics if `ids.len() != codes.len()`.
    pub(crate) fn load_entries(&mut self, ids: Vec<NodeId>, codes: Vec<RabitqCode>) {
        assert_eq!(
            ids.len(),
            codes.len(),
            "ids and codes must have equal length"
        );
        self.ids = ids;
        self.codes = codes;
    }
}

/// Two-stage nearest-neighbour search: a RaBitQ coarse pass over compact
/// 1-bit codes, then an int8 rerank of the top candidates for an accurate
/// ordering. Targets ~97–99% recall while the coarse codes are 32×
/// smaller than `f32` vectors.
#[derive(Debug, Clone)]
pub struct TwoStageVectorIndex {
    coarse: RabitqIndex,
    scalar: ScalarQuantizer,
    /// int8 codes parallel to `coarse.ids()`, for the rerank stage.
    int8: Vec<Vec<u8>>,
    /// `NodeId` → row offset, for O(1) candidate lookup.
    id_to_row: FxHashMap<NodeId, u32>,
}

impl TwoStageVectorIndex {
    /// Builds the index from a full set of vectors, training the int8
    /// [`ScalarQuantizer`] on the same vectors. `seed` fixes the RaBitQ
    /// rotation.
    ///
    /// # Preconditions
    /// `vectors` must not contain two entries with the same `NodeId`.
    /// In debug builds a duplicate is caught by a `debug_assert!`; in
    /// release builds the second entry's int8 row wins in `id_to_row`,
    /// leaving the first row unreachable via reranking.
    ///
    /// # Panics
    /// Panics if `vectors` is empty or any vector's length is not `dim`.
    /// In debug builds, also panics on a duplicate `NodeId`.
    #[must_use]
    pub fn build(vectors: &[(NodeId, Vec<f32>)], dim: usize, seed: u64) -> Self {
        assert!(!vectors.is_empty(), "cannot build index from no vectors");
        let refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
        let scalar = ScalarQuantizer::train(&refs);

        let mut coarse = RabitqIndex::with_quantizer(RabitqQuantizer::new(dim, seed));
        let mut int8 = Vec::with_capacity(vectors.len());
        let mut id_to_row = FxHashMap::default();
        for (row, (id, v)) in vectors.iter().enumerate() {
            assert_eq!(v.len(), dim, "vector dimension mismatch");
            coarse.insert(*id, v);
            int8.push(scalar.quantize(v));
            debug_assert!(
                !id_to_row.contains_key(id),
                "duplicate NodeId in TwoStageVectorIndex::build input"
            );
            // reason: row count bounded by input length, fits u32 for any real index
            #[allow(clippy::cast_possible_truncation)]
            id_to_row.insert(*id, row as u32);
        }
        Self {
            coarse,
            scalar,
            int8,
            id_to_row,
        }
    }

    /// Reconstructs an index from already-decoded parts (blob deserialization).
    ///
    /// # Panics
    /// Panics if `int8.len() != coarse.len()`.
    #[must_use]
    pub(crate) fn from_parts(
        coarse: RabitqIndex,
        scalar: ScalarQuantizer,
        int8: Vec<Vec<u8>>,
    ) -> Self {
        assert_eq!(
            int8.len(),
            coarse.len(),
            "int8 and coarse must have equal length"
        );
        let mut id_to_row = FxHashMap::default();
        for (row, &id) in coarse.ids().iter().enumerate() {
            // reason: row count fits u32 for any real index
            #[allow(clippy::cast_possible_truncation)]
            id_to_row.insert(id, row as u32);
        }
        Self {
            coarse,
            scalar,
            int8,
            id_to_row,
        }
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coarse.len()
    }

    /// True if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coarse.is_empty()
    }

    /// Searches for the `k` nearest neighbours of `query`.
    ///
    /// The coarse pass keeps `k · rerank_factor` candidates; the rerank
    /// pass reorders them by int8 asymmetric Euclidean distance. A larger
    /// `rerank_factor` trades query time for recall; 8–16 is typical.
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize, rerank_factor: usize) -> Vec<(NodeId, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let candidate_n = k.saturating_mul(rerank_factor.max(1)).min(self.len());
        let candidates = self.coarse.coarse_search(query, candidate_n);

        let mut reranked: Vec<(NodeId, f32)> = candidates
            .iter()
            .filter_map(|&(id, _)| {
                let row = *self.id_to_row.get(&id)? as usize;
                let dist = self.scalar.asymmetric_distance(query, &self.int8[row]);
                Some((id, dist))
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }

    /// Accessors used by blob serialization.
    #[must_use]
    pub(crate) fn parts(&self) -> (&RabitqIndex, &ScalarQuantizer, &[Vec<u8>]) {
        (&self.coarse, &self.scalar, &self.int8)
    }
}

/// Current RaBitQ blob format version.
const BLOB_VERSION: u8 = 1;

/// Appends zero bytes until `buf.len()` is a multiple of `align`.
fn pad_to(buf: &mut Vec<u8>, align: usize) {
    while !buf.len().is_multiple_of(align) {
        buf.push(0);
    }
}

/// Reads a little-endian `u32` at `*pos`, advancing `*pos`.
fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, RabitqError> {
    let end = *pos + 4;
    let slice = buf.get(*pos..end).ok_or(RabitqError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, RabitqError> {
    let end = *pos + 8;
    let slice = buf.get(*pos..end).ok_or(RabitqError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

/// Reads a little-endian `f32` at `*pos`, advancing `*pos`.
fn read_f32(buf: &[u8], pos: &mut usize) -> Result<f32, RabitqError> {
    Ok(f32::from_bits(read_u32(buf, pos)?))
}

impl TwoStageVectorIndex {
    /// Serializes the index to a self-describing, position-independent blob.
    ///
    /// The layout honours the Plan 2 zero-copy contract: a fixed header,
    /// naturally-aligned arrays, blob-relative `u64` offsets to each section,
    /// and a trailing CRC32. See `BLOB_VERSION` for the wire-format version.
    ///
    /// Header layout (64 bytes):
    /// ```text
    /// offset 0   "GRBQ"               magic (4)
    ///        4   version = 1          u8
    ///        5   flags = 0            u8
    ///        6   padding              u16
    ///        8   dim                  u32
    ///       12   count                u32
    ///       16   seed                 u64
    ///       24   words                u32
    ///       28   quantizer_len        u32
    ///       32   rotation_offset      u64   (blob-relative)
    ///       40   ids_offset           u64   (blob-relative)
    ///       48   codes_offset         u64   (blob-relative)
    ///       56   int8_offset          u64   (blob-relative)
    ///       64   quantizer bytes ... + pad to 8
    ///            ... sections in offset-table order
    ///            trailing CRC32 (4 bytes)
    /// ```
    ///
    /// # Panics
    /// Panics if the internal `ScalarQuantizer` fails to serialize via
    /// `bincode` (should never happen in practice).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let (coarse, scalar, int8) = self.parts();
        let quantizer = coarse.quantizer();
        let dim = quantizer.dim();
        let words = quantizer.words();
        let count = coarse.len();

        let quant_blob = bincode::serde::encode_to_vec(scalar, bincode::config::standard())
            .expect("ScalarQuantizer is serializable");

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GRBQ");
        buf.push(BLOB_VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // padding
        // reason: dim/count/words bounded well below u32::MAX in practice
        #[allow(clippy::cast_possible_truncation)]
        {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
            buf.extend_from_slice(&(count as u32).to_le_bytes());
            buf.extend_from_slice(&quantizer.seed().to_le_bytes());
            buf.extend_from_slice(&(words as u32).to_le_bytes());
            buf.extend_from_slice(&(quant_blob.len() as u32).to_le_bytes());
        }
        // 4×u64 placeholder for section offsets, patched at the end.
        let offsets_pos = buf.len(); // = 32
        buf.extend_from_slice(&[0u8; 32]); // 4 × u64

        buf.extend_from_slice(&quant_blob);
        pad_to(&mut buf, 8);

        // reason: section offsets fit in u64 for any realistic blob size
        #[allow(clippy::cast_possible_truncation)]
        let rotation_offset = buf.len() as u64;
        // Rotation matrix (dim*dim f32).
        for &m in quantizer.rotation().matrix() {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        // reason: section offsets fit in u64 for any realistic blob size
        #[allow(clippy::cast_possible_truncation)]
        let ids_offset = buf.len() as u64;
        // ids.
        for &id in coarse.ids() {
            buf.extend_from_slice(&id.as_u64().to_le_bytes());
        }

        // reason: section offsets fit in u64 for any realistic blob size
        #[allow(clippy::cast_possible_truncation)]
        let codes_offset = buf.len() as u64;
        // codes: bit words, then dot_oo, then norm.
        for code in coarse.codes() {
            for &w in &code.bits {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&code.dot_oo().to_le_bytes());
            buf.extend_from_slice(&code.norm().to_le_bytes());
        }

        // reason: section offsets fit in u64 for any realistic blob size
        #[allow(clippy::cast_possible_truncation)]
        let int8_offset = buf.len() as u64;
        // int8 codes.
        for row in int8 {
            buf.extend_from_slice(row);
        }
        pad_to(&mut buf, 4);

        // Patch offsets into placeholder.
        buf[offsets_pos..offsets_pos + 8].copy_from_slice(&rotation_offset.to_le_bytes());
        buf[offsets_pos + 8..offsets_pos + 16].copy_from_slice(&ids_offset.to_le_bytes());
        buf[offsets_pos + 16..offsets_pos + 24].copy_from_slice(&codes_offset.to_le_bytes());
        buf[offsets_pos + 24..offsets_pos + 32].copy_from_slice(&int8_offset.to_le_bytes());

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`RabitqError`] on a bad magic, unsupported version,
    /// truncation, CRC mismatch, or a corrupt scalar-quantizer sub-blob.
    ///
    /// # Panics
    /// Panics if the trailing 4-byte CRC slice cannot be converted to `[u8; 4]`
    /// (impossible when the buffer length check at entry passes).
    pub fn from_bytes(buf: &[u8]) -> Result<Self, RabitqError> {
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
        // Verify CRC over everything but the trailing 4 bytes.
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
        // reason: offsets fit in usize on any platform that can hold the blob in memory
        #[allow(clippy::cast_possible_truncation)]
        let rotation_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let ids_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let codes_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let int8_offset = read_u64(buf, &mut pos)? as usize;

        let quant_slice = buf
            .get(pos..pos + quant_len)
            .ok_or(RabitqError::Truncated {
                need: pos + quant_len,
                have: buf.len(),
            })?;
        let (scalar, _): (ScalarQuantizer, usize) =
            bincode::serde::decode_from_slice(quant_slice, bincode::config::standard())
                .map_err(|e| RabitqError::Quantizer(e.to_string()))?;
        pos += quant_len;
        pos = pos.next_multiple_of(8);
        debug_assert_eq!(
            pos, rotation_offset,
            "rotation_offset mismatch: writer/reader diverged"
        );

        // Rotation matrix.
        let mut matrix = Vec::with_capacity(dim * dim);
        for _ in 0..dim * dim {
            matrix.push(read_f32(buf, &mut pos)?);
        }
        pos = pos.next_multiple_of(8);
        debug_assert_eq!(
            pos, ids_offset,
            "ids_offset mismatch: writer/reader diverged"
        );

        let quantizer = RabitqQuantizer::from_parts(dim, seed, Rotation::from_matrix(dim, matrix));
        let mut coarse = RabitqIndex::with_quantizer(quantizer);

        // ids.
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(NodeId::new(read_u64(buf, &mut pos)?));
        }
        debug_assert_eq!(
            pos, codes_offset,
            "codes_offset mismatch: writer/reader diverged"
        );

        // codes.
        let mut codes = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bits = Vec::with_capacity(words);
            for _ in 0..words {
                bits.push(read_u64(buf, &mut pos)?);
            }
            let dot_oo = read_f32(buf, &mut pos)?;
            let norm = read_f32(buf, &mut pos)?;
            codes.push(RabitqCode { bits, dot_oo, norm });
        }
        debug_assert_eq!(
            pos, int8_offset,
            "int8_offset mismatch: writer/reader diverged"
        );
        coarse.load_entries(ids, codes);

        // int8 codes.
        let mut int8 = Vec::with_capacity(count);
        for _ in 0..count {
            let slice = buf.get(pos..pos + dim).ok_or(RabitqError::Truncated {
                need: pos + dim,
                have: buf.len(),
            })?;
            int8.push(slice.to_vec());
            pos += dim;
        }

        Ok(Self::from_parts(coarse, scalar, int8))
    }

    /// Opens a blob shared via [`bytes::Bytes`] into an owned index.
    ///
    /// This still copies the codes and int8 arrays into owned `Vec`s.
    /// For a true borrowing reader that holds `Bytes` slices and decodes
    /// query-hot data on demand, use [`RabitqView::open`] instead.
    ///
    /// # Errors
    /// Same as [`Self::from_bytes`].
    pub fn from_bytes_shared(blob: bytes::Bytes) -> Result<Self, RabitqError> {
        Self::from_bytes(&blob)
    }
}

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
    ///
    /// # Panics
    /// Panics if the CRC trailer slice is not exactly 4 bytes — an internal
    /// invariant upheld by `to_bytes` that cannot occur on a well-formed blob.
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
        // reason: offsets fit in usize on any platform that can hold the blob in memory
        #[allow(clippy::cast_possible_truncation)]
        let rotation_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let ids_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let codes_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
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
        let rotation_quantizer =
            RabitqQuantizer::from_parts(dim, seed, Rotation::from_matrix(dim, matrix));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix64_gaussian_is_roughly_centred() {
        let mut rng = SplitMix64::new(7);
        let n = 50_000;
        let mean: f32 = (0..n).map(|_| rng.next_gaussian()).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.01, "gaussian mean drifted: {mean}");
    }

    #[test]
    fn rotation_preserves_l2_norm() {
        let rot = Rotation::new_seeded(64, 123);
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let norm_in: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rotated = rot.apply(&v);
        let norm_out: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_in - norm_out).abs() < 1e-3,
            "rotation changed norm: {norm_in} -> {norm_out}"
        );
    }

    #[test]
    fn rotation_is_deterministic_for_a_seed() {
        let a = Rotation::new_seeded(32, 99);
        let b = Rotation::new_seeded(32, 99);
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(a.apply(&v), b.apply(&v));
    }

    #[test]
    fn encode_256_dim_yields_32_byte_code() {
        let q = RabitqQuantizer::new(256, 1);
        let v: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let code = q.encode(&v);
        assert_eq!(
            code.code_bytes(),
            32,
            "256 sign bits must pack into 32 bytes"
        );
        assert!(code.dot_oo() > 0.0 && code.dot_oo() <= 1.0 + 1e-4);
        assert!(code.norm() > 0.0);
    }

    #[test]
    fn encode_zero_vector_does_not_panic() {
        let q = RabitqQuantizer::new(8, 1);
        let code = q.encode(&[0.0; 8]);
        assert_eq!(code.norm(), 0.0);
    }

    #[test]
    fn encode_non_multiple_of_64_dim() {
        // 100 dims -> ceil(100/64) = 2 words = 16 bytes.
        let q = RabitqQuantizer::new(100, 5);
        let v: Vec<f32> = (0..100).map(|i| i as f32 - 50.0).collect();
        assert_eq!(q.encode(&v).code_bytes(), 16);
    }

    #[test]
    fn estimate_distance_ranks_self_closest() {
        let q = RabitqQuantizer::new(128, 3);
        let target: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let far: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).cos() * 3.0).collect();

        let query = q.encode_query(&target);
        let code_self = q.encode(&target);
        let code_far = q.encode(&far);

        let d_self = q.estimate_distance(&query, &code_self);
        let d_far = q.estimate_distance(&query, &code_far);
        assert!(
            d_self < d_far,
            "self {d_self} should be closer than far {d_far}"
        );
        assert!(d_self >= 0.0);
    }

    #[test]
    fn estimate_distance_orders_a_small_set() {
        let q = RabitqQuantizer::new(64, 11);
        let base: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let query = q.encode_query(&base);

        // Increasingly perturbed copies must estimate increasingly far.
        let mut last = -1.0f32;
        for scale in [0.0f32, 0.5, 1.0, 2.0] {
            let v: Vec<f32> = base.iter().map(|&x| x + scale).collect();
            let d = q.estimate_distance(&query, &q.encode(&v));
            assert!(
                d >= last - 0.5,
                "distance not monotone at scale {scale}: {d} < {last}"
            );
            last = d;
        }
    }

    #[test]
    fn rabitq_index_coarse_search_returns_sorted_candidates() {
        use grafeo_common::types::NodeId;

        let mut index = RabitqIndex::new(32, 17);
        // Cluster A near 0.0, cluster B near 5.0.
        for i in 0..10 {
            let a: Vec<f32> = (0..32)
                .map(|d| (d as f32 * 0.1).sin() + i as f32 * 0.01)
                .collect();
            index.insert(NodeId::new(i + 1), &a);
        }
        for i in 0..10 {
            let b: Vec<f32> = (0..32).map(|d| (d as f32 * 0.1).sin() + 5.0).collect();
            index.insert(NodeId::new(100 + i), &b);
        }
        assert_eq!(index.len(), 20);

        let query: Vec<f32> = (0..32).map(|d| (d as f32 * 0.1).sin()).collect();
        let hits = index.coarse_search(&query, 5);
        assert_eq!(hits.len(), 5);
        // Sorted ascending by estimated distance.
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        // The nearest hits should come from cluster A (ids 1..=10).
        assert!(hits[0].0.as_u64() <= 10, "nearest hit not from cluster A");
    }

    #[test]
    fn two_stage_search_beats_coarse_alone_on_recall() {
        use grafeo_common::types::NodeId;

        let dim = 64;
        // 6 well-separated clusters of 20 points each.
        let mut rng = SplitMix64::new(2024);
        let mut centres: Vec<Vec<f32>> = Vec::new();
        for _ in 0..6 {
            centres.push((0..dim).map(|_| rng.next_gaussian() * 5.0).collect());
        }
        let mut vectors: Vec<(NodeId, Vec<f32>)> = Vec::new();
        let mut id = 1u64;
        for centre in &centres {
            for _ in 0..20 {
                let v: Vec<f32> = centre
                    .iter()
                    .map(|&c| c + rng.next_gaussian() * 0.3)
                    .collect();
                vectors.push((NodeId::new(id), v));
                id += 1;
            }
        }

        let index = TwoStageVectorIndex::build(&vectors, dim, 1);
        assert_eq!(index.len(), 120);

        // Query = first point of cluster 0; its 10 true neighbours are in cluster 0.
        let query = vectors[0].1.clone();
        let hits = index.search(&query, 10, 16);
        assert_eq!(hits.len(), 10);
        // Ascending distance.
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        // All 10 should be cluster-0 points (ids 1..=20).
        let from_cluster0 = hits.iter().filter(|(id, _)| id.as_u64() <= 20).count();
        assert!(
            from_cluster0 >= 9,
            "expected >=9 cluster-0 hits, got {from_cluster0}"
        );
    }

    #[test]
    fn two_stage_search_empty_and_k_zero() {
        use grafeo_common::types::NodeId;
        let vectors = vec![(NodeId::new(1), vec![1.0f32; 8])];
        let index = TwoStageVectorIndex::build(&vectors, 8, 1);
        assert!(index.search(&[1.0; 8], 0, 4).is_empty());
        assert_eq!(index.search(&[1.0; 8], 5, 4).len(), 1);
    }

    #[test]
    fn blob_round_trip_preserves_search_results() {
        use grafeo_common::types::NodeId;

        let dim = 48;
        let mut rng = SplitMix64::new(555);
        let vectors: Vec<(NodeId, Vec<f32>)> = (0..80)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.next_gaussian()).collect();
                (NodeId::new(i + 1), v)
            })
            .collect();

        let index = TwoStageVectorIndex::build(&vectors, dim, 9);
        let blob = index.to_bytes();

        // Header contract: magic, version, 8-byte aligned total length.
        assert_eq!(&blob[0..4], b"GRBQ");
        assert_eq!(blob[4], 1);

        // New header fields (offsets at 32, 40, 48, 56).
        let rotation_offset = u64::from_le_bytes(blob[32..40].try_into().unwrap());
        let ids_offset = u64::from_le_bytes(blob[40..48].try_into().unwrap());
        let codes_offset = u64::from_le_bytes(blob[48..56].try_into().unwrap());
        let int8_offset = u64::from_le_bytes(blob[56..64].try_into().unwrap());
        assert!(rotation_offset >= 64);
        assert!(rotation_offset < ids_offset);
        assert!(ids_offset < codes_offset);
        assert!(codes_offset < int8_offset);
        assert!(int8_offset < blob.len() as u64 - 4);

        let reopened = TwoStageVectorIndex::from_bytes(&blob).expect("from_bytes");
        assert_eq!(reopened.len(), index.len());

        // Identical query results before and after a round trip.
        let query = vectors[3].1.clone();
        assert_eq!(index.search(&query, 10, 8), reopened.search(&query, 10, 8),);
    }

    #[test]
    fn blob_rejects_bad_magic_and_crc() {
        use grafeo_common::types::NodeId;
        let vectors = vec![(NodeId::new(1), vec![1.0f32; 8])];
        let mut blob = TwoStageVectorIndex::build(&vectors, 8, 1).to_bytes();

        let mut bad_magic = blob.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            TwoStageVectorIndex::from_bytes(&bad_magic),
            Err(RabitqError::BadMagic)
        ));

        // Corrupt a body byte; the trailing CRC must catch it.
        let mid = blob.len() / 2;
        blob[mid] ^= 0xFF;
        assert!(matches!(
            TwoStageVectorIndex::from_bytes(&blob),
            Err(RabitqError::CrcMismatch { .. })
        ));
    }

    #[test]
    fn blob_from_bytes_shared_round_trip() {
        use grafeo_common::types::NodeId;
        let vectors = vec![(NodeId::new(1), vec![1.0f32; 8])];
        let index = TwoStageVectorIndex::build(&vectors, 8, 1);
        let blob = bytes::Bytes::from(index.to_bytes());
        let reopened = TwoStageVectorIndex::from_bytes_shared(blob).expect("from_bytes_shared");
        assert_eq!(reopened.len(), 1);
    }

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
}
