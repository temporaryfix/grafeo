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
    /// A vector or reconstructed component has the wrong dimension.
    #[error("rabitq dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Required dimension.
        expected: usize,
        /// Supplied dimension.
        actual: usize,
    },
    /// A required collection or dimension was empty.
    #[error("rabitq invalid input: {0}")]
    InvalidInput(&'static str),
    /// The build input repeated a node identifier.
    #[error("rabitq duplicate node id: {0:?}")]
    DuplicateNodeId(NodeId),
    /// Decoded or supplied components disagree structurally.
    #[error("rabitq invariant violation: {0}")]
    InvariantViolation(&'static str),
    /// Size or offset arithmetic overflowed.
    #[error("rabitq size overflow in {0}")]
    SizeOverflow(&'static str),
    /// A section offset, width, reserved field, or padding byte is invalid.
    #[error("rabitq blob: invalid layout: {0}")]
    InvalidLayout(&'static str),
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
    /// # Errors
    ///
    /// Returns an error if `dim` is zero or a generated row degenerates.
    pub fn new_seeded(dim: usize, seed: u64) -> Result<Self, RabitqError> {
        if dim == 0 {
            return Err(RabitqError::InvalidInput(
                "rotation dimension must be greater than zero",
            ));
        }
        dim.checked_mul(dim)
            .ok_or(RabitqError::SizeOverflow("rotation matrix"))?;
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
            if norm <= f32::EPSILON {
                return Err(RabitqError::InvariantViolation(
                    "generated rotation row collapsed to a near-zero norm",
                ));
            }
            let inv = 1.0 / norm;
            for x in &mut rows[i] {
                *x *= inv;
            }
        }

        Ok(Self {
            dim,
            matrix: rows.into_iter().flatten().collect(),
        })
    }

    /// Reconstructs a rotation from an already-computed matrix (used by
    /// blob deserialization).
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension is zero or the matrix is not square.
    pub(crate) fn from_matrix(dim: usize, matrix: Vec<f32>) -> Result<Self, RabitqError> {
        if dim == 0 {
            return Err(RabitqError::InvalidInput(
                "rotation dimension must be greater than zero",
            ));
        }
        let expected = dim
            .checked_mul(dim)
            .ok_or(RabitqError::SizeOverflow("rotation matrix"))?;
        if matrix.len() != expected {
            return Err(RabitqError::DimensionMismatch {
                expected,
                actual: matrix.len(),
            });
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(RabitqError::InvalidInput(
                "rotation matrix values must be finite",
            ));
        }
        Ok(Self { dim, matrix })
    }

    /// Returns the rotated vector `M · v`.
    ///
    /// # Errors
    ///
    /// Returns an error if `v` has the wrong dimension.
    pub fn apply(&self, v: &[f32]) -> Result<Vec<f32>, RabitqError> {
        if v.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: v.len(),
            });
        }
        if v.iter().any(|value| !value.is_finite()) {
            return Err(RabitqError::InvalidInput("vector values must be finite"));
        }
        Ok(self.apply_validated(v))
    }

    fn apply_validated(&self, v: &[f32]) -> Vec<f32> {
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
    /// # Errors
    ///
    /// Returns an error if `dim` is zero or rotation construction fails.
    pub fn new(dim: usize, seed: u64) -> Result<Self, RabitqError> {
        Ok(Self {
            dim,
            seed,
            rotation: Rotation::new_seeded(dim, seed)?,
        })
    }

    /// Reconstructs a quantizer from a stored rotation matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the rotation and declared dimensions disagree.
    pub(crate) fn from_parts(
        dim: usize,
        seed: u64,
        rotation: Rotation,
    ) -> Result<Self, RabitqError> {
        if dim == 0 {
            return Err(RabitqError::InvalidInput(
                "quantizer dimension must be greater than zero",
            ));
        }
        if rotation.dim() != dim {
            return Err(RabitqError::DimensionMismatch {
                expected: dim,
                actual: rotation.dim(),
            });
        }
        Ok(Self {
            dim,
            seed,
            rotation,
        })
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
    fn rotate_unit(&self, vector: &[f32]) -> Result<(Vec<f32>, f32), RabitqError> {
        if vector.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(RabitqError::InvalidInput("vector values must be finite"));
        }
        let norm_f64 = vector
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>()
            .sqrt();
        if !norm_f64.is_finite() || norm_f64 > f64::from(f32::MAX) {
            return Err(RabitqError::InvalidInput(
                "vector norm must be representable as f32",
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        let norm = norm_f64 as f32;
        let inv = if norm > f32::EPSILON { 1.0 / norm } else { 0.0 };
        let unit: Vec<f32> = vector.iter().map(|&x| x * inv).collect();
        Ok((self.rotation.apply_validated(&unit), norm))
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
    /// # Errors
    ///
    /// Returns an error if `vector` has the wrong dimension.
    pub fn encode(&self, vector: &[f32]) -> Result<RabitqCode, RabitqError> {
        let (rotated, norm) = self.rotate_unit(vector)?;
        let bits = self.sign_bits(&rotated);
        // ō_i = ±1/√D, so ⟨o, ō⟩ = (1/√D) · Σ|o_i|.
        let abs_sum: f32 = rotated.iter().map(|x| x.abs()).sum();
        // reason: dim is small and positive, cast is exact
        #[allow(clippy::cast_precision_loss)]
        let dot_oo = abs_sum / (self.dim as f32).sqrt();
        Ok(RabitqCode { bits, dot_oo, norm })
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
    /// # Errors
    ///
    /// Returns an error if `query` has the wrong dimension.
    pub fn encode_query(&self, query: &[f32]) -> Result<RabitqQuery, RabitqError> {
        let (rotated, norm) = self.rotate_unit(query)?;
        Ok(RabitqQuery {
            bits: self.sign_bits(&rotated),
            norm,
        })
    }

    /// Estimates the Euclidean distance between `query` and the vector
    /// behind `code`. Lower is closer.
    ///
    /// This is the coarse-stage score; [`TwoStageVectorIndex`] reranks the
    /// top candidates against int8 vectors for an accurate ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if either packed code width does not match this
    /// quantizer.
    pub fn estimate_distance(
        &self,
        query: &RabitqQuery,
        code: &RabitqCode,
    ) -> Result<f32, RabitqError> {
        let words = self.words();
        if query.bits.len() != words || code.bits.len() != words {
            return Err(RabitqError::InvariantViolation(
                "packed code width does not match the quantizer dimension",
            ));
        }
        Ok(self.estimate_distance_validated(query, code))
    }

    /// Computes a score after the owning index or blob parser has established
    /// that both packed widths match this quantizer.
    fn estimate_distance_validated(&self, query: &RabitqQuery, code: &RabitqCode) -> f32 {
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
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension or rotation is invalid.
    pub fn new(dim: usize, seed: u64) -> Result<Self, RabitqError> {
        Ok(Self {
            quantizer: RabitqQuantizer::new(dim, seed)?,
            ids: Vec::new(),
            codes: Vec::new(),
        })
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
    ///
    /// # Errors
    ///
    /// Returns an error if `vector` has the wrong dimension.
    pub fn insert(&mut self, id: NodeId, vector: &[f32]) -> Result<(), RabitqError> {
        let code = self.quantizer.encode(vector)?;
        self.codes.push(code);
        self.ids.push(id);
        Ok(())
    }

    /// Returns the `n` nearest candidates to `query` by RaBitQ distance
    /// estimate, sorted ascending (closest first).
    ///
    /// # Errors
    ///
    /// Returns an error if `query` has the wrong dimension.
    pub fn coarse_search(
        &self,
        query: &[f32],
        n: usize,
    ) -> Result<Vec<(NodeId, f32)>, RabitqError> {
        if query.len() != self.quantizer.dim() {
            return Err(RabitqError::DimensionMismatch {
                expected: self.quantizer.dim(),
                actual: query.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let q = self.quantizer.encode_query(query)?;
        let mut scored: Vec<(NodeId, f32)> = self
            .ids
            .iter()
            .zip(&self.codes)
            .map(|(&id, code)| (id, self.quantizer.estimate_distance_validated(&q, code)))
            .collect();
        let cmp = |a: &(NodeId, f32), b: &(NodeId, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        // Select the n smallest (unordered), then sort just those —
        // O(N + n log n) instead of O(N log N).
        if n < scored.len() {
            scored.select_nth_unstable_by(n - 1, cmp);
            scored.truncate(n);
        }
        scored.sort_unstable_by(cmp);
        Ok(scored)
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
    /// # Errors
    ///
    /// Returns an error if IDs and codes do not have the same length or a
    /// code has the wrong packed width.
    pub(crate) fn load_entries(
        &mut self,
        ids: Vec<NodeId>,
        codes: Vec<RabitqCode>,
    ) -> Result<(), RabitqError> {
        if ids.len() != codes.len() {
            return Err(RabitqError::InvariantViolation(
                "ids and codes must have equal length",
            ));
        }
        if codes
            .iter()
            .any(|code| code.bits.len() != self.quantizer.words())
        {
            return Err(RabitqError::InvariantViolation(
                "code width does not match the quantizer dimension",
            ));
        }
        self.ids = ids;
        self.codes = codes;
        Ok(())
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
    /// # Errors
    ///
    /// Returns an error when the dimension or vectors are empty, a vector has
    /// the wrong dimension, an ID is duplicated, or the index is too large for
    /// its on-disk row representation.
    pub fn build(
        vectors: &[(NodeId, Vec<f32>)],
        dim: usize,
        seed: u64,
    ) -> Result<Self, RabitqError> {
        if dim == 0 {
            return Err(RabitqError::InvalidInput(
                "index dimension must be greater than zero",
            ));
        }
        if vectors.is_empty() {
            return Err(RabitqError::InvalidInput(
                "cannot build an index from no vectors",
            ));
        }
        u32::try_from(vectors.len()).map_err(|_| RabitqError::SizeOverflow("entry count"))?;
        let mut id_to_row = FxHashMap::with_capacity_and_hasher(vectors.len(), Default::default());
        for (row, (id, vector)) in vectors.iter().enumerate() {
            if vector.len() != dim {
                return Err(RabitqError::DimensionMismatch {
                    expected: dim,
                    actual: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(RabitqError::InvalidInput("vector values must be finite"));
            }
            let row =
                u32::try_from(row).map_err(|_| RabitqError::SizeOverflow("index row number"))?;
            if id_to_row.insert(*id, row).is_some() {
                return Err(RabitqError::DuplicateNodeId(*id));
            }
        }

        let refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
        let scalar = ScalarQuantizer::train(&refs);
        if !scalar.has_valid_storage() {
            return Err(RabitqError::InvalidInput(
                "training data exceeds the scalar quantizer's numeric range",
            ));
        }

        let mut coarse = RabitqIndex::with_quantizer(RabitqQuantizer::new(dim, seed)?);
        let mut int8 = Vec::with_capacity(vectors.len());
        for (id, v) in vectors {
            coarse.insert(*id, v)?;
            int8.push(scalar.quantize(v));
        }
        Ok(Self {
            coarse,
            scalar,
            int8,
            id_to_row,
        })
    }

    /// Reconstructs an index from already-decoded parts (blob deserialization).
    ///
    /// # Errors
    ///
    /// Returns an error if the component counts, dimensions, or IDs disagree.
    pub(crate) fn from_parts(
        coarse: RabitqIndex,
        scalar: ScalarQuantizer,
        int8: Vec<Vec<u8>>,
    ) -> Result<Self, RabitqError> {
        let dim = coarse.quantizer().dim();
        if int8.len() != coarse.len() {
            return Err(RabitqError::InvariantViolation(
                "int8 and coarse entries must have equal length",
            ));
        }
        if scalar.dimensions() != dim {
            return Err(RabitqError::DimensionMismatch {
                expected: dim,
                actual: scalar.dimensions(),
            });
        }
        if int8.iter().any(|row| row.len() != dim) {
            return Err(RabitqError::InvariantViolation(
                "int8 row width does not match the index dimension",
            ));
        }
        let mut id_to_row = FxHashMap::with_capacity_and_hasher(coarse.len(), Default::default());
        for (row, &id) in coarse.ids().iter().enumerate() {
            let row =
                u32::try_from(row).map_err(|_| RabitqError::SizeOverflow("index row number"))?;
            if id_to_row.insert(id, row).is_some() {
                return Err(RabitqError::DuplicateNodeId(id));
            }
        }
        Ok(Self {
            coarse,
            scalar,
            int8,
            id_to_row,
        })
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
    ///
    /// # Errors
    ///
    /// Returns an error if `query` has the wrong dimension or the index's
    /// internal row mapping is inconsistent.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
    ) -> Result<Vec<(NodeId, f32)>, RabitqError> {
        if query.len() != self.scalar.dimensions() {
            return Err(RabitqError::DimensionMismatch {
                expected: self.scalar.dimensions(),
                actual: query.len(),
            });
        }
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let candidate_n = k.saturating_mul(rerank_factor.max(1)).min(self.len());
        let candidates = self.coarse.coarse_search(query, candidate_n)?;

        let mut reranked = Vec::with_capacity(candidates.len());
        for (id, _) in candidates {
            let row = self
                .id_to_row
                .get(&id)
                .copied()
                .ok_or(RabitqError::InvariantViolation(
                    "candidate ID is missing from the row map",
                ))? as usize;
            let code = self.int8.get(row).ok_or(RabitqError::InvariantViolation(
                "candidate row is outside the int8 code table",
            ))?;
            reranked.push((id, self.scalar.asymmetric_distance(query, code)));
        }
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked)
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

#[inline]
fn checked_add(start: usize, len: usize, field: &'static str) -> Result<usize, RabitqError> {
    start
        .checked_add(len)
        .ok_or(RabitqError::SizeOverflow(field))
}

#[inline]
fn checked_mul(left: usize, right: usize, field: &'static str) -> Result<usize, RabitqError> {
    left.checked_mul(right)
        .ok_or(RabitqError::SizeOverflow(field))
}

#[inline]
fn checked_align(value: usize, align: usize, field: &'static str) -> Result<usize, RabitqError> {
    let mask = align - 1;
    value
        .checked_add(mask)
        .map(|end| end & !mask)
        .ok_or(RabitqError::SizeOverflow(field))
}

#[inline]
fn checked_slice<'a>(
    buf: &'a [u8],
    start: usize,
    len: usize,
    limit: usize,
    field: &'static str,
) -> Result<&'a [u8], RabitqError> {
    let end = checked_add(start, len, field)?;
    if end > limit {
        return Err(RabitqError::Truncated {
            need: end,
            have: limit,
        });
    }
    buf.get(start..end).ok_or(RabitqError::Truncated {
        need: end,
        have: buf.len(),
    })
}

#[inline]
fn read_array<const N: usize>(buf: &[u8], pos: &mut usize) -> Result<[u8; N], RabitqError> {
    let end = checked_add(*pos, N, "fixed-width field")?;
    let slice = buf.get(*pos..end).ok_or(RabitqError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    let bytes = <[u8; N]>::try_from(slice)
        .map_err(|_| RabitqError::InvariantViolation("fixed-width field has the wrong length"))?;
    *pos = end;
    Ok(bytes)
}

/// Reads a little-endian `u32` at `*pos`, advancing `*pos`.
#[inline]
fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, RabitqError> {
    Ok(u32::from_le_bytes(read_array(buf, pos)?))
}

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
#[inline]
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, RabitqError> {
    Ok(u64::from_le_bytes(read_array(buf, pos)?))
}

/// Reads a little-endian `f32` at `*pos`, advancing `*pos`.
#[inline]
fn read_f32(buf: &[u8], pos: &mut usize) -> Result<f32, RabitqError> {
    Ok(f32::from_bits(read_u32(buf, pos)?))
}

fn require_zeroes(bytes: &[u8], field: &'static str) -> Result<(), RabitqError> {
    if bytes.iter().any(|&byte| byte != 0) {
        return Err(RabitqError::InvalidLayout(field));
    }
    Ok(())
}

fn require_offset(actual: usize, expected: usize, field: &'static str) -> Result<(), RabitqError> {
    if actual != expected {
        return Err(RabitqError::InvalidLayout(field));
    }
    Ok(())
}

struct ParsedRabitqBlob {
    dim: usize,
    count: usize,
    words: usize,
    scalar: ScalarQuantizer,
    rotation_quantizer: RabitqQuantizer,
    ids: Vec<NodeId>,
    codes_offset: usize,
    code_stride: usize,
    int8_offset: usize,
}

/// Validates the complete canonical layout shared by the owned and borrowing
/// readers, then decodes their small common state.
fn parse_rabitq_blob(buf: &[u8]) -> Result<ParsedRabitqBlob, RabitqError> {
    if buf.len() < 8 {
        return Err(RabitqError::Truncated {
            need: 8,
            have: buf.len(),
        });
    }
    if buf.get(..4) != Some(b"GRBQ") {
        return Err(RabitqError::BadMagic);
    }
    if buf[4] != BLOB_VERSION {
        return Err(RabitqError::BadVersion(buf[4]));
    }
    require_zeroes(&buf[5..8], "flags and reserved header bytes must be zero")?;

    let body_end = buf.len() - 4;
    let mut crc_pos = body_end;
    let stored = read_u32(buf, &mut crc_pos)?;
    let computed = crc32fast::hash(&buf[..body_end]);
    if stored != computed {
        return Err(RabitqError::CrcMismatch { stored, computed });
    }

    let mut pos = 8;
    let dim = usize::try_from(read_u32(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("dimension"))?;
    let count = usize::try_from(read_u32(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("entry count"))?;
    let seed = read_u64(buf, &mut pos)?;
    let words = usize::try_from(read_u32(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("code word count"))?;
    let quant_len = usize::try_from(read_u32(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("scalar quantizer blob"))?;
    let rotation_offset = usize::try_from(read_u64(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("rotation offset"))?;
    let ids_offset = usize::try_from(read_u64(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("IDs offset"))?;
    let codes_offset = usize::try_from(read_u64(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("codes offset"))?;
    let int8_offset = usize::try_from(read_u64(buf, &mut pos)?)
        .map_err(|_| RabitqError::SizeOverflow("int8 offset"))?;

    if dim == 0 {
        return Err(RabitqError::InvalidLayout(
            "dimension must be greater than zero",
        ));
    }
    if count == 0 {
        return Err(RabitqError::InvalidLayout(
            "entry count must be greater than zero",
        ));
    }
    if words != dim.div_ceil(64) {
        return Err(RabitqError::InvalidLayout(
            "code word count does not match the dimension",
        ));
    }

    let expected_quant_len = checked_mul(
        checked_mul(dim, 3, "scalar quantizer values")?,
        size_of::<f32>(),
        "scalar quantizer bytes",
    )?;
    if quant_len != expected_quant_len {
        return Err(RabitqError::InvalidLayout(
            "scalar quantizer length does not match the dimension",
        ));
    }
    let quant_slice = checked_slice(buf, pos, quant_len, body_end, "scalar quantizer section")?;
    let quant_end = checked_add(pos, quant_len, "scalar quantizer section")?;
    let expected_rotation = checked_align(quant_end, 8, "rotation alignment")?;
    require_offset(
        rotation_offset,
        expected_rotation,
        "rotation offset is not canonical",
    )?;
    require_zeroes(
        checked_slice(
            buf,
            quant_end,
            rotation_offset - quant_end,
            body_end,
            "scalar quantizer padding",
        )?,
        "scalar quantizer padding must be zero",
    )?;

    let matrix_values = checked_mul(dim, dim, "rotation matrix elements")?;
    let matrix_bytes = checked_mul(matrix_values, 4, "rotation matrix bytes")?;
    let rotation_end = checked_add(rotation_offset, matrix_bytes, "rotation matrix")?;
    checked_slice(
        buf,
        rotation_offset,
        matrix_bytes,
        body_end,
        "rotation matrix",
    )?;
    let expected_ids = checked_align(rotation_end, 8, "IDs alignment")?;
    require_offset(ids_offset, expected_ids, "IDs offset is not canonical")?;
    require_zeroes(
        checked_slice(
            buf,
            rotation_end,
            ids_offset - rotation_end,
            body_end,
            "rotation padding",
        )?,
        "rotation padding must be zero",
    )?;

    let ids_bytes = checked_mul(count, 8, "ID table")?;
    let expected_codes = checked_add(ids_offset, ids_bytes, "ID table")?;
    require_offset(
        codes_offset,
        expected_codes,
        "codes offset is not canonical",
    )?;
    checked_slice(buf, ids_offset, ids_bytes, body_end, "ID table")?;

    let bits_bytes = checked_mul(words, 8, "code bits")?;
    let code_stride = checked_add(bits_bytes, 8, "code stride")?;
    let codes_bytes = checked_mul(count, code_stride, "codes section")?;
    let expected_int8 = checked_add(codes_offset, codes_bytes, "codes section")?;
    require_offset(int8_offset, expected_int8, "int8 offset is not canonical")?;
    let codes_slice = checked_slice(buf, codes_offset, codes_bytes, body_end, "codes section")?;

    let used_bits_in_last_word = dim % 64;
    for row_bytes in codes_slice.chunks_exact(code_stride) {
        if used_bits_in_last_word != 0 {
            let last_word_offset = checked_mul(words - 1, 8, "last code word offset")?;
            let mut last_word_pos = last_word_offset;
            let last_word = read_u64(row_bytes, &mut last_word_pos)?;
            let padding_mask = !((1u64 << used_bits_in_last_word) - 1);
            if last_word & padding_mask != 0 {
                return Err(RabitqError::InvalidLayout(
                    "unused sign-code bits must be zero",
                ));
            }
        }
        let mut factors_pos = bits_bytes;
        let dot_oo = read_f32(row_bytes, &mut factors_pos)?;
        let norm = read_f32(row_bytes, &mut factors_pos)?;
        if !dot_oo.is_finite()
            || !(0.0..=1.0 + 1e-4).contains(&dot_oo)
            || !norm.is_finite()
            || norm < 0.0
        {
            return Err(RabitqError::InvalidLayout(
                "code factors are outside their finite canonical range",
            ));
        }
    }

    let int8_bytes = checked_mul(count, dim, "int8 section")?;
    let int8_end = checked_add(int8_offset, int8_bytes, "int8 section")?;
    checked_slice(buf, int8_offset, int8_bytes, body_end, "int8 section")?;
    let expected_body_end = checked_align(int8_end, 4, "CRC alignment")?;
    require_offset(
        body_end,
        expected_body_end,
        "blob has trailing or missing data",
    )?;
    require_zeroes(
        checked_slice(buf, int8_end, body_end - int8_end, body_end, "int8 padding")?,
        "int8 padding must be zero",
    )?;

    let mut quant_pos = 0;
    let mut read_quant_values = || -> Result<Vec<f32>, RabitqError> {
        let mut values = Vec::with_capacity(dim);
        for _ in 0..dim {
            values.push(read_f32(quant_slice, &mut quant_pos)?);
        }
        Ok(values)
    };
    let min = read_quant_values()?;
    let scale = read_quant_values()?;
    let inv_scale = read_quant_values()?;
    require_offset(
        quant_pos,
        quant_slice.len(),
        "scalar quantizer length is invalid",
    )?;
    let scalar = ScalarQuantizer::from_storage_parts(dim, min, scale, inv_scale)
        .map_err(RabitqError::InvalidLayout)?;

    let mut matrix = Vec::with_capacity(matrix_values);
    let mut matrix_pos = rotation_offset;
    for _ in 0..matrix_values {
        matrix.push(read_f32(buf, &mut matrix_pos)?);
    }
    require_offset(
        matrix_pos,
        rotation_end,
        "rotation matrix length is invalid",
    )?;
    let rotation = Rotation::from_matrix(dim, matrix)?;
    let rotation_quantizer = RabitqQuantizer::from_parts(dim, seed, rotation)?;

    let mut ids = Vec::with_capacity(count);
    let mut seen_ids = FxHashMap::with_capacity_and_hasher(count, Default::default());
    let mut ids_pos = ids_offset;
    for row in 0..count {
        let id = NodeId::new(read_u64(buf, &mut ids_pos)?);
        if seen_ids.insert(id, row).is_some() {
            return Err(RabitqError::DuplicateNodeId(id));
        }
        ids.push(id);
    }
    require_offset(ids_pos, codes_offset, "ID table length is invalid")?;

    Ok(ParsedRabitqBlob {
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
    /// # Errors
    ///
    /// Returns an error if a component cannot be serialized or a length or
    /// offset exceeds the wire format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RabitqError> {
        let (coarse, scalar, int8) = self.parts();
        let quantizer = coarse.quantizer();
        let dim = quantizer.dim();
        let words = quantizer.words();
        let count = coarse.len();

        if !scalar.has_valid_storage() || scalar.dimensions() != dim {
            return Err(RabitqError::InvariantViolation(
                "scalar quantizer storage does not match the index dimension",
            ));
        }
        let quant_len = checked_mul(
            checked_mul(dim, 3, "scalar quantizer values")?,
            size_of::<f32>(),
            "scalar quantizer bytes",
        )?;
        let dim_u32 = u32::try_from(dim).map_err(|_| RabitqError::SizeOverflow("dimension"))?;
        let count_u32 =
            u32::try_from(count).map_err(|_| RabitqError::SizeOverflow("entry count"))?;
        let words_u32 =
            u32::try_from(words).map_err(|_| RabitqError::SizeOverflow("code word count"))?;
        let quant_len_u32 = u32::try_from(quant_len)
            .map_err(|_| RabitqError::SizeOverflow("scalar quantizer blob"))?;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GRBQ");
        buf.push(BLOB_VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // padding
        buf.extend_from_slice(&dim_u32.to_le_bytes());
        buf.extend_from_slice(&count_u32.to_le_bytes());
        buf.extend_from_slice(&quantizer.seed().to_le_bytes());
        buf.extend_from_slice(&words_u32.to_le_bytes());
        buf.extend_from_slice(&quant_len_u32.to_le_bytes());
        // 4×u64 placeholder for section offsets, patched at the end.
        let offsets_pos = buf.len(); // = 32
        buf.extend_from_slice(&[0u8; 32]); // 4 × u64

        let (min, scale, inv_scale) = scalar.storage_parts();
        for &value in min.iter().chain(scale).chain(inv_scale) {
            buf.extend_from_slice(&value.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        let rotation_offset =
            u64::try_from(buf.len()).map_err(|_| RabitqError::SizeOverflow("rotation offset"))?;
        // Rotation matrix (dim*dim f32).
        for &m in quantizer.rotation().matrix() {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        let ids_offset =
            u64::try_from(buf.len()).map_err(|_| RabitqError::SizeOverflow("IDs offset"))?;
        // ids.
        for &id in coarse.ids() {
            buf.extend_from_slice(&id.as_u64().to_le_bytes());
        }

        let codes_offset =
            u64::try_from(buf.len()).map_err(|_| RabitqError::SizeOverflow("codes offset"))?;
        // codes: bit words, then dot_oo, then norm.
        for code in coarse.codes() {
            for &w in &code.bits {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&code.dot_oo().to_le_bytes());
            buf.extend_from_slice(&code.norm().to_le_bytes());
        }

        let int8_offset =
            u64::try_from(buf.len()).map_err(|_| RabitqError::SizeOverflow("int8 offset"))?;
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
        Ok(buf)
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`RabitqError`] on a bad magic, unsupported version,
    /// truncation, CRC mismatch, or a corrupt scalar-quantizer sub-blob.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, RabitqError> {
        let parsed = parse_rabitq_blob(buf)?;
        let mut pos = parsed.codes_offset;
        let mut codes = Vec::with_capacity(parsed.count);
        for _ in 0..parsed.count {
            let mut bits = Vec::with_capacity(parsed.words);
            for _ in 0..parsed.words {
                bits.push(read_u64(buf, &mut pos)?);
            }
            let dot_oo = read_f32(buf, &mut pos)?;
            let norm = read_f32(buf, &mut pos)?;
            codes.push(RabitqCode { bits, dot_oo, norm });
        }
        require_offset(pos, parsed.int8_offset, "codes section length is invalid")?;

        let mut coarse = RabitqIndex::with_quantizer(parsed.rotation_quantizer);
        coarse.load_entries(parsed.ids, codes)?;

        let mut int8 = Vec::with_capacity(parsed.count);
        pos = parsed.int8_offset;
        for _ in 0..parsed.count {
            let slice = checked_slice(buf, pos, parsed.dim, buf.len(), "int8 row")?;
            int8.push(slice.to_vec());
            pos = checked_add(pos, parsed.dim, "int8 row")?;
        }

        Self::from_parts(coarse, parsed.scalar, int8)
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
    pub fn open(blob: bytes::Bytes) -> Result<Self, RabitqError> {
        let parsed = parse_rabitq_blob(blob.as_ref())?;

        Ok(Self {
            blob,
            dim: parsed.dim,
            count: parsed.count,
            words: parsed.words,
            scalar: parsed.scalar,
            rotation_quantizer: parsed.rotation_quantizer,
            ids: parsed.ids,
            codes_offset: parsed.codes_offset,
            code_stride: parsed.code_stride,
            int8_offset: parsed.int8_offset,
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

    /// Searches for the `k` nearest neighbours of `query`.
    ///
    /// Identical results to [`TwoStageVectorIndex::search`] (see the
    /// parity proptest).
    ///
    /// # Errors
    ///
    /// Returns an error if `query` has the wrong dimension or the held blob's
    /// validated row layout is inconsistent.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
    ) -> Result<Vec<(NodeId, f32)>, RabitqError> {
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let candidate_n = k.saturating_mul(rerank_factor.max(1)).min(self.count);

        // Coarse pass: one reused code buffer, no per-row allocation.
        let q = self.rotation_quantizer.encode_query(query)?;
        let mut code = RabitqCode {
            bits: Vec::with_capacity(self.words),
            dot_oo: 0.0,
            norm: 0.0,
        };
        let code_section_len = checked_mul(self.count, self.code_stride, "codes section")?;
        let code_section = checked_slice(
            self.blob.as_ref(),
            self.codes_offset,
            code_section_len,
            self.blob.len(),
            "codes section",
        )?;
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.count);
        for (row, row_bytes) in code_section.chunks_exact(self.code_stride).enumerate() {
            code.bits.clear();
            let mut row_pos = 0;
            for _ in 0..self.words {
                code.bits.push(read_u64(row_bytes, &mut row_pos)?);
            }
            code.dot_oo = read_f32(row_bytes, &mut row_pos)?;
            code.norm = read_f32(row_bytes, &mut row_pos)?;
            let est = self
                .rotation_quantizer
                .estimate_distance_validated(&q, &code);
            scored.push((row, est));
        }
        let cmp = |a: &(usize, f32), b: &(usize, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        // Keep the candidate_n smallest, sorted — O(N + n log n) vs O(N log N).
        if candidate_n < scored.len() {
            scored.select_nth_unstable_by(candidate_n - 1, cmp);
            scored.truncate(candidate_n);
        }
        scored.sort_unstable_by(cmp);

        // Rerank by int8.
        let int8_section_len = checked_mul(self.count, self.dim, "int8 section")?;
        let int8_section = checked_slice(
            self.blob.as_ref(),
            self.int8_offset,
            int8_section_len,
            self.blob.len(),
            "int8 section",
        )?;
        let mut reranked = Vec::with_capacity(scored.len());
        for (row, _) in scored {
            let int8_start = checked_mul(row, self.dim, "int8 row offset")?;
            let int8 = checked_slice(
                int8_section,
                int8_start,
                self.dim,
                int8_section.len(),
                "int8 row",
            )?;
            let id = self
                .ids
                .get(row)
                .copied()
                .ok_or(RabitqError::InvariantViolation(
                    "ID row is outside the index",
                ))?;
            reranked.push((id, self.scalar.asymmetric_distance(query, int8)));
        }
        let rcmp = |a: &(NodeId, f32), b: &(NodeId, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        if k < reranked.len() {
            reranked.select_nth_unstable_by(k - 1, rcmp);
            reranked.truncate(k);
        }
        reranked.sort_unstable_by(rcmp);
        Ok(reranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refresh_crc(blob: &mut [u8]) {
        let body_end = blob.len() - 4;
        let crc = crc32fast::hash(&blob[..body_end]);
        blob[body_end..].copy_from_slice(&crc.to_le_bytes());
    }

    fn one_row_blob() -> Vec<u8> {
        TwoStageVectorIndex::build(&[(NodeId::new(1), vec![1.0; 8])], 8, 1)
            .expect("valid index")
            .to_bytes()
            .expect("serialize index")
    }

    #[test]
    fn invalid_public_inputs_return_errors() {
        assert!(Rotation::new_seeded(0, 1).is_err());

        let quantizer = RabitqQuantizer::new(8, 1).expect("valid quantizer");
        assert!(quantizer.encode(&[0.0; 7]).is_err());
        assert!(quantizer.encode_query(&[0.0; 9]).is_err());
        let mut non_finite = [0.0; 8];
        non_finite[3] = f32::NAN;
        assert!(
            Rotation::new_seeded(8, 1)
                .expect("valid rotation")
                .apply(&non_finite)
                .is_err()
        );
        assert!(quantizer.encode(&non_finite).is_err());
        assert!(quantizer.encode_query(&non_finite).is_err());
        assert!(quantizer.encode(&[f32::MAX; 8]).is_err());

        let other_quantizer = RabitqQuantizer::new(65, 2).expect("valid quantizer");
        let query = quantizer
            .encode_query(&[0.0; 8])
            .expect("matching query dimension");
        let code = other_quantizer
            .encode(&[0.0; 65])
            .expect("matching vector dimension");
        assert!(quantizer.estimate_distance(&query, &code).is_err());

        assert!(TwoStageVectorIndex::build(&[], 8, 1).is_err());
        assert!(TwoStageVectorIndex::build(&[(NodeId::new(1), vec![0.0; 7])], 8, 1).is_err());
        let duplicate = vec![
            (NodeId::new(1), vec![0.0; 8]),
            (NodeId::new(1), vec![1.0; 8]),
        ];
        assert!(TwoStageVectorIndex::build(&duplicate, 8, 1).is_err());
    }

    #[test]
    fn owned_and_view_reject_crc_valid_noncanonical_layouts() {
        let pristine = one_row_blob();
        let mut cases = Vec::new();

        let mut reserved = pristine.clone();
        reserved[5] = 1;
        refresh_crc(&mut reserved);
        cases.push(reserved);

        let mut wrong_words = pristine.clone();
        wrong_words[24..28].copy_from_slice(&0u32.to_le_bytes());
        refresh_crc(&mut wrong_words);
        cases.push(wrong_words);

        let mut wrong_quant_len = pristine.clone();
        wrong_quant_len[28..32].copy_from_slice(&(8 * 3 * 4 + 4u32).to_le_bytes());
        refresh_crc(&mut wrong_quant_len);
        cases.push(wrong_quant_len);

        let mut non_finite_quantizer = pristine.clone();
        non_finite_quantizer[64..68].copy_from_slice(&f32::NAN.to_le_bytes());
        refresh_crc(&mut non_finite_quantizer);
        cases.push(non_finite_quantizer);

        let mut wrong_offset = pristine.clone();
        let rotation_offset = u64::from_le_bytes(
            wrong_offset[32..40]
                .try_into()
                .expect("rotation offset field"),
        );
        wrong_offset[32..40].copy_from_slice(&(rotation_offset + 8).to_le_bytes());
        refresh_crc(&mut wrong_offset);
        cases.push(wrong_offset);

        let mut nonzero_padding_bits = pristine.clone();
        let codes_offset = usize::try_from(u64::from_le_bytes(
            nonzero_padding_bits[48..56]
                .try_into()
                .expect("codes offset field"),
        ))
        .expect("test blob offset fits usize");
        nonzero_padding_bits[codes_offset + 7] |= 0x80;
        refresh_crc(&mut nonzero_padding_bits);
        cases.push(nonzero_padding_bits);

        let mut non_finite_factor = pristine.clone();
        non_finite_factor[codes_offset + 8..codes_offset + 12]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        refresh_crc(&mut non_finite_factor);
        cases.push(non_finite_factor);

        let mut trailing = pristine;
        trailing.truncate(trailing.len() - 4);
        trailing.extend_from_slice(&[0; 8]);
        refresh_crc(&mut trailing);
        cases.push(trailing);

        for blob in cases {
            assert!(matches!(
                TwoStageVectorIndex::from_bytes(&blob),
                Err(RabitqError::InvalidLayout(_))
            ));
            assert!(matches!(
                RabitqView::open(bytes::Bytes::from(blob)),
                Err(RabitqError::InvalidLayout(_))
            ));
        }
    }

    #[test]
    fn owned_and_view_search_reject_wrong_query_dimensions() {
        let blob = one_row_blob();
        let owned = TwoStageVectorIndex::from_bytes(&blob).expect("open owned index");
        let view = RabitqView::open(bytes::Bytes::from(blob)).expect("open view");

        assert!(owned.search(&[0.0; 7], 1, 8).is_err());
        assert!(view.search(&[0.0; 9], 1, 8).is_err());
    }

    #[test]
    fn scalar_quantizer_wire_storage_is_fixed_width() {
        let blob = one_row_blob();
        let quant_len = usize::try_from(u32::from_le_bytes(
            blob[28..32]
                .try_into()
                .expect("scalar quantizer length field"),
        ))
        .expect("u32 fits usize on supported targets");
        assert_eq!(quant_len, 8 * 3 * size_of::<f32>());
    }

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
        let rot = Rotation::new_seeded(64, 123).expect("valid rotation");
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let norm_in: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rotated = rot.apply(&v).expect("matching vector dimension");
        let norm_out: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_in - norm_out).abs() < 1e-3,
            "rotation changed norm: {norm_in} -> {norm_out}"
        );
    }

    #[test]
    fn rotation_is_deterministic_for_a_seed() {
        let a = Rotation::new_seeded(32, 99).expect("valid rotation");
        let b = Rotation::new_seeded(32, 99).expect("valid rotation");
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(
            a.apply(&v).expect("matching vector dimension"),
            b.apply(&v).expect("matching vector dimension")
        );
    }

    #[test]
    fn encode_256_dim_yields_32_byte_code() {
        let q = RabitqQuantizer::new(256, 1).expect("valid quantizer");
        let v: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let code = q.encode(&v).expect("matching vector dimension");
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
        let q = RabitqQuantizer::new(8, 1).expect("valid quantizer");
        let code = q.encode(&[0.0; 8]).expect("matching vector dimension");
        assert_eq!(code.norm(), 0.0);
    }

    #[test]
    fn encode_non_multiple_of_64_dim() {
        // 100 dims -> ceil(100/64) = 2 words = 16 bytes.
        let q = RabitqQuantizer::new(100, 5).expect("valid quantizer");
        let v: Vec<f32> = (0..100).map(|i| i as f32 - 50.0).collect();
        assert_eq!(
            q.encode(&v)
                .expect("matching vector dimension")
                .code_bytes(),
            16
        );
    }

    #[test]
    fn estimate_distance_ranks_self_closest() {
        let q = RabitqQuantizer::new(128, 3).expect("valid quantizer");
        let target: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let far: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).cos() * 3.0).collect();

        let query = q.encode_query(&target).expect("matching query dimension");
        let code_self = q.encode(&target).expect("matching vector dimension");
        let code_far = q.encode(&far).expect("matching vector dimension");

        let d_self = q
            .estimate_distance(&query, &code_self)
            .expect("matching code widths");
        let d_far = q
            .estimate_distance(&query, &code_far)
            .expect("matching code widths");
        assert!(
            d_self < d_far,
            "self {d_self} should be closer than far {d_far}"
        );
        assert!(d_self >= 0.0);
    }

    #[test]
    fn estimate_distance_orders_a_small_set() {
        let q = RabitqQuantizer::new(64, 11).expect("valid quantizer");
        let base: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let query = q.encode_query(&base).expect("matching query dimension");

        // Increasingly perturbed copies must estimate increasingly far.
        let mut last = -1.0f32;
        for scale in [0.0f32, 0.5, 1.0, 2.0] {
            let v: Vec<f32> = base.iter().map(|&x| x + scale).collect();
            let code = q.encode(&v).expect("matching vector dimension");
            let d = q
                .estimate_distance(&query, &code)
                .expect("matching code widths");
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

        let mut index = RabitqIndex::new(32, 17).expect("valid index");
        // Cluster A near 0.0, cluster B near 5.0.
        for i in 0..10 {
            let a: Vec<f32> = (0..32)
                .map(|d| (d as f32 * 0.1).sin() + i as f32 * 0.01)
                .collect();
            index
                .insert(NodeId::new(i + 1), &a)
                .expect("matching vector dimension");
        }
        for i in 0..10 {
            let b: Vec<f32> = (0..32).map(|d| (d as f32 * 0.1).sin() + 5.0).collect();
            index
                .insert(NodeId::new(100 + i), &b)
                .expect("matching vector dimension");
        }
        assert_eq!(index.len(), 20);

        let query: Vec<f32> = (0..32).map(|d| (d as f32 * 0.1).sin()).collect();
        let hits = index
            .coarse_search(&query, 5)
            .expect("matching query dimension");
        assert_eq!(hits.len(), 5);
        // Sorted ascending by estimated distance.
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        // The nearest hits should come from cluster A (ids 1..=10).
        assert!(hits[0].0.as_u64() <= 10, "nearest hit not from cluster A");
    }

    #[test]
    fn coarse_search_selection_matches_full_sort_prefix() {
        use grafeo_common::types::NodeId;

        let mut index = RabitqIndex::new(16, 3).expect("valid index");
        for i in 0..40u64 {
            let v: Vec<f32> = (0..16)
                .map(|d| (d as f32 * 0.2).cos() + i as f32 * 0.05)
                .collect();
            index
                .insert(NodeId::new(i + 1), &v)
                .expect("matching vector dimension");
        }
        let query: Vec<f32> = (0..16).map(|d| (d as f32 * 0.2).cos()).collect();

        let top = index
            .coarse_search(&query, 8)
            .expect("matching query dimension");
        let full = index
            .coarse_search(&query, index.len())
            .expect("matching query dimension");
        assert_eq!(top.len(), 8);
        // select_nth must pick the same 8 smallest, in the same sorted order
        // as a full sort + truncate.
        assert_eq!(top, full[..8].to_vec());
        for w in top.windows(2) {
            assert!(w[0].1 <= w[1].1, "coarse_search not ascending");
        }
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

        let index = TwoStageVectorIndex::build(&vectors, dim, 1).expect("valid index");
        assert_eq!(index.len(), 120);

        // Query = first point of cluster 0; its 10 true neighbours are in cluster 0.
        let query = vectors[0].1.clone();
        let hits = index
            .search(&query, 10, 16)
            .expect("matching query dimension");
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
        let index = TwoStageVectorIndex::build(&vectors, 8, 1).expect("valid index");
        assert!(
            index
                .search(&[1.0; 8], 0, 4)
                .expect("matching query dimension")
                .is_empty()
        );
        assert_eq!(
            index
                .search(&[1.0; 8], 5, 4)
                .expect("matching query dimension")
                .len(),
            1
        );
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

        let index = TwoStageVectorIndex::build(&vectors, dim, 9).expect("valid index");
        let blob = index.to_bytes().expect("serialize index");

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
        assert_eq!(
            index
                .search(&query, 10, 8)
                .expect("matching query dimension"),
            reopened
                .search(&query, 10, 8)
                .expect("matching query dimension"),
        );
    }

    #[test]
    fn blob_rejects_bad_magic_and_crc() {
        use grafeo_common::types::NodeId;
        let vectors = vec![(NodeId::new(1), vec![1.0f32; 8])];
        let mut blob = TwoStageVectorIndex::build(&vectors, 8, 1)
            .expect("valid index")
            .to_bytes()
            .expect("serialize index");

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
        let index = TwoStageVectorIndex::build(&vectors, 8, 1).expect("valid index");
        let blob = bytes::Bytes::from(index.to_bytes().expect("serialize index"));
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
        let owned = TwoStageVectorIndex::build(&vectors, dim, 7).expect("valid index");
        let blob = bytes::Bytes::from(owned.to_bytes().expect("serialize index"));
        let view = RabitqView::open(blob).expect("open");

        assert_eq!(view.len(), owned.len());

        let query = vectors[3].1.clone();
        let owned_hits = owned
            .search(&query, 10, 8)
            .expect("matching query dimension");
        let view_hits = view
            .search(&query, 10, 8)
            .expect("matching query dimension");
        assert_eq!(view_hits, owned_hits);
    }

    #[test]
    fn view_rejects_bad_magic() {
        use grafeo_common::types::NodeId;
        let owned = TwoStageVectorIndex::build(&[(NodeId::new(1), vec![1.0f32; 8])], 8, 1)
            .expect("valid index");
        let mut bad = owned.to_bytes().expect("serialize index");
        bad[0] = b'X';
        assert!(matches!(
            RabitqView::open(bytes::Bytes::from(bad)),
            Err(RabitqError::BadMagic)
        ));
    }
}
