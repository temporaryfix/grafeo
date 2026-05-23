# Sub-plan 2a — RaBitQ + int8 Two-Stage Vector Codec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RaBitQ 1-bit-per-dimension vector quantization codec to the Grafeo fork, with a two-stage search (RaBitQ coarse pass → int8 rerank), exposed through the WASM bindings, so Nota's build pipeline can compress vector blobs.

**Architecture:** A new in-tree, dependency-free module `index/vector/rabitq.rs`. Each vector is unit-normalised, rotated by a fixed random orthogonal matrix, and sign-quantized to one bit per dimension; a per-vector correction factor unbiases the popcount-based distance estimator. The existing `ScalarQuantizer` (int8, 4× compression) is reused unchanged for the rerank stage. `TwoStageVectorIndex` runs the coarse RaBitQ scan, then reranks the top `k·rerank_factor` candidates by int8 distance. A blob `to_bytes`/`from_bytes` round-trip implements the Plan 2 zero-copy layout contract. The existing `ProductQuantizer` stays as the documented fallback; a criterion benchmark compares the two.

**Tech Stack:** Rust 2024, `grafeo-core` (`index/vector/`), `crates/bindings/wasm`, `criterion` (benchmark), `proptest` (recall-floor regression test). No new runtime dependencies — the rotation PRNG is an in-tree SplitMix64.

---

## File Structure

- **Create** `crates/grafeo-core/src/index/vector/rabitq.rs` — the codec: `SplitMix64`, `Rotation`, `RabitqQuantizer`, `RabitqCode`, `RabitqQuery`, `RabitqIndex`, `TwoStageVectorIndex`, `RabitqError`, blob serialization, unit tests.
- **Modify** `crates/grafeo-core/src/index/vector/mod.rs` — register `pub mod rabitq;` and re-export the public types.
- **Modify** `crates/grafeo-core/Cargo.toml` — add `proptest` dev-dependency and the `rabitq_vs_pq` bench entry.
- **Create** `crates/grafeo-core/tests/rabitq_recall.rs` — recall-floor proptest + fixed-seed regression cases.
- **Create** `crates/grafeo-core/benches/rabitq_vs_pq.rs` — criterion benchmark, RaBitQ two-stage vs PQ.
- **Create** `crates/bindings/wasm/src/codecs.rs` — `#[wasm_bindgen]` `RabitqCodec` wrapper.
- **Modify** `crates/bindings/wasm/src/lib.rs` — register `mod codecs;`.
- **Modify** `crates/bindings/wasm/tests/web.rs` — wasm round-trip test.
- **Modify** `CHANGELOG.md` — note the new codec.

Reference facts from the existing code (read before starting):
- `crates/grafeo-core/src/index/vector/quantization.rs:430` — `BinaryQuantizer` (sign-bit packing into `u64`, the pattern `rabitq.rs` follows).
- `quantization.rs:940` — `pub fn hamming_distance_simd(a: &[u64], b: &[u64]) -> u32` — reuse for the popcount distance; do not reimplement.
- `quantization.rs:168` — `ScalarQuantizer`; `train(&[&[f32]])`, `quantize(&[f32]) -> Vec<u8>`, `asymmetric_distance(&[f32], &[u8]) -> f32`. Derives `Serialize`/`Deserialize`.
- `quantization.rs:541` — `ProductQuantizer`; the documented fallback, used only in the benchmark.
- `index/vector/mod.rs:84` — `pub mod quantization;` is **not** feature-gated. `rabitq` is added the same way (a codec, not an index).

---

## Task 1: Module scaffold — `SplitMix64` PRNG and `RabitqError`

**Files:**
- Create: `crates/grafeo-core/src/index/vector/rabitq.rs`
- Modify: `crates/grafeo-core/src/index/vector/mod.rs:84` (add module), `:109` (re-export)

- [ ] **Step 1: Write the failing test**

Create `crates/grafeo-core/src/index/vector/rabitq.rs` with this content:

```rust
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
            (self.next_u64() >> 40) as f32 / f32::from(1u32 << 12) / f32::from(1u32 << 12)
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
        assert!(mean.abs() < 0.05, "gaussian mean drifted: {mean}");
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/grafeo-core/src/index/vector/mod.rs`, after the line `pub mod quantization;` (line 84), add:

```rust
pub mod rabitq;
```

After the line `pub use quantization::{BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer};` (line 109), add:

```rust
pub use rabitq::{
    RabitqCode, RabitqError, RabitqIndex, RabitqQuantizer, RabitqQuery, TwoStageVectorIndex,
};
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq -- --nocapture`
Expected: `splitmix64_is_deterministic` and `splitmix64_gaussian_is_roughly_centred` both PASS, 2 tests.

(The re-export of not-yet-defined types would fail to compile — they are added in later tasks. To keep this task self-contained, add the `pub use rabitq::{...}` line only after Task 6. For Step 3 here, register just `pub mod rabitq;` and run the test; add the re-export line in Task 6 Step 5.)

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs crates/grafeo-core/src/index/vector/mod.rs
git commit -m "feat(rabitq): module scaffold with SplitMix64 PRNG and error type"
```

---

## Task 2: Random orthogonal `Rotation`

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `rabitq.rs`:

```rust
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `cannot find type/value 'Rotation' in this scope`.

- [ ] **Step 3: Write the implementation**

Add to `rabitq.rs`, after the `RabitqError` definition:

```rust
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
            let inv = if norm > f32::EPSILON { 1.0 / norm } else { 1.0 };
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
    #[must_use]
    pub(crate) fn from_matrix(dim: usize, matrix: Vec<f32>) -> Self {
        debug_assert_eq!(matrix.len(), dim * dim, "matrix must be dim*dim");
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs
git commit -m "feat(rabitq): seeded random orthogonal rotation matrix"
```

---

## Task 3: `RabitqQuantizer` and `RabitqCode` — encode

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn encode_256_dim_yields_32_byte_code() {
    let q = RabitqQuantizer::new(256, 1);
    let v: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let code = q.encode(&v);
    assert_eq!(code.code_bytes(), 32, "256 sign bits must pack into 32 bytes");
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `cannot find type 'RabitqQuantizer'`.

- [ ] **Step 3: Write the implementation**

Add to `rabitq.rs`, after `Rotation`:

```rust
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
        Self { dim, seed, rotation }
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs
git commit -m "feat(rabitq): RabitqQuantizer encode with sign-bit code and correction factor"
```

---

## Task 4: `encode_query` and popcount distance estimate

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
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
    assert!(d_self < d_far, "self {d_self} should be closer than far {d_far}");
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
        assert!(d >= last - 0.5, "distance not monotone at scale {scale}: {d} < {last}");
        last = d;
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `no method named 'encode_query'`.

- [ ] **Step 3: Write the implementation**

Add to `rabitq.rs`. First, at the top of the file (after the module doc comment), add the import:

```rust
use super::quantization::{ScalarQuantizer, hamming_distance_simd};
```

Then add a method block for `RabitqQuantizer` (extend the existing `impl`, or add a second `impl RabitqQuantizer` block) plus the `RabitqQuery` type:

```rust
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
            let cos_est =
                (ip_quant / code.dot_oo.max(f32::MIN_POSITIVE)).clamp(-1.0, 1.0);
            // d² = |a|² + |b|² − 2|a||b|cosθ.
            let d2 = code.norm.mul_add(code.norm, query.norm * query.norm)
                - 2.0 * code.norm * query.norm * cos_est;
            d2.max(0.0).sqrt()
        }
    }
}
```

Note: the `ScalarQuantizer` import is unused until Task 6 — add `#[allow(unused_imports)]` above the `use` line for now, or import only `hamming_distance_simd` here and add `ScalarQuantizer` in Task 6. Prefer the latter: import just `use super::quantization::hamming_distance_simd;` now.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 9 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs
git commit -m "feat(rabitq): query encoding and popcount distance estimate"
```

---

## Task 5: `RabitqIndex` — coarse search

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn rabitq_index_coarse_search_returns_sorted_candidates() {
    use grafeo_common::types::NodeId;

    let mut index = RabitqIndex::new(32, 17);
    // Cluster A near 0.0, cluster B near 5.0.
    for i in 0..10 {
        let a: Vec<f32> = (0..32).map(|d| (d as f32 * 0.1).sin() + i as f32 * 0.01).collect();
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `cannot find type 'RabitqIndex'`.

- [ ] **Step 3: Write the implementation**

Add to `rabitq.rs`. Add the import for `NodeId` near the top:

```rust
use grafeo_common::types::NodeId;
```

Then add:

```rust
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
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs
git commit -m "feat(rabitq): RabitqIndex with coarse nearest-neighbour scan"
```

---

## Task 6: `TwoStageVectorIndex` — build and int8-rerank search

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`, `crates/grafeo-core/src/index/vector/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
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
            let v: Vec<f32> = centre.iter().map(|&c| c + rng.next_gaussian() * 0.3).collect();
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
    assert!(from_cluster0 >= 9, "expected >=9 cluster-0 hits, got {from_cluster0}");
}

#[test]
fn two_stage_search_empty_and_k_zero() {
    use grafeo_common::types::NodeId;
    let vectors = vec![(NodeId::new(1), vec![1.0f32; 8])];
    let index = TwoStageVectorIndex::build(&vectors, 8, 1);
    assert!(index.search(&[1.0; 8], 0, 4).is_empty());
    assert_eq!(index.search(&[1.0; 8], 5, 4).len(), 1);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `cannot find type 'TwoStageVectorIndex'`.

- [ ] **Step 3: Write the implementation**

In `rabitq.rs`, change the quantization import to also bring in `ScalarQuantizer`:

```rust
use super::quantization::{ScalarQuantizer, hamming_distance_simd};
```

Add the `FxHashMap` import near the top:

```rust
use grafeo_common::utils::hash::FxHashMap;
```

Then add:

```rust
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
    /// # Panics
    /// Panics if `vectors` is empty or any vector's length is not `dim`.
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
    #[must_use]
    pub(crate) fn from_parts(
        coarse: RabitqIndex,
        scalar: ScalarQuantizer,
        int8: Vec<Vec<u8>>,
    ) -> Self {
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 12 tests.

- [ ] **Step 5: Add the public re-export**

In `crates/grafeo-core/src/index/vector/mod.rs`, after line 109, add (if not already added in Task 1):

```rust
pub use rabitq::{
    RabitqCode, RabitqError, RabitqIndex, RabitqQuantizer, RabitqQuery, TwoStageVectorIndex,
};
```

Run: `cargo build -p grafeo-core` — Expected: clean build.

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs crates/grafeo-core/src/index/vector/mod.rs
git commit -m "feat(rabitq): TwoStageVectorIndex with int8 rerank search"
```

---

## Task 7: Blob `to_bytes` / `from_bytes` — zero-copy layout contract

This implements the Plan 2 cross-cutting zero-copy contract (fixed header, natural alignment, blob-relative offsets, self-describing). The `from_bytes` here parses into owned `Vec`s; sub-plan 2d adds a borrowing view over the *same* layout.

**Files:**
- Modify: `crates/grafeo-core/src/index/vector/rabitq.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
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

    let reopened = TwoStageVectorIndex::from_bytes(&blob).expect("from_bytes");
    assert_eq!(reopened.len(), index.len());

    // Identical query results before and after a round trip.
    let query = vectors[3].1.clone();
    assert_eq!(
        index.search(&query, 10, 8),
        reopened.search(&query, 10, 8),
    );
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: FAIL — `no method named 'to_bytes'`.

- [ ] **Step 3: Write the implementation**

Add to `rabitq.rs`. The layout, all little-endian:

```text
offset 0   "GRBQ"                         magic (4 bytes)
       4   version = 1                    u8
       5   flags = 0                      u8
       6   padding                        u16
       8   dim                            u32
      12   count                          u32
      16   seed                           u64
      24   words = ceil(dim/64)            u32
      28   quantizer_len                   u32   bincode(ScalarQuantizer) length
      32   quantizer bytes ... + pad to 8
           rotation: dim*dim × f32         (then pad to 8)
           ids: count × u64
           codes: count × (words×u64, dot_oo f32, norm f32)
           int8: count × dim × u8          (then pad to 4)
           crc32 of everything above       u32
```

Implementation:

```rust
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
    /// naturally-aligned arrays, blob-relative offsets, and a trailing CRC32.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let (coarse, scalar, int8) = self.parts();
        let quantizer = coarse.quantizer();
        let dim = quantizer.dim();
        let words = quantizer.words();
        let count = coarse.len();

        let quant_blob =
            bincode::serde::encode_to_vec(scalar, bincode::config::standard())
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
        buf.extend_from_slice(&quant_blob);
        pad_to(&mut buf, 8);

        // Rotation matrix (dim*dim f32).
        for &m in quantizer.rotation().matrix() {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        pad_to(&mut buf, 8);

        // ids.
        for &id in coarse.ids() {
            buf.extend_from_slice(&id.as_u64().to_le_bytes());
        }

        // codes: bit words, then dot_oo, then norm.
        for code in coarse.codes() {
            for &w in &code.bits {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&code.dot_oo().to_le_bytes());
            buf.extend_from_slice(&code.norm().to_le_bytes());
        }

        // int8 codes.
        for row in int8 {
            buf.extend_from_slice(row);
        }
        pad_to(&mut buf, 4);

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`RabitqError`] on a bad magic, unsupported version,
    /// truncation, CRC mismatch, or a corrupt scalar-quantizer sub-blob.
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

        let quant_slice = buf.get(pos..pos + quant_len).ok_or(RabitqError::Truncated {
            need: pos + quant_len,
            have: buf.len(),
        })?;
        let (scalar, _): (ScalarQuantizer, usize) =
            bincode::serde::decode_from_slice(quant_slice, bincode::config::standard())
                .map_err(|e| RabitqError::Quantizer(e.to_string()))?;
        pos += quant_len;
        pos = pos.next_multiple_of(8);

        // Rotation matrix.
        let mut matrix = Vec::with_capacity(dim * dim);
        for _ in 0..dim * dim {
            matrix.push(read_f32(buf, &mut pos)?);
        }
        pos = pos.next_multiple_of(8);

        let quantizer =
            RabitqQuantizer::from_parts(dim, seed, Rotation::from_matrix(dim, matrix));
        let mut coarse = RabitqIndex::with_quantizer(quantizer);

        // ids.
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(NodeId::new(read_u64(buf, &mut pos)?));
        }

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
}
```

Add this helper to the `impl RabitqIndex` block (it lets `from_bytes` populate an index directly):

```rust
    /// Replaces the index contents with pre-decoded ids and codes.
    pub(crate) fn load_entries(&mut self, ids: Vec<NodeId>, codes: Vec<RabitqCode>) {
        debug_assert_eq!(ids.len(), codes.len());
        self.ids = ids;
        self.codes = codes;
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p grafeo-core --lib index::vector::rabitq`
Expected: PASS — 14 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-core/src/index/vector/rabitq.rs
git commit -m "feat(rabitq): zero-copy-layout blob serialization with crc"
```

---

## Task 8: Recall-floor proptest

**Files:**
- Modify: `crates/grafeo-core/Cargo.toml`
- Create: `crates/grafeo-core/tests/rabitq_recall.rs`

- [ ] **Step 1: Add the `proptest` dev-dependency**

In `crates/grafeo-core/Cargo.toml`, under `[dev-dependencies]`, add:

```toml
proptest.workspace = true
```

(`proptest = "1"` is already declared in the workspace `Cargo.toml`.)

- [ ] **Step 2: Write the failing test**

Create `crates/grafeo-core/tests/rabitq_recall.rs`:

```rust
//! Recall-floor regression test for the RaBitQ two-stage vector codec.
//!
//! Mirrors the fork's property-test discipline (see
//! `grafeo-engine/tests/compact_roundtrip_proptest.rs`): a `proptest!`
//! block generating clustered datasets, plus fixed-seed regression cases.
//! The oracle is exact brute-force Euclidean k-NN; the codec must keep
//! recall@10 at or above the floor.
//!
//! ```bash
//! cargo test -p grafeo-core --test rabitq_recall
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test rabitq_recall
//! ```

use grafeo_common::types::NodeId;
use grafeo_core::index::vector::TwoStageVectorIndex;
use proptest::prelude::*;

/// SplitMix64 — duplicated here because the in-crate one is private.
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

/// Builds a clustered dataset deterministically from `seed`.
fn clustered_dataset(
    seed: u64,
    dim: usize,
    clusters: usize,
    per_cluster: usize,
) -> Vec<(NodeId, Vec<f32>)> {
    let mut rng = Rng(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gaussian() * 6.0).collect())
        .collect();
    let mut out = Vec::new();
    let mut id = 1u64;
    for centre in &centres {
        for _ in 0..per_cluster {
            let v = centre.iter().map(|&c| c + rng.gaussian() * 0.4).collect();
            out.push((NodeId::new(id), v));
            id += 1;
        }
    }
    out
}

/// Exact brute-force Euclidean k-NN — the recall oracle.
fn brute_force(
    vectors: &[(NodeId, Vec<f32>)],
    query: &[f32],
    k: usize,
) -> Vec<NodeId> {
    let mut scored: Vec<(NodeId, f32)> = vectors
        .iter()
        .map(|(id, v)| {
            let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (*id, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// recall@k of `got` against the exact `truth`.
fn recall(truth: &[NodeId], got: &[(NodeId, f32)]) -> f32 {
    let hits = got.iter().filter(|(id, _)| truth.contains(id)).count();
    hits as f32 / truth.len() as f32
}

/// The codec must clear this recall@10 on clustered data.
const RECALL_FLOOR: f32 = 0.90;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// recall@10 stays at or above the floor across generated datasets.
    #[test]
    fn rabitq_two_stage_clears_recall_floor(
        seed in any::<u64>(),
        clusters in 4usize..=8,
        per_cluster in 15usize..=30,
        query_pick in 0usize..200,
    ) {
        let dim = 64;
        let vectors = clustered_dataset(seed, dim, clusters, per_cluster);
        let index = TwoStageVectorIndex::build(&vectors, dim, seed ^ 0xA5A5);

        let query = vectors[query_pick % vectors.len()].1.clone();
        let truth = brute_force(&vectors, &query, 10);
        let got = index.search(&query, 10, 16);

        let r = recall(&truth, &got);
        prop_assert!(
            r >= RECALL_FLOOR,
            "recall {r} below floor {RECALL_FLOOR} (seed={seed}, clusters={clusters})"
        );
    }
}

// ── Fixed regression seeds ───────────────────────────────────────

#[test]
fn recall_floor_fixed_seed_small() {
    let dim = 64;
    let vectors = clustered_dataset(1, dim, 5, 20);
    let index = TwoStageVectorIndex::build(&vectors, dim, 7);
    let query = vectors[0].1.clone();
    let truth = brute_force(&vectors, &query, 10);
    let got = index.search(&query, 10, 16);
    assert!(recall(&truth, &got) >= RECALL_FLOOR);
}

#[test]
fn recall_floor_survives_blob_round_trip() {
    let dim = 64;
    let vectors = clustered_dataset(42, dim, 6, 25);
    let index = TwoStageVectorIndex::build(&vectors, dim, 13);
    let reopened = TwoStageVectorIndex::from_bytes(&index.to_bytes()).expect("from_bytes");

    let query = vectors[10].1.clone();
    let truth = brute_force(&vectors, &query, 10);
    assert!(recall(&truth, &reopened.search(&query, 10, 16)) >= RECALL_FLOOR);
}
```

- [ ] **Step 3: Run to verify it passes**

Run: `cargo test -p grafeo-core --test rabitq_recall`
Expected: PASS — `rabitq_two_stage_clears_recall_floor` (128 proptest cases), `recall_floor_fixed_seed_small`, `recall_floor_survives_blob_round_trip`.

If the proptest fails, do **not** lower `RECALL_FLOOR` blindly. Run `superpowers:systematic-debugging`: print the failing seed, rebuild that dataset, and inspect whether the coarse pass dropped a true neighbour (raise `rerank_factor`) or the int8 rerank mis-ordered (a `ScalarQuantizer` training issue). Only adjust the floor if the shrunk case is a genuine degenerate dataset (e.g. fewer than 10 vectors).

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/Cargo.toml crates/grafeo-core/tests/rabitq_recall.rs
git commit -m "test(rabitq): recall-floor proptest and fixed regression seeds"
```

---

## Task 9: WASM bindings

**Files:**
- Create: `crates/bindings/wasm/src/codecs.rs`
- Modify: `crates/bindings/wasm/src/lib.rs`
- Modify: `crates/bindings/wasm/tests/web.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/bindings/wasm/tests/web.rs`:

```rust
#[wasm_bindgen_test]
fn rabitq_codec_encode_open_search_round_trip() {
    use grafeo_wasm::codecs::RabitqCodec;

    let dim = 32usize;
    let count = 40u32;
    // Two clusters: ids 1..=20 near 0, ids 21..=40 near 10.
    let mut ids = Vec::new();
    let mut flat = Vec::new();
    for i in 0..count {
        ids.push(i + 1);
        let base = if i < 20 { 0.0f32 } else { 10.0f32 };
        for d in 0..dim {
            flat.push(base + (d as f32 * 0.05).sin());
        }
    }

    let blob = RabitqCodec::encode(&ids, &flat, dim, 1.0).expect("encode");
    assert_eq!(&blob[0..4], b"GRBQ");

    let codec = RabitqCodec::open(&blob).expect("open");
    // Query from cluster 0 -> hits should be ids 1..=20.
    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.05).sin()).collect();
    let hits = codec.search(&query, 5, 8);
    assert_eq!(hits.len(), 5);
    for id in hits {
        assert!(id <= 20, "expected cluster-0 hit, got id {id}");
    }
}
```

(Confirm the wasm crate's package name with `grep '^name' crates/bindings/wasm/Cargo.toml`; if it is not `grafeo-wasm`, the test import path `grafeo_wasm::codecs::RabitqCodec` uses the crate's lib name with hyphens turned to underscores. Adjust the `use` accordingly.)

- [ ] **Step 2: Run to verify it fails**

Run: `wasm-pack test --node crates/bindings/wasm -- --test web rabitq`
Expected: FAIL — `unresolved import` / `module 'codecs' not found`.

(If `wasm-pack` is unavailable, the binding still type-checks under `cargo build -p grafeo-wasm --target wasm32-unknown-unknown`; run that as the fallback verification.)

- [ ] **Step 3: Write the implementation**

Create `crates/bindings/wasm/src/codecs.rs`:

```rust
//! WASM bindings for the Plan 2 compression codecs.
//!
//! The TypeScript build pipeline calls these to encode a data component
//! to a compressed blob and to open and query that blob.

use grafeo_common::types::NodeId;
use grafeo_core::index::vector::TwoStageVectorIndex;
use wasm_bindgen::prelude::*;

/// A JS-facing handle to a RaBitQ two-stage vector index.
#[wasm_bindgen]
pub struct RabitqCodec {
    inner: TwoStageVectorIndex,
}

#[wasm_bindgen]
impl RabitqCodec {
    /// Encodes a batch of vectors into a compressed blob.
    ///
    /// `ids` is one node id per vector; `flat` holds `ids.length * dim`
    /// `f32` values, row-major (vector `r` occupies `flat[r*dim .. r*dim+dim]`).
    /// `seed` fixes the RaBitQ rotation. Returns the blob bytes.
    ///
    /// # Errors
    /// Returns a `JsError` if `dim` is zero, `ids` is empty, or `flat`'s
    /// length is not `ids.length * dim`.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode(ids: &[u32], flat: &[f32], dim: usize, seed: f64) -> Result<Vec<u8>, JsError> {
        if dim == 0 || ids.is_empty() {
            return Err(JsError::new("ids must be non-empty and dim must be > 0"));
        }
        if flat.len() != ids.len() * dim {
            return Err(JsError::new("flat.length must equal ids.length * dim"));
        }
        let vectors: Vec<(NodeId, Vec<f32>)> = ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let start = row * dim;
                (NodeId::new(u64::from(id)), flat[start..start + dim].to_vec())
            })
            .collect();
        // reason: seed arrives as a JS number; truncating the fraction is fine
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let seed_u64 = seed as u64;
        Ok(TwoStageVectorIndex::build(&vectors, dim, seed_u64).to_bytes())
    }

    /// Opens a blob produced by [`RabitqCodec::encode`] for querying.
    ///
    /// # Errors
    /// Returns a `JsError` if the blob is malformed (bad magic, version,
    /// truncation, or CRC mismatch).
    #[wasm_bindgen(js_name = "open")]
    pub fn open(blob: &[u8]) -> Result<RabitqCodec, JsError> {
        let inner = TwoStageVectorIndex::from_bytes(blob)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Searches for the `k` nearest neighbours of `query`. Returns node ids
    /// nearest-first. `rerank_factor` controls the recall/latency trade-off
    /// (8–16 is typical).
    #[wasm_bindgen(js_name = "search")]
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize, rerank_factor: usize) -> Vec<u32> {
        // reason: node ids in a snapshot fit u32 for the JS surface
        #[allow(clippy::cast_possible_truncation)]
        self.inner
            .search(query, k, rerank_factor)
            .into_iter()
            .map(|(id, _)| id.as_u64() as u32)
            .collect()
    }

    /// Number of indexed vectors.
    #[wasm_bindgen(js_name = "len")]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if the index holds no vectors.
    #[wasm_bindgen(js_name = "isEmpty")]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
```

In `crates/bindings/wasm/src/lib.rs`, add alongside the other `mod`/`use` declarations near the top (after the `use wasm_bindgen::prelude::*;` block, around line 28):

```rust
pub mod codecs;
```

- [ ] **Step 4: Run to verify it passes**

Run: `wasm-pack test --node crates/bindings/wasm -- --test web rabitq`
Expected: PASS — `rabitq_codec_encode_open_search_round_trip`.

Fallback if `wasm-pack` is unavailable: `cargo build -p grafeo-wasm --target wasm32-unknown-unknown` builds clean.

- [ ] **Step 5: Commit**

```bash
git add crates/bindings/wasm/src/codecs.rs crates/bindings/wasm/src/lib.rs crates/bindings/wasm/tests/web.rs
git commit -m "feat(wasm): RabitqCodec binding for encode/open/search"
```

---

## Task 10: Benchmark — RaBitQ two-stage vs Product Quantization

This produces the evidence for the "PQ is the fallback if RaBitQ proves too costly" decision.

**Files:**
- Create: `crates/grafeo-core/benches/rabitq_vs_pq.rs`
- Modify: `crates/grafeo-core/Cargo.toml`

- [ ] **Step 1: Register the bench**

In `crates/grafeo-core/Cargo.toml`, after the existing `[[bench]]` entries, add:

```toml
[[bench]]
name = "rabitq_vs_pq"
harness = false
```

- [ ] **Step 2: Write the benchmark**

Create `crates/grafeo-core/benches/rabitq_vs_pq.rs`:

```rust
//! RaBitQ two-stage vs Product Quantization: query latency and recall@10.
//!
//! Run: `cargo bench -p grafeo-core --bench rabitq_vs_pq`
//!
//! The recall numbers are printed once before the timed loops; use them
//! together with the criterion latency report to decide whether RaBitQ
//! replaces PQ or PQ remains the fallback.

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_core::index::vector::ProductQuantizer;
use grafeo_core::index::vector::TwoStageVectorIndex;
use std::hint::black_box;

const DIM: usize = 256;
const COUNT: usize = 5_000;
const K: usize = 10;

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

fn dataset() -> Vec<(NodeId, Vec<f32>)> {
    let mut rng = Rng(99);
    let centres: Vec<Vec<f32>> = (0..40)
        .map(|_| (0..DIM).map(|_| rng.gaussian() * 6.0).collect())
        .collect();
    let mut out = Vec::with_capacity(COUNT);
    for i in 0..COUNT {
        let c = &centres[i % centres.len()];
        let v = c.iter().map(|&x| x + rng.gaussian() * 0.5).collect();
        out.push((NodeId::new(i as u64 + 1), v));
    }
    out
}

fn brute_force_top_k(vectors: &[(NodeId, Vec<f32>)], query: &[f32]) -> Vec<NodeId> {
    let mut s: Vec<(NodeId, f32)> = vectors
        .iter()
        .map(|(id, v)| {
            (*id, v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum())
        })
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(K).map(|(id, _)| id).collect()
}

fn recall(truth: &[NodeId], got: &[NodeId]) -> f32 {
    got.iter().filter(|id| truth.contains(id)).count() as f32 / truth.len() as f32
}

fn bench(c: &mut Criterion) {
    let vectors = dataset();
    let query = vectors[7].1.clone();
    let truth = brute_force_top_k(&vectors, &query);

    // RaBitQ two-stage.
    let rabitq = TwoStageVectorIndex::build(&vectors, DIM, 1);
    let rabitq_hits: Vec<NodeId> =
        rabitq.search(&query, K, 16).into_iter().map(|(id, _)| id).collect();

    // PQ baseline: 32 subvectors, asymmetric distance scan.
    let refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
    let pq = ProductQuantizer::train(&refs, 32, 256, 10);
    let pq_codes: Vec<(NodeId, Vec<u8>)> =
        vectors.iter().map(|(id, v)| (*id, pq.quantize(v))).collect();
    let pq_search = |q: &[f32]| -> Vec<NodeId> {
        let table = pq.build_distance_table(q);
        let mut s: Vec<(NodeId, f32)> = pq_codes
            .iter()
            .map(|(id, codes)| (*id, pq.distance_with_table(&table, codes)))
            .collect();
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        s.into_iter().take(K).map(|(id, _)| id).collect()
    };
    let pq_hits = pq_search(&query);

    eprintln!("── recall@{K} (oracle = exact Euclidean) ──");
    eprintln!("  RaBitQ two-stage : {:.3}", recall(&truth, &rabitq_hits));
    eprintln!("  Product Quant.   : {:.3}", recall(&truth, &pq_hits));
    eprintln!("  RaBitQ code size : 32 B/vec (+8 B factors)  PQ: 32 B/vec");

    let mut group = c.benchmark_group("vector_search_5k_256d");
    group.bench_function("rabitq_two_stage", |b| {
        b.iter(|| black_box(rabitq.search(black_box(&query), K, 16)));
    });
    group.bench_function("product_quantization", |b| {
        b.iter(|| black_box(pq_search(black_box(&query))));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

- [ ] **Step 3: Run the benchmark**

Run: `cargo bench -p grafeo-core --bench rabitq_vs_pq`
Expected: compiles and runs; prints recall for both methods and criterion latency estimates. There is no pass/fail assertion — this is a measurement. Record the recall and latency numbers in the commit message.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-core/benches/rabitq_vs_pq.rs crates/grafeo-core/Cargo.toml
git commit -m "bench(rabitq): two-stage RaBitQ vs Product Quantization recall and latency"
```

---

## Task 11: CHANGELOG and full verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a CHANGELOG entry**

In `CHANGELOG.md`, under the current unreleased/working section, add a line consistent with the file's existing format, e.g.:

```markdown
### Added
- `index::vector::rabitq` — RaBitQ 1-bit vector quantization codec with a
  two-stage search (RaBitQ coarse pass + int8 rerank), zero-copy blob
  serialization, and a `RabitqCodec` WASM binding.
```

- [ ] **Step 2: Run the full verification suite**

Run each and confirm output before claiming completion (use `superpowers:verification-before-completion`):

```bash
cargo test -p grafeo-core --lib index::vector::rabitq
cargo test -p grafeo-core --test rabitq_recall
cargo clippy -p grafeo-core --all-targets -- -D warnings
cargo build -p grafeo-core --target wasm32-unknown-unknown
```

Expected: all green. The wasm build confirms the codec stays within the WASM toolchain (no `std`-only or non-portable dependency crept in).

If `wasm-pack` is installed, also run:

```bash
wasm-pack test --node crates/bindings/wasm -- --test web rabitq
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note RaBitQ vector codec"
```

---

## Self-Review (completed by the plan author)

**Spec coverage** — every Plan 2a requirement maps to a task:
- 1-bit-per-dimension binary quantization → Task 3.
- SIMD popcount/XOR distance → Task 4 (reuses `hamming_distance_simd`).
- 256-dim `f32` (1024 B) → 32 B → Task 3 (`encode_256_dim_yields_32_byte_code`).
- Two-stage search (RaBitQ coarse → int8 rerank) → Tasks 5–6.
- Confirm/add int8 scalar quantization → Task 6 reuses the existing
  `ScalarQuantizer` unchanged (confirmed sufficient — no new int8 code needed).
- Benchmark RaBitQ vs PQ → Task 10.
- WASM bindings → Task 9.
- Recall-floor regression test → Task 8.
- Component-wise encode to bytes / open and query → Task 7 (`to_bytes`/`from_bytes`).
- Zero-copy contract from the Plan 2 overview → Task 7 layout (fixed header,
  alignment, blob-relative offsets, self-describing, CRC).

**Type consistency** — checked across tasks: `RabitqQuantizer`, `RabitqCode`
(`bits`/`dot_oo`/`norm`), `RabitqQuery`, `RabitqIndex` (`with_quantizer`,
`load_entries`, `ids`, `codes`, `quantizer`), `TwoStageVectorIndex`
(`build`, `from_parts`, `parts`, `search`, `to_bytes`, `from_bytes`),
`RabitqError`, `Rotation` (`new_seeded`, `from_matrix`, `apply`, `matrix`)
all referenced consistently. `RabitqQuantizer::from_parts` and the
`pub(crate)` accessors are defined in Tasks 2–6 and consumed in Task 7.

**Placeholder scan** — no TBD/TODO; every code step shows complete code.

**Known follow-ups (out of scope for 2a, tracked in the Plan 2 overview):**
- A borrowing zero-copy *view* over the Task 7 blob layout — sub-plan 2d.
- HNSW topology over RaBitQ codes for sub-linear coarse search — the current
  `coarse_search` is a full linear scan, adequate for the snapshot sizes Plan 2
  targets and for the benchmark; a graph index is a separate optimisation.
