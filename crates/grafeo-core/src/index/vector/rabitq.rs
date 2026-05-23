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
}
