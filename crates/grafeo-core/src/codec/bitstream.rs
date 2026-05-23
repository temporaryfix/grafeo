//! Bit-level reader/writer with Elias gamma codes.
//!
//! Universal variable-length integer codes used by static graph and
//! sequence codecs. `gamma(n)` for `n >= 1` is `floor(log2(n))` zeros
//! followed by the binary of `n` (a unary prefix + the bits below the
//! top one). `gamma(1) = "1"`, `gamma(2) = "010"`, `gamma(4) = "00100"`.
//!
//! `zigzag_gamma(n)` for any `i64` maps `n` to `n >= 0 ? 2n+1 : -2n` and
//! gamma-encodes the result, giving variable-length encoding for signed
//! values (used for the first gap in an adjacency list, which may be
//! negative when `dst < src`).

/// Maximum integer encodable in `gamma` without overflow on the decoder
/// side: `2^63 - 1`. The encoded length for the maximum is 127 bits
/// (63 zeros + 64-bit binary).
pub(crate) const GAMMA_MAX: u64 = u64::MAX >> 1;

/// A growable bit-packed write buffer.
///
/// Bits are appended MSB-first within each byte; a `BitReader` over the
/// same buffer reads them in the same order. The total bit length is
/// tracked separately from the byte length so trailing zero-padding in
/// the last byte is not interpreted as part of the stream.
#[derive(Debug, Clone, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Number of bits actually written. Always `<= bytes.len() * 8`.
    bit_len: u64,
}

impl BitWriter {
    /// Creates an empty writer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the number of bits written so far.
    pub(crate) fn bit_len(&self) -> u64 {
        self.bit_len
    }

    /// Appends a single bit (LSB of `bit`).
    pub(crate) fn write_bit(&mut self, bit: u8) {
        // reason: bit fits in u8::from(bool)
        #[allow(clippy::cast_possible_truncation)]
        let byte_idx = (self.bit_len / 8) as usize;
        let bit_in_byte = 7 - (self.bit_len % 8) as u8;
        if byte_idx == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit & 1 != 0 {
            self.bytes[byte_idx] |= 1u8 << bit_in_byte;
        }
        self.bit_len += 1;
    }

    /// Appends the low `nbits` of `value`, most-significant bit first.
    pub(crate) fn write_bits(&mut self, value: u64, nbits: u8) {
        debug_assert!(nbits <= 64, "nbits must be <= 64");
        for i in (0..nbits).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Appends `gamma(n)` for `n >= 1`.
    ///
    /// # Panics
    /// Panics if `n == 0` or `n > GAMMA_MAX`.
    pub(crate) fn write_gamma(&mut self, n: u64) {
        assert!(n >= 1, "gamma encodes n >= 1, got 0");
        assert!(n <= GAMMA_MAX, "gamma overflow: n={n} > GAMMA_MAX");
        let bits_needed = 64 - n.leading_zeros() as u8; // 1..=64
        // unary prefix: (bits_needed - 1) zeros + a terminating 1.
        for _ in 0..(bits_needed - 1) {
            self.write_bit(0);
        }
        // The remaining body is the binary of n in `bits_needed` bits,
        // MSB first — but the MSB is the terminating 1 of the unary
        // prefix, so we write the full `bits_needed`-bit value here.
        self.write_bits(n, bits_needed);
    }

    /// Appends `zigzag_gamma(n)`: maps `n` to non-negative, then gamma.
    pub(crate) fn write_zigzag_gamma(&mut self, n: i64) {
        // n >= 0 -> 2n + 1; n < 0 -> -2n. Always yields a positive u64.
        let folded: u64 = if n >= 0 {
            (n as u64).checked_mul(2).expect("zigzag overflow")
                + 1
        } else {
            // reason: n is strictly negative, abs fits in u64
            #[allow(clippy::cast_sign_loss)]
            {
                ((-(n + 1)) as u64)
                    .checked_mul(2)
                    .expect("zigzag overflow")
                    + 2
            }
        };
        self.write_gamma(folded);
    }

    /// Consumes the writer and returns the underlying bytes plus the
    /// exact bit length.
    pub(crate) fn into_bytes(self) -> (Vec<u8>, u64) {
        (self.bytes, self.bit_len)
    }
}

/// A bit-level reader over a borrowed byte slice.
///
/// `bit_pos` tracks the current bit offset from the start of the slice;
/// callers can save and restore it to implement random access (the
/// WebGraph codec uses this for per-node seek via the offsets index).
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: u64,
    bit_len: u64,
}

impl<'a> BitReader<'a> {
    /// Creates a reader over `bytes` with the given total `bit_len`.
    pub(crate) fn new(bytes: &'a [u8], bit_len: u64) -> Self {
        Self {
            bytes,
            bit_pos: 0,
            bit_len,
        }
    }

    /// Seeks to `bit_pos` from the start of the stream.
    pub(crate) fn seek(&mut self, bit_pos: u64) {
        self.bit_pos = bit_pos;
    }

    /// Current bit position from the start of the stream.
    pub(crate) fn bit_pos(&self) -> u64 {
        self.bit_pos
    }

    /// Reads one bit (returns 0 or 1).
    ///
    /// # Errors
    /// Returns `None` if past the end of the stream.
    pub(crate) fn read_bit(&mut self) -> Option<u8> {
        if self.bit_pos >= self.bit_len {
            return None;
        }
        // reason: bit_pos < bit_len <= bytes.len() * 8, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let byte_idx = (self.bit_pos / 8) as usize;
        let bit_in_byte = 7 - (self.bit_pos % 8) as u8;
        self.bit_pos += 1;
        Some((self.bytes[byte_idx] >> bit_in_byte) & 1)
    }

    /// Reads `nbits` bits MSB-first into a `u64`.
    pub(crate) fn read_bits(&mut self, nbits: u8) -> Option<u64> {
        debug_assert!(nbits <= 64);
        let mut acc = 0u64;
        for _ in 0..nbits {
            acc = (acc << 1) | u64::from(self.read_bit()?);
        }
        Some(acc)
    }

    /// Reads one gamma-encoded integer.
    pub(crate) fn read_gamma(&mut self) -> Option<u64> {
        // Count leading zeros (unary prefix length).
        let mut zeros: u8 = 0;
        loop {
            match self.read_bit()? {
                0 => zeros += 1,
                _ => break,
            }
            if zeros > 63 {
                return None; // Malformed: prefix would overflow.
            }
        }
        // We consumed `zeros + 1` bits; the leading 1 is the high bit of
        // the value. Read the remaining `zeros` low bits.
        if zeros == 0 {
            return Some(1);
        }
        let low = self.read_bits(zeros)?;
        Some((1u64 << zeros) | low)
    }

    /// Reads one zigzag-gamma-encoded signed integer.
    pub(crate) fn read_zigzag_gamma(&mut self) -> Option<i64> {
        let folded = self.read_gamma()?;
        // odd -> non-negative, even -> negative.
        Some(if folded & 1 == 1 {
            // reason: folded - 1 is even, half fits i64 in practice
            #[allow(clippy::cast_possible_wrap)]
            (((folded - 1) >> 1) as i64)
        } else {
            // reason: folded is even and >= 2, half fits i64
            #[allow(clippy::cast_possible_wrap)]
            (-((folded >> 1) as i64))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_round_trip_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0b1011_0100, 8);
        w.write_bits(0b11, 2);
        let (bytes, bits) = w.into_bytes();
        assert_eq!(bits, 10);
        let mut r = BitReader::new(&bytes, bits);
        assert_eq!(r.read_bits(8), Some(0b1011_0100));
        assert_eq!(r.read_bits(2), Some(0b11));
        assert_eq!(r.read_bit(), None);
    }

    #[test]
    fn gamma_known_values() {
        // gamma(1) = "1" (1 bit), gamma(2) = "010" (3 bits),
        // gamma(3) = "011" (3 bits), gamma(4) = "00100" (5 bits),
        // gamma(7) = "00111" (5 bits), gamma(8) = "0001000" (7 bits).
        for n in [1u64, 2, 3, 4, 7, 8, 100, 12345] {
            let mut w = BitWriter::new();
            w.write_gamma(n);
            let (bytes, bits) = w.into_bytes();
            let mut r = BitReader::new(&bytes, bits);
            assert_eq!(r.read_gamma(), Some(n), "round-trip failed for {n}");
        }
    }

    #[test]
    fn zigzag_gamma_round_trip() {
        for n in [
            0i64,
            1,
            -1,
            2,
            -2,
            42,
            -42,
            1000,
            -1000,
            i32::MAX as i64,
            i32::MIN as i64,
        ] {
            let mut w = BitWriter::new();
            w.write_zigzag_gamma(n);
            let (bytes, bits) = w.into_bytes();
            let mut r = BitReader::new(&bytes, bits);
            assert_eq!(r.read_zigzag_gamma(), Some(n), "round-trip failed for {n}");
        }
    }

    #[test]
    fn gamma_then_bits_then_gamma() {
        // Multiple codes interleaved must round-trip without bit drift.
        let mut w = BitWriter::new();
        w.write_gamma(13);
        w.write_bits(0b101, 3);
        w.write_gamma(1);
        w.write_zigzag_gamma(-5);
        let (bytes, bits) = w.into_bytes();
        let mut r = BitReader::new(&bytes, bits);
        assert_eq!(r.read_gamma(), Some(13));
        assert_eq!(r.read_bits(3), Some(0b101));
        assert_eq!(r.read_gamma(), Some(1));
        assert_eq!(r.read_zigzag_gamma(), Some(-5));
    }

    #[test]
    fn bitwriter_records_exact_bit_length() {
        let mut w = BitWriter::new();
        for _ in 0..10 {
            w.write_bit(1);
        }
        assert_eq!(w.bit_len(), 10);
        let (bytes, bits) = w.into_bytes();
        // 10 bits → 2 bytes; the second byte has 2 high bits set.
        assert_eq!(bytes.len(), 2);
        assert_eq!(bits, 10);
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1] & 0b1100_0000, 0b1100_0000);
    }

    #[test]
    fn seek_enables_random_access() {
        let mut w = BitWriter::new();
        w.write_gamma(1);
        let pos_after_first = w.bit_len();
        w.write_gamma(2);
        let pos_after_second = w.bit_len();
        w.write_gamma(3);
        let (bytes, bits) = w.into_bytes();

        let mut r = BitReader::new(&bytes, bits);
        r.seek(pos_after_second);
        assert_eq!(r.read_gamma(), Some(3));

        r.seek(pos_after_first);
        assert_eq!(r.read_gamma(), Some(2));

        r.seek(0);
        assert_eq!(r.read_gamma(), Some(1));
    }
}
