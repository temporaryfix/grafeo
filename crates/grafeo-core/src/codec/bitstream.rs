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

use crate::codec::delta::{zigzag_decode, zigzag_encode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum BitWriterError {
    #[error("bit width {0} exceeds 64")]
    InvalidBitWidth(u8),
    #[error("gamma value {0} is outside 1..={GAMMA_MAX}")]
    InvalidGamma(u64),
    #[error("zigzag-gamma cannot represent {0}")]
    InvalidZigzagGamma(i64),
    #[error("bit-stream length exceeds u64::MAX")]
    LengthOverflow,
    #[error("bit-stream byte index exceeds the current address space")]
    AddressSpaceOverflow,
    #[error("cannot allocate {bytes} bytes for the bit stream")]
    AllocationFailed { bytes: usize },
    #[error("bit-stream writer state is inconsistent")]
    InconsistentState,
}

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

    fn reserve_through(&mut self, bit_len: u64) -> Result<(), BitWriterError> {
        let required_bytes = usize::try_from(bit_len.div_ceil(8))
            .map_err(|_| BitWriterError::AddressSpaceOverflow)?;
        if required_bytes > self.bytes.len() {
            let additional = required_bytes - self.bytes.len();
            self.bytes
                .try_reserve(additional)
                .map_err(|_| BitWriterError::AllocationFailed {
                    bytes: required_bytes,
                })?;
            while self.bytes.len() < required_bytes {
                self.bytes.push(0);
            }
        }
        Ok(())
    }

    fn advance_zeros(&mut self, bit_count: u8) -> Result<(), BitWriterError> {
        let next_bit_len = self
            .bit_len
            .checked_add(u64::from(bit_count))
            .ok_or(BitWriterError::LengthOverflow)?;
        self.reserve_through(next_bit_len)?;
        self.bit_len = next_bit_len;
        Ok(())
    }

    /// Appends the low `nbits` of `value`, most-significant bit first.
    pub(crate) fn write_bits(&mut self, value: u64, nbits: u8) -> Result<(), BitWriterError> {
        if nbits > 64 {
            return Err(BitWriterError::InvalidBitWidth(nbits));
        }
        let next_bit_len = self
            .bit_len
            .checked_add(u64::from(nbits))
            .ok_or(BitWriterError::LengthOverflow)?;
        self.reserve_through(next_bit_len)?;

        let mut byte_index =
            usize::try_from(self.bit_len / 8).map_err(|_| BitWriterError::AddressSpaceOverflow)?;
        let mut bit_in_byte =
            u8::try_from(self.bit_len % 8).map_err(|_| BitWriterError::InconsistentState)?;
        for shift in (0..nbits).rev() {
            if ((value >> shift) & 1) != 0 {
                let byte = self
                    .bytes
                    .get_mut(byte_index)
                    .ok_or(BitWriterError::InconsistentState)?;
                *byte |= 1_u8 << (7 - bit_in_byte);
            }
            bit_in_byte += 1;
            if bit_in_byte == 8 {
                bit_in_byte = 0;
                byte_index = byte_index
                    .checked_add(1)
                    .ok_or(BitWriterError::AddressSpaceOverflow)?;
            }
        }
        self.bit_len = next_bit_len;
        Ok(())
    }

    /// Appends `gamma(n)` for `n >= 1`.
    ///
    /// # Errors
    /// Returns [`BitWriterError`] if `n` is outside the supported gamma
    /// domain, the stream length overflows, or allocation fails.
    pub(crate) fn write_gamma(&mut self, n: u64) -> Result<(), BitWriterError> {
        if !(1..=GAMMA_MAX).contains(&n) {
            return Err(BitWriterError::InvalidGamma(n));
        }
        // reason: leading_zeros() for a u64 is in 0..=64, always fits u8
        #[allow(clippy::cast_possible_truncation)]
        let bits_needed = 64 - n.leading_zeros() as u8; // 1..=64
        // The unwritten tail is already zero-filled, so advancing over the
        // unary prefix avoids a checked write for every leading zero.
        self.advance_zeros(bits_needed - 1)?;
        // The remaining body is the binary of n in `bits_needed` bits,
        // MSB first — but the MSB is the terminating 1 of the unary
        // prefix, so we write the full `bits_needed`-bit value here.
        self.write_bits(n, bits_needed)
    }

    /// Appends `zigzag_gamma(n)`: folds `n` to non-negative via the canonical
    /// zig-zag map, then gamma-codes `folded + 1` (gamma needs `>= 1`).
    ///
    /// Supported domain: `|n|` up to roughly `2^62` (bounded by `GAMMA_MAX`),
    /// far beyond any graph's first-gap (which is bounded by the node count).
    /// # Errors
    /// Returns [`BitWriterError`] if `n` is not representable by the gamma
    /// domain, the stream length overflows, or allocation fails.
    pub(crate) fn write_zigzag_gamma(&mut self, n: i64) -> Result<(), BitWriterError> {
        let folded = zigzag_encode(n)
            .checked_add(1)
            .ok_or(BitWriterError::InvalidZigzagGamma(n))?;
        self.write_gamma(folded)
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

    /// Reads one bit (returns 0 or 1).
    ///
    /// # Errors
    /// Returns `None` if past the end of the stream.
    pub(crate) fn read_bit(&mut self) -> Option<u8> {
        if self.bit_pos >= self.bit_len {
            return None;
        }
        let byte_idx = usize::try_from(self.bit_pos / 8).ok()?;
        let bit_in_byte = 7 - (self.bit_pos % 8) as u8;
        let byte = self.bytes.get(byte_idx)?;
        self.bit_pos += 1;
        Some((byte >> bit_in_byte) & 1)
    }

    /// Reads `nbits` bits MSB-first into a `u64`.
    pub(crate) fn read_bits(&mut self, nbits: u8) -> Option<u64> {
        if nbits > 64 {
            return None;
        }
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
        // read_gamma returns >= 1, so folded - 1 >= 0 is a valid zig-zag code.
        let folded = self.read_gamma()?;
        Some(zigzag_decode(folded - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_gamma_round_trips_representable_domain() {
        // gamma caps at GAMMA_MAX = 2^63-1, so zigzag-gamma represents roughly
        // |n| <= 2^62-1 — far beyond any graph's first-gap (bounded by node
        // count). Exercise the full representable range including its edges.
        let max_repr = (1i64 << 62) - 1;
        let cases = [
            0i64,
            1,
            -1,
            2,
            -2,
            1000,
            -1000,
            max_repr,
            -max_repr,
            i32::MIN as i64,
            i32::MAX as i64,
            1i64 << 40,
            -(1i64 << 40),
        ];
        for &n in &cases {
            let mut w = BitWriter::new();
            w.write_zigzag_gamma(n).unwrap();
            let (bytes, bits) = w.into_bytes();
            let mut r = BitReader::new(&bytes, bits);
            assert_eq!(r.read_zigzag_gamma(), Some(n), "round-trip failed for {n}");
        }
    }

    #[test]
    fn zigzag_gamma_i64_min_returns_error() {
        let mut w = BitWriter::new();
        assert!(w.write_zigzag_gamma(i64::MIN).is_err());
    }

    #[test]
    fn gamma_outside_encodable_domain_returns_error() {
        let mut w = BitWriter::new();
        assert!(w.write_gamma(0).is_err());
        assert!(w.write_gamma(GAMMA_MAX + 1).is_err());
    }

    #[test]
    fn invalid_bit_width_is_rejected_without_mutating_the_stream() {
        let mut writer = BitWriter::new();
        assert!(matches!(
            writer.write_bits(0, 65),
            Err(BitWriterError::InvalidBitWidth(65))
        ));
        assert_eq!(writer.bit_len(), 0);
        assert!(writer.into_bytes().0.is_empty());

        let mut reader = BitReader::new(&[0], 8);
        assert_eq!(reader.read_bits(65), None);
        assert_eq!(reader.read_bits(8), Some(0));
    }

    #[test]
    fn reader_rejects_a_bit_length_larger_than_its_storage() {
        let mut reader = BitReader::new(&[], 1);
        assert_eq!(reader.read_bit(), None);
    }

    #[test]
    fn bits_round_trip_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0b1011_0100, 8).unwrap();
        w.write_bits(0b11, 2).unwrap();
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
            w.write_gamma(n).unwrap();
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
            w.write_zigzag_gamma(n).unwrap();
            let (bytes, bits) = w.into_bytes();
            let mut r = BitReader::new(&bytes, bits);
            assert_eq!(r.read_zigzag_gamma(), Some(n), "round-trip failed for {n}");
        }
    }

    #[test]
    fn gamma_then_bits_then_gamma() {
        // Multiple codes interleaved must round-trip without bit drift.
        let mut w = BitWriter::new();
        w.write_gamma(13).unwrap();
        w.write_bits(0b101, 3).unwrap();
        w.write_gamma(1).unwrap();
        w.write_zigzag_gamma(-5).unwrap();
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
            w.write_bits(1, 1).unwrap();
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
        w.write_gamma(1).unwrap();
        let pos_after_first = w.bit_len();
        w.write_gamma(2).unwrap();
        let pos_after_second = w.bit_len();
        w.write_gamma(3).unwrap();
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
