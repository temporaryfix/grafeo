//! Canonical application valid-time coordinates.
//!
//! Transaction time in Grafeo is expressed by [`super::EpochId`]. Application
//! (phenomenon) time is deliberately a different axis: signed nanoseconds on a
//! continuous TAI scale. Civil calendars, UTC offsets, and leap-second policy
//! belong at API boundaries and are not stored in the graph kernel.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Signed nanoseconds since `1970-01-01T00:00:00 TAI`.
///
/// `i128` is intentional. It keeps nanosecond precision without constraining
/// the engine to the roughly 584-year span of an `i64` nanosecond counter, and
/// makes conversion from the legacy microsecond representation exact.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct TaiNanoseconds(i128);

impl TaiNanoseconds {
    /// Nanoseconds in one legacy microsecond.
    pub const NANOS_PER_MICRO: i128 = 1_000;

    /// Creates a coordinate from its canonical signed nanosecond value.
    #[must_use]
    pub const fn new(value: i128) -> Self {
        Self(value)
    }

    /// Returns the canonical signed nanosecond value.
    #[must_use]
    pub const fn as_i128(self) -> i128 {
        self.0
    }

    /// Losslessly promotes the legacy signed microsecond coordinate.
    ///
    /// This is a deterministic unit migration (`×1000`), not a claim that
    /// historical bytes carried UTC, civil-time, or TAI scale metadata.
    #[must_use]
    pub fn from_legacy_micros(micros: i64) -> Self {
        Self(i128::from(micros) * Self::NANOS_PER_MICRO)
    }

    /// Converts to the legacy microsecond coordinate when no precision or
    /// range would be lost.
    #[must_use]
    pub fn to_legacy_micros_exact(self) -> Option<i64> {
        if self.0 % Self::NANOS_PER_MICRO != 0 {
            return None;
        }
        i64::try_from(self.0 / Self::NANOS_PER_MICRO).ok()
    }
}

impl From<i128> for TaiNanoseconds {
    fn from(value: i128) -> Self {
        Self::new(value)
    }
}

impl From<TaiNanoseconds> for i128 {
    fn from(value: TaiNanoseconds) -> Self {
        value.as_i128()
    }
}

impl fmt::Display for TaiNanoseconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} TAI ns", self.0)
    }
}

/// Error returned when a half-open valid-time interval is empty or inverted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("valid-time interval must satisfy from < to, got {from}..{to}")]
pub struct InvalidValidTimeInterval {
    from: TaiNanoseconds,
    to: TaiNanoseconds,
}

impl InvalidValidTimeInterval {
    /// Invalid inclusive lower bound.
    #[must_use]
    pub const fn from(self) -> TaiNanoseconds {
        self.from
    }

    /// Invalid exclusive upper bound.
    #[must_use]
    pub const fn to(self) -> TaiNanoseconds {
        self.to
    }
}

/// A non-empty half-open application valid-time interval `[from, to)`.
///
/// Construction and deserialization enforce `from < to`, so storage and query
/// code can rely on the interval invariant without repeating ad-hoc checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidTimeInterval {
    from: TaiNanoseconds,
    to: TaiNanoseconds,
}

impl ValidTimeInterval {
    /// Creates a non-empty half-open interval.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidValidTimeInterval`] unless `from < to`.
    pub const fn new(
        from: TaiNanoseconds,
        to: TaiNanoseconds,
    ) -> Result<Self, InvalidValidTimeInterval> {
        if from.as_i128() < to.as_i128() {
            Ok(Self { from, to })
        } else {
            Err(InvalidValidTimeInterval { from, to })
        }
    }

    /// Creates an interval directly from canonical TAI nanosecond values.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidValidTimeInterval`] unless `from < to`.
    pub const fn from_tai_nanoseconds(
        from: i128,
        to: i128,
    ) -> Result<Self, InvalidValidTimeInterval> {
        Self::new(TaiNanoseconds::new(from), TaiNanoseconds::new(to))
    }

    /// Losslessly promotes a legacy microsecond interval.
    ///
    /// This is a deterministic unit migration (`×1000`), not civil-time or
    /// leap-second conversion; the old format did not declare a time scale.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidValidTimeInterval`] unless `from_micros < to_micros`.
    pub fn from_legacy_micros(
        from_micros: i64,
        to_micros: i64,
    ) -> Result<Self, InvalidValidTimeInterval> {
        Self::new(
            TaiNanoseconds::from_legacy_micros(from_micros),
            TaiNanoseconds::from_legacy_micros(to_micros),
        )
    }

    /// Inclusive lower bound.
    #[must_use]
    pub const fn from(self) -> TaiNanoseconds {
        self.from
    }

    /// Exclusive upper bound.
    #[must_use]
    pub const fn to(self) -> TaiNanoseconds {
        self.to
    }

    /// Whether `instant` lies in this half-open interval.
    #[must_use]
    pub const fn contains(self, instant: TaiNanoseconds) -> bool {
        self.from.as_i128() <= instant.as_i128() && instant.as_i128() < self.to.as_i128()
    }
}

impl TryFrom<(TaiNanoseconds, TaiNanoseconds)> for ValidTimeInterval {
    type Error = InvalidValidTimeInterval;

    fn try_from((from, to): (TaiNanoseconds, TaiNanoseconds)) -> Result<Self, Self::Error> {
        Self::new(from, to)
    }
}

impl Serialize for ValidTimeInterval {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (self.from, self.to).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ValidTimeInterval {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (from, to) = <(TaiNanoseconds, TaiNanoseconds)>::deserialize(deserializer)?;
        Self::new(from, to).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_microseconds_promote_without_loss() {
        let interval = ValidTimeInterval::from_legacy_micros(-1, 2).unwrap();
        assert_eq!(interval.from().as_i128(), -1_000);
        assert_eq!(interval.to().as_i128(), 2_000);
        assert_eq!(interval.from().to_legacy_micros_exact(), Some(-1));
        assert_eq!(TaiNanoseconds::new(1_001).to_legacy_micros_exact(), None);
    }

    #[test]
    fn intervals_are_non_empty_and_half_open() {
        let interval = ValidTimeInterval::from_tai_nanoseconds(10, 11).unwrap();
        assert!(interval.contains(TaiNanoseconds::new(10)));
        assert!(!interval.contains(TaiNanoseconds::new(11)));
        assert!(ValidTimeInterval::from_tai_nanoseconds(10, 10).is_err());
        assert!(ValidTimeInterval::from_tai_nanoseconds(11, 10).is_err());
    }

    #[test]
    fn deserialization_rechecks_the_interval_invariant() {
        let bytes = bincode::serde::encode_to_vec(
            (TaiNanoseconds::new(2), TaiNanoseconds::new(1)),
            bincode::config::standard(),
        )
        .unwrap();
        let decoded = bincode::serde::decode_from_slice::<ValidTimeInterval, _>(
            &bytes,
            bincode::config::standard(),
        );
        assert!(decoded.is_err());
    }
}
