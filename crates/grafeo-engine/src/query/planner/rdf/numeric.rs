//! Exact numeric values shared by RDF expressions, aggregates, and ordering.
//!
//! SPARQL numeric promotion is not SQL-style `f64` coercion. Integer and
//! decimal values stay arbitrary precision, `xsd:float` stays `f32`, and only
//! `xsd:double` uses `f64`.

use std::cmp::Ordering;
#[cfg(feature = "spill")]
use std::io::{Cursor, Read, Write};
use std::str::FromStr;

use bigdecimal::{BigDecimal, FromPrimitive, RoundingMode, ToPrimitive, Zero};
use grafeo_common::types::Value;
use grafeo_core::graph::rdf::Literal;

use super::{NumericKind, parse_xsd_double, parse_xsd_float, rdf_numeric_literal_is_valid};

const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";

#[derive(Clone)]
pub(super) struct RdfNumeric {
    kind: NumericKind,
    repr: RdfNumericRepr,
}

#[derive(Clone)]
enum RdfNumericRepr {
    Exact(BigDecimal),
    Float(f32),
    Double(f64),
}

// BigDecimal's owned multiplication by zero calls BigInt::set_zero, which
// clears a potentially large backing vector without shrinking it. Keep the
// exact scale while discarding that inaccessible spare capacity, so retained
// accounting can safely depend on the normalized coefficient length.
fn compact_zero_coefficient(repr: RdfNumericRepr) -> RdfNumericRepr {
    match repr {
        RdfNumericRepr::Exact(value) if value.is_zero() => {
            let (_, scale) = value.as_bigint_and_scale();
            RdfNumericRepr::Exact(BigDecimal::new(0.into(), scale))
        }
        repr => repr,
    }
}

/// Covers literal validation, lexical copies, coefficient parsing and its
/// retained result. RDF integer/decimal grammar rejects exponent notation;
/// IEEE parsing has bounded exponent handling rather than decimal expansion.
#[cfg(test)]
pub(super) fn numeric_parse_scratch_bytes(value: &Value) -> Option<usize> {
    numeric_workspace(value.retained_size_bytes()?)
}

// Pinned BigDecimal/num-bigint arithmetic keeps a constant number of full
// coefficients, radix buffers, and geometrically shrinking multiplication
// scratch arrays live. 256 bytes per (overestimated) decimal digit covers
// these simultaneous buffers, limb capacity slack, and overlapping String
// growth. The fixed reserve also covers IEEE conversion and formatting.
fn numeric_workspace(digits: usize) -> Option<usize> {
    digits
        .checked_add(32)?
        .checked_mul(256)?
        .checked_add(64 * 1024)
}

impl RdfNumeric {
    /// Conservative coefficient storage, without cloning or formatting it.
    ///
    /// Pinned num-bigint 0.4.6 normalizes nonzero arithmetic results and shrinks
    /// when length is below capacity / 4. Eight-byte limbs cover both limb
    /// configurations; the three spare limbs cover the integer division in
    /// that threshold. Arithmetic zero results are compacted below because
    /// BigDecimal multiplication can clear a coefficient without shrinking it.
    /// Aggregate updates must also retain their admitted peak
    /// until the previous value has been dropped.
    pub(super) fn retained_bytes(&self) -> Option<usize> {
        let heap = match &self.repr {
            RdfNumericRepr::Exact(value) => {
                let (coefficient, _) = value.as_bigint_and_scale();
                let limbs = usize::try_from(coefficient.bits()).ok()?.div_ceil(32);
                limbs.checked_mul(4)?.checked_add(3)?.checked_mul(8)?
            }
            RdfNumericRepr::Float(_) | RdfNumericRepr::Double(_) => 0,
        };
        std::mem::size_of::<Self>().checked_add(heap)
    }

    /// Peak temporary bytes for consuming addition, including both operands
    /// and its result. Scale alignment is charged before constructing 10^n.
    #[cfg(any(test, feature = "spill"))]
    pub(super) fn add_scratch_bytes(&self, other: &Self) -> Option<usize> {
        let (left, left_scale) = self.coefficient_bound()?;
        let (right, right_scale) = other.coefficient_bound()?;
        let scale_gap = left_scale.checked_sub(right_scale)?.unsigned_abs();
        let span = left
            .checked_add(right)?
            .checked_add(usize::try_from(scale_gap).ok()?)?;
        numeric_workspace(span)?
            .checked_add(self.retained_bytes()?)?
            .checked_add(other.retained_bytes()?)
    }

    /// Peak for cloning, optional AVG division, and conversion to an RDF value.
    /// The bound includes plain notation: a tiny coefficient with a large
    /// exponent cannot trigger unadmitted zero padding during formatting.
    pub(super) fn finalize_scratch_bytes(&self, average_count: Option<u64>) -> Option<usize> {
        let (digits, scale) = self.coefficient_bound()?;
        let division_digits = if average_count.is_some() {
            let precision = bigdecimal::Context::default().precision().get();
            let extra = precision.checked_add(21)?;
            // BigDecimal division increments its signed scale internally.
            scale.checked_add(i64::try_from(extra).ok()?)?;
            usize::try_from(extra).ok()?
        } else {
            0
        };
        let span = digits
            .checked_add(usize::try_from(scale.unsigned_abs()).ok()?)?
            .checked_add(division_digits)?;
        numeric_workspace(span)?.checked_add(self.retained_bytes()?)
    }

    // Decimal digit count itself allocates in BigDecimal. Bit count is a
    // deliberately larger, allocation-free decimal-digit bound instead.
    fn coefficient_bound(&self) -> Option<(usize, i64)> {
        match &self.repr {
            RdfNumericRepr::Exact(value) => {
                let (coefficient, scale) = value.as_bigint_and_scale();
                Some((usize::try_from(coefficient.bits()).ok()?.max(1), scale))
            }
            RdfNumericRepr::Float(_) | RdfNumericRepr::Double(_) => Some((1075, 0)),
        }
    }

    pub(super) fn from_literal(literal: &Literal) -> Option<Self> {
        let kind = super::numeric_kind(literal.datatype())?;
        if !rdf_numeric_literal_is_valid(literal) {
            return None;
        }
        let repr = match kind {
            NumericKind::Integer | NumericKind::Decimal => {
                RdfNumericRepr::Exact(BigDecimal::from_str(literal.value()).ok()?)
            }
            NumericKind::Float => RdfNumericRepr::Float(parse_xsd_float(literal.value())?),
            NumericKind::Double => RdfNumericRepr::Double(parse_xsd_double(literal.value())?),
        };
        Some(Self { kind, repr })
    }

    /// Parses native RDF values without treating plain strings as numbers.
    pub(super) fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int64(value) => Some(Self {
                kind: NumericKind::Integer,
                repr: RdfNumericRepr::Exact(BigDecimal::from(*value)),
            }),
            Value::Float64(value) => Some(Self {
                kind: NumericKind::Double,
                repr: RdfNumericRepr::Double(*value),
            }),
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } => Self::from_literal(&Literal::typed(lexical.as_str(), datatype.as_str())),
            _ => None,
        }
    }

    /// Retains the historical compatibility coercion for numeric-looking
    /// strings at expression boundaries. Exact RDF literals never use it.
    pub(super) fn from_compatible_value(value: &Value) -> Option<Self> {
        Self::from_value(value).or_else(|| match value {
            Value::String(value) => value.parse().ok().map(|value| Self {
                kind: NumericKind::Double,
                repr: RdfNumericRepr::Double(value),
            }),
            _ => None,
        })
    }

    pub(super) fn checked_add(self, other: Self) -> Option<Self> {
        self.binary(
            other,
            |left, right| left + right,
            |left, right| left + right,
            |left, right| left + right,
        )
    }

    pub(super) fn checked_sub(self, other: Self) -> Option<Self> {
        self.binary(
            other,
            |left, right| left - right,
            |left, right| left - right,
            |left, right| left - right,
        )
    }

    pub(super) fn checked_mul(self, other: Self) -> Option<Self> {
        self.binary(
            other,
            |left, right| left * right,
            |left, right| left * right,
            |left, right| left * right,
        )
    }

    pub(super) fn checked_div(self, other: Self) -> Option<Self> {
        let target = if self.kind == NumericKind::Integer && other.kind == NumericKind::Integer {
            NumericKind::Decimal
        } else {
            self.kind.max(other.kind)
        };
        let (left, right) = Self::promote_pair(self, other, target)?;
        let repr = match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) if !right.is_zero() => {
                RdfNumericRepr::Exact(left / right)
            }
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) => {
                RdfNumericRepr::Float(left / right)
            }
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) => {
                RdfNumericRepr::Double(left / right)
            }
            _ => return None,
        };
        Some(Self {
            kind: target,
            repr: compact_zero_coefficient(repr),
        })
    }

    pub(super) fn checked_rem(self, other: Self) -> Option<Self> {
        let target = self.kind.max(other.kind);
        let (left, right) = Self::promote_pair(self, other, target)?;
        let repr = match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) if !right.is_zero() => {
                RdfNumericRepr::Exact(left % right)
            }
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) if right != 0.0 => {
                RdfNumericRepr::Float(left % right)
            }
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) if right != 0.0 => {
                RdfNumericRepr::Double(left % right)
            }
            _ => return None,
        };
        Some(Self {
            kind: target,
            repr: compact_zero_coefficient(repr),
        })
    }

    pub(super) fn negated(mut self) -> Self {
        self.repr = match self.repr {
            RdfNumericRepr::Exact(value) => RdfNumericRepr::Exact(-value),
            RdfNumericRepr::Float(value) => RdfNumericRepr::Float(-value),
            RdfNumericRepr::Double(value) => RdfNumericRepr::Double(-value),
        };
        self
    }

    pub(super) fn absolute(mut self) -> Self {
        self.repr = match self.repr {
            RdfNumericRepr::Exact(value) => RdfNumericRepr::Exact(value.abs()),
            RdfNumericRepr::Float(value) => RdfNumericRepr::Float(value.abs()),
            RdfNumericRepr::Double(value) => RdfNumericRepr::Double(value.abs()),
        };
        self
    }

    pub(super) fn ceiling(mut self) -> Self {
        self.repr = match self.repr {
            RdfNumericRepr::Exact(value) if self.kind == NumericKind::Integer => {
                RdfNumericRepr::Exact(value)
            }
            RdfNumericRepr::Exact(value) => {
                RdfNumericRepr::Exact(value.with_scale_round(0, RoundingMode::Ceiling))
            }
            RdfNumericRepr::Float(value) => RdfNumericRepr::Float(value.ceil()),
            RdfNumericRepr::Double(value) => RdfNumericRepr::Double(value.ceil()),
        };
        self
    }

    pub(super) fn floor(mut self) -> Self {
        self.repr = match self.repr {
            RdfNumericRepr::Exact(value) if self.kind == NumericKind::Integer => {
                RdfNumericRepr::Exact(value)
            }
            RdfNumericRepr::Exact(value) => {
                RdfNumericRepr::Exact(value.with_scale_round(0, RoundingMode::Floor))
            }
            RdfNumericRepr::Float(value) => RdfNumericRepr::Float(value.floor()),
            RdfNumericRepr::Double(value) => RdfNumericRepr::Double(value.floor()),
        };
        self
    }

    pub(super) fn rounded(mut self) -> Self {
        self.repr = match self.repr {
            RdfNumericRepr::Exact(value) if self.kind == NumericKind::Integer => {
                RdfNumericRepr::Exact(value)
            }
            RdfNumericRepr::Exact(value) => {
                let mode = if value < BigDecimal::zero() {
                    RoundingMode::HalfDown
                } else {
                    RoundingMode::HalfUp
                };
                RdfNumericRepr::Exact(value.with_scale_round(0, mode))
            }
            RdfNumericRepr::Float(value) => {
                RdfNumericRepr::Float(round_f32_toward_positive_infinity(value))
            }
            RdfNumericRepr::Double(value) => {
                RdfNumericRepr::Double(round_f64_toward_positive_infinity(value))
            }
        };
        self
    }

    pub(super) fn compare(&self, other: &Self) -> Option<Ordering> {
        let target = self.kind.max(other.kind);
        let (left, right) = Self::promote_pair(self.clone(), other.clone(), target)?;
        match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) => left.partial_cmp(&right),
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) => left.partial_cmp(&right),
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) => {
                left.partial_cmp(&right)
            }
            _ => None,
        }
    }

    /// Numeric equality is defined for NaN even though partial ordering is not:
    /// NaN is unequal to every numeric value, including itself.
    pub(super) fn equal(&self, other: &Self) -> Option<bool> {
        let target = self.kind.max(other.kind);
        let (left, right) = Self::promote_pair(self.clone(), other.clone(), target)?;
        match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) => Some(left == right),
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) => Some(left == right),
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) => Some(left == right),
            _ => None,
        }
    }

    pub(super) fn less_than(&self, other: &Self) -> Option<bool> {
        let target = self.kind.max(other.kind);
        let (left, right) = Self::promote_pair(self.clone(), other.clone(), target)?;
        match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) => Some(left < right),
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) => Some(left < right),
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) => Some(left < right),
            _ => None,
        }
    }

    pub(super) fn greater_than(&self, other: &Self) -> Option<bool> {
        let target = self.kind.max(other.kind);
        let (left, right) = Self::promote_pair(self.clone(), other.clone(), target)?;
        match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) => Some(left > right),
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) => Some(left > right),
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) => Some(left > right),
            _ => None,
        }
    }

    /// Deterministic extension ordering for blocking operators. SPARQL leaves
    /// the relative order of NaN and ordinary numbers undefined, and pairwise
    /// SPARQL promotion itself is not transitive across integer/float/double.
    /// Convert every finite representation to its exact mathematical decimal
    /// value so the blocking comparator has one stable equivalence relation.
    pub(super) fn compare_for_order(&self, other: &Self) -> Option<Ordering> {
        Some(self.order_key()?.compare(&other.order_key()?))
    }

    pub(super) fn effective_boolean_value(&self) -> bool {
        match &self.repr {
            RdfNumericRepr::Exact(value) => !value.is_zero(),
            RdfNumericRepr::Float(value) => *value != 0.0 && !value.is_nan(),
            RdfNumericRepr::Double(value) => *value != 0.0 && !value.is_nan(),
        }
    }

    pub(super) fn as_f64(&self) -> f64 {
        match &self.repr {
            RdfNumericRepr::Exact(value) => decimal_to_f64(value).unwrap_or_else(|| {
                if value < &BigDecimal::zero() {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }),
            RdfNumericRepr::Float(value) => f64::from(*value),
            RdfNumericRepr::Double(value) => *value,
        }
    }

    pub(super) fn as_f32(&self) -> Option<f32> {
        match &self.repr {
            RdfNumericRepr::Exact(value) => decimal_to_f32(value),
            RdfNumericRepr::Float(value) => Some(*value),
            RdfNumericRepr::Double(value) => value.to_f32(),
        }
    }

    pub(super) fn average(self, count: u64) -> Option<Self> {
        if count == 0 {
            return None;
        }
        match self.repr {
            RdfNumericRepr::Exact(value) => Some(Self {
                kind: NumericKind::Decimal,
                repr: RdfNumericRepr::Exact(value / BigDecimal::from(count)),
            }),
            RdfNumericRepr::Float(value) => Some(Self {
                kind: NumericKind::Float,
                repr: RdfNumericRepr::Float(value / count as f32),
            }),
            RdfNumericRepr::Double(value) => Some(Self {
                kind: NumericKind::Double,
                repr: RdfNumericRepr::Double(value / count as f64),
            }),
        }
    }

    pub(super) fn into_value(self) -> Value {
        match (self.repr, self.kind) {
            (RdfNumericRepr::Exact(value), NumericKind::Integer) => integer_value(&value),
            (RdfNumericRepr::Exact(value), NumericKind::Decimal) => decimal_value(value),
            (RdfNumericRepr::Float(value), NumericKind::Float) => float_value(value),
            (RdfNumericRepr::Double(value), NumericKind::Double) => Value::Float64(value),
            _ => Value::Null,
        }
    }

    #[cfg(feature = "spill")]
    pub(super) fn write_spill<W: Write + ?Sized>(&self, writer: &mut W) -> std::io::Result<()> {
        let kind = match self.kind {
            NumericKind::Integer => 0,
            NumericKind::Decimal => 1,
            NumericKind::Float => 2,
            NumericKind::Double => 3,
        };
        writer.write_all(&[kind])?;
        match &self.repr {
            RdfNumericRepr::Exact(value) => {
                if !matches!(self.kind, NumericKind::Integer | NumericKind::Decimal) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "exact RDF numeric state has a floating-point kind",
                    ));
                }
                if self.kind == NumericKind::Integer && !value.is_integer() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "integer RDF numeric state has a fractional value",
                    ));
                }
                let value = value.to_string();
                let length = u32::try_from(value.len()).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "exact RDF numeric state exceeds the spill format",
                    )
                })?;
                writer.write_all(&length.to_le_bytes())?;
                writer.write_all(value.as_bytes())?;
            }
            RdfNumericRepr::Float(value) if self.kind == NumericKind::Float => {
                writer.write_all(&value.to_bits().to_le_bytes())?;
            }
            RdfNumericRepr::Double(value) if self.kind == NumericKind::Double => {
                writer.write_all(&value.to_bits().to_le_bytes())?;
            }
            RdfNumericRepr::Float(_) | RdfNumericRepr::Double(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "RDF numeric representation does not match its promotion kind",
                ));
            }
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    pub(super) fn read_spill(reader: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        let mut kind = [0u8; 1];
        reader.read_exact(&mut kind)?;
        match kind[0] {
            exact_kind @ (0 | 1) => {
                let mut length = [0u8; 4];
                reader.read_exact(&mut length)?;
                let length = usize::try_from(u32::from_le_bytes(length)).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "exact RDF numeric length does not fit this platform",
                    )
                })?;
                let position = usize::try_from(reader.position()).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "RDF numeric reader position does not fit this platform",
                    )
                })?;
                let remaining = reader
                    .get_ref()
                    .len()
                    .checked_sub(position)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "RDF numeric reader advanced beyond its payload",
                        )
                    })?;
                if length > remaining {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "exact RDF numeric length exceeds the remaining spill record",
                    ));
                }
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(length).map_err(|error| {
                    std::io::Error::other(format!(
                        "cannot reserve exact RDF numeric spill state: {error}"
                    ))
                })?;
                bytes.resize(length, 0);
                reader.read_exact(&mut bytes)?;
                let lexical = std::str::from_utf8(&bytes).map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("exact RDF numeric state is not UTF-8: {error}"),
                    )
                })?;
                let value = BigDecimal::from_str(lexical).map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("invalid exact RDF numeric state: {error}"),
                    )
                })?;
                if exact_kind == 0 && !value.is_integer() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "integer RDF numeric state has a fractional value",
                    ));
                }
                Ok(Self {
                    kind: if exact_kind == 0 {
                        NumericKind::Integer
                    } else {
                        NumericKind::Decimal
                    },
                    repr: RdfNumericRepr::Exact(value),
                })
            }
            2 => {
                let mut bits = [0u8; 4];
                reader.read_exact(&mut bits)?;
                Ok(Self {
                    kind: NumericKind::Float,
                    repr: RdfNumericRepr::Float(f32::from_bits(u32::from_le_bytes(bits))),
                })
            }
            3 => {
                let mut bits = [0u8; 8];
                reader.read_exact(&mut bits)?;
                Ok(Self {
                    kind: NumericKind::Double,
                    repr: RdfNumericRepr::Double(f64::from_bits(u64::from_le_bytes(bits))),
                })
            }
            tag => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown RDF numeric spill tag {tag}"),
            )),
        }
    }

    fn binary(
        self,
        other: Self,
        exact: impl FnOnce(BigDecimal, BigDecimal) -> BigDecimal,
        float: impl FnOnce(f32, f32) -> f32,
        double: impl FnOnce(f64, f64) -> f64,
    ) -> Option<Self> {
        let target = self.kind.max(other.kind);
        let (left, right) = Self::promote_pair(self, other, target)?;
        let repr = match (left, right) {
            (RdfNumericRepr::Exact(left), RdfNumericRepr::Exact(right)) => {
                RdfNumericRepr::Exact(exact(left, right))
            }
            (RdfNumericRepr::Float(left), RdfNumericRepr::Float(right)) => {
                RdfNumericRepr::Float(float(left, right))
            }
            (RdfNumericRepr::Double(left), RdfNumericRepr::Double(right)) => {
                RdfNumericRepr::Double(double(left, right))
            }
            _ => return None,
        };
        Some(Self {
            kind: target,
            repr: compact_zero_coefficient(repr),
        })
    }

    fn promote_pair(
        left: Self,
        right: Self,
        target: NumericKind,
    ) -> Option<(RdfNumericRepr, RdfNumericRepr)> {
        Some((left.promote(target)?, right.promote(target)?))
    }

    fn promote(self, target: NumericKind) -> Option<RdfNumericRepr> {
        match (self.repr, target) {
            (RdfNumericRepr::Exact(value), NumericKind::Integer | NumericKind::Decimal) => {
                Some(RdfNumericRepr::Exact(value))
            }
            (RdfNumericRepr::Exact(value), NumericKind::Float) => {
                decimal_to_f32(&value).map(RdfNumericRepr::Float)
            }
            (RdfNumericRepr::Exact(value), NumericKind::Double) => {
                decimal_to_f64(&value).map(RdfNumericRepr::Double)
            }
            (RdfNumericRepr::Float(value), NumericKind::Float) => {
                Some(RdfNumericRepr::Float(value))
            }
            (RdfNumericRepr::Float(value), NumericKind::Double) => {
                Some(RdfNumericRepr::Double(f64::from(value)))
            }
            (RdfNumericRepr::Double(value), NumericKind::Double) => {
                Some(RdfNumericRepr::Double(value))
            }
            _ => None,
        }
    }

    fn order_key(&self) -> Option<RdfNumericOrderKey> {
        match &self.repr {
            RdfNumericRepr::Exact(value) => Some(RdfNumericOrderKey::Finite(value.clone())),
            RdfNumericRepr::Float(value) if value.is_nan() => Some(RdfNumericOrderKey::NaN),
            RdfNumericRepr::Float(value) if *value == f32::NEG_INFINITY => {
                Some(RdfNumericOrderKey::NegativeInfinity)
            }
            RdfNumericRepr::Float(value) if *value == f32::INFINITY => {
                Some(RdfNumericOrderKey::PositiveInfinity)
            }
            RdfNumericRepr::Float(value) => {
                BigDecimal::from_f32(*value).map(RdfNumericOrderKey::Finite)
            }
            RdfNumericRepr::Double(value) if value.is_nan() => Some(RdfNumericOrderKey::NaN),
            RdfNumericRepr::Double(value) if *value == f64::NEG_INFINITY => {
                Some(RdfNumericOrderKey::NegativeInfinity)
            }
            RdfNumericRepr::Double(value) if *value == f64::INFINITY => {
                Some(RdfNumericOrderKey::PositiveInfinity)
            }
            RdfNumericRepr::Double(value) => {
                BigDecimal::from_f64(*value).map(RdfNumericOrderKey::Finite)
            }
        }
    }
}

enum RdfNumericOrderKey {
    NaN,
    NegativeInfinity,
    Finite(BigDecimal),
    PositiveInfinity,
}

impl RdfNumericOrderKey {
    fn compare(&self, other: &Self) -> Ordering {
        use RdfNumericOrderKey::{Finite, NaN, NegativeInfinity, PositiveInfinity};

        match (self, other) {
            (NaN, NaN)
            | (NegativeInfinity, NegativeInfinity)
            | (PositiveInfinity, PositiveInfinity) => Ordering::Equal,
            (NaN, _) | (_, PositiveInfinity) => Ordering::Less,
            (_, NaN) | (PositiveInfinity, _) => Ordering::Greater,
            (NegativeInfinity, _) => Ordering::Less,
            (_, NegativeInfinity) => Ordering::Greater,
            (Finite(left), Finite(right)) => left.cmp(right),
        }
    }
}

fn round_f32_toward_positive_infinity(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let rounded = value.round();
    let rounded = if value.is_sign_negative() && value.fract().abs() == 0.5 {
        rounded + 1.0
    } else {
        rounded
    };
    if rounded == 0.0 && value.is_sign_negative() {
        -0.0
    } else {
        rounded
    }
}

fn round_f64_toward_positive_infinity(value: f64) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let rounded = value.round();
    let rounded = if value.is_sign_negative() && value.fract().abs() == 0.5 {
        rounded + 1.0
    } else {
        rounded
    };
    if rounded == 0.0 && value.is_sign_negative() {
        -0.0
    } else {
        rounded
    }
}

fn integer_value(value: &BigDecimal) -> Value {
    if let Some(value) = value.to_i64() {
        Value::Int64(value)
    } else {
        Value::RdfLiteral {
            lexical: value.with_scale(0).to_string().into(),
            language: None,
            datatype: Some(Literal::XSD_INTEGER.into()),
        }
    }
}

fn decimal_value(value: BigDecimal) -> Value {
    // BigDecimal's Display switches to exponent notation outside its display
    // thresholds, but exponent notation is not in the xsd:decimal lexical
    // space. Preserve a valid decimal term across subquery/result boundaries.
    let mut lexical = value.normalized().to_plain_string();
    if !lexical.contains('.') {
        lexical.push_str(".0");
    }
    Value::RdfLiteral {
        lexical: lexical.into(),
        language: None,
        datatype: Some(Literal::XSD_DECIMAL.into()),
    }
}

fn float_value(value: f32) -> Value {
    let lexical = match value {
        value if value.is_nan() => "NaN".to_string(),
        f32::INFINITY => "INF".to_string(),
        f32::NEG_INFINITY => "-INF".to_string(),
        value => value.to_string(),
    };
    Value::RdfLiteral {
        lexical: lexical.into(),
        language: None,
        datatype: Some(XSD_FLOAT.into()),
    }
}

fn decimal_to_f32(value: &BigDecimal) -> Option<f32> {
    value.to_f32().or_else(|| {
        (!value.is_zero()).then(|| {
            if value < &BigDecimal::zero() {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        })
    })
}

fn decimal_to_f64(value: &BigDecimal) -> Option<f64> {
    value.to_f64().or_else(|| {
        (!value.is_zero()).then(|| {
            if value < &BigDecimal::zero() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric(lexical: &str, datatype: &str) -> RdfNumeric {
        RdfNumeric::from_literal(&Literal::typed(lexical, datatype)).unwrap()
    }

    #[test]
    fn zero_multiplication_discards_hidden_coefficient_capacity() {
        let huge = numeric(&"9".repeat(8192), Literal::XSD_INTEGER);
        let zero = huge
            .checked_mul(numeric("0.00", Literal::XSD_DECIMAL))
            .unwrap();
        let RdfNumericRepr::Exact(coefficient) = &zero.repr else {
            panic!("exact result");
        };
        assert_eq!(coefficient.as_bigint_and_scale().1, 2);
        assert!(zero.retained_bytes().unwrap() < 1024);
        assert_eq!(
            zero.clone().into_value(),
            numeric("0.0", Literal::XSD_DECIMAL).into_value()
        );
        let rhs = numeric("0.125", Literal::XSD_DECIMAL);
        let scratch = zero.add_scratch_bytes(&rhs).unwrap();
        let sum = zero.checked_add(rhs).unwrap();
        assert!(scratch >= sum.retained_bytes().unwrap());
        let final_bound = sum.finalize_scratch_bytes(None).unwrap();
        let output = sum.into_value();
        assert!(final_bound >= output.retained_size_bytes().unwrap());
        assert_eq!(output, numeric("0.125", Literal::XSD_DECIMAL).into_value());
    }

    #[test]
    fn aggregate_numeric_bounds_cover_scale_alignment_and_final_values() {
        let left = numeric("999999999999999999999999999999", Literal::XSD_INTEGER);
        let right = numeric("0.000000000000000000000000000001", Literal::XSD_DECIMAL);
        let peak = left.add_scratch_bytes(&right).unwrap();
        let sum = left.checked_add(right).unwrap();
        assert!(peak >= sum.retained_bytes().unwrap());
        for count in [None, Some(3), Some(u64::MAX)] {
            let peak = sum.finalize_scratch_bytes(count).unwrap();
            let output = match count {
                Some(count) => sum.clone().average(count).unwrap().into_value(),
                None => sum.clone().into_value(),
            };
            assert!(peak >= output.retained_size_bytes().unwrap());
        }
        let cancellation = sum.clone().checked_add(sum.negated()).unwrap();
        assert!(cancellation.retained_bytes().unwrap() < 1024);
    }

    #[test]
    fn aggregate_numeric_bounds_reject_exponent_amplification_before_allocation() {
        let huge = RdfNumeric {
            kind: NumericKind::Decimal,
            repr: RdfNumericRepr::Exact(BigDecimal::new(1.into(), i64::MAX)),
        };
        assert!(huge.finalize_scratch_bytes(None).is_none());
        assert!(huge.finalize_scratch_bytes(Some(3)).is_none());
        assert!(
            huge.add_scratch_bytes(&numeric("1", Literal::XSD_INTEGER))
                .is_none()
        );
        assert!(numeric_workspace(usize::MAX).is_none());
        let literal = Value::RdfLiteral {
            lexical: "9".repeat(4096).into(),
            language: None,
            datatype: Some(Literal::XSD_INTEGER.into()),
        };
        let bound = numeric_parse_scratch_bytes(&literal).unwrap();
        let parsed = RdfNumeric::from_value(&literal).unwrap();
        assert!(bound >= parsed.retained_bytes().unwrap());
    }

    #[test]
    fn exact_arithmetic_crosses_i64_without_overflow() {
        let value = numeric("9223372036854775807", Literal::XSD_INTEGER)
            .checked_add(numeric("1", Literal::XSD_INTEGER))
            .unwrap()
            .into_value();
        assert!(matches!(
            value,
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "9223372036854775808"
                    && datatype.as_str() == Literal::XSD_INTEGER
        ));
    }

    #[test]
    fn exact_decimal_addition_does_not_round_through_binary_float() {
        let value = numeric("0.3", Literal::XSD_DECIMAL)
            .checked_add(numeric("0.1", Literal::XSD_DECIMAL))
            .unwrap()
            .into_value();
        assert!(matches!(
            value,
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "0.4"
                    && datatype.as_str() == Literal::XSD_DECIMAL
        ));
    }

    #[test]
    fn decimal_values_always_use_plain_xsd_lexicals() {
        for (input, expected) in [
            ("0.0000001", "0.0000001"),
            ("10000000000000000", "10000000000000000.0"),
            ("+001.2300", "1.23"),
            ("-0.000", "0.0"),
        ] {
            let value = decimal_value(BigDecimal::from_str(input).unwrap());
            assert!(matches!(
                value,
                Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                    if lexical.as_str() == expected
                        && !lexical.contains(['e', 'E'])
                        && datatype.as_str() == Literal::XSD_DECIMAL
            ));
        }
    }

    #[test]
    fn blocking_order_uses_one_transitive_cross_kind_numeric_domain() {
        let integer = numeric("16777217", Literal::XSD_INTEGER);
        let float = numeric("16777216", XSD_FLOAT);
        let double = numeric("16777217", Literal::XSD_DOUBLE);

        assert_eq!(
            integer.compare_for_order(&float),
            Some(Ordering::Greater),
            "integer ORDER keys must not round through a pairwise f32 promotion"
        );
        assert_eq!(integer.compare_for_order(&double), Some(Ordering::Equal));
        assert_eq!(float.compare_for_order(&double), Some(Ordering::Less));
    }
}
