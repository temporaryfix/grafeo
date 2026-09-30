//! Shared accumulator types for both pull-based and push-based aggregate operators.
//!
//! Provides the canonical definitions of [`AggregateFunction`], [`AggregateExpr`],
//! [`AggregateState`], and [`HashableValue`] used by both `aggregate.rs` (pull)
//! and `push/aggregate.rs`.

// Re-export AggregateState so both pull and push operators import from one place.
pub use super::aggregate::AggregateState;

use grafeo_common::types::Value;

/// Aggregation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AggregateFunction {
    /// Count of rows (COUNT(*)).
    Count,
    /// Count of non-null values (COUNT(column)).
    CountNonNull,
    /// Sum of values.
    Sum,
    /// Average of values.
    Avg,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// First value in the group.
    First,
    /// Last value in the group.
    Last,
    /// Collect values into a list.
    Collect,
    /// Sample standard deviation (STDEV).
    StdDev,
    /// Population standard deviation (STDEVP).
    StdDevPop,
    /// Sample variance (VAR_SAMP / VARIANCE).
    Variance,
    /// Population variance (VAR_POP).
    VariancePop,
    /// Discrete percentile (PERCENTILE_DISC).
    PercentileDisc,
    /// Continuous percentile (PERCENTILE_CONT).
    PercentileCont,
    /// Concatenate values with separator (GROUP_CONCAT).
    GroupConcat,
    /// Return an arbitrary value from the group (SAMPLE).
    Sample,
    /// Sample covariance (COVAR_SAMP(y, x)).
    CovarSamp,
    /// Population covariance (COVAR_POP(y, x)).
    CovarPop,
    /// Pearson correlation coefficient (CORR(y, x)).
    Corr,
    /// Regression slope (REGR_SLOPE(y, x)).
    RegrSlope,
    /// Regression intercept (REGR_INTERCEPT(y, x)).
    RegrIntercept,
    /// Coefficient of determination (REGR_R2(y, x)).
    RegrR2,
    /// Regression count of non-null pairs (REGR_COUNT(y, x)).
    RegrCount,
    /// Regression sum of squares for x (REGR_SXX(y, x)).
    RegrSxx,
    /// Regression sum of squares for y (REGR_SYY(y, x)).
    RegrSyy,
    /// Regression sum of cross-products (REGR_SXY(y, x)).
    RegrSxy,
    /// Regression average of x (REGR_AVGX(y, x)).
    RegrAvgx,
    /// Regression average of y (REGR_AVGY(y, x)).
    RegrAvgy,
}

/// An aggregation expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// The aggregation function.
    pub function: AggregateFunction,
    /// Column index to aggregate (None for COUNT(*), y column for binary set functions).
    pub column: Option<usize>,
    /// Second column index for binary set functions (x column for COVAR, CORR, REGR_*).
    pub column2: Option<usize>,
    /// Optional independent identity column used only for DISTINCT tracking.
    ///
    /// The aggregate function still consumes [`Self::column`]. This channel
    /// lets a planner preserve richer identity semantics without encoding
    /// control data inside a public [`Value`].
    pub distinct_key_column: Option<usize>,
    /// Whether to aggregate distinct values only.
    pub distinct: bool,
    /// Output alias (for naming the result column).
    pub alias: Option<String>,
    /// Percentile parameter for PERCENTILE_DISC/PERCENTILE_CONT (0.0 to 1.0).
    pub percentile: Option<f64>,
    /// Separator string for GROUP_CONCAT / LISTAGG.
    pub separator: Option<String>,
}

impl AggregateExpr {
    /// Creates a COUNT(*) expression.
    pub fn count_star() -> Self {
        Self {
            function: AggregateFunction::Count,
            column: None,
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a COUNT(column) expression.
    pub fn count(column: usize) -> Self {
        Self {
            function: AggregateFunction::CountNonNull,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a SUM(column) expression.
    pub fn sum(column: usize) -> Self {
        Self {
            function: AggregateFunction::Sum,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates an AVG(column) expression.
    pub fn avg(column: usize) -> Self {
        Self {
            function: AggregateFunction::Avg,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a MIN(column) expression.
    pub fn min(column: usize) -> Self {
        Self {
            function: AggregateFunction::Min,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a MAX(column) expression.
    pub fn max(column: usize) -> Self {
        Self {
            function: AggregateFunction::Max,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a FIRST(column) expression.
    pub fn first(column: usize) -> Self {
        Self {
            function: AggregateFunction::First,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a LAST(column) expression.
    pub fn last(column: usize) -> Self {
        Self {
            function: AggregateFunction::Last,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a COLLECT(column) expression.
    pub fn collect(column: usize) -> Self {
        Self {
            function: AggregateFunction::Collect,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a STDEV(column) expression (sample standard deviation).
    pub fn stdev(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDev,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a STDEVP(column) expression (population standard deviation).
    pub fn stdev_pop(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDevPop,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }
    }

    /// Creates a PERCENTILE_DISC(column, percentile) expression.
    ///
    /// # Arguments
    /// * `column` - Column index to aggregate
    /// * `percentile` - Percentile value between 0.0 and 1.0 (e.g., 0.5 for median)
    pub fn percentile_disc(column: usize, percentile: f64) -> Self {
        Self {
            function: AggregateFunction::PercentileDisc,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
            separator: None,
        }
    }

    /// Creates a PERCENTILE_CONT(column, percentile) expression.
    ///
    /// # Arguments
    /// * `column` - Column index to aggregate
    /// * `percentile` - Percentile value between 0.0 and 1.0 (e.g., 0.5 for median)
    pub fn percentile_cont(column: usize, percentile: f64) -> Self {
        Self {
            function: AggregateFunction::PercentileCont,
            column: Some(column),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
            separator: None,
        }
    }

    /// Sets the distinct flag.
    pub fn with_distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Uses a separate column as the identity tracked by DISTINCT aggregates.
    #[must_use]
    pub fn with_distinct_key_column(mut self, column: usize) -> Self {
        self.distinct_key_column = Some(column);
        self
    }

    /// Sets the output alias.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }
}

/// A wrapper for [`Value`] that can be hashed (for DISTINCT tracking).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HashableValue {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int64(i64),
    /// Float as raw bits (for deterministic hashing).
    Float64Bits(u64),
    /// String value.
    String(String),
    /// Fallback for other types (uses Debug representation).
    Other(String),
}

impl From<&Value> for HashableValue {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => HashableValue::Null,
            Value::Bool(b) => HashableValue::Bool(*b),
            Value::Int64(i) => HashableValue::Int64(*i),
            // Canonicalize -0.0 to +0.0 so the two zeros group together.
            Value::Float64(f) => {
                HashableValue::Float64Bits(grafeo_common::types::canonical_f64_bits(*f))
            }
            Value::String(s) => HashableValue::String(s.to_string()),
            other => HashableValue::Other(format!("{other:?}")),
        }
    }
}

impl From<Value> for HashableValue {
    fn from(v: Value) -> Self {
        Self::from(&v)
    }
}

// These declarations inspect built-in Value data only; they never format or
// allocate before the resource caller has admitted their returned peak.
#[cfg(feature = "spill")]
impl HashableValue {
    pub(super) fn retained_heap_bytes(&self) -> usize {
        match self {
            Self::String(value) | Self::Other(value) => value.capacity(),
            Self::Null | Self::Bool(_) | Self::Int64(_) | Self::Float64Bits(_) => 0,
        }
    }

    /// Capacity of the newly retained identity string, excluding formatting
    /// scratch. The byte Vec growth and fmt initial-hint proof below gives
    /// at most max(2*output_length, 8) bytes for its final backing.
    pub(super) fn retained_heap_bound(value: &Value) -> Option<usize> {
        value.retained_size_bytes()?;
        let text = match value {
            Value::Null | Value::Bool(_) | Value::Int64(_) | Value::Float64(_) => return Some(0),
            Value::String(value) => value.len(),
            other => value_debug_text_bound(other, 0)?,
        };
        text.checked_mul(2).map(|bytes| bytes.max(8))
    }

    pub(super) fn construction_peak_bytes(value: &Value) -> Option<usize> {
        // Also enforce the existing bounded traversal/immutable backing proof.
        value.retained_size_bytes()?;
        match value {
            Value::Null | Value::Bool(_) | Value::Int64(_) | Value::Float64(_) => Some(0),
            Value::String(value) => formatted_string_peak(value.len()),
            other => formatted_string_peak(value_debug_text_bound(other, 0)?),
        }
    }
}

/// Upper bound for String formatting, including a moving reallocation and the
/// temporal Display implementations' short fractional-second scratch String.
/// Rust 1.97 RawVec grows to max(2*capacity, required, 8) for bytes. Before a
/// growth, the old capacity is smaller than the eventual output; the new one
/// is at most twice it. fmt::format's initial hint is at most twice its literal
/// bytes. The extra 64 covers the at-most-ten-byte temporal fraction formatter
/// and its old/new buffers. This is allocation capacity, not a length estimate.
#[cfg(feature = "spill")]
pub(super) fn formatted_string_peak(text_bytes: usize) -> Option<usize> {
    text_bytes.max(8).checked_mul(3)?.checked_add(64)
}

/// Nonformatting bound for the exact current Value Debug implementation.
/// A Rust debug string emits at most ten bytes per source UTF-8 byte (the
/// longest Unicode escape), plus delimiters. Fixed-width numeric/temporal
/// values, lengths and counter totals fit in 512 bytes: even f64's nonexponent
/// decimal representation is at most 327, while each temporal value has at
/// most eight bounded integer fields. Container delimiters are counted below.
#[cfg(feature = "spill")]
pub(super) fn value_debug_text_bound(value: &Value, depth: usize) -> Option<usize> {
    if depth > 256 {
        return None;
    }
    match value {
        Value::String(value) => value.len().checked_mul(10)?.checked_add(10),
        Value::List(values) => values.iter().try_fold(8usize, |total, value| {
            total
                .checked_add(value_debug_text_bound(value, depth + 1)?)?
                .checked_add(2)
        }),
        Value::Map(values) => values.iter().try_fold(7usize, |total, (key, value)| {
            total
                .checked_add(key.as_str().len().checked_mul(10)?)?
                .checked_add(16)?
                .checked_add(value_debug_text_bound(value, depth + 1)?)?
                .checked_add(4)
        }),
        Value::RdfLiteral {
            lexical,
            language,
            datatype,
        } => {
            let mut bytes = lexical.len().checked_add(32)?;
            for part in [language, datatype].into_iter().flatten() {
                bytes = bytes.checked_add(part.len())?;
            }
            Some(bytes)
        }
        Value::Null => Some(4),
        Value::Bool(value) => Some(if *value { 10 } else { 11 }),
        Value::Int64(value) => {
            let magnitude = value.unsigned_abs();
            let digits = if magnitude == 0 {
                1
            } else {
                magnitude.ilog10() as usize + 1
            };
            digits.checked_add(usize::from(*value < 0))?.checked_add(7)
        }
        Value::Float64(value) => {
            // Only f64's built-in Display formatter runs here. It writes from
            // stack scratch, unlike the allocating temporal Display paths.
            struct ByteCount(usize);
            impl std::fmt::Write for ByteCount {
                fn write_str(&mut self, text: &str) -> std::fmt::Result {
                    self.0 = self.0.checked_add(text.len()).ok_or(std::fmt::Error)?;
                    Ok(())
                }
            }
            let mut count = ByteCount(0);
            std::fmt::write(&mut count, format_args!("Float64({value})")).ok()?;
            Some(count.0)
        }
        Value::Bytes(_)
        | Value::Timestamp(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_)
        | Value::ZonedDatetime(_)
        | Value::Vector(_)
        | Value::Path { .. }
        | Value::GCounter(_)
        | Value::OnCounter { .. } => Some(512),
        _ => None,
    }
}

#[cfg(all(test, feature = "spill"))]
mod resource_bounds_tests {
    use super::*;
    use grafeo_common::types::{Duration, Time, Timestamp};
    use std::sync::Arc;

    #[test]
    fn distinct_format_bounds_cover_escaped_nested_and_temporal_values() {
        let nested = Value::List(Arc::from([
            Value::String("\u{1}\n\"\\".repeat(129).into()),
            Value::Duration(Duration::from_nanos(123_456_789)),
            Value::Time(Time::from_hms(3, 4, 5).unwrap()),
            Value::Timestamp(Timestamp::from_micros(i64::MAX)),
            Value::Float64(f64::MAX),
            Value::Float64(f64::MIN_POSITIVE),
            Value::Float64(f64::from_bits(1)),
            Value::Int64(i64::MIN),
            Value::Bytes(Arc::from([255u8; 7])),
        ]));
        for value in [
            nested,
            Value::Int64(i64::MIN),
            Value::Float64(f64::NEG_INFINITY),
            Value::Bool(false),
            Value::Null,
            Value::String("escaped\ntext".into()),
        ] {
            let text_bound = value_debug_text_bound(&value, 0).unwrap();
            let hash_bound = HashableValue::retained_heap_bound(&value).unwrap();
            let peak = HashableValue::construction_peak_bytes(&value).unwrap();
            assert!(format!("{value:?}").len() <= text_bound);
            let key = HashableValue::from(&value);
            assert!(key.retained_heap_bytes() <= hash_bound);
            assert!(hash_bound <= peak);
        }
    }
}
