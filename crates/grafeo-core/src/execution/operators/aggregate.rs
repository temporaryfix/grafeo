//! Aggregation operators for GROUP BY and aggregation functions.
//!
//! This module provides:
//! - [`HashAggregateOperator`]: Hash-based grouping with aggregation functions
//! - [`SimpleAggregateOperator`]: Global aggregation without GROUP BY
//!
//! Shared types ([`AggregateFunction`], [`AggregateExpr`], [`HashableValue`]) live in
//! the [`super::accumulator`] module.

use indexmap::IndexMap;
use std::sync::Arc;

use grafeo_common::utils::hash::FxHashSet;

use arcstr::ArcStr;
use grafeo_common::types::{LogicalType, PropertyKey, Value};

use super::accumulator::{AggregateExpr, AggregateFunction, HashableValue};
use super::{Operator, OperatorError, OperatorPipelineDecomposition, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// State for a single aggregation computation.
///
/// Used by both the pull-based [`HashAggregateOperator`] and the push-based
/// `AggregatePushOperator`.
/// Supports all [`AggregateFunction`] variants including Welford's algorithm
/// for online statistics, Kahan summation, distinct tracking, and bivariate
/// regression functions.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum AggregateState {
    /// Count state.
    Count(i64),
    /// Count distinct state (count, seen values).
    CountDistinct(i64, FxHashSet<HashableValue>),
    /// Sum state (integer sum, count of values added).
    SumInt(i64, i64),
    /// Sum distinct state (integer sum, count, seen values).
    SumIntDistinct(i64, i64, FxHashSet<HashableValue>),
    /// Sum state (float sum, count of values added).
    SumFloat(f64, f64, i64),
    /// Sum distinct state (float sum, compensation, count, seen values).
    SumFloatDistinct(f64, f64, i64, FxHashSet<HashableValue>),
    /// Average state (sum, count).
    Avg(f64, i64),
    /// Average distinct state (sum, count, seen values).
    AvgDistinct(f64, i64, FxHashSet<HashableValue>),
    /// Min state.
    Min(Option<Value>),
    /// Minimum over the first non-null operand admitted for each identity.
    MinDistinct(Option<Value>, FxHashSet<HashableValue>),
    /// Max state.
    Max(Option<Value>),
    /// Maximum over the first non-null operand admitted for each identity.
    MaxDistinct(Option<Value>, FxHashSet<HashableValue>),
    /// First state.
    First(Option<Value>),
    /// Last state.
    Last(Option<Value>),
    /// Last operand admitted with a previously unseen identity.
    LastDistinct(Option<Value>, FxHashSet<HashableValue>),
    /// Collect state.
    Collect(Vec<Value>),
    /// Collect distinct state (values, seen).
    CollectDistinct(Vec<Value>, FxHashSet<HashableValue>),
    /// Sample standard deviation state using Welford's algorithm (count, mean, M2).
    StdDev {
        count: i64,
        mean: f64,
        m2: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// Population standard deviation state using Welford's algorithm (count, mean, M2).
    StdDevPop {
        count: i64,
        mean: f64,
        m2: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// Discrete percentile state (values, percentile).
    PercentileDisc {
        values: Vec<f64>,
        percentile: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// Continuous percentile state (values, percentile).
    PercentileCont {
        values: Vec<f64>,
        percentile: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// GROUP_CONCAT / LISTAGG state (collected string values, separator).
    GroupConcat(Vec<String>, String),
    /// GROUP_CONCAT / LISTAGG distinct state (collected string values, separator, seen).
    GroupConcatDistinct(Vec<String>, String, FxHashSet<HashableValue>),
    /// SAMPLE state (first non-null value encountered).
    Sample(Option<Value>),
    /// Sample variance state using Welford's algorithm (count, mean, M2).
    Variance {
        count: i64,
        mean: f64,
        m2: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// Population variance state using Welford's algorithm (count, mean, M2).
    VariancePop {
        count: i64,
        mean: f64,
        m2: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// Two-variable online statistics (Welford generalization for covariance/regression).
    Bivariate {
        /// Which binary set function this state will finalize to.
        kind: AggregateFunction,
        count: i64,
        mean_x: f64,
        mean_y: f64,
        m2_x: f64,
        m2_y: f64,
        c_xy: f64,
        seen: Option<FxHashSet<HashableValue>>,
    },
    /// Explicit terminal value. Live accumulators never use this representation
    /// during spill; only an already Frozen value remains terminal on reload.
    Frozen(Value),
}

impl AggregateState {
    /// Creates initial state for an aggregation function.
    pub fn new(
        function: AggregateFunction,
        distinct: bool,
        percentile: Option<f64>,
        separator: Option<&str>,
    ) -> Self {
        match (function, distinct) {
            (AggregateFunction::Count | AggregateFunction::CountNonNull, false) => {
                AggregateState::Count(0)
            }
            (AggregateFunction::Count | AggregateFunction::CountNonNull, true) => {
                AggregateState::CountDistinct(0, FxHashSet::default())
            }
            (AggregateFunction::Sum, false) => AggregateState::SumInt(0, 0),
            (AggregateFunction::Sum, true) => {
                AggregateState::SumIntDistinct(0, 0, FxHashSet::default())
            }
            (AggregateFunction::Avg, false) => AggregateState::Avg(0.0, 0),
            (AggregateFunction::Avg, true) => {
                AggregateState::AvgDistinct(0.0, 0, FxHashSet::default())
            }
            (AggregateFunction::Min, false) => AggregateState::Min(None),
            (AggregateFunction::Min, true) => {
                AggregateState::MinDistinct(None, FxHashSet::default())
            }
            (AggregateFunction::Max, false) => AggregateState::Max(None),
            (AggregateFunction::Max, true) => {
                AggregateState::MaxDistinct(None, FxHashSet::default())
            }
            (AggregateFunction::First, _) => AggregateState::First(None),
            (AggregateFunction::Last, false) => AggregateState::Last(None),
            (AggregateFunction::Last, true) => {
                AggregateState::LastDistinct(None, FxHashSet::default())
            }
            (AggregateFunction::Collect, false) => AggregateState::Collect(Vec::new()),
            (AggregateFunction::Collect, true) => {
                AggregateState::CollectDistinct(Vec::new(), FxHashSet::default())
            }
            // Statistical functions (Welford's algorithm for online computation)
            (AggregateFunction::StdDev, _) => AggregateState::StdDev {
                count: 0,
                mean: 0.0,
                m2: 0.0,
                seen: distinct.then(FxHashSet::default),
            },
            (AggregateFunction::StdDevPop, _) => AggregateState::StdDevPop {
                count: 0,
                mean: 0.0,
                m2: 0.0,
                seen: distinct.then(FxHashSet::default),
            },
            (AggregateFunction::PercentileDisc, _) => AggregateState::PercentileDisc {
                values: Vec::new(),
                percentile: percentile.unwrap_or(0.5),
                seen: distinct.then(FxHashSet::default),
            },
            (AggregateFunction::PercentileCont, _) => AggregateState::PercentileCont {
                values: Vec::new(),
                percentile: percentile.unwrap_or(0.5),
                seen: distinct.then(FxHashSet::default),
            },
            (AggregateFunction::GroupConcat, false) => {
                AggregateState::GroupConcat(Vec::new(), separator.unwrap_or(" ").to_string())
            }
            (AggregateFunction::GroupConcat, true) => AggregateState::GroupConcatDistinct(
                Vec::new(),
                separator.unwrap_or(" ").to_string(),
                FxHashSet::default(),
            ),
            (AggregateFunction::Sample, _) => AggregateState::Sample(None),
            // Binary set functions (all share the same Bivariate state)
            (
                AggregateFunction::CovarSamp
                | AggregateFunction::CovarPop
                | AggregateFunction::Corr
                | AggregateFunction::RegrSlope
                | AggregateFunction::RegrIntercept
                | AggregateFunction::RegrR2
                | AggregateFunction::RegrCount
                | AggregateFunction::RegrSxx
                | AggregateFunction::RegrSyy
                | AggregateFunction::RegrSxy
                | AggregateFunction::RegrAvgx
                | AggregateFunction::RegrAvgy,
                _,
            ) => AggregateState::Bivariate {
                kind: function,
                count: 0,
                mean_x: 0.0,
                mean_y: 0.0,
                m2_x: 0.0,
                m2_y: 0.0,
                c_xy: 0.0,
                seen: distinct.then(FxHashSet::default),
            },
            (AggregateFunction::Variance, _) => AggregateState::Variance {
                count: 0,
                mean: 0.0,
                m2: 0.0,
                seen: distinct.then(FxHashSet::default),
            },
            (AggregateFunction::VariancePop, _) => AggregateState::VariancePop {
                count: 0,
                mean: 0.0,
                m2: 0.0,
                seen: distinct.then(FxHashSet::default),
            },
        }
    }

    /// Updates the state with a new value.
    ///
    /// For `COUNT(*)`, pass `None` to count all rows. For column-specific
    /// aggregates, pass `Some(value)` (nulls are skipped by most functions).
    pub fn update(&mut self, value: Option<Value>) {
        self.update_with_distinct_key(value, None);
    }

    /// Updates the state while optionally tracking a separate DISTINCT key.
    ///
    /// The key controls duplicate elimination only. The aggregate function
    /// always consumes `value`. Passing no key retains ordinary value-based
    /// DISTINCT behavior.
    pub fn update_with_distinct_key(&mut self, value: Option<Value>, distinct_key: Option<Value>) {
        match self {
            AggregateState::Count(count) => {
                *count += 1;
            }
            AggregateState::CountDistinct(count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(distinct_key.as_ref().unwrap_or(v));
                    if seen.insert(hashable) {
                        *count += 1;
                    }
                }
            }
            AggregateState::SumInt(sum, count) => {
                if let Some(Value::Int64(v)) = value {
                    *sum += v;
                    *count += 1;
                } else if let Some(Value::Float64(v)) = value {
                    // Convert to float sum, carrying count forward
                    *self = AggregateState::SumFloat(*sum as f64 + v, 0.0, *count + 1);
                } else if let Some(ref v) = value {
                    // RDF stores numeric literals as strings - try to parse
                    if let Some(num) = value_to_f64(v) {
                        *self = AggregateState::SumFloat(*sum as f64 + num, 0.0, *count + 1);
                    }
                }
            }
            AggregateState::SumIntDistinct(sum, count, seen) => {
                if let Some(ref v) = value
                    && let Some(num) = value_to_f64(v)
                {
                    let hashable = HashableValue::from(distinct_key.as_ref().unwrap_or(v));
                    if seen.insert(hashable) {
                        if let Value::Int64(i) = v {
                            *sum += i;
                            *count += 1;
                        } else {
                            // Convert to float distinct: move the seen set instead of cloning
                            let moved_seen = std::mem::take(seen);
                            *self = AggregateState::SumFloatDistinct(
                                *sum as f64 + num,
                                0.0,
                                *count + 1,
                                moved_seen,
                            );
                        }
                    }
                }
            }
            AggregateState::SumFloat(sum, comp, count) => {
                if let Some(ref v) = value {
                    // Use value_to_f64 which now handles strings
                    if let Some(num) = value_to_f64(v) {
                        // Kahan compensated summation to reduce rounding error
                        let y = num - *comp;
                        let t = *sum + y;
                        *comp = (t - *sum) - y;
                        *sum = t;
                        *count += 1;
                    }
                }
            }
            AggregateState::SumFloatDistinct(sum, comp, count, seen) => {
                if let Some(ref v) = value
                    && let Some(num) = value_to_f64(v)
                {
                    let hashable = HashableValue::from(distinct_key.as_ref().unwrap_or(v));
                    if seen.insert(hashable) {
                        let y = num - *comp;
                        let t = *sum + y;
                        *comp = (t - *sum) - y;
                        *sum = t;
                        *count += 1;
                    }
                }
            }
            AggregateState::Avg(sum, count) => {
                if let Some(ref v) = value
                    && let Some(num) = value_to_f64(v)
                {
                    *sum += num;
                    *count += 1;
                }
            }
            AggregateState::AvgDistinct(sum, count, seen) => {
                if let Some(ref v) = value
                    && let Some(num) = value_to_f64(v)
                {
                    let hashable = HashableValue::from(distinct_key.as_ref().unwrap_or(v));
                    if seen.insert(hashable) {
                        *sum += num;
                        *count += 1;
                    }
                }
            }
            AggregateState::Min(min) => {
                if let Some(v) = value {
                    match min {
                        None => *min = Some(v),
                        Some(current) => {
                            if compare_values(&v, current) == Some(std::cmp::Ordering::Less) {
                                *min = Some(v);
                            }
                        }
                    }
                }
            }
            AggregateState::Max(max) => {
                if let Some(v) = value {
                    match max {
                        None => *max = Some(v),
                        Some(current) => {
                            if compare_values(&v, current) == Some(std::cmp::Ordering::Greater) {
                                *max = Some(v);
                            }
                        }
                    }
                }
            }
            AggregateState::MinDistinct(min, seen) => {
                if let Some(v) = value
                    && !matches!(v, Value::Null)
                    && seen.insert(HashableValue::from(distinct_key.as_ref().unwrap_or(&v)))
                    && min.as_ref().is_none_or(|current| {
                        compare_values(&v, current) == Some(std::cmp::Ordering::Less)
                    })
                {
                    *min = Some(v);
                }
            }
            AggregateState::MaxDistinct(max, seen) => {
                if let Some(v) = value
                    && !matches!(v, Value::Null)
                    && seen.insert(HashableValue::from(distinct_key.as_ref().unwrap_or(&v)))
                    && max.as_ref().is_none_or(|current| {
                        compare_values(&v, current) == Some(std::cmp::Ordering::Greater)
                    })
                {
                    *max = Some(v);
                }
            }
            AggregateState::First(first) => {
                if first.is_none() {
                    *first = value;
                }
            }
            AggregateState::Last(last) => {
                if value.is_some() {
                    *last = value;
                }
            }
            AggregateState::LastDistinct(last, seen) => {
                if let Some(v) = value
                    && seen.insert(HashableValue::from(distinct_key.as_ref().unwrap_or(&v)))
                {
                    *last = Some(v);
                }
            }
            AggregateState::Collect(list) => {
                if let Some(v) = value {
                    list.push(v);
                }
            }
            AggregateState::CollectDistinct(list, seen) => {
                if let Some(v) = value {
                    let hashable = HashableValue::from(distinct_key.as_ref().unwrap_or(&v));
                    if seen.insert(hashable) {
                        list.push(v);
                    }
                }
            }
            // Statistical functions using Welford's online algorithm
            AggregateState::StdDev {
                count,
                mean,
                m2,
                seen,
            }
            | AggregateState::StdDevPop {
                count,
                mean,
                m2,
                seen,
            }
            | AggregateState::Variance {
                count,
                mean,
                m2,
                seen,
            }
            | AggregateState::VariancePop {
                count,
                mean,
                m2,
                seen,
            } => {
                if let Some(ref v) = value
                    && let Some(x) = value_to_f64(v)
                    && seen.as_mut().is_none_or(|seen| {
                        seen.insert(HashableValue::from(distinct_key.as_ref().unwrap_or(v)))
                    })
                {
                    *count += 1;
                    let delta = x - *mean;
                    *mean += delta / *count as f64;
                    let delta2 = x - *mean;
                    *m2 += delta * delta2;
                }
            }
            AggregateState::PercentileDisc { values, seen, .. }
            | AggregateState::PercentileCont { values, seen, .. } => {
                if let Some(ref v) = value
                    && let Some(x) = value_to_f64(v)
                    && seen.as_mut().is_none_or(|seen| {
                        seen.insert(HashableValue::from(distinct_key.as_ref().unwrap_or(v)))
                    })
                {
                    values.push(x);
                }
            }
            AggregateState::GroupConcat(list, _sep) => {
                if let Some(v) = value {
                    list.push(agg_value_to_string(&v));
                }
            }
            AggregateState::GroupConcatDistinct(list, _sep, seen) => {
                if let Some(v) = value {
                    let hashable = HashableValue::from(distinct_key.as_ref().unwrap_or(&v));
                    if seen.insert(hashable) {
                        list.push(agg_value_to_string(&v));
                    }
                }
            }
            AggregateState::Sample(sample) => {
                if sample.is_none() {
                    *sample = value;
                }
            }
            AggregateState::Bivariate { .. } => {
                // Bivariate functions require two values; use update_bivariate() instead.
                // Single-value update is a no-op for bivariate state.
            }
            AggregateState::Frozen(_) => {}
        }
    }

    /// Updates a bivariate (two-variable) aggregate state with a pair of values.
    ///
    /// Uses the two-variable Welford online algorithm for numerically stable computation
    /// of covariance and related statistics. Skips the update if either value is null.
    pub fn update_bivariate(&mut self, y_val: Option<Value>, x_val: Option<Value>) {
        self.update_bivariate_with_distinct_key(y_val, x_val, None);
    }

    /// Updates a pair, using an explicit DISTINCT identity when supplied and
    /// the ordered `(y, x)` value pair otherwise. Invalid pairs consume no key.
    pub fn update_bivariate_with_distinct_key(
        &mut self,
        y_val: Option<Value>,
        x_val: Option<Value>,
        distinct_key: Option<Value>,
    ) {
        if let AggregateState::Bivariate {
            count,
            mean_x,
            mean_y,
            m2_x,
            m2_y,
            c_xy,
            seen,
            ..
        } = self
        {
            // Skip if either value is null (SQL semantics: exclude non-pairs)
            if let (Some(y), Some(x)) = (&y_val, &x_val)
                && let (Some(y_f), Some(x_f)) = (value_to_f64(y), value_to_f64(x))
            {
                if let Some(seen) = seen {
                    let key = distinct_key
                        .unwrap_or_else(|| Value::List(Arc::from([y.clone(), x.clone()])));
                    if !seen.insert(HashableValue::from(key)) {
                        return;
                    }
                }
                *count += 1;
                let n = *count as f64;
                let dx = x_f - *mean_x;
                let dy = y_f - *mean_y;
                *mean_x += dx / n;
                *mean_y += dy / n;
                let dx2 = x_f - *mean_x; // post-update delta
                let dy2 = y_f - *mean_y; // post-update delta
                *m2_x += dx * dx2;
                *m2_y += dy * dy2;
                *c_xy += dx * dy2;
            }
        }
    }

    /// Finalizes the state and returns the result value.
    pub fn finalize(&self) -> Value {
        match self {
            AggregateState::Count(count) | AggregateState::CountDistinct(count, _) => {
                Value::Int64(*count)
            }
            AggregateState::SumInt(sum, count) | AggregateState::SumIntDistinct(sum, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Int64(*sum)
                }
            }
            AggregateState::SumFloat(sum, _, count)
            | AggregateState::SumFloatDistinct(sum, _, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*sum)
                }
            }
            AggregateState::Avg(sum, count) | AggregateState::AvgDistinct(sum, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*sum / *count as f64)
                }
            }
            AggregateState::Min(min) | AggregateState::MinDistinct(min, _) => {
                min.clone().unwrap_or(Value::Null)
            }
            AggregateState::Max(max) | AggregateState::MaxDistinct(max, _) => {
                max.clone().unwrap_or(Value::Null)
            }
            AggregateState::First(first) => first.clone().unwrap_or(Value::Null),
            AggregateState::Last(last) | AggregateState::LastDistinct(last, _) => {
                last.clone().unwrap_or(Value::Null)
            }
            AggregateState::Collect(list) | AggregateState::CollectDistinct(list, _) => {
                Value::List(list.clone().into())
            }
            // Sample standard deviation: sqrt(M2 / (n - 1))
            AggregateState::StdDev { count, m2, .. } => {
                if *count < 2 {
                    Value::Null
                } else {
                    Value::Float64((*m2 / (*count - 1) as f64).sqrt())
                }
            }
            // Population standard deviation: sqrt(M2 / n)
            AggregateState::StdDevPop { count, m2, .. } => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64((*m2 / *count as f64).sqrt())
                }
            }
            // Sample variance: M2 / (n - 1)
            AggregateState::Variance { count, m2, .. } => {
                if *count < 2 {
                    Value::Null
                } else {
                    Value::Float64(*m2 / (*count - 1) as f64)
                }
            }
            // Population variance: M2 / n
            AggregateState::VariancePop { count, m2, .. } => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*m2 / *count as f64)
                }
            }
            // Discrete percentile: return actual value at percentile position
            AggregateState::PercentileDisc {
                values, percentile, ..
            } => {
                if values.is_empty() {
                    Value::Null
                } else {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    // Index calculation per SQL standard: floor(p * (n - 1))
                    // reason: percentile index is bounded by sorted.len(), fits usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let index = (percentile * (sorted.len() - 1) as f64).floor() as usize;
                    Value::Float64(sorted[index])
                }
            }
            // Continuous percentile: interpolate between values
            AggregateState::PercentileCont {
                values, percentile, ..
            } => {
                if values.is_empty() {
                    Value::Null
                } else {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    // Linear interpolation per SQL standard
                    let rank = percentile * (sorted.len() - 1) as f64;
                    // reason: rank is bounded by sorted.len() - 1, fits usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let lower_idx = rank.floor() as usize;
                    // reason: rank is a non-negative f64 from percentile calculation, fits usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let upper_idx = rank.ceil() as usize;
                    if lower_idx == upper_idx {
                        Value::Float64(sorted[lower_idx])
                    } else {
                        let fraction = rank - lower_idx as f64;
                        let result =
                            sorted[lower_idx] + fraction * (sorted[upper_idx] - sorted[lower_idx]);
                        Value::Float64(result)
                    }
                }
            }
            // GROUP_CONCAT: join strings with space separator (SPARQL default)
            AggregateState::GroupConcat(list, sep)
            | AggregateState::GroupConcatDistinct(list, sep, _) => {
                Value::String(list.join(sep).into())
            }
            // SAMPLE: return the first non-null value seen
            AggregateState::Sample(sample) => sample.clone().unwrap_or(Value::Null),
            AggregateState::Frozen(val) => val.clone(),
            // Binary set functions: dispatch on kind
            AggregateState::Bivariate {
                kind,
                count,
                mean_x,
                mean_y,
                m2_x,
                m2_y,
                c_xy,
                ..
            } => {
                let n = *count;
                match kind {
                    AggregateFunction::CovarSamp => {
                        if n < 2 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / (n - 1) as f64)
                        }
                    }
                    AggregateFunction::CovarPop => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / n as f64)
                        }
                    }
                    AggregateFunction::Corr => {
                        if n == 0 || *m2_x == 0.0 || *m2_y == 0.0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / (*m2_x * *m2_y).sqrt())
                        }
                    }
                    AggregateFunction::RegrSlope => {
                        if n == 0 || *m2_x == 0.0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / *m2_x)
                        }
                    }
                    AggregateFunction::RegrIntercept => {
                        if n == 0 || *m2_x == 0.0 {
                            Value::Null
                        } else {
                            let slope = *c_xy / *m2_x;
                            Value::Float64(*mean_y - slope * *mean_x)
                        }
                    }
                    AggregateFunction::RegrR2 => {
                        if n == 0 || *m2_x == 0.0 || *m2_y == 0.0 {
                            Value::Null
                        } else {
                            Value::Float64((*c_xy * *c_xy) / (*m2_x * *m2_y))
                        }
                    }
                    AggregateFunction::RegrCount => Value::Int64(n),
                    AggregateFunction::RegrSxx => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*m2_x)
                        }
                    }
                    AggregateFunction::RegrSyy => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*m2_y)
                        }
                    }
                    AggregateFunction::RegrSxy => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy)
                        }
                    }
                    AggregateFunction::RegrAvgx => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*mean_x)
                        }
                    }
                    AggregateFunction::RegrAvgy => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*mean_y)
                        }
                    }
                    _ => Value::Null, // non-bivariate functions never reach here
                }
            }
        }
    }
}

/// Direct allocation measurements before a single admitted update. This
/// contains no borrowed/owned accumulator data and allocates no metadata.
#[cfg(feature = "spill")]
#[derive(Clone, Copy)]
pub(crate) struct AggregateRetainedSnapshot {
    direct_and_single_bytes: usize,
    values_len: usize,
    distinct_len: usize,
}

/// Resource admission declarations for the existing accumulator operations.
/// The state and Value graphs expose no interior mutation through a shared
/// reference. Declarations never clone, format, reserve or mutate their inputs.
#[cfg(feature = "spill")]
impl AggregateState {
    /// Checked retained heap authority, excluding the inline AggregateState.
    /// Shared Value allocations are charged per ownership edge using the
    /// existing Value layout proof, even when another owner also charges them.
    pub(crate) fn retained_heap_bytes(&self) -> Option<usize> {
        let mut bytes = match self {
            Self::Min(value)
            | Self::MinDistinct(value, _)
            | Self::Max(value)
            | Self::MaxDistinct(value, _)
            | Self::First(value)
            | Self::Last(value)
            | Self::LastDistinct(value, _)
            | Self::Sample(value) => value.as_ref().map_or(Some(0), aggregate_value_heap_bytes)?,
            Self::Frozen(value) => aggregate_value_heap_bytes(value)?,
            Self::Collect(values) | Self::CollectDistinct(values, _) => values.iter().try_fold(
                values.capacity().checked_mul(size_of::<Value>())?,
                |total, value| total.checked_add(aggregate_value_heap_bytes(value)?),
            )?,
            Self::PercentileDisc { values, .. } | Self::PercentileCont { values, .. } => {
                values.capacity().checked_mul(size_of::<f64>())?
            }
            Self::GroupConcat(values, separator)
            | Self::GroupConcatDistinct(values, separator, _) => values.iter().try_fold(
                values
                    .capacity()
                    .checked_mul(size_of::<String>())?
                    .checked_add(separator.capacity())?,
                |total, value| total.checked_add(value.capacity()),
            )?,
            Self::Count(_)
            | Self::CountDistinct(_, _)
            | Self::SumInt(_, _)
            | Self::SumIntDistinct(_, _, _)
            | Self::SumFloat(_, _, _)
            | Self::SumFloatDistinct(_, _, _, _)
            | Self::Avg(_, _)
            | Self::AvgDistinct(_, _, _)
            | Self::StdDev { .. }
            | Self::StdDevPop { .. }
            | Self::Variance { .. }
            | Self::VariancePop { .. }
            | Self::Bivariate { .. } => 0,
        };
        if let Some(seen) = self.distinct_values() {
            bytes = seen.iter().try_fold(
                bytes.checked_add(seen.allocation_size())?,
                |total, value| total.checked_add(value.retained_heap_bytes()),
            )?;
        }
        Some(bytes)
    }

    /// Records direct backing capacities and the one optional Value payload.
    /// Does not traverse retained collection entries or DISTINCT identities.
    pub(crate) fn retained_update_snapshot(&self) -> Option<AggregateRetainedSnapshot> {
        let (mut direct, values_len) = match self {
            Self::Min(value)
            | Self::MinDistinct(value, _)
            | Self::Max(value)
            | Self::MaxDistinct(value, _)
            | Self::First(value)
            | Self::Last(value)
            | Self::LastDistinct(value, _)
            | Self::Sample(value) => (
                value.as_ref().map_or(Some(0), aggregate_value_heap_bytes)?,
                0,
            ),
            Self::Frozen(value) => (aggregate_value_heap_bytes(value)?, 0),
            Self::Collect(values) | Self::CollectDistinct(values, _) => (
                values.capacity().checked_mul(size_of::<Value>())?,
                values.len(),
            ),
            Self::PercentileDisc { values, .. } | Self::PercentileCont { values, .. } => (
                values.capacity().checked_mul(size_of::<f64>())?,
                values.len(),
            ),
            Self::GroupConcat(values, separator)
            | Self::GroupConcatDistinct(values, separator, _) => (
                values
                    .capacity()
                    .checked_mul(size_of::<String>())?
                    .checked_add(separator.capacity())?,
                values.len(),
            ),
            Self::Count(_)
            | Self::CountDistinct(_, _)
            | Self::SumInt(_, _)
            | Self::SumIntDistinct(_, _, _)
            | Self::SumFloat(_, _, _)
            | Self::SumFloatDistinct(_, _, _, _)
            | Self::Avg(_, _)
            | Self::AvgDistinct(_, _, _)
            | Self::StdDev { .. }
            | Self::StdDevPop { .. }
            | Self::Variance { .. }
            | Self::VariancePop { .. }
            | Self::Bivariate { .. } => (0, 0),
        };
        let distinct_len = if let Some(seen) = self.distinct_values() {
            direct = direct.checked_add(seen.allocation_size())?;
            seen.len()
        } else {
            0
        };
        Some(AggregateRetainedSnapshot {
            direct_and_single_bytes: direct,
            values_len,
            distinct_len,
        })
    }

    /// Reconciles one successful ordinary update without rescanning old
    /// collection members. Apply `(cached - removed) + added`, checked, while
    /// the full update peak remains admitted. Container backing uses actual
    /// observed capacities. Newly retained Values use the existing nested
    /// bound; the one new DISTINCT identity uses the proven formatted String
    /// capacity bound. Repeated identities add no charge. Numeric promotion
    /// moves its set and preserves these measurements.
    pub(crate) fn retained_update_delta(
        &self,
        before: AggregateRetainedSnapshot,
        expr: &AggregateExpr,
        value: Option<&Value>,
        second: Option<&Value>,
        distinct_key: Option<&Value>,
    ) -> Option<(usize, usize)> {
        let after = self.retained_update_snapshot()?;
        if after.values_len.checked_sub(before.values_len)? > 1
            || after.distinct_len.checked_sub(before.distinct_len)? > 1
        {
            return None;
        }
        let mut added = after.direct_and_single_bytes;
        if after.values_len != before.values_len {
            let tail = match self {
                Self::Collect(values) | Self::CollectDistinct(values, _) => {
                    aggregate_value_heap_bytes(values.last()?)?
                }
                Self::GroupConcat(values, _) | Self::GroupConcatDistinct(values, _, _) => {
                    values.last()?.capacity()
                }
                _ => 0, // Percentile elements are inline f64 slots.
            };
            added = added.checked_add(tail)?;
        }
        if after.distinct_len != before.distinct_len {
            let key_bytes = if distinct_key.is_none() && expr.column2.is_some() {
                let (Some(y), Some(x)) = (value, second) else {
                    return None;
                };
                super::accumulator::value_debug_text_bound(y, 0)?
                    .checked_add(super::accumulator::value_debug_text_bound(x, 0)?)?
                    .checked_add(12)?
                    .checked_mul(2)?
                    .max(8)
            } else {
                HashableValue::retained_heap_bound(distinct_key.or(value)?)?
            };
            added = added.checked_add(key_bytes)?;
        }
        Some((before.direct_and_single_bytes, added))
    }

    /// Peak for cloning an existing state (or constructing a new one), then
    /// calling the ordinary single/bivariate update with cloned inputs.
    /// Includes the resulting retained heap and transient old/new backing
    /// overlap. The caller separately admits inline state/container slots and
    /// continues to own the original state under its original grant.
    pub(crate) fn replacement_peak_bytes(
        previous: Option<&Self>,
        expr: &AggregateExpr,
        value: Option<&Value>,
        second: Option<&Value>,
        distinct_key: Option<&Value>,
    ) -> Option<usize> {
        let initial = match previous {
            Some(previous) => previous.retained_heap_bytes()?,
            None if expr.function == AggregateFunction::GroupConcat => {
                super::accumulator::formatted_string_peak(
                    expr.separator.as_deref().unwrap_or(" ").len(),
                )?
            }
            None => 0,
        };
        initial.checked_add(Self::update_storage_peak_bytes(
            previous,
            expr,
            value,
            second,
            distinct_key,
            true,
        )?)
    }

    /// Peak additional authority for one in-place update. The old state's
    /// retained heap remains admitted separately; spare Vec/set capacity is
    /// reused. No clone or mutation occurs while computing this declaration.
    pub(crate) fn update_peak_bytes(
        &self,
        expr: &AggregateExpr,
        value: Option<&Value>,
        second: Option<&Value>,
        distinct_key: Option<&Value>,
    ) -> Option<usize> {
        Self::update_storage_peak_bytes(Some(self), expr, value, second, distinct_key, false)
    }

    fn update_storage_peak_bytes(
        previous: Option<&Self>,
        expr: &AggregateExpr,
        value: Option<&Value>,
        second: Option<&Value>,
        distinct_key: Option<&Value>,
        cloned: bool,
    ) -> Option<usize> {
        use super::accumulator::{formatted_string_peak, value_debug_text_bound};
        let mut peak = 0usize;
        for input in [value, second, distinct_key].into_iter().flatten() {
            // Cloning a Value shares immutable backing; charge its entire
            // ownership edge before that clone can outlive the input chunk.
            peak = peak.checked_add(input.retained_size_bytes()?)?;
        }
        #[cfg(feature = "triple-store")]
        if let (
            Some(
                Self::Min(Some(current))
                | Self::MinDistinct(Some(current), _)
                | Self::Max(Some(current))
                | Self::MaxDistinct(Some(current), _),
            ),
            Some(value),
        ) = (previous, value)
        {
            // compare_values parses internal RDF identities, retaining both
            // parsed Terms while recursively comparing visible literals.
            peak = peak
                .checked_add(aggregate_rdf_comparison_peak(current, 0)?)?
                .checked_add(aggregate_rdf_comparison_peak(value, 0)?)?;
        }
        let seen = previous.and_then(Self::distinct_values);
        if seen.is_some() || expr.distinct {
            // hashbrown 0.17.1 RawTable::clone preserves the bucket count,
            // including sparse capacity. reserve_rehash_inner for one insert
            // either rehashes in place or at most doubles it. Both tables can
            // coexist; the old clone is in retained_heap_bytes above. The
            // public allocation_size includes alignment and control bytes.
            // allocator-api2 0.2.21 Global returns the requested length, so no
            // oversized-allocation bucket expansion occurs. Initial minimum
            // is four HashableValue buckets plus the maximal 16-byte group.
            let table = seen.map_or(0, FxHashSet::allocation_size);
            let new_table = if seen.is_some_and(|seen| seen.len() < seen.capacity()) {
                0
            } else if table == 0 {
                size_of::<HashableValue>()
                    .checked_add(1)?
                    .checked_mul(4)?
                    .checked_add(16)?
            } else {
                table.checked_mul(2)?
            };
            peak = peak.checked_add(new_table)?;
            if let Some(key) = distinct_key.or(value) {
                peak = peak.checked_add(HashableValue::construction_peak_bytes(key)?)?;
            }
            if distinct_key.is_none()
                && expr.column2.is_some()
                && let (Some(y), Some(x)) = (value, second)
            {
                // The ordinary bivariate update builds an Arc<[Value; 2]>
                // and then formats that pair as the DISTINCT identity.
                let pair = size_of::<usize>()
                    .checked_mul(2)?
                    .checked_add(align_of::<Value>() - 1)?
                    .checked_add(y.retained_size_bytes()?)?
                    .checked_add(x.retained_size_bytes()?)?;
                let text = value_debug_text_bound(y, 0)?
                    .checked_add(value_debug_text_bound(x, 0)?)?
                    .checked_add(12)?;
                peak = peak
                    .checked_add(pair)?
                    .checked_add(formatted_string_peak(text)?)?;
            }
        }
        let vector = match previous {
            Some(Self::Collect(values) | Self::CollectDistinct(values, _)) => {
                Some((values.len(), values.capacity(), size_of::<Value>()))
            }
            Some(Self::PercentileDisc { values, .. } | Self::PercentileCont { values, .. }) => {
                Some((values.len(), values.capacity(), size_of::<f64>()))
            }
            Some(Self::GroupConcat(values, _) | Self::GroupConcatDistinct(values, _, _)) => {
                Some((values.len(), values.capacity(), size_of::<String>()))
            }
            None => match expr.function {
                AggregateFunction::Collect => Some((0, 0, size_of::<Value>())),
                AggregateFunction::PercentileDisc | AggregateFunction::PercentileCont => {
                    Some((0, 0, size_of::<f64>()))
                }
                AggregateFunction::GroupConcat => Some((0, 0, size_of::<String>())),
                _ => None,
            },
            _ => None,
        };
        if let Some((len, capacity, slot)) = vector {
            let capacity = if cloned { len } else { capacity };
            // Rust 1.97 Vec::clone allocates len slots, no more than the
            // observed capacity. One push grows to max(2*cap, len+1, 4) for
            // each of these element types. Charge a whole new backing while
            // retaining the old cloned backing, including an allocator move.
            if len == capacity {
                let grown = capacity.checked_mul(2)?.max(len.checked_add(1)?).max(4);
                peak = peak.checked_add(grown.checked_mul(slot)?)?;
            }
        }
        if matches!(
            previous,
            Some(Self::GroupConcat(..) | Self::GroupConcatDistinct(..))
        ) || (previous.is_none() && expr.function == AggregateFunction::GroupConcat)
        {
            let text = match value {
                Some(Value::String(value)) => value.len(),
                Some(Value::Int64(_)) => 20,
                Some(Value::Float64(_)) => 327,
                Some(Value::Bool(_)) => 5,
                None | Some(Value::Null) => 0,
                Some(value) => value_debug_text_bound(value, 0)?,
            };
            peak = peak.checked_add(formatted_string_peak(text)?)?;
        }
        Some(peak)
    }

    /// Peak extra heap for the existing finalize implementation, including
    /// its returned Value's retained backing. Inline output slots are owned
    /// by the caller. The original state remains admitted throughout.
    pub(crate) fn finalize_peak_bytes(&self) -> Option<usize> {
        let arc_header = size_of::<usize>().checked_mul(2)?;
        match self {
            Self::Min(value)
            | Self::MinDistinct(value, _)
            | Self::Max(value)
            | Self::MaxDistinct(value, _)
            | Self::First(value)
            | Self::Last(value)
            | Self::LastDistinct(value, _)
            | Self::Sample(value) => value.as_ref().map_or(Some(0), aggregate_value_heap_bytes),
            Self::Frozen(value) => aggregate_value_heap_bytes(value),
            Self::Collect(values) | Self::CollectDistinct(values, _) => {
                // Vec clone and Arc slice coexist during conversion. Value
                // clones share immutable child backing, charged once here
                // for the output ownership edge independently of the state.
                values.iter().try_fold(
                    values
                        .len()
                        .checked_mul(size_of::<Value>())?
                        .checked_mul(2)?
                        .checked_add(arc_header)?
                        .checked_add(align_of::<Value>() - 1)?,
                    |total, value| total.checked_add(aggregate_value_heap_bytes(value)?),
                )
            }
            Self::PercentileDisc { values, .. } | Self::PercentileCont { values, .. } => {
                // Rust 1.97 stable driftsort requests at most len elements of
                // temporary storage while the full cloned Vec remains live.
                values.len().checked_mul(size_of::<f64>())?.checked_mul(2)
            }
            Self::GroupConcat(values, separator)
            | Self::GroupConcatDistinct(values, separator, _) => {
                let text = values.iter().try_fold(
                    values
                        .len()
                        .saturating_sub(1)
                        .checked_mul(separator.len())?,
                    |total, value| total.checked_add(value.len()),
                )?;
                // join's exact-capacity String coexists with ArcStr 1.2's
                // byte copy (two-word header plus at most seven pad bytes).
                text.checked_mul(2)?.checked_add(arc_header)?.checked_add(7)
            }
            Self::Count(_)
            | Self::CountDistinct(_, _)
            | Self::SumInt(_, _)
            | Self::SumIntDistinct(_, _, _)
            | Self::SumFloat(_, _, _)
            | Self::SumFloatDistinct(_, _, _, _)
            | Self::Avg(_, _)
            | Self::AvgDistinct(_, _, _)
            | Self::StdDev { .. }
            | Self::StdDevPop { .. }
            | Self::Variance { .. }
            | Self::VariancePop { .. }
            | Self::Bivariate { .. } => Some(0),
        }
    }

    fn distinct_values(&self) -> Option<&FxHashSet<HashableValue>> {
        match self {
            Self::CountDistinct(_, seen)
            | Self::SumIntDistinct(_, _, seen)
            | Self::SumFloatDistinct(_, _, _, seen)
            | Self::AvgDistinct(_, _, seen)
            | Self::MinDistinct(_, seen)
            | Self::MaxDistinct(_, seen)
            | Self::LastDistinct(_, seen)
            | Self::CollectDistinct(_, seen)
            | Self::GroupConcatDistinct(_, _, seen) => Some(seen),
            Self::StdDev { seen, .. }
            | Self::StdDevPop { seen, .. }
            | Self::Variance { seen, .. }
            | Self::VariancePop { seen, .. }
            | Self::PercentileDisc { seen, .. }
            | Self::PercentileCont { seen, .. }
            | Self::Bivariate { seen, .. } => seen.as_ref(),
            _ => None,
        }
    }
}

/// Term::from_ntriples uses a growing lexical String, optional Unicode-hex
/// scratch (at most eight UTF-8 chars), and at most three Arc<str> backings.
/// Decoding cannot expand beyond the source bytes. Four times the source
/// length covers lexical old/new reallocation and its final Arc conversion
/// alongside datatype/language copies; 96 bounds the hex old/new buffers.
/// The fixed datatype and three two-word headers are charged separately.
#[cfg(all(feature = "spill", feature = "triple-store"))]
fn aggregate_rdf_comparison_peak(value: &Value, depth: usize) -> Option<usize> {
    if depth > 256 {
        return None;
    }
    let Value::List(values) = value else {
        return Some(0);
    };
    let [visible, Value::String(exact), Value::String(marker)] = values.as_ref() else {
        return Some(0);
    };
    if marker.as_str() != grafeo_common::types::INTERNAL_RDF_TAGGED_TERM_MARKER {
        return Some(0);
    }
    exact
        .len()
        .max(8)
        .checked_mul(4)?
        .checked_add(96)?
        .checked_add(crate::graph::rdf::Literal::RDF_LANG_STRING.len())?
        .checked_add(size_of::<usize>().checked_mul(6)?)?
        .checked_add(aggregate_rdf_comparison_peak(visible, depth + 1)?)
}

#[cfg(feature = "spill")]
fn aggregate_value_heap_bytes(value: &Value) -> Option<usize> {
    value.retained_size_bytes()?.checked_sub(size_of::<Value>())
}

use super::value_utils::{compare_values, value_to_f64};

/// Converts a Value to its string representation for GROUP_CONCAT.
fn agg_value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => format!("{other:?}"),
    }
}

/// A group key for hash-based aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey(Vec<GroupKeyPart>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKeyPart {
    Null,
    Bool(bool),
    Int64(i64),
    String(ArcStr),
    Bytes(Arc<[u8]>),
    Date(grafeo_common::types::Date),
    Time(grafeo_common::types::Time),
    Timestamp(grafeo_common::types::Timestamp),
    Duration(grafeo_common::types::Duration),
    ZonedDatetime(grafeo_common::types::ZonedDatetime),
    List(Vec<GroupKeyPart>),
    Map(Vec<(ArcStr, GroupKeyPart)>),
}

impl GroupKeyPart {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(b),
            Value::Int64(i) => Self::Int64(i),
            // reason: intentional bit-level reinterpretation for grouping equality;
            // canonicalize -0.0 to +0.0 so GROUP BY treats the two zeros as one
            // group key (IEEE-754 -0.0 == 0.0).
            #[allow(clippy::cast_possible_wrap)]
            Value::Float64(f) => Self::Int64(grafeo_common::types::canonical_f64_bits(f) as i64),
            Value::String(s) => Self::String(s.clone()),
            Value::Bytes(b) => Self::Bytes(b),
            Value::Date(d) => Self::Date(d),
            Value::Time(t) => Self::Time(t),
            Value::Timestamp(ts) => Self::Timestamp(ts),
            Value::Duration(d) => Self::Duration(d),
            Value::ZonedDatetime(zdt) => Self::ZonedDatetime(zdt),
            Value::List(items) => Self::List(items.iter().cloned().map(Self::from_value).collect()),
            Value::Map(map) => {
                // BTreeMap already iterates in key order, so this is deterministic
                let entries: Vec<(ArcStr, GroupKeyPart)> = map
                    .iter()
                    .map(|(k, v)| (ArcStr::from(k.as_str()), Self::from_value(v.clone())))
                    .collect();
                Self::Map(entries)
            }
            // Path, Vector, GCounter, OnCounter: use Debug string as fallback
            other => Self::String(ArcStr::from(format!("{other:?}"))),
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(*b),
            Self::Int64(i) => Value::Int64(*i),
            Self::String(s) => Value::String(s.clone()),
            Self::Bytes(b) => Value::Bytes(Arc::clone(b)),
            Self::Date(d) => Value::Date(*d),
            Self::Time(t) => Value::Time(*t),
            Self::Timestamp(ts) => Value::Timestamp(*ts),
            Self::Duration(d) => Value::Duration(*d),
            Self::ZonedDatetime(zdt) => Value::ZonedDatetime(*zdt),
            Self::List(parts) => {
                let values: Vec<Value> = parts.iter().map(Self::to_value).collect();
                Value::List(Arc::from(values.into_boxed_slice()))
            }
            Self::Map(entries) => {
                let map: std::collections::BTreeMap<PropertyKey, Value> = entries
                    .iter()
                    .map(|(k, v)| (PropertyKey::new(k.as_str()), v.to_value()))
                    .collect();
                Value::Map(Arc::new(map))
            }
        }
    }
}

impl GroupKey {
    /// Creates a group key from column values.
    fn from_row(chunk: &DataChunk, row: usize, group_columns: &[usize]) -> Self {
        let parts: Vec<GroupKeyPart> = group_columns
            .iter()
            .map(|&col_idx| {
                chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row))
                    .map_or(GroupKeyPart::Null, GroupKeyPart::from_value)
            })
            .collect();
        GroupKey(parts)
    }

    /// Converts the group key back to values.
    fn to_values(&self) -> Vec<Value> {
        self.0.iter().map(GroupKeyPart::to_value).collect()
    }
}

fn distinct_key_from_row(
    aggregate: &AggregateExpr,
    chunk: &DataChunk,
    row: usize,
) -> Result<Option<Value>, OperatorError> {
    if !aggregate.distinct {
        return Ok(None);
    }
    let Some(column_index) = aggregate.distinct_key_column else {
        return Ok(None);
    };
    let column = chunk
        .column(column_index)
        .ok_or_else(|| OperatorError::ColumnNotFound(column_index.to_string()))?;
    let key = column.get_value(row).ok_or_else(|| {
        OperatorError::Execution(format!(
            "DISTINCT key column {column_index} had no value for row {row}"
        ))
    })?;
    Ok(Some(key))
}

/// Hash-based aggregate operator.
///
/// Groups input by key columns and computes aggregations for each group.
pub struct HashAggregateOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Columns to group by.
    group_columns: Vec<usize>,
    /// Aggregation expressions.
    aggregates: Vec<AggregateExpr>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Ordered map: group key -> aggregate states (IndexMap for deterministic iteration order).
    groups: IndexMap<GroupKey, Vec<AggregateState>>,
    /// Whether aggregation is complete.
    aggregation_complete: bool,
    /// Results iterator.
    results: Option<std::vec::IntoIter<(GroupKey, Vec<AggregateState>)>>,
}

impl HashAggregateOperator {
    /// Creates a new hash aggregate operator.
    ///
    /// # Arguments
    /// * `child` - Child operator to read from.
    /// * `group_columns` - Column indices to group by.
    /// * `aggregates` - Aggregation expressions.
    /// * `output_schema` - Schema of the output (group columns + aggregate results).
    pub fn new(
        child: Box<dyn Operator>,
        group_columns: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            group_columns,
            aggregates,
            output_schema,
            groups: IndexMap::new(),
            aggregation_complete: false,
            results: None,
        }
    }

    /// Decomposes this operator for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<usize>, Vec<AggregateExpr>) {
        (self.child, self.group_columns, self.aggregates)
    }

    /// Performs the aggregation.
    fn aggregate(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            for row in chunk.selected_indices() {
                let key = GroupKey::from_row(&chunk, row, &self.group_columns);

                // Get or create aggregate states for this group
                let states = self.groups.entry(key).or_insert_with(|| {
                    self.aggregates
                        .iter()
                        .map(|agg| {
                            AggregateState::new(
                                agg.function,
                                agg.distinct,
                                agg.percentile,
                                agg.separator.as_deref(),
                            )
                        })
                        .collect()
                });

                // Update each aggregate
                for (i, agg) in self.aggregates.iter().enumerate() {
                    // Binary set functions: read two column values
                    if agg.column2.is_some() {
                        let y_val = agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        let x_val = agg
                            .column2
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        states[i].update_bivariate_with_distinct_key(
                            y_val,
                            x_val,
                            distinct_key_from_row(agg, &chunk, row)?,
                        );
                        continue;
                    }

                    let value = match (agg.function, agg.distinct) {
                        // COUNT(*) without DISTINCT doesn't need a value
                        (AggregateFunction::Count, false) => None,
                        // COUNT DISTINCT needs the actual value to track unique values
                        (AggregateFunction::Count, true) => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                        _ => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                    };
                    let distinct_key = distinct_key_from_row(agg, &chunk, row)?;
                    // For COUNT without DISTINCT, always update. For others, skip nulls.
                    match (agg.function, agg.distinct) {
                        (AggregateFunction::Count, false) => states[i].update(None),
                        (AggregateFunction::Count, true) => {
                            // COUNT DISTINCT needs the value to track unique values
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update_with_distinct_key(value, distinct_key);
                            }
                        }
                        (AggregateFunction::CountNonNull, _) => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update_with_distinct_key(value, distinct_key);
                            }
                        }
                        _ => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update_with_distinct_key(value, distinct_key);
                            }
                        }
                    }
                }
            }
        }

        self.aggregation_complete = true;

        // Convert to results iterator (IndexMap::drain takes a range)
        let results: Vec<_> = self.groups.drain(..).collect();
        self.results = Some(results.into_iter());

        Ok(())
    }
}

impl Operator for HashAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        // Perform aggregation if not done
        if !self.aggregation_complete {
            self.aggregate()?;
        }

        // Special case: no groups (global aggregation with no data)
        if self.groups.is_empty() && self.results.is_none() && self.group_columns.is_empty() {
            // For global aggregation (no GROUP BY), return one row with initial values
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);

            for agg in &self.aggregates {
                let state = AggregateState::new(
                    agg.function,
                    agg.distinct,
                    agg.percentile,
                    agg.separator.as_deref(),
                );
                let value = state.finalize();
                if let Some(col) = builder.column_mut(self.group_columns.len()) {
                    col.push_value(value);
                }
            }
            builder.advance_row();

            self.results = Some(Vec::new().into_iter()); // Mark as done
            return Ok(Some(builder.finish()));
        }

        let Some(results) = &mut self.results else {
            return Ok(None);
        };

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        for (key, states) in results.by_ref() {
            // Output group key columns
            let key_values = key.to_values();
            for (i, value) in key_values.into_iter().enumerate() {
                if let Some(col) = builder.column_mut(i) {
                    col.push_value(value);
                }
            }

            // Output aggregate results
            for (i, state) in states.iter().enumerate() {
                let col_idx = self.group_columns.len() + i;
                if let Some(col) = builder.column_mut(col_idx) {
                    col.push_value(state.finalize());
                }
            }

            builder.advance_row();

            if builder.is_full() {
                return Ok(Some(builder.finish()));
            }
        }

        if builder.row_count() > 0 {
            Ok(Some(builder.finish()))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.groups.clear();
        self.aggregation_complete = false;
        self.results = None;
    }

    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }

    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, crate::execution::QueryResourceContextError> {
        let (child, group_columns, aggregates) = (*self).into_parts();
        #[cfg(feature = "spill")]
        if resources.has_spill_manager() {
            return Ok(OperatorPipelineDecomposition::unary(
                child,
                Box::new(
                    super::push::SpillableAggregatePushOperator::with_qualified_resource_context(
                        group_columns,
                        aggregates,
                        resources.clone(),
                    )?,
                ),
            ));
        }
        Ok(OperatorPipelineDecomposition::unary(
            child,
            Box::new(super::push::AggregatePushOperator::with_resource_context(
                group_columns,
                aggregates,
                resources.clone(),
            )?),
        ))
    }
}

/// Simple (non-grouping) aggregate operator for global aggregations.
///
/// Used when there's no GROUP BY clause - aggregates all input into a single row.
pub struct SimpleAggregateOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Aggregation expressions.
    aggregates: Vec<AggregateExpr>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Aggregate states.
    states: Vec<AggregateState>,
    /// Whether aggregation is complete.
    done: bool,
    /// Shared cancellation/deadline state for the blocking input drain.
    cancellation: Option<crate::execution::QueryCancellationToken>,
}

impl SimpleAggregateOperator {
    fn check_cancellation(&self) -> Result<(), OperatorError> {
        if let Some(token) = &self.cancellation {
            token.check()?;
        }
        Ok(())
    }

    /// Creates a new simple aggregate operator.
    pub fn new(
        child: Box<dyn Operator>,
        aggregates: Vec<AggregateExpr>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        let states = aggregates
            .iter()
            .map(|agg| {
                AggregateState::new(
                    agg.function,
                    agg.distinct,
                    agg.percentile,
                    agg.separator.as_deref(),
                )
            })
            .collect();

        Self {
            child,
            aggregates,
            output_schema,
            states,
            done: false,
            cancellation: None,
        }
    }
}

impl Operator for SimpleAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }

        // Process all input
        loop {
            self.check_cancellation()?;
            let chunk = self.child.next()?;
            self.check_cancellation()?;
            let Some(chunk) = chunk else {
                break;
            };
            for (position, row) in chunk.selected_indices().enumerate() {
                if position % 128 == 0 {
                    self.check_cancellation()?;
                }
                for (i, agg) in self.aggregates.iter().enumerate() {
                    // Binary set functions: read two column values
                    if agg.column2.is_some() {
                        let y_val = agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        let x_val = agg
                            .column2
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        self.states[i].update_bivariate_with_distinct_key(
                            y_val,
                            x_val,
                            distinct_key_from_row(agg, &chunk, row)?,
                        );
                        continue;
                    }

                    let value = match (agg.function, agg.distinct) {
                        // COUNT(*) without DISTINCT doesn't need a value
                        (AggregateFunction::Count, false) => None,
                        // COUNT DISTINCT needs the actual value to track unique values
                        (AggregateFunction::Count, true) => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                        _ => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                    };
                    let distinct_key = distinct_key_from_row(agg, &chunk, row)?;

                    match (agg.function, agg.distinct) {
                        (AggregateFunction::Count, false) => self.states[i].update(None),
                        (AggregateFunction::Count, true) => {
                            // COUNT DISTINCT needs the value to track unique values
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update_with_distinct_key(value, distinct_key);
                            }
                        }
                        (AggregateFunction::CountNonNull, _) => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update_with_distinct_key(value, distinct_key);
                            }
                        }
                        _ => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update_with_distinct_key(value, distinct_key);
                            }
                        }
                    }
                }
            }
        }

        // Output single result row
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);

        for (i, state) in self.states.iter().enumerate() {
            if let Some(col) = builder.column_mut(i) {
                col.push_value(state.finalize());
            }
        }
        builder.advance_row();

        self.done = true;
        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.child.reset();
        self.states = self
            .aggregates
            .iter()
            .map(|agg| {
                AggregateState::new(
                    agg.function,
                    agg.distinct,
                    agg.percentile,
                    agg.separator.as_deref(),
                )
            })
            .collect();
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "SimpleAggregate"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)?;
        self.cancellation = Some(resources.cancellation_token().clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn create_test_chunk() -> DataChunk {
        // Create: [(group, value)] = [(1, 10), (1, 20), (2, 30), (2, 40), (2, 50)]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);

        let data = [(1i64, 10i64), (1, 20), (2, 30), (2, 40), (2, 50)];
        for (group, value) in data {
            builder.column_mut(0).unwrap().push_int64(group);
            builder.column_mut(1).unwrap().push_int64(value);
            builder.advance_row();
        }

        builder.finish()
    }

    #[test]
    fn simple_aggregate_checks_cancellation_after_child_pull() {
        use crate::execution::{QueryExecutionControl, QueryResourceContext};
        use grafeo_common::memory::buffer::BufferManager;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CancelAfterPull {
            handle: crate::execution::QueryCancellationHandle,
            pulls: Arc<AtomicUsize>,
        }
        impl Operator for CancelAfterPull {
            fn next(&mut self) -> OperatorResult {
                assert_eq!(
                    self.pulls.fetch_add(1, Ordering::SeqCst),
                    0,
                    "cancelled source must not be pulled twice"
                );
                self.handle.cancel();
                let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], 1);
                chunk.column_mut(0).unwrap().push_int64(1);
                chunk.set_count(1);
                Ok(Some(chunk))
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "CancelAfterPull"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }
        let control = QueryExecutionControl::new();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .unwrap();
        let pulls = Arc::new(AtomicUsize::new(0));
        let mut aggregate = SimpleAggregateOperator::new(
            Box::new(CancelAfterPull {
                handle: control.cancellation_handle(),
                pulls: pulls.clone(),
            }),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        );
        aggregate.install_resource_context(&resources).unwrap();
        assert!(matches!(
            aggregate.next(),
            Err(OperatorError::QueryCancelled(
                crate::execution::QueryCancellationError::Cancelled,
            ))
        ));
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_simple_count() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(5));

        // Should be done
        assert!(agg.next().unwrap().is_none());
    }

    #[test]
    fn test_simple_sum() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::sum(1)], // Sum of column 1
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sum: 10 + 20 + 30 + 40 + 50 = 150
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(150));
    }

    #[test]
    fn test_simple_avg() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::avg(1)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Avg: 150 / 5 = 30.0
        let avg = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((avg - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_simple_min_max() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::min(1), AggregateExpr::max(1)],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10)); // Min
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(50)); // Max
    }

    #[test]
    fn test_sum_with_string_values() {
        // Test SUM with string values (like RDF stores numeric literals)
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        builder.column_mut(0).unwrap().push_string("30");
        builder.advance_row();
        builder.column_mut(0).unwrap().push_string("25");
        builder.advance_row();
        builder.column_mut(0).unwrap().push_string("35");
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::sum(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Should parse strings and sum: 30 + 25 + 35 = 90
        let sum_val = result.column(0).unwrap().get_float64(0).unwrap();
        assert!(
            (sum_val - 90.0).abs() < 0.001,
            "Expected 90.0, got {}",
            sum_val
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn grouped_query_decomposition_rejects_undeclared_spill_hooks() {
        use crate::execution::spill::{BorrowedSpillFixture, PartitionAdmissionError, SpillIo};
        use crate::execution::{QueryExecutionControl, QueryResourceContextError};
        use grafeo_common::memory::buffer::BufferManager;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct UndeclaredHooks(AtomicUsize);
        impl SpillIo for UndeclaredHooks {
            fn check(
                &self,
                _operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }

        let directory = tempfile::TempDir::new().unwrap();
        let buffer_manager = BufferManager::with_budget(1 << 20);
        let control = QueryExecutionControl::new();
        let io = Arc::new(UndeclaredHooks(AtomicUsize::new(0)));
        let (resources, manager) = BorrowedSpillFixture::new(directory.path())
            .io(Arc::clone(&io) as Arc<dyn SpillIo>)
            .build_operator_resources(Arc::clone(&buffer_manager), control.token())
            .unwrap();
        let baseline_bytes = resources.query_stats().allocated_bytes;
        let baseline_consumers = buffer_manager.stats().consumer_count;
        let aggregate = HashAggregateOperator::new(
            Box::new(MockOperator::new(vec![create_test_chunk()])),
            vec![0],
            vec![AggregateExpr::sum(1)],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        assert!(matches!(
            Box::new(aggregate).decompose_pipeline_with_resources(&resources),
            Err(QueryResourceContextError::PartitionAdmission(
                PartitionAdmissionError::UnboundedHooks
            ))
        ));
        assert_eq!(io.0.load(Ordering::Relaxed), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, baseline_bytes);
        assert_eq!(buffer_manager.stats().consumer_count, baseline_consumers);
    }

    #[test]
    fn test_grouped_aggregation() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        // GROUP BY column 0, SUM(column 1)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],                     // Group by column 0
            vec![AggregateExpr::sum(1)], // Sum of column 1
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let sum = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((group, sum));
            }
        }

        results.sort_by_key(|(g, _)| *g);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 30)); // Group 1: 10 + 20 = 30
        assert_eq!(results[1], (2, 120)); // Group 2: 30 + 40 + 50 = 120
    }

    #[test]
    fn test_grouped_count() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        // GROUP BY column 0, COUNT(*)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((group, count));
            }
        }

        results.sort_by_key(|(g, _)| *g);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 2)); // Group 1: 2 rows
        assert_eq!(results[1], (2, 3)); // Group 2: 3 rows
    }

    #[test]
    fn test_multiple_aggregates() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        // GROUP BY column 0, COUNT(*), SUM(column 1), AVG(column 1)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![
                AggregateExpr::count_star(),
                AggregateExpr::sum(1),
                AggregateExpr::avg(1),
            ],
            vec![
                LogicalType::Int64,   // Group key
                LogicalType::Int64,   // COUNT
                LogicalType::Int64,   // SUM
                LogicalType::Float64, // AVG
            ],
        );

        let mut results: Vec<(i64, i64, i64, f64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                let sum = chunk.column(2).unwrap().get_int64(row).unwrap();
                let avg = chunk.column(3).unwrap().get_float64(row).unwrap();
                results.push((group, count, sum, avg));
            }
        }

        results.sort_by_key(|(g, _, _, _)| *g);
        assert_eq!(results.len(), 2);

        // Group 1: COUNT=2, SUM=30, AVG=15.0
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, 2);
        assert_eq!(results[0].2, 30);
        assert!((results[0].3 - 15.0).abs() < 0.001);

        // Group 2: COUNT=3, SUM=120, AVG=40.0
        assert_eq!(results[1].0, 2);
        assert_eq!(results[1].1, 3);
        assert_eq!(results[1].2, 120);
        assert!((results[1].3 - 40.0).abs() < 0.001);
    }

    fn create_test_chunk_with_duplicates() -> DataChunk {
        // Create data with duplicate values in column 1
        // [(group, value)] = [(1, 10), (1, 10), (1, 20), (2, 30), (2, 30), (2, 30)]
        // GROUP 1: values [10, 10, 20] -> distinct count = 2
        // GROUP 2: values [30, 30, 30] -> distinct count = 1
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);

        let data = [(1i64, 10i64), (1, 10), (1, 20), (2, 30), (2, 30), (2, 30)];
        for (group, value) in data {
            builder.column_mut(0).unwrap().push_int64(group);
            builder.column_mut(1).unwrap().push_int64(value);
            builder.advance_row();
        }

        builder.finish()
    }

    #[test]
    fn count_distinct_never_interprets_public_marker_shaped_lists() {
        let ordinary_list = |visible: &str| {
            Value::List(
                vec![
                    Value::String(visible.into()),
                    Value::String("same second item".into()),
                    Value::String("\0grafeo:rdf-tagged-term-pair".into()),
                ]
                .into(),
            )
        };
        let mut state = AggregateState::new(AggregateFunction::CountNonNull, true, None, None);
        state.update(Some(ordinary_list("first")));
        state.update(Some(ordinary_list("second")));

        assert_eq!(state.finalize(), Value::Int64(2));
    }

    #[test]
    fn distinct_aggregates_use_the_explicit_key_column_and_original_value() {
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String, LogicalType::String]);
        for (number, text, key) in [
            (1, "same", "term-1"),
            (1, "same", "term-2"),
            (3, "other", "term-3"),
            (9, "ignored duplicate key", "term-1"),
        ] {
            builder.column_mut(0).unwrap().push_int64(number);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::String(text.into()));
            builder
                .column_mut(2)
                .unwrap()
                .push_value(Value::String(key.into()));
            builder.advance_row();
        }

        let mut group_concat = AggregateExpr::count(1)
            .with_distinct()
            .with_distinct_key_column(2);
        group_concat.function = AggregateFunction::GroupConcat;
        group_concat.separator = Some("|".to_string());
        let mut aggregate = SimpleAggregateOperator::new(
            Box::new(MockOperator::new(vec![builder.finish()])),
            vec![
                AggregateExpr::count(0)
                    .with_distinct()
                    .with_distinct_key_column(2),
                AggregateExpr::sum(0)
                    .with_distinct()
                    .with_distinct_key_column(2),
                AggregateExpr::avg(0)
                    .with_distinct()
                    .with_distinct_key_column(2),
                group_concat,
            ],
            vec![
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Float64,
                LogicalType::String,
            ],
        );

        let result = aggregate.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Int64(3))
        );
        assert_eq!(
            result.column(1).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
        let Some(Value::Float64(average)) = result.column(2).unwrap().get_value(0) else {
            panic!("AVG DISTINCT must produce a floating-point result");
        };
        assert!((average - 5.0 / 3.0).abs() < f64::EPSILON);
        assert_eq!(
            result.column(3).unwrap().get_value(0),
            Some(Value::String("same|same|other".into()))
        );
    }

    #[test]
    fn grouped_distinct_uses_the_explicit_key_column() {
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64, LogicalType::String]);
        for (value, key) in [(1, "term-1"), (1, "term-2"), (3, "term-3"), (9, "term-1")] {
            builder.column_mut(0).unwrap().push_int64(7);
            builder.column_mut(1).unwrap().push_int64(value);
            builder
                .column_mut(2)
                .unwrap()
                .push_value(Value::String(key.into()));
            builder.advance_row();
        }

        let mut aggregate = HashAggregateOperator::new(
            Box::new(MockOperator::new(vec![builder.finish()])),
            vec![0],
            vec![
                AggregateExpr::sum(1)
                    .with_distinct()
                    .with_distinct_key_column(2),
            ],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = aggregate.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Int64(7))
        );
        assert_eq!(
            result.column(1).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
    }

    #[test]
    fn test_count_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // COUNT(DISTINCT column 1)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::count(1).with_distinct()],
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Total distinct values: 10, 20, 30 = 3 distinct values
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(3));
    }

    #[test]
    fn test_grouped_count_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // GROUP BY column 0, COUNT(DISTINCT column 1)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![AggregateExpr::count(1).with_distinct()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((group, count));
            }
        }

        results.sort_by_key(|(g, _)| *g);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 2)); // Group 1: [10, 10, 20] -> 2 distinct values
        assert_eq!(results[1], (2, 1)); // Group 2: [30, 30, 30] -> 1 distinct value
    }

    #[test]
    fn test_sum_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // SUM(DISTINCT column 1)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::sum(1).with_distinct()],
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sum of distinct values: 10 + 20 + 30 = 60
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(60));
    }

    #[test]
    fn test_avg_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // AVG(DISTINCT column 1)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::avg(1).with_distinct()],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Avg of distinct values: (10 + 20 + 30) / 3 = 20.0
        let avg = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((avg - 20.0).abs() < 0.001);
    }

    fn create_statistical_test_chunk() -> DataChunk {
        // Create data: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Sample StdDev = 2.138, Population StdDev = 2.0
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);

        for value in [2i64, 4, 4, 4, 5, 5, 7, 9] {
            builder.column_mut(0).unwrap().push_int64(value);
            builder.advance_row();
        }

        builder.finish()
    }

    fn distinct_prerequisite_pull_results(
        data: &[[i64; 3]],
        expressions: Vec<AggregateExpr>,
        grouped: bool,
    ) -> Vec<Value> {
        // Operand, second operand, independent identity, constant group.
        let mut builder = DataChunkBuilder::new(&[
            LogicalType::Int64,
            LogicalType::Int64,
            LogicalType::Int64,
            LogicalType::Int64,
        ]);
        for row in data {
            for (column, value) in row.iter().copied().chain([0]).enumerate() {
                builder.column_mut(column).unwrap().push_int64(value);
            }
            builder.advance_row();
        }
        let width = expressions.len();
        let input = Box::new(MockOperator::new(vec![builder.finish()]));
        let mut operator: Box<dyn Operator> = if grouped {
            Box::new(HashAggregateOperator::new(
                input,
                vec![3],
                expressions,
                vec![LogicalType::Any; width + 1],
            ))
        } else {
            Box::new(SimpleAggregateOperator::new(
                input,
                expressions,
                vec![LogicalType::Any; width],
            ))
        };
        let chunk = operator.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        let offset = usize::from(grouped);
        (offset..offset + width)
            .map(|column| chunk.column(column).unwrap().get_value(0).unwrap())
            .collect()
    }

    #[test]
    fn distinct_identity_extrema_keep_first_operand_per_key() {
        for grouped in [false, true] {
            for (function, data) in [
                (AggregateFunction::Min, [[100, 0, 1], [1, 0, 1], [50, 0, 2]]),
                (AggregateFunction::Max, [[1, 0, 1], [100, 0, 1], [50, 0, 2]]),
            ] {
                let mut expression = AggregateExpr::min(0)
                    .with_distinct()
                    .with_distinct_key_column(2);
                expression.function = function;
                assert_eq!(
                    distinct_prerequisite_pull_results(&data, vec![expression], grouped),
                    vec![Value::Int64(50)],
                    "{function:?}, grouped={grouped}"
                );
            }
        }
    }

    #[test]
    fn distinct_identity_extrema_and_last_preserve_null_admission() {
        for function in [AggregateFunction::Min, AggregateFunction::Max] {
            let mut state = AggregateState::new(function, true, None, None);
            state.update_with_distinct_key(None, Some(Value::Int64(1)));
            state.update_with_distinct_key(Some(Value::Null), Some(Value::Int64(1)));
            state.update_with_distinct_key(Some(Value::Int64(10)), Some(Value::Int64(1)));
            assert_eq!(state.finalize(), Value::Int64(10), "{function:?}");
        }
        let mut state = AggregateState::new(AggregateFunction::Last, true, None, None);
        state.update_with_distinct_key(None, Some(Value::Int64(1)));
        state.update_with_distinct_key(Some(Value::Int64(10)), Some(Value::Int64(1)));
        assert_eq!(state.finalize(), Value::Int64(10));
        state.update_with_distinct_key(Some(Value::Null), Some(Value::Int64(2)));
        state.update_with_distinct_key(Some(Value::Int64(20)), Some(Value::Int64(2)));
        assert_eq!(
            state.finalize(),
            Value::Null,
            "Some(Null) claims a LAST identity"
        );
        state.update_with_distinct_key(None, Some(Value::Int64(3)));
        state.update_with_distinct_key(Some(Value::Int64(30)), Some(Value::Int64(3)));
        assert_eq!(state.finalize(), Value::Int64(30));
    }

    #[test]
    fn distinct_identity_last_keeps_last_new_identity() {
        for grouped in [false, true] {
            let mut expression = AggregateExpr::min(0)
                .with_distinct()
                .with_distinct_key_column(2);
            expression.function = AggregateFunction::Last;
            assert_eq!(
                distinct_prerequisite_pull_results(
                    &[[1, 0, 1], [3, 0, 2], [2, 0, 1]],
                    vec![expression],
                    grouped,
                ),
                vec![Value::Int64(3)]
            );
        }
        let mut state = AggregateState::new(AggregateFunction::Last, true, None, None);
        for value in [1, 3, 1] {
            state.update(Some(Value::Int64(value)));
        }
        assert_eq!(state.finalize(), Value::Int64(3));
    }

    #[test]
    fn distinct_identity_sum_avg_invalid_input_does_not_claim_key() {
        use crate::execution::vector::ValueVector;
        for grouped in [false, true] {
            for prefix in [None, Some(Value::Int64(0)), Some(Value::Float64(0.0))] {
                for operand in [
                    Value::Int64(10),
                    Value::Float64(10.0),
                    Value::String("10".into()),
                ] {
                    let mut values: Vec<_> = prefix.clone().into_iter().collect();
                    let mut keys = vec![Value::Int64(0); values.len()];
                    values.extend([
                        Value::Null,
                        Value::Bool(true),
                        Value::String("bad".into()),
                        Value::List(Arc::from([Value::Int64(999)])),
                        operand.clone(),
                    ]);
                    keys.extend(vec![Value::Int64(1); 5]);
                    let chunk = DataChunk::new(vec![
                        ValueVector::from_values(&values),
                        ValueVector::from_values(&keys),
                        ValueVector::from_values(&vec![Value::Int64(0); values.len()]),
                    ]);
                    let expressions = vec![
                        AggregateExpr::sum(0)
                            .with_distinct()
                            .with_distinct_key_column(1),
                        AggregateExpr::avg(0)
                            .with_distinct()
                            .with_distinct_key_column(1),
                    ];
                    let input = Box::new(MockOperator::new(vec![chunk]));
                    let mut operator: Box<dyn Operator> = if grouped {
                        Box::new(HashAggregateOperator::new(
                            input,
                            vec![2],
                            expressions,
                            vec![LogicalType::Any; 3],
                        ))
                    } else {
                        Box::new(SimpleAggregateOperator::new(
                            input,
                            expressions,
                            vec![LogicalType::Any; 2],
                        ))
                    };
                    let result = operator.next().unwrap().unwrap();
                    let offset = usize::from(grouped);
                    let sum = result.column(offset).unwrap().get_value(0).unwrap();
                    let expected_sum = if matches!(prefix, Some(Value::Float64(_)))
                        || !matches!(operand, Value::Int64(_))
                    {
                        Value::Float64(10.0)
                    } else {
                        Value::Int64(10)
                    };
                    assert_eq!(
                        sum, expected_sum,
                        "grouped={grouped}, prefix={prefix:?}, operand={operand:?}"
                    );
                    assert_eq!(
                        result.column(offset + 1).unwrap().get_value(0),
                        Some(Value::Float64(if prefix.is_some() { 5.0 } else { 10.0 }))
                    );
                }
            }
        }
    }

    #[test]
    fn distinct_statistics_and_percentiles_use_value_and_identity_keys() {
        for grouped in [false, true] {
            for explicit_key in [false, true] {
                let data = if explicit_key {
                    // Equal operands with different identities both count;
                    // a later changed operand with identity 1 must not count.
                    vec![[1, 0, 1], [1, 0, 2], [3, 0, 3], [9, 0, 1]]
                } else {
                    vec![[1, 0, 1], [3, 0, 2], [3, 0, 3]]
                };
                let expressions = [
                    AggregateExpr::stdev_pop(0),
                    AggregateExpr::percentile_disc(0, 0.5),
                    AggregateExpr::percentile_cont(0, 0.5),
                ]
                .into_iter()
                .map(|expression| {
                    let expression = expression.with_distinct();
                    if explicit_key {
                        expression.with_distinct_key_column(2)
                    } else {
                        expression
                    }
                })
                .collect();
                let actual = distinct_prerequisite_pull_results(&data, expressions, grouped);
                let expected = if explicit_key {
                    [(8.0_f64 / 9.0).sqrt(), 1.0, 1.0]
                } else {
                    [1.0, 1.0, 2.0]
                };
                assert!(
                    actual.iter().zip(expected).all(|(value, expected)| {
                        matches!(value, Value::Float64(actual) if (actual - expected).abs() < 1e-12)
                    }),
                    "grouped={grouped}, explicit_key={explicit_key}: {actual:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn distinct_bivariate_counts_pairs_or_explicit_identities() {
        for grouped in [false, true] {
            for explicit_key in [false, true] {
                let (data, expected) = if explicit_key {
                    (vec![[1, 10, 1], [1, 10, 2], [3, 30, 3], [9, 90, 1]], 3)
                } else {
                    // Pair identity must retain both x values for the same y.
                    (vec![[1, 10, 1], [1, 10, 2], [1, 20, 3]], 2)
                };
                let expression = AggregateExpr {
                    function: AggregateFunction::RegrCount,
                    column: Some(0),
                    column2: Some(1),
                    distinct_key_column: explicit_key.then_some(2),
                    distinct: true,
                    alias: None,
                    percentile: None,
                    separator: None,
                };
                let actual = distinct_prerequisite_pull_results(&data, vec![expression], grouped);
                assert_eq!(
                    actual,
                    vec![Value::Int64(expected)],
                    "grouped={grouped}, explicit_key={explicit_key}"
                );
            }
        }
    }

    #[test]
    fn distinct_numeric_invalid_operands_do_not_consume_identity() {
        for function in [
            AggregateFunction::StdDevPop,
            AggregateFunction::PercentileCont,
        ] {
            let mut state = AggregateState::new(function, true, Some(0.5), None);
            for invalid in [Value::Null, Value::String("not-numeric".into())] {
                state.update_with_distinct_key(Some(invalid), Some(Value::Int64(1)));
            }
            state.update_with_distinct_key(Some(Value::Int64(1)), Some(Value::Int64(1)));
            state.update_with_distinct_key(Some(Value::Int64(3)), Some(Value::Int64(2)));
            state.update_with_distinct_key(Some(Value::Int64(99)), Some(Value::Int64(1)));
            assert_eq!(
                state.finalize(),
                Value::Float64(if function == AggregateFunction::StdDevPop {
                    1.0
                } else {
                    2.0
                })
            );
        }
        let mut state = AggregateState::new(AggregateFunction::RegrCount, true, None, None);
        for invalid in [Value::Null, Value::String("not-numeric".into())] {
            state.update_bivariate_with_distinct_key(
                Some(Value::Int64(1)),
                Some(invalid),
                Some(Value::Int64(1)),
            );
        }
        state.update_bivariate_with_distinct_key(
            Some(Value::Int64(1)),
            Some(Value::Int64(10)),
            Some(Value::Int64(1)),
        );
        state.update_bivariate_with_distinct_key(
            Some(Value::Int64(99)),
            Some(Value::Int64(90)),
            Some(Value::Int64(1)),
        );
        assert_eq!(state.finalize(), Value::Int64(1));
    }

    #[test]
    fn test_stdev_sample() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sample standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Variance = 32/7 = 4.571, StdDev = 2.138
        let stdev = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((stdev - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_stdev_population() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev_pop(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Population standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Variance = 32/8 = 4.0, StdDev = 2.0
        let stdev = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((stdev - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_disc() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        // Median (50th percentile discrete)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::percentile_disc(0, 0.5)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sorted: [2, 4, 4, 4, 5, 5, 7, 9], index = floor(0.5 * 7) = 3, value = 4
        let percentile = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((percentile - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_cont() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        // Median (50th percentile continuous)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::percentile_cont(0, 0.5)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sorted: [2, 4, 4, 4, 5, 5, 7, 9], rank = 0.5 * 7 = 3.5
        // Interpolate between index 3 (4) and index 4 (5): 4 + 0.5 * (5 - 4) = 4.5
        let percentile = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((percentile - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile_extremes() {
        // Test 0th and 100th percentiles
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![
                AggregateExpr::percentile_disc(0, 0.0),
                AggregateExpr::percentile_disc(0, 1.0),
            ],
            vec![LogicalType::Float64, LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // 0th percentile = minimum = 2
        let p0 = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((p0 - 2.0).abs() < 0.01);
        // 100th percentile = maximum = 9
        let p100 = result.column(1).unwrap().get_float64(0).unwrap();
        assert!((p100 - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_stdev_single_value() {
        // Single value should return null for sample stdev
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sample stdev of single value is undefined (null)
        assert!(matches!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Null)
        ));
    }

    #[test]
    fn test_first_and_last() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::first(1), AggregateExpr::last(1)],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // First: 10, Last: 50
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(50));
    }

    #[test]
    fn test_collect() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::collect(1)],
            vec![LogicalType::Any],
        );

        let result = agg.next().unwrap().unwrap();
        let val = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::List(items) = val {
            assert_eq!(items.len(), 5);
        } else {
            panic!("Expected List value");
        }
    }

    #[test]
    fn test_collect_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::collect(1).with_distinct()],
            vec![LogicalType::Any],
        );

        let result = agg.next().unwrap().unwrap();
        let val = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::List(items) = val {
            // [10, 10, 20, 30, 30, 30] -> distinct: [10, 20, 30]
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected List value");
        }
    }

    #[test]
    fn test_group_concat() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        for s in ["hello", "world", "foo"] {
            builder.column_mut(0).unwrap().push_string(s);
            builder.advance_row();
        }
        let chunk = builder.finish();
        let mock = MockOperator::new(vec![chunk]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::GroupConcat,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg =
            SimpleAggregateOperator::new(Box::new(mock), vec![agg_expr], vec![LogicalType::String]);

        let result = agg.next().unwrap().unwrap();
        let val = result.column(0).unwrap().get_value(0).unwrap();
        assert_eq!(val, Value::String("hello world foo".into()));
    }

    #[test]
    fn test_sample() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::Sample,
            column: Some(1),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg =
            SimpleAggregateOperator::new(Box::new(mock), vec![agg_expr], vec![LogicalType::Int64]);

        let result = agg.next().unwrap().unwrap();
        // Sample should return the first non-null value (10)
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
    }

    #[test]
    fn test_variance_sample() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::Variance,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![agg_expr],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        // Sample variance of [2, 4, 4, 4, 5, 5, 7, 9]: M2/(n-1) = 32/7 = 4.571
        let variance = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((variance - 32.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_population() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::VariancePop,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![agg_expr],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        // Population variance: M2/n = 32/8 = 4.0
        let variance = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((variance - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_single_value() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();
        let mock = MockOperator::new(vec![chunk]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::Variance,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![agg_expr],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        // Sample variance of single value is undefined (null)
        assert!(matches!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Null)
        ));
    }

    #[test]
    fn test_empty_aggregation() {
        // No input rows: COUNT should be 0, SUM/AVG/MIN/MAX should be NULL
        // (ISO/IEC 39075 Section 20.9)
        let mock = MockOperator::new(vec![]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![
                AggregateExpr::count_star(),
                AggregateExpr::sum(0),
                AggregateExpr::avg(0),
                AggregateExpr::min(0),
                AggregateExpr::max(0),
            ],
            vec![
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Float64,
                LogicalType::Int64,
                LogicalType::Int64,
            ],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(0)); // COUNT
        assert!(matches!(
            result.column(1).unwrap().get_value(0),
            Some(Value::Null)
        )); // SUM
        assert!(matches!(
            result.column(2).unwrap().get_value(0),
            Some(Value::Null)
        )); // AVG
        assert!(matches!(
            result.column(3).unwrap().get_value(0),
            Some(Value::Null)
        )); // MIN
        assert!(matches!(
            result.column(4).unwrap().get_value(0),
            Some(Value::Null)
        )); // MAX
    }

    #[test]
    fn test_stdev_pop_single_value() {
        // Single value should return 0 for population stdev
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev_pop(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Population stdev of single value is 0
        let stdev = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((stdev - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_hash_aggregate_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<HashAggregateOperator>().is_ok());
    }

    #[test]
    fn test_simple_aggregate_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<SimpleAggregateOperator>().is_ok());
    }

    #[test]
    fn test_hash_aggregate_into_parts() {
        let mock = MockOperator::new(vec![]);
        let op = HashAggregateOperator::new(
            Box::new(mock),
            vec![0, 2],
            vec![AggregateExpr::sum(1), AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64],
        );
        let (mut child, group_columns, aggregates) = op.into_parts();
        assert_eq!(group_columns, vec![0, 2]);
        assert_eq!(aggregates.len(), 2);
        assert!(child.next().unwrap().is_none());
    }
}

#[cfg(all(test, feature = "spill"))]
mod resource_bounds_tests {
    use super::*;
    use grafeo_common::memory::buffer::{BufferManager, MemoryRegion};

    fn expression(function: AggregateFunction, distinct: bool) -> AggregateExpr {
        AggregateExpr {
            function,
            column: Some(0),
            column2: matches!(
                function,
                AggregateFunction::CovarSamp
                    | AggregateFunction::CovarPop
                    | AggregateFunction::Corr
                    | AggregateFunction::RegrSlope
                    | AggregateFunction::RegrIntercept
                    | AggregateFunction::RegrR2
                    | AggregateFunction::RegrCount
                    | AggregateFunction::RegrSxx
                    | AggregateFunction::RegrSyy
                    | AggregateFunction::RegrSxy
                    | AggregateFunction::RegrAvgx
                    | AggregateFunction::RegrAvgy
            )
            .then_some(1),
            distinct_key_column: None,
            distinct,
            alias: None,
            percentile: Some(0.5),
            separator: Some(" / ".repeat(17)),
        }
    }

    fn nested_value() -> Value {
        Value::Map(Arc::new(std::collections::BTreeMap::from([
            (
                PropertyKey::from("escaped\nkey"),
                Value::List(Arc::from([
                    Value::String(ArcStr::from("\u{1}\n\"\\".repeat(31))),
                    Value::Bytes(Arc::from([1u8; 113])),
                ])),
            ),
            (PropertyKey::from("number"), Value::Int64(7)),
        ])))
    }

    fn apply(state: &mut AggregateState, expr: &AggregateExpr, value: &Value) {
        if expr.column2.is_some() {
            state.update_bivariate_with_distinct_key(
                Some(value.clone()),
                Some(Value::Int64(3)),
                None,
            );
        } else {
            state.update(Some(value.clone()));
        }
    }

    #[test]
    fn accumulator_bounds_cover_all_functions_updates_and_outputs() {
        use AggregateFunction as F;
        let functions = [
            F::Count,
            F::CountNonNull,
            F::Sum,
            F::Avg,
            F::Min,
            F::Max,
            F::First,
            F::Last,
            F::Collect,
            F::StdDev,
            F::StdDevPop,
            F::Variance,
            F::VariancePop,
            F::PercentileDisc,
            F::PercentileCont,
            F::GroupConcat,
            F::Sample,
            F::CovarSamp,
            F::CovarPop,
            F::Corr,
            F::RegrSlope,
            F::RegrIntercept,
            F::RegrR2,
            F::RegrCount,
            F::RegrSxx,
            F::RegrSyy,
            F::RegrSxy,
            F::RegrAvgx,
            F::RegrAvgy,
        ];
        let inputs = [
            Value::Int64(7),
            Value::Float64(2.5),
            Value::String(ArcStr::from("23.75")),
            Value::Null,
            nested_value(),
            Value::Int64(7),
        ];
        for function in functions {
            for distinct in [false, true] {
                let expr = expression(function, distinct);
                let empty_peak =
                    AggregateState::replacement_peak_bytes(None, &expr, None, None, None).unwrap();
                let mut state = AggregateState::new(
                    expr.function,
                    expr.distinct,
                    expr.percentile,
                    expr.separator.as_deref(),
                );
                assert!(state.retained_heap_bytes().unwrap() <= empty_peak);
                let mut cached = state.retained_heap_bytes().unwrap();
                for input in &inputs {
                    let second = expr.column2.map(|_| Value::Int64(3));
                    let replacement = AggregateState::replacement_peak_bytes(
                        Some(&state),
                        &expr,
                        Some(input),
                        second.as_ref(),
                        None,
                    )
                    .unwrap();
                    let retained = state.retained_heap_bytes().unwrap();
                    let extra = state
                        .update_peak_bytes(&expr, Some(input), second.as_ref(), None)
                        .unwrap();
                    let before = state.retained_update_snapshot().unwrap();
                    let mut cloned = state.clone();
                    apply(&mut cloned, &expr, input);
                    apply(&mut state, &expr, input);
                    let (removed, added) = state
                        .retained_update_delta(before, &expr, Some(input), second.as_ref(), None)
                        .unwrap();
                    let prior_cached = cached;
                    cached = cached
                        .checked_sub(removed)
                        .unwrap()
                        .checked_add(added)
                        .unwrap();
                    let observed = state.retained_heap_bytes().unwrap();
                    assert!(
                        observed <= cached,
                        "cached {function:?} distinct={distinct}"
                    );
                    assert!(cached <= prior_cached.checked_add(extra).unwrap());
                    assert!(
                        observed <= retained.checked_add(extra).unwrap(),
                        "in-place {function:?} distinct={distinct}"
                    );
                    assert!(
                        observed <= replacement,
                        "replacement {function:?} distinct={distinct}"
                    );
                    assert_eq!(state.finalize(), cloned.finalize());
                    let output = state.finalize();
                    assert!(
                        aggregate_value_heap_bytes(&output).unwrap()
                            <= state.finalize_peak_bytes().unwrap(),
                        "output {function:?} distinct={distinct}"
                    );
                }
            }
        }
        let frozen = AggregateState::Frozen(nested_value());
        assert_eq!(frozen.retained_heap_bytes(), frozen.finalize_peak_bytes());
    }

    #[test]
    fn accumulator_bounds_cover_sparse_hash_clone_and_every_growth_boundary() {
        let expr = expression(AggregateFunction::Collect, true);
        let mut state = AggregateState::new(expr.function, true, None, None);
        for index in 0..257 {
            let value = Value::String(ArcStr::from(format!("value-{index:04}")));
            let old = state.retained_heap_bytes().unwrap();
            let extra = state
                .update_peak_bytes(&expr, Some(&value), None, None)
                .unwrap();
            state.update(Some(value));
            assert!(state.retained_heap_bytes().unwrap() <= old + extra);
        }
        let AggregateState::CollectDistinct(values, seen) = &mut state else {
            unreachable!();
        };
        values.clear();
        seen.retain(|key| matches!(key, HashableValue::String(value) if value.ends_with("0000")));
        let source_table = seen.allocation_size();
        let input = nested_value();
        let declared =
            AggregateState::replacement_peak_bytes(Some(&state), &expr, Some(&input), None, None)
                .unwrap();
        let mut cloned = state.clone();
        assert_eq!(
            cloned.distinct_values().unwrap().allocation_size(),
            source_table
        );
        cloned.update(Some(input));
        assert!(cloned.retained_heap_bytes().unwrap() <= declared);
        // The sparse live owner was not mutated by construction of its replacement.
        let AggregateState::CollectDistinct(values, seen) = &state else {
            unreachable!();
        };
        assert!(values.is_empty());
        assert_eq!(seen.len(), 1);
    }

    #[test]
    fn accumulator_declaration_denial_preserves_live_state_and_shared_input() {
        let expr = expression(AggregateFunction::Collect, true);
        let mut state = AggregateState::new(expr.function, true, None, None);
        state.update(Some(Value::Int64(7)));
        let before = state.finalize();
        let retained = state.retained_heap_bytes();
        let input = nested_value();
        let input_before = input.clone();
        let peak = state
            .update_peak_bytes(&expr, Some(&input), None, None)
            .unwrap();
        let memory = BufferManager::with_budget(1);
        let grant = memory.try_allocate(peak, MemoryRegion::ExecutionBuffers);
        assert!(grant.is_none());
        // The real caller only invokes update after a successful admission.
        if grant.is_some() {
            state.update(Some(input.clone()));
        }
        assert_eq!(state.finalize(), before);
        assert_eq!(state.retained_heap_bytes(), retained);
        assert_eq!(input, input_before);
        assert_eq!(memory.allocated(), 0);

        let mut excessive = Value::Null;
        for _ in 0..258 {
            excessive = Value::List(Arc::from([excessive]));
        }
        assert!(
            state
                .update_peak_bytes(&expr, Some(&excessive), None, None)
                .is_none()
        );
        assert_eq!(state.finalize(), before);
        assert!(super::super::accumulator::formatted_string_peak(usize::MAX).is_none());
    }

    #[test]
    fn accumulator_finalize_bounds_cover_large_collect_concat_and_percentile_scratch() {
        for function in [
            AggregateFunction::Collect,
            AggregateFunction::GroupConcat,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
        ] {
            let expr = expression(function, false);
            let mut state =
                AggregateState::new(function, false, Some(0.5), expr.separator.as_deref());
            for index in (0..2048).rev() {
                state.update(Some(Value::Int64(index)));
            }
            let peak = state.finalize_peak_bytes().unwrap();
            let output = state.finalize();
            assert!(aggregate_value_heap_bytes(&output).unwrap() <= peak);
            if matches!(
                function,
                AggregateFunction::PercentileDisc | AggregateFunction::PercentileCont
            ) {
                assert_eq!(peak, 2 * 2048 * size_of::<f64>());
                assert!(matches!(output, Value::Float64(_)));
            }
        }
    }

    #[test]
    fn accumulator_incremental_bounds_do_not_grow_for_repeated_distinct_identity() {
        let expr = expression(AggregateFunction::Collect, true);
        let mut state = AggregateState::new(expr.function, true, None, None);
        let input = nested_value();
        state.update(Some(input.clone()));
        let cached = state.retained_heap_bytes().unwrap();
        for _ in 0..128 {
            let before = state.retained_update_snapshot().unwrap();
            state.update(Some(input.clone()));
            let (removed, added) = state
                .retained_update_delta(before, &expr, Some(&input), None, None)
                .unwrap();
            assert_eq!(removed, added);
            assert_eq!(
                cached.checked_sub(removed).unwrap().checked_add(added),
                Some(cached)
            );
        }
        assert_eq!(state.retained_heap_bytes(), Some(cached));
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn accumulator_min_bound_includes_recursive_rdf_term_comparison_scratch() {
        fn tagged(mut visible: Value) -> Value {
            for _ in 0..16 {
                visible = Value::List(Arc::from([
                    visible,
                    Value::String("\"escaped\\u00E9 lexical\"@en".into()),
                    Value::String(grafeo_common::types::INTERNAL_RDF_TAGGED_TERM_MARKER.into()),
                ]));
            }
            visible
        }
        let expr = expression(AggregateFunction::Min, false);
        let initial = tagged(Value::Int64(7));
        let input = tagged(Value::Int64(3));
        let mut state = AggregateState::Min(Some(initial.clone()));
        let comparison = aggregate_rdf_comparison_peak(&initial, 0).unwrap()
            + aggregate_rdf_comparison_peak(&input, 0).unwrap();
        let peak = state
            .update_peak_bytes(&expr, Some(&input), None, None)
            .unwrap();
        assert!(peak >= comparison + input.retained_size_bytes().unwrap());
        assert!(comparison >= 32 * (96 + 6 * size_of::<usize>()));
        let snapshot = state.retained_update_snapshot().unwrap();
        let old_heap = state.retained_heap_bytes().unwrap();
        state.update(Some(input.clone()));
        assert_eq!(state.finalize(), input);
        let (removed, added) = state
            .retained_update_delta(snapshot, &expr, Some(&input), None, None)
            .unwrap();
        assert_eq!(
            old_heap - removed + added,
            state.retained_heap_bytes().unwrap()
        );
        assert_eq!(initial, tagged(Value::Int64(7)));
    }
}
