//! Typed property-index routing; candidates always require evaluator residuals.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::cmp::Ordering;
use std::ops::Bound;

use crate::graph::PropertyIndexPredicate;
use grafeo_common::types::{
    ArcStr, Date, HashableValue, OrderedFloat64, Time, Timestamp, Value, canonical_f64_bits,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Lane {
    Integer,
    Float,
    FloatNan,
    String,
    StringAsInt,
    StringAsFloat,
    Timestamp,
    Date,
    TimeLocalRaw,
    TimeOffsetRaw,
    TimeOffsetUtc,
}

/// Each lane uses exactly one scalar variant. In particular, integer and float
/// values never invoke the existing lossy cross-type OrderableValue comparator.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Scalar {
    Integer(i64),
    Float(OrderedFloat64),
    String(ArcStr),
    Timestamp(Timestamp),
    Date(Date),
    TimeNanos(u64),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Position {
    Minimum,
    Value(Scalar),
    Maximum,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Identity {
    Native,
    FloatBits(u64),
    String(ArcStr),
    Time(u64, Option<i32>),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Suffix {
    Lower,
    Source(Identity),
    Upper,
}

/// An ordered coordinate and its exact hash-bucket identity. Sentinels have no
/// source. Coordinates that collapse different spellings/NaNs/times retain an
/// identity suffix so no distinct source bucket disappears from the directory.
#[derive(Clone, Debug)]
pub(super) struct PropertyOrderKey {
    lane: Lane,
    position: Position,
    suffix: Suffix,
    source: Option<HashableValue>,
}

impl PropertyOrderKey {
    pub(super) fn source_key(&self) -> Option<&HashableValue> {
        self.source.as_ref()
    }

    fn sentinel(lane: Lane, position: Position, suffix: Suffix) -> Self {
        Self {
            lane,
            position,
            suffix,
            source: None,
        }
    }
}

impl Ord for PropertyOrderKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.lane
            .cmp(&other.lane)
            .then_with(|| self.position.cmp(&other.position))
            .then_with(|| self.suffix.cmp(&other.suffix))
    }
}
impl PartialOrd for PropertyOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for PropertyOrderKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for PropertyOrderKey {}

fn time_utc(time: Time) -> u64 {
    let offset = i128::from(time.offset_seconds().unwrap_or(0));
    let nanos =
        (i128::from(time.as_nanos()) - offset * 1_000_000_000).rem_euclid(86_400_000_000_000);
    // The Euclidean remainder lies in 0..nanoseconds_per_day and fits u64.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        nanos as u64
    }
}

/// Constructs only scalar routes. Composite equality requires a caller fallback:
/// the evaluator recursively coerces elements while exact hash equality does not.
pub(super) fn keys_for_value(value: &Value) -> Vec<PropertyOrderKey> {
    keys_for_value_fixed(value).into_iter().flatten().collect()
}

/// The preparation path uses stack storage: scalar Value/ArcStr clones do not
/// allocate, and no value has more than its native key plus two aliases.
pub(super) fn keys_for_value_fixed(value: &Value) -> [Option<PropertyOrderKey>; 3] {
    let mut keys = [None, None, None];
    let mut next = 0;
    let mut push = |lane, scalar, identity| {
        // The match below emits at most three keys (only numeric strings use
        // all three slots); no public caller can change that bound.
        keys[next] = Some(PropertyOrderKey {
            lane,
            position: Position::Value(scalar),
            suffix: Suffix::Source(identity),
            source: Some(HashableValue::new(value.clone())),
        });
        next += 1;
    };
    match value {
        Value::Int64(i) => push(Lane::Integer, Scalar::Integer(*i), Identity::Native),
        Value::Float64(f) => push(
            if f.is_nan() {
                Lane::FloatNan
            } else {
                Lane::Float
            },
            Scalar::Float(OrderedFloat64(*f)),
            Identity::FloatBits(canonical_f64_bits(*f)),
        ),
        Value::String(s) => {
            push(Lane::String, Scalar::String(s.clone()), Identity::Native);
            if let Ok(i) = s.parse::<i64>() {
                push(
                    Lane::StringAsInt,
                    Scalar::Integer(i),
                    Identity::String(s.clone()),
                );
            }
            if let Ok(f) = s.parse::<f64>()
                && !f.is_nan()
            {
                push(
                    Lane::StringAsFloat,
                    Scalar::Float(OrderedFloat64(f)),
                    Identity::String(s.clone()),
                );
            }
        }
        Value::Timestamp(t) => push(Lane::Timestamp, Scalar::Timestamp(*t), Identity::Native),
        Value::Date(d) => push(Lane::Date, Scalar::Date(*d), Identity::Native),
        Value::Time(t) => {
            let identity = Identity::Time(t.as_nanos(), t.offset_seconds());
            if t.offset_seconds().is_some() {
                push(
                    Lane::TimeOffsetRaw,
                    Scalar::TimeNanos(t.as_nanos()),
                    identity.clone(),
                );
                push(
                    Lane::TimeOffsetUtc,
                    Scalar::TimeNanos(time_utc(*t)),
                    identity,
                );
            } else {
                push(
                    Lane::TimeLocalRaw,
                    Scalar::TimeNanos(t.as_nanos()),
                    identity,
                );
            }
        }
        _ => {}
    }
    keys
}

type Route = (Bound<PropertyOrderKey>, Bound<PropertyOrderKey>);

/// A span uses inclusive sentinel endpoints, even when the scalar bound is
/// exclusive: Lower/Upper sort outside every original spelling at that scalar.
struct Span {
    lower: PropertyOrderKey,
    upper: PropertyOrderKey,
}

impl Span {
    fn all(lane: Lane) -> Self {
        Self {
            lower: PropertyOrderKey::sentinel(lane, Position::Minimum, Suffix::Lower),
            upper: PropertyOrderKey::sentinel(lane, Position::Maximum, Suffix::Upper),
        }
    }

    fn constrain(&mut self, scalar: Scalar, lower: bool, inclusive: bool) {
        let suffix = if lower == inclusive {
            Suffix::Lower
        } else {
            Suffix::Upper
        };
        let bound = PropertyOrderKey::sentinel(self.lower.lane, Position::Value(scalar), suffix);
        if lower {
            if bound > self.lower {
                self.lower = bound;
            }
        } else if bound < self.upper {
            self.upper = bound;
        }
    }

    fn append(self, routes: &mut Vec<Route>) {
        if self.lower < self.upper {
            routes.push((Bound::Included(self.lower), Bound::Included(self.upper)));
        }
    }
}

/// First integer for which a monotone false-to-true predicate holds. The i128
/// exclusive endpoint represents MAX+1 without saturation at the i64 boundary.
fn first_integer(mut predicate: impl FnMut(i64) -> bool) -> i128 {
    let mut low = i128::from(i64::MIN);
    let mut high = i128::from(i64::MAX) + 1;
    while low < high {
        let middle = low + (high - low) / 2;
        // middle is strictly below MAX+1 and at least MIN.
        #[allow(clippy::cast_possible_truncation)]
        let integer = middle as i64;
        if predicate(integer) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

fn constrain_integer_float(span: &mut Span, bound: f64, lower: bool, inclusive: bool) -> bool {
    if bound.is_nan() {
        return false;
    }
    let cut = if lower {
        first_integer(|i| {
            if inclusive {
                i as f64 >= bound
            } else {
                i as f64 > bound
            }
        })
    } else {
        first_integer(|i| {
            if inclusive {
                i as f64 > bound
            } else {
                i as f64 >= bound
            }
        }) - 1
    };
    if cut < i128::from(i64::MIN) || cut > i128::from(i64::MAX) {
        return false;
    }
    // Checked against the complete i64 domain above.
    #[allow(clippy::cast_possible_truncation)]
    span.constrain(Scalar::Integer(cut as i64), lower, true);
    true
}

fn float_span(lane: Lane, center: f64, routes: &mut Vec<Route>) {
    if !center.is_finite() {
        return;
    }
    let mut span = Span::all(lane);
    // Rounded subtraction/addition may shrink the mathematical EPSILON band.
    // One outward representable step is conservative, including near zero.
    span.constrain(
        Scalar::Float(OrderedFloat64((center - f64::EPSILON).next_down())),
        true,
        true,
    );
    span.constrain(
        Scalar::Float(OrderedFloat64((center + f64::EPSILON).next_up())),
        false,
        true,
    );
    span.append(routes);
}

fn integer_point(lane: Lane, value: i64, routes: &mut Vec<Route>) {
    let mut span = Span::all(lane);
    span.constrain(Scalar::Integer(value), true, true);
    span.constrain(Scalar::Integer(value), false, true);
    span.append(routes);
}

fn equality(value: &Value, routes: &mut Vec<Route>) {
    match value {
        Value::Int64(i) => {
            float_span(Lane::Float, *i as f64, routes);
            integer_point(Lane::StringAsInt, *i, routes);
        }
        Value::Float64(f) if f.is_finite() => {
            float_span(Lane::Float, *f, routes);
            float_span(Lane::StringAsFloat, *f, routes);
            let mut span = Span::all(Lane::Integer);
            if constrain_integer_float(&mut span, (*f - f64::EPSILON).next_down(), true, true)
                && constrain_integer_float(&mut span, (*f + f64::EPSILON).next_up(), false, true)
            {
                span.append(routes);
            }
        }
        Value::String(s) => {
            if let Ok(i) = s.parse::<i64>() {
                integer_point(Lane::Integer, i, routes);
            }
            if let Ok(f) = s.parse::<f64>() {
                float_span(Lane::Float, f, routes);
            }
        }
        _ => {}
    }
}

fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn apply_bound(span: &mut Span, value: &Value, lower: bool, inclusive: bool) -> bool {
    match span.lower.lane {
        Lane::Integer => match value {
            Value::Int64(i) => {
                span.constrain(Scalar::Integer(*i), lower, inclusive);
                true
            }
            _ => numeric(value).is_some_and(|f| constrain_integer_float(span, f, lower, inclusive)),
        },
        Lane::Float => {
            let Some(f) = numeric(value) else {
                return false;
            };
            if f.is_nan() {
                // Only native Float/Float comparison treats NaN as compare=0.
                return matches!(value, Value::Float64(_)) && inclusive;
            }
            span.constrain(Scalar::Float(OrderedFloat64(f)), lower, inclusive);
            true
        }
        Lane::FloatNan => matches!(value, Value::Float64(_)) && inclusive,
        Lane::String => {
            let Value::String(s) = value else {
                return false;
            };
            span.constrain(Scalar::String(s.clone()), lower, inclusive);
            true
        }
        Lane::StringAsFloat => match value {
            // String/string is lexical. A numeric bound selected this alias
            // lane; the lexical bound must be checked by the residual evaluator.
            Value::String(_) => true,
            Value::Int64(_) | Value::Float64(_) => {
                let Some(f) = numeric(value).filter(|f| !f.is_nan()) else {
                    return false;
                };
                span.constrain(Scalar::Float(OrderedFloat64(f)), lower, inclusive);
                true
            }
            _ => false,
        },
        Lane::Timestamp => {
            let Value::Timestamp(t) = value else {
                return false;
            };
            span.constrain(Scalar::Timestamp(*t), lower, inclusive);
            true
        }
        Lane::Date => {
            let Value::Date(d) = value else {
                return false;
            };
            span.constrain(Scalar::Date(*d), lower, inclusive);
            true
        }
        Lane::TimeLocalRaw | Lane::TimeOffsetRaw | Lane::TimeOffsetUtc => {
            let Value::Time(t) = value else {
                return false;
            };
            let utc = span.lower.lane == Lane::TimeOffsetUtc;
            if span.lower.lane != Lane::TimeLocalRaw && utc != t.offset_seconds().is_some() {
                // Offset/local bounds use different coordinates. Seek with one
                // coordinate, then evaluate the other bound on candidates.
                return true;
            }
            span.constrain(
                Scalar::TimeNanos(if utc { time_utc(*t) } else { t.as_nanos() }),
                lower,
                inclusive,
            );
            true
        }
        Lane::StringAsInt => false, // Equality-only alias lane.
    }
}

/// Produces scalar candidate ranges, never query truth. The parent must also
/// probe exact hash equality and run the real evaluator on every candidate.
/// Composite equality and a completely unbounded property request require a
/// caller path that covers nonordered values as well as this scalar directory.
pub(super) fn routes(predicate: PropertyIndexPredicate<'_>) -> Vec<Route> {
    let mut routes = Vec::new();
    match predicate {
        PropertyIndexPredicate::Equal(value) => equality(value, &mut routes),
        PropertyIndexPredicate::In(values) => {
            for value in values {
                equality(value, &mut routes);
            }
        }
        PropertyIndexPredicate::Range {
            min,
            max,
            min_inclusive,
            max_inclusive,
        } => {
            let bounds = [(min, true, min_inclusive), (max, false, max_inclusive)];
            let numeric_string_lane = [min, max]
                .into_iter()
                .flatten()
                .any(|v| matches!(v, Value::Int64(_) | Value::Float64(_)));
            let offset_time_lane = [min, max]
                .into_iter()
                .flatten()
                .any(|v| matches!(v, Value::Time(t) if t.offset_seconds().is_some()));
            for lane in [
                Lane::Integer,
                Lane::Float,
                Lane::FloatNan,
                if numeric_string_lane {
                    Lane::StringAsFloat
                } else {
                    Lane::String
                },
                Lane::Timestamp,
                Lane::Date,
                Lane::TimeLocalRaw,
                if offset_time_lane {
                    Lane::TimeOffsetUtc
                } else {
                    Lane::TimeOffsetRaw
                },
            ] {
                let mut span = Span::all(lane);
                if bounds.into_iter().all(|(value, lower, inclusive)| {
                    value.is_none_or(|v| apply_bound(&mut span, v, lower, inclusive))
                }) {
                    span.append(&mut routes);
                }
            }
        }
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::super::ordered::{OrderedDirectory, PreparedKey};
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::operators::{BinaryFilterOp, ExpressionPredicate, FilterExpression};
    use crate::graph::lpg::LpgStore;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn oracle(left: &Value, op: BinaryFilterOp, right: &Value, store: &Arc<LpgStore>) -> bool {
        let expression = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(left.clone())),
            op,
            right: Box::new(FilterExpression::Literal(right.clone())),
        };
        let predicate = ExpressionPredicate::new(expression, HashMap::new(), store.clone());
        matches!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::Bool(true))
        )
    }

    fn samples() -> Vec<Value> {
        let mut values: Vec<_> = [
            i64::MIN,
            i64::MIN + 1,
            -9007199254740993,
            -9007199254740992,
            -1,
            0,
            1,
            9007199254740991,
            9007199254740992,
            9007199254740993,
            i64::MAX - 1,
            i64::MAX,
        ]
        .into_iter()
        .map(Value::Int64)
        .collect();
        values.extend(
            [
                f64::NEG_INFINITY,
                -f64::MAX,
                i64::MIN as f64,
                -9007199254740992.0,
                -1.0,
                -f64::EPSILON,
                -f64::EPSILON / 2.0,
                -0.0,
                0.0,
                f64::from_bits(1),
                f64::EPSILON / 2.0,
                f64::EPSILON,
                1.0_f64.next_down(),
                1.0,
                1.0_f64.next_up(),
                9007199254740992.0_f64.next_down(),
                9007199254740992.0,
                9007199254740992.0_f64.next_up(),
                i64::MAX as f64,
                f64::MAX,
                f64::INFINITY,
                f64::NAN,
                f64::from_bits(0xfff8000000000001),
            ]
            .into_iter()
            .map(Value::Float64),
        );
        values.extend(
            [
                "0",
                "-0",
                "+0",
                "00",
                "0.0",
                "1",
                "01",
                "1.0",
                "1e0",
                "2",
                "10",
                "9007199254740992",
                "9007199254740993",
                "9223372036854775807",
                "9223372036854775808",
                "-9223372036854775808",
                "NaN",
                "inf",
                "-inf",
                "word",
                "",
            ]
            .into_iter()
            .map(|s| Value::String(s.into())),
        );
        values.extend([
            Value::Bool(true),
            Value::Null,
            Value::Timestamp(Timestamp::from_micros(1)),
            Value::Timestamp(Timestamp::from_micros(2)),
            Value::Date(Date::from_days(1)),
            Value::Date(Date::from_days(2)),
        ]);
        for hour in [0, 1, 2, 23] {
            let time = Time::from_hms(hour, 0, 0).unwrap();
            values.extend([
                Value::Time(time),
                Value::Time(time.with_offset(3600)),
                Value::Time(time.with_offset(-3600)),
            ]);
        }
        values
    }

    fn directory(values: &[Value]) -> OrderedDirectory<PropertyOrderKey> {
        let mut directory = OrderedDirectory::new();
        for value in values {
            for key in keys_for_value(value) {
                let _ = directory.insert_prepared(PreparedKey::new(key).unwrap());
            }
        }
        directory
    }

    fn candidates(
        directory: &OrderedDirectory<PropertyOrderKey>,
        predicate: PropertyIndexPredicate<'_>,
    ) -> Vec<HashableValue> {
        let mut found = Vec::new();
        for (lower, upper) in routes(predicate) {
            directory.visit(lower.as_ref(), upper.as_ref(), |key| {
                found.push(key.source_key().unwrap().clone());
                true
            });
        }
        found
    }

    #[test]
    fn routing_equality_includes_actual_evaluator_numeric_matches() {
        let values = samples();
        let directory = directory(&values);
        let store = Arc::new(LpgStore::new().unwrap());
        for probe in &values {
            let mut found = candidates(&directory, PropertyIndexPredicate::Equal(probe));
            found.push(HashableValue::new(probe.clone())); // Parent's mandatory exact probe.
            for candidate in &values {
                if oracle(candidate, BinaryFilterOp::Eq, probe, &store) {
                    assert!(
                        found.contains(&HashableValue::new(candidate.clone())),
                        "missing {candidate:?} = {probe:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn routing_ranges_include_actual_evaluator_matches() {
        let values = samples();
        let directory = directory(&values);
        let store = Arc::new(LpgStore::new().unwrap());
        for probe in &values {
            for (lower, inclusive, op) in [
                (true, true, BinaryFilterOp::Ge),
                (true, false, BinaryFilterOp::Gt),
                (false, true, BinaryFilterOp::Le),
                (false, false, BinaryFilterOp::Lt),
            ] {
                let found = candidates(
                    &directory,
                    PropertyIndexPredicate::Range {
                        min: lower.then_some(probe),
                        max: (!lower).then_some(probe),
                        min_inclusive: inclusive,
                        max_inclusive: inclusive,
                    },
                );
                for candidate in &values {
                    if oracle(candidate, op, probe, &store) {
                        assert!(
                            found.contains(&HashableValue::new(candidate.clone())),
                            "missing {candidate:?} {op:?} {probe:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn routing_intersects_bounds_without_mixing_string_or_time_coordinates() {
        let values = samples();
        let directory = directory(&values);
        let store = Arc::new(LpgStore::new().unwrap());
        let pairs = [
            (Value::String("1".into()), Value::Int64(2)),
            (
                Value::Float64(9007199254740992.0),
                Value::Int64(9007199254740993),
            ),
            (Value::Float64(f64::NAN), Value::Float64(1.0)),
            (
                Value::Time(Time::from_hms(0, 0, 0).unwrap().with_offset(3600)),
                Value::Time(Time::from_hms(2, 0, 0).unwrap()),
            ),
        ];
        for (min, max) in pairs {
            for inclusive in [false, true] {
                let found = candidates(
                    &directory,
                    PropertyIndexPredicate::Range {
                        min: Some(&min),
                        max: Some(&max),
                        min_inclusive: inclusive,
                        max_inclusive: inclusive,
                    },
                );
                for candidate in &values {
                    if oracle(
                        candidate,
                        if inclusive {
                            BinaryFilterOp::Ge
                        } else {
                            BinaryFilterOp::Gt
                        },
                        &min,
                        &store,
                    ) && oracle(
                        candidate,
                        if inclusive {
                            BinaryFilterOp::Le
                        } else {
                            BinaryFilterOp::Lt
                        },
                        &max,
                        &store,
                    ) {
                        assert!(
                            found.contains(&HashableValue::new(candidate.clone())),
                            "missing {min:?} .. {candidate:?} .. {max:?}"
                        );
                    }
                }
            }
        }
        let min = Value::Int64(9);
        let max = Value::Int64(1);
        assert!(
            routes(PropertyIndexPredicate::Range {
                min: Some(&min),
                max: Some(&max),
                min_inclusive: true,
                max_inclusive: true
            })
            .is_empty()
        );
    }

    #[test]
    fn routing_order_is_lawful_and_suffix_bounds_cover_alias_spellings() {
        let values = samples();
        let keys: Vec<_> = values.iter().flat_map(keys_for_value).collect();
        for a in &keys {
            for b in &keys {
                assert_eq!(a.cmp(b), b.cmp(a).reverse());
                assert_eq!(a == b, a.cmp(b).is_eq());
                if a == b {
                    assert_eq!(a.source_key(), b.source_key());
                }
                for c in &keys {
                    if a <= b && b <= c {
                        assert!(a <= c);
                    }
                }
            }
        }
        let found = candidates(
            &directory(&values),
            PropertyIndexPredicate::Equal(&Value::Int64(0)),
        );
        for alias in ["0", "-0", "+0", "00"] {
            assert!(found.contains(&HashableValue::new(Value::String(alias.into()))));
        }
    }

    #[test]
    fn routing_bounded_numeric_seek_visits_only_selected_native_and_alias_keys() {
        let values: Vec<_> = (0..512)
            .flat_map(|i| [Value::Int64(i), Value::String(i.to_string().into())])
            .collect();
        let directory = directory(&values);
        let bound = Value::Int64(256);
        let found = candidates(
            &directory,
            PropertyIndexPredicate::Range {
                min: Some(&bound),
                max: Some(&bound),
                min_inclusive: true,
                max_inclusive: true,
            },
        );
        assert_eq!(
            found.len(),
            2,
            "range routing must not enumerate every numeric key"
        );
        assert!(found.contains(&HashableValue::new(bound.clone())));
        assert!(found.contains(&HashableValue::new(Value::String("256".into()))));
        let probes = [Value::Int64(256), Value::String("257.0".into())];
        let found = candidates(&directory, PropertyIndexPredicate::In(&probes));
        assert!(found.contains(&HashableValue::new(Value::String("256".into()))));
        // String-to-integer equality parses i64, so a decimal spelling must not
        // route an integer bucket just because its f64 parse is integral.
        assert!(!found.contains(&HashableValue::new(Value::Int64(257))));
    }

    #[test]
    fn routing_fixed_preparation_has_no_allocator_traffic() {
        let values = samples();
        let mut owned = Vec::with_capacity(values.len());
        crate::allocation_test::start();
        for value in &values {
            owned.push(keys_for_value_fixed(value));
        }
        let traffic = crate::allocation_test::stop();
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        for (value, keys) in values.iter().zip(owned) {
            assert_eq!(
                keys.into_iter().flatten().collect::<Vec<_>>(),
                keys_for_value(value)
            );
        }
    }

    #[test]
    fn routing_fixed_keys_preserve_every_native_and_alias_coordinate() {
        let offset = Time::from_hms(2, 0, 0).unwrap().with_offset(3600);
        let fixtures = [
            (
                Value::String("01".into()),
                vec![Lane::String, Lane::StringAsInt, Lane::StringAsFloat],
            ),
            (
                Value::String("1.0".into()),
                vec![Lane::String, Lane::StringAsFloat],
            ),
            (Value::String("word".into()), vec![Lane::String]),
            (
                Value::Time(offset),
                vec![Lane::TimeOffsetRaw, Lane::TimeOffsetUtc],
            ),
            (
                Value::Time(Time::from_hms(2, 0, 0).unwrap()),
                vec![Lane::TimeLocalRaw],
            ),
        ];
        for (value, expected) in fixtures {
            let fixed = keys_for_value_fixed(&value);
            let keys: Vec<_> = fixed.into_iter().flatten().collect();
            assert_eq!(
                keys.iter().map(|key| key.lane).collect::<Vec<_>>(),
                expected
            );
            for key in keys {
                assert_eq!(key.source_key(), Some(&HashableValue::new(value.clone())));
            }
        }
        let values = [Value::String("01".into()), Value::Time(offset)];
        let directory = directory(&values);
        for bound in [
            Value::String("01".into()),
            Value::Int64(1),
            Value::Time(Time::from_hms(2, 0, 0).unwrap()),
        ] {
            let found = candidates(
                &directory,
                PropertyIndexPredicate::Range {
                    min: Some(&bound),
                    max: Some(&bound),
                    min_inclusive: true,
                    max_inclusive: true,
                },
            );
            let expected = if matches!(bound, Value::Time(_)) {
                &values[1]
            } else {
                &values[0]
            };
            assert!(
                found.contains(&HashableValue::new(expected.clone())),
                "missing coordinate for {bound:?}"
            );
        }
    }
}
