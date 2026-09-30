//! Shared value comparison and conversion utilities.
//!
//! Used by pull-based and push-based aggregate, filter, and sort operators
//! to avoid duplicating comparison logic across six different files.

use std::cmp::Ordering;

#[cfg(feature = "triple-store")]
use grafeo_common::types::INTERNAL_RDF_TAGGED_TERM_MARKER;
use grafeo_common::types::Value;

#[cfg(feature = "triple-store")]
use crate::graph::rdf::Term;

use super::sort::NullOrder;

/// Converts a value to `f64` for numeric aggregations.
///
/// Supports RDF values stored as strings by attempting numeric parsing.
pub fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        // RDF stores numeric literals as strings - try to parse them
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Compares two values with partial ordering (returns `None` for incomparable types).
///
/// Handles cross-type comparisons between Int64/Float64/String, including
/// RDF numeric strings that need parsing before comparison.
pub fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => {
            // Try numeric comparison first if both look like numbers
            if let (Ok(a_num), Ok(b_num)) = (a.parse::<f64>(), b.parse::<f64>()) {
                a_num.partial_cmp(&b_num)
            } else {
                Some(a.cmp(b))
            }
        }
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        // String-to-numeric comparisons for RDF
        (Value::String(s), Value::Int64(i)) => s.parse::<f64>().ok()?.partial_cmp(&(*i as f64)),
        (Value::String(s), Value::Float64(f)) => s.parse::<f64>().ok()?.partial_cmp(f),
        (Value::Int64(i), Value::String(s)) => (*i as f64).partial_cmp(&s.parse::<f64>().ok()?),
        (Value::Float64(f), Value::String(s)) => f.partial_cmp(&s.parse::<f64>().ok()?),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        // Internal RDF mutation aggregates carry sealed tagged pairs. Do not
        // make ordinary user lists orderable as a side effect.
        #[cfg(feature = "triple-store")]
        (Value::List(a), Value::List(b)) => compare_tagged_rdf_terms(a, b),
        _ => None,
    }
}

#[cfg(feature = "triple-store")]
fn compare_tagged_rdf_terms(a: &[Value], b: &[Value]) -> Option<Ordering> {
    let (a_visible, a_exact, a_term) = decode_tagged_rdf_term(a)?;
    let (b_visible, b_exact, b_term) = decode_tagged_rdf_term(b)?;

    let kind_rank = |term: &Term| match term {
        Term::BlankNode(_) => 0,
        Term::Iri(_) => 1,
        Term::Literal(_) => 2,
    };
    let kind_order = kind_rank(&a_term).cmp(&kind_rank(&b_term));
    if kind_order != Ordering::Equal {
        return Some(kind_order);
    }

    match (&a_term, &b_term) {
        (Term::BlankNode(a), Term::BlankNode(b)) => Some(a.id().cmp(b.id())),
        (Term::Iri(a), Term::Iri(b)) => Some(a.as_str().cmp(b.as_str())),
        (Term::Literal(_), Term::Literal(_)) => match compare_values(a_visible, b_visible) {
            Some(Ordering::Equal) | None => Some(a_exact.cmp(b_exact)),
            ordering => ordering,
        },
        _ => Some(a_exact.cmp(b_exact)),
    }
}

#[cfg(feature = "triple-store")]
fn decode_tagged_rdf_term(values: &[Value]) -> Option<(&Value, &str, Term)> {
    let [visible, Value::String(exact), Value::String(marker)] = values else {
        return None;
    };
    if marker.as_str() != INTERNAL_RDF_TAGGED_TERM_MARKER {
        return None;
    }
    let term = Term::from_ntriples(exact.as_str())?;
    Some((visible, exact.as_str(), term))
}

/// Compares two values with total ordering (returns `Equal` for incomparable types).
///
/// Used by sort operators where a total order is required.
pub fn compare_values_total(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Int64(a), Value::Float64(b)) => {
            (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
        }
        (Value::Float64(a), Value::Int64(b)) => {
            a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
        }
        (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
        (Value::Date(a), Value::Date(b)) => a.cmp(b),
        (Value::Time(a), Value::Time(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Compares two optional values with null handling.
///
/// Used by sort and top-K operators to handle the `NULLS FIRST` / `NULLS LAST`
/// directive uniformly. Both `None` and `Some(Value::Null)` are treated as null.
pub fn compare_values_with_nulls(
    a: &Option<Value>,
    b: &Option<Value>,
    null_order: NullOrder,
) -> Ordering {
    match (a, b) {
        (None, None) | (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
        (None, _) | (Some(Value::Null), _) => match null_order {
            NullOrder::NullsFirst => Ordering::Less,
            NullOrder::NullsLast => Ordering::Greater,
        },
        (_, None) | (_, Some(Value::Null)) => match null_order {
            NullOrder::NullsFirst => Ordering::Greater,
            NullOrder::NullsLast => Ordering::Less,
        },
        (Some(a), Some(b)) => compare_values_total(a, b),
    }
}

/// Returns `true` if `new` is less than `current` (for MIN aggregation).
///
/// Returns `true` when `current` is `None` (first value always wins).
pub fn is_less_than(current: &Option<Value>, new: &Value) -> bool {
    match current {
        None => true,
        Some(curr) => compare_values(new, curr) == Some(Ordering::Less),
    }
}

/// Returns `true` if `new` is greater than `current` (for MAX aggregation).
///
/// Returns `true` when `current` is `None` (first value always wins).
pub fn is_greater_than(current: &Option<Value>, new: &Value) -> bool {
    match current {
        None => true,
        Some(curr) => compare_values(new, curr) == Some(Ordering::Greater),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_to_f64_int() {
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
    }

    #[test]
    fn value_to_f64_float() {
        assert_eq!(value_to_f64(&Value::Float64(2.72)), Some(2.72));
    }

    #[test]
    fn value_to_f64_numeric_string() {
        assert_eq!(value_to_f64(&Value::String("2.5".into())), Some(2.5));
    }

    #[test]
    fn value_to_f64_non_numeric_string() {
        assert_eq!(value_to_f64(&Value::String("abc".into())), None);
    }

    #[test]
    fn value_to_f64_null() {
        assert_eq!(value_to_f64(&Value::Null), None);
    }

    #[test]
    fn compare_same_type_int() {
        assert_eq!(
            compare_values(&Value::Int64(1), &Value::Int64(2)),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn compare_cross_type_int_float() {
        assert_eq!(
            compare_values(&Value::Int64(2), &Value::Float64(2.0)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn compare_rdf_numeric_strings() {
        assert_eq!(
            compare_values(&Value::String("10".into()), &Value::String("9".into())),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn compare_incomparable() {
        assert_eq!(compare_values(&Value::Bool(true), &Value::Int64(1)), None);
    }

    #[test]
    fn ordinary_lists_remain_incomparable() {
        let left = Value::List(vec![Value::Int64(1)].into());
        let right = Value::List(vec![Value::Int64(2)].into());
        assert_eq!(compare_values(&left, &right), None);
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn sealed_rdf_pairs_compare_by_term_kind_and_preserve_exact_ties() {
        let tagged = |visible: Value, exact: &str| {
            Value::List(
                vec![
                    visible,
                    Value::String(exact.into()),
                    Value::String(INTERNAL_RDF_TAGGED_TERM_MARKER.into()),
                ]
                .into(),
            )
        };
        let blank = tagged(Value::String("same".into()), "_:same");
        let iri = tagged(Value::String("same".into()), "<same>");
        let literal = tagged(Value::String("same".into()), "\"same\"");
        let lang = tagged(Value::String("same".into()), "\"same\"@en");

        assert_eq!(compare_values(&blank, &iri), Some(Ordering::Less));
        assert_eq!(compare_values(&iri, &literal), Some(Ordering::Less));
        assert_ne!(compare_values(&literal, &lang), Some(Ordering::Equal));
    }

    #[test]
    fn total_ordering_incomparable_returns_equal() {
        assert_eq!(
            compare_values_total(&Value::Bool(true), &Value::Int64(1)),
            Ordering::Equal
        );
    }

    #[test]
    fn is_less_than_none_always_true() {
        assert!(is_less_than(&None, &Value::Int64(5)));
    }

    #[test]
    fn is_less_than_smaller() {
        assert!(is_less_than(&Some(Value::Int64(10)), &Value::Int64(5)));
    }

    #[test]
    fn is_less_than_larger() {
        assert!(!is_less_than(&Some(Value::Int64(3)), &Value::Int64(5)));
    }

    #[test]
    fn is_greater_than_none_always_true() {
        assert!(is_greater_than(&None, &Value::Int64(5)));
    }

    #[test]
    fn is_greater_than_larger() {
        assert!(is_greater_than(&Some(Value::Int64(3)), &Value::Int64(5)));
    }

    #[test]
    fn is_greater_than_smaller() {
        assert!(!is_greater_than(&Some(Value::Int64(10)), &Value::Int64(5)));
    }
}
