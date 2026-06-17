//! Conflict-key granularity: an optional property tag riding alongside `EntityId`.
//!
//! Part G's property-level conflict granularity tracks rw-antidependencies per
//! `(entity, property)` instead of per `entity`. The property is carried as a
//! [`PropTag`]:
//!
//! - `None` — a structural / whole-entity read or write (existence, labels,
//!   delete, or *any* read under the entity-level default granularity).
//! - `Some(hash)` — a specific property (a stable 64-bit hash of the property key).
//!
//! Two participants on the **same entity** actually conflict only if their tags
//! are [`prop_compatible`]. A structural (`None`) tag is a wildcard — it touches
//! the whole entity, so it conflicts with anything; two property tags conflict
//! iff equal. Hash collisions only ever *merge* keys (a safe over-approximation —
//! a false conflict at worst, never a missed one), so soundness is preserved.

// These items are consumed when the read-set/registry generalization lands
// (Part G Task 2); this module-level allow is removed there.
#![allow(dead_code)]

/// Optional property tag on a conflict key. `None` is the entity-level default
/// (and the only value produced under `ConflictGranularity::Entity`).
pub type PropTag = Option<u64>;

/// Whether two rw-conflict participants on the **same entity** actually conflict.
///
/// A structural (`None`) read/write touches the whole entity, so it is compatible
/// with (conflicts with) anything; two property tags conflict iff equal.
#[inline]
#[must_use]
pub fn prop_compatible(a: PropTag, b: PropTag) -> bool {
    a.is_none() || b.is_none() || a == b
}

/// Stable 64-bit tag for a property key.
///
/// Collisions only *merge* keys (a safe over-approximation — a false conflict at
/// worst, never a missed conflict), so this never makes SSI unsound.
#[inline]
#[must_use]
pub fn prop_tag(key: &str) -> u64 {
    grafeo_common::utils::hash::hash_one(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prop_compatible_none_is_wildcard_on_both_sides() {
        assert!(prop_compatible(None, None));
        assert!(prop_compatible(None, Some(1)));
        assert!(prop_compatible(Some(1), None));
    }

    #[test]
    fn prop_compatible_property_tags_conflict_iff_equal() {
        assert!(prop_compatible(Some(1), Some(1)));
        assert!(!prop_compatible(Some(1), Some(2)));
    }

    #[test]
    fn prop_tag_is_stable_and_distinguishing() {
        assert_eq!(prop_tag("balance"), prop_tag("balance"));
        assert_ne!(prop_tag("balance"), prop_tag("id"));
    }
}
