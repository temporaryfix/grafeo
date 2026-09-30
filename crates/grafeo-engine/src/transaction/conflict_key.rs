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

/// Optional property tag on a conflict key. `None` is the entity-level default
/// (and the only value produced under `ConflictGranularity::Entity`).
pub type PropTag = Option<u64>;

/// Reserved [`PropTag`] payload for a **structural / whole-entity read** recorded
/// under `ConflictGranularity::Property`.
///
/// Purely structural reads (label-scan visits, existence/label checks, `count`)
/// touch the entity without naming a property. Under `Property` granularity they
/// are tagged `Some(STRUCT_TAG)` rather than dropped (which would miss a
/// concurrent structural write — unsound) or recorded as `None` (which would
/// conflict with *every* property write, defeating the per-property knob).
///
/// Soundness/precision of `Some(STRUCT_TAG)`:
/// - vs a structural write (`DELETE`/label change, tagged `None`):
///   `prop_compatible(Some(STRUCT_TAG), None) == true` → the rw-antidependency is
///   **detected** (sound — a delete invalidates every read of the entity).
/// - vs a disjoint property write (e.g. `balance`, tagged `Some(prop_tag("balance"))`):
///   `STRUCT_TAG != prop_tag("balance")` → **no false conflict** (the knob holds —
///   a structural visit does not clash with a disjoint-property write).
///
/// The value is a fixed reserved sentinel. [`prop_tag`] is a full 64-bit hash, so
/// the probability any real property key hashes to exactly this value is ~2^-64
/// (negligible); a collision would only ever *merge* keys — a false conflict at
/// worst, never a missed one — so soundness is preserved regardless.
pub const STRUCT_TAG: u64 = 0xFFFF_FFFF_FFFF_FFFE;

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
