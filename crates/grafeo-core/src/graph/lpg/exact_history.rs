//! Shared structural-history invariants for exact LPG persistence.
//!
//! These predicates do not resolve identities or choose an endpoint policy.
//! Complete images resolve endpoints locally; transported snapshots resolve
//! them over their admitted sibling union before testing lifetime coverage.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};

/// One retained structural incarnation, with an exclusive deletion boundary.
/// A same-epoch create/delete is legal but is never visible.
pub type Lifetime = (EpochId, Option<EpochId>);

/// Validates a nonempty, ordered structural history at a committed cut.
///
/// Adjacent and zero-width incarnations are legal; overlapping incarnations,
/// an incarnation following an open one, and uncommitted epochs are not.
///
/// # Errors
///
/// Returns a serialization error for a malformed history or a PENDING cut.
pub fn validate_lifetimes(entity: &str, lives: &[Lifetime], cut: EpochId) -> Result<()> {
    if cut == EpochId::PENDING {
        return Err(Error::Serialization(format!(
            "{entity} committed cut cannot be PENDING"
        )));
    }
    if lives.is_empty() {
        return Err(Error::Serialization(format!(
            "{entity} has no structural lifetimes"
        )));
    }
    for (index, &(created, deleted)) in lives.iter().enumerate() {
        validate_epoch(entity, created, cut)?;
        if let Some(deleted) = deleted {
            validate_epoch(entity, deleted, cut)?;
            if deleted < created {
                return Err(Error::Serialization(format!(
                    "{entity} lifetime {index} has invalid delete epoch {deleted}: must satisfy created <= deleted"
                )));
            }
        }
        if let Some((_, previous_deleted)) = index.checked_sub(1).and_then(|i| lives.get(i)) {
            let Some(previous_deleted) = previous_deleted else {
                return Err(Error::Serialization(format!(
                    "{entity} lifetime {index} follows an open lifetime"
                )));
            };
            if created < *previous_deleted {
                return Err(Error::Serialization(format!(
                    "{entity} lifetime {index} overlaps its predecessor"
                )));
            }
        }
    }
    Ok(())
}

fn validate_epoch(entity: &str, epoch: EpochId, cut: EpochId) -> Result<()> {
    if epoch == EpochId::PENDING {
        return Err(Error::Serialization(format!(
            "{entity} structural history contains PENDING"
        )));
    }
    if epoch > cut {
        return Err(Error::Serialization(format!(
            "{entity} structural history reaches epoch {epoch}, beyond committed epoch {cut}"
        )));
    }
    Ok(())
}

/// Tests membership in validated structural lifetimes.
///
/// Set `include_delete_boundary` for retained property/label history entries:
/// tombstones at deletion must persist even though the entity is not visible.
/// The caller validates the queried epoch against its committed cut separately.
#[must_use]
pub fn epoch_is_inside_lifetime(
    epoch: EpochId,
    lives: &[Lifetime],
    include_delete_boundary: bool,
) -> bool {
    lives.iter().any(|(created, deleted)| {
        *created <= epoch
            && deleted.is_none_or(|deleted| {
                if include_delete_boundary {
                    epoch <= deleted
                } else {
                    epoch < deleted
                }
            })
    })
}

/// Tests whether one validated lifetime fits within one endpoint incarnation.
///
/// Both arguments must already have passed structural validation. Coverage
/// never bridges two endpoint incarnations, even when their boundaries touch.
/// A zero-width lifetime at an endpoint's deletion boundary remains admissible.
#[must_use]
pub fn lifetime_is_covered_by(life: Lifetime, endpoint_lives: &[Lifetime]) -> bool {
    endpoint_lives.iter().any(|(created, deleted)| {
        *created <= life.0
            && match (life.1, *deleted) {
                (None, None) => true,
                (Some(inner_deleted), Some(outer_deleted)) => inner_deleted <= outer_deleted,
                (Some(_), None) => true,
                (None, Some(_)) => false,
            }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lives(values: &[(u64, Option<u64>)]) -> Vec<Lifetime> {
        values
            .iter()
            .map(|&(created, deleted)| (EpochId::new(created), deleted.map(EpochId::new)))
            .collect()
    }

    #[test]
    fn structural_history_accepts_zero_width_and_adjacent_incarnations() {
        for history in [
            vec![(1, None)],
            vec![(3, Some(3))],
            vec![(1, Some(3)), (3, None)],
        ] {
            validate_lifetimes("node", &lives(&history), EpochId::new(5))
                .expect("valid structural history");
        }
    }

    #[test]
    fn structural_history_rejects_empty_overlapping_and_uncommitted_lives() {
        for history in [
            vec![],
            vec![(3, Some(2))],
            vec![(1, Some(4)), (3, None)],
            vec![(1, None), (3, None)],
            vec![(6, None)],
            vec![(1, Some(6))],
            vec![(u64::MAX, None)],
            vec![(1, Some(u64::MAX))],
        ] {
            assert!(
                validate_lifetimes("node", &lives(&history), EpochId::new(5)).is_err(),
                "{history:?}"
            );
        }
        assert!(validate_lifetimes("node", &lives(&[(1, None)]), EpochId::PENDING).is_err());
    }

    #[test]
    fn deletion_boundary_entries_survive_without_making_entities_visible() {
        let history = lives(&[(1, Some(3)), (4, Some(4))]);
        for (epoch, visible, retained_entry) in [
            (0, false, false),
            (1, true, true),
            (2, true, true),
            (3, false, true),
            (4, false, true),
            (5, false, false),
        ] {
            assert_eq!(
                epoch_is_inside_lifetime(EpochId::new(epoch), &history, false),
                visible
            );
            assert_eq!(
                epoch_is_inside_lifetime(EpochId::new(epoch), &history, true),
                retained_entry
            );
        }
        assert!(!epoch_is_inside_lifetime(EpochId::new(1), &[], true));
    }

    #[test]
    fn endpoint_coverage_never_bridges_gaps_or_recreation_boundaries() {
        let edge = (EpochId::new(2), Some(EpochId::new(5)));
        assert!(!lifetime_is_covered_by(
            edge,
            &lives(&[(1, Some(3)), (4, None)])
        ));
        assert!(!lifetime_is_covered_by(
            edge,
            &lives(&[(1, Some(3)), (3, None)])
        ));
        assert!(!lifetime_is_covered_by(edge, &[]));
    }

    #[test]
    fn endpoint_coverage_preserves_inclusive_history_and_open_lifetimes() {
        for (edge, endpoint, expected) in [
            ((1, Some(3)), (1, Some(3)), true),
            ((3, Some(3)), (1, Some(3)), true),
            ((1, None), (1, Some(3)), false),
            ((2, None), (1, None), true),
            ((2, Some(5)), (1, None), true),
            ((0, Some(2)), (1, None), false),
            ((2, Some(4)), (1, Some(3)), false),
        ] {
            let edge = (EpochId::new(edge.0), edge.1.map(EpochId::new));
            assert_eq!(lifetime_is_covered_by(edge, &lives(&[endpoint])), expected);
        }
    }
}
