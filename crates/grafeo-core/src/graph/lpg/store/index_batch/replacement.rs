//! One continuously fenced data/registry replacement at an existing store.

use super::{
    DataRebindError, IndexRegistryWorkspace, RegistryAuthority, conflict, prepare_under_authority,
};
use crate::graph::lpg::store::LpgReplacementWorkspace;
use grafeo_common::utils::error::Error;

/// Prepares and publishes an exact replacement without changing the root Arc.
///
/// The registry workspace must borrow independently retained graph anchors and
/// include the target root even when it has no index edits. All incoming index
/// contents and displaced payloads remain owned outside publication guards.
/// The preparation callback may fail; the publication callback must only install
/// already prepared companions, without allocation, validation or lock entry.
///
/// # Errors
/// Rejects missing root enrollment, stale registrations, denied authority and
/// any backing or companion preparation failure before changing live state.
pub fn with_prepared_lpg_replacement<T, R, E>(
    data: &mut LpgReplacementWorkspace,
    indexes: &mut IndexRegistryWorkspace<'_>,
    prepare_companions: impl FnOnce() -> std::result::Result<T, E>,
    publish_companions: impl FnOnce(T) -> R,
) -> std::result::Result<R, E>
where
    E: From<Error> + From<DataRebindError>,
{
    // Borrow the root from the independent registry anchors, never from data:
    // the data workspace must remain mutably borrowable for its backing swap.
    let target = indexes
        .registry
        .stores
        .iter()
        .find(|pending| std::ptr::eq(pending.store, data.target().as_ref()))
        .map(|pending| pending.store)
        .ok_or_else(|| conflict("replacement root is not enrolled"))?;
    indexes.prepare_inputs()?;
    let IndexRegistryWorkspace {
        registry,
        authority,
    } = indexes;
    let authority = RegistryAuthority::acquire(&mut registry.stores, authority)?;
    let retained = &*authority.workspace;
    data.validate_under_transitions(retained.retained_transitions())?;
    let transition = retained.transition(target)?;
    data.with_prepared_replacement(
        transition,
        move || {
            let released = prepare_under_authority(registry, retained)?;
            let prepared = released.rebind()?;
            let companions = prepare_companions()?;
            Ok((prepared, companions))
        },
        |(mut prepared, companions)| {
            // The borrowed fence retains the same registry/Text guards as the
            // standalone InstalledIndexRegistryFence. Its authority is owned
            // by this aggregate driver across the data and companion swap.
            prepared.fences.install();
            let result = publish_companions(companions);
            drop(prepared);
            result
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::lpg::{
        IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey, LpgStore, StoreIndexEdits,
    };
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use grafeo_common::types::{NodeId, PropertyKey, Value};
    use grafeo_common::utils::error::Result;
    use std::sync::Arc;

    #[derive(Debug)]
    struct PreparationFailed;

    impl From<Error> for PreparationFailed {
        fn from(_: Error) -> Self {
            Self
        }
    }

    impl From<DataRebindError> for PreparationFailed {
        fn from(_: DataRebindError) -> Self {
            Self
        }
    }

    fn incoming_property() -> IndexRegistryEdit {
        IndexRegistryEdit::Create {
            key: IndexRegistryKey::Property(PropertyKey::new("incoming")),
            contents: IndexRegistryContents::Property(vec![(NodeId::new(0), Value::from("new"))]),
        }
    }

    #[test]
    fn replacement_companion_failure_keeps_data_and_registries() -> Result<()> {
        let target = Arc::new(LpgStore::new()?);
        let old = target.create_node(&["Old"]);
        target.create_property_index("old");
        let authority = WriteAuthority::new();
        assert!(target.seal_unframed_writes(&authority));
        let candidate = LpgStore::new()?;
        candidate.create_node(&["New"]);
        candidate.create_node(&["New"]);
        let mut data = LpgReplacementWorkspace::new(Arc::clone(&target), candidate)?;
        let mut indexes = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &target,
            edits: vec![incoming_property()],
        }]);
        let result = with_authority(&authority, || {
            with_prepared_lpg_replacement(
                &mut data,
                &mut indexes,
                || Err::<(), _>(PreparationFailed),
                |()| (),
            )
        });
        assert!(result.is_err());
        assert_eq!(target.node_count(), 1);
        assert!(target.get_node(old).is_some());
        assert!(target.has_property_index("old"));
        assert!(!target.has_property_index("incoming"));
        assert!(target.property_indexes.try_write().is_some());
        Ok(())
    }

    #[test]
    fn replacement_retains_data_and_registry_writers_through_publication() -> Result<()> {
        let target = Arc::new(LpgStore::new()?);
        target.create_node(&["Old"]);
        let authority = WriteAuthority::new();
        assert!(target.seal_unframed_writes(&authority));
        let candidate = LpgStore::new()?;
        candidate.create_node(&["New"]);
        candidate.create_node(&["New"]);
        let mut data = LpgReplacementWorkspace::new(Arc::clone(&target), candidate)?;
        let mut indexes = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &target,
            edits: vec![incoming_property()],
        }]);
        let result = with_authority(&authority, || {
            with_prepared_lpg_replacement(
                &mut data,
                &mut indexes,
                || Ok::<_, PreparationFailed>(()),
                |()| {
                    assert!(target.property_indexes.try_read().is_none());
                    assert!(target.node_labels.try_read().is_none());
                },
            )
        });
        assert!(result.is_ok());
        assert_eq!(target.node_count(), 2);
        assert!(target.has_property_index("incoming"));
        assert!(target.property_indexes.try_write().is_some());
        Ok(())
    }

    #[test]
    fn replacement_requires_root_enrollment_even_without_indexes() -> Result<()> {
        let target = Arc::new(LpgStore::new()?);
        target.create_node(&["Old"]);
        let authority = WriteAuthority::new();
        assert!(target.seal_unframed_writes(&authority));
        let mut data = LpgReplacementWorkspace::new(Arc::clone(&target), LpgStore::new()?)?;
        let mut indexes = IndexRegistryWorkspace::new(Vec::new());
        let result = with_authority(&authority, || {
            with_prepared_lpg_replacement(
                &mut data,
                &mut indexes,
                || Ok::<_, PreparationFailed>(()),
                |()| (),
            )
        });
        assert!(result.is_err());
        assert_eq!(target.node_count(), 1);
        Ok(())
    }
}
