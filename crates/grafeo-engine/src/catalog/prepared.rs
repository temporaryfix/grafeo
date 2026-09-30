//! Consuming logical catalog preparation under one retained catalog writer.
//!
//! Workspaces own candidates and retired payloads outside publication-authority
//! lifecycle, write-authority, RDF, publication, current-schema and catalog
//! gates. Per-Session operation/historical gates may remain held by an outer
//! invocation while that owned metadata retires. Proof objects borrow the
//! workspace; dropping a proof releases only its guard, never the payload.

#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
use std::ops::Deref;

#[cfg(any(feature = "lpg", test))]
use super::state::CatalogStateLock;
#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
use parking_lot::RwLockReadGuard;
use parking_lot::RwLockWriteGuard;
use std::sync::Arc;

#[cfg(any(all(feature = "lpg", feature = "wal"), test))]
use super::CatalogWalStateV1;
use super::{
    Catalog, CatalogError, CatalogState, CatalogStateSnapshot, EdgeTypeCatalog, IndexCatalog,
    LabelCatalog, PropertyCatalog, SchemaState,
};

/// A read of one already-held catalog cut, with no lock acquisition.
#[derive(Clone, Copy)]
#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
pub(crate) struct CatalogRead<'a> {
    state: &'a CatalogState,
}

#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
impl Deref for CatalogRead<'_> {
    type Target = CatalogState;
    fn deref(&self) -> &Self::Target {
        self.state
    }
}

#[cfg(feature = "lpg")]
impl CatalogRead<'_> {
    /// Resolves one exact natural physical owner in this already-held cut.
    pub(crate) fn physical_index_owner(
        self,
        key: &grafeo_core::graph::lpg::PhysicalIndexKey,
    ) -> Option<super::IndexDefinition> {
        let owner = self.state.indexes.physical_owners.get(key)?;
        self.state.indexes.get(*owner)
    }
}

/// The public-catalog read guard owns exactly one acquisition.
#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
pub(crate) struct CatalogReadGuard<'a> {
    guard: RwLockReadGuard<'a, Arc<CatalogState>>,
}

#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql"))]
impl CatalogReadGuard<'_> {
    pub(crate) fn view(&self) -> CatalogRead<'_> {
        CatalogRead { state: &self.guard }
    }
}

#[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
impl Deref for CatalogReadGuard<'_> {
    type Target = CatalogState;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

/// Outer ownership, initially vacant or explicitly populated by a decoder.
///
/// The optional field represents these real preparation states. Mandatory
/// ready/installed evidence below contains direct borrows, never an Option.
pub(crate) struct CatalogWorkspace {
    candidate: Option<Catalog>,
}

impl CatalogWorkspace {
    /// Keep failed and successful private DDL payloads outside authority gates.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn transaction_candidate(&mut self, source: &Catalog) -> &Catalog {
        self.candidate.insert(source.snapshot())
    }

    #[cfg(any(feature = "lpg", test))]
    pub(crate) fn new() -> Self {
        Self { candidate: None }
    }
    pub(crate) fn replacement(candidate: Catalog) -> Self {
        Self {
            candidate: Some(candidate),
        }
    }
}

/// One writer plus a uniquely owned detached candidate being edited.
#[cfg(any(feature = "lpg", test))]
pub(crate) struct CatalogEdit<'catalog, 'workspace> {
    live: RwLockWriteGuard<'catalog, Arc<CatalogState>>,
    candidate: &'workspace mut Catalog,
}

#[cfg(any(feature = "lpg", test))]
impl<'catalog, 'workspace> CatalogEdit<'catalog, 'workspace> {
    #[cfg(feature = "lpg")]
    pub(crate) fn view(&self) -> CatalogRead<'_> {
        CatalogRead { state: &self.live }
    }

    pub(crate) fn candidate(&self) -> &Catalog {
        self.candidate
    }

    pub(crate) fn finish(self) -> ReadyCatalog<'catalog, 'workspace> {
        ReadyCatalog {
            live: self.live,
            candidate: self.candidate.state.inner.get_mut(),
        }
    }
}

/// All fallible logical construction is complete; its candidate cannot mutate.
pub(crate) struct ReadyCatalog<'catalog, 'workspace> {
    live: RwLockWriteGuard<'catalog, Arc<CatalogState>>,
    candidate: &'workspace mut Arc<CatalogState>,
}

impl<'catalog, 'workspace> ReadyCatalog<'catalog, 'workspace> {
    /// Metadata publishes before the owner envelope during recovery. Preserve
    /// its exact owner preimage; only IndexOwnerBatch changes logical owners.
    #[cfg(all(feature = "lpg", feature = "wal"))]
    pub(crate) fn encode_transaction_metadata(&self) -> Result<Vec<u8>, String> {
        let mut state = self.candidate.as_ref().clone();
        state.indexes = self.live.indexes.clone();
        Catalog {
            state: CatalogStateLock::new(state),
        }
        .encode_current_state_v2()
    }
    #[cfg(feature = "lpg")]
    pub(crate) fn preimage(&self) -> CatalogRead<'_> {
        CatalogRead { state: &self.live }
    }

    #[cfg(any(all(feature = "lpg", feature = "wal"), test))]
    pub(crate) fn view(&self) -> CatalogRead<'_> {
        CatalogRead {
            state: self.candidate,
        }
    }

    pub(crate) fn install(mut self) -> InstalledCatalogFence<'catalog, 'workspace> {
        std::mem::swap(&mut *self.live, self.candidate);
        InstalledCatalogFence {
            live: self.live,
            _retired: self.candidate,
        }
    }
}

/// Logical installation retaining the writer and actual outer-workspace borrow.
/// Production callers preserve already-durable outcomes on later epoch errors;
/// swap-back below is only a test witness, not an aggregate rollback API.
pub(crate) struct InstalledCatalogFence<'catalog, 'workspace> {
    live: RwLockWriteGuard<'catalog, Arc<CatalogState>>,
    // Hold the actual workspace borrow through writer release, even though
    // production installation is not followed by a catalog-only rollback.
    _retired: &'workspace mut Arc<CatalogState>,
}

impl InstalledCatalogFence<'_, '_> {
    #[cfg(test)]
    pub(crate) fn view(&self) -> CatalogRead<'_> {
        CatalogRead { state: &self.live }
    }

    pub(crate) fn finish(self) {
        drop(self.live);
    }

    #[cfg(test)]
    pub(crate) fn rollback(mut self) {
        std::mem::swap(&mut *self.live, self._retired);
    }
}

impl Catalog {
    #[cfg(any(feature = "lpg", feature = "triple-store", feature = "gql", test))]
    pub(crate) fn read(&self) -> CatalogReadGuard<'_> {
        CatalogReadGuard {
            guard: self.state.read(),
        }
    }

    #[cfg(any(feature = "lpg", test))]
    pub(crate) fn prepare_edit<'catalog, 'workspace>(
        &'catalog self,
        workspace: &'workspace mut CatalogWorkspace,
    ) -> Result<CatalogEdit<'catalog, 'workspace>, CatalogError> {
        if workspace.candidate.is_some() {
            return Err(CatalogError::InvalidState(
                "catalog edit workspace is not vacant".to_string(),
            ));
        }
        let live = self.state.inner.write();
        let candidate = workspace.candidate.insert(Self {
            state: CatalogStateLock::shared(Arc::clone(&live)),
        });
        #[cfg(all(test, feature = "lpg", feature = "gql"))]
        run_preparation_rendezvous()?;
        Ok(CatalogEdit { live, candidate })
    }

    #[cfg(all(test, feature = "lpg", feature = "gql"))]
    pub(crate) fn writer_is_held_for_test(&self) -> bool {
        self.state.inner.try_write().is_none()
    }

    /// Bind an edited transaction cut to its exact committed preimage.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_transaction_edit<'catalog, 'workspace>(
        &'catalog self,
        base: &Catalog,
        candidate: Catalog,
        workspace: &'workspace mut CatalogWorkspace,
    ) -> Result<CatalogEdit<'catalog, 'workspace>, CatalogError> {
        if workspace.candidate.is_some() {
            return Err(CatalogError::InvalidState(
                "catalog edit workspace is not vacant".into(),
            ));
        }
        let expected = Arc::clone(&base.state.read());
        let live = self.state.inner.write();
        if !Arc::ptr_eq(&live, &expected) {
            return Err(CatalogError::InvalidState(
                "catalog changed during transaction preparation".into(),
            ));
        }
        let candidate = workspace.candidate.insert(candidate);
        Ok(CatalogEdit { live, candidate })
    }

    pub(crate) fn prepare_replacement<'catalog, 'workspace>(
        &'catalog self,
        workspace: &'workspace mut CatalogWorkspace,
    ) -> Result<ReadyCatalog<'catalog, 'workspace>, CatalogError> {
        let candidate = workspace.candidate.as_mut().ok_or_else(|| {
            CatalogError::InvalidState("catalog replacement has not been decoded".to_string())
        })?;
        let live = self.state.inner.write();
        Ok(ReadyCatalog {
            live,
            candidate: candidate.state.inner.get_mut(),
        })
    }

    /// WAL dictionary/schema bytes do not contain logical indexes.
    #[cfg(any(all(feature = "lpg", feature = "wal"), test))]
    pub(crate) fn prepare_wal_replacement<'catalog, 'workspace>(
        &'catalog self,
        workspace: &'workspace mut CatalogWorkspace,
    ) -> Result<ReadyCatalog<'catalog, 'workspace>, CatalogError> {
        let candidate = workspace.candidate.as_mut().ok_or_else(|| {
            CatalogError::InvalidState("catalog WAL replacement has not been decoded".to_string())
        })?;
        let live = self.state.inner.write();
        let candidate = candidate.state.inner.get_mut();
        if !candidate.indexes.indexes.is_empty() {
            return Err(CatalogError::InvalidState(
                "catalog WAL bytes cannot contain index definitions".to_string(),
            ));
        }
        for definition in live.indexes.indexes.values() {
            if live.labels.get_name(definition.label) != candidate.labels.get_name(definition.label)
                || live.property_keys.get_name(definition.property_key)
                    != candidate.property_keys.get_name(definition.property_key)
            {
                return Err(CatalogError::InvalidState(format!(
                    "catalog WAL replacement changes dictionary referents for index '{}'",
                    definition.name
                )));
            }
        }
        Arc::make_mut(candidate).indexes = live.indexes.clone();
        Ok(ReadyCatalog { live, candidate })
    }

    /// A transaction metadata image may not smuggle in owner changes.
    #[cfg(all(feature = "lpg", feature = "wal"))]
    pub(crate) fn prepare_transaction_metadata_replacement<'catalog, 'workspace>(
        &'catalog self,
        workspace: &'workspace mut CatalogWorkspace,
    ) -> Result<ReadyCatalog<'catalog, 'workspace>, CatalogError> {
        let ready = self.prepare_replacement(workspace)?;
        if ready.candidate.indexes.indexes != ready.live.indexes.indexes
            || ready.candidate.indexes.next_id != ready.live.indexes.next_id
            || ready.live.indexes.indexes.values().any(|owner| {
                ready.live.labels.get_name(owner.label)
                    != ready.candidate.labels.get_name(owner.label)
                    || ready.live.property_keys.get_name(owner.property_key)
                        != ready.candidate.property_keys.get_name(owner.property_key)
            })
        {
            return Err(CatalogError::InvalidState(
                "transaction metadata changes index owner preimage".into(),
            ));
        }
        Ok(ready)
    }

    #[cfg(any(all(feature = "lpg", feature = "wal"), test))]
    pub(crate) fn decode_wal_state_v1(data: &[u8]) -> Result<Self, String> {
        let (state, consumed): (CatalogWalStateV1, usize) =
            bincode::serde::decode_from_slice(data, bincode::config::standard())
                .map_err(|error| format!("failed to decode catalog WAL state v1: {error}"))?;
        if consumed != data.len() {
            return Err(format!(
                "catalog WAL state v1 has {} trailing bytes",
                data.len() - consumed
            ));
        }
        let state = CatalogState::from_snapshot(state.into_snapshot()?)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            state: CatalogStateLock::new(state),
        })
    }

    /// Current schema-only WAL fixture helper; production uses prepared publication.
    #[cfg(test)]
    pub(crate) fn restore_wal_state_v1(&self, data: &[u8]) -> Result<(), String> {
        let mut workspace = CatalogWorkspace::replacement(Self::decode_wal_state_v1(data)?);
        self.prepare_wal_replacement(&mut workspace)
            .map_err(|error| error.to_string())?
            .install()
            .finish();
        Ok(())
    }
}

impl CatalogState {
    pub(super) fn from_snapshot(snapshot: CatalogStateSnapshot) -> Result<Self, CatalogError> {
        Ok(Self {
            labels: LabelCatalog::from_names(snapshot.labels)?,
            property_keys: PropertyCatalog::from_names(snapshot.property_keys)?,
            edge_types: EdgeTypeCatalog::from_names(snapshot.edge_types)?,
            indexes: IndexCatalog::new(),
            schema: snapshot.schema.map(|schema| SchemaState {
                unique_constraints: schema.unique_constraints,
                required_properties: schema.required_properties,
                named_constraints: schema.named_constraints,
                node_types: schema.node_types,
                edge_types: schema.edge_types,
                graph_types: schema.graph_types,
                schemas: schema.schemas,
                graph_type_bindings: schema.graph_type_bindings,
                procedures: schema.procedures,
            }),
            #[cfg(test)]
            retirement_probe: None,
        })
    }
}

#[cfg(test)]
thread_local! {
    static COMPLETE_STATE_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    #[cfg(all(feature = "lpg", feature = "gql"))]
    static PREPARATION_RENDEZVOUS: std::cell::RefCell<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, feature = "lpg", feature = "gql"))]
pub(crate) fn install_preparation_rendezvous(
    prepared: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
) -> Result<(), CatalogError> {
    PREPARATION_RENDEZVOUS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(CatalogError::InvalidState(
                "test preparation rendezvous already installed".to_string(),
            ));
        }
        *slot = Some((prepared, release));
        Ok(())
    })
}

#[cfg(all(test, feature = "lpg", feature = "gql"))]
fn run_preparation_rendezvous() -> Result<(), CatalogError> {
    let rendezvous = PREPARATION_RENDEZVOUS.with(|slot| slot.borrow_mut().take());
    if let Some((prepared, release)) = rendezvous {
        prepared
            .send(())
            .map_err(|error| CatalogError::InvalidState(error.to_string()))?;
        release
            .recv()
            .map_err(|error| CatalogError::InvalidState(error.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn record_state_copy() {
    COMPLETE_STATE_COPIES.with(|copies| copies.set(copies.get() + 1));
}

#[cfg(test)]
pub(crate) fn state_copy_count() -> usize {
    COMPLETE_STATE_COPIES.with(std::cell::Cell::get)
}

#[cfg(test)]
#[derive(Clone)]
pub(super) struct CatalogRetirementProbe {
    check: Arc<dyn Fn() -> bool + Send + Sync>,
    releases: std::sync::Arc<parking_lot::Mutex<Vec<bool>>>,
}

#[cfg(test)]
impl Drop for CatalogRetirementProbe {
    fn drop(&mut self) {
        self.releases.lock().push((self.check)());
    }
}

#[cfg(all(test, feature = "lpg", feature = "gql"))]
impl Catalog {
    pub(crate) fn observe_retirement(
        &self,
        check: Arc<dyn Fn() -> bool + Send + Sync>,
        releases: Arc<parking_lot::Mutex<Vec<bool>>>,
    ) {
        self.state.write().retirement_probe = Some(CatalogRetirementProbe { check, releases });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{
        IndexConfiguration, IndexType, NamedConstraintDefinition, NamedConstraintKind,
    };
    use grafeo_common::types::GraphPath;
    use std::sync::{Arc, mpsc};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn catalog_foundation_ordinary_admission_never_copies_complete_state() -> TestResult {
        let catalog = Catalog::new();
        let before = state_copy_count();
        for index in 0..1000 {
            let name = format!("name_{index}");
            let label = catalog.get_or_create_label(&name)?;
            let property = catalog.get_or_create_property_key(&name)?;
            catalog.get_or_create_edge_type(&name)?;
            catalog.create_index(
                Some(&name),
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Property,
            )?;
            assert_eq!(catalog.get_or_create_label(&name)?, label);
        }
        assert_eq!(state_copy_count(), before);
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn catalog_transaction_snapshots_and_reads_share_one_cut_until_an_edit() -> TestResult {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("present")?;
        let before = state_copy_count();
        let base = catalog.snapshot();
        let working = base.snapshot();
        let savepoint = working.snapshot();
        assert!(catalog.same_cut(&working));
        for _ in 0..100 {
            assert_eq!(working.get_or_create_label("present")?, label);
        }
        assert_eq!(state_copy_count(), before);
        working.get_or_create_label("private")?;
        assert_eq!(state_copy_count(), before + 1);
        assert!(!working.same_cut(&base));
        assert_eq!(base.get_label_id("private"), None);
        assert_eq!(savepoint.get_label_id("private"), None);
        assert!(catalog.same_cut(&base));
        Ok(())
    }

    #[test]
    fn catalog_foundation_late_property_exhaustion_leaves_no_constraint_prefix() -> TestResult {
        let catalog = Catalog::new();
        catalog.get_or_create_label("existing")?;
        catalog.get_or_create_property_key("existing")?;
        catalog.state.write().property_keys.next_id = u32::MAX - 1;
        let before = catalog.encode_wal_state_v1()?;
        let result = catalog.create_named_constraint(NamedConstraintDefinition {
            name: "new_owner".to_string(),
            label: "new_label".to_string(),
            properties: vec!["first".to_string(), "second".to_string()],
            kind: NamedConstraintKind::NodeKey,
        });
        assert_eq!(result, Err(CatalogError::IdExhausted("property key")));
        assert_eq!(catalog.encode_wal_state_v1()?, before);
        assert_eq!(catalog.state.read().property_keys.next_id, u32::MAX - 1);
        assert_eq!(catalog.get_label_id("new_label"), None);
        assert_eq!(catalog.get_property_key_id("first"), None);
        assert_eq!(catalog.get_named_constraint("new_owner"), None);
        Ok(())
    }

    #[test]
    fn catalog_foundation_all_counter_failures_preserve_owner_and_names() -> TestResult {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("L")?;
        let property = catalog.get_or_create_property_key("p")?;
        catalog.get_or_create_edge_type("E")?;
        let owner = catalog.create_index(
            Some("owner"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        {
            let mut state = catalog.state.write();
            state.labels.next_id = u32::MAX;
            state.property_keys.next_id = u32::MAX;
            state.edge_types.next_id = u32::MAX;
            state.indexes.next_id = u32::MAX;
        }
        let before = catalog.encode_wal_state_v1()?;
        assert_eq!(
            catalog.get_or_create_label("new"),
            Err(CatalogError::IdExhausted("label"))
        );
        assert_eq!(
            catalog.get_or_create_property_key("new"),
            Err(CatalogError::IdExhausted("property key"))
        );
        assert_eq!(
            catalog.get_or_create_edge_type("new"),
            Err(CatalogError::IdExhausted("edge type"))
        );
        assert_eq!(
            catalog.create_index(
                Some("new"),
                label,
                property,
                GraphPath::from_components(&["new"])?,
                IndexConfiguration::Property
            ),
            Err(CatalogError::IdExhausted("index"))
        );
        assert_eq!(catalog.encode_wal_state_v1()?, before);
        assert_eq!(catalog.find_index_by_name("owner"), Some(owner));
        assert_eq!(catalog.index_count(), 1);
        assert_eq!(catalog.state.read().indexes.next_id, u32::MAX);
        assert_eq!(catalog.get_or_create_label("L")?, label);
        Ok(())
    }

    #[test]
    fn catalog_foundation_duplicate_index_name_cannot_replace_owner() -> TestResult {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("L")?;
        let property = catalog.get_or_create_property_key("p")?;
        let owner = catalog.create_index(
            Some("owner"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        let next = catalog.state.read().indexes.next_id;
        assert_eq!(
            catalog.create_index(
                Some("owner"),
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::BTree
            ),
            Err(CatalogError::IndexAlreadyExists("owner".to_string()))
        );
        assert_eq!(catalog.find_index_by_name("owner"), Some(owner));
        assert_eq!(
            catalog.get_index(owner).ok_or("owner missing")?.index_type,
            IndexType::Hash
        );
        assert_eq!(catalog.state.read().indexes.next_id, next);
        assert!(catalog.drop_index(owner));
        assert_eq!(catalog.find_index_by_name("owner"), None);
        assert_eq!(catalog.index_count(), 0);
        Ok(())
    }

    #[test]
    fn failed_detached_owner_edit_does_not_advance_live_allocator() -> TestResult {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("L")?;
        let property = catalog.get_or_create_property_key("p")?;
        let mut workspace = CatalogWorkspace::new();
        {
            let edit = catalog.prepare_edit(&mut workspace)?;
            let provisional = edit.candidate().create_index(
                None,
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Property,
            )?;
            assert_eq!(edit.candidate().index_allocator_high_water(), 1);
            assert_eq!(
                edit.candidate().create_index(
                    Some("alias"),
                    label,
                    property,
                    GraphPath::root(),
                    IndexConfiguration::BTree
                ),
                Err(CatalogError::IndexPhysicalAlreadyOwned(provisional))
            );
            drop(edit.finish());
        }
        assert_eq!(catalog.index_allocator_high_water(), 0);
        assert_eq!(catalog.index_count(), 0);
        assert_eq!(catalog.find_index_by_name("@grafeo-index:0"), None);
        drop(workspace);
        let owner = catalog.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert_eq!(owner.as_u32(), 0);
        assert_eq!(catalog.index_allocator_high_water(), 1);
        let before = catalog.get_index(owner);
        let mut workspace = CatalogWorkspace::new();
        {
            let edit = catalog.prepare_edit(&mut workspace)?;
            assert!(edit.candidate().drop_index(owner));
            edit.candidate().create_index(
                None,
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::BTree,
            )?;
            drop(edit);
        }
        assert_eq!(catalog.get_index(owner), before);
        assert_eq!(catalog.index_allocator_high_water(), 1);
        Ok(())
    }

    #[test]
    fn catalog_foundation_retirement_follows_outer_gates_for_all_outcomes() -> TestResult {
        for outcome in ["abort", "error", "success", "rollback", "unwind"] {
            let outer = Arc::new(parking_lot::Mutex::new(()));
            let releases = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let catalog = Catalog::new();
            catalog.get_or_create_label("old")?;
            catalog.state.write().retirement_probe = Some(CatalogRetirementProbe {
                check: {
                    let outer = Arc::clone(&outer);
                    Arc::new(move || outer.try_lock().is_some())
                },
                releases: Arc::clone(&releases),
            });
            let mut workspace = CatalogWorkspace::new();
            let operation =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> TestResult {
                    let _outer = outer.lock();
                    let edit = catalog.prepare_edit(&mut workspace)?;
                    edit.candidate().get_or_create_label("new")?;
                    match outcome {
                        "abort" => drop(edit.finish()),
                        "error" => {
                            edit.candidate().state.write().labels.next_id = u32::MAX;
                            assert_eq!(
                                edit.candidate().get_or_create_label("failed"),
                                Err(CatalogError::IdExhausted("label"))
                            );
                            drop(edit);
                        }
                        "success" => {
                            let fence = edit.finish().install();
                            assert!(fence.view().get_label_id("new").is_some());
                            fence.finish();
                        }
                        "rollback" => {
                            let fence = edit.finish().install();
                            assert!(fence.view().get_label_id("new").is_some());
                            fence.rollback();
                        }
                        "unwind" => {
                            std::panic::resume_unwind(Box::new("intentional retirement probe"))
                        }
                        _ => return Err("invalid test outcome".into()),
                    }
                    assert!(
                        releases.lock().is_empty(),
                        "payload retired under outer gate"
                    );
                    Ok(())
                }));
            if outcome == "unwind" {
                assert!(operation.is_err());
            } else {
                operation.map_err(|_| "unexpected unwind")??;
            }
            assert!(
                releases.lock().is_empty(),
                "workspace must own the payload after guard release"
            );
            drop(workspace);
            assert_eq!(*releases.lock(), vec![true], "outcome {outcome}");
            assert_eq!(catalog.get_label_id("new").is_some(), outcome == "success");
        }
        Ok(())
    }

    #[test]
    fn catalog_foundation_direct_mutation_and_prepared_successor_are_both_retained() -> TestResult {
        let catalog = Arc::new(Catalog::new());
        let mut workspace = CatalogWorkspace::new();
        let edit = catalog.prepare_edit(&mut workspace)?;
        edit.candidate()
            .register_schema_namespace("prepared".to_string())?;
        assert!(
            catalog.state.inner.try_write().is_none(),
            "preparation must retain actual writer authority"
        );
        let (started_tx, started_rx) = mpsc::channel();
        let other = Arc::clone(&catalog);
        let worker = std::thread::spawn(move || -> Result<(), CatalogError> {
            let sent = started_tx.send(());
            if sent.is_err() {
                return Err(CatalogError::InvalidState(
                    "test rendezvous closed".to_string(),
                ));
            }
            other.register_schema_namespace("direct".to_string())
        });
        started_rx.recv()?;
        let ready = edit.finish();
        assert!(ready.view().schema_exists("prepared"));
        ready.install().finish();
        worker
            .join()
            .map_err(|_| "direct mutation worker panicked")??;
        let view = catalog.read();
        assert!(view.schema_exists("prepared"));
        assert!(view.schema_exists("direct"));
        Ok(())
    }

    #[test]
    fn catalog_foundation_wal_retains_counter_and_full_replace_removes_target_only_owners()
    -> TestResult {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("L")?;
        let property = catalog.get_or_create_property_key("p")?;
        let owner = catalog.create_index(
            Some("owner"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        catalog.state.write().indexes.next_id = 37;
        let bytes = catalog.encode_wal_state_v1()?;
        catalog.restore_wal_state_v1(&bytes)?;
        assert_eq!(catalog.find_index_by_name("owner"), Some(owner));
        assert_eq!(catalog.state.read().indexes.next_id, 37);
        let mut workspace = CatalogWorkspace::replacement(Catalog::decode_wal_state_v1(&bytes)?);
        catalog
            .prepare_replacement(&mut workspace)?
            .install()
            .finish();
        assert_eq!(catalog.find_index_by_name("owner"), None);
        assert_eq!(catalog.state.read().indexes.next_id, 0);
        Ok(())
    }
}
