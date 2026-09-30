//! Engine-owned inputs for the connected LPG data/index commit segment.
//!
//! Anchors and row images precede the separately borrowed core workspace. The
//! scoped core driver releases topology before Session's existing lifecycle
//! tail; no ordinary physical/data installer runs inside that scope.

use super::{
    Arc, CatalogRead, CatalogWorkspace, EpochId, GraphPath, GraphStoreMut, IndexConfiguration,
    LpgStore, Node, NodeId, PendingGraphTypeBinding, PendingIndexDdl, PendingIndexKind,
    PhysicalIndexFamily, PhysicalIndexKey, PreparedLogicalCatalog, Result, Session, TransactionId,
    Value,
};
use grafeo_common::types::PropertyKey;
#[cfg(feature = "vector-index")]
use grafeo_common::utils::hash::FxHashMap;
#[cfg(feature = "vector-index")]
use grafeo_core::graph::lpg::{IndexRegistrationObservation, VectorCommitInput};
use grafeo_core::graph::lpg::{
    IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey, IndexRegistryMaintenance,
    LpgCommitWorkspace, StoreCommitInput,
};
#[cfg(feature = "vector-index")]
use grafeo_core::index::vector::{VectorIndexView, value_to_vector};

#[derive(Default)]
pub(super) struct EngineCommitCapture {
    stores: Vec<CapturedStore>,
    pending: Vec<PendingIndexDdl>,
    bindings: std::collections::HashMap<GraphPath, PendingGraphTypeBinding>,
    dropped_binding_roots: std::collections::HashSet<GraphPath>,
    #[cfg(feature = "wal")]
    graph_coordinates: Vec<(Arc<LpgStore>, GraphPath)>,
    #[cfg(feature = "wal")]
    index_coordinates: Vec<(Arc<LpgStore>, GraphPath)>,
    #[cfg(feature = "wal")]
    index_wal_required: bool,
    #[cfg(feature = "vector-index")]
    vectors: Vec<CapturedVector>,
}

struct CapturedStore {
    store: Arc<LpgStore>,
    source: Arc<dyn GraphStoreMut>,
    graph: GraphPath,
    graph_root: Arc<LpgStore>,
    graph_suffix: GraphPath,
    publish_data: bool,
    before: Vec<Node>,
    after: Vec<Node>,
    changed: Vec<NodeId>,
    definitions: std::collections::HashMap<PhysicalIndexKey, Option<usize>>,
    keys: std::collections::HashSet<PhysicalIndexKey>,
    edits: Vec<IndexRegistryEdit>,
}

#[cfg(feature = "vector-index")]
struct CapturedVector {
    store: Arc<LpgStore>,
    property: PropertyKey,
    view: VectorIndexView,
    expected: IndexRegistrationObservation,
    rows: Vec<(NodeId, Option<Arc<[f32]>>)>,
    routing: FxHashMap<NodeId, Arc<[f32]>>,
}

impl EngineCommitCapture {
    fn remember(
        &mut self,
        session: &Session,
        store: Arc<LpgStore>,
        graph: GraphPath,
        publish_data: bool,
    ) -> Result<usize> {
        if Arc::ptr_eq(&store, &session.store) != graph.components().is_empty() {
            return Err(Session::index_ddl_error(
                "captured store/path root identity mismatch",
            ));
        }
        if let Some(index) = self
            .stores
            .iter()
            .position(|known| Arc::ptr_eq(&known.store, &store))
        {
            if self.stores[index].graph != graph {
                return Err(Session::index_ddl_error(
                    "captured store has ambiguous graph coordinates",
                ));
            }
            self.stores[index].publish_data |= publish_data;
            return Ok(index);
        }
        let source = if Arc::ptr_eq(&store, &session.store) {
            session
                .graph_store_mut
                .as_ref()
                .map_or_else(|| Arc::clone(&store) as Arc<dyn GraphStoreMut>, Arc::clone)
        } else {
            Arc::clone(&store) as Arc<dyn GraphStoreMut>
        };
        let mut graph_root = Arc::clone(&session.store);
        let mut graph_suffix = graph.clone();
        if !graph.components().is_empty() {
            // A transaction-local graph is intentionally not published beneath
            // the database root yet. Anchor its actual detached incarnation;
            // Session's lifecycle proof still owns its outer publication.
            // Nested staged children may not yet belong to their staged parent,
            // so prefer the longest literal prefix that resolves this exact Arc.
            let mut candidates: Vec<_> = session
                .pending_created_graphs
                .lock()
                .iter()
                .filter(|(path, _)| graph.components().starts_with(path.components()))
                .map(|(path, pending)| (path.clone(), Arc::clone(&pending.store)))
                .collect();
            candidates.extend(
                session
                    .cancelled_created_graphs
                    .lock()
                    .iter()
                    .filter(|(path, _)| graph.components().starts_with(path.components()))
                    .flat_map(|(path, stores)| {
                        stores
                            .iter()
                            .map(move |store| (path.clone(), Arc::clone(store)))
                    }),
            );
            candidates.sort_by(|(left, _), (right, _)| {
                right
                    .components()
                    .len()
                    .cmp(&left.components().len())
                    .then_with(|| left.cmp(right))
            });
            for (prefix, candidate) in candidates {
                let Some(rest) = graph.components().strip_prefix(prefix.components()) else {
                    continue;
                };
                let mut resolved = Some(Arc::clone(&candidate));
                for component in rest {
                    resolved = resolved.and_then(|store| store.graph(component));
                }
                if resolved
                    .as_ref()
                    .is_some_and(|target| Arc::ptr_eq(target, &store))
                {
                    graph_root = candidate;
                    graph_suffix = GraphPath::from_components(
                        &rest.iter().map(String::as_str).collect::<Vec<_>>(),
                    )
                    .map_err(|error| Session::index_ddl_error(error.to_string()))?;
                    break;
                }
            }
        }
        let index = self.stores.len();
        self.stores.push(CapturedStore {
            store,
            source,
            graph,
            graph_root,
            graph_suffix,
            publish_data,
            before: Vec::new(),
            after: Vec::new(),
            changed: Vec::new(),
            definitions: std::collections::HashMap::new(),
            keys: std::collections::HashSet::new(),
            edits: Vec::new(),
        });
        Ok(index)
    }

    pub(super) fn prepare(
        &mut self,
        session: &Session,
        touched: &[GraphPath],
        transaction: TransactionId,
        publication: EpochId,
        commit: EpochId,
    ) -> Result<()> {
        // Move the actual descriptors out of Session: successful publication,
        // rejection and unwind all retire them after the enclosing gates.
        self.pending = std::mem::take(&mut *session.pending_index_ddl.lock());
        self.bindings = std::mem::take(&mut *session.pending_graph_type_bindings.lock());
        self.dropped_binding_roots
            .extend(session.pending_dropped_graphs.lock().keys().cloned());
        self.dropped_binding_roots
            .extend(session.cancelled_created_graphs.lock().keys().cloned());
        if !self.pending.is_empty() {
            session.validate_pending_index_ddl_at(
                session.catalog_view().read().view(),
                &self.pending,
            )?;
        }
        for graph in touched {
            let store = session.pinned_graph_path(graph);
            if let Some(store) = store {
                let survives = session.lpg_incarnation_survives_pending_lifecycle(graph, &store);
                self.remember(session, store, graph.clone(), survives)?;
            } else if !session.cancelled_created_graphs.lock().contains_key(graph) {
                return Err(Session::index_ddl_error(
                    "touched graph lost its captured incarnation",
                ));
            }
        }
        let superseded = session.superseded_graph_touches.lock().clone();
        for (graph, store) in superseded {
            self.remember(session, store, graph, false)?;
        }
        // Zero-event detached/cancelled incarnations are not necessarily in
        // the touched-path set. Keep their exact anchors through the tail.
        // remember() resolves staged path witnesses through these maps. Own
        // the entries first so that traversal never reenters their mutexes.
        // Lifecycle keys retain every literal component, including '/'.
        let created: Vec<_> = session
            .pending_created_graphs
            .lock()
            .iter()
            .map(|(graph, pending)| (graph.clone(), Arc::clone(&pending.store)))
            .collect();
        for (graph, store) in created {
            let survives = session.lpg_incarnation_survives_pending_lifecycle(&graph, &store);
            self.remember(session, store, graph, survives)?;
        }
        let dropped = session.pending_dropped_graphs.lock().clone();
        for (graph, store) in dropped {
            self.remember(session, store, graph, false)?;
        }
        let cancelled = session.cancelled_created_graphs.lock().clone();
        for (graph, stores) in cancelled {
            for store in stores {
                self.remember(session, store, graph.clone(), false)?;
            }
        }
        for index in 0..self.pending.len() {
            let store = Arc::clone(&self.pending[index].target);
            let graph = self.pending[index].graph.clone();
            let survives = session.index_target_survives(&graph, &store);
            self.remember(session, store, graph, survives)?;
        }

        // Fold physical ownership in the same operation order as the logical
        // catalog. A DROP with another surviving logical owner is not a
        // physical removal; repeated DDL produces one final edit per key.
        let catalog = session.catalog.read();
        let mut owners: std::collections::HashMap<
            PhysicalIndexKey,
            std::collections::HashSet<String>,
        > = std::collections::HashMap::new();
        if !self.pending.is_empty() {
            for definition in catalog.all_indexes() {
                owners
                    .entry(definition.key)
                    .or_default()
                    .insert(definition.name);
            }
        }
        for (ordinal, ddl) in self.pending.iter().enumerate() {
            let key = Session::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind);
            let current = owners.entry(key.clone()).or_default();
            if ddl.create {
                current.insert(ddl.name.clone().unwrap_or_else(|| "<unnamed>".to_owned()));
            } else if let Some(name) = &ddl.name {
                current.remove(name);
            } else {
                current.remove("<unnamed>");
            }
            if ddl.create || current.is_empty() {
                let target = self
                    .stores
                    .iter_mut()
                    .find(|store| Arc::ptr_eq(&store.store, &ddl.target))
                    .ok_or_else(|| {
                        Session::index_ddl_error("prepared DDL target was not retained")
                    })?;
                let local = key.with_graph(GraphPath::root());
                target
                    .definitions
                    .insert(local, ddl.create.then_some(ordinal));
            }
        }
        drop(catalog);

        for ordinal in 0..self.stores.len() {
            let captured = &mut self.stores[ordinal];
            if !captured.publish_data {
                // Discarded incarnations have no live path to revalidate.
                // Their exact self-anchor still fences registry retirement.
                captured.graph_root = Arc::clone(&captured.store);
                captured.graph_suffix = GraphPath::root();
            }
            captured.keys =
                Session::physical_index_keys_for_store(&GraphPath::root(), &captured.store);
            captured.keys.extend(captured.definitions.keys().cloned());
            if captured.keys.is_empty() {
                continue;
            }
            if captured.publish_data {
                captured.changed = captured.source.overlay_touched_entities(transaction).0;
                captured
                    .changed
                    .extend(captured.source.pending_node_creates(transaction));
                captured
                    .changed
                    .extend(captured.source.pending_node_deletes_peek(transaction));
                captured.changed.sort_unstable();
                captured.changed.dedup();
            }
            let builds_index = captured.definitions.values().any(Option::is_some);
            if !captured.changed.is_empty() {
                captured.before = captured.source.prepare_index_node_rows_by_id(
                    publication,
                    None,
                    &captured.changed,
                )?;
            }
            if builds_index {
                // A fresh physical index needs its whole final population.
                // Surviving indexes below still receive only touched rows.
                captured.after = captured
                    .source
                    .prepare_index_node_rows(publication, Some(transaction))?;
            } else if !captured.changed.is_empty() {
                captured.after = captured.source.prepare_index_node_rows_by_id(
                    publication,
                    Some(transaction),
                    &captured.changed,
                )?;
            }
            captured.before.sort_unstable_by_key(|node| node.id);
            captured.after.sort_unstable_by_key(|node| node.id);
            captured.changed.retain(|id| {
                let before = row(&captured.before, *id);
                let after = row(&captured.after, *id);
                match (before, after) {
                    (Some(before), Some(after)) => {
                        before.properties != after.properties
                            || before.labels.len() != after.labels.len()
                            || before
                                .labels
                                .iter()
                                .any(|label| !after.has_label(label.as_str()))
                    }
                    (None, None) => false,
                    _ => true,
                }
            });
            for key in &captured.keys {
                #[cfg(feature = "wal")]
                if matches!(
                    key.family(),
                    PhysicalIndexFamily::Text | PhysicalIndexFamily::Vector
                ) {
                    self.index_wal_required = true;
                }
                let expected = observe(&captured.store, key);
                if let Some(definition) = captured.definitions.get(key) {
                    match definition {
                        Some(definition) => {
                            let ddl = &mut self.pending[*definition];
                            #[cfg(feature = "text-index")]
                            if ddl.rebuild && matches!(ddl.kind, PendingIndexKind::Text { .. }) {
                                let Some(IndexConfiguration::Text {
                                    config,
                                    min_token_length,
                                }) = &ddl.configuration
                                else {
                                    return Err(Session::index_ddl_error(
                                        "Text rebuild lacks canonical configuration",
                                    ));
                                };
                                let expected = expected.ok_or_else(|| {
                                    Session::index_ddl_error(
                                        "Text rebuild registration disappeared",
                                    )
                                })?;
                                captured.edits.push(IndexRegistryEdit::Maintain {
                                    expected,
                                    changes: IndexRegistryMaintenance::TextRebuild {
                                        config: config.clone(),
                                        min_token_length: *min_token_length,
                                        rows: captured
                                            .after
                                            .iter()
                                            .filter_map(|node| {
                                                text(Some(node), &ddl.label, &ddl.property)
                                                    .map(|text| (node.id, Some(text.to_owned())))
                                            })
                                            .collect(),
                                        frontier: publication,
                                        commit_epoch: commit,
                                        transaction_id: transaction,
                                    },
                                });
                                continue;
                            }
                            let (contents, configuration) =
                                build_contents(ddl, &captured.store, &captured.after, commit)?;
                            ddl.configuration = Some(configuration);
                            captured.edits.push(match expected {
                                Some(expected) => IndexRegistryEdit::Replace { expected, contents },
                                None => IndexRegistryEdit::Create {
                                    key: registry_key(key)?,
                                    contents,
                                },
                            });
                        }
                        None => {
                            if let Some(expected) = expected {
                                captured.edits.push(IndexRegistryEdit::Drop { expected });
                            }
                        }
                    }
                    continue;
                }
                if !captured.publish_data {
                    continue;
                }
                let expected = expected.ok_or_else(|| {
                    Session::index_ddl_error("surviving index disappeared during capture")
                })?;
                match key.family() {
                    PhysicalIndexFamily::Property => {
                        let changes: Vec<_> = captured
                            .changed
                            .iter()
                            .filter_map(|id| {
                                let before =
                                    property(row(&captured.before, *id), key.property_name())
                                        .cloned();
                                let after =
                                    property(row(&captured.after, *id), key.property_name())
                                        .cloned();
                                (before != after).then_some((*id, before, after))
                            })
                            .collect();
                        if changes.is_empty() {
                            continue;
                        }
                        captured.edits.push(IndexRegistryEdit::Maintain {
                            expected,
                            changes: IndexRegistryMaintenance::PropertyAt {
                                rows: changes,
                                commit_epoch: commit,
                            },
                        });
                    }
                    PhysicalIndexFamily::Text => {
                        #[cfg(feature = "text-index")]
                        {
                            let label = key.label().ok_or_else(|| {
                                Session::index_ddl_error("Text key lacks a label")
                            })?;
                            let rows: Vec<_> = captured
                                .changed
                                .iter()
                                .filter_map(|id| {
                                    let before = text(
                                        row(&captured.before, *id),
                                        label,
                                        key.property_name(),
                                    );
                                    let after =
                                        text(row(&captured.after, *id), label, key.property_name());
                                    (before != after).then(|| (*id, after.map(str::to_owned)))
                                })
                                .collect();
                            if rows.is_empty() {
                                continue;
                            }
                            captured.edits.push(IndexRegistryEdit::Maintain {
                                expected,
                                changes: IndexRegistryMaintenance::Text {
                                    rows,
                                    frontier: publication,
                                    commit_epoch: commit,
                                    transaction_id: transaction,
                                },
                            });
                        }
                        #[cfg(not(feature = "text-index"))]
                        return Err(Session::index_ddl_error(
                            "captured Text index without Text support",
                        ));
                    }
                    PhysicalIndexFamily::Vector => {
                        #[cfg(feature = "vector-index")]
                        {
                            let label = key.label().ok_or_else(|| {
                                Session::index_ddl_error("Vector key lacks a label")
                            })?;
                            if !captured.changed.iter().any(|id| {
                                vector_value(row(&captured.before, *id), label, key.property_name())
                                    != vector_value(
                                        row(&captured.after, *id),
                                        label,
                                        key.property_name(),
                                    )
                            }) {
                                continue;
                            }
                            let view = captured
                                .store
                                .get_vector_index(label, key.property_name())
                                .ok_or_else(|| {
                                    Session::index_ddl_error(
                                        "Vector target disappeared during capture",
                                    )
                                })?;
                            self.vectors.push(CapturedVector {
                                store: Arc::clone(&captured.store),
                                property: PropertyKey::new(key.property_name()),
                                view,
                                expected,
                                rows: Vec::new(),
                                routing: FxHashMap::default(),
                            });
                            let vector = self.vectors.last_mut().ok_or_else(|| {
                                Session::index_ddl_error("Vector capture slot is absent")
                            })?;
                            for id in &captured.changed {
                                let before = vector_value(
                                    row(&captured.before, *id),
                                    label,
                                    key.property_name(),
                                );
                                let after = vector_value(
                                    row(&captured.after, *id),
                                    label,
                                    key.property_name(),
                                );
                                if before != after {
                                    vector.rows.push((*id, after));
                                }
                            }
                            // Deleted rows retain their committed routing vector
                            // while topology is staged; surviving final rows win.
                            for id in &captured.changed {
                                let routing = vector_value(
                                    row(&captured.after, *id),
                                    label,
                                    key.property_name(),
                                )
                                .or_else(|| {
                                    vector_value(
                                        row(&captured.before, *id),
                                        label,
                                        key.property_name(),
                                    )
                                });
                                if let Some(value) = routing {
                                    vector.routing.insert(*id, value);
                                }
                            }
                        }
                        #[cfg(not(feature = "vector-index"))]
                        return Err(Session::index_ddl_error(
                            "captured Vector index without Vector support",
                        ));
                    }
                }
            }
        }
        #[cfg(feature = "wal")]
        if session.wal.is_some() {
            if self.index_wal_required {
                self.index_coordinates.extend(
                    self.stores
                        .iter()
                        .map(|store| (Arc::clone(&store.store), store.graph.clone())),
                );
            }
            for store in self.stores.iter().filter(|store| store.publish_data) {
                self.graph_coordinates
                    .push((Arc::clone(&store.store), store.graph.clone()));
            }
        }
        Ok(())
    }

    pub(super) fn workspace(
        &mut self,
        transaction: TransactionId,
        publication: EpochId,
        commit: EpochId,
    ) -> (LpgCommitWorkspace<'_>, LogicalCommitInputs<'_>) {
        let stores = self
            .stores
            .iter_mut()
            .map(|store| StoreCommitInput {
                store: &store.store,
                graph: Some((&store.graph_root, &store.graph_suffix)),
                source: store.source.as_ref(),
                publish_data: store.publish_data,
                edits: std::mem::take(&mut store.edits),
            })
            .collect();
        #[cfg(feature = "vector-index")]
        let vectors = self
            .vectors
            .iter_mut()
            .map(|vector| VectorCommitInput {
                store: &vector.store,
                property: vector.property.clone(),
                view: &vector.view,
                expected: &vector.expected,
                changes: grafeo_core::graph::lpg::VectorCommitChanges::Rows(std::mem::take(
                    &mut vector.rows,
                )),
                routing: std::mem::take(&mut vector.routing),
            })
            .collect();
        (
            LpgCommitWorkspace::new(
                stores,
                #[cfg(feature = "vector-index")]
                vectors,
                transaction,
                publication,
                commit,
            ),
            LogicalCommitInputs {
                pending: &self.pending,
                bindings: &self.bindings,
                dropped_binding_roots: &self.dropped_binding_roots,
                #[cfg(feature = "wal")]
                graph_coordinates: &self.graph_coordinates,
                #[cfg(feature = "wal")]
                index_coordinates: &self.index_coordinates,
                #[cfg(feature = "wal")]
                index_wal_required: self.index_wal_required,
                #[cfg(feature = "wal")]
                frontier: publication,
                #[cfg(feature = "wal")]
                commit,
            },
        )
    }
}

/// Facts captured before core authority. Preparing the logical companion may
/// read only catalog state: Vector/Layered readers are already excluded here.
pub(super) struct LogicalCommitInputs<'capture> {
    pending: &'capture [PendingIndexDdl],
    bindings: &'capture std::collections::HashMap<GraphPath, PendingGraphTypeBinding>,
    dropped_binding_roots: &'capture std::collections::HashSet<GraphPath>,
    #[cfg(feature = "wal")]
    graph_coordinates: &'capture [(Arc<LpgStore>, GraphPath)],
    #[cfg(feature = "wal")]
    index_coordinates: &'capture [(Arc<LpgStore>, GraphPath)],
    #[cfg(feature = "wal")]
    index_wal_required: bool,
    #[cfg(feature = "wal")]
    frontier: EpochId,
    #[cfg(feature = "wal")]
    commit: EpochId,
}

impl LogicalCommitInputs<'_> {
    #[cfg(feature = "wal")]
    pub(super) fn write_wal_publication(
        &self,
        session: &Session,
        prepared: &grafeo_core::graph::lpg::ReleasedLpgCommit<'_, '_>,
        transaction_id: TransactionId,
    ) -> Result<()> {
        use grafeo_storage::wal::{LpgMutationOp, WalRecord};

        let Some(wal) = &session.wal else {
            return Ok(());
        };
        // A surviving touched child publishes the commit cut even when its
        // label operations normalize away or never create a buffered delta.
        // The root obtains its cut from the global Committed marker instead.
        for (_, graph) in self.graph_coordinates {
            if !graph.components().is_empty() {
                wal.log(&WalRecord::lpg(
                    transaction_id,
                    graph.clone(),
                    LpgMutationOp::PublishGraph,
                ))?;
            }
        }
        for (store, images) in prepared.label_images() {
            let (_, graph) = self
                .graph_coordinates
                .iter()
                .find(|(captured, _)| std::ptr::eq(captured.as_ref(), store))
                .ok_or_else(|| {
                    Session::index_ddl_error("prepared labels lack a surviving captured graph")
                })?;
            for labels in images {
                wal.log(&WalRecord::lpg(
                    transaction_id,
                    graph.clone(),
                    LpgMutationOp::NodeLabelImages {
                        id: labels.id(),
                        birth: labels.birth(),
                        images: labels.images().to_vec(),
                    },
                ))?;
            }
        }
        Ok(())
    }

    /// IDs are read from the real prepared catalog after allocation. No store
    /// reads or catalog reacquisition are allowed while core readers are fenced.
    #[cfg(feature = "wal")]
    pub(super) fn write_wal_owners(
        &self,
        session: &Session,
        publication: &PreparedLogicalCatalog<'_, '_>,
        prepared: &grafeo_core::graph::lpg::ReleasedLpgCommit<'_, '_>,
        transaction_id: TransactionId,
    ) -> Result<()> {
        use crate::catalog::{IndexOwnerBatch, IndexOwnerChange, IndexOwnerImage};
        let Some(wal) = &session.wal else {
            return Ok(());
        };
        if self.pending.is_empty() && !self.index_wal_required {
            return Ok(());
        }
        let (before, after) = match publication {
            PreparedLogicalCatalog::NoCatalogChange => return Ok(()),
            PreparedLogicalCatalog::OwnersPinned(owners) => (owners.view(), owners.view()),
            PreparedLogicalCatalog::Changed(ready) => (ready.preimage(), ready.view()),
        };
        let capture = |catalog, id| {
            IndexOwnerImage::capture(catalog, id)
                .map_err(|error| Session::index_ddl_error(error.to_string()))
        };
        let mut changes = Vec::new();
        for ddl in self.pending {
            if ddl.rebuild {
                let id = ddl
                    .expected_owner
                    .ok_or_else(|| Session::index_ddl_error("rebuild lacks owner"))?;
                changes.push(IndexOwnerChange::Rebuild(capture(before, id)?));
            } else if ddl.create {
                if ddl.configuration.is_some() {
                    let id =
                        ddl.owner_result.get().copied().ok_or_else(|| {
                            Session::index_ddl_error("prepared create lacks owner")
                        })?;
                    changes.push(IndexOwnerChange::Create(capture(after, id)?));
                }
            } else if let Some(id) = ddl.expected_owner {
                changes.push(IndexOwnerChange::Drop(capture(before, id)?));
            }
        }
        let mut text: Vec<crate::database::index_commit_wire::TextPostimage> = Vec::new();
        #[cfg(feature = "text-index")]
        for (store, key, birth, bytes) in prepared.text_postimages() {
            let (_, graph) = self
                .index_coordinates
                .iter()
                .find(|(captured, _)| std::ptr::eq(captured.as_ref(), store))
                .ok_or_else(|| Session::index_ddl_error("prepared Text lacks a captured graph"))?;
            let (label, property) =
                grafeo_core::graph::lpg::decode_index_key(key).ok_or_else(|| {
                    Session::index_ddl_error("prepared Text has invalid physical key")
                })?;
            let key =
                grafeo_core::graph::lpg::PhysicalIndexKey::text(graph.clone(), label, property);
            let owner = after.physical_index_owner(&key).ok_or_else(|| {
                Session::index_ddl_error("prepared Text lacks exact logical owner")
            })?;
            text.push(crate::database::index_commit_wire::TextPostimage {
                owner: capture(after, owner.id)?,
                birth,
                payload: bytes.to_vec(),
            });
        }
        #[cfg(not(feature = "text-index"))]
        let _ = (&prepared, &mut text, &self.index_coordinates);
        let mut vectors: Vec<crate::database::index_commit_wire::VectorPostimage> = Vec::new();
        #[cfg(feature = "vector-index")]
        for (store, key, complete, bytes) in prepared.vector_postimages() {
            let (_, graph) = self
                .index_coordinates
                .iter()
                .find(|(captured, _)| std::ptr::eq(captured.as_ref(), store))
                .ok_or_else(|| Session::index_ddl_error("prepared Vector lacks captured graph"))?;
            let (label, property) = grafeo_core::graph::lpg::decode_index_key(key)
                .ok_or_else(|| Session::index_ddl_error("prepared Vector key is invalid"))?;
            let key = PhysicalIndexKey::vector(graph.clone(), label, property);
            let owner = after.physical_index_owner(&key).ok_or_else(|| {
                Session::index_ddl_error("prepared Vector lacks exact logical owner")
            })?;
            vectors.push(crate::database::index_commit_wire::VectorPostimage {
                owner: capture(after, owner.id)?,
                complete,
                payload: bytes.to_vec(),
            });
        }
        text.sort_unstable_by_key(|image| image.owner.id);
        vectors.sort_unstable_by_key(|image| image.owner.id);
        if changes.is_empty() && !self.index_wal_required {
            return Ok(());
        }
        let owners = IndexOwnerBatch {
            expected_floor: before.index_allocator_high_water(),
            next_floor: after.index_allocator_high_water(),
            changes,
        };
        let payload = crate::database::index_commit_wire::IndexCommitBatch {
            frontier: self.frontier,
            commit: self.commit,
            owners,
            text,
            vectors,
        }
        .encode()?;
        wal.log(&grafeo_storage::wal::WalRecord::IndexOwnerBatch {
            transaction_id,
            payload,
        })?;
        Ok(())
    }

    pub(super) fn prepare<'catalog, 'workspace>(
        &self,
        session: &'catalog Session,
        workspace: &'workspace mut CatalogWorkspace,
    ) -> Result<PreparedLogicalCatalog<'catalog, 'workspace>> {
        let catalog_changed = session.catalog_changed();
        if !catalog_changed
            && self.bindings.is_empty()
            && self.dropped_binding_roots.is_empty()
            && self.pending.iter().all(|ddl| ddl.rebuild)
        {
            if self.pending.is_empty() && {
                #[cfg(feature = "wal")]
                {
                    !self.index_wal_required
                }
                #[cfg(not(feature = "wal"))]
                {
                    true
                }
            } {
                return Ok(PreparedLogicalCatalog::NoCatalogChange);
            }
            let owners = session.catalog.read();
            if !self.pending.is_empty() {
                self.validate(owners.view())?;
            }
            return Ok(PreparedLogicalCatalog::OwnersPinned(owners));
        }
        let edit = if catalog_changed {
            let snapshot = session.transaction_catalog.lock();
            let snapshot = snapshot.as_ref().ok_or_else(|| {
                Session::index_ddl_error("changed catalog lacks its transaction snapshot")
            })?;
            session.catalog.prepare_transaction_edit(
                &snapshot.base,
                snapshot.current.snapshot(),
                workspace,
            )
        } else {
            session.catalog.prepare_edit(workspace)
        }
        .map_err(|error| Session::index_ddl_error(error.to_string()))?;
        self.validate(edit.candidate().read().view())?;
        if !self.dropped_binding_roots.is_empty() {
            for (path, _) in edit.view().all_graph_type_bindings() {
                if self
                    .dropped_binding_roots
                    .iter()
                    .any(|root| path.components().starts_with(root.components()))
                    && !self.bindings.contains_key(&path)
                {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                            "graph type binding for {path:?} appeared after subtree DROP"
                        )),
                    ));
                }
            }
        }
        let candidate = edit.candidate();
        for ddl in self.pending {
            if ddl.rebuild {
                continue;
            }
            if ddl.create {
                // A transient create erased by a later operation has no final
                // physical image and must not allocate a public owner.
                let Some(configuration) = &ddl.configuration else {
                    continue;
                };
                let label = candidate
                    .get_or_create_label(&ddl.label)
                    .map_err(|error| Session::index_ddl_error(error.to_string()))?;
                let property = candidate
                    .get_or_create_property_key(&ddl.property)
                    .map_err(|error| Session::index_ddl_error(error.to_string()))?;
                let owner = candidate
                    .create_index(
                        ddl.name.as_deref(),
                        label,
                        property,
                        ddl.graph.clone(),
                        configuration.clone(),
                    )
                    .map_err(|error| Session::index_ddl_error(error.to_string()))?;
                ddl.owner_result.set(owner).map_err(|_| {
                    Session::index_ddl_error("index creation result was already prepared")
                })?;
            } else if let Some(owner) = ddl.expected_owner {
                candidate.drop_index(owner);
            }
        }
        for (name, binding) in self.bindings {
            if !edit
                .view()
                .graph_type_binding_matches(name, binding.expected.as_deref())
            {
                return Err(Session::index_ddl_error(format!(
                    "graph type binding for {name:?} changed during preparation"
                )));
            }
            if let Some(replacement) = binding.replacement.as_deref() {
                candidate
                    .validate_graph_type_binding_target(replacement)
                    .map_err(|error| Session::index_ddl_error(error.to_string()))?;
            }
            let expected = candidate.get_graph_type_binding(name);
            if !candidate.publish_graph_type_binding_if_same(
                name,
                expected.as_deref(),
                binding.replacement.clone(),
            ) {
                return Err(Session::index_ddl_error(format!(
                    "graph type binding for {name:?} changed during preparation"
                )));
            }
        }
        Ok(PreparedLogicalCatalog::Changed(edit.finish()))
    }

    fn validate(&self, catalog: CatalogRead<'_>) -> Result<()> {
        let mut names: std::collections::HashMap<_, _> = catalog
            .all_indexes()
            .into_iter()
            .map(|definition| (definition.name, Some(definition.id)))
            .collect();
        let mut owners: std::collections::HashMap<
            PhysicalIndexKey,
            std::collections::HashSet<String>,
        > = std::collections::HashMap::new();
        for definition in catalog.all_indexes() {
            owners
                .entry(definition.key)
                .or_default()
                .insert(definition.name);
        }
        for ddl in self.pending {
            if ddl.rebuild {
                Session::validate_index_rebuild_owner(catalog, ddl)?;
                continue;
            }
            if ddl.create
                && ddl
                    .owner_schema
                    .as_ref()
                    .is_some_and(|owner| !catalog.schema_exists(&owner.name))
            {
                return Err(Session::index_ddl_error(
                    "owning schema disappeared during index preparation",
                ));
            }
            let key = Session::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind);
            if let Some(name) = &ddl.name {
                let changed = if ddl.create {
                    names.insert(name.clone(), None).is_some()
                } else {
                    names.remove(name) != Some(ddl.expected_owner)
                };
                if changed {
                    return Err(Session::index_ddl_error(format!(
                        "index '{name}' changed during preparation"
                    )));
                }
            }
            let current = owners.entry(key).or_default();
            if ddl.create {
                if !current.is_empty() {
                    return Err(Session::index_ddl_error(
                        "physical index gained a logical owner during preparation",
                    ));
                }
                current.insert(ddl.name.clone().unwrap_or_else(|| "<unnamed>".to_owned()));
            } else if let Some(name) = &ddl.name {
                current.remove(name);
            } else {
                if current.iter().any(|owner| owner != "<unnamed>") {
                    return Err(Session::index_ddl_error(
                        "unnamed physical index gained a logical owner during preparation",
                    ));
                }
                current.remove("<unnamed>");
            }
        }
        Ok(())
    }
}

fn row(rows: &[Node], id: NodeId) -> Option<&Node> {
    rows.binary_search_by_key(&id, |node| node.id)
        .ok()
        .map(|index| &rows[index])
}

fn property<'row>(node: Option<&'row Node>, key: &str) -> Option<&'row Value> {
    node.and_then(|node| node.get_property(key))
        .filter(|value| !value.is_null())
}

#[cfg(feature = "text-index")]
fn text<'row>(node: Option<&'row Node>, label: &str, key: &str) -> Option<&'row str> {
    match property(node.filter(|node| node.has_label(label)), key) {
        Some(Value::String(text)) => Some(text.as_str()),
        _ => None,
    }
}

#[cfg(feature = "vector-index")]
fn vector_value(node: Option<&Node>, label: &str, key: &str) -> Option<Arc<[f32]>> {
    property(node.filter(|node| node.has_label(label)), key).and_then(value_to_vector)
}

fn observe(
    store: &LpgStore,
    key: &PhysicalIndexKey,
) -> Option<grafeo_core::graph::lpg::IndexRegistrationObservation> {
    match key.family() {
        PhysicalIndexFamily::Property => store.observe_property_index(key.property_name()),
        PhysicalIndexFamily::Text => {
            #[cfg(feature = "text-index")]
            {
                store.observe_text_index(key.label()?, key.property_name())
            }
            #[cfg(not(feature = "text-index"))]
            {
                None
            }
        }
        PhysicalIndexFamily::Vector => {
            #[cfg(feature = "vector-index")]
            {
                store.observe_vector_index(key.label()?, key.property_name())
            }
            #[cfg(not(feature = "vector-index"))]
            {
                None
            }
        }
    }
}

fn registry_key(key: &PhysicalIndexKey) -> Result<IndexRegistryKey> {
    Ok(match key.family() {
        PhysicalIndexFamily::Property => {
            IndexRegistryKey::Property(PropertyKey::new(key.property_name()))
        }
        #[cfg(feature = "text-index")]
        PhysicalIndexFamily::Text => IndexRegistryKey::Text {
            label: key
                .label()
                .ok_or_else(|| Session::index_ddl_error("Text key lacks a label"))?
                .to_owned(),
            property: key.property_name().to_owned(),
        },
        #[cfg(feature = "vector-index")]
        PhysicalIndexFamily::Vector => IndexRegistryKey::Vector {
            label: key
                .label()
                .ok_or_else(|| Session::index_ddl_error("Vector key lacks a label"))?
                .to_owned(),
            property: key.property_name().to_owned(),
        },
        #[cfg(not(all(feature = "text-index", feature = "vector-index")))]
        _ => {
            return Err(Session::index_ddl_error(
                "physical index family is unavailable",
            ));
        }
    })
}

fn build_contents(
    ddl: &PendingIndexDdl,
    store: &LpgStore,
    rows: &[Node],
    commit: EpochId,
) -> Result<(IndexRegistryContents, IndexConfiguration)> {
    match ddl.kind {
        PendingIndexKind::Property | PendingIndexKind::BTree => {
            let mut image = store.property_index_image(&ddl.property)?;
            image.project_final_rows(rows, &ddl.property, commit)?;
            Ok((
                IndexRegistryContents::PropertyHistory(image),
                if matches!(ddl.kind, PendingIndexKind::BTree) {
                    IndexConfiguration::BTree
                } else {
                    IndexConfiguration::Property
                },
            ))
        }
        PendingIndexKind::Text {
            min_token_length: _min_token_length,
        } => {
            #[cfg(feature = "text-index")]
            {
                let configuration =
                    ddl.configuration
                        .clone()
                        .unwrap_or_else(|| IndexConfiguration::Text {
                            config: grafeo_core::index::text::BM25Config::default(),
                            min_token_length: _min_token_length.unwrap_or(2),
                        });
                let IndexConfiguration::Text {
                    config,
                    min_token_length,
                } = &configuration
                else {
                    return Err(Session::index_ddl_error(
                        "text rebuild owner has a different configuration family",
                    ));
                };
                configuration
                    .validate()
                    .map_err(|error| Session::index_ddl_error(error.to_string()))?;
                let mut index = grafeo_core::index::text::InvertedIndex::with_simple_tokenizer(
                    config.clone(),
                    *min_token_length,
                );
                for node in rows {
                    if let Some(text) = text(Some(node), &ddl.label, &ddl.property) {
                        index.insert_versioned(node.id, text, commit, None);
                    }
                }
                Ok((IndexRegistryContents::Text(index), configuration))
            }
            #[cfg(not(feature = "text-index"))]
            Err(Session::index_ddl_error(
                "Text index support requires the 'text-index' feature",
            ))
        }
        PendingIndexKind::Vector { .. } => {
            #[cfg(feature = "vector-index")]
            {
                let index = Session::build_pending_vector_contents(ddl, rows)?;
                let configuration = IndexConfiguration::Vector {
                    config: index.config().clone(),
                    quantization: index
                        .quantization_type()
                        .unwrap_or(grafeo_core::index::vector::QuantizationType::None),
                };
                Ok((IndexRegistryContents::Vector(index), configuration))
            }
            #[cfg(not(feature = "vector-index"))]
            Err(Session::index_ddl_error(
                "Vector index support requires the 'vector-index' feature",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_capture_qualifies_nested_detached_and_cancelled_prefixes()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = crate::database::GrafeoDB::new_in_memory();
        let session = db.session();
        let parent_path = GraphPath::from_components(&["a"])?;
        let child_path = GraphPath::from_components(&["a", "b"])?;
        let literal_path = GraphPath::from_components(&["a/b"])?;
        let parent = Arc::new(LpgStore::new()?);
        let attached = parent.graph_or_create("b")?;
        let detached = Arc::new(LpgStore::new()?);
        let wrong_incarnation = Arc::new(LpgStore::new()?);
        session.cancelled_created_graphs.lock().insert(
            parent_path,
            vec![Arc::clone(&wrong_incarnation), Arc::clone(&parent)],
        );
        session
            .cancelled_created_graphs
            .lock()
            .insert(child_path.clone(), vec![Arc::clone(&detached)]);
        // A slash inside one name must not qualify as the nested prefix,
        // even when the candidate happens to have the requested identity.
        session
            .cancelled_created_graphs
            .lock()
            .insert(literal_path, vec![Arc::clone(&attached)]);

        let mut capture = EngineCommitCapture::default();
        capture.remember(&session, Arc::clone(&detached), child_path.clone(), false)?;
        capture.remember(&session, Arc::clone(&attached), child_path.clone(), false)?;
        let detached_capture = capture
            .stores
            .iter()
            .find(|captured| Arc::ptr_eq(&captured.store, &detached))
            .ok_or("detached child capture is absent")?;
        assert!(Arc::ptr_eq(&detached_capture.graph_root, &detached));
        assert_eq!(detached_capture.graph_suffix, GraphPath::root());
        assert_eq!(detached_capture.graph, child_path);
        let attached_capture = capture
            .stores
            .iter()
            .find(|captured| Arc::ptr_eq(&captured.store, &attached))
            .ok_or("attached child capture is absent")?;
        assert!(Arc::ptr_eq(&attached_capture.graph_root, &parent));
        assert_eq!(
            attached_capture.graph_suffix,
            GraphPath::from_components(&["b"])?
        );
        assert_eq!(attached_capture.graph, child_path);
        Ok(())
    }
}
