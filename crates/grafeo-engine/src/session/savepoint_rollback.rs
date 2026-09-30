//! Savepoint rollback through retained exact graph incarnations.

#[cfg(feature = "lpg")]
use super::{Arc, EdgeId, GraphPath, GraphStoreMut, LpgLifecycleSavepoint, LpgStore, NodeId};
use super::{Result, Session};

#[cfg(feature = "lpg")]
#[derive(Clone, Copy)]
enum SelectorLifecycle<'state> {
    Created,
    Dropped(&'state GraphPath, &'state Arc<LpgStore>),
    Cancelled,
}

/// A drop/cancellation of any ancestor makes its descendants absent. Otherwise
/// the deepest staged create supplies the detached incarnation for this path.
#[cfg(feature = "lpg")]
fn selector_lifecycle<'state>(
    state: &'state LpgLifecycleSavepoint,
    path: &GraphPath,
) -> Result<Option<SelectorLifecycle<'state>>> {
    let mut prefix = GraphPath::root();
    let mut selected = None;
    for component in path.components() {
        prefix = prefix
            .child(component)
            .map_err(|error| Session::index_ddl_error(error.to_string()))?;
        if state.pending_created_graphs.contains_key(&prefix) {
            selected = Some(SelectorLifecycle::Created);
        } else if let Some((path, dropped)) = state.pending_dropped_graphs.get_key_value(&prefix) {
            return Ok(Some(SelectorLifecycle::Dropped(path, dropped)));
        } else if state.cancelled_created_graphs.contains_key(&prefix) {
            return Ok(Some(SelectorLifecycle::Cancelled));
        }
    }
    Ok(selected)
}

impl Session {
    pub(super) fn rollback_to_savepoint_authorized(&self, name: &str) -> Result<()> {
        let transaction_id = self.current_transaction.lock().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        let mut savepoints = self.savepoints.lock();

        // Find the savepoint by name (search from the end for nested savepoints)
        let pos = savepoints
            .iter()
            .rposition(|sp| sp.name == name)
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(format!(
                        "Savepoint '{name}' not found"
                    )),
                )
            })?;

        let sp_state = savepoints[pos].clone();

        // Capture every exact graph incarnation that currently owns state for
        // this transaction before restoring the lifecycle maps. Incarnations
        // first touched (or detached/replaced) after the savepoint need full
        // cleanup, including Serializable trackers and layered overlays.
        #[cfg(feature = "lpg")]
        let mut current_stores: Vec<(Arc<LpgStore>, Arc<dyn GraphStoreMut>)> = Vec::new();
        #[cfg(feature = "lpg")]
        let mut remember_store = |store: Arc<LpgStore>, mutation_store: Arc<dyn GraphStoreMut>| {
            if !current_stores.iter().any(|(known_store, known_mutation)| {
                Arc::ptr_eq(known_store, &store) && Arc::ptr_eq(known_mutation, &mutation_store)
            }) {
                current_stores.push((store, mutation_store));
            }
        };
        #[cfg(feature = "lpg")]
        for graph_name in self.touched_graphs.lock().iter() {
            remember_store(
                self.resolve_store(graph_name)?,
                self.resolve_mutation_store(graph_name)?,
            );
        }
        #[cfg(feature = "lpg")]
        let current_lifecycle = self.lpg_lifecycle_snapshot();

        // Graph/schema selectors deliberately survive rollback-to-savepoint,
        // so their snapshot expectation must survive with them. Re-resolving
        // the surviving name after restoring the savepoint can otherwise
        // upgrade Snapshot Isolation/Serializable from Missing to a graph that
        // appeared later, or from an old exact Arc to a replacement.
        #[cfg(feature = "lpg")]
        enum SurvivingSelectorExpectation {
            Exact(Arc<LpgStore>),
            Missing,
            Refresh,
        }
        #[cfg(feature = "lpg")]
        let current_selector_key = self.active_graph_storage_key();
        #[cfg(feature = "lpg")]
        let read_committed = self.transaction_manager.isolation_level(transaction_id)
            == Some(crate::transaction::IsolationLevel::ReadCommitted);
        #[cfg(feature = "lpg")]
        let selector_expectations = {
            let components: Vec<&str> = current_selector_key
                .components()
                .iter()
                .map(String::as_str)
                .collect();
            let mut expectations: Vec<(GraphPath, SurvivingSelectorExpectation)> = Vec::new();
            for end in 1..=components.len() {
                let key = GraphPath::from_components(&components[..end])
                    .map_err(|error| Self::index_ddl_error(error.to_string()))?;
                let lifecycle_target = |prefix: &GraphPath, root: &Arc<LpgStore>| {
                    let mut target = Some(Arc::clone(root));
                    for component in key.components().iter().skip(prefix.components().len()) {
                        target = target.and_then(|store| store.graph(component));
                    }
                    target.map_or(
                        SurvivingSelectorExpectation::Missing,
                        SurvivingSelectorExpectation::Exact,
                    )
                };
                let saved_lifecycle = selector_lifecycle(&sp_state.lpg_lifecycle, &key)?;
                let current_action = selector_lifecycle(&current_lifecycle, &key)?;
                let saved_expectation = || {
                    if matches!(saved_lifecycle, Some(SelectorLifecycle::Created)) {
                        if let Some(exact) = sp_state.lpg_lifecycle.touched_named_graphs.get(&key) {
                            return Some(SurvivingSelectorExpectation::Exact(Arc::clone(exact)));
                        }
                        if sp_state.lpg_lifecycle.missing_named_graphs.contains(&key) {
                            return Some(SurvivingSelectorExpectation::Missing);
                        }
                        return Some(
                            Self::resolve_lpg_lifecycle_path(
                                &self.store,
                                &key,
                                &sp_state.lpg_lifecycle.pending_created_graphs,
                                &sp_state.lpg_lifecycle.pending_dropped_graphs,
                                &sp_state.lpg_lifecycle.cancelled_created_graphs,
                            )
                            .map_or(
                                SurvivingSelectorExpectation::Missing,
                                SurvivingSelectorExpectation::Exact,
                            ),
                        );
                    }
                    if matches!(
                        saved_lifecycle,
                        Some(SelectorLifecycle::Dropped(..) | SelectorLifecycle::Cancelled)
                    ) {
                        return Some(SurvivingSelectorExpectation::Missing);
                    }
                    if let Some(exact) = sp_state.lpg_lifecycle.touched_named_graphs.get(&key) {
                        return Some(SurvivingSelectorExpectation::Exact(Arc::clone(exact)));
                    }
                    sp_state
                        .lpg_lifecycle
                        .missing_named_graphs
                        .contains(&key)
                        .then_some(SurvivingSelectorExpectation::Missing)
                };
                let fallback = || {
                    if read_committed {
                        SurvivingSelectorExpectation::Refresh
                    } else {
                        SurvivingSelectorExpectation::Missing
                    }
                };

                let rolled_back_drop = || {
                    current_lifecycle
                        .pending_dropped_graphs
                        .iter()
                        .filter(|(prefix, _)| key.components().starts_with(prefix.components()))
                        .min_by_key(|(prefix, _)| prefix.components().len())
                        .map(|(prefix, store)| lifecycle_target(prefix, store))
                };
                let expectation = if let Some(SelectorLifecycle::Created) = current_action {
                    // The leaf's store alone is insufficient: retaining it
                    // under a replaced staged ancestor would cross graph
                    // incarnations. Every staged parent link must survive.
                    let survives = current_lifecycle
                        .pending_created_graphs
                        .iter()
                        .filter(|(prefix, _)| key.components().starts_with(prefix.components()))
                        .all(|(prefix, current)| {
                            sp_state
                                .lpg_lifecycle
                                .pending_created_graphs
                                .get(prefix)
                                .is_some_and(|saved| {
                                    Arc::ptr_eq(&saved.store, &current.store)
                                        && Arc::ptr_eq(&saved.parent, &current.parent)
                                })
                        });
                    if survives {
                        if let Some(exact) = current_lifecycle.touched_named_graphs.get(&key) {
                            SurvivingSelectorExpectation::Exact(Arc::clone(exact))
                        } else if current_lifecycle.missing_named_graphs.contains(&key) {
                            SurvivingSelectorExpectation::Missing
                        } else {
                            Self::resolve_lpg_lifecycle_path(
                                &self.store,
                                &key,
                                &current_lifecycle.pending_created_graphs,
                                &current_lifecycle.pending_dropped_graphs,
                                &current_lifecycle.cancelled_created_graphs,
                            )
                            .map_or(
                                SurvivingSelectorExpectation::Missing,
                                SurvivingSelectorExpectation::Exact,
                            )
                        }
                    } else if let Some(saved) = saved_expectation() {
                        saved
                    } else if let Some(rolled_back_drop) = rolled_back_drop() {
                        rolled_back_drop
                    } else {
                        fallback()
                    }
                } else if let Some(SelectorLifecycle::Dropped(prefix, dropped)) = current_action {
                    saved_expectation().unwrap_or_else(|| {
                        // The DROP happened after the savepoint. Rolling it
                        // back reveals the same exact pre-drop incarnation.
                        lifecycle_target(prefix, dropped)
                    })
                } else if matches!(current_action, Some(SelectorLifecycle::Cancelled)) {
                    saved_expectation().unwrap_or_else(fallback)
                } else if read_committed && saved_expectation().is_none() {
                    // This prefix was first observed after the savepoint.
                    // Once its later work is undone, Read Committed must
                    // resolve the surviving selector at a fresh cut, not
                    // reinstall a discarded statement's exact Arc.
                    SurvivingSelectorExpectation::Refresh
                } else if let Some(exact) = current_lifecycle.touched_named_graphs.get(&key) {
                    SurvivingSelectorExpectation::Exact(Arc::clone(exact))
                } else if current_lifecycle.missing_named_graphs.contains(&key) {
                    SurvivingSelectorExpectation::Missing
                } else {
                    saved_expectation().unwrap_or_else(fallback)
                };
                // A child cannot retain an exact selector below a parent
                // that vanished when a post-savepoint CREATE was undone.
                // Likewise a refreshed RC ancestor requires fresh descent.
                let expectation = match expectations.last().map(|(_, expected)| expected) {
                    Some(SurvivingSelectorExpectation::Missing) => {
                        SurvivingSelectorExpectation::Missing
                    }
                    Some(SurvivingSelectorExpectation::Refresh) => {
                        SurvivingSelectorExpectation::Refresh
                    }
                    Some(SurvivingSelectorExpectation::Exact(_)) | None => expectation,
                };
                expectations.push((key, expectation));
            }
            expectations
        };
        #[cfg(feature = "lpg")]
        for pending in current_lifecycle.pending_created_graphs.values() {
            remember_store(
                Arc::clone(&pending.store),
                Arc::clone(&pending.store) as Arc<dyn GraphStoreMut>,
            );
        }
        #[cfg(feature = "lpg")]
        for store in current_lifecycle
            .pending_dropped_graphs
            .values()
            .chain(current_lifecycle.touched_named_graphs.values())
        {
            remember_store(
                Arc::clone(store),
                Arc::clone(store) as Arc<dyn GraphStoreMut>,
            );
        }
        #[cfg(feature = "lpg")]
        for store in current_lifecycle
            .cancelled_created_graphs
            .values()
            .flatten()
        {
            remember_store(
                Arc::clone(store),
                Arc::clone(store) as Arc<dyn GraphStoreMut>,
            );
        }
        #[cfg(feature = "lpg")]
        for (_, store) in &current_lifecycle.superseded_graph_touches {
            remember_store(
                Arc::clone(store),
                Arc::clone(store) as Arc<dyn GraphStoreMut>,
            );
        }

        // Exact-target and selector preparation above is still abortable.
        // Record the rollback only after all new fallible resolution is done.
        #[cfg(feature = "wal")]
        self.log_wal_record(
            &grafeo_storage::wal::WalRecord::TransactionRollbackToSavepoint {
                transaction_id,
                name: name.to_string(),
            },
        )?;
        // SQL/GQL retains the target savepoint and releases only later ones.
        savepoints.truncate(pos + 1);
        drop(savepoints);

        // Roll back each graph that was captured in the savepoint.
        #[cfg(feature = "lpg")]
        for gs in &sp_state.graph_snapshots {
            // Replay property/label undo entries recorded after the savepoint
            // and restore deferred structural queues/base tombstones through
            // the exact incarnation and mutation layer captured at SAVEPOINT.
            gs.store
                .rollback_transaction_properties_to(transaction_id, gs.undo_log_position);

            // Restore the buffered property delta to the savepoint snapshot.
            // This discards any buffered property writes made after the savepoint,
            // including SET/REMOVE on existing properties and newly added properties.
            gs.mutation_store
                .tx_overlay_restore(transaction_id, gs.overlay_snapshot.clone());
            if let Err(message) = gs
                .mutation_store
                .tx_structural_restore(transaction_id, gs.structural_snapshot.clone())
            {
                // The WAL rollback marker is already durable, while the live
                // transaction has now failed to reach the state recovery will
                // reconstruct. This is a fail-stop durability boundary: never
                // allow an outer commit to authenticate the divergent runtime
                // cut. A full rollback remains available for cleanup.
                self.poison_durability();
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                        "rollback to savepoint '{name}' was recorded durably but structural state could not be restored: {message}; roll back and reopen before continuing"
                    )),
                ));
            }

            // Discard entities created after the savepoint
            let current_next_node = gs.store.peek_next_node_id();
            let current_next_edge = gs.store.peek_next_edge_id();

            let node_ids: Vec<NodeId> = (gs.next_node_id..current_next_node)
                .map(NodeId::new)
                .collect();
            let edge_ids: Vec<EdgeId> = (gs.next_edge_id..current_next_edge)
                .map(EdgeId::new)
                .collect();

            if !node_ids.is_empty() || !edge_ids.is_empty() {
                gs.store
                    .discard_entities_by_id(transaction_id, &node_ids, &edge_ids);
            }
        }

        #[cfg(feature = "lpg")]
        let saved_lifecycle_contains = |candidate: &Arc<LpgStore>| {
            sp_state
                .lpg_lifecycle
                .pending_created_graphs
                .values()
                .any(|saved| Arc::ptr_eq(&saved.store, candidate))
                || sp_state
                    .lpg_lifecycle
                    .pending_dropped_graphs
                    .values()
                    .chain(sp_state.lpg_lifecycle.touched_named_graphs.values())
                    .any(|saved| Arc::ptr_eq(saved, candidate))
                || sp_state
                    .lpg_lifecycle
                    .cancelled_created_graphs
                    .values()
                    .flatten()
                    .any(|saved| Arc::ptr_eq(saved, candidate))
                || sp_state
                    .lpg_lifecycle
                    .superseded_graph_touches
                    .iter()
                    .any(|(_, saved)| Arc::ptr_eq(saved, candidate))
        };
        #[cfg(feature = "lpg")]
        for (store, mutation_store) in current_stores {
            let captured = sp_state.graph_snapshots.iter().any(|snapshot| {
                Arc::ptr_eq(&snapshot.store, &store)
                    && Arc::ptr_eq(&snapshot.mutation_store, &mutation_store)
            });
            let was_touched = sp_state.graph_snapshots.iter().any(|snapshot| {
                snapshot.was_touched
                    && Arc::ptr_eq(&snapshot.store, &store)
                    && Arc::ptr_eq(&snapshot.mutation_store, &mutation_store)
            });
            if !captured && !saved_lifecycle_contains(&store) {
                self.discard_lpg_graph_transaction_via(&store, &mutation_store, transaction_id);
            } else if !was_touched {
                // A lifecycle-only store can be first selected after the
                // savepoint. Its structural state is restored above, but its
                // per-store SSI bridges are not part of the saved touch set.
                // Remove them now; the surviving current selector is
                // re-tracked below when appropriate.
                mutation_store.unregister_read_tracker(transaction_id);
                mutation_store.unregister_write_tracker(transaction_id);
            }
        }

        #[cfg(feature = "lpg")]
        self.restore_lpg_lifecycle_snapshot(sp_state.lpg_lifecycle.clone());
        #[cfg(feature = "lpg")]
        {
            *self.pending_index_ddl.lock() = sp_state.index_ddl_snapshot;
            *self.transaction_catalog.lock() = sp_state.catalog_snapshot;
        }
        #[cfg(feature = "lpg")]
        {
            *self.pending_projection_ddl.lock() = sp_state.projection_ddl_snapshot;
        }

        #[cfg(feature = "triple-store")]
        self.rdf_store
            .restore_transaction_savepoint(transaction_id, &sp_state.rdf_snapshot);

        // Truncate CDC event buffer to the savepoint position.
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            pending.truncate(sp_state.cdc_event_position);
        }

        // Restore the exact coordinate set, not the broader lifecycle snapshot
        // set used to rewind detached store state.
        #[cfg(feature = "lpg")]
        let mut restored_touched = sp_state.touched_graphs;

        #[cfg(feature = "lpg")]
        {
            let mut exact = self.touched_named_graphs.lock();
            let mut missing = self.missing_named_graphs.lock();
            let mut superseded = self.superseded_graph_touches.lock();
            for (key, expectation) in selector_expectations {
                let previous = match expectation {
                    SurvivingSelectorExpectation::Exact(store) => {
                        missing.remove(&key);
                        exact.insert(key.clone(), store)
                    }
                    SurvivingSelectorExpectation::Missing => {
                        missing.insert(key.clone());
                        exact.remove(&key)
                    }
                    SurvivingSelectorExpectation::Refresh => {
                        missing.remove(&key);
                        exact.remove(&key)
                    }
                };
                if let Some(previous) = previous
                    && !exact
                        .get(&key)
                        .is_some_and(|store| Arc::ptr_eq(store, &previous))
                    && restored_touched.contains(&key)
                {
                    // The saved target still owns its restored writes. Keep
                    // it even when the surviving selector names another
                    // incarnation, and let tracking admit the new target.
                    if !superseded
                        .iter()
                        .any(|(path, store)| path == &key && Arc::ptr_eq(store, &previous))
                    {
                        superseded.push((key.clone(), previous));
                    }
                    restored_touched.retain(|path| path != &key);
                }
            }
        }
        #[cfg(feature = "lpg")]
        {
            *self.touched_graphs.lock() = restored_touched;
        }

        // Graph/schema selectors deliberately survive rollback-to-savepoint.
        // Re-track that current coordinate after restoring the saved touched
        // set so a subsequent mutation is finalized and WAL/CDC-published
        // through the same exact incarnation instead of being orphaned.
        #[cfg(feature = "lpg")]
        if let Err(error) = self.track_graph_touch() {
            self.poison_durability();
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                    "savepoint rollback restored data but current graph tracking failed: {error}; roll back and reopen before continuing"
                )),
            ));
        }

        Ok(())
    }
}
