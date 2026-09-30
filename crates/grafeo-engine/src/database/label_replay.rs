//! Borrowed scheduling of authoritative committed label images.
//!
//! Creation intents and deletes can precede the commit-time image record. Only
//! labels move: all other WAL effects retain their original replay position.

use std::collections::HashMap;

use arcstr::ArcStr;
use grafeo_common::types::{EpochId, GraphPath, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result};
use grafeo_storage::wal::{LpgMutationOp, WalEntry, WalRecord};

use super::{GrafeoDB, LpgRecoveryTarget};

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct NodeKey<'a> {
    transaction: TransactionId,
    graph: &'a GraphPath,
    incarnation: usize,
    id: NodeId,
}

#[derive(Default)]
struct Images<'a> {
    records: Vec<(usize, bool, &'a [Vec<String>])>,
    create: Option<usize>,
    delete: Option<usize>,
}

struct Append<'a> {
    id: NodeId,
    epoch: EpochId,
    images: &'a [Vec<String>],
}

pub(super) struct LabelReplaySchedule<'a> {
    births: HashMap<usize, &'a [String]>,
    appends: HashMap<usize, Vec<Append<'a>>>,
}

fn invalid(reason: &str) -> Error {
    Error::Serialization(format!("invalid WAL label images: {reason}"))
}

impl<'a> LabelReplaySchedule<'a> {
    pub(super) fn prepare(
        records: &'a [WalRecord],
        initial_graphs: &[GraphPath],
        epochs: &HashMap<TransactionId, EpochId>,
    ) -> Result<Self> {
        for record in records {
            record.validate_recovery().map_err(Error::Serialization)?;
            if let WalRecord::LpgMutation {
                transaction_id,
                op: LpgMutationOp::PublishGraph,
                ..
            } = record
            {
                record.validate_recovery().map_err(Error::Serialization)?;
                if epochs
                    .get(transaction_id)
                    .is_none_or(|epoch| *epoch == EpochId::PENDING)
                {
                    return Err(invalid("graph publication lacks a committed epoch"));
                }
            }
        }
        let mut groups = HashMap::<NodeKey<'a>, Images<'a>>::new();
        walk(records, initial_graphs, |index, key, op| {
            if let LpgMutationOp::NodeLabelImages { birth, images, .. } = op {
                records[index]
                    .validate_recovery()
                    .map_err(Error::Serialization)?;
                if key.transaction == TransactionId::SYSTEM
                    || !epochs.contains_key(&key.transaction)
                {
                    return Err(invalid("image record has no committed transaction epoch"));
                }
                groups
                    .entry(key)
                    .or_default()
                    .records
                    .push((index, *birth, images));
            }
            Ok(())
        })?;
        if groups.is_empty() {
            return Ok(Self {
                births: HashMap::new(),
                appends: HashMap::new(),
            });
        }

        // The second scan stores structural positions only for nodes with image
        // records; ordinary WAL traffic does not create per-node scratch.
        walk(records, initial_graphs, |index, key, op| {
            let Some(group) = groups.get_mut(&key) else {
                return Ok(());
            };
            match op {
                LpgMutationOp::CreateNode { .. } => {
                    if group.create.replace(index).is_some() {
                        return Err(invalid("ambiguous creation intent"));
                    }
                }
                LpgMutationOp::DeleteNode { .. } => {
                    if group.delete.replace(index).is_some() {
                        return Err(invalid("ambiguous node deletion"));
                    }
                }
                LpgMutationOp::AddNodeLabel { .. } | LpgMutationOp::RemoveNodeLabel { .. } => {
                    return Err(invalid(
                        "intent label deltas mixed with authoritative images",
                    ));
                }
                _ => {}
            }
            Ok(())
        })?;

        let mut schedule = Self {
            births: HashMap::new(),
            appends: HashMap::new(),
        };
        for (key, group) in groups {
            let epoch = epochs
                .get(&key.transaction)
                .copied()
                .ok_or_else(|| invalid("image record lacks commit epoch"))?;
            if epoch == EpochId::PENDING {
                return Err(invalid("image record uses the PENDING commit epoch"));
            }
            let mut saw_birth = false;
            for (ordinal, (index, birth, images)) in group.records.into_iter().enumerate() {
                let mut remaining = images;
                if birth {
                    if saw_birth || ordinal != 0 {
                        return Err(invalid("multiple authoritative birth images"));
                    }
                    let create = group.create.ok_or_else(|| {
                        invalid("birth has no matching creation intent in its graph incarnation")
                    })?;
                    if create >= index || group.delete.is_some_and(|delete| delete < create) {
                        return Err(invalid(
                            "birth precedes creation or follows a conflicting lifetime",
                        ));
                    }
                    let (first, rest) = images
                        .split_first()
                        .ok_or_else(|| invalid("empty birth image sequence"))?;
                    schedule.births.insert(create, first);
                    remaining = rest;
                    saw_birth = true;
                } else if group.create.is_some_and(|create| index < create) {
                    return Err(invalid("label images precede creation"));
                }
                if !remaining.is_empty() {
                    let position = group.delete.map_or(index, |delete| delete.min(index));
                    schedule.appends.entry(position).or_default().push(Append {
                        id: key.id,
                        epoch,
                        images: remaining,
                    });
                }
            }
        }
        Ok(schedule)
    }

    pub(super) fn apply(
        &self,
        index: usize,
        target: &LpgRecoveryTarget,
        op: &LpgMutationOp,
    ) -> Result<()> {
        if let Some(appends) = self.appends.get(&index) {
            for append in appends {
                for labels in append.images {
                    let labels: Vec<ArcStr> = labels
                        .iter()
                        .map(|label| ArcStr::from(label.as_str()))
                        .collect();
                    match target {
                        LpgRecoveryTarget::Flat(store) => {
                            store.replay_node_labels_at_epoch(append.id, append.epoch, &labels)?;
                        }
                        #[cfg(feature = "compact-store")]
                        LpgRecoveryTarget::Layered(store) => {
                            store.replay_node_labels_at_epoch(append.id, append.epoch, &labels)?;
                        }
                    }
                }
            }
        }
        if let LpgMutationOp::CreateNode { id, .. } = op
            && let Some(labels) = self.births.get(&index)
        {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            return target.recover_create_node_with_id(*id, &labels);
        }
        if matches!(
            op,
            LpgMutationOp::NodeLabelImages { .. } | LpgMutationOp::PublishGraph
        ) {
            // prepare() validated and placed every image and qualified each
            // graph publication. The caller has already synchronized the exact
            // destination epoch; publication adds no entity/history mutation.
            return Ok(());
        }
        GrafeoDB::apply_lpg_mutation_op(target, op)
    }
}

fn walk<'a>(
    records: &'a [WalRecord],
    initial_graphs: &[GraphPath],
    mut visit: impl FnMut(usize, NodeKey<'a>, &'a LpgMutationOp) -> Result<()>,
) -> Result<()> {
    // A generation changes only when an absent path acquires a new store.
    // Retiring a parent also retires every descendant incarnation.
    let mut graphs: HashMap<GraphPath, (usize, bool)> = initial_graphs
        .iter()
        .map(|path| (path.clone(), (0, true)))
        .collect();
    graphs.insert(GraphPath::root(), (0, true));
    for (index, record) in records.iter().enumerate() {
        match record {
            WalRecord::CreateLpgGraph { graph, .. } => {
                ensure_graph(&mut graphs, graph)?;
            }
            WalRecord::DropLpgGraph { graph, .. } => {
                drop_graph(&mut graphs, graph)?;
            }
            WalRecord::SetGraphTypeBinding {
                graph, graph_type, ..
            } => {
                // A committed graph drop is followed by its binding removals,
                // including descendants retired by the parent's removal.
                if graph_type.is_some() || !graphs.contains_key(graph) {
                    existing_graph(&graphs, graph)?;
                }
            }
            WalRecord::CatalogBatchV3 {
                created_graphs,
                dropped_graphs,
                ..
            } => {
                for path in dropped_graphs {
                    drop_graph(&mut graphs, path)?;
                }
                for path in created_graphs {
                    ensure_graph(&mut graphs, path)?;
                }
            }
            _ => {}
        }
        let (transaction, graph, op) = match record {
            WalRecord::LpgMutation {
                transaction_id,
                graph,
                op,
            } => (*transaction_id, graph, op),
            _ => continue,
        };
        if matches!(op, LpgMutationOp::PublishGraph) {
            if graph.components().is_empty() {
                return Err(invalid("graph publication targets an absent named graph"));
            }
            existing_graph(&graphs, graph)?;
            continue;
        }
        let incarnation = existing_graph(&graphs, graph)?;
        let id = match op {
            LpgMutationOp::CreateNode { id, .. }
            | LpgMutationOp::DeleteNode { id }
            | LpgMutationOp::NodeLabelImages { id, .. }
            | LpgMutationOp::AddNodeLabel { id, .. }
            | LpgMutationOp::RemoveNodeLabel { id, .. } => *id,
            _ => continue,
        };
        visit(
            index,
            NodeKey {
                transaction,
                graph,
                incarnation,
                id,
            },
            op,
        )?;
    }
    Ok(())
}

fn existing_graph(graphs: &HashMap<GraphPath, (usize, bool)>, path: &GraphPath) -> Result<usize> {
    graphs
        .get(path)
        .filter(|(_, present)| *present)
        .map(|(generation, _)| *generation)
        .ok_or_else(|| GrafeoDB::missing_wal_graph(path))
}

fn lifecycle_parent(path: &GraphPath) -> Result<GraphPath> {
    path.parent()
        .map_err(|error| Error::Serialization(error.to_string()))?
        .ok_or_else(|| invalid("root graph has no lifecycle parent"))
}

fn ensure_graph(graphs: &mut HashMap<GraphPath, (usize, bool)>, path: &GraphPath) -> Result<usize> {
    existing_graph(graphs, &lifecycle_parent(path)?)?;
    if let Some((generation, present)) = graphs.get_mut(path) {
        if !*present {
            *generation = generation
                .checked_add(1)
                .ok_or_else(|| invalid("graph incarnation space exhausted"))?;
            *present = true;
        }
        return Ok(*generation);
    }
    graphs.insert(path.clone(), (0, true));
    Ok(0)
}

fn drop_graph(graphs: &mut HashMap<GraphPath, (usize, bool)>, path: &GraphPath) -> Result<()> {
    existing_graph(graphs, &lifecycle_parent(path)?)?;
    for (candidate, (_, present)) in graphs.iter_mut() {
        if candidate.components().starts_with(path.components()) {
            *present = false;
        }
    }
    graphs.entry(path.clone()).or_insert((0, false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::catalog::Catalog;
    use grafeo_core::graph::lpg::LpgStore;

    fn path(graph: Option<&str>) -> GraphPath {
        graph.map_or_else(GraphPath::root, |name| {
            GraphPath::from_components(&[name]).unwrap()
        })
    }

    fn mutation(graph: Option<&str>, op: LpgMutationOp) -> WalRecord {
        WalRecord::lpg(TransactionId::new(7), path(graph), op)
    }

    fn create(graph: Option<&str>) -> WalRecord {
        mutation(
            graph,
            LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Intent".into()],
            },
        )
    }

    fn images(graph: Option<&str>, birth: bool, labels: &[&[&str]]) -> WalRecord {
        mutation(
            graph,
            LpgMutationOp::NodeLabelImages {
                id: NodeId::new(1),
                birth,
                images: labels
                    .iter()
                    .map(|image| image.iter().map(|label| (*label).to_owned()).collect())
                    .collect(),
            },
        )
    }

    fn committed() -> WalRecord {
        WalRecord::Committed {
            transaction_id: TransactionId::new(7),
            epoch: EpochId::new(5),
        }
    }

    fn replay(store: &Arc<LpgStore>, records: &[WalRecord]) -> Result<()> {
        let catalog = Catalog::new();
        replay_with_catalog(store, &catalog, records)
    }

    fn replay_with_catalog(
        store: &Arc<LpgStore>,
        catalog: &Catalog,
        records: &[WalRecord],
    ) -> Result<()> {
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        #[cfg(feature = "triple-store")]
        let projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
        GrafeoDB::apply_wal_records(
            store,
            catalog,
            #[cfg(feature = "triple-store")]
            &rdf_store,
            #[cfg(feature = "triple-store")]
            &projections,
            records,
        )
    }

    fn history(store: &LpgStore) -> Vec<(EpochId, Vec<String>)> {
        store
            .node_label_history(NodeId::new(1))
            .into_iter()
            .map(|(epoch, labels)| (epoch, labels.iter().map(ToString::to_string).collect()))
            .collect()
    }

    #[test]
    fn birth_replacement_precedes_properties_and_retains_exact_zero_width_images() {
        use grafeo_common::types::Value;
        let store = Arc::new(LpgStore::new().unwrap());
        replay(
            &store,
            &[
                create(None),
                mutation(
                    None,
                    LpgMutationOp::SetNodeProperty {
                        id: NodeId::new(1),
                        key: "p".into(),
                        value: Value::Int64(42),
                    },
                ),
                mutation(None, LpgMutationOp::DeleteNode { id: NodeId::new(1) }),
                images(None, true, &[&["Born"], &["Final"], &["Final"], &[]]),
                committed(),
            ],
        )
        .unwrap();
        assert_eq!(
            history(&store),
            vec![
                (EpochId::new(5), vec!["Born".into()]),
                (EpochId::new(5), vec!["Final".into()]),
                (EpochId::new(5), vec!["Final".into()]),
                (EpochId::new(5), Vec::new()),
            ]
        );
        assert!(store.get_node(NodeId::new(1)).is_none());
        assert!(store.nodes_by_label("Born").is_empty());
        assert!(store.nodes_by_label("Final").is_empty());
    }

    #[test]
    fn existing_images_move_before_deletion_without_inventing_intermediates() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.sync_epoch(EpochId::new(2));
        store
            .create_node_with_id(NodeId::new(1), &["A", "B"])
            .unwrap();
        replay(
            &store,
            &[
                mutation(None, LpgMutationOp::DeleteNode { id: NodeId::new(1) }),
                images(None, false, &[&["C", "D"]]),
                committed(),
            ],
        )
        .unwrap();
        assert_eq!(
            history(&store),
            vec![
                (EpochId::new(2), vec!["A".into(), "B".into()]),
                (EpochId::new(5), vec!["C".into(), "D".into()]),
            ]
        );
        assert!(store.get_node(NodeId::new(1)).is_none());
    }

    #[test]
    fn birth_images_keep_default_and_empty_named_graph_distinct() {
        let store = Arc::new(LpgStore::new().unwrap());
        replay(
            &store,
            &[
                create(None),
                WalRecord::CreateLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: path(Some("")),
                    transaction_id: TransactionId::new(7),
                },
                create(Some("")),
                images(None, true, &[&["Root"]]),
                images(Some(""), true, &[&["EmptyName"]]),
                committed(),
            ],
        )
        .unwrap();
        assert_eq!(
            history(&store),
            vec![(EpochId::new(5), vec!["Root".into()])]
        );
        assert_eq!(
            history(&store.graph("").unwrap()),
            vec![(EpochId::new(5), vec!["EmptyName".into()])]
        );
    }

    #[test]
    fn immediate_system_label_changes_remain_individual_images() {
        let store = Arc::new(LpgStore::new().unwrap());
        replay(
            &store,
            &[
                create(None),
                images(None, true, &[&["A"]]),
                committed(),
                WalRecord::lpg(
                    TransactionId::SYSTEM,
                    GraphPath::root(),
                    LpgMutationOp::AddNodeLabel {
                        id: NodeId::new(1),
                        label: "B".into(),
                    },
                ),
                WalRecord::lpg(
                    TransactionId::SYSTEM,
                    GraphPath::root(),
                    LpgMutationOp::RemoveNodeLabel {
                        id: NodeId::new(1),
                        label: "A".into(),
                    },
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            history(&store),
            vec![
                (EpochId::new(5), vec!["A".into()]),
                (EpochId::new(5), vec!["A".into(), "B".into()]),
                (EpochId::new(5), vec!["B".into()]),
            ]
        );
    }

    #[test]
    fn named_graph_publication_advances_only_its_cut_without_history_entries() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.sync_epoch(EpochId::new(2));
        store
            .create_node_with_id(NodeId::new(1), &["Root"])
            .unwrap();
        for name in ["", "untouched"] {
            store.create_graph(name).unwrap();
            let graph = store.graph(name).unwrap();
            graph.sync_epoch(EpochId::new(2));
            graph
                .create_node_with_id(NodeId::new(1), &["Named"])
                .unwrap();
        }
        let empty = store.graph("").unwrap();
        let untouched = store.graph("untouched").unwrap();
        let root_history = history(&store);
        let empty_history = history(&empty);
        let untouched_history = history(&untouched);
        replay(
            &store,
            &[mutation(Some(""), LpgMutationOp::PublishGraph), committed()],
        )
        .unwrap();
        assert_eq!(store.current_epoch(), EpochId::new(5));
        assert_eq!(empty.current_epoch(), EpochId::new(5));
        assert_eq!(untouched.current_epoch(), EpochId::new(2));
        assert_eq!(history(&store), root_history);
        assert_eq!(history(&empty), empty_history);
        assert_eq!(history(&untouched), untouched_history);
        assert_eq!(empty.node_count(), 1);
    }

    #[test]
    fn named_graph_publication_rejects_missing_graph_epoch_and_root_before_replay() {
        let groups = [
            vec![
                mutation(Some("missing"), LpgMutationOp::PublishGraph),
                committed(),
            ],
            vec![mutation(None, LpgMutationOp::PublishGraph), committed()],
            vec![mutation(Some(""), LpgMutationOp::PublishGraph)],
            vec![
                mutation(Some(""), LpgMutationOp::PublishGraph),
                WalRecord::Committed {
                    transaction_id: TransactionId::new(7),
                    epoch: EpochId::PENDING,
                },
            ],
            vec![
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: path(Some("")),
                    transaction_id: TransactionId::new(7),
                },
                mutation(Some(""), LpgMutationOp::PublishGraph),
                committed(),
            ],
        ];
        for records in groups {
            let store = Arc::new(LpgStore::new().unwrap());
            store.sync_epoch(EpochId::new(2));
            store.create_graph("").unwrap();
            let empty = store.graph("").unwrap();
            empty.sync_epoch(EpochId::new(2));
            assert!(replay(&store, &records).is_err());
            assert_eq!(store.current_epoch(), EpochId::new(2));
            assert_eq!(empty.current_epoch(), EpochId::new(2));
            assert!(store.graph("missing").is_none());
            assert!(Arc::ptr_eq(&empty, &store.graph("").unwrap()));
            assert!(history(&empty).is_empty());
        }
    }

    #[test]
    fn birth_images_match_only_the_surviving_named_incarnation() {
        let store = Arc::new(LpgStore::new().unwrap());
        replay(
            &store,
            &[
                WalRecord::CreateLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: path(Some("g")),
                    transaction_id: TransactionId::new(7),
                },
                create(Some("g")),
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: path(Some("g")),
                    transaction_id: TransactionId::new(7),
                },
                WalRecord::CreateLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(2),
                    graph: path(Some("g")),
                    transaction_id: TransactionId::new(7),
                },
                create(Some("g")),
                images(Some("g"), true, &[&["Survivor"]]),
                committed(),
            ],
        )
        .unwrap();
        assert_eq!(
            history(&store.graph("g").unwrap()),
            vec![(EpochId::new(5), vec!["Survivor".into()])]
        );
    }

    #[test]
    fn catalog_postimage_drop_cannot_supply_a_birth_from_the_retired_graph() {
        let records = [
            create(Some("g")),
            WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![],
                dropped_graph_incarnations: vec![grafeo_common::types::GraphIncarnationId::new(1)],
                version: 2,
                epoch: EpochId::new(5),
                catalog_state: Vec::new(),
                created_graphs: Vec::new(),
                dropped_graphs: vec![path(Some("g"))],
            },
            images(Some("g"), true, &[&["WrongIncarnation"]]),
            committed(),
        ];
        let epochs = HashMap::from([(TransactionId::new(7), EpochId::new(5))]);
        let error = LabelReplaySchedule::prepare(&records, &[path(Some("g"))], &epochs)
            .err()
            .unwrap();
        assert!(error.to_string().contains("absent named LPG graph"));
    }

    #[test]
    fn nested_birth_images_keep_literal_components_and_parent_incarnations() {
        let store = Arc::new(LpgStore::new().unwrap());
        let parent = GraphPath::from_components(&["literal"]).unwrap();
        let nested = GraphPath::from_components(&["literal", "slash"]).unwrap();
        let literal = path(Some("literal/slash"));
        let transaction_id = TransactionId::new(7);
        let create_graph = |graph: &GraphPath, incarnation| WalRecord::CreateLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(incarnation),
            graph: graph.clone(),
            transaction_id,
        };
        let create_node = |graph: &GraphPath| {
            WalRecord::lpg(
                transaction_id,
                graph.clone(),
                LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Intent".into()],
                },
            )
        };
        let birth = |graph: &GraphPath, label: &str| {
            WalRecord::lpg(
                transaction_id,
                graph.clone(),
                LpgMutationOp::NodeLabelImages {
                    id: NodeId::new(1),
                    birth: true,
                    images: vec![vec![label.into()]],
                },
            )
        };
        replay(
            &store,
            &[
                create_graph(&literal, 1),
                create_node(&literal),
                create_graph(&parent, 2),
                create_graph(&nested, 3),
                create_node(&nested),
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(2),
                    graph: parent.clone(),
                    transaction_id,
                },
                create_graph(&parent, 4),
                create_graph(&nested, 5),
                create_node(&nested),
                birth(&literal, "Literal"),
                birth(&nested, "Survivor"),
                committed(),
            ],
        )
        .unwrap();
        assert_eq!(
            history(&GrafeoDB::resolve_wal_graph(&store, &literal).unwrap()),
            vec![(EpochId::new(5), vec!["Literal".into()])]
        );
        assert_eq!(
            history(&GrafeoDB::resolve_wal_graph(&store, &nested).unwrap()),
            vec![(EpochId::new(5), vec!["Survivor".into()])]
        );
        assert_eq!(store.node_count(), 0);
    }

    #[test]
    fn absent_parent_and_schema_batch_lifecycle_reject_before_replay() {
        let orphan = GraphPath::from_components(&["missing", "child"]).unwrap();
        let transaction_id = TransactionId::new(7);
        let invalid_records = [
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: orphan.clone(),
                transaction_id,
            },
            WalRecord::lpg(
                transaction_id,
                orphan.clone(),
                LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: Vec::new(),
                },
            ),
            WalRecord::SetGraphTypeBinding {
                graph: orphan,
                graph_type: None,
                transaction_id,
            },
            WalRecord::CatalogBatchV2 {
                version: 1,
                records: vec![WalRecord::CreateLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: path(Some("hidden")),
                    transaction_id,
                }],
            },
        ];
        for invalid_record in invalid_records {
            let store = Arc::new(LpgStore::new().unwrap());
            assert!(replay(&store, &[create(None), invalid_record, committed()]).is_err());
            assert_eq!(store.node_count(), 0);
            assert!(store.named_graph_entries().is_empty());
        }
    }

    #[test]
    fn dropped_parent_cannot_supply_descendant_birth_to_recreated_topology() {
        let parent = path(Some("parent"));
        let nested = GraphPath::from_components(&["parent", "child"]).unwrap();
        let transaction_id = TransactionId::new(7);
        let records = [
            WalRecord::lpg(
                transaction_id,
                nested.clone(),
                LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: Vec::new(),
                },
            ),
            WalRecord::DropLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: parent.clone(),
                transaction_id,
            },
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: parent.clone(),
                transaction_id,
            },
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: nested.clone(),
                transaction_id,
            },
            WalRecord::lpg(
                transaction_id,
                nested.clone(),
                LpgMutationOp::NodeLabelImages {
                    id: NodeId::new(1),
                    birth: true,
                    images: vec![vec!["Retired".into()]],
                },
            ),
            committed(),
        ];
        let epochs = HashMap::from([(transaction_id, EpochId::new(5))]);
        let error = LabelReplaySchedule::prepare(&records, &[parent, nested], &epochs)
            .err()
            .unwrap();
        assert!(error.to_string().contains("no matching creation intent"));
    }

    #[test]
    fn parent_drop_retires_only_component_descendant_bindings() {
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Catalog::new();
        let parent = path(Some("parent"));
        let nested = GraphPath::from_components(&["parent", "child"]).unwrap();
        let literal = path(Some("parent/child"));
        catalog
            .register_graph_type(crate::catalog::GraphTypeDefinition {
                name: "Bound".into(),
                allowed_node_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                open: true,
            })
            .unwrap();
        store.create_graph("parent").unwrap();
        store
            .graph("parent")
            .unwrap()
            .create_graph("child")
            .unwrap();
        store.create_graph("parent/child").unwrap();
        for graph in [&parent, &nested, &literal] {
            catalog.bind_graph_type(graph, "Bound".into()).unwrap();
        }
        let transaction_id = TransactionId::new(7);
        replay_with_catalog(
            &store,
            &catalog,
            &[
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: parent.clone(),
                    transaction_id,
                },
                WalRecord::SetGraphTypeBinding {
                    graph: parent.clone(),
                    graph_type: None,
                    transaction_id,
                },
                WalRecord::SetGraphTypeBinding {
                    graph: nested.clone(),
                    graph_type: None,
                    transaction_id,
                },
                committed(),
            ],
        )
        .unwrap();
        assert!(store.graph("parent").is_none());
        assert!(store.graph("parent/child").is_some());
        assert!(catalog.get_graph_type_binding(&parent).is_none());
        assert!(catalog.get_graph_type_binding(&nested).is_none());
        assert_eq!(
            catalog.get_graph_type_binding(&literal).as_deref(),
            Some("Bound")
        );
    }

    #[test]
    fn malformed_birth_groups_reject_before_any_replay_mutation() {
        let bad_groups = [
            vec![images(None, true, &[&["A"]]), committed()],
            vec![
                create(None),
                create(None),
                images(None, true, &[&["A"]]),
                committed(),
            ],
            vec![
                create(None),
                images(None, true, &[&["A"]]),
                images(None, true, &[&["B"]]),
                committed(),
            ],
            vec![create(None), images(None, true, &[]), committed()],
            vec![
                create(None),
                images(None, true, &[&["A", "A"]]),
                committed(),
            ],
            vec![create(None), images(Some(""), true, &[&["A"]]), committed()],
            vec![create(None), images(None, true, &[&["A"]])],
            vec![
                create(Some("g")),
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: path(Some("g")),
                    transaction_id: TransactionId::new(7),
                },
                images(Some("g"), true, &[&["A"]]),
                committed(),
            ],
            vec![
                create(None),
                mutation(
                    None,
                    LpgMutationOp::AddNodeLabel {
                        id: NodeId::new(1),
                        label: "B".into(),
                    },
                ),
                images(None, true, &[&["A"]]),
                committed(),
            ],
        ];
        for records in bad_groups {
            let store = Arc::new(LpgStore::new().unwrap());
            assert!(replay(&store, &records).is_err());
            assert_eq!(store.node_count(), 0);
            assert!(store.named_graph_entries().is_empty());
        }
    }
}
