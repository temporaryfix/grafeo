//! Validate native lifetime transitions over the complete committed suffix
//! before replay can mutate any model, catalog or graph.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use grafeo_common::types::{GraphIncarnationId, GraphPath};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_storage::wal::{WalEntry, WalRecord};

pub(super) fn validate(
    initial: &[(GraphPath, Arc<LpgStore>)],
    floor: u64,
    records: &[WalRecord],
) -> Result<()> {
    let mut graphs: HashMap<_, _> = initial
        .iter()
        .map(|(path, graph)| (path.clone(), graph.graph_incarnation_id()))
        .collect();
    let mut allocated = HashSet::new();
    for record in records {
        record.validate_recovery().map_err(Error::Serialization)?;
        match record {
            WalRecord::CreateLpgGraph {
                graph, incarnation, ..
            } => {
                create(&mut graphs, &mut allocated, floor, graph, *incarnation)?;
            }
            WalRecord::DropLpgGraph {
                graph, incarnation, ..
            } => {
                drop_graph(&mut graphs, graph, *incarnation)?;
            }
            WalRecord::CatalogBatchV3 {
                created_graphs,
                dropped_graphs,
                created_graph_incarnations,
                dropped_graph_incarnations,
                ..
            } => {
                for (path, id) in dropped_graphs.iter().zip(dropped_graph_incarnations) {
                    drop_graph(&mut graphs, path, *id)?;
                }
                for (path, id) in created_graphs.iter().zip(created_graph_incarnations) {
                    create(&mut graphs, &mut allocated, floor, path, *id)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn invalid(reason: &str) -> Error {
    Error::Serialization(format!("invalid WAL graph incarnation: {reason}"))
}

fn create(
    graphs: &mut HashMap<GraphPath, GraphIncarnationId>,
    allocated: &mut HashSet<GraphIncarnationId>,
    floor: u64,
    path: &GraphPath,
    id: GraphIncarnationId,
) -> Result<()> {
    let parent = path
        .parent()
        .map_err(|error| invalid(&error.to_string()))?
        .ok_or_else(|| invalid("cannot create root"))?;
    if graphs.contains_key(path) || !graphs.contains_key(&parent) {
        return Err(invalid(
            "create requires an absent path and existing parent",
        ));
    }
    if id.as_u64() < floor || id.as_u64() == u64::MAX || !allocated.insert(id) {
        return Err(invalid("create reuses a reserved or retired incarnation"));
    }
    graphs.insert(path.clone(), id);
    Ok(())
}

fn drop_graph(
    graphs: &mut HashMap<GraphPath, GraphIncarnationId>,
    path: &GraphPath,
    id: GraphIncarnationId,
) -> Result<()> {
    if path.components().is_empty() || graphs.get(path) != Some(&id) {
        return Err(invalid("drop does not name the current lifetime"));
    }
    graphs.retain(|graph, _| !graph.components().starts_with(path.components()));
    Ok(())
}
