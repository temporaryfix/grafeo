//! Graph traversal algorithms: BFS and DFS.
//!
//! These algorithms use the visitor pattern to allow flexible customization
//! of traversal behavior, including early termination and edge filtering.

use std::collections::VecDeque;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType};
use super::traits::{
    Control, NodeValueResultBuilder, TraversalEvent, impl_algorithm, node_id_from_param,
};

// ============================================================================
// BFS Implementation
// ============================================================================

/// Performs breadth-first search from a starting node.
///
/// Returns the set of visited nodes in BFS order.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
///
/// # Returns
///
/// A vector of node IDs in the order they were discovered.
pub fn bfs(store: &dyn GraphStore, start: NodeId) -> Vec<NodeId> {
    let mut visited = Vec::new();
    bfs_with_visitor(store, start, |event| -> Control<()> {
        if let TraversalEvent::Discover(node) = event {
            visited.push(node);
        }
        Control::Continue
    });
    visited
}

/// Performs breadth-first search with a visitor callback.
///
/// The visitor is called for each traversal event, allowing custom
/// behavior such as early termination or path recording.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
/// * `visitor` - Callback function receiving traversal events
///
/// # Returns
///
/// `Some(B)` if the visitor returned `Control::Break(B)`, otherwise `None`.
pub fn bfs_with_visitor<B, F>(store: &dyn GraphStore, start: NodeId, mut visitor: F) -> Option<B>
where
    F: FnMut(TraversalEvent) -> Control<B>,
{
    let mut discovered: FxHashSet<NodeId> = FxHashSet::default();
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    // Check if start node exists
    store.get_node(start)?;

    // Discover the start node
    discovered.insert(start);
    queue.push_back(start);

    match visitor(TraversalEvent::Discover(start)) {
        Control::Break(b) => return Some(b),
        Control::Prune => {
            // Prune means don't explore neighbors, but we still finish the node
            match visitor(TraversalEvent::Finish(start)) {
                Control::Break(b) => return Some(b),
                _ => return None,
            }
        }
        Control::Continue => {}
    }

    while let Some(node) = queue.pop_front() {
        // Iterate over outgoing edges
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if discovered.insert(neighbor) {
                // Tree edge - neighbor not yet discovered
                match visitor(TraversalEvent::TreeEdge {
                    source: node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => continue, // Don't add to queue
                    Control::Continue => {}
                }

                match visitor(TraversalEvent::Discover(neighbor)) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => continue, // Don't explore neighbors
                    Control::Continue => {}
                }

                queue.push_back(neighbor);
            } else {
                // Non-tree edge - neighbor already discovered
                match visitor(TraversalEvent::NonTreeEdge {
                    source: node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    _ => {}
                }
            }
        }

        // Node processing complete
        match visitor(TraversalEvent::Finish(node)) {
            Control::Break(b) => return Some(b),
            _ => {}
        }
    }

    None
}

/// BFS layers - returns nodes grouped by their distance from the start.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
///
/// # Returns
///
/// A vector of vectors, where `result[i]` contains all nodes at distance `i` from start.
pub fn bfs_layers(store: &dyn GraphStore, start: NodeId) -> Vec<Vec<NodeId>> {
    bfs_layers_filtered(store, start, None, None)
}

/// BFS layers restricted to one edge type and/or bounded in depth.
///
/// Traverses outgoing edges, as [`bfs_layers`] does. A caller that needs an edge-type filter or a
/// depth bound gets it here instead of re-running a traversal of its own.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
/// * `edge_type` - When given, only edges of this type are followed, matched
///   case-insensitively as elsewhere in the engine
/// * `max_depth` - When given, the largest distance from `start` that is returned: `Some(0)` yields
///   the start node alone, `Some(2)` yields at most three layers
///
/// # Returns
///
/// A vector of vectors, where `result[i]` contains all nodes at distance `i` from start.
pub fn bfs_layers_filtered(
    store: &dyn GraphStore,
    start: NodeId,
    edge_type: Option<&str>,
    max_depth: Option<usize>,
) -> Vec<Vec<NodeId>> {
    bfs_layers_with_direction(store, start, edge_type, max_depth, Direction::Outgoing)
}

/// BFS layers restricted by edge type, depth, and traversal direction.
///
/// The start node is always returned at distance zero when it exists. Parallel
/// edges and cycles do not duplicate a node because discovery is tracked by
/// node identity.
pub fn bfs_layers_with_direction(
    store: &dyn GraphStore,
    start: NodeId,
    edge_type: Option<&str>,
    max_depth: Option<usize>,
    direction: Direction,
) -> Vec<Vec<NodeId>> {
    let mut layers: Vec<Vec<NodeId>> = Vec::new();
    let mut discovered: FxHashSet<NodeId> = FxHashSet::default();
    let mut current_layer: Vec<NodeId> = Vec::new();
    let mut next_layer: Vec<NodeId> = Vec::new();

    if store.get_node(start).is_none() {
        return layers;
    }

    discovered.insert(start);
    current_layer.push(start);

    while !current_layer.is_empty() {
        layers.push(current_layer.clone());

        // The layer just pushed sits at depth `layers.len() - 1`; stop once that is the bound.
        if max_depth.is_some_and(|bound| layers.len() > bound) {
            break;
        }

        for &node in &current_layer {
            for (neighbor, edge_id) in store.edges_from(node, direction) {
                if let Some(wanted) = edge_type
                    && !store
                        .edge_type(edge_id)
                        .is_some_and(|actual| actual.as_str().eq_ignore_ascii_case(wanted))
                {
                    continue;
                }
                if discovered.insert(neighbor) {
                    next_layer.push(neighbor);
                }
            }
        }

        current_layer.clear();
        std::mem::swap(&mut current_layer, &mut next_layer);
    }

    layers
}

// ============================================================================
// DFS Implementation
// ============================================================================

/// Node state during DFS traversal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeColor {
    /// Not yet discovered
    White,
    /// Discovered but not finished (on stack)
    Gray,
    /// Finished processing
    Black,
}

/// Performs depth-first search from a starting node.
///
/// Returns nodes in the order they were finished (post-order).
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
///
/// # Returns
///
/// A vector of node IDs in post-order (finished order).
pub fn dfs(store: &dyn GraphStore, start: NodeId) -> Vec<NodeId> {
    let mut finished = Vec::new();
    dfs_with_visitor(store, start, |event| -> Control<()> {
        if let TraversalEvent::Finish(node) = event {
            finished.push(node);
        }
        Control::Continue
    });
    finished
}

/// Performs depth-first search with a visitor callback.
///
/// Uses an explicit stack to avoid stack overflow on deep graphs.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
/// * `visitor` - Callback function receiving traversal events
///
/// # Returns
///
/// `Some(B)` if the visitor returned `Control::Break(B)`, otherwise `None`.
pub fn dfs_with_visitor<B, F>(store: &dyn GraphStore, start: NodeId, mut visitor: F) -> Option<B>
where
    F: FnMut(TraversalEvent) -> Control<B>,
{
    let mut color: FxHashMap<NodeId, NodeColor> = FxHashMap::default();

    // Stack entries: (node, edge_iterator_index, is_first_visit)
    // We use indices to track progress through neighbors
    let mut stack: Vec<(NodeId, Vec<(NodeId, grafeo_common::types::EdgeId)>, usize)> = Vec::new();

    // Check if start node exists
    store.get_node(start)?;

    // Discover start node
    color.insert(start, NodeColor::Gray);
    match visitor(TraversalEvent::Discover(start)) {
        Control::Break(b) => return Some(b),
        Control::Prune => {
            color.insert(start, NodeColor::Black);
            match visitor(TraversalEvent::Finish(start)) {
                Control::Break(b) => return Some(b),
                _ => return None,
            }
        }
        Control::Continue => {}
    }

    let neighbors: Vec<_> = store
        .edges_from(start, Direction::Outgoing)
        .into_iter()
        .collect();
    stack.push((start, neighbors, 0));

    while let Some((node, neighbors, idx)) = stack.last_mut() {
        if *idx >= neighbors.len() {
            // All neighbors processed, finish this node
            let node = *node;
            stack.pop();
            color.insert(node, NodeColor::Black);
            match visitor(TraversalEvent::Finish(node)) {
                Control::Break(b) => return Some(b),
                _ => {}
            }
            continue;
        }

        let (neighbor, edge_id) = neighbors[*idx];
        *idx += 1;

        match color.get(&neighbor).copied().unwrap_or(NodeColor::White) {
            NodeColor::White => {
                // Tree edge - undiscovered node
                match visitor(TraversalEvent::TreeEdge {
                    source: *node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => continue,
                    Control::Continue => {}
                }

                color.insert(neighbor, NodeColor::Gray);
                match visitor(TraversalEvent::Discover(neighbor)) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => {
                        color.insert(neighbor, NodeColor::Black);
                        match visitor(TraversalEvent::Finish(neighbor)) {
                            Control::Break(b) => return Some(b),
                            _ => {}
                        }
                        continue;
                    }
                    Control::Continue => {}
                }

                let neighbor_neighbors: Vec<_> = store
                    .edges_from(neighbor, Direction::Outgoing)
                    .into_iter()
                    .collect();
                stack.push((neighbor, neighbor_neighbors, 0));
            }
            NodeColor::Gray => {
                // Back edge - node is on the stack (ancestor)
                match visitor(TraversalEvent::BackEdge {
                    source: *node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    _ => {}
                }
            }
            NodeColor::Black => {
                // Non-tree edge (cross/forward) - already finished
                match visitor(TraversalEvent::NonTreeEdge {
                    source: *node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    _ => {}
                }
            }
        }
    }

    None
}

/// Performs DFS on all nodes, visiting each connected component.
///
/// Returns nodes in reverse post-order (useful for topological sort).
pub fn dfs_all(store: &dyn GraphStore) -> Vec<NodeId> {
    let mut finished = Vec::new();
    let mut visited: FxHashSet<NodeId> = FxHashSet::default();

    for node_id in store.node_ids() {
        if visited.contains(&node_id) {
            continue;
        }

        dfs_with_visitor(store, node_id, |event| -> Control<()> {
            match event {
                TraversalEvent::Discover(n) => {
                    visited.insert(n);
                }
                TraversalEvent::Finish(n) => {
                    finished.push(n);
                }
                _ => {}
            }
            Control::Continue
        });
    }

    finished
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for BFS algorithm.
static BFS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn bfs_params() -> &'static [ParameterDef] {
    BFS_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "start".to_string(),
                description: "Starting node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            // Optional parameters are appended, never inserted: positional arguments map by index.
            ParameterDef {
                name: "edge_type".to_string(),
                description: "Follow only edges of this type (default: every type)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "max_depth".to_string(),
                description: "Largest distance from `start` to return; 0 yields the start node \
                              alone (default: unbounded)"
                    .to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "direction".to_string(),
                description: "Traversal direction: outgoing, incoming, or both (default: outgoing)"
                    .to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("outgoing".to_string()),
            },
        ]
    })
}

/// BFS algorithm wrapper for the plugin registry.
pub struct BfsAlgorithm;

impl_algorithm! {
    BfsAlgorithm,
    name: "bfs",
    description: "Breadth-first search traversal from a starting node",
    params: bfs_params,
    execute(store, params) {
        let start_id = params.get_int("start").ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue("start parameter required".to_string())
        })?;

        let start = node_id_from_param(start_id, "start")?;
        let edge_type = match params.get_string("edge_type") {
            Some(value) => Some(value),
            None
                if params.get_int("edge_type").is_some()
                    || params.get_float("edge_type").is_some()
                    || params.get_bool("edge_type").is_some()
                    || params.get_list("edge_type").is_some() =>
            {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    "edge_type must be a string".to_string(),
                ));
            }
            None => None,
        };
        let max_depth = match params.get_int("max_depth") {
            Some(v) if v < 0 => {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    format!("max_depth must be non-negative, got {v}"),
                ));
            }
            Some(v) => Some(usize::try_from(v).map_err(|_| {
                grafeo_common::utils::error::Error::InvalidValue(
                    format!("max_depth value {v} exceeds maximum supported size"),
                )
            })?),
            None
                if params.get_float("max_depth").is_some()
                    || params.get_string("max_depth").is_some()
                    || params.get_bool("max_depth").is_some()
                    || params.get_list("max_depth").is_some() =>
            {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    "max_depth must be an integer".to_string(),
                ));
            }
            None => None,
        };
        let direction_value = match params.get_string("direction") {
            Some(value) => value,
            None
                if params.get_int("direction").is_some()
                    || params.get_float("direction").is_some()
                    || params.get_bool("direction").is_some()
                    || params.get_list("direction").is_some() =>
            {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    "direction must be a string".to_string(),
                ));
            }
            None => "outgoing",
        };
        let direction = match direction_value.to_ascii_lowercase().as_str() {
            "outgoing" => Direction::Outgoing,
            "incoming" => Direction::Incoming,
            "both" => Direction::Both,
            value => {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    format!(
                        "direction must be one of outgoing, incoming, or both, got '{value}'"
                    ),
                ));
            }
        };
        let layers = bfs_layers_with_direction(store, start, edge_type, max_depth, direction);

        let mut result = AlgorithmResult::new(vec!["node_id".to_string(), "distance".to_string()]);

        for (distance, layer) in layers.iter().enumerate() {
            for &node in layer {
                // reason: Node IDs and BFS distances are small, well within i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                result.add_row(vec![
                    Value::Int64(node.0 as i64),
                    Value::Int64(distance as i64),
                ]);
            }
        }

        Ok(result)
    }
}

/// Static parameter definitions for DFS algorithm.
static DFS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn dfs_params() -> &'static [ParameterDef] {
    DFS_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "start".to_string(),
            description: "Starting node ID".to_string(),
            param_type: ParameterType::NodeId,
            required: true,
            default: None,
        }]
    })
}

/// DFS algorithm wrapper for the plugin registry.
pub struct DfsAlgorithm;

impl_algorithm! {
    DfsAlgorithm,
    name: "dfs",
    description: "Depth-first search traversal from a starting node",
    params: dfs_params,
    execute(store, params) {
        let start_id = params.get_int("start").ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue("start parameter required".to_string())
        })?;

        let start = node_id_from_param(start_id, "start")?;
        let finished = dfs(store, start);

        let mut builder = NodeValueResultBuilder::with_capacity("finish_order", finished.len());
        for (order, node) in finished.iter().enumerate() {
            // reason: DFS finish order is bounded by node count, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            builder.push(*node, Value::Int64(order as i64));
        }

        Ok(builder.build())
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;

    fn create_test_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        // Create a simple graph:
        //   0 -> 1 -> 2
        //   |    |
        //   v    v
        //   3 -> 4
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);
        let n4 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n0, n3, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n1, n4, "EDGE");
        store.create_edge(n3, n4, "EDGE");

        store
    }

    #[test]
    fn test_bfs_simple() {
        let store = create_test_graph();
        let visited = bfs(&store, NodeId::new(0));

        assert!(!visited.is_empty());
        assert_eq!(visited[0], NodeId::new(0));
        // Node 0 should be first
    }

    #[test]
    fn test_bfs_layers() {
        let store = create_test_graph();
        let layers = bfs_layers(&store, NodeId::new(0));

        assert!(!layers.is_empty());
        assert_eq!(layers[0], vec![NodeId::new(0)]);
        // Distance 0: just the start node
    }

    /// A chain of two edge types: `a -KNOWS-> b -KNOWS-> c` and `a -LIKES-> d -LIKES-> e`.
    ///
    /// Unfiltered BFS reaches both branches; filtering on one type must reach only it, and a depth
    /// bound must cut the layers it returns.
    fn create_two_edge_type_graph() -> (LpgStore, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..5).map(|_| store.create_node(&["Node"])).collect();
        store.create_edge(nodes[0], nodes[1], "KNOWS");
        store.create_edge(nodes[1], nodes[2], "KNOWS");
        store.create_edge(nodes[0], nodes[3], "LIKES");
        store.create_edge(nodes[3], nodes[4], "LIKES");
        (store, nodes)
    }

    #[test]
    fn bfs_layers_filtered_restricts_to_one_edge_type() {
        let (store, nodes) = create_two_edge_type_graph();

        let all = bfs_layers(&store, nodes[0]);
        assert_eq!(all.len(), 3, "both branches are two hops deep: {all:?}");
        assert_eq!(
            all[1].len(),
            2,
            "unfiltered, layer 1 holds b and d: {all:?}"
        );

        let knows = bfs_layers_filtered(&store, nodes[0], Some("KNOWS"), None);
        assert_eq!(
            knows,
            vec![vec![nodes[0]], vec![nodes[1]], vec![nodes[2]]],
            "KNOWS must not reach the LIKES branch"
        );

        let likes = bfs_layers_filtered(&store, nodes[0], Some("LIKES"), None);
        assert_eq!(
            likes,
            vec![vec![nodes[0]], vec![nodes[3]], vec![nodes[4]]],
            "LIKES must not reach the KNOWS branch"
        );

        // Edge types match case-insensitively, as they do everywhere else in the engine.
        assert_eq!(
            bfs_layers_filtered(&store, nodes[0], Some("knows"), None),
            knows
        );

        assert_eq!(
            bfs_layers_filtered(&store, nodes[0], Some("NOPE"), None),
            vec![vec![nodes[0]]],
            "an unused edge type leaves the start node alone"
        );
    }

    #[test]
    fn bfs_layers_filtered_truncates_at_max_depth() {
        let (store, nodes) = create_two_edge_type_graph();

        assert_eq!(
            bfs_layers_filtered(&store, nodes[0], None, Some(0)),
            vec![vec![nodes[0]]],
            "depth 0 is the start node alone"
        );
        let depth_one = bfs_layers_filtered(&store, nodes[0], None, Some(1));
        assert_eq!(depth_one.len(), 2, "depth 1 returns two layers");
        assert_eq!(depth_one[1].len(), 2);

        assert_eq!(
            bfs_layers_filtered(&store, nodes[0], Some("KNOWS"), Some(1)),
            vec![vec![nodes[0]], vec![nodes[1]]],
            "the filter and the bound compose"
        );
        assert_eq!(
            bfs_layers_filtered(&store, nodes[0], None, Some(9)),
            bfs_layers(&store, nodes[0]),
            "a bound past the graph's depth changes nothing"
        );
    }

    #[test]
    fn bfs_algorithm_accepts_the_edge_type_and_depth_parameters() {
        use super::super::traits::GraphAlgorithm;

        let (store, nodes) = create_two_edge_type_graph();
        let parameters = BfsAlgorithm.parameters();
        assert_eq!(parameters.len(), 4);
        assert_eq!(parameters[0].name, "start");
        assert_eq!(parameters[1].name, "edge_type");
        assert_eq!(parameters[2].name, "max_depth");
        assert_eq!(parameters[3].name, "direction");
        assert_eq!(parameters[3].default.as_deref(), Some("outgoing"));
        assert!(!parameters[1].required && !parameters[2].required && !parameters[3].required);

        let mut params = super::super::super::Parameters::new();
        // reason: node IDs are sequential counters, well within i64::MAX
        #[allow(clippy::cast_possible_wrap)]
        params.set_int("start", nodes[0].0 as i64);
        assert_eq!(
            BfsAlgorithm.execute(&store, &params).unwrap().row_count(),
            5,
            "unfiltered BFS reaches every node"
        );

        params.set_string("edge_type", "KNOWS");
        assert_eq!(
            BfsAlgorithm.execute(&store, &params).unwrap().row_count(),
            3,
            "the KNOWS branch is three nodes"
        );

        params.set_int("max_depth", 1);
        assert_eq!(
            BfsAlgorithm.execute(&store, &params).unwrap().row_count(),
            2,
            "depth 1 on the KNOWS branch is two nodes"
        );

        params.set_int("max_depth", -1);
        assert!(
            BfsAlgorithm.execute(&store, &params).is_err(),
            "a negative depth is rejected, not silently treated as unbounded"
        );
    }

    #[test]
    fn bfs_layers_direction_handles_cycles_parallel_edges_and_disconnected_start() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let isolated = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        store.create_edge(d, b, "LIKES");

        assert_eq!(
            bfs_layers_with_direction(&store, c, Some("KNOWS"), Some(2), Direction::Incoming),
            vec![vec![c], vec![b], vec![a]]
        );
        assert_eq!(
            bfs_layers_with_direction(&store, b, None, Some(1), Direction::Both),
            vec![vec![b], vec![c, a, d]]
        );
        assert_eq!(
            bfs_layers_with_direction(&store, isolated, None, None, Direction::Both),
            vec![vec![isolated]]
        );
    }

    #[test]
    fn bfs_algorithm_direction_parameter_is_positional_and_validated() {
        use super::super::traits::GraphAlgorithm;

        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "R");

        let mut params = super::super::super::Parameters::new();
        #[allow(clippy::cast_possible_wrap)]
        params.set_int("start", b.0 as i64);
        params.set_string("direction", "incoming");
        assert_eq!(
            BfsAlgorithm.execute(&store, &params).unwrap().row_count(),
            2
        );

        params.set_string("direction", "sideways");
        assert!(BfsAlgorithm.execute(&store, &params).is_err());

        let mut wrong_type = super::super::super::Parameters::new();
        #[allow(clippy::cast_possible_wrap)]
        wrong_type.set_int("start", a.0 as i64);
        wrong_type.set_float("max_depth", 1.5);
        assert!(
            BfsAlgorithm.execute(&store, &wrong_type).is_err(),
            "a fractional max_depth must not silently become unbounded"
        );

        let mut wrong_type = super::super::super::Parameters::new();
        #[allow(clippy::cast_possible_wrap)]
        wrong_type.set_int("start", a.0 as i64);
        wrong_type.set_list("edge_type", vec![Value::Int64(1)]);
        assert!(
            BfsAlgorithm.execute(&store, &wrong_type).is_err(),
            "a non-string edge_type must not silently remove the filter"
        );

        let mut wrong_type = super::super::super::Parameters::new();
        #[allow(clippy::cast_possible_wrap)]
        wrong_type.set_int("start", a.0 as i64);
        wrong_type.set_bool("direction", true);
        assert!(
            BfsAlgorithm.execute(&store, &wrong_type).is_err(),
            "a non-string direction must not silently use outgoing"
        );
    }

    #[test]
    fn test_dfs_simple() {
        let store = create_test_graph();
        let finished = dfs(&store, NodeId::new(0));

        assert!(!finished.is_empty());
        // Post-order means leaves are finished first
    }

    #[test]
    fn test_bfs_nonexistent_start() {
        let store = LpgStore::new().unwrap();
        let visited = bfs(&store, NodeId::new(999));
        assert!(visited.is_empty());
    }

    #[test]
    fn test_dfs_nonexistent_start() {
        let store = LpgStore::new().unwrap();
        let finished = dfs(&store, NodeId::new(999));
        assert!(finished.is_empty());
    }

    #[test]
    fn test_bfs_early_termination() {
        let store = create_test_graph();
        let target = NodeId::new(2);

        let found = bfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == target
            {
                return Control::Break(true);
            }
            Control::Continue
        });

        assert_eq!(found, Some(true));
    }

    #[test]
    fn test_bfs_visits_all_reachable() {
        let store = create_test_graph();
        let visited = bfs(&store, NodeId::new(0));
        // All 5 nodes are reachable from node 0
        assert_eq!(visited.len(), 5);
    }

    #[test]
    fn test_bfs_layers_distances() {
        let store = create_test_graph();
        let layers = bfs_layers(&store, NodeId::new(0));

        // Layer 0: node 0
        // Layer 1: nodes 1, 3 (direct neighbors)
        // Layer 2: nodes 2, 4 (distance 2)
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].len(), 1);
        assert_eq!(layers[1].len(), 2);
        assert_eq!(layers[2].len(), 2);
    }

    #[test]
    fn test_bfs_layers_empty_graph() {
        let store = LpgStore::new().unwrap();
        let layers = bfs_layers(&store, NodeId::new(0));
        assert!(layers.is_empty());
    }

    #[test]
    fn test_bfs_single_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let visited = bfs(&store, n0);
        assert_eq!(visited, vec![n0]);
    }

    #[test]
    fn test_bfs_layers_single_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let layers = bfs_layers(&store, n0);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0], vec![n0]);
    }

    #[test]
    fn test_bfs_with_visitor_prune_on_start() {
        let store = create_test_graph();
        let result: Option<()> = bfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == NodeId::new(0)
            {
                return Control::Prune;
            }
            Control::Continue
        });
        // Pruning start node means no further traversal
        assert!(result.is_none());
    }

    #[test]
    fn test_bfs_with_visitor_collects_tree_edges() {
        let store = create_test_graph();
        let mut tree_edges = Vec::new();

        bfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            if let TraversalEvent::TreeEdge { source, target, .. } = event {
                tree_edges.push((source, target));
            }
            Control::Continue
        });

        // BFS tree from node 0 has 4 tree edges (one per non-start node)
        assert_eq!(tree_edges.len(), 4);
    }

    #[test]
    fn test_bfs_with_visitor_detects_non_tree_edges() {
        let store = create_test_graph();
        let mut non_tree_edges = Vec::new();

        bfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            if let TraversalEvent::NonTreeEdge { source, target, .. } = event {
                non_tree_edges.push((source, target));
            }
            Control::Continue
        });

        // There's at least one non-tree edge (3->4 or 1->4)
        assert!(!non_tree_edges.is_empty());
    }

    #[test]
    fn test_dfs_visits_all_reachable() {
        let store = create_test_graph();
        let finished = dfs(&store, NodeId::new(0));
        assert_eq!(finished.len(), 5);
    }

    #[test]
    fn test_dfs_post_order() {
        let store = create_test_graph();
        let finished = dfs(&store, NodeId::new(0));
        // In post-order, the start node is finished last
        assert_eq!(*finished.last().unwrap(), NodeId::new(0));
    }

    #[test]
    fn test_dfs_with_visitor_early_termination() {
        let store = create_test_graph();
        let found = dfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == NodeId::new(4)
            {
                return Control::Break(node);
            }
            Control::Continue
        });
        assert_eq!(found, Some(NodeId::new(4)));
    }

    #[test]
    fn test_dfs_with_visitor_prune() {
        let store = create_test_graph();
        let mut discovered = Vec::new();

        dfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            if let TraversalEvent::Discover(node) = event {
                discovered.push(node);
                if node == NodeId::new(1) {
                    return Control::Prune; // Don't explore node 1's children
                }
            }
            Control::Continue
        });

        // Node 1 is discovered but its children (2, 4) may not all be
        assert!(discovered.contains(&NodeId::new(0)));
        assert!(discovered.contains(&NodeId::new(1)));
        // Node 2 should not be discovered since we pruned node 1
        assert!(!discovered.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_dfs_with_visitor_back_edge() {
        // Create a cycle: 0 -> 1 -> 2 -> 0
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");

        let mut back_edges = Vec::new();
        dfs_with_visitor(&store, n0, |event| -> Control<()> {
            if let TraversalEvent::BackEdge { source, target, .. } = event {
                back_edges.push((source, target));
            }
            Control::Continue
        });

        // Edge 2->0 is a back edge (0 is ancestor of 2)
        assert_eq!(back_edges.len(), 1);
        assert_eq!(back_edges[0], (n2, n0));
    }

    #[test]
    fn test_dfs_single_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let finished = dfs(&store, n0);
        assert_eq!(finished, vec![n0]);
    }

    #[test]
    fn test_dfs_all_visits_all_components() {
        let store = LpgStore::new().unwrap();
        // Component 1: 0 -> 1
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");

        // Component 2: 2 -> 3
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);
        store.create_edge(n2, n3, "EDGE");

        let finished = dfs_all(&store);
        assert_eq!(finished.len(), 4);
    }

    #[test]
    fn test_dfs_all_empty_graph() {
        let store = LpgStore::new().unwrap();
        let finished = dfs_all(&store);
        assert!(finished.is_empty());
    }

    #[test]
    fn test_bfs_prune_tree_edge() {
        let store = create_test_graph();
        let mut discovered = Vec::new();

        bfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            match event {
                TraversalEvent::TreeEdge { target, .. } => {
                    if target == NodeId::new(1) {
                        return Control::Prune; // Skip node 1
                    }
                    Control::Continue
                }
                TraversalEvent::Discover(node) => {
                    discovered.push(node);
                    Control::Continue
                }
                _ => Control::Continue,
            }
        });

        // Node 1 should not be discovered due to pruned tree edge
        assert!(discovered.contains(&NodeId::new(0)));
        assert!(!discovered.contains(&NodeId::new(1)));
    }

    #[test]
    fn test_dfs_with_visitor_prune_start() {
        let store = create_test_graph();
        let result: Option<()> = dfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == NodeId::new(0)
            {
                return Control::Prune;
            }
            Control::Continue
        });
        assert!(result.is_none());
    }

    #[test]
    fn test_bfs_with_self_loop() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n0, "SELF"); // self-loop
        store.create_edge(n0, n1, "EDGE");

        let visited = bfs(&store, n0);
        // Self-loop should not cause infinite traversal or duplicates
        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0], n0);
        assert!(visited.contains(&n1));
    }

    #[test]
    fn test_dfs_with_self_loop() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n0, "SELF"); // self-loop
        store.create_edge(n0, n1, "EDGE");

        let finished = dfs(&store, n0);
        assert_eq!(finished.len(), 2);
        assert!(finished.contains(&n0));
        assert!(finished.contains(&n1));
    }
}
