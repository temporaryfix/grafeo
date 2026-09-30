//! Clustering coefficient algorithms: Local, Global, and Triangle counting.
//!
//! These algorithms measure how tightly connected the neighbors of each node are.
//! A high clustering coefficient indicates that neighbors tend to be connected to each other.

#[cfg(feature = "parallel")]
use std::sync::Arc;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;
#[cfg(feature = "parallel")]
use super::traits::ParallelGraphAlgorithm;

// ============================================================================
// Result Types
// ============================================================================

/// Result of clustering coefficient computation.
#[derive(Debug, Clone)]
pub struct ClusteringCoefficientResult {
    /// Local clustering coefficient for each node (0.0 to 1.0).
    pub coefficients: FxHashMap<NodeId, f64>,
    /// Number of triangles containing each node.
    pub triangle_counts: FxHashMap<NodeId, u64>,
    /// Total number of unique triangles in the graph.
    pub total_triangles: u64,
    /// Global (average) clustering coefficient.
    pub global_coefficient: f64,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Builds undirected neighbor sets for all nodes.
///
/// Treats the graph as undirected by combining both outgoing and incoming edges.
fn build_undirected_neighbors(store: &dyn GraphStore) -> FxHashMap<NodeId, FxHashSet<NodeId>> {
    let nodes = store.node_ids();
    let mut neighbors: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();

    // Initialize all nodes with empty sets
    for &node in &nodes {
        neighbors.insert(node, FxHashSet::default());
    }

    // Add edges in both directions (undirected treatment)
    for &node in &nodes {
        // Outgoing edges: node -> neighbor
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            // Add reverse direction for undirected
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }

        // Incoming edges: neighbor -> node (ensures we capture all connections)
        for (neighbor, _) in store.edges_from(node, Direction::Incoming) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }
    }

    neighbors
}

/// Counts triangles for a single node given its neighbors and the full neighbor map.
fn count_node_triangles(
    node_neighbors: &FxHashSet<NodeId>,
    all_neighbors: &FxHashMap<NodeId, FxHashSet<NodeId>>,
) -> u64 {
    let neighbor_list: Vec<NodeId> = node_neighbors.iter().copied().collect();
    let k = neighbor_list.len();
    let mut triangles = 0u64;

    // For each pair of neighbors, check if they're connected
    for i in 0..k {
        for j in (i + 1)..k {
            let u = neighbor_list[i];
            let w = neighbor_list[j];

            // Check if u and w are neighbors (completing a triangle)
            if let Some(u_neighbors) = all_neighbors.get(&u)
                && u_neighbors.contains(&w)
            {
                triangles += 1;
            }
        }
    }

    triangles
}

// ============================================================================
// Core Algorithm Functions
// ============================================================================

/// Counts the number of triangles containing each node.
///
/// A triangle is a set of three nodes where each pair is connected.
/// Each triangle is counted once for each of its three vertices.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Map from NodeId to the number of triangles containing that node.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn triangle_count(store: &dyn GraphStore) -> FxHashMap<NodeId, u64> {
    let neighbors = build_undirected_neighbors(store);
    let mut counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let triangles = count_node_triangles(node_neighbors, &neighbors);
        counts.insert(node, triangles);
    }

    counts
}

/// Computes the local clustering coefficient for each node.
///
/// The local clustering coefficient measures how close a node's neighbors are
/// to being a complete graph (clique). For a node v with degree k and T triangles:
///
/// C(v) = 2T / (k * (k-1)) for undirected graphs
///
/// Nodes with degree < 2 have coefficient 0.0 (cannot form triangles).
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Map from NodeId to local clustering coefficient (0.0 to 1.0).
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn local_clustering_coefficient(store: &dyn GraphStore) -> FxHashMap<NodeId, f64> {
    let neighbors = build_undirected_neighbors(store);
    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let k = node_neighbors.len();

        if k < 2 {
            // Cannot form triangles with fewer than 2 neighbors
            coefficients.insert(node, 0.0);
        } else {
            let triangles = count_node_triangles(node_neighbors, &neighbors);
            let max_triangles = (k * (k - 1)) / 2;
            let coefficient = triangles as f64 / max_triangles as f64;
            coefficients.insert(node, coefficient);
        }
    }

    coefficients
}

/// Computes the global (average) clustering coefficient.
///
/// The global clustering coefficient is the average of all local coefficients
/// across all nodes in the graph.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Average clustering coefficient (0.0 to 1.0).
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn global_clustering_coefficient(store: &dyn GraphStore) -> f64 {
    let local = local_clustering_coefficient(store);

    if local.is_empty() {
        return 0.0;
    }

    let sum: f64 = local.values().sum();
    sum / local.len() as f64
}

/// Counts the total number of unique triangles in the graph.
///
/// Each triangle is counted exactly once using degree-ordered merge intersection.
/// Builds the oriented adjacency directly from `GraphStore` without intermediate
/// hash sets, making it significantly faster on CSR-backed stores.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Total number of unique triangles.
///
/// # Complexity
///
/// O(m * sqrt(m)) where m is the number of edges
pub fn total_triangles(store: &dyn GraphStore) -> u64 {
    let (oriented_adj, edge_list) = build_oriented_adjacency(store);
    let mut total = 0u64;
    for &(_u, v) in &edge_list {
        total += sorted_intersection_count(&oriented_adj[_u], &oriented_adj[v]);
    }
    total
}

/// Builds degree-oriented adjacency lists directly from a GraphStore.
///
/// Skips the FxHashSet intermediary — collects undirected neighbors as sorted
/// Vec<usize> per node, computes degrees, then orients edges from low-degree
/// to high-degree. Returns (oriented_adj, edge_list) for triangle counting.
fn build_oriented_adjacency(store: &dyn GraphStore) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let node_list = store.node_ids();
    let n = node_list.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Map NodeId -> contiguous index
    let node_to_idx: FxHashMap<NodeId, usize> =
        node_list.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    // Build undirected adjacency as sorted Vec<usize> per node.
    // Collect both outgoing and incoming, deduplicate via sort+dedup.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, &node_u) in node_list.iter().enumerate() {
        for (neighbor, _) in store.edges_from(node_u, Direction::Outgoing) {
            if let Some(&v) = node_to_idx.get(&neighbor) {
                adj[u].push(v);
                adj[v].push(u);
            }
        }
    }

    // Sort and deduplicate each list
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }

    // Degrees from the undirected adjacency
    let degrees: Vec<usize> = adj.iter().map(Vec::len).collect();

    // Orient: u -> v only if deg(u) < deg(v), or (== and u < v)
    let mut oriented_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edge_list: Vec<(usize, usize)> = Vec::new();

    for u in 0..n {
        for &v in &adj[u] {
            if degrees[u] < degrees[v] || (degrees[u] == degrees[v] && u < v) {
                oriented_adj[u].push(v);
                edge_list.push((u, v));
            }
        }
        oriented_adj[u].sort_unstable();
    }

    (oriented_adj, edge_list)
}

/// Computes all clustering metrics in a single pass.
///
/// More efficient than calling each function separately since it builds
/// the neighbor structure only once.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Complete clustering coefficient result including local coefficients,
/// triangle counts, total triangles, and global coefficient.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn clustering_coefficient(store: &dyn GraphStore) -> ClusteringCoefficientResult {
    let neighbors = build_undirected_neighbors(store);
    let n = neighbors.len();

    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut triangle_counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let k = node_neighbors.len();
        let triangles = count_node_triangles(node_neighbors, &neighbors);

        triangle_counts.insert(node, triangles);

        let coefficient = if k < 2 {
            0.0
        } else {
            let max_triangles = (k * (k - 1)) / 2;
            triangles as f64 / max_triangles as f64
        };
        coefficients.insert(node, coefficient);
    }

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

// ============================================================================
// Directed Clustering Coefficient
// ============================================================================

/// Directed neighbourhood of every node, as the directed clustering coefficient needs it.
struct DirectedNeighborhood {
    /// Out-neighbours of each node: `j` is present when the edge `i -> j` is.
    out: FxHashMap<NodeId, FxHashSet<NodeId>>,
    /// Undirected neighbours: every node joined to `i` by an edge in either direction.
    all: FxHashMap<NodeId, FxHashSet<NodeId>>,
}

impl DirectedNeighborhood {
    /// True when the directed edge `x -> y` is present.
    fn has_edge(&self, x: NodeId, y: NodeId) -> bool {
        self.out.get(&x).is_some_and(|set| set.contains(&y))
    }

    /// Local directed clustering coefficient using the LDBC Graphalytics definition: neighbours
    /// are the union of incoming and outgoing neighbours, and directed edges between distinct
    /// neighbours are counted once per direction.
    fn coefficient(&self, node: NodeId) -> (f64, u64) {
        let Some(neighbors) = self.all.get(&node) else {
            return (0.0, 0);
        };
        let neighbor_list: Vec<NodeId> = neighbors.iter().copied().collect();

        let degree = neighbor_list.len() as u64;
        let mut closed_edges = 0u64;
        let mut triangles = 0u64;
        for (j_index, &j) in neighbor_list.iter().enumerate() {
            for &k in neighbor_list.iter().skip(j_index + 1) {
                let j_to_k = self.has_edge(j, k);
                let k_to_j = self.has_edge(k, j);
                closed_edges += u64::from(j_to_k) + u64::from(k_to_j);
                if j_to_k || k_to_j {
                    triangles += 1;
                }
            }
        }

        let possible = degree * degree.saturating_sub(1);
        if possible == 0 {
            return (0.0, triangles);
        }

        (closed_edges as f64 / possible as f64, triangles)
    }
}

/// Builds the directed neighbourhood of every node: out-neighbours and undirected neighbours.
///
/// Self-loops are skipped: a node cannot form a triangle with itself.
fn build_directed_neighbors(store: &dyn GraphStore) -> DirectedNeighborhood {
    let nodes = store.node_ids();
    let mut out: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();
    let mut all: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();

    for &node in &nodes {
        out.insert(node, FxHashSet::default());
        all.insert(node, FxHashSet::default());
    }

    for &node in &nodes {
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if neighbor == node || !all.contains_key(&neighbor) {
                continue;
            }
            if let Some(set) = out.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = all.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = all.get_mut(&neighbor) {
                set.insert(node);
            }
        }
    }

    DirectedNeighborhood { out, all }
}

/// Computes the **directed** local clustering coefficient of every node.
///
/// Implements the directed local coefficient from the LDBC Graphalytics specification — see
/// `DirectedNeighborhood::coefficient` for the neighbourhood and edge-count definition.
/// Direction is honoured: in the transitive triangle `a -> b`, `b -> c`, `a -> c` every node scores
/// 1/2, because only one of the two ordered pairs of its neighbours is joined, where the
/// undirected coefficient ([`clustering_coefficient`]) scores 1.
///
/// `triangle_counts` hold unique triangles in the underlying undirected graph through each
/// node, and `total_triangles` counts each triangle once.
///
/// # Arguments
///
/// * `store` - The graph store, read as directed
///
/// # Returns
///
/// Complete clustering coefficient result for the directed reading of the graph.
///
/// # Complexity
///
/// O(V * d^2) where d is the average total degree
pub fn clustering_coefficient_directed(store: &dyn GraphStore) -> ClusteringCoefficientResult {
    let neighborhood = build_directed_neighbors(store);
    let n = neighborhood.all.len();

    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut triangle_counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for &node in neighborhood.all.keys() {
        let (coefficient, triangles) = neighborhood.coefficient(node);
        coefficients.insert(node, coefficient);
        triangle_counts.insert(node, triangles);
    }

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

/// Computes directed clustering coefficients in parallel using rayon.
///
/// Falls back to [`clustering_coefficient_directed`] below `parallel_threshold` nodes.
///
/// # Arguments
///
/// * `store` - The graph store, read as directed
/// * `parallel_threshold` - Minimum node count to enable parallelism
///
/// # Returns
///
/// Complete clustering coefficient result for the directed reading of the graph.
///
/// # Complexity
///
/// O(V * d^2 / threads) where d is the average total degree
#[cfg(feature = "parallel")]
pub fn clustering_coefficient_directed_parallel(
    store: &dyn GraphStore,
    parallel_threshold: usize,
) -> ClusteringCoefficientResult {
    let neighborhood = build_directed_neighbors(store);
    let n = neighborhood.all.len();

    if n < parallel_threshold {
        return clustering_coefficient_directed(store);
    }

    let neighborhood = Arc::new(neighborhood);
    let nodes: Vec<NodeId> = neighborhood.all.keys().copied().collect();

    let (coefficients, triangle_counts): (FxHashMap<NodeId, f64>, FxHashMap<NodeId, u64>) = nodes
        .par_iter()
        .fold(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut coeffs, mut triangles), &node| {
                let (coefficient, count) = neighborhood.coefficient(node);
                coeffs.insert(node, coefficient);
                triangles.insert(node, count);
                (coeffs, triangles)
            },
        )
        .reduce(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut c1, mut t1), (c2, t2)| {
                c1.extend(c2);
                t1.extend(t2);
                (c1, t1)
            },
        );

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}
// ============================================================================
// Parallel Implementation
// ============================================================================

/// Computes clustering coefficients in parallel using rayon.
///
/// Automatically falls back to sequential execution for small graphs
/// to avoid parallelization overhead.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
/// * `parallel_threshold` - Minimum node count to enable parallelism
///
/// # Returns
///
/// Complete clustering coefficient result.
///
/// # Panics
///
/// Panics if the internal neighbor map is missing an expected node entry.
///
/// # Complexity
///
/// O(V * d^2 / threads) where d is the average degree
#[cfg(feature = "parallel")]
pub fn clustering_coefficient_parallel(
    store: &dyn GraphStore,
    parallel_threshold: usize,
) -> ClusteringCoefficientResult {
    let neighbors = build_undirected_neighbors(store);
    let n = neighbors.len();

    if n < parallel_threshold {
        // Fall back to sequential for small graphs
        return clustering_coefficient(store);
    }

    // Use Arc for shared neighbor data across threads
    let neighbors = Arc::new(neighbors);
    let nodes: Vec<NodeId> = neighbors.keys().copied().collect();

    // Parallel computation using fold-reduce pattern
    let (coefficients, triangle_counts): (FxHashMap<NodeId, f64>, FxHashMap<NodeId, u64>) = nodes
        .par_iter()
        .fold(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut coeffs, mut triangles), &node| {
                let node_neighbors = neighbors.get(&node).expect("node in neighbor map");
                let k = node_neighbors.len();

                let t = count_node_triangles(node_neighbors, &neighbors);

                triangles.insert(node, t);

                let coefficient = if k < 2 {
                    0.0
                } else {
                    let max_triangles = (k * (k - 1)) / 2;
                    t as f64 / max_triangles as f64
                };
                coeffs.insert(node, coefficient);

                (coeffs, triangles)
            },
        )
        .reduce(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut c1, mut t1), (c2, t2)| {
                c1.extend(c2);
                t1.extend(t2);
                (c1, t1)
            },
        );

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

/// Counts the total number of unique triangles in parallel.
///
/// This is a dedicated parallel path optimized purely for triangle counting,
/// without the overhead of computing per-node clustering coefficients. Uses
/// three key optimizations:
///
/// 1. **Degree ordering**: For each edge (u, v), only process when
///    degree(u) <= degree(v). This ensures each triangle is counted exactly
///    once and halves the work.
/// 2. **Sorted neighbor intersection**: Pre-sort adjacency lists and use
///    merge-based intersection instead of hash lookups. This is cache-friendly.
/// 3. **Atomic accumulator**: Uses `AtomicU64` instead of per-thread maps,
///    avoiding lock contention during the reduce phase.
///
/// Falls back to sequential `total_triangles` for small graphs.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
/// * `parallel_threshold` - Minimum node count to enable parallelism
///
/// # Returns
///
/// Total number of unique triangles.
///
/// # Complexity
///
/// O(m * sqrt(m) / threads) where m is the number of edges
#[cfg(feature = "parallel")]
pub fn total_triangles_parallel(store: &dyn GraphStore, parallel_threshold: usize) -> u64 {
    let nodes = store.node_ids();
    if nodes.len() < parallel_threshold {
        return total_triangles(store);
    }

    let (oriented_adj, edge_list) = build_oriented_adjacency(store);
    if oriented_adj.is_empty() {
        return 0;
    }

    // Parallel triangle counting over the edge list.
    let oriented_adj = Arc::new(oriented_adj);
    let counter = std::sync::atomic::AtomicU64::new(0);

    edge_list.par_iter().for_each(|&(u, v)| {
        let count = sorted_intersection_count(&oriented_adj[u], &oriented_adj[v]);
        counter.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    });

    counter.load(std::sync::atomic::Ordering::Relaxed)
}

/// Counts the size of the intersection of two sorted slices.
///
/// Uses a merge-based approach: O(min(|a|, |b|)) with excellent cache behavior.
fn sorted_intersection_count(a: &[usize], b: &[usize]) -> u64 {
    let mut count = 0u64;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

// ============================================================================
// Algorithm Wrapper for Plugin Registry
// ============================================================================

/// Static parameter definitions for Total Triangles algorithm.
static TOTAL_TRIANGLES_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn total_triangles_params() -> &'static [ParameterDef] {
    TOTAL_TRIANGLES_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "parallel".to_string(),
                description: "Enable parallel computation (default: true)".to_string(),
                param_type: ParameterType::Boolean,
                required: false,
                default: Some("true".to_string()),
            },
            ParameterDef {
                name: "parallel_threshold".to_string(),
                description: "Minimum nodes for parallel execution (default: 50)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("50".to_string()),
            },
        ]
    })
}

/// Total Triangles algorithm wrapper for the plugin registry.
///
/// Returns a single-row result with the total triangle count.
/// Uses the optimized parallel path when available and enabled.
pub struct TotalTrianglesAlgorithm;

impl GraphAlgorithm for TotalTrianglesAlgorithm {
    fn name(&self) -> &str {
        "total_triangles"
    }

    fn description(&self) -> &str {
        "Count the total number of unique triangles in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        total_triangles_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        #[cfg(feature = "parallel")]
        let count = {
            let parallel = params.get_bool("parallel").unwrap_or(true);
            let threshold =
                usize::try_from(params.get_int("parallel_threshold").unwrap_or(50)).unwrap_or(50);

            if parallel {
                total_triangles_parallel(store, threshold)
            } else {
                total_triangles(store)
            }
        };

        #[cfg(not(feature = "parallel"))]
        let count = {
            let _ = params;
            total_triangles(store)
        };

        let mut output = AlgorithmResult::new(vec!["total_triangles".to_string()]);
        // reason: Triangle count is bounded by graph size, well within i64::MAX
        #[allow(clippy::cast_possible_wrap)]
        output.add_row(vec![Value::Int64(count as i64)]);
        Ok(output)
    }
}

/// Static parameter definitions for Clustering Coefficient algorithm.
static CLUSTERING_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn clustering_params() -> &'static [ParameterDef] {
    CLUSTERING_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "parallel".to_string(),
                description: "Enable parallel computation (default: true)".to_string(),
                param_type: ParameterType::Boolean,
                required: false,
                default: Some("true".to_string()),
            },
            ParameterDef {
                name: "parallel_threshold".to_string(),
                description: "Minimum nodes for parallel execution (default: 50)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("50".to_string()),
            },
            // Appended last on purpose: positional arguments map by index, so a new parameter
            // goes after the existing ones or it changes what `(true)` means.
            ParameterDef {
                name: "directed".to_string(),
                description: "Honour edge direction, computing Fagiolo's total-pattern directed \
                              coefficient instead of the undirected one (default: false)"
                    .to_string(),
                param_type: ParameterType::Boolean,
                required: false,
                default: Some("false".to_string()),
            },
        ]
    })
}

/// Clustering Coefficient algorithm wrapper for the plugin registry.
pub struct ClusteringCoefficientAlgorithm;

fn validate_directed_parameter(params: &Parameters) -> Result<()> {
    if params.get_bool("directed").is_none()
        && (params.get_int("directed").is_some()
            || params.get_float("directed").is_some()
            || params.get_string("directed").is_some()
            || params.get_list("directed").is_some())
    {
        return Err(Error::InvalidValue(
            "directed parameter must be a boolean".to_string(),
        ));
    }
    Ok(())
}

impl GraphAlgorithm for ClusteringCoefficientAlgorithm {
    fn name(&self) -> &str {
        "clustering_coefficient"
    }

    fn description(&self) -> &str {
        "Local and global clustering coefficients with triangle counts"
    }

    fn parameters(&self) -> &[ParameterDef] {
        clustering_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        validate_directed_parameter(params)?;
        let directed = params.get_bool("directed").unwrap_or(false);

        #[cfg(feature = "parallel")]
        let result = {
            let parallel = params.get_bool("parallel").unwrap_or(true);
            let threshold =
                usize::try_from(params.get_int("parallel_threshold").unwrap_or(50)).unwrap_or(50);

            match (directed, parallel) {
                (true, true) => clustering_coefficient_directed_parallel(store, threshold),
                (true, false) => clustering_coefficient_directed(store),
                (false, true) => clustering_coefficient_parallel(store, threshold),
                (false, false) => clustering_coefficient(store),
            }
        };

        #[cfg(not(feature = "parallel"))]
        let result = {
            if directed {
                clustering_coefficient_directed(store)
            } else {
                clustering_coefficient(store)
            }
        };

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "clustering_coefficient".to_string(),
            "triangle_count".to_string(),
        ]);

        for (node, coefficient) in result.coefficients {
            let triangles = *result.triangle_counts.get(&node).unwrap_or(&0);
            // reason: Node IDs and triangle counts are bounded by graph size, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(coefficient),
                Value::Int64(triangles as i64),
            ]);
        }

        Ok(output)
    }
}

#[cfg(feature = "parallel")]
impl ParallelGraphAlgorithm for ClusteringCoefficientAlgorithm {
    fn parallel_threshold(&self) -> usize {
        50
    }

    fn execute_parallel(
        &self,
        store: &dyn GraphStore,
        params: &Parameters,
        _num_threads: usize,
    ) -> Result<AlgorithmResult> {
        validate_directed_parameter(params)?;
        let result = if params.get_bool("directed").unwrap_or(false) {
            clustering_coefficient_directed_parallel(store, self.parallel_threshold())
        } else {
            clustering_coefficient_parallel(store, self.parallel_threshold())
        };

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "clustering_coefficient".to_string(),
            "triangle_count".to_string(),
        ]);

        for (node, coefficient) in result.coefficients {
            let triangles = *result.triangle_counts.get(&node).unwrap_or(&0);
            // reason: Node IDs and triangle counts are bounded by graph size, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(coefficient),
                Value::Int64(triangles as i64),
            ]);
        }

        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;

    fn create_triangle_graph() -> LpgStore {
        // Simple triangle: 0 - 1 - 2 - 0
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        // Bidirectional edges for undirected treatment
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        store
    }

    fn create_star_graph() -> LpgStore {
        // Star: center (0) connected to leaves (1, 2, 3, 4)
        // No triangles because leaves don't connect to each other
        let store = LpgStore::new().unwrap();
        let center = store.create_node(&["Center"]);

        for _ in 0..4 {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center, leaf, "EDGE");
            store.create_edge(leaf, center, "EDGE");
        }

        store
    }

    fn create_complete_graph(n: usize) -> LpgStore {
        // K_n: complete graph with n nodes (all pairs connected)
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        store
    }

    fn create_path_graph() -> LpgStore {
        // Path: 0 - 1 - 2 - 3
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        store
    }

    #[test]
    fn test_triangle_graph_clustering() {
        let store = create_triangle_graph();
        let result = clustering_coefficient(&store);

        // All nodes in a triangle have coefficient 1.0
        for (_, coeff) in &result.coefficients {
            assert!(
                (*coeff - 1.0).abs() < 1e-10,
                "Expected coefficient 1.0, got {}",
                coeff
            );
        }

        // One unique triangle
        assert_eq!(result.total_triangles, 1);

        // Global coefficient should be 1.0
        assert!(
            (result.global_coefficient - 1.0).abs() < 1e-10,
            "Expected global 1.0, got {}",
            result.global_coefficient
        );
    }

    #[test]
    fn test_star_graph_clustering() {
        let store = create_star_graph();
        let result = clustering_coefficient(&store);

        // All coefficients should be 0 (no triangles in a star)
        for (_, coeff) in &result.coefficients {
            assert_eq!(*coeff, 0.0);
        }

        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.global_coefficient, 0.0);
    }

    #[test]
    fn test_complete_graph_clustering() {
        let store = create_complete_graph(5);
        let result = clustering_coefficient(&store);

        // In a complete graph, all coefficients are 1.0
        for (_, coeff) in &result.coefficients {
            assert!((*coeff - 1.0).abs() < 1e-10, "Expected 1.0, got {}", coeff);
        }

        // K_5 has C(5,3) = 10 triangles
        assert_eq!(result.total_triangles, 10);
    }

    #[test]
    fn test_path_graph_clustering() {
        let store = create_path_graph();
        let result = clustering_coefficient(&store);

        // Path has no triangles
        assert_eq!(result.total_triangles, 0);

        // All coefficients should be 0 (endpoints have degree 1, middle have no triangles)
        for (_, coeff) in &result.coefficients {
            assert_eq!(*coeff, 0.0);
        }
    }

    #[test]
    fn test_empty_graph() {
        let store = LpgStore::new().unwrap();
        let result = clustering_coefficient(&store);

        assert!(result.coefficients.is_empty());
        assert!(result.triangle_counts.is_empty());
        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.global_coefficient, 0.0);
    }

    #[test]
    fn test_single_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);

        let result = clustering_coefficient(&store);

        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(*result.coefficients.get(&n0).unwrap(), 0.0);
        assert_eq!(result.total_triangles, 0);
    }

    /// The transitive triangle `a -> b`, `b -> c`, `a -> c` separates the directed coefficient
    /// from the undirected one: every node has two neighbours joined in exactly one of the two
    /// possible directions, so the directed coefficient is 1/2 where the undirected one is 1.
    #[test]
    fn directed_coefficient_of_the_transitive_triangle_is_one_half() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, c, "EDGE");
        store.create_edge(a, c, "EDGE");

        let undirected = clustering_coefficient(&store);
        let directed = clustering_coefficient_directed(&store);

        for node in [a, b, c] {
            let plain = *undirected.coefficients.get(&node).unwrap();
            assert!(
                (plain - 1.0).abs() < 1e-12,
                "undirected coefficient in a triangle is 1.0, got {plain}"
            );
            let with_direction = *directed.coefficients.get(&node).unwrap();
            assert!(
                (with_direction - 0.5).abs() < 1e-12,
                "directed coefficient in the transitive triangle is 0.5, got {with_direction}"
            );
            assert_eq!(*directed.triangle_counts.get(&node).unwrap(), 1);
        }
        assert_eq!(directed.total_triangles, 1);
    }

    #[test]
    fn directed_coefficient_uses_union_neighborhood() {
        let store = LpgStore::new().unwrap();
        let i = store.create_node(&["Node"]);
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(i, a, "EDGE");
        store.create_edge(i, b, "EDGE");
        store.create_edge(i, c, "EDGE");
        store.create_edge(a, i, "EDGE");
        store.create_edge(a, b, "EDGE");

        let directed = clustering_coefficient_directed(&store);
        let coefficient = *directed.coefficients.get(&i).unwrap();
        assert!((coefficient - (1.0 / 6.0)).abs() < 1e-12);
        assert_eq!(*directed.triangle_counts.get(&i).unwrap(), 1);
    }

    /// On a fully reciprocal graph the directed coefficient reduces to the undirected one.
    #[test]
    fn directed_coefficient_matches_undirected_on_a_reciprocal_triangle() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        for (from, to) in [(a, b), (b, a), (b, c), (c, b), (a, c), (c, a)] {
            store.create_edge(from, to, "EDGE");
        }

        let directed = clustering_coefficient_directed(&store);
        for node in [a, b, c] {
            let value = *directed.coefficients.get(&node).unwrap();
            assert!(
                (value - 1.0).abs() < 1e-12,
                "a reciprocal triangle scores 1.0 under the directed formula too, got {value}"
            );
            assert_eq!(*directed.triangle_counts.get(&node).unwrap(), 1);
        }
        assert_eq!(directed.total_triangles, 1);
    }

    /// A node with fewer than two neighbours has no pair to close, so the coefficient is 0 and
    /// the normalisation must not divide by zero.
    #[test]
    fn directed_coefficient_of_low_degree_nodes_is_zero() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let isolated = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");

        let directed = clustering_coefficient_directed(&store);
        for node in [a, b, isolated] {
            assert_eq!(*directed.coefficients.get(&node).unwrap(), 0.0);
            assert_eq!(*directed.triangle_counts.get(&node).unwrap(), 0);
        }
        assert_eq!(directed.total_triangles, 0);
    }

    #[test]
    fn directed_triangle_counts_are_unique_for_asymmetric_spoke() {
        let store = LpgStore::new().unwrap();
        let i = store.create_node(&["Node"]);
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        for (source, target) in [(i, a), (i, b), (i, c), (a, i), (a, b)] {
            store.create_edge(source, target, "EDGE");
        }

        let directed = clustering_coefficient_directed(&store);
        for node in [i, a, b] {
            assert_eq!(*directed.triangle_counts.get(&node).unwrap(), 1);
        }
        assert_eq!(*directed.triangle_counts.get(&c).unwrap(), 0);
        assert_eq!(directed.total_triangles, 1);
        assert!((directed.coefficients[&i] - 1.0 / 6.0).abs() < 1e-12);
        assert!((directed.coefficients[&a] - 1.0 / 2.0).abs() < 1e-12);
        assert!((directed.coefficients[&b] - 1.0).abs() < 1e-12);
        assert_eq!(directed.coefficients[&c], 0.0);
    }

    #[test]
    fn test_two_connected_nodes() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");

        let result = clustering_coefficient(&store);

        // Both nodes have degree 1, so coefficient is 0
        assert_eq!(*result.coefficients.get(&n0).unwrap(), 0.0);
        assert_eq!(*result.coefficients.get(&n1).unwrap(), 0.0);
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_triangle_count_function() {
        let store = create_triangle_graph();
        let counts = triangle_count(&store);

        // Each node in a triangle has 1 triangle
        for (_, count) in counts {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_local_clustering_coefficient_function() {
        let store = create_complete_graph(4);
        let coefficients = local_clustering_coefficient(&store);

        // K_4: all nodes have coefficient 1.0
        for (_, coeff) in coefficients {
            assert!((coeff - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_global_clustering_coefficient_function() {
        let store = create_triangle_graph();
        let global = global_clustering_coefficient(&store);
        assert!((global - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_triangles_function() {
        let store = create_complete_graph(4);
        let total = total_triangles(&store);
        // K_4 has C(4,3) = 4 triangles
        assert_eq!(total, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_sequential() {
        let store = create_complete_graph(20);

        let sequential = clustering_coefficient(&store);
        let parallel = clustering_coefficient_parallel(&store, 1); // Force parallel

        // Results should match
        for (node, seq_coeff) in &sequential.coefficients {
            let par_coeff = parallel.coefficients.get(node).unwrap();
            assert!(
                (seq_coeff - par_coeff).abs() < 1e-10,
                "Mismatch for node {:?}: seq={}, par={}",
                node,
                seq_coeff,
                par_coeff
            );
        }

        assert_eq!(sequential.total_triangles, parallel.total_triangles);
        assert!((sequential.global_coefficient - parallel.global_coefficient).abs() < 1e-10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_threshold_fallback() {
        let store = create_triangle_graph();

        // With threshold higher than node count, should use sequential
        let result = clustering_coefficient_parallel(&store, 100);

        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.total_triangles, 1);
    }

    #[test]
    fn test_algorithm_wrapper() {
        let store = create_triangle_graph();
        let algo = ClusteringCoefficientAlgorithm;

        assert_eq!(algo.name(), "clustering_coefficient");
        assert!(!algo.description().is_empty());
        // parallel, parallel_threshold, directed — `directed` appended last so positional
        // arguments keep their meaning.
        assert_eq!(algo.parameters().len(), 3);
        assert_eq!(algo.parameters()[2].name, "directed");

        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();

        assert_eq!(result.columns.len(), 3);
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_algorithm_wrapper_sequential() {
        let store = create_triangle_graph();
        let algo = ClusteringCoefficientAlgorithm;

        let mut params = Parameters::new();
        params.set_bool("parallel", false);

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_algorithm_wrapper_rejects_wrong_directed_type() {
        let store = create_triangle_graph();
        let mut params = Parameters::new();
        params.set_string("directed", "true");

        let error = ClusteringCoefficientAlgorithm
            .execute(&store, &params)
            .err()
            .expect("invalid algorithm arguments must fail");
        assert!(
            error
                .to_string()
                .contains("directed parameter must be a boolean")
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_algorithm_trait() {
        let store = create_complete_graph(10);
        let algo = ClusteringCoefficientAlgorithm;

        assert_eq!(algo.parallel_threshold(), 50);

        let params = Parameters::new();
        let result = algo.execute_parallel(&store, &params, 4).unwrap();

        assert_eq!(result.row_count(), 10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_total_triangles_parallel_matches_sequential() {
        let store = create_complete_graph(20);

        let sequential = total_triangles(&store);
        let parallel = total_triangles_parallel(&store, 1); // Force parallel

        assert_eq!(
            sequential, parallel,
            "Sequential ({}) and parallel ({}) triangle counts diverge",
            sequential, parallel
        );

        // K_20 has C(20,3) = 1140 triangles
        assert_eq!(sequential, 1140);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_total_triangles_parallel_triangle_graph() {
        let store = create_triangle_graph();
        let count = total_triangles_parallel(&store, 1);
        assert_eq!(count, 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_total_triangles_parallel_star_graph() {
        let store = create_star_graph();
        let count = total_triangles_parallel(&store, 1);
        assert_eq!(count, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_total_triangles_parallel_path_graph() {
        let store = create_path_graph();
        let count = total_triangles_parallel(&store, 1);
        assert_eq!(count, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_total_triangles_parallel_empty_graph() {
        let store = LpgStore::new().unwrap();
        let count = total_triangles_parallel(&store, 1);
        assert_eq!(count, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_total_triangles_parallel_threshold_fallback() {
        let store = create_triangle_graph();
        // Threshold higher than node count: should use sequential path
        let count = total_triangles_parallel(&store, 100);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_total_triangles_algorithm_wrapper() {
        let store = create_complete_graph(5);
        let algo = TotalTrianglesAlgorithm;

        assert_eq!(algo.name(), "total_triangles");
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 1);
        // K_5 has 10 triangles
        assert_eq!(result.rows[0][0], Value::Int64(10));
    }

    #[test]
    fn test_two_triangles_sharing_edge() {
        // Two triangles sharing edge 0-1:
        //     2
        //    / \
        //   0---1
        //    \ /
        //     3
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        // Triangle 0-1-2
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        // Triangle 0-1-3
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");
        store.create_edge(n3, n0, "EDGE");
        store.create_edge(n0, n3, "EDGE");

        let result = clustering_coefficient(&store);

        // 2 unique triangles
        assert_eq!(result.total_triangles, 2);

        // Nodes 0 and 1 are in both triangles
        assert_eq!(*result.triangle_counts.get(&n0).unwrap(), 2);
        assert_eq!(*result.triangle_counts.get(&n1).unwrap(), 2);

        // Nodes 2 and 3 are in one triangle each
        assert_eq!(*result.triangle_counts.get(&n2).unwrap(), 1);
        assert_eq!(*result.triangle_counts.get(&n3).unwrap(), 1);

        // Node 0 has 3 neighbors (1, 2, 3), 2 triangles
        // max_triangles = 3*2/2 = 3, coefficient = 2/3
        let coeff_0 = *result.coefficients.get(&n0).unwrap();
        assert!(
            (coeff_0 - 2.0 / 3.0).abs() < 1e-10,
            "Expected 2/3, got {}",
            coeff_0
        );
    }

    // ---- Cross-model: RDF adapter produces same results as LPG ----

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_triangle_count_rdf_matches_lpg() {
        use grafeo_core::graph::rdf::{RdfGraphStoreAdapter, RdfStore, Term, Triple};

        // Build a K_4 graph in RDF
        let rdf = RdfStore::new();
        let nodes = ["a", "b", "c", "d"];
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let u = Term::iri(format!("http://example.org/{}", nodes[i]));
                let v = Term::iri(format!("http://example.org/{}", nodes[j]));
                let pred = Term::iri("http://example.org/knows");
                rdf.insert(Triple::new(u.clone(), pred.clone(), v.clone()));
                rdf.insert(Triple::new(v, pred, u));
            }
        }

        let adapter = RdfGraphStoreAdapter::new(&rdf);

        // K_4 on RDF adapter
        let rdf_triangles = total_triangles(&adapter);

        // K_4 on LPG
        let lpg = create_complete_graph(4);
        let lpg_triangles = total_triangles(&lpg);

        assert_eq!(
            rdf_triangles, lpg_triangles,
            "RDF adapter ({}) and LPG ({}) triangle counts must match for K_4",
            rdf_triangles, lpg_triangles
        );
        // K_4 has C(4,3) = 4 triangles
        assert_eq!(rdf_triangles, 4);
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_clustering_coefficient_rdf_matches_lpg() {
        use grafeo_core::graph::rdf::{RdfGraphStoreAdapter, RdfStore, Term, Triple};

        // Build a triangle in RDF
        let rdf = RdfStore::new();
        let pred = Term::iri("http://example.org/knows");
        let pairs = [("a", "b"), ("b", "c"), ("c", "a")];
        for (s, o) in pairs {
            let subj = Term::iri(format!("http://example.org/{s}"));
            let obj = Term::iri(format!("http://example.org/{o}"));
            rdf.insert(Triple::new(subj.clone(), pred.clone(), obj.clone()));
            rdf.insert(Triple::new(obj, pred.clone(), subj));
        }

        let adapter = RdfGraphStoreAdapter::new(&rdf);
        let rdf_result = clustering_coefficient(&adapter);

        // All coefficients should be 1.0 (triangle)
        for (_, coeff) in &rdf_result.coefficients {
            assert!(
                (*coeff - 1.0).abs() < 1e-10,
                "RDF triangle coefficient should be 1.0, got {}",
                coeff
            );
        }
        assert_eq!(rdf_result.total_triangles, 1);
    }
}
