//! Traversal and iteration methods for the LPG store.

use super::LpgStore;
use crate::graph::Direction;
use crate::graph::lpg::{Edge, Node};
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};

impl LpgStore {
    // === Traversal ===

    /// Iterates over neighbors of a node in the specified direction.
    ///
    /// This is the fast path for graph traversal - goes straight to the
    /// adjacency index without loading full node data.
    pub fn neighbors(
        &self,
        node: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = NodeId> + '_ {
        let forward: Box<dyn Iterator<Item = NodeId>> = match direction {
            Direction::Outgoing | Direction::Both => {
                Box::new(self.forward_adj.neighbors(node).into_iter())
            }
            Direction::Incoming => Box::new(std::iter::empty()),
        };

        let backward: Box<dyn Iterator<Item = NodeId>> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    Box::new(adj.neighbors(node).into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Direction::Outgoing => Box::new(std::iter::empty()),
        };

        forward.chain(backward)
    }

    /// Returns edges from a node with their targets.
    ///
    /// Returns an iterator of (target_node, edge_id) pairs.
    pub fn edges_from(
        &self,
        node: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let forward: Box<dyn Iterator<Item = (NodeId, EdgeId)>> = match direction {
            Direction::Outgoing | Direction::Both => {
                Box::new(self.forward_adj.edges_from(node).into_iter())
            }
            Direction::Incoming => Box::new(std::iter::empty()),
        };

        let backward: Box<dyn Iterator<Item = (NodeId, EdgeId)>> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    Box::new(adj.edges_from(node).into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Direction::Outgoing => Box::new(std::iter::empty()),
        };

        forward.chain(backward)
    }

    /// Returns edges from a node that are visible at a transaction's snapshot,
    /// and records each visible edge into the SSI read-set.
    ///
    /// Uses raw adjacency (including soft-deleted entries) so that an edge
    /// deleted *after* the snapshot's start is still examined by the version
    /// chain and correctly returned as visible.  The `deleted` filter in the
    /// regular `edges_from` path would hide such edges before the version chain
    /// gets a chance to evaluate them — which would be both unsound (wrong path)
    /// and a missed read for SSI purposes.
    ///
    /// The read-recording is a side-effect of `is_edge_visible_versioned`: it is
    /// a no-op for non-Serializable transactions (no tracker registered).
    pub fn edges_from_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Vec<(NodeId, EdgeId)> {
        let forward: Vec<(NodeId, EdgeId)> = match direction {
            Direction::Outgoing | Direction::Both => self
                .forward_adj
                .edges_from_including_deleted(node)
                .into_iter()
                .filter(|&(_, eid)| self.is_edge_visible_versioned(eid, epoch, tx))
                .collect(),
            Direction::Incoming => Vec::new(),
        };

        let backward: Vec<(NodeId, EdgeId)> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    adj.edges_from_including_deleted(node)
                        .into_iter()
                        .filter(|&(_, eid)| self.is_edge_visible_versioned(eid, epoch, tx))
                        .collect()
                } else {
                    Vec::new()
                }
            }
            Direction::Outgoing => Vec::new(),
        };

        forward.into_iter().chain(backward).collect()
    }

    /// Returns neighbor node IDs reachable via snapshot-visible edges.
    ///
    /// Derives targets from `edges_from_versioned` so that a node reachable
    /// only via an invisible edge is never yielded.
    pub fn neighbors_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Vec<NodeId> {
        let mut targets: Vec<NodeId> = self
            .edges_from_versioned(node, direction, epoch, tx)
            .into_iter()
            .map(|(target, _)| target)
            .collect();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    /// Returns edges to a node (where the node is the destination).
    ///
    /// Returns (source_node, edge_id) pairs for all edges pointing TO this node.
    /// Uses the backward adjacency index for O(degree) lookup.
    ///
    /// # Example
    ///
    /// ```
    /// # use grafeo_core::graph::lpg::LpgStore;
    /// # use grafeo_common::types::Value;
    /// let store = LpgStore::new().expect("arena allocation");
    /// let a = store.create_node(&["Node"]);
    /// let b = store.create_node(&["Node"]);
    /// let c = store.create_node(&["Node"]);
    /// let _e1 = store.create_edge(a, b, "LINKS");
    /// let _e2 = store.create_edge(c, b, "LINKS");
    ///
    /// // For edges: A->B, C->B
    /// let incoming = store.edges_to(b);
    /// assert_eq!(incoming.len(), 2);
    /// ```
    pub fn edges_to(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        if let Some(ref backward) = self.backward_adj {
            backward.edges_from(node)
        } else {
            // Fallback: scan all edges (slow but correct)
            self.all_edges()
                .filter_map(|edge| {
                    if edge.dst == node {
                        Some((edge.src, edge.id))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    /// Returns the out-degree of a node (number of outgoing edges).
    ///
    /// Uses the forward adjacency index for O(1) lookup.
    #[must_use]
    pub fn out_degree(&self, node: NodeId) -> usize {
        self.forward_adj.out_degree(node)
    }

    /// Returns the in-degree of a node (number of incoming edges).
    ///
    /// Uses the backward adjacency index for O(1) lookup if available,
    /// otherwise falls back to scanning edges.
    #[must_use]
    pub fn in_degree(&self, node: NodeId) -> usize {
        if let Some(ref backward) = self.backward_adj {
            backward.in_degree(node)
        } else {
            // Fallback: count edges (slow)
            self.all_edges().filter(|edge| edge.dst == node).count()
        }
    }

    // === Admin API: Iteration ===

    /// Returns an iterator over all nodes in the database.
    ///
    /// This creates a snapshot of all visible nodes at the current epoch.
    /// Useful for dump/export operations.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn all_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        let epoch = self.current_epoch();
        let node_ids: Vec<NodeId> = self
            .nodes
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();

        node_ids.into_iter().filter_map(move |id| self.get_node(id))
    }

    /// Returns an iterator over all nodes in the database.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn all_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        let node_ids = self.node_ids();
        node_ids.into_iter().filter_map(move |id| self.get_node(id))
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// This creates a snapshot of all visible edges at the current epoch.
    /// Useful for dump/export operations.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        let epoch = self.current_epoch();
        let edge_ids: Vec<EdgeId> = self
            .edges
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();

        edge_ids.into_iter().filter_map(move |id| self.get_edge(id))
    }

    /// Returns an iterator over all edges in the database.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        let epoch = self.current_epoch();
        let versions = self.edge_versions.read();
        let edge_ids: Vec<EdgeId> = versions
            .iter()
            .filter_map(|(id, index)| {
                index.visible_at(epoch).and_then(|vref| {
                    self.read_edge_record(&vref)
                        .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
                })
            })
            .collect();

        edge_ids.into_iter().filter_map(move |id| self.get_edge(id))
    }

    /// Returns an iterator over nodes with a specific label.
    pub fn nodes_with_label<'a>(&'a self, label: &str) -> impl Iterator<Item = Node> + 'a {
        let node_ids = self.nodes_by_label(label);
        node_ids.into_iter().filter_map(move |id| self.get_node(id))
    }

    /// Returns an iterator over edges with a specific type.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edges_with_type<'a>(&'a self, edge_type: &str) -> impl Iterator<Item = Edge> + 'a {
        let epoch = self.current_epoch();
        let type_to_id = self.edge_type_to_id.read();

        if let Some(&type_id) = type_to_id.get(edge_type) {
            let edge_ids: Vec<EdgeId> = self
                .edges
                .read()
                .iter()
                .filter_map(|(id, chain)| {
                    chain.visible_at(epoch).and_then(|r| {
                        if !r.is_deleted() && r.type_id == type_id {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            // Return a boxed iterator for the found edges
            Box::new(edge_ids.into_iter().filter_map(move |id| self.get_edge(id)))
                as Box<dyn Iterator<Item = Edge> + 'a>
        } else {
            // Return empty iterator
            Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Edge> + 'a>
        }
    }

    /// Returns an iterator over edges with a specific type.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn edges_with_type<'a>(&'a self, edge_type: &str) -> impl Iterator<Item = Edge> + 'a {
        let epoch = self.current_epoch();
        let type_to_id = self.edge_type_to_id.read();

        if let Some(&type_id) = type_to_id.get(edge_type) {
            let versions = self.edge_versions.read();
            let edge_ids: Vec<EdgeId> = versions
                .iter()
                .filter_map(|(id, index)| {
                    index.visible_at(epoch).and_then(|vref| {
                        self.read_edge_record(&vref).and_then(|r| {
                            if !r.is_deleted() && r.type_id == type_id {
                                Some(*id)
                            } else {
                                None
                            }
                        })
                    })
                })
                .collect();

            Box::new(edge_ids.into_iter().filter_map(move |id| self.get_edge(id)))
                as Box<dyn Iterator<Item = Edge> + 'a>
        } else {
            Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Edge> + 'a>
        }
    }
}
