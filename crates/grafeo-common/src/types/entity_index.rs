//! `EntityIndex` — bidirectional content-address ↔ store-handle resolution.
//!
//! Maps durable content-addressed [`EntityRef`]s to the store's dense [`NodeId`]
//! handles and back. Lets callers resolve a node by its *durable* identity
//! (deterministic projection/rebuild, dedup, cross-store reference) while the hot
//! path keeps using `NodeId`. The `EntityRef` is the key that survives compaction
//! renumbering of `NodeId`s — [`bind`](EntityIndex::bind) re-points an entity at a
//! new handle while keeping the bimap consistent. Populated by ingest/projection.

use crate::types::{EntityRef, NodeId};
use crate::utils::hash::FxHashMap;

/// Bidirectional [`EntityRef`] ↔ [`NodeId`] index. See module docs.
#[derive(Clone, Debug, Default)]
pub struct EntityIndex {
    by_ref: FxHashMap<EntityRef, NodeId>,
    by_node: FxHashMap<NodeId, EntityRef>,
}

impl EntityIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_ref: FxHashMap::default(),
            by_node: FxHashMap::default(),
        }
    }

    /// Binds a durable `entity` to a store `node`, keeping both directions
    /// consistent. Idempotent for an existing pair; re-binding an entity to a new
    /// node (e.g. compaction renumber) drops the stale reverse entry, and binding a
    /// node already held by another entity drops that entity's stale forward entry.
    pub fn bind(&mut self, entity: EntityRef, node: NodeId) {
        if let Some(old_node) = self.by_ref.insert(entity, node)
            && old_node != node
        {
            self.by_node.remove(&old_node);
        }
        if let Some(old_entity) = self.by_node.insert(node, entity)
            && old_entity != entity
        {
            self.by_ref.remove(&old_entity);
        }
    }

    /// Resolves a durable [`EntityRef`] to its current store [`NodeId`].
    #[must_use]
    pub fn node_of(&self, entity: &EntityRef) -> Option<NodeId> {
        self.by_ref.get(entity).copied()
    }

    /// Resolves a store [`NodeId`] to its durable [`EntityRef`].
    #[must_use]
    pub fn entity_of(&self, node: NodeId) -> Option<EntityRef> {
        self.by_node.get(&node).copied()
    }

    /// Whether a durable entity is bound.
    #[must_use]
    pub fn contains_entity(&self, entity: &EntityRef) -> bool {
        self.by_ref.contains_key(entity)
    }

    /// Number of bound entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_ref.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_ref.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eref(b: u8) -> EntityRef {
        EntityRef::from_bytes([b; 16])
    }

    #[test]
    fn test_bind_and_resolve_both_directions() {
        let mut ix = EntityIndex::new();
        let e = eref(1);
        ix.bind(e, NodeId::new(42));
        assert_eq!(ix.node_of(&e), Some(NodeId::new(42)));
        assert_eq!(ix.entity_of(NodeId::new(42)), Some(e));
        assert_eq!(ix.len(), 1);
        assert!(ix.contains_entity(&e));
    }

    #[test]
    fn test_bind_is_idempotent_for_same_pair() {
        let mut ix = EntityIndex::new();
        let e = eref(2);
        ix.bind(e, NodeId::new(7));
        ix.bind(e, NodeId::new(7));
        assert_eq!(ix.len(), 1);
        assert_eq!(ix.node_of(&e), Some(NodeId::new(7)));
        assert_eq!(ix.entity_of(NodeId::new(7)), Some(e));
    }

    #[test]
    fn test_rebind_entity_to_new_node_drops_stale_reverse() {
        // Compaction renumber: entity keeps its identity, gets a new NodeId.
        let mut ix = EntityIndex::new();
        let e = eref(3);
        ix.bind(e, NodeId::new(1));
        ix.bind(e, NodeId::new(2));
        assert_eq!(ix.node_of(&e), Some(NodeId::new(2)));
        assert_eq!(ix.entity_of(NodeId::new(2)), Some(e));
        assert_eq!(ix.entity_of(NodeId::new(1)), None); // stale reverse dropped
        assert_eq!(ix.len(), 1);
    }

    #[test]
    fn test_rebind_node_to_new_entity_drops_stale_forward() {
        let mut ix = EntityIndex::new();
        let (e1, e2) = (eref(4), eref(5));
        ix.bind(e1, NodeId::new(9));
        ix.bind(e2, NodeId::new(9));
        assert_eq!(ix.entity_of(NodeId::new(9)), Some(e2));
        assert_eq!(ix.node_of(&e2), Some(NodeId::new(9)));
        assert_eq!(ix.node_of(&e1), None); // stale forward dropped
        assert_eq!(ix.len(), 1);
    }

    #[test]
    fn test_empty() {
        let ix = EntityIndex::new();
        assert!(ix.is_empty());
        assert_eq!(ix.node_of(&eref(0)), None);
        assert_eq!(ix.entity_of(NodeId::new(0)), None);
    }
}
