//! Outer-owned authority guards and topology scratch for registry/commit batches.

use super::{
    DataRebindError, LpgStore, PendingStore, PinnedLpgTransition, conflict, physical_order,
    reserve_vec,
};
#[cfg(feature = "text-index")]
use crate::index::text::{InvertedIndex, TextScopeTransition};
#[cfg(feature = "vector-index")]
use crate::index::vector::{HnswScopeTransition, VectorIndexKind};
use grafeo_common::types::GraphPath;
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::MutexGuard;
use std::sync::Arc;

/// Buffers outlive the complete enclosing publication scope, including errors.
/// Guards borrow only independent store anchors, never this workspace's fields.
pub(super) struct RegistryAuthorityWorkspace<'store> {
    transitions: Vec<PinnedLpgTransition<'store>>,
    bindings: Option<BindingFences>,
    nodes: Vec<TopologyNode<'store>>,
    identities: FxHashMap<usize, usize>,
    ordered: Vec<&'store LpgStore>,
    frontier: Vec<Visit>,
    attempted: bool,
}

struct TopologyNode<'store> {
    anchor: TopologyAnchor<'store>,
    state: VisitState,
}

#[derive(Clone)]
enum TopologyAnchor<'store> {
    Target(&'store LpgStore),
    Descendant(Arc<LpgStore>),
}

impl TopologyAnchor<'_> {
    fn store(&self) -> &LpgStore {
        match self {
            Self::Target(store) => store,
            Self::Descendant(store) => store,
        }
    }
}

enum VisitState {
    Fresh,
    Active,
    Complete,
}

enum Visit {
    Enter(usize),
    Leave(usize),
}

impl<'store> RegistryAuthorityWorkspace<'store> {
    pub(super) fn new() -> Self {
        Self {
            transitions: Vec::new(),
            bindings: None,
            nodes: Vec::new(),
            identities: FxHashMap::default(),
            ordered: Vec::new(),
            frontier: Vec::new(),
            attempted: false,
        }
    }

    /// The enclosing commit cleanup owner exists before entering this method.
    /// It drains every component before these raw authority guards on failure,
    /// unwind, or a deliberately forgotten callback proof.
    pub(super) fn acquire_pending(&mut self, pending: &mut [PendingStore<'store>]) -> Result<()> {
        self.reserve(pending.len())?;
        if pending.is_empty() {
            // Read-only/RDF-only engine commits have no LPG authority to pin.
            // Do not serialize unrelated databases on global index/topology
            // gates just to return an empty physical companion.
            return Ok(());
        }
        self.bindings = Some(BindingFences::acquire());
        self.order_stores(pending)?;
        for (index, store) in self.ordered.iter().enumerate() {
            let position = pending
                .iter()
                .position(|candidate| std::ptr::eq(candidate.store, *store))
                .ok_or_else(|| conflict("ordered store does not belong to the workspace"))?;
            pending.swap(index, position);
        }
        for candidate in pending {
            let transition = candidate
                .store
                .pin_exclusive_unframed_transition()
                .ok_or_else(|| conflict("mutation authority denied or representation retired"))?;
            self.transitions.push(transition);
        }
        Ok(())
    }

    fn reserve(&mut self, count: usize) -> Result<()> {
        if self.attempted {
            return Err(conflict("authority workspace acquisition is one-shot"));
        }
        self.attempted = true;
        reserve_vec(&mut self.transitions, count)?;
        reserve_vec(&mut self.nodes, count)?;
        reserve_vec(&mut self.ordered, count)?;
        reserve_vec(&mut self.frontier, count)?;
        super::reserve_map(&mut self.identities, count)
    }

    fn order_stores(&mut self, pending: &[PendingStore<'store>]) -> Result<()> {
        // Insert all targets first, so each target keeps its independent outer
        // reference even when discovered through another target's descendants.
        for item in pending {
            let identity = physical_order(item.store);
            if self.identities.contains_key(&identity) {
                return Err(conflict("duplicate physical store"));
            }
            self.identities.insert(identity, self.nodes.len());
            self.nodes.push(TopologyNode {
                anchor: TopologyAnchor::Target(item.store),
                state: VisitState::Fresh,
            });
        }
        self.frontier
            .extend((0..self.nodes.len()).map(Visit::Enter));
        self.frontier.sort_unstable_by_key(|visit| match visit {
            Visit::Enter(index) | Visit::Leave(index) => self
                .nodes
                .get(*index)
                .map(|node| physical_order(node.anchor.store())),
        });
        // One iterative DFS over the union of reachable stores, rather than a
        // complete descendant set per target. Reverse postorder qualifies all
        // parent-before-child constraints, including targets encountered first
        // through another root. Anchors and scratch remain outer-owned.
        while let Some(visit) = self.frontier.pop() {
            match visit {
                Visit::Leave(index) => {
                    let node = self
                        .nodes
                        .get_mut(index)
                        .ok_or_else(|| conflict("topology exit lacks its captured node"))?;
                    if !matches!(node.state, VisitState::Active) {
                        return Err(conflict("topology exit is not active"));
                    }
                    node.state = VisitState::Complete;
                    if let TopologyAnchor::Target(store) = &node.anchor {
                        self.ordered.push(*store);
                    }
                }
                Visit::Enter(index) => {
                    let node = self
                        .nodes
                        .get_mut(index)
                        .ok_or_else(|| conflict("topology entry lacks its captured node"))?;
                    match node.state {
                        VisitState::Complete => continue,
                        VisitState::Active => {
                            return Err(conflict("named graph topology contains a cycle"));
                        }
                        VisitState::Fresh => {}
                    }
                    node.state = VisitState::Active;
                    // This temporary clone cannot retire a store: the original
                    // anchor stays in nodes through complete guard release.
                    let anchor = node.anchor.clone();
                    reserve_vec(&mut self.frontier, 1)?;
                    self.frontier.push(Visit::Leave(index));
                    let children = anchor.store().named_graphs.read();
                    reserve_vec(&mut self.nodes, children.len())?;
                    reserve_vec(&mut self.frontier, children.len())?;
                    super::reserve_map(&mut self.identities, children.len())?;
                    for child in children.values() {
                        let identity = physical_order(child);
                        let index = if let Some(index) = self.identities.get(&identity) {
                            *index
                        } else {
                            let index = self.nodes.len();
                            self.identities.insert(identity, index);
                            self.nodes.push(TopologyNode {
                                anchor: TopologyAnchor::Descendant(Arc::clone(child)),
                                state: VisitState::Fresh,
                            });
                            index
                        };
                        self.frontier.push(Visit::Enter(index));
                    }
                }
            }
        }
        self.ordered.reverse();
        Ok(())
    }

    pub(super) fn transition(
        &self,
        store: &LpgStore,
    ) -> std::result::Result<&PinnedLpgTransition<'store>, DataRebindError> {
        self.retained_transitions()
            .iter()
            .find(|transition| transition.pins_store(store))
            .ok_or_else(|| DataRebindError::new("registry store lacks the retained transition"))
    }

    pub(super) fn retained_transitions(&self) -> &[PinnedLpgTransition<'store>] {
        &self.transitions
    }

    pub(super) fn bindings(&self) -> Result<&BindingFences> {
        self.bindings
            .as_ref()
            .ok_or_else(|| conflict("registry binding authority was not retained"))
    }

    pub(super) fn release_guards(&mut self) {
        while let Some(transition) = self.transitions.pop() {
            drop(transition);
        }
        drop(self.bindings.take());
    }
}

impl Drop for RegistryAuthorityWorkspace<'_> {
    fn drop(&mut self) {
        // Covers a forgotten borrowed authority as well as partial acquisition.
        // Guard release precedes every scratch/anchor field's destructor.
        self.release_guards();
    }
}

/// One continuous authority loan; Drop releases guards but never their buffers.
pub(super) struct RegistryAuthority<'store, 'workspace> {
    pub(super) workspace: &'workspace mut RegistryAuthorityWorkspace<'store>,
}

impl<'store, 'workspace> RegistryAuthority<'store, 'workspace> {
    pub(super) fn acquire(
        pending: &mut [PendingStore<'store>],
        workspace: &'workspace mut RegistryAuthorityWorkspace<'store>,
    ) -> Result<Self> {
        let authority = Self { workspace };
        authority.workspace.acquire_pending(pending)?;
        Ok(authority)
    }

    #[cfg(test)]
    pub(super) fn transition(
        &self,
        store: &LpgStore,
    ) -> std::result::Result<&PinnedLpgTransition<'store>, DataRebindError> {
        self.workspace.transition(store)
    }
}

impl Drop for RegistryAuthority<'_, '_> {
    fn drop(&mut self) {
        self.workspace.release_guards();
    }
}

pub(super) struct BindingFences {
    // Concrete proofs release in reverse acquisition order without freeing a
    // closure box beneath enclosing lifecycle/publication gates.
    #[cfg(feature = "text-index")]
    text: TextScopeTransition,
    #[cfg(feature = "vector-index")]
    vector: HnswScopeTransition,
    _topology: MutexGuard<'static, ()>,
}

impl BindingFences {
    fn acquire() -> Self {
        let topology = super::super::NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text = InvertedIndex::pin_scope_transition();
        Self {
            #[cfg(feature = "text-index")]
            text,
            #[cfg(feature = "vector-index")]
            vector,
            _topology: topology,
        }
    }

    pub(super) fn validate_graph(
        &self,
        root: &Arc<LpgStore>,
        path: &GraphPath,
        target: &LpgStore,
    ) -> Result<()> {
        // This proof continuously owns the global topology gate. Every
        // temporary Arc remains owned by its still-attached parent (or the
        // caller's independent root), so traversal cannot retire a store.
        // Do not call graph(): its maintenance admission would reenter an
        // already-retained exclusive target transition.
        let mut current = Arc::clone(root);
        for component in path.components() {
            let next = current.named_graphs.read().get(component.as_str()).cloned();
            current =
                next.ok_or_else(|| conflict("commit graph path was detached during admission"))?;
        }
        if !std::ptr::eq(current.as_ref(), target) {
            return Err(conflict("commit graph path resolves to another store"));
        }
        Ok(())
    }

    #[cfg(feature = "text-index")]
    pub(super) fn bind_text(
        &self,
        index: &InvertedIndex,
        owner: u64,
        slot: u64,
        scope: u64,
    ) -> Result<()> {
        if !index.binding_is_compatible(owner, slot, &self.text)
            || if scope == 0 {
                !index.scope_is_unsealed(&self.text)
            } else {
                !index.scope_is_compatible(scope, &self.text)
            }
        {
            return Err(conflict("Text contents have foreign binding or authority"));
        }
        if !index.bind_under_transition(owner, slot, &self.text)
            || (scope != 0 && !index.seal_with_scope_under_transition(scope, &self.text))
        {
            return Err(conflict("private Text binding was rejected"));
        }
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    pub(super) fn bind_vector(
        &self,
        index: &VectorIndexKind,
        owner: u64,
        slot: u64,
        scope: u64,
    ) -> Result<()> {
        if !index.binding_is_compatible(owner, slot, &self.vector)
            || if scope == 0 {
                !index.scope_is_unsealed(&self.vector)
            } else {
                !index.scope_is_compatible(scope, &self.vector)
            }
        {
            return Err(conflict(
                "Vector contents have foreign binding or authority",
            ));
        }
        if !index.bind_under_transition(owner, slot, &self.vector)
            || (scope != 0 && !index.seal_with_scope_under_transition(scope, &self.vector))
        {
            return Err(conflict("private Vector binding was rejected"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
