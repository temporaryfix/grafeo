//! Native graph lifetimes. Physical representations and preparation views share
//! one allocator; exact replacement images bring their own validated authority.

use super::LpgStore;
use grafeo_common::memory::arena::AllocError;
use grafeo_common::types::{GraphIncarnationId, GraphPath};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone)]
pub(super) struct GraphIdentity {
    incarnation: GraphIncarnationId,
    next: Arc<AtomicU64>,
}

impl Default for GraphIdentity {
    fn default() -> Self {
        Self {
            incarnation: GraphIncarnationId::DEFAULT_GRAPH,
            next: Arc::new(AtomicU64::new(1)),
        }
    }
}

impl LpgStore {
    /// Native lifetime, qualified by the owning StoreId and the LPG model.
    #[must_use]
    pub fn graph_incarnation_id(&self) -> GraphIncarnationId {
        let _maintenance = self.pin_maintenance();
        self.graph_identity.read().incarnation
    }

    /// First unallocated named lifetime, including reservations and retired IDs.
    #[must_use]
    pub fn next_graph_incarnation_id(&self) -> u64 {
        let _maintenance = self.pin_maintenance();
        self.graph_identity.read().next.load(Ordering::Acquire)
    }

    /// Prepares a named lifetime before transaction publication or its WAL marker.
    ///
    /// # Errors
    /// Returns allocation, identity exhaustion, or denied mutation authority.
    #[doc(hidden)]
    pub fn new_named_graph_candidate(&self) -> std::result::Result<Self, AllocError> {
        let _mutation = self.pin_mutation().ok_or(AllocError::InsufficientSpace)?;
        self.allocate_named_graph()
    }

    pub(super) fn allocate_named_graph(&self) -> std::result::Result<Self, AllocError> {
        let identity = self.graph_identity.read();
        let incarnation = identity
            .next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
            .map_err(|_| AllocError::InsufficientSpace)?;
        self.graph_candidate(
            GraphIncarnationId::new(incarnation),
            Arc::clone(&identity.next),
        )
    }

    fn graph_candidate(
        &self,
        incarnation: GraphIncarnationId,
        next: Arc<AtomicU64>,
    ) -> std::result::Result<Self, AllocError> {
        let mut graph = Self::new()?;
        *graph.graph_identity.get_mut() = GraphIdentity { incarnation, next };
        graph.mutation_scope.store(
            self.mutation_scope.load(Ordering::Acquire),
            Ordering::Release,
        );
        Ok(graph)
    }

    /// A detached registry for catalog preparation, with the same native owner.
    ///
    /// # Errors
    /// Returns allocation failure or denied mutation authority.
    #[doc(hidden)]
    pub fn new_graph_topology_candidate(&self) -> std::result::Result<Self, AllocError> {
        let _mutation = self.pin_mutation().ok_or(AllocError::InsufficientSpace)?;
        let identity = self.graph_identity.read();
        self.graph_candidate(identity.incarnation, Arc::clone(&identity.next))
    }

    /// Prepares an authenticated committed WAL lifetime. Recovery must first
    /// reject repeated IDs and IDs below the checkpoint allocation floor across
    /// the complete replay stream, including graphs subsequently dropped.
    ///
    /// # Errors
    /// Returns an invalid identity, allocation failure, or denied authority.
    #[doc(hidden)]
    pub fn new_replayed_graph_candidate(&self, incarnation: GraphIncarnationId) -> Result<Self> {
        let _mutation = self
            .pin_mutation()
            .ok_or_else(|| invalid("replay authority denied"))?;
        let next = incarnation
            .as_u64()
            .checked_add(1)
            .filter(|_| !incarnation.is_default_graph())
            .ok_or_else(|| invalid("invalid replay incarnation"))?;
        let identity = self.graph_identity.read();
        let graph = self.graph_candidate(incarnation, Arc::clone(&identity.next))?;
        identity.next.fetch_max(next, Ordering::AcqRel);
        Ok(graph)
    }

    pub(super) fn shares_graph_allocator(&self, child: &LpgStore) -> bool {
        let owner = self.graph_identity.read();
        let child = child.graph_identity.read();
        !child.incarnation.is_default_graph()
            && child.incarnation != owner.incarnation
            && Arc::ptr_eq(&owner.next, &child.next)
    }

    /// Validates a complete exact image's lifetime coordinates before building it.
    /// A standalone section may describe a named graph as its relative root.
    ///
    /// # Errors
    /// Rejects missing/duplicate paths, IDs, parents, or an invalid allocator floor.
    #[doc(hidden)]
    pub fn validate_graph_incarnations(
        rows: &[(GraphPath, GraphIncarnationId)],
        next: u64,
    ) -> Result<()> {
        if next == 0
            || rows
                .first()
                .is_none_or(|(path, _)| !path.components().is_empty())
        {
            return Err(invalid("missing root or invalid allocator floor"));
        }
        let mut ids = FxHashSet::default();
        for (index, (path, id)) in rows.iter().enumerate() {
            if id.as_u64() >= next || (index != 0 && id.is_default_graph()) || !ids.insert(*id) {
                return Err(invalid("duplicate or out-of-range incarnation"));
            }
            if index != 0 && rows[index - 1].0 >= *path {
                return Err(invalid("paths must be sorted and unique"));
            }
            if let Some(parent) = path.parent().map_err(|error| invalid(&error.to_string()))?
                && rows[..index]
                    .binary_search_by(|(path, _)| path.cmp(&parent))
                    .is_err()
            {
                return Err(invalid("missing incarnation parent"));
            }
        }
        Ok(())
    }

    /// Installs exact coordinates only into a uniquely owned detached tree.
    /// All validation and allocation precede the infallible metadata assignment.
    ///
    /// # Errors
    /// Rejects invalid coordinates, aliases, sealed stores, or incomplete coverage.
    #[doc(hidden)]
    pub fn restore_graph_incarnations(
        &mut self,
        rows: &[(GraphPath, GraphIncarnationId)],
        next: u64,
    ) -> Result<()> {
        Self::validate_graph_incarnations(rows, next)?;
        fn check(store: &mut LpgStore) -> Result<usize> {
            if *store.mutation_scope.get_mut() != 0 {
                return Err(invalid("identity restore requires an unsealed image"));
            }
            let mut count = 1usize;
            for graph in store.named_graphs.get_mut().values_mut() {
                let graph = Arc::get_mut(graph).ok_or_else(|| invalid("aliased restore graph"))?;
                count = count
                    .checked_add(check(graph)?)
                    .ok_or_else(|| invalid("graph count overflow"))?;
            }
            Ok(count)
        }
        fn at<'a>(mut graph: &'a mut LpgStore, path: &GraphPath) -> Option<&'a mut LpgStore> {
            for component in path.components() {
                graph = Arc::get_mut(graph.named_graphs.get_mut().get_mut(component)?)?;
            }
            Some(graph)
        }
        if check(self)? != rows.len() {
            return Err(invalid("incomplete incarnation coverage"));
        }
        for (path, _) in rows {
            if at(self, path).is_none() {
                return Err(invalid("incarnation path absent from image"));
            }
        }
        let allocator = Arc::new(AtomicU64::new(next));
        for (path, incarnation) in rows {
            let graph = at(self, path).expect("validated uniquely owned graph path");
            *graph.graph_identity.get_mut() = GraphIdentity {
                incarnation: *incarnation,
                next: Arc::clone(&allocator),
            };
        }
        Ok(())
    }
}

fn invalid(reason: &str) -> Error {
    Error::Serialization(format!("invalid LPG graph identity: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_native_creates_and_capture_denial_keep_one_allocation_order() {
        let root = LpgStore::new().unwrap();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let root = &root;
                scope.spawn(move || {
                    for graph in 0..8 {
                        assert!(root.create_graph(&format!("{worker}-{graph}")).unwrap());
                    }
                });
            }
        });
        let ids: std::collections::BTreeSet<_> = root
            .named_graph_entries()
            .values()
            .map(|graph| graph.graph_incarnation_id().as_u64())
            .collect();
        assert_eq!(ids, (1..65).collect());
        root.with_pinned_recursive_capture(|root, _| {
            assert!(root.new_named_graph_candidate().is_err());
            assert!(root.new_graph_topology_candidate().is_err());
            assert_eq!(root.next_graph_incarnation_id(), 65);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn native_lifetimes_share_one_allocator_and_never_reuse_dropped_ids() {
        let root = LpgStore::new().unwrap();
        let a = root.graph_or_create("a").unwrap();
        let nested = a.graph_or_create("b").unwrap();
        let literal = root.graph_or_create("a/b").unwrap();
        assert_eq!(
            (
                root.graph_incarnation_id().as_u64(),
                a.graph_incarnation_id().as_u64(),
                nested.graph_incarnation_id().as_u64(),
                literal.graph_incarnation_id().as_u64()
            ),
            (0, 1, 2, 3)
        );
        let staged = root.new_graph_topology_candidate().unwrap();
        let abandoned = staged.new_named_graph_candidate().unwrap();
        assert_eq!(abandoned.graph_incarnation_id().as_u64(), 4);
        assert!(root.drop_graph("a"));
        let replacement = root.graph_or_create("a").unwrap();
        assert_eq!(replacement.graph_incarnation_id().as_u64(), 5);
        assert_eq!(a.graph_incarnation_id().as_u64(), 1);
        assert_eq!(nested.next_graph_incarnation_id(), 6);
        assert!(!root.install_graph_if_absent("foreign", Arc::new(LpgStore::new().unwrap())));
    }

    #[test]
    fn exact_identity_restore_validates_before_mutating_and_exhaustion_is_checked() {
        let mut root = LpgStore::new().unwrap();
        root.create_graph("a").unwrap();
        let path = GraphPath::root().child("a").unwrap();
        let rows = vec![
            (GraphPath::root(), GraphIncarnationId::DEFAULT_GRAPH),
            (path.clone(), GraphIncarnationId::new(11)),
        ];
        root.restore_graph_incarnations(&rows, u64::MAX).unwrap();
        assert_eq!(root.graph("a").unwrap().graph_incarnation_id().as_u64(), 11);
        assert!(root.create_graph("exhausted").is_err());
        assert!(root.graph("exhausted").is_none());
        let duplicate = vec![
            (GraphPath::root(), GraphIncarnationId::DEFAULT_GRAPH),
            (path, GraphIncarnationId::DEFAULT_GRAPH),
        ];
        assert!(root.restore_graph_incarnations(&duplicate, 1).is_err());
        assert_eq!(root.graph("a").unwrap().graph_incarnation_id().as_u64(), 11);
        assert_eq!(root.next_graph_incarnation_id(), u64::MAX);
        let alias = root.graph("a").unwrap();
        assert!(root.restore_graph_incarnations(&rows, 12).is_err());
        assert_eq!(alias.next_graph_incarnation_id(), u64::MAX);
    }
}
