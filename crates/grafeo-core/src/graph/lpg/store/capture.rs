//! Coherent recursive section capture under existing topology/transition gates.

use super::{EXCLUSIVE_BULK_RESTORE_STACK, ExclusiveMode, LpgStore, NAMED_GRAPH_TOPOLOGY_GATE};
use grafeo_common::types::GraphPath;
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashSet;
use std::sync::Arc;

type CapturedGraphs = Vec<(GraphPath, Arc<LpgStore>)>;

struct ReadContext {
    previous_depth: usize,
}

impl ReadContext {
    fn enter(root: &LpgStore, children: &[(GraphPath, Arc<LpgStore>)]) -> Result<Self> {
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow_mut(|stack| {
            let previous_depth = stack.len();
            let count = children
                .len()
                .checked_add(1)
                .ok_or_else(|| Error::Serialization("recursive capture count overflow".into()))?;
            stack
                .try_reserve(count)
                .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            stack.push((std::ptr::from_ref(root).addr(), ExclusiveMode::ReadOnly));
            for (_, child) in children {
                if !std::ptr::eq(root, child.as_ref()) {
                    stack.push((Arc::as_ptr(child).addr(), ExclusiveMode::ReadOnly));
                }
            }
            Ok(Self { previous_depth })
        })
    }
}

impl Drop for ReadContext {
    fn drop(&mut self) {
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow_mut(|stack| stack.truncate(self.previous_depth));
    }
}

fn collect_children(
    store: &LpgStore,
    parent: &GraphPath,
    children: &mut Vec<(GraphPath, Arc<LpgStore>)>,
    seen: &mut FxHashSet<usize>,
) -> Result<()> {
    for (name, child) in store.named_graphs.read().iter() {
        let path = parent
            .child(name)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        seen.try_reserve(1)
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        if !seen.insert(Arc::as_ptr(child).addr()) {
            return Err(Error::Serialization(
                "recursive LPG capture contains a cycle or aliased graph".into(),
            ));
        }
        children
            .try_reserve(1)
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        children.push((path, Arc::clone(child)));
    }
    Ok(())
}

impl LpgStore {
    /// Pins one recursive topology and every store's existing mutation barrier.
    /// Nested accessors may reuse the read proof; logical mutators cannot.
    pub(crate) fn with_pinned_recursive_capture<T>(
        &self,
        capture: impl FnOnce(&LpgStore, &[(GraphPath, Arc<LpgStore>)]) -> Result<T>,
    ) -> Result<T> {
        self.with_pinned_capture_graphs(None, |children| capture(self, children))
            .map(|(_, result)| result)
    }

    /// Returns the actual root-inclusive graph owners observed by this pin.
    /// The callback cannot supply a replacement or stale graph set.
    pub(crate) fn with_pinned_owned_capture<T>(
        self: &Arc<Self>,
        capture: impl FnOnce(&[(GraphPath, Arc<LpgStore>)]) -> Result<T>,
    ) -> Result<(CapturedGraphs, T)> {
        self.with_pinned_capture_graphs(Some(Arc::clone(self)), capture)
    }

    fn with_pinned_capture_graphs<T>(
        &self,
        root: Option<Arc<LpgStore>>,
        capture: impl FnOnce(&[(GraphPath, Arc<LpgStore>)]) -> Result<T>,
    ) -> Result<(CapturedGraphs, T)> {
        if EXCLUSIVE_BULK_RESTORE_STACK.with_borrow(|stack| !stack.is_empty()) {
            return Err(Error::Serialization(
                "recursive LPG capture cannot nest inside an exclusive store transition".into(),
            ));
        }
        // Owning graph/path scratch outlives the complete pin scope. Returning
        // these exact Arcs never makes a second recursive topology observation.
        let mut seen = FxHashSet::default();
        seen.try_reserve(1)
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        seen.insert(std::ptr::from_ref(self).addr());
        let mut children = Vec::new();
        let includes_root = root.is_some();
        if let Some(root) = root {
            if !std::ptr::eq(self, root.as_ref()) {
                return Err(Error::Serialization(
                    "recursive capture root identity differs".into(),
                ));
            }
            children
                .try_reserve(1)
                .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            children.push((GraphPath::root(), root));
        }
        let result = {
            let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
            if !includes_root {
                collect_children(self, &GraphPath::root(), &mut children, &mut seen)?;
            }
            let mut next = 0;
            while next < children.len() {
                let (path, child) = children[next].clone();
                collect_children(&child, &path, &mut children, &mut seen)?;
                next += 1;
            }
            children.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            let mut child_pins = Vec::new();
            child_pins
                .try_reserve(children.len())
                .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            let _root_pin = self.pin_exclusive_maintenance();
            for (_, child) in &children {
                if !std::ptr::eq(self, child.as_ref()) {
                    child_pins.push(child.pin_exclusive_maintenance());
                }
            }
            let _context = ReadContext::enter(self, &children)?;
            capture(&children)?
        };
        Ok((children, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStoreSection;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use grafeo_common::storage::section::Section;
    use grafeo_common::types::{EpochId, PropertyKey, Value};

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn recursive_capture_reads_sealed_sections_without_write_authority() -> TestResult {
        let source = Arc::new(LpgStore::new()?);
        let node = source.create_node(&["Root"]);
        source.set_node_property(node, "name", Value::from("retained"));
        source.create_graph("a/b")?;
        let parent = source.graph_or_create("a")?;
        let child = parent.graph_or_create("b")?;
        assert!(child.create_node(&["Child"]).is_valid());
        let owner = WriteAuthority::new();
        assert!(source.seal_unframed_writes(&owner));
        let bytes = LpgStoreSection::new(Arc::clone(&source)).serialize()?;
        let restored = Arc::new(LpgStore::new()?);
        LpgStoreSection::new(Arc::clone(&restored)).deserialize(&bytes)?;
        assert_eq!(
            restored.get_node_property(node, &PropertyKey::new("name")),
            Some(Value::from("retained"))
        );
        assert!(restored.graph("a/b").is_some());
        assert_eq!(
            restored
                .graph("a")
                .and_then(|graph| graph.graph("b"))
                .ok_or("missing nested restored graph")?
                .node_count(),
            1
        );
        assert!(!source.create_node(&["Denied"]).is_valid());
        assert!(!child.create_node(&["Denied"]).is_valid());
        Ok(())
    }

    #[test]
    fn recursive_capture_denies_logical_mutation_even_with_held_authority() -> TestResult {
        let source = LpgStore::new()?;
        let root_node = source.create_node(&["Root"]);
        let child = source.graph_or_create("child")?;
        let child_node = child.create_node(&["Child"]);
        let owner = WriteAuthority::new();
        assert!(source.seal_unframed_writes(&owner));
        with_authority(&owner, || {
            source.with_pinned_recursive_capture(|root, children| {
                assert_eq!(children.len(), 1);
                assert!(root.graph("child").is_some());
                for (store, node) in [(root, root_node), (child.as_ref(), child_node)] {
                    let before = (
                        store.next_node_id(),
                        store.next_edge_id(),
                        store.current_epoch(),
                    );
                    assert!(store.get_node(node).is_some());
                    assert!(!store.create_node(&["Denied"]).is_valid());
                    assert!(!store.delete_node(node));
                    assert!(!store.add_label(node, "Denied"));
                    store.set_node_property(node, "denied", Value::Int64(1));
                    store.sync_epoch(EpochId::new(40));
                    store.clear();
                    assert!(store.restore_allocator_high_water_exact(100, 100).is_err());
                    assert_eq!(
                        (
                            store.next_node_id(),
                            store.next_edge_id(),
                            store.current_epoch()
                        ),
                        before
                    );
                    assert!(
                        store
                            .get_node_property(node, &PropertyKey::new("denied"))
                            .is_none()
                    );
                    assert_eq!(store.node_count(), 1);
                }
                assert!(root.with_pinned_recursive_capture(|_, _| Ok(())).is_err());
                Ok(())
            })
        })?;
        assert!(with_authority(&owner, || source.create_node(&["After"])).is_valid());
        Ok(())
    }

    #[test]
    fn recursive_capture_rejects_topology_reentry_on_captured_and_unrelated_stores() -> TestResult {
        let source = LpgStore::new()?;
        let child = source.graph_or_create("child")?;
        let unrelated = LpgStore::new()?;
        let replacement = Arc::new(LpgStore::new()?);
        let owner = WriteAuthority::new();
        source.with_pinned_recursive_capture(|root, _| {
            for store in [root, &unrelated] {
                assert!(store.graph_or_create("denied").is_err());
                assert!(!store.create_graph("denied").is_ok_and(|created| created));
                assert!(!store.drop_graph("child"));
                assert!(!store.install_graph_if_absent("denied", Arc::clone(&replacement)));
                assert!(!store.drop_graph_if_same("child", &child));
                assert!(!store.replace_graph_if_same("child", &child, Arc::clone(&replacement)));
                assert!(!store.seal_unframed_writes(&owner));
            }
            Ok(())
        })?;
        assert!(
            source
                .graph("child")
                .is_some_and(|actual| Arc::ptr_eq(&actual, &child))
        );
        assert!(unrelated.graph_names().is_empty());
        assert!(unrelated.create_graph("after")?);
        Ok(())
    }

    #[test]
    fn recursive_capture_error_and_unwind_release_context_and_all_pins() -> TestResult {
        let source = LpgStore::new()?;
        let child = source.graph_or_create("child")?;
        let owner = WriteAuthority::new();
        assert!(source.seal_unframed_writes(&owner));
        let rejected: Result<()> = source.with_pinned_recursive_capture(|_, _| {
            Err(Error::Serialization("injected capture rejection".into()))
        });
        assert!(rejected.is_err());
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<()> = source.with_pinned_recursive_capture(|_, _| {
                std::panic::resume_unwind(Box::new("injected capture unwind"))
            });
        }));
        assert!(unwind.is_err());
        assert!(EXCLUSIVE_BULK_RESTORE_STACK.with_borrow(Vec::is_empty));
        assert!(!source.create_node(&["NoLeakedAuthority"]).is_valid());
        with_authority(&owner, || {
            assert!(source.create_node(&["After"]).is_valid());
            assert!(child.create_node(&["After"]).is_valid());
        });
        source.with_pinned_recursive_capture(|root, children| {
            assert_eq!(root.node_count(), 1);
            assert_eq!(children.len(), 1);
            assert_eq!(children[0].1.node_count(), 1);
            Ok(())
        })?;
        Ok(())
    }
}
