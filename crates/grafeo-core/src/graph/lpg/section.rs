//! LPG section serializer for the `.grafeo` container format.
//!
//! Implements the [`Section`] trait for LPG graph data (nodes, edges,
//! properties, named graphs). LPG4 is the only accepted format. Exact temporal
//! state and recursive literal graph paths are fully prepared on a detached
//! candidate before one pristine-target installation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::GraphPath;
use grafeo_common::utils::error::Result;

use crate::graph::lpg::LpgStore;

#[path = "section_v4.rs"]
mod v4;

/// Current LPG section format version (v4 = recursive exact temporal state).
const LPG_SECTION_VERSION: u8 = 4;

// ── Section implementation ──────────────────────────────────────────

/// LPG store section for the `.grafeo` container.
///
/// Wraps an `Arc<LpgStore>` and implements the [`Section`] trait for
/// serialization/deserialization of exact LPG graph state using format v4.
pub struct LpgStoreSection {
    store: Arc<LpgStore>,
    dirty: AtomicBool,
}

/// Eager LPG4 bytes and the exact sorted root-inclusive physical graph set
/// from one recursive topology/store pin.
///
/// The graph Arcs retain identities, not a viewing lease. An engine collecting
/// Catalog/Text/Vector or other sections afterwards must continuously retain
/// its existing publication barrier from before capture through those reads.
pub struct LpgSectionCapture {
    /// Already encoded immutable LPG4 payload; no later store reread is needed.
    pub bytes: Vec<u8>,
    /// Root first, then literal component-qualified paths in canonical order.
    pub graphs: Vec<(GraphPath, Arc<LpgStore>)>,
}

impl LpgStoreSection {
    /// Serializes a replacement workspace's already qualified owner set.
    /// The private root is exclusively borrowed; every escaped child is either
    /// still private or retained under its exact aggregate transition.
    pub(crate) fn serialize_replacement_image(
        root: &LpgStore,
        graphs: &[(GraphPath, Arc<LpgStore>)],
    ) -> Result<Vec<u8>> {
        v4::serialize_replacement_image(root, graphs)
    }

    /// Captures LPG bytes and graph owners together without exposing an
    /// arbitrary-cut serializer or invoking external code under store pins.
    ///
    /// The caller must retain its publication barrier continuously across this
    /// call and any subsequent catalog/auxiliary capture using `graphs`. This
    /// releases LPG pins before auxiliary snapshot gates are acquired.
    ///
    /// # Errors
    /// Rejects nested capture, invalid topology/history, and allocation failure.
    pub fn capture(&self) -> Result<LpgSectionCapture> {
        let (graphs, bytes) = self
            .store
            .with_pinned_owned_capture(v4::serialize_pinned_graphs)?;
        Ok(LpgSectionCapture { bytes, graphs })
    }

    /// Captures only the same canonical root-inclusive graph owner set.
    ///
    /// Recovery can qualify section owners without serializing a just-restored
    /// database. As with [`Self::capture`], keep the existing publication
    /// barrier across later consumers; retained Arcs alone do not freeze data.
    ///
    /// # Errors
    /// Rejects nested capture, invalid topology, and allocation failure.
    pub fn capture_graphs(&self) -> Result<Vec<(GraphPath, Arc<LpgStore>)>> {
        self.store
            .with_pinned_owned_capture(|_| Ok(()))
            .map(|(graphs, ())| graphs)
    }

    /// Create a new LPG section wrapping the given store.
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self {
            store,
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty (has unsaved changes).
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Access the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }
}

impl Section for LpgStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::LpgStore
    }

    fn version(&self) -> u8 {
        LPG_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        v4::serialize(&self.store)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        v4::deserialize_into(&self.store, data)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        let (store, indexes, mvcc, string_pool) = self.store.memory_breakdown();
        store.total_bytes + indexes.total_bytes + mvcc.total_bytes + string_pool.total_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use arcstr::ArcStr;
    use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn shared_capture_returns_exact_root_and_literal_path_owners_with_eager_bytes()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Arc::new(LpgStore::new()?);
        let root_node = source.create_node(&["Root"]);
        let empty = source.graph_or_create("")?;
        let parent = source.graph_or_create("a")?;
        let nested = parent.graph_or_create("b")?;
        let literal = source.graph_or_create("a/b")?;
        let nested_node = nested.create_node(&["Before"]);
        let owner = WriteAuthority::new();
        assert!(source.seal_unframed_writes(&owner));
        let section = LpgStoreSection::new(Arc::clone(&source));
        let captured = section.capture()?;
        let expected = [
            (GraphPath::root(), &source),
            (GraphPath::from_components(&[""])?, &empty),
            (GraphPath::from_components(&["a"])?, &parent),
            (GraphPath::from_components(&["a", "b"])?, &nested),
            (GraphPath::from_components(&["a/b"])?, &literal),
        ];
        assert_eq!(captured.graphs.len(), expected.len());
        for ((path, graph), (expected_path, expected_graph)) in
            captured.graphs.iter().zip(&expected)
        {
            assert_eq!(path, expected_path);
            assert!(Arc::ptr_eq(graph, expected_graph));
        }
        let topology_only = section.capture_graphs()?;
        assert_eq!(topology_only.len(), captured.graphs.len());
        for ((path, graph), (captured_path, captured_graph)) in
            topology_only.iter().zip(&captured.graphs)
        {
            assert_eq!(path, captured_path);
            assert!(Arc::ptr_eq(graph, captured_graph));
        }
        assert_eq!(captured.bytes, section.serialize()?);
        // Returned Arcs preserve identity, not a data lease. The bytes are
        // already encoded and cannot silently reobserve this later mutation.
        with_authority(&owner, || {
            source.set_node_property(root_node, "later", Value::Int64(1));
            assert!(nested.add_label(nested_node, "After"));
        });
        let restored = Arc::new(LpgStore::new()?);
        LpgStoreSection::new(Arc::clone(&restored)).deserialize(&captured.bytes)?;
        assert!(
            restored
                .get_node_property(root_node, &PropertyKey::new("later"))
                .is_none()
        );
        let restored_nested = restored
            .graph("a")
            .and_then(|graph| graph.graph("b"))
            .ok_or("missing nested image")?;
        assert!(
            restored_nested
                .get_node(nested_node)
                .is_some_and(|node| node.labels.len() == 1)
        );
        assert!(
            nested
                .get_node(nested_node)
                .is_some_and(|node| node.labels.len() == 2)
        );
        Ok(())
    }

    #[test]
    fn shared_capture_topology_only_skips_history_encoding_and_reentry_fails_closed()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Arc::new(LpgStore::new()?);
        let pending =
            source.create_node_versioned(&["Pending"], EpochId::PENDING, TransactionId::new(1));
        assert!(pending.is_valid());
        let section = LpgStoreSection::new(Arc::clone(&source));
        assert_eq!(section.capture_graphs()?.len(), 1);
        assert!(section.capture().is_err());
        source.with_pinned_recursive_capture(|_, _| {
            assert!(section.capture().is_err());
            assert!(section.capture_graphs().is_err());
            Ok(())
        })?;
        // Neither codec rejection nor nested admission leaks a read-only scope.
        assert!(source.create_node(&["After"]).is_valid());
        assert!(source.create_graph("after")?);
        Ok(())
    }

    fn node_lifetimes(store: &LpgStore, id: NodeId) -> Vec<(EpochId, Option<EpochId>)> {
        let mut history: Vec<_> = store
            .get_node_history(id)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect();
        history.reverse();
        history
    }

    fn edge_lifetimes(store: &LpgStore, id: EdgeId) -> Vec<(EpochId, Option<EpochId>)> {
        let mut history: Vec<_> = store
            .get_edge_history(id)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect();
        history.reverse();
        history
    }

    #[test]
    fn lpg_section_round_trip() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", Value::String("Alix".into()));
        store.set_node_property(n2, "name", Value::String("Gus".into()));
        store.create_edge(n1, n2, "KNOWS");

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"LPG4"));

        // Deserialize into a fresh store
        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(section2.store().node_count(), 2);
        assert_eq!(section2.store().edge_count(), 1);
    }

    #[test]
    fn lpg_section_dirty_tracking() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(store);

        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn lpg_section_type() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(store);
        assert_eq!(section.section_type(), SectionType::LpgStore);
        assert_eq!(section.version(), LPG_SECTION_VERSION);
    }

    #[test]
    fn lpg_section_empty_round_trip() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        section2.deserialize(&bytes).unwrap();
        assert_eq!(section2.store().node_count(), 0);
        assert_eq!(section2.store().edge_count(), 0);
    }

    #[test]
    fn lpg_section_properties_preserved() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n = store.create_node(&["Person"]);
        store.set_node_property(n, "name", Value::String("Alix".into()));
        store.set_node_property(n, "age", Value::Int64(30));
        store.set_node_property(n, "active", Value::Bool(true));

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        let node = store2.get_node(n).unwrap();
        let name_key: PropertyKey = "name".into();
        let age_key: PropertyKey = "age".into();
        let active_key: PropertyKey = "active".into();
        assert_eq!(
            node.properties.get(&name_key),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(node.properties.get(&age_key), Some(&Value::Int64(30)));
        assert_eq!(node.properties.get(&active_key), Some(&Value::Bool(true)));
    }

    #[test]
    fn lpg_section_named_graphs() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Root"]);
        store.create_graph("social").unwrap();

        if let Some(g) = store.graph("social") {
            g.create_node(&["Friend"]);
        }

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        assert_eq!(store2.node_count(), 1);
        assert!(store2.graph("social").is_some());
        assert_eq!(store2.graph("social").unwrap().node_count(), 1);
    }

    #[test]
    fn lpg_section_crc_integrity() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Test"]);

        let section = LpgStoreSection::new(Arc::clone(&store));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn predecessor_payloads_reject_at_magic_without_mutation() {
        let target = Arc::new(LpgStore::new().unwrap());
        let existing = target.create_node(&["Existing"]);
        let before = LpgStoreSection::new(Arc::clone(&target))
            .serialize()
            .unwrap();
        for payload in [b"LPG1".as_slice(), b"LPGB", b"LPG2", b"LPG3", b"LPG5", &[]] {
            let error = LpgStoreSection::new(Arc::clone(&target))
                .deserialize(payload)
                .expect_err("predecessor/unknown generation must reject");
            assert!(
                error
                    .to_string()
                    .contains("unsupported LPG section generation"),
                "{error}"
            );
            assert_eq!(
                LpgStoreSection::new(Arc::clone(&target))
                    .serialize()
                    .unwrap(),
                before
            );
            assert!(target.get_node(existing).is_some());
        }
    }

    #[test]
    fn sealed_lpg_section_restore_requires_exact_authority() {
        let source = Arc::new(LpgStore::new().unwrap());
        source.create_node(&["Restored"]);
        source.create_graph("archive").unwrap();
        source
            .graph("archive")
            .unwrap()
            .create_node(&["NamedRestored"]);
        let bytes = LpgStoreSection::new(source).serialize().unwrap();

        let target = Arc::new(LpgStore::new().unwrap());
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(target.seal_unframed_writes(&owner));

        let restore = || {
            let mut section = LpgStoreSection::new(Arc::clone(&target));
            section.deserialize(&bytes)
        };
        let error = restore().expect_err("raw section restore must fail closed");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.node_count(), 0);

        let error = with_authority(&foreign, restore)
            .expect_err("foreign authority must not restore a sealed store");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.node_count(), 0);

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || {
                target.with_exclusive_bulk_restore(|_| {
                    panic!("injected bulk-restore panic");
                });
            });
        }));
        assert!(unwind.is_err());
        let error = restore().expect_err("bulk-restore unwind must release authority context");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.node_count(), 0);

        with_authority(&owner, || restore().expect("owner recovery restore"));
        assert_eq!(target.node_count(), 1);
        assert_eq!(target.graph("archive").unwrap().node_count(), 1);

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || panic!("injected section panic"));
        }));
        let error = restore().expect_err("caught panic must release authority");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.node_count(), 1);
    }

    #[test]
    fn lpg_v4_round_trip_preserves_exact_history_epochs_and_allocator_high_water() {
        let source = Arc::new(LpgStore::new().unwrap());
        let e1 = EpochId::new(1);
        let e2 = EpochId::new(2);
        let e3 = EpochId::new(3);
        let e4 = EpochId::new(4);
        let e5 = EpochId::new(5);
        let e7 = EpochId::new(7);
        let e8 = EpochId::new(8);
        let e9 = EpochId::new(9);
        let e10 = EpochId::new(10);
        let e20 = EpochId::new(20);
        let recreated = NodeId::new(42);
        let anchor = NodeId::new(43);
        let relation = EdgeId::new(70);

        source
            .restore_node_history_exact(
                recreated,
                &[(e2, Some(e5)), (e7, None)],
                &[
                    (e2, vec![ArcStr::from("Draft")]),
                    (e4, vec![ArcStr::from("Draft"), ArcStr::from("Person")]),
                    (e5, vec![ArcStr::from("DeletedBoundary")]),
                    (e7, vec![ArcStr::from("Person")]),
                    (e9, vec![ArcStr::from("Person"), ArcStr::from("Researcher")]),
                ],
            )
            .unwrap();
        source
            .restore_node_history_exact(
                anchor,
                &[(e1, None)],
                &[(e1, vec![ArcStr::from("Anchor")])],
            )
            .unwrap();
        source.set_node_property_at_epoch(recreated, "name", Value::from("draft"), e2);
        source.set_node_property_at_epoch(recreated, "name", Value::Null, e5);
        source.set_node_property_at_epoch(recreated, "name", Value::from("Alix"), e7);
        source.set_node_property_at_epoch(recreated, "name", Value::from("Alix R."), e9);
        source
            .restore_edge_history_exact(
                relation,
                recreated,
                anchor,
                "KNOWS",
                &[(e3, Some(e5)), (e8, Some(e10))],
            )
            .unwrap();
        source.set_edge_property_at_epoch(relation, "weight", Value::Int64(1), e3);
        source.set_edge_property_at_epoch(relation, "weight", Value::Null, e5);
        source.set_edge_property_at_epoch(relation, "weight", Value::Int64(2), e8);
        source.set_edge_property_at_epoch(relation, "weight", Value::Null, e10);
        source.sync_epoch(e20);
        source.set_next_node_id(500);
        source.set_next_edge_id(600);

        source.create_graph("research").unwrap();
        let named = source.graph("research").unwrap();
        let named_person = NodeId::new(8);
        named
            .restore_node_history_exact(
                named_person,
                &[(e3, Some(e4)), (e8, None)],
                &[
                    (e3, vec![ArcStr::from("Draft")]),
                    (e4, Vec::new()),
                    (e8, vec![ArcStr::from("Paper")]),
                ],
            )
            .unwrap();
        named.set_node_property_at_epoch(named_person, "title", Value::from("Grafeo"), e8);
        named.sync_epoch(EpochId::new(30));
        named.set_next_node_id(700);
        named.set_next_edge_id(800);

        let bytes = LpgStoreSection::new(Arc::clone(&source))
            .serialize()
            .unwrap();
        let restored = Arc::new(LpgStore::new().unwrap());
        LpgStoreSection::new(Arc::clone(&restored))
            .deserialize(&bytes)
            .unwrap();

        assert_eq!(restored.current_epoch(), e20);
        assert_eq!(restored.next_node_id(), 500);
        assert_eq!(restored.next_edge_id(), 600);
        assert_eq!(
            node_lifetimes(&restored, recreated),
            vec![(e2, Some(e5)), (e7, None)]
        );
        assert_eq!(
            restored.node_label_history(recreated),
            source.node_label_history(recreated)
        );
        assert_eq!(
            restored.node_property_history_for_key(recreated, "name"),
            source.node_property_history_for_key(recreated, "name")
        );
        assert_eq!(
            edge_lifetimes(&restored, relation),
            vec![(e3, Some(e5)), (e8, Some(e10))]
        );
        assert_eq!(
            restored.edge_property_history(relation),
            source.edge_property_history(relation)
        );
        let restored_edge = restored.get_edge_at_epoch(relation, e8).unwrap();
        assert_eq!(restored_edge.src, recreated);
        assert_eq!(restored_edge.dst, anchor);
        assert_eq!(restored_edge.edge_type.as_str(), "KNOWS");

        let restored_named = restored.graph("research").unwrap();
        assert_eq!(restored_named.current_epoch(), EpochId::new(30));
        assert_eq!(restored_named.next_node_id(), 700);
        assert_eq!(restored_named.next_edge_id(), 800);
        assert_eq!(
            node_lifetimes(&restored_named, named_person),
            vec![(e3, Some(e4)), (e8, None)]
        );
        assert_eq!(
            restored_named.node_label_history(named_person),
            named.node_label_history(named_person)
        );
        assert_eq!(
            restored_named.node_property_history_for_key(named_person, "title"),
            named.node_property_history_for_key(named_person, "title")
        );
    }

    fn canonical_source(reverse: bool) -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().unwrap());
        let epoch = EpochId::new(4);
        let ids = if reverse {
            [NodeId::new(9), NodeId::new(2)]
        } else {
            [NodeId::new(2), NodeId::new(9)]
        };
        for id in ids {
            let labels = if reverse {
                vec![ArcStr::from("Zulu"), ArcStr::from("Alpha")]
            } else {
                vec![ArcStr::from("Alpha"), ArcStr::from("Zulu")]
            };
            store
                .restore_node_history_exact(id, &[(epoch, None)], &[(epoch, labels)])
                .unwrap();
        }

        let mut counter = HashMap::new();
        if reverse {
            counter.insert("replica-z".to_owned(), 9);
            counter.insert("replica-a".to_owned(), 2);
            store.set_node_property_at_epoch(NodeId::new(2), "zeta", Value::Int64(2), epoch);
            store.set_node_property_at_epoch(
                NodeId::new(2),
                "counter",
                Value::GCounter(Arc::new(counter)),
                epoch,
            );
        } else {
            counter.insert("replica-a".to_owned(), 2);
            counter.insert("replica-z".to_owned(), 9);
            store.set_node_property_at_epoch(
                NodeId::new(2),
                "counter",
                Value::GCounter(Arc::new(counter)),
                epoch,
            );
            store.set_node_property_at_epoch(NodeId::new(2), "zeta", Value::Int64(2), epoch);
        }

        // Hold native lifetimes fixed while varying registry insertion order.
        // Different CREATE orders now describe different durable histories.
        let a = Arc::new(store.new_named_graph_candidate().unwrap());
        let z = Arc::new(store.new_named_graph_candidate().unwrap());
        let entries = if reverse {
            [("z", z), ("a", a)]
        } else {
            [("a", a), ("z", z)]
        };
        for (name, graph) in entries {
            assert!(store.install_graph_if_absent(name, graph));
        }
        store.sync_epoch(EpochId::new(12));
        store.set_next_node_id(50);
        store.set_next_edge_id(60);
        store
    }

    #[test]
    fn lpg_v4_encoding_is_canonical_across_insertion_orders() {
        let left = LpgStoreSection::new(canonical_source(false))
            .serialize()
            .unwrap();
        let right = LpgStoreSection::new(canonical_source(true))
            .serialize()
            .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn lpg_v4_preserves_lossless_nested_and_distributed_property_values() {
        use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};

        let source = Arc::new(LpgStore::new().unwrap());
        let node = source.create_node(&["Values"]);
        let epoch = source.current_epoch();
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("nested"), Value::Int64(7));
        let mut positive = HashMap::new();
        positive.insert("b".to_owned(), 8);
        positive.insert("a".to_owned(), 3);
        let mut negative = HashMap::new();
        negative.insert("c".to_owned(), 2);
        let values = [
            ("null", Value::Null),
            ("bool", Value::Bool(true)),
            ("int", Value::Int64(-42)),
            ("float", Value::Float64(2.5)),
            ("string", Value::from("hello")),
            ("bytes", Value::Bytes(vec![1, 2, 3].into())),
            (
                "timestamp",
                Value::Timestamp(Timestamp::from_micros(1_234_567)),
            ),
            ("date", Value::Date(Date::from_days(19_000))),
            (
                "time",
                Value::Time(Time::from_nanos(123_000).unwrap().with_offset(3600)),
            ),
            ("duration", Value::Duration(Duration::new(2, 3, 4))),
            (
                "zoned",
                Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                    Timestamp::from_micros(5_678_901),
                    -3600,
                )),
            ),
            (
                "list",
                Value::List(vec![Value::Bool(false), Value::Int64(9)].into()),
            ),
            ("map", Value::Map(Arc::new(map))),
            ("vector", Value::Vector(vec![1.0, 2.0, 3.0].into())),
            (
                "path",
                Value::Path {
                    nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
                    edges: vec![Value::from("LINK")].into(),
                },
            ),
            ("gcounter", Value::GCounter(Arc::new(positive.clone()))),
            (
                "oncounter",
                Value::OnCounter {
                    pos: Arc::new(positive),
                    neg: Arc::new(negative),
                },
            ),
            (
                "rdf",
                Value::RdfLiteral {
                    lexical: ArcStr::from("bonjour"),
                    language: Some(ArcStr::from("fr")),
                    datatype: None,
                },
            ),
        ];
        for (key, value) in &values {
            source.set_node_property_at_epoch(node, key, value.clone(), epoch);
        }
        let float_bits = 0x7ff8_0000_0000_0042_u64;
        source.set_node_property_at_epoch(
            node,
            "float_bits",
            Value::Float64(f64::from_bits(float_bits)),
            epoch,
        );
        let vector_bits = [0x8000_0000_u32, 0x7fc0_0042_u32];
        source.set_node_property_at_epoch(
            node,
            "vector_bits",
            Value::Vector(vector_bits.map(f32::from_bits).to_vec().into()),
            epoch,
        );

        let bytes = LpgStoreSection::new(Arc::clone(&source))
            .serialize()
            .unwrap();
        let restored = Arc::new(LpgStore::new().unwrap());
        LpgStoreSection::new(Arc::clone(&restored))
            .deserialize(&bytes)
            .unwrap();

        for (key, _) in values {
            assert_eq!(
                restored.node_property_history_for_key(node, key),
                source.node_property_history_for_key(node, key),
                "property history {key}"
            );
        }
        match restored.get_node_property(node, &PropertyKey::new("float_bits")) {
            Some(Value::Float64(value)) => assert_eq!(value.to_bits(), float_bits),
            value => panic!("unexpected float_bits value: {value:?}"),
        }
        match restored.get_node_property(node, &PropertyKey::new("vector_bits")) {
            Some(Value::Vector(values)) => assert_eq!(
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                vector_bits
            ),
            value => panic!("unexpected vector_bits value: {value:?}"),
        }
        match restored.get_node_property(node, &PropertyKey::new("zoned")) {
            Some(Value::ZonedDatetime(value)) => {
                assert_eq!(value.as_timestamp(), Timestamp::from_micros(5_678_901));
                assert_eq!(value.offset_seconds(), -3600);
            }
            value => panic!("unexpected zoned value: {value:?}"),
        }
    }

    #[test]
    fn exact_allocator_restore_validates_both_high_water_marks_before_installing_either() {
        let store = LpgStore::new().unwrap();
        let left = store.create_node(&["Left"]);
        let right = store.create_node(&["Right"]);
        store.create_edge(left, right, "LINK");
        let before_node = store.next_node_id();
        let before_edge = store.next_edge_id();

        let error = store
            .restore_allocator_high_water_exact(20, 0)
            .expect_err("edge allocator regression must reject both marks");
        assert!(error.contains("next edge id"));
        assert_eq!(store.next_node_id(), before_node);
        assert_eq!(store.next_edge_id(), before_edge);

        store
            .restore_allocator_high_water_exact(20, 30)
            .expect("monotonic high-water restore");
        assert_eq!(store.next_node_id(), 20);
        assert_eq!(store.next_edge_id(), 30);
    }

    #[test]
    fn lpg_v4_retains_allocator_gaps_left_by_aborted_creates() {
        let source = Arc::new(LpgStore::new().unwrap());
        let left = source.create_node(&["Left"]);
        let right = source.create_node(&["Right"]);
        let transaction = TransactionId::new(99);
        let abandoned_node =
            source.create_node_versioned(&["Abandoned"], source.current_epoch(), transaction);
        let abandoned_edge = source.create_edge_versioned(
            left,
            right,
            "ABANDONED",
            source.current_epoch(),
            transaction,
        );
        source.discard_entities_by_id(transaction, &[abandoned_node], &[abandoned_edge]);
        let next_node = source.next_node_id();
        let next_edge = source.next_edge_id();

        let bytes = LpgStoreSection::new(source).serialize().unwrap();
        let restored = Arc::new(LpgStore::new().unwrap());
        LpgStoreSection::new(Arc::clone(&restored))
            .deserialize(&bytes)
            .unwrap();

        assert_eq!(restored.next_node_id(), next_node);
        assert_eq!(restored.next_edge_id(), next_edge);
        assert_eq!(restored.create_node(&["After"]), NodeId::new(next_node));
        assert_eq!(
            restored.create_edge(left, right, "AFTER"),
            EdgeId::new(next_edge)
        );
    }

    #[test]
    fn lpg_v4_rejects_trailing_bytes_before_mutating_target() {
        let source = Arc::new(LpgStore::new().unwrap());
        source.create_node(&["Source"]);
        let mut bytes = LpgStoreSection::new(source).serialize().unwrap();
        bytes.push(0xA5);

        let target = Arc::new(LpgStore::new().unwrap());
        target.set_next_node_id(100);
        let sentinel = target.create_node(&["Sentinel"]);
        let before_next = target.next_node_id();
        let error = LpgStoreSection::new(Arc::clone(&target))
            .deserialize(&bytes)
            .expect_err("trailing bytes must fail closed");
        assert!(error.to_string().contains("trailing"));
        assert!(target.get_node(sentinel).is_some());
        assert_eq!(target.node_count(), 1);
        assert_eq!(target.next_node_id(), before_next);
    }
}
