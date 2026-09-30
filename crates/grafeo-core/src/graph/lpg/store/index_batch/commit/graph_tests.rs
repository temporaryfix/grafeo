use super::{LpgCommitWorkspace, StoreCommitInput, with_prepared_lpg_commit};
use crate::graph::lpg::store::NAMED_GRAPH_TOPOLOGY_GATE;
use crate::graph::lpg::{IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey, LpgStore};
use crate::graph::write_permit::{WriteAuthority, with_authority};
use grafeo_common::types::{EpochId, GraphPath, PropertyKey, TransactionId};
use grafeo_common::utils::error::Result;
use std::sync::Arc;

fn capture<'target>(
    root: &'target Arc<LpgStore>,
    path: &'target GraphPath,
    target: &'target LpgStore,
) -> LpgCommitWorkspace<'target> {
    LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            store: target,
            source: target,
            graph: Some((root, path)),
            publish_data: false,
            edits: vec![IndexRegistryEdit::Create {
                key: IndexRegistryKey::Property(PropertyKey::new("value")),
                contents: IndexRegistryContents::Property(vec![]),
            }],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TransactionId::new(1900),
        EpochId::INITIAL,
        EpochId::new(1),
    )
}

#[test]
fn graph_witness_rejects_ancestor_detach_and_replacement_after_capture() {
    for replace in [false, true] {
        let root = Arc::new(LpgStore::new().unwrap());
        let parent = root.graph_or_create("parent").unwrap();
        let target = parent.graph_or_create("leaf").unwrap();
        let replacement = Arc::new(root.new_named_graph_candidate().unwrap());
        let replacement_target = replacement.graph_or_create("leaf").unwrap();
        let path = GraphPath::from_components(&["parent", "leaf"]).unwrap();
        let mut workspace = capture(&root, &path, &target);

        // A captured leaf Arc remains writable when an ancestor is removed.
        // The graph witness, not target authority alone, must reject it.
        if replace {
            assert!(root.replace_graph_if_same("parent", &parent, Arc::clone(&replacement)));
        } else {
            assert!(root.drop_graph("parent"));
        }
        let authority = WriteAuthority::new();
        let mut callback_ran = false;
        let result: Result<()> = with_authority(&authority, || {
            with_prepared_lpg_commit(&mut workspace, &authority, |_| {
                callback_ran = true;
                Ok(())
            })
        });
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("commit graph path")
        );
        assert!(!callback_ran);
        assert!(!target.has_property_index("value"));
        assert!(!replacement_target.has_property_index("value"));
        assert_eq!(target.current_epoch(), EpochId::INITIAL);
        assert!(target.mutation_scope_gate.try_write().is_some());
        // Failure released the global topology gate as well as target guards.
        assert!(root.install_graph_if_absent(
            "after_failure",
            Arc::new(root.new_named_graph_candidate().unwrap())
        ));
    }
}

#[test]
fn graph_witness_distinguishes_literal_components_and_retains_admission_through_install() {
    let root = Arc::new(LpgStore::new().unwrap());
    let direct = root.graph_or_create("a/b").unwrap();
    let parent = root.graph_or_create("a").unwrap();
    let nested = parent.graph_or_create("b").unwrap();
    let empty = root.graph_or_create("").unwrap();
    let cases = [
        (
            direct,
            GraphPath::from_components(&["a", "b"]).unwrap(),
            GraphPath::from_components(&["a/b"]).unwrap(),
        ),
        (
            nested,
            GraphPath::from_components(&["a/b"]).unwrap(),
            GraphPath::from_components(&["a", "b"]).unwrap(),
        ),
        (
            empty,
            GraphPath::root(),
            GraphPath::from_components(&[""]).unwrap(),
        ),
        (
            Arc::clone(&root),
            GraphPath::from_components(&[""]).unwrap(),
            GraphPath::root(),
        ),
    ];
    for (target, wrong, exact) in &cases {
        let mut rejected = capture(&root, wrong, target);
        let mut accepted = capture(&root, exact, target);
        let authority = WriteAuthority::new();
        with_authority(&authority, || {
            let result: Result<()> = with_prepared_lpg_commit(&mut rejected, &authority, |_| {
                panic!("a different literal graph path reached publication")
            });
            assert!(result.unwrap_err().to_string().contains("another store"));
            assert!(!target.has_property_index("value"));
            with_prepared_lpg_commit(&mut accepted, &authority, |released| {
                assert!(NAMED_GRAPH_TOPOLOGY_GATE.try_lock().is_none());
                let installed = released.rebind().unwrap().install();
                assert!(NAMED_GRAPH_TOPOLOGY_GATE.try_lock().is_none());
                drop(installed);
                assert!(NAMED_GRAPH_TOPOLOGY_GATE.try_lock().is_none());
                Ok(())
            })
            .unwrap();
        });
        assert!(target.has_property_index("value"));
    }
}
