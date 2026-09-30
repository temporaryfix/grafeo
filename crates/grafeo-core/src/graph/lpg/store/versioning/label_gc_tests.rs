use super::LpgStore;
use crate::graph::lpg::LpgStoreSection;
use arcstr::ArcStr;
use grafeo_common::storage::Section;
use grafeo_common::types::{EpochId, NodeId};
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn labels(entries: &[(u64, &[&str])]) -> Vec<(EpochId, Vec<ArcStr>)> {
    entries
        .iter()
        .map(|(epoch, names)| {
            (
                EpochId::new(*epoch),
                names.iter().map(|name| ArcStr::from(*name)).collect(),
            )
        })
        .collect()
}

fn assert_exact_roundtrip(store: &Arc<LpgStore>, ids: &[NodeId]) -> TestResult {
    let bytes = LpgStoreSection::new(Arc::clone(store)).serialize()?;
    let restored = Arc::new(LpgStore::new()?);
    let mut section = LpgStoreSection::new(Arc::clone(&restored));
    section.deserialize(&bytes)?;
    assert_eq!(section.serialize()?, bytes);
    for &id in ids {
        assert_eq!(
            restored.node_label_history(id),
            store.node_label_history(id)
        );
        let lifetimes = |store: &LpgStore| {
            store
                .get_node_history(id)
                .into_iter()
                .map(|(created, deleted, _)| (created, deleted))
                .collect::<Vec<_>>()
        };
        assert_eq!(lifetimes(&restored), lifetimes(store));
    }
    Ok(())
}

#[test]
fn label_gc_preserves_live_birth_and_exact_baseline_without_retaining_all_changes() -> TestResult {
    let store = Arc::new(LpgStore::new()?);
    let id = NodeId::new(0);
    store.restore_node_history_exact(
        id,
        &[(EpochId::new(1), None)],
        &labels(&[
            (1, &[]),
            (1, &["Born"]),
            (3, &["Discardable"]),
            (5, &["Baseline"]),
            (7, &["Recent"]),
            (10, &["Current"]),
        ]),
    )?;
    let before: Vec<_> = (6..=10)
        .map(|epoch| {
            store
                .get_node_at_epoch(id, EpochId::new(epoch))
                .map(|node| node.labels)
        })
        .collect();
    store.gc_versions(EpochId::new(6));
    assert_eq!(
        store.node_label_history(id),
        labels(&[
            (1, &[]),
            (1, &["Born"]),
            (5, &["Baseline"]),
            (7, &["Recent"]),
            (10, &["Current"])
        ])
    );
    let after: Vec<_> = (6..=10)
        .map(|epoch| {
            store
                .get_node_at_epoch(id, EpochId::new(epoch))
                .map(|node| node.labels)
        })
        .collect();
    assert_eq!(after, before);
    assert_exact_roundtrip(&store, &[id])?;

    // Restore advances the clock through the last label change (10). Model
    // a later committed epoch before advancing the retained floor to 11.
    store.set_epoch(EpochId::new(11));
    store.gc_versions(EpochId::new(11));
    assert_eq!(
        store.node_label_history(id),
        labels(&[(1, &[]), (1, &["Born"]), (10, &["Current"])])
    );
    assert_exact_roundtrip(&store, &[id])
}

#[test]
fn label_gc_prunes_removed_lifetime_births_but_preserves_retained_deleted_births() -> TestResult {
    let store = Arc::new(LpgStore::new()?);
    let reincarnated = NodeId::new(0);
    let deleted = NodeId::new(1);
    store.restore_node_history_exact(
        reincarnated,
        &[
            (EpochId::new(1), Some(EpochId::new(4))),
            (EpochId::new(6), Some(EpochId::new(10))),
            (EpochId::new(12), Some(EpochId::new(18))),
            (EpochId::new(20), None),
        ],
        &labels(&[
            (1, &["First"]),
            (3, &["FirstChange"]),
            (4, &[]),
            (6, &["Second"]),
            (9, &["SecondChange"]),
            (10, &[]),
            (12, &["Third"]),
            (13, &["ThirdDiscardable"]),
            (14, &["ThirdBaseline"]),
            (16, &["ThirdRecent"]),
            (18, &[]),
            (20, &["Fourth"]),
            (22, &["FourthCurrent"]),
        ]),
    )?;
    store.restore_node_history_exact(
        deleted,
        &[(EpochId::new(1), Some(EpochId::new(8)))],
        &labels(&[(1, &["DeletedBirth"]), (3, &["DeletedChange"]), (8, &[])]),
    )?;
    assert_exact_roundtrip(&store, &[reincarnated, deleted])?;
    store.gc_versions(EpochId::new(15));
    assert_eq!(
        store.node_label_history(reincarnated),
        labels(&[
            (12, &["Third"]),
            (14, &["ThirdBaseline"]),
            (16, &["ThirdRecent"]),
            (18, &[]),
            (20, &["Fourth"]),
            (22, &["FourthCurrent"])
        ])
    );
    assert_eq!(store.get_node_history(reincarnated).len(), 2);
    assert_eq!(
        store.node_label_history(deleted),
        labels(&[(1, &["DeletedBirth"]), (8, &[])])
    );
    assert!(store.get_node_at_epoch(deleted, EpochId::new(1)).is_some());
    assert!(store.get_node_at_epoch(deleted, EpochId::new(8)).is_none());
    assert!(
        store
            .get_node_at_epoch(reincarnated, EpochId::new(18))
            .is_none()
    );
    assert!(
        store
            .get_node_at_epoch(reincarnated, EpochId::new(20))
            .is_some()
    );
    assert_exact_roundtrip(&store, &[reincarnated, deleted])?;

    // A GC horizon beyond the last restored change (22) needs a matching
    // committed database clock, even though no further labels changed.
    store.set_epoch(EpochId::new(23));
    store.gc_versions(EpochId::new(23));
    assert_eq!(store.get_node_history(reincarnated).len(), 1);
    assert_eq!(
        store.node_label_history(reincarnated),
        labels(&[(20, &["Fourth"]), (22, &["FourthCurrent"])])
    );
    assert_eq!(
        store.node_label_history(deleted),
        labels(&[(1, &["DeletedBirth"]), (8, &[])])
    );
    assert_exact_roundtrip(&store, &[reincarnated, deleted])
}
