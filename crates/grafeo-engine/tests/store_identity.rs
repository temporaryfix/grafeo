//! Model-neutral logical-store identity qualification.

#![cfg(feature = "lpg")]

#[cfg(any(feature = "wal", feature = "grafeo-file"))]
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

#[test]
fn portable_snapshot_restore_preserves_lpg_only_identity() {
    let source =
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    assert!(source.create_node(&["World"]).is_valid());
    let expected = source.store_id();

    let restored = GrafeoDB::import_snapshot(&source.export_snapshot().unwrap()).unwrap();
    assert_eq!(restored.store_id(), expected);
    assert_eq!(restored.node_count(), 1);
}

#[cfg(feature = "wal")]
#[test]
fn wal_directory_reopen_preserves_lpg_only_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("world");
    let expected = {
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Lpg)
                .with_storage_format(StorageFormat::WalDirectory),
        )
        .unwrap();
        assert!(db.create_node(&["World"]).is_valid());
        let store_id = db.store_id();
        db.close().unwrap();
        store_id
    };

    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Lpg)
            .with_storage_format(StorageFormat::WalDirectory),
    )
    .unwrap();
    assert_eq!(reopened.store_id(), expected);
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}

#[cfg(feature = "grafeo-file")]
#[test]
fn current_single_file_checkpoint_preserves_lpg_only_identity() {
    use grafeo_storage::file::GrafeoFileManager;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("current.grafeo");
    let config = Config::persistent(&path)
        .with_graph_model(GraphModel::Lpg)
        .with_storage_format(StorageFormat::SingleFile);
    let expected = {
        let source = GrafeoDB::with_config(config.clone()).unwrap();
        assert!(source.create_node(&["World"]).is_valid());
        let identity = source.world_identity();
        source.close().unwrap();
        identity
    };

    // Inspect the real checkpoint only after its writer has released the lock.
    // A portable blob wrapped in a monolithic container is not this format.
    {
        let manager = GrafeoFileManager::open_read_only(&path).unwrap();
        assert!(manager.read_section_directory().unwrap().is_some());
        manager.close().unwrap();
    }

    let reopened = GrafeoDB::with_config(config).unwrap();
    assert_eq!(reopened.world_identity(), expected);
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}
