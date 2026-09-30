//! Persistent configuration must select storage available in this build.

#![cfg(any(feature = "lpg", feature = "triple-store"))]

use grafeo_common::utils::error::Error;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB};
use std::path::{Path, PathBuf};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Files = Vec<(PathBuf, bool, Vec<u8>)>;

fn inventory(root: &Path) -> std::io::Result<Files> {
    fn visit(root: &Path, path: &Path, files: &mut Files) -> std::io::Result<()> {
        for entry in path.read_dir()? {
            let path = entry?.path();
            let directory = path.is_dir();
            files.push((
                path.strip_prefix(root).expect("fixture path").to_path_buf(),
                directory,
                if directory {
                    Vec::new()
                } else {
                    std::fs::read(&path)?
                },
            ));
            if directory {
                visit(root, &path, files)?;
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, root, &mut files)?;
    files.sort();
    Ok(files)
}

fn rejects_without_changes(root: &Path, config: Config) -> TestResult {
    let before = inventory(root)?;
    let error = GrafeoDB::with_config(config).err();
    assert_eq!(inventory(root)?, before, "rejection changed the filesystem");
    assert!(
        matches!(&error, Some(Error::InvalidValue(message)) if message.contains("persist")),
        "expected structured persistence admission error, got {error:?}",
    );
    Ok(())
}

#[test]
fn persistent_directory_requires_enabled_wal() -> TestResult {
    for format in [StorageFormat::Auto, StorageFormat::WalDirectory] {
        for existing in [false, true] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("directory");
            if existing {
                std::fs::create_dir(&path)?;
                std::fs::write(path.join("sentinel"), b"retain existing bytes")?;
            }
            let mut config = Config::persistent(&path).with_storage_format(format);
            config.wal_enabled = false;
            rejects_without_changes(dir.path(), config)?;
        }
    }
    Ok(())
}

#[cfg(not(feature = "wal"))]
#[test]
fn persistent_directory_requires_compiled_wal() -> TestResult {
    let dir = tempfile::tempdir()?;
    for format in [StorageFormat::Auto, StorageFormat::WalDirectory] {
        let config = Config::persistent(dir.path().join("directory")).with_storage_format(format);
        assert!(config.wal_enabled);
        rejects_without_changes(dir.path(), config)?;
    }
    Ok(())
}

#[cfg(not(feature = "grafeo-file"))]
#[test]
fn persistent_configuration_requires_a_storage_capability() -> TestResult {
    let dir = tempfile::tempdir()?;
    for format in [
        StorageFormat::Auto,
        StorageFormat::SingleFile,
        StorageFormat::WalDirectory,
    ] {
        rejects_without_changes(
            dir.path(),
            Config::persistent(dir.path().join("missing.grafeo")).with_storage_format(format),
        )?;
    }
    let path = dir.path().join("existing.grafeo");
    std::fs::write(&path, b"retain unsupported input without opening it")?;
    rejects_without_changes(dir.path(), Config::read_only(path))?;
    Ok(())
}

#[cfg(feature = "grafeo-file")]
fn seed_data(db: &GrafeoDB) -> TestResult {
    #[cfg(feature = "lpg")]
    assert!(
        db.session()
            .create_node_with_props(
                &["Persisted"],
                [("value", grafeo_common::types::Value::Int64(42))],
            )?
            .is_valid()
    );
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    {
        use grafeo_core::graph::rdf::{Term, Triple};
        assert_eq!(
            db.batch_insert_rdf([Triple::new(
                Term::iri("http://example.org/s"),
                Term::iri("http://example.org/p"),
                Term::iri("http://example.org/o"),
            )])?,
            1
        );
    }
    Ok(())
}

#[cfg(feature = "grafeo-file")]
#[test]
fn standalone_container_without_wal_roundtrips_exact_data() -> TestResult {
    for (name, format) in [
        ("store.grafeo", StorageFormat::Auto),
        ("container", StorageFormat::SingleFile),
    ] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(name);
        let mut config = Config::persistent(&path).with_storage_format(format);
        config.wal_enabled = false;
        let db = GrafeoDB::with_config(config)?;
        seed_data(&db)?;
        let exact = db.export_snapshot()?;
        db.close()?;
        assert!(path.is_file());
        let mut config = Config::persistent(&path);
        config.wal_enabled = false;
        let reopened = GrafeoDB::with_config(config)?;
        assert_eq!(reopened.export_snapshot()?, exact);
        reopened.close()?;
        let bytes = std::fs::read(&path)?;
        let read_only = GrafeoDB::with_config(Config::read_only(&path))?;
        assert_eq!(read_only.export_snapshot()?, exact);
        read_only.close()?;
        assert_eq!(std::fs::read(&path)?, bytes);
    }
    Ok(())
}

#[cfg(feature = "wal")]
#[test]
fn enabled_wal_directory_roundtrips_exact_data() -> TestResult {
    for format in [StorageFormat::Auto, StorageFormat::WalDirectory] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("directory");
        let db = GrafeoDB::with_config(Config::persistent(&path).with_storage_format(format))?;
        seed_data(&db)?;
        let exact = db.export_snapshot()?;
        db.close()?;
        assert!(path.is_dir());
        let reopened = GrafeoDB::with_config(Config::persistent(&path))?;
        assert_eq!(reopened.export_snapshot()?, exact);
        reopened.close()?;
    }
    Ok(())
}
