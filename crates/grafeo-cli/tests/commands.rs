//! Integration tests for CLI commands.

use std::path::Path;
use tempfile::TempDir;

#[derive(Debug, PartialEq, Eq)]
enum RestoreArtifactKind {
    Directory,
    File(Vec<u8>),
    Symlink(std::path::PathBuf),
}

#[derive(Debug, PartialEq, Eq)]
struct RestoreArtifact {
    kind: RestoreArtifactKind,
    #[cfg(unix)]
    coordination_identity: Option<(u64, u64)>,
}

fn restore_tree_image(
    root: &Path,
) -> Option<std::collections::BTreeMap<std::path::PathBuf, RestoreArtifact>> {
    fn visit(
        root: &Path,
        path: &Path,
        image: &mut std::collections::BTreeMap<std::path::PathBuf, RestoreArtifact>,
    ) {
        let metadata = std::fs::symlink_metadata(path).expect("snapshot artifact metadata");
        let kind = if metadata.file_type().is_symlink() {
            RestoreArtifactKind::Symlink(std::fs::read_link(path).expect("snapshot symlink"))
        } else if metadata.is_dir() {
            for entry in std::fs::read_dir(path).expect("snapshot directory") {
                visit(
                    root,
                    &entry.expect("snapshot directory entry").path(),
                    image,
                );
            }
            RestoreArtifactKind::Directory
        } else {
            assert!(
                metadata.is_file(),
                "unexpected fixture artifact: {}",
                path.display()
            );
            RestoreArtifactKind::File(std::fs::read(path).expect("snapshot file"))
        };
        #[cfg(unix)]
        let coordination_identity = {
            use std::os::unix::fs::MetadataExt;
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(".grafeo-owner"))
                .then(|| (metadata.dev(), metadata.ino()))
        };
        image.insert(
            path.strip_prefix(root).unwrap().to_owned(),
            RestoreArtifact {
                kind,
                #[cfg(unix)]
                coordination_identity,
            },
        );
    }

    match std::fs::symlink_metadata(root) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("snapshot root: {error}"),
    }
    let mut image = std::collections::BTreeMap::new();
    visit(root, root, &mut image);
    Some(image)
}

fn native_restore_cli(backup: &Path, target: &Path, force: bool) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_grafeo"));
    command.args(["backup", "restore"]).arg(backup).arg(target);
    if force {
        command.arg("--force");
    }
    let output = command.output().expect("spawn and reap actual grafeo CLI");
    assert!(
        output.status.code().is_some(),
        "CLI terminated by signal: {output:?}"
    );
    output
}

#[test]
fn native_restore_force_preserves_a_live_directory_before_backup_validation() {
    let temporary = TempDir::new().unwrap();
    let target = temporary.path().join("held-database");
    let db = create_test_db(&target);
    db.wal().unwrap().sync().unwrap();
    let coordinate = target.join("wal.grafeo-owner");
    assert!(
        coordinate.is_file(),
        "fixture must hold its actual inner W coordinate"
    );
    let before = restore_tree_image(&target);
    assert!(before.is_some());

    let output = native_restore_cli(&temporary.path().join("missing-backup"), &target, true);
    // Old code returns runtime 1 after deleting the target; fixed code rejects
    // the removed option with parser 2. Preservation is the decisive oracle.
    assert!(matches!(output.status.code(), Some(1 | 2)), "{output:?}");
    assert_eq!(
        restore_tree_image(&target),
        before,
        "native restore removed or changed a live target before backup validation; missing tree is explicit None; child={output:?}"
    );
    db.wal().unwrap().sync().unwrap();
    db.close().unwrap();
}

#[test]
fn native_restore_rejects_removed_force_flag() {
    let temporary = TempDir::new().unwrap();
    let target = temporary.path().join("never-created");
    let output = native_restore_cli(&temporary.path().join("missing-backup"), &target, true);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--force"));
    assert!(!target.exists());
}

#[test]
fn native_restore_help_omits_force() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_grafeo"))
        .args(["backup", "restore", "--help"])
        .output()
        .expect("spawn and reap actual CLI help");
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("<BACKUP>"), "{help}");
    assert!(help.contains("<PATH>"), "{help}");
    assert!(!help.contains("--force"), "{help}");
}

#[test]
fn native_restore_preserves_occupied_targets_with_valid_backup() {
    for target_name in ["occupied-directory", "occupied.grafeo"] {
        let temporary = TempDir::new().unwrap();
        let backup = temporary.path().join("backup.grafeo");
        let source = create_test_db(&backup);
        source.close().unwrap();
        drop(source);

        // Snapshot the complete target area: file C, sidecar W and their
        // permanent coordinates for file mode, or the whole directory tree.
        let target_area = temporary.path().join("target-area");
        std::fs::create_dir(&target_area).unwrap();
        let target = target_area.join(target_name);
        let held = create_test_db(&target);
        let marker = held.create_node(&["HeldTargetOnly"]);
        assert!(marker.is_valid());
        held.set_node_property(marker, "sentinel", "occupied-not-backup".into())
            .expect("set occupied-target sentinel");
        assert_eq!(
            held.execute("MATCH (n:HeldTargetOnly) RETURN n.sentinel")
                .unwrap()
                .scalar::<String>()
                .unwrap(),
            "occupied-not-backup"
        );
        held.wal().unwrap().sync().unwrap();
        let before = restore_tree_image(&target_area);
        assert!(before.is_some());

        let output = native_restore_cli(&backup, &target, false);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("already exists"), "{error}");
        assert!(error.contains("new destination"), "{error}");
        assert!(error.contains(&target.display().to_string()), "{error}");
        assert!(!error.contains("--force"), "{error}");
        assert_eq!(restore_tree_image(&target_area), before);
        assert_eq!(held.info().node_count, 4);
        held.wal().unwrap().sync().unwrap();
        held.close().unwrap();
    }
}

#[test]
fn native_restore_to_absent_targets_reopens_concrete_graph() {
    for target_name in ["restored-directory", "restored.grafeo"] {
        let temporary = TempDir::new().unwrap();
        let backup = temporary.path().join("backup.grafeo");
        let source = create_test_db(&backup);
        source.close().unwrap();
        drop(source);
        let target = temporary.path().join(target_name);
        assert!(!target.exists());

        let output = native_restore_cli(&backup, &target, false);
        assert_eq!(output.status.code(), Some(0), "{output:?}");
        let restored = grafeo_engine::GrafeoDB::open(&target).unwrap();
        assert_eq!(restored.info().node_count, 3);
        assert_eq!(restored.info().edge_count, 2);
        for (query, expected) in [
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN p.name, q.name",
                ["Alix", "Gus"],
            ),
            (
                "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name",
                ["Alix", "Acme Corp"],
            ),
        ] {
            let result = restored.execute(query).unwrap();
            assert_eq!(
                result.rows(),
                &[expected
                    .into_iter()
                    .map(grafeo_common::types::Value::from)
                    .collect::<Vec<_>>()],
                "restored query: {query}"
            );
        }
        restored.close().unwrap();
    }
}

/// Helper to create a test database.
fn create_test_db(dir: &Path) -> grafeo_engine::GrafeoDB {
    let db = grafeo_engine::GrafeoDB::open(dir).expect("Failed to create test database");

    // Add some test data
    let n1 = db.create_node(&["Person"]);
    let n2 = db.create_node(&["Person"]);
    let n3 = db.create_node(&["Company"]);

    db.set_node_property(n1, "name", "Alix".into())
        .expect("set node property");
    db.set_node_property(n2, "name", "Gus".into())
        .expect("set node property");
    db.set_node_property(n3, "name", "Acme Corp".into())
        .expect("set node property");

    db.create_edge(n1, n2, "KNOWS");
    db.create_edge(n1, n3, "WORKS_AT");

    db
}

#[test]
fn test_database_can_be_opened() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
    drop(db);

    // Reopen to verify persistence
    let db2 = grafeo_engine::GrafeoDB::open(&db_path).expect("reopen db");
    let info = db2.info();
    assert!(info.is_persistent);
}

#[test]
fn test_database_info() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let info = db.info();

    assert_eq!(info.node_count, 3);
    assert_eq!(info.edge_count, 2);
    assert!(info.is_persistent);
}

#[test]
fn test_database_stats() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let stats = db.detailed_stats();

    assert_eq!(stats.node_count, 3);
    assert_eq!(stats.edge_count, 2);
    assert_eq!(stats.label_count, 2); // Person, Company
    assert_eq!(stats.edge_type_count, 2); // KNOWS, WORKS_AT
    assert!(stats.property_key_count >= 1); // name
    // Note: memory_bytes may be 0 depending on implementation
}

#[test]
fn test_query_execution() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);

    // Test a simple query
    let result = db
        .execute("MATCH (n:Person) RETURN n.name")
        .expect("execute query");
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_in_memory_database() {
    let db = grafeo_engine::GrafeoDB::new_in_memory();
    let info = db.info();

    assert!(!info.is_persistent);
    assert_eq!(info.node_count, 0);
    assert_eq!(info.edge_count, 0);
}

#[test]
fn test_node_and_edge_creation() {
    let db = grafeo_engine::GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Test"]);
    let n2 = db.create_node(&["Test"]);
    let e1 = db.create_edge(n1, n2, "LINKS");

    let info = db.info();
    assert_eq!(info.node_count, 2);
    assert_eq!(info.edge_count, 1);

    // Verify edge exists
    let edge = db.get_edge(e1);
    assert!(edge.is_some());
}

#[test]
fn test_validate_passes_on_good_database() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let result = db.validate();

    assert!(result.errors.is_empty(), "expected no validation errors");
}

#[test]
fn test_schema_inspection() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let schema = db.schema();

    match schema {
        grafeo_engine::SchemaInfo::Lpg(lpg) => {
            let label_names: Vec<&str> = lpg.labels.iter().map(|l| l.name.as_str()).collect();
            assert!(label_names.contains(&"Person"));
            assert!(label_names.contains(&"Company"));

            let edge_names: Vec<&str> = lpg.edge_types.iter().map(|e| e.name.as_str()).collect();
            assert!(edge_names.contains(&"KNOWS"));
            assert!(edge_names.contains(&"WORKS_AT"));

            assert!(lpg.property_keys.contains(&"name".to_string()));
        }
        other => panic!("expected LPG schema, got {other:?}"),
    }
}

#[test]
fn test_admin_service_trait() {
    use grafeo_engine::AdminService;

    let db = grafeo_engine::GrafeoDB::new_in_memory();
    let n1 = db.create_node(&["Test"]);
    db.set_node_property(n1, "key", "value".into())
        .expect("set node property");

    // AdminService methods should work
    let info = AdminService::info(&db);
    assert_eq!(info.node_count, 1);

    let stats = AdminService::detailed_stats(&db);
    assert_eq!(stats.node_count, 1);

    let schema = AdminService::schema(&db);
    assert!(matches!(schema, grafeo_engine::SchemaInfo::Lpg(_)));

    let validation = AdminService::validate(&db);
    assert!(validation.errors.is_empty());
}

#[test]
fn test_query_with_parameters() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);

    // Test parameterized query
    let mut params = std::collections::HashMap::new();
    params.insert(
        "name".to_string(),
        grafeo_common::types::Value::from("Alix"),
    );

    let session = db.session();
    let result = session
        .execute_with_params("MATCH (n {name: $name}) RETURN n.name", params)
        .expect("execute parameterized query");
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_init_creates_database() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("new.grafeo");

    let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
    let info = db.info();
    assert_eq!(info.node_count, 0);
    assert_eq!(info.edge_count, 0);
    assert!(info.is_persistent);
}
