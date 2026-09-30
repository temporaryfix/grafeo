//! Atomic and crash-durable standalone catalog DDL.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
#[cfg(feature = "testing-crash-injection")]
use std::sync::Arc;

#[cfg(feature = "testing-crash-injection")]
use grafeo_common::testing::wal_failure::{
    disable_catalog_batch_ack_failure, disable_catalog_batch_log_failure,
    enable_catalog_batch_ack_failure_once, enable_catalog_batch_log_failure_once,
};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn persistent(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent LPG database")
}

fn sidecar(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(".wal");
    PathBuf::from(value)
}

fn copy_live_database(src: &Path, dst: &Path) {
    std::fs::copy(src, dst).expect("copy live container");
    let source_wal = sidecar(src);
    if source_wal.exists() {
        let target_wal = sidecar(dst);
        std::fs::create_dir_all(&target_wal).expect("create copied WAL directory");
        for entry in std::fs::read_dir(source_wal).expect("read live WAL directory") {
            let entry = entry.expect("WAL directory entry");
            if entry.path().is_file() {
                std::fs::copy(entry.path(), target_wal.join(entry.file_name()))
                    .expect("copy WAL segment");
            }
        }
    }
}

fn constraint_names(db: &GrafeoDB) -> Vec<String> {
    db.session()
        .execute("SHOW CONSTRAINTS")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| row[0].as_str().map(ToString::to_string))
        .collect()
}

#[cfg(feature = "testing-crash-injection")]
fn schema_names(db: &GrafeoDB) -> Vec<String> {
    let mut names: Vec<String> = db
        .session()
        .execute("SHOW SCHEMAS")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| row[0].as_str().map(ToString::to_string))
        .collect();
    names.sort_unstable();
    names
}

fn index_names(db: &GrafeoDB) -> Vec<String> {
    let mut names: Vec<String> = db
        .session()
        .execute("SHOW INDEXES")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| row[0].as_str().map(ToString::to_string))
        .collect();
    names.sort_unstable();
    names
}

#[cfg(feature = "testing-crash-injection")]
fn checkpointed_catalog_anchor(db: &GrafeoDB) -> (Vec<String>, Vec<String>) {
    db.session().execute("CREATE SCHEMA anchor").unwrap();
    db.wal_checkpoint().unwrap();
    let schemas = schema_names(db);
    let mut graphs = db.list_graphs();
    graphs.sort_unstable();
    assert!(schemas.contains(&"anchor".to_string()));
    assert!(graphs.contains(&"anchor/__default__".to_string()));
    (schemas, graphs)
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn create_schema_preappend_failure_keeps_live_and_recovered_preimage() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-preappend.grafeo");
    let db = persistent(&path);
    let (schemas_before, graphs_before) = checkpointed_catalog_anchor(&db);
    let anchor_before = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("anchor/__default__")
        .unwrap();
    let session = db.session();

    let before_epoch = db.current_epoch();
    enable_catalog_batch_log_failure_once();
    let result = session.execute("CREATE SCHEMA ghost");
    disable_catalog_batch_log_failure();
    assert_eq!(db.current_epoch(), before_epoch);

    assert!(result.is_err(), "the injected WAL failure must surface");
    assert!(db.is_durability_poisoned());
    db.set_current_schema(Some("anchor"))
        .expect("the existing catalog preimage must remain live");
    db.set_current_schema(None).unwrap();
    assert!(
        db.set_current_schema(Some("ghost")).is_err(),
        "the rejected catalog post-image must never become live"
    );
    let mut live_graphs = db.list_graphs();
    live_graphs.sort_unstable();
    assert_eq!(live_graphs, graphs_before);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("ghost/__default__")
            .is_none()
    );
    let anchor_after = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("anchor/__default__")
        .unwrap();
    assert!(
        Arc::ptr_eq(&anchor_before, &anchor_after),
        "failed preparation must preserve existing graph incarnations"
    );

    drop(session);
    assert!(
        db.close().is_err(),
        "poisoned close must not checkpoint tentative state"
    );
    drop(db);
    let reopened = persistent(&path);
    assert_eq!(schema_names(&reopened), schemas_before);
    let mut recovered_graphs = reopened.list_graphs();
    recovered_graphs.sort_unstable();
    assert_eq!(recovered_graphs, graphs_before);
    assert!(reopened.set_current_schema(Some("ghost")).is_err());
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .graph("ghost/__default__")
            .is_none()
    );
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn create_schema_lost_ack_keeps_live_preimage_but_reopens_durable_postimage() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-lost-ack.grafeo");
    let db = persistent(&path);
    let (schemas_before, graphs_before) = checkpointed_catalog_anchor(&db);
    let anchor_before = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("anchor/__default__")
        .unwrap();
    let session = db.session();

    let before_epoch = db.current_epoch();
    enable_catalog_batch_ack_failure_once();
    let result = session.execute("CREATE SCHEMA durable_after_reopen");
    disable_catalog_batch_ack_failure();
    assert_eq!(db.current_epoch(), before_epoch);

    assert!(result.is_err(), "the lost acknowledgement must surface");
    assert!(db.is_durability_poisoned());
    db.set_current_schema(Some("anchor"))
        .expect("the existing catalog preimage must remain live");
    db.set_current_schema(None).unwrap();
    assert!(
        db.set_current_schema(Some("durable_after_reopen")).is_err(),
        "an ambiguous write must leave the live preimage fail-stopped"
    );
    let mut live_graphs = db.list_graphs();
    live_graphs.sort_unstable();
    assert_eq!(live_graphs, graphs_before);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("durable_after_reopen/__default__")
            .is_none()
    );
    let anchor_after = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("anchor/__default__")
        .unwrap();
    assert!(Arc::ptr_eq(&anchor_before, &anchor_after));

    drop(session);
    assert!(
        db.close().is_err(),
        "poisoned close must not checkpoint an ambiguous live state"
    );
    drop(db);
    let reopened = persistent(&path);
    let mut schemas_after = schemas_before;
    schemas_after.push("durable_after_reopen".to_string());
    schemas_after.sort_unstable();
    assert_eq!(schema_names(&reopened), schemas_after);
    let mut graphs_after = graphs_before;
    graphs_after.push("durable_after_reopen/__default__".to_string());
    graphs_after.sort_unstable();
    let mut recovered_graphs = reopened.list_graphs();
    recovered_graphs.sort_unstable();
    assert_eq!(recovered_graphs, graphs_after);
    reopened
        .set_current_schema(Some("durable_after_reopen"))
        .expect("recovery must install the durable catalog post-image");
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .graph("durable_after_reopen/__default__")
            .is_some()
    );
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn drop_schema_preappend_failure_preserves_session_context_and_graph_incarnation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-drop-preappend.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session.execute("CREATE SCHEMA retained").unwrap();
    db.wal_checkpoint().unwrap();
    session.execute("SESSION SET SCHEMA retained").unwrap();
    let graph_before = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("retained/__default__")
        .unwrap();

    let before_epoch = db.current_epoch();
    enable_catalog_batch_log_failure_once();
    let result = session.execute("DROP SCHEMA retained");
    disable_catalog_batch_log_failure();
    assert_eq!(db.current_epoch(), before_epoch);

    assert!(result.is_err());
    assert!(db.is_durability_poisoned());
    assert_eq!(session.current_schema().as_deref(), Some("retained"));
    db.set_current_schema(Some("retained"))
        .expect("failed drop must preserve the live namespace");
    db.set_current_schema(None).unwrap();
    let graph_after = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("retained/__default__")
        .unwrap();
    assert!(Arc::ptr_eq(&graph_before, &graph_after));

    drop(session);
    assert!(db.close().is_err());
    drop(db);
    let reopened = persistent(&path);
    reopened
        .set_current_schema(Some("retained"))
        .expect("pre-append failure must recover the schema preimage");
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .graph("retained/__default__")
            .is_some()
    );
}

#[test]
fn standalone_catalog_publication_preserves_transactional_indexes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-preserves-index.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute("CREATE INDEX idx_person_name FOR (n:Person) ON (n.name)")
        .unwrap();
    assert_eq!(index_names(&db), vec!["idx_person_name"]);
    assert!(db.has_property_index("name"));

    session.execute("CREATE SCHEMA metadata").unwrap();
    session
        .execute("CREATE NODE TYPE PersonRecord (name STRING)")
        .unwrap();

    assert_eq!(index_names(&db), vec!["idx_person_name"]);
    assert!(
        db.has_property_index("name"),
        "detached catalog publication must not replace physical indexes"
    );
}

#[test]
fn in_memory_catalog_publication_preserves_raw_named_graph_capability() {
    let db = GrafeoDB::new_in_memory();
    grafeo_engine::database::testing::root_lpg_store(&db)
        .create_graph("existing")
        .unwrap();
    let existing = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("existing")
        .unwrap();
    assert!(existing.create_node(&["BeforeDdl"]).is_valid());

    db.session().execute("CREATE SCHEMA Added").unwrap();

    assert!(
        existing.create_node(&["AfterDdl"]).is_valid(),
        "detached preparation must not seal a shared in-memory graph Arc"
    );
    let added = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("Added/__default__")
        .unwrap();
    assert!(
        added.create_node(&["RawNewPartition"]).is_valid(),
        "publishing into an unsealed in-memory root must preserve raw mutation semantics"
    );
}

#[test]
fn failed_in_memory_catalog_preparation_preserves_raw_named_graph_capability() {
    let db = GrafeoDB::new_in_memory();
    grafeo_engine::database::testing::root_lpg_store(&db)
        .create_graph("existing")
        .unwrap();
    let existing = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("existing")
        .unwrap();
    assert!(existing.create_node(&["BeforeRejectedDdl"]).is_valid());

    assert!(db.session().execute("DROP SCHEMA missing").is_err());

    assert!(
        existing.create_node(&["AfterRejectedDdl"]).is_valid(),
        "a rejected detached statement must leave every live capability unchanged"
    );
}

#[test]
fn named_constraint_is_enforced_droppable_and_crash_durable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog.grafeo");
    let crash_copy = directory.path().join("catalog-crash.grafeo");
    let db = persistent(&path);
    db.session()
        .execute("INSERT (:Person {email: 'first@example.com'})")
        .unwrap();
    db.wal_checkpoint().unwrap();

    let session = db.session();
    session
        .execute("CREATE CONSTRAINT unique_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    assert_eq!(constraint_names(&db), vec!["unique_email"]);
    assert!(
        session
            .execute("INSERT (:Person {email: 'first@example.com'})")
            .is_err(),
        "the published constraint must be enforced"
    );

    // Sync durability requires the successful DDL call itself to be enough;
    // deliberately do not call wal.sync() before taking the crash image.
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&crash_copy);
    assert_eq!(constraint_names(&recovered), vec!["unique_email"]);
    assert!(
        recovered
            .session()
            .execute("INSERT (:Person {email: 'first@example.com'})")
            .is_err(),
        "WAL replay must restore enforcement as well as SHOW metadata"
    );
    recovered
        .session()
        .execute("DROP CONSTRAINT unique_email")
        .unwrap();
    assert!(constraint_names(&recovered).is_empty());
    recovered
        .session()
        .execute("INSERT (:Person {email: 'first@example.com'})")
        .expect("dropping a constraint must remove enforcement");
}

#[test]
fn failed_multi_alter_rolls_back_its_live_and_durable_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("alter.grafeo");
    let crash_copy = directory.path().join("alter-crash.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Device (serial STRING)")
        .unwrap();
    db.wal_checkpoint().unwrap();

    assert!(
        session
            .execute("ALTER NODE TYPE Device ADD location STRING DROP missing")
            .is_err()
    );
    session
        .execute("ALTER NODE TYPE Device ADD location STRING")
        .expect("the successful first alteration must have been rolled back");
    session
        .execute("ALTER NODE TYPE Device DROP location")
        .unwrap();

    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&crash_copy);
    recovered
        .session()
        .execute("ALTER NODE TYPE Device ADD location STRING")
        .expect("recovery must not replay a prefix from the failed statement");
}

#[test]
fn constraint_creation_rejects_preexisting_violations_without_catalog_leaks() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'same'})").unwrap();
    session.execute("INSERT (:Person {name: 'same'})").unwrap();

    assert!(
        session
            .execute("CREATE CONSTRAINT uniq_name FOR (n:Person) ON (n.name) UNIQUE")
            .is_err()
    );
    assert!(constraint_names(&db).is_empty());
    session
        .execute("DROP CONSTRAINT IF EXISTS uniq_name")
        .expect("a rejected create must not leave a ghost definition");
}

#[test]
fn composite_unique_uses_tuples_and_sees_same_transaction_writes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT pair_key FOR (n:Pair) ON (n.a, n.b) UNIQUE")
        .unwrap();

    session.execute("INSERT (:Pair {a: 1, b: 1})").unwrap();
    session
        .execute("INSERT (:Pair {a: 1, b: 2})")
        .expect("composite uniqueness compares the tuple, not either scalar");
    assert!(
        session.execute("INSERT (:Pair {a: 1, b: 1})").is_err(),
        "an equal composite tuple must be rejected"
    );
    session.execute("INSERT (:Pair {a: 1, b: null})").unwrap();
    session
        .execute("INSERT (:Pair {a: 1, b: null})")
        .expect("UNIQUE skips tuples containing NULL");

    let mut transaction = db.session();
    transaction.begin_transaction().unwrap();
    transaction.execute("INSERT (:Pair {a: 9, b: 9})").unwrap();
    assert!(
        transaction.execute("INSERT (:Pair {a: 9, b: 9})").is_err(),
        "a second create must see the first create's tx-local tuple"
    );
    let _ = transaction.rollback();
}

#[test]
fn required_constraint_checks_every_complete_post_image() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT email_required FOR (n:Person) ON (n.email) NOT NULL")
        .unwrap();

    assert!(session.execute("INSERT (:Person {id: 'missing'})").is_err());
    assert!(
        session
            .execute("INSERT (:Person {id: 'explicit', email: null})")
            .is_err()
    );
    session
        .execute("INSERT (:Person {id: 'kept', email: 'kept@example.com'})")
        .unwrap();
    assert!(
        session
            .execute("MATCH (n:Person {id: 'kept'}) SET n.email = null")
            .is_err()
    );
    assert!(
        session
            .execute("MATCH (n:Person {id: 'kept'}) REMOVE n.email")
            .is_err()
    );
    assert!(
        session
            .execute("MATCH (n:Person {id: 'kept'}) SET n += {email: null}")
            .is_err()
    );
    assert!(
        session
            .execute("MATCH (n:Person {id: 'kept'}) SET n = {id: 'kept'}")
            .is_err()
    );
    assert!(
        session
            .execute("MERGE (n:Person {id: 'merge-missing'})")
            .is_err(),
        "MERGE create must validate completeness"
    );
    assert!(
        session
            .execute("MERGE (n:Person {id: 'kept'}) ON MATCH SET n.email = null")
            .is_err(),
        "MERGE ON MATCH must validate the complete post-image"
    );

    let email = session
        .execute("MATCH (n:Person {id: 'kept'}) RETURN n.email")
        .unwrap();
    assert_eq!(email.rows()[0][0].as_str(), Some("kept@example.com"));

    session
        .execute("CREATE NODE TYPE Defaulted (code STRING NOT NULL DEFAULT 'fallback')")
        .unwrap();
    session
        .execute("INSERT (:Defaulted {id: 'defaulted'})")
        .expect("CREATE injects the declared default");
    assert!(
        session
            .execute("MATCH (n:Defaulted {id: 'defaulted'}) REMOVE n.code")
            .is_err(),
        "a default is creation-time input, not permission to leave NOT NULL absent"
    );
    assert!(
        session
            .execute("MATCH (n:Defaulted {id: 'defaulted'}) SET n = {id: 'defaulted'}")
            .is_err(),
        "map replacement cannot remove a defaulted NOT NULL property"
    );
}

#[test]
fn add_label_cannot_bypass_constraints_and_same_value_set_excludes_self() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT uniq_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    session
        .execute("CREATE CONSTRAINT required_email FOR (n:Person) ON (n.email) NOT NULL")
        .unwrap();
    session
        .execute("INSERT (:Person {id: 'owner', email: 'same@example.com'})")
        .unwrap();
    session
        .execute("MATCH (n:Person {id: 'owner'}) SET n.email = 'same@example.com'")
        .expect("a node must be excluded from its own uniqueness lookup");

    session.execute("INSERT (:Other {id: 'missing'})").unwrap();
    assert!(
        session
            .execute("MATCH (n:Other {id: 'missing'}) SET n:Person")
            .is_err(),
        "adding a constrained label must validate existing properties"
    );
    session
        .execute("INSERT (:Other {id: 'duplicate', email: 'same@example.com'})")
        .unwrap();
    assert!(
        session
            .execute("MATCH (n:Other {id: 'duplicate'}) SET n:Person")
            .is_err(),
        "adding a label must enforce its UNIQUE tuple"
    );
}

#[test]
fn edge_required_properties_survive_set_remove_and_map_replace() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE EDGE TYPE NeedsCode (code STRING NOT NULL)")
        .unwrap();
    session.execute("INSERT (:End {id: 1})").unwrap();
    session.execute("INSERT (:End {id: 2})").unwrap();
    session
        .execute(
            "MATCH (a:End {id: 1}), (b:End {id: 2}) INSERT (a)-[:NeedsCode {code: 'kept'}]->(b)",
        )
        .unwrap();

    assert!(
        session
            .execute("MATCH ()-[e:NeedsCode]->() SET e.code = null")
            .is_err()
    );
    assert!(
        session
            .execute("MATCH ()-[e:NeedsCode]->() REMOVE e.code")
            .is_err()
    );
    assert!(
        session
            .execute("MATCH ()-[e:NeedsCode]->() SET e += {code: null}")
            .is_err()
    );
    assert!(
        session
            .execute("MATCH ()-[e:NeedsCode]->() SET e = {other: 'value'}")
            .is_err()
    );
    let code = session
        .execute("MATCH ()-[e:NeedsCode]->() RETURN e.code")
        .unwrap();
    assert_eq!(code.rows()[0][0].as_str(), Some("kept"));
}

#[test]
fn concurrent_duplicate_commits_have_exactly_one_winner() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE CONSTRAINT uniq_token FOR (n:Token) ON (n.value) UNIQUE")
        .unwrap();

    let mut left = db.session();
    let mut right = db.session();
    left.begin_transaction().unwrap();
    right.begin_transaction().unwrap();
    left.execute("INSERT (:Token {value: 'contended'})")
        .unwrap();
    right
        .execute("INSERT (:Token {value: 'contended'})")
        .expect("the competing write remains invisible until publication");

    assert!(left.commit().is_ok());
    assert!(
        right.commit().is_err(),
        "publication-time revalidation must reject the second duplicate"
    );
    let count = db
        .session()
        .execute("MATCH (n:Token {value: 'contended'}) RETURN count(n)")
        .unwrap();
    assert_eq!(count.rows()[0][0].as_int64(), Some(1));
}

#[test]
fn anonymous_names_are_property_sensitive_and_required_null_reopens() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("anonymous.grafeo");
    let crash_copy = directory.path().join("anonymous-crash.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    session
        .execute("CREATE CONSTRAINT FOR (n:Person) ON (n.username) UNIQUE")
        .expect("anonymous constraints on distinct properties need distinct names");
    session
        .execute("CREATE CONSTRAINT required_code FOR (n:Item) ON (n.code) NOT NULL")
        .unwrap();
    let names = constraint_names(&db);
    assert_eq!(names.len(), 3);
    assert_ne!(names[0], names[1]);

    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);
    let recovered = persistent(&crash_copy);
    assert_eq!(constraint_names(&recovered).len(), 3);
    assert!(
        recovered
            .session()
            .execute("INSERT (:Item {code: null})")
            .is_err(),
        "replayed NOT NULL metadata must reject explicit NULL"
    );
}

#[test]
fn complete_catalog_post_image_preserves_full_ddl_semantics_after_crash() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-v3.grafeo");
    let crash_copy = directory.path().join("catalog-v3-crash.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Entity (id STRING)")
        .unwrap();
    db.wal_checkpoint().unwrap();

    session
        .execute(
            "CREATE NODE TYPE Person EXTENDS Entity (name STRING NOT NULL DEFAULT 'anonymous')",
        )
        .unwrap();
    session
        .execute("CREATE NODE TYPE Company (name STRING NOT NULL)")
        .unwrap();
    session
        .execute(
            "CREATE EDGE TYPE WorksAt CONNECTING (Person) TO (Company) \
             (since INTEGER NOT NULL DEFAULT 2020)",
        )
        .unwrap();
    session
        .execute(
            "CREATE GRAPH TYPE Employment \
             (NODE TYPE Person, NODE TYPE Company, EDGE TYPE WorksAt)",
        )
        .unwrap();
    session
        .execute("ALTER GRAPH TYPE Employment DROP EDGE TYPE WorksAt")
        .unwrap();
    session
        .execute("ALTER GRAPH TYPE Employment ADD EDGE TYPE WorksAt")
        .unwrap();
    session
        .execute(
            "CREATE PROCEDURE list_people() RETURNS (name STRING) AS { \
             MATCH (c:Company) RETURN c.name AS name }",
        )
        .unwrap();
    session
        .execute(
            "CREATE OR REPLACE PROCEDURE list_people() RETURNS (name STRING) AS { \
             MATCH (p:Person) RETURN p.name AS name }",
        )
        .unwrap();
    session.execute("CREATE SCHEMA analytics").unwrap();

    // Every successful V3 catalog publication is force-synced. Take a live
    // process image without close(), explicit WAL sync, or another checkpoint.
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&crash_copy);
    let session = recovered.session();
    let node_types = session.execute("SHOW NODE TYPES").unwrap();
    let person = node_types
        .rows()
        .iter()
        .find(|row| row[0].as_str() == Some("Person"))
        .expect("Person type recovered");
    assert_eq!(person[3].as_str(), Some("Entity"));

    let edge_types = session.execute("SHOW EDGE TYPES").unwrap();
    let works_at = edge_types
        .rows()
        .iter()
        .find(|row| row[0].as_str() == Some("WorksAt"))
        .expect("WorksAt type recovered");
    assert_eq!(works_at[2].as_str(), Some("Person"));
    assert_eq!(works_at[3].as_str(), Some("Company"));

    let graph_types = session.execute("SHOW GRAPH TYPES").unwrap();
    let employment = graph_types
        .rows()
        .iter()
        .find(|row| row[0].as_str() == Some("Employment"))
        .expect("Employment graph type recovered");
    assert_eq!(employment[2].as_str(), Some("Person, Company"));
    assert_eq!(employment[3].as_str(), Some("WorksAt"));

    session.execute("INSERT (:Person {id: 'p1'})").unwrap();
    session.execute("INSERT (:Company {name: 'Acme'})").unwrap();
    let defaulted = session
        .execute("MATCH (p:Person {id: 'p1'}) RETURN p.name")
        .unwrap();
    assert_eq!(defaulted.rows()[0][0].as_str(), Some("anonymous"));
    session
        .execute(
            "MATCH (p:Person {id: 'p1'}), (c:Company {name: 'Acme'}) \
             INSERT (p)-[:WorksAt {since: 2020}]->(c)",
        )
        .unwrap();
    let since = session
        .execute("MATCH ()-[e:WorksAt]->() RETURN e.since")
        .unwrap();
    assert_eq!(since.rows()[0][0].as_int64(), Some(2020));
    assert!(
        session
            .execute(
                "MATCH (p:Person {id: 'p1'}), (c:Company {name: 'Acme'}) \
                 INSERT (c)-[:WorksAt {since: 2026}]->(p)",
            )
            .is_err(),
        "replayed CONNECTING endpoints must still be enforced"
    );

    #[cfg(feature = "algos")]
    {
        let called = session.execute("CALL list_people()").unwrap();
        assert_eq!(called.rows()[0][0].as_str(), Some("anonymous"));
    }
    session
        .execute("DROP PROCEDURE list_people")
        .expect("the replaced stored procedure must exist after recovery");
    let schemas = session.execute("SHOW SCHEMAS").unwrap();
    assert!(
        schemas
            .rows()
            .iter()
            .any(|row| row[0].as_str() == Some("analytics"))
    );
    session.execute("SESSION SET SCHEMA analytics").unwrap();
    session
        .execute("INSERT (:RecoveredDefaultGraph {ok: true})")
        .expect("the schema default graph partition must be recreated from V3 lifecycle state");
}

#[test]
fn complete_catalog_post_image_replays_removals_over_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-v3-drop.grafeo");
    let crash_copy = directory.path().join("catalog-v3-drop-crash.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Obsolete (value STRING)")
        .unwrap();
    session.execute("CREATE SCHEMA temporary").unwrap();
    db.wal_checkpoint().unwrap();

    session.execute("DROP NODE TYPE Obsolete").unwrap();
    session.execute("DROP SCHEMA temporary").unwrap();
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&crash_copy);
    let session = recovered.session();
    assert!(
        session
            .execute("SHOW NODE TYPES")
            .unwrap()
            .rows()
            .iter()
            .all(|row| row[0].as_str() != Some("Obsolete"))
    );
    assert!(
        session
            .execute("SHOW SCHEMAS")
            .unwrap()
            .rows()
            .iter()
            .all(|row| row[0].as_str() != Some("temporary"))
    );
    assert!(session.execute("SESSION SET SCHEMA temporary").is_err());
}
