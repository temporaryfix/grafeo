//! RDF→LPG receipt/row atomicity across real child-process failpoints.
//!
//! ```text
//! cargo test --locked -p grafeo-engine \
//!   --features "lpg,gql,triple-store,sparql,wal,grafeo-file,testing-crash-injection" \
//!   --test rdf_projection_crash -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "sparql",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};

use grafeo_core::graph::rdf::RdfLpgProjectionDefinition;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::wal::{WalRecord, WalRecovery};

const PERSON: &str = "http://ex.org/Person";

fn both_sync(path: &Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Both)
        .with_wal_durability(DurabilityMode::Sync);
    #[cfg(feature = "cdc")]
    let config = config.with_cdc();
    GrafeoDB::with_config(config).expect("open persistent Both database")
}

fn sidecar_wal_dir(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(".wal");
    PathBuf::from(path)
}

fn projection_child() {
    let path = std::env::var("RDF_PROJECTION_CRASH_PATH").expect("child path");
    let db = both_sync(Path::new(&path));
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    let crashed = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = db.rebuild_rdf_lpg_projection(id);
    }));
    assert!(crashed.is_err(), "named projection failpoint did not fire");
    std::mem::forget(db);
    std::process::abort();
}

fn projected_rows(db: &GrafeoDB) -> usize {
    db.session()
        .execute("MATCH (n:Person) RETURN n")
        .unwrap()
        .row_count()
}

#[cfg(feature = "cdc")]
fn assert_feed(db: &GrafeoDB, projected: bool) -> serde_json::Value {
    use grafeo_common::types::Value;
    use grafeo_core::graph::rdf::{
        RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
    };
    use grafeo_engine::cdc::ChangeKind;
    let page = db.session().changes_after(None, 3, 1024 * 1024).unwrap();
    assert_eq!(page.events.len(), 1 + usize::from(projected));
    assert!(page.events[0].entity_id.is_triple());
    assert_eq!(page.events[0].kind, ChangeKind::Create);
    if projected {
        let event = &page.events[1];
        assert!(event.entity_id.is_node());
        assert_eq!(event.kind, ChangeKind::Create);
        assert_eq!(event.labels.as_ref().unwrap(), &["Person"]);
        assert!(event.before.is_none());
        let props = event.after.as_ref().unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(
            props.get(RDF_LPG_PROJECTION_IRI_PROPERTY),
            Some(&Value::from("http://ex.org/alix"))
        );
        assert_eq!(
            props.get(RDF_LPG_PROJECTION_OWNER_PROPERTY),
            Some(&Value::from(
                RdfLpgProjectionDefinition::new(PERSON, "Person").owner_marker()
            ))
        );
    }
    serde_json::to_value(page).unwrap()
}

#[test]
fn receipt_and_rows_follow_the_same_commit_across_child_crashes() {
    if std::env::var("RDF_PROJECTION_CRASH_CHILD").as_deref() == Ok("1") {
        projection_child();
        return;
    }

    let cases = [
        ("projection:before_receipt", false),
        ("projection:after_receipt_before_commit", false),
        ("projection:after_commit_before_publication", true),
    ];
    let projection_id = RdfLpgProjectionDefinition::new(PERSON, "Person").id();

    for (site, committed) in cases {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("projection-crash.grafeo");
        let exe = std::env::current_exe().expect("current test executable");
        let status = std::process::Command::new(exe)
            .arg("receipt_and_rows_follow_the_same_commit_across_child_crashes")
            .arg("--exact")
            .env("RDF_PROJECTION_CRASH_CHILD", "1")
            .env("RDF_PROJECTION_CRASH_PATH", &path)
            .env("GRAFEO_CRASH_NAMED", site)
            .status()
            .expect("spawn projection failpoint child");
        assert!(!status.success(), "child must terminate at {site}");

        let recovered_records = WalRecovery::new(sidecar_wal_dir(&path))
            .unwrap()
            .recover()
            .expect("recover committed WAL groups");
        let has_receipt = recovered_records
            .iter()
            .any(|record| matches!(record, WalRecord::RdfLpgProjectionPublishedV3 { .. }));
        assert_eq!(
            has_receipt, committed,
            "receipt committed-only filtering disagrees at {site}"
        );

        let db = both_sync(&path);
        let definition = db
            .rdf_lpg_projection(projection_id)
            .expect("force-synced declaration survives child crash");
        assert_eq!(definition.generation(), u64::from(committed), "{site}");
        assert_eq!(definition.receipt().is_some(), committed, "{site}");
        assert_eq!(projected_rows(&db), usize::from(committed), "{site}");

        #[cfg(feature = "cdc")]
        assert_feed(&db, committed);

        // Exact replay/reconciliation remains idempotent whichever side of the
        // durable marker the process died on.
        assert_eq!(db.rebuild_rdf_lpg_projection(projection_id).unwrap(), 1);
        assert_eq!(projected_rows(&db), 1, "{site}");
        assert_eq!(
            db.rdf_lpg_projection(projection_id).unwrap().generation(),
            if committed { 2 } else { 1 },
            "{site}"
        );
        #[cfg(feature = "cdc")]
        let expected = assert_feed(&db, true);
        db.close().unwrap();
        for _ in 0..2 {
            let reopened = both_sync(&path);
            assert_eq!(projected_rows(&reopened), 1, "{site}");
            assert!(
                reopened
                    .rdf_lpg_projection(projection_id)
                    .unwrap()
                    .receipt()
                    .is_some()
            );
            #[cfg(feature = "cdc")]
            assert_eq!(assert_feed(&reopened, true), expected, "{site}");
            assert_eq!(
                reopened.rebuild_rdf_lpg_projection(projection_id).unwrap(),
                1
            );
            #[cfg(feature = "cdc")]
            assert_eq!(assert_feed(&reopened, true), expected, "{site}");
            reopened.close().unwrap();
        }
    }
}
