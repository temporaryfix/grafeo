//! Manual, local bulk-load measurement; never a competitive benchmark claim.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};

fn fields(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (PropertyKey::new(key), value))
            .collect::<BTreeMap<_, _>>()
            .into(),
    )
}

fn params(rows: Vec<Value>) -> HashMap<String, Value> {
    HashMap::from([("rows".into(), Value::List(rows.into()))])
}

#[test]
#[ignore = "manual wall-clock probe; run explicitly on a quiet host"]
fn unwind_10k_nodes_20k_edges() -> Result<(), Box<dyn std::error::Error>> {
    const NODES: usize = 10_000;
    const EDGES: usize = 20_000;
    const BATCH: usize = 1_000;
    for sample in 0..3 {
        let db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
        let session = db.session();
        for start in (0..NODES).step_by(BATCH) {
            let rows = (start..(start + BATCH).min(NODES))
                .map(|id| fields([("id", Value::from(format!("n{id}")))]))
                .collect();
            session
                .execute_with_params("UNWIND $rows AS e CREATE (:Node {id:e.id})", params(rows))?;
        }
        session.execute("CREATE INDEX probe_node_id FOR (n:Node) ON (n.id)")?;
        let batches: Vec<_> = (0..EDGES)
            .step_by(BATCH)
            .map(|start| {
                params(
                    (start..(start + BATCH).min(EDGES))
                        .map(|edge| {
                            fields([
                                ("s", Value::from(format!("n{}", edge % NODES))),
                                ("t", Value::from(format!("n{}", (edge + 1) % NODES))),
                            ])
                        })
                        .collect(),
                )
            })
            .collect();
        let started = Instant::now();
        for batch in batches {
            session.execute_with_params(
                "UNWIND $rows AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[:LINKS]->(t)",
                batch,
            )?;
        }
        let elapsed = started.elapsed();
        assert_eq!(db.node_count(), NODES);
        assert_eq!(db.edge_count(), EDGES);
        println!(
            "{}",
            serde_json::json!({
                "probe": "unwind_endpoint_load", "sample": sample,
                "nodes": NODES, "edges": EDGES, "batch_size": BATCH,
                "elapsed_ns": elapsed.as_nanos(), "language": "default-gql",
                "storage": "in-memory", "comparison": "none-local-measurement-only",
            })
        );
    }
    Ok(())
}
