//! Public owned GROUP BY route: exact results, retained grants and process peak.
#![cfg(all(feature = "gql", feature = "lpg", feature = "spill"))]

use grafeo_common::memory::buffer::MemoryRegion;
use grafeo_common::types::Value;
use grafeo_common::utils::error::ErrorCode;
use grafeo_engine::{Config, GrafeoDB};

const ROWS: usize = 4096;
const BUDGET: usize = 3 << 20;
const PAYLOAD_BYTES: usize = 512;
// Same fixed headroom and isolated-process protocol as async_sort/memory_tests.
// Store setup is a separate baseline; neither envelope scales with input.
const PROCESS_HEADROOM: usize = 32 << 20;
const HEAP_QUERY: &str = "MATCH (n:AggregatePeak) RETURN n.bucket AS bucket, count(*) AS count, collect(n.payload) AS collected, collect(DISTINCT n.payload) AS unique_payloads";
const SCALAR_QUERY: &str =
    "MATCH (n:AggregatePeak) RETURN n.bucket AS bucket, sum(n.value) AS total, count(*) AS count";

fn payload(group: usize, variant: usize) -> String {
    let prefix = format!("group-{group:05}-kind-{variant}-");
    let mut text = prefix.clone();
    text.push_str(&"x".repeat(PAYLOAD_BYTES - prefix.len()));
    assert_eq!(text.len(), PAYLOAD_BYTES);
    text
}

fn seed_heap(database: &GrafeoDB, rows: usize, skew: bool) {
    for pass in 0..4 {
        for slot in 0..rows / 4 {
            let group = if skew {
                if slot % 4 == 0 { 0 } else { slot - slot / 4 }
            } else {
                slot
            };
            database.create_node_with_props(
                &["AggregatePeak"],
                [
                    ("bucket", Value::Int64(i64::try_from(group).unwrap())),
                    ("payload", Value::from(payload(group, pass % 2))),
                ],
            );
        }
    }
}

fn assert_heap_rows(result: &[Vec<Value>], rows: usize, skew: bool) {
    let groups = if skew { 1 + rows * 3 / 16 } else { rows / 4 };
    assert_eq!(result.len(), groups);
    let mut seen = vec![false; groups];
    let mut total = 0;
    for row in result {
        let [
            Value::Int64(group),
            Value::Int64(count),
            Value::List(collected),
            Value::List(unique),
        ] = row.as_slice()
        else {
            panic!("unexpected grouped output: {row:?}");
        };
        let group = usize::try_from(*group).unwrap();
        assert!(group < groups && !seen[group]);
        seen[group] = true;
        let expected_count = if skew && group == 0 { rows / 4 } else { 4 };
        assert_eq!(usize::try_from(*count).unwrap(), expected_count);
        assert_eq!(collected.len(), expected_count);
        assert_eq!(unique.len(), 2);
        let expected = [
            Value::from(payload(group, 0)),
            Value::from(payload(group, 1)),
        ];
        for value in &expected {
            assert_eq!(
                collected.iter().filter(|actual| *actual == value).count(),
                expected_count / 2
            );
            assert_eq!(unique.iter().filter(|actual| *actual == value).count(), 1);
        }
        total += expected_count;
    }
    assert!(seen.into_iter().all(|value| value));
    assert_eq!(total, rows);
}

fn assert_no_query_leaf(directory: &std::path::Path, database: &GrafeoDB) {
    let namespace = directory.join(format!("grafeo-store-{}", database.store_id()));
    for entry in std::fs::read_dir(namespace).unwrap() {
        assert!(
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("grafeo-query-")
        );
    }
}

#[test]
fn public_grouped_scalar_spill_succeeds_and_zero_quota_denies() {
    // Independent completing spill witness: the heap matrix may correctly
    // reject its eager output above 3 MiB. These pressure constants match the
    // predeclared scalar cost fixture, not a relaxation of that memory gate.
    const SCALAR_BUDGET: usize = 32 << 20;
    const HELD_PRESSURE: usize = 28 << 20;
    for denied in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let config = Config::in_memory()
            .with_memory_limit(SCALAR_BUDGET)
            .with_spill_path(directory.path())
            .with_max_query_spill_bytes(if denied { 0 } else { 64 << 20 });
        let database = GrafeoDB::with_config(config).unwrap();
        for pass in 0..4 {
            for group in 0..ROWS / 4 {
                database.create_node_with_props(
                    &["AggregatePeak"],
                    [
                        ("bucket", Value::Int64(i64::try_from(group).unwrap())),
                        ("value", Value::Int64(pass + 1)),
                    ],
                );
            }
        }
        let baseline = database.buffer_manager().allocated();
        let pressure = database
            .buffer_manager()
            .try_allocate(HELD_PRESSURE, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut session = database.session();
        // Active transactions bypass the reusable borrowed pull-plan cache.
        // Bare group/aggregate output variables keep HashAggregate at the root,
        // so owned execution decomposes it into the configured push operator.
        session.begin_transaction().unwrap();
        let outcome = session.execute(SCALAR_QUERY);
        if denied {
            let error = outcome.as_ref().unwrap_err();
            assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
            assert!(
                error.to_string().contains("spill disk quota exceeded"),
                "{error}"
            );
        } else {
            let result = outcome.as_ref().unwrap();
            assert_eq!(result.row_count(), ROWS / 4);
            let mut seen = vec![false; ROWS / 4];
            for row in result.rows() {
                let [Value::Int64(group), Value::Int64(10), Value::Int64(4)] = row.as_slice()
                else {
                    panic!("unexpected grouped scalar output: {row:?}");
                };
                let group = usize::try_from(*group).unwrap();
                assert!(group < seen.len() && !seen[group]);
                seen[group] = true;
            }
            assert!(seen.into_iter().all(|value| value));
            assert!(database.buffer_manager().allocated() > baseline + HELD_PRESSURE);
        }
        assert!(database.buffer_manager().allocated() <= SCALAR_BUDGET);
        drop(outcome);
        session.rollback().unwrap();
        drop(session);
        drop(pressure);
        assert_eq!(database.buffer_manager().allocated(), baseline);
        assert_no_query_leaf(directory.path(), &database);
    }
}

#[cfg(target_os = "linux")]
fn process_memory() -> (usize, usize) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let bytes = |name: &str| {
        let line = status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .unwrap();
        let mut fields = line.split_whitespace();
        let kib: usize = fields.next().unwrap().parse().unwrap();
        assert_eq!(fields.next(), Some("kB"));
        kib.checked_mul(1024).unwrap()
    };
    (bytes("VmRSS:"), bytes("VmHWM:"))
}

#[cfg(target_os = "macos")]
fn process_rss() -> usize {
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .env("LC_ALL", "C")
        .output()
        .unwrap();
    assert!(output.status.success());
    let kib: usize = std::str::from_utf8(&output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(kib > 0);
    kib.checked_mul(1024).unwrap()
}

#[test]
#[cfg(target_os = "linux")]
fn native_aggregate_process_peak_linux() {
    native_aggregate_process_peak("native_aggregate_process_peak_linux");
}

#[test]
#[cfg(target_os = "macos")]
fn native_aggregate_process_peak_macos() {
    native_aggregate_process_peak("native_aggregate_process_peak_macos");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn native_aggregate_process_peak(test_name: &str) {
    const CHILD_ROWS: &str = "GRAFEO_AGGREGATE_PEAK_CHILD_ROWS";
    const CHILD_MODE: &str = "GRAFEO_AGGREGATE_PEAK_CHILD_MODE";
    let Some(rows) = std::env::var_os(CHILD_ROWS) else {
        for rows in [4096, 8192, 16384] {
            for mode in ["balanced", "skew", "quota0"] {
                #[cfg(target_os = "linux")]
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                #[cfg(target_os = "macos")]
                let mut command = {
                    let mut command = std::process::Command::new("/usr/bin/time");
                    command.arg("-l").arg(std::env::current_exe().unwrap());
                    command.env("LC_ALL", "C");
                    command
                };
                let output = command
                    .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
                    .env(CHILD_ROWS, rows.to_string())
                    .env(CHILD_MODE, mode)
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    output.status.success(),
                    "rows={rows} mode={mode}: {}\n{stdout}\n{stderr}",
                    output.status
                );
                #[cfg(target_os = "linux")]
                assert!(
                    stdout.contains("aggregate process peak:"),
                    "{stdout}\n{stderr}"
                );
                #[cfg(target_os = "macos")]
                {
                    let mut samples = stdout.lines().filter_map(|line| {
                        line.split_once("aggregate process sample: ")
                            .map(|(_, value)| value)
                    });
                    let sample = samples.next().expect("missing child memory sample");
                    assert!(samples.next().is_none());
                    let counter = |name: &str| -> usize {
                        let mut values = sample
                            .split_whitespace()
                            .filter_map(|field| field.split_once('='))
                            .filter_map(|(key, value)| (key == name).then_some(value));
                        let value = values.next().unwrap().parse().unwrap();
                        assert!(values.next().is_none());
                        value
                    };
                    assert_eq!(counter("rows"), rows);
                    assert_eq!(counter("budget"), BUDGET);
                    assert_eq!(counter("released_query_bytes"), 0);
                    assert!(
                        sample
                            .split_whitespace()
                            .any(|field| field == format!("mode={mode}"))
                    );
                    let ceiling = counter("runtime_rss")
                        .max(counter("store_rss"))
                        .checked_add(PROCESS_HEADROOM)
                        .unwrap();
                    assert_eq!(counter("ceiling"), ceiling);
                    let mut maxima = stderr
                        .lines()
                        .filter_map(|line| line.trim().strip_suffix("maximum resident set size"));
                    let peak: usize = maxima
                        .next()
                        .expect("missing time-l max RSS")
                        .trim()
                        .parse()
                        .unwrap();
                    assert!(maxima.next().is_none());
                    assert!(
                        peak > 0 && peak <= ceiling,
                        "peak={peak} ceiling={ceiling}\n{stdout}\n{stderr}"
                    );
                    println!(
                        "aggregate process peak: platform=macos rows={rows} mode={mode} peak={peak} ceiling={ceiling}"
                    );
                }
                print!("{stdout}");
            }
        }
        return;
    };
    let rows: usize = rows.to_str().unwrap().parse().unwrap();
    assert!([4096, 8192, 16384].contains(&rows));
    let mode = std::env::var(CHILD_MODE).unwrap();
    assert!(["balanced", "skew", "quota0"].contains(&mode.as_str()));
    let skew = mode == "skew";
    #[cfg(target_os = "linux")]
    let runtime = process_memory();
    #[cfg(target_os = "macos")]
    let runtime = process_rss();
    let directory = tempfile::tempdir().unwrap();
    let mut config = Config::in_memory()
        .with_memory_limit(BUDGET)
        .with_spill_path(directory.path());
    if mode == "quota0" {
        config = config.with_max_query_spill_bytes(0);
    }
    let database = GrafeoDB::with_config(config).unwrap();
    seed_heap(&database, rows, skew);
    let baseline = database.buffer_manager().allocated();
    #[cfg(target_os = "linux")]
    let store = process_memory();
    #[cfg(target_os = "macos")]
    let store = process_rss();
    let mut session = database.session();
    session.begin_transaction().unwrap();
    let outcome = session.execute(HEAP_QUERY);
    let denied = outcome.is_err();
    match &outcome {
        Ok(result) => {
            assert_ne!(
                mode, "quota0",
                "zero quota must reject this spilling workload"
            );
            assert_eq!(
                result.columns,
                ["bucket", "count", "collected", "unique_payloads"]
            );
            assert_heap_rows(result.rows(), rows, skew);
            assert!(
                database.buffer_manager().allocated() > baseline,
                "retained result must retain its authority"
            );
        }
        Err(error) => assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}"),
    }
    let retained = database.buffer_manager().allocated();
    assert!(retained <= BUDGET);
    drop(outcome);
    session.rollback().unwrap();
    drop(session);
    assert_eq!(database.buffer_manager().allocated(), baseline);
    assert_no_query_leaf(directory.path(), &database);
    let follow_up = database.execute("RETURN 7 AS value").unwrap();
    assert_eq!(follow_up.rows(), &[vec![Value::Int64(7)]]);
    drop(follow_up);
    assert_eq!(database.buffer_manager().allocated(), baseline);
    assert_no_query_leaf(directory.path(), &database);
    #[cfg(target_os = "linux")]
    {
        let final_memory = process_memory();
        let ceiling = runtime
            .1
            .max(store.1)
            .checked_add(PROCESS_HEADROOM)
            .unwrap();
        println!(
            "aggregate process peak: platform=linux rows={rows} mode={mode} denied={denied} budget={BUDGET} runtime_rss={} runtime_hwm={} store_rss={} store_hwm={} final_rss={} peak={} ceiling={ceiling} retained_query_bytes={retained} released_query_bytes=0",
            runtime.0, runtime.1, store.0, store.1, final_memory.0, final_memory.1
        );
        assert!(final_memory.1 > 0 && final_memory.1 <= ceiling);
    }
    #[cfg(target_os = "macos")]
    {
        let ceiling = runtime.max(store).checked_add(PROCESS_HEADROOM).unwrap();
        println!(
            "aggregate process sample: rows={rows} mode={mode} denied={denied} budget={BUDGET} runtime_rss={runtime} store_rss={store} final_rss={} ceiling={ceiling} retained_query_bytes={retained} released_query_bytes=0",
            process_rss()
        );
    }
}
