//! Native process and retained-grant controls for the public async sort route.

use super::*;
use crate::query::executor::ExecutionOptions;
use crate::{Config, GrafeoDB};
use grafeo_common::types::Value;
use std::collections::HashMap;

#[cfg(target_os = "linux")]
fn process_memory() -> (usize, usize) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let bytes = |name: &str| {
        let line = status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .unwrap_or_else(|| panic!("missing Linux process counter {name}"));
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
    assert!(output.status.success(), "process RSS measurement failed");
    let text = std::str::from_utf8(&output.stdout).unwrap();
    let kib: usize = text.trim().parse().expect("one process RSS value in KiB");
    assert!(kib > 0, "missing process RSS");
    kib.checked_mul(1024).unwrap()
}

// Follow rdf_aggregate_process_peak_linux's isolated-process protocol. Linux
// process high-water includes allocations outside grants and allocator reuse;
// it cannot prove ownership. The independently retained query account below
// supplies the grant high-water and exact terminal-release assertions.
#[test]
#[cfg(target_os = "linux")]
fn async_sort_process_peak_linux() {
    async_sort_process_peak(
        "query::executor::async_sort::memory_tests::async_sort_process_peak_linux",
    );
}

#[test]
#[cfg(target_os = "macos")]
fn async_sort_process_peak_macos() {
    async_sort_process_peak(
        "query::executor::async_sort::memory_tests::async_sort_process_peak_macos",
    );
}

fn async_sort_process_peak(test_name: &str) {
    const CHILD_SCALE: &str = "GRAFEO_ASYNC_SORT_PEAK_CHILD_SCALE";
    const CHILD_DENIED: &str = "GRAFEO_ASYNC_SORT_PEAK_CHILD_DENIED";
    const ROWS: usize = 4096;
    const BUDGET: usize = 3 << 20;
    const KEY_BYTES: usize = 128;
    // Predeclared fixed process envelope, matching the existing RDF control's
    // 32 MiB runtime/planner/allocator headroom. Neither this headroom nor the
    // query budget grows with input. Store setup is measured separately. The
    // 3 MiB/128-byte seed is the already-qualified completing async cost fixture;
    // N/2N/4N exercise more runs and eager output under that same query budget.
    const PROCESS_HEADROOM: usize = 32 << 20;
    const QUERY: &str = "MATCH (n:AsyncPeak) RETURN n.value AS value ORDER BY n.key";

    let Some(scale) = std::env::var_os(CHILD_SCALE) else {
        for scale in [1, 2, 4] {
            for denied in [false, true] {
                #[cfg(target_os = "linux")]
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                #[cfg(target_os = "macos")]
                let mut command = {
                    // Execute the test child directly, without a shell. time's
                    // maximum resident set size is bytes on macOS; its separate
                    // peak-memory-footprint counter is not this RSS measure.
                    let mut command = std::process::Command::new("/usr/bin/time");
                    command.arg("-l").arg(std::env::current_exe().unwrap());
                    command.env("LC_ALL", "C");
                    command
                };
                let output = command
                    .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
                    .env(CHILD_SCALE, scale.to_string())
                    .env(CHILD_DENIED, if denied { "1" } else { "0" })
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    output.status.success(),
                    "async resource child scale={scale}, denied={denied}: {}\n{stdout}\n{stderr}",
                    output.status
                );
                #[cfg(target_os = "linux")]
                assert!(
                    stdout.contains("async sort process peak:"),
                    "{stdout}\n{stderr}"
                );
                #[cfg(target_os = "macos")]
                {
                    let mut samples = stdout.lines().filter_map(|line| {
                        line.split_once("async sort process sample: platform=macos ")
                            .map(|(_, sample)| sample)
                    });
                    let sample = samples.next().expect("missing child memory sample");
                    assert!(samples.next().is_none(), "duplicate child memory sample");
                    let counter = |name: &str| {
                        let mut matches = sample
                            .split_whitespace()
                            .filter_map(|field| field.split_once('='))
                            .filter_map(|(key, value)| (key == name).then_some(value));
                        let value: usize = matches
                            .next()
                            .unwrap_or_else(|| panic!("missing child counter {name}"))
                            .parse()
                            .unwrap();
                        assert!(matches.next().is_none(), "duplicate child counter {name}");
                        value
                    };
                    assert_eq!(counter("rows"), ROWS * scale);
                    assert_eq!(counter("budget"), BUDGET);
                    assert_eq!(counter("released_query_bytes"), 0);
                    assert!(sample.split_whitespace().any(|field| {
                        field
                            == if denied {
                                "denied=true"
                            } else {
                                "denied=false"
                            }
                    }));
                    let runtime_rss = counter("runtime_rss");
                    let store_rss = counter("store_rss");
                    assert!(runtime_rss > 0 && store_rss > 0);
                    let ceiling = runtime_rss
                        .max(store_rss)
                        .checked_add(PROCESS_HEADROOM)
                        .unwrap();
                    assert_eq!(counter("ceiling"), ceiling);
                    let mut maxima = stderr
                        .lines()
                        .filter_map(|line| line.trim().strip_suffix("maximum resident set size"));
                    let process_peak: usize = maxima
                        .next()
                        .expect("missing macOS maximum resident set size")
                        .trim()
                        .parse()
                        .expect("macOS maximum resident set size must be bytes");
                    assert!(maxima.next().is_none(), "duplicate process peak metric");
                    assert!(process_peak > 0);
                    println!(
                        "async sort process peak: platform=macos rows={} denied={denied} budget={BUDGET} runtime_rss={runtime_rss} store_rss={store_rss} peak={process_peak} ceiling={ceiling} measurement=time-l-max-rss baseline=ps-current-rss",
                        ROWS * scale,
                    );
                    assert!(
                        process_peak <= ceiling,
                        "process peak {process_peak} exceeds fixed ceiling {ceiling}\n{stdout}\n{stderr}"
                    );
                }
                print!("{stdout}");
            }
        }
        return;
    };
    let scale: usize = scale.to_str().unwrap().parse().unwrap();
    assert!([1, 2, 4].contains(&scale));
    let denied = match std::env::var(CHILD_DENIED).unwrap().as_str() {
        "0" => false,
        "1" => true,
        other => panic!("invalid child quota mode: {other}"),
    };
    let rows = ROWS * scale;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async { tokio::task::spawn_blocking(|| {}).await.unwrap() });
    #[cfg(target_os = "linux")]
    let runtime_baseline = process_memory();
    #[cfg(target_os = "macos")]
    let runtime_baseline = process_rss();
    let directory = tempfile::tempdir().unwrap();
    let mut config = Config::in_memory()
        .with_memory_limit(BUDGET)
        .with_spill_path(directory.path());
    if denied {
        config = config.with_max_query_spill_bytes(0);
    }
    let database = Arc::new(GrafeoDB::with_config(config).unwrap());
    for index in 0..rows {
        database.create_node_with_props(
            &["AsyncPeak"],
            [
                ("value", Value::Int64(i64::try_from(index / 2).unwrap())),
                (
                    "key",
                    Value::from(format!(
                        "{:05}-{}",
                        (rows - 1 - index) / 2,
                        "x".repeat(KEY_BYTES)
                    )),
                ),
            ],
        );
    }
    #[cfg(target_os = "linux")]
    let store_baseline = process_memory();
    #[cfg(target_os = "macos")]
    let store_baseline = process_rss();
    #[cfg(target_os = "linux")]
    let ceiling = runtime_baseline
        .1
        .max(store_baseline.1)
        .checked_add(PROCESS_HEADROOM)
        .unwrap();
    // These macOS baselines are current RSS, not high-water samples. The
    // parent time process measures the child's lifetime high-water, including
    // setup and teardown, and compares it against this fixed-headroom ceiling.
    #[cfg(target_os = "macos")]
    let ceiling = runtime_baseline
        .max(store_baseline)
        .checked_add(PROCESS_HEADROOM)
        .unwrap();

    let peak = runtime.block_on(async {
        // This is the actual binding dispatch/preparation entrypoint. Never
        // substitute a synthetic operator or accept a synchronous fallback.
        let preparing_database = Arc::clone(&database);
        let dispatch = tokio::task::spawn_blocking(move || {
            preparing_database.execute_or_prepare_async_sort(
                QUERY,
                HashMap::new(),
                ExecutionOptions::default(),
            )
        })
        .await
        .unwrap()
        .unwrap();
        let AsyncSortDispatch::Prepared(prepared) = dispatch else {
            panic!("memory qualification must exercise scheduled sort");
        };
        let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
        let resources = shared.resources.clone();
        let weak_owner = Arc::downgrade(&shared);
        let outcome = prepared.execute().await;
        assert!(shared.leaf_finish_attempted.load(Ordering::Acquire));
        assert!(shared.state.lock().is_none());
        assert!(shared.publication.lock().is_none());

        let peak = resources.profile_stats();
        assert!((1..=BUDGET).contains(&peak.resident_peak_bytes), "{peak:?}");
        if denied {
            let error = outcome.as_ref().unwrap_err();
            assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
            assert!(
                error.to_string().contains("spill disk quota exceeded"),
                "{error}"
            );
            assert_eq!(peak.spilled_bytes, 0);
            assert_eq!(peak.spill_runs, 0);
        } else {
            let result = outcome.as_ref().unwrap();
            assert_eq!(result.columns, ["value"]);
            assert_eq!(result.row_count(), rows);
            // Identical values and keys occur twice. Check every duplicate
            // and its position across the complete merged output without a
            // second input-sized expected-result allocation in the probe.
            for (position, row) in result.rows().iter().enumerate() {
                assert_eq!(
                    row.as_slice(),
                    &[Value::Int64(
                        i64::try_from((rows - 1 - position) / 2).unwrap()
                    )],
                    "row {position} of {rows}"
                );
            }
            assert!(peak.spilled_bytes > 0, "{peak:?}");
            assert!(peak.spill_runs >= 2, "{peak:?}");
        }
        let manager = resources.spill_manager().expect("admitted spill manager");
        assert_eq!(manager.active_file_count(), 0);
        let disk = manager.disk_stats();
        assert_eq!(disk.reserved_live_bytes, 0);
        assert_eq!(disk.published_live_bytes, 0);
        let physical = peak.spill_physical.expect("retained physical accounting");
        assert_eq!(physical.reserved_bytes, 0, "{physical:?}");
        assert_eq!(physical.observed_file_bytes, 0, "{physical:?}");
        assert_eq!(physical.cleanup_debt_bytes, 0, "{physical:?}");
        assert!(
            !physical.cleanup_failed && !physical.reservation_uncertain,
            "{physical:?}"
        );
        if !denied {
            assert!(disk.peak_reserved_bytes > 0);
            assert!(physical.peak_observed_file_bytes > 0, "{physical:?}");
        }
        drop(outcome);
        drop(shared);
        // The terminal oneshot precedes the physical worker's last scheduler
        // bookkeeping drops. Wait for those already-finished owners, as the
        // existing async lifecycle controls do; never rerun the query.
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while weak_owner.upgrade().is_some()
                || resources.query_stats().allocated_bytes != 0
                || resources.buffer_stats().total_allocated != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal async sort owners did not release their grants");
        assert!(weak_owner.upgrade().is_none(), "terminal owner leaked");
        assert_eq!(
            resources.query_stats().allocated_bytes,
            0,
            "query grants leaked"
        );
        assert_eq!(
            resources.buffer_stats().total_allocated,
            0,
            "database grants leaked"
        );
        peak
    });

    let assert_cleanup = || {
        let namespace = directory
            .path()
            .join(format!("grafeo-store-{}", database.store_id()));
        for entry in std::fs::read_dir(namespace).unwrap() {
            assert!(
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("grafeo-query-"),
                "terminal query leaf leaked"
            );
        }
    };
    assert_cleanup();
    let follow_up = database.execute("RETURN 7 AS value").unwrap();
    assert_eq!(follow_up.rows(), &[vec![Value::Int64(7)]]);
    drop(follow_up);
    assert_cleanup();
    #[cfg(target_os = "linux")]
    {
        let final_memory = process_memory();
        println!(
            "async sort process peak: rows={rows} denied={denied} budget={BUDGET} runtime_rss={} runtime_hwm={} store_rss={} store_hwm={} final_rss={} peak={} ceiling={ceiling} resident_peak_bytes={} spilled_bytes_total={} spill_runs_total={} released_query_bytes=0",
            runtime_baseline.0,
            runtime_baseline.1,
            store_baseline.0,
            store_baseline.1,
            final_memory.0,
            final_memory.1,
            peak.resident_peak_bytes,
            peak.spilled_bytes,
            peak.spill_runs,
        );
        assert!(final_memory.1 > 0);
        assert!(
            final_memory.1 <= ceiling,
            "process peak {} exceeds fixed ceiling {ceiling}",
            final_memory.1
        );
    }
    #[cfg(target_os = "macos")]
    println!(
        "async sort process sample: platform=macos rows={rows} denied={denied} budget={BUDGET} runtime_rss={runtime_baseline} store_rss={store_baseline} final_rss={} ceiling={ceiling} resident_peak_bytes={} spilled_bytes_total={} spill_runs_total={} released_query_bytes=0 baseline=ps-current-rss",
        process_rss(),
        peak.resident_peak_bytes,
        peak.spilled_bytes,
        peak.spill_runs,
    );
}
