//! Decision Gate pins on **production** types (not the Task 0 IntervalAdj spike).
//!
//! Run:
//! `CARGO_TARGET_DIR=/tmp/grafeo-gate cargo bench --locked -p grafeo-core --features compact-store,lpg,statement-table --bench asof_production_gate`
//!
//! `GRAFEO_GATE=wholegraph,seeded` runs only those sections (`current`, `frame`,
//! `compact`, `wholegraph`, `seeded`, `layered`, `stmt`, or `all`). Unset = everything.
//!
//! Prints `[pin]` / `[gate]` lines. Set `GRAFEO_GATE_STRICT=1` on the declared
//! reference runner to return non-zero when any selected gate misses.

#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use grafeo_common::storage::section::Section;
use grafeo_common::types::{EdgeId, EpochId, EpochInterval, TransactionId, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::compact::column::ColumnCodec;
use grafeo_core::graph::compact::csr::{
    CsrAdjacency, PackedOpenAdjacency, TemporalEdgeRow, pack_open_prefix,
};
use grafeo_core::graph::compact::from_graph_store_preserving_ids;
use grafeo_core::graph::compact::layered::LayeredStore;
use grafeo_core::graph::compact::section::CompactStoreSection;
use grafeo_core::graph::compact::statement_table::{StatementIngest, StatementTable};
use grafeo_core::graph::compact::temporal_column::TemporalColumn;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::{GraphStore, GraphStoreMut};

static GATE_FAILURES: AtomicUsize = AtomicUsize::new(0);

/// the application's declared 60 Hz whole-graph admission budget. This is intentionally distinct
/// from the older synthetic kernel-frame target below.
const WHOLE_GRAPH_FRAME_BUDGET_MS: f64 = 16.6;

fn pin_ms(label: &str, warmup: u32, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ms = start.elapsed().as_secs_f64() * 1_000.0 / f64::from(iters);
    eprintln!("[pin] {label}: {ms:.3} ms/iter (warmup={warmup} iters={iters})");
    ms
}

fn gate(name: &str, ok: bool, detail: &str) {
    let mark = if ok { "PASS" } else { "FAIL" };
    if !ok {
        GATE_FAILURES.fetch_add(1, Ordering::Relaxed);
    }
    eprintln!("[gate] {mark}  {name}  {detail}");
}

fn strict_gate_enabled() -> bool {
    std::env::var("GRAFEO_GATE_STRICT").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    })
}

fn gate_wants(name: &str) -> bool {
    match std::env::var("GRAFEO_GATE") {
        Ok(s) if !s.trim().is_empty() => s.split(',').any(|p| {
            let p = p.trim();
            p == name || p == "all"
        }),
        _ => true,
    }
}

fn dest(src: u32, k: u32, generation: u32, num_nodes: u32) -> u32 {
    let mixed = src
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(k.wrapping_mul(0x85EB_CA6B))
        .wrapping_add(generation.wrapping_mul(0xC2B2_AE35));
    let mut dst = mixed % num_nodes;
    if dst == src {
        dst = (dst + 1) % num_nodes;
    }
    dst
}

fn build_rows(num_nodes: u32, degree: u32) -> Vec<TemporalEdgeRow> {
    let mut rows = Vec::with_capacity((num_nodes * degree * 2) as usize);
    let e10 = EpochId::new(10);
    let e20 = EpochId::new(20);
    let e30 = EpochId::new(30);
    let mut eid = 1u64;
    for src in 0..num_nodes {
        for k in 0..degree {
            let dst0 = dest(src, k, 0, num_nodes);
            if k % 2 == 0 {
                rows.push(TemporalEdgeRow {
                    src,
                    dst: dst0,
                    validity: EpochInterval::closed(e10, e20),
                    edge_id: EdgeId::new(eid),
                });
                eid += 1;
                rows.push(TemporalEdgeRow {
                    src,
                    dst: dest(src, k, 1, num_nodes),
                    validity: EpochInterval::open(e20),
                    edge_id: EdgeId::new(eid),
                });
                eid += 1;
            } else if k % 4 == 1 {
                rows.push(TemporalEdgeRow {
                    src,
                    dst: dst0,
                    validity: EpochInterval::closed(e10, e30),
                    edge_id: EdgeId::new(eid),
                });
                eid += 1;
                rows.push(TemporalEdgeRow {
                    src,
                    dst: dest(src, k, 2, num_nodes),
                    validity: EpochInterval::open(e30),
                    edge_id: EdgeId::new(eid),
                });
                eid += 1;
            } else {
                rows.push(TemporalEdgeRow {
                    src,
                    dst: dst0,
                    validity: EpochInterval::open(e10),
                    edge_id: EdgeId::new(eid),
                });
                eid += 1;
            }
        }
    }
    rows
}

fn build_node_column(num_nodes: usize) -> TemporalColumn {
    let mut values = Vec::with_capacity(num_nodes * 3);
    let mut validity = Vec::with_capacity(num_nodes * 3);
    let e10 = EpochId::new(10);
    let e20 = EpochId::new(20);
    let e30 = EpochId::new(30);
    for i in 0..num_nodes {
        let base = i as i64;
        values.push(base);
        validity.push(EpochInterval::closed(e10, e20));
        values.push(base.saturating_mul(2));
        validity.push(EpochInterval::closed(e20, e30));
        values.push(base.saturating_mul(3));
        validity.push(EpochInterval::open(e30));
    }
    TemporalColumn::new(ColumnCodec::raw_i64(values), validity)
}

fn expand_csr(csr: &CsrAdjacency) -> u64 {
    let mut acc = 0u64;
    let n = csr.num_nodes() as u32;
    for src in 0..n {
        for &t in csr.neighbors(src) {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

fn expand_packed_current(packed: &PackedOpenAdjacency) -> u64 {
    let mut acc = 0u64;
    let n = packed.num_nodes() as u32;
    for src in 0..n {
        for &t in packed.current_neighbors(src) {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

fn expand_packed_asof(packed: &PackedOpenAdjacency, epoch: EpochId, buf: &mut Vec<u32>) -> u64 {
    let mut acc = 0u64;
    let n = packed.num_nodes() as u32;
    for src in 0..n {
        packed.fill_neighbors_at_epoch(src, epoch, buf);
        for &t in buf.iter() {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

fn scrub_edges(packed: &PackedOpenAdjacency, epoch: EpochId) -> usize {
    let mut n = 0usize;
    let srcs = packed.num_nodes() as u32;
    let mut buf = Vec::new();
    for src in 0..srcs {
        packed.fill_neighbors_at_epoch(src, epoch, &mut buf);
        n += buf.len();
    }
    n
}

fn scrub_nodes(col: &TemporalColumn, epoch: EpochId) -> i64 {
    let nodes = col.len() / 3;
    let mut sum = 0i64;
    for i in 0..nodes {
        if let Some(Value::Int64(v)) = col.value_in_range_as_of(i * 3, 3, epoch) {
            sum = sum.wrapping_add(v);
        }
    }
    sum
}

fn pin_budget(n: u32) -> (u32, u32) {
    match n {
        0..=40_000 => (16, 80),
        40_001..=80_000 => (12, 48),
        80_001..=200_000 => (8, 32),
        200_001..=400_000 => (4, 16),
        _ => (2, 8),
    }
}

struct CurrentRow {
    n: u32,
    open: usize,
    csr_ms: f64,
    derived_ms: f64,
    prefix_ms: f64,
}

fn kernel_current_at(n: u32, degree: u32) -> CurrentRow {
    let rows = build_rows(n, degree);
    let packed = pack_open_prefix(n as usize, &rows);
    let derived = packed.derive_current_csr();
    let mut open: Vec<(u32, u32)> = rows
        .iter()
        .filter(|r| r.validity.is_open())
        .map(|r| (r.src, r.dst))
        .collect();
    open.sort_unstable();
    let today = CsrAdjacency::from_sorted_edges(n as usize, &open);
    let (w, i) = pin_budget(n);
    let csr_ms = pin_ms(&format!("sweep/current/{n}/csr_today"), w, i, || {
        black_box(expand_csr(&today));
    });
    let derived_ms = pin_ms(&format!("sweep/current/{n}/derived_csr"), w, i, || {
        black_box(expand_csr(&derived));
    });
    let prefix_ms = pin_ms(&format!("sweep/current/{n}/packed_prefix"), w, i, || {
        black_box(expand_packed_current(&packed));
    });
    let regress = 100.0 * (derived_ms / csr_ms - 1.0);
    gate(
        &format!(
            "current 1-hop derived ≤10% (n={n} open={})",
            today.num_edges()
        ),
        regress <= 10.0,
        &format!("{derived_ms:.3} / {csr_ms:.3} ms = {regress:+.1}%"),
    );
    CurrentRow {
        n,
        open: today.num_edges(),
        csr_ms,
        derived_ms,
        prefix_ms,
    }
}

struct FrameRow {
    n: u32,
    node_ms: f64,
    expand_ms: f64,
    edge_ms: f64,
}

fn kernel_frame_at(n: u32, degree: u32) -> FrameRow {
    let rows = build_rows(n, degree);
    let packed = pack_open_prefix(n as usize, &rows);
    let nodes = build_node_column(n as usize);
    let epoch = EpochId::new(25);
    let mut buf = Vec::with_capacity(8);
    let (w, i) = pin_budget(n);
    let node_ms = pin_ms(&format!("sweep/frame/{n}/node_scrub"), w, i, || {
        black_box(scrub_nodes(&nodes, epoch));
    });
    let expand_ms = pin_ms(&format!("sweep/frame/{n}/asof_expand"), w, i, || {
        black_box(expand_packed_asof(&packed, epoch, &mut buf));
    });
    let edge_ms = pin_ms(&format!("sweep/frame/{n}/edge_scrub"), w, i, || {
        black_box(scrub_edges(&packed, epoch));
    });
    let frame = node_ms + expand_ms + edge_ms;
    gate(
        &format!("as-of frame ≤16 ms (n={n})"),
        frame <= 16.0,
        &format!("{frame:.3} ms = {:.1}% of budget", 100.0 * frame / 16.0),
    );
    FrameRow {
        n,
        node_ms,
        expand_ms,
        edge_ms,
    }
}

fn production_whole_graph_frame(n: usize, degree: usize) {
    let store = LpgStore::new().unwrap();
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        ids.push(store.create_node(&["Entity"]));
    }
    for (i, &src) in ids.iter().enumerate() {
        for k in 0..degree {
            let dst = ids[(i + k + 1) % n];
            store.create_edge(src, dst, "PROV");
        }
    }
    let compact = from_graph_store_preserving_ids(&store).unwrap();
    let epoch = EpochId::PENDING;
    let expand_ms = pin_ms(&format!("prod/wholegraph/{n}/fill_all_src"), 4, 16, || {
        let mut acc = 0u64;
        compact.visit_neighbors_at_epoch(Direction::Outgoing, epoch, |_, dests| {
            acc = acc.wrapping_add(dests.len() as u64);
        });
        black_box(acc);
    });
    let scrub_ms = pin_ms(&format!("prod/wholegraph/{n}/graph_scrub"), 4, 16, || {
        black_box(compact.graph_scrub_at_epoch(epoch));
    });
    let frame = expand_ms + scrub_ms;
    eprintln!(
        "[note] production whole-graph frame n={n}: expand {expand_ms:.3} + scrub {scrub_ms:.3} = {frame:.3} ms"
    );
    gate(
        &format!("production whole-graph frame ≤{WHOLE_GRAPH_FRAME_BUDGET_MS:.1} ms (n={n})"),
        frame <= WHOLE_GRAPH_FRAME_BUDGET_MS,
        &format!(
            "{frame:.3} ms = {:.1}% of budget",
            100.0 * frame / WHOLE_GRAPH_FRAME_BUDGET_MS
        ),
    );
}

fn temporal_workload_seeded(n: usize) {
    let n_src = (n / 20).max(8);
    let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
    let layered = LayeredStore::new(empty, (n as u64) + 16, (n * 8) as u64 + 16).unwrap();
    let overlay = layered.overlay_store();
    overlay.set_epoch(EpochId::new(10));
    let mut ents = Vec::with_capacity(n);
    for i in 0..n {
        let id = overlay.create_node(&["Entity"]);
        overlay.set_node_property(id, "id", Value::from(i.to_string()));
        ents.push(id);
    }
    let mut sources = Vec::with_capacity(n_src);
    for i in 0..n_src {
        let id = overlay.create_node(&["Source"]);
        overlay.set_node_property(id, "id", Value::from(format!("s{i}")));
        sources.push(id);
    }
    for (i, &ent) in ents.iter().enumerate() {
        overlay.create_edge(sources[i % n_src], ent, "OBSERVED_BY");
        overlay.create_edge(sources[(i + 1) % n_src], ent, "OBSERVED_BY");
        let j = (i * 7 + 3) % n;
        if j != i {
            overlay.create_edge(ent, ents[j], "CORRELATED_WITH");
        }
    }
    overlay.set_epoch(EpochId::new(20));
    layered.merge_overlay_temporal().unwrap();
    let h = ents[1];
    let epoch = EpochId::new(15);
    let mut buf = Vec::new();
    let obs_ty = [String::from("OBSERVED_BY")];
    let corr_ty = [String::from("CORRELATED_WITH")];
    let obs = pin_ms(&format!("prod/seeded/{n}/observers_in"), 8, 32, || {
        layered.fill_neighbors_of_types_at_epoch(h, Direction::Incoming, epoch, &obs_ty, &mut buf);
        black_box(buf.len());
    });
    let hop2 = pin_ms(&format!("prod/seeded/{n}/cluster_2hop_fill"), 8, 32, || {
        layered.fill_neighbors_of_types_at_epoch(h, Direction::Outgoing, epoch, &corr_ty, &mut buf);
        let mid = buf.clone();
        let mut acc = 0usize;
        for m in &mid {
            layered.fill_neighbors_of_types_at_epoch(
                *m,
                Direction::Outgoing,
                epoch,
                &corr_ty,
                &mut buf,
            );
            acc = acc.wrapping_add(buf.len());
        }
        black_box(acc);
    });
    eprintln!("[note] Temporal workload n={n}: observers {obs:.3} ms, cluster fill {hop2:.3} ms");
    gate(
        &format!("Seeded observers+cluster < 0.2 ms (n={n})"),
        obs < 0.2 && hop2 < 0.2,
        &format!("in {obs:.3} / 2-hop {hop2:.3}"),
    );
}

fn compact_current_only_320k() {
    let n = 80_000usize;
    let degree = 4usize;
    let store = LpgStore::new().unwrap();
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        ids.push(store.create_node(&["Entity"]));
    }
    for (i, &src) in ids.iter().enumerate() {
        for k in 0..degree {
            let dst = ids[(i + k + 1) % n];
            store.create_edge(src, dst, "PROV");
        }
    }
    let compact = from_graph_store_preserving_ids(&store).unwrap();
    assert_eq!(compact.node_count(), n);
    assert_eq!(compact.edge_count(), n * degree);
    let bpe = compact.memory_bytes() as f64 / n as f64;
    eprintln!(
        "[size] all-open CompactStore 80k/320k: {} B ({bpe:.1} B/entity)",
        compact.memory_bytes()
    );
    let ids = compact.node_ids();
    let ms = pin_ms("prod/compact/320k/neighbors_current", 8, 40, || {
        let mut acc = 0u64;
        for id in &ids {
            acc = acc.wrapping_add(compact.neighbors(*id, Direction::Outgoing).len() as u64);
        }
        black_box(acc);
    });
    gate(
        "all-open CompactStore current expand ran",
        ms > 0.0,
        &format!("{ms:.3} ms/iter over 80k src (production neighbors())"),
    );
}

struct LayeredRow {
    n: usize,
    setup_s: f64,
    current_ms: f64,
    asof_ms: f64,
    overlay_ms: f64,
    ser_ms: f64,
    de_ms: f64,
    density_x: f64,
}

fn layered_temporal_and_persist(n: usize, degree: usize) -> LayeredRow {
    eprintln!("[setup] LayeredStore temporal n={n} degree={degree} …");
    let t0 = Instant::now();
    let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
    let layered = LayeredStore::new(empty, (n as u64) + 16, (n * degree * 3) as u64 + 16).unwrap();
    let overlay = layered.overlay_store();
    overlay.set_epoch(EpochId::new(10));
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = overlay.create_node(&["Entity"]);
        overlay.set_node_property_at_epoch(id, "score", Value::Int64(i as i64), EpochId::new(10));
        ids.push(id);
    }
    let mut replace = Vec::new();
    for (i, &src) in ids.iter().enumerate() {
        for k in 0..degree {
            let dst = ids[(i + k + 1) % n];
            let eid = overlay.create_edge_versioned(
                src,
                dst,
                "PROV",
                EpochId::new(10),
                TransactionId::SYSTEM,
            );
            if k % 2 == 0 {
                replace.push((src, ids[(i + k + 2) % n], eid));
            }
        }
    }
    overlay.set_epoch(EpochId::new(20));
    for (src, new_dst, eid) in &replace {
        assert!(overlay.delete_edge(*eid));
        overlay.create_edge_versioned(
            *src,
            *new_dst,
            "PROV",
            EpochId::new(20),
            TransactionId::SYSTEM,
        );
    }
    overlay.set_epoch(EpochId::new(20));
    layered.merge_overlay_temporal().unwrap();
    let setup_s = t0.elapsed().as_secs_f64();
    eprintln!("[setup] merge_overlay_temporal in {setup_s:.1}s");

    let base = layered.base_store_arc();
    assert_eq!(base.node_count(), n);
    let bpe = base.memory_bytes() as f64 / n as f64;
    eprintln!(
        "[size] temporal CompactStore n={n}: {} B ({bpe:.1} B/entity) closed_edges={}",
        base.memory_bytes(),
        base.closed_edge_ids().len()
    );
    let open_only = {
        let s = LpgStore::new().unwrap();
        let mut oids = Vec::with_capacity(n);
        for _ in 0..n {
            oids.push(s.create_node(&["Entity"]));
        }
        for (i, &src) in oids.iter().enumerate() {
            for k in 0..degree {
                s.create_edge(src, oids[(i + k + 1) % n], "PROV");
            }
        }
        from_graph_store_preserving_ids(&s).unwrap()
    };
    let open_bpe = open_only.memory_bytes() as f64 / n as f64;
    let density_ratio = bpe / open_bpe;
    eprintln!(
        "[size] all-open CompactStore n={n}: {} B ({open_bpe:.1} B/entity); temporal/open = {density_ratio:.2}×",
        open_only.memory_bytes()
    );
    gate(
        "temporal density ≤1.25× all-open CompactStore",
        density_ratio <= 1.25,
        &format!("{bpe:.1} / {open_bpe:.1} B/entity = {density_ratio:.2}×"),
    );

    let epoch = EpochId::new(15);
    let current_ms = pin_ms(
        &format!("prod/layered/{n}/neighbors_current"),
        4,
        20,
        || {
            let mut acc = 0u64;
            for id in &ids {
                acc = acc.wrapping_add(base.neighbors(*id, Direction::Outgoing).len() as u64);
            }
            black_box(acc);
        },
    );
    let asof_ms = pin_ms(
        &format!("prod/layered/{n}/neighbors_at_epoch_15"),
        4,
        20,
        || {
            let mut acc = 0u64;
            for id in &ids {
                acc = acc.wrapping_add(
                    base.neighbors_at_epoch(*id, Direction::Outgoing, epoch)
                        .len() as u64,
                );
            }
            black_box(acc);
        },
    );
    let mut fill_buf = Vec::new();
    let fill_ms = pin_ms(
        &format!("prod/layered/{n}/fill_neighbors_at_epoch_15"),
        4,
        20,
        || {
            let mut acc = 0u64;
            for id in &ids {
                base.fill_neighbors_at_epoch(*id, Direction::Outgoing, epoch, &mut fill_buf);
                acc = acc.wrapping_add(fill_buf.len() as u64);
            }
            black_box(acc);
        },
    );
    eprintln!(
        "[note] production CompactStore as-of expand {n} src: vec {asof_ms:.3} ms; fill {fill_ms:.3} ms; current {current_ms:.3} ms"
    );

    // Overlay-hot as-of: delete a handful of base edges without re-merge.
    for id in ids.iter().take(64) {
        let edges = layered.edges_from(*id, Direction::Outgoing);
        if let Some((_, eid)) = edges.first() {
            layered.delete_edge(*eid);
        }
    }
    let overlay_asof = pin_ms(
        &format!("prod/layered/{n}/neighbors_at_epoch_after_overlay_delete"),
        4,
        20,
        || {
            let mut acc = 0u64;
            for id in ids.iter().take(1024) {
                acc = acc.wrapping_add(
                    layered
                        .neighbors_at_epoch(*id, Direction::Outgoing, epoch)
                        .len() as u64,
                );
            }
            black_box(acc);
        },
    );
    eprintln!("[note] LayeredStore as-of 1024 src after 64 overlay deletes: {overlay_asof:.3} ms");

    let section = CompactStoreSection::new(Arc::clone(&base));
    let bytes = section.serialize().unwrap();
    eprintln!(
        "[size] v5 serialize n={n}: {} B (version={})",
        bytes.len(),
        bytes[4]
    );
    assert_eq!(bytes[4], 5);
    let ser_ms = pin_ms(&format!("prod/persist/{n}/serialize_v5"), 2, 8, || {
        black_box(section.serialize().unwrap());
    });
    let de_ms = pin_ms(&format!("prod/persist/{n}/deserialize_v5"), 2, 8, || {
        let mut s = CompactStoreSection::empty();
        s.deserialize(&bytes).unwrap();
        black_box(s.store().unwrap().node_count());
    });
    let mut restored = CompactStoreSection::empty();
    restored.deserialize(&bytes).unwrap();
    let r = restored.store().unwrap();
    let sample = ids[0];
    let before = base.neighbors_at_epoch(sample, Direction::Outgoing, epoch);
    let after = r.neighbors_at_epoch(sample, Direction::Outgoing, epoch);
    gate(
        "v5 persist preserves as-of neighbors (sample src)",
        before == after,
        &format!(
            "serialize {ser_ms:.3} ms, deserialize {de_ms:.3} ms, {} B",
            bytes.len()
        ),
    );
    LayeredRow {
        n,
        setup_s,
        current_ms,
        asof_ms,
        overlay_ms: overlay_asof,
        ser_ms,
        de_ms,
        density_x: density_ratio,
    }
}

struct StmtRow {
    n: usize,
    insert_ms: f64,
    ser_ms: f64,
    q_ms: f64,
}

fn statement_ingest(n: usize) -> StmtRow {
    let mut batch = Vec::with_capacity(n);
    for i in 0..n {
        let mut target = [0u8; 32];
        target[..8].copy_from_slice(&(i as u64).to_le_bytes());
        let mut source = [0u8; 32];
        source[0] = (i % 251) as u8;
        batch.push(StatementIngest {
            target_kind: 1,
            target_ref: target,
            target_row: i as u64,
            meta_kind: 0,
            assertion: vec![(i % 255) as u8; 16],
            timestamp_ns: 1_000 + i as i64,
            source_id: source,
        });
    }
    let insert_iters = if n >= 400_000 { 2 } else { 4 };
    let insert_ms = pin_ms(
        &format!("sweep/statements/{n}/insert"),
        1,
        insert_iters,
        || {
            let mut t = StatementTable::new();
            black_box(t.insert_batch(&batch, EpochId::new(10)));
        },
    );
    let mut table = StatementTable::new();
    table.insert_batch(&batch, EpochId::new(10));
    let mut bytes = Vec::new();
    table.write_to(&mut bytes);
    let ser_ms = pin_ms(&format!("sweep/statements/{n}/serialize"), 2, 8, || {
        let mut b = Vec::new();
        table.write_to(&mut b);
        black_box(b.len());
    });
    let q_ms = pin_ms(&format!("sweep/statements/{n}/at_epoch"), 4, 16, || {
        black_box(table.statements_at_epoch(EpochId::new(15)).len());
    });
    let restored = StatementTable::read_from(&bytes).unwrap();
    let mut again = Vec::new();
    restored.write_to(&mut again);
    gate(
        &format!("statement ingest determinism (n={n})"),
        bytes == again,
        &format!(
            "insert {insert_ms:.3} ms, serialize {ser_ms:.3} ms ({} B), query {q_ms:.3} ms",
            bytes.len()
        ),
    );
    StmtRow {
        n,
        insert_ms,
        ser_ms,
        q_ms,
    }
}

fn main() {
    eprintln!("=== Grafeo Decision Gate — size sweep ===");
    if let Ok(sel) = std::env::var("GRAFEO_GATE") {
        eprintln!("[note] GRAFEO_GATE={sel}");
    }

    let current: Vec<CurrentRow> = if gate_wants("current") {
        eprintln!("\n-- current 1-hop (degree 4, open = 4n) --");
        [20_000, 40_000, 80_000, 160_000, 320_000]
            .into_iter()
            .map(|n| kernel_current_at(n, 4))
            .collect()
    } else {
        Vec::new()
    };

    let frames: Vec<FrameRow> = if gate_wants("frame") {
        eprintln!("\n-- as-of frame (degree 2, node+expand+edge, 16 ms budget) --");
        [50_000, 100_000, 200_000, 400_000, 800_000]
            .into_iter()
            .map(|n| kernel_frame_at(n, 2))
            .collect()
    } else {
        Vec::new()
    };

    if gate_wants("compact") {
        eprintln!("\n-- CompactStore current-only (80k / 320k open, reference) --");
        compact_current_only_320k();
    }

    if gate_wants("wholegraph") {
        eprintln!(
            "\n-- production whole-graph frame (CompactStore fill + scrub, {WHOLE_GRAPH_FRAME_BUDGET_MS:.1} ms) --"
        );
        production_whole_graph_frame(50_000, 2);
        production_whole_graph_frame(200_000, 2);
    }

    if gate_wants("seeded") {
        eprintln!("\n-- Temporal workload seeded as-of (incoming observers + 2-hop) --");
        temporal_workload_seeded(10_000);
    }

    let layered: Vec<LayeredRow> = if gate_wants("layered") {
        eprintln!("\n-- LayeredStore temporal + persist (degree 2) --");
        [10_000, 20_000, 40_000, 80_000, 160_000]
            .into_iter()
            .map(|n| layered_temporal_and_persist(n, 2))
            .collect()
    } else {
        Vec::new()
    };

    let stmts: Vec<StmtRow> = if gate_wants("stmt") {
        eprintln!("\n-- statement ingest --");
        [50_000, 100_000, 200_000, 400_000, 800_000]
            .into_iter()
            .map(statement_ingest)
            .collect()
    } else {
        Vec::new()
    };

    eprintln!("\n========== SWEEP SUMMARY ==========");
    eprintln!("current 1-hop  (≤10% derived vs CSR)");
    eprintln!("     n     open      csr     derived    prefix    derived%");
    for r in &current {
        let pct = 100.0 * (r.derived_ms / r.csr_ms - 1.0);
        eprintln!(
            "{:>7} {:>8} {:>8.3} {:>10.3} {:>8.3} {:>+8.1}%",
            r.n, r.open, r.csr_ms, r.derived_ms, r.prefix_ms, pct
        );
    }
    eprintln!("as-of frame  (≤16 ms)");
    eprintln!("     n     node   expand     edge    frame   %budget");
    for r in &frames {
        let frame = r.node_ms + r.expand_ms + r.edge_ms;
        eprintln!(
            "{:>7} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>7.1}%",
            r.n,
            r.node_ms,
            r.expand_ms,
            r.edge_ms,
            frame,
            100.0 * frame / 16.0
        );
    }
    eprintln!("LayeredStore + persist");
    eprintln!("     n   setup_s  current    asof  overlay     ser      de   dens×");
    for r in &layered {
        eprintln!(
            "{:>7} {:>8.2} {:>8.3} {:>7.3} {:>8.3} {:>7.3} {:>7.3} {:>6.2}",
            r.n, r.setup_s, r.current_ms, r.asof_ms, r.overlay_ms, r.ser_ms, r.de_ms, r.density_x
        );
    }
    eprintln!("statements");
    eprintln!("     n   insert      ser   query");
    for r in &stmts {
        eprintln!(
            "{:>7} {:>8.3} {:>8.3} {:>7.3}",
            r.n, r.insert_ms, r.ser_ms, r.q_ms
        );
    }
    eprintln!("===================================");

    let failures = GATE_FAILURES.load(Ordering::Relaxed);
    if strict_gate_enabled() {
        eprintln!("[gate] strict mode: {failures} selected gate failure(s)");
        if failures != 0 {
            std::process::exit(1);
        }
    }
}
