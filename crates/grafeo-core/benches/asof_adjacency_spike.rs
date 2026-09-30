//! As-of adjacency spike (Task A0 / unification Phase 1 Task 0).
//!
//! Run:
//! `CARGO_TARGET_DIR=/tmp/grafeo-proto cargo bench -p grafeo-core --features compact-store,lpg --bench asof_adjacency_spike -- --quick`
//!
//! Isolated prototypes only. Production (Task 4) now has
//! [`grafeo_core::graph::compact::csr::PackedOpenAdjacency`] plus
//! `build_current_csr_from_open_edges`: after `merge_overlay_temporal`,
//! `RelTable.fwd` is the **derived current CSR** (open prefix) and
//! `RelTable.packed_fwd` is the fat interval run. Current 1-hop is that
//! derived CSR / prefix slice — never `filter(is_open)` on the fat list.
//! v5 persist writes packed adjacency and the closed-edge sidecar.
//!
//! Spike numbers below (Task 0, 2026-08-15, Apple M4): A packed prefix
//! +3.9% vs today's CSR; A-naive `is_open` filter +13.6% pin / +57%
//! criterion (banned). Production current path is the derived CSR, same
//! access pattern as `CsrAdjacency::neighbors`.
//!
//! Candidate historical-adjacency layouts:
//!
//! * **A** — interval-annotated CSR of every edge version; open rows packed
//!   as a prefix of each neighbor list (current = slice; as-of = filter).
//! * **B** — today's current CSR untouched + cold closed-edge CSR; as-of
//!   merges current (gated by `from`) with a closed scan.
//! * **C** — one `CsrAdjacency` per sampled epoch (density only).
// Benchmarks are not public API — suppress doc and pedantic lints.
#![allow(
    missing_docs,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::{EpochId, EpochInterval, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::compact::column::ColumnCodec;
use grafeo_core::graph::compact::csr::CsrAdjacency;
use grafeo_core::graph::compact::from_graph_store_preserving_ids;
use grafeo_core::graph::compact::layered::LayeredStore;
use grafeo_core::graph::compact::temporal_column::TemporalColumn;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::{GraphStore, GraphStoreMut};

/// Half-open validity stamped on each historical edge version.
#[derive(Clone, Copy)]
struct EdgeLife {
    src: u32,
    dst: u32,
    interval: EpochInterval,
}

/// Option A: one CSR of *all* versions, `validity` parallel to `targets`.
///
/// Open (still-current) rows are packed at the front of each source's
/// neighbor run so current 1-hop is a prefix slice — the same access
/// pattern as [`CsrAdjacency::neighbors`].
struct IntervalAdj {
    offsets: Vec<u32>,
    /// Exclusive end of the open prefix for node `i` (`offsets[i]..open_ends[i]`).
    open_ends: Vec<u32>,
    targets: Vec<u32>,
    validity: Vec<EpochInterval>,
}

impl IntervalAdj {
    fn from_lives(num_nodes: usize, lives: &[EdgeLife]) -> Self {
        let mut open: Vec<Vec<(u32, EpochInterval)>> = vec![Vec::new(); num_nodes];
        let mut closed: Vec<Vec<(u32, EpochInterval)>> = vec![Vec::new(); num_nodes];
        for life in lives {
            let bucket = if life.interval.is_open() {
                &mut open[life.src as usize]
            } else {
                &mut closed[life.src as usize]
            };
            bucket.push((life.dst, life.interval));
        }

        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut open_ends = Vec::with_capacity(num_nodes);
        let mut targets = Vec::with_capacity(lives.len());
        let mut validity = Vec::with_capacity(lives.len());
        offsets.push(0);
        for i in 0..num_nodes {
            for &(dst, iv) in &open[i] {
                targets.push(dst);
                validity.push(iv);
            }
            open_ends.push(u32::try_from(targets.len()).expect("csr targets fit u32"));
            for &(dst, iv) in &closed[i] {
                targets.push(dst);
                validity.push(iv);
            }
            offsets.push(u32::try_from(targets.len()).expect("csr targets fit u32"));
        }
        Self {
            offsets,
            open_ends,
            targets,
            validity,
        }
    }

    fn num_nodes(&self) -> usize {
        self.open_ends.len()
    }

    #[inline]
    fn current_neighbors(&self, src: u32) -> &[u32] {
        let i = src as usize;
        if i >= self.open_ends.len() {
            return &[];
        }
        let start = self.offsets[i] as usize;
        let end = self.open_ends[i] as usize;
        &self.targets[start..end]
    }

    /// Naive current path: scan the whole historical run, keep `is_open`.
    #[inline]
    fn current_neighbors_filter_open(&self, src: u32, out: &mut Vec<u32>) {
        out.clear();
        let i = src as usize;
        if i + 1 >= self.offsets.len() {
            return;
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        for e in start..end {
            if self.validity[e].is_open() {
                out.push(self.targets[e]);
            }
        }
    }

    #[inline]
    fn neighbors_at(&self, src: u32, epoch: EpochId, out: &mut Vec<u32>) {
        out.clear();
        let i = src as usize;
        if i + 1 >= self.offsets.len() {
            return;
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        if epoch == EpochId::PENDING {
            out.extend_from_slice(&self.targets[start..self.open_ends[i] as usize]);
            return;
        }
        for e in start..end {
            if self.validity[e].contains(epoch) {
                out.push(self.targets[e]);
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        self.offsets.len() * 4
            + self.open_ends.len() * 4
            + self.targets.len() * 4
            + self.validity.len() * std::mem::size_of::<EpochInterval>()
    }
}

/// Option B: production current CSR + cold closed-edge CSR.
///
/// Open edges still need a `from` stamp so as-of at a past epoch can exclude
/// edges created *after* that epoch. That column is not on today's
/// [`CsrAdjacency`]; it is extra as-of state, not on the current 1-hop path.
struct DualAdj {
    current: CsrAdjacency,
    /// Creation epoch, parallel to `current.targets`.
    current_from: Vec<EpochId>,
    closed_offsets: Vec<u32>,
    closed_targets: Vec<u32>,
    closed_validity: Vec<EpochInterval>,
}

impl DualAdj {
    fn from_lives(num_nodes: usize, lives: &[EdgeLife]) -> Self {
        let mut open: Vec<Vec<(u32, EpochId)>> = vec![Vec::new(); num_nodes];
        let mut closed: Vec<Vec<(u32, EpochInterval)>> = vec![Vec::new(); num_nodes];
        for life in lives {
            if life.interval.is_open() {
                open[life.src as usize].push((life.dst, life.interval.from()));
            } else {
                closed[life.src as usize].push((life.dst, life.interval));
            }
        }

        let mut current_pairs = Vec::new();
        let mut current_from = Vec::new();
        for (src, list) in open.iter().enumerate() {
            for &(dst, from) in list {
                current_pairs.push((src as u32, dst));
                current_from.push(from);
            }
        }
        let current = CsrAdjacency::from_sorted_edges(num_nodes, &current_pairs);

        let mut closed_offsets = Vec::with_capacity(num_nodes + 1);
        let mut closed_targets = Vec::new();
        let mut closed_validity = Vec::new();
        closed_offsets.push(0);
        for list in &closed {
            for &(dst, iv) in list {
                closed_targets.push(dst);
                closed_validity.push(iv);
            }
            closed_offsets.push(u32::try_from(closed_targets.len()).expect("closed fit u32"));
        }

        Self {
            current,
            current_from,
            closed_offsets,
            closed_targets,
            closed_validity,
        }
    }

    #[inline]
    fn neighbors_at(&self, src: u32, epoch: EpochId, out: &mut Vec<u32>) {
        out.clear();
        if epoch == EpochId::PENDING {
            out.extend_from_slice(self.current.neighbors(src));
            return;
        }
        let i = src as usize;
        let cur = self.current.neighbors(src);
        let cur_start = self.current.offset_of(src) as usize;
        for (k, &dst) in cur.iter().enumerate() {
            if self.current_from[cur_start + k] <= epoch {
                out.push(dst);
            }
        }
        if i + 1 >= self.closed_offsets.len() {
            return;
        }
        let start = self.closed_offsets[i] as usize;
        let end = self.closed_offsets[i + 1] as usize;
        for e in start..end {
            if self.closed_validity[e].contains(epoch) {
                out.push(self.closed_targets[e]);
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        self.current.memory_bytes()
            + self.current_from.len() * std::mem::size_of::<EpochId>()
            + self.closed_offsets.len() * 4
            + self.closed_targets.len() * 4
            + self.closed_validity.len() * std::mem::size_of::<EpochInterval>()
    }
}

/// Option C: a full current-shaped CSR at each sampled epoch.
struct EpochSnapshots {
    csrs: Vec<CsrAdjacency>,
}

impl EpochSnapshots {
    fn from_lives(num_nodes: usize, lives: &[EdgeLife], epochs: &[EpochId]) -> Self {
        let csrs = epochs
            .iter()
            .map(|&epoch| {
                let mut pairs: Vec<(u32, u32)> = lives
                    .iter()
                    .filter(|e| e.interval.contains(epoch))
                    .map(|e| (e.src, e.dst))
                    .collect();
                pairs.sort_unstable_by_key(|&(s, _)| s);
                CsrAdjacency::from_sorted_edges(num_nodes, &pairs)
            })
            .collect();
        Self { csrs }
    }

    fn heap_bytes(&self) -> usize {
        self.csrs.iter().map(CsrAdjacency::memory_bytes).sum()
    }
}

struct Fixture {
    num_open: usize,
    current: CsrAdjacency,
    a: IntervalAdj,
    b: DualAdj,
    c: EpochSnapshots,
    node_col: TemporalColumn,
    sample_hops: Vec<(u32, EpochId)>,
}

/// Deterministic “provenance” lifetime: each source starts with `degree`
/// edges at epoch 10; half close-and-replace at 20 and again at 30.
fn build_lives(num_nodes: u32, degree: u32) -> Vec<EdgeLife> {
    let mut lives = Vec::with_capacity((num_nodes * degree * 2) as usize);
    let e10 = EpochId::new(10);
    let e20 = EpochId::new(20);
    let e30 = EpochId::new(30);
    for src in 0..num_nodes {
        for k in 0..degree {
            let dst0 = dest(src, k, 0, num_nodes);
            if k % 2 == 0 {
                // Closes at 20, replaced; replacement stays open.
                lives.push(EdgeLife {
                    src,
                    dst: dst0,
                    interval: EpochInterval::closed(e10, e20),
                });
                lives.push(EdgeLife {
                    src,
                    dst: dest(src, k, 1, num_nodes),
                    interval: EpochInterval::open(e20),
                });
            } else if k % 4 == 1 {
                // Lives [10,30), then replaced.
                lives.push(EdgeLife {
                    src,
                    dst: dst0,
                    interval: EpochInterval::closed(e10, e30),
                });
                lives.push(EdgeLife {
                    src,
                    dst: dest(src, k, 2, num_nodes),
                    interval: EpochInterval::open(e30),
                });
            } else {
                // Open from the start (never closed).
                lives.push(EdgeLife {
                    src,
                    dst: dst0,
                    interval: EpochInterval::open(e10),
                });
            }
        }
    }
    lives
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

fn build_current_csr(num_nodes: usize, lives: &[EdgeLife]) -> CsrAdjacency {
    let pairs: Vec<(u32, u32)> = lives
        .iter()
        .filter(|e| e.interval.is_open())
        .map(|e| (e.src, e.dst))
        .collect();
    CsrAdjacency::from_sorted_edges(num_nodes, &pairs)
}

fn sample_hops(num_nodes: u32, n: usize) -> Vec<(u32, EpochId)> {
    const EPOCHS: [u64; 3] = [15, 25, 35];
    (0..n)
        .map(|i| {
            let src = (i as u32).wrapping_mul(0x9E37_79B9) % num_nodes;
            let epoch = EpochId::new(EPOCHS[i % EPOCHS.len()]);
            (src, epoch)
        })
        .collect()
}

/// 200k nodes × 3 version rows — node-scrub stand-in (no LayeredStore merge).
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

fn build_fixture(name: &'static str, num_nodes: u32, degree: u32) -> Fixture {
    let lives = build_lives(num_nodes, degree);
    let n = num_nodes as usize;
    let num_open = lives.iter().filter(|e| e.interval.is_open()).count();
    let num_closed = lives.len() - num_open;
    let current = build_current_csr(n, &lives);
    let a = IntervalAdj::from_lives(n, &lives);
    let b = DualAdj::from_lives(n, &lives);
    let snap_epochs = [EpochId::new(15), EpochId::new(25), EpochId::new(35)];
    let c = EpochSnapshots::from_lives(n, &lives, &snap_epochs);
    assert_layouts_agree(n, &a, &b, &current, &snap_epochs);
    let node_col = build_node_column(n);
    let hops = sample_hops(num_nodes, 10_000);
    eprintln!(
        "[{name}] nodes={n} open={num_open} closed={num_closed} \
         current_csr={}B A={}B B={}B C(3 epochs)={}B \
         bytes/hist-edge A={:.1} B={:.1} C={:.1} \
         EpochInterval={}B",
        current.memory_bytes(),
        a.heap_bytes(),
        b.heap_bytes(),
        c.heap_bytes(),
        extra_bytes_per_closed(a.heap_bytes(), current.memory_bytes(), num_closed),
        extra_bytes_per_closed(b.heap_bytes(), current.memory_bytes(), num_closed),
        extra_bytes_per_closed(c.heap_bytes(), current.memory_bytes(), num_closed),
        std::mem::size_of::<EpochInterval>(),
    );
    Fixture {
        num_open,
        current,
        a,
        b,
        c,
        node_col,
        sample_hops: hops,
    }
}

fn extra_bytes_per_closed(total: usize, current: usize, num_closed: usize) -> f64 {
    if num_closed == 0 {
        0.0
    } else {
        (total.saturating_sub(current) as f64) / (num_closed as f64)
    }
}

fn assert_layouts_agree(
    num_nodes: usize,
    a: &IntervalAdj,
    b: &DualAdj,
    current: &CsrAdjacency,
    epochs: &[EpochId],
) {
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    // Every 64th node, plus a handful of tail nodes.
    let samples: Vec<u32> = (0..num_nodes)
        .step_by(64)
        .chain([num_nodes.saturating_sub(1)])
        .map(|i| i as u32)
        .collect();
    for src in samples {
        let mut cur: Vec<u32> = current.neighbors(src).to_vec();
        cur.sort_unstable();
        let mut packed: Vec<u32> = a.current_neighbors(src).to_vec();
        packed.sort_unstable();
        assert_eq!(
            packed, cur,
            "A packed-prefix current != production CSR at src={src}"
        );
        assert_eq!(
            b.current.neighbors(src),
            current.neighbors(src),
            "B current CSR != production CSR at src={src}"
        );
        for &epoch in epochs {
            a.neighbors_at(src, epoch, &mut buf_a);
            b.neighbors_at(src, epoch, &mut buf_b);
            buf_a.sort_unstable();
            buf_b.sort_unstable();
            assert_eq!(buf_a, buf_b, "A/B as-of disagree src={src} epoch={epoch:?}");
        }
    }
}

#[inline]
fn expand_current_csr(csr: &CsrAdjacency) -> u64 {
    let mut acc = 0u64;
    let n = csr.num_nodes() as u32;
    for src in 0..n {
        for &t in csr.neighbors(src) {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn expand_a_packed(a: &IntervalAdj) -> u64 {
    let mut acc = 0u64;
    let n = a.num_nodes() as u32;
    for src in 0..n {
        for &t in a.current_neighbors(src) {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn expand_a_filter_open(a: &IntervalAdj, buf: &mut Vec<u32>) -> u64 {
    let mut acc = 0u64;
    let n = a.num_nodes() as u32;
    for src in 0..n {
        a.current_neighbors_filter_open(src, buf);
        for &t in buf.iter() {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn expand_asof_a(a: &IntervalAdj, epoch: EpochId, buf: &mut Vec<u32>) -> u64 {
    let mut acc = 0u64;
    let n = a.num_nodes() as u32;
    for src in 0..n {
        a.neighbors_at(src, epoch, buf);
        for &t in buf.iter() {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn expand_asof_b(b: &DualAdj, epoch: EpochId, buf: &mut Vec<u32>) -> u64 {
    let mut acc = 0u64;
    let n = b.current.num_nodes() as u32;
    for src in 0..n {
        b.neighbors_at(src, epoch, buf);
        for &t in buf.iter() {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn expand_random(a: &IntervalAdj, hops: &[(u32, EpochId)], buf: &mut Vec<u32>) -> u64 {
    let mut acc = 0u64;
    for &(src, epoch) in hops {
        a.neighbors_at(src, epoch, buf);
        acc = acc.wrapping_add(buf.len() as u64);
        for &t in buf.iter() {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn expand_random_b(b: &DualAdj, hops: &[(u32, EpochId)], buf: &mut Vec<u32>) -> u64 {
    let mut acc = 0u64;
    for &(src, epoch) in hops {
        b.neighbors_at(src, epoch, buf);
        acc = acc.wrapping_add(buf.len() as u64);
        for &t in buf.iter() {
            acc = acc.wrapping_add(u64::from(t));
        }
    }
    acc
}

#[inline]
fn scrub_edges_a(a: &IntervalAdj, epoch: EpochId) -> usize {
    if epoch == EpochId::PENDING {
        return a
            .open_ends
            .iter()
            .zip(a.offsets.iter())
            .map(|(end, start)| (*end - *start) as usize)
            .sum();
    }
    a.validity.iter().filter(|iv| iv.contains(epoch)).count()
}

#[inline]
fn scrub_edges_b(b: &DualAdj, epoch: EpochId) -> usize {
    if epoch == EpochId::PENDING {
        return b.current.num_edges();
    }
    let open = b.current_from.iter().filter(|&&from| from <= epoch).count();
    let closed = b
        .closed_validity
        .iter()
        .filter(|iv| iv.contains(epoch))
        .count();
    open + closed
}

#[inline]
fn scrub_nodes(col: &TemporalColumn, epoch: EpochId) -> i64 {
    // 3 physical rows per logical node (mirrors TemporalColumn range as-of).
    let nodes = col.len() / 3;
    let mut sum = 0i64;
    for n in 0..nodes {
        if let Some(Value::Int64(v)) = col.value_in_range_as_of(n * 3, 3, epoch) {
            sum = sum.wrapping_add(v);
        }
    }
    sum
}

#[inline]
fn frame_a(fix: &Fixture, epoch: EpochId, buf: &mut Vec<u32>) -> u64 {
    let nodes = scrub_nodes(&fix.node_col, epoch);
    let hops = expand_asof_a(&fix.a, epoch, buf);
    let edges = scrub_edges_a(&fix.a, epoch);
    (nodes as u64).wrapping_add(hops).wrapping_add(edges as u64)
}

#[inline]
fn frame_b(fix: &Fixture, epoch: EpochId, buf: &mut Vec<u32>) -> u64 {
    let nodes = scrub_nodes(&fix.node_col, epoch);
    let hops = expand_asof_b(&fix.b, epoch, buf);
    let edges = scrub_edges_b(&fix.b, epoch);
    (nodes as u64).wrapping_add(hops).wrapping_add(edges as u64)
}

fn configure(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(12);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
}

/// Wall-clock pin used in the decision record. Criterion `--quick` first-function
/// samples on this host were too noisy to trust for the 10% current-CSR gate.
fn pin_ms(label: &str, warmup: u32, iters: u32, mut f: impl FnMut()) {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ms = start.elapsed().as_secs_f64() * 1_000.0 / f64::from(iters);
    eprintln!("[pin] {label}: {ms:.3} ms/iter (warmup={warmup} iters={iters})");
}

fn bench_current_320k(c: &mut Criterion) {
    // ~320k open provenance edges, 80k nodes, degree 4. Sacred constraint
    // fixture: current 1-hop must stay within 10% of today's CSR.
    let fix = build_fixture("prov_320k", 80_000, 4);
    assert_eq!(fix.num_open, 320_000, "320k current-edge fixture");
    let epoch = EpochId::new(25);
    pin_ms("320k/csr_today", 32, 200, || {
        black_box(expand_current_csr(&fix.current));
    });
    pin_ms("320k/A_packed_prefix", 32, 200, || {
        black_box(expand_a_packed(&fix.a));
    });
    let mut pin_buf = Vec::with_capacity(16);
    pin_ms("320k/A_filter_is_open", 32, 200, || {
        black_box(expand_a_filter_open(&fix.a, &mut pin_buf));
    });
    pin_ms("320k/B_untouched_csr", 32, 200, || {
        black_box(expand_current_csr(&fix.b.current));
    });
    pin_ms("320k/A_asof_e25", 16, 80, || {
        black_box(expand_asof_a(&fix.a, epoch, &mut pin_buf));
    });
    pin_ms("320k/B_asof_e25", 16, 80, || {
        black_box(expand_asof_b(&fix.b, epoch, &mut pin_buf));
    });
    pin_ms("320k/C_snapshot_e25", 16, 80, || {
        black_box(expand_current_csr(&fix.c.csrs[1]));
    });
    pin_ms("320k/A_random_10k", 16, 80, || {
        black_box(expand_random(&fix.a, &fix.sample_hops, &mut pin_buf));
    });
    pin_ms("320k/B_random_10k", 16, 80, || {
        black_box(expand_random_b(&fix.b, &fix.sample_hops, &mut pin_buf));
    });

    let mut group = c.benchmark_group("current_1hop_320k");
    configure(&mut group);

    group.bench_function("csr_today", |ben| {
        ben.iter(|| black_box(expand_current_csr(black_box(&fix.current))));
    });
    group.bench_function("A_packed_prefix", |ben| {
        ben.iter(|| black_box(expand_a_packed(black_box(&fix.a))));
    });
    group.bench_function("A_filter_is_open", |ben| {
        let mut buf = Vec::with_capacity(16);
        ben.iter(|| black_box(expand_a_filter_open(black_box(&fix.a), &mut buf)));
    });
    group.bench_function("B_untouched_csr", |ben| {
        ben.iter(|| black_box(expand_current_csr(black_box(&fix.b.current))));
    });
    group.finish();

    let mut asof = c.benchmark_group("asof_1hop_320k");
    configure(&mut asof);
    let epoch = EpochId::new(25);
    asof.bench_function("A_filter_intervals", |ben| {
        let mut buf = Vec::with_capacity(16);
        ben.iter(|| black_box(expand_asof_a(black_box(&fix.a), black_box(epoch), &mut buf)));
    });
    asof.bench_function("B_merge_csr_closed", |ben| {
        let mut buf = Vec::with_capacity(16);
        ben.iter(|| black_box(expand_asof_b(black_box(&fix.b), black_box(epoch), &mut buf)));
    });
    asof.bench_function("C_snapshot_csr", |ben| {
        let csr = &fix.c.csrs[1]; // epoch 25
        ben.iter(|| black_box(expand_current_csr(black_box(csr))));
    });
    asof.bench_function("A_random_epochs_10k_hops", |ben| {
        let mut buf = Vec::with_capacity(16);
        ben.iter(|| black_box(expand_random(black_box(&fix.a), &fix.sample_hops, &mut buf)));
    });
    asof.bench_function("B_random_epochs_10k_hops", |ben| {
        let mut buf = Vec::with_capacity(16);
        ben.iter(|| {
            black_box(expand_random_b(
                black_box(&fix.b),
                &fix.sample_hops,
                &mut buf,
            ))
        });
    });
    asof.finish();
}

fn bench_entity_200k(c: &mut Criterion) {
    // 200k-entity fixture: degree 2 → 400k open + 400k closed. Frame budget
    // is 16 ms for node scrub + as-of expand + edge scrub.
    let fix = build_fixture("entity_200k", 200_000, 2);
    let epoch = EpochId::new(25);
    pin_ms("200k/node_scrub", 16, 80, || {
        black_box(scrub_nodes(&fix.node_col, epoch));
    });
    let mut pin_buf = Vec::with_capacity(8);
    pin_ms("200k/A_asof_expand", 16, 80, || {
        black_box(expand_asof_a(&fix.a, epoch, &mut pin_buf));
    });
    pin_ms("200k/B_asof_expand", 16, 80, || {
        black_box(expand_asof_b(&fix.b, epoch, &mut pin_buf));
    });
    pin_ms("200k/A_edge_scrub", 16, 80, || {
        black_box(scrub_edges_a(&fix.a, epoch));
    });
    pin_ms("200k/B_edge_scrub", 16, 80, || {
        black_box(scrub_edges_b(&fix.b, epoch));
    });
    pin_ms("200k/A_frame", 16, 40, || {
        black_box(frame_a(&fix, epoch, &mut pin_buf));
    });
    pin_ms("200k/B_frame", 16, 40, || {
        black_box(frame_b(&fix, epoch, &mut pin_buf));
    });
    pin_ms("200k/A_random_10k", 16, 80, || {
        black_box(expand_random(&fix.a, &fix.sample_hops, &mut pin_buf));
    });

    let mut group = c.benchmark_group("asof_frame_200k");
    configure(&mut group);

    group.bench_function("node_scrub_only", |ben| {
        ben.iter(|| black_box(scrub_nodes(black_box(&fix.node_col), black_box(epoch))));
    });
    group.bench_function("A_asof_expand", |ben| {
        let mut buf = Vec::with_capacity(8);
        ben.iter(|| black_box(expand_asof_a(black_box(&fix.a), black_box(epoch), &mut buf)));
    });
    group.bench_function("B_asof_expand", |ben| {
        let mut buf = Vec::with_capacity(8);
        ben.iter(|| black_box(expand_asof_b(black_box(&fix.b), black_box(epoch), &mut buf)));
    });
    group.bench_function("A_edge_scrub", |ben| {
        ben.iter(|| black_box(scrub_edges_a(black_box(&fix.a), black_box(epoch))));
    });
    group.bench_function("B_edge_scrub", |ben| {
        ben.iter(|| black_box(scrub_edges_b(black_box(&fix.b), black_box(epoch))));
    });
    group.bench_function("A_frame_node_plus_expand_plus_scrub", |ben| {
        let mut buf = Vec::with_capacity(8);
        ben.iter(|| black_box(frame_a(black_box(&fix), black_box(epoch), &mut buf)));
    });
    group.bench_function("B_frame_node_plus_expand_plus_scrub", |ben| {
        let mut buf = Vec::with_capacity(8);
        ben.iter(|| black_box(frame_b(black_box(&fix), black_box(epoch), &mut buf)));
    });
    group.bench_function("A_random_epochs_10k_hops", |ben| {
        let mut buf = Vec::with_capacity(8);
        ben.iter(|| black_box(expand_random(black_box(&fix.a), &fix.sample_hops, &mut buf)));
    });
    group.finish();
}

/// Overlay writes during as-of *reads*.
///
/// Task 4 landed `neighbors_at_epoch` on RelTable / CompactStore / LayeredStore
/// (PENDING = derived current CSR; else packed validity). This overlay bench
/// still measures current CSR hops and layered `edges_from` / `get_edge_at_epoch`
/// under concurrent overlay property writes (the original production surface).
fn bench_overlay(c: &mut Criterion) {
    const N: usize = 8_000;
    let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
    let layered = LayeredStore::new(empty, (N as u64) + 10, (N as u64) * 4).unwrap();
    let overlay = layered.overlay_store();
    let mut ids = Vec::with_capacity(N);
    for _ in 0..N {
        ids.push(overlay.create_node(&["Entity"]));
    }
    for i in 0..N {
        let src = ids[i];
        let dst = ids[(i + 1) % N];
        overlay.create_edge(src, dst, "PROV");
    }
    overlay.set_epoch(EpochId::new(10));
    layered.merge_overlay_in_place().unwrap();

    // A handful of hot overlay edges so writes have somewhere to land without
    // rebuilding the base. create_edge on LayeredStore promotes endpoints.
    let mut hot_edges = Vec::new();
    for i in 0..64 {
        let eid = layered.create_edge(ids[i], ids[i + 1], "HOT");
        layered.set_edge_property(eid, "w", Value::Int64(1));
        hot_edges.push(eid);
    }
    let epoch = EpochId::new(10);
    let store = Arc::new(layered);
    let stop = Arc::new(AtomicBool::new(false));
    let writes = Arc::new(AtomicU64::new(0));
    let w_store = Arc::clone(&store);
    let w_stop = Arc::clone(&stop);
    let w_count = Arc::clone(&writes);
    let hot = hot_edges.clone();
    let writer = thread::spawn(move || {
        let mut tick = 0i64;
        while !w_stop.load(Ordering::Relaxed) {
            let eid = hot[(tick as usize) % hot.len()];
            w_store.set_edge_property(eid, "w", Value::Int64(tick));
            tick += 1;
            w_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    let base = store.base_store_arc();
    let mut group = c.benchmark_group("overlay_writes_during_reads");
    configure(&mut group);
    group.bench_function("base_current_neighbors", |ben| {
        ben.iter(|| {
            let mut acc = 0usize;
            for id in ids.iter().step_by(8) {
                acc += base.neighbors(*id, Direction::Outgoing).len();
            }
            black_box(acc)
        });
    });
    group.bench_function("layered_edges_from_current", |ben| {
        ben.iter(|| {
            let mut acc = 0usize;
            for id in ids.iter().step_by(8) {
                acc += store.edges_from(*id, Direction::Outgoing).len();
            }
            black_box(acc)
        });
    });
    group.bench_function("layered_get_edge_at_epoch", |ben| {
        ben.iter(|| {
            let mut acc = 0usize;
            for eid in &hot_edges {
                if store.get_edge_at_epoch(*eid, epoch).is_some() {
                    acc += 1;
                }
            }
            black_box(acc)
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    writer.join().expect("overlay writer");
    eprintln!(
        "[overlay] concurrent set_edge_property writes during read benches: {}",
        writes.load(Ordering::Relaxed)
    );
}

criterion_group!(
    benches,
    bench_current_320k,
    bench_entity_200k,
    bench_overlay
);
criterion_main!(benches);
