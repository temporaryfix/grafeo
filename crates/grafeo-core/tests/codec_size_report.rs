//! Compression-ratio measurements for the Plan 2 codecs against the
//! pre-existing alternatives.
//!
//! Not a regression test — a measurement printout. Run with
//! `cargo test -p grafeo-core --test codec_size_report -- --nocapture`.
//!
//! Datasets are synthetic but shaped to look like real workloads. Results
//! depend on data characteristics and should not be quoted as absolute
//! claims; the integration measurement against Nota's actual data is the
//! authoritative source.

use grafeo_core::codec::{DictionaryBuilder, FsstCodec, WebGraphBuilder};
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo)
    }
}

#[test]
fn measure_codec_sizes() {
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  Plan 2 codec compression measurements (synthetic datasets)");
    println!("════════════════════════════════════════════════════════════════");

    measure_fsst_random_ascii();
    measure_fsst_name_like();
    measure_fsst_url_like();
    measure_webgraph_random();
    measure_webgraph_power_law();
    measure_rabitq_by_construction();

    println!("════════════════════════════════════════════════════════════════");
}

// ── FSST -------------------------------------------------------------

fn dict_size_bytes(strings: &[Vec<u8>]) -> usize {
    // DictionaryEncoding stores each unique string once + 4 bytes per row.
    let mut builder = DictionaryBuilder::new();
    for s in strings {
        if let Ok(utf8) = std::str::from_utf8(s) {
            builder.add(utf8);
        }
    }
    let dict: grafeo_core::codec::DictionaryEncoding = builder.build();
    let dict_bytes: usize = dict
        .dictionary()
        .iter()
        .map(|s: &Arc<str>| s.len())
        .sum();
    let codes_bytes = dict.codes_bytes().len();
    // Plus a small fixed header (4-byte counts × 2 = 8 bytes; rounded up).
    dict_bytes + codes_bytes + 16
}

fn report_fsst(label: &str, strings: &[Vec<u8>]) {
    let raw: usize = strings.iter().map(|s| s.len()).sum::<usize>() + strings.len() * 4; // 4-byte offset per string

    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let fsst = FsstCodec::build(&refs);
    let fsst_size = fsst.to_bytes().len();

    let dict_size = dict_size_bytes(strings);

    println!();
    println!("─── FSST: {label} ({} strings) ───", strings.len());
    println!("  raw (bytes + offsets):    {raw:>9}");
    println!("  DictionaryEncoding:       {dict_size:>9}  ({:.2}× vs raw)", raw as f64 / dict_size as f64);
    println!("  FSST blob:                {fsst_size:>9}  ({:.2}× vs raw, {:.2}× vs Dict)",
        raw as f64 / fsst_size as f64,
        dict_size as f64 / fsst_size as f64);
}

fn measure_fsst_random_ascii() {
    let mut rng = Rng::new(42);
    let strings: Vec<Vec<u8>> = (0..1000)
        .map(|_| {
            let len = rng.range(5, 20) as usize;
            (0..len).map(|_| (rng.range(0, 26) as u8) + b'a').collect()
        })
        .collect();
    report_fsst("random lowercase 5-20 chars", &strings);
}

fn measure_fsst_name_like() {
    // Names with realistic repetition: 50 first names × 50 last names, 2000 rows
    // (≈ 1.5 rows per name pair).
    let firsts = [
        "Vincent", "Mia", "Butch", "Jules", "Marsellus", "Honey", "Pumpkin",
        "Lance", "Jody", "Esmeralda", "Winston", "Captain", "Floyd", "Pete",
        "Marvin", "Brett", "Roger", "Ringo", "Yolanda", "Raquel",
        "Antwan", "Jimmie", "Bonnie", "Maynard", "Zed", "Trudi", "Fabienne",
        "Paul", "English", "Wolf", "Smith", "Jones", "Brown", "Lee",
        "Walker", "Lewis", "Robinson", "Green", "Adams", "Nelson",
        "Carter", "Mitchell", "Roberts", "Phillips", "Campbell", "Parker",
        "Evans", "Edwards", "Collins", "Stewart",
    ];
    let lasts = [
        "Vega", "Wallace", "Coolidge", "Winnfield", "Bunny", "Marvin",
        "Wolf", "Maximus", "Cooper", "Lockhart", "Spencer", "Hayes",
        "Bryant", "Henderson", "Murphy", "Sullivan", "Foster", "Webb",
        "Hardy", "Stokes", "Russo", "Ferraro", "Dimitri", "Kowalski",
        "Petrov", "Yamamoto", "Chen", "Patel", "Singh", "Tanaka",
        "Garcia", "Lopez", "Hernandez", "Gomez", "Martinez", "Rivera",
        "Schmidt", "Müller", "Schneider", "Weber", "Fischer", "Becker",
        "Kowalski", "Nowak", "Wójcik", "Krawczyk", "Lewandowski",
        "Andersen", "Olsen", "Johansen",
    ];
    let mut rng = Rng::new(99);
    let strings: Vec<Vec<u8>> = (0..2000)
        .map(|_| {
            let f = firsts[(rng.next() as usize) % firsts.len()];
            let l = lasts[(rng.next() as usize) % lasts.len()];
            format!("{f} {l}").into_bytes()
        })
        .collect();
    report_fsst("realistic names (50 first × 50 last, 2000 rows)", &strings);
}

fn measure_fsst_url_like() {
    let hosts = [
        "api.example.com", "cdn.example.com", "assets.example.com",
        "static.example.com", "www.acme.com", "store.acme.com",
    ];
    let paths = [
        "/v1/users/", "/v1/posts/", "/v1/comments/", "/v2/profile/",
        "/static/images/", "/assets/css/", "/api/search?q=",
    ];
    let mut rng = Rng::new(77);
    let strings: Vec<Vec<u8>> = (0..1000)
        .map(|_| {
            let host = hosts[(rng.next() as usize) % hosts.len()];
            let path = paths[(rng.next() as usize) % paths.len()];
            let id = rng.range(1000, 999999);
            format!("https://{host}{path}{id}").into_bytes()
        })
        .collect();
    report_fsst("URL-like (shared host + path prefixes)", &strings);
}

// ── WebGraph ---------------------------------------------------------

fn report_webgraph(label: &str, num_nodes: u64, edges: &[(u64, u64)]) {
    let mut b = WebGraphBuilder::new(num_nodes);
    for &(s, d) in edges {
        b.add_edge(s, d).unwrap();
    }
    let codec = b.build();
    let blob = codec.to_bytes();
    let blob_size = blob.len();

    // Raw adjacency: per edge, store the destination as u64 = 8 bytes.
    // Plus a u64 offset per node so successors() can index.
    let raw_size = (edges.len() * 8) + ((num_nodes as usize + 1) * 8);

    let bits_per_edge_blob = (blob_size as f64 * 8.0) / edges.len() as f64;
    let bits_per_edge_raw = (raw_size as f64 * 8.0) / edges.len() as f64;

    println!();
    println!("─── WebGraph: {label} ({} nodes, {} edges) ───", num_nodes, edges.len());
    println!("  raw (u64 dst + u64 offsets): {raw_size:>9} bytes  ({bits_per_edge_raw:>5.1} bits/edge)");
    println!("  WebGraph blob:               {blob_size:>9} bytes  ({bits_per_edge_blob:>5.1} bits/edge,  {:.2}× vs raw)",
        raw_size as f64 / blob_size as f64);
}

fn measure_webgraph_random() {
    let n: u64 = 1000;
    let mut rng = Rng::new(7);
    let mut edges: Vec<(u64, u64)> = (0..10_000)
        .map(|_| (rng.next() % n, rng.next() % n))
        .collect();
    edges.sort_unstable();
    edges.dedup();
    report_webgraph("random graph", n, &edges);
}

fn measure_webgraph_power_law() {
    // Power-law-ish: a few high-degree nodes, many low-degree.
    let n: u64 = 1000;
    let mut rng = Rng::new(11);
    let mut edges: Vec<(u64, u64)> = Vec::new();
    for src in 0..n {
        // out-degree biased toward small values
        let degree_log = rng.next() % 10; // 0..=9
        let degree = 1u64 << degree_log; // 1, 2, 4, 8, ..., 512
        let degree = degree.min(50); // cap
        for _ in 0..degree {
            // Destinations biased toward low ids (popular nodes).
            let r = rng.next();
            let dst = if r % 4 == 0 {
                r % 50 // 25% to top-50 popular nodes
            } else {
                r % n
            };
            edges.push((src, dst));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    report_webgraph("power-law-ish (popular-node bias)", n, &edges);
}

// ── RaBitQ -----------------------------------------------------------

fn measure_rabitq_by_construction() {
    println!();
    println!("─── RaBitQ: by construction ───");
    println!("  256-dim f32 vector:        1024 bytes");
    println!("  RaBitQ bit code:             32 bytes      (32.0× — fixed)");
    println!("  + correction factors:         8 bytes      (25.6× including factors)");
    println!("  + int8 rerank code:         256 bytes      (3.6× total with rerank)");
    println!();
    println!("  The rerank int8 is what TwoStageVectorIndex stores per vector to");
    println!("  achieve the 1.000 recall@10 measured in the rabitq_vs_pq bench. A");
    println!("  RaBitQ-only index (no rerank) would be a 32× compression at the");
    println!("  recall cost documented in the SIGMOD 2024 paper (~80% recall).");
}
