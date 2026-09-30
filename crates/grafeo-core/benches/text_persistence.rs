//! Text-index persistence throughput.

#![allow(missing_docs)]

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use grafeo_common::storage::section::Section;
use grafeo_common::types::{GraphPath, NodeId};
use grafeo_core::graph::lpg::PhysicalIndexKey;
use grafeo_core::index::text::{BM25Config, InvertedIndex, TextIndexSection};
use parking_lot::RwLock;

const DOCUMENTS: u64 = 10_000;

fn fixture_index() -> Arc<RwLock<InvertedIndex>> {
    let mut index = InvertedIndex::new(BM25Config::default());
    for document in 0..DOCUMENTS {
        index.insert(
            NodeId::new(document),
            &format!(
                "graph database persistence document {document} cohort {}",
                document % 100
            ),
        );
    }
    Arc::new(RwLock::new(index))
}

fn bench_text_persistence(c: &mut Criterion) {
    let source = fixture_index();
    let section = TextIndexSection::new(vec![(
        PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
        source,
    )]);
    let bytes = section.serialize().expect("serialize fixture Text index");

    let mut group = c.benchmark_group("text_persistence_10k_documents");
    group.throughput(Throughput::Elements(DOCUMENTS));
    group.bench_function("encode", |bencher| {
        bencher.iter(|| black_box(section.serialize().expect("serialize Text index")));
    });
    group.bench_function("decode", |bencher| {
        bencher.iter(|| {
            let target = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
            let mut target_section = TextIndexSection::for_unpublished_recovery(vec![(
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::clone(&target),
            )]);
            target_section
                .deserialize(black_box(&bytes))
                .expect("deserialize Text index");
            black_box(target);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_text_persistence);
criterion_main!(benches);
