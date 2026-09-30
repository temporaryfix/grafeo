//! Packed Ring persistence throughput.

#![allow(missing_docs)]

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use grafeo_common::storage::section::Section;
use grafeo_core::graph::rdf::{RdfStore, Term, Triple};
use grafeo_core::index::ring::{RdfRingSection, TripleRing};

const SUBJECTS: usize = 2_000;
const PREDICATES: usize = 5;
const TRIPLES: usize = SUBJECTS * PREDICATES;

fn fixture_store() -> Arc<RdfStore> {
    let triples: Vec<_> = (0..SUBJECTS)
        .flat_map(|subject| {
            (0..PREDICATES).map(move |predicate| {
                Triple::new(
                    Term::iri(format!("https://example.test/subject/{subject}")),
                    Term::iri(format!("https://example.test/predicate/{predicate}")),
                    Term::literal(format!("value-{subject}-{predicate}")),
                )
            })
        })
        .collect();
    let store = Arc::new(RdfStore::new());
    store.bulk_load(triples.iter().cloned());
    // Primary hash-table iteration randomizes dictionary IDs. Fix the input
    // order so every process measures the same packed persistence fixture.
    store.set_ring(TripleRing::from_triples(triples.into_iter()));
    store
}

fn bench_ring_persistence(c: &mut Criterion) {
    let store = fixture_store();
    let section = RdfRingSection::new(Arc::clone(&store));
    let bytes = section.serialize().expect("serialize fixture Ring");
    eprintln!(
        "Ring fixture: {TRIPLES} triples, {} bytes, BLAKE3 {}",
        bytes.len(),
        blake3::hash(&bytes)
    );

    let mut group = c.benchmark_group("ring_persistence_10k_triples");
    group.throughput(Throughput::Elements(TRIPLES as u64));
    group.bench_function("encode", |bencher| {
        bencher.iter(|| black_box(section.serialize().expect("serialize Ring")));
    });
    group.bench_function("decode", |bencher| {
        bencher.iter(|| {
            let target = Arc::new(RdfStore::new());
            let mut target_section = RdfRingSection::new(target);
            target_section
                .deserialize(black_box(&bytes))
                .expect("deserialize Ring");
            black_box(target_section);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_ring_persistence);
criterion_main!(benches);
