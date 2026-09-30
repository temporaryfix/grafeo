// Real production-seam witnesses for owned generation publication.

struct GenerationDropProbe {
    layered: Arc<LayeredStore>,
    overlay: Arc<LpgStore>,
    #[cfg(feature = "text-index")]
    text_gates:
        Option<Arc<RwLock<Vec<std::sync::Weak<RwLock<crate::index::text::InvertedIndex>>>>>>,
    dropped: Arc<AtomicUsize>,
    guards_drained: Arc<AtomicBool>,
}

impl Drop for GenerationDropProbe {
    fn drop(&mut self) {
        let mutation = self.layered.merge_guard.try_write().is_some();
        let publication = self.layered.publication_guard.try_write().is_some();
        #[cfg(feature = "text-index")]
        let text = self.text_gates.as_ref().is_none_or(|gates| {
            gates.try_read().is_some_and(|gates| {
                gates.iter().all(|gate| {
                    gate.upgrade()
                        .is_some_and(|gate| gate.try_write().is_some())
                })
            })
        });
        #[cfg(not(feature = "text-index"))]
        let text = true;
        self.guards_drained.store(
            mutation && publication && self.overlay.generation_guards_drained_for_test() && text,
            Ordering::SeqCst,
        );
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn generation_totality_temporal_revision_exhaustion_is_an_error_before_external_prepare() {
    let layered = empty_layered();
    let original_base = layered.base_store_arc();
    let original_overlay = layered.overlay_store();
    let node = original_overlay.create_node(&["Document"]);
    original_overlay.set_node_property(node, "title", Value::from("retained"));
    layered
        .generation_revision
        .store(u64::MAX, Ordering::Release);
    let prepares = AtomicUsize::new(0);
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        layered.merge_overlay_temporal_with_publication(
            |base| {
                prepares.fetch_add(1, Ordering::SeqCst);
                Ok((base, ()))
            },
            |()| {
                publishes.fetch_add(1, Ordering::SeqCst);
            },
            |()| {
                rollbacks.fetch_add(1, Ordering::SeqCst);
            },
        )
    }));
    assert_eq!(prepares.load(Ordering::SeqCst), 0);
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    assert!(Arc::ptr_eq(&original_base, &layered.base_store_arc()));
    assert!(Arc::ptr_eq(&original_overlay, &layered.overlay_store()));
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        u64::MAX
    );
    assert_eq!(
        original_overlay.get_node_property(node, &PropertyKey::new("title")),
        Some(Value::from("retained"))
    );
    assert!(
        outcome.is_ok(),
        "revision exhaustion must return an error, not unwind"
    );
    assert!(outcome.unwrap().is_err());
}

#[test]
fn generation_totality_temporal_revision_last_value_publishes_once() {
    let layered = empty_layered();
    layered.overlay_store().create_node(&["Retained"]);
    layered
        .generation_revision
        .store(u64::MAX - 1, Ordering::Release);
    let prepares = AtomicUsize::new(0);
    layered
        .merge_overlay_temporal_with_publication(
            |base| {
                prepares.fetch_add(1, Ordering::SeqCst);
                Ok((base, ()))
            },
            |()| (),
            |()| (),
        )
        .unwrap();
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        u64::MAX
    );
    let base = layered.base_store_arc();
    let overlay = layered.overlay_store();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        layered.merge_overlay_temporal_with_publication(
            |base| {
                prepares.fetch_add(1, Ordering::SeqCst);
                Ok((base, ()))
            },
            |()| (),
            |()| (),
        )
    }));
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
    assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
    assert!(outcome.is_ok(), "MAX is terminal without wrap or panic");
    assert!(outcome.unwrap().is_err());
}

fn generation_receipt_fixture(
    mixed: bool,
) -> (LayeredStore, Vec<TransportEdgeReceipt>, WriteAuthority) {
    let source = Arc::new(LpgStore::new().unwrap());
    let src = source.create_node(&["Source"]);
    let dst = source.create_node(&["Destination"]);
    source.sync_epoch(EpochId::new(1));
    let receipts: Vec<_> = (0..4)
        .map(|i| {
            source
                .create_transport_edge_with_id(EdgeId::new(9_700_000 + i), src, dst, "CARRIED")
                .unwrap()
                .unwrap()
        })
        .collect();
    let layered = compact_transport_source(source);
    layered.overlay_store().sync_epoch(EpochId::new(2));
    for (i, receipt) in receipts.iter().enumerate() {
        if mixed && i % 2 == 0 {
            layered.set_edge_property(receipt.edge_id(), "promoted", Value::Bool(true));
        }
        assert!(layered.delete_edge(receipt.edge_id()));
        assert_eq!(
            layered
                .overlay_store()
                .all_known_edge_ids()
                .contains(&receipt.edge_id()),
            mixed && i % 2 == 0
        );
        assert!(receipt_belongs_to_current_overlay(&layered, receipt));
    }
    let owner = WriteAuthority::new();
    assert!(layered.overlay_store().seal_unframed_writes(&owner));
    (layered, receipts, owner)
}

fn assert_generation_revision_purge(mixed: bool, last_publication: bool) {
    let (layered, receipts, owner) = generation_receipt_fixture(mixed);
    let prepares = AtomicUsize::new(0);
    layered.generation_revision.store(
        if last_publication {
            u64::MAX - 1
        } else {
            u64::MAX
        },
        Ordering::Release,
    );
    if last_publication {
        assert!(
            with_authority(&owner, || layered.purge_transport_extract_edges(
                &[&receipts[0], &receipts[1]],
                &owner,
                |base| {
                    prepares.fetch_add(1, Ordering::SeqCst);
                    Ok((base, ()))
                },
                |()| (),
                |()| (),
            ))
            .unwrap()
        );
        assert!(!receipt_belongs_to_current_overlay(&layered, &receipts[0]));
        assert!(!receipt_belongs_to_current_overlay(&layered, &receipts[1]));
    }
    let original_base = layered.base_store_arc();
    let original_overlay = layered.overlay_store();
    for receipt in &receipts[2..] {
        assert!(receipt_belongs_to_current_overlay(&layered, receipt));
        assert_eq!(layered.complete_edge_history(receipt.edge_id()).len(), 1);
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || {
            layered.purge_transport_extract_edges(
                &[&receipts[2], &receipts[3]],
                &owner,
                |base| {
                    prepares.fetch_add(1, Ordering::SeqCst);
                    Ok((base, ()))
                },
                |()| (),
                |()| (),
            )
        })
    }));
    assert_eq!(
        prepares.load(Ordering::SeqCst),
        usize::from(last_publication),
        "exhaustion must precede external preparation on still-qualified receipts"
    );
    assert!(
        outcome.is_ok(),
        "exhaustion is a build error, never unwind or qualification miss"
    );
    assert!(outcome.unwrap().is_err());
    assert!(Arc::ptr_eq(&original_base, &layered.base_store_arc()));
    assert!(Arc::ptr_eq(&original_overlay, &layered.overlay_store()));
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        u64::MAX
    );
    for receipt in &receipts[2..] {
        assert!(receipt_belongs_to_current_overlay(&layered, receipt));
        assert_eq!(layered.complete_edge_history(receipt.edge_id()).len(), 1);
    }
}

#[test]
fn generation_totality_base_only_revision_exhaustion_precedes_external() {
    assert_generation_revision_purge(false, false);
}

#[test]
fn generation_totality_mixed_purge_revision_exhaustion_precedes_external() {
    assert_generation_revision_purge(true, false);
}

#[test]
fn generation_totality_base_only_revision_last_value_publishes_once() {
    assert_generation_revision_purge(false, true);
}

#[test]
fn generation_totality_mixed_purge_revision_last_value_publishes_once() {
    assert_generation_revision_purge(true, true);
}

#[test]
fn generation_totality_invalid_successor_defers_external() {
    let layered = Arc::new(empty_layered());
    let original_overlay = layered.overlay_store();
    let original_base = layered.base_store_arc();
    let dropped = Arc::new(AtomicUsize::new(0));
    let guards_drained = Arc::new(AtomicBool::new(false));
    let publishes = AtomicUsize::new(0);
    let retirements = GenerationRetirements::new();
    let mut mutations = MutationWriteScope::enter(&layered);
    let topology = original_overlay.pin_named_graph_topology();
    let transition = original_overlay
        .pin_exclusive_unframed_transition()
        .unwrap();
    let result = layered.install_with_pinned_empty_successor_and_publication(
        PinnedEmptySuccessorPublication {
            current_overlay: &original_overlay,
            representation_topology: Some(&topology),
            transition: &transition,
            mutation: &mut mutations,
            retirements: &retirements,
        },
        |_scope, _successor| {
            Ok((
                PreparedLayerGeneration::empty(
                    Arc::clone(&original_base),
                    Arc::new(LpgStore::new().unwrap()),
                ),
                GenerationDropProbe {
                    layered: Arc::clone(&layered),
                    overlay: Arc::clone(&original_overlay),
                    #[cfg(feature = "text-index")]
                    text_gates: None,
                    dropped: Arc::clone(&dropped),
                    guards_drained: Arc::clone(&guards_drained),
                },
            ))
        },
        |probe| {
            publishes.fetch_add(1, Ordering::SeqCst);
            probe
        },
        |probe| probe,
    );
    drop(transition);
    drop(topology);
    drop(mutations);
    drop(retirements);
    assert!(result.is_err());
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(
        guards_drained.load(Ordering::SeqCst),
        "owned prepared metadata retired beneath outer guards"
    );
    assert!(Arc::ptr_eq(&original_overlay, &layered.overlay_store()));
    assert!(Arc::ptr_eq(&original_base, &layered.base_store_arc()));
    layered.merge_overlay_temporal().unwrap();
}

fn assert_generation_late_resident_retirement(unwind: bool) {
    use crate::graph::write_permit::with_authority;
    let (layered, receipts, owner) = generation_receipt_fixture(true);
    let layered = Arc::new(layered);
    let overlay = layered.overlay_store();
    let base = layered.base_store_arc();
    let revision = layered.generation_revision.load(Ordering::Acquire);
    let histories: Vec<_> = receipts
        .iter()
        .map(|receipt| layered.edge_full_history(receipt.edge_id()))
        .collect();
    let dropped = Arc::new(AtomicUsize::new(0));
    let drained = Arc::new(AtomicBool::new(false));
    let prepares = AtomicUsize::new(0);
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let batch: Vec<_> = receipts.iter().collect();
    overlay.generation_purge_late_once_for_test(unwind);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || {
            layered.purge_transport_extract_edges(
                &batch,
                &owner,
                |base| {
                    prepares.fetch_add(1, Ordering::SeqCst);
                    Ok((
                        base,
                        GenerationDropProbe {
                            layered: Arc::clone(&layered),
                            overlay: Arc::clone(&overlay),
                            #[cfg(feature = "text-index")]
                            text_gates: None,
                            dropped: Arc::clone(&dropped),
                            guards_drained: Arc::clone(&drained),
                        },
                    ))
                },
                |probe| {
                    publishes.fetch_add(1, Ordering::SeqCst);
                    probe
                },
                |probe| {
                    rollbacks.fetch_add(1, Ordering::SeqCst);
                    probe
                },
            )
        })
    }));
    if unwind {
        assert_eq!(
            result.unwrap_err().downcast_ref::<&str>(),
            Some(&"generation late resident qualification unwind")
        );
    } else {
        assert!(!result.unwrap().unwrap());
    }
    assert_eq!(overlay.generation_purge_late_hits_for_test(), 1);
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(drained.load(Ordering::SeqCst));
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        revision
    );
    assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
    assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
    for (receipt, history) in receipts.iter().zip(histories) {
        let actual = layered.edge_full_history(receipt.edge_id());
        assert_eq!(
            (
                actual.src,
                actual.dst,
                actual.edge_type,
                actual.lifetimes,
                actual.properties
            ),
            (
                history.src,
                history.dst,
                history.edge_type,
                history.lifetimes,
                history.properties
            )
        );
        assert!(overlay.transport_receipt_belongs_to_current_incarnation(receipt));
    }
    assert!(with_authority(&owner, || layered
        .purge_transport_extract_edges(&batch, &owner, |base| Ok((base, ())), |()| (), |()| ())
        .unwrap()));
    assert_eq!(overlay.generation_purge_late_hits_for_test(), 1);
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        revision + 1
    );
}

#[test]
fn generation_totality_resident_late_rejection_retires_after_guards() {
    assert_generation_late_resident_retirement(false);
}

#[test]
fn generation_totality_resident_late_unwind_retires_after_guards() {
    assert_generation_late_resident_retirement(true);
}

fn assert_generation_reseed_retirement(unwind: bool) {
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    let layered = Arc::new(empty_layered());
    let overlay = layered.overlay_store();
    let base = layered.base_store_arc();
    overlay.create_property_index("score");
    let node = overlay.create_node(&["Item"]);
    overlay.set_node_property_at_epoch(node, "score", Value::Int64(10), EpochId::new(10));
    overlay.set_node_property_at_epoch(node, "score", Value::Int64(30), EpochId::new(30));
    overlay.set_epoch(EpochId::new(30));
    let history = layered.node_property_full_history(node);
    assert!(
        history
            .iter()
            .any(|(_, values)| values.iter().any(|(epoch, _)| *epoch > EpochId::new(25)))
    );
    let owner = WriteAuthority::new();
    assert!(overlay.seal_unframed_writes(&owner));
    let revision = layered.generation_revision.load(Ordering::Acquire);
    let prepares = AtomicUsize::new(0);
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let drained = Arc::new(AtomicBool::new(false));
    layered
        .generation_reseed_action
        .store(if unwind { 2 } else { 1 }, Ordering::SeqCst);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || {
            layered.merge_overlay_temporal_retaining_with_publication(
                Some(EpochId::new(25)),
                false,
                |base| {
                    prepares.fetch_add(1, Ordering::SeqCst);
                    Ok((
                        base,
                        GenerationDropProbe {
                            layered: Arc::clone(&layered),
                            overlay: Arc::clone(&overlay),
                            #[cfg(feature = "text-index")]
                            text_gates: None,
                            dropped: Arc::clone(&dropped),
                            guards_drained: Arc::clone(&drained),
                        },
                    ))
                },
                |probe| {
                    publishes.fetch_add(1, Ordering::SeqCst);
                    probe
                },
                |probe| {
                    rollbacks.fetch_add(1, Ordering::SeqCst);
                    probe
                },
            )
        })
    }));
    if unwind {
        assert_eq!(
            result.err().unwrap().downcast_ref::<&str>(),
            Some(&"generation retained-hot reseed unwind")
        );
    } else {
        assert!(
            matches!(result.unwrap().map_err(|message| message.into_reason()), Err(message) if message == "generation retained-hot reseed rejected")
        );
    }
    assert_eq!(layered.generation_reseed_hits.load(Ordering::SeqCst), 1);
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(drained.load(Ordering::SeqCst));
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        revision
    );
    assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
    assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
    assert_eq!(layered.node_property_full_history(node), history);
    assert!(overlay.has_property_index("score"));
    with_authority(&owner, || {
        layered.merge_overlay_temporal_retaining(EpochId::new(25))
    })
    .unwrap();
    assert_eq!(layered.generation_reseed_hits.load(Ordering::SeqCst), 1);
    assert!(layered.overlay_store().get_node(node).is_some());
    assert_eq!(
        layered.get_node_property(node, &PropertyKey::new("score")),
        Some(Value::Int64(30))
    );
}

#[test]
fn generation_totality_reseed_rejection_defers_external() {
    assert_generation_reseed_retirement(false);
}

#[test]
fn generation_totality_reseed_unwind_defers_external() {
    assert_generation_reseed_retirement(true);
}

#[test]
fn generation_totality_final_image_unwind_defers_external() {
    let layered = Arc::new(empty_layered());
    let overlay = layered.overlay_store();
    let base = layered.base_store_arc();
    let revision = layered.generation_revision.load(Ordering::Acquire);
    let prepares = AtomicUsize::new(0);
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let dropped = Arc::new(AtomicUsize::new(0));
    let drained = Arc::new(AtomicBool::new(false));
    layered
        .generation_image_validation_panic
        .store(true, Ordering::SeqCst);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        layered.merge_overlay_temporal_with_publication(
            |base| {
                prepares.fetch_add(1, Ordering::SeqCst);
                Ok((
                    base,
                    GenerationDropProbe {
                        layered: Arc::clone(&layered),
                        overlay: Arc::clone(&overlay),
                        #[cfg(feature = "text-index")]
                        text_gates: None,
                        dropped: Arc::clone(&dropped),
                        guards_drained: Arc::clone(&drained),
                    },
                ))
            },
            |probe| {
                publishes.fetch_add(1, Ordering::SeqCst);
                probe
            },
            |probe| {
                rollbacks.fetch_add(1, Ordering::SeqCst);
                probe
            },
        )
    }));
    assert_eq!(
        result.err().unwrap().downcast_ref::<&str>(),
        Some(&"generation final image validation unwind")
    );
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(drained.load(Ordering::SeqCst));
    assert_eq!(
        layered.generation_revision.load(Ordering::Acquire),
        revision
    );
    assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
    assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
    layered.merge_overlay_temporal().unwrap();
}

#[cfg(feature = "text-index")]
struct GenerationTokenizerProbe {
    targets: Arc<RwLock<Vec<std::sync::Weak<RwLock<crate::index::text::InvertedIndex>>>>>,
    dropped: Arc<AtomicUsize>,
    all_fences_drained: Arc<AtomicBool>,
}

#[cfg(feature = "text-index")]
impl crate::index::text::Tokenizer for GenerationTokenizerProbe {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }
}

#[cfg(feature = "text-index")]
impl Drop for GenerationTokenizerProbe {
    fn drop(&mut self) {
        let drained = self.targets.try_read().is_some_and(|targets| {
            targets.iter().all(|target| {
                target
                    .upgrade()
                    .is_none_or(|target| target.try_write().is_some())
            })
        });
        self.all_fences_drained.fetch_and(drained, Ordering::SeqCst);
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

/// A real clear on a different thread is sequenced after the original source
/// cut but before outer-bank retirement. Only weak diagnostic identities are
/// retained by the fixture; the production view anchors must keep payloads.
#[cfg(feature = "text-index")]
fn assert_generation_transfer_anchors(case: u8, retain_anchors: bool) {
    use crate::index::text::{BM25Config, InvertedIndex};
    let layered = Arc::new(empty_layered());
    let overlay = layered.overlay_store();
    let base = layered.base_store_arc();
    let targets = Arc::new(RwLock::new(Vec::new()));
    let tokenizer_drops = Arc::new(AtomicUsize::new(0));
    let tokenizer_drained = Arc::new(AtomicBool::new(true));
    for property in ["alpha", "beta"] {
        let caller = Arc::new(RwLock::new(InvertedIndex::with_tokenizer(
            BM25Config::default(),
            Box::new(GenerationTokenizerProbe {
                targets: Arc::clone(&targets),
                dropped: Arc::clone(&tokenizer_drops),
                all_fences_drained: Arc::clone(&tokenizer_drained),
            }),
        )));
        targets.write().push(Arc::downgrade(&caller));
        overlay.add_text_index("Document", property, caller);
    }
    targets
        .write()
        .extend(overlay.generation_text_targets_for_test());
    assert_eq!(targets.read().len(), 4);
    #[cfg(feature = "vector-index")]
    {
        use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};
        overlay.add_vector_index(
            "Document",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
                HnswConfig::new(2, DistanceMetric::Euclidean),
                17,
            ))),
        );
    }
    let dropped = Arc::new(AtomicUsize::new(0));
    let drained = Arc::new(AtomicBool::new(false));
    let prepares = AtomicUsize::new(0);
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let retirements = GenerationRetirements::<GenerationDropProbe, GenerationDropProbe>::new();
    let mut mutations = MutationWriteScope::enter(&layered);
    let topology = overlay.pin_named_graph_topology();
    let transition = overlay.pin_exclusive_unframed_transition().unwrap();
    if case == 3 {
        layered
            .generation_image_validation_panic
            .store(true, Ordering::SeqCst);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if case == 4 {
            let revision = mutations.checked_revision().unwrap();
            let successor = Arc::new(
                transition
                    .prepare_same_incarnation_empty_successor()
                    .unwrap(),
            );
            *retirements.anchors.borrow_mut() =
                Some(GenerationRetirementAnchors::capture(&overlay, &successor));
            let transfer = transition
                .prepare_same_incarnation_representation_transfer(
                    &topology,
                    Arc::clone(&overlay),
                    Arc::clone(&successor),
                )
                .unwrap();
            transfer.validate_unpublished_target().unwrap();
            let mut image = PreparedLayerGeneration::empty(Arc::clone(&base), successor);
            image.representation_transfer = Some(transfer);
            prepares.fetch_add(1, Ordering::SeqCst);
            let ready = ReadyLayerPublication {
                revision,
                image,
                external: GenerationDropProbe {
                    layered: Arc::clone(&layered),
                    overlay: Arc::clone(&overlay),
                    #[cfg(feature = "text-index")]
                    text_gates: Some(Arc::clone(&targets)),
                    dropped: Arc::clone(&dropped),
                    guards_drained: Arc::clone(&drained),
                },
            };
            *retirements.unpublished.borrow_mut() = Some(ready.into_retirement());
            return Err("explicit ready abandonment".to_owned());
        }
        layered
            .install_with_pinned_empty_successor_and_publication(
                PinnedEmptySuccessorPublication {
                    current_overlay: &overlay,
                    representation_topology: Some(&topology),
                    transition: &transition,
                    mutation: &mut mutations,
                    retirements: &retirements,
                },
                |_scope, successor| {
                    prepares.fetch_add(1, Ordering::SeqCst);
                    // A real completed transfer already fences both concrete targets.
                    assert!(
                        targets.read().iter().all(|target| target
                            .upgrade()
                            .unwrap()
                            .try_write()
                            .is_none())
                    );
                    if case == 0 {
                        return Err("external preparation rejected".to_owned());
                    }
                    assert!(case != 1, "external preparation unwind");
                    let overlay_image = if case == 2 {
                        Arc::new(LpgStore::new().unwrap())
                    } else {
                        successor
                    };
                    Ok((
                        PreparedLayerGeneration::empty(Arc::clone(&base), overlay_image),
                        GenerationDropProbe {
                            layered: Arc::clone(&layered),
                            overlay: Arc::clone(&overlay),
                            #[cfg(feature = "text-index")]
                            text_gates: Some(Arc::clone(&targets)),
                            dropped: Arc::clone(&dropped),
                            guards_drained: Arc::clone(&drained),
                        },
                    ))
                },
                |probe| {
                    publishes.fetch_add(1, Ordering::SeqCst);
                    probe
                },
                |probe| {
                    rollbacks.fetch_add(1, Ordering::SeqCst);
                    probe
                },
            )
            .map(|_| ())
    }));
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    match case {
        1 => assert_eq!(
            result.err().unwrap().downcast_ref::<&str>(),
            Some(&"external preparation unwind")
        ),
        3 => assert_eq!(
            result.err().unwrap().downcast_ref::<&str>(),
            Some(&"generation final image validation unwind")
        ),
        _ => assert!(result.unwrap().is_err()),
    }
    assert_eq!(tokenizer_drops.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    assert_eq!(
        retirements.anchors.borrow().as_ref().unwrap()._text.len(),
        2
    );
    #[cfg(feature = "vector-index")]
    assert_eq!(
        retirements.anchors.borrow().as_ref().unwrap()._vector.len(),
        1
    );
    drop(transition);
    drop(topology);
    drop(mutations);
    assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
    assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
    assert_eq!(layered.generation_revision.load(Ordering::Acquire), 0);
    let clear_target = Arc::clone(&overlay);
    std::thread::spawn(move || clear_target.clear())
        .join()
        .unwrap();
    assert!(overlay.text_index_entries().is_empty());
    assert!(
        targets
            .read()
            .iter()
            .all(|target| target.upgrade().is_some())
    );
    assert_eq!(tokenizer_drops.load(Ordering::SeqCst), 0);
    if !retain_anchors {
        // Deliberately remove ONLY the production anchors as a positive
        // witness-strength control. No source cut is held; Drop merely records.
        drop(retirements.anchors.borrow_mut().take());
    }
    drop(retirements);
    assert_eq!(tokenizer_drops.load(Ordering::SeqCst), 2);
    assert_eq!(tokenizer_drained.load(Ordering::SeqCst), retain_anchors);
    assert!(
        targets
            .read()
            .iter()
            .all(|target| target.upgrade().is_none())
    );
    assert_eq!(dropped.load(Ordering::SeqCst), usize::from(case >= 2));
    if case >= 2 {
        assert!(drained.load(Ordering::SeqCst));
    }
    layered.merge_overlay_temporal().unwrap();
}

#[cfg(feature = "text-index")]
#[test]
fn generation_totality_separate_transfer_prepare_error_preserves_exact_anchors() {
    assert_generation_transfer_anchors(0, true);
    assert_generation_transfer_anchors(0, false);
}

#[cfg(feature = "text-index")]
#[test]
fn generation_totality_separate_transfer_prepare_unwind_preserves_exact_anchors() {
    assert_generation_transfer_anchors(1, true);
}

#[cfg(feature = "text-index")]
#[test]
fn generation_totality_separate_transfer_image_error_preserves_exact_anchors() {
    assert_generation_transfer_anchors(2, true);
}

#[cfg(feature = "text-index")]
#[test]
fn generation_totality_separate_transfer_image_unwind_preserves_exact_anchors() {
    assert_generation_transfer_anchors(3, true);
}

#[cfg(feature = "text-index")]
#[test]
fn generation_totality_attached_transfer_abandonment_preserves_exact_anchors() {
    assert_generation_transfer_anchors(4, true);
}
