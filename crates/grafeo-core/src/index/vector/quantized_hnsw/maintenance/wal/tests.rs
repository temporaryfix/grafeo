use super::super::{ReleasedHnswMaintenance, ReleasedQuantizedMaintenance};
use super::{
    Arc, Code, Image, LIMIT, MAGIC, NodeId, QuantizationType, QuantizedHnswIndex,
    QuantizedMaintenanceWorkspace, Reader, Writer, decode, encode,
};
use crate::graph::lpg::PhysicalIndexKey;
use crate::index::vector::{DistanceMetric, HnswConfig, VectorIndexKind, VectorStoreSection};
use grafeo_common::storage::Section;
use grafeo_common::types::GraphPath;

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn kinds() -> [QuantizationType; 4] {
    [
        QuantizationType::None,
        QuantizationType::Scalar,
        QuantizationType::Binary,
        QuantizationType::Product { num_subvectors: 2 },
    ]
}

fn vector(id: u16) -> Arc<[f32]> {
    Arc::from([f32::from(id), f32::from(id % 7), -0.0, 0.5])
}

fn fixture(kind: QuantizationType, count: u16) -> Arc<VectorIndexKind> {
    let index = QuantizedHnswIndex::with_seed(
        HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4),
        kind,
        41,
    )
    .with_training_threshold(10)
    .with_rescore_factor(7)
    .without_rescore();
    for id in 0..count {
        index.insert(NodeId::new(u64::from(id)), &vector(id));
    }
    Arc::new(VectorIndexKind::Quantized(index))
}

fn concrete(index: &VectorIndexKind) -> TestResult<&QuantizedHnswIndex> {
    match index {
        VectorIndexKind::Quantized(index) => Ok(index),
        VectorIndexKind::Hnsw(_) => Err("fixture requires a quantized target".into()),
    }
}

fn exact(index: &Arc<VectorIndexKind>) -> TestResult<Vec<u8>> {
    Ok(VectorStoreSection::new(vec![(
        PhysicalIndexKey::vector(GraphPath::root(), "Node", "v"),
        Arc::clone(index),
    )])
    .serialize()?)
}

fn clone_index(index: &QuantizedHnswIndex) -> TestResult<Arc<VectorIndexKind>> {
    let target = fixture(index.quantization_type(), 0);
    let target_index = concrete(&target)?;
    target_index
        .apply_prepared_exact_restore(target_index.prepare_exact_restore(index.snapshot_exact()?)?);
    Ok(target)
}

fn apply(
    index: &QuantizedHnswIndex,
    workspace: &mut QuantizedMaintenanceWorkspace,
) -> TestResult<Vec<u8>> {
    workspace.topology.capture_wal()?;
    let pin = index.pin_maintenance()?;
    pin.prepare_workspace(workspace)?;
    let payload = encode(index, workspace)?;
    let released = ReleasedQuantizedMaintenance {
        topology: ReleasedHnswMaintenance {
            workspace: &mut workspace.topology,
            pin: &pin.topology,
        },
        auxiliary: &mut workspace.auxiliary,
        pin: &pin,
    };
    let readers = pin.exclude_readers();
    crate::allocation_test::start();
    let ready = released.rebind(&readers);
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    let ready = ready.map_err(|error| error.into_error())?;
    crate::allocation_test::start();
    drop(ready.install());
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    Ok(payload)
}

#[test]
fn quantized_recorded_all_kinds_preserve_training_options_routing_and_next_mutation() -> TestResult
{
    for kind in kinds() {
        for count in [8, 9, 12] {
            let source = fixture(kind, count);
            let target = clone_index(concrete(&source)?)?;
            let mut workspace = QuantizedMaintenanceWorkspace::new(vec![
                (NodeId::new(0), Some(vector(30))),
                (NodeId::new(20), Some(vector(20))),
                (NodeId::new(2), None),
                (NodeId::new(999), None),
            ]);
            let payload = apply(concrete(&source)?, &mut workspace)?;
            let image = decode(&payload)?;
            assert_eq!(image.options.rescore_factor, 7);
            assert!(!image.options.rescore);
            if matches!(
                kind,
                QuantizationType::Scalar | QuantizationType::Product { .. }
            ) && count < 10
            {
                assert!(image.first_training);
                assert_eq!(image.sample_count, usize::from(count));
                assert_eq!(image.samples.len(), 10 - usize::from(count));
            }
            let mut replay = QuantizedMaintenanceWorkspace::from_recorded(payload.clone());
            assert_eq!(apply(concrete(&target)?, &mut replay)?, payload);
            assert_eq!(exact(&target)?, exact(&source)?);
            assert!(
                concrete(&target)?
                    .vectors
                    .read()
                    .contains_key(&NodeId::new(2)),
                "deleted routing is retained"
            );

            // A new ordinary mutation exercises restored RNG, trained model and
            // runtime options, not merely a matching one-time imported result.
            let rows = vec![(NodeId::new(40), Some(vector(40))), (NodeId::new(0), None)];
            let source_next = apply(
                concrete(&source)?,
                &mut QuantizedMaintenanceWorkspace::new(rows.clone()),
            )?;
            let target_next = apply(
                concrete(&target)?,
                &mut QuantizedMaintenanceWorkspace::new(rows),
            )?;
            assert_eq!(source_next, target_next);
            assert_eq!(exact(&target)?, exact(&source)?);
        }
    }
    Ok(())
}

#[test]
fn quantized_recorded_sample_prefix_includes_repeated_identity_and_remains_untrained() -> TestResult
{
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let source = fixture(kind, 3);
        concrete(&source)?.insert(NodeId::new(0), &vector(31));
        let target = clone_index(concrete(&source)?)?;
        let payload = apply(
            concrete(&source)?,
            &mut QuantizedMaintenanceWorkspace::new(vec![
                (NodeId::new(0), Some(vector(32))),
                (NodeId::new(1), None),
            ]),
        )?;
        let image = decode(&payload)?;
        assert_eq!(image.sample_count, 4);
        assert_eq!(image.samples.len(), 1);
        assert!(!image.first_training);
        apply(
            concrete(&target)?,
            &mut QuantizedMaintenanceWorkspace::from_recorded(payload),
        )?;
        assert_eq!(exact(&target)?, exact(&source)?);
    }
    Ok(())
}

fn encode_image(image: &Image) -> TestResult<Vec<u8>> {
    let mut writer = Writer(Vec::new());
    writer.bytes(MAGIC)?;
    writer.usize(image.dimensions)?;
    writer.kind(image.kind)?;
    writer.boolean(image.options.rescore)?;
    writer.usize(image.options.rescore_factor)?;
    writer.usize(image.options.training_threshold)?;
    writer.boolean(image.trained)?;
    for count in image.counts {
        writer.usize(count)?;
    }
    writer.usize(image.sample_count)?;
    writer.usize(image.baseline_centroids)?;
    writer.bytes(&image.calibration)?;
    writer.boolean(image.first_training)?;
    writer.model(image.model.scalar.as_ref(), image.model.product.as_ref())?;
    writer.count(image.samples.len())?;
    for sample in &image.samples {
        writer.vector(sample)?;
    }
    writer.count(image.rows.len())?;
    for row in &image.rows {
        writer.u64(row.id.as_u64())?;
        writer.byte(match row.operation {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        })?;
        writer.optional_vector(row.before.as_deref())?;
        writer.optional_vector(row.after.as_deref())?;
        for code in [&row.before_code, &row.after_code] {
            writer.boolean(code.is_some())?;
            match code {
                Some(Code::Bytes(bytes)) => {
                    writer.count(bytes.len())?;
                    writer.bytes(bytes)?;
                }
                Some(Code::Words(words)) => {
                    writer.count(words.len())?;
                    for word in words {
                        writer.u64(*word)?;
                    }
                }
                None => {}
            }
        }
    }
    writer.count(image.topology.len())?;
    writer.bytes(&image.topology)?;
    Ok(writer.0)
}

fn refuses_unchanged(target: &Arc<VectorIndexKind>, payload: Vec<u8>) -> TestResult {
    let before = exact(target)?;
    let mut workspace = QuantizedMaintenanceWorkspace::from_recorded(payload);
    let pin = concrete(target)?.pin_maintenance()?;
    assert!(pin.prepare_workspace(&mut workspace).is_err());
    drop(pin);
    assert_eq!(exact(target)?, before);
    Ok(())
}

#[test]
fn quantized_recorded_malformed_stale_and_code_preimage_fail_without_mutation() -> TestResult {
    for kind in kinds() {
        let source = fixture(kind, 12);
        let target = clone_index(concrete(&source)?)?;
        let payload = apply(
            concrete(&source)?,
            &mut QuantizedMaintenanceWorkspace::new(vec![
                (NodeId::new(0), Some(vector(33))),
                (NodeId::new(2), None),
            ]),
        )?;
        let mut trailing = payload.clone();
        trailing.push(0);
        refuses_unchanged(&target, trailing)?;
        refuses_unchanged(
            &target,
            payload
                .get(..payload.len() / 2)
                .ok_or("missing payload prefix")?
                .to_vec(),
        )?;
        let mut image = decode(&payload)?;
        image.options.rescore = !image.options.rescore;
        refuses_unchanged(&target, encode_image(&image)?)?;
        let mut image = decode(&payload)?;
        let value = image
            .rows
            .first_mut()
            .and_then(|row| row.before.as_mut())
            .and_then(|values| values.first_mut())
            .ok_or("missing routing preimage")?;
        *value = 55.0;
        refuses_unchanged(&target, encode_image(&image)?)?;
        if kind != QuantizationType::None {
            let mut image = decode(&payload)?;
            let code = image
                .rows
                .first_mut()
                .and_then(|row| row.before_code.as_mut())
                .ok_or("missing code preimage")?;
            match code {
                Code::Bytes(bytes) => {
                    *bytes.first_mut().ok_or("missing byte")? ^= 1;
                }
                Code::Words(words) => {
                    *words.first_mut().ok_or("missing word")? ^= 1;
                }
            }
            refuses_unchanged(&target, encode_image(&image)?)?;
        }
        apply(
            concrete(&target)?,
            &mut QuantizedMaintenanceWorkspace::from_recorded(payload.clone()),
        )?;
        refuses_unchanged(&target, payload)?;
    }
    Ok(())
}

#[test]
fn quantized_recorded_rejects_omitted_training_codes_and_forged_sample_prefix() -> TestResult {
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let source = fixture(kind, 9);
        let target = clone_index(concrete(&source)?)?;
        let payload = apply(
            concrete(&source)?,
            &mut QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(20), Some(vector(20)))]),
        )?;
        let mut image = decode(&payload)?;
        image.rows.retain(|row| row.id != NodeId::new(2));
        refuses_unchanged(&target, encode_image(&image)?)?;
        let mut changed = concrete(&target)?.snapshot_exact()?;
        *changed
            .training_samples
            .first_mut()
            .and_then(|sample| sample.first_mut())
            .ok_or("missing sample")? = 77.0;
        concrete(&target)?
            .apply_prepared_exact_restore(concrete(&target)?.prepare_exact_restore(changed)?);
        refuses_unchanged(&target, payload)?;
    }
    Ok(())
}

#[test]
fn quantized_recorded_declared_allocation_and_payload_limits_are_bounded() -> TestResult {
    let mut reader = Reader {
        bytes: &u32::MAX.to_le_bytes(),
        position: 0,
        allocated: 0,
    };
    assert!(
        reader
            .sequence::<Vec<f32>>(|reader| reader.vector())
            .is_err()
    );
    let mut reader = Reader {
        bytes: &1_u32.to_le_bytes(),
        position: 0,
        allocated: LIMIT,
    };
    assert!(reader.sequence::<u8>(|reader| reader.byte()).is_err());
    let mut oversized = vec![0; LIMIT + 1];
    oversized
        .get_mut(..MAGIC.len())
        .ok_or("missing header")?
        .copy_from_slice(MAGIC);
    assert!(decode(&oversized).is_err());
    let source = fixture(QuantizationType::None, 0);
    let mut workspace = QuantizedMaintenanceWorkspace::new(Vec::new());
    workspace.topology.capture_wal()?;
    let pin = concrete(&source)?.pin_maintenance()?;
    pin.prepare_workspace(&mut workspace)?;
    assert!(decode(&encode(concrete(&source)?, &workspace)?).is_ok());
    Ok(())
}

#[test]
fn quantized_recorded_delete_size_does_not_copy_unrelated_directories() -> TestResult {
    for kind in kinds() {
        let mut lengths = Vec::new();
        for count in [16, 512] {
            let source = fixture(kind, count);
            let payload = apply(
                concrete(&source)?,
                &mut QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(2), None)]),
            )?;
            let image = decode(&payload)?;
            assert_eq!(image.rows.len(), 1);
            assert!(image.samples.is_empty());
            lengths.push(payload.len());
        }
        let minimum = lengths.iter().min().ok_or("missing lengths")?;
        let maximum = lengths.iter().max().ok_or("missing lengths")?;
        assert!(
            maximum - minimum < 128,
            "unrelated population must not become WAL postimages"
        );
    }
    Ok(())
}

#[test]
fn quantized_recorded_rejects_changed_model_bits_with_identical_metadata_and_codes() -> TestResult {
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let source = fixture(kind, 12);
        let target = clone_index(concrete(&source)?)?;
        let payload = apply(
            concrete(&source)?,
            &mut QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(2), None)]),
        )?;
        let mut changed = concrete(&target)?.snapshot_exact()?;
        let zero = match kind {
            QuantizationType::Scalar => changed
                .scalar_quantizer
                .as_mut()
                .and_then(|model| model.min.get_mut(2))
                .ok_or("missing scalar zero dimension")?,
            QuantizationType::Product { .. } => {
                let model = changed
                    .product_quantizer
                    .as_mut()
                    .ok_or("missing product model")?;
                let offset = model
                    .num_centroids
                    .checked_mul(model.subvector_dim)
                    .ok_or("centroid offset overflow")?;
                model
                    .centroids
                    .get_mut(offset)
                    .ok_or("missing product zero dimension")?
            }
            _ => return Err("fixture requires a trained quantizer".into()),
        };
        assert_eq!(*zero, 0.0);
        *zero = f32::from_bits(zero.to_bits() ^ (1_u32 << 31));
        // Signed zero changes exact model identity without altering any code.
        concrete(&target)?
            .apply_prepared_exact_restore(concrete(&target)?.prepare_exact_restore(changed)?);
        refuses_unchanged(&target, payload)?;
    }
    Ok(())
}

#[test]
fn quantized_recorded_payload_does_not_grow_with_unchanged_training_prefix_or_model() -> TestResult
{
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let mut lengths = Vec::new();
        for count in [4, 256] {
            let index = QuantizedHnswIndex::with_seed(
                HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4),
                kind,
                41,
            )
            .with_training_threshold(1000);
            for _ in 0..count {
                index.insert(NodeId::new(0), &vector(3));
            }
            let payload = apply(
                &index,
                &mut QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), Some(vector(30)))]),
            )?;
            let image = decode(&payload)?;
            assert_eq!(image.sample_count, count);
            assert_eq!(image.samples.len(), 1);
            assert!(image.model.scalar.is_none() && image.model.product.is_none());
            lengths.push(payload.len());
        }
        assert!(
            lengths.iter().max().ok_or("missing lengths")?
                - lengths.iter().min().ok_or("missing lengths")?
                < 16
        );
    }

    let source = fixture(QuantizationType::Product { num_subvectors: 2 }, 12);
    let compact_model = clone_index(concrete(&source)?)?;
    let mut state = concrete(&compact_model)?.snapshot_exact()?;
    let used_centroids = usize::from(
        state
            .product_codes
            .iter()
            .flat_map(|(_, codes)| codes.iter())
            .copied()
            .max()
            .ok_or("missing product codes")?,
    ) + 1;
    let model = state
        .product_quantizer
        .as_mut()
        .ok_or("missing product model")?;
    let old_centroids = model.centroids.clone();
    let old_width = model
        .num_centroids
        .checked_mul(model.subvector_dim)
        .ok_or("centroid width overflow")?;
    let retained_width = used_centroids
        .checked_mul(model.subvector_dim)
        .ok_or("retained centroid width overflow")?;
    let mut centroids = Vec::new();
    for subvector in 0..model.num_subvectors {
        let begin = subvector
            .checked_mul(old_width)
            .ok_or("centroid offset overflow")?;
        let end = begin
            .checked_add(old_width)
            .ok_or("centroid end overflow")?;
        let block = old_centroids
            .get(begin..end)
            .ok_or("missing centroid block")?;
        // Discard only unused trailing centroids, keeping every exact code.
        centroids.extend_from_slice(
            block
                .get(..retained_width)
                .ok_or("missing retained centroids")?,
        );
    }
    assert!(old_centroids.len() > centroids.len() * 10);
    model.num_centroids = used_centroids;
    model.centroids = centroids;
    concrete(&compact_model)?
        .apply_prepared_exact_restore(concrete(&compact_model)?.prepare_exact_restore(state)?);
    let rows = vec![(NodeId::new(2), None)];
    let small = apply(
        concrete(&compact_model)?,
        &mut QuantizedMaintenanceWorkspace::new(rows.clone()),
    )?;
    let large = apply(
        concrete(&source)?,
        &mut QuantizedMaintenanceWorkspace::new(rows),
    )?;
    assert_eq!(
        small.len(),
        large.len(),
        "unchanged PQ table is a fixed-width fingerprint"
    );
    assert_ne!(decode(&small)?.calibration, decode(&large)?.calibration);
    Ok(())
}
