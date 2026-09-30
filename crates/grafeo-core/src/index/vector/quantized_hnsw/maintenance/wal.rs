//! Exact sparse auxiliary transitions; topology uses the shared HNSW codec.
//!
//! Little-endian fixed-width values preserve floating-point bits. Every sequence
//! is length checked against both remaining input and a shared allocation budget
//! before allocating. Recorded codes and calibration models are authoritative:
//! replay validates their shape and exact preimages, never retrains or quantizes.

use super::super::QuantizedOptions;
use super::{
    Arc, HnswMaintenanceWorkspace, NodeId, ProductQuantizer, QuantizationType, QuantizedHnswIndex,
    QuantizedMaintenancePin, QuantizedMaintenanceWorkspace, Result, ScalarQuantizer, invalid,
    qualify_product_layout, qualify_product_model, qualify_scalar_model, qualify_vector,
    reserve_map, reserve_vec,
};
use crate::index::vector::quantization::ProductQuantizerExactState;
use grafeo_common::memory::AllocError;

const MAGIC: &[u8; 4] = b"QVS1";
const LIMIT: usize = 16 * 1024 * 1024;

#[derive(Default)]
struct Model {
    scalar: Option<ScalarQuantizer>,
    product: Option<ProductQuantizer>,
}

#[derive(Debug, PartialEq, Eq)]
enum Code {
    Bytes(Vec<u8>),
    Words(Vec<u64>),
}

struct Row {
    id: NodeId,
    operation: Option<bool>,
    before: Option<Vec<f32>>,
    after: Option<Vec<f32>>,
    before_code: Option<Code>,
    after_code: Option<Code>,
}

struct Image {
    dimensions: usize,
    kind: QuantizationType,
    options: QuantizedOptions,
    trained: bool,
    counts: [usize; 4],
    sample_count: usize,
    baseline_centroids: usize,
    calibration: [u8; 32],
    first_training: bool,
    model: Model,
    samples: Vec<Vec<f32>>,
    rows: Vec<Row>,
    topology: Vec<u8>,
}

pub(super) fn encode(
    index: &QuantizedHnswIndex,
    workspace: &QuantizedMaintenanceWorkspace,
) -> Result<Vec<u8>> {
    if !workspace.prepared {
        return Err(invalid("quantized WAL postimage is not prepared"));
    }
    let topology = workspace.topology.encode_wal_postimage()?;
    let auxiliary = &workspace.auxiliary;
    let mut operations = Vec::new();
    for operation in workspace.topology.prepared_final_presence() {
        if operations.len() >= LIMIT / std::mem::size_of::<(NodeId, bool)>() {
            return Err(invalid("quantized WAL operation budget exceeded"));
        }
        reserve_vec(&mut operations, 1)?;
        operations.push(operation);
    }
    let mut ids = Vec::new();
    reserve_vec(&mut ids, operations.len())?;
    ids.extend(operations.iter().map(|(id, _)| *id));
    if auxiliary.first_training {
        reserve_vec(&mut ids, auxiliary.scalar_vectors.len())?;
        ids.extend(auxiliary.scalar_vectors.keys().copied());
        reserve_vec(&mut ids, auxiliary.product_codes.len())?;
        ids.extend(auxiliary.product_codes.keys().copied());
    }
    ids.sort_unstable();
    ids.dedup();

    // The caller still retains the exact maintenance pin. No final state
    // writer is held while encoding; all these readers observe its preimage.
    let vectors = index.vectors.read();
    let scalar_model = index.scalar_quantizer.read();
    let product_model = index.product_quantizer.read();
    let scalar = index.scalar_vectors.read();
    let binary = index.binary_vectors.read();
    let product = index.product_codes.read();
    let samples = index.training_samples.read();
    let trained = *index.quantizer_trained.read();
    let options = *index.options.read();
    let mut writer = Writer(Vec::new());
    writer.bytes(MAGIC)?;
    writer.usize(index.config().dimensions)?;
    writer.kind(index.quantization_type)?;
    writer.boolean(options.rescore)?;
    writer.usize(options.rescore_factor)?;
    writer.usize(options.training_threshold)?;
    writer.boolean(trained)?;
    for count in [vectors.len(), scalar.len(), binary.len(), product.len()] {
        writer.usize(count)?;
    }
    writer.usize(samples.len())?;
    writer.usize(
        product_model
            .as_ref()
            .map_or(0, ProductQuantizer::num_centroids),
    )?;
    writer.bytes(&calibration_fingerprint(
        scalar_model.as_ref(),
        product_model.as_ref(),
        &samples,
    )?)?;
    writer.boolean(auxiliary.first_training)?;
    writer.model(
        auxiliary.scalar_quantizer.as_ref(),
        auxiliary.product_quantizer.as_ref(),
    )?;
    writer.samples(&auxiliary.samples)?;
    writer.count(ids.len())?;
    for id in ids {
        writer.u64(id.as_u64())?;
        let operation = operations
            .binary_search_by_key(&id, |(id, _)| *id)
            .ok()
            .and_then(|position| operations.get(position))
            .map(|(_, present)| *present);
        writer.byte(match operation {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        })?;
        writer.optional_vector(vectors.get(&id).map(AsRef::as_ref))?;
        writer.optional_vector(auxiliary.vectors.get(&id).map(AsRef::as_ref))?;
        writer.code(
            index.quantization_type,
            scalar.get(&id),
            binary.get(&id),
            product.get(&id),
        )?;
        writer.code(
            index.quantization_type,
            auxiliary.scalar_vectors.get(&id),
            auxiliary.binary_vectors.get(&id),
            auxiliary.product_codes.get(&id),
        )?;
    }
    writer.count(topology.len())?;
    writer.bytes(&topology)?;
    // Encoders cannot produce input rejected by the bounded current decoder.
    let _ = decode(&writer.0)?;
    Ok(writer.0)
}

pub(super) fn prepare_recorded(
    pin: &QuantizedMaintenancePin<'_>,
    workspace: &mut QuantizedMaintenanceWorkspace,
    payload: &[u8],
) -> Result<()> {
    let mut image = decode(payload)?;
    validate_preimage(pin.index, &image)?;
    let auxiliary = &mut workspace.auxiliary;
    reserve_map(&mut auxiliary.vectors, image.rows.len())?;
    let codes = image
        .rows
        .iter()
        .filter(|row| row.after_code.is_some())
        .count();
    match image.kind {
        QuantizationType::Scalar => reserve_map(&mut auxiliary.scalar_vectors, codes)?,
        QuantizationType::Binary => reserve_map(&mut auxiliary.binary_vectors, codes)?,
        QuantizationType::Product { .. } => reserve_map(&mut auxiliary.product_codes, codes)?,
        QuantizationType::None => {}
    }
    reserve_vec(&mut auxiliary.samples, image.samples.len())?;
    for row in &mut image.rows {
        if let Some(vector) = row.after.take() {
            auxiliary.vectors.insert(row.id, Arc::from(vector));
        }
        match row.after_code.take() {
            Some(Code::Bytes(code)) if image.kind == QuantizationType::Scalar => {
                auxiliary.scalar_vectors.insert(row.id, code);
            }
            Some(Code::Bytes(code)) => {
                auxiliary.product_codes.insert(row.id, code);
            }
            Some(Code::Words(code)) => {
                auxiliary.binary_vectors.insert(row.id, code);
            }
            None => {}
        }
    }
    auxiliary
        .samples
        .extend(image.samples.drain(..).map(Arc::from));
    auxiliary.scalar_quantizer = image.model.scalar.take();
    auxiliary.product_quantizer = image.model.product.take();
    auxiliary.first_training = image.first_training;
    workspace.topology = HnswMaintenanceWorkspace::from_recorded(image.topology);
    let baseline = pin.index.vectors.read();
    let accessor = |id| {
        auxiliary
            .vectors
            .get(&id)
            .or_else(|| baseline.get(&id))
            .cloned()
    };
    pin.topology
        .prepare_workspace(&mut workspace.topology, &accessor)?;
    drop(baseline);
    if !workspace.topology.prepared_final_presence().eq(image
        .rows
        .iter()
        .filter_map(|row| row.operation.map(|present| (row.id, present))))
    {
        return Err(invalid(
            "recorded auxiliary operations differ from topology",
        ));
    }
    auxiliary.reserve_targets(pin.index)
}

fn same_vector(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn same_optional_vector(left: Option<&[f32]>, right: Option<&[f32]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => same_vector(left, right),
        _ => false,
    }
}

/// Domain-separated, framed streaming commitment to the unchanged calibration.
/// No model or historical sample-prefix copy becomes part of a sparse payload.
fn calibration_fingerprint(
    scalar: Option<&ScalarQuantizer>,
    product: Option<&ProductQuantizer>,
    samples: &[Arc<[f32]>],
) -> Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new_derive_key("grafeo.vector.quantized-wal.calibration.v1");
    match (scalar, product) {
        (None, None) => {
            hasher.update(&[0]);
        }
        (Some(model), None) => {
            hasher.update(&[1]);
            hash_usize(&mut hasher, model.dimensions())?;
            let (minimum, scale, inverse) = model.storage_parts();
            hash_vector(&mut hasher, minimum)?;
            hash_vector(&mut hasher, scale)?;
            hash_vector(&mut hasher, inverse)?;
        }
        (None, Some(model)) => {
            hasher.update(&[2]);
            hash_usize(&mut hasher, model.dimensions())?;
            hash_usize(&mut hasher, model.num_subvectors())?;
            hash_usize(&mut hasher, model.num_centroids())?;
            hash_usize(&mut hasher, model.subvector_dim())?;
            hash_vector(&mut hasher, model.centroid_values())?;
        }
        _ => return Err(invalid("quantized calibration contains two models")),
    }
    hash_usize(&mut hasher, samples.len())?;
    for sample in samples {
        hash_vector(&mut hasher, sample)?;
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_usize(hasher: &mut blake3::Hasher, value: usize) -> Result<()> {
    let value = u64::try_from(value).map_err(|_| invalid("calibration length exceeds u64"))?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

fn hash_vector(hasher: &mut blake3::Hasher, values: &[f32]) -> Result<()> {
    hash_usize(hasher, values.len())?;
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    Ok(())
}

fn validate_preimage(index: &QuantizedHnswIndex, image: &Image) -> Result<()> {
    if index.config().dimensions != image.dimensions
        || index.quantization_type != image.kind
        || *index.options.read() != image.options
        || *index.quantizer_trained.read() != image.trained
    {
        return Err(invalid("recorded quantizer baseline differs from target"));
    }
    let scalar_model = index.scalar_quantizer.read();
    let product_model = index.product_quantizer.read();
    let samples = index.training_samples.read();
    if samples.len() != image.sample_count
        || product_model
            .as_ref()
            .map_or(0, ProductQuantizer::num_centroids)
            != image.baseline_centroids
        || calibration_fingerprint(scalar_model.as_ref(), product_model.as_ref(), &samples)?
            != image.calibration
    {
        return Err(invalid(
            "recorded model or training prefix differs from target",
        ));
    }
    match (image.kind, image.trained) {
        (QuantizationType::Scalar, true) => {
            qualify_scalar_model(
                scalar_model
                    .as_ref()
                    .ok_or_else(|| invalid("live scalar model is absent"))?,
                image.dimensions,
            )?;
            if product_model.is_some() {
                return Err(invalid("scalar target has a foreign model"));
            }
        }
        (QuantizationType::Product { .. }, true) => {
            qualify_product_model(
                product_model
                    .as_ref()
                    .ok_or_else(|| invalid("live product model is absent"))?,
                image.kind,
                image.dimensions,
                false,
            )?;
            if scalar_model.is_some() {
                return Err(invalid("product target has a foreign model"));
            }
        }
        _ if scalar_model.is_some() || product_model.is_some() => {
            return Err(invalid("untrained target has a model"));
        }
        _ => {}
    }
    for sample in samples.iter() {
        qualify_vector(sample, image.dimensions)?;
    }
    let vectors = index.vectors.read();
    let scalar = index.scalar_vectors.read();
    let binary = index.binary_vectors.read();
    let product = index.product_codes.read();
    if image.counts != [vectors.len(), scalar.len(), binary.len(), product.len()] {
        return Err(invalid("recorded auxiliary directory counts differ"));
    }
    for row in &image.rows {
        if !same_optional_vector(
            vectors.get(&row.id).map(AsRef::as_ref),
            row.before.as_deref(),
        ) {
            return Err(invalid("recorded routing vector preimage differs"));
        }
        let matches = match &row.before_code {
            Some(Code::Bytes(code)) if image.kind == QuantizationType::Scalar => {
                scalar.get(&row.id) == Some(code)
            }
            Some(Code::Bytes(code)) => product.get(&row.id) == Some(code),
            Some(Code::Words(code)) => binary.get(&row.id) == Some(code),
            None => {
                !scalar.contains_key(&row.id)
                    && !binary.contains_key(&row.id)
                    && !product.contains_key(&row.id)
            }
        };
        if !matches {
            return Err(invalid("recorded code preimage differs"));
        }
    }
    Ok(())
}

fn validate(image: &Image) -> Result<()> {
    if image.dimensions == 0
        || image.options.rescore_factor == 0
        || image.options.training_threshold < 10
    {
        return Err(invalid("invalid recorded quantizer configuration"));
    }
    if let QuantizationType::Product { num_subvectors } = image.kind {
        qualify_product_layout(image.dimensions, num_subvectors)?;
    }
    let [raw_count, scalar_count, binary_count, product_count] = image.counts;
    let training_kind = matches!(
        image.kind,
        QuantizationType::Scalar | QuantizationType::Product { .. }
    );
    if (!training_kind && (image.trained || image.sample_count != 0))
        || (image.trained && image.sample_count != 0)
        || (!image.trained && image.sample_count >= image.options.training_threshold)
        || (image.trained
            && matches!(image.kind, QuantizationType::Product { .. })
            && !(1..=256).contains(&image.baseline_centroids))
        || ((!image.trained || !matches!(image.kind, QuantizationType::Product { .. }))
            && image.baseline_centroids != 0)
    {
        return Err(invalid("invalid recorded calibration baseline"));
    }
    let expected_counts = match image.kind {
        QuantizationType::None => [raw_count, 0, 0, 0],
        QuantizationType::Binary => [raw_count, 0, raw_count, 0],
        QuantizationType::Scalar if image.trained => [raw_count, raw_count, 0, 0],
        QuantizationType::Product { .. } if image.trained => [raw_count, 0, 0, raw_count],
        QuantizationType::Scalar | QuantizationType::Product { .. } => [raw_count, 0, 0, 0],
    };
    if image.counts != expected_counts
        || (scalar_count > 0 && image.kind != QuantizationType::Scalar)
        || (binary_count > 0 && image.kind != QuantizationType::Binary)
        || (product_count > 0 && !matches!(image.kind, QuantizationType::Product { .. }))
    {
        return Err(invalid("recorded quantizer has foreign code directories"));
    }
    let upserts = image
        .rows
        .iter()
        .filter(|row| row.operation == Some(true))
        .count();
    let needed = image.options.training_threshold - image.sample_count;
    let additional = if training_kind && !image.trained {
        needed.min(upserts)
    } else {
        0
    };
    if image.samples.len() != additional
        || image.first_training != (training_kind && !image.trained && additional == needed)
    {
        return Err(invalid("recorded first-training boundary differs"));
    }
    if image.first_training {
        match image.kind {
            QuantizationType::Scalar => qualify_scalar_model(
                image
                    .model
                    .scalar
                    .as_ref()
                    .ok_or_else(|| invalid("recorded scalar calibration is missing"))?,
                image.dimensions,
            )?,
            QuantizationType::Product { .. } => qualify_product_model(
                image
                    .model
                    .product
                    .as_ref()
                    .ok_or_else(|| invalid("recorded product calibration is missing"))?,
                image.kind,
                image.dimensions,
                true,
            )?,
            QuantizationType::None | QuantizationType::Binary => {
                return Err(invalid("non-training kind has calibration"));
            }
        }
    } else if image.model.scalar.is_some() || image.model.product.is_some() {
        return Err(invalid("recorded maintenance unexpectedly replaces model"));
    }
    for sample in &image.samples {
        qualify_vector(sample, image.dimensions)?;
    }
    for (sample, vector) in image
        .samples
        .iter()
        .zip(image.rows.iter().filter_map(|row| row.after.as_deref()))
    {
        if !same_vector(sample, vector) {
            return Err(invalid(
                "recorded appended samples differ from ordered upserts",
            ));
        }
    }
    let mut previous = None;
    let mut existing = 0usize;
    for row in &image.rows {
        if !row.id.is_valid()
            || previous.is_some_and(|id| id >= row.id)
            || row.after.is_some() != (row.operation == Some(true))
            || (row.operation.is_none() && (!image.first_training || row.before.is_none()))
        {
            return Err(invalid("non-canonical recorded auxiliary row"));
        }
        previous = Some(row.id);
        if let Some(vector) = &row.before {
            if vector.len() != image.dimensions {
                return Err(invalid("recorded vector preimage has wrong dimensions"));
            }
            existing = existing
                .checked_add(1)
                .ok_or(AllocError::InsufficientSpace)?;
        }
        if let Some(vector) = &row.after {
            qualify_vector(vector, image.dimensions)?;
        }
        let had_codes = image.kind == QuantizationType::Binary || image.trained;
        if row.before_code.is_some() != (had_codes && row.before.is_some()) {
            return Err(invalid("recorded code preimage coverage differs"));
        }
        let needs_code = if image.first_training {
            row.before.is_some() || row.after.is_some()
        } else {
            had_codes && row.after.is_some()
        };
        if row.after_code.is_some() != needs_code {
            return Err(invalid("recorded code postimage coverage differs"));
        }
        validate_code(row.before_code.as_ref(), image, false)?;
        validate_code(row.after_code.as_ref(), image, image.first_training)?;
    }
    if image.first_training && existing != raw_count {
        return Err(invalid("first-training codes omit retained routing rows"));
    }
    Ok(())
}

fn validate_code(code: Option<&Code>, image: &Image, first: bool) -> Result<()> {
    let Some(code) = code else {
        return Ok(());
    };
    let valid = match (image.kind, code) {
        (QuantizationType::Scalar, Code::Bytes(bytes)) => bytes.len() == image.dimensions,
        (QuantizationType::Binary, Code::Words(words)) => {
            let padding = image.dimensions % 64;
            words.len() == image.dimensions.div_ceil(64)
                && (padding == 0
                    || words
                        .last()
                        .is_some_and(|word| word & !((1_u64 << padding) - 1) == 0))
        }
        (QuantizationType::Product { num_subvectors }, Code::Bytes(bytes)) => {
            let centroids = if first {
                image
                    .model
                    .product
                    .as_ref()
                    .map_or(0, ProductQuantizer::num_centroids)
            } else {
                image.baseline_centroids
            };
            bytes.len() == num_subvectors && bytes.iter().all(|code| usize::from(*code) < centroids)
        }
        _ => false,
    };
    if !valid {
        return Err(invalid("recorded code has invalid shape or range"));
    }
    Ok(())
}

struct Writer(Vec<u8>);

impl Writer {
    fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > LIMIT.saturating_sub(self.0.len()) {
            return Err(invalid("quantized WAL exceeds 16 MiB"));
        }
        reserve_vec(&mut self.0, bytes.len())?;
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    fn byte(&mut self, byte: u8) -> Result<()> {
        self.bytes(&[byte])
    }
    fn boolean(&mut self, value: bool) -> Result<()> {
        self.byte(u8::from(value))
    }
    fn u64(&mut self, value: u64) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }
    fn usize(&mut self, value: usize) -> Result<()> {
        self.u64(u64::try_from(value).map_err(|_| invalid("WAL integer exceeds u64"))?)
    }
    fn count(&mut self, count: usize) -> Result<()> {
        self.bytes(
            &u32::try_from(count)
                .map_err(|_| invalid("WAL sequence exceeds u32"))?
                .to_le_bytes(),
        )
    }
    fn kind(&mut self, kind: QuantizationType) -> Result<()> {
        match kind {
            QuantizationType::None => self.byte(0),
            QuantizationType::Scalar => self.byte(1),
            QuantizationType::Binary => self.byte(2),
            QuantizationType::Product { num_subvectors } => {
                self.byte(3)?;
                self.usize(num_subvectors)
            }
        }
    }
    fn vector(&mut self, values: &[f32]) -> Result<()> {
        self.count(values.len())?;
        for value in values {
            self.bytes(&value.to_bits().to_le_bytes())?;
        }
        Ok(())
    }
    fn optional_vector(&mut self, values: Option<&[f32]>) -> Result<()> {
        self.boolean(values.is_some())?;
        if let Some(values) = values {
            self.vector(values)?;
        }
        Ok(())
    }
    fn samples(&mut self, samples: &[Arc<[f32]>]) -> Result<()> {
        self.count(samples.len())?;
        for sample in samples {
            self.vector(sample)?;
        }
        Ok(())
    }
    fn model(
        &mut self,
        scalar: Option<&ScalarQuantizer>,
        product: Option<&ProductQuantizer>,
    ) -> Result<()> {
        match (scalar, product) {
            (None, None) => self.byte(0),
            (Some(model), None) => {
                self.byte(1)?;
                let (minimum, scale, inverse) = model.storage_parts();
                self.vector(minimum)?;
                self.vector(scale)?;
                self.vector(inverse)
            }
            (None, Some(model)) => {
                self.byte(2)?;
                self.usize(model.num_centroids())?;
                self.vector(model.centroid_values())
            }
            _ => Err(invalid("quantized WAL contains two models")),
        }
    }
    fn code(
        &mut self,
        kind: QuantizationType,
        scalar: Option<&Vec<u8>>,
        binary: Option<&Vec<u64>>,
        product: Option<&Vec<u8>>,
    ) -> Result<()> {
        if (scalar.is_some() && kind != QuantizationType::Scalar)
            || (binary.is_some() && kind != QuantizationType::Binary)
            || (product.is_some() && !matches!(kind, QuantizationType::Product { .. }))
        {
            return Err(invalid("quantized WAL contains foreign codes"));
        }
        let bytes = scalar.or(product);
        self.boolean(bytes.is_some() || binary.is_some())?;
        if let Some(bytes) = bytes {
            self.count(bytes.len())?;
            self.bytes(bytes)?;
        }
        if let Some(words) = binary {
            self.count(words.len())?;
            for word in words {
                self.u64(*word)?;
            }
        }
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    allocated: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| invalid("WAL length overflow"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated quantized WAL"))?;
        self.position = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> Result<u8> {
        self.take(1)?
            .first()
            .copied()
            .ok_or_else(|| invalid("truncated WAL byte"))
    }
    fn boolean(&mut self) -> Result<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("non-canonical WAL boolean")),
        }
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| invalid("truncated WAL u64"))?,
        ))
    }
    fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid("WAL integer exceeds usize"))
    }
    fn count<T>(&mut self) -> Result<usize> {
        let count = u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| invalid("truncated WAL count"))?,
        );
        let count = usize::try_from(count).map_err(|_| invalid("WAL count exceeds usize"))?;
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(AllocError::InsufficientSpace)?;
        self.charge(bytes)?;
        if count > self.bytes.len().saturating_sub(self.position) {
            return Err(invalid("WAL count exceeds remaining payload"));
        }
        Ok(count)
    }
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.allocated = self
            .allocated
            .checked_add(bytes)
            .ok_or(AllocError::InsufficientSpace)?;
        if self.allocated > LIMIT {
            return Err(invalid("quantized WAL declared allocation exceeds 16 MiB"));
        }
        Ok(())
    }
    fn sequence<T>(&mut self, mut read: impl FnMut(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        let count = self.count::<T>()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| AllocError::OutOfMemory)?;
        for _ in 0..count {
            values.push(read(self)?);
        }
        Ok(values)
    }
    fn vector(&mut self) -> Result<Vec<f32>> {
        self.sequence(|reader| {
            Ok(f32::from_bits(u32::from_le_bytes(
                reader
                    .take(4)?
                    .try_into()
                    .map_err(|_| invalid("truncated WAL float"))?,
            )))
        })
    }
    fn bytes_vec(&mut self) -> Result<Vec<u8>> {
        let count = self.count::<u8>()?;
        let source = self.take(count)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(count)
            .map_err(|_| AllocError::OutOfMemory)?;
        bytes.extend_from_slice(source);
        Ok(bytes)
    }
    fn optional_vector(&mut self) -> Result<Option<Vec<f32>>> {
        if self.boolean()? {
            Ok(Some(self.vector()?))
        } else {
            Ok(None)
        }
    }
    fn model(&mut self, dimensions: usize, kind: QuantizationType) -> Result<Model> {
        match self.byte()? {
            0 => Ok(Model::default()),
            1 if kind == QuantizationType::Scalar => {
                let minimum = self.vector()?;
                let scale = self.vector()?;
                let inverse = self.vector()?;
                let scalar =
                    ScalarQuantizer::from_storage_parts(dimensions, minimum, scale, inverse)
                        .map_err(invalid)?;
                Ok(Model {
                    scalar: Some(scalar),
                    product: None,
                })
            }
            2 => {
                let QuantizationType::Product { num_subvectors } = kind else {
                    return Err(invalid("WAL model belongs to different quantizer"));
                };
                qualify_product_layout(dimensions, num_subvectors)?;
                let num_centroids = self.usize()?;
                let centroids = self.vector()?;
                let product = ProductQuantizer::from_exact_state(ProductQuantizerExactState {
                    dimensions,
                    num_subvectors,
                    num_centroids,
                    subvector_dim: dimensions / num_subvectors,
                    centroids,
                })
                .map_err(|reason| invalid(&reason))?;
                Ok(Model {
                    scalar: None,
                    product: Some(product),
                })
            }
            _ => Err(invalid("invalid WAL quantizer model tag")),
        }
    }
    fn code(&mut self, kind: QuantizationType) -> Result<Option<Code>> {
        if !self.boolean()? {
            return Ok(None);
        }
        match kind {
            QuantizationType::Scalar | QuantizationType::Product { .. } => {
                Ok(Some(Code::Bytes(self.bytes_vec()?)))
            }
            QuantizationType::Binary => Ok(Some(Code::Words(self.sequence(Self::u64)?))),
            QuantizationType::None => Err(invalid("unquantized WAL carries codes")),
        }
    }
}

fn decode(payload: &[u8]) -> Result<Image> {
    if payload.len() > LIMIT || !payload.starts_with(MAGIC) {
        return Err(invalid("unsupported or oversized quantized WAL payload"));
    }
    let mut reader = Reader {
        bytes: payload,
        position: MAGIC.len(),
        allocated: 0,
    };
    let dimensions = reader.usize()?;
    let kind = match reader.byte()? {
        0 => QuantizationType::None,
        1 => QuantizationType::Scalar,
        2 => QuantizationType::Binary,
        3 => QuantizationType::Product {
            num_subvectors: reader.usize()?,
        },
        _ => return Err(invalid("invalid WAL quantization kind")),
    };
    let options = QuantizedOptions {
        rescore: reader.boolean()?,
        rescore_factor: reader.usize()?,
        training_threshold: reader.usize()?,
    };
    let trained = reader.boolean()?;
    let counts = [
        reader.usize()?,
        reader.usize()?,
        reader.usize()?,
        reader.usize()?,
    ];
    let sample_count = reader.usize()?;
    let baseline_centroids = reader.usize()?;
    let calibration = reader
        .take(32)?
        .try_into()
        .map_err(|_| invalid("truncated calibration fingerprint"))?;
    let first_training = reader.boolean()?;
    let model = reader.model(dimensions, kind)?;
    let samples = reader.sequence(Reader::vector)?;
    let rows = reader.sequence(|reader| {
        let id = NodeId::new(reader.u64()?);
        let operation = match reader.byte()? {
            0 => None,
            1 => Some(false),
            2 => Some(true),
            _ => return Err(invalid("invalid WAL operation tag")),
        };
        Ok(Row {
            id,
            operation,
            before: reader.optional_vector()?,
            after: reader.optional_vector()?,
            before_code: reader.code(kind)?,
            after_code: reader.code(kind)?,
        })
    })?;
    let topology = reader.bytes_vec()?;
    reader.charge(HnswMaintenanceWorkspace::recorded_allocation_bytes(
        &topology,
    )?)?;
    if reader.position != payload.len() {
        return Err(invalid("quantized WAL has trailing bytes"));
    }
    let image = Image {
        dimensions,
        kind,
        options,
        trained,
        counts,
        sample_count,
        baseline_centroids,
        calibration,
        first_training,
        model,
        samples,
        rows,
        topology,
    };
    validate(&image)?;
    Ok(image)
}

#[cfg(test)]
mod tests;
