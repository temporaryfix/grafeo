//! Sparse quantized postimages under the exact inner HNSW mutation authority.
//!
//! The caller owns handles, workspaces and pins in separate outer scopes. All
//! preparation is private; final rebind is try-only, and installation retains
//! displaced payloads until every enclosing publication fence has drained.

use super::{
    Arc, BinaryQuantizer, HashMap, NodeId, ProductQuantizer, QuantizationType, QuantizedHnswIndex,
    ScalarQuantizer,
};
#[cfg(test)]
use super::{HnswConfig, QuantizedExactState};
use crate::graph::lpg::DataRebindError;
use crate::index::vector::hnsw::maintenance::{
    HnswGuardSlot, HnswMaintenancePin, HnswMaintenanceWorkspace,
};
#[cfg(test)]
use crate::index::vector::hnsw::maintenance::{
    HnswReaderFence, InstalledHnswMaintenance, ReadyHnswMaintenance, ReleasedHnswMaintenance,
};
use grafeo_common::memory::AllocError;
use grafeo_common::utils::error::{Error, Result, TransactionError};
use parking_lot::RwLockWriteGuard;
use std::cell::Cell;

mod wal;

pub(crate) struct QuantizedMaintenancePin<'index> {
    index: &'index QuantizedHnswIndex,
    topology: HnswMaintenancePin<'index>,
    preparation_claimed: Cell<bool>,
}

impl QuantizedHnswIndex {
    pub(crate) fn pin_maintenance(&self) -> Result<QuantizedMaintenancePin<'_>> {
        let topology = self.hnsw.pin_maintenance()?;
        Ok(QuantizedMaintenancePin {
            index: self,
            topology,
            preparation_claimed: Cell::new(false),
        })
    }
}

impl QuantizedMaintenancePin<'_> {
    pub(in crate::index::vector) fn pins_topology(&self, pin: &HnswMaintenancePin<'_>) -> bool {
        std::ptr::eq(std::ptr::from_ref(&self.topology), pin)
    }

    fn prepare_workspace(&self, workspace: &mut QuantizedMaintenanceWorkspace) -> Result<()> {
        if self.preparation_claimed.replace(true) {
            return Err(invalid("maintenance pin preparation is one-shot"));
        }
        if workspace.attempted {
            return Err(invalid("workspace preparation is one-shot"));
        }
        workspace.attempted = true;
        if let Some(payload) = workspace.recorded.take() {
            wal::prepare_recorded(self, workspace, &payload)?;
            workspace.prepared = true;
            return Ok(());
        }
        let dimensions = self.index.config().dimensions;
        if dimensions == 0 {
            return Err(invalid("vector dimensions must be positive"));
        }
        let operations = workspace.topology.normalized_operations(dimensions)?;
        let auxiliary = &mut workspace.auxiliary;
        reserve_map(&mut auxiliary.vectors, operations.len())?;
        for (id, vector) in operations {
            if let Some(vector) = vector {
                qualify_vector(vector, dimensions)?;
                auxiliary.vectors.insert(*id, Arc::clone(vector));
            }
        }
        auxiliary.prepare(self.index, operations)?;

        // No whole-directory clone: absent final rows keep their old routing
        // vector, while every upsert is read from the same final-row overlay.
        let baseline = self.index.vectors.read();
        let accessor = |id| {
            auxiliary
                .vectors
                .get(&id)
                .or_else(|| baseline.get(&id))
                .cloned()
        };
        self.topology
            .prepare_workspace(&mut workspace.topology, &accessor)?;
        drop(baseline);
        // Binary searches retain code readers while entering topology. Never
        // acquire auxiliary guards while the initial topology writers remain.
        auxiliary.reserve_targets(self.index)?;
        workspace.prepared = true;
        Ok(())
    }
}

#[cfg(test)]
impl<'index> QuantizedMaintenancePin<'index> {
    pub(crate) fn exclude_readers(&self) -> HnswReaderFence<'index, '_> {
        self.topology.exclude_readers()
    }

    pub(crate) fn prepare<'workspace, 'pin>(
        &'pin self,
        workspace: &'workspace mut QuantizedMaintenanceWorkspace,
    ) -> Result<ReleasedQuantizedMaintenance<'index, 'workspace, 'pin>> {
        self.prepare_workspace(workspace)?;
        Ok(ReleasedQuantizedMaintenance {
            topology: ReleasedHnswMaintenance {
                workspace: &mut workspace.topology,
                pin: &self.topology,
            },
            auxiliary: &mut workspace.auxiliary,
            pin: self,
        })
    }
}

pub(crate) struct QuantizedMaintenanceWorkspace {
    topology: HnswMaintenanceWorkspace,
    auxiliary: AuxiliaryWorkspace,
    attempted: bool,
    prepared: bool,
    recorded: Option<Vec<u8>>,
}

impl QuantizedMaintenanceWorkspace {
    pub(crate) fn new(operations: Vec<(NodeId, Option<Arc<[f32]>>)>) -> Self {
        Self {
            topology: HnswMaintenanceWorkspace::new(operations),
            auxiliary: AuxiliaryWorkspace::default(),
            attempted: false,
            prepared: false,
            recorded: None,
        }
    }

    fn from_recorded(payload: Vec<u8>) -> Self {
        let mut workspace = Self::new(Vec::new());
        workspace.recorded = Some(payload);
        workspace
    }
}

#[derive(Default)]
struct AuxiliaryWorkspace {
    vectors: HashMap<NodeId, Arc<[f32]>>,
    scalar_vectors: HashMap<NodeId, Vec<u8>>,
    binary_vectors: HashMap<NodeId, Vec<u64>>,
    product_codes: HashMap<NodeId, Vec<u8>>,
    samples: Vec<Arc<[f32]>>,
    // These hold the candidate before installation and the old model after it.
    scalar_quantizer: Option<ScalarQuantizer>,
    product_quantizer: Option<ProductQuantizer>,
    first_training: bool,
    retired_vectors: Vec<Arc<[f32]>>,
    retired_scalar_vectors: Vec<Vec<u8>>,
    retired_binary_vectors: Vec<Vec<u64>>,
    retired_product_codes: Vec<Vec<u8>>,
    retired_samples: Vec<Arc<[f32]>>,
}

impl AuxiliaryWorkspace {
    fn prepare(
        &mut self,
        index: &QuantizedHnswIndex,
        operations: &[(NodeId, Option<Arc<[f32]>>)],
    ) -> Result<()> {
        let options = *index.options.read();
        if options.rescore_factor == 0 || options.training_threshold < 10 {
            return Err(invalid("invalid quantized runtime options"));
        }
        let dimensions = index.config().dimensions;
        let trained = *index.quantizer_trained.read();
        let samples = index.training_samples.read();
        let scalar = index.scalar_quantizer.read();
        let product = index.product_quantizer.read();
        match index.quantization_type {
            QuantizationType::None | QuantizationType::Binary => {
                if trained || !samples.is_empty() || scalar.is_some() || product.is_some() {
                    return Err(invalid("untrained kind contains calibration state"));
                }
                if !index.scalar_vectors.read().is_empty()
                    || !index.product_codes.read().is_empty()
                    || (index.quantization_type == QuantizationType::None
                        && !index.binary_vectors.read().is_empty())
                {
                    return Err(invalid("quantized kind contains foreign codes"));
                }
                if index.quantization_type == QuantizationType::Binary {
                    reserve_map(&mut self.binary_vectors, self.vectors.len())?;
                    for (&id, vector) in &self.vectors {
                        self.binary_vectors
                            .insert(id, BinaryQuantizer::quantize(vector));
                    }
                }
            }
            QuantizationType::Scalar | QuantizationType::Product { .. } => {
                let is_scalar = index.quantization_type == QuantizationType::Scalar;
                if !index.binary_vectors.read().is_empty()
                    || (is_scalar && (product.is_some() || !index.product_codes.read().is_empty()))
                    || (!is_scalar && (scalar.is_some() || !index.scalar_vectors.read().is_empty()))
                {
                    return Err(invalid("quantized kind contains foreign model or codes"));
                }
                if let QuantizationType::Product { num_subvectors } = index.quantization_type {
                    qualify_product_layout(dimensions, num_subvectors)?;
                }
                if trained {
                    if !samples.is_empty() {
                        return Err(invalid("trained quantizer retains pending samples"));
                    }
                    if is_scalar {
                        let quantizer = scalar
                            .as_ref()
                            .ok_or_else(|| invalid("trained scalar model is absent"))?;
                        qualify_scalar_model(quantizer, dimensions)?;
                        reserve_map(&mut self.scalar_vectors, self.vectors.len())?;
                        for (&id, vector) in &self.vectors {
                            self.scalar_vectors.insert(id, quantizer.quantize(vector));
                        }
                    } else {
                        let quantizer = product
                            .as_ref()
                            .ok_or_else(|| invalid("trained product model is absent"))?;
                        qualify_product_model(
                            quantizer,
                            index.quantization_type,
                            dimensions,
                            false,
                        )?;
                        reserve_map(&mut self.product_codes, self.vectors.len())?;
                        for (&id, vector) in &self.vectors {
                            self.product_codes.insert(id, quantizer.quantize(vector));
                        }
                    }
                } else {
                    if scalar.is_some()
                        || product.is_some()
                        || !index.scalar_vectors.read().is_empty()
                        || !index.product_codes.read().is_empty()
                        || samples.len() >= options.training_threshold
                    {
                        return Err(invalid("untrained quantizer state is inconsistent"));
                    }
                    let needed = options.training_threshold - samples.len();
                    let additional = needed.min(self.vectors.len());
                    reserve_vec(&mut self.samples, additional)?;
                    for (_, vector) in operations {
                        if self.samples.len() == additional {
                            break;
                        }
                        if let Some(vector) = vector {
                            self.samples.push(Arc::clone(vector));
                        }
                    }
                    self.first_training = self.samples.len() == needed;
                    if self.first_training {
                        // Preserve the historical prefix (including repeated
                        // identities), then only the batch prefix to crossing.
                        let total = samples
                            .len()
                            .checked_add(self.samples.len())
                            .ok_or(AllocError::InsufficientSpace)?;
                        checked_elements(total, dimensions)?;
                        let mut references = Vec::new();
                        reserve_vec(&mut references, total)?;
                        for sample in samples.iter().chain(&self.samples) {
                            qualify_vector(sample, dimensions)?;
                            references.push(sample.as_ref());
                        }
                        if references.is_empty() {
                            return Err(invalid("training prefix is empty"));
                        }
                        match index.quantization_type {
                            QuantizationType::Scalar => {
                                self.scalar_quantizer = Some(ScalarQuantizer::train(&references));
                            }
                            QuantizationType::Product { num_subvectors } => {
                                self.product_quantizer = Some(ProductQuantizer::train(
                                    &references,
                                    num_subvectors,
                                    256,
                                    10,
                                ));
                            }
                            QuantizationType::None | QuantizationType::Binary => {
                                return Err(invalid("non-training kind selected calibration"));
                            }
                        }
                        drop(references);
                        self.prepare_first_codes(index)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn prepare_first_codes(&mut self, index: &QuantizedHnswIndex) -> Result<()> {
        let baseline = index.vectors.read();
        let dimensions = index.config().dimensions;
        let missing = self
            .vectors
            .keys()
            .filter(|id| !baseline.contains_key(*id))
            .count();
        let total = baseline
            .len()
            .checked_add(missing)
            .ok_or(AllocError::InsufficientSpace)?;
        // Codes are globally affected only at the first calibration. Full
        // routing vectors and topology remain borrowed, never copied wholesale.
        if let Some(quantizer) = &self.scalar_quantizer {
            qualify_scalar_model(quantizer, dimensions)?;
            reserve_map(&mut self.scalar_vectors, total)?;
            for (&id, vector) in baseline
                .iter()
                .filter(|(id, _)| !self.vectors.contains_key(*id))
                .chain(self.vectors.iter())
            {
                qualify_vector(vector, dimensions)?;
                self.scalar_vectors.insert(id, quantizer.quantize(vector));
            }
        } else if let Some(quantizer) = &self.product_quantizer {
            qualify_product_model(quantizer, index.quantization_type, dimensions, true)?;
            reserve_map(&mut self.product_codes, total)?;
            for (&id, vector) in baseline
                .iter()
                .filter(|(id, _)| !self.vectors.contains_key(*id))
                .chain(self.vectors.iter())
            {
                qualify_vector(vector, dimensions)?;
                self.product_codes.insert(id, quantizer.quantize(vector));
            }
        } else {
            return Err(invalid("first calibration did not produce a model"));
        }
        Ok(())
    }

    fn reserve_targets(&mut self, index: &QuantizedHnswIndex) -> Result<()> {
        reserve_merge(
            &mut index.vectors.write(),
            &self.vectors,
            &mut self.retired_vectors,
        )?;
        reserve_merge(
            &mut index.scalar_vectors.write(),
            &self.scalar_vectors,
            &mut self.retired_scalar_vectors,
        )?;
        reserve_merge(
            &mut index.binary_vectors.write(),
            &self.binary_vectors,
            &mut self.retired_binary_vectors,
        )?;
        reserve_merge(
            &mut index.product_codes.write(),
            &self.product_codes,
            &mut self.retired_product_codes,
        )?;
        if !self.first_training {
            reserve_vec(&mut index.training_samples.write(), self.samples.len())?;
        }
        Ok(())
    }

    fn install(&mut self, guards: &mut AuxiliaryGuards<'_>) {
        install_merge(
            &mut guards.vectors,
            &mut self.vectors,
            &mut self.retired_vectors,
        );
        install_merge(
            &mut guards.scalar_vectors,
            &mut self.scalar_vectors,
            &mut self.retired_scalar_vectors,
        );
        install_merge(
            &mut guards.binary_vectors,
            &mut self.binary_vectors,
            &mut self.retired_binary_vectors,
        );
        install_merge(
            &mut guards.product_codes,
            &mut self.product_codes,
            &mut self.retired_product_codes,
        );
        if self.first_training {
            std::mem::swap(&mut *guards.scalar_quantizer, &mut self.scalar_quantizer);
            std::mem::swap(&mut *guards.product_quantizer, &mut self.product_quantizer);
            std::mem::swap(&mut *guards.samples, &mut self.retired_samples);
            *guards.trained = true;
        } else {
            guards.samples.extend(self.samples.drain(..));
        }
    }
}

// Fields drop in reverse acquisition order. Options are immutable under the
// retained alias pin and are not changed by this component.
struct AuxiliaryGuards<'index> {
    trained: RwLockWriteGuard<'index, bool>,
    samples: RwLockWriteGuard<'index, Vec<Arc<[f32]>>>,
    product_codes: RwLockWriteGuard<'index, HashMap<NodeId, Vec<u8>>>,
    binary_vectors: RwLockWriteGuard<'index, HashMap<NodeId, Vec<u64>>>,
    scalar_vectors: RwLockWriteGuard<'index, HashMap<NodeId, Vec<u8>>>,
    product_quantizer: RwLockWriteGuard<'index, Option<ProductQuantizer>>,
    scalar_quantizer: RwLockWriteGuard<'index, Option<ScalarQuantizer>>,
    vectors: RwLockWriteGuard<'index, HashMap<NodeId, Arc<[f32]>>>,
}

impl<'index> AuxiliaryGuards<'index> {
    fn try_acquire(
        index: &'index QuantizedHnswIndex,
    ) -> std::result::Result<Self, DataRebindError> {
        let vectors = index
            .vectors
            .try_write()
            .ok_or(DataRebindError::Conflict("quantized vectors are busy"))?;
        let scalar_quantizer = index
            .scalar_quantizer
            .try_write()
            .ok_or(DataRebindError::Conflict("scalar model is busy"))?;
        let product_quantizer = index
            .product_quantizer
            .try_write()
            .ok_or(DataRebindError::Conflict("product model is busy"))?;
        let scalar_vectors = index
            .scalar_vectors
            .try_write()
            .ok_or(DataRebindError::Conflict("scalar codes are busy"))?;
        let binary_vectors = index
            .binary_vectors
            .try_write()
            .ok_or(DataRebindError::Conflict("binary codes are busy"))?;
        let product_codes = index
            .product_codes
            .try_write()
            .ok_or(DataRebindError::Conflict("product codes are busy"))?;
        let samples = index
            .training_samples
            .try_write()
            .ok_or(DataRebindError::Conflict("quantized samples are busy"))?;
        let trained = index
            .quantizer_trained
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "quantized training state is busy",
            ))?;
        Ok(Self {
            trained,
            samples,
            product_codes,
            binary_vectors,
            scalar_vectors,
            product_quantizer,
            scalar_quantizer,
            vectors,
        })
    }
}

#[cfg(test)]
pub(crate) struct ReleasedQuantizedMaintenance<'index, 'workspace, 'pin> {
    topology: ReleasedHnswMaintenance<'index, 'workspace, 'pin>,
    auxiliary: &'workspace mut AuxiliaryWorkspace,
    pin: &'pin QuantizedMaintenancePin<'index>,
}

#[cfg(test)]
pub(crate) struct ReadyQuantizedMaintenance<'index, 'workspace, 'pin, 'fence> {
    guards: AuxiliaryGuards<'index>,
    topology: ReadyHnswMaintenance<'index, 'workspace, 'pin, 'fence>,
    auxiliary: &'workspace mut AuxiliaryWorkspace,
    pin: &'pin QuantizedMaintenancePin<'index>,
}

#[cfg(test)]
pub(crate) struct InstalledQuantizedMaintenance<'index, 'workspace, 'pin, 'fence> {
    _guards: AuxiliaryGuards<'index>,
    _topology: InstalledHnswMaintenance<'index, 'workspace, 'pin, 'fence>,
    _auxiliary: &'workspace mut AuxiliaryWorkspace,
}

#[cfg(test)]
impl<'index, 'workspace, 'pin> ReleasedQuantizedMaintenance<'index, 'workspace, 'pin> {
    pub(crate) fn rebind<'fence>(
        self,
        readers: &'fence HnswReaderFence<'index, 'pin>,
    ) -> std::result::Result<
        ReadyQuantizedMaintenance<'index, 'workspace, 'pin, 'fence>,
        DataRebindError,
    > {
        let topology = self.topology.rebind(readers)?;
        let guards = AuxiliaryGuards::try_acquire(self.pin.index)?;
        Ok(ReadyQuantizedMaintenance {
            guards,
            topology,
            auxiliary: self.auxiliary,
            pin: self.pin,
        })
    }
}

#[cfg(test)]
impl<'index, 'workspace, 'pin, 'fence> ReadyQuantizedMaintenance<'index, 'workspace, 'pin, 'fence> {
    pub(crate) fn release(self) -> ReleasedQuantizedMaintenance<'index, 'workspace, 'pin> {
        drop(self.guards);
        ReleasedQuantizedMaintenance {
            topology: self.topology.release(),
            auxiliary: self.auxiliary,
            pin: self.pin,
        }
    }

    pub(crate) fn install(self) -> InstalledQuantizedMaintenance<'index, 'workspace, 'pin, 'fence> {
        let Self {
            mut guards,
            topology,
            auxiliary,
            ..
        } = self;
        let topology = topology.install();
        auxiliary.install(&mut guards);
        InstalledQuantizedMaintenance {
            _guards: guards,
            _topology: topology,
            _auxiliary: auxiliary,
        }
    }
}

/// Raw guards and their intrinsic candidate pairing, retained in the outer
/// batch buffer. No field borrows a local maintenance pin or another field.
pub(in crate::index::vector) struct QuantizedMaintenanceSlot<'index> {
    auxiliary_guards: Option<AuxiliaryGuards<'index>>,
    topology_guards: HnswGuardSlot<'index>,
    index: &'index QuantizedHnswIndex,
    workspace: QuantizedMaintenanceWorkspace,
}

impl<'index> QuantizedMaintenanceSlot<'index> {
    pub(in crate::index::vector) fn new(
        index: &'index QuantizedHnswIndex,
        rows: Vec<(NodeId, Option<Arc<[f32]>>)>,
    ) -> Self {
        Self {
            auxiliary_guards: None,
            topology_guards: HnswGuardSlot::new(&index.hnsw),
            index,
            workspace: QuantizedMaintenanceWorkspace::new(rows),
        }
    }

    pub(in crate::index::vector) fn from_recorded(
        index: &'index QuantizedHnswIndex,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            auxiliary_guards: None,
            topology_guards: HnswGuardSlot::new(&index.hnsw),
            index,
            workspace: QuantizedMaintenanceWorkspace::from_recorded(payload),
        }
    }

    pub(in crate::index::vector) fn encode_wal_postimage(&self) -> Result<Vec<u8>> {
        wal::encode(self.index, &self.workspace)
    }

    pub(in crate::index::vector) fn capture_wal(&mut self) -> Result<()> {
        self.workspace.topology.capture_wal()
    }

    pub(in crate::index::vector) fn matches(&self, pin: &QuantizedMaintenancePin<'_>) -> bool {
        std::ptr::eq(self.index, pin.index) && self.topology_guards.matches(&pin.topology)
    }

    pub(in crate::index::vector) fn prepare(
        &mut self,
        pin: &QuantizedMaintenancePin<'index>,
    ) -> Result<()> {
        if !self.matches(pin) {
            return Err(invalid("maintenance slot belongs to a different target"));
        }
        pin.prepare_workspace(&mut self.workspace)
    }

    pub(in crate::index::vector) fn exclude_readers(
        &mut self,
        pin: &QuantizedMaintenancePin<'index>,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'index, '_>,
    ) -> std::result::Result<(), DataRebindError> {
        if !scope.pins_quantized(pin) || !self.matches(pin) || self.auxiliary_guards.is_some() {
            return Err(DataRebindError::new("quantized reader slot target changed"));
        }
        self.topology_guards.exclude_readers(&pin.topology, scope)
    }

    pub(in crate::index::vector) fn rebind(
        &mut self,
        pin: &QuantizedMaintenancePin<'index>,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'index, '_>,
    ) -> std::result::Result<(), DataRebindError> {
        if !scope.pins_quantized(pin) || !self.matches(pin) || self.auxiliary_guards.is_some() {
            return Err(DataRebindError::new("quantized state slot target changed"));
        }
        self.topology_guards
            .rebind(&pin.topology, &self.workspace.topology, scope)?;
        match AuxiliaryGuards::try_acquire(self.index) {
            Ok(guards) => {
                self.auxiliary_guards = Some(guards);
                Ok(())
            }
            Err(error) => {
                self.topology_guards.release_state();
                Err(error)
            }
        }
    }

    pub(in crate::index::vector) fn install(
        &mut self,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'_, '_>,
    ) {
        self.workspace.prepared = false;
        self.topology_guards
            .install(&mut self.workspace.topology, scope);
        // The ready batch retains exclusive access to this paired slot from
        // successful binding through installation. No target lookup remains.
        if let Some(guards) = &mut self.auxiliary_guards {
            self.workspace.auxiliary.install(guards);
        }
    }

    pub(in crate::index::vector) fn release_state(&mut self) {
        drop(self.auxiliary_guards.take());
        self.topology_guards.release_state();
    }

    pub(in crate::index::vector) fn release_readers(&mut self) {
        self.topology_guards.release_readers();
    }
}

impl Drop for QuantizedMaintenanceSlot<'_> {
    fn drop(&mut self) {
        self.release_state();
        self.release_readers();
    }
}

fn qualify_vector(vector: &[f32], dimensions: usize) -> Result<()> {
    if vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "vector must have the configured dimensions and finite values",
        ));
    }
    Ok(())
}

fn checked_elements(rows: usize, dimensions: usize) -> Result<()> {
    if rows
        .checked_mul(dimensions)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(AllocError::InsufficientSpace.into());
    }
    Ok(())
}

fn qualify_product_layout(dimensions: usize, partitions: usize) -> Result<()> {
    if dimensions == 0 || partitions == 0 || !dimensions.is_multiple_of(partitions) {
        return Err(invalid(
            "product partitions must divide positive vector dimensions",
        ));
    }
    checked_elements(256, dimensions)
}

fn qualify_scalar_model(model: &ScalarQuantizer, dimensions: usize) -> Result<()> {
    if model.dimensions() != dimensions || !model.has_valid_storage() {
        return Err(invalid(
            "scalar model has invalid dimensions or numeric range",
        ));
    }
    Ok(())
}

fn qualify_product_model(
    model: &ProductQuantizer,
    kind: QuantizationType,
    dimensions: usize,
    new_model: bool,
) -> Result<()> {
    let QuantizationType::Product { num_subvectors } = kind else {
        return Err(invalid("product model belongs to another quantized kind"));
    };
    qualify_product_layout(dimensions, num_subvectors)?;
    if model.dimensions() != dimensions
        || model.num_subvectors() != num_subvectors
        || model.num_centroids() == 0
        || model.num_centroids() > 256
        || model.subvector_dim() != dimensions / num_subvectors
    {
        return Err(invalid("product model has inconsistent parameters"));
    }
    let expected = dimensions
        .checked_mul(model.num_centroids())
        .ok_or(AllocError::InsufficientSpace)?;
    if model.centroid_values().len() != expected
        || (new_model
            && model
                .centroid_values()
                .iter()
                .any(|value| !value.is_finite()))
    {
        return Err(invalid(
            "product model has invalid centroid storage or numeric range",
        ));
    }
    Ok(())
}

fn reserve_vec<T>(values: &mut Vec<T>, additional: usize) -> Result<()> {
    values
        .try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

fn reserve_map<T>(values: &mut HashMap<NodeId, T>, additional: usize) -> Result<()> {
    values
        .try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

fn reserve_merge<T>(
    target: &mut HashMap<NodeId, T>,
    candidate: &HashMap<NodeId, T>,
    retirement: &mut Vec<T>,
) -> Result<()> {
    let missing = candidate
        .keys()
        .filter(|id| !target.contains_key(*id))
        .count();
    reserve_map(target, missing)?;
    reserve_vec(retirement, candidate.len())
}

fn install_merge<T>(
    target: &mut HashMap<NodeId, T>,
    candidate: &mut HashMap<NodeId, T>,
    retirement: &mut Vec<T>,
) {
    for (id, value) in candidate.drain() {
        if let Some(occupied) = target.get_mut(&id) {
            retirement.push(std::mem::replace(occupied, value));
        } else {
            target.entry(id).or_insert(value);
        }
    }
}

fn invalid(reason: &str) -> Error {
    TransactionError::InvalidState(format!("quantized maintenance preparation: {reason}")).into()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(super) fn binary_read_seam() {
    tests::binary_read_seam();
}
