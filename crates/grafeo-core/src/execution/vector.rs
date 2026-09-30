//! ValueVector for columnar data storage.

#[cfg(feature = "spill")]
use allocator_api2::alloc::Global;
#[cfg(feature = "spill")]
use allocator_api2::vec::Vec as ExactVec;
use arcstr::ArcStr;

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};
use std::collections::TryReserveError;
use std::fmt;

/// Default vector capacity (tuples per vector).
pub const DEFAULT_VECTOR_CAPACITY: usize = 2048;

/// Failure while reserving or measuring resident vector capacity.
///
/// Allocation failures retain the allocator's original [`TryReserveError`]
/// and a static container label. Constructing this error therefore does not
/// allocate another diagnostic string while the allocator is already under
/// pressure.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ResidentCapacityError {
    /// The allocator refused a requested container capacity.
    Allocation {
        /// Static identity of the container that could not grow.
        container: &'static str,
        /// Original allocation failure.
        source: TryReserveError,
    },
    /// The pinned exact allocator refused a requested container capacity.
    #[cfg(feature = "spill")]
    ExactAllocation {
        /// Static identity of the container that could not grow.
        container: &'static str,
        /// Original exact-allocator failure.
        source: ExactVectorAllocationError,
    },
    /// The pinned exact allocator violated its requested-capacity contract.
    #[cfg(feature = "spill")]
    ExactCapacityContract {
        /// Static identity of the affected container.
        container: &'static str,
        /// Requested element capacity.
        requested: usize,
        /// Capacity reported by the allocation.
        observed: usize,
    },
    /// A capacity-to-byte calculation exceeded the platform address space.
    ArithmeticOverflow {
        /// Static identity of the capacity calculation that overflowed.
        container: &'static str,
    },
}

impl ResidentCapacityError {
    fn allocation(container: &'static str, source: TryReserveError) -> Self {
        Self::Allocation { container, source }
    }

    fn overflow(container: &'static str) -> Self {
        Self::ArithmeticOverflow { container }
    }

    #[cfg(feature = "spill")]
    pub(crate) fn exact_allocation(
        container: &'static str,
        source: allocator_api2::collections::TryReserveError,
    ) -> Self {
        Self::ExactAllocation {
            container,
            source: ExactVectorAllocationError::from_allocator_api2(source),
        }
    }

    /// Returns the static identity of the container that failed.
    #[must_use]
    pub const fn container(&self) -> &'static str {
        match self {
            Self::Allocation { container, .. } | Self::ArithmeticOverflow { container } => {
                container
            }
            #[cfg(feature = "spill")]
            Self::ExactAllocation { container, .. }
            | Self::ExactCapacityContract { container, .. } => container,
        }
    }
}

/// Stable classification of an exact vector reservation failure.
#[cfg(feature = "spill")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExactVectorAllocationKind {
    /// The requested capacity exceeded the collection maximum.
    CapacityOverflow,
    /// The allocator refused an otherwise valid layout.
    AllocatorRefused {
        /// Requested allocation size.
        requested_bytes: usize,
        /// Requested allocation alignment.
        alignment: usize,
    },
}

/// Grafeo-owned envelope for the pinned exact-vector allocator failure.
///
/// The dependency error remains the private source, keeping allocator-api2
/// out of Grafeo's public semver surface while preserving full diagnostics.
#[cfg(feature = "spill")]
#[derive(Clone, Debug)]
pub struct ExactVectorAllocationError {
    kind: ExactVectorAllocationKind,
    source: Option<allocator_api2::collections::TryReserveError>,
}

#[cfg(feature = "spill")]
impl ExactVectorAllocationError {
    fn from_allocator_api2(source: allocator_api2::collections::TryReserveError) -> Self {
        use allocator_api2::collections::TryReserveErrorKind;
        let kind = match source.kind() {
            TryReserveErrorKind::CapacityOverflow => ExactVectorAllocationKind::CapacityOverflow,
            TryReserveErrorKind::AllocError { layout, .. } => {
                ExactVectorAllocationKind::AllocatorRefused {
                    requested_bytes: layout.size(),
                    alignment: layout.align(),
                }
            }
        };
        Self {
            kind,
            source: Some(source),
        }
    }

    /// Returns the stable allocation-failure classification.
    #[must_use]
    pub const fn kind(&self) -> ExactVectorAllocationKind {
        self.kind
    }
}

#[cfg(feature = "spill")]
impl fmt::Display for ExactVectorAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ExactVectorAllocationKind::CapacityOverflow => formatter
                .write_str("computed exact vector capacity exceeded the collection maximum"),
            ExactVectorAllocationKind::AllocatorRefused {
                requested_bytes,
                alignment,
            } => write!(
                formatter,
                "allocator refused a {requested_bytes}-byte exact vector layout with {alignment}-byte alignment"
            ),
        }
    }
}

#[cfg(feature = "spill")]
impl std::error::Error for ExactVectorAllocationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

impl fmt::Display for ResidentCapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation { container, source } => {
                write!(
                    formatter,
                    "allocator refused {container} capacity: {source}"
                )
            }
            #[cfg(feature = "spill")]
            Self::ExactAllocation { container, source } => {
                write!(
                    formatter,
                    "exact allocator refused {container} capacity: {source}"
                )
            }
            #[cfg(feature = "spill")]
            Self::ExactCapacityContract {
                container,
                requested,
                observed,
            } => write!(
                formatter,
                "exact allocator reported {observed} {container} slots for {requested} requested"
            ),
            Self::ArithmeticOverflow { container } => {
                write!(
                    formatter,
                    "{container} capacity exceeds the platform address space"
                )
            }
        }
    }
}

impl std::error::Error for ResidentCapacityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation { source, .. } => Some(source),
            #[cfg(feature = "spill")]
            Self::ExactAllocation { source, .. } => Some(source),
            #[cfg(feature = "spill")]
            Self::ExactCapacityContract { .. } => None,
            Self::ArithmeticOverflow { .. } => None,
        }
    }
}

/// Allocates through allocator-api2's `Global`, validates its exact reported
/// layout, then hands the allocation to std `Vec` without reallocating.
///
/// `allocator_api2::alloc::Global` forwards to the same `alloc`/`dealloc`
/// entry points used by std's global `Vec`. The bridge preserves the element
/// type, pointer, length, and capacity exactly, so std later deallocates with
/// the identical global allocator and `Layout::array::<T>(capacity)` contract.
#[cfg(feature = "spill")]
#[allow(
    unsafe_code,
    reason = "audited raw-parts bridge between allocator-api2 Global and std global Vec"
)]
fn try_exact_global_vec<T>(
    capacity: usize,
    container: &'static str,
) -> Result<Vec<T>, ResidentCapacityError> {
    let mut exact = ExactVec::new_in(Global);
    exact
        .try_reserve_exact(capacity)
        .map_err(|source| ResidentCapacityError::exact_allocation(container, source))?;
    let observed = exact.capacity();
    if observed != capacity {
        return Err(ResidentCapacityError::ExactCapacityContract {
            container,
            requested: capacity,
            observed,
        });
    }
    let (pointer, length, capacity, _global) = exact.into_raw_parts_with_alloc();
    // SAFETY: allocator-api2 `Global` and std `Vec` both use the process global
    // allocator. The exact pointer/type/length/capacity are unchanged, and the
    // allocator-api2 owner was consumed without deallocating the allocation.
    Ok(unsafe { Vec::from_raw_parts(pointer, length, capacity) })
}

/// Creates exact-capacity outer storage for a sealed column catalog.
#[cfg(feature = "spill")]
pub(crate) fn try_exact_value_vector_catalog(
    capacity: usize,
) -> Result<Vec<ValueVector>, ResidentCapacityError> {
    try_exact_global_vec(capacity, "exact value-vector catalog")
}

fn try_vec_with_capacity<T>(
    capacity: usize,
    container: &'static str,
) -> Result<Vec<T>, ResidentCapacityError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|source| ResidentCapacityError::allocation(container, source))?;
    Ok(values)
}

fn checked_capacity_bytes<T>(
    capacity: usize,
    container: &'static str,
) -> Result<usize, ResidentCapacityError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| ResidentCapacityError::overflow(container))
}

/// Reuses the admitted exact Global Vec allocation as a single std Box.
/// Layout::array::<LogicalType>(1) equals Layout::new::<LogicalType>(); the
/// initialized element and allocation transfer once, with no reallocation.
#[cfg(feature = "spill")]
#[allow(
    unsafe_code,
    reason = "audited exact one-element Global Vec to std Box ownership bridge"
)]
fn try_exact_edge_schema_box() -> Result<Box<LogicalType>, ResidentCapacityError> {
    let mut storage = try_exact_global_vec(1, "exact edge-list schema")?;
    storage.push(LogicalType::Edge);
    let mut storage = std::mem::ManuallyDrop::new(storage);
    // SAFETY: the exact vector owns one initialized LogicalType, with length
    // and capacity one, allocated by the same Global allocator as std Box.
    Ok(unsafe { Box::from_raw(storage.as_mut_ptr()) })
}

/// A columnar vector of values.
///
/// ValueVector stores data in columnar format for efficient SIMD processing
/// and cache utilization during query execution.
#[derive(Debug, Clone)]
pub struct ValueVector {
    /// The logical type of values in this vector.
    data_type: LogicalType,
    /// The actual data storage.
    data: VectorData,
    /// Number of valid entries.
    len: usize,
    /// Validity bitmap (true = valid, false = null).
    validity: Option<Vec<bool>>,
}

fn output_schema_heap_bytes(ty: &LogicalType, depth: usize) -> Option<usize> {
    if depth > 128 {
        return None;
    }
    let boxed = |inner: &LogicalType| {
        std::mem::size_of::<LogicalType>().checked_add(output_schema_heap_bytes(inner, depth + 1)?)
    };
    match ty {
        LogicalType::List(inner) => boxed(inner),
        LogicalType::Map { key, value } => boxed(key)?.checked_add(boxed(value)?),
        LogicalType::Struct(fields) => {
            let mut bytes = fields
                .capacity()
                .checked_mul(std::mem::size_of::<(String, LogicalType)>())?;
            for (name, inner) in fields {
                bytes = bytes
                    .checked_add(name.capacity())?
                    .checked_add(output_schema_heap_bytes(inner, depth + 1)?)?;
            }
            Some(bytes)
        }
        LogicalType::Any
        | LogicalType::Null
        | LogicalType::Bool
        | LogicalType::Int8
        | LogicalType::Int16
        | LogicalType::Int32
        | LogicalType::Int64
        | LogicalType::Float32
        | LogicalType::Float64
        | LogicalType::String
        | LogicalType::Bytes
        | LogicalType::Date
        | LogicalType::Time
        | LogicalType::Timestamp
        | LogicalType::Duration
        | LogicalType::ZonedTime
        | LogicalType::ZonedDatetime
        | LogicalType::Node
        | LogicalType::Edge
        | LogicalType::Path
        | LogicalType::Vector(_) => Some(0),
        _ => None,
    }
}

/// Internal storage for vector data.
#[derive(Debug, Clone)]
enum VectorData {
    /// Boolean values.
    Bool(Vec<bool>),
    /// 64-bit integers.
    Int64(Vec<i64>),
    /// 64-bit floats.
    Float64(Vec<f64>),
    /// Strings (stored as ArcStr for cheap cloning).
    String(Vec<ArcStr>),
    /// Node IDs.
    NodeId(Vec<NodeId>),
    /// Edge IDs.
    EdgeId(Vec<EdgeId>),
    /// Generic values (fallback for complex types).
    Generic(Vec<Value>),
}

impl ValueVector {
    /// Creates the sealed one-row Generic lane used by exact accounted output.
    ///
    /// The one `Value` slot is allocated exactly and the consuming push cannot
    /// grow it. Null remains an embedded Generic value, avoiding a separate
    /// `Vec<bool>` allocation whose capacity could not share this proof.
    #[cfg(all(test, feature = "spill"))]
    pub(crate) fn try_exact_generic_one(value: Value) -> Result<Self, ResidentCapacityError> {
        Self::try_exact_generic_one_with_sort_type(
            value,
            super::accounted_chunk::SortValueType::Ordinary,
        )
    }

    /// Exact Generic storage with a closed private sort-provenance tag.
    /// Caller admits the Value slot and, for EdgeList only, its schema box.
    /// Taking a tag avoids allocating a boxed LogicalType before admission.
    #[cfg(feature = "spill")]
    pub(crate) fn try_exact_generic_one_with_sort_type(
        value: Value,
        value_type: super::accounted_chunk::SortValueType,
    ) -> Result<Self, ResidentCapacityError> {
        let mut values = try_exact_global_vec(1, "exact generic value vector")?;
        values.push(value);
        let data_type = match value_type {
            super::accounted_chunk::SortValueType::Ordinary => LogicalType::Any,
            super::accounted_chunk::SortValueType::EdgeList => {
                LogicalType::List(try_exact_edge_schema_box()?)
            }
            super::accounted_chunk::SortValueType::Edge => LogicalType::Edge,
            super::accounted_chunk::SortValueType::Node => LogicalType::Node,
        };
        Ok(Self {
            data_type,
            data: VectorData::Generic(values),
            len: 1,
            validity: None,
        })
    }

    /// Creates a new empty generic vector.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(LogicalType::Any, DEFAULT_VECTOR_CAPACITY)
    }

    /// Creates a new empty vector with the given type.
    #[must_use]
    pub fn with_type(data_type: LogicalType) -> Self {
        Self::with_capacity(data_type, DEFAULT_VECTOR_CAPACITY)
    }

    /// Creates a vector from a slice of values.
    pub fn from_values(values: &[Value]) -> Self {
        let mut vec = Self::new();
        for value in values {
            vec.push_value(value.clone());
        }
        vec
    }

    /// Creates a new vector with the given capacity.
    #[must_use]
    pub fn with_capacity(data_type: LogicalType, capacity: usize) -> Self {
        let data = match &data_type {
            LogicalType::Bool => VectorData::Bool(Vec::with_capacity(capacity)),
            LogicalType::Int8 | LogicalType::Int16 | LogicalType::Int32 | LogicalType::Int64 => {
                VectorData::Int64(Vec::with_capacity(capacity))
            }
            LogicalType::Float32 | LogicalType::Float64 => {
                VectorData::Float64(Vec::with_capacity(capacity))
            }
            LogicalType::String => VectorData::String(Vec::with_capacity(capacity)),
            LogicalType::Node => VectorData::NodeId(Vec::with_capacity(capacity)),
            LogicalType::Edge => VectorData::EdgeId(Vec::with_capacity(capacity)),
            _ => VectorData::Generic(Vec::with_capacity(capacity)),
        };

        Self {
            data_type,
            data,
            len: 0,
            validity: None,
        }
    }

    /// Fallibly creates a new vector with the requested backing capacity.
    ///
    /// Unlike [`Self::with_capacity`], allocator refusal is returned with its
    /// original structured source instead of aborting through an infallible
    /// `Vec` allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ResidentCapacityError::Allocation`] if the backing `Vec`
    /// cannot reserve the requested capacity.
    pub fn try_with_capacity(
        data_type: LogicalType,
        capacity: usize,
    ) -> Result<Self, ResidentCapacityError> {
        let data = match &data_type {
            LogicalType::Bool => {
                VectorData::Bool(try_vec_with_capacity(capacity, "boolean value vector")?)
            }
            LogicalType::Int8 | LogicalType::Int16 | LogicalType::Int32 | LogicalType::Int64 => {
                VectorData::Int64(try_vec_with_capacity(capacity, "integer value vector")?)
            }
            LogicalType::Float32 | LogicalType::Float64 => VectorData::Float64(
                try_vec_with_capacity(capacity, "floating-point value vector")?,
            ),
            LogicalType::String => {
                VectorData::String(try_vec_with_capacity(capacity, "string value vector")?)
            }
            LogicalType::Node => {
                VectorData::NodeId(try_vec_with_capacity(capacity, "node-id value vector")?)
            }
            LogicalType::Edge => {
                VectorData::EdgeId(try_vec_with_capacity(capacity, "edge-id value vector")?)
            }
            _ => VectorData::Generic(try_vec_with_capacity(capacity, "generic value vector")?),
        };

        Ok(Self {
            data_type,
            data,
            len: 0,
            validity: None,
        })
    }

    /// Fallibly reserves optional null-validity storage to at least `capacity`.
    ///
    /// Existing rows are represented as valid when the bitmap is created.
    /// A replacement bitmap is fully reserved off-object before publication,
    /// so a failed first allocation does not change logical values or nullness.
    /// An existing bitmap can gain physical capacity on a failed later reserve;
    /// callers can reconcile that growth through
    /// [`Self::observed_capacity_bytes`].
    ///
    /// # Errors
    ///
    /// Returns a structured allocation failure if the bitmap cannot reserve
    /// the requested capacity.
    pub fn try_reserve_validity_capacity(
        &mut self,
        capacity: usize,
    ) -> Result<(), ResidentCapacityError> {
        let target = capacity.max(self.len);
        if target == 0 {
            return Ok(());
        }

        if let Some(validity) = &mut self.validity {
            if validity.capacity() < target {
                let additional = target.saturating_sub(validity.len());
                validity.try_reserve_exact(additional).map_err(|source| {
                    ResidentCapacityError::allocation("value vector validity", source)
                })?;
            }
            if validity.len() < self.len {
                validity.resize(self.len, true);
            }
            return Ok(());
        }

        let mut validity = try_vec_with_capacity(target, "value vector validity")?;
        validity.resize(self.len, true);
        self.validity = Some(validity);
        Ok(())
    }

    /// Returns the observed direct backing capacity owned by this vector.
    ///
    /// This counts the concrete data `Vec` and optional byte-per-bool validity vector.
    /// It deliberately does not recursively charge pointees shared through
    /// `Arc`/`ArcStr`; ownership accounting for those payloads must travel with
    /// the authority that originally admitted them.
    ///
    /// # Errors
    ///
    /// Returns [`ResidentCapacityError::ArithmeticOverflow`] if converting an
    /// observed element capacity to bytes exceeds `usize`.
    pub fn observed_capacity_bytes(&self) -> Result<usize, ResidentCapacityError> {
        let data_bytes = match &self.data {
            VectorData::Bool(values) => {
                checked_capacity_bytes::<bool>(values.capacity(), "boolean value vector")?
            }
            VectorData::Int64(values) => {
                checked_capacity_bytes::<i64>(values.capacity(), "integer value vector")?
            }
            VectorData::Float64(values) => {
                checked_capacity_bytes::<f64>(values.capacity(), "floating-point value vector")?
            }
            VectorData::String(values) => {
                checked_capacity_bytes::<ArcStr>(values.capacity(), "string value vector")?
            }
            VectorData::NodeId(values) => {
                checked_capacity_bytes::<NodeId>(values.capacity(), "node-id value vector")?
            }
            VectorData::EdgeId(values) => {
                checked_capacity_bytes::<EdgeId>(values.capacity(), "edge-id value vector")?
            }
            VectorData::Generic(values) => {
                checked_capacity_bytes::<Value>(values.capacity(), "generic value vector")?
            }
        };
        let validity_bytes = match &self.validity {
            Some(validity) => {
                checked_capacity_bytes::<bool>(validity.capacity(), "value vector validity")?
            }
            None => 0,
        };
        // List(Edge) carries a concrete schema box in the sort provenance lane.
        let schema_bytes = if matches!(&self.data_type, LogicalType::List(item) if item.as_ref() == &LogicalType::Edge)
        {
            std::mem::size_of::<LogicalType>()
        } else {
            0
        };
        data_bytes
            .checked_add(validity_bytes)
            .and_then(|bytes| bytes.checked_add(schema_bytes))
            .ok_or_else(|| ResidentCapacityError::overflow("value vector data and validity"))
    }

    /// Bounds all storage retained by this vector at the public output boundary.
    ///
    /// Includes spare direct capacity, nested schema and every physically stored
    /// shared payload, even when validity or logical length hides that payload.
    /// Shared pointees are conservatively charged once per ownership edge. This
    /// measurement does not mint a sealed operator admission or prove that the
    /// upstream allocation was reserved before construction.
    ///
    /// # Errors
    /// Returns an arithmetic error for overflow, unsupported schema, or nesting
    /// beyond the bounded measurement depth.
    pub fn output_retained_bytes(&self) -> Result<usize, ResidentCapacityError> {
        let measure = || {
            let mut bytes = self.observed_capacity_bytes().ok()?;
            let schema = output_schema_heap_bytes(&self.data_type, 0)?;
            let measured_schema = if matches!(&self.data_type, LogicalType::List(inner) if **inner == LogicalType::Edge)
            {
                std::mem::size_of::<LogicalType>()
            } else {
                0
            };
            bytes = bytes.checked_add(schema.checked_sub(measured_schema)?)?;
            match &self.data {
                VectorData::String(values) => {
                    let per_pointee = 2 * std::mem::size_of::<usize>() + 7;
                    bytes = bytes.checked_add(values.len().checked_mul(per_pointee)?)?;
                    let mut lengths = 0usize;
                    let mut overflowed = false;
                    for value in values {
                        let (next, overflow) = lengths.overflowing_add(value.len());
                        lengths = next;
                        overflowed |= overflow;
                    }
                    // Every term is nonnegative: any wrap invalidates the bound,
                    // including when subsequent additions wrap back again.
                    if overflowed {
                        return None;
                    }
                    bytes = bytes.checked_add(lengths)?;
                }
                VectorData::Generic(values) => {
                    for value in values {
                        bytes = bytes.checked_add(
                            value
                                .retained_size_bytes()?
                                .checked_sub(std::mem::size_of::<Value>())?,
                        )?;
                    }
                }
                VectorData::Bool(_)
                | VectorData::Int64(_)
                | VectorData::Float64(_)
                | VectorData::NodeId(_)
                | VectorData::EdgeId(_) => {}
            }
            Some(bytes)
        };
        measure().ok_or_else(|| ResidentCapacityError::overflow("complete output vector storage"))
    }

    /// Returns the observed validity capacity in boolean elements.
    #[must_use]
    pub fn validity_capacity(&self) -> usize {
        self.validity.as_ref().map_or(0, Vec::capacity)
    }

    /// Returns the data type of this vector.
    #[must_use]
    pub fn data_type(&self) -> &LogicalType {
        &self.data_type
    }

    /// Returns the number of entries in this vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the value at index is null.
    #[must_use]
    pub fn is_null(&self, index: usize) -> bool {
        match &self.validity {
            Some(validity) => !validity.get(index).copied().unwrap_or(true),
            None => matches!(
                &self.data,
                VectorData::Generic(values) if matches!(values.get(index), Some(Value::Null))
            ),
        }
    }

    /// Sets the value at index to null.
    pub fn set_null(&mut self, index: usize) {
        if self.validity.is_none() {
            self.validity = Some(vec![true; index + 1]);
        }
        if let Some(validity) = &mut self.validity {
            if validity.len() <= index {
                validity.resize(index + 1, true);
            }
            validity[index] = false;
        }
    }

    /// Pushes a boolean value.
    pub fn push_bool(&mut self, value: bool) {
        match &mut self.data {
            VectorData::Bool(vec) => {
                vec.push(value);
                self.len += 1;
            }
            VectorData::Generic(vec) => {
                vec.push(Value::Bool(value));
                self.len += 1;
            }
            _ => {}
        }
    }

    /// Pushes an integer value.
    pub fn push_int64(&mut self, value: i64) {
        match &mut self.data {
            VectorData::Int64(vec) => {
                vec.push(value);
                self.len += 1;
            }
            VectorData::Generic(vec) => {
                vec.push(Value::Int64(value));
                self.len += 1;
            }
            _ => {}
        }
    }

    /// Pushes a float value.
    pub fn push_float64(&mut self, value: f64) {
        match &mut self.data {
            VectorData::Float64(vec) => {
                vec.push(value);
                self.len += 1;
            }
            VectorData::Generic(vec) => {
                vec.push(Value::Float64(value));
                self.len += 1;
            }
            _ => {}
        }
    }

    /// Pushes a string value.
    pub fn push_string(&mut self, value: impl Into<ArcStr>) {
        match &mut self.data {
            VectorData::String(vec) => {
                vec.push(value.into());
                self.len += 1;
            }
            VectorData::Generic(vec) => {
                vec.push(Value::String(value.into()));
                self.len += 1;
            }
            _ => {}
        }
    }

    /// Pushes a node ID.
    pub fn push_node_id(&mut self, value: NodeId) {
        match &mut self.data {
            VectorData::NodeId(vec) => {
                vec.push(value);
                self.len += 1;
            }
            VectorData::Generic(vec) => {
                // reason: entity IDs stored as i64, standard encoding
                #[allow(clippy::cast_possible_wrap)]
                vec.push(Value::Int64(value.as_u64() as i64));
                self.len += 1;
            }
            _ => {}
        }
    }

    /// Pushes an edge ID.
    pub fn push_edge_id(&mut self, value: EdgeId) {
        match &mut self.data {
            VectorData::EdgeId(vec) => {
                vec.push(value);
                self.len += 1;
            }
            VectorData::Generic(vec) => {
                // reason: entity IDs stored as i64, standard encoding
                #[allow(clippy::cast_possible_wrap)]
                vec.push(Value::Int64(value.as_u64() as i64));
                self.len += 1;
            }
            _ => {}
        }
    }

    /// Pushes a generic value.
    pub fn push_value(&mut self, value: Value) {
        // Handle null values specially - push a default and mark as null
        if matches!(value, Value::Null) {
            match &mut self.data {
                VectorData::Bool(vec) => vec.push(false),
                VectorData::Int64(vec) => vec.push(0),
                VectorData::Float64(vec) => vec.push(0.0),
                VectorData::String(vec) => vec.push(ArcStr::new()),
                VectorData::NodeId(vec) => vec.push(NodeId::new(0)),
                VectorData::EdgeId(vec) => vec.push(EdgeId::new(0)),
                VectorData::Generic(vec) => vec.push(Value::Null),
            }
            self.len += 1;
            self.set_null(self.len - 1);
            return;
        }

        match (&mut self.data, &value) {
            (VectorData::Bool(vec), Value::Bool(b)) => vec.push(*b),
            (VectorData::Int64(vec), Value::Int64(i)) => vec.push(*i),
            (VectorData::Float64(vec), Value::Float64(f)) => vec.push(*f),
            (VectorData::String(vec), Value::String(s)) => vec.push(s.clone()),
            // Handle Int64 -> NodeId conversion (from get_value roundtrip)
            // reason: ID encoding: i64 <-> u64 round-trip
            #[allow(clippy::cast_sign_loss)]
            (VectorData::NodeId(vec), Value::Int64(i)) => vec.push(NodeId::new(*i as u64)),
            // Handle Int64 -> EdgeId conversion (from get_value roundtrip)
            // reason: ID encoding: i64 <-> u64 round-trip
            #[allow(clippy::cast_sign_loss)]
            (VectorData::EdgeId(vec), Value::Int64(i)) => vec.push(EdgeId::new(*i as u64)),
            (VectorData::Generic(vec), _) => vec.push(value),
            _ => {
                // Type mismatch - push a default value to maintain vector alignment
                match &mut self.data {
                    VectorData::Bool(vec) => vec.push(false),
                    VectorData::Int64(vec) => vec.push(0),
                    VectorData::Float64(vec) => vec.push(0.0),
                    VectorData::String(vec) => vec.push(ArcStr::new()),
                    VectorData::NodeId(vec) => vec.push(NodeId::new(0)),
                    VectorData::EdgeId(vec) => vec.push(EdgeId::new(0)),
                    VectorData::Generic(vec) => vec.push(value),
                }
            }
        }
        self.len += 1;
    }

    /// Fallibly reserves every vector component needed for one value, then
    /// publishes that value using the existing push semantics.
    ///
    /// No row is logically visible until the data backing and any null
    /// validity growth have both succeeded. If validity reservation fails
    /// after the data `Vec` grew, `len` and all values remain unchanged and
    /// [`Self::observed_capacity_bytes`] reports the capacity that must remain
    /// accounted.
    ///
    /// # Errors
    ///
    /// Returns a structured allocation or capacity-overflow error before the
    /// new value is logically published.
    pub fn try_push_value(&mut self, value: Value) -> Result<(), ResidentCapacityError> {
        let new_len = self
            .len
            .checked_add(1)
            .ok_or_else(|| ResidentCapacityError::overflow("value vector length"))?;
        self.try_push_value_with_validity_capacity(value, new_len)
    }

    fn try_push_value_with_validity_capacity(
        &mut self,
        value: Value,
        requested_validity_capacity: usize,
    ) -> Result<(), ResidentCapacityError> {
        let new_len = self
            .len
            .checked_add(1)
            .ok_or_else(|| ResidentCapacityError::overflow("value vector length"))?;

        match &mut self.data {
            VectorData::Bool(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("boolean value vector", source)
            })?,
            VectorData::Int64(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("integer value vector", source)
            })?,
            VectorData::Float64(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("floating-point value vector", source)
            })?,
            VectorData::String(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("string value vector", source)
            })?,
            VectorData::NodeId(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("node-id value vector", source)
            })?,
            VectorData::EdgeId(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("edge-id value vector", source)
            })?,
            VectorData::Generic(values) => values.try_reserve_exact(1).map_err(|source| {
                ResidentCapacityError::allocation("generic value vector", source)
            })?,
        }

        let replacement_validity = if matches!(value, Value::Null) {
            let validity_target = requested_validity_capacity.max(new_len);
            if let Some(validity) = &mut self.validity {
                if validity.capacity() < validity_target {
                    let additional = validity_target.saturating_sub(validity.len());
                    validity.try_reserve_exact(additional).map_err(|source| {
                        ResidentCapacityError::allocation("value vector validity", source)
                    })?;
                }
                None
            } else {
                let mut validity = try_vec_with_capacity(validity_target, "value vector validity")?;
                validity.resize(self.len, true);
                Some(validity)
            }
        } else {
            None
        };

        if let Some(validity) = replacement_validity {
            self.validity = Some(validity);
        }
        self.push_value(value);
        debug_assert_eq!(self.len, new_len);
        Ok(())
    }

    /// Gets a boolean value at index.
    #[must_use]
    pub fn get_bool(&self, index: usize) -> Option<bool> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::Bool(vec) = &self.data {
            vec.get(index).copied()
        } else {
            None
        }
    }

    /// Gets an integer value at index.
    #[must_use]
    pub fn get_int64(&self, index: usize) -> Option<i64> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::Int64(vec) = &self.data {
            vec.get(index).copied()
        } else {
            None
        }
    }

    /// Gets a float value at index.
    #[must_use]
    pub fn get_float64(&self, index: usize) -> Option<f64> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::Float64(vec) = &self.data {
            vec.get(index).copied()
        } else {
            None
        }
    }

    /// Gets a string value at index.
    #[must_use]
    pub fn get_string(&self, index: usize) -> Option<&str> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::String(vec) = &self.data {
            vec.get(index).map(|s| s.as_ref())
        } else {
            None
        }
    }

    /// Gets a node ID at index.
    #[must_use]
    pub fn get_node_id(&self, index: usize) -> Option<NodeId> {
        if self.is_null(index) {
            return None;
        }
        match &self.data {
            VectorData::NodeId(vec) => vec.get(index).copied(),
            // Handle Generic vectors that contain node IDs stored as Int64
            VectorData::Generic(vec) if self.data_type != LogicalType::Edge => match vec.get(index)
            {
                // reason: ID encoding: i64 <-> u64 round-trip
                #[allow(clippy::cast_sign_loss)]
                Some(Value::Int64(i)) => Some(NodeId::new(*i as u64)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Gets an edge ID at index.
    #[must_use]
    pub fn get_edge_id(&self, index: usize) -> Option<EdgeId> {
        if self.is_null(index) {
            return None;
        }
        match &self.data {
            VectorData::EdgeId(vec) => vec.get(index).copied(),
            // Handle Generic vectors that contain edge IDs stored as Int64
            VectorData::Generic(vec) if self.data_type != LogicalType::Node => match vec.get(index)
            {
                // reason: ID encoding: i64 <-> u64 round-trip
                #[allow(clippy::cast_sign_loss)]
                Some(Value::Int64(i)) => Some(EdgeId::new(*i as u64)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Gets a value at index as a generic Value.
    #[must_use]
    pub fn get_value(&self, index: usize) -> Option<Value> {
        if self.is_null(index) {
            return Some(Value::Null);
        }

        match &self.data {
            VectorData::Bool(vec) => vec.get(index).map(|&v| Value::Bool(v)),
            VectorData::Int64(vec) => vec.get(index).map(|&v| Value::Int64(v)),
            VectorData::Float64(vec) => vec.get(index).map(|&v| Value::Float64(v)),
            VectorData::String(vec) => vec.get(index).map(|v| Value::String(v.clone())),
            // reason: entity IDs stored as i64, standard encoding
            VectorData::NodeId(vec) => vec.get(index).map(|&v| {
                // reason: entity IDs are sequential counters, well within i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int64(v.as_u64() as i64);
                val
            }),
            // reason: entity IDs stored as i64, standard encoding
            // reason: entity IDs are sequential counters, well within i64::MAX
            VectorData::EdgeId(vec) => vec.get(index).map(|&v| {
                // reason: entity IDs are sequential counters, well within i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int64(v.as_u64() as i64);
                val
            }),
            VectorData::Generic(vec) => vec.get(index).cloned(),
        }
    }

    /// Bounds the retained Value produced at `index` without cloning it.
    ///
    /// Includes the output Value slot and all retained shared backings. Returns
    /// `None` for an invalid index or an unrepresentable resident bound.
    #[must_use]
    pub fn retained_value_bytes(&self, index: usize) -> Option<usize> {
        if index >= self.len {
            return None;
        }
        if self.is_null(index) {
            return Some(std::mem::size_of::<Value>());
        }
        match &self.data {
            VectorData::Generic(values) => values.get(index)?.retained_size_bytes(),
            VectorData::String(values) => std::mem::size_of::<Value>()
                .checked_add(values.get(index)?.len())?
                .checked_add(2 * std::mem::size_of::<usize>())?
                .checked_add(7),
            _ => Some(std::mem::size_of::<Value>()),
        }
    }

    /// Alias for get_value.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        self.get_value(index)
    }

    /// Alias for push_value.
    pub fn push(&mut self, value: Value) {
        self.push_value(value);
    }

    /// Returns a slice of the underlying boolean data.
    #[must_use]
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        if let VectorData::Bool(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying integer data.
    #[must_use]
    pub fn as_int64_slice(&self) -> Option<&[i64]> {
        if let VectorData::Int64(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying float data.
    #[must_use]
    pub fn as_float64_slice(&self) -> Option<&[f64]> {
        if let VectorData::Float64(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying node ID data.
    #[must_use]
    pub fn as_node_id_slice(&self) -> Option<&[NodeId]> {
        if let VectorData::NodeId(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Appends `other` when both sides are dense typed integers / node ids.
    ///
    /// Used by last-hop `id(c)` so Project does not `get_value`/`push_value`
    /// 640k times.
    pub fn try_extend_from(&mut self, other: &Self) -> bool {
        if self.validity.is_some() || other.validity.is_some() {
            return false;
        }
        match (&mut self.data, &other.data) {
            (VectorData::Int64(dst), VectorData::Int64(src)) => {
                dst.extend_from_slice(src);
                self.len += src.len();
                true
            }
            (VectorData::Int64(dst), VectorData::NodeId(src)) => {
                dst.extend(src.iter().map(|n| {
                    // reason: entity IDs are sequential counters, well within i64::MAX
                    #[allow(clippy::cast_possible_wrap)]
                    {
                        n.as_u64() as i64
                    }
                }));
                self.len += src.len();
                true
            }
            (VectorData::NodeId(dst), VectorData::NodeId(src)) => {
                dst.extend_from_slice(src);
                self.len += src.len();
                true
            }
            _ => false,
        }
    }

    /// Returns a slice of the underlying edge ID data.
    #[must_use]
    pub fn as_edge_id_slice(&self) -> Option<&[EdgeId]> {
        if let VectorData::EdgeId(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns the logical type of this vector.
    #[must_use]
    pub fn logical_type(&self) -> LogicalType {
        self.data_type.clone()
    }

    /// Copies a row from this vector to the destination vector.
    ///
    /// The destination vector should have a compatible type. The value at `row`
    /// is read from this vector and pushed to the destination vector.
    pub fn copy_row_to(&self, row: usize, dest: &mut ValueVector) {
        if self.is_null(row) {
            dest.push_value(Value::Null);
            return;
        }

        match &self.data {
            VectorData::Bool(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_bool(v);
                }
            }
            VectorData::Int64(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_int64(v);
                }
            }
            VectorData::Float64(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_float64(v);
                }
            }
            VectorData::String(vec) => {
                if let Some(v) = vec.get(row) {
                    dest.push_string(v.clone());
                }
            }
            VectorData::NodeId(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_node_id(v);
                }
            }
            VectorData::EdgeId(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_edge_id(v);
                }
            }
            VectorData::Generic(vec) => {
                if let Some(v) = vec.get(row) {
                    dest.push_value(v.clone());
                }
            }
        }
    }

    /// Clears all data from this vector.
    pub fn clear(&mut self) {
        match &mut self.data {
            VectorData::Bool(vec) => vec.clear(),
            VectorData::Int64(vec) => vec.clear(),
            VectorData::Float64(vec) => vec.clear(),
            VectorData::String(vec) => vec.clear(),
            VectorData::NodeId(vec) => vec.clear(),
            VectorData::EdgeId(vec) => vec.clear(),
            VectorData::Generic(vec) => vec.clear(),
        }
        self.len = 0;
        self.validity = None;
    }
}

impl Default for ValueVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_storage_counts_masked_and_logically_truncated_payloads() {
        for ty in [LogicalType::String, LogicalType::Any] {
            let mut vector = ValueVector::with_capacity(ty, 8);
            vector.push_value(Value::from("x".repeat(8192)));
            vector.push_value(Value::from("y".repeat(4096)));
            let original = vector.output_retained_bytes().unwrap();
            vector.set_null(0);
            let masked = vector.output_retained_bytes().unwrap();
            assert!(masked >= original);
            vector.len = 0;
            assert_eq!(vector.output_retained_bytes().unwrap(), masked);
            assert!(masked >= vector.observed_capacity_bytes().unwrap() + 12288);
        }
    }

    #[test]
    fn output_storage_counts_schema_spare_capacity_and_rejects_deep_schema() {
        let mut name = String::with_capacity(1024);
        name.push_str("field");
        let mut fields = Vec::with_capacity(16);
        fields.push((
            name,
            LogicalType::Map {
                key: Box::new(LogicalType::String),
                value: Box::new(LogicalType::List(Box::new(LogicalType::Edge))),
            },
        ));
        let vector = ValueVector::with_capacity(LogicalType::Struct(fields), 32);
        assert!(
            vector.output_retained_bytes().unwrap()
                >= vector.observed_capacity_bytes().unwrap()
                    + 1024
                    + 16 * std::mem::size_of::<(String, LogicalType)>()
                    + 3 * std::mem::size_of::<LogicalType>()
        );
        let mut ty = LogicalType::Any;
        for _ in 0..130 {
            ty = LogicalType::List(Box::new(ty));
        }
        assert!(ValueVector::with_type(ty).output_retained_bytes().is_err());
    }

    #[test]
    fn retained_value_bytes_bounds_typed_string_generic_and_null_without_mutation() {
        let mut strings = ValueVector::with_type(LogicalType::String);
        strings.push_value(Value::String("a deliberately retained string".into()));
        let before = strings.len();
        let bound = strings.retained_value_bytes(0).unwrap();
        assert!(bound >= strings.get_value(0).unwrap().retained_size_bytes().unwrap());
        assert_eq!(strings.len(), before);

        let mut typed = ValueVector::with_type(LogicalType::Int64);
        typed.push_value(Value::Int64(42));
        assert!(
            typed.retained_value_bytes(0).unwrap()
                >= typed.get_value(0).unwrap().retained_size_bytes().unwrap()
        );

        let mut generic = ValueVector::with_type(LogicalType::Any);
        generic.push_value(Value::List(vec![Value::String("nested".into())].into()));
        assert!(
            generic.retained_value_bytes(0).unwrap()
                >= generic.get_value(0).unwrap().retained_size_bytes().unwrap()
        );
        generic.push_value(Value::Null);
        assert_eq!(
            generic.retained_value_bytes(1),
            Some(std::mem::size_of::<Value>())
        );
    }

    #[test]
    fn fallible_capacity_failure_keeps_static_context_and_source() {
        let error = ValueVector::try_with_capacity(LogicalType::Any, usize::MAX)
            .expect_err("an impossible generic vector capacity must fail");

        assert_eq!(error.container(), "generic value vector");
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn capacity_byte_overflow_keeps_static_context_without_a_source() {
        let error = checked_capacity_bytes::<Value>(usize::MAX, "test vector capacity")
            .expect_err("an impossible byte capacity must fail");

        assert_eq!(error.container(), "test vector capacity");
        assert!(matches!(
            error,
            ResidentCapacityError::ArithmeticOverflow { .. }
        ));
    }

    #[test]
    fn fallible_null_push_preserves_existing_and_trailing_valid_rows() {
        let mut vector = ValueVector::try_with_capacity(LogicalType::Int64, 4).unwrap();

        vector.try_push_value(Value::Int64(7)).unwrap();
        vector.try_push_value(Value::Null).unwrap();
        vector.try_push_value(Value::Int64(9)).unwrap();
        vector.try_push_value(Value::Null).unwrap();

        assert_eq!(vector.len(), 4);
        assert_eq!(vector.get_value(0), Some(Value::Int64(7)));
        assert_eq!(vector.get_value(1), Some(Value::Null));
        assert_eq!(vector.get_value(2), Some(Value::Int64(9)));
        assert_eq!(vector.get_value(3), Some(Value::Null));
        assert!(!vector.is_null(0));
        assert!(vector.is_null(1));
        assert!(!vector.is_null(2));
        assert!(vector.is_null(3));
    }

    #[test]
    fn failed_validity_reserve_does_not_publish_or_reclassify_values() {
        let mut vector = ValueVector::try_with_capacity(LogicalType::Int64, 2).unwrap();
        vector.try_push_value(Value::Int64(11)).unwrap();
        let capacity_before = vector.observed_capacity_bytes().unwrap();

        let error = vector
            .try_reserve_validity_capacity(usize::MAX)
            .expect_err("an impossible validity capacity must fail");

        assert_eq!(error.container(), "value vector validity");
        assert_eq!(vector.len(), 1);
        assert_eq!(vector.get_value(0), Some(Value::Int64(11)));
        assert!(!vector.is_null(0));
        assert_eq!(vector.observed_capacity_bytes().unwrap(), capacity_before);
    }

    #[test]
    fn failed_null_push_can_grow_capacity_but_does_not_publish_a_row() {
        let mut vector = ValueVector::try_with_capacity(LogicalType::Int64, 0).unwrap();
        let capacity_before = vector.observed_capacity_bytes().unwrap();

        let error = vector
            .try_push_value_with_validity_capacity(Value::Null, usize::MAX)
            .expect_err("an impossible validity capacity must fail after data preflight");

        assert_eq!(error.container(), "value vector validity");
        assert_eq!(vector.len(), 0);
        assert!(vector.is_empty());
        assert_eq!(vector.get_value(0), None);
        assert!(!vector.is_null(0));
        assert!(vector.observed_capacity_bytes().unwrap() > capacity_before);
    }

    #[test]
    fn boolean_output_counts_byte_sized_data_and_validity_capacity() {
        let mut vector = ValueVector::try_with_capacity(LogicalType::Bool, 257).unwrap();
        vector.try_push_value(Value::Bool(true)).unwrap();
        vector.try_reserve_validity_capacity(513).unwrap();
        let VectorData::Bool(values) = &vector.data else {
            panic!("boolean vector must retain boolean data");
        };
        let expected =
            (values.capacity() + vector.validity_capacity()) * std::mem::size_of::<bool>();
        assert_eq!(vector.observed_capacity_bytes().unwrap(), expected);
        assert_eq!(vector.output_retained_bytes().unwrap(), expected);
    }

    #[test]
    fn observed_capacity_includes_null_validity_storage() {
        let mut vector = ValueVector::try_with_capacity(LogicalType::Int64, 4).unwrap();
        let data_only = vector.observed_capacity_bytes().unwrap();

        vector.try_reserve_validity_capacity(4).unwrap();
        let with_validity = vector.observed_capacity_bytes().unwrap();

        assert_eq!(
            with_validity - data_only,
            vector.validity_capacity() * std::mem::size_of::<bool>()
        );
        assert!(vector.validity_capacity() >= 4);
    }

    #[test]
    fn fallible_push_matches_all_existing_value_and_mismatch_semantics() {
        let cases = [
            (
                LogicalType::Bool,
                vec![
                    Value::Bool(true),
                    Value::Null,
                    Value::Int64(3),
                    Value::Bool(false),
                ],
            ),
            (
                LogicalType::Int64,
                vec![
                    Value::Int64(-7),
                    Value::Null,
                    Value::String("mismatch".into()),
                    Value::Int64(9),
                ],
            ),
            (
                LogicalType::Float64,
                vec![
                    Value::Float64(1.25),
                    Value::Null,
                    Value::Bool(true),
                    Value::Float64(-0.5),
                ],
            ),
            (
                LogicalType::String,
                vec![
                    Value::String("first".into()),
                    Value::Null,
                    Value::Int64(4),
                    Value::String("last".into()),
                ],
            ),
            (
                LogicalType::Node,
                vec![
                    Value::Int64(12),
                    Value::Null,
                    Value::String("mismatch".into()),
                    Value::Int64(13),
                ],
            ),
            (
                LogicalType::Edge,
                vec![
                    Value::Int64(21),
                    Value::Null,
                    Value::Bool(false),
                    Value::Int64(22),
                ],
            ),
            (
                LogicalType::Any,
                vec![
                    Value::String("generic".into()),
                    Value::Null,
                    Value::Bool(true),
                    Value::Int64(99),
                ],
            ),
        ];

        for (data_type, values) in cases {
            let mut existing = ValueVector::with_capacity(data_type.clone(), values.len());
            let mut fallible =
                ValueVector::try_with_capacity(data_type.clone(), values.len()).unwrap();
            for value in values {
                existing.push_value(value.clone());
                fallible.try_push_value(value).unwrap();
            }

            assert_eq!(fallible.len(), existing.len(), "type {data_type:?}");
            for index in 0..existing.len() {
                assert_eq!(
                    fallible.get_value(index),
                    existing.get_value(index),
                    "type {data_type:?}, row {index}"
                );
                assert_eq!(
                    fallible.is_null(index),
                    existing.is_null(index),
                    "type {data_type:?}, row {index}"
                );
            }
        }
    }

    #[test]
    fn test_int64_vector() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);

        vec.push_int64(1);
        vec.push_int64(2);
        vec.push_int64(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get_int64(0), Some(1));
        assert_eq!(vec.get_int64(1), Some(2));
        assert_eq!(vec.get_int64(2), Some(3));
    }

    #[test]
    fn test_string_vector() {
        let mut vec = ValueVector::with_type(LogicalType::String);

        vec.push_string("hello");
        vec.push_string("world");

        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_string(0), Some("hello"));
        assert_eq!(vec.get_string(1), Some("world"));
    }

    #[test]
    fn test_null_values() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);

        vec.push_int64(1);
        vec.push_int64(2);
        vec.push_int64(3);

        assert!(!vec.is_null(1));
        vec.set_null(1);
        assert!(vec.is_null(1));

        assert_eq!(vec.get_int64(0), Some(1));
        assert_eq!(vec.get_int64(1), None); // Null
        assert_eq!(vec.get_int64(2), Some(3));
    }

    #[test]
    fn test_get_value() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(42);

        let value = vec.get_value(0);
        assert_eq!(value, Some(Value::Int64(42)));
    }

    #[test]
    fn test_slice_access() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(1);
        vec.push_int64(2);
        vec.push_int64(3);

        let slice = vec.as_int64_slice().unwrap();
        assert_eq!(slice, &[1, 2, 3]);
    }

    /// Typed push methods fall back to VectorData::Generic when the vector
    /// was created with LogicalType::Any. This exercises the safety-net arms
    /// added to prevent silent data loss on type mismatch.
    #[test]
    fn test_generic_fallback_push_int64() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_int64(42);
        vec.push_int64(-7);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::Int64(42)));
        assert_eq!(vec.get_value(1), Some(Value::Int64(-7)));
    }

    #[test]
    fn test_generic_fallback_push_bool() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_bool(true);
        vec.push_bool(false);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::Bool(true)));
        assert_eq!(vec.get_value(1), Some(Value::Bool(false)));
    }

    #[test]
    fn test_generic_fallback_push_float64() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_float64(1.23);
        vec.push_float64(-0.5);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::Float64(1.23)));
        assert_eq!(vec.get_value(1), Some(Value::Float64(-0.5)));
    }

    #[test]
    fn test_generic_fallback_push_string() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_string("hello");
        vec.push_string("world");
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::String("hello".into())));
        assert_eq!(vec.get_value(1), Some(Value::String("world".into())));
    }

    /// Mixed typed pushes into a Generic vector preserve each value's type.
    #[test]
    fn test_generic_fallback_mixed_types() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_int64(1);
        vec.push_string("two");
        vec.push_bool(true);
        vec.push_float64(99.5);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get_value(0), Some(Value::Int64(1)));
        assert_eq!(vec.get_value(1), Some(Value::String("two".into())));
        assert_eq!(vec.get_value(2), Some(Value::Bool(true)));
        assert_eq!(vec.get_value(3), Some(Value::Float64(99.5)));
    }

    /// Pushing a typed value into a mismatched non-Generic vector is a no-op.
    #[test]
    fn test_type_mismatch_noop() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_string("wrong type");
        assert_eq!(vec.len(), 0);

        let mut vec = ValueVector::with_type(LogicalType::String);
        vec.push_int64(42);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(1);
        vec.push_int64(2);

        vec.clear();

        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn scalar_provenance_exact_generic_accessor_excludes_opposite_entity() {
        // Model the exact output lane: one fallibly allocated Generic Value
        // slot plus a genuine scalar schema, never an inferred integer kind.
        for data_type in [LogicalType::Node, LogicalType::Edge] {
            let column = ValueVector::try_exact_generic_one_with_sort_type(
                Value::Int64(7),
                super::super::accounted_chunk::SortValueType::from_logical(&data_type),
            )
            .unwrap();
            assert_eq!(column.get_value(0), Some(Value::Int64(7)));
            assert_eq!(
                column.get_node_id(0),
                (data_type == LogicalType::Node).then_some(NodeId(7))
            );
            assert_eq!(
                column.get_edge_id(0),
                (data_type == LogicalType::Edge).then_some(EdgeId(7))
            );
        }
        let ordinary = ValueVector::try_exact_generic_one(Value::Int64(7)).unwrap();
        assert_eq!(ordinary.get_node_id(0), Some(NodeId(7)));
        assert_eq!(ordinary.get_edge_id(0), Some(EdgeId(7)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_generic_one_has_exact_capacity_embedded_null_and_no_validity_allocation() {
        let value = ValueVector::try_exact_generic_one(Value::Int64(7)).unwrap();
        assert_eq!(value.len(), 1);
        assert_eq!(value.observed_capacity_bytes().unwrap(), size_of::<Value>());
        assert_eq!(value.validity_capacity(), 0);
        assert!(!value.is_null(0));
        assert_eq!(value.get_value(0), Some(Value::Int64(7)));

        let null = ValueVector::try_exact_generic_one(Value::Null).unwrap();
        assert_eq!(null.observed_capacity_bytes().unwrap(), size_of::<Value>());
        assert_eq!(null.validity_capacity(), 0);
        assert!(null.is_null(0));
        assert_eq!(null.get_value(0), Some(Value::Null));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_global_bridge_moves_recursive_arc_and_outer_catalog_drops_it_once() {
        let list = std::sync::Arc::from([Value::Int64(1), Value::from("nested")]);
        let weak = std::sync::Arc::downgrade(&list);
        let column = ValueVector::try_exact_generic_one(Value::List(list)).unwrap();
        assert_eq!(weak.strong_count(), 1);
        let mut columns = try_exact_value_vector_catalog(1).unwrap();
        assert_eq!(columns.capacity(), 1);
        columns.push(column);
        assert_eq!(weak.strong_count(), 1);
        let retrieved = columns[0].get_value(0).unwrap();
        assert_eq!(weak.strong_count(), 2);
        drop(retrieved);
        assert_eq!(weak.strong_count(), 1);
        drop(columns);
        assert_eq!(weak.strong_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_global_bridge_returns_structured_impossible_capacity_failure() {
        let error = try_exact_value_vector_catalog(usize::MAX).unwrap_err();
        assert_eq!(error.container(), "exact value-vector catalog");
        assert!(matches!(
            error,
            ResidentCapacityError::ExactAllocation { .. }
        ));
        assert!(std::error::Error::source(&error).is_some());
    }
}
