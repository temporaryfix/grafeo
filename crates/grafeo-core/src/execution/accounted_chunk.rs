//! Move-only ownership for an admitted execution chunk.

use super::chunk::DataChunk;
#[cfg(feature = "spill")]
use super::value_codec::{DecodedFramedRow, DecodedPayloadReceipt};
#[cfg(feature = "spill")]
use super::vector::{ResidentCapacityError, ValueVector};
use grafeo_common::memory::buffer::MemoryGrant;
#[cfg(feature = "spill")]
use grafeo_common::memory::buffer::MemoryGrantError;
use grafeo_common::types::{LogicalType, Value};

/// Closed private transport tags. Ordinary rows have no trailer at all;
/// a typed row packs four columns per byte without changing public Values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SortValueType {
    Ordinary = 0,
    EdgeList = 1,
    Edge = 2,
    Node = 3,
}

impl SortValueType {
    pub(crate) fn from_logical(data_type: &LogicalType) -> Self {
        match data_type {
            LogicalType::List(item) if item.as_ref() == &LogicalType::Edge => Self::EdgeList,
            LogicalType::Edge => Self::Edge,
            LogicalType::Node => Self::Node,
            _ => Self::Ordinary,
        }
    }

    /// Ordinary output may materialize its schema normally. Exact accounted
    /// output instead passes this tag to the fallible vector constructor.
    pub(crate) fn logical_type(self) -> LogicalType {
        match self {
            Self::Ordinary => LogicalType::Any,
            Self::EdgeList => LogicalType::List(Box::new(LogicalType::Edge)),
            Self::Edge => LogicalType::Edge,
            Self::Node => LogicalType::Node,
        }
    }
}

/// Private sort-owner row contract. Generic spill callers remain strict;
/// only the graph sort owner may attach one validated provenance value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SortRowShape {
    pub(crate) logical_columns: usize,
    edge_trailer: bool,
}

impl SortRowShape {
    #[cfg(feature = "spill")]
    pub(crate) const fn strict(logical_columns: usize) -> Self {
        Self {
            logical_columns,
            edge_trailer: false,
        }
    }

    pub(crate) const fn with_edge_trailer(logical_columns: usize) -> Self {
        Self {
            logical_columns,
            edge_trailer: true,
        }
    }

    #[cfg(feature = "spill")]
    pub(crate) fn physical_columns(self, payload: &[u8]) -> Result<usize, &'static str> {
        let count = payload
            .get(..8)
            .ok_or("sort row is missing its column count")?;
        let count = usize::try_from(u64::from_le_bytes(
            count
                .try_into()
                .map_err(|_| "sort row column count is truncated")?,
        ))
        .map_err(|_| "sort row column count is not addressable")?;
        if count == self.logical_columns
            || (self.edge_trailer && self.logical_columns.checked_add(1) == Some(count))
        {
            Ok(count)
        } else {
            Err("sort row does not match its admitted logical width")
        }
    }

    pub(crate) fn edge_mask(self, values: &[Value]) -> Result<Option<&[u8]>, &'static str> {
        if values.len() == self.logical_columns {
            return Ok(None);
        }
        if !self.edge_trailer || self.logical_columns.checked_add(1) != Some(values.len()) {
            return Err("sort row does not match its admitted logical width");
        }
        let Some(Value::Bytes(mask)) = values.last() else {
            return Err("sort provenance trailer is not Bytes");
        };
        let mask_len = self.logical_columns.div_ceil(4);
        if mask.len() != mask_len || !mask.iter().any(|byte| *byte != 0) {
            return Err("sort provenance mask has invalid length or no typed columns");
        }
        let remainder = 2 * (self.logical_columns % 4);
        if remainder != 0 && mask.last().is_some_and(|byte| *byte >> remainder != 0) {
            return Err("sort provenance mask has nonzero padding");
        }
        Ok(Some(mask))
    }

    pub(crate) fn column_type(mask: Option<&[u8]>, column: usize) -> SortValueType {
        let tag = mask
            .and_then(|mask| mask.get(column / 4))
            .map_or(0, |byte| (byte >> (2 * (column % 4))) & 3);
        match tag {
            1 => SortValueType::EdgeList,
            2 => SortValueType::Edge,
            3 => SortValueType::Node,
            _ => SortValueType::Ordinary,
        }
    }
}

#[cfg(feature = "spill")]
pub(crate) trait AccountedRowGrantObserver {
    fn replace_row(&self, previous: usize, current: usize) -> Result<(), MemoryGrantError>;
}

#[cfg(feature = "spill")]
pub(crate) trait AccountedChunkGrantObserver {
    fn publish_retained(&self, bytes: usize) -> Result<(), MemoryGrantError>;
    fn publish_retained_preserving_primary(&self, bytes: usize);
    fn poison_unacknowledged_retained(&self);
}

#[cfg(all(test, feature = "spill"))]
std::thread_local! {
    static PANIC_DURING_SORT_CHUNK_CONSTRUCTION: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

/// Test-only arm whose drop prevents a failed test from contaminating a later
/// test scheduled on the same harness thread.
#[cfg(all(test, feature = "spill"))]
pub(crate) struct SortChunkConstructionPanicGuard;

#[cfg(all(test, feature = "spill"))]
impl Drop for SortChunkConstructionPanicGuard {
    fn drop(&mut self) {
        PANIC_DURING_SORT_CHUNK_CONSTRUCTION.with(|armed| armed.set(false));
    }
}

/// Arms one thread-local panic after a decoded row and its grant have entered
/// the output-construction owner.
#[cfg(all(test, feature = "spill"))]
pub(crate) fn inject_sort_chunk_construction_panic_once() -> SortChunkConstructionPanicGuard {
    PANIC_DURING_SORT_CHUNK_CONSTRUCTION.with(|armed| armed.set(true));
    SortChunkConstructionPanicGuard
}

#[cfg(all(test, feature = "spill"))]
fn maybe_panic_during_sort_chunk_construction() {
    PANIC_DURING_SORT_CHUNK_CONSTRUCTION.with(|armed| {
        assert!(
            !armed.replace(false),
            "deterministic accounted sort-chunk construction panic"
        );
    });
}

/// Decoder-minted sort row whose physical values and exact authority cannot
/// be separated or cloned.
#[cfg(feature = "spill")]
#[derive(Debug)]
#[must_use = "the decoded spill row must remain coupled to its memory grant"]
pub(crate) struct AccountedOrdinalRow {
    values: Option<Vec<Value>>,
    receipt: Option<DecodedPayloadReceipt>,
    ordinal: u64,
    grant: Option<MemoryGrant>,
}

#[cfg(feature = "spill")]
#[derive(Debug)]
pub(crate) enum AccountedSortRowError {
    Memory(MemoryGrantError),
    Arithmetic,
    InvalidAuthority,
}

/// Failed decoded-row construction together with the still-live authority
/// that covered every value and receipt until their destruction.
#[cfg(feature = "spill")]
#[derive(Debug)]
pub(crate) struct AccountedSortRowBuildError {
    primary: AccountedSortRowError,
    grant: MemoryGrant,
}

#[cfg(feature = "spill")]
impl AccountedSortRowBuildError {
    fn new(primary: AccountedSortRowError, grant: MemoryGrant) -> Self {
        Self { primary, grant }
    }

    pub(crate) fn into_parts(self) -> (AccountedSortRowError, MemoryGrant) {
        (self.primary, self.grant)
    }
}

#[cfg(feature = "spill")]
impl AccountedOrdinalRow {
    /// Consumes the decoder-minted row and shrinks its transferred grant to the
    /// exact top-level backing plus the unforgeable recursive-payload receipt.
    /// No sibling receives arbitrary live resize authority.
    pub(crate) fn try_from_decoded(
        decoded: DecodedFramedRow,
        ordinal: u64,
        mut grant: MemoryGrant,
        observer: Option<&dyn AccountedRowGrantObserver>,
    ) -> Result<Self, AccountedSortRowBuildError> {
        let (values, receipt) = decoded.into_parts();
        let direct_bytes = match std::alloc::Layout::array::<Value>(values.capacity()) {
            Ok(layout) => layout.size(),
            Err(_) => {
                return Err(AccountedSortRowBuildError::new(
                    AccountedSortRowError::Arithmetic,
                    grant,
                ));
            }
        };
        let Some(retained_bytes) = direct_bytes.checked_add(receipt.retained_bytes()) else {
            return Err(AccountedSortRowBuildError::new(
                AccountedSortRowError::Arithmetic,
                grant,
            ));
        };
        if grant.size() < retained_bytes {
            return Err(AccountedSortRowBuildError::new(
                AccountedSortRowError::InvalidAuthority,
                grant,
            ));
        }
        let previous = grant.size();
        if let Err(error) = grant.try_resize(retained_bytes) {
            return Err(AccountedSortRowBuildError::new(
                AccountedSortRowError::Memory(error),
                grant,
            ));
        }
        if let Some(Err(error)) =
            observer.map(|observer| observer.replace_row(previous, grant.size()))
        {
            return Err(AccountedSortRowBuildError::new(
                AccountedSortRowError::Memory(error),
                grant,
            ));
        }
        Ok(Self {
            values: Some(values),
            receipt: Some(receipt),
            ordinal,
            grant: Some(grant),
        })
    }

    pub(crate) fn values(&self) -> &[Value] {
        self.values
            .as_deref()
            .expect("live accounted row retains its physical values")
    }

    pub(crate) const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    pub(crate) fn granted_bytes(&self) -> usize {
        self.grant
            .as_ref()
            .expect("live accounted row retains its memory grant")
            .size()
    }

    #[cfg(test)]
    pub(crate) fn top_level_capacity_bytes(&self) -> usize {
        self.values
            .as_ref()
            .expect("live accounted row retains its physical values")
            .capacity()
            .checked_mul(std::mem::size_of::<Value>())
            .expect("accounted row top-level capacity was admitted")
    }

    fn into_chunk_parts(mut self) -> (Vec<Value>, DecodedPayloadReceipt, MemoryGrant) {
        let values = self
            .values
            .take()
            .expect("live accounted row retains its physical values");
        let receipt = self
            .receipt
            .take()
            .expect("live accounted row retains its decoder receipt");
        let grant = self
            .grant
            .take()
            .expect("live accounted row retains its memory grant");
        (values, receipt, grant)
    }

    /// Destroys the decoded row before returning its still-accounted grant.
    ///
    /// Exact terminal paths use this consuming seam instead of `Drop` when a
    /// release result must remain observable and retryable. The physical
    /// top-level values and recursive-payload receipt cease before authority
    /// can move back into a sorter-owned frontier.
    #[must_use = "the returned grant remains live release or transfer authority"]
    pub(crate) fn into_released_grant(mut self) -> MemoryGrant {
        drop(self.values.take());
        let _ = self.receipt.take();
        self.grant
            .take()
            .expect("live accounted row transfers its memory grant exactly once")
    }
}

#[cfg(feature = "spill")]
impl Drop for AccountedOrdinalRow {
    fn drop(&mut self) {
        drop(self.values.take());
        let _ = self.receipt.take();
        let Some(mut grant) = self.grant.take() else {
            return;
        };
        if grant.try_resize(0).is_err() {
            // A destructor cannot surface the release error or return its
            // retry token. Keep the charge fail-closed instead of delegating
            // to MemoryGrant's unobservable best-effort Drop path.
            std::mem::forget(grant);
        }
    }
}

/// A data chunk coupled to the unique authority accounting for every resident
/// allocation whose lifetime was transferred with it.
///
/// The envelope is intentionally move-only. Qualified execution can retain or
/// forward it without releasing the charge while the chunk remains resident,
/// but cannot duplicate the accounting capability through [`Clone`].
///
/// The public API exposes only a shared chunk reference and the charged byte
/// count. It has no owned-inner or mutable access that could detach the chunk
/// from its grant or grow storage behind the accounting authority.
///
/// Calling [`DataChunk::clone`] through [`Self::chunk`] creates a separate
/// compatibility allocation which is not owned by this envelope. Qualified
/// transport must remain move-only unless a future fallible accounted-copy
/// API admits a distinct grant for that allocation.
///
/// Construction is sealed inside this module. The qualified sort-output
/// builder mints the private admission only after validating the chunk's
/// shape and transferring authority for direct columns, selections, zone
/// hints, and shared `Arc` payloads as applicable. A direct-column capacity
/// measurement alone is not proof that an arbitrary [`DataChunk`] is admitted.
/// Successful sink-owned chunks retain the established RAII contract: field
/// order destroys the physical chunk before ordinary [`MemoryGrant`] Drop.
/// Callers that require a fallible, retryable terminal release must keep that
/// responsibility in an upstream move-only owner; the richer construction
/// error below exists specifically so failed construction never depends on
/// the successful chunk's best-effort Drop contract.
///
/// ```compile_fail
/// use grafeo_core::execution::AccountedDataChunk;
///
/// fn duplicate(chunk: &AccountedDataChunk) -> AccountedDataChunk {
///     chunk.clone()
/// }
/// ```
///
/// ```compile_fail
/// use grafeo_common::memory::buffer::MemoryGrant;
/// use grafeo_core::execution::{AccountedDataChunk, DataChunk};
///
/// fn forge(chunk: DataChunk, grant: MemoryGrant) -> AccountedDataChunk {
///     AccountedDataChunk::new(chunk, grant)
/// }
/// ```
#[derive(Debug)]
#[must_use = "dropping an accounted chunk releases its resident-memory grant"]
pub struct AccountedDataChunk {
    admission: AccountedChunkAdmission,
}

/// Module-private proof that the chunk's complete retained allocation shape
/// and every transferred payload authority have been admitted.
///
/// Fields stay private so sibling execution modules cannot substitute an
/// arbitrary chunk/grant pair.
#[derive(Debug)]
struct AccountedChunkAdmission {
    /// Declared first so all physical chunk storage drops before its authority.
    chunk: DataChunk,
    /// Unique authority for all resident lifetimes transferred with `chunk`.
    grant: MemoryGrant,
}

impl AccountedDataChunk {
    /// Consumes the module-private proof without reopening its invariants.
    fn from_admission(admission: AccountedChunkAdmission) -> Self {
        Self { admission }
    }

    /// Returns shared, non-growing access to the accounted chunk.
    #[must_use]
    pub const fn chunk(&self) -> &DataChunk {
        &self.admission.chunk
    }

    /// Returns the number of resident bytes owned by this envelope's grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.admission.grant.size()
    }
}

/// Validates the complete retained shape of an already pre-admitted DISTINCT
/// output and transfers its authority into the sealed, move-only transport.
/// The state admits capacity before construction; this final check prevents an
/// incorrect estimate or a selected/hinted chunk from minting transport proof.
pub(crate) fn try_accounted_distinct_output(
    chunk: DataChunk,
    grant: MemoryGrant,
) -> Result<AccountedDataChunk, super::operators::OperatorError> {
    use super::operators::OperatorError;
    // Storage is declared before authority, including on a validation failure.
    let admission = AccountedChunkAdmission { chunk, grant };
    let chunk = &admission.chunk;
    let invalid = |message| OperatorError::ResidentContainerInvariant {
        container: "DISTINCT accounted output",
        message,
    };
    if chunk.selection().is_some() || chunk.zone_hints().is_some() {
        return Err(invalid("output must have no selection or zone hints"));
    }
    let mut bytes = chunk
        .observed_column_capacity_bytes()
        .map_err(|_| invalid("direct capacity overflow"))?;
    for column in chunk.columns() {
        let schema_heap = distinct_schema_heap_bytes(column.data_type(), 0)
            .ok_or_else(|| invalid("schema capacity overflow or excessive nesting"))?;
        // Vector's direct-capacity measurement already includes this specific
        // provenance box; all other nested schema allocations are added here.
        let already_measured = if matches!(column.data_type(), LogicalType::List(inner) if **inner == LogicalType::Edge)
        {
            std::mem::size_of::<LogicalType>()
        } else {
            0
        };
        bytes = bytes
            .checked_add(
                schema_heap
                    .checked_sub(already_measured)
                    .ok_or_else(|| invalid("schema capacity mismatch"))?,
            )
            .ok_or_else(|| invalid("schema capacity overflow"))?;
        for row in 0..column.len() {
            let backing = column
                .retained_value_bytes(row)
                .and_then(|value| value.checked_sub(std::mem::size_of::<Value>()))
                .ok_or_else(|| invalid("nested payload capacity overflow"))?;
            bytes = bytes
                .checked_add(backing)
                .ok_or_else(|| invalid("payload capacity overflow"))?;
        }
    }
    if bytes > admission.grant.size() {
        return Err(invalid("output exceeds its pre-admitted reservation"));
    }
    Ok(AccountedDataChunk::from_admission(admission))
}

fn distinct_schema_heap_bytes(ty: &LogicalType, depth: usize) -> Option<usize> {
    if depth > 128 {
        return None;
    }
    let boxed = |inner: &LogicalType| {
        std::mem::size_of::<LogicalType>()
            .checked_add(distinct_schema_heap_bytes(inner, depth + 1)?)
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
                    .checked_add(distinct_schema_heap_bytes(inner, depth + 1)?)?;
            }
            Some(bytes)
        }
        _ => Some(0),
    }
}

/// Typed reason why an exact decoded row could not become a columnar chunk.
#[cfg(feature = "spill")]
#[derive(Debug)]
pub(crate) enum AccountedSortChunkPrimary {
    Memory(MemoryGrantError),
    Capacity(ResidentCapacityError),
    Arithmetic(&'static str),
    InvalidShape(&'static str),
    CapacityContract,
}

#[cfg(feature = "spill")]
impl std::fmt::Display for AccountedSortChunkPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::Capacity(error) => std::fmt::Display::fmt(error, formatter),
            Self::Arithmetic(context) => write!(formatter, "{context} exceeds addressable memory"),
            Self::InvalidShape(message) => formatter.write_str(message),
            Self::CapacityContract => formatter.write_str(
                "accounted sort output exceeded its pre-admitted direct-capacity envelope",
            ),
        }
    }
}

#[cfg(feature = "spill")]
impl std::error::Error for AccountedSortChunkPrimary {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Memory(error) => Some(error),
            Self::Capacity(error) => Some(error),
            Self::Arithmetic(_) | Self::InvalidShape(_) | Self::CapacityContract => None,
        }
    }
}

#[cfg(feature = "spill")]
impl From<MemoryGrantError> for AccountedSortChunkPrimary {
    fn from(error: MemoryGrantError) -> Self {
        Self::Memory(error)
    }
}

#[cfg(feature = "spill")]
impl From<ResidentCapacityError> for AccountedSortChunkPrimary {
    fn from(error: ResidentCapacityError) -> Self {
        Self::Capacity(error)
    }
}

/// Failed exact row-to-chunk construction with its sole release capability.
///
/// Every physical chunk, column, source row, and decoder receipt is destroyed
/// before this value becomes visible. The remaining grant is deliberately
/// non-detachable except through [`Self::into_parts`], so a caller can merge
/// it into a terminal frontier or explicitly reconcile it without relying on
/// ordinary [`MemoryGrant`] Drop's unobservable best-effort release.
#[cfg(feature = "spill")]
#[derive(Debug)]
#[must_use = "the failed construction retains live memory-release authority"]
pub(crate) struct AccountedSortChunkError {
    primary: Option<AccountedSortChunkPrimary>,
    grant: Option<MemoryGrant>,
}

#[cfg(feature = "spill")]
impl AccountedSortChunkError {
    fn new(primary: AccountedSortChunkPrimary, grant: MemoryGrant) -> Self {
        Self {
            primary: Some(primary),
            grant: Some(grant),
        }
    }

    fn primary(&self) -> &AccountedSortChunkPrimary {
        self.primary
            .as_ref()
            .expect("live accounted chunk failure retains its primary")
    }

    /// Recovers both the typed diagnostic and the complete retry authority.
    #[must_use = "the returned grant must be explicitly released or transferred"]
    pub(crate) fn into_parts(mut self) -> (AccountedSortChunkPrimary, MemoryGrant) {
        (
            self.primary
                .take()
                .expect("accounted chunk failure transfers its primary exactly once"),
            self.grant
                .take()
                .expect("accounted chunk failure transfers its grant exactly once"),
        )
    }

    #[cfg(test)]
    fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }
}

#[cfg(feature = "spill")]
impl std::fmt::Display for AccountedSortChunkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.primary(), formatter)
    }
}

#[cfg(feature = "spill")]
impl std::error::Error for AccountedSortChunkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary())
    }
}

#[cfg(feature = "spill")]
impl Drop for AccountedSortChunkError {
    fn drop(&mut self) {
        let _ = self.primary.take();
        let Some(mut grant) = self.grant.take() else {
            return;
        };
        if grant.try_resize(0).is_err() {
            // `Drop` cannot return the only retry token. Retain it fail-closed
            // instead of letting raw grant Drop hide a second release failure.
            std::mem::forget(grant);
        }
    }
}

/// Construction owner whose physical allocations always precede the grant.
#[cfg(feature = "spill")]
struct AccountedSortChunkConstruction<'a, 'observer> {
    chunk: Option<DataChunk>,
    columns: Option<Vec<ValueVector>>,
    values: Option<Vec<grafeo_common::types::Value>>,
    tail: Option<Value>,
    receipt: Option<DecodedPayloadReceipt>,
    grant: Option<MemoryGrant>,
    observer: &'a (dyn AccountedChunkGrantObserver + 'observer),
}

#[cfg(feature = "spill")]
impl AccountedSortChunkConstruction<'_, '_> {
    fn destroy_physical_storage(&mut self) {
        drop(self.chunk.take());
        drop(self.columns.take());
        drop(self.values.take());
        drop(self.tail.take());
        let _ = self.receipt.take();
    }

    fn into_error(mut self, primary: AccountedSortChunkPrimary) -> AccountedSortChunkError {
        self.destroy_physical_storage();
        let grant = self
            .grant
            .take()
            .expect("failed accounted chunk construction retains its grant");
        let retained = grant.size();
        let error = AccountedSortChunkError::new(primary, grant);
        // The grant remains charged after every physical allocation has gone.
        // Publish that conservative retained owner until the exact cursor
        // explicitly merges or releases the returned capability.
        self.observer.publish_retained_preserving_primary(retained);
        error
    }
}

#[cfg(feature = "spill")]
impl Drop for AccountedSortChunkConstruction<'_, '_> {
    fn drop(&mut self) {
        self.destroy_physical_storage();
        let Some(mut grant) = self.grant.take() else {
            return;
        };
        if std::thread::panicking() {
            // Do not invoke another fallible accounting transition during an
            // existing unwind. The physical storage is gone, but authority
            // has no remaining retry recipient. Latch that loss of authority
            // before forgetting the grant so later cursor cleanup cannot
            // overwrite the conservative signal with a false zero.
            std::mem::forget(grant);
            self.observer.poison_unacknowledged_retained();
            return;
        }
        let release =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| grant.try_resize(0)));
        match release {
            Ok(Ok(())) => self.observer.publish_retained_preserving_primary(0),
            Ok(Err(_)) => {
                std::mem::forget(grant);
                self.observer.poison_unacknowledged_retained();
            }
            Err(payload) => {
                std::mem::forget(grant);
                self.observer.poison_unacknowledged_retained();
                std::panic::resume_unwind(payload);
            }
        }
    }
}

#[cfg(feature = "spill")]
fn checked_add(
    current: usize,
    additional: usize,
    context: &'static str,
) -> Result<usize, AccountedSortChunkPrimary> {
    current
        .checked_add(additional)
        .ok_or(AccountedSortChunkPrimary::Arithmetic(context))
}

/// Moves one decoder-minted row into a one-row accounted output chunk.
///
/// The deliberately narrow one-row shape makes every direct allocation
/// precomputable: an exact outer column vector and one exact Generic `Value`
/// slot per column. Null is embedded directly in that Generic slot, so this
/// lane never allocates a validity bitmap. The decoder-witnessed recursive
/// payload bytes transfer after removing the private provenance Bytes receipt.
/// Source row backing and the mask are destroyed before
/// the child grant shrinks to the final direct column capacity plus that
/// payload receipt.
#[cfg(feature = "spill")]
pub(crate) fn try_accounted_chunk_from_sort_row(
    row: AccountedOrdinalRow,
    num_columns: usize,
    observer: &dyn AccountedChunkGrantObserver,
) -> Result<AccountedDataChunk, AccountedSortChunkError> {
    let (values, receipt, grant) = row.into_chunk_parts();
    let mut owner = AccountedSortChunkConstruction {
        chunk: None,
        columns: None,
        values: Some(values),
        tail: None,
        receipt: Some(receipt),
        grant: Some(grant),
        observer,
    };
    #[cfg(test)]
    maybe_panic_during_sort_chunk_construction();
    macro_rules! fail {
        ($primary:expr) => {
            return Err(owner.into_error($primary))
        };
    }
    let source_values = owner
        .values
        .as_ref()
        .expect("construction retains source values");
    if num_columns == 0
        || (source_values.len() != num_columns
            && num_columns.checked_add(1) != Some(source_values.len()))
    {
        fail!(AccountedSortChunkPrimary::InvalidShape(
            "decoded sort row does not match its non-empty output schema",
        ));
    }
    let mask = match SortRowShape::with_edge_trailer(num_columns).edge_mask(source_values) {
        Ok(mask) => mask,
        Err(message) => fail!(AccountedSortChunkPrimary::InvalidShape(message)),
    };
    let mask_bytes = mask.map(<[u8]>::len);
    // Scalar tags need no heap schema. Only each List(Edge) owns a box.
    let typed_columns = (0..num_columns)
        .filter(|&column| SortRowShape::column_type(mask, column) == SortValueType::EdgeList)
        .count();
    let source_direct =
        match std::alloc::Layout::array::<grafeo_common::types::Value>(source_values.capacity()) {
            Ok(layout) => layout.size(),
            Err(_) => fail!(AccountedSortChunkPrimary::Arithmetic(
                "decoded sort-row backing"
            )),
        };
    let payload_bytes = owner
        .receipt
        .as_ref()
        .expect("construction retains decoder receipt")
        .retained_bytes();
    let expected_source =
        match checked_add(payload_bytes, source_direct, "decoded sort-row authority") {
            Ok(bytes) => bytes,
            Err(primary) => fail!(primary),
        };
    if owner
        .grant
        .as_ref()
        .expect("construction retains source authority")
        .size()
        != expected_source
    {
        fail!(AccountedSortChunkPrimary::InvalidShape(
            "decoded sort-row authority does not equal its witnessed retained shape",
        ));
    }

    let outer_bytes = match std::alloc::Layout::array::<ValueVector>(num_columns) {
        Ok(layout) => layout.size(),
        Err(_) => fail!(AccountedSortChunkPrimary::Arithmetic(
            "sort output column catalog"
        )),
    };
    let value_bytes = match std::alloc::Layout::array::<grafeo_common::types::Value>(num_columns) {
        Ok(layout) => layout.size(),
        Err(_) => fail!(AccountedSortChunkPrimary::Arithmetic("sort output values")),
    };
    // The sealed exact Generic lane embeds `Value::Null`; it never allocates
    // the compatibility validity bitmap.
    let Some(schema_bytes) =
        typed_columns.checked_mul(std::mem::size_of::<grafeo_common::types::LogicalType>())
    else {
        fail!(AccountedSortChunkPrimary::Arithmetic(
            "sort output schema boxes"
        ));
    };
    let direct_envelope = match checked_add(outer_bytes, value_bytes, "sort output direct capacity")
        .and_then(|bytes| checked_add(bytes, schema_bytes, "sort output schema boxes"))
    {
        Ok(bytes) => bytes,
        Err(primary) => fail!(primary),
    };

    let construction_bytes = match checked_add(
        owner
            .grant
            .as_ref()
            .expect("construction retains grant")
            .size(),
        direct_envelope,
        "sort output construction peak",
    ) {
        Ok(bytes) => bytes,
        Err(primary) => fail!(primary),
    };
    // The construction peak is exact. Publish it before admission can invoke
    // eviction machinery, keeping telemetry conservative across the grant
    // transition itself.
    if let Err(error) = observer.publish_retained(construction_bytes) {
        fail!(AccountedSortChunkPrimary::Memory(error));
    }
    if let Err(error) = owner
        .grant
        .as_mut()
        .expect("construction retains grant")
        .try_resize(construction_bytes)
    {
        observer.publish_retained_preserving_primary(
            owner
                .grant
                .as_ref()
                .expect("failed construction retains grant")
                .size(),
        );
        fail!(AccountedSortChunkPrimary::Memory(error));
    }

    let columns = match crate::execution::vector::try_exact_value_vector_catalog(num_columns) {
        Ok(columns) => columns,
        Err(error) => fail!(AccountedSortChunkPrimary::Capacity(error)),
    };
    owner.columns = Some(columns);
    if mask_bytes.is_some() {
        let Some(values) = owner.values.as_mut() else {
            fail!(AccountedSortChunkPrimary::InvalidShape(
                "construction lost its source values"
            ));
        };
        owner.tail = values.pop();
    }
    let mut values = owner
        .values
        .take()
        .expect("construction consumes source values exactly once")
        .into_iter();
    let mut column_index = 0;
    while let Some(value) = values.next() {
        let mask = match owner.tail.as_ref() {
            Some(Value::Bytes(mask)) => Some(mask.as_ref()),
            _ => None,
        };
        let value_type = SortRowShape::column_type(mask, column_index);
        let column = match ValueVector::try_exact_generic_one_with_sort_type(value, value_type) {
            Ok(column) => column,
            Err(error) => {
                // Destroy the still-owned source allocation and every
                // unconsumed recursive payload before authority can escape.
                drop(values);
                fail!(AccountedSortChunkPrimary::Capacity(error));
            }
        };
        owner
            .columns
            .as_mut()
            .expect("construction retains its exact column catalog")
            .push(column);
        column_index += 1;
    }
    // `IntoIter` retains the source Vec allocation after yielding its final
    // value. End that physical lifetime before the grant can shrink from the
    // exact construction peak to the output-only retained shape.
    drop(values);
    // Keep the mask in the construction owner until every schema box exists.
    // Its physical lifetime ends before subtracting its decoder contribution.
    drop(owner.tail.take());
    let Some(receipt) = owner.receipt.as_mut() else {
        fail!(AccountedSortChunkPrimary::InvalidShape(
            "construction lost its decoder receipt"
        ));
    };
    if let Some(bytes) = mask_bytes
        && let Err(error) = receipt.release_sort_bytes_tail(bytes)
    {
        fail!(AccountedSortChunkPrimary::Memory(error));
    }
    let payload_bytes = receipt.retained_bytes();
    owner.chunk = Some(DataChunk::new(
        owner
            .columns
            .take()
            .expect("constructed columns transfer exactly once"),
    ));
    let direct_bytes = owner
        .chunk
        .as_ref()
        .expect("constructed chunk remains owned")
        .observed_column_capacity_bytes();
    let direct_bytes = match direct_bytes {
        Ok(bytes) => bytes,
        Err(error) => fail!(AccountedSortChunkPrimary::Capacity(error)),
    };
    if direct_bytes > direct_envelope {
        fail!(AccountedSortChunkPrimary::CapacityContract);
    }
    let retained_bytes =
        match checked_add(payload_bytes, direct_bytes, "sort output retained bytes") {
            Ok(bytes) => bytes,
            Err(primary) => fail!(primary),
        };
    let resize = owner
        .grant
        .as_mut()
        .expect("constructed chunk retains grant")
        .try_resize(retained_bytes);
    let observation = observer.publish_retained(
        owner
            .grant
            .as_ref()
            .expect("constructed chunk retains grant")
            .size(),
    );
    if let Err(error) = resize {
        fail!(AccountedSortChunkPrimary::Memory(error));
    }
    if let Err(error) = observation {
        fail!(AccountedSortChunkPrimary::Memory(error));
    }

    let chunk = owner
        .chunk
        .take()
        .expect("accounted chunk transfers exactly once");
    let grant = owner
        .grant
        .take()
        .expect("accounted chunk authority transfers exactly once");
    let _ = owner.receipt.take();
    Ok(AccountedDataChunk::from_admission(
        AccountedChunkAdmission { chunk, grant },
    ))
}

/// Test-only admission bypass for exercising ownership and drop semantics.
///
/// This is deliberately not a capacity calculator or a production proof. The
/// qualified sort-output builder must validate the complete retained shape and
/// transfer every payload authority before it gains a production minter.
#[cfg(test)]
pub(crate) fn accounted_chunk_for_test(chunk: DataChunk, grant: MemoryGrant) -> AccountedDataChunk {
    AccountedDataChunk::from_admission(AccountedChunkAdmission { chunk, grant })
}

#[cfg(test)]
mod tests {
    #[test]
    fn distinct_output_proof_rejects_uncovered_nested_payload_and_selection() {
        use super::try_accounted_distinct_output;
        use crate::execution::selection::SelectionVector;
        use crate::execution::{DataChunk, QueryResourceContext, ValueVector};
        use grafeo_common::memory::buffer::BufferManager;
        use grafeo_common::types::Value;
        let resources = QueryResourceContext::new(BufferManager::with_budget(1024 * 1024)).unwrap();
        let nested = Value::List(vec![Value::from("x".repeat(8192))].into());
        let chunk = DataChunk::new(vec![ValueVector::from_values(&[nested])]);
        assert!(
            try_accounted_distinct_output(chunk, resources.try_allocate(256).unwrap()).is_err()
        );
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        let mut selected = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(1)])]);
        selected.set_selection(SelectionVector::new_all(1));
        assert!(
            try_accounted_distinct_output(selected, resources.try_allocate(4096).unwrap()).is_err()
        );
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    use super::accounted_chunk_for_test;
    #[cfg(feature = "spill")]
    use super::{
        AccountedChunkGrantObserver, AccountedOrdinalRow, AccountedRowGrantObserver,
        AccountedSortChunkPrimary, try_accounted_chunk_from_sort_row,
    };
    #[cfg(feature = "spill")]
    use crate::execution::value_codec::{
        CodecLimits, deserialize_framed_row_exact_with_receipt, serialize_row_with_limits,
    };
    use crate::execution::{DataChunk, QueryResourceContext, ValueVector};
    use grafeo_common::memory::buffer::BufferManager;
    #[cfg(feature = "spill")]
    use grafeo_common::memory::buffer::MemoryGrantError;
    use grafeo_common::types::{LogicalType, Value};
    #[cfg(feature = "spill")]
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    #[cfg(feature = "spill")]
    struct NoopGrantObserver;

    #[cfg(feature = "spill")]
    impl AccountedRowGrantObserver for NoopGrantObserver {
        fn replace_row(&self, _previous: usize, _current: usize) -> Result<(), MemoryGrantError> {
            Ok(())
        }
    }

    #[cfg(feature = "spill")]
    impl AccountedChunkGrantObserver for NoopGrantObserver {
        fn publish_retained(&self, _bytes: usize) -> Result<(), MemoryGrantError> {
            Ok(())
        }

        fn publish_retained_preserving_primary(&self, _bytes: usize) {}

        fn poison_unacknowledged_retained(&self) {}
    }

    #[cfg(feature = "spill")]
    #[derive(Default)]
    struct TrackingGrantObserver {
        retained: Cell<usize>,
    }

    #[cfg(feature = "spill")]
    impl AccountedChunkGrantObserver for TrackingGrantObserver {
        fn publish_retained(&self, bytes: usize) -> Result<(), MemoryGrantError> {
            self.retained.set(bytes);
            Ok(())
        }

        fn publish_retained_preserving_primary(&self, bytes: usize) {
            self.retained.set(bytes);
        }

        fn poison_unacknowledged_retained(&self) {
            self.retained.set(usize::MAX);
        }
    }

    fn one_value_chunk() -> DataChunk {
        let mut column = ValueVector::try_with_capacity(LogicalType::Int64, 1).unwrap();
        column.try_push_value(Value::Int64(42)).unwrap();
        DataChunk::new(vec![column])
    }

    #[test]
    fn retained_chunk_keeps_query_and_global_bytes_charged_until_drop() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let chunk = one_value_chunk();
        let direct_column_bytes = chunk.observed_column_capacity_bytes().unwrap();
        // Stand in for transferred selection/zone/shared-payload authority:
        // admission is explicitly broader than direct column measurement.
        let granted_bytes = direct_column_bytes + 37;
        let grant = context.try_allocate(granted_bytes).unwrap();
        let retained = Some(accounted_chunk_for_test(chunk, grant));

        let accounted = retained.as_ref().unwrap();
        assert_eq!(accounted.granted_bytes(), granted_bytes);
        assert_eq!(accounted.chunk().len(), 1);
        assert_eq!(context.query_stats().allocated_bytes, granted_bytes);
        assert_eq!(manager.allocated(), granted_bytes);

        drop(context);
        assert_eq!(retained.as_ref().unwrap().granted_bytes(), granted_bytes);
        assert_eq!(manager.allocated(), granted_bytes);
        drop(retained);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn unwind_drops_chunk_envelope_and_releases_query_and_global_bytes() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let chunk = one_value_chunk();
        let granted_bytes = chunk.observed_column_capacity_bytes().unwrap();
        let unwind_context = context.clone();

        let result = catch_unwind(AssertUnwindSafe(move || {
            let grant = unwind_context.try_allocate(granted_bytes).unwrap();
            let retained = accounted_chunk_for_test(chunk, grant);
            assert_eq!(unwind_context.query_stats().allocated_bytes, granted_bytes);
            assert_eq!(manager.allocated(), granted_bytes);
            let _keep_alive_until_unwind = retained;
            panic!("exercise accounted chunk unwind cleanup");
        }));

        assert!(result.is_err());
        assert_eq!(context.query_stats().allocated_bytes, 0);
        assert_eq!(context.buffer_stats().total_allocated, 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn decoder_receipt_and_recursive_arc_move_into_chunk_under_one_exact_grant() {
        let manager = Arc::new(BufferManager::with_budget(2 * 1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let source = vec![Value::List(Arc::from([
            Value::from("nested"),
            Value::Int64(7),
        ]))];
        let mut encoded = Vec::new();
        serialize_row_with_limits(&source, &mut encoded, CodecLimits::format_max()).unwrap();
        let decoded =
            deserialize_framed_row_exact_with_receipt(&encoded, 1, CodecLimits::format_max())
                .unwrap();
        let construction_grant = context.try_allocate(1024 * 1024).unwrap();
        let row = AccountedOrdinalRow::try_from_decoded(
            decoded,
            11,
            construction_grant,
            Some(&NoopGrantObserver),
        )
        .unwrap();
        let Value::List(row_list) = &row.values()[0] else {
            panic!("decoded nested value changed shape")
        };
        let decoded_list_ptr = Arc::as_ptr(row_list);
        let decoded_list_weak = Arc::downgrade(row_list);
        assert_eq!(Arc::strong_count(row_list), 1);
        let row_bytes = row.granted_bytes();
        assert_eq!(manager.allocated(), row_bytes);

        let chunk = try_accounted_chunk_from_sort_row(row, 1, &NoopGrantObserver).unwrap();
        assert!(chunk.granted_bytes() > 0);
        assert_eq!(
            decoded_list_weak.strong_count(),
            1,
            "conversion moves the sole recursive Arc owner without cloning it"
        );
        let moved = chunk.chunk().column(0).unwrap().get_value(0).unwrap();
        let Value::List(chunk_list) = moved else {
            panic!("accounted chunk nested value changed shape")
        };
        assert_eq!(Arc::as_ptr(&chunk_list), decoded_list_ptr);
        assert_eq!(
            Arc::strong_count(&chunk_list),
            2,
            "the chunk plus this single retrieval are the only live Arc owners"
        );
        assert_eq!(manager.allocated(), chunk.granted_bytes());

        drop(chunk_list);
        let moved_again = chunk.chunk().column(0).unwrap().get_value(0).unwrap();
        let Value::List(chunk_list_again) = moved_again else {
            panic!("accounted chunk nested value changed shape")
        };
        assert_eq!(Arc::as_ptr(&chunk_list_again), decoded_list_ptr);
        assert_eq!(Arc::strong_count(&chunk_list_again), 2);
        drop(chunk_list_again);
        drop(chunk);
        assert_eq!(decoded_list_weak.strong_count(), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn accounted_row_releases_physical_payload_before_returning_live_grant() {
        let manager = Arc::new(BufferManager::with_budget(2 * 1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let source = vec![Value::List(Arc::from([
            Value::from("released"),
            Value::Int64(17),
        ]))];
        let mut encoded = Vec::new();
        serialize_row_with_limits(&source, &mut encoded, CodecLimits::format_max()).unwrap();
        let decoded =
            deserialize_framed_row_exact_with_receipt(&encoded, 1, CodecLimits::format_max())
                .unwrap();
        let row = AccountedOrdinalRow::try_from_decoded(
            decoded,
            29,
            context.try_allocate(1024 * 1024).unwrap(),
            Some(&NoopGrantObserver),
        )
        .unwrap();
        let Value::List(list) = &row.values()[0] else {
            panic!("decoded nested value changed shape")
        };
        let weak = Arc::downgrade(list);
        let row_bytes = row.granted_bytes();
        drop(source);

        let mut grant = row.into_released_grant();

        assert_eq!(weak.strong_count(), 0);
        assert_eq!(grant.size(), row_bytes);
        assert_eq!(manager.allocated(), row_bytes);
        grant.try_resize(0).unwrap();
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(all(feature = "spill", feature = "lpg"))]
    fn scalar_provenance_accounted_mixed_tags_charge_schema_and_cleanup() {
        for fail_schema in [false, true] {
            let manager = BufferManager::with_budget(2 * 1024 * 1024);
            let context = QueryResourceContext::new(manager.clone()).unwrap();
            // Edge, Node, ordinary, ordinary | List(Edge): second byte matters.
            let source = [
                Value::Int64(7),
                Value::Int64(7),
                Value::Null,
                Value::Bytes(vec![0xff].into()),
                Value::List(vec![Value::Int64(7)].into()),
                Value::Bytes(vec![0b0000_1110, 0b0000_0001].into()),
            ];
            let mut encoded = Vec::new();
            serialize_row_with_limits(&source, &mut encoded, CodecLimits::format_max()).unwrap();
            let decoded =
                deserialize_framed_row_exact_with_receipt(&encoded, 6, CodecLimits::format_max())
                    .unwrap();
            let row = AccountedOrdinalRow::try_from_decoded(
                decoded,
                0,
                context.try_allocate(1024 * 1024).unwrap(),
                Some(&NoopGrantObserver),
            )
            .unwrap();
            let Value::Bytes(mask) = &row.values()[5] else {
                panic!("missing mask")
            };
            let mask_weak = Arc::downgrade(mask);
            let Value::List(list) = &row.values()[4] else {
                panic!("missing list")
            };
            let list_weak = Arc::downgrade(list);
            let observer = TrackingGrantObserver::default();
            if fail_schema {
                let schema = std::alloc::Layout::new::<LogicalType>();
                let skip =
                    usize::from(std::alloc::Layout::array::<ValueVector>(5).unwrap() == schema)
                        + 5 * usize::from(std::alloc::Layout::new::<Value>() == schema);
                let (result, fired) = crate::allocation_test::with_failure(
                    schema.size(),
                    schema.align(),
                    skip,
                    || try_accounted_chunk_from_sort_row(row, 5, &observer),
                );
                assert!(
                    fired,
                    "must reach fallible List(Edge) schema after scalar columns"
                );
                let error = result.unwrap_err();
                assert!(matches!(
                    error.primary(),
                    AccountedSortChunkPrimary::Capacity(_)
                ));
                assert_eq!(mask_weak.strong_count(), 0);
                assert_eq!(list_weak.strong_count(), 0);
                assert_eq!(manager.allocated(), error.granted_bytes());
                let (_, mut grant) = error.into_parts();
                grant.try_resize(0).unwrap();
                observer.publish_retained(0).unwrap();
                assert_eq!(manager.allocated(), 0);
                let decoded = deserialize_framed_row_exact_with_receipt(
                    &encoded,
                    6,
                    CodecLimits::format_max(),
                )
                .unwrap();
                let row = AccountedOrdinalRow::try_from_decoded(
                    decoded,
                    0,
                    context.try_allocate(1024 * 1024).unwrap(),
                    Some(&NoopGrantObserver),
                )
                .unwrap();
                let retry = try_accounted_chunk_from_sort_row(row, 5, &observer).unwrap();
                assert_eq!(retry.chunk().column(0).unwrap().get_node_id(0), None);
                drop(retry);
            } else {
                let chunk = try_accounted_chunk_from_sort_row(row, 5, &observer).unwrap();
                let expected = [
                    LogicalType::Edge,
                    LogicalType::Node,
                    LogicalType::Any,
                    LogicalType::Any,
                    LogicalType::List(Box::new(LogicalType::Edge)),
                ];
                assert_eq!(chunk.chunk().column_count(), 5);
                for (index, expected) in expected.iter().enumerate() {
                    assert_eq!(chunk.chunk().column(index).unwrap().data_type(), expected);
                }
                assert_eq!(
                    chunk.chunk().column(0).unwrap().get_edge_id(0),
                    Some(grafeo_common::types::EdgeId(7))
                );
                assert_eq!(chunk.chunk().column(0).unwrap().get_node_id(0), None);
                assert_eq!(
                    chunk.chunk().column(1).unwrap().get_node_id(0),
                    Some(grafeo_common::types::NodeId(7))
                );
                assert_eq!(chunk.chunk().column(1).unwrap().get_edge_id(0), None);
                assert_eq!(mask_weak.strong_count(), 0);
                assert_eq!(
                    chunk.chunk().observed_column_capacity_bytes().unwrap(),
                    5 * (std::mem::size_of::<ValueVector>() + std::mem::size_of::<Value>())
                        + std::mem::size_of::<LogicalType>()
                );
                assert_eq!(manager.allocated(), chunk.granted_bytes());
                drop(chunk);
                assert_eq!(list_weak.strong_count(), 0);
            }
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn sort_provenance_mask_is_destroyed_before_its_receipt_is_released() {
        let manager = BufferManager::with_budget(2 * 1024 * 1024);
        let context = QueryResourceContext::new(manager.clone()).unwrap();
        let source = [
            Value::List(vec![Value::Int64(7)].into()),
            Value::Bytes(vec![1u8].into()),
        ];
        let mut encoded = Vec::new();
        serialize_row_with_limits(&source, &mut encoded, CodecLimits::format_max()).unwrap();
        let decoded =
            deserialize_framed_row_exact_with_receipt(&encoded, 2, CodecLimits::format_max())
                .unwrap();
        let (_, receipt) =
            deserialize_framed_row_exact_with_receipt(&encoded, 2, CodecLimits::format_max())
                .unwrap()
                .into_parts();
        let payload_before = receipt.retained_bytes();
        let row = AccountedOrdinalRow::try_from_decoded(
            decoded,
            0,
            context.try_allocate(1024 * 1024).unwrap(),
            Some(&NoopGrantObserver),
        )
        .unwrap();
        let Value::Bytes(mask) = &row.values()[1] else {
            panic!("expected private mask")
        };
        let weak = Arc::downgrade(mask);
        let chunk = try_accounted_chunk_from_sort_row(row, 1, &NoopGrantObserver).unwrap();
        assert_eq!(weak.strong_count(), 0);
        assert_eq!(chunk.chunk().column_count(), 1);
        assert_eq!(
            chunk.chunk().column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        let direct = chunk.chunk().observed_column_capacity_bytes().unwrap();
        assert_eq!(
            direct,
            std::mem::size_of::<ValueVector>()
                + std::mem::size_of::<Value>()
                + std::mem::size_of::<LogicalType>()
        );
        assert_eq!(
            chunk.granted_bytes(),
            direct + payload_before
                - super::super::value_codec::decoded_bytes_payload_retained_bytes(1).unwrap()
        );
        assert_eq!(manager.allocated(), chunk.granted_bytes());
        drop(chunk);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(all(feature = "spill", feature = "lpg"))]
    fn sort_provenance_schema_allocation_failure_releases_payload_and_preserves_retry_grant() {
        let manager = BufferManager::with_budget(2 * 1024 * 1024);
        let context = QueryResourceContext::new(manager.clone()).unwrap();
        let source = [
            Value::List(vec![Value::Int64(7)].into()),
            Value::Bytes(vec![1u8].into()),
        ];
        let mut encoded = Vec::new();
        serialize_row_with_limits(&source, &mut encoded, CodecLimits::format_max()).unwrap();
        let decoded =
            deserialize_framed_row_exact_with_receipt(&encoded, 2, CodecLimits::format_max())
                .unwrap();
        let row = AccountedOrdinalRow::try_from_decoded(
            decoded,
            0,
            context.try_allocate(1024 * 1024).unwrap(),
            Some(&NoopGrantObserver),
        )
        .unwrap();
        let Value::List(list) = &row.values()[0] else {
            panic!("expected edge IDs")
        };
        let list_weak = Arc::downgrade(list);
        let Value::Bytes(mask) = &row.values()[1] else {
            panic!("expected private mask")
        };
        let mask_weak = Arc::downgrade(mask);
        let schema = std::alloc::Layout::new::<LogicalType>();
        // Skip matching one-column catalog/Value-slot allocations, if their
        // layouts happen to equal the schema layout on this architecture.
        let skip = usize::from(std::alloc::Layout::new::<ValueVector>() == schema)
            + usize::from(std::alloc::Layout::new::<Value>() == schema);
        let observer = TrackingGrantObserver::default();
        let (result, fired) =
            crate::allocation_test::with_failure(schema.size(), schema.align(), skip, || {
                try_accounted_chunk_from_sort_row(row, 1, &observer)
            });
        assert!(fired, "must refuse the actual fallible schema allocation");
        let error = result.unwrap_err();
        assert!(matches!(
            error.primary(),
            AccountedSortChunkPrimary::Capacity(
                crate::execution::vector::ResidentCapacityError::ExactAllocation {
                    container: "exact edge-list schema",
                    ..
                }
            )
        ));
        assert_eq!(list_weak.strong_count(), 0);
        assert_eq!(mask_weak.strong_count(), 0);
        assert!(error.granted_bytes() > 0);
        assert_eq!(manager.allocated(), error.granted_bytes());
        assert_eq!(observer.retained.get(), error.granted_bytes());
        let (_, mut grant) = error.into_parts();
        grant.try_resize(0).unwrap();
        observer.publish_retained(0).unwrap();
        assert_eq!(manager.allocated(), 0);
        let decoded =
            deserialize_framed_row_exact_with_receipt(&encoded, 2, CodecLimits::format_max())
                .unwrap();
        let row = AccountedOrdinalRow::try_from_decoded(
            decoded,
            0,
            context.try_allocate(1024 * 1024).unwrap(),
            Some(&NoopGrantObserver),
        )
        .unwrap();
        let chunk = try_accounted_chunk_from_sort_row(row, 1, &observer).unwrap();
        assert_eq!(
            chunk.chunk().column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        drop(chunk);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn failed_chunk_construction_returns_authority_after_destroying_physical_payload() {
        let manager = Arc::new(BufferManager::with_budget(2 * 1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let source = vec![Value::List(Arc::from([
            Value::from("failed"),
            Value::Int64(23),
        ]))];
        let mut encoded = Vec::new();
        serialize_row_with_limits(&source, &mut encoded, CodecLimits::format_max()).unwrap();
        let decoded =
            deserialize_framed_row_exact_with_receipt(&encoded, 1, CodecLimits::format_max())
                .unwrap();
        let row = AccountedOrdinalRow::try_from_decoded(
            decoded,
            31,
            context.try_allocate(1024 * 1024).unwrap(),
            Some(&NoopGrantObserver),
        )
        .unwrap();
        let Value::List(list) = &row.values()[0] else {
            panic!("decoded nested value changed shape")
        };
        let weak = Arc::downgrade(list);
        let row_bytes = row.granted_bytes();
        drop(source);
        let observer = TrackingGrantObserver::default();

        let error = try_accounted_chunk_from_sort_row(row, 2, &observer).unwrap_err();

        assert_eq!(weak.strong_count(), 0);
        assert_eq!(error.granted_bytes(), row_bytes);
        assert_eq!(observer.retained.get(), row_bytes);
        assert_eq!(manager.allocated(), row_bytes);
        let (primary, mut grant) = error.into_parts();
        assert!(matches!(
            primary,
            AccountedSortChunkPrimary::InvalidShape(
                "decoded sort row does not match its non-empty output schema"
            )
        ));
        grant.try_resize(0).unwrap();
        observer.publish_retained(0).unwrap();
        assert_eq!(observer.retained.get(), 0);
        assert_eq!(manager.allocated(), 0);
    }
}
