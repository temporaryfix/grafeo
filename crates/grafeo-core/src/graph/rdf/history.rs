//! Authoritative, graph-qualified RDF temporal history.
//!
//! The live RDF indexes answer current queries.  This module is the durable
//! history contract: typed quads, named-graph lifetimes, full statement
//! handles, transaction-time cuts, ordered diffs, and CDC pages.  CDC is
//! derived from the same persisted interval records instead of maintaining a
//! second volatile event log.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;
use std::collections::{BTreeMap, BTreeSet};

use grafeo_common::types::{
    EpochId, EpochInterval, GraphIncarnationId, HistoryCompleteness, StatementHandle, StoreId,
    TaiNanoseconds, ValidTimeInterval,
};
use serde::{Deserialize, Serialize};

use super::{Quad, Term};

const STATEMENT_HANDLE_CONTEXT: &str = "org.grafeo.rdf.statement-handle.v1";
const CURSOR_PREFIX: &str = "rh1_";
/// Defensive limit for graph names copied into an opaque cursor token.
pub const MAX_CURSOR_GRAPH_NAME_BYTES: usize = 64 * 1024;
const MAX_CURSOR_PAYLOAD_BYTES: usize =
    StoreId::LEN + 8 + 1 + 1 + 4 + MAX_CURSOR_GRAPH_NAME_BYTES + 8 + StatementHandle::LEN;
/// Maximum encoded cursor-token length, including the version prefix.
pub const MAX_CURSOR_TOKEN_BYTES: usize = CURSOR_PREFIX.len() + MAX_CURSOR_PAYLOAD_BYTES * 2;

/// The default graph or one specific lifetime of a named graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RdfGraphIdentity {
    name: Option<String>,
    incarnation: GraphIncarnationId,
}

impl RdfGraphIdentity {
    /// Permanent identity of the default graph.
    #[must_use]
    pub const fn default_graph() -> Self {
        Self {
            name: None,
            incarnation: GraphIncarnationId::DEFAULT_GRAPH,
        }
    }

    /// Constructs one named-graph lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`RdfHistoryError::InvalidGraphIdentity`] if `incarnation` is
    /// the reserved default-graph value.
    pub fn named(
        name: impl Into<String>,
        incarnation: GraphIncarnationId,
    ) -> Result<Self, RdfHistoryError> {
        let name = name.into();
        if incarnation.is_default_graph() || name.is_empty() || name.chars().any(char::is_control) {
            return Err(RdfHistoryError::InvalidGraphIdentity {
                graph: Some(name),
                incarnation,
            });
        }
        if name.len() > MAX_CURSOR_GRAPH_NAME_BYTES {
            return Err(RdfHistoryError::GraphNameTooLong {
                bytes: name.len(),
                maximum: MAX_CURSOR_GRAPH_NAME_BYTES,
            });
        }
        Ok(Self {
            name: Some(name),
            incarnation,
        })
    }

    /// Constructs the identity matching an RDF quad's graph component.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph term and incarnation do not describe
    /// the same default or named graph identity.
    pub fn for_quad(quad: &Quad, incarnation: GraphIncarnationId) -> Result<Self, RdfHistoryError> {
        match quad.graph() {
            None if incarnation.is_default_graph() => Ok(Self::default_graph()),
            Some(name) if !incarnation.is_default_graph() => {
                Self::named(name.to_string(), incarnation)
            }
            graph => Err(RdfHistoryError::InvalidGraphIdentity {
                graph: graph.map(str::to_owned),
                incarnation,
            }),
        }
    }

    /// Named graph IRI, or `None` for the default graph.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Durable graph-lifetime identifier.
    #[must_use]
    pub const fn incarnation(&self) -> GraphIncarnationId {
        self.incarnation
    }

    /// Whether this is the permanent default graph.
    #[must_use]
    pub const fn is_default_graph(&self) -> bool {
        self.name.is_none()
    }
}

/// Transaction-time lifetime of one named-graph incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfGraphLife {
    graph: RdfGraphIdentity,
    tx: EpochInterval,
}

impl RdfGraphLife {
    /// Constructs a validated named-graph lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error for the default graph or an invalid transaction-time
    /// interval.
    pub fn new(graph: RdfGraphIdentity, tx: EpochInterval) -> Result<Self, RdfHistoryError> {
        if graph.is_default_graph() {
            return Err(RdfHistoryError::DefaultGraphHasLifecycle);
        }
        validate_tx_interval(tx)?;
        Ok(Self { graph, tx })
    }

    /// Named-graph identity.
    #[must_use]
    pub const fn graph(&self) -> &RdfGraphIdentity {
        &self.graph
    }

    /// Transaction-time interval over which the graph existed.
    #[must_use]
    pub const fn tx(&self) -> EpochInterval {
        self.tx
    }
}

/// One persisted transaction-time version of a typed RDF quad.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfQuadVersion {
    quad: Quad,
    graph_incarnation: GraphIncarnationId,
    statement: StatementHandle,
    tx: EpochInterval,
    valid: Option<ValidTimeInterval>,
}

impl RdfQuadVersion {
    /// Constructs a quad version and computes its stable statement handle.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid graph identity or transaction-time
    /// interval.
    pub fn new(
        store_id: StoreId,
        quad: Quad,
        graph_incarnation: GraphIncarnationId,
        tx: EpochInterval,
        valid: Option<ValidTimeInterval>,
    ) -> Result<Self, RdfHistoryError> {
        validate_tx_interval(tx)?;
        let statement = statement_handle(store_id, &quad, graph_incarnation)?;
        Ok(Self {
            quad,
            graph_incarnation,
            statement,
            tx,
            valid,
        })
    }

    /// Restores a persisted version while verifying its handle.
    ///
    /// This detects corruption or a producer that used a different graph
    /// incarnation/canonicalisation contract.
    ///
    /// # Errors
    ///
    /// Returns an error when any persisted invariant is invalid or the stored
    /// handle does not match the canonical handle.
    pub fn from_persisted(
        store_id: StoreId,
        quad: Quad,
        graph_incarnation: GraphIncarnationId,
        statement: StatementHandle,
        tx: EpochInterval,
        valid: Option<ValidTimeInterval>,
    ) -> Result<Self, RdfHistoryError> {
        let version = Self::new(store_id, quad, graph_incarnation, tx, valid)?;
        if version.statement != statement {
            return Err(RdfHistoryError::StatementHandleMismatch {
                expected: version.statement,
                stored: statement,
            });
        }
        Ok(version)
    }

    /// Graph-qualified typed quad.
    #[must_use]
    pub const fn quad(&self) -> &Quad {
        &self.quad
    }

    /// Named-graph incarnation, or the reserved default incarnation.
    #[must_use]
    pub const fn graph_incarnation(&self) -> GraphIncarnationId {
        self.graph_incarnation
    }

    /// Full stable statement handle.
    #[must_use]
    pub const fn statement(&self) -> StatementHandle {
        self.statement
    }

    /// Transaction-time interval.
    #[must_use]
    pub const fn tx(&self) -> EpochInterval {
        self.tx
    }

    /// Optional application valid-time interval.
    #[must_use]
    pub const fn valid(&self) -> Option<ValidTimeInterval> {
        self.valid
    }

    fn graph_identity(&self) -> Result<RdfGraphIdentity, RdfHistoryError> {
        RdfGraphIdentity::for_quad(&self.quad, self.graph_incarnation)
    }
}

/// A quad visible in an RDF dataset cut.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfHistoricalQuad {
    /// Graph-qualified typed RDF statement.
    pub quad: Quad,
    /// Exact graph lifetime containing the statement.
    pub graph_incarnation: GraphIncarnationId,
    /// Stable 256-bit statement handle.
    pub statement: StatementHandle,
    /// Optional application valid-time interval.
    pub valid: Option<ValidTimeInterval>,
}

/// Complete RDF dataset membership at one committed epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfHistoryCut {
    /// Logical store whose history produced this cut.
    pub store_id: StoreId,
    /// Transaction-time cut.
    pub epoch: EpochId,
    /// Named-graph incarnations alive at the cut, in canonical order.
    pub named_graphs: Vec<RdfGraphIdentity>,
    /// Quads visible at the cut, in stable handle order.
    pub quads: Vec<RdfHistoricalQuad>,
    /// Provenance boundary inherited from the persisted history.
    pub completeness: HistoryCompleteness,
}

/// One canonical transition derived from durable interval history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfHistoryTransition {
    /// Commit epoch at which the transition became visible.
    pub epoch: EpochId,
    /// Exact graph incarnation affected by the transition.
    pub graph: RdfGraphIdentity,
    /// Lifecycle or statement transition.
    pub kind: RdfHistoryTransitionKind,
}

impl RdfHistoryTransition {
    /// Stable statement handle for statement transitions.
    #[must_use]
    pub const fn statement(&self) -> Option<StatementHandle> {
        match &self.kind {
            RdfHistoryTransitionKind::StatementAsserted { statement, .. }
            | RdfHistoryTransitionKind::StatementRetracted { statement, .. } => Some(*statement),
            RdfHistoryTransitionKind::GraphCreated | RdfHistoryTransitionKind::GraphDropped => None,
        }
    }

    fn key(&self, store_id: StoreId) -> TransitionKey {
        TransitionKey {
            store_id,
            epoch: self.epoch,
            phase: self.kind.phase(),
            graph: self.graph.clone(),
            statement: self.statement().unwrap_or_else(zero_statement_handle),
        }
    }
}

/// Kind and payload of one RDF history transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RdfHistoryTransitionKind {
    /// A named-graph incarnation became visible.
    GraphCreated,
    /// A typed quad became visible.
    StatementAsserted {
        /// Graph-qualified typed quad.
        quad: Quad,
        /// Stable statement identity.
        statement: StatementHandle,
        /// Optional application valid-time interval.
        valid: Option<ValidTimeInterval>,
    },
    /// A typed quad ceased to be visible.
    StatementRetracted {
        /// Graph-qualified typed quad.
        quad: Quad,
        /// Stable statement identity.
        statement: StatementHandle,
        /// Optional application valid-time interval carried by the assertion.
        valid: Option<ValidTimeInterval>,
    },
    /// A named-graph incarnation ceased to exist.
    GraphDropped,
}

impl RdfHistoryTransitionKind {
    // Canonical atomic-epoch ordering: remove the old graph lifetime before
    // introducing a replacement with the same name, while preserving
    // retract-before-drop and create-before-assert dependencies.
    const fn phase(&self) -> u8 {
        match self {
            Self::StatementRetracted { .. } => 0,
            Self::GraphDropped => 1,
            Self::GraphCreated => 2,
            Self::StatementAsserted { .. } => 3,
        }
    }
}

/// Ordered transition range `(from, through]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfHistoryDiff {
    /// Logical store whose history produced this diff.
    pub store_id: StoreId,
    /// Exclusive lower transaction-time bound.
    pub from: EpochId,
    /// Inclusive upper transaction-time bound.
    pub through: EpochId,
    /// Canonically ordered lifecycle and statement transitions.
    pub transitions: Vec<RdfHistoryTransition>,
    /// Provenance boundary inherited from the persisted history.
    pub completeness: HistoryCompleteness,
}

/// Opaque stable position in canonical RDF history order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfHistoryCursor(TransitionKey);

impl RdfHistoryCursor {
    /// Encodes this cursor as a portable opaque token.
    ///
    /// # Errors
    ///
    /// Returns [`RdfHistoryCursorError::GraphNameTooLong`] for a cursor
    /// constructed by unchecked deserialization with an oversized graph name.
    pub fn to_token(&self) -> Result<String, RdfHistoryCursorError> {
        let graph_name_len = self.0.graph.name().map_or(0, str::len);
        if graph_name_len > MAX_CURSOR_GRAPH_NAME_BYTES {
            return Err(RdfHistoryCursorError::GraphNameTooLong {
                bytes: graph_name_len,
                maximum: MAX_CURSOR_GRAPH_NAME_BYTES,
            });
        }
        let mut bytes = Vec::with_capacity(118 + graph_name_len);
        bytes.extend_from_slice(self.0.store_id.as_bytes());
        bytes.extend_from_slice(&self.0.epoch.as_u64().to_be_bytes());
        bytes.push(self.0.phase);
        match self.0.graph.name() {
            None => bytes.push(0),
            Some(name) => {
                bytes.push(1);
                let len = u32::try_from(name.len()).map_err(|_| {
                    RdfHistoryCursorError::GraphNameTooLong {
                        bytes: name.len(),
                        maximum: MAX_CURSOR_GRAPH_NAME_BYTES,
                    }
                })?;
                bytes.extend_from_slice(&len.to_be_bytes());
                bytes.extend_from_slice(name.as_bytes());
            }
        }
        bytes.extend_from_slice(&self.0.graph.incarnation().as_u64().to_be_bytes());
        bytes.extend_from_slice(self.0.statement.as_bytes());
        Ok(format!("{CURSOR_PREFIX}{}", encode_hex(&bytes)))
    }

    /// Logical store whose transition ordering this cursor addresses.
    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.0.store_id
    }

    /// Epoch of the transition immediately preceding the next page.
    #[must_use]
    pub const fn epoch(&self) -> EpochId {
        self.0.epoch
    }
}

impl fmt::Display for RdfHistoryCursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_token().map_err(|_| fmt::Error)?)
    }
}

impl FromStr for RdfHistoryCursor {
    type Err = RdfHistoryCursorError;

    fn from_str(token: &str) -> Result<Self, Self::Err> {
        if token.len() > MAX_CURSOR_TOKEN_BYTES {
            return Err(RdfHistoryCursorError::TokenTooLong {
                bytes: token.len(),
                maximum: MAX_CURSOR_TOKEN_BYTES,
            });
        }
        let encoded = token
            .strip_prefix(CURSOR_PREFIX)
            .ok_or(RdfHistoryCursorError::InvalidPrefix)?;
        let bytes = decode_hex(encoded)?;
        let mut pos = 0usize;
        let store_end = pos
            .checked_add(StoreId::LEN)
            .ok_or(RdfHistoryCursorError::Truncated)?;
        let store_bytes: [u8; StoreId::LEN] = bytes
            .get(pos..store_end)
            .ok_or(RdfHistoryCursorError::Truncated)?
            .try_into()
            .map_err(|_| RdfHistoryCursorError::Truncated)?;
        pos = store_end;
        let store_id = StoreId::from_bytes(store_bytes)
            .map_err(|_| RdfHistoryCursorError::InvalidStoreIdentity)?;
        let epoch = EpochId::new(read_u64(&bytes, &mut pos)?);
        let phase = read_byte(&bytes, &mut pos)?;
        if phase > 3 {
            return Err(RdfHistoryCursorError::InvalidPhase(phase));
        }
        let graph_tag = read_byte(&bytes, &mut pos)?;
        let graph_name = match graph_tag {
            0 => None,
            1 => {
                let len = read_u32(&bytes, &mut pos)? as usize;
                if len > MAX_CURSOR_GRAPH_NAME_BYTES {
                    return Err(RdfHistoryCursorError::GraphNameTooLong {
                        bytes: len,
                        maximum: MAX_CURSOR_GRAPH_NAME_BYTES,
                    });
                }
                let end = pos
                    .checked_add(len)
                    .ok_or(RdfHistoryCursorError::Truncated)?;
                let raw = bytes
                    .get(pos..end)
                    .ok_or(RdfHistoryCursorError::Truncated)?;
                pos = end;
                Some(
                    core::str::from_utf8(raw)
                        .map_err(|_| RdfHistoryCursorError::InvalidGraphName)?
                        .to_string(),
                )
            }
            tag => return Err(RdfHistoryCursorError::InvalidGraphTag(tag)),
        };
        let incarnation = GraphIncarnationId::new(read_u64(&bytes, &mut pos)?);
        let handle_end = pos
            .checked_add(StatementHandle::LEN)
            .ok_or(RdfHistoryCursorError::Truncated)?;
        let handle_bytes: [u8; StatementHandle::LEN] = bytes
            .get(pos..handle_end)
            .ok_or(RdfHistoryCursorError::Truncated)?
            .try_into()
            .map_err(|_| RdfHistoryCursorError::Truncated)?;
        pos = handle_end;
        if pos != bytes.len() {
            return Err(RdfHistoryCursorError::TrailingBytes);
        }
        let graph = match graph_name {
            None if incarnation.is_default_graph() => RdfGraphIdentity::default_graph(),
            Some(name) if !incarnation.is_default_graph() && !name.is_empty() => {
                RdfGraphIdentity::named(name, incarnation)
                    .map_err(|_| RdfHistoryCursorError::InvalidGraphIdentity)?
            }
            _ => return Err(RdfHistoryCursorError::InvalidGraphIdentity),
        };
        Ok(Self(TransitionKey {
            store_id,
            epoch,
            phase,
            graph,
            statement: StatementHandle::from_bytes(handle_bytes),
        }))
    }
}

/// Cursor decoding failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RdfHistoryCursorError {
    /// Token version/prefix is not supported.
    #[error("invalid RDF history cursor prefix")]
    InvalidPrefix,
    /// Encoded token exceeds the total defensive allocation limit.
    #[error("RDF history cursor has {bytes} bytes; maximum is {maximum}")]
    TokenTooLong {
        /// Supplied encoded byte length.
        bytes: usize,
        /// Maximum accepted encoded byte length.
        maximum: usize,
    },
    /// Hex payload is malformed.
    #[error("RDF history cursor contains invalid hexadecimal data")]
    InvalidHex,
    /// Store identity is the reserved zero value.
    #[error("RDF history cursor contains an invalid store identity")]
    InvalidStoreIdentity,
    /// Payload ended before a complete cursor could be read.
    #[error("RDF history cursor is truncated")]
    Truncated,
    /// Unknown transition phase.
    #[error("RDF history cursor has invalid transition phase {0}")]
    InvalidPhase(u8),
    /// Unknown graph tag.
    #[error("RDF history cursor has invalid graph tag {0}")]
    InvalidGraphTag(u8),
    /// Named graph is not UTF-8.
    #[error("RDF history cursor graph name is not UTF-8")]
    InvalidGraphName,
    /// Named graph exceeds the defensive cursor-token limit.
    #[error("RDF history cursor graph name has {bytes} bytes; maximum is {maximum}")]
    GraphNameTooLong {
        /// Supplied byte length.
        bytes: usize,
        /// Maximum accepted byte length.
        maximum: usize,
    },
    /// Graph kind and incarnation disagree.
    #[error("RDF history cursor has an invalid graph incarnation")]
    InvalidGraphIdentity,
    /// Payload has unframed trailing bytes.
    #[error("RDF history cursor contains trailing bytes")]
    TrailingBytes,
}

/// Stable page of durable RDF CDC transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfCdcPage {
    /// Logical store whose durable history produced this page.
    pub store_id: StoreId,
    /// Canonically ordered transitions.
    pub transitions: Vec<RdfHistoryTransition>,
    /// Cursor for the last delivered transition. Preserve it even when this
    /// bounded page is exhausted so a durable consumer can later extend `through`.
    pub next_cursor: Option<RdfHistoryCursor>,
    /// Whether another row exists within the requested bound.
    pub has_more: bool,
    /// Provenance boundary inherited from persisted history.
    pub completeness: HistoryCompleteness,
}

/// Immutable validated view over persisted RDF dataset history.
#[derive(Debug, Clone)]
pub struct RdfDatasetHistory {
    store_id: StoreId,
    completeness: HistoryCompleteness,
    next_graph_incarnation: GraphIncarnationId,
    graph_lives: Vec<RdfGraphLife>,
    quad_versions: Vec<RdfQuadVersion>,
}

impl RdfDatasetHistory {
    /// Validates and indexes persisted graph and quad interval records.
    ///
    /// # Errors
    ///
    /// Returns an error when graph lifetimes, quad intervals, or canonical
    /// statement handles are inconsistent.
    pub fn new(
        store_id: StoreId,
        completeness: HistoryCompleteness,
        graph_lives: Vec<RdfGraphLife>,
        quad_versions: Vec<RdfQuadVersion>,
    ) -> Result<Self, RdfHistoryError> {
        let maximum = graph_lives
            .iter()
            .map(|life| life.graph.incarnation())
            .max()
            .unwrap_or(GraphIncarnationId::DEFAULT_GRAPH);
        let next_graph_incarnation = maximum
            .checked_next()
            .ok_or(RdfHistoryError::GraphIncarnationExhausted)?
            .max(GraphIncarnationId::FIRST_NAMED);
        Self::new_with_high_water(
            store_id,
            completeness,
            next_graph_incarnation,
            graph_lives,
            quad_versions,
        )
    }

    /// Validates persisted history together with its durable allocator
    /// high-water mark.
    ///
    /// The high-water mark may exceed the visible maximum because aborted
    /// graph reservations intentionally leave permanent gaps.
    ///
    /// # Errors
    ///
    /// Returns an error when the allocator could reuse a persisted
    /// incarnation or any dataset-history invariant is inconsistent.
    pub fn new_with_high_water(
        store_id: StoreId,
        completeness: HistoryCompleteness,
        next_graph_incarnation: GraphIncarnationId,
        mut graph_lives: Vec<RdfGraphLife>,
        mut quad_versions: Vec<RdfQuadVersion>,
    ) -> Result<Self, RdfHistoryError> {
        validate_dataset(store_id, &graph_lives, &quad_versions)?;
        let maximum = graph_lives
            .iter()
            .map(|life| life.graph.incarnation())
            .max()
            .unwrap_or(GraphIncarnationId::DEFAULT_GRAPH);
        if next_graph_incarnation.is_default_graph() || next_graph_incarnation <= maximum {
            return Err(RdfHistoryError::InvalidGraphIncarnationHighWater {
                next: next_graph_incarnation,
                maximum,
            });
        }
        graph_lives.sort_by_key(|life| (life.graph.clone(), life.tx.from()));
        quad_versions.sort_by_key(|version| (version.statement, version.tx.from()));
        Ok(Self {
            store_id,
            completeness,
            next_graph_incarnation,
            graph_lives,
            quad_versions,
        })
    }

    /// Portable identity of the logical store owning this history.
    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.store_id
    }

    /// Provenance boundary for this history.
    #[must_use]
    pub const fn completeness(&self) -> HistoryCompleteness {
        self.completeness
    }

    /// First named-graph incarnation not yet reserved by this dataset.
    #[must_use]
    pub const fn next_graph_incarnation(&self) -> GraphIncarnationId {
        self.next_graph_incarnation
    }

    /// Creates a detached logical fork under a new store identity.
    ///
    /// Transaction intervals, valid time, graph lifetimes, incarnation values,
    /// completeness, and allocator high-water are preserved exactly. Every
    /// statement handle is recomputed under `new_store_id`, then the detached
    /// history is validated and canonicalized before it is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if `new_store_id` is unchanged or if recomputing the
    /// detached history exposes an invalid persisted invariant.
    pub fn reidentify_for_fork(&self, new_store_id: StoreId) -> Result<Self, RdfHistoryError> {
        if new_store_id == self.store_id {
            return Err(RdfHistoryError::ForkStoreIdentityUnchanged);
        }
        let quad_versions = self
            .quad_versions
            .iter()
            .map(|version| {
                RdfQuadVersion::new(
                    new_store_id,
                    version.quad.clone(),
                    version.graph_incarnation,
                    version.tx,
                    version.valid,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new_with_high_water(
            new_store_id,
            self.completeness,
            self.next_graph_incarnation,
            self.graph_lives.clone(),
            quad_versions,
        )
    }

    /// Persistable named-graph lifecycle records.
    #[must_use]
    pub fn graph_lives(&self) -> &[RdfGraphLife] {
        &self.graph_lives
    }

    /// Persistable graph-qualified quad interval records.
    #[must_use]
    pub fn quad_versions(&self) -> &[RdfQuadVersion] {
        &self.quad_versions
    }

    /// Returns the complete typed-quad dataset visible at `epoch`.
    ///
    /// # Errors
    ///
    /// Returns an error for a pending cut or a cut before the truthful legacy
    /// history boundary.
    pub fn cut(&self, epoch: EpochId) -> Result<RdfHistoryCut, RdfHistoryError> {
        self.cut_at(epoch, None)
    }

    /// Returns the typed-quad dataset at a transaction cut and optional valid instant.
    ///
    /// Versions without an application valid-time interval are always valid.
    /// Unlike the legacy valid-time API, this considers the transaction-time
    /// version selected by `epoch`, not only versions that are currently open.
    ///
    /// # Errors
    ///
    /// Returns an error for a pending cut or a cut before the truthful legacy
    /// history boundary.
    pub fn cut_at(
        &self,
        epoch: EpochId,
        valid_at: Option<TaiNanoseconds>,
    ) -> Result<RdfHistoryCut, RdfHistoryError> {
        self.require_cut(epoch)?;
        let mut named_graphs: Vec<RdfGraphIdentity> = self
            .graph_lives
            .iter()
            .filter(|life| life.tx.contains(epoch))
            .map(|life| life.graph.clone())
            .collect();
        named_graphs.sort();

        let mut quads: Vec<RdfHistoricalQuad> = self
            .quad_versions
            .iter()
            .filter(|version| {
                version.tx.contains(epoch)
                    && valid_at.is_none_or(|instant| {
                        version.valid.is_none_or(|valid| valid.contains(instant))
                    })
            })
            .map(|version| RdfHistoricalQuad {
                quad: version.quad.clone(),
                graph_incarnation: version.graph_incarnation,
                statement: version.statement,
                valid: version.valid,
            })
            .collect();
        quads.sort_by_key(|quad| quad.statement);
        Ok(RdfHistoryCut {
            store_id: self.store_id,
            epoch,
            named_graphs,
            quads,
            completeness: self.completeness,
        })
    }

    /// Returns lifecycle and statement transitions in canonical `(from, through]` order.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, pending, or incomplete bounds.
    pub fn ordered_diff(
        &self,
        from: EpochId,
        through: EpochId,
    ) -> Result<RdfHistoryDiff, RdfHistoryError> {
        self.require_range(from, through)?;
        let transitions = self.transitions_in_range(from, through)?;
        Ok(RdfHistoryDiff {
            store_id: self.store_id,
            from,
            through,
            transitions,
            completeness: self.completeness,
        })
    }

    /// Returns one stable page of durable RDF CDC derived from interval history.
    ///
    /// `from` is exclusive and `through` is inclusive.  Keep `through` fixed
    /// while following `next_cursor` to obtain a repeatable finite scan.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bounds, a zero page size, a cross-store or
    /// forged cursor, or a cut before the truthful legacy history boundary.
    pub fn cdc_page(
        &self,
        from: EpochId,
        through: EpochId,
        after: Option<&RdfHistoryCursor>,
        limit: usize,
    ) -> Result<RdfCdcPage, RdfHistoryError> {
        self.require_range(from, through)?;
        if limit == 0 {
            return Err(RdfHistoryError::ZeroPageSize);
        }
        if let Some(cursor) = after {
            if cursor.0.store_id != self.store_id {
                return Err(RdfHistoryError::CrossStoreCursor {
                    expected: self.store_id,
                    cursor: cursor.0.store_id,
                });
            }
            if cursor.0.epoch <= from || cursor.0.epoch > through {
                return Err(RdfHistoryError::CursorOutsideRange {
                    cursor_epoch: cursor.0.epoch,
                    from,
                    through,
                });
            }
        }

        let all_transitions = self.transitions_in_range(from, through)?;
        if let Some(cursor) = after
            && !all_transitions
                .iter()
                .any(|row| row.key(self.store_id) == cursor.0)
        {
            return Err(RdfHistoryError::CursorNotFound);
        }
        let after_key = after.map(|cursor| &cursor.0);
        let mut rows = all_transitions.into_iter().filter(|row| {
            after_key.is_none_or(|cursor| row.key(self.store_id).cmp(cursor) == Ordering::Greater)
        });
        let mut transitions = Vec::with_capacity(limit);
        for _ in 0..limit {
            let Some(row) = rows.next() else {
                break;
            };
            transitions.push(row);
        }
        let has_more = rows.next().is_some();
        let next_cursor = transitions
            .last()
            .map(|row| RdfHistoryCursor(row.key(self.store_id)))
            .or_else(|| after.cloned());
        Ok(RdfCdcPage {
            store_id: self.store_id,
            transitions,
            next_cursor,
            has_more,
            completeness: self.completeness,
        })
    }

    fn require_cut(&self, epoch: EpochId) -> Result<(), RdfHistoryError> {
        if epoch == EpochId::PENDING {
            return Err(RdfHistoryError::PendingEpoch);
        }
        if let Some(authoritative_from) = self.completeness.authoritative_from()
            && epoch < authoritative_from
        {
            return Err(RdfHistoryError::IncompleteHistory {
                requested_from: epoch,
                authoritative_from,
            });
        }
        Ok(())
    }

    fn require_range(&self, from: EpochId, through: EpochId) -> Result<(), RdfHistoryError> {
        if from == EpochId::PENDING || through == EpochId::PENDING {
            return Err(RdfHistoryError::PendingEpoch);
        }
        if through < from {
            return Err(RdfHistoryError::InvertedRange { from, through });
        }
        if let Some(authoritative_from) = self.completeness.authoritative_from()
            && from < authoritative_from
        {
            return Err(RdfHistoryError::IncompleteHistory {
                requested_from: from,
                authoritative_from,
            });
        }
        Ok(())
    }

    fn transitions_in_range(
        &self,
        from: EpochId,
        through: EpochId,
    ) -> Result<Vec<RdfHistoryTransition>, RdfHistoryError> {
        let mut transitions = Vec::new();
        for life in &self.graph_lives {
            if from < life.tx.from() && life.tx.from() <= through {
                transitions.push(RdfHistoryTransition {
                    epoch: life.tx.from(),
                    graph: life.graph.clone(),
                    kind: RdfHistoryTransitionKind::GraphCreated,
                });
            }
            if !life.tx.is_open() && from < life.tx.to() && life.tx.to() <= through {
                transitions.push(RdfHistoryTransition {
                    epoch: life.tx.to(),
                    graph: life.graph.clone(),
                    kind: RdfHistoryTransitionKind::GraphDropped,
                });
            }
        }
        for version in &self.quad_versions {
            let graph = version.graph_identity()?;
            if from < version.tx.from() && version.tx.from() <= through {
                transitions.push(RdfHistoryTransition {
                    epoch: version.tx.from(),
                    graph: graph.clone(),
                    kind: RdfHistoryTransitionKind::StatementAsserted {
                        quad: version.quad.clone(),
                        statement: version.statement,
                        valid: version.valid,
                    },
                });
            }
            if !version.tx.is_open() && from < version.tx.to() && version.tx.to() <= through {
                transitions.push(RdfHistoryTransition {
                    epoch: version.tx.to(),
                    graph,
                    kind: RdfHistoryTransitionKind::StatementRetracted {
                        quad: version.quad.clone(),
                        statement: version.statement,
                        valid: version.valid,
                    },
                });
            }
        }
        transitions.sort_by_cached_key(|transition| transition.key(self.store_id));
        Ok(transitions)
    }
}

/// RDF history validation or query failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RdfHistoryError {
    /// An explicit fork must enter a distinct statement-handle namespace.
    #[error("RDF history fork requires a different store identity")]
    ForkStoreIdentityUnchanged,
    /// The named-graph incarnation namespace cannot allocate another value.
    #[error("RDF named-graph incarnation space is exhausted")]
    GraphIncarnationExhausted,
    /// Persisted allocator high-water could reuse an existing incarnation.
    #[error("invalid RDF graph-incarnation high-water {next}; persisted maximum is {maximum}")]
    InvalidGraphIncarnationHighWater {
        /// First claimed unreserved incarnation.
        next: GraphIncarnationId,
        /// Maximum incarnation already present in history.
        maximum: GraphIncarnationId,
    },
    /// Graph kind and incarnation disagree.
    #[error("invalid RDF graph identity {graph:?} with incarnation {incarnation}")]
    InvalidGraphIdentity {
        /// Named graph IRI, or `None` for default.
        graph: Option<String>,
        /// Supplied graph incarnation.
        incarnation: GraphIncarnationId,
    },
    /// Named graph exceeds the defensive public cursor limit.
    #[error("RDF graph name has {bytes} bytes; maximum is {maximum}")]
    GraphNameTooLong {
        /// Supplied UTF-8 byte length.
        bytes: usize,
        /// Maximum accepted UTF-8 byte length.
        maximum: usize,
    },
    /// Default graph does not have create/drop lifecycle records.
    #[error("the permanent RDF default graph must not have a lifecycle record")]
    DefaultGraphHasLifecycle,
    /// Transaction interval is empty, inverted, or begins at the pending sentinel.
    #[error("invalid RDF transaction-time interval {from}..{to}")]
    InvalidTransactionInterval {
        /// Inclusive lower bound.
        from: EpochId,
        /// Exclusive upper bound or pending sentinel.
        to: EpochId,
    },
    /// Persisted handle does not match canonical content.
    #[error("persisted RDF statement handle mismatch: expected {expected}, stored {stored}")]
    StatementHandleMismatch {
        /// Canonically recomputed handle.
        expected: StatementHandle,
        /// Persisted handle.
        stored: StatementHandle,
    },
    /// One incarnation identifier was reused.
    #[error("RDF graph incarnation {0} is reused")]
    ReusedGraphIncarnation(GraphIncarnationId),
    /// Lifetimes for the same graph IRI overlap.
    #[error("RDF graph <{graph}> has overlapping incarnation lifetimes")]
    OverlappingGraphLives {
        /// Named graph IRI.
        graph: String,
    },
    /// A named-graph quad has no matching lifecycle.
    #[error("RDF quad references unknown graph incarnation {incarnation} for <{graph}>")]
    UnknownGraphIncarnation {
        /// Named graph IRI.
        graph: String,
        /// Referenced incarnation.
        incarnation: GraphIncarnationId,
    },
    /// Quad transaction interval escapes its graph lifetime.
    #[error("RDF statement {statement} escapes graph incarnation {incarnation} lifetime")]
    QuadOutsideGraphLife {
        /// Stable statement handle.
        statement: StatementHandle,
        /// Referenced incarnation.
        incarnation: GraphIncarnationId,
    },
    /// The same handle maps to conflicting statement content.
    #[error("RDF statement handle collision for {0}")]
    StatementHandleCollision(StatementHandle),
    /// Version intervals for one statement overlap.
    #[error("RDF statement {0} has overlapping transaction-time versions")]
    OverlappingStatementLives(StatementHandle),
    /// Requested history predates the truthful provenance boundary.
    #[error(
        "RDF history requested from epoch {requested_from}, but legacy history is authoritative only from epoch {authoritative_from}"
    )]
    IncompleteHistory {
        /// Requested cut or exclusive lower bound.
        requested_from: EpochId,
        /// Earliest authoritative boundary.
        authoritative_from: EpochId,
    },
    /// Upper bound precedes lower bound.
    #[error("inverted RDF history range ({from}, {through}]")]
    InvertedRange {
        /// Exclusive lower bound.
        from: EpochId,
        /// Inclusive upper bound.
        through: EpochId,
    },
    /// Pending is an uncommitted sentinel, not a reproducible history cut.
    #[error("EpochId::PENDING is not a committed RDF history bound")]
    PendingEpoch,
    /// A CDC page must make forward progress.
    #[error("RDF CDC page size must be greater than zero")]
    ZeroPageSize,
    /// Cursor does not belong to requested bounds.
    #[error("RDF CDC cursor epoch {cursor_epoch} is outside ({from}, {through}]")]
    CursorOutsideRange {
        /// Cursor epoch.
        cursor_epoch: EpochId,
        /// Exclusive lower bound.
        from: EpochId,
        /// Inclusive upper bound.
        through: EpochId,
    },
    /// Cursor belongs to another logical store.
    #[error("RDF CDC cursor belongs to store {cursor}, expected store {expected}")]
    CrossStoreCursor {
        /// Store serving the request.
        expected: StoreId,
        /// Store encoded by the cursor.
        cursor: StoreId,
    },
    /// Cursor key was forged or no longer names a transition in this range.
    #[error("RDF CDC cursor does not identify a transition in the requested range")]
    CursorNotFound,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TransitionKey {
    store_id: StoreId,
    epoch: EpochId,
    phase: u8,
    graph: RdfGraphIdentity,
    statement: StatementHandle,
}

impl Ord for TransitionKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.store_id,
            self.epoch,
            self.phase,
            &self.graph,
            self.statement,
        )
            .cmp(&(
                other.store_id,
                other.epoch,
                other.phase,
                &other.graph,
                other.statement,
            ))
    }
}

impl PartialOrd for TransitionKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Computes the canonical 256-bit handle for a quad in one store and graph incarnation.
///
/// The BLAKE3 derive-key context `org.grafeo.rdf.statement-handle.v1` is part of
/// the identity contract, followed by the full store identity, graph identity,
/// incarnation, and lossless typed RDF terms.
///
/// # Errors
///
/// Returns an error when the quad graph and supplied incarnation do not form a
/// valid graph identity.
pub fn statement_handle(
    store_id: StoreId,
    quad: &Quad,
    graph_incarnation: GraphIncarnationId,
) -> Result<StatementHandle, RdfHistoryError> {
    let graph = RdfGraphIdentity::for_quad(quad, graph_incarnation)?;
    let mut hasher = blake3::Hasher::new_derive_key(STATEMENT_HANDLE_CONTEXT);
    hasher.update(store_id.as_bytes());
    match graph.name() {
        None => {
            hasher.update(&[0]);
        }
        Some(name) => {
            hasher.update(&[1]);
            hash_bytes(&mut hasher, name.as_bytes());
        }
    }
    hasher.update(&graph.incarnation().as_u64().to_be_bytes());
    hash_term(&mut hasher, quad.triple().subject());
    hash_term(&mut hasher, quad.triple().predicate());
    hash_term(&mut hasher, quad.triple().object());
    Ok(StatementHandle::from_bytes(*hasher.finalize().as_bytes()))
}

fn hash_term(hasher: &mut blake3::Hasher, term: &Term) {
    match term {
        Term::Iri(iri) => {
            hasher.update(&[0]);
            hash_bytes(hasher, iri.as_str().as_bytes());
        }
        Term::BlankNode(blank) => {
            hasher.update(&[1]);
            hash_bytes(hasher, blank.id().as_bytes());
        }
        Term::Literal(literal) => {
            hasher.update(&[2]);
            hash_bytes(hasher, literal.value().as_bytes());
            hash_bytes(hasher, literal.datatype().as_bytes());
            match literal.language() {
                None => {
                    hasher.update(&[0]);
                }
                Some(language) => {
                    hasher.update(&[1]);
                    hash_bytes(hasher, language.as_bytes());
                }
            }
        }
    }
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn validate_tx_interval(tx: EpochInterval) -> Result<(), RdfHistoryError> {
    if tx.from() == EpochId::PENDING || (!tx.is_open() && tx.from() >= tx.to()) {
        return Err(RdfHistoryError::InvalidTransactionInterval {
            from: tx.from(),
            to: tx.to(),
        });
    }
    Ok(())
}

fn validate_dataset(
    store_id: StoreId,
    graph_lives: &[RdfGraphLife],
    quad_versions: &[RdfQuadVersion],
) -> Result<(), RdfHistoryError> {
    let mut incarnations = BTreeSet::new();
    let mut by_name: BTreeMap<&str, Vec<&RdfGraphLife>> = BTreeMap::new();
    let mut by_incarnation: BTreeMap<GraphIncarnationId, &RdfGraphLife> = BTreeMap::new();
    for life in graph_lives {
        validate_graph_identity(&life.graph, false)?;
        validate_tx_interval(life.tx)?;
        if !incarnations.insert(life.graph.incarnation()) {
            return Err(RdfHistoryError::ReusedGraphIncarnation(
                life.graph.incarnation(),
            ));
        }
        let Some(name) = life.graph.name() else {
            return Err(RdfHistoryError::DefaultGraphHasLifecycle);
        };
        by_name.entry(name).or_default().push(life);
        by_incarnation.insert(life.graph.incarnation(), life);
    }
    for (graph, lives) in &mut by_name {
        lives.sort_by_key(|life| life.tx.from());
        for pair in lives.windows(2) {
            let left = pair[0].tx;
            let right = pair[1].tx;
            if left.is_open() || left.to() > right.from() {
                return Err(RdfHistoryError::OverlappingGraphLives {
                    graph: (*graph).to_string(),
                });
            }
        }
    }

    let mut handle_content: BTreeMap<StatementHandle, (&Quad, GraphIncarnationId)> =
        BTreeMap::new();
    // Lossless spellings may have different stable handles, but membership
    // permits only one transaction-time lifetime per canonical graph/triple.
    let mut canonical_lives: BTreeMap<(RdfGraphIdentity, [String; 3]), Vec<&RdfQuadVersion>> =
        BTreeMap::new();
    for version in quad_versions {
        validate_tx_interval(version.tx)?;
        let graph = version.graph_identity()?;
        validate_graph_identity(&graph, true)?;
        let expected = statement_handle(store_id, &version.quad, version.graph_incarnation)?;
        if expected != version.statement {
            return Err(RdfHistoryError::StatementHandleMismatch {
                expected,
                stored: version.statement,
            });
        }
        if let Some(name) = graph.name() {
            let life = by_incarnation.get(&graph.incarnation()).ok_or_else(|| {
                RdfHistoryError::UnknownGraphIncarnation {
                    graph: name.to_string(),
                    incarnation: graph.incarnation(),
                }
            })?;
            if life.graph.name() != Some(name) || !interval_contains(life.tx, version.tx) {
                return Err(RdfHistoryError::QuadOutsideGraphLife {
                    statement: version.statement,
                    incarnation: graph.incarnation(),
                });
            }
        }
        if let Some((known_quad, known_incarnation)) = handle_content.get(&version.statement)
            && (*known_quad != &version.quad || *known_incarnation != version.graph_incarnation)
        {
            return Err(RdfHistoryError::StatementHandleCollision(version.statement));
        }
        handle_content.insert(
            version.statement,
            (&version.quad, version.graph_incarnation),
        );
        canonical_lives
            .entry((graph, version.quad.triple().canonical_identity_key()))
            .or_default()
            .push(version);
    }
    for lives in canonical_lives.values_mut() {
        lives.sort_by_key(|version| version.tx.from());
        for pair in lives.windows(2) {
            if pair[0].tx.is_open() || pair[0].tx.to() > pair[1].tx.from() {
                return Err(RdfHistoryError::OverlappingStatementLives(
                    pair[0].statement,
                ));
            }
        }
    }
    Ok(())
}

fn validate_graph_identity(
    graph: &RdfGraphIdentity,
    allow_default: bool,
) -> Result<(), RdfHistoryError> {
    match graph.name() {
        None if allow_default && graph.incarnation().is_default_graph() => Ok(()),
        None if graph.incarnation().is_default_graph() => {
            Err(RdfHistoryError::DefaultGraphHasLifecycle)
        }
        Some(name)
            if !graph.incarnation().is_default_graph()
                && !name.is_empty()
                && !name.chars().any(char::is_control)
                && name.len() <= MAX_CURSOR_GRAPH_NAME_BYTES =>
        {
            Ok(())
        }
        Some(name) if name.len() > MAX_CURSOR_GRAPH_NAME_BYTES => {
            Err(RdfHistoryError::GraphNameTooLong {
                bytes: name.len(),
                maximum: MAX_CURSOR_GRAPH_NAME_BYTES,
            })
        }
        name => Err(RdfHistoryError::InvalidGraphIdentity {
            graph: name.map(str::to_owned),
            incarnation: graph.incarnation(),
        }),
    }
}

fn interval_contains(outer: EpochInterval, inner: EpochInterval) -> bool {
    outer.from() <= inner.from()
        && match (outer.is_open(), inner.is_open()) {
            (true, _) => true,
            (false, true) => false,
            (false, false) => inner.to() <= outer.to(),
        }
}

fn zero_statement_handle() -> StatementHandle {
    StatementHandle::from_bytes([0; StatementHandle::LEN])
}

fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn decode_hex(encoded: &str) -> Result<Vec<u8>, RdfHistoryCursorError> {
    if !encoded.len().is_multiple_of(2) {
        return Err(RdfHistoryCursorError::InvalidHex);
    }
    let mut bytes = Vec::with_capacity(encoded.len() / 2);
    for pair in encoded.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0]).ok_or(RdfHistoryCursorError::InvalidHex)?;
        let low = hex_nibble(pair[1]).ok_or(RdfHistoryCursorError::InvalidHex)?;
        bytes.push((high << 4) | low);
    }
    Ok(bytes)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn read_byte(bytes: &[u8], pos: &mut usize) -> Result<u8, RdfHistoryCursorError> {
    let byte = *bytes.get(*pos).ok_or(RdfHistoryCursorError::Truncated)?;
    *pos += 1;
    Ok(byte)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, RdfHistoryCursorError> {
    let end = pos.checked_add(4).ok_or(RdfHistoryCursorError::Truncated)?;
    let raw: [u8; 4] = bytes
        .get(*pos..end)
        .ok_or(RdfHistoryCursorError::Truncated)?
        .try_into()
        .map_err(|_| RdfHistoryCursorError::Truncated)?;
    *pos = end;
    Ok(u32::from_be_bytes(raw))
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, RdfHistoryCursorError> {
    let end = pos.checked_add(8).ok_or(RdfHistoryCursorError::Truncated)?;
    let raw: [u8; 8] = bytes
        .get(*pos..end)
        .ok_or(RdfHistoryCursorError::Truncated)?
        .try_into()
        .map_err(|_| RdfHistoryCursorError::Truncated)?;
    *pos = end;
    Ok(u64::from_be_bytes(raw))
}

#[cfg(test)]
mod tests {
    use super::super::Triple;
    use super::*;

    fn triple(subject: &str) -> super::super::Triple {
        super::super::Triple::new(
            Term::iri(subject),
            Term::iri("urn:p"),
            Term::lang_literal("café", "fr"),
        )
    }

    fn e(value: u64) -> EpochId {
        EpochId::new(value)
    }

    fn sid(value: u8) -> StoreId {
        StoreId::from_bytes([value; StoreId::LEN]).unwrap()
    }

    #[test]
    fn canonical_alias_history_rejects_overlap_but_retains_separate_lifetimes() {
        let upper = Quad::new(Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::lang_literal("value", "EN"),
        ));
        let lower = Quad::new(Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::lang_literal("value", "en"),
        ));
        for first_interval in [EpochInterval::open(e(1)), EpochInterval::closed(e(1), e(4))] {
            let first = RdfQuadVersion::new(
                sid(1),
                upper.clone(),
                GraphIncarnationId::DEFAULT_GRAPH,
                first_interval,
                None,
            )
            .unwrap();
            let second = RdfQuadVersion::new(
                sid(1),
                lower.clone(),
                GraphIncarnationId::DEFAULT_GRAPH,
                EpochInterval::open(e(3)),
                None,
            )
            .unwrap();
            assert_ne!(first.statement(), second.statement());
            assert!(matches!(
                RdfDatasetHistory::new(
                    sid(1),
                    HistoryCompleteness::Complete,
                    vec![],
                    vec![second, first]
                ),
                Err(RdfHistoryError::OverlappingStatementLives(_))
            ));
        }
        let first = RdfQuadVersion::new(
            sid(1),
            upper.clone(),
            GraphIncarnationId::DEFAULT_GRAPH,
            EpochInterval::closed(e(1), e(3)),
            None,
        )
        .unwrap();
        let second = RdfQuadVersion::new(
            sid(1),
            lower.clone(),
            GraphIncarnationId::DEFAULT_GRAPH,
            EpochInterval::open(e(3)),
            None,
        )
        .unwrap();
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![],
            vec![second.clone(), first.clone()],
        )
        .unwrap();
        let before = history.cut(e(2)).unwrap();
        assert_eq!(before.quads[0].quad, upper);
        assert_eq!(before.quads[0].statement, first.statement());
        let after = history.cut(e(3)).unwrap();
        assert_eq!(after.quads[0].quad, lower);
        assert_eq!(after.quads[0].statement, second.statement());
    }

    #[test]
    fn canonical_history_membership_is_graph_qualified() {
        let make = |language| {
            Triple::new(
                Term::iri("urn:s"),
                Term::iri("urn:p"),
                Term::lang_literal("value", language),
            )
        };
        let default = RdfQuadVersion::new(
            sid(1),
            Quad::new(make("EN")),
            GraphIncarnationId::DEFAULT_GRAPH,
            EpochInterval::open(e(1)),
            None,
        )
        .unwrap();
        let named = RdfQuadVersion::new(
            sid(1),
            Quad::named(make("en"), "urn:g"),
            GraphIncarnationId::new(1),
            EpochInterval::open(e(1)),
            None,
        )
        .unwrap();
        let alias = RdfQuadVersion::new(
            sid(1),
            Quad::named(make("EN"), "urn:g"),
            GraphIncarnationId::new(1),
            EpochInterval::open(e(2)),
            None,
        )
        .unwrap();
        let graph = RdfGraphLife::new(
            RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(1)).unwrap(),
            EpochInterval::open(e(1)),
        )
        .unwrap();
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![graph.clone()],
            vec![default.clone(), named.clone()],
        )
        .unwrap();
        assert_eq!(history.cut(e(2)).unwrap().quads.len(), 2);
        assert!(matches!(
            RdfDatasetHistory::new(
                sid(1),
                HistoryCompleteness::Complete,
                vec![graph],
                vec![default, named, alias]
            ),
            Err(RdfHistoryError::OverlappingStatementLives(_))
        ));
    }

    #[test]
    fn handles_are_full_typed_graph_incarnation_identities() {
        let default = Quad::new(triple("urn:s"));
        let named = Quad::named(triple("urn:s"), "urn:g");
        let h_default =
            statement_handle(sid(1), &default, GraphIncarnationId::DEFAULT_GRAPH).unwrap();
        let h_named_1 = statement_handle(sid(1), &named, GraphIncarnationId::new(1)).unwrap();
        let h_named_2 = statement_handle(sid(1), &named, GraphIncarnationId::new(2)).unwrap();
        assert_ne!(h_default, h_named_1);
        assert_ne!(h_named_1, h_named_2);
        assert_eq!(h_default.as_bytes().len(), 32);

        let plain = Quad::new(super::super::Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::literal("café"),
        ));
        assert_ne!(
            h_default,
            statement_handle(sid(1), &plain, GraphIncarnationId::DEFAULT_GRAPH).unwrap()
        );
        assert_ne!(
            h_default,
            statement_handle(sid(2), &default, GraphIncarnationId::DEFAULT_GRAPH).unwrap()
        );
    }

    #[test]
    fn recreate_retains_both_lifetimes_and_changes_handles() {
        let graph_1 = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(1)).unwrap();
        let graph_2 = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(2)).unwrap();
        let quad = Quad::named(triple("urn:s"), "urn:g");
        let version_1 = RdfQuadVersion::new(
            sid(1),
            quad.clone(),
            GraphIncarnationId::new(1),
            EpochInterval::closed(e(2), e(3)),
            None,
        )
        .unwrap();
        let version_2 = RdfQuadVersion::new(
            sid(1),
            quad,
            GraphIncarnationId::new(2),
            EpochInterval::open(e(5)),
            None,
        )
        .unwrap();
        assert_ne!(version_1.statement(), version_2.statement());

        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![
                RdfGraphLife::new(graph_1, EpochInterval::closed(e(1), e(3))).unwrap(),
                RdfGraphLife::new(graph_2, EpochInterval::open(e(4))).unwrap(),
            ],
            vec![version_1.clone(), version_2.clone()],
        )
        .unwrap();
        assert_eq!(
            history.cut(e(2)).unwrap().quads[0].statement,
            version_1.statement()
        );
        assert!(history.cut(e(3)).unwrap().quads.is_empty());
        assert_eq!(
            history.cut(e(5)).unwrap().quads[0].statement,
            version_2.statement()
        );
    }

    #[test]
    fn diff_is_atomic_epoch_ordered_and_graph_qualified() {
        let graph = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(1)).unwrap();
        let quad = Quad::named(triple("urn:s"), "urn:g");
        let version = RdfQuadVersion::new(
            sid(1),
            quad,
            GraphIncarnationId::new(1),
            EpochInterval::closed(e(2), e(3)),
            None,
        )
        .unwrap();
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![RdfGraphLife::new(graph, EpochInterval::closed(e(2), e(3))).unwrap()],
            vec![version],
        )
        .unwrap();

        let kinds: Vec<&'static str> = history
            .ordered_diff(e(1), e(3))
            .unwrap()
            .transitions
            .iter()
            .map(|transition| match transition.kind {
                RdfHistoryTransitionKind::GraphCreated => "create",
                RdfHistoryTransitionKind::StatementAsserted { .. } => "assert",
                RdfHistoryTransitionKind::StatementRetracted { .. } => "retract",
                RdfHistoryTransitionKind::GraphDropped => "drop",
            })
            .collect();
        assert_eq!(kinds, ["create", "assert", "retract", "drop"]);
    }

    #[test]
    fn same_epoch_drop_recreate_pages_old_lifetime_before_replacement() {
        let old_graph = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(1)).unwrap();
        let new_graph = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(2)).unwrap();
        let old_version = RdfQuadVersion::new(
            sid(1),
            Quad::named(triple("urn:old"), "urn:g"),
            GraphIncarnationId::new(1),
            EpochInterval::closed(e(2), e(3)),
            None,
        )
        .unwrap();
        let new_version = RdfQuadVersion::new(
            sid(1),
            Quad::named(triple("urn:new"), "urn:g"),
            GraphIncarnationId::new(2),
            EpochInterval::open(e(3)),
            None,
        )
        .unwrap();
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![
                RdfGraphLife::new(old_graph, EpochInterval::closed(e(1), e(3))).unwrap(),
                RdfGraphLife::new(new_graph, EpochInterval::open(e(3))).unwrap(),
            ],
            vec![old_version, new_version],
        )
        .unwrap();

        let expected = ["retract", "drop", "create", "assert"];
        let diff = history.ordered_diff(e(2), e(3)).unwrap();
        let kinds: Vec<_> = diff
            .transitions
            .iter()
            .map(|transition| match transition.kind {
                RdfHistoryTransitionKind::StatementRetracted { .. } => "retract",
                RdfHistoryTransitionKind::GraphDropped => "drop",
                RdfHistoryTransitionKind::GraphCreated => "create",
                RdfHistoryTransitionKind::StatementAsserted { .. } => "assert",
            })
            .collect();
        assert_eq!(kinds, expected);

        let mut cursor = None;
        let mut paged = Vec::new();
        loop {
            let page = history.cdc_page(e(2), e(3), cursor.as_ref(), 1).unwrap();
            assert_eq!(page.transitions.len(), 1);
            paged.push(page.transitions[0].clone());
            cursor = page.next_cursor;
            if !page.has_more {
                break;
            }
        }
        assert_eq!(paged, diff.transitions);
        let terminal = cursor.unwrap();
        assert_eq!(
            terminal
                .to_token()
                .unwrap()
                .parse::<RdfHistoryCursor>()
                .unwrap(),
            terminal
        );
    }

    #[test]
    fn cdc_cursor_pages_are_stable_and_round_trip() {
        let versions: Vec<RdfQuadVersion> = (0..5)
            .map(|index| {
                RdfQuadVersion::new(
                    sid(1),
                    Quad::new(triple(&format!("urn:s:{index}"))),
                    GraphIncarnationId::DEFAULT_GRAPH,
                    EpochInterval::open(e(2)),
                    None,
                )
                .unwrap()
            })
            .collect();
        let history =
            RdfDatasetHistory::new(sid(1), HistoryCompleteness::Complete, Vec::new(), versions)
                .unwrap();
        let first = history.cdc_page(e(1), e(2), None, 2).unwrap();
        assert_eq!(first.transitions.len(), 2);
        assert!(first.has_more);
        let cursor = first.next_cursor.unwrap();
        let decoded: RdfHistoryCursor = cursor.to_token().unwrap().parse().unwrap();
        assert_eq!(decoded, cursor);

        let second = history.cdc_page(e(1), e(2), Some(&decoded), 2).unwrap();
        let third = history
            .cdc_page(e(1), e(2), second.next_cursor.as_ref(), 2)
            .unwrap();
        assert_eq!(second.transitions.len(), 2);
        assert_eq!(third.transitions.len(), 1);
        assert!(!third.has_more);
        assert!(
            third.next_cursor.is_some(),
            "exhausted pages remain resumable"
        );

        let mut handles = BTreeSet::new();
        for transition in first
            .transitions
            .into_iter()
            .chain(second.transitions)
            .chain(third.transitions)
        {
            assert!(handles.insert(transition.statement().unwrap()));
        }
        assert_eq!(handles.len(), 5);
    }

    #[test]
    fn legacy_current_state_never_claims_missing_history() {
        let completeness = HistoryCompleteness::LegacyCurrentState {
            observed_at: e(10),
            source_version: 4,
        };
        let history = RdfDatasetHistory::new(sid(1), completeness, Vec::new(), Vec::new()).unwrap();
        assert!(matches!(
            history.cut(e(9)),
            Err(RdfHistoryError::IncompleteHistory { .. })
        ));
        assert!(history.cut(e(10)).is_ok());
        assert!(matches!(
            history.ordered_diff(e(9), e(11)),
            Err(RdfHistoryError::IncompleteHistory { .. })
        ));
        assert!(history.ordered_diff(e(10), e(11)).is_ok());
    }

    #[test]
    fn graph_lifetime_must_contain_every_quad_version() {
        let graph = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(1)).unwrap();
        let version = RdfQuadVersion::new(
            sid(1),
            Quad::named(triple("urn:s"), "urn:g"),
            GraphIncarnationId::new(1),
            EpochInterval::open(e(2)),
            None,
        )
        .unwrap();
        let result = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![RdfGraphLife::new(graph, EpochInterval::closed(e(1), e(3))).unwrap()],
            vec![version],
        );
        assert!(matches!(
            result,
            Err(RdfHistoryError::QuadOutsideGraphLife { .. })
        ));
    }

    #[test]
    fn snapshot_identity_preserves_handles_and_explicit_fork_changes_them() {
        let original = RdfQuadVersion::new(
            sid(1),
            Quad::new(triple("urn:s")),
            GraphIncarnationId::DEFAULT_GRAPH,
            EpochInterval::open(e(2)),
            None,
        )
        .unwrap();
        let history = RdfDatasetHistory::new_with_high_water(
            sid(1),
            HistoryCompleteness::Complete,
            GraphIncarnationId::FIRST_NAMED,
            Vec::new(),
            vec![original.clone()],
        )
        .unwrap();

        let transferred = history.clone();
        let forked = history.reidentify_for_fork(sid(2)).unwrap();

        assert_eq!(transferred.store_id(), sid(1));
        assert_eq!(forked.store_id(), sid(2));
        assert_eq!(forked.completeness(), history.completeness());
        assert_eq!(
            forked.next_graph_incarnation(),
            history.next_graph_incarnation()
        );
        assert_eq!(forked.graph_lives(), history.graph_lives());
        assert_eq!(forked.quad_versions()[0].quad(), original.quad());
        assert_eq!(forked.quad_versions()[0].tx(), original.tx());
        assert_eq!(forked.quad_versions()[0].valid(), original.valid());
        assert_eq!(
            original.statement(),
            transferred.quad_versions()[0].statement()
        );
        assert_ne!(original.statement(), forked.quad_versions()[0].statement());
        assert_eq!(
            history.reidentify_for_fork(sid(1)).unwrap_err(),
            RdfHistoryError::ForkStoreIdentityUnchanged
        );
    }

    #[test]
    fn persisted_handles_and_lifecycle_dtos_are_revalidated() {
        let valid = RdfQuadVersion::new(
            sid(1),
            Quad::new(triple("urn:s")),
            GraphIncarnationId::DEFAULT_GRAPH,
            EpochInterval::open(e(2)),
            None,
        )
        .unwrap();
        let corrupt = RdfQuadVersion {
            statement: StatementHandle::from_bytes([0x55; 32]),
            ..valid
        };
        assert!(matches!(
            RdfDatasetHistory::new(
                sid(1),
                HistoryCompleteness::Complete,
                Vec::new(),
                vec![corrupt]
            ),
            Err(RdfHistoryError::StatementHandleMismatch { .. })
        ));

        let forged_default_life = RdfGraphLife {
            graph: RdfGraphIdentity::default_graph(),
            tx: EpochInterval::open(e(1)),
        };
        assert!(matches!(
            RdfDatasetHistory::new(
                sid(1),
                HistoryCompleteness::Complete,
                vec![forged_default_life],
                Vec::new()
            ),
            Err(RdfHistoryError::DefaultGraphHasLifecycle)
        ));
    }

    #[test]
    fn cdc_rejects_cross_store_and_forged_cursors() {
        let version = RdfQuadVersion::new(
            sid(1),
            Quad::new(triple("urn:s")),
            GraphIncarnationId::DEFAULT_GRAPH,
            EpochInterval::open(e(2)),
            None,
        )
        .unwrap();
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            Vec::new(),
            vec![version],
        )
        .unwrap();
        let real = history
            .cdc_page(e(1), e(2), None, 1)
            .unwrap()
            .next_cursor
            .unwrap();
        let cross_store = RdfHistoryCursor(TransitionKey {
            store_id: sid(2),
            ..real.0.clone()
        });
        assert!(matches!(
            history.cdc_page(e(1), e(2), Some(&cross_store), 1),
            Err(RdfHistoryError::CrossStoreCursor { .. })
        ));

        let forged = RdfHistoryCursor(TransitionKey {
            statement: StatementHandle::from_bytes([0x77; 32]),
            ..real.0
        });
        assert!(matches!(
            history.cdc_page(e(1), e(2), Some(&forged), 1),
            Err(RdfHistoryError::CursorNotFound)
        ));
    }

    #[test]
    fn composed_cut_applies_sub_micro_validity_to_historical_named_quad() {
        let graph = RdfGraphIdentity::named("urn:g", GraphIncarnationId::new(1)).unwrap();
        let valid = ValidTimeInterval::from_tai_nanoseconds(1_000, 1_002).unwrap();
        let version = RdfQuadVersion::new(
            sid(1),
            Quad::named(triple("urn:s"), "urn:g"),
            GraphIncarnationId::new(1),
            EpochInterval::closed(e(2), e(4)),
            Some(valid),
        )
        .unwrap();
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![RdfGraphLife::new(graph, EpochInterval::open(e(1))).unwrap()],
            vec![version],
        )
        .unwrap();

        let inside = history
            .cut_at(e(3), Some(TaiNanoseconds::new(1_001)))
            .unwrap();
        assert_eq!(inside.quads.len(), 1);
        assert_eq!(inside.quads[0].quad.graph(), Some("urn:g"));
        assert!(
            history
                .cut_at(e(3), Some(TaiNanoseconds::new(1_002)))
                .unwrap()
                .quads
                .is_empty()
        );
        assert!(history.cut_at(e(4), None).unwrap().quads.is_empty());
    }

    #[test]
    fn store_history_retains_drop_recreate_and_exact_restore() {
        let store = super::super::RdfStore::new();
        store.try_set_commit_epoch(e(1)).unwrap();
        assert!(store.create_graph("urn:g"));
        let first_graph = store.graph("urn:g").unwrap();
        store.try_set_commit_epoch(e(2)).unwrap();
        assert!(first_graph.insert(triple("urn:old")));

        store.try_set_commit_epoch(e(3)).unwrap();
        assert!(store.drop_graph("urn:g"));
        store.try_set_commit_epoch(e(4)).unwrap();
        assert!(store.create_graph("urn:g"));
        let second_graph = store.graph("urn:g").unwrap();
        assert!(second_graph.insert(triple("urn:new")));

        let history = store.dataset_history().unwrap();
        assert_eq!(history.graph_lives().len(), 2);
        let old_cut = history.cut(e(2)).unwrap();
        let new_cut = history.cut(e(4)).unwrap();
        assert_eq!(old_cut.quads.len(), 1);
        assert_eq!(new_cut.quads.len(), 1);
        assert_ne!(old_cut.quads[0].statement, new_cut.quads[0].statement);
        assert!(history.cut(e(3)).unwrap().quads.is_empty());

        let source_id = history.store_id();
        let source_diff = history.ordered_diff(e(0), e(4)).unwrap();
        let restored = super::super::RdfStore::new();
        restored
            .replace_dataset_history_exact(history, e(4))
            .unwrap();
        assert_eq!(restored.store_id(), source_id);
        let restored_history = restored.dataset_history().unwrap();
        assert_eq!(
            restored_history.ordered_diff(e(0), e(4)).unwrap(),
            source_diff
        );
        assert_eq!(restored_history.cut(e(2)).unwrap(), old_cut);
        assert_eq!(restored_history.cut(e(4)).unwrap(), new_cut);

        store.try_set_commit_epoch(e(5)).unwrap();
        assert!(store.drop_graph("urn:g"));
        store.try_set_commit_epoch(e(6)).unwrap();
        assert!(store.create_graph("urn:g"));
        let latest = store.graph("urn:g").unwrap();
        assert_eq!(latest.graph_incarnation(), GraphIncarnationId::new(3));
    }

    #[test]
    fn same_epoch_insert_delete_and_create_drop_leave_no_false_history() {
        let store = super::super::RdfStore::new();
        store.try_set_commit_epoch(e(7)).unwrap();
        let statement = triple("urn:transient");
        assert!(store.insert(statement.clone()));
        assert!(store.remove(&statement));
        assert!(store.create_graph("urn:transient-graph"));
        let graph = store.graph("urn:transient-graph").unwrap();
        assert!(graph.insert(statement));
        assert!(store.drop_graph("urn:transient-graph"));

        let history = store.dataset_history().unwrap();
        assert!(history.graph_lives().is_empty());
        assert!(history.quad_versions().is_empty());
        assert!(
            history
                .ordered_diff(e(6), e(7))
                .unwrap()
                .transitions
                .is_empty()
        );
    }

    #[test]
    fn constructor_canonicalizes_equivalent_history_bytes() {
        let graph_a = RdfGraphLife::new(
            RdfGraphIdentity::named("urn:a", GraphIncarnationId::new(1)).unwrap(),
            EpochInterval::open(e(1)),
        )
        .unwrap();
        let graph_b = RdfGraphLife::new(
            RdfGraphIdentity::named("urn:b", GraphIncarnationId::new(2)).unwrap(),
            EpochInterval::open(e(1)),
        )
        .unwrap();
        let quad_a = RdfQuadVersion::new(
            sid(1),
            Quad::named(triple("urn:s:a"), "urn:a"),
            GraphIncarnationId::new(1),
            EpochInterval::open(e(2)),
            None,
        )
        .unwrap();
        let quad_b = RdfQuadVersion::new(
            sid(1),
            Quad::named(triple("urn:s:b"), "urn:b"),
            GraphIncarnationId::new(2),
            EpochInterval::open(e(3)),
            None,
        )
        .unwrap();

        let canonical = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![graph_a.clone(), graph_b.clone()],
            vec![quad_a.clone(), quad_b.clone()],
        )
        .unwrap();
        let permuted = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            vec![graph_b, graph_a],
            vec![quad_b, quad_a],
        )
        .unwrap();

        let encode = |history: &RdfDatasetHistory| {
            bincode::serde::encode_to_vec(
                (history.graph_lives(), history.quad_versions()),
                bincode::config::standard(),
            )
            .unwrap()
        };
        assert_eq!(encode(&canonical), encode(&permuted));
        assert_eq!(
            canonical.ordered_diff(e(0), e(3)).unwrap(),
            permuted.ordered_diff(e(0), e(3)).unwrap()
        );
    }

    #[test]
    fn pending_epoch_is_rejected_as_a_non_reproducible_cut() {
        let history = RdfDatasetHistory::new(
            sid(1),
            HistoryCompleteness::Complete,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            history.cut(EpochId::PENDING),
            Err(RdfHistoryError::PendingEpoch)
        );
        assert_eq!(
            history.ordered_diff(e(0), EpochId::PENDING),
            Err(RdfHistoryError::PendingEpoch)
        );
        assert_eq!(
            history.cdc_page(e(0), EpochId::PENDING, None, 1),
            Err(RdfHistoryError::PendingEpoch)
        );
    }

    #[test]
    fn cursor_rejects_oversized_token_before_hex_decode() {
        // Invalid hex sits after the bound. The length check must win so hostile
        // input cannot force proportional decoding or allocation.
        let token = format!("{CURSOR_PREFIX}{}z", "0".repeat(MAX_CURSOR_TOKEN_BYTES));
        assert_eq!(
            token.parse::<RdfHistoryCursor>(),
            Err(RdfHistoryCursorError::TokenTooLong {
                bytes: token.len(),
                maximum: MAX_CURSOR_TOKEN_BYTES,
            })
        );
    }

    #[test]
    fn graph_identity_rejects_control_characters() {
        assert!(matches!(
            RdfGraphIdentity::named("urn:bad\u{1f}", GraphIncarnationId::new(1)),
            Err(RdfHistoryError::InvalidGraphIdentity { .. })
        ));
    }
}
