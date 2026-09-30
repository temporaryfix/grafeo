//! Schema metadata - what labels, properties, and indexes exist.
//!
//! The catalog is the "dictionary" of your database. When you write `(:Person)`,
//! the catalog maps "Person" to an internal LabelId. This indirection keeps
//! storage compact while names stay readable.
//!
//! | What it tracks | Why it matters |
//! | -------------- | -------------- |
//! | Labels | Maps "Person" → LabelId for efficient storage |
//! | Property keys | Maps "name" → PropertyKeyId |
//! | Edge types | Maps "KNOWS" → EdgeTypeId |
//! | Indexes | Which properties are indexed for fast lookups |

mod check_eval;
mod current_wire;
mod index_owner;
#[cfg(all(feature = "lpg", feature = "wal"))]
mod owner_wal;
mod prepared;
mod state;

#[cfg(any(
    feature = "lpg",
    feature = "triple-store",
    feature = "grafeo-file",
    test
))]
pub(crate) use current_wire::CURRENT_CATALOG_STATE_VERSION;
#[cfg(any(feature = "lpg", feature = "triple-store"))]
pub(crate) use current_wire::CatalogStateV2;
#[cfg(any(feature = "lpg", feature = "triple-store"))]
pub(crate) use current_wire::decode_bounded as decode_current_catalog_bounded;
pub use index_owner::{ANONYMOUS_INDEX_PREFIX, IndexConfiguration};
#[cfg(all(feature = "lpg", feature = "wal"))]
pub(crate) use owner_wal::{IndexOwnerBatch, IndexOwnerChange, IndexOwnerImage};

#[cfg(any(feature = "lpg", feature = "triple-store"))]
pub(crate) use prepared::CatalogRead;
#[cfg(any(
    feature = "lpg",
    feature = "triple-store",
    feature = "grafeo-file",
    test
))]
pub(crate) use prepared::CatalogWorkspace;
#[cfg(all(test, feature = "lpg", feature = "gql"))]
pub(crate) use prepared::install_preparation_rendezvous;
#[cfg(all(test, feature = "lpg"))]
pub(crate) use prepared::state_copy_count;
#[cfg(feature = "lpg")]
pub(crate) use prepared::{CatalogReadGuard, ReadyCatalog};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use state::CatalogStateLock;

use grafeo_common::types::{
    EdgeTypeId, EpochId, GraphPath, IndexId, LabelId, NodeId, PropertyKey, PropertyKeyId,
    TransactionId, Value,
};
use grafeo_core::graph::lpg::PhysicalIndexKey;

/// The database's schema dictionary - maps names to compact internal IDs.
///
/// You rarely interact with this directly. The query processor uses it to
/// resolve names like "Person" and "name" to internal IDs.
pub struct Catalog {
    state: CatalogStateLock,
}

pub(crate) struct CatalogState {
    /// Label name-to-ID mappings.
    labels: LabelCatalog,
    /// Property key name-to-ID mappings.
    property_keys: PropertyCatalog,
    /// Edge type name-to-ID mappings.
    edge_types: EdgeTypeCatalog,
    /// Index definitions.
    indexes: IndexCatalog,
    /// Optional schema constraints.
    schema: Option<SchemaState>,
    #[cfg(test)]
    retirement_probe: Option<prepared::CatalogRetirementProbe>,
}

impl Clone for CatalogState {
    fn clone(&self) -> Self {
        #[cfg(test)]
        prepared::record_state_copy();
        Self {
            labels: self.labels.clone(),
            property_keys: self.property_keys.clone(),
            edge_types: self.edge_types.clone(),
            indexes: self.indexes.clone(),
            schema: self.schema.clone(),
            #[cfg(test)]
            retirement_probe: self.retirement_probe.clone(),
        }
    }
}

/// Current wire projection, not a complete runtime rollback/preparation image.
/// Index definitions have a separate current-format payload; prepared runtime
/// state includes them together with these dictionary and schema fields.
pub(crate) struct CatalogStateSnapshot {
    labels: Vec<Arc<str>>,
    property_keys: Vec<Arc<str>>,
    edge_types: Vec<Arc<str>>,
    schema: Option<SchemaStateSnapshot>,
}

struct SchemaStateSnapshot {
    unique_constraints: HashSet<(LabelId, PropertyKeyId)>,
    required_properties: HashSet<(LabelId, PropertyKeyId)>,
    named_constraints: HashMap<String, NamedConstraintDefinition>,
    node_types: HashMap<String, NodeTypeDefinition>,
    edge_types: HashMap<String, EdgeTypeDefinition>,
    graph_types: HashMap<String, GraphTypeDefinition>,
    schemas: Vec<String>,
    graph_type_bindings: HashMap<GraphPath, String>,
    procedures: HashMap<String, ProcedureDefinition>,
}

/// Version-1 engine payload carried by `CatalogBatchV3`.
///
/// Maps and sets are represented as sorted vectors so identical catalog state
/// always produces identical bytes. Keeping this DTO separate from the live
/// lock-backed catalog keeps locking details out of the current WAL bytes.
#[derive(serde::Serialize, serde::Deserialize)]
struct CatalogWalStateV1 {
    labels: Vec<String>,
    property_keys: Vec<String>,
    edge_types: Vec<String>,
    schema: SchemaWalStateV1,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SchemaWalStateV1 {
    unique_constraints: Vec<(u32, u32)>,
    required_properties: Vec<(u32, u32)>,
    named_constraints: Vec<NamedConstraintWalV1>,
    node_types: Vec<NodeTypeWalV1>,
    edge_types: Vec<EdgeTypeWalV1>,
    graph_types: Vec<GraphTypeWalV1>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<(String, String)>,
    procedures: Vec<ProcedureWalV1>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TypedPropertyWalV1 {
    name: String,
    data_type: String,
    nullable: bool,
    default_value: Option<Value>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TypeConstraintWalV1 {
    kind: u8,
    properties: Vec<String>,
    name: Option<String>,
    expression: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct NamedConstraintWalV1 {
    name: String,
    label: String,
    properties: Vec<String>,
    kind: u8,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct NodeTypeWalV1 {
    name: String,
    properties: Vec<TypedPropertyWalV1>,
    constraints: Vec<TypeConstraintWalV1>,
    parent_types: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct EdgeTypeWalV1 {
    name: String,
    properties: Vec<TypedPropertyWalV1>,
    constraints: Vec<TypeConstraintWalV1>,
    source_node_types: Vec<String>,
    target_node_types: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GraphTypeWalV1 {
    name: String,
    allowed_node_types: Vec<String>,
    allowed_edge_types: Vec<String>,
    open: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ProcedureWalV1 {
    name: String,
    params: Vec<(String, String)>,
    returns: Vec<(String, String)>,
    body: String,
}

impl Catalog {
    /// Creates an empty catalog with schema support enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: CatalogStateLock::new(CatalogState::new()),
        }
    }

    /// Creates an empty catalog with schema support enabled.
    #[must_use]
    pub fn with_schema() -> Self {
        Self::new()
    }

    /// Captures the dictionary/schema wire projection, excluding index owners.
    #[cfg(test)]
    pub(crate) fn snapshot_state(&self) -> CatalogStateSnapshot {
        self.state.read().snapshot_state()
    }

    /// Encodes the deterministic dictionary/schema WAL projection. Complete
    /// runtime preparation also carries logical indexes and checked counters.
    #[cfg(any(all(feature = "lpg", feature = "wal", feature = "gql"), test))]
    pub(crate) fn encode_wal_state_v1(&self) -> Result<Vec<u8>, String> {
        self.state.read().encode_wal_state_v1()
    }

    /// Canonical internal DDL comparison, including native binding identities.
    /// This is not a durable payload: V1 persistence still rejects root/nested bindings.
    #[cfg(any(all(feature = "lpg", feature = "gql"), test))]
    pub(crate) fn ddl_comparison_state(
        &self,
    ) -> Result<(Vec<u8>, Vec<(GraphPath, String)>), String> {
        self.state.read().ddl_comparison_state()
    }

    // === Label Operations ===

    /// Gets or creates a label ID for the given label name.
    ///
    /// # Errors
    /// Returns an exhaustion or reservation error before admission.
    pub fn get_or_create_label(&self, name: &str) -> Result<LabelId, CatalogError> {
        if let Some(id) = self.state.read().get_label_id(name) {
            return Ok(id);
        }
        self.state.write().get_or_create_label(name)
    }

    /// Gets the label ID for a label name, if it exists.
    #[must_use]
    pub fn get_label_id(&self, name: &str) -> Option<LabelId> {
        self.state.read().get_label_id(name)
    }

    /// Gets the label name for a label ID, if it exists.
    #[must_use]
    pub fn get_label_name(&self, id: LabelId) -> Option<Arc<str>> {
        self.state.read().get_label_name(id)
    }

    /// Returns the number of distinct labels.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.state.read().label_count()
    }

    /// Returns all label names.
    #[must_use]
    pub fn all_labels(&self) -> Vec<Arc<str>> {
        self.state.read().all_labels()
    }

    // === Property Key Operations ===

    /// Gets or creates a property key ID for the given property key name.
    ///
    /// # Errors
    /// Returns an exhaustion or reservation error before admission.
    pub fn get_or_create_property_key(&self, name: &str) -> Result<PropertyKeyId, CatalogError> {
        if let Some(id) = self.state.read().get_property_key_id(name) {
            return Ok(id);
        }
        self.state.write().get_or_create_property_key(name)
    }

    /// Gets the property key ID for a property key name, if it exists.
    #[must_use]
    pub fn get_property_key_id(&self, name: &str) -> Option<PropertyKeyId> {
        self.state.read().get_property_key_id(name)
    }

    /// Gets the property key name for a property key ID, if it exists.
    #[must_use]
    pub fn get_property_key_name(&self, id: PropertyKeyId) -> Option<Arc<str>> {
        self.state.read().get_property_key_name(id)
    }

    /// Returns the number of distinct property keys.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        self.state.read().property_key_count()
    }

    /// Returns all property key names.
    #[must_use]
    pub fn all_property_keys(&self) -> Vec<Arc<str>> {
        self.state.read().all_property_keys()
    }

    // === Edge Type Operations ===

    /// Gets or creates an edge type ID for the given edge type name.
    ///
    /// # Errors
    /// Returns an exhaustion or reservation error before admission.
    pub fn get_or_create_edge_type(&self, name: &str) -> Result<EdgeTypeId, CatalogError> {
        if let Some(id) = self.state.read().get_edge_type_id(name) {
            return Ok(id);
        }
        self.state.write().get_or_create_edge_type(name)
    }

    /// Gets the edge type ID for an edge type name, if it exists.
    #[must_use]
    pub fn get_edge_type_id(&self, name: &str) -> Option<EdgeTypeId> {
        self.state.read().get_edge_type_id(name)
    }

    /// Gets the edge type name for an edge type ID, if it exists.
    #[must_use]
    pub fn get_edge_type_name(&self, id: EdgeTypeId) -> Option<Arc<str>> {
        self.state.read().get_edge_type_name(id)
    }

    /// Returns the number of distinct edge types.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.state.read().edge_type_count()
    }

    /// Returns all edge type names.
    #[must_use]
    pub fn all_edge_types(&self) -> Vec<Arc<str>> {
        self.state.read().all_edge_types()
    }

    // === Index Operations ===

    /// Creates one canonical owner for an exact graph and resolved configuration.
    /// An absent name allocates a catalog-reserved anonymous name.
    ///
    /// # Errors
    /// Rejects invalid names/configurations/referents, duplicate logical or
    /// physical ownership, exhausted IDs, and failed capacity reservations.
    pub fn create_index(
        &self,
        name: Option<&str>,
        label: LabelId,
        property_key: PropertyKeyId,
        graph: GraphPath,
        configuration: IndexConfiguration,
    ) -> Result<IndexId, CatalogError> {
        self.state
            .write()
            .create_index(name, label, property_key, graph, configuration)
    }

    /// Drops an index by ID.
    pub fn drop_index(&self, id: IndexId) -> bool {
        self.state.write().drop_index(id)
    }

    /// Finds an index by its user-defined name.
    #[must_use]
    pub fn find_index_by_name(&self, name: &str) -> Option<IndexId> {
        self.state.read().find_index_by_name(name)
    }

    /// Gets the index definition for an index ID.
    #[must_use]
    pub fn get_index(&self, id: IndexId) -> Option<IndexDefinition> {
        self.state.read().get_index(id)
    }

    /// Returns the exact graph path. `None` means the owner does not exist.
    #[must_use]
    pub fn index_graph(&self, id: IndexId) -> Option<GraphPath> {
        self.state.read().index_graph(id)
    }

    /// Next owner identity, including allocations whose owners were dropped.
    #[must_use]
    pub fn index_allocator_high_water(&self) -> u32 {
        self.state.read().index_allocator_high_water()
    }

    /// Finds indexes for a given label.
    #[must_use]
    pub fn indexes_for_label(&self, label: LabelId) -> Vec<IndexId> {
        self.state.read().indexes_for_label(label)
    }

    /// Finds indexes for a given label and property key.
    #[must_use]
    pub fn indexes_for_label_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Vec<IndexId> {
        self.state
            .read()
            .indexes_for_label_property(label, property_key)
    }

    /// Returns all index definitions.
    #[must_use]
    pub fn all_indexes(&self) -> Vec<IndexDefinition> {
        self.state.read().all_indexes()
    }

    /// Returns the number of indexes.
    #[must_use]
    pub fn index_count(&self) -> usize {
        self.state.read().index_count()
    }

    // === Schema Operations ===

    /// Returns whether schema constraints are enabled.
    #[must_use]
    pub fn has_schema(&self) -> bool {
        self.state.read().has_schema()
    }

    /// Adds a uniqueness constraint.
    ///
    /// Returns an error if schema is not enabled or constraint already exists.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or a
    /// schema-specific error if the operation fails (e.g. duplicate constraint).
    pub fn add_unique_constraint(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .add_unique_constraint(label, property_key)
    }

    /// Adds a required property constraint (NOT NULL).
    ///
    /// Returns an error if schema is not enabled or constraint already exists.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or a
    /// schema-specific error if the operation fails (e.g. duplicate constraint).
    pub fn add_required_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .add_required_property(label, property_key)
    }

    /// Checks if a property is required for a label.
    #[must_use]
    pub fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.state.read().is_property_required(label, property_key)
    }

    /// Checks if a property must be unique for a label.
    #[must_use]
    pub fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.state.read().is_property_unique(label, property_key)
    }

    /// Returns physical uniqueness constraints translated back to stable
    /// dictionary names. Portable snapshot merge uses names because dictionary
    /// IDs are local to each source catalog.
    #[cfg(feature = "lpg")]
    pub(crate) fn all_unique_constraints(&self) -> Vec<(String, String)> {
        self.state.read().all_unique_constraints()
    }

    /// Returns physical required-property constraints translated to names.
    #[cfg(feature = "lpg")]
    pub(crate) fn all_required_properties(&self) -> Vec<(String, String)> {
        self.state.read().all_required_properties()
    }

    /// Creates a durable, user-visible constraint and installs its enforcement
    /// metadata as one catalog operation.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/duplicated property list, duplicate name
    /// or target, or when schema support is unavailable.
    pub fn create_named_constraint(
        &self,
        definition: NamedConstraintDefinition,
    ) -> Result<(), CatalogError> {
        self.state.write().create_named_constraint(definition)
    }

    /// Restores a named constraint from a trusted catalog snapshot.
    ///
    /// Unlike a fresh CREATE, an equivalent type-level constraint is expected:
    /// current snapshots persist both the type definition and its named owner.
    #[cfg(any(feature = "lpg", test))]
    pub(crate) fn restore_named_constraint(
        &self,
        definition: NamedConstraintDefinition,
    ) -> Result<(), CatalogError> {
        self.state.write().restore_named_constraint(definition)
    }

    /// Restores a named constraint from legacy delta WAL over a v1-v3 catalog
    /// snapshot that retained enforcement metadata but lost the user's owner
    /// name.
    ///
    /// Legacy snapshots synthesize reserved, content-derived owners. When the
    /// corresponding old `CreateConstraint` record is still present, recovery
    /// replaces those owners with the real WAL identity without removing and
    /// re-adding type or physical enforcement metadata. The operation is
    /// idempotent and never adopts a merely prefix-matching user name.
    #[cfg(any(all(feature = "lpg", feature = "wal"), test))]
    pub(crate) fn restore_named_constraint_from_wal(
        &self,
        definition: NamedConstraintDefinition,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .restore_named_constraint_from_wal(definition)
    }

    /// Drops a named constraint and removes the enforcement metadata owned by
    /// that definition.
    ///
    /// # Errors
    ///
    /// Returns `ConstraintNotFound` when the name is unknown.
    pub fn drop_named_constraint(
        &self,
        name: &str,
    ) -> Result<NamedConstraintDefinition, CatalogError> {
        self.state.write().drop_named_constraint(name)
    }

    /// Returns one named constraint definition.
    #[must_use]
    pub fn get_named_constraint(&self, name: &str) -> Option<NamedConstraintDefinition> {
        self.state.read().get_named_constraint(name)
    }

    /// Returns all named constraints in deterministic name order.
    #[must_use]
    pub fn all_named_constraints(&self) -> Vec<NamedConstraintDefinition> {
        self.state.read().all_named_constraints()
    }

    /// Whether this catalog still contains a synthetic owner adopted from a
    /// legacy checkpoint that did not persist user-visible constraint names.
    ///
    /// Recovery uses this only to reject an otherwise-unresolvable legacy
    /// `DropConstraint`: silently ignoring the missing name could leave the
    /// type-level enforcement active after the WAL says it was dropped.
    #[must_use]
    #[cfg(all(feature = "lpg", feature = "wal"))]
    pub(crate) fn has_legacy_constraint_owners(&self) -> bool {
        self.state.read().has_legacy_constraint_owners()
    }

    /// Whether an equivalent constraint target already has a named owner.
    #[must_use]
    pub fn has_equivalent_named_constraint(&self, definition: &NamedConstraintDefinition) -> bool {
        self.state
            .read()
            .has_equivalent_named_constraint(definition)
    }

    /// Registers a node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a type with the same name exists.
    pub fn register_node_type(&self, def: NodeTypeDefinition) -> Result<(), CatalogError> {
        self.state.write().register_node_type(def)
    }

    /// Registers or replaces a node type definition.
    pub fn register_or_replace_node_type(&self, def: NodeTypeDefinition) {
        self.state.write().register_or_replace_node_type(def);
    }

    /// Drops a node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the type does not exist.
    pub fn drop_node_type(&self, name: &str) -> Result<(), CatalogError> {
        self.state.write().drop_node_type(name)
    }

    /// Gets a node type definition by name.
    #[must_use]
    pub fn get_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.state.read().get_node_type(name)
    }

    /// Gets a resolved node type with inherited properties from parents.
    #[must_use]
    pub fn resolved_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.state.read().resolved_node_type(name)
    }

    /// Returns all registered node type names.
    #[must_use]
    pub fn all_node_type_names(&self) -> Vec<String> {
        self.state.read().all_node_type_names()
    }

    /// Returns all registered edge type definition names.
    #[must_use]
    pub fn all_edge_type_names(&self) -> Vec<String> {
        self.state.read().all_edge_type_names()
    }

    /// Registers an edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if an edge type with the same name exists.
    pub fn register_edge_type_def(&self, def: EdgeTypeDefinition) -> Result<(), CatalogError> {
        self.state.write().register_edge_type_def(def)
    }

    /// Registers or replaces an edge type definition.
    pub fn register_or_replace_edge_type_def(&self, def: EdgeTypeDefinition) {
        self.state.write().register_or_replace_edge_type_def(def);
    }

    /// Drops an edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    pub fn drop_edge_type_def(&self, name: &str) -> Result<(), CatalogError> {
        self.state.write().drop_edge_type_def(name)
    }

    /// Gets an edge type definition by name.
    #[must_use]
    pub fn get_edge_type_def(&self, name: &str) -> Option<EdgeTypeDefinition> {
        self.state.read().get_edge_type_def(name)
    }

    /// Registers a graph type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a graph type with the same name exists.
    pub fn register_graph_type(&self, def: GraphTypeDefinition) -> Result<(), CatalogError> {
        self.state.write().register_graph_type(def)
    }

    /// Registers or replaces a graph type while preserving graph bindings.
    pub fn register_or_replace_graph_type(&self, def: GraphTypeDefinition) {
        self.state.write().register_or_replace_graph_type(def);
    }

    /// Drops a graph type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn drop_graph_type(&self, name: &str) -> Result<(), CatalogError> {
        self.state.write().drop_graph_type(name)
    }

    /// Returns all registered graph type names.
    #[must_use]
    pub fn all_graph_type_names(&self) -> Vec<String> {
        self.state.read().all_graph_type_names()
    }

    /// Gets a graph type definition by name.
    #[must_use]
    pub fn get_graph_type_def(&self, name: &str) -> Option<GraphTypeDefinition> {
        self.state.read().get_graph_type_def(name)
    }

    /// Registers a schema namespace.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::SchemaAlreadyExists` if the namespace already exists.
    pub fn register_schema_namespace(&self, name: String) -> Result<(), CatalogError> {
        self.state.write().register_schema_namespace(name)
    }

    /// Drops a schema namespace.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::SchemaNotFound` if the namespace does not exist.
    pub fn drop_schema_namespace(&self, name: &str) -> Result<(), CatalogError> {
        self.state.write().drop_schema_namespace(name)
    }

    /// Checks whether a schema namespace exists.
    #[must_use]
    pub fn schema_exists(&self, name: &str) -> bool {
        self.state.read().schema_exists(name)
    }

    /// Returns all registered schema namespace names.
    #[must_use]
    pub fn schema_names(&self) -> Vec<String> {
        self.state.read().schema_names()
    }

    /// Adds a constraint to an existing node type, creating a minimal type if needed.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled.
    pub fn add_constraint_to_type(
        &self,
        label: &str,
        constraint: TypeConstraint,
    ) -> Result<(), CatalogError> {
        self.state.write().add_constraint_to_type(label, constraint)
    }

    /// Adds a property to a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub fn alter_node_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_node_type_add_property(type_name, property)
    }

    /// Drops a property from a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type or property does not exist.
    pub fn alter_node_type_drop_property(
        &self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_node_type_drop_property(type_name, property_name)
    }

    /// Adds a property to an edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub fn alter_edge_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_edge_type_add_property(type_name, property)
    }

    /// Drops a property from an edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type or property does not exist.
    pub fn alter_edge_type_drop_property(
        &self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_edge_type_drop_property(type_name, property_name)
    }

    /// Adds a node type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_node_type(
        &self,
        graph_type_name: &str,
        node_type: String,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_graph_type_add_node_type(graph_type_name, node_type)
    }

    /// Drops a node type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_node_type(
        &self,
        graph_type_name: &str,
        node_type: &str,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_graph_type_drop_node_type(graph_type_name, node_type)
    }

    /// Adds an edge type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_edge_type(
        &self,
        graph_type_name: &str,
        edge_type: String,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_graph_type_add_edge_type(graph_type_name, edge_type)
    }

    /// Drops an edge type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_edge_type(
        &self,
        graph_type_name: &str,
        edge_type: &str,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .alter_graph_type_drop_edge_type(graph_type_name, edge_type)
    }

    /// Binds a graph instance to a graph type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn bind_graph_type(
        &self,
        graph_path: &GraphPath,
        graph_type: String,
    ) -> Result<(), CatalogError> {
        self.state.write().bind_graph_type(graph_path, graph_type)
    }

    /// Validates that a graph type exists without publishing a binding.
    ///
    /// Transactional graph creation uses this before its durable marker, then
    /// publishes the already-validated binding together with graph lifecycle.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::SchemaNotEnabled`] when schema support is
    /// disabled, or [`CatalogError::TypeNotFound`] when `graph_type` is not
    /// defined in the catalog.
    pub fn validate_graph_type_binding_target(&self, graph_type: &str) -> Result<(), CatalogError> {
        self.state
            .read()
            .validate_graph_type_binding_target(graph_type)
    }

    /// Returns whether a graph's binding is still the exact expected value.
    #[must_use]
    pub fn graph_type_binding_matches(
        &self,
        graph_path: &GraphPath,
        expected: Option<&str>,
    ) -> bool {
        self.state
            .read()
            .graph_type_binding_matches(graph_path, expected)
    }

    /// Atomically replaces or removes a pre-validated graph type binding if it
    /// still matches.
    ///
    /// Returns `false` on a compare-and-swap mismatch and leaves the catalog
    /// untouched. The caller must validate any replacement with
    /// [`Self::validate_graph_type_binding_target`] before entering a durable
    /// publication phase; this operation is deliberately infallible so commit
    /// publication contains no catalog validation after its WAL marker.
    #[must_use]
    pub fn publish_graph_type_binding_if_same(
        &self,
        graph_path: &GraphPath,
        expected: Option<&str>,
        replacement: Option<String>,
    ) -> bool {
        self.state
            .write()
            .publish_graph_type_binding_if_same(graph_path, expected, replacement)
    }

    /// Gets the graph type binding for a graph instance.
    pub fn get_graph_type_binding(&self, graph_path: &GraphPath) -> Option<String> {
        self.state.read().get_graph_type_binding(graph_path)
    }

    /// Replays a durable graph-type binding exactly during database recovery.
    ///
    /// Unlike the compare-and-swap publication path, recovery is the sole
    /// catalog writer and must deterministically reproduce the committed WAL
    /// stream over any older checkpoint. A referenced graph type must already
    /// exist in that recovered catalog cut.
    #[cfg(all(feature = "lpg", feature = "wal"))]
    pub(crate) fn replay_graph_type_binding(
        &self,
        graph_path: &GraphPath,
        graph_type: Option<String>,
    ) -> Result<(), CatalogError> {
        self.state
            .write()
            .replay_graph_type_binding(graph_path, graph_type)
    }

    /// Registers a stored procedure.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a procedure with the same name exists.
    pub fn register_procedure(&self, def: ProcedureDefinition) -> Result<(), CatalogError> {
        self.state.write().register_procedure(def)
    }

    /// Replaces or creates a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled.
    pub fn replace_procedure(&self, def: ProcedureDefinition) -> Result<(), CatalogError> {
        self.state.write().replace_procedure(def)
    }

    /// Drops a stored procedure.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the procedure does not exist.
    pub fn drop_procedure(&self, name: &str) -> Result<(), CatalogError> {
        self.state.write().drop_procedure(name)
    }

    /// Gets a stored procedure by name.
    pub fn get_procedure(&self, name: &str) -> Option<ProcedureDefinition> {
        self.state.read().get_procedure(name)
    }

    /// Returns all registered node type definitions.
    #[must_use]
    pub fn all_node_type_defs(&self) -> Vec<NodeTypeDefinition> {
        self.state.read().all_node_type_defs()
    }

    /// Returns all registered edge type definitions.
    #[must_use]
    pub fn all_edge_type_defs(&self) -> Vec<EdgeTypeDefinition> {
        self.state.read().all_edge_type_defs()
    }

    /// Returns all registered graph type definitions.
    #[must_use]
    pub fn all_graph_type_defs(&self) -> Vec<GraphTypeDefinition> {
        self.state.read().all_graph_type_defs()
    }

    /// Returns all registered procedure definitions.
    #[must_use]
    pub fn all_procedure_defs(&self) -> Vec<ProcedureDefinition> {
        self.state.read().all_procedure_defs()
    }

    /// Returns all graph type bindings (graph_path, type_name).
    #[must_use]
    pub fn all_graph_type_bindings(&self) -> Vec<(GraphPath, String)> {
        self.state.read().all_graph_type_bindings()
    }
}

impl CatalogState {
    fn new() -> Self {
        Self {
            labels: LabelCatalog::new(),
            property_keys: PropertyCatalog::new(),
            edge_types: EdgeTypeCatalog::new(),
            indexes: IndexCatalog::new(),
            schema: Some(SchemaState::new()),
            #[cfg(test)]
            retirement_probe: None,
        }
    }

    /// Captures the dictionary/schema wire projection, excluding index owners.
    pub(crate) fn snapshot_state(&self) -> CatalogStateSnapshot {
        CatalogStateSnapshot {
            labels: self.labels.all_names(),
            property_keys: self.property_keys.all_names(),
            edge_types: self.edge_types.all_names(),
            schema: self.schema.as_ref().map(|schema| SchemaStateSnapshot {
                unique_constraints: schema.unique_constraints.clone(),
                required_properties: schema.required_properties.clone(),
                named_constraints: schema.named_constraints.clone(),
                node_types: schema.node_types.clone(),
                edge_types: schema.edge_types.clone(),
                graph_types: schema.graph_types.clone(),
                schemas: schema.schemas.clone(),
                graph_type_bindings: schema.graph_type_bindings.clone(),
                procedures: schema.procedures.clone(),
            }),
        }
    }

    /// Encodes the deterministic dictionary/schema WAL projection. Complete
    /// runtime preparation also carries logical indexes and checked counters.
    #[cfg(any(all(feature = "lpg", feature = "wal", feature = "gql"), test))]
    pub(crate) fn encode_wal_state_v1(&self) -> Result<Vec<u8>, String> {
        CatalogWalStateV1::encode_snapshot(self.snapshot_state())
    }

    /// Compares DDL state without imposing a wire format on native graph paths.
    #[cfg(any(all(feature = "lpg", feature = "gql"), test))]
    pub(crate) fn ddl_comparison_state(
        &self,
    ) -> Result<(Vec<u8>, Vec<(GraphPath, String)>), String> {
        let mut snapshot = self.snapshot_state();
        let schema = snapshot
            .schema
            .as_mut()
            .ok_or_else(|| "catalog DDL comparison requires schema support".to_string())?;
        let mut bindings: Vec<_> = std::mem::take(&mut schema.graph_type_bindings)
            .into_iter()
            .collect();
        bindings.sort_unstable();
        Ok((CatalogWalStateV1::encode_snapshot(snapshot)?, bindings))
    }

    // === Label Operations ===

    /// Gets or creates a label ID for the given label name.
    ///
    /// # Errors
    /// Returns an exhaustion or reservation error before admission.
    pub(crate) fn get_or_create_label(&mut self, name: &str) -> Result<LabelId, CatalogError> {
        self.labels.get_or_create(name)
    }

    /// Gets the label ID for a label name, if it exists.
    #[must_use]
    pub(crate) fn get_label_id(&self, name: &str) -> Option<LabelId> {
        self.labels.get_id(name)
    }

    /// Gets the label name for a label ID, if it exists.
    #[must_use]
    pub(crate) fn get_label_name(&self, id: LabelId) -> Option<Arc<str>> {
        self.labels.get_name(id)
    }

    /// Returns the number of distinct labels.
    #[must_use]
    pub(crate) fn label_count(&self) -> usize {
        self.labels.count()
    }

    /// Returns all label names.
    #[must_use]
    pub(crate) fn all_labels(&self) -> Vec<Arc<str>> {
        self.labels.all_names()
    }

    // === Property Key Operations ===

    /// Gets or creates a property key ID for the given property key name.
    ///
    /// # Errors
    /// Returns an exhaustion or reservation error before admission.
    pub(crate) fn get_or_create_property_key(
        &mut self,
        name: &str,
    ) -> Result<PropertyKeyId, CatalogError> {
        self.property_keys.get_or_create(name)
    }

    /// Gets the property key ID for a property key name, if it exists.
    #[must_use]
    pub(crate) fn get_property_key_id(&self, name: &str) -> Option<PropertyKeyId> {
        self.property_keys.get_id(name)
    }

    /// Gets the property key name for a property key ID, if it exists.
    #[must_use]
    pub(crate) fn get_property_key_name(&self, id: PropertyKeyId) -> Option<Arc<str>> {
        self.property_keys.get_name(id)
    }

    /// Returns the number of distinct property keys.
    #[must_use]
    pub(crate) fn property_key_count(&self) -> usize {
        self.property_keys.count()
    }

    /// Returns all property key names.
    #[must_use]
    pub(crate) fn all_property_keys(&self) -> Vec<Arc<str>> {
        self.property_keys.all_names()
    }

    // === Edge Type Operations ===

    /// Gets or creates an edge type ID for the given edge type name.
    ///
    /// # Errors
    /// Returns an exhaustion or reservation error before admission.
    pub(crate) fn get_or_create_edge_type(
        &mut self,
        name: &str,
    ) -> Result<EdgeTypeId, CatalogError> {
        self.edge_types.get_or_create(name)
    }

    /// Gets the edge type ID for an edge type name, if it exists.
    #[must_use]
    pub(crate) fn get_edge_type_id(&self, name: &str) -> Option<EdgeTypeId> {
        self.edge_types.get_id(name)
    }

    /// Gets the edge type name for an edge type ID, if it exists.
    #[must_use]
    pub(crate) fn get_edge_type_name(&self, id: EdgeTypeId) -> Option<Arc<str>> {
        self.edge_types.get_name(id)
    }

    /// Returns the number of distinct edge types.
    #[must_use]
    pub(crate) fn edge_type_count(&self) -> usize {
        self.edge_types.count()
    }

    /// Returns all edge type names.
    #[must_use]
    pub(crate) fn all_edge_types(&self) -> Vec<Arc<str>> {
        self.edge_types.all_names()
    }

    // === Index Operations ===

    /// Creates one owner from an exact graph and resolved configuration.
    ///
    /// # Errors
    /// Rejects invalid referents/configuration, duplicate ownership, exhausted
    /// IDs, and failed reservations before logical admission.
    pub(crate) fn create_index(
        &mut self,
        name: Option<&str>,
        label: LabelId,
        property_key: PropertyKeyId,
        graph: GraphPath,
        configuration: IndexConfiguration,
    ) -> Result<IndexId, CatalogError> {
        let label_name = self
            .labels
            .get_name(label)
            .ok_or_else(|| CatalogError::LabelNotFound(label.to_string()))?;
        let property_name = self
            .property_keys
            .get_name(property_key)
            .ok_or_else(|| CatalogError::PropertyKeyNotFound(property_key.to_string()))?;
        configuration.validate()?;
        let key = configuration.physical_key(graph, &label_name, &property_name);
        self.indexes
            .create(name, label, property_key, key, configuration)
    }

    /// Drops an index by ID.
    pub(crate) fn drop_index(&mut self, id: IndexId) -> bool {
        self.indexes.drop(id)
    }

    /// Finds an index by its user-defined name.
    #[must_use]
    pub(crate) fn find_index_by_name(&self, name: &str) -> Option<IndexId> {
        self.indexes.find_by_name(name)
    }

    /// Gets the index definition for an index ID.
    #[must_use]
    pub(crate) fn get_index(&self, id: IndexId) -> Option<IndexDefinition> {
        self.indexes.get(id)
    }

    /// Returns the exact graph path. `None` means the owner does not exist.
    #[must_use]
    pub(crate) fn index_graph(&self, id: IndexId) -> Option<GraphPath> {
        self.indexes.graph(id)
    }

    /// Next owner identity, retaining every committed allocator gap.
    #[must_use]
    pub(crate) fn index_allocator_high_water(&self) -> u32 {
        self.indexes.next_id
    }

    /// Finds indexes for a given label.
    #[must_use]
    pub(crate) fn indexes_for_label(&self, label: LabelId) -> Vec<IndexId> {
        self.indexes.for_label(label)
    }

    /// Finds indexes for a given label and property key.
    #[must_use]
    pub(crate) fn indexes_for_label_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Vec<IndexId> {
        self.indexes.for_label_property(label, property_key)
    }

    // SHOW borrows the held catalog cut; collection/formatting admission belongs
    // to the result producer, before any output allocation.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_indexes(&self) -> impl Iterator<Item = &IndexDefinition> {
        self.indexes.indexes.values()
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_constraints(&self) -> impl Iterator<Item = &NamedConstraintDefinition> {
        self.schema
            .iter()
            .flat_map(|schema| schema.named_constraints.values())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_node_types(&self) -> impl Iterator<Item = (&str, &NodeTypeDefinition)> {
        self.schema.iter().flat_map(|schema| {
            schema
                .node_types
                .iter()
                .map(|(name, def)| (name.as_str(), def))
        })
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_edge_types(&self) -> impl Iterator<Item = (&str, &EdgeTypeDefinition)> {
        self.schema.iter().flat_map(|schema| {
            schema
                .edge_types
                .iter()
                .map(|(name, def)| (name.as_str(), def))
        })
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_graph_types(&self) -> impl Iterator<Item = (&str, &GraphTypeDefinition)> {
        self.schema.iter().flat_map(|schema| {
            schema
                .graph_types
                .iter()
                .map(|(name, def)| (name.as_str(), def))
        })
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_schemas(&self) -> impl Iterator<Item = &str> {
        self.schema
            .iter()
            .flat_map(|schema| schema.schemas.iter().map(String::as_str))
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn show_graph_type_binding(&self, path: &GraphPath) -> Option<&str> {
        self.schema
            .as_ref()?
            .graph_type_bindings
            .get(path)
            .map(String::as_str)
    }

    /// Returns all index definitions.
    #[must_use]
    pub(crate) fn all_indexes(&self) -> Vec<IndexDefinition> {
        self.indexes.all()
    }

    /// Returns the number of indexes.
    #[must_use]
    pub(crate) fn index_count(&self) -> usize {
        self.indexes.count()
    }

    // === Schema Operations ===

    /// Returns whether schema constraints are enabled.
    #[must_use]
    pub(crate) fn has_schema(&self) -> bool {
        self.schema.is_some()
    }

    /// Adds a uniqueness constraint.
    ///
    /// Returns an error if schema is not enabled or constraint already exists.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or a
    /// schema-specific error if the operation fails (e.g. duplicate constraint).
    pub(crate) fn add_unique_constraint(
        &mut self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.add_unique_constraint(label, property_key),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a required property constraint (NOT NULL).
    ///
    /// Returns an error if schema is not enabled or constraint already exists.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or a
    /// schema-specific error if the operation fails (e.g. duplicate constraint).
    pub(crate) fn add_required_property(
        &mut self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.add_required_property(label, property_key),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Checks if a property is required for a label.
    #[must_use]
    pub(crate) fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.schema
            .as_ref()
            .is_some_and(|s| s.is_property_required(label, property_key))
    }

    /// Checks if a property must be unique for a label.
    #[must_use]
    pub(crate) fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.schema
            .as_ref()
            .is_some_and(|s| s.is_property_unique(label, property_key))
    }

    /// Returns physical uniqueness constraints translated back to stable
    /// dictionary names. Portable snapshot merge uses names because dictionary
    /// IDs are local to each source catalog.
    #[cfg(feature = "lpg")]
    pub(crate) fn all_unique_constraints(&self) -> Vec<(String, String)> {
        let Some(schema) = &self.schema else {
            return Vec::new();
        };
        let mut constraints: Vec<_> = schema
            .unique_constraints
            .iter()
            .filter_map(|(label, property)| {
                Some((
                    self.get_label_name(*label)?.to_string(),
                    self.get_property_key_name(*property)?.to_string(),
                ))
            })
            .collect();
        constraints.sort();
        constraints
    }

    /// Returns physical required-property constraints translated to names.
    #[cfg(feature = "lpg")]
    pub(crate) fn all_required_properties(&self) -> Vec<(String, String)> {
        let Some(schema) = &self.schema else {
            return Vec::new();
        };
        let mut constraints: Vec<_> = schema
            .required_properties
            .iter()
            .filter_map(|(label, property)| {
                Some((
                    self.get_label_name(*label)?.to_string(),
                    self.get_property_key_name(*property)?.to_string(),
                ))
            })
            .collect();
        constraints.sort();
        constraints
    }

    /// Creates a durable, user-visible constraint and installs its enforcement
    /// metadata as one catalog operation.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/duplicated property list, duplicate name
    /// or target, or when schema support is unavailable.
    pub(crate) fn create_named_constraint(
        &mut self,
        definition: NamedConstraintDefinition,
    ) -> Result<(), CatalogError> {
        self.install_named_constraint(definition, false)
    }

    /// Restores a named constraint from a trusted catalog snapshot.
    ///
    /// Unlike a fresh CREATE, an equivalent type-level constraint is expected:
    /// current snapshots persist both the type definition and its named owner.
    #[cfg(any(feature = "lpg", test))]
    pub(crate) fn restore_named_constraint(
        &mut self,
        definition: NamedConstraintDefinition,
    ) -> Result<(), CatalogError> {
        self.install_named_constraint(definition, true)
    }

    /// Restores a named constraint from legacy delta WAL over a v1-v3 catalog
    /// snapshot that retained enforcement metadata but lost the user's owner
    /// name.
    ///
    /// Legacy snapshots synthesize reserved, content-derived owners. When the
    /// corresponding old `CreateConstraint` record is still present, recovery
    /// replaces those owners with the real WAL identity without removing and
    /// re-adding type or physical enforcement metadata. The operation is
    /// idempotent and never adopts a merely prefix-matching user name.
    #[cfg(any(all(feature = "lpg", feature = "wal"), test))]
    pub(crate) fn restore_named_constraint_from_wal(
        &mut self,
        definition: NamedConstraintDefinition,
    ) -> Result<(), CatalogError> {
        validate_named_constraint_definition(&definition)?;
        let schema = self.schema.as_mut().ok_or(CatalogError::SchemaNotEnabled)?;
        let named = &mut schema.named_constraints;

        if let Some(existing) = named.get(&definition.name) {
            return if existing == &definition {
                Ok(())
            } else {
                Err(CatalogError::ConstraintAlreadyExists)
            };
        }

        let legacy_names: Vec<String> = match definition.kind {
            NamedConstraintKind::Unique | NamedConstraintKind::NodeKey => named
                .values()
                .filter(|existing| {
                    is_authentic_legacy_constraint_owner(existing)
                        && existing.has_same_target(&definition)
                })
                .map(|existing| existing.name.clone())
                .collect(),
            NamedConstraintKind::NotNull | NamedConstraintKind::Exists => {
                let wanted: HashSet<&str> =
                    definition.properties.iter().map(String::as_str).collect();
                named
                    .values()
                    .filter(|existing| {
                        is_authentic_legacy_constraint_owner(existing)
                            && existing.label == definition.label
                            && existing.kind == NamedConstraintKind::NotNull
                            && existing.properties.len() == 1
                            && wanted.contains(existing.properties[0].as_str())
                    })
                    .map(|existing| existing.name.clone())
                    .collect()
            }
        };

        let expected_legacy_owners = match definition.kind {
            NamedConstraintKind::Unique | NamedConstraintKind::NodeKey => 1,
            NamedConstraintKind::NotNull | NamedConstraintKind::Exists => {
                definition.properties.len()
            }
        };
        if legacy_names.is_empty() {
            return self.install_named_constraint(definition, true);
        }
        if legacy_names.len() != expected_legacy_owners {
            return Err(CatalogError::InvalidConstraint(format!(
                "legacy owner set for constraint '{}' is incomplete",
                definition.name
            )));
        }

        for legacy_name in legacy_names {
            named.remove(&legacy_name);
        }
        named.insert(definition.name.clone(), definition);
        Ok(())
    }

    pub(crate) fn install_named_constraint(
        &mut self,
        definition: NamedConstraintDefinition,
        restoring: bool,
    ) -> Result<(), CatalogError> {
        validate_named_constraint_definition(&definition)?;

        let schema = self.schema.as_mut().ok_or(CatalogError::SchemaNotEnabled)?;
        let Some(type_definition) = schema.prepare_named_constraint(&definition, restoring)? else {
            return Ok(());
        };
        let physical_label = definition
            .label
            .rsplit_once('/')
            .map_or(definition.label.as_str(), |(_, label)| label);
        let labels = self.labels.prepare_names(&[physical_label])?;
        let label_id = labels.ids.first().copied().ok_or_else(|| {
            CatalogError::InvalidState("constraint preparation omitted its label".to_string())
        })?;
        let mut property_names = Vec::new();
        property_names
            .try_reserve(definition.properties.len())
            .map_err(|_| CatalogError::Capacity("constraint properties"))?;
        property_names.extend(definition.properties.iter().map(String::as_str));
        let properties = self.property_keys.prepare_names(&property_names)?;

        // All semantic checks and reservations precede this complete admission.
        schema.install_prepared_constraint(definition, label_id, &properties.ids, type_definition);
        self.labels.install_names(labels);
        self.property_keys.install_names(properties);
        Ok(())
    }

    /// Drops a named constraint and removes the enforcement metadata owned by
    /// that definition.
    ///
    /// # Errors
    ///
    /// Returns `ConstraintNotFound` when the name is unknown.
    pub(crate) fn drop_named_constraint(
        &mut self,
        name: &str,
    ) -> Result<NamedConstraintDefinition, CatalogError> {
        let schema = self.schema.as_mut().ok_or(CatalogError::SchemaNotEnabled)?;
        let definition = schema
            .get_named_constraint(name)
            .ok_or_else(|| CatalogError::ConstraintNotFound(name.to_string()))?;
        let physical_label = definition
            .label
            .rsplit_once('/')
            .map_or(definition.label.as_str(), |(_, label)| label);
        let label_id = self.labels.get_id(physical_label);
        let property_ids: Vec<Option<PropertyKeyId>> = definition
            .properties
            .iter()
            .map(|property| self.property_keys.get_id(property))
            .collect();
        schema.drop_named_constraint(name, label_id, &property_ids)
    }

    /// Returns one named constraint definition.
    #[must_use]
    pub(crate) fn get_named_constraint(&self, name: &str) -> Option<NamedConstraintDefinition> {
        self.schema
            .as_ref()
            .and_then(|schema| schema.get_named_constraint(name))
    }

    /// Returns all named constraints in deterministic name order.
    #[must_use]
    pub(crate) fn all_named_constraints(&self) -> Vec<NamedConstraintDefinition> {
        let mut definitions = self
            .schema
            .as_ref()
            .map(SchemaState::all_named_constraints)
            .unwrap_or_default();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    /// Whether this catalog still contains a synthetic owner adopted from a
    /// legacy checkpoint that did not persist user-visible constraint names.
    ///
    /// Recovery uses this only to reject an otherwise-unresolvable legacy
    /// `DropConstraint`: silently ignoring the missing name could leave the
    /// type-level enforcement active after the WAL says it was dropped.
    #[must_use]
    #[cfg(all(feature = "lpg", feature = "wal"))]
    pub(crate) fn has_legacy_constraint_owners(&self) -> bool {
        self.schema.as_ref().is_some_and(|schema| {
            schema
                .named_constraints
                .values()
                .any(is_authentic_legacy_constraint_owner)
        })
    }

    /// Whether an equivalent constraint target already has a named owner.
    #[must_use]
    pub(crate) fn has_equivalent_named_constraint(
        &self,
        definition: &NamedConstraintDefinition,
    ) -> bool {
        self.schema.as_ref().is_some_and(|schema| {
            schema
                .named_constraints
                .values()
                .any(|existing| existing.has_same_target(definition))
        })
    }

    /// Registers a node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a type with the same name exists.
    pub(crate) fn register_node_type(
        &mut self,
        def: NodeTypeDefinition,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.register_node_type(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Registers or replaces a node type definition.
    pub(crate) fn register_or_replace_node_type(&mut self, def: NodeTypeDefinition) {
        if let Some(schema) = &mut self.schema {
            schema.register_or_replace_node_type(def);
        }
    }

    /// Drops a node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the type does not exist.
    pub(crate) fn drop_node_type(&mut self, name: &str) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.drop_node_type(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets a node type definition by name.
    #[must_use]
    pub(crate) fn get_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.schema.as_ref().and_then(|s| s.get_node_type(name))
    }

    /// Gets a resolved node type with inherited properties from parents.
    #[must_use]
    pub(crate) fn resolved_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.schema
            .as_ref()
            .and_then(|s| s.resolved_node_type(name))
    }

    /// Returns all registered node type names.
    #[must_use]
    pub(crate) fn all_node_type_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(SchemaState::all_node_types)
            .unwrap_or_default()
    }

    /// Returns all registered edge type definition names.
    #[must_use]
    pub(crate) fn all_edge_type_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(SchemaState::all_edge_types)
            .unwrap_or_default()
    }

    /// Registers an edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if an edge type with the same name exists.
    pub(crate) fn register_edge_type_def(
        &mut self,
        def: EdgeTypeDefinition,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.register_edge_type(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Registers or replaces an edge type definition.
    pub(crate) fn register_or_replace_edge_type_def(&mut self, def: EdgeTypeDefinition) {
        if let Some(schema) = &mut self.schema {
            schema.register_or_replace_edge_type(def);
        }
    }

    /// Drops an edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    pub(crate) fn drop_edge_type_def(&mut self, name: &str) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.drop_edge_type(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets an edge type definition by name.
    #[must_use]
    pub(crate) fn get_edge_type_def(&self, name: &str) -> Option<EdgeTypeDefinition> {
        self.schema.as_ref().and_then(|s| s.get_edge_type(name))
    }

    /// Registers a graph type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a graph type with the same name exists.
    pub(crate) fn register_graph_type(
        &mut self,
        def: GraphTypeDefinition,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.register_graph_type(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Registers or replaces a graph type while preserving graph bindings.
    pub(crate) fn register_or_replace_graph_type(&mut self, def: GraphTypeDefinition) {
        if let Some(schema) = &mut self.schema {
            schema.register_or_replace_graph_type(def);
        }
    }

    /// Drops a graph type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the graph type does not exist.
    pub(crate) fn drop_graph_type(&mut self, name: &str) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.drop_graph_type(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Returns all registered graph type names.
    #[must_use]
    pub(crate) fn all_graph_type_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(SchemaState::all_graph_types)
            .unwrap_or_default()
    }

    /// Gets a graph type definition by name.
    #[must_use]
    pub(crate) fn get_graph_type_def(&self, name: &str) -> Option<GraphTypeDefinition> {
        self.schema.as_ref().and_then(|s| s.get_graph_type(name))
    }

    /// Registers a schema namespace.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::SchemaAlreadyExists` if the namespace already exists.
    pub(crate) fn register_schema_namespace(&mut self, name: String) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.register_schema(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a schema namespace.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::SchemaNotFound` if the namespace does not exist.
    pub(crate) fn drop_schema_namespace(&mut self, name: &str) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.drop_schema(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Checks whether a schema namespace exists.
    #[must_use]
    pub(crate) fn schema_exists(&self, name: &str) -> bool {
        self.schema.as_ref().is_some_and(|s| s.schema_exists(name))
    }

    /// Returns all registered schema namespace names.
    #[must_use]
    pub(crate) fn schema_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(|s| s.schema_names())
            .unwrap_or_default()
    }

    /// Adds a constraint to an existing node type, creating a minimal type if needed.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled.
    pub(crate) fn add_constraint_to_type(
        &mut self,
        label: &str,
        constraint: TypeConstraint,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.add_constraint_to_type(label, constraint),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a property to a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub(crate) fn alter_node_type_add_property(
        &mut self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_node_type_add_property(type_name, property),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a property from a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type or property does not exist.
    pub(crate) fn alter_node_type_drop_property(
        &mut self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_node_type_drop_property(type_name, property_name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a property to an edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub(crate) fn alter_edge_type_add_property(
        &mut self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_edge_type_add_property(type_name, property),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a property from an edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type or property does not exist.
    pub(crate) fn alter_edge_type_drop_property(
        &mut self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_edge_type_drop_property(type_name, property_name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a node type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub(crate) fn alter_graph_type_add_node_type(
        &mut self,
        graph_type_name: &str,
        node_type: String,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_graph_type_add_node_type(graph_type_name, node_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a node type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub(crate) fn alter_graph_type_drop_node_type(
        &mut self,
        graph_type_name: &str,
        node_type: &str,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_graph_type_drop_node_type(graph_type_name, node_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds an edge type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub(crate) fn alter_graph_type_add_edge_type(
        &mut self,
        graph_type_name: &str,
        edge_type: String,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_graph_type_add_edge_type(graph_type_name, edge_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops an edge type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub(crate) fn alter_graph_type_drop_edge_type(
        &mut self,
        graph_type_name: &str,
        edge_type: &str,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.alter_graph_type_drop_edge_type(graph_type_name, edge_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Binds a graph instance to a graph type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the graph type does not exist.
    pub(crate) fn bind_graph_type(
        &mut self,
        graph_path: &GraphPath,
        graph_type: String,
    ) -> Result<(), CatalogError> {
        self.validate_graph_type_binding_target(&graph_type)?;
        self.schema
            .as_mut()
            .ok_or(CatalogError::SchemaNotEnabled)?
            .graph_type_bindings
            .insert(graph_path.clone(), graph_type);
        Ok(())
    }

    /// Validates that a graph type exists without publishing a binding.
    ///
    /// Transactional graph creation uses this before its durable marker, then
    /// publishes the already-validated binding together with graph lifecycle.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::SchemaNotEnabled`] when schema support is
    /// disabled, or [`CatalogError::TypeNotFound`] when `graph_type` is not
    /// defined in the catalog.
    pub(crate) fn validate_graph_type_binding_target(
        &self,
        graph_type: &str,
    ) -> Result<(), CatalogError> {
        let schema = self.schema.as_ref().ok_or(CatalogError::SchemaNotEnabled)?;
        let definition = schema
            .get_graph_type(graph_type)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type.to_string()))?;
        for node_type in &definition.allowed_node_types {
            if schema.get_node_type(node_type).is_none() {
                return Err(CatalogError::TypeNotFound(node_type.clone()));
            }
        }
        for edge_type in &definition.allowed_edge_types {
            if schema.get_edge_type(edge_type).is_none() {
                return Err(CatalogError::EdgeTypeNotFound(edge_type.clone()));
            }
        }
        Ok(())
    }

    /// Returns whether a graph's binding is still the exact expected value.
    #[must_use]
    pub(crate) fn graph_type_binding_matches(
        &self,
        graph_path: &GraphPath,
        expected: Option<&str>,
    ) -> bool {
        self.schema.as_ref().is_some_and(|schema| {
            schema
                .graph_type_bindings
                .get(graph_path)
                .map(String::as_str)
                == expected
        })
    }

    /// Atomically replaces or removes a pre-validated graph type binding if it
    /// still matches.
    ///
    /// Returns `false` on a compare-and-swap mismatch and leaves the catalog
    /// untouched. The caller must validate any replacement with
    /// [`Self::validate_graph_type_binding_target`] before entering a durable
    /// publication phase; this operation is deliberately infallible so commit
    /// publication contains no catalog validation after its WAL marker.
    #[must_use]
    pub(crate) fn publish_graph_type_binding_if_same(
        &mut self,
        graph_path: &GraphPath,
        expected: Option<&str>,
        replacement: Option<String>,
    ) -> bool {
        let Some(schema) = self.schema.as_mut() else {
            return false;
        };
        let bindings = &mut schema.graph_type_bindings;
        if bindings.get(graph_path).map(String::as_str) != expected {
            return false;
        }
        match replacement {
            Some(graph_type) => {
                bindings.insert(graph_path.clone(), graph_type);
            }
            None => {
                bindings.remove(graph_path);
            }
        }
        true
    }

    /// Gets the graph type binding for a graph instance.
    pub(crate) fn get_graph_type_binding(&self, graph_path: &GraphPath) -> Option<String> {
        self.schema
            .as_ref()?
            .graph_type_bindings
            .get(graph_path)
            .cloned()
    }

    /// Replays a durable graph-type binding exactly during database recovery.
    ///
    /// Unlike the compare-and-swap publication path, recovery is the sole
    /// catalog writer and must deterministically reproduce the committed WAL
    /// stream over any older checkpoint. A referenced graph type must already
    /// exist in that recovered catalog cut.
    #[cfg(all(feature = "lpg", feature = "wal"))]
    pub(crate) fn replay_graph_type_binding(
        &mut self,
        graph_path: &GraphPath,
        graph_type: Option<String>,
    ) -> Result<(), CatalogError> {
        if let Some(graph_type) = graph_type.as_deref() {
            self.validate_graph_type_binding_target(graph_type)?;
        }
        let schema = self.schema.as_mut().ok_or(CatalogError::SchemaNotEnabled)?;
        let bindings = &mut schema.graph_type_bindings;
        match graph_type {
            Some(graph_type) => {
                bindings.insert(graph_path.clone(), graph_type);
            }
            None => {
                bindings.remove(graph_path);
            }
        }
        Ok(())
    }

    /// Registers a stored procedure.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a procedure with the same name exists.
    pub(crate) fn register_procedure(
        &mut self,
        def: ProcedureDefinition,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.register_procedure(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Replaces or creates a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled.
    pub(crate) fn replace_procedure(
        &mut self,
        def: ProcedureDefinition,
    ) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => {
                schema.replace_procedure(def);
                Ok(())
            }
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a stored procedure.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the procedure does not exist.
    pub(crate) fn drop_procedure(&mut self, name: &str) -> Result<(), CatalogError> {
        match &mut self.schema {
            Some(schema) => schema.drop_procedure(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets a stored procedure by name.
    pub(crate) fn get_procedure(&self, name: &str) -> Option<ProcedureDefinition> {
        self.schema.as_ref()?.get_procedure(name)
    }

    /// Returns all registered node type definitions.
    #[must_use]
    pub(crate) fn all_node_type_defs(&self) -> Vec<NodeTypeDefinition> {
        self.schema
            .as_ref()
            .map(SchemaState::all_node_type_defs)
            .unwrap_or_default()
    }

    /// Returns all registered edge type definitions.
    #[must_use]
    pub(crate) fn all_edge_type_defs(&self) -> Vec<EdgeTypeDefinition> {
        self.schema
            .as_ref()
            .map(SchemaState::all_edge_type_defs)
            .unwrap_or_default()
    }

    /// Returns all registered graph type definitions.
    #[must_use]
    pub(crate) fn all_graph_type_defs(&self) -> Vec<GraphTypeDefinition> {
        self.schema
            .as_ref()
            .map(SchemaState::all_graph_type_defs)
            .unwrap_or_default()
    }

    /// Returns all registered procedure definitions.
    #[must_use]
    pub(crate) fn all_procedure_defs(&self) -> Vec<ProcedureDefinition> {
        self.schema
            .as_ref()
            .map(SchemaState::all_procedure_defs)
            .unwrap_or_default()
    }

    /// Returns all graph type bindings (graph_path, type_name).
    #[must_use]
    pub(crate) fn all_graph_type_bindings(&self) -> Vec<(GraphPath, String)> {
        self.schema
            .as_ref()
            .map(SchemaState::all_graph_type_bindings)
            .unwrap_or_default()
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

// === Plain catalog dictionaries ===

trait DictionaryId: Copy {
    const KIND: &'static str;
    fn from_u32(value: u32) -> Self;
    fn as_u32(self) -> u32;
}

macro_rules! dictionary_id {
    ($id:ty, $kind:literal) => {
        impl DictionaryId for $id {
            const KIND: &'static str = $kind;
            fn from_u32(value: u32) -> Self {
                Self::new(value)
            }
            fn as_u32(self) -> u32 {
                <$id>::as_u32(&self)
            }
        }
    };
}

dictionary_id!(LabelId, "label");
dictionary_id!(PropertyKeyId, "property key");
dictionary_id!(EdgeTypeId, "edge type");

#[derive(Clone)]
struct NameDictionary<Id> {
    name_to_id: HashMap<Arc<str>, Id>,
    id_to_name: Vec<Arc<str>>,
    next_id: u32,
}

type LabelCatalog = NameDictionary<LabelId>;
type PropertyCatalog = NameDictionary<PropertyKeyId>;
type EdgeTypeCatalog = NameDictionary<EdgeTypeId>;

struct DictionaryAdmission<Id> {
    entries: Vec<(Arc<str>, Id)>,
    ids: Vec<Id>,
    next_id: u32,
}

impl<Id: DictionaryId> NameDictionary<Id> {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            id_to_name: Vec::new(),
            next_id: 0,
        }
    }

    fn get_or_create(&mut self, name: &str) -> Result<Id, CatalogError> {
        if let Some(id) = self.get_id(name) {
            return Ok(id);
        }
        let next = self
            .next_id
            .checked_add(1)
            .ok_or(CatalogError::IdExhausted(Id::KIND))?;
        self.name_to_id
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        self.id_to_name
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        let id = Id::from_u32(self.next_id);
        let name: Arc<str> = name.into();
        self.id_to_name.push(Arc::clone(&name));
        self.name_to_id.insert(name, id);
        self.next_id = next;
        Ok(id)
    }

    fn get_id(&self, name: &str) -> Option<Id> {
        self.name_to_id.get(name).copied()
    }
    fn get_name(&self, id: Id) -> Option<Arc<str>> {
        self.id_to_name.get(id.as_u32() as usize).cloned()
    }
    fn count(&self) -> usize {
        self.id_to_name.len()
    }
    fn all_names(&self) -> Vec<Arc<str>> {
        self.id_to_name.clone()
    }

    fn prepare_names(&mut self, names: &[&str]) -> Result<DictionaryAdmission<Id>, CatalogError> {
        let mut entries = Vec::new();
        let mut ids = Vec::new();
        let mut selected = HashMap::<Arc<str>, Id>::new();
        entries
            .try_reserve(names.len())
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        ids.try_reserve(names.len())
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        selected
            .try_reserve(names.len())
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        let mut next_id = self.next_id;
        for name in names {
            if let Some(id) = self.get_id(name).or_else(|| selected.get(*name).copied()) {
                ids.push(id);
                continue;
            }
            let id = Id::from_u32(next_id);
            next_id = next_id
                .checked_add(1)
                .ok_or(CatalogError::IdExhausted(Id::KIND))?;
            let name: Arc<str> = (*name).into();
            selected.insert(Arc::clone(&name), id);
            entries.push((name, id));
            ids.push(id);
        }
        self.name_to_id
            .try_reserve(entries.len())
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        self.id_to_name
            .try_reserve(entries.len())
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        Ok(DictionaryAdmission {
            entries,
            ids,
            next_id,
        })
    }

    fn install_names(&mut self, admission: DictionaryAdmission<Id>) {
        for (name, id) in admission.entries {
            self.id_to_name.push(Arc::clone(&name));
            self.name_to_id.insert(name, id);
        }
        self.next_id = admission.next_id;
    }

    fn from_names(names: Vec<Arc<str>>) -> Result<Self, CatalogError> {
        let next_id =
            u32::try_from(names.len()).map_err(|_| CatalogError::IdExhausted(Id::KIND))?;
        let mut name_to_id = HashMap::new();
        name_to_id
            .try_reserve(names.len())
            .map_err(|_| CatalogError::Capacity(Id::KIND))?;
        for (position, name) in names.iter().enumerate() {
            let position =
                u32::try_from(position).map_err(|_| CatalogError::IdExhausted(Id::KIND))?;
            if name_to_id
                .insert(Arc::clone(name), Id::from_u32(position))
                .is_some()
            {
                return Err(CatalogError::InvalidState(format!(
                    "duplicate {} dictionary name",
                    Id::KIND
                )));
            }
        }
        Ok(Self {
            name_to_id,
            id_to_name: names,
            next_id,
        })
    }
}

// === Index Catalog ===

/// Type of index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexType {
    /// Hash index for equality lookups.
    Hash,
    /// BTree index for range queries.
    BTree,
    /// Full-text index for text search.
    FullText,
    /// HNSW vector similarity index.
    Vector,
}

/// Index definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDefinition {
    /// The index ID.
    pub id: IndexId,
    /// Explicit name or catalog-allocated `@grafeo-index:<id>` name.
    pub name: String,
    /// The label this index applies to.
    pub label: LabelId,
    /// The property key being indexed.
    pub property_key: PropertyKeyId,
    /// Authoritative graph-qualified natural physical identity.
    pub key: PhysicalIndexKey,
    /// Authoritative resolved creation contract.
    pub configuration: IndexConfiguration,
    /// Read-only type projection of `configuration`.
    pub index_type: IndexType,
}

/// Manages index definitions.
#[derive(Clone)]
struct IndexCatalog {
    indexes: HashMap<IndexId, IndexDefinition>,
    label_indexes: HashMap<LabelId, Vec<IndexId>>,
    label_property_indexes: HashMap<(LabelId, PropertyKeyId), Vec<IndexId>>,
    name_index: HashMap<String, IndexId>,
    physical_owners: HashMap<PhysicalIndexKey, IndexId>,
    next_id: u32,
}

impl IndexCatalog {
    fn new() -> Self {
        Self {
            indexes: HashMap::new(),
            label_indexes: HashMap::new(),
            label_property_indexes: HashMap::new(),
            name_index: HashMap::new(),
            physical_owners: HashMap::new(),
            next_id: 0,
        }
    }

    fn create(
        &mut self,
        name: Option<&str>,
        label: LabelId,
        property_key: PropertyKeyId,
        key: PhysicalIndexKey,
        configuration: IndexConfiguration,
    ) -> Result<IndexId, CatalogError> {
        if let Some(name) = name
            && name.starts_with(ANONYMOUS_INDEX_PREFIX)
        {
            return Err(CatalogError::InvalidIndex(
                "explicit index name uses the reserved anonymous namespace".into(),
            ));
        }
        let id = IndexId::new(self.next_id);
        let name = name.map_or_else(
            || format!("{ANONYMOUS_INDEX_PREFIX}{}", id.as_u32()),
            str::to_owned,
        );
        self.create_exact(IndexDefinition {
            id,
            name,
            label,
            property_key,
            key,
            index_type: configuration.index_type(),
            configuration,
        })
    }

    /// Installs precisely the recorded next identity, sharing ordinary owner
    /// admission and reservation checks without allocating a replacement ID.
    fn create_exact(&mut self, definition: IndexDefinition) -> Result<IndexId, CatalogError> {
        let IndexDefinition {
            id,
            name,
            label,
            property_key,
            key,
            configuration,
            index_type,
        } = &definition;
        if id.as_u32() != self.next_id {
            return Err(CatalogError::InvalidIndex(
                "recorded owner ID differs from the next allocator identity".into(),
            ));
        }
        if name.starts_with(ANONYMOUS_INDEX_PREFIX)
            && *name != format!("{ANONYMOUS_INDEX_PREFIX}{}", id.as_u32())
        {
            return Err(CatalogError::InvalidIndex(
                "anonymous owner name disagrees with its ID".into(),
            ));
        }
        configuration.validate()?;
        if *index_type != configuration.index_type() {
            return Err(CatalogError::InvalidIndex(
                "owner kind disagrees with its configuration".into(),
            ));
        }
        if self.name_index.contains_key(name) {
            return Err(CatalogError::IndexAlreadyExists(name.clone()));
        }
        if let Some(owner) = self.physical_owners.get(key) {
            return Err(CatalogError::IndexPhysicalAlreadyOwned(*owner));
        }
        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or(CatalogError::IdExhausted("index"))?;
        self.indexes
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("index"))?;
        self.label_indexes
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("index labels"))?;
        self.label_property_indexes
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("index properties"))?;
        self.name_index
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("index names"))?;
        self.physical_owners
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("physical index owners"))?;
        let mut new_label_ids = Vec::new();
        match self.label_indexes.get_mut(label) {
            Some(ids) => ids.try_reserve(1),
            None => new_label_ids.try_reserve(1),
        }
        .map_err(|_| CatalogError::Capacity("index label entries"))?;
        let mut new_property_ids = Vec::new();
        match self
            .label_property_indexes
            .get_mut(&(*label, *property_key))
        {
            Some(ids) => ids.try_reserve(1),
            None => new_property_ids.try_reserve(1),
        }
        .map_err(|_| CatalogError::Capacity("index property entries"))?;
        let lookup_name = name.clone();
        let lookup_key = key.clone();
        let (id, label, property_key) = (*id, *label, *property_key);
        self.indexes.insert(id, definition);
        self.label_indexes
            .entry(label)
            .or_insert(new_label_ids)
            .push(id);
        self.label_property_indexes
            .entry((label, property_key))
            .or_insert(new_property_ids)
            .push(id);
        self.name_index.insert(lookup_name, id);
        self.physical_owners.insert(lookup_key, id);
        self.next_id = next_id;
        Ok(id)
    }

    fn drop(&mut self, id: IndexId) -> bool {
        let indexes = &mut self.indexes;
        let label_indexes = &mut self.label_indexes;
        let label_property_indexes = &mut self.label_property_indexes;

        if let Some(definition) = indexes.remove(&id) {
            // Remove from label index
            if let Some(ids) = label_indexes.get_mut(&definition.label) {
                ids.retain(|&i| i != id);
            }
            // Remove from label-property index
            if let Some(ids) =
                label_property_indexes.get_mut(&(definition.label, definition.property_key))
            {
                ids.retain(|&i| i != id);
            }
            // Remove from name index
            self.name_index.remove(&definition.name);
            self.physical_owners.remove(&definition.key);
            true
        } else {
            false
        }
    }

    fn find_by_name(&self, name: &str) -> Option<IndexId> {
        self.name_index.get(name).copied()
    }

    fn get(&self, id: IndexId) -> Option<IndexDefinition> {
        self.indexes.get(&id).cloned()
    }

    fn graph(&self, id: IndexId) -> Option<GraphPath> {
        self.indexes
            .get(&id)
            .map(|definition| definition.key.graph().clone())
    }

    fn for_label(&self, label: LabelId) -> Vec<IndexId> {
        self.label_indexes.get(&label).cloned().unwrap_or_default()
    }

    fn for_label_property(&self, label: LabelId, property_key: PropertyKeyId) -> Vec<IndexId> {
        self.label_property_indexes
            .get(&(label, property_key))
            .cloned()
            .unwrap_or_default()
    }

    fn count(&self) -> usize {
        self.indexes.len()
    }

    fn all(&self) -> Vec<IndexDefinition> {
        self.indexes.values().cloned().collect()
    }
}

// === Type Definitions ===

/// Data type for a typed property in a node or edge type definition.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum PropertyDataType {
    /// UTF-8 string.
    String,
    /// 64-bit signed integer.
    Int64,
    /// 64-bit floating point.
    Float64,
    /// Boolean.
    Bool,
    /// Calendar date.
    Date,
    /// Time of day.
    Time,
    /// Timestamp (date + time).
    Timestamp,
    /// Duration / interval.
    Duration,
    /// Ordered list of values (untyped).
    List,
    /// Typed list: `LIST<element_type>` (ISO sec 4.16.9).
    ListTyped(Box<PropertyDataType>),
    /// Key-value map.
    Map,
    /// Raw bytes.
    Bytes,
    /// Node reference type (ISO sec 4.15.1).
    Node,
    /// Edge reference type (ISO sec 4.15.1).
    Edge,
    /// Any type (no enforcement).
    Any,
}

impl PropertyDataType {
    /// Parses a type name string (case-insensitive) into a `PropertyDataType`.
    #[must_use]
    pub fn from_type_name(name: &str) -> Self {
        let upper = name.to_uppercase();
        // Handle parameterized LIST<element_type>
        if let Some(inner) = upper
            .strip_prefix("LIST<")
            .and_then(|s| s.strip_suffix('>'))
        {
            return Self::ListTyped(Box::new(Self::from_type_name(inner)));
        }
        match upper.as_str() {
            "STRING" | "VARCHAR" | "TEXT" => Self::String,
            "INT" | "INT64" | "INTEGER" | "BIGINT" => Self::Int64,
            "FLOAT" | "FLOAT64" | "DOUBLE" | "REAL" => Self::Float64,
            "BOOL" | "BOOLEAN" => Self::Bool,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "TIMESTAMP" | "DATETIME" => Self::Timestamp,
            "DURATION" | "INTERVAL" => Self::Duration,
            "LIST" | "ARRAY" => Self::List,
            "MAP" | "RECORD" => Self::Map,
            "BYTES" | "BINARY" | "BLOB" => Self::Bytes,
            "NODE" => Self::Node,
            "EDGE" | "RELATIONSHIP" => Self::Edge,
            _ => Self::Any,
        }
    }

    /// Checks whether a value conforms to this type.
    #[must_use]
    pub fn matches(&self, value: &Value) -> bool {
        match (self, value) {
            (Self::Any, _) | (_, Value::Null) => true,
            (Self::String, Value::String(_)) => true,
            (Self::Int64, Value::Int64(_)) => true,
            (Self::Float64, Value::Float64(_)) => true,
            (Self::Bool, Value::Bool(_)) => true,
            (Self::Date, Value::Date(_)) => true,
            (Self::Time, Value::Time(_)) => true,
            (Self::Timestamp, Value::Timestamp(_)) => true,
            (Self::Duration, Value::Duration(_)) => true,
            (Self::List, Value::List(_)) => true,
            (Self::ListTyped(elem_type), Value::List(items)) => {
                items.iter().all(|item| elem_type.matches(item))
            }
            (Self::Map, Value::Map(_)) => true,
            (Self::Bytes, Value::Bytes(_)) => true,
            // Node/Edge reference types match Map values (graph elements are
            // represented as maps with _id, _labels/_type, and properties)
            (Self::Node | Self::Edge, Value::Map(_)) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for PropertyDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "STRING"),
            Self::Int64 => write!(f, "INT64"),
            Self::Float64 => write!(f, "FLOAT64"),
            Self::Bool => write!(f, "BOOLEAN"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::Duration => write!(f, "DURATION"),
            Self::List => write!(f, "LIST"),
            Self::ListTyped(elem) => write!(f, "LIST<{elem}>"),
            Self::Map => write!(f, "MAP"),
            Self::Bytes => write!(f, "BYTES"),
            Self::Node => write!(f, "NODE"),
            Self::Edge => write!(f, "EDGE"),
            Self::Any => write!(f, "ANY"),
        }
    }
}

/// A typed property within a node or edge type definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TypedProperty {
    /// Property name.
    pub name: String,
    /// Expected data type.
    pub data_type: PropertyDataType,
    /// Whether NULL values are allowed.
    pub nullable: bool,
    /// Default value (used when property is not explicitly set).
    pub default_value: Option<Value>,
}

/// A constraint on a node or edge type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum TypeConstraint {
    /// Primary key (implies UNIQUE + NOT NULL).
    PrimaryKey(Vec<String>),
    /// Uniqueness constraint on one or more properties.
    Unique(Vec<String>),
    /// NOT NULL constraint on a single property.
    NotNull(String),
    /// CHECK constraint with a named expression string.
    Check {
        /// Optional constraint name.
        name: Option<String>,
        /// Expression (stored as string for now).
        expression: String,
    },
}

/// Durable kind tag for a user-visible named constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NamedConstraintKind {
    /// Values (or value tuples) must be unique.
    Unique,
    /// Values form a required unique node key.
    NodeKey,
    /// Every listed property must be present and non-null.
    NotNull,
    /// Every listed property must exist (represented as NOT NULL by the
    /// current LPG value model, where absence and null are both invalid).
    Exists,
}

impl NamedConstraintKind {
    /// Stable WAL and display spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unique => "unique",
            Self::NodeKey => "node_key",
            Self::NotNull => "not_null",
            Self::Exists => "exists",
        }
    }

    /// Parses the stable WAL spelling.
    #[must_use]
    pub fn from_wal_str(value: &str) -> Option<Self> {
        match value {
            "unique" => Some(Self::Unique),
            "node_key" => Some(Self::NodeKey),
            "not_null" => Some(Self::NotNull),
            "exists" => Some(Self::Exists),
            _ => None,
        }
    }
}

/// Durable metadata for a constraint created by schema DDL.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NamedConstraintDefinition {
    /// Schema-qualified constraint name.
    pub name: String,
    /// Schema-qualified node type/label key.
    pub label: String,
    /// Properties participating in the constraint.
    pub properties: Vec<String>,
    /// Constraint semantics.
    pub kind: NamedConstraintKind,
}

impl NamedConstraintDefinition {
    fn type_constraints(&self) -> Vec<TypeConstraint> {
        match self.kind {
            NamedConstraintKind::Unique => {
                vec![TypeConstraint::Unique(self.properties.clone())]
            }
            NamedConstraintKind::NodeKey => {
                vec![TypeConstraint::PrimaryKey(self.properties.clone())]
            }
            NamedConstraintKind::NotNull | NamedConstraintKind::Exists => self
                .properties
                .iter()
                .cloned()
                .map(TypeConstraint::NotNull)
                .collect(),
        }
    }

    fn has_same_target(&self, other: &Self) -> bool {
        if self.label != other.label || self.kind != other.kind {
            return false;
        }
        let mut left = self.properties.clone();
        let mut right = other.properties.clone();
        left.sort_unstable();
        right.sort_unstable();
        left == right
    }
}

fn validate_named_constraint_definition(
    definition: &NamedConstraintDefinition,
) -> Result<(), CatalogError> {
    if definition.name.is_empty() {
        return Err(CatalogError::InvalidConstraint(
            "constraint name must not be empty".to_string(),
        ));
    }
    if definition.label.is_empty() {
        return Err(CatalogError::InvalidConstraint(
            "constraint label must not be empty".to_string(),
        ));
    }
    if definition.properties.is_empty() {
        return Err(CatalogError::InvalidConstraint(
            "constraint must reference at least one property".to_string(),
        ));
    }
    let mut distinct = HashSet::with_capacity(definition.properties.len());
    if definition
        .properties
        .iter()
        .any(|property| property.is_empty() || !distinct.insert(property.as_str()))
    {
        return Err(CatalogError::InvalidConstraint(
            "constraint property names must be non-empty and distinct".to_string(),
        ));
    }
    Ok(())
}

#[cfg(any(all(feature = "lpg", feature = "wal"), test))]
fn legacy_constraint_hex(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len().saturating_mul(2));
    for byte in value.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(any(all(feature = "lpg", feature = "wal"), test))]
fn authentic_legacy_constraint_owner_name(definition: &NamedConstraintDefinition) -> String {
    let schema_prefix = definition
        .label
        .rsplit_once('/')
        .map_or("", |(schema, _)| schema);
    let mut local_name = format!(
        "__grafeo_legacy_{}_{}_{}",
        legacy_constraint_hex(definition.kind.as_str()),
        legacy_constraint_hex(&definition.label),
        definition.properties.len()
    );
    for property in &definition.properties {
        local_name.push('_');
        local_name.push_str(&legacy_constraint_hex(property));
    }
    if schema_prefix.is_empty() {
        local_name
    } else {
        format!("{schema_prefix}/{local_name}")
    }
}

#[cfg(any(all(feature = "lpg", feature = "wal"), test))]
fn is_authentic_legacy_constraint_owner(definition: &NamedConstraintDefinition) -> bool {
    definition.name == authentic_legacy_constraint_owner_name(definition)
}

/// Definition of a node type (label schema).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeTypeDefinition {
    /// Type name (corresponds to a label).
    pub name: String,
    /// Typed property definitions.
    pub properties: Vec<TypedProperty>,
    /// Type-level constraints.
    pub constraints: Vec<TypeConstraint>,
    /// Parent type names for inheritance (GQL `EXTENDS`).
    pub parent_types: Vec<String>,
}

/// Definition of an edge type (relationship type schema).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeTypeDefinition {
    /// Type name (corresponds to an edge type / relationship type).
    pub name: String,
    /// Typed property definitions.
    pub properties: Vec<TypedProperty>,
    /// Type-level constraints.
    pub constraints: Vec<TypeConstraint>,
    /// Allowed source node types (empty = any).
    pub source_node_types: Vec<String>,
    /// Allowed target node types (empty = any).
    pub target_node_types: Vec<String>,
}

/// Definition of a graph type (constrains which node/edge types a graph allows).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphTypeDefinition {
    /// Graph type name.
    pub name: String,
    /// Allowed node types (empty = open).
    pub allowed_node_types: Vec<String>,
    /// Allowed edge types (empty = open).
    pub allowed_edge_types: Vec<String>,
    /// Whether unlisted types are permitted.
    pub open: bool,
}

/// Definition of a stored procedure.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProcedureDefinition {
    /// Procedure name.
    pub name: String,
    /// Parameter definitions: (name, type).
    pub params: Vec<(String, String)>,
    /// Return column definitions: (name, type).
    pub returns: Vec<(String, String)>,
    /// Raw GQL query body.
    pub body: String,
}

impl CatalogWalStateV1 {
    #[cfg(any(all(feature = "lpg", feature = "gql"), test))]
    fn encode_snapshot(snapshot: CatalogStateSnapshot) -> Result<Vec<u8>, String> {
        let state = Self::from_snapshot(snapshot)?;
        bincode::serde::encode_to_vec(&state, bincode::config::standard())
            .map_err(|error| format!("failed to encode catalog WAL state v1: {error}"))
    }

    fn from_snapshot(snapshot: CatalogStateSnapshot) -> Result<Self, String> {
        let schema = snapshot
            .schema
            .ok_or_else(|| "catalog WAL state requires schema support".to_string())?;

        let mut unique_constraints: Vec<_> = schema
            .unique_constraints
            .into_iter()
            .map(|(label, property)| (label.as_u32(), property.as_u32()))
            .collect();
        unique_constraints.sort_unstable();
        let mut required_properties: Vec<_> = schema
            .required_properties
            .into_iter()
            .map(|(label, property)| (label.as_u32(), property.as_u32()))
            .collect();
        required_properties.sort_unstable();

        let mut named_constraints = Vec::with_capacity(schema.named_constraints.len());
        for (name, definition) in schema.named_constraints {
            if name != definition.name {
                return Err(format!(
                    "named constraint map key '{name}' disagrees with definition '{}'",
                    definition.name
                ));
            }
            named_constraints.push(NamedConstraintWalV1::from_live(definition));
        }
        named_constraints.sort_unstable_by(|left, right| left.name.cmp(&right.name));

        let mut node_types = Vec::with_capacity(schema.node_types.len());
        for (name, definition) in schema.node_types {
            if name != definition.name {
                return Err(format!(
                    "node type map key '{name}' disagrees with definition '{}'",
                    definition.name
                ));
            }
            node_types.push(NodeTypeWalV1::from_live(definition));
        }
        node_types.sort_unstable_by(|left, right| left.name.cmp(&right.name));

        let mut edge_types = Vec::with_capacity(schema.edge_types.len());
        for (name, definition) in schema.edge_types {
            if name != definition.name {
                return Err(format!(
                    "edge type map key '{name}' disagrees with definition '{}'",
                    definition.name
                ));
            }
            edge_types.push(EdgeTypeWalV1::from_live(definition));
        }
        edge_types.sort_unstable_by(|left, right| left.name.cmp(&right.name));

        let mut graph_types = Vec::with_capacity(schema.graph_types.len());
        for (name, definition) in schema.graph_types {
            if name != definition.name {
                return Err(format!(
                    "graph type map key '{name}' disagrees with definition '{}'",
                    definition.name
                ));
            }
            graph_types.push(GraphTypeWalV1::from_live(definition));
        }
        graph_types.sort_unstable_by(|left, right| left.name.cmp(&right.name));

        let mut graph_type_bindings = Vec::with_capacity(schema.graph_type_bindings.len());
        for (path, graph_type) in schema.graph_type_bindings {
            let [name] = path.components() else {
                return Err(
                    "catalog WAL state v1 cannot encode root or nested graph-type bindings"
                        .to_string(),
                );
            };
            graph_type_bindings.push((name.clone(), graph_type));
        }
        graph_type_bindings.sort_unstable();

        let mut procedures = Vec::with_capacity(schema.procedures.len());
        for (name, definition) in schema.procedures {
            if name != definition.name {
                return Err(format!(
                    "procedure map key '{name}' disagrees with definition '{}'",
                    definition.name
                ));
            }
            procedures.push(ProcedureWalV1::from_live(definition));
        }
        procedures.sort_unstable_by(|left, right| left.name.cmp(&right.name));

        Ok(Self {
            labels: snapshot
                .labels
                .into_iter()
                .map(|name| name.to_string())
                .collect(),
            property_keys: snapshot
                .property_keys
                .into_iter()
                .map(|name| name.to_string())
                .collect(),
            edge_types: snapshot
                .edge_types
                .into_iter()
                .map(|name| name.to_string())
                .collect(),
            schema: SchemaWalStateV1 {
                unique_constraints,
                required_properties,
                named_constraints,
                node_types,
                edge_types,
                graph_types,
                schemas: schema.schemas,
                graph_type_bindings,
                procedures,
            },
        })
    }

    fn into_snapshot(self) -> Result<CatalogStateSnapshot, String> {
        validate_dictionary("label", &self.labels)?;
        validate_dictionary("property-key", &self.property_keys)?;
        validate_dictionary("edge-type", &self.edge_types)?;

        let SchemaWalStateV1 {
            unique_constraints,
            required_properties,
            named_constraints,
            node_types,
            edge_types,
            graph_types,
            schemas,
            graph_type_bindings,
            procedures,
        } = self.schema;

        let unique_constraints = decode_physical_constraints(
            "unique",
            unique_constraints,
            self.labels.len(),
            self.property_keys.len(),
        )?;
        let required_properties = decode_physical_constraints(
            "required-property",
            required_properties,
            self.labels.len(),
            self.property_keys.len(),
        )?;

        let mut named_constraint_map = HashMap::with_capacity(named_constraints.len());
        for definition in named_constraints {
            let definition = definition.into_live()?;
            if named_constraint_map
                .insert(definition.name.clone(), definition)
                .is_some()
            {
                return Err("catalog WAL state contains duplicate named constraints".to_string());
            }
        }

        let mut node_type_map = HashMap::with_capacity(node_types.len());
        for definition in node_types {
            let definition = definition.into_live()?;
            if node_type_map
                .insert(definition.name.clone(), definition)
                .is_some()
            {
                return Err("catalog WAL state contains duplicate node types".to_string());
            }
        }

        let mut edge_type_map = HashMap::with_capacity(edge_types.len());
        for definition in edge_types {
            let definition = definition.into_live()?;
            if edge_type_map
                .insert(definition.name.clone(), definition)
                .is_some()
            {
                return Err("catalog WAL state contains duplicate edge types".to_string());
            }
        }

        let mut graph_type_map = HashMap::with_capacity(graph_types.len());
        for definition in graph_types {
            let definition = definition.into_live();
            if graph_type_map
                .insert(definition.name.clone(), definition)
                .is_some()
            {
                return Err("catalog WAL state contains duplicate graph types".to_string());
            }
        }

        validate_unique_strings("schema namespace", &schemas)?;

        let mut binding_map = HashMap::with_capacity(graph_type_bindings.len());
        for (graph, graph_type) in graph_type_bindings {
            let path = GraphPath::from_components(&[graph.as_str()])
                .map_err(|error| format!("invalid catalog WAL graph-type binding path: {error}"))?;
            if !graph_type_map.contains_key(&graph_type) {
                return Err(format!(
                    "graph '{graph}' is bound to missing graph type '{graph_type}'"
                ));
            }
            if binding_map.insert(path, graph_type).is_some() {
                return Err(format!(
                    "catalog WAL state contains duplicate binding for graph '{graph}'"
                ));
            }
        }

        let mut procedure_map = HashMap::with_capacity(procedures.len());
        for definition in procedures {
            let definition = definition.into_live();
            if procedure_map
                .insert(definition.name.clone(), definition)
                .is_some()
            {
                return Err("catalog WAL state contains duplicate procedures".to_string());
            }
        }

        Ok(CatalogStateSnapshot {
            labels: self.labels.into_iter().map(Arc::<str>::from).collect(),
            property_keys: self
                .property_keys
                .into_iter()
                .map(Arc::<str>::from)
                .collect(),
            edge_types: self.edge_types.into_iter().map(Arc::<str>::from).collect(),
            schema: Some(SchemaStateSnapshot {
                unique_constraints,
                required_properties,
                named_constraints: named_constraint_map,
                node_types: node_type_map,
                edge_types: edge_type_map,
                graph_types: graph_type_map,
                schemas,
                graph_type_bindings: binding_map,
                procedures: procedure_map,
            }),
        })
    }
}

fn validate_dictionary(kind: &str, names: &[String]) -> Result<(), String> {
    if u32::try_from(names.len()).is_err() {
        return Err(format!("catalog WAL state has too many {kind} names"));
    }
    validate_unique_strings(&format!("{kind} dictionary entry"), names)
}

fn validate_unique_strings(kind: &str, values: &[String]) -> Result<(), String> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !seen.insert(value.as_str()) {
            return Err(format!(
                "catalog WAL state contains duplicate {kind} '{value}'"
            ));
        }
    }
    Ok(())
}

fn decode_physical_constraints(
    kind: &str,
    values: Vec<(u32, u32)>,
    label_count: usize,
    property_count: usize,
) -> Result<HashSet<(LabelId, PropertyKeyId)>, String> {
    let mut decoded = HashSet::with_capacity(values.len());
    for (label, property) in values {
        if usize::try_from(label).map_or(true, |id| id >= label_count) {
            return Err(format!(
                "catalog WAL {kind} constraint references label id {label} outside dictionary"
            ));
        }
        if usize::try_from(property).map_or(true, |id| id >= property_count) {
            return Err(format!(
                "catalog WAL {kind} constraint references property id {property} outside dictionary"
            ));
        }
        if !decoded.insert((LabelId::new(label), PropertyKeyId::new(property))) {
            return Err(format!(
                "catalog WAL state contains duplicate {kind} physical constraint"
            ));
        }
    }
    Ok(decoded)
}

impl TypedPropertyWalV1 {
    fn from_live(property: TypedProperty) -> Self {
        Self {
            name: property.name,
            data_type: property.data_type.to_string(),
            nullable: property.nullable,
            default_value: property.default_value,
        }
    }

    fn into_live(self) -> Result<TypedProperty, String> {
        let data_type = PropertyDataType::from_type_name(&self.data_type);
        if data_type.to_string() != self.data_type {
            return Err(format!(
                "catalog WAL state contains non-canonical or unknown property type '{}'",
                self.data_type
            ));
        }
        Ok(TypedProperty {
            name: self.name,
            data_type,
            nullable: self.nullable,
            default_value: self.default_value,
        })
    }
}

impl TypeConstraintWalV1 {
    fn from_live(constraint: TypeConstraint) -> Self {
        match constraint {
            TypeConstraint::PrimaryKey(properties) => Self {
                kind: 0,
                properties,
                name: None,
                expression: None,
            },
            TypeConstraint::Unique(properties) => Self {
                kind: 1,
                properties,
                name: None,
                expression: None,
            },
            TypeConstraint::NotNull(property) => Self {
                kind: 2,
                properties: vec![property],
                name: None,
                expression: None,
            },
            TypeConstraint::Check { name, expression } => Self {
                kind: 3,
                properties: Vec::new(),
                name,
                expression: Some(expression),
            },
        }
    }

    fn into_live(self) -> Result<TypeConstraint, String> {
        match self.kind {
            0 if self.name.is_none()
                && self.expression.is_none()
                && !self.properties.is_empty() =>
            {
                Ok(TypeConstraint::PrimaryKey(self.properties))
            }
            1 if self.name.is_none()
                && self.expression.is_none()
                && !self.properties.is_empty() =>
            {
                Ok(TypeConstraint::Unique(self.properties))
            }
            2 if self.name.is_none() && self.expression.is_none() && self.properties.len() == 1 => {
                let property = self.properties.into_iter().next().ok_or_else(|| {
                    "catalog WAL NOT NULL constraint is missing its property".to_string()
                })?;
                Ok(TypeConstraint::NotNull(property))
            }
            3 if self.properties.is_empty() => {
                let expression = self.expression.ok_or_else(|| {
                    "catalog WAL CHECK constraint is missing its expression".to_string()
                })?;
                Ok(TypeConstraint::Check {
                    name: self.name,
                    expression,
                })
            }
            kind => Err(format!(
                "catalog WAL state contains invalid type constraint encoding (kind {kind})"
            )),
        }
    }
}

impl NamedConstraintWalV1 {
    fn from_live(definition: NamedConstraintDefinition) -> Self {
        let kind = match definition.kind {
            NamedConstraintKind::Unique => 0,
            NamedConstraintKind::NodeKey => 1,
            NamedConstraintKind::NotNull => 2,
            NamedConstraintKind::Exists => 3,
        };
        Self {
            name: definition.name,
            label: definition.label,
            properties: definition.properties,
            kind,
        }
    }

    fn into_live(self) -> Result<NamedConstraintDefinition, String> {
        if self.name.is_empty() || self.label.is_empty() || self.properties.is_empty() {
            return Err(
                "catalog WAL state contains an incomplete named constraint definition".to_string(),
            );
        }
        validate_unique_strings("named-constraint property", &self.properties)?;
        let kind = match self.kind {
            0 => NamedConstraintKind::Unique,
            1 => NamedConstraintKind::NodeKey,
            2 => NamedConstraintKind::NotNull,
            3 => NamedConstraintKind::Exists,
            kind => {
                return Err(format!(
                    "catalog WAL state contains unknown named constraint kind {kind}"
                ));
            }
        };
        Ok(NamedConstraintDefinition {
            name: self.name,
            label: self.label,
            properties: self.properties,
            kind,
        })
    }
}

impl NodeTypeWalV1 {
    fn from_live(definition: NodeTypeDefinition) -> Self {
        Self {
            name: definition.name,
            properties: definition
                .properties
                .into_iter()
                .map(TypedPropertyWalV1::from_live)
                .collect(),
            constraints: definition
                .constraints
                .into_iter()
                .map(TypeConstraintWalV1::from_live)
                .collect(),
            parent_types: definition.parent_types,
        }
    }

    fn into_live(self) -> Result<NodeTypeDefinition, String> {
        Ok(NodeTypeDefinition {
            name: self.name,
            properties: self
                .properties
                .into_iter()
                .map(TypedPropertyWalV1::into_live)
                .collect::<Result<_, _>>()?,
            constraints: self
                .constraints
                .into_iter()
                .map(TypeConstraintWalV1::into_live)
                .collect::<Result<_, _>>()?,
            parent_types: self.parent_types,
        })
    }
}

impl EdgeTypeWalV1 {
    fn from_live(definition: EdgeTypeDefinition) -> Self {
        Self {
            name: definition.name,
            properties: definition
                .properties
                .into_iter()
                .map(TypedPropertyWalV1::from_live)
                .collect(),
            constraints: definition
                .constraints
                .into_iter()
                .map(TypeConstraintWalV1::from_live)
                .collect(),
            source_node_types: definition.source_node_types,
            target_node_types: definition.target_node_types,
        }
    }

    fn into_live(self) -> Result<EdgeTypeDefinition, String> {
        Ok(EdgeTypeDefinition {
            name: self.name,
            properties: self
                .properties
                .into_iter()
                .map(TypedPropertyWalV1::into_live)
                .collect::<Result<_, _>>()?,
            constraints: self
                .constraints
                .into_iter()
                .map(TypeConstraintWalV1::into_live)
                .collect::<Result<_, _>>()?,
            source_node_types: self.source_node_types,
            target_node_types: self.target_node_types,
        })
    }
}

impl GraphTypeWalV1 {
    fn from_live(definition: GraphTypeDefinition) -> Self {
        Self {
            name: definition.name,
            allowed_node_types: definition.allowed_node_types,
            allowed_edge_types: definition.allowed_edge_types,
            open: definition.open,
        }
    }

    fn into_live(self) -> GraphTypeDefinition {
        GraphTypeDefinition {
            name: self.name,
            allowed_node_types: self.allowed_node_types,
            allowed_edge_types: self.allowed_edge_types,
            open: self.open,
        }
    }
}

impl ProcedureWalV1 {
    fn from_live(definition: ProcedureDefinition) -> Self {
        Self {
            name: definition.name,
            params: definition.params,
            returns: definition.returns,
            body: definition.body,
        }
    }

    fn into_live(self) -> ProcedureDefinition {
        ProcedureDefinition {
            name: self.name,
            params: self.params,
            returns: self.returns,
            body: self.body,
        }
    }
}

// === Schema Catalog ===

/// Schema constraints and type definitions.
#[derive(Clone, PartialEq)]
struct SchemaState {
    /// Properties that must be unique for a given label.
    unique_constraints: HashSet<(LabelId, PropertyKeyId)>,
    /// Properties that are required (NOT NULL) for a given label.
    required_properties: HashSet<(LabelId, PropertyKeyId)>,
    /// User-visible named constraint definitions.
    named_constraints: HashMap<String, NamedConstraintDefinition>,
    /// Registered node type definitions.
    node_types: HashMap<String, NodeTypeDefinition>,
    /// Registered edge type definitions.
    edge_types: HashMap<String, EdgeTypeDefinition>,
    /// Registered graph type definitions.
    graph_types: HashMap<String, GraphTypeDefinition>,
    /// Schema namespaces.
    schemas: Vec<String>,
    /// Graph instance to graph type bindings.
    graph_type_bindings: HashMap<GraphPath, String>,
    /// Stored procedure definitions.
    procedures: HashMap<String, ProcedureDefinition>,
}

impl SchemaState {
    fn new() -> Self {
        Self {
            unique_constraints: HashSet::new(),
            required_properties: HashSet::new(),
            named_constraints: HashMap::new(),
            node_types: HashMap::new(),
            edge_types: HashMap::new(),
            graph_types: HashMap::new(),
            schemas: Vec::new(),
            graph_type_bindings: HashMap::new(),
            procedures: HashMap::new(),
        }
    }

    // --- Node type operations ---

    /// Registers a new node type definition.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a type with the same name exists.
    pub fn register_node_type(&mut self, mut def: NodeTypeDefinition) -> Result<(), CatalogError> {
        let named = &self.named_constraints;
        for constraint in named
            .values()
            .filter(|constraint| constraint.label == def.name)
        {
            for type_constraint in constraint.type_constraints() {
                if !def.constraints.contains(&type_constraint) {
                    def.constraints.push(type_constraint);
                }
            }
        }
        let types = &mut self.node_types;
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
        types.insert(def.name.clone(), def);
        Ok(())
    }

    /// Registers or replaces a node type definition.
    pub fn register_or_replace_node_type(&mut self, mut def: NodeTypeDefinition) {
        let named = &self.named_constraints;
        for constraint in named
            .values()
            .filter(|constraint| constraint.label == def.name)
        {
            for type_constraint in constraint.type_constraints() {
                if !def.constraints.contains(&type_constraint) {
                    def.constraints.push(type_constraint);
                }
            }
        }
        self.node_types.insert(def.name.clone(), def);
    }

    /// Drops a node type definition by name.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no type with the given name exists.
    pub fn drop_node_type(&mut self, name: &str) -> Result<(), CatalogError> {
        if self
            .named_constraints
            .values()
            .any(|constraint| constraint.label == name)
        {
            return Err(CatalogError::TypeHasConstraints(name.to_string()));
        }
        let types = &mut self.node_types;
        if types.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets a node type definition by name.
    #[must_use]
    pub fn get_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.node_types.get(name).cloned()
    }

    /// Gets a resolved node type with inherited properties and constraints from parents.
    ///
    /// Walks the parent chain depth-first, collecting properties and constraints.
    /// Detects cycles via a visited set. Child properties override parent ones
    /// with the same name.
    #[must_use]
    pub fn resolved_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        let types = &self.node_types;
        let base = types.get(name)?;
        if base.parent_types.is_empty() {
            return Some(base.clone());
        }
        let mut visited = HashSet::new();
        visited.insert(name.to_string());
        let mut all_properties = Vec::new();
        let mut all_constraints = Vec::new();
        Self::collect_inherited(
            types,
            name,
            &mut visited,
            &mut all_properties,
            &mut all_constraints,
        );
        Some(NodeTypeDefinition {
            name: base.name.clone(),
            properties: all_properties,
            constraints: all_constraints,
            parent_types: base.parent_types.clone(),
        })
    }

    /// Recursively collects properties and constraints from a type and its parents.
    fn collect_inherited(
        types: &HashMap<String, NodeTypeDefinition>,
        name: &str,
        visited: &mut HashSet<String>,
        properties: &mut Vec<TypedProperty>,
        constraints: &mut Vec<TypeConstraint>,
    ) {
        let Some(def) = types.get(name) else { return };
        // Walk parents first (depth-first) so child properties override
        for parent in &def.parent_types {
            if visited.insert(parent.clone()) {
                Self::collect_inherited(types, parent, visited, properties, constraints);
            }
        }
        // Add own properties, overriding parent ones with same name
        for prop in &def.properties {
            if let Some(pos) = properties.iter().position(|p| p.name == prop.name) {
                properties[pos] = prop.clone();
            } else {
                properties.push(prop.clone());
            }
        }
        // Append own constraints (no dedup, constraints are additive)
        constraints.extend(def.constraints.iter().cloned());
    }

    /// Returns all registered node type names.
    #[must_use]
    pub fn all_node_types(&self) -> Vec<String> {
        self.node_types.keys().cloned().collect()
    }

    /// Returns all registered node type definitions.
    #[must_use]
    pub fn all_node_type_defs(&self) -> Vec<NodeTypeDefinition> {
        self.node_types.values().cloned().collect()
    }

    // --- Edge type operations ---

    /// Registers a new edge type definition.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if an edge type with the same name exists.
    pub fn register_edge_type(&mut self, def: EdgeTypeDefinition) -> Result<(), CatalogError> {
        let types = &mut self.edge_types;
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
        types.insert(def.name.clone(), def);
        Ok(())
    }

    /// Registers or replaces an edge type definition.
    pub fn register_or_replace_edge_type(&mut self, def: EdgeTypeDefinition) {
        self.edge_types.insert(def.name.clone(), def);
    }

    /// Drops an edge type definition by name.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no edge type with the given name exists.
    pub fn drop_edge_type(&mut self, name: &str) -> Result<(), CatalogError> {
        let types = &mut self.edge_types;
        if types.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets an edge type definition by name.
    #[must_use]
    pub fn get_edge_type(&self, name: &str) -> Option<EdgeTypeDefinition> {
        self.edge_types.get(name).cloned()
    }

    /// Returns all registered edge type names.
    #[must_use]
    pub fn all_edge_types(&self) -> Vec<String> {
        self.edge_types.keys().cloned().collect()
    }

    /// Returns all registered edge type definitions.
    #[must_use]
    pub fn all_edge_type_defs(&self) -> Vec<EdgeTypeDefinition> {
        self.edge_types.values().cloned().collect()
    }

    // --- Graph type operations ---

    /// Registers a new graph type definition.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a graph type with the same name exists.
    pub fn register_graph_type(&mut self, def: GraphTypeDefinition) -> Result<(), CatalogError> {
        let types = &mut self.graph_types;
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
        types.insert(def.name.clone(), def);
        Ok(())
    }

    /// Registers or replaces a graph type while retaining instance bindings.
    pub fn register_or_replace_graph_type(&mut self, def: GraphTypeDefinition) {
        self.graph_types.insert(def.name.clone(), def);
    }

    /// Drops a graph type definition by name.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no graph type with the given name exists.
    pub fn drop_graph_type(&mut self, name: &str) -> Result<(), CatalogError> {
        if self
            .graph_type_bindings
            .values()
            .any(|graph_type| graph_type == name)
        {
            return Err(CatalogError::GraphTypeHasBindings(name.to_string()));
        }
        let types = &mut self.graph_types;
        if types.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets a graph type definition by name.
    #[must_use]
    pub fn get_graph_type(&self, name: &str) -> Option<GraphTypeDefinition> {
        self.graph_types.get(name).cloned()
    }

    /// Returns all registered graph type names.
    #[must_use]
    pub fn all_graph_types(&self) -> Vec<String> {
        self.graph_types.keys().cloned().collect()
    }

    /// Returns all registered graph type definitions.
    #[must_use]
    pub fn all_graph_type_defs(&self) -> Vec<GraphTypeDefinition> {
        self.graph_types.values().cloned().collect()
    }

    // --- Schema namespace operations ---

    /// Registers a schema namespace.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaAlreadyExists` if the namespace already exists.
    pub fn register_schema(&mut self, name: String) -> Result<(), CatalogError> {
        let schemas = &mut self.schemas;
        if schemas.contains(&name) {
            return Err(CatalogError::SchemaAlreadyExists(name));
        }
        schemas.push(name);
        Ok(())
    }

    /// Drops a schema namespace.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotFound` if the namespace does not exist.
    pub fn drop_schema(&mut self, name: &str) -> Result<(), CatalogError> {
        let schemas = &mut self.schemas;
        if let Some(pos) = schemas.iter().position(|s| s == name) {
            schemas.remove(pos);
            Ok(())
        } else {
            Err(CatalogError::SchemaNotFound(name.to_string()))
        }
    }

    /// Checks whether a schema namespace exists.
    #[must_use]
    pub fn schema_exists(&self, name: &str) -> bool {
        self.schemas.iter().any(|s| s.eq_ignore_ascii_case(name))
    }

    /// Returns all registered schema namespace names.
    #[must_use]
    pub fn schema_names(&self) -> Vec<String> {
        self.schemas.clone()
    }

    // --- ALTER operations ---

    /// Adds a constraint to an existing node type, creating a minimal type if needed.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    pub fn add_constraint_to_type(
        &mut self,
        label: &str,
        constraint: TypeConstraint,
    ) -> Result<(), CatalogError> {
        let types = &mut self.node_types;
        if let Some(def) = types.get_mut(label) {
            def.constraints.push(constraint);
        } else {
            // Auto-create a minimal type definition for the label
            types.insert(
                label.to_string(),
                NodeTypeDefinition {
                    name: label.to_string(),
                    properties: Vec::new(),
                    constraints: vec![constraint],
                    parent_types: Vec::new(),
                },
            );
        }
        Ok(())
    }

    /// Adds a property to an existing node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeNotFound` if the node type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub fn alter_node_type_add_property(
        &mut self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        let types = &mut self.node_types;
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        if def.properties.iter().any(|p| p.name == property.name) {
            return Err(CatalogError::TypeAlreadyExists(format!(
                "property {} on {}",
                property.name, type_name
            )));
        }
        def.properties.push(property);
        Ok(())
    }

    /// Drops a property from an existing node type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the node type or property does not exist.
    pub fn alter_node_type_drop_property(
        &mut self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        let types = &mut self.node_types;
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        let len_before = def.properties.len();
        def.properties.retain(|p| p.name != property_name);
        if def.properties.len() == len_before {
            return Err(CatalogError::TypeNotFound(format!(
                "property {} on {}",
                property_name, type_name
            )));
        }
        Ok(())
    }

    /// Adds a property to an existing edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub fn alter_edge_type_add_property(
        &mut self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        let types = &mut self.edge_types;
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        if def.properties.iter().any(|p| p.name == property.name) {
            return Err(CatalogError::TypeAlreadyExists(format!(
                "property {} on {}",
                property.name, type_name
            )));
        }
        def.properties.push(property);
        Ok(())
    }

    /// Drops a property from an existing edge type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the edge type or property does not exist.
    pub fn alter_edge_type_drop_property(
        &mut self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        let types = &mut self.edge_types;
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        let len_before = def.properties.len();
        def.properties.retain(|p| p.name != property_name);
        if def.properties.len() == len_before {
            return Err(CatalogError::TypeNotFound(format!(
                "property {} on {}",
                property_name, type_name
            )));
        }
        Ok(())
    }

    /// Adds a node type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_node_type(
        &mut self,
        graph_type_name: &str,
        node_type: String,
    ) -> Result<(), CatalogError> {
        let types = &mut self.graph_types;
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        if !def.allowed_node_types.contains(&node_type) {
            def.allowed_node_types.push(node_type);
        }
        Ok(())
    }

    /// Drops a node type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_node_type(
        &mut self,
        graph_type_name: &str,
        node_type: &str,
    ) -> Result<(), CatalogError> {
        let types = &mut self.graph_types;
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        def.allowed_node_types.retain(|t| t != node_type);
        Ok(())
    }

    /// Adds an edge type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_edge_type(
        &mut self,
        graph_type_name: &str,
        edge_type: String,
    ) -> Result<(), CatalogError> {
        let types = &mut self.graph_types;
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        if !def.allowed_edge_types.contains(&edge_type) {
            def.allowed_edge_types.push(edge_type);
        }
        Ok(())
    }

    /// Drops an edge type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_edge_type(
        &mut self,
        graph_type_name: &str,
        edge_type: &str,
    ) -> Result<(), CatalogError> {
        let types = &mut self.graph_types;
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        def.allowed_edge_types.retain(|t| t != edge_type);
        Ok(())
    }

    // --- Procedure operations ---

    /// Registers a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a procedure with the same name exists.
    pub fn register_procedure(&mut self, def: ProcedureDefinition) -> Result<(), CatalogError> {
        let procs = &mut self.procedures;
        if procs.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name.clone()));
        }
        procs.insert(def.name.clone(), def);
        Ok(())
    }

    /// Replaces or creates a stored procedure.
    pub fn replace_procedure(&mut self, def: ProcedureDefinition) {
        self.procedures.insert(def.name.clone(), def);
    }

    /// Drops a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no procedure with the given name exists.
    pub fn drop_procedure(&mut self, name: &str) -> Result<(), CatalogError> {
        let procs = &mut self.procedures;
        if procs.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets a stored procedure by name.
    pub fn get_procedure(&self, name: &str) -> Option<ProcedureDefinition> {
        self.procedures.get(name).cloned()
    }

    /// Returns all registered procedure definitions.
    #[must_use]
    pub fn all_procedure_defs(&self) -> Vec<ProcedureDefinition> {
        self.procedures.values().cloned().collect()
    }

    /// Returns all graph type bindings (graph_path, type_name).
    #[must_use]
    pub fn all_graph_type_bindings(&self) -> Vec<(GraphPath, String)> {
        self.graph_type_bindings
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn prepare_named_constraint(
        &mut self,
        definition: &NamedConstraintDefinition,
        restoring: bool,
    ) -> Result<Option<NodeTypeDefinition>, CatalogError> {
        if let Some(existing) = self.named_constraints.get(&definition.name) {
            if restoring && existing == definition {
                return Ok(None);
            }
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        if self
            .named_constraints
            .values()
            .any(|existing| existing.has_same_target(definition))
        {
            return Err(CatalogError::ConstraintAlreadyExists);
        }

        let wanted = definition.type_constraints();
        let mut type_definition = self
            .node_types
            .get(&definition.label)
            .cloned()
            .unwrap_or_else(|| NodeTypeDefinition {
                name: definition.label.clone(),
                properties: Vec::new(),
                constraints: Vec::new(),
                parent_types: Vec::new(),
            });
        if !restoring
            && wanted
                .iter()
                .any(|constraint| type_definition.constraints.contains(constraint))
        {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        type_definition
            .constraints
            .try_reserve(wanted.len())
            .map_err(|_| CatalogError::Capacity("type constraints"))?;
        for constraint in wanted {
            if !type_definition.constraints.contains(&constraint) {
                type_definition.constraints.push(constraint);
            }
        }

        self.named_constraints
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("named constraints"))?;
        self.node_types
            .try_reserve(1)
            .map_err(|_| CatalogError::Capacity("node types"))?;
        self.unique_constraints
            .try_reserve(definition.properties.len())
            .map_err(|_| CatalogError::Capacity("unique constraints"))?;
        self.required_properties
            .try_reserve(definition.properties.len())
            .map_err(|_| CatalogError::Capacity("required properties"))?;
        Ok(Some(type_definition))
    }

    fn install_prepared_constraint(
        &mut self,
        definition: NamedConstraintDefinition,
        label_id: LabelId,
        property_ids: &[PropertyKeyId],
        type_definition: NodeTypeDefinition,
    ) {
        self.node_types
            .insert(definition.label.clone(), type_definition);
        match definition.kind {
            NamedConstraintKind::Unique => {
                let unique = &mut self.unique_constraints;
                unique.extend(property_ids.iter().map(|property| (label_id, *property)));
            }
            NamedConstraintKind::NodeKey => {
                let unique = &mut self.unique_constraints;
                let required = &mut self.required_properties;
                unique.extend(property_ids.iter().map(|property| (label_id, *property)));
                required.extend(property_ids.iter().map(|property| (label_id, *property)));
            }
            NamedConstraintKind::NotNull | NamedConstraintKind::Exists => {
                let required = &mut self.required_properties;
                required.extend(property_ids.iter().map(|property| (label_id, *property)));
            }
        }
        self.named_constraints
            .insert(definition.name.clone(), definition);
    }

    fn drop_named_constraint(
        &mut self,
        name: &str,
        label_id: Option<LabelId>,
        property_ids: &[Option<PropertyKeyId>],
    ) -> Result<NamedConstraintDefinition, CatalogError> {
        let named = &mut self.named_constraints;
        let definition = named
            .remove(name)
            .ok_or_else(|| CatalogError::ConstraintNotFound(name.to_string()))?;

        if let Some(type_definition) = self.node_types.get_mut(&definition.label) {
            for wanted in definition.type_constraints() {
                if let Some(position) = type_definition
                    .constraints
                    .iter()
                    .position(|constraint| constraint == &wanted)
                {
                    type_definition.constraints.remove(position);
                }
            }
        }

        if let Some(label_id) = label_id {
            for (property, property_id) in definition.properties.iter().zip(property_ids) {
                let Some(property_id) = property_id else {
                    continue;
                };
                let still_unique = named.values().any(|remaining| {
                    remaining.label == definition.label
                        && matches!(
                            remaining.kind,
                            NamedConstraintKind::Unique | NamedConstraintKind::NodeKey
                        )
                        && remaining.properties.iter().any(|item| item == property)
                });
                let still_required = named.values().any(|remaining| {
                    remaining.label == definition.label
                        && matches!(
                            remaining.kind,
                            NamedConstraintKind::NodeKey
                                | NamedConstraintKind::NotNull
                                | NamedConstraintKind::Exists
                        )
                        && remaining.properties.iter().any(|item| item == property)
                });
                if !still_unique {
                    self.unique_constraints.remove(&(label_id, *property_id));
                }
                if !still_required {
                    self.required_properties.remove(&(label_id, *property_id));
                }
            }
        }

        Ok(definition)
    }

    fn get_named_constraint(&self, name: &str) -> Option<NamedConstraintDefinition> {
        self.named_constraints.get(name).cloned()
    }

    fn all_named_constraints(&self) -> Vec<NamedConstraintDefinition> {
        self.named_constraints.values().cloned().collect()
    }

    fn add_unique_constraint(
        &mut self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        let constraints = &mut self.unique_constraints;
        let key = (label, property_key);
        if !constraints.insert(key) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        Ok(())
    }

    fn add_required_property(
        &mut self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        let required = &mut self.required_properties;
        let key = (label, property_key);
        if !required.insert(key) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        Ok(())
    }

    fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.required_properties.contains(&(label, property_key))
    }

    fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.unique_constraints.contains(&(label, property_key))
    }
}

// === Errors ===

/// Catalog-related errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CatalogError {
    /// A checked catalog identity counter has no remaining values.
    IdExhausted(&'static str),
    /// Storage for an admission could not be reserved before publication.
    Capacity(&'static str),
    /// A prepared state or replacement violates catalog invariants.
    InvalidState(String),
    /// Schema constraints are not enabled.
    SchemaNotEnabled,
    /// The constraint already exists.
    ConstraintAlreadyExists,
    /// No named constraint with this name exists.
    ConstraintNotFound(String),
    /// The constraint definition is malformed.
    InvalidConstraint(String),
    /// A node type still owns named constraints.
    TypeHasConstraints(String),
    /// A graph type is still bound to one or more graph instances.
    GraphTypeHasBindings(String),
    /// The label does not exist.
    LabelNotFound(String),
    /// The property key does not exist.
    PropertyKeyNotFound(String),
    /// The edge type does not exist.
    EdgeTypeNotFound(String),
    /// The index does not exist.
    IndexNotFound(IndexId),
    /// An index with this logical name already exists.
    IndexAlreadyExists(String),
    /// The physical index key already has a logical owner.
    IndexPhysicalAlreadyOwned(IndexId),
    /// An index name or resolved creation configuration is invalid.
    InvalidIndex(String),
    /// A type with this name already exists.
    TypeAlreadyExists(String),
    /// No type with this name exists.
    TypeNotFound(String),
    /// A schema with this name already exists.
    SchemaAlreadyExists(String),
    /// No schema with this name exists.
    SchemaNotFound(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdExhausted(kind) => write!(f, "Catalog {kind} identity space is exhausted"),
            Self::Capacity(kind) => write!(f, "Could not reserve catalog {kind} capacity"),
            Self::InvalidState(message) => write!(f, "Invalid catalog state: {message}"),
            Self::SchemaNotEnabled => write!(f, "Schema constraints are not enabled"),
            Self::ConstraintAlreadyExists => write!(f, "Constraint already exists"),
            Self::ConstraintNotFound(name) => write!(f, "Constraint not found: {name}"),
            Self::InvalidConstraint(message) => write!(f, "Invalid constraint: {message}"),
            Self::TypeHasConstraints(name) => {
                write!(f, "Type has named constraints: {name}")
            }
            Self::GraphTypeHasBindings(name) => {
                write!(f, "Graph type is still bound to graph instances: {name}")
            }
            Self::LabelNotFound(name) => write!(f, "Label not found: {name}"),
            Self::PropertyKeyNotFound(name) => write!(f, "Property key not found: {name}"),
            Self::EdgeTypeNotFound(name) => write!(f, "Edge type not found: {name}"),
            Self::IndexNotFound(id) => write!(f, "Index not found: {id}"),
            Self::IndexAlreadyExists(name) => write!(f, "Index already exists: {name}"),
            Self::IndexPhysicalAlreadyOwned(id) => {
                write!(f, "Physical index already belongs to owner {id}")
            }
            Self::InvalidIndex(message) => write!(f, "Invalid index: {message}"),
            Self::TypeAlreadyExists(name) => write!(f, "Type already exists: {name}"),
            Self::TypeNotFound(name) => write!(f, "Type not found: {name}"),
            Self::SchemaAlreadyExists(name) => write!(f, "Schema already exists: {name}"),
            Self::SchemaNotFound(name) => write!(f, "Schema not found: {name}"),
        }
    }
}

impl std::error::Error for CatalogError {}

// === Constraint Validator ===

use grafeo_core::execution::operators::ConstraintValidator;
use grafeo_core::execution::operators::OperatorError;

/// Source for the active graph-type binding seen by a transaction validator.
enum GraphTypeBindingOverride {
    Catalog,
    Session(Option<String>),
}

/// Validates schema constraints during mutation operations using the Catalog.
///
/// Checks type definitions, NOT NULL constraints, and UNIQUE constraints
/// against registered node/edge type definitions.
pub struct CatalogConstraintValidator {
    catalog: Arc<Catalog>,
    /// Optional exact graph path for graph-type-bound validation.
    graph_path: Option<GraphPath>,
    /// Transaction-view override for the active graph's type binding.
    graph_type_binding_override: GraphTypeBindingOverride,
    /// Current session schema. Node/edge TYPE keys are stored as "schema/Type",
    /// so bare labels are qualified with this before catalog lookups, and stored
    /// (qualified) allowed-type names are stripped before comparing to labels.
    current_schema: Option<String>,
    /// Optional graph store for UNIQUE constraint enforcement via index lookup.
    store: Option<Arc<dyn grafeo_core::graph::GraphStore>>,
    /// Optional maximum property value size in bytes.
    max_property_size: Option<usize>,
    /// Narrow engine-only authority for one RDF→LPG reconciliation.
    ///
    /// The public mutation surface never receives this capability. It exists
    /// solely so a rebuild can repair/delete its own reserved rows while the
    /// same validator rejects user attempts to forge the marker or mutate an
    /// already-owned node.
    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    rdf_projection_authority: Option<RdfProjectionMutationAuthority>,
}

#[cfg(all(feature = "lpg", feature = "triple-store"))]
#[derive(Clone)]
struct RdfProjectionMutationAuthority {
    owner_marker: String,
    node_label: String,
    desired_iris: Arc<std::collections::BTreeSet<String>>,
}

impl CatalogConstraintValidator {
    /// Creates a new validator wrapping the given catalog.
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            graph_path: None,
            graph_type_binding_override: GraphTypeBindingOverride::Catalog,
            current_schema: None,
            store: None,
            max_property_size: None,
            #[cfg(all(feature = "lpg", feature = "triple-store"))]
            rdf_projection_authority: None,
        }
    }

    /// Sets the exact graph path for graph-type-bound validation.
    pub fn with_graph_path(mut self, path: GraphPath) -> Self {
        self.graph_path = Some(path);
        self
    }

    /// Uses a transaction-local graph type binding for validation.
    #[must_use]
    pub fn with_graph_type_binding_override(mut self, binding: Option<String>) -> Self {
        self.graph_type_binding_override = GraphTypeBindingOverride::Session(binding);
        self
    }

    /// Attaches a graph store for UNIQUE constraint enforcement.
    pub fn with_store(mut self, store: Arc<dyn grafeo_core::graph::GraphStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Sets the maximum property value size in bytes.
    pub fn with_max_property_size(mut self, limit: Option<usize>) -> Self {
        self.max_property_size = limit;
        self
    }

    /// Grants this validator the narrowly-scoped capability used by one
    /// RDF→LPG rebuild transaction.
    ///
    /// This is crate-private by design: callers outside the engine cannot
    /// manufacture projection ownership or mutate projection-owned nodes.
    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    pub(crate) fn with_rdf_projection_authority(
        mut self,
        owner_marker: String,
        node_label: String,
        desired_iris: Arc<std::collections::BTreeSet<String>>,
    ) -> Self {
        self.rdf_projection_authority = Some(RdfProjectionMutationAuthority {
            owner_marker,
            node_label,
            desired_iris,
        });
        self
    }

    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    fn validate_rdf_projection_post_image(
        &self,
        current_node: Option<NodeId>,
        labels: &[String],
        properties: &[(String, Value)],
        viewing_epoch: EpochId,
    ) -> Result<(), OperatorError> {
        use grafeo_core::graph::rdf::{
            RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
        };

        let reserved = |key: &str| key.starts_with("__grafeo");
        let owner_key = PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY);
        let committed_pre_image = current_node.and_then(|node_id| {
            self.store
                .as_ref()?
                .get_node_at_epoch(node_id, viewing_epoch)
        });
        let pre_image_is_reserved = committed_pre_image.as_ref().is_some_and(|node| {
            node.properties
                .iter()
                .any(|(key, _)| reserved(key.as_str()))
        });
        let post_image_is_reserved = properties.iter().any(|(key, _)| reserved(key));

        if !pre_image_is_reserved && !post_image_is_reserved {
            return Ok(());
        }

        let Some(authority) = self.rdf_projection_authority.as_ref() else {
            return Err(OperatorError::ConstraintViolation(
                "properties in the reserved '__grafeo' namespace and RDF projection-owned nodes may only be mutated by the projection rebuild engine"
                    .to_string(),
            ));
        };

        if committed_pre_image.as_ref().is_some_and(|node| {
            node.properties.iter().any(|(key, _)| {
                reserved(key.as_str()) && key.as_str() != RDF_LPG_PROJECTION_OWNER_PROPERTY
            })
        }) || properties
            .iter()
            .any(|(key, _)| reserved(key) && key != RDF_LPG_PROJECTION_OWNER_PROPERTY)
        {
            return Err(OperatorError::ConstraintViolation(
                "projection rebuild encountered unsupported reserved metadata on an LPG node"
                    .to_string(),
            ));
        }

        let expected_owner = authority.owner_marker.as_str();
        let pre_owner = committed_pre_image
            .as_ref()
            .and_then(|node| node.properties.get(&owner_key))
            .and_then(Value::as_str);
        if pre_image_is_reserved && pre_owner != Some(expected_owner) {
            return Err(OperatorError::ConstraintViolation(
                "projection rebuild authority cannot mutate a row owned by another projection"
                    .to_string(),
            ));
        }

        let post_owner = properties
            .iter()
            .find(|(key, _)| key == RDF_LPG_PROJECTION_OWNER_PROPERTY)
            .and_then(|(_, value)| value.as_str());
        let post_iri = properties
            .iter()
            .find(|(key, _)| key == RDF_LPG_PROJECTION_IRI_PROPERTY)
            .and_then(|(_, value)| value.as_str());
        if post_owner != Some(expected_owner)
            || post_iri.is_none_or(|iri| !authority.desired_iris.contains(iri))
            || !labels.iter().any(|label| label == &authority.node_label)
        {
            return Err(OperatorError::ConstraintViolation(
                "projection rebuild attempted to publish an invalid owned-row post-image"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Authorizes deletion of a reserved node. Public validators have no
    /// authority, while a rebuild may delete only rows bearing its exact owner
    /// marker; the final commit-time target predicate decides whether that
    /// deletion leaves the complete generation valid.
    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    pub(crate) fn validate_rdf_projection_node_delete(
        &self,
        node_id: NodeId,
        viewing_epoch: EpochId,
    ) -> Result<(), OperatorError> {
        use grafeo_core::graph::rdf::RDF_LPG_PROJECTION_OWNER_PROPERTY;

        let Some(node) = self
            .store
            .as_ref()
            .and_then(|store| store.get_node_at_epoch(node_id, viewing_epoch))
        else {
            return Ok(());
        };
        let has_reserved = node
            .properties
            .iter()
            .any(|(key, _)| key.as_str().starts_with("__grafeo"));
        if !has_reserved {
            return Ok(());
        }
        let Some(authority) = self.rdf_projection_authority.as_ref() else {
            return Err(OperatorError::ConstraintViolation(
                "RDF projection-owned nodes may only be deleted by the projection rebuild engine"
                    .to_string(),
            ));
        };
        let owner = node
            .properties
            .get(&PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY))
            .and_then(Value::as_str);
        if owner != Some(authority.owner_marker.as_str()) {
            return Err(OperatorError::ConstraintViolation(
                "projection rebuild authority cannot delete a row owned by another projection"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Sets the current session schema, so bare labels are qualified to the
    /// "schema/Type" keys the catalog stores under a named schema.
    #[must_use]
    pub fn with_schema(mut self, schema: Option<String>) -> Self {
        self.current_schema = schema;
        self
    }

    /// Resolves a bare node label to the exact element-type coordinate governed
    /// by the bound graph type. This matters for cross-schema bindings: a graph
    /// in `s2` bound to `s1/G` must validate `:Person` against `s1/Person`, not
    /// silently fall back to `s2/Person`.
    fn node_type_key(&self, name: &str) -> String {
        if name.contains('/') {
            return name.to_string();
        }
        if let Some(graph_path) = self.graph_path.as_ref()
            && let Some(graph_type) = self.graph_type_binding(graph_path)
            && let Some(definition) = self.catalog.get_graph_type_def(&graph_type)
            && let Some(allowed) = definition
                .allowed_node_types
                .iter()
                .find(|allowed| Self::strip_schema(allowed) == name)
        {
            return allowed.clone();
        }
        self.current_schema
            .as_ref()
            .map_or_else(|| name.to_string(), |s| format!("{s}/{name}"))
    }

    /// Edge-type analogue of [`Self::node_type_key`].
    fn edge_type_key(&self, name: &str) -> String {
        if name.contains('/') {
            return name.to_string();
        }
        if let Some(graph_path) = self.graph_path.as_ref()
            && let Some(graph_type) = self.graph_type_binding(graph_path)
            && let Some(definition) = self.catalog.get_graph_type_def(&graph_type)
            && let Some(allowed) = definition
                .allowed_edge_types
                .iter()
                .find(|allowed| Self::strip_schema(allowed) == name)
        {
            return allowed.clone();
        }
        self.current_schema
            .as_ref()
            .map_or_else(|| name.to_string(), |s| format!("{s}/{name}"))
    }

    fn graph_type_binding(&self, graph_path: &GraphPath) -> Option<String> {
        match &self.graph_type_binding_override {
            GraphTypeBindingOverride::Catalog => self.catalog.get_graph_type_binding(graph_path),
            GraphTypeBindingOverride::Session(binding) => binding.clone(),
        }
    }

    /// Strips a leading "schema/" prefix from a stored type key so it compares
    /// against a bare label (a type name itself never contains '/').
    fn strip_schema(qualified: &str) -> &str {
        qualified.split_once('/').map_or(qualified, |(_, t)| t)
    }
}

impl ConstraintValidator for CatalogConstraintValidator {
    fn validate_node_delete(
        &self,
        node_id: NodeId,
        viewing_epoch: EpochId,
        _transaction_id: Option<TransactionId>,
    ) -> Result<(), OperatorError> {
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        {
            self.validate_rdf_projection_node_delete(node_id, viewing_epoch)
        }
        #[cfg(not(all(feature = "lpg", feature = "triple-store")))]
        {
            let _ = (node_id, viewing_epoch);
            Ok(())
        }
    }

    fn validate_node_property(
        &self,
        labels: &[String],
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError> {
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        if key.starts_with("__grafeo") {
            use grafeo_core::graph::rdf::RDF_LPG_PROJECTION_OWNER_PROPERTY;

            let authorized_owner = self
                .rdf_projection_authority
                .as_ref()
                .is_some_and(|_| key == RDF_LPG_PROJECTION_OWNER_PROPERTY);
            if !authorized_owner {
                return Err(OperatorError::ConstraintViolation(
                    "properties in the reserved '__grafeo' namespace may only be written by the projection rebuild engine"
                        .to_string(),
                ));
            }
        }
        if let Some(limit) = self.max_property_size {
            let size = value.estimated_size_bytes();
            if size > limit {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' value exceeds maximum size of {} MiB ({size} bytes)",
                    limit / (1024 * 1024)
                )));
            }
        }
        for label in labels {
            if let Some(type_def) = self.catalog.resolved_node_type(&self.node_type_key(label))
                && let Some(typed_prop) = type_def.properties.iter().find(|p| p.name == key)
            {
                // Check NOT NULL
                if !typed_prop.nullable && *value == Value::Null {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} is NOT NULL, cannot set to null"
                    )));
                }
                // Check type compatibility
                if *value != Value::Null && !typed_prop.data_type.matches(value) {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} expects {:?}, got {:?}",
                        typed_prop.data_type, value
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_node_complete(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        let props: std::collections::HashMap<&str, &Value> = properties
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();

        for label in labels {
            if let Some(type_def) = self.catalog.resolved_node_type(&self.node_type_key(label)) {
                // Check that all NOT NULL properties are present
                for typed_prop in &type_def.properties {
                    if !typed_prop.nullable
                        && props
                            .get(typed_prop.name.as_str())
                            .is_none_or(|value| **value == Value::Null)
                    {
                        return Err(OperatorError::ConstraintViolation(format!(
                            "missing required property '{}' on :{label}",
                            typed_prop.name
                        )));
                    }
                }
                // Check type-level constraints
                for constraint in &type_def.constraints {
                    match constraint {
                        TypeConstraint::NotNull(prop_name) => {
                            if props
                                .get(prop_name.as_str())
                                .is_none_or(|value| **value == Value::Null)
                            {
                                return Err(OperatorError::ConstraintViolation(format!(
                                    "missing required property '{prop_name}' on :{label} (NOT NULL constraint)"
                                )));
                            }
                        }
                        TypeConstraint::PrimaryKey(key_props) => {
                            for pk in key_props {
                                if props
                                    .get(pk.as_str())
                                    .is_none_or(|value| **value == Value::Null)
                                {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "missing primary key property '{pk}' on :{label}"
                                    )));
                                }
                            }
                        }
                        TypeConstraint::Check { name, expression } => {
                            match check_eval::evaluate_check(expression, properties) {
                                Ok(true) => {}
                                Ok(false) => {
                                    let constraint_name = name.as_deref().unwrap_or("unnamed");
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "CHECK constraint '{constraint_name}' violated on :{label}"
                                    )));
                                }
                                Err(err) => {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "CHECK constraint evaluation error: {err}"
                                    )));
                                }
                            }
                        }
                        TypeConstraint::Unique(_) => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn check_unique_node_property(
        &self,
        labels: &[String],
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError> {
        // Skip uniqueness check for NULL values (NULLs are never duplicates)
        if *value == Value::Null {
            return Ok(());
        }
        for label in labels {
            if let Some(type_def) = self.catalog.resolved_node_type(&self.node_type_key(label)) {
                for constraint in &type_def.constraints {
                    let is_unique = match constraint {
                        TypeConstraint::Unique(props) | TypeConstraint::PrimaryKey(props) => {
                            props.len() == 1 && props[0] == key
                        }
                        _ => false,
                    };
                    if is_unique && let Some(ref store) = self.store {
                        let existing = store.find_nodes_by_property(key, value);
                        for node_id in existing {
                            if let Some(node) = store.get_node(node_id) {
                                let has_label = node.labels.iter().any(|l| l.as_str() == label);
                                if has_label {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "UNIQUE constraint violation: property '{key}' \
                                             with value {value:?} already exists on :{label}"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_node_post_image(
        &self,
        current_node: Option<NodeId>,
        labels: &[String],
        properties: &[(String, Value)],
        viewing_epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Result<(), OperatorError> {
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        self.validate_rdf_projection_post_image(current_node, labels, properties, viewing_epoch)?;

        for (key, value) in properties {
            self.validate_node_property(labels, key, value)?;
        }
        self.validate_node_complete(labels, properties)?;

        let Some(store) = self.store.as_ref() else {
            return Ok(());
        };
        let post_image: std::collections::HashMap<&str, &Value> = properties
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();

        for label in labels {
            let Some(type_def) = self.catalog.resolved_node_type(&self.node_type_key(label)) else {
                continue;
            };
            for constraint in &type_def.constraints {
                let (keys, nulls_skip) = match constraint {
                    TypeConstraint::Unique(keys) => (keys, true),
                    TypeConstraint::PrimaryKey(keys) => (keys, false),
                    _ => continue,
                };
                let tuple: Option<Vec<Value>> = keys
                    .iter()
                    .map(|key| post_image.get(key.as_str()).map(|value| (**value).clone()))
                    .collect();
                let Some(tuple) = tuple else {
                    if nulls_skip {
                        continue;
                    }
                    // Completeness validation above reports the authoritative
                    // NODE KEY error before this branch is reachable.
                    continue;
                };
                if tuple.iter().any(Value::is_null) {
                    if nulls_skip {
                        continue;
                    }
                    continue;
                }

                let mut candidates = store.nodes_by_label_visible(label, transaction_id);
                if let Some(tid) = transaction_id {
                    // Inline CREATE/MERGE labels live in PENDING version chains,
                    // not the committed label index or label-op overlay. Include
                    // every tx-local create, then confirm its visible labels
                    // below. This is what makes two duplicate creates in one
                    // transaction conflict before commit.
                    candidates.extend(store.pending_node_creates(tid));
                }
                candidates.sort_unstable();
                candidates.dedup();
                for candidate in candidates {
                    if current_node == Some(candidate) {
                        continue;
                    }
                    let visible = transaction_id.map_or_else(
                        || store.get_node_at_epoch(candidate, viewing_epoch),
                        |tid| store.get_node_versioned(candidate, viewing_epoch, tid),
                    );
                    if visible.is_none() {
                        continue;
                    }
                    if !store
                        .read_node_labels_visible(candidate, viewing_epoch, transaction_id)
                        .iter()
                        .any(|candidate_label| candidate_label.as_str() == label)
                    {
                        continue;
                    }
                    let candidate_props = store.read_node_properties_visible(
                        candidate,
                        viewing_epoch,
                        transaction_id,
                    );
                    let candidate_tuple: Option<Vec<Value>> = keys
                        .iter()
                        .map(|key| {
                            candidate_props
                                .get(&PropertyKey::new(key.as_str()))
                                .cloned()
                        })
                        .collect();
                    let Some(candidate_tuple) = candidate_tuple else {
                        continue;
                    };
                    if candidate_tuple.iter().any(Value::is_null) {
                        continue;
                    }
                    if candidate_tuple == tuple {
                        return Err(OperatorError::ConstraintViolation(format!(
                            "{} constraint violation on :{label}({}): value tuple already exists",
                            if nulls_skip { "UNIQUE" } else { "NODE KEY" },
                            keys.join(", ")
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_edge_property(
        &self,
        edge_type: &str,
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError> {
        #[cfg(feature = "triple-store")]
        if key.starts_with("__grafeo") {
            return Err(OperatorError::ConstraintViolation(
                "properties in the reserved '__grafeo' namespace may not be attached to user-authored edges"
                    .to_string(),
            ));
        }
        if let Some(limit) = self.max_property_size {
            let size = value.estimated_size_bytes();
            if size > limit {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' value exceeds maximum size of {} MiB ({size} bytes)",
                    limit / (1024 * 1024)
                )));
            }
        }
        if let Some(type_def) = self
            .catalog
            .get_edge_type_def(&self.edge_type_key(edge_type))
            && let Some(typed_prop) = type_def.properties.iter().find(|p| p.name == key)
        {
            // Check NOT NULL
            if !typed_prop.nullable && *value == Value::Null {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' on :{edge_type} is NOT NULL, cannot set to null"
                )));
            }
            // Check type compatibility
            if *value != Value::Null && !typed_prop.data_type.matches(value) {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' on :{edge_type} expects {:?}, got {:?}",
                    typed_prop.data_type, value
                )));
            }
        }
        Ok(())
    }

    fn validate_edge_complete(
        &self,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        if let Some(type_def) = self
            .catalog
            .get_edge_type_def(&self.edge_type_key(edge_type))
        {
            let prop_names: std::collections::HashSet<&str> =
                properties.iter().map(|(n, _)| n.as_str()).collect();

            for typed_prop in &type_def.properties {
                if !typed_prop.nullable
                    && typed_prop.default_value.is_none()
                    && !prop_names.contains(typed_prop.name.as_str())
                {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "missing required property '{}' on :{edge_type}",
                        typed_prop.name
                    )));
                }
            }

            for constraint in &type_def.constraints {
                if let TypeConstraint::Check { name, expression } = constraint {
                    match check_eval::evaluate_check(expression, properties) {
                        Ok(true) => {}
                        Ok(false) => {
                            let constraint_name = name.as_deref().unwrap_or("unnamed");
                            return Err(OperatorError::ConstraintViolation(format!(
                                "CHECK constraint '{constraint_name}' violated on :{edge_type}"
                            )));
                        }
                        Err(err) => {
                            return Err(OperatorError::ConstraintViolation(format!(
                                "CHECK constraint evaluation error: {err}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_node_labels_allowed(&self, labels: &[String]) -> Result<(), OperatorError> {
        let Some(ref graph_path) = self.graph_path else {
            return Ok(());
        };
        let Some(type_name) = self.graph_type_binding(graph_path) else {
            return Ok(());
        };
        let Some(gt) = self.catalog.get_graph_type_def(&type_name) else {
            return Ok(());
        };
        if !gt.open {
            // `allowed_node_types` are stored schema-qualified ("schema/Person");
            // labels are bare. Strip the prefix so the comparison is correct
            // under both the default and a named schema. A closed graph admits
            // only a non-empty label set in which every label is declared;
            // one allowed label must not smuggle arbitrary extra labels.
            let allowed = !labels.is_empty()
                && labels.iter().all(|l| {
                    gt.allowed_node_types
                        .iter()
                        .any(|a| Self::strip_schema(a) == l)
                });
            if !allowed {
                return Err(OperatorError::ConstraintViolation(format!(
                    "node labels {labels:?} are not allowed by graph type '{}'",
                    gt.name
                )));
            }
        }
        Ok(())
    }

    fn validate_edge_type_allowed(&self, edge_type: &str) -> Result<(), OperatorError> {
        let Some(ref graph_path) = self.graph_path else {
            return Ok(());
        };
        let Some(type_name) = self.graph_type_binding(graph_path) else {
            return Ok(());
        };
        let Some(gt) = self.catalog.get_graph_type_def(&type_name) else {
            return Ok(());
        };
        if !gt.open {
            let allowed = gt
                .allowed_edge_types
                .iter()
                .any(|a| Self::strip_schema(a) == edge_type);
            if !allowed {
                return Err(OperatorError::ConstraintViolation(format!(
                    "edge type '{edge_type}' is not allowed by graph type '{}'",
                    gt.name
                )));
            }
        }
        Ok(())
    }

    fn validate_edge_endpoints(
        &self,
        edge_type: &str,
        source_labels: &[String],
        target_labels: &[String],
    ) -> Result<(), OperatorError> {
        let Some(type_def) = self
            .catalog
            .get_edge_type_def(&self.edge_type_key(edge_type))
        else {
            return Ok(());
        };
        // source/target_node_types may be stored schema-qualified ("schema/Person")
        // while endpoint labels are bare; strip the prefix so the comparison is
        // correct under both the default and a named schema (no false rejection).
        if !type_def.source_node_types.is_empty() {
            let source_ok = source_labels.iter().any(|l| {
                type_def
                    .source_node_types
                    .iter()
                    .any(|s| Self::strip_schema(s) == l)
            });
            if !source_ok {
                return Err(OperatorError::ConstraintViolation(format!(
                    "source node labels {source_labels:?} are not allowed for edge type '{edge_type}', \
                     expected one of {:?}",
                    type_def.source_node_types
                )));
            }
        }
        if !type_def.target_node_types.is_empty() {
            let target_ok = target_labels.iter().any(|l| {
                type_def
                    .target_node_types
                    .iter()
                    .any(|t| Self::strip_schema(t) == l)
            });
            if !target_ok {
                return Err(OperatorError::ConstraintViolation(format!(
                    "target node labels {target_labels:?} are not allowed for edge type '{edge_type}', \
                     expected one of {:?}",
                    type_def.target_node_types
                )));
            }
        }
        Ok(())
    }

    fn inject_defaults(&self, labels: &[String], properties: &mut Vec<(String, Value)>) {
        for label in labels {
            if let Some(type_def) = self.catalog.resolved_node_type(&self.node_type_key(label)) {
                for typed_prop in &type_def.properties {
                    if let Some(ref default) = typed_prop.default_value {
                        let already_set = properties.iter().any(|(n, _)| n == &typed_prop.name);
                        if !already_set {
                            properties.push((typed_prop.name.clone(), default.clone()));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn catalog_foundation_exhausted_label_admission_preserves_the_dictionary() {
        let catalog = Catalog::new();
        let _ = catalog.get_or_create_label("existing");
        let before = catalog.all_labels();
        catalog.state.write().labels.next_id = u32::MAX;

        let _ = catalog.get_or_create_label("must_not_be_published");

        assert_eq!(catalog.all_labels(), before);
        assert_eq!(catalog.get_label_id("must_not_be_published"), None);
        assert_eq!(catalog.state.read().labels.next_id, u32::MAX);
    }

    #[test]
    fn catalog_foundation_wal_replacement_rejects_changed_retained_index_referents()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();
        let _ = catalog.get_or_create_label("original_label");
        let _ = catalog.get_or_create_property_key("original_property");
        let label = catalog
            .get_label_id("original_label")
            .ok_or("missing label")?;
        let property = catalog
            .get_property_key_id("original_property")
            .ok_or("missing property")?;
        catalog.create_index(
            Some("retained"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        let before = catalog.encode_wal_state_v1()?;
        let owner = catalog.find_index_by_name("retained");
        let incoming = Catalog::new();
        let _ = incoming.get_or_create_label("different_label");
        let _ = incoming.get_or_create_property_key("different_property");

        let result = catalog.restore_wal_state_v1(&incoming.encode_wal_state_v1()?);

        assert!(
            result.is_err(),
            "replacement must not retarget a retained index"
        );
        assert_eq!(catalog.encode_wal_state_v1()?, before);
        assert_eq!(catalog.find_index_by_name("retained"), owner);
        Ok(())
    }

    #[test]
    fn test_catalog_labels() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        // Get or create labels
        let person_id = catalog.get_or_create_label("Person")?;
        let company_id = catalog.get_or_create_label("Company")?;

        // IDs should be different
        assert_ne!(person_id, company_id);

        // Getting the same label should return the same ID
        assert_eq!(catalog.get_or_create_label("Person")?, person_id);

        // Should be able to look up by name
        assert_eq!(catalog.get_label_id("Person"), Some(person_id));
        assert_eq!(catalog.get_label_id("Company"), Some(company_id));
        assert_eq!(catalog.get_label_id("Unknown"), None);

        // Should be able to look up by ID
        assert_eq!(catalog.get_label_name(person_id).as_deref(), Some("Person"));
        assert_eq!(
            catalog.get_label_name(company_id).as_deref(),
            Some("Company")
        );

        // Count should be correct
        assert_eq!(catalog.label_count(), 2);
        Ok(())
    }

    #[test]
    fn test_catalog_property_keys() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        let name_id = catalog.get_or_create_property_key("name")?;
        let age_id = catalog.get_or_create_property_key("age")?;

        assert_ne!(name_id, age_id);
        assert_eq!(catalog.get_or_create_property_key("name")?, name_id);
        assert_eq!(catalog.get_property_key_id("name"), Some(name_id));
        assert_eq!(
            catalog.get_property_key_name(name_id).as_deref(),
            Some("name")
        );
        assert_eq!(catalog.property_key_count(), 2);
        Ok(())
    }

    #[test]
    fn test_catalog_edge_types() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        let knows_id = catalog.get_or_create_edge_type("KNOWS")?;
        let works_at_id = catalog.get_or_create_edge_type("WORKS_AT")?;

        assert_ne!(knows_id, works_at_id);
        assert_eq!(catalog.get_or_create_edge_type("KNOWS")?, knows_id);
        assert_eq!(catalog.get_edge_type_id("KNOWS"), Some(knows_id));
        assert_eq!(
            catalog.get_edge_type_name(knows_id).as_deref(),
            Some("KNOWS")
        );
        assert_eq!(catalog.edge_type_count(), 2);
        Ok(())
    }

    #[test]
    fn test_catalog_indexes() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person")?;
        let name_id = catalog.get_or_create_property_key("name")?;
        let age_id = catalog.get_or_create_property_key("age")?;

        // Create indexes
        let idx1 = catalog.create_index(
            Some("idx_person_name"),
            person_id,
            name_id,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        let idx2 = catalog.create_index(
            Some("idx_person_age"),
            person_id,
            age_id,
            GraphPath::root(),
            IndexConfiguration::BTree,
        )?;

        assert_ne!(idx1, idx2);
        assert_eq!(catalog.index_count(), 2);

        // Look up by label
        let label_indexes = catalog.indexes_for_label(person_id);
        assert_eq!(label_indexes.len(), 2);
        assert!(label_indexes.contains(&idx1));
        assert!(label_indexes.contains(&idx2));

        // Look up by label and property
        let name_indexes = catalog.indexes_for_label_property(person_id, name_id);
        assert_eq!(name_indexes.len(), 1);
        assert_eq!(name_indexes[0], idx1);

        // Get definition
        let def = catalog.get_index(idx1).unwrap();
        assert_eq!(def.label, person_id);
        assert_eq!(def.property_key, name_id);
        assert_eq!(def.index_type, IndexType::Hash);

        // Drop index
        assert!(catalog.drop_index(idx1));
        assert_eq!(catalog.index_count(), 1);
        assert!(catalog.get_index(idx1).is_none());
        assert_eq!(catalog.indexes_for_label(person_id).len(), 1);
        Ok(())
    }

    #[test]
    fn test_catalog_schema_constraints() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::with_schema();

        let person_id = catalog.get_or_create_label("Person")?;
        let email_id = catalog.get_or_create_property_key("email")?;
        let name_id = catalog.get_or_create_property_key("name")?;

        // Add constraints
        assert!(catalog.add_unique_constraint(person_id, email_id).is_ok());
        assert!(catalog.add_required_property(person_id, name_id).is_ok());

        // Check constraints
        assert!(catalog.is_property_unique(person_id, email_id));
        assert!(!catalog.is_property_unique(person_id, name_id));
        assert!(catalog.is_property_required(person_id, name_id));
        assert!(!catalog.is_property_required(person_id, email_id));

        // Duplicate constraint should fail
        assert_eq!(
            catalog.add_unique_constraint(person_id, email_id),
            Err(CatalogError::ConstraintAlreadyExists)
        );
        Ok(())
    }

    #[test]
    fn test_catalog_schema_always_enabled() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Catalog::new() always enables schema
        let catalog = Catalog::new();
        assert!(catalog.has_schema());

        let person_id = catalog.get_or_create_label("Person")?;
        let email_id = catalog.get_or_create_property_key("email")?;

        // Should succeed with schema enabled
        assert_eq!(catalog.add_unique_constraint(person_id, email_id), Ok(()));
        Ok(())
    }

    // === Additional tests for comprehensive coverage ===

    #[test]
    fn test_catalog_default() {
        let catalog = Catalog::default();
        assert!(catalog.has_schema());
        assert_eq!(catalog.label_count(), 0);
        assert_eq!(catalog.property_key_count(), 0);
        assert_eq!(catalog.edge_type_count(), 0);
        assert_eq!(catalog.index_count(), 0);
    }

    #[test]
    fn test_catalog_all_labels() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        catalog.get_or_create_label("Person")?;
        catalog.get_or_create_label("Company")?;
        catalog.get_or_create_label("Product")?;

        let all = catalog.all_labels();
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|l| l.as_ref() == "Person"));
        assert!(all.iter().any(|l| l.as_ref() == "Company"));
        assert!(all.iter().any(|l| l.as_ref() == "Product"));
        Ok(())
    }

    #[test]
    fn test_catalog_all_property_keys() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        catalog.get_or_create_property_key("name")?;
        catalog.get_or_create_property_key("age")?;
        catalog.get_or_create_property_key("email")?;

        let all = catalog.all_property_keys();
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|k| k.as_ref() == "name"));
        assert!(all.iter().any(|k| k.as_ref() == "age"));
        assert!(all.iter().any(|k| k.as_ref() == "email"));
        Ok(())
    }

    #[test]
    fn test_catalog_all_edge_types() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        catalog.get_or_create_edge_type("KNOWS")?;
        catalog.get_or_create_edge_type("WORKS_AT")?;
        catalog.get_or_create_edge_type("LIVES_IN")?;

        let all = catalog.all_edge_types();
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|t| t.as_ref() == "KNOWS"));
        assert!(all.iter().any(|t| t.as_ref() == "WORKS_AT"));
        assert!(all.iter().any(|t| t.as_ref() == "LIVES_IN"));
        Ok(())
    }

    #[test]
    fn test_catalog_invalid_id_lookup() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        // Create one label to ensure IDs are allocated
        let _ = catalog.get_or_create_label("Person")?;

        // Try to look up non-existent IDs
        let invalid_label = LabelId::new(999);
        let invalid_property = PropertyKeyId::new(999);
        let invalid_edge_type = EdgeTypeId::new(999);
        let invalid_index = IndexId::new(999);

        assert!(catalog.get_label_name(invalid_label).is_none());
        assert!(catalog.get_property_key_name(invalid_property).is_none());
        assert!(catalog.get_edge_type_name(invalid_edge_type).is_none());
        assert!(catalog.get_index(invalid_index).is_none());
        Ok(())
    }

    #[test]
    fn test_catalog_drop_nonexistent_index() {
        let catalog = Catalog::new();
        let invalid_index = IndexId::new(999);
        assert!(!catalog.drop_index(invalid_index));
    }

    #[test]
    fn test_catalog_indexes_for_nonexistent_label() {
        let catalog = Catalog::new();
        let invalid_label = LabelId::new(999);
        let invalid_property = PropertyKeyId::new(999);

        assert!(catalog.indexes_for_label(invalid_label).is_empty());
        assert!(
            catalog
                .indexes_for_label_property(invalid_label, invalid_property)
                .is_empty()
        );
    }

    #[test]
    fn test_catalog_rejects_property_family_aliases()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person")?;
        let name_id = catalog.get_or_create_property_key("name")?;

        let hash_idx = catalog.create_index(
            Some("idx_hash"),
            person_id,
            name_id,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert_eq!(
            catalog.create_index(
                Some("idx_btree"),
                person_id,
                name_id,
                GraphPath::root(),
                IndexConfiguration::BTree
            ),
            Err(CatalogError::IndexPhysicalAlreadyOwned(hash_idx)),
        );
        assert_eq!(catalog.index_count(), 1);

        let indexes = catalog.indexes_for_label_property(person_id, name_id);
        assert_eq!(indexes.len(), 1);
        assert!(indexes.contains(&hash_idx));

        // Verify each has the correct type
        assert_eq!(
            catalog.get_index(hash_idx).unwrap().index_type,
            IndexType::Hash
        );
        assert_eq!(catalog.index_allocator_high_water(), 1);
        Ok(())
    }

    #[test]
    fn test_catalog_schema_required_property_duplicate()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::with_schema();

        let person_id = catalog.get_or_create_label("Person")?;
        let name_id = catalog.get_or_create_property_key("name")?;

        // First should succeed
        assert!(catalog.add_required_property(person_id, name_id).is_ok());

        // Duplicate should fail
        assert_eq!(
            catalog.add_required_property(person_id, name_id),
            Err(CatalogError::ConstraintAlreadyExists)
        );
        Ok(())
    }

    #[test]
    fn test_catalog_schema_check_without_constraints()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person")?;
        let name_id = catalog.get_or_create_property_key("name")?;

        // Without schema enabled, these should return false
        assert!(!catalog.is_property_unique(person_id, name_id));
        assert!(!catalog.is_property_required(person_id, name_id));
        Ok(())
    }

    #[test]
    fn test_catalog_has_schema() {
        // Both new() and with_schema() enable schema by default
        let catalog = Catalog::new();
        assert!(catalog.has_schema());

        let with_schema = Catalog::with_schema();
        assert!(with_schema.has_schema());
    }

    #[test]
    fn test_catalog_error_display() {
        assert_eq!(
            CatalogError::SchemaNotEnabled.to_string(),
            "Schema constraints are not enabled"
        );
        assert_eq!(
            CatalogError::ConstraintAlreadyExists.to_string(),
            "Constraint already exists"
        );
        assert_eq!(
            CatalogError::LabelNotFound("Person".to_string()).to_string(),
            "Label not found: Person"
        );
        assert_eq!(
            CatalogError::PropertyKeyNotFound("name".to_string()).to_string(),
            "Property key not found: name"
        );
        assert_eq!(
            CatalogError::EdgeTypeNotFound("KNOWS".to_string()).to_string(),
            "Edge type not found: KNOWS"
        );
        let idx = IndexId::new(42);
        assert!(CatalogError::IndexNotFound(idx).to_string().contains("42"));
    }

    #[test]
    fn test_catalog_concurrent_label_creation()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;

        let catalog = Arc::new(Catalog::new());
        let mut handles = vec![];

        // Spawn multiple threads trying to create the same labels
        for i in 0..10 {
            let catalog = Arc::clone(&catalog);
            handles.push(thread::spawn(move || {
                let label_name = format!("Label{}", i % 3); // Only 3 unique labels
                catalog.get_or_create_label(&label_name)
            }));
        }

        let mut ids: Vec<LabelId> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>, _>>()?;
        ids.sort_by_key(|id| id.as_u32());
        ids.dedup();

        // Should only have 3 unique label IDs
        assert_eq!(ids.len(), 3);
        assert_eq!(catalog.label_count(), 3);
        Ok(())
    }

    #[test]
    fn test_catalog_concurrent_property_key_creation()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;

        let catalog = Arc::new(Catalog::new());
        let mut handles = vec![];

        for i in 0..10 {
            let catalog = Arc::clone(&catalog);
            handles.push(thread::spawn(move || {
                let key_name = format!("key{}", i % 4);
                catalog.get_or_create_property_key(&key_name)
            }));
        }

        let mut ids: Vec<PropertyKeyId> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>, _>>()?;
        ids.sort_by_key(|id| id.as_u32());
        ids.dedup();

        assert_eq!(ids.len(), 4);
        assert_eq!(catalog.property_key_count(), 4);
        Ok(())
    }

    #[test]
    fn test_catalog_concurrent_index_operations()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;

        let catalog = Arc::new(Catalog::new());
        let label = catalog.get_or_create_label("Node")?;

        let mut handles = vec![];

        // Create indexes concurrently
        for i in 0..5 {
            let catalog = Arc::clone(&catalog);
            handles.push(thread::spawn(move || {
                let prop = catalog.get_or_create_property_key(&format!("property_{i}"))?;
                catalog.create_index(
                    Some(&format!("idx_{i}")),
                    label,
                    prop,
                    GraphPath::root(),
                    IndexConfiguration::Property,
                )
            }));
        }

        let ids: Vec<IndexId> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(ids.len(), 5);
        assert_eq!(catalog.index_count(), 5);
        Ok(())
    }

    #[test]
    fn test_catalog_special_characters_in_names()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        // Test with various special characters
        let label1 = catalog.get_or_create_label("Label With Spaces")?;
        let label2 = catalog.get_or_create_label("Label-With-Dashes")?;
        let label3 = catalog.get_or_create_label("Label_With_Underscores")?;
        let label4 = catalog.get_or_create_label("LabelWithUnicode\u{00E9}")?;

        assert_ne!(label1, label2);
        assert_ne!(label2, label3);
        assert_ne!(label3, label4);

        assert_eq!(
            catalog.get_label_name(label1).as_deref(),
            Some("Label With Spaces")
        );
        assert_eq!(
            catalog.get_label_name(label4).as_deref(),
            Some("LabelWithUnicode\u{00E9}")
        );
        Ok(())
    }

    #[test]
    fn test_catalog_empty_names() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();

        // Empty names should be valid (edge case)
        let empty_label = catalog.get_or_create_label("")?;
        let empty_prop = catalog.get_or_create_property_key("")?;
        let empty_edge = catalog.get_or_create_edge_type("")?;

        assert_eq!(catalog.get_label_name(empty_label).as_deref(), Some(""));
        assert_eq!(
            catalog.get_property_key_name(empty_prop).as_deref(),
            Some("")
        );
        assert_eq!(catalog.get_edge_type_name(empty_edge).as_deref(), Some(""));

        // Calling again should return same ID
        assert_eq!(catalog.get_or_create_label("")?, empty_label);
        Ok(())
    }

    #[test]
    fn test_catalog_large_number_of_entries() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let catalog = Catalog::new();

        // Create many labels
        for i in 0..1000 {
            catalog.get_or_create_label(&format!("Label{}", i))?;
        }

        assert_eq!(catalog.label_count(), 1000);

        // Verify we can retrieve them all
        let all = catalog.all_labels();
        assert_eq!(all.len(), 1000);

        // Verify a specific one
        let id = catalog.get_label_id("Label500").unwrap();
        assert_eq!(catalog.get_label_name(id).as_deref(), Some("Label500"));
        Ok(())
    }

    #[test]
    fn test_index_definition_debug() {
        let def = IndexDefinition {
            id: IndexId::new(1),
            name: "test_index".to_string(),
            label: LabelId::new(2),
            property_key: PropertyKeyId::new(3),
            key: PhysicalIndexKey::property(GraphPath::root(), "value"),
            configuration: IndexConfiguration::Property,
            index_type: IndexType::Hash,
        };

        // Should be able to debug print
        let debug_str = format!("{:?}", def);
        assert!(debug_str.contains("IndexDefinition"));
        assert!(debug_str.contains("Hash"));
    }

    #[test]
    fn test_index_type_equality() {
        assert_eq!(IndexType::Hash, IndexType::Hash);
        assert_ne!(IndexType::Hash, IndexType::BTree);
        assert_ne!(IndexType::BTree, IndexType::FullText);

        // Clone
        let t = IndexType::Hash;
        let t2 = t;
        assert_eq!(t, t2);
    }

    #[test]
    fn test_catalog_error_equality() {
        assert_eq!(
            CatalogError::SchemaNotEnabled,
            CatalogError::SchemaNotEnabled
        );
        assert_eq!(
            CatalogError::ConstraintAlreadyExists,
            CatalogError::ConstraintAlreadyExists
        );
        assert_eq!(
            CatalogError::LabelNotFound("X".to_string()),
            CatalogError::LabelNotFound("X".to_string())
        );
        assert_ne!(
            CatalogError::LabelNotFound("X".to_string()),
            CatalogError::LabelNotFound("Y".to_string())
        );
    }

    #[test]
    fn catalog_wal_state_v1_roundtrips_complete_exact_post_image()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Catalog::new();
        let person = source.get_or_create_label("Person")?;
        let email = source.get_or_create_property_key("email")?;
        let tenant = source.get_or_create_property_key("tenant")?;
        source.get_or_create_edge_type("WORKS_AT")?;
        source
            .register_node_type(NodeTypeDefinition {
                name: "Person".to_string(),
                properties: vec![
                    TypedProperty {
                        name: "email".to_string(),
                        data_type: PropertyDataType::String,
                        nullable: false,
                        default_value: Some(Value::String("unknown@example.test".into())),
                    },
                    TypedProperty {
                        name: "scores".to_string(),
                        data_type: PropertyDataType::ListTyped(Box::new(PropertyDataType::Int64)),
                        nullable: true,
                        default_value: Some(Value::List(vec![Value::Int64(7)].into())),
                    },
                ],
                constraints: vec![TypeConstraint::Check {
                    name: Some("email_shape".to_string()),
                    expression: "email IS NOT NULL".to_string(),
                }],
                parent_types: vec!["Entity".to_string()],
            })
            .unwrap();
        source
            .register_node_type(NodeTypeDefinition {
                name: "Company".to_string(),
                properties: Vec::new(),
                constraints: Vec::new(),
                parent_types: vec!["Entity".to_string()],
            })
            .unwrap();
        source
            .register_edge_type_def(EdgeTypeDefinition {
                name: "WORKS_AT".to_string(),
                properties: vec![TypedProperty {
                    name: "since".to_string(),
                    data_type: PropertyDataType::Int64,
                    nullable: false,
                    default_value: Some(Value::Int64(2026)),
                }],
                constraints: vec![TypeConstraint::NotNull("since".to_string())],
                source_node_types: vec!["Person".to_string()],
                target_node_types: vec!["Company".to_string()],
            })
            .unwrap();
        source
            .register_graph_type(GraphTypeDefinition {
                name: "Employment".to_string(),
                allowed_node_types: vec!["Person".to_string(), "Company".to_string()],
                allowed_edge_types: vec!["WORKS_AT".to_string()],
                open: false,
            })
            .unwrap();
        source.register_schema_namespace("hr".to_string()).unwrap();
        source
            .bind_graph_type(
                &GraphPath::from_components(&["hr/main"])?,
                "Employment".to_string(),
            )
            .unwrap();
        source
            .register_procedure(ProcedureDefinition {
                name: "find_people".to_string(),
                params: vec![("tenant".to_string(), "STRING".to_string())],
                returns: vec![("email".to_string(), "STRING".to_string())],
                body: "MATCH (p:Person) RETURN p.email".to_string(),
            })
            .unwrap();
        source
            .create_named_constraint(NamedConstraintDefinition {
                name: "person_identity".to_string(),
                label: "Person".to_string(),
                properties: vec!["tenant".to_string(), "email".to_string()],
                kind: NamedConstraintKind::NodeKey,
            })
            .unwrap();
        source.add_unique_constraint(person, tenant).unwrap_err();
        assert!(source.is_property_unique(person, email));
        assert!(source.is_property_required(person, tenant));

        let encoded = source.encode_wal_state_v1().unwrap();
        assert_eq!(source.encode_wal_state_v1().unwrap(), encoded);

        let restored = Catalog::new();
        restored.get_or_create_label("stale")?;
        restored
            .register_procedure(ProcedureDefinition {
                name: "stale".to_string(),
                params: Vec::new(),
                returns: Vec::new(),
                body: String::new(),
            })
            .unwrap();
        restored.restore_wal_state_v1(&encoded).unwrap();

        assert_eq!(restored.encode_wal_state_v1().unwrap(), encoded);
        assert_eq!(restored.get_label_name(person).as_deref(), Some("Person"));
        assert!(restored.get_label_id("stale").is_none());
        let node = restored.get_node_type("Person").unwrap();
        assert!(restored.get_node_type("Company").is_some());
        assert_eq!(node.parent_types, ["Entity"]);
        assert_eq!(
            node.properties[0].default_value,
            Some(Value::String("unknown@example.test".into()))
        );
        assert_eq!(
            node.properties[1].data_type,
            PropertyDataType::ListTyped(Box::new(PropertyDataType::Int64))
        );
        let edge = restored.get_edge_type_def("WORKS_AT").unwrap();
        assert_eq!(edge.source_node_types, ["Person"]);
        assert_eq!(edge.target_node_types, ["Company"]);
        assert_eq!(edge.properties[0].default_value, Some(Value::Int64(2026)));
        assert_eq!(
            restored
                .get_graph_type_binding(&GraphPath::from_components(&["hr/main"])?)
                .as_deref(),
            Some("Employment")
        );
        assert_eq!(
            restored.get_procedure("find_people").unwrap().body,
            "MATCH (p:Person) RETURN p.email"
        );
        assert!(restored.get_procedure("stale").is_none());
        assert_eq!(
            restored
                .get_named_constraint("person_identity")
                .unwrap()
                .kind,
            NamedConstraintKind::NodeKey
        );
        Ok(())
    }

    #[test]
    fn ddl_comparison_keeps_native_binding_identity_without_weakening_wire()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog::new();
        for (name, open) in [("Open", true), ("Closed", false)] {
            catalog.register_graph_type(GraphTypeDefinition {
                name: name.to_string(),
                allowed_node_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                open,
            })?;
        }
        let nested = GraphPath::from_components(&["a", "b"])?;
        catalog.bind_graph_type(&nested, "Open".to_string())?;
        catalog.bind_graph_type(&GraphPath::root(), "Open".to_string())?;
        let before = catalog.ddl_comparison_state()?;
        assert_eq!(catalog.ddl_comparison_state()?, before);
        assert_eq!(catalog.read().ddl_comparison_state()?, before);
        assert_eq!(
            before.1,
            vec![
                (GraphPath::root(), "Open".to_string()),
                (nested.clone(), "Open".to_string()),
            ]
        );
        assert!(catalog.encode_wal_state_v1().is_err());

        catalog.bind_graph_type(&nested, "Closed".to_string())?;
        let binding_changed = catalog.ddl_comparison_state()?;
        assert_eq!(binding_changed.0, before.0);
        assert_ne!(binding_changed.1, before.1);
        catalog.register_graph_type(GraphTypeDefinition {
            name: "Additional".to_string(),
            allowed_node_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            open: true,
        })?;
        let type_changed = catalog.ddl_comparison_state()?;
        assert_ne!(type_changed.0, binding_changed.0);
        assert_eq!(type_changed.1, binding_changed.1);
        assert!(catalog.encode_wal_state_v1().is_err());
        assert_eq!(
            catalog.get_graph_type_binding(&nested).as_deref(),
            Some("Closed")
        );
        Ok(())
    }

    #[test]
    fn graph_type_bindings_and_validator_preserve_exact_paths()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = Arc::new(Catalog::new());
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["default"])?,
            GraphPath::from_components(&["schema/graph"])?,
            GraphPath::from_components(&["schema", "graph"])?,
        ];
        for (position, path) in paths.iter().enumerate() {
            let node_type = format!("owner{position}/Person");
            let graph_type = format!("owner{position}/Graph");
            catalog.register_node_type(NodeTypeDefinition {
                name: node_type.clone(),
                properties: Vec::new(),
                constraints: Vec::new(),
                parent_types: Vec::new(),
            })?;
            catalog.register_graph_type(GraphTypeDefinition {
                name: graph_type.clone(),
                allowed_node_types: vec![node_type],
                allowed_edge_types: Vec::new(),
                open: false,
            })?;
            catalog.bind_graph_type(path, graph_type)?;
        }
        let bindings = catalog.all_graph_type_bindings();
        assert_eq!(bindings.len(), paths.len());
        let snapshot = catalog.snapshot_state();
        let schema = snapshot.schema.ok_or("schema snapshot missing")?;
        assert_eq!(schema.graph_type_bindings.len(), paths.len());
        for (position, path) in paths.iter().enumerate() {
            let graph_type = format!("owner{position}/Graph");
            assert_eq!(
                catalog.get_graph_type_binding(path),
                Some(graph_type.clone())
            );
            assert!(catalog.graph_type_binding_matches(path, Some(&graph_type)));
            assert_eq!(schema.graph_type_bindings.get(path), Some(&graph_type));
            assert!(bindings.contains(&(path.clone(), graph_type)));
            let validator = CatalogConstraintValidator::new(Arc::clone(&catalog))
                .with_graph_path(path.clone())
                .with_schema(Some("unrelated".to_string()));
            assert_eq!(
                validator.node_type_key("Person"),
                format!("owner{position}/Person")
            );
            assert!(
                validator
                    .validate_node_labels_allowed(&["Person".to_string()])
                    .is_ok()
            );
            assert!(
                validator
                    .validate_node_labels_allowed(&["Other".to_string()])
                    .is_err()
            );
        }
        assert!(!catalog.publish_graph_type_binding_if_same(&paths[3], Some("wrong"), None));
        assert!(catalog.publish_graph_type_binding_if_same(&paths[3], Some("owner3/Graph"), None));
        assert_eq!(catalog.get_graph_type_binding(&paths[3]), None);
        assert_eq!(
            catalog.get_graph_type_binding(&paths[4]).as_deref(),
            Some("owner4/Graph")
        );
        assert!(matches!(
            catalog.drop_graph_type("owner4/Graph"),
            Err(CatalogError::GraphTypeHasBindings(_))
        ));
        Ok(())
    }

    #[test]
    fn catalog_wal_state_v1_rejects_root_and_nested_graph_type_bindings()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for path in [GraphPath::root(), GraphPath::from_components(&["a", "b"])?] {
            let catalog = Catalog::new();
            catalog.register_graph_type(GraphTypeDefinition {
                name: "Graph".to_string(),
                allowed_node_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                open: true,
            })?;
            catalog.bind_graph_type(&path, "Graph".to_string())?;
            let before = catalog.all_graph_type_bindings();
            assert_eq!(
                catalog.encode_wal_state_v1(),
                Err(
                    "catalog WAL state v1 cannot encode root or nested graph-type bindings"
                        .to_string()
                )
            );
            assert_eq!(catalog.all_graph_type_bindings(), before);
        }
        Ok(())
    }

    #[test]
    fn catalog_wal_state_v1_keeps_single_component_graph_bindings_literal()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Catalog::new();
        source.register_graph_type(GraphTypeDefinition {
            name: "Graph".to_string(),
            allowed_node_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            open: true,
        })?;
        for name in ["", "default", "a/b", "é\0"] {
            source.bind_graph_type(&GraphPath::from_components(&[name])?, "Graph".to_string())?;
        }
        let encoded = source.encode_wal_state_v1()?;
        let restored = Catalog::new();
        restored.restore_wal_state_v1(&encoded)?;
        assert_eq!(restored.encode_wal_state_v1()?, encoded);
        assert_eq!(restored.all_graph_type_bindings().len(), 4);
        for name in ["", "default", "a/b", "é\0"] {
            assert_eq!(
                restored
                    .get_graph_type_binding(&GraphPath::from_components(&[name])?)
                    .as_deref(),
                Some("Graph")
            );
        }
        assert_eq!(restored.get_graph_type_binding(&GraphPath::root()), None);
        assert_eq!(
            restored.get_graph_type_binding(&GraphPath::from_components(&["a", "b"])?),
            None
        );

        let before = restored.encode_wal_state_v1()?;
        let oversized = "x".repeat(grafeo_common::types::MAX_WORLD_GRAPH_NAME_BYTES + 1);
        for bindings in [
            vec![(oversized, "Graph".to_string())],
            vec![("same".to_string(), "Graph".to_string()); 2],
        ] {
            let mut state = CatalogWalStateV1::from_snapshot(source.snapshot_state())?;
            state.schema.graph_type_bindings = bindings;
            let invalid = bincode::serde::encode_to_vec(&state, bincode::config::standard())?;
            assert!(restored.restore_wal_state_v1(&invalid).is_err());
            assert_eq!(restored.encode_wal_state_v1()?, before);
        }
        Ok(())
    }

    #[test]
    fn catalog_wal_state_v1_rejects_trailing_and_invalid_dictionary_ids()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Catalog::new();
        let label = source.get_or_create_label("Person")?;
        let property = source.get_or_create_property_key("email")?;
        source.add_unique_constraint(label, property).unwrap();

        let mut trailing = source.encode_wal_state_v1().unwrap();
        trailing.push(0);
        let target = Catalog::new();
        let before = target.encode_wal_state_v1().unwrap();
        assert!(target.restore_wal_state_v1(&trailing).is_err());
        assert_eq!(target.encode_wal_state_v1().unwrap(), before);

        let mut state = CatalogWalStateV1::from_snapshot(source.snapshot_state()).unwrap();
        state.schema.unique_constraints = vec![(99, 0)];
        let invalid = bincode::serde::encode_to_vec(&state, bincode::config::standard()).unwrap();
        assert!(target.restore_wal_state_v1(&invalid).is_err());
        assert_eq!(target.encode_wal_state_v1().unwrap(), before);
        Ok(())
    }

    #[test]
    fn legacy_constraint_owners_are_adopted_without_losing_enforcement() {
        let catalog = Catalog::new();
        catalog.register_or_replace_node_type(NodeTypeDefinition {
            name: "hr/Employee".to_string(),
            properties: Vec::new(),
            constraints: vec![
                TypeConstraint::Unique(vec!["tenant".to_string(), "email".to_string()]),
                TypeConstraint::NotNull("tenant".to_string()),
                TypeConstraint::NotNull("email".to_string()),
            ],
            parent_types: Vec::new(),
        });

        let synthetic_unique = NamedConstraintDefinition {
            name: String::new(),
            label: "hr/Employee".to_string(),
            properties: vec!["tenant".to_string(), "email".to_string()],
            kind: NamedConstraintKind::Unique,
        };
        let mut synthetic_unique = synthetic_unique;
        synthetic_unique.name = authentic_legacy_constraint_owner_name(&synthetic_unique);
        catalog
            .restore_named_constraint(synthetic_unique.clone())
            .unwrap();

        let mut synthetic_required = Vec::new();
        for property in ["tenant", "email"] {
            let mut definition = NamedConstraintDefinition {
                name: String::new(),
                label: "hr/Employee".to_string(),
                properties: vec![property.to_string()],
                kind: NamedConstraintKind::NotNull,
            };
            definition.name = authentic_legacy_constraint_owner_name(&definition);
            catalog
                .restore_named_constraint(definition.clone())
                .unwrap();
            synthetic_required.push(definition);
        }

        catalog
            .restore_named_constraint_from_wal(NamedConstraintDefinition {
                name: "hr/employee_identity".to_string(),
                label: "hr/Employee".to_string(),
                properties: vec!["tenant".to_string(), "email".to_string()],
                kind: NamedConstraintKind::Unique,
            })
            .unwrap();
        catalog
            .restore_named_constraint_from_wal(NamedConstraintDefinition {
                name: "hr/employee_required".to_string(),
                label: "hr/Employee".to_string(),
                properties: vec!["tenant".to_string(), "email".to_string()],
                kind: NamedConstraintKind::Exists,
            })
            .unwrap();

        assert!(
            catalog
                .get_named_constraint(&synthetic_unique.name)
                .is_none()
        );
        assert!(
            synthetic_required
                .iter()
                .all(|definition| catalog.get_named_constraint(&definition.name).is_none())
        );
        assert!(
            catalog
                .get_named_constraint("hr/employee_identity")
                .is_some()
        );
        assert!(
            catalog
                .get_named_constraint("hr/employee_required")
                .is_some()
        );

        let label = catalog.get_label_id("Employee").unwrap();
        let tenant = catalog.get_property_key_id("tenant").unwrap();
        assert!(catalog.is_property_unique(label, tenant));
        assert!(catalog.is_property_required(label, tenant));
        catalog
            .drop_named_constraint("hr/employee_identity")
            .unwrap();
        catalog
            .drop_named_constraint("hr/employee_required")
            .unwrap();
        assert!(!catalog.is_property_unique(label, tenant));
        assert!(!catalog.is_property_required(label, tenant));
    }
}
