//! Snapshot10's Catalog7 and exact Text5/Vector4 transport lane.
//! Graph histories and endpoint policy remain in the portable adapter.

#[cfg(feature = "triple-store")]
use super::SNAPSHOT_RDF_HISTORY_PAYLOAD_VERSION;
#[cfg(all(feature = "triple-store", feature = "lpg"))]
use super::decode_rdf_lpg_projections;
use super::{SNAPSHOT_VERSION, Snapshot, resolved_snapshot_graph_model};
#[cfg(feature = "lpg")]
use super::{SchemaMergePolicy, SnapshotSchema};
use crate::catalog::Catalog;
#[cfg(feature = "lpg")]
use crate::catalog::{
    CatalogRead, EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
};
use crate::config::GraphModel;
#[cfg(feature = "lpg")]
use crate::database::{catalog_section::CatalogSection, index_sections::LpgIndexGraphCut};
#[cfg(all(feature = "lpg", any(feature = "text-index", feature = "vector-index")))]
use grafeo_common::storage::Section;
#[cfg(all(feature = "lpg", feature = "triple-store"))]
use grafeo_common::types::NodeId;
use grafeo_common::types::{
    AuthoritativeFormat, EpochId, GraphModelTag, GraphPath, ModelFormatVersion, ProjectionCut,
    SchemaCut, WorldCutDescriptor,
};
use grafeo_common::utils::error::{Error, Result};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{LpgStore, PhysicalIndexFamily, PhysicalIndexKey};
#[cfg(feature = "lpg")]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "lpg")]
use std::borrow::Cow;
#[cfg(feature = "lpg")]
use std::ops::Range;
#[cfg(feature = "lpg")]
use std::sync::Arc;

#[cfg(feature = "lpg")]
pub(super) fn normalize_schema_bytes(schema: &SnapshotSchema) -> Result<Vec<u8>> {
    let mut node_types = schema.node_types.clone();
    node_types.sort_by(|a, b| a.name.cmp(&b.name));
    let mut edge_types = schema.edge_types.clone();
    edge_types.sort_by(|a, b| a.name.cmp(&b.name));
    let mut graph_types = schema.graph_types.clone();
    graph_types.sort_by(|a, b| a.name.cmp(&b.name));
    let mut procedures = schema.procedures.clone();
    procedures.sort_by(|a, b| a.name.cmp(&b.name));
    let mut schemas = schema.schemas.clone();
    schemas.sort();
    let mut bindings = schema.graph_type_bindings.clone();
    bindings.sort();

    let bindings: Vec<_> = bindings
        .iter()
        .map(|(path, name)| (path.components(), name))
        .collect();
    bincode::serde::encode_to_vec(
        (
            node_types,
            edge_types,
            graph_types,
            procedures,
            schemas,
            bindings,
        ),
        bincode::config::standard(),
    )
    .map_err(|error| Error::Serialization(format!("cannot normalize snapshot schema: {error}")))
}

#[cfg(feature = "lpg")]
pub(super) fn merge_snapshot_schemas(
    snapshots: &[Snapshot],
    policy: SchemaMergePolicy,
) -> Result<SnapshotSchema> {
    let source_schemas = snapshots
        .iter()
        .map(|snapshot| {
            stage_snapshot_catalog(snapshot).and_then(|catalog| collect_schema(&catalog))
        })
        .collect::<Result<Vec<_>>>()?;
    let first = source_schemas
        .first()
        .ok_or_else(|| Error::Serialization("snapshot schema union is empty".into()))?;
    match policy {
        SchemaMergePolicy::StrictEquality => {
            let canonical = normalize_schema_bytes(first)?;
            for (idx, snap) in source_schemas.iter().enumerate().skip(1) {
                if normalize_schema_bytes(snap)? != canonical {
                    return Err(Error::Internal(format!(
                        "snapshot[{idx}] schema does not match snapshot[0] schema \
                         (SchemaMergePolicy::StrictEquality); all snapshots must \
                         declare byte-identical schemas after canonical ordering"
                    )));
                }
            }
            Ok(first.clone())
        }
        SchemaMergePolicy::UnionWithConflictCheck => {
            // Each typed catalog field is HashMap<name, def>. Inserting
            // again with the same name requires the definitions to be
            // byte-equal (post-normalization within each definition).
            // Same-name+different-shape rejects with a diagnostic.

            let mut node_types: HashMap<String, NodeTypeDefinition> = HashMap::new();
            let mut edge_types: HashMap<String, EdgeTypeDefinition> = HashMap::new();
            let mut graph_types: HashMap<String, GraphTypeDefinition> = HashMap::new();
            let mut procedures: HashMap<String, ProcedureDefinition> = HashMap::new();
            let mut schemas: HashSet<String> = HashSet::new();
            let mut bindings: HashMap<GraphPath, String> = HashMap::new();

            let cfg = bincode::config::standard();

            for (idx, snap) in source_schemas.iter().enumerate() {
                for def in &snap.node_types {
                    match node_types.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines NodeType {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for def in &snap.edge_types {
                    match edge_types.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines EdgeType {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for def in &snap.graph_types {
                    match graph_types.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines GraphType {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for def in &snap.procedures {
                    match procedures.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).map_err(|error| {
                                    Error::Serialization(format!(
                                        "cannot encode schema entry: {error}"
                                    ))
                                })?;
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines Procedure {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for s in &snap.schemas {
                    schemas.insert(s.clone());
                }
                for (gname, gtype) in &snap.graph_type_bindings {
                    match bindings.entry(gname.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            if existing.get() != gtype {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] binds graph {gname:?} to \
                                     {gtype:?} but an earlier snapshot bound it \
                                     to {:?}",
                                    existing.get()
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(gtype.clone());
                        }
                    }
                }
            }

            let mut merged = SnapshotSchema {
                node_types: node_types.into_values().collect(),
                edge_types: edge_types.into_values().collect(),
                graph_types: graph_types.into_values().collect(),
                procedures: procedures.into_values().collect(),
                schemas: schemas.into_iter().collect(),
                graph_type_bindings: bindings.into_iter().collect(),
            };
            // Sort the merged vecs so the result is deterministic and
            // the eventual round-trip through export_snapshot is
            // reproducible.
            merged.node_types.sort_by(|a, b| a.name.cmp(&b.name));
            merged.edge_types.sort_by(|a, b| a.name.cmp(&b.name));
            merged.graph_types.sort_by(|a, b| a.name.cmp(&b.name));
            merged.procedures.sort_by(|a, b| a.name.cmp(&b.name));
            merged.schemas.sort();
            merged.graph_type_bindings.sort();
            Ok(merged)
        }
    }
}

#[cfg(feature = "lpg")]
pub(super) fn restore_schema_from_snapshot(
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &Catalog,
    schema: &SnapshotSchema,
) -> Result<()> {
    for def in &schema.node_types {
        catalog.register_or_replace_node_type(def.clone());
    }
    for def in &schema.edge_types {
        catalog.register_or_replace_edge_type_def(def.clone());
    }
    for def in &schema.graph_types {
        catalog
            .register_graph_type(def.clone())
            .map_err(|error| Error::Serialization(error.to_string()))?;
    }
    for def in &schema.procedures {
        catalog
            .replace_procedure(def.clone())
            .map_err(|error| Error::Serialization(error.to_string()))?;
    }
    for name in &schema.schemas {
        catalog
            .register_schema_namespace(name.clone())
            .map_err(|error| Error::Serialization(error.to_string()))?;
        // Ensure the schema's default graph partition exists
        let default_key = format!("{name}/__default__");
        store.create_graph(&default_key)?;
    }
    for (path, type_name) in &schema.graph_type_bindings {
        catalog
            .bind_graph_type(path, type_name.clone())
            .map_err(|error| Error::Serialization(error.to_string()))?;
    }
    Ok(())
}

#[cfg(feature = "lpg")]
pub(super) fn collect_schema(catalog: &Catalog) -> Result<SnapshotSchema> {
    let catalog = catalog.read();
    collect_schema_at(catalog.view())
}

#[cfg(feature = "lpg")]
pub(super) fn collect_schema_at(catalog: CatalogRead<'_>) -> Result<SnapshotSchema> {
    let mut schema = SnapshotSchema {
        node_types: catalog.all_node_type_defs(),
        edge_types: catalog.all_edge_type_defs(),
        graph_types: catalog.all_graph_type_defs(),
        procedures: catalog.all_procedure_defs(),
        schemas: catalog.schema_names(),
        graph_type_bindings: catalog.all_graph_type_bindings(),
    };
    schema
        .node_types
        .sort_by(|left, right| left.name.cmp(&right.name));
    schema
        .edge_types
        .sort_by(|left, right| left.name.cmp(&right.name));
    schema
        .graph_types
        .sort_by(|left, right| left.name.cmp(&right.name));
    schema
        .procedures
        .sort_by(|left, right| left.name.cmp(&right.name));
    schema.schemas.sort();
    schema.graph_type_bindings.sort();
    Ok(schema)
}

pub(super) fn snapshot_world_descriptor(snapshot: &Snapshot) -> Result<WorldCutDescriptor> {
    if snapshot.version != SNAPSHOT_VERSION {
        return Err(Error::Serialization(format!(
            "integrity-sealed snapshot artifacts require snapshot v{SNAPSHOT_VERSION}; got v{}",
            snapshot.version
        )));
    }
    let graph_model = resolved_snapshot_graph_model(snapshot)?;
    let graph_model_tag = GraphModelTag::from_u8(graph_model.as_u8()).map_err(|error| {
        Error::Serialization(format!(
            "invalid snapshot graph model for world cut: {error}"
        ))
    })?;

    let catalog = stage_snapshot_catalog(snapshot)?;
    let canonical_catalog =
        crate::database::catalog_wire::encode_catalog_read(catalog.read().view(), snapshot.epoch)?;
    if canonical_catalog != snapshot.catalog_state {
        return Err(Error::Serialization(
            "snapshot catalog state is valid but not canonically encoded".to_string(),
        ));
    }
    let logical_catalog = catalog.encode_current_state_v2().map_err(|error| {
        Error::Serialization(format!(
            "canonicalize snapshot logical catalog for world cut: {error}"
        ))
    })?;
    let schema = SchemaCut::from_canonical_post_image(
        u16::from(crate::catalog::CURRENT_CATALOG_STATE_VERSION),
        &logical_catalog,
    )
    .map_err(|error| Error::Serialization(format!("seal snapshot schema: {error}")))?;

    let mut formats = vec![
        ModelFormatVersion::new(AuthoritativeFormat::Catalog, 7),
        ModelFormatVersion::new(AuthoritativeFormat::Cdc, 1),
        ModelFormatVersion::new(
            AuthoritativeFormat::PortableSnapshot,
            u16::from(SNAPSHOT_VERSION),
        ),
    ]
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()
    .map_err(|error| Error::Serialization(format!("seal snapshot formats: {error}")))?;
    if matches!(graph_model, GraphModel::Lpg | GraphModel::Both) {
        formats.push(
            ModelFormatVersion::new(AuthoritativeFormat::Lpg, u16::from(SNAPSHOT_VERSION))
                .map_err(|error| {
                    Error::Serialization(format!("seal LPG snapshot format: {error}"))
                })?,
        );
    }
    if matches!(graph_model, GraphModel::Rdf | GraphModel::Both) {
        #[cfg(not(feature = "triple-store"))]
        return Err(Error::Serialization(
            "RDF snapshot descriptor requires triple-store support".to_string(),
        ));
        #[cfg(feature = "triple-store")]
        {
            formats.push(
                ModelFormatVersion::new(AuthoritativeFormat::Rdf, u16::from(SNAPSHOT_VERSION))
                    .map_err(|error| {
                        Error::Serialization(format!("seal RDF snapshot format: {error}"))
                    })?,
            );
            formats.push(
                ModelFormatVersion::new(
                    AuthoritativeFormat::RdfHistory,
                    u16::from(SNAPSHOT_RDF_HISTORY_PAYLOAD_VERSION),
                )
                .map_err(|error| {
                    Error::Serialization(format!("seal RDF history format: {error}"))
                })?,
            );
        }
    }

    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    let projections = decode_rdf_lpg_projections(
        &snapshot.rdf_lpg_projections,
        snapshot.world_identity.store_id(),
    )?
    .into_iter()
    .filter_map(|definition| definition.receipt().cloned())
    .map(|receipt| {
        receipt
            .to_world_cut()
            .map_err(|error| Error::Serialization(format!("seal projection receipt: {error}")))
    })
    .collect::<Result<Vec<ProjectionCut>>>()?;
    #[cfg(not(all(feature = "triple-store", feature = "lpg")))]
    let projections = Vec::<ProjectionCut>::new();

    WorldCutDescriptor::new(
        snapshot.world_identity.store_id(),
        EpochId::new(snapshot.epoch),
        graph_model_tag,
        formats,
        schema,
        projections,
        snapshot.world_identity.history(),
    )
    .map_err(|error| Error::Serialization(format!("invalid snapshot world cut: {error}")))
}

#[cfg(feature = "lpg")]
pub(super) fn merge_snapshot_catalogs(
    snapshots: &[Snapshot],
    merged_schema: &SnapshotSchema,
) -> Result<Catalog> {
    use std::collections::{BTreeMap, BTreeSet};

    let sources = snapshots
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            stage_snapshot_catalog(snapshot).map_err(|error| {
                Error::Serialization(format!("snapshot[{index}] catalog: {error}"))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let merged = Catalog::new();
    let scratch = std::sync::Arc::new(
        grafeo_core::graph::lpg::LpgStore::new()
            .map_err(|error| Error::Internal(error.to_string()))?,
    );
    restore_schema_from_snapshot(&scratch, &merged, merged_schema)?;

    let mut labels = BTreeSet::new();
    let mut properties = BTreeSet::new();
    let mut edge_types = BTreeSet::new();
    let mut unique = BTreeSet::new();
    let mut required = BTreeSet::new();
    let mut constraints: BTreeMap<String, (usize, crate::catalog::NamedConstraintDefinition)> =
        BTreeMap::new();
    let mut targets: BTreeMap<(String, String, Vec<String>), (usize, String)> = BTreeMap::new();

    for (source_index, catalog) in sources.iter().enumerate() {
        labels.extend(
            catalog
                .all_labels()
                .into_iter()
                .map(|name| name.to_string()),
        );
        properties.extend(
            catalog
                .all_property_keys()
                .into_iter()
                .map(|name| name.to_string()),
        );
        edge_types.extend(
            catalog
                .all_edge_types()
                .into_iter()
                .map(|name| name.to_string()),
        );
        unique.extend(catalog.all_unique_constraints());
        required.extend(catalog.all_required_properties());

        for definition in catalog.all_named_constraints() {
            if let Some((existing_source, existing)) = constraints.get(&definition.name) {
                if existing != &definition {
                    return Err(Error::Serialization(format!(
                        "open_multi: named constraint '{}' conflicts between snapshot[{existing_source}] and snapshot[{source_index}]",
                        definition.name
                    )));
                }
                continue;
            }
            let mut canonical_properties = definition.properties.clone();
            canonical_properties.sort();
            let target = (
                definition.label.clone(),
                definition.kind.as_str().to_string(),
                canonical_properties,
            );
            if let Some((existing_source, existing_name)) = targets.get(&target) {
                return Err(Error::Serialization(format!(
                    "open_multi: named constraints '{existing_name}' (snapshot[{existing_source}]) and '{}' (snapshot[{source_index}]) claim the same canonical target",
                    definition.name
                )));
            }
            targets.insert(target, (source_index, definition.name.clone()));
            constraints.insert(definition.name.clone(), (source_index, definition));
        }
    }

    for label in labels {
        merged
            .get_or_create_label(&label)
            .map_err(|error| Error::Serialization(error.to_string()))?;
    }
    for property in properties {
        merged
            .get_or_create_property_key(&property)
            .map_err(|error| Error::Serialization(error.to_string()))?;
    }
    for edge_type in edge_types {
        merged
            .get_or_create_edge_type(&edge_type)
            .map_err(|error| Error::Serialization(error.to_string()))?;
    }
    for (_, definition) in constraints.into_values() {
        merged
            .restore_named_constraint(definition)
            .map_err(|error| {
                Error::Serialization(format!("open_multi: restore named constraint: {error}"))
            })?;
    }
    for (label, property) in unique {
        let label = merged
            .get_or_create_label(&label)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        let property = merged
            .get_or_create_property_key(&property)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        if !merged.is_property_unique(label, property) {
            merged
                .add_unique_constraint(label, property)
                .map_err(|error| {
                    Error::Serialization(format!("open_multi: restore unique constraint: {error}"))
                })?;
        }
    }
    for (label, property) in required {
        let label = merged
            .get_or_create_label(&label)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        let property = merged
            .get_or_create_property_key(&property)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        if !merged.is_property_required(label, property) {
            merged
                .add_required_property(label, property)
                .map_err(|error| {
                    Error::Serialization(format!("open_multi: restore required property: {error}"))
                })?;
        }
    }
    merged
        .merge_current_index_owners(&sources)
        .map_err(Error::Serialization)
}

/// One exact family image and the owner-qualified target subset it describes.
#[cfg(feature = "lpg")]
struct SnapshotIndexChunk<'a> {
    family: PhysicalIndexFamily,
    keys: Vec<PhysicalIndexKey>,
    bytes: Cow<'a, [u8]>,
}

/// Retains immutable source bytes until every unpublished target has imported.
/// Whole chunks stay borrowed; only partially retained union chunks are copied.
#[cfg(feature = "lpg")]
pub(super) struct SnapshotIndexImages<'a> {
    chunks: Vec<SnapshotIndexChunk<'a>>,
}

#[cfg(feature = "lpg")]
pub(super) fn capture_snapshot_indexes(
    catalog: &Arc<Catalog>,
    view: CatalogRead<'_>,
    graphs: Vec<(GraphPath, Arc<LpgStore>)>,
    epoch: u64,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if graphs.is_empty() {
        if view.index_allocator_high_water() != 0 || !view.all_graph_type_bindings().is_empty() {
            return Err(Error::Serialization(
                "snapshot catalog has LPG owners without graph topology".into(),
            ));
        }
        return Ok((
            crate::database::catalog_wire::encode_catalog_read(view, epoch)?,
            Vec::new(),
            Vec::new(),
        ));
    }
    let cut = LpgIndexGraphCut::from_graphs(graphs);
    let section =
        CatalogSection::new_with_graphs(Arc::clone(catalog), cut.graphs(), move || epoch)?;
    let catalog_state = section.serialize_from_read(view, epoch)?;
    #[cfg(feature = "text-index")]
    let text_indexes = {
        let views = cut.text_views()?;
        if views.is_empty() {
            Vec::new()
        } else {
            grafeo_core::index::text::TextIndexSection::from_views(views).serialize()?
        }
    };
    #[cfg(not(feature = "text-index"))]
    let text_indexes = Vec::new();
    #[cfg(feature = "vector-index")]
    let vector_indexes = {
        let views = cut.vector_views()?;
        if views.is_empty() {
            Vec::new()
        } else {
            grafeo_core::index::vector::VectorStoreSection::from_views(views).serialize()?
        }
    };
    #[cfg(not(feature = "vector-index"))]
    let vector_indexes = Vec::new();
    Ok((catalog_state, text_indexes, vector_indexes))
}

pub(super) fn stage_snapshot_catalog(snapshot: &Snapshot) -> Result<Catalog> {
    let catalog =
        crate::database::catalog_wire::decode_catalog(&snapshot.catalog_state, snapshot.epoch)?;
    let graph_exists = |path: &GraphPath| {
        snapshot
            .graphs
            .binary_search_by(|graph| graph.path.cmp(path))
            .is_ok()
    };
    for (path, _) in catalog.all_graph_type_bindings() {
        if !graph_exists(&path) {
            return Err(Error::Serialization(format!(
                "snapshot catalog binding targets missing graph {path:?}"
            )));
        }
    }
    for owner in catalog.all_indexes() {
        if !graph_exists(owner.key.graph()) {
            return Err(Error::Serialization(format!(
                "snapshot catalog owner {} targets missing graph {:?}",
                owner.id,
                owner.key.graph()
            )));
        }
    }
    for namespace in catalog.schema_names() {
        let name = format!("{namespace}/__default__");
        let path = GraphPath::from_components(&[&name])
            .map_err(|error| Error::Serialization(error.to_string()))?;
        if !graph_exists(&path) {
            return Err(Error::Serialization(format!(
                "snapshot catalog namespace requires graph {path:?}"
            )));
        }
    }
    if snapshot.graphs.is_empty() && catalog.index_allocator_high_water() != 0 {
        return Err(Error::Serialization(
            "snapshot without LPG cannot carry its index allocator floor".into(),
        ));
    }
    Ok(catalog)
}

#[cfg(feature = "lpg")]
pub(super) fn stage_snapshot_indexes<'a>(
    snapshot: &'a Snapshot,
    catalog: &Catalog,
) -> Result<SnapshotIndexImages<'a>> {
    let owners = catalog.all_indexes();
    let mut chunks = Vec::new();
    for (family, bytes) in [
        (PhysicalIndexFamily::Text, snapshot.text_indexes.as_slice()),
        (
            PhysicalIndexFamily::Vector,
            snapshot.vector_indexes.as_slice(),
        ),
    ] {
        let mut keys: Vec<_> = owners
            .iter()
            .filter(|owner| owner.key.family() == family)
            .map(|owner| owner.key.clone())
            .collect();
        keys.sort_unstable();
        if bytes.is_empty() {
            if !keys.is_empty() {
                return Err(Error::Serialization(format!(
                    "snapshot lacks authoritative {family:?} image for its owners"
                )));
            }
            continue;
        }
        let actual = auxiliary_payload_keys(family, bytes)?;
        if keys.is_empty() || actual != keys {
            return Err(Error::Serialization(format!(
                "snapshot {family:?} payload keys do not match its exact catalog owners"
            )));
        }
        chunks.push(SnapshotIndexChunk {
            family,
            keys,
            bytes: Cow::Borrowed(bytes),
        });
    }
    Ok(SnapshotIndexImages { chunks })
}

/// Forks remove store-bound projection history, but unrelated exact indexes
/// retain their owner and complete image. Property indexes have no auxiliary
/// image and are derived from the fork's rows during detached installation.
#[cfg(all(feature = "lpg", feature = "triple-store"))]
pub(super) fn validate_snapshot_fork_indexes(
    snapshot: &Snapshot,
    removed_nodes: &HashSet<NodeId>,
) -> Result<()> {
    let catalog = stage_snapshot_catalog(snapshot)?;
    let images = stage_snapshot_indexes(snapshot, &catalog)?;
    if removed_nodes.is_empty() {
        return Ok(());
    }
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    let is_removed = |key: &PhysicalIndexKey, node: NodeId| {
        key.graph().components().is_empty() && removed_nodes.contains(&node)
    };
    for chunk in images.chunks {
        let overlaps: bool = match chunk.family {
            PhysicalIndexFamily::Text => {
                #[cfg(feature = "text-index")]
                {
                    grafeo_core::index::text::TextIndexSection::payload_references_nodes(
                        &chunk.bytes,
                        is_removed,
                    )
                }
                #[cfg(not(feature = "text-index"))]
                Err(Error::Serialization(
                    "snapshot Text state requires text-index support".into(),
                ))
            }
            PhysicalIndexFamily::Vector => {
                #[cfg(feature = "vector-index")]
                {
                    grafeo_core::index::vector::VectorStoreSection::payload_references_nodes(
                        &chunk.bytes,
                        is_removed,
                    )
                }
                #[cfg(not(feature = "vector-index"))]
                Err(Error::Serialization(
                    "snapshot Vector state requires vector-index support".into(),
                ))
            }
            PhysicalIndexFamily::Property => Err(Error::Serialization(
                "Property indexes have no auxiliary snapshot image".into(),
            )),
        }?;
        if overlaps {
            return Err(Error::Serialization(format!(
                "snapshot fork cannot retain exact {:?} index state referencing removed materialized projection rows",
                chunk.family
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "lpg")]
fn auxiliary_payload_keys(
    family: PhysicalIndexFamily,
    bytes: &[u8],
) -> Result<Vec<PhysicalIndexKey>> {
    match family {
        PhysicalIndexFamily::Text => {
            #[cfg(feature = "text-index")]
            {
                grafeo_core::index::text::TextIndexSection::payload_keys(bytes)
            }
            #[cfg(not(feature = "text-index"))]
            {
                Err(Error::Serialization(format!(
                    "snapshot Text state ({} bytes) requires text-index support",
                    bytes.len()
                )))
            }
        }
        PhysicalIndexFamily::Vector => {
            #[cfg(feature = "vector-index")]
            {
                grafeo_core::index::vector::VectorStoreSection::payload_keys(bytes)
            }
            #[cfg(not(feature = "vector-index"))]
            {
                Err(Error::Serialization(format!(
                    "snapshot Vector state ({} bytes) requires vector-index support",
                    bytes.len()
                )))
            }
        }
        PhysicalIndexFamily::Property => Err(Error::Serialization(
            "Property indexes have no auxiliary snapshot image".into(),
        )),
    }
}

#[cfg(feature = "lpg")]
fn auxiliary_payload_entry_ranges(
    family: PhysicalIndexFamily,
    bytes: &[u8],
) -> Result<Vec<(PhysicalIndexKey, Range<usize>)>> {
    match family {
        PhysicalIndexFamily::Text => {
            #[cfg(feature = "text-index")]
            {
                grafeo_core::index::text::TextIndexSection::payload_entry_ranges(bytes)
            }
            #[cfg(not(feature = "text-index"))]
            {
                Err(Error::Serialization(format!(
                    "snapshot Text state ({} bytes) requires text-index support",
                    bytes.len()
                )))
            }
        }
        PhysicalIndexFamily::Vector => {
            #[cfg(feature = "vector-index")]
            {
                grafeo_core::index::vector::VectorStoreSection::payload_entry_ranges(bytes)
            }
            #[cfg(not(feature = "vector-index"))]
            {
                Err(Error::Serialization(format!(
                    "snapshot Vector state ({} bytes) requires vector-index support",
                    bytes.len()
                )))
            }
        }
        PhysicalIndexFamily::Property => Err(Error::Serialization(
            "Property indexes have no auxiliary snapshot image".into(),
        )),
    }
}

#[cfg(feature = "lpg")]
fn select_auxiliary_payload(
    family: PhysicalIndexFamily,
    bytes: &[u8],
    keys: &[PhysicalIndexKey],
) -> Result<Vec<u8>> {
    match family {
        PhysicalIndexFamily::Text => {
            #[cfg(feature = "text-index")]
            {
                grafeo_core::index::text::TextIndexSection::select_payload_keys(bytes, keys)
            }
            #[cfg(not(feature = "text-index"))]
            {
                Err(Error::Serialization(format!(
                    "snapshot Text selection ({} owners, {} bytes) requires text-index support",
                    keys.len(),
                    bytes.len()
                )))
            }
        }
        PhysicalIndexFamily::Vector => {
            #[cfg(feature = "vector-index")]
            {
                grafeo_core::index::vector::VectorStoreSection::select_payload_keys(bytes, keys)
            }
            #[cfg(not(feature = "vector-index"))]
            {
                Err(Error::Serialization(format!(
                    "snapshot Vector selection ({} owners, {} bytes) requires vector-index support",
                    keys.len(),
                    bytes.len()
                )))
            }
        }
        PhysicalIndexFamily::Property => Err(Error::Serialization(
            "Property indexes have no auxiliary snapshot image".into(),
        )),
    }
}

#[cfg(feature = "lpg")]
pub(super) fn merge_snapshot_indexes<'a>(
    snapshots: &'a [Snapshot],
    merged_catalog: &Catalog,
) -> Result<SnapshotIndexImages<'a>> {
    let mut sources = Vec::new();
    for snapshot in snapshots {
        let source_catalog = stage_snapshot_catalog(snapshot)?;
        sources.extend(stage_snapshot_indexes(snapshot, &source_catalog)?.chunks);
    }

    // Decide ownership before inspecting entry bytes. Disjoint families take
    // the existing borrowed-chunk route without another payload traversal.
    let mut key_sources = std::collections::BTreeMap::<PhysicalIndexKey, usize>::new();
    let mut retained = Vec::with_capacity(sources.len());
    let mut duplicates = Vec::new();
    for (position, chunk) in sources.iter().enumerate() {
        let mut keys = Vec::new();
        for key in &chunk.keys {
            if let Some(prior) = key_sources.get(key) {
                duplicates.push((key.clone(), *prior, position));
            } else {
                key_sources.insert(key.clone(), position);
                keys.push(key.clone());
            }
        }
        retained.push(keys);
    }

    let mut ranges =
        HashMap::<usize, std::collections::BTreeMap<PhysicalIndexKey, Range<usize>>>::new();
    for (key, prior, position) in duplicates {
        for source in [prior, position] {
            if let hashbrown::hash_map::Entry::Vacant(entry) = ranges.entry(source) {
                let chunk = sources.get(source).ok_or_else(|| {
                    Error::Serialization(
                        "snapshot index source qualification is inconsistent".into(),
                    )
                })?;
                entry.insert(
                    auxiliary_payload_entry_ranges(chunk.family, &chunk.bytes)?
                        .into_iter()
                        .collect(),
                );
            }
        }
        let image = |source: usize| -> Result<&[u8]> {
            let bytes = sources.get(source).map(|chunk| chunk.bytes.as_ref());
            let range = ranges.get(&source).and_then(|entries| entries.get(&key));
            bytes
                .zip(range)
                .and_then(|(bytes, range)| bytes.get(range.clone()))
                .ok_or_else(|| {
                    Error::Serialization(
                        "snapshot index entry qualification is inconsistent".into(),
                    )
                })
        };
        if image(prior)? != image(position)? {
            return Err(Error::Serialization(format!(
                "open_multi: overlapping {:?} owner images are not byte-identical; exact index state cannot be merged",
                key.family(),
            )));
        }
    }

    let mut merged = SnapshotIndexImages { chunks: Vec::new() };
    for (mut chunk, keys) in sources.into_iter().zip(retained) {
        if keys.is_empty() {
            continue;
        }
        if keys.len() != chunk.keys.len() {
            chunk.bytes = Cow::Owned(select_auxiliary_payload(chunk.family, &chunk.bytes, &keys)?);
            chunk.keys = keys;
        }
        merged.chunks.push(chunk);
    }
    let mut expected: Vec<_> = merged_catalog
        .all_indexes()
        .into_iter()
        .filter(|owner| owner.key.family() != PhysicalIndexFamily::Property)
        .map(|owner| owner.key)
        .collect();
    expected.sort_unstable();
    let mut actual: Vec<_> = merged
        .chunks
        .iter()
        .flat_map(|chunk| chunk.keys.iter().cloned())
        .collect();
    actual.sort_unstable();
    if actual != expected {
        return Err(Error::Serialization(
            "open_multi exact index images do not match merged owners".into(),
        ));
    }
    Ok(merged)
}

#[cfg(feature = "lpg")]
pub(super) fn install_snapshot_catalog_and_indexes(
    db: &crate::database::GrafeoDB,
    staged_catalog: Catalog,
    staged_indexes: &SnapshotIndexImages<'_>,
) -> Result<()> {
    install_snapshot_catalog_and_indexes_into(
        db.store_arc(),
        &db.catalog,
        staged_catalog,
        staged_indexes,
    )
}

/// Validate complete physical images before an explicit union rebuild discards
/// them. Recovery checks semantic state and owner configuration, not just keys.
#[cfg(feature = "lpg")]
pub(super) fn validate_snapshot_indexes_for_rebuild(snapshot: &Snapshot) -> Result<()> {
    let staged_catalog = stage_snapshot_catalog(snapshot)?;
    let staged_indexes = stage_snapshot_indexes(snapshot, &staged_catalog)?;
    let store = Arc::new(LpgStore::new()?);
    for graph in &snapshot.graphs {
        super::create_snapshot_graph(&store, &graph.path)?;
    }
    // Exact Text and Vector recovery owns its postings/history/topology in the
    // auxiliary images. Empty graph targets suffice: no row population or
    // endpoint resolution is needed, including for dangling transport edges.
    let catalog = Arc::new(Catalog::new());
    install_snapshot_catalog_and_indexes_into(&store, &catalog, staged_catalog, &staged_indexes)
}

/// Populate canonical owners once from an unpublished union's retained rows.
#[cfg(feature = "lpg")]
pub(super) fn install_rebuilt_snapshot_catalog(
    db: &crate::database::GrafeoDB,
    catalog: Catalog,
) -> Result<()> {
    let cut = LpgIndexGraphCut::capture(Arc::clone(db.store_arc()))?;
    CatalogSection::new_with_graphs(Arc::clone(&db.catalog), cut.graphs(), || 0)?
        .install_unpublished_catalog_rebuilt(catalog)
}

/// All supplied targets must remain unpublished until this entire import succeeds.
#[cfg(feature = "lpg")]
pub(super) fn install_snapshot_catalog_and_indexes_into(
    store: &Arc<LpgStore>,
    catalog: &Arc<Catalog>,
    staged_catalog: Catalog,
    staged_indexes: &SnapshotIndexImages<'_>,
) -> Result<()> {
    let cut = LpgIndexGraphCut::capture(Arc::clone(store))?;
    let section = CatalogSection::new_with_graphs(Arc::clone(catalog), cut.graphs(), || 0)?;
    #[cfg(feature = "vector-index")]
    let section = section.with_vector_payloads(
        staged_indexes
            .chunks
            .iter()
            .filter(|chunk| chunk.family == PhysicalIndexFamily::Vector)
            .map(|chunk| chunk.bytes.as_ref()),
    )?;
    section.install_unpublished_catalog(staged_catalog)?;
    #[cfg(feature = "text-index")]
    let text_views: std::collections::BTreeMap<_, _> = cut.text_views()?.into_iter().collect();
    #[cfg(feature = "vector-index")]
    let vector_views: std::collections::BTreeMap<_, _> = cut.vector_views()?.into_iter().collect();
    staged_indexes
        .chunks
        .iter()
        .try_for_each(|chunk| match chunk.family {
            PhysicalIndexFamily::Text => {
                #[cfg(feature = "text-index")]
                {
                    let views = chunk
                        .keys
                        .iter()
                        .map(|key| {
                            text_views
                                .get(key)
                                .cloned()
                                .map(|view| (key.clone(), view))
                                .ok_or_else(|| {
                                    Error::Serialization(
                                        "snapshot Text recovery target is missing".into(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    grafeo_core::index::text::TextIndexSection::for_unpublished_recovery_views(
                        views,
                    )
                    .deserialize(&chunk.bytes)
                }
                #[cfg(not(feature = "text-index"))]
                Err(Error::Serialization(
                    "snapshot Text state requires text-index support".into(),
                ))
            }
            PhysicalIndexFamily::Vector => {
                #[cfg(feature = "vector-index")]
                {
                    let views = chunk
                        .keys
                        .iter()
                        .map(|key| {
                            vector_views
                                .get(key)
                                .cloned()
                                .map(|view| (key.clone(), view))
                                .ok_or_else(|| {
                                    Error::Serialization(
                                        "snapshot Vector recovery target is missing".into(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    grafeo_core::index::vector::VectorStoreSection::for_unpublished_recovery_views(
                        views,
                    )
                    .deserialize(&chunk.bytes)
                }
                #[cfg(not(feature = "vector-index"))]
                Err(Error::Serialization(
                    "snapshot Vector state requires vector-index support".into(),
                ))
            }
            PhysicalIndexFamily::Property => Err(Error::Serialization(
                "unexpected Property auxiliary image".into(),
            )),
        })
}

#[cfg(all(test, feature = "lpg", feature = "text-index"))]
mod tests {
    use super::super::{SchemaMergePolicy, decode_snapshot_bytes};
    use super::{
        merge_snapshot_catalogs, merge_snapshot_indexes, merge_snapshot_schemas,
        stage_snapshot_catalog, stage_snapshot_indexes,
    };
    use crate::database::{CreateIndexRequest, GrafeoDB, IndexCreateKind};
    use grafeo_common::types::{GraphPath, Value};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn rebuild_validation_checks_text_configuration_beyond_payload_keys() -> TestResult {
        let snapshot_with_tokenizer = |minimum| -> Result<_, Box<dyn std::error::Error>> {
            let source = GrafeoDB::new_in_memory();
            let node = source.create_node(&["Item"]);
            source.set_node_property(node, "body", Value::from("alpha beta"))?;
            source.create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("text".into()),
                label: Some("Item".into()),
                property: "body".into(),
                kind: IndexCreateKind::Text {
                    min_token_length: Some(minimum),
                },
            })?;
            Ok(decode_snapshot_bytes(&source.export_snapshot()?)?)
        };
        let mut snapshot = snapshot_with_tokenizer(2)?;
        super::validate_snapshot_indexes_for_rebuild(&snapshot)?;
        let other = snapshot_with_tokenizer(5)?;
        snapshot.text_indexes = other.text_indexes;
        let catalog = stage_snapshot_catalog(&snapshot)?;
        // The keys and complete wire shape remain valid. Exact recovery must
        // still reject the tokenizer/image mismatch before a rebuild discards it.
        stage_snapshot_indexes(&snapshot, &catalog)?;
        let error = super::validate_snapshot_indexes_for_rebuild(&snapshot)
            .expect_err("rebuild must validate the source image's owner configuration");
        assert!(matches!(
            error,
            grafeo_common::utils::error::Error::Serialization(_)
        ));
        Ok(())
    }

    #[test]
    fn snapshot10_exact_text_transport_keeps_owner_and_retained_image() -> TestResult {
        let source = GrafeoDB::new_in_memory();
        let node = source.create_node(&["Item"]);
        source.set_node_property(node, "body", Value::from("alpha beta"))?;
        let owner = source.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: Some("text".into()),
            label: Some("Item".into()),
            property: "body".into(),
            kind: IndexCreateKind::Text {
                min_token_length: None,
            },
        })?;
        source.set_node_property(node, "body", Value::from("gamma delta"))?;
        let bytes = source.export_snapshot()?;
        let snapshot = decode_snapshot_bytes(&bytes)?;
        let restored = GrafeoDB::import_snapshot(&bytes)?;
        assert_eq!(
            restored.catalog.get_index(owner),
            source.catalog.get_index(owner)
        );
        let round_trip = decode_snapshot_bytes(&restored.export_snapshot()?)?;
        assert_eq!(round_trip.text_indexes, snapshot.text_indexes);
        assert_eq!(round_trip.catalog_state, snapshot.catalog_state);

        let mut missing = snapshot.clone();
        missing.text_indexes.clear();
        let catalog = stage_snapshot_catalog(&missing)?;
        assert!(stage_snapshot_indexes(&missing, &catalog).is_err());
        Ok(())
    }

    #[test]
    fn snapshot10_shared_owner_requires_identical_exact_image() -> TestResult {
        let source = GrafeoDB::new_in_memory();
        let node = source.create_node(&["Item"]);
        source.set_node_property(node, "body", Value::from("alpha"))?;
        source.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: Some("text".into()),
            label: Some("Item".into()),
            property: "body".into(),
            kind: IndexCreateKind::Text {
                min_token_length: None,
            },
        })?;
        let snapshot = decode_snapshot_bytes(&source.export_snapshot()?)?;
        let identical = [snapshot.clone(), snapshot.clone()];
        let schema = merge_snapshot_schemas(&identical, SchemaMergePolicy::StrictEquality)?;
        let catalog = merge_snapshot_catalogs(&identical, &schema)?;
        let merged = merge_snapshot_indexes(&identical, &catalog)?;
        assert_eq!(merged.chunks.len(), 1);
        assert!(matches!(
            merged.chunks[0].bytes,
            std::borrow::Cow::Borrowed(_)
        ));

        source.set_node_property(node, "body", Value::from("different"))?;
        let changed = [snapshot, decode_snapshot_bytes(&source.export_snapshot()?)?];
        assert!(merge_snapshot_indexes(&changed, &catalog).is_err());
        Ok(())
    }

    #[test]
    fn snapshot10_partial_owner_union_copies_only_the_retained_partial_chunk() -> TestResult {
        let source = GrafeoDB::new_in_memory();
        let create = |db: &GrafeoDB, name: &str| {
            db.create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some(name.into()),
                label: Some("Item".into()),
                property: name.into(),
                kind: IndexCreateKind::Text {
                    min_token_length: None,
                },
            })
        };
        create(&source, "a")?;
        let base = source.export_snapshot()?;
        let left = GrafeoDB::import_snapshot(&base)?;
        let right = GrafeoDB::import_snapshot(&base)?;
        create(&left, "b")?;
        let burned = create(&right, "burned")?;
        assert!(right.drop_index(burned)?);
        create(&right, "c")?;
        let snapshots = [
            decode_snapshot_bytes(&left.export_snapshot()?)?,
            decode_snapshot_bytes(&right.export_snapshot()?)?,
        ];
        for pair in [
            snapshots.clone(),
            [snapshots[1].clone(), snapshots[0].clone()],
        ] {
            let schema = merge_snapshot_schemas(&pair, SchemaMergePolicy::StrictEquality)?;
            let catalog = merge_snapshot_catalogs(&pair, &schema)?;
            let merged = merge_snapshot_indexes(&pair, &catalog)?;
            assert_eq!(merged.chunks.len(), 2);
            assert_eq!(
                merged
                    .chunks
                    .iter()
                    .map(|chunk| chunk.keys.len())
                    .sum::<usize>(),
                3
            );
            assert!(matches!(
                merged.chunks[0].bytes,
                std::borrow::Cow::Borrowed(_)
            ));
            assert!(matches!(merged.chunks[1].bytes, std::borrow::Cow::Owned(_)));
            assert!(merged.chunks[1].bytes.len() < pair[1].text_indexes.len());
        }
        Ok(())
    }

    #[test]
    fn snapshot10_info_and_import_reject_mismatched_orphan_and_empty_exact_keys() -> TestResult {
        use crate::catalog::Catalog;
        use grafeo_common::storage::Section;
        use grafeo_core::graph::lpg::PhysicalIndexKey;
        use grafeo_core::index::text::{BM25Config, InvertedIndex, TextIndexSection};
        #[cfg(feature = "lpg")]
        use std::sync::Arc;

        let source = GrafeoDB::new_in_memory();
        source.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: Some("text".into()),
            label: Some("Item".into()),
            property: "body".into(),
            kind: IndexCreateKind::Text {
                min_token_length: None,
            },
        })?;
        let snapshot = decode_snapshot_bytes(&source.export_snapshot()?)?;
        let wrong_image = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Wrong", "property"),
            Arc::new(parking_lot::RwLock::new(InvertedIndex::new(
                BM25Config::default(),
            ))),
        )])
        .serialize()?;
        let mut mismatch = snapshot.clone();
        mismatch.text_indexes = wrong_image;
        let mut orphan = snapshot.clone();
        let empty_catalog = Catalog::new();
        orphan.catalog_state = crate::database::catalog_wire::encode_catalog_read(
            empty_catalog.read().view(),
            orphan.epoch,
        )?;
        let mut encoded_empty = orphan.clone();
        encoded_empty.text_indexes = TextIndexSection::new(Vec::new()).serialize()?;

        let mut cases = vec![
            ("mismatched Text key", mismatch),
            ("orphan Text key", orphan),
            ("noncanonical empty Text image", encoded_empty),
        ];
        #[cfg(feature = "vector-index")]
        {
            use grafeo_core::index::vector::{
                DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection,
            };
            let source = GrafeoDB::new_in_memory();
            source.create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("vector".into()),
                label: Some("Item".into()),
                property: "embedding".into(),
                kind: IndexCreateKind::Vector {
                    dimensions: Some(2),
                    metric: None,
                    m: None,
                    ef_construction: None,
                    ef: None,
                    quantization: None,
                },
            })?;
            let snapshot = decode_snapshot_bytes(&source.export_snapshot()?)?;
            let mut mismatch = snapshot.clone();
            mismatch.vector_indexes = VectorStoreSection::new(vec![(
                PhysicalIndexKey::vector(GraphPath::root(), "Wrong", "property"),
                Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                    2,
                    DistanceMetric::Cosine,
                )))),
            )])
            .serialize()?;
            let mut orphan = snapshot;
            orphan.catalog_state = crate::database::catalog_wire::encode_catalog_read(
                empty_catalog.read().view(),
                orphan.epoch,
            )?;
            let mut encoded_empty = orphan.clone();
            encoded_empty.vector_indexes = VectorStoreSection::new(Vec::new()).serialize()?;
            cases.extend([
                ("mismatched Vector key", mismatch),
                ("orphan Vector key", orphan),
                ("noncanonical empty Vector image", encoded_empty),
            ]);
        }
        for (case, snapshot) in cases.drain(..) {
            let bytes = super::super::encode_snapshot_bytes(&snapshot)?;
            let info_error = super::super::snapshot_info(&bytes)
                .err()
                .ok_or("snapshot_info accepted invalid keys")?;
            assert!(
                info_error.to_string().contains("payload keys"),
                "{case}: {info_error}"
            );
            let import_error = GrafeoDB::import_snapshot(&bytes)
                .err()
                .ok_or("import accepted invalid keys")?;
            assert!(
                import_error.to_string().contains("payload keys"),
                "{case}: {import_error}"
            );
        }
        Ok(())
    }
}
