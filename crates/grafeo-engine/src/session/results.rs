//! Administrative output borrows its source and admits every output allocation.

#![cfg(all(feature = "lpg", feature = "gql"))]

use crate::database::QueryResult;
use crate::query::executor::{BoundedResultWriter, ResultAccumulator, ResultLimits};
use grafeo_common::memory::buffer::MemoryGrant;
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::execution::QueryResourceContext;
use std::fmt::{self, Write};

fn exhausted(message: impl Into<String>) -> Error {
    Error::Storage(grafeo_common::utils::error::StorageError::Full).with_context(message)
}

fn checked(bytes: Option<usize>) -> Result<usize> {
    bytes.ok_or_else(|| exhausted("administrative result size overflow"))
}

/// Joins directly into the admitted formatter, without intermediate strings.
struct Joined<'a, T>(&'a [T]);
impl<T: fmt::Display> fmt::Display for Joined<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, item) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

struct Properties<'a>(&'a [crate::catalog::TypedProperty]);
impl fmt::Display for Properties<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, p) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "{} {}{}",
                p.name,
                p.data_type,
                if p.nullable { "" } else { " NOT NULL" }
            )?;
        }
        Ok(())
    }
}

struct Constraints<'a>(&'a [crate::catalog::TypeConstraint]);
impl fmt::Display for Constraints<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, c) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            write!(f, "{c:?}")?;
        }
        Ok(())
    }
}

fn local_name<'a>(name: &'a str, schema: Option<&str>) -> Option<&'a str> {
    match schema {
        Some(schema) => name.strip_prefix(schema)?.strip_prefix('/'),
        None => (!name.contains('/')).then_some(name),
    }
}

struct LocalNames<'a>(&'a [String], Option<&'a str>);
impl fmt::Display for LocalNames<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, name) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            f.write_str(local_name(name, self.1).unwrap_or(name))?;
        }
        Ok(())
    }
}

enum Cell<'a> {
    Text(fmt::Arguments<'a>),
    Bool(bool),
    Null,
}

// Field order releases source storage before its admission, including on
// iterator short-circuit, cancellation, formatting failure, and unwind.
struct AdmittedItems<T> {
    values: std::vec::IntoIter<T>,
    _grant: MemoryGrant,
}

impl<T> Iterator for AdmittedItems<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.values.next()
    }
}

struct AdminRows {
    accumulator: ResultAccumulator,
    resources: QueryResourceContext,
    limits: ResultLimits,
    rows: usize,
}

impl AdminRows {
    fn new<const N: usize>(session: &super::Session, names: [&str; N]) -> Result<Self> {
        let resources = session.result_resources()?;
        let bytes = names
            .iter()
            .try_fold(0usize, |bytes, name| checked(bytes.checked_add(name.len())))?;
        let _metadata = resources
            .try_allocate(bytes)
            .map_err(|e| exhausted(e.to_string()))?;
        let columns = names.map(str::to_owned);
        let limits = session.effective_result_limits();
        let accumulator = ResultAccumulator::new(&columns, &[], resources.clone(), limits)?;
        Ok(Self {
            accumulator,
            resources,
            limits,
            rows: 0,
        })
    }

    // Copy the small session coordinate under admission, then release its lock
    // before taking catalog/registry/overlay locks, as the original snapshot did.
    fn context(
        &self,
        session: &super::Session,
    ) -> Result<(MemoryGrant, super::SessionGraphContext)> {
        let context = session.current_context.lock();
        let bytes = checked(
            context
                .storage_key
                .components()
                .len()
                .checked_mul(size_of::<String>()),
        )?;
        let bytes = context
            .storage_key
            .components()
            .iter()
            .map(String::len)
            .chain(context.graph.as_ref().map(String::len))
            .chain(context.schema.as_ref().map(String::len))
            .try_fold(bytes, |bytes, len| checked(bytes.checked_add(len)))?;
        let grant = self
            .resources
            .try_allocate(bytes)
            .map_err(|e| exhausted(e.to_string()))?;
        Ok((grant, context.clone()))
    }

    fn append<const N: usize>(&mut self, cells: [Cell<'_>; N]) -> Result<()> {
        if self.rows >= self.limits.max_rows {
            return Err(exhausted("administrative result exceeds row limit"));
        }
        let mut writers: [Option<BoundedResultWriter>; N] = std::array::from_fn(|_| None);
        let mut grants: [Option<MemoryGrant>; N] = std::array::from_fn(|_| None);
        let mut values = [const { Value::Null }; N];
        for (index, cell) in cells.into_iter().enumerate() {
            values[index] = match cell {
                Cell::Text(arguments) => {
                    let mut writer =
                        BoundedResultWriter::new(self.resources.clone(), self.limits.max_bytes, 0)?;
                    let outcome = writer.write_fmt(arguments);
                    writer.resolve(outcome)?;
                    // Pinned ArcStr header/alignment allowance; the String and
                    // shared string coexist until the accumulator owns the row.
                    let bytes =
                        checked(writer.output().len().checked_add(128 + size_of::<Value>()))?;
                    grants[index] = Some(
                        self.resources
                            .try_allocate(bytes)
                            .map_err(|e| exhausted(e.to_string()))?,
                    );
                    let value = Value::from(writer.output());
                    writers[index] = Some(writer);
                    value
                }
                Cell::Bool(value) => Value::Bool(value),
                Cell::Null => Value::Null,
            };
        }
        self.accumulator.append_values(&values)?;
        self.rows += 1;
        Ok(())
    }

    /// Retains only borrowed keys, charging capacity before each growth. The
    /// guard must outlive this buffer; the caller keeps both until emission.
    fn sorted<T>(
        &self,
        items: impl Iterator<Item = T>,
        compare: impl Fn(&T, &T) -> std::cmp::Ordering,
    ) -> Result<AdmittedItems<T>> {
        self.sorted_source(items, compare, true)
    }

    fn sorted_source<T>(
        &self,
        items: impl Iterator<Item = T>,
        compare: impl Fn(&T, &T) -> std::cmp::Ordering,
        output_rows: bool,
    ) -> Result<AdmittedItems<T>> {
        let mut grant = self
            .resources
            .try_allocate(0)
            .map_err(|e| exhausted(e.to_string()))?;
        let mut values = Vec::new();
        for item in items {
            self.resources
                .check_cancelled()
                .map_err(crate::query::executor::convert_cancellation_error)?;
            if output_rows && values.len() >= self.limits.max_rows {
                return Err(exhausted("administrative result exceeds row limit"));
            }
            if values.len() == values.capacity() {
                let capacity = values
                    .capacity()
                    .saturating_mul(2)
                    .max(1)
                    .min(if output_rows {
                        self.limits.max_rows
                    } else {
                        usize::MAX
                    });
                let bytes = checked(capacity.checked_mul(size_of::<T>()))?;
                grant
                    .try_resize(bytes)
                    .map_err(|e| exhausted(e.to_string()))?;
                values
                    .try_reserve_exact(capacity - values.len())
                    .map_err(|e| exhausted(e.to_string()))?;
                if values.capacity() > capacity {
                    return Err(exhausted(
                        "administrative sort allocator exceeded admitted capacity",
                    ));
                }
            }
            values.push(item);
        }
        values.sort_unstable_by(compare);
        Ok(AdmittedItems {
            values: values.into_iter(),
            _grant: grant,
        })
    }

    fn finish(self) -> QueryResult {
        self.accumulator.finish()
    }
}

impl super::Session {
    pub(super) fn execute_show_indexes(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name", "type", "label", "property"])?;
        let catalog_owner = self.catalog_view();
        // Keep the historic catalog -> pending-index lock order.
        let catalog = catalog_owner.read();
        #[cfg(feature = "lpg")]
        let pending = self.pending_index_ddl.lock();
        let base = catalog
            .show_indexes()
            .filter(|def| {
                self.index_path_grant_allows(def.key.graph(), crate::auth::Role::ReadOnly)
            })
            .map(|def| (def.name.as_str(), 0, IndexSource::Catalog(def)));
        #[cfg(feature = "lpg")]
        let base = base.chain(
            pending
                .iter()
                .enumerate()
                .filter(|(_, ddl)| {
                    self.index_path_grant_allows(&ddl.graph, crate::auth::Role::ReadOnly)
                })
                .filter_map(|(index, ddl)| {
                    ddl.name
                        .as_deref()
                        .map(|name| (name, index + 1, IndexSource::Pending(ddl)))
                }),
        );
        // Sort borrowed source descriptors; latest visible pending operation wins
        // for a name, including DROP. Duplicate candidates are not output rows.
        let entries = output.sorted_source(base, |a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)), false)?;
        let mut entries = entries.peekable();
        while let Some((name, _, source)) = entries.next() {
            if entries.peek().is_some_and(|next| next.0 == name) {
                continue;
            }
            match source {
                IndexSource::Catalog(def) => {
                    let label = catalog.get_label_name(def.label);
                    let property = catalog.get_property_key_name(def.property_key);
                    output.append([
                        Cell::Text(format_args!("{name}")),
                        Cell::Text(format_args!("{:?}", def.index_type)),
                        Cell::Text(format_args!("{}", label.as_deref().unwrap_or("?"))),
                        Cell::Text(format_args!("{}", property.as_deref().unwrap_or("?"))),
                    ])?;
                }
                #[cfg(feature = "lpg")]
                IndexSource::Pending(ddl) if ddl.create => {
                    output.append([
                        Cell::Text(format_args!("{name}")),
                        Cell::Text(format_args!("{:?}", Self::catalog_index_kind(&ddl.kind))),
                        Cell::Text(format_args!("{}", ddl.label)),
                        Cell::Text(format_args!("{}", ddl.property)),
                    ])?;
                }
                #[cfg(feature = "lpg")]
                IndexSource::Pending(_) => {}
            }
        }
        Ok(output.finish())
    }

    pub(super) fn execute_show_constraints(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name", "type", "label", "properties"])?;
        let (_context_grant, context) = output.context(self)?;
        let owner = self.catalog_view();
        let catalog = owner.read();
        let definitions = catalog.show_constraints().filter_map(|def| {
            Some((
                local_name(&def.name, context.schema.as_deref())?,
                local_name(&def.label, context.schema.as_deref())?,
                def,
            ))
        });
        let definitions = output.sorted(definitions, |a, b| a.0.cmp(b.0))?;
        for (name, label, def) in definitions {
            output.append([
                Cell::Text(format_args!("{name}")),
                Cell::Text(format_args!("{}", def.kind.as_str())),
                Cell::Text(format_args!("{label}")),
                Cell::Text(format_args!("{}", Joined(&def.properties))),
            ])?;
        }
        Ok(output.finish())
    }

    pub(super) fn execute_show_node_types(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name", "properties", "constraints", "parents"])?;
        let (_context_grant, context) = output.context(self)?;
        let owner = self.catalog_view();
        let catalog = owner.read();
        for (key, def) in catalog.show_node_types() {
            if let Some(name) = local_name(key, context.schema.as_deref()) {
                output.append([
                    Cell::Text(format_args!("{name}")),
                    Cell::Text(format_args!("{}", Properties(&def.properties))),
                    Cell::Text(format_args!("{}", Constraints(&def.constraints))),
                    Cell::Text(format_args!("{}", Joined(&def.parent_types))),
                ])?;
            }
        }
        Ok(output.finish())
    }

    pub(super) fn execute_show_edge_types(&self) -> Result<QueryResult> {
        let mut output =
            AdminRows::new(self, ["name", "properties", "source_types", "target_types"])?;
        let (_context_grant, context) = output.context(self)?;
        let owner = self.catalog_view();
        let catalog = owner.read();
        for (key, def) in catalog.show_edge_types() {
            if let Some(name) = local_name(key, context.schema.as_deref()) {
                output.append([
                    Cell::Text(format_args!("{name}")),
                    Cell::Text(format_args!("{}", Properties(&def.properties))),
                    Cell::Text(format_args!("{}", Joined(&def.source_node_types))),
                    Cell::Text(format_args!("{}", Joined(&def.target_node_types))),
                ])?;
            }
        }
        Ok(output.finish())
    }

    pub(super) fn execute_show_graph_types(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name", "open", "node_types", "edge_types"])?;
        let (_context_grant, context) = output.context(self)?;
        let owner = self.catalog_view();
        let catalog = owner.read();
        for (key, def) in catalog.show_graph_types() {
            if let Some(name) = local_name(key, context.schema.as_deref()) {
                output.append([
                    Cell::Text(format_args!("{name}")),
                    Cell::Bool(def.open),
                    Cell::Text(format_args!(
                        "{}",
                        LocalNames(&def.allowed_node_types, context.schema.as_deref())
                    )),
                    Cell::Text(format_args!(
                        "{}",
                        LocalNames(&def.allowed_edge_types, context.schema.as_deref())
                    )),
                ])?;
            }
        }
        Ok(output.finish())
    }

    #[cfg(feature = "lpg")]
    pub(super) fn execute_show_graphs(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name"])?;
        let (_context_grant, context) = output.context(self)?;
        self.store.with_graph_names(|names| {
            // Auth constructs a GraphPath from the borrowed storage key. Admit
            // its source-dependent backing before invoking that existing check.
            let mut auth_error = None;
            let names = names
                .filter(|name| {
                    let Some(bytes) = name.len().checked_add(size_of::<String>()) else {
                        auth_error = Some(exhausted("graph authorization size overflow"));
                        return false;
                    };
                    let Ok(_grant) = output.resources.try_allocate(bytes) else {
                        auth_error = Some(exhausted("graph authorization exceeds query grant"));
                        return false;
                    };
                    self.graph_grant_allows(Some(name), crate::auth::Role::ReadOnly)
                })
                .filter_map(|name| local_name(name, context.schema.as_deref()))
                .filter(|name| context.schema.is_none() || *name != super::SCHEMA_DEFAULT_GRAPH);
            let names = output.sorted(names, |a, b| a.cmp(b))?;
            if let Some(error) = auth_error {
                return Err(error);
            }
            for name in names {
                output.append([Cell::Text(format_args!("{name}"))])?;
            }
            Ok(output.finish())
        })
    }

    pub(super) fn execute_show_schemas(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name"])?;
        let owner = self.catalog_view();
        let catalog = owner.read();
        let names = output.sorted(catalog.show_schemas(), |a, b| a.cmp(b))?;
        for name in names {
            output.append([Cell::Text(format_args!("{name}"))])?;
        }
        Ok(output.finish())
    }

    pub(super) fn execute_show_graph_type(&self, name: &str) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name", "open", "node_types", "edge_types"])?;
        let (_context_grant, context) = output.context(self)?;
        let owner = self.catalog_view();
        let catalog = owner.read();
        let def = catalog
            .show_graph_types()
            .find(|(key, _)| match context.schema.as_deref() {
                Some(schema) => {
                    key.strip_prefix(schema)
                        .and_then(|key| key.strip_prefix('/'))
                        == Some(name)
                }
                None => *key == name,
            })
            .map(|(_, def)| def)
            .ok_or_else(|| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("Graph type '{name}' not found"),
                ))
            })?;
        output.append([
            Cell::Text(format_args!("{name}")),
            Cell::Bool(def.open),
            Cell::Text(format_args!("{}", Joined(&def.allowed_node_types))),
            Cell::Text(format_args!("{}", Joined(&def.allowed_edge_types))),
        ])?;
        Ok(output.finish())
    }

    pub(super) fn execute_show_current_graph_type(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(
            self,
            ["graph", "graph_type", "open", "node_types", "edge_types"],
        )?;
        let (_context_grant, context) = output.context(self)?;
        self.require_graph_path_grant(&context.storage_key, crate::auth::Role::ReadOnly)?;
        // Preserve pending-binding -> catalog order and a pending clear.
        #[cfg(feature = "lpg")]
        let pending = self.pending_graph_type_bindings.lock();
        #[cfg(feature = "lpg")]
        let pending_binding = pending
            .get(&context.storage_key)
            .map(|pending| pending.replacement.as_deref());
        #[cfg(not(feature = "lpg"))]
        let pending_binding: Option<Option<&str>> = None;
        let owner = self.catalog_view();
        let catalog = owner.read();
        let binding = pending_binding
            .unwrap_or_else(|| catalog.show_graph_type_binding(&context.storage_key));
        let graph_name = GraphName(&context);
        if let Some(type_name) = binding
            && let Some((_, def)) = catalog
                .show_graph_types()
                .find(|(key, _)| *key == type_name)
        {
            output.append([
                Cell::Text(format_args!("{graph_name}")),
                Cell::Text(format_args!("{type_name}")),
                Cell::Bool(def.open),
                Cell::Text(format_args!("{}", Joined(&def.allowed_node_types))),
                Cell::Text(format_args!("{}", Joined(&def.allowed_edge_types))),
            ])?;
        } else {
            output.append([
                Cell::Text(format_args!("{graph_name}")),
                Cell::Null,
                Cell::Null,
                Cell::Null,
                Cell::Null,
            ])?;
        }
        Ok(output.finish())
    }

    #[cfg(feature = "lpg")]
    pub(super) fn execute_show_projections(&self) -> Result<QueryResult> {
        let mut output = AdminRows::new(self, ["name"])?;
        let transaction_id = self.current_transaction_id();
        if transaction_id.is_some_and(|id| {
            self.transaction_manager.isolation_level(id)
                == Some(crate::transaction::IsolationLevel::Serializable)
        }) {
            self.projection_registry_read
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let _publication = self.publication_read_guard();
        let registry = self.projections.read();
        let snapshot = self.projection_registry_snapshot.lock();
        let base = if transaction_id.is_some() {
            snapshot.as_ref().unwrap_or(&registry)
        } else {
            &registry
        };
        let pending = self.pending_projection_ddl.lock();
        let root = grafeo_common::types::GraphPath::root();
        let base = base.iter().filter(|(name, _)| !pending.contains_key(*name));
        let additions = pending.iter().filter_map(|(name, change)| {
            change
                .replacement
                .as_ref()
                .map(|projection| (name, projection))
        });
        let names = base
            .chain(additions)
            .filter(|(_, projection)| {
                self.index_path_grant_allows(
                    projection.source_graph().map_or(&root, |(key, _)| key),
                    crate::auth::Role::ReadOnly,
                )
            })
            .map(|(name, _)| name.as_str());
        let names = output.sorted(names, |a, b| a.cmp(b))?;
        for name in names {
            output.append([Cell::Text(format_args!("{name}"))])?;
        }
        Ok(output.finish())
    }
}

enum IndexSource<'a> {
    Catalog(&'a crate::catalog::IndexDefinition),
    #[cfg(feature = "lpg")]
    Pending(&'a super::PendingIndexDdl),
}

struct GraphName<'a>(&'a super::SessionGraphContext);
impl fmt::Display for GraphName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.graph.as_deref() {
            Some(name) => f.write_str(name),
            None if self.0.native => write!(f, "{:?}", self.0.storage_key),
            None => f.write_str("default"),
        }
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::GrafeoDB;
    use crate::query::executor::ExecutionOptions;
    use std::collections::HashMap;

    fn bounded(
        session: &super::super::Session,
        query: &str,
        rows: usize,
        bytes: usize,
    ) -> Result<QueryResult> {
        session.execute_with_options(
            query,
            HashMap::new(),
            ExecutionOptions {
                result_limits: Some(ResultLimits {
                    max_rows: rows,
                    max_bytes: bytes,
                }),
                ..ExecutionOptions::default()
            },
        )
    }

    #[test]
    fn bounded_show_schemas_sort_and_exact_row_boundary() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for name in ["z", "a", "m"] {
            session
                .catalog
                .register_schema_namespace(name.to_owned())
                .unwrap();
        }
        assert!(bounded(&session, "SHOW SCHEMAS", 2, 1 << 20).is_err());
        let result = bounded(&session, "SHOW SCHEMAS", 3, 1 << 20).unwrap();
        assert_eq!(
            result.rows(),
            &[
                vec![Value::from("a")],
                vec![Value::from("m")],
                vec![Value::from("z")]
            ]
        );
        assert!(bounded(&session, "SHOW SCHEMAS", 3, 1).is_err());
        assert!(bounded(&session, "SHOW SCHEMAS", 0, 0).is_err());
        assert_eq!(
            bounded(&session, "SHOW INDEXES", 0, 0).unwrap().row_count(),
            0
        );
        assert_eq!(
            bounded(&session, "SHOW PROJECTIONS", 0, 0)
                .unwrap()
                .row_count(),
            0
        );
    }

    #[test]
    fn bounded_show_preserves_property_format_and_schema_display() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INTEGER)")
            .unwrap();
        let result = bounded(&session, "SHOW NODE TYPES", 1, 1 << 20).unwrap();
        assert_eq!(
            result.rows()[0][1],
            Value::from("name STRING NOT NULL, age INT64")
        );
        assert!(bounded(&session, "SHOW NODE TYPES", 1, 8).is_err());
        session
            .catalog
            .register_schema_namespace("scope".into())
            .unwrap();
        session
            .catalog
            .register_graph_type(crate::catalog::GraphTypeDefinition {
                name: "scope/social".into(),
                allowed_node_types: vec!["scope/Person".into(), "external/City".into()],
                allowed_edge_types: Vec::new(),
                open: false,
            })
            .unwrap();
        session.execute("SESSION SET SCHEMA scope").unwrap();
        let plural = bounded(&session, "SHOW GRAPH TYPES", 1, 1 << 20).unwrap();
        assert_eq!(plural.rows()[0][0], Value::from("social"));
        assert_eq!(plural.rows()[0][2], Value::from("Person, external/City"));
        let singular = bounded(&session, "SHOW GRAPH TYPE social", 1, 1 << 20).unwrap();
        assert_eq!(
            singular.rows()[0][2],
            Value::from("scope/Person, external/City")
        );
    }

    #[test]
    fn bounded_show_indexes_preserves_pending_drop_and_create() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("CREATE INDEX old_owner FOR (n:Person) ON (n.email)")
            .unwrap();
        session.execute("START TRANSACTION").unwrap();
        session.execute("DROP INDEX old_owner").unwrap();
        assert_eq!(
            bounded(&session, "SHOW INDEXES", 0, 0).unwrap().row_count(),
            0
        );
        session
            .execute("CREATE INDEX new_owner FOR (n:Person) ON (n.email)")
            .unwrap();
        let result = bounded(&session, "SHOW INDEXES", 1, 1 << 20).unwrap();
        assert_eq!(
            result.rows()[0],
            vec![
                Value::from("new_owner"),
                Value::from("Hash"),
                Value::from("Person"),
                Value::from("email")
            ]
        );
        assert!(bounded(&session, "SHOW INDEXES", 0, 1 << 20).is_err());
        session.execute("ROLLBACK").unwrap();
        let result = session.execute("SHOW INDEXES").unwrap();
        assert_eq!(result.rows()[0][0], Value::from("old_owner"));
    }
}
