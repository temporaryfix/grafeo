//! Canonical procedure resolution and transitive effect analysis.
//!
//! Stored procedure source is resolved into an immutable snapshot before
//! authorization. The same snapshot is then handed to the physical planner,
//! preventing a concurrent DROP/REPLACE from changing which body runs after
//! the Session has selected read versus write authority.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use crate::catalog::{Catalog, ProcedureDefinition, PropertyDataType};
use crate::procedures::{Procedure, ProcedureEffect, builtin_registry};
use crate::query::plan::{CallProcedureOp, LogicalOperator, LogicalPlan};

/// Maximum number of catalog procedure frames in one invocation chain.
pub(crate) const MAX_PROCEDURE_CALL_DEPTH: usize = 64;

/// A catalog procedure body parsed and classified at one catalog cut.
#[derive(Clone)]
pub(crate) struct ResolvedCatalogProcedure {
    pub(crate) definition: Arc<ProcedureDefinition>,
    pub(crate) parameter_types: Vec<PropertyDataType>,
    pub(crate) return_types: Vec<PropertyDataType>,
    pub(crate) logical_body: LogicalPlan,
    pub(crate) effect: ProcedureEffect,
}

/// Immutable procedure definitions and parsed bodies for one statement.
#[derive(Clone, Default)]
pub(crate) struct ResolvedProcedureCatalog {
    resolved: HashMap<String, Arc<ResolvedCatalogProcedure>>,
    fingerprint: Vec<ProcedureFingerprint>,
}

type ProcedureFingerprint = (String, Vec<(String, String)>, Vec<(String, String)>, String);

impl ResolvedProcedureCatalog {
    /// Shared immutable empty state avoids an allocation for ordinary queries.
    pub(crate) fn empty() -> Arc<Self> {
        static EMPTY: OnceLock<Arc<ResolvedProcedureCatalog>> = OnceLock::new();
        Arc::clone(EMPTY.get_or_init(|| Arc::new(Self::default())))
    }

    /// Resolves a call using the exact precedence used during effect analysis.
    pub(crate) fn resolve(&self, call: &CallProcedureOp) -> Result<ResolvedProcedure> {
        if is_listing_call(&call.name) {
            return Ok(ResolvedProcedure::Listing);
        }

        if let Some(name) = catalog_lookup_name(&call.name)
            && let Some(procedure) = self.resolved.get(name)
        {
            return Ok(ResolvedProcedure::Catalog(Arc::clone(procedure)));
        }

        builtin_registry()
            .get(&call.name)
            .map(ResolvedProcedure::Builtin)
            .ok_or_else(|| unknown_procedure(&call.name))
    }

    /// Confirms that the live catalog still matches the statement snapshot.
    ///
    /// Session holds its publication read guard from this check through
    /// physical execution. A catalog DDL publication in the small interval
    /// between initial qualification and transaction start therefore causes a
    /// clean retry error rather than executing a stale definition.
    pub(crate) fn validate_live_catalog(&self, catalog: &Catalog) -> Result<()> {
        let live = catalog_fingerprint(catalog.all_procedure_defs());
        if live == self.fingerprint {
            return Ok(());
        }
        Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "stored procedure catalog changed while preparing the statement; retry the statement",
        )))
    }

    /// Returns whether `root` contains a transitively write-capable call.
    ///
    /// Resolution uses this statement's immutable procedure snapshot, so a
    /// physical short-circuit decision cannot disagree with the effect that
    /// Session authorized and framed.
    pub(crate) fn contains_mutating_call(&self, root: &LogicalOperator) -> Result<bool> {
        if let LogicalOperator::CallProcedure(call) = root {
            let mutates = match self.resolve(call)? {
                ResolvedProcedure::Listing => false,
                ResolvedProcedure::Builtin(procedure) => {
                    procedure.effect() == ProcedureEffect::MayWrite
                }
                ResolvedProcedure::Catalog(procedure) => {
                    procedure.effect == ProcedureEffect::MayWrite
                }
            };
            if mutates {
                return Ok(true);
            }
        }

        for child in root.children() {
            if self.contains_mutating_call(child)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// One canonical call resolution result shared by analysis and planning.
pub(crate) enum ResolvedProcedure {
    Listing,
    Builtin(Arc<dyn Procedure>),
    Catalog(Arc<ResolvedCatalogProcedure>),
}

/// Security- and transaction-relevant properties of an LPG logical plan.
#[derive(Clone)]
pub(crate) struct LpgPlanEffects {
    pub(crate) mutates: bool,
    pub(crate) contains_call: bool,
    pub(crate) contains_mutating_call: bool,
    pub(crate) procedures: Arc<ResolvedProcedureCatalog>,
}

impl LpgPlanEffects {
    /// The captured resolutions are the authority for whether catalog bodies run.
    pub(crate) fn contains_catalog_call(&self) -> bool {
        !self.procedures.resolved.is_empty()
    }
}

/// Resolves every reachable procedure and computes its transitive effect.
pub(crate) fn analyze_procedure_effects(
    root: &LogicalOperator,
    catalog: &Catalog,
) -> Result<LpgPlanEffects> {
    if !root.contains_procedure_call() {
        return Ok(LpgPlanEffects {
            mutates: root.has_mutations(),
            contains_call: false,
            contains_mutating_call: false,
            procedures: ResolvedProcedureCatalog::empty(),
        });
    }

    let definitions = catalog
        .all_procedure_defs()
        .into_iter()
        .map(|definition| (definition.name.clone(), Arc::new(definition)))
        .collect::<HashMap<_, _>>();
    let fingerprint = catalog_fingerprint(
        definitions
            .values()
            .map(|definition| definition.as_ref().clone())
            .collect(),
    );
    let mut analyzer = Analyzer {
        definitions,
        resolved: HashMap::new(),
        visits: HashMap::new(),
        stack: Vec::new(),
        contains_mutating_call: false,
    };
    let effect = analyzer.visit_operator(root)?;
    let snapshot = ResolvedProcedureCatalog {
        resolved: analyzer.resolved,
        fingerprint,
    };

    Ok(LpgPlanEffects {
        mutates: root.has_mutations() || effect == ProcedureEffect::MayWrite,
        contains_call: true,
        contains_mutating_call: analyzer.contains_mutating_call,
        procedures: Arc::new(snapshot),
    })
}

#[derive(Clone, Copy)]
enum Visit {
    Visiting,
    Done(ProcedureEffect),
}

struct Analyzer {
    definitions: HashMap<String, Arc<ProcedureDefinition>>,
    resolved: HashMap<String, Arc<ResolvedCatalogProcedure>>,
    visits: HashMap<String, Visit>,
    stack: Vec<String>,
    contains_mutating_call: bool,
}

impl Analyzer {
    fn visit_operator(&mut self, operator: &LogicalOperator) -> Result<ProcedureEffect> {
        let mut effect = if operator.has_mutations() {
            ProcedureEffect::MayWrite
        } else {
            ProcedureEffect::ReadOnly
        };

        if let LogicalOperator::CallProcedure(call) = operator {
            let call_effect = self.visit_call(call)?;
            if call_effect == ProcedureEffect::MayWrite {
                self.contains_mutating_call = true;
            }
            effect = effect.max(call_effect);
        }
        for child in operator.children() {
            effect = effect.max(self.visit_operator(child)?);
        }
        Ok(effect)
    }

    fn visit_call(&mut self, call: &CallProcedureOp) -> Result<ProcedureEffect> {
        if is_listing_call(&call.name) {
            return Ok(ProcedureEffect::ReadOnly);
        }
        if let Some(name) = catalog_lookup_name(&call.name)
            && self.definitions.contains_key(name)
        {
            return self.visit_catalog(name);
        }
        builtin_registry()
            .get(&call.name)
            .map(|procedure| procedure.effect())
            .ok_or_else(|| unknown_procedure(&call.name))
    }

    fn visit_catalog(&mut self, name: &str) -> Result<ProcedureEffect> {
        match self.visits.get(name).copied() {
            Some(Visit::Done(effect)) => return Ok(effect),
            Some(Visit::Visiting) => return Err(self.cycle_error(name)),
            None => {}
        }
        if self.stack.len() >= MAX_PROCEDURE_CALL_DEPTH {
            let mut path = self.stack.clone();
            path.push(name.to_string());
            return Err(semantic_error(format!(
                "procedure call depth exceeded {MAX_PROCEDURE_CALL_DEPTH}: {}",
                path.join(" -> ")
            )));
        }

        let definition = self.definitions.get(name).cloned().ok_or_else(|| {
            Error::Internal(format!(
                "procedure snapshot lost reachable definition '{name}'"
            ))
        })?;
        self.visits.insert(name.to_string(), Visit::Visiting);
        self.stack.push(name.to_string());

        let (parameter_types, return_types) = resolve_signature(&definition)?;
        let logical_body = crate::query::translators::gql::translate(&definition.body)?;
        if logical_body.explain || logical_body.profile {
            return Err(semantic_error(format!(
                "procedure '{name}' body cannot contain EXPLAIN or PROFILE"
            )));
        }
        let effect = self.visit_operator(&logical_body.root)?;

        let popped = self.stack.pop();
        debug_assert_eq!(popped.as_deref(), Some(name));
        self.visits.insert(name.to_string(), Visit::Done(effect));
        self.resolved.insert(
            name.to_string(),
            Arc::new(ResolvedCatalogProcedure {
                definition,
                parameter_types,
                return_types,
                logical_body,
                effect,
            }),
        );
        Ok(effect)
    }

    fn cycle_error(&self, repeated: &str) -> Error {
        let start = self
            .stack
            .iter()
            .position(|name| name == repeated)
            .unwrap_or(0);
        let mut path = self.stack[start..].to_vec();
        path.push(repeated.to_string());
        semantic_error(format!(
            "procedure call cycle detected: {}",
            path.join(" -> ")
        ))
    }
}

fn resolve_signature(
    definition: &ProcedureDefinition,
) -> Result<(Vec<PropertyDataType>, Vec<PropertyDataType>)> {
    fn resolve_columns(
        procedure: &str,
        kind: &str,
        columns: &[(String, String)],
    ) -> Result<Vec<PropertyDataType>> {
        let mut seen = std::collections::HashSet::new();
        columns
            .iter()
            .map(|(name, data_type)| {
                if !seen.insert(name) {
                    return Err(semantic_error(format!(
                        "procedure '{procedure}' declares duplicate {kind} name '{name}'"
                    )));
                }
                parse_procedure_type(data_type).ok_or_else(|| {
                    semantic_error(format!(
                        "procedure '{procedure}' {kind} '{name}' uses unsupported type '{data_type}'"
                    ))
                })
            })
            .collect()
    }

    Ok((
        resolve_columns(&definition.name, "parameter", &definition.params)?,
        resolve_columns(&definition.name, "return", &definition.returns)?,
    ))
}

fn parse_procedure_type(name: &str) -> Option<PropertyDataType> {
    let upper = name.trim().to_ascii_uppercase();
    if let Some(inner) = upper
        .strip_prefix("LIST<")
        .and_then(|value| value.strip_suffix('>'))
    {
        return parse_procedure_type(inner)
            .map(|element| PropertyDataType::ListTyped(Box::new(element)));
    }
    match upper.as_str() {
        "STRING" | "VARCHAR" | "TEXT" => Some(PropertyDataType::String),
        "INT" | "INT64" | "INTEGER" | "BIGINT" => Some(PropertyDataType::Int64),
        "FLOAT" | "FLOAT64" | "DOUBLE" | "REAL" => Some(PropertyDataType::Float64),
        "BOOL" | "BOOLEAN" => Some(PropertyDataType::Bool),
        "DATE" => Some(PropertyDataType::Date),
        "TIME" => Some(PropertyDataType::Time),
        "TIMESTAMP" | "DATETIME" => Some(PropertyDataType::Timestamp),
        "DURATION" | "INTERVAL" => Some(PropertyDataType::Duration),
        "LIST" | "ARRAY" => Some(PropertyDataType::List),
        "MAP" | "RECORD" => Some(PropertyDataType::Map),
        "BYTES" | "BINARY" | "BLOB" => Some(PropertyDataType::Bytes),
        "NODE" => Some(PropertyDataType::Node),
        "EDGE" | "RELATIONSHIP" => Some(PropertyDataType::Edge),
        "ANY" => Some(PropertyDataType::Any),
        _ => None,
    }
}

fn is_listing_call(name: &[String]) -> bool {
    matches!(name, [single] if single == "procedures")
        || matches!(name, [namespace, procedure]
            if namespace.eq_ignore_ascii_case("grafeo") && procedure == "procedures")
}

fn catalog_lookup_name(name: &[String]) -> Option<&str> {
    match name {
        // Qualified namespaces are reserved for builtin dispatch. Catalog
        // procedures are intentionally addressed by their declared, bare name
        // so `grafeo.*` and `db.*` cannot be shadowed downstream.
        [name] => Some(name),
        _ => None,
    }
}

fn unknown_procedure(name: &[String]) -> Error {
    semantic_error(format!(
        "unknown procedure '{}'; use CALL grafeo.procedures() to list available procedures",
        name.join(".")
    ))
}

fn semantic_error(message: impl Into<String>) -> Error {
    Error::Query(QueryError::new(QueryErrorKind::Semantic, message))
}

fn catalog_fingerprint(definitions: Vec<ProcedureDefinition>) -> Vec<ProcedureFingerprint> {
    let mut fingerprint = definitions
        .into_iter()
        .map(|definition| {
            (
                definition.name,
                definition.params,
                definition.returns,
                definition.body,
            )
        })
        .collect::<Vec<_>>();
    fingerprint.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    fingerprint
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(name: &str, body: impl Into<String>) -> ProcedureDefinition {
        ProcedureDefinition {
            name: name.to_string(),
            params: Vec::new(),
            returns: vec![("value".to_string(), "ANY".to_string())],
            body: body.into(),
        }
    }

    fn analyze(query: &str, catalog: &Catalog) -> Result<LpgPlanEffects> {
        let plan = crate::query::translators::gql::translate(query)?;
        analyze_procedure_effects(&plan.root, catalog)
    }

    #[test]
    fn ordinary_queries_share_empty_resolution_without_catalog_state() -> Result<()> {
        let populated = Catalog::new();
        populated
            .register_procedure(definition("unused", "RETURN 1 AS value"))
            .map_err(|error| Error::Internal(error.to_string()))?;
        let first = analyze("MATCH (n) RETURN n", &populated)?;
        let second = analyze("MATCH (n) RETURN n", &Catalog::new())?;
        assert!(Arc::ptr_eq(&first.procedures, &second.procedures));
        for effects in [first, second] {
            assert!(!effects.mutates);
            assert!(!effects.contains_call);
            assert!(!effects.contains_mutating_call);
            assert!(!effects.contains_catalog_call());
            assert!(effects.procedures.fingerprint.is_empty());
        }
        Ok(())
    }

    fn analyze_error(query: &str, catalog: &Catalog) -> Error {
        match analyze(query, catalog) {
            Ok(_) => panic!("expected procedure analysis to fail"),
            Err(error) => error,
        }
    }

    #[test]
    fn nested_read_chain_stays_read_only() {
        let catalog = Catalog::new();
        catalog
            .register_procedure(definition("leaf", "RETURN 1 AS value"))
            .unwrap();
        catalog
            .register_procedure(definition("outer", "CALL leaf() YIELD value RETURN value"))
            .unwrap();

        let effects = analyze("CALL outer()", &catalog).unwrap();
        assert!(!effects.mutates);
        assert!(effects.contains_call);
        assert!(effects.contains_catalog_call());
    }

    #[test]
    fn trusted_namespace_cannot_be_shadowed_by_catalog_leaf_name() {
        let catalog = Catalog::new();
        catalog
            .register_procedure(definition("labels", "INSERT (n:Shadow) RETURN n AS value"))
            .unwrap();

        let qualified = analyze("CALL grafeo.labels()", &catalog).unwrap();
        assert!(!qualified.mutates);
        assert!(!qualified.contains_catalog_call());

        let unqualified = analyze("CALL labels()", &catalog).unwrap();
        assert!(unqualified.mutates, "bare catalog names retain precedence");
        assert!(unqualified.contains_catalog_call());
    }

    #[test]
    fn direct_cycle_is_a_structured_semantic_error() {
        let catalog = Catalog::new();
        catalog
            .register_procedure(definition(
                "looping",
                "CALL looping() YIELD value RETURN value",
            ))
            .unwrap();

        let error = analyze_error("CALL looping()", &catalog);
        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("looping -> looping")
        ));
    }

    #[test]
    fn depth_limit_reports_the_complete_boundary() {
        let catalog = Catalog::new();
        for index in 0..MAX_PROCEDURE_CALL_DEPTH {
            catalog
                .register_procedure(definition(
                    &format!("p{index}"),
                    format!("CALL p{}() YIELD value RETURN value", index + 1),
                ))
                .unwrap();
        }
        catalog
            .register_procedure(definition(
                &format!("p{MAX_PROCEDURE_CALL_DEPTH}"),
                "RETURN 1 AS value",
            ))
            .unwrap();

        let error = analyze_error("CALL p0()", &catalog);
        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("procedure call depth exceeded 64")
        ));
    }

    #[test]
    fn unknown_procedure_is_semantic_not_internal() {
        let error = analyze_error("CALL no_such_procedure()", &Catalog::new());
        assert!(matches!(
            error,
            Error::Query(ref query) if query.kind == QueryErrorKind::Semantic
        ));
    }

    #[test]
    fn unsupported_signature_type_is_semantic() {
        let catalog = Catalog::new();
        let mut procedure = definition("typed", "RETURN 1 AS value");
        procedure.returns[0].1 = "MYSTERY".to_string();
        catalog.register_procedure(procedure).unwrap();

        let error = analyze_error("CALL typed()", &catalog);
        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("unsupported type 'MYSTERY'")
        ));
    }

    #[test]
    fn duplicate_signature_names_are_semantic() {
        let catalog = Catalog::new();
        let mut procedure = definition("typed", "RETURN 1 AS value, 2 AS other");
        procedure
            .returns
            .push(("value".to_string(), "INT".to_string()));
        catalog.register_procedure(procedure).unwrap();

        let error = analyze_error("CALL typed()", &catalog);
        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("duplicate return name 'value'")
        ));
    }
}
