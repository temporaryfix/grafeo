//! RDF CDC must share the transaction's publication boundary.

#![cfg(all(
    feature = "lpg",
    feature = "triple-store",
    feature = "sparql",
    feature = "cdc"
))]
#![allow(missing_docs)]

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::cdc::ChangeKind;
use grafeo_engine::{Config, GrafeoDB, GraphModel, RdfHistoryTransitionKind, Term, Triple};

fn all_changes(db: &GrafeoDB) -> Vec<grafeo_engine::cdc::ChangeEvent> {
    db.fixture_changes(EpochId::new(0)..=EpochId::new(u64::MAX))
        .expect("CDC history")
}

#[test]
fn rdf_events_are_invisible_until_commit_and_discarded_on_rollback() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { <http://example.org/rolled-back> <http://example.org/p> "dead" . }"#,
        )
        .unwrap();
    assert!(
        all_changes(&db).is_empty(),
        "an active RDF transaction must not publish CDC"
    );
    session.rollback().unwrap();
    assert!(
        all_changes(&db).is_empty(),
        "rolled-back RDF events must be discarded"
    );

    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { <http://example.org/committed> <http://example.org/p> "live" . }"#,
        )
        .unwrap();
    let commit_epoch = session.commit().unwrap();

    let changes = all_changes(&db);
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].kind, ChangeKind::Create);
    assert_eq!(changes[0].epoch, commit_epoch);
    assert_eq!(
        changes[0].triple_subject.as_deref(),
        Some("<http://example.org/committed>")
    );
}

#[test]
#[cfg(feature = "testing-statement-injection")]
fn rdf_cancel_after_write_keeps_committed_state_and_feed_then_publishes_once() {
    use grafeo_common::utils::error::{Error, QueryErrorKind};
    use grafeo_core::execution::QueryExecutionControl;
    use grafeo_engine::QueryCancellationTestPhase;
    use grafeo_engine::query::executor::ExecutionOptions;
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier};

    const SELECT: &str =
        "SELECT ?s ?o WHERE { GRAPH <urn:cdc-cancel:g> { ?s <urn:cdc-cancel:p> ?o } } ORDER BY ?s";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .unwrap();
    let session = db.session();
    session
        .execute_sparql(
            r#"INSERT DATA { GRAPH <urn:cdc-cancel:g> { <urn:cdc-cancel:retained> <urn:cdc-cancel:p> "retained" } }"#,
        )
        .unwrap();
    let before_epoch = db.current_epoch();
    let before_rdf_epoch = db.rdf_store_commit_epoch();
    let before_cut = db.rdf_history_cut(before_rdf_epoch).unwrap();
    assert_eq!(before_cut.named_graphs.len(), 1);
    assert_eq!(before_cut.named_graphs[0].name(), Some("urn:cdc-cancel:g"));
    let incarnation = before_cut.named_graphs[0].incarnation();
    assert_eq!(before_cut.quads.len(), 1);
    assert_eq!(before_cut.quads[0].graph_incarnation, incarnation);
    let before_history = db.rdf_dataset_history().unwrap();
    let before_high_water = db.rdf_store().next_graph_incarnation();
    let before_rows = vec![vec![
        Value::from("urn:cdc-cancel:retained"),
        Value::from("retained"),
    ]];
    let before_events = all_changes(&db);
    assert_eq!(before_events.len(), 1);
    assert_eq!(before_events[0].graph_incarnation, Some(incarnation));
    let before_feed = serde_json::to_value(&before_events).unwrap();

    let assert_committed_unchanged = || {
        assert_eq!(db.current_epoch(), before_epoch);
        assert_eq!(db.rdf_store_commit_epoch(), before_rdf_epoch);
        assert_eq!(db.rdf_history_cut(before_rdf_epoch).unwrap(), before_cut);
        let history = db.rdf_dataset_history().unwrap();
        assert_eq!(history.graph_lives(), before_history.graph_lives());
        assert_eq!(history.quad_versions(), before_history.quad_versions());
        assert_eq!(db.rdf_store().next_graph_incarnation(), before_high_water);
        assert_eq!(db.execute_sparql(SELECT).unwrap().rows(), &before_rows);
        assert_eq!(serde_json::to_value(all_changes(&db)).unwrap(), before_feed);
    };
    assert_committed_unchanged();

    let reached = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    session.set_query_cancellation_test_hook(
        QueryCancellationTestPhase::BeforeStatementCompletion,
        Arc::clone(&reached),
        Arc::clone(&release),
    );
    let control = QueryExecutionControl::new();
    let cancel = control.cancellation_handle();
    let result = std::thread::scope(|scope| {
        let observer = scope.spawn(|| {
            reached.wait();
            // This boundary is after the mutation body and before publication.
            // The final typed cancellation below also proves that the body did
            // not fail: a concrete body error takes precedence over cancellation.
            let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert!(session.in_transaction());
                assert_committed_unchanged();
            }));
            cancel.cancel();
            release.wait();
            // Always release the blocked query before propagating an assertion.
            if let Err(payload) = observed {
                std::panic::resume_unwind(payload);
            }
        });
        let result = session.execute_with_options(
            r#"INSERT DATA { GRAPH <urn:cdc-cancel:g> { <urn:cdc-cancel:discarded> <urn:cdc-cancel:p> "pending" } }"#,
            HashMap::new(),
            ExecutionOptions {
                control,
                language: Some("sparql".into()),
                ..ExecutionOptions::default()
            },
        );
        observer.join().unwrap();
        result
    });
    assert!(
        matches!(&result, Err(Error::Query(error)) if error.kind == QueryErrorKind::Cancelled),
        "{result:?}"
    );
    assert!(!session.in_transaction());
    assert!(!db.is_durability_poisoned());
    assert_committed_unchanged();

    session
        .execute_sparql(
            r#"INSERT DATA { GRAPH <urn:cdc-cancel:g> { <urn:cdc-cancel:accepted> <urn:cdc-cancel:p> "accepted" } }"#,
        )
        .unwrap();
    let committed = db.current_epoch();
    assert!(committed > before_epoch);
    assert_eq!(db.rdf_store_commit_epoch(), committed);
    assert_eq!(db.rdf_store().next_graph_incarnation(), before_high_water);
    assert_eq!(
        db.execute_sparql(SELECT).unwrap().rows(),
        &[
            vec![
                Value::from("urn:cdc-cancel:accepted"),
                Value::from("accepted")
            ],
            before_rows[0].clone(),
        ]
    );
    assert_eq!(db.rdf_history_cut(before_rdf_epoch).unwrap(), before_cut);
    let after_cut = db.rdf_history_cut(committed).unwrap();
    assert_eq!(after_cut.named_graphs, before_cut.named_graphs);
    assert_eq!(after_cut.quads.len(), 2);
    assert!(
        after_cut
            .quads
            .iter()
            .all(|quad| quad.graph_incarnation == incarnation)
    );
    let events = all_changes(&db);
    assert_eq!(events.len(), 2, "the follow-up publishes exactly once");
    assert_eq!(serde_json::to_value(&events[..1]).unwrap(), before_feed);
    let accepted = &events[1];
    assert_eq!(accepted.kind, ChangeKind::Create);
    assert_eq!(accepted.epoch, committed);
    assert_eq!(accepted.graph_incarnation, Some(incarnation));
    assert_eq!(accepted.triple_graph.as_deref(), Some("urn:cdc-cancel:g"));
    assert_eq!(
        accepted.triple_subject.as_deref(),
        Some("<urn:cdc-cancel:accepted>")
    );
    assert_eq!(
        accepted.triple_predicate.as_deref(),
        Some("<urn:cdc-cancel:p>")
    );
    assert_eq!(accepted.triple_object.as_deref(), Some(r#""accepted""#));
    assert!(accepted.timestamp > before_events[0].timestamp);
}

#[test]
fn rdf_pattern_mutations_attribute_named_graph_cdc() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(
        r#"INSERT DATA {
            GRAPH <http://example.org/source> {
                <http://example.org/s> <http://example.org/p> "value" .
            }
        }"#,
    )
    .unwrap();

    db.execute_sparql(
        r#"INSERT { GRAPH <http://example.org/target> { ?s <http://example.org/p> ?o } }
           WHERE  { GRAPH <http://example.org/source> { ?s <http://example.org/p> ?o } }"#,
    )
    .unwrap();
    let before_delete = db.current_epoch();
    db.execute_sparql(
        r#"DELETE WHERE {
            GRAPH <http://example.org/target> { ?s <http://example.org/p> ?o }
        }"#,
    )
    .unwrap();
    let delete_epoch = db.current_epoch();

    let remaining = db
        .execute_sparql(
            r#"SELECT ?s WHERE {
                GRAPH <http://example.org/target> { ?s <http://example.org/p> ?o }
            }"#,
        )
        .unwrap();
    assert!(
        remaining.rows().is_empty(),
        "DELETE WHERE must mutate the named graph, not merely evaluate its WHERE clause"
    );

    let target_changes: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| event.triple_graph.as_deref() == Some("http://example.org/target"))
        .collect();
    assert_eq!(target_changes.len(), 2);
    assert_eq!(target_changes[0].kind, ChangeKind::Create);
    assert_eq!(target_changes[1].kind, ChangeKind::Delete);

    let durable = db
        .rdf_history_diff(before_delete, delete_epoch)
        .expect("durable history diff for DELETE WHERE");
    assert!(matches!(
        durable.transitions.as_slice(),
        [transition]
            if transition.graph.name() == Some("http://example.org/target")
                && matches!(
                    transition.kind,
                    RdfHistoryTransitionKind::StatementRetracted { .. }
                )
    ));
}

#[test]
fn delete_where_graph_variable_mutates_each_bound_named_graph_exactly_once() {
    const FIRST: &str = "http://example.org/first";
    const SECOND: &str = "http://example.org/second";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            GRAPH <{FIRST}> {{ <http://example.org/a> <http://example.org/p> "one" . }}
            GRAPH <{SECOND}> {{ <http://example.org/b> <http://example.org/p> "two" . }}
        }}"#
    ))
    .unwrap();

    let before_delete = db.current_epoch();
    db.execute_sparql(
        r#"DELETE WHERE {
            GRAPH ?graph { ?subject <http://example.org/p> ?object }
        }"#,
    )
    .unwrap();
    let delete_epoch = db.current_epoch();

    assert!(
        db.execute_sparql(
            r#"SELECT ?graph ?subject WHERE {
                GRAPH ?graph { ?subject <http://example.org/p> ?object }
            }"#,
        )
        .unwrap()
        .rows()
        .is_empty(),
        "the graph binding must select the mutation target for every solution"
    );

    let mut volatile_graphs: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| event.kind == ChangeKind::Delete)
        .filter_map(|event| event.triple_graph)
        .collect();
    volatile_graphs.sort();
    assert_eq!(volatile_graphs, vec![FIRST, SECOND]);

    let durable = db
        .rdf_history_diff(before_delete, delete_epoch)
        .expect("durable graph-variable DELETE WHERE diff");
    let mut durable_graphs: Vec<_> = durable
        .transitions
        .iter()
        .filter(|transition| {
            matches!(
                transition.kind,
                RdfHistoryTransitionKind::StatementRetracted { .. }
            )
        })
        .filter_map(|transition| transition.graph.name())
        .collect();
    durable_graphs.sort_unstable();
    assert_eq!(durable_graphs, vec![FIRST, SECOND]);
}

#[test]
fn modify_graph_variable_keeps_delete_insert_and_cdc_on_each_bound_graph() {
    const FIRST: &str = "http://example.org/modify-first";
    const SECOND: &str = "http://example.org/modify-second";
    const PREDICATE: &str = "http://example.org/status";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            GRAPH <{FIRST}> {{ <http://example.org/a> <{PREDICATE}> "old-a" . }}
            GRAPH <{SECOND}> {{ <http://example.org/b> <{PREDICATE}> "old-b" . }}
        }}"#
    ))
    .unwrap();

    let before_modify = db.current_epoch();
    db.execute_sparql(&format!(
        r#"DELETE {{ GRAPH ?graph {{ ?subject <{PREDICATE}> ?old }} }}
           INSERT {{ GRAPH ?graph {{ ?subject <{PREDICATE}> "new" }} }}
           WHERE  {{ GRAPH ?graph {{ ?subject <{PREDICATE}> ?old }} }}"#
    ))
    .unwrap();
    let modify_epoch = db.current_epoch();

    let rows = db
        .execute_sparql(&format!(
            r#"SELECT ?graph ?subject ?value WHERE {{
                GRAPH ?graph {{ ?subject <{PREDICATE}> ?value }}
            }} ORDER BY ?graph"#
        ))
        .unwrap();
    assert_eq!(rows.rows().len(), 2);
    assert!(rows.rows().iter().all(|row| row[2].as_str() == Some("new")));

    let mut volatile: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| event.epoch == modify_epoch)
        .map(|event| (event.triple_graph.expect("named graph"), event.kind))
        .collect();
    volatile.sort_by(|left, right| {
        left.0.cmp(&right.0).then_with(|| {
            let rank = |kind: &ChangeKind| match kind {
                ChangeKind::Delete => 0,
                ChangeKind::Create => 1,
                ChangeKind::Update => 2,
                _ => 3,
            };
            rank(&left.1).cmp(&rank(&right.1))
        })
    });
    assert_eq!(
        volatile,
        vec![
            (FIRST.to_string(), ChangeKind::Delete),
            (FIRST.to_string(), ChangeKind::Create),
            (SECOND.to_string(), ChangeKind::Delete),
            (SECOND.to_string(), ChangeKind::Create),
        ]
    );

    let durable = db
        .rdf_history_diff(before_modify, modify_epoch)
        .expect("durable graph-variable MODIFY diff");
    assert_eq!(durable.transitions.len(), 4);
    for graph in [FIRST, SECOND] {
        let transitions: Vec<_> = durable
            .transitions
            .iter()
            .filter(|transition| transition.graph.name() == Some(graph))
            .collect();
        assert_eq!(transitions.len(), 2);
        assert!(matches!(
            transitions[0].kind,
            RdfHistoryTransitionKind::StatementRetracted { .. }
        ));
        assert!(matches!(
            transitions[1].kind,
            RdfHistoryTransitionKind::StatementAsserted { .. }
        ));
    }
}

#[test]
fn modify_filter_uses_the_selected_physical_binding() {
    const GRAPH: &str = "http://example.org/filter-target";
    const PREDICATE: &str = "http://example.org/filter-status";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{ GRAPH <{GRAPH}> {{
            <http://example.org/filter-a> <{PREDICATE}> "old" .
            <http://example.org/filter-b> <{PREDICATE}> "old" .
            <http://example.org/filter-c> <{PREDICATE}> "old" .
        }} }}"#
    ))
    .unwrap();

    let scan = db
        .execute_sparql(&format!(
            r#"SELECT ?subject WHERE {{ GRAPH <{GRAPH}> {{
                ?subject <{PREDICATE}> "old"
            }} }}"#
        ))
        .unwrap();
    assert_eq!(scan.rows().len(), 3);
    let selected = scan.rows()[1][0]
        .as_str()
        .expect("subject IRI binding")
        .to_string();

    let before_modify = db.current_epoch();
    db.execute_sparql(&format!(
        r#"DELETE {{ GRAPH <{GRAPH}> {{ ?subject <{PREDICATE}> "old" }} }}
           INSERT {{ GRAPH <{GRAPH}> {{ ?subject <{PREDICATE}> "new" }} }}
           WHERE  {{ GRAPH <{GRAPH}> {{ ?subject <{PREDICATE}> "old" }}
                    FILTER (?subject = <{selected}>) }}"#
    ))
    .unwrap();
    let modify_epoch = db.current_epoch();

    let rows = db
        .execute_sparql(&format!(
            r#"SELECT ?subject ?status WHERE {{ GRAPH <{GRAPH}> {{
                ?subject <{PREDICATE}> ?status
            }} }}"#
        ))
        .unwrap();
    assert_eq!(rows.rows().len(), 3);
    for row in rows.rows() {
        let subject = row[0].as_str().expect("subject IRI");
        let status = row[1].as_str().expect("status literal");
        assert_eq!(status, if subject == selected { "new" } else { "old" });
    }

    let epoch_events: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| event.epoch == modify_epoch)
        .collect();
    assert_eq!(epoch_events.len(), 2);
    assert!(epoch_events.iter().all(|event| {
        event.triple_subject.as_deref() == Some(format!("<{selected}>").as_str())
    }));

    let durable = db
        .rdf_history_diff(before_modify, modify_epoch)
        .expect("filtered MODIFY durable diff");
    assert_eq!(durable.transitions.len(), 2);
    assert!(durable.transitions.iter().all(|transition| {
        let (RdfHistoryTransitionKind::StatementAsserted { quad, .. }
        | RdfHistoryTransitionKind::StatementRetracted { quad, .. }) = &transition.kind
        else {
            return false;
        };
        quad.triple()
            .subject()
            .as_iri()
            .is_some_and(|iri| iri.as_str() == selected)
    }));
}

#[test]
fn graph_templates_accept_only_bound_iris() {
    const VALID_SCAN_GRAPH: &str = "http://example.org/from-scan";
    const VALUES_GRAPH: &str = "http://example.org/from-values";
    const BIND_GRAPH: &str = "http://example.org/from-bind";
    const IRI_FUNCTION_GRAPH: &str = "http://example.org/from-iri-function";
    const ALIAS_GRAPH: &str = "http://example.org/from-alias";
    const MARKER: &str = "http://example.org/graph-marker";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <http://example.org/source-iri> <http://example.org/candidate> <{VALID_SCAN_GRAPH}> .
            <http://example.org/source-string> <http://example.org/candidate> "literal-graph" .
            <http://example.org/source-typed> <http://example.org/candidate> "42"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <http://example.org/source-blank> <http://example.org/candidate> _:not-a-graph .
        }}"#
    ))
    .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&format!(
            r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> ?graph }} }}
               WHERE  {{ ?source <http://example.org/candidate> ?graph }}"#
        ))
        .expect("non-IRI graph bindings omit their template quad");
    session.commit().unwrap();

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "values" }} }}
           WHERE  {{ VALUES ?graph {{ <{VALUES_GRAPH}> "values-literal-graph" }} }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "bind" }} }}
           WHERE  {{ BIND (<{BIND_GRAPH}> AS ?graph) }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "literal" }} }}
           WHERE  {{ BIND ("bind-literal-graph" AS ?graph) }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?unbound {{ <{MARKER}> <http://example.org/value> "unbound" }} }}
           WHERE  {{ VALUES ?other {{ "bound" }} }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "iri-function" }} }}
           WHERE  {{ BIND (IRI("{IRI_FUNCTION_GRAPH}") AS ?graph) }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "alias" }} }}
           WHERE  {{ VALUES ?source {{ <{ALIAS_GRAPH}> }} BIND (?source AS ?graph) }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "invalid" }} }}
           WHERE  {{ VALUES ?graph {{
               "http://example.org/any-uri-literal"^^<http://www.w3.org/2001/XMLSchema#anyURI>
               7
           }} }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "str" }} }}
           WHERE  {{ BIND (STR(<http://example.org/str-is-a-literal>) AS ?graph) }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "blank" }} }}
           WHERE  {{ BIND (BNODE("graph") AS ?graph) }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{MARKER}> <http://example.org/value> "forged" }} }}
           WHERE  {{ VALUES (?graph ?__term_graph) {{
               ("forged-visible-graph" "<http://example.org/forged-target>")
           }} }}"#
    ))
    .unwrap();

    for graph in [
        VALID_SCAN_GRAPH,
        VALUES_GRAPH,
        BIND_GRAPH,
        IRI_FUNCTION_GRAPH,
        ALIAS_GRAPH,
    ] {
        assert_eq!(
            db.rdf_store()
                .graph(graph)
                .unwrap_or_else(|| {
                    panic!(
                        "IRI graph {graph} should exist; available: {:?}",
                        db.rdf_store().graph_names()
                    )
                })
                .len(),
            1
        );
    }
    for forbidden in [
        "literal-graph",
        "42",
        "values-literal-graph",
        "bind-literal-graph",
        "http://example.org/any-uri-literal",
        "7",
        "http://example.org/str-is-a-literal",
        "_:bgraph",
        "forged-visible-graph",
        "http://example.org/forged-target",
    ] {
        assert!(
            db.rdf_store().graph(forbidden).is_none(),
            "non-IRI binding must not create graph {forbidden}"
        );
    }
    assert!(
        db.rdf_store()
            .graph_names()
            .iter()
            .all(|graph| !graph.starts_with("_:")),
        "a blank-node binding must never become a graph name"
    );
}

#[test]
fn graph_template_preserves_iri_identity_through_subselect_projection() {
    const GRAPH: &str = "http://example.org/from-subselect";
    const FUNCTION_GRAPH: &str = "http://example.org/from-subselect-function";
    const SUBJECT: &str = "http://example.org/subselect-subject";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{SUBJECT}> <http://example.org/value> "projected" }} }}
           WHERE  {{ {{ SELECT (<{GRAPH}> AS ?graph) WHERE {{ }} }} }}"#
    ))
    .expect("an IRI projected by a subselect remains an IRI graph name");

    assert_eq!(
        db.rdf_store()
            .graph(GRAPH)
            .expect("projected IRI graph")
            .len(),
        1
    );

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <{SUBJECT}> <http://example.org/function> "projected" }} }}
           WHERE  {{ {{ SELECT (IRI("{FUNCTION_GRAPH}") AS ?graph) WHERE {{ }} }} }}"#
    ))
    .expect("a computed IRI projection retains exact term identity");
    assert_eq!(
        db.rdf_store()
            .graph(FUNCTION_GRAPH)
            .expect("computed projected IRI graph")
            .len(),
        1
    );

    db.execute_sparql(
        r#"INSERT { <http://example.org/subselect-literal-subject>
                       <http://example.org/subselect-literal-predicate> ?object }
           WHERE  { { SELECT (STR(<http://example.org/literal-object>) AS ?object) WHERE { } } }"#,
    )
    .expect("a projected STR result remains an RDF literal");
    let triples = db.rdf_store().triples();
    let projected = triples
        .iter()
        .find(|triple| {
            triple
                .subject()
                .as_iri()
                .is_some_and(|iri| iri.as_str() == "http://example.org/subselect-literal-subject")
        })
        .expect("projected STR object triple");
    assert!(projected.object().is_literal());
}

#[test]
fn mutation_helper_bindings_preserve_capability_without_weakening_exact_joins() {
    const PREDICATE: &str = "http://example.org/helper-join";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <http://example.org/source> <{PREDICATE}> <http://example.org/same-text>
        }}"#
    ))
    .unwrap();

    db.execute_sparql(
        r#"INSERT { <http://example.org/helper-result> <http://example.org/value> "ok" }
           WHERE  { BIND (<urn:unsupported-helper>("x") AS ?unused) }"#,
    )
    .expect("an unrelated extension helper retains the established update surface");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("http://example.org/helper-result"),
        Term::iri("http://example.org/value"),
        Term::literal("ok"),
    )));

    db.execute_sparql(&format!(
        r#"INSERT {{ <http://example.org/false-join> <http://example.org/value> "bad" }}
           WHERE  {{
               <http://example.org/source> <{PREDICATE}> ?join
               VALUES ?join {{ "http://example.org/same-text" }}
           }}"#
    ))
    .expect("term-exact helper join executes");
    assert!(
        !db.rdf_store().contains(&Triple::new(
            Term::iri("http://example.org/false-join"),
            Term::iri("http://example.org/value"),
            Term::literal("bad"),
        )),
        "an IRI and literal with the same visible string must not join"
    );

    db.execute_sparql(
        r#"INSERT { GRAPH ?graph {
               <http://example.org/unbound-alias> <http://example.org/value> "omitted"
           } }
           WHERE { BIND (?missing AS ?graph) }"#,
    )
    .expect("an absent BIND source remains an unbound expression result");

    db.execute_sparql(
        r#"INSERT { <http://example.org/empty-values> <http://example.org/value> "bad" }
           WHERE  { VALUES ?nothing { } }"#,
    )
    .expect("an empty VALUES table is a valid zero-solution pattern");
    assert!(!db.rdf_store().contains(&Triple::new(
        Term::iri("http://example.org/empty-values"),
        Term::iri("http://example.org/value"),
        Term::literal("bad"),
    )));
}

#[test]
fn mutation_values_with_undef_join_on_exact_rdf_terms() {
    const SOURCE: &str = "urn:values-undef-source";
    const SELECTED: &str = "urn:values-undef-selected";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <urn:values-iri-subject> <{SOURCE}> <urn:values-same-text> .
            <urn:values-literal-subject> <{SOURCE}> "urn:values-same-text" .
        }}"#
    ))
    .unwrap();

    db.execute_sparql(&format!(
        r#"INSERT {{ ?subject <{SELECTED}> "selected" }}
           WHERE  {{
               ?subject <{SOURCE}> ?value
               VALUES (?value ?unused) {{ (<urn:values-same-text> UNDEF) }}
           }}"#
    ))
    .expect("VALUES rows containing UNDEF still join on exact RDF terms");

    let matches = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: None,
            predicate: Some(Term::iri(SELECTED)),
            object: None,
        });
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].subject(), &Term::iri("urn:values-iri-subject"));
}

#[test]
fn mutation_filters_use_exact_rdf_term_kinds() {
    const SOURCE: &str = "http://example.org/filter-term-source";
    const IRI_SUBJECT: &str = "http://example.org/filter-term-iri";
    const LITERAL_SUBJECT: &str = "http://example.org/filter-term-literal";
    const BLANK_SUBJECT: &str = "http://example.org/filter-term-blank";
    const BLANK_SHAPE_SOURCE: &str = "http://example.org/filter-blank-shape-source";
    const BLANK_SHAPE_LITERAL_SUBJECT: &str = "http://example.org/filter-blank-shape-literal";
    const SAME_TEXT: &str = "http://example.org/same-visible-text";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <{IRI_SUBJECT}> <{SOURCE}> <{SAME_TEXT}> .
            <{LITERAL_SUBJECT}> <{SOURCE}> "{SAME_TEXT}" .
            <{BLANK_SUBJECT}> <{SOURCE}> _:source-blank .
            <{BLANK_SUBJECT}> <{BLANK_SHAPE_SOURCE}> _:source-blank .
            <{BLANK_SHAPE_LITERAL_SUBJECT}> <{BLANK_SHAPE_SOURCE}> "_:source-blank" .
        }}"#
    ))
    .unwrap();

    for (predicate, filter) in [
        (
            "http://example.org/selected-equal",
            format!("?value = <{SAME_TEXT}>"),
        ),
        (
            "http://example.org/selected-same-term",
            format!("sameTerm(?value, <{SAME_TEXT}>)"),
        ),
        (
            "http://example.org/selected-iri",
            "isIRI(?value)".to_string(),
        ),
        (
            "http://example.org/selected-uri",
            "isURI(?value)".to_string(),
        ),
    ] {
        db.execute_sparql(&format!(
            r#"INSERT {{ ?subject <{predicate}> "selected" }}
               WHERE  {{ ?subject <{SOURCE}> ?value FILTER ({filter}) }}"#
        ))
        .expect("term-sensitive IRI filter update");
        let matches = db
            .rdf_store()
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri(predicate)),
                object: None,
            });
        assert_eq!(matches.len(), 1, "{filter} must select one RDF term kind");
        assert_eq!(matches[0].subject(), &Term::iri(IRI_SUBJECT));
    }

    db.execute_sparql(&format!(
        r#"INSERT {{ ?subject <http://example.org/selected-not-equal> "selected" }}
           WHERE  {{ ?subject <{SOURCE}> ?value FILTER (?value != <{SAME_TEXT}>) }}"#
    ))
    .expect("term-sensitive inequality update");
    let not_equal_subjects: std::collections::HashSet<_> = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://example.org/selected-not-equal")),
            object: None,
        })
        .into_iter()
        .map(|triple| triple.subject().clone())
        .collect();
    assert_eq!(
        not_equal_subjects,
        std::collections::HashSet::from([Term::iri(LITERAL_SUBJECT), Term::iri(BLANK_SUBJECT),])
    );

    for (predicate, filter, expected) in [
        (
            "http://example.org/selected-literal",
            "isLiteral(?value)",
            LITERAL_SUBJECT,
        ),
        (
            "http://example.org/selected-blank",
            "isBlank(?value)",
            BLANK_SUBJECT,
        ),
    ] {
        db.execute_sparql(&format!(
            r#"INSERT {{ ?subject <{predicate}> "selected" }}
               WHERE  {{ ?subject <{SOURCE}> ?value FILTER ({filter}) }}"#
        ))
        .expect("term-kind filter update");
        let matches = db
            .rdf_store()
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri(predicate)),
                object: None,
            });
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].subject(), &Term::iri(expected));
    }

    db.execute_sparql(&format!(
        r#"INSERT {{ ?subject <http://example.org/selected-real-blank> "selected" }}
           WHERE  {{ ?subject <{BLANK_SHAPE_SOURCE}> ?value FILTER (isBlank(?value)) }}"#
    ))
    .expect("blank-kind filter must not inspect a literal's lexical prefix");
    let matches = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://example.org/selected-real-blank")),
            object: None,
        });
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].subject(), &Term::iri(BLANK_SUBJECT));
}

#[test]
fn mutation_filter_value_equality_obeys_rdf_datatypes() {
    const SOURCE: &str = "http://example.org/filter-datatype-source";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <urn:string-01> <{SOURCE}> "01" .
            <urn:string-1> <{SOURCE}> "1" .
            <urn:number-01> <{SOURCE}> "01"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <urn:number-1> <{SOURCE}> 1 .
            <urn:byte-leading-zeroes> <{SOURCE}> "+0000000000000000000000000000000000000000000000000000000000000001"^^<http://www.w3.org/2001/XMLSchema#byte> .
            <urn:unknown-a> <{SOURCE}> "a"^^<urn:unknown-datatype> .
            <urn:language-upper> <{SOURCE}> "hello"@EN .
            <urn:large-a> <{SOURCE}> "9007199254740992"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <urn:large-b> <{SOURCE}> "9007199254740993"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <urn:decimal-a> <{SOURCE}> "1.0000000000000000001"^^<http://www.w3.org/2001/XMLSchema#decimal> .
            <urn:decimal-b> <{SOURCE}> "1.0000000000000000002"^^<http://www.w3.org/2001/XMLSchema#decimal> .
            <urn:nan> <{SOURCE}> "NaN"^^<http://www.w3.org/2001/XMLSchema#double> .
            <urn:invalid-integer> <{SOURCE}> "1.5"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <urn:precise-time> <{SOURCE}> "2000-01-01T00:00:00.0000001Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:precise-time-other> <{SOURCE}> "2000-01-01T00:00:00.0000002Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:precise-time-offset> <{SOURCE}> "1999-12-31T19:00:00.00000010-05:00"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:end-of-day> <{SOURCE}> "1999-12-31T24:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:next-midnight> <{SOURCE}> "2000-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:invalid-negative-hour> <{SOURCE}> "2000-01-01T-1:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:valid-previous-hour> <{SOURCE}> "1999-12-31T23:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:positive-infinity> <{SOURCE}> "+INF"^^<http://www.w3.org/2001/XMLSchema#double> .
            <urn:invalid-infinity> <{SOURCE}> "infinity"^^<http://www.w3.org/2001/XMLSchema#double> .
            <urn:date-z> <{SOURCE}> "2004-12-25Z"^^<http://www.w3.org/2001/XMLSchema#date> .
            <urn:date-plus-zero> <{SOURCE}> "2004-12-25+00:00"^^<http://www.w3.org/2001/XMLSchema#date> .
            <urn:large-date-a> <{SOURCE}> "6000000-01-01"^^<http://www.w3.org/2001/XMLSchema#date> .
            <urn:large-date-b> <{SOURCE}> "6000001-01-01"^^<http://www.w3.org/2001/XMLSchema#date> .
            <urn:unbounded-date> <{SOURCE}> "100000000000000000000000000000000000000000000000000-01-01Z"^^<http://www.w3.org/2001/XMLSchema#date> .
            <urn:unbounded-date-time> <{SOURCE}> "100000000000000000000000000000000000000000000000000-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            <urn:negative-zero-date> <{SOURCE}> "-0000-02-29Z"^^<http://www.w3.org/2001/XMLSchema#date> .
        }}"#
    ))
    .unwrap();

    let select_subjects = |predicate: &str, filter: &str| {
        db.execute_sparql(&format!(
            r#"INSERT {{ ?subject <{predicate}> "selected" }}
               WHERE  {{ ?subject <{SOURCE}> ?value FILTER ({filter}) }}"#
        ))
        .expect("datatype-sensitive mutation filter");
        db.rdf_store()
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri(predicate)),
                object: None,
            })
            .into_iter()
            .map(|triple| triple.subject().clone())
            .collect::<std::collections::HashSet<_>>()
    };

    assert_eq!(
        select_subjects("urn:selected-string", r#"?value = "1""#),
        std::collections::HashSet::from([Term::iri("urn:string-1")]),
        "numeric-looking xsd:string values compare lexically"
    );
    assert_eq!(
        select_subjects("urn:selected-number", "?value = 1"),
        std::collections::HashSet::from([
            Term::iri("urn:number-01"),
            Term::iri("urn:number-1"),
            Term::iri("urn:byte-leading-zeroes"),
        ]),
        "numeric equality compares values across lexical representations"
    );
    assert_eq!(
        select_subjects("urn:selected-number-term", "sameTerm(?value, 1)"),
        std::collections::HashSet::from([Term::iri("urn:number-1")]),
        "sameTerm preserves the numeric literal's lexical identity"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-language-term",
            r#"sameTerm(?value, "hello"@en)"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:language-upper")]),
        "RDF language tags are identity-normalized case-insensitively"
    );
    assert!(
        select_subjects(
            "urn:selected-unknown-not-equal",
            r#"?subject = <urn:unknown-a> && ?value != "z"^^<urn:unknown-datatype>"#,
        )
        .is_empty(),
        "an unsupported typed-literal equality error must survive !="
    );
    assert_eq!(
        select_subjects(
            "urn:selected-large-integer",
            r#"?value = "9007199254740992"^^<http://www.w3.org/2001/XMLSchema#integer>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:large-a")]),
        "integer equality must not round through f64"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-precise-decimal",
            r#"?value = "1.0000000000000000001"^^<http://www.w3.org/2001/XMLSchema#decimal>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:decimal-a")]),
        "decimal equality must preserve arbitrary lexical precision"
    );
    assert!(
        select_subjects(
            "urn:selected-nan",
            r#"?value = "NaN"^^<http://www.w3.org/2001/XMLSchema#double>"#,
        )
        .is_empty(),
        "NaN is not value-equal to itself"
    );
    assert!(
        select_subjects(
            "urn:selected-invalid-integer",
            r#"?value = "1.5"^^<http://www.w3.org/2001/XMLSchema#integer>"#,
        )
        .contains(&Term::iri("urn:invalid-integer")),
        "identical ill-typed literals are the same RDF term before value comparison"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-byte-leading-zeroes",
            r#"?value = "1"^^<http://www.w3.org/2001/XMLSchema#byte>"#,
        ),
        std::collections::HashSet::from([
            Term::iri("urn:number-01"),
            Term::iri("urn:number-1"),
            Term::iri("urn:byte-leading-zeroes"),
        ]),
        "bounded integer facets inspect the canonical value, not raw lexical width"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-precise-time",
            r#"?value = "2000-01-01T00:00:00.0000001Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>"#,
        ),
        std::collections::HashSet::from([
            Term::iri("urn:precise-time"),
            Term::iri("urn:precise-time-offset"),
        ]),
        "dateTime equality keeps arbitrary precision and normalizes offsets"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-end-of-day",
            r#"?value = "2000-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>"#,
        ),
        std::collections::HashSet::from([
            Term::iri("urn:end-of-day"),
            Term::iri("urn:next-midnight"),
        ]),
        "XSD 24:00:00 normalizes to the following midnight"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-valid-hour",
            r#"?value = "1999-12-31T23:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:valid-previous-hour")]),
        "signed time components are not valid XSD dateTime lexicals"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-positive-infinity",
            r#"?value = "+INF"^^<http://www.w3.org/2001/XMLSchema#double>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:positive-infinity")]),
        "XSD +INF is accepted without accepting Rust-only infinity spellings"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-zoned-date",
            r#"?value = "2004-12-25Z"^^<http://www.w3.org/2001/XMLSchema#date>"#,
        ),
        std::collections::HashSet::from(
            [Term::iri("urn:date-z"), Term::iri("urn:date-plus-zero"),]
        ),
        "XSD date equality accepts and normalizes timezone suffixes"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-large-date",
            r#"?value = "6000000-01-01"^^<http://www.w3.org/2001/XMLSchema#date>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:large-date-a")]),
        "large XSD dates must not collide through a saturated day count"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-unbounded-date",
            r#"?value = "100000000000000000000000000000000000000000000000000-01-01+00:00"^^<http://www.w3.org/2001/XMLSchema#date>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:unbounded-date")]),
        "XSD's unbounded year space remains exact across timezone lexicals"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-unbounded-date-time",
            r#"?value = "99999999999999999999999999999999999999999999999999-12-31T19:00:00-05:00"^^<http://www.w3.org/2001/XMLSchema#dateTime>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:unbounded-date-time")]),
        "dateTime normalization crosses an unbounded-year boundary without truncation"
    );
    assert_eq!(
        select_subjects(
            "urn:selected-negative-zero-date",
            r#"?value = "0000-02-29+00:00"^^<http://www.w3.org/2001/XMLSchema#date>"#,
        ),
        std::collections::HashSet::from([Term::iri("urn:negative-zero-date")]),
        "XSD 1.1 accepts signed year zero and applies Gregorian leap rules"
    );

    db.execute_sparql(
        r#"INSERT DATA {
            <urn:plain-language-control> <urn:language-control> "hello" .
            <urn:tagged-language-control> <urn:language-control> "hello"@en .
        }"#,
    )
    .unwrap();
    db.execute_sparql(
        r#"INSERT { ?subject <urn:language-different> "selected" }
           WHERE  {
               ?subject <urn:language-control> ?value
               FILTER (?value != "hello"@en)
           }"#,
    )
    .expect("Grafeo's LangTagAwareness extension remains available in mutations");
    let language_different = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: None,
            predicate: Some(Term::iri("urn:language-different")),
            object: None,
        });
    assert_eq!(language_different.len(), 1);
    assert_eq!(
        language_different[0].subject(),
        &Term::iri("urn:plain-language-control")
    );

    db.execute_sparql(
        r#"INSERT { <urn:language-vs-numeric> <urn:status> "selected" }
           WHERE  { FILTER ("1"@en != 1) }"#,
    )
    .expect("language-tagged and numeric literals are determinably different RDF terms");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:language-vs-numeric"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:error-or-true> <urn:status> "selected" }
           WHERE  {
               FILTER (("a"^^<urn:unknown-datatype> = "b"^^<urn:unknown-datatype>) || true)
           }"#,
    )
    .expect("SPARQL OR truth table preserves true over expression error");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:error-or-true"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));
    db.execute_sparql(
        r#"INSERT { <urn:error-and-false> <urn:status> "bad" }
           WHERE  {
               FILTER (("a"^^<urn:unknown-datatype> = "b"^^<urn:unknown-datatype>) && false)
           }"#,
    )
    .expect("SPARQL AND truth table preserves false over expression error");
    assert!(!db.rdf_store().contains(&Triple::new(
        Term::iri("urn:error-and-false"),
        Term::iri("urn:status"),
        Term::literal("bad"),
    )));
}

#[test]
fn mutation_membership_and_is_numeric_follow_rdf_semantics() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    for (subject, filter) in [
        (
            "urn:in-numeric-value",
            r#"1 IN ("1.0"^^<http://www.w3.org/2001/XMLSchema#decimal>)"#,
        ),
        (
            "urn:not-in-term-kind",
            r#"<urn:member> NOT IN ("urn:member")"#,
        ),
        ("urn:in-language-case", r#""a"@EN IN ("a"@en)"#),
        (
            "urn:in-string-syntax",
            r#""a" IN ("a"^^<http://www.w3.org/2001/XMLSchema#string>)"#,
        ),
        ("urn:in-error-before-match", "2 IN (1 / 0, 2)"),
        ("urn:in-error-after-match", "2 IN (2, 1 / 0)"),
        (
            "urn:in-unsupported-same-term-after-error",
            r#""a"^^<urn:unsupported> IN ("b"^^<urn:unsupported>, "a"^^<urn:unsupported>)"#,
        ),
        ("urn:not-in-empty", "2 NOT IN ()"),
        (
            "urn:numeric-unsigned-long",
            r#"ISNUMERIC("18446744073709551615"^^<http://www.w3.org/2001/XMLSchema#unsignedLong>)"#,
        ),
        (
            "urn:numeric-nan",
            r#"ISNUMERIC("NaN"^^<http://www.w3.org/2001/XMLSchema#double>)"#,
        ),
        (
            "urn:numeric-strdt-infinity",
            r#"ISNUMERIC(STRDT("INF", <http://www.w3.org/2001/XMLSchema#double>))"#,
        ),
        (
            "urn:nonnumeric-strdt-ill-typed",
            r#"!ISNUMERIC(STRDT("pumpkin", <http://www.w3.org/2001/XMLSchema#integer>))"#,
        ),
        ("urn:numeric-computed-infinity", "ISNUMERIC(1e308 * 1e308)"),
        ("urn:nonnumeric-string", r#"!ISNUMERIC("12")"#),
        (
            "urn:nonnumeric-facet",
            r#"!ISNUMERIC("256"^^<http://www.w3.org/2001/XMLSchema#unsignedByte>)"#,
        ),
        (
            "urn:nonnumeric-lexical",
            r#"!ISNUMERIC("pumpkin"^^<http://www.w3.org/2001/XMLSchema#integer>)"#,
        ),
    ] {
        db.execute_sparql(&format!(
            r#"INSERT {{ <{subject}> <urn:status> "selected" }}
               WHERE  {{ FILTER ({filter}) }}"#,
        ))
        .unwrap_or_else(|error| {
            panic!("membership/numeric selector must execute: {filter}: {error}")
        });
        assert!(
            db.rdf_store().contains(&Triple::new(
                Term::iri(subject),
                Term::iri("urn:status"),
                Term::literal("selected"),
            )),
            "selector should be true: {filter}",
        );
    }

    for (subject, filter) in [
        (
            "urn:in-term-kind-mismatch",
            r#"<urn:member> IN ("urn:member")"#,
        ),
        (
            "urn:not-in-numeric-value",
            r#"1 NOT IN ("1.0"^^<http://www.w3.org/2001/XMLSchema#decimal>)"#,
        ),
        ("urn:in-error-without-match", "2 IN (3, 1 / 0)"),
        ("urn:not-in-error-without-match", "2 NOT IN (3, 1 / 0)"),
        (
            "urn:in-unsupported-error-without-match",
            r#""a"^^<urn:unsupported> IN ("b"^^<urn:unsupported>)"#,
        ),
        ("urn:in-empty", "2 IN ()"),
        ("urn:not-in-empty-lhs-error", "(1 / 0) NOT IN ()"),
        ("urn:string-is-not-numeric", r#"ISNUMERIC("12")"#),
        (
            "urn:out-of-range-is-not-numeric",
            r#"ISNUMERIC("256"^^<http://www.w3.org/2001/XMLSchema#unsignedByte>)"#,
        ),
        (
            "urn:ill-typed-is-not-numeric",
            r#"ISNUMERIC("pumpkin"^^<http://www.w3.org/2001/XMLSchema#integer>)"#,
        ),
    ] {
        db.execute_sparql(&format!(
            r#"INSERT {{ <{subject}> <urn:status> "bad" }}
               WHERE  {{ FILTER ({filter}) }}"#,
        ))
        .unwrap_or_else(|error| panic!("false/error selector must execute: {filter}: {error}"));
        assert!(
            !db.rdf_store().contains(&Triple::new(
                Term::iri(subject),
                Term::iri("urn:status"),
                Term::literal("bad"),
            )),
            "selector should be false or an expression error: {filter}",
        );
    }

    db.execute_sparql(
        r#"INSERT DATA {
            <urn:member-iri-row> <urn:member-value> <urn:member> .
            <urn:member-literal-row> <urn:member-value> "urn:member" .
        }"#,
    )
    .unwrap();
    db.execute_sparql(
        r#"INSERT { ?row <urn:member-selected> ?value }
           WHERE  {
               ?row <urn:member-value> ?value .
               FILTER (?value IN (<urn:member>))
           }"#,
    )
    .expect("membership over scanned variables retains exact RDF identity");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:member-iri-row"),
        Term::iri("urn:member-selected"),
        Term::iri("urn:member"),
    )));
    assert!(!db.rdf_store().contains(&Triple::new(
        Term::iri("urn:member-literal-row"),
        Term::iri("urn:member-selected"),
        Term::literal("urn:member"),
    )));

    db.execute_sparql(
        r#"INSERT DATA {
            <urn:numeric-valid-row> <urn:numeric-value> "01"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <urn:numeric-string-row> <urn:numeric-value> "12" .
            <urn:numeric-ill-typed-row> <urn:numeric-value> "pumpkin"^^<http://www.w3.org/2001/XMLSchema#integer> .
            <urn:numeric-out-of-range-row> <urn:numeric-value> "256"^^<http://www.w3.org/2001/XMLSchema#unsignedByte> .
        }"#,
    )
    .unwrap();
    db.execute_sparql(
        r#"INSERT { ?row <urn:numeric-selected> ?value }
           WHERE  {
               ?row <urn:numeric-value> ?value .
               FILTER (ISNUMERIC(?value))
           }"#,
    )
    .expect("isNumeric over scanned terms uses datatype, lexical, and facet identity");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:numeric-valid-row"),
        Term::iri("urn:numeric-selected"),
        Term::typed_literal("01", "http://www.w3.org/2001/XMLSchema#integer"),
    )));
    for subject in [
        "urn:numeric-string-row",
        "urn:numeric-ill-typed-row",
        "urn:numeric-out-of-range-row",
    ] {
        assert!(
            db.rdf_store()
                .find(&grafeo_core::graph::rdf::TriplePattern {
                    subject: Some(Term::iri(subject)),
                    predicate: Some(Term::iri("urn:numeric-selected")),
                    object: None,
                })
                .is_empty(),
            "non-numeric scanned term selected: {subject}",
        );
    }

    db.execute_sparql(
        r#"INSERT { <urn:in-undef-match> <urn:status> "selected" }
           WHERE  {
               VALUES (?value ?missing) { (2 UNDEF) }
               FILTER (?value IN (?missing, 2))
           }"#,
    )
    .expect("a membership match wins over an unbound list-item error");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:in-undef-match"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:in-undef-no-match> <urn:status> "bad" }
           WHERE  {
               VALUES (?value ?missing) { (2 UNDEF) }
               FILTER (?value IN (3, ?missing))
           }"#,
    )
    .expect("an unbound list item becomes a filter error when no item matches");
    assert!(!db.rdf_store().contains(&Triple::new(
        Term::iri("urn:in-undef-no-match"),
        Term::iri("urn:status"),
        Term::literal("bad"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:membership-control-alias> <urn:status> "selected" }
           WHERE  {
               VALUES ?value { <urn:member> }
               BIND (?value IN (<urn:member>) AS ?selected)
               FILTER (?selected)
           }"#,
    )
    .expect("a control-only membership alias is evaluated term-exactly");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:membership-control-alias"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:numeric-control-alias> <urn:status> "selected" }
           WHERE  {
               VALUES ?value { "12"^^<http://www.w3.org/2001/XMLSchema#integer> }
               BIND (ISNUMERIC(?value) AS ?selected)
               FILTER (?selected)
           }"#,
    )
    .expect("a control-only isNumeric alias is evaluated from RDF identity");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:numeric-control-alias"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:unused-numeric-helper> <urn:status> "accepted" }
           WHERE  {
               VALUES ?value { "12" }
               BIND (ISNUMERIC(?value) AS ?unused)
           }"#,
    )
    .expect("an unused exact-capable helper does not require hidden identity columns");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:unused-numeric-helper"),
        Term::iri("urn:status"),
        Term::literal("accepted"),
    )));

    assert!(
        db.execute_sparql(
            r#"INSERT { <urn:opaque-membership> <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>() AS ?value)
                   FILTER (?value IN ("x"))
               }"#,
        )
        .is_err(),
        "a term-sensitive membership selector must fail closed on opaque identity",
    );

    assert!(
        db.execute_sparql(
            r#"INSERT { <urn:opaque-numeric> <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>() AS ?value)
                   BIND (ISNUMERIC(?value) AS ?selected)
                   FILTER (?selected)
               }"#,
        )
        .is_err(),
        "a control-only isNumeric alias must fail closed on opaque identity",
    );

    let arity_error = db
        .execute_sparql(
            r#"INSERT { <urn:numeric-arity> <urn:status> "bad" }
               WHERE  { FILTER (ISNUMERIC(1, 2)) }"#,
        )
        .expect_err("exact isNumeric rejects extra arguments");
    assert!(
        arity_error
            .to_string()
            .contains("ISNUMERIC requires exactly 1 argument"),
        "unexpected isNumeric arity error: {arity_error}",
    );
}

#[test]
fn aggregate_having_keeps_existing_mutation_capability() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(
        r#"INSERT DATA {
            <urn:a> <urn:aggregate-source> "a" .
            <urn:b> <urn:aggregate-source> "b" .
        }"#,
    )
    .unwrap();

    db.execute_sparql(
        r#"INSERT { <urn:aggregate-having> <urn:status> "selected" }
           WHERE  {
               { SELECT (COUNT(?value) AS ?count)
                   WHERE { ?subject <urn:aggregate-source> ?value }
                   HAVING (COUNT(?value) = 2) }
           }"#,
    )
    .expect("aggregate HAVING equality remains executable in a mutation");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:aggregate-having"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:aggregate-having-membership> <urn:status> "selected" }
           WHERE  {
               { SELECT (COUNT(?value) AS ?count)
                   WHERE { ?subject <urn:aggregate-source> ?value }
                   HAVING (
                       ISNUMERIC(COUNT(?value)) &&
                       COUNT(?value) IN (2)
                   ) }
           }"#,
    )
    .expect("aggregate HAVING retains exact numeric and membership semantics");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:aggregate-having-membership"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:unused-aggregate-helper> <urn:status> "accepted" }
           WHERE  {
               { SELECT (COUNT(<urn:opaque-helper>() = "x") AS ?unused) WHERE { } }
           }"#,
    )
    .expect("an unrelated aggregate helper retains the ordinary expression surface");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:unused-aggregate-helper"),
        Term::iri("urn:status"),
        Term::literal("accepted"),
    )));

    db.execute_sparql(
        r#"INSERT DATA {
            <urn:having-iri-row> <urn:having-kind> <urn:same-having-text> .
            <urn:having-literal-row> <urn:having-kind> "urn:same-having-text" .
        }"#,
    )
    .unwrap();
    db.execute_sparql(
        r#"INSERT { <urn:having-results> <urn:selected-kind> ?kind }
           WHERE  {
               { SELECT ?kind (COUNT(*) AS ?count)
                   WHERE { ?row <urn:having-kind> ?kind }
                   GROUP BY ?kind
                   HAVING (?kind = <urn:same-having-text>) }
           }"#,
    )
    .expect("a non-aggregate HAVING operand stays term-exact beside COUNT");
    let selected = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: Some(Term::iri("urn:having-results")),
            predicate: Some(Term::iri("urn:selected-kind")),
            object: None,
        });
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].object(), &Term::iri("urn:same-having-text"));

    db.execute_sparql(
        r#"INSERT { <urn:distinct-identity> <urn:status> "selected" }
           WHERE  {
               { SELECT (COUNT(DISTINCT ?kind) AS ?count)
                   WHERE { ?row <urn:having-kind> ?kind }
                   HAVING (COUNT(DISTINCT ?kind) = 2) }
           }"#,
    )
    .expect("COUNT DISTINCT uses RDF-term identity in a mutation selector");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:distinct-identity"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    for (subject, values) in [
        ("urn:distinct-language-case", r#""a"@en "a"@EN"#),
        (
            "urn:distinct-explicit-string",
            r#""a" "a"^^<http://www.w3.org/2001/XMLSchema#string>"#,
        ),
    ] {
        db.execute_sparql(&format!(
            r#"INSERT {{ <{subject}> <urn:status> "selected" }}
               WHERE  {{
                   {{ SELECT (COUNT(DISTINCT ?value) AS ?count)
                       WHERE {{ VALUES ?value {{ {values} }} }}
                       HAVING (COUNT(DISTINCT ?value) = 1) }}
               }}"#,
        ))
        .unwrap_or_else(|error| {
            panic!("DISTINCT canonicalizes alternate spellings for {subject}: {error}")
        });
        assert!(db.rdf_store().contains(&Triple::new(
            Term::iri(subject),
            Term::iri("urn:status"),
            Term::literal("selected"),
        )));
    }
}

#[test]
fn exact_filter_dependencies_fail_closed_without_rejecting_unused_helpers() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(
        r#"INSERT { <urn:unused-helper> <urn:status> "accepted" }
           WHERE  { BIND (<urn:opaque-helper>("x") = "x" AS ?unused) }"#,
    )
    .expect("an unrelated opaque comparison retains the ordinary helper surface");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:unused-helper"),
        Term::iri("urn:status"),
        Term::literal("accepted"),
    )));

    let ordinary = db
        .execute_sparql(
            r#"SELECT ?kind WHERE { BIND (isIRI("http://ordinary-visible-value") AS ?kind) }"#,
        )
        .expect("ordinary SELECT keeps its established public expression behavior");
    assert_eq!(ordinary.rows(), &[vec![Value::Bool(true)]]);

    db.execute_sparql(
        r#"INSERT { <urn:known-filter-alias> <urn:status> "selected" }
           WHERE  {
               BIND (IF(true, <urn:selected-iri>, "urn:selected-iri") AS ?selector)
               FILTER (isIRI(?selector))
           }"#,
    )
    .expect("a filter-only alias receives a lossless exact companion");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:known-filter-alias"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:branch-local-helper> <urn:status> "selected" }
           WHERE  {
               { BIND (<urn:opaque-helper>("unused") AS ?branch) }
               UNION
               {
                   BIND (IF(true, <urn:branch-iri>, "urn:branch-iri") AS ?branch)
                   FILTER (isIRI(?branch))
               }
           }"#,
    )
    .expect("a selector in one UNION branch does not reject an unrelated opaque branch");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:branch-local-helper"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:branch-values-helper> <urn:status> "selected" }
           WHERE  {
               { BIND (<urn:opaque-helper>("unused") AS ?value) }
               UNION
               { VALUES (?value ?unused) { (<urn:branch-value> UNDEF) } }
           }"#,
    )
    .expect("a VALUES binding in one UNION branch does not taint a sibling helper");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:branch-values-helper"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:exists-branch-local-helper> <urn:status> "selected" }
           WHERE  {
               FILTER EXISTS {
                   { BIND (<urn:opaque-helper>("unused") AS ?local) }
                   UNION
                   {
                       BIND (IF(true, <urn:exists-iri>, "urn:exists-iri") AS ?local)
                       FILTER (isIRI(?local))
                   }
               }
           }"#,
    )
    .expect("EXISTS keeps selector requirements local to each UNION alternative");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:exists-branch-local-helper"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:order-without-slice> <urn:status> "selected" }
           WHERE  {
               { SELECT ?keep ?sort WHERE {
                       BIND ("keep" AS ?keep)
                       BIND (<urn:opaque-helper>("unused") AS ?sort)
                   }
                   ORDER BY (isIRI(?sort)) }
           }"#,
    )
    .expect("ORDER BY without LIMIT or OFFSET cannot alter the mutation solution multiset");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:order-without-slice"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:undef-does-not-constrain> <urn:status> "selected" }
           WHERE  {
               BIND (<urn:opaque-helper>("unused") AS ?value)
               VALUES ?value { UNDEF }
           }"#,
    )
    .expect("an all-UNDEF VALUES column imposes no RDF compatibility constraint");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:undef-does-not-constrain"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    db.execute_sparql(
        r#"INSERT { <urn:scoped-filter-alias> <urn:status> "selected" }
           WHERE  {
               BIND (<urn:opaque-helper>("unused") AS ?selector)
               { SELECT ?keep WHERE {
                   BIND (IF(true, <urn:inner-iri>, "urn:inner-iri") AS ?selector)
                   BIND ("keep" AS ?keep)
                   FILTER (isIRI(?selector))
               } }
           }"#,
    )
    .expect("a subselect selector is exact without tainting a hidden outer alias");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:scoped-filter-alias"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    let error = db
        .execute_sparql(
            r#"INSERT { <urn:opaque-local-filter> <urn:status> "bad" }
               WHERE  {
                   { SELECT ?keep WHERE {
                       BIND (<urn:opaque-helper>("x") AS ?local)
                       BIND ("keep" AS ?keep)
                       FILTER (isIRI(?local))
                   } }
               }"#,
        )
        .expect_err("a subselect-local term selector must fail closed on opaque identity");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected local-selector error: {error}"
    );

    db.execute_sparql(
        r#"INSERT { ?outer <urn:status> "selected" }
           WHERE  {
               BIND (<urn:outer-template-term> AS ?outer)
               { SELECT ?keep WHERE {
                   BIND (<urn:opaque-helper>("hidden") AS ?outer)
                   BIND ("keep" AS ?keep)
               } }
           }"#,
    )
    .expect("a hidden subselect alias does not inherit an outer exact requirement");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:outer-template-term"),
        Term::iri("urn:status"),
        Term::literal("selected"),
    )));

    let error = db
        .execute_sparql(
            r#"INSERT { <urn:opaque-values-selector> <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>("x") AS ?value)
                   VALUES (?value ?unused) { (<urn:x> UNDEF) }
               }"#,
        )
        .expect_err("a VALUES equality cannot silently consume opaque RDF identity");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected VALUES-selector error: {error}"
    );

    for update in [
        r#"INSERT { <urn:opaque-values-first-join> <urn:status> "bad" }
           WHERE  {
               VALUES ?value { <urn:x> }
               { SELECT (<urn:opaque-helper>("x") AS ?value) WHERE { } }
           }"#,
        r#"INSERT { <urn:opaque-pattern-join> <urn:status> "bad" }
           WHERE  {
               { SELECT (<urn:opaque-helper>("x") AS ?value) WHERE { } }
               ?subject <urn:join-source> ?value
           }"#,
    ] {
        let error = db
            .execute_sparql(update)
            .expect_err("shared graph-pattern variables join on exact RDF identity");
        assert!(
            error
                .to_string()
                .contains("cannot preserve RDF term identity"),
            "unexpected exact-join error: {error}"
        );
    }

    let error = db
        .execute_sparql(
            r#"INSERT { <urn:opaque-control-alias> <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>("x") AS ?source)
                   BIND (isIRI(?source) AS ?keep)
                   FILTER (?keep)
               }"#,
        )
        .expect_err("a selector alias must trace its term-sensitive input");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected control-alias error: {error}"
    );

    let error = db
        .execute_sparql(
            r#"INSERT { ?output <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>("x") AS ?source)
                   BIND (isIRI(?source) AS ?choose)
                   BIND (IF(?choose, <urn:i>, "lit") AS ?output)
               }"#,
        )
        .expect_err("an exact IF template output traces its condition as control data");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected template IF control error: {error}"
    );

    let error = db
        .execute_sparql(
            r#"INSERT { <urn:opaque-correlated-not-exists> <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>("x") AS ?selector)
                   FILTER NOT EXISTS { FILTER (isIRI(?selector)) }
               }"#,
        )
        .expect_err("a correlated NOT EXISTS selector traces outer RDF identity");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected correlated NOT EXISTS error: {error}"
    );

    let error = db
        .execute_sparql(
            r#"INSERT { <urn:opaque-correlated-union> <urn:status> "bad" }
               WHERE  {
                   BIND (<urn:opaque-helper>("x") AS ?source)
                   FILTER EXISTS {
                       { BIND (?source AS ?local) }
                       UNION
                       { BIND (<urn:i> AS ?local) }
                       FILTER (isIRI(?local))
                   }
               }"#,
        )
        .expect_err("EXISTS requirements flow into each UNION child without crossing siblings");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected correlated UNION error: {error}"
    );

    let error = db
        .execute_sparql(
            r#"INSERT { <urn:opaque-group-key> <urn:status> "bad" }
               WHERE  {
                   { SELECT ?key (COUNT(*) AS ?count) WHERE {
                           BIND (<urn:opaque-helper>("x") AS ?key)
                       }
                       GROUP BY ?key }
               }"#,
        )
        .expect_err("GROUP BY variables use exact RDF identity");
    assert!(
        error
            .to_string()
            .contains("cannot preserve RDF term identity"),
        "unexpected GROUP BY identity error: {error}"
    );

    for update in [
        r#"INSERT { <urn:opaque-filter-alias> <urn:status> "bad" }
           WHERE  {
               BIND (<urn:opaque-helper>("x") AS ?selector)
               FILTER (isIRI(?selector))
           }"#,
        r#"INSERT { <urn:opaque-subselect-alias> <urn:status> "bad" }
           WHERE  {
               { SELECT (<urn:opaque-helper>("x") AS ?selector) WHERE { } }
               FILTER (isIRI(?selector))
           }"#,
        r#"INSERT { <urn:opaque-order-alias> <urn:status> "bad" }
           WHERE  {
               { SELECT (<urn:opaque-helper>("x") AS ?selector)
                   WHERE { }
                   ORDER BY (isIRI(?selector))
                   LIMIT 1 }
           }"#,
    ] {
        let error = db
            .execute_sparql(update)
            .expect_err("opaque RDF kinds must not control mutation selection");
        assert!(
            error
                .to_string()
                .contains("cannot preserve RDF term identity"),
            "unexpected opaque-selector error: {error}"
        );
    }

    for (expression, expected) in [
        ("isIRI(<urn:a>, <urn:b>)", "exactly 1 argument"),
        ("sameTerm(<urn:a>)", "exactly 2 arguments"),
    ] {
        let error = db
            .execute_sparql(&format!(
                r#"INSERT {{ <urn:wrong-arity> <urn:status> "bad" }}
                   WHERE  {{ FILTER ({expression}) }}"#
            ))
            .expect_err("term-sensitive filter arity is validated");
        assert!(
            error.to_string().contains(expected),
            "unexpected arity error: {error}"
        );
    }
}

#[test]
fn dynamic_graph_bindings_are_tagged_once_and_keep_runtime_term_kind() {
    const CONDITIONAL_GRAPH: &str = "http://example.org/conditional-graph";
    const VALUES_GRAPH: &str = "http://example.org/values-after-undef";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{ <http://example.org/if-subject>
                                      <http://example.org/value> ?graph }} }}
           WHERE  {{ BIND (IF(true, <{CONDITIONAL_GRAPH}>, "not-a-graph") AS ?graph) }}"#
    ))
    .expect("IF selects and preserves its runtime RDF term kind");
    assert_eq!(
        db.rdf_store()
            .graph(CONDITIONAL_GRAPH)
            .expect("conditional IRI graph")
            .len(),
        1
    );

    db.execute_sparql(
        r#"INSERT { GRAPH ?graph {
                       <http://example.org/coalesce-subject> <http://example.org/value> "v"
                   } }
           WHERE  { BIND (COALESCE(IRI(?unbound), "fallback-literal") AS ?graph) }"#,
    )
    .expect("a literal COALESCE result omits the GRAPH template without retargeting");
    assert!(db.rdf_store().graph("fallback-literal").is_none());

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{
                       <http://example.org/values-subject> <http://example.org/value> ?graph
                   }} }}
           WHERE  {{ VALUES ?graph {{ UNDEF <{VALUES_GRAPH}> }} }}"#
    ))
    .expect("UNDEF and bound VALUES rows have aligned sealed schemas");
    assert_eq!(
        db.rdf_store()
            .graph(VALUES_GRAPH)
            .expect("bound VALUES graph")
            .len(),
        1
    );

    db.execute_sparql(
        r#"INSERT { GRAPH ?graph {
                       <http://example.org/uuid-subject> <http://example.org/value> ?graph
                   } }
           WHERE  { BIND (UUID() AS ?graph) }"#,
    )
    .expect("volatile UUID is evaluated once for visible and exact term views");
    let uuid_pair = db
        .execute_sparql(
            r#"SELECT ?graph ?object WHERE {
                GRAPH ?graph { <http://example.org/uuid-subject>
                                   <http://example.org/value> ?object }
            }"#,
        )
        .unwrap();
    assert_eq!(uuid_pair.rows().len(), 1);
    assert_eq!(uuid_pair.rows()[0][0], uuid_pair.rows()[0][1]);
}

#[test]
fn union_aligns_disjoint_visible_and_exact_binding_schemas_by_name() {
    const BOUND_GRAPH: &str = "http://example.org/union-bound";
    const OTHER_GRAPH: &str = "http://example.org/union-other";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{
                       <http://example.org/union-subject> <http://example.org/value> ?graph
                   }} }}
           WHERE  {{
               {{ VALUES ?graph {{ <{BOUND_GRAPH}> }} }}
               UNION
               {{ VALUES ?other {{ <{OTHER_GRAPH}> }} }}
           }}"#
    ))
    .expect("UNION null-fills disjoint variable and exact-companion columns");

    assert_eq!(
        db.rdf_store()
            .graph(BOUND_GRAPH)
            .expect("bound UNION graph")
            .len(),
        1
    );
    assert!(
        db.rdf_store().graph(OTHER_GRAPH).is_none(),
        "a positional column collision must not forge a ?graph binding"
    );
}

#[test]
fn aggregate_aliases_preserve_exact_identity() {
    const SAMPLE_GRAPH: &str = "http://example.org/sample-graph";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(&format!(
        r#"INSERT {{ GRAPH ?graph {{
                       <http://example.org/sample-subject> <http://example.org/value> ?graph
                   }} }}
           WHERE  {{ {{ SELECT (SAMPLE(?source) AS ?graph)
                        WHERE {{ VALUES ?source {{ <{SAMPLE_GRAPH}> }} }} }} }}"#
    ))
    .expect("SAMPLE carries the selected visible term and exact identity in lockstep");
    assert_eq!(
        db.rdf_store()
            .graph(SAMPLE_GRAPH)
            .expect("SAMPLE-selected graph")
            .len(),
        1
    );

    db.execute_sparql(
        r#"INSERT { GRAPH ?graph {
                       <http://example.org/count-subject> <http://example.org/value> "v"
                   } }
           WHERE  { { SELECT (COUNT(*) AS ?graph) WHERE { VALUES ?v { 1 } } } }"#,
    )
    .expect("numeric aggregate identity is an RDF literal and omits GRAPH");
    assert!(db.rdf_store().graph("1").is_none());

    db.execute_sparql(
        r#"INSERT { <http://example.org/unused-min> <http://example.org/value> "ok" }
           WHERE  { { SELECT (MIN(?source) AS ?unused)
                        WHERE { VALUES ?source { "b" "a" } } } }"#,
    )
    .expect("an aggregate that cannot carry term provenance remains usable outside the template");

    db.execute_sparql(
        r#"INSERT { GRAPH ?graph {
                       <http://example.org/min-subject> <http://example.org/value> ?graph
                   } }
           WHERE  { { SELECT (MIN(?source) AS ?graph)
                        WHERE { VALUES ?source {
                            <http://example.org/z-graph> <http://example.org/a-graph>
                        } } } }"#,
    )
    .expect("MIN carries the selected RDF term and exact identity together");
    assert_eq!(
        db.rdf_store()
            .graph("http://example.org/a-graph")
            .expect("MIN-selected graph")
            .len(),
        1
    );

    db.execute_sparql(
        r#"INSERT {
               <http://example.org/mixed-min> <http://example.org/value> ?minimum
           }
           WHERE  { { SELECT (MIN(?source) AS ?minimum)
                        WHERE { VALUES ?source { <http://example.org/z-iri> "a-literal" } } } }"#,
    )
    .expect("MIN orders and returns one sealed RDF term rather than split columns");
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("http://example.org/mixed-min"),
        Term::iri("http://example.org/value"),
        Term::iri("http://example.org/z-iri"),
    )));

    db.execute_sparql(
        r#"INSERT { GRAPH ?graph {
                       <http://example.org/chained-min> <http://example.org/value> ?graph
                   } }
           WHERE  {
               { SELECT (MIN(?source) AS ?minimum)
                   WHERE { VALUES ?source {
                       <http://example.org/z-chain> <http://example.org/a-chain>
                   } } }
               BIND (?minimum AS ?graph)
           }"#,
    )
    .expect("aggregate identity survives a subsequent alias binding");
    assert_eq!(
        db.rdf_store()
            .graph("http://example.org/a-chain")
            .expect("chained MIN-selected graph")
            .len(),
        1
    );
}

#[test]
fn mutation_strlang_and_strdt_preserve_exact_rdf_terms() {
    const PREDICATE: &str = "http://example.org/value";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(
        r#"INSERT {
               <http://example.org/lang-subject> <http://example.org/value> ?lang .
               <http://example.org/typed-subject> <http://example.org/value> ?typed .
           }
           WHERE {
               BIND (STRLANG("colour", "en-GB") AS ?lang)
               BIND (STRDT("01", <http://www.w3.org/2001/XMLSchema#integer>) AS ?typed)
           }"#,
    )
    .expect("literal constructors retain language, datatype, and lexical form");

    let store = db.rdf_store();
    assert!(store.contains(&Triple::new(
        Term::iri("http://example.org/lang-subject"),
        Term::iri(PREDICATE),
        Term::lang_literal("colour", "en-GB"),
    )));
    assert!(store.contains(&Triple::new(
        Term::iri("http://example.org/typed-subject"),
        Term::iri(PREDICATE),
        Term::typed_literal("01", "http://www.w3.org/2001/XMLSchema#integer"),
    )));

    db.execute_sparql(
        r#"INSERT { <http://example.org/control-subject> <http://example.org/value> ?value }
           WHERE  { BIND ("back\bform\funit\u0001" AS ?value) }"#,
    )
    .expect("all N-Triples control escapes survive sealed mutation bindings");
    assert!(store.contains(&Triple::new(
        Term::iri("http://example.org/control-subject"),
        Term::iri(PREDICATE),
        Term::literal("back\u{0008}form\u{000C}unit\u{0001}"),
    )));

    db.execute_sparql(
        r#"INSERT {
               <http://example.org/volatile-typed> <http://example.org/value> ?typed .
               <http://example.org/volatile-visible> <http://example.org/value> ?visible .
           }
           WHERE {
               BIND (STRDT(STRUUID(), <urn:volatile-datatype>) AS ?typed)
               BIND (STR(?typed) AS ?visible)
           }"#,
    )
    .expect("volatile typed-literal input is evaluated once");
    let object_for = |subject: &str| {
        store
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: Some(Term::iri(subject)),
                predicate: Some(Term::iri(PREDICATE)),
                object: None,
            })
            .first()
            .expect("inserted volatile literal")
            .object()
            .clone()
    };
    let Term::Literal(typed) = object_for("http://example.org/volatile-typed") else {
        panic!("STRDT must store a literal")
    };
    let Term::Literal(visible) = object_for("http://example.org/volatile-visible") else {
        panic!("STR must store a literal")
    };
    assert_eq!(typed.datatype(), "urn:volatile-datatype");
    assert_eq!(typed.value(), visible.value());

    for invalid in [
        r#"STRDT("x", <urn:datatype>, "extra")"#,
        r#"STRLANG("x", "en", "extra")"#,
    ] {
        let error = db
            .execute_sparql(&format!(
                r#"INSERT {{ <http://example.org/invalid-arity> <{PREDICATE}> ?value }}
                   WHERE  {{ BIND ({invalid} AS ?value) }}"#
            ))
            .expect_err("literal constructors with the wrong arity must be rejected");
        assert!(
            error.to_string().contains("requires exactly two arguments"),
            "unexpected wrong-arity error: {error}"
        );
    }
}

#[test]
fn grouped_mutation_uses_canonical_identity_and_lossless_representatives() {
    const PREDICATE: &str = "http://example.org/grouped-value";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    for (subject, values) in [
        (
            "http://example.org/grouped-language",
            r#""colour"@EN "colour"@en"#,
        ),
        (
            "http://example.org/grouped-string",
            r#""x" "x"^^<http://www.w3.org/2001/XMLSchema#string>"#,
        ),
    ] {
        db.execute_sparql(&format!(
            r#"INSERT {{ <{subject}> <{PREDICATE}> ?value }}
               WHERE  {{
                   {{ SELECT ?value (COUNT(*) AS ?count)
                       WHERE {{ VALUES ?value {{ {values} }} }}
                       GROUP BY ?value
                       HAVING (COUNT(*) = 2) }}
               }}"#,
        ))
        .unwrap_or_else(|error| {
            panic!("canonical GROUP BY must retain a lossless mutation representative: {error}")
        });
    }

    let language = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: Some(Term::iri("http://example.org/grouped-language")),
            predicate: Some(Term::iri(PREDICATE)),
            object: None,
        });
    assert_eq!(language.len(), 1, "language spellings form one RDF group");
    assert_eq!(
        language[0].object(),
        &Term::lang_literal("colour", "EN"),
        "the group carries the deterministic minimum raw N-Triples representative"
    );

    let string = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: Some(Term::iri("http://example.org/grouped-string")),
            predicate: Some(Term::iri(PREDICATE)),
            object: None,
        });
    assert_eq!(
        string.len(),
        1,
        "plain and explicit xsd:string form one RDF group"
    );
    assert_eq!(
        string[0].object(),
        &Term::literal("x"),
        "the representative remains a valid lossless RDF term"
    );
}

#[test]
fn invalid_template_term_positions_are_omitted_without_panicking() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    for update in [
        r#"INSERT { ?subject <http://example.org/p> "object" }
            WHERE { BIND ("http://example.org/literal-subject" AS ?subject) }"#,
        r#"INSERT { <http://example.org/subject> ?predicate "object" }
            WHERE { BIND ("http://example.org/literal-predicate" AS ?predicate) }"#,
        r#"INSERT { <http://example.org/subject> ?predicate "object" }
            WHERE { BIND (BNODE("blank-predicate") AS ?predicate) }"#,
        r#"INSERT { ?subject <http://example.org/p> "object" }
            WHERE { { SELECT (STR(<http://example.org/projected-literal>) AS ?subject)
                      WHERE { } } }"#,
    ] {
        db.execute_sparql(update)
            .expect("invalid instantiated triples are successful no-ops");
    }
    assert!(db.rdf_store().is_empty());

    db.execute_sparql(
        r#"INSERT { ?subject <http://example.org/p> "object" }
           WHERE  { BIND (BNODE("valid-blank-subject") AS ?subject) }"#,
    )
    .expect("blank nodes remain valid subjects");
    assert_eq!(db.rdf_store().len(), 1);
}

#[test]
fn property_path_bindings_keep_exact_term_identity_in_mutations() {
    const P: &str = "http://example.org/path-edge";
    const REACHES: &str = "http://example.org/reaches";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
               <http://example.org/a> <{P}> <http://example.org/b> .
               <http://example.org/b> <{P}> <http://example.org/c> .
           }}"#
    ))
    .unwrap();

    db.execute_sparql(&format!(
        r#"INSERT {{ ?start <{REACHES}> ?end }}
           WHERE  {{ ?start <{P}>+ ?end }}"#
    ))
    .expect("property-path endpoints retain exact IRI identity in update templates");

    for (start, end) in [
        ("http://example.org/a", "http://example.org/b"),
        ("http://example.org/a", "http://example.org/c"),
        ("http://example.org/b", "http://example.org/c"),
    ] {
        assert!(db.rdf_store().contains(&Triple::new(
            Term::iri(start),
            Term::iri(REACHES),
            Term::iri(end),
        )));
    }
}

#[test]
fn insert_template_blank_nodes_are_fresh_per_solution_and_shared_per_label() {
    use std::collections::HashSet;

    const SOURCE: &str = "http://example.org/source-value";
    const LINK: &str = "http://example.org/link";
    const KIND: &str = "http://example.org/kind";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
               <http://example.org/source-a> <{SOURCE}> "a" .
               <http://example.org/source-b> <{SOURCE}> "b" .
           }}"#
    ))
    .unwrap();

    let update = format!(
        r#"INSERT {{
               _:result <{LINK}> ?value .
               _:result <{KIND}> <http://example.org/Result> .
           }}
           WHERE {{ ?source <{SOURCE}> ?value }}"#
    );
    db.execute_sparql(&update)
        .expect("INSERT template blank nodes are constructed per solution");
    db.execute_sparql(&update)
        .expect("a repeated statement constructs fresh blank nodes again");

    let subjects_for = |predicate: &str| {
        db.rdf_store()
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri(predicate)),
                object: None,
            })
            .into_iter()
            .map(|triple| triple.subject().clone())
            .collect::<HashSet<_>>()
    };
    let link_subjects = subjects_for(LINK);
    let kind_subjects = subjects_for(KIND);
    assert_eq!(
        link_subjects.len(),
        4,
        "one fresh blank node per solution and execution"
    );
    assert_eq!(
        link_subjects, kind_subjects,
        "one label is shared per solution"
    );
    assert!(link_subjects.iter().all(Term::is_blank_node));
}

#[test]
fn relative_graph_iri_starting_with_question_mark_is_not_a_variable() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(
        r#"INSERT { GRAPH <?constant> {
                       <http://example.org/relative-subject> <http://example.org/value> "v"
                   } }
           WHERE { }"#,
    )
    .expect("accepted relative GRAPH IRI is not decoded as a variable");

    assert_eq!(
        db.rdf_store()
            .graph("?constant")
            .expect("relative constant graph")
            .len(),
        1
    );
}

#[test]
fn modify_with_and_using_define_the_where_dataset() {
    const TARGET: &str = "http://example.org/with-target";
    const SOURCE: &str = "http://example.org/using-source";
    const P: &str = "http://example.org/p";
    const COPIED: &str = "http://example.org/copied";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(&format!(
        r#"INSERT DATA {{
               <http://example.org/default-subject> <{P}> "default" .
               GRAPH <{TARGET}> {{ <http://example.org/target-subject> <{P}> "target" }}
               GRAPH <{SOURCE}> {{ <http://example.org/source-subject> <{P}> "source" }}
           }}"#
    ))
    .unwrap();

    db.execute_sparql(&format!(
        r#"WITH <{TARGET}>
           DELETE {{ ?s <{P}> ?o }}
           WHERE  {{ ?s <{P}> ?o }}"#
    ))
    .expect("WITH makes its graph the default WHERE dataset");
    assert!(db.rdf_store().graph(TARGET).unwrap().is_empty());
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("http://example.org/default-subject"),
        Term::iri(P),
        Term::literal("default"),
    )));

    db.execute_sparql(&format!(
        r#"INSERT DATA {{ GRAPH <{TARGET}> {{
               <http://example.org/nested-target> <{P}> "nested"
           }} }}"#
    ))
    .unwrap();
    db.execute_sparql(&format!(
        r#"WITH <{TARGET}>
           DELETE {{ ?s <{P}> ?o }}
           WHERE  {{ {{ SELECT ?s ?o WHERE {{ ?s <{P}> ?o }} }} }}"#
    ))
    .expect("a subselect inherits the WITH dataset");
    assert!(db.rdf_store().graph(TARGET).unwrap().is_empty());

    db.execute_sparql(&format!(
        r#"WITH <{TARGET}>
           INSERT {{ ?s <{COPIED}> ?o }}
           USING <{SOURCE}>
           WHERE  {{ {{ SELECT ?s ?o WHERE {{ ?s <{P}> ?o }} }} }}"#
    ))
    .expect("a subselect inherits USING while WITH still targets the template");
    let target = db.rdf_store().graph(TARGET).expect("WITH target graph");
    assert!(target.contains(&Triple::new(
        Term::iri("http://example.org/source-subject"),
        Term::iri(COPIED),
        Term::literal("source"),
    )));
    assert!(!target.contains(&Triple::new(
        Term::iri("http://example.org/default-subject"),
        Term::iri(COPIED),
        Term::literal("default"),
    )));

    db.execute_sparql(&format!(
        r#"WITH <{TARGET}>
           INSERT {{ ?s <http://example.org/not-named> ?graph }}
           USING <{SOURCE}>
           WHERE  {{ GRAPH ?graph {{ ?s <{P}> ?o }} }}"#
    ))
    .expect("USING without USING NAMED exposes no named graphs");
    assert!(
        target
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri("http://example.org/not-named")),
                object: None,
            })
            .is_empty()
    );
}

#[test]
fn using_datasets_have_rdf_set_semantics() {
    const FIRST: &str = "http://example.org/set-source-first";
    const SECOND: &str = "http://example.org/set-source-second";
    const SOURCE_PREDICATE: &str = "http://example.org/set-source-predicate";
    const DEFAULT_RESULT: &str = "http://example.org/default-set-result";
    const NAMED_RESULT: &str = "http://example.org/named-set-result";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");

    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            GRAPH <{FIRST}> {{ <http://example.org/shared> <{SOURCE_PREDICATE}> "same" }}
            GRAPH <{SECOND}> {{ <http://example.org/shared> <{SOURCE_PREDICATE}> "same" }}
        }}"#
    ))
    .unwrap();

    db.execute_sparql(&format!(
        r#"INSERT {{ _:result <{DEFAULT_RESULT}> ?value }}
           USING <{FIRST}>
           USING <{SECOND}>
           WHERE {{ <http://example.org/shared> <{SOURCE_PREDICATE}> ?value }}"#
    ))
    .expect("overlapping USING graphs form one set-valued default graph");
    assert_eq!(
        db.rdf_store()
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri(DEFAULT_RESULT)),
                object: None,
            })
            .len(),
        1,
        "an identical triple in two USING graphs is one WHERE solution"
    );

    db.execute_sparql(&format!(
        r#"INSERT {{ _:result <{NAMED_RESULT}> ?graph }}
           USING NAMED <{FIRST}>
           USING NAMED <{FIRST}>
           WHERE {{ GRAPH ?graph {{ <http://example.org/shared> <{SOURCE_PREDICATE}> ?value }} }}"#
    ))
    .expect("duplicate USING NAMED clauses name one graph");
    assert_eq!(
        db.rdf_store()
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: None,
                predicate: Some(Term::iri(NAMED_RESULT)),
                object: None,
            })
            .len(),
        1,
        "a duplicate named-graph clause must not duplicate GRAPH solutions"
    );
}

#[test]
fn modify_insert_cdc_and_history_report_only_real_set_transitions() {
    const TARGET: &str = "http://example.org/deduplicated-target";
    const SUBJECT: &str = "http://example.org/deduplicated-subject";
    const PREDICATE: &str = "http://example.org/deduplicated-predicate";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(
        r#"INSERT DATA {
            <http://example.org/source-a> <http://example.org/matches> "one" .
            <http://example.org/source-b> <http://example.org/matches> "two" .
        }"#,
    )
    .unwrap();

    let before_insert = db.current_epoch();
    let update = format!(
        r#"INSERT {{ GRAPH <{TARGET}> {{ <{SUBJECT}> <{PREDICATE}> "value" }} }}
           WHERE  {{ ?source <http://example.org/matches> ?match }}"#
    );
    db.execute_sparql(&update).unwrap();
    let insert_epoch = db.current_epoch();

    assert_eq!(db.rdf_store().graph(TARGET).unwrap().len(), 1);
    let first_events: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| event.epoch == insert_epoch)
        .filter(|event| event.triple_graph.as_deref() == Some(TARGET))
        .collect();
    assert_eq!(first_events.len(), 1);
    assert_eq!(first_events[0].kind, ChangeKind::Create);
    let first_diff = db
        .rdf_history_diff(before_insert, insert_epoch)
        .expect("deduplicated insert history");
    assert_eq!(
        first_diff
            .transitions
            .iter()
            .filter(|transition| matches!(
                transition.kind,
                RdfHistoryTransitionKind::StatementAsserted { .. }
            ))
            .count(),
        1
    );

    db.execute_sparql(&update).unwrap();
    let noop_epoch = db.current_epoch();
    assert_eq!(db.rdf_store().graph(TARGET).unwrap().len(), 1);
    assert!(
        all_changes(&db)
            .into_iter()
            .filter(|event| event.epoch == noop_epoch)
            .find(|event| event.triple_graph.as_deref() == Some(TARGET))
            .is_none(),
        "an already-present quad must not emit volatile create CDC"
    );
    assert!(
        db.rdf_history_diff(insert_epoch, noop_epoch)
            .expect("no-op insert history")
            .transitions
            .is_empty()
    );
}

#[test]
fn delete_where_materializes_all_templates_from_one_preimage() {
    const SUBJECT: &str = "http://example.org/preimage-subject";
    const FIRST: &str = "http://example.org/preimage-first";
    const SECOND: &str = "http://example.org/preimage-second";

    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_cdc(),
    )
    .expect("RDF CDC database");
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <{SUBJECT}> <{FIRST}> "one" .
            <{SUBJECT}> <{SECOND}> "two" .
        }}"#
    ))
    .unwrap();

    let before_delete = db.current_epoch();
    db.execute_sparql(&format!(
        r#"DELETE WHERE {{
            <{SUBJECT}> <{FIRST}> ?first .
            <{SUBJECT}> <{SECOND}> ?second .
        }}"#
    ))
    .unwrap();
    let delete_epoch = db.current_epoch();

    assert!(db.rdf_store().is_empty());
    let deletes: Vec<_> = all_changes(&db)
        .into_iter()
        .filter(|event| event.epoch == delete_epoch && event.kind == ChangeKind::Delete)
        .collect();
    assert_eq!(deletes.len(), 2);
    let durable = db
        .rdf_history_diff(before_delete, delete_epoch)
        .expect("multi-template DELETE WHERE history");
    assert_eq!(durable.transitions.len(), 2);
    assert!(durable.transitions.iter().all(|transition| matches!(
        transition.kind,
        RdfHistoryTransitionKind::StatementRetracted { .. }
    )));
}

#[test]
fn canonical_aliases_emit_one_lossless_event_per_real_transition() {
    for model in [GraphModel::Rdf, GraphModel::Both] {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(model).with_cdc()).unwrap();
        for graph in [None, Some("urn:canonical")] {
            let wrap = |body: &str| {
                graph.map_or_else(|| body.to_string(), |g| format!("GRAPH <{g}> {{ {body} }}"))
            };
            let before = all_changes(&db).len();
            db.execute_sparql(&format!(
                "INSERT DATA {{ {} }}",
                wrap(r#"<urn:s> <urn:p> "hello"@EN . <urn:s> <urn:p> "hello"@en ."#)
            ))
            .unwrap();
            let created = all_changes(&db);
            assert_eq!(created.len(), before + 1);
            assert_eq!(created[before].kind, ChangeKind::Create);
            assert_eq!(
                created[before].triple_object.as_deref(),
                Some(r#""hello"@EN"#)
            );
            assert_eq!(created[before].triple_graph.as_deref(), graph);
            db.execute_sparql(&format!(
                "INSERT {{ {} }} WHERE {{ VALUES ?unused {{ 1 2 }} }}",
                wrap(r#"<urn:s> <urn:p> "hello"@en"#)
            ))
            .unwrap();
            assert_eq!(
                all_changes(&db).len(),
                before + 1,
                "equivalent template inserts are set no-ops"
            );
            db.execute_sparql(&format!(
                "DELETE {{ {} }} WHERE {{ VALUES ?unused {{ 1 2 }} }}",
                wrap(r#"<urn:s> <urn:p> "hello"@en"#)
            ))
            .unwrap();
            let deleted = all_changes(&db);
            assert_eq!(deleted.len(), before + 2);
            assert_eq!(deleted[before + 1].kind, ChangeKind::Delete);
            assert_eq!(
                deleted[before + 1].triple_object.as_deref(),
                Some(r#""hello"@EN"#)
            );
            db.execute_sparql(&format!(
                "DELETE DATA {{ {} }}",
                wrap(r#"<urn:s> <urn:p> "hello"@en"#)
            ))
            .unwrap();
            assert_eq!(
                all_changes(&db).len(),
                before + 2,
                "absent aliases emit no delete"
            );
            db.execute_sparql(&format!(
                "INSERT DATA {{ {} }}",
                wrap(r#"<urn:s> <urn:p> "hello"@en"#)
            ))
            .unwrap();
            let reinserted = all_changes(&db);
            assert_eq!(reinserted.len(), before + 3);
            assert_eq!(
                reinserted[before + 2].triple_object.as_deref(),
                Some(r#""hello"@en"#)
            );
        }
    }
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
