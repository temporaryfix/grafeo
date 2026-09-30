//! Complete, owned native change-event JSON for bounded binding pages.

use grafeo_engine::cdc::{ChangeEvent, ChangeKind};
use serde_json::{Value, json};

/// Converts one admitted event, preserving coordinates as decimal strings.
///
/// Callers apply native row/byte limits and authorization before conversion.
/// Creation labels and edge endpoints are part of the event, not reconstructed
/// from a possibly newer graph state.
#[must_use]
pub fn change_event_to_json(event: &ChangeEvent) -> Value {
    let entity_type = if event.entity_id.is_node() {
        "node"
    } else if event.entity_id.is_triple() {
        "triple"
    } else {
        "edge"
    };
    let kind = match event.kind {
        ChangeKind::Create => "create",
        ChangeKind::Update => "update",
        ChangeKind::Delete => "delete",
        _ => "unknown",
    };
    let properties =
        |properties: &Option<std::collections::HashMap<String, grafeo_common::types::Value>>| {
            properties.as_ref().map(|properties| {
                properties
                    .iter()
                    .map(|(key, value)| (key.clone(), crate::json::value_to_json(value)))
                    .collect::<serde_json::Map<String, Value>>()
            })
        };
    json!({
        "entity_id": event.entity_id.as_u64().to_string(),
        "entity_type": entity_type,
        "kind": kind,
        "epoch": event.epoch.as_u64().to_string(),
        "timestamp": event.timestamp.as_u64().to_string(),
        "graph_incarnation": event.graph_incarnation.map(|id| id.as_u64().to_string()),
        "before": properties(&event.before),
        "after": properties(&event.after),
        "labels": event.labels,
        "edge_type": event.edge_type,
        "src_id": event.src_id.map(|id| id.to_string()),
        "dst_id": event.dst_id.map(|id| id.to_string()),
        "lpg_graph": event.graph_path().map(grafeo_common::types::GraphPath::components),
        "triple_graph": event.triple_graph,
        "triple_subject": event.triple_subject,
        "triple_predicate": event.triple_predicate,
        "triple_object": event.triple_object,
    })
}
