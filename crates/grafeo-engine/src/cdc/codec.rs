//! Canonical native event images shared by WAL and retained checkpoints.
#[cfg(feature = "lpg")]
use super::cdc_capacity_error;
use super::{ChangeEvent, ChangeKind, EntityId};
use grafeo_common::types::{EpochId, GraphIncarnationId, GraphPath, HlcTimestamp, NodeId, Value};
use grafeo_common::utils::error::{Error, Result, StorageError};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{decode_value_exact, encode_value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type Properties = Option<Vec<(String, Vec<u8>)>>;

#[derive(Serialize, Deserialize)]
pub(super) struct Event {
    pub(super) timestamp: u64,
    pub(super) incarnation: u64,
    pub(super) kind: ChangeKind,
    pub(super) payload: Payload,
}

#[derive(Serialize, Deserialize)]
pub(super) enum Payload {
    Lpg {
        #[serde(with = "grafeo_common::types::graph_path_bytes")]
        graph: GraphPath,
        entity: EntityId,
        before: Properties,
        after: Properties,
        labels: Option<Vec<String>>,
        edge_type: Option<String>,
        src: Option<u64>,
        dst: Option<u64>,
    },
    Rdf {
        graph: Option<String>,
        subject: String,
        predicate: String,
        object: String,
    },
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Storage(StorageError::InvalidWalEntry(format!(
        "CDC batch: {message}"
    )))
}

#[cfg(feature = "lpg")]
fn encode_properties(properties: &Option<HashMap<String, Value>>) -> Result<Properties> {
    properties
        .as_ref()
        .map(|properties| {
            let mut encoded = Vec::new();
            encoded
                .try_reserve(properties.len())
                .map_err(|_| cdc_capacity_error())?;
            for (key, value) in properties {
                encoded.push((key.clone(), encode_value(value)?));
            }
            encoded.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            Ok(encoded)
        })
        .transpose()
}

#[cfg(feature = "lpg")]
fn decode_properties(properties: Properties) -> Result<Option<HashMap<String, Value>>> {
    properties
        .map(|properties| {
            if properties.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                return Err(invalid("unordered or duplicate property keys"));
            }
            let mut decoded = HashMap::new();
            decoded
                .try_reserve(properties.len())
                .map_err(|_| cdc_capacity_error())?;
            for (key, bytes) in properties {
                decoded.insert(key, decode_value_exact(&bytes)?);
            }
            Ok(decoded)
        })
        .transpose()
}

#[cfg(not(feature = "lpg"))]
fn encode_properties(_: &Option<HashMap<String, Value>>) -> Result<Properties> {
    Err(invalid("LPG CDC requires LPG support"))
}

#[cfg(not(feature = "lpg"))]
fn decode_properties(_: Properties) -> Result<Option<HashMap<String, Value>>> {
    Err(invalid("LPG CDC requires LPG support"))
}

impl Event {
    pub(super) fn from_event(event: &ChangeEvent) -> Result<Self> {
        let incarnation = event
            .graph_incarnation
            .ok_or_else(|| invalid("event has no native graph owner"))?
            .as_u64();
        let payload = if event.entity_id.is_triple() {
            if event.lpg_graph.is_some()
                || event.before.is_some()
                || event.after.is_some()
                || event.labels.is_some()
                || event.edge_type.is_some()
                || event.src_id.is_some()
                || event.dst_id.is_some()
            {
                return Err(invalid("RDF event carries LPG fields"));
            }
            Payload::Rdf {
                graph: event.triple_graph.clone(),
                subject: event
                    .triple_subject
                    .clone()
                    .ok_or_else(|| invalid("missing RDF subject"))?,
                predicate: event
                    .triple_predicate
                    .clone()
                    .ok_or_else(|| invalid("missing RDF predicate"))?,
                object: event
                    .triple_object
                    .clone()
                    .ok_or_else(|| invalid("missing RDF object"))?,
            }
        } else {
            if event.triple_graph.is_some()
                || event.triple_subject.is_some()
                || event.triple_predicate.is_some()
                || event.triple_object.is_some()
            {
                return Err(invalid("LPG event carries RDF fields"));
            }
            Payload::Lpg {
                graph: event
                    .lpg_graph
                    .clone()
                    .ok_or_else(|| invalid("missing LPG path"))?,
                entity: event.entity_id,
                before: encode_properties(&event.before)?,
                after: encode_properties(&event.after)?,
                labels: event.labels.clone(),
                edge_type: event.edge_type.clone(),
                src: event.src_id,
                dst: event.dst_id,
            }
        };
        let wire = Self {
            timestamp: event.timestamp.as_u64(),
            incarnation,
            kind: event.kind.clone(),
            payload,
        };
        wire.validate(if event.entity_id.is_triple() { 2 } else { 1 })?;
        Ok(wire)
    }

    pub(super) fn validate(&self, model: u8) -> Result<()> {
        if self.timestamp == 0 || self.timestamp == u64::MAX || self.incarnation == u64::MAX {
            return Err(invalid("reserved timestamp or incarnation"));
        }
        let default_graph = match &self.payload {
            Payload::Lpg {
                graph,
                entity,
                labels,
                edge_type,
                src,
                dst,
                before,
                after,
            } => {
                if model != 1 || entity.is_triple() || entity.as_u64() == u64::MAX {
                    return Err(invalid("invalid LPG model or entity"));
                }
                if matches!(self.kind, ChangeKind::Create) && before.is_some()
                    || matches!(self.kind, ChangeKind::Delete) && after.is_some()
                    || labels.is_some() && !entity.is_node()
                    || (edge_type.is_some() || src.is_some() || dst.is_some())
                        && (entity.is_node() || self.kind != ChangeKind::Create)
                    || src.is_some_and(|id| id == u64::MAX)
                    || dst.is_some_and(|id| id == u64::MAX)
                {
                    return Err(invalid("inconsistent LPG change image"));
                }
                graph.components().is_empty()
            }
            Payload::Rdf {
                graph,
                subject,
                predicate,
                object,
            } => {
                if model != 2 || self.kind == ChangeKind::Update {
                    return Err(invalid("invalid RDF model or change kind"));
                }
                #[cfg(feature = "triple-store")]
                {
                    use grafeo_core::graph::rdf::Term;
                    let mut terms = Vec::new();
                    for text in [subject, predicate, object] {
                        let term = Term::from_ntriples(text)
                            .ok_or_else(|| invalid("malformed RDF term"))?;
                        if term.to_string() != *text {
                            return Err(invalid("noncanonical RDF term encoding"));
                        }
                        terms.push(term);
                    }
                    if terms[0].is_literal() || !terms[1].is_iri() {
                        return Err(invalid("invalid RDF subject or predicate"));
                    }
                }
                #[cfg(not(feature = "triple-store"))]
                {
                    let _ = (graph, subject, predicate, object);
                    return Err(invalid("RDF CDC requires triple-store support"));
                }
                #[cfg(feature = "triple-store")]
                graph.is_none()
            }
        };
        if default_graph != (self.incarnation == 0) {
            return Err(invalid("graph coordinate and native incarnation disagree"));
        }
        Ok(())
    }

    pub(super) fn into_event(self, epoch: EpochId) -> Result<ChangeEvent> {
        let mut event = ChangeEvent {
            entity_id: EntityId::Node(NodeId::new(0)),
            kind: self.kind,
            epoch,
            timestamp: HlcTimestamp::from_u64(self.timestamp),
            graph_incarnation: Some(GraphIncarnationId::new(self.incarnation)),
            before: None,
            after: None,
            labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            lpg_graph: None,
            triple_graph: None,
        };
        match self.payload {
            Payload::Lpg {
                graph,
                entity,
                before,
                after,
                labels,
                edge_type,
                src,
                dst,
            } => {
                event.entity_id = entity;
                event.lpg_graph = Some(graph);
                event.before = decode_properties(before)?;
                event.after = decode_properties(after)?;
                event.labels = labels;
                event.edge_type = edge_type;
                event.src_id = src;
                event.dst_id = dst;
            }
            Payload::Rdf {
                graph,
                subject,
                predicate,
                object,
            } => {
                event.entity_id = EntityId::Triple(super::triple_hash(
                    &subject,
                    &predicate,
                    &object,
                    graph.as_deref(),
                ));
                event.triple_graph = graph;
                event.triple_subject = Some(subject);
                event.triple_predicate = Some(predicate);
                event.triple_object = Some(object);
            }
        }
        Ok(event)
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use grafeo_common::types::EdgeId;

    #[test]
    fn node_label_images_survive_every_change_kind_but_edges_reject_labels() {
        for kind in [ChangeKind::Create, ChangeKind::Update, ChangeKind::Delete] {
            let wire = Event {
                timestamp: 1,
                incarnation: 0,
                kind,
                payload: Payload::Lpg {
                    graph: GraphPath::root(),
                    entity: EntityId::Node(NodeId::new(1)),
                    before: None,
                    after: None,
                    labels: Some(vec!["Person".to_owned()]),
                    edge_type: None,
                    src: None,
                    dst: None,
                },
            };
            wire.validate(1).unwrap();
            let mut event = wire.into_event(EpochId(1)).unwrap();
            let restored = Event::from_event(&event)
                .unwrap()
                .into_event(EpochId(1))
                .unwrap();
            assert_eq!(
                serde_json::to_value(&restored).unwrap(),
                serde_json::to_value(&event).unwrap()
            );
            event.entity_id = EntityId::Edge(EdgeId::new(1));
            assert!(Event::from_event(&event).is_err());
        }
    }
}
