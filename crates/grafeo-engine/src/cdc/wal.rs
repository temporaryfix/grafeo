//! Canonical native event batches carried inside authenticated WAL groups.

use super::codec::Event;
use super::{CdcLog, ChangeEvent, EntityId, PreparedCdcBatch, cdc_capacity_error};
#[cfg(all(test, feature = "lpg"))]
use super::{ChangeKind, codec::Payload};
use grafeo_common::types::{EpochId, HlcTimestamp, TransactionId};
#[cfg(any(test, not(feature = "lpg")))]
use grafeo_common::types::{GraphIncarnationId, GraphPath};
#[cfg(all(test, feature = "lpg"))]
use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::{Error, Result, StorageError};
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::encode_value;
use grafeo_storage::wal::WalRecord;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;

const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_EVENTS: usize = 65_536;
const MAGIC: &[u8; 5] = b"GCDC\x01";

// Deliberately independent of the public JSON shape: no skipped fields,
// hash-map iteration order, pointers or process-local identity on disk.
#[derive(Serialize, Deserialize)]
struct Batch {
    transaction: u64,
    epoch: u64,
    model: u8,
    events: Vec<Event>,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Storage(StorageError::InvalidWalEntry(format!(
        "CDC batch: {message}"
    )))
}

struct BoundedWriter(Vec<u8>);
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("CDC batch exceeds 16 MiB"));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(std::io::Error::other)?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode(batch: &Batch) -> Result<Vec<u8>> {
    let mut writer = BoundedWriter(Vec::new());
    writer.write_all(MAGIC)?;
    bincode::serde::encode_into_std_write(batch, &mut writer, bincode::config::standard())
        .map_err(invalid)?;
    Ok(writer.0)
}

impl PreparedCdcBatch<'_> {
    /// Serialize before final publication binding and before the durable marker.
    pub(crate) fn wal_records(
        &self,
        transaction_id: TransactionId,
        epoch: EpochId,
    ) -> Result<Vec<WalRecord>> {
        let mut ordered = Vec::new();
        for events in self.groups.values() {
            ordered
                .try_reserve(events.len())
                .map_err(|_| cdc_capacity_error())?;
            ordered.extend(events);
        }
        if ordered.len() > MAX_EVENTS {
            return Err(invalid("too many events in one transaction"));
        }
        ordered.sort_unstable_by_key(|event| event.timestamp);
        if ordered
            .windows(2)
            .any(|pair| pair[0].timestamp >= pair[1].timestamp)
        {
            return Err(invalid("duplicate event timestamps"));
        }
        let mut records = Vec::new();
        records.try_reserve(2).map_err(|_| cdc_capacity_error())?;
        for model in [1, 2] {
            let mut events = Vec::new();
            for event in &ordered {
                if if event.entity_id.is_triple() { 2 } else { 1 } == model {
                    if event.epoch != epoch {
                        return Err(invalid("event epoch differs from commit"));
                    }
                    events.try_reserve(1).map_err(|_| cdc_capacity_error())?;
                    events.push(Event::from_event(event)?);
                }
            }
            if !events.is_empty() {
                let payload = encode(&Batch {
                    transaction: transaction_id.as_u64(),
                    epoch: epoch.as_u64(),
                    model,
                    events,
                })?;
                records.push(WalRecord::CdcBatch {
                    transaction_id,
                    epoch,
                    model,
                    payload,
                });
            }
        }
        Ok(records)
    }
}

fn decode(
    transaction_id: TransactionId,
    epoch: EpochId,
    model: u8,
    payload: &[u8],
) -> Result<Vec<ChangeEvent>> {
    if payload.len() > MAX_BYTES || !payload.starts_with(MAGIC) {
        return Err(invalid("unsupported schema or size"));
    }
    let (batch, consumed): (Batch, usize) = bincode::serde::decode_from_slice(
        &payload[MAGIC.len()..],
        bincode::config::standard().with_limit::<MAX_BYTES>(),
    )
    .map_err(invalid)?;
    if consumed + MAGIC.len() != payload.len()
        || batch.transaction != transaction_id.as_u64()
        || batch.epoch != epoch.as_u64()
        || batch.model != model
        || batch.events.is_empty()
        || batch.events.len() > MAX_EVENTS
    {
        return Err(invalid(
            "foreign coordinate, invalid count or trailing bytes",
        ));
    }
    if batch
        .events
        .windows(2)
        .any(|pair| pair[0].timestamp >= pair[1].timestamp)
    {
        return Err(invalid("event timestamps are not strictly ordered"));
    }
    // Reject alternative encodings before installing any event or native state.
    if encode(&batch)? != payload {
        return Err(invalid("noncanonical batch encoding"));
    }
    let mut events = Vec::new();
    events
        .try_reserve(batch.events.len())
        .map_err(|_| cdc_capacity_error())?;
    for event in batch.events {
        event.validate(model)?;
        events.push(event.into_event(epoch)?);
    }
    Ok(events)
}

/// Private startup destination and the native checkpoint topology it extends.
pub(crate) struct RecoverySource<'a> {
    pub(crate) log: &'a CdcLog,
    #[cfg(feature = "lpg")]
    pub(crate) root: &'a std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
}

/// Decode the complete suffix before native recovery. This startup-only log
/// cannot escape the database constructor until every native replay succeeds.
pub(crate) fn recover(source: RecoverySource<'_>, records: &[WalRecord]) -> Result<()> {
    let log = source.log;
    let mut groups: hashbrown::HashMap<EntityId, Vec<ChangeEvent>> = hashbrown::HashMap::new();
    let mut high_timestamp = HlcTimestamp::zero();
    #[cfg(feature = "lpg")]
    let mut graphs: HashMap<_, _> =
        grafeo_core::graph::lpg::LpgStoreSection::new(std::sync::Arc::clone(source.root))
            .capture_graphs()?
            .into_iter()
            .map(|(path, graph)| (path, graph.graph_incarnation_id()))
            .collect();
    #[cfg(not(feature = "lpg"))]
    let mut graphs: HashMap<GraphPath, GraphIncarnationId> = HashMap::new();
    let mut rdf_owners = std::collections::HashSet::new();
    let mut timestamps = std::collections::HashSet::new();
    for record in records {
        match record {
            WalRecord::CreateLpgGraph {
                graph, incarnation, ..
            } => {
                graphs.insert(graph.clone(), *incarnation);
            }
            WalRecord::DropLpgGraph { graph, .. } => {
                graphs.retain(|path, _| !path.components().starts_with(graph.components()));
            }
            WalRecord::CatalogBatchV3 {
                created_graphs,
                dropped_graphs,
                created_graph_incarnations,
                ..
            } => {
                for graph in dropped_graphs {
                    graphs.retain(|path, _| !path.components().starts_with(graph.components()));
                }
                for (graph, incarnation) in created_graphs.iter().zip(created_graph_incarnations) {
                    graphs.insert(graph.clone(), *incarnation);
                }
            }
            WalRecord::InsertRdfQuadV3 {
                transaction_id,
                graph,
                graph_incarnation: incarnation,
                ..
            }
            | WalRecord::DeleteRdfQuadV3 {
                transaction_id,
                graph,
                graph_incarnation: incarnation,
                ..
            } => {
                rdf_owners
                    .try_reserve(1)
                    .map_err(|_| cdc_capacity_error())?;
                rdf_owners.insert((*transaction_id, graph.clone(), *incarnation));
            }
            WalRecord::CdcBatch {
                transaction_id,
                epoch,
                model,
                payload,
            } => {
                for event in decode(*transaction_id, *epoch, *model, payload)? {
                    if timestamps.len() == MAX_EVENTS {
                        return Err(invalid("transaction event limit exceeded"));
                    }
                    timestamps
                        .try_reserve(1)
                        .map_err(|_| cdc_capacity_error())?;
                    if !timestamps.insert(event.timestamp) {
                        return Err(invalid("duplicate timestamp across model batches"));
                    }
                    let incarnation = event
                        .graph_incarnation
                        .ok_or_else(|| invalid("missing native lifetime"))?;
                    if let Some(path) = &event.lpg_graph {
                        if graphs.get(path) != Some(&incarnation) {
                            return Err(invalid("LPG event names a foreign native lifetime"));
                        }
                    } else if !rdf_owners.contains(&(
                        *transaction_id,
                        event.triple_graph.clone(),
                        incarnation,
                    )) {
                        return Err(invalid("RDF event has no matching native mutation owner"));
                    }
                    high_timestamp = high_timestamp.max(event.timestamp);
                    if !groups.contains_key(&event.entity_id) {
                        groups.try_reserve(1).map_err(|_| cdc_capacity_error())?;
                    }
                    let target = groups.entry(event.entity_id).or_default();
                    target.try_reserve(1).map_err(|_| cdc_capacity_error())?;
                    target.push(event);
                }
            }
            WalRecord::CommittedWithCdc { .. }
            | WalRecord::Committed { .. }
            | WalRecord::TransactionCommit { .. } => {
                rdf_owners.clear();
                timestamps.clear();
            }
            WalRecord::CdcRetention {
                epoch,
                generation,
                previous_floor,
                floor,
                next_sequence,
            } => {
                // Publish this prefix into the private startup destination so
                // the floor is checked against its exact preimage, not against
                // later events that happened to be in the same WAL suffix.
                PreparedCdcBatch { log, groups }
                    .prepare_publication()?
                    .publish();
                groups = hashbrown::HashMap::new();
                recover_retention(
                    log,
                    *epoch,
                    *generation,
                    *previous_floor,
                    *floor,
                    *next_sequence,
                )?;
            }
            _ => {}
        }
    }
    let prepared = PreparedCdcBatch { log, groups }.prepare_publication()?;
    prepared.publish();
    if high_timestamp != HlcTimestamp::zero() {
        log.clock.update(high_timestamp);
    }
    Ok(())
}

fn recover_retention(
    log: &CdcLog,
    epoch: EpochId,
    generation: u64,
    previous_floor: u64,
    floor: u64,
    next_sequence: u64,
) -> Result<()> {
    let mut state = log.events.write();
    if generation != state.generation {
        return Err(invalid("retention names a foreign feed generation"));
    }
    // A checkpoint can already contain a floor record left in its WAL prefix.
    // It may not move that installed cut backwards or invent later sequences.
    if floor <= state.floor && next_sequence <= state.next_sequence {
        return Ok(());
    }
    if previous_floor != state.floor
        || next_sequence != state.next_sequence
        || floor <= previous_floor
        || floor > next_sequence
        || state.back().is_some_and(|event| event.epoch > epoch)
    {
        return Err(invalid(
            "retention does not match its native sequence preimage",
        ));
    }
    let count = usize::try_from(floor - previous_floor)
        .map_err(|_| invalid("retention prefix length overflow"))?;
    if count > state.len()
        || state
            .get(count)
            .is_some_and(|event| event.epoch == state[count - 1].epoch)
    {
        return Err(invalid("retention splits a committed epoch"));
    }
    state.drain(..count);
    state.floor = floor;
    state.trim_entity_index();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "lpg")]
    #[test]
    fn retention_replay_validates_preimage_and_whole_epoch_before_pruning() {
        let log = CdcLog::new();
        let events = decode(
            TransactionId::new(1),
            EpochId::new(2),
            1,
            &encode(&batch()).unwrap(),
        )
        .unwrap();
        let mut second = events[0].clone();
        second.timestamp = HlcTimestamp::from_u64(second.timestamp.as_u64() + 1);
        log.record_batch(events.into_iter().chain([second]));
        let before = super::super::checkpoint::encode(&log).unwrap();
        for (epoch, generation, previous, floor, next) in [
            (2, 2, 1, 3, 3), // foreign generation
            (2, 1, 2, 3, 3), // foreign preimage
            (2, 1, 1, 2, 3), // split epoch
            (2, 1, 1, 4, 4), // invented sequence
            (1, 1, 1, 3, 3), // cut before retained events
        ] {
            assert!(
                recover_retention(&log, EpochId::new(epoch), generation, previous, floor, next)
                    .is_err()
            );
            assert_eq!(super::super::checkpoint::encode(&log).unwrap(), before);
        }
        recover_retention(&log, EpochId::new(2), 1, 1, 3, 3).unwrap();
        assert_eq!(log.events.read().floor, 3);
        assert!(log.events.read().is_empty());
        // An installed checkpoint may already cover this exact transition.
        recover_retention(&log, EpochId::new(2), 1, 1, 3, 3).unwrap();
    }

    #[cfg(feature = "lpg")]
    fn batch() -> Batch {
        Batch {
            transaction: 1,
            epoch: 2,
            model: 1,
            events: vec![Event {
                timestamp: 1234,
                incarnation: 0,
                kind: ChangeKind::Update,
                payload: Payload::Lpg {
                    graph: GraphPath::root(),
                    entity: EntityId::Node(NodeId::new(4)),
                    before: Some(vec![("a".into(), encode_value(&Value::Int64(1)).unwrap())]),
                    after: Some(vec![
                        ("a".into(), encode_value(&Value::Float64(-0.0)).unwrap()),
                        ("z".into(), encode_value(&Value::from("bytes\n\0")).unwrap()),
                    ]),
                    labels: None,
                    edge_type: None,
                    src: None,
                    dst: None,
                },
            }],
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn canonical_images_round_trip_without_map_order_or_float_loss() {
        let bytes = encode(&batch()).unwrap();
        let events = decode(TransactionId::new(1), EpochId::new(2), 1, &bytes).unwrap();
        let event = &events[0];
        assert_eq!(
            event.graph_incarnation,
            Some(GraphIncarnationId::DEFAULT_GRAPH)
        );
        let Value::Float64(value) = event.after.as_ref().unwrap()["a"] else {
            panic!("float lost");
        };
        assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
        let mut second = batch();
        second.events = vec![Event::from_event(event).unwrap()];
        assert_eq!(encode(&second).unwrap(), bytes);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn malformed_payloads_fail_before_any_feed_publication() {
        let valid = encode(&batch()).unwrap();
        for (tx, epoch, model) in [(2, 2, 1), (1, 3, 1), (1, 2, 2)] {
            assert!(decode(TransactionId::new(tx), EpochId::new(epoch), model, &valid).is_err());
        }
        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(decode(TransactionId::new(1), EpochId::new(2), 1, &trailing).is_err());
        let mut unknown = valid.clone();
        unknown[4] = 2;
        assert!(decode(TransactionId::new(1), EpochId::new(2), 1, &unknown).is_err());
        for case in 0..5 {
            let mut invalid_batch = batch();
            match case {
                0 => invalid_batch.events[0].timestamp = u64::MAX,
                1 => invalid_batch.events[0].incarnation = 3,
                2 => invalid_batch.events.clear(),
                3 => {
                    if let Payload::Lpg {
                        after: Some(properties),
                        ..
                    } = &mut invalid_batch.events[0].payload
                    {
                        properties.reverse();
                    }
                }
                _ => {
                    if let Payload::Lpg {
                        after: Some(properties),
                        ..
                    } = &mut invalid_batch.events[0].payload
                    {
                        properties[1].0 = "a".into();
                    }
                }
            }
            let log = CdcLog::new();
            assert!(
                recover(
                    RecoverySource {
                        log: &log,
                        root: &std::sync::Arc::new(
                            grafeo_core::graph::lpg::LpgStore::new().unwrap()
                        )
                    },
                    &[WalRecord::CdcBatch {
                        transaction_id: TransactionId::new(1),
                        epoch: EpochId::new(2),
                        model: 1,
                        payload: encode(&invalid_batch).unwrap()
                    }]
                )
                .is_err()
            );
            assert_eq!(log.event_count(), 0);
        }
        // Standard-bincode header then a malicious u64 vector length. The
        // decoder's allocation budget rejects it before allocating that count.
        let mut huge_count = MAGIC.to_vec();
        huge_count.extend_from_slice(&[1, 2, 1, 253]);
        huge_count.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode(TransactionId::new(1), EpochId::new(2), 1, &huge_count).is_err());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn forged_named_lifetime_rejects_the_whole_feed_postimage() {
        let root = std::sync::Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        root.create_graph("named").unwrap();
        let mut valid = batch();
        valid.transaction = 11;
        let mut forged = batch();
        forged.transaction = 12;
        forged.epoch = 3;
        forged.events[0].timestamp += 1;
        forged.events[0].incarnation = 42;
        if let Payload::Lpg { graph, .. } = &mut forged.events[0].payload {
            *graph = GraphPath::from_components(&["named"]).unwrap();
        }
        let log = CdcLog::new();
        let records = [
            WalRecord::CdcBatch {
                transaction_id: TransactionId::new(11),
                epoch: EpochId::new(2),
                model: 1,
                payload: encode(&valid).unwrap(),
            },
            WalRecord::CommittedWithCdc {
                transaction_id: TransactionId::new(11),
                epoch: EpochId::new(2),
                models: 1,
            },
            WalRecord::CdcBatch {
                transaction_id: TransactionId::new(12),
                epoch: EpochId::new(3),
                model: 1,
                payload: encode(&forged).unwrap(),
            },
            WalRecord::CommittedWithCdc {
                transaction_id: TransactionId::new(12),
                epoch: EpochId::new(3),
                models: 1,
            },
        ];
        let error = recover(
            RecoverySource {
                log: &log,
                root: &root,
            },
            &records,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("foreign native lifetime"),
            "{error}"
        );
        assert_eq!(
            log.event_count(),
            0,
            "a late invalid event must not install the valid prefix"
        );
    }

    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    #[test]
    fn rdf_directory_reopen_restores_the_feed() {
        use crate::config::{DurabilityMode, StorageFormat};
        use crate::{Config, GrafeoDB, GraphModel};
        let dir = tempfile::tempdir().unwrap();
        let config = Config::persistent(dir.path())
            .with_graph_model(GraphModel::Rdf)
            .with_cdc()
            .with_storage_format(StorageFormat::WalDirectory)
            .with_wal_durability(DurabilityMode::Sync);
        let db = GrafeoDB::with_config(config.clone()).unwrap();
        db.session().execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "line\nquote\"" . GRAPH <urn:g> { <urn:s> <urn:p> "01"^^<http://www.w3.org/2001/XMLSchema#integer> . } }"#).unwrap();
        let expected = serde_json::to_value(
            db.changes_between(EpochId::INITIAL, EpochId::PENDING)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(expected.as_array().unwrap().len(), 2);
        db.close().unwrap();
        drop(db);
        for _ in 0..2 {
            let db = GrafeoDB::with_config(config.clone()).unwrap();
            assert_eq!(
                serde_json::to_value(
                    db.changes_between(EpochId::INITIAL, EpochId::PENDING)
                        .unwrap()
                )
                .unwrap(),
                expected
            );
            assert_eq!(
                db.session()
                    .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
                    .unwrap()
                    .row_count(),
                1
            );
            db.close().unwrap();
        }
    }
}
