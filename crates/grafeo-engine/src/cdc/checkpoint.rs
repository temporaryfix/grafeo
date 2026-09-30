//! Bounded retained-window postimages. Sequence/floor survive an empty window.
use super::{CdcLog, FeedState, cdc_capacity_error, codec::Event};
use grafeo_common::types::{EpochId, HlcTimestamp};
use grafeo_common::utils::error::{Error, Result};

pub(crate) const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_EVENTS: usize = 1_000_000;
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;

fn invalid(message: &str) -> Error {
    Error::Serialization(format!("CDC checkpoint: {message}"))
}
fn append(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_BYTES.saturating_sub(out.len()) {
        return Err(invalid("retained window exceeds 64 MiB"));
    }
    out.try_reserve(bytes.len())
        .map_err(|_| cdc_capacity_error())?;
    out.extend_from_slice(bytes);
    Ok(())
}

pub(crate) fn encode(log: &CdcLog) -> Result<Vec<u8>> {
    let state = log.events.read();
    let count = state.len();
    if count > MAX_EVENTS || state.next_sequence.checked_sub(state.floor) != Some(count as u64) {
        return Err(invalid("invalid retained sequence window"));
    }
    let mut out = Vec::new();
    for value in [
        state.generation,
        state.floor,
        state.next_sequence,
        state.high_timestamp.as_u64(),
    ] {
        append(&mut out, &value.to_le_bytes())?;
    }
    append(
        &mut out,
        &u32::try_from(count)
            .map_err(|_| invalid("event count overflow"))?
            .to_le_bytes(),
    )?;
    for event in state.iter() {
        let native = Event::from_event(event)?;
        let bytes = bincode::serde::encode_to_vec(&native, bincode::config::standard())
            .map_err(|error| Error::Serialization(error.to_string()))?;
        if bytes.len() > MAX_EVENT_BYTES {
            return Err(invalid("event exceeds 16 MiB"));
        }
        append(&mut out, &event.epoch.as_u64().to_le_bytes())?;
        append(&mut out, &[if event.entity_id.is_triple() { 2 } else { 1 }])?;
        append(
            &mut out,
            &u32::try_from(bytes.len())
                .map_err(|_| invalid("event size overflow"))?
                .to_le_bytes(),
        )?;
        append(&mut out, &bytes)?;
    }
    Ok(out)
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self.0.get(..n).ok_or_else(|| invalid("truncated image"))?;
        self.0 = &self.0[n..];
        Ok(bytes)
    }
    fn u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }
}

pub(crate) struct PreparedRestore(FeedState);
impl PreparedRestore {
    /// No decoding or allocation remains when the publication fence is held.
    pub(crate) fn install(self, log: &CdcLog) {
        let high = self.0.high_timestamp;
        *log.events.write() = self.0;
        if high != HlcTimestamp::zero() {
            log.clock.update(high);
        }
    }
}

pub(crate) fn empty() -> PreparedRestore {
    PreparedRestore(FeedState::default())
}

pub(crate) fn prepare(bytes: &[u8], cut_epoch: EpochId) -> Result<PreparedRestore> {
    if bytes.len() > MAX_BYTES {
        return Err(invalid("image exceeds 64 MiB"));
    }
    let mut reader = Reader(bytes);
    let generation = reader.u64()?;
    let floor = reader.u64()?;
    let next_sequence = reader.u64()?;
    let high_timestamp = HlcTimestamp::from_u64(reader.u64()?);
    let count = reader.u32()? as usize;
    if generation == 0
        || generation == u64::MAX
        || floor == 0
        || next_sequence == u64::MAX
        || high_timestamp.as_u64() == u64::MAX
        || count > MAX_EVENTS
        || next_sequence.checked_sub(floor) != Some(count as u64)
        || count > MAX_BYTES / std::mem::size_of::<super::ChangeEvent>()
    {
        return Err(invalid("reserved coordinate or invalid bounded window"));
    }
    let mut state = FeedState {
        generation,
        floor,
        next_sequence,
        high_timestamp,
        ..FeedState::default()
    };
    let mut previous = None;
    for _ in 0..count {
        let epoch = EpochId::new(reader.u64()?);
        let model = reader.take(1)?[0];
        let length = reader.u32()? as usize;
        if epoch == EpochId::PENDING
            || epoch.as_u64() == 0
            || epoch > cut_epoch
            || length > MAX_EVENT_BYTES
        {
            return Err(invalid("foreign publication coordinate or oversized event"));
        }
        let bytes = reader.take(length)?;
        let (native, consumed): (Event, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_EVENT_BYTES>(),
        )
        .map_err(|error| Error::Serialization(error.to_string()))?;
        if consumed != bytes.len()
            || bincode::serde::encode_to_vec(&native, bincode::config::standard())
                .map_err(|error| Error::Serialization(error.to_string()))?
                != bytes
        {
            return Err(invalid("noncanonical native event"));
        }
        native.validate(model)?;
        let event = native.into_event(epoch)?;
        let coordinate = (epoch, event.timestamp);
        if previous.is_some_and(|last| coordinate <= last) || event.timestamp > high_timestamp {
            return Err(invalid(
                "unordered/duplicate event or regressive clock high-water",
            ));
        }
        previous = Some(coordinate);
        state.try_reserve(1).map_err(|_| cdc_capacity_error())?;
        if !state.by_entity.contains_key(&event.entity_id) {
            state
                .by_entity
                .try_reserve(1)
                .map_err(|_| cdc_capacity_error())?;
        }
        let sequence = state.floor + state.len() as u64;
        let positions = state.by_entity.entry(event.entity_id).or_default();
        positions.try_reserve(1).map_err(|_| cdc_capacity_error())?;
        positions.push_back(sequence);
        state.push_back(event);
    }
    if !reader.0.is_empty() {
        return Err(invalid("trailing bytes"));
    }
    Ok(PreparedRestore(state))
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use grafeo_common::types::{GraphIncarnationId, GraphPath, NodeId};
    fn event(log: &CdcLog, epoch: u64) -> super::super::ChangeEvent {
        let mut event = log.node_create_event(
            NodeId::new(epoch),
            EpochId::new(epoch),
            None,
            None,
            &GraphPath::root(),
        );
        event.graph_incarnation = Some(GraphIncarnationId::DEFAULT_GRAPH);
        event.timestamp = HlcTimestamp::from_u64(100 + epoch);
        event
    }
    #[test]
    fn retained_sequence_floor_and_clock_survive_an_empty_checkpoint() {
        let log = CdcLog::new();
        let future = HlcTimestamp::new(HlcTimestamp::now().physical_ms() + 3_600_000, 7);
        for epoch in 2..5 {
            let mut event = event(&log, epoch);
            if epoch == 4 {
                event.timestamp = future;
            }
            log.record(event);
        }
        log.prune_before(EpochId::new(9));
        assert_eq!(log.heap_memory_bytes(), (0, 0, 0));
        let bytes = encode(&log).unwrap();
        let restored = CdcLog::new();
        prepare(&bytes, EpochId::new(9)).unwrap().install(&restored);
        assert_eq!(encode(&restored).unwrap(), bytes);
        let state = restored.events.read();
        assert_eq!(
            (state.generation, state.floor, state.next_sequence),
            (1, 4, 4)
        );
        assert_eq!(state.high_timestamp, future);
        drop(state);
        assert!(restored.next_timestamp() > future);
        restored.record(event(&restored, 10));
        let state = restored.events.read();
        assert_eq!((state.floor, state.next_sequence), (4, 5));
    }
    #[test]
    fn entity_index_survives_checkpoint_prefix_pruning_and_reinsertion() {
        let log = CdcLog::new();
        for (epoch, id) in [(2, 1), (3, 2), (4, 1), (5, 3)] {
            let mut row = event(&log, epoch);
            row.entity_id = super::super::EntityId::Node(NodeId::new(id));
            log.record(row);
        }
        let restored = CdcLog::new();
        prepare(&encode(&log).unwrap(), EpochId::new(5))
            .unwrap()
            .install(&restored);
        let epochs = |log: &CdcLog, id| {
            log.history(super::super::EntityId::Node(NodeId::new(id)))
                .iter()
                .map(|row| row.epoch.as_u64())
                .collect::<Vec<_>>()
        };
        assert_eq!(epochs(&restored, 1), [2, 4]);
        assert_eq!(epochs(&restored, 2), [3]);
        assert_eq!(epochs(&restored, 3), [5]);
        restored.prune_before(EpochId::new(4));
        assert_eq!(epochs(&restored, 1), [4]);
        assert!(epochs(&restored, 2).is_empty());
        assert_eq!(epochs(&restored, 3), [5]);
        let mut row = event(&restored, 6);
        row.entity_id = super::super::EntityId::Node(NodeId::new(2));
        restored.record(row);
        let reopened = CdcLog::new();
        prepare(&encode(&restored).unwrap(), EpochId::new(6))
            .unwrap()
            .install(&reopened);
        for (id, expected) in [(1, vec![4]), (2, vec![6]), (3, vec![5])] {
            assert_eq!(epochs(&restored, id), expected);
            assert_eq!(epochs(&reopened, id), expected);
        }
    }

    #[test]
    fn malformed_window_count_clock_order_and_lengths_fail_before_install() {
        let log = CdcLog::new();
        log.record(event(&log, 2));
        log.record(event(&log, 3));
        let bytes = encode(&log).unwrap();
        for field in [0, 8, 16, 24] {
            let mut forged = bytes.clone();
            forged[field..field + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(prepare(&forged, EpochId::new(3)).is_err());
        }
        let mut forged = bytes.clone();
        forged[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(prepare(&forged, EpochId::new(3)).is_err());
        let mut forged = bytes.clone();
        forged[45..49].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(prepare(&forged, EpochId::new(3)).is_err());
        let mut forged = bytes.clone();
        forged.push(0);
        assert!(prepare(&forged, EpochId::new(3)).is_err());
        assert!(prepare(&bytes, EpochId::new(1)).is_err());
        for end in [0, 8, 32, 36, bytes.len() - 1] {
            assert!(prepare(&bytes[..end], EpochId::new(3)).is_err());
        }
        // A second complete copy of the first row has a valid byte encoding but
        // repeats its causal coordinate instead of supplying the next event.
        let mut length = [0; 4];
        length.copy_from_slice(&bytes[45..49]);
        let row_end = 49 + u32::from_le_bytes(length) as usize;
        let mut duplicate = bytes[..row_end].to_vec();
        duplicate.extend_from_slice(&bytes[36..row_end]);
        assert!(prepare(&duplicate, EpochId::new(3)).is_err());
    }
}

/// Native model/allocator authority already validated by the enclosing image.
pub(crate) struct NativeCut<'a> {
    pub(crate) model: u8,
    pub(crate) lpg_next: u64,
    pub(crate) lpg_owners: &'a [(
        grafeo_common::types::GraphPath,
        grafeo_common::types::GraphIncarnationId,
    )],
    pub(crate) rdf_next: u64,
}
fn validate_native(state: &FeedState, cut: NativeCut<'_>) -> Result<()> {
    let mut owners = hashbrown::HashMap::new();
    owners
        .try_reserve(cut.lpg_owners.len())
        .map_err(|_| cdc_capacity_error())?;
    for (path, incarnation) in cut.lpg_owners {
        owners.insert(*incarnation, path);
    }
    for event in state.iter() {
        let incarnation = event
            .graph_incarnation
            .ok_or_else(|| invalid("missing native owner"))?;
        if event.entity_id.is_triple() {
            if !matches!(cut.model, 1 | 2) || incarnation.as_u64() >= cut.rdf_next {
                return Err(invalid(
                    "RDF event exceeds native model/allocator authority",
                ));
            }
        } else if !matches!(cut.model, 0 | 2)
            || incarnation.as_u64() >= cut.lpg_next
            || owners
                .get(&incarnation)
                .is_some_and(|path| event.lpg_graph.as_ref() != Some(*path))
        {
            return Err(invalid(
                "LPG event exceeds native model/allocator/path authority",
            ));
        }
    }
    Ok(())
}
#[cfg(any(feature = "lpg", feature = "triple-store"))]
impl PreparedRestore {
    pub(crate) fn validate_native(&self, cut: NativeCut<'_>) -> Result<()> {
        validate_native(&self.0, cut)
    }
}
#[cfg(all(
    feature = "grafeo-file",
    any(feature = "lpg", feature = "triple-store")
))]
impl CdcLog {
    pub(crate) fn validate_checkpoint_native(&self, cut: NativeCut<'_>) -> Result<()> {
        validate_native(&self.events.read(), cut)
    }
}

#[cfg(all(
    test,
    not(feature = "lpg"),
    feature = "triple-store",
    feature = "sparql",
    feature = "wal"
))]
#[test]
fn rdf_only_container_checkpoint_preserves_native_feed_twice() {
    use crate::{Config, GrafeoDB, GraphModel};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rdf-feed.grafeo");
    let cfg = Config::persistent(&path)
        .with_cdc()
        .with_graph_model(GraphModel::Rdf);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    db.session()
        .execute_sparql("INSERT DATA { <urn:s> <urn:p> 1 . GRAPH <urn:g> { <urn:s> <urn:p> 2 . } }")
        .unwrap();
    let events = |db: &GrafeoDB| {
        serde_json::to_value(
            db.changes_between(EpochId::INITIAL, EpochId::PENDING)
                .unwrap(),
        )
        .unwrap()
    };
    let expected = events(&db);
    assert_eq!(expected.as_array().unwrap().len(), 2);
    let cursor = db.changes_after(None, 1, 4096).unwrap().next;
    assert_eq!(cursor.feed.model, 2);
    let tail = db.changes_after(Some(&cursor), 1, 4096).unwrap();
    assert_eq!(tail.next.sequence, 2);
    let cut = db.world_cut().unwrap();
    db.wal_checkpoint().unwrap();
    db.close().unwrap();
    drop(db);
    for _ in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert_eq!(events(&db), expected);
        let resumed = db.session().changes_after(Some(&cursor), 1, 4096).unwrap();
        assert_eq!(resumed.next, tail.next);
        assert_eq!(
            serde_json::to_value(resumed.events).unwrap(),
            serde_json::to_value(&tail.events).unwrap()
        );
        assert_eq!(db.world_cut().unwrap(), cut);
        db.close().unwrap();
    }
}
