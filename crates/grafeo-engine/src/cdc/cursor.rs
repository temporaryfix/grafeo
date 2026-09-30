//! Bounded readers over the retained commit-ordered feed.

use super::{CdcLog, ChangeEvent, EntityId, cdc_capacity_error};
use grafeo_common::types::{DurableCursor, EpochId, FeedId, GraphPath, StoreId};
use grafeo_common::utils::error::{Error, Result, StorageError};

/// A bounded page and its exclusive resume coordinate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChangePage {
    /// Native events in retained commit order.
    pub events: Vec<ChangeEvent>,
    /// Position after the inspected prefix, including rows hidden by Session
    /// visibility. An empty filtered page can advance this position; an
    /// unchanged cursor means there are no more retained rows to inspect.
    pub next: DurableCursor,
}

/// Model-specific coordinate selection for one entity's retained history.
#[derive(Debug, Clone, Default)]
pub enum HistoryGraph {
    /// Aggregate colliding local IDs across all authorized graph coordinates.
    #[default]
    All,
    /// One exact LPG component path, including its successive incarnations.
    Lpg(GraphPath),
    /// One exact RDF graph name; `None` selects the RDF default graph.
    Rdf(Option<String>),
}

/// Selection applied to bounded pages of one entity's retained native history.
///
/// A cursor describes the shared feed, not this query. To widen the selection
/// and recover earlier events, start at `None` or an appropriately earlier cut.
#[derive(Debug, Clone)]
pub struct EntityHistoryQuery {
    /// Native typed entity ID. Local LPG IDs can collide across graph paths.
    pub entity_id: EntityId,
    /// Exact graph coordinate, or all authorized coordinates.
    pub graph: HistoryGraph,
    /// Inclusive minimum commit epoch.
    pub since_epoch: EpochId,
}

impl EntityHistoryQuery {
    /// Selects this entity across authorized coordinates from the initial epoch.
    #[must_use]
    pub fn new(entity_id: impl Into<EntityId>) -> Self {
        Self {
            entity_id: entity_id.into(),
            graph: HistoryGraph::All,
            since_epoch: EpochId::INITIAL,
        }
    }

    pub(crate) fn matches(&self, event: &ChangeEvent) -> bool {
        event.epoch >= self.since_epoch
            && match &self.graph {
                HistoryGraph::All => true,
                HistoryGraph::Lpg(path) => event.graph_path() == Some(path),
                HistoryGraph::Rdf(name) => {
                    event.entity_id.is_triple() && event.triple_graph.as_deref() == name.as_deref()
                }
            }
    }
}

/// Counts serialized bytes without allocating an intermediate event image.
struct ByteBudget {
    remaining: usize,
    exceeded: bool,
}

impl bincode::enc::write::Writer for ByteBudget {
    #[inline]
    fn write(&mut self, bytes: &[u8]) -> std::result::Result<(), bincode::error::EncodeError> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(bincode::error::EncodeError::UnexpectedEnd);
        }
        self.remaining -= bytes.len();
        Ok(())
    }
}

impl CdcLog {
    pub(crate) fn page(
        &self,
        authority: (StoreId, FeedId, EpochId),
        entity: Option<EntityId>,
        cursor: Option<&DurableCursor>,
        max_events: usize,
        max_bytes: usize,
        visible: impl Fn(&ChangeEvent) -> bool,
    ) -> Result<ChangePage> {
        let (store, feed, epoch) = authority;
        let state = self.events.read();
        let initial = if let Some(cursor) = cursor {
            cursor.validate()?;
            if cursor.store_id != store || cursor.feed != feed {
                return Err(StorageError::CursorForeign.into());
            }
            if cursor.generation != state.generation || cursor.epoch > epoch {
                return Err(StorageError::CursorInvalid.into());
            }
            if cursor.sequence < state.floor - 1 {
                return Err(StorageError::CursorEvicted.into());
            }
            if cursor.sequence >= state.next_sequence {
                return Err(StorageError::CursorInvalid.into());
            }
            if cursor.sequence >= state.floor {
                let index = usize::try_from(cursor.sequence - state.floor)
                    .map_err(|_| StorageError::CursorInvalid)?;
                if state
                    .get(index)
                    .is_none_or(|event| event.epoch != cursor.epoch)
                {
                    return Err(StorageError::CursorInvalid.into());
                }
            } else if cursor.sequence == 0 && cursor.epoch != EpochId::INITIAL {
                return Err(StorageError::CursorInvalid.into());
            }
            (cursor.sequence, cursor.epoch)
        } else {
            (state.floor - 1, EpochId::INITIAL)
        };
        if max_events == 0 || max_bytes == 0 {
            return Err(Error::InvalidValue(
                "change pages require positive row and byte limits".into(),
            ));
        }
        let positions = entity.and_then(|id| state.by_entity.get(&id));
        let start = if entity.is_some() {
            positions.map_or(0, |positions| {
                positions.partition_point(|sequence| *sequence <= initial.0)
            })
        } else {
            usize::try_from(initial.0 + 1 - state.floor).map_err(|_| StorageError::CursorInvalid)?
        };
        let available = if entity.is_some() {
            positions.map_or(0, |positions| positions.len() - start)
        } else {
            state.len() - start
        };
        let capacity = max_events.min(available).min(max_bytes);
        let mut events = Vec::new();
        let mut budget = ByteBudget {
            remaining: max_bytes,
            exceeded: false,
        };
        let mut position = initial;
        // An exhausted entity index proves the remaining feed contains no
        // candidate for this entity. Advance to the shared tail in constant
        // time; do not scan or copy unrelated events just to discover EOF.
        if entity.is_some()
            && available == 0
            && let Some(tail) = state.back()
        {
            position = (state.next_sequence - 1, tail.epoch);
        }
        for offset in 0..available.min(max_events) {
            let sequence = positions.map_or_else(
                || state.floor + (start + offset) as u64,
                |positions| positions[start + offset],
            );
            let event = sequence
                .checked_sub(state.floor)
                .and_then(|offset| usize::try_from(offset).ok())
                .and_then(|offset| state.get(offset))
                .ok_or_else(|| {
                    Error::Internal("retained CDC entity index is inconsistent".into())
                })?;
            if !visible(event) {
                position = (sequence, event.epoch);
                continue;
            }
            if let Err(error) =
                bincode::serde::encode_into_writer(event, &mut budget, bincode::config::standard())
            {
                if !budget.exceeded {
                    return Err(Error::Serialization(format!(
                        "change event byte accounting: {error}"
                    )));
                }
                if events.is_empty() {
                    return Err(StorageError::Full.into());
                }
                break;
            }
            if events.capacity() == 0 {
                // Allocate only after the first visible, in-budget event is
                // ready. Hidden-only pages require no owned event slots.
                events
                    .try_reserve_exact(capacity)
                    .map_err(|_| cdc_capacity_error())?;
            }
            events.push(event.clone());
            position = (sequence, event.epoch);
        }
        let next = match cursor {
            Some(cursor) if position == initial => *cursor,
            _ => DurableCursor::new(store, feed, state.generation, position.0, position.1)?,
        };
        Ok(ChangePage { events, next })
    }
}
