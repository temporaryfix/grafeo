//! Finite fixture collection through the production bounded feed reader.

use std::ops::RangeInclusive;

use grafeo_common::types::{DurableCursor, EpochId};
use grafeo_common::utils::error::Result;
use grafeo_engine::cdc::{ChangeEvent, ChangePage, EntityHistoryQuery};

pub enum FixtureSelection {
    Epochs(RangeInclusive<EpochId>),
    Entity(EntityHistoryQuery),
}

impl From<RangeInclusive<EpochId>> for FixtureSelection {
    fn from(epochs: RangeInclusive<EpochId>) -> Self {
        Self::Epochs(epochs)
    }
}

impl From<EntityHistoryQuery> for FixtureSelection {
    fn from(query: EntityHistoryQuery) -> Self {
        Self::Entity(query)
    }
}

pub trait CdcFixtureChanges {
    fn fixture_page(
        &self,
        selection: &FixtureSelection,
        cursor: Option<&DurableCursor>,
    ) -> Result<ChangePage>;

    fn fixture_changes(&self, selection: impl Into<FixtureSelection>) -> Result<Vec<ChangeEvent>> {
        let selection = selection.into();
        let mut cursor = None;
        let mut events = Vec::new();
        loop {
            let page = self.fixture_page(&selection, cursor.as_ref())?;
            assert!(page.events.len() <= 2);
            if cursor == Some(page.next) {
                break;
            }
            cursor = Some(page.next);
            events.extend(page.events.into_iter().filter(|event| match &selection {
                FixtureSelection::Epochs(epochs) => epochs.contains(&event.epoch),
                FixtureSelection::Entity(_) => true,
            }));
        }
        Ok(events)
    }
}

impl CdcFixtureChanges for grafeo_engine::GrafeoDB {
    fn fixture_page(
        &self,
        selection: &FixtureSelection,
        cursor: Option<&DurableCursor>,
    ) -> Result<ChangePage> {
        match selection {
            FixtureSelection::Epochs(_) => self.changes_after(cursor, 2, 64 * 1024 * 1024),
            FixtureSelection::Entity(query) => {
                self.history_after(query, cursor, 2, 64 * 1024 * 1024)
            }
        }
    }
}

impl CdcFixtureChanges for grafeo_engine::Session {
    fn fixture_page(
        &self,
        selection: &FixtureSelection,
        cursor: Option<&DurableCursor>,
    ) -> Result<ChangePage> {
        match selection {
            FixtureSelection::Epochs(_) => self.changes_after(cursor, 2, 64 * 1024 * 1024),
            FixtureSelection::Entity(query) => {
                self.history_after(query, cursor, 2, 64 * 1024 * 1024)
            }
        }
    }
}
