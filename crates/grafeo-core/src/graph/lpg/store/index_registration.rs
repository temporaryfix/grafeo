//! Exact observations of physical index registrations.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::{LpgStore, TransportExtractAuthority};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use crate::graph::lpg::encode_index_key;
#[cfg(feature = "text-index")]
use crate::index::text::InvertedIndex;
#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;
use grafeo_common::types::PropertyKey;
use grafeo_common::utils::error::{Error, Result, TransactionError};
use std::sync::{Arc, Weak};

/// Equality evidence for one physical representation, stable if its value moves.
pub(super) struct PhysicalStoreIdentity;

/// Retained equality evidence for one successful registry publication, not authority.
pub(super) struct RegistrationIncarnation;

pub(super) use super::property_index::PropertyIndexRows;

/// Identity and payload travel together. Cloning Property's Arc retains its rows
/// cheaply for a future prepared registry postimage; it does not copy row contents.
#[derive(Clone)]
pub(super) struct RegisteredIndex<T> {
    pub(super) registration: Arc<RegistrationIncarnation>,
    pub(super) payload: T,
}

impl<T> RegisteredIndex<T> {
    pub(super) fn new(payload: T) -> Self {
        Self {
            registration: Arc::new(RegistrationIncarnation),
            payload,
        }
    }

    /// A representation transfer preserves the logical registration but may
    /// replace its physical payload with empty membership or a frozen snapshot.
    #[cfg(feature = "compact-store")]
    pub(super) fn with_same_registration<U>(&self, payload: U) -> RegisteredIndex<U> {
        RegisteredIndex {
            registration: Arc::clone(&self.registration),
            payload,
        }
    }
}

impl<T> std::ops::Deref for RegisteredIndex<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.payload
    }
}

pub(super) enum ObservedIndex {
    Property(PropertyKey, Weak<PropertyIndexRows>),
    #[cfg(feature = "vector-index")]
    Vector(String, Weak<VectorIndexKind>),
    #[cfg(feature = "text-index")]
    Text(String, Weak<parking_lot::RwLock<InvertedIndex>>),
}

/// An opaque observation of one physical store's index registration.
///
/// This is equality evidence, not write authority or a content snapshot. It
/// must be reacquired after representation replacement. An absent observation
/// does not prove a history of absence. Weak target identities retain control
/// blocks against address reuse, but do not keep dropped index contents alive.
pub struct IndexRegistrationObservation {
    physical: Arc<PhysicalStoreIdentity>,
    incarnation: Arc<TransportExtractAuthority>,
    registration: Arc<RegistrationIncarnation>,
    index: ObservedIndex,
}

impl IndexRegistrationObservation {
    pub(super) fn target(&self) -> &ObservedIndex {
        &self.index
    }

    pub(super) fn matches_property(
        &self,
        current: &RegisteredIndex<Arc<PropertyIndexRows>>,
    ) -> bool {
        matches!(&self.index, ObservedIndex::Property(_, target)
            if Arc::ptr_eq(&current.registration, &self.registration)
                && Weak::ptr_eq(&Arc::downgrade(&current.payload), target))
    }

    #[cfg(feature = "vector-index")]
    pub(super) fn matches_vector(&self, current: &RegisteredIndex<Arc<VectorIndexKind>>) -> bool {
        matches!(&self.index, ObservedIndex::Vector(_, target)
            if Arc::ptr_eq(&current.registration, &self.registration)
                && Weak::ptr_eq(&Arc::downgrade(&current.payload), target))
    }

    #[cfg(feature = "text-index")]
    pub(super) fn matches_text(
        &self,
        current: &RegisteredIndex<crate::index::text::RegisteredTextIndex>,
    ) -> bool {
        matches!(&self.index, ObservedIndex::Text(_, target)
            if Arc::ptr_eq(&current.registration, &self.registration)
                && Weak::ptr_eq(&current.target_identity(), target))
    }
}

fn conflict(reason: &str) -> Error {
    Error::Transaction(TransactionError::WriteConflict(format!(
        "index registration: {reason}"
    )))
}

impl LpgStore {
    /// Observes an existing property-index registration without granting write authority.
    #[must_use]
    pub fn observe_property_index(&self, property: &str) -> Option<IndexRegistrationObservation> {
        let _maintenance = self.pin_maintenance();
        let incarnation = self.transport_extract_authority.read();
        let key = PropertyKey::new(property);
        let indexes = self.property_indexes.read();
        let current = indexes.get(&key)?;
        Some(self.registration_observation(
            &incarnation,
            &current.registration,
            ObservedIndex::Property(key, Arc::downgrade(&current.payload)),
        ))
    }

    /// Observes an existing vector-index registration, not its current contents.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn observe_vector_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<IndexRegistrationObservation> {
        let _maintenance = self.pin_maintenance();
        let incarnation = self.transport_extract_authority.read();
        let key = encode_index_key(label, property);
        let indexes = self.vector_indexes.read();
        let current = indexes.get(&key)?;
        Some(self.registration_observation(
            &incarnation,
            &current.registration,
            ObservedIndex::Vector(key, Arc::downgrade(&current.payload)),
        ))
    }

    /// Observes the registry-owned concrete text target, not a replaceable caller shell.
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn observe_text_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<IndexRegistrationObservation> {
        let _maintenance = self.pin_maintenance();
        let incarnation = self.transport_extract_authority.read();
        let key = encode_index_key(label, property);
        let indexes = self.text_indexes.read();
        let current = indexes.get(&key)?;
        Some(self.registration_observation(
            &incarnation,
            &current.registration,
            ObservedIndex::Text(key, current.target_identity()),
        ))
    }

    fn registration_observation(
        &self,
        incarnation: &Arc<TransportExtractAuthority>,
        registration: &Arc<RegistrationIncarnation>,
        index: ObservedIndex,
    ) -> IndexRegistrationObservation {
        IndexRegistrationObservation {
            physical: Arc::clone(&self.index_physical_identity),
            incarnation: Arc::clone(incarnation),
            registration: Arc::clone(registration),
            index,
        }
    }

    pub(super) fn validate_registration_store(
        &self,
        observation: &IndexRegistrationObservation,
        incarnation: &Arc<TransportExtractAuthority>,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.index_physical_identity, &observation.physical) {
            return Err(conflict("observation belongs to another physical store"));
        }
        if !Arc::ptr_eq(incarnation, &observation.incarnation) {
            return Err(conflict("store incarnation changed"));
        }
        Ok(())
    }

    /// Checks registration equality and current mutation authority at this instant.
    ///
    /// This does not retain a publication fence. Use [`Self::drop_index_if_unchanged`]
    /// for atomic conditional removal; validate-then-ordinary-drop is not equivalent.
    ///
    /// # Errors
    /// Returns a write conflict for absent/stale/foreign registrations or when the
    /// current thread cannot mutate this sealed or retired representation.
    pub fn validate_index_registration(
        &self,
        observation: &IndexRegistrationObservation,
    ) -> Result<()> {
        let _mutation = self
            .pin_mutation()
            .ok_or_else(|| conflict("mutation authority denied or representation retired"))?;
        let incarnation = self.transport_extract_authority.read();
        self.validate_registration_store(observation, &incarnation)?;
        let matches = match &observation.index {
            ObservedIndex::Property(key, target) => self
                .property_indexes
                .read()
                .get(key)
                .is_some_and(|current| {
                    Arc::ptr_eq(&current.registration, &observation.registration)
                        && Weak::ptr_eq(&Arc::downgrade(&current.payload), target)
                }),
            #[cfg(feature = "vector-index")]
            ObservedIndex::Vector(key, target) => {
                self.vector_indexes.read().get(key).is_some_and(|current| {
                    Arc::ptr_eq(&current.registration, &observation.registration)
                        && Weak::ptr_eq(&Arc::downgrade(&current.payload), target)
                })
            }
            #[cfg(feature = "text-index")]
            ObservedIndex::Text(key, target) => {
                self.text_indexes.read().get(key).is_some_and(|current| {
                    Arc::ptr_eq(&current.registration, &observation.registration)
                        && Weak::ptr_eq(&current.target_identity(), target)
                })
            }
        };
        if !matches {
            return Err(conflict("registration is absent or changed"));
        }
        Ok(())
    }

    /// Removes exactly the observed registration under retained authority and registry guards.
    ///
    /// The observation supplies no write permission. Comparison and removal share
    /// one registry write guard; no unlocked lookup-then-drop is performed.
    ///
    /// # Errors
    /// Returns a write conflict without removing anything if the physical store,
    /// incarnation, registration or current mutation authority no longer matches.
    pub fn drop_index_if_unchanged(&self, observation: IndexRegistrationObservation) -> Result<()> {
        #[cfg(feature = "vector-index")]
        let _vector_transition = matches!(&observation.index, ObservedIndex::Vector(..))
            .then(VectorIndexKind::pin_scope_transition);
        #[cfg(feature = "text-index")]
        let _text_transition = matches!(&observation.index, ObservedIndex::Text(..))
            .then(InvertedIndex::pin_scope_transition);
        let _mutation = self
            .pin_mutation()
            .ok_or_else(|| conflict("mutation authority denied or representation retired"))?;
        let incarnation = self.transport_extract_authority.read();
        self.validate_registration_store(&observation, &incarnation)?;
        match &observation.index {
            ObservedIndex::Property(key, target) => {
                let mut indexes = self.property_indexes.write();
                if !indexes.get(key).is_some_and(|current| {
                    Arc::ptr_eq(&current.registration, &observation.registration)
                        && Weak::ptr_eq(&Arc::downgrade(&current.payload), target)
                }) {
                    return Err(conflict("registration is absent or changed"));
                }
                indexes.remove(key);
            }
            #[cfg(feature = "vector-index")]
            ObservedIndex::Vector(key, target) => {
                let mut indexes = self.vector_indexes.write();
                if !indexes.get(key).is_some_and(|current| {
                    Arc::ptr_eq(&current.registration, &observation.registration)
                        && Weak::ptr_eq(&Arc::downgrade(&current.payload), target)
                }) {
                    return Err(conflict("registration is absent or changed"));
                }
                indexes.remove(key);
            }
            #[cfg(feature = "text-index")]
            ObservedIndex::Text(key, target) => {
                let mut indexes = self.text_indexes.write();
                let current = indexes
                    .get(key)
                    .ok_or_else(|| conflict("registration is absent or changed"))?;
                if !Arc::ptr_eq(&current.registration, &observation.registration)
                    || !Weak::ptr_eq(&current.target_identity(), target)
                {
                    return Err(conflict("registration is absent or changed"));
                }
                // The matched entry stays registered under this same map guard.
                // Retain its real gate and target, not the observation's Weak.
                let registered = current.payload.clone();
                let _target_guard = registered.write();
                indexes.remove(key);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod aba_tests;

#[cfg(test)]
mod lifecycle_tests;

#[cfg(all(test, feature = "compact-store"))]
mod compact_tests;

#[cfg(test)]
mod tests {
    use super::super::LpgStore;
    use grafeo_common::types::Value;

    #[test]
    fn property_conditional_drop_removes_only_the_observed_registration() {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Item"]);
        store.set_node_property(node, "code", Value::Int64(7));
        store.create_property_index("code");
        store.create_property_index("other");
        let observed = store.observe_property_index("code").unwrap();
        assert!(store.validate_index_registration(&observed).is_ok());
        store.drop_index_if_unchanged(observed).unwrap();
        assert!(!store.has_property_index("code"));
        assert!(store.has_property_index("other"));
        assert_eq!(
            store.find_nodes_by_property("code", &Value::Int64(7)),
            vec![node]
        );
        assert!(store.observe_property_index("missing").is_none());
    }

    #[cfg(feature = "vector-index")]
    pub(super) fn vector() -> std::sync::Arc<crate::index::vector::VectorIndexKind> {
        use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
        std::sync::Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(3, DistanceMetric::Cosine),
            47,
        )))
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn vector_conditional_drop_does_not_remove_another_key() {
        let store = LpgStore::new().unwrap();
        store.add_vector_index("Item", "embedding", vector());
        store.add_vector_index("Other", "embedding", vector());
        let observed = store.observe_vector_index("Item", "embedding").unwrap();
        assert!(store.validate_index_registration(&observed).is_ok());
        store.drop_index_if_unchanged(observed).unwrap();
        assert!(store.get_vector_index("Item", "embedding").is_none());
        assert!(store.get_vector_index("Other", "embedding").is_some());
    }

    #[cfg(feature = "text-index")]
    pub(super) fn text() -> std::sync::Arc<parking_lot::RwLock<crate::index::text::InvertedIndex>> {
        use crate::index::text::{BM25Config, InvertedIndex};
        std::sync::Arc::new(parking_lot::RwLock::new(InvertedIndex::new(
            BM25Config::default(),
        )))
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn text_conditional_drop_does_not_remove_another_key() {
        let store = LpgStore::new().unwrap();
        store.add_text_index("Doc", "body", text());
        store.add_text_index("Other", "body", text());
        let observed = store.observe_text_index("Doc", "body").unwrap();
        assert!(store.validate_index_registration(&observed).is_ok());
        store.drop_index_if_unchanged(observed).unwrap();
        assert!(store.get_text_index("Doc", "body").is_none());
        assert!(store.get_text_index("Other", "body").is_some());
    }
}
