//! Leapfrog iteration for Ring Index.
//!
//! Provides complete multi-pattern joins using the Ring Index structure.
//! The query-local preparation pass canonicalizes D dictionary terms, inspects
//! the N Ring rows for each of P relations, and builds ordered matching leaves
//! (O(D + PN + sum(Mi log Mi)) time, O(D + sum(Mi)) memory). Enumeration is
//! streaming LFTJ plus emitted witness bags; the succinct direct-trie
//! preparation target remains future optimization.

use super::triple_ring::TripleRing;
use crate::graph::rdf::{Term, Triple, TriplePattern};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

/// Iterator over a single component of the Ring Index.
///
/// Efficiently iterates over triples filtered by a specific term binding.
#[derive(Debug)]
pub struct RingIterator<'a> {
    ring: &'a TripleRing,
    /// Current position in the sequence.
    pos: usize,
    /// End position (exclusive).
    end: usize,
    /// Component being iterated (0 = subject, 1 = predicate, 2 = object).
    component: u8,
    /// Bound term ID for filtering.
    bound_id: Option<u32>,
    /// Current rank within the bound term's occurrences.
    rank: usize,
    /// Total count of bound term.
    count: usize,
    /// Whether this is an "iterate all" iterator (vs bound search).
    iterate_all: bool,
}

impl<'a> RingIterator<'a> {
    /// Creates an iterator over all triples.
    pub fn all(ring: &'a TripleRing) -> Self {
        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 0,
            bound_id: None,
            rank: 0,
            count: ring.len(),
            iterate_all: true,
        }
    }

    /// Creates an iterator over triples with a specific subject.
    pub fn with_subject(ring: &'a TripleRing, subject: &Term) -> Self {
        let (bound_id, count) = if let Some(id) = ring.dictionary().get_id(subject) {
            let count = ring.subjects_wt().count(id as u64);
            (Some(id), count)
        } else {
            (None, 0)
        };

        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 0,
            bound_id,
            rank: 0,
            count,
            iterate_all: false,
        }
    }

    /// Creates an iterator over triples with a specific predicate.
    pub fn with_predicate(ring: &'a TripleRing, predicate: &Term) -> Self {
        let (bound_id, count) = if let Some(id) = ring.dictionary().get_id(predicate) {
            let count = ring.predicates_wt().count(id as u64);
            (Some(id), count)
        } else {
            (None, 0)
        };

        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 1,
            bound_id,
            rank: 0,
            count,
            iterate_all: false,
        }
    }

    /// Creates an iterator over triples with a specific object.
    pub fn with_object(ring: &'a TripleRing, object: &Term) -> Self {
        let (bound_id, count) = if let Some(id) = ring.dictionary().get_id(object) {
            let count = ring.objects_wt().count(id as u64);
            (Some(id), count)
        } else {
            (None, 0)
        };

        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 2,
            bound_id,
            rank: 0,
            count,
            iterate_all: false,
        }
    }

    /// Returns the current position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns whether there are more elements.
    #[must_use]
    pub fn has_next(&self) -> bool {
        if self.iterate_all {
            self.pos < self.end
        } else if self.bound_id.is_some() {
            self.rank < self.count
        } else {
            // Searching for a term that wasn't found
            false
        }
    }

    /// Returns the term ID at the current position for a given component.
    ///
    /// Used by the leapfrog algorithm to compare term IDs across iterators.
    #[must_use]
    pub fn current_term_id(&self, component: u8) -> Option<u32> {
        if self.pos >= self.end {
            return None;
        }
        let wt = match component {
            0 => self.ring.subjects_wt(),
            1 => self.ring.predicates_wt(),
            _ => self.ring.objects_wt(),
        };
        // reason: wavelet tree values are dictionary IDs, fit u32
        #[allow(clippy::cast_possible_truncation)]
        Some(wt.access(self.pos) as u32)
    }

    /// Seeks to the next position where the given component's term ID >= target_id.
    ///
    /// Returns true if such a position was found.
    pub fn seek_term(&mut self, component: u8, target_id: u32) -> bool {
        while self.pos < self.end {
            if let Some(current_id) = self.current_term_id(component)
                && current_id >= target_id
            {
                return true;
            }
            // Advance to next position
            if self.iterate_all {
                self.pos += 1;
            } else if let Some(bound) = self.bound_id {
                self.rank += 1;
                if !self.has_next() {
                    return false;
                }
                let wt = match self.component {
                    0 => self.ring.subjects_wt(),
                    1 => self.ring.predicates_wt(),
                    _ => self.ring.objects_wt(),
                };
                if let Some(next_pos) = wt.select(bound as u64, self.rank) {
                    self.pos = next_pos;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        false
    }

    /// Seeks to the first position >= target.
    ///
    /// For leapfrog join, this is the key operation.
    ///
    /// # Panics
    ///
    pub fn seek(&mut self, target: usize) {
        if self.iterate_all {
            // For iterate-all, just move position
            self.pos = target.min(self.end);
        } else if let Some(bound) = self.bound_id {
            // For bound iterators, we need to find the next occurrence >= target
            while self.has_next() {
                let wt = match self.component {
                    0 => self.ring.subjects_wt(),
                    1 => self.ring.predicates_wt(),
                    _ => self.ring.objects_wt(),
                };

                if let Some(next_pos) = wt.select(bound as u64, self.rank) {
                    if next_pos >= target {
                        self.pos = next_pos;
                        return;
                    }
                    self.rank += 1;
                } else {
                    break;
                }
            }
            // No more elements
            self.pos = self.end;
        }
        // If bound_id is None and not iterate_all, do nothing (term not found)
    }
}

impl Iterator for RingIterator<'_> {
    type Item = Triple;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        let pos = if self.iterate_all {
            // Iterate all triples
            let p = self.pos;
            self.pos += 1;
            p
        } else {
            let id = self.bound_id?;
            // Get next position for this term using wavelet tree select
            let wt = match self.component {
                0 => self.ring.subjects_wt(),
                1 => self.ring.predicates_wt(),
                _ => self.ring.objects_wt(),
            };

            let next_pos = wt.select(id as u64, self.rank)?;
            self.rank += 1;
            self.pos = next_pos + 1;
            next_pos
        };

        self.ring.get_spo(pos)
    }
}

/// A triple pattern annotated with variable names for each position.
///
/// Variable positions use `Some("var_name")`, bound positions use `None`.
#[derive(Debug, Clone)]
pub struct AnnotatedPattern {
    /// The underlying triple pattern (bound terms).
    pub pattern: TriplePattern,
    /// Variable name for subject position (None if bound).
    pub subject_var: Option<String>,
    /// Variable name for predicate position (None if bound).
    pub predicate_var: Option<String>,
    /// Variable name for object position (None if bound).
    pub object_var: Option<String>,
}

/// Query-local canonical RDF identity used by the frozen Ring tries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalTermId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct VarId(u32);

impl VarId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy)]
struct Witness {
    exact_terms: [u32; 3],
}

#[derive(Debug)]
struct FrozenLeaf {
    keys: Vec<CanonicalTermId>,
    witnesses: Vec<Witness>,
}

/// One immutable trie relation. `leaves` are lexicographically ordered by the
/// relation's unique variables in the query-global variable order.
#[derive(Debug)]
struct FrozenTrie {
    variables: Vec<VarId>,
    leaves: Vec<FrozenLeaf>,
}

impl FrozenTrie {
    fn variable_position(&self, variable: VarId) -> Option<usize> {
        self.variables.binary_search(&variable).ok()
    }

    fn prefix_range(
        &self,
        local_position: usize,
        bindings: &[Option<CanonicalTermId>],
    ) -> Option<(usize, usize)> {
        let prefix = self.variables[..local_position]
            .iter()
            .map(|variable| bindings[variable.index()])
            .collect::<Option<Vec<_>>>()?;
        let start = self
            .leaves
            .partition_point(|leaf| leaf.keys[..local_position].cmp(&prefix) == Ordering::Less);
        let end = self
            .leaves
            .partition_point(|leaf| leaf.keys[..local_position].cmp(&prefix) != Ordering::Greater);
        Some((start, end))
    }

    fn cursor_at<'a>(
        &'a self,
        variable: VarId,
        bindings: &[Option<CanonicalTermId>],
        after: Option<CanonicalTermId>,
    ) -> Option<TrieValueCursor<'a>> {
        let local_position = self.variable_position(variable)?;
        let (start, end) = self.prefix_range(local_position, bindings)?;
        let offset = after.map_or(0, |bound| {
            self.leaves[start..end].partition_point(|leaf| leaf.keys[local_position] <= bound)
        });
        let cursor = TrieValueCursor {
            leaves: &self.leaves,
            local_position,
            position: start + offset,
            end,
        };
        cursor.current().map(|_| cursor)
    }

    fn leaf_for(&self, bindings: &[Option<CanonicalTermId>]) -> Option<usize> {
        let keys = self
            .variables
            .iter()
            .map(|variable| bindings[variable.index()])
            .collect::<Option<Vec<_>>>()?;
        self.leaves
            .binary_search_by(|leaf| leaf.keys.cmp(&keys))
            .ok()
    }
}

struct TrieValueCursor<'a> {
    leaves: &'a [FrozenLeaf],
    local_position: usize,
    position: usize,
    end: usize,
}

impl TrieValueCursor<'_> {
    fn current(&self) -> Option<CanonicalTermId> {
        (self.position < self.end).then(|| self.leaves[self.position].keys[self.local_position])
    }

    fn seek_at_least(&mut self, target: CanonicalTermId) -> Option<CanonicalTermId> {
        let offset = self.leaves[self.position..self.end]
            .partition_point(|leaf| leaf.keys[self.local_position] < target);
        self.position += offset;
        self.current()
    }
}

#[derive(Default)]
struct CanonicalTermTable {
    by_key: HashMap<String, CanonicalTermId>,
    exact_terms: Vec<Term>,
    exact_to_canonical: Vec<CanonicalTermId>,
}

impl CanonicalTermTable {
    fn intern(&mut self, key: String) -> Result<CanonicalTermId, RingJoinError> {
        if let Some(id) = self.by_key.get(&key) {
            return Ok(*id);
        }
        let id = CanonicalTermId(u32::try_from(self.by_key.len()).map_err(|_| {
            RingJoinError::InvalidPattern(
                "Ring query canonical identity domain exceeds u32 IDs".to_string(),
            )
        })?);
        self.by_key.insert(key, id);
        Ok(id)
    }

    fn from_ring<G: RingJoinGuard>(
        ring: &TripleRing,
        guard: &mut G,
    ) -> Result<Self, RingJoinError> {
        let mut table = Self::default();
        table.exact_terms.reserve(ring.num_terms());
        table.exact_to_canonical.reserve(ring.num_terms());
        let mut pending_work = 0;
        for index in 0..ring.num_terms() {
            let exact_id = u32::try_from(index).map_err(|_| {
                RingJoinError::InvalidPattern("Ring dictionary exceeds u32 IDs".to_string())
            })?;
            let term = ring.term_by_id(exact_id).ok_or_else(|| {
                RingJoinError::InvalidPattern(format!(
                    "Ring dictionary term {exact_id} is unavailable"
                ))
            })?;
            let canonical = table.intern(term.canonical_identity_key())?;
            table.exact_terms.push(term.clone());
            table.exact_to_canonical.push(canonical);
            pending_work += 1;
            if pending_work == RING_GUARD_BATCH {
                guard.checkpoint(pending_work, 0)?;
                pending_work = 0;
            }
        }
        if pending_work != 0 {
            guard.checkpoint(pending_work, 0)?;
        }
        Ok(table)
    }

    fn triple(&self, witness: Witness) -> Option<Triple> {
        let [subject, predicate, object] = witness.exact_terms;
        Some(Triple::new_unchecked(
            self.exact_terms.get(subject as usize)?.clone(),
            self.exact_terms.get(predicate as usize)?.clone(),
            self.exact_terms.get(object as usize)?.clone(),
        ))
    }
}

const RING_GUARD_BATCH: usize = 256;

/// Cooperative interruption reported by Ring preparation or enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingJoinError {
    /// The caller's work guard cancelled the operation.
    Cancelled,
    /// The annotated native pattern cannot be represented safely.
    InvalidPattern(String),
}

impl std::fmt::Display for RingJoinError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("Ring join cancelled"),
            Self::InvalidPattern(message) => {
                write!(formatter, "invalid Ring join pattern: {message}")
            }
        }
    }
}

impl std::error::Error for RingJoinError {}

/// Bounded-work callback used during native Ring preparation and enumeration.
pub trait RingJoinGuard {
    /// Checks whether work may continue before a partial result is exposed.
    ///
    /// # Errors
    ///
    /// Returns [`RingJoinError::Cancelled`] when the caller interrupts the
    /// operation, or another structured Ring error supplied by the guard.
    fn checkpoint(&mut self, work_delta: usize, emitted_delta: usize) -> Result<(), RingJoinError>;

    /// Optional safe cap on native solutions. Callers must prove that no
    /// ordering, distinct, or aggregate boundary needs rows beyond the cap.
    fn output_cap(&self) -> Option<usize> {
        None
    }
}

/// A guard with no cancellation and an optional caller-proven output cap.
#[derive(Debug, Default)]
pub struct UnboundedRingJoinGuard {
    output_cap: Option<usize>,
}

impl UnboundedRingJoinGuard {
    /// Creates an uninterruptible guard with no result cap.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a caller-proven native result cap.
    #[must_use]
    pub fn with_output_cap(mut self, output_cap: usize) -> Self {
        self.output_cap = Some(output_cap);
        self
    }
}

impl RingJoinGuard for UnboundedRingJoinGuard {
    fn checkpoint(
        &mut self,
        _work_delta: usize,
        _emitted_delta: usize,
    ) -> Result<(), RingJoinError> {
        Ok(())
    }

    fn output_cap(&self) -> Option<usize> {
        self.output_cap
    }
}

/// One exact native solution: canonical bindings plus one witness per pattern.
#[derive(Debug)]
pub struct RingSolution {
    canonical_bindings: Vec<CanonicalTermId>,
    witnesses: Vec<Triple>,
}

impl RingSolution {
    /// Returns the canonical binding tuple in query-global variable order.
    #[must_use]
    pub fn canonical_bindings(&self) -> &[CanonicalTermId] {
        &self.canonical_bindings
    }

    /// Returns exact physical witnesses in logical pattern order.
    #[must_use]
    pub fn witnesses(&self) -> &[Triple] {
        &self.witnesses
    }

    /// Consumes the solution and returns its exact witnesses.
    #[must_use]
    pub fn into_witnesses(self) -> Vec<Triple> {
        self.witnesses
    }
}

/// Immutable, query-local canonical tries prepared from a Ring snapshot.
pub struct PreparedRingJoin {
    relations: Vec<FrozenTrie>,
    relations_by_variable: Vec<Vec<usize>>,
    variable_count: usize,
    canonical_terms: CanonicalTermTable,
    empty: bool,
}

impl PreparedRingJoin {
    /// Compiles annotated patterns into canonical frozen tries.
    ///
    /// # Errors
    ///
    /// Returns cancellation from `guard` or a structured invalid-pattern error.
    pub fn prepare<G: RingJoinGuard>(
        ring: &TripleRing,
        annotated: &[AnnotatedPattern],
        guard: &mut G,
    ) -> Result<Self, RingJoinError> {
        if annotated.is_empty() {
            return Err(RingJoinError::InvalidPattern(
                "at least one pattern is required".to_string(),
            ));
        }
        let mut canonical_terms = CanonicalTermTable::from_ring(ring, guard)?;
        let mut variables = HashMap::<String, VarId>::new();
        let mut pending_pattern_work = 0;
        for pattern in annotated {
            for component in [
                pattern.subject_var.as_ref(),
                pattern.predicate_var.as_ref(),
                pattern.object_var.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if !variables.contains_key(component) {
                    let id = VarId(u32::try_from(variables.len()).map_err(|_| {
                        RingJoinError::InvalidPattern(
                            "Ring query variable domain exceeds u32 IDs".to_string(),
                        )
                    })?);
                    variables.insert(component.clone(), id);
                }
            }
            pending_pattern_work += 1;
            if pending_pattern_work == RING_GUARD_BATCH {
                guard.checkpoint(pending_pattern_work, 0)?;
                pending_pattern_work = 0;
            }
        }
        if pending_pattern_work != 0 {
            guard.checkpoint(pending_pattern_work, 0)?;
        }

        let mut relations = Vec::with_capacity(annotated.len());
        let mut pending_pattern_work = 0;
        for pattern in annotated {
            let component_names = [
                pattern.subject_var.as_ref(),
                pattern.predicate_var.as_ref(),
                pattern.object_var.as_ref(),
            ];
            let component_variables = component_names.map(|name| name.map(|name| variables[name]));
            let mut constants = [None; 3];
            for (component, term) in [
                pattern.pattern.subject.as_ref(),
                pattern.pattern.predicate.as_ref(),
                pattern.pattern.object.as_ref(),
            ]
            .into_iter()
            .enumerate()
            {
                if let Some(term) = term {
                    constants[component] =
                        Some(canonical_terms.intern(term.canonical_identity_key())?);
                }
            }
            for component in 0..3 {
                if component_variables[component].is_some() && constants[component].is_some() {
                    return Err(RingJoinError::InvalidPattern(
                        "one triple position cannot be both constant and variable".to_string(),
                    ));
                }
            }
            let mut relation_variables = component_variables
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            relation_variables.sort_unstable();
            relation_variables.dedup();

            let mut leaves_by_key = BTreeMap::<Vec<CanonicalTermId>, Vec<Witness>>::new();
            let mut pending_work = 0;
            for position in 0..ring.len() {
                pending_work += 1;
                if pending_work == RING_GUARD_BATCH {
                    guard.checkpoint(pending_work, 0)?;
                    pending_work = 0;
                }
                let exact_terms = ring.get_spo_ids(position).ok_or_else(|| {
                    RingJoinError::InvalidPattern(format!(
                        "Ring SPO witness {position} is unavailable"
                    ))
                })?;
                let mut canonical = [CanonicalTermId(0); 3];
                for (component, exact) in exact_terms.into_iter().enumerate() {
                    canonical[component] = canonical_terms
                        .exact_to_canonical
                        .get(exact as usize)
                        .copied()
                        .ok_or_else(|| {
                            let component_name = ["subject", "predicate", "object"][component];
                            RingJoinError::InvalidPattern(format!(
                                "Ring SPO witness {position} {component_name} references missing dictionary term ID {exact}"
                            ))
                        })?;
                }
                if constants.iter().enumerate().any(|(component, constant)| {
                    constant.is_some_and(|constant| constant != canonical[component])
                }) {
                    continue;
                }

                let mut scratch = Vec::<(VarId, CanonicalTermId)>::with_capacity(3);
                let mut repeated_conflict = false;
                for component in 0..3 {
                    let Some(variable) = component_variables[component] else {
                        continue;
                    };
                    if let Some((_, previous)) =
                        scratch.iter().find(|(existing, _)| *existing == variable)
                    {
                        if *previous != canonical[component] {
                            repeated_conflict = true;
                            break;
                        }
                    } else {
                        scratch.push((variable, canonical[component]));
                    }
                }
                if repeated_conflict {
                    continue;
                }
                scratch.sort_unstable_by_key(|(variable, _)| *variable);
                let keys = scratch.into_iter().map(|(_, key)| key).collect();
                leaves_by_key
                    .entry(keys)
                    .or_default()
                    .push(Witness { exact_terms });
            }
            if pending_work != 0 {
                guard.checkpoint(pending_work, 0)?;
            }
            let mut leaves = Vec::with_capacity(leaves_by_key.len());
            let mut pending_work = 0;
            for (keys, witnesses) in leaves_by_key {
                leaves.push(FrozenLeaf { keys, witnesses });
                pending_work += 1;
                if pending_work == RING_GUARD_BATCH {
                    guard.checkpoint(pending_work, 0)?;
                    pending_work = 0;
                }
            }
            if pending_work != 0 {
                guard.checkpoint(pending_work, 0)?;
            }
            relations.push(FrozenTrie {
                variables: relation_variables,
                leaves,
            });
            pending_pattern_work += 1;
            if pending_pattern_work == RING_GUARD_BATCH {
                guard.checkpoint(pending_pattern_work, 0)?;
                pending_pattern_work = 0;
            }
        }
        if pending_pattern_work != 0 {
            guard.checkpoint(pending_pattern_work, 0)?;
        }

        let mut relations_by_variable = vec![Vec::new(); variables.len()];
        let mut pending_edge_work = 0;
        for (relation_index, relation) in relations.iter().enumerate() {
            for variable in &relation.variables {
                relations_by_variable[variable.index()].push(relation_index);
                pending_edge_work += 1;
                if pending_edge_work == RING_GUARD_BATCH {
                    guard.checkpoint(pending_edge_work, 0)?;
                    pending_edge_work = 0;
                }
            }
        }
        if pending_edge_work != 0 {
            guard.checkpoint(pending_edge_work, 0)?;
        }
        let empty = relations.iter().any(|relation| relation.leaves.is_empty());
        guard.checkpoint(0, 0)?;
        Ok(Self {
            relations,
            relations_by_variable,
            variable_count: variables.len(),
            canonical_terms,
            empty,
        })
    }

    /// Creates fresh resumable state for this prepared join.
    #[must_use]
    pub fn new_state(&self) -> RingJoinState {
        RingJoinState {
            bindings: vec![None; self.variable_count],
            frames: if self.variable_count == 0 || self.empty {
                Vec::new()
            } else {
                vec![LevelFrame {
                    level: 0,
                    resume_after: None,
                }]
            },
            leaf_product: None,
            emitted: 0,
            done: self.empty,
        }
    }

    fn next_common_value<G: RingJoinGuard>(
        &self,
        variable: VarId,
        bindings: &[Option<CanonicalTermId>],
        after: Option<CanonicalTermId>,
        guard: &mut G,
    ) -> Result<Option<CanonicalTermId>, RingJoinError> {
        let Some(mut cursors) = self.relations_by_variable[variable.index()]
            .iter()
            .map(|relation| self.relations[*relation].cursor_at(variable, bindings, after))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let Some(mut target) = cursors.iter().filter_map(TrieValueCursor::current).max() else {
            return Ok(None);
        };
        loop {
            let mut converged = true;
            for cursor in &mut cursors {
                let Some(value) = cursor.seek_at_least(target) else {
                    return Ok(None);
                };
                if value > target {
                    target = value;
                    converged = false;
                }
            }
            guard.checkpoint(cursors.len().max(1), 0)?;
            if converged
                && cursors
                    .iter()
                    .all(|cursor| cursor.current() == Some(target))
            {
                return Ok(Some(target));
            }
        }
    }

    fn leaf_product(&self, bindings: &[Option<CanonicalTermId>]) -> Option<WitnessProductState> {
        let leaves = self
            .relations
            .iter()
            .map(|relation| relation.leaf_for(bindings))
            .collect::<Option<Vec<_>>>()?;
        Some(WitnessProductState {
            offsets: vec![0; leaves.len()],
            leaves,
        })
    }

    fn emit_product(&self, product: &WitnessProductState) -> Option<RingSolution> {
        let witnesses = product
            .leaves
            .iter()
            .zip(&product.offsets)
            .enumerate()
            .map(|(relation, (leaf, offset))| {
                let witness = self.relations[relation].leaves[*leaf].witnesses[*offset];
                self.canonical_terms.triple(witness)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(RingSolution {
            canonical_bindings: Vec::new(),
            witnesses,
        })
    }

    fn advance_product(&self, product: &mut WitnessProductState) -> bool {
        for relation in (0..product.offsets.len()).rev() {
            product.offsets[relation] += 1;
            let witness_count = self.relations[relation].leaves[product.leaves[relation]]
                .witnesses
                .len();
            if product.offsets[relation] < witness_count {
                return true;
            }
            product.offsets[relation] = 0;
        }
        false
    }

    /// Emits one complete native solution and preserves state for the next call.
    /// `state` must have been created by [`Self::new_state`] on this prepared
    /// join; states are query-local and must not be exchanged between joins.
    ///
    /// # Errors
    ///
    /// Returns cancellation from `guard` before exposing a partial solution.
    /// Any error makes `state` terminal because trie cursors and witness
    /// odometers advance incrementally; callers restart with [`Self::new_state`].
    pub fn next_solution<G: RingJoinGuard>(
        &self,
        state: &mut RingJoinState,
        guard: &mut G,
    ) -> Result<Option<RingSolution>, RingJoinError> {
        let result = self.next_solution_inner(state, guard);
        if result.is_err() {
            state.poison();
        }
        result
    }

    fn next_solution_inner<G: RingJoinGuard>(
        &self,
        state: &mut RingJoinState,
        guard: &mut G,
    ) -> Result<Option<RingSolution>, RingJoinError> {
        if state.done || guard.output_cap().is_some_and(|cap| state.emitted >= cap) {
            state.done = true;
            return Ok(None);
        }
        loop {
            if let Some(mut product) = state.leaf_product.take() {
                guard.checkpoint(1, 1)?;
                let mut solution = self.emit_product(&product).ok_or_else(|| {
                    RingJoinError::InvalidPattern(
                        "prepared Ring witness cannot be reconstructed".to_string(),
                    )
                })?;
                solution.canonical_bindings = state
                    .bindings
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        RingJoinError::InvalidPattern(
                            "complete Ring solution has an unbound variable".to_string(),
                        )
                    })?;
                state.emitted += 1;
                if self.advance_product(&mut product) {
                    state.leaf_product = Some(product);
                } else if self.variable_count == 0 {
                    state.done = true;
                } else {
                    state.bindings[self.variable_count - 1] = None;
                }
                return Ok(Some(solution));
            }

            if self.variable_count == 0 {
                state.leaf_product = self.leaf_product(&state.bindings);
                if state.leaf_product.is_none() {
                    state.done = true;
                    return Ok(None);
                }
                continue;
            }
            let Some(frame) = state.frames.last_mut() else {
                state.done = true;
                return Ok(None);
            };
            let level = frame.level;
            state.bindings[level] = None;
            let candidate = self.next_common_value(
                VarId(u32::try_from(level).map_err(|_| {
                    RingJoinError::InvalidPattern(
                        "Ring join depth exceeds u32 variable IDs".to_string(),
                    )
                })?),
                &state.bindings,
                frame.resume_after,
                guard,
            )?;
            let Some(candidate) = candidate else {
                state.frames.pop();
                if let Some(parent) = state.frames.last() {
                    state.bindings[parent.level] = None;
                }
                guard.checkpoint(1, 0)?;
                continue;
            };
            state.bindings[level] = Some(candidate);
            frame.resume_after = Some(candidate);
            if level + 1 == self.variable_count {
                state.leaf_product = self.leaf_product(&state.bindings);
                if state.leaf_product.is_none() {
                    state.bindings[level] = None;
                }
            } else {
                state.frames.push(LevelFrame {
                    level: level + 1,
                    resume_after: None,
                });
            }
        }
    }
}

/// Resumable state for [`PreparedRingJoin`].
pub struct RingJoinState {
    bindings: Vec<Option<CanonicalTermId>>,
    frames: Vec<LevelFrame>,
    leaf_product: Option<WitnessProductState>,
    emitted: usize,
    done: bool,
}

impl RingJoinState {
    fn poison(&mut self) {
        self.bindings.fill(None);
        self.frames.clear();
        self.leaf_product = None;
        self.done = true;
    }
}

struct LevelFrame {
    level: usize,
    resume_after: Option<CanonicalTermId>,
}

struct WitnessProductState {
    leaves: Vec<usize>,
    offsets: Vec<usize>,
}

/// Streaming Leapfrog Triejoin wrapper over a single immutable Ring snapshot.
///
/// This convenience `Iterator` cannot expose structured preparation or guard
/// errors and therefore terminates on either. Call [`PreparedRingJoin::prepare`]
/// and [`PreparedRingJoin::next_solution`] when an error must reach the caller.
pub struct LeapfrogRing<'a> {
    ring: &'a TripleRing,
    patterns: Vec<TriplePattern>,
    annotated: Option<Vec<AnnotatedPattern>>,
    prepared: Option<PreparedRingJoin>,
    state: Option<RingJoinState>,
    guard: UnboundedRingJoinGuard,
    exhausted: bool,
}

impl<'a> LeapfrogRing<'a> {
    /// Creates a native cross-pattern join without named correlations.
    pub fn new(ring: &'a TripleRing, patterns: Vec<TriplePattern>) -> Self {
        let exhausted = patterns.is_empty() || ring.is_empty();
        Self {
            ring,
            patterns,
            annotated: None,
            prepared: None,
            state: None,
            guard: UnboundedRingJoinGuard::new(),
            exhausted,
        }
    }

    /// Creates a native join with explicit variable ownership per RDF position.
    pub fn with_variables(ring: &'a TripleRing, annotated: Vec<AnnotatedPattern>) -> Self {
        let exhausted = annotated.is_empty() || ring.is_empty();
        let patterns = annotated
            .iter()
            .map(|pattern| pattern.pattern.clone())
            .collect();
        Self {
            ring,
            patterns,
            annotated: Some(annotated),
            prepared: None,
            state: None,
            guard: UnboundedRingJoinGuard::new(),
            exhausted,
        }
    }

    /// Applies a caller-proven output cap to the streaming iterator.
    #[must_use]
    pub fn with_output_limit(mut self, output_limit: usize) -> Self {
        self.guard = UnboundedRingJoinGuard::new().with_output_cap(output_limit);
        self
    }

    /// Returns the patterns being joined.
    #[must_use]
    pub fn patterns(&self) -> &[TriplePattern] {
        &self.patterns
    }

    /// Returns whether the join is exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    fn prepare(&mut self) -> Result<(), RingJoinError> {
        if self.prepared.is_some() {
            return Ok(());
        }
        let annotated = self.annotated.clone().unwrap_or_else(|| {
            self.patterns
                .iter()
                .cloned()
                .map(|pattern| AnnotatedPattern {
                    pattern,
                    subject_var: None,
                    predicate_var: None,
                    object_var: None,
                })
                .collect()
        });
        let prepared = PreparedRingJoin::prepare(self.ring, &annotated, &mut self.guard)?;
        self.state = Some(prepared.new_state());
        self.prepared = Some(prepared);
        Ok(())
    }
}

impl Iterator for LeapfrogRing<'_> {
    type Item = Vec<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted || self.guard.output_cap() == Some(0) || self.prepare().is_err() {
            self.exhausted = true;
            return None;
        }
        let prepared = self.prepared.as_ref()?;
        let state = self.state.as_mut()?;
        match prepared.next_solution(state, &mut self.guard) {
            Ok(Some(solution)) => Some(solution.into_witnesses()),
            Ok(None) | Err(_) => {
                self.exhausted = true;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(Term::iri(s), Term::iri(p), Term::iri(o))
    }

    #[derive(Default)]
    struct CountingGuard {
        work: usize,
        emitted: usize,
        cancel_after_work: Option<usize>,
        output_cap: Option<usize>,
    }

    impl RingJoinGuard for CountingGuard {
        fn checkpoint(
            &mut self,
            work_delta: usize,
            emitted_delta: usize,
        ) -> Result<(), RingJoinError> {
            self.work += work_delta;
            if self
                .cancel_after_work
                .is_some_and(|limit| self.work > limit)
            {
                return Err(RingJoinError::Cancelled);
            }
            self.emitted += emitted_delta;
            Ok(())
        }

        fn output_cap(&self) -> Option<usize> {
            self.output_cap
        }
    }

    #[test]
    fn test_ring_iterator_all() {
        let triples = vec![make_triple("s1", "p1", "o1"), make_triple("s2", "p2", "o2")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::all(&ring);
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_with_subject() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("alix", "knows", "harm"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_subject(&ring, &Term::iri("alix"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_with_predicate() {
        let triples = vec![
            make_triple("s1", "type", "Person"),
            make_triple("s2", "type", "Place"),
            make_triple("s1", "name", "Alix"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_predicate(&ring, &Term::iri("type"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_not_found() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_subject(&ring, &Term::iri("nonexistent"));
        let results: Vec<Triple> = iter.collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_leapfrog_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let lf = LeapfrogRing::new(&ring, vec![]);
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_leapfrog_single_pattern() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        let result = lf.next();
        assert!(result.is_some());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject(), &Term::iri("alix"));
    }

    #[test]
    fn test_ring_iterator_with_object() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("harm", "knows", "gus"),
            make_triple("dave", "likes", "eve"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_object(&ring, &Term::iri("gus"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);

        // Verify all results have gus as object
        for triple in &results {
            assert_eq!(triple.object(), &Term::iri("gus"));
        }
    }

    #[test]
    fn test_ring_iterator_with_object_not_found() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_object(&ring, &Term::iri("nonexistent"));
        let results: Vec<Triple> = iter.collect();
        assert!(results.is_empty());
    }

    #[test]
    fn bound_ring_iterator_preserves_exact_symbol_occurrence_count() {
        let ring = TripleRing::from_triples(["EN", "en"].into_iter().map(|language| {
            Triple::new(
                Term::iri(format!("urn:{language}")),
                Term::iri("urn:p"),
                Term::lang_literal("x", language),
            )
        }));
        let exact = Term::lang_literal("x", "EN");
        let iter = RingIterator::with_object(&ring, &exact);

        assert!(iter.has_next());
        let rows = iter.collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].object(), &exact);
        assert_eq!(
            ring.count(&TriplePattern::with_object(exact)),
            2,
            "pattern count deliberately follows canonical RDF identity"
        );
    }

    #[test]
    fn test_ring_iterator_position() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s2", "p2", "o2"),
            make_triple("s3", "p3", "o3"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        assert_eq!(iter.position(), 0);

        iter.next();
        assert_eq!(iter.position(), 1);

        iter.next();
        assert_eq!(iter.position(), 2);
    }

    #[test]
    fn test_ring_iterator_seek_iterate_all() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s2", "p2", "o2"),
            make_triple("s3", "p3", "o3"),
            make_triple("s4", "p4", "o4"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        assert_eq!(iter.position(), 0);

        // Seek to position 2
        iter.seek(2);
        assert_eq!(iter.position(), 2);
        assert!(iter.has_next());

        // Continue iteration from position 2
        let remaining: Vec<Triple> = iter.collect();
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn test_ring_iterator_seek_past_end() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        iter.seek(100);

        // Should be clamped to end
        assert!(!iter.has_next());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ring_iterator_seek_bound() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("harm", "knows", "dave"),
            make_triple("alix", "likes", "eve"),
            make_triple("frank", "knows", "alix"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::with_subject(&ring, &Term::iri("alix"));

        // Verify initial state
        assert!(iter.has_next());

        // Seek should find next occurrence >= target
        iter.seek(1);

        // The iterator should still be usable
        let results: Vec<Triple> = iter.collect();
        // All remaining results should have alix as subject
        for triple in &results {
            assert_eq!(triple.subject(), &Term::iri("alix"));
        }
    }

    #[test]
    fn test_ring_iterator_seek_not_found_term() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::with_subject(&ring, &Term::iri("nonexistent"));

        // Seek on a term that doesn't exist should do nothing
        iter.seek(0);
        assert!(!iter.has_next());
    }

    #[test]
    fn test_ring_iterator_has_next_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());

        let iter = RingIterator::all(&ring);
        assert!(!iter.has_next());
    }

    #[test]
    fn test_leapfrog_patterns_accessor() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern1 = TriplePattern::with_subject(Term::iri("alix"));
        let pattern2 = TriplePattern::with_predicate(Term::iri("knows"));
        let lf = LeapfrogRing::new(&ring, vec![pattern1.clone(), pattern2.clone()]);

        let patterns = lf.patterns();
        assert_eq!(patterns.len(), 2);
    }

    #[test]
    fn test_leapfrog_multi_pattern() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
            make_triple("harm", "likes", "alix"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Create patterns that should both match
        let pattern1 = TriplePattern::with_subject(Term::iri("alix"));
        let pattern2 = TriplePattern::with_predicate(Term::iri("knows"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern1, pattern2]);

        let result = lf.next();
        assert!(result.is_some());
        let matched = result.unwrap();
        // Should have matched both patterns
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_leapfrog_no_match() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Pattern that doesn't match any triple
        let pattern = TriplePattern::with_subject(Term::iri("nonexistent"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        let result = lf.next();
        assert!(result.is_none());
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_leapfrog_exhausted_after_iteration() {
        let triples = vec![make_triple("alix", "knows", "gus")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        assert!(!lf.is_exhausted());
        let result = lf.next();
        assert!(result.is_some());

        // After consuming all results, next returns None and marks exhausted
        let second_result = lf.next();
        assert!(second_result.is_none());
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_leapfrog_empty_ring_with_patterns() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let lf = LeapfrogRing::new(&ring, vec![pattern]);

        // Should be exhausted immediately when ring is empty
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_ring_iterator_predicate_not_found() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::with_predicate(&ring, &Term::iri("nonexistent"));
        assert!(!iter.has_next());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ring_iterator_all_single_triple() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        assert!(iter.has_next());

        let triple = iter.next().unwrap();
        assert_eq!(triple.subject(), &Term::iri("s"));
        assert_eq!(triple.predicate(), &Term::iri("p"));
        assert_eq!(triple.object(), &Term::iri("o"));

        assert!(!iter.has_next());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ring_iterator_seek_to_zero() {
        let triples = vec![make_triple("s1", "p1", "o1"), make_triple("s2", "p2", "o2")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        iter.seek(0);
        assert_eq!(iter.position(), 0);

        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_leapfrog_shared_subject() {
        // (?x knows bob) AND (?x knows harm) -> find subjects knowing both
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("alix", "knows", "harm"),
            make_triple("dave", "knows", "bob"),
            make_triple("eve", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("bob")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("harm")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
        ];

        let lf = LeapfrogRing::with_variables(&ring, annotated);
        let results: Vec<Vec<Triple>> = lf.collect();

        // Only alix knows both bob and harm
        assert_eq!(results.len(), 1);
        assert_eq!(results[0][0].subject(), &Term::iri("alix"));
        assert_eq!(results[0][1].subject(), &Term::iri("alix"));
    }

    #[test]
    fn test_leapfrog_triangle() {
        // (?x knows ?y) AND (?y knows ?z) AND (?z knows ?x) -> find triangles
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("bob", "knows", "harm"),
            make_triple("harm", "knows", "alix"),
            make_triple("dave", "knows", "eve"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: None,
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: Some("y".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: None,
                },
                subject_var: Some("y".to_string()),
                predicate_var: None,
                object_var: Some("z".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: None,
                },
                subject_var: Some("z".to_string()),
                predicate_var: None,
                object_var: Some("x".to_string()),
            },
        ];

        let lf = LeapfrogRing::with_variables(&ring, annotated);
        let results: Vec<Vec<Triple>> = lf.collect();

        // Should find the triangle in 3 rotations: alix->bob->harm->alix
        assert_eq!(results.len(), 3, "Expected three rotations of the triangle");
        assert_eq!(results[0].len(), 3);
    }

    #[test]
    fn leapfrog_backtracks_past_a_dead_continuation_and_emits_every_closure() {
        let triples = vec![
            make_triple("urn:a", "urn:p", "urn:b"),
            make_triple("urn:b", "urn:q", "urn:c0-dead"),
            make_triple("urn:b", "urn:q", "urn:c1-good"),
            make_triple("urn:b", "urn:q", "urn:c2-good"),
            make_triple("urn:c1-good", "urn:r", "urn:a"),
            make_triple("urn:c2-good", "urn:r", "urn:a"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:p")),
                    object: None,
                },
                subject_var: Some("a".to_string()),
                predicate_var: None,
                object_var: Some("b".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:q")),
                    object: None,
                },
                subject_var: Some("b".to_string()),
                predicate_var: None,
                object_var: Some("c".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:r")),
                    object: None,
                },
                subject_var: Some("c".to_string()),
                predicate_var: None,
                object_var: Some("a".to_string()),
            },
        ];

        let results = LeapfrogRing::with_variables(&ring, annotated).collect::<Vec<_>>();
        let mut closing_subjects = results
            .iter()
            .map(|row| row[2].subject().to_string())
            .collect::<Vec<_>>();
        closing_subjects.sort();

        assert_eq!(closing_subjects, ["<urn:c1-good>", "<urn:c2-good>"]);
    }

    #[test]
    fn leapfrog_preserves_bag_multiplicity_across_independent_closing_choices() {
        let triples = vec![
            make_triple("urn:a", "urn:p", "urn:b"),
            make_triple("urn:b", "urn:q", "urn:c1"),
            make_triple("urn:b", "urn:q", "urn:c2"),
            make_triple("urn:c1", "urn:r1", "urn:a"),
            make_triple("urn:c1", "urn:r2", "urn:a"),
            make_triple("urn:c2", "urn:r1", "urn:a"),
            make_triple("urn:c2", "urn:r2", "urn:a"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:p")),
                    object: None,
                },
                subject_var: Some("a".to_string()),
                predicate_var: None,
                object_var: Some("b".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:q")),
                    object: None,
                },
                subject_var: Some("b".to_string()),
                predicate_var: None,
                object_var: Some("c".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern::any(),
                subject_var: Some("c".to_string()),
                predicate_var: Some("closing_predicate".to_string()),
                object_var: Some("a".to_string()),
            },
        ];

        let results = LeapfrogRing::with_variables(&ring, annotated).collect::<Vec<_>>();

        assert_eq!(
            results.len(),
            4,
            "two-by-two continuations are four mappings"
        );
        assert!(
            results
                .iter()
                .all(|row| row[0].subject() == &Term::iri("urn:a"))
        );
    }

    #[test]
    fn leapfrog_emits_the_cartesian_product_of_canonical_leaf_witnesses() {
        let ring = TripleRing::from_triples(
            [
                Triple::new(
                    Term::iri("urn:p1"),
                    Term::iri("urn:p"),
                    Term::lang_literal("x", "EN"),
                ),
                Triple::new(
                    Term::iri("urn:p2"),
                    Term::iri("urn:p"),
                    Term::lang_literal("x", "en"),
                ),
                Triple::new(
                    Term::iri("urn:q1"),
                    Term::iri("urn:q"),
                    Term::lang_literal("x", "EN"),
                ),
                Triple::new(
                    Term::iri("urn:q2"),
                    Term::iri("urn:q"),
                    Term::lang_literal("x", "en"),
                ),
                Triple::new(
                    Term::iri("urn:q3"),
                    Term::iri("urn:q"),
                    Term::lang_literal("x", "En"),
                ),
            ]
            .into_iter(),
        );
        let annotated = ["urn:p", "urn:q"]
            .into_iter()
            .map(|predicate| AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri(predicate)),
                    object: None,
                },
                subject_var: None,
                predicate_var: None,
                object_var: Some("term".to_string()),
            })
            .collect();

        assert_eq!(
            LeapfrogRing::with_variables(&ring, annotated).count(),
            6,
            "two witnesses times three canonical-equal witnesses is a six-row bag"
        );
    }

    #[test]
    fn leapfrog_enforces_repeated_variables_within_each_triple_pattern() {
        let triples = vec![
            make_triple("urn:self", "urn:edge", "urn:self"),
            make_triple("urn:left", "urn:edge", "urn:right"),
            make_triple("urn:sp", "urn:sp", "urn:o"),
            make_triple("urn:s", "urn:po", "urn:po"),
            make_triple("urn:all", "urn:all", "urn:all"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let cases = [
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:edge")),
                    object: None,
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: Some("x".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: None,
                    object: Some(Term::iri("urn:o")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: Some("x".to_string()),
                object_var: None,
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: Some(Term::iri("urn:s")),
                    predicate: None,
                    object: None,
                },
                subject_var: None,
                predicate_var: Some("x".to_string()),
                object_var: Some("x".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern::any(),
                subject_var: Some("x".to_string()),
                predicate_var: Some("x".to_string()),
                object_var: Some("x".to_string()),
            },
        ];

        for pattern in cases {
            let results = LeapfrogRing::with_variables(&ring, vec![pattern]).collect::<Vec<_>>();
            assert_eq!(
                results.len(),
                1,
                "each repeated-position shape has one match"
            );
        }
    }

    #[test]
    fn leapfrog_enforces_a_repeated_position_against_an_existing_binding() {
        let ring = TripleRing::from_triples(
            [
                make_triple("urn:seed", "urn:bind", "urn:self"),
                make_triple("urn:self", "urn:loop", "urn:self"),
                make_triple("urn:self", "urn:loop", "urn:other"),
            ]
            .into_iter(),
        );
        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: Some(Term::iri("urn:seed")),
                    predicate: Some(Term::iri("urn:bind")),
                    object: None,
                },
                subject_var: None,
                predicate_var: None,
                object_var: Some("x".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("urn:loop")),
                    object: None,
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: Some("x".to_string()),
            },
        ];

        assert_eq!(LeapfrogRing::with_variables(&ring, annotated).count(), 1);
    }

    #[test]
    fn leapfrog_uses_canonical_rdf_identity_without_collapsing_term_kinds() {
        let language_ring = TripleRing::from_triples(
            [
                Triple::new(
                    Term::iri("urn:left"),
                    Term::iri("urn:p"),
                    Term::lang_literal("x", "EN"),
                ),
                Triple::new(
                    Term::iri("urn:right"),
                    Term::iri("urn:q"),
                    Term::lang_literal("x", "en"),
                ),
            ]
            .into_iter(),
        );
        let join_on_object = |left_predicate: &str, right_predicate: &str| {
            vec![
                AnnotatedPattern {
                    pattern: TriplePattern {
                        subject: None,
                        predicate: Some(Term::iri(left_predicate)),
                        object: None,
                    },
                    subject_var: Some("left".to_string()),
                    predicate_var: None,
                    object_var: Some("term".to_string()),
                },
                AnnotatedPattern {
                    pattern: TriplePattern {
                        subject: None,
                        predicate: Some(Term::iri(right_predicate)),
                        object: None,
                    },
                    subject_var: Some("right".to_string()),
                    predicate_var: None,
                    object_var: Some("term".to_string()),
                },
            ]
        };
        assert_eq!(
            LeapfrogRing::with_variables(&language_ring, join_on_object("urn:p", "urn:q")).count(),
            1,
            "language-tag case is one RDF identity"
        );

        let collision_ring = TripleRing::from_triples(
            [
                Triple::new(
                    Term::iri("urn:left"),
                    Term::iri("urn:p"),
                    Term::iri("urn:value"),
                ),
                Triple::new(
                    Term::iri("urn:right"),
                    Term::iri("urn:q"),
                    Term::literal("urn:value"),
                ),
            ]
            .into_iter(),
        );
        assert_eq!(
            LeapfrogRing::with_variables(&collision_ring, join_on_object("urn:p", "urn:q")).count(),
            0,
            "an IRI and same-spelled literal are distinct RDF identities"
        );

        let string_ring = TripleRing::from_triples(
            [
                Triple::new(
                    Term::iri("urn:left"),
                    Term::iri("urn:p"),
                    Term::literal("x"),
                ),
                Triple::new(
                    Term::iri("urn:right"),
                    Term::iri("urn:q"),
                    Term::typed_literal("x", crate::graph::rdf::Literal::XSD_STRING),
                ),
            ]
            .into_iter(),
        );
        assert_eq!(
            LeapfrogRing::with_variables(&string_ring, join_on_object("urn:p", "urn:q")).count(),
            1,
            "plain and explicit xsd:string literals have one RDF identity"
        );
    }

    #[test]
    fn leapfrog_never_cross_joins_same_spelled_iri_and_literal_identities() {
        let mut triples = Vec::new();
        for (subject, predicate) in [
            ("urn:s1", "urn:p"),
            ("urn:s2", "urn:q"),
            ("urn:s3", "urn:r"),
        ] {
            triples.push(Triple::new(
                Term::iri(subject),
                Term::iri(predicate),
                Term::iri("urn:x"),
            ));
            triples.push(Triple::new(
                Term::iri(subject),
                Term::iri(predicate),
                Term::literal("urn:x"),
            ));
        }
        let ring = TripleRing::from_triples(triples.into_iter());
        let annotated = ["urn:p", "urn:q", "urn:r"]
            .into_iter()
            .map(|predicate| AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri(predicate)),
                    object: None,
                },
                subject_var: None,
                predicate_var: None,
                object_var: Some("term".to_string()),
            })
            .collect();
        let results = LeapfrogRing::with_variables(&ring, annotated).collect::<Vec<_>>();

        assert_eq!(results.len(), 2, "one all-IRI row and one all-literal row");
        assert!(results.iter().any(|row| row[0].object().is_iri()));
        assert!(results.iter().any(|row| row[0].object().is_literal()));
    }

    #[test]
    fn leapfrog_is_invariant_to_dictionary_and_pattern_order() {
        use std::collections::BTreeMap;

        let edges = [
            ("urn:a", "urn:p", "urn:b"),
            ("urn:d", "urn:p", "urn:b"),
            ("urn:b", "urn:q", "urn:c"),
            ("urn:b", "urn:q", "urn:d"),
            ("urn:c", "urn:r", "urn:a"),
            ("urn:d", "urn:r", "urn:a"),
            ("urn:c", "urn:r", "urn:d"),
        ];
        let pattern_orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut dictionary_orders = vec![(0..edges.len()).collect::<Vec<_>>()];
        dictionary_orders.push((0..edges.len()).rev().collect());
        for rotation in 1..edges.len() {
            dictionary_orders.push(
                (0..edges.len())
                    .cycle()
                    .skip(rotation)
                    .take(edges.len())
                    .collect(),
            );
        }
        let base_patterns = [
            ("a", "urn:p", "b"),
            ("b", "urn:q", "c"),
            ("c", "urn:r", "a"),
        ];
        let expected = BTreeMap::from([
            (
                vec!["<urn:a>", "<urn:b>", "<urn:c>"]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
                1,
            ),
            (
                vec!["<urn:a>", "<urn:b>", "<urn:d>"]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
                1,
            ),
            (
                vec!["<urn:d>", "<urn:b>", "<urn:c>"]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
                1,
            ),
        ]);

        for dictionary_order in dictionary_orders {
            let ring = TripleRing::from_triples(dictionary_order.into_iter().map(|edge| {
                let (subject, predicate, object) = edges[edge];
                make_triple(subject, predicate, object)
            }));
            for pattern_order in pattern_orders {
                let annotated = pattern_order
                    .into_iter()
                    .map(|pattern| {
                        let (subject, predicate, object) = base_patterns[pattern];
                        AnnotatedPattern {
                            pattern: TriplePattern::with_predicate(Term::iri(predicate)),
                            subject_var: Some(subject.to_string()),
                            predicate_var: None,
                            object_var: Some(object.to_string()),
                        }
                    })
                    .collect::<Vec<_>>();
                let mut bag = BTreeMap::<Vec<String>, usize>::new();
                for witnesses in LeapfrogRing::with_variables(&ring, annotated.clone()) {
                    let mut bindings = HashMap::<&str, &Term>::new();
                    for (pattern, witness) in annotated.iter().zip(&witnesses) {
                        for (variable, term) in [
                            (pattern.subject_var.as_deref(), witness.subject()),
                            (pattern.predicate_var.as_deref(), witness.predicate()),
                            (pattern.object_var.as_deref(), witness.object()),
                        ] {
                            if let Some(variable) = variable {
                                bindings.entry(variable).or_insert(term);
                            }
                        }
                    }
                    let tuple = ["a", "b", "c"]
                        .map(|variable| bindings[variable].to_string())
                        .to_vec();
                    *bag.entry(tuple).or_default() += 1;
                }
                assert_eq!(bag, expected);
            }
        }
    }

    #[test]
    fn leapfrog_preserves_rdf_set_semantics_for_exact_duplicate_input() {
        let duplicate = make_triple("urn:a", "urn:p", "urn:b");
        let ring = TripleRing::from_triples([duplicate.clone(), duplicate].into_iter());
        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern::with_predicate(Term::iri("urn:p")),
                subject_var: Some("s".to_string()),
                predicate_var: None,
                object_var: Some("o".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern::with_predicate(Term::iri("urn:p")),
                subject_var: Some("s".to_string()),
                predicate_var: None,
                object_var: Some("o".to_string()),
            },
        ];

        assert_eq!(LeapfrogRing::with_variables(&ring, annotated).count(), 1);
    }

    #[test]
    fn prepared_ring_join_cancellation_is_bounded_and_never_emits_a_partial_row() {
        let ring =
            TripleRing::from_triples((0..600).map(|index| {
                make_triple(&format!("urn:s{index}"), "urn:p", &format!("urn:o{index}"))
            }));
        let annotated = vec![AnnotatedPattern {
            pattern: TriplePattern::with_predicate(Term::iri("urn:p")),
            subject_var: Some("s".to_string()),
            predicate_var: None,
            object_var: Some("o".to_string()),
        }];
        let mut preparation_guard = CountingGuard {
            cancel_after_work: Some(8),
            ..CountingGuard::default()
        };
        assert!(matches!(
            PreparedRingJoin::prepare(&ring, &annotated, &mut preparation_guard),
            Err(RingJoinError::Cancelled)
        ));
        assert_eq!(preparation_guard.emitted, 0);

        let mut unbounded = UnboundedRingJoinGuard::new();
        let prepared = PreparedRingJoin::prepare(&ring, &annotated, &mut unbounded).unwrap();
        let mut state = prepared.new_state();
        let mut enumeration_guard = CountingGuard {
            cancel_after_work: Some(0),
            ..CountingGuard::default()
        };
        assert!(matches!(
            prepared.next_solution(&mut state, &mut enumeration_guard),
            Err(RingJoinError::Cancelled)
        ));
        assert_eq!(enumeration_guard.emitted, 0);
        assert!(
            state.done,
            "cancellation poisons the partially advanced state"
        );
        assert!(state.frames.is_empty());
        assert!(state.leaf_product.is_none());
        assert!(
            prepared
                .next_solution(&mut state, &mut UnboundedRingJoinGuard::new())
                .unwrap()
                .is_none(),
            "a cancelled state is terminal; callers restart from new_state()"
        );
        assert!(
            prepared
                .next_solution(
                    &mut prepared.new_state(),
                    &mut UnboundedRingJoinGuard::new(),
                )
                .unwrap()
                .is_some(),
            "fresh state restarts a cancelled enumeration"
        );
    }

    #[test]
    fn prepared_ring_join_can_cancel_dense_leaf_finalization_after_scans() {
        let ring =
            TripleRing::from_triples((0..600).map(|index| {
                make_triple(&format!("urn:s{index}"), "urn:p", &format!("urn:o{index}"))
            }));
        let annotated = vec![AnnotatedPattern {
            pattern: TriplePattern::with_predicate(Term::iri("urn:p")),
            subject_var: Some("s".to_string()),
            predicate_var: None,
            object_var: Some("o".to_string()),
        }];
        let scan_work = ring.num_terms() + ring.len();
        let mut guard = CountingGuard {
            cancel_after_work: Some(scan_work),
            ..CountingGuard::default()
        };

        assert!(matches!(
            PreparedRingJoin::prepare(&ring, &annotated, &mut guard),
            Err(RingJoinError::Cancelled)
        ));
        assert!(
            guard.work > scan_work,
            "guard must interrupt ordered leaf finalization after dictionary and row scans"
        );
        assert_eq!(guard.emitted, 0);
    }

    #[test]
    fn prepared_ring_join_checks_large_pattern_and_relation_edge_compilation() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let annotated = (0..600)
            .map(|index| AnnotatedPattern {
                pattern: TriplePattern::with_predicate(Term::iri(format!("urn:p{index}"))),
                subject_var: Some(format!("s{index}")),
                predicate_var: None,
                object_var: Some(format!("o{index}")),
            })
            .collect::<Vec<_>>();

        let mut variable_guard = CountingGuard {
            cancel_after_work: Some(0),
            ..CountingGuard::default()
        };
        assert!(matches!(
            PreparedRingJoin::prepare(&ring, &annotated, &mut variable_guard),
            Err(RingJoinError::Cancelled)
        ));
        assert_eq!(variable_guard.work, RING_GUARD_BATCH);

        // Two complete O(P) passes account for 1,200 units. The next guarded
        // batch is relation-to-variable edge construction (two edges/pattern).
        let mut edge_guard = CountingGuard {
            cancel_after_work: Some(1_200),
            ..CountingGuard::default()
        };
        assert!(matches!(
            PreparedRingJoin::prepare(&ring, &annotated, &mut edge_guard),
            Err(RingJoinError::Cancelled)
        ));
        assert_eq!(edge_guard.work, 1_200 + RING_GUARD_BATCH);
        assert_eq!(edge_guard.emitted, 0);
    }

    #[test]
    fn ring_reconstruction_rejects_a_witness_outside_the_dictionary() {
        use super::super::{PackedTermDictionary, SuccinctPermutation};
        use crate::codec::succinct::WaveletTree;
        use crate::index::ring::triple_ring::{TermDictionary, TripleRingInvariantError};

        let mut dictionary = TermDictionary::new();
        dictionary.get_or_insert(Term::iri("urn:only-term"));
        let packed = PackedTermDictionary::from_term_dict(&dictionary)
            .expect("test dictionary is representable");
        let invalid_component = WaveletTree::new(&[7]);
        let Err(error) = TripleRing::from_packed_parts(
            packed,
            1,
            invalid_component.clone(),
            invalid_component.clone(),
            invalid_component,
            SuccinctPermutation::new(&[0]),
            SuccinctPermutation::new(&[0]),
        ) else {
            panic!("malformed dictionary reference must fail during reconstruction");
        };
        assert_eq!(
            error,
            TripleRingInvariantError::DictionaryIdOutOfRange {
                component: "subjects",
                id: 7,
                dictionary_len: 1,
            }
        );
    }

    #[test]
    fn prepared_ring_join_safe_cap_stops_before_enumerating_the_full_bag() {
        let mut triples = vec![make_triple("urn:a", "urn:p", "urn:b")];
        for index in 0..100 {
            let closing = format!("urn:c{index:03}");
            triples.push(make_triple("urn:b", "urn:q", &closing));
            triples.push(make_triple(&closing, "urn:r", "urn:a"));
        }
        let ring = TripleRing::from_triples(triples.into_iter());
        let annotated = [
            ("a", "urn:p", "b"),
            ("b", "urn:q", "c"),
            ("c", "urn:r", "a"),
        ]
        .into_iter()
        .map(|(subject, predicate, object)| AnnotatedPattern {
            pattern: TriplePattern::with_predicate(Term::iri(predicate)),
            subject_var: Some(subject.to_string()),
            predicate_var: None,
            object_var: Some(object.to_string()),
        })
        .collect::<Vec<_>>();
        let mut preparation_guard = UnboundedRingJoinGuard::new();
        let prepared =
            PreparedRingJoin::prepare(&ring, &annotated, &mut preparation_guard).unwrap();
        let mut state = prepared.new_state();
        let mut enumeration_guard = CountingGuard {
            output_cap: Some(1),
            ..CountingGuard::default()
        };

        assert!(
            prepared
                .next_solution(&mut state, &mut enumeration_guard)
                .unwrap()
                .is_some()
        );
        assert!(
            prepared
                .next_solution(&mut state, &mut enumeration_guard)
                .unwrap()
                .is_none()
        );
        assert_eq!(enumeration_guard.emitted, 1);
        assert!(
            enumeration_guard.work < 100,
            "safe LIMIT 1 must not enumerate all 100 closing branches: {}",
            enumeration_guard.work
        );
    }

    #[test]
    fn prepared_ring_join_fresh_state_replays_the_same_bag() {
        let ring = TripleRing::from_triples(
            [
                make_triple("urn:a", "urn:p", "urn:b"),
                make_triple("urn:b", "urn:q", "urn:c1"),
                make_triple("urn:b", "urn:q", "urn:c2"),
                make_triple("urn:c1", "urn:r", "urn:a"),
                make_triple("urn:c2", "urn:r", "urn:a"),
            ]
            .into_iter(),
        );
        let annotated = [
            ("a", "urn:p", "b"),
            ("b", "urn:q", "c"),
            ("c", "urn:r", "a"),
        ]
        .into_iter()
        .map(|(subject, predicate, object)| AnnotatedPattern {
            pattern: TriplePattern::with_predicate(Term::iri(predicate)),
            subject_var: Some(subject.to_string()),
            predicate_var: None,
            object_var: Some(object.to_string()),
        })
        .collect::<Vec<_>>();
        let mut preparation_guard = UnboundedRingJoinGuard::new();
        let prepared =
            PreparedRingJoin::prepare(&ring, &annotated, &mut preparation_guard).unwrap();
        let collect = |mut state: RingJoinState| {
            let mut guard = UnboundedRingJoinGuard::new();
            let mut rows = Vec::new();
            while let Some(solution) = prepared.next_solution(&mut state, &mut guard).unwrap() {
                rows.push(solution.witnesses()[1].object().canonical_identity_key());
            }
            rows
        };

        assert_eq!(collect(prepared.new_state()), collect(prepared.new_state()));
    }

    #[test]
    fn test_leapfrog_empty_intersection() {
        // (?x knows bob) AND (?x knows dave) -> no one knows both
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("harm", "knows", "dave"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("bob")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("dave")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
        ];

        let lf = LeapfrogRing::with_variables(&ring, annotated);
        let results: Vec<Vec<Triple>> = lf.collect();

        assert!(results.is_empty(), "Expected no matches");
    }

    #[test]
    fn test_ring_iterator_current_term_id() {
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("harm", "likes", "dave"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::all(&ring);
        // Should return Some for valid positions
        let id = iter.current_term_id(0);
        assert!(id.is_some());
    }

    #[test]
    fn test_ring_iterator_current_term_id_past_end() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        iter.next(); // consume the only triple
        let id = iter.current_term_id(0);
        assert!(id.is_none());
    }
}
