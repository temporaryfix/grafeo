//! RDF-local ordering.
//!
//! The generic sorter intentionally knows only the native scalar value model.
//! SPARQL results can carry arbitrary-precision numeric literals, so RDF plans
//! need a comparator that applies numeric promotion before falling back to the
//! generic total order.

use std::cmp::Ordering;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::operators::value_utils::compare_values_total;
use grafeo_core::execution::operators::{
    AccountedValueComparator, NullOrder, Operator, OperatorResult, SemanticComparisonError,
    SortKey, SortOperator,
};
use grafeo_core::execution::{QueryResourceContext, QueryResourceContextError};
use grafeo_core::graph::rdf::{Literal, Term};

use super::{
    compare_xsd_dates, compare_xsd_datetimes, decode_tagged_rdf_filter_term, numeric::RdfNumeric,
};

/// RDF semantics remain local; physical ownership and execution use the
/// resource-qualified native sorter, including hidden input key columns.
pub(super) struct RdfSortOperator {
    inner: SortOperator,
}

impl RdfSortOperator {
    pub(super) fn new(
        child: Box<dyn Operator>,
        sort_keys: Vec<SortKey>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            inner: SortOperator::new(child, sort_keys, output_schema)
                .with_semantic_comparator(Arc::new(RdfValueComparator)),
        }
    }
}

impl Operator for RdfSortOperator {
    fn next(&mut self) -> OperatorResult {
        self.inner.next()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        self.inner.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfInMemorySort"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct RdfValueComparator;

impl AccountedValueComparator for RdfValueComparator {
    fn scratch_bytes(
        &self,
        left: Option<&Value>,
        right: Option<&Value>,
    ) -> Result<usize, SemanticComparisonError> {
        let retained = |value: Option<&Value>| {
            value.map_or(Some(0), Value::retained_size_bytes).ok_or(
                SemanticComparisonError::Resource(
                    grafeo_common::memory::buffer::MemoryGrantError::ArithmeticOverflow {
                        current_bytes: usize::MAX,
                        additional_bytes: 1,
                    },
                ),
            )
        };
        rdf_comparison_scratch(retained(left)?, retained(right)?)
    }

    fn compare(
        &self,
        left: Option<&Value>,
        right: Option<&Value>,
    ) -> Result<Ordering, SemanticComparisonError> {
        Ok(rdf_compare_borrowed_values_with_nulls(
            left,
            right,
            NullOrder::NullsFirst,
        ))
    }
}

/// Bound temporary allocations of the pinned RDF comparison implementation.
///
/// N-Triples unescaping never expands its input; lexical validation, term
/// ownership, exact date/year normalization and their overlapping buffers are
/// linear in input bytes. Integer/decimal validation forbids exponent notation
/// before BigDecimal parsing, so coefficient and scale are lexical-length
/// bounded. bigdecimal 0.4.10 compares scaled coefficients by digit counts and
/// radix digits, without allocating exponent-sized zero padding. Float/double
/// lexical exponents first become IEEE values; their exact decimal conversion
/// has at most 1074 fractional digits, covered by the fixed allowance.
///
/// The 256 multiplier reserves four overlapping 64-byte-per-retained-byte
/// envelopes: decoded terms/lexical validation, numeric coefficients and
/// clones, radix conversion scratch, and fallback formatting. String escaping
/// expands one input byte by at most six; old/new growth backing and both Debug
/// strings fit the formatting envelope. Value Debug prints containers linearly
/// (no pretty-print indentation), and summarizes paths, vectors and counters.
/// Recursive retained measurement
/// includes every container entry and rejects nesting deeper than 256. No
/// comparison scratch or dynamic error payload survives the callback.
fn rdf_comparison_scratch(left: usize, right: usize) -> Result<usize, SemanticComparisonError> {
    left.checked_add(right)
        .and_then(|bytes| bytes.checked_mul(256))
        .and_then(|bytes| bytes.checked_add(64 << 10))
        .ok_or(SemanticComparisonError::Resource(
            grafeo_common::memory::buffer::MemoryGrantError::ArithmeticOverflow {
                current_bytes: left,
                additional_bytes: right,
            },
        ))
}

#[cfg(test)]
fn rdf_compare_values_with_nulls(
    left: &Option<Value>,
    right: &Option<Value>,
    null_order: NullOrder,
) -> Ordering {
    rdf_compare_borrowed_values_with_nulls(left.as_ref(), right.as_ref(), null_order)
}

fn rdf_compare_borrowed_values_with_nulls(
    left: Option<&Value>,
    right: Option<&Value>,
    null_order: NullOrder,
) -> Ordering {
    match (left, right) {
        (None | Some(Value::Null), None | Some(Value::Null)) => Ordering::Equal,
        (None | Some(Value::Null), _) => {
            if matches!(null_order, NullOrder::NullsFirst) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (_, None | Some(Value::Null)) => {
            if matches!(null_order, NullOrder::NullsFirst) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Some(left), Some(right)) => {
            if let (Some((_, left_term)), Some((_, right_term))) = (
                decode_tagged_rdf_filter_term(left),
                decode_tagged_rdf_filter_term(right),
            ) {
                return rdf_compare_terms(&left_term, &right_term);
            }
            match (RdfNumeric::from_value(left), RdfNumeric::from_value(right)) {
                (Some(left), Some(right)) => {
                    left.compare_for_order(&right).unwrap_or(Ordering::Equal)
                }
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => rdf_compare_values_total(left, right),
            }
        }
    }
}

pub(super) fn rdf_compare_terms(left: &Term, right: &Term) -> Ordering {
    let category = rdf_term_category(left).cmp(&rdf_term_category(right));
    if category != Ordering::Equal {
        return category;
    }

    match (left, right) {
        (Term::BlankNode(left), Term::BlankNode(right)) => left.id().cmp(right.id()),
        (Term::Iri(left), Term::Iri(right)) => left.as_str().cmp(right.as_str()),
        (Term::Literal(left), Term::Literal(right)) => {
            let literal_category = rdf_literal_category(left).cmp(&rdf_literal_category(right));
            if literal_category != Ordering::Equal {
                return literal_category;
            }
            if let (Some(left), Some(right)) = (numeric_literal(left), numeric_literal(right)) {
                return left.compare_for_order(&right).unwrap_or(Ordering::Equal);
            }
            match rdf_literal_category(left) {
                1 => left
                    .value()
                    .cmp(right.value())
                    .then_with(|| compare_languages(left.language(), right.language()))
                    .then_with(|| left.datatype().cmp(right.datatype())),
                2 => left
                    .as_boolean()
                    .expect("boolean category contains valid lexical forms")
                    .cmp(
                        &right
                            .as_boolean()
                            .expect("boolean category contains valid lexical forms"),
                    ),
                3 => compare_xsd_dates(left.value(), right.value())
                    .expect("date category contains valid lexical forms"),
                4 => compare_xsd_datetimes(left.value(), right.value())
                    .expect("dateTime category contains valid lexical forms"),
                _ => left
                    .datatype()
                    .cmp(right.datatype())
                    .then_with(|| compare_languages(left.language(), right.language()))
                    .then_with(|| left.value().cmp(right.value())),
            }
        }
        _ => Ordering::Equal,
    }
}

fn numeric_literal(literal: &Literal) -> Option<RdfNumeric> {
    RdfNumeric::from_literal(literal)
}

/// SPARQL leaves several cross-datatype literal pairs unordered. A blocking
/// sorter nevertheless requires one transitive total order, so place every
/// comparable value family in a stable class before applying its value order.
fn rdf_literal_category(literal: &Literal) -> u8 {
    if numeric_literal(literal).is_some() {
        return 0;
    }
    if literal.datatype() == Literal::XSD_STRING || literal.language().is_some() {
        return 1;
    }
    match literal.datatype() {
        Literal::XSD_BOOLEAN if literal.as_boolean().is_some() => 2,
        Literal::XSD_DATE if compare_xsd_dates(literal.value(), literal.value()).is_some() => 3,
        Literal::XSD_DATETIME
            if compare_xsd_datetimes(literal.value(), literal.value()).is_some() =>
        {
            4
        }
        _ => 5,
    }
}

fn compare_languages(left: Option<&str>, right: Option<&str>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left
            .bytes()
            .map(|byte| byte.to_ascii_lowercase())
            .cmp(right.bytes().map(|byte| byte.to_ascii_lowercase())),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    }
}

fn rdf_term_category(term: &Term) -> u8 {
    match term {
        Term::BlankNode(_) => 0,
        Term::Iri(_) => 1,
        Term::Literal(_) => 2,
        _ => u8::MAX,
    }
}

/// Deterministic extension ordering for RDF execution values. SPARQL leaves
/// some cross-category pairs unordered, while Rust's stable sort still
/// requires transitivity.
pub(super) fn rdf_compare_values_total(left: &Value, right: &Value) -> Ordering {
    let category = rdf_value_category(left).cmp(&rdf_value_category(right));
    if category != Ordering::Equal {
        return category;
    }
    let native = compare_values_total(left, right);
    if native != Ordering::Equal || left == right {
        return native;
    }
    format!("{left:?}").cmp(&format!("{right:?}"))
}

fn rdf_value_category(value: &Value) -> u8 {
    match value {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Int64(_) | Value::Float64(_) => 2,
        Value::String(_) => 3,
        Value::RdfLiteral { .. } => 4,
        Value::Date(_) => 5,
        Value::Time(_) => 6,
        Value::Timestamp(_) | Value::ZonedDatetime(_) => 7,
        Value::Duration(_) => 8,
        Value::Bytes(_) => 9,
        Value::List(_) => 10,
        Value::Map(_) => 11,
        Value::Vector(_) => 12,
        Value::Path { .. } => 13,
        Value::GCounter(_) | Value::OnCounter { .. } => 14,
        _ => u8::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::super::tagged_rdf_term;
    use super::*;

    #[test]
    fn qualified_comparator_checks_scratch_and_preserves_exact_numeric_order() {
        let comparator = RdfValueComparator;
        let left = Value::RdfLiteral {
            lexical: "9007199254740993".into(),
            datatype: Some(Literal::XSD_INTEGER.into()),
            language: None,
        };
        let right = Value::Float64(9007199254740992.0);
        assert!(comparator.scratch_bytes(Some(&left), Some(&right)).unwrap() > 64 << 10);
        assert_eq!(
            comparator.compare(Some(&left), Some(&right)).unwrap(),
            Ordering::Greater
        );
        assert_eq!(
            comparator.compare(None, Some(&right)).unwrap(),
            Ordering::Less
        );
        assert!(rdf_comparison_scratch(usize::MAX, 1).is_err());
        assert!(rdf_comparison_scratch(usize::MAX / 256, 1).is_err());
    }

    #[test]
    fn qualified_comparator_does_not_expand_lexical_exponents() {
        let comparator = RdfValueComparator;
        let enormous_exponent = "1e999999999999999999999999999999999999999";
        let invalid_decimal = Value::RdfLiteral {
            lexical: enormous_exponent.into(),
            datatype: Some(Literal::XSD_DECIMAL.into()),
            language: None,
        };
        // Decimal exponents are invalid lexical forms, not requests to
        // materialize that many zeros. Preserve the existing fallback order.
        assert!(RdfNumeric::from_value(&invalid_decimal).is_none());
        assert_eq!(
            comparator
                .compare(Some(&invalid_decimal), Some(&Value::Int64(1)))
                .unwrap(),
            Ordering::Greater
        );
        let mut nested = Value::Null;
        for _ in 0..258 {
            nested = Value::List(vec![nested].into());
        }
        assert!(comparator.scratch_bytes(Some(&nested), None).is_err());
    }

    #[cfg(feature = "spill")]
    #[test]
    fn exact_rdf_runs_span_one_two_and_many_input_chunks() {
        use grafeo_common::memory::buffer::BufferManager;
        use grafeo_core::execution::DataChunk;
        use grafeo_core::execution::operators::OperatorError;
        use grafeo_core::execution::spill::{
            CleartextSpillRecordProvider, SpillFrameLimits, SpillIo, SpillIoOperation,
        };
        use grafeo_core::execution::vector::ValueVector;
        use std::mem::size_of;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

        struct Source(std::vec::IntoIter<DataChunk>);
        impl Operator for Source {
            fn next(&mut self) -> OperatorResult {
                Ok(self.0.next())
            }
            fn reset(&mut self) {
                self.0 = Vec::new().into_iter();
            }
            fn name(&self) -> &'static str {
                "FixedRdfSortChunks"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }
        struct RunCounts {
            creates: AtomicUsize,
            payloads: AtomicUsize,
            completed: AtomicUsize,
            rows: [AtomicUsize; 4],
            read_started: AtomicBool,
            merge_creates: AtomicUsize,
        }
        impl SpillIo for RunCounts {
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                match operation {
                    SpillIoOperation::Create => {
                        self.creates.fetch_add(1, AtomicOrdering::Relaxed);
                        self.payloads.store(0, AtomicOrdering::Relaxed);
                        if self.read_started.load(AtomicOrdering::Relaxed) {
                            self.merge_creates.fetch_add(1, AtomicOrdering::Relaxed);
                        }
                    }
                    SpillIoOperation::WritePayload => {
                        self.payloads.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    SpillIoOperation::Flush => {
                        let index = self.completed.fetch_add(1, AtomicOrdering::Relaxed);
                        if let Some(rows) = self.rows.get(index) {
                            // Current framed sort format: FileStart,
                            // SortRunStart, one frame per row, FileEnd.
                            rows.store(
                                self.payloads
                                    .load(AtomicOrdering::Relaxed)
                                    .checked_sub(3)
                                    .ok_or(std::io::ErrorKind::InvalidData)?,
                                AtomicOrdering::Relaxed,
                            );
                        }
                    }
                    SpillIoOperation::ReadOpen => {
                        self.read_started.store(true, AtomicOrdering::Relaxed);
                    }
                    _ => {}
                }
                Ok(())
            }
            fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
                // Atomic observations and ErrorKind failures allocate no heap,
                // including across all callbacks and retained sort failures.
                Some(0)
            }
            fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
                Some(0)
            }
        }
        fn key(index: usize) -> Value {
            match index % 4 {
                0 => Value::RdfLiteral {
                    lexical: "9007199254740993".into(),
                    datatype: Some(Literal::XSD_INTEGER.into()),
                    language: None,
                },
                2 => Value::Float64(9007199254740992.0),
                _ => Value::RdfLiteral {
                    lexical: "1.25".into(),
                    datatype: Some(Literal::XSD_DECIMAL.into()),
                    language: None,
                },
            }
        }
        fn input(chunk_index: usize, padding_capacity: usize) -> DataChunk {
            let mut keys = ValueVector::with_capacity(LogicalType::Any, 4);
            let mut ordinals = ValueVector::with_capacity(LogicalType::Int64, 4);
            let mut padding = ValueVector::with_capacity(LogicalType::Int64, padding_capacity);
            for index in 0..4 {
                keys.push_value(key(index));
                ordinals.push_int64(i64::try_from(chunk_index * 4 + index).unwrap());
                padding.push_int64(0);
            }
            DataChunk::new(vec![keys, ordinals, padding])
        }
        fn execute(
            chunks: Vec<DataChunk>,
            resources: &QueryResourceContext,
        ) -> Result<Vec<Vec<Value>>, OperatorError> {
            let mut sort = RdfSortOperator::new(
                Box::new(Source(chunks.into_iter())),
                vec![SortKey::ascending(0)],
                vec![LogicalType::Any, LogicalType::Int64],
            );
            sort.install_resource_context(resources).unwrap();
            let mut rows = Vec::new();
            while let Some(chunk) = sort.next()? {
                assert_eq!(chunk.column_count(), 2, "hidden padding column leaked");
                for row in chunk.selected_indices() {
                    rows.push(
                        (0..2)
                            .map(|column| chunk.column(column).unwrap().get_value(row).unwrap())
                            .collect(),
                    );
                }
            }
            sort.reset();
            Ok(rows)
        }
        const BUDGET: usize = 8 << 20;
        const QUARTER: usize = BUDGET / 4;
        for chunks_per_run in [1, 2, 8] {
            let make_chunks = || {
                let mut chunks = vec![input(0, QUARTER / size_of::<i64>())];
                assert!(chunks[0].output_retained_bytes().unwrap() > QUARTER);
                for index in 1..=3 * chunks_per_run {
                    let chunk = input(index, QUARTER / chunks_per_run / size_of::<i64>());
                    let bytes = chunk.output_retained_bytes().unwrap();
                    assert!(bytes * chunks_per_run >= QUARTER);
                    assert!(bytes * (chunks_per_run - 1) < QUARTER);
                    chunks.push(chunk);
                }
                chunks
            };
            let count = 4 * (1 + 3 * chunks_per_run);
            let expected: Vec<_> = [1usize, 2, 0]
                .into_iter()
                .flat_map(|class| {
                    (0..count)
                        .filter(move |index| match class {
                            1 => index % 4 == 1 || index % 4 == 3,
                            other => index % 4 == other,
                        })
                        .map(|index| vec![key(index), Value::Int64(i64::try_from(index).unwrap())])
                })
                .collect();
            let resident = BufferManager::with_budget(64 << 20);
            let resident_resources = QueryResourceContext::new(resident.clone()).unwrap();
            assert_eq!(
                execute(make_chunks(), &resident_resources).unwrap(),
                expected
            );
            assert_eq!(resident.allocated(), 0);
            let directory = tempfile::tempdir().unwrap();
            let counts = Arc::new(RunCounts {
                creates: AtomicUsize::new(0),
                payloads: AtomicUsize::new(0),
                completed: AtomicUsize::new(0),
                rows: std::array::from_fn(|_| AtomicUsize::new(0)),
                read_started: AtomicBool::new(false),
                merge_creates: AtomicUsize::new(0),
            });
            let memory = BufferManager::with_budget(BUDGET);
            let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
                directory.path(),
                memory.clone(),
                grafeo_core::execution::QueryExecutionControl::new().token(),
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
                counts.clone(),
                grafeo_core::execution::SpillDiskQuota::new(u64::MAX),
            );
            assert_eq!(execute(make_chunks(), &resources).unwrap(), expected);
            assert_eq!(counts.creates.load(AtomicOrdering::Relaxed), 4);
            assert_eq!(counts.completed.load(AtomicOrdering::Relaxed), 4);
            assert_eq!(
                counts.merge_creates.load(AtomicOrdering::Relaxed),
                0,
                "four initial runs fit the existing final merge fan-in"
            );
            assert!(counts.read_started.load(AtomicOrdering::Relaxed));
            assert_eq!(counts.rows[0].load(AtomicOrdering::Relaxed), 4);
            for rows in &counts.rows[1..] {
                assert_eq!(rows.load(AtomicOrdering::Relaxed), 4 * chunks_per_run);
            }
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(memory.allocated(), 0);
        }
    }

    #[test]
    fn mixed_values_do_not_break_sort_transitivity() {
        let one = Value::Int64(1);
        let two = Value::Int64(2);
        let text = Value::String("between".into());

        assert_eq!(rdf_compare_values_total(&one, &two), Ordering::Less);
        assert_eq!(rdf_compare_values_total(&one, &text), Ordering::Less);
        assert_eq!(rdf_compare_values_total(&two, &text), Ordering::Less);
    }

    #[test]
    fn tagged_terms_follow_sparql_category_order() {
        let blank = tagged_rdf_term(Value::String("_:b".into()), Term::blank("b"));
        let iri = tagged_rdf_term(Value::String("urn:z".into()), Term::iri("urn:z"));
        let literal = tagged_rdf_term(Value::String("a".into()), Term::literal("a"));

        assert_eq!(
            rdf_compare_values_with_nulls(&Some(blank), &Some(iri.clone()), NullOrder::NullsFirst),
            Ordering::Less
        );
        assert_eq!(
            rdf_compare_values_with_nulls(&Some(iri), &Some(literal), NullOrder::NullsFirst),
            Ordering::Less
        );
    }

    #[test]
    fn tagged_literal_order_is_transitive_across_numeric_representations() {
        let decimal = tagged_rdf_term(
            Value::RdfLiteral {
                lexical: "1.0".into(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
                language: None,
            },
            Term::typed_literal("1.0", "http://www.w3.org/2001/XMLSchema#decimal"),
        );
        let integer = tagged_rdf_term(
            Value::Int64(2),
            Term::typed_literal("2", "http://www.w3.org/2001/XMLSchema#integer"),
        );
        let text = tagged_rdf_term(Value::String("text".into()), Term::literal("text"));

        assert_eq!(
            rdf_compare_values_with_nulls(
                &Some(decimal.clone()),
                &Some(integer.clone()),
                NullOrder::NullsFirst,
            ),
            Ordering::Less
        );
        assert_eq!(
            rdf_compare_values_with_nulls(
                &Some(integer),
                &Some(text.clone()),
                NullOrder::NullsFirst,
            ),
            Ordering::Less
        );
        assert_eq!(
            rdf_compare_values_with_nulls(&Some(decimal), &Some(text), NullOrder::NullsFirst,),
            Ordering::Less
        );
    }
}
