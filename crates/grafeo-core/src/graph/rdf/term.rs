//! RDF Terms.
//!
//! RDF terms are the building blocks of triples. There are four types:
//! - IRIs (Internationalized Resource Identifiers)
//! - Blank nodes (anonymous nodes)
//! - Literals (data values)
//! - Variables (for query patterns, not stored)

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// An RDF term.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Term {
    /// An IRI (Internationalized Resource Identifier).
    Iri(Iri),
    /// A blank node (anonymous node).
    BlankNode(BlankNode),
    /// A literal value.
    Literal(Literal),
}

impl Term {
    /// Creates an IRI term.
    #[inline]
    pub fn iri(value: impl Into<Arc<str>>) -> Self {
        Term::Iri(Iri::new(value))
    }

    /// Creates a blank node term.
    #[inline]
    pub fn blank(id: impl Into<Arc<str>>) -> Self {
        Term::BlankNode(BlankNode::new(id))
    }

    /// Creates a simple literal (xsd:string).
    #[inline]
    pub fn literal(value: impl Into<Arc<str>>) -> Self {
        Term::Literal(Literal::simple(value))
    }

    /// Creates a typed literal.
    #[inline]
    pub fn typed_literal(value: impl Into<Arc<str>>, datatype: impl Into<Arc<str>>) -> Self {
        Term::Literal(Literal::typed(value, datatype))
    }

    /// Creates a language-tagged literal.
    #[inline]
    pub fn lang_literal(value: impl Into<Arc<str>>, lang: impl Into<Arc<str>>) -> Self {
        Term::Literal(Literal::with_language(value, lang))
    }

    /// Returns true if this term is an IRI.
    #[inline]
    #[must_use]
    pub fn is_iri(&self) -> bool {
        matches!(self, Term::Iri(_))
    }

    /// Returns true if this term is a blank node.
    #[inline]
    #[must_use]
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Term::BlankNode(_))
    }

    /// Returns true if this term is a literal.
    #[inline]
    #[must_use]
    pub fn is_literal(&self) -> bool {
        matches!(self, Term::Literal(_))
    }

    /// Returns the IRI if this term is an IRI.
    #[inline]
    #[must_use]
    pub fn as_iri(&self) -> Option<&Iri> {
        match self {
            Term::Iri(iri) => Some(iri),
            _ => None,
        }
    }

    /// Returns the blank node if this term is a blank node.
    #[inline]
    #[must_use]
    pub fn as_blank_node(&self) -> Option<&BlankNode> {
        match self {
            Term::BlankNode(bn) => Some(bn),
            _ => None,
        }
    }

    /// Returns the literal if this term is a literal.
    #[inline]
    #[must_use]
    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            Term::Literal(lit) => Some(lit),
            _ => None,
        }
    }
}

impl Term {
    /// Parses an N-Triples encoded term string.
    ///
    /// Supported formats:
    /// - `<iri>` for IRIs
    /// - `_:id` for blank nodes
    /// - `"value"` for simple literals
    /// - `"value"^^<type>` for typed literals
    /// - `"value"@lang` for language-tagged literals
    pub fn from_ntriples(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some(inner) = s.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
            Some(Term::Iri(Iri::new(inner)))
        } else if let Some(id) = s.strip_prefix("_:") {
            Some(Term::BlankNode(BlankNode::new(id)))
        } else if let Some(rest) = s.strip_prefix('"') {
            let (value, after) = unescape_ntriples_literal(rest)?;
            if let Some(lang) = after.strip_prefix('@') {
                Some(Term::Literal(Literal::with_language(value, lang)))
            } else if let Some(typed) = after.strip_prefix("^^<").and_then(|s| s.strip_suffix('>'))
            {
                Some(Term::Literal(Literal::typed(value, typed)))
            } else if after.is_empty() {
                Some(Term::Literal(Literal::simple(value)))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Parses exactly the canonical byte spelling emitted by this type's
    /// N-Triples writer. Unlike [`from_ntriples`](Self::from_ntriples), this
    /// rejects surrounding whitespace and alternate escape spellings while
    /// decoding literals in a single pass.
    #[cfg(any(test, feature = "ring-index"))]
    pub(crate) fn from_canonical_ntriples(s: &str) -> Option<Self> {
        if let Some(inner) = s
            .strip_prefix('<')
            .and_then(|value| value.strip_suffix('>'))
        {
            return Some(Self::Iri(Iri::new(inner)));
        }
        if let Some(id) = s.strip_prefix("_:") {
            return Some(Self::BlankNode(BlankNode::new(id)));
        }
        let rest = s.strip_prefix('"')?;
        let (value, suffix) = unescape_canonical_ntriples_literal(rest)?;
        if let Some(language) = suffix.strip_prefix('@') {
            return Some(Self::Literal(Literal::with_language(value, language)));
        }
        if let Some(datatype) = suffix
            .strip_prefix("^^<")
            .and_then(|value| value.strip_suffix('>'))
        {
            if datatype == Literal::XSD_STRING {
                return None;
            }
            return Some(Self::Literal(Literal::typed(value, datatype)));
        }
        suffix
            .is_empty()
            .then(|| Self::Literal(Literal::simple(value)))
    }

    /// Converts this term to its N-Triples string representation.
    ///
    /// Round-trips with [`from_ntriples`](Self::from_ntriples).
    pub fn to_ntriples(&self) -> String {
        self.to_string()
    }

    /// Returns the canonical key used for RDF term identity joins.
    ///
    /// RDF language tags are case-insensitive and are normalized to lowercase
    /// in the abstract syntax. `Display` already canonicalizes an explicit
    /// `xsd:string` datatype to the same spelling as a simple literal.
    #[must_use]
    pub fn canonical_identity_key(&self) -> String {
        if let Self::Literal(literal) = self
            && let Some(language) = literal.language()
        {
            return Self::lang_literal(literal.value().to_string(), language.to_ascii_lowercase())
                .to_ntriples();
        }
        self.to_ntriples()
    }

    /// Returns whether two terms denote the same RDF term identity.
    ///
    /// Lossless source representations can differ only in language-tag case
    /// while remaining the same RDF term. The structural fast path avoids
    /// allocating canonical strings during ordinary pattern scans.
    #[must_use]
    pub fn same_identity(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        let (Self::Literal(left), Self::Literal(right)) = (self, other) else {
            return false;
        };
        left.value() == right.value()
            && left.datatype() == right.datatype()
            && left
                .language()
                .zip(right.language())
                .is_some_and(|(left, right)| left.eq_ignore_ascii_case(right))
    }
}

/// Unescape an N-Triples quoted literal body. `s` starts just after the opening `"`.
///
/// Returns `(lexical, remainder after closing quote)`.
fn unescape_ntriples_literal(s: &str) -> Option<(String, &str)> {
    let mut value = String::new();
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some((value, chars.as_str())),
            '\\' => {
                let esc = chars.next()?;
                match esc {
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    '\'' => value.push('\''),
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    'b' => value.push('\u{0008}'),
                    'f' => value.push('\u{000C}'),
                    'u' => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if hex.len() != 4 {
                            return None;
                        }
                        let cp = u32::from_str_radix(&hex, 16).ok()?;
                        value.push(char::from_u32(cp)?);
                    }
                    'U' => {
                        let hex: String = chars.by_ref().take(8).collect();
                        if hex.len() != 8 {
                            return None;
                        }
                        let cp = u32::from_str_radix(&hex, 16).ok()?;
                        value.push(char::from_u32(cp)?);
                    }
                    other => {
                        value.push('\\');
                        value.push(other);
                    }
                }
            }
            other => value.push(other),
        }
    }
    None
}

/// Decode the writer's one canonical literal spelling in a single pass.
#[cfg(any(test, feature = "ring-index"))]
fn unescape_canonical_ntriples_literal(s: &str) -> Option<(String, &str)> {
    let mut value = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some((value, chars.as_str())),
            '\\' => match chars.next()? {
                '"' => value.push('"'),
                '\\' => value.push('\\'),
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                'b' => value.push('\u{0008}'),
                'f' => value.push('\u{000C}'),
                'u' => {
                    let mut code = 0u32;
                    for _ in 0..4 {
                        code = code
                            .checked_mul(16)?
                            .checked_add(canonical_hex(chars.next()?)?)?;
                    }
                    let character = char::from_u32(code)?;
                    if !(character <= '\u{001F}' || character == '\u{007F}')
                        || matches!(character, '\u{0008}' | '\t' | '\n' | '\u{000C}' | '\r')
                    {
                        return None;
                    }
                    value.push(character);
                }
                _ => return None,
            },
            control if control <= '\u{001F}' || control == '\u{007F}' => return None,
            other => value.push(other),
        }
    }
    None
}

#[cfg(any(test, feature = "ring-index"))]
fn canonical_hex(character: char) -> Option<u32> {
    match character {
        '0'..='9' => Some(u32::from(character) - u32::from('0')),
        'A'..='F' => Some(u32::from(character) - u32::from('A') + 10),
        _ => None,
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Iri(iri) => write!(f, "{}", iri),
            Term::BlankNode(bn) => write!(f, "{}", bn),
            Term::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

/// An IRI (Internationalized Resource Identifier).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Iri {
    /// The IRI string.
    value: Arc<str>,
}

impl Iri {
    /// Creates a new IRI.
    #[inline]
    pub fn new(value: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
        }
    }

    /// Returns the IRI string.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Returns the local name (part after last # or /).
    #[must_use]
    pub fn local_name(&self) -> &str {
        if let Some(pos) = self.value.rfind('#') {
            &self.value[pos + 1..]
        } else if let Some(pos) = self.value.rfind('/') {
            &self.value[pos + 1..]
        } else {
            &self.value
        }
    }

    /// Returns the namespace (part before local name).
    #[must_use]
    pub fn namespace(&self) -> &str {
        if let Some(pos) = self.value.rfind('#') {
            &self.value[..=pos]
        } else if let Some(pos) = self.value.rfind('/') {
            &self.value[..=pos]
        } else {
            ""
        }
    }
}

impl fmt::Display for Iri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.value)
    }
}

impl From<&str> for Iri {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Iri {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// A blank node (anonymous node).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlankNode {
    /// The blank node identifier.
    id: Arc<str>,
}

impl BlankNode {
    /// Creates a new blank node with the given identifier.
    #[inline]
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self { id: id.into() }
    }

    /// Returns the blank node identifier.
    #[inline]
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for BlankNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_:{}", self.id)
    }
}

/// An RDF literal (data value).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    /// The lexical form (string value).
    value: Arc<str>,
    /// The datatype IRI (e.g., xsd:string, xsd:integer).
    datatype: Arc<str>,
    /// Optional language tag (e.g., "en", "de").
    language: Option<Arc<str>>,
}

impl Literal {
    /// XSD namespace.
    pub const XSD: &'static str = "http://www.w3.org/2001/XMLSchema#";

    /// RDF namespace.
    pub const RDF: &'static str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

    /// xsd:string datatype IRI.
    pub const XSD_STRING: &'static str = "http://www.w3.org/2001/XMLSchema#string";

    /// xsd:integer datatype IRI.
    pub const XSD_INTEGER: &'static str = "http://www.w3.org/2001/XMLSchema#integer";

    /// xsd:decimal datatype IRI.
    pub const XSD_DECIMAL: &'static str = "http://www.w3.org/2001/XMLSchema#decimal";

    /// xsd:double datatype IRI.
    pub const XSD_DOUBLE: &'static str = "http://www.w3.org/2001/XMLSchema#double";

    /// xsd:boolean datatype IRI.
    pub const XSD_BOOLEAN: &'static str = "http://www.w3.org/2001/XMLSchema#boolean";

    /// xsd:dateTime datatype IRI.
    pub const XSD_DATETIME: &'static str = "http://www.w3.org/2001/XMLSchema#dateTime";

    /// xsd:date datatype IRI.
    pub const XSD_DATE: &'static str = "http://www.w3.org/2001/XMLSchema#date";

    /// rdf:langString datatype IRI.
    pub const RDF_LANG_STRING: &'static str =
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

    /// Creates a simple string literal.
    #[inline]
    pub fn simple(value: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
            datatype: Self::XSD_STRING.into(),
            language: None,
        }
    }

    /// Creates a typed literal.
    #[inline]
    pub fn typed(value: impl Into<Arc<str>>, datatype: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
            datatype: datatype.into(),
            language: None,
        }
    }

    /// Creates a language-tagged literal.
    #[inline]
    pub fn with_language(value: impl Into<Arc<str>>, language: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
            datatype: Self::RDF_LANG_STRING.into(),
            language: Some(language.into()),
        }
    }

    /// Creates an integer literal.
    #[inline]
    pub fn integer(value: i64) -> Self {
        Self::typed(value.to_string(), Self::XSD_INTEGER)
    }

    /// Creates a double literal.
    #[inline]
    pub fn double(value: f64) -> Self {
        Self::typed(value.to_string(), Self::XSD_DOUBLE)
    }

    /// Creates a boolean literal.
    #[inline]
    pub fn boolean(value: bool) -> Self {
        Self::typed(if value { "true" } else { "false" }, Self::XSD_BOOLEAN)
    }

    /// Returns the lexical form.
    #[inline]
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the datatype IRI.
    #[inline]
    #[must_use]
    pub fn datatype(&self) -> &str {
        &self.datatype
    }

    /// Returns the language tag, if any.
    #[inline]
    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Returns true if this is a simple string literal.
    #[inline]
    #[must_use]
    pub fn is_simple(&self) -> bool {
        self.datatype.as_ref() == Self::XSD_STRING && self.language.is_none()
    }

    /// Returns true if this literal has a language tag.
    #[inline]
    #[must_use]
    pub fn is_lang_string(&self) -> bool {
        self.language.is_some()
    }

    /// Attempts to parse the literal as an integer.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        self.value.parse().ok()
    }

    /// Attempts to parse the literal as a double.
    #[must_use]
    pub fn as_double(&self) -> Option<f64> {
        self.value.parse().ok()
    }

    /// Attempts to parse the literal as a boolean.
    #[must_use]
    pub fn as_boolean(&self) -> Option<bool> {
        match self.value.as_ref() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Escape special characters in the value
        write!(f, "\"")?;
        for ch in self.value.chars() {
            match ch {
                '"' => write!(f, "\\\"")?,
                '\\' => write!(f, "\\\\")?,
                '\n' => write!(f, "\\n")?,
                '\r' => write!(f, "\\r")?,
                '\t' => write!(f, "\\t")?,
                '\u{0008}' => write!(f, "\\b")?,
                '\u{000C}' => write!(f, "\\f")?,
                control if control <= '\u{001F}' || control == '\u{007F}' => {
                    write!(f, "\\u{:04X}", u32::from(control))?;
                }
                _ => write!(f, "{}", ch)?,
            }
        }
        write!(f, "\"")?;

        if let Some(ref lang) = self.language {
            write!(f, "@{}", lang)
        } else if self.datatype.as_ref() != Self::XSD_STRING {
            write!(f, "^^<{}>", self.datatype)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iri_creation() {
        let iri = Iri::new("http://example.org/resource");
        assert_eq!(iri.as_str(), "http://example.org/resource");
        assert_eq!(iri.local_name(), "resource");
        assert_eq!(iri.namespace(), "http://example.org/");
    }

    #[test]
    fn test_iri_with_fragment() {
        let iri = Iri::new("http://xmlns.com/foaf/0.1#Person");
        assert_eq!(iri.local_name(), "Person");
        assert_eq!(iri.namespace(), "http://xmlns.com/foaf/0.1#");
    }

    #[test]
    fn test_blank_node() {
        let bn = BlankNode::new("b0");
        assert_eq!(bn.id(), "b0");
        assert_eq!(bn.to_string(), "_:b0");
    }

    #[test]
    fn test_simple_literal() {
        let lit = Literal::simple("Hello");
        assert_eq!(lit.value(), "Hello");
        assert_eq!(lit.datatype(), Literal::XSD_STRING);
        assert!(lit.is_simple());
        assert!(!lit.is_lang_string());
    }

    #[test]
    fn test_typed_literal() {
        let lit = Literal::integer(42);
        assert_eq!(lit.value(), "42");
        assert_eq!(lit.datatype(), Literal::XSD_INTEGER);
        assert_eq!(lit.as_integer(), Some(42));
    }

    #[test]
    fn test_lang_literal() {
        let lit = Literal::with_language("Bonjour", "fr");
        assert_eq!(lit.value(), "Bonjour");
        assert_eq!(lit.language(), Some("fr"));
        assert!(lit.is_lang_string());
    }

    #[test]
    fn canonical_identity_key_normalizes_language_and_xsd_string() {
        assert_eq!(
            Term::lang_literal("value", "EN").canonical_identity_key(),
            Term::lang_literal("value", "en").canonical_identity_key()
        );
        assert_eq!(
            Term::typed_literal("value", Literal::XSD_STRING).canonical_identity_key(),
            Term::literal("value").canonical_identity_key()
        );
    }

    #[test]
    fn same_identity_is_structural_and_language_case_insensitive() {
        assert!(
            Term::lang_literal("value", "EN").same_identity(&Term::lang_literal("value", "en"))
        );
        assert!(
            !Term::lang_literal("other", "EN").same_identity(&Term::lang_literal("value", "en"))
        );
        assert!(!Term::iri("value").same_identity(&Term::literal("value")));
    }

    #[test]
    fn test_term_display() {
        assert_eq!(
            Term::iri("http://example.org").to_string(),
            "<http://example.org>"
        );
        assert_eq!(Term::blank("b0").to_string(), "_:b0");
        assert_eq!(Term::literal("Hello").to_string(), "\"Hello\"");
        assert_eq!(
            Term::lang_literal("Bonjour", "fr").to_string(),
            "\"Bonjour\"@fr"
        );
        assert_eq!(
            Term::typed_literal("42", Literal::XSD_INTEGER).to_string(),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }

    #[test]
    fn canonical_ntriples_parser_matches_writer_in_one_pass() {
        let terms = [
            Term::iri("https://example.test/a"),
            Term::blank("b0"),
            Term::literal("plain"),
            Term::literal("quote \" slash \\ newline\n control\u{0001} café"),
            Term::lang_literal("bonjour", "fr"),
            Term::typed_literal("42", Literal::XSD_INTEGER),
        ];
        for term in terms {
            assert_eq!(
                Term::from_canonical_ntriples(&term.to_ntriples()),
                Some(term)
            );
        }

        assert!(Term::from_canonical_ntriples(" <https://example.test/a>").is_none());
        assert!(
            Term::from_canonical_ntriples("\"value\"^^<http://www.w3.org/2001/XMLSchema#string>")
                .is_none()
        );
        assert!(Term::from_canonical_ntriples("\"caf\\u00E9\"").is_none());
        assert!(Term::from_canonical_ntriples("\"line\\u000A\"").is_none());
        assert_eq!(
            Term::from_canonical_ntriples("\"nul\\u0000\""),
            Some(Term::literal("nul\0"))
        );
    }

    #[test]
    fn from_ntriples_unicode_plain_literal() {
        let original = Term::literal("café 日本語 🎵");
        let nt = original.to_ntriples();
        let parsed = Term::from_ntriples(&nt).expect("parse");
        assert_eq!(parsed, original, "nt={nt:?}");
    }

    #[test]
    fn from_ntriples_unicode_lang_literal() {
        let original = Term::lang_literal("日本語", "ja");
        let parsed = Term::from_ntriples(&original.to_ntriples()).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn from_ntriples_unicode_typed_literal() {
        let original = Term::typed_literal("Москва", "http://ex.org/City");
        let parsed = Term::from_ntriples(&original.to_ntriples()).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn from_ntriples_unicode_escape() {
        let parsed = Term::from_ntriples(r#""caf\u00E9""#).expect("parse");
        assert_eq!(parsed, Term::literal("café"));
    }
}
