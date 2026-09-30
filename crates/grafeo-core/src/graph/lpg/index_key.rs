//! Collision-free keys for indexes scoped to an LPG graph, label and property.
//!
//! Index registries historically used `"label:property"`. That representation
//! is ambiguous as soon as either component contains `':'`. New keys use a
//! versioned, byte-length-prefixed label; the property occupies the remainder:
//!
//! ```text
//! @idx1:<label UTF-8 byte length>:<label><property>
//! ```
//!
//! A byte length is unambiguous and preserves arbitrary Unicode without
//! escaping. The decoder also accepts the old representation when it contains
//! exactly one colon, which keeps simple legacy keys readable without guessing
//! how an ambiguous legacy key should be split.
//!
//! Flat scoped runtime keys add one outer, canonical scope prefix:
//!
//! ```text
//! @gidx1:<graph UTF-8 byte length>:<graph>@idx1:<label byte length>:<label><property>
//! ```
//!
//! The explicit prefix keeps the default graph distinct from every named graph,
//! including a named graph whose name is empty or `"default"`. Named keys have
//! no legacy representation, so both their outer length and nested index key
//! must be canonical.
//!
//! Current persistence uses the checked [`PhysicalIndexKey`] byte representation,
//! never these scoped strings or their predecessor aliases.

use grafeo_common::types::GraphPath;
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Physical membership family, independent of optional implementation features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhysicalIndexFamily {
    /// Property and BTree indexes share label-independent physical membership.
    Property,
    /// Full-text membership for one label and property.
    Text,
    /// Vector membership for one label and property.
    Vector,
}

/// Validated physical index identity, ordered by graph, family, label, property.
///
/// Its current Serde representation validates graph bytes, family and label
/// shape on decode. This identity never supplies mutation authority.
/// Names remain literal, including empty strings, NULs and separators. Creation
/// policy and optional-family support are validated separately by the engine.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalIndexKey {
    graph: GraphPath,
    family: PhysicalIndexFamily,
    label: Option<String>,
    property: String,
}

#[derive(Serialize)]
struct PhysicalIndexKeyWireRef<'a> {
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    graph: &'a GraphPath,
    family: u8,
    label: Option<&'a str>,
    property: &'a str,
}

#[derive(Deserialize)]
struct PhysicalIndexKeyWire {
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    graph: GraphPath,
    family: u8,
    label: Option<String>,
    property: String,
}

impl Serialize for PhysicalIndexKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        PhysicalIndexKeyWireRef {
            graph: &self.graph,
            family: match self.family {
                PhysicalIndexFamily::Property => 0,
                PhysicalIndexFamily::Text => 1,
                PhysicalIndexFamily::Vector => 2,
            },
            label: self.label.as_deref(),
            property: &self.property,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PhysicalIndexKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = PhysicalIndexKeyWire::deserialize(deserializer)?;
        let family = match wire.family {
            0 => PhysicalIndexFamily::Property,
            1 => PhysicalIndexFamily::Text,
            2 => PhysicalIndexFamily::Vector,
            _ => return Err(serde::de::Error::custom("invalid physical index family")),
        };
        Self::new(wire.graph, family, wire.label, wire.property).map_err(serde::de::Error::custom)
    }
}

impl PhysicalIndexKey {
    /// Constructs an identity from independently owned components.
    ///
    /// # Errors
    /// Rejects labels on Property keys and absent labels on Text/Vector keys.
    /// An explicitly empty label is present and remains representable.
    pub fn new(
        graph: GraphPath,
        family: PhysicalIndexFamily,
        label: Option<String>,
        property: String,
    ) -> Result<Self> {
        match (family, label.is_some()) {
            (PhysicalIndexFamily::Property, true) => {
                return Err(Error::InvalidValue(
                    "Property physical index keys must not have a label".to_owned(),
                ));
            }
            (PhysicalIndexFamily::Text | PhysicalIndexFamily::Vector, false) => {
                return Err(Error::InvalidValue(
                    "Text and Vector physical index keys require a label".to_owned(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            graph,
            family,
            label,
            property,
        })
    }

    /// Constructs a label-independent Property/BTree identity.
    #[must_use]
    pub fn property(graph: GraphPath, property: impl Into<String>) -> Self {
        Self {
            graph,
            family: PhysicalIndexFamily::Property,
            label: None,
            property: property.into(),
        }
    }

    /// Constructs a Text identity with an explicitly present label.
    #[must_use]
    pub fn text(graph: GraphPath, label: impl Into<String>, property: impl Into<String>) -> Self {
        Self {
            graph,
            family: PhysicalIndexFamily::Text,
            label: Some(label.into()),
            property: property.into(),
        }
    }

    /// Constructs a Vector identity with an explicitly present label.
    #[must_use]
    pub fn vector(graph: GraphPath, label: impl Into<String>, property: impl Into<String>) -> Self {
        Self {
            graph,
            family: PhysicalIndexFamily::Vector,
            label: Some(label.into()),
            property: property.into(),
        }
    }

    /// Literal root-relative graph identity.
    #[must_use]
    pub const fn graph(&self) -> &GraphPath {
        &self.graph
    }

    /// Physical membership family.
    #[must_use]
    pub const fn family(&self) -> PhysicalIndexFamily {
        self.family
    }

    /// Label, absent only for the Property family.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Literal property name.
    #[must_use]
    pub fn property_name(&self) -> &str {
        &self.property
    }

    /// Moves this identity to another graph without cloning its other names.
    #[must_use]
    pub fn with_graph(mut self, graph: GraphPath) -> Self {
        self.graph = graph;
        self
    }
}

const PREFIX: &str = "@idx1:";
const GRAPH_PREFIX: &str = "@gidx1:";

fn is_canonical_decimal(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && (value == "0" || !value.starts_with('0'))
}

/// Encodes a `(label, property)` pair as a collision-free registry key.
#[must_use]
pub fn encode_index_key(label: &str, property: &str) -> String {
    let label_len = label.len().to_string();
    let mut key =
        String::with_capacity(PREFIX.len() + label_len.len() + 1 + label.len() + property.len());
    key.push_str(PREFIX);
    key.push_str(&label_len);
    key.push(':');
    key.push_str(label);
    key.push_str(property);
    key
}

/// Decodes a collision-free key or an unambiguous legacy `"label:property"` key.
///
/// Legacy keys containing more than one colon are rejected deliberately: no
/// decoder can determine whether a colon belonged to the label or property.
#[must_use]
pub fn decode_index_key(key: &str) -> Option<(&str, &str)> {
    if let Some(encoded) = key.strip_prefix(PREFIX)
        && let Some((label_len, components)) = encoded.split_once(':')
        && !label_len.is_empty()
        && label_len.bytes().all(|byte| byte.is_ascii_digit())
        && let Ok(label_len) = label_len.parse::<usize>()
        && let (Some(label), Some(property)) =
            (components.get(..label_len), components.get(label_len..))
    {
        return Some((label, property));
    }

    let (label, property) = key.split_once(':')?;
    (!property.contains(':')).then_some((label, property))
}

/// Converts either supported representation to the canonical encoded form.
#[must_use]
pub fn canonicalize_index_key(key: &str) -> Option<String> {
    let (label, property) = decode_index_key(key)?;
    Some(encode_index_key(label, property))
}

/// Returns whether two encoded or legacy-simple keys identify the same index.
#[must_use]
pub fn index_keys_match(left: &str, right: &str) -> bool {
    decode_index_key(left)
        .zip(decode_index_key(right))
        .is_some_and(|(left, right)| left == right)
}

/// Encodes a graph-qualified `(label, property)` index key.
///
/// `None` preserves the established default-graph bytes exactly. `Some`
/// always emits the named-graph envelope, even when the graph name is empty.
#[must_use]
pub fn encode_scoped_index_key(graph: Option<&str>, label: &str, property: &str) -> String {
    let index_key = encode_index_key(label, property);
    let Some(graph) = graph else {
        return index_key;
    };

    let graph_len = graph.len().to_string();
    let mut key = String::with_capacity(
        GRAPH_PREFIX.len() + graph_len.len() + 1 + graph.len() + index_key.len(),
    );
    key.push_str(GRAPH_PREFIX);
    key.push_str(&graph_len);
    key.push(':');
    key.push_str(graph);
    key.push_str(&index_key);
    key
}

/// Decodes a graph-qualified canonical key or a supported default-graph key.
///
/// Default keys retain the compatibility behavior of [`decode_index_key`]. A
/// named key is deliberately stricter: the graph byte length has one canonical
/// decimal spelling and the nested key must be the canonical `@idx1` form.
#[must_use]
pub fn decode_scoped_index_key(key: &str) -> Option<(Option<&str>, &str, &str)> {
    if let Some(encoded) = key.strip_prefix(GRAPH_PREFIX) {
        let (graph_len, components) = encoded.split_once(':')?;
        if !is_canonical_decimal(graph_len) {
            return None;
        }
        let graph_len = graph_len.parse::<usize>().ok()?;
        let graph = components.get(..graph_len)?;
        let index_key = components.get(graph_len..)?;
        let (label, property) = decode_index_key(index_key)?;
        if index_key != encode_index_key(label, property) {
            return None;
        }
        return Some((Some(graph), label, property));
    }

    let (label, property) = decode_index_key(key)?;
    Some((None, label, property))
}

/// Converts a supported scoped representation to its canonical bytes.
#[must_use]
pub fn canonicalize_scoped_index_key(key: &str) -> Option<String> {
    let (graph, label, property) = decode_scoped_index_key(key)?;
    Some(encode_scoped_index_key(graph, label, property))
}

/// Returns whether two keys identify the same graph-qualified index.
#[must_use]
pub fn scoped_index_keys_match(left: &str, right: &str) -> bool {
    decode_scoped_index_key(left)
        .zip(decode_scoped_index_key(right))
        .is_some_and(|(left, right)| left == right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_key_wire_preserves_literal_recursive_paths()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let paths: &[&[&str]] = &[&[], &[""], &["default"], &["a/b"], &["a", "b"]];
        let mut encoded = std::collections::BTreeSet::new();
        for components in paths {
            let key = PhysicalIndexKey::text(GraphPath::from_components(components)?, "", "a:\0b");
            let bytes = bincode::serde::encode_to_vec(&key, bincode::config::standard())?;
            let (decoded, consumed): (PhysicalIndexKey, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
            assert_eq!(decoded, key);
            assert_eq!(consumed, bytes.len());
            assert!(encoded.insert(bytes));
        }
        Ok(())
    }

    #[test]
    fn physical_key_wire_rejects_family_shape_and_unbounded_path()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for (family, label) in [
            (0, Some("forbidden")),
            (1, None),
            (2, None),
            (3, Some("Doc")),
        ] {
            let wire = PhysicalIndexKeyWireRef {
                graph: &GraphPath::root(),
                family,
                label,
                property: "body",
            };
            let bytes = bincode::serde::encode_to_vec(wire, bincode::config::standard())?;
            assert!(
                bincode::serde::decode_from_slice::<PhysicalIndexKey, _>(
                    &bytes,
                    bincode::config::standard(),
                )
                .is_err()
            );
        }
        let mut oversized_path = vec![253];
        oversized_path.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            bincode::serde::decode_from_slice::<PhysicalIndexKey, _>(
                &oversized_path,
                bincode::config::standard(),
            )
            .is_err()
        );
        Ok(())
    }

    type PhysicalKeyTestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn physical_key_shape_distinguishes_absent_and_empty_labels() -> PhysicalKeyTestResult {
        use grafeo_common::utils::error::ErrorCode;

        for (family, label) in [
            (PhysicalIndexFamily::Property, Some(String::new())),
            (PhysicalIndexFamily::Text, None),
            (PhysicalIndexFamily::Vector, None),
        ] {
            assert!(matches!(
                PhysicalIndexKey::new(GraphPath::root(), family, label, String::new()),
                Err(error) if error.error_code() == ErrorCode::InvalidInput
            ));
        }
        let keys = [
            PhysicalIndexKey::property(GraphPath::root(), ""),
            PhysicalIndexKey::text(GraphPath::root(), "", ""),
            PhysicalIndexKey::vector(GraphPath::root(), "", ""),
        ];
        for key in keys {
            assert_eq!(key.graph(), &GraphPath::root());
            assert_eq!(key.property_name(), "");
            assert_eq!(
                key.label(),
                (key.family() != PhysicalIndexFamily::Property).then_some(""),
            );
            assert_eq!(
                PhysicalIndexKey::new(
                    key.graph().clone(),
                    key.family(),
                    key.label().map(str::to_owned),
                    key.property_name().to_owned(),
                )?,
                key,
            );
        }
        Ok(())
    }

    #[test]
    fn physical_key_hash_and_order_preserve_every_literal_component() -> PhysicalKeyTestResult {
        use std::collections::{BTreeSet, HashSet};

        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["a", "b"])?,
            GraphPath::from_components(&["a/b"])?,
            GraphPath::from_components(&["nul\0graph", "知識:🧠"])?,
        ];
        let mut expected = Vec::new();
        for graph in paths {
            expected.extend([
                PhysicalIndexKey::property(graph.clone(), "a"),
                PhysicalIndexKey::property(graph.clone(), "b"),
                PhysicalIndexKey::text(graph.clone(), "a", "b:c"),
                PhysicalIndexKey::text(graph.clone(), "a:b", "c"),
                PhysicalIndexKey::vector(graph.clone(), "a", "b:c"),
                PhysicalIndexKey::vector(graph, "a:b", "c"),
            ]);
        }
        let hashed: HashSet<_> = expected.iter().cloned().collect();
        let ordered: BTreeSet<_> = expected.iter().rev().cloned().collect();
        assert_eq!(hashed.len(), expected.len());
        assert_eq!(ordered.into_iter().collect::<Vec<_>>(), expected);
        for key in &expected {
            assert!(hashed.contains(&key.clone()));
        }
        Ok(())
    }

    #[test]
    fn physical_key_requalification_moves_names_without_changing_shape() -> PhysicalKeyTestResult {
        let graph = GraphPath::from_components(&["", "a/b\0🧠"])?;
        let key = PhysicalIndexKey::text(GraphPath::root(), "label\0:🧠", "property\0:知識");
        let label_pointer = key.label().ok_or("Text key lost its label")?.as_ptr();
        let property_pointer = key.property_name().as_ptr();
        let moved = key.with_graph(graph.clone());
        assert_eq!(moved.graph(), &graph);
        assert_eq!(moved.family(), PhysicalIndexFamily::Text);
        assert_eq!(moved.label(), Some("label\0:🧠"));
        assert_eq!(moved.property_name(), "property\0:知識");
        assert_eq!(
            moved.label().ok_or("Text key lost its label")?.as_ptr(),
            label_pointer
        );
        assert_eq!(moved.property_name().as_ptr(), property_pointer);
        Ok(())
    }

    #[test]
    fn encoded_keys_distinguish_colons_in_either_component() {
        let colon_in_label = encode_index_key("tenant:Doc", "embedding");
        let colon_in_property = encode_index_key("tenant", "Doc:embedding");

        assert_ne!(colon_in_label, colon_in_property);
        assert_eq!(
            decode_index_key(&colon_in_label),
            Some(("tenant:Doc", "embedding"))
        );
        assert_eq!(
            decode_index_key(&colon_in_property),
            Some(("tenant", "Doc:embedding"))
        );
    }

    #[test]
    fn encoded_keys_use_utf8_byte_lengths_safely() {
        let key = encode_index_key("知識:文档", "嵌入:🧠");
        assert_eq!(decode_index_key(&key), Some(("知識:文档", "嵌入:🧠")));
    }

    #[test]
    fn only_unambiguous_legacy_keys_are_accepted() {
        assert_eq!(decode_index_key("Doc:body"), Some(("Doc", "body")));
        assert_eq!(decode_index_key(":embedding"), Some(("", "embedding")));
        assert_eq!(decode_index_key("Doc:body:stemmed"), None);
        assert!(index_keys_match(
            "Doc:body",
            &encode_index_key("Doc", "body")
        ));
    }

    #[test]
    fn default_scoped_keys_preserve_the_established_wire_bytes() {
        let established = encode_index_key("知識:Doc", "embedding:🧠");
        assert_eq!(
            encode_scoped_index_key(None, "知識:Doc", "embedding:🧠"),
            established
        );
        assert_eq!(
            decode_scoped_index_key(&established),
            Some((None, "知識:Doc", "embedding:🧠"))
        );
        assert_eq!(
            canonicalize_scoped_index_key("Doc:body"),
            Some(encode_index_key("Doc", "body"))
        );
    }

    #[test]
    fn named_scoped_keys_are_collision_free_for_arbitrary_components() {
        let cases = [
            ("", "", ""),
            ("default", "Doc", "body"),
            ("tenant:west/@idx1:", "知識:文档", "嵌入:🧠"),
            ("nul\0graph", "nul\0label", "nul\0property"),
        ];
        for (graph, label, property) in cases {
            let key = encode_scoped_index_key(Some(graph), label, property);
            assert_eq!(
                decode_scoped_index_key(&key),
                Some((Some(graph), label, property))
            );
            assert_eq!(canonicalize_scoped_index_key(&key), Some(key));
        }

        let default = encode_scoped_index_key(None, "Doc", "body");
        let named = encode_scoped_index_key(Some("default"), "Doc", "body");
        assert_ne!(default, named);
        assert!(!scoped_index_keys_match(&default, &named));
    }

    #[test]
    fn named_scoped_decoder_rejects_noncanonical_or_unsafe_lengths() {
        let canonical = encode_scoped_index_key(Some("é"), "Doc", "body");
        assert_eq!(canonical, "@gidx1:2:é@idx1:3:Docbody");

        for malformed in [
            "@gidx1::@idx1:3:Docbody",
            "@gidx1:00:@idx1:3:Docbody",
            "@gidx1:02:é@idx1:3:Docbody",
            "@gidx1:1:é@idx1:3:Docbody",
            "@gidx1:3:é@idx1:3:Docbody",
            "@gidx1:2:éDoc:body",
            "@gidx1:2:é@idx1:03:Docbody",
            "@gidx1:184467440737095516160:x@idx1:3:Docbody",
        ] {
            assert_eq!(decode_scoped_index_key(malformed), None, "{malformed:?}");
        }
    }

    #[test]
    fn scoped_matching_accepts_only_default_legacy_aliases() {
        let default = encode_scoped_index_key(None, "Doc", "body");
        assert!(scoped_index_keys_match("Doc:body", &default));

        let named = encode_scoped_index_key(Some("g"), "Doc", "body");
        assert!(scoped_index_keys_match(&named, &named));
        assert!(!scoped_index_keys_match("@gidx1:1:gDoc:body", &named));
    }
}
