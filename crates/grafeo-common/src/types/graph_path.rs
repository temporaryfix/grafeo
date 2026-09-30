//! Canonical component-qualified graph identity shared by durable formats.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::MAX_WORLD_GRAPH_NAME_BYTES;

/// Maximum nesting depth of a graph identity.
pub const MAX_GRAPH_PATH_COMPONENTS: usize = 256;

/// A root-relative sequence of graph names, ordered by components.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphPath {
    components: Vec<String>,
}

/// A graph identity could not be constructed or decoded.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphPathError {
    /// Too many nested graph names.
    #[error("graph path exceeds {MAX_GRAPH_PATH_COMPONENTS} components")]
    TooDeep,
    /// A name exceeds the shared graph-name byte limit.
    #[error("graph path component {index} exceeds {MAX_WORLD_GRAPH_NAME_BYTES} bytes")]
    NameTooLong {
        /// Zero-based component position.
        index: usize,
    },
    /// The enclosing format's remaining byte budget was exceeded.
    #[error("graph path exceeds its {maximum} byte budget")]
    ByteLimit {
        /// Remaining byte budget supplied by the enclosing format.
        maximum: usize,
    },
    /// A fixed-width field or name is incomplete.
    #[error("truncated graph path")]
    Truncated,
    /// A graph name is not UTF-8.
    #[error("graph path component {index} is not UTF-8")]
    InvalidUtf8 {
        /// Zero-based component position.
        index: usize,
    },
    /// Bytes follow the complete identity.
    #[error("trailing bytes after graph path")]
    TrailingBytes,
    /// Memory for a validated identity could not be reserved.
    #[error("cannot allocate graph path")]
    Allocation,
}

impl GraphPath {
    /// The default graph has no name components.
    #[must_use]
    pub const fn root() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Constructs an identity without interpreting names as separators.
    ///
    /// # Errors
    /// Returns an error for excessive depth/name length or allocation failure.
    pub fn from_components(components: &[&str]) -> Result<Self, GraphPathError> {
        if components.len() > MAX_GRAPH_PATH_COMPONENTS {
            return Err(GraphPathError::TooDeep);
        }
        // Validate all names before allocating any owned state.
        for (index, name) in components.iter().enumerate() {
            if name.len() > MAX_WORLD_GRAPH_NAME_BYTES {
                return Err(GraphPathError::NameTooLong { index });
            }
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(components.len())
            .map_err(|_| GraphPathError::Allocation)?;
        for name in components {
            let mut value = String::new();
            value
                .try_reserve_exact(name.len())
                .map_err(|_| GraphPathError::Allocation)?;
            value.push_str(name);
            owned.push(value);
        }
        Ok(Self { components: owned })
    }

    /// Returns the individual names, with no normalization or escaping.
    #[must_use]
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// Appends one named child, including the legal empty name.
    ///
    /// # Errors
    /// Returns an error if the resulting identity exceeds a limit or cannot allocate.
    pub fn child(&self, name: &str) -> Result<Self, GraphPathError> {
        let mut names = [""; MAX_GRAPH_PATH_COMPONENTS];
        for (slot, component) in names.iter_mut().zip(&self.components) {
            *slot = component.as_str();
        }
        let index = self.components.len();
        *names.get_mut(index).ok_or(GraphPathError::TooDeep)? = name;
        Self::from_components(&names[..=index])
    }

    /// Returns the parent, or `None` for the root.
    ///
    /// # Errors
    /// Returns an error if the parent identity cannot allocate.
    pub fn parent(&self) -> Result<Option<Self>, GraphPathError> {
        let Some(depth) = self.components.len().checked_sub(1) else {
            return Ok(None);
        };
        let mut names = [""; MAX_GRAPH_PATH_COMPONENTS];
        for (slot, component) in names.iter_mut().zip(&self.components).take(depth) {
            *slot = component.as_str();
        }
        Self::from_components(&names[..depth]).map(Some)
    }

    /// Encodes this identity within the enclosing format's byte budget.
    ///
    /// # Errors
    /// Returns an error if the budget is exceeded or allocation fails.
    pub fn to_bytes(&self, maximum: usize) -> Result<Vec<u8>, GraphPathError> {
        let length = self.components.iter().try_fold(4_usize, |total, name| {
            total
                .checked_add(4)
                .and_then(|total| total.checked_add(name.len()))
                .ok_or(GraphPathError::ByteLimit { maximum })
        })?;
        if length > maximum {
            return Err(GraphPathError::ByteLimit { maximum });
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| GraphPathError::Allocation)?;
        let count = u32::try_from(self.components.len()).map_err(|_| GraphPathError::TooDeep)?;
        bytes.extend_from_slice(&count.to_le_bytes());
        for (index, name) in self.components.iter().enumerate() {
            let length =
                u32::try_from(name.len()).map_err(|_| GraphPathError::NameTooLong { index })?;
            bytes.extend_from_slice(&length.to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
        }
        Ok(bytes)
    }

    /// Decodes exactly one identity, validating the entire frame before allocation.
    ///
    /// `maximum` is the enclosing section/snapshot's remaining byte budget.
    ///
    /// # Errors
    /// Rejects malformed, oversized, non-UTF-8, or trailing data and allocation failure.
    pub fn from_bytes(bytes: &[u8], maximum: usize) -> Result<Self, GraphPathError> {
        if bytes.len() > maximum {
            return Err(GraphPathError::ByteLimit { maximum });
        }
        let mut remaining = bytes;
        let count =
            usize::try_from(read_u32(&mut remaining)?).map_err(|_| GraphPathError::TooDeep)?;
        if count > MAX_GRAPH_PATH_COMPONENTS {
            return Err(GraphPathError::TooDeep);
        }
        // A fixed-size borrowed staging area prevents hostile fields from
        // driving an allocation before every component and the frame is valid.
        let mut names = [""; MAX_GRAPH_PATH_COMPONENTS];
        for (index, slot) in names.iter_mut().take(count).enumerate() {
            let length = usize::try_from(read_u32(&mut remaining)?)
                .map_err(|_| GraphPathError::NameTooLong { index })?;
            if length > MAX_WORLD_GRAPH_NAME_BYTES {
                return Err(GraphPathError::NameTooLong { index });
            }
            let (name, rest) = remaining
                .split_at_checked(length)
                .ok_or(GraphPathError::Truncated)?;
            *slot = std::str::from_utf8(name).map_err(|_| GraphPathError::InvalidUtf8 { index })?;
            remaining = rest;
        }
        if !remaining.is_empty() {
            return Err(GraphPathError::TrailingBytes);
        }
        Self::from_components(&names[..count])
    }
}

fn read_u32(bytes: &mut &[u8]) -> Result<u32, GraphPathError> {
    let (value, remaining) = bytes
        .split_first_chunk::<4>()
        .ok_or(GraphPathError::Truncated)?;
    *bytes = remaining;
    Ok(u32::from_le_bytes(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_identity_distinguishes_root_empty_name_and_component_boundaries() {
        let cases: &[(&[&str], &[u8])] = &[
            (&[], &[0, 0, 0, 0]),
            (&[""], &[1, 0, 0, 0, 0, 0, 0, 0]),
            (
                &["a", "b"],
                &[2, 0, 0, 0, 1, 0, 0, 0, b'a', 1, 0, 0, 0, b'b'],
            ),
            (&["a/b"], &[1, 0, 0, 0, 3, 0, 0, 0, b'a', b'/', b'b']),
            (
                &["é", ""],
                &[2, 0, 0, 0, 2, 0, 0, 0, 0xc3, 0xa9, 0, 0, 0, 0],
            ),
        ];
        for &(names, bytes) in cases {
            let path = GraphPath::from_components(names).unwrap();
            assert_eq!(path.to_bytes(bytes.len()).unwrap(), bytes);
            assert_eq!(GraphPath::from_bytes(bytes, bytes.len()).unwrap(), path);
        }
        assert_ne!(GraphPath::root(), GraphPath::root().child("").unwrap());
    }

    #[test]
    fn component_order_not_length_prefix_order_controls_canonical_sort() {
        let mut paths: Vec<_> = [&["z"][..], &["aa"], &["a", ""], &["a"], &[]]
            .into_iter()
            .map(|names| GraphPath::from_components(names).unwrap())
            .collect();
        paths.sort();
        let names: Vec<_> = paths.iter().map(GraphPath::components).collect();
        assert_eq!(
            names,
            vec![
                Vec::<String>::new(),
                vec!["a".into()],
                vec!["a".into(), String::new()],
                vec!["aa".into()],
                vec!["z".into()]
            ]
        );
    }

    #[test]
    fn child_and_parent_preserve_names_and_reject_excess_depth() {
        let root = GraphPath::root();
        assert_eq!(root.parent().unwrap(), None);
        let child = root.child("a/b\0é").unwrap().child("").unwrap();
        assert_eq!(child.components(), &["a/b\0é", ""]);
        assert_eq!(child.parent().unwrap(), Some(root.child("a/b\0é").unwrap()));
        let deepest = GraphPath::from_components(&[""; 256]).unwrap();
        assert_eq!(deepest.child(""), Err(GraphPathError::TooDeep));
        assert_eq!(
            GraphPath::from_components(&[""; 257]),
            Err(GraphPathError::TooDeep)
        );
        let bytes = deepest.to_bytes(1028).unwrap();
        assert_eq!(GraphPath::from_bytes(&bytes, 1028).unwrap(), deepest);
    }

    #[test]
    fn utf8_byte_length_limit_is_enforced_by_construction_and_decode() {
        let legal = "é".repeat(32_768);
        let path = GraphPath::root().child(&legal).unwrap();
        let bytes = path.to_bytes(65_544).unwrap();
        assert_eq!(GraphPath::from_bytes(&bytes, bytes.len()).unwrap(), path);
        let excessive = format!("{legal}a");
        assert_eq!(
            GraphPath::root().child(&excessive),
            Err(GraphPathError::NameTooLong { index: 0 })
        );
        assert_eq!(
            GraphPath::from_components(&["", &excessive]),
            Err(GraphPathError::NameTooLong { index: 1 })
        );
        // Advertised oversized length must fail before attempting to read/allocate it.
        assert_eq!(
            GraphPath::from_bytes(&[1, 0, 0, 0, 1, 0, 1, 0], 8),
            Err(GraphPathError::NameTooLong { index: 0 })
        );
    }

    #[test]
    fn malformed_frames_fail_with_structured_errors() {
        for bytes in [
            &[][..],
            &[0],
            &[0, 0, 0],
            &[1, 0, 0, 0],
            &[1, 0, 0, 0, 1, 0, 0, 0],
        ] {
            assert_eq!(
                GraphPath::from_bytes(bytes, 100),
                Err(GraphPathError::Truncated)
            );
        }
        assert_eq!(
            GraphPath::from_bytes(&[0, 0, 0, 0, 0], 100),
            Err(GraphPathError::TrailingBytes)
        );
        assert_eq!(
            GraphPath::from_bytes(&[1, 0, 0, 0, 1, 0, 0, 0, 0xff], 100),
            Err(GraphPathError::InvalidUtf8 { index: 0 })
        );
        assert_eq!(
            GraphPath::from_bytes(&[1, 1, 0, 0], 100),
            Err(GraphPathError::TooDeep)
        );
        assert_eq!(
            GraphPath::from_bytes(&[255, 255, 255, 255], 100),
            Err(GraphPathError::TooDeep)
        );
    }

    #[test]
    fn enclosing_byte_budget_is_required_for_read_and_write() {
        assert_eq!(
            GraphPath::root().to_bytes(3),
            Err(GraphPathError::ByteLimit { maximum: 3 })
        );
        assert_eq!(
            GraphPath::from_bytes(&[0, 0, 0, 0], 3),
            Err(GraphPathError::ByteLimit { maximum: 3 })
        );
        let path = GraphPath::root().child("ab").unwrap();
        assert_eq!(
            path.to_bytes(9),
            Err(GraphPathError::ByteLimit { maximum: 9 })
        );
        assert_eq!(
            path.to_bytes(10).unwrap(),
            [1, 0, 0, 0, 2, 0, 0, 0, b'a', b'b']
        );
    }
}
