//! Canonical catalog-owned index mutation ABI.

use grafeo_common::types::{GraphPath, IndexId, MAX_GRAPH_PATH_COMPONENTS};
use grafeo_engine::database::{CreateIndexRequest, IndexCreateKind};

use crate::error::{GrafeoStatus, set_error, set_last_error};
use crate::types::GrafeoDatabase;

/// Borrowed UTF-8 bytes; a null pointer is permitted only for an empty span.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GrafeoUtf8 {
    /// Readable bytes for a nonempty span.
    pub data: *const u8,
    /// Byte length, excluding any optional terminator.
    pub len: usize,
}

/// Canonical index request. All spans remain valid until the call returns.
/// Kind: Property=0, BTree=1, Text=2, Vector=3.
/// Option bits: name=1, label=2, dimensions=4, metric=8, m=16,
/// ef_construction=32, quantization=64, min_token_length=128.
/// Presence is independent of value.
#[repr(C)]
#[derive(Default)]
pub struct GrafeoIndexRequest {
    /// Property=0, BTree=1, Text=2, Vector=3.
    pub kind: u32,
    /// Explicit option-presence bitmask documented above.
    pub options: u32,
    /// Component array; no string is split on separators.
    pub graph: *const GrafeoUtf8,
    /// Zero selects the root graph.
    pub graph_count: usize,
    /// Optional user name (bit 1).
    pub name: GrafeoUtf8,
    /// Optional node label (bit 2).
    pub label: GrafeoUtf8,
    /// Required property name.
    pub property: GrafeoUtf8,
    /// Optional vector metric (bit 8).
    pub metric: GrafeoUtf8,
    /// Optional vector quantization (bit 64).
    pub quantization: GrafeoUtf8,
    /// Optional vector width (bit 4).
    pub dimensions: usize,
    /// Optional HNSW connection count (bit 16).
    pub m: usize,
    /// Optional HNSW construction beam width (bit 32).
    pub ef_construction: usize,
    /// Optional Text minimum token length (bit 128); zero is valid.
    pub min_token_length: usize,
}

fn invalid(message: &str) -> GrafeoStatus {
    set_last_error(message);
    GrafeoStatus::ErrorDatabase
}

fn utf8<'a>(span: GrafeoUtf8) -> Result<&'a str, GrafeoStatus> {
    if span.len == 0 {
        return Ok("");
    }
    if span.data.is_null() {
        set_last_error("Null pointer for nonempty UTF-8 span");
        return Err(GrafeoStatus::ErrorNullPointer);
    }
    if isize::try_from(span.len).is_err() {
        return Err(invalid("UTF-8 span exceeds addressable length"));
    }
    // SAFETY: C caller supplies span.len readable bytes for each nonempty span.
    let bytes = unsafe { std::slice::from_raw_parts(span.data, span.len) };
    std::str::from_utf8(bytes).map_err(|_| {
        set_last_error("Invalid UTF-8 in index request");
        GrafeoStatus::ErrorInvalidUtf8
    })
}

fn decode(request: &GrafeoIndexRequest) -> Result<CreateIndexRequest, GrafeoStatus> {
    if request.options & !255 != 0 || request.kind > 3 {
        return Err(invalid("Unknown index kind or option bits"));
    }
    if request.kind != 3 && request.options & 0x7c != 0 {
        return Err(invalid("Vector options require the Vector index kind"));
    }
    if request.kind != 2 && request.options & 128 != 0 {
        return Err(invalid("min_token_length requires the Text index kind"));
    }
    if request.graph_count > MAX_GRAPH_PATH_COMPONENTS {
        return Err(invalid("Graph path exceeds maximum depth"));
    }
    let components = if request.graph_count == 0 {
        &[][..]
    } else {
        if request.graph.is_null() {
            set_last_error("Null graph component pointer for nonempty path");
            return Err(GrafeoStatus::ErrorNullPointer);
        }
        // SAFETY: C caller supplies graph_count initialized GrafeoUtf8 entries.
        unsafe { std::slice::from_raw_parts(request.graph, request.graph_count) }
    };
    let names = components
        .iter()
        .map(|span| utf8(*span))
        .collect::<Result<Vec<_>, _>>()?;
    let graph = GraphPath::from_components(&names).map_err(|error| invalid(&error.to_string()))?;
    let optional = |bit, span| {
        if request.options & bit == 0 {
            Ok(None)
        } else {
            utf8(span).map(|value| Some(value.to_owned()))
        }
    };
    let kind = match request.kind {
        0 => IndexCreateKind::Property,
        1 => IndexCreateKind::BTree,
        2 => IndexCreateKind::Text {
            min_token_length: (request.options & 128 != 0).then_some(request.min_token_length),
        },
        3 => IndexCreateKind::Vector {
            dimensions: (request.options & 4 != 0).then_some(request.dimensions),
            metric: optional(8, request.metric)?,
            m: (request.options & 16 != 0).then_some(request.m),
            ef_construction: (request.options & 32 != 0).then_some(request.ef_construction),
            ef: None,
            quantization: optional(64, request.quantization)?,
        },
        _ => return Err(invalid("Unknown index kind")),
    };
    Ok(CreateIndexRequest {
        graph,
        name: optional(1, request.name)?,
        label: optional(2, request.label)?,
        property: utf8(request.property)?.to_owned(),
        kind,
    })
}

/// Creates one catalog-owned index. Writes the owner only on success.
/// All pointers must be aligned, live and valid for their declared lengths.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_create_index(
    db: *mut GrafeoDatabase,
    request: *const GrafeoIndexRequest,
    out_id: *mut u32,
) -> GrafeoStatus {
    if db.is_null() || request.is_null() || out_id.is_null() {
        set_last_error("Null database, index request, or output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: C caller supplies live aligned database and request pointers.
    let (db, request) = unsafe { (&*db, &*request) };
    let request = match decode(request) {
        Ok(request) => request,
        Err(error) => return error,
    };
    match db.inner.read().create_index(request) {
        Ok(id) => {
            // SAFETY: C caller supplies a writable u32 output pointer.
            unsafe { *out_id = id.as_u32() };
            GrafeoStatus::Ok
        }
        Err(error) => set_error(&error),
    }
}

/// Drops exactly one catalog owner; missing owners succeed with out_dropped=0.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_drop_index(
    db: *mut GrafeoDatabase,
    id: u32,
    out_dropped: *mut i32,
) -> GrafeoStatus {
    if db.is_null() || out_dropped.is_null() {
        set_last_error("Null database or drop output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: C caller supplies a live database pointer.
    let db = unsafe { &*db };
    match db.inner.read().drop_index(IndexId::new(id)) {
        Ok(dropped) => {
            // SAFETY: C caller supplies a writable i32 output pointer.
            unsafe { *out_dropped = i32::from(dropped) };
            GrafeoStatus::Ok
        }
        Err(error) => set_error(&error),
    }
}

/// Rebuilds exactly one catalog owner. A missing owner is an error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_rebuild_index(db: *mut GrafeoDatabase, id: u32) -> GrafeoStatus {
    if db.is_null() {
        set_last_error("Null database pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: C caller supplies a live database pointer.
    let db = unsafe { &*db };
    match db.inner.read().rebuild_index(IndexId::new(id)) {
        Ok(()) => GrafeoStatus::Ok,
        Err(error) => set_error(&error),
    }
}

#[cfg(test)]
mod tests;
