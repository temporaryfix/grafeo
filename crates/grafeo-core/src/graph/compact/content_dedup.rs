//! In-memory content-addressed block pool plus persistence backends for
//! cross-time (cross-generation) structural sharing of the temporal cold base.
//!
//! A column's serialized value block is hashed to a [`grafeo_common::types::ContentId`]
//! ([`crate::graph::compact::content_hash::physical_block_hash`]). Across base generations an *unchanged* column hashes
//! to the same id, so a new generation need only store the blocks whose ids are
//! new — the unchanged blocks are shared by reference (the "compress version
//! history across time" density win).
//!
//! The dedup unit is a **whole column's serialized value block** — per-column
//! granularity. Sub-column dedup (sharing individual physical 1024-row blocks
//! within a column) is a deliberate non-goal: a column either hashes identically
//! across generations and is shared whole, or it differs and is stored whole.
//!
//! [`crate::graph::compact::content_dedup::BlockPool`] holds those blocks in memory, keyed by content id. Durable
//! sharing is provided by the persistence backends: the synchronous
//! [`crate::graph::compact::content_dedup::BlockBackend`] trait (native filesystem via [`crate::graph::compact::content_dedup::FsBlockBackend`]), whole-pool
//! serialization ([`crate::graph::compact::content_dedup::BlockPool::to_bytes`]/[`crate::graph::compact::content_dedup::BlockPool::from_bytes`]), and
//! incremental per-block persistence ([`crate::graph::compact::content_dedup::BlockPool::persist_to`] /
//! [`crate::graph::compact::content_dedup::BlockPool::load_from`]). The block producer is the content-addressed
//! serialization mode in [`crate::graph::compact::section`]; this module is the pool and
//! its durable backends, not a second block-id scheme.

use std::sync::Arc;

use grafeo_common::types::ContentId;
use grafeo_common::utils::hash::FxHashMap;

use super::content_hash::physical_block_hash;

/// An in-memory content-addressed block pool: column value blocks keyed by their
/// physical content id, stored once and shared across base generations.
///
/// Interning a block from a new generation is a no-op when an identical block
/// (same [`ContentId`]) was already stored by a prior generation — the on-disk
/// realization of cross-time structural sharing. The pool is backend-agnostic
/// (a persistence layer maps `ContentId -> bytes`).
#[derive(Clone, Debug, Default)]
pub struct BlockPool {
    blocks: FxHashMap<ContentId, Arc<[u8]>>,
}

impl BlockPool {
    /// An empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns a block, returning its content id. The bytes are stored only if
    /// the id is new — an unchanged block from a prior generation is reused, not
    /// re-stored (cross-time dedup).
    pub fn intern(&mut self, bytes: Vec<u8>) -> ContentId {
        let id = physical_block_hash(&bytes);
        self.blocks
            .entry(id)
            .or_insert_with(|| Arc::from(bytes.into_boxed_slice()));
        id
    }

    /// Whether a block with this id is already stored.
    #[must_use]
    pub fn contains(&self, id: ContentId) -> bool {
        self.blocks.contains_key(&id)
    }

    /// The raw bytes of a stored block.
    #[must_use]
    pub fn get(&self, id: ContentId) -> Option<&[u8]> {
        self.blocks.get(&id).map(|b| &**b)
    }

    /// Number of distinct blocks stored.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total bytes of all distinct stored blocks.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.blocks.values().map(|b| b.len()).sum()
    }

    /// Iterates the stored `(content id, block bytes)` pairs — for a persistence
    /// backend that writes each block under its id, skipping ids it already holds.
    pub fn blocks(&self) -> impl Iterator<Item = (ContentId, &[u8])> {
        self.blocks.iter().map(|(id, block)| (*id, &**block))
    }

    /// Serializes the pool to a deterministic blob (blocks sorted by content id,
    /// each `[32-byte id][u32 len][bytes]`) for whole-pool persistence. A backend
    /// targeting incremental on-disk dedup can instead persist per-block via
    /// [`blocks`](Self::blocks) and [`contains`](Self::contains).
    ///
    /// # Panics
    ///
    /// Panics if the pool holds more than `u32::MAX` blocks or a single block
    /// exceeds `u32::MAX` bytes — neither occurs for realistic stores.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut entries: Vec<(&ContentId, &Arc<[u8]>)> = self.blocks.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut buf = Vec::new();
        let count = u32::try_from(entries.len()).expect("block count exceeds u32::MAX");
        buf.extend_from_slice(&count.to_le_bytes());
        for (id, block) in entries {
            buf.extend_from_slice(id.as_bytes());
            let len = u32::try_from(block.len()).expect("block exceeds u32::MAX");
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(block);
        }
        buf
    }

    /// Reconstructs a pool from [`to_bytes`](Self::to_bytes) output.
    ///
    /// # Errors
    ///
    /// Returns an error if the blob is truncated or malformed.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut pos = 0;
        let count = read_u32_le(data, &mut pos)? as usize;
        let mut blocks: FxHashMap<ContentId, Arc<[u8]>> =
            FxHashMap::with_capacity_and_hasher(count, Default::default());
        for _ in 0..count {
            let id_end = pos.checked_add(32).ok_or("offset overflow")?;
            let id_slice = data.get(pos..id_end).ok_or("truncated content id")?;
            let id_arr: [u8; 32] = id_slice.try_into().map_err(|_| "content id length")?;
            let id = ContentId::from_bytes(id_arr);
            pos = id_end;
            let len = read_u32_le(data, &mut pos)? as usize;
            let end = pos.checked_add(len).ok_or("offset overflow")?;
            let block = data.get(pos..end).ok_or("truncated block body")?;
            blocks.insert(id, Arc::from(block));
            pos = end;
        }
        Ok(Self { blocks })
    }

    /// Persists this pool's blocks to a durable [`BlockBackend`], writing only
    /// the blocks whose id the backend does **not** already
    /// [`contains`](BlockBackend::contains) — the incremental on-disk dedup. A
    /// later base generation that re-interns unchanged columns therefore writes
    /// nothing for them; only the changed blocks reach the backend. Returns the
    /// number of blocks newly written (the ones not previously held).
    ///
    /// This is WASM-safe: it routes entirely through the backend trait and uses
    /// no `std::fs`.
    ///
    /// # Errors
    ///
    /// Returns the first backend error from a `contains`/`put` call.
    pub fn persist_to(&self, backend: &impl BlockBackend) -> Result<usize, String> {
        let mut written = 0;
        for (id, block) in self.blocks() {
            if !backend.contains(id)? {
                backend.put(id, block)?;
                written += 1;
            }
        }
        Ok(written)
    }

    /// Loads a fresh pool from a durable [`BlockBackend`], pulling the blocks for
    /// `ids`. The inverse of [`persist_to`](Self::persist_to). Repeated ids are
    /// loaded once. WASM-safe: routes entirely through the backend trait.
    ///
    /// # Errors
    ///
    /// Returns a backend error, or `"missing block <hex id>"` if the backend does
    /// not hold a requested id.
    pub fn load_from(backend: &impl BlockBackend, ids: &[ContentId]) -> Result<Self, String> {
        let mut pool = Self::new();
        for &id in ids {
            if pool.blocks.contains_key(&id) {
                continue;
            }
            let bytes = backend
                .get(id)?
                .ok_or_else(|| format!("missing block {}", hex_lower(id.as_bytes())))?;
            pool.blocks.insert(id, Arc::from(bytes.into_boxed_slice()));
        }
        Ok(pool)
    }
}

/// Lowercase-hex encoding of a content id's raw bytes — the
/// content-addressed file name a [`FsBlockBackend`] stores a block under (and the
/// identifier in `load_from`'s "missing block" error). Pure, WASM-safe.
#[must_use]
fn hex_lower(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// A native filesystem [`BlockBackend`]: each block is one file in a directory,
/// named by the lowercase hex of its [`ContentId`] (64 hex chars). Because the
/// name *is* the content address, `put` is idempotent — a block already on disk
/// is skipped — which gives [`BlockPool::persist_to`] its incremental on-disk
/// dedup: a new generation writes only the files whose ids are new.
///
/// Native-only (`std::fs`); gated with `#[cfg(not(target_arch = "wasm32"))]` so
/// the wasm build is unaffected. The browser-side counterpart lives in the wasm
/// crate (`grafeo-wasm`, `opfs` feature) as async OPFS (Origin-Private File
/// System) free functions — the same content-addressed naming (one file per id),
/// but NOT this sync [`BlockBackend`] trait, since OPFS access is inherently
/// asynchronous.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct FsBlockBackend {
    /// Directory holding one file per block, named by lowercase-hex content id.
    dir: std::path::PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl FsBlockBackend {
    /// Opens (creating if needed) a filesystem block store rooted at `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dir` could not be created.
    pub fn open(dir: impl Into<std::path::PathBuf>) -> Result<Self, String> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("create_dir_all {}: {e}", dir.display()))?;
        Ok(Self { dir })
    }

    /// The on-disk path for a block id: `dir/<lowercase-hex-id>`.
    fn path_for(&self, id: ContentId) -> std::path::PathBuf {
        self.dir.join(hex_lower(id.as_bytes()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl BlockBackend for FsBlockBackend {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), String> {
        let path = self.path_for(id);
        // Content-addressed: the file name fixes the bytes, so if it already
        // exists the content is identical — skip the write (idempotent).
        if path.exists() {
            return Ok(());
        }
        std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
    }

    fn get(&self, id: ContentId) -> Result<Option<Vec<u8>>, String> {
        let path = self.path_for(id);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    fn contains(&self, id: ContentId) -> Result<bool, String> {
        Ok(self.path_for(id).exists())
    }
}

/// A durable, content-addressed block store: a backend that maps a block's
/// [`ContentId`] to its bytes.
///
/// Backends are **content-addressed**: a block's id is fixed by its bytes (the
/// physical block hash), so every `put` is *idempotent* — writing a block whose
/// id is already held is a no-op, and any backend that holds an id holds the
/// exact bytes that hash to it. This is what makes [`BlockPool::persist_to`]
/// incremental: a new base generation writes only the blocks the backend does
/// not already [`contains`](BlockBackend::contains), sharing the rest by id.
///
/// This is the **synchronous** backend contract — the native filesystem
/// ([`FsBlockBackend`]). The browser-side OPFS (Origin-Private File System)
/// counterpart mirrors the same content-addressed semantics (one file per id,
/// same naming) but is exposed as **async free functions in the wasm crate**, not
/// through this trait: OPFS access is inherently async and cannot satisfy these
/// synchronous methods. Errors are reported as `Result<_, String>` so any backend
/// can carry its own failure detail without a shared error type.
pub trait BlockBackend {
    /// Stores `bytes` under `id`. Idempotent: because backends are
    /// content-addressed, if `id` is already [`contains`](Self::contains)ed the
    /// stored bytes are identical, so an implementation may skip the write.
    ///
    /// # Errors
    ///
    /// Returns an error if the block could not be durably stored.
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), String>;

    /// Reads the block stored under `id`, or `None` if the backend does not hold
    /// it.
    ///
    /// # Errors
    ///
    /// Returns an error if a held block could not be read back.
    fn get(&self, id: ContentId) -> Result<Option<Vec<u8>>, String>;

    /// Whether the backend already holds a block with this id.
    ///
    /// # Errors
    ///
    /// Returns an error if existence could not be determined.
    fn contains(&self, id: ContentId) -> Result<bool, String>;
}

/// Reads a little-endian `u32` at `pos`, advancing it.
fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let end = pos.checked_add(4).ok_or("offset overflow")?;
    let slice = data.get(*pos..end).ok_or("truncated u32")?;
    *pos = end;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("4-byte u32 slice"),
    ))
}

#[cfg(test)]
mod tests {
    use super::BlockPool;

    #[test]
    fn block_pool_serializes_and_reloads() {
        let mut pool = BlockPool::new();
        let id1 = pool.intern(vec![1, 2, 3, 4]);
        let id2 = pool.intern(vec![5, 6, 7]);
        let blob = pool.to_bytes();

        let reloaded = BlockPool::from_bytes(&blob).unwrap();
        assert_eq!(reloaded.block_count(), 2);
        assert_eq!(reloaded.get(id1), Some(&[1, 2, 3, 4][..]));
        assert_eq!(reloaded.get(id2), Some(&[5, 6, 7][..]));
        // Deterministic and idempotent.
        assert_eq!(reloaded.to_bytes(), blob);
        // Per-block enumeration exposes both ids (for incremental backends).
        let ids: Vec<_> = pool.blocks().map(|(id, _)| id).collect();
        assert!(ids.contains(&id1) && ids.contains(&id2));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn block_pool_round_trips_through_fs_backend() {
        use super::FsBlockBackend;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let backend = FsBlockBackend::open(dir.path()).unwrap();

        // Intern several blocks into a pool.
        let mut pool = BlockPool::new();
        let payloads: Vec<Vec<u8>> = vec![
            vec![1, 2, 3, 4],
            vec![9, 9, 9],
            (0u8..32).collect(),
            Vec::new(), // empty block round-trips too
        ];
        let ids: Vec<_> = payloads.iter().map(|p| pool.intern(p.clone())).collect();
        assert_eq!(pool.block_count(), 4);

        // Persist all blocks to disk (all new).
        let written = pool.persist_to(&backend).unwrap();
        assert_eq!(written, 4);

        // Load into a fresh pool from the backend by id; bytes/count match.
        let reloaded = BlockPool::load_from(&backend, &ids).unwrap();
        assert_eq!(reloaded.block_count(), pool.block_count());
        for (id, payload) in ids.iter().zip(&payloads) {
            assert_eq!(reloaded.get(*id), Some(payload.as_slice()));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn persist_to_is_incremental_dedup() {
        use super::{BlockBackend, FsBlockBackend};
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let backend = FsBlockBackend::open(dir.path()).unwrap();

        let mut pool = BlockPool::new();
        pool.intern(vec![1, 2, 3]);
        pool.intern(vec![4, 5, 6, 7]);
        pool.intern(vec![8]);

        // First persist writes every block.
        assert_eq!(pool.persist_to(&backend).unwrap(), 3);

        // Persisting the same pool again writes zero new files: all ids already
        // on disk (content-addressed idempotence).
        assert_eq!(pool.persist_to(&backend).unwrap(), 0);

        // A second generation that shares two blocks and adds one new block
        // writes exactly one file.
        let mut gen2 = BlockPool::new();
        gen2.intern(vec![1, 2, 3]); // shared
        gen2.intern(vec![4, 5, 6, 7]); // shared
        gen2.intern(vec![100, 101]); // new
        assert_eq!(gen2.persist_to(&backend).unwrap(), 1);

        // Backend still holds exactly the file per distinct block id, and each
        // distinct id reports contained.
        for (id, _) in pool.blocks() {
            assert!(backend.contains(id).unwrap());
        }
        // The on-disk file count equals the number of distinct blocks (4).
        let file_count = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(file_count, 4);
    }
}
