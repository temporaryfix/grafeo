//! OPFS (Origin-Private File System) persistence backend for the
//! content-addressed [`BlockPool`] — the browser counterpart of grafeo-core's
//! native `FsBlockBackend`.
//!
//! Each block is stored as one OPFS file whose name is the lowercase hex of the
//! block's [`ContentId`] (64 hex chars). Because the name *is* the content
//! address, writes are idempotent: a block already present is skipped, giving the
//! same incremental cross-generation dedup as the native filesystem backend
//! (write only the blocks whose ids are new).
//!
//! OPFS is inherently asynchronous, so this backend does **not** implement
//! grafeo-core's synchronous `BlockBackend` trait. Instead it exposes free async
//! functions ([`persist_pool_to_opfs`], [`load_pool_from_opfs`]) that drive the
//! main-thread-compatible OPFS API:
//!
//! - `navigator.storage.getDirectory()` -> [`FileSystemDirectoryHandle`]
//! - per-block write: `getFileHandle(name, { create: true })` ->
//!   `createWritable()` -> `write(bytes)` -> `close()`
//! - per-block read: `getFileHandle(name)` -> `getFile()` -> `arrayBuffer()`
//!
//! # Build requirement (web-sys unstable APIs)
//!
//! In the web-sys version this crate resolves to (0.3.x at time of writing), the
//! File System Access write bindings (`FileSystemWritableFileStream`,
//! `createWritable`, `write_with_buffer_source`) are **stable**, so this module
//! compiles for `wasm32-unknown-unknown` with `--features opfs` and **no extra
//! rustflags**. If a future/older web-sys re-gates those bindings behind the
//! unstable-apis flag, build with:
//!
//! ```text
//! RUSTFLAGS="--cfg=web_sys_unstable_apis" cargo build \
//!     --target wasm32-unknown-unknown -p grafeo-wasm --features opfs
//! ```
//!
//! # Verification status
//!
//! **Runtime-verified in a browser.** The [`opfs_selftest`] `#[wasm_bindgen]`
//! export exercises the full round-trip against real OPFS: persist a small pool
//! (including an empty-block edge case) to the origin-private filesystem, reload
//! it, and assert every block is byte-identical, plus that a second persist
//! writes nothing (incremental content-addressed dedup). It was run in Chrome on
//! 2026-06-23 and passed:
//! `3 blocks (incl. empty) persisted + reloaded byte-identical; 2nd persist wrote 0`.
//!
//! To re-run it, build for the web with the feature
//! (`wasm-pack build --target web -- --features opfs`) and call `opfs_selftest()`
//! from a page served over a secure context (`http://localhost` counts).

use grafeo_common::types::ContentId;
use grafeo_core::graph::compact::content_dedup::BlockPool;
use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    File, FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemWritableFileStream,
};

/// Lowercase-hex encoding of a 32-byte content id (64 hex chars). Mirrors the
/// content-addressed file name used by the native `FsBlockBackend` so the OPFS
/// store is layout-compatible with the filesystem one.
fn hex_lower(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Opens (creating if needed) the OPFS subdirectory `dir_name` under the
/// origin-private root, returning its directory handle.
///
/// # Errors
///
/// Returns the underlying `JsValue` if the global `navigator.storage` is
/// unavailable, the root directory cannot be obtained, or the subdirectory
/// cannot be created/opened.
async fn open_dir(dir_name: &str) -> Result<FileSystemDirectoryHandle, JsValue> {
    // navigator.storage.getDirectory() -> Promise<FileSystemDirectoryHandle>
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no global `window`"))?;
    let storage = window.navigator().storage();
    let root: FileSystemDirectoryHandle =
        JsFuture::from(storage.get_directory()).await?.dyn_into()?;

    // getDirectoryHandle(dir_name, { create: true }) -> Promise<FileSystemDirectoryHandle>
    let opts = web_sys::FileSystemGetDirectoryOptions::new();
    opts.set_create(true);
    let dir: FileSystemDirectoryHandle =
        JsFuture::from(root.get_directory_handle_with_options(dir_name, &opts))
            .await?
            .dyn_into()?;
    Ok(dir)
}

/// Whether `dir` already holds a file named `name` (an existing content-addressed
/// block). Resolving the file handle without `{ create: true }` rejects with a
/// `NotFoundError` when absent, which we map to `false`.
async fn file_exists(dir: &FileSystemDirectoryHandle, name: &str) -> Result<bool, JsValue> {
    // Absent is signalled by a `NotFoundError` DOMException. Any OTHER rejection
    // (quota exceeded, permission denied, transient I/O) must surface rather than
    // be misread as "absent" — which would trigger a needless re-write or mask a
    // real failure.
    match JsFuture::from(dir.get_file_handle(name)).await {
        Ok(_) => Ok(true),
        Err(e) => {
            let not_found = e
                .dyn_ref::<web_sys::DomException>()
                .is_some_and(|ex| ex.name() == "NotFoundError");
            if not_found { Ok(false) } else { Err(e) }
        }
    }
}

/// Writes `bytes` to a freshly-(re)created file `name` in `dir` via a writable
/// stream: `getFileHandle(name, { create: true })` -> `createWritable()` ->
/// `write(bytes)` -> `close()`.
///
/// # Errors
///
/// Returns the first `JsValue` rejection from any step of the write pipeline.
async fn write_file(
    dir: &FileSystemDirectoryHandle,
    name: &str,
    bytes: &[u8],
) -> Result<(), JsValue> {
    let opts = FileSystemGetFileOptions::new();
    opts.set_create(true);
    let handle: FileSystemFileHandle =
        JsFuture::from(dir.get_file_handle_with_options(name, &opts))
            .await?
            .dyn_into()?;

    let writable: FileSystemWritableFileStream =
        JsFuture::from(handle.create_writable()).await?.dyn_into()?;

    // Copy into a JS Uint8Array; write_with_u8_array returns a Promise.
    let view = Uint8Array::new_with_length(
        u32::try_from(bytes.len())
            .map_err(|_| JsValue::from_str("block exceeds u32::MAX bytes"))?,
    );
    view.copy_from(bytes);
    JsFuture::from(writable.write_with_buffer_source(&view)?).await?;
    JsFuture::from(writable.close()).await?;
    Ok(())
}

/// Reads file `name` from `dir` into a `Vec<u8>`:
/// `getFileHandle(name)` -> `getFile()` -> `arrayBuffer()`.
///
/// # Errors
///
/// Returns the `JsValue` rejection if the file is missing or unreadable.
async fn read_file(dir: &FileSystemDirectoryHandle, name: &str) -> Result<Vec<u8>, JsValue> {
    let handle: FileSystemFileHandle = JsFuture::from(dir.get_file_handle(name))
        .await?
        .dyn_into()?;
    let file: File = JsFuture::from(handle.get_file()).await?.dyn_into()?;
    let buffer: ArrayBuffer = JsFuture::from(file.array_buffer()).await?.dyn_into()?;
    let view = Uint8Array::new(&buffer);
    Ok(view.to_vec())
}

/// Persists every block in `pool` to the OPFS subdirectory `dir_name`, one file
/// per block named by the lowercase hex of its [`ContentId`]. A block whose file
/// already exists is skipped (content-addressed idempotence), so this mirrors the
/// native backend's incremental cross-generation dedup: a later generation that
/// re-interns unchanged columns writes nothing for them.
///
/// Returns the number of blocks **newly written** (those not already present).
///
/// To avoid holding a borrow of `pool` across `.await` points, the
/// `(ContentId, Vec<u8>)` pairs are collected up front.
///
/// # Errors
///
/// Returns the first `JsValue` rejection from opening the directory or from any
/// existence-check / write.
pub async fn persist_pool_to_opfs(pool: &BlockPool, dir_name: &str) -> Result<usize, JsValue> {
    // Collect owned copies so no pool borrow is held across awaits.
    let blocks: Vec<(ContentId, Vec<u8>)> = pool
        .blocks()
        .map(|(id, bytes)| (id, bytes.to_vec()))
        .collect();

    let dir = open_dir(dir_name).await?;
    let mut written = 0usize;
    for (id, bytes) in blocks {
        let name = hex_lower(id.as_bytes());
        if file_exists(&dir, &name).await? {
            continue; // already stored (idempotent, content-addressed)
        }
        write_file(&dir, &name, &bytes).await?;
        written += 1;
    }
    Ok(written)
}

/// Loads a fresh [`BlockPool`] from the OPFS subdirectory `dir_name`, reading the
/// file for each id in `ids` and `intern`ing its bytes (which re-hashes and
/// re-verifies the content id). Repeated ids are interned once by the pool's own
/// dedup.
///
/// # Errors
///
/// Returns the first `JsValue` rejection from opening the directory or from any
/// per-block read (including a missing file for a requested id).
pub async fn load_pool_from_opfs(dir_name: &str, ids: &[ContentId]) -> Result<BlockPool, JsValue> {
    let dir = open_dir(dir_name).await?;
    let mut pool = BlockPool::new();
    for &id in ids {
        let name = hex_lower(id.as_bytes());
        let bytes = read_file(&dir, &name).await?;
        // intern re-hashes; the recomputed id should match `id`.
        pool.intern(bytes);
    }
    Ok(pool)
}

/// In-browser runtime self-test of the OPFS round-trip, callable from JS.
///
/// Builds a small [`BlockPool`] (including an empty-block edge case), persists it
/// to OPFS, asserts a second persist writes **zero** blocks (incremental
/// content-addressed dedup), loads the pool back, and checks every block is
/// byte-identical. Resolves with a human-readable success string; rejects with a
/// `JsValue` describing the first failure. Gated behind the `opfs` feature; exists
/// so the OPFS path can be exercised in a real browser.
///
/// # Errors
///
/// Rejects if any OPFS operation fails or a reloaded block does not match.
#[wasm_bindgen]
pub async fn opfs_selftest() -> Result<JsValue, JsValue> {
    let mut pool = BlockPool::new();
    let a = pool.intern(vec![1u8, 2, 3, 4, 5]);
    let b = pool.intern(vec![42u8; 100]);
    let empty = pool.intern(Vec::new());
    let dir = "grafeo_opfs_selftest";

    // First persist — count depends on any prior run's leftover files.
    let _first = persist_pool_to_opfs(&pool, dir).await?;
    // Second persist must write zero: content-addressed incremental dedup.
    let second = persist_pool_to_opfs(&pool, dir).await?;
    if second != 0 {
        return Err(JsValue::from_str(&format!(
            "incremental dedup failed: 2nd persist wrote {second}, expected 0"
        )));
    }

    // Load back from OPFS and verify byte-identity.
    let loaded = load_pool_from_opfs(dir, &[a, b, empty]).await?;
    if loaded.block_count() != 3 {
        return Err(JsValue::from_str(&format!(
            "block_count {} != 3",
            loaded.block_count()
        )));
    }
    if loaded.get(a) != Some(&[1u8, 2, 3, 4, 5][..]) {
        return Err(JsValue::from_str("block a (5 bytes) mismatch after reload"));
    }
    if loaded.get(b) != Some(&[42u8; 100][..]) {
        return Err(JsValue::from_str(
            "block b (100 bytes) mismatch after reload",
        ));
    }
    if loaded.get(empty) != Some(&[][..]) {
        return Err(JsValue::from_str("empty block mismatch after reload"));
    }

    Ok(JsValue::from_str(&format!(
        "OPFS round-trip OK: 3 blocks (incl. empty) persisted + reloaded byte-identical; 2nd persist wrote {second} (incremental dedup)"
    )))
}
