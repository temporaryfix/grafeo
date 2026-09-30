//! Durable generation publication and recovery for the existing backup caller.

use std::fs;
#[cfg(any(test, feature = "lpg"))]
use std::fs::{File, OpenOptions};
use std::io::Read;
#[cfg(any(test, feature = "lpg"))]
use std::io::Write;
use std::path::Path;

#[cfg(any(test, feature = "lpg"))]
use grafeo_common::testing::crash::maybe_crash;
#[cfg(any(test, feature = "lpg"))]
use grafeo_common::testing::wal_failure::check_backup_publication_failure;
use grafeo_common::utils::error::{Error, Result};
#[cfg(any(test, feature = "lpg"))]
use grafeo_storage::wal::WalCapture;

#[cfg(any(test, feature = "lpg"))]
use super::BackupCursor;
use super::{BackupManifest, MANIFEST_FILENAME, chain};

fn incomplete(message: &str) -> Error {
    Error::Serialization(format!("incomplete backup generation: {message}"))
}

fn generation_name(prefix: &str, chain_id: &[u8; 32], generation: u64) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut chain = String::with_capacity(64);
    for byte in chain_id {
        chain.push(char::from(HEX[usize::from(byte >> 4)]));
        chain.push(char::from(HEX[usize::from(byte & 15)]));
    }
    format!("backup_{prefix}_{chain}_{generation:020}.meta")
}

#[cfg(any(test, feature = "lpg"))]
fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path.parent().ok_or_else(|| incomplete("missing parent"))?)?.sync_all()?;
    // Match the storage ownership protocol where portable directory sync is unavailable.
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn read_metadata(path: &Path) -> Result<Vec<u8>> {
    let maximum = chain::MAX_PAYLOAD_BYTES as u64 + 17;
    let file = grafeo_storage::file::open_backup_source(path)?;
    if file.metadata()?.len() > maximum {
        return Err(incomplete("metadata exceeds its size bound"));
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(incomplete("metadata exceeds its size bound"));
    }
    Ok(bytes)
}

#[cfg(any(test, feature = "lpg"))]
fn require_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => Err(incomplete(
            "immutable generation or installing debris already exists",
        )),
    }
}

#[cfg(any(test, feature = "lpg"))]
fn install_immutable(path: &Path, bytes: &[u8], manifest: bool) -> Result<()> {
    require_absent(path)?;
    let installing = path.with_extension("meta.installing");
    require_absent(&installing)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&installing)?;
    if manifest {
        check_backup_publication_failure("backup:manifest_write")?;
    }
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    if manifest {
        maybe_crash("backup:manifest_installing_sync");
    }
    require_absent(path)?;
    if manifest {
        check_backup_publication_failure("backup:manifest_rename_failure")?;
    }
    fs::rename(&installing, path)?;
    if manifest {
        maybe_crash("backup:manifest_rename");
    }
    if manifest {
        check_backup_publication_failure("backup:manifest_parent_sync")?;
    }
    sync_parent(path)?;
    if manifest {
        maybe_crash("backup:manifest_directory_sync");
    }
    Ok(())
}

/// An offline reader accepts only a manifest with a matching durable witness.
/// The witness is written after the source cursor's directory barrier.
pub(super) fn recover_manifest(dir: &Path) -> Result<Option<BackupManifest>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut selected: Option<BackupManifest> = None;
    let mut artifacts = false;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // The permanent coordination entry alone does not establish a chain.
        if name.ends_with(".grafeo-owner") {
            continue;
        }
        artifacts |= name.starts_with("backup_");
        if !name.starts_with("backup_commit_") || name.ends_with(".installing") {
            continue;
        }
        let cursor = chain::decode_cursor(&read_metadata(&entry.path())?)?;
        let expected = generation_name("commit", &cursor.chain_id, cursor.generation);
        if name != expected {
            return Err(incomplete(
                "commit witness name disagrees with its identity",
            ));
        }
        let manifest_path = dir.join(generation_name(
            "manifest",
            &cursor.chain_id,
            cursor.generation,
        ));
        let manifest = chain::decode_manifest(&read_metadata(&manifest_path)?)?;
        super::validate_manifest(&manifest)?;
        if !chain::matching_generation(
            &chain::generation_from_manifest(&manifest)?,
            &chain::generation_from_cursor(&cursor),
        ) {
            return Err(incomplete("manifest and commit witness disagree"));
        }
        if let Some(previous) = &selected {
            if previous.chain_id != manifest.chain_id {
                return Err(incomplete(
                    "multiple chain identities share a backup directory",
                ));
            }
            if previous.generation >= manifest.generation {
                continue;
            }
        }
        selected = Some(manifest);
    }
    let Some(manifest) = selected else {
        return if artifacts {
            Err(incomplete("no matching committed pair"))
        } else {
            Ok(None)
        };
    };
    // The old public filename is an advisory locator, never standalone authority.
    let locator = dir.join(MANIFEST_FILENAME);
    if locator.try_exists()? {
        let advertised = chain::decode_manifest(&read_metadata(&locator)?)?;
        super::validate_manifest(&advertised)?;
        if advertised.chain_id != manifest.chain_id {
            return Err(incomplete("locator names a foreign chain"));
        }
        let immutable = dir.join(generation_name(
            "manifest",
            &advertised.chain_id,
            advertised.generation,
        ));
        let immutable = chain::decode_manifest(&read_metadata(&immutable)?)?;
        if chain::encode_manifest(&advertised)? != chain::encode_manifest(&immutable)? {
            return Err(incomplete("locator differs from its immutable manifest"));
        }
    }
    Ok(Some(manifest))
}

#[cfg(feature = "lpg")]
pub(super) fn recover_cursor(
    capture: &mut WalCapture<'_>,
    manifest: &BackupManifest,
) -> Result<BackupCursor> {
    let bytes = capture
        .read_backup_generation_bytes(&manifest.chain_id, manifest.generation)?
        .ok_or_else(|| incomplete("source cursor generation is absent"))?;
    let cursor = chain::decode_cursor(&bytes)?;
    let generation = chain::generation_from_manifest(manifest)?;
    if !chain::matching_generation(&generation, &chain::generation_from_cursor(&cursor)) {
        return Err(incomplete(
            "source cursor does not match the committed manifest",
        ));
    }
    if let Some(advertised) = super::read_backup_cursor(capture)? {
        let encoded = capture
            .read_backup_generation_bytes(&advertised.chain_id, advertised.generation)?
            .ok_or_else(|| incomplete("cursor locator has no immutable generation"))?;
        let immutable = chain::decode_cursor(&encoded)?;
        if !chain::matching_generation(
            &chain::generation_from_cursor(&advertised),
            &chain::generation_from_cursor(&immutable),
        ) {
            return Err(incomplete("cursor does not match its immutable generation"));
        }
        if advertised.chain_id == manifest.chain_id && advertised.generation > manifest.generation {
            return Err(incomplete(
                "cursor locator is newer than the committed backup",
            ));
        }
    }
    Ok(cursor)
}

#[cfg(any(test, feature = "lpg"))]
pub(super) fn publish(
    dir: &Path,
    manifest: &BackupManifest,
    mut capture: Option<&mut WalCapture<'_>>,
) -> Result<()> {
    super::validate_manifest(manifest)?;
    let generation = chain::generation_from_manifest(manifest)?;
    let cursor = BackupCursor {
        chain_id: generation.chain_id,
        generation: generation.generation,
        manifest_digest: generation.manifest_digest,
        backed_up_epoch: generation.end_epoch,
        log_sequence: generation.end_sequence,
        timestamp_ms: super::now_ms(),
    };
    let cursor_bytes = chain::encode_cursor(&cursor)?;
    fs::create_dir_all(dir)?;
    install_immutable(
        &dir.join(generation_name(
            "manifest",
            &manifest.chain_id,
            manifest.generation,
        )),
        &chain::encode_manifest(manifest)?,
        true,
    )?;
    if let Some(wal) = capture.as_mut() {
        wal.write_backup_generation_bytes(&manifest.chain_id, manifest.generation, &cursor_bytes)?;
    }
    // This local witness certifies that both immutable sides passed their barriers.
    install_immutable(
        &dir.join(generation_name(
            "commit",
            &manifest.chain_id,
            manifest.generation,
        )),
        &cursor_bytes,
        false,
    )?;
    maybe_crash("backup:commit_directory_sync");
    super::write_manifest(dir, manifest)?;
    if let Some(wal) = capture {
        super::write_backup_cursor(wal, &cursor)?;
    }
    Ok(())
}

#[cfg(feature = "lpg")]
pub(super) fn write_segment(path: &Path, bytes: &[u8]) -> Result<()> {
    require_absent(path)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    check_backup_publication_failure("backup:segment_write_failure")?;
    file.write_all(bytes)?;
    maybe_crash("backup:segment_write");
    file.sync_all()?;
    drop(file);
    sync_parent(path)?;
    maybe_crash("backup:segment_sync");
    let mut file = File::open(path)?;
    if file.metadata()?.len() != bytes.len() as u64 {
        return Err(incomplete("segment size changed before publication"));
    }
    let mut chunk = vec![0u8; 64 * 1024];
    let mut offset = 0;
    while offset < bytes.len() {
        let count = (bytes.len() - offset).min(chunk.len());
        file.read_exact(&mut chunk[..count])?;
        if chunk[..count] != bytes[offset..offset + count] {
            return Err(incomplete(
                "segment reread differs from the validated WAL cut",
            ));
        }
        offset += count;
    }
    drop(file);
    maybe_crash("backup:segment_validated");
    Ok(())
}
