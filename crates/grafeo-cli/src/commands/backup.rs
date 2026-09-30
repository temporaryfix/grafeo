//! Backup management commands.

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

use crate::output;
use crate::{BackupCommands, OutputFormat};

/// Run backup commands.
pub fn run(cmd: BackupCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        BackupCommands::Create { path, output: out } => {
            output::status(&format!("Creating backup of {}...", path.display()), quiet);

            let db = super::open_existing(&path)?;
            db.save(&out)
                .with_context(|| format!("Failed to create backup at {}", out.display()))?;

            output::success(&format!("Backup created at {}", out.display()), quiet);
        }
        BackupCommands::Restore { backup, path } => {
            if path.exists() {
                anyhow::bail!(
                    "Target path {} already exists. Choose a new destination for native restore.",
                    path.display()
                );
            }

            output::status(&format!("Restoring from {}...", backup.display()), quiet);

            let db = super::open_existing(&backup)
                .with_context(|| format!("Failed to open backup at {}", backup.display()))?;
            db.save(&path)
                .with_context(|| format!("Failed to restore to {}", path.display()))?;

            output::success(&format!("Database restored to {}", path.display()), quiet);
        }
        BackupCommands::Full { path, output: out } => {
            output::status(
                &format!("Creating full backup of {}...", path.display()),
                quiet,
            );

            let db = super::open_existing(&path)?;
            let segment = db
                .backup_full(&out)
                .with_context(|| format!("Failed to create full backup at {}", out.display()))?;

            output::success(
                &format!(
                    "Full backup created: {} ({} bytes, epoch 0-{})",
                    segment.filename,
                    segment.size_bytes,
                    segment.end_epoch.as_u64()
                ),
                quiet,
            );
        }
        BackupCommands::Incremental { path, output: out } => {
            output::status(
                &format!("Creating incremental backup of {}...", path.display()),
                quiet,
            );

            let db = super::open_existing(&path)?;
            let segment = db.backup_incremental(&out).with_context(|| {
                format!("Failed to create incremental backup at {}", out.display())
            })?;

            output::success(
                &format!(
                    "Incremental backup created: {} ({} bytes, epoch {}-{})",
                    segment.filename,
                    segment.size_bytes,
                    segment.start_epoch.as_u64(),
                    segment.end_epoch.as_u64()
                ),
                quiet,
            );
        }
        BackupCommands::Status { path } => {
            let manifest = GrafeoDB::read_backup_manifest(&path)
                .with_context(|| format!("Failed to read manifest at {}", path.display()))?;

            match manifest {
                Some(m) => {
                    println!("Backup manifest (version {})", m.version);
                    println!("Segments: {}", m.segments.len());
                    if let Some((start, end)) = m.epoch_range() {
                        println!("Epoch range: {} - {}", start.as_u64(), end.as_u64());
                    }
                    println!();
                    for (i, seg) in m.segments.iter().enumerate() {
                        println!(
                            "  [{i}] {:?} {} (epoch {}-{}, {} bytes)",
                            seg.kind,
                            seg.filename,
                            seg.start_epoch.as_u64(),
                            seg.end_epoch.as_u64(),
                            seg.size_bytes
                        );
                    }
                }
                None => {
                    println!("No backup manifest found at {}", path.display());
                }
            }
        }
        BackupCommands::RestoreToEpoch {
            backup_dir,
            epoch,
            output: out,
        } => {
            output::status(
                &format!(
                    "Restoring to epoch {epoch} from {}...",
                    backup_dir.display()
                ),
                quiet,
            );

            let target = grafeo_common::types::EpochId::new(epoch);
            GrafeoDB::restore_to_epoch(&backup_dir, target, &out)
                .with_context(|| format!("Failed to restore to epoch {epoch}"))?;

            output::success(
                &format!("Database restored to epoch {epoch} at {}", out.display()),
                quiet,
            );
        }
    }

    Ok(())
}
