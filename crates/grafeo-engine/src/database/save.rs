//! Exact current-format container copies shared by both native graph models.

use std::path::{Path, PathBuf};

use grafeo_common::types::GraphModelTag;
use grafeo_common::utils::error::{Error, Result};

use super::GrafeoDB;
use crate::config::GraphModel;

#[cfg(feature = "wal")]
fn absolute_save_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

/// A completed single-file container staged beside its destination. The file
/// manager creates the stage with `create_new`; publication then uses an
/// atomic no-replace rename so concurrent savers cannot clobber one another.
#[cfg(feature = "grafeo-file")]
struct StagedGrafeoFile {
    stage: grafeo_storage::file::OwnedContainerStage,
}

#[cfg(feature = "grafeo-file")]
impl StagedGrafeoFile {
    fn create(
        destination: grafeo_storage::file::ContainerDestination,
        graph_model: u8,
    ) -> Result<Self> {
        Ok(Self {
            stage: destination.into_stage(graph_model)?,
        })
    }

    fn install(self) -> Result<()> {
        self.stage
            .install(super::atomic_install::rename_file_noreplace)
    }
}

#[cfg(feature = "grafeo-file")]
fn ensure_destination_absent(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(Error::Internal(format!(
            "refusing to save over existing destination {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

impl GrafeoDB {
    /// Saves an exact, independent container copy at an unused file path.
    ///
    /// The filename extension does not select a different format. The copy
    /// preserves the captured epoch, lineage, temporal history, catalog, exact
    /// indexes and projection receipts. It can be opened with [`Self::open`].
    /// The original database remains unchanged, including WAL-directory sources.
    ///
    /// # Errors
    ///
    /// Returns an error if the save operation fails.
    ///
    /// Requires the `wal` feature for persistence support.
    #[cfg(feature = "wal")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        // Resolve once. The process working directory can change concurrently;
        // carrying one owned absolute path prevents staging, publication, and
        // cleanup from drifting into different directories.
        let path = absolute_save_path(path.as_ref())?;

        self.save_as_grafeo_file(&path)
    }

    /// Saves the database to a single `.grafeo` file as one complete,
    /// manifest-verified, versioned container image.
    ///
    /// The copy uses the same section serializers and WorldMetadata manifest as
    /// checkpoint publication, whether or not the source has been compacted.
    /// Capturing a copy never marks the source's section wrappers clean.
    #[cfg(feature = "grafeo-file")]
    fn save_as_grafeo_file(&self, path: &Path) -> Result<()> {
        // Serialize every section exactly once behind one coherent publication
        // boundary. Destination I/O uses only these immutable bytes after all
        // source locks have been released.
        let (encoded, epoch, transaction_id, node_count, edge_count, graph_model) = {
            let _capture = self.acquire_quiescent_capture("save")?;
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            self.validate_live_rdf_lpg_projection_rows()?;

            let (sections, captured_lpg) = self.build_sections(false)?;
            let section_refs: Vec<&dyn grafeo_common::storage::Section> =
                sections.iter().map(|section| section.as_ref()).collect();
            let mut encoded = super::world_metadata::encode_sections(&section_refs, captured_lpg)?;
            let epoch = self.transaction_manager.current_epoch();
            let graph_model_tag =
                GraphModelTag::from_u8(self.config.graph_model.as_u8()).map_err(|error| {
                    Error::Serialization(format!("capture saved graph model: {error}"))
                })?;
            let world_identity = self.world_identity();
            #[cfg(feature = "triple-store")]
            let world_identity = super::world_metadata::validate_live_world_identity(
                world_identity,
                graph_model_tag,
                &self.rdf_store,
            )?;
            let world_cut = super::world_metadata::capture_world_cut_inputs(
                world_identity,
                epoch,
                graph_model_tag,
                &self.catalog,
                #[cfg(all(feature = "triple-store", feature = "lpg"))]
                &self.rdf_projections,
            )?;
            let transaction_id = self
                .transaction_manager
                .last_assigned_transaction_id()
                .map_or(0, |transaction| transaction.0);
            let (node_count, edge_count) = if self.config.graph_model == GraphModel::Rdf {
                (0, 0)
            } else {
                #[cfg(feature = "lpg")]
                {
                    let node_count = self.read_graph_view().node_count() as u64;
                    let edge_count = self.read_graph_view().edge_count() as u64;
                    (node_count, edge_count)
                }
                #[cfg(not(feature = "lpg"))]
                return Err(Error::Internal(
                    "native LPG save requires the lpg feature".into(),
                ));
            };
            let recovery_coordinates = super::world_metadata::RecoveryCoordinates::new(
                epoch.as_u64(),
                transaction_id,
                graph_model_tag,
                node_count,
                edge_count,
            );
            super::world_metadata::append_world_metadata(
                &mut encoded,
                world_cut,
                recovery_coordinates,
            )?;
            self.require_quiescent("save")?;
            (
                encoded,
                epoch.as_u64(),
                transaction_id,
                node_count,
                edge_count,
                self.config.graph_model.as_u8(),
            )
        };

        let destination = grafeo_storage::file::ContainerDestination::acquire(path)?;
        ensure_destination_absent(destination.path())?;
        let staging = StagedGrafeoFile::create(destination, graph_model)?;
        let writes: Vec<_> = encoded
            .iter()
            .map(|section| {
                grafeo_storage::file::SectionWrite::new(
                    section.section_type(),
                    section.version(),
                    section.bytes(),
                )
            })
            .collect();
        staging.stage.manager().write_versioned_sections(
            &writes,
            epoch,
            transaction_id,
            node_count,
            edge_count,
        )?;
        staging.install()
    }
}
