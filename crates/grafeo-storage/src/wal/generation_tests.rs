//! Independent byte-level admission oracles for the current WAL generation.

use super::{
    CheckpointMetadata, TestWalDirectory, TypedWal, WalConfig, WalEntry, WalManager, WalRecord,
    WalRecovery, test_wal_dir,
};
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, ErrorCode, Result as StorageResult, StorageError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn image(path: &Path) -> std::io::Result<BTreeMap<OsString, Vec<u8>>> {
    std::fs::read_dir(path)?
        .map(|entry| {
            let entry = entry?;
            Ok((entry.file_name(), std::fs::read(entry.path())?))
        })
        .collect()
}

fn current<R: WalEntry>(record: &R) -> TestResult<Vec<u8>> {
    let mut payload = b"GRAFOWAL".to_vec();
    payload.extend_from_slice(&5u16.to_le_bytes());
    let role = if record.is_commit() {
        1
    } else if record.is_abort() {
        2
    } else if record.is_checkpoint() {
        4
    } else if record.is_metadata() {
        3
    } else {
        0
    };
    payload.push(role);
    payload.extend_from_slice(
        &record
            .transaction_id()
            .map_or(u64::MAX, |id| id.as_u64())
            .to_le_bytes(),
    );
    payload.extend(bincode::serde::encode_to_vec(
        record,
        bincode::config::standard(),
    )?);
    Ok(payload)
}

fn frame(payload: &[u8], bad_crc: bool) -> TestResult<Vec<u8>> {
    let mut bytes = u32::try_from(payload.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&(crc32fast::hash(payload) ^ u32::from(bad_crc)).to_le_bytes());
    Ok(bytes)
}

fn checkpoint(path: &Path) -> TestResult {
    let metadata = CheckpointMetadata {
        format_version: 5,
        retired_before: 0,
        epoch: EpochId::new(1),
        log_sequence: 1,
        timestamp_ms: 0,
        transaction_id: TransactionId::new(1),
    };
    std::fs::write(
        path.join("checkpoint.meta"),
        bincode::serde::encode_to_vec(metadata, bincode::config::standard())?,
    )?;
    Ok(())
}

fn fixture(bytes: &[u8], retained_prefix: bool) -> TestResult<TestWalDirectory> {
    let dir = test_wal_dir()?;
    std::fs::write(dir.path().join("wal_00000000.log"), bytes)?;
    if retained_prefix {
        checkpoint(dir.path())?;
        std::fs::write(dir.path().join("wal_00000001.log"), [])?;
    }
    Ok(dir)
}

fn predecessor_payloads() -> TestResult<Vec<Vec<u8>>> {
    let record = WalRecord::EpochAdvance {
        epoch: EpochId::new(1),
    };
    let mut unknown = current(&record)?;
    unknown[8..10].copy_from_slice(&6u16.to_le_bytes());
    let predecessor = include_bytes!("../../tests/fixtures/golden_wal_v4.bin");
    let len = u32::from_le_bytes(predecessor[..4].try_into()?) as usize;
    Ok(vec![
        predecessor[4..4 + len].to_vec(),
        bincode::serde::encode_to_vec(record, bincode::config::standard())?,
        unknown,
        b"GRAFOWAL\x01\x00".to_vec(),
        b"GRAFOWAL\x02\x00".to_vec(),
        b"GRAFOWAL\x03\x00".to_vec(),
        b"GRAFOWAL\x04\x00".to_vec(),
        b"GRAF".to_vec(),
        b"GRAFOWAL".to_vec(),
        b"GRAFOWAL\x01".to_vec(),
    ])
}

fn refused<T>(result: StorageResult<T>) -> TestResult<Error> {
    result
        .err()
        .ok_or_else(|| "invalid WAL was admitted".into())
}

fn invalid_generation<T>(result: StorageResult<T>) -> TestResult {
    let error = refused(result)?;
    assert!(
        matches!(error, Error::Storage(StorageError::InvalidWalEntry(_))),
        "{error}"
    );
    Ok(())
}

#[test]
fn predecessor_generation_is_refused_without_mutating_any_segment() -> TestResult {
    for payload in predecessor_payloads()? {
        for prefix in [false, true] {
            let dir = fixture(&frame(&payload, false)?, prefix)?;
            let before = image(dir.path())?;
            invalid_generation(WalManager::open(dir.path()))?;
            assert_eq!(image(dir.path())?, before);
            invalid_generation(TypedWal::<WalRecord>::open(dir.path()))?;
            assert_eq!(image(dir.path())?, before);
            let mut recovery = WalRecovery::new(dir.path())?;
            invalid_generation(recovery.recover_report())?;
            assert_eq!(image(dir.path())?, before);
            assert!(recovery.into_wal(WalConfig::default()).is_err());
            assert_eq!(image(dir.path())?, before);
        }
    }
    Ok(())
}

#[test]
fn predecessor_projection_records_are_refused_without_mutating_any_segment() -> TestResult {
    let bodies: &[&[u8]] = &[
        &[
            36, 17, 25, 104, 116, 116, 112, 58, 47, 47, 101, 120, 97, 109, 112, 108, 101, 46, 111,
            114, 103, 47, 80, 101, 114, 115, 111, 110, 6, 80, 101, 114, 115, 111, 110,
        ],
        &[37, 17, 3, 41, 9],
        &[
            39, 17, 25, 104, 116, 116, 112, 58, 47, 47, 101, 120, 97, 109, 112, 108, 101, 46, 111,
            114, 103, 47, 80, 101, 114, 115, 111, 110, 6, 80, 101, 114, 115, 111, 110, 42,
        ],
        &[40, 17, 4, 41, 43, 9],
    ];
    for body in bodies {
        let payload = [
            b"GRAFOWAL\x05\x00\x03\xff\xff\xff\xff\xff\xff\xff\xff".as_slice(),
            *body,
        ]
        .concat();
        let dir = fixture(&frame(&payload, false)?, false)?;
        let before = image(dir.path())?;
        WalManager::open(dir.path())?.close()?;
        assert_eq!(image(dir.path())?, before);
        TypedWal::<WalRecord>::open(dir.path())?.close()?;
        assert_eq!(image(dir.path())?, before);
        let mut recovery = WalRecovery::new(dir.path())?;
        invalid_generation(recovery.recover_report())?;
        assert_eq!(image(dir.path())?, before);
    }
    Ok(())
}

#[test]
fn predecessor_rdf_records_are_refused_without_mutating_any_segment() -> TestResult {
    macro_rules! captured {
        ($name:literal) => {
            (
                $name,
                include_bytes!(concat!(
                    "../../tests/fixtures/rdf-wal-predecessors/",
                    $name,
                    ".body.bin"
                ))
                .as_slice(),
                include_bytes!(concat!(
                    "../../tests/fixtures/rdf-wal-predecessors/",
                    $name,
                    ".payload.bin"
                ))
                .as_slice(),
                include_bytes!(concat!(
                    "../../tests/fixtures/rdf-wal-predecessors/",
                    $name,
                    ".wal.bin"
                ))
                .as_slice(),
            )
        };
    }
    for (name, body, payload, sealed_wal) in [
        captured!("retired-15-insert-default"),
        captured!("retired-16-delete-default"),
        captured!("retired-17-clear-default"),
        captured!("retired-31-micros-default"),
        captured!("retired-41-tai-v2-default"),
        captured!("retired-15-insert-named"),
        captured!("retired-16-delete-named"),
        captured!("retired-17-clear-named"),
        captured!("retired-31-micros-named"),
        captured!("retired-41-tai-v2-named"),
        captured!("retired-26-drop-named"),
        captured!("retired-28-create-named"),
        // Names retain the original capture's current/retired classification;
        // these graph records became predecessors in the subsequent cut.
        captured!("current-18-create-live"),
        captured!("current-19-drop-live-default"),
        captured!("current-19-drop-live-named"),
    ] {
        // The retained parent body now selects only an invalid unit slot. Full
        // bodies additionally fail the decoder's exact-consumption requirement.
        let (reserved, consumed): (WalRecord, usize) =
            bincode::serde::decode_from_slice(body, bincode::config::standard())?;
        assert_eq!(consumed, 1, "{name}");
        assert!(consumed < body.len(), "{name}");
        assert!(reserved.validate_recovery().is_err(), "{name}");
        assert!(super::encode_record(&reserved).is_err(), "{name}");
        assert_eq!(&payload[19..], body, "{name}");
        // Preserve the authentic parent's envelope coordinate even for the
        // bare-tag case, so refusal cannot depend on a bad envelope or CRC.
        let mut bare_tag = payload[..19].to_vec();
        bare_tag.push(body[0]);
        for (representation, bytes) in [
            ("sealed committed WAL", sealed_wal.to_vec()),
            ("full body", frame(payload, false)?),
            ("bare tag", frame(&bare_tag, false)?),
        ] {
            let dir = fixture(&bytes, false)?;
            let before = image(dir.path())?;
            // Structural writer admission does not decode record semantics.
            WalManager::open(dir.path())?.close()?;
            assert_eq!(image(dir.path())?, before, "{name}: {representation}");
            TypedWal::<WalRecord>::open(dir.path())?.close()?;
            assert_eq!(image(dir.path())?, before, "{name}: {representation}");
            for attempt in 0..2 {
                let mut recovery = WalRecovery::new(dir.path())?;
                let error = refused(recovery.recover_report())?;
                assert!(
                    matches!(error, Error::Storage(StorageError::InvalidWalEntry(_))),
                    "{name}: {representation}, attempt {attempt}: {error}"
                );
                assert_eq!(image(dir.path())?, before, "{name}: {representation}");
                assert!(recovery.into_wal(WalConfig::default()).is_err(), "{name}");
                assert_eq!(image(dir.path())?, before, "{name}: {representation}");
            }
        }
    }
    Ok(())
}

#[test]
fn retained_generation_one_fixture_is_refused_before_record_decode() -> TestResult {
    for bytes in [
        include_bytes!("../../tests/fixtures/golden_wal_v1.bin").as_slice(),
        include_bytes!("../../tests/fixtures/golden_wal_v2.bin").as_slice(),
        include_bytes!("../../tests/fixtures/golden_wal_v3.bin").as_slice(),
        include_bytes!("../../tests/fixtures/golden_wal_v4.bin").as_slice(),
    ] {
        let dir = fixture(bytes, false)?;
        let before = image(dir.path())?;
        invalid_generation(WalManager::open(dir.path()))?;
        invalid_generation(TypedWal::<WalRecord>::open(dir.path()))?;
        invalid_generation(WalRecovery::new(dir.path())?.recover_report())?;
        assert_eq!(image(dir.path())?, before);
    }
    Ok(())
}

#[test]
fn malformed_native_graph_paths_fail_closed_inside_current_envelopes() -> TestResult {
    // The current mutation starts with variant 25 and transaction 1, followed
    // by a byte vector containing the canonical path. An operation need not be
    // supplied: path validation must refuse before the operation is decoded.
    let mut paths = vec![
        vec![0; 3],                           // Missing component-count byte.
        257_u32.to_le_bytes().to_vec(),       // Too many components.
        vec![0, 0, 0, 0, 9],                  // Trailing path bytes.
        vec![1, 0, 0, 0, 255, 255, 255, 255], // Unbounded component length.
        vec![1, 0, 0, 0, 1, 0, 0, 0, 255],    // Invalid UTF-8.
    ];
    for path in paths.drain(..) {
        let mut payload = b"GRAFOWAL\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x19\x01".to_vec();
        payload.extend(bincode::serde::encode_to_vec(
            path,
            bincode::config::standard(),
        )?);
        let dir = fixture(&frame(&payload, false)?, false)?;
        let before = image(dir.path())?;
        invalid_generation(WalRecovery::new(dir.path())?.recover_report())?;
        assert_eq!(image(dir.path())?, before);
    }
    let mut payload = b"GRAFOWAL\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x19\x01".to_vec();
    payload.extend(bincode::serde::encode_to_vec(
        u64::MAX,
        bincode::config::standard(),
    )?);
    let dir = fixture(&frame(&payload, false)?, false)?;
    let before = image(dir.path())?;
    invalid_generation(WalRecovery::new(dir.path())?.recover_report())?;
    assert_eq!(image(dir.path())?, before);
    Ok(())
}

#[test]
fn checksum_failure_precedes_generation_and_direct_open_never_repairs() -> TestResult {
    let payload = predecessor_payloads()?.remove(1);
    let dir = fixture(&frame(&payload, true)?, false)?;
    let before = image(dir.path())?;
    for error in [
        refused(WalManager::open(dir.path()))?,
        refused(TypedWal::<WalRecord>::open(dir.path()))?,
    ] {
        assert!(error.error_code() == ErrorCode::StorageCorrupted, "{error}");
        assert!(
            error.to_string().to_ascii_lowercase().contains("checksum"),
            "{error}"
        );
        assert_eq!(image(dir.path())?, before);
    }
    let mut recovery = WalRecovery::new(dir.path())?;
    let error = refused(recovery.recover_validated(|_, _, _| -> StorageResult<()> {
        Err(Error::Internal(
            "checksum-invalid input reached validation".into(),
        ))
    }))?;
    assert!(error.error_code() == ErrorCode::StorageCorrupted, "{error}");
    assert_eq!(image(dir.path())?, before);
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CustomRecord([u8; 4]);

impl WalEntry for CustomRecord {
    fn requires_sync(&self) -> bool {
        false
    }
    fn is_commit(&self) -> bool {
        false
    }
    fn is_abort(&self) -> bool {
        false
    }
    fn is_checkpoint(&self) -> bool {
        false
    }
    fn is_metadata(&self) -> bool {
        true
    }
    fn make_checkpoint(_: TransactionId) -> Self {
        Self([0; 4])
    }
}

#[test]
fn custom_current_record_survives_direct_reopen_and_typed_handoff() -> TestResult {
    let record = CustomRecord([255; 4]);
    let body = bincode::serde::encode_to_vec(&record, bincode::config::standard())?;
    assert!(
        bincode::serde::decode_from_slice::<WalRecord, _>(&body, bincode::config::standard())
            .is_err()
    );
    assert_eq!(super::encode_record(&record)?, current(&record)?);
    assert!(
        super::encode_record(&WalRecord::EpochAdvance {
            epoch: EpochId::PENDING
        })
        .is_err()
    );
    let dir = fixture(&frame(&current(&record)?, false)?, false)?;
    let wal = WalManager::open(dir.path())?;
    wal.close()?;
    drop(wal);
    let wal = TypedWal::<CustomRecord>::open(dir.path())?;
    wal.log(&record)?;
    wal.close()?;
    drop(wal);
    let mut recovery = WalRecovery::new(dir.path())?;
    assert_eq!(
        recovery.recover_report_as::<CustomRecord>()?.committed,
        vec![record.clone(), record]
    );
    let wal = TypedWal::<CustomRecord>::from_manager(recovery.into_wal(WalConfig::default())?);
    wal.close()?;
    Ok(())
}

#[test]
fn current_prefix_with_latest_empty_checkpoint_segment_is_admissible() -> TestResult {
    let record = WalRecord::EpochAdvance {
        epoch: EpochId::new(1),
    };
    let dir = fixture(&frame(&current(&record)?, false)?, true)?;
    let wal = WalManager::open(dir.path())?;
    wal.close()?;
    drop(wal);
    let mut recovery = WalRecovery::new(dir.path())?;
    assert!(recovery.recover_report()?.committed.is_empty());
    let wal = recovery.into_wal(WalConfig::default())?;
    wal.close()?;
    drop(wal);
    let wal = WalManager::open(dir.path())?;
    wal.close()?;
    Ok(())
}

#[test]
fn torn_tail_requires_explicit_recovery_before_writer_handoff() -> TestResult {
    let complete = frame(
        &current(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })?,
        false,
    )?;
    let mut torn = complete.clone();
    torn.extend_from_slice(&[12, 0, 0]);
    let dir = fixture(&torn, false)?;
    let before = image(dir.path())?;
    refused(WalManager::open(dir.path()))?;
    assert_eq!(image(dir.path())?, before);
    refused(TypedWal::<WalRecord>::open(dir.path()))?;
    assert_eq!(image(dir.path())?, before);
    let mut recovery = WalRecovery::new(dir.path())?;
    assert_eq!(recovery.recover_report()?.committed.len(), 1);
    assert_eq!(
        std::fs::read(dir.path().join("wal_00000000.log"))?,
        complete
    );
    let wal = recovery.into_wal(WalConfig::default())?;
    wal.log(&WalRecord::EpochAdvance {
        epoch: EpochId::new(2),
    })?;
    wal.close()?;
    drop(wal);
    assert_eq!(
        WalRecovery::new(dir.path())?
            .recover_report()?
            .committed
            .len(),
        2
    );
    Ok(())
}

fn namespace_fixture(case: u8) -> TestResult<TestWalDirectory> {
    let dir = fixture(&[], false)?;
    match case {
        0 => std::fs::write(dir.path().join("WAL_CORRUPT"), b"retained quarantine")?,
        1 => std::fs::write(dir.path().join("wal_0.log"), [])?,
        _ => checkpoint(dir.path())?,
    }
    Ok(dir)
}

#[cfg(feature = "encryption")]
#[test]
fn encrypted_generation_admission_authenticates_before_rejecting_without_repair() -> TestResult {
    use grafeo_common::encryption::{KEY_SIZE, KeyChain, build_nonce};
    let chain = KeyChain::new([37; KEY_SIZE]);
    let encryptor = || chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes());
    for payload in predecessor_payloads()? {
        for prefix in [false, true] {
            let encrypted = encryptor().encrypt(&payload, &build_nonce(0, 0), b"grafeo-wal")?;
            let mut bytes = u32::try_from(encrypted.len())?.to_le_bytes().to_vec();
            bytes.extend_from_slice(&encrypted);
            let dir = fixture(&bytes, prefix)?;
            let before = image(dir.path())?;
            invalid_generation(WalManager::with_config_and_encryptor(
                dir.path(),
                WalConfig::default(),
                encryptor(),
            ))?;
            assert_eq!(image(dir.path())?, before);
            let mut recovery = WalRecovery::with_encryptor(dir.path(), encryptor())?;
            invalid_generation(recovery.recover_report())?;
            assert!(recovery.into_wal(WalConfig::default()).is_err());
            assert_eq!(image(dir.path())?, before);

            *bytes.last_mut().ok_or("missing authentication tag")? ^= 1;
            std::fs::write(dir.path().join("wal_00000000.log"), &bytes)?;
            let before = image(dir.path())?;
            let error = refused(WalManager::with_config_and_encryptor(
                dir.path(),
                WalConfig::default(),
                encryptor(),
            ))?;
            assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
            assert!(error.to_string().contains("decryption failed"));
            assert_eq!(image(dir.path())?, before);
        }
    }
    Ok(())
}

#[test]
fn direct_open_respects_quarantine_sequence_and_checkpoint_boundaries() -> TestResult {
    for case in 0..3 {
        let dir = namespace_fixture(case)?;
        let before = image(dir.path())?;
        refused(WalManager::open(dir.path()))?;
        assert_eq!(image(dir.path())?, before);
        refused(TypedWal::<WalRecord>::open(dir.path()))?;
        assert_eq!(image(dir.path())?, before);
    }
    Ok(())
}

#[cfg(feature = "async-storage")]
#[tokio::test]
async fn async_direct_open_uses_identical_nonmutating_generation_admission() -> TestResult {
    for payload in predecessor_payloads()? {
        for prefix in [false, true] {
            let dir = fixture(&frame(&payload, false)?, prefix)?;
            let before = image(dir.path())?;
            invalid_generation(super::AsyncWalManager::open(dir.path()).await)?;
            assert_eq!(image(dir.path())?, before);
            invalid_generation(super::AsyncTypedWal::<CustomRecord>::open(dir.path()).await)?;
            assert_eq!(image(dir.path())?, before);
        }
    }
    for dir in [
        namespace_fixture(0)?,
        namespace_fixture(1)?,
        namespace_fixture(2)?,
        fixture(&[12, 0, 0], false)?,
    ] {
        let before = image(dir.path())?;
        refused(super::AsyncWalManager::open(dir.path()).await)?;
        assert_eq!(image(dir.path())?, before);
        refused(super::AsyncTypedWal::<CustomRecord>::open(dir.path()).await)?;
        assert_eq!(image(dir.path())?, before);
    }
    let dir = fixture(&frame(&predecessor_payloads()?.remove(1), true)?, false)?;
    let before = image(dir.path())?;
    for error in [
        refused(super::AsyncWalManager::open(dir.path()).await)?,
        refused(super::AsyncTypedWal::<CustomRecord>::open(dir.path()).await)?,
    ] {
        assert!(error.error_code() == ErrorCode::StorageCorrupted, "{error}");
        assert!(error.to_string().contains("checksum"), "{error}");
        assert_eq!(image(dir.path())?, before);
    }
    let record = CustomRecord([255; 4]);
    let dir = fixture(&frame(&current(&record)?, false)?, false)?;
    let wal = super::AsyncTypedWal::<CustomRecord>::open(dir.path()).await?;
    wal.log(&record).await?;
    wal.close().await?;
    drop(wal);
    assert_eq!(
        WalRecovery::new(dir.path())?
            .recover_report_as::<CustomRecord>()?
            .committed,
        vec![record.clone(), record]
    );
    Ok(())
}
