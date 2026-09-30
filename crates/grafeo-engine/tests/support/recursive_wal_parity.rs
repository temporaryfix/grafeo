//! Exact typed sync/async replay of records emitted by the real recursive fixture.

use super::*;
use grafeo_storage::wal::{
    AsyncLpgWal, LpgWal, WalConfig, WalEntry, WalRecord, WalRecovery, encode_record,
};
use std::collections::{BTreeMap, BTreeSet};

type Files = BTreeMap<PathBuf, Vec<u8>>;

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".wal");
    name.into()
}

fn files(dir: &Path) -> TestResult<Files> {
    let mut result = Files::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            result.insert(
                PathBuf::from(entry.file_name()),
                std::fs::read(entry.path())?,
            );
        }
    }
    Ok(result)
}

fn assert_files(actual: &Files, expected: &Files, context: &str) {
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "{context}: file set"
    );
    for (name, bytes) in expected {
        let received = &actual[name];
        assert!(
            received == bytes,
            "{context}: {} lengths {}/{}; first differing byte {:?}",
            name.display(),
            received.len(),
            bytes.len(),
            received.iter().zip(bytes).position(|(a, b)| a != b)
        );
    }
}

fn capture_wal(db: &GrafeoDB, path: &Path) -> TestResult<Files> {
    let wal = db.wal().ok_or("missing fixture WAL")?;
    wal.sync()?;
    // This fixture has the sole writer and disables background GC. Use public
    // durability controls without adding raw append/capture access to GrafeoDB.
    let before = (
        wal.record_count(),
        wal.current_sequence(),
        wal.size_bytes()?,
    );
    let result = files(&sidecar(path))?;
    assert_eq!(
        before,
        (
            wal.record_count(),
            wal.current_sequence(),
            wal.size_bytes()?
        )
    );
    Ok(result)
}

fn install(path: &Path, container: &[u8], wal: &Files) -> TestResult {
    std::fs::write(path, container)?;
    let dir = sidecar(path);
    std::fs::create_dir(&dir)?;
    for (name, bytes) in wal {
        std::fs::write(dir.join(name), bytes)?;
    }
    Ok(())
}

fn records(mut bytes: &[u8]) -> TestResult<Vec<WalRecord>> {
    let mut result = Vec::new();
    while !bytes.is_empty() {
        let length = u32::from_le_bytes(bytes.get(..4).ok_or("short frame length")?.try_into()?);
        let end = 4usize
            .checked_add(usize::try_from(length)?)
            .ok_or("frame length overflow")?;
        let payload = bytes.get(4..end).ok_or("short frame payload")?;
        let checksum_end = end.checked_add(4).ok_or("checksum offset overflow")?;
        let checksum = u32::from_le_bytes(
            bytes
                .get(end..checksum_end)
                .ok_or("short checksum")?
                .try_into()?,
        );
        assert_eq!(checksum, crc32fast::hash(payload));
        let envelope = payload
            .strip_prefix(b"GRAFOWAL\x05\x00")
            .ok_or("not a current WAL frame")?;
        let (coordinate, sealed_body) = envelope
            .split_at_checked(9)
            .ok_or("short WAL group coordinate")?;
        let seal_len = if matches!(coordinate[0], 1 | 2) {
            40
        } else {
            0
        };
        let body_len = sealed_body
            .len()
            .checked_sub(seal_len)
            .ok_or("short group seal")?;
        let body = &sealed_body[..body_len];
        let (record, consumed): (WalRecord, usize) =
            bincode::serde::decode_from_slice(body, bincode::config::standard())?;
        assert_eq!(consumed, body.len());
        let encoded = encode_record(&record)?;
        assert_eq!(encoded.len(), payload.len());
        let unsigned_len = payload.len() - seal_len;
        assert_eq!(
            &encoded[..unsigned_len],
            &payload[..unsigned_len],
            "record body and group coordinate must be canonical"
        );
        // encode_record reserves the seal; the sole WAL writer fills it.
        // report() verifies every complete group, and assert_files() below
        // compares full physical images including the real seal bytes.
        assert!(encoded[unsigned_len..].iter().all(|byte| *byte == 0));
        result.push(record);
        bytes = &bytes[checksum_end..];
    }
    Ok(result)
}

fn discarded_writes(db: &GrafeoDB) -> TestResult {
    let mut session = db.session();
    session.use_graph_path(&GraphPath::from_components(&["a", "b"])?)?;
    session.begin_transaction()?;
    session.set_node_property(NodeId::new(0), "discarded", Value::from("abort"))?;
    session.rollback()?;
    session.begin_transaction()?;
    session.savepoint("parity")?;
    session.set_node_property(NodeId::new(0), "discarded", Value::from("savepoint"))?;
    session.rollback_to_savepoint("parity")?;
    session.commit()?;
    Ok(())
}

fn report(path: &Path) -> TestResult<(Vec<Vec<u8>>, Option<TransactionId>)> {
    let recovered = WalRecovery::new(sidecar(path))?.recover_report()?;
    Ok((
        recovered
            .committed
            .iter()
            .map(encode_record)
            .collect::<Result<_, _>>()?,
        recovered.max_transaction_id,
    ))
}

#[tokio::test]
async fn real_recursive_tail_has_identical_sync_async_bytes_and_replay() -> TestResult {
    let temp = tempfile::tempdir()?;
    let memory = blank_recursive_memory(&temp.path().join("topology.grafeo"))?;
    let old_text_epoch = seed(&memory)?;
    let baseline = memory.current_epoch();
    let source_path = temp.path().join("source.grafeo");
    memory.save(&source_path)?;
    memory.close()?;
    drop(memory);
    let source = persistent(&source_path)?;
    source.wal_checkpoint()?;
    let container = std::fs::read(&source_path)?;
    let before = capture_wal(&source, &source_path)?;
    let first_tail_sequence = source
        .wal()
        .ok_or("missing initial sequence")?
        .current_sequence();
    discarded_writes(&source)?;
    source.wal().ok_or("missing rotation caller")?.rotate()?;
    tail(&source)?;
    let expected = capture(&source, baseline)?;
    assert_exact(&source, &expected, baseline, old_text_epoch, true)?;
    let after = capture_wal(&source, &source_path)?;
    assert_eq!(
        std::fs::read(&source_path)?,
        container,
        "all tested writes remain WAL-only"
    );

    let mut batches = BTreeMap::new();
    for (name, bytes) in &after {
        if name.extension().is_some_and(|extension| extension == "log") {
            let prefix = before.get(name).map_or(&[][..], Vec::as_slice);
            assert!(bytes.starts_with(prefix), "pre-tail segment changed");
            let sequence: u64 = name
                .file_stem()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_prefix("wal_"))
                .ok_or("invalid segment name")?
                .parse()?;
            if sequence < first_tail_sequence {
                assert_eq!(bytes, prefix, "retained comparison-only prefix changed");
                continue;
            }
            batches.insert(sequence, records(&bytes[prefix.len()..])?);
        } else {
            assert_eq!(
                before.get(name),
                Some(bytes),
                "tail changed non-segment metadata"
            );
        }
    }
    assert!(batches.len() >= 2, "real rotation must split the tail");
    assert_eq!(
        batches.first_key_value().map(|(sequence, _)| *sequence),
        Some(first_tail_sequence)
    );
    let all: Vec<_> = batches.values().flatten().collect();
    assert!(all.iter().any(|record| record.is_abort()));
    assert!(
        all.iter()
            .any(|record| record.savepoint_name() == Some("parity"))
    );
    assert!(
        all.iter()
            .any(|record| record.rollback_to_savepoint_name() == Some("parity"))
    );
    assert!(
        all.iter()
            .any(|record| matches!(record, WalRecord::IndexOwnerBatch { .. }))
    );
    assert!(
        all.iter()
            .any(|record| matches!(record, WalRecord::InsertRdfQuadV3 { .. }))
    );
    let paths: BTreeSet<_> = all
        .iter()
        .filter_map(|record| match record {
            WalRecord::LpgMutation { graph, .. } => Some(graph.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(paths, data_paths()?.into_iter().collect());

    let oracle_path = temp.path().join("oracle.grafeo");
    install(&oracle_path, &container, &after)?;
    let expected_report = report(&oracle_path)?;
    let mut replay_paths = vec![oracle_path];
    use grafeo_storage::wal::DurabilityMode as WalDurability;
    for (name, durability) in [
        ("sync", WalDurability::Sync),
        (
            "batch",
            WalDurability::Batch {
                max_delay_ms: 10,
                max_records: 4,
            },
        ),
        (
            "adaptive",
            WalDurability::Adaptive {
                target_interval_ms: 10,
            },
        ),
        ("nosync", WalDurability::NoSync),
    ] {
        let sync_path = temp.path().join(format!("{name}-sync.grafeo"));
        let async_path = temp.path().join(format!("{name}-async.grafeo"));
        install(&sync_path, &container, &before)?;
        install(&async_path, &container, &before)?;
        let config = WalConfig {
            durability,
            ..WalConfig::default()
        };
        let sync = LpgWal::with_config(sidecar(&sync_path), config.clone())?;
        let asynchronous = AsyncLpgWal::with_config(sidecar(&async_path), config).await?;
        let invalid = WalRecord::Committed {
            transaction_id: TransactionId::INVALID,
            epoch: EpochId::PENDING,
        };
        let counts = (sync.record_count(), asynchronous.record_count());
        assert!(sync.log(&invalid).is_err());
        assert!(asynchronous.log(&invalid).await.is_err());
        assert!(!sync.is_poisoned() && !asynchronous.is_poisoned());
        assert_eq!((sync.record_count(), asynchronous.record_count()), counts);
        assert_files(&files(sync.dir())?, &before, "sync invalid admission");
        assert_files(
            &files(asynchronous.dir())?,
            &before,
            "async invalid admission",
        );
        for (position, (_, records)) in batches.iter().enumerate() {
            if position != 0 {
                sync.rotate()?;
                asynchronous.rotate().await?;
            }
            for record in records {
                sync.log(record)?;
                asynchronous.log(record).await?;
            }
        }
        sync.sync()?;
        asynchronous.sync().await?;
        assert_eq!(sync.record_count(), asynchronous.record_count());
        sync.close()?;
        asynchronous.close().await?;
        drop(sync);
        drop(asynchronous);
        assert_files(&files(&sidecar(&sync_path))?, &after, "sync physical image");
        assert_files(
            &files(&sidecar(&async_path))?,
            &after,
            "async physical image",
        );
        assert_eq!(report(&sync_path)?, expected_report);
        assert_eq!(report(&async_path)?, expected_report);
        replay_paths.extend([sync_path, async_path]);
    }
    source.close()?;
    drop(source);
    for path in &replay_paths {
        let reopened = persistent(path)?;
        assert_exact(&reopened, &expected, baseline, old_text_epoch, true)?;
        reopened.close()?;
        drop(reopened);
        let reopened = persistent(path)?;
        assert_exact(&reopened, &expected, baseline, old_text_epoch, true)?;
        reopened.close()?;
    }
    eprintln!(
        "recursive WAL parity: {} paths, {} tail segments, {} frames, {} recovered records, 4 policies, {} twice-opened copies",
        paths.len(),
        batches.len(),
        all.len(),
        expected_report.0.len(),
        replay_paths.len()
    );
    Ok(())
}
