//! Golden fixture tests for WAL frame format stability.
//!
//! These tests deserialize a committed binary fixture (`golden_wal_v5.bin`)
//! containing hand-crafted current-generation WAL frames and verify the code can
//! parse them. If a code change alters the frame layout (length prefix, CRC
//! position), bincode encoding of `WalRecord`, or enum variant ordering,
//! these tests fail immediately.
//!
//! Frame: `[length: u32 LE][GRAFOWAL][generation: u16 LE][role: u8][transaction: u64 LE][bincode][optional group seal][crc32: u32 LE]`.
//! Length and CRC cover the complete envelope plus bincode payload.
//!
//! ## When these tests fail
//!
//! - **Accidental breakage**: fix the regression.
//! - **Intentional format change**: regenerate the fixture:
//!   ```
//!   cargo +1.97.1 test --locked -p grafeo-storage --features wal --test golden_wal_frames -- regenerate_wal_fixture --ignored
//!   ```

#![cfg(feature = "wal")]

use grafeo_common::types::{EdgeId, NodeId, TransactionId, Value};
use grafeo_storage::wal::WalRecord;

/// The set of WAL records embedded in the golden fixture.
/// Order matters: it must match the order in the fixture file.
fn golden_records() -> Vec<WalRecord> {
    vec![
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
        ),
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(1),
                key: "name".to_string(),
                value: Value::String("Alix".into()),
            },
        ),
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Person".to_string()],
            },
        ),
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::CreateEdge {
                id: EdgeId::new(1),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: "KNOWS".to_string(),
            },
        ),
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::SetEdgeProperty {
                id: EdgeId::new(1),
                key: "since".to_string(),
                value: Value::Int64(2020),
            },
        ),
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::DeleteEdge { id: EdgeId::new(1) },
        ),
        WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            grafeo_storage::wal::LpgMutationOp::DeleteNode { id: NodeId::new(2) },
        ),
        WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        },
        WalRecord::Checkpoint {
            transaction_id: TransactionId::new(1),
        },
    ]
}

/// Independently encode the fixed one-transaction oracle, without the writer codec.
fn encode_frames(records: &[WalRecord]) -> Vec<u8> {
    let mut frames = Vec::new();
    let mut count = 0_u64;
    let mut records_hash = blake3::Hasher::new_derive_key("grafeo/wal/group-record/v5");
    for record in records {
        let role = match record {
            WalRecord::LpgMutation { .. } => 0,
            WalRecord::TransactionCommit { .. } => 1,
            WalRecord::Checkpoint { .. } => 4,
            _ => panic!("unexpected golden record"),
        };
        let body = bincode::serde::encode_to_vec(record, bincode::config::standard()).unwrap();
        let mut data = b"GRAFOWAL\x05\x00".to_vec();
        data.push(role);
        data.extend_from_slice(&1_u64.to_le_bytes());
        data.extend_from_slice(&body);
        if role == 0 {
            records_hash.update(&count.to_le_bytes());
            records_hash.update(&u64::try_from(data.len()).unwrap().to_le_bytes());
            records_hash.update(&data);
            count += 1;
        } else if role == 1 {
            let mut hash = blake3::Hasher::new_derive_key("grafeo/wal/commit-group/v5");
            hash.update(&data);
            hash.update(&1_u64.to_le_bytes());
            hash.update(&count.to_le_bytes());
            hash.update(records_hash.finalize().as_bytes());
            data.extend_from_slice(&count.to_le_bytes());
            data.extend_from_slice(hash.finalize().as_bytes());
            count = 0;
            records_hash = blake3::Hasher::new_derive_key("grafeo/wal/group-record/v5");
        } else {
            assert_eq!(count, 0);
        }
        frames.extend_from_slice(&u32::try_from(data.len()).unwrap().to_le_bytes());
        frames.extend_from_slice(&data);
        frames.extend_from_slice(&crc32fast::hash(&data).to_le_bytes());
    }
    frames
}

/// Parse all frames from raw bytes, returning `(record, data_bytes)` pairs.
fn parse_frames(mut bytes: &[u8]) -> Vec<(WalRecord, Vec<u8>)> {
    let mut results = Vec::new();
    while bytes.len() >= 8 {
        let len = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        if bytes.len() < 4 + len + 4 {
            break;
        }
        let data = &bytes[4..4 + len];
        let stored_crc = u32::from_le_bytes(bytes[4 + len..4 + len + 4].try_into().unwrap());
        let actual_crc = crc32fast::hash(data);
        assert_eq!(stored_crc, actual_crc, "CRC mismatch in WAL frame");
        assert!(data.len() >= 19, "missing current WAL envelope");
        assert_eq!(&data[..8], b"GRAFOWAL", "invalid WAL magic");
        assert_eq!(
            u16::from_le_bytes(data[8..10].try_into().unwrap()),
            5,
            "unsupported WAL generation"
        );

        assert_eq!(u64::from_le_bytes(data[11..19].try_into().unwrap()), 1);
        let body_end = data.len() - if data[10] == 1 { 40 } else { 0 };
        let (record, consumed): (WalRecord, usize) =
            bincode::serde::decode_from_slice(&data[19..body_end], bincode::config::standard())
                .unwrap();
        assert_eq!(consumed, body_end - 19, "trailing record bytes");
        results.push((record, data.to_vec()));
        bytes = &bytes[4 + len + 4..];
    }
    assert!(bytes.is_empty(), "incomplete golden WAL frame");
    results
}

fn golden_bytes() -> &'static [u8] {
    include_bytes!("fixtures/golden_wal_v5.bin")
}

// ---------------------------------------------------------------------------
// Current-generation tests
// ---------------------------------------------------------------------------

#[test]
fn golden_wal_frame_count() {
    let frames = parse_frames(golden_bytes());
    assert_eq!(
        frames.len(),
        golden_records().len(),
        "expected {} frames, got {}",
        golden_records().len(),
        frames.len(),
    );
}

#[test]
fn golden_wal_crc_integrity() {
    // parse_frames asserts CRC per frame, so if this succeeds, all CRCs match.
    let _ = parse_frames(golden_bytes());
}

#[test]
fn golden_wal_create_node() {
    let frames = parse_frames(golden_bytes());
    match &frames[0].0 {
        WalRecord::LpgMutation {
            op: grafeo_storage::wal::LpgMutationOp::CreateNode { id, labels },
            ..
        } => {
            assert_eq!(*id, NodeId::new(1));
            assert_eq!(labels, &["Person"]);
        }
        other => panic!("expected CreateNode, got {other:?}"),
    }
}

#[test]
fn golden_wal_set_node_property() {
    let frames = parse_frames(golden_bytes());
    match &frames[1].0 {
        WalRecord::LpgMutation {
            op: grafeo_storage::wal::LpgMutationOp::SetNodeProperty { id, key, value },
            ..
        } => {
            assert_eq!(*id, NodeId::new(1));
            assert_eq!(key, "name");
            assert_eq!(*value, Value::String("Alix".into()));
        }
        other => panic!("expected SetNodeProperty, got {other:?}"),
    }
}

#[test]
fn golden_wal_create_edge() {
    let frames = parse_frames(golden_bytes());
    match &frames[3].0 {
        WalRecord::LpgMutation {
            op:
                grafeo_storage::wal::LpgMutationOp::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type,
                },
            ..
        } => {
            assert_eq!(*id, EdgeId::new(1));
            assert_eq!(*src, NodeId::new(1));
            assert_eq!(*dst, NodeId::new(2));
            assert_eq!(edge_type, "KNOWS");
        }
        other => panic!("expected CreateEdge, got {other:?}"),
    }
}

#[test]
fn golden_wal_commit_and_checkpoint() {
    let frames = parse_frames(golden_bytes());
    match &frames[7].0 {
        WalRecord::TransactionCommit { transaction_id } => {
            assert_eq!(*transaction_id, TransactionId::new(1));
        }
        other => panic!("expected TransactionCommit, got {other:?}"),
    }
    match &frames[8].0 {
        WalRecord::Checkpoint { transaction_id } => {
            assert_eq!(*transaction_id, TransactionId::new(1));
        }
        other => panic!("expected Checkpoint, got {other:?}"),
    }
}

#[test]
fn golden_wal_byte_equality() {
    // Re-encode all golden records and verify byte-for-byte match.
    let fresh = encode_frames(&golden_records());
    assert_eq!(
        fresh.as_slice(),
        golden_bytes(),
        "WAL frame bytes differ, encoding may have changed",
    );
}

#[test]
fn golden_wal_payloads_match_shared_current_encoder() -> Result<(), Box<dyn std::error::Error>> {
    let records = golden_records();
    let frames = parse_frames(golden_bytes());
    assert_eq!(frames.len(), records.len());
    for (record, (_, payload)) in records.iter().zip(frames) {
        let prepared = grafeo_storage::wal::encode_record(record)?;
        if matches!(record, WalRecord::TransactionCommit { .. }) {
            let at = payload.len() - 40;
            assert_eq!(&prepared[..at], &payload[..at]);
            assert_eq!(&prepared[at..], &[0; 40]);
        } else {
            assert_eq!(prepared, payload);
        }
    }
    let dir = tempfile::tempdir()?;
    let wal_dir = dir.path().join("wal");
    let writer = grafeo_storage::wal::WalManager::open(&wal_dir)?;
    for record in records {
        writer.log(&record)?;
    }
    writer.close()?;
    assert_eq!(
        std::fs::read(wal_dir.join("wal_00000000.log"))?,
        golden_bytes()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_wal_fixture() {
    let bytes = encode_frames(&golden_records());

    let dest = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golden_wal_v5.bin"
    );
    let dest = std::env::var("GRAFEO_WAL_FIXTURE_PATH").unwrap_or_else(|_| dest.into());
    std::fs::write(&dest, &bytes).unwrap();
    println!(
        "Wrote {} bytes ({} frames) to {dest}",
        bytes.len(),
        golden_records().len()
    );
}
