//! One-pass frame preparation with an explicit retained-capacity limit.

use super::MAX_WAL_FRAME_BYTES;
use bincode::enc::write::Writer;
use bincode::error::EncodeError;
use grafeo_common::utils::error::{Error, Result, StorageError};
use serde::Serialize;

/// Current record envelope, inside the existing checksum/authentication boundary.
pub(super) const RECORD_HEADER: &[u8; 10] = super::group::HEADER;

pub(super) fn record_bytes(data: &[u8]) -> std::result::Result<&[u8], String> {
    super::group::envelope(data).map(|envelope| envelope.record)
}

/// Encodes one validated record as a bounded current-generation WAL payload.
///
/// Pass these bytes to an async backend's `write_wal_batch`; physical framing
/// and checksum/authentication are applied by the backend, not this function.
/// Commit and abort payloads reserve a zeroed group seal that the serialized
/// WAL owner fills after accepting the preceding records in physical order.
///
/// # Errors
/// Returns record-validation, serialization or frame-capacity errors.
pub fn encode_record<R: super::WalEntry>(record: &R) -> Result<Vec<u8>> {
    record
        .validate_recovery()
        .map_err(|reason| Error::InvalidValue(format!("invalid WAL record: {reason}")))?;
    encode_frame(record)
}

pub(super) struct CappedFrameBuffer {
    bytes: Vec<u8>,
    failed: Option<Error>,
    #[cfg(test)]
    allocation_fault: Option<AllocationFault>,
}

#[cfg(test)]
enum AllocationFault {
    Refuse,
    Oversized,
}

impl CappedFrameBuffer {
    pub(super) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            failed: None,
            #[cfg(test)]
            allocation_fault: None,
        }
    }

    fn required_length(length: usize, incoming: usize) -> Result<usize> {
        let required = length
            .checked_add(incoming)
            .ok_or_else(|| Error::InvalidValue("WAL frame length overflow".into()))?;
        super::validate_wal_frame_payload_len(required)?;
        Ok(required)
    }

    fn growth_target(capacity: usize, required: usize) -> Result<usize> {
        capacity
            .checked_mul(2)
            .ok_or(Error::Storage(StorageError::Full))
            .map(|doubled| doubled.max(256).max(required).min(MAX_WAL_FRAME_BYTES))
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        #[cfg(test)]
        match self.allocation_fault.take() {
            Some(AllocationFault::Refuse) => {
                self.bytes
                    .try_reserve_exact(usize::MAX)
                    .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            }
            Some(AllocationFault::Oversized) => {
                self.bytes
                    .try_reserve_exact(MAX_WAL_FRAME_BYTES + 1)
                    .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            }
            None => {}
        }
        if required > self.bytes.capacity() {
            let target = Self::growth_target(self.bytes.capacity(), required)?;
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        }
        if self.bytes.capacity() > MAX_WAL_FRAME_BYTES {
            self.bytes = Vec::new();
            return Err(Error::Storage(StorageError::Full)
                .with_context("WAL allocator exceeded the retained frame capacity limit"));
        }
        Ok(())
    }

    pub(super) fn append(&mut self, incoming: &[u8]) -> Result<()> {
        if self.failed.is_some() {
            return Err(Error::InvalidValue(
                "WAL frame preparation already failed".into(),
            ));
        }
        let result = Self::required_length(self.bytes.len(), incoming.len())
            .and_then(|required| self.grow(required));
        if let Err(error) = result {
            self.failed = Some(error);
            return Err(Error::InvalidValue("WAL frame preparation failed".into()));
        }
        // Capacity has been proven. Unlike bulk insertion, push guarantees it
        // does not reallocate while spare capacity exists.
        for byte in incoming {
            self.bytes.push(*byte);
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<Vec<u8>> {
        match self.failed {
            Some(error) => Err(error),
            None => Ok(self.bytes),
        }
    }
}

impl Writer for CappedFrameBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::result::Result<(), EncodeError> {
        self.append(bytes)
            .map_err(|_| EncodeError::Other("WAL frame preparation failed"))
    }
}

pub(super) fn encode_frame<R: super::WalEntry>(record: &R) -> Result<Vec<u8>> {
    encode_parts(record, super::group::Coordinate::of(record)?)
}

fn encode_parts<R: Serialize>(record: &R, coordinate: super::group::Coordinate) -> Result<Vec<u8>> {
    let mut buffer = CappedFrameBuffer::new();
    let encoded = buffer
        .write(RECORD_HEADER)
        .and_then(|()| buffer.write(&coordinate.bytes()))
        .and_then(|()| {
            bincode::serde::encode_into_writer(record, &mut buffer, bincode::config::standard())
        })
        .and_then(|()| {
            if coordinate.has_seal() {
                // The sole serialized WAL owner fills this reservation only after
                // every earlier group record has been accepted in physical order.
                buffer.write(&[0; super::group::SEAL_LEN])
            } else {
                Ok(())
            }
        });
    // A serializer can swallow Writer errors; finalization owns the original
    // latched cause and must run before inspecting the encoder result.
    let bytes = buffer.finish()?;
    encoded.map_err(|error| Error::Serialization(error.to_string()))?;
    Ok(bytes)
}

pub(super) fn copy_frame(bytes: &[u8]) -> Result<Vec<u8>> {
    super::validate_wal_frame_payload_len(bytes.len())?;
    record_bytes(bytes).map_err(Error::InvalidValue)?;
    let mut buffer = CappedFrameBuffer::new();
    let copied = buffer.append(bytes);
    let result = buffer.finish()?;
    copied?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_limits_are_inclusive_and_writer_failure_is_sticky() {
        let mut buffer = CappedFrameBuffer::new();
        buffer.append(&[]).unwrap();
        assert_eq!(buffer.bytes.capacity(), 0);
        buffer.append(&vec![7; MAX_WAL_FRAME_BYTES]).unwrap();
        assert_eq!(buffer.bytes.len(), MAX_WAL_FRAME_BYTES);
        assert!(buffer.bytes.capacity() <= MAX_WAL_FRAME_BYTES);
        assert!(buffer.write(&[8]).is_err());
        assert!(buffer.write(&[]).is_err());
        assert!(buffer.finish().is_err());
        assert!(CappedFrameBuffer::required_length(usize::MAX, 1).is_err());
        assert!(copy_frame(&vec![0; MAX_WAL_FRAME_BYTES + 1]).is_err());
    }

    #[test]
    fn geometric_growth_preserves_every_byte_within_capacity() {
        let mut buffer = CappedFrameBuffer::new();
        for _ in 0..513 {
            buffer.append(&[1, 2, 3]).unwrap();
            assert!(buffer.bytes.capacity() <= MAX_WAL_FRAME_BYTES);
        }
        assert_eq!(buffer.finish().unwrap(), [1, 2, 3].repeat(513));
        // Independently derived standard-bincode u16 encoding.
        assert_eq!(
            encode_parts(&300_u16, super::super::group::Coordinate::untagged_data()).unwrap(),
            b"GRAFOWAL\x05\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfb\x2c\x01"
        );
        assert!(CappedFrameBuffer::growth_target(usize::MAX, 1).is_err());
    }

    #[test]
    fn typed_records_match_literal_and_retained_golden_payloads() {
        use crate::wal::WalRecord;
        use grafeo_common::types::{NodeId, TransactionId};

        let fixture = include_bytes!("../../tests/fixtures/golden_wal_v5.bin");
        // Mutation variant 25, transaction 1, four root-path bytes, operation 0,
        // node 1, one label, six UTF-8 bytes: no second encoder.
        let create_node = b"GRAFOWAL\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x19\x01\x04\x00\x00\x00\x00\x00\x01\x01\x06Person";
        assert_eq!(&fixture[..4], &[36, 0, 0, 0]);
        assert_eq!(&fixture[4..40], create_node);
        assert_eq!(
            encode_frame(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".into()],
                }
            ))
            .unwrap(),
            create_node
        );

        // Current envelope, variant 22 and transaction 1.
        let checkpoint = b"GRAFOWAL\x05\x00\x04\x01\x00\x00\x00\x00\x00\x00\x00\x16\x01";
        let last_frame = &fixture[fixture.len() - 29..];
        assert_eq!(&last_frame[..4], &[21, 0, 0, 0]);
        assert_eq!(&last_frame[4..25], checkpoint);
        assert_eq!(
            encode_frame(&WalRecord::Checkpoint {
                transaction_id: TransactionId::new(1),
            })
            .unwrap(),
            checkpoint
        );
    }

    #[test]
    fn reservation_error_keeps_its_original_cause_and_oversized_capacity_is_retired() {
        let mut refused = CappedFrameBuffer::new();
        refused.allocation_fault = Some(AllocationFault::Refuse);
        assert!(refused.append(&[1]).is_err());
        let error = refused.finish().unwrap_err();
        assert!(matches!(error, Error::Io(error)
            if error.get_ref().is_some_and(|cause| cause.is::<std::collections::TryReserveError>())));
        let mut oversized = CappedFrameBuffer::new();
        oversized.allocation_fault = Some(AllocationFault::Oversized);
        assert!(oversized.append(&[1]).is_err());
        assert_eq!(
            oversized.bytes.capacity(),
            0,
            "rejected allocation must retire before dispatch"
        );
        assert!(oversized.append(&[]).is_err());
        assert!(oversized.finish().is_err());
    }

    #[test]
    fn swallowed_serializer_error_cannot_finalize_a_frame() {
        struct Swallow;
        impl Serialize for Swallow {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                use serde::ser::SerializeTuple;
                let mut tuple = serializer.serialize_tuple(2)?;
                let _ignored = tuple.serialize_element(&vec![0_u8; MAX_WAL_FRAME_BYTES + 1]);
                tuple.end()
            }
        }
        assert!(encode_parts(&Swallow, super::super::group::Coordinate::untagged_data()).is_err());
    }
}
