//! Canonical bounded Value codec extracted from the exact LPG section.
//!
//! Tags, ordering, floating-point bits, temporal precision, and limits retain
//! the existing section bytes and are shared by current portable snapshots.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{Date, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime};
use grafeo_common::utils::error::{Error, Result};

const MAX_PAYLOAD_BYTES: usize = u32::MAX as usize;
const MAX_VALUE_DEPTH: usize = 128;

// ── Canonical Value codec ─────────────────────────────────────────

const VALUE_NULL: u8 = 0;
const VALUE_BOOL: u8 = 1;
const VALUE_I64: u8 = 2;
const VALUE_F64: u8 = 3;
const VALUE_STRING: u8 = 4;
const VALUE_BYTES: u8 = 5;
const VALUE_TIMESTAMP: u8 = 6;
const VALUE_DATE: u8 = 7;
const VALUE_TIME: u8 = 8;
const VALUE_DURATION: u8 = 9;
const VALUE_ZONED_DATETIME: u8 = 10;
const VALUE_LIST: u8 = 11;
const VALUE_MAP: u8 = 12;
const VALUE_VECTOR: u8 = 13;
const VALUE_PATH: u8 = 14;
const VALUE_GCOUNTER: u8 = 15;
const VALUE_ONCOUNTER: u8 = 16;
const VALUE_RDF_LITERAL: u8 = 17;

/// Encodes one value using the canonical bounded LPG property grammar.
///
/// # Errors
/// Rejects values beyond the shared nesting or encoded-size limits.
pub fn encode_value(value: &Value) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    encode_value_into(value, 0, &mut encoded)?;
    Ok(encoded)
}

#[allow(clippy::too_many_lines)]
fn encode_value_into(value: &Value, depth: usize, encoded: &mut Vec<u8>) -> Result<()> {
    if depth > MAX_VALUE_DEPTH {
        return Err(serialization(format!(
            "LPG v3 property value exceeds nesting depth {MAX_VALUE_DEPTH}"
        )));
    }
    match value {
        Value::Null => put_u8(encoded, VALUE_NULL)?,
        Value::Bool(value) => {
            put_u8(encoded, VALUE_BOOL)?;
            put_u8(encoded, u8::from(*value))?;
        }
        Value::Int64(value) => {
            put_u8(encoded, VALUE_I64)?;
            put_bytes(encoded, &value.to_le_bytes())?;
        }
        Value::Float64(value) => {
            put_u8(encoded, VALUE_F64)?;
            put_bytes(encoded, &value.to_bits().to_le_bytes())?;
        }
        Value::String(value) => {
            put_u8(encoded, VALUE_STRING)?;
            put_string(encoded, value)?;
        }
        Value::Bytes(value) => {
            put_u8(encoded, VALUE_BYTES)?;
            put_len(encoded, value.len())?;
            put_bytes(encoded, value)?;
        }
        Value::Timestamp(value) => {
            put_u8(encoded, VALUE_TIMESTAMP)?;
            put_bytes(encoded, &value.as_micros().to_le_bytes())?;
        }
        Value::Date(value) => {
            put_u8(encoded, VALUE_DATE)?;
            put_bytes(encoded, &value.as_days().to_le_bytes())?;
        }
        Value::Time(value) => {
            put_u8(encoded, VALUE_TIME)?;
            put_bytes(encoded, &value.as_nanos().to_le_bytes())?;
            match value.offset_seconds() {
                Some(offset) => {
                    put_u8(encoded, 1)?;
                    put_bytes(encoded, &offset.to_le_bytes())?;
                }
                None => put_u8(encoded, 0)?,
            }
        }
        Value::Duration(value) => {
            put_u8(encoded, VALUE_DURATION)?;
            put_bytes(encoded, &value.months().to_le_bytes())?;
            put_bytes(encoded, &value.days().to_le_bytes())?;
            put_bytes(encoded, &value.nanos().to_le_bytes())?;
        }
        Value::ZonedDatetime(value) => {
            put_u8(encoded, VALUE_ZONED_DATETIME)?;
            put_bytes(encoded, &value.as_timestamp().as_micros().to_le_bytes())?;
            put_bytes(encoded, &value.offset_seconds().to_le_bytes())?;
        }
        Value::List(values) => {
            put_u8(encoded, VALUE_LIST)?;
            put_len(encoded, values.len())?;
            for value in values.iter() {
                encode_value_into(value, depth + 1, encoded)?;
            }
        }
        Value::Map(values) => {
            put_u8(encoded, VALUE_MAP)?;
            put_len(encoded, values.len())?;
            for (key, value) in values.iter() {
                put_string(encoded, key.as_str())?;
                encode_value_into(value, depth + 1, encoded)?;
            }
        }
        Value::Vector(values) => {
            put_u8(encoded, VALUE_VECTOR)?;
            put_len(encoded, values.len())?;
            for value in values.iter() {
                put_bytes(encoded, &value.to_bits().to_le_bytes())?;
            }
        }
        Value::Path { nodes, edges } => {
            put_u8(encoded, VALUE_PATH)?;
            put_len(encoded, nodes.len())?;
            for value in nodes.iter() {
                encode_value_into(value, depth + 1, encoded)?;
            }
            put_len(encoded, edges.len())?;
            for value in edges.iter() {
                encode_value_into(value, depth + 1, encoded)?;
            }
        }
        Value::GCounter(values) => {
            put_u8(encoded, VALUE_GCOUNTER)?;
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
            put_len(encoded, entries.len())?;
            for (key, value) in entries {
                put_string(encoded, key)?;
                put_bytes(encoded, &value.to_le_bytes())?;
            }
        }
        Value::OnCounter { pos, neg } => {
            put_u8(encoded, VALUE_ONCOUNTER)?;
            encode_counter_map(pos, encoded)?;
            encode_counter_map(neg, encoded)?;
        }
        Value::RdfLiteral {
            lexical,
            language,
            datatype,
        } => {
            put_u8(encoded, VALUE_RDF_LITERAL)?;
            put_string(encoded, lexical)?;
            put_optional_string(encoded, language.as_deref())?;
            put_optional_string(encoded, datatype.as_deref())?;
        }
        _ => {
            return Err(serialization(
                "LPG v3 cannot encode an unknown future Value variant",
            ));
        }
    }
    Ok(())
}

fn encode_counter_map(values: &HashMap<String, u64>, encoded: &mut Vec<u8>) -> Result<()> {
    let mut entries: Vec<_> = values.iter().collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    put_len(encoded, entries.len())?;
    for (key, value) in entries {
        put_string(encoded, key)?;
        put_bytes(encoded, &value.to_le_bytes())?;
    }
    Ok(())
}

fn put_optional_string(encoded: &mut Vec<u8>, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => {
            put_u8(encoded, 1)?;
            put_string(encoded, value)
        }
        None => put_u8(encoded, 0),
    }
}

fn put_string(encoded: &mut Vec<u8>, value: &str) -> Result<()> {
    put_len(encoded, value.len())?;
    put_bytes(encoded, value.as_bytes())
}

fn put_len(encoded: &mut Vec<u8>, len: usize) -> Result<()> {
    let len = u32::try_from(len)
        .map_err(|_| serialization("LPG v3 value collection exceeds u32::MAX"))?;
    put_bytes(encoded, &len.to_le_bytes())
}

fn put_u8(encoded: &mut Vec<u8>, value: u8) -> Result<()> {
    put_bytes(encoded, &[value])
}

fn put_bytes(encoded: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let new_len = encoded
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| serialization("LPG v3 value length overflow"))?;
    if u32::try_from(new_len).is_err() {
        return Err(serialization(format!(
            "LPG v3 property value exceeds {MAX_PAYLOAD_BYTES} bytes"
        )));
    }
    encoded.extend_from_slice(bytes);
    Ok(())
}

/// Decodes exactly one canonical LPG property value.
///
/// # Errors
/// Rejects malformed, over-deep, truncated or trailing value bytes.
pub fn decode_value_exact(encoded: &[u8]) -> Result<Value> {
    let mut decoder = ValueDecoder {
        encoded,
        position: 0,
    };
    let value = decoder.decode(0)?;
    if decoder.position != encoded.len() {
        return Err(serialization(format!(
            "trailing LPG v3 property-value bytes: {}",
            encoded.len() - decoder.position
        )));
    }
    Ok(value)
}

struct ValueDecoder<'a> {
    encoded: &'a [u8],
    position: usize,
}

impl ValueDecoder<'_> {
    #[allow(clippy::too_many_lines)]
    fn decode(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_VALUE_DEPTH {
            return Err(serialization(format!(
                "LPG v3 property value exceeds nesting depth {MAX_VALUE_DEPTH}"
            )));
        }
        match self.u8()? {
            VALUE_NULL => Ok(Value::Null),
            VALUE_BOOL => match self.u8()? {
                0 => Ok(Value::Bool(false)),
                1 => Ok(Value::Bool(true)),
                value => Err(serialization(format!(
                    "invalid LPG v3 boolean payload {value}"
                ))),
            },
            VALUE_I64 => Ok(Value::Int64(i64::from_le_bytes(self.array()?))),
            VALUE_F64 => Ok(Value::Float64(f64::from_bits(u64::from_le_bytes(
                self.array()?,
            )))),
            VALUE_STRING => Ok(Value::String(ArcStr::from(self.string()?))),
            VALUE_BYTES => Ok(Value::Bytes(Arc::from(self.length_delimited()?))),
            VALUE_TIMESTAMP => Ok(Value::Timestamp(Timestamp::from_micros(
                i64::from_le_bytes(self.array()?),
            ))),
            VALUE_DATE => Ok(Value::Date(Date::from_days(i32::from_le_bytes(
                self.array()?,
            )))),
            VALUE_TIME => {
                let nanos = u64::from_le_bytes(self.array()?);
                let mut time = Time::from_nanos(nanos).ok_or_else(|| {
                    serialization(format!("invalid LPG v3 time nanoseconds {nanos}"))
                })?;
                match self.u8()? {
                    0 => {}
                    1 => time = time.with_offset(i32::from_le_bytes(self.array()?)),
                    flag => {
                        return Err(serialization(format!(
                            "invalid LPG v3 time offset flag {flag}"
                        )));
                    }
                }
                Ok(Value::Time(time))
            }
            VALUE_DURATION => Ok(Value::Duration(Duration::new(
                i64::from_le_bytes(self.array()?),
                i64::from_le_bytes(self.array()?),
                i64::from_le_bytes(self.array()?),
            ))),
            VALUE_ZONED_DATETIME => Ok(Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(i64::from_le_bytes(self.array()?)),
                i32::from_le_bytes(self.array()?),
            ))),
            VALUE_LIST => {
                let count = self.collection_count(1)?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.decode(depth + 1)?);
                }
                Ok(Value::List(values.into()))
            }
            VALUE_MAP => {
                let count = self.collection_count(5)?;
                let mut values = BTreeMap::new();
                let mut prior = None;
                for _ in 0..count {
                    let key = self.string()?.to_owned();
                    ensure_strict_key_order(prior.as_deref(), &key, "map")?;
                    prior = Some(key.clone());
                    values.insert(PropertyKey::new(key), self.decode(depth + 1)?);
                }
                Ok(Value::Map(Arc::new(values)))
            }
            VALUE_VECTOR => {
                let count = self.collection_count(4)?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(f32::from_bits(u32::from_le_bytes(self.array()?)));
                }
                Ok(Value::Vector(values.into()))
            }
            VALUE_PATH => {
                let node_count = self.collection_count(1)?;
                let mut nodes = Vec::with_capacity(node_count);
                for _ in 0..node_count {
                    nodes.push(self.decode(depth + 1)?);
                }
                let edge_count = self.collection_count(1)?;
                let mut edges = Vec::with_capacity(edge_count);
                for _ in 0..edge_count {
                    edges.push(self.decode(depth + 1)?);
                }
                Ok(Value::Path {
                    nodes: nodes.into(),
                    edges: edges.into(),
                })
            }
            VALUE_GCOUNTER => Ok(Value::GCounter(Arc::new(self.counter_map()?))),
            VALUE_ONCOUNTER => Ok(Value::OnCounter {
                pos: Arc::new(self.counter_map()?),
                neg: Arc::new(self.counter_map()?),
            }),
            VALUE_RDF_LITERAL => Ok(Value::RdfLiteral {
                lexical: ArcStr::from(self.string()?),
                language: self.optional_string()?.map(ArcStr::from),
                datatype: self.optional_string()?.map(ArcStr::from),
            }),
            tag => Err(serialization(format!(
                "unknown LPG v3 property-value tag {tag}"
            ))),
        }
    }

    fn counter_map(&mut self) -> Result<HashMap<String, u64>> {
        let count = self.collection_count(12)?;
        let mut values = HashMap::with_capacity(count);
        let mut prior = None;
        for _ in 0..count {
            let key = self.string()?.to_owned();
            ensure_strict_key_order(prior.as_deref(), &key, "counter")?;
            prior = Some(key.clone());
            values.insert(key, u64::from_le_bytes(self.array()?));
        }
        Ok(values)
    }

    fn optional_string(&mut self) -> Result<Option<&str>> {
        match self.u8()? {
            0 => Ok(None),
            1 => self.string().map(Some),
            flag => Err(serialization(format!(
                "invalid LPG v3 optional-string flag {flag}"
            ))),
        }
    }

    fn collection_count(&mut self, minimum_encoded_bytes: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        let minimum = count
            .checked_mul(minimum_encoded_bytes)
            .ok_or_else(|| serialization("LPG v3 value collection length overflow"))?;
        if minimum > self.remaining() {
            return Err(serialization(format!(
                "hostile LPG v3 collection length {count} exceeds remaining value bytes"
            )));
        }
        Ok(count)
    }

    fn length_delimited(&mut self) -> Result<&[u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn string(&mut self) -> Result<&str> {
        let bytes = self.length_delimited()?;
        std::str::from_utf8(bytes)
            .map_err(|error| serialization(format!("invalid LPG v3 UTF-8 string: {error}")))
    }

    fn u8(&mut self) -> Result<u8> {
        let [byte] = self.array()?;
        Ok(byte)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?
            .try_into()
            .map_err(|_| serialization("truncated LPG v3 scalar"))
    }

    fn take(&mut self, len: usize) -> Result<&[u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| serialization("LPG v3 value offset overflow"))?;
        let bytes = self.encoded.get(self.position..end).ok_or_else(|| {
            serialization(format!(
                "truncated LPG v3 property value at byte {}",
                self.position
            ))
        })?;
        self.position = end;
        Ok(bytes)
    }

    fn remaining(&self) -> usize {
        self.encoded.len() - self.position
    }
}

fn ensure_strict_key_order(prior: Option<&str>, current: &str, kind: &str) -> Result<()> {
    if prior.is_some_and(|prior| prior >= current) {
        return Err(serialization(format!(
            "LPG v3 {kind} keys are not canonical and unique"
        )));
    }
    Ok(())
}

fn serialization(message: impl Into<String>) -> Error {
    Error::Serialization(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_wire_examples_preserve_signed_and_utf8_values() {
        let integer = [2, 214, 255, 255, 255, 255, 255, 255, 255];
        assert_eq!(
            encode_value(&Value::Int64(-42)).expect("integer encoding"),
            integer
        );
        assert_eq!(
            decode_value_exact(&integer).expect("integer decoding"),
            Value::Int64(-42)
        );
        let string = [4, 2, 0, 0, 0, 195, 169];
        assert_eq!(
            encode_value(&Value::from("é")).expect("UTF-8 encoding"),
            string
        );
        assert_eq!(
            decode_value_exact(&string).expect("UTF-8 decoding"),
            Value::from("é")
        );
        for end in 0..integer.len() {
            assert!(decode_value_exact(&integer[..end]).is_err());
        }
    }

    #[test]
    fn malformed_tags_lengths_and_trailing_bytes_fail_closed() {
        for bytes in [
            vec![],
            vec![255],
            vec![1, 2],
            vec![0, 0],
            vec![11, 255, 255, 255, 255],
            vec![13, 255, 255, 255, 255],
            vec![15, 255, 255, 255, 255],
            vec![4, 1, 0, 0, 0, 255],
        ] {
            assert!(decode_value_exact(&bytes).is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn duplicate_and_descending_map_keys_are_not_repaired() {
        for keys in [*b"aa", *b"ba"] {
            let bytes = [
                12, 2, 0, 0, 0, 1, 0, 0, 0, keys[0], 0, 1, 0, 0, 0, keys[1], 0,
            ];
            assert!(decode_value_exact(&bytes).is_err());
        }
    }

    #[test]
    fn nesting_limit_is_inclusive_for_both_writer_and_reader() {
        let mut value = Value::Null;
        for _ in 0..128 {
            value = Value::List(vec![value].into());
        }
        let bytes = encode_value(&value).expect("128 nested values are admitted");
        assert_eq!(decode_value_exact(&bytes).expect("bounded decoding"), value);
        assert!(encode_value(&Value::List(vec![value].into())).is_err());
        let mut too_deep = vec![11, 1, 0, 0, 0];
        too_deep.extend_from_slice(&bytes);
        assert!(decode_value_exact(&too_deep).is_err());
    }
}
