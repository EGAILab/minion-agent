//! In-memory tool arguments after preparation (TOOL-041).
//!
//! Raw tool calls remain JSON-compatible. Preparation may additionally produce
//! non-finite binary64 numbers and arbitrary UTF-16 strings, which must never be replaced by JSON nulls,
//! strings, or clamped values. This vocabulary deliberately does not implement
//! `Serialize`: any future serialization boundary must specify its projection.

use std::collections::BTreeMap;

use serde_json::{Number, Value};
use thiserror::Error;

/// A JavaScript string, including unpaired UTF-16 surrogate code units.
///
/// Scalar text is available only when conversion is lossless. The UTF-16
/// sequence remains authoritative; replacement is an explicit later boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedString {
    units: Vec<u16>,
    scalar: Option<String>,
}

impl PreparedString {
    pub fn from_code_units(units: Vec<u16>) -> Self {
        let scalar = String::from_utf16(&units).ok();
        Self { units, scalar }
    }

    pub fn code_units(&self) -> &[u16] {
        &self.units
    }

    pub fn as_str(&self) -> Option<&str> {
        self.scalar.as_deref()
    }

    pub fn code_point_len(&self) -> usize {
        char::decode_utf16(self.units.iter().copied()).count()
    }

    /// Explicit Node Buffer.from(string, "utf8") projection, not a runtime
    /// normalization. Each unpaired unit becomes one replacement character.
    pub fn to_utf8_lossy(&self) -> String {
        String::from_utf16_lossy(&self.units)
    }
}

impl From<String> for PreparedString {
    fn from(value: String) -> Self {
        Self {
            units: value.encode_utf16().collect(),
            scalar: Some(value),
        }
    }
}

impl From<&str> for PreparedString {
    fn from(value: &str) -> Self {
        value.to_owned().into()
    }
}

impl PartialOrd for PreparedString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PreparedString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.units.cmp(&other.units)
    }
}

/// The numeric domain of prepared tool arguments.
///
/// Finite JSON numbers retain their existing representation, including signed
/// floating-point zero. The three non-finite categories are explicit, rather
/// than being encoded into a JSON-compatible escape hatch.
#[derive(Clone, Debug, PartialEq)]
pub enum PreparedNumber {
    Finite(Number),
    PositiveInfinity,
    NegativeInfinity,
    NaN,
}

impl PreparedNumber {
    pub fn from_f64(value: f64) -> Self {
        if value.is_nan() {
            Self::NaN
        } else if value == f64::INFINITY {
            Self::PositiveInfinity
        } else if value == f64::NEG_INFINITY {
            Self::NegativeInfinity
        } else {
            Self::Finite(Number::from_f64(value).expect("finite binary64 is a JSON number"))
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Finite(number) => number.as_f64().expect("JSON number fits binary64"),
            Self::PositiveInfinity => f64::INFINITY,
            Self::NegativeInfinity => f64::NEG_INFINITY,
            Self::NaN => f64::NAN,
        }
    }

    pub fn is_finite(&self) -> bool {
        matches!(self, Self::Finite(_))
    }
}

/// JSON-shaped in-memory data with the additional prepared numeric domain.
#[derive(Clone, Debug, PartialEq)]
pub enum PreparedValue {
    Null,
    Bool(bool),
    Number(PreparedNumber),
    String(PreparedString),
    Array(Vec<Self>),
    Object(BTreeMap<PreparedString, Self>),
}

impl PreparedValue {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn number(value: f64) -> Self {
        Self::Number(PreparedNumber::from_f64(value))
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(number) => Some(number.as_f64()),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => value.as_str(),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<PreparedString, Self>> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<PreparedString, Self>> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(array) => Some(array),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Self> {
        self.get_code_units(&PreparedString::from(key))
    }

    pub fn get_code_units(&self, key: &PreparedString) -> Option<&Self> {
        self.as_object()?.get(key)
    }

    /// Returns JSON only when the entire value already belongs to that domain.
    ///
    /// Failure is explicit, with the first non-finite value's JSON pointer.
    /// This is not a diagnostic projection and never changes a runtime value.
    pub fn try_to_json(&self) -> Result<Value, NonJsonPreparedValue> {
        self.json_at("")
    }

    fn json_at(&self, pointer: &str) -> Result<Value, NonJsonPreparedValue> {
        match self {
            Self::Null => Ok(Value::Null),
            Self::Bool(value) => Ok(Value::Bool(*value)),
            Self::Number(PreparedNumber::Finite(number)) => Ok(Value::Number(number.clone())),
            Self::Number(_) => Err(NonJsonPreparedValue {
                pointer: pointer.to_owned(),
            }),
            Self::String(value) => value
                .as_str()
                .map(|text| Value::String(text.to_owned()))
                .ok_or_else(|| NonJsonPreparedValue {
                    pointer: pointer.to_owned(),
                }),
            Self::Array(array) => array
                .iter()
                .enumerate()
                .map(|(index, value)| value.json_at(&format!("{pointer}/{index}")))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array),
            Self::Object(object) => object
                .iter()
                .map(|(key, value)| {
                    let key = key.as_str().ok_or_else(|| NonJsonPreparedValue {
                        pointer: pointer.to_owned(),
                    })?;
                    let escaped = key.replace('~', "~0").replace('/', "~1");
                    value
                        .json_at(&format!("{pointer}/{escaped}"))
                        .map(|value| (key.to_owned(), value))
                })
                .collect::<Result<serde_json::Map<_, _>, _>>()
                .map(Value::Object),
        }
    }
}

impl std::ops::Index<&str> for PreparedValue {
    type Output = Self;
    fn index(&self, key: &str) -> &Self {
        self.get(key).unwrap_or(&Self::Null)
    }
}

impl std::ops::IndexMut<&str> for PreparedValue {
    fn index_mut(&mut self, key: &str) -> &mut Self {
        if self.is_null() {
            *self = Self::Object(BTreeMap::new());
        }
        self.as_object_mut()
            .expect("prepared value is an object")
            .entry(key.into())
            .or_insert(Self::Null)
    }
}

impl PartialEq<Value> for PreparedValue {
    fn eq(&self, other: &Value) -> bool {
        self == &Self::from(other.clone())
    }
}

impl PartialEq<i32> for PreparedValue {
    fn eq(&self, other: &i32) -> bool {
        self.as_f64() == Some(f64::from(*other))
    }
}

impl From<Value> for PreparedValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(number) => Self::Number(PreparedNumber::Finite(number)),
            Value::String(value) => Self::String(value.into()),
            Value::Array(array) => Self::Array(array.into_iter().map(Self::from).collect()),
            Value::Object(object) => Self::Object(
                object
                    .into_iter()
                    .map(|(key, value)| (key.into(), Self::from(value)))
                    .collect(),
            ),
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("prepared runtime value at {pointer:?} is not representable by serde_json")]
pub struct NonJsonPreparedValue {
    pub pointer: String,
}
