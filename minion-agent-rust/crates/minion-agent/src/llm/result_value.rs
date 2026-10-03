//! Live tool-result values (L0506-D003). JSON is a separate, fallible boundary.
use std::{collections::BTreeMap, fmt, ops::Index};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::javascript::JsString;

/// UTF-16 authority with a cached scalar view only when conversion is lossless.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ResultString {
    units: JsString,
    scalar: Option<String>,
}
impl ResultString {
    pub fn from_code_units(units: Vec<u16>) -> Self {
        let units = JsString::from_code_units(units);
        let scalar = units.to_string();
        Self { units, scalar }
    }
    pub fn code_units(&self) -> &[u16] {
        self.units.code_units()
    }
    pub fn as_str(&self) -> Option<&str> {
        self.scalar.as_deref()
    }
}
impl From<&str> for ResultString {
    fn from(value: &str) -> Self {
        Self::from_code_units(value.encode_utf16().collect())
    }
}
impl From<String> for ResultString {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}
impl From<&String> for ResultString {
    fn from(value: &String) -> Self {
        Self::from(value.as_str())
    }
}
impl From<JsString> for ResultString {
    fn from(value: JsString) -> Self {
        Self::from_code_units(value.code_units().to_vec())
    }
}
impl From<&ResultString> for JsString {
    fn from(value: &ResultString) -> Self {
        Self::from_code_units(value.code_units().to_vec())
    }
}
impl From<&ResultString> for ResultString {
    fn from(value: &ResultString) -> Self {
        value.clone()
    }
}
impl PartialEq<str> for ResultString {
    fn eq(&self, other: &str) -> bool {
        self.code_units().iter().copied().eq(other.encode_utf16())
    }
}
impl PartialEq<&str> for ResultString {
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}
impl PartialEq<String> for ResultString {
    fn eq(&self, other: &String) -> bool {
        self == other.as_str()
    }
}
impl fmt::Display for ResultString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in char::decode_utf16(self.code_units().iter().copied()) {
            match c {
                Ok(c) => write!(f, "{c}")?,
                Err(e) => write!(f, "\\u{:04x}", e.unpaired_surrogate())?,
            }
        }
        Ok(())
    }
}
impl Serialize for ResultString {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_str()
            .ok_or_else(|| serde::ser::Error::custom("tool-result UTF-16 is not scalar JSON"))?
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for ResultString {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d).map(Self::from)
    }
}

/// Binary64 including signed zero, infinities and NaN; never widens RawNumber.
#[derive(Clone, Copy, Debug)]
pub struct ResultNumber(f64);
impl ResultNumber {
    pub fn new(value: f64) -> Self {
        Self(value)
    }
    pub fn as_f64(self) -> f64 {
        self.0
    }
}
impl PartialEq for ResultNumber {
    fn eq(&self, other: &Self) -> bool {
        (self.0.is_nan() && other.0.is_nan()) || self.0.to_bits() == other.0.to_bits()
    }
}

/// Recursive runtime result domain. No tagged JSON escape hatches in live state.
#[derive(Clone, Debug, PartialEq)]
pub enum ResultValue {
    Null,
    Bool(bool),
    Number(ResultNumber),
    String(ResultString),
    Array(Vec<Self>),
    Object(BTreeMap<ResultString, Self>),
}
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("tool-result value is not representable by scalar finite JSON")]
pub struct ResultValueError;
impl ResultValue {
    pub fn number(value: f64) -> Self {
        Self::Number(ResultNumber::new(value))
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(values) => values.get(&key.into()),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => s.as_str(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(n.as_f64()),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
    pub fn try_to_json(&self) -> Result<Value, ResultValueError> {
        Ok(match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(*b),
            Self::Number(n) => {
                let v = n.as_f64();
                if !v.is_finite() {
                    return Err(ResultValueError);
                }
                if v == 0.0 && v.is_sign_negative() {
                    serde_json::json!(-0.0)
                } else if v.fract() == 0.0 && v >= i64::MIN as f64 && v < 9223372036854775808.0 {
                    Value::from(v as i64)
                } else if v.fract() == 0.0 && (0.0..18446744073709551616.0).contains(&v) {
                    Value::from(v as u64)
                } else {
                    Value::Number(serde_json::Number::from_f64(v).ok_or(ResultValueError)?)
                }
            }
            Self::String(s) => Value::String(s.as_str().ok_or(ResultValueError)?.into()),
            Self::Array(values) => Value::Array(
                values
                    .iter()
                    .map(Self::try_to_json)
                    .collect::<Result<_, _>>()?,
            ),
            Self::Object(values) => Value::Object(
                values
                    .iter()
                    .map(|(k, v)| {
                        Ok((k.as_str().ok_or(ResultValueError)?.into(), v.try_to_json()?))
                    })
                    .collect::<Result<_, ResultValueError>>()?,
            ),
        })
    }
}
impl From<Value> for ResultValue {
    fn from(v: Value) -> Self {
        match v {
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(b),
            Value::Number(n) => Self::number(n.as_f64().expect("JSON binary64")),
            Value::String(s) => Self::String(s.into()),
            Value::Array(v) => Self::Array(v.into_iter().map(Self::from).collect()),
            Value::Object(v) => {
                Self::Object(v.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
            }
        }
    }
}
impl Index<&str> for ResultValue {
    type Output = Self;
    fn index(&self, key: &str) -> &Self {
        self.get(key).unwrap_or(&Self::Null)
    }
}
impl PartialEq<Value> for ResultValue {
    fn eq(&self, other: &Value) -> bool {
        self == &Self::from(other.clone())
    }
}
impl Serialize for ResultValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.try_to_json()
            .map_err(serde::ser::Error::custom)?
            .serialize(s)
    }
}
impl<'de> Deserialize<'de> for ResultValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Value::deserialize(d).map(Self::from)
    }
}

/// Result-only text. Assistant/user text remain at their certified vocabulary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultTextBlock {
    #[serde(rename = "type")]
    kind: ResultTextKind,
    pub text: ResultString,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResultTextKind {
    Text,
}
impl ResultTextBlock {
    pub fn new(text: impl Into<ResultString>) -> Self {
        Self {
            kind: ResultTextKind::Text,
            text: text.into(),
            text_signature: None,
        }
    }
}
impl From<super::TextBlock> for ResultTextBlock {
    fn from(block: super::TextBlock) -> Self {
        Self {
            kind: ResultTextKind::Text,
            text: block.text.into(),
            text_signature: block.text_signature,
        }
    }
}
