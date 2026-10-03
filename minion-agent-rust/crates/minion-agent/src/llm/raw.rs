//! Live raw tool arguments: UTF-16 strings/keys and binary64 numbers.
//! JSON projection is explicit and fallible, never the live log.
use crate::argument_graph::{ArgumentArray, ArgumentObjectRef};
pub use crate::javascript::JsString as RawString;
use crate::javascript::{JsJsonValue, js_json_loads};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum RawValue {
    Null,
    Bool(bool),
    Number(RawNumber),
    String(RawString),
    Array(ArgumentArray<Self>),
    Object(ArgumentObjectRef<RawString, Self>),
}

/// Binary64, including signed zero and infinities, but not NaN.
#[derive(Clone, Copy, Debug)]
pub struct RawNumber(f64);
impl RawNumber {
    pub fn new(value: f64) -> Result<Self, RawValueError> {
        if value.is_nan() {
            Err(RawValueError::NaN)
        } else {
            Ok(Self(value))
        }
    }
    pub fn as_f64(self) -> f64 {
        self.0
    }
}
impl PartialEq for RawNumber {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RawValueError {
    #[error("NaN is outside the raw JSON.parse value domain")]
    NaN,
    #[error("raw value is not representable by serde_json")]
    NonJson,
    #[error("invalid raw argument JSON: {0}")]
    Decode(String),
}
impl RawValue {
    /// Decode JavaScript argument text even when the text itself contains a
    /// literal unpaired surrogate inside a JSON string (not only a `\\u` escape).
    pub fn decode_utf16(text: &RawString) -> Result<Self, RawValueError> {
        use std::fmt::Write;
        let mut scalar = String::new();
        let mut quoted = false;
        let mut escaped = false;
        for character in char::decode_utf16(text.code_units().iter().copied()) {
            match character {
                Ok(character) => {
                    scalar.push(character);
                    if quoted && escaped {
                        escaped = false;
                    } else if quoted && character == '\\' {
                        escaped = true;
                    } else if character == '"' {
                        quoted = !quoted;
                    }
                }
                Err(error) if quoted && !escaped => {
                    write!(scalar, "\\u{:04x}", error.unpaired_surrogate())
                        .expect("writing to String cannot fail");
                }
                Err(_) => {
                    return Err(RawValueError::Decode(
                        "invalid surrogate outside a JSON string or after an escape".into(),
                    ));
                }
            }
        }
        Self::decode(&scalar)
    }

    pub fn decode(text: &str) -> Result<Self, RawValueError> {
        let value =
            js_json_loads(text).map_err(|error| RawValueError::Decode(error.to_string()))?;
        Self::from_javascript(value)
    }
    fn from_javascript(value: JsJsonValue) -> Result<Self, RawValueError> {
        Ok(match value {
            JsJsonValue::Null => Self::Null,
            JsJsonValue::Bool(value) => Self::Bool(value),
            JsJsonValue::Number(value) => Self::Number(RawNumber::new(value)?),
            JsJsonValue::String(value) => Self::String(value),
            JsJsonValue::Array(values) => Self::Array(
                values
                    .into_iter()
                    .map(Self::from_javascript)
                    .collect::<Result<_, _>>()?,
            ),
            JsJsonValue::Object(values) => Self::Object(
                values
                    .into_iter()
                    .map(|(key, value)| Ok((key, Self::from_javascript(value)?)))
                    .collect::<Result<_, RawValueError>>()?,
            ),
        })
    }
    pub fn get(&self, key: &str) -> Option<Self> {
        let Self::Object(values) = self else {
            return None;
        };
        values.get(&RawString::from_code_units(key.encode_utf16().collect()))
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(value.as_f64()),
            _ => None,
        }
    }
    pub fn as_string(&self) -> Option<&RawString> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
    pub fn try_to_json(&self) -> Result<Value, RawValueError> {
        Ok(match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => {
                let value = value.as_f64();
                if !value.is_finite() {
                    return Err(RawValueError::NonJson);
                }
                if value == 0.0 && value.is_sign_negative() {
                    serde_json::json!(-0.0)
                } else if value.fract() == 0.0
                    && value >= i64::MIN as f64
                    && value < 9223372036854775808.0
                {
                    Value::from(value as i64)
                } else if value.fract() == 0.0 && (0.0..18446744073709551616.0).contains(&value) {
                    Value::from(value as u64)
                } else {
                    Value::Number(
                        serde_json::Number::from_f64(value).ok_or(RawValueError::NonJson)?,
                    )
                }
            }
            Self::String(value) => Value::String(value.to_string().ok_or(RawValueError::NonJson)?),
            Self::Array(values) => Value::Array(
                values
                    .iter()
                    .map(|value| value.try_to_json())
                    .collect::<Result<_, _>>()?,
            ),
            Self::Object(values) => Value::Object(
                values
                    .iter()
                    .map(|(key, value)| {
                        Ok((
                            key.to_string().ok_or(RawValueError::NonJson)?,
                            value.try_to_json()?,
                        ))
                    })
                    .collect::<Result<_, RawValueError>>()?,
            ),
        })
    }
}
impl From<Value> for RawValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(RawNumber(
                value
                    .as_f64()
                    .expect("serde_json number is binary64 convertible"),
            )),
            Value::String(value) => {
                Self::String(RawString::from_code_units(value.encode_utf16().collect()))
            }
            Value::Array(values) => Self::Array(values.into_iter().map(Self::from).collect()),
            Value::Object(values) => Self::Object(
                values
                    .into_iter()
                    .map(|(key, value)| {
                        (
                            RawString::from_code_units(key.encode_utf16().collect()),
                            Self::from(value),
                        )
                    })
                    .collect(),
            ),
        }
    }
}
impl From<BTreeMap<String, Value>> for RawValue {
    fn from(values: BTreeMap<String, Value>) -> Self {
        Self::from(Value::Object(values.into_iter().collect()))
    }
}
impl PartialEq<Value> for RawValue {
    fn eq(&self, other: &Value) -> bool {
        self == &Self::from(other.clone())
    }
}
impl Serialize for RawValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.try_to_json()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for RawValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Value::deserialize(deserializer).map(Self::from)
    }
}
