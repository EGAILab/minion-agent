//! Live runtime-validation schemas. Provider JSON projection is a distinct,
//! fallible boundary; it never supplies the schema used by Layer 06.
use super::PreparedValue;
use crate::llm::JsonSchemaObject;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeSchemaObject(PreparedValue);

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RuntimeSchemaError {
    #[error("runtime-validation schema must be an object with finite numeric leaves")]
    InvalidDomain,
    #[error("runtime-validation schema cannot be projected to scalar-only provider JSON")]
    NonScalarProjection,
}

impl RuntimeSchemaObject {
    pub fn as_value(&self) -> &PreparedValue {
        &self.0
    }

    pub fn try_to_json(&self) -> Result<JsonSchemaObject, RuntimeSchemaError> {
        let value = self
            .0
            .try_to_json()
            .map_err(|_| RuntimeSchemaError::NonScalarProjection)?;
        JsonSchemaObject::try_from(value).map_err(|_| RuntimeSchemaError::InvalidDomain)
    }
}

impl TryFrom<PreparedValue> for RuntimeSchemaObject {
    type Error = RuntimeSchemaError;
    fn try_from(value: PreparedValue) -> Result<Self, Self::Error> {
        fn finite(value: &PreparedValue) -> bool {
            match value {
                PreparedValue::Number(n) => n.is_finite(),
                PreparedValue::Array(a) => a.iter().all(finite),
                PreparedValue::Object(o) => o.values().all(finite),
                _ => true,
            }
        }
        if value.as_object().is_none() || !finite(&value) {
            return Err(RuntimeSchemaError::InvalidDomain);
        }
        Ok(Self(value))
    }
}

impl From<JsonSchemaObject> for RuntimeSchemaObject {
    fn from(value: JsonSchemaObject) -> Self {
        Self(PreparedValue::from(serde_json::Value::from(value)))
    }
}
