use serde_json::{Value, json};
use thiserror::Error;

use super::{Message, ToolCallIdNormalizer, TransformTarget, transform_messages};

/// Failure to decode a legacy/dynamic message into the certified typed vocabulary.
#[derive(Debug, Error)]
#[error("legacy message is not valid Minion message data: {0}")]
pub struct TransformCompatError(#[from] serde_json::Error);

/// Normalize legacy null/absent content at the dynamic boundary, then invoke typed XFORM.
pub fn transform_legacy_messages(
    messages: &[Value],
    target: &TransformTarget,
    normalizer: Option<&mut dyn ToolCallIdNormalizer>,
) -> Result<Vec<Message>, TransformCompatError> {
    let typed = messages
        .iter()
        .cloned()
        .map(normalize_and_decode)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(transform_messages(&typed, target, normalizer))
}

fn normalize_and_decode(mut value: Value) -> Result<Message, serde_json::Error> {
    if let Some(object) = value.as_object_mut()
        && object.get("content").is_none_or(Value::is_null)
    {
        object.insert("content".into(), json!([]));
    }
    serde_json::from_value(value)
}
