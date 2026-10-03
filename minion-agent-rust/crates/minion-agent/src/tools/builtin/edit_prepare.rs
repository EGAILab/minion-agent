//! Pi's prepareEditArguments, consuming the certified binary64 JSON parser and TOOL-041 domain.

use crate::{
    llm::{RawString, RawValue},
    tools::{PreparedValue, ToolCapabilityError},
};

fn is_string(value: PreparedValue) -> bool {
    matches!(value, PreparedValue::String(_))
}

fn single(value: &PreparedValue) -> bool {
    value.get("oldText").is_some_and(is_string) && value.get("newText").is_some_and(is_string)
}

/// Normalize Pi's array, single-object, JSON-string and legacy edit argument forms.
/// Extra runtime numbers remain binary64, including infinities and signed zero.
pub fn prepare_edit_arguments(
    raw: impl Into<RawValue>,
) -> Result<PreparedValue, ToolCapabilityError> {
    let mut value = PreparedValue::from(raw.into());
    let Some(args) = value.as_object_mut() else {
        return Ok(value);
    };
    if let Some(edits) = args.get(&"edits".into()) {
        if let PreparedValue::String(text) = edits {
            if let Ok(parsed) =
                RawValue::decode_utf16(&RawString::from_code_units(text.code_units().to_vec()))
            {
                let parsed = PreparedValue::from(parsed);
                if parsed.as_array().is_some() {
                    args.insert("edits".into(), parsed);
                } else if single(&parsed) {
                    args.insert("edits".into(), PreparedValue::Array(vec![parsed].into()));
                }
            }
        } else if single(&edits) {
            args.insert("edits".into(), PreparedValue::Array(vec![edits].into()));
        }
    }
    if single(&PreparedValue::Object(args.clone())) {
        let old = args
            .remove(&"oldText".into())
            .expect("single edit has oldText");
        let new = args
            .remove(&"newText".into())
            .expect("single edit has newText");
        let mut edits = args
            .get(&"edits".into())
            .and_then(|v| v.as_array())
            .map(|v| v.to_vec())
            .unwrap_or_default();
        edits.push(PreparedValue::Object(
            [("oldText".into(), old), ("newText".into(), new)]
                .into_iter()
                .collect(),
        ));
        args.insert("edits".into(), PreparedValue::Array(edits.into()));
    }
    Ok(value)
}
