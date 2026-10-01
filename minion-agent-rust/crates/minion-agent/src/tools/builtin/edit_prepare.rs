//! Pi's prepareEditArguments, consuming the certified binary64 JSON parser and TOOL-041 domain.

use crate::{
    auth::{JsJsonValue, js_json_loads},
    tools::{PreparedValue, ToolCapabilityError},
};
use serde_json::Value;

fn runtime(value: JsJsonValue) -> Result<PreparedValue, ToolCapabilityError> {
    Ok(match value {
        JsJsonValue::Null => PreparedValue::Null,
        JsJsonValue::Bool(v) => PreparedValue::Bool(v),
        JsJsonValue::Number(v) => PreparedValue::number(v),
        JsJsonValue::String(v) => PreparedValue::String(v.to_string().ok_or_else(|| ToolCapabilityError::new("Unpaired surrogate is not representable in the certified Rust argument domain"))?),
        JsJsonValue::Array(v) => PreparedValue::Array(v.into_iter().map(runtime).collect::<Result<_,_>>()?),
        JsJsonValue::Object(v) => PreparedValue::Object(v.into_iter().map(|(key, value)| Ok((key.to_string().ok_or_else(|| ToolCapabilityError::new("Unpaired surrogate key is not representable in the certified Rust argument domain"))?, runtime(value)?))).collect::<Result<_,ToolCapabilityError>>()?),
    })
}

fn single(value: &PreparedValue) -> bool {
    value
        .get("oldText")
        .and_then(PreparedValue::as_str)
        .is_some()
        && value
            .get("newText")
            .and_then(PreparedValue::as_str)
            .is_some()
}

/// Normalize Pi's array, single-object, JSON-string and legacy edit argument forms.
/// Extra runtime numbers remain binary64, including infinities and signed zero.
pub fn prepare_edit_arguments(raw: Value) -> Result<PreparedValue, ToolCapabilityError> {
    let mut value = PreparedValue::from(raw);
    let Some(args) = value.as_object_mut() else {
        return Ok(value);
    };
    if let Some(edits) = args.get("edits") {
        if let Some(text) = edits.as_str() {
            if let Ok(parsed) = js_json_loads(text) {
                let parsed = runtime(parsed)?;
                if parsed.as_array().is_some() {
                    args.insert("edits".into(), parsed);
                } else if single(&parsed) {
                    args.insert("edits".into(), PreparedValue::Array(vec![parsed]));
                }
            }
        } else if single(edits) {
            args.insert("edits".into(), PreparedValue::Array(vec![edits.clone()]));
        }
    }
    if single(&PreparedValue::Object(args.clone())) {
        let old = args.remove("oldText").expect("single edit has oldText");
        let new = args.remove("newText").expect("single edit has newText");
        let mut edits = args
            .get("edits")
            .and_then(PreparedValue::as_array)
            .unwrap_or(&[])
            .to_vec();
        edits.push(PreparedValue::Object(
            [("oldText".into(), old), ("newText".into(), new)]
                .into_iter()
                .collect(),
        ));
        args.insert("edits".into(), PreparedValue::Array(edits));
    }
    Ok(value)
}
