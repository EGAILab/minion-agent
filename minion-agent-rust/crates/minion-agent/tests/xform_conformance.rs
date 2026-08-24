#![cfg(feature = "conformance")]

use std::{collections::BTreeMap, fs, path::PathBuf};

use minion_agent::llm::{
    AssistantMessage, Message, ModelIdentity, TransformTarget, transform_legacy_messages,
};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn canonical_numbers(value: Value) -> Value {
    match value {
        Value::Number(number) => number
            .as_f64()
            .filter(|value| value.fract() == 0.0)
            .and_then(|value| serde_json::Number::from_i128(value as i128))
            .map_or(Value::Number(number), Value::Number),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_numbers).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonical_numbers(value)))
                .collect(),
        ),
        other => other,
    }
}

fn normalize_content(value: &mut Value) {
    let Some(content) = value.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for block in content {
        let object = block.as_object_mut().unwrap();
        match object.get("type").and_then(Value::as_str) {
            Some("text") => {
                object.entry("text_signature").or_insert(Value::Null);
            }
            Some("thinking") => {
                object.entry("thinking_signature").or_insert(Value::Null);
                object.entry("redacted").or_insert(Value::Bool(false));
            }
            Some("tool_call") => {
                object.entry("thought_signature").or_insert(Value::Null);
                object.entry("namespace").or_insert(Value::Null);
            }
            _ => {}
        }
    }
}

fn normalize_message(message: &Message, expected: &Value) -> Value {
    let mut value = serde_json::to_value(message).unwrap();
    normalize_content(&mut value);
    let object = value.as_object_mut().unwrap();
    match object.get("role").and_then(Value::as_str) {
        Some("assistant") => {
            for field in [
                "response_model",
                "response_id",
                "diagnostics",
                "deferred",
                "error_message",
                "raw_stop_reason",
                "end_turn",
            ] {
                object.entry(field).or_insert(Value::Null);
            }
            if let Some(usage) = object.get_mut("usage").and_then(Value::as_object_mut) {
                usage.entry("cache_write_1h").or_insert(Value::Null);
                usage.entry("reasoning").or_insert(Value::Null);
            }
            if let Some(diagnostics) = object.get_mut("diagnostics").and_then(Value::as_array_mut) {
                for diagnostic in diagnostics {
                    let diagnostic = diagnostic.as_object_mut().unwrap();
                    diagnostic.entry("error").or_insert(Value::Null);
                    diagnostic.entry("details").or_insert(Value::Null);
                }
            }
            if let Some(deferred) = object.get_mut("deferred").and_then(Value::as_object_mut) {
                for field in ["expires_at", "poll_after_ms", "data"] {
                    deferred.entry(field).or_insert(Value::Null);
                }
            }
        }
        Some("tool_result") => {
            for field in ["details", "usage", "added_tool_names"] {
                object.entry(field).or_insert(Value::Null);
            }
            if expected.get("timestamp").is_none() {
                object.remove("timestamp");
            }
            if let Some(usage) = object.get_mut("usage").and_then(Value::as_object_mut) {
                usage.entry("cache_write_1h").or_insert(Value::Null);
                usage.entry("reasoning").or_insert(Value::Null);
            }
        }
        _ => {}
    }
    canonical_numbers(value)
}

#[test]
fn all_layer_04_scenarios_drive_the_real_typed_rust_transformer() {
    let directory = root().join("conformance/agent");
    let mut scenarios = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .filter_map(|path| {
            let document: Value =
                serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            document
                .get("transform")
                .is_some()
                .then_some((path, document))
        })
        .collect::<Vec<_>>();
    scenarios.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(scenarios.len(), 14);

    for (path, document) in scenarios {
        let transform = document["transform"].as_object().unwrap();
        let target_spec = transform["target"].as_object().unwrap();
        let target = TransformTarget::new(
            ModelIdentity::new(
                target_spec["provider"].as_str().unwrap(),
                target_spec["api"].as_str().unwrap(),
                target_spec["model_id"].as_str().unwrap(),
            )
            .unwrap(),
            target_spec["supports_images"].as_bool().unwrap(),
        );
        let messages = transform["messages"].as_array().unwrap().clone();
        let mapping = transform
            .get("normalize_tool_call_ids")
            .and_then(Value::as_object)
            .map(|mapping| {
                mapping
                    .iter()
                    .map(|(key, value)| (key.clone(), value.as_str().unwrap().to_owned()))
                    .collect::<BTreeMap<_, _>>()
            });
        let mut policy = |id: &str, _: &TransformTarget, _: &AssistantMessage| {
            mapping
                .as_ref()
                .and_then(|mapping| mapping.get(id))
                .cloned()
                .unwrap_or_else(|| id.to_owned())
        };
        let actual = transform_legacy_messages(
            &messages,
            &target,
            mapping
                .as_ref()
                .map(|_| &mut policy as &mut dyn minion_agent::llm::ToolCallIdNormalizer),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let expected = document["expect"]["messages"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len(), "{}", path.display());
        let normalized = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| normalize_message(actual, expected))
            .collect::<Vec<_>>();
        assert_eq!(
            Value::Array(normalized),
            Value::Array(expected.clone()),
            "{}",
            path.display()
        );
    }
}
