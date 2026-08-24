use std::{collections::BTreeMap, fs, path::PathBuf};

use minion_agent::{
    llm::{
        AssistantContentBlock, AssistantMessage, AssistantMessageDiagnostic, Cost, DeferredHandle,
        Message, ModelIdentity, ToolDefinition, ToolResultContentBlock, ToolResultMessage, Usage,
        UserContent, UserMessage,
    },
    session::{EventKind, Session, SessionEvent},
};
use serde_json::{Map, Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn usage(raw: Option<&Value>) -> Usage {
    let raw = raw.and_then(Value::as_object);
    let number = |name| {
        raw.and_then(|v| v.get(name))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let optional = |name| raw.and_then(|v| v.get(name)).and_then(Value::as_u64);
    let cost = raw.and_then(|v| v.get("cost")).and_then(Value::as_object);
    let cost_number = |name| {
        cost.and_then(|v| v.get(name))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    };
    Usage {
        input: number("input"),
        output: number("output"),
        cache_read: number("cache_read"),
        cache_write: number("cache_write"),
        cache_write_1h: optional("cache_write_1h"),
        reasoning: optional("reasoning"),
        total_tokens: number("total_tokens"),
        cost: Cost {
            input: cost_number("input"),
            output: cost_number("output"),
            cache_read: cost_number("cache_read"),
            cache_write: cost_number("cache_write"),
            total: cost_number("total"),
        },
    }
}

fn content_values(spec: &Map<String, Value>) -> Vec<Value> {
    spec.get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| {
            spec.get("text")
                .and_then(Value::as_str)
                .map_or_else(Vec::new, |text| vec![json!({"type":"text", "text":text})])
        })
}

fn make_message(role: &str, spec: &Map<String, Value>) -> Message {
    let timestamp = spec.get("timestamp").and_then(Value::as_f64).unwrap_or(1.0);
    let blocks = content_values(spec);
    if role != "assistant" && role != "tool_result" {
        let content = if let Some(text) = spec.get("text").and_then(Value::as_str) {
            UserContent::Text(text.into())
        } else {
            UserContent::Blocks(
                blocks
                    .into_iter()
                    .map(|v| serde_json::from_value(v).unwrap())
                    .collect(),
            )
        };
        return Message::User(UserMessage::new(content, timestamp));
    }
    if role == "assistant" {
        let provider = spec
            .get("provider")
            .and_then(Value::as_str)
            .unwrap_or("mock");
        let api = spec.get("api").and_then(Value::as_str).unwrap_or("mock");
        let model = spec
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("mock-1");
        let identity = ModelIdentity::new(provider, api, model).unwrap();
        let content: Vec<AssistantContentBlock> = blocks
            .into_iter()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect();
        let stop_reason = serde_json::from_value(Value::String(
            spec.get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("stop")
                .into(),
        ))
        .unwrap();
        let mut message = AssistantMessage::new(
            identity,
            content,
            usage(spec.get("usage")),
            stop_reason,
            timestamp,
        );
        message.error_message = spec
            .get("error_message")
            .and_then(Value::as_str)
            .map(str::to_owned);
        message.response_model = spec
            .get("response_model")
            .and_then(Value::as_str)
            .map(str::to_owned);
        message.response_id = spec
            .get("response_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        message.raw_stop_reason = spec
            .get("raw_stop_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        message.end_turn = spec.get("end_turn").and_then(Value::as_bool);
        message.diagnostics = spec
            .get("diagnostics")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .cloned()
                    .map(|v| serde_json::from_value::<AssistantMessageDiagnostic>(v).unwrap())
                    .collect()
            });
        message.deferred = spec
            .get("deferred")
            .cloned()
            .map(|v| serde_json::from_value::<DeferredHandle>(v).unwrap());
        return Message::Assistant(Box::new(message));
    }
    let content: Vec<ToolResultContentBlock> = blocks
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
    let mut message = ToolResultMessage::new(
        spec.get("tool_call_id")
            .and_then(Value::as_str)
            .unwrap_or("t1"),
        spec.get("tool_name").and_then(Value::as_str).unwrap(),
        content,
        false,
        timestamp,
    );
    message.details = spec.get("details").cloned();
    message.usage = spec.get("usage").map(|v| usage(Some(v)));
    message.added_tool_names = spec
        .get("added_tool_names")
        .and_then(Value::as_array)
        .map(|v| {
            v.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        });
    Message::ToolResult(Box::new(message))
}

fn text_of(message: &Message) -> String {
    match message {
        Message::User(message) => match &message.content {
            UserContent::Text(text) => text.clone(),
            UserContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| {
                    serde_json::to_value(b)
                        .ok()?
                        .get("text")?
                        .as_str()
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
                .join(""),
        },
        Message::Assistant(message) => message
            .content
            .iter()
            .filter_map(|b| {
                serde_json::to_value(b)
                    .ok()?
                    .get("text")?
                    .as_str()
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>()
            .join(""),
        Message::ToolResult(message) => message
            .content
            .iter()
            .filter_map(|b| {
                serde_json::to_value(b)
                    .ok()?
                    .get("text")?
                    .as_str()
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn normalize_optional_fields(mut value: Value, tool_result: bool) -> Value {
    let object = value.as_object_mut().unwrap();
    object.remove("role");
    let optional = if tool_result {
        vec!["details", "usage", "added_tool_names"]
    } else {
        vec![
            "error_message",
            "response_model",
            "response_id",
            "diagnostics",
            "deferred",
            "raw_stop_reason",
            "end_turn",
        ]
    };
    for name in optional {
        object.entry(name).or_insert(Value::Null);
    }
    if let Some(Value::Object(usage)) = object.get_mut("usage") {
        usage.entry("cache_write_1h").or_insert(Value::Null);
        usage.entry("reasoning").or_insert(Value::Null);
    }
    if !tool_result {
        if let Some(Value::Array(diagnostics)) = object.get_mut("diagnostics") {
            for diagnostic in diagnostics {
                let d = diagnostic.as_object_mut().unwrap();
                d.entry("error").or_insert(Value::Null);
                d.entry("details").or_insert(Value::Null);
            }
        }
        if let Some(Value::Object(deferred)) = object.get_mut("deferred") {
            for name in ["expires_at", "poll_after_ms", "data"] {
                deferred.entry(name).or_insert(Value::Null);
            }
        }
    }
    if let Some(Value::Array(content)) = object.get_mut("content") {
        for block in content {
            let b = block.as_object_mut().unwrap();
            match b.get("type").and_then(Value::as_str) {
                Some("text") => {
                    b.entry("text_signature").or_insert(Value::Null);
                }
                Some("thinking") => {
                    b.entry("thinking_signature").or_insert(Value::Null);
                    b.entry("redacted").or_insert(Value::Bool(false));
                }
                Some("tool_call") => {
                    b.entry("thought_signature").or_insert(Value::Null);
                    b.entry("namespace").or_insert(Value::Null);
                }
                _ => {}
            }
        }
    }
    canonical_numbers(value)
}

fn canonical_numbers(value: Value) -> Value {
    match value {
        Value::Number(number) => number
            .as_f64()
            .filter(|v| v.fract() == 0.0)
            .and_then(|v| serde_json::Number::from_i128(v as i128))
            .map_or(Value::Number(number), Value::Number),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_numbers).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(k, v)| (k, canonical_numbers(v)))
                .collect(),
        ),
        other => other,
    }
}

#[test]
fn all_current_layer_session_scenarios_drive_the_real_typed_rust_session() {
    let directory = root().join("conformance/session");
    let mut files = fs::read_dir(directory)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 19);
    let mut executed = 0;
    for path in files {
        if path.file_name().unwrap() == "request-reconstruction-after-target-transform.yaml" {
            continue;
        }
        let document: Value = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let object = document.as_object().unwrap();
        let surface = object
            .get("surface_kinds")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        let mut session = Session::new("scenario", surface).unwrap();
        let mut last_header: Option<SessionEvent> = None;
        let mut actual_error: Option<Value> = None;
        for step in object["steps"].as_array().unwrap() {
            let step = step.as_object().unwrap();
            if let Some(spec) = step.get("append").and_then(Value::as_object) {
                let role = spec["role"].as_str().unwrap();
                session
                    .append_projectable(
                        EventKind::new(match role {
                            "user" => "user/message",
                            "assistant" => "assistant/message",
                            "tool_result" => "tool/result",
                            other => other,
                        })
                        .unwrap(),
                        make_message(role, spec),
                    )
                    .unwrap();
            } else if let Some(spec) = step.get("record_header").and_then(Value::as_object) {
                let components = spec["components"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap().into()))
                    .collect::<BTreeMap<_, _>>();
                let tools = spec
                    .get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .cloned()
                    .map(|v| serde_json::from_value::<ToolDefinition>(v).unwrap())
                    .collect();
                last_header = Some(
                    session
                        .record_header(components, spec["model"].as_str().unwrap(), tools)
                        .unwrap(),
                );
            } else if let Some(spec) = step.get("fork").and_then(Value::as_object) {
                match session.fork("fork", spec.get("at").and_then(Value::as_u64)) {
                    Ok(child) => session = child,
                    Err(
                        error @ minion_agent::session::SessionError::InvalidForkBoundary { .. },
                    ) => {
                        actual_error = Some(json!({
                            "type": "InvalidForkBoundaryError",
                            "message": error.to_string()
                        }));
                        break;
                    }
                    Err(error) => panic!("unexpected fork error in {}: {error}", path.display()),
                }
            } else if step.contains_key("reset") {
                session.reset().unwrap();
            } else if let Some(spec) = step.get("compact").and_then(Value::as_object) {
                session
                    .compact(
                        spec["summary"].as_str().unwrap(),
                        spec.get("keep").and_then(Value::as_u64).unwrap_or(0) as usize,
                    )
                    .unwrap();
            }
        }
        match object.get("expect_error") {
            Some(Value::Object(expected)) => {
                let actual = actual_error
                    .as_ref()
                    .and_then(Value::as_object)
                    .unwrap_or_else(|| panic!("{} expected an error", path.display()));
                assert_eq!(actual["type"], expected["type"], "{}", path.display());
                if let Some(fragment) = expected.get("message_contains").and_then(Value::as_str) {
                    assert!(
                        actual["message"].as_str().unwrap().contains(fragment),
                        "{}",
                        path.display()
                    );
                }
            }
            None => assert!(actual_error.is_none(), "{}", path.display()),
            Some(_) => panic!("{} has malformed expect_error", path.display()),
        }
        let messages = session.derive_messages().unwrap();
        let summaries = messages.iter().map(|m| json!({"role": match m { Message::User(_) => "user", Message::Assistant(_) => "assistant", Message::ToolResult(_) => "tool_result" }, "text": text_of(m)})).collect::<Vec<_>>();
        assert_eq!(
            Value::Array(summaries),
            object["expect_messages"],
            "{}",
            path.display()
        );
        if let Some(expected) = object.get("expect_assistant_details") {
            let actual = messages
                .iter()
                .filter_map(|m| {
                    if let Message::Assistant(v) = m {
                        Some(normalize_optional_fields(
                            serde_json::to_value(v.as_ref()).unwrap(),
                            false,
                        ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(Value::Array(actual), *expected, "{}", path.display());
        }
        if let Some(expected) = object.get("expect_tool_result_details") {
            let actual = messages
                .iter()
                .filter_map(|m| {
                    if let Message::ToolResult(v) = m {
                        Some(normalize_optional_fields(
                            serde_json::to_value(v.as_ref()).unwrap(),
                            true,
                        ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(Value::Array(actual), *expected, "{}", path.display());
        }
        if let Some(expected) = object.get("expect_artifact_count").and_then(Value::as_u64) {
            assert_eq!(
                session.artifact_count() as u64,
                expected,
                "{}",
                path.display()
            );
        }
        if let Some(expected) = object.get("expect_reconstructed_header") {
            let reconstructed = session
                .reconstruct_header(last_header.as_ref().unwrap())
                .unwrap();
            assert_eq!(
                json!({"components": reconstructed.components, "tools": reconstructed.tools}),
                *expected,
                "{}",
                path.display()
            );
        }
        if let Some(expected) = object.get("expect_event_kinds") {
            let actual = session
                .events()
                .into_iter()
                .map(|event| Value::String(event.kind.as_str().to_owned()))
                .collect::<Vec<_>>();
            assert_eq!(Value::Array(actual), *expected, "{}", path.display());
        }
        executed += 1;
    }
    assert_eq!(executed, 18);
}
