#![cfg(feature = "conformance")]

use minion_agent::{
    PluginInitError, PluginSpec, Runtime,
    agent::{AgentDefinition, AgentInstance},
    agent_loop::{AgentEvent, AgentLoop, PromptInput, register_agent_listener},
    llm::{
        AssistantContentBlock, AssistantMessage, DoneReason, LlmService, Message, ModelIdentity,
        RawString, RawValue, ResultString, ResultTextBlock, ResultValue, Script, ScriptItem,
        ScriptedAdapter, StopReason, StreamChunk, TextBlock, ToolCall, ToolResultContentBlock,
        ToolResultMessage, Usage, UserContent, UserMessage,
    },
    session::Session,
    tools::{
        AfterToolCallOverride, AgentToolResult, ToolCapabilityError, ToolDefinition,
        register_after_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

fn string(v: &Value) -> ResultString {
    ResultString::from_code_units(
        v["utf16"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| u16::try_from(n.as_u64().unwrap()).unwrap())
            .collect(),
    )
}
fn preflight(v: &Value) -> Result<(), String> {
    match v {
        Value::Object(o) => {
            if let Some(Value::String(token)) = o.get("number")
                && !["NaN", "+Infinity", "-Infinity", "-0"].contains(&token.as_str())
            {
                let n = token
                    .parse::<f64>()
                    .map_err(|_| format!("invalid number token: {token}"))?;
                if !n.is_finite() || ryu_js::Buffer::new().format(n) != token {
                    return Err(format!("noncanonical finite binary64 token: {token}"));
                }
            }
            for value in o.values() {
                preflight(value)?;
            }
        }
        Value::Array(a) => {
            for value in a {
                preflight(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn decode(v: &Value) -> ResultValue {
    if v.get("utf16").is_some() {
        return ResultValue::String(string(v));
    }
    if let Some(n) = v.get("number") {
        let n = match n.as_str().unwrap() {
            "NaN" => f64::NAN,
            "+Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            "-0" => -0.0,
            n => n.parse::<f64>().unwrap(),
        };
        return ResultValue::number(n);
    }
    if let Some(keys) = v.get("$keys") {
        return ResultValue::Object(
            keys.as_array()
                .unwrap()
                .iter()
                .map(|entry| {
                    let k = ResultString::from_code_units(
                        entry[0]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| u16::try_from(x.as_u64().unwrap()).unwrap())
                            .collect(),
                    );
                    (k, decode(&entry[1]))
                })
                .collect(),
        );
    }
    match v {
        Value::Array(a) => ResultValue::Array(a.iter().map(decode).collect()),
        Value::Object(o) => ResultValue::Object(
            o.iter()
                .map(|(k, v)| (k.as_str().into(), decode(v)))
                .collect(),
        ),
        _ => v.clone().into(),
    }
}
fn observe(v: &ResultValue) -> Value {
    match v {
        ResultValue::String(s) => json!({"utf16":s.code_units()}),
        ResultValue::Number(n) => {
            let f = n.as_f64();
            let token = if f.is_nan() {
                "NaN".into()
            } else if f == f64::INFINITY {
                "+Infinity".into()
            } else if f == f64::NEG_INFINITY {
                "-Infinity".into()
            } else if f == 0.0 && f.is_sign_negative() {
                "-0".into()
            } else {
                ryu_js::Buffer::new().format(f).to_owned()
            };
            json!({"number":token})
        }
        ResultValue::Array(a) => Value::Array(a.iter().map(observe).collect()),
        ResultValue::Object(o) => {
            json!({"$keys":o.iter().map(|(k,v)|json!([k.code_units(),observe(v)])).collect::<Vec<_>>()})
        }
        ResultValue::Bool(b) => json!(b),
        ResultValue::Null => Value::Null,
    }
}
// Expectations are scenario observations, not values produced by the decoder.
// Only unordered object key-set observations are sorted, on both sides.
fn canonical(v: &Value) -> Value {
    match v {
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        Value::Object(o) if o.contains_key("$keys") => {
            let mut pairs = o["$keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(canonical)
                .collect::<Vec<_>>();
            pairs.sort_by_key(|x| {
                x[0].as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap())
                    .collect::<Vec<_>>()
            });
            json!({"$keys":pairs})
        }
        Value::Object(o) => {
            Value::Object(o.iter().map(|(k, v)| (k.clone(), canonical(v))).collect())
        }
        _ => v.clone(),
    }
}
fn content(v: &Value) -> Vec<ToolResultContentBlock> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| ToolResultContentBlock::Text(ResultTextBlock::new(string(s))))
        .collect()
}
fn observation(
    content: &[ToolResultContentBlock],
    details: Option<&ResultValue>,
    error: Option<bool>,
) -> Value {
    let mut v = json!({"content":content.iter().map(|b|match b {
        ToolResultContentBlock::Text(t) => json!({"utf16":t.text.code_units()}),
        _ => panic!("text corpus"),
    }).collect::<Vec<_>>(),"details": details.map(observe).unwrap_or(Value::Null)});
    if let Some(e) = error {
        v["is_error"] = json!(e);
    }
    v
}
fn message_observation(m: &ToolResultMessage, error: bool) -> Value {
    observation(&m.content, m.details.as_ref(), error.then_some(m.is_error))
}
fn identity() -> ModelIdentity {
    ModelIdentity::new("p", "a", "m").unwrap()
}
fn script(called: bool, edit: Option<&Value>) -> Script {
    let content = if called {
        let arguments = edit.map_or_else(
            || RawValue::from(json!({})),
            |edit| {
                RawValue::Object(
                    edit["arguments"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, v)| {
                            let value = if v.get("utf16").is_some() {
                                RawValue::String(RawString::from_code_units(
                                    string(v).code_units().to_vec(),
                                ))
                            } else {
                                RawValue::from(v.clone())
                            };
                            (RawString::from(k.as_str()), value)
                        })
                        .collect(),
                )
            },
        );
        vec![AssistantContentBlock::ToolCall(ToolCall::new_raw(
            "call-1",
            if edit.is_some() { "edit" } else { "probe" },
            arguments,
        ))]
    } else {
        vec![AssistantContentBlock::Text(TextBlock::new("done"))]
    };
    let message = AssistantMessage::new(
        identity(),
        content,
        Usage::default(),
        if called {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        },
        0.0,
    );
    Script::new([ScriptItem::Chunk(Box::new(StreamChunk::Done {
        reason: if called {
            DoneReason::ToolUse
        } else {
            DoneReason::Stop
        },
        message,
    }))])
}
async fn run_case(case: &Value, lossy_edit_control: bool) -> Value {
    let runtime = Runtime::new();
    let edit_root = case.get("edit").map(|edit| {
        let root = tempfile::tempdir().unwrap();
        let hex = edit["file_utf8_hex"].as_str().unwrap();
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        std::fs::write(root.path().join("f.txt"), bytes).unwrap();
        let mut tool = minion_agent::tools::builtin::create_edit_tool(Arc::new(
            minion_agent::execution::LocalFileSystem::new(root.path()),
        ));
        if lossy_edit_control {
            let execute = tool.execute().clone();
            let schema = tool.schema().unwrap();
            tool = ToolDefinition::new(
                schema.name,
                schema.description,
                schema.parameters,
                "edit",
                move |request| {
                    let execute = execute.clone();
                    Box::pin(async move {
                        let mut result = execute(request).await?;
                        fn lose(value: ResultValue) -> ResultValue {
                            match value {
                                ResultValue::String(s) => ResultValue::String(
                                    String::from_utf16_lossy(s.code_units()).into(),
                                ),
                                ResultValue::Object(o) => ResultValue::Object(
                                    o.into_iter().map(|(k, v)| (k, lose(v))).collect(),
                                ),
                                ResultValue::Array(a) => {
                                    ResultValue::Array(a.into_iter().map(lose).collect())
                                }
                                value => value,
                            }
                        }
                        // Deliberately incorrect early filesystem-style projection of
                        // runtime details; the real tool still performs all execution.
                        result.details = lose(result.details);
                        Ok(result)
                    })
                },
            )
            .with_prepare_raw_arguments(minion_agent::tools::builtin::prepare_edit_arguments);
        }
        runtime.tools().register_for_scope(None, tool).unwrap();
        root
    });
    let fixture = case["tool"].clone();
    if edit_root.is_none() {
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "probe",
                    "probe",
                    serde_json::from_value(json!({"type":"object"})).unwrap(),
                    "probe",
                    move |_| {
                        let fixture = fixture.clone();
                        Box::pin(async move {
                            if let Some(s) = fixture.get("throws") {
                                return Err(ToolCapabilityError::new(string(s)));
                            }
                            let v = &fixture["returns"];
                            Ok(AgentToolResult {
                                content: content(&v["content"]),
                                details: decode(&v["details"]),
                                usage: None,
                                added_tool_names: None,
                                terminate: None,
                            })
                        })
                    },
                ),
            )
            .unwrap();
    }
    let seen = Arc::new(Mutex::new(
        json!({"hook":null,"execution_end":null,"message":null,"session":null}),
    ));
    let hook = case["hook"].clone();
    let plugin_seen = seen.clone();
    let plugin = PluginSpec::<Value>::new(
        "result-domain-observers",
        vec![],
        || json!({}),
        move |context, _| {
            let seen = plugin_seen.clone();
            let hook = hook.clone();
            async move {
                if hook["mode"] != "none" {
                    let hs = seen.clone();
                    register_after_tool_call_hook(&context, move |result| {
                        let hs = hs.clone();
                        let hook = hook.clone();
                        async move {
                            hs.lock()["hook"] = observation(
                                &result.content,
                                result.details.as_ref(),
                                Some(result.is_error),
                            );
                            let replacement = match hook["mode"].as_str().unwrap() {
                                "observe" => None,
                                "same" => Some(
                                    AfterToolCallOverride::default()
                                        .with_content(result.content)
                                        .with_details(result.details.unwrap()),
                                ),
                                "null" => Some(
                                    AfterToolCallOverride::default()
                                        .with_details(ResultValue::Null),
                                ),
                                "replace" => {
                                    let mut value = AfterToolCallOverride::default();
                                    if let Some(c) = hook.get("content") {
                                        value = value.with_content(content(c));
                                    }
                                    if let Some(d) = hook.get("details") {
                                        value = value.with_details(decode(d));
                                    }
                                    Some(value)
                                }
                                "throws" => {
                                    return Err(ToolCapabilityError::new(string(&hook["message"])));
                                }
                                other => panic!("unknown mode {other}"),
                            };
                            Ok(replacement)
                        }
                    })
                    .map_err(|e| PluginInitError::new(e.to_string()))?;
                }
                register_agent_listener(&context, move |event| {
                    let seen = seen.clone();
                    async move {
                        match event {
                            AgentEvent::ToolExecutionEnd(e) => {
                                seen.lock()["execution_end"] = observation(
                                    &e.result.content,
                                    e.result.details.as_ref(),
                                    Some(e.result.is_error),
                                )
                            }
                            AgentEvent::MessageEnd(Message::ToolResult(m)) => {
                                seen.lock()["message"] = message_observation(&m, true)
                            }
                            _ => {}
                        };
                        Ok(())
                    }
                })
                .map_err(|e| PluginInitError::new(e.to_string()))?;
                Ok(())
            }
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let llm = Arc::new(LlmService::new());
    llm.register(
        identity(),
        Arc::new(ScriptedAdapter::new([
            script(true, case.get("edit")),
            script(false, None),
        ])),
    );
    let session = Session::new("room", [] as [&str; 0]).unwrap();
    let agent = Arc::new(AgentInstance::new(
        "room",
        AgentDefinition::new("probe", "system", identity()),
        session,
        Some(runtime.context()),
        None,
    ));
    let driver = AgentLoop::new(agent.clone(), runtime.context(), llm);
    let messages = driver
        .prompt(PromptInput::Message(Message::User(UserMessage::new(
            UserContent::Text("go".into()),
            0.0,
        ))))
        .await
        .unwrap();
    assert!(
        messages.iter().any(|m| matches!(m, Message::ToolResult(_))),
        "{}: tool message missing",
        case["id"]
    );
    let derived = agent.session().derive_messages().unwrap();
    let tool = derived
        .iter()
        .find_map(|m| match m {
            Message::ToolResult(t) => Some(t),
            _ => None,
        })
        .unwrap();
    seen.lock()["session"] = message_observation(tool, false);
    if let Some(root) = edit_root {
        seen.lock()["file_utf8_hex"] = json!(
            std::fs::read(root.path().join("f.txt"))
                .unwrap()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
    }
    seen.lock().clone()
}

#[test]
fn wp132_real_edit_result_preserves_surrogate_through_agent_and_session() {
    let doc: Value = serde_json::from_slice(
        &std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../conformance/agent/tool-result-domain/gate-wp132-edit-result.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(doc["gate"], "WP-13.2");
    let cases = doc["tool_result_domain"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1);
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for case in cases {
        let actual = executor.block_on(run_case(case, false));
        for boundary in [
            "hook",
            "execution_end",
            "message",
            "session",
            "file_utf8_hex",
        ] {
            assert_eq!(
                canonical(&actual[boundary]),
                canonical(&case["expect"][boundary]),
                "{}: {boundary}",
                case["id"]
            );
        }
        let mutant = executor.block_on(run_case(case, true));
        assert_eq!(
            mutant["file_utf8_hex"], case["expect"]["file_utf8_hex"],
            "control must not alter file encoding"
        );
        for boundary in ["hook", "execution_end", "message", "session"] {
            assert_ne!(
                canonical(&mutant[boundary]),
                canonical(&case["expect"][boundary]),
                "lossy edit-details control survived at {boundary}"
            );
        }
    }
}
fn cases() -> Vec<Value> {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/tool-result-domain");
    let mut files = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect::<Vec<_>>();
    files.sort();
    files
        .into_iter()
        .filter_map(|p| {
            let doc: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            (doc["gate"] != "WP-13.2").then(|| {
                doc["tool_result_domain"]["cases"]
                    .as_array()
                    .unwrap()
                    .clone()
            })
        })
        .flatten()
        .collect()
}
#[test]
fn real_agent_session_tool_result_domain() {
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases = cases();
    assert_eq!(cases.len(), 142, "approved delta gate");
    let mut failures = Vec::new();
    for case in cases {
        preflight(&case).expect("canonical document preflight before dispatch");
        let actual = executor.block_on(run_case(&case, false));
        for boundary in ["hook", "execution_end", "message", "session"] {
            if canonical(&actual[boundary]) != canonical(&case["expect"][boundary]) {
                failures.push(format!("{}: {boundary}", case["id"]));
                break;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {failures:?}",
        failures.len()
    );
}

#[test]
fn result_number_preflight_refuses_noncanonical_or_overflowing_literals() {
    for token in ["1.0", "1e18", "1e999", "-0.0", "nan", "inf", "-Infinityx"] {
        assert!(preflight(&json!({"number":token})).is_err(), "{token}");
    }
    for token in [
        "NaN",
        "+Infinity",
        "-Infinity",
        "-0",
        "0",
        "0.1",
        "1000000000000000100",
        "1e+21",
    ] {
        assert!(preflight(&json!({"number":token})).is_ok(), "{token}");
    }
}
