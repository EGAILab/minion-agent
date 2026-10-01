#![cfg(feature = "conformance")]

use minion_agent::{
    PluginSpec, Runtime,
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, RawNumber, RawString,
        RawValue, StopReason, TextBlock, ToolCall, ToolResultContentBlock, Usage,
    },
    session::Session,
    tools::{
        AgentToolResult, BeforeToolCallAction, PreparedValue, ToolDefinition, ToolExecutionOptions,
        ToolExecutionRequest, execute_tool_calls, register_before_tool_call_hook,
        tool_execution_start_spec, tool_execution_update_spec,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
fn units(value: &Value) -> Vec<u16> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn token(value: &str) -> f64 {
    match value {
        "+Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        "-0" => -0.0,
        literal => {
            let n = literal.parse::<f64>().unwrap();
            assert!(n.is_finite());
            assert_eq!(
                ryu_js::Buffer::new().format(n),
                literal,
                "non-canonical number token"
            );
            n
        }
    }
}
fn fixture(value: &Value) -> RawValue {
    match value {
        Value::Object(o) if o.contains_key("utf16") => {
            RawValue::String(RawString::from_code_units(units(&o["utf16"])))
        }
        Value::Object(o) if o.contains_key("number") => {
            RawValue::Number(RawNumber::new(token(o["number"].as_str().unwrap())).unwrap())
        }
        Value::Object(o) if o.contains_key("$keys") => RawValue::Object(
            o["$keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| {
                    (
                        RawString::from_code_units(units(&entry[0])),
                        fixture(&entry[1]),
                    )
                })
                .collect(),
        ),
        Value::Object(o) => RawValue::Object(
            o.iter()
                .map(|(key, value)| {
                    (
                        RawString::from_code_units(key.encode_utf16().collect()),
                        fixture(value),
                    )
                })
                .collect(),
        ),
        Value::Array(values) => RawValue::Array(values.iter().map(fixture).collect()),
        _ => value.clone().into(),
    }
}
fn number(n: f64) -> Value {
    let text = if n == f64::INFINITY {
        "+Infinity".into()
    } else if n == f64::NEG_INFINITY {
        "-Infinity".into()
    } else if n == 0.0 && n.is_sign_negative() {
        "-0".into()
    } else {
        ryu_js::Buffer::new().format(n).to_owned()
    };
    json!({"number":text})
}
fn object(mut entries: Vec<(Vec<u16>, Value)>) -> Value {
    // Key membership/value fidelity only. K1 enumeration is deliberately not asserted.
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    json!({"$keys":entries})
}
fn observe(value: &RawValue) -> Value {
    match value {
        RawValue::Null => Value::Null,
        RawValue::Bool(v) => json!(v),
        RawValue::Number(v) => number(v.as_f64()),
        RawValue::String(v) => json!({"utf16":v.code_units()}),
        RawValue::Array(a) => Value::Array(a.iter().map(observe).collect()),
        RawValue::Object(o) => object(
            o.iter()
                .map(|(k, v)| (k.code_units().to_vec(), observe(v)))
                .collect(),
        ),
    }
}
fn observe_prepared(value: &PreparedValue) -> Value {
    match value {
        PreparedValue::Null => Value::Null,
        PreparedValue::Bool(v) => json!(v),
        PreparedValue::Number(v) => number(v.as_f64()),
        PreparedValue::String(v) => json!({"utf16":v.code_units()}),
        PreparedValue::Array(a) => Value::Array(a.iter().map(observe_prepared).collect()),
        PreparedValue::Object(o) => object(
            o.iter()
                .map(|(k, v)| (k.code_units().to_vec(), observe_prepared(v)))
                .collect(),
        ),
    }
}
// Expected tokens/units come directly from scenario text, never fixture().
fn expected(value: &Value) -> Value {
    match value {
        Value::Object(o) if o.contains_key("utf16") || o.contains_key("number") => value.clone(),
        Value::Object(o) if o.contains_key("$keys") => object(
            o["$keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| (units(&entry[0]), expected(&entry[1])))
                .collect(),
        ),
        Value::Object(o) => object(
            o.iter()
                .map(|(key, value)| (key.encode_utf16().collect(), expected(value)))
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(expected).collect()),
        _ => value.clone(),
    }
}
fn result() -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new("ok").into())],
        details: json!({}).into(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

async fn run(case: &Value) {
    let expected = expected(&case["arguments"]);
    let raw = fixture(&case["arguments"]);
    assert_eq!(
        observe(&RawValue::decode(case["provider_text"].as_str().unwrap()).unwrap()),
        expected,
        "decoder {}",
        case["id"]
    );
    let call = ToolCall::new_raw("call", "probe", raw);
    assert_eq!(observe(&call.arguments), expected);
    let message = Message::Assistant(Box::new(AssistantMessage::new(
        ModelIdentity::new("mock", "model", "mock").unwrap(),
        vec![AssistantContentBlock::ToolCall(call)],
        Usage::default(),
        StopReason::ToolUse,
        0.0,
    )));
    let session = Session::new("raw", std::iter::empty::<String>()).unwrap();
    let appended = session.append_message(message.clone()).unwrap();
    assert_eq!(appended.data["message"].as_message(), Some(&message));
    let replayed = session.derive_messages().unwrap();
    assert_eq!(replayed, vec![message]);
    let Message::Assistant(replayed) = &replayed[0] else {
        panic!()
    };
    let AssistantContentBlock::ToolCall(call) = &replayed.content[0] else {
        panic!()
    };
    assert_eq!(observe(&call.arguments), expected);
    let runtime = Runtime::new();
    let observations = Arc::new(Mutex::new(BTreeMap::<String, Vec<Value>>::new()));
    let execution = observations.clone();
    runtime
        .tools()
        .register_for_scope(
            None,
            ToolDefinition::new(
                "probe",
                "probe",
                serde_json::from_value(json!({"type":"object","properties":{}})).unwrap(),
                "probe",
                move |request: ToolExecutionRequest| {
                    execution
                        .lock()
                        .entry("execute".into())
                        .or_default()
                        .push(observe_prepared(&request.params));
                    if let Some(update) = request.on_update {
                        update(result());
                    }
                    Box::pin(async { Ok(result()) })
                },
            ),
        )
        .unwrap();
    let event_observations = observations.clone();
    let plugin = PluginSpec::<Value>::new(
        "raw-observer",
        vec![],
        || json!({}),
        move |context, _| {
            let observations = event_observations.clone();
            async move {
                let bus = context.events().unwrap();
                let start = tool_execution_start_spec();
                let update = tool_execution_update_spec();
                bus.declare(&start).unwrap();
                bus.declare(&update).unwrap();
                bus.on_emit(&start, &context.effect_store(), context.scope(), {
                    let o = observations.clone();
                    move |e| {
                        o.lock()
                            .entry("start".into())
                            .or_default()
                            .push(observe(&e.arguments));
                    }
                })
                .unwrap();
                bus.on_emit(&update, &context.effect_store(), context.scope(), {
                    let o = observations.clone();
                    move |e| {
                        o.lock()
                            .entry("update-event".into())
                            .or_default()
                            .push(observe(&e.arguments));
                    }
                })
                .unwrap();
                register_before_tool_call_hook(&context, move |current| {
                    observations
                        .lock()
                        .entry("hook".into())
                        .or_default()
                        .push(observe_prepared(&current.arguments));
                    async { Ok(BeforeToolCallAction::Proceed(None)) }
                })
                .unwrap();
                Ok(())
            }
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let update_delivery = observations.clone();
    let options =
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_execution_update(move |e| {
            update_delivery
                .lock()
                .entry("update-delivery".into())
                .or_default()
                .push(observe(&e.arguments));
            async { Ok(()) }
        });
    let batch = execute_tool_calls(&runtime.context(), std::slice::from_ref(call), options)
        .await
        .unwrap();
    assert!(
        !batch.messages[0].is_error,
        "{} {:?}",
        case["id"], batch.messages
    );
    let observations = observations.lock();
    assert_eq!(observations.len(), 5);
    for (boundary, values) in observations.iter() {
        assert_eq!(values, &vec![expected.clone()], "{} {boundary}", case["id"]);
    }
}
#[tokio::test]
async fn raw_arguments_canonical_live_boundaries() {
    let schema: Value = serde_json::from_str(
        &fs::read_to_string(root().join("conformance/schema/raw-arguments-scenario.schema.json"))
            .unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mut cases = vec![];
    for file in fs::read_dir(root().join("conformance/agent/raw-arguments")).unwrap() {
        let document: Value =
            serde_json::from_str(&fs::read_to_string(file.unwrap().path()).unwrap()).unwrap();
        validator.validate(&document).unwrap();
        cases.extend(
            document["raw_arguments"]["cases"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
    }
    cases.sort_by_key(|case| case["id"].as_str().unwrap().to_owned());
    eprintln!("raw argument delta gate: {} cases", cases.len());
    assert!(!cases.is_empty());
    for case in cases {
        run(&case).await;
    }
}
