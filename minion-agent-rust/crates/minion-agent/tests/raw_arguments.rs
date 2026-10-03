use minion_agent::{
    Runtime,
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, RawNumber, RawString,
        RawValue, StopReason, ToolCall, Usage,
    },
    session::Session,
    tools::{
        AgentToolResult, PreparedValue, ToolDefinition, ToolExecutionOptions, ToolExecutionRequest,
        execute_tool_calls,
    },
};
use parking_lot::Mutex;
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

fn message(raw: RawValue) -> Message {
    Message::Assistant(Box::new(AssistantMessage::new(
        ModelIdentity::new("mock", "model", "mock").unwrap(),
        vec![AssistantContentBlock::ToolCall(ToolCall::new_raw(
            "c", "probe", raw,
        ))],
        Usage::default(),
        StopReason::ToolUse,
        0.0,
    )))
}
fn arguments(message: &Message) -> &RawValue {
    let Message::Assistant(message) = message else {
        panic!()
    };
    let AssistantContentBlock::ToolCall(call) = &message.content[0] else {
        panic!()
    };
    &call.arguments
}
#[test]
fn decoder_is_binary64_utf16_and_never_repairs_with_replacement() {
    let raw=RawValue::decode(r#"{"high":"\ud800","low":"\udc00","pair":"\ud83d\ude00","\ud800":-0,"n":1e999,"big":1000000000000000100}"#).unwrap();
    assert_eq!(
        raw.get("high").unwrap().as_string().unwrap().code_units(),
        [0xd800]
    );
    assert_eq!(
        raw.get("low").unwrap().as_string().unwrap().code_units(),
        [0xdc00]
    );
    assert_eq!(
        raw.get("pair").unwrap().as_string().unwrap().code_units(),
        [0xd83d, 0xde00]
    );
    assert_eq!(raw.get("n").unwrap().as_f64(), Some(f64::INFINITY));
    assert_eq!(
        raw.get("big").unwrap().as_f64(),
        Some(1000000000000000128.0)
    );
    let RawValue::Object(entries) = &raw else {
        panic!()
    };
    let zero = entries
        .iter()
        .find(|(k, _)| k.code_units() == [0xd800])
        .unwrap()
        .1
        .as_f64()
        .unwrap();
    assert!(zero.is_sign_negative());
    assert!(RawValue::decode("NaN").is_err());
    assert!(RawNumber::new(f64::NAN).is_err());
    assert!(RawValue::decode("01").is_err());
    assert!(RawValue::decode("+1").is_err());
    assert!(RawValue::decode("1.").is_err());
    assert!(RawValue::decode("1e").is_err());
}

#[test]
fn literal_utf16_argument_text_is_not_forced_through_utf8_replacement() {
    let text = RawString::from_code_units(vec![34, 0xd800, 0xd83d, 0xde00, 0xdc00, 34]);
    assert_eq!(
        RawValue::decode_utf16(&text)
            .unwrap()
            .as_string()
            .unwrap()
            .code_units(),
        [0xd800, 0xd83d, 0xde00, 0xdc00]
    );
    for units in [vec![0xd800], vec![34, 92, 0xd800, 34]] {
        assert!(RawValue::decode_utf16(&RawString::from_code_units(units)).is_err());
    }
    let mut keys = indexmap::IndexMap::new();
    keys.insert(
        RawString::from_code_units(vec![0xd800]),
        RawValue::Bool(true),
    );
    keys.insert(
        RawString::from_code_units(vec![0xd800]),
        RawValue::Bool(false),
    );
    assert_eq!(
        keys.len(),
        1,
        "raw objects have unique keys by construction"
    );
    assert_eq!(
        RawValue::decode(r#"{"a":1,"a":2}"#)
            .unwrap()
            .get("a")
            .unwrap()
            .as_f64(),
        Some(2.0)
    );
}
#[test]
fn log_is_live_typed_and_fork_compaction_replay_keep_values() {
    let raw = RawValue::decode(r#"{"s":"\ud800","n":-1e999,"z":-0}"#).unwrap();
    let session = Session::new("parent", std::iter::empty::<String>()).unwrap();
    let event = session.append_message(message(raw.clone())).unwrap();
    assert_eq!(arguments(event.data["message"].as_message().unwrap()), &raw);
    assert!(
        serde_json::to_value(&event).is_err(),
        "a JSON projection cannot silently erase live values"
    );
    let child = session.fork("child", Some(event.seq)).unwrap();
    assert_eq!(arguments(&child.derive_messages().unwrap()[0]), &raw);
    session.compact("summary", 1).unwrap();
    assert_eq!(arguments(&session.derive_messages().unwrap()[1]), &raw);
    assert_eq!(arguments(&child.derive_messages().unwrap()[0]), &raw);
    session.reset().unwrap();
    assert!(session.derive_messages().unwrap().is_empty());
    assert_eq!(arguments(&child.derive_messages().unwrap()[0]), &raw);
}
#[test]
fn projections_are_explicit_and_json_compatible_values_remain_compatible() {
    let raw = RawValue::from(json!({"a":[1,true,null,"ok"]}));
    assert_eq!(raw.try_to_json().unwrap(), json!({"a":[1,true,null,"ok"]}));
    let call = ToolCall::new("c", "probe", BTreeMap::from([("n".into(), json!(1))]));
    let encoded = serde_json::to_value(call).unwrap();
    assert_eq!(encoded["arguments"], json!({"n":1}));
    for text in [r#"{"s":"\ud800"}"#, r#"{"\udc00":0}"#, r#"{"n":1e999}"#] {
        let raw = RawValue::decode(text).unwrap();
        assert!(raw.try_to_json().is_err());
        assert!(serde_json::to_value(raw).is_err());
    }
    assert_ne!(
        RawValue::Number(RawNumber::new(-0.0).unwrap()),
        RawValue::Number(RawNumber::new(0.0).unwrap())
    );
    assert_ne!(
        RawValue::String(RawString::from_code_units(vec![0xd800])),
        RawValue::from(json!("\u{fffd}"))
    );
}
#[tokio::test]
async fn raw_preparation_callback_observes_original_domain() {
    let raw = RawValue::decode(r#"{"s":"\ud800","\udc00":-0,"n":1e999}"#).unwrap();
    let seen = Arc::new(Mutex::new(None));
    let preparation = seen.clone();
    let execute = Arc::new(Mutex::new(None));
    let executed = execute.clone();
    let tool = ToolDefinition::new(
        "probe",
        "probe",
        serde_json::from_value(json!({"type":"object","properties":{}})).unwrap(),
        "probe",
        move |request: ToolExecutionRequest| {
            *executed.lock() = Some(request.params);
            Box::pin(async {
                Ok(AgentToolResult {
                    content: vec![],
                    details: json!({}).into(),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    )
    .with_prepare_raw_arguments(move |raw| {
        *preparation.lock() = Some(raw.clone());
        Ok(PreparedValue::from(raw))
    });
    let runtime = Runtime::new();
    runtime.tools().register_for_scope(None, tool).unwrap();
    let batch = execute_tool_calls(
        &runtime.context(),
        &[ToolCall::new_raw("c", "probe", raw.clone())],
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
    )
    .await
    .unwrap();
    assert!(!batch.messages[0].is_error);
    assert_eq!(seen.lock().as_ref(), Some(&raw));
    let prepared = PreparedValue::from(raw);
    assert_eq!(execute.lock().as_ref(), Some(&prepared));
}
