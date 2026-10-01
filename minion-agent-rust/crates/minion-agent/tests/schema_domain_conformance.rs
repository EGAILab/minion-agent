#![cfg(feature = "conformance")]

use minion_agent::{
    Runtime,
    llm::{RawString, RawValue, StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        AgentToolResult, PreparedValue, RuntimeSchemaObject, ToolDefinition, ToolExecutionOptions,
        execute_tool_calls,
    },
};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

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

// Decode only the shared transport grammar. No role, keyword, verdict or
// normalization logic belongs to this runner.
fn decode(value: &Value) -> RawValue {
    match value {
        Value::Object(o) if o.contains_key("utf16") => {
            RawValue::String(RawString::from_code_units(units(&o["utf16"])))
        }
        Value::Object(o) if o.contains_key("$keys") => RawValue::Object(
            o["$keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    (
                        RawString::from_code_units(units(&pair[0])),
                        decode(&pair[1]),
                    )
                })
                .collect(),
        ),
        Value::Array(a) => RawValue::Array(a.iter().map(decode).collect()),
        _ => RawValue::from(value.clone()),
    }
}

async fn run(case: &Value) -> bool {
    let runtime = Runtime::new();
    let count = Arc::new(AtomicUsize::new(0));
    let executed = count.clone();
    let schema =
        RuntimeSchemaObject::try_from(PreparedValue::from(decode(&case["schema"]))).unwrap();
    let tool =
        ToolDefinition::new_with_runtime_schema("probe", "probe", schema, "probe", move |_| {
            executed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Ok(AgentToolResult {
                    content: vec![ToolResultContentBlock::Text(TextBlock::new("ok"))],
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        });
    assert!(tool.prepare_arguments().is_none());
    runtime.tools().register_for_scope(None, tool).unwrap();
    let call = ToolCall::new_raw("call", "probe", decode(&case["arguments"]));
    let source = call.arguments.clone();
    let batch = execute_tool_calls(
        &runtime.context(),
        std::slice::from_ref(&call),
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
    )
    .await
    .unwrap();
    assert_eq!(call.arguments, source);
    assert_eq!(batch.messages.len(), 1);
    if batch.messages[0].is_error {
        assert_eq!(count.load(Ordering::SeqCst), 0);
        let text = batch.messages[0]
            .content
            .iter()
            .filter_map(|block| match block {
                ToolResultContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(text.starts_with("invalid arguments for tool"), "{text}");
        assert!(
            !text.contains("invalid schema"),
            "schema error is not a canonical rejection: {text}"
        );
        false
    } else {
        assert_eq!(count.load(Ordering::SeqCst), 1);
        true
    }
}

#[tokio::test]
async fn schema_domain_cases_use_real_registration_and_unprepared_execution() {
    let schema: Value = serde_json::from_str(
        &fs::read_to_string(root().join("conformance/schema/schema-domain-scenario.schema.json"))
            .unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mut documents = fs::read_dir(root().join("conformance/agent/schema-domain"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect::<Vec<_>>();
    documents.sort();
    assert_eq!(documents.len(), 10);
    let mut count = 0;
    let mut mismatches = Vec::new();
    for path in documents {
        let document: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        validator.validate(&document).unwrap();
        for case in document["schema_domain"]["cases"].as_array().unwrap() {
            if run(case).await != (case["expect"] == "accept") {
                mismatches.push(case["id"].as_str().unwrap().to_owned());
            }
            count += 1;
        }
    }
    assert_eq!(count, 810);
    assert!(
        mismatches.is_empty(),
        "schema-domain verdict mismatches: {mismatches:?}"
    );
    eprintln!("schema domain delta gate: {count}/810 cases");
}
