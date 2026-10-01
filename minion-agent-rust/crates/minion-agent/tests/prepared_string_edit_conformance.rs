#![cfg(feature = "conformance")]

use minion_agent::{
    PluginInitError, PluginSpec, Runtime,
    execution::LocalFileSystem,
    llm::{StopReason, ToolCall, ToolResultContentBlock},
    tools::{
        BeforeToolCallAction, PreparedValue, ToolExecutionOptions, builtin::create_edit_tool,
        execute_tool_calls, register_before_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

#[tokio::test]
async fn gate_wp132_real_edit_preserves_all_prepared_strings_and_file_bytes() {
    let doc: Value = serde_yaml::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/prepared-runtime-string/prepared-string-edit-json-string.yaml")).unwrap()).unwrap();
    assert_eq!(doc["gate"], "WP-13.2");
    let cases = doc["prepared_string"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 21);
    for case in cases {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("f.txt"), b"a\n").unwrap();
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(
                None,
                create_edit_tool(Arc::new(LocalFileSystem::new(root.path()))),
            )
            .unwrap();
        let seen = Arc::new(Mutex::new(None));
        let capture = seen.clone();
        let plugin = PluginSpec::<Value>::new(
            "observe",
            vec![],
            || json!({}),
            move |context, _| {
                let capture = capture.clone();
                async move {
                    register_before_tool_call_hook(&context, move |value| {
                        *capture.lock() = Some(value.arguments.clone());
                        async { Ok(BeforeToolCallAction::Proceed(None)) }
                    })
                    .map_err(|e| PluginInitError::new(e.to_string()))?;
                    Ok(())
                }
            },
        )
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();
        let call = ToolCall::new(
            "call",
            "edit",
            serde_json::from_value(case["arguments"].clone()).unwrap(),
        );
        let batch = execute_tool_calls(
            &runtime.context(),
            &[call],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();
        let message = &batch.messages[0];
        assert!(!message.is_error, "{}: {:?}", case["id"], message.content);
        let ToolResultContentBlock::Text(text) = &message.content[0] else {
            panic!("text result")
        };
        assert_eq!(
            text.text.as_str().unwrap(),
            case["expect"]["result_text"].as_str().unwrap(),
            "{}",
            case["id"]
        );
        let observed = seen.lock().clone().unwrap();
        for pointer in case["observe"].as_array().unwrap() {
            let pointer = pointer.as_str().unwrap();
            let mut value = &observed;
            for key in pointer.strip_prefix('/').unwrap().split('/') {
                value = if let PreparedValue::Array(values) = value {
                    &values[key.parse::<usize>().unwrap()]
                } else {
                    value.get(key).unwrap()
                };
            }
            let PreparedValue::String(value) = value else {
                panic!("prepared string")
            };
            assert_eq!(
                json!(value.code_units()),
                case["expect"]["observed"][pointer],
                "{}: {pointer}",
                case["id"]
            );
        }
        let bytes = std::fs::read(root.path().join("f.txt")).unwrap();
        let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex,
            case["expect"]["file_utf8_hex"].as_str().unwrap(),
            "{}",
            case["id"]
        );
    }
}
