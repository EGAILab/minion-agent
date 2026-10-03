#![cfg(feature = "conformance")]

use minion_agent::{
    PluginInitError, PluginSpec, Runtime,
    execution::LocalFileSystem,
    llm::{StopReason, ToolCall},
    tools::{
        BeforeToolCallAction, PreparedValue, ToolExecutionOptions, builtin::create_edit_tool,
        execute_tool_calls, register_before_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::sync::Arc;

#[tokio::test]
async fn gate_wp132_real_edit_preserves_prepared_numeric_domain_at_the_hook() {
    let document: Value = serde_yaml::from_str(&std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/prepared-runtime/prepared-runtime-edit-json-string-numbers.yaml")).unwrap()).unwrap();
    assert_eq!(document["gate"], "WP-13.2");
    let cases = document["prepared_runtime"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), b"a").unwrap();
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(
                None,
                create_edit_tool(Arc::new(LocalFileSystem::new(dir.path()))),
            )
            .unwrap();
        let observation = Arc::new(Mutex::new(None));
        let capture = observation.clone();
        let plugin = PluginSpec::<Value>::new(
            "observer",
            vec![],
            || json!({}),
            move |context, _| {
                let capture = capture.clone();
                async move {
                    register_before_tool_call_hook(&context, move |current| {
                        *capture.lock() = Some(current.arguments.clone());
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
        let raw = case["arguments"].clone();
        let call = ToolCall::new("edit", "edit", serde_json::from_value(raw.clone()).unwrap());
        let starts = Arc::new(Mutex::new(Vec::new()));
        let capture = starts.clone();
        let options = ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_execution_start(
            move |event| {
                capture.lock().push(event.arguments);
                async { Ok(()) }
            },
        );
        let batch = execute_tool_calls(&runtime.context(), &[call], options)
            .await
            .unwrap();
        assert!(!batch.messages[0].is_error, "{}", case["id"]);
        let result = serde_json::to_value(&batch.messages[0]).unwrap();
        assert_eq!(result["content"][0]["text"], case["expect"]["result_text"]);
        assert_eq!(std::fs::read(dir.path().join("f.txt")).unwrap(), b"b");
        assert_eq!(*starts.lock(), vec![raw]);
        let value = observation.lock().clone().unwrap();
        for pointer in case["observe"].as_array().unwrap() {
            let pointer = pointer.as_str().unwrap();
            let mut at = value.clone();
            for key in pointer.strip_prefix('/').unwrap().split('/') {
                at = if let PreparedValue::Array(a) = &at {
                    a.get(key.parse::<usize>().unwrap()).unwrap()
                } else {
                    at.get(key).unwrap()
                };
            }
            let n = at.as_f64().unwrap();
            let token = if n == f64::INFINITY {
                "+Infinity".into()
            } else if n == f64::NEG_INFINITY {
                "-Infinity".into()
            } else if n == 0.0 && n.is_sign_negative() {
                "-0".into()
            } else {
                ryu_js::Buffer::new().format(n).to_owned()
            };
            assert_eq!(
                Value::String(token),
                case["expect"]["observed"][pointer],
                "{}",
                case["id"]
            );
        }
    }
}
