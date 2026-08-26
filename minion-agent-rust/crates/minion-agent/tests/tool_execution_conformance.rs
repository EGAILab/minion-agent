#![cfg(feature = "conformance")]

use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc};

use minion_agent::{
    DynPluginSpec, PluginInitError, PluginSpec, Runtime,
    llm::{StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        AfterToolCallOverride, AgentToolResult, BeforeToolCallAction, ExecutionMode,
        ToolCapabilityError, ToolDefinition, ToolExecutionEnd, ToolExecutionOptions,
        ToolExecutionRequest, ToolExecutionUpdate, execute_tool_calls,
        register_after_tool_call_hook, register_before_tool_call_hook, tool_execution_end_spec,
        tool_execution_update_spec,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};

const SCENARIOS: [&str; 10] = [
    "after-hook-failure-replaces-result-with-tool-error",
    "before-hook-failure-becomes-tool-error",
    "execute-failure-becomes-tool-error",
    "late-tool-update-ignored",
    "length-stop-executes-no-tools",
    "parallel-tool-completion-vs-message-order",
    "prepare-arguments-failure-becomes-tool-error",
    "schema-validation-failure-becomes-tool-error",
    "tool-batch-parallel",
    "tool-batch-sequential-contagion",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn text_result(text: impl Into<String>) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(text))],
        details: Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn content_text(content: &[ToolResultContentBlock]) -> &str {
    match &content[0] {
        ToolResultContentBlock::Text(block) => &block.text,
        ToolResultContentBlock::Image(_) => panic!("Layer-06 fixtures expect text"),
    }
}

fn parse_calls(document: &Value) -> (Vec<ToolCall>, StopReason) {
    let response = &document["provider_script"][0];
    let calls = response["content"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter(|block| block["type"] == "tool_call")
        .map(|block| {
            ToolCall::new(
                block["id"].as_str().unwrap(),
                block["name"].as_str().unwrap(),
                serde_json::from_value::<BTreeMap<String, Value>>(block["arguments"].clone())
                    .unwrap(),
            )
        })
        .collect();
    let stop_reason = match response["stop_reason"].as_str().unwrap() {
        "length" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        other => panic!("unexpected Layer-06 stop reason {other}"),
    };
    (calls, stop_reason)
}

fn scripted_tool(name: &str, script: &Value) -> ToolDefinition {
    let parameters = serde_json::from_value(
        script
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
    )
    .unwrap();
    let result_text = script["result"]["text"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let raises = script
        .get("raises")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let delay_ticks = script
        .get("delay_ticks")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let updates = script
        .get("emits_updates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|value| value.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let late = script
        .get("late_update")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let terminate = script
        .get("terminate")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut definition = ToolDefinition::new(
        name,
        name,
        parameters,
        name,
        move |request: ToolExecutionRequest| {
            let result_text = result_text.clone();
            let raises = raises.clone();
            let updates = updates.clone();
            let late = late.clone();
            Box::pin(async move {
                for _ in 0..delay_ticks {
                    tokio::task::yield_now().await;
                }
                if let Some(update) = &request.on_update {
                    for value in updates {
                        update(text_result(value));
                    }
                    if let Some(value) = late {
                        let update = Arc::clone(update);
                        tokio::spawn(async move {
                            tokio::task::yield_now().await;
                            update(text_result(value));
                        });
                    }
                }
                if let Some(message) = raises {
                    return Err(ToolCapabilityError::new(message));
                }
                let mut result = text_result(result_text);
                result.terminate = Some(terminate);
                Ok(result)
            })
        },
    );
    if script["execution_mode"] == "sequential" {
        definition = definition.with_execution_mode(ExecutionMode::Sequential);
    }
    if let Some(prepare) = script.get("prepare_arguments") {
        let raises = prepare
            .get("raises")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let set = prepare.get("set").cloned();
        definition = definition.with_prepare_arguments(move |mut arguments| {
            if let Some(message) = &raises {
                return Err(ToolCapabilityError::new(message));
            }
            if let (Some(target), Some(values)) = (
                arguments.as_object_mut(),
                set.as_ref().and_then(Value::as_object),
            ) {
                target.extend(values.clone());
            }
            Ok(arguments)
        });
    }
    definition
}

fn observation_plugin(
    listeners: Vec<Value>,
    completions: Arc<Mutex<Vec<String>>>,
    updates: Arc<Mutex<Vec<(String, String)>>>,
) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "layer-06-observer",
        vec![],
        || json!({}),
        move |context, _config| {
            let listeners = listeners.clone();
            let completions = Arc::clone(&completions);
            let updates = Arc::clone(&updates);
            async move {
                let events = context
                    .events()
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let end_spec = tool_execution_end_spec();
                let update_spec = tool_execution_update_spec();
                events
                    .declare(&end_spec)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                events
                    .declare(&update_spec)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let effects = context.effect_store();
                events
                    .on_emit(
                        &end_spec,
                        &effects,
                        context.scope(),
                        move |event: &ToolExecutionEnd| {
                            completions.lock().push(event.tool_call_id.clone());
                        },
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                events
                    .on_emit(
                        &update_spec,
                        &effects,
                        context.scope(),
                        move |event: &ToolExecutionUpdate| {
                            updates.lock().push((
                                event.tool_call_id.clone(),
                                content_text(&event.update.content).to_owned(),
                            ));
                        },
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                for listener in listeners {
                    let event = listener["event"].as_str().unwrap();
                    let action = listener["action"].as_str().unwrap().to_owned();
                    let message = listener
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    let label = listener
                        .get("label")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    if event == "tools/pre-execute" {
                        register_before_tool_call_hook(&context, move |_current| {
                            let action = action.clone();
                            let message = message.clone();
                            async move {
                                match action.as_str() {
                                    "raise" => Err(ToolCapabilityError::new(message)),
                                    "block" => Ok(BeforeToolCallAction::Block(message)),
                                    _ => Ok(BeforeToolCallAction::Proceed(None)),
                                }
                            }
                        })
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    } else {
                        register_after_tool_call_hook(&context, move |current| {
                            let action = action.clone();
                            let message = message.clone();
                            let label = label.clone();
                            async move {
                                match action.as_str() {
                                    "raise" => Err(ToolCapabilityError::new(message)),
                                    "annotate_result" => Ok(Some(
                                        AfterToolCallOverride::default().with_content(vec![
                                            ToolResultContentBlock::Text(TextBlock::new(format!(
                                                "{}-{label}",
                                                content_text(&current.content)
                                            ))),
                                        ]),
                                    )),
                                    _ => Ok(None),
                                }
                            }
                        })
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    }
                }
                Ok(())
            }
        },
    )
    .erase()
}

fn run_scenario(document: &Value) {
    let runtime = Runtime::new();
    for (name, script) in document["tools"].as_object().into_iter().flatten() {
        runtime
            .tools()
            .register_for_scope(None, scripted_tool(name, script))
            .unwrap();
    }
    let completions = Arc::new(Mutex::new(Vec::new()));
    let updates = Arc::new(Mutex::new(Vec::new()));
    let listeners = document["listeners"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let plugin = observation_plugin(listeners, Arc::clone(&completions), Arc::clone(&updates));
    runtime.mount(&plugin, json!({})).unwrap();
    let (calls, stop_reason) = parse_calls(document);
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    tokio.block_on(runtime.reconcile()).unwrap();
    let batch = tokio
        .block_on(execute_tool_calls(
            &runtime.context(),
            &calls,
            ToolExecutionOptions::new(stop_reason, 0.0),
        ))
        .unwrap();
    tokio.block_on(async { tokio::task::yield_now().await });

    let expected_results = document["expect_messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool_result")
        .collect::<Vec<_>>();
    assert_eq!(batch.messages.len(), expected_results.len());
    for (actual, expected) in batch.messages.iter().zip(expected_results) {
        let actual = content_text(&actual.content);
        if let Some(exact) = expected.get("text").and_then(Value::as_str) {
            assert_eq!(actual, exact);
        }
        if let Some(fragment) = expected.get("text_contains").and_then(Value::as_str) {
            assert!(
                actual.contains(fragment),
                "{actual:?} does not contain {fragment:?}"
            );
        }
    }
    if let Some(expected) = document.get("expect_tool_completion_order") {
        assert_eq!(
            serde_json::to_value(completions.lock().clone()).unwrap(),
            *expected
        );
    }
    if let Some(expected) = document.get("expect_updates") {
        assert_eq!(
            serde_json::to_value(updates.lock().clone()).unwrap(),
            *expected
        );
    }
}

#[test]
fn all_layer_06_scenarios_drive_the_real_rust_tool_executor() {
    let directory = root().join("conformance/agent");
    let mut scenarios = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .filter_map(|path| {
            let stem = path.file_stem()?.to_str()?;
            SCENARIOS.contains(&stem).then_some(path)
        })
        .collect::<Vec<_>>();
    scenarios.sort();
    assert_eq!(scenarios.len(), SCENARIOS.len());
    for path in scenarios {
        let document: Value = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        run_scenario(&document);
    }
}
