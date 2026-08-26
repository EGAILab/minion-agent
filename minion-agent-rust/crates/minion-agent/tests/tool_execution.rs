use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use minion_agent::{
    DynPluginSpec, PluginInitError, PluginSpec, Runtime,
    llm::{StopReason, TextBlock, ToolCall, ToolResultContentBlock, Usage},
    tools::{
        AfterToolCallOverride, AgentToolResult, BeforeToolCallAction, ToolCapabilityError,
        ToolDefinition, ToolExecutionEnd, ToolExecutionOptions, ToolExecutionRequest,
        ToolExecutionSignal, ToolExecutionUpdate, after_tool_call_spec, execute_tool_calls,
        register_after_tool_call_hook, register_before_tool_call_hook, tool_execution_end_spec,
        tool_execution_start_spec, tool_execution_update_spec,
    },
};
use serde_json::{Value, json};

type ProtectedObservation = (String, String, Option<Vec<String>>);

struct TestSignal;

impl ToolExecutionSignal for TestSignal {
    fn is_cancelled(&self) -> bool {
        false
    }
}

fn run(future: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(future);
}

fn before_hook_plugin(executed: Arc<parking_lot::Mutex<Option<Value>>>) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "before-hooks",
        vec![],
        || json!({}),
        move |context, _config| {
            let executed = Arc::clone(&executed);
            async move {
                context
                    .tools()
                    .map_err(|error| PluginInitError::new(error.to_string()))?
                    .register(
                        &context,
                        ToolDefinition::new(
                            "prepared",
                            "prepared",
                            serde_json::from_value(json!({
                                "type": "object",
                                "properties": {"x": {"type": "string"}},
                                "required": ["x"]
                            }))
                            .unwrap(),
                            "prepared",
                            move |request: ToolExecutionRequest| {
                                *executed.lock() = Some(request.params);
                                Box::pin(async { Ok(result("done")) })
                            },
                        ),
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                register_before_tool_call_hook(&context, |current| async move {
                    let mut arguments = current.arguments;
                    arguments["stage"] = json!(1);
                    Ok(BeforeToolCallAction::Proceed(Some(arguments)))
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                register_before_tool_call_hook(&context, |current| async move {
                    assert_eq!(current.arguments["stage"], 1);
                    let mut arguments = current.arguments;
                    arguments["stage"] = json!(2);
                    Ok(BeforeToolCallAction::Proceed(Some(arguments)))
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn event_observer_plugin(events: Arc<parking_lot::Mutex<Vec<String>>>) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "execution-events",
        vec![],
        || json!({}),
        move |context, _config| {
            let events = Arc::clone(&events);
            async move {
                let bus = context
                    .events()
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let starts = tool_execution_start_spec();
                let ends = tool_execution_end_spec();
                bus.declare(&starts)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                bus.declare(&ends)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let effects = context.effect_store();
                bus.on_emit(&starts, &effects, context.scope(), {
                    let events = Arc::clone(&events);
                    move |event| events.lock().push(format!("start:{}", event.tool_call_id))
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                bus.on_emit(
                    &ends,
                    &effects,
                    context.scope(),
                    move |event: &ToolExecutionEnd| {
                        events.lock().push(format!("end:{}", event.tool_call_id));
                    },
                )
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn plugin_with_after_hooks(
    seen: Arc<parking_lot::Mutex<Vec<ProtectedObservation>>>,
) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "execution-hooks",
        vec![],
        || json!({}),
        move |context, _config| {
            let seen = Arc::clone(&seen);
            async move {
                context
                    .tools()
                    .map_err(|error| PluginInitError::new(error.to_string()))?
                    .register(
                        &context,
                        ToolDefinition::new(
                            "hooked",
                            "hooked",
                            serde_json::from_value(json!({"type": "object"})).unwrap(),
                            "hooked",
                            |_request: ToolExecutionRequest| {
                                Box::pin(async {
                                    let mut value = result("original");
                                    value.added_tool_names = Some(vec!["alpha".into()]);
                                    Ok(value)
                                })
                            },
                        ),
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                register_after_tool_call_hook(&context, |_current| async {
                    Ok(Some(AfterToolCallOverride::default().with_content(vec![
                        ToolResultContentBlock::Text(TextBlock::new("changed")),
                    ])))
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                register_after_tool_call_hook(&context, move |current| {
                    let seen = Arc::clone(&seen);
                    async move {
                        seen.lock().push((
                            current.tool_call_id.clone(),
                            current.tool_name.clone(),
                            current.added_tool_names.clone(),
                        ));
                        Ok(Some(AfterToolCallOverride::default().with_terminate(true)))
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn plugin_with_raw_after_attack(
    seen: Arc<parking_lot::Mutex<Vec<ProtectedObservation>>>,
) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "raw-after-attack",
        vec![],
        || json!({}),
        move |context, _config| {
            let seen = Arc::clone(&seen);
            async move {
                context
                    .tools()
                    .map_err(|error| PluginInitError::new(error.to_string()))?
                    .register(
                        &context,
                        ToolDefinition::new(
                            "raw-hooked",
                            "raw-hooked",
                            serde_json::from_value(json!({})).unwrap(),
                            "raw-hooked",
                            |_request| {
                                Box::pin(async {
                                    let mut output = result("original");
                                    output.added_tool_names = Some(vec!["alpha".into()]);
                                    Ok(output)
                                })
                            },
                        ),
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let spec = after_tool_call_spec();
                let events = context
                    .events()
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                events
                    .declare(&spec)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let effects = context.effect_store();
                events
                    .on_waterfall(
                        &spec,
                        &effects,
                        context.scope(),
                        |mut current, next| async move {
                            current.tool_call_id = "evil-id".into();
                            current.tool_name = "evil-name".into();
                            current.added_tool_names = Some(vec!["evil".into()]);
                            current.content =
                                vec![ToolResultContentBlock::Text(TextBlock::new("allowed"))];
                            next.call(Some(current)).await
                        },
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                register_after_tool_call_hook(&context, move |current| {
                    let seen = Arc::clone(&seen);
                    async move {
                        seen.lock().push((
                            current.tool_call_id.clone(),
                            current.tool_name.clone(),
                            current.added_tool_names.clone(),
                        ));
                        Ok(None)
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall::new(
        id,
        name,
        serde_json::from_value::<BTreeMap<String, Value>>(arguments).unwrap(),
    )
}

fn result(text: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(text))],
        details: Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        name,
        serde_json::from_value(json!({"type": "object"})).unwrap(),
        name,
        |_request: ToolExecutionRequest| Box::pin(async { Ok(result("ok")) }),
    )
}

fn text(message: &minion_agent::llm::ToolResultMessage) -> &str {
    match &message.content[0] {
        ToolResultContentBlock::Text(block) => &block.text,
        ToolResultContentBlock::Image(_) => panic!("expected text result"),
    }
}

#[test]
fn batch_execution_uses_the_authoritative_registry_and_preserves_source_identity() {
    run(async {
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(None, tool("echo"))
            .unwrap();
        let context = runtime.context();
        let calls = vec![call("t1", "echo", json!({}))];

        let batch = execute_tool_calls(
            &context,
            &calls,
            ToolExecutionOptions::new(StopReason::ToolUse, 42.0),
        )
        .await
        .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert_eq!(batch.messages[0].tool_call_id, "t1");
        assert_eq!(batch.messages[0].tool_name, "echo");
        assert!(!batch.messages[0].is_error);
        assert!(!batch.terminate);
    });
}

#[test]
fn unknown_tool_is_an_isolated_semantic_error() {
    run(async {
        let runtime = Runtime::new();
        let context = runtime.context();
        let calls = vec![call("missing-1", "missing", json!({}))];

        let batch = execute_tool_calls(
            &context,
            &calls,
            ToolExecutionOptions::new(StopReason::ToolUse, 7.0),
        )
        .await
        .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert_eq!(batch.messages[0].tool_call_id, "missing-1");
        assert_eq!(batch.messages[0].tool_name, "missing");
        assert!(batch.messages[0].is_error);
        assert_eq!(text(&batch.messages[0]), "Tool missing not found");
    });
}

#[test]
fn prepared_arguments_are_validated_against_the_raw_json_schema() {
    run(async {
        let runtime = Runtime::new();
        let executions = Arc::new(AtomicUsize::new(0));
        let definition = ToolDefinition::new(
            "strict",
            "strict",
            serde_json::from_value(json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .unwrap(),
            "strict",
            {
                let executions = Arc::clone(&executions);
                move |_request: ToolExecutionRequest| {
                    executions.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async { Ok(result("executed")) })
                }
            },
        );
        runtime
            .tools()
            .register_for_scope(None, definition)
            .unwrap();
        let context = runtime.context();

        let batch = execute_tool_calls(
            &context,
            &[call("bad", "strict", json!({"x": 123}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert!(batch.messages[0].is_error);
        assert!(text(&batch.messages[0]).contains("invalid arguments"));
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn prepare_runs_before_validation_and_can_repair_raw_arguments() {
    run(async {
        let runtime = Runtime::new();
        let definition = ToolDefinition::new(
            "repair",
            "repair",
            serde_json::from_value(json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .unwrap(),
            "repair",
            |request: ToolExecutionRequest| {
                Box::pin(async move {
                    assert_eq!(request.params, json!({"x": "repaired"}));
                    Ok(result("executed"))
                })
            },
        )
        .with_prepare_arguments(|_raw| Ok(json!({"x": "repaired"})));
        runtime
            .tools()
            .register_for_scope(None, definition)
            .unwrap();
        let context = runtime.context();
        let source = call("t1", "repair", json!({"x": 123}));
        let source_copy = source.clone();

        let batch = execute_tool_calls(
            &context,
            &[source],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert!(!batch.messages[0].is_error);
        assert_eq!(
            source_copy.arguments,
            BTreeMap::from([("x".into(), json!(123))])
        );
    });
}

#[test]
fn prepare_can_invalidate_arguments_and_capability_errors_keep_only_the_message() {
    run(async {
        let runtime = Runtime::new();
        let executions = Arc::new(AtomicUsize::new(0));
        let definition = ToolDefinition::new(
            "invalidate",
            "invalidate",
            serde_json::from_value(json!({"type": "string"})).unwrap(),
            "invalidate",
            {
                let executions = Arc::clone(&executions);
                move |_request: ToolExecutionRequest| {
                    executions.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async { Ok(result("should not run")) })
                }
            },
        )
        .with_prepare_arguments(|_raw| Ok(json!(42)));
        runtime
            .tools()
            .register_for_scope(None, definition)
            .unwrap();
        let context = runtime.context();
        let batch = execute_tool_calls(
            &context,
            &[call("t1", "invalidate", json!({}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();
        assert!(batch.messages[0].is_error);
        assert_eq!(executions.load(Ordering::SeqCst), 0);

        let failure = ToolDefinition::new(
            "failure",
            "failure",
            serde_json::from_value(json!({})).unwrap(),
            "failure",
            |_request: ToolExecutionRequest| {
                Box::pin(async { Err(ToolCapabilityError::new("boom")) })
            },
        );
        runtime.tools().register_for_scope(None, failure).unwrap();
        let batch = execute_tool_calls(
            &context,
            &[call("t2", "failure", json!({}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();
        assert_eq!(text(&batch.messages[0]), "boom");
    });
}

#[test]
fn after_hooks_accumulate_allowed_fields_and_preserve_authoritative_fields_per_listener() {
    run(async {
        let runtime = Runtime::new();
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let plugin = plugin_with_after_hooks(Arc::clone(&seen));
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("t1", "hooked", json!({}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert_eq!(text(&batch.messages[0]), "changed");
        assert_eq!(batch.messages[0].tool_call_id, "t1");
        assert_eq!(batch.messages[0].tool_name, "hooked");
        assert_eq!(
            batch.messages[0].added_tool_names.as_deref(),
            Some(["alpha".into()].as_slice())
        );
        assert!(batch.terminate);
        assert_eq!(
            seen.lock().as_slice(),
            &[("t1".into(), "hooked".into(), Some(vec!["alpha".into()]))]
        );
    });
}

#[test]
fn raw_after_listener_is_normalized_before_the_next_listener_and_final_result() {
    run(async {
        let runtime = Runtime::new();
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        runtime
            .mount(&plugin_with_raw_after_attack(Arc::clone(&seen)), json!({}))
            .unwrap();
        runtime.reconcile().await.unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("t1", "raw-hooked", json!({}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert_eq!(text(&batch.messages[0]), "allowed");
        assert_eq!(batch.messages[0].tool_call_id, "t1");
        assert_eq!(batch.messages[0].tool_name, "raw-hooked");
        assert_eq!(
            batch.messages[0].added_tool_names.as_deref(),
            Some(["alpha".to_owned()].as_slice())
        );
        assert_eq!(
            seen.lock().as_slice(),
            &[("t1".into(), "raw-hooked".into(), Some(vec!["alpha".into()]))]
        );
    });
}

#[test]
fn malformed_schema_is_an_isolated_validation_error_instead_of_a_panic() {
    run(async {
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "malformed",
                    "malformed",
                    serde_json::from_value(json!({"type": 42})).unwrap(),
                    "malformed",
                    |_request| Box::pin(async { Ok(result("must not run")) }),
                ),
            )
            .unwrap();
        runtime
            .tools()
            .register_for_scope(None, tool("healthy"))
            .unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[
                call("bad", "malformed", json!({})),
                call("good", "healthy", json!({})),
            ],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert!(batch.messages[0].is_error);
        assert!(text(&batch.messages[0]).contains("invalid arguments"));
        assert!(!batch.messages[1].is_error);
    });
}

#[test]
fn raw_after_listener_failure_replaces_one_result_without_aborting_siblings() {
    run(async {
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(None, tool("raw-fails"))
            .unwrap();
        runtime
            .tools()
            .register_for_scope(None, tool("healthy-sibling"))
            .unwrap();
        let plugin = PluginSpec::<Value>::new(
            "raw-failure",
            vec![],
            || json!({}),
            |context, _config| async move {
                let spec = after_tool_call_spec();
                let events = context
                    .events()
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                events
                    .declare(&spec)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                let effects = context.effect_store();
                events
                    .on_waterfall(
                        &spec,
                        &effects,
                        context.scope(),
                        |current, next| async move {
                            let first = next.call(None).await?;
                            if current.tool_name == "raw-fails" {
                                next.call(None).await
                            } else {
                                Ok(first)
                            }
                        },
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            },
        )
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[
                call("bad", "raw-fails", json!({})),
                call("good", "healthy-sibling", json!({})),
            ],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert!(batch.messages[0].is_error);
        assert!(text(&batch.messages[0]).contains("next"));
        assert_eq!(batch.messages[0].tool_call_id, "bad");
        assert!(!batch.messages[1].is_error);
        assert_eq!(batch.messages[1].tool_call_id, "good");
    });
}

#[test]
fn parallel_completion_events_and_final_messages_have_distinct_deterministic_orders() {
    run(async {
        let runtime = Runtime::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));
        runtime
            .mount(&event_observer_plugin(Arc::clone(&events)), json!({}))
            .unwrap();
        runtime.reconcile().await.unwrap();
        let release_slow = Arc::new(tokio::sync::Notify::new());
        let slow_started = Arc::new(tokio::sync::Notify::new());
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "slow",
                    "slow",
                    serde_json::from_value(json!({})).unwrap(),
                    "slow",
                    {
                        let release_slow = Arc::clone(&release_slow);
                        let slow_started = Arc::clone(&slow_started);
                        move |_request| {
                            let release_slow = Arc::clone(&release_slow);
                            let slow_started = Arc::clone(&slow_started);
                            Box::pin(async move {
                                slow_started.notify_one();
                                release_slow.notified().await;
                                Ok(result("slow"))
                            })
                        }
                    },
                ),
            )
            .unwrap();
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "fast",
                    "fast",
                    serde_json::from_value(json!({})).unwrap(),
                    "fast",
                    {
                        let release_slow = Arc::clone(&release_slow);
                        let slow_started = Arc::clone(&slow_started);
                        move |_request| {
                            let release_slow = Arc::clone(&release_slow);
                            let slow_started = Arc::clone(&slow_started);
                            Box::pin(async move {
                                slow_started.notified().await;
                                release_slow.notify_one();
                                Ok(result("fast"))
                            })
                        }
                    },
                ),
            )
            .unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("a", "slow", json!({})), call("b", "fast", json!({}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert_eq!(
            events.lock().as_slice(),
            ["start:a", "start:b", "end:b", "end:a"]
        );
        assert_eq!(
            batch
                .messages
                .iter()
                .map(|message| message.tool_call_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    });
}

#[test]
fn length_stop_emits_results_without_resolving_or_invoking_tools() {
    run(async {
        let runtime = Runtime::new();
        let executions = Arc::new(AtomicUsize::new(0));
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "real",
                    "real",
                    serde_json::from_value(json!({})).unwrap(),
                    "real",
                    {
                        let executions = Arc::clone(&executions);
                        move |_request| {
                            executions.fetch_add(1, Ordering::SeqCst);
                            Box::pin(async { Ok(result("bad")) })
                        }
                    },
                ),
            )
            .unwrap();
        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("t1", "real", json!({}))],
            ToolExecutionOptions::new(StopReason::Length, 0.0),
        )
        .await
        .unwrap();

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(batch.messages[0].is_error);
        assert_eq!(
            text(&batch.messages[0]),
            "Tool call \"real\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments."
        );
        assert!(!batch.terminate);
    });
}

#[test]
fn before_hooks_form_a_registration_order_waterfall_after_validation() {
    run(async {
        let runtime = Runtime::new();
        let executed = Arc::new(parking_lot::Mutex::new(None));
        runtime
            .mount(&before_hook_plugin(Arc::clone(&executed)), json!({}))
            .unwrap();
        runtime.reconcile().await.unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("t1", "prepared", json!({"x": "valid"}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert!(!batch.messages[0].is_error);
        assert_eq!(executed.lock().as_ref().unwrap()["stage"], 2);
    });
}

#[test]
fn updates_are_emitted_live_and_ignored_after_execute_settles() {
    run(async {
        let runtime = Runtime::new();
        let updates = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let saved = Arc::new(parking_lot::Mutex::new(None));
        let plugin = PluginSpec::<Value>::new("update-observer", vec![], || json!({}), {
            let updates = Arc::clone(&updates);
            let saved = Arc::clone(&saved);
            move |context, _config| {
                let updates = Arc::clone(&updates);
                let saved = Arc::clone(&saved);
                async move {
                    let bus = context
                        .events()
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    let spec = tool_execution_update_spec();
                    bus.declare(&spec)
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    let effects = context.effect_store();
                    bus.on_emit(
                        &spec,
                        &effects,
                        context.scope(),
                        move |event: &ToolExecutionUpdate| {
                            updates.lock().push(event.update.content.clone());
                        },
                    )
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    context
                        .tools()
                        .map_err(|error| PluginInitError::new(error.to_string()))?
                        .register(
                            &context,
                            ToolDefinition::new(
                                "chatty",
                                "chatty",
                                serde_json::from_value(json!({})).unwrap(),
                                "chatty",
                                move |request: ToolExecutionRequest| {
                                    let saved = Arc::clone(&saved);
                                    Box::pin(async move {
                                        let update = request.on_update.unwrap();
                                        update(result("live"));
                                        *saved.lock() = Some(update);
                                        Ok(result("done"))
                                    })
                                },
                            ),
                        )
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                }
            }
        })
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        execute_tool_calls(
            &runtime.context(),
            &[call("t1", "chatty", json!({}))],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();
        saved.lock().as_ref().unwrap()(result("late"));

        assert_eq!(updates.lock().len(), 1);
    });
}

#[test]
fn hook_failures_short_circuit_with_pi_stage_reachability() {
    run(async {
        let runtime = Runtime::new();
        let before_executions = Arc::new(AtomicUsize::new(0));
        let plugin = PluginSpec::<Value>::new("failing-hooks", vec![], || json!({}), {
            let before_executions = Arc::clone(&before_executions);
            move |context, _config| {
                let before_executions = Arc::clone(&before_executions);
                async move {
                    let tools = context
                        .tools()
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    tools
                        .register(
                            &context,
                            ToolDefinition::new(
                                "before-fails",
                                "before-fails",
                                serde_json::from_value(json!({})).unwrap(),
                                "before-fails",
                                move |_request| {
                                    before_executions.fetch_add(1, Ordering::SeqCst);
                                    Box::pin(async { Ok(result("bad")) })
                                },
                            ),
                        )
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    tools
                        .register(
                            &context,
                            ToolDefinition::new(
                                "execute-fails",
                                "execute-fails",
                                serde_json::from_value(json!({})).unwrap(),
                                "execute-fails",
                                |_request| {
                                    Box::pin(async {
                                        Err(ToolCapabilityError::new("execute boom"))
                                    })
                                },
                            ),
                        )
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    register_before_tool_call_hook(&context, |current| async move {
                        if current.tool_name == "before-fails" {
                            Err(ToolCapabilityError::new("before boom"))
                        } else {
                            Ok(BeforeToolCallAction::Proceed(None))
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    register_after_tool_call_hook(&context, |current| async move {
                        if current.tool_name == "execute-fails" {
                            Err(ToolCapabilityError::new("after boom"))
                        } else {
                            Ok(None)
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                }
            }
        })
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[
                call("t1", "before-fails", json!({})),
                call("t2", "execute-fails", json!({})),
            ],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert_eq!(before_executions.load(Ordering::SeqCst), 0);
        assert_eq!(text(&batch.messages[0]), "before boom");
        assert_eq!(text(&batch.messages[1]), "after boom");
        assert!(batch.messages.iter().all(|message| message.is_error));
    });
}

#[test]
fn one_sequential_tool_makes_the_entire_batch_sequential() {
    run(async {
        let runtime = Runtime::new();
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        for (name, sequential) in [("shared", false), ("exclusive", true)] {
            let definition = ToolDefinition::new(
                name,
                name,
                serde_json::from_value(json!({})).unwrap(),
                name,
                {
                    let active = Arc::clone(&active);
                    let maximum = Arc::clone(&maximum);
                    move |_request| {
                        let active = Arc::clone(&active);
                        let maximum = Arc::clone(&maximum);
                        Box::pin(async move {
                            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                            maximum.fetch_max(now, Ordering::SeqCst);
                            tokio::task::yield_now().await;
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok(result("done"))
                        })
                    }
                },
            );
            let definition = if sequential {
                definition.with_execution_mode(minion_agent::tools::ExecutionMode::Sequential)
            } else {
                definition
            };
            runtime
                .tools()
                .register_for_scope(None, definition)
                .unwrap();
        }

        execute_tool_calls(
            &runtime.context(),
            &[
                call("t1", "shared", json!({})),
                call("t2", "exclusive", json!({})),
            ],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();

        assert_eq!(maximum.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn signal_and_execution_metadata_pass_through_without_namespace_lookup_semantics() {
    run(async {
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "meta",
                    "meta",
                    serde_json::from_value(json!({})).unwrap(),
                    "meta",
                    |request: ToolExecutionRequest| {
                        Box::pin(async move {
                            assert!(!request.signal.unwrap().is_cancelled());
                            let mut output = result("done");
                            output.details = json!({"trace": 1});
                            output.usage = Some(Usage::default());
                            output.added_tool_names = Some(vec!["alpha".into()]);
                            output.terminate = Some(true);
                            Ok(output)
                        })
                    },
                ),
            )
            .unwrap();
        let call = call("t1", "meta", json!({})).with_namespace("ignored");

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_signal(Arc::new(TestSignal)),
        )
        .await
        .unwrap();

        assert_eq!(batch.messages[0].details, Some(json!({"trace": 1})));
        assert_eq!(batch.messages[0].usage, Some(Usage::default()));
        assert_eq!(
            batch.messages[0].added_tool_names.as_deref(),
            Some(["alpha".to_owned()].as_slice())
        );
        assert!(batch.terminate);
    });
}
