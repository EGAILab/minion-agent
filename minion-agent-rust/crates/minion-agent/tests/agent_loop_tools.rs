use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use minion_agent::{
    Runtime,
    llm::{StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        AgentToolResult, ToolDefinition, ToolExecutionOptions, ToolExecutionRequest,
        ToolLifecycleError, execute_tool_calls,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::Notify;

fn run(future: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(future);
}

fn call(id: &str, name: &str) -> ToolCall {
    ToolCall::new(id, name, BTreeMap::new())
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

fn text(result: &minion_agent::llm::ToolResultMessage) -> &str {
    match result.content.first().unwrap() {
        ToolResultContentBlock::Text(block) => &block.text,
        ToolResultContentBlock::Image(_) => panic!("expected text result"),
    }
}

#[test]
fn live_start_failure_prevents_the_execute_body() {
    run(async {
        let runtime = Runtime::new();
        let executions = Arc::new(AtomicUsize::new(0));
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "guarded",
                    "guarded",
                    serde_json::from_value(json!({})).unwrap(),
                    "guarded",
                    {
                        let executions = Arc::clone(&executions);
                        move |_request: ToolExecutionRequest| {
                            executions.fetch_add(1, Ordering::SeqCst);
                            Box::pin(async { Ok(result("should-not-run")) })
                        }
                    },
                ),
            )
            .unwrap();

        let error = execute_tool_calls(
            &runtime.context(),
            &[call("call-1", "guarded")],
            ToolExecutionOptions::new(StopReason::ToolUse, 1.0).with_execution_start(
                |_event| async { Err(ToolLifecycleError::new("start listener failed")) },
            ),
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "start listener failed");
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn structured_update_starts_eagerly_and_joins_before_end() {
    run(async {
        let runtime = Runtime::new();
        let trace = Arc::new(Mutex::new(Vec::new()));
        let partial = AgentToolResult {
            content: vec![ToolResultContentBlock::Text(TextBlock::new("partial"))],
            details: json!({"progress": 1}),
            usage: None,
            added_tool_names: Some(vec!["introduced".into()]),
            terminate: Some(false),
        };
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "chatty",
                    "chatty",
                    serde_json::from_value(json!({})).unwrap(),
                    "chatty",
                    {
                        let trace = Arc::clone(&trace);
                        let partial = partial.clone();
                        move |request: ToolExecutionRequest| {
                            let trace = Arc::clone(&trace);
                            let partial = partial.clone();
                            Box::pin(async move {
                                request.on_update.unwrap()(partial);
                                trace.lock().push("tool-continued");
                                Ok(result("final"))
                            })
                        }
                    },
                ),
            )
            .unwrap();

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("call-1", "chatty")],
            ToolExecutionOptions::new(StopReason::ToolUse, 1.0)
                .with_execution_update({
                    let trace = Arc::clone(&trace);
                    let expected = partial.clone();
                    move |event| {
                        let trace = Arc::clone(&trace);
                        let expected = expected.clone();
                        async move {
                            assert_eq!(event.update, expected);
                            trace.lock().push("update-entered");
                            tokio::task::yield_now().await;
                            trace.lock().push("update-finished");
                            Ok(())
                        }
                    }
                })
                .with_execution_end({
                    let trace = Arc::clone(&trace);
                    move |event| {
                        let trace = Arc::clone(&trace);
                        async move {
                            assert_eq!(event.tool_name, "chatty");
                            assert_eq!(event.result.content, result("final").content);
                            trace.lock().push("end");
                            Ok(())
                        }
                    }
                }),
        )
        .await
        .unwrap();

        assert_eq!(text(&batch.messages[0]), "final");
        assert_eq!(
            trace.lock().as_slice(),
            ["update-entered", "tool-continued", "update-finished", "end"]
        );
    });
}

#[test]
fn update_listener_failure_propagates_before_finalization_and_end() {
    run(async {
        let runtime = Runtime::new();
        runtime
            .tools()
            .register_for_scope(
                None,
                ToolDefinition::new(
                    "chatty",
                    "chatty",
                    serde_json::from_value(json!({})).unwrap(),
                    "chatty",
                    |request: ToolExecutionRequest| {
                        Box::pin(async move {
                            request.on_update.unwrap()(result("partial"));
                            Ok(result("final"))
                        })
                    },
                ),
            )
            .unwrap();
        let ends = Arc::new(AtomicUsize::new(0));

        let error = execute_tool_calls(
            &runtime.context(),
            &[call("call-1", "chatty")],
            ToolExecutionOptions::new(StopReason::ToolUse, 1.0)
                .with_execution_update(|_event| async {
                    tokio::task::yield_now().await;
                    Err(ToolLifecycleError::new("update listener failed"))
                })
                .with_execution_end({
                    let ends = Arc::clone(&ends);
                    move |_event| {
                        let ends = Arc::clone(&ends);
                        async move {
                            ends.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        }
                    }
                }),
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "update listener failed");
        assert_eq!(ends.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn parallel_end_callbacks_follow_completion_while_messages_remain_source_ordered() {
    run(async {
        let runtime = Runtime::new();
        let release_a = Arc::new(Notify::new());
        for name in ["a", "b"] {
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        name,
                        name,
                        serde_json::from_value(json!({})).unwrap(),
                        name,
                        {
                            let release_a = Arc::clone(&release_a);
                            move |_request: ToolExecutionRequest| {
                                let release_a = Arc::clone(&release_a);
                                Box::pin(async move {
                                    if name == "a" {
                                        release_a.notified().await;
                                    } else {
                                        release_a.notify_one();
                                    }
                                    Ok(result(name))
                                })
                            }
                        },
                    ),
                )
                .unwrap();
        }
        let ends = Arc::new(Mutex::new(Vec::new()));

        let batch = execute_tool_calls(
            &runtime.context(),
            &[call("call-a", "a"), call("call-b", "b")],
            ToolExecutionOptions::new(StopReason::ToolUse, 1.0).with_execution_end({
                let ends = Arc::clone(&ends);
                move |event| {
                    let ends = Arc::clone(&ends);
                    async move {
                        ends.lock().push(event.tool_name);
                        Ok(())
                    }
                }
            }),
        )
        .await
        .unwrap();

        assert_eq!(ends.lock().as_slice(), ["b", "a"]);
        assert_eq!(
            batch.messages.iter().map(text).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    });
}
