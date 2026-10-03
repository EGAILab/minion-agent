use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use minion_agent::{
    Runtime,
    llm::StopReason,
    tools::{
        AgentToolResult, ExecutionMode, ToolCapabilityError, ToolDefinition, ToolExecutionContext,
        ToolExecutionOptions, ToolExecutionRequest, ToolExecutionSignal, execute_tool_calls,
    },
};
use parking_lot::Mutex;
use serde_json::json;

fn output() -> AgentToolResult {
    AgentToolResult {
        content: vec![],
        details: json!({}).into(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn calls() -> Vec<minion_agent::llm::ToolCall> {
    ["a", "b"]
        .into_iter()
        .map(|id| minion_agent::llm::ToolCall::new(id, "probe", BTreeMap::new()))
        .collect()
}

fn probe(runtime: &Runtime, observed: Arc<Mutex<Vec<Option<ToolExecutionContext>>>>) {
    runtime
        .tools()
        .register_for_scope(
            None,
            ToolDefinition::new(
                "probe",
                "probe",
                serde_json::from_value(json!({"type":"object"})).unwrap(),
                "probe",
                move |request: ToolExecutionRequest| {
                    observed.lock().push(request.context);
                    Box::pin(async { Ok(output()) })
                },
            )
            .with_execution_mode(ExecutionMode::Sequential),
        )
        .unwrap();
}

#[tokio::test]
async fn context_absent_without_provider_and_ignoring_context_is_valid() {
    let runtime = Runtime::new();
    let observed = Arc::new(Mutex::new(vec![]));
    probe(&runtime, observed.clone());
    let batch = execute_tool_calls(
        &runtime.context(),
        &calls(),
        ToolExecutionOptions::new(StopReason::ToolUse, 1.0),
    )
    .await
    .unwrap();
    assert_eq!(*observed.lock(), vec![None, None]);
    assert_eq!(batch.messages.len(), 2);
    assert!(batch.messages.iter().all(|message| !message.is_error));
}

#[tokio::test]
async fn provider_failure_is_a_tool_error_with_end_and_healthy_sibling() {
    let runtime = Runtime::new();
    let finalized = Arc::new(Mutex::new(vec![]));
    let plugin = minion_agent::PluginSpec::<serde_json::Value>::new(
        "context-finalize",
        vec![],
        || json!({}),
        {
            let finalized = finalized.clone();
            move |context, _| {
                let finalized = finalized.clone();
                async move {
                    minion_agent::tools::register_after_tool_call_hook(&context, move |result| {
                        finalized
                            .lock()
                            .push((result.tool_call_id, result.is_error));
                        async { Ok(None) }
                    })
                    .map_err(|e| minion_agent::PluginInitError::new(e.to_string()))?;
                    Ok(())
                }
            }
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let observed = Arc::new(Mutex::new(vec![]));
    probe(&runtime, observed.clone());
    let sampled = Arc::new(AtomicUsize::new(0));
    let count = sampled.clone();
    let ends = Arc::new(Mutex::new(vec![]));
    let delivered = ends.clone();
    let options = ToolExecutionOptions::new(StopReason::ToolUse, 1.0)
        .with_context_provider(move || {
            if count.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(ToolCapabilityError::new("context boom"))
            } else {
                Ok(None)
            }
        })
        .with_execution_end(move |end| {
            delivered.lock().push(end);
            Box::pin(async { Ok(()) })
        });
    let batch = execute_tool_calls(&runtime.context(), &calls(), options)
        .await
        .unwrap();
    assert_eq!(sampled.load(Ordering::SeqCst), 2);
    assert_eq!(*observed.lock(), vec![None]);
    assert!(batch.messages[0].is_error);
    assert!(!batch.messages[1].is_error);
    assert!(format!("{:?}", batch.messages[0].content).contains("context boom"));
    assert_eq!(ends.lock().len(), 2);
    assert_eq!(
        *finalized.lock(),
        vec![("a".into(), true), ("b".into(), false)]
    );
}

struct Aborted;
impl ToolExecutionSignal for Aborted {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn provider_not_sampled_for_missing_or_preaborted_calls() {
    let runtime = Runtime::new();
    let observed = Arc::new(Mutex::new(vec![]));
    probe(&runtime, observed);
    let count = Arc::new(AtomicUsize::new(0));
    for (name, signal) in [
        ("missing", None),
        (
            "probe",
            Some(Arc::new(Aborted) as Arc<dyn ToolExecutionSignal>),
        ),
    ] {
        let sampled = count.clone();
        let mut options = ToolExecutionOptions::new(StopReason::ToolUse, 1.0)
            .with_context_provider(move || {
                sampled.fetch_add(1, Ordering::SeqCst);
                Ok(None)
            });
        if let Some(signal) = signal {
            options = options.with_signal(signal);
        }
        let call = minion_agent::llm::ToolCall::new("a", name, BTreeMap::new());
        execute_tool_calls(&runtime.context(), &[call], options)
            .await
            .unwrap();
    }
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn provider_runs_after_before_hooks_and_never_for_a_blocked_call() {
    use minion_agent::{
        PluginInitError, PluginSpec,
        tools::{
            BeforeToolCallAction, register_after_tool_call_hook, register_before_tool_call_hook,
        },
    };
    let runtime = Runtime::new();
    let stage = Arc::new(AtomicUsize::new(0));
    let finalized = Arc::new(AtomicUsize::new(0));
    let plugin = PluginSpec::<serde_json::Value>::new("context-hooks", vec![], || json!({}), {
        let stage = stage.clone();
        let finalized = finalized.clone();
        move |context, _| {
            let stage = stage.clone();
            let finalized = finalized.clone();
            async move {
                register_before_tool_call_hook(&context, move |current| {
                    let stage = stage.clone();
                    async move {
                        stage.fetch_add(1, Ordering::SeqCst);
                        if current.tool_call_id == "a" {
                            Ok(BeforeToolCallAction::Block {
                                reason: Some("blocked".into()),
                                terminate: false,
                            })
                        } else {
                            Ok(BeforeToolCallAction::Proceed(None))
                        }
                    }
                })
                .map_err(|e| PluginInitError::new(e.to_string()))?;
                register_after_tool_call_hook(&context, move |_current| {
                    finalized.fetch_add(1, Ordering::SeqCst);
                    async move { Ok(None) }
                })
                .map_err(|e| PluginInitError::new(e.to_string()))?;
                Ok(())
            }
        }
    })
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let observed = Arc::new(Mutex::new(vec![]));
    probe(&runtime, observed.clone());
    let sampled = Arc::new(AtomicUsize::new(0));
    let count = sampled.clone();
    let options =
        ToolExecutionOptions::new(StopReason::ToolUse, 1.0).with_context_provider(move || {
            assert_eq!(stage.load(Ordering::SeqCst), 2);
            count.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        });
    let batch = execute_tool_calls(&runtime.context(), &calls(), options)
        .await
        .unwrap();
    assert_eq!(sampled.load(Ordering::SeqCst), 1);
    assert_eq!(*observed.lock(), vec![None]);
    assert!(batch.messages[0].is_error);
    assert!(!batch.messages[1].is_error);
    assert_eq!(finalized.load(Ordering::SeqCst), 1);
}

#[test]
fn snapshot_is_owned_and_preserves_absent_and_off() {
    let mut model = "first".to_owned();
    let snapshot = ToolExecutionContext::new(
        "session".into(),
        None,
        Some("provider".into()),
        Some(model.clone()),
        Some("off".into()),
    );
    model.replace_range(.., "second");
    assert_eq!(snapshot.session_id(), "session");
    assert_eq!(snapshot.session_file(), None);
    assert_eq!(snapshot.provider(), Some("provider"));
    assert_eq!(snapshot.model(), Some("first"));
    assert_eq!(snapshot.reasoning_level(), Some("off"));
}

#[tokio::test]
async fn a_tool_ignoring_the_new_field_has_identical_results() {
    let runtime = Runtime::new();
    runtime
        .tools()
        .register_for_scope(
            None,
            ToolDefinition::new(
                "probe",
                "probe",
                serde_json::from_value(json!({"type":"object"})).unwrap(),
                "probe",
                |_request| Box::pin(async { Ok(output()) }),
            ),
        )
        .unwrap();
    let plain = execute_tool_calls(
        &runtime.context(),
        &calls(),
        ToolExecutionOptions::new(StopReason::ToolUse, 1.0),
    )
    .await
    .unwrap();
    let explicit = execute_tool_calls(
        &runtime.context(),
        &calls(),
        ToolExecutionOptions::new(StopReason::ToolUse, 1.0).with_context_provider(|| {
            Ok(Some(ToolExecutionContext::new(
                "s".into(),
                None,
                Some("p".into()),
                Some("m".into()),
                Some("off".into()),
            )))
        }),
    )
    .await
    .unwrap();
    assert_eq!(plain.messages, explicit.messages);
    assert_eq!(plain.terminate, explicit.terminate);
}
