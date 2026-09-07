use std::{sync::Arc, task::Poll};

use futures::future::poll_fn;

use minion_agent::{
    DynPluginSpec, PluginInitError, PluginSpec, Runtime,
    agent::ThinkingLevel,
    agent_loop::{
        AgentEndReason, AgentEvent, AgentEventKind, AgentListenerError, Enter, PreStepContext,
        PreStepDecision, PreStepReason, PrepareNextTurnContext, Reject, RunConfig, RunConfigUpdate,
        RunContext, ShouldStopAfterTurnContext, TurnStopping, dispatch_agent_event,
        register_agent_listener,
    },
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, StreamChunk, TextBlock,
        ToolResultContentBlock, ToolResultMessage, Usage, UserContent, UserMessage,
    },
    tools::{
        AfterToolCallResult, AgentToolResult, ToolDefinition, ToolExecutionEnd,
        ToolExecutionRequest, ToolExecutionStart, ToolExecutionUpdate,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::oneshot;

fn run(future: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(future);
}

fn identity() -> ModelIdentity {
    ModelIdentity::new("provider", "api", "model").unwrap()
}

fn assistant(text: &str) -> AssistantMessage {
    let mut message = AssistantMessage::new(
        identity(),
        vec![AssistantContentBlock::Text(TextBlock::new(text))],
        Usage::default(),
        minion_agent::llm::StopReason::Stop,
        7.0,
    );
    message.response_id = Some("response-1".into());
    message
}

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn tool_result(text: &str) -> ToolResultMessage {
    let mut message = ToolResultMessage::new(
        "call-1",
        "lookup",
        vec![ToolResultContentBlock::Text(TextBlock::new(text))],
        false,
        8.0,
    );
    message.details = Some(json!({"source": "fixture"}));
    message.added_tool_names = Some(vec!["introduced".into()]);
    message
}

fn tool() -> Arc<ToolDefinition> {
    Arc::new(ToolDefinition::new(
        "lookup",
        "Lookup a value",
        serde_json::from_value(json!({"type": "object"})).unwrap(),
        "Lookup",
        |_request: ToolExecutionRequest| {
            Box::pin(async {
                Ok(AgentToolResult {
                    content: vec![],
                    details: Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    ))
}

#[test]
fn run_and_decision_vocabulary_uses_owned_typed_snapshots_and_exact_wire_names() {
    let definition = tool();
    let context = RunContext {
        system_prompt: "system".into(),
        messages: vec![user("history")],
        tools: vec![Arc::clone(&definition)],
    };
    let cloned = context.clone();
    assert_ne!(context.messages.as_ptr(), cloned.messages.as_ptr());
    assert_ne!(context.tools.as_ptr(), cloned.tools.as_ptr());
    assert!(Arc::ptr_eq(&context.tools[0], &cloned.tools[0]));

    let config = RunConfig {
        model: identity(),
        thinking_level: ThinkingLevel::XHigh,
    };
    assert_eq!(config.model, identity());
    assert_eq!(config.thinking_level, ThinkingLevel::XHigh);

    let update = RunConfigUpdate {
        context: Some(cloned.clone()),
        model: Some(identity()),
        thinking_level: Some(ThinkingLevel::Off),
    };
    assert_eq!(update.context.as_ref().unwrap().system_prompt, "system");
    assert_eq!(update.model, Some(identity()));
    assert_eq!(update.thinking_level, Some(ThinkingLevel::Off));
    assert!(RunConfigUpdate::default().context.is_none());

    let pre_step = PreStepContext {
        messages: vec![user("entering")],
        reason: PreStepReason::ToolResults,
    };
    let decision = PreStepDecision::Enter(Enter {
        messages: pre_step.messages.clone(),
        system_override: None,
        history_window: Some(3),
    });
    assert!(matches!(
        decision,
        PreStepDecision::Enter(Enter { messages, .. }) if messages == pre_step.messages
    ));
    let rejection = PreStepDecision::Reject(Reject {
        reason: String::new(),
    });
    assert!(matches!(
        rejection,
        PreStepDecision::Reject(Reject { reason }) if reason.is_empty()
    ));

    let message = assistant("answer");
    let results = vec![tool_result("result")];
    let new_messages = vec![
        user("entering"),
        Message::Assistant(Box::new(message.clone())),
    ];
    let prepare = PrepareNextTurnContext {
        message: message.clone(),
        tool_results: results.clone(),
        context: cloned.clone(),
        new_messages: new_messages.clone(),
    };
    let stopping = ShouldStopAfterTurnContext {
        message: message.clone(),
        tool_results: results.clone(),
        context: cloned,
        new_messages: new_messages.clone(),
    };
    assert_eq!(prepare.message, message);
    assert_eq!(prepare.tool_results, results);
    assert_eq!(prepare.new_messages, new_messages);
    assert_eq!(stopping.message, prepare.message);
    assert_eq!(stopping.tool_results, prepare.tool_results);
    assert_eq!(
        stopping.context.system_prompt,
        prepare.context.system_prompt
    );
    assert_eq!(stopping.new_messages, prepare.new_messages);

    assert_eq!(
        [
            PreStepReason::Initial,
            PreStepReason::ToolResults,
            PreStepReason::Steering,
            PreStepReason::NextTurn,
            PreStepReason::Continuation,
        ]
        .map(|reason| serde_json::to_value(reason).unwrap()),
        [
            json!("initial"),
            json!("tool_results"),
            json!("steering"),
            json!("next_turn"),
            json!("continuation"),
        ]
    );
    assert_eq!(
        serde_json::to_value(TurnStopping::NoOpinion).unwrap(),
        json!("no_opinion")
    );
    assert_eq!(
        serde_json::to_value(TurnStopping::Continue).unwrap(),
        json!("continue")
    );
    assert_eq!(
        serde_json::to_value(TurnStopping::Stop).unwrap(),
        json!("stop")
    );
    assert_eq!(
        [
            AgentEndReason::Completed,
            AgentEndReason::Terminated,
            AgentEndReason::Stopped,
            AgentEndReason::Rejected,
            AgentEndReason::Error,
            AgentEndReason::Aborted,
            AgentEndReason::Failed,
        ]
        .map(|reason| serde_json::to_value(reason).unwrap()),
        [
            json!("completed"),
            json!("terminated"),
            json!("stopped"),
            json!("rejected"),
            json!("error"),
            json!("aborted"),
            json!("failed"),
        ]
    );
}

#[test]
fn every_agent_event_variant_preserves_its_complete_typed_payload() {
    let partial = assistant("partial");
    let chunk = StreamChunk::TextDelta {
        content_index: 0,
        delta: "ial".into(),
        partial: partial.clone(),
    };
    let start = ToolExecutionStart {
        tool_call_id: "call-1".into(),
        tool_name: "lookup".into(),
        arguments: json!({"query": "rust"}),
    };
    let update = ToolExecutionUpdate {
        tool_call_id: "call-1".into(),
        tool_name: "lookup".into(),
        arguments: json!({"query": "rust"}),
        update: AgentToolResult {
            content: vec![ToolResultContentBlock::Text(TextBlock::new("half"))],
            details: json!({"progress": 0.5}),
            usage: Some(Usage::default()),
            added_tool_names: Some(vec!["introduced".into()]),
            terminate: Some(false),
        },
    };
    let end = ToolExecutionEnd {
        tool_call_id: "call-1".into(),
        tool_name: "lookup".into(),
        result: AfterToolCallResult {
            tool_call_id: "call-1".into(),
            tool_name: "lookup".into(),
            content: vec![ToolResultContentBlock::Text(TextBlock::new("done"))],
            details: Some(json!({"complete": true})),
            usage: Some(Usage::default()),
            added_tool_names: Some(vec!["introduced".into()]),
            is_error: false,
            terminate: Some(true),
        },
    };
    let result = tool_result("result");
    let message = assistant("answer");
    let events = vec![
        AgentEvent::AgentStart { causes: vec![] },
        AgentEvent::TurnStart,
        AgentEvent::MessageStart(user("prompt")),
        AgentEvent::MessageUpdate {
            event: chunk.clone(),
            partial: partial.clone(),
        },
        AgentEvent::MessageEnd(Message::Assistant(Box::new(message.clone()))),
        AgentEvent::ToolExecutionStart(start.clone()),
        AgentEvent::ToolExecutionUpdate(update.clone()),
        AgentEvent::ToolExecutionEnd(end.clone()),
        AgentEvent::TurnEnd {
            message: message.clone(),
            tool_results: vec![result.clone()],
        },
        AgentEvent::AgentEnd {
            reason: minion_agent::agent_loop::AgentEndReason::Completed,
            causes: vec![],
            messages: vec![
                user("prompt"),
                Message::Assistant(Box::new(message.clone())),
            ],
        },
    ];

    let kinds: Vec<_> = events.iter().map(AgentEvent::kind).collect();
    assert_eq!(
        kinds,
        vec![
            AgentEventKind::AgentStart,
            AgentEventKind::TurnStart,
            AgentEventKind::MessageStart,
            AgentEventKind::MessageUpdate,
            AgentEventKind::MessageEnd,
            AgentEventKind::ToolExecutionStart,
            AgentEventKind::ToolExecutionUpdate,
            AgentEventKind::ToolExecutionEnd,
            AgentEventKind::TurnEnd,
            AgentEventKind::AgentEnd,
        ]
    );
    assert_eq!(
        kinds
            .into_iter()
            .map(|kind| serde_json::to_value(kind).unwrap())
            .collect::<Vec<_>>(),
        vec![
            json!("agent_start"),
            json!("turn_start"),
            json!("message_start"),
            json!("message_update"),
            json!("message_end"),
            json!("tool_execution_start"),
            json!("tool_execution_update"),
            json!("tool_execution_end"),
            json!("turn_end"),
            json!("agent_end"),
        ]
    );
    assert!(matches!(&events[2], AgentEvent::MessageStart(value) if *value == user("prompt")));
    assert!(matches!(
        &events[3],
        AgentEvent::MessageUpdate { event, partial: value }
            if event == &chunk && value == &partial
    ));
    assert!(
        matches!(&events[4], AgentEvent::MessageEnd(Message::Assistant(value)) if value.as_ref() == &message)
    );
    assert!(matches!(&events[5], AgentEvent::ToolExecutionStart(value) if value == &start));
    assert!(matches!(&events[6], AgentEvent::ToolExecutionUpdate(value) if value == &update));
    assert!(matches!(&events[7], AgentEvent::ToolExecutionEnd(value) if value == &end));
    assert!(matches!(
        &events[8],
        AgentEvent::TurnEnd { message: value, tool_results }
            if value == &message && tool_results == &vec![result.clone()]
    ));
    assert!(matches!(
        &events[9],
        AgentEvent::AgentEnd { messages, .. }
            if messages == &vec![user("prompt"), Message::Assistant(Box::new(message))]
    ));
}

fn listeners_plugin(
    seen: Arc<Mutex<Vec<&'static str>>>,
    gate: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    started: Arc<Mutex<Option<oneshot::Sender<()>>>>,
) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "agent-event-listeners",
        vec![],
        || json!({}),
        move |context, _config| {
            let seen = Arc::clone(&seen);
            let gate = Arc::clone(&gate);
            let started = Arc::clone(&started);
            async move {
                let first_seen = Arc::clone(&seen);
                register_agent_listener(&context, move |_event| {
                    let seen = Arc::clone(&first_seen);
                    let receiver = gate.lock().take();
                    let started = started.lock().take();
                    async move {
                        seen.lock().push("first");
                        if let Some(started) = started {
                            started.send(()).unwrap();
                        }
                        if let Some(receiver) = receiver {
                            receiver.await.unwrap();
                        }
                        Ok(())
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                let second_seen = Arc::clone(&seen);
                register_agent_listener(&context, move |_event| {
                    let seen = Arc::clone(&second_seen);
                    async move {
                        seen.lock().push("second");
                        Ok(())
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

#[test]
fn agent_listeners_are_awaited_serially_in_registration_order() {
    run(async {
        let runtime = Runtime::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (release, gate) = oneshot::channel();
        let (started, first_started) = oneshot::channel();
        let plugin = listeners_plugin(
            Arc::clone(&seen),
            Arc::new(Mutex::new(Some(gate))),
            Arc::new(Mutex::new(Some(started))),
        );
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let context = runtime.context();
        let dispatch = tokio::spawn(async move {
            dispatch_agent_event(&context, AgentEvent::TurnStart)
                .await
                .unwrap();
        });
        first_started.await.unwrap();
        assert_eq!(&*seen.lock(), &["first"]);
        release.send(()).unwrap();
        dispatch.await.unwrap();
        assert_eq!(&*seen.lock(), &["first", "second"]);
    });
}

#[test]
fn first_agent_listener_error_prevents_later_listeners() {
    run(async {
        let runtime = Runtime::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&seen);
        let plugin = PluginSpec::<Value>::new(
            "failing-agent-event-listeners",
            vec![],
            || json!({}),
            move |context, _config| {
                let seen = Arc::clone(&observed);
                async move {
                    let first_seen = Arc::clone(&seen);
                    register_agent_listener(&context, move |_event| {
                        let seen = Arc::clone(&first_seen);
                        async move {
                            seen.lock().push("failed");
                            Err(AgentListenerError::new("listener exploded"))
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    let second_seen = Arc::clone(&seen);
                    register_agent_listener(&context, move |_event| {
                        let seen = Arc::clone(&second_seen);
                        async move {
                            seen.lock().push("must-not-run");
                            Ok(())
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                }
            },
        )
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let error = dispatch_agent_event(
            &runtime.context(),
            AgentEvent::AgentStart { causes: vec![] },
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.listener_error().unwrap().message(),
            "listener exploded"
        );
        assert_eq!(&*seen.lock(), &["failed"]);
    });
}

#[test]
fn eagerly_ready_listeners_still_have_a_scheduling_boundary_between_them() {
    run(async {
        let runtime = Runtime::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&seen);
        let plugin = PluginSpec::<Value>::new(
            "yielding-agent-event-listeners",
            vec![],
            || json!({}),
            move |context, _config| {
                let seen = Arc::clone(&observed);
                async move {
                    let first_seen = Arc::clone(&seen);
                    register_agent_listener(&context, move |_event| {
                        let seen = Arc::clone(&first_seen);
                        async move {
                            seen.lock().push("first");
                            Ok(())
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    let second_seen = Arc::clone(&seen);
                    register_agent_listener(&context, move |_event| {
                        let seen = Arc::clone(&second_seen);
                        async move {
                            seen.lock().push("second");
                            Ok(())
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                }
            },
        )
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let context = runtime.context();
        let mut dispatch = Box::pin(dispatch_agent_event(&context, AgentEvent::TurnStart));
        poll_fn(|task_context| match dispatch.as_mut().poll(task_context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(result) => {
                panic!("dispatch completed without a per-listener boundary: {result:?}")
            }
        })
        .await;
        assert_eq!(&*seen.lock(), &["first"]);
        dispatch.await.unwrap();
        assert_eq!(&*seen.lock(), &["first", "second"]);
    });
}
