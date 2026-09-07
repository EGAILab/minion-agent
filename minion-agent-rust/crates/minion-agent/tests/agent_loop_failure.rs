use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use minion_agent::{
    DynPluginSpec, PluginInitError, PluginSpec, Runtime,
    agent::{AgentDefinition, AgentInstance, AgentStatus},
    agent_loop::{
        AgentEvent, AgentEventKind, AgentListenerError, AgentLoop, AgentLoopError, PromptInput,
        RunConfigUpdate, register_agent_listener, register_prepare_next_turn_listener,
    },
    llm::{
        AssistantContentBlock, AssistantMessage, DoneReason, ImageBlock, LlmService, Message,
        ModelIdentity, Script, ScriptItem, ScriptedAdapter, StopReason, StreamChunk, TextBlock,
        ToolResultContentBlock, ToolResultMessage, Usage, UserContent, UserContentBlock,
        UserMessage,
    },
    session::Session,
};
use parking_lot::Mutex;
use serde_json::{Value, json};

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

fn named_identity(name: &str) -> ModelIdentity {
    ModelIdentity::new(
        format!("provider-{name}"),
        format!("api-{name}"),
        format!("model-{name}"),
    )
    .unwrap()
}

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn text_turn(text: &str) -> Script {
    let message = AssistantMessage::new(
        identity(),
        vec![AssistantContentBlock::Text(TextBlock::new(text))],
        Usage::default(),
        StopReason::Stop,
        2.0,
    );
    Script::new([ScriptItem::Chunk(Box::new(StreamChunk::Done {
        reason: DoneReason::Stop,
        message,
    }))])
}

fn setup(scripts: impl IntoIterator<Item = Script>) -> (Runtime, AgentLoop, Arc<AgentInstance>) {
    let runtime = Runtime::new();
    let llm = Arc::new(LlmService::new());
    llm.register(identity(), Arc::new(ScriptedAdapter::new(scripts)));
    let session = Session::new("room-a", [] as [&str; 0]).unwrap();
    let agent = Arc::new(AgentInstance::new(
        "room-a",
        AgentDefinition::new("ada", "system", identity()),
        session,
        Some(runtime.context()),
        None,
    ));
    let driver = AgentLoop::new(Arc::clone(&agent), runtime.context(), llm);
    (runtime, driver, agent)
}

type Listener = Arc<dyn Fn(AgentEvent) -> Result<(), AgentListenerError> + Send + Sync>;

fn listener_plugin(name: &'static str, listener: Listener) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        name,
        vec![],
        || json!({}),
        move |context, _config| {
            let listener = Arc::clone(&listener);
            async move {
                register_agent_listener(&context, move |event| {
                    let listener = Arc::clone(&listener);
                    async move { listener(event) }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn override_model_then_fail_plugin(
    agent: Arc<AgentInstance>,
    run_local_model: ModelIdentity,
    persistent_model_at_failure: Option<ModelIdentity>,
) -> DynPluginSpec {
    let turn_starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    PluginSpec::<Value>::new(
        "model-identity-failure-listeners",
        vec![],
        || json!({}),
        move |context, _config| {
            let agent = Arc::clone(&agent);
            let run_local_model = run_local_model.clone();
            let persistent_model_at_failure = persistent_model_at_failure.clone();
            let turn_starts = Arc::clone(&turn_starts);
            async move {
                let event_agent = Arc::clone(&agent);
                register_agent_listener(&context, move |event| {
                    let agent = Arc::clone(&event_agent);
                    let persistent_model_at_failure = persistent_model_at_failure.clone();
                    let turn_starts = Arc::clone(&turn_starts);
                    async move {
                        match event {
                            AgentEvent::TurnEnd { .. } => {
                                agent.steer(user("continue to failing turn"), None);
                            }
                            AgentEvent::TurnStart
                                if turn_starts.fetch_add(1, Ordering::SeqCst) == 1 =>
                            {
                                if let Some(model) = persistent_model_at_failure {
                                    agent.set_model(model);
                                }
                                return Err(AgentListenerError::new("second turn failed"));
                            }
                            _ => {}
                        }
                        Ok(())
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                register_prepare_next_turn_listener(&context, move |_current, _next| {
                    let model = run_local_model.clone();
                    async move {
                        Ok(RunConfigUpdate {
                            model: Some(model),
                            ..RunConfigUpdate::default()
                        })
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn failure_message(messages: &[Message]) -> &AssistantMessage {
    match messages {
        [Message::Assistant(message)] if message.stop_reason == StopReason::Error => message,
        other => panic!("expected one synthesized assistant failure, got {other:?}"),
    }
}

fn message_is_failure(event: &AgentEvent) -> bool {
    match event {
        AgentEvent::MessageStart(Message::Assistant(message))
        | AgentEvent::MessageEnd(Message::Assistant(message)) => {
            message.stop_reason == StopReason::Error
        }
        AgentEvent::TurnEnd { message, .. } => message.stop_reason == StopReason::Error,
        AgentEvent::AgentEnd { messages } => matches!(
            messages.as_slice(),
            [Message::Assistant(message)] if message.stop_reason == StopReason::Error
        ),
        AgentEvent::AgentStart
        | AgentEvent::TurnStart
        | AgentEvent::MessageStart(_)
        | AgentEvent::MessageUpdate { .. }
        | AgentEvent::MessageEnd(_)
        | AgentEvent::ToolExecutionStart(_)
        | AgentEvent::ToolExecutionUpdate(_)
        | AgentEvent::ToolExecutionEnd(_) => false,
    }
}

#[test]
fn public_prompt_runs_the_complete_state_machine_and_settles_idle() {
    run(async {
        let (_runtime, driver, agent) = setup([text_turn("answer")]);

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(agent.messages().unwrap(), messages);
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.streaming_message(), None);
        assert!(agent.pending_tool_calls().is_empty());
        assert_eq!(agent.error_message(), None);
    });
}

#[test]
fn public_prompt_preserves_typed_batches_and_text_image_convenience_input() {
    run(async {
        let (_runtime, driver, _agent) = setup([
            text_turn("first answer"),
            text_turn("second answer"),
            text_turn("third answer"),
        ]);
        let assistant_input = Message::Assistant(Box::new(AssistantMessage::new(
            identity(),
            vec![AssistantContentBlock::Text(TextBlock::new(
                "assistant input",
            ))],
            Usage::default(),
            StopReason::Stop,
            3.0,
        )));
        let tool_input = Message::ToolResult(Box::new(ToolResultMessage::new(
            "call-1",
            "lookup",
            vec![ToolResultContentBlock::Text(TextBlock::new("tool input"))],
            false,
            4.0,
        )));

        let single = driver
            .prompt(PromptInput::Message(user("single")))
            .await
            .unwrap();
        assert_eq!(single.first(), Some(&user("single")));

        let batch = driver
            .prompt(PromptInput::Messages(vec![
                assistant_input.clone(),
                tool_input.clone(),
            ]))
            .await
            .unwrap();
        assert_eq!(&batch[..2], &[assistant_input, tool_input]);

        let image = ImageBlock::data("image/png", "aW1hZ2U=");
        let convenience = driver
            .prompt(PromptInput::Text {
                text: "describe".into(),
                images: vec![image.clone()],
            })
            .await
            .unwrap();
        let Message::User(prompt) = &convenience[0] else {
            panic!("convenience prompt must produce one user message")
        };
        assert_eq!(
            prompt.content,
            UserContent::Blocks(vec![
                UserContentBlock::Text(TextBlock::new("describe")),
                UserContentBlock::Image(image),
            ])
        );
    });
}

#[test]
fn ordinary_listener_failure_recovers_without_inventing_turn_start() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("unused")]);
        let trace = Arc::new(Mutex::new(Vec::new()));
        let fail_turn = Arc::new(AtomicBool::new(true));
        runtime
            .mount(
                &listener_plugin("recovering-listener", {
                    let trace = Arc::clone(&trace);
                    let fail_turn = Arc::clone(&fail_turn);
                    let observed_agent = Arc::clone(&agent);
                    Arc::new(move |event| {
                        assert_eq!(observed_agent.status(), AgentStatus::Running);
                        trace.lock().push(event.kind());
                        if event.kind() == AgentEventKind::TurnStart
                            && fail_turn.swap(false, Ordering::SeqCst)
                        {
                            return Err(AgentListenerError::new("turn listener failed"));
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();
        let failure = failure_message(&messages);

        assert_eq!(
            failure.error_message.as_deref(),
            Some("turn listener failed")
        );
        assert_eq!(
            trace.lock().as_slice(),
            [
                AgentEventKind::AgentStart,
                AgentEventKind::TurnStart,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
                AgentEventKind::TurnEnd,
                AgentEventKind::AgentEnd,
            ]
        );
        assert_eq!(agent.messages().unwrap(), messages);
        assert_eq!(
            agent.error_message().as_deref(),
            Some("turn listener failed")
        );
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.streaming_message(), None);
        assert!(agent.pending_tool_calls().is_empty());
    });
}

#[test]
fn recovery_listener_failure_interrupts_recovery_but_guard_still_settles() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("unused")]);
        let trace = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin("interrupting-recovery-listener", {
                    let trace = Arc::clone(&trace);
                    Arc::new(move |event| {
                        trace.lock().push(event.kind());
                        match event {
                            AgentEvent::TurnStart => {
                                Err(AgentListenerError::new("ordinary failure"))
                            }
                            AgentEvent::MessageEnd(Message::Assistant(message))
                                if message.stop_reason == StopReason::Error =>
                            {
                                Err(AgentListenerError::new("recovery message-end failed"))
                            }
                            _ => Ok(()),
                        }
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        let error = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap_err();

        assert_eq!(
            error.listener_error().unwrap().message(),
            "recovery message-end failed"
        );
        assert_eq!(
            trace.lock().as_slice(),
            [
                AgentEventKind::AgentStart,
                AgentEventKind::TurnStart,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
            ]
        );
        let transcript = agent.messages().unwrap();
        assert_eq!(
            failure_message(&transcript).error_message.as_deref(),
            Some("ordinary failure")
        );
        assert_eq!(agent.error_message(), None);
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.streaming_message(), None);
        assert!(agent.pending_tool_calls().is_empty());
    });
}

#[test]
fn every_recovery_listener_boundary_interrupts_only_after_its_own_reduction() {
    run(async {
        for boundary in [
            AgentEventKind::MessageStart,
            AgentEventKind::MessageEnd,
            AgentEventKind::TurnEnd,
            AgentEventKind::AgentEnd,
        ] {
            let (runtime, driver, agent) = setup([text_turn("unused")]);
            let trace = Arc::new(Mutex::new(Vec::new()));
            runtime
                .mount(
                    &listener_plugin("recovery-boundary-listener", {
                        let trace = Arc::clone(&trace);
                        Arc::new(move |event| {
                            trace.lock().push(event.kind());
                            if matches!(event, AgentEvent::TurnStart) {
                                return Err(AgentListenerError::new("ordinary failure"));
                            }
                            if event.kind() == boundary && message_is_failure(&event) {
                                return Err(AgentListenerError::new("recovery failure"));
                            }
                            Ok(())
                        })
                    }),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();

            let error = driver
                .prompt(PromptInput::Message(user("prompt")))
                .await
                .unwrap_err();
            assert_eq!(
                error.listener_error().unwrap().message(),
                "recovery failure"
            );
            assert_eq!(trace.lock().last().copied(), Some(boundary));
            assert_eq!(agent.status(), AgentStatus::Idle);
            assert_eq!(agent.streaming_message(), None);
            assert!(agent.pending_tool_calls().is_empty());

            let transcript = agent.messages().unwrap();
            if boundary == AgentEventKind::MessageStart {
                assert!(transcript.is_empty());
            } else {
                assert_eq!(
                    failure_message(&transcript).error_message.as_deref(),
                    Some("ordinary failure")
                );
            }
            assert_eq!(
                agent.error_message().as_deref(),
                matches!(boundary, AgentEventKind::TurnEnd | AgentEventKind::AgentEnd)
                    .then_some("ordinary failure")
            );
        }
    });
}

#[test]
fn no_start_provider_message_listener_failure_enters_recovery() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("answer")]);
        let trace = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin("provider-message-listener", {
                    let trace = Arc::clone(&trace);
                    Arc::new(move |event| {
                        trace.lock().push(event.kind());
                        if matches!(
                            event,
                            AgentEvent::MessageStart(Message::Assistant(ref message))
                                if message.stop_reason == StopReason::Stop
                        ) {
                            return Err(AgentListenerError::new("provider message-start failed"));
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();

        assert_eq!(
            failure_message(&messages).error_message.as_deref(),
            Some("provider message-start failed")
        );
        assert_eq!(
            trace.lock().as_slice(),
            [
                AgentEventKind::AgentStart,
                AgentEventKind::TurnStart,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
                AgentEventKind::TurnEnd,
                AgentEventKind::AgentEnd,
            ]
        );
        assert_eq!(agent.status(), AgentStatus::Idle);
    });
}

#[test]
fn post_turn_listener_failure_enters_recovery_with_raw_message() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("answer")]);
        let plugin = PluginSpec::<Value>::new(
            "failing-post-turn-listener",
            vec![],
            || json!({}),
            move |context, _config| async move {
                register_prepare_next_turn_listener(&context, |_current, _next| async move {
                    Err(minion_agent::WaterfallError::ListenerFailed(
                        "post-turn failed".into(),
                    ))
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            },
        )
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();

        assert_eq!(
            failure_message(&messages).error_message.as_deref(),
            Some("post-turn failed")
        );
        assert_eq!(agent.error_message().as_deref(), Some("post-turn failed"));
        assert_eq!(agent.status(), AgentStatus::Idle);
    });
}

#[test]
fn ordinary_agent_end_listener_failure_is_recovered_inside_the_run_boundary() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("answer")]);
        let failed = Arc::new(AtomicBool::new(false));
        runtime
            .mount(
                &listener_plugin("agent-end-listener", {
                    let failed = Arc::clone(&failed);
                    Arc::new(move |event| {
                        if matches!(event, AgentEvent::AgentEnd { ref messages } if messages.len() > 1)
                            && !failed.swap(true, Ordering::SeqCst)
                        {
                            return Err(AgentListenerError::new("agent-end failed"));
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();

        assert_eq!(
            failure_message(&messages).error_message.as_deref(),
            Some("agent-end failed")
        );
        assert_eq!(agent.error_message().as_deref(), Some("agent-end failed"));
        assert_eq!(agent.status(), AgentStatus::Idle);
    });
}

#[test]
fn public_entry_errors_keep_their_exact_distinct_messages() {
    run(async {
        let (_runtime, driver, agent) = setup([text_turn("unused")]);

        let empty = driver.continue_run().await.unwrap_err();
        assert_eq!(empty.to_string(), "No messages to continue from");

        agent.try_begin_run().unwrap();
        let prompt = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap_err();
        let continuation = driver.continue_run().await.unwrap_err();
        assert_eq!(
            prompt.to_string(),
            "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
        );
        assert_eq!(
            continuation.to_string(),
            "Agent is already processing. Wait for completion before continuing."
        );
        agent.finish_run();
    });
}

#[test]
fn public_continue_runs_from_an_existing_non_assistant_transcript() {
    run(async {
        let (_runtime, driver, agent) = setup([text_turn("continued")]);
        agent.session().append_message(user("existing")).unwrap();

        let messages = driver.continue_run().await.unwrap();

        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0], Message::Assistant(_)));
        assert_eq!(agent.messages().unwrap().len(), 2);
        assert_eq!(agent.status(), AgentStatus::Idle);
    });
}

#[test]
fn unknown_model_remains_eager_and_does_not_synthesize_failure() {
    run(async {
        let runtime = Runtime::new();
        let agent = Arc::new(AgentInstance::new(
            "room-a",
            AgentDefinition::new("ada", "system", identity()),
            Session::new("room-a", [] as [&str; 0]).unwrap(),
            Some(runtime.context()),
            None,
        ));
        let driver = AgentLoop::new(
            Arc::clone(&agent),
            runtime.context(),
            Arc::new(LlmService::new()),
        );

        let error = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap_err();

        assert!(matches!(error, AgentLoopError::LlmStart(_)));
        assert_eq!(agent.messages().unwrap(), vec![user("prompt")]);
        assert_eq!(agent.error_message(), None);
        assert_eq!(agent.status(), AgentStatus::Idle);
    });
}

#[test]
fn adapter_start_failure_inside_the_run_enters_recovery() {
    run(async {
        let (_runtime, driver, agent) = setup([]);

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();

        assert!(
            failure_message(&messages)
                .error_message
                .as_deref()
                .unwrap()
                .contains("scripted adapter has no remaining script")
        );
        assert_eq!(
            agent.error_message(),
            failure_message(&messages).error_message
        );
        assert_eq!(agent.status(), AgentStatus::Idle);
    });
}

#[test]
fn run_local_model_replacement_does_not_change_synthesized_failure_identity() {
    run(async {
        let persistent_a = named_identity("a");
        let run_local_b = named_identity("b");
        let runtime = Runtime::new();
        let llm = Arc::new(LlmService::new());
        llm.register(
            persistent_a.clone(),
            Arc::new(ScriptedAdapter::new([Script::new([ScriptItem::Chunk(
                Box::new(StreamChunk::Done {
                    reason: DoneReason::Stop,
                    message: AssistantMessage::new(
                        persistent_a.clone(),
                        vec![AssistantContentBlock::Text(TextBlock::new("first"))],
                        Usage::default(),
                        StopReason::Stop,
                        2.0,
                    ),
                }),
            )])])),
        );
        let agent = Arc::new(AgentInstance::new(
            "room-a",
            AgentDefinition::new("ada", "system", persistent_a.clone()),
            Session::new("room-a", [] as [&str; 0]).unwrap(),
            Some(runtime.context()),
            None,
        ));
        runtime
            .mount(
                &override_model_then_fail_plugin(Arc::clone(&agent), run_local_b, None),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();
        let driver = AgentLoop::new(Arc::clone(&agent), runtime.context(), llm);

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();
        let failure = failure_message(&messages);

        assert_eq!(failure.provider, persistent_a.provider());
        assert_eq!(failure.api, persistent_a.api());
        assert_eq!(failure.model, persistent_a.model_id());
    });
}

#[test]
fn persistent_model_mutation_during_run_is_read_live_by_failure_settlement() {
    run(async {
        let persistent_a = named_identity("a");
        let run_local_b = named_identity("b");
        let persistent_c = named_identity("c");
        let runtime = Runtime::new();
        let llm = Arc::new(LlmService::new());
        llm.register(
            persistent_a.clone(),
            Arc::new(ScriptedAdapter::new([Script::new([ScriptItem::Chunk(
                Box::new(StreamChunk::Done {
                    reason: DoneReason::Stop,
                    message: AssistantMessage::new(
                        persistent_a,
                        vec![AssistantContentBlock::Text(TextBlock::new("first"))],
                        Usage::default(),
                        StopReason::Stop,
                        2.0,
                    ),
                }),
            )])])),
        );
        let agent = Arc::new(AgentInstance::new(
            "room-a",
            AgentDefinition::new("ada", "system", named_identity("a")),
            Session::new("room-a", [] as [&str; 0]).unwrap(),
            Some(runtime.context()),
            None,
        ));
        runtime
            .mount(
                &override_model_then_fail_plugin(
                    Arc::clone(&agent),
                    run_local_b,
                    Some(persistent_c.clone()),
                ),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();
        let driver = AgentLoop::new(Arc::clone(&agent), runtime.context(), llm);

        let messages = driver
            .prompt(PromptInput::Message(user("prompt")))
            .await
            .unwrap();
        let failure = failure_message(&messages);

        assert_eq!(failure.provider, persistent_c.provider());
        assert_eq!(failure.api, persistent_c.api());
        assert_eq!(failure.model, persistent_c.model_id());
    });
}
