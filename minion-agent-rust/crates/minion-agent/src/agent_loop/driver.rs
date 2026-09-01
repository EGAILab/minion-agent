// Tasks 4-5 deliberately stage private admission and provider-turn
// transactions before Task 8 wires the complete public run entries.
#![allow(dead_code)]

use std::{
    fmt,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use futures::{StreamExt, future::BoxFuture};

use crate::{
    Context,
    agent::{
        AgentInstance, AgentRunError, AgentStatus, ClaimPolicy, InboxTarget,
        ThinkingLevel as AgentThinkingLevel,
    },
    llm::{
        AssistantContentBlock, AssistantMessage, ImageBlock, LlmContext, LlmRequest, LlmService,
        Message, SimpleStreamOptions, StopReason, StreamChunk, TextBlock,
        ThinkingLevel as LlmThinkingLevel, UserContent, UserContentBlock, UserMessage,
    },
    tools::{
        ToolExecutionBatchResult, ToolExecutionOptions, ToolLifecycleError, execute_tool_calls,
    },
};

use super::{
    AgentEvent, AgentLoopError, RunConfig, RunContext, RunSnapshot, dispatch_agent_event,
    reduce_event,
};

/// Input accepted by the eventual public prompt entry point.
#[derive(Clone, Debug, PartialEq)]
pub enum PromptInput {
    Message(Message),
    Messages(Vec<Message>),
    Text {
        text: String,
        images: Vec<ImageBlock>,
    },
}

/// Layer-08 run driver.
///
/// The Agent remains the sole authority for mutable run state. The Context and
/// LLM values are shared handles to their existing lower-layer authorities.
pub struct AgentLoop {
    agent: Arc<AgentInstance>,
    context: Context,
    llm: Arc<LlmService>,
}

struct PreparedRun {
    agent: Arc<AgentInstance>,
    context: RunContext,
    config: RunConfig,
    new_messages: Vec<Message>,
}

#[derive(Debug, PartialEq)]
struct ProviderTurn {
    message: AssistantMessage,
    disposition: ProviderTurnDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderTurnDisposition {
    Continue,
    RepresentedTerminal,
}

impl fmt::Debug for PreparedRun {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedRun")
            .field("context", &self.context)
            .field("config", &self.config)
            .field("new_messages", &self.new_messages)
            .finish_non_exhaustive()
    }
}

impl Drop for PreparedRun {
    fn drop(&mut self) {
        self.agent.finish_run();
    }
}

impl AgentLoop {
    pub fn new(agent: Arc<AgentInstance>, context: Context, llm: Arc<LlmService>) -> Self {
        Self {
            agent,
            context,
            llm,
        }
    }

    async fn prepare_prompt_run(&self, input: PromptInput) -> Result<PreparedRun, AgentLoopError> {
        if self.agent.status() != AgentStatus::Idle {
            return Err(AgentLoopError::PromptActive);
        }
        let entering = normalize_prompt_input(input);
        let snapshot = self.agent.try_begin_run().map_err(map_prompt_entry_error)?;
        self.prepare_first_turn(snapshot, entering, false).await
    }

    async fn prepare_continue_run(&self) -> Result<PreparedRun, AgentLoopError> {
        if self.agent.status() != AgentStatus::Idle {
            return Err(AgentLoopError::ContinueActive);
        }

        let messages = self.agent.session().derive_messages()?;
        let Some(last_message) = messages.last() else {
            return Err(AgentLoopError::NoMessagesToContinue);
        };
        let assistant_last = matches!(last_message, Message::Assistant(_));
        if assistant_last && !self.agent.has_queued_messages() {
            return Err(AgentLoopError::CannotContinueFromAssistant);
        }

        let snapshot = self
            .agent
            .try_begin_run()
            .map_err(map_continue_entry_error)?;
        if !assistant_last {
            return self.prepare_first_turn(snapshot, Vec::new(), false).await;
        }

        let steering = self.claim(InboxTarget::Steering);
        if !steering.is_empty() {
            return self.prepare_first_turn(snapshot, steering, true).await;
        }
        let follow_up = self.claim(InboxTarget::FollowUp);
        if !follow_up.is_empty() {
            return self.prepare_first_turn(snapshot, follow_up, false).await;
        }

        // Another claimant may have drained the observed queue between the
        // validation and this run's atomic entry.
        self.agent.finish_run();
        Err(AgentLoopError::CannotContinueFromAssistant)
    }

    async fn prepare_first_turn(
        &self,
        snapshot: RunSnapshot,
        entering: Vec<Message>,
        skip_initial_steering_poll: bool,
    ) -> Result<PreparedRun, AgentLoopError> {
        let mut prepared = PreparedRun {
            agent: Arc::clone(&self.agent),
            context: snapshot.context,
            config: snapshot.config,
            new_messages: Vec::new(),
        };
        self.dispatch(AgentEvent::AgentStart).await?;
        self.dispatch(AgentEvent::TurnStart).await?;
        self.admit(&mut prepared, entering).await?;

        if !skip_initial_steering_poll {
            let steering = self.claim(InboxTarget::Steering);
            self.admit(&mut prepared, steering).await?;
        }
        Ok(prepared)
    }

    async fn admit(
        &self,
        prepared: &mut PreparedRun,
        messages: Vec<Message>,
    ) -> Result<(), AgentLoopError> {
        for message in messages {
            self.dispatch(AgentEvent::MessageStart(message.clone()))
                .await?;
            self.dispatch(AgentEvent::MessageEnd(message.clone()))
                .await?;
            prepared.context.messages.push(message.clone());
            prepared.new_messages.push(message);
        }
        Ok(())
    }

    async fn run_provider_turn(
        &self,
        prepared: &mut PreparedRun,
    ) -> Result<ProviderTurn, AgentLoopError> {
        let request = LlmRequest {
            model: prepared.config.model.clone(),
            context: LlmContext {
                system_prompt: Some(prepared.context.system_prompt.clone()),
                messages: prepared.context.messages.clone(),
                tools: Some(
                    prepared
                        .context
                        .tools
                        .iter()
                        .map(|tool| tool.schema())
                        .collect(),
                ),
            },
            options: SimpleStreamOptions {
                reasoning: provider_thinking_level(prepared.config.thinking_level),
                ..SimpleStreamOptions::default()
            },
        };
        let mut stream = self.llm.stream(request)?;
        let mut started = false;

        while let Some(event) = stream.next().await {
            let partial = event.partial().clone();
            match event {
                StreamChunk::Start { .. } => {
                    self.dispatch(AgentEvent::MessageStart(Message::Assistant(Box::new(
                        partial,
                    ))))
                    .await?;
                    started = true;
                }
                StreamChunk::Done { .. } | StreamChunk::Error { .. } => {
                    let message = partial;
                    if !started {
                        self.dispatch(AgentEvent::MessageStart(Message::Assistant(Box::new(
                            message.clone(),
                        ))))
                        .await?;
                    }
                    self.dispatch(AgentEvent::MessageEnd(Message::Assistant(Box::new(
                        message.clone(),
                    ))))
                    .await?;
                    prepared
                        .context
                        .messages
                        .push(Message::Assistant(Box::new(message.clone())));
                    prepared
                        .new_messages
                        .push(Message::Assistant(Box::new(message.clone())));
                    let disposition = match message.stop_reason {
                        StopReason::Error | StopReason::Aborted => {
                            ProviderTurnDisposition::RepresentedTerminal
                        }
                        StopReason::Pending
                        | StopReason::Stop
                        | StopReason::Length
                        | StopReason::ToolUse
                        | StopReason::Deferred => ProviderTurnDisposition::Continue,
                    };
                    return Ok(ProviderTurn {
                        message,
                        disposition,
                    });
                }
                event => {
                    if started {
                        self.dispatch(AgentEvent::MessageUpdate { event, partial })
                            .await?;
                    }
                }
            }
        }

        unreachable!("AssistantStream emits a terminal chunk before it fuses")
    }

    async fn run_tool_calls(
        &self,
        prepared: &mut PreparedRun,
        assistant: &AssistantMessage,
    ) -> Result<ToolExecutionBatchResult, AgentLoopError> {
        let calls = assistant
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::ToolCall(call) => Some(call.clone()),
                AssistantContentBlock::Text(_) | AssistantContentBlock::Thinking(_) => None,
            })
            .collect::<Vec<_>>();
        let start_agent = Arc::clone(&self.agent);
        let start_context = self.context.clone();
        let update_agent = Arc::clone(&self.agent);
        let update_context = self.context.clone();
        let end_agent = Arc::clone(&self.agent);
        let end_context = self.context.clone();
        let options = ToolExecutionOptions::new(assistant.stop_reason, now_millis())
            .with_execution_tools(prepared.context.tools.clone())
            .with_execution_start(move |event| {
                live_tool_event(
                    Arc::clone(&start_agent),
                    start_context.clone(),
                    AgentEvent::ToolExecutionStart(event),
                )
            })
            .with_execution_update(move |event| {
                live_tool_event(
                    Arc::clone(&update_agent),
                    update_context.clone(),
                    AgentEvent::ToolExecutionUpdate(event),
                )
            })
            .with_execution_end(move |event| {
                live_tool_event(
                    Arc::clone(&end_agent),
                    end_context.clone(),
                    AgentEvent::ToolExecutionEnd(event),
                )
            });
        let batch = execute_tool_calls(&self.context, &calls, options).await?;

        for message in &batch.messages {
            self.admit(
                prepared,
                vec![Message::ToolResult(Box::new(message.clone()))],
            )
            .await?;
            self.extend_run_tools(prepared, message.added_tool_names.as_deref())?;
        }
        Ok(batch)
    }

    fn extend_run_tools(
        &self,
        prepared: &mut PreparedRun,
        added_tool_names: Option<&[String]>,
    ) -> Result<(), AgentLoopError> {
        let Some(added_tool_names) = added_tool_names else {
            return Ok(());
        };
        let registry = self.context.tools()?;
        for name in added_tool_names {
            if prepared
                .context
                .tools
                .iter()
                .any(|tool| tool.name() == name)
            {
                continue;
            }
            if let Some(tool) = registry.resolve(name, self.context.scope()) {
                prepared.context.tools.push(tool);
            }
        }
        Ok(())
    }

    async fn dispatch(&self, event: AgentEvent) -> Result<(), AgentLoopError> {
        reduce_event(&self.agent, &event)?;
        dispatch_agent_event(&self.context, event).await
    }

    fn claim(&self, target: InboxTarget) -> Vec<Message> {
        self.agent
            .inbox()
            .claim(target, ClaimPolicy::OneAtATime)
            .into_iter()
            .map(|envelope| envelope.message)
            .collect()
    }
}

fn live_tool_event(
    agent: Arc<AgentInstance>,
    context: Context,
    event: AgentEvent,
) -> BoxFuture<'static, Result<(), ToolLifecycleError>> {
    let reduction = reduce_event(&agent, &event).map_err(tool_lifecycle_error);
    Box::pin(async move {
        reduction?;
        dispatch_agent_event(&context, event)
            .await
            .map_err(tool_lifecycle_error)
    })
}

fn tool_lifecycle_error(error: AgentLoopError) -> ToolLifecycleError {
    if let Some(listener) = error.listener_error() {
        return ToolLifecycleError::new(listener.message());
    }
    ToolLifecycleError::new(error.to_string())
}

fn provider_thinking_level(level: AgentThinkingLevel) -> Option<LlmThinkingLevel> {
    match level {
        AgentThinkingLevel::Off => None,
        AgentThinkingLevel::Minimal => Some(LlmThinkingLevel::Minimal),
        AgentThinkingLevel::Low => Some(LlmThinkingLevel::Low),
        AgentThinkingLevel::Medium => Some(LlmThinkingLevel::Medium),
        AgentThinkingLevel::High => Some(LlmThinkingLevel::High),
        AgentThinkingLevel::XHigh => Some(LlmThinkingLevel::Xhigh),
        AgentThinkingLevel::Max => Some(LlmThinkingLevel::Max),
    }
}

fn map_prompt_entry_error(error: AgentRunError) -> AgentLoopError {
    match error {
        AgentRunError::Active => AgentLoopError::PromptActive,
        other => other.into(),
    }
}

fn map_continue_entry_error(error: AgentRunError) -> AgentLoopError {
    match error {
        AgentRunError::Active => AgentLoopError::ContinueActive,
        other => other.into(),
    }
}

fn normalize_prompt_input(input: PromptInput) -> Vec<Message> {
    match input {
        PromptInput::Message(message) => vec![message],
        PromptInput::Messages(messages) => messages,
        PromptInput::Text { text, images } => {
            let mut content = Vec::with_capacity(images.len() + 1);
            content.push(UserContentBlock::Text(TextBlock::new(text)));
            content.extend(images.into_iter().map(UserContentBlock::Image));
            vec![Message::User(UserMessage::new(
                UserContent::Blocks(content),
                now_millis(),
            ))]
        }
    }
}

fn now_millis() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must not precede the Unix epoch")
        .as_millis() as f64
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    use parking_lot::Mutex;
    use serde_json::{Value, json};

    use crate::{
        DynPluginSpec, PluginInitError, PluginSpec, Runtime,
        agent::{AgentDefinition, AgentInstance, AgentStatus},
        llm::{
            AssistantContentBlock, AssistantMessage, DoneReason, ErrorReason, ImageBlock,
            LlmService, LlmStartError, Message, ModelIdentity, Script, ScriptItem, ScriptedAdapter,
            StopReason, StreamChunk, TextBlock, ThinkingBlock, ThinkingLevel as LlmThinkingLevel,
            ToolCall, ToolResultContentBlock, ToolResultMessage, Usage, UserContent,
            UserContentBlock, UserMessage,
        },
        session::Session,
        tools::{AgentToolResult, ToolDefinition, ToolExecutionError, ToolExecutionRequest},
    };

    use super::{AgentLoop, PromptInput, ProviderTurnDisposition};
    use crate::agent_loop::{
        AgentEvent, AgentEventKind, AgentListenerError, AgentLoopError, register_agent_listener,
    };

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

    fn user(text: &str) -> Message {
        Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
    }

    fn assistant(text: &str) -> Message {
        Message::Assistant(Box::new(AssistantMessage::new(
            identity(),
            vec![AssistantContentBlock::Text(TextBlock::new(text))],
            Usage::default(),
            StopReason::Stop,
            2.0,
        )))
    }

    fn tool_result(text: &str) -> Message {
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "call-1",
            "lookup",
            vec![ToolResultContentBlock::Text(TextBlock::new(text))],
            false,
            3.0,
        )))
    }

    fn loop_for(runtime: &Runtime, session: Session) -> (AgentLoop, Arc<AgentInstance>) {
        loop_for_with_llm(runtime, session, Arc::new(LlmService::new()))
    }

    fn loop_for_with_llm(
        runtime: &Runtime,
        session: Session,
        llm: Arc<LlmService>,
    ) -> (AgentLoop, Arc<AgentInstance>) {
        let context = runtime.context();
        let agent = Arc::new(AgentInstance::new(
            "room-a",
            AgentDefinition::new("ada", "system", identity()),
            session,
            Some(context.clone()),
            None,
        ));
        let driver = AgentLoop::new(Arc::clone(&agent), context, llm);
        (driver, agent)
    }

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(
            name,
            format!("{name} description"),
            serde_json::from_value(json!({"type": "object"})).unwrap(),
            name,
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
        )
    }

    type EventHook = Arc<dyn Fn(AgentEvent) + Send + Sync>;

    fn listener_plugin(hook: EventHook) -> DynPluginSpec {
        PluginSpec::<Value>::new(
            "agent-loop-entry-listener",
            vec![],
            || json!({}),
            move |context, _config| {
                let hook = Arc::clone(&hook);
                async move {
                    register_agent_listener(&context, move |event| {
                        let hook = Arc::clone(&hook);
                        async move {
                            hook(event);
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

    fn now_millis() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as f64
    }

    #[test]
    fn active_entry_errors_keep_their_distinct_exact_messages() {
        run(async {
            let runtime = Runtime::new();
            let (driver, agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            agent.try_begin_run().unwrap();

            let prompt_error = driver
                .prepare_prompt_run(PromptInput::Message(user("later")))
                .await
                .unwrap_err();
            assert_eq!(
                prompt_error.to_string(),
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
            );

            let continue_error = driver.prepare_continue_run().await.unwrap_err();
            assert_eq!(
                continue_error.to_string(),
                "Agent is already processing. Wait for completion before continuing."
            );
            assert_eq!(agent.status(), AgentStatus::Running);
            agent.finish_run();
        });
    }

    #[test]
    fn continuation_validation_is_exact_and_does_not_start_a_run() {
        run(async {
            let runtime = Runtime::new();
            let (empty_driver, empty_agent) =
                loop_for(&runtime, Session::new("empty", [] as [&str; 0]).unwrap());
            empty_agent.set_error_message(Some("keep-empty".into()));
            let empty_error = empty_driver.prepare_continue_run().await.unwrap_err();
            assert_eq!(empty_error.to_string(), "No messages to continue from");
            assert_eq!(empty_agent.status(), AgentStatus::Idle);
            assert_eq!(empty_agent.error_message().as_deref(), Some("keep-empty"));

            let session = Session::new("assistant", [] as [&str; 0]).unwrap();
            session.append_message(assistant("done")).unwrap();
            let (assistant_driver, assistant_agent) = loop_for(&runtime, session);
            assistant_agent.set_error_message(Some("keep-assistant".into()));
            let assistant_error = assistant_driver.prepare_continue_run().await.unwrap_err();
            assert_eq!(
                assistant_error.to_string(),
                "Cannot continue from message role: assistant"
            );
            assert_eq!(assistant_agent.status(), AgentStatus::Idle);
            assert_eq!(
                assistant_agent.error_message().as_deref(),
                Some("keep-assistant")
            );
        });
    }

    #[test]
    fn prompt_accepts_every_typed_message_variant_without_normalizing_it() {
        run(async {
            let runtime = Runtime::new();
            let (driver, agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            let entering = vec![user("question"), assistant("prior"), tool_result("answer")];

            let prepared = driver
                .prepare_prompt_run(PromptInput::Messages(entering.clone()))
                .await
                .unwrap();

            assert_eq!(prepared.context.messages, entering);
            assert_eq!(prepared.new_messages, prepared.context.messages);
            assert_eq!(agent.messages().unwrap(), prepared.context.messages);
            assert_eq!(agent.status(), AgentStatus::Running);
            drop(prepared);
            assert_eq!(agent.status(), AgentStatus::Idle);
        });
    }

    #[test]
    fn text_and_images_normalize_to_one_user_message_in_exact_order() {
        run(async {
            let runtime = Runtime::new();
            let (driver, _agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            let first = ImageBlock::data("image/png", "first");
            let second = ImageBlock::data("image/jpeg", "second");
            let before = now_millis();

            let prepared = driver
                .prepare_prompt_run(PromptInput::Text {
                    text: "describe".into(),
                    images: vec![first.clone(), second.clone()],
                })
                .await
                .unwrap();
            let after = now_millis();

            let Message::User(message) = &prepared.new_messages[0] else {
                panic!("text input must normalize to a user message")
            };
            assert_eq!(
                message.content,
                UserContent::Blocks(vec![
                    UserContentBlock::Text(TextBlock::new("describe")),
                    UserContentBlock::Image(first),
                    UserContentBlock::Image(second),
                ])
            );
            assert!((before..=after).contains(&message.timestamp));
            assert_eq!(prepared.new_messages.len(), 1);
        });
    }

    #[test]
    fn prompt_is_reduced_before_dispatch_and_completed_before_steering_is_claimed() {
        run(async {
            let runtime = Runtime::new();
            let events = Arc::new(Mutex::new(Vec::new()));
            let agent_slot = Arc::new(Mutex::new(None::<Arc<AgentInstance>>));
            let prompt = user("prompt");
            let steering = user("steering");
            let observed_events = Arc::clone(&events);
            let observed_agent = Arc::clone(&agent_slot);
            let prompt_for_listener = prompt.clone();
            let steering_for_listener = steering.clone();
            let plugin = listener_plugin(Arc::new(move |event| {
                let agent = observed_agent.lock().clone().unwrap();
                match &event {
                    AgentEvent::MessageStart(message) => {
                        assert_eq!(agent.streaming_message().as_ref(), Some(message));
                        assert!(!agent.messages().unwrap().contains(message));
                    }
                    AgentEvent::MessageEnd(message) => {
                        assert_eq!(agent.streaming_message(), None);
                        assert!(agent.messages().unwrap().contains(message));
                        if message == &prompt_for_listener {
                            agent.steer(steering_for_listener.clone(), None);
                        }
                    }
                    _ => {}
                }
                observed_events.lock().push(event);
            }));
            runtime.mount(&plugin, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            let (driver, agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            *agent_slot.lock() = Some(Arc::clone(&agent));

            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(prompt.clone()))
                .await
                .unwrap();

            assert_eq!(
                prepared.context.messages,
                vec![prompt.clone(), steering.clone()]
            );
            assert_eq!(
                prepared.new_messages,
                vec![prompt.clone(), steering.clone()]
            );
            assert_eq!(agent.messages().unwrap(), vec![prompt, steering]);
            assert_eq!(
                events
                    .lock()
                    .iter()
                    .map(AgentEvent::kind)
                    .collect::<Vec<_>>(),
                vec![
                    AgentEventKind::AgentStart,
                    AgentEventKind::TurnStart,
                    AgentEventKind::MessageStart,
                    AgentEventKind::MessageEnd,
                    AgentEventKind::MessageStart,
                    AgentEventKind::MessageEnd,
                ]
            );
        });
    }

    #[test]
    fn assistant_last_steering_pre_drain_is_admitted_once_without_a_second_claim() {
        run(async {
            let runtime = Runtime::new();
            let session = Session::new("room-a", [] as [&str; 0]).unwrap();
            let history = assistant("done");
            session.append_message(history.clone()).unwrap();
            let (driver, agent) = loop_for(&runtime, session);
            let first = user("first");
            let second = user("second");
            agent.steer(first.clone(), None);
            agent.steer(second.clone(), None);

            let prepared = driver.prepare_continue_run().await.unwrap();

            assert_eq!(prepared.context.messages, vec![history, first.clone()]);
            assert_eq!(prepared.new_messages, vec![first]);
            assert_eq!(
                agent
                    .inbox()
                    .claim(
                        crate::agent::InboxTarget::Steering,
                        crate::agent::ClaimPolicy::OneAtATime,
                    )
                    .into_iter()
                    .map(|envelope| envelope.message)
                    .collect::<Vec<_>>(),
                vec![second]
            );
        });
    }

    #[test]
    fn assistant_last_follow_up_pre_drain_still_performs_the_initial_steering_poll() {
        run(async {
            let runtime = Runtime::new();
            let agent_slot = Arc::new(Mutex::new(None::<Arc<AgentInstance>>));
            let follow_up = user("follow-up");
            let steering = user("steering");
            let observed_agent = Arc::clone(&agent_slot);
            let follow_up_for_listener = follow_up.clone();
            let steering_for_listener = steering.clone();
            let plugin = listener_plugin(Arc::new(move |event| {
                if matches!(&event, AgentEvent::MessageEnd(message) if message == &follow_up_for_listener)
                {
                    observed_agent
                        .lock()
                        .as_ref()
                        .unwrap()
                        .steer(steering_for_listener.clone(), None);
                }
            }));
            runtime.mount(&plugin, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            let session = Session::new("room-a", [] as [&str; 0]).unwrap();
            let history = assistant("done");
            session.append_message(history.clone()).unwrap();
            let (driver, agent) = loop_for(&runtime, session);
            *agent_slot.lock() = Some(Arc::clone(&agent));
            agent.follow_up(follow_up.clone(), None);

            let prepared = driver.prepare_continue_run().await.unwrap();

            assert_eq!(
                prepared.context.messages,
                vec![history, follow_up.clone(), steering.clone()]
            );
            assert_eq!(prepared.new_messages, vec![follow_up, steering]);
            assert!(!agent.has_queued_messages());
        });
    }

    #[test]
    fn plain_continuation_keeps_history_out_of_the_run_accumulator() {
        run(async {
            let runtime = Runtime::new();
            let session = Session::new("room-a", [] as [&str; 0]).unwrap();
            let history = user("history");
            session.append_message(history.clone()).unwrap();
            let (driver, agent) = loop_for(&runtime, session);
            let steering = tool_result("ambient");
            agent.steer(steering.clone(), None);

            let prepared = driver.prepare_continue_run().await.unwrap();

            assert_eq!(prepared.context.messages, vec![history, steering.clone()]);
            assert_eq!(prepared.new_messages, vec![steering]);
        });
    }

    #[test]
    fn provider_turn_uses_the_prepared_snapshot_and_forwards_every_complete_partial() {
        run(async {
            let runtime = Runtime::new();
            let first_tool = runtime
                .tools()
                .register_for_scope(None, tool("first"))
                .unwrap();
            let start = AssistantMessage::pending(identity(), 10.0);
            let mut text_partial = start.clone();
            text_partial.response_id = Some("response-1".into());
            text_partial.content = vec![AssistantContentBlock::Text(
                TextBlock::new("answer").with_signature("text-signature"),
            )];
            let mut thinking_partial = text_partial.clone();
            thinking_partial
                .content
                .push(AssistantContentBlock::Thinking(
                    ThinkingBlock::new("reason").with_signature("thinking-signature"),
                ));
            let mut tool_call = ToolCall::new(
                "call-1",
                "first",
                BTreeMap::from([("query".into(), json!("rust"))]),
            )
            .with_namespace("fixture");
            tool_call.thought_signature = Some("tool-signature".into());
            let mut tool_partial = thinking_partial.clone();
            tool_partial
                .content
                .push(AssistantContentBlock::ToolCall(tool_call.clone()));
            tool_partial.response_model = Some("provider-model".into());
            tool_partial.raw_stop_reason = Some("tool_calls".into());
            tool_partial.end_turn = Some(false);
            let mut final_message = tool_partial.clone();
            final_message.stop_reason = StopReason::ToolUse;
            final_message.usage.input = 11;
            final_message.usage.output = 7;
            final_message.usage.total_tokens = 18;
            let updates = [
                StreamChunk::TextDelta {
                    content_index: 0,
                    delta: "answer".into(),
                    partial: text_partial.clone(),
                },
                StreamChunk::ThinkingDelta {
                    content_index: 1,
                    delta: "reason".into(),
                    partial: thinking_partial.clone(),
                },
                StreamChunk::ToolCallEnd {
                    content_index: 2,
                    tool_call,
                    partial: tool_partial.clone(),
                },
            ];
            let script = Script::new(
                [ScriptItem::Chunk(Box::new(StreamChunk::Start {
                    partial: start.clone(),
                }))]
                .into_iter()
                .chain(
                    updates
                        .iter()
                        .cloned()
                        .map(|chunk| ScriptItem::Chunk(Box::new(chunk))),
                )
                .chain([ScriptItem::Chunk(Box::new(StreamChunk::Done {
                    reason: DoneReason::ToolUse,
                    message: final_message.clone(),
                }))]),
            );
            let adapter = Arc::new(ScriptedAdapter::new([script]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let events = Arc::new(Mutex::new(Vec::new()));
            let reductions = Arc::new(Mutex::new(Vec::new()));
            let agent_slot = Arc::new(Mutex::new(None::<Arc<AgentInstance>>));
            let observed_events = Arc::clone(&events);
            let observed_reductions = Arc::clone(&reductions);
            let observed_agent = Arc::clone(&agent_slot);
            let plugin = listener_plugin(Arc::new(move |event| {
                let agent = observed_agent.lock().clone().unwrap();
                observed_reductions.lock().push((
                    event.kind(),
                    agent.streaming_message(),
                    agent.messages().unwrap(),
                ));
                observed_events.lock().push(event);
            }));
            runtime.mount(&plugin, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            *agent_slot.lock() = Some(Arc::clone(&agent));
            agent.set_thinking_level(crate::agent::ThinkingLevel::High);
            let prompt = user("prompt");
            let mut prepared = driver
                .prepare_prompt_run(PromptInput::Message(prompt.clone()))
                .await
                .unwrap();

            agent.set_system_prompt("changed-after-entry");
            agent.set_model(ModelIdentity::new("provider", "api", "changed").unwrap());
            agent.set_thinking_level(crate::agent::ThinkingLevel::Off);
            agent
                .session()
                .append_message(user("external-late"))
                .unwrap();
            let second_tool = runtime
                .tools()
                .register_for_scope(None, tool("second"))
                .unwrap();

            let turn = driver.run_provider_turn(&mut prepared).await.unwrap();

            assert_eq!(turn.message, final_message);
            assert_eq!(turn.disposition, ProviderTurnDisposition::Continue);
            assert_eq!(
                prepared.context.messages,
                vec![
                    prompt.clone(),
                    Message::Assistant(Box::new(final_message.clone()))
                ]
            );
            assert_eq!(
                prepared.new_messages,
                vec![
                    prompt.clone(),
                    Message::Assistant(Box::new(final_message.clone()))
                ]
            );
            let requests = adapter.requests();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].model, identity());
            assert_eq!(requests[0].context.system_prompt.as_deref(), Some("system"));
            assert_eq!(requests[0].context.messages, vec![prompt.clone()]);
            assert_eq!(
                requests[0]
                    .context
                    .tools
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|schema| schema.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["first"]
            );
            assert_eq!(requests[0].options.reasoning, Some(LlmThinkingLevel::High));

            let events = events.lock();
            assert!(
                matches!(&events[4], AgentEvent::MessageStart(message) if message == &Message::Assistant(Box::new(start.clone())))
            );
            for (offset, expected) in updates.iter().enumerate() {
                assert!(matches!(
                    &events[5 + offset],
                    AgentEvent::MessageUpdate { event, partial }
                        if event == expected && partial == expected.partial()
                ));
            }
            assert!(
                matches!(&events[8], AgentEvent::MessageEnd(message) if message == &Message::Assistant(Box::new(final_message.clone())))
            );
            drop(events);

            let reductions = reductions.lock();
            assert_eq!(reductions[4].1, Some(Message::Assistant(Box::new(start))));
            assert_eq!(reductions[4].2, vec![prompt.clone(), user("external-late")]);
            assert_eq!(
                reductions[5].1,
                Some(Message::Assistant(Box::new(text_partial)))
            );
            assert_eq!(
                reductions[6].1,
                Some(Message::Assistant(Box::new(thinking_partial)))
            );
            assert_eq!(
                reductions[7].1,
                Some(Message::Assistant(Box::new(tool_partial)))
            );
            assert_eq!(reductions[8].1, None);
            assert_eq!(
                reductions[8].2,
                vec![
                    prompt,
                    user("external-late"),
                    Message::Assistant(Box::new(final_message))
                ]
            );
            drop(reductions);
            assert_eq!(agent.status(), AgentStatus::Running);
            drop(prepared);
            assert_eq!(agent.status(), AgentStatus::Idle);
            drop((first_tool, second_tool));
        });
    }

    #[test]
    fn represented_error_and_aborted_results_are_classified_without_post_turn_work() {
        run(async {
            for (reason, error_reason) in [
                (StopReason::Error, ErrorReason::Error),
                (StopReason::Aborted, ErrorReason::Aborted),
            ] {
                let runtime = Runtime::new();
                let mut final_message = AssistantMessage::pending(identity(), 4.0);
                final_message.stop_reason = reason;
                final_message.error_message = Some(format!("{reason:?}"));
                let adapter = Arc::new(ScriptedAdapter::new([Script::new([ScriptItem::Chunk(
                    Box::new(StreamChunk::Error {
                        reason: error_reason,
                        error: final_message.clone(),
                    }),
                )])]));
                let llm = Arc::new(LlmService::new());
                llm.register(identity(), adapter);
                let (driver, agent) = loop_for_with_llm(
                    &runtime,
                    Session::new("room-a", [] as [&str; 0]).unwrap(),
                    llm,
                );
                let mut prepared = driver
                    .prepare_prompt_run(PromptInput::Message(user("prompt")))
                    .await
                    .unwrap();

                let turn = driver.run_provider_turn(&mut prepared).await.unwrap();

                assert_eq!(turn.message, final_message);
                assert_eq!(
                    turn.disposition,
                    ProviderTurnDisposition::RepresentedTerminal
                );
                assert_eq!(prepared.new_messages.len(), 2);
                assert_eq!(agent.messages().unwrap().len(), 2);
                assert_eq!(agent.status(), AgentStatus::Running);
                drop(prepared);
                assert_eq!(agent.status(), AgentStatus::Idle);
            }
        });
    }

    #[test]
    fn eager_llm_start_errors_remain_typed_and_settle_without_assistant_events() {
        run(async {
            for registered_but_exhausted in [false, true] {
                let runtime = Runtime::new();
                let llm = Arc::new(LlmService::new());
                if registered_but_exhausted {
                    llm.register(identity(), Arc::new(ScriptedAdapter::new([])));
                }
                let (driver, agent) = loop_for_with_llm(
                    &runtime,
                    Session::new("room-a", [] as [&str; 0]).unwrap(),
                    llm,
                );
                let prompt = user("prompt");
                let mut prepared = driver
                    .prepare_prompt_run(PromptInput::Message(prompt.clone()))
                    .await
                    .unwrap();

                let error = driver.run_provider_turn(&mut prepared).await.unwrap_err();

                if registered_but_exhausted {
                    assert!(matches!(
                        error,
                        AgentLoopError::LlmStart(LlmStartError::AdapterStart(_))
                    ));
                } else {
                    assert!(matches!(
                        error,
                        AgentLoopError::LlmStart(LlmStartError::UnknownModel { .. })
                    ));
                }
                assert_eq!(agent.messages().unwrap(), vec![prompt]);
                assert_eq!(prepared.new_messages.len(), 1);
                assert_eq!(agent.status(), AgentStatus::Running);
                drop(prepared);
                assert_eq!(agent.status(), AgentStatus::Idle);
            }
        });
    }

    #[test]
    fn tool_events_are_live_and_pending_state_spans_the_update_window() {
        run(async {
            let runtime = Runtime::new();
            let trace = Arc::new(Mutex::new(Vec::new()));
            let agent_slot = Arc::new(Mutex::new(None::<Arc<AgentInstance>>));
            runtime
                .mount(
                    &listener_plugin({
                        let trace = Arc::clone(&trace);
                        let agent_slot = Arc::clone(&agent_slot);
                        Arc::new(move |event| {
                            let agent = agent_slot.lock().as_ref().unwrap().clone();
                            match event {
                                AgentEvent::ToolExecutionStart(start) => {
                                    assert!(
                                        agent.pending_tool_calls().contains(&start.tool_call_id)
                                    );
                                    trace.lock().push("start");
                                }
                                AgentEvent::ToolExecutionUpdate(update) => {
                                    assert_eq!(update.update.details, json!({"progress": 1}));
                                    assert!(
                                        agent.pending_tool_calls().contains(&update.tool_call_id)
                                    );
                                    trace.lock().push("update");
                                }
                                AgentEvent::ToolExecutionEnd(end) => {
                                    assert!(
                                        !agent.pending_tool_calls().contains(&end.tool_call_id)
                                    );
                                    trace.lock().push("end");
                                }
                                _ => {}
                            }
                        })
                    }),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();
            let executions = Arc::new(AtomicUsize::new(0));
            let registration = runtime
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
                            let executions = Arc::clone(&executions);
                            move |request: ToolExecutionRequest| {
                                let trace = Arc::clone(&trace);
                                let executions = Arc::clone(&executions);
                                Box::pin(async move {
                                    executions.fetch_add(1, Ordering::SeqCst);
                                    request.on_update.unwrap()(AgentToolResult {
                                        content: vec![ToolResultContentBlock::Text(
                                            TextBlock::new("partial"),
                                        )],
                                        details: json!({"progress": 1}),
                                        usage: None,
                                        added_tool_names: None,
                                        terminate: None,
                                    });
                                    trace.lock().push("tool-continued");
                                    Ok(AgentToolResult {
                                        content: vec![ToolResultContentBlock::Text(
                                            TextBlock::new("done"),
                                        )],
                                        details: Value::Null,
                                        usage: None,
                                        added_tool_names: None,
                                        terminate: None,
                                    })
                                })
                            }
                        },
                    ),
                )
                .unwrap();
            let (driver, agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            *agent_slot.lock() = Some(Arc::clone(&agent));
            let mut prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            trace.lock().clear();
            let assistant = AssistantMessage::new(
                identity(),
                vec![AssistantContentBlock::ToolCall(ToolCall::new(
                    "call-1",
                    "chatty",
                    BTreeMap::new(),
                ))],
                Usage::default(),
                StopReason::ToolUse,
                2.0,
            );

            let batch = driver
                .run_tool_calls(&mut prepared, &assistant)
                .await
                .unwrap();

            assert_eq!(executions.load(Ordering::SeqCst), 1);
            assert_eq!(
                trace.lock().as_slice(),
                ["start", "update", "tool-continued", "end"]
            );
            assert!(agent.pending_tool_calls().is_empty());
            assert_eq!(batch.messages.len(), 1);
            assert!(matches!(
                prepared.new_messages.last(),
                Some(Message::ToolResult(_))
            ));
            drop((prepared, registration));
        });
    }

    #[test]
    fn tool_listener_failure_preserves_the_raw_listener_message() {
        run(async {
            let runtime = Runtime::new();
            let plugin = PluginSpec::<Value>::new(
                "failing-tool-listener",
                vec![],
                || json!({}),
                |context, _config| async move {
                    register_agent_listener(&context, |event| async move {
                        match event {
                            AgentEvent::ToolExecutionStart(_) => {
                                Err(AgentListenerError::new("boom"))
                            }
                            _ => Ok(()),
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                },
            )
            .erase();
            runtime.mount(&plugin, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            let executions = Arc::new(AtomicUsize::new(0));
            let registration = runtime
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
                                Box::pin(async {
                                    Ok(AgentToolResult {
                                        content: vec![],
                                        details: Value::Null,
                                        usage: None,
                                        added_tool_names: None,
                                        terminate: None,
                                    })
                                })
                            }
                        },
                    ),
                )
                .unwrap();
            let (driver, _agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            let mut prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            let assistant = AssistantMessage::new(
                identity(),
                vec![AssistantContentBlock::ToolCall(ToolCall::new(
                    "call-1",
                    "guarded",
                    BTreeMap::new(),
                ))],
                Usage::default(),
                StopReason::ToolUse,
                2.0,
            );

            let error = driver
                .run_tool_calls(&mut prepared, &assistant)
                .await
                .unwrap_err();

            assert!(matches!(
                &error,
                AgentLoopError::ToolExecution(ToolExecutionError::Lifecycle(error))
                    if error.message() == "boom"
            ));
            assert_eq!(error.to_string(), "boom");
            assert_eq!(executions.load(Ordering::SeqCst), 0);
            drop((prepared, registration));
        });
    }

    #[test]
    fn added_tool_names_extend_only_the_run_local_snapshot_in_order() {
        run(async {
            let runtime = Runtime::new();
            let initial = runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "initial",
                        "initial",
                        serde_json::from_value(json!({})).unwrap(),
                        "initial",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async {
                                Ok(AgentToolResult {
                                    content: vec![],
                                    details: Value::Null,
                                    usage: None,
                                    added_tool_names: Some(vec![
                                        "introduced".into(),
                                        "initial".into(),
                                        "missing".into(),
                                        "introduced".into(),
                                    ]),
                                    terminate: None,
                                })
                            })
                        },
                    ),
                )
                .unwrap();
            let (driver, agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            let mut prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            assert_eq!(
                prepared
                    .context
                    .tools
                    .iter()
                    .map(|tool| tool.name())
                    .collect::<Vec<_>>(),
                vec!["initial"]
            );
            let introduced = runtime
                .tools()
                .register_for_scope(None, tool("introduced"))
                .unwrap();
            let unrelated = runtime
                .tools()
                .register_for_scope(None, tool("unrelated"))
                .unwrap();
            let assistant = AssistantMessage::new(
                identity(),
                vec![AssistantContentBlock::ToolCall(ToolCall::new(
                    "call-1",
                    "initial",
                    BTreeMap::new(),
                ))],
                Usage::default(),
                StopReason::ToolUse,
                2.0,
            );

            driver
                .run_tool_calls(&mut prepared, &assistant)
                .await
                .unwrap();

            assert_eq!(
                prepared
                    .context
                    .tools
                    .iter()
                    .map(|tool| tool.name())
                    .collect::<Vec<_>>(),
                vec!["initial", "introduced"]
            );
            assert_eq!(
                agent
                    .tools()
                    .iter()
                    .map(|tool| tool.name())
                    .collect::<Vec<_>>(),
                vec!["initial", "introduced", "unrelated"]
            );
            drop((prepared, initial, introduced, unrelated));
        });
    }

    #[test]
    fn tool_execution_uses_the_run_snapshot_after_live_registry_replacement() {
        run(async {
            let runtime = Runtime::new();
            let old_executions = Arc::new(AtomicUsize::new(0));
            let old = runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "versioned",
                        "old",
                        serde_json::from_value(json!({})).unwrap(),
                        "old",
                        {
                            let old_executions = Arc::clone(&old_executions);
                            move |_request: ToolExecutionRequest| {
                                let old_executions = Arc::clone(&old_executions);
                                Box::pin(async move {
                                    old_executions.fetch_add(1, Ordering::SeqCst);
                                    Ok(AgentToolResult {
                                        content: vec![ToolResultContentBlock::Text(
                                            TextBlock::new("old"),
                                        )],
                                        details: Value::Null,
                                        usage: None,
                                        added_tool_names: None,
                                        terminate: None,
                                    })
                                })
                            }
                        },
                    ),
                )
                .unwrap();
            let (driver, _agent) =
                loop_for(&runtime, Session::new("room-a", [] as [&str; 0]).unwrap());
            let mut prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            old.withdraw();
            let new_executions = Arc::new(AtomicUsize::new(0));
            let replacement = runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "versioned",
                        "new",
                        serde_json::from_value(json!({})).unwrap(),
                        "new",
                        {
                            let new_executions = Arc::clone(&new_executions);
                            move |_request: ToolExecutionRequest| {
                                let new_executions = Arc::clone(&new_executions);
                                Box::pin(async move {
                                    new_executions.fetch_add(1, Ordering::SeqCst);
                                    Ok(AgentToolResult {
                                        content: vec![ToolResultContentBlock::Text(
                                            TextBlock::new("new"),
                                        )],
                                        details: Value::Null,
                                        usage: None,
                                        added_tool_names: None,
                                        terminate: None,
                                    })
                                })
                            }
                        },
                    ),
                )
                .unwrap();
            let assistant = AssistantMessage::new(
                identity(),
                vec![AssistantContentBlock::ToolCall(ToolCall::new(
                    "call-1",
                    "versioned",
                    BTreeMap::new(),
                ))],
                Usage::default(),
                StopReason::ToolUse,
                2.0,
            );

            let batch = driver
                .run_tool_calls(&mut prepared, &assistant)
                .await
                .unwrap();

            assert_eq!(old_executions.load(Ordering::SeqCst), 1);
            assert_eq!(new_executions.load(Ordering::SeqCst), 0);
            assert!(matches!(
                batch.messages[0].content.as_slice(),
                [ToolResultContentBlock::Text(block)] if block.text == "old"
            ));
            drop((prepared, replacement));
        });
    }
}
