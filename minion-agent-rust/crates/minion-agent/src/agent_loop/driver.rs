// Task 4 deliberately stages these private transactions before Task 5 wires
// their only production caller (the complete public provider-backed entries).
#![allow(dead_code)]

use std::{
    fmt,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    Context,
    agent::{AgentInstance, AgentRunError, AgentStatus, ClaimPolicy, InboxTarget},
    llm::{ImageBlock, LlmService, Message, TextBlock, UserContent, UserContentBlock, UserMessage},
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
    _llm: Arc<LlmService>,
}

struct PreparedRun {
    agent: Arc<AgentInstance>,
    context: RunContext,
    config: RunConfig,
    new_messages: Vec<Message>,
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
            _llm: llm,
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
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use parking_lot::Mutex;
    use serde_json::{Value, json};

    use crate::{
        DynPluginSpec, PluginInitError, PluginSpec, Runtime,
        agent::{AgentDefinition, AgentInstance, AgentStatus},
        llm::{
            AssistantContentBlock, AssistantMessage, ImageBlock, LlmService, Message,
            ModelIdentity, StopReason, TextBlock, ToolResultContentBlock, ToolResultMessage, Usage,
            UserContent, UserContentBlock, UserMessage,
        },
        session::Session,
    };

    use super::{AgentLoop, PromptInput};
    use crate::agent_loop::{AgentEvent, AgentEventKind, register_agent_listener};

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
        let context = runtime.context();
        let agent = Arc::new(AgentInstance::new(
            "room-a",
            AgentDefinition::new("ada", "system", identity()),
            session,
            Some(context.clone()),
            None,
        ));
        let driver = AgentLoop::new(Arc::clone(&agent), context, Arc::new(LlmService::new()));
        (driver, agent)
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
}
