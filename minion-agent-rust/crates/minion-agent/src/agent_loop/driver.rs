use std::{
    fmt,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use futures::{StreamExt, future::BoxFuture};

use crate::{
    Context,
    agent::{
        AgentInstance, AgentRunError, AgentStatus, ClaimPolicy, InboxTarget, InputEnvelope,
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

use super::decisions::{pre_step, prepare_next_turn, should_stop_after_turn, transform_context};
use super::events::dispatch_agent_event_with_signal;
use super::{
    AgentEndReason, AgentEvent, AgentLoopError, Enter, PreStepContext, PreStepDecision,
    PreStepReason, PrepareNextTurnContext, RunCause, RunConfig, RunContext, RunSnapshot,
    ShouldStopAfterTurnContext, TransformContext, reduce_event,
};

/// Input accepted by the public prompt entry point.
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
    next_step_policy: ClaimPolicy,
    next_turn_policy: ClaimPolicy,
}

struct PreparedRun {
    agent: Arc<AgentInstance>,
    context: RunContext,
    config: RunConfig,
    new_messages: Vec<Message>,
    causes: Vec<RunCause>,
    decision: Option<Enter>,
    signal: crate::runtime::RunSignal,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TurnBoundary {
    AlreadyOpen,
    NeedsStart,
}

#[derive(Clone, Debug, PartialEq)]
enum LoopDecision {
    Continue(Enter),
    Exhausted,
    Stop,
}

impl fmt::Debug for PreparedRun {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedRun")
            .field("context", &self.context)
            .field("config", &self.config)
            .field("new_messages", &self.new_messages)
            .field("causes", &self.causes)
            .field("decision", &self.decision)
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

impl Drop for PreparedRun {
    fn drop(&mut self) {
        self.agent.finish_run();
    }
}

impl PreparedRun {
    fn new(agent: Arc<AgentInstance>, snapshot: RunSnapshot, causes: Vec<RunCause>) -> Self {
        Self {
            agent,
            context: snapshot.context,
            config: snapshot.config,
            new_messages: Vec::new(),
            causes,
            decision: None,
            signal: snapshot.signal,
        }
    }
}

impl AgentLoop {
    pub fn new(agent: Arc<AgentInstance>, context: Context, llm: Arc<LlmService>) -> Self {
        Self {
            agent,
            context,
            llm,
            next_step_policy: ClaimPolicy::OneAtATime,
            next_turn_policy: ClaimPolicy::OneAtATime,
        }
    }

    pub fn set_next_step_policy(&mut self, policy: ClaimPolicy) {
        self.next_step_policy = policy;
    }

    pub fn set_next_turn_policy(&mut self, policy: ClaimPolicy) {
        self.next_turn_policy = policy;
    }

    /// Starts a complete Pi-equivalent run from typed or convenience input.
    pub async fn prompt(&self, input: PromptInput) -> Result<Vec<Message>, AgentLoopError> {
        if self.agent.status() != AgentStatus::Idle {
            return Err(AgentLoopError::PromptActive);
        }
        let entering = normalize_prompt_input(input);
        let snapshot = self.agent.try_begin_run().map_err(map_prompt_entry_error)?;
        self.run_wrapped(
            PreparedRun::new(Arc::clone(&self.agent), snapshot, Vec::new()),
            entering,
            false,
        )
        .await
    }

    /// Continues a complete Pi-equivalent run from the current transcript.
    pub async fn continue_run(&self) -> Result<Vec<Message>, AgentLoopError> {
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
            return self
                .run_wrapped(
                    PreparedRun::new(Arc::clone(&self.agent), snapshot, Vec::new()),
                    Vec::new(),
                    false,
                )
                .await;
        }

        let steering = self.claim(InboxTarget::Steering);
        if !steering.is_empty() {
            let causes = envelope_causes(&steering);
            return self
                .run_wrapped(
                    PreparedRun::new(Arc::clone(&self.agent), snapshot, causes),
                    envelope_messages(&steering),
                    true,
                )
                .await;
        }
        let follow_up = self.claim(InboxTarget::FollowUp);
        if !follow_up.is_empty() {
            let causes = envelope_causes(&follow_up);
            return self
                .run_wrapped(
                    PreparedRun::new(Arc::clone(&self.agent), snapshot, causes),
                    envelope_messages(&follow_up),
                    false,
                )
                .await;
        }

        // Another claimant may have drained the observed queue between the
        // validation and this run's atomic entry.
        self.agent.finish_run();
        Err(AgentLoopError::CannotContinueFromAssistant)
    }

    /// Runs queued follow-up input until the next-turn queue is empty.
    ///
    /// This Minion host pump opens one complete run for each batch it must
    /// claim. A run may itself consume later follow-ups before returning.
    pub async fn run_until_idle(&self) -> Result<(), AgentLoopError> {
        while !self.agent.inbox().pending(InboxTarget::FollowUp).is_empty() {
            let claimed = self.claim(InboxTarget::FollowUp);
            if claimed.is_empty() {
                continue;
            }
            let causes = envelope_causes(&claimed);
            let entering = envelope_messages(&claimed);
            let snapshot = self.agent.try_begin_run()?;
            self.run_wrapped(
                PreparedRun::new(Arc::clone(&self.agent), snapshot, causes),
                entering,
                false,
            )
            .await?;
        }
        self.agent.inbox().take_wake();
        Ok(())
    }

    #[cfg(test)]
    async fn prepare_prompt_run(&self, input: PromptInput) -> Result<PreparedRun, AgentLoopError> {
        if self.agent.status() != AgentStatus::Idle {
            return Err(AgentLoopError::PromptActive);
        }
        let entering = normalize_prompt_input(input);
        let snapshot = self.agent.try_begin_run().map_err(map_prompt_entry_error)?;
        self.prepare_first_turn(snapshot, entering, false, Vec::new())
            .await
    }

    #[cfg(test)]
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
            return self
                .prepare_first_turn(snapshot, Vec::new(), false, Vec::new())
                .await;
        }

        let steering = self.claim(InboxTarget::Steering);
        if !steering.is_empty() {
            let causes = envelope_causes(&steering);
            return self
                .prepare_first_turn(snapshot, envelope_messages(&steering), true, causes)
                .await;
        }
        let follow_up = self.claim(InboxTarget::FollowUp);
        if !follow_up.is_empty() {
            let causes = envelope_causes(&follow_up);
            return self
                .prepare_first_turn(snapshot, envelope_messages(&follow_up), false, causes)
                .await;
        }

        // Another claimant may have drained the observed queue between the
        // validation and this run's atomic entry.
        self.agent.finish_run();
        Err(AgentLoopError::CannotContinueFromAssistant)
    }

    #[cfg(test)]
    async fn prepare_first_turn(
        &self,
        snapshot: RunSnapshot,
        entering: Vec<Message>,
        skip_initial_steering_poll: bool,
        causes: Vec<RunCause>,
    ) -> Result<PreparedRun, AgentLoopError> {
        let mut prepared = PreparedRun::new(Arc::clone(&self.agent), snapshot, causes);
        self.open_first_turn(&mut prepared, entering, skip_initial_steering_poll)
            .await?;
        Ok(prepared)
    }

    async fn open_first_turn(
        &self,
        prepared: &mut PreparedRun,
        entering: Vec<Message>,
        skip_initial_steering_poll: bool,
    ) -> Result<(), AgentLoopError> {
        self.dispatch(AgentEvent::AgentStart {
            causes: prepared.causes.clone(),
        })
        .await?;
        self.dispatch(AgentEvent::TurnStart).await?;
        let decision = self.decide(entering, PreStepReason::Initial).await?;
        let PreStepDecision::Enter(mut decision) = decision else {
            return Ok(());
        };
        self.admit(prepared, decision.messages.clone()).await?;

        if !skip_initial_steering_poll {
            let steering = self.claim(InboxTarget::Steering);
            if !steering.is_empty() {
                let steering_decision = self
                    .decide(envelope_messages(&steering), PreStepReason::Steering)
                    .await?;
                let PreStepDecision::Enter(next) = steering_decision else {
                    return Ok(());
                };
                self.admit(prepared, next.messages.clone()).await?;
                decision = next;
            }
        }
        prepared.decision = Some(decision);
        Ok(())
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

    #[cfg(test)]
    async fn run_provider_turn(
        &self,
        prepared: &mut PreparedRun,
    ) -> Result<ProviderTurn, AgentLoopError> {
        let decision = prepared.decision.clone().unwrap_or_else(|| Enter {
            messages: Vec::new(),
            system_override: None,
            history_window: None,
        });
        self.run_provider_turn_with_decision(prepared, &decision)
            .await
    }

    async fn run_provider_turn_with_decision(
        &self,
        prepared: &mut PreparedRun,
        decision: &Enter,
    ) -> Result<ProviderTurn, AgentLoopError> {
        let first_visible = decision.history_window.map_or(0, |window| {
            prepared.context.messages.len().saturating_sub(window)
        });
        let request_messages = transform_context(
            &self.context,
            TransformContext {
                agent: Arc::clone(&prepared.agent),
                messages: prepared.context.messages[first_visible..].to_vec(),
                signal: prepared.signal.clone(),
            },
        )
        .await?;
        let request = LlmRequest {
            model: prepared.config.model.clone(),
            context: LlmContext {
                system_prompt: Some(
                    decision
                        .system_override
                        .clone()
                        .unwrap_or_else(|| prepared.context.system_prompt.clone()),
                ),
                messages: request_messages,
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
            signal: Some(prepared.signal.clone()),
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
            .with_signal(Arc::new(prepared.signal.clone()))
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

    #[cfg(test)]
    async fn run_successful(
        &self,
        mut prepared: PreparedRun,
    ) -> Result<Vec<Message>, AgentLoopError> {
        self.run_successful_inner(&mut prepared).await
    }

    async fn run_successful_inner(
        &self,
        prepared: &mut PreparedRun,
    ) -> Result<Vec<Message>, AgentLoopError> {
        let Some(mut decision) = prepared.decision.take() else {
            return self
                .finish_successful_run(prepared, AgentEndReason::Rejected)
                .await;
        };
        let mut boundary = TurnBoundary::AlreadyOpen;
        let mut last_terminated: bool;
        loop {
            loop {
                if boundary == TurnBoundary::NeedsStart {
                    self.dispatch(AgentEvent::TurnStart).await?;
                    self.admit(prepared, decision.messages.clone()).await?;
                }
                boundary = TurnBoundary::NeedsStart;

                let turn = self
                    .run_provider_turn_with_decision(prepared, &decision)
                    .await?;
                if turn.disposition == ProviderTurnDisposition::RepresentedTerminal {
                    let reason = match turn.message.stop_reason {
                        StopReason::Error => AgentEndReason::Error,
                        StopReason::Aborted => AgentEndReason::Aborted,
                        StopReason::Pending
                        | StopReason::Stop
                        | StopReason::Length
                        | StopReason::ToolUse
                        | StopReason::Deferred => unreachable!("represented terminal is exact"),
                    };
                    self.dispatch(AgentEvent::TurnEnd {
                        message: turn.message,
                        tool_results: Vec::new(),
                    })
                    .await?;
                    return self.finish_successful_run(prepared, reason).await;
                }

                let has_tool_calls = turn
                    .message
                    .content
                    .iter()
                    .any(|block| matches!(block, AssistantContentBlock::ToolCall(_)));
                let batch = if has_tool_calls {
                    self.run_tool_calls(prepared, &turn.message).await?
                } else {
                    ToolExecutionBatchResult {
                        messages: Vec::new(),
                        terminate: false,
                    }
                };
                let has_more_tool_calls = has_tool_calls && !batch.terminate;
                last_terminated = batch.terminate;
                self.dispatch(AgentEvent::TurnEnd {
                    message: turn.message.clone(),
                    tool_results: batch.messages.clone(),
                })
                .await?;

                let decision_context = PrepareNextTurnContext {
                    message: turn.message.clone(),
                    tool_results: batch.messages.clone(),
                    context: prepared.context.clone(),
                    new_messages: prepared.new_messages.clone(),
                    signal: prepared.signal.clone(),
                };
                let update = prepare_next_turn(&self.context, decision_context).await?;
                if let Some(context) = update.context {
                    prepared.context = context;
                }
                if let Some(model) = update.model {
                    prepared.config.model = model;
                }
                if let Some(thinking_level) = update.thinking_level {
                    prepared.config.thinking_level = thinking_level;
                }
                if should_stop_after_turn(
                    &self.context,
                    ShouldStopAfterTurnContext {
                        message: turn.message,
                        tool_results: batch.messages,
                        context: prepared.context.clone(),
                        new_messages: prepared.new_messages.clone(),
                        signal: prepared.signal.clone(),
                    },
                )
                .await?
                {
                    return self
                        .finish_successful_run(prepared, AgentEndReason::Stopped)
                        .await;
                }

                match self.select_next_inner_turn(has_more_tool_calls).await? {
                    LoopDecision::Continue(next) => decision = next,
                    LoopDecision::Exhausted => break,
                    LoopDecision::Stop => {
                        return self
                            .finish_successful_run(prepared, AgentEndReason::Rejected)
                            .await;
                    }
                }
            }

            match self.select_follow_up(prepared).await? {
                LoopDecision::Continue(next) => decision = next,
                LoopDecision::Exhausted => {
                    let reason = if last_terminated {
                        AgentEndReason::Terminated
                    } else {
                        AgentEndReason::Completed
                    };
                    return self.finish_successful_run(prepared, reason).await;
                }
                LoopDecision::Stop => {
                    return self
                        .finish_successful_run(prepared, AgentEndReason::Rejected)
                        .await;
                }
            }
        }
    }

    async fn finish_successful_run(
        &self,
        prepared: &PreparedRun,
        reason: AgentEndReason,
    ) -> Result<Vec<Message>, AgentLoopError> {
        let messages = prepared.new_messages.clone();
        self.dispatch(AgentEvent::AgentEnd {
            reason,
            causes: prepared.causes.clone(),
            messages: messages.clone(),
        })
        .await?;
        Ok(messages)
    }

    async fn run_wrapped(
        &self,
        mut prepared: PreparedRun,
        entering: Vec<Message>,
        skip_initial_steering_poll: bool,
    ) -> Result<Vec<Message>, AgentLoopError> {
        let outcome = async {
            self.open_first_turn(&mut prepared, entering, skip_initial_steering_poll)
                .await?;
            self.run_successful_inner(&mut prepared).await
        }
        .await;

        match outcome {
            Ok(messages) => Ok(messages),
            Err(error) if error.is_eager() => Err(error),
            Err(error) => {
                self.settle_run_failure(&error, prepared.causes.clone())
                    .await
            }
        }
    }

    async fn settle_run_failure(
        &self,
        error: &AgentLoopError,
        causes: Vec<RunCause>,
    ) -> Result<Vec<Message>, AgentLoopError> {
        let mut failure = AssistantMessage::new(
            self.agent.model(),
            vec![AssistantContentBlock::Text(TextBlock::new(""))],
            Default::default(),
            if self.agent.signal().is_some_and(|signal| signal.aborted()) {
                StopReason::Aborted
            } else {
                StopReason::Error
            },
            now_millis(),
        );
        failure.error_message = Some(error.failure_message());
        let message = Message::Assistant(Box::new(failure.clone()));
        self.dispatch(AgentEvent::MessageStart(message.clone()))
            .await?;
        self.dispatch(AgentEvent::MessageEnd(message.clone()))
            .await?;
        self.dispatch(AgentEvent::TurnEnd {
            message: failure,
            tool_results: Vec::new(),
        })
        .await?;
        let messages = vec![message];
        self.dispatch(AgentEvent::AgentEnd {
            reason: AgentEndReason::Failed,
            causes,
            messages: messages.clone(),
        })
        .await?;
        Ok(messages)
    }

    async fn decide(
        &self,
        messages: Vec<Message>,
        reason: PreStepReason,
    ) -> Result<PreStepDecision, AgentLoopError> {
        let signal = self
            .agent
            .signal()
            .expect("pre-step is only dispatched during an active run");
        pre_step(
            &self.context,
            PreStepContext {
                messages,
                reason,
                signal,
            },
        )
        .await
    }

    async fn select_next_inner_turn(
        &self,
        has_more_tool_calls: bool,
    ) -> Result<LoopDecision, AgentLoopError> {
        let steering = self.claim(InboxTarget::Steering);
        if !has_more_tool_calls && steering.is_empty() {
            return Ok(LoopDecision::Exhausted);
        }
        let reason = if steering.is_empty() {
            PreStepReason::ToolResults
        } else {
            PreStepReason::Steering
        };
        Ok(
            match self.decide(envelope_messages(&steering), reason).await? {
                PreStepDecision::Enter(next) => LoopDecision::Continue(next),
                PreStepDecision::Reject(_) => LoopDecision::Stop,
            },
        )
    }

    async fn select_follow_up(
        &self,
        prepared: &mut PreparedRun,
    ) -> Result<LoopDecision, AgentLoopError> {
        let follow_up = self.claim(InboxTarget::FollowUp);
        if follow_up.is_empty() {
            return Ok(LoopDecision::Exhausted);
        }
        prepared.causes.extend(envelope_causes(&follow_up));
        Ok(
            match self
                .decide(envelope_messages(&follow_up), PreStepReason::NextTurn)
                .await?
            {
                PreStepDecision::Enter(next) => LoopDecision::Continue(next),
                PreStepDecision::Reject(_) => LoopDecision::Stop,
            },
        )
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
        dispatch_agent_event_with_signal(&self.context, event, self.agent.signal()).await
    }

    fn claim(&self, target: InboxTarget) -> Vec<InputEnvelope> {
        let policy = match target {
            InboxTarget::Steering => self.next_step_policy,
            InboxTarget::FollowUp => self.next_turn_policy,
        };
        self.agent.inbox().claim(target, policy)
    }
}

fn envelope_messages(envelopes: &[InputEnvelope]) -> Vec<Message> {
    envelopes
        .iter()
        .map(|envelope| envelope.message.clone())
        .collect()
}

fn envelope_causes(envelopes: &[InputEnvelope]) -> Vec<RunCause> {
    envelopes
        .iter()
        .map(|envelope| RunCause {
            id: envelope.id.clone(),
            origin: envelope.origin.clone(),
        })
        .collect()
}

fn live_tool_event(
    agent: Arc<AgentInstance>,
    context: Context,
    event: AgentEvent,
) -> BoxFuture<'static, Result<(), ToolLifecycleError>> {
    let reduction = reduce_event(&agent, &event).map_err(tool_lifecycle_error);
    Box::pin(async move {
        reduction?;
        dispatch_agent_event_with_signal(&context, event, agent.signal())
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
        agent::{AgentDefinition, AgentInstance, AgentStatus, ClaimPolicy},
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
        AgentEvent, AgentEventKind, AgentListenerError, AgentLoopError, Enter, PreStepContext,
        PreStepDecision, PreStepReason, PrepareNextTurnContext, RunConfigUpdate, RunContext,
        ShouldStopAfterTurnContext, TransformContextAction, TurnStopping, register_agent_listener,
        register_agent_listener_with_signal, register_pre_step_listener,
        register_prepare_next_turn_listener, register_should_stop_after_turn_listener,
        register_transform_context_listener,
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

    #[test]
    fn one_signal_reaches_lifecycle_transform_and_provider_then_clears() {
        run(async {
            let runtime = Runtime::new();
            let lifecycle_signal = Arc::new(Mutex::new(None));
            let transform_signal = Arc::new(Mutex::new(None));
            let lifecycle_observed = Arc::clone(&lifecycle_signal);
            let transform_observed = Arc::clone(&transform_signal);
            let plugin = PluginSpec::<Value>::new(
                "abort-observers",
                vec![],
                || json!({}),
                move |context, _config| {
                    let lifecycle_observed = Arc::clone(&lifecycle_observed);
                    let transform_observed = Arc::clone(&transform_observed);
                    async move {
                        register_agent_listener_with_signal(&context, move |current| {
                            let lifecycle_observed = Arc::clone(&lifecycle_observed);
                            async move {
                                if current.event.kind() == AgentEventKind::AgentStart {
                                    *lifecycle_observed.lock() = current.signal;
                                }
                                Ok(())
                            }
                        })
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                        register_transform_context_listener(&context, move |current| {
                            let transform_observed = Arc::clone(&transform_observed);
                            async move {
                                *transform_observed.lock() = Some(current.signal);
                                Ok(TransformContextAction::Next(None))
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
            let adapter = Arc::new(ScriptedAdapter::new([text_turn("done")]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("signal", [] as [&str; 0]).unwrap(),
                llm,
            );

            driver
                .prompt(PromptInput::Message(user("hello")))
                .await
                .unwrap();

            let lifecycle = lifecycle_signal.lock().clone().unwrap();
            let transform = transform_signal.lock().clone().unwrap();
            let provider = adapter.requests()[0].signal.clone().unwrap();
            assert_eq!(lifecycle, transform);
            assert_eq!(transform, provider);
            assert!(agent.signal().is_none());
        });
    }

    #[test]
    fn transform_context_is_provider_local_and_cannot_replace_signal_metadata() {
        run(async {
            let runtime = Runtime::new();
            let plugin = PluginSpec::<Value>::new(
                "transform",
                vec![],
                || json!({}),
                |context, _config| async move {
                    register_transform_context_listener(&context, |current| async move {
                        assert!(!current.signal.aborted());
                        assert_eq!(current.agent.signal(), Some(current.signal.clone()));
                        Ok(TransformContextAction::Next(Some(vec![user(
                            "provider-only",
                        )])))
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                },
            )
            .erase();
            runtime.mount(&plugin, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            let adapter = Arc::new(ScriptedAdapter::new([text_turn("done")]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let session = Session::new("transform", [] as [&str; 0]).unwrap();
            let (driver, agent) = loop_for_with_llm(&runtime, session, llm);

            driver
                .prompt(PromptInput::Message(user("persistent")))
                .await
                .unwrap();

            assert_eq!(
                adapter.requests()[0].context.messages,
                vec![user("provider-only")]
            );
            assert_eq!(agent.messages().unwrap()[0], user("persistent"));
        });
    }

    #[test]
    fn exception_after_abort_is_represented_as_aborted_and_uses_live_persistent_model() {
        run(async {
            let runtime = Runtime::new();
            let agent_slot: Arc<Mutex<Option<Arc<AgentInstance>>>> = Arc::new(Mutex::new(None));
            let listener_slot = Arc::clone(&agent_slot);
            let plugin = PluginSpec::<Value>::new(
                "abort-and-fail",
                vec![],
                || json!({}),
                move |context, _config| {
                    let listener_slot = Arc::clone(&listener_slot);
                    async move {
                        register_agent_listener(&context, move |event| {
                            let listener_slot = Arc::clone(&listener_slot);
                            async move {
                                if event.kind() == AgentEventKind::TurnStart {
                                    listener_slot.lock().as_ref().unwrap().abort();
                                    return Err(AgentListenerError::new("boom"));
                                }
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
            let (driver, agent) =
                loop_for(&runtime, Session::new("failure", [] as [&str; 0]).unwrap());
            *agent_slot.lock() = Some(Arc::clone(&agent));

            let messages = driver
                .prompt(PromptInput::Message(user("hello")))
                .await
                .unwrap();
            let Message::Assistant(message) = &messages[0] else {
                panic!("failure must be assistant")
            };
            assert_eq!(message.stop_reason, StopReason::Aborted);
            assert_eq!(message.model, agent.model().model_id());
            assert!(agent.signal().is_none());
            assert_eq!(agent.status(), AgentStatus::Idle);
        });
    }

    #[test]
    fn abort_requested_during_agent_end_settles_before_the_signal_clears() {
        run(async {
            let runtime = Runtime::new();
            let agent_slot: Arc<Mutex<Option<Arc<AgentInstance>>>> = Arc::new(Mutex::new(None));
            let listener_slot = Arc::clone(&agent_slot);
            let observed = Arc::new(Mutex::new(false));
            let listener_observed = Arc::clone(&observed);
            let plugin = PluginSpec::<Value>::new(
                "abort-during-agent-end",
                vec![],
                || json!({}),
                move |context, _config| {
                    let listener_slot = Arc::clone(&listener_slot);
                    let listener_observed = Arc::clone(&listener_observed);
                    async move {
                        register_agent_listener_with_signal(&context, move |current| {
                            let listener_slot = Arc::clone(&listener_slot);
                            let listener_observed = Arc::clone(&listener_observed);
                            async move {
                                if current.event.kind() == AgentEventKind::AgentEnd {
                                    let signal = current.signal.expect("run signal remains live");
                                    listener_slot.lock().as_ref().unwrap().abort();
                                    *listener_observed.lock() = signal.aborted();
                                }
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
            let adapter = Arc::new(ScriptedAdapter::new([text_turn("done")]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter);
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("abort-agent-end", [] as [&str; 0]).unwrap(),
                llm,
            );
            *agent_slot.lock() = Some(Arc::clone(&agent));

            driver
                .prompt(PromptInput::Message(user("hello")))
                .await
                .unwrap();

            assert!(*observed.lock());
            assert!(agent.signal().is_none());
            assert_eq!(agent.status(), AgentStatus::Idle);
        });
    }

    #[test]
    fn pre_step_prepare_and_stopping_observe_the_active_signal() {
        run(async {
            let runtime = Runtime::new();
            let agent_slot: Arc<Mutex<Option<Arc<AgentInstance>>>> = Arc::new(Mutex::new(None));
            let pre_slot = Arc::clone(&agent_slot);
            let prepare_slot = Arc::clone(&agent_slot);
            let stop_slot = Arc::clone(&agent_slot);
            let plugin = PluginSpec::<Value>::new(
                "decision-signals",
                vec![],
                || json!({}),
                move |context, _config| {
                    let pre_slot = Arc::clone(&pre_slot);
                    let prepare_slot = Arc::clone(&prepare_slot);
                    let stop_slot = Arc::clone(&stop_slot);
                    async move {
                        register_pre_step_listener(&context, move |current, next| {
                            let agent = pre_slot.lock().clone().unwrap();
                            async move {
                                assert_eq!(agent.signal(), Some(current.signal.clone()));
                                next.call(None).await
                            }
                        })
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                        register_prepare_next_turn_listener(&context, move |current, _next| {
                            let agent = prepare_slot.lock().clone().unwrap();
                            async move {
                                assert_eq!(agent.signal(), Some(current.signal));
                                Ok(RunConfigUpdate::default())
                            }
                        })
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                        register_should_stop_after_turn_listener(&context, move |current| {
                            let agent = stop_slot.lock().clone().unwrap();
                            async move {
                                assert_eq!(agent.signal(), Some(current.signal));
                                Ok(TurnStopping::Stop)
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
            let adapter = Arc::new(ScriptedAdapter::new([text_turn("done")]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter);
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("decisions", [] as [&str; 0]).unwrap(),
                llm,
            );
            *agent_slot.lock() = Some(agent);
            driver
                .prompt(PromptInput::Message(user("hello")))
                .await
                .unwrap();
        });
    }

    fn tool_turn(call_id: &str, tool_name: &str) -> Script {
        let message = AssistantMessage::new(
            identity(),
            vec![AssistantContentBlock::ToolCall(ToolCall::new(
                call_id,
                tool_name,
                BTreeMap::new(),
            ))],
            Usage::default(),
            StopReason::ToolUse,
            2.0,
        );
        Script::new([ScriptItem::Chunk(Box::new(StreamChunk::Done {
            reason: DoneReason::ToolUse,
            message,
        }))])
    }

    fn tool_output(text: &str) -> AgentToolResult {
        AgentToolResult {
            content: vec![ToolResultContentBlock::Text(TextBlock::new(text))],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        }
    }

    fn message_text(message: &AssistantMessage) -> &str {
        match message.content.first() {
            Some(AssistantContentBlock::Text(block)) => &block.text,
            Some(AssistantContentBlock::Thinking(_))
            | Some(AssistantContentBlock::ToolCall(_))
            | None => "",
        }
    }

    fn decision_plugin<P, S>(prepare: P, stop: S) -> DynPluginSpec
    where
        P: Fn(PrepareNextTurnContext) -> RunConfigUpdate + Send + Sync + 'static,
        S: Fn(ShouldStopAfterTurnContext) -> TurnStopping + Send + Sync + 'static,
    {
        let prepare = Arc::new(prepare);
        let stop = Arc::new(stop);
        PluginSpec::<Value>::new(
            "decision-listeners",
            vec![],
            || json!({}),
            move |context, _config| {
                let prepare = Arc::clone(&prepare);
                let stop = Arc::clone(&stop);
                async move {
                    register_prepare_next_turn_listener(&context, move |current, _next| {
                        let update = prepare(current);
                        async move { Ok(update) }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    register_should_stop_after_turn_listener(&context, move |current| {
                        let decision = stop(current);
                        async move { Ok(decision) }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                }
            },
        )
        .erase()
    }

    fn prepare_once_plugin(update: RunConfigUpdate) -> DynPluginSpec {
        let update = Arc::new(Mutex::new(Some(update)));
        decision_plugin(
            move |_context| update.lock().take().unwrap_or_default(),
            |_context| TurnStopping::Continue,
        )
    }

    fn counting_decision_plugin(count: Arc<AtomicUsize>) -> DynPluginSpec {
        decision_plugin(
            {
                let count = Arc::clone(&count);
                move |_context| {
                    count.fetch_add(1, Ordering::SeqCst);
                    RunConfigUpdate::default()
                }
            },
            move |_context| {
                count.fetch_add(1, Ordering::SeqCst);
                TurnStopping::Continue
            },
        )
    }

    fn pre_step_plugin<F>(listener: F) -> DynPluginSpec
    where
        F: Fn(PreStepContext) -> Option<Vec<Message>> + Send + Sync + 'static,
    {
        let listener = Arc::new(listener);
        PluginSpec::<Value>::new(
            "pre-step-listener",
            vec![],
            || json!({}),
            move |context, _config| {
                let listener = Arc::clone(&listener);
                async move {
                    register_pre_step_listener(&context, move |current, next| {
                        let replacement = listener(current.clone());
                        async move {
                            match replacement {
                                Some(messages) => Ok(PreStepDecision::Enter(Enter {
                                    messages,
                                    system_override: None,
                                    history_window: None,
                                })),
                                None => next.call(None).await,
                            }
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

    #[test]
    fn successful_run_continues_for_tools_steering_and_follow_up_in_pi_order() {
        run(async {
            let runtime = Runtime::new();
            let trace = Arc::new(Mutex::new(Vec::<String>::new()));
            let scripts = [
                tool_turn("call-1", "echo"),
                text_turn("after-steering"),
                text_turn("after-follow-up"),
            ];
            let adapter = Arc::new(ScriptedAdapter::new(scripts));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "echo",
                        "echo",
                        serde_json::from_value(json!({})).unwrap(),
                        "echo",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async { Ok(tool_output("done")) })
                        },
                    ),
                )
                .unwrap();
            runtime
                .mount(
                    &decision_plugin(
                        {
                            let trace = Arc::clone(&trace);
                            move |context| {
                                trace
                                    .lock()
                                    .push(format!("prepare:{}", message_text(&context.message)));
                                RunConfigUpdate::default()
                            }
                        },
                        {
                            let trace = Arc::clone(&trace);
                            move |context| {
                                trace
                                    .lock()
                                    .push(format!("stop:{}", message_text(&context.message)));
                                TurnStopping::Continue
                            }
                        },
                    ),
                    json!({}),
                )
                .unwrap();
            runtime
                .mount(
                    &PluginSpec::<Value>::new("turn-end-listener", vec![], || json!({}), {
                        let trace = Arc::clone(&trace);
                        move |context, _config| {
                            let trace = Arc::clone(&trace);
                            async move {
                                register_agent_listener(&context, move |event| {
                                    let trace = Arc::clone(&trace);
                                    async move {
                                        if event.kind() == AgentEventKind::TurnEnd {
                                            trace.lock().push("turn_end".into());
                                        }
                                        Ok(())
                                    }
                                })
                                .map_err(|error| PluginInitError::new(error.to_string()))?;
                                Ok(())
                            }
                        }
                    })
                    .erase(),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();
            let (mut driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            driver.set_next_step_policy(ClaimPolicy::OneAtATime);
            driver.set_next_turn_policy(ClaimPolicy::OneAtATime);
            agent.follow_up(user("follow"), None);
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            agent.steer(user("steer"), None);

            let messages = driver.run_successful(prepared).await.unwrap();

            assert_eq!(adapter.requests().len(), 3);
            assert_eq!(messages.len(), 7);
            assert_eq!(agent.status(), AgentStatus::Idle);
            assert!(!agent.has_queued_messages());
            assert_eq!(
                trace.lock().as_slice(),
                [
                    "turn_end",
                    "prepare:",
                    "stop:",
                    "turn_end",
                    "prepare:after-steering",
                    "stop:after-steering",
                    "turn_end",
                    "prepare:after-follow-up",
                    "stop:after-follow-up",
                ]
            );
        });
    }

    #[test]
    fn prepare_next_turn_replaces_context_model_and_thinking_for_this_run_only() {
        run(async {
            let runtime = Runtime::new();
            let replacement_model = ModelIdentity::new("provider", "api", "replacement").unwrap();
            let replacement_adapter = Arc::new(ScriptedAdapter::new([text_turn("replacement")]));
            let initial_adapter = Arc::new(ScriptedAdapter::new([tool_turn("call-1", "echo")]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), initial_adapter.clone());
            llm.register(replacement_model.clone(), replacement_adapter.clone());
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "echo",
                        "echo",
                        serde_json::from_value(json!({})).unwrap(),
                        "echo",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async { Ok(tool_output("done")) })
                        },
                    ),
                )
                .unwrap();
            let replacement_context = RunContext {
                system_prompt: "replacement-system".into(),
                messages: vec![user("replacement-history")],
                tools: Vec::new(),
            };
            runtime
                .mount(
                    &prepare_once_plugin(RunConfigUpdate {
                        context: Some(replacement_context.clone()),
                        model: Some(replacement_model.clone()),
                        thinking_level: Some(crate::agent::ThinkingLevel::High),
                    }),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();

            driver.run_successful(prepared).await.unwrap();

            let replacement_requests = replacement_adapter.requests();
            assert_eq!(replacement_requests.len(), 1);
            assert_eq!(replacement_requests[0].model, replacement_model);
            assert_eq!(
                replacement_requests[0].context.system_prompt.as_deref(),
                Some("replacement-system")
            );
            assert_eq!(
                replacement_requests[0].context.messages,
                replacement_context.messages
            );
            assert!(
                replacement_requests[0]
                    .context
                    .tools
                    .as_ref()
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                replacement_requests[0].options.reasoning,
                Some(LlmThinkingLevel::High)
            );
            assert_eq!(agent.system_prompt(), "system");
            assert_eq!(agent.model(), identity());
            assert_eq!(agent.thinking_level(), crate::agent::ThinkingLevel::Off);
        });
    }

    #[test]
    fn represented_terminals_skip_every_post_turn_decision_and_leave_queues_unclaimed() {
        run(async {
            for reason in [ErrorReason::Error, ErrorReason::Aborted] {
                let runtime = Runtime::new();
                let decisions = Arc::new(AtomicUsize::new(0));
                runtime
                    .mount(&counting_decision_plugin(Arc::clone(&decisions)), json!({}))
                    .unwrap();
                runtime.reconcile().await.unwrap();
                let mut terminal = AssistantMessage::pending(identity(), 2.0);
                terminal.stop_reason = match reason {
                    ErrorReason::Error => StopReason::Error,
                    ErrorReason::Aborted => StopReason::Aborted,
                };
                let adapter = Arc::new(ScriptedAdapter::new([Script::new([ScriptItem::Chunk(
                    Box::new(StreamChunk::Error {
                        reason,
                        error: terminal,
                    }),
                )])]));
                let llm = Arc::new(LlmService::new());
                llm.register(identity(), adapter);
                let (driver, agent) = loop_for_with_llm(
                    &runtime,
                    Session::new("room-a", [] as [&str; 0]).unwrap(),
                    llm,
                );
                agent.follow_up(user("queued-follow-up"), None);
                let prepared = driver
                    .prepare_prompt_run(PromptInput::Message(user("prompt")))
                    .await
                    .unwrap();
                agent.steer(user("queued-steering"), None);

                driver.run_successful(prepared).await.unwrap();

                assert_eq!(decisions.load(Ordering::SeqCst), 0);
                assert!(agent.has_queued_messages());
            }
        });
    }

    #[test]
    fn twenty_one_tool_turns_are_not_capped() {
        run(async {
            let runtime = Runtime::new();
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "echo",
                        "echo",
                        serde_json::from_value(json!({})).unwrap(),
                        "echo",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async { Ok(tool_output("done")) })
                        },
                    ),
                )
                .unwrap();
            let scripts = (0..21)
                .map(|index| tool_turn(&format!("call-{index}"), "echo"))
                .chain([text_turn("finished")]);
            let adapter = Arc::new(ScriptedAdapter::new(scripts));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, _agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();

            driver.run_successful(prepared).await.unwrap();

            assert_eq!(adapter.requests().len(), 22);
        });
    }

    #[test]
    fn steering_without_tool_continuation_starts_the_next_turn() {
        run(async {
            let runtime = Runtime::new();
            let adapter = Arc::new(ScriptedAdapter::new([
                text_turn("first"),
                text_turn("steered"),
            ]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            agent.steer(user("late-steering"), None);

            driver.run_successful(prepared).await.unwrap();

            assert_eq!(adapter.requests().len(), 2);
            assert_eq!(
                adapter.requests()[1].context.messages.last(),
                Some(&user("late-steering"))
            );
        });
    }

    #[test]
    fn terminate_suppresses_only_tool_driven_continuation() {
        run(async {
            let runtime = Runtime::new();
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "ending",
                        "ending",
                        serde_json::from_value(json!({})).unwrap(),
                        "ending",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async {
                                let mut output = tool_output("done");
                                output.terminate = Some(true);
                                Ok(output)
                            })
                        },
                    ),
                )
                .unwrap();
            let decisions = Arc::new(AtomicUsize::new(0));
            runtime
                .mount(&counting_decision_plugin(Arc::clone(&decisions)), json!({}))
                .unwrap();
            runtime.reconcile().await.unwrap();
            let adapter = Arc::new(ScriptedAdapter::new([
                tool_turn("call-1", "ending"),
                text_turn("steered"),
            ]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            agent.steer(user("late-steering"), None);

            driver.run_successful(prepared).await.unwrap();

            assert_eq!(adapter.requests().len(), 2);
            assert_eq!(decisions.load(Ordering::SeqCst), 4);
        });
    }

    #[test]
    fn stop_precedes_queue_claim_and_leaves_follow_up_pending() {
        run(async {
            let runtime = Runtime::new();
            runtime
                .mount(
                    &decision_plugin(
                        |_context| RunConfigUpdate::default(),
                        |_context| TurnStopping::Stop,
                    ),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();
            let adapter = Arc::new(ScriptedAdapter::new([text_turn("first")]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            agent.follow_up(user("later"), None);
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();

            driver.run_successful(prepared).await.unwrap();

            assert_eq!(adapter.requests().len(), 1);
            assert!(agent.has_queued_messages());
        });
    }

    #[test]
    fn steering_and_follow_up_use_independently_configurable_all_policies() {
        run(async {
            let runtime = Runtime::new();
            let adapter = Arc::new(ScriptedAdapter::new([
                text_turn("first"),
                text_turn("after-steering"),
                text_turn("after-follow-up"),
            ]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (mut driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            driver.set_next_step_policy(ClaimPolicy::All);
            driver.set_next_turn_policy(ClaimPolicy::All);
            agent.follow_up(user("follow-1"), None);
            agent.follow_up(user("follow-2"), None);
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            agent.steer(user("steer-1"), None);
            agent.steer(user("steer-2"), None);

            driver.run_successful(prepared).await.unwrap();

            let requests = adapter.requests();
            assert_eq!(requests.len(), 3);
            assert_eq!(
                &requests[1].context.messages[requests[1].context.messages.len() - 2..],
                [user("steer-1"), user("steer-2")]
            );
            assert_eq!(
                &requests[2].context.messages[requests[2].context.messages.len() - 2..],
                [user("follow-1"), user("follow-2")]
            );
        });
    }

    #[test]
    fn pre_step_observes_reasons_and_can_replace_admitted_messages() {
        run(async {
            let runtime = Runtime::new();
            let reasons = Arc::new(Mutex::new(Vec::new()));
            runtime
                .mount(
                    &pre_step_plugin({
                        let reasons = Arc::clone(&reasons);
                        move |context| {
                            reasons.lock().push(context.reason);
                            (context.reason == PreStepReason::Steering)
                                .then(|| vec![user("rewritten-steering")])
                        }
                    }),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();
            let adapter = Arc::new(ScriptedAdapter::new([
                text_turn("first"),
                text_turn("second"),
            ]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            agent.steer(user("original-steering"), None);

            driver.run_successful(prepared).await.unwrap();

            assert_eq!(
                reasons.lock().as_slice(),
                [PreStepReason::Initial, PreStepReason::Steering]
            );
            assert_eq!(
                adapter.requests()[1].context.messages.last(),
                Some(&user("rewritten-steering"))
            );
            assert!(
                !agent
                    .messages()
                    .unwrap()
                    .contains(&user("original-steering"))
            );
        });
    }

    #[test]
    fn dynamically_added_tool_is_available_to_the_next_turn_only_through_run_growth() {
        run(async {
            let runtime = Runtime::new();
            let introduced_executions = Arc::new(AtomicUsize::new(0));
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "introducer",
                        "introducer",
                        serde_json::from_value(json!({})).unwrap(),
                        "introducer",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async {
                                let mut output = tool_output("introduced");
                                output.added_tool_names = Some(vec!["introduced".into()]);
                                Ok(output)
                            })
                        },
                    ),
                )
                .unwrap();
            let adapter = Arc::new(ScriptedAdapter::new([
                tool_turn("call-1", "introducer"),
                tool_turn("call-2", "introduced"),
                text_turn("finished"),
            ]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), adapter.clone());
            let (driver, _agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let prepared = driver
                .prepare_prompt_run(PromptInput::Message(user("prompt")))
                .await
                .unwrap();
            let introduced = runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "introduced",
                        "introduced",
                        serde_json::from_value(json!({})).unwrap(),
                        "introduced",
                        {
                            let introduced_executions = Arc::clone(&introduced_executions);
                            move |_request: ToolExecutionRequest| {
                                let introduced_executions = Arc::clone(&introduced_executions);
                                Box::pin(async move {
                                    introduced_executions.fetch_add(1, Ordering::SeqCst);
                                    Ok(tool_output("used"))
                                })
                            }
                        },
                    ),
                )
                .unwrap();

            driver.run_successful(prepared).await.unwrap();

            assert_eq!(introduced_executions.load(Ordering::SeqCst), 1);
            assert_eq!(
                adapter.requests()[1]
                    .context
                    .tools
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["introducer", "introduced"]
            );
            drop(introduced);
        });
    }

    #[test]
    fn prepare_next_turn_replacement_does_not_leak_to_a_later_run_snapshot() {
        run(async {
            let runtime = Runtime::new();
            runtime
                .tools()
                .register_for_scope(
                    None,
                    ToolDefinition::new(
                        "echo",
                        "echo",
                        serde_json::from_value(json!({})).unwrap(),
                        "echo",
                        |_request: ToolExecutionRequest| {
                            Box::pin(async { Ok(tool_output("done")) })
                        },
                    ),
                )
                .unwrap();
            let replacement_model = ModelIdentity::new("provider", "api", "replacement").unwrap();
            let replacement_adapter = Arc::new(ScriptedAdapter::new([text_turn("replacement")]));
            let original_adapter = Arc::new(ScriptedAdapter::new([
                tool_turn("call-1", "echo"),
                text_turn("second-run"),
            ]));
            let llm = Arc::new(LlmService::new());
            llm.register(identity(), original_adapter.clone());
            llm.register(replacement_model.clone(), replacement_adapter);
            runtime
                .mount(
                    &prepare_once_plugin(RunConfigUpdate {
                        context: Some(RunContext {
                            system_prompt: "replacement-system".into(),
                            messages: vec![user("replacement-history")],
                            tools: Vec::new(),
                        }),
                        model: Some(replacement_model),
                        thinking_level: Some(crate::agent::ThinkingLevel::Max),
                    }),
                    json!({}),
                )
                .unwrap();
            runtime.reconcile().await.unwrap();
            let (driver, _agent) = loop_for_with_llm(
                &runtime,
                Session::new("room-a", [] as [&str; 0]).unwrap(),
                llm,
            );
            let first = driver
                .prepare_prompt_run(PromptInput::Message(user("first")))
                .await
                .unwrap();
            driver.run_successful(first).await.unwrap();
            let second = driver
                .prepare_prompt_run(PromptInput::Message(user("second")))
                .await
                .unwrap();
            driver.run_successful(second).await.unwrap();

            let requests = original_adapter.requests();
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[1].model, identity());
            assert_eq!(requests[1].context.system_prompt.as_deref(), Some("system"));
            assert!(requests[1].context.messages.contains(&user("second")));
            assert_ne!(
                requests[1].context.messages,
                vec![user("replacement-history")]
            );
            assert_eq!(requests[1].options.reasoning, None);
        });
    }
}
