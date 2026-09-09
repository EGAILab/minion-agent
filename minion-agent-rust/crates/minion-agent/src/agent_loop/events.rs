use std::{future::Future, sync::Arc};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    Context, DispatchMode, EventListenerHandle, EventName, EventSpec,
    llm::{AssistantMessage, Message, StreamChunk, ToolResultMessage},
    runtime::RunSignal,
    tools::{ToolExecutionEnd, ToolExecutionStart, ToolExecutionUpdate},
};

use super::{AgentListenerError, AgentLoopError};

/// Provenance for one queued input that caused or extended a run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunCause {
    pub id: String,
    pub origin: Option<serde_json::Value>,
}

/// Why one complete run settled.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEndReason {
    Completed,
    Terminated,
    Stopped,
    Rejected,
    Error,
    Aborted,
    Failed,
}

/// Complete live Agent lifecycle vocabulary.
///
/// The complete owned stream event and partial are intentionally kept in the
/// public variant rather than hidden behind implementation-specific boxing.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    AgentStart {
        causes: Vec<RunCause>,
    },
    TurnStart,
    MessageStart(Message),
    MessageUpdate {
        event: StreamChunk,
        partial: AssistantMessage,
    },
    MessageEnd(Message),
    ToolExecutionStart(ToolExecutionStart),
    ToolExecutionUpdate(ToolExecutionUpdate),
    ToolExecutionEnd(ToolExecutionEnd),
    TurnEnd {
        message: AssistantMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    AgentEnd {
        reason: AgentEndReason,
        causes: Vec<RunCause>,
        messages: Vec<Message>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEventKind {
    AgentStart,
    TurnStart,
    MessageStart,
    MessageUpdate,
    MessageEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolExecutionEnd,
    TurnEnd,
    AgentEnd,
}

impl AgentEvent {
    pub fn kind(&self) -> AgentEventKind {
        match self {
            Self::AgentStart { .. } => AgentEventKind::AgentStart,
            Self::TurnStart => AgentEventKind::TurnStart,
            Self::MessageStart(_) => AgentEventKind::MessageStart,
            Self::MessageUpdate { .. } => AgentEventKind::MessageUpdate,
            Self::MessageEnd(_) => AgentEventKind::MessageEnd,
            Self::ToolExecutionStart(_) => AgentEventKind::ToolExecutionStart,
            Self::ToolExecutionUpdate(_) => AgentEventKind::ToolExecutionUpdate,
            Self::ToolExecutionEnd(_) => AgentEventKind::ToolExecutionEnd,
            Self::TurnEnd { .. } => AgentEventKind::TurnEnd,
            Self::AgentEnd { .. } => AgentEventKind::AgentEnd,
        }
    }
}

#[derive(Clone)]
struct AgentEventDispatch {
    event: AgentEvent,
    signal: Option<RunSignal>,
    first_error: Arc<Mutex<Option<AgentListenerError>>>,
}

#[derive(Clone, Debug)]
pub struct AgentLifecycleContext {
    pub event: AgentEvent,
    pub signal: Option<RunSignal>,
}

fn agent_lifecycle_spec() -> EventSpec<AgentEventDispatch, ()> {
    EventSpec::new(
        EventName::new("agent/lifecycle-event").expect("normative event name is valid"),
        DispatchMode::Serial,
        |_| (),
    )
}

/// Registers a scope/fiber-owned, fallible async lifecycle listener.
///
/// EventBus retains registration order. Each wrapper observes the dispatch's
/// typed first-error cell before invoking user code, so a listener failure
/// prevents all later listeners for that event without converting the error
/// into an untyped payload.
pub fn register_agent_listener<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, AgentLoopError>
where
    F: Fn(AgentEvent) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), AgentListenerError>> + Send + 'static,
{
    register_agent_listener_with_signal(context, move |dispatch| listener(dispatch.event))
}

/// Registers a lifecycle listener that also observes the active run signal.
pub fn register_agent_listener_with_signal<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, AgentLoopError>
where
    F: Fn(AgentLifecycleContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), AgentListenerError>> + Send + 'static,
{
    let events = context.events()?;
    let spec = agent_lifecycle_spec();
    events.declare(&spec)?;
    let effects = context.effect_store();
    Ok(
        events.on_serial(&spec, &effects, context.scope(), move |dispatch| {
            let already_failed = dispatch.first_error.lock().is_some();
            let future = if already_failed {
                None
            } else {
                Some(listener(AgentLifecycleContext {
                    event: dispatch.event.clone(),
                    signal: dispatch.signal.clone(),
                }))
            };
            async move {
                let Some(future) = future else {
                    return;
                };
                if let Err(error) = future.await {
                    let mut first_error = dispatch.first_error.lock();
                    if first_error.is_none() {
                        *first_error = Some(error);
                    }
                }
                // Pi's awaited serial listener loop suspends after every
                // invoked listener, even when its future completed eagerly.
                tokio::task::yield_now().await;
            }
        })?,
    )
}

/// Dispatches one complete lifecycle event through the public serial seam.
pub async fn dispatch_agent_event(
    context: &Context,
    event: AgentEvent,
) -> Result<(), AgentLoopError> {
    dispatch_agent_event_with_signal(context, event, None).await
}

pub(crate) async fn dispatch_agent_event_with_signal(
    context: &Context,
    event: AgentEvent,
    signal: Option<RunSignal>,
) -> Result<(), AgentLoopError> {
    let events = context.events()?;
    let spec = agent_lifecycle_spec();
    events.declare(&spec)?;
    let first_error = Arc::new(Mutex::new(None));
    events
        .serial(
            &spec,
            AgentEventDispatch {
                event,
                signal,
                first_error: Arc::clone(&first_error),
            },
            context.scope(),
        )
        .await?;
    let listener_error = first_error.lock().clone();
    match listener_error {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}
