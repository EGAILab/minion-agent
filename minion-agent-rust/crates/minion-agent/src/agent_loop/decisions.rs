use std::{future::Future, sync::Arc};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    Context, DispatchMode, EventError, EventListenerHandle, EventName, EventSpec, Next,
    WaterfallError,
    llm::{AssistantMessage, Message, ToolResultMessage},
};

use super::{AgentListenerError, AgentLoopError, RunConfigUpdate, RunContext};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Reject {
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Enter {
    pub messages: Vec<Message>,
    pub system_override: Option<String>,
    pub history_window: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreStepDecision {
    Reject(Reject),
    Enter(Enter),
}

/// Why an input batch is being considered for admission.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreStepReason {
    Initial,
    ToolResults,
    Steering,
    NextTurn,
    Continuation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreStepContext {
    pub messages: Vec<Message>,
    pub reason: PreStepReason,
}

/// A listener's ordered opinion on whether the run should continue.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStopping {
    #[default]
    NoOpinion,
    Continue,
    Stop,
}

/// Owned snapshot delivered to `prepare-next-turn` listeners.
#[derive(Clone, Debug)]
pub struct PrepareNextTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: RunContext,
    pub new_messages: Vec<Message>,
}

/// Owned snapshot delivered to `should-stop-after-turn` listeners.
#[derive(Clone, Debug)]
pub struct ShouldStopAfterTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: RunContext,
    pub new_messages: Vec<Message>,
}

/// Resolves the intentional multi-listener extension in registration order.
///
/// Pinned Pi has one callback. Minion permits several listeners while keeping
/// the first concrete opinion authoritative; absence of an opinion continues.
pub fn resolve_stopping(decisions: impl IntoIterator<Item = TurnStopping>) -> TurnStopping {
    decisions
        .into_iter()
        .find(|decision| *decision != TurnStopping::NoOpinion)
        .unwrap_or(TurnStopping::Continue)
}

fn pre_step_spec() -> EventSpec<PreStepContext, PreStepDecision> {
    EventSpec::new(
        EventName::new("agent/pre-step").expect("normative event name is valid"),
        DispatchMode::Waterfall,
        |current: &PreStepContext| {
            PreStepDecision::Enter(Enter {
                messages: current.messages.clone(),
                system_override: None,
                history_window: None,
            })
        },
    )
}

/// Registers an ordered pre-step waterfall listener.
pub fn register_pre_step_listener<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, EventError>
where
    F: Fn(PreStepContext, Next<PreStepContext, PreStepDecision>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<PreStepDecision, WaterfallError>> + Send + 'static,
{
    let events = context.events()?;
    let spec = pre_step_spec();
    events.declare(&spec)?;
    let effects = context.effect_store();
    events.on_waterfall(&spec, &effects, context.scope(), listener)
}

pub(crate) async fn pre_step(
    runtime: &Context,
    context: PreStepContext,
) -> Result<PreStepDecision, AgentLoopError> {
    let events = runtime.events()?;
    let spec = pre_step_spec();
    events.declare(&spec)?;
    Ok(events.waterfall(&spec, context, runtime.scope()).await?)
}

fn prepare_next_turn_spec() -> EventSpec<PrepareNextTurnContext, RunConfigUpdate> {
    EventSpec::new(
        EventName::new("agent/prepare-next-turn").expect("normative event name is valid"),
        DispatchMode::Waterfall,
        |_| RunConfigUpdate::default(),
    )
}

/// Registers a prepare-next-turn waterfall listener.
///
/// Returning directly owns the decision; calling `next` delegates to later
/// listeners and ultimately the no-op terminal update.
pub fn register_prepare_next_turn_listener<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, EventError>
where
    F: Fn(PrepareNextTurnContext, Next<PrepareNextTurnContext, RunConfigUpdate>) -> Fut
        + Send
        + Sync
        + 'static,
    Fut: Future<Output = Result<RunConfigUpdate, WaterfallError>> + Send + 'static,
{
    let events = context.events()?;
    let spec = prepare_next_turn_spec();
    events.declare(&spec)?;
    let effects = context.effect_store();
    events.on_waterfall(&spec, &effects, context.scope(), listener)
}

pub(crate) async fn prepare_next_turn(
    runtime: &Context,
    context: PrepareNextTurnContext,
) -> Result<RunConfigUpdate, AgentLoopError> {
    let events = runtime.events()?;
    let spec = prepare_next_turn_spec();
    events.declare(&spec)?;
    Ok(events.waterfall(&spec, context, runtime.scope()).await?)
}

#[derive(Clone)]
struct StoppingDispatch {
    context: ShouldStopAfterTurnContext,
    opinions: Arc<Mutex<Vec<TurnStopping>>>,
    first_error: Arc<Mutex<Option<AgentListenerError>>>,
}

fn should_stop_after_turn_spec() -> EventSpec<StoppingDispatch, ()> {
    EventSpec::new(
        EventName::new("agent/turn-stopping").expect("normative event name is valid"),
        DispatchMode::Serial,
        |_| (),
    )
}

/// Registers a fallible, ordered turn-stopping listener.
pub fn register_should_stop_after_turn_listener<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, EventError>
where
    F: Fn(ShouldStopAfterTurnContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<TurnStopping, AgentListenerError>> + Send + 'static,
{
    let events = context.events()?;
    let spec = should_stop_after_turn_spec();
    events.declare(&spec)?;
    let effects = context.effect_store();
    events.on_serial(&spec, &effects, context.scope(), move |dispatch| {
        let future = if dispatch.first_error.lock().is_some() {
            None
        } else {
            Some(listener(dispatch.context.clone()))
        };
        async move {
            let Some(future) = future else {
                return;
            };
            match future.await {
                Ok(opinion) => dispatch.opinions.lock().push(opinion),
                Err(error) => {
                    let mut first_error = dispatch.first_error.lock();
                    if first_error.is_none() {
                        *first_error = Some(error);
                    }
                }
            }
        }
    })
}

pub(crate) async fn should_stop_after_turn(
    runtime: &Context,
    context: ShouldStopAfterTurnContext,
) -> Result<bool, AgentLoopError> {
    let events = runtime.events()?;
    let spec = should_stop_after_turn_spec();
    events.declare(&spec)?;
    let opinions = Arc::new(Mutex::new(Vec::new()));
    let first_error = Arc::new(Mutex::new(None));
    events
        .serial(
            &spec,
            StoppingDispatch {
                context,
                opinions: Arc::clone(&opinions),
                first_error: Arc::clone(&first_error),
            },
            runtime.scope(),
        )
        .await?;
    if let Some(error) = first_error.lock().clone() {
        return Err(error.into());
    }
    let stopping = resolve_stopping(opinions.lock().iter().copied());
    Ok(stopping == TurnStopping::Stop)
}
