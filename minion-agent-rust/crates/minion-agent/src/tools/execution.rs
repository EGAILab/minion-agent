use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::{FutureExt, StreamExt, stream::FuturesUnordered};
use serde_json::Value;
use thiserror::Error;

use crate::{
    Context, DispatchMode, EventBus, EventError, EventListenerHandle, EventName, EventSpec,
    RuntimeError, ScopeHandle,
    llm::{StopReason, TextBlock, ToolCall, ToolResultContentBlock, ToolResultMessage, Usage},
};

use super::{
    AgentToolResult, ExecutionMode, ToolDefinition, ToolExecutionRequest, ToolExecutionSignal,
};

/// Batch-level execution inputs owned by Layer 06.
#[derive(Clone)]
pub struct ToolExecutionOptions {
    pub stop_reason: StopReason,
    pub default_mode: ExecutionMode,
    pub signal: Option<Arc<dyn ToolExecutionSignal>>,
    pub timestamp: f64,
}

impl ToolExecutionOptions {
    pub fn new(stop_reason: StopReason, timestamp: f64) -> Self {
        Self {
            stop_reason,
            default_mode: ExecutionMode::Parallel,
            signal: None,
            timestamp,
        }
    }

    pub fn with_default_mode(mut self, mode: ExecutionMode) -> Self {
        self.default_mode = mode;
        self
    }

    pub fn with_signal(mut self, signal: Arc<dyn ToolExecutionSignal>) -> Self {
        self.signal = Some(signal);
        self
    }
}

#[derive(Debug)]
pub struct ToolExecutionBatchResult {
    pub messages: Vec<ToolResultMessage>,
    pub terminate: bool,
}

#[derive(Debug, Error)]
pub enum ToolExecutionError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Event(#[from] EventError),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolExecutionStart {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolExecutionEnd {
    pub tool_call_id: String,
    pub tool_name: String,
    pub result: AfterToolCallResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolExecutionUpdate {
    pub tool_call_id: String,
    pub tool_name: String,
    pub update: AgentToolResult,
}

pub fn tool_execution_start_spec() -> EventSpec<ToolExecutionStart, ()> {
    EventSpec::new(
        EventName::new("tools/execution-start").expect("normative event name is valid"),
        DispatchMode::Emit,
        |_| (),
    )
}

pub fn tool_execution_end_spec() -> EventSpec<ToolExecutionEnd, ()> {
    EventSpec::new(
        EventName::new("tools/execution-end").expect("normative event name is valid"),
        DispatchMode::Emit,
        |_| (),
    )
}

pub fn tool_execution_update_spec() -> EventSpec<ToolExecutionUpdate, ()> {
    EventSpec::new(
        EventName::new("tools/execution-update").expect("normative event name is valid"),
        DispatchMode::Emit,
        |_| (),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeforeToolCallContext {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BeforeToolCallAction {
    Proceed(Option<Value>),
    Block(String),
}

#[derive(Clone)]
enum BeforeHookOutcome {
    Proceed(BeforeToolCallContext),
    Blocked(String),
    Failed(String),
}

fn before_tool_call_spec() -> EventSpec<BeforeToolCallContext, BeforeHookOutcome> {
    EventSpec::new(
        EventName::new("tools/pre-execute").expect("normative event name is valid"),
        DispatchMode::Waterfall,
        |current: &BeforeToolCallContext| BeforeHookOutcome::Proceed(current.clone()),
    )
}

pub fn register_before_tool_call_hook<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, EventError>
where
    F: Fn(BeforeToolCallContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<BeforeToolCallAction, super::ToolCapabilityError>> + Send + 'static,
{
    let events = context.events()?;
    let spec = before_tool_call_spec();
    events.declare(&spec)?;
    let effects = context.effect_store();
    events.on_waterfall(&spec, &effects, context.scope(), move |current, next| {
        let future = listener(current.clone());
        async move {
            match future.await {
                Ok(BeforeToolCallAction::Proceed(arguments)) => {
                    let replacement = BeforeToolCallContext {
                        arguments: arguments.unwrap_or(current.arguments),
                        ..current
                    };
                    next.call(Some(replacement)).await
                }
                Ok(BeforeToolCallAction::Block(message)) => Ok(BeforeHookOutcome::Blocked(message)),
                Err(error) => Ok(BeforeHookOutcome::Failed(error.message().to_owned())),
            }
        }
    })
}

/// The accumulated result visible to each post-execute listener.
#[derive(Clone, Debug, PartialEq)]
pub struct AfterToolCallResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ToolResultContentBlock>,
    pub details: Option<Value>,
    pub usage: Option<Usage>,
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
    pub terminate: Option<bool>,
}

/// Pi's constrained successful after-hook replacement surface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterToolCallOverride {
    content: Option<Vec<ToolResultContentBlock>>,
    details: Option<Option<Value>>,
    usage: Option<Option<Usage>>,
    is_error: Option<bool>,
    terminate: Option<Option<bool>>,
}

impl AfterToolCallOverride {
    pub fn with_content(mut self, content: Vec<ToolResultContentBlock>) -> Self {
        self.content = Some(content);
        self
    }

    pub fn with_details(mut self, details: Option<Value>) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_usage(mut self, usage: Option<Usage>) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_is_error(mut self, is_error: bool) -> Self {
        self.is_error = Some(is_error);
        self
    }

    pub fn with_terminate(mut self, terminate: bool) -> Self {
        self.terminate = Some(Some(terminate));
        self
    }

    fn apply(self, mut current: AfterToolCallResult) -> AfterToolCallResult {
        if let Some(content) = self.content {
            current.content = content;
        }
        if let Some(details) = self.details {
            current.details = details;
        }
        if let Some(usage) = self.usage {
            current.usage = usage;
        }
        if let Some(is_error) = self.is_error {
            current.is_error = is_error;
        }
        if let Some(terminate) = self.terminate {
            current.terminate = terminate;
        }
        current
    }
}

#[derive(Clone)]
enum AfterHookOutcome {
    Result(Box<AfterToolCallResult>),
    Failed(String),
}

fn after_tool_call_spec() -> EventSpec<AfterToolCallResult, AfterHookOutcome> {
    EventSpec::new(
        EventName::new("tools/post-execute").expect("normative event name is valid"),
        DispatchMode::Waterfall,
        |current: &AfterToolCallResult| AfterHookOutcome::Result(Box::new(current.clone())),
    )
}

/// Registers a scope/fiber-owned post-execute hook in Runtime registration order.
pub fn register_after_tool_call_hook<F, Fut>(
    context: &Context,
    listener: F,
) -> Result<EventListenerHandle, EventError>
where
    F: Fn(AfterToolCallResult) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Option<AfterToolCallOverride>, super::ToolCapabilityError>>
        + Send
        + 'static,
{
    let events = context.events()?;
    let spec = after_tool_call_spec();
    events.declare(&spec)?;
    let effects = context.effect_store();
    events.on_waterfall(&spec, &effects, context.scope(), move |current, next| {
        let future = listener(current.clone());
        async move {
            match future.await {
                Ok(replacement) => {
                    let replacement = match replacement {
                        Some(value) => value.apply(current),
                        None => current,
                    };
                    next.call(Some(replacement)).await
                }
                Err(error) => Ok(AfterHookOutcome::Failed(error.message().to_owned())),
            }
        }
    })
}

pub async fn execute_tool_calls(
    context: &Context,
    calls: &[ToolCall],
    options: ToolExecutionOptions,
) -> Result<ToolExecutionBatchResult, ToolExecutionError> {
    let registry = context.tools()?;
    let events = context.events()?;
    let start_spec = tool_execution_start_spec();
    let update_spec = tool_execution_update_spec();
    let end_spec = tool_execution_end_spec();
    let before_spec = before_tool_call_spec();
    let after_spec = after_tool_call_spec();
    events.declare(&start_spec)?;
    events.declare(&update_spec)?;
    events.declare(&end_spec)?;
    events.declare(&before_spec)?;
    events.declare(&after_spec)?;
    let scope = context.scope().cloned();
    for call in calls {
        events.emit(
            &start_spec,
            &ToolExecutionStart {
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                arguments: arguments_value(&call.arguments),
            },
            scope.as_ref(),
        )?;
    }

    if options.stop_reason == StopReason::Length {
        let mut messages = Vec::with_capacity(calls.len());
        for call in calls {
            let message = format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                call.name
            );
            let result = immediate_error(call, &message);
            events.emit(
                &end_spec,
                &ToolExecutionEnd {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    result: result.clone(),
                },
                scope.as_ref(),
            )?;
            messages.push(result.into_message(options.timestamp));
        }
        return Ok(ToolExecutionBatchResult {
            messages,
            terminate: false,
        });
    }

    let resolved: Vec<_> = calls
        .iter()
        .map(|call| registry.resolve(&call.name, scope.as_ref()))
        .collect();
    let sequential = options.default_mode == ExecutionMode::Sequential
        || resolved
            .iter()
            .flatten()
            .any(|tool| tool.execution_mode() == Some(ExecutionMode::Sequential));
    let mut indexed = Vec::with_capacity(calls.len());
    if sequential {
        for (index, (call, tool)) in calls.iter().cloned().zip(resolved).enumerate() {
            indexed.push(
                execute_one(
                    index,
                    call,
                    tool,
                    events.clone(),
                    scope.clone(),
                    after_spec.clone(),
                    before_spec.clone(),
                    update_spec.clone(),
                    end_spec.clone(),
                    options.signal.clone(),
                    options.timestamp,
                )
                .await?,
            );
        }
    } else {
        let mut running = FuturesUnordered::new();
        for (index, (call, tool)) in calls.iter().cloned().zip(resolved).enumerate() {
            running.push(
                execute_one(
                    index,
                    call,
                    tool,
                    events.clone(),
                    scope.clone(),
                    after_spec.clone(),
                    before_spec.clone(),
                    update_spec.clone(),
                    end_spec.clone(),
                    options.signal.clone(),
                    options.timestamp,
                )
                .boxed(),
            );
        }
        while let Some(outcome) = running.next().await {
            indexed.push(outcome?);
        }
    }
    indexed.sort_by_key(|(index, _, _)| *index);
    let terminate = !indexed.is_empty() && indexed.iter().all(|(_, _, terminate)| *terminate);
    let messages = indexed.into_iter().map(|(_, message, _)| message).collect();
    Ok(ToolExecutionBatchResult {
        messages,
        terminate,
    })
}

#[allow(clippy::too_many_arguments)]
async fn execute_one(
    index: usize,
    call: ToolCall,
    tool: Option<Arc<ToolDefinition>>,
    events: EventBus,
    scope: Option<ScopeHandle>,
    after_spec: EventSpec<AfterToolCallResult, AfterHookOutcome>,
    before_spec: EventSpec<BeforeToolCallContext, BeforeHookOutcome>,
    update_spec: EventSpec<ToolExecutionUpdate, ()>,
    end_spec: EventSpec<ToolExecutionEnd, ()>,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
    timestamp: f64,
) -> Result<(usize, ToolResultMessage, bool), ToolExecutionError> {
    let executed = match tool {
        None => {
            return finish_immediate(
                index,
                call.clone(),
                &format!("Tool {} not found", call.name),
                events,
                scope,
                end_spec,
                timestamp,
            );
        }
        Some(tool) => {
            let mut params = arguments_value(&call.arguments);
            if let Some(prepare) = tool.prepare_arguments() {
                match prepare(params) {
                    Ok(prepared) => params = prepared,
                    Err(error) => {
                        let message = error.message().to_owned();
                        return finish_immediate(
                            index, call, &message, events, scope, end_spec, timestamp,
                        );
                    }
                }
            }
            let schema = Value::from(tool.schema().parameters);
            let validator = jsonschema::validator_for(&schema)
                .expect("ToolDefinition contains a schema accepted at the shared model boundary");
            if let Err(error) = validator.validate(&params) {
                return finish_immediate(
                    index,
                    call.clone(),
                    &format!("invalid arguments for tool \"{}\": {error}", call.name),
                    events,
                    scope,
                    end_spec,
                    timestamp,
                );
            }
            let before = BeforeToolCallContext {
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                arguments: params,
            };
            let before = match events
                .waterfall(&before_spec, before, scope.as_ref())
                .await?
            {
                BeforeHookOutcome::Proceed(current) => current,
                BeforeHookOutcome::Blocked(message) | BeforeHookOutcome::Failed(message) => {
                    return finish_immediate(
                        index, call, &message, events, scope, end_spec, timestamp,
                    );
                }
            };
            let accepting_updates = Arc::new(AtomicBool::new(true));
            let update_callback = {
                let accepting_updates = Arc::clone(&accepting_updates);
                let events = events.clone();
                let update_spec = update_spec.clone();
                let scope = scope.clone();
                let tool_call_id = call.id.clone();
                let tool_name = call.name.clone();
                Arc::new(move |update: AgentToolResult| {
                    if accepting_updates.load(Ordering::Acquire) {
                        let _ = events.emit(
                            &update_spec,
                            &ToolExecutionUpdate {
                                tool_call_id: tool_call_id.clone(),
                                tool_name: tool_name.clone(),
                                update,
                            },
                            scope.as_ref(),
                        );
                    }
                })
            };
            let request = ToolExecutionRequest {
                tool_call_id: call.id.clone(),
                params: before.arguments,
                signal,
                on_update: Some(update_callback),
            };
            let outcome = (tool.execute())(request).await;
            accepting_updates.store(false, Ordering::Release);
            match outcome {
                Ok(result) => AfterToolCallResult {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    content: result.content,
                    details: (result.details != Value::Null).then_some(result.details),
                    usage: result.usage,
                    added_tool_names: result.added_tool_names,
                    is_error: false,
                    terminate: result.terminate,
                },
                Err(error) => immediate_error(&call, error.message()),
            }
        }
    };
    let finalized = match events
        .waterfall(&after_spec, executed, scope.as_ref())
        .await?
    {
        AfterHookOutcome::Result(result) => *result,
        AfterHookOutcome::Failed(message) => immediate_error(&call, &message),
    };
    events.emit(
        &end_spec,
        &ToolExecutionEnd {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            result: finalized.clone(),
        },
        scope.as_ref(),
    )?;
    let terminate = finalized.terminate.unwrap_or(false);
    Ok((index, finalized.into_message(timestamp), terminate))
}

fn finish_immediate(
    index: usize,
    call: ToolCall,
    message: &str,
    events: EventBus,
    scope: Option<ScopeHandle>,
    end_spec: EventSpec<ToolExecutionEnd, ()>,
    timestamp: f64,
) -> Result<(usize, ToolResultMessage, bool), ToolExecutionError> {
    let result = immediate_error(&call, message);
    events.emit(
        &end_spec,
        &ToolExecutionEnd {
            tool_call_id: call.id,
            tool_name: call.name,
            result: result.clone(),
        },
        scope.as_ref(),
    )?;
    Ok((index, result.into_message(timestamp), false))
}

fn immediate_error(call: &ToolCall, message: &str) -> AfterToolCallResult {
    AfterToolCallResult {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: error_content(message),
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: true,
        terminate: None,
    }
}

fn arguments_value(arguments: &BTreeMap<String, Value>) -> Value {
    Value::Object(arguments.clone().into_iter().collect())
}

fn error_content(message: &str) -> Vec<ToolResultContentBlock> {
    vec![ToolResultContentBlock::Text(TextBlock::new(message))]
}

impl AfterToolCallResult {
    fn into_message(self, timestamp: f64) -> ToolResultMessage {
        let mut message = ToolResultMessage::new(
            self.tool_call_id,
            self.tool_name,
            self.content,
            self.is_error,
            timestamp,
        );
        message.details = self.details;
        message.usage = self.usage;
        message.added_tool_names = self.added_tool_names;
        message
    }
}

#[allow(dead_code)]
fn _scope_type_guard(_: Option<&ScopeHandle>) {}
