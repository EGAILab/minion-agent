use std::{
    collections::BTreeMap,
    future::Future,
    mem,
    sync::Arc,
    task::{Context as TaskContext, Poll},
};

use futures::{
    FutureExt, StreamExt,
    future::{BoxFuture, poll_fn},
    stream::FuturesUnordered,
    task::{AtomicWaker, noop_waker_ref},
};
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
    execution_tools: Option<Vec<Arc<ToolDefinition>>>,
    on_execution_start: Option<ToolExecutionStartCallback>,
    on_execution_update: Option<ToolExecutionUpdateCallback>,
    on_execution_end: Option<ToolExecutionEndCallback>,
}

impl ToolExecutionOptions {
    pub fn new(stop_reason: StopReason, timestamp: f64) -> Self {
        Self {
            stop_reason,
            default_mode: ExecutionMode::Parallel,
            signal: None,
            timestamp,
            execution_tools: None,
            on_execution_start: None,
            on_execution_update: None,
            on_execution_end: None,
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

    /// Uses an already-resolved run-local tool snapshot for this batch.
    ///
    /// The default remains the live Layer-05 registry so existing Layer-06
    /// callers retain their certified behavior.
    pub fn with_execution_tools(mut self, tools: Vec<Arc<ToolDefinition>>) -> Self {
        self.execution_tools = Some(tools);
        self
    }

    pub fn with_execution_start<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(ToolExecutionStart) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), ToolLifecycleError>> + Send + 'static,
    {
        self.on_execution_start = Some(Arc::new(move |event| callback(event).boxed()));
        self
    }

    pub fn with_execution_update<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(ToolExecutionUpdate) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), ToolLifecycleError>> + Send + 'static,
    {
        self.on_execution_update = Some(Arc::new(move |event| callback(event).boxed()));
        self
    }

    pub fn with_execution_end<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(ToolExecutionEnd) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), ToolLifecycleError>> + Send + 'static,
    {
        self.on_execution_end = Some(Arc::new(move |event| callback(event).boxed()));
        self
    }
}

pub type ToolExecutionStartCallback = Arc<
    dyn Fn(ToolExecutionStart) -> BoxFuture<'static, Result<(), ToolLifecycleError>>
        + Send
        + Sync
        + 'static,
>;
pub type ToolExecutionUpdateCallback = Arc<
    dyn Fn(ToolExecutionUpdate) -> BoxFuture<'static, Result<(), ToolLifecycleError>>
        + Send
        + Sync
        + 'static,
>;
pub type ToolExecutionEndCallback = Arc<
    dyn Fn(ToolExecutionEnd) -> BoxFuture<'static, Result<(), ToolLifecycleError>>
        + Send
        + Sync
        + 'static,
>;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct ToolLifecycleError {
    message: String,
}

impl ToolLifecycleError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
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
    #[error(transparent)]
    Lifecycle(#[from] ToolLifecycleError),
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
    pub arguments: Value,
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
    details: Option<Value>,
    usage: Option<Usage>,
    is_error: Option<bool>,
    terminate: Option<bool>,
}

struct PreparedToolCall {
    index: usize,
    call: ToolCall,
    tool: Arc<ToolDefinition>,
    arguments: Value,
}

struct UpdateDispatchState {
    accepting: bool,
    reservations: usize,
    pending: Vec<BoxFuture<'static, Result<(), ToolLifecycleError>>>,
    first_error: Option<ToolLifecycleError>,
}

impl Default for UpdateDispatchState {
    fn default() -> Self {
        Self {
            accepting: true,
            reservations: 0,
            pending: Vec::new(),
            first_error: None,
        }
    }
}

struct LiveUpdateDispatches {
    state: parking_lot::Mutex<UpdateDispatchState>,
    waker: AtomicWaker,
}

struct UpdateDispatchReservation {
    owner: Arc<LiveUpdateDispatches>,
    completed: bool,
}

impl UpdateDispatchReservation {
    fn dispatch(mut self, mut future: BoxFuture<'static, Result<(), ToolLifecycleError>>) {
        let mut task_context = TaskContext::from_waker(noop_waker_ref());
        let result = future.as_mut().poll(&mut task_context);
        self.owner.finish_reservation(result, future);
        self.completed = true;
    }
}

impl Drop for UpdateDispatchReservation {
    fn drop(&mut self) {
        if !self.completed {
            self.owner.finish_empty_reservation();
        }
    }
}

impl LiveUpdateDispatches {
    fn new() -> Self {
        Self {
            state: parking_lot::Mutex::new(UpdateDispatchState::default()),
            waker: AtomicWaker::new(),
        }
    }

    fn reserve(self: &Arc<Self>) -> Option<UpdateDispatchReservation> {
        let mut state = self.state.lock();
        if !state.accepting {
            return None;
        }
        state.reservations += 1;
        Some(UpdateDispatchReservation {
            owner: Arc::clone(self),
            completed: false,
        })
    }

    fn close(&self) {
        self.state.lock().accepting = false;
        self.waker.wake();
    }

    fn is_complete(&self) -> bool {
        let state = self.state.lock();
        state.reservations == 0 && state.pending.is_empty()
    }

    fn poll_once(&self, task_context: &mut TaskContext<'_>) {
        let mut current = {
            let mut state = self.state.lock();
            mem::take(&mut state.pending)
        };
        let mut still_pending = Vec::with_capacity(current.len());
        for mut future in current.drain(..) {
            match future.as_mut().poll(task_context) {
                Poll::Ready(result) => self.record_result(result),
                Poll::Pending => still_pending.push(future),
            }
        }
        let mut state = self.state.lock();
        still_pending.append(&mut state.pending);
        state.pending = still_pending;
    }

    fn finish_reservation(
        &self,
        result: Poll<Result<(), ToolLifecycleError>>,
        future: BoxFuture<'static, Result<(), ToolLifecycleError>>,
    ) {
        let mut state = self.state.lock();
        state.reservations -= 1;
        match result {
            Poll::Ready(Ok(())) => {}
            Poll::Ready(Err(error)) => {
                if state.first_error.is_none() {
                    state.first_error = Some(error);
                }
            }
            Poll::Pending => state.pending.push(future),
        }
        drop(state);
        self.waker.wake();
    }

    fn finish_empty_reservation(&self) {
        self.state.lock().reservations -= 1;
        self.waker.wake();
    }

    fn record_result(&self, result: Result<(), ToolLifecycleError>) {
        if let Err(error) = result {
            let mut state = self.state.lock();
            if state.first_error.is_none() {
                state.first_error = Some(error);
            }
        }
    }

    fn take_error(&self) -> Option<ToolLifecycleError> {
        self.state.lock().first_error.take()
    }
}

enum PreflightOutcome {
    Immediate((usize, ToolResultMessage, bool)),
    Prepared(PreparedToolCall),
}

impl AfterToolCallOverride {
    pub fn with_content(mut self, content: Vec<ToolResultContentBlock>) -> Self {
        self.content = Some(content);
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_is_error(mut self, is_error: bool) -> Self {
        self.is_error = Some(is_error);
        self
    }

    pub fn with_terminate(mut self, terminate: bool) -> Self {
        self.terminate = Some(terminate);
        self
    }

    fn apply(self, mut current: AfterToolCallResult) -> AfterToolCallResult {
        if let Some(content) = self.content {
            current.content = content;
        }
        if let Some(details) = self.details {
            current.details = Some(details);
        }
        if let Some(usage) = self.usage {
            current.usage = Some(usage);
        }
        if let Some(is_error) = self.is_error {
            current.is_error = is_error;
        }
        if let Some(terminate) = self.terminate {
            current.terminate = Some(terminate);
        }
        current
    }
}

/// Public Runtime waterfall event for post-execute listeners. Raw listeners
/// may return a whole result; production dispatch constrains every handoff.
pub fn after_tool_call_spec() -> EventSpec<AfterToolCallResult, AfterToolCallResult> {
    EventSpec::new(
        EventName::new("tools/post-execute").expect("normative event name is valid"),
        DispatchMode::Waterfall,
        |current: &AfterToolCallResult| current.clone(),
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
                Err(error) => Err(crate::runtime::WaterfallError::ListenerFailed(
                    error.message().to_owned(),
                )),
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
    if options.stop_reason == StopReason::Length {
        let mut messages = Vec::with_capacity(calls.len());
        for call in calls {
            emit_start(
                events,
                &start_spec,
                call,
                scope.as_ref(),
                options.on_execution_start.as_ref(),
            )
            .await?;
            let message = format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                call.name
            );
            let result = immediate_error(call, &message);
            emit_end(
                events,
                &end_spec,
                ToolExecutionEnd {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    result: result.clone(),
                },
                scope.as_ref(),
                options.on_execution_end.as_ref(),
            )
            .await?;
            messages.push(result.into_message(options.timestamp));
        }
        return Ok(ToolExecutionBatchResult {
            messages,
            terminate: false,
        });
    }

    let resolved: Vec<_> = calls
        .iter()
        .map(|call| match options.execution_tools.as_ref() {
            Some(tools) => tools.iter().find(|tool| tool.name() == call.name).cloned(),
            None => registry.resolve(&call.name, scope.as_ref()),
        })
        .collect();
    let sequential = options.default_mode == ExecutionMode::Sequential
        || resolved
            .iter()
            .flatten()
            .any(|tool| tool.execution_mode() == Some(ExecutionMode::Sequential));
    let mut indexed = Vec::with_capacity(calls.len());
    if sequential {
        for (index, call) in calls.iter().cloned().enumerate() {
            emit_start(
                events,
                &start_spec,
                &call,
                scope.as_ref(),
                options.on_execution_start.as_ref(),
            )
            .await?;
            let tool = resolved[index].clone();
            match preflight_one(
                index,
                call,
                tool,
                events.clone(),
                scope.clone(),
                before_spec.clone(),
                end_spec.clone(),
                options.on_execution_end.clone(),
                options.timestamp,
            )
            .await?
            {
                PreflightOutcome::Immediate(outcome) => indexed.push(outcome),
                PreflightOutcome::Prepared(prepared) => {
                    indexed.push(
                        execute_and_finalize_prepared(
                            prepared,
                            events.clone(),
                            scope.clone(),
                            after_spec.clone(),
                            update_spec.clone(),
                            end_spec.clone(),
                            options.signal.clone(),
                            options.on_execution_update.clone(),
                            options.on_execution_end.clone(),
                            options.timestamp,
                        )
                        .await?,
                    );
                }
            }
        }
    } else {
        let mut prepared = Vec::new();
        for (index, call) in calls.iter().cloned().enumerate() {
            emit_start(
                events,
                &start_spec,
                &call,
                scope.as_ref(),
                options.on_execution_start.as_ref(),
            )
            .await?;
            let tool = resolved[index].clone();
            match preflight_one(
                index,
                call,
                tool,
                events.clone(),
                scope.clone(),
                before_spec.clone(),
                end_spec.clone(),
                options.on_execution_end.clone(),
                options.timestamp,
            )
            .await?
            {
                PreflightOutcome::Immediate(outcome) => indexed.push(outcome),
                PreflightOutcome::Prepared(call) => prepared.push(call),
            }
        }
        let mut running = FuturesUnordered::new();
        for prepared in prepared {
            running.push(
                execute_and_finalize_prepared(
                    prepared,
                    events.clone(),
                    scope.clone(),
                    after_spec.clone(),
                    update_spec.clone(),
                    end_spec.clone(),
                    options.signal.clone(),
                    options.on_execution_update.clone(),
                    options.on_execution_end.clone(),
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
async fn preflight_one(
    index: usize,
    call: ToolCall,
    tool: Option<Arc<ToolDefinition>>,
    events: EventBus,
    scope: Option<ScopeHandle>,
    before_spec: EventSpec<BeforeToolCallContext, BeforeHookOutcome>,
    end_spec: EventSpec<ToolExecutionEnd, ()>,
    on_execution_end: Option<ToolExecutionEndCallback>,
    timestamp: f64,
) -> Result<PreflightOutcome, ToolExecutionError> {
    let tool = match tool {
        None => {
            return Ok(PreflightOutcome::Immediate(
                finish_immediate(
                    index,
                    call.clone(),
                    &format!("Tool {} not found", call.name),
                    events,
                    scope,
                    end_spec,
                    on_execution_end,
                    timestamp,
                )
                .await?,
            ));
        }
        Some(tool) => tool,
    };
    let mut params = arguments_value(&call.arguments);
    if let Some(prepare) = tool.prepare_arguments() {
        match prepare(params) {
            Ok(prepared) => params = prepared,
            Err(error) => {
                let message = error.message().to_owned();
                return Ok(PreflightOutcome::Immediate(
                    finish_immediate(
                        index,
                        call,
                        &message,
                        events,
                        scope,
                        end_spec,
                        on_execution_end,
                        timestamp,
                    )
                    .await?,
                ));
            }
        }
    }
    let schema = Value::from(tool.schema().parameters);
    let validator = match jsonschema::validator_for(&schema) {
        Ok(validator) => validator,
        Err(error) => {
            return Ok(PreflightOutcome::Immediate(
                finish_immediate(
                    index,
                    call.clone(),
                    &format!(
                        "invalid arguments for tool \"{}\": invalid schema: {error}",
                        call.name
                    ),
                    events,
                    scope,
                    end_spec,
                    on_execution_end,
                    timestamp,
                )
                .await?,
            ));
        }
    };
    if let Err(error) = validator.validate(&params) {
        return Ok(PreflightOutcome::Immediate(
            finish_immediate(
                index,
                call.clone(),
                &format!("invalid arguments for tool \"{}\": {error}", call.name),
                events,
                scope,
                end_spec,
                on_execution_end,
                timestamp,
            )
            .await?,
        ));
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
            return Ok(PreflightOutcome::Immediate(
                finish_immediate(
                    index,
                    call,
                    &message,
                    events,
                    scope,
                    end_spec,
                    on_execution_end,
                    timestamp,
                )
                .await?,
            ));
        }
    };
    Ok(PreflightOutcome::Prepared(PreparedToolCall {
        index,
        call,
        tool,
        arguments: before.arguments,
    }))
}

#[allow(clippy::too_many_arguments)]
async fn execute_and_finalize_prepared(
    prepared: PreparedToolCall,
    events: EventBus,
    scope: Option<ScopeHandle>,
    after_spec: EventSpec<AfterToolCallResult, AfterToolCallResult>,
    update_spec: EventSpec<ToolExecutionUpdate, ()>,
    end_spec: EventSpec<ToolExecutionEnd, ()>,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
    on_execution_update: Option<ToolExecutionUpdateCallback>,
    on_execution_end: Option<ToolExecutionEndCallback>,
    timestamp: f64,
) -> Result<(usize, ToolResultMessage, bool), ToolExecutionError> {
    let PreparedToolCall {
        index,
        call,
        tool,
        arguments,
    } = prepared;
    let executed = {
        let live_updates = Arc::new(LiveUpdateDispatches::new());
        let update_callback = {
            let live_updates = Arc::clone(&live_updates);
            let events = events.clone();
            let update_spec = update_spec.clone();
            let scope = scope.clone();
            let on_execution_update = on_execution_update.clone();
            let tool_call_id = call.id.clone();
            let tool_name = call.name.clone();
            let original_arguments = arguments_value(&call.arguments);
            Arc::new(move |update: AgentToolResult| {
                let Some(reservation) = live_updates.reserve() else {
                    return;
                };
                let event = ToolExecutionUpdate {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    arguments: original_arguments.clone(),
                    update,
                };
                let _ = events.emit(&update_spec, &event, scope.as_ref());
                if let Some(callback) = on_execution_update.as_ref() {
                    reservation.dispatch(callback(event));
                }
            })
        };
        let request = ToolExecutionRequest {
            tool_call_id: call.id.clone(),
            params: arguments,
            signal,
            on_update: Some(update_callback),
        };
        let outcome =
            execute_with_live_updates((tool.execute())(request), Arc::clone(&live_updates)).await?;
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
    };
    let protected = executed.clone();
    let normalization_authority = protected.clone();
    let normalization_state = Arc::new(parking_lot::Mutex::new(executed.clone()));
    let step_state = Arc::clone(&normalization_state);
    let finalized = match events
        .waterfall_normalized(&after_spec, executed, scope.as_ref(), move |candidate| {
            let mut previous = step_state.lock();
            let normalized =
                normalize_successful_after_result(candidate, &previous, &normalization_authority);
            previous.clone_from(&normalized);
            normalized
        })
        .await
    {
        Ok(finalized) => {
            normalize_successful_after_result(finalized, &normalization_state.lock(), &protected)
        }
        Err(EventError::Waterfall(error)) => immediate_error(&call, &error.to_string()),
        Err(error) => return Err(error.into()),
    };
    emit_end(
        &events,
        &end_spec,
        ToolExecutionEnd {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            result: finalized.clone(),
        },
        scope.as_ref(),
        on_execution_end.as_ref(),
    )
    .await?;
    let terminate = finalized.terminate.unwrap_or(false);
    Ok((index, finalized.into_message(timestamp), terminate))
}

async fn execute_with_live_updates(
    mut execution: BoxFuture<'static, Result<AgentToolResult, super::ToolCapabilityError>>,
    updates: Arc<LiveUpdateDispatches>,
) -> Result<Result<AgentToolResult, super::ToolCapabilityError>, ToolLifecycleError> {
    let mut outcome = None;
    poll_fn(|task_context| {
        updates.waker.register(task_context.waker());
        if outcome.is_none()
            && let Poll::Ready(result) = execution.as_mut().poll(task_context)
        {
            updates.close();
            outcome = Some(result);
        }
        updates.poll_once(task_context);
        if outcome.is_some() && updates.is_complete() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;
    if let Some(error) = updates.take_error() {
        return Err(error);
    }
    Ok(outcome.expect("execution is complete when update dispatches settle"))
}

async fn emit_start(
    events: &EventBus,
    start_spec: &EventSpec<ToolExecutionStart, ()>,
    call: &ToolCall,
    scope: Option<&ScopeHandle>,
    callback: Option<&ToolExecutionStartCallback>,
) -> Result<(), ToolExecutionError> {
    let event = ToolExecutionStart {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        arguments: arguments_value(&call.arguments),
    };
    events.emit(start_spec, &event, scope)?;
    if let Some(callback) = callback {
        callback(event).await?;
    }
    Ok(())
}

async fn emit_end(
    events: &EventBus,
    end_spec: &EventSpec<ToolExecutionEnd, ()>,
    event: ToolExecutionEnd,
    scope: Option<&ScopeHandle>,
    callback: Option<&ToolExecutionEndCallback>,
) -> Result<(), ToolExecutionError> {
    events.emit(end_spec, &event, scope)?;
    if let Some(callback) = callback {
        callback(event).await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn finish_immediate(
    index: usize,
    call: ToolCall,
    message: &str,
    events: EventBus,
    scope: Option<ScopeHandle>,
    end_spec: EventSpec<ToolExecutionEnd, ()>,
    on_execution_end: Option<ToolExecutionEndCallback>,
    timestamp: f64,
) -> Result<(usize, ToolResultMessage, bool), ToolExecutionError> {
    let result = immediate_error(&call, message);
    emit_end(
        &events,
        &end_spec,
        ToolExecutionEnd {
            tool_call_id: call.id,
            tool_name: call.name,
            result: result.clone(),
        },
        scope.as_ref(),
        on_execution_end.as_ref(),
    )
    .await?;
    Ok((index, result.into_message(timestamp), false))
}

fn immediate_error(call: &ToolCall, message: &str) -> AfterToolCallResult {
    AfterToolCallResult {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: error_content(message),
        details: empty_error_details(),
        usage: None,
        added_tool_names: None,
        is_error: true,
        terminate: None,
    }
}

fn empty_error_details() -> Option<Value> {
    Some(Value::Object(serde_json::Map::new()))
}

fn restore_protected(
    mut candidate: AfterToolCallResult,
    authoritative: &AfterToolCallResult,
) -> AfterToolCallResult {
    candidate
        .tool_call_id
        .clone_from(&authoritative.tool_call_id);
    candidate.tool_name.clone_from(&authoritative.tool_name);
    candidate
        .added_tool_names
        .clone_from(&authoritative.added_tool_names);
    candidate
}

fn normalize_successful_after_result(
    mut candidate: AfterToolCallResult,
    previous: &AfterToolCallResult,
    authoritative: &AfterToolCallResult,
) -> AfterToolCallResult {
    if candidate.details.is_none() {
        candidate.details.clone_from(&previous.details);
    }
    if candidate.usage.is_none() {
        candidate.usage.clone_from(&previous.usage);
    }
    if candidate.terminate.is_none() {
        candidate.terminate = previous.terminate;
    }
    restore_protected(candidate, authoritative)
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

#[cfg(test)]
mod live_update_tests {
    use std::sync::Arc;

    use super::LiveUpdateDispatches;

    #[test]
    fn an_update_reserved_before_execute_settles_keeps_the_join_open() {
        let updates = Arc::new(LiveUpdateDispatches::new());
        let reservation = updates.reserve().expect("updates initially accepted");

        updates.close();

        assert!(!updates.is_complete());
        drop(reservation);
        assert!(updates.is_complete());
        assert!(updates.reserve().is_none());
    }
}
