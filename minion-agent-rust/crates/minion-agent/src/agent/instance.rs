use std::{collections::BTreeSet, sync::Arc};

use parking_lot::Mutex;
use serde_json::Value;
use thiserror::Error;

use crate::{
    agent_loop::{RunConfig, RunContext, RunSnapshot},
    llm::{Message, ModelIdentity},
    runtime::{Context, RunAbortController, RunSignal, ScopeHandle},
    session::{Session, SessionError},
    tools::ToolDefinition,
};

use super::{AgentDefinition, AgentStatus, Inbox, InboxTarget, InputEnvelope, ThinkingLevel};

#[derive(Debug, Eq, Error, PartialEq)]
pub enum AgentError {
    #[error("Agent is already processing. Wait for completion before resetting.")]
    Active,
    #[error("{0}")]
    Session(String),
}

#[derive(Debug, Eq, Error, PartialEq)]
pub enum AgentRunError {
    #[error("Agent is already processing.")]
    Active,
    #[error(transparent)]
    Session(#[from] SessionError),
}

struct RunEntryRollback {
    streaming_message: Option<Message>,
    error_message: Option<String>,
}

#[derive(Clone)]
struct AgentMutableState {
    system_prompt: String,
    model: ModelIdentity,
    thinking_level: ThinkingLevel,
    status: AgentStatus,
    streaming_message: Option<Message>,
    pending_tool_calls: BTreeSet<String>,
    error_message: Option<String>,
    abort_controller: Option<RunAbortController>,
}

pub struct AgentInstance {
    id: String,
    definition: AgentDefinition,
    session: Session,
    context: Option<Context>,
    scope: Option<ScopeHandle>,
    state: Mutex<AgentMutableState>,
    status_gate: Mutex<()>,
    inbox: Inbox,
}

impl AgentInstance {
    pub fn new(
        id: impl Into<String>,
        definition: AgentDefinition,
        session: Session,
        context: Option<Context>,
        scope: Option<ScopeHandle>,
    ) -> Self {
        let state = AgentMutableState {
            system_prompt: definition.system_prompt.clone(),
            model: definition.model.clone(),
            thinking_level: ThinkingLevel::Off,
            status: AgentStatus::Idle,
            streaming_message: None,
            pending_tool_calls: BTreeSet::new(),
            error_message: None,
            abort_controller: None,
        };
        Self {
            id: id.into(),
            definition,
            session,
            context,
            scope,
            state: Mutex::new(state),
            status_gate: Mutex::new(()),
            inbox: Inbox::new(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn definition(&self) -> &AgentDefinition {
        &self.definition
    }
    pub fn session(&self) -> &Session {
        &self.session
    }
    pub fn inbox(&self) -> &Inbox {
        &self.inbox
    }

    pub fn steer(&self, message: Message, origin: Option<Value>) -> InputEnvelope {
        self.inbox.steer(message, origin)
    }

    pub fn follow_up(&self, message: Message, origin: Option<Value>) -> InputEnvelope {
        self.inbox.follow_up(message, origin)
    }

    pub fn inject(&self, message: Message, origin: Option<Value>) -> InputEnvelope {
        self.inbox.inject(message, origin)
    }

    pub fn clear_steering_queue(&self) {
        self.inbox.clear(InboxTarget::Steering);
    }

    pub fn clear_follow_up_queue(&self) {
        self.inbox.clear(InboxTarget::FollowUp);
    }

    pub fn clear_all_queues(&self) {
        self.inbox.clear_all();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.inbox.has_pending()
    }
    pub fn system_prompt(&self) -> String {
        self.state.lock().system_prompt.clone()
    }
    pub fn model(&self) -> ModelIdentity {
        self.state.lock().model.clone()
    }
    pub fn thinking_level(&self) -> ThinkingLevel {
        self.state.lock().thinking_level
    }
    pub fn status(&self) -> AgentStatus {
        self.state.lock().status
    }
    pub fn streaming_message(&self) -> Option<Message> {
        self.state.lock().streaming_message.clone()
    }
    pub fn pending_tool_calls(&self) -> BTreeSet<String> {
        self.state.lock().pending_tool_calls.clone()
    }
    pub fn error_message(&self) -> Option<String> {
        self.state.lock().error_message.clone()
    }

    /// Returns the read-only cancellation signal for the active run.
    pub fn signal(&self) -> Option<RunSignal> {
        self.state
            .lock()
            .abort_controller
            .as_ref()
            .map(RunAbortController::signal)
    }

    /// Requests cooperative cancellation. Calling this while idle is a no-op.
    pub fn abort(&self) {
        if let Some(controller) = self.state.lock().abort_controller.as_ref() {
            controller.abort();
        }
    }

    pub fn set_system_prompt(&self, value: impl Into<String>) {
        self.state.lock().system_prompt = value.into();
    }
    pub fn set_model(&self, value: ModelIdentity) {
        self.state.lock().model = value;
    }
    pub fn set_thinking_level(&self, value: ThinkingLevel) {
        self.state.lock().thinking_level = value;
    }
    pub fn set_streaming_message(&self, value: Option<Message>) {
        self.state.lock().streaming_message = value;
    }
    pub fn set_pending_tool_calls(&self, value: BTreeSet<String>) {
        self.state.lock().pending_tool_calls = value;
    }
    pub fn set_error_message(&self, value: Option<String>) {
        self.state.lock().error_message = value;
    }

    pub fn set_status(&self, status: AgentStatus) {
        let _gate = self.status_gate.lock();
        self.state.lock().status = status;
    }

    pub fn try_begin_run(&self) -> Result<RunSnapshot, AgentRunError> {
        let (system_prompt, model, thinking_level, signal, rollback) = {
            let _gate = self.status_gate.lock();
            let mut state = self.state.lock();
            if state.status != AgentStatus::Idle {
                return Err(AgentRunError::Active);
            }
            let rollback = RunEntryRollback {
                streaming_message: state.streaming_message.clone(),
                error_message: state.error_message.clone(),
            };
            let controller = RunAbortController::default();
            let signal = controller.signal();
            state.abort_controller = Some(controller);
            state.status = AgentStatus::Running;
            state.streaming_message = None;
            state.error_message = None;
            (
                state.system_prompt.clone(),
                state.model.clone(),
                state.thinking_level,
                signal,
                rollback,
            )
        };

        let messages = match self.session.derive_messages() {
            Ok(messages) => messages,
            Err(error) => {
                self.rollback_run_entry(rollback);
                return Err(error.into());
            }
        };
        let tools = self.tools();

        Ok(RunSnapshot {
            context: RunContext {
                system_prompt,
                messages,
                tools,
            },
            config: RunConfig {
                model,
                thinking_level,
            },
            signal,
        })
    }

    pub fn finish_run(&self) {
        let _gate = self.status_gate.lock();
        let mut state = self.state.lock();
        state.abort_controller = None;
        state.status = AgentStatus::Idle;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
    }

    pub(crate) fn add_pending_tool_call(&self, tool_call_id: String) {
        self.state.lock().pending_tool_calls.insert(tool_call_id);
    }

    pub(crate) fn remove_pending_tool_call(&self, tool_call_id: &str) {
        self.state.lock().pending_tool_calls.remove(tool_call_id);
    }

    fn rollback_run_entry(&self, rollback: RunEntryRollback) {
        let _gate = self.status_gate.lock();
        let mut state = self.state.lock();
        state.abort_controller = None;
        state.status = AgentStatus::Idle;
        state.streaming_message = rollback.streaming_message;
        state.error_message = rollback.error_message;
    }

    pub fn messages(&self) -> Result<Vec<Message>, AgentError> {
        self.session
            .derive_messages()
            .map_err(|error| AgentError::Session(error.to_string()))
    }

    pub fn tools(&self) -> Vec<Arc<ToolDefinition>> {
        self.context
            .as_ref()
            .and_then(|context| context.tools().ok())
            .map_or_else(Vec::new, |registry| registry.visible(self.scope.as_ref()))
    }

    pub fn reset(&self) -> Result<(), AgentError> {
        let _gate = self.status_gate.lock();
        if self.state.lock().status != AgentStatus::Idle {
            return Err(AgentError::Active);
        }
        self.session
            .reset()
            .map_err(|error| AgentError::Session(error.to_string()))?;
        {
            let mut state = self.state.lock();
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message = None;
        }
        self.inbox.clear_all();
        Ok(())
    }
}
