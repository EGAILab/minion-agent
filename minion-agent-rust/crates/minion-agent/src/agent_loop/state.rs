use crate::{agent::AgentInstance, llm::Message, runtime::RunSignal};

use super::{AgentEvent, AgentLoopError, RunConfig, RunContext};

#[derive(Clone, Debug)]
pub struct RunSnapshot {
    pub context: RunContext,
    pub config: RunConfig,
    pub signal: RunSignal,
}

/// Applies the Pi-equivalent state transition for one lifecycle event.
///
/// This function never dispatches listeners. Every Agent state lock is
/// released before the Session append performed for `MessageEnd`.
pub fn reduce_event(agent: &AgentInstance, event: &AgentEvent) -> Result<(), AgentLoopError> {
    match event {
        AgentEvent::MessageStart(message) => {
            agent.set_streaming_message(Some(message.clone()));
        }
        AgentEvent::MessageUpdate { partial, .. } => {
            agent.set_streaming_message(Some(Message::Assistant(Box::new(partial.clone()))));
        }
        AgentEvent::MessageEnd(message) => {
            agent.set_streaming_message(None);
            agent.session().append_message(message.clone())?;
        }
        AgentEvent::ToolExecutionStart(start) => {
            agent.add_pending_tool_call(start.tool_call_id.clone());
        }
        AgentEvent::ToolExecutionEnd(end) => {
            agent.remove_pending_tool_call(&end.tool_call_id);
        }
        AgentEvent::TurnEnd { message, .. } => {
            if let Some(error_message) = message
                .error_message
                .as_ref()
                .filter(|message| !message.is_empty())
            {
                agent.set_error_message(Some(error_message.clone()));
            }
        }
        AgentEvent::AgentEnd { .. } => {
            agent.set_streaming_message(None);
        }
        AgentEvent::AgentStart { .. }
        | AgentEvent::TurnStart
        | AgentEvent::ToolExecutionUpdate(_) => {}
    }
    Ok(())
}
