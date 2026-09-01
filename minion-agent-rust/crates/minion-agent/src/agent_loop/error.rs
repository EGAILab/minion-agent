use thiserror::Error;

use crate::{EventError, RuntimeError, agent::AgentRunError, session::SessionError};

/// Failure returned by a public Agent lifecycle listener.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("agent listener failed: {message}")]
pub struct AgentListenerError {
    message: String,
}

impl AgentListenerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Error)]
pub enum AgentLoopError {
    #[error(
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
    )]
    PromptActive,
    #[error("Agent is already processing. Wait for completion before continuing.")]
    ContinueActive,
    #[error("No messages to continue from")]
    NoMessagesToContinue,
    #[error("Cannot continue from message role: assistant")]
    CannotContinueFromAssistant,
    #[error(transparent)]
    Run(#[from] AgentRunError),
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Event(#[from] EventError),
    #[error(transparent)]
    Listener(#[from] AgentListenerError),
}

impl AgentLoopError {
    pub fn listener_error(&self) -> Option<&AgentListenerError> {
        match self {
            Self::Listener(error) => Some(error),
            Self::PromptActive
            | Self::ContinueActive
            | Self::NoMessagesToContinue
            | Self::CannotContinueFromAssistant
            | Self::Run(_)
            | Self::Session(_)
            | Self::Runtime(_)
            | Self::Event(_) => None,
        }
    }
}
