use thiserror::Error;

use crate::{EventError, RuntimeError};

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
            Self::Runtime(_) | Self::Event(_) => None,
        }
    }
}
