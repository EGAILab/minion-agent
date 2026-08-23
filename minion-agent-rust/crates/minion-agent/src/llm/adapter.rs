use std::pin::Pin;

use futures::Stream;
use thiserror::Error;

use super::{LlmRequest, StreamChunk};

pub type RawAssistantStream =
    Pin<Box<dyn Stream<Item = Result<StreamChunk, AdapterStreamError>> + Send>>;

/// Provider-specific stream creation and decoding.
///
/// Returning [`AdapterStartError`] is an eager failure because no stream exists.
/// Once returned, expected operational failures use [`AdapterStreamError`].
/// Implementations must not duplicate Minion terminal fusion or premature-EOF
/// settlement; [`crate::llm::AssistantStream`] owns those rules.
pub trait LlmAdapter: Send + Sync {
    fn start(&self, request: LlmRequest) -> Result<RawAssistantStream, AdapterStartError>;
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum AdapterStartError {
    #[error("adapter rejected request before stream creation: {0}")]
    Rejected(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdapterStreamErrorKind {
    Provider,
    Network,
    Model,
    Cancelled,
    Protocol,
    Runtime,
}

#[derive(Clone, Debug, Error, PartialEq)]
#[error("{kind:?}: {message}")]
pub struct AdapterStreamError {
    pub kind: AdapterStreamErrorKind,
    pub message: String,
}

impl AdapterStreamError {
    pub fn new(kind: AdapterStreamErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}
