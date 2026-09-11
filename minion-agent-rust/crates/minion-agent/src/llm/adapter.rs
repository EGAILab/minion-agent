use std::pin::Pin;

use futures::Stream;
use thiserror::Error;

use super::{LlmRequest, StreamChunk};

pub type RawAssistantStream =
    Pin<Box<dyn Stream<Item = Result<StreamChunk, AdapterStreamError>> + Send>>;

/// Provider-specific stream creation and decoding.
///
/// Once invoked for a resolved model, expected request/provider/runtime failures
/// use [`AdapterStreamError`] in the returned stream. There is deliberately no
/// eager expected-error channel at this boundary.
/// Implementations must not duplicate Minion terminal fusion or premature-EOF
/// settlement; [`crate::llm::AssistantStream`] owns those rules.
pub trait LlmAdapter: Send + Sync {
    fn start(&self, request: LlmRequest) -> RawAssistantStream;
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
