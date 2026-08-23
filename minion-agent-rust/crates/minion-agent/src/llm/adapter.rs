use std::pin::Pin;

use futures::Stream;
use thiserror::Error;

use super::{LlmRequest, StreamChunk};

pub type RawAssistantStream =
    Pin<Box<dyn Stream<Item = Result<StreamChunk, AdapterStreamError>> + Send>>;

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
