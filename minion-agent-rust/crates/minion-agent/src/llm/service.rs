use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;
use thiserror::Error;

use super::{
    AdapterStartError, AssistantMessage, AssistantStream, LlmAdapter, LlmRequest, ModelIdentity,
};

/// Resolves strict three-part model identities before creating assistant streams.
///
/// Lookup and adapter-start failures are eager typed errors. The adapter is
/// cloned out of the registry before user/provider code runs, so no registry
/// lock is held while starting a stream.
#[derive(Default)]
pub struct LlmService {
    adapters: RwLock<HashMap<ModelIdentity, Arc<dyn LlmAdapter>>>,
}

impl LlmService {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, identity: ModelIdentity, adapter: Arc<dyn LlmAdapter>) {
        self.adapters.write().insert(identity, adapter);
    }

    pub fn stream(&self, request: LlmRequest) -> Result<AssistantStream, LlmStartError> {
        let adapter = self
            .adapters
            .read()
            .get(&request.model)
            .cloned()
            .ok_or_else(|| LlmStartError::UnknownModel {
                model: request.model.clone(),
            })?;
        let partial = AssistantMessage::pending(request.model.clone(), 0.0);
        let raw = adapter
            .start(request)
            .map_err(LlmStartError::AdapterStart)?;
        Ok(AssistantStream::new(raw, partial))
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum LlmStartError {
    #[error("unknown model: {model:?}")]
    UnknownModel { model: ModelIdentity },
    #[error(transparent)]
    AdapterStart(#[from] AdapterStartError),
}
