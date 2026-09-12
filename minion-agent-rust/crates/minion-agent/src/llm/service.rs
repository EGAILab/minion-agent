use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use parking_lot::RwLock;
use thiserror::Error;

use super::{AssistantMessage, AssistantStream, LlmAdapter, LlmRequest, ModelIdentity};

struct RegisteredAdapter {
    adapter: Arc<dyn LlmAdapter>,
    owner: Arc<()>,
}

type Registry = HashMap<ModelIdentity, RegisteredAdapter>;

/// Resolves strict three-part model identities before creating assistant streams.
///
/// Unknown identities fail eagerly. Once a resolved adapter is invoked, its
/// expected failures settle in the returned stream. Registry locks are never
/// held while adapter code runs.
#[derive(Clone, Default)]
pub struct LlmService {
    adapters: Arc<RwLock<Registry>>,
}

impl LlmService {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one identity and return a repeatable, idempotent withdrawal handle.
    pub fn register(
        &self,
        identity: ModelIdentity,
        adapter: Arc<dyn LlmAdapter>,
    ) -> LlmRegistration {
        self.register_models([identity], adapter)
    }

    /// Register every identity as one ownership unit.
    ///
    /// A later registration replaces an earlier one for the same identity.
    /// Withdrawing a stale handle never removes that later registration.
    pub fn register_models(
        &self,
        identities: impl IntoIterator<Item = ModelIdentity>,
        adapter: Arc<dyn LlmAdapter>,
    ) -> LlmRegistration {
        let owner = Arc::new(());
        let identities: Vec<_> = identities.into_iter().collect();
        {
            let mut entries = self.adapters.write();
            for identity in &identities {
                entries.insert(
                    identity.clone(),
                    RegisteredAdapter {
                        adapter: Arc::clone(&adapter),
                        owner: Arc::clone(&owner),
                    },
                );
            }
        }
        LlmRegistration {
            adapters: Arc::downgrade(&self.adapters),
            identities,
            owner,
        }
    }

    /// Return the currently resolvable full identities in deterministic order.
    pub fn models(&self) -> Vec<ModelIdentity> {
        let mut identities: Vec<_> = self.adapters.read().keys().cloned().collect();
        identities.sort_by(|left, right| {
            (left.provider(), left.api(), left.model_id()).cmp(&(
                right.provider(),
                right.api(),
                right.model_id(),
            ))
        });
        identities
    }

    pub fn stream(&self, request: LlmRequest) -> Result<AssistantStream, LlmStartError> {
        let adapter = self
            .adapters
            .read()
            .get(&request.model)
            .map(|entry| Arc::clone(&entry.adapter))
            .ok_or_else(|| LlmStartError::UnknownModel {
                model: request.model.clone(),
            })?;
        let partial = AssistantMessage::pending(request.model.clone(), 0.0);
        let raw = adapter.start(request);
        Ok(AssistantStream::new(raw, partial))
    }
}

/// Repeatable withdrawal authority for exactly one registration call.
pub struct LlmRegistration {
    adapters: Weak<RwLock<Registry>>,
    identities: Vec<ModelIdentity>,
    owner: Arc<()>,
}

impl LlmRegistration {
    pub fn withdraw(&self) {
        let Some(adapters) = self.adapters.upgrade() else {
            return;
        };
        let mut entries = adapters.write();
        for identity in &self.identities {
            let owned = entries
                .get(identity)
                .is_some_and(|entry| Arc::ptr_eq(&entry.owner, &self.owner));
            if owned {
                entries.remove(identity);
            }
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum LlmStartError {
    #[error("unknown model: {model:?}")]
    UnknownModel { model: ModelIdentity },
}
