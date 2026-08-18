use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use thiserror::Error;

use super::RuntimeError;

type Disposer = Box<dyn FnOnce() -> BoxFuture<'static, Result<(), DisposeError>> + Send>;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("disposer {label} failed: {message}")]
pub struct DisposeError {
    pub label: String,
    pub message: String,
}

impl DisposeError {
    pub fn new(label: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Error)]
#[error("errors while disposing")]
pub struct DisposeErrors(Vec<DisposeError>);

impl DisposeErrors {
    pub fn as_slice(&self) -> &[DisposeError] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<DisposeError> {
        self.0
    }
}

struct EffectSlot {
    label: String,
    disposer: Mutex<Option<Disposer>>,
}

impl EffectSlot {
    async fn dispose(&self) -> Result<(), DisposeError> {
        let disposer = self.disposer.lock().take();
        let Some(disposer) = disposer else {
            return Ok(());
        };

        disposer().await.map_err(|mut error| {
            error.label.clone_from(&self.label);
            error
        })
    }
}

struct EffectState {
    accepting: bool,
    entries: Vec<Arc<EffectSlot>>,
}

pub struct EffectStore {
    state: Mutex<EffectState>,
}

impl EffectStore {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(EffectState {
                accepting: true,
                entries: Vec::new(),
            }),
        }
    }

    pub fn push<F>(
        &self,
        label: impl Into<String>,
        disposer: F,
    ) -> Result<EffectHandle, RuntimeError>
    where
        F: FnOnce() -> BoxFuture<'static, Result<(), DisposeError>> + Send + 'static,
    {
        let mut state = self.state.lock();
        if !state.accepting {
            return Err(RuntimeError::InactiveOwner {
                owner: "effect store".to_owned(),
            });
        }

        let slot = Arc::new(EffectSlot {
            label: label.into(),
            disposer: Mutex::new(Some(Box::new(disposer))),
        });
        state.entries.push(Arc::clone(&slot));
        Ok(EffectHandle { slot })
    }

    pub fn close(&self) {
        self.close_entries();
    }

    pub async fn close_and_dispose(&self) -> Result<(), DisposeErrors> {
        let entries = self.close_entries();
        let mut errors = Vec::new();
        for entry in entries {
            if let Err(error) = entry.dispose().await {
                errors.push(error);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(DisposeErrors(errors))
        }
    }

    fn close_entries(&self) -> Vec<Arc<EffectSlot>> {
        let mut state = self.state.lock();
        state.accepting = false;
        state.entries.iter().rev().cloned().collect()
    }
}

impl Default for EffectStore {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EffectHandle {
    slot: Arc<EffectSlot>,
}

impl EffectHandle {
    pub async fn dispose(&self) -> Result<(), DisposeError> {
        self.slot.dispose().await
    }
}
