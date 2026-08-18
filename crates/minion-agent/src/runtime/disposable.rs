use std::sync::{Arc, Weak};

use futures::future::{BoxFuture, FutureExt, Shared};
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

#[derive(Clone, Debug, Error)]
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
    in_flight_unwind: Option<InFlightUnwind>,
    next_generation: u64,
}

type BulkUnwind = Shared<BoxFuture<'static, Result<(), DisposeErrors>>>;

struct InFlightUnwind {
    generation: u64,
    unwind: BulkUnwind,
}

pub struct EffectStore {
    state: Arc<Mutex<EffectState>>,
}

impl EffectStore {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(EffectState {
                accepting: true,
                entries: Vec::new(),
                in_flight_unwind: None,
                next_generation: 0,
            })),
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
        self.state.lock().accepting = false;
    }

    pub async fn close_and_dispose(&self) -> Result<(), DisposeErrors> {
        let unwind = {
            let mut state = self.state.lock();
            state.accepting = false;
            if let Some(unwind) = &state.in_flight_unwind {
                unwind.unwind.clone()
            } else {
                let entries = state.entries.iter().rev().cloned().collect();
                let generation = issue_counter(&mut state.next_generation);
                let unwind = dispose_generation(entries, Arc::downgrade(&self.state), generation)
                    .boxed()
                    .shared();
                state.in_flight_unwind = Some(InFlightUnwind {
                    generation,
                    unwind: unwind.clone(),
                });
                unwind
            }
        };

        unwind.await
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

async fn dispose_entries(entries: Vec<Arc<EffectSlot>>) -> Result<(), DisposeErrors> {
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

async fn dispose_generation(
    entries: Vec<Arc<EffectSlot>>,
    state: Weak<Mutex<EffectState>>,
    generation: u64,
) -> Result<(), DisposeErrors> {
    let result = dispose_entries(entries).await;
    if let Some(state) = state.upgrade() {
        let mut state = state.lock();
        if state
            .in_flight_unwind
            .as_ref()
            .is_some_and(|unwind| unwind.generation == generation)
        {
            state.in_flight_unwind = None;
        }
    }
    result
}

fn issue_counter(counter: &mut u64) -> u64 {
    let value = *counter;
    *counter = counter.wrapping_add(1);
    value
}
