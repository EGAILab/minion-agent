use std::sync::Arc;

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
    next_owner_epoch: u64,
}

type BulkUnwind = Shared<BoxFuture<'static, Result<(), DisposeErrors>>>;

struct InFlightUnwind {
    generation: u64,
    owner_epoch: u64,
    owner_claimed: bool,
    unwind: BulkUnwind,
}

enum BulkUnwindRole {
    Owner {
        unwind: BulkUnwind,
        lease: OwnerLease,
    },
    Follower {
        unwind: BulkUnwind,
        generation: u64,
    },
}

struct OwnerLease {
    state: Arc<Mutex<EffectState>>,
    generation: u64,
    owner_epoch: u64,
    active: bool,
}

impl OwnerLease {
    fn new(state: Arc<Mutex<EffectState>>, generation: u64, owner_epoch: u64) -> Self {
        Self {
            state,
            generation,
            owner_epoch,
            active: true,
        }
    }

    fn complete(&mut self) -> bool {
        let mut state = self.state.lock();
        let owns_unwind = state.in_flight_unwind.as_ref().is_some_and(|unwind| {
            unwind.generation == self.generation && unwind.owner_epoch == self.owner_epoch
        });
        if owns_unwind {
            state.in_flight_unwind = None;
        }
        self.active = false;
        owns_unwind
    }
}

impl Drop for OwnerLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let mut state = self.state.lock();
        if let Some(unwind) = &mut state.in_flight_unwind
            && unwind.generation == self.generation
            && unwind.owner_epoch == self.owner_epoch
        {
            unwind.owner_claimed = false;
        }
    }
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
                next_owner_epoch: 0,
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
        let role = {
            let mut state = self.state.lock();
            state.accepting = false;
            if state
                .in_flight_unwind
                .as_ref()
                .is_some_and(|unwind| unwind.owner_claimed)
            {
                let unwind = state.in_flight_unwind.as_ref().unwrap();
                BulkUnwindRole::Follower {
                    unwind: unwind.unwind.clone(),
                    generation: unwind.generation,
                }
            } else if state.in_flight_unwind.is_some() {
                let owner_epoch = issue_counter(&mut state.next_owner_epoch);
                let unwind = state.in_flight_unwind.as_mut().unwrap();
                unwind.owner_claimed = true;
                unwind.owner_epoch = owner_epoch;
                BulkUnwindRole::Owner {
                    unwind: unwind.unwind.clone(),
                    lease: OwnerLease::new(Arc::clone(&self.state), unwind.generation, owner_epoch),
                }
            } else {
                let entries = state.entries.iter().rev().cloned().collect();
                let unwind = dispose_entries(entries).boxed().shared();
                let generation = issue_counter(&mut state.next_generation);
                let owner_epoch = issue_counter(&mut state.next_owner_epoch);
                state.in_flight_unwind = Some(InFlightUnwind {
                    generation,
                    owner_epoch,
                    owner_claimed: true,
                    unwind: unwind.clone(),
                });
                BulkUnwindRole::Owner {
                    unwind,
                    lease: OwnerLease::new(Arc::clone(&self.state), generation, owner_epoch),
                }
            }
        };

        match role {
            BulkUnwindRole::Owner { unwind, mut lease } => {
                let result = unwind.await;
                if lease.complete() { result } else { Ok(()) }
            }
            BulkUnwindRole::Follower { unwind, generation } => {
                let result = unwind.await;
                self.claim_completed_unwind(generation)
                    .map_or(
                        Ok(()),
                        |mut lease| {
                            if lease.complete() { result } else { Ok(()) }
                        },
                    )
            }
        }
    }

    fn claim_completed_unwind(&self, generation: u64) -> Option<OwnerLease> {
        let mut state = self.state.lock();
        let unwind = state.in_flight_unwind.as_ref()?;
        if unwind.generation != generation || unwind.owner_claimed {
            return None;
        }

        let owner_epoch = issue_counter(&mut state.next_owner_epoch);
        let unwind = state.in_flight_unwind.as_mut().unwrap();
        unwind.owner_claimed = true;
        unwind.owner_epoch = owner_epoch;
        Some(OwnerLease::new(
            Arc::clone(&self.state),
            generation,
            owner_epoch,
        ))
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

fn issue_counter(counter: &mut u64) -> u64 {
    let value = *counter;
    *counter = counter.wrapping_add(1);
    value
}
