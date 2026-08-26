use std::sync::{Arc, Mutex};

use parking_lot::Mutex as ParkingMutex;

use super::{DisposeError, RuntimeError, ScopeHandle, ScopeId, ScopeTree};

pub struct ScopedRegistry<T> {
    tree: ScopeTree,
    state: Arc<ParkingMutex<RegistryState<T>>>,
}

struct RegistryState<T> {
    entries: Vec<Option<RegistryEntry<T>>>,
}

struct RegistryEntry<T> {
    owner: Option<ScopeId>,
    value: Arc<T>,
}

#[derive(Clone)]
pub struct RegistrationHandle {
    removal: Arc<RegistrationRemoval>,
}

struct RegistrationRemoval {
    remove: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl<T> ScopedRegistry<T> {
    pub fn new(tree: impl Into<ScopeTree>) -> Self {
        Self {
            tree: tree.into(),
            state: Arc::new(ParkingMutex::new(RegistryState {
                entries: Vec::new(),
            })),
        }
    }

    pub fn register(
        &self,
        owner: Option<&ScopeHandle>,
        value: T,
    ) -> Result<RegistrationHandle, RuntimeError>
    where
        T: Send + Sync + 'static,
    {
        if let Some(owner) = owner
            && (!owner.belongs_to(&self.tree) || !owner.is_active())
        {
            return Err(RuntimeError::InactiveOwner {
                owner: format!("scope {}", owner.id().as_u64()),
            });
        }

        let owner_id = owner.map(ScopeHandle::id);
        let entry_index = {
            let mut state = self.state.lock();
            let index = state.entries.len();
            state.entries.push(Some(RegistryEntry {
                owner: owner_id,
                value: Arc::new(value),
            }));
            index
        };
        let state = Arc::clone(&self.state);
        let removal = Arc::new(RegistrationRemoval {
            remove: Mutex::new(Some(Box::new(move || {
                state.lock().entries[entry_index] = None;
            }))),
        });
        let handle = RegistrationHandle {
            removal: Arc::clone(&removal),
        };

        if let Some(owner) = owner {
            let owned_removal = Arc::clone(&removal);
            if let Err(error) = owner
                .effects()
                .push("scoped registry registration", move || {
                    Box::pin(async move {
                        owned_removal.remove();
                        Ok::<(), DisposeError>(())
                    })
                })
            {
                handle.remove();
                return Err(error);
            }
        }
        Ok(handle)
    }

    pub fn visible_from(&self, request: ScopeId) -> Vec<Arc<T>> {
        let Some(chain) = self.tree.active_ancestor_chain(request) else {
            return Vec::new();
        };
        let state = self.state.lock();
        let mut visible = Vec::new();
        for scope in chain {
            visible.extend(
                state
                    .entries
                    .iter()
                    .filter_map(|entry| entry.as_ref())
                    .filter(|entry| entry.owner == Some(scope))
                    .map(|entry| Arc::clone(&entry.value)),
            );
        }
        visible.extend(
            state
                .entries
                .iter()
                .filter_map(|entry| entry.as_ref())
                .filter(|entry| entry.owner.is_none())
                .map(|entry| Arc::clone(&entry.value)),
        );
        visible
    }

    pub fn visible_from_scope(&self, request: Option<&ScopeHandle>) -> Vec<Arc<T>> {
        match request {
            Some(request) if request.belongs_to(&self.tree) => self.visible_from(request.id()),
            Some(_) => Vec::new(),
            None => self.visible_untagged(),
        }
    }

    fn visible_untagged(&self) -> Vec<Arc<T>> {
        self.state
            .lock()
            .entries
            .iter()
            .filter_map(|entry| entry.as_ref())
            .filter(|entry| entry.owner.is_none())
            .map(|entry| Arc::clone(&entry.value))
            .collect()
    }
}

impl From<&ScopeTree> for ScopeTree {
    fn from(tree: &ScopeTree) -> Self {
        tree.clone()
    }
}

impl RegistrationHandle {
    pub fn withdraw(&self) {
        self.remove();
    }

    pub async fn dispose(&self) -> Result<(), DisposeError> {
        self.withdraw();
        Ok(())
    }

    fn remove(&self) {
        if let Some(remove) = self
            .removal
            .remove
            .lock()
            .expect("registration removal lock poisoned")
            .take()
        {
            remove();
        }
    }
}

impl RegistrationRemoval {
    fn remove(&self) {
        if let Some(remove) = self
            .remove
            .lock()
            .expect("registration removal lock poisoned")
            .take()
        {
            remove();
        }
    }
}
