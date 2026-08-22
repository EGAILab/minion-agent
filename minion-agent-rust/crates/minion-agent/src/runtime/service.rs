use std::{
    any::{Any, TypeId, type_name},
    collections::HashMap,
    fmt,
    sync::{Arc, Weak},
};

use futures::future::BoxFuture;
use parking_lot::Mutex;

use super::{DisposeError, EffectHandle, EffectStore, RuntimeError, ServiceName};

pub trait Service: Send + Sync + 'static {
    const NAME: &'static str;
}

pub trait ServiceOwner: Send + Sync + 'static {
    fn service_owner_name(&self) -> String;

    fn service_owner_is_active(&self) -> bool;

    fn service_effect_store(&self) -> Arc<EffectStore>;
}

pub type ServiceCheck = Arc<dyn Fn() -> bool + Send + Sync + 'static>;

#[derive(Clone)]
pub struct ServiceRegistry {
    state: Arc<Mutex<ServiceRegistryState>>,
    on_revoked: Arc<Mutex<Option<ServiceRevoked>>>,
}

pub(crate) type ServiceRevoked =
    Arc<dyn Fn(ServiceName) -> BoxFuture<'static, Result<(), DisposeError>> + Send + Sync>;

struct ServiceRegistryState {
    contracts: HashMap<ServiceName, ServiceContract>,
    registrations: HashMap<ServiceName, ServiceEntry>,
    next_registration_id: u64,
}

struct ServiceContract {
    type_id: TypeId,
    type_name: &'static str,
}

struct ServiceEntry {
    id: u64,
    holder: String,
    owner: Arc<dyn ServiceOwner>,
    value: Arc<dyn Any + Send + Sync>,
    check: Option<ServiceCheck>,
}

struct ServiceSnapshot {
    owner: Arc<dyn ServiceOwner>,
    value: Arc<dyn Any + Send + Sync>,
    check: Option<ServiceCheck>,
}

pub struct ServiceRegistration {
    effect: EffectHandle,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(ServiceRegistryState {
                contracts: HashMap::new(),
                registrations: HashMap::new(),
                next_registration_id: 0,
            })),
            on_revoked: Arc::new(Mutex::new(None)),
        }
    }

    pub fn provide<S>(
        &self,
        owner: Arc<dyn ServiceOwner>,
        value: Arc<S>,
        check: Option<ServiceCheck>,
    ) -> Result<ServiceRegistration, RuntimeError>
    where
        S: Service,
    {
        let name = ServiceName::new(S::NAME)?;
        let holder = owner.service_owner_name();
        let registration_id = {
            let mut state = self.state.lock();
            validate_registration::<S>(&state, &name)?;
            let id = state.next_registration_id;
            state.next_registration_id = state
                .next_registration_id
                .checked_add(1)
                .expect("service registration ID space exhausted");
            id
        };
        let effects = owner.service_effect_store();
        let erased_value: Arc<dyn Any + Send + Sync> = value;
        let removal_state = Arc::downgrade(&self.state);
        let removal_name = name.clone();
        let on_revoked = Arc::clone(&self.on_revoked);
        let commit_state = Arc::clone(&self.state);
        let commit_name = name.clone();
        let pending_entry = Box::new(ServiceEntry {
            id: registration_id,
            holder,
            owner,
            value: erased_value,
            check,
        });

        let (effect, ()) = effects.try_push_with_commit(
            format!("provide({name})"),
            move || {
                Box::pin(async move {
                    if remove_registration(removal_state, &removal_name, registration_id) {
                        let callback = on_revoked.lock().clone();
                        if let Some(callback) = callback {
                            callback(removal_name).await?;
                        }
                    }
                    Ok(())
                })
            },
            pending_entry,
            move |entry| {
                let mut state = commit_state.lock();
                if let Err(error) = validate_registration::<S>(&state, &commit_name) {
                    return Err((error, entry));
                }

                state
                    .contracts
                    .entry(commit_name.clone())
                    .or_insert(ServiceContract {
                        type_id: TypeId::of::<S>(),
                        type_name: type_name::<S>(),
                    });
                state.registrations.insert(commit_name.clone(), *entry);
                Ok(())
            },
        )?;

        Ok(ServiceRegistration { effect })
    }

    pub fn require<S>(&self) -> Result<Arc<S>, RuntimeError>
    where
        S: Service,
    {
        let name = ServiceName::new(S::NAME)?;
        let snapshot = {
            let state = self.state.lock();
            if let Some(contract) = state.contracts.get(&name)
                && contract.type_id != TypeId::of::<S>()
            {
                return Err(RuntimeError::ServiceTypeMismatch {
                    name,
                    expected: contract.type_name,
                    actual: type_name::<S>(),
                });
            }
            state.registrations.get(&name).map(|entry| ServiceSnapshot {
                owner: Arc::clone(&entry.owner),
                value: Arc::clone(&entry.value),
                check: entry.check.clone(),
            })
        };
        let Some(snapshot) = snapshot else {
            return Err(RuntimeError::ServiceUnavailable { name });
        };
        if !snapshot.owner.service_owner_is_active()
            || snapshot.check.as_ref().is_some_and(|check| !check())
        {
            return Err(RuntimeError::ServiceUnavailable { name });
        }

        match snapshot.value.downcast::<S>() {
            Ok(value) => Ok(value),
            Err(_) => panic!("service contract metadata disagrees with erased storage for {name}"),
        }
    }

    pub(crate) fn is_visible(&self, name: &ServiceName) -> bool {
        let snapshot = {
            let state = self.state.lock();
            state.registrations.get(name).map(|entry| ServiceSnapshot {
                owner: Arc::clone(&entry.owner),
                value: Arc::clone(&entry.value),
                check: entry.check.clone(),
            })
        };
        snapshot.is_some_and(|snapshot| {
            snapshot.owner.service_owner_is_active()
                && snapshot.check.as_ref().is_none_or(|check| check())
        })
    }

    pub(crate) fn set_revocation_callback(&self, callback: ServiceRevoked) {
        *self.on_revoked.lock() = Some(callback);
    }

    pub(crate) fn names_held_by(&self, holder: &str) -> Vec<ServiceName> {
        self.state
            .lock()
            .registrations
            .iter()
            .filter(|(_, entry)| entry.holder == holder)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceRegistration {
    pub async fn dispose(&self) -> Result<(), DisposeError> {
        self.effect.dispose().await
    }
}

impl fmt::Debug for ServiceRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ServiceRegistration { .. }")
    }
}

fn remove_registration(
    state: Weak<Mutex<ServiceRegistryState>>,
    name: &ServiceName,
    expected_id: u64,
) -> bool {
    let Some(state) = state.upgrade() else {
        return false;
    };
    let removed = {
        let mut state = state.lock();
        if state
            .registrations
            .get(name)
            .is_some_and(|entry| entry.id == expected_id)
        {
            state.registrations.remove(name)
        } else {
            None
        }
    };
    let was_removed = removed.is_some();
    drop(removed);
    was_removed
}

fn validate_registration<S>(
    state: &ServiceRegistryState,
    name: &ServiceName,
) -> Result<(), RuntimeError>
where
    S: Service,
{
    if let Some(contract) = state.contracts.get(name)
        && contract.type_id != TypeId::of::<S>()
    {
        return Err(RuntimeError::ServiceTypeMismatch {
            name: name.clone(),
            expected: contract.type_name,
            actual: type_name::<S>(),
        });
    }
    if let Some(existing) = state.registrations.get(name) {
        return Err(RuntimeError::ServiceConflict {
            name: name.clone(),
            holder: existing.holder.clone(),
        });
    }
    Ok(())
}
