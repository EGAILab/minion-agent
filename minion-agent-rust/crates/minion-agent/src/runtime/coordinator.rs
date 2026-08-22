use std::sync::Arc;

use futures::FutureExt;
use parking_lot::Mutex;
use serde_json::Value;

use super::{
    DynPluginSpec, EventBus, FiberError, FiberHandle, FiberInitContext, PluginConfigError,
    ScopeTree, ServiceOwner, ServiceRegistry,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeObservation {
    FiberState {
        plugin: String,
        state: super::FiberState,
    },
    EffectCreated {
        plugin: String,
        label: String,
    },
    EffectDisposed {
        plugin: String,
        label: String,
    },
    ServiceProvided {
        plugin: String,
        service: super::ServiceName,
    },
    ServiceRevoked {
        plugin: String,
        service: super::ServiceName,
    },
    ScopeDisposed {
        scope: super::ScopeId,
    },
}

pub trait RuntimeObserver: Send + Sync + 'static {
    fn observe(&self, observation: RuntimeObservation);
}

struct NoopObserver;

impl RuntimeObserver for NoopObserver {
    fn observe(&self, _observation: RuntimeObservation) {}
}

#[derive(Clone)]
pub struct Runtime {
    core: Arc<RuntimeCore>,
}

struct RuntimeCore {
    services: ServiceRegistry,
    events: EventBus,
    scopes: ScopeTree,
    fibers: Mutex<Vec<FiberHandle>>,
    observer: Arc<dyn RuntimeObserver>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::with_observer(Arc::new(NoopObserver))
    }

    pub fn with_observer(observer: Arc<dyn RuntimeObserver>) -> Self {
        let scopes = ScopeTree::new();
        let services = ServiceRegistry::new();
        let core = Arc::new(RuntimeCore {
            services: services.clone(),
            events: EventBus::new(scopes.clone()),
            scopes,
            fibers: Mutex::new(Vec::new()),
            observer,
        });
        let weak = Arc::downgrade(&core);
        let revoke_observer = Arc::clone(&core.observer);
        services.set_revocation_callback(Arc::new(move |name, holder| {
            let weak = weak.clone();
            let observer = Arc::clone(&revoke_observer);
            async move {
                observer.observe(RuntimeObservation::ServiceRevoked {
                    plugin: holder,
                    service: name.clone(),
                });
                let Some(core) = weak.upgrade() else {
                    return Ok(());
                };
                core.reconcile_dependents(&name).await.map_err(|error| {
                    super::DisposeError::new(format!("reconcile({name})"), error.to_string())
                })
            }
            .boxed()
        }));
        Self { core }
    }

    pub fn mount(
        &self,
        spec: &DynPluginSpec,
        config: Value,
    ) -> Result<FiberHandle, PluginConfigError> {
        let services = self.core.services.clone();
        let inject = spec.inject().to_vec();
        let owner = Arc::new(Mutex::new(None::<FiberHandle>));
        let context_owner = Arc::clone(&owner);
        let context_services = self.core.services.clone();
        let context_observer = Arc::clone(&self.core.observer);
        let plugin_name = spec.name().to_owned();
        let state_observer = Arc::clone(&self.core.observer);
        let state_plugin = plugin_name.clone();
        let fiber = spec.mount_with_context_factory(
            config,
            move || inject.iter().all(|name| services.is_visible(name)),
            Arc::new(move |effects| {
                let fiber = context_owner
                    .lock()
                    .clone()
                    .expect("coordinated fiber is installed before initialization");
                let owner: Arc<dyn ServiceOwner> = Arc::new(fiber);
                FiberInitContext::coordinated(
                    effects,
                    context_services.clone(),
                    owner,
                    plugin_name.clone(),
                    Arc::clone(&context_observer),
                )
            }),
            Arc::new(move |state| {
                state_observer.observe(RuntimeObservation::FiberState {
                    plugin: state_plugin.clone(),
                    state,
                });
            }),
        )?;
        *owner.lock() = Some(fiber.clone());
        self.core.fibers.lock().push(fiber.clone());
        Ok(fiber)
    }

    pub async fn reconcile(&self) -> Result<(), FiberError> {
        loop {
            let fibers = self.core.fibers.lock().clone();
            let before: Vec<_> = fibers.iter().map(FiberHandle::state).collect();
            for (fiber, before_state) in fibers.iter().zip(&before) {
                fiber.dependencies_changed();
                fiber.reconcile().await?;
                if *before_state != super::FiberState::Active
                    && fiber.state() == super::FiberState::Active
                {
                    for name in self.core.services.names_held_by(fiber.name()) {
                        self.core.reconcile_dependents(&name).await?;
                    }
                }
            }
            let changed = fibers
                .iter()
                .zip(before)
                .any(|(fiber, before)| fiber.state() != before);
            if !changed {
                return Ok(());
            }
        }
    }

    pub async fn unmount(&self, fiber: &FiberHandle) -> Result<(), FiberError> {
        fiber.dispose().await
    }

    pub fn services(&self) -> &ServiceRegistry {
        &self.core.services
    }

    pub fn events(&self) -> &EventBus {
        &self.core.events
    }

    pub fn scopes(&self) -> &ScopeTree {
        &self.core.scopes
    }
}

impl RuntimeCore {
    async fn reconcile_dependents(&self, name: &super::ServiceName) -> Result<(), FiberError> {
        let fibers: Vec<_> = self
            .fibers
            .lock()
            .iter()
            .filter(|fiber| fiber.inject().contains(name))
            .cloned()
            .collect();
        for fiber in fibers {
            fiber.dependencies_changed();
            fiber.reconcile().await?;
        }
        Ok(())
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
