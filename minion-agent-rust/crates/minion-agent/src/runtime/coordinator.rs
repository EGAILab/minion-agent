use std::sync::Arc;

use futures::FutureExt;
use parking_lot::Mutex;
use serde_json::Value;

use crate::tools::ToolRegistry;

use super::{
    DynPluginSpec, EventBus, FiberError, FiberHandle, FiberInitContext, PluginConfigError,
    ScopeTree, ServiceOwner, ServiceRegistry, fiber::ContextResources,
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
    tools: ToolRegistry,
    fibers: Mutex<Vec<FiberHandle>>,
    pending_revocations: Mutex<Vec<super::ServiceName>>,
    observer: Arc<dyn RuntimeObserver>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::with_observer(Arc::new(NoopObserver))
    }

    pub fn with_observer(observer: Arc<dyn RuntimeObserver>) -> Self {
        let scopes = ScopeTree::new();
        let services = ServiceRegistry::new();
        let tools = ToolRegistry::new(scopes.clone());
        let core = Arc::new(RuntimeCore {
            services: services.clone(),
            events: EventBus::new(scopes.clone()),
            scopes,
            tools,
            fibers: Mutex::new(Vec::new()),
            pending_revocations: Mutex::new(Vec::new()),
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
                core.invalidate_dependents(&name).await.map_err(|error| {
                    super::DisposeError::new(format!("reconcile({name})"), format!("{error:?}"))
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
        self.mount_in(spec, config, None)
    }

    pub fn mount_in(
        &self,
        spec: &DynPluginSpec,
        config: Value,
        scope: Option<super::ScopeHandle>,
    ) -> Result<FiberHandle, PluginConfigError> {
        let services = self.core.services.clone();
        let inject = spec.inject().to_vec();
        let owner = Arc::new(Mutex::new(None::<FiberHandle>));
        let context_owner = Arc::clone(&owner);
        let context_resources = ContextResources {
            services: self.core.services.clone(),
            events: self.core.events.clone(),
            tools: self.core.tools.clone(),
        };
        let context_observer = Arc::clone(&self.core.observer);
        let context_scope = scope.clone();
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
                    context_resources.clone(),
                    owner,
                    plugin_name.clone(),
                    Arc::clone(&context_observer),
                    context_scope.clone(),
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

    pub fn create_scope(
        &self,
        parent: Option<&super::ScopeHandle>,
    ) -> Result<super::ScopeHandle, super::RuntimeError> {
        let scope = match parent {
            Some(parent) => self.core.scopes.create_child(parent)?,
            None => self.core.scopes.create_root(),
        };
        let observer = Arc::clone(&self.core.observer);
        let id = scope.id();
        scope.effects().push("runtime.scope-observer", move || {
            Box::pin(async move {
                observer.observe(RuntimeObservation::ScopeDisposed { scope: id });
                Ok(())
            })
        })?;
        Ok(scope)
    }

    pub fn context(&self) -> super::Context {
        FiberInitContext::runtime_view(ContextResources {
            services: self.core.services.clone(),
            events: self.core.events.clone(),
            tools: self.core.tools.clone(),
        })
    }

    pub async fn reconcile(&self) -> Result<(), FiberError> {
        loop {
            self.core.reconcile_pending_revocations().await?;
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
        let disposal = fiber.dispose().await;
        let reconciliation = self.core.reconcile_pending_revocations().await;
        disposal.and(reconciliation)
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

    pub fn tools(&self) -> &ToolRegistry {
        &self.core.tools
    }
}

impl RuntimeCore {
    async fn invalidate_dependents(&self, name: &super::ServiceName) -> Result<(), FiberError> {
        let fibers: Vec<_> = self
            .fibers
            .lock()
            .iter()
            .filter(|fiber| fiber.inject().contains(name))
            .cloned()
            .collect();
        let states: Vec<_> = fibers.iter().map(FiberHandle::state).collect();
        for fiber in &fibers {
            fiber.dependencies_changed();
        }
        let mut deferred = false;
        let mut first_error = None;
        for (fiber, state) in fibers.into_iter().zip(states) {
            match state {
                super::FiberState::Active => {
                    if let Err(error) = fiber.reconcile().await
                        && first_error.is_none()
                    {
                        first_error = Some(error);
                    }
                }
                super::FiberState::Loading => deferred = true,
                _ => {}
            }
        }
        if deferred {
            let mut pending = self.pending_revocations.lock();
            if !pending.contains(name) {
                pending.push(name.clone());
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn reconcile_pending_revocations(&self) -> Result<(), FiberError> {
        let names = std::mem::take(&mut *self.pending_revocations.lock());
        let mut first_error = None;
        for name in names {
            if let Err(error) = self.reconcile_dependents(&name).await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn reconcile_dependents(&self, name: &super::ServiceName) -> Result<(), FiberError> {
        let fibers: Vec<_> = self
            .fibers
            .lock()
            .iter()
            .filter(|fiber| fiber.inject().contains(name))
            .cloned()
            .collect();
        let mut first_error = None;
        for fiber in fibers {
            fiber.dependencies_changed();
            if let Err(error) = fiber.reconcile().await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
