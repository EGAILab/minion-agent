use std::sync::Arc;

use futures::FutureExt;
use parking_lot::Mutex;
use serde_json::Value;

use super::{
    DynPluginSpec, EventBus, FiberError, FiberHandle, FiberInitContext, PluginConfigError,
    ScopeTree, ServiceOwner, ServiceRegistry,
};

#[derive(Clone)]
pub struct Runtime {
    core: Arc<RuntimeCore>,
}

struct RuntimeCore {
    services: ServiceRegistry,
    events: EventBus,
    scopes: ScopeTree,
    fibers: Mutex<Vec<FiberHandle>>,
}

impl Runtime {
    pub fn new() -> Self {
        let scopes = ScopeTree::new();
        let services = ServiceRegistry::new();
        let core = Arc::new(RuntimeCore {
            services: services.clone(),
            events: EventBus::new(scopes.clone()),
            scopes,
            fibers: Mutex::new(Vec::new()),
        });
        let weak = Arc::downgrade(&core);
        services.set_revocation_callback(Arc::new(move |name| {
            let weak = weak.clone();
            async move {
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
        let fiber = spec.mount_with_context_factory(
            config,
            move || inject.iter().all(|name| services.is_visible(name)),
            Arc::new(move |effects| {
                let fiber = context_owner
                    .lock()
                    .clone()
                    .expect("coordinated fiber is installed before initialization");
                let owner: Arc<dyn ServiceOwner> = Arc::new(fiber);
                FiberInitContext::coordinated(effects, context_services.clone(), owner)
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
            for fiber in &fibers {
                fiber.dependencies_changed();
                fiber.reconcile().await?;
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
