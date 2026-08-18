use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::Arc,
};

use futures::{
    FutureExt,
    future::{BoxFuture, Either, select},
};
use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::{Mutex as AsyncMutex, watch};

use super::{
    DisposeError, DisposeErrors, EffectHandle, EffectStore, RuntimeError, ServiceName,
    ServiceOwner,
    plugin::{ErasedConfig, ErasedInitializer, PluginInitError},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FiberState {
    Pending,
    Loading,
    Active,
    Failed,
    Unloading,
    Disposed,
}

#[derive(Clone, Debug, Error)]
pub enum FiberError {
    #[error(transparent)]
    Initialization(PluginInitError),
    #[error(transparent)]
    Cleanup(DisposeErrors),
    #[error("plugin initialization failed and cleanup also failed")]
    InitializationAndCleanup {
        initialization: PluginInitError,
        cleanup: DisposeErrors,
    },
}

#[derive(Clone)]
pub struct FiberInitContext {
    effects: Arc<EffectStore>,
}

impl std::fmt::Debug for FiberInitContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("FiberInitContext { .. }")
    }
}

impl FiberInitContext {
    pub fn effect<F>(
        &self,
        label: impl Into<String>,
        disposer: F,
    ) -> Result<EffectHandle, RuntimeError>
    where
        F: FnOnce() -> BoxFuture<'static, Result<(), DisposeError>> + Send + 'static,
    {
        self.effects.push(label, disposer)
    }

    pub fn effect_store(&self) -> Arc<EffectStore> {
        Arc::clone(&self.effects)
    }
}

#[derive(Clone)]
pub struct FiberHandle {
    inner: Arc<FiberInner>,
}

struct FiberInner {
    name: String,
    inject: Vec<ServiceName>,
    initializer: ErasedInitializer,
    config: Arc<dyn Any + Send + Sync>,
    dependencies_visible: Arc<dyn Fn() -> bool + Send + Sync>,
    transition: AsyncMutex<()>,
    lifecycle: Mutex<Lifecycle>,
    trace: watch::Sender<Vec<FiberState>>,
}

struct Lifecycle {
    state: FiberState,
    generation: u64,
    current: Option<Generation>,
    dispose_requested: bool,
    dependency_invalidated: bool,
    failure: Option<PluginInitError>,
    last_cleanup: Option<(u64, Result<(), DisposeErrors>)>,
}

struct Generation {
    id: u64,
    effects: Arc<EffectStore>,
    cancellation: CancellationToken,
}

#[derive(Clone)]
struct CancellationToken {
    cancelled: watch::Sender<bool>,
}

impl CancellationToken {
    fn new() -> Self {
        let (cancelled, _) = watch::channel(false);
        Self { cancelled }
    }

    fn cancel(&self) {
        self.cancelled.send_replace(true);
    }

    async fn cancelled(&self) {
        let mut receiver = self.cancelled.subscribe();
        if *receiver.borrow_and_update() {
            return;
        }
        while receiver.changed().await.is_ok() {
            if *receiver.borrow_and_update() {
                return;
            }
        }
    }
}

enum InitializerOutcome {
    Cancelled,
    Completed(Result<(), PluginInitError>),
    Panicked(Box<dyn Any + Send>),
}

impl FiberHandle {
    pub(crate) fn new(
        name: String,
        inject: Vec<ServiceName>,
        initializer: ErasedInitializer,
        config: ErasedConfig,
        dependencies_visible: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        let (trace, _) = watch::channel(vec![FiberState::Pending]);
        Self {
            inner: Arc::new(FiberInner {
                name,
                inject,
                initializer,
                config,
                dependencies_visible,
                transition: AsyncMutex::new(()),
                lifecycle: Mutex::new(Lifecycle {
                    state: FiberState::Pending,
                    generation: 0,
                    current: None,
                    dispose_requested: false,
                    dependency_invalidated: false,
                    failure: None,
                    last_cleanup: None,
                }),
                trace,
            }),
        }
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn inject(&self) -> &[ServiceName] {
        &self.inner.inject
    }

    pub fn state(&self) -> FiberState {
        self.inner.lifecycle.lock().state
    }

    pub fn failure(&self) -> Option<PluginInitError> {
        self.inner.lifecycle.lock().failure.clone()
    }

    pub fn trace(&self) -> Vec<FiberState> {
        self.inner.trace.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<Vec<FiberState>> {
        self.inner.trace.subscribe()
    }

    pub fn dependencies_changed(&self) {
        if !(self.inner.dependencies_visible)() {
            let mut lifecycle = self.inner.lifecycle.lock();
            if lifecycle.state == FiberState::Active {
                lifecycle.dependency_invalidated = true;
                let generation = lifecycle
                    .current
                    .as_ref()
                    .expect("active fiber has no owned generation");
                let id = generation.id;
                let effects = Arc::clone(&generation.effects);
                let cancellation = generation.cancellation.clone();
                if lifecycle.generation == id {
                    lifecycle.generation = lifecycle
                        .generation
                        .checked_add(1)
                        .expect("fiber generation space exhausted");
                }
                effects.close();
                cancellation.cancel();
            }
            drop(lifecycle);
            self.invalidate_loading(false);
        }
    }

    pub fn reconcile(&self) -> BoxFuture<'static, Result<(), FiberError>> {
        self.dependencies_changed();
        let fiber = self.clone();
        async move { fiber.reconcile_serialized().await }.boxed()
    }

    pub fn dispose(&self) -> BoxFuture<'static, Result<(), FiberError>> {
        let invalidated_generation = self.invalidate_loading(true);
        let fiber = self.clone();
        async move {
            fiber.dispose_serialized().await?;
            if let Some(generation) = invalidated_generation {
                fiber.cleanup_outcome(generation)
            } else {
                Ok(())
            }
        }
        .boxed()
    }

    async fn reconcile_serialized(&self) -> Result<(), FiberError> {
        let _transition = self.inner.transition.lock().await;
        match self.state() {
            FiberState::Pending
                if !self.disposal_requested() && (self.inner.dependencies_visible)() =>
            {
                self.load().await
            }
            FiberState::Active
                if self.dependency_invalidated() || !(self.inner.dependencies_visible)() =>
            {
                self.unload(false).await
            }
            FiberState::Pending
            | FiberState::Active
            | FiberState::Failed
            | FiberState::Disposed => Ok(()),
            FiberState::Loading | FiberState::Unloading => {
                panic!("serialized transition lock observed an in-flight fiber state")
            }
        }
    }

    async fn dispose_serialized(&self) -> Result<(), FiberError> {
        let _transition = self.inner.transition.lock().await;
        match self.state() {
            FiberState::Pending | FiberState::Failed => {
                let mut lifecycle = self.inner.lifecycle.lock();
                lifecycle.current = None;
                self.transition_to(&mut lifecycle, FiberState::Disposed);
                Ok(())
            }
            FiberState::Active => self.unload(true).await,
            FiberState::Disposed => Ok(()),
            FiberState::Loading | FiberState::Unloading => {
                panic!("serialized transition lock observed an in-flight fiber state")
            }
        }
    }

    async fn load(&self) -> Result<(), FiberError> {
        let (generation_id, effects, cancellation) = {
            let mut lifecycle = self.inner.lifecycle.lock();
            assert_eq!(lifecycle.state, FiberState::Pending);
            assert!(!lifecycle.dispose_requested);
            lifecycle.dependency_invalidated = false;
            lifecycle.failure = None;
            lifecycle.generation = lifecycle
                .generation
                .checked_add(1)
                .expect("fiber generation space exhausted");
            let generation_id = lifecycle.generation;
            let effects = Arc::new(EffectStore::new());
            let cancellation = CancellationToken::new();
            lifecycle.current = Some(Generation {
                id: generation_id,
                effects: Arc::clone(&effects),
                cancellation: cancellation.clone(),
            });
            self.transition_to(&mut lifecycle, FiberState::Loading);
            (generation_id, effects, cancellation)
        };

        let context = FiberInitContext {
            effects: Arc::clone(&effects),
        };
        let initializer = match catch_unwind(AssertUnwindSafe(|| {
            (self.inner.initializer)(context, Arc::clone(&self.inner.config))
        })) {
            Ok(initializer) => initializer,
            Err(payload) => {
                let _ = self
                    .finish_loading_cleanup(generation_id, FiberState::Disposed, None)
                    .await;
                resume_unwind(payload)
            }
        };
        let guarded_initializer = AssertUnwindSafe(initializer).catch_unwind().boxed();
        let cancelled = cancellation.cancelled().boxed();
        let outcome = match select(cancelled, guarded_initializer).await {
            Either::Left(((), _)) => InitializerOutcome::Cancelled,
            Either::Right((result, _)) => match result {
                Ok(result) => InitializerOutcome::Completed(result),
                Err(payload) => InitializerOutcome::Panicked(payload),
            },
        };

        match outcome {
            InitializerOutcome::Completed(Ok(())) => {
                if self.try_commit_active(generation_id) {
                    Ok(())
                } else {
                    self.finish_loading_cleanup(generation_id, FiberState::Pending, None)
                        .await
                }
            }
            InitializerOutcome::Cancelled => {
                self.finish_loading_cleanup(generation_id, FiberState::Pending, None)
                    .await
            }
            InitializerOutcome::Completed(Err(initialization)) => {
                let target = if self.loading_generation_is_live(generation_id) {
                    FiberState::Failed
                } else {
                    FiberState::Pending
                };
                let failure = (target == FiberState::Failed).then_some(initialization);
                self.finish_loading_cleanup(generation_id, target, failure)
                    .await
            }
            InitializerOutcome::Panicked(payload) => {
                let _ = self
                    .finish_loading_cleanup(generation_id, FiberState::Disposed, None)
                    .await;
                resume_unwind(payload)
            }
        }
    }

    fn try_commit_active(&self, generation_id: u64) -> bool {
        if !(self.inner.dependencies_visible)() {
            self.invalidate_loading(false);
            return false;
        }
        let mut lifecycle = self.inner.lifecycle.lock();
        let valid = lifecycle.state == FiberState::Loading
            && lifecycle.generation == generation_id
            && !lifecycle.dispose_requested
            && !lifecycle.dependency_invalidated;
        if valid {
            self.transition_to(&mut lifecycle, FiberState::Active);
        }
        valid
    }

    fn loading_generation_is_live(&self, generation_id: u64) -> bool {
        let lifecycle = self.inner.lifecycle.lock();
        lifecycle.state == FiberState::Loading
            && lifecycle.generation == generation_id
            && !lifecycle.dispose_requested
            && !lifecycle.dependency_invalidated
    }

    fn dependency_invalidated(&self) -> bool {
        self.inner.lifecycle.lock().dependency_invalidated
    }

    fn disposal_requested(&self) -> bool {
        self.inner.lifecycle.lock().dispose_requested
    }

    async fn finish_loading_cleanup(
        &self,
        generation_id: u64,
        default_target: FiberState,
        initialization: Option<PluginInitError>,
    ) -> Result<(), FiberError> {
        let effects = self.close_generation(generation_id);
        let cleanup = effects.close_and_dispose().await;
        let target = {
            let mut lifecycle = self.inner.lifecycle.lock();
            if lifecycle
                .current
                .as_ref()
                .is_some_and(|generation| generation.id == generation_id)
            {
                lifecycle.current = None;
            }
            let target = if lifecycle.dispose_requested {
                FiberState::Disposed
            } else {
                default_target
            };
            lifecycle.failure = if target == FiberState::Failed {
                initialization.clone()
            } else {
                None
            };
            lifecycle.last_cleanup = Some((generation_id, cleanup.clone()));
            self.transition_to(&mut lifecycle, target);
            target
        };

        if target != FiberState::Failed {
            return cleanup.map_err(FiberError::Cleanup);
        }
        match (initialization, cleanup) {
            (Some(initialization), Ok(())) => Err(FiberError::Initialization(initialization)),
            (Some(initialization), Err(cleanup)) => Err(FiberError::InitializationAndCleanup {
                initialization,
                cleanup,
            }),
            (None, _) => panic!("failed fiber requires a represented initialization error"),
        }
    }

    async fn unload(&self, disposing: bool) -> Result<(), FiberError> {
        let (generation_id, effects) = {
            let mut lifecycle = self.inner.lifecycle.lock();
            assert_eq!(lifecycle.state, FiberState::Active);
            lifecycle.dispose_requested |= disposing;
            let generation = lifecycle
                .current
                .as_ref()
                .expect("active fiber has no owned generation");
            let generation_id = generation.id;
            let effects = Arc::clone(&generation.effects);
            let cancellation = generation.cancellation.clone();
            lifecycle.generation = lifecycle
                .generation
                .checked_add(1)
                .expect("fiber generation space exhausted");
            effects.close();
            cancellation.cancel();
            self.transition_to(&mut lifecycle, FiberState::Unloading);
            (generation_id, effects)
        };

        let cleanup = effects.close_and_dispose().await;
        {
            let mut lifecycle = self.inner.lifecycle.lock();
            lifecycle.current = None;
            let target = if lifecycle.dispose_requested {
                FiberState::Disposed
            } else {
                FiberState::Pending
            };
            lifecycle.last_cleanup = Some((generation_id, cleanup.clone()));
            self.transition_to(&mut lifecycle, target);
        }
        cleanup.map_err(FiberError::Cleanup)
    }

    fn invalidate_loading(&self, disposing: bool) -> Option<u64> {
        let mut lifecycle = self.inner.lifecycle.lock();
        lifecycle.dispose_requested |= disposing;
        if lifecycle.state != FiberState::Loading {
            let current = lifecycle.current.as_ref().map(|generation| {
                (
                    generation.id,
                    Arc::clone(&generation.effects),
                    generation.cancellation.clone(),
                )
            });
            if disposing
                && lifecycle.state == FiberState::Active
                && let Some((id, effects, cancellation)) = &current
            {
                if lifecycle.generation == *id {
                    lifecycle.generation = lifecycle
                        .generation
                        .checked_add(1)
                        .expect("fiber generation space exhausted");
                }
                effects.close();
                cancellation.cancel();
            }
            return disposing.then(|| current.map(|(id, _, _)| id)).flatten();
        }
        let Some(generation) = lifecycle.current.as_ref() else {
            panic!("loading fiber has no owned generation");
        };
        let id = generation.id;
        if lifecycle.generation == id {
            let effects = Arc::clone(&generation.effects);
            let cancellation = generation.cancellation.clone();
            lifecycle.generation = lifecycle
                .generation
                .checked_add(1)
                .expect("fiber generation space exhausted");
            effects.close();
            cancellation.cancel();
        }
        Some(id)
    }

    fn close_generation(&self, generation_id: u64) -> Arc<EffectStore> {
        let mut lifecycle = self.inner.lifecycle.lock();
        let generation = lifecycle
            .current
            .as_ref()
            .filter(|generation| generation.id == generation_id)
            .expect("loading generation disappeared before cleanup");
        let effects = Arc::clone(&generation.effects);
        let cancellation = generation.cancellation.clone();
        if lifecycle.generation == generation_id {
            lifecycle.generation = lifecycle
                .generation
                .checked_add(1)
                .expect("fiber generation space exhausted");
        }
        effects.close();
        cancellation.cancel();
        effects
    }

    fn cleanup_outcome(&self, generation_id: u64) -> Result<(), FiberError> {
        self.inner
            .lifecycle
            .lock()
            .last_cleanup
            .as_ref()
            .filter(|(id, _)| *id == generation_id)
            .map_or(Ok(()), |(_, result)| {
                result.clone().map_err(FiberError::Cleanup)
            })
    }

    fn transition_to(&self, lifecycle: &mut Lifecycle, state: FiberState) {
        if lifecycle.state == state {
            return;
        }
        lifecycle.state = state;
        self.inner.trace.send_modify(|trace| trace.push(state));
    }

    fn current_effect_store(&self) -> Arc<EffectStore> {
        self.inner
            .lifecycle
            .lock()
            .current
            .as_ref()
            .map(|generation| Arc::clone(&generation.effects))
            .unwrap_or_else(|| {
                let effects = Arc::new(EffectStore::new());
                effects.close();
                effects
            })
    }
}

impl ServiceOwner for FiberHandle {
    fn service_owner_name(&self) -> String {
        self.inner.name.clone()
    }

    fn service_owner_is_active(&self) -> bool {
        let lifecycle = self.inner.lifecycle.lock();
        lifecycle.state == FiberState::Active
            && !lifecycle.dispose_requested
            && !lifecycle.dependency_invalidated
    }

    fn service_effect_store(&self) -> Arc<EffectStore> {
        self.current_effect_store()
    }
}

impl std::fmt::Debug for FiberHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FiberHandle")
            .field("name", &self.inner.name)
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}
