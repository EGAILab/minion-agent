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

use crate::tools::ToolRegistry;

use super::{
    DisposeError, DisposeErrors, EffectHandle, EffectStore, EventBus, RuntimeError,
    RuntimeObservation, RuntimeObserver, ScopeHandle, Service, ServiceCheck, ServiceName,
    ServiceOwner, ServiceRegistration, ServiceRegistry,
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
    services: Option<ServiceRegistry>,
    owner: Option<Arc<dyn ServiceOwner>>,
    plugin: Option<String>,
    observer: Option<Arc<dyn RuntimeObserver>>,
    events: Option<EventBus>,
    scope: Option<ScopeHandle>,
    tools: Option<ToolRegistry>,
}

pub type Context = FiberInitContext;

#[derive(Clone)]
pub(crate) struct ContextResources {
    pub services: ServiceRegistry,
    pub events: EventBus,
    pub tools: ToolRegistry,
}

pub(crate) type InitContextFactory =
    Arc<dyn Fn(Arc<EffectStore>) -> FiberInitContext + Send + Sync>;
pub(crate) type FiberStateObserver = Arc<dyn Fn(FiberState) + Send + Sync>;

impl std::fmt::Debug for FiberInitContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("FiberInitContext { .. }")
    }
}

impl FiberInitContext {
    pub(crate) fn standalone(effects: Arc<EffectStore>) -> Self {
        Self {
            effects,
            services: None,
            owner: None,
            plugin: None,
            observer: None,
            events: None,
            scope: None,
            tools: None,
        }
    }

    pub(crate) fn coordinated(
        effects: Arc<EffectStore>,
        resources: ContextResources,
        owner: Arc<dyn ServiceOwner>,
        plugin: String,
        observer: Arc<dyn RuntimeObserver>,
        scope: Option<ScopeHandle>,
    ) -> Self {
        Self {
            effects,
            services: Some(resources.services),
            owner: Some(owner),
            plugin: Some(plugin),
            observer: Some(observer),
            events: Some(resources.events),
            scope,
            tools: Some(resources.tools),
        }
    }

    pub(crate) fn runtime_view(resources: ContextResources) -> Self {
        let effects = Arc::new(EffectStore::new());
        effects.close();
        Self {
            effects,
            services: Some(resources.services),
            owner: None,
            plugin: None,
            observer: None,
            events: Some(resources.events),
            scope: None,
            tools: Some(resources.tools),
        }
    }

    pub fn effect<F>(
        &self,
        label: impl Into<String>,
        disposer: F,
    ) -> Result<EffectHandle, RuntimeError>
    where
        F: FnOnce() -> BoxFuture<'static, Result<(), DisposeError>> + Send + 'static,
    {
        let label = label.into();
        let owner_effects = self
            .scope
            .as_ref()
            .map_or(self.effects.as_ref(), ScopeHandle::effects);
        let Some(observer) = self.observer.as_ref() else {
            return owner_effects.push(label, disposer);
        };
        let plugin = self
            .plugin
            .as_ref()
            .expect("coordinated context has a plugin name")
            .clone();
        let disposal_observer = Arc::clone(observer);
        let disposal_plugin = plugin.clone();
        let disposal_label = label.clone();
        let handle = owner_effects.push(label.clone(), move || {
            let future = disposer();
            Box::pin(async move {
                let result = future.await;
                disposal_observer.observe(RuntimeObservation::EffectDisposed {
                    plugin: disposal_plugin,
                    label: disposal_label,
                });
                result
            })
        })?;
        observer.observe(RuntimeObservation::EffectCreated { plugin, label });
        Ok(handle)
    }

    pub fn effect_store(&self) -> Arc<EffectStore> {
        self.scope
            .as_ref()
            .map_or_else(|| Arc::clone(&self.effects), ScopeHandle::effect_store)
    }

    pub fn events(&self) -> Result<&EventBus, RuntimeError> {
        self.events
            .as_ref()
            .ok_or(RuntimeError::UncoordinatedContext)
    }

    pub fn scope(&self) -> Option<&ScopeHandle> {
        self.scope.as_ref()
    }

    pub fn tools(&self) -> Result<&ToolRegistry, RuntimeError> {
        self.tools
            .as_ref()
            .ok_or(RuntimeError::UncoordinatedContext)
    }

    pub fn provide<S>(
        &self,
        value: Arc<S>,
        check: Option<ServiceCheck>,
    ) -> Result<ServiceRegistration, RuntimeError>
    where
        S: Service,
    {
        let services = self
            .services
            .as_ref()
            .ok_or(RuntimeError::UncoordinatedContext)?;
        let owner = self
            .owner
            .as_ref()
            .ok_or(RuntimeError::UncoordinatedContext)?;
        let registration = services.provide(Arc::clone(owner), value, check)?;
        self.observer
            .as_ref()
            .expect("coordinated context has an observer")
            .observe(RuntimeObservation::ServiceProvided {
                plugin: self
                    .plugin
                    .as_ref()
                    .expect("coordinated context has a plugin name")
                    .clone(),
                service: ServiceName::new(S::NAME)
                    .expect("Service constants must use normative names"),
            });
        Ok(registration)
    }

    pub fn require<S>(&self) -> Result<Arc<S>, RuntimeError>
    where
        S: Service,
    {
        self.services
            .as_ref()
            .ok_or(RuntimeError::UncoordinatedContext)?
            .require::<S>()
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
    context_factory: InitContextFactory,
    state_observer: FiberStateObserver,
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
    next_transition_id: u64,
    in_flight: Option<InFlightTransition>,
}

struct InFlightTransition {
    id: u64,
    outcome: watch::Sender<Option<TransitionOutcome>>,
}

#[derive(Clone)]
enum TransitionOutcome {
    Completed(Result<(), FiberError>),
    Panicked(PanicReport),
}

#[derive(Clone, Debug)]
struct PanicReport {
    message: String,
}

impl PanicReport {
    fn from_payload(payload: Box<dyn Any + Send>) -> Self {
        let message = payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string plugin or runtime panic".to_owned());
        Self { message }
    }

    fn resume(self) -> ! {
        panic!("{}", self.message)
    }
}

#[derive(Clone, Copy)]
enum LifecycleRequest {
    Reconcile,
    Dispose,
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
        context_factory: InitContextFactory,
        state_observer: FiberStateObserver,
    ) -> Self {
        let (trace, _) = watch::channel(vec![FiberState::Pending]);
        Self {
            inner: Arc::new(FiberInner {
                name,
                inject,
                initializer,
                config,
                dependencies_visible,
                context_factory,
                state_observer,
                transition: AsyncMutex::new(()),
                lifecycle: Mutex::new(Lifecycle {
                    state: FiberState::Pending,
                    generation: 0,
                    current: None,
                    dispose_requested: false,
                    dependency_invalidated: false,
                    failure: None,
                    next_transition_id: 0,
                    in_flight: None,
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
        drop(self.signal_dependency_change());
    }

    pub fn reconcile(&self) -> BoxFuture<'static, Result<(), FiberError>> {
        let joined = self.signal_dependency_change();
        let fiber = self.clone();
        async move {
            fiber
                .drive_transition(LifecycleRequest::Reconcile, joined)
                .await
        }
        .boxed()
    }

    pub fn dispose(&self) -> BoxFuture<'static, Result<(), FiberError>> {
        let joined = self.request_disposal();
        let fiber = self.clone();
        async move {
            fiber
                .drive_transition(LifecycleRequest::Dispose, joined)
                .await
        }
        .boxed()
    }

    async fn drive_transition(
        &self,
        request: LifecycleRequest,
        mut joined: Option<watch::Receiver<Option<TransitionOutcome>>>,
    ) -> Result<(), FiberError> {
        let mut prior_error = None;
        loop {
            let mut outcome = joined
                .take()
                .unwrap_or_else(|| self.join_or_start_transition(request));
            let outcome = loop {
                if let Some(outcome) = outcome.borrow_and_update().clone() {
                    break outcome;
                }
                outcome
                    .changed()
                    .await
                    .expect("in-flight fiber transition dropped without an outcome");
            };
            match outcome {
                TransitionOutcome::Completed(Err(error)) => prior_error = Some(error),
                TransitionOutcome::Completed(Ok(())) => {}
                TransitionOutcome::Panicked(report) => report.resume(),
            }
            if !self.needs_follow_up_transition() {
                return prior_error.map_or(Ok(()), Err);
            }
        }
    }

    fn join_or_start_transition(
        &self,
        request: LifecycleRequest,
    ) -> watch::Receiver<Option<TransitionOutcome>> {
        let (transition_id, sender, receiver) = {
            let mut lifecycle = self.inner.lifecycle.lock();
            if let Some(in_flight) = &lifecycle.in_flight {
                return in_flight.outcome.subscribe();
            }
            let transition_id = lifecycle.next_transition_id;
            lifecycle.next_transition_id = lifecycle
                .next_transition_id
                .checked_add(1)
                .expect("fiber transition ID space exhausted");
            let (sender, receiver) = watch::channel(None);
            lifecycle.in_flight = Some(InFlightTransition {
                id: transition_id,
                outcome: sender.clone(),
            });
            (transition_id, sender, receiver)
        };

        let fiber = self.clone();
        tokio::spawn(async move {
            let transition = fiber.run_serialized_transition(request);
            let outcome = match AssertUnwindSafe(transition).catch_unwind().await {
                Ok(result) => TransitionOutcome::Completed(result),
                Err(payload) => TransitionOutcome::Panicked(PanicReport::from_payload(payload)),
            };
            sender.send_replace(Some(outcome));
            let mut lifecycle = fiber.inner.lifecycle.lock();
            if lifecycle
                .in_flight
                .as_ref()
                .is_some_and(|in_flight| in_flight.id == transition_id)
            {
                lifecycle.in_flight = None;
            }
        });

        receiver
    }

    async fn run_serialized_transition(&self, request: LifecycleRequest) -> Result<(), FiberError> {
        let _transition = self.inner.transition.lock().await;
        if self.disposal_requested() {
            return self.dispose_serialized().await;
        }
        if matches!(request, LifecycleRequest::Dispose) {
            panic!("dispose transition lost its sticky disposal request");
        }
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
        match self.state() {
            FiberState::Pending | FiberState::Failed => {
                let changed = {
                    let mut lifecycle = self.inner.lifecycle.lock();
                    lifecycle.current = None;
                    self.transition_to(&mut lifecycle, FiberState::Disposed)
                };
                self.notify_state(changed, FiberState::Disposed);
                Ok(())
            }
            FiberState::Active => self.unload(true).await,
            FiberState::Disposed => Ok(()),
            FiberState::Loading | FiberState::Unloading => {
                panic!("serialized transition lock observed an in-flight fiber state")
            }
        }
    }

    fn needs_follow_up_transition(&self) -> bool {
        let lifecycle = self.inner.lifecycle.lock();
        (lifecycle.dispose_requested && lifecycle.state != FiberState::Disposed)
            || (lifecycle.dependency_invalidated && lifecycle.state == FiberState::Active)
    }

    async fn load(&self) -> Result<(), FiberError> {
        let (generation_id, effects, cancellation, changed) = {
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
            let changed = self.transition_to(&mut lifecycle, FiberState::Loading);
            (generation_id, effects, cancellation, changed)
        };
        self.notify_state(changed, FiberState::Loading);

        let context = (self.inner.context_factory)(Arc::clone(&effects));
        let initializer = match catch_unwind(AssertUnwindSafe(|| {
            (self.inner.initializer)(context, Arc::clone(&self.inner.config))
        })) {
            Ok(initializer) => initializer,
            Err(payload) => self.finish_panicked_loading(generation_id, payload).await,
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
                self.finish_panicked_loading(generation_id, payload).await
            }
        }
    }

    async fn finish_panicked_loading(&self, generation_id: u64, payload: Box<dyn Any + Send>) -> ! {
        match self
            .finish_loading_cleanup(generation_id, FiberState::Disposed, None)
            .await
        {
            Ok(()) => resume_unwind(payload),
            Err(cleanup) => {
                let panic = PanicReport::from_payload(payload);
                panic!("{}; cleanup also failed: {cleanup:?}", panic.message)
            }
        }
    }

    fn try_commit_active(&self, generation_id: u64) -> bool {
        if !(self.inner.dependencies_visible)() {
            let mut lifecycle = self.inner.lifecycle.lock();
            if lifecycle.state == FiberState::Loading {
                Self::close_current_generation(&mut lifecycle);
            }
            return false;
        }
        let mut lifecycle = self.inner.lifecycle.lock();
        let valid = lifecycle.state == FiberState::Loading
            && lifecycle.generation == generation_id
            && !lifecycle.dispose_requested
            && !lifecycle.dependency_invalidated;
        let changed = valid && self.transition_to(&mut lifecycle, FiberState::Active);
        drop(lifecycle);
        self.notify_state(changed, FiberState::Active);
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
        let (target, changed) = {
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
            let changed = self.transition_to(&mut lifecycle, target);
            (target, changed)
        };
        self.notify_state(changed, target);

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
        let (effects, changed) = {
            let mut lifecycle = self.inner.lifecycle.lock();
            assert_eq!(lifecycle.state, FiberState::Active);
            lifecycle.dispose_requested |= disposing;
            let generation = lifecycle
                .current
                .as_ref()
                .expect("active fiber has no owned generation");
            let effects = Arc::clone(&generation.effects);
            let cancellation = generation.cancellation.clone();
            lifecycle.generation = lifecycle
                .generation
                .checked_add(1)
                .expect("fiber generation space exhausted");
            effects.close();
            cancellation.cancel();
            let changed = self.transition_to(&mut lifecycle, FiberState::Unloading);
            (effects, changed)
        };
        self.notify_state(changed, FiberState::Unloading);

        let cleanup = effects.close_and_dispose().await;
        let (target, changed) = {
            let mut lifecycle = self.inner.lifecycle.lock();
            lifecycle.current = None;
            let target = if lifecycle.dispose_requested {
                FiberState::Disposed
            } else {
                FiberState::Pending
            };
            let changed = self.transition_to(&mut lifecycle, target);
            (target, changed)
        };
        self.notify_state(changed, target);
        cleanup.map_err(FiberError::Cleanup)
    }

    fn signal_dependency_change(&self) -> Option<watch::Receiver<Option<TransitionOutcome>>> {
        let dependencies_visible = (self.inner.dependencies_visible)();
        let mut lifecycle = self.inner.lifecycle.lock();
        if !dependencies_visible {
            match lifecycle.state {
                FiberState::Active => {
                    lifecycle.dependency_invalidated = true;
                    Self::close_current_generation(&mut lifecycle);
                }
                FiberState::Loading => Self::close_current_generation(&mut lifecycle),
                FiberState::Pending
                | FiberState::Failed
                | FiberState::Unloading
                | FiberState::Disposed => {}
            }
        }
        Self::subscribe_locked(&lifecycle)
    }

    fn request_disposal(&self) -> Option<watch::Receiver<Option<TransitionOutcome>>> {
        let mut lifecycle = self.inner.lifecycle.lock();
        lifecycle.dispose_requested = true;
        if matches!(lifecycle.state, FiberState::Active | FiberState::Loading) {
            Self::close_current_generation(&mut lifecycle);
        }
        Self::subscribe_locked(&lifecycle)
    }

    fn close_current_generation(lifecycle: &mut Lifecycle) {
        let generation = lifecycle
            .current
            .as_ref()
            .expect("active or loading fiber has no owned generation");
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

    fn subscribe_locked(
        lifecycle: &Lifecycle,
    ) -> Option<watch::Receiver<Option<TransitionOutcome>>> {
        lifecycle
            .in_flight
            .as_ref()
            .map(|in_flight| in_flight.outcome.subscribe())
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

    fn transition_to(&self, lifecycle: &mut Lifecycle, state: FiberState) -> bool {
        if lifecycle.state == state {
            return false;
        }
        lifecycle.state = state;
        self.inner.trace.send_modify(|trace| trace.push(state));
        true
    }

    fn notify_state(&self, changed: bool, state: FiberState) {
        if changed {
            (self.inner.state_observer)(state);
        }
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
