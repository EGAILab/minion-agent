use std::{
    any::{Any, TypeId, type_name},
    collections::HashMap,
    future::Future,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::future::{BoxFuture, FutureExt, join_all, ready};
use parking_lot::Mutex;
use thiserror::Error;

use super::{DisposeError, EffectStore, EventName, RuntimeError, ScopeHandle, ScopeId, ScopeTree};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchMode {
    Emit,
    Parallel,
    Serial,
    Waterfall,
}

pub struct EventSpec<P, R> {
    name: EventName,
    mode: DispatchMode,
    terminal: Arc<dyn Fn(&P) -> R + Send + Sync>,
}

impl<P, R> EventSpec<P, R> {
    pub fn new(
        name: EventName,
        mode: DispatchMode,
        terminal: impl Fn(&P) -> R + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            mode,
            terminal: Arc::new(terminal),
        }
    }

    pub fn name(&self) -> &EventName {
        &self.name
    }

    pub fn mode(&self) -> DispatchMode {
        self.mode
    }
}

impl<P, R> Clone for EventSpec<P, R> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            mode: self.mode,
            terminal: Arc::clone(&self.terminal),
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("event listener failed: {message}")]
pub struct EventListenerError {
    pub message: String,
}

impl EventListenerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("parallel event listeners failed")]
pub struct ParallelErrors(Vec<EventListenerError>);

impl ParallelErrors {
    pub fn as_slice(&self) -> &[EventListenerError] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<EventListenerError> {
        self.0
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WaterfallError {
    #[error("waterfall `next` may be called at most once per listener")]
    NextAlreadyCalled,
}

#[derive(Debug, Error)]
pub enum EventError {
    #[error("event {name} is not declared")]
    Undeclared { name: EventName },
    #[error("event {name} is already declared as {expected:?}, not {actual:?}")]
    DeclarationModeMismatch {
        name: EventName,
        expected: DispatchMode,
        actual: DispatchMode,
    },
    #[error(
        "event {name} has Rust contract ({expected_payload}, {expected_result}), not ({actual_payload}, {actual_result})"
    )]
    DeclarationTypeMismatch {
        name: EventName,
        expected_payload: &'static str,
        expected_result: &'static str,
        actual_payload: &'static str,
        actual_result: &'static str,
    },
    #[error("event {name} requires {expected:?} dispatch, not {actual:?}")]
    DispatchModeMismatch {
        name: EventName,
        expected: DispatchMode,
        actual: DispatchMode,
    },
    #[error("scope {scope} belongs to another scope tree")]
    ForeignScope { scope: u64 },
    #[error("scope {scope} is inactive")]
    InactiveScope { scope: u64 },
    #[error(transparent)]
    Lifecycle(#[from] RuntimeError),
    #[error(transparent)]
    Parallel(#[from] ParallelErrors),
    #[error(transparent)]
    Waterfall(#[from] WaterfallError),
}

#[derive(Clone)]
pub struct EventBus {
    tree: ScopeTree,
    state: Arc<Mutex<EventState>>,
}

struct EventState {
    declarations: HashMap<EventName, EventDeclaration>,
}

struct EventDeclaration {
    mode: DispatchMode,
    payload_type: TypeId,
    result_type: TypeId,
    payload_name: &'static str,
    result_name: &'static str,
    terminal: ErasedValue,
    listeners: Vec<Option<ListenerEntry>>,
}

type ErasedValue = Arc<dyn Any + Send + Sync>;

struct ListenerEntry {
    scope: Option<ScopeId>,
    callback: ErasedValue,
}

struct Terminal<P, R>(Arc<dyn Fn(&P) -> R + Send + Sync>);
struct EmitCallback<P>(Arc<dyn Fn(&P) + Send + Sync>);
struct ParallelCallback<P>(
    Arc<dyn Fn(P) -> BoxFuture<'static, Result<(), EventListenerError>> + Send + Sync>,
);
struct SerialCallback<P, R>(Arc<dyn Fn(P) -> BoxFuture<'static, R> + Send + Sync>);
type WaterfallListener<P, R> =
    dyn Fn(P, Next<P, R>) -> BoxFuture<'static, Result<R, WaterfallError>> + Send + Sync;
struct WaterfallCallback<P, R>(Arc<WaterfallListener<P, R>>);
type DispatchSnapshot<P, R, C> = (Vec<Arc<C>>, Arc<Terminal<P, R>>);

pub struct EventListenerHandle {
    removal: Arc<ListenerRemoval>,
}

struct ListenerRemoval {
    remove: StdMutex<Option<Box<dyn FnOnce() + Send>>>,
}

pub struct Next<P, R> {
    used: AtomicBool,
    current: P,
    continuation: Arc<dyn Fn(P) -> BoxFuture<'static, Result<R, WaterfallError>> + Send + Sync>,
}

impl EventBus {
    pub fn new(tree: impl Into<ScopeTree>) -> Self {
        Self {
            tree: tree.into(),
            state: Arc::new(Mutex::new(EventState {
                declarations: HashMap::new(),
            })),
        }
    }

    pub fn declare<P, R>(&self, spec: &EventSpec<P, R>) -> Result<(), EventError>
    where
        P: 'static,
        R: 'static,
    {
        let mut state = self.state.lock();
        if let Some(declaration) = state.declarations.get(spec.name()) {
            validate_declaration::<P, R>(declaration, spec)?;
            return Ok(());
        }

        state.declarations.insert(
            spec.name.clone(),
            EventDeclaration {
                mode: spec.mode,
                payload_type: TypeId::of::<P>(),
                result_type: TypeId::of::<R>(),
                payload_name: type_name::<P>(),
                result_name: type_name::<R>(),
                terminal: Arc::new(Terminal(Arc::clone(&spec.terminal))),
                listeners: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn on_emit<P, F>(
        &self,
        spec: &EventSpec<P, ()>,
        effects: &EffectStore,
        scope: Option<&ScopeHandle>,
        listener: F,
    ) -> Result<EventListenerHandle, EventError>
    where
        P: 'static,
        F: Fn(&P) + Send + Sync + 'static,
    {
        self.register_callback(
            spec,
            DispatchMode::Emit,
            effects,
            scope,
            EmitCallback(Arc::new(listener)),
        )
    }

    pub fn on_parallel<P, F, Fut>(
        &self,
        spec: &EventSpec<P, ()>,
        effects: &EffectStore,
        scope: Option<&ScopeHandle>,
        listener: F,
    ) -> Result<EventListenerHandle, EventError>
    where
        P: 'static,
        F: Fn(P) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), EventListenerError>> + Send + 'static,
    {
        self.register_callback(
            spec,
            DispatchMode::Parallel,
            effects,
            scope,
            ParallelCallback(Arc::new(move |payload| listener(payload).boxed())),
        )
    }

    pub fn on_serial<P, R, F, Fut>(
        &self,
        spec: &EventSpec<P, R>,
        effects: &EffectStore,
        scope: Option<&ScopeHandle>,
        listener: F,
    ) -> Result<EventListenerHandle, EventError>
    where
        P: 'static,
        R: 'static,
        F: Fn(P) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
    {
        self.register_callback(
            spec,
            DispatchMode::Serial,
            effects,
            scope,
            SerialCallback(Arc::new(move |payload| listener(payload).boxed())),
        )
    }

    pub fn on_waterfall<P, R, F, Fut>(
        &self,
        spec: &EventSpec<P, R>,
        effects: &EffectStore,
        scope: Option<&ScopeHandle>,
        listener: F,
    ) -> Result<EventListenerHandle, EventError>
    where
        P: 'static,
        R: 'static,
        F: Fn(P, Next<P, R>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, WaterfallError>> + Send + 'static,
    {
        self.register_callback(
            spec,
            DispatchMode::Waterfall,
            effects,
            scope,
            WaterfallCallback(Arc::new(move |payload, next| {
                listener(payload, next).boxed()
            })),
        )
    }

    pub fn emit<P>(
        &self,
        spec: &EventSpec<P, ()>,
        payload: &P,
        scope: Option<&ScopeHandle>,
    ) -> Result<(), EventError>
    where
        P: 'static,
    {
        let (callbacks, _) =
            self.snapshot::<P, (), EmitCallback<P>>(spec, DispatchMode::Emit, scope)?;
        for callback in callbacks {
            (callback.0)(payload);
        }
        Ok(())
    }

    pub async fn parallel<P>(
        &self,
        spec: &EventSpec<P, ()>,
        payload: P,
        scope: Option<&ScopeHandle>,
    ) -> Result<(), EventError>
    where
        P: Clone + Send + 'static,
    {
        let (callbacks, _) =
            self.snapshot::<P, (), ParallelCallback<P>>(spec, DispatchMode::Parallel, scope)?;
        let outcomes = join_all(
            callbacks
                .into_iter()
                .map(|callback| (callback.0)(payload.clone())),
        )
        .await;
        let errors: Vec<_> = outcomes.into_iter().filter_map(Result::err).collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ParallelErrors(errors).into())
        }
    }

    pub async fn serial<P, R>(
        &self,
        spec: &EventSpec<P, R>,
        payload: P,
        scope: Option<&ScopeHandle>,
    ) -> Result<Option<R>, EventError>
    where
        P: Clone + 'static,
        R: 'static,
    {
        let (callbacks, _) =
            self.snapshot::<P, R, SerialCallback<P, R>>(spec, DispatchMode::Serial, scope)?;
        let mut result = None;
        for callback in callbacks {
            result = Some((callback.0)(payload.clone()).await);
        }
        Ok(result)
    }

    pub async fn waterfall<P, R>(
        &self,
        spec: &EventSpec<P, R>,
        payload: P,
        scope: Option<&ScopeHandle>,
    ) -> Result<R, EventError>
    where
        P: Clone + Send + 'static,
        R: Send + 'static,
    {
        let (callbacks, terminal) =
            self.snapshot::<P, R, WaterfallCallback<P, R>>(spec, DispatchMode::Waterfall, scope)?;
        run_waterfall(callbacks.into(), terminal, 0, payload)
            .await
            .map_err(EventError::from)
    }

    fn register_callback<P, R, C>(
        &self,
        spec: &EventSpec<P, R>,
        mode: DispatchMode,
        effects: &EffectStore,
        scope: Option<&ScopeHandle>,
        callback: C,
    ) -> Result<EventListenerHandle, EventError>
    where
        P: 'static,
        R: 'static,
        C: Send + Sync + 'static,
    {
        let scope_id = self.registration_scope(scope)?;
        let entry_index =
            {
                let mut state = self.state.lock();
                let declaration = state.declarations.get_mut(spec.name()).ok_or_else(|| {
                    EventError::Undeclared {
                        name: spec.name.clone(),
                    }
                })?;
                validate_dispatch::<P, R>(declaration, spec, mode)?;
                let index = declaration.listeners.len();
                declaration.listeners.push(Some(ListenerEntry {
                    scope: scope_id,
                    callback: Arc::new(callback),
                }));
                index
            };

        let state = Arc::clone(&self.state);
        let name = spec.name.clone();
        let removal = Arc::new(ListenerRemoval {
            remove: StdMutex::new(Some(Box::new(move || {
                let mut state = state.lock();
                if let Some(declaration) = state.declarations.get_mut(&name) {
                    declaration.listeners[entry_index] = None;
                }
            }))),
        });
        let handle = EventListenerHandle {
            removal: Arc::clone(&removal),
        };
        let owned_removal = Arc::clone(&removal);
        if let Err(error) = effects.push(format!("event listener {}", spec.name), move || {
            Box::pin(async move {
                owned_removal.remove();
                Ok(())
            })
        }) {
            handle.remove();
            return Err(error.into());
        }
        Ok(handle)
    }

    fn snapshot<P, R, C>(
        &self,
        spec: &EventSpec<P, R>,
        mode: DispatchMode,
        scope: Option<&ScopeHandle>,
    ) -> Result<DispatchSnapshot<P, R, C>, EventError>
    where
        P: 'static,
        R: 'static,
        C: Send + Sync + 'static,
    {
        let admitted_scopes = match scope {
            Some(scope) => {
                self.ensure_scope_belongs(scope)?;
                self.tree.active_ancestor_chain(scope.id())
            }
            None => None,
        };
        let state = self.state.lock();
        let declaration =
            state
                .declarations
                .get(spec.name())
                .ok_or_else(|| EventError::Undeclared {
                    name: spec.name.clone(),
                })?;
        validate_dispatch::<P, R>(declaration, spec, mode)?;
        let callbacks = declaration
            .listeners
            .iter()
            .filter_map(Option::as_ref)
            .filter(|entry| {
                entry.scope.is_none()
                    || admitted_scopes.as_ref().is_some_and(|scopes| {
                        entry.scope.is_some_and(|scope| scopes.contains(&scope))
                    })
            })
            .map(|entry| {
                Arc::clone(&entry.callback)
                    .downcast::<C>()
                    .expect("validated event callback contract")
            })
            .collect();
        let terminal = Arc::clone(&declaration.terminal)
            .downcast::<Terminal<P, R>>()
            .expect("validated event terminal contract");
        Ok((callbacks, terminal))
    }

    fn registration_scope(
        &self,
        scope: Option<&ScopeHandle>,
    ) -> Result<Option<ScopeId>, EventError> {
        let Some(scope) = scope else {
            return Ok(None);
        };
        self.ensure_scope_belongs(scope)?;
        if !scope.is_active() {
            return Err(EventError::InactiveScope {
                scope: scope.id().as_u64(),
            });
        }
        Ok(Some(scope.id()))
    }

    fn ensure_scope_belongs(&self, scope: &ScopeHandle) -> Result<(), EventError> {
        if scope.belongs_to(&self.tree) {
            Ok(())
        } else {
            Err(EventError::ForeignScope {
                scope: scope.id().as_u64(),
            })
        }
    }
}

impl EventListenerHandle {
    pub async fn dispose(&self) -> Result<(), DisposeError> {
        self.remove();
        Ok(())
    }

    fn remove(&self) {
        self.removal.remove();
    }
}

impl ListenerRemoval {
    fn remove(&self) {
        if let Some(remove) = self
            .remove
            .lock()
            .expect("listener removal lock poisoned")
            .take()
        {
            remove();
        }
    }
}

impl<P, R> Next<P, R>
where
    P: Clone,
    R: Send + 'static,
{
    pub fn call(&self, replacement: Option<P>) -> BoxFuture<'static, Result<R, WaterfallError>> {
        if self.used.swap(true, Ordering::AcqRel) {
            return ready(Err(WaterfallError::NextAlreadyCalled)).boxed();
        }
        (self.continuation)(replacement.unwrap_or_else(|| self.current.clone()))
    }
}

fn validate_declaration<P, R>(
    declaration: &EventDeclaration,
    spec: &EventSpec<P, R>,
) -> Result<(), EventError>
where
    P: 'static,
    R: 'static,
{
    if declaration.mode != spec.mode {
        return Err(EventError::DeclarationModeMismatch {
            name: spec.name.clone(),
            expected: declaration.mode,
            actual: spec.mode,
        });
    }
    validate_contract::<P, R>(declaration, spec.name())
}

fn validate_dispatch<P, R>(
    declaration: &EventDeclaration,
    spec: &EventSpec<P, R>,
    mode: DispatchMode,
) -> Result<(), EventError>
where
    P: 'static,
    R: 'static,
{
    if spec.mode != mode {
        return Err(EventError::DispatchModeMismatch {
            name: spec.name.clone(),
            expected: mode,
            actual: spec.mode,
        });
    }
    if declaration.mode != mode {
        return Err(EventError::DispatchModeMismatch {
            name: spec.name.clone(),
            expected: mode,
            actual: declaration.mode,
        });
    }
    validate_contract::<P, R>(declaration, spec.name())
}

fn validate_contract<P, R>(
    declaration: &EventDeclaration,
    name: &EventName,
) -> Result<(), EventError>
where
    P: 'static,
    R: 'static,
{
    if declaration.payload_type == TypeId::of::<P>() && declaration.result_type == TypeId::of::<R>()
    {
        return Ok(());
    }
    Err(EventError::DeclarationTypeMismatch {
        name: name.clone(),
        expected_payload: declaration.payload_name,
        expected_result: declaration.result_name,
        actual_payload: type_name::<P>(),
        actual_result: type_name::<R>(),
    })
}

fn run_waterfall<P, R>(
    callbacks: Arc<[Arc<WaterfallCallback<P, R>>]>,
    terminal: Arc<Terminal<P, R>>,
    index: usize,
    payload: P,
) -> BoxFuture<'static, Result<R, WaterfallError>>
where
    P: Clone + Send + 'static,
    R: Send + 'static,
{
    async move {
        let Some(callback) = callbacks.get(index).cloned() else {
            return Ok((terminal.0)(&payload));
        };
        let continuation_callbacks = Arc::clone(&callbacks);
        let continuation_terminal = Arc::clone(&terminal);
        let continuation = Arc::new(move |replacement| {
            run_waterfall(
                Arc::clone(&continuation_callbacks),
                Arc::clone(&continuation_terminal),
                index + 1,
                replacement,
            )
        });
        let next = Next {
            used: AtomicBool::new(false),
            current: payload.clone(),
            continuation,
        };
        (callback.0)(payload, next).await
    }
    .boxed()
}
