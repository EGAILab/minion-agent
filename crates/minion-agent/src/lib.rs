#![forbid(unsafe_code)]

pub mod runtime;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use runtime::{
    DispatchMode, DisposeError, DisposeErrors, EffectHandle, EffectStore, EventBus, EventError,
    EventListenerError, EventListenerHandle, EventName, EventSpec, Next, ParallelErrors,
    RegistrationHandle, RuntimeError, ScopeHandle, ScopeId, ScopeTree, ScopedRegistry, Service,
    ServiceCheck, ServiceName, ServiceOwner, ServiceRegistration, ServiceRegistry, WaterfallError,
};
