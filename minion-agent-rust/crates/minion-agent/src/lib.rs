#![forbid(unsafe_code)]

pub mod runtime;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use runtime::{
    DispatchMode, DisposeError, DisposeErrors, DynPluginSpec, EffectHandle, EffectStore, EventBus,
    EventError, EventListenerError, EventListenerHandle, EventName, EventSpec, FiberError,
    FiberHandle, FiberInitContext, FiberState, Next, ParallelErrors, PluginConfigError,
    PluginInitError, PluginSpec, RegistrationHandle, RuntimeError, ScopeHandle, ScopeId, ScopeTree,
    ScopedRegistry, Service, ServiceCheck, ServiceName, ServiceOwner, ServiceRegistration,
    ServiceRegistry, WaterfallError,
};
