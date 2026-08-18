#![forbid(unsafe_code)]

pub mod runtime;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use runtime::{
    DisposeError, DisposeErrors, EffectHandle, EffectStore, EventName, RegistrationHandle,
    RuntimeError, ScopeHandle, ScopeId, ScopeTree, ScopedRegistry, ServiceName,
};
