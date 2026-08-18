use thiserror::Error;

use super::ServiceName;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("invalid normative name {0:?}")]
    InvalidName(String),
    #[error("cannot create effect on inactive owner {owner}")]
    InactiveOwner { owner: String },
    #[error("service {name} is already provided by {holder}")]
    ServiceConflict { name: ServiceName, holder: String },
    #[error("service {name} has Rust contract {expected}, not {actual}")]
    ServiceTypeMismatch {
        name: ServiceName,
        expected: &'static str,
        actual: &'static str,
    },
}
