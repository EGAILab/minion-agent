//! Provider-neutral LLM vocabulary and streaming boundary.
//!
//! Adapters decode provider protocols into typed raw chunks. [`AssistantStream`]
//! owns Minion's provider-neutral settlement and fusion rules.

mod adapter;
mod assistant_stream;
mod model;
mod raw;
mod result_value;
mod scripted;
mod service;
mod transform;
#[cfg(feature = "conformance")]
mod transform_compat;
mod vocabulary;

pub use adapter::*;
pub use assistant_stream::AssistantStream;
pub use model::{ModelIdentity, ModelIdentityError};
pub use raw::*;
pub use result_value::*;
pub use scripted::{Script, ScriptItem, ScriptedAdapter};
pub use service::{LlmRegistration, LlmService, LlmStartError};
pub use transform::{ToolCallIdNormalizer, TransformTarget, transform_messages};
#[cfg(feature = "conformance")]
pub use transform_compat::{TransformCompatError, transform_legacy_messages};
pub use vocabulary::*;
