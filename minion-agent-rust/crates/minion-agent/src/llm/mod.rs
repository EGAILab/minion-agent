//! Provider-neutral LLM vocabulary and streaming boundary.
//!
//! Adapters decode provider protocols into typed raw chunks. [`AssistantStream`]
//! owns Minion's provider-neutral settlement and fusion rules.

mod adapter;
mod assistant_stream;
mod model;
mod scripted;
mod service;
mod transform;
mod transform_compat;
mod vocabulary;

pub use adapter::*;
pub use assistant_stream::AssistantStream;
pub use model::{ModelIdentity, ModelIdentityError};
pub use scripted::{Script, ScriptItem, ScriptedAdapter};
pub use service::{LlmService, LlmStartError};
pub use transform::{ToolCallIdNormalizer, TransformTarget, transform_messages};
pub use transform_compat::{TransformCompatError, transform_legacy_messages};
pub use vocabulary::*;
