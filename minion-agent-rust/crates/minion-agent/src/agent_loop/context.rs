use std::sync::Arc;

use crate::{
    agent::ThinkingLevel,
    llm::{Message, ModelIdentity},
    tools::ToolDefinition,
};

/// Invocation-local model context.
///
/// The vectors are owned top-level snapshots. Tool capabilities retain their
/// existing shared identity through [`Arc`].
#[derive(Clone, Debug)]
pub struct RunContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<ToolDefinition>>,
}

/// Invocation-local provider configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunConfig {
    pub model: ModelIdentity,
    pub thinking_level: ThinkingLevel,
}

/// Optional replacements applied to the next request in the current run.
///
/// `context`, when present, replaces the whole context rather than merging
/// individual fields. No replacement is persisted to the Agent instance.
#[derive(Clone, Debug, Default)]
pub struct RunConfigUpdate {
    pub context: Option<RunContext>,
    pub model: Option<ModelIdentity>,
    pub thinking_level: Option<ThinkingLevel>,
}
