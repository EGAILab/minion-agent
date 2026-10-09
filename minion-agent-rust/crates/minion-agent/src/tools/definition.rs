use std::{fmt, sync::Arc};

use super::{PreparedValue, RuntimeSchemaError, RuntimeSchemaObject};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::llm::{
    ConstrainedSampling, JsonSchemaObject, ToolResultContentBlock, ToolSchema, Usage,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentToolResult {
    pub content: Vec<ToolResultContentBlock>,
    pub details: crate::llm::ResultValue,
    pub usage: Option<Usage>,
    pub added_tool_names: Option<Vec<String>>,
    pub terminate: Option<bool>,
}

pub trait ToolExecutionSignal: Send + Sync + 'static {
    fn is_cancelled(&self) -> bool;
}

/// Cloneable read-only signal view exposed to tool hooks.
#[derive(Clone)]
pub struct ExecutionSignal(Arc<dyn ToolExecutionSignal>);

impl ExecutionSignal {
    pub fn new(signal: Arc<dyn ToolExecutionSignal>) -> Self {
        Self(signal)
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

impl fmt::Debug for ExecutionSignal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionSignal")
            .field("is_cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl PartialEq for ExecutionSignal {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ExecutionSignal {}

pub type ToolUpdateCallback = Arc<dyn Fn(AgentToolResult) + Send + Sync + 'static>;

/// Immutable, per-execute projection of the executing agent's authoritative state (TOOL-042).
/// No agent reference or mutable access is exposed. Absence is distinct from a present `off`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolExecutionContext {
    session_id: String,
    session_file: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    reasoning_level: Option<String>,
}

impl ToolExecutionContext {
    pub fn new(
        session_id: String,
        session_file: Option<String>,
        provider: Option<String>,
        model: Option<String>,
        reasoning_level: Option<String>,
    ) -> Self {
        Self {
            session_id,
            session_file,
            provider,
            model,
            reasoning_level,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn session_file(&self) -> Option<&str> {
        self.session_file.as_deref()
    }
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
    pub fn reasoning_level(&self) -> Option<&str> {
        self.reasoning_level.as_deref()
    }
}

/// Explicit per-call factory. A typed failure settles as an ordinary tool execution failure.
pub type ToolContextProvider = Arc<
    dyn Fn() -> Result<Option<ToolExecutionContext>, ToolCapabilityError> + Send + Sync + 'static,
>;

pub struct ToolExecutionRequest {
    pub tool_call_id: String,
    pub params: PreparedValue,
    pub signal: Option<Arc<dyn ToolExecutionSignal>>,
    pub on_update: Option<ToolUpdateCallback>,
    /// Sampled at execute invocation, after preflight and hooks; absent outside an agent.
    pub context: Option<ToolExecutionContext>,
}

pub type PrepareArguments = Arc<
    dyn Fn(crate::llm::RawValue) -> Result<PreparedValue, ToolCapabilityError>
        + Send
        + Sync
        + 'static,
>;
pub type ExecuteTool = Arc<
    dyn Fn(ToolExecutionRequest) -> BoxFuture<'static, Result<AgentToolResult, ToolCapabilityError>>
        + Send
        + Sync
        + 'static,
>;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("tool capability failed: {message}")]
pub struct ToolCapabilityError {
    message: crate::llm::ResultString,
}

impl ToolCapabilityError {
    pub fn new(message: impl Into<crate::llm::ResultString>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the semantic capability error message without a Rust error-type prefix.
    pub fn message(&self) -> &crate::llm::ResultString {
        &self.message
    }
}

#[derive(Clone)]
pub struct ToolDefinition {
    name: String,
    description: String,
    parameters: RuntimeSchemaObject,
    constrained_sampling: Option<ConstrainedSampling>,
    label: String,
    prepare_arguments: Option<PrepareArguments>,
    execute: ExecuteTool,
    execution_mode: Option<ExecutionMode>,
    prompt_snippet: Option<String>,
    prompt_guidelines: Option<Vec<String>>,
}

impl ToolDefinition {
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonSchemaObject,
        label: impl Into<String>,
        execute: F,
    ) -> Self
    where
        F: Fn(
                ToolExecutionRequest,
            ) -> BoxFuture<'static, Result<AgentToolResult, ToolCapabilityError>>
            + Send
            + Sync
            + 'static,
    {
        Self::new_with_runtime_schema(name, description, parameters.into(), label, execute)
    }

    /// Register a lossless JavaScript-string schema for runtime validation.
    /// No scalar projection or surrogate normalization occurs here.
    pub fn new_with_runtime_schema<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: RuntimeSchemaObject,
        label: impl Into<String>,
        execute: F,
    ) -> Self
    where
        F: Fn(
                ToolExecutionRequest,
            ) -> BoxFuture<'static, Result<AgentToolResult, ToolCapabilityError>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            constrained_sampling: None,
            label: label.into(),
            prepare_arguments: None,
            execute: Arc::new(execute),
            execution_mode: None,
            prompt_snippet: None,
            prompt_guidelines: None,
        }
    }

    pub fn with_constrained_sampling(mut self, value: ConstrainedSampling) -> Self {
        self.constrained_sampling = Some(value);
        self
    }

    pub fn with_prepare_arguments<F>(mut self, prepare: F) -> Self
    where
        F: Fn(Value) -> Result<Value, ToolCapabilityError> + Send + Sync + 'static,
    {
        self.prepare_arguments = Some(Arc::new(move |raw| {
            let json = raw
                .try_to_json()
                .map_err(|error| ToolCapabilityError::new(error.to_string()))?;
            prepare(json).map(PreparedValue::from)
        }));
        self
    }

    /// JSON-input compatibility adapter. Use `with_prepare_raw_arguments` for
    /// a callback accepting the full UTF-16/binary64 raw domain.
    pub fn with_prepare_runtime_arguments<F>(mut self, prepare: F) -> Self
    where
        F: Fn(Value) -> Result<PreparedValue, ToolCapabilityError> + Send + Sync + 'static,
    {
        self.prepare_arguments = Some(Arc::new(move |raw| {
            let json = raw
                .try_to_json()
                .map_err(|error| ToolCapabilityError::new(error.to_string()))?;
            prepare(json)
        }));
        self
    }

    /// Receive raw arguments unchanged, including non-scalar strings/keys,
    /// signed zero and infinities. Preparation owns any subsequent conversion.
    pub fn with_prepare_raw_arguments<F>(mut self, prepare: F) -> Self
    where
        F: Fn(crate::llm::RawValue) -> Result<PreparedValue, ToolCapabilityError>
            + Send
            + Sync
            + 'static,
    {
        self.prepare_arguments = Some(Arc::new(prepare));
        self
    }

    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }

    /// Optional model-facing prose; never part of the executable tool schema.
    pub fn with_prompt_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.prompt_snippet = Some(snippet.into());
        self
    }

    pub fn with_prompt_guidelines(mut self, guidelines: Vec<String>) -> Self {
        self.prompt_guidelines = Some(guidelines);
        self
    }

    pub fn prompt_snippet(&self) -> Option<&str> {
        self.prompt_snippet.as_deref()
    }

    pub fn prompt_guidelines(&self) -> Option<&[String]> {
        self.prompt_guidelines.as_deref()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn prepare_arguments(&self) -> Option<&PrepareArguments> {
        self.prepare_arguments.as_ref()
    }

    pub fn execute(&self) -> &ExecuteTool {
        &self.execute
    }

    pub fn execution_mode(&self) -> Option<ExecutionMode> {
        self.execution_mode
    }

    pub fn parameters(&self) -> &RuntimeSchemaObject {
        &self.parameters
    }

    /// Scalar-only provider projection. Non-scalar schemas remain fully usable
    /// by the runtime, but cannot silently pass through a serde_json boundary.
    pub fn schema(&self) -> Result<ToolSchema, RuntimeSchemaError> {
        Ok(ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.try_to_json()?,
            constrained_sampling: self.constrained_sampling.clone(),
        })
    }
}

impl fmt::Debug for ToolDefinition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("constrained_sampling", &self.constrained_sampling)
            .field("label", &self.label)
            .field("has_prepare_arguments", &self.prepare_arguments.is_some())
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}
