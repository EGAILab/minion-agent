use std::{fmt, sync::Arc};

use futures::future::BoxFuture;
use serde_json::Value;
use thiserror::Error;

use crate::llm::{
    ConstrainedSampling, JsonSchemaObject, ToolResultContentBlock, ToolSchema, Usage,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentToolResult {
    pub content: Vec<ToolResultContentBlock>,
    pub details: Value,
    pub usage: Option<Usage>,
    pub added_tool_names: Option<Vec<String>>,
    pub terminate: Option<bool>,
}

pub trait ToolExecutionSignal: Send + Sync + 'static {
    fn is_cancelled(&self) -> bool;
}

pub type ToolUpdateCallback = Arc<dyn Fn(AgentToolResult) + Send + Sync + 'static>;

pub struct ToolExecutionRequest {
    pub tool_call_id: String,
    pub params: Value,
    pub signal: Option<Arc<dyn ToolExecutionSignal>>,
    pub on_update: Option<ToolUpdateCallback>,
}

pub type PrepareArguments =
    Arc<dyn Fn(Value) -> Result<Value, ToolCapabilityError> + Send + Sync + 'static>;
pub type ExecuteTool = Arc<
    dyn Fn(ToolExecutionRequest) -> BoxFuture<'static, Result<AgentToolResult, ToolCapabilityError>>
        + Send
        + Sync
        + 'static,
>;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("tool capability failed: {message}")]
pub struct ToolCapabilityError {
    message: String,
}

impl ToolCapabilityError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone)]
pub struct ToolDefinition {
    name: String,
    description: String,
    parameters: JsonSchemaObject,
    constrained_sampling: Option<ConstrainedSampling>,
    label: String,
    prepare_arguments: Option<PrepareArguments>,
    execute: ExecuteTool,
    execution_mode: Option<ExecutionMode>,
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
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            constrained_sampling: None,
            label: label.into(),
            prepare_arguments: None,
            execute: Arc::new(execute),
            execution_mode: None,
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
        self.prepare_arguments = Some(Arc::new(prepare));
        self
    }

    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
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

    pub fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            constrained_sampling: self.constrained_sampling.clone(),
        }
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
