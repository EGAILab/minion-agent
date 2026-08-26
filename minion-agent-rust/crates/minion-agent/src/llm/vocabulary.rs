use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use super::ModelIdentity;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextBlock {
    #[serde(rename = "type")]
    kind: TextKind,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            kind: TextKind::Text,
            text: text.into(),
            text_signature: None,
        }
    }

    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.text_signature = Some(signature.into());
        self
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TextKind {
    Text,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ThinkingBlock {
    #[serde(rename = "type")]
    kind: ThinkingKind,
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(default)]
    pub redacted: bool,
}

impl ThinkingBlock {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            kind: ThinkingKind::Thinking,
            thinking: thinking.into(),
            thinking_signature: None,
            redacted: false,
        }
    }

    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.thinking_signature = Some(signature.into());
        self
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ThinkingKind {
    Thinking,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ImageSource {
    Data { data: String },
    Reference { reference: ArtifactHash },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ArtifactHash(String);

impl ArtifactHash {
    pub fn new(value: impl Into<String>) -> Result<Self, ArtifactHashError> {
        let value = value.into();
        let digest = value.strip_prefix("sha256:").ok_or(ArtifactHashError)?;
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(ArtifactHashError);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ArtifactHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("artifact identity must be sha256 followed by 64 hexadecimal digits")]
pub struct ArtifactHashError;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageBlock {
    #[serde(rename = "type")]
    kind: ImageKind,
    pub mime_type: String,
    #[serde(flatten)]
    pub source: ImageSource,
}

impl ImageBlock {
    pub fn data(mime_type: impl Into<String>, base64_data: impl Into<String>) -> Self {
        Self {
            kind: ImageKind::Image,
            mime_type: mime_type.into(),
            source: ImageSource::Data {
                data: base64_data.into(),
            },
        }
    }

    pub fn reference(mime_type: impl Into<String>, reference: ArtifactHash) -> Self {
        Self {
            kind: ImageKind::Image,
            mime_type: mime_type.into(),
            source: ImageSource::Reference { reference },
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImageKind {
    Image,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCall {
    #[serde(rename = "type")]
    kind: ToolCallKind,
    pub id: String,
    pub name: String,
    pub arguments: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: BTreeMap<String, Value>,
    ) -> Self {
        Self {
            kind: ToolCallKind::ToolCall,
            id: id.into(),
            name: name.into(),
            arguments,
            thought_signature: None,
            namespace: None,
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ToolCallKind {
    ToolCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<UserContentBlock>),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UserContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum AssistantContentBlock {
    Text(TextBlock),
    Thinking(ThinkingBlock),
    ToolCall(ToolCall),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolResultContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UserMessage {
    role: UserRole,
    pub content: UserContent,
    pub timestamp: f64,
}

impl UserMessage {
    pub fn new(content: UserContent, timestamp: f64) -> Self {
        Self {
            role: UserRole::User,
            content,
            timestamp,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum UserRole {
    User,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: Cost,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeferredHandle {
    pub provider: String,
    pub model_id: String,
    pub api: String,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum DiagnosticCode {
    String(String),
    Number(f64),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DiagnosticError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<DiagnosticCode>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub diagnostic_type: String,
    pub timestamp: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AssistantMessage {
    role: AssistantRole,
    pub content: Vec<AssistantContentBlock>,
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    pub timestamp: f64,
}

impl AssistantMessage {
    pub fn new(
        identity: ModelIdentity,
        content: Vec<AssistantContentBlock>,
        usage: Usage,
        stop_reason: StopReason,
        timestamp: f64,
    ) -> Self {
        Self {
            role: AssistantRole::Assistant,
            content,
            api: identity.api().into(),
            provider: identity.provider().into(),
            model: identity.model_id().into(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage,
            stop_reason,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp,
        }
    }

    pub fn pending(identity: ModelIdentity, timestamp: f64) -> Self {
        Self::new(
            identity,
            Vec::new(),
            Usage::default(),
            StopReason::Pending,
            timestamp,
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AssistantRole {
    Assistant,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolResultMessage {
    role: ToolResultRole,
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ToolResultContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
    pub timestamp: f64,
}

impl ToolResultMessage {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<ToolResultContentBlock>,
        is_error: bool,
        timestamp: f64,
    ) -> Self {
        Self {
            role: ToolResultRole::ToolResult,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            details: None,
            usage: None,
            added_tool_names: None,
            is_error,
            timestamp,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ToolResultRole {
    ToolResult,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Message {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
}

/// An owned JSON object containing a tool's parameter schema.
///
/// Object-valued describes the schema representation, not the instance type
/// described by the schema. A top-level `type: string` or `oneOf` is valid.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct JsonSchemaObject(Map<String, Value>);

impl JsonSchemaObject {
    pub fn new(values: Map<String, Value>) -> Self {
        Self(values)
    }

    pub fn as_map(&self) -> &Map<String, Value> {
        &self.0
    }

    pub fn into_map(self) -> Map<String, Value> {
        self.0
    }
}

impl TryFrom<Value> for JsonSchemaObject {
    type Error = Value;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Object(values) => Ok(Self(values)),
            other => Err(other),
        }
    }
}

impl From<JsonSchemaObject> for Value {
    fn from(schema: JsonSchemaObject) -> Self {
        Self::Object(schema.into_map())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonSchemaStrictness {
    Prefer,
    Require,
}

/// Pi's closed grammar format map. Both keys are independently optional,
/// including the empty map at the model boundary.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarVariants {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstrainedSampling {
    Disabled,
    JsonSchema { strict: JsonSchemaStrictness },
    Grammar { variants: GrammarVariants },
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ConstrainedSamplingConfig {
    JsonSchema { strict: JsonSchemaStrictness },
    Grammar { variants: GrammarVariants },
}

impl Serialize for ConstrainedSampling {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Disabled => false.serialize(serializer),
            Self::JsonSchema { strict } => ConstrainedSamplingConfig::JsonSchema {
                strict: *strict,
            }
            .serialize(serializer),
            Self::Grammar { variants } => ConstrainedSamplingConfig::Grammar {
                variants: variants.clone(),
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ConstrainedSampling {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value == Value::Bool(false) {
            return Ok(Self::Disabled);
        }
        if value == Value::Bool(true) {
            return Err(serde::de::Error::custom(
                "constrained_sampling accepts false, never true",
            ));
        }
        match serde_json::from_value::<ConstrainedSamplingConfig>(value)
            .map_err(serde::de::Error::custom)?
        {
            ConstrainedSamplingConfig::JsonSchema { strict } => Ok(Self::JsonSchema { strict }),
            ConstrainedSamplingConfig::Grammar { variants } => Ok(Self::Grammar { variants }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchemaObject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSampling>,
}

impl ToolSchema {
    /// Canonical model-facing observation, which writes optional absence as
    /// explicit null while semantic input continues to use omission.
    pub fn as_json(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("ToolSchema serialization is infallible");
        value
            .as_object_mut()
            .expect("ToolSchema serializes as an object")
            .entry("constrained_sampling")
            .or_insert(Value::Null);
        value
    }
}

pub type ToolDefinition = ToolSchema;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LlmContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ProviderRequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, Option<String>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<BTreeMap<String, String>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    Sse,
    Websocket,
    WebsocketCached,
    Auto,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StreamOptions {
    #[serde(flatten)]
    pub provider: ProviderRequestOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<BTreeMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket_connect_timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ThinkingBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum DeferredWindow {
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DeferredOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<DeferredWindow>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum DeferredRequest {
    Enabled(bool),
    Options(DeferredOptions),
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SimpleStreamOptions {
    #[serde(flatten)]
    pub stream: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ThinkingLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<ThinkingBudgets>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LlmRequest {
    pub model: ModelIdentity,
    pub context: LlmContext,
    #[serde(default)]
    pub options: SimpleStreamOptions,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoneReason {
    Stop,
    Length,
    ToolUse,
    Deferred,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorReason {
    Error,
    Aborted,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    Done {
        reason: DoneReason,
        message: AssistantMessage,
    },
    Error {
        reason: ErrorReason,
        error: AssistantMessage,
    },
}

impl StreamChunk {
    pub fn partial(&self) -> &AssistantMessage {
        match self {
            Self::Start { partial }
            | Self::TextStart { partial, .. }
            | Self::TextDelta { partial, .. }
            | Self::TextEnd { partial, .. }
            | Self::ThinkingStart { partial, .. }
            | Self::ThinkingDelta { partial, .. }
            | Self::ThinkingEnd { partial, .. }
            | Self::ToolCallStart { partial, .. }
            | Self::ToolCallDelta { partial, .. }
            | Self::ToolCallEnd { partial, .. } => partial,
            Self::Done { message, .. } => message,
            Self::Error { error, .. } => error,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error { .. })
    }
}
