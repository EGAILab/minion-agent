#![cfg(feature = "conformance")]

use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc};

use minion_agent::{
    PluginInitError, PluginSpec, RegistrationHandle, Runtime,
    agent::{AgentDefinition, AgentInstance, ClaimPolicy},
    agent_loop::{
        AgentEvent, AgentListenerError, AgentLoop, AgentLoopError, register_agent_listener,
    },
    llm::{
        AssistantContentBlock, AssistantMessage, AssistantMessageDiagnostic, Cost, DeferredHandle,
        DiagnosticCode, DoneReason, ErrorReason, LlmService, Message, ModelIdentity, Script,
        ScriptItem, ScriptedAdapter, StopReason, StreamChunk, TextBlock, ThinkingBlock, ToolCall,
        ToolResultContentBlock, Usage, UserContent, UserMessage,
    },
    session::Session,
    tools::{
        AfterToolCallOverride, AgentToolResult, BeforeToolCallAction, ExecutionMode,
        ToolCapabilityError, ToolDefinition, ToolExecutionRequest, register_after_tool_call_hook,
        register_before_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::task::JoinHandle;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[derive(Debug, Eq, PartialEq)]
enum AgentDocumentKind {
    Primitive,
    Placeholder,
    Executable,
    Unclassified,
}

fn contains_placeholder(value: &Value) -> bool {
    match value {
        Value::String(value) => value.starts_with("TO_BE_"),
        Value::Array(values) => values.iter().any(contains_placeholder),
        Value::Object(values) => values.values().any(contains_placeholder),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

fn classify(document: &Value) -> AgentDocumentKind {
    if ["transform", "tool_registry", "agent_inbox", "llm_service"]
        .iter()
        .any(|key| document.get(key).is_some())
    {
        AgentDocumentKind::Primitive
    } else if contains_placeholder(document) {
        AgentDocumentKind::Placeholder
    } else if document.get("provider_script").is_some() && document.get("steps").is_some() {
        AgentDocumentKind::Executable
    } else {
        AgentDocumentKind::Unclassified
    }
}

fn discover() -> Vec<(PathBuf, Value)> {
    let mut documents = fs::read_dir(root().join("conformance/agent"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .map(|entry| {
            let path = entry.path();
            let document = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            (path, document)
        })
        .collect::<Vec<_>>();
    documents.sort_by(|left, right| left.0.cmp(&right.0));
    documents
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn identity(model: &str) -> ModelIdentity {
    ModelIdentity::new("mock", "mock", model).expect("canonical model identity is valid")
}

fn usage(raw: Option<&Value>) -> Usage {
    let raw = raw.and_then(Value::as_object);
    let number = |key: &str| raw.and_then(|raw| raw.get(key)).and_then(Value::as_u64);
    let cost = raw
        .and_then(|raw| raw.get("cost"))
        .and_then(Value::as_object);
    let money = |key: &str| cost.and_then(|cost| cost.get(key)).and_then(Value::as_f64);
    Usage {
        input: number("input").unwrap_or(0),
        output: number("output").unwrap_or(0),
        cache_read: number("cache_read").unwrap_or(0),
        cache_write: number("cache_write").unwrap_or(0),
        cache_write_1h: number("cache_write_1h"),
        reasoning: number("reasoning"),
        total_tokens: number("total_tokens").unwrap_or(0),
        cost: Cost {
            input: money("input").unwrap_or(0.0),
            output: money("output").unwrap_or(0.0),
            cache_read: money("cache_read").unwrap_or(0.0),
            cache_write: money("cache_write").unwrap_or(0.0),
            total: money("total").unwrap_or(0.0),
        },
    }
}

fn assistant_block(raw: &Value) -> Result<AssistantContentBlock, String> {
    match string(raw, "type")? {
        "text" => {
            let mut block = TextBlock::new(raw.get("text").and_then(Value::as_str).unwrap_or(""));
            if let Some(signature) = raw.get("text_signature").and_then(Value::as_str) {
                block = block.with_signature(signature);
            }
            Ok(AssistantContentBlock::Text(block))
        }
        "thinking" => {
            let mut block =
                ThinkingBlock::new(raw.get("thinking").and_then(Value::as_str).unwrap_or(""));
            if let Some(signature) = raw.get("thinking_signature").and_then(Value::as_str) {
                block = block.with_signature(signature);
            }
            block.redacted = raw
                .get("redacted")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Ok(AssistantContentBlock::Thinking(block))
        }
        "tool_call" => {
            let arguments = serde_json::from_value::<BTreeMap<String, Value>>(
                raw.get("arguments").cloned().unwrap_or_else(|| json!({})),
            )
            .map_err(|error| error.to_string())?;
            let mut call = ToolCall::new(string(raw, "id")?, string(raw, "name")?, arguments);
            call.thought_signature = raw
                .get("thought_signature")
                .and_then(Value::as_str)
                .map(str::to_owned);
            call.namespace = raw
                .get("namespace")
                .and_then(Value::as_str)
                .map(str::to_owned);
            Ok(AssistantContentBlock::ToolCall(call))
        }
        other => Err(format!("unsupported assistant content block {other}")),
    }
}

fn response_message(raw: &Value, timestamp: f64) -> Result<AssistantMessage, String> {
    let content = raw
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(assistant_block)
        .collect::<Result<Vec<_>, _>>()?;
    let stop_reason = match string(raw, "stop_reason")? {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "error" => StopReason::Error,
        "aborted" => StopReason::Aborted,
        "deferred" => StopReason::Deferred,
        other => return Err(format!("unsupported stop reason {other}")),
    };
    let mut message = AssistantMessage::new(
        identity("mock-1"),
        content,
        usage(raw.get("usage")),
        stop_reason,
        timestamp,
    );
    message.error_message = raw
        .get("error_message")
        .and_then(Value::as_str)
        .map(str::to_owned);
    message.response_model = raw
        .get("response_model")
        .and_then(Value::as_str)
        .map(str::to_owned);
    message.response_id = raw
        .get("response_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    message.raw_stop_reason = raw
        .get("raw_stop_reason")
        .and_then(Value::as_str)
        .map(str::to_owned);
    message.end_turn = raw.get("end_turn").and_then(Value::as_bool);
    message.diagnostics = raw
        .get("diagnostics")
        .map(|value| {
            serde_json::from_value::<Vec<AssistantMessageDiagnostic>>(value.clone())
                .map_err(|error| error.to_string())
        })
        .transpose()?;
    message.deferred = raw
        .get("deferred")
        .map(|value| {
            serde_json::from_value::<DeferredHandle>(value.clone())
                .map_err(|error| error.to_string())
        })
        .transpose()?;
    Ok(message)
}

fn scripted_response(raw: &Value, timestamp: f64) -> Result<Script, String> {
    let message = response_message(raw, timestamp)?;
    let mut partial = message.clone();
    partial.stop_reason = StopReason::Pending;
    partial.usage = Usage::default();
    let mut items = vec![ScriptItem::Chunk(Box::new(StreamChunk::Start {
        partial: AssistantMessage::pending(identity("mock-1"), timestamp),
    }))];
    let has_visible_delta = message.content.iter().any(|block| {
        matches!(
            block,
            AssistantContentBlock::Text(_) | AssistantContentBlock::Thinking(_)
        )
    });
    if has_visible_delta {
        items.push(ScriptItem::Chunk(Box::new(StreamChunk::TextDelta {
            content_index: 0,
            delta: String::new(),
            partial,
        })));
    }
    if !raw
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let terminal = match message.stop_reason {
            StopReason::Error => StreamChunk::Error {
                reason: ErrorReason::Error,
                error: message,
            },
            StopReason::Aborted => StreamChunk::Error {
                reason: ErrorReason::Aborted,
                error: message,
            },
            StopReason::Stop => StreamChunk::Done {
                reason: DoneReason::Stop,
                message,
            },
            StopReason::Length => StreamChunk::Done {
                reason: DoneReason::Length,
                message,
            },
            StopReason::ToolUse => StreamChunk::Done {
                reason: DoneReason::ToolUse,
                message,
            },
            StopReason::Deferred => StreamChunk::Done {
                reason: DoneReason::Deferred,
                message,
            },
            StopReason::Pending => return Err("provider response cannot stop pending".into()),
        };
        items.push(ScriptItem::Chunk(Box::new(terminal)));
        for _ in 0..raw
            .get("chunks_after_terminal")
            .and_then(Value::as_u64)
            .unwrap_or(0)
        {
            items.push(ScriptItem::Chunk(Box::new(StreamChunk::TextDelta {
                content_index: 0,
                delta: "ignored".into(),
                partial: AssistantMessage::pending(identity("mock-1"), timestamp),
            })));
        }
    }
    Ok(Script::new(items))
}

fn result_text(content: &[ToolResultContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ToolResultContentBlock::Text(block) => Some(block.text.as_str()),
            ToolResultContentBlock::Image(_) => None,
        })
        .collect()
}

fn message_text(message: &Message) -> String {
    match message {
        Message::User(message) => match &message.content {
            UserContent::Text(text) => text.clone(),
            UserContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    minion_agent::llm::UserContentBlock::Text(block) => Some(block.text.as_str()),
                    minion_agent::llm::UserContentBlock::Image(_) => None,
                })
                .collect(),
        },
        Message::Assistant(message) => message
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text(block) => Some(block.text.as_str()),
                AssistantContentBlock::Thinking(_) | AssistantContentBlock::ToolCall(_) => None,
            })
            .collect(),
        Message::ToolResult(message) => result_text(&message.content),
    }
}

fn partial_result(raw: &Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(
            raw.get("text").and_then(Value::as_str).unwrap_or(""),
        ))],
        details: raw.get("details").cloned().unwrap_or_else(|| json!({})),
        usage: raw.get("usage").map(|value| usage(Some(value))),
        added_tool_names: raw.get("added_tool_names").map(|names| {
            names
                .as_array()
                .expect("schema validates added_tool_names")
                .iter()
                .map(|name| name.as_str().unwrap().to_owned())
                .collect()
        }),
        terminate: raw.get("terminate").and_then(Value::as_bool),
    }
}

fn encode_partial(partial: &AgentToolResult) -> Value {
    let mut encoded = json!({
        "text": result_text(&partial.content),
        "details": partial.details,
    });
    if let Some(usage) = &partial.usage {
        encoded["usage"] = normalize_usage(usage);
    }
    if let Some(terminate) = partial.terminate {
        encoded["terminate"] = json!(terminate);
    }
    if let Some(names) = &partial.added_tool_names {
        encoded["added_tool_names"] = json!(names);
    }
    encoded
}

fn text_result(text: impl Into<String>) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(text))],
        details: Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: Some(false),
    }
}

fn scripted_tool(
    name: &str,
    raw: &Value,
    registry: minion_agent::tools::ToolRegistry,
    registrations: Arc<Mutex<Vec<RegistrationHandle>>>,
    late_updates: Arc<Mutex<Vec<JoinHandle<()>>>>,
    trace: Arc<Mutex<Vec<Value>>>,
) -> Result<ToolDefinition, String> {
    let parameters = serde_json::from_value(
        raw.get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    )
    .map_err(|error| error.to_string())?;
    let result = raw
        .get("result")
        .and_then(|result| result.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let raises = raw.get("raises").and_then(Value::as_str).map(str::to_owned);
    let delay_ticks = raw.get("delay_ticks").and_then(Value::as_u64).unwrap_or(0);
    let terminate = raw
        .get("terminate")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let added_names = raw
        .get("adds_tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|name| name.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let updates = raw
        .get("emits_updates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(partial_result)
        .collect::<Vec<_>>();
    let late = raw.get("late_update").map(partial_result);
    let mut tool = ToolDefinition::new(
        name,
        name,
        parameters,
        name,
        move |request: ToolExecutionRequest| {
            let result = result.clone();
            let raises = raises.clone();
            let added_names = added_names.clone();
            let updates = updates.clone();
            let late = late.clone();
            let registry = registry.clone();
            let registrations = Arc::clone(&registrations);
            let late_updates = Arc::clone(&late_updates);
            let trace = Arc::clone(&trace);
            Box::pin(async move {
                trace.lock().push(json!(["execute", request.tool_call_id]));
                for _ in 0..delay_ticks {
                    tokio::task::yield_now().await;
                }
                if let Some(message) = raises {
                    return Err(ToolCapabilityError::new(message));
                }
                if let Some(update) = &request.on_update {
                    for partial in updates {
                        update(partial);
                    }
                    if let Some(partial) = late {
                        let update = Arc::clone(update);
                        late_updates.lock().push(tokio::spawn(async move {
                            tokio::task::yield_now().await;
                            update(partial);
                        }));
                    }
                }
                for added_name in &added_names {
                    let added_result = added_name.clone();
                    let registration = registry
                        .register_for_scope(
                            None,
                            ToolDefinition::new(
                                added_name,
                                added_name,
                                serde_json::from_value(json!({
                                    "type": "object",
                                    "properties": {}
                                }))
                                .unwrap(),
                                added_name,
                                move |_request: ToolExecutionRequest| {
                                    let text = added_result.clone();
                                    Box::pin(async move { Ok(text_result(text)) })
                                },
                            ),
                        )
                        .map_err(|error| ToolCapabilityError::new(error.to_string()))?;
                    registrations.lock().push(registration);
                }
                let mut output = text_result(result);
                output.terminate = Some(terminate);
                output.added_tool_names = Some(added_names);
                Ok(output)
            })
        },
    );
    if raw.get("execution_mode").and_then(Value::as_str) == Some("sequential") {
        tool = tool.with_execution_mode(ExecutionMode::Sequential);
    }
    if let Some(prepare) = raw.get("prepare_arguments") {
        let raises = prepare
            .get("raises")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let replacements = prepare.get("set").cloned();
        tool = tool.with_prepare_arguments(move |mut arguments| {
            if let Some(message) = &raises {
                return Err(ToolCapabilityError::new(message));
            }
            if let (Some(arguments), Some(replacements)) = (
                arguments.as_object_mut(),
                replacements.as_ref().and_then(Value::as_object),
            ) {
                arguments.extend(replacements.clone());
            }
            Ok(arguments)
        });
    }
    Ok(tool)
}

fn normalize_usage(usage: &Usage) -> Value {
    json!({
        "input": usage.input,
        "output": usage.output,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "cache_write_1h": usage.cache_write_1h,
        "reasoning": usage.reasoning,
        "total_tokens": usage.total_tokens,
        "cost": {
            "input": canonical_float(usage.cost.input),
            "output": canonical_float(usage.cost.output),
            "cache_read": canonical_float(usage.cost.cache_read),
            "cache_write": canonical_float(usage.cost.cache_write),
            "total": canonical_float(usage.cost.total),
        }
    })
}

fn canonical_float(value: f64) -> Value {
    if value.fract() == 0.0 {
        json!(value as i64)
    } else {
        json!(value)
    }
}

#[derive(Default)]
struct Observation {
    events: Vec<String>,
    messages: Vec<Value>,
    causes: Vec<Vec<Value>>,
    agent_end_messages: Vec<Vec<String>>,
    assistant_stop_reasons: Vec<String>,
    assistant_details: Vec<Value>,
    tool_completion_order: Vec<String>,
    request_tools: Vec<Vec<String>>,
    updates: Vec<Value>,
    tool_trace: Vec<Value>,
    error: Option<Value>,
}

fn event_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart { .. } => "agent_start",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::MessageStart(_) => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd(_) => "message_end",
        AgentEvent::ToolExecutionStart(_) => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate(_) => "tool_execution_update",
        AgentEvent::ToolExecutionEnd(_) => "tool_execution_end",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::AgentEnd { .. } => "agent_end",
    }
}

fn stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Pending => "pending",
        StopReason::Stop => "stop",
        StopReason::Length => "length",
        StopReason::ToolUse => "tool_use",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
        StopReason::Deferred => "deferred",
    }
}

fn normalize_block(block: &AssistantContentBlock) -> Value {
    match block {
        AssistantContentBlock::Text(block) => json!({
            "type": "text",
            "text": block.text,
            "text_signature": block.text_signature,
        }),
        AssistantContentBlock::Thinking(block) => json!({
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }),
        AssistantContentBlock::ToolCall(call) => json!({
            "type": "tool_call",
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "thought_signature": call.thought_signature,
            "namespace": call.namespace,
        }),
    }
}

fn normalize_assistant(message: &AssistantMessage) -> Value {
    let diagnostics = message.diagnostics.as_ref().map(|diagnostics| {
        diagnostics
            .iter()
            .map(|diagnostic| {
                let error = diagnostic.error.as_ref().map(|error| {
                    let code = error.code.as_ref().map(|code| match code {
                        DiagnosticCode::String(value) => json!(value),
                        DiagnosticCode::Number(value) => canonical_float(*value),
                    });
                    json!({
                        "message": error.message,
                        "name": error.name,
                        "stack": error.stack,
                        "code": code,
                    })
                });
                json!({
                    "type": diagnostic.diagnostic_type,
                    "timestamp": canonical_float(diagnostic.timestamp),
                    "error": error,
                    "details": diagnostic.details,
                })
            })
            .collect::<Vec<_>>()
    });
    json!({
        "api": message.api,
        "provider": message.provider,
        "model": message.model,
        "timestamp": canonical_float(message.timestamp),
        "response_model": message.response_model,
        "response_id": message.response_id,
        "stop_reason": stop_reason(message.stop_reason),
        "raw_stop_reason": message.raw_stop_reason,
        "end_turn": message.end_turn,
        "error_message": message.error_message,
        "usage": normalize_usage(&message.usage),
        "diagnostics": diagnostics,
        "deferred": message.deferred,
        "content": message.content.iter().map(normalize_block).collect::<Vec<_>>(),
    })
}

fn normalize_message(message: &Message) -> Value {
    match message {
        Message::User(_) => json!({"role": "user", "text": message_text(message)}),
        Message::Assistant(_) => json!({"role": "assistant", "text": message_text(message)}),
        Message::ToolResult(message) => {
            let mut value = json!({"role": "tool_result", "text": result_text(&message.content)});
            if let Some(details) = &message.details {
                value["details"] = details.clone();
            }
            value
        }
    }
}

fn policy(raw: Option<&Value>) -> ClaimPolicy {
    match raw.and_then(Value::as_str) {
        Some("all") => ClaimPolicy::All,
        Some("one-at-a-time") | None => ClaimPolicy::OneAtATime,
        Some(other) => panic!("schema admitted unsupported claim policy {other}"),
    }
}

fn input(raw: &Value) -> (Message, Option<Value>) {
    let (text, origin) = match raw {
        Value::String(text) => (text.clone(), None),
        Value::Object(raw) => (
            raw.get("text").and_then(Value::as_str).unwrap().to_owned(),
            raw.get("origin").cloned(),
        ),
        _ => panic!("schema admitted unsupported input"),
    };
    (
        Message::User(UserMessage::new(UserContent::Text(text), 1.0)),
        origin,
    )
}

fn observe_event(
    event: AgentEvent,
    events: &Arc<Mutex<Vec<String>>>,
    causes: &Arc<Mutex<Vec<Vec<Value>>>>,
    agent_end_messages: &Arc<Mutex<Vec<Vec<String>>>>,
    completion: &Arc<Mutex<Vec<String>>>,
    updates: &Arc<Mutex<Vec<Value>>>,
    trace: &Arc<Mutex<Vec<Value>>>,
) {
    events.lock().push(event_name(&event).to_owned());
    match event {
        AgentEvent::AgentEnd {
            causes: run_causes,
            messages,
            ..
        } => {
            causes.lock().push(
                run_causes
                    .into_iter()
                    .map(|cause| json!({"id": cause.id, "origin": cause.origin}))
                    .collect(),
            );
            agent_end_messages
                .lock()
                .push(messages.iter().map(message_text).collect());
        }
        AgentEvent::ToolExecutionStart(event) => {
            trace.lock().push(json!(["start", event.tool_call_id]))
        }
        AgentEvent::ToolExecutionUpdate(event) => updates.lock().push(json!({
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "arguments": event.arguments,
            "partial": encode_partial(&event.update),
        })),
        AgentEvent::ToolExecutionEnd(event) => {
            completion.lock().push(event.tool_call_id.clone());
            trace.lock().push(json!(["end", event.tool_call_id]));
        }
        AgentEvent::TurnStart
        | AgentEvent::MessageStart(_)
        | AgentEvent::MessageUpdate { .. }
        | AgentEvent::MessageEnd(_)
        | AgentEvent::TurnEnd { .. } => {}
        AgentEvent::AgentStart { .. } => {}
    }
}

fn install_tool_listener(
    context: &minion_agent::Context,
    spec: &Value,
    trace: Arc<Mutex<Vec<Value>>>,
    trace_enabled: bool,
) -> Result<(), String> {
    let event = string(spec, "event")?;
    let action = string(spec, "action")?.to_owned();
    let only = spec
        .get("only_tool")
        .and_then(Value::as_str)
        .map(str::to_owned);
    match event {
        "tools/pre-execute" => {
            let spec = spec.clone();
            register_before_tool_call_hook(context, move |call| {
                let spec = spec.clone();
                let action = action.clone();
                let only = only.clone();
                let trace = Arc::clone(&trace);
                async move {
                    if only.as_deref().is_some_and(|only| only != call.tool_name) {
                        return Ok(BeforeToolCallAction::Proceed(None));
                    }
                    let terminal = matches!(action.as_str(), "block" | "raise");
                    if trace_enabled && terminal {
                        trace.lock().push(json!(["before", call.tool_call_id]));
                    }
                    match action.as_str() {
                        "block" => Ok(BeforeToolCallAction::Block {
                            reason: spec
                                .get("reason")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            terminate: spec
                                .get("terminate")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                        }),
                        "narrow_arguments" => Ok(BeforeToolCallAction::Proceed(Some(
                            spec.get("arguments").cloned().unwrap_or_else(|| json!({})),
                        ))),
                        "abstain" => Ok(BeforeToolCallAction::Proceed(None)),
                        "raise" => Err(ToolCapabilityError::new(
                            spec.get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("before-hook failed"),
                        )),
                        other => Err(ToolCapabilityError::new(format!(
                            "unsupported pre-execute action {other}"
                        ))),
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        }
        "tools/post-execute" => {
            let spec = spec.clone();
            register_after_tool_call_hook(context, move |result| {
                let spec = spec.clone();
                let action = action.clone();
                let only = only.clone();
                async move {
                    if only.as_deref().is_some_and(|only| only != result.tool_name) {
                        return Ok(None);
                    }
                    match action.as_str() {
                        "annotate_result" => {
                            let label = spec.get("label").and_then(Value::as_str).unwrap_or("seen");
                            Ok(Some(AfterToolCallOverride::default().with_content(vec![
                                ToolResultContentBlock::Text(TextBlock::new(format!(
                                    "{}-{label}",
                                    result_text(&result.content)
                                ))),
                            ])))
                        }
                        "abstain" => Ok(None),
                        "raise" => Err(ToolCapabilityError::new(
                            spec.get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("after-hook failed"),
                        )),
                        other => Err(ToolCapabilityError::new(format!(
                            "unsupported post-execute action {other}"
                        ))),
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        }
        other => return Err(format!("unsupported listener event {other}")),
    }
    Ok(())
}

fn compare_observation(document: &Value, actual: &Observation) -> Result<(), String> {
    if let Some(expected) = document.get("expect_events") {
        let actual = json!(actual.events);
        if &actual != expected {
            return Err(format!("events\nexpected: {expected}\nactual:   {actual}"));
        }
    }
    if let Some(expected) = document.get("expect_messages").and_then(Value::as_array) {
        if actual.messages.len() != expected.len() {
            return Err(format!(
                "message count expected {} actual {}: {}",
                expected.len(),
                actual.messages.len(),
                json!(actual.messages)
            ));
        }
        for (index, (actual, expected)) in actual.messages.iter().zip(expected).enumerate() {
            if actual.get("role") != expected.get("role") {
                return Err(format!(
                    "message {index} role: expected {expected}, actual {actual}"
                ));
            }
            if let Some(fragment) = expected.get("text_contains").and_then(Value::as_str) {
                if !actual
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains(fragment))
                {
                    return Err(format!("message {index} missing {fragment:?}: {actual}"));
                }
            } else if actual.get("text") != expected.get("text") {
                return Err(format!(
                    "message {index} text: expected {expected}, actual {actual}"
                ));
            }
            if expected.get("details").is_some() && actual.get("details") != expected.get("details")
            {
                return Err(format!(
                    "message {index} details: expected {expected}, actual {actual}"
                ));
            }
        }
    }
    let exact = [
        (
            "expect_agent_end_messages",
            json!(actual.agent_end_messages),
        ),
        (
            "expect_assistant_stop_reasons",
            json!(actual.assistant_stop_reasons),
        ),
        ("expect_assistant_details", json!(actual.assistant_details)),
        (
            "expect_tool_completion_order",
            json!(actual.tool_completion_order),
        ),
        ("expect_request_tools", json!(actual.request_tools)),
        ("expect_updates", json!(actual.updates)),
        ("expect_tool_trace", json!(actual.tool_trace)),
    ];
    for (key, actual) in exact {
        if let Some(expected) = document.get(key)
            && &actual != expected
        {
            return Err(format!("{key}\nexpected: {expected}\nactual:   {actual}"));
        }
    }
    if let Some(expected) = document.get("expect_causes") {
        let origins = actual
            .causes
            .iter()
            .map(|causes| {
                causes
                    .iter()
                    .map(|cause| cause.get("origin").cloned().unwrap_or(Value::Null))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if json!(origins) != *expected {
            return Err(format!(
                "expect_causes\nexpected: {expected}\nactual:   {}",
                json!(origins)
            ));
        }
    }
    match document.get("expect_error") {
        None if actual.error.is_some() => {
            return Err(format!(
                "unexpected error: {}",
                actual.error.as_ref().unwrap()
            ));
        }
        Some(expected) => {
            let Some(actual) = actual.error.as_ref() else {
                return Err("expected error but scenario completed".into());
            };
            if actual.get("type") != expected.get("type") {
                return Err(format!("error type expected {expected}, actual {actual}"));
            }
            if let Some(fragment) = expected.get("message_contains").and_then(Value::as_str)
                && !actual
                    .get("message")
                    .and_then(Value::as_str)
                    .is_some_and(|message| message.contains(fragment))
            {
                return Err(format!("error missing {fragment:?}: {actual}"));
            }
        }
        None => {}
    }
    Ok(())
}

fn run_scenario_through_real_agent_loop(document: &Value) -> Result<(), String> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .map_err(|error| error.to_string())?
        .block_on(run_scenario(document))
}

async fn run_scenario(document: &Value) -> Result<(), String> {
    let runtime = Runtime::new();
    let context = runtime.context();
    let config = document.get("config").unwrap_or(&Value::Null);
    let configured_model = config
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("mock-1");

    let scripts = document
        .get("provider_script")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, response)| scripted_response(response, (index + 1) as f64))
        .collect::<Result<Vec<_>, _>>()?;
    let adapter = Arc::new(ScriptedAdapter::new(scripts));
    let llm = Arc::new(LlmService::new());
    llm.register(identity("mock-1"), adapter.clone());

    let registrations = Arc::new(Mutex::new(Vec::<RegistrationHandle>::new()));
    let late_updates = Arc::new(Mutex::new(Vec::<JoinHandle<()>>::new()));
    let trace = Arc::new(Mutex::new(Vec::<Value>::new()));
    if let Some(tools) = document.get("tools").and_then(Value::as_object) {
        for (name, raw) in tools {
            let tool = scripted_tool(
                name,
                raw,
                runtime.tools().clone(),
                Arc::clone(&registrations),
                Arc::clone(&late_updates),
                Arc::clone(&trace),
            )?;
            let registration = runtime
                .tools()
                .register_for_scope(None, tool)
                .map_err(|error| error.to_string())?;
            registrations.lock().push(registration);
        }
    }

    let seen_events = Arc::new(Mutex::new(Vec::new()));
    let seen_causes = Arc::new(Mutex::new(Vec::new()));
    let seen_agent_end_messages = Arc::new(Mutex::new(Vec::new()));
    let seen_completion = Arc::new(Mutex::new(Vec::new()));
    let seen_updates = Arc::new(Mutex::new(Vec::new()));
    let listeners = document
        .get("listeners")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let trace_enabled = document.get("expect_tool_trace").is_some();
    let listener_plugin =
        PluginSpec::<Value>::new("agent-loop-conformance-observer", vec![], || json!({}), {
            let events = Arc::clone(&seen_events);
            let causes = Arc::clone(&seen_causes);
            let agent_end_messages = Arc::clone(&seen_agent_end_messages);
            let completion = Arc::clone(&seen_completion);
            let updates = Arc::clone(&seen_updates);
            let trace = Arc::clone(&trace);
            move |plugin_context, _config| {
                let events = Arc::clone(&events);
                let causes = Arc::clone(&causes);
                let agent_end_messages = Arc::clone(&agent_end_messages);
                let completion = Arc::clone(&completion);
                let updates = Arc::clone(&updates);
                let trace = Arc::clone(&trace);
                let listeners = listeners.clone();
                async move {
                    for listener in &listeners {
                        install_tool_listener(
                            &plugin_context,
                            listener,
                            Arc::clone(&trace),
                            trace_enabled,
                        )
                        .map_err(PluginInitError::new)?;
                    }
                    if trace_enabled {
                        let trace_before = Arc::clone(&trace);
                        register_before_tool_call_hook(&plugin_context, move |call| {
                            let trace = Arc::clone(&trace_before);
                            async move {
                                trace.lock().push(json!(["before", call.tool_call_id]));
                                Ok(BeforeToolCallAction::Proceed(None))
                            }
                        })
                        .map_err(|error| PluginInitError::new(error.to_string()))?;
                    }
                    register_agent_listener(&plugin_context, move |event| {
                        let events = Arc::clone(&events);
                        let causes = Arc::clone(&causes);
                        let agent_end_messages = Arc::clone(&agent_end_messages);
                        let completion = Arc::clone(&completion);
                        let updates = Arc::clone(&updates);
                        let trace = Arc::clone(&trace);
                        async move {
                            observe_event(
                                event,
                                &events,
                                &causes,
                                &agent_end_messages,
                                &completion,
                                &updates,
                                &trace,
                            );
                            Ok::<(), AgentListenerError>(())
                        }
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                    Ok(())
                }
            }
        })
        .erase();
    runtime
        .mount(&listener_plugin, json!({}))
        .map_err(|error| error.to_string())?;
    runtime
        .reconcile()
        .await
        .map_err(|error| error.to_string())?;
    let session = Session::new("scenario", [] as [&str; 0]).map_err(|error| error.to_string())?;
    let agent = Arc::new(AgentInstance::new(
        "scenario",
        AgentDefinition::new(
            "scenario",
            config.get("system").and_then(Value::as_str).unwrap_or(""),
            identity(configured_model),
        ),
        session,
        Some(context.clone()),
        None,
    ));
    let mut driver = AgentLoop::new(Arc::clone(&agent), context, llm);
    driver.set_next_turn_policy(policy(config.get("next_turn_policy")));
    driver.set_next_step_policy(policy(config.get("next_step_policy")));

    let mut run_error: Option<AgentLoopError> = None;
    for step in document.get("steps").and_then(Value::as_array).unwrap() {
        if let Some(raw) = step.get("followup") {
            let (message, origin) = input(raw);
            agent.follow_up(message, origin);
        } else if let Some(raw) = step.get("steer") {
            let (message, origin) = input(raw);
            agent.steer(message, origin);
        } else if let Some(raw) = step.get("inject") {
            let (message, origin) = input(raw);
            agent.inject(message, origin);
        } else if step.get("await_idle").and_then(Value::as_bool) == Some(true) {
            if let Err(error) = driver.run_until_idle().await {
                run_error = Some(error);
                break;
            }
        } else if step.get("continue").and_then(Value::as_bool) == Some(true) {
            if let Err(error) = driver.continue_run().await {
                run_error = Some(error);
                break;
            }
        } else if step.get("abort").is_some() {
            return Err("an executable Layer-09 abort step reached the Layer-08 adapter".into());
        } else {
            return Err(format!("schema admitted unsupported step {step}"));
        }
    }

    let handles = std::mem::take(&mut *late_updates.lock());
    for handle in handles {
        handle.await.map_err(|error| error.to_string())?;
    }

    let messages = agent.messages().map_err(|error| error.to_string())?;
    let assistant_messages = messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(message) => Some(message.as_ref()),
            Message::User(_) | Message::ToolResult(_) => None,
        })
        .collect::<Vec<_>>();
    let request_tools = adapter
        .requests()
        .iter()
        .map(|request| {
            request
                .context
                .tools
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    let error = run_error.map(|error| {
        let error_type = if matches!(
            error,
            AgentLoopError::LlmStart(minion_agent::llm::LlmStartError::UnknownModel { .. })
        ) {
            "UnknownModelError"
        } else {
            "AgentLoopError"
        };
        json!({"type": error_type, "message": error.to_string()})
    });
    let observation = Observation {
        events: seen_events.lock().clone(),
        messages: messages.iter().map(normalize_message).collect(),
        causes: seen_causes.lock().clone(),
        agent_end_messages: seen_agent_end_messages.lock().clone(),
        assistant_stop_reasons: assistant_messages
            .iter()
            .map(|message| stop_reason(message.stop_reason).to_owned())
            .collect(),
        assistant_details: assistant_messages
            .iter()
            .map(|message| normalize_assistant(message))
            .collect(),
        tool_completion_order: seen_completion.lock().clone(),
        request_tools,
        updates: seen_updates.lock().clone(),
        tool_trace: trace.lock().clone(),
        error,
    };
    compare_observation(document, &observation)
}

#[test]
fn discovery_classifies_every_agent_document_by_semantic_shape() {
    let documents = discover();
    let primitives = documents
        .iter()
        .filter(|(_, document)| classify(document) == AgentDocumentKind::Primitive)
        .count();
    let placeholders = documents
        .iter()
        .filter(|(_, document)| classify(document) == AgentDocumentKind::Placeholder)
        .count();
    let executable = documents
        .iter()
        .filter(|(_, document)| classify(document) == AgentDocumentKind::Executable)
        .count();
    let unclassified = documents
        .iter()
        .filter(|(_, document)| classify(document) == AgentDocumentKind::Unclassified)
        .map(|(path, _)| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    assert!(
        unclassified.is_empty(),
        "unclassified Agent documents: {unclassified:?}"
    );
    assert_eq!(documents.len(), primitives + placeholders + executable);
    assert!(primitives > 0);
    assert!(placeholders > 0);
    assert!(executable > 0);
}

#[test]
fn all_layer_08_scenarios_drive_the_real_rust_agent_loop() {
    let scenarios = discover()
        .into_iter()
        .filter(|(_, document)| classify(document) == AgentDocumentKind::Executable)
        .collect::<Vec<_>>();
    assert!(!scenarios.is_empty());

    for (path, document) in scenarios {
        run_scenario_through_real_agent_loop(&document)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }
}
