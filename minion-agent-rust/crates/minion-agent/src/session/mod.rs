//! Append-only sessions, deterministic surface derivation, and immutable artifacts.

use crate::llm::{ArtifactHash, Message, ToolDefinition};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};
use thiserror::Error;

const USER: &str = "user/message";
const ASSISTANT: &str = "assistant/message";
const TOOL_RESULT: &str = "tool/result";
const RESET: &str = "session/reset";
const FORKED: &str = "session/forked";
const COMPACTION: &str = "session/compaction";
const REQUEST_HEADER: &str = "request/header";

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct EventKind(String);

impl EventKind {
    pub fn new(value: impl Into<String>) -> Result<Self, SessionError> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.split('/').all(|part| {
                let mut chars = part.chars();
                chars.next().is_some_and(|c| c.is_ascii_lowercase())
                    && chars.all(|c| {
                        c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'
                    })
            });
        valid
            .then_some(Self(value))
            .ok_or(SessionError::InvalidEventKind)
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn user_message() -> Self {
        Self(USER.into())
    }
    pub fn assistant_message() -> Self {
        Self(ASSISTANT.into())
    }
    pub fn tool_result() -> Self {
        Self(TOOL_RESULT.into())
    }
}

impl<'de> Deserialize<'de> for EventKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionEvent {
    pub seq: u64,
    pub kind: EventKind,
    pub data: Map<String, Value>,
}

#[derive(Clone)]
pub struct Session {
    inner: Arc<SessionInner>,
}

struct SessionInner {
    id: String,
    events: Mutex<Vec<SessionEvent>>,
    ancestor: Option<Session>,
    boundary: u64,
    surface: HashSet<EventKind>,
    artifacts: Arc<ArtifactStore>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum SessionError {
    #[error("invalid session event kind")]
    InvalidEventKind,
    #[error("event data must be a JSON object")]
    InvalidEventData,
    #[error("message serialization failed: {0}")]
    Message(String),
    #[error("artifact is missing: {0}")]
    MissingArtifact(String),
    #[error("request header is malformed")]
    InvalidHeader,
    #[error("fork boundary {boundary} is beyond committed tip {tip}")]
    InvalidForkBoundary { boundary: u64, tip: u64 },
}

impl Session {
    pub fn new<I, S>(id: impl Into<String>, extra_surface: I) -> Result<Self, SessionError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::with_artifacts(id, extra_surface, Arc::new(ArtifactStore::new()))
    }

    pub fn with_artifacts<I, S>(
        id: impl Into<String>,
        extra_surface: I,
        artifacts: Arc<ArtifactStore>,
    ) -> Result<Self, SessionError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut surface = [USER, ASSISTANT, TOOL_RESULT]
            .into_iter()
            .map(EventKind::new)
            .collect::<Result<HashSet<_>, _>>()?;
        for kind in extra_surface {
            surface.insert(EventKind::new(kind)?);
        }
        Ok(Self {
            inner: Arc::new(SessionInner {
                id: id.into(),
                events: Mutex::new(Vec::new()),
                ancestor: None,
                boundary: 0,
                surface,
                artifacts,
            }),
        })
    }

    pub fn id(&self) -> &str {
        &self.inner.id
    }
    pub fn events(&self) -> Vec<SessionEvent> {
        self.inner.events.lock().clone()
    }
    pub fn artifact_count(&self) -> usize {
        self.inner.artifacts.len()
    }

    pub fn append(&self, kind: EventKind, data: Map<String, Value>) -> SessionEvent {
        let mut events = self.inner.events.lock();
        Self::append_locked(&mut events, kind, data)
    }

    pub fn append_raw(
        &self,
        kind: impl Into<String>,
        data: Value,
    ) -> Result<SessionEvent, SessionError> {
        let kind = EventKind::new(kind)?;
        let Value::Object(data) = data else {
            return Err(SessionError::InvalidEventData);
        };
        Ok(self.append(kind, data))
    }

    fn append_locked(
        events: &mut Vec<SessionEvent>,
        kind: EventKind,
        data: Map<String, Value>,
    ) -> SessionEvent {
        let event = SessionEvent {
            seq: events.len() as u64 + 1,
            kind,
            data,
        };
        events.push(event.clone());
        event
    }

    pub fn append_message(&self, message: Message) -> Result<SessionEvent, SessionError> {
        let kind = match &message {
            Message::User(_) => EventKind::user_message(),
            Message::Assistant(_) => EventKind::assistant_message(),
            Message::ToolResult(_) => EventKind::tool_result(),
        };
        self.append_projectable(kind, message)
    }

    pub fn append_projectable(
        &self,
        kind: EventKind,
        message: Message,
    ) -> Result<SessionEvent, SessionError> {
        let message =
            serde_json::to_value(message).map_err(|e| SessionError::Message(e.to_string()))?;
        let Value::Object(data) = serde_json::json!({"message": message}) else {
            unreachable!()
        };
        Ok(self.append(kind, data))
    }

    pub fn reset(&self) -> Result<SessionEvent, SessionError> {
        self.append_raw(RESET, serde_json::json!({}))
    }

    pub fn compact(
        &self,
        summary: impl Into<String>,
        keep: usize,
    ) -> Result<SessionEvent, SessionError> {
        let mut events = self.inner.events.lock();
        let floor = events
            .iter()
            .rev()
            .find(|event| event.kind.as_str() == RESET)
            .map_or(0, |event| event.seq);
        let surface = events
            .iter()
            .filter(|event| event.seq > floor && self.inner.surface.contains(&event.kind))
            .collect::<Vec<_>>();
        let retained = surface
            .iter()
            .rev()
            .take(keep)
            .map(|event| event.seq)
            .collect::<Vec<_>>();
        let through = surface.last().map_or(0, |event| event.seq);
        let Value::Object(data) = serde_json::json!({"summary": summary.into(), "superseded_through": through, "retained": retained.into_iter().rev().collect::<Vec<_>>() })
        else {
            unreachable!()
        };
        Ok(Self::append_locked(
            &mut events,
            EventKind::new(COMPACTION)?,
            data,
        ))
    }

    pub fn fork(&self, id: impl Into<String>, at: Option<u64>) -> Result<Self, SessionError> {
        let tip = self.inner.events.lock().len() as u64;
        let boundary = at.unwrap_or(tip);
        if boundary > tip {
            return Err(SessionError::InvalidForkBoundary { boundary, tip });
        }
        let child = Self {
            inner: Arc::new(SessionInner {
                id: id.into(),
                events: Mutex::new(Vec::new()),
                ancestor: Some(self.clone()),
                boundary,
                surface: self.inner.surface.clone(),
                artifacts: self.inner.artifacts.clone(),
            }),
        };
        child.append_raw(
            FORKED,
            serde_json::json!({"source": self.id(), "boundary": boundary}),
        )?;
        Ok(child)
    }

    pub fn derive_messages(&self) -> Result<Vec<Message>, SessionError> {
        self.derive_until(u64::MAX)
    }

    fn derive_until(&self, limit: u64) -> Result<Vec<Message>, SessionError> {
        let events = self.events();
        let reset = events
            .iter()
            .rev()
            .find(|e| e.seq <= limit && e.kind.as_str() == RESET);
        let floor = reset.map_or(0, |e| e.seq);
        let surface = events
            .iter()
            .filter(|e| e.seq <= limit && e.seq > floor && self.inner.surface.contains(&e.kind))
            .cloned()
            .collect::<Vec<_>>();
        if let Some(compaction) = events
            .iter()
            .rev()
            .find(|e| e.seq <= limit && e.seq > floor && e.kind.as_str() == COMPACTION)
        {
            let through = compaction
                .data
                .get("superseded_through")
                .and_then(Value::as_u64)
                .ok_or(SessionError::InvalidEventData)?;
            let retained_values = compaction
                .data
                .get("retained")
                .and_then(Value::as_array)
                .ok_or(SessionError::InvalidEventData)?;
            let retained = retained_values
                .iter()
                .map(Value::as_u64)
                .collect::<Option<HashSet<_>>>()
                .ok_or(SessionError::InvalidEventData)?;
            let summary = compaction
                .data
                .get("summary")
                .and_then(Value::as_str)
                .ok_or(SessionError::InvalidEventData)?;
            let mut result = vec![Message::User(crate::llm::UserMessage::new(
                crate::llm::UserContent::Text(summary.into()),
                0.0,
            ))];
            result.extend(
                self.decode_events(
                    surface
                        .into_iter()
                        .filter(|e| e.seq > through || retained.contains(&e.seq)),
                )?,
            );
            return Ok(result);
        }
        let mut result = if reset.is_none() {
            self.inner
                .ancestor
                .as_ref()
                .map_or(Ok(Vec::new()), |a| a.derive_until(self.inner.boundary))?
        } else {
            Vec::new()
        };
        result.extend(self.decode_events(surface)?);
        Ok(result)
    }

    fn decode_events<I>(&self, events: I) -> Result<Vec<Message>, SessionError>
    where
        I: IntoIterator<Item = SessionEvent>,
    {
        events
            .into_iter()
            .map(|event| {
                let value = event
                    .data
                    .get("message")
                    .cloned()
                    .ok_or(SessionError::InvalidEventData)?;
                serde_json::from_value(value).map_err(|e| SessionError::Message(e.to_string()))
            })
            .collect()
    }

    pub fn record_header(
        &self,
        components: BTreeMap<String, String>,
        model: impl Into<String>,
        tools: Vec<ToolDefinition>,
    ) -> Result<SessionEvent, SessionError> {
        let references = components
            .into_iter()
            .map(|(name, value)| (name, self.inner.artifacts.put(value.as_bytes())))
            .collect::<BTreeMap<_, _>>();
        let tools_json =
            serde_json::to_vec(&tools).map_err(|e| SessionError::Message(e.to_string()))?;
        let tools_ref = self.inner.artifacts.put(&tools_json);
        self.append_raw(REQUEST_HEADER, serde_json::json!({"model": model.into(), "components": references, "tools": tools_ref}))
    }

    pub fn reconstruct_header(
        &self,
        event: &SessionEvent,
    ) -> Result<ReconstructedHeader, SessionError> {
        if event.kind.as_str() != REQUEST_HEADER {
            return Err(SessionError::InvalidHeader);
        }
        let references = event
            .data
            .get("components")
            .and_then(Value::as_object)
            .ok_or(SessionError::InvalidHeader)?;
        let mut components = BTreeMap::new();
        for (name, reference) in references {
            let reference = reference.as_str().ok_or(SessionError::InvalidHeader)?;
            let raw = self.inner.artifacts.get(reference)?;
            components.insert(
                name.clone(),
                String::from_utf8(raw).map_err(|_| SessionError::InvalidHeader)?,
            );
        }
        let tools_ref = event
            .data
            .get("tools")
            .and_then(Value::as_str)
            .ok_or(SessionError::InvalidHeader)?;
        let tools = serde_json::from_slice(&self.inner.artifacts.get(tools_ref)?)
            .map_err(|e| SessionError::Message(e.to_string()))?;
        let model = event
            .data
            .get("model")
            .and_then(Value::as_str)
            .ok_or(SessionError::InvalidHeader)?
            .to_owned();
        Ok(ReconstructedHeader {
            model,
            assembled_system: assemble_system(&components),
            components,
            tools,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReconstructedHeader {
    pub model: String,
    pub components: BTreeMap<String, String>,
    pub tools: Vec<ToolDefinition>,
    pub assembled_system: String,
}

pub fn assemble_system(components: &BTreeMap<String, String>) -> String {
    components
        .values()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[derive(Default)]
pub struct ArtifactStore {
    content: Mutex<HashMap<String, Vec<u8>>>,
}

impl ArtifactStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn put(&self, content: &[u8]) -> ArtifactHash {
        let hash = format!("sha256:{:x}", Sha256::digest(content));
        self.content
            .lock()
            .entry(hash.clone())
            .or_insert_with(|| content.to_vec());
        ArtifactHash::new(hash).expect("SHA-256 output always satisfies ArtifactHash")
    }
    pub fn get(&self, reference: &str) -> Result<Vec<u8>, SessionError> {
        self.content
            .lock()
            .get(reference)
            .cloned()
            .ok_or_else(|| SessionError::MissingArtifact(reference.into()))
    }
    pub fn len(&self) -> usize {
        self.content.lock().len()
    }
    pub fn is_empty(&self) -> bool {
        self.content.lock().is_empty()
    }
}
