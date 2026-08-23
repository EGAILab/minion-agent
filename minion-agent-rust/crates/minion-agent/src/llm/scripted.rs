use std::collections::VecDeque;

use futures::stream;
use parking_lot::Mutex;

use super::{
    AdapterStartError, AdapterStreamError, LlmAdapter, LlmRequest, RawAssistantStream, StreamChunk,
};

#[derive(Clone, Debug)]
pub enum ScriptItem {
    Chunk(StreamChunk),
    Error(AdapterStreamError),
}

#[derive(Clone, Debug)]
pub struct Script {
    items: Vec<ScriptItem>,
}

impl Script {
    pub fn new(items: impl IntoIterator<Item = ScriptItem>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }
}

#[derive(Debug)]
pub struct ScriptedAdapter {
    scripts: Mutex<VecDeque<Script>>,
    requests: Mutex<Vec<LlmRequest>>,
}

impl ScriptedAdapter {
    pub fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn requests(&self) -> Vec<LlmRequest> {
        self.requests.lock().clone()
    }
}

impl LlmAdapter for ScriptedAdapter {
    fn start(&self, request: LlmRequest) -> Result<RawAssistantStream, AdapterStartError> {
        self.requests.lock().push(request);
        let script = self.scripts.lock().pop_front().ok_or_else(|| {
            AdapterStartError::Rejected("scripted adapter has no remaining script".into())
        })?;
        Ok(Box::pin(stream::iter(script.items.into_iter().map(
            |item| match item {
                ScriptItem::Chunk(chunk) => Ok(chunk),
                ScriptItem::Error(error) => Err(error),
            },
        ))))
    }
}
