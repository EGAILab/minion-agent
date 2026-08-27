use std::{collections::VecDeque, sync::Arc};

use parking_lot::Mutex;
use serde_json::Value;

use crate::llm::Message;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InboxTarget {
    Steering,
    FollowUp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimPolicy {
    All,
    OneAtATime,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InputEnvelope {
    pub target: InboxTarget,
    pub message: Message,
    pub origin: Option<Value>,
}

#[derive(Default)]
struct InboxState {
    steering: VecDeque<InputEnvelope>,
    follow_up: VecDeque<InputEnvelope>,
    wake_requested: bool,
}

#[derive(Clone, Default)]
pub struct Inbox {
    state: Arc<Mutex<InboxState>>,
}

impl Inbox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn steer(&self, message: Message, origin: Option<Value>) -> InputEnvelope {
        self.enqueue(InboxTarget::Steering, message, origin, true)
    }

    pub fn follow_up(&self, message: Message, origin: Option<Value>) -> InputEnvelope {
        self.enqueue(InboxTarget::FollowUp, message, origin, true)
    }

    pub fn inject(&self, message: Message, origin: Option<Value>) -> InputEnvelope {
        self.enqueue(InboxTarget::Steering, message, origin, false)
    }

    fn enqueue(
        &self,
        target: InboxTarget,
        message: Message,
        origin: Option<Value>,
        request_wake: bool,
    ) -> InputEnvelope {
        let envelope = InputEnvelope {
            target,
            message,
            origin,
        };
        let mut state = self.state.lock();
        match target {
            InboxTarget::Steering => state.steering.push_back(envelope.clone()),
            InboxTarget::FollowUp => state.follow_up.push_back(envelope.clone()),
        }
        state.wake_requested |= request_wake;
        envelope
    }

    pub fn claim(&self, target: InboxTarget, policy: ClaimPolicy) -> Vec<InputEnvelope> {
        let mut state = self.state.lock();
        let queue = match target {
            InboxTarget::Steering => &mut state.steering,
            InboxTarget::FollowUp => &mut state.follow_up,
        };
        match policy {
            ClaimPolicy::All => queue.drain(..).collect(),
            ClaimPolicy::OneAtATime => queue.pop_front().into_iter().collect(),
        }
    }

    pub fn has_pending(&self) -> bool {
        let state = self.state.lock();
        !state.steering.is_empty() || !state.follow_up.is_empty()
    }

    pub fn clear(&self, target: InboxTarget) {
        let mut state = self.state.lock();
        match target {
            InboxTarget::Steering => state.steering.clear(),
            InboxTarget::FollowUp => state.follow_up.clear(),
        }
    }

    pub fn clear_all(&self) {
        let mut state = self.state.lock();
        state.steering.clear();
        state.follow_up.clear();
    }

    pub fn wake_requested(&self) -> bool {
        self.state.lock().wake_requested
    }

    pub fn take_wake(&self) -> bool {
        let mut state = self.state.lock();
        let requested = state.wake_requested;
        state.wake_requested = false;
        requested
    }
}
