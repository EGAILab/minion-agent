use serde::{Deserialize, Serialize};

use crate::llm::{AssistantMessage, Message, ToolResultMessage};

use super::RunContext;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Reject {
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Enter {
    pub messages: Vec<Message>,
    pub system_override: Option<String>,
    pub history_window: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreStepDecision {
    Reject(Reject),
    Enter(Enter),
}

/// Why an input batch is being considered for admission.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreStepReason {
    Initial,
    ToolResults,
    Steering,
    NextTurn,
    Continuation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreStepContext {
    pub messages: Vec<Message>,
    pub reason: PreStepReason,
}

/// A listener's ordered opinion on whether the run should continue.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStopping {
    #[default]
    NoOpinion,
    Continue,
    Stop,
}

/// Owned snapshot delivered to `prepare-next-turn` listeners.
#[derive(Clone, Debug)]
pub struct PrepareNextTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: RunContext,
    pub new_messages: Vec<Message>,
}

/// Owned snapshot delivered to `should-stop-after-turn` listeners.
#[derive(Clone, Debug)]
pub struct ShouldStopAfterTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: RunContext,
    pub new_messages: Vec<Message>,
}
