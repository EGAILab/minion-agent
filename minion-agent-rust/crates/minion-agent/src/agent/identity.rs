use serde::{Deserialize, Serialize};

use crate::llm::ModelIdentity;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentStatus {
    #[default]
    Idle,
    Running,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentDefinition {
    pub name: String,
    pub system_prompt: String,
    pub model: ModelIdentity,
}

impl AgentDefinition {
    pub fn new(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        model: ModelIdentity,
    ) -> Self {
        Self {
            name: name.into(),
            system_prompt: system_prompt.into(),
            model,
        }
    }
}
