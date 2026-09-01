mod context;
mod decisions;
mod driver;
mod error;
mod events;
mod state;

pub use context::{RunConfig, RunConfigUpdate, RunContext};
pub use decisions::{
    Enter, PreStepContext, PreStepDecision, PreStepReason, PrepareNextTurnContext, Reject,
    ShouldStopAfterTurnContext, TurnStopping, register_pre_step_listener,
    register_prepare_next_turn_listener, register_should_stop_after_turn_listener,
    resolve_stopping,
};
pub use driver::{AgentLoop, PromptInput};
pub use error::{AgentListenerError, AgentLoopError};
pub use events::{AgentEvent, AgentEventKind, dispatch_agent_event, register_agent_listener};
pub use state::{RunSnapshot, reduce_event};
