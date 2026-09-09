mod context;
mod decisions;
mod driver;
mod error;
mod events;
mod state;

pub use context::{RunConfig, RunConfigUpdate, RunContext};
pub use decisions::{
    Enter, PreStepContext, PreStepDecision, PreStepReason, PrepareNextTurnContext, Reject,
    ShouldStopAfterTurnContext, TransformContext, TransformContextAction, TurnStopping,
    register_pre_step_listener, register_prepare_next_turn_listener,
    register_should_stop_after_turn_listener, register_transform_context_listener,
    resolve_stopping,
};
pub use driver::{AgentLoop, PromptInput};
pub use error::{AgentListenerError, AgentLoopError};
pub use events::{
    AgentEndReason, AgentEvent, AgentEventKind, AgentLifecycleContext, RunCause,
    dispatch_agent_event, register_agent_listener, register_agent_listener_with_signal,
};
pub use state::{RunSnapshot, reduce_event};
