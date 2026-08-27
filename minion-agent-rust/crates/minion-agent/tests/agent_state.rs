use std::collections::BTreeSet;

use minion_agent::{
    Runtime,
    agent::{
        AgentDefinition, AgentError, AgentInstance, AgentStatus, ClaimPolicy, InboxTarget,
        ThinkingLevel,
    },
    llm::{Message, ModelIdentity, UserContent, UserMessage},
    session::Session,
    tools::{ToolDefinition, ToolExecutionRequest},
};
use serde_json::json;

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn definition() -> AgentDefinition {
    AgentDefinition::new(
        "ada",
        "system",
        ModelIdentity::new("mock", "mock", "m").unwrap(),
    )
}

fn instance(runtime: Option<&Runtime>) -> AgentInstance {
    AgentInstance::new(
        "room-a",
        definition(),
        Session::new("room-a", [] as [&str; 0]).unwrap(),
        runtime.map(Runtime::context),
        None,
    )
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        "description",
        serde_json::from_value(json!({"type":"object"})).unwrap(),
        name,
        |_request: ToolExecutionRequest| Box::pin(async { unreachable!() }),
    )
}

#[test]
fn state_has_the_approved_initial_vocabulary_and_mutable_configuration() {
    let agent = instance(None);
    assert_eq!(agent.system_prompt(), "system");
    assert_eq!(agent.thinking_level(), ThinkingLevel::Off);
    assert_eq!(agent.status(), AgentStatus::Idle);
    assert_eq!(agent.streaming_message(), None);
    assert_eq!(agent.pending_tool_calls(), BTreeSet::new());
    assert_eq!(agent.error_message(), None);
    agent.set_system_prompt("changed");
    agent.set_thinking_level(ThinkingLevel::High);
    assert_eq!(agent.system_prompt(), "changed");
    assert_eq!(agent.thinking_level(), ThinkingLevel::High);
    assert_eq!(
        serde_json::to_value(ThinkingLevel::XHigh).unwrap(),
        json!("xhigh")
    );
}

#[test]
fn messages_are_a_live_session_projection_and_tools_are_total() {
    let runtime = Runtime::new();
    let agent = instance(Some(&runtime));
    assert!(agent.messages().unwrap().is_empty());
    agent.session().append_message(user("hello")).unwrap();
    assert_eq!(agent.messages().unwrap(), vec![user("hello")]);
    assert!(instance(None).tools().is_empty());
    assert!(agent.tools().is_empty());
}

#[test]
fn tools_are_a_live_layer_05_projection_and_survive_reset() {
    let runtime = Runtime::new();
    let agent = instance(Some(&runtime));
    let registration = runtime
        .tools()
        .register_for_scope(None, tool("echo"))
        .unwrap();
    assert_eq!(agent.tools()[0].name(), "echo");
    agent.reset().unwrap();
    assert_eq!(agent.tools()[0].name(), "echo");
    registration.withdraw();
    assert!(agent.tools().is_empty());
}

#[test]
fn reset_uses_session_reset_and_preserves_wake_configuration_and_identity() {
    let agent = instance(None);
    agent.session().append_message(user("before")).unwrap();
    agent.inbox().steer(user("queued"), None);
    agent.set_streaming_message(Some(user("partial")));
    agent.set_pending_tool_calls(BTreeSet::from(["t1".into()]));
    agent.set_error_message(Some("boom".into()));
    let before_events = agent.session().events().len();

    agent.reset().unwrap();

    assert!(agent.messages().unwrap().is_empty());
    assert_eq!(agent.session().events().len(), before_events + 1);
    assert!(!agent.inbox().has_pending());
    assert!(agent.inbox().wake_requested());
    assert_eq!(agent.system_prompt(), "system");
    assert_eq!(agent.id(), "room-a");
    assert_eq!(agent.streaming_message(), None);
    assert!(agent.pending_tool_calls().is_empty());
    assert_eq!(agent.error_message(), None);
}

#[test]
fn active_reset_is_exact_and_atomic() {
    let agent = instance(None);
    agent.session().append_message(user("before")).unwrap();
    agent.inbox().steer(user("queued"), None);
    agent.set_error_message(Some("keep".into()));
    agent.set_status(AgentStatus::Running);
    let events = agent.session().events();

    assert_eq!(agent.reset(), Err(AgentError::Active));
    assert_eq!(
        AgentError::Active.to_string(),
        "Agent is already processing. Wait for completion before resetting."
    );
    assert_eq!(agent.session().events(), events);
    assert!(agent.inbox().has_pending());
    assert_eq!(agent.error_message().as_deref(), Some("keep"));
}

#[test]
fn agent_level_inbox_surface_delegates_without_adding_run_timing() {
    let agent = instance(None);
    agent.steer(user("S"), None);
    agent.follow_up(user("F"), None);
    agent.inject(user("I"), None);
    assert!(agent.has_queued_messages());
    assert_eq!(
        agent
            .inbox()
            .claim(InboxTarget::Steering, ClaimPolicy::All)
            .len(),
        2
    );
    agent.clear_follow_up_queue();
    assert!(!agent.has_queued_messages());
    agent.steer(user("S2"), None);
    agent.follow_up(user("F2"), None);
    agent.clear_all_queues();
    assert!(!agent.has_queued_messages());
}
