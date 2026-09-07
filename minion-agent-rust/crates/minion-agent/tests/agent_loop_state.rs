use std::{
    collections::BTreeSet,
    sync::{Arc, Barrier, mpsc},
};

use minion_agent::{
    Runtime,
    agent::{AgentDefinition, AgentInstance, AgentRunError, AgentStatus, ThinkingLevel},
    agent_loop::{AgentEvent, reduce_event},
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, StopReason, StreamChunk,
        TextBlock, Usage, UserContent, UserMessage,
    },
    session::Session,
    tools::{
        AfterToolCallResult, ToolDefinition, ToolExecutionEnd, ToolExecutionRequest,
        ToolExecutionStart,
    },
};
use serde_json::json;

fn identity(model: &str) -> ModelIdentity {
    ModelIdentity::new("provider", "api", model).unwrap()
}

fn definition() -> AgentDefinition {
    AgentDefinition::new("ada", "system", identity("initial"))
}

fn instance(runtime: Option<&Runtime>, session: Session) -> AgentInstance {
    AgentInstance::new(
        "room-a",
        definition(),
        session,
        runtime.map(Runtime::context),
        None,
    )
}

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn assistant(text: &str, error_message: Option<&str>) -> AssistantMessage {
    let mut message = AssistantMessage::new(
        identity("initial"),
        vec![AssistantContentBlock::Text(TextBlock::new(text))],
        Usage::default(),
        StopReason::Stop,
        2.0,
    );
    message.error_message = error_message.map(str::to_owned);
    message
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        "description",
        serde_json::from_value(json!({"type": "object"})).unwrap(),
        name,
        |_request: ToolExecutionRequest| Box::pin(async { unreachable!() }),
    )
}

fn tool_start(call_id: &str) -> ToolExecutionStart {
    ToolExecutionStart {
        tool_call_id: call_id.into(),
        tool_name: "lookup".into(),
        arguments: json!({"query": "rust"}),
    }
}

fn tool_end(call_id: &str) -> ToolExecutionEnd {
    ToolExecutionEnd {
        tool_call_id: call_id.into(),
        tool_name: "lookup".into(),
        result: AfterToolCallResult {
            tool_call_id: call_id.into(),
            tool_name: "lookup".into(),
            content: vec![],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            terminate: None,
        },
    }
}

#[test]
fn run_snapshot_is_one_time_owned_top_level_state_with_shared_tool_identity() {
    let runtime = Runtime::new();
    let session = Session::new("room-a", [] as [&str; 0]).unwrap();
    session.append_message(user("history-1")).unwrap();
    let agent = instance(Some(&runtime), session);
    let first_registration = runtime
        .tools()
        .register_for_scope(None, tool("first"))
        .unwrap();
    let projected_before = agent.tools();

    let snapshot = agent.try_begin_run().unwrap();

    assert_eq!(snapshot.context.system_prompt, "system");
    assert_eq!(snapshot.context.messages, vec![user("history-1")]);
    assert_eq!(snapshot.context.tools.len(), 1);
    assert!(Arc::ptr_eq(
        &snapshot.context.tools[0],
        &projected_before[0]
    ));
    assert_eq!(snapshot.config.model, identity("initial"));
    assert_eq!(snapshot.config.thinking_level, ThinkingLevel::Off);

    agent.set_system_prompt("changed");
    agent.set_model(identity("changed"));
    agent.set_thinking_level(ThinkingLevel::High);
    agent.session().append_message(user("history-2")).unwrap();
    let second_registration = runtime
        .tools()
        .register_for_scope(None, tool("second"))
        .unwrap();

    assert_eq!(snapshot.context.system_prompt, "system");
    assert_eq!(snapshot.context.messages, vec![user("history-1")]);
    assert_eq!(snapshot.context.tools.len(), 1);
    assert_eq!(snapshot.config.model, identity("initial"));
    assert_eq!(snapshot.config.thinking_level, ThinkingLevel::Off);

    agent.finish_run();
    let next = agent.try_begin_run().unwrap();
    assert_eq!(next.context.system_prompt, "changed");
    assert_eq!(
        next.context.messages,
        vec![user("history-1"), user("history-2")]
    );
    assert_eq!(
        next.context
            .tools
            .iter()
            .map(|definition| definition.name())
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert_eq!(next.config.model, identity("changed"));
    assert_eq!(next.config.thinking_level, ThinkingLevel::High);
    agent.finish_run();

    drop((first_registration, second_registration));
}

#[test]
fn begin_run_is_an_atomic_single_authority_transition() {
    let agent = Arc::new(instance(
        None,
        Session::new("room-a", [] as [&str; 0]).unwrap(),
    ));
    let start = Arc::new(Barrier::new(3));
    let (send, receive) = mpsc::channel();
    let mut threads = Vec::new();

    for _ in 0..2 {
        let agent = Arc::clone(&agent);
        let start = Arc::clone(&start);
        let send = send.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            send.send(agent.try_begin_run()).unwrap();
        }));
    }
    start.wait();

    let results = [receive.recv().unwrap(), receive.recv().unwrap()];
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(AgentRunError::Active)))
            .count(),
        1
    );
    assert_eq!(
        AgentRunError::Active.to_string(),
        "Agent is already processing."
    );
    assert_eq!(agent.status(), AgentStatus::Running);

    agent.finish_run();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(agent.status(), AgentStatus::Idle);
}

#[test]
fn failed_snapshot_resolution_rolls_back_every_entry_mutation() {
    let session = Session::new("room-a", [] as [&str; 0]).unwrap();
    session
        .append_raw("assistant/message", json!({"message": "malformed"}))
        .unwrap();
    let agent = instance(None, session);
    agent.set_streaming_message(Some(user("pre-existing")));
    agent.set_error_message(Some("previous failure".into()));

    let error = agent.try_begin_run().unwrap_err();

    assert!(matches!(error, AgentRunError::Session(_)));
    assert_eq!(agent.status(), AgentStatus::Idle);
    assert_eq!(agent.streaming_message(), Some(user("pre-existing")));
    assert_eq!(agent.error_message().as_deref(), Some("previous failure"));
}

#[test]
fn message_reduction_tracks_complete_partial_and_appends_only_at_end() {
    let session = Session::new("room-a", [] as [&str; 0]).unwrap();
    let agent = instance(None, session);
    agent.try_begin_run().unwrap();
    let initial = assistant("p", None);
    let partial = assistant("partial", None);
    let final_message = assistant("complete", None);

    reduce_event(
        &agent,
        &AgentEvent::MessageStart(Message::Assistant(Box::new(initial.clone()))),
    )
    .unwrap();
    assert_eq!(
        agent.streaming_message(),
        Some(Message::Assistant(Box::new(initial)))
    );
    assert!(agent.messages().unwrap().is_empty());

    reduce_event(
        &agent,
        &AgentEvent::MessageUpdate {
            event: StreamChunk::TextDelta {
                content_index: 0,
                delta: "artial".into(),
                partial: partial.clone(),
            },
            partial: partial.clone(),
        },
    )
    .unwrap();
    assert_eq!(
        agent.streaming_message(),
        Some(Message::Assistant(Box::new(partial)))
    );
    assert!(agent.messages().unwrap().is_empty());

    reduce_event(
        &agent,
        &AgentEvent::MessageEnd(Message::Assistant(Box::new(final_message.clone()))),
    )
    .unwrap();
    assert_eq!(agent.streaming_message(), None);
    assert_eq!(
        agent.messages().unwrap(),
        vec![Message::Assistant(Box::new(final_message))]
    );
    agent.finish_run();
}

#[test]
fn tool_reduction_adds_and_removes_only_the_matching_pending_call() {
    let agent = instance(None, Session::new("room-a", [] as [&str; 0]).unwrap());
    agent.try_begin_run().unwrap();

    reduce_event(&agent, &AgentEvent::ToolExecutionStart(tool_start("a"))).unwrap();
    reduce_event(&agent, &AgentEvent::ToolExecutionStart(tool_start("b"))).unwrap();
    assert_eq!(
        agent.pending_tool_calls(),
        BTreeSet::from(["a".into(), "b".into()])
    );

    reduce_event(&agent, &AgentEvent::ToolExecutionEnd(tool_end("a"))).unwrap();
    assert_eq!(agent.pending_tool_calls(), BTreeSet::from(["b".into()]));
    agent.finish_run();
}

#[test]
fn error_persists_after_agent_end_and_finish_until_the_next_run_starts() {
    let agent = instance(None, Session::new("room-a", [] as [&str; 0]).unwrap());
    agent.try_begin_run().unwrap();
    let failed = assistant("", Some("provider failed"));

    reduce_event(
        &agent,
        &AgentEvent::TurnEnd {
            message: failed,
            tool_results: vec![],
        },
    )
    .unwrap();
    reduce_event(
        &agent,
        &AgentEvent::AgentEnd {
            reason: minion_agent::agent_loop::AgentEndReason::Completed,
            causes: vec![],
            messages: vec![],
        },
    )
    .unwrap();
    assert_eq!(agent.error_message().as_deref(), Some("provider failed"));

    agent.finish_run();
    assert_eq!(agent.status(), AgentStatus::Idle);
    assert_eq!(agent.error_message().as_deref(), Some("provider failed"));

    agent.try_begin_run().unwrap();
    assert_eq!(agent.error_message(), None);
    agent.finish_run();
}

#[test]
fn finish_run_unconditionally_clears_transient_state_and_returns_idle() {
    let agent = instance(None, Session::new("room-a", [] as [&str; 0]).unwrap());
    agent.try_begin_run().unwrap();
    reduce_event(
        &agent,
        &AgentEvent::MessageStart(Message::Assistant(Box::new(assistant("partial", None)))),
    )
    .unwrap();
    reduce_event(
        &agent,
        &AgentEvent::ToolExecutionStart(tool_start("call-1")),
    )
    .unwrap();

    agent.finish_run();

    assert_eq!(agent.status(), AgentStatus::Idle);
    assert_eq!(agent.streaming_message(), None);
    assert!(agent.pending_tool_calls().is_empty());
}
