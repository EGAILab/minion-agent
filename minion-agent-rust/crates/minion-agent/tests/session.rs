use std::{collections::BTreeMap, sync::Arc};

use minion_agent::{
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, StopReason, TextBlock,
        ToolDefinition, Usage, UserContent, UserMessage,
    },
    session::{ArtifactStore, EventKind, Session, SessionError},
};

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn assistant(text: &str) -> Message {
    Message::Assistant(Box::new(AssistantMessage::new(
        ModelIdentity::new("mock", "mock", "mock-1").unwrap(),
        vec![AssistantContentBlock::Text(TextBlock::new(text))],
        Usage::default(),
        StopReason::Stop,
        1.0,
    )))
}

#[test]
fn event_identity_is_open_validated_and_compared_by_value() {
    let literal = EventKind::new("plugin/note").unwrap();
    assert_eq!(
        literal,
        EventKind::new(String::from("plugin/note")).unwrap()
    );
    assert_eq!(
        EventKind::new("Plugin/Note"),
        Err(SessionError::InvalidEventKind)
    );
}

#[test]
fn append_is_gapless_and_message_round_trips_through_the_log() {
    let session = Session::new("s1", [] as [&str; 0]).unwrap();
    let first = session.append_message(user("hello")).unwrap();
    let second = session.append_message(assistant("world")).unwrap();
    assert_eq!((first.seq, second.seq), (1, 2));
    assert_eq!(
        session.derive_messages().unwrap(),
        vec![user("hello"), assistant("world")]
    );
}

#[test]
fn concurrent_appends_allocate_sequence_in_committed_log_order() {
    let session = Arc::new(Session::new("concurrent", [] as [&str; 0]).unwrap());
    let workers = (0..32)
        .map(|index| {
            let session = session.clone();
            std::thread::spawn(move || {
                session
                    .append("plugin/audit", serde_json::json!({"index": index}))
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    let mut returned = workers
        .into_iter()
        .map(|worker| worker.join().unwrap().seq)
        .collect::<Vec<_>>();
    returned.sort_unstable();
    assert_eq!(returned, (1..=32).collect::<Vec<_>>());
    assert_eq!(
        session
            .events()
            .iter()
            .map(|event| event.seq)
            .collect::<Vec<_>>(),
        (1..=32).collect::<Vec<_>>()
    );
}

#[test]
fn reset_compaction_and_fork_are_derived_by_the_real_session() {
    let parent = Session::new("parent", [] as [&str; 0]).unwrap();
    parent.append_message(user("shared")).unwrap();
    let child = parent.fork("child", None).unwrap();
    parent.append_message(user("parent later")).unwrap();
    child.append_message(user("one")).unwrap();
    child.append_message(user("two")).unwrap();
    child.compact("summary", 1).unwrap();
    let summary = Message::User(UserMessage::new(UserContent::Text("summary".into()), 0.0));
    assert_eq!(child.derive_messages().unwrap(), vec![summary, user("two")]);
    assert_eq!(
        parent.derive_messages().unwrap(),
        vec![user("shared"), user("parent later")]
    );

    child.reset().unwrap();
    child.append_message(user("after reset")).unwrap();
    assert_eq!(child.derive_messages().unwrap(), vec![user("after reset")]);
}

#[test]
fn open_event_is_log_only_unless_explicitly_surface_admitted() {
    let hidden = Session::new("hidden", [] as [&str; 0]).unwrap();
    hidden
        .append_projectable("plugin/note", user("hidden"))
        .unwrap();
    assert!(hidden.derive_messages().unwrap().is_empty());

    let visible = Session::new("visible", ["plugin/note"]).unwrap();
    visible
        .append_projectable("plugin/note", user("visible"))
        .unwrap();
    assert_eq!(visible.derive_messages().unwrap(), vec![user("visible")]);
}

#[test]
fn artifacts_are_content_addressed_and_headers_reconstruct_real_state() {
    let store = Arc::new(ArtifactStore::new());
    let session = Session::with_artifacts("s1", [] as [&str; 0], store.clone()).unwrap();
    let mut components = BTreeMap::new();
    components.insert("memory".into(), "remember".into());
    components.insert("system_base".into(), "be helpful".into());
    let tools = vec![ToolDefinition {
        name: "lookup".into(),
        description: "look up".into(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    let header = session
        .record_header(components.clone(), "mock-1", tools.clone())
        .unwrap();
    let reconstructed = session.reconstruct_header(&header).unwrap();
    assert_eq!(reconstructed.components, components);
    assert_eq!(reconstructed.tools, tools);
    assert_eq!(store.len(), 3);
    assert_eq!(reconstructed.assembled_system, "remember\n\nbe helpful");
}
