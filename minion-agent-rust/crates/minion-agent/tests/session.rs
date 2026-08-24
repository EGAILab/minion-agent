use std::{
    collections::BTreeMap,
    sync::{Arc, Barrier},
};

use minion_agent::{
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, StopReason, TextBlock,
        ToolDefinition, Usage, UserContent, UserContentBlock, UserMessage,
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

    for valid in [
        "plugin/foo",
        "plugin/foo-bar",
        "plugin/foo_bar",
        "plugin2/foo",
    ] {
        assert!(EventKind::new(valid).is_ok(), "{valid} must be accepted");
    }
    for invalid in [
        "Plugin/foo",
        "plugin-name/foo",
        "plugin//foo",
        "/foo",
        "plugin/",
    ] {
        assert!(
            EventKind::new(invalid).is_err(),
            "{invalid} must be rejected"
        );
    }
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
fn string_and_block_user_content_remain_distinct_through_fork_derivation() {
    let parent = Session::new("parent", [] as [&str; 0]).unwrap();
    let string_message = Message::User(UserMessage::new(UserContent::Text("hello".into()), 1.0));
    let block_message = Message::User(UserMessage::new(
        UserContent::Blocks(vec![UserContentBlock::Text(TextBlock::new("hello"))]),
        2.0,
    ));
    parent.append_message(string_message.clone()).unwrap();
    parent.append_message(block_message.clone()).unwrap();

    assert_eq!(
        parent.derive_messages().unwrap(),
        vec![string_message.clone(), block_message.clone()]
    );

    let child = parent.fork("child", None).unwrap();
    let after_fork = Message::User(UserMessage::new(
        UserContent::Text("still a string after fork".into()),
        3.0,
    ));
    child.append_message(after_fork.clone()).unwrap();
    assert_eq!(
        child.derive_messages().unwrap(),
        vec![string_message, block_message, after_fork]
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
                    .append_raw("plugin/audit", serde_json::json!({"index": index}))
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
fn concurrent_compaction_marker_provenance_matches_its_serialized_snapshot() {
    let session = Arc::new(Session::new("linearized", [] as [&str; 0]).unwrap());
    let barrier = Arc::new(Barrier::new(8));
    let workers = (0..8)
        .map(|worker| {
            let session = session.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for iteration in 0..64 {
                    match (worker + iteration) % 5 {
                        0 => {
                            session.reset().unwrap();
                        }
                        1 => {
                            session
                                .compact(format!("summary-{worker}-{iteration}"), 0)
                                .unwrap();
                        }
                        _ => {
                            session
                                .append_message(user(&format!("message-{worker}-{iteration}")))
                                .unwrap();
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }

    let events = session.events();
    assert_eq!(
        events.iter().map(|event| event.seq).collect::<Vec<_>>(),
        (1..=events.len() as u64).collect::<Vec<_>>()
    );
    for marker in events
        .iter()
        .filter(|event| event.kind.as_str() == "session/compaction")
    {
        let floor = events
            .iter()
            .rev()
            .find(|event| event.seq < marker.seq && event.kind.as_str() == "session/reset")
            .map_or(0, |event| event.seq);
        let expected_through = events
            .iter()
            .filter(|event| {
                event.seq > floor
                    && event.seq < marker.seq
                    && matches!(
                        event.kind.as_str(),
                        "user/message" | "assistant/message" | "tool/result"
                    )
            })
            .map(|event| event.seq)
            .max()
            .unwrap_or(0);
        assert_eq!(
            marker.data["superseded_through"],
            serde_json::json!(expected_through),
            "compaction marker {} must describe exactly its serialized snapshot",
            marker.seq
        );
    }
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
fn a_fork_rejects_a_boundary_beyond_the_committed_tip() {
    let session = Session::new("parent", [] as [&str; 0]).unwrap();
    assert_eq!(
        session.fork("child", Some(1)).err(),
        Some(SessionError::InvalidForkBoundary {
            boundary: 1,
            tip: 0
        })
    );
}

#[test]
fn open_event_is_log_only_unless_explicitly_surface_admitted() {
    let hidden = Session::new("hidden", [] as [&str; 0]).unwrap();
    hidden
        .append_projectable(EventKind::new("plugin/note").unwrap(), user("hidden"))
        .unwrap();
    assert!(hidden.derive_messages().unwrap().is_empty());

    let visible = Session::new("visible", ["plugin/note"]).unwrap();
    visible
        .append_projectable(EventKind::new("plugin/note").unwrap(), user("visible"))
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
    assert_eq!(reconstructed.model, "mock-1");
    assert_eq!(store.len(), 3);
    assert_eq!(reconstructed.assembled_system, "remember\n\nbe helpful");
}
