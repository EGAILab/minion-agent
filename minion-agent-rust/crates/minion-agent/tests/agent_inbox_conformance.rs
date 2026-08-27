#![cfg(feature = "conformance")]

use std::{collections::BTreeMap, fs, path::PathBuf};

use minion_agent::{
    agent::{ClaimPolicy, Inbox, InboxTarget},
    llm::{Message, UserContent, UserMessage},
};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn text_message(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn text(message: &Message) -> &str {
    match message {
        Message::User(message) => match &message.content {
            UserContent::Text(value) => value,
            UserContent::Blocks(_) => panic!("canonical inbox fixture uses text messages"),
        },
        _ => panic!("canonical inbox fixture uses user messages"),
    }
}

fn target(value: &str) -> InboxTarget {
    match value {
        "steering" => InboxTarget::Steering,
        "follow_up" => InboxTarget::FollowUp,
        _ => panic!("invalid canonical queue {value}"),
    }
}

fn run(document: &Value) -> BTreeMap<String, Value> {
    let inbox = Inbox::new();
    let mut observed = BTreeMap::new();
    for action in document["agent_inbox"]["actions"].as_array().unwrap() {
        let observe = action.get("observe").and_then(Value::as_str);
        if let Some(spec) = action.get("steer") {
            inbox.steer(text_message(spec["text"].as_str().unwrap()), None);
        } else if let Some(spec) = action.get("follow_up") {
            inbox.follow_up(text_message(spec["text"].as_str().unwrap()), None);
        } else if let Some(spec) = action.get("inject") {
            inbox.inject(text_message(spec["text"].as_str().unwrap()), None);
        } else if let Some(spec) = action.get("clear") {
            match spec["queue"].as_str().unwrap() {
                "all" => inbox.clear_all(),
                queue => inbox.clear(target(queue)),
            }
        } else if let Some(spec) = action.get("claim") {
            let policy = match spec["mode"].as_str().unwrap() {
                "all" => ClaimPolicy::All,
                "one-at-a-time" => ClaimPolicy::OneAtATime,
                value => panic!("invalid canonical claim mode {value}"),
            };
            let claimed = inbox.claim(target(spec["queue"].as_str().unwrap()), policy);
            if let Some(name) = observe {
                observed.insert(
                    name.into(),
                    json!(claimed.iter().map(|e| text(&e.message)).collect::<Vec<_>>()),
                );
            }
        } else if action.get("has_queued_messages").is_some() {
            if let Some(name) = observe {
                observed.insert(name.into(), json!(inbox.has_pending()));
            }
        } else {
            panic!("schema-valid action has one supported operation");
        }
    }
    observed
}

#[test]
fn all_layer_07_inbox_scenarios_drive_the_real_rust_inbox() {
    let directory = root().join("conformance/agent");
    let mut scenarios = fs::read_dir(directory)
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let source = fs::read_to_string(entry.path()).ok()?;
            let document: Value = serde_yaml::from_str(&source).ok()?;
            document.get("agent_inbox")?;
            Some((entry.path(), document))
        })
        .collect::<Vec<_>>();
    scenarios.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(scenarios.len(), 2);
    for (path, document) in scenarios {
        let actual = run(&document);
        let expected: BTreeMap<String, Value> =
            serde_json::from_value(document["expect"].clone()).unwrap();
        assert_eq!(actual, expected, "{}", path.display());
    }
}
