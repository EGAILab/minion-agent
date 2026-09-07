use minion_agent::{
    agent::{ClaimPolicy, Inbox, InboxTarget},
    llm::{AssistantMessage, Message, ModelIdentity, ToolResultMessage, UserContent, UserMessage},
};
use serde_json::json;

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn model() -> ModelIdentity {
    ModelIdentity::new("mock", "mock", "m").unwrap()
}

#[test]
fn inbox_accepts_the_complete_pinned_agent_message_domain() {
    let inbox = Inbox::new();
    inbox.steer(user("user"), None);
    inbox.steer(
        Message::Assistant(Box::new(AssistantMessage::pending(model(), 1.0))),
        None,
    );
    inbox.steer(
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "t1",
            "echo",
            vec![],
            false,
            1.0,
        ))),
        None,
    );
    assert_eq!(
        inbox.claim(InboxTarget::Steering, ClaimPolicy::All).len(),
        3
    );
}

#[test]
fn claims_are_fifo_and_queue_modes_are_exact() {
    let inbox = Inbox::new();
    for value in ["A", "B", "C"] {
        inbox.steer(user(value), None);
    }
    let first = inbox.claim(InboxTarget::Steering, ClaimPolicy::OneAtATime);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].message, user("A"));
    let rest = inbox.claim(InboxTarget::Steering, ClaimPolicy::All);
    assert_eq!(
        rest.iter().map(|v| &v.message).collect::<Vec<_>>(),
        vec![&user("B"), &user("C")]
    );
    assert!(
        inbox
            .claim(InboxTarget::Steering, ClaimPolicy::All)
            .is_empty()
    );
}

#[test]
fn queues_clear_independently_and_clear_all_preserves_wake() {
    let inbox = Inbox::new();
    inbox.steer(user("S"), None);
    inbox.follow_up(user("F"), None);
    assert!(inbox.has_pending());
    assert!(inbox.wake_requested());
    inbox.clear(InboxTarget::Steering);
    assert!(
        inbox
            .claim(InboxTarget::Steering, ClaimPolicy::All)
            .is_empty()
    );
    assert_eq!(
        inbox.claim(InboxTarget::FollowUp, ClaimPolicy::All).len(),
        1
    );
    inbox.steer(user("S2"), None);
    inbox.follow_up(user("F2"), None);
    inbox.clear_all();
    assert!(!inbox.has_pending());
    assert!(inbox.wake_requested());
    assert!(inbox.take_wake());
    assert!(!inbox.take_wake());
}

#[test]
fn injection_is_steering_without_requesting_wake_or_consuming_content() {
    let inbox = Inbox::new();
    inbox.inject(user("ambient"), None);
    assert!(!inbox.wake_requested());
    assert_eq!(
        inbox.claim(InboxTarget::Steering, ClaimPolicy::All).len(),
        1
    );
    inbox.follow_up(user("queued"), None);
    assert!(inbox.take_wake());
    assert_eq!(
        inbox.claim(InboxTarget::FollowUp, ClaimPolicy::All).len(),
        1
    );
}

#[test]
fn envelopes_have_unique_stable_ids_and_preserve_opaque_origins() {
    let first_origin = json!({"nested": [1, null, {"flag": true}]});
    let inbox = Inbox::new();

    let first = inbox.steer(user("first"), Some(first_origin.clone()));
    let second = inbox.follow_up(user("second"), Some(serde_json::Value::Null));
    let absent = inbox.inject(user("third"), None);
    let other_inbox = Inbox::new();
    let other = other_inbox.steer(user("other"), None);

    assert!(!first.id.is_empty());
    assert!(!second.id.is_empty());
    assert!(!absent.id.is_empty());
    assert_ne!(first.id, second.id);
    assert_ne!(first.id, absent.id);
    assert_ne!(second.id, absent.id);
    assert_ne!(first.id, other.id);
    assert_ne!(second.id, other.id);
    assert_ne!(absent.id, other.id);
    assert_eq!(first.origin, Some(first_origin));
    assert_eq!(second.origin, Some(serde_json::Value::Null));
    assert_eq!(absent.origin, None);
}

#[test]
fn target_pending_is_a_non_consuming_fifo_snapshot() {
    let inbox = Inbox::new();
    let first = inbox.follow_up(user("first"), Some(json!("one")));
    let second = inbox.follow_up(user("second"), Some(json!("two")));
    inbox.steer(user("steering"), None);

    let pending = inbox.pending(InboxTarget::FollowUp);
    assert_eq!(pending, vec![first.clone(), second.clone()]);
    assert_eq!(inbox.pending(InboxTarget::FollowUp), pending);
    assert_eq!(
        inbox.claim(InboxTarget::FollowUp, ClaimPolicy::All),
        vec![first, second]
    );
    assert!(inbox.pending(InboxTarget::FollowUp).is_empty());
    assert_eq!(inbox.pending(InboxTarget::Steering).len(), 1);
}
