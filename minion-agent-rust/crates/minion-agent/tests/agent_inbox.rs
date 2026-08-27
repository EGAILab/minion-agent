use minion_agent::{
    agent::{ClaimPolicy, Inbox, InboxTarget},
    llm::{AssistantMessage, Message, ModelIdentity, ToolResultMessage, UserContent, UserMessage},
};

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
