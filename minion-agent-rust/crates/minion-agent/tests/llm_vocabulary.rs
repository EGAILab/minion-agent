use minion_agent::llm::{
    AssistantContentBlock, AssistantMessage, Cost, ModelIdentity, ModelIdentityError, StopReason,
    TextBlock, ThinkingBlock, ToolCall, Usage,
};

#[test]
fn model_identity_requires_all_three_non_empty_components() {
    assert!(ModelIdentity::new("openai", "responses", "gpt-5").is_ok());

    assert_eq!(
        ModelIdentity::new("", "responses", "gpt-5")
            .expect_err("an empty provider must fail")
            .field(),
        "provider"
    );
    assert_eq!(
        ModelIdentity::new("openai", "", "gpt-5")
            .expect_err("an empty api must fail")
            .field(),
        "api"
    );
    assert_eq!(
        ModelIdentity::new("openai", "responses", "")
            .expect_err("an empty model id must fail")
            .field(),
        "model_id"
    );
}

#[test]
fn model_identity_has_value_equality_and_round_trips() {
    let identity = ModelIdentity::new("openai", "responses", "gpt-5")
        .expect("complete model identity must be valid");
    let equal_value = ModelIdentity::new(
        String::from("openai"),
        String::from("responses"),
        String::from("gpt-5"),
    )
    .expect("separately allocated equal strings must be equal identities");

    assert_eq!(identity, equal_value);

    let json = serde_json::to_value(&identity).expect("identity must serialize");
    assert_eq!(
        json,
        serde_json::json!({
            "provider": "openai",
            "api": "responses",
            "model_id": "gpt-5"
        })
    );
    assert_eq!(
        serde_json::from_value::<ModelIdentity>(json).expect("identity must deserialize"),
        identity
    );
}

#[test]
fn model_identity_deserialization_uses_the_same_validation_boundary() {
    let error = serde_json::from_value::<ModelIdentity>(serde_json::json!({
        "provider": "openai",
        "api": "",
        "model_id": "gpt-5"
    }))
    .expect_err("deserialization must not bypass identity validation");

    assert!(error.to_string().contains("api"));

    let direct = ModelIdentity::new("openai", "", "gpt-5")
        .expect_err("direct construction must reject the same value");
    assert_eq!(
        direct,
        ModelIdentityError::MissingComponent { field: "api" }
    );
}

#[test]
fn core_vocabulary_uses_canonical_snake_case_and_omits_absent_fields() {
    let text = TextBlock::new("answer");
    let thinking = ThinkingBlock::new("reason").with_signature("opaque");
    let tool_call = ToolCall::new("call-1", "lookup", serde_json::json!({"query": "rust"}))
        .with_namespace("web");

    assert_eq!(
        serde_json::to_value(StopReason::ToolUse).unwrap(),
        "tool_use"
    );
    assert_eq!(
        serde_json::to_value(StopReason::Deferred).unwrap(),
        "deferred"
    );
    assert_eq!(serde_json::to_value(&text).unwrap()["type"], "text");
    assert_eq!(serde_json::to_value(&thinking).unwrap()["redacted"], false);
    assert_eq!(
        serde_json::to_value(&tool_call).unwrap()["type"],
        "tool_call"
    );
    assert!(
        serde_json::to_value(&text)
            .unwrap()
            .get("text_signature")
            .is_none()
    );

    let usage = Usage {
        input: 1,
        output: 2,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: Some(1),
        total_tokens: 3,
        cost: Cost::default(),
    };
    let assistant = AssistantMessage::new(
        ModelIdentity::new("openai", "responses", "gpt-5").unwrap(),
        vec![
            AssistantContentBlock::Text(text),
            AssistantContentBlock::Thinking(thinking),
            AssistantContentBlock::ToolCall(tool_call),
        ],
        usage,
        StopReason::Stop,
        42.0,
    );
    let json = serde_json::to_value(&assistant).unwrap();

    assert_eq!(json["role"], "assistant");
    assert_eq!(json["api"], "responses");
    assert_eq!(json["provider"], "openai");
    assert_eq!(json["model"], "gpt-5");
    assert_eq!(json["usage"]["total_tokens"], 3);
    assert!(json.get("response_id").is_none());
    assert_eq!(
        serde_json::from_value::<AssistantMessage>(json).unwrap(),
        assistant
    );
}

#[test]
fn thinking_redacted_defaults_false_when_absent() {
    let value = serde_json::json!({"type": "thinking", "thinking": "visible"});
    let block: ThinkingBlock = serde_json::from_value(value).unwrap();
    assert!(!block.redacted);
}
