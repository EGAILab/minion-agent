use std::collections::BTreeMap;

#[cfg(feature = "conformance")]
use minion_agent::llm::transform_legacy_messages;
use minion_agent::llm::{
    AssistantContentBlock, AssistantMessage, AssistantMessageDiagnostic, DeferredHandle,
    DiagnosticCode, DiagnosticError, ImageBlock, Message, ModelIdentity, StopReason, TextBlock,
    ThinkingBlock, ToolCall, ToolResultContentBlock, ToolResultMessage, TransformTarget, Usage,
    UserContent, UserContentBlock, UserMessage, transform_messages,
};

fn target(provider: &str, api: &str, model: &str, supports_images: bool) -> TransformTarget {
    TransformTarget::new(
        ModelIdentity::new(provider, api, model).unwrap(),
        supports_images,
    )
}

fn user_text(value: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(value.into()), 1.0))
}

fn image() -> ImageBlock {
    ImageBlock::data("image/png", "eA==")
}

fn user_blocks(blocks: Vec<UserContentBlock>) -> Message {
    Message::User(UserMessage::new(UserContent::Blocks(blocks), 1.0))
}

fn tool_blocks(blocks: Vec<ToolResultContentBlock>) -> Message {
    Message::ToolResult(Box::new(ToolResultMessage::new(
        "call-1", "lookup", blocks, false, 2.0,
    )))
}

fn assistant(
    provider: &str,
    api: &str,
    model: &str,
    content: Vec<AssistantContentBlock>,
) -> Message {
    assistant_with_reason(provider, api, model, content, StopReason::Stop)
}

fn assistant_with_reason(
    provider: &str,
    api: &str,
    model: &str,
    content: Vec<AssistantContentBlock>,
    stop_reason: StopReason,
) -> Message {
    Message::Assistant(Box::new(AssistantMessage::new(
        ModelIdentity::new(provider, api, model).unwrap(),
        content,
        Usage::default(),
        stop_reason,
        3.0,
    )))
}

fn thinking(text: &str, signature: Option<&str>, redacted: bool) -> AssistantContentBlock {
    let mut block = ThinkingBlock::new(text);
    block.thinking_signature = signature.map(str::to_owned);
    block.redacted = redacted;
    AssistantContentBlock::Thinking(block)
}

fn tool_call(id: &str, name: &str) -> AssistantContentBlock {
    AssistantContentBlock::ToolCall(ToolCall::new(id, name, BTreeMap::new()))
}

#[test]
fn string_user_content_is_preserved_for_every_target_capability_and_identity() {
    for value in [
        "hello",
        "",
        "   ",
        "(image omitted: model does not support images)",
    ] {
        let source = vec![user_text(value)];
        for target in [
            target("p", "a", "m", true),
            target("p", "a", "m", false),
            target("other", "a", "m", false),
            target("p", "other", "m", false),
            target("p", "a", "other", false),
        ] {
            assert_eq!(transform_messages(&source, &target, None), source);
        }
    }
}

#[test]
fn transform_returns_values_without_mutating_the_source_history() {
    let source = vec![user_text("hello")];
    let snapshot = source.clone();
    let transformed = transform_messages(&source, &target("p", "a", "m", false), None);

    assert_eq!(source, snapshot);
    assert_eq!(transformed, snapshot);
}

#[test]
fn image_capability_preserves_or_downgrades_role_specific_blocks() {
    let source = vec![
        user_blocks(vec![UserContentBlock::Image(image())]),
        tool_blocks(vec![ToolResultContentBlock::Image(image())]),
    ];

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", true), None),
        source
    );
    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", false), None),
        vec![
            user_blocks(vec![UserContentBlock::Text(TextBlock::new(
                "(image omitted: model does not support images)",
            ))]),
            tool_blocks(vec![ToolResultContentBlock::Text(TextBlock::new(
                "(tool image omitted: model does not support images)",
            ))]),
        ]
    );
}

#[test]
fn image_placeholder_suppression_is_per_run_and_per_message() {
    let placeholder = "(image omitted: model does not support images)";
    let source = vec![
        user_blocks(vec![
            UserContentBlock::Image(image()),
            UserContentBlock::Image(image()),
            UserContentBlock::Image(image()),
            UserContentBlock::Text(TextBlock::new("break")),
            UserContentBlock::Image(image()),
            UserContentBlock::Text(TextBlock::new(placeholder)),
            UserContentBlock::Image(image()),
        ]),
        user_blocks(vec![UserContentBlock::Image(image())]),
    ];

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", false), None),
        vec![
            user_blocks(vec![
                UserContentBlock::Text(TextBlock::new(placeholder)),
                UserContentBlock::Text(TextBlock::new("break")),
                UserContentBlock::Text(TextBlock::new(placeholder)),
                UserContentBlock::Text(TextBlock::new(placeholder)),
            ]),
            user_blocks(vec![UserContentBlock::Text(TextBlock::new(placeholder))]),
        ]
    );
}

#[test]
fn same_model_thinking_retains_only_the_certified_cases() {
    let source = vec![assistant(
        "p",
        "a",
        "m",
        vec![
            thinking("", None, true),
            thinking("", Some("signed"), false),
            thinking("visible", None, false),
            thinking("   ", None, false),
        ],
    )];

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", true), None),
        vec![assistant(
            "p",
            "a",
            "m",
            vec![
                thinking("", None, true),
                thinking("", Some("signed"), false),
                thinking("visible", None, false),
            ],
        )]
    );
}

#[test]
fn cross_model_thinking_and_signatures_are_made_compatible() {
    let mut call = ToolCall::new("call-1", "lookup", BTreeMap::new());
    call.thought_signature = Some("thought-sig".into());
    call.namespace = Some("tools".into());
    let source = vec![
        assistant(
            "p",
            "a",
            "m1",
            vec![
                thinking("opaque", Some("redacted-sig"), true),
                thinking("reasoning", Some("thinking-sig"), false),
                thinking("", Some("empty-sig"), false),
                AssistantContentBlock::Text(TextBlock::new("answer").with_signature("text-sig")),
                AssistantContentBlock::ToolCall(call),
            ],
        ),
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "call-1",
            "lookup",
            Vec::new(),
            false,
            4.0,
        ))),
    ];
    let mut expected_call = ToolCall::new("call-1", "lookup", BTreeMap::new());
    expected_call.namespace = Some("tools".into());

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m2", true), None),
        vec![
            assistant(
                "p",
                "a",
                "m1",
                vec![
                    AssistantContentBlock::Text(TextBlock::new("reasoning")),
                    AssistantContentBlock::Text(TextBlock::new("answer")),
                    AssistantContentBlock::ToolCall(expected_call),
                ],
            ),
            Message::ToolResult(Box::new(ToolResultMessage::new(
                "call-1",
                "lookup",
                Vec::new(),
                false,
                4.0,
            ))),
        ]
    );
}

#[test]
fn every_identity_component_participates_in_signature_compatibility() {
    let source = vec![assistant(
        "p",
        "a",
        "m",
        vec![AssistantContentBlock::Text(
            TextBlock::new("answer").with_signature("sig"),
        )],
    )];
    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", true), None),
        source
    );
    for cross in [
        target("other", "a", "m", true),
        target("p", "other", "m", true),
        target("p", "a", "other", true),
    ] {
        assert_eq!(
            transform_messages(&source, &cross, None),
            vec![assistant(
                "p",
                "a",
                "m",
                vec![AssistantContentBlock::Text(TextBlock::new("answer"))],
            )]
        );
    }
}

#[test]
fn injected_id_policy_receives_original_assistant_and_rewrites_matching_results() {
    let source_assistant = assistant(
        "p",
        "a",
        "m1",
        vec![
            thinking("reasoning", Some("sig"), false),
            tool_call("old", "lookup"),
        ],
    );
    let mut matched = match tool_blocks(vec![ToolResultContentBlock::Text(TextBlock::new("ok"))]) {
        Message::ToolResult(message) => *message,
        _ => unreachable!(),
    };
    matched.tool_call_id = "old".into();
    matched.details = Some(serde_json::json!({"kept": true}));
    matched.usage = Some(Usage::default());
    matched.added_tool_names = Some(vec!["later".into()]);
    let mut unrelated = matched.clone();
    unrelated.tool_call_id = "unrelated".into();
    let source = vec![
        source_assistant.clone(),
        Message::ToolResult(Box::new(matched.clone())),
        Message::ToolResult(Box::new(unrelated.clone())),
    ];
    let mut observations = Vec::new();
    let mut policy = |id: &str, _target: &TransformTarget, source: &AssistantMessage| {
        observations.push((id.to_owned(), source.clone()));
        format!("normalized-{id}")
    };

    let result = transform_messages(&source, &target("p", "a", "m2", true), Some(&mut policy));

    assert_eq!(
        observations,
        vec![(
            "old".into(),
            match &source_assistant {
                Message::Assistant(message) => message.as_ref().clone(),
                _ => unreachable!(),
            }
        )]
    );
    let transformed_assistant = match &result[0] {
        Message::Assistant(message) => message,
        _ => unreachable!(),
    };
    let transformed_call = match &transformed_assistant.content[1] {
        AssistantContentBlock::ToolCall(call) => call,
        _ => unreachable!(),
    };
    assert_eq!(transformed_call.id, "normalized-old");
    assert!(matches!(
        transformed_assistant.content[0],
        AssistantContentBlock::Text(_)
    ));
    let transformed_matched = match &result[1] {
        Message::ToolResult(message) => message,
        _ => unreachable!(),
    };
    assert_eq!(transformed_matched.tool_call_id, "normalized-old");
    let mut expected_matched = matched;
    expected_matched.tool_call_id = "normalized-old".into();
    assert_eq!(transformed_matched.as_ref(), &expected_matched);
    assert_eq!(result[2], Message::ToolResult(Box::new(unrelated)));
}

#[test]
fn same_model_never_invokes_the_id_policy() {
    let source = vec![
        assistant("p", "a", "m", vec![tool_call("old", "lookup")]),
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "old",
            "lookup",
            Vec::new(),
            false,
            4.0,
        ))),
    ];
    let mut called = false;
    let mut policy = |_id: &str, _target: &TransformTarget, _source: &AssistantMessage| {
        called = true;
        "changed".to_owned()
    };

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", true), Some(&mut policy),),
        source
    );
    assert!(!called);
}

#[test]
fn id_policy_runs_in_transcript_order_and_rewrites_each_matching_result() {
    let source = vec![
        assistant("p", "a", "m1", vec![tool_call("first", "lookup")]),
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "first",
            "lookup",
            Vec::new(),
            false,
            1.0,
        ))),
        assistant("p", "a", "m1", vec![tool_call("second", "fetch")]),
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "second",
            "fetch",
            Vec::new(),
            false,
            2.0,
        ))),
    ];
    let mut calls = Vec::new();
    let mut policy = |id: &str, _target: &TransformTarget, _source: &AssistantMessage| {
        calls.push(id.to_owned());
        format!("normalized-{id}")
    };

    let result = transform_messages(&source, &target("p", "a", "m2", true), Some(&mut policy));

    assert_eq!(calls, ["first", "second"]);
    let result_ids = result
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(message) => Some(message.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(result_ids, ["normalized-first", "normalized-second"]);
}

#[test]
fn error_and_aborted_assistants_are_excluded_without_orphan_results() {
    let source = vec![
        assistant_with_reason(
            "p",
            "a",
            "m",
            vec![tool_call("error-call", "lookup")],
            StopReason::Error,
        ),
        assistant_with_reason(
            "p",
            "a",
            "m",
            vec![tool_call("aborted-call", "lookup")],
            StopReason::Aborted,
        ),
        assistant_with_reason(
            "p",
            "a",
            "m",
            vec![AssistantContentBlock::Text(TextBlock::new("pending stays"))],
            StopReason::Pending,
        ),
    ];

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m", true), None),
        vec![assistant_with_reason(
            "p",
            "a",
            "m",
            vec![AssistantContentBlock::Text(TextBlock::new("pending stays"))],
            StopReason::Pending,
        )]
    );
}

#[test]
fn unresolved_calls_synthesize_ordered_required_results_before_interruptions() {
    let source = vec![
        assistant(
            "p",
            "a",
            "m",
            vec![tool_call("c1", "first"), tool_call("c2", "second")],
        ),
        user_text("next"),
    ];

    let result = transform_messages(&source, &target("p", "a", "m", true), None);
    assert_eq!(result.len(), 4);
    assert_eq!(result[0], source[0]);
    for (message, id, name) in [(&result[1], "c1", "first"), (&result[2], "c2", "second")] {
        let Message::ToolResult(message) = message else {
            panic!("expected synthetic tool result")
        };
        assert_eq!(message.tool_call_id, id);
        assert_eq!(message.tool_name, name);
        assert!(message.is_error);
        assert_eq!(
            message.content,
            vec![ToolResultContentBlock::Text(TextBlock::new(
                "No result provided"
            ))]
        );
    }
    assert_eq!(result[3], source[1]);
}

#[test]
fn resolved_calls_do_not_synthesize_and_normalized_orphans_use_transformed_ids() {
    let source = vec![
        assistant(
            "p",
            "a",
            "m1",
            vec![
                tool_call("resolved", "lookup"),
                tool_call("orphan", "search"),
            ],
        ),
        Message::ToolResult(Box::new(ToolResultMessage::new(
            "resolved",
            "lookup",
            vec![ToolResultContentBlock::Text(TextBlock::new("ok"))],
            false,
            4.0,
        ))),
    ];
    let mut policy = |id: &str, _: &TransformTarget, _: &AssistantMessage| format!("new-{id}");

    let result = transform_messages(&source, &target("p", "a", "m2", true), Some(&mut policy));

    assert_eq!(result.len(), 3);
    let Message::ToolResult(real) = &result[1] else {
        panic!("expected real result")
    };
    assert_eq!(real.tool_call_id, "new-resolved");
    let Message::ToolResult(orphan) = &result[2] else {
        panic!("expected synthetic result")
    };
    assert_eq!(orphan.tool_call_id, "new-orphan");
    assert_eq!(orphan.tool_name, "search");
}

#[cfg(feature = "conformance")]
#[test]
fn legacy_null_content_is_normalized_by_the_library_before_typed_transformation() {
    let usage = serde_json::json!({
        "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
        "total_tokens": 0,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0}
    });
    let legacy = vec![
        serde_json::json!({"role":"user", "content":null, "timestamp":1}),
        serde_json::json!({
            "role":"assistant", "content":null, "provider":"p", "api":"a", "model":"m",
            "usage":usage, "stop_reason":"stop", "timestamp":2
        }),
        serde_json::json!({
            "role":"tool_result", "content":null, "tool_call_id":"c", "tool_name":"lookup",
            "is_error":false, "timestamp":3
        }),
    ];

    let mut empty_assistant = match assistant("p", "a", "m", Vec::new()) {
        Message::Assistant(message) => *message,
        _ => unreachable!(),
    };
    empty_assistant.timestamp = 2.0;
    assert_eq!(
        transform_legacy_messages(&legacy, &target("p", "a", "m", true), None).unwrap(),
        vec![
            user_blocks(Vec::new()),
            Message::Assistant(Box::new(empty_assistant)),
            Message::ToolResult(Box::new(ToolResultMessage::new(
                "c",
                "lookup",
                Vec::new(),
                false,
                3.0,
            ))),
        ]
    );
}

#[test]
fn transforming_assistant_content_preserves_every_unrelated_rich_field() {
    let mut source_message = match assistant(
        "p",
        "a",
        "m1",
        vec![AssistantContentBlock::Text(
            TextBlock::new("answer").with_signature("strip-me"),
        )],
    ) {
        Message::Assistant(message) => *message,
        _ => unreachable!(),
    };
    source_message.response_model = Some("response-model".into());
    source_message.response_id = Some("response-id".into());
    source_message.diagnostics = Some(vec![AssistantMessageDiagnostic {
        diagnostic_type: "provider".into(),
        timestamp: 9.0,
        error: Some(DiagnosticError {
            message: "detail".into(),
            name: Some("ProviderError".into()),
            stack: Some("stack".into()),
            code: Some(DiagnosticCode::String("E1".into())),
        }),
        details: Some(BTreeMap::from([("retry".into(), serde_json::json!(false))])),
    }]);
    source_message.usage.input = 11;
    source_message.deferred = Some(DeferredHandle {
        provider: "p".into(),
        model_id: "m1".into(),
        api: "a".into(),
        id: "deferred-id".into(),
        expires_at: Some(10.0),
        poll_after_ms: Some(20),
        data: Some(serde_json::json!({"key":"value"})),
    });
    source_message.error_message = Some("retained".into());
    source_message.raw_stop_reason = Some("native".into());
    source_message.end_turn = Some(true);
    let source = vec![Message::Assistant(Box::new(source_message.clone()))];
    let mut expected = source_message;
    expected.content = vec![AssistantContentBlock::Text(TextBlock::new("answer"))];

    assert_eq!(
        transform_messages(&source, &target("p", "a", "m2", true), None)[0],
        Message::Assistant(Box::new(expected))
    );
}
