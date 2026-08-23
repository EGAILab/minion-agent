use std::{path::PathBuf, sync::Arc};

use futures::StreamExt;
use minion_agent::llm::{
    AssistantContentBlock, AssistantMessage, DoneReason, ErrorReason, LlmContext, LlmRequest,
    LlmService, ModelIdentity, Script, ScriptItem, ScriptedAdapter, SimpleStreamOptions,
    StopReason, StreamChunk, TextBlock,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Scenario {
    provider_script: Vec<Response>,
    #[serde(default)]
    config: Config,
    #[serde(default)]
    expect_assistant_stop_reasons: Vec<StopReason>,
    #[serde(default)]
    expect_messages: Vec<ExpectedMessage>,
}

#[derive(Default, Deserialize)]
struct Config {
    model: Option<String>,
}

#[derive(Deserialize)]
struct Response {
    content: Vec<Content>,
    stop_reason: StopReason,
    #[serde(default)]
    error_message: Option<String>,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    chunks_after_terminal: usize,
}

#[derive(Deserialize)]
struct Content {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

#[derive(Deserialize)]
struct ExpectedMessage {
    role: String,
    text: String,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent")
}

fn identity(model: &str) -> ModelIdentity {
    ModelIdentity::new("mock", "mock", model).unwrap()
}

fn partial(model: &ModelIdentity, response: &Response) -> AssistantMessage {
    let mut message = AssistantMessage::pending(model.clone(), 0.0);
    message.content = response
        .content
        .iter()
        .map(|content| {
            assert_eq!(
                content.kind, "text",
                "this partial Layer-02 runner accepts only text fixtures"
            );
            AssistantContentBlock::Text(TextBlock::new(&content.text))
        })
        .collect();
    message
}

fn script(model: &ModelIdentity, response: &Response) -> Script {
    let partial = partial(model, response);
    let mut items = Vec::new();
    if !response.content.is_empty() {
        items.push(ScriptItem::Chunk(Box::new(StreamChunk::TextDelta {
            content_index: 0,
            delta: response.content[0].text.clone(),
            partial: partial.clone(),
        })));
    }
    if !response.truncated {
        let terminal = match response.stop_reason {
            StopReason::Error | StopReason::Aborted => {
                let mut error = partial.clone();
                error.stop_reason = response.stop_reason;
                error.error_message = response.error_message.clone();
                StreamChunk::Error {
                    reason: if response.stop_reason == StopReason::Aborted {
                        ErrorReason::Aborted
                    } else {
                        ErrorReason::Error
                    },
                    error,
                }
            }
            reason => {
                let mut message = partial;
                let reason = match reason {
                    StopReason::Stop => DoneReason::Stop,
                    StopReason::Length => DoneReason::Length,
                    StopReason::ToolUse => DoneReason::ToolUse,
                    StopReason::Deferred => DoneReason::Deferred,
                    StopReason::Pending | StopReason::Error | StopReason::Aborted => unreachable!(),
                };
                message.stop_reason = response.stop_reason;
                StreamChunk::Done { reason, message }
            }
        };
        items.push(ScriptItem::Chunk(Box::new(terminal)));
        for _ in 0..response.chunks_after_terminal {
            items.push(ScriptItem::Chunk(Box::new(StreamChunk::TextDelta {
                content_index: 0,
                delta: "ignored".into(),
                partial: AssistantMessage::pending(model.clone(), 0.0),
            })));
        }
    }
    Script::new(items)
}

#[test]
fn layer_02_stream_contract_cases_drive_the_real_typed_rust_seam() {
    futures::executor::block_on(async {
        for file in [
            "premature-eof-synthesizes-error-terminal.yaml",
            "premature-eof-preserves-partial-message.yaml",
            "public-stream-fuses-after-first-terminal.yaml",
            "represented-provider-error-rides-stream.yaml",
        ] {
            let scenario: Scenario =
                serde_yaml::from_str(&std::fs::read_to_string(root().join(file)).unwrap()).unwrap();
            let model = identity("model");
            let adapter = Arc::new(ScriptedAdapter::new(
                scenario
                    .provider_script
                    .iter()
                    .map(|response| script(&model, response)),
            ));
            let service = LlmService::new();
            service.register(model.clone(), adapter);
            let mut stream = service
                .stream(LlmRequest {
                    model,
                    context: LlmContext::default(),
                    options: SimpleStreamOptions::default(),
                })
                .unwrap();
            let mut terminal = None;
            while let Some(chunk) = stream.next().await {
                if chunk.is_terminal() {
                    terminal = Some(chunk);
                }
            }
            let terminal = terminal.expect("canonical provider script must settle");
            assert_eq!(
                scenario.expect_assistant_stop_reasons,
                vec![terminal.partial().stop_reason],
                "{file}"
            );
            if let Some(expected) = scenario
                .expect_messages
                .iter()
                .find(|message| message.role == "assistant")
            {
                assert!(
                    serde_json::to_value(&terminal)
                        .unwrap()
                        .to_string()
                        .contains(&expected.text),
                    "{file}"
                );
            }
        }
    });
}

#[test]
fn eager_invalid_model_case_uses_real_service_lookup() {
    let scenario: Scenario = serde_yaml::from_str(
        &std::fs::read_to_string(root().join("eager-invalid-model-fails-before-stream.yaml"))
            .unwrap(),
    )
    .unwrap();
    let service = LlmService::new();
    let requested = scenario
        .config
        .model
        .expect("canonical case names a missing model");
    assert!(
        service
            .stream(LlmRequest {
                model: identity(&requested),
                context: LlmContext::default(),
                options: SimpleStreamOptions::default()
            })
            .is_err()
    );
}
