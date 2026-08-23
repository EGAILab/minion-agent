use std::sync::Arc;

use futures::{StreamExt, stream};
use minion_agent::llm::{
    AdapterStartError, AdapterStreamError, AdapterStreamErrorKind, AssistantContentBlock,
    AssistantMessage, DoneReason, ErrorReason, LlmAdapter, LlmContext, LlmRequest, LlmService,
    ModelIdentity, RawAssistantStream, StopReason, StreamChunk, StreamOptions, TextBlock,
};

struct ItemsAdapter(Vec<Result<StreamChunk, AdapterStreamError>>);

impl LlmAdapter for ItemsAdapter {
    fn start(&self, _: LlmRequest) -> Result<RawAssistantStream, AdapterStartError> {
        Ok(Box::pin(stream::iter(self.0.clone())))
    }
}

fn identity() -> ModelIdentity {
    ModelIdentity::new("openai", "responses", "gpt-5").unwrap()
}

fn request() -> LlmRequest {
    LlmRequest {
        model: identity(),
        context: LlmContext::default(),
        options: StreamOptions::default(),
    }
}

fn partial(text: &str) -> AssistantMessage {
    let mut message = AssistantMessage::pending(identity(), 7.0);
    message
        .content
        .push(AssistantContentBlock::Text(TextBlock::new(text)));
    message
}

fn service(items: Vec<Result<StreamChunk, AdapterStreamError>>) -> LlmService {
    let service = LlmService::new();
    service.register(identity(), Arc::new(ItemsAdapter(items)));
    service
}

#[test]
fn normal_terminal_fuses_without_exposing_post_terminal_items() {
    futures::executor::block_on(async {
        let prefix = partial("hel");
        let mut final_message = partial("hello");
        final_message.stop_reason = StopReason::Stop;
        let extra = partial("must not appear");
        let service = service(vec![
            Ok(StreamChunk::TextDelta {
                content_index: 0,
                delta: "hel".into(),
                partial: prefix.clone(),
            }),
            Ok(StreamChunk::Done {
                reason: DoneReason::Stop,
                message: final_message.clone(),
            }),
            Ok(StreamChunk::TextDelta {
                content_index: 0,
                delta: "bad".into(),
                partial: extra,
            }),
        ]);

        let mut public = service.stream(request()).unwrap();
        assert_eq!(public.next().await.unwrap().partial(), &prefix);
        assert_eq!(public.next().await.unwrap().partial(), &final_message);
        assert!(public.next().await.is_none());
        assert!(public.next().await.is_none());
    });
}

#[test]
fn adapter_error_settles_in_band_and_preserves_partial() {
    futures::executor::block_on(async {
        let observed = partial("partial answer");
        let service = service(vec![
            Ok(StreamChunk::TextDelta {
                content_index: 0,
                delta: "partial answer".into(),
                partial: observed.clone(),
            }),
            Err(AdapterStreamError::new(
                AdapterStreamErrorKind::Network,
                "connection reset",
            )),
        ]);

        let mut public = service.stream(request()).unwrap();
        public.next().await.unwrap();
        let terminal = public.next().await.unwrap();
        assert!(matches!(
            terminal,
            StreamChunk::Error {
                reason: ErrorReason::Error,
                ..
            }
        ));
        assert_eq!(terminal.partial().content, observed.content);
        assert_eq!(terminal.partial().stop_reason, StopReason::Error);
        assert_eq!(
            terminal.partial().error_message.as_deref(),
            Some("connection reset")
        );
        assert!(public.next().await.is_none());
    });
}

#[test]
fn cancellation_settles_aborted() {
    futures::executor::block_on(async {
        let service = service(vec![Err(AdapterStreamError::new(
            AdapterStreamErrorKind::Cancelled,
            "cancelled",
        ))]);
        let terminal = service.stream(request()).unwrap().next().await.unwrap();
        assert!(matches!(
            terminal,
            StreamChunk::Error {
                reason: ErrorReason::Aborted,
                ..
            }
        ));
        assert_eq!(terminal.partial().stop_reason, StopReason::Aborted);
    });
}

#[test]
fn premature_eof_synthesizes_error_terminal_preserving_partial() {
    futures::executor::block_on(async {
        let observed = partial("unfinished");
        let service = service(vec![Ok(StreamChunk::TextDelta {
            content_index: 0,
            delta: "unfinished".into(),
            partial: observed.clone(),
        })]);
        let mut public = service.stream(request()).unwrap();
        public.next().await.unwrap();
        let terminal = public.next().await.unwrap();
        assert!(matches!(
            terminal,
            StreamChunk::Error {
                reason: ErrorReason::Error,
                ..
            }
        ));
        assert_eq!(terminal.partial().content, observed.content);
        assert_eq!(terminal.partial().usage, observed.usage);
        assert_eq!(terminal.partial().stop_reason, StopReason::Error);
        assert!(
            terminal
                .partial()
                .error_message
                .as_deref()
                .is_some_and(|message| !message.is_empty())
        );
        assert!(public.next().await.is_none());
    });
}
