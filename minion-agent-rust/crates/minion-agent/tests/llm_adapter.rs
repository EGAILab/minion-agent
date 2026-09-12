use std::{
    pin::Pin,
    sync::{Arc, Mutex},
};

use futures::{Stream, stream};
use minion_agent::llm::{
    AdapterStreamError, AdapterStreamErrorKind, LlmAdapter, LlmContext, LlmRequest, LlmService,
    LlmStartError, ModelIdentity, RawAssistantStream, Script, ScriptItem, ScriptedAdapter,
    SimpleStreamOptions, StopReason,
};

#[derive(Clone)]
struct RecordingAdapter {
    requests: Arc<Mutex<Vec<LlmRequest>>>,
}

impl LlmAdapter for RecordingAdapter {
    fn start(&self, request: LlmRequest) -> RawAssistantStream {
        self.requests.lock().unwrap().push(request);
        let raw: Pin<Box<dyn Stream<Item = Result<_, AdapterStreamError>> + Send>> =
            Box::pin(stream::empty());
        raw
    }
}

struct RejectingAdapter;

impl LlmAdapter for RejectingAdapter {
    fn start(&self, _: LlmRequest) -> RawAssistantStream {
        Box::pin(stream::once(async {
            Err(AdapterStreamError::new(
                AdapterStreamErrorKind::Provider,
                "invalid provider configuration",
            ))
        }))
    }
}

fn request(identity: ModelIdentity) -> LlmRequest {
    LlmRequest {
        model: identity,
        context: LlmContext::default(),
        options: SimpleStreamOptions::default(),
        signal: None,
    }
}

#[test]
fn unknown_model_fails_before_adapter_stream_creation() {
    let service = LlmService::new();
    let identity = ModelIdentity::new("openai", "responses", "missing").unwrap();
    let result = service.stream(request(identity.clone()));
    assert!(matches!(result, Err(LlmStartError::UnknownModel { model }) if model == identity));
}

#[test]
fn resolved_adapter_detection_failure_settles_in_band() {
    futures::executor::block_on(async {
        use futures::StreamExt;

        let identity = ModelIdentity::new("openai", "responses", "gpt-5").unwrap();
        let service = LlmService::new();
        service.register(identity.clone(), Arc::new(RejectingAdapter));

        let terminal = service
            .stream(request(identity))
            .expect("resolved adapter invocation returns a stream")
            .next()
            .await
            .expect("expected adapter failure settles terminally");
        assert_eq!(terminal.partial().stop_reason, StopReason::Error);
        assert_eq!(
            terminal.partial().error_message.as_deref(),
            Some("invalid provider configuration")
        );
    });
}

#[test]
fn scripted_adapter_records_requests_and_only_emits_raw_script_items() {
    futures::executor::block_on(async {
        use futures::StreamExt;

        let identity = ModelIdentity::new("openai", "responses", "gpt-5").unwrap();
        let adapter = Arc::new(ScriptedAdapter::new([Script::new([ScriptItem::Error(
            AdapterStreamError::new(
                minion_agent::llm::AdapterStreamErrorKind::Network,
                "offline",
            ),
        )])]));
        let service = LlmService::new();
        service.register(identity.clone(), adapter.clone());
        let original = request(identity);

        let terminal = service
            .stream(original.clone())
            .unwrap()
            .next()
            .await
            .unwrap();
        assert_eq!(terminal.partial().error_message.as_deref(), Some("offline"));
        assert_eq!(adapter.requests(), vec![original]);
    });
}

#[test]
fn exhausted_scripted_adapter_settles_in_band() {
    futures::executor::block_on(async {
        use futures::StreamExt;

        let identity = ModelIdentity::new("openai", "responses", "gpt-5").unwrap();
        let adapter = Arc::new(ScriptedAdapter::new([]));
        let service = LlmService::new();
        service.register(identity.clone(), adapter);
        let terminal = service
            .stream(request(identity))
            .unwrap()
            .next()
            .await
            .unwrap();
        assert_eq!(terminal.partial().stop_reason, StopReason::Error);
        assert_eq!(
            terminal.partial().error_message.as_deref(),
            Some("scripted adapter has no remaining script")
        );
    });
}

#[test]
fn registration_handles_are_repeatable_stale_safe_and_owned_per_call() {
    let identity = ModelIdentity::new("mock", "mock", "alpha").unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let adapter: Arc<dyn LlmAdapter> = Arc::new(RecordingAdapter {
        requests: requests.clone(),
    });
    let service = LlmService::new();

    let first = service.register(identity.clone(), adapter.clone());
    let second = service.register(identity.clone(), adapter);
    assert_eq!(service.models(), vec![identity.clone()]);

    first.withdraw();
    assert_eq!(service.models(), vec![identity.clone()]);
    first.withdraw();
    assert_eq!(service.models(), vec![identity.clone()]);

    second.withdraw();
    assert!(service.models().is_empty());
    second.withdraw();
    assert!(service.models().is_empty());
}

#[test]
fn one_registration_handle_owns_every_identity_in_that_call() {
    let alpha = ModelIdentity::new("mock", "mock", "alpha").unwrap();
    let beta = ModelIdentity::new("mock", "mock", "beta").unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let adapter: Arc<dyn LlmAdapter> = Arc::new(RecordingAdapter { requests });
    let service = LlmService::new();

    let handle = service.register_models(vec![beta.clone(), alpha.clone()], adapter);
    assert_eq!(service.models(), vec![alpha, beta]);
    handle.withdraw();
    assert!(service.models().is_empty());
}

#[test]
fn multi_identity_withdrawal_removes_only_entries_the_call_still_owns() {
    let alpha = ModelIdentity::new("mock", "mock", "alpha").unwrap();
    let beta = ModelIdentity::new("mock", "mock", "beta").unwrap();
    let first_adapter: Arc<dyn LlmAdapter> = Arc::new(RecordingAdapter {
        requests: Arc::new(Mutex::new(Vec::new())),
    });
    let replacement: Arc<dyn LlmAdapter> = Arc::new(RecordingAdapter {
        requests: Arc::new(Mutex::new(Vec::new())),
    });
    let service = LlmService::new();

    let first = service.register_models([alpha.clone(), beta.clone()], first_adapter);
    let beta_replacement = service.register(beta.clone(), replacement);
    first.withdraw();

    assert_eq!(service.models(), vec![beta.clone()]);
    beta_replacement.withdraw();
    assert!(service.models().is_empty());
}
