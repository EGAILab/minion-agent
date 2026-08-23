use std::{
    pin::Pin,
    sync::{Arc, Mutex},
};

use futures::{Stream, stream};
use minion_agent::llm::{
    AdapterStartError, AdapterStreamError, LlmAdapter, LlmContext, LlmRequest, LlmService,
    LlmStartError, ModelIdentity, RawAssistantStream, Script, ScriptItem, ScriptedAdapter,
    StreamOptions,
};

#[derive(Clone)]
struct RecordingAdapter {
    requests: Arc<Mutex<Vec<LlmRequest>>>,
    reject: bool,
}

impl LlmAdapter for RecordingAdapter {
    fn start(&self, request: LlmRequest) -> Result<RawAssistantStream, AdapterStartError> {
        self.requests.lock().unwrap().push(request);
        if self.reject {
            return Err(AdapterStartError::Rejected(
                "invalid provider configuration".into(),
            ));
        }
        let raw: Pin<Box<dyn Stream<Item = Result<_, AdapterStreamError>> + Send>> =
            Box::pin(stream::empty());
        Ok(raw)
    }
}

fn request(identity: ModelIdentity) -> LlmRequest {
    LlmRequest {
        model: identity,
        context: LlmContext::default(),
        options: StreamOptions::default(),
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
fn adapter_start_failure_remains_eager_and_typed() {
    let identity = ModelIdentity::new("openai", "responses", "gpt-5").unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let service = LlmService::new();
    service.register(
        identity.clone(),
        Arc::new(RecordingAdapter {
            requests: requests.clone(),
            reject: true,
        }),
    );

    let original = request(identity);
    assert!(matches!(
        service.stream(original.clone()),
        Err(LlmStartError::AdapterStart(_))
    ));
    assert_eq!(*requests.lock().unwrap(), vec![original]);
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
fn exhausted_scripted_adapter_fails_before_stream_creation() {
    let identity = ModelIdentity::new("openai", "responses", "gpt-5").unwrap();
    let adapter = Arc::new(ScriptedAdapter::new([]));
    let service = LlmService::new();
    service.register(identity.clone(), adapter);
    assert!(matches!(
        service.stream(request(identity)),
        Err(LlmStartError::AdapterStart(_))
    ));
}
