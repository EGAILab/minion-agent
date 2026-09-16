use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use minion_agent::auth::*;

struct NullTransport;

#[async_trait]
impl HttpTransport for NullTransport {
    async fn post(
        &self,
        _request: HttpRequest,
        _signal: Arc<dyn Abortable>,
    ) -> Result<HttpResponse, HttpTransportError> {
        Err(HttpTransportError::Request("unused".into()))
    }
}

struct Interaction {
    signal: Arc<dyn Abortable>,
    answer: String,
    prompts: Arc<Mutex<Vec<AuthPrompt>>>,
}

impl AuthInteraction for Interaction {
    fn signal(&self) -> Option<Arc<dyn Abortable>> {
        Some(self.signal.clone())
    }

    fn prompt(&self, prompt: AuthPrompt) -> InteractionFuture<String> {
        self.prompts.lock().unwrap().push(prompt);
        let answer = self.answer.clone();
        Box::pin(async move { Ok(answer) })
    }

    fn notify(&self, _event: AuthEvent) -> Result<(), AuthInteractionError> {
        Ok(())
    }
}

impl ProviderAuthInteraction for Interaction {}

#[test]
fn auth_vocabulary_fields_are_mutable_and_subscription_is_three_valued() {
    let oauth = OpenAiCodexOAuth::new(Arc::new(NullTransport));
    let mut method = oauth.as_oauth_auth();
    assert_eq!(method.is_subscription, Some(true));
    method.name = "renamed".into();
    method.is_subscription = Some(false);
    assert_eq!(method.is_subscription, Some(false));
    method.is_subscription = None;
    assert_eq!(method.is_subscription, None);

    let mut option = AuthPromptOption {
        id: "first".into(),
        label: "First".into(),
        description: None,
    };
    option.id = "second".into();
    option.description = Some("mutable".into());
    assert_eq!(option.id, "second");
}

#[test]
fn provider_auth_enforces_at_least_one_method() {
    assert_eq!(
        ProviderAuth::new(None, None).err(),
        Some(EmptyProviderAuthError)
    );
    let oauth = OpenAiCodexOAuth::new(Arc::new(NullTransport)).as_oauth_auth();
    let mut provider = ProviderAuth::new(None, Some(oauth)).unwrap();
    assert!(provider.oauth().is_some());
    provider.oauth = None;
    assert!(provider.oauth().is_none());
}

#[tokio::test]
async fn provider_interaction_is_an_auth_interaction_and_select_has_no_signal() {
    fn accepts_auth_interaction<T: AuthInteraction + ?Sized>(_value: &T) {}

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let interaction = Arc::new(Interaction {
        signal: AuthAbortController::default().signal(),
        answer: "unsupported".into(),
        prompts: prompts.clone(),
    });
    accepts_auth_interaction(interaction.as_ref());
    let provider: Arc<dyn ProviderAuthInteraction> = interaction;
    let error = OpenAiCodexOAuth::new(Arc::new(NullTransport))
        .login(provider)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Unknown OpenAI Codex login method: unsupported"
    );
    let AuthPrompt::Select(prompt) = &prompts.lock().unwrap()[0] else {
        panic!("expected select prompt")
    };
    assert!(prompt.signal.is_none());
    assert_eq!(
        prompt
            .options
            .iter()
            .map(|option| option.id.as_str())
            .collect::<Vec<_>>(),
        [LOGIN_METHOD_BROWSER, LOGIN_METHOD_DEVICE_CODE]
    );
}

#[test]
fn codex_oauth_composite_has_the_approved_public_identity() {
    let method = OpenAiCodexOAuth::new(Arc::new(NullTransport)).as_oauth_auth();
    assert_eq!(method.name, "OpenAI (ChatGPT Plus/Pro)");
    assert_eq!(method.is_subscription, Some(true));
    assert_eq!(method.login_label, None);
}
