use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use serde_json::Value;
use serde_json::json;
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{Mutex, oneshot},
    task::JoinHandle,
};
use url::{Url, form_urlencoded};

use super::{
    Abortable, AuthAbortController, AuthEvent, AuthEventDeviceCode, AuthEventUrl,
    AuthInteractionError, AuthMethodError, AuthPrompt, AuthPromptManualCode, AuthPromptOption,
    AuthPromptSelect, CodexProjectionError, DeviceClock, DeviceFlowError, DeviceFlowOptions,
    DevicePollResult, HttpRequest, HttpResponse, HttpTransport, HttpTransportError, JsJsonError,
    JsJsonValue, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction, ReqwestTransport,
    SystemDeviceClock, codex_to_auth, credentials_from_token, generate_pkce, js_json_loads,
    js_json_stringify, js_trim, poll_device_code_flow,
};

pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const AUTH_BASE_URL: &str = "https://auth.openai.com";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub const DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
pub const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
pub const DEVICE_CODE_TIMEOUT_SECONDS: f64 = 900.0;
pub const SCOPE: &str = "openid profile email offline_access";
pub const LOGIN_METHOD_BROWSER: &str = "browser";
pub const LOGIN_METHOD_DEVICE_CODE: &str = "device_code";
pub const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";

#[derive(Debug, Error)]
pub enum CodexOAuthError {
    #[error(transparent)]
    Interaction(#[from] AuthInteractionError),
    #[error(transparent)]
    Transport(#[from] HttpTransportError),
    #[error(transparent)]
    Json(#[from] JsJsonError),
    #[error(transparent)]
    Projection(#[from] CodexProjectionError),
    #[error("{0}")]
    DeviceFlow(#[from] DeviceFlowError),
    #[error("{0}")]
    Message(String),
}

impl From<CodexOAuthError> for AuthMethodError {
    fn from(value: CodexOAuthError) -> Self {
        Self::new(value.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedAuthorizationInput {
    pub code: Option<String>,
    pub state: Option<String>,
}

pub fn parse_authorization_input(input: &str) -> ParsedAuthorizationInput {
    let value = js_trim(input);
    if value.is_empty() {
        return ParsedAuthorizationInput {
            code: None,
            state: None,
        };
    }
    if let Ok(url) = Url::parse(value) {
        return query_result(url.query_pairs());
    }
    if value.contains('#') {
        let mut parts = value.splitn(3, '#');
        return ParsedAuthorizationInput {
            code: parts.next().map(ToOwned::to_owned),
            state: parts.next().map(ToOwned::to_owned),
        };
    }
    if value.contains("code=") {
        let query = value.strip_prefix('?').unwrap_or(value);
        return query_result(form_urlencoded::parse(query.as_bytes()));
    }
    ParsedAuthorizationInput {
        code: Some(value.to_owned()),
        state: None,
    }
}

fn query_result<'a>(
    pairs: impl Iterator<Item = (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>)>,
) -> ParsedAuthorizationInput {
    let mut code = None;
    let mut state = None;
    for (key, value) in pairs {
        match key.as_ref() {
            "code" if code.is_none() => code = Some(value.into_owned()),
            "state" if state.is_none() => state = Some(value.into_owned()),
            _ => {}
        }
    }
    ParsedAuthorizationInput { code, state }
}

#[derive(Clone)]
pub struct OpenAiCodexOAuth {
    transport: Arc<dyn HttpTransport>,
}

impl OpenAiCodexOAuth {
    pub fn new(transport: Arc<dyn HttpTransport>) -> Self {
        Self { transport }
    }

    pub fn production() -> Result<Self, CodexOAuthError> {
        Ok(Self::new(Arc::new(ReqwestTransport::new()?)))
    }

    pub async fn login(
        &self,
        interaction: Arc<dyn ProviderAuthInteraction>,
    ) -> Result<OAuthCredential, CodexOAuthError> {
        let method = interaction
            .prompt(AuthPrompt::Select(AuthPromptSelect {
                message: "Select OpenAI Codex login method:".into(),
                options: Arc::from([
                    AuthPromptOption {
                        id: LOGIN_METHOD_BROWSER.into(),
                        label: "Browser login (default)".into(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: LOGIN_METHOD_DEVICE_CODE.into(),
                        label: "Device code login (headless)".into(),
                        description: None,
                    },
                ]),
                signal: None,
            }))
            .await?;
        match method.as_str() {
            LOGIN_METHOD_BROWSER => self.login_browser(interaction, CALLBACK_PORT).await,
            LOGIN_METHOD_DEVICE_CODE => {
                self.login_device_code(interaction, &SystemDeviceClock::default())
                    .await
            }
            _ => Err(CodexOAuthError::Message(format!(
                "Unknown OpenAI Codex login method: {method}"
            ))),
        }
    }

    pub async fn refresh(
        &self,
        credential: OAuthCredential,
        signal: Arc<dyn Abortable>,
    ) -> Result<OAuthCredential, CodexOAuthError> {
        let token = self
            .refresh_access_token(&credential.refresh(), signal)
            .await?;
        Ok(credentials_from_token(
            token.access,
            token.refresh,
            token.expires,
        )?)
    }

    pub async fn to_auth(&self, credential: OAuthCredential) -> Result<ModelAuth, CodexOAuthError> {
        Ok(codex_to_auth(&credential))
    }

    pub fn as_oauth_auth(&self) -> OAuthAuth {
        let login = self.clone();
        let refresh = self.clone();
        let to_auth = self.clone();
        OAuthAuth {
            name: "OpenAI (ChatGPT Plus/Pro)".into(),
            login: Arc::new(move |interaction| {
                let this = login.clone();
                Box::pin(async move { this.login(interaction).await.map_err(Into::into) })
            }),
            refresh: Arc::new(move |credential, signal| {
                let this = refresh.clone();
                Box::pin(async move { this.refresh(credential, signal).await.map_err(Into::into) })
            }),
            to_auth: Arc::new(move |credential| {
                let this = to_auth.clone();
                Box::pin(async move { this.to_auth(credential).await.map_err(Into::into) })
            }),
            is_subscription: Some(true),
            login_label: None,
        }
    }

    async fn login_device_code(
        &self,
        interaction: Arc<dyn ProviderAuthInteraction>,
        clock: &dyn DeviceClock,
    ) -> Result<OAuthCredential, CodexOAuthError> {
        let signal = interaction.provider_signal();
        let device = self.start_device_auth(signal.clone()).await?;
        interaction.notify(AuthEvent::DeviceCode(AuthEventDeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: DEVICE_VERIFICATION_URI.into(),
            interval_seconds: Some(device.interval_seconds),
            expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS),
        }))?;
        let result = self
            .poll_device_auth(&device, signal.clone(), clock)
            .await?;
        self.exchange_for_credentials(
            &result.authorization_code,
            &result.code_verifier,
            DEVICE_REDIRECT_URI,
            signal,
        )
        .await
    }

    async fn start_device_auth(
        &self,
        signal: Arc<dyn Abortable>,
    ) -> Result<DeviceAuthInfo, CodexOAuthError> {
        let response = self
            .fetch_login(
                HttpRequest {
                    url: DEVICE_USER_CODE_URL.into(),
                    headers: content_type("application/json"),
                    body: serde_json::to_vec(&json!({ "client_id": CLIENT_ID }))
                        .expect("static JSON is serializable"),
                },
                signal.clone(),
            )
            .await?;
        if !response.is_success() {
            if response.status == 404 {
                response.discard().await;
                return Err(CodexOAuthError::Message(
                    "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL.".into(),
                ));
            }
            let body = text_or_empty(&response, signal).await?;
            let suffix = if body.is_empty() {
                String::new()
            } else {
                format!(": {body}")
            };
            return Err(CodexOAuthError::Message(format!(
                "OpenAI Codex device code request failed with status {}{suffix}",
                response.status
            )));
        }
        let parsed = js_json_loads(&response.text(signal).await?)?;
        let device_auth_id = nonempty_string(&parsed, "device_auth_id");
        let user_code = nonempty_string(&parsed, "user_code");
        let interval = parsed.get("interval").and_then(coerce_interval);
        let Some((device_auth_id, user_code, interval_seconds)) = device_auth_id
            .zip(user_code)
            .zip(interval)
            .map(|((a, b), c)| (a, b, c))
        else {
            return Err(CodexOAuthError::Message(format!(
                "Invalid OpenAI Codex device code response: {}",
                js_json_stringify(&parsed)
            )));
        };
        Ok(DeviceAuthInfo {
            device_auth_id,
            user_code,
            interval_seconds,
        })
    }

    async fn poll_device_auth(
        &self,
        device: &DeviceAuthInfo,
        signal: Arc<dyn Abortable>,
        clock: &dyn DeviceClock,
    ) -> Result<DeviceTokenSuccess, CodexOAuthError> {
        let transport = self.transport.clone();
        let interval_seconds = device.interval_seconds;
        let device = device.clone();
        let poll_signal = signal.clone();
        poll_device_code_flow(
            move || {
                let transport = transport.clone();
                let device = device.clone();
                let signal = poll_signal.clone();
                async move {
                    match poll_device_once(transport, &device, signal).await {
                        Ok(DevicePollResult::Pending) => DevicePollResult::Pending,
                        Ok(DevicePollResult::SlowDown { interval_seconds }) => {
                            DevicePollResult::SlowDown { interval_seconds }
                        }
                        Ok(DevicePollResult::Failed { message }) => {
                            DevicePollResult::Failed { message }
                        }
                        Ok(DevicePollResult::Complete(value)) => {
                            DevicePollResult::Complete(Ok(value))
                        }
                        Err(error) => DevicePollResult::Complete(Err(error)),
                    }
                }
            },
            DeviceFlowOptions {
                interval_seconds: Some(interval_seconds),
                expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS),
                wait_before_first_poll: false,
                signal: Some(signal),
            },
            clock,
        )
        .await?
    }

    async fn exchange_for_credentials(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        signal: Arc<dyn Abortable>,
    ) -> Result<OAuthCredential, CodexOAuthError> {
        let token = self
            .exchange_authorization_code(code, verifier, redirect_uri, signal)
            .await?;
        Ok(credentials_from_token(
            token.access,
            token.refresh,
            token.expires,
        )?)
    }

    async fn exchange_authorization_code(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        signal: Arc<dyn Abortable>,
    ) -> Result<OAuthToken, CodexOAuthError> {
        let body = form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "authorization_code")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("code", code)
            .append_pair("code_verifier", verifier)
            .append_pair("redirect_uri", redirect_uri)
            .finish();
        let response = self
            .fetch_login(
                HttpRequest {
                    url: TOKEN_URL.into(),
                    headers: content_type("application/x-www-form-urlencoded"),
                    body: body.into_bytes(),
                },
                signal.clone(),
            )
            .await?;
        read_token_response(&response, "exchange", signal).await
    }

    async fn refresh_access_token(
        &self,
        refresh_token: &str,
        signal: Arc<dyn Abortable>,
    ) -> Result<OAuthToken, CodexOAuthError> {
        let body = form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "refresh_token")
            .append_pair("refresh_token", refresh_token)
            .append_pair("client_id", CLIENT_ID)
            .finish();
        let response = self
            .transport
            .post(
                HttpRequest {
                    url: TOKEN_URL.into(),
                    headers: content_type("application/x-www-form-urlencoded"),
                    body: body.into_bytes(),
                },
                signal.clone(),
            )
            .await
            .map_err(|error| {
                CodexOAuthError::Message(format!("OpenAI Codex token refresh error: {error}"))
            })?;
        read_token_response(&response, "refresh", signal).await
    }

    async fn fetch_login(
        &self,
        request: HttpRequest,
        signal: Arc<dyn Abortable>,
    ) -> Result<HttpResponse, CodexOAuthError> {
        match self.transport.post(request, signal.clone()).await {
            Ok(response) => Ok(response),
            Err(_) if signal.aborted() => Err(CodexOAuthError::Message("Login cancelled".into())),
            Err(error) => Err(error.into()),
        }
    }

    async fn login_browser(
        &self,
        interaction: Arc<dyn ProviderAuthInteraction>,
        port: u16,
    ) -> Result<OAuthCredential, CodexOAuthError> {
        let pkce = generate_pkce().map_err(|error| CodexOAuthError::Message(error.to_string()))?;
        let state = random_state()?;
        let authorization_url = authorization_url(&pkce.challenge, &state);
        let host = callback_host();
        let mut server = CallbackServer::start(&host, port, state.clone()).await;
        let flow_signal = interaction.provider_signal();
        let server_abort = server.as_ref().map(CallbackServer::abort_controller);
        let watcher = server_abort.as_ref().map(|abort| {
            let abort = abort.clone();
            let signal = flow_signal.clone();
            tokio::spawn(async move {
                while !signal.aborted() {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                abort.abort();
            })
        });
        if flow_signal.aborted()
            && let Some(abort) = &server_abort
        {
            abort.abort();
        }

        // Deliberately before the cleanup boundary, matching pinned Pi.
        interaction.notify(AuthEvent::AuthUrl(AuthEventUrl {
            url: authorization_url,
            instructions: Some("A browser window should open. Complete login to finish.".into()),
        }))?;

        let manual_abort = AuthAbortController::default();
        let prompt = interaction.prompt(AuthPrompt::ManualCode(AuthPromptManualCode {
            message: "Complete login in your browser, or paste the authorization code / redirect URL here:".into(),
            placeholder: Some(REDIRECT_URI.into()),
            signal: Some(manual_abort.signal()),
        }));
        tokio::pin!(prompt);

        let result: Result<ParsedAuthorizationInput, AuthInteractionError> =
            if let Some(server_ref) = server.as_mut() {
                tokio::select! {
                    server_code = server_ref.wait() => {
                        match server_code {
                            Some(code) => {
                                manual_abort.abort();
                                Ok(ParsedAuthorizationInput { code: Some(code), state: None })
                            }
                            None => prompt.await.map(|value| parse_authorization_input(&value)),
                        }
                    }
                    manual = &mut prompt => {
                        server_ref.abort();
                        manual.map(|value| parse_authorization_input(&value))
                    }
                }
            } else {
                prompt.await.map(|value| parse_authorization_input(&value))
            };

        manual_abort.abort();
        if let Some(server) = server.as_mut() {
            server.close().await;
        }
        if let Some(watcher) = watcher {
            watcher.abort();
        }
        let result = result?;

        if result
            .state
            .as_deref()
            .is_some_and(|value| !value.is_empty() && value != state)
        {
            return Err(CodexOAuthError::Message("State mismatch".into()));
        }
        let code = result
            .code
            .filter(|value| !value.is_empty())
            .ok_or_else(|| CodexOAuthError::Message("Missing authorization code".into()))?;
        self.exchange_for_credentials(&code, &pkce.verifier, REDIRECT_URI, flow_signal)
            .await
    }
}

fn callback_host() -> String {
    match std::env::var("PI_OAUTH_CALLBACK_HOST") {
        Ok(value) if !value.is_empty() => value,
        _ => "127.0.0.1".into(),
    }
}

fn random_state() -> Result<String, CodexOAuthError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes).map_err(|error| CodexOAuthError::Message(error.to_string()))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn authorization_url(challenge: &str, state: &str) -> String {
    let mut url = Url::parse(AUTHORIZE_URL).expect("fixed authorization URL is valid");
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", "pi");
    url.into()
}

fn content_type(value: &str) -> BTreeMap<String, String> {
    BTreeMap::from([("Content-Type".into(), value.into())])
}

fn nonempty_string(value: &JsJsonValue, key: &str) -> Option<String> {
    value
        .get(key)?
        .as_string()
        .filter(|value| !value.is_empty())
}

fn coerce_interval(value: &JsJsonValue) -> Option<f64> {
    let value = match value {
        JsJsonValue::Number(value) => *value,
        JsJsonValue::String(value) => js_number_coerce(&value.to_string()?),
        _ => return None,
    };
    (value.is_finite() && value >= 0.0).then_some(value)
}

fn js_number_coerce(value: &str) -> f64 {
    let value = js_trim(value);
    if value.is_empty() {
        return 0.0;
    }
    match value {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    for (prefixes, radix) in [(["0x", "0X"], 16), (["0o", "0O"], 8), (["0b", "0B"], 2)] {
        if let Some(digits) = prefixes
            .iter()
            .find_map(|prefix| value.strip_prefix(prefix))
        {
            if digits.is_empty()
                || !digits
                    .bytes()
                    .all(|byte| ascii_digit(byte, radix).is_some())
            {
                return f64::NAN;
            }
            return digits.bytes().fold(0.0, |number, byte| {
                number * f64::from(radix) + f64::from(ascii_digit(byte, radix).unwrap())
            });
        }
    }
    if valid_decimal(value) {
        value.parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

fn ascii_digit(byte: u8, radix: u32) -> Option<u32> {
    let value = match byte {
        b'0'..=b'9' => u32::from(byte - b'0'),
        b'a'..=b'f' => u32::from(byte - b'a') + 10,
        b'A'..=b'F' => u32::from(byte - b'A') + 10,
        _ => return None,
    };
    (value < radix).then_some(value)
}

fn valid_decimal(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let start_digits = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    let mut digits = index - start_digits;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        digits += index - fraction;
    }
    if digits == 0 {
        return false;
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let exponent = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == exponent {
            return false;
        }
    }
    index == bytes.len()
}

async fn poll_device_once(
    transport: Arc<dyn HttpTransport>,
    device: &DeviceAuthInfo,
    signal: Arc<dyn Abortable>,
) -> Result<DevicePollResult<DeviceTokenSuccess>, CodexOAuthError> {
    let request = HttpRequest {
        url: DEVICE_TOKEN_URL.into(),
        headers: content_type("application/json"),
        body: serde_json::to_vec(&json!({
            "device_auth_id": device.device_auth_id,
            "user_code": device.user_code,
        }))
        .expect("typed request is serializable"),
    };
    let response = match transport.post(request, signal.clone()).await {
        Ok(response) => response,
        Err(_) if signal.aborted() => {
            return Err(CodexOAuthError::Message("Login cancelled".into()));
        }
        Err(error) => return Err(error.into()),
    };
    if response.is_success() {
        let parsed = js_json_loads(&response.text(signal).await?)?;
        let result = nonempty_string(&parsed, "authorization_code")
            .zip(nonempty_string(&parsed, "code_verifier"));
        return Ok(match result {
            Some((authorization_code, code_verifier)) => {
                DevicePollResult::Complete(DeviceTokenSuccess {
                    authorization_code,
                    code_verifier,
                })
            }
            None => DevicePollResult::Failed {
                message: format!(
                    "Invalid OpenAI Codex device auth token response: {}",
                    js_json_stringify(&parsed)
                ),
            },
        });
    }
    if matches!(response.status, 403 | 404) {
        response.discard().await;
        return Ok(DevicePollResult::Pending);
    }
    let body = text_or_empty(&response, signal).await?;
    let error_code = js_json_loads(&body).ok().and_then(|value| {
        let error = value.get("error")?;
        error.as_string().or_else(|| error.get("code")?.as_string())
    });
    Ok(match error_code.as_deref() {
        Some("deviceauth_authorization_pending") => DevicePollResult::Pending,
        Some("slow_down") => DevicePollResult::SlowDown {
            interval_seconds: None,
        },
        _ => {
            let suffix = if body.is_empty() {
                String::new()
            } else {
                format!(": {body}")
            };
            DevicePollResult::Failed {
                message: format!(
                    "OpenAI Codex device auth failed with status {}{suffix}",
                    response.status
                ),
            }
        }
    })
}

async fn text_or_empty(
    response: &HttpResponse,
    signal: Arc<dyn Abortable>,
) -> Result<String, CodexOAuthError> {
    match response.text(signal).await {
        Ok(value) => Ok(value),
        Err(HttpTransportError::Cancelled) => Err(HttpTransportError::Cancelled.into()),
        Err(_) => Ok(String::new()),
    }
}

#[derive(Debug)]
struct OAuthToken {
    access: String,
    refresh: String,
    expires: f64,
}

async fn read_token_response(
    response: &HttpResponse,
    operation: &str,
    signal: Arc<dyn Abortable>,
) -> Result<OAuthToken, CodexOAuthError> {
    if !response.is_success() {
        let body = text_or_empty(response, signal).await?;
        let detail = if body.is_empty() {
            &response.reason_phrase
        } else {
            &body
        };
        return Err(CodexOAuthError::Message(format!(
            "OpenAI Codex token {operation} failed ({}): {detail}",
            response.status
        )));
    }
    let parsed = js_json_loads(&response.text(signal).await?)?;
    let access = nonempty_string(&parsed, "access_token");
    let refresh = nonempty_string(&parsed, "refresh_token");
    let expires_in = parsed.get("expires_in").and_then(JsJsonValue::as_f64);
    let Some((access, refresh, expires_in)) = access
        .zip(refresh)
        .zip(expires_in)
        .map(|((a, b), c)| (a, b, c))
    else {
        return Err(CodexOAuthError::Message(format!(
            "OpenAI Codex token {operation} response missing fields: {}",
            js_json_stringify(&parsed)
        )));
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0;
    Ok(OAuthToken {
        access,
        refresh,
        expires: now + expires_in * 1000.0,
    })
}

#[derive(Clone, Debug)]
struct DeviceAuthInfo {
    device_auth_id: String,
    user_code: String,
    interval_seconds: f64,
}
struct DeviceTokenSuccess {
    authorization_code: String,
    code_verifier: String,
}

struct CallbackServer {
    abort: AuthAbortController,
    result: Arc<Mutex<Option<oneshot::Receiver<Option<String>>>>>,
    task: Option<JoinHandle<()>>,
}

impl CallbackServer {
    async fn start(host: &str, port: u16, expected_state: String) -> Option<Self> {
        let listener = TcpListener::bind((host, port)).await.ok()?;
        let abort = AuthAbortController::default();
        let signal = abort.signal();
        let (sender, receiver) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut sender = Some(sender);
            loop {
                if signal.aborted() {
                    if let Some(sender) = sender.take() {
                        let _ = sender.send(None);
                    }
                    return;
                }
                tokio::select! {
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { continue };
                        if let Some(code) = handle_callback(stream, &expected_state).await {
                            if let Some(sender) = sender.take() { let _ = sender.send(Some(code)); }
                            return;
                        }
                    }
                    () = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
                }
            }
        });
        Some(Self {
            abort,
            result: Arc::new(Mutex::new(Some(receiver))),
            task: Some(task),
        })
    }
    fn abort_controller(&self) -> AuthAbortController {
        self.abort.clone()
    }
    fn abort(&self) {
        self.abort.abort();
    }
    async fn wait(&mut self) -> Option<String> {
        let receiver = self.result.lock().await.take()?;
        receiver.await.ok().flatten()
    }
    async fn close(&mut self) {
        self.abort.abort();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

async fn handle_callback(mut stream: TcpStream, expected_state: &str) -> Option<String> {
    let mut buffer = vec![0_u8; 8192];
    let result = async {
        let read = stream.read(&mut buffer).await?;
        let request = std::str::from_utf8(&buffer[..read]).map_err(std::io::Error::other)?;
        let target = request
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| std::io::Error::other("missing target"))?;
        let url =
            Url::parse(&format!("http://localhost{target}")).map_err(std::io::Error::other)?;
        let parsed = query_result(url.query_pairs());
        let (status, message, code) = if url.path() != CALLBACK_PATH {
            (404, "Callback route not found.", None)
        } else if parsed.state.as_deref() != Some(expected_state) {
            (400, "State mismatch.", None)
        } else if parsed.code.as_deref().is_none_or(str::is_empty) {
            (400, "Missing authorization code.", None)
        } else {
            (200, "Authentication successful.", parsed.code)
        };
        write_callback_response(&mut stream, status, message).await?;
        Ok::<_, std::io::Error>(code)
    }
    .await;
    match result {
        Ok(code) => code,
        Err(_) => {
            let _ = write_callback_response(&mut stream, 500, "Internal server error.").await;
            None
        }
    }
}

async fn write_callback_response(
    stream: &mut TcpStream,
    status: u16,
    body: &str,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Mutex as StdMutex,
            atomic::{AtomicU64, Ordering},
        },
    };

    use async_trait::async_trait;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    use super::*;
    use crate::auth::AuthInteraction;

    #[derive(Clone)]
    struct FakeTransport {
        responses: Arc<Mutex<VecDeque<Result<HttpResponse, HttpTransportError>>>>,
        requests: Arc<Mutex<Vec<HttpRequest>>>,
    }

    impl FakeTransport {
        fn new(responses: impl IntoIterator<Item = HttpResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses.into_iter().map(Ok).collect())),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl HttpTransport for FakeTransport {
        async fn post(
            &self,
            request: HttpRequest,
            _signal: Arc<dyn Abortable>,
        ) -> Result<HttpResponse, HttpTransportError> {
            self.requests.lock().await.push(request);
            self.responses
                .lock()
                .await
                .pop_front()
                .unwrap_or_else(|| Err(HttpTransportError::Request("unscripted request".into())))
        }
    }

    struct TestInteraction {
        signal: Arc<dyn Abortable>,
        selection: String,
        manual: Option<String>,
        events: Arc<StdMutex<Vec<AuthEvent>>>,
        prompts: Arc<StdMutex<Vec<&'static str>>>,
        fail_notify: bool,
    }

    impl AuthInteraction for TestInteraction {
        fn signal(&self) -> Option<Arc<dyn Abortable>> {
            Some(self.signal.clone())
        }

        fn prompt(&self, prompt: AuthPrompt) -> super::super::InteractionFuture<String> {
            let answer = match prompt {
                AuthPrompt::Select(prompt) => {
                    assert!(prompt.signal.is_none());
                    self.prompts.lock().unwrap().push("select");
                    self.selection.clone()
                }
                AuthPrompt::ManualCode(prompt) => {
                    assert!(prompt.signal.is_some());
                    self.prompts.lock().unwrap().push("manual_code");
                    if let Some(value) = &self.manual {
                        value.clone()
                    } else {
                        let events = self.events.lock().unwrap();
                        let AuthEvent::AuthUrl(event) = events.last().unwrap() else {
                            panic!("auth-url notification must precede the prompt")
                        };
                        let url = Url::parse(&event.url).unwrap();
                        let state = url.query_pairs().find(|(key, _)| key == "state").unwrap().1;
                        format!("?code=manual&state={state}")
                    }
                }
                _ => panic!("unexpected prompt"),
            };
            Box::pin(async move { Ok(answer) })
        }

        fn notify(&self, event: AuthEvent) -> Result<(), AuthInteractionError> {
            if self.fail_notify {
                return Err(AuthInteractionError::new("notify failed"));
            }
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    impl ProviderAuthInteraction for TestInteraction {}

    struct InstantClock {
        millis: AtomicU64,
    }

    #[async_trait]
    impl DeviceClock for InstantClock {
        fn elapsed_seconds(&self) -> f64 {
            self.millis.load(Ordering::Acquire) as f64 / 1000.0
        }
        async fn sleep_seconds(&self, seconds: f64) {
            self.millis
                .fetch_add((seconds * 1000.0) as u64, Ordering::AcqRel);
        }
    }

    fn interaction(selection: &str) -> Arc<TestInteraction> {
        Arc::new(TestInteraction {
            signal: AuthAbortController::default().signal(),
            selection: selection.into(),
            manual: None,
            events: Arc::new(StdMutex::new(Vec::new())),
            prompts: Arc::new(StdMutex::new(Vec::new())),
            fail_notify: false,
        })
    }

    fn access_token() -> String {
        let payload = STANDARD
            .encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct"}}"#)
            .trim_end_matches('=')
            .to_owned();
        format!("x.{payload}.y")
    }

    fn token_response() -> HttpResponse {
        HttpResponse::new(
            200,
            "OK",
            serde_json::to_string(&json!({
                "access_token": access_token(),
                "refresh_token": "refresh",
                "expires_in": 3600,
            }))
            .unwrap(),
        )
    }

    #[test]
    fn parser_preserves_whatwg_boundaries() {
        assert_eq!(
            parse_authorization_input("?code=&state=xyz"),
            ParsedAuthorizationInput {
                code: Some(String::new()),
                state: Some("xyz".into())
            }
        );
        assert_eq!(
            parse_authorization_input("?code=+++"),
            ParsedAuthorizationInput {
                code: Some("   ".into()),
                state: None,
            }
        );
        assert_eq!(
            parse_authorization_input("??code=1&state=s"),
            ParsedAuthorizationInput {
                code: None,
                state: Some("s".into())
            }
        );
        assert_eq!(
            parse_authorization_input("code#state#discarded"),
            ParsedAuthorizationInput {
                code: Some("code".into()),
                state: Some("state".into())
            }
        );
        assert_eq!(
            parse_authorization_input("http://example.com:bad?code=x&state=s"),
            ParsedAuthorizationInput {
                code: None,
                state: Some("s".into())
            }
        );
        assert_eq!(
            parse_authorization_input("mailto:x?code=c&state=s"),
            ParsedAuthorizationInput {
                code: Some("c".into()),
                state: Some("s".into())
            }
        );
    }

    #[test]
    fn number_coercion_matches_javascript_discriminators() {
        assert_eq!(js_number_coerce("\u{feff}  "), 0.0);
        assert_eq!(js_number_coerce("0x1a"), 26.0);
        assert_eq!(js_number_coerce("0o10"), 8.0);
        assert_eq!(js_number_coerce("0b10"), 2.0);
        assert!(js_number_coerce("+0x1").is_nan());
        assert!(js_number_coerce("١").is_nan());
        assert!(js_number_coerce(&format!("0x{}", "f".repeat(1000))).is_infinite());
    }

    #[tokio::test]
    async fn device_flow_uses_real_poller_emits_event_and_sends_exact_requests() {
        let transport = FakeTransport::new([
            HttpResponse::new(
                200,
                "OK",
                r#"{"device_auth_id":"device","user_code":"user","interval":""}"#,
            ),
            HttpResponse::new(403, "Forbidden", "body-must-not-be-needed"),
            HttpResponse::new(
                200,
                "OK",
                r#"{"authorization_code":"authorization","code_verifier":"verifier"}"#,
            ),
            token_response(),
        ]);
        let oauth = OpenAiCodexOAuth::new(Arc::new(transport.clone()));
        let interaction = interaction(LOGIN_METHOD_DEVICE_CODE);
        let credential = oauth
            .login_device_code(
                interaction.clone(),
                &InstantClock {
                    millis: AtomicU64::new(0),
                },
            )
            .await
            .unwrap();
        assert_eq!(credential.extra().read()["account_id"], "acct");
        assert!(matches!(
            &interaction.events.lock().unwrap()[0],
            AuthEvent::DeviceCode(event)
                if event.user_code == "user"
                    && event.verification_uri == DEVICE_VERIFICATION_URI
                    && event.interval_seconds == Some(0.0)
                    && event.expires_in_seconds == Some(900.0)
        ));
        let requests = transport.requests.lock().await;
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[0].url, DEVICE_USER_CODE_URL);
        assert_eq!(requests[1].url, DEVICE_TOKEN_URL);
        assert_eq!(requests[2].url, DEVICE_TOKEN_URL);
        assert_eq!(requests[3].url, TOKEN_URL);
        assert_eq!(requests[0].headers["Content-Type"], "application/json");
        assert_eq!(
            js_json_loads(std::str::from_utf8(&requests[0].body).unwrap()).unwrap()["client_id"]
                .as_string()
                .as_deref(),
            Some(CLIENT_ID)
        );
        assert_eq!(
            js_json_loads(std::str::from_utf8(&requests[1].body).unwrap()).unwrap()
                ["device_auth_id"]
                .as_string()
                .as_deref(),
            Some("device")
        );
        let form = form_urlencoded::parse(&requests[3].body).collect::<BTreeMap<_, _>>();
        assert_eq!(form["redirect_uri"], DEVICE_REDIRECT_URI);
        assert_eq!(form["code_verifier"], "verifier");
    }

    #[tokio::test]
    async fn browser_manual_path_notifies_then_prompts_and_exchanges_with_local_pkce() {
        let transport = FakeTransport::new([token_response()]);
        let oauth = OpenAiCodexOAuth::new(Arc::new(transport.clone()));
        let interaction = interaction(LOGIN_METHOD_BROWSER);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let credential = oauth
            .login_browser(interaction.clone(), port)
            .await
            .unwrap();
        assert_eq!(credential.extra().read()["account_id"], "acct");
        assert_eq!(&*interaction.prompts.lock().unwrap(), &["manual_code"]);
        let event_url = {
            let events = interaction.events.lock().unwrap();
            let AuthEvent::AuthUrl(event) = &events[0] else {
                panic!()
            };
            event.url.clone()
        };
        let url = Url::parse(&event_url).unwrap();
        let names = url
            .query_pairs()
            .map(|(name, _)| name.into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "response_type",
                "client_id",
                "redirect_uri",
                "scope",
                "code_challenge",
                "code_challenge_method",
                "state",
                "id_token_add_organizations",
                "codex_cli_simplified_flow",
                "originator"
            ]
        );
        let requests = transport.requests.lock().await;
        let form = form_urlencoded::parse(&requests[0].body).collect::<BTreeMap<_, _>>();
        assert_eq!(form["code"], "manual");
        assert_eq!(form["redirect_uri"], REDIRECT_URI);
        assert!(!form["code_verifier"].is_empty());
    }

    #[tokio::test]
    async fn browser_bind_failure_silently_falls_back_to_manual_input() {
        let occupied = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = occupied.local_addr().unwrap().port();
        let transport = FakeTransport::new([token_response()]);
        let oauth = OpenAiCodexOAuth::new(Arc::new(transport));
        let interaction = interaction(LOGIN_METHOD_BROWSER);

        let credential = oauth.login_browser(interaction, port).await.unwrap();

        assert_eq!(credential.extra().read()["account_id"], "acct");
        drop(occupied);
    }

    #[tokio::test]
    async fn browser_callback_wins_the_race_and_supplies_the_exchange_code() {
        struct PendingInteraction {
            signal: Arc<dyn Abortable>,
            events: Arc<StdMutex<Vec<AuthEvent>>>,
        }
        impl AuthInteraction for PendingInteraction {
            fn signal(&self) -> Option<Arc<dyn Abortable>> {
                Some(self.signal.clone())
            }
            fn prompt(&self, prompt: AuthPrompt) -> super::super::InteractionFuture<String> {
                assert!(matches!(prompt, AuthPrompt::ManualCode(_)));
                Box::pin(std::future::pending())
            }
            fn notify(&self, event: AuthEvent) -> Result<(), AuthInteractionError> {
                self.events.lock().unwrap().push(event);
                Ok(())
            }
        }
        impl ProviderAuthInteraction for PendingInteraction {}

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let events = Arc::new(StdMutex::new(Vec::new()));
        let interaction = Arc::new(PendingInteraction {
            signal: AuthAbortController::default().signal(),
            events: events.clone(),
        });
        let transport = FakeTransport::new([token_response()]);
        let requests = transport.requests.clone();
        let oauth = OpenAiCodexOAuth::new(Arc::new(transport));
        let run = tokio::spawn(async move { oauth.login_browser(interaction, port).await });
        let state = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let event = events.lock().unwrap().first().cloned();
                if let Some(AuthEvent::AuthUrl(event)) = event {
                    let url = Url::parse(&event.url).unwrap();
                    break url
                        .query_pairs()
                        .find(|(key, _)| key == "state")
                        .unwrap()
                        .1
                        .into_owned();
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let response = send_callback(
            port,
            &format!("/auth/callback?state={state}&code=server-code"),
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 200"));
        run.await.unwrap().unwrap();
        let requests = requests.lock().await;
        let form = form_urlencoded::parse(&requests[0].body).collect::<BTreeMap<_, _>>();
        assert_eq!(form["code"], "server-code");
    }

    #[tokio::test]
    async fn empty_manual_input_cancels_the_server_and_fails_without_exchange() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let interaction = Arc::new(TestInteraction {
            signal: AuthAbortController::default().signal(),
            selection: LOGIN_METHOD_BROWSER.into(),
            manual: Some("  \t".into()),
            events: Arc::new(StdMutex::new(Vec::new())),
            prompts: Arc::new(StdMutex::new(Vec::new())),
            fail_notify: false,
        });
        let transport = FakeTransport::new([]);
        let requests = transport.requests.clone();
        let error = OpenAiCodexOAuth::new(Arc::new(transport))
            .login_browser(interaction, port)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "Missing authorization code");
        assert!(requests.lock().await.is_empty());
        assert!(TcpListener::bind(("127.0.0.1", port)).await.is_ok());
    }

    #[tokio::test]
    async fn browser_notify_failure_remains_before_the_cleanup_boundary() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let controller = AuthAbortController::default();
        let interaction = Arc::new(TestInteraction {
            signal: controller.signal(),
            selection: LOGIN_METHOD_BROWSER.into(),
            manual: Some("unused".into()),
            events: Arc::new(StdMutex::new(Vec::new())),
            prompts: Arc::new(StdMutex::new(Vec::new())),
            fail_notify: true,
        });
        let error = OpenAiCodexOAuth::new(Arc::new(FakeTransport::new([])))
            .login_browser(interaction.clone(), port)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "notify failed");
        assert!(interaction.prompts.lock().unwrap().is_empty());
        assert!(TcpListener::bind(("127.0.0.1", port)).await.is_err());

        controller.abort();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)).await {
                    drop(listener);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn strict_response_field_types_cover_all_six_prov_016_fields() {
        for field in ["device_auth_id", "user_code"] {
            for invalid in [Value::Null, json!({"truthy":"object"})] {
                let mut body = json!({"device_auth_id":"device","user_code":"user","interval":1});
                body[field] = invalid;
                let transport = FakeTransport::new([HttpResponse::new(
                    200,
                    "OK",
                    serde_json::to_string(&body).unwrap(),
                )]);
                let error = OpenAiCodexOAuth::new(Arc::new(transport))
                    .start_device_auth(AuthAbortController::default().signal())
                    .await
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .starts_with("Invalid OpenAI Codex device code response:")
                );
            }
        }

        for field in ["authorization_code", "code_verifier"] {
            for invalid in [Value::Null, json!(7)] {
                let mut body = json!({"authorization_code":"code","code_verifier":"verifier"});
                body[field] = invalid;
                let transport = FakeTransport::new([HttpResponse::new(
                    200,
                    "OK",
                    serde_json::to_string(&body).unwrap(),
                )]);
                let result = poll_device_once(
                    Arc::new(transport),
                    &DeviceAuthInfo {
                        device_auth_id: "device".into(),
                        user_code: "user".into(),
                        interval_seconds: 1.0,
                    },
                    AuthAbortController::default().signal(),
                )
                .await
                .unwrap();
                assert!(
                    matches!(result, DevicePollResult::Failed { message } if message.starts_with("Invalid OpenAI Codex device auth token response:"))
                );
            }
        }

        for field in ["access_token", "refresh_token"] {
            for invalid in [Value::Null, json!(true)] {
                let mut body =
                    json!({"access_token":"access","refresh_token":"refresh","expires_in":1});
                body[field] = invalid;
                let response = HttpResponse::new(200, "OK", serde_json::to_string(&body).unwrap());
                let error = read_token_response(
                    &response,
                    "exchange",
                    AuthAbortController::default().signal(),
                )
                .await
                .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .starts_with("OpenAI Codex token exchange response missing fields:")
                );
            }
        }
    }

    #[tokio::test]
    async fn invalid_response_error_preserves_javascript_surrogate_rendering() {
        let transport = FakeTransport::new([HttpResponse::new(
            200,
            "OK",
            r#"{"device_auth_id":"device","user_code":"user","interval":null,"lone":"\ud800"}"#,
        )]);
        let error = OpenAiCodexOAuth::new(Arc::new(transport))
            .start_device_auth(AuthAbortController::default().signal())
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            r#"Invalid OpenAI Codex device code response: {"device_auth_id":"device","user_code":"user","interval":null,"lone":"\ud800"}"#
        );
    }

    #[tokio::test]
    async fn status_only_branches_do_not_read_bodies_and_non_success_read_failures_fall_back() {
        struct NeverRead;
        #[async_trait]
        impl super::super::HttpResponseBody for NeverRead {
            async fn text(
                self: Box<Self>,
                _signal: Arc<dyn Abortable>,
            ) -> Result<String, HttpTransportError> {
                panic!("body must not be read")
            }
        }
        let transport = FakeTransport::new([HttpResponse::deferred(
            404,
            "Not Found",
            Box::new(NeverRead),
        )]);
        let error = OpenAiCodexOAuth::new(Arc::new(transport))
            .start_device_auth(AuthAbortController::default().signal())
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
        );

        struct BrokenBody;
        #[async_trait]
        impl super::super::HttpResponseBody for BrokenBody {
            async fn text(
                self: Box<Self>,
                _signal: Arc<dyn Abortable>,
            ) -> Result<String, HttpTransportError> {
                Err(HttpTransportError::Body("broken".into()))
            }
        }
        let response = HttpResponse::deferred(500, "Server Error", Box::new(BrokenBody));
        let error = read_token_response(
            &response,
            "exchange",
            AuthAbortController::default().signal(),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "OpenAI Codex token exchange failed (500): Server Error"
        );
    }

    #[tokio::test]
    async fn callback_route_checks_path_state_and_empty_code() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut server = CallbackServer::start("127.0.0.1", port, "state".into())
            .await
            .unwrap();
        let response = send_callback(port, "/auth/callback?state=state&code=").await;
        assert!(response.starts_with("HTTP/1.1 400"));
        assert!(response.contains("Content-Type: text/html; charset=utf-8"));
        let response = send_callback(port, "/wrong").await;
        assert!(response.starts_with("HTTP/1.1 404"));
        let response = send_callback(port, "/auth/callback?state=wrong&code=ok").await;
        assert!(response.starts_with("HTTP/1.1 400"));
        let response = send_callback(port, "/auth/callback?state=state").await;
        assert!(response.starts_with("HTTP/1.1 400"));
        let response = send_raw_callback(
            port,
            b"GET /auth/callback?state=state&code=\xff HTTP/1.1\r\n\r\n",
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 500"));
        let response = send_callback(port, "/auth/callback?state=state&code=ok").await;
        assert!(response.starts_with("HTTP/1.1 200"));
        assert_eq!(server.wait().await.as_deref(), Some("ok"));
        server.close().await;
    }

    async fn send_callback(port: u16, target: &str) -> String {
        send_raw_callback(
            port,
            format!("GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes(),
        )
        .await
    }

    async fn send_raw_callback(port: u16, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        stream.write_all(request).await.unwrap();
        let mut output = String::new();
        stream.read_to_string(&mut output).await.unwrap();
        output
    }
}
