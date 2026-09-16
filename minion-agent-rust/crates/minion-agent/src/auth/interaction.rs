use std::{future::Future, pin::Pin, sync::Arc};

use thiserror::Error;

use super::{
    Abortable, ApiKeyCredential, AuthCheck, AuthContext, AuthResult, ModelAuth, OAuthCredential,
};

pub type AuthFuture<T> = Pin<Box<dyn Future<Output = Result<T, AuthMethodError>> + Send>>;
pub type InteractionFuture<T> =
    Pin<Box<dyn Future<Output = Result<T, AuthInteractionError>> + Send>>;

#[derive(Clone)]
pub struct AuthPromptText {
    pub message: String,
    pub placeholder: Option<String>,
    pub signal: Option<Arc<dyn Abortable>>,
}

#[derive(Clone)]
pub struct AuthPromptSecret {
    pub message: String,
    pub placeholder: Option<String>,
    pub signal: Option<Arc<dyn Abortable>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthPromptOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Clone)]
pub struct AuthPromptSelect {
    pub message: String,
    pub options: Arc<[AuthPromptOption]>,
    pub signal: Option<Arc<dyn Abortable>>,
}

#[derive(Clone)]
pub struct AuthPromptManualCode {
    pub message: String,
    pub placeholder: Option<String>,
    pub signal: Option<Arc<dyn Abortable>>,
}

#[derive(Clone)]
pub enum AuthPrompt {
    Text(AuthPromptText),
    Secret(AuthPromptSecret),
    Select(AuthPromptSelect),
    ManualCode(AuthPromptManualCode),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthInfoLink {
    pub url: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthEventInfo {
    pub message: String,
    pub links: Option<Arc<[AuthInfoLink]>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthEventUrl {
    pub url: String,
    pub instructions: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthEventDeviceCode {
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: Option<f64>,
    pub expires_in_seconds: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthEventProgress {
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AuthEvent {
    Info(AuthEventInfo),
    AuthUrl(AuthEventUrl),
    DeviceCode(AuthEventDeviceCode),
    Progress(AuthEventProgress),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct AuthInteractionError {
    pub message: String,
}

impl AuthInteractionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Login interaction surface. Notification is deliberately synchronous.
pub trait AuthInteraction: Send + Sync {
    fn signal(&self) -> Option<Arc<dyn Abortable>>;
    fn prompt(&self, prompt: AuthPrompt) -> InteractionFuture<String>;
    fn notify(&self, event: AuthEvent) -> Result<(), AuthInteractionError>;
}

/// Provider-normalized interaction: the whole-flow signal is always present.
///
/// This supertrait relationship is Rust's sound representation of the adopted
/// `ProviderAuthInteraction IS-A AuthInteraction` rule. Access through either
/// trait is read-only, as allowed by PROV-015.
pub trait ProviderAuthInteraction: AuthInteraction {
    fn provider_signal(&self) -> Arc<dyn Abortable> {
        self.signal()
            .expect("ProviderAuthInteraction requires a signal")
    }
}

pub type ApiKeyResolve = Arc<
    dyn Fn(
            Arc<dyn AuthContext>,
            Option<ApiKeyCredential>,
            Arc<dyn Abortable>,
        ) -> AuthFuture<Option<AuthResult>>
        + Send
        + Sync,
>;
pub type ApiKeyCheck = Arc<
    dyn Fn(
            Arc<dyn AuthContext>,
            Option<ApiKeyCredential>,
            Arc<dyn Abortable>,
        ) -> AuthFuture<Option<AuthCheck>>
        + Send
        + Sync,
>;
pub type ApiKeyLogin =
    Arc<dyn Fn(Arc<dyn ProviderAuthInteraction>) -> AuthFuture<ApiKeyCredential> + Send + Sync>;

pub struct ApiKeyAuth {
    pub name: String,
    pub resolve: ApiKeyResolve,
    pub login: Option<ApiKeyLogin>,
    pub check: Option<ApiKeyCheck>,
}

pub type OAuthLogin =
    Arc<dyn Fn(Arc<dyn ProviderAuthInteraction>) -> AuthFuture<OAuthCredential> + Send + Sync>;
pub type OAuthRefresh =
    Arc<dyn Fn(OAuthCredential, Arc<dyn Abortable>) -> AuthFuture<OAuthCredential> + Send + Sync>;
pub type OAuthToAuth = Arc<dyn Fn(OAuthCredential) -> AuthFuture<ModelAuth> + Send + Sync>;

pub struct OAuthAuth {
    pub name: String,
    pub login: OAuthLogin,
    pub refresh: OAuthRefresh,
    pub to_auth: OAuthToAuth,
    pub is_subscription: Option<bool>,
    pub login_label: Option<String>,
}

#[derive(Debug, Error, Eq, PartialEq)]
#[error("provider auth must define api_key, oauth, or both")]
pub struct EmptyProviderAuthError;

pub struct ProviderAuth {
    pub api_key: Option<ApiKeyAuth>,
    pub oauth: Option<OAuthAuth>,
}

impl ProviderAuth {
    pub fn new(
        api_key: Option<ApiKeyAuth>,
        oauth: Option<OAuthAuth>,
    ) -> Result<Self, EmptyProviderAuthError> {
        if api_key.is_none() && oauth.is_none() {
            Err(EmptyProviderAuthError)
        } else {
            Ok(Self { api_key, oauth })
        }
    }

    pub fn api_key(&self) -> Option<&ApiKeyAuth> {
        self.api_key.as_ref()
    }

    pub fn oauth(&self) -> Option<&OAuthAuth> {
        self.oauth.as_ref()
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct AuthMethodError {
    pub message: String,
}

impl AuthMethodError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<AuthInteractionError> for AuthMethodError {
    fn from(value: AuthInteractionError) -> Self {
        Self::new(value.message)
    }
}
