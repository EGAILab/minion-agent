use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type SharedEnvironment = Arc<RwLock<BTreeMap<String, String>>>;
pub type SharedExtra = Arc<RwLock<BTreeMap<String, Value>>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    ApiKey,
    Oauth,
}

#[derive(Clone, Debug)]
pub struct ApiKeyCredential(Arc<RwLock<ApiKeyCredentialData>>);

#[derive(Clone, Debug)]
struct ApiKeyCredentialData {
    key: Option<String>,
    env: Option<SharedEnvironment>,
}

impl ApiKeyCredential {
    pub fn new(key: Option<String>, env: Option<SharedEnvironment>) -> Self {
        Self(Arc::new(RwLock::new(ApiKeyCredentialData { key, env })))
    }
    pub fn key(&self) -> Option<String> {
        self.0.read().key.clone()
    }
    pub fn set_key(&self, key: Option<String>) {
        self.0.write().key = key;
    }
    pub fn env(&self) -> Option<SharedEnvironment> {
        self.0.read().env.clone()
    }
    pub fn set_env(&self, env: Option<SharedEnvironment>) {
        self.0.write().env = env;
    }
}

#[derive(Clone, Debug)]
pub struct OAuthCredential(Arc<RwLock<OAuthCredentialData>>);

#[derive(Clone, Debug)]
struct OAuthCredentialData {
    access: String,
    refresh: String,
    expires: f64,
    extra: SharedExtra,
}

impl OAuthCredential {
    pub fn new(access: String, refresh: String, expires: f64, extra: SharedExtra) -> Self {
        Self(Arc::new(RwLock::new(OAuthCredentialData {
            access,
            refresh,
            expires,
            extra,
        })))
    }
    pub fn access(&self) -> String {
        self.0.read().access.clone()
    }
    pub fn set_access(&self, value: impl Into<String>) {
        self.0.write().access = value.into();
    }
    pub fn refresh(&self) -> String {
        self.0.read().refresh.clone()
    }
    pub fn set_refresh(&self, value: impl Into<String>) {
        self.0.write().refresh = value.into();
    }
    pub fn expires(&self) -> f64 {
        self.0.read().expires
    }
    pub fn set_expires(&self, value: f64) {
        self.0.write().expires = value;
    }
    pub fn extra(&self) -> SharedExtra {
        self.0.read().extra.clone()
    }
    pub fn set_extra(&self, value: SharedExtra) {
        self.0.write().extra = value;
    }
}

#[derive(Clone, Debug)]
pub enum Credential {
    ApiKey(ApiKeyCredential),
    OAuth(OAuthCredential),
}

impl Credential {
    pub fn auth_type(&self) -> AuthType {
        match self {
            Self::ApiKey(_) => AuthType::ApiKey,
            Self::OAuth(_) => AuthType::Oauth,
        }
    }
    pub fn as_oauth(&self) -> Option<OAuthCredential> {
        match self {
            Self::OAuth(value) => Some(value.clone()),
            Self::ApiKey(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub auth_type: AuthType,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAuth {
    pub api_key: Option<String>,
    pub headers: Option<BTreeMap<String, String>>,
    pub base_url: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuthResult {
    pub auth: ModelAuth,
    pub env: Option<BTreeMap<String, String>>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthCheck {
    pub auth_type: AuthType,
    pub source: Option<String>,
}
