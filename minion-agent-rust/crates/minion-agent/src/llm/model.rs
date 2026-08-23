use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct ModelIdentity {
    provider: String,
    api: String,
    model_id: String,
}

impl ModelIdentity {
    pub fn new(
        provider: impl Into<String>,
        api: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<Self, ModelIdentityError> {
        let provider = provider.into();
        let api = api.into();
        let model_id = model_id.into();

        if provider.is_empty() {
            return Err(ModelIdentityError::MissingComponent { field: "provider" });
        }
        if api.is_empty() {
            return Err(ModelIdentityError::MissingComponent { field: "api" });
        }
        if model_id.is_empty() {
            return Err(ModelIdentityError::MissingComponent { field: "model_id" });
        }

        Ok(Self {
            provider,
            api,
            model_id,
        })
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn api(&self) -> &str {
        &self.api
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl<'de> Deserialize<'de> for ModelIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawModelIdentity {
            provider: String,
            api: String,
            model_id: String,
        }

        let raw = RawModelIdentity::deserialize(deserializer)?;
        Self::new(raw.provider, raw.api, raw.model_id).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ModelIdentityError {
    #[error("model identity requires non-empty {field}")]
    MissingComponent { field: &'static str },
}

impl ModelIdentityError {
    pub fn field(&self) -> &'static str {
        match self {
            Self::MissingComponent { field } => field,
        }
    }
}
