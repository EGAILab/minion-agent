use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use thiserror::Error;

use super::{
    Abortable, AuthOperationOptions, CombinedSignal, Credential, CredentialStore,
    CredentialStoreError, ModifyError, OAuthCredential,
};

pub const DEFAULT_MINIMUM_VALIDITY_MS: f64 = 300_000.0;
pub const DEFAULT_REFRESH_TIMEOUT: Duration = Duration::from_secs(15);

pub type RefreshFuture = Pin<Box<dyn Future<Output = Result<OAuthCredential, String>> + Send>>;
pub type RefreshOperation =
    Arc<dyn Fn(OAuthCredential, Arc<dyn Abortable>) -> RefreshFuture + Send + Sync>;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AuthorityError {
    #[error("OAuth refresh failed for {provider_id:?}: {message}")]
    OAuthRefresh {
        provider_id: String,
        message: String,
    },
    #[error("credential store {operation} failed for {provider_id:?}: {message}")]
    CredentialStore {
        provider_id: String,
        operation: &'static str,
        message: String,
    },
}

pub async fn refresh_if_expiring<S: CredentialStore>(
    store: &S,
    provider_id: &str,
    refresh: RefreshOperation,
    minimum_validity_ms: Option<f64>,
    options: AuthOperationOptions,
) -> Result<Option<OAuthCredential>, AuthorityError> {
    refresh_if_expiring_at(
        store,
        provider_id,
        refresh,
        minimum_validity_ms,
        options,
        now_ms,
    )
    .await
}

pub async fn refresh_if_expiring_at<S, N>(
    store: &S,
    provider_id: &str,
    refresh: RefreshOperation,
    minimum_validity_ms: Option<f64>,
    options: AuthOperationOptions,
    now: N,
) -> Result<Option<OAuthCredential>, AuthorityError>
where
    S: CredentialStore,
    N: Fn() -> f64 + Send + Sync + 'static,
{
    let stored = store
        .read(provider_id, options.clone())
        .await
        .map_err(|error| store_error(provider_id, "read", error))?;
    let Some(stored) = stored.and_then(|value| value.as_oauth()) else {
        return Ok(None);
    };
    let threshold = js_max(
        DEFAULT_MINIMUM_VALIDITY_MS,
        minimum_validity_ms.unwrap_or(0.0),
    );
    if !expires_soon(&stored, threshold, now()) {
        return Ok(Some(stored));
    }

    let provider = provider_id.to_owned();
    let provider_for_callback = provider.clone();
    let now = Arc::new(now);
    let callback_now = now.clone();
    let callback_options = options.clone();
    let modified = store
        .modify(
            provider.clone(),
            move |current| {
                let refresh = refresh.clone();
                let now = callback_now.clone();
                let provider = provider_for_callback.clone();
                let caller = callback_options.signal.clone();
                async move {
                    let Some(current) = current.and_then(|value| value.as_oauth()) else {
                        return Ok(None);
                    };
                    if !expires_soon(&current, threshold, now()) {
                        return Ok(None);
                    }
                    let signal: Arc<dyn Abortable> =
                        Arc::new(CombinedSignal::new(caller, DEFAULT_REFRESH_TIMEOUT));
                    refresh(current, signal)
                        .await
                        .map(|credential| Some(Credential::OAuth(credential)))
                        .map_err(|message| AuthorityError::OAuthRefresh {
                            provider_id: provider,
                            message,
                        })
                }
            },
            options,
        )
        .await;

    let result = match modified {
        Ok(value) => value,
        Err(ModifyError::Callback(error)) => return Err(error),
        Err(ModifyError::Store(error)) => return Err(store_error(&provider, "modify", error)),
    };
    let Some(result) = result.and_then(|value| value.as_oauth()) else {
        return Ok(None);
    };
    if minimum_validity_ms.is_some() && expires_soon(&result, threshold, now()) {
        return Err(AuthorityError::OAuthRefresh {
            provider_id: provider,
            message: "refreshed token expires too soon".into(),
        });
    }
    Ok(Some(result))
}

fn expires_soon(credential: &OAuthCredential, minimum: f64, now: f64) -> bool {
    now + minimum >= credential.expires()
}

fn js_max(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.max(right)
    }
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

fn store_error(
    provider_id: &str,
    operation: &'static str,
    error: CredentialStoreError,
) -> AuthorityError {
    AuthorityError::CredentialStore {
        provider_id: provider_id.to_owned(),
        operation,
        message: error.to_string(),
    }
}
