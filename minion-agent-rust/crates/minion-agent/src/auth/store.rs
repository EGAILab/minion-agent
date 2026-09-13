use std::{collections::HashMap, future::Future, sync::Arc, time::Duration};

use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use thiserror::Error;
use tokio::{sync::Mutex as AsyncMutex, task::JoinHandle};

use super::{AuthOperationOptions, Credential, CredentialInfo};

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CredentialStoreError {
    #[error("credential operation cancelled")]
    Cancelled,
    #[error("credential operation task failed: {0}")]
    Task(String),
    #[error("credential storage backend failed: {0}")]
    Backend(String),
}

#[derive(Debug, Error)]
pub enum ModifyError<E> {
    #[error(transparent)]
    Store(#[from] CredentialStoreError),
    #[error("credential modifier failed")]
    Callback(E),
}

struct StoreInner {
    credentials: RwLock<IndexMap<String, Credential>>,
    gates: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

#[derive(Clone)]
pub struct InMemoryCredentialStore {
    inner: Arc<StoreInner>,
}

/// App-owned credential persistence with serialized provider-local mutation.
pub trait CredentialStore: Clone + Send + Sync + 'static {
    fn read(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> impl Future<Output = Result<Option<Credential>, CredentialStoreError>> + Send;

    fn list(
        &self,
        options: AuthOperationOptions,
    ) -> impl Future<Output = Result<Vec<CredentialInfo>, CredentialStoreError>> + Send;

    fn modify<F, Fut, E>(
        &self,
        provider_id: impl Into<String> + Send,
        callback: F,
        options: AuthOperationOptions,
    ) -> impl Future<Output = Result<Option<Credential>, ModifyError<E>>> + Send
    where
        F: FnOnce(Option<Credential>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Option<Credential>, E>> + Send + 'static,
        E: Send + 'static;

    fn delete(
        &self,
        provider_id: impl Into<String> + Send,
        options: AuthOperationOptions,
    ) -> impl Future<Output = Result<(), CredentialStoreError>> + Send;
}

impl Default for InMemoryCredentialStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(StoreInner {
                credentials: RwLock::new(IndexMap::new()),
                gates: Mutex::new(HashMap::new()),
            }),
        }
    }
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn check(options: &AuthOperationOptions) -> Result<(), CredentialStoreError> {
        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            Err(CredentialStoreError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn gate(&self, provider_id: &str) -> Arc<AsyncMutex<()>> {
        self.inner
            .gates
            .lock()
            .entry(provider_id.to_owned())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    pub async fn read(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>, CredentialStoreError> {
        Self::check(&options)?;
        Ok(self.inner.credentials.read().get(provider_id).cloned())
    }

    pub async fn list(
        &self,
        options: AuthOperationOptions,
    ) -> Result<Vec<CredentialInfo>, CredentialStoreError> {
        Self::check(&options)?;
        Ok(self
            .inner
            .credentials
            .read()
            .iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id: provider_id.clone(),
                auth_type: credential.auth_type(),
            })
            .collect())
    }

    pub async fn modify<F, Fut, E>(
        &self,
        provider_id: impl Into<String>,
        callback: F,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>, ModifyError<E>>
    where
        F: FnOnce(Option<Credential>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Option<Credential>, E>> + Send + 'static,
        E: Send + 'static,
    {
        Self::check(&options)?;
        let provider_id = provider_id.into();
        let gate = self.gate(&provider_id);
        let inner = self.inner.clone();
        let task_options = options.clone();
        let task: JoinHandle<Result<Option<Credential>, ModifyError<E>>> =
            tokio::spawn(async move {
                let _guard = gate.lock().await;
                Self::check(&task_options)?;
                let current = inner.credentials.read().get(&provider_id).cloned();
                let next = callback(current.clone())
                    .await
                    .map_err(ModifyError::Callback)?;
                Self::check(&task_options)?;
                if let Some(next) = next {
                    inner.credentials.write().insert(provider_id, next.clone());
                    Ok(Some(next))
                } else {
                    Ok(current)
                }
            });
        wait_for_modify(task, options).await
    }

    pub async fn delete(
        &self,
        provider_id: impl Into<String>,
        options: AuthOperationOptions,
    ) -> Result<(), CredentialStoreError> {
        Self::check(&options)?;
        let provider_id = provider_id.into();
        let gate = self.gate(&provider_id);
        let inner = self.inner.clone();
        let task_options = options.clone();
        let task = tokio::spawn(async move {
            let _guard = gate.lock().await;
            Self::check(&task_options)?;
            inner.credentials.write().shift_remove(&provider_id);
            Ok(())
        });
        wait_for_task(task, options).await
    }
}

impl CredentialStore for InMemoryCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>, CredentialStoreError> {
        InMemoryCredentialStore::read(self, provider_id, options).await
    }

    async fn list(
        &self,
        options: AuthOperationOptions,
    ) -> Result<Vec<CredentialInfo>, CredentialStoreError> {
        InMemoryCredentialStore::list(self, options).await
    }

    async fn modify<F, Fut, E>(
        &self,
        provider_id: impl Into<String> + Send,
        callback: F,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>, ModifyError<E>>
    where
        F: FnOnce(Option<Credential>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Option<Credential>, E>> + Send + 'static,
        E: Send + 'static,
    {
        InMemoryCredentialStore::modify(self, provider_id, callback, options).await
    }

    async fn delete(
        &self,
        provider_id: impl Into<String> + Send,
        options: AuthOperationOptions,
    ) -> Result<(), CredentialStoreError> {
        InMemoryCredentialStore::delete(self, provider_id, options).await
    }
}

async fn wait_for_task<T>(
    mut task: JoinHandle<Result<T, CredentialStoreError>>,
    options: AuthOperationOptions,
) -> Result<T, CredentialStoreError> {
    if options.signal.is_none() {
        return task
            .await
            .map_err(|error| CredentialStoreError::Task(error.to_string()))?;
    }
    loop {
        tokio::select! {
            result = &mut task => return result.map_err(|error| CredentialStoreError::Task(error.to_string()))?,
            () = tokio::time::sleep(Duration::from_millis(10)) => {
                SelfCheck::check(&options)?;
            }
        }
    }
}

async fn wait_for_modify<T, E>(
    mut task: JoinHandle<Result<T, ModifyError<E>>>,
    options: AuthOperationOptions,
) -> Result<T, ModifyError<E>> {
    if options.signal.is_none() {
        return task
            .await
            .map_err(|error| CredentialStoreError::Task(error.to_string()))?;
    }
    loop {
        tokio::select! {
            result = &mut task => return result.map_err(|error| CredentialStoreError::Task(error.to_string()))?,
            () = tokio::time::sleep(Duration::from_millis(10)) => {
                SelfCheck::check(&options)?;
            }
        }
    }
}

struct SelfCheck;
impl SelfCheck {
    fn check(options: &AuthOperationOptions) -> Result<(), CredentialStoreError> {
        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            Err(CredentialStoreError::Cancelled)
        } else {
            Ok(())
        }
    }
}
