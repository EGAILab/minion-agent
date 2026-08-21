use std::{any::Any, future::Future, sync::Arc};

use futures::future::{BoxFuture, FutureExt};
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

use super::{FiberHandle, FiberInitContext, ServiceName};

pub(crate) type ErasedConfig = Arc<dyn Any + Send + Sync>;
pub(crate) type ErasedInitializer = Arc<
    dyn Fn(FiberInitContext, ErasedConfig) -> BoxFuture<'static, Result<(), PluginInitError>>
        + Send
        + Sync,
>;
type TypedInitializer<C> = Arc<
    dyn Fn(FiberInitContext, Arc<C>) -> BoxFuture<'static, Result<(), PluginInitError>>
        + Send
        + Sync,
>;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("plugin initialization failed: {message}")]
pub struct PluginInitError {
    message: String,
}

impl PluginInitError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Error)]
pub enum PluginConfigError {
    #[error("invalid configuration for plugin {plugin}: {source}")]
    Deserialize {
        plugin: String,
        #[source]
        source: serde_json::Error,
    },
}

pub struct PluginSpec<C> {
    name: String,
    inject: Vec<ServiceName>,
    schema: Arc<dyn Fn() -> Value + Send + Sync>,
    initializer: TypedInitializer<C>,
}

impl<C> PluginSpec<C>
where
    C: DeserializeOwned + Send + Sync + 'static,
{
    pub fn new<S, F, Fut>(
        name: impl Into<String>,
        inject: Vec<ServiceName>,
        schema: S,
        initializer: F,
    ) -> Self
    where
        S: Fn() -> Value + Send + Sync + 'static,
        F: Fn(FiberInitContext, Arc<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), PluginInitError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            inject,
            schema: Arc::new(schema),
            initializer: Arc::new(move |context, config| initializer(context, config).boxed()),
        }
    }

    pub fn erase(self) -> DynPluginSpec {
        let name = self.name;
        let deserialize_name = name.clone();
        let typed_initializer = self.initializer;
        DynPluginSpec {
            name,
            inject: self.inject,
            schema: self.schema,
            deserialize: Arc::new(move |value| {
                serde_json::from_value::<C>(value)
                    .map(|config| Arc::new(config) as ErasedConfig)
                    .map_err(|source| PluginConfigError::Deserialize {
                        plugin: deserialize_name.clone(),
                        source,
                    })
            }),
            initializer: Arc::new(move |context, config| {
                let config = Arc::downcast::<C>(config)
                    .expect("typed plugin config disagrees with its erased deserializer");
                typed_initializer(context, config)
            }),
        }
    }
}

type ConfigDeserializer =
    Arc<dyn Fn(Value) -> Result<ErasedConfig, PluginConfigError> + Send + Sync>;

#[derive(Clone)]
pub struct DynPluginSpec {
    name: String,
    inject: Vec<ServiceName>,
    schema: Arc<dyn Fn() -> Value + Send + Sync>,
    deserialize: ConfigDeserializer,
    initializer: ErasedInitializer,
}

impl DynPluginSpec {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn inject(&self) -> &[ServiceName] {
        &self.inject
    }

    pub fn schema(&self) -> Value {
        (self.schema)()
    }

    pub fn mount<F>(
        &self,
        config: Value,
        dependencies_visible: F,
    ) -> Result<FiberHandle, PluginConfigError>
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        let config = (self.deserialize)(config)?;
        Ok(FiberHandle::new(
            self.name.clone(),
            self.inject.clone(),
            Arc::clone(&self.initializer),
            config,
            Arc::new(dependencies_visible),
        ))
    }
}

impl std::fmt::Debug for DynPluginSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynPluginSpec")
            .field("name", &self.name)
            .field("inject", &self.inject)
            .finish_non_exhaustive()
    }
}
