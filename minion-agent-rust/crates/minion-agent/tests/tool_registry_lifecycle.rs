use std::sync::Arc;

use minion_agent::{
    DynPluginSpec, PluginInitError, PluginSpec, RegistrationHandle, Runtime, ScopeHandle,
    tools::{ToolDefinition, ToolExecutionRequest},
};
use parking_lot::Mutex;
use serde_json::{Value, json};

fn run(future: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(future);
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        name,
        serde_json::from_value(json!({"type": "object", "properties": {}})).unwrap(),
        name,
        |_request: ToolExecutionRequest| Box::pin(async { unreachable!() }),
    )
}

fn registering_plugin(
    name: &str,
    definition: ToolDefinition,
    handle: Arc<Mutex<Option<RegistrationHandle>>>,
) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        name,
        vec![],
        || json!({}),
        move |context, _config| {
            let definition = definition.clone();
            let handle = Arc::clone(&handle);
            async move {
                let registration = context
                    .tools()
                    .map_err(|error| PluginInitError::new(error.to_string()))?
                    .register(&context, definition)
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                *handle.lock() = Some(registration);
                Ok(())
            }
        },
    )
    .erase()
}

#[test]
fn unscoped_registration_is_owned_by_the_registering_fiber() {
    run(async {
        let runtime = Runtime::new();
        let handle = Arc::new(Mutex::new(None));
        let plugin = registering_plugin("global-plugin", tool("global"), handle);
        let fiber = runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        assert!(runtime.tools().resolve("global", None).is_some());
        runtime.unmount(&fiber).await.unwrap();
        assert!(runtime.tools().resolve("global", None).is_none());
    });
}

#[test]
fn scoped_registration_survives_plugin_unmount_and_is_owned_by_scope() {
    run(async {
        let runtime = Runtime::new();
        let scope = runtime.create_scope(None).unwrap();
        let handle = Arc::new(Mutex::new(None));
        let plugin = registering_plugin("scoped-plugin", tool("scoped"), handle);
        let fiber = runtime
            .mount_in(&plugin, json!({}), Some(scope.clone()))
            .unwrap();
        runtime.reconcile().await.unwrap();

        runtime.unmount(&fiber).await.unwrap();
        assert!(runtime.tools().resolve("scoped", Some(&scope)).is_some());
        scope.dispose().await.unwrap();
        assert!(runtime.tools().resolve("scoped", Some(&scope)).is_none());
    });
}

#[test]
fn explicit_withdrawal_composes_safely_with_both_lifecycle_owners() {
    run(async {
        let runtime = Runtime::new();
        let scope = runtime.create_scope(None).unwrap();
        let slot = Arc::new(Mutex::new(None));
        let plugin = registering_plugin("scoped-plugin", tool("scoped"), Arc::clone(&slot));
        let fiber = runtime
            .mount_in(&plugin, json!({}), Some(scope.clone()))
            .unwrap();
        runtime.reconcile().await.unwrap();

        let handle = slot.lock().clone().unwrap();
        handle.dispose().await.unwrap();
        handle.dispose().await.unwrap();
        scope.dispose().await.unwrap();
        runtime.unmount(&fiber).await.unwrap();
        assert!(runtime.tools().visible(Some(&scope)).is_empty());
    });
}

#[test]
fn disposed_scope_observation_uses_the_real_runtime_scope_handle() {
    run(async {
        let runtime = Runtime::new();
        let scope: ScopeHandle = runtime.create_scope(None).unwrap();
        let slot = Arc::new(Mutex::new(None));
        let plugin = registering_plugin("scoped-plugin", tool("scoped"), slot);
        let _fiber = runtime
            .mount_in(&plugin, json!({}), Some(scope.clone()))
            .unwrap();
        runtime.reconcile().await.unwrap();
        assert!(runtime.tools().resolve("scoped", Some(&scope)).is_some());

        scope.dispose().await.unwrap();
        assert!(runtime.tools().visible(Some(&scope)).is_empty());
        assert!(runtime.tools().resolve("scoped", Some(&scope)).is_none());
    });
}
