use std::sync::Arc;

use minion_agent::{FiberState, PluginInitError, PluginSpec, Runtime, Service, ServiceName};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct EmptyConfig {}

fn run(future: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(future);
}

fn dependent_spec() -> minion_agent::DynPluginSpec {
    PluginSpec::<EmptyConfig>::new(
        "dependent",
        vec![ServiceName::new("tools").unwrap()],
        || json!({ "type": "object" }),
        |_context, _config| async { Ok::<_, PluginInitError>(()) },
    )
    .erase()
}

#[derive(Debug)]
struct Tools;

impl Service for Tools {
    const NAME: &'static str = "tools";
}

fn provider_spec() -> minion_agent::DynPluginSpec {
    PluginSpec::<EmptyConfig>::new(
        "provider",
        vec![],
        || json!({ "type": "object" }),
        |context, _config| async move {
            context
                .provide(Arc::new(Tools), None)
                .map_err(|error| PluginInitError::new(error.to_string()))?;
            Ok(())
        },
    )
    .erase()
}

#[test]
fn provider_absence_keeps_a_real_runtime_dependent_pending() {
    run(async {
        let runtime = Runtime::new();
        let dependent = runtime.mount(&dependent_spec(), json!({})).unwrap();

        runtime.reconcile().await.unwrap();

        assert_eq!(dependent.state(), FiberState::Pending);
    });
}

#[test]
fn provider_appearance_activates_a_real_runtime_dependent() {
    run(async {
        let runtime = Runtime::new();
        let dependent = runtime.mount(&dependent_spec(), json!({})).unwrap();
        runtime.reconcile().await.unwrap();
        assert_eq!(dependent.state(), FiberState::Pending);

        let provider = runtime.mount(&provider_spec(), json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        assert_eq!(provider.state(), FiberState::Active);
        assert_eq!(dependent.state(), FiberState::Active);
        assert!(runtime.services().require::<Tools>().is_ok());
    });
}

#[test]
fn provider_revocation_reconciles_the_dependent_without_a_caller_driven_pass() {
    run(async {
        let runtime = Runtime::new();
        let dependent = runtime.mount(&dependent_spec(), json!({})).unwrap();
        let provider = runtime.mount(&provider_spec(), json!({})).unwrap();
        runtime.reconcile().await.unwrap();
        assert_eq!(dependent.state(), FiberState::Active);

        runtime.unmount(&provider).await.unwrap();

        assert_eq!(provider.state(), FiberState::Disposed);
        assert_eq!(dependent.state(), FiberState::Pending);
        assert_eq!(
            dependent.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Active,
                FiberState::Unloading,
                FiberState::Pending,
            ]
        );
    });
}

#[test]
fn one_provider_reconciles_multiple_independent_dependents() {
    run(async {
        let runtime = Runtime::new();
        let first = runtime.mount(&dependent_spec(), json!({})).unwrap();
        let second_spec = PluginSpec::<EmptyConfig>::new(
            "second-dependent",
            vec![ServiceName::new("tools").unwrap()],
            || json!({ "type": "object" }),
            |_context, _config| async { Ok::<_, PluginInitError>(()) },
        )
        .erase();
        let second = runtime.mount(&second_spec, json!({})).unwrap();
        let provider = runtime.mount(&provider_spec(), json!({})).unwrap();

        runtime.reconcile().await.unwrap();
        assert_eq!(first.state(), FiberState::Active);
        assert_eq!(second.state(), FiberState::Active);

        runtime.unmount(&provider).await.unwrap();
        assert_eq!(first.state(), FiberState::Pending);
        assert_eq!(second.state(), FiberState::Pending);
    });
}
