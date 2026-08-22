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

#[test]
fn revocation_settles_all_affected_dependents_when_one_cleanup_fails() {
    run(async {
        let runtime = Runtime::new();
        let failing_spec = PluginSpec::<EmptyConfig>::new(
            "failing-dependent",
            vec![ServiceName::new("tools").unwrap()],
            || json!({ "type": "object" }),
            |context, _config| async move {
                context
                    .effect("fails", || {
                        Box::pin(async {
                            Err(minion_agent::DisposeError::new(
                                "failing-dependent",
                                "cleanup failed",
                            ))
                        })
                    })
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            },
        )
        .erase();
        let failing = runtime.mount(&failing_spec, json!({})).unwrap();
        let healthy_spec = PluginSpec::<EmptyConfig>::new(
            "healthy-dependent",
            vec![ServiceName::new("tools").unwrap()],
            || json!({ "type": "object" }),
            |_context, _config| async { Ok::<_, PluginInitError>(()) },
        )
        .erase();
        let healthy = runtime.mount(&healthy_spec, json!({})).unwrap();
        let provider = runtime.mount(&provider_spec(), json!({})).unwrap();
        runtime.reconcile().await.unwrap();

        let error = runtime.unmount(&provider).await.unwrap_err();

        assert_eq!(failing.state(), FiberState::Pending);
        assert_eq!(healthy.state(), FiberState::Pending);
        assert_eq!(provider.state(), FiberState::Disposed);
        assert!(
            format!("{error:?}").contains("cleanup failed"),
            "unexpected error: {error:?}"
        );
    });
}

#[test]
fn a_failed_reconciliation_settles_the_fiber_and_does_not_poison_the_runtime() {
    run(async {
        let runtime = Runtime::new();
        let failing_spec = PluginSpec::<EmptyConfig>::new(
            "failing",
            vec![],
            || json!({ "type": "object" }),
            |_context, _config| async { Err(PluginInitError::new("expected failure")) },
        )
        .erase();
        let failing = runtime.mount(&failing_spec, json!({})).unwrap();

        let error = runtime.reconcile().await.unwrap_err();
        assert!(error.to_string().contains("expected failure"));
        assert_eq!(failing.state(), FiberState::Failed);

        let healthy_spec = PluginSpec::<EmptyConfig>::new(
            "healthy",
            vec![],
            || json!({ "type": "object" }),
            |_context, _config| async { Ok::<_, PluginInitError>(()) },
        )
        .erase();
        let healthy = runtime.mount(&healthy_spec, json!({})).unwrap();
        runtime.reconcile().await.unwrap();
        assert_eq!(healthy.state(), FiberState::Active);
        assert_eq!(failing.state(), FiberState::Failed);
    });
}

#[test]
fn provider_appearance_reaches_dependents_before_an_unrelated_failure_aborts_the_pass() {
    run(async {
        let runtime = Runtime::new();
        let provider = runtime.mount(&provider_spec(), json!({})).unwrap();
        let failing_spec = PluginSpec::<EmptyConfig>::new(
            "unrelated-failure",
            vec![],
            || json!({ "type": "object" }),
            |_context, _config| async { Err(PluginInitError::new("expected failure")) },
        )
        .erase();
        let failing = runtime.mount(&failing_spec, json!({})).unwrap();
        let dependent = runtime.mount(&dependent_spec(), json!({})).unwrap();

        assert!(runtime.reconcile().await.is_err());

        assert_eq!(provider.state(), FiberState::Active);
        assert_eq!(failing.state(), FiberState::Failed);
        assert_eq!(dependent.state(), FiberState::Active);
    });
}

#[test]
fn dependency_revocation_from_inside_loading_invalidates_without_self_joining() {
    run(async {
        let runtime = Runtime::new();
        let provider = runtime.mount(&provider_spec(), json!({})).unwrap();
        runtime.reconcile().await.unwrap();
        let provider_slot = Arc::new(std::sync::Mutex::new(Some(provider.clone())));
        let subject_spec = PluginSpec::<EmptyConfig>::new(
            "loading-subject",
            vec![ServiceName::new("tools").unwrap()],
            || json!({ "type": "object" }),
            {
                let runtime = runtime.clone();
                let provider_slot = Arc::clone(&provider_slot);
                move |_context, _config| {
                    let runtime = runtime.clone();
                    let provider = provider_slot.lock().unwrap().clone().unwrap();
                    async move {
                        runtime.unmount(&provider).await.unwrap();
                        Ok::<_, PluginInitError>(())
                    }
                }
            },
        )
        .erase();
        let subject = runtime.mount(&subject_spec, json!({})).unwrap();

        runtime.reconcile().await.unwrap();

        assert_eq!(provider.state(), FiberState::Disposed);
        assert_eq!(subject.state(), FiberState::Pending);
        assert!(!subject.trace().contains(&FiberState::Active));
    });
}
