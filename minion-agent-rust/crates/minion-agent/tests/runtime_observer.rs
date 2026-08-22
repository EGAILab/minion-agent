use std::sync::{Arc, Mutex};

use minion_agent::{
    FiberState, PluginInitError, PluginSpec, Runtime, RuntimeObservation, RuntimeObserver, Service,
    ServiceName,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct EmptyConfig {}

#[derive(Debug)]
struct Tools;

impl Service for Tools {
    const NAME: &'static str = "tools";
}

struct Recorder(Arc<Mutex<Vec<RuntimeObservation>>>);

impl RuntimeObserver for Recorder {
    fn observe(&self, observation: RuntimeObservation) {
        self.0.lock().unwrap().push(observation);
    }
}

#[test]
fn observer_records_inline_revocation_order_without_driving_transitions() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(async {
            let observations = Arc::new(Mutex::new(Vec::new()));
            let runtime = Runtime::with_observer(Arc::new(Recorder(Arc::clone(&observations))));
            let provider_spec = PluginSpec::<EmptyConfig>::new(
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
            .erase();
            let dependent_spec = PluginSpec::<EmptyConfig>::new(
                "dependent",
                vec![ServiceName::new("tools").unwrap()],
                || json!({ "type": "object" }),
                |_context, _config| async { Ok::<_, PluginInitError>(()) },
            )
            .erase();
            let provider = runtime.mount(&provider_spec, json!({})).unwrap();
            let dependent = runtime.mount(&dependent_spec, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            observations.lock().unwrap().clear();

            runtime.unmount(&provider).await.unwrap();

            assert_eq!(
                *observations.lock().unwrap(),
                vec![
                    RuntimeObservation::FiberState {
                        plugin: "provider".into(),
                        state: FiberState::Unloading,
                    },
                    RuntimeObservation::ServiceRevoked {
                        plugin: "provider".into(),
                        service: ServiceName::new("tools").unwrap(),
                    },
                    RuntimeObservation::FiberState {
                        plugin: "dependent".into(),
                        state: FiberState::Unloading,
                    },
                    RuntimeObservation::FiberState {
                        plugin: "dependent".into(),
                        state: FiberState::Pending,
                    },
                    RuntimeObservation::FiberState {
                        plugin: "provider".into(),
                        state: FiberState::Disposed,
                    },
                ]
            );
            assert_eq!(dependent.state(), FiberState::Pending);
        });
}

#[test]
fn observer_records_effect_boundaries_from_the_real_context() {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(async {
            let observations = Arc::new(Mutex::new(Vec::new()));
            let runtime = Runtime::with_observer(Arc::new(Recorder(Arc::clone(&observations))));
            let spec = PluginSpec::<EmptyConfig>::new(
                "owner",
                vec![],
                || json!({ "type": "object" }),
                |context, _config| async move {
                    context
                        .effect("resource", || Box::pin(async { Ok(()) }))
                        .unwrap();
                    Ok::<_, PluginInitError>(())
                },
            )
            .erase();
            let fiber = runtime.mount(&spec, json!({})).unwrap();
            runtime.reconcile().await.unwrap();
            runtime.unmount(&fiber).await.unwrap();

            let observations = observations.lock().unwrap();
            assert!(observations.contains(&RuntimeObservation::EffectCreated {
                plugin: "owner".into(),
                label: "resource".into(),
            }));
            assert!(observations.contains(&RuntimeObservation::EffectDisposed {
                plugin: "owner".into(),
                label: "resource".into(),
            }));
        });
}
