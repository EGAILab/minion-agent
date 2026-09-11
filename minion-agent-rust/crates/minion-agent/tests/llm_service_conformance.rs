#![cfg(feature = "conformance")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    sync::Arc,
};

use futures::{StreamExt, stream};
use minion_agent::llm::{
    AdapterStreamError, AdapterStreamErrorKind, AssistantMessage, DoneReason, LlmAdapter,
    LlmContext, LlmRegistration, LlmRequest, LlmService, ModelIdentity, RawAssistantStream,
    SimpleStreamOptions, StopReason, StreamChunk,
};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[derive(Deserialize)]
struct Scenario {
    llm_service: ServiceScenario,
    expect: Value,
}

#[derive(Deserialize)]
struct ServiceScenario {
    adapters: Vec<AdapterSpec>,
    steps: Vec<Step>,
    #[serde(default)]
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct AdapterSpec {
    id: String,
    provider: String,
    api: String,
    models: Vec<String>,
    behavior: AdapterBehavior,
    reject_message: Option<String>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AdapterBehavior {
    Ok,
    Reject,
}

#[derive(Deserialize)]
struct Step {
    register: Option<RegisterStep>,
    withdraw: Option<String>,
    stream: Option<StreamStep>,
}

#[derive(Deserialize)]
struct RegisterStep {
    adapter: String,
    #[serde(rename = "as")]
    handle: String,
}

#[derive(Deserialize)]
struct StreamStep {
    identity: CanonicalIdentity,
    #[serde(rename = "as")]
    observation: String,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    resolve: Option<CanonicalIdentity>,
    introspect: Option<String>,
}

#[derive(Clone, Deserialize)]
struct CanonicalIdentity {
    provider: String,
    model: String,
    api: String,
}

impl CanonicalIdentity {
    fn typed(&self) -> Result<ModelIdentity, String> {
        ModelIdentity::new(&self.provider, &self.api, &self.model)
            .map_err(|error| error.to_string())
    }
}

struct CanonicalAdapter {
    behavior: AdapterBehavior,
    reject_message: Option<String>,
    requests: Mutex<Vec<LlmRequest>>,
}

impl CanonicalAdapter {
    fn request_count(&self) -> usize {
        self.requests.lock().len()
    }
}

impl LlmAdapter for CanonicalAdapter {
    fn start(&self, request: LlmRequest) -> RawAssistantStream {
        let model = request.model.clone();
        self.requests.lock().push(request);
        match self.behavior {
            AdapterBehavior::Ok => {
                let message = AssistantMessage::pending(model, 0.0);
                Box::pin(stream::once(async move {
                    Ok(StreamChunk::Done {
                        reason: DoneReason::Stop,
                        message,
                    })
                }))
            }
            AdapterBehavior::Reject => {
                let message = self
                    .reject_message
                    .clone()
                    .expect("schema requires reject_message for reject behavior");
                Box::pin(stream::once(async move {
                    Err(AdapterStreamError::new(
                        AdapterStreamErrorKind::Provider,
                        message,
                    ))
                }))
            }
        }
    }
}

fn request(identity: ModelIdentity) -> LlmRequest {
    LlmRequest {
        model: identity,
        context: LlmContext::default(),
        options: SimpleStreamOptions::default(),
        signal: None,
    }
}

fn validate_references(scenario: &ServiceScenario) -> Result<(), String> {
    let mut adapter_ids = BTreeSet::new();
    for adapter in &scenario.adapters {
        if !adapter_ids.insert(adapter.id.as_str()) {
            return Err(format!("duplicate adapter id {}", adapter.id));
        }
    }

    let mut handles = BTreeSet::new();
    let mut observations = BTreeSet::new();
    for step in &scenario.steps {
        let operation_count = usize::from(step.register.is_some())
            + usize::from(step.withdraw.is_some())
            + usize::from(step.stream.is_some());
        if operation_count != 1 {
            return Err("each step must contain exactly one operation".into());
        }
        if let Some(register) = &step.register {
            if !adapter_ids.contains(register.adapter.as_str()) {
                return Err(format!("unknown adapter {}", register.adapter));
            }
            if !handles.insert(register.handle.as_str()) {
                return Err(format!("duplicate handle id {}", register.handle));
            }
        }
        if let Some(withdraw) = &step.withdraw
            && !handles.contains(withdraw.as_str())
        {
            return Err(format!("unknown withdrawal handle {withdraw}"));
        }
        if let Some(stream) = &step.stream
            && !observations.insert(stream.observation.as_str())
        {
            return Err(format!("duplicate observation id {}", stream.observation));
        }
    }
    for query in &scenario.queries {
        if query.resolve.is_some() == query.introspect.is_some() {
            return Err(format!(
                "query {} must contain exactly one operation",
                query.id
            ));
        }
        if !observations.insert(query.id.as_str()) {
            return Err(format!("duplicate observation id {}", query.id));
        }
    }
    Ok(())
}

async fn observe_stream(service: &LlmService, identity: ModelIdentity) -> Value {
    let Ok(mut result) = service.stream(request(identity)) else {
        return json!({"raised": true});
    };
    while let Some(chunk) = result.next().await {
        if chunk.is_terminal() {
            let message = chunk.partial();
            return match message.stop_reason {
                StopReason::Error | StopReason::Aborted => {
                    let mut value = json!({"settled": "error"});
                    if let Some(error_message) = &message.error_message {
                        value["error_message"] = Value::String(error_message.clone());
                    }
                    value
                }
                StopReason::Stop
                | StopReason::Length
                | StopReason::ToolUse
                | StopReason::Deferred => json!({"settled": "ok"}),
                StopReason::Pending => return json!({"raised": true}),
            };
        }
    }
    json!({"raised": true})
}

fn canonical_identity(identity: &ModelIdentity) -> Value {
    json!({
        "provider": identity.provider(),
        "model": identity.model_id(),
        "api": identity.api(),
    })
}

async fn run_scenario(scenario: &Scenario) -> Result<Value, String> {
    validate_references(&scenario.llm_service)?;
    let service = LlmService::new();
    let adapters = scenario
        .llm_service
        .adapters
        .iter()
        .map(|adapter| {
            (
                adapter.id.clone(),
                (
                    adapter,
                    Arc::new(CanonicalAdapter {
                        behavior: adapter.behavior,
                        reject_message: adapter.reject_message.clone(),
                        requests: Mutex::new(Vec::new()),
                    }),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut handles = BTreeMap::<String, LlmRegistration>::new();
    let mut actual = serde_json::Map::new();

    for step in &scenario.llm_service.steps {
        if let Some(register) = &step.register {
            let (spec, adapter) = &adapters[&register.adapter];
            let identities = spec
                .models
                .iter()
                .map(|model| ModelIdentity::new(&spec.provider, &spec.api, model))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            handles.insert(
                register.handle.clone(),
                service.register_models(identities, adapter.clone()),
            );
        } else if let Some(withdraw) = &step.withdraw {
            handles[withdraw].withdraw();
        } else if let Some(stream) = &step.stream {
            let observation = observe_stream(&service, stream.identity.typed()?).await;
            if scenario.expect.get(&stream.observation).is_some() {
                actual.insert(stream.observation.clone(), observation);
            }
        }
    }

    for query in &scenario.llm_service.queries {
        if query.introspect.as_deref() == Some("models") {
            actual.insert(
                query.id.clone(),
                json!({
                    "models": service
                        .models()
                        .iter()
                        .map(canonical_identity)
                        .collect::<Vec<_>>()
                }),
            );
        } else if let Some(identity) = &query.resolve {
            let before = adapters
                .iter()
                .map(|(id, (_, adapter))| (id, adapter.request_count()))
                .collect::<BTreeMap<_, _>>();
            let found = match service.stream(request(identity.typed()?)) {
                Ok(mut result) => {
                    while result.next().await.is_some() {}
                    true
                }
                Err(_) => false,
            };
            if !found {
                actual.insert(query.id.clone(), json!({"resolve": {"found": false}}));
                continue;
            }
            let owners = adapters
                .iter()
                .filter(|(id, (_, adapter))| adapter.request_count() == before[id] + 1)
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>();
            if owners.len() != 1 {
                return Err(format!(
                    "resolve {} reached {} adapters instead of exactly one",
                    query.id,
                    owners.len()
                ));
            }
            actual.insert(
                query.id.clone(),
                json!({"resolve": {"found": true, "adapter": owners[0]}}),
            );
        }
    }
    Ok(Value::Object(actual))
}

#[test]
fn all_layer_10_scenarios_drive_the_real_rust_llm_service() {
    futures::executor::block_on(async {
        let mut scenarios = fs::read_dir(root().join("conformance/agent"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("llm-service-") && name.ends_with(".yaml"))
            })
            .collect::<Vec<_>>();
        scenarios.sort();
        assert_eq!(scenarios.len(), 7);

        for path in scenarios {
            let scenario: Scenario =
                serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            let actual = run_scenario(&scenario)
                .await
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert_eq!(actual, scenario.expect, "{}", path.display());
        }
    });
}

#[test]
fn canonical_reference_validation_rejects_ambiguous_or_dangling_operations() {
    let scenario = ServiceScenario {
        adapters: Vec::new(),
        steps: vec![Step {
            register: None,
            withdraw: Some("missing".into()),
            stream: None,
        }],
        queries: Vec::new(),
    };
    assert_eq!(
        validate_references(&scenario).unwrap_err(),
        "unknown withdrawal handle missing"
    );
}
