use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use minion_agent::{
    Context, DispatchMode, DynPluginSpec, EventName, EventSpec, FiberHandle, FiberState,
    PluginInitError, PluginSpec, Runtime, RuntimeObservation, RuntimeObserver, ScopeHandle,
    ScopeId, Service, ServiceCheck, ServiceName,
};
use serde::Deserialize;
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, String>;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    name: String,
    #[serde(default)]
    description: Option<String>,
    plugins: Vec<PluginDefinition>,
    steps: Vec<Step>,
    expect_trace: Vec<Value>,
    #[serde(default)]
    expect_result: Option<Value>,
    #[serde(default)]
    expect_error: Option<ExpectedError>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PluginDefinition {
    id: String,
    #[serde(default)]
    inject: Vec<String>,
    #[serde(default)]
    provides: Option<Provides>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    scope_parent: Option<String>,
    #[serde(default)]
    config: Value,
    #[serde(default)]
    fails: bool,
    #[serde(default)]
    effects: Vec<EffectDefinition>,
    #[serde(default)]
    listeners: Vec<ListenerDefinition>,
    #[serde(default)]
    during_load: Vec<Step>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Provides {
    Name(String),
    Detailed(ProvidedService),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvidedService {
    name: String,
    #[serde(default = "default_true")]
    visible: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectDefinition {
    label: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListenerDefinition {
    event: String,
    action: String,
    tag: String,
    #[serde(default)]
    replacement: Value,
    #[serde(default)]
    returns: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Step {
    Mount(MountStep),
    Unmount(UnmountStep),
    DisposeScope(DisposeScopeStep),
    AttemptEffect(AttemptEffectStep),
    Dispatch(DispatchStepWrapper),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MountStep {
    mount: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UnmountStep {
    unmount: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DisposeScopeStep {
    dispose_scope: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttemptEffectStep {
    attempt_effect: EffectAttempt,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectAttempt {
    plugin: String,
    #[serde(default = "default_effect_label")]
    label: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DispatchStepWrapper {
    dispatch: DispatchStep,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DispatchStep {
    event: String,
    mode: String,
    #[serde(default)]
    args: Vec<Value>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    terminal: Value,
    #[serde(default)]
    terminal_from_args: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedError {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    message_contains: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_effect_label() -> String {
    "attempted-effect".to_owned()
}

#[derive(Clone)]
struct Shared {
    runtime: Runtime,
    trace: Arc<Mutex<Vec<Value>>>,
    fibers: Arc<Mutex<HashMap<String, FiberHandle>>>,
    contexts: Arc<Mutex<HashMap<String, Context>>>,
    scopes: Arc<Mutex<HashMap<String, ScopeHandle>>>,
    scope_names: Arc<Mutex<HashMap<ScopeId, String>>>,
    event_specs: Arc<HashMap<String, DynamicEventSpec>>,
}

struct RecordingObserver {
    trace: Arc<Mutex<Vec<Value>>>,
    scope_names: Arc<Mutex<HashMap<ScopeId, String>>>,
}

impl RuntimeObserver for RecordingObserver {
    fn observe(&self, observation: RuntimeObservation) {
        let entry = match observation {
            RuntimeObservation::FiberState { plugin, state } => json!({
                "event": "fiber_state",
                "plugin": plugin,
                "state": state_name(state),
            }),
            RuntimeObservation::EffectCreated { plugin, label } => json!({
                "event": "effect_created",
                "plugin": plugin,
                "label": label,
            }),
            RuntimeObservation::EffectDisposed { plugin, label } => json!({
                "event": "effect_disposed",
                "plugin": plugin,
                "label": label,
            }),
            RuntimeObservation::ServiceProvided { plugin, service } => json!({
                "event": "service_provided",
                "plugin": plugin,
                "service": service.as_str(),
            }),
            RuntimeObservation::ScopeDisposed { scope } => {
                let name = self
                    .scope_names
                    .lock()
                    .unwrap()
                    .get(&scope)
                    .cloned()
                    .unwrap_or_else(|| format!("scope-{}", scope.as_u64()));
                json!({ "event": "scope_disposed", "scope": name })
            }
            RuntimeObservation::ServiceRevoked { .. } => return,
        };
        self.trace.lock().unwrap().push(entry);
    }
}

#[derive(Clone)]
enum DynamicEventSpec {
    Emit(EventSpec<Vec<Value>, ()>),
    Parallel(EventSpec<Vec<Value>, ()>),
    Serial(EventSpec<Vec<Value>, Value>),
    Waterfall(EventSpec<Vec<Value>, Value>),
}

#[derive(Debug)]
struct Tools;
impl Service for Tools {
    const NAME: &'static str = "tools";
}

#[derive(Debug)]
struct GatedService;
impl Service for GatedService {
    const NAME: &'static str = "gated_service";
}

#[derive(Debug)]
struct SharedService;
impl Service for SharedService {
    const NAME: &'static str = "shared_service";
}

pub fn run_all() -> TestResult {
    let root = canonical_root()?;
    let mut paths: Vec<_> = fs::read_dir(&root)
        .map_err(|error| format!("read {}: {error}", root.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .collect();
    paths.sort();
    let mut failures = Vec::new();
    for path in paths {
        if let Err(error) = run_path(&path) {
            failures.push(format!("{}: {error}", path.display()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n\n"))
    }
}

fn canonical_root() -> TestResult<PathBuf> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let monorepo = crate_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| "Rust crate is not nested in the monorepo".to_owned())?;
    Ok(monorepo.join("conformance/runtime"))
}

fn run_path(path: &Path) -> TestResult {
    let source = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let scenario: Scenario = serde_yaml::from_str(&source).map_err(|error| error.to_string())?;
    let _ = &scenario.description;
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .map_err(|error| error.to_string())?
        .block_on(run_scenario(scenario))
}

async fn run_scenario(scenario: Scenario) -> TestResult {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let scope_names = Arc::new(Mutex::new(HashMap::new()));
    let observer = Arc::new(RecordingObserver {
        trace: Arc::clone(&trace),
        scope_names: Arc::clone(&scope_names),
    });
    let runtime = Runtime::with_observer(observer);
    let event_specs = match declare_events(&runtime, &scenario.steps) {
        Ok(specs) => specs,
        Err(error) => return compare_outcome(&scenario, &trace, None, Some(error)),
    };
    let shared = Shared {
        runtime,
        trace,
        fibers: Arc::new(Mutex::new(HashMap::new())),
        contexts: Arc::new(Mutex::new(HashMap::new())),
        scopes: Arc::new(Mutex::new(HashMap::new())),
        scope_names,
        event_specs: Arc::new(event_specs),
    };
    let definitions: HashMap<_, _> = scenario
        .plugins
        .iter()
        .cloned()
        .map(|plugin| (plugin.id.clone(), plugin))
        .collect();
    let mut result = None;
    let mut error = None;
    for step in &scenario.steps {
        match execute_step(step, &definitions, &shared).await {
            Ok(step_result) => {
                if step_result.is_some() {
                    result = step_result;
                }
            }
            Err(step_error) => {
                error = Some(step_error);
                break;
            }
        }
    }
    compare_outcome(&scenario, &shared.trace, result, error)
}

fn declare_events(
    runtime: &Runtime,
    steps: &[Step],
) -> Result<HashMap<String, DynamicEventSpec>, String> {
    let mut specs = HashMap::new();
    for step in steps {
        let Step::Dispatch(wrapper) = step else {
            continue;
        };
        let dispatch = &wrapper.dispatch;
        let name = EventName::new(&dispatch.event).map_err(|error| error.to_string())?;
        let dynamic = match dispatch.mode.as_str() {
            "emit" => DynamicEventSpec::Emit(EventSpec::new(name, DispatchMode::Emit, |_| ())),
            "parallel" => {
                DynamicEventSpec::Parallel(EventSpec::new(name, DispatchMode::Parallel, |_| ()))
            }
            "serial" => {
                DynamicEventSpec::Serial(EventSpec::new(name, DispatchMode::Serial, |_| {
                    Value::Null
                }))
            }
            "waterfall" => {
                let fixed = dispatch.terminal.clone();
                let from_args = dispatch.terminal_from_args;
                DynamicEventSpec::Waterfall(EventSpec::new(
                    name,
                    DispatchMode::Waterfall,
                    move |payload: &Vec<Value>| {
                        if from_args {
                            payload_value(payload)
                        } else {
                            fixed.clone()
                        }
                    },
                ))
            }
            mode => return Err(format!("unsupported dispatch mode {mode}")),
        };
        declare_dynamic(runtime, &dynamic).map_err(|error| error.to_string())?;
        specs.entry(dispatch.event.clone()).or_insert(dynamic);
    }
    Ok(specs)
}

fn declare_dynamic(
    runtime: &Runtime,
    spec: &DynamicEventSpec,
) -> Result<(), minion_agent::EventError> {
    match spec {
        DynamicEventSpec::Emit(spec) => runtime.events().declare(spec),
        DynamicEventSpec::Parallel(spec) => runtime.events().declare(spec),
        DynamicEventSpec::Serial(spec) => runtime.events().declare(spec),
        DynamicEventSpec::Waterfall(spec) => runtime.events().declare(spec),
    }
}

async fn execute_step(
    step: &Step,
    definitions: &HashMap<String, PluginDefinition>,
    shared: &Shared,
) -> Result<Option<Value>, String> {
    match step {
        Step::Mount(step) => {
            mount_plugin(&step.mount, definitions, shared).await?;
            Ok(None)
        }
        Step::Unmount(step) => {
            let fiber = shared
                .fibers
                .lock()
                .unwrap()
                .get(&step.unmount)
                .cloned()
                .ok_or_else(|| format!("unknown mounted plugin {}", step.unmount))?;
            shared
                .runtime
                .unmount(&fiber)
                .await
                .map_err(|error| error.to_string())?;
            Ok(None)
        }
        Step::DisposeScope(step) => {
            let scope = shared
                .scopes
                .lock()
                .unwrap()
                .get(&step.dispose_scope)
                .cloned()
                .ok_or_else(|| format!("unknown scope {}", step.dispose_scope))?;
            scope.dispose().await.map_err(|error| error.to_string())?;
            Ok(None)
        }
        Step::AttemptEffect(step) => {
            let context = shared
                .contexts
                .lock()
                .unwrap()
                .get(&step.attempt_effect.plugin)
                .cloned()
                .ok_or_else(|| format!("unknown plugin context {}", step.attempt_effect.plugin))?;
            context
                .effect(step.attempt_effect.label.clone(), || {
                    Box::pin(async { Ok(()) })
                })
                .map_err(|error| match error {
                    minion_agent::RuntimeError::InactiveOwner { .. } => {
                        "disposed-owner: cannot create effect".to_owned()
                    }
                    other => other.to_string(),
                })?;
            Ok(None)
        }
        Step::Dispatch(wrapper) => dispatch(&wrapper.dispatch, shared).await.map(Some),
    }
}

async fn mount_plugin(
    id: &str,
    definitions: &HashMap<String, PluginDefinition>,
    shared: &Shared,
) -> Result<(), String> {
    let definition = definitions
        .get(id)
        .cloned()
        .ok_or_else(|| format!("unknown plugin {id}"))?;
    let scope = match definition.scope.as_deref() {
        Some(name) => Some(ensure_scope(name, definitions, shared)?),
        None => None,
    };
    let spec = scripted_spec(definition.clone(), shared.clone())?;
    let fiber = shared
        .runtime
        .mount_in(&spec, definition.config.clone(), scope)
        .map_err(|error| error.to_string())?;
    shared.fibers.lock().unwrap().insert(id.to_owned(), fiber);
    match shared.runtime.reconcile().await {
        Ok(()) => Ok(()),
        Err(_) if definition.fails => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn ensure_scope(
    name: &str,
    definitions: &HashMap<String, PluginDefinition>,
    shared: &Shared,
) -> Result<ScopeHandle, String> {
    if let Some(scope) = shared.scopes.lock().unwrap().get(name).cloned() {
        return Ok(scope);
    }
    let parent_name = definitions
        .values()
        .find(|definition| definition.scope.as_deref() == Some(name))
        .and_then(|definition| definition.scope_parent.clone());
    let parent = parent_name
        .as_deref()
        .map(|parent| ensure_scope(parent, definitions, shared))
        .transpose()?;
    let scope = shared
        .runtime
        .create_scope(parent.as_ref())
        .map_err(|error| error.to_string())?;
    shared
        .scope_names
        .lock()
        .unwrap()
        .insert(scope.id(), name.to_owned());
    shared
        .scopes
        .lock()
        .unwrap()
        .insert(name.to_owned(), scope.clone());
    Ok(scope)
}

fn scripted_spec(definition: PluginDefinition, shared: Shared) -> TestResult<DynPluginSpec> {
    let inject = definition
        .inject
        .iter()
        .map(ServiceName::new)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let plugin = definition.id.clone();
    Ok(PluginSpec::<Value>::new(
        plugin.clone(),
        inject,
        || json!({ "type": "object" }),
        move |context, _config| {
            let definition = definition.clone();
            let shared = shared.clone();
            let plugin = plugin.clone();
            async move {
                shared
                    .contexts
                    .lock()
                    .unwrap()
                    .insert(plugin.clone(), context.clone());
                for effect in &definition.effects {
                    context
                        .effect(effect.label.clone(), || Box::pin(async { Ok(()) }))
                        .map_err(init_error)?;
                }
                if let Some(provides) = &definition.provides {
                    provide_scripted(&context, provides).map_err(init_error)?;
                }
                for listener in &definition.listeners {
                    register_listener(&context, &plugin, listener, &shared).map_err(init_error)?;
                }
                for step in &definition.during_load {
                    execute_during_load(step, &shared).await?;
                }
                if definition.fails {
                    return Err(PluginInitError::new(format!(
                        "{plugin} initialization failed"
                    )));
                }
                Ok(())
            }
        },
    )
    .erase())
}

fn init_error(error: impl ToString) -> PluginInitError {
    PluginInitError::new(error.to_string())
}

fn provide_scripted(
    context: &Context,
    provides: &Provides,
) -> Result<(), minion_agent::RuntimeError> {
    let (name, visible) = match provides {
        Provides::Name(name) => (name.as_str(), true),
        Provides::Detailed(detail) => (detail.name.as_str(), detail.visible),
    };
    let check: Option<ServiceCheck> = (!visible).then(|| Arc::new(|| false) as ServiceCheck);
    match name {
        "tools" => context.provide(Arc::new(Tools), check).map(|_| ()),
        "gated_service" => context.provide(Arc::new(GatedService), check).map(|_| ()),
        "shared_service" => context.provide(Arc::new(SharedService), check).map(|_| ()),
        other => Err(minion_agent::RuntimeError::InvalidName(format!(
            "unsupported scripted service {other}"
        ))),
    }
}

fn register_listener(
    context: &Context,
    plugin: &str,
    listener: &ListenerDefinition,
    shared: &Shared,
) -> Result<(), String> {
    let spec = shared
        .event_specs
        .get(&listener.event)
        .ok_or_else(|| format!("listener references undeclared event {}", listener.event))?;
    let effects = context.effect_store();
    let scope = context.scope();
    let bus = context.events().map_err(|error| error.to_string())?;
    let plugin = plugin.to_owned();
    let listener = listener.clone();
    let trace = Arc::clone(&shared.trace);
    match spec {
        DynamicEventSpec::Emit(spec) => bus
            .on_emit(spec, &effects, scope, move |_payload| {
                listener_entered(&trace, &plugin, &listener.tag);
            })
            .map(|_| ())
            .map_err(|error| error.to_string()),
        DynamicEventSpec::Parallel(spec) => bus
            .on_parallel(spec, &effects, scope, move |_payload| {
                let trace = Arc::clone(&trace);
                let plugin = plugin.clone();
                let tag = listener.tag.clone();
                async move {
                    listener_entered(&trace, &plugin, &tag);
                    Ok(())
                }
            })
            .map(|_| ())
            .map_err(|error| error.to_string()),
        DynamicEventSpec::Serial(spec) => bus
            .on_serial(spec, &effects, scope, move |_payload| {
                let trace = Arc::clone(&trace);
                let plugin = plugin.clone();
                let tag = listener.tag.clone();
                let returned = listener.returns.clone();
                async move {
                    listener_entered(&trace, &plugin, &tag);
                    returned
                }
            })
            .map(|_| ())
            .map_err(|error| error.to_string()),
        DynamicEventSpec::Waterfall(spec) => bus
            .on_waterfall(spec, &effects, scope, move |payload, next| {
                let trace = Arc::clone(&trace);
                let plugin = plugin.clone();
                let listener = listener.clone();
                async move {
                    listener_entered(&trace, &plugin, &listener.tag);
                    match listener.action.as_str() {
                        "delegate" => next.call(None).await,
                        "transform" => next.call(Some(vec![listener.replacement])).await,
                        "delegate_twice" => {
                            let _ = next.call(None).await?;
                            next.call(None).await
                        }
                        "echo_args" => Ok(Value::Array(payload)),
                        "short_circuit" | "observe" => Ok(listener.returns),
                        action => panic!("unsupported scripted action {action}"),
                    }
                }
            })
            .map(|_| ())
            .map_err(|error| error.to_string()),
    }
}

async fn execute_during_load(step: &Step, shared: &Shared) -> Result<(), PluginInitError> {
    let Step::Unmount(step) = step else {
        return Err(PluginInitError::new(
            "only unmount is supported inside canonical during_load",
        ));
    };
    let fiber = shared
        .fibers
        .lock()
        .unwrap()
        .get(&step.unmount)
        .cloned()
        .ok_or_else(|| PluginInitError::new(format!("unknown mounted plugin {}", step.unmount)))?;
    shared.runtime.unmount(&fiber).await.map_err(init_error)
}

async fn dispatch(step: &DispatchStep, shared: &Shared) -> Result<Value, String> {
    let spec = shared
        .event_specs
        .get(&step.event)
        .ok_or_else(|| format!("undeclared event {}", step.event))?;
    let scope = step
        .scope
        .as_ref()
        .map(|name| {
            shared
                .scopes
                .lock()
                .unwrap()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unknown scope {name}"))
        })
        .transpose()?;
    match spec {
        DynamicEventSpec::Emit(spec) => shared
            .runtime
            .events()
            .emit(spec, &step.args, scope.as_ref())
            .map(|()| Value::Null)
            .map_err(|error| error.to_string()),
        DynamicEventSpec::Parallel(spec) => shared
            .runtime
            .events()
            .parallel(spec, step.args.clone(), scope.as_ref())
            .await
            .map(|()| Value::Null)
            .map_err(|error| error.to_string()),
        DynamicEventSpec::Serial(spec) => shared
            .runtime
            .events()
            .serial(spec, step.args.clone(), scope.as_ref())
            .await
            .map(|value| value.unwrap_or(Value::Null))
            .map_err(|error| error.to_string()),
        DynamicEventSpec::Waterfall(spec) => shared
            .runtime
            .events()
            .waterfall(spec, step.args.clone(), scope.as_ref())
            .await
            .map_err(|error| error.to_string()),
    }
}

fn listener_entered(trace: &Mutex<Vec<Value>>, plugin: &str, tag: &str) {
    trace.lock().unwrap().push(json!({
        "event": "listener_entered",
        "plugin": plugin,
        "tag": tag,
    }));
}

fn payload_value(payload: &[Value]) -> Value {
    match payload {
        [only] => only.clone(),
        values => Value::Array(values.to_vec()),
    }
}

fn compare_outcome(
    scenario: &Scenario,
    trace: &Mutex<Vec<Value>>,
    result: Option<Value>,
    error: Option<String>,
) -> TestResult {
    let actual_trace = trace.lock().unwrap().clone();
    if actual_trace != scenario.expect_trace {
        return Err(format!(
            "{} trace mismatch\nexpected: {:#?}\nactual: {actual_trace:#?}",
            scenario.name, scenario.expect_trace
        ));
    }
    match (&scenario.expect_error, error) {
        (Some(expected), Some(actual)) => {
            let actual_kind = normalize_error_kind(&actual);
            if actual_kind != expected.kind {
                return Err(format!(
                    "{} error kind: expected {}, got {} ({actual})",
                    scenario.name, expected.kind, actual_kind
                ));
            }
            if let Some(fragment) = &expected.message_contains
                && !actual.contains(fragment)
            {
                return Err(format!(
                    "{} error did not contain {fragment:?}: {actual}",
                    scenario.name
                ));
            }
        }
        (Some(expected), None) => {
            return Err(format!(
                "{} expected error {}",
                scenario.name, expected.kind
            ));
        }
        (None, Some(actual)) => {
            return Err(format!("{} unexpected error: {actual}", scenario.name));
        }
        (None, None) => {}
    }
    if let Some(expected) = &scenario.expect_result {
        let actual = result.unwrap_or(Value::Null);
        if &actual != expected {
            return Err(format!(
                "{} result mismatch: expected {expected}, got {actual}",
                scenario.name
            ));
        }
    }
    Ok(())
}

fn normalize_error_kind(message: &str) -> String {
    if message.contains("already provided") {
        "ServiceConflictError"
    } else if message.contains("inactive owner") || message.contains("disposed-owner") {
        "InactiveFiberError"
    } else if message.contains("no active provider") {
        "ServiceNotFoundError"
    } else if message.contains("mode") {
        "EventModeError"
    } else if message.contains("at most once") {
        "WaterfallError"
    } else {
        "UnknownError"
    }
    .to_owned()
}

fn state_name(state: FiberState) -> &'static str {
    match state {
        FiberState::Pending => "pending",
        FiberState::Loading => "loading",
        FiberState::Active => "active",
        FiberState::Failed => "failed",
        FiberState::Unloading => "unloading",
        FiberState::Disposed => "disposed",
    }
}
