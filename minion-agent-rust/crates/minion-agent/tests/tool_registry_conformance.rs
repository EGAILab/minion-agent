#![cfg(feature = "conformance")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    sync::Arc,
};

use minion_agent::{
    DynPluginSpec, FiberHandle, PluginInitError, PluginSpec, RegistrationHandle, Runtime,
    ScopeHandle,
    llm::{ConstrainedSampling, JsonSchemaObject},
    tools::{ExecutionMode, ToolDefinition, ToolExecutionRequest},
};
use parking_lot::Mutex;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn object<'a>(value: &'a Value, field: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing object field {field}"))
}

fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field {field}"))
}

fn validate_references(document: &Value) -> Result<(), String> {
    let registry = object(document, "tool_registry")?;
    let plugins = registry["plugins"]
        .as_array()
        .ok_or("plugins must be an array")?;
    let plugin_ids = plugins
        .iter()
        .map(|plugin| string(plugin, "id").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if plugin_ids.len() != plugins.len() {
        return Err("duplicate plugin id".into());
    }
    let scopes = plugins
        .iter()
        .filter_map(|plugin| plugin.get("scope").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    for plugin in plugins {
        if let Some(parent) = plugin.get("scope_parent").and_then(Value::as_str)
            && !scopes.contains(parent)
        {
            return Err(format!("unknown parent scope {parent}"));
        }
        if plugin.get("scope").and_then(Value::as_str).is_some()
            && plugin.get("scope").and_then(Value::as_str)
                == plugin.get("scope_parent").and_then(Value::as_str)
        {
            return Err("scope cannot parent itself".into());
        }
    }
    let parents = plugins
        .iter()
        .filter_map(|plugin| {
            Some((
                plugin.get("scope")?.as_str()?,
                plugin.get("scope_parent")?.as_str()?,
            ))
        })
        .collect::<BTreeMap<_, _>>();
    for start in &scopes {
        let mut seen = BTreeSet::new();
        let mut current = Some(*start);
        while let Some(scope) = current {
            if !seen.insert(scope) {
                return Err(format!("scope parent cycle at {scope}"));
            }
            current = parents.get(scope).copied();
        }
    }
    for step in registry["steps"]
        .as_array()
        .ok_or("steps must be an array")?
    {
        let step = step.as_object().ok_or("step must be an object")?;
        for operation in ["mount", "unmount", "withdraw"] {
            if let Some(plugin) = step.get(operation).and_then(Value::as_str)
                && !plugin_ids.contains(plugin)
            {
                return Err(format!("unknown {operation} plugin {plugin}"));
            }
        }
        if let Some(scope) = step.get("dispose_scope").and_then(Value::as_str)
            && !scopes.contains(scope)
        {
            return Err(format!("unknown disposal scope {scope}"));
        }
    }
    for query in registry["queries"]
        .as_array()
        .ok_or("queries must be an array")?
    {
        if let Some(scope) = query.get("scope").and_then(Value::as_str)
            && !scopes.contains(scope)
        {
            return Err(format!("unknown query scope {scope}"));
        }
    }
    Ok(())
}

fn parse_tool(value: &Value) -> Result<ToolDefinition, String> {
    let parameters: JsonSchemaObject = serde_json::from_value(
        value
            .get("parameters")
            .cloned()
            .ok_or("missing parameters")?,
    )
    .map_err(|error| error.to_string())?;
    let mut tool = ToolDefinition::new(
        string(value, "name")?,
        string(value, "description")?,
        parameters,
        string(value, "label")?,
        |_request: ToolExecutionRequest| {
            Box::pin(async { panic!("Layer 05 canonical runner must never execute a tool") })
        },
    );
    if let Some(sampling) = value.get("constrained_sampling") {
        if sampling.is_null() {
            return Err("explicit null constrained_sampling input is invalid".into());
        }
        tool = tool.with_constrained_sampling(
            serde_json::from_value::<ConstrainedSampling>(sampling.clone())
                .map_err(|error| error.to_string())?,
        );
    }
    if let Some(mode) = value.get("execution_mode").and_then(Value::as_str) {
        tool = tool.with_execution_mode(match mode {
            "parallel" => ExecutionMode::Parallel,
            "sequential" => ExecutionMode::Sequential,
            other => return Err(format!("unknown execution_mode {other}")),
        });
    }
    Ok(tool)
}

struct PluginFixture {
    spec: DynPluginSpec,
    scope: Option<String>,
    registrations: Arc<Mutex<Vec<RegistrationHandle>>>,
}

fn plugin_fixture(value: &Value) -> Result<PluginFixture, String> {
    let id = string(value, "id")?.to_owned();
    let tools = value["tools"]
        .as_array()
        .ok_or("plugin tools must be an array")?
        .iter()
        .map(parse_tool)
        .collect::<Result<Vec<_>, _>>()?;
    let registrations = Arc::new(Mutex::new(Vec::new()));
    let registration_sink = Arc::clone(&registrations);
    let spec = PluginSpec::<Value>::new(
        id,
        vec![],
        || serde_json::json!({}),
        move |context, _config| {
            let tools = tools.clone();
            let registrations = Arc::clone(&registration_sink);
            async move {
                let registry = context
                    .tools()
                    .map_err(|error| PluginInitError::new(error.to_string()))?;
                for tool in tools {
                    registrations.lock().push(
                        registry
                            .register(&context, tool)
                            .map_err(|error| PluginInitError::new(error.to_string()))?,
                    );
                }
                Ok(())
            }
        },
    )
    .erase();
    Ok(PluginFixture {
        spec,
        scope: value
            .get("scope")
            .and_then(Value::as_str)
            .map(str::to_owned),
        registrations,
    })
}

fn create_scopes(
    runtime: &Runtime,
    plugins: &[Value],
) -> Result<BTreeMap<String, ScopeHandle>, String> {
    let mut pending = plugins
        .iter()
        .filter_map(|plugin| {
            Some((
                plugin.get("scope")?.as_str()?.to_owned(),
                plugin
                    .get("scope_parent")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let mut scopes = BTreeMap::new();
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .find(|(_, parent)| {
                parent
                    .as_ref()
                    .is_none_or(|parent| scopes.contains_key(parent))
            })
            .map(|(scope, _)| scope.clone())
            .ok_or("scope declarations contain a cycle")?;
        let parent = pending.remove(&ready).unwrap();
        let handle = runtime
            .create_scope(parent.as_ref().and_then(|parent| scopes.get(parent)))
            .map_err(|error| error.to_string())?;
        scopes.insert(ready, handle);
    }
    Ok(scopes)
}

fn run_scenario(document: &Value) -> Result<Value, String> {
    validate_references(document)?;
    let registry_spec = object(document, "tool_registry")?;
    let plugin_values = registry_spec["plugins"]
        .as_array()
        .ok_or("plugins must be an array")?;
    let runtime = Runtime::new();
    let scopes = create_scopes(&runtime, plugin_values)?;
    let fixtures = plugin_values
        .iter()
        .map(|plugin| Ok((string(plugin, "id")?.to_owned(), plugin_fixture(plugin)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut fibers = BTreeMap::<String, FiberHandle>::new();

    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .map_err(|error| error.to_string())?;
    for step in registry_spec["steps"]
        .as_array()
        .ok_or("steps must be an array")?
    {
        let step = step.as_object().ok_or("step must be an object")?;
        if let Some(id) = step.get("mount").and_then(Value::as_str) {
            let fixture = &fixtures[id];
            let fiber = runtime
                .mount_in(
                    &fixture.spec,
                    serde_json::json!({}),
                    fixture
                        .scope
                        .as_ref()
                        .and_then(|scope| scopes.get(scope))
                        .cloned(),
                )
                .map_err(|error| error.to_string())?;
            tokio
                .block_on(runtime.reconcile())
                .map_err(|error| format!("{error:?}"))?;
            fibers.insert(id.to_owned(), fiber);
        } else if let Some(id) = step.get("unmount").and_then(Value::as_str) {
            tokio
                .block_on(runtime.unmount(&fibers[id]))
                .map_err(|error| format!("{error:?}"))?;
        } else if let Some(id) = step.get("withdraw").and_then(Value::as_str) {
            for registration in fixtures[id].registrations.lock().iter() {
                registration.withdraw();
            }
        } else if let Some(scope) = step.get("dispose_scope").and_then(Value::as_str) {
            tokio
                .block_on(scopes[scope].dispose())
                .map_err(|error| format!("{error:?}"))?;
        }
    }

    let expected = object(document, "expect")?;
    let mut actual = serde_json::Map::new();
    for query in registry_spec["queries"]
        .as_array()
        .ok_or("queries must be an array")?
    {
        let id = string(query, "id")?;
        let expected_query = expected[id]
            .as_object()
            .ok_or("expected query must be object")?;
        let scope = query
            .get("scope")
            .and_then(Value::as_str)
            .and_then(|scope| scopes.get(scope));
        let mut observation = serde_json::Map::new();
        observation.insert(
            "names".into(),
            Value::Array(
                runtime
                    .tools()
                    .visible(scope)
                    .into_iter()
                    .map(|tool| Value::String(tool.name().to_owned()))
                    .collect(),
            ),
        );
        if let Some(names) = query.get("resolve").and_then(Value::as_array) {
            let mut resolved = serde_json::Map::new();
            for name in names.iter().map(|name| name.as_str().unwrap()) {
                let value = runtime.tools().resolve(name, scope).map_or_else(
                    || serde_json::json!({"found": false}),
                    |tool| serde_json::json!({"found": true, "label": tool.label()}),
                );
                resolved.insert(name.to_owned(), value);
            }
            observation.insert("resolve".into(), Value::Object(resolved));
        }
        if expected_query.contains_key("schemas") {
            observation.insert(
                "schemas".into(),
                Value::Array(
                    runtime
                        .tools()
                        .schemas(scope)
                        .iter()
                        .map(|schema| schema.as_json())
                        .collect(),
                ),
            );
        }
        actual.insert(id.to_owned(), Value::Object(observation));
    }
    Ok(Value::Object(actual))
}

#[test]
fn all_layer_05_scenarios_drive_the_real_runtime_tool_registry() {
    let mut scenarios = fs::read_dir(root().join("conformance/agent"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("tool-registry-") && name.ends_with(".yaml"))
        })
        .map(|path| {
            let document: Value =
                serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            (path, document)
        })
        .collect::<Vec<_>>();
    scenarios.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(scenarios.len(), 9);

    for (path, document) in scenarios {
        let actual =
            run_scenario(&document).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert_eq!(actual, document["expect"], "{}", path.display());
    }
}

fn validation_document(plugins: Value, steps: Value, queries: Value) -> Value {
    serde_json::json!({
        "tool_registry": {"plugins": plugins, "steps": steps, "queries": queries},
        "expect": {},
    })
}

#[test]
fn unknown_scope_parent_is_rejected_before_runtime_construction() {
    let document = validation_document(
        serde_json::json!([{"id": "p", "scope": "child", "scope_parent": "missing"}]),
        serde_json::json!([]),
        serde_json::json!([]),
    );
    assert_eq!(
        validate_references(&document).unwrap_err(),
        "unknown parent scope missing"
    );
}

#[test]
fn unknown_query_scope_is_rejected_directly() {
    let document = validation_document(
        serde_json::json!([{"id": "p"}]),
        serde_json::json!([]),
        serde_json::json!([{"id": "q", "scope": "missing"}]),
    );
    assert_eq!(
        validate_references(&document).unwrap_err(),
        "unknown query scope missing"
    );
}

#[test]
fn self_parent_is_rejected_directly() {
    let document = validation_document(
        serde_json::json!([{"id": "p", "scope": "same", "scope_parent": "same"}]),
        serde_json::json!([]),
        serde_json::json!([]),
    );
    assert_eq!(
        validate_references(&document).unwrap_err(),
        "scope cannot parent itself"
    );
}

#[test]
fn scope_parent_cycle_is_rejected_directly() {
    let document = validation_document(
        serde_json::json!([
            {"id": "a", "scope": "a", "scope_parent": "b"},
            {"id": "b", "scope": "b", "scope_parent": "a"}
        ]),
        serde_json::json!([]),
        serde_json::json!([]),
    );
    assert!(
        validate_references(&document)
            .unwrap_err()
            .contains("scope parent cycle")
    );
}

#[test]
fn unknown_mount_plugin_is_rejected_directly() {
    let document = validation_document(
        serde_json::json!([{"id": "p"}]),
        serde_json::json!([{"mount": "missing"}]),
        serde_json::json!([]),
    );
    assert_eq!(
        validate_references(&document).unwrap_err(),
        "unknown mount plugin missing"
    );
}

#[test]
fn unknown_disposal_scope_is_rejected_directly() {
    let document = validation_document(
        serde_json::json!([{"id": "p"}]),
        serde_json::json!([{"dispose_scope": "missing"}]),
        serde_json::json!([]),
    );
    assert_eq!(
        validate_references(&document).unwrap_err(),
        "unknown disposal scope missing"
    );
}

#[test]
fn unknown_withdrawal_plugin_is_rejected_directly() {
    let document = validation_document(
        serde_json::json!([{"id": "p"}]),
        serde_json::json!([{"withdraw": "missing"}]),
        serde_json::json!([]),
    );
    assert_eq!(
        validate_references(&document).unwrap_err(),
        "unknown withdraw plugin missing"
    );
}
