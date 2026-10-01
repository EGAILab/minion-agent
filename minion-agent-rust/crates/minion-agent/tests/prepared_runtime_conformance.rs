#![cfg(feature = "conformance")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    sync::Arc,
};

use minion_agent::{
    PluginInitError, PluginSpec, Runtime,
    llm::{StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        AgentToolResult, BeforeToolCallAction, PreparedValue, ToolCapabilityError, ToolDefinition,
        ToolExecutionOptions, ToolExecutionRequest, execute_tool_calls,
        register_before_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn token(value: &str) -> Result<f64, String> {
    match value {
        "+Infinity" => Ok(f64::INFINITY),
        "-Infinity" => Ok(f64::NEG_INFINITY),
        "NaN" => Ok(f64::NAN),
        "-0" => Ok(-0.0),
        literal => {
            let value: f64 = literal
                .parse()
                .map_err(|_| format!("invalid token {literal}"))?;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(format!("non-finite literal {literal}"))
            }
        }
    }
}

fn observed(
    value: &PreparedValue,
    pointers: &[String],
) -> Result<BTreeMap<String, String>, String> {
    pointers
        .iter()
        .map(|p| {
            let mut at = value;
            for part in p
                .strip_prefix('/')
                .ok_or_else(|| format!("invalid pointer {p}"))?
                .split('/')
            {
                let key = part.replace("~1", "/").replace("~0", "~");
                at = match at {
                    PreparedValue::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
                    _ => at.get(&key),
                }
                .ok_or_else(|| format!("missing pointer {p}"))?;
            }
            let n = at
                .as_f64()
                .ok_or_else(|| format!("not a runtime number at {p}: {at:?}"))?;
            let text = if n.is_nan() {
                "NaN".into()
            } else if n == f64::INFINITY {
                "+Infinity".into()
            } else if n == f64::NEG_INFINITY {
                "-Infinity".into()
            } else if n == 0.0 && n.is_sign_negative() {
                "-0".into()
            } else {
                ryu_js::Buffer::new().format(n).to_owned()
            };
            Ok((p.clone(), text))
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Mutation {
    None,
    RejectNonFinite,
    NullNonFinite,
    ClampNonFinite,
    StringNonFinite,
    LoseZeroSign,
    LoseAtHook,
    ReplaceAtHook,
}

fn mutate(value: PreparedValue, mutation: Mutation) -> Result<PreparedValue, ToolCapabilityError> {
    match value {
        PreparedValue::Number(_) => {
            let n = value.as_f64().unwrap();
            match mutation {
                Mutation::RejectNonFinite if !n.is_finite() => Err(ToolCapabilityError::new(
                    "JSON-only runtime rejects non-finite",
                )),
                Mutation::NullNonFinite if !n.is_finite() => Ok(PreparedValue::Null),
                Mutation::ClampNonFinite if !n.is_finite() => Ok(PreparedValue::number(f64::MAX)),
                Mutation::StringNonFinite if !n.is_finite() => {
                    Ok(PreparedValue::String(n.to_string()))
                }
                Mutation::LoseZeroSign if n == 0.0 => Ok(PreparedValue::number(0.0)),
                _ => Ok(value),
            }
        }
        PreparedValue::Array(a) => a
            .into_iter()
            .map(|v| mutate(v, mutation))
            .collect::<Result<Vec<_>, _>>()
            .map(PreparedValue::Array),
        PreparedValue::Object(o) => o
            .into_iter()
            .map(|(k, v)| mutate(v, mutation).map(|v| (k, v)))
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(PreparedValue::Object),
        _ => Ok(value),
    }
}

fn result() -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(
            "probe success",
        ))],
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

struct Observation {
    error: bool,
    hook: Option<PreparedValue>,
    execute: Option<PreparedValue>,
    raw_events: Vec<Value>,
}

fn run_case(case: &Value, mutation: Mutation) -> Observation {
    assert_eq!(case["tool"], "custom");
    let runtime = Runtime::new();
    let hook = Arc::new(Mutex::new(None));
    let executed = Arc::new(Mutex::new(None));
    let execute_observation = executed.clone();
    let schema = match case["schema"].as_str().unwrap() {
        "open" => json!({"type":"object","properties":{}}),
        ty @ ("number" | "integer") => {
            json!({"type":"object","properties":{"limit":{"type":ty}},"required":["limit"]})
        }
        kind => {
            let constraint = match kind {
                "bound-maximum" => json!({"maximum":0}),
                "bound-minimum" => json!({"minimum":0}),
                "bound-exclusive-maximum" => json!({"exclusiveMaximum":0}),
                "bound-exclusive-minimum" => json!({"exclusiveMinimum":0}),
                "multiple-of" => json!({"multipleOf":2}),
                "one-of-bounds" => json!({"oneOf":[{"maximum":0},{"minimum":1}]}),
                "not-bound" => json!({"not":{"maximum":0}}),
                "number-bound" => json!({"type":"number","maximum":0}),
                other => panic!("unknown schema {other}"),
            };
            json!({"type":"object","properties":{"limit":constraint}})
        }
    };
    let set = case["prepare_set"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), token(v.as_str().unwrap()).unwrap()))
        .collect::<Vec<_>>();
    let definition = ToolDefinition::new(
        "probe",
        "probe",
        serde_json::from_value(schema).unwrap(),
        "probe",
        move |request: ToolExecutionRequest| {
            *execute_observation.lock() = Some(request.params);
            if let Some(update) = request.on_update {
                update(result());
            }
            Box::pin(async { Ok(result()) })
        },
    )
    .with_prepare_runtime_arguments(move |raw| {
        let mut prepared = PreparedValue::from(raw);
        for (key, n) in &set {
            prepared[key.as_str()] = PreparedValue::number(*n);
        }
        mutate(prepared, mutation)
    });
    runtime
        .tools()
        .register_for_scope(None, definition)
        .unwrap();
    let hook_observation = hook.clone();
    let plugin = PluginSpec::<Value>::new(
        "prepared-observer",
        vec![],
        || json!({}),
        move |context, _| {
            let hook = hook_observation.clone();
            async move {
                register_before_tool_call_hook(&context, move |current| {
                    *hook.lock() = Some(current.arguments.clone());
                    async move {
                        let replacement = match mutation {
                            Mutation::LoseAtHook => {
                                Some(mutate(current.arguments, Mutation::NullNonFinite)?)
                            }
                            Mutation::ReplaceAtHook => {
                                let mut replacement = current.arguments;
                                replacement["extra"] = PreparedValue::number(f64::INFINITY);
                                Some(replacement)
                            }
                            _ => None,
                        };
                        Ok(BeforeToolCallAction::Proceed(replacement))
                    }
                })
                .map_err(|e| PluginInitError::new(e.to_string()))?;
                Ok(())
            }
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(runtime.reconcile()).unwrap();
    let raw = case["arguments"].clone();
    let call = ToolCall::new(
        "probe-1",
        "probe",
        serde_json::from_value(raw.clone()).unwrap(),
    );
    let calls = vec![call];
    let events = Arc::new(Mutex::new(Vec::new()));
    let starts = events.clone();
    let updates = events.clone();
    let options = ToolExecutionOptions::new(StopReason::ToolUse, 0.0)
        .with_execution_start(move |e| {
            starts.lock().push(e.arguments);
            async { Ok(()) }
        })
        .with_execution_update(move |e| {
            updates.lock().push(e.arguments);
            async { Ok(()) }
        });
    let batch = executor
        .block_on(execute_tool_calls(&runtime.context(), &calls, options))
        .unwrap();
    assert_eq!(batch.messages.len(), 1);
    assert_eq!(serde_json::to_value(&calls[0].arguments).unwrap(), raw);
    let raw_events = events.lock().clone();
    assert!(!raw_events.is_empty());
    assert!(
        raw_events.iter().all(|e| e == &raw),
        "prepared values must not replace raw lifecycle arguments"
    );
    Observation {
        error: batch.messages[0].is_error,
        hook: hook.lock().clone(),
        execute: executed.lock().clone(),
        raw_events,
    }
}

fn matches_expected(case: &Value, result: &Observation) -> bool {
    if case["expect"]["outcome"] == "argument_validation_failure" {
        return result.error && result.hook.is_none() && result.execute.is_none();
    }
    let pointers = case["observe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let expected =
        serde_json::from_value::<BTreeMap<String, String>>(case["expect"]["observed"].clone())
            .unwrap();
    !result.error
        && result
            .hook
            .as_ref()
            .and_then(|v| observed(v, &pointers).ok())
            .as_ref()
            == Some(&expected)
        && result
            .execute
            .as_ref()
            .and_then(|v| observed(v, &pointers).ok())
            .as_ref()
            == Some(&expected)
        && result.raw_events.len() == 2
}

fn documents() -> Vec<Value> {
    let schema: Value = serde_json::from_str(
        &fs::read_to_string(
            root().join("conformance/schema/prepared-runtime-scenario.schema.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mut files = fs::read_dir(root().join("conformance/agent/prepared-runtime"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
        .collect::<Vec<_>>();
    files.sort();
    files
        .into_iter()
        .map(|p| serde_yaml::from_str::<Value>(&fs::read_to_string(p).unwrap()).unwrap())
        .filter(|d| d["gate"] == "L0506-D001")
        .inspect(|d| {
            validator.validate(d).unwrap();
            preflight(d).unwrap();
        })
        .collect()
}

fn preflight(document: &Value) -> Result<(), String> {
    for case in document["prepared_runtime"]["cases"].as_array().unwrap() {
        for tokens in [&case["prepare_set"], &case["expect"]["observed"]] {
            if let Some(tokens) = tokens.as_object() {
                for t in tokens.values() {
                    token(t.as_str().unwrap())?;
                }
            }
        }
        if case["expect"]["outcome"] == "prepared" {
            let actual = case["expect"]["observed"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            let required = case["observe"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect::<BTreeSet<_>>();
            if actual != required {
                return Err("observed pointer set differs from observe".into());
            }
        }
    }
    Ok(())
}

#[test]
fn canonical_delta_gate_uses_real_preparation_validation_hooks_and_execute() {
    let documents = documents();
    let mut count = 0;
    let mut failures = Vec::new();
    for document in &documents {
        for case in document["prepared_runtime"]["cases"].as_array().unwrap() {
            let result = run_case(case, Mutation::None);
            if !matches_expected(case, &result) {
                failures.push(format!(
                    "{}: hook {:?}, execute {:?}, error {}",
                    case["id"], result.hook, result.execute, result.error
                ));
            }
            count += 1;
        }
    }
    eprintln!(
        "L0506-D001: {} discovered documents, {count} real pipeline cases executed; {} failed; WP-13.2 excluded by gate",
        documents.len(),
        failures.len()
    );
    assert!(count > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn five_preparation_mutants_and_hook_loss_are_killed_by_the_same_canonical_observations() {
    let documents = documents();
    for mutation in [
        Mutation::RejectNonFinite,
        Mutation::NullNonFinite,
        Mutation::ClampNonFinite,
        Mutation::StringNonFinite,
        Mutation::LoseZeroSign,
        Mutation::LoseAtHook,
    ] {
        let mut killed = false;
        for document in &documents {
            for case in document["prepared_runtime"]["cases"].as_array().unwrap() {
                killed |= !matches_expected(case, &run_case(case, mutation));
            }
        }
        assert!(killed, "canonical corpus failed to kill {mutation:?}");
    }
}

#[test]
fn preflight_rejects_wrong_pointer_sets_and_overflow_finite_tokens() {
    let mut doc = documents().remove(0);
    let case = doc["prepared_runtime"]["cases"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["expect"]["outcome"] == "prepared")
        .unwrap();
    case["expect"]["observed"] = json!({"/wrong":"0"});
    assert!(preflight(&doc).is_err());
    assert!(token("1e999").is_err());
    assert!(token("-1e999").is_err());
}

#[test]
fn pre_execute_replacement_carries_the_runtime_domain_without_revalidation_or_json_projection() {
    let docs = documents();
    let case = docs
        .iter()
        .flat_map(|d| d["prepared_runtime"]["cases"].as_array().unwrap())
        .find(|c| c["id"] == "undeclared-zero")
        .unwrap();
    let result = run_case(case, Mutation::ReplaceAtHook);
    assert!(!result.error);
    assert_eq!(result.hook.as_ref().unwrap()["extra"].as_f64(), Some(0.0));
    assert_eq!(
        result.execute.as_ref().unwrap()["extra"].as_f64(),
        Some(f64::INFINITY)
    );
    assert_eq!(result.raw_events, vec![json!({}), json!({})]);
}
