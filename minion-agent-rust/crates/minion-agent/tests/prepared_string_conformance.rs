#![cfg(feature = "conformance")]

use minion_agent::{
    PluginInitError, PluginSpec, Runtime,
    llm::{StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        AgentToolResult, BeforeToolCallAction, PreparedString, PreparedValue, ToolCapabilityError,
        ToolDefinition, ToolExecutionOptions, ToolExecutionRequest, execute_tool_calls,
        register_before_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    sync::Arc,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
fn units(value: &Value) -> Vec<u16> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|n| u16::try_from(n.as_u64().unwrap()).unwrap())
        .collect()
}
fn decode(value: &Value) -> PreparedValue {
    match value {
        Value::Object(o) if o.contains_key("utf16") => {
            PreparedValue::String(PreparedString::from_code_units(units(&o["utf16"])))
        }
        Value::Object(o) if o.contains_key("$key") => PreparedValue::Object(BTreeMap::from([(
            PreparedString::from_code_units(units(&o["$key"])),
            decode(&o["value"]),
        )])),
        Value::Object(o) => PreparedValue::Object(
            o.iter()
                .map(|(k, v)| (k.as_str().into(), decode(v)))
                .collect(),
        ),
        Value::Array(a) => PreparedValue::Array(a.iter().map(decode).collect()),
        _ => PreparedValue::from(value.clone()),
    }
}
fn set(mut value: PreparedValue, replacements: &Value) -> PreparedValue {
    for (pointer, replacement) in replacements.as_object().unwrap() {
        value[pointer.strip_prefix('/').unwrap()] = decode(replacement);
    }
    value
}
fn at<'a>(value: &'a PreparedValue, pointer: &str) -> Option<&'a PreparedValue> {
    let mut node = value;
    for key in pointer.strip_prefix('/')?.split('/') {
        let key = key.replace("~1", "/").replace("~0", "~");
        node = match node {
            PreparedValue::Array(a) => a.get(key.parse::<usize>().ok()?)?,
            _ => node.get(&key)?,
        };
    }
    Some(node)
}
fn observe(value: &PreparedValue, pointers: &Value, keys: &Value) -> Option<Value> {
    let mut values = serde_json::Map::new();
    let mut observed_keys = serde_json::Map::new();
    for pointer in pointers.as_array()? {
        let p = pointer.as_str()?;
        let PreparedValue::String(s) = at(value, p)? else {
            return None;
        };
        values.insert(p.to_owned(), json!(s.code_units()));
    }
    if let Some(keys) = keys.as_array() {
        for pointer in keys {
            let p = pointer.as_str()?;
            observed_keys.insert(
                p.to_owned(),
                json!(
                    at(value, p)?
                        .as_object()?
                        .keys()
                        .map(PreparedString::code_units)
                        .collect::<Vec<_>>()
                ),
            );
        }
    }
    Some(json!({"values":values,"keys":observed_keys}))
}
#[derive(Clone, Copy, Debug)]
enum Mutation {
    None,
    Reject,
    Replace,
    StrictUtf8,
    PairCorruption,
    LowOnly,
    KeyReplace,
    HookReplace,
    ExecuteReplace,
}
fn mutate(value: PreparedValue, mutation: Mutation) -> Result<PreparedValue, ToolCapabilityError> {
    let string = |s: PreparedString| -> Result<PreparedString, ToolCapabilityError> {
        if s.as_str().is_none() && matches!(mutation, Mutation::Reject | Mutation::StrictUtf8) {
            return Err(ToolCapabilityError::new("scalar-only string boundary"));
        }
        if matches!(mutation, Mutation::PairCorruption) {
            return Ok(PreparedString::from_code_units(
                s.code_units()
                    .iter()
                    .map(|u| {
                        if (0xd800..=0xdfff).contains(u) {
                            0xfffd
                        } else {
                            *u
                        }
                    })
                    .collect(),
            ));
        }
        if matches!(mutation, Mutation::LowOnly) {
            return Ok(PreparedString::from_code_units(
                s.code_units()
                    .iter()
                    .enumerate()
                    .map(|(i, u)| {
                        if (0xdc00..=0xdfff).contains(u)
                            && (i == 0 || !(0xd800..=0xdbff).contains(&s.code_units()[i - 1]))
                        {
                            0xfffd
                        } else {
                            *u
                        }
                    })
                    .collect(),
            ));
        }
        Ok(s.to_utf8_lossy().into())
    };
    match value {
        PreparedValue::String(s) if !matches!(mutation, Mutation::None | Mutation::KeyReplace) => {
            string(s).map(PreparedValue::String)
        }
        PreparedValue::Array(a) => a
            .into_iter()
            .map(|v| mutate(v, mutation))
            .collect::<Result<Vec<_>, _>>()
            .map(PreparedValue::Array),
        PreparedValue::Object(o) => o
            .into_iter()
            .map(|(k, v)| {
                Ok((
                    if matches!(mutation, Mutation::None) {
                        k
                    } else {
                        string(k)?
                    },
                    mutate(v, mutation)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(PreparedValue::Object),
        _ => Ok(value),
    }
}
fn schema(name: &str) -> Value {
    let constraint = match name {
        "open" => return json!({"type":"object","properties":{}}),
        "string" => json!({"type":"string"}),
        "min-length-2" => json!({"type":"string","minLength":2}),
        "max-length-1" => json!({"type":"string","maxLength":1}),
        "pattern-one-char" => json!({"type":"string","pattern":"^.$"}),
        "pattern-two-chars" => json!({"type":"string","pattern":"^..$"}),
        "const-pair" => json!({"const":"😀"}),
        "enum-fffd" => json!({"enum":["�"]}),
        _ => panic!("unknown schema {name}"),
    };
    let mut s = json!({"type":"object","properties":{"text":constraint}});
    if name == "string" {
        s["required"] = json!(["text"]);
    }
    s
}
fn result() -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new("ok").into())],
        details: json!({}).into(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}
fn run_case(case: &Value, mutation: Mutation) -> bool {
    assert_eq!(case["tool"], "custom");
    let runtime = Runtime::new();
    let hook = Arc::new(Mutex::new(None));
    let executed = Arc::new(Mutex::new(None));
    let execute_observation = executed.clone();
    let execute_case = case.clone();
    let has_replacement = case.get("hook_replace_set").is_some();
    let definition = ToolDefinition::new(
        "probe",
        "probe",
        serde_json::from_value(schema(case["schema"].as_str().unwrap())).unwrap(),
        "probe",
        move |request: ToolExecutionRequest| {
            let mut params = request.params;
            if matches!(mutation, Mutation::ExecuteReplace) {
                params = mutate(params, Mutation::Replace).unwrap();
            }
            let values = execute_case
                .get("execute_observe")
                .unwrap_or(&execute_case["observe"]);
            let keys = execute_case
                .get("execute_observe_keys")
                .or_else(|| {
                    if has_replacement {
                        None
                    } else {
                        execute_case.get("observe_keys")
                    }
                })
                .unwrap_or(&Value::Null);
            *execute_observation.lock() = observe(&params, values, keys);
            Box::pin(async { Ok(result()) })
        },
    )
    .with_prepare_runtime_arguments({
        let set_value = case["prepare_set"].clone();
        move |raw| {
            let prepared = set(PreparedValue::from(raw), &set_value);
            mutate(
                prepared,
                if matches!(mutation, Mutation::HookReplace | Mutation::ExecuteReplace) {
                    Mutation::None
                } else {
                    mutation
                },
            )
        }
    });
    runtime
        .tools()
        .register_for_scope(None, definition)
        .unwrap();
    let hook_case = case.clone();
    let seen = hook.clone();
    let plugin = PluginSpec::<Value>::new(
        "prepared-string-observer",
        vec![],
        || json!({}),
        move |context, _| {
            let case = hook_case.clone();
            let seen = seen.clone();
            async move {
                register_before_tool_call_hook(&context, move |current| {
                    let case = case.clone();
                    let seen = seen.clone();
                    async move {
                        let observed = if matches!(mutation, Mutation::HookReplace) {
                            mutate(current.arguments.clone(), Mutation::Replace)?
                        } else {
                            current.arguments.clone()
                        };
                        *seen.lock() = observe(&observed, &case["observe"], &case["observe_keys"]);
                        Ok(BeforeToolCallAction::Proceed(
                            case.get("hook_replace_set")
                                .map(|replacements| set(current.arguments, replacements)),
                        ))
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
        "call-1",
        "probe",
        serde_json::from_value(raw.clone()).unwrap(),
    );
    let calls = vec![call];
    let batch = executor
        .block_on(execute_tool_calls(
            &runtime.context(),
            &calls,
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        ))
        .unwrap();
    assert_eq!(serde_json::to_value(&calls[0].arguments).unwrap(), raw);
    let expected = &case["expect"];
    if expected["outcome"] == "argument_validation_failure" {
        return batch.messages[0].is_error && hook.lock().is_none() && executed.lock().is_none();
    }
    let expected_hook = json!({"values":expected["observed"],"keys":expected.get("observed_keys").cloned().unwrap_or(json!({}))});
    let expected_execute = if has_replacement {
        json!({"values":expected["execute_observed"],"keys":expected.get("execute_observed_keys").cloned().unwrap_or(json!({}))})
    } else {
        expected_hook.clone()
    };
    !batch.messages[0].is_error
        && hook.lock().as_ref() == Some(&expected_hook)
        && executed.lock().as_ref() == Some(&expected_execute)
}
fn preflight(case: &Value) -> Result<(), String> {
    if case["expect"]["outcome"] != "prepared" {
        return Ok(());
    }
    let expected = &case["expect"];
    for (expected_key, pointers_key) in [
        ("observed", "observe"),
        ("observed_keys", "observe_keys"),
        ("execute_observed_keys", "execute_observe_keys"),
    ] {
        let actual = expected
            .get(expected_key)
            .and_then(Value::as_object)
            .map(|o| o.keys().cloned().collect::<BTreeSet<_>>())
            .unwrap_or_default();
        let pointers = case
            .get(pointers_key)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|p| p.as_str().unwrap().to_owned())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if actual != pointers {
            return Err(format!("{}: {expected_key}", case["id"]));
        }
    }
    let replaced = case.get("hook_replace_set").is_some();
    if replaced != expected.get("execute_observed").is_some() {
        return Err("replacement observation presence".into());
    }
    if replaced {
        let actual = expected["execute_observed"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let pointers = case["execute_observe"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        if actual != pointers {
            return Err("execute observation pointers".into());
        }
    }
    Ok(())
}
fn cases() -> Vec<Value> {
    let schema: Value = serde_json::from_str(
        &fs::read_to_string(root().join("conformance/schema/prepared-string-scenario.schema.json"))
            .unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mut files = fs::read_dir(root().join("conformance/agent/prepared-runtime-string"))
        .unwrap()
        .map(|f| f.unwrap().path())
        .collect::<Vec<_>>();
    files.sort();
    files
        .into_iter()
        .flat_map(|path| {
            let document: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
            validator.validate(&document).unwrap();
            if document["gate"] == "L0506-D002" {
                document["prepared_string"]["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .inspect(|case| preflight(case).unwrap())
                    .cloned()
                    .collect()
            } else {
                vec![]
            }
        })
        .collect()
}
#[test]
fn delta_canonical_through_real_preparation_hooks_and_execute() {
    let cases = cases();
    assert!(!cases.is_empty());
    eprintln!("prepared string delta gate: {} cases", cases.len());
    for case in cases {
        assert!(run_case(&case, Mutation::None), "{}", case["id"]);
    }
}
#[test]
fn wrong_string_boundaries_are_discriminated() {
    let cases = cases();
    for mutation in [
        Mutation::Reject,
        Mutation::Replace,
        Mutation::StrictUtf8,
        Mutation::PairCorruption,
        Mutation::LowOnly,
        Mutation::KeyReplace,
        Mutation::HookReplace,
        Mutation::ExecuteReplace,
    ] {
        assert!(
            cases.iter().any(|case| !run_case(case, mutation)),
            "surviving mutant {mutation:?}"
        );
    }
}
