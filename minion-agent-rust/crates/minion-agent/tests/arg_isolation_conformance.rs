//! L0506-D005: the acyclic corpus through real preparation/validation/hooks/updates.
//! No validation algorithm, clone, ordering or expectation is implemented here.
#![cfg(feature = "conformance")]

use minion_agent::{
    PluginSpec, Runtime,
    llm::{RawValue, StopReason, ToolCall},
    tools::{
        AgentToolResult, BeforeToolCallAction, PreparedString, PreparedValue, ToolDefinition,
        ToolExecutionOptions, ToolExecutionRequest, execute_tool_calls,
        register_before_tool_call_hook, tool_execution_update_spec,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{fs, path::PathBuf, sync::Arc};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn number(n: f64) -> Value {
    let token = if n.is_nan() {
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
    json!({"n":token})
}

fn observe(v: &PreparedValue) -> Value {
    match v {
        PreparedValue::Null => Value::Null,
        PreparedValue::Bool(v) => json!(v),
        PreparedValue::Number(v) => number(v.as_f64()),
        PreparedValue::String(v) => json!({"u":v.code_units()}),
        PreparedValue::Array(v) => json!({"a":v.iter().map(|v| observe(&v)).collect::<Vec<_>>()}),
        PreparedValue::Object(v) => {
            json!({"o":v.iter().map(|(k,v)| json!([k.as_str().expect("scalar fixture key"), observe(&v)])).collect::<Vec<_>>()})
        }
    }
}

// Observe the actual raw handles, not a converted/ordered copy.
fn observe_raw(v: &RawValue) -> Value {
    match v {
        RawValue::Null => Value::Null,
        RawValue::Bool(v) => json!(v),
        RawValue::Number(v) => number(v.as_f64()),
        RawValue::String(v) => json!({"u":v.code_units()}),
        RawValue::Array(v) => json!({"a":v.iter().map(|v| observe_raw(&v)).collect::<Vec<_>>()}),
        RawValue::Object(v) => {
            json!({"o":v.iter().map(|(k,v)| json!([String::from_utf16(k.code_units()).expect("scalar fixture key"), observe_raw(&v)])).collect::<Vec<_>>()})
        }
    }
}

fn fixture(v: &Value) -> PreparedValue {
    match v {
        Value::Null => PreparedValue::Null,
        Value::Bool(v) => PreparedValue::Bool(*v),
        Value::Object(o) if o.contains_key("n") => {
            let text = o["n"].as_str().unwrap();
            PreparedValue::number(match text {
                "NaN" => f64::NAN,
                "+Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                _ => text.parse().unwrap(),
            })
        }
        Value::Object(o) if o.contains_key("u") => {
            PreparedValue::String(PreparedString::from_code_units(
                o["u"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
                    .collect(),
            ))
        }
        Value::Object(o) if o.contains_key("a") => {
            PreparedValue::Array(o["a"].as_array().unwrap().iter().map(fixture).collect())
        }
        Value::Object(o) if o.contains_key("o") => PreparedValue::Object(
            o["o"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (pair[0].as_str().unwrap().into(), fixture(&pair[1])))
                .collect(),
        ),
        _ => panic!("invalid fixture {v}"),
    }
}

fn at(root: &PreparedValue, path: &Value) -> PreparedValue {
    path.as_array().unwrap().iter().fold(root.clone(), |v, k| {
        if let Some(k) = k.as_str() {
            v.get(k).unwrap()
        } else {
            v.as_array()
                .unwrap()
                .get(usize::try_from(k.as_u64().unwrap()).unwrap())
                .unwrap()
        }
    })
}

fn same(a: &PreparedValue, b: &PreparedValue) -> bool {
    match (a, b) {
        (PreparedValue::Object(a), PreparedValue::Object(b)) => a.same_identity(b),
        (PreparedValue::Array(a), PreparedValue::Array(b)) => a.same_identity(b),
        _ => panic!("identity fact must name containers"),
    }
}

fn program(args: &PreparedValue, ops: &Value) {
    for op in ops.as_array().unwrap() {
        let target = at(args, &op["path"]);
        match op["op"].as_str().unwrap() {
            "set" => {
                target.set(op["key"].as_str().unwrap(), fixture(&op["value"]));
            }
            "push" => target.as_array().unwrap().push(fixture(&op["value"])),
            "delete" => {
                target
                    .as_object()
                    .unwrap()
                    .remove(&op["key"].as_str().unwrap().into());
            }
            other => panic!("unknown operation {other}"),
        }
    }
}

fn result() -> AgentToolResult {
    AgentToolResult {
        content: vec![],
        details: json!({}).into(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

#[derive(Default)]
struct Seen {
    entries: Vec<Value>,
    facts: Option<Vec<bool>>,
    executed: Option<Value>,
    updates: Vec<Value>,
    deliveries: Vec<Value>,
    handles: Vec<PreparedValue>,
    executed_handle: Option<PreparedValue>,
    // Rust raw/prepared containers have different concrete element types and
    // cannot share an allocation. Retain the shim's converted source too, to
    // test the stronger condition: validation owns a fresh copy of that graph.
    shim_source: Option<PreparedValue>,
}

async fn run(case: &Value) {
    let raw = RawValue::decode(case["raw_text"].as_str().unwrap()).unwrap();
    let seen = Arc::new(Mutex::new(Seen::default()));
    let runtime = Runtime::new();
    let execute_seen = seen.clone();
    let mut tool = ToolDefinition::new(
        "probe",
        "probe",
        case.get("schema")
            .cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}}))
            .try_into()
            .unwrap(),
        "probe",
        move |request: ToolExecutionRequest| {
            {
                let mut seen = execute_seen.lock();
                seen.executed = Some(observe(&request.params));
                seen.executed_handle = Some(request.params.clone());
            }
            if let Some(update) = request.on_update {
                update(result());
            }
            Box::pin(async { Ok(result()) })
        },
    );
    if let Some(shim) = case.get("prepare") {
        let shim = shim.as_str().unwrap().to_owned();
        let prepare_seen = seen.clone();
        tool = tool.with_prepare_raw_arguments(move |raw| {
            let converted = PreparedValue::from(raw);
            let value = match shim.as_str() {
                "alias" => {
                    let child = PreparedValue::from(json!({"k":1}));
                    PreparedValue::Object([("p".into(), child.clone()), ("q".into(), child)].into())
                }
                "non-finite" => {
                    converted.set("nan", PreparedValue::number(f64::NAN));
                    converted.set("inf", PreparedValue::number(f64::INFINITY));
                    converted.set("ninf", PreparedValue::number(f64::NEG_INFINITY));
                    converted.set("nz", PreparedValue::number(-0.0));
                    converted
                }
                "reuse-raw-child" => PreparedValue::Object(
                    [
                        ("o".into(), converted.get("o").unwrap()),
                        ("extra".into(), PreparedValue::number(1.0)),
                    ]
                    .into(),
                ),
                // Cyclic validation is NOT part of this gate (Owner Q001 / #193).
                other => panic!("non-certifying shim {other}"),
            };
            prepare_seen.lock().shim_source = Some(value.clone());
            Ok(value)
        });
    }
    runtime.tools().register_for_scope(None, tool).unwrap();
    let plugin_seen = seen.clone();
    let plugin_case = case.clone();
    let plugin = PluginSpec::<Value>::new(
        "isolation",
        vec![],
        || json!({}),
        move |context, _| {
            let seen = plugin_seen.clone();
            let case = plugin_case.clone();
            async move {
                let spec = tool_execution_update_spec();
                let events = context.events().unwrap();
                events.declare(&spec).unwrap();
                let update_seen = seen.clone();
                events
                    .on_emit(
                        &spec,
                        &context.effect_store(),
                        context.scope(),
                        move |event| {
                            update_seen
                                .lock()
                                .updates
                                .push(observe_raw(&event.arguments));
                        },
                    )
                    .unwrap();
                let programs = case["hooks"].as_array().unwrap();
                for (index, ops) in programs.iter().enumerate() {
                    let ops = ops.clone();
                    let facts = case.get("facts").cloned().unwrap_or_else(|| json!([]));
                    let block = case.get("block").is_some() && index + 1 == programs.len();
                    let seen = seen.clone();
                    register_before_tool_call_hook(&context, move |current| {
                        {
                            let mut seen = seen.lock();
                            seen.entries.push(observe(&current.arguments));
                            seen.handles.push(current.arguments.clone());
                            if index == 0 {
                                seen.facts = Some(
                                    facts
                                        .as_array()
                                        .unwrap()
                                        .iter()
                                        .map(|f| {
                                            if let Some(paths) = f.get("same") {
                                                same(
                                                    &at(&current.arguments, &paths[0]),
                                                    &at(&current.arguments, &paths[1]),
                                                )
                                            } else {
                                                let source = seen
                                                    .shim_source
                                                    .as_ref()
                                                    .expect("raw child reuse shim");
                                                !same(
                                                    &at(
                                                        &current.arguments,
                                                        &f["distinct_from_raw"],
                                                    ),
                                                    &at(source, &f["distinct_from_raw"]),
                                                )
                                            }
                                        })
                                        .collect(),
                                );
                            }
                        }
                        program(&current.arguments, &ops);
                        async move {
                            Ok(if block {
                                BeforeToolCallAction::Block {
                                    reason: Some("blocked".into()),
                                    terminate: false,
                                }
                            } else {
                                BeforeToolCallAction::Proceed(None)
                            })
                        }
                    })
                    .unwrap();
                }
                Ok(())
            }
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let delivery_seen = seen.clone();
    let options =
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_execution_update(move |event| {
            delivery_seen
                .lock()
                .deliveries
                .push(observe_raw(&event.arguments));
            async { Ok(()) }
        });
    let call = ToolCall::new_raw("c1", "probe", raw.clone());
    let batch = execute_tool_calls(&runtime.context(), &[call], options)
        .await
        .unwrap();
    let seen = seen.lock();
    let actual = json!({"outcome":if batch.messages[0].is_error {"immediate_error"} else {"executed"}, "hook_entries":seen.entries, "facts":seen.facts, "execute":seen.executed, "updates":seen.updates, "raw_after":observe_raw(&raw)});
    assert_eq!(actual, case["expect"], "canonical isolation observation");
    assert_eq!(seen.deliveries, seen.updates, "raw update delivery");
    for later in seen
        .handles
        .iter()
        .skip(1)
        .chain(seen.executed_handle.iter())
    {
        assert!(
            same(&seen.handles[0], later),
            "one validated graph across listeners and execute"
        );
    }
    if let Some(source) = &seen.shim_source {
        if let Some(first) = seen.handles.first() {
            assert!(!same(source, first), "clone isolates prepared source root");
        }
    }
}

async fn named(name: &str) {
    let doc: Value = serde_json::from_slice(
        &fs::read(root().join(format!(
            "conformance/agent/arg-isolation/arg-isolation-{name}.json"
        )))
        .unwrap(),
    )
    .unwrap();
    run(&doc["arg_isolation"]).await;
}

#[tokio::test]
async fn complete_acyclic_corpus_through_real_preflight() {
    let mut files = fs::read_dir(root().join("conformance/agent/arg-isolation"))
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 21);
    for file in files {
        let doc: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        assert_ne!(doc["arg_isolation"]["prepare"], "cycle", "#193 excluded");
        println!("canonical {}", doc["name"]);
        run(&doc["arg_isolation"]).await;
    }
}

#[tokio::test]
async fn alias_identity_survives_validation() {
    named("prepared-alias-stays-shared-in-clone").await;
}
#[tokio::test]
async fn reused_child_is_isolated_from_the_prepared_source() {
    named("prepared-reused-raw-child-is-isolated").await;
}
#[tokio::test]
async fn one_graph_reaches_every_listener_and_execute() {
    named("two-hooks-share-the-validated-graph").await;
}
#[tokio::test]
async fn runtime_values_survive_validation() {
    named("prepared-non-finite-values-survive-clone").await;
    named("nested-runtime-values-survive-the-clone").await;
}
#[tokio::test]
async fn raw_mutation_isolation_and_update_delivery() {
    named("hook-pushes-object-into-nested-array").await;
    named("blocked-after-nested-mutation").await;
}
#[tokio::test]
async fn enumeration_order_survives_validation() {
    named("hook-reorders-nested-by-index-key").await;
}

#[tokio::test]
async fn retained_aliases_across_object_array_frontiers_are_fresh_and_shared() {
    let raw = RawValue::decode(r#"{"o":{"old":1},"a":[{"old":2}]}"#).unwrap();
    let original_raw = observe_raw(&raw);
    let child = PreparedValue::from(json!({"b":1,"a":2}));
    let array = PreparedValue::Array(vec![child.clone()].into());
    let source = PreparedValue::Object(
        [
            ("direct".into(), child.clone()),
            ("array".into(), array.clone()),
            ("array_alias".into(), array.clone()),
        ]
        .into(),
    );
    let source_before = observe(&source);
    let handles = Arc::new(Mutex::new(Vec::<PreparedValue>::new()));
    let execution_handles = handles.clone();
    let source_for_prepare = source.clone();
    let runtime = Runtime::new();
    runtime
        .tools()
        .register_for_scope(
            None,
            ToolDefinition::new(
                "probe",
                "probe",
                json!({"type":"object","properties":{}}).try_into().unwrap(),
                "probe",
                move |request: ToolExecutionRequest| {
                    let retained = execution_handles.lock();
                    assert_eq!(retained.len(), 2);
                    assert!(
                        same(&retained[0], &retained[1]),
                        "listeners share root identity"
                    );
                    assert!(
                        same(&retained[0], &request.params),
                        "execute shares root identity"
                    );
                    let direct = request.params.get("direct").unwrap();
                    let through_array = request
                        .params
                        .get("array")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .get(0)
                        .unwrap();
                    assert!(
                        same(&direct, &through_array),
                        "object/array alias preserved"
                    );
                    assert!(
                        same(
                            &request.params.get("array").unwrap(),
                            &request.params.get("array_alias").unwrap()
                        ),
                        "array alias preserved"
                    );
                    direct.set("execute", PreparedValue::Bool(true));
                    request
                        .params
                        .get("array")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .push(PreparedValue::Null);
                    assert_eq!(
                        request
                            .params
                            .get("array_alias")
                            .unwrap()
                            .as_array()
                            .unwrap()
                            .len(),
                        2
                    );
                    assert_eq!(
                        through_array.get("execute"),
                        Some(PreparedValue::Bool(true))
                    );
                    Box::pin(async { Ok(result()) })
                },
            )
            .with_prepare_raw_arguments(move |_| Ok(source_for_prepare.clone())),
        )
        .unwrap();
    let hook_handles = handles.clone();
    let hook_source = source.clone();
    let plugin = PluginSpec::<Value>::new(
        "frontier",
        vec![],
        || json!({}),
        move |context, _| {
            let handles = hook_handles.clone();
            let source = hook_source.clone();
            async move {
                for index in 0..2 {
                    let handles = handles.clone();
                    let source = source.clone();
                    register_before_tool_call_hook(&context, move |current| {
                        assert!(!same(&source, &current.arguments), "prepared root isolated");
                        assert!(
                            !same(
                                &source.get("direct").unwrap(),
                                &current.arguments.get("direct").unwrap()
                            ),
                            "prepared child isolated"
                        );
                        assert!(
                            !same(
                                &source.get("array").unwrap(),
                                &current.arguments.get("array").unwrap()
                            ),
                            "prepared array isolated"
                        );
                        let direct = current.arguments.get("direct").unwrap();
                        if index == 0 {
                            direct.set("1", PreparedValue::number(3.0));
                        } else {
                            assert_eq!(direct.get("1"), Some(PreparedValue::number(3.0)));
                        }
                        handles.lock().push(current.arguments);
                        async { Ok(BeforeToolCallAction::Proceed(None)) }
                    })
                    .unwrap();
                }
                Ok(())
            }
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let call = ToolCall::new_raw("c1", "probe", raw.clone());
    let batch = execute_tool_calls(
        &runtime.context(),
        &[call],
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
    )
    .await
    .unwrap();
    assert!(!batch.messages[0].is_error);
    assert_eq!(
        observe(&source),
        source_before,
        "execute never mutates retained prepared source"
    );
    assert_eq!(
        observe_raw(&raw),
        original_raw,
        "execute never mutates raw input"
    );
    let keys = handles.lock()[0]
        .get("direct")
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        ["1", "b", "a", "execute"],
        "K1 survives clone and in-place mutation"
    );
}
