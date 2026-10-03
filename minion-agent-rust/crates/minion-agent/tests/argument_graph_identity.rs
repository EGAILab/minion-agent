//! K1 reference/provenance regressions. No Python native-alias latitude exists.
use std::sync::Arc;

use minion_agent::{
    PluginSpec, Runtime,
    argument_graph::{ArgumentArray, ArgumentObjectRef},
    llm::{RawValue, StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        AgentToolResult, BeforeToolCallAction, PreparedValue, ToolDefinition, ToolExecutionOptions,
        ToolExecutionRequest, execute_tool_calls, register_before_tool_call_hook,
        tool_execution_start_spec, tool_execution_update_spec,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};

fn keys(value: &PreparedValue) -> Vec<String> {
    value
        .as_object()
        .unwrap()
        .keys()
        .map(|key| key.as_str().unwrap().to_owned())
        .collect()
}
fn raw_keys(value: &RawValue) -> Vec<String> {
    let RawValue::Object(object) = value else {
        panic!("object")
    };
    object.keys().map(|key| key.to_string().unwrap()).collect()
}
fn object(
    value: &PreparedValue,
) -> ArgumentObjectRef<minion_agent::tools::PreparedString, PreparedValue> {
    value.as_object().unwrap().clone()
}
fn child() -> PreparedValue {
    PreparedValue::from(RawValue::decode(r#"{"z":1,"2":2,"1":3}"#).unwrap())
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

#[test]
fn object_attachment_keeps_the_exact_retained_child_identity() {
    let root = PreparedValue::from(json!({}));
    let retained = child();
    root.set("a", retained.clone());
    root.set("b", retained.clone());
    assert!(object(&root.get("a").unwrap()).same_identity(&object(&retained)));
    retained.set("0", PreparedValue::Bool(true));
    assert_eq!(keys(&root.get("a").unwrap()), ["0", "1", "2", "z"]);
    assert_eq!(root.get("a"), root.get("b"));
}

#[test]
fn array_append_insert_replace_and_extend_keep_the_supplied_handles() {
    let array = ArgumentArray::default();
    let retained = child();
    array.push(retained.clone());
    array.insert(0, retained.clone());
    array.set(1, retained.clone()).unwrap();
    array.extend([retained.clone()]);
    retained.set("0", PreparedValue::Null);
    for value in array.iter() {
        assert!(object(&value).same_identity(&object(&retained)));
        assert_eq!(keys(&value), ["0", "1", "2", "z"]);
    }
}

#[test]
fn structured_clone_copies_the_graph_but_not_its_aliases_in_either_frontier_order() {
    for reverse in [false, true] {
        let shared = child();
        let array = PreparedValue::Array(vec![shared.clone()].into());
        let root = PreparedValue::from(json!({}));
        for (key, value) in if reverse {
            [("plain", shared.clone()), ("graph", array)]
        } else {
            [("graph", array), ("plain", shared.clone())]
        } {
            root.set(key, value);
        }
        let copied = root.structured_clone();
        let first = copied.get("plain").unwrap();
        let second = copied
            .get("graph")
            .unwrap()
            .as_array()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(object(&first).same_identity(&object(&second)));
        assert!(!object(&first).same_identity(&object(&shared)));
        first.set("0", PreparedValue::Null);
        assert_eq!(keys(&second), ["0", "1", "2", "z"]);
        assert_eq!(keys(&shared), ["1", "2", "z"]);
    }
}

#[test]
fn raw_to_prepared_preserves_shared_object_and_array_aliases() {
    let retained = RawValue::decode(r#"{"z":1,"2":2}"#).unwrap();
    let array = ArgumentArray::from(vec![retained.clone()]);
    let raw = RawValue::Object(ArgumentObjectRef::from([
        ("a".into(), RawValue::Array(array.clone())),
        ("b".into(), retained),
        ("c".into(), RawValue::Array(array)),
    ]));
    let prepared = PreparedValue::from(raw);
    let a = prepared.get("a").unwrap().as_array().unwrap();
    let c = prepared.get("c").unwrap().as_array().unwrap();
    assert!(a.same_identity(&c));
    let b = prepared.get("b").unwrap();
    assert!(object(&a.get(0).unwrap()).same_identity(&object(&b)));
    b.set("1", PreparedValue::Bool(true));
    assert_eq!(keys(&c.get(0).unwrap()), ["1", "2", "z"]);
}

#[tokio::test]
async fn shim_objects_are_isolated_but_hooks_share_one_alias_preserving_graph() {
    let shim_child = child();
    let shim = PreparedValue::from(json!({}));
    shim.set("a", shim_child.clone());
    shim.set("b", PreparedValue::Array(vec![shim_child.clone()].into()));
    let shim_input = shim.clone();
    let runtime = Runtime::new();
    let seen = Arc::new(Mutex::new(None));
    let executed = seen.clone();
    runtime
        .tools()
        .register_for_scope(
            None,
            ToolDefinition::new(
                "probe",
                "probe",
                serde_json::from_value(json!({})).unwrap(),
                "probe",
                move |r: ToolExecutionRequest| {
                    *executed.lock() = Some(r.params);
                    Box::pin(async { Ok(result()) })
                },
            )
            .with_prepare_raw_arguments(move |_| Ok(shim_input.clone())),
        )
        .unwrap();
    let plugin = PluginSpec::<Value>::new(
        "observer",
        vec![],
        || json!({}),
        move |context, _| async move {
            register_before_tool_call_hook(&context, |call| async move {
                let a = call.arguments.get("a").unwrap();
                let b = call
                    .arguments
                    .get("b")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .get(0)
                    .unwrap();
                assert!(object(&a).same_identity(&object(&b)));
                a.set("0", PreparedValue::Null);
                // Proceed(None) must not discard this in-place mutation.
                Ok(BeforeToolCallAction::Proceed(None))
            })
            .unwrap();
            register_before_tool_call_hook(&context, |call| async move {
                assert_eq!(
                    keys(
                        &call
                            .arguments
                            .get("b")
                            .unwrap()
                            .as_array()
                            .unwrap()
                            .get(0)
                            .unwrap()
                    ),
                    ["0", "1", "2", "z"]
                );
                Ok(BeforeToolCallAction::Proceed(None))
            })
            .unwrap();
            Ok(())
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let batch = execute_tool_calls(
        &runtime.context(),
        &[ToolCall::new_raw("c", "probe", json!({}).into())],
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
    )
    .await
    .unwrap();
    assert!(!batch.messages[0].is_error);
    assert_eq!(
        keys(&seen.lock().as_ref().unwrap().get("a").unwrap()),
        ["0", "1", "2", "z"]
    );
    assert_eq!(keys(&shim_child), ["1", "2", "z"]);
}

#[tokio::test]
async fn raw_event_listeners_live_deliveries_and_retained_array_root_share_identity() {
    let raw_child = RawValue::decode(r#"{"z":1}"#).unwrap();
    let root = RawValue::Array(vec![raw_child.clone()].into());
    let retained = root.clone();
    let runtime = Runtime::new();
    runtime
        .tools()
        .register_for_scope(
            None,
            ToolDefinition::new(
                "probe",
                "probe",
                serde_json::from_value(json!({})).unwrap(),
                "probe",
                |request: ToolExecutionRequest| {
                    if let Some(update) = request.on_update {
                        update(result());
                    }
                    Box::pin(async { Ok(result()) })
                },
            ),
        )
        .unwrap();
    let plugin = PluginSpec::<Value>::new(
        "raw-events",
        vec![],
        || json!({}),
        move |context, _| async move {
            let events = context.events().unwrap();
            let start = tool_execution_start_spec();
            let update = tool_execution_update_spec();
            events.declare(&start).unwrap();
            events.declare(&update).unwrap();
            events
                .on_emit(&start, &context.effect_store(), context.scope(), |e| {
                    let RawValue::Array(a) = &e.arguments else {
                        panic!()
                    };
                    let RawValue::Object(o) = a.get(0).unwrap() else {
                        panic!()
                    };
                    o.insert("2".into(), RawValue::Null);
                })
                .unwrap();
            events
                .on_emit(&start, &context.effect_store(), context.scope(), |e| {
                    let RawValue::Array(a) = &e.arguments else {
                        panic!()
                    };
                    assert_eq!(raw_keys(&a.get(0).unwrap()), ["2", "z"]);
                })
                .unwrap();
            events
                .on_emit(&update, &context.effect_store(), context.scope(), |e| {
                    let RawValue::Array(a) = &e.arguments else {
                        panic!()
                    };
                    let RawValue::Object(o) = a.get(0).unwrap() else {
                        panic!()
                    };
                    o.insert("1".into(), RawValue::Null);
                })
                .unwrap();
            Ok(())
        },
    )
    .erase();
    runtime.mount(&plugin, json!({})).unwrap();
    runtime.reconcile().await.unwrap();
    let options = ToolExecutionOptions::new(StopReason::ToolUse, 0.0)
        .with_execution_start(|e| {
            let RawValue::Array(a) = e.arguments else {
                panic!()
            };
            assert_eq!(raw_keys(&a.get(0).unwrap()), ["2", "z"]);
            async { Ok(()) }
        })
        .with_execution_update(|e| {
            let RawValue::Array(a) = e.arguments else {
                panic!()
            };
            assert_eq!(raw_keys(&a.get(0).unwrap()), ["1", "2", "z"]);
            async { Ok(()) }
        });
    let batch = execute_tool_calls(
        &runtime.context(),
        &[ToolCall::new_raw("c", "probe", root)],
        options,
    )
    .await
    .unwrap();
    assert!(!batch.messages[0].is_error);
    let RawValue::Array(array) = retained else {
        panic!()
    };
    assert_eq!(raw_keys(&array.get(0).unwrap()), ["1", "2", "z"]);
    assert_eq!(raw_keys(&raw_child), ["1", "2", "z"]);
}
