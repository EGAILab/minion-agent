#![cfg(feature = "conformance")]
//! Thin K1 adapter: insertion sequences and mutation programs drive real typed
//! argument handles. Observers iterate those handles without sorting anything.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use minion_agent::{
    PluginSpec, Runtime,
    argument_graph::ArgumentArray,
    execution::LocalFileSystem,
    llm::{
        AssistantContentBlock, AssistantMessage, Message, ModelIdentity, RawValue, StopReason,
        TextBlock, ToolCall, ToolResultContentBlock, Usage,
    },
    session::Session,
    tools::{
        AgentToolResult, BeforeToolCallAction, PreparedValue, ToolDefinition, ToolExecutionOptions,
        ToolExecutionRequest,
        builtin::{create_edit_tool, prepare_edit_arguments},
        execute_tool_calls, register_before_tool_call_hook, tool_execution_start_spec,
        tool_execution_update_spec,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};

trait Graph: Clone {
    fn scalar(value: Value) -> Self;
    fn object(entries: Vec<(String, Self)>) -> Self;
    fn array(values: Vec<Self>) -> Self;
    fn get(&self, key: &str) -> Self;
    fn set(&self, key: &str, value: Self);
    fn elements(&self) -> ArgumentArray<Self>;
    fn observe(&self) -> Value;
}

impl Graph for RawValue {
    fn scalar(value: Value) -> Self {
        value.into()
    }
    fn object(entries: Vec<(String, Self)>) -> Self {
        Self::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    fn array(values: Vec<Self>) -> Self {
        Self::Array(values.into())
    }
    fn get(&self, key: &str) -> Self {
        self.get(key).expect("program property")
    }
    fn set(&self, key: &str, value: Self) {
        let Self::Object(o) = self else {
            panic!("program object")
        };
        o.insert(key.into(), value);
    }
    fn elements(&self) -> ArgumentArray<Self> {
        let Self::Array(a) = self else {
            panic!("program array")
        };
        a.clone()
    }
    fn observe(&self) -> Value {
        match self {
            Self::Object(o) => {
                json!({"o":o.iter().map(|(k,v)| json!([k.to_string().unwrap(),v.observe()])).collect::<Vec<_>>()})
            }
            Self::Array(a) => json!({"a":a.iter().map(|v| v.observe()).collect::<Vec<_>>()}),
            _ => self.try_to_json().unwrap(),
        }
    }
}

impl Graph for PreparedValue {
    fn scalar(value: Value) -> Self {
        value.into()
    }
    fn object(entries: Vec<(String, Self)>) -> Self {
        Self::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    fn array(values: Vec<Self>) -> Self {
        Self::Array(values.into())
    }
    fn get(&self, key: &str) -> Self {
        self.get(key).expect("program property")
    }
    fn set(&self, key: &str, value: Self) {
        self.set(key, value);
    }
    fn elements(&self) -> ArgumentArray<Self> {
        self.as_array().expect("program array")
    }
    fn observe(&self) -> Value {
        match self {
            Self::Object(o) => {
                json!({"o":o.iter().map(|(k,v)| json!([k.as_str().unwrap(),v.observe()])).collect::<Vec<_>>()})
            }
            Self::Array(a) => json!({"a":a.iter().map(|v| v.observe()).collect::<Vec<_>>()}),
            _ => self.try_to_json().unwrap(),
        }
    }
}

fn fixture<G: Graph>(value: &Value) -> G {
    if let Some(entries) = value.get("$o") {
        G::object(
            entries
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_str().unwrap().to_owned(), fixture(&p[1])))
                .collect(),
        )
    } else if let Value::Array(values) = value {
        G::array(values.iter().map(fixture).collect())
    } else {
        G::scalar(value.clone())
    }
}

fn program<G: Graph>(root: &G, ops: Option<&Value>) -> Vec<Value> {
    let Some(ops) = ops else { return Vec::new() };
    let mut handles = BTreeMap::from([("args".to_owned(), root.clone())]);
    let mut reads = Vec::new();
    for op in ops.as_array().unwrap() {
        let kind = op["op"].as_str().unwrap();
        if kind == "read" {
            let mut at = root.clone();
            for part in op["path"].as_array().unwrap() {
                at = if let Some(key) = part.as_str() {
                    at.get(key)
                } else {
                    at.elements().get(part.as_u64().unwrap() as usize).unwrap()
                };
            }
            reads.push(at.observe());
            continue;
        }
        let target = handles[op["target"].as_str().unwrap()].clone();
        let key = op["key"].as_str().unwrap();
        if kind == "get" {
            handles.insert(op["as"].as_str().unwrap().to_owned(), target.get(key));
            continue;
        }
        if kind == "extend" {
            target
                .get(key)
                .elements()
                .extend(op["values"].as_array().unwrap().iter().map(fixture));
            continue;
        }
        let value = if let Some(name) = op.get("ref") {
            handles[name.as_str().unwrap()].clone()
        } else {
            fixture(&op["value"])
        };
        if let Some(name) = op.get("as") {
            handles.insert(name.as_str().unwrap().to_owned(), value.clone());
        }
        match kind {
            "set" => target.set(key, value),
            "push" => target.get(key).elements().push(value),
            "insert" => target
                .get(key)
                .elements()
                .insert(op["index"].as_u64().unwrap() as usize, value),
            "replace" => {
                target
                    .get(key)
                    .elements()
                    .set(op["index"].as_u64().unwrap() as usize, value)
                    .unwrap();
            }
            _ => panic!("unknown op {kind}"),
        }
    }
    reads
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

type Observations = Arc<Mutex<BTreeMap<String, Value>>>;
fn record(o: &Observations, name: &str, value: Value) {
    assert!(
        o.lock().insert(name.to_owned(), value).is_none(),
        "duplicate observation {name}"
    );
}

async fn run(case: &Value) {
    let observations: Observations = Arc::default();
    let raw: RawValue = fixture(&case["arguments"]);
    let call = ToolCall::new_raw("call", "probe", raw);
    record(&observations, "raw", call.arguments.observe());
    program(&call.arguments, case.get("raw_program"));
    let message = Message::Assistant(Box::new(AssistantMessage::new(
        ModelIdentity::new("mock", "model", "mock").unwrap(),
        vec![AssistantContentBlock::ToolCall(call.clone())],
        Usage::default(),
        StopReason::ToolUse,
        0.0,
    )));
    let session = Session::new("key-order", std::iter::empty::<String>()).unwrap();
    let entry = session.append_message(message).unwrap();
    // Explicit scalar persisted form: decoding it must not change property order.
    let serialized = serde_json::to_string(&entry).unwrap();
    let encoded: Value = serde_json::from_str(&serialized).unwrap();
    let serialized_args =
        RawValue::from(encoded["data"]["message"]["content"][0]["arguments"].clone());
    assert_eq!(
        serialized_args.observe(),
        call.arguments.observe(),
        "session serialization {}",
        case["id"]
    );
    let messages = session.derive_messages().unwrap();
    let Message::Assistant(replayed) = &messages[0] else {
        panic!("assistant")
    };
    let AssistantContentBlock::ToolCall(replayed) = &replayed.content[0] else {
        panic!("call")
    };
    record(&observations, "replay", replayed.arguments.observe());

    let runtime = Runtime::new();
    let execute_observation = observations.clone();
    let parameters = if case["schema"] == "edit" {
        let dir = tempfile::tempdir().unwrap();
        create_edit_tool(Arc::new(LocalFileSystem::new(dir.path())))
            .parameters()
            .clone()
    } else {
        minion_agent::tools::RuntimeSchemaObject::try_from(PreparedValue::from(
            case["schema"].clone(),
        ))
        .unwrap()
    };
    let mut tool = ToolDefinition::new_with_runtime_schema(
        "probe",
        "probe",
        parameters,
        "probe",
        move |request: ToolExecutionRequest| {
            record(&execute_observation, "execute", request.params.observe());
            if let Some(update) = request.on_update {
                update(result());
            }
            Box::pin(async { Ok(result()) })
        },
    );
    if case.get("prepare").is_some() {
        tool = tool.with_prepare_raw_arguments(prepare_edit_arguments);
    }
    runtime.tools().register_for_scope(None, tool).unwrap();
    let plugin_case = case.clone();
    let plugin_observations = observations.clone();
    let plugin = PluginSpec::<Value>::new(
        "k1",
        vec![],
        || json!({}),
        move |context, _| {
            let case = plugin_case.clone();
            let o = plugin_observations.clone();
            async move {
                let events = context.events().unwrap();
                let start = tool_execution_start_spec();
                let update = tool_execution_update_spec();
                events.declare(&start).unwrap();
                events.declare(&update).unwrap();
                let start_program = case.get("start_program").cloned();
                events
                    .on_emit(
                        &start,
                        &context.effect_store(),
                        context.scope(),
                        move |event| {
                            program(&event.arguments, start_program.as_ref());
                        },
                    )
                    .unwrap();
                let update_program = case.get("update_program").cloned();
                events
                    .on_emit(
                        &update,
                        &context.effect_store(),
                        context.scope(),
                        move |event| {
                            program(&event.arguments, update_program.as_ref());
                        },
                    )
                    .unwrap();
                let hook_case = case.clone();
                let hook_o = o.clone();
                register_before_tool_call_hook(&context, move |current| {
                    record(&hook_o, "hook", current.arguments.observe());
                    if let Some(pairs) = hook_case.get("mutate") {
                        for p in pairs.as_array().unwrap() {
                            current
                                .arguments
                                .set(p[0].as_str().unwrap(), fixture(&p[1]));
                        }
                    }
                    let reads = program(&current.arguments, hook_case.get("program"));
                    if !reads.is_empty() {
                        record(&hook_o, "hook_reads", Value::Array(reads));
                    }
                    let replacement = hook_case.get("replace").map(fixture);
                    async move { Ok(BeforeToolCallAction::Proceed(replacement)) }
                })
                .unwrap();
                if case.get("observe_second").is_some() {
                    register_before_tool_call_hook(&context, move |current| {
                        record(&o, "second", current.arguments.observe());
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
    let start_o = observations.clone();
    let update_o = observations.clone();
    let options = ToolExecutionOptions::new(StopReason::ToolUse, 0.0)
        .with_execution_start(move |event| {
            record(&start_o, "start", event.arguments.observe());
            async { Ok(()) }
        })
        .with_execution_update(move |event| {
            record(&update_o, "update", event.arguments.observe());
            async { Ok(()) }
        });
    let batch = execute_tool_calls(&runtime.context(), &[call], options)
        .await
        .unwrap();
    assert!(
        !batch.messages[0].is_error,
        "execution {}: {:?}",
        case["id"], batch.messages
    );
    let actual = observations.lock();
    for (name, expected) in case["expect"].as_object().unwrap() {
        assert_eq!(actual.get(name), Some(expected), "{} / {name}", case["id"]);
    }
}

#[tokio::test]
async fn complete_key_order_corpus_through_real_tool_events_and_session() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let document: Value = serde_json::from_slice(
        &std::fs::read(root.join("conformance/agent/key-order/key-order.json")).unwrap(),
    )
    .unwrap();
    let cases = document["key_order"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 49);
    for case in cases {
        run(case).await;
    }
}
