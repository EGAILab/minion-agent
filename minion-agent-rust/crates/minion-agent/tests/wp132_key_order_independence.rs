#![cfg(feature = "conformance")]
use minion_agent::{
    PluginInitError, PluginSpec, Runtime,
    execution::LocalFileSystem,
    llm::{RawValue, ResultTextBlock, StopReason, ToolCall, ToolResultContentBlock},
    tools::{
        AfterToolCallOverride, ToolExecutionOptions,
        builtin::{create_edit_tool, create_write_tool},
        execute_tool_calls, register_after_tool_call_hook,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::sync::Arc;

struct Signal(bool);
impl minion_agent::tools::ToolExecutionSignal for Signal {
    fn is_cancelled(&self) -> bool {
        self.0
    }
}

fn variants(value: &RawValue) -> Vec<RawValue> {
    let RawValue::Object(object) = value else {
        panic!("object fixture")
    };
    let entries = object
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<Vec<_>>();
    let mut reversed = entries.clone();
    reversed.reverse();
    let mut index_first = entries.clone();
    index_first.sort_by_key(|(k, _)| match k.to_string().unwrap().parse::<u32>() {
        Ok(n) => (0, n, String::new()),
        Err(_) => (1, 0, String::new()),
    });
    let mut sorted = entries.clone();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    [entries, reversed, index_first, sorted]
        .into_iter()
        .map(|v| RawValue::Object(v.into_iter().collect()))
        .collect()
}
fn order(value: &RawValue) -> Vec<String> {
    let RawValue::Object(object) = value else {
        panic!("object")
    };
    object.keys().map(|k| k.to_string().unwrap()).collect()
}
async fn run(
    calls: &[(&str, RawValue)],
    aborted: bool,
    leaky: bool,
) -> (
    Vec<minion_agent::llm::ToolResultMessage>,
    Vec<u8>,
    Vec<Vec<String>>,
) {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("f.txt"), b"alpha\nbeta\ngamma\n").unwrap();
    let runtime = Runtime::new();
    let fs = Arc::new(LocalFileSystem::new(root.path()));
    runtime
        .tools()
        .register_for_scope(None, create_edit_tool(fs.clone()))
        .unwrap();
    runtime
        .tools()
        .register_for_scope(None, create_write_tool(fs))
        .unwrap();
    let observed = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    if leaky {
        let capture = observed.clone();
        let plugin = PluginSpec::<Value>::new(
            "leaky-order-control",
            vec![],
            || json!({}),
            move |context, _| {
                let capture = capture.clone();
                async move {
                    register_after_tool_call_hook(&context, move |result| {
                        let capture = capture.clone();
                        async move {
                            let ToolResultContentBlock::Text(text) = &result.content[0] else {
                                panic!("text")
                            };
                            let text = format!(
                                "{} {}",
                                text.text,
                                capture.lock().last().unwrap().join(",")
                            );
                            Ok(Some(AfterToolCallOverride::default().with_content(vec![
                                ToolResultContentBlock::Text(ResultTextBlock::new(text)),
                            ])))
                        }
                    })
                    .map_err(|e| PluginInitError::new(e.to_string()))?;
                    Ok(())
                }
            },
        )
        .erase();
        runtime.mount(&plugin, json!({})).unwrap();
        runtime.reconcile().await.unwrap();
    }
    let signal = Signal(aborted);
    let calls = calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| ToolCall::new_raw(format!("c{i}"), *name, args.clone()))
        .collect::<Vec<_>>();
    let capture = observed.clone();
    let options = ToolExecutionOptions::new(StopReason::ToolUse, 0.0)
        .with_signal(Arc::new(signal))
        .with_execution_start(move |event| {
            capture.lock().push(order(&event.arguments));
            async { Ok(()) }
        });
    let batch = execute_tool_calls(&runtime.context(), &calls, options)
        .await
        .unwrap();
    let bytes = std::fs::read(root.path().join("f.txt")).unwrap();
    let seen = observed.lock().clone();
    (batch.messages, bytes, seen)
}

#[tokio::test]
async fn owned_outputs_are_independent_of_distinct_raw_key_enumerations() {
    let fixtures = [
        (
            "write",
            r#"{"path":"f.txt","content":"new\ncontent\n","0":"x","10":"y","1":"z","b":true}"#,
        ),
        (
            "edit",
            r#"{"path":"f.txt","edits":[{"oldText":"beta","newText":"BETA","0":"x","10":"y","1":"z","b":true}],"0":"x","10":"y","1":"z","b":true}"#,
        ),
        (
            "edit",
            r#"{"path":"f.txt","edits":[{"oldText":"alpha","newText":"A"},{"oldText":"gamma","newText":"G"}],"0":"x","10":"y","1":"z","b":true}"#,
        ),
        (
            "edit",
            r#"{"path":"f.txt","edits":[{"oldText":"missing","newText":"x"}],"0":"x","10":"y","1":"z","b":true}"#,
        ),
        (
            "write",
            r#"{"path":".","content":"x","0":"x","10":"y","1":"z","b":true}"#,
        ),
    ];
    for (name, text) in fixtures {
        let permutations = variants(&RawValue::decode(text).unwrap());
        assert_eq!(
            permutations
                .iter()
                .map(order)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            2
        );
        for aborted in [false, true] {
            let mut baseline = None;
            for args in permutations.clone() {
                let expected_order = order(&args);
                let (messages, bytes, observed) = run(&[(name, args)], aborted, false).await;
                if !aborted {
                    assert_eq!(observed, vec![expected_order]);
                }
                let output = (messages, bytes);
                if let Some(base) = &baseline {
                    assert_eq!(&output, base);
                } else {
                    baseline = Some(output);
                }
            }
        }
    }
}

#[tokio::test]
async fn nested_json_string_and_same_target_queue_are_order_independent() {
    let item = RawValue::decode(
        r#"{"oldText":"beta","newText":"BETA","0":"x","10":"y","1":"z","b":true}"#,
    )
    .unwrap();
    let mut outcomes = Vec::new();
    for nested in variants(&item) {
        let nested_orders = order(&nested);
        assert_eq!(nested_orders.len(), 6);
        for as_json in [false, true] {
            let edits = if as_json {
                RawValue::from(json!(format!(
                    "[{}]",
                    serde_json::to_string(&nested).unwrap()
                )))
            } else {
                RawValue::Array(vec![nested.clone()].into())
            };
            let base = RawValue::decode(
                r#"{"path":"f.txt","edits":[],"0":"x","10":"y","1":"z","b":true}"#,
            )
            .unwrap();
            for mut args in variants(&base) {
                let RawValue::Object(ref mut object) = args else {
                    unreachable!()
                };
                object.insert("edits".into(), edits.clone());
                let result = run(&[("edit", args)], false, false).await;
                outcomes.push((result.0, result.1));
            }
        }
    }
    assert!(outcomes.iter().all(|v| v == &outcomes[0]));
    let first=RawValue::decode(r#"{"path":"f.txt","edits":[{"oldText":"beta","newText":"B1"}],"0":"x","10":"y","1":"z","b":true}"#).unwrap();
    let second = RawValue::decode(
        r#"{"path":"f.txt","content":"after\n","0":"x","10":"y","1":"z","b":true}"#,
    )
    .unwrap();
    let mut outputs = Vec::new();
    for (a, b) in variants(&first).into_iter().zip(variants(&second)) {
        let r = run(&[("edit", a), ("write", b)], false, false).await;
        outputs.push((r.0, r.1));
    }
    assert!(outputs.iter().all(|v| v == &outputs[0]));
}

#[tokio::test]
async fn witness_detects_an_order_leaking_result_control() {
    let args=RawValue::decode(r#"{"path":"f.txt","edits":[{"oldText":"beta","newText":"BETA"}],"0":"x","10":"y","1":"z","b":true}"#).unwrap();
    let mut outputs = Vec::new();
    for args in variants(&args) {
        let r = run(&[("edit", args)], false, true).await;
        outputs.push((r.0, r.1));
    }
    assert!(
        outputs.iter().any(|v| v != &outputs[0]),
        "raw-key-order leak mutant survived"
    );
}
