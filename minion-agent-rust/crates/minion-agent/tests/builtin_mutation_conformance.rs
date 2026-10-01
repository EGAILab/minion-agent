#![cfg(feature = "conformance")]
#[path = "support/mutation_fs.rs"]
mod fixture;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fixture::{FixtureFs, Gate, Signal};
use futures::FutureExt;
use minion_agent::{
    Runtime,
    llm::{StopReason, ToolCall},
    tools::{
        ToolExecutionOptions,
        builtin::{create_edit_tool, create_write_tool},
        execute_tool_calls,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, sync::Arc, time::Duration};

fn bytes(value: &Value) -> Vec<u8> {
    value["base64"].as_str().map_or_else(
        || value["text"].as_str().unwrap().as_bytes().to_vec(),
        |s| STANDARD.decode(s).unwrap(),
    )
}
fn build(root: &Path, entries: &Value) {
    for entry in entries.as_array().into_iter().flatten() {
        let path = root.join(entry["path"].as_str().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        if entry["dir"] == true {
            std::fs::create_dir_all(path).unwrap();
        } else if let Some(target) = entry["symlink"].as_str() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(root.join(target), path).unwrap();
            #[cfg(windows)]
            {
                if root.join(target).is_dir() {
                    std::os::windows::fs::symlink_dir(root.join(target), path).unwrap();
                } else {
                    std::os::windows::fs::symlink_file(root.join(target), path).unwrap();
                }
            }
        } else {
            std::fs::write(path, bytes(&entry["file"])).unwrap();
        }
    }
}
fn files(root: &Path, expected: &Value) {
    for file in expected.as_array().into_iter().flatten() {
        let path = root.join(file["path"].as_str().unwrap());
        if file["absent"] == true {
            assert!(!path.exists(), "{}", path.display());
        } else {
            assert_eq!(
                std::fs::read(&path).unwrap(),
                bytes(file),
                "{}",
                path.display()
            );
        }
    }
}
fn runtime(fs: Arc<FixtureFs>) -> Arc<Runtime> {
    runtime_with_abort_control(fs, None)
}

fn runtime_with_abort_control(fs: Arc<FixtureFs>, delay: Option<Duration>) -> Arc<Runtime> {
    let runtime = Arc::new(Runtime::new());
    let mut write = create_write_tool(fs.clone());
    if let Some(delay) = delay {
        let execute = write.execute().clone();
        let schema = write.schema();
        write = minion_agent::tools::ToolDefinition::new(
            schema.name,
            schema.description,
            schema.parameters,
            "write",
            move |request: minion_agent::tools::ToolExecutionRequest| {
                let signal = request.signal.clone();
                let task = tokio::spawn(execute(request));
                Box::pin(async move {
                    let mut task = task;
                    loop {
                        if signal.as_ref().is_some_and(|s| s.is_cancelled()) {
                            return Err(minion_agent::tools::ToolCapabilityError::new(
                                "Operation aborted",
                            ));
                        }
                        tokio::select! {
                            result = &mut task => return result.unwrap(),
                            () = tokio::time::sleep(delay) => {},
                        }
                    }
                })
            },
        );
    }
    runtime.tools().register_for_scope(None, write).unwrap();
    runtime
        .tools()
        .register_for_scope(None, create_edit_tool(fs))
        .unwrap();
    runtime
}
async fn invoke(runtime: &Runtime, call: &Value, signal: Signal) -> Value {
    let call = ToolCall::new(
        call["id"].as_str().unwrap_or("case"),
        call["tool"].as_str().unwrap(),
        serde_json::from_value(call["arguments"].clone()).unwrap(),
    );
    let batch = execute_tool_calls(
        &runtime.context(),
        &[call],
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_signal(Arc::new(signal)),
    )
    .await
    .unwrap();
    let message = serde_json::to_value(&batch.messages[0]).unwrap();
    json!({"is_error":message["is_error"], "text":message["content"][0]["text"], "details":message["details"]})
}
fn result(observed: &Value, expected: &Value) {
    assert_eq!(observed["is_error"], expected["is_error"]);
    if expected["argument_validation_failure"] == true {
        return;
    }
    for key in ["text", "details"] {
        if let Some(expected) = expected.get(key) {
            assert_eq!(&observed[key], expected, "{key}");
        }
    }
}
async fn run_case(document: &Value, case: &Value) {
    let dir = tempfile::tempdir().unwrap();
    build(dir.path(), &document["builtin_mutation"]["fixture"]);
    build(dir.path(), &case["fixture"]);
    let signal = Signal::default();
    if case["signal"] == "pre_aborted" {
        signal.abort();
    }
    let fs = Arc::new(FixtureFs::new(
        dir.path(),
        "p",
        document["builtin_mutation"]["provider"].clone(),
        Arc::default(),
        vec![],
        signal.clone(),
        case["abort_after"].as_str().map(str::to_owned),
    ));
    let observed = invoke(&runtime(fs.clone()), case, signal).await;
    result(&observed, &case["expect"]);
    if let Some(expected) = case["expect"].get("fs_calls") {
        assert_eq!(
            serde_json::to_value(fs.calls.lock().clone()).unwrap(),
            *expected
        );
    }
    files(dir.path(), &case["expect"]["files_after"]);
}

// With a paused current-thread runtime, Tokio advances to the next timer only
// after runnable work and blocking filesystem work have drained. Sleeping a
// virtual minute therefore fires all intermediate timers, not a real settle
// window nor a fixed number of scheduler turns. Timer controls below prove this.
async fn quiesce() {
    tokio::time::sleep(Duration::from_secs(60)).await;
}

async fn run_queue(document: &Value, abort_control: Option<Duration>) {
    let dir = tempfile::tempdir().unwrap();
    let input = &document["builtin_mutation"];
    let queue = &input["queue"];
    build(dir.path(), &input["fixture"]);
    let gates: Vec<_> = queue["gates"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|v| Arc::new(Gate::new(v.clone())))
        .collect();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut providers = BTreeMap::new();
    let ids = queue["providers"]
        .as_array()
        .cloned()
        .unwrap_or_else(|| vec![json!("p")]);
    for id in ids {
        let id = id.as_str().unwrap();
        providers.insert(
            id.to_owned(),
            runtime_with_abort_control(
                Arc::new(FixtureFs::new(
                    dir.path(),
                    id,
                    input["provider"].clone(),
                    log.clone(),
                    gates.clone(),
                    Signal::default(),
                    None,
                )),
                abort_control,
            ),
        );
    }
    let mut calls = BTreeMap::new();
    let mut signals = BTreeMap::new();
    for call in queue["calls"].as_array().unwrap() {
        let started = Arc::new(Mutex::new(false));
        let id = call["id"].as_str().unwrap().to_owned();
        let runtime = providers[call["provider"].as_str().unwrap_or("p")].clone();
        let signal = Signal::default();
        signals.insert(id.clone(), signal.clone());
        let call = call.clone();
        let log = log.clone();
        let id2 = id.clone();
        let started2 = started.clone();
        let task = tokio::spawn(async move {
            // Invoke is polled before the next call is launched. The registration
            // happens on that first poll; it must not be reordered by spawning.
            let future = invoke(&runtime, &call, signal);
            tokio::pin!(future);
            let value = futures::future::poll_fn(|cx| {
                let poll = future.as_mut().poll(cx);
                *started2.lock() = true;
                poll
            })
            .await;
            log.lock().push(format!("result {id2}"));
            value
        });
        while !*started.lock() {
            tokio::task::yield_now().await;
        }
        calls.insert(id, task);
    }
    quiesce().await;
    for step in queue["steps"].as_array().into_iter().flatten() {
        if let Some(id) = step["abort"].as_str() {
            signals[id].abort();
        } else {
            let gate = gates
                .iter()
                .find(|g| g.spec["id"] == step["release"])
                .unwrap();
            gate.release(
                step.get("error")
                    .map(|v| serde_json::from_value(v.clone()).unwrap()),
            );
        }
        quiesce().await;
    }
    for (id, task) in calls {
        assert!(
            task.is_finished(),
            "lingering call {id}; log: {:?}",
            log.lock()
        );
        result(&task.await.unwrap(), &queue["expect"]["results"][&id]);
    }
    let log = log.lock();
    for pair in queue["expect"]["order"].as_array().into_iter().flatten() {
        let a = log
            .iter()
            .position(|s| s == pair[0].as_str().unwrap())
            .unwrap_or_else(|| panic!("missing {} in {log:?}", pair[0]));
        let b = log
            .iter()
            .position(|s| s == pair[1].as_str().unwrap())
            .unwrap_or_else(|| panic!("missing {} in {log:?}", pair[1]));
        assert!(a < b, "{} before {} violated in {log:?}", pair[0], pair[1]);
    }
    for event in queue["expect"]["logged"].as_array().into_iter().flatten() {
        assert!(log.iter().any(|s| s == event.as_str().unwrap()));
    }
    for event in queue["expect"]["never"].as_array().into_iter().flatten() {
        assert!(!log.iter().any(|s| s == event.as_str().unwrap()));
    }
    files(dir.path(), &queue["expect"]["files_after"]);
}

#[tokio::test(start_paused = true)]
async fn canonical_non_corpus_mutations_and_all_queue_documents() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/builtin-mutation");
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| !p.file_name().unwrap().to_string_lossy().contains("corpus"))
        .collect();
    paths.sort();
    let mut case_count = 0;
    let mut queues = 0;
    for path in paths {
        eprintln!("canonical: {}", path.display());
        let source = std::fs::read_to_string(path).unwrap();
        let document: Value = if let Some((header, _)) = source.split_once("\n  cases:") {
            let mut document: Value =
                serde_yaml::from_str(&format!("{header}\n  cases: []\n")).unwrap();
            let cases: Vec<Value> = source
                .split("\n    - id:")
                .skip(1)
                .filter_map(|block| {
                    let mut lines = block.lines();
                    let body = format!(
                        "id:{}\n{}",
                        lines.next().unwrap(),
                        lines
                            .map(|s| s.strip_prefix("      ").unwrap_or(s))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                    match serde_yaml::from_str(&body) {
                        Ok(case) => Some(case),
                        Err(e) => {
                            assert!(body.contains("unpaired_surrogate_arguments: true"), "{e}");
                            assert!(serde_json::from_str::<Value>(r#""\ud800""#).is_err());
                            None
                        }
                    }
                })
                .collect();
            document["builtin_mutation"]["cases"] = Value::Array(cases);
            document
        } else {
            serde_yaml::from_str(&source).unwrap()
        };
        if document["builtin_mutation"].get("queue").is_some() {
            run_queue(&document, None).await;
            queues += 1;
        } else {
            for case in document["builtin_mutation"]["cases"].as_array().unwrap() {
                run_case(&document, case).await;
                case_count += 1;
            }
        }
    }
    assert_eq!(queues, 11);
    assert_eq!(case_count, 42); // 43 discovered, one explicitly unreachable lone-surrogate case.
}

#[tokio::test(start_paused = true)]
async fn virtual_quiescence_fires_abort_listener_timers_even_at_ten_seconds() {
    for delay in [
        Duration::from_millis(10),
        Duration::from_millis(500),
        Duration::from_secs(10),
    ] {
        let timer = tokio::spawn(async move {
            tokio::time::sleep(delay).await;
        });
        quiesce().await;
        assert!(timer.is_finished(), "quiescence missed {delay:?}");
        timer.await.unwrap();
    }
}

#[tokio::test(start_paused = true)]
async fn queue_evidence_kills_early_abort_result_controls_at_all_three_timer_scales() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/builtin-mutation/builtin-mutation-queue-aborted-call-holds-lock-until-its-write-settles.yaml");
    let document: Value = serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for delay in [
        Duration::from_millis(10),
        Duration::from_millis(500),
        Duration::from_secs(10),
    ] {
        let rejected = std::panic::AssertUnwindSafe(run_queue(&document, Some(delay)))
            .catch_unwind()
            .await;
        assert!(
            rejected.is_err(),
            "early-result abort mutant survived at {delay:?}"
        );
    }
}
