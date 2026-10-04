#![cfg(feature = "conformance")]
//! Real local capabilities and Layer 06. Only fixture/path and observation normalization.
use minion_agent::{
    Runtime,
    execution::{LocalFileSystem, LocalSubprocess},
    llm::{RawNumber, RawValue, StopReason, ToolCall},
    tools::{
        ToolExecutionOptions, ToolExecutionSignal,
        builtin::{BashToolOptions, create_bash_tool},
        execute_tool_calls,
    },
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Default)]
struct Signal(AtomicBool);
impl ToolExecutionSignal for Signal {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[tokio::test]
async fn real_detached_descendant_inherits_pipes_but_is_not_killed_by_settlement() {
    assert_eq!(String::from_utf8(std::process::Command::new("node").arg("--version").output().unwrap().stdout).unwrap().trim(),"v22.15.1");
    let dir=tempfile::tempdir().unwrap();std::fs::write(dir.path().join("parent.cjs"),include_bytes!("data/bash-descendant.cjs")).unwrap();
    let tool=create_bash_tool(Arc::new(LocalFileSystem::new(dir.path())),Arc::new(LocalSubprocess::new(dir.path())),BashToolOptions::default()).unwrap();
    let result=tokio::time::timeout(std::time::Duration::from_secs(10),(tool.execute())(minion_agent::tools::ToolExecutionRequest{tool_call_id:"descendant".into(),params:json!({"command":"node parent.cjs"}).into(),signal:None,on_update:None,context:None})).await;
    let pid=std::fs::read_to_string(dir.path().join("descendant.pid")).unwrap();
    let alive=std::process::Command::new("node").args(["-e",&format!("process.kill({pid},0)")]).status().unwrap().success();
    // Harness cleanup, NOT tool settlement. Always performed before assertions.
    let _=std::process::Command::new("node").args(["-e",&format!("try {{process.kill({pid},'SIGKILL')}} catch {{}}")]).status();
    assert!(result.is_ok(),"own exit must settle despite descendant-held pipes");assert!(alive,"pipe disposal must not kill descendants");
    let result=serde_json::to_value(result.unwrap().unwrap().content).unwrap();assert_eq!(result[0]["text"],"descendant-ready\n");
}

#[tokio::test]
async fn canonical_bash_real_capabilities_and_layer_six() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/builtin-bash");
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    let mut count = 0;
    for path in paths {
        let document: Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let case = &document["builtin_bash"];
        let expected = &case["expect"][if cfg!(windows) { "win32" } else { "linux" }];
        if expected.is_null() {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let cwd = if case["missing_cwd"] == true {
            dir.path().join("missing")
        } else {
            dir.path().to_path_buf()
        };
        let runtime = Runtime::new();
        let tool = create_bash_tool(
            Arc::new(LocalFileSystem::new(dir.path())),
            Arc::new(LocalSubprocess::new(&cwd)),
            BashToolOptions::default(),
        )
        .unwrap();
        runtime.tools().register_for_scope(None, tool).unwrap();
        let signal = Arc::new(Signal::default());
        if case["signal"] == "pre_aborted" {
            signal.0.store(true, Ordering::SeqCst);
        }
        let mut arguments: RawValue =
            serde_json::from_value(json!({"command":case["command"]})).unwrap();
        if let Some(timeout) = case.get("timeout") {
            let value = if timeout == "Infinity" {
                RawValue::Number(RawNumber::new(f64::INFINITY).unwrap())
            } else {
                serde_json::from_value(timeout.clone()).unwrap()
            };
            let RawValue::Object(object) = &mut arguments else {
                unreachable!()
            };
            object.insert("timeout".into(), value);
        }
        let abort = case["abort_after_ms"].as_u64().map(|ms| {
            let signal = signal.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                signal.0.store(true, Ordering::SeqCst);
            })
        });
        let batch = execute_tool_calls(
            &runtime.context(),
            &[ToolCall::new_raw("case", "bash", arguments)],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_signal(signal),
        )
        .await
        .unwrap();
        if let Some(abort) = abort {
            abort.abort();
        }
        let result = serde_json::to_value(&batch.messages[0]).unwrap();
        let mut text = result["content"][0]["text"].as_str().unwrap().to_owned();
        let mut details = result["details"].clone();
        let full_path = details["fullOutputPath"]
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                text.split("Full output: ")
                    .nth(1)
                    .and_then(|tail| tail.split(']').next())
                    .map(str::to_owned)
            });
        let full_output = if let Some(path) = &full_path {
            let bytes = std::fs::read(path).unwrap();
            std::fs::remove_file(path).unwrap();
            text = text.replace(path, "<FULL_OUTPUT>");
            json!({"size":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))})
        } else {
            Value::Null
        };
        text = text.replace(&cwd.to_string_lossy().into_owned(), "<CWD>");
        if let Some(truncation) = details.get_mut("truncation") {
            let content = truncation
                .as_object_mut()
                .unwrap()
                .remove("content")
                .unwrap();
            assert!(
                text.starts_with(content.as_str().unwrap()),
                "{} content",
                document["name"]
            );
            details["fullOutputPath"] = json!("<FULL_OUTPUT>");
        }
        assert_eq!(
            result["is_error"], expected["is_error"],
            "{}",
            document["name"]
        );
        if expected["validation_rejected"] == true {
            assert!(text.contains("timeout"));
        }
        if let Some(expected) = expected.get("text") {
            assert_eq!(json!(text), *expected, "{}", document["name"]);
        }
        let units: Vec<_> = text.encode_utf16().collect();
        if let Some(length) = expected.get("text_length") {
            assert_eq!(json!(units.len()), *length, "{}", document["name"]);
        }
        for (key, slice) in [
            ("text_head", &units[..units.len().min(200)]),
            ("text_tail", &units[units.len().saturating_sub(400)..]),
        ] {
            if let Some(expected) = expected.get(key) {
                assert_eq!(
                    json!(String::from_utf16(slice).unwrap()),
                    *expected,
                    "{} {key}",
                    document["name"]
                );
            }
        }
        assert_eq!(details, expected["details"], "{}", document["name"]);
        assert_eq!(full_output, expected["full_output"], "{}", document["name"]);
        count += 1;
    }
    assert!(count > 0, "canonical corpus must not pass vacuously");
    eprintln!("builtin bash: {count} canonical documents on {}", if cfg!(windows) { "win32" } else { "linux" });
}
