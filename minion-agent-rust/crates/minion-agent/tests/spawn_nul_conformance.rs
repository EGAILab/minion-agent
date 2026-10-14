#![cfg(feature = "conformance")]
//! Canonical UTF-16 decoding -> real LocalSubprocess / Layer-06 tool pipeline.
//! The scripted engine resolver supplies only a binary name; no engine can run
//! in these rejected cases. It neither validates nor manufactures diagnostics.
use async_trait::async_trait;
use minion_agent::{
    Runtime,
    execution::{ExecutionWorldIdentity, LocalFileSystem, LocalSubprocess},
    javascript::JsString,
    llm::{RawValue, ResultString, StopReason, ToolCall, ToolResultContentBlock},
    tools::{
        ToolCapabilityError, ToolExecutionOptions, ToolExecutionSignal,
        builtin::{
            BashToolOptions, create_bash_tool, create_find_tool, create_grep_tool,
            search_engines::{SearchEngine, SearchEngines},
        },
        execute_tool_calls,
    },
};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
#[path = "../src/spawn_nul_evidence.rs"]
mod evidence;
#[path = "support/fs_containment.rs"]
mod fs_containment;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        Self(fs_containment::sandbox("spawn-nul"))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs_containment::cleanup(&self.0);
    }
}
struct EngineNames;
#[async_trait]
impl SearchEngines for EngineNames {
    async fn resolve(
        &self,
        engine: SearchEngine,
        _: &ExecutionWorldIdentity,
    ) -> Result<PathBuf, ToolCapabilityError> {
        Ok(PathBuf::from(engine.executable()))
    }
}
#[derive(Default)]
struct Signal(AtomicBool);
impl ToolExecutionSignal for Signal {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
fn string(v: &Value) -> JsString {
    JsString::from_code_units(
        v["utf16"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| u.as_u64().unwrap() as u16)
            .collect(),
    )
}

#[tokio::test]
async fn spawn_nul_seven_canonical_tools_through_real_pipeline() {
    let data: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../conformance/agent/spawn-nul/spawn-nul.json"
    )))
    .unwrap();
    let platform = if cfg!(windows) { "win32" } else { "linux" };
    let rows = data["spawn_nul"]["tools"].as_array().unwrap();
    assert_eq!(rows.len(), 7);
    for row in rows {
        let fixture = Fixture::new();
        let fs = Arc::new(LocalFileSystem::new(&fixture.0));
        let subprocess = Arc::new(LocalSubprocess::new(&fixture.0));
        let engines = Arc::new(EngineNames);
        let name = row["tool"].as_str().unwrap();
        let tool = match name {
            "find" => create_find_tool(fs, subprocess, engines).unwrap(),
            "grep" => create_grep_tool(fs, subprocess, engines).unwrap(),
            "bash" => create_bash_tool(
                fs,
                subprocess,
                BashToolOptions {
                    shell_path: Some(
                        if cfg!(windows) {
                            "C:/Program Files/Git/bin/bash.exe"
                        } else {
                            "/bin/bash"
                        }
                        .into(),
                    ),
                    ..BashToolOptions::default()
                },
            )
            .unwrap(),
            _ => panic!("unsupported canonical tool"),
        };
        let runtime = Runtime::new();
        runtime.tools().register_for_scope(None, tool).unwrap();
        let mut arguments: RawValue = serde_json::from_str("{}").unwrap();
        let RawValue::Object(object) = &mut arguments else {
            unreachable!()
        };
        for (key, value) in row["arguments"][platform].as_object().unwrap() {
            object.insert(key.as_str().into(), RawValue::String(string(value)));
        }
        let signal = Arc::new(Signal::default());
        signal
            .0
            .store(row["pre_aborted"].as_bool().unwrap(), Ordering::SeqCst);
        let batch = execute_tool_calls(
            &runtime.context(),
            &[ToolCall::new_raw("nul", name, arguments)],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_signal(signal),
        )
        .await
        .unwrap();
        let message = &batch.messages[0];
        let expected = &row["expect"][platform];
        assert_eq!(
            message.is_error,
            expected["is_error"].as_bool().unwrap(),
            "{}: is_error",
            row["id"]
        );
        assert_eq!(message.content.len(), 1);
        let ToolResultContentBlock::Text(text) = &message.content[0] else {
            panic!("text error expected")
        };
        assert_eq!(
            text.text,
            ResultString::from(string(&expected["text"])),
            "{}: exact tool error units",
            row["id"]
        );
    }
}

#[tokio::test]
async fn spawn_nul_tools_preserve_lone_surrogate_and_cut_units() {
    evidence::install();
    for (units, inspected) in [
        (
            vec![0xd800, 0, 98],
            "'\\ud800\\x00b'".encode_utf16().collect::<Vec<_>>(),
        ),
        (
            {
                let mut u = vec![65; 126];
                u.extend([0xd83d, 0xde00, 0]);
                u
            },
            {
                let mut u = vec![39];
                u.extend(vec![65; 126]);
                u.push(0xd83d);
                u.extend("...".encode_utf16());
                u
            },
        ),
    ] {
        for (name, index) in [("find", 7), ("grep", 5), ("bash", 1)] {
            let fixture = Fixture::new();
            let fs = Arc::new(LocalFileSystem::new(&fixture.0));
            let subprocess = Arc::new(LocalSubprocess::new(&fixture.0));
            let engines = Arc::new(EngineNames);
            let tool = match name {
                "find" => create_find_tool(fs, subprocess, engines).unwrap(),
                "grep" => create_grep_tool(fs, subprocess, engines).unwrap(),
                _ => create_bash_tool(
                    fs,
                    subprocess,
                    BashToolOptions {
                        shell_path: Some(
                            if cfg!(windows) {
                                "C:/Program Files/Git/bin/bash.exe"
                            } else {
                                "/bin/bash"
                            }
                            .into(),
                        ),
                        ..BashToolOptions::default()
                    },
                )
                .unwrap(),
            };
            let runtime = Runtime::new();
            runtime.tools().register_for_scope(None, tool).unwrap();
            let mut arguments: RawValue = serde_json::from_str("{}").unwrap();
            let RawValue::Object(object) = &mut arguments else {
                unreachable!()
            };
            object.insert(
                if name == "bash" { "command" } else { "pattern" }.into(),
                RawValue::String(JsString::from_code_units(units.clone())),
            );
            let batch = execute_tool_calls(
                &runtime.context(),
                &[ToolCall::new_raw("lossless", name, arguments)],
                ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
            )
            .await
            .unwrap();
            let ToolResultContentBlock::Text(text) = &batch.messages[0].content[0] else {
                panic!("text result required")
            };
            let mut expected: Vec<u16> = format!(
                "The argument 'args[{index}]' must be a string without null bytes. Received "
            )
            .encode_utf16()
            .collect();
            expected.extend_from_slice(&inspected);
            assert_eq!(
                text.text.code_units(),
                expected,
                "nul-tool-lossless-unit-message"
            );
            assert!(batch.messages[0].is_error, "nul-tool-lossless-is-error");
        }
    }
}
