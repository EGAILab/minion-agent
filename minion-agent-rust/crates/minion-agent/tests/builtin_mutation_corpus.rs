#![cfg(feature = "conformance")]

#[path = "support/mutation_fs.rs"]
mod fixture;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use minion_agent::{
    Runtime,
    llm::{StopReason, ToolCall},
    tools::{
        ToolExecutionOptions,
        builtin::{create_edit_tool, create_write_tool},
        execute_tool_calls,
    },
};
use serde_json::Value;
use std::{path::Path, sync::Arc};

#[tokio::test]
async fn provider_gate_released_before_arrival_is_level_triggered() {
    use minion_agent::execution::FileSystem;
    let dir = tempfile::tempdir().unwrap();
    let gate = Arc::new(fixture::Gate::new(
        serde_json::json!({"id":"gate", "operation":"absolute_path", "path":"."}),
    ));
    gate.release(None);
    let fs = fixture::FixtureFs::new(
        dir.path(),
        "p",
        Value::Null,
        Arc::default(),
        vec![gate],
        fixture::Signal::default(),
        None,
    );
    assert!(fs.absolute_path(".", None).await.is_ok());
    assert_eq!(*fs.calls.lock(), vec!["absolute_path ."]);
}

fn cases(source: &str) -> Vec<Value> {
    source
        .split("\n    - id:")
        .skip(1)
        .filter_map(|block| {
            let mut lines = block.lines();
            let body = format!(
                "id:{}\n{}",
                lines.next().unwrap(),
                lines
                    .map(|l| l.strip_prefix("      ").unwrap_or(l))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            match serde_yaml::from_str(&body) {
                Ok(case) => Some(case),
                Err(e) => {
                    assert!(body.contains("unpaired_surrogate_arguments: true"), "{e}");
                    assert!(serde_json::from_str::<Value>(r#""\ud800""#).is_err());
                    // This legacy scalar-JSON subset does not decode UTF-16 YAML.
                    // The lossless 417-case runner executes this case separately.
                    None
                }
            }
        })
        .collect()
}

fn bytes(file: &Value) -> Vec<u8> {
    if let Some(encoded) = file["base64"].as_str() {
        STANDARD.decode(encoded).unwrap()
    } else {
        file["text"].as_str().unwrap().as_bytes().to_vec()
    }
}

#[tokio::test]
async fn scalar_json_authority_subset_runs_through_real_tools_and_layer_six() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/builtin-mutation");
    let mut count = 0;
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .contains("-corpus-")
                || p.file_name().unwrap() == "builtin-write-corpus.yaml"
        })
        .collect();
    paths.sort();
    for path in paths {
        for case in cases(&std::fs::read_to_string(&path).unwrap()) {
            let dir = tempfile::tempdir().unwrap();
            for fixture in case["fixture"].as_array().into_iter().flatten() {
                let path = dir.path().join(fixture["path"].as_str().unwrap());
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, bytes(&fixture["file"])).unwrap();
            }
            let fs = Arc::new(fixture::FixtureFs::new(
                dir.path(),
                "p",
                Value::Null,
                Arc::default(),
                vec![],
                fixture::Signal::default(),
                None,
            ));
            let runtime = Runtime::new();
            runtime
                .tools()
                .register_for_scope(None, create_edit_tool(fs.clone()))
                .unwrap();
            runtime
                .tools()
                .register_for_scope(None, create_write_tool(fs.clone()))
                .unwrap();
            let call = ToolCall::new(
                "case",
                case["tool"].as_str().unwrap(),
                serde_json::from_value(case["arguments"].clone()).unwrap(),
            );
            let batch = execute_tool_calls(
                &runtime.context(),
                &[call],
                ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
            )
            .await
            .unwrap();
            let result = &batch.messages[0];
            let expected = &case["expect"];
            assert_eq!(
                result.is_error,
                expected["is_error"].as_bool().unwrap(),
                "{} {}",
                path.display(),
                case["id"]
            );
            let serialized = serde_json::to_value(result).unwrap();
            if expected["argument_validation_failure"] != true {
                assert_eq!(
                    serialized["content"][0]["text"], expected["text"],
                    "{}",
                    case["id"]
                );
                assert_eq!(serialized["details"], expected["details"], "{}", case["id"]);
            }
            if let Some(files) = expected["files_after"].as_array() {
                for file in files {
                    assert_eq!(
                        std::fs::read(dir.path().join(file["path"].as_str().unwrap())).unwrap(),
                        bytes(file),
                        "{}",
                        case["id"]
                    );
                }
            }
            if let Some(expected) = expected.get("fs_calls") {
                assert_eq!(
                    serde_json::to_value(fs.calls.lock().clone()).unwrap(),
                    *expected,
                    "{}",
                    case["id"]
                );
            }
            count += 1;
        }
    }
    assert_eq!(
        count, 372,
        "372 scalar subset cases; all 374 are covered by the lossless runner"
    );
    eprintln!(
        "scalar JSON subset: {count}; full UTF-16 corpus is exercised by builtin_mutation_conformance"
    );
}
