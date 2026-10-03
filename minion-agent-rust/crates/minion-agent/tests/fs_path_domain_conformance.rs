//! Thin adapter for the accepted L12-D001 corpus. All path behavior is production-owned.
use minion_agent::{
    execution::{FileKind, FileSystem, FsError, FsPath, LocalFileSystem},
    llm::{RawNumber, RawString, RawValue, StopReason, ToolCall, ToolResultContentBlock},
    runtime::Runtime,
    tools::{
        ToolExecutionOptions,
        builtin::{
            ReadToolOptions, create_edit_tool, create_ls_tool, create_read_tool, create_write_tool,
        },
        execute_tool_calls,
    },
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

fn units(value: &Value) -> Vec<u16> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|u| u.as_u64().unwrap() as u16)
        .collect()
}
fn scalar_path(path: &Path) -> String {
    let p = path.to_string_lossy();
    if cfg!(windows) {
        p.strip_prefix("\\\\?\\").unwrap_or(&p).to_owned()
    } else {
        p.into_owned()
    }
}
fn input(value: &Value, root: &Path) -> FsPath {
    if let Some(u) = value.get("utf16") {
        return FsPath::from_code_units(units(u));
    }
    let mut prefix = url::Url::from_directory_path(root)
        .unwrap()
        .to_string()
        .encode_utf16()
        .collect::<Vec<_>>();
    prefix.extend(units(&value["file_url_tail"]));
    FsPath::from_code_units(prefix)
}
fn path_observation(path: &FsPath, root: &Path) -> Value {
    let root: Vec<u16> = scalar_path(root).encode_utf16().collect();
    let value = path.code_units();
    if !value.starts_with(&root) {
        return json!({"outside":true});
    }
    let rest = &value[root.len()..];
    let components: Vec<Vec<u16>> = rest
        .split(|u| *u == 47 || (cfg!(windows) && *u == 92))
        .filter(|p| !p.is_empty())
        .map(<[u16]>::to_vec)
        .collect();
    json!({"components":components})
}
fn error_observation(error: FsError, root: &Path) -> Value {
    json!({"error": error.code, "path":error.path.as_ref().map(|p| path_observation(p, root))})
}
async fn step(fs: &LocalFileSystem, root: &Path, spec: &Value) -> Result<Value, FsError> {
    let path = input(&spec["path"], root);
    let content = spec
        .get("content")
        .map(units)
        .map(|u| String::from_utf16(&u).unwrap().into_bytes())
        .unwrap_or_default();
    let recursive = spec["recursive"].as_bool().unwrap_or(false);
    match spec["op"].as_str().unwrap() {
        "write_file" => fs.write_file(&path, &content, None).await.map(|_| json!({"ok":null})),
        "append_file" => fs.append_file(&path, &content, None).await.map(|_| json!({"ok":null})),
        "create_dir" => fs.create_dir(&path, recursive, None).await.map(|_| json!({"ok":null})),
        "remove" => fs.remove(&path, recursive, false, None).await.map(|_| json!({"ok":null})),
        "rename_file" => fs.rename_file(&path, input(&spec["to"],root), None).await.map(|_| json!({"ok":null})),
        "read_text_file" => fs.read_text_file(&path, None).await.map(|s| json!({"ok":s.encode_utf16().collect::<Vec<_>>()})),
        "read_text_lines" => fs.read_text_lines(&path, None, None).await.map(|lines| json!({"ok":lines.iter().map(|s| s.encode_utf16().collect::<Vec<_>>()).collect::<Vec<_>>()})),
        "read_binary_file" => fs.read_binary_file(&path, None).await.map(|bytes| json!({"ok":bytes})),
        "exists" => fs.exists(&path, None).await.map(|v| json!({"ok":v})),
        "absolute_path" => fs.absolute_path(&path, None).await.map(|p| {
            if spec["observe"] == "last_component" {
                let last = p.code_units().rsplit(|u| *u==47 || (cfg!(windows) && *u==92)).next().unwrap();
                json!({"last":last})
            } else { path_observation(&p,root) }
        }),
        "canonical_path" => fs.canonical_path(&path, None).await.map(|p| path_observation(&p,root)),
        "target_key" => {
            let target = fs.resolve(&path,None).await?;
            fs.process_path(&target).await.map(|p| path_observation(&p,root))
        },
        "file_info" => fs.file_info(&path,None).await.map(|i| json!({"kind":match i.kind {FileKind::File=>"file",FileKind::Directory=>"directory",FileKind::Symlink=>"symlink"},"name":i.name.code_units()})),
        "list_dir" => fs.list_dir(&path,None).await.map(|entries| {
            let mut names: Vec<Vec<u16>> = entries.into_iter().map(|i| i.name.code_units().to_vec()).collect();
            names.sort(); json!({"names":names})
        }),
        op => panic!("unhandled operation {op}"),
    }
}

fn raw(value: &Value) -> RawValue {
    if value.get("utf16").is_some() {
        return RawValue::String(RawString::from_code_units(units(&value["utf16"])));
    }
    match value {
        Value::Null => RawValue::Null,
        Value::Bool(v) => RawValue::Bool(*v),
        Value::Number(v) => RawValue::Number(RawNumber::new(v.as_f64().unwrap()).unwrap()),
        Value::String(v) => RawValue::String(v.as_str().into()),
        Value::Array(v) => RawValue::Array(v.iter().map(raw).collect()),
        Value::Object(v) => {
            RawValue::Object(v.iter().map(|(k, v)| (k.as_str().into(), raw(v))).collect())
        }
    }
}
async fn tools(fs: Arc<LocalFileSystem>, spec: &Value) -> Value {
    let runtime = Runtime::new();
    for tool in [
        create_write_tool(fs.clone()),
        create_read_tool(fs.clone(), ReadToolOptions::default()),
        create_ls_tool(fs.clone()),
        create_edit_tool(fs),
    ] {
        runtime.tools().register_for_scope(None, tool).unwrap();
    }
    let RawValue::Object(arguments) = raw(&spec["arguments"]) else {
        panic!()
    };
    let call = ToolCall::new_raw(
        "case",
        spec["tool"].as_str().unwrap(),
        RawValue::Object(arguments),
    );
    let batch = execute_tool_calls(
        &runtime.context(),
        &[call],
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
    )
    .await
    .unwrap();
    let message = &batch.messages[0];
    let ToolResultContentBlock::Text(text) = &message.content[0] else {
        panic!()
    };
    json!({"is_error":message.is_error,"text":{"utf16":text.text.code_units()}})
}

#[tokio::test]
async fn full_accepted_fs_path_domain_corpus() {
    let directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/fs-path-domain");
    let platform = if cfg!(windows) { "win32" } else { "linux" };
    let mut ran = BTreeMap::new();
    let mut skipped = 0;
    for file in std::fs::read_dir(directory).unwrap() {
        let path = file.unwrap().path();
        let doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let family = if doc.get("fs_path_domain").is_some() {
            "fs_path_domain"
        } else {
            "fs_path_tools"
        };
        let mut count = 0;
        for case in doc[family]["cases"].as_array().unwrap() {
            if let Some(platforms) = case["platforms"].as_array()
                && !platforms.iter().any(|p| p == platform)
            {
                eprintln!("excluded {}: {}", case["id"], case["platform_note"]);
                skipped += 1;
                continue;
            }
            let temp = std::env::temp_dir().join(format!("minion-fs-domain-{}", Uuid::new_v4()));
            std::fs::create_dir(&temp).unwrap();
            let root = PathBuf::from(scalar_path(&std::fs::canonicalize(&temp).unwrap()));
            let fs = Arc::new(LocalFileSystem::new(&root));
            for (i, spec) in case["steps"].as_array().unwrap().iter().enumerate() {
                let actual = if family == "fs_path_tools" {
                    tools(fs.clone(), spec).await
                } else {
                    step(&fs, &root, spec)
                        .await
                        .unwrap_or_else(|e| error_observation(e, &root))
                };
                let expected = spec
                    .get("expect")
                    .unwrap_or(&spec["expect_by_platform"][platform]);
                assert_eq!(
                    &actual, expected,
                    "{} / {} step {i}",
                    doc["name"], case["id"]
                );
            }
            std::fs::remove_dir_all(temp).unwrap();
            count += 1;
        }
        ran.insert(doc["name"].as_str().unwrap().to_owned(), count);
    }
    assert_eq!(
        ran.values().sum::<usize>(),
        if cfg!(windows) { 117 } else { 127 }
    );
    assert_eq!(skipped, if cfg!(windows) { 10 } else { 0 });
    eprintln!("L12-D001 cases: {ran:?}; excluded {skipped}");
}
