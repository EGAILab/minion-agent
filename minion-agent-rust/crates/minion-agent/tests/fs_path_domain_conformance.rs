//! Thin adapter for the accepted L12-D001 corpus. All path behavior is production-owned.
use minion_agent::{
    execution::{
        AbortSignal, CancellationController, FileKind, FileSystem, FsError, FsPath, LocalFileSystem,
    },
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
#[path = "support/fs_containment.rs"]
mod containment;

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
    let resolved = fs.absolute_path(&path, None).await.unwrap();
    let native = PathBuf::from(String::from_utf16_lossy(resolved.code_units()));
    if native != root {
        containment::check(root, &native);
    }
    if matches!(
        spec["op"].as_str().unwrap(),
        "write_file" | "append_file" | "create_dir" | "remove" | "rename_file"
    ) {
        containment::argument(fs, root, &path).await;
        if spec["op"] == "rename_file" {
            containment::argument(fs, root, &input(&spec["to"], root)).await;
        }
    }
    let content = spec
        .get("content")
        .map(units)
        .map(|u| String::from_utf16(&u).unwrap().into_bytes())
        .unwrap_or_default();
    let recursive = spec["recursive"].as_bool().unwrap_or(false);
    let controller = CancellationController::default();
    controller.abort();
    let cancelled = controller.signal();
    let signal = spec["aborted"]
        .as_bool()
        .unwrap_or(false)
        .then_some(&cancelled as &dyn AbortSignal);
    match spec["op"].as_str().unwrap() {
        "write_file" => fs.write_file(&path, &content, signal).await.map(|_| json!({"ok":null})),
        "append_file" => fs.append_file(&path, &content, None).await.map(|_| json!({"ok":null})),
        "create_dir" => fs.create_dir(&path, recursive, None).await.map(|_| json!({"ok":null})),
        "remove" => fs.remove(&path, recursive, spec["force"].as_bool().unwrap_or(false), None).await.map(|_| json!({"ok":null})),
        "rename_file" => fs.rename_file(&path, input(&spec["to"],root), signal).await.map(|_| json!({"ok":null})),
        "read_text_file" => fs.read_text_file(&path, signal).await.map(|s| json!({"ok":s.encode_utf16().collect::<Vec<_>>()})),
        "read_text_lines" => fs.read_text_lines(&path, spec["max_lines"].as_i64().map(|n| isize::try_from(n).unwrap()), signal).await.map(|lines| json!({"ok":lines.iter().map(|s| s.encode_utf16().collect::<Vec<_>>()).collect::<Vec<_>>()})),
        "read_binary_file" => fs.read_binary_file(&path, signal).await.map(|bytes| json!({"ok":bytes})),
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
        "list_dir" => fs.list_dir(&path,signal).await.map(|entries| {
            let mut names: Vec<Vec<u16>> = entries.into_iter().map(|i| i.name.code_units().to_vec()).collect();
            names.sort(); json!({"names":names})
        }),
        "list_dir_raw" => fs.list_dir_raw(&path, None).await.map(|names| {
            let mut names: Vec<Vec<u16>> = names.iter().map(|s| s.encode_utf16().collect()).collect();
            names.sort(); json!({"names":names})
        }),
        "probe_dir_entry" => fs.probe_dir_entry(&path, None).await.map(|p| json!({"kind":p.kind,"name":p.name.code_units()})),
        "check_readable" => fs.check_readable(&path, None).await.map(|_| json!({"ok":null})),
        "check_read_write" => fs.check_read_write(&path, None).await.map(|_| json!({"ok":null})),
        op => panic!("unhandled operation {op}"),
    }
}

#[derive(Default)]
struct Fixtures {
    holds: Vec<std::fs::File>,
    denied: Vec<(PathBuf, bool, std::fs::Permissions)>,
}
impl Fixtures {
    fn release(&mut self, root: &Path) {
        self.holds.clear();
        for (path, directory, permissions) in self.denied.drain(..) {
            containment::check_entry(root, &path);
            match std::fs::symlink_metadata(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Ok(meta) if !meta.file_type().is_symlink() && meta.is_dir() == directory => (),
                Ok(_) => continue,
                Err(e) => panic!("cannot prove restore target: {e}"),
            }
            containment::check(root, &path);
            #[cfg(windows)]
            {
                let _ = permissions;
                assert!(
                    std::process::Command::new("icacls")
                        .arg(&path)
                        .args(["/remove:d", "*S-1-1-0"])
                        .output()
                        .unwrap()
                        .status
                        .success()
                );
            }
            #[cfg(not(windows))]
            std::fs::set_permissions(&path, permissions).unwrap();
        }
    }
    fn step(&mut self, root: &Path, spec: &Value) -> Option<Value> {
        let op = spec["op"].as_str()?;
        if !matches!(
            op,
            "make_symlink" | "deny_access" | "hold_exclusive" | "lock_range"
        ) {
            return None;
        }
        let relative = String::from_utf16(&units(&spec["path"]["utf16"])).unwrap();
        let path = root.join(relative);
        containment::check(root, &path);
        match op {
            "make_symlink" => {
                containment::check_entry(root, &path);
                let to = String::from_utf16(&units(&spec["to"]["utf16"])).unwrap();
                let destination = path.parent().unwrap().join(&to);
                containment::check(root, &destination);
                #[cfg(windows)]
                std::os::windows::fs::symlink_file(to, &path).unwrap();
                #[cfg(unix)]
                std::os::unix::fs::symlink(to, &path).unwrap();
            }
            "deny_access" => {
                let meta = std::fs::metadata(&path).unwrap();
                self.denied
                    .push((path.clone(), meta.is_dir(), meta.permissions()));
                #[cfg(windows)]
                assert!(
                    std::process::Command::new("icacls")
                        .arg(&path)
                        .args(["/deny", "*S-1-1-0:(RD,REA,RA,S)"])
                        .output()
                        .unwrap()
                        .status
                        .success()
                );
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0)).unwrap();
                }
            }
            "hold_exclusive" | "lock_range" => {
                #[cfg(windows)]
                self.holds
                    .push(minion_agent_native_fs::fixture_hold(&path, op == "lock_range").unwrap());
                #[cfg(not(windows))]
                panic!("Windows-only fixture selected");
            }
            _ => unreachable!(),
        }
        Some(json!({"ok":null}))
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
async fn tools(fs: Arc<LocalFileSystem>, root: &Path, spec: &Value) -> Value {
    if spec["arguments"].get("path").is_some() {
        containment::argument(&fs, root, &input(&spec["arguments"]["path"], root)).await;
    }
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
    #[cfg(unix)]
    let mut privileged_skipped = 0;
    #[cfg(not(unix))]
    let privileged_skipped = 0;
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
            #[cfg(unix)]
            if nix::unistd::Uid::effective().is_root()
                && case["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|s| s["op"] == "deny_access")
            {
                eprintln!(
                    "excluded privileged caller: {} (requires uid-1000 corpus run)",
                    case["id"]
                );
                privileged_skipped += 1;
                continue;
            }
            let temp = containment::sandbox("minion-fs-domain");
            let root = PathBuf::from(scalar_path(&std::fs::canonicalize(&temp).unwrap()));
            let fs = Arc::new(LocalFileSystem::new(&root));
            let mut fixtures = Fixtures::default();
            for (i, spec) in case["steps"].as_array().unwrap().iter().enumerate() {
                let mut actual = if let Some(value) = fixtures.step(&root, spec) {
                    value
                } else if family == "fs_path_tools" {
                    tools(fs.clone(), &root, spec).await
                } else {
                    step(&fs, &root, spec)
                        .await
                        .unwrap_or_else(|e| error_observation(e, &root))
                };
                if doc["name"] == "fs-error-codes"
                    && spec["op"] == "file_info"
                    && actual.get("ok").is_some()
                {
                    actual = actual["ok"].take();
                }
                let expected = spec
                    .get("expect")
                    .unwrap_or(&spec["expect_by_platform"][platform]);
                if &actual != expected {
                    fixtures.release(&root);
                }
                assert_eq!(
                    &actual, expected,
                    "{} / {} step {i}",
                    doc["name"], case["id"]
                );
            }
            fixtures.release(&root);
            containment::cleanup(&temp);
            count += 1;
        }
        ran.insert(doc["name"].as_str().unwrap().to_owned(), count);
    }
    assert_eq!(
        ran.values().sum::<usize>(),
        (if cfg!(windows) {
            127 + 237 + 240
        } else {
            127 + 237 + 160
        }) - privileged_skipped
    );
    assert_eq!(skipped, if cfg!(windows) { 0 } else { 80 });
    assert_eq!(ran["fs-path-nul"], 237);
    #[cfg(unix)]
    assert_eq!(
        privileged_skipped,
        if nix::unistd::Uid::effective().is_root() {
            48
        } else {
            0
        }
    );
    eprintln!(
        "L12-D001 + L12-D006 + L12-D007 cases: {ran:?}; platform excluded {skipped}; privileged excluded {privileged_skipped}"
    );
}
