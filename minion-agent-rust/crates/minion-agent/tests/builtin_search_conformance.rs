#![cfg(feature = "conformance")]
//! Fixture construction + real Layer 06 execution + contract-defined observations only.
use minion_agent::{
    Runtime,
    execution::{LocalFileSystem, LocalSubprocess},
    llm::{StopReason, ToolCall},
    tools::{
        ToolExecutionOptions, ToolExecutionSignal,
        builtin::{
            create_find_tool, create_grep_tool,
            search_engines::{
                SearchEngine, SearchEngineStore, SearchEngines, provision_search_engines,
            },
        },
        execute_tool_calls,
    },
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
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
fn hex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
        .collect()
}
fn link(target: &Path, path: &Path, directory: bool) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    #[cfg(windows)]
    {
        if directory {
            let result = std::process::Command::new("cmd")
                .args(["/c", "mklink", "/J"])
                .arg(path.to_string_lossy().replace('/', "\\"))
                .arg(target.to_string_lossy().replace('/', "\\"))
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "junction {:?} -> {:?}: {} {}",
                path,
                target,
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        } else {
            std::os::windows::fs::symlink_file(target, path).unwrap();
        }
    }
    #[cfg(unix)]
    {
        let _ = directory;
        std::os::unix::fs::symlink(target, path).unwrap();
    }
}
fn fixture(root: &Path, spec: &Value) {
    for (name, data) in spec["files"].as_object().unwrap() {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = if let Some(hexadecimal) = data["hex"].as_str() {
            hex(hexadecimal)
        } else {
            data["text"].as_str().unwrap().as_bytes().to_vec()
        };
        std::fs::write(path, bytes).unwrap();
    }
    for dir in spec["directories"].as_array().unwrap() {
        std::fs::create_dir_all(root.join(dir.as_str().unwrap())).unwrap();
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        if let Some(names) = spec["raw_name_files"]["linux"].as_object() {
            for (relative, data) in names {
                let parent = root.join(relative).parent().unwrap().to_path_buf();
                std::fs::create_dir_all(&parent).unwrap();
                let path = parent.join(std::ffi::OsString::from_vec(hex(data["name_hex"]
                    .as_str()
                    .unwrap())));
                std::fs::write(path, data["text"].as_str().unwrap()).unwrap();
            }
        }
    }
    for entry in spec["links"].as_array().unwrap() {
        link(
            &root.join(entry["target"].as_str().unwrap()),
            &root.join(entry["path"].as_str().unwrap()),
            entry["kind"] == "dir",
        );
    }
    if spec["dangling_git_junction"] == true {
        let target = root.join("gone-target");
        std::fs::create_dir(&target).unwrap();
        link(&target, &root.join(".git"), true);
        std::fs::remove_dir(target).unwrap();
    }
}
fn normalize(value: &mut Value, root: &str) {
    match value {
        Value::String(s) => {
            *s = s
                .replace(root, "<ROOT>")
                .replace(&root.replace('\\', "/"), "<ROOT>");
        }
        Value::Object(m) => {
            for v in m.values_mut() {
                normalize(v, root);
            }
        }
        Value::Array(a) => {
            for v in a {
                normalize(v, root);
            }
        }
        _ => {}
    }
}
fn multiset(lines: impl IntoIterator<Item = String>) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for l in lines {
        *out.entry(l).or_default() += 1;
    }
    out
}
fn file_key(line: &str) -> &str {
    for (index, c) in line.char_indices() {
        if c == ':' || c == '-' {
            let tail = &line[index + 1..];
            if let Some((number, _)) = tail.split_once(if c == ':' { ": " } else { "- " })
                && number.parse::<f64>().is_ok()
            {
                return &line[..index];
            }
        }
    }
    ""
}
fn compare(text: &str, mut details: Value, is_error: bool, expected: &Value) {
    assert_eq!(is_error, expected["is_error"].as_bool().unwrap());
    let (body, notice) = if expected["mode"] == "exact" {
        (text, None)
    } else {
        match text.split_once("\n\n[") {
            Some((body, n)) => (body, Some(format!("[{n}"))),
            None => (text, None),
        }
    };
    if expected["mode"] == "exact" {
        assert_eq!(text, expected["text"].as_str().unwrap());
    } else {
        assert_eq!(json!(notice), expected["notice"]);
        match expected["mode"].as_str().unwrap() {
            "find_multiset" => assert_eq!(
                multiset(body.split('\n').map(str::to_owned)),
                multiset(
                    expected["entries"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                )
            ),
            "find_subset" => {
                let entries = multiset(body.split('\n').map(str::to_owned));
                assert_eq!(
                    entries.values().sum::<usize>(),
                    expected["count"].as_u64().unwrap() as usize
                );
                let full = multiset(
                    expected["of"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned()),
                );
                assert!(
                    entries
                        .iter()
                        .all(|(k, n)| full.get(k).is_some_and(|m| n <= m))
                );
            }
            "grep_by_file" => {
                let mut blocks: Vec<(&str, Vec<&str>)> = Vec::new();
                let mut seen = BTreeSet::new();
                for line in body.split('\n') {
                    let key = file_key(line);
                    if blocks.last().is_none_or(|(k, _)| *k != key) {
                        assert!(seen.insert(key), "noncontiguous file block");
                        blocks.push((key, Vec::new()));
                    }
                    blocks.last_mut().unwrap().1.push(line);
                }
                let partial = expected["details"].get("truncation").is_some()
                    || expected["details"].get("matchLimitReached").is_some();
                let full = expected["files"].as_object().unwrap();
                for (i, (key, lines)) in blocks.iter().enumerate() {
                    let allowed: Vec<_> = full
                        .get(*key)
                        .expect("unexpected file")
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap())
                        .collect();
                    if partial && i == blocks.len() - 1 {
                        assert_eq!(lines, &allowed[..lines.len().min(allowed.len())]);
                    } else {
                        assert_eq!(lines, &allowed);
                    }
                }
                if !partial {
                    assert_eq!(seen.len(), full.len());
                } else if expected["details"].get("truncation").is_none() {
                    assert_eq!(
                        blocks.iter().map(|(_, b)| b.len()).sum::<usize>(),
                        full.values()
                            .map(|v| v.as_array().unwrap().len())
                            .sum::<usize>()
                    );
                }
            }
            other => panic!("unknown comparison {other}"),
        }
    }
    if let Some(truncation) = details.get_mut("truncation") {
        let content = truncation
            .as_object_mut()
            .unwrap()
            .remove("content")
            .unwrap();
        assert_eq!(content, body);
    }
    assert_eq!(details, expected["details"]);
}
#[tokio::test]
async fn canonical_search_real_engines_and_layer_six() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/builtin-search");
    let corpora: Value =
        serde_json::from_slice(&std::fs::read(root.join("corpus.json")).unwrap()).unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let store = Arc::new(SearchEngineStore::new(store_dir.path()));
    let source = std::env::var_os("MINION_SEARCH_ENGINE_ARTIFACTS")
        .expect("pinned artifacts are mandatory, no vacuous skip");
    provision_search_engines(&store, Some(Path::new(&source)))
        .await
        .unwrap();
    for engine in [SearchEngine::Fd, SearchEngine::Ripgrep] {
        let path = store
            .resolve(
                engine,
                &minion_agent::execution::ExecutionWorldIdentity::local(),
            )
            .await
            .unwrap();
        let output = std::process::Command::new(path)
            .arg("--version")
            .output()
            .unwrap();
        assert!(
            String::from_utf8(output.stdout)
                .unwrap()
                .contains(engine.version())
        );
    }
    let platform = if cfg!(windows) { "win32" } else { "linux" };
    let mut paths: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
        .collect();
    paths.sort();
    let total = paths.len();
    let selected = std::env::var("MINION_SEARCH_CASE").ok();
    let mut count = 0;
    for path in paths {
        let doc: Value = serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        if selected
            .as_ref()
            .is_some_and(|name| doc["name"].as_str() != Some(name))
        {
            continue;
        }
        let case = &doc["builtin_search"];
        let expected = &case["expect"][platform];
        if expected.is_null() {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path(), &corpora[case["corpus"].as_str().unwrap()]);
        let fs = Arc::new(LocalFileSystem::new(dir.path()));
        let subprocess = Arc::new(LocalSubprocess::new(dir.path()));
        let tool = if case["tool"] == "find" {
            create_find_tool(fs, subprocess, store.clone())
        } else {
            create_grep_tool(fs, subprocess, store.clone())
        }
        .unwrap();
        let runtime = Runtime::new();
        runtime.tools().register_for_scope(None, tool).unwrap();
        let mut args = case["arguments"].clone();
        let dirname = dir.path().to_string_lossy().into_owned();
        for v in args.as_object_mut().unwrap().values_mut() {
            if let Value::String(s) = v {
                *s = s.replace("<ROOT>", &dirname);
            }
        }
        let signal = Arc::new(Signal::default());
        signal
            .0
            .store(case["signal"] == "pre_aborted", Ordering::SeqCst);
        let batch = execute_tool_calls(
            &runtime.context(),
            &[ToolCall::new(
                "case",
                case["tool"].as_str().unwrap(),
                args.as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_signal(signal),
        )
        .await
        .unwrap();
        let mut result = serde_json::to_value(&batch.messages[0]).unwrap();
        normalize(&mut result, &dirname);
        eprintln!("search case {}", doc["name"]);
        compare(
            result["content"][0]["text"].as_str().unwrap(),
            result["details"].clone(),
            result["is_error"].as_bool().unwrap(),
            expected,
        );
        count += 1;
    }
    assert_eq!(total, 219);
    assert_eq!(
        count,
        if selected.is_some() { 1 } else { 218 },
        "no vacuous canonical run"
    );
    eprintln!("search: {count}/{total} canonical scenarios on {platform}");
}

#[test]
fn comparisons_reject_fabrication_duplicates_and_wrong_order() {
    let e = json!({"is_error":false,"mode":"find_subset","count":2,"of":["a","b","c"],"notice":null,"details":{}});
    compare("c\nb", json!({}), false, &e);
    assert!(std::panic::catch_unwind(|| compare("Q\nQ", json!({}), false, &e)).is_err());
    assert!(std::panic::catch_unwind(|| compare("a\na", json!({}), false, &e)).is_err());
    let e = json!({"is_error":false,"mode":"grep_by_file","files":{"a":["a:1: x","a:2: y"]},"notice":null,"details":{}});
    assert!(std::panic::catch_unwind(|| compare("a:2: y\na:1: x", json!({}), false, &e)).is_err());
}
