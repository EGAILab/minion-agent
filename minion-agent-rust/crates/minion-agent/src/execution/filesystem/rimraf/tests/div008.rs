//! Probe through the real provider and its native-operation injection seam.
//! Trace labels describe calls, never infer terminality from an opcode.
use super::*;

#[derive(Debug)]
struct TraceOperations {
    inner: Injected,
    trace: Mutex<Vec<String>>,
    scripted_order: bool,
}
impl TraceOperations {
    fn call(&self, op: &str, p: &Path) -> io::Result<()> {
        let rel = p
            .strip_prefix(&self.inner.root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        self.trace.lock().unwrap().push(format!("{op} {rel}"));
        let result = self.inner.before(op, p);
        if let Err(e) = &result {
            self.error("settle", op, &rel, e);
        }
        result
    }
    fn error(&self, event: &str, op: &str, rel: &str, e: &io::Error) {
        let code = match e.kind() {
            _ if e.raw_os_error() == errno("EACCES").raw_os_error() => "EACCES",
            _ if e.raw_os_error() == errno("EPERM").raw_os_error() => "EPERM",
            io::ErrorKind::DirectoryNotEmpty => "ENOTEMPTY",
            io::ErrorKind::PermissionDenied => "EACCES",
            io::ErrorKind::IsADirectory => "EISDIR",
            io::ErrorKind::NotADirectory => "ENOTDIR",
            io::ErrorKind::NotFound => "ENOENT",
            _ if e.raw_os_error() == errno("EBUSY").raw_os_error() => "EBUSY",
            _ => panic!("unclassified probe failure: {e:?}"),
        };
        self.trace
            .lock()
            .unwrap()
            .push(format!("{event} {op} {rel} {code}"));
    }
    fn finish<T>(&self, op: &str, p: &Path, result: io::Result<T>) -> io::Result<T> {
        if let Err(e) = &result {
            let rel = p
                .strip_prefix(&self.inner.root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            self.error("fail", op, &rel, e);
        }
        result
    }
}
#[async_trait]
impl Operations for TraceOperations {
    async fn lstat(&self, p: &Path) -> io::Result<std::fs::Metadata> {
        self.call("lstat", p)?;
        self.finish("lstat", p, NativeOperations.lstat(p).await)
    }
    async fn stat(&self, p: &Path) -> io::Result<std::fs::Metadata> {
        self.call("stat", p)?;
        self.finish("stat", p, NativeOperations.stat(p).await)
    }
    async fn unlink(&self, p: &Path) -> io::Result<()> {
        self.call("unlink", p)?;
        self.finish("unlink", p, NativeOperations.unlink(p).await)
    }
    async fn rmdir(&self, p: &Path) -> io::Result<()> {
        self.call("rmdir", p)?;
        self.finish("rmdir", p, NativeOperations.rmdir(p).await)
    }
    async fn readdir(&self, p: &Path) -> io::Result<Vec<PathBuf>> {
        self.call("readdir", p)?;
        let mut entries = self.finish("readdir", p, NativeOperations.readdir(p).await)?;
        // Only the named propagation witness scripts its dependency's order.
        // Matrix probes retain native enumeration order without sorting.
        if self.scripted_order {
            entries.sort();
        }
        Ok(entries)
    }
    async fn chmod(&self, p: &Path) -> io::Result<()> {
        self.call("chmod", p)?;
        self.finish("chmod", p, NativeOperations.chmod(p).await)
    }
}

async fn probe(spec: &Value, run: usize) -> Value {
    let temp = sandbox();
    let root = temp.path();
    for item in spec["fixture"].as_array().unwrap() {
        if let Some(rel) = item.as_str() {
            let p = relative(root, rel.trim_end_matches('/'));
            prove(root, &p, true);
            if rel.ends_with('/') {
                std::fs::create_dir(&p).unwrap();
            } else {
                std::fs::write(&p, b"x").unwrap();
            }
        } else {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let p = relative(root, item["ro_dir"].as_str().unwrap());
                prove(root, &p, true);
                std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o555)).unwrap();
            }
            #[cfg(windows)]
            panic!("POSIX fixture on Windows");
        }
    }
    let ops = Arc::new(TraceOperations {
        scripted_order: spec["scripted_order"] == true,
        inner: Injected {
            root: root.to_owned(),
            faults: spec["inject"].as_array().cloned().unwrap_or_default(),
            calls: Mutex::new(BTreeMap::new()),
        },
        trace: Mutex::new(Vec::new()),
    });
    let fs = LocalFileSystem {
        remove_operations: ops.clone(),
        ..LocalFileSystem::new(root)
    };
    let result = fs.remove("tree", true, false, None).await;
    let (code, path) = match result {
        Ok(()) => (json!("ok"), Value::Null),
        Err(e) => (
            json!(e.code),
            json!(
                native(&e.path.unwrap())
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            ),
        ),
    };
    let snapshot =
        json!({"calls":ops.inner.calls.lock().unwrap().clone(), "remains":listing(root).0});
    let calls = ops.inner.calls.lock().unwrap();
    let unfired: Vec<_> = ops
        .inner
        .faults
        .iter()
        .filter_map(|f| {
            let key = format!(
                "{} {}",
                f["op"].as_str().unwrap(),
                f["rel"].as_str().unwrap()
            );
            (calls.get(&key).copied().unwrap_or(0) < f["nth"].as_u64().unwrap() as usize)
                .then(|| format!("{key} #{}", f["nth"]))
        })
        .collect();
    drop(calls);
    let actual = json!({"platform":if cfg!(windows) {"win32"} else {"linux"}, "id":spec["id"], "run":run,
        "result":code, "path":path, "seq":ops.trace.lock().unwrap().clone(), "at_settle":snapshot, "after_drain":snapshot,
        "late_calls":[], "drained":true, "settle_incomplete":false, "unfired":unfired});
    // Restore only proven, existing ordinary entries before guarded cleanup.
    #[cfg(unix)]
    for item in spec["fixture"].as_array().unwrap() {
        if let Some(rel) = item["ro_dir"].as_str() {
            use std::os::unix::fs::PermissionsExt;
            let p = relative(root, rel);
            prove(root, &p, true);
            let m = std::fs::symlink_metadata(&p).unwrap();
            assert!(m.is_dir() && !m.file_type().is_symlink());
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    clean(root);
    actual
}

#[tokio::test]
async fn div008_controlled_rows_continue_and_preserve_drained_extent() {
    let matrix: Value =
        serde_json::from_str(include_str!("../../div008-data/div008-matrix.json")).unwrap();
    let oracle = if cfg!(windows) {
        include_str!("../../div008-data/div008-pi-win32.jsonl")
    } else {
        include_str!("../../div008-data/div008-pi-linux.jsonl")
    };
    let oracle: Vec<Value> = oracle
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let natural = env::var("DIV008_NATURAL").is_ok();
    let mut count = 0;
    for spec in matrix["rows"].as_array().unwrap() {
        if spec.get("platforms").is_some() && (!natural || cfg!(windows)) {
            continue;
        }
        for run in 1..=spec["repeat"].as_u64().unwrap_or(1) as usize {
            let actual = probe(spec, run).await;
            println!("DIV008 {}", serde_json::to_string(&actual).unwrap());
            let pi = oracle
                .iter()
                .find(|v| v["id"] == spec["id"] && v["run"] == run)
                .unwrap();
            assert_eq!(
                actual["unfired"],
                json!([]),
                "injection completeness: {}",
                spec["id"]
            );
            assert_eq!(
                actual["after_drain"]["remains"], pi["after_drain"]["remains"],
                "drained extent: {}",
                spec["id"]
            );
            assert_eq!(
                actual["result"] == "ok",
                pi["result"] == "ok",
                "terminal vs recovered outcome: {}",
                spec["id"]
            );
            count += 1;
        }
    }
    assert_eq!(count, if natural && !cfg!(windows) { 45 } else { 25 });
}

#[tokio::test]
async fn div008_reports_first_terminal_error_unchanged_not_last() {
    let actual = probe(
        &json!({"id":"first-not-last", "fixture":["tree/","tree/a","tree/b"],
        "inject":[{"op":"unlink","rel":"tree/a","nth":1,"code":"EACCES"},
                  {"op":"unlink","rel":"tree/b","nth":1,"code":"EBUSY"}]}),
        1,
    )
    .await;
    assert_eq!(
        actual["unfired"],
        json!([]),
        "both terminal injections must fire"
    );
    let first = actual["seq"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .find(|s| s.starts_with("settle unlink "))
        .unwrap();
    let words: Vec<_> = first.split(' ').collect();
    assert_eq!(actual["path"], words[2], "first terminal origin, not last");
    assert_eq!(
        actual["result"],
        if words[3] == "EACCES" {
            "permission_denied"
        } else {
            "unknown"
        },
        "first terminal code unchanged"
    );
}

#[tokio::test]
async fn div008_failed_subtree_keeps_ancestors_but_successful_sibling_directory_is_removed() {
    let actual = probe(
        &json!({"id":"subtree-local", "scripted_order":true, "fixture":["tree/","tree/a/","tree/a/x","tree/b/","tree/b/y"],
        "inject":[{"op":"unlink","rel":"tree/a/x","nth":1,"code":"EACCES"}]}),
        1,
    )
    .await;
    assert_eq!(actual["unfired"], json!([]));
    assert_eq!(actual["result"], "permission_denied");
    assert_eq!(actual["path"], "tree/a/x");
    assert_eq!(
        actual["after_drain"]["remains"],
        json!(["tree/", "tree/a/", "tree/a/x"]),
        "continue siblings, retaining only the failed subtree and ancestors"
    );
    assert_eq!(
        actual["after_drain"]["calls"]["rmdir tree/a"], 1,
        "no final rmdir of failed subtree"
    );
    assert_eq!(
        actual["after_drain"]["calls"]["rmdir tree"], 1,
        "no final rmdir of failed ancestor"
    );
    assert_eq!(
        actual["after_drain"]["calls"]["rmdir tree/b"], 2,
        "unaffected sibling must finish"
    );
}
