use super::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Mutex};

#[cfg(windows)]
#[derive(Debug, Default)]
struct DeniedHandles(std::sync::atomic::AtomicUsize);
#[cfg(windows)]
#[async_trait]
impl HandleOperations for DeniedHandles {
    async fn read_end(&self, _: &mut tokio::fs::File, _: &mut Vec<u8>) -> io::Result<usize> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(io::Error::from_raw_os_error(5))
    }
    async fn read_line(
        &self,
        _: &mut BufReader<tokio::fs::File>,
        _: &mut Vec<u8>,
    ) -> io::Result<usize> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(io::Error::from_raw_os_error(5))
    }
    async fn write(&self, _: &mut tokio::fs::File, _: &[u8]) -> io::Result<()> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(io::Error::from_raw_os_error(5))
    }
}

#[cfg(windows)]
#[tokio::test]
async fn actual_provider_handle_failures_are_unknown_before_eof_and_keep_the_logical_path() {
    let fixture = sandbox();
    let path = FsPath::from_code_units(vec![102, 0xd800]);
    let native_path = fixture.path().join("f\u{fffd}");
    prove(fixture.path(), &native_path, false);
    std::fs::write(&native_path, b"x").unwrap();
    let operations = Arc::new(DeniedHandles::default());
    let fs = LocalFileSystem {
        handle_operations: operations.clone(),
        ..LocalFileSystem::new(fixture.path())
    };
    let logical = fs.resolved(&path);
    for result in [
        fs.read_text_file(&path, None).await.map(|_| ()),
        fs.read_text_lines(&path, None, None).await.map(|_| ()),
        fs.read_binary_file(&path, None).await.map(|_| ()),
        {
            prove(fixture.path(), &native_path, true);
            fs.write_file(&path, b"x", None).await
        },
        {
            prove(fixture.path(), &native_path, true);
            fs.append_file(&path, b"x", None).await
        },
    ] {
        let error = result.unwrap_err();
        assert_eq!(
            error.code,
            FsErrorCode::Unknown,
            "handle failure is not EOF or an open error"
        );
        assert_eq!(error.path.as_ref(), Some(&logical));
    }
    assert_eq!(operations.0.load(std::sync::atomic::Ordering::SeqCst), 5);
    clean(fixture.path());
}

#[derive(Debug)]
struct DeepTree {
    metadata: std::fs::Metadata,
    root: PathBuf,
    directories: usize,
    visited: Mutex<std::collections::HashSet<PathBuf>>,
}

#[derive(Debug)]
struct VanishingChild {
    root: PathBuf,
    removed: std::sync::atomic::AtomicBool,
}
#[async_trait]
impl Operations for VanishingChild {
    async fn lstat(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        if path == self.root.join("tree/child") {
            prove(&self.root, path, false);
            NativeOperations.unlink(path).await?;
            self.removed
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        NativeOperations.lstat(path).await
    }
}

#[tokio::test]
async fn a_child_vanishing_between_enumeration_and_lstat_is_removed_on_every_platform() {
    let fixture = sandbox();
    let root = fixture.path();
    let tree = root.join("tree");
    prove(root, &tree, false);
    std::fs::create_dir(&tree).unwrap();
    let child = tree.join("child");
    prove(root, &child, false);
    std::fs::write(&child, b"x").unwrap();
    let operations = Arc::new(VanishingChild {
        root: root.to_owned(),
        removed: false.into(),
    });
    let fs = LocalFileSystem {
        remove_operations: operations.clone(),
        ..LocalFileSystem::new(root)
    };
    fs.remove("tree", true, false, None).await.unwrap();
    assert!(operations.removed.load(std::sync::atomic::Ordering::SeqCst));
    assert!(!tree.exists());
    let missing = fs.remove("missing", true, false, None).await.unwrap_err();
    assert_eq!(missing.code, FsErrorCode::NotFound);
    clean(root);
}
#[async_trait]
impl Operations for DeepTree {
    async fn lstat(&self, _: &Path) -> io::Result<std::fs::Metadata> {
        Ok(self.metadata.clone())
    }
    async fn rmdir(&self, path: &Path) -> io::Result<()> {
        if path.strip_prefix(&self.root).unwrap().components().count() + 1 < self.directories
            && !self.visited.lock().unwrap().contains(path)
        {
            Err(io::Error::from(io::ErrorKind::DirectoryNotEmpty))
        } else {
            Ok(())
        }
    }
    async fn readdir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        self.visited.lock().unwrap().insert(path.to_owned());
        Ok(vec![path.join("d")])
    }
}

/// The injected operations own a virtual tree; no long-path native mutation is
/// attempted. The real provider still owns validation, classification and walk.
#[test]
fn removal_walk_uses_constant_interpreter_stack_for_deep_trees() {
    let fixture = sandbox();
    let root = fixture.path().join("tree");
    prove(fixture.path(), &root, false);
    std::fs::create_dir(&root).unwrap();
    let operations = Arc::new(DeepTree {
        metadata: std::fs::metadata(&root).unwrap(),
        root: root.clone(),
        directories: 2048,
        visited: Mutex::new(std::collections::HashSet::new()),
    });
    let observer = operations.clone();
    let fs = LocalFileSystem {
        remove_operations: operations,
        ..LocalFileSystem::new(fixture.path())
    };
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(fs.remove("tree", true, false, None))
                .unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(observer.visited.lock().unwrap().len(), 2047);
    clean(fixture.path());
}

// Fixed matrix fixtures are closed graphs inside an explicitly project-local
// sandbox. Every mutation checks its addressed parent, then its referent where
// that operation follows links. Cleanup is a no-follow per-entry walk.
fn prove(root: &Path, path: &Path, follow: bool) {
    assert!(path.starts_with(root) && path != root);
    assert!(!path.components().any(|c| matches!(c, Component::ParentDir)));
    let canonical = std::fs::canonicalize(root).unwrap();
    let canonical = canonical.to_string_lossy();
    assert_eq!(
        Path::new(canonical.strip_prefix(r"\\?\").unwrap_or(&canonical)),
        root
    );
    let mut pending = if follow {
        path.to_owned()
    } else {
        path.parent().unwrap().to_owned()
    };
    let mut seen = std::collections::HashSet::new();
    for _ in 0..64 {
        assert!(pending.starts_with(root));
        let parts = pending
            .components()
            .skip(root.components().count())
            .collect::<Vec<_>>();
        let mut current = root.to_owned();
        let mut redirect = false;
        for (i, part) in parts.iter().enumerate() {
            let Component::Normal(part) = part else {
                panic!("non-normal fixture component")
            };
            current.push(part);
            match std::fs::symlink_metadata(&current) {
                Ok(m) if m.file_type().is_symlink() => {
                    let link = std::fs::read_link(&current).unwrap();
                    assert!(!link.components().any(|c| matches!(c, Component::ParentDir)));
                    let mut next = if link.is_absolute() {
                        link
                    } else {
                        current.parent().unwrap().join(link)
                    };
                    assert!(next.starts_with(root));
                    for part in &parts[i + 1..] {
                        next.push(part.as_os_str());
                    }
                    if !seen.insert((current.clone(), next.clone())) {
                        return;
                    }
                    pending = next;
                    redirect = true;
                    break;
                }
                Ok(m) if !m.is_dir() && i + 1 < parts.len() => return,
                Ok(_) => (),
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                    ) =>
                {
                    return;
                }
                Err(e) => panic!("unproven fixture target {path:?}: {e}"),
            }
        }
        if !redirect {
            return;
        }
    }
    panic!("fixture resolution budget exhausted");
}
fn relative(root: &Path, rel: &str) -> PathBuf {
    assert!(!rel.is_empty() && !rel.contains(':') && !rel.starts_with(['/', '\\']));
    assert!(rel.split('/').all(|p| !matches!(p, "." | "..")));
    root.join(rel)
}
fn sandbox() -> tempfile::TempDir {
    let base = env::temp_dir();
    #[cfg(windows)]
    assert!(std::fs::canonicalize(&base).unwrap().starts_with(
        std::fs::canonicalize("E:/AI/Projects/OpenMinds/Minions/Minion-Agent").unwrap()
    ));
    #[cfg(not(windows))]
    assert!(base.starts_with("/tmp"));
    tempfile::tempdir_in(base).unwrap()
}
fn clean(root: &Path) {
    let mut stack = std::fs::read_dir(root)
        .unwrap()
        .map(|e| (e.unwrap().path(), false))
        .collect::<Vec<_>>();
    while let Some((p, visited)) = stack.pop() {
        prove(root, &p, false);
        let meta = std::fs::symlink_metadata(&p).unwrap();
        if meta.is_dir() && !meta.file_type().is_symlink() && !visited {
            stack.push((p.clone(), true));
            stack.extend(
                std::fs::read_dir(&p)
                    .unwrap()
                    .map(|e| (e.unwrap().path(), false)),
            );
        } else {
            #[cfg(windows)]
            minion_agent_native_fs::delete_entry(&p, meta.is_dir()).unwrap();
            #[cfg(not(windows))]
            if meta.is_dir() && !meta.file_type().is_symlink() {
                std::fs::remove_dir(&p).unwrap();
            } else {
                std::fs::remove_file(&p).unwrap();
            }
        }
    }
}
fn listing(root: &Path) -> (Vec<String>, Vec<String>) {
    let mut left = Vec::new();
    let mut readonly = Vec::new();
    let mut stack = vec![root.to_owned()];
    while let Some(p) = stack.pop() {
        for e in std::fs::read_dir(p).unwrap() {
            let p = e.unwrap().path();
            prove(root, &p, false);
            let m = std::fs::symlink_metadata(&p).unwrap();
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if m.file_type().is_symlink() {
                left.push(format!("{rel}@"));
            } else if m.is_dir() {
                left.push(format!("{rel}/"));
                stack.push(p);
            } else {
                left.push(rel.clone());
            }
            #[cfg(windows)]
            if m.permissions().readonly() {
                readonly.push(rel);
            }
        }
    }
    left.sort();
    readonly.sort();
    (left, readonly)
}
fn errno(code: &str) -> io::Error {
    let n = match (cfg!(windows), code) {
        (true, "EIO") => 1117,
        (true, "EACCES") => 1920,
        (true, "EBUSY") => 32,
        (true, "ENOENT") => 2,
        (true, "EPERM") => 5,
        (true, "ENOTDIR") => return io::Error::from(io::ErrorKind::NotADirectory),
        (false, "EIO") => 5,
        (false, "EACCES") => 13,
        (false, "EBUSY") => 16,
        (false, "ENOENT") => 2,
        (false, "EPERM") => 1,
        (false, "ENOTDIR") => 20,
        _ => panic!("unexpected injected errno {code}"),
    };
    io::Error::from_raw_os_error(n)
}
#[derive(Debug)]
struct Injected {
    root: PathBuf,
    faults: Vec<Value>,
    calls: Mutex<BTreeMap<String, usize>>,
}
impl Injected {
    fn before(&self, op: &str, path: &Path) -> io::Result<()> {
        prove(&self.root, path, op == "stat" || op == "readdir");
        let rel = path
            .strip_prefix(&self.root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let key = format!("{op} {rel}");
        let mut calls = self.calls.lock().unwrap();
        let nth = calls.entry(key).or_default();
        *nth += 1;
        for f in &self.faults {
            if f["op"] == op && f["rel"] == rel && f["nth"].as_u64().unwrap() == *nth as u64 {
                return Err(errno(f["code"].as_str().unwrap()));
            }
        }
        Ok(())
    }
}
#[async_trait]
impl Operations for Injected {
    async fn lstat(&self, p: &Path) -> io::Result<std::fs::Metadata> {
        self.before("lstat", p)?;
        NativeOperations.lstat(p).await
    }
    async fn stat(&self, p: &Path) -> io::Result<std::fs::Metadata> {
        self.before("stat", p)?;
        NativeOperations.stat(p).await
    }
    async fn unlink(&self, p: &Path) -> io::Result<()> {
        self.before("unlink", p)?;
        NativeOperations.unlink(p).await
    }
    async fn rmdir(&self, p: &Path) -> io::Result<()> {
        self.before("rmdir", p)?;
        NativeOperations.rmdir(p).await
    }
    async fn readdir(&self, p: &Path) -> io::Result<Vec<PathBuf>> {
        self.before("readdir", p)?;
        NativeOperations.readdir(p).await
    }
    async fn chmod(&self, p: &Path) -> io::Result<()> {
        self.before("chmod", p)?;
        NativeOperations.chmod(p).await
    }
}

async fn row(spec: &Value, recovery: bool) -> Value {
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
        } else if let Some(rel) = item["ro"].as_str() {
            #[cfg(windows)]
            {
                let p = relative(root, rel);
                prove(root, &p, true);
                let mut permissions = std::fs::metadata(&p).unwrap().permissions();
                permissions.set_readonly(true);
                std::fs::set_permissions(&p, permissions).unwrap();
            }
            #[cfg(not(windows))]
            let _ = rel;
        } else {
            let p = relative(root, item["link"].as_str().unwrap());
            let to = relative(root, item["to"].as_str().unwrap());
            prove(root, &p, false);
            prove(root, &to, true);
            #[cfg(windows)]
            std::os::windows::fs::symlink_file(to, p).unwrap();
            #[cfg(not(windows))]
            std::os::unix::fs::symlink(to, p).unwrap();
        }
    }
    let faults = match &spec["inject"] {
        Value::Array(v) => v.clone(),
        Value::Null => vec![],
        v => vec![v.clone()],
    };
    let ops = Arc::new(Injected {
        root: root.to_owned(),
        faults,
        calls: Mutex::new(BTreeMap::new()),
    });
    let mut fs = LocalFileSystem::new(root);
    fs.remove_operations = ops.clone();
    let rel = spec["remove"]["path"].as_str().unwrap_or("tree");
    let result = fs
        .remove(
            rel,
            spec["remove"]["recursive"].as_bool().unwrap_or(true),
            spec["remove"]["force"].as_bool().unwrap_or(false),
            None,
        )
        .await;
    let (code, path) = match result {
        Ok(()) => (json!("ok"), Value::Null),
        Err(e) => (
            json!(e.code),
            e.path
                .map(|p| {
                    let p = native(&p);
                    json!(
                        p.strip_prefix(root)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/")
                    )
                })
                .unwrap_or(Value::Null),
        ),
    };
    let (left, ro) = listing(root);
    let calls = ops.calls.lock().unwrap().clone();
    let unfired = ops
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
        .collect::<Vec<_>>();
    let mut observed = json!({"platform": if cfg!(windows) { "win32" } else { "linux" }, "id":spec["id"],"result":code,"path":path,"calls":calls,"remains":left,"unfired":unfired});
    if recovery {
        observed["readonly"] = json!(ro);
    }
    clean(root);
    observed
}

fn view(row: &Value) -> Value {
    let calls = row["calls"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| {
            **k != "lstat tree"
                && matches!(
                    k.split(' ').next().unwrap(),
                    "rmdir" | "unlink" | "readdir" | "lstat"
                )
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<serde_json::Map<_, _>>();
    let mut v =
        json!({"result":row["result"],"path":row["path"],"remains":row["remains"],"calls":calls});
    if let Some(ro) = row.get("readonly") {
        v["readonly"] = ro.clone();
    }
    v
}

#[tokio::test]
async fn l12d007_all_41_pi_routing_rows_match_and_injections_fire() {
    let matrices = [
        (
            include_str!("../ce02-data/ce02-matrix.json"),
            include_str!("../ce02-data/ce02-pi-win32.jsonl"),
            include_str!("../ce02-data/ce02-pi-linux.jsonl"),
            false,
        ),
        (
            include_str!("../ce02-data/ce02-recovery-matrix.json"),
            include_str!("../ce02-data/ce02-recovery-pi-win32.jsonl"),
            include_str!("../ce02-data/ce02-recovery-pi-linux.jsonl"),
            true,
        ),
        (
            include_str!("../ce02-data/ce02-validation-matrix.json"),
            include_str!("../ce02-data/ce02-validation-pi-win32.jsonl"),
            include_str!("../ce02-data/ce02-validation-pi-linux.jsonl"),
            true,
        ),
    ];
    let mut count = 0;
    let selected = env::var("L12D007_CASE").ok();
    for (matrix, win, linux, recovery) in matrices {
        let matrix: Value = serde_json::from_str(matrix).unwrap();
        let oracle = if cfg!(windows) { win } else { linux }
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect::<Vec<_>>();
        for spec in matrix["rows"].as_array().unwrap() {
            if selected
                .as_ref()
                .is_some_and(|id| spec["id"] != id.as_str())
            {
                continue;
            }
            let expected = oracle.iter().find(|r| r["id"] == spec["id"]).unwrap();
            let actual = row(spec, recovery).await;
            println!("CE02 {}", serde_json::to_string(&actual).unwrap());
            assert_eq!(
                actual["unfired"],
                expected.get("unfired").cloned().unwrap_or(json!([])),
                "binding-only UNFIRED: {}",
                spec["id"]
            );
            assert_eq!(view(&actual), view(expected), "Pi routing: {}", spec["id"]);
            count += 1;
        }
    }
    assert_eq!(count, if selected.is_some() { 1 } else { 41 });
}
