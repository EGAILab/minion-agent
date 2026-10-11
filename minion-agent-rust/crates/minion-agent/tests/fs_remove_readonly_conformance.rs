//! Fixture-only adapter: observations come from the real LocalFileSystem::remove.
use minion_agent::execution::LocalFileSystem;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[path = "support/fs_containment.rs"]
mod containment;

fn guard(root: &Path, path: &Path, entry: bool) {
    if path == root {
        containment::check(&containment::base(), root);
    } else if entry {
        containment::check_entry(root, path);
    } else {
        containment::check(root, path);
    }
}
fn target(root: &Path, rel: &str) -> PathBuf {
    assert!(!rel.is_empty() && !rel.contains(':') && !rel.starts_with(['/', '\\']));
    assert!(rel.split(['/', '\\']).all(|p| !matches!(p, "." | "..")));
    let path = root.join(rel);
    guard(root, &path, true);
    path
}

fn command(name: &str, args: &[&str]) {
    let output = std::process::Command::new(name)
        .args(args)
        .output()
        .unwrap();
    assert!(output.status.success(), "{name} {args:?}: {:?}", output);
}

struct Fixture {
    root: PathBuf,
    permissions: Vec<PathBuf>,
    acls: Vec<PathBuf>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        #[cfg(windows)]
        for path in self.acls.iter().rev() {
            guard(&self.root, path, true);
            match fs::symlink_metadata(path) {
                Ok(meta) if !meta.file_type().is_symlink() => (),
                Ok(_) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => panic!("ACL restore target not inspectable: {e}"),
            }
            guard(&self.root, path, false);
            command(
                "icacls",
                &[
                    path.to_str().unwrap(),
                    "/remove:d",
                    &std::env::var("USERNAME").unwrap(),
                ],
            );
        }
        for path in &self.permissions {
            guard(&self.root, path, true);
            if !path.try_exists().unwrap() {
                continue;
            }
            #[cfg(windows)]
            {
                minion_agent_native_fs::clear_readonly_entry(path).unwrap();
            }
            #[cfg(unix)]
            if let Ok(metadata) = fs::metadata(path) {
                use std::os::unix::fs::PermissionsExt;
                guard(&self.root, path, false);
                fs::set_permissions(
                    path,
                    fs::Permissions::from_mode(if metadata.is_dir() { 0o755 } else { 0o644 }),
                )
                .unwrap();
            }
        }
        containment::cleanup(&self.root);
    }
}
fn prepare(root: &Path, steps: &Value, fixture: &mut Fixture) {
    for step in steps.as_array().unwrap() {
        if let Some(rel) = step["file"].as_str() {
            let path = target(root, rel);
            guard(root, path.parent().unwrap(), false);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            guard(root, &path, false);
            fs::write(path, step["text"].as_str().unwrap_or("x")).unwrap();
        } else if let Some(rel) = step["dir"].as_str() {
            let path = target(root, rel);
            guard(root, &path, false);
            fs::create_dir_all(path).unwrap();
        } else if let Some(rel) = step["readonly"].as_str() {
            let path = target(root, rel);
            guard(root, &path, false);
            #[cfg(windows)]
            command("attrib", &["+R", path.to_str().unwrap()]);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(
                    &path,
                    fs::Permissions::from_mode(if path.is_dir() { 0o555 } else { 0o444 }),
                )
                .unwrap();
            }
            fixture.permissions.push(path);
        } else if let Some(rel) = step["readonly_link"].as_str() {
            let path = target(root, rel);
            command("attrib", &["+R", "/L", path.to_str().unwrap()]);
            fixture.permissions.push(path);
        } else if let Some(rel) = step["symlink"].as_str() {
            let destination = target(root, step["to"].as_str().unwrap());
            guard(root, &destination, false);
            let link = target(root, rel);
            #[cfg(windows)]
            if step["kind"] == "dir" {
                std::os::windows::fs::symlink_dir(destination, link).unwrap();
            } else {
                std::os::windows::fs::symlink_file(destination, link).unwrap();
            }
            #[cfg(unix)]
            std::os::unix::fs::symlink(destination, link).unwrap();
        } else {
            let (rel, access) = if let Some(rel) = step["deny_delete"].as_str() {
                (rel, "D")
            } else {
                (step["deny_write_attributes"].as_str().unwrap(), "WA")
            };
            let path = target(root, rel);
            guard(root, &path, false);
            let user = std::env::var("USERNAME").unwrap();
            command(
                "icacls",
                &[
                    path.to_str().unwrap(),
                    "/deny",
                    &format!("{user}:({access})"),
                ],
            );
            fixture.acls.push(path.clone());
            if access == "D" {
                let parent = path.parent().unwrap();
                guard(root, parent, false);
                command(
                    "icacls",
                    &[parent.to_str().unwrap(), "/deny", &format!("{user}:(DC)")],
                );
                fixture.acls.push(parent.to_owned());
            }
        }
    }
}
fn left(root: &Path, path: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(path) else {
        return vec![];
    };
    let mut entries = entries.map(Result::unwrap).collect::<Vec<_>>();
    entries.sort_by_key(|e| e.file_name());
    let mut result = vec![];
    for entry in entries {
        let p = entry.path();
        let rel = p
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let kind = fs::symlink_metadata(&p).unwrap().file_type();
        if kind.is_symlink() {
            result.push(format!("{rel}@"));
        } else if kind.is_dir() {
            result.push(format!("{rel}/"));
            result.extend(left(root, &p));
        } else {
            result.push(rel);
        }
    }
    result
}
fn readonly(path: &Path) -> bool {
    #[cfg(windows)]
    {
        fs::metadata(path).unwrap().permissions().readonly()
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).unwrap().permissions().mode() & 0o222 == 0
    }
}

#[tokio::test]
async fn canonical_fs_remove_readonly() {
    let corpus =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/fs-remove-readonly");
    let mut documents = fs::read_dir(corpus)
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    documents.sort_by_key(|e| e.file_name());
    assert_eq!(documents.len(), 21);
    let platform = if cfg!(windows) { "win32" } else { "linux" };
    let mut run = 0;
    for document in documents {
        let doc: Value = serde_json::from_slice(&fs::read(document.path()).unwrap()).unwrap();
        if !doc["platforms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p == platform)
        {
            continue;
        }
        let case = &doc["fs_remove"];
        #[cfg(unix)]
        if nix::unistd::Uid::effective().is_root() && case["expect"].get("error").is_some() {
            eprintln!("SKIP privileged host: {}", doc["name"]);
            continue;
        }
        let mut fixture = Fixture {
            root: containment::sandbox("minion-readonly"),
            permissions: vec![],
            acls: vec![],
        };
        let root = fixture.root.clone();
        prepare(&root, &case["fixture"], &mut fixture);
        let fs = LocalFileSystem::new(&root);
        containment::argument(&fs, &root, &case["remove"]["path"].as_str().unwrap().into()).await;
        let result = fs
            .remove(
                case["remove"]["path"].as_str().unwrap(),
                case["remove"]["recursive"].as_bool().unwrap(),
                false,
                None,
            )
            .await;
        let observed = match result {
            Ok(()) => json!({"ok":true}),
            Err(error) => {
                let path = error.path.map(|p| {
                    let p = PathBuf::from(String::from_utf16(p.code_units()).unwrap());
                    let p = p.strip_prefix(&root).unwrap();
                    p.components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                });
                json!({"error":error.code,"path":path})
            }
        };
        assert_eq!(observed, case["expect"], "{}", doc["name"]);
        assert_eq!(
            json!(left(&root, &root)),
            case["expect_left"],
            "{}",
            doc["name"]
        );
        if let Some(external) = case.get("expect_external") {
            for expected in external.as_array().unwrap() {
                let rel = expected["path"].as_str().unwrap();
                let path = root.join(rel);
                let mut observed = json!({"path":rel,"exists":path.exists()});
                if path.exists() {
                    observed["readonly"] = json!(readonly(&path));
                    if path.is_file() {
                        observed["text"] = json!(fs::read_to_string(&path).unwrap());
                    }
                }
                assert_eq!(&observed, expected, "{}", doc["name"]);
            }
        }
        run += 1;
        eprintln!("PASS {}", doc["name"]);
    }
    assert!(run > 0);
    eprintln!("L12-D005: {run}/21 documents executed on {platform}");
}
