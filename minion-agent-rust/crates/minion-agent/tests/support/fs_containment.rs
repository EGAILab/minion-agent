//! Test-only safety boundary. Never changes filesystem observations/expectations.
#![allow(dead_code)]
use minion_agent::execution::{AbortSignal, FsError, FsPath, LocalFileSystem};
use std::{
    io,
    ops::Deref,
    path::{Component, Path, PathBuf},
};

fn plain(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let text = path.to_string_lossy();
        PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text))
    }
    #[cfg(not(windows))]
    {
        path
    }
}

pub fn base() -> PathBuf {
    #[cfg(windows)]
    let base = PathBuf::from("E:/AI/Projects/OpenMinds/Minions/Minion-Agent/.tmp/process-temp");
    #[cfg(not(windows))]
    let base = {
        let value = std::env::var_os("MINION_FIXTURE_ROOT")
            .expect("explicit container fixture sandbox required");
        let path = PathBuf::from(value);
        assert!(
            path.starts_with("/tmp") && path != Path::new("/tmp"),
            "container-private fixture sandbox required"
        );
        path
    };
    assert_eq!(
        plain(std::fs::canonicalize(&base).expect("fixture base must exist")),
        base,
        "fixture base must not traverse links"
    );
    base
}

/// Both lexical and real containment, including missing/NUL leaves. Fail closed
/// on unresolved links, permission errors, parent traversal, or the sandbox root.
pub fn check(root: &Path, target: &Path) {
    assert!(
        target.is_absolute() && target.starts_with(root) && target != root,
        "fixture target outside sandbox: {target:?}"
    );
    assert!(
        !target
            .components()
            .any(|c| matches!(c, Component::ParentDir)),
        "fixture parent traversal"
    );
    let real_root = plain(std::fs::canonicalize(root).expect("sandbox must exist"));
    assert_eq!(real_root, root, "sandbox ancestry must be link-free");
    let mut pending = target.to_path_buf();
    let mut visited = std::collections::HashSet::new();
    for _ in 0..64 {
        assert!(pending.starts_with(root));
        let mut current = root.to_owned();
        // Do not reparse a relative suffix: `b:name` becomes a drive prefix
        // when detached from its proven absolute parent, although it is an
        // ordinary (invalid-on-Windows) filename in the addressed full path.
        let components = pending
            .components()
            .skip(root.components().count())
            .collect::<Vec<_>>();
        let mut redirected = false;
        for (i, part) in components.iter().enumerate() {
            let Component::Normal(name) = part else {
                panic!("non-normal fixture component")
            };
            if name.to_string_lossy().contains('\0') {
                return;
            }
            current.push(name);
            match std::fs::symlink_metadata(&current) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    let link = std::fs::read_link(&current).expect("link must be inspectable");
                    assert!(
                        !link.components().any(|p| matches!(p, Component::ParentDir)),
                        "dot-dot link text refused"
                    );
                    let mut next = if link.is_absolute() {
                        plain(link)
                    } else {
                        current.parent().unwrap().join(link)
                    };
                    for c in &components[i + 1..] {
                        next.push(c.as_os_str());
                    }
                    assert!(next.starts_with(root), "outward fixture link");
                    // Each link is checked before a repeated state is credited.
                    if !visited.insert((current.clone(), next.clone())) {
                        return;
                    }
                    pending = next;
                    redirected = true;
                    break;
                }
                Ok(meta) if !meta.is_dir() && i + 1 < components.len() => return,
                Ok(_) => (),
                Err(e)
                    if e.kind() == io::ErrorKind::NotFound
                        || e.kind() == io::ErrorKind::NotADirectory =>
                {
                    return;
                }
                #[cfg(windows)]
                Err(e) if matches!(e.raw_os_error(), Some(123 | 161)) => return,
                #[cfg(unix)]
                Err(e) if e.raw_os_error() == Some(36) => {
                    use std::os::unix::ffi::OsStrExt;
                    let dir = current.parent().unwrap();
                    assert_eq!(plain(std::fs::canonicalize(dir).unwrap()), dir);
                    let limit = |key| {
                        let output = std::process::Command::new("getconf")
                            .arg(key)
                            .arg(dir)
                            .output()
                            .unwrap();
                        assert!(output.status.success());
                        let value: usize = String::from_utf8(output.stdout)
                            .unwrap()
                            .trim()
                            .parse()
                            .unwrap();
                        assert!(value > 0);
                        value
                    };
                    assert!(name.as_bytes().len() > limit("NAME_MAX"));
                    assert!(pending.as_os_str().as_bytes().len() < limit("PATH_MAX"));
                    return;
                }
                Err(e) => panic!("fixture containment cannot be established: {e}"),
            }
        }
        if !redirected {
            return;
        }
    }
    panic!("fixture proof budget exhausted");
}

pub fn sandbox(label: &str) -> PathBuf {
    assert!(
        label
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
    );
    let base = base();
    let target = base.join(format!("{label}-{}", uuid::Uuid::new_v4()));
    check(&base, &target);
    std::fs::create_dir(&target).unwrap();
    target
}

pub fn cleanup(target: &Path) {
    let base = base();
    let mut stack = vec![(target.to_path_buf(), false)];
    while let Some((path, visited)) = stack.pop() {
        check_entry(&base, &path);
        let meta = std::fs::symlink_metadata(&path).unwrap();
        if meta.file_type().is_symlink() {
            check_entry(&base, &path);
            #[cfg(windows)]
            minion_agent_native_fs::delete_entry(&path, meta.is_dir()).unwrap();
            #[cfg(not(windows))]
            std::fs::remove_file(&path).unwrap();
        } else if meta.is_dir() && !visited {
            // A successful provider rename may move a mode-000 directory away
            // from its recorded restore path. Cleanup owns this *new* entry,
            // proves it independently, and never restores via the stale path.
            #[cfg(unix)]
            {
                check(&base, &path);
                nix::sys::stat::fchmodat(
                    nix::fcntl::AT_FDCWD,
                    &path,
                    nix::sys::stat::Mode::S_IRWXU,
                    nix::sys::stat::FchmodatFlags::NoFollowSymlink,
                )
                .unwrap();
            }
            stack.push((path.clone(), true));
            for entry in std::fs::read_dir(&path).unwrap() {
                stack.push((entry.unwrap().path(), false));
            }
        } else if meta.is_dir() {
            check(&base, &path);
            std::fs::remove_dir(&path).unwrap();
        } else {
            check(&base, &path);
            std::fs::remove_file(&path).unwrap();
        }
    }
}

/// ENTRY proof for a verified no-follow primitive; never authorizes referent IO.
pub fn check_entry(root: &Path, target: &Path) {
    assert!(target.starts_with(root) && target != root);
    assert!(
        !target
            .components()
            .any(|c| matches!(c, Component::ParentDir))
    );
    let parent = target.parent().unwrap();
    if parent != root {
        check(root, parent);
    } else {
        assert_eq!(plain(std::fs::canonicalize(root).unwrap()), root);
    }
    assert!(matches!(
        target.components().next_back(),
        Some(Component::Normal(_))
    ));
}

pub async fn argument(fs: &LocalFileSystem, root: &Path, path: &FsPath) {
    let text = String::from_utf16_lossy(path.code_units());
    if !text.starts_with("file://") {
        relative(&text);
    }
    // File URLs are constructed by the fixture, not accepted as arbitrary
    // absolute targets. Resolution below must still stay inside this sandbox.
    let resolved = fs
        .absolute_path(path, None)
        .await
        .expect("fixture resolution failed");
    let native = PathBuf::from(String::from_utf16_lossy(resolved.code_units()));
    if native == root {
        // A canonical "." may address its own case sandbox. Prove that
        // sandbox from the explicit fixture base; never admit the base itself.
        check(&base(), root);
    } else {
        check(root, &native);
    }
}

fn relative(text: &str) {
    let raw = Path::new(text);
    assert!(
        !raw.is_absolute() && !text.starts_with(['/', '\\']),
        "absolute fixture mutation input"
    );
    assert!(
        !text.split(['/', '\\']).any(|p| p == ".."),
        "fixture parent traversal input"
    );
    assert!(
        !text.as_bytes().get(1).is_some_and(|c| *c == b':'),
        "drive-relative fixture input"
    );
    assert!(
        !text.contains(':') || text.starts_with("./"),
        "colon fixture names require ./"
    );
}

#[test]
fn containment_refuses_unsafe_raw_mutation_names_before_dispatch() {
    for name in [
        "../../..",
        "a/../b",
        "E:",
        "E:foo",
        "E:/",
        "/",
        "\\\\host\\share",
        "bad:name",
    ] {
        assert!(
            std::panic::catch_unwind(|| relative(name)).is_err(),
            "unsafe target accepted: {name}"
        );
    }
    for name in ["leaf", "a/f\0x", "./bad:name"] {
        relative(name);
    }
}

pub struct GuardedFs {
    pub inner: LocalFileSystem,
    root: PathBuf,
}
impl Deref for GuardedFs {
    type Target = LocalFileSystem;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl GuardedFs {
    pub fn new(root: &Path) -> Self {
        Self {
            inner: LocalFileSystem::new(root),
            root: root.to_owned(),
        }
    }
    pub async fn write_file(
        &self,
        path: impl Into<FsPath>,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = path.into();
        argument(&self.inner, &self.root, &path).await;
        self.inner.write_file(path, content, signal).await
    }
    pub async fn append_file(
        &self,
        path: impl Into<FsPath>,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = path.into();
        argument(&self.inner, &self.root, &path).await;
        self.inner.append_file(path, content, signal).await
    }
    pub async fn create_dir(
        &self,
        path: impl Into<FsPath>,
        recursive: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = path.into();
        argument(&self.inner, &self.root, &path).await;
        self.inner.create_dir(path, recursive, signal).await
    }
    pub async fn remove(
        &self,
        path: impl Into<FsPath>,
        recursive: bool,
        force: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = path.into();
        argument(&self.inner, &self.root, &path).await;
        self.inner.remove(path, recursive, force, signal).await
    }
    pub async fn rename_file(
        &self,
        source: impl Into<FsPath>,
        destination: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let source = source.into();
        let destination = destination.into();
        argument(&self.inner, &self.root, &source).await;
        argument(&self.inner, &self.root, &destination).await;
        self.inner.rename_file(source, destination, signal).await
    }
}
