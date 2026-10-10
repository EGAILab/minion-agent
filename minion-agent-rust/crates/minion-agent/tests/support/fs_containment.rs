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
    let mut ancestor = target;
    loop {
        match std::fs::canonicalize(ancestor) {
            Ok(real) => {
                assert!(
                    plain(real).starts_with(&real_root),
                    "fixture link escapes sandbox: {target:?}"
                );
                break;
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound
                        | io::ErrorKind::InvalidInput
                        | io::ErrorKind::NotADirectory
                ) =>
            {
                if let Ok(meta) = std::fs::symlink_metadata(ancestor) {
                    assert!(!meta.file_type().is_symlink(), "unresolved fixture link");
                }
                ancestor = ancestor.parent().expect("no contained ancestor");
            }
            Err(error) => panic!("fixture containment cannot be established: {error}"),
        }
    }
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
        check(&base, &path);
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(
            !meta.file_type().is_symlink(),
            "cleanup refuses fixture links"
        );
        if meta.is_dir() && !visited {
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
    check(
        root,
        &PathBuf::from(String::from_utf16_lossy(resolved.code_units())),
    );
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
