use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use minion_agent::execution::{
    AbortSignal, CancellationController, DirEntryProbe, DirEntryProbeKind, ExecutionWorldIdentity,
    FileInfo, FileKind, FileSystem, FsError, FsErrorCode, FsTarget, LocalFileSystem,
};
use uuid::Uuid;

fn temp_root() -> PathBuf {
    std::env::temp_dir().join(format!("minion-execution-test-{}", Uuid::new_v4()))
}

#[cfg(unix)]
fn symlink_file(target: impl AsRef<std::path::Path>, link: impl AsRef<std::path::Path>) {
    std::os::unix::fs::symlink(target, link).unwrap();
}

#[cfg(windows)]
fn symlink_file(target: impl AsRef<std::path::Path>, link: impl AsRef<std::path::Path>) {
    std::os::windows::fs::symlink_file(target, link).unwrap();
}

#[cfg(unix)]
fn symlink_directory(target: impl AsRef<std::path::Path>, link: impl AsRef<std::path::Path>) {
    std::os::unix::fs::symlink(target, link).unwrap();
}

struct UnsupportedExtensionProvider {
    cwd: PathBuf,
    world: ExecutionWorldIdentity,
}

#[async_trait::async_trait]
impl FileSystem for UnsupportedExtensionProvider {
    fn cwd(&self) -> &Path {
        &self.cwd
    }

    fn execution_world(&self) -> &ExecutionWorldIdentity {
        &self.world
    }

    async fn absolute_path(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        unreachable!()
    }

    async fn join_path(
        &self,
        _parts: &[&str],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        unreachable!()
    }

    async fn read_text_file(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        unreachable!()
    }

    async fn read_text_lines(
        &self,
        _path: &str,
        _max_lines: Option<isize>,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        unreachable!()
    }

    async fn read_binary_file(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        unreachable!()
    }

    async fn write_file(
        &self,
        _path: &str,
        _content: &[u8],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        unreachable!()
    }

    async fn append_file(
        &self,
        _path: &str,
        _content: &[u8],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        unreachable!()
    }

    async fn rename_file(
        &self,
        _source: &str,
        _destination: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        unreachable!()
    }

    async fn file_info(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        unreachable!()
    }

    async fn list_dir(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        unreachable!()
    }

    async fn canonical_path(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        unreachable!()
    }

    async fn exists(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<bool, FsError> {
        unreachable!()
    }

    async fn create_dir(
        &self,
        _path: &str,
        _recursive: bool,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        unreachable!()
    }

    async fn remove(
        &self,
        _path: &str,
        _recursive: bool,
        _force: bool,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        unreachable!()
    }

    async fn create_temp_dir(
        &self,
        _prefix: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        unreachable!()
    }

    async fn create_temp_file(
        &self,
        _prefix: &str,
        _suffix: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        unreachable!()
    }

    async fn resolve(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        unreachable!()
    }

    async fn process_path(&self, _target: &FsTarget) -> Result<String, FsError> {
        unreachable!()
    }

    async fn cleanup(&self) {
        unreachable!()
    }
}

#[cfg(windows)]
fn symlink_directory(target: impl AsRef<std::path::Path>, link: impl AsRef<std::path::Path>) {
    std::os::windows::fs::symlink_dir(target, link).unwrap();
}

#[tokio::test]
async fn local_filesystem_covers_reads_writes_metadata_and_removal() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    let fs = LocalFileSystem::new(&root);

    fs.write_file("nested/file.txt", b"one\ntwo\nthree\n", None)
        .await
        .unwrap();
    assert_eq!(
        fs.read_text_lines("nested/file.txt", Some(2), None)
            .await
            .unwrap(),
        ["one", "two"]
    );
    fs.append_file("nested/file.txt", b"four\n", None)
        .await
        .unwrap();
    assert_eq!(
        fs.read_text_file("nested/file.txt", None).await.unwrap(),
        "one\ntwo\nthree\nfour\n"
    );
    let info = fs.file_info("nested/file.txt", None).await.unwrap();
    assert_eq!(info.kind, FileKind::File);
    assert_eq!(info.name, "file.txt");
    assert_eq!(fs.list_dir("nested", None).await.unwrap().len(), 1);
    assert!(fs.exists("nested/file.txt", None).await.unwrap());

    fs.rename_file("nested/file.txt", "nested/moved.txt", None)
        .await
        .unwrap();
    assert!(!fs.exists("nested/file.txt", None).await.unwrap());
    assert!(fs.exists("nested/moved.txt", None).await.unwrap());
    fs.remove("nested", true, false, None).await.unwrap();
    assert!(!fs.exists("nested", None).await.unwrap());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn max_lines_at_or_below_zero_does_not_touch_the_file() {
    let root = temp_root();
    let fs = LocalFileSystem::new(&root);
    assert!(
        fs.read_text_lines("missing.txt", Some(0), None)
            .await
            .unwrap()
            .is_empty()
    );
    let controller = CancellationController::default();
    let signal = controller.signal();
    controller.abort();
    assert_eq!(
        fs.read_text_lines("missing.txt", Some(0), Some(&signal))
            .await
            .unwrap_err()
            .code,
        FsErrorCode::Aborted
    );
}

#[tokio::test]
async fn lexical_join_and_utf8_decoding_match_the_pi_surface() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("invalid.txt"), [b'a', 0xff, b'\n']).unwrap();
    let fs = LocalFileSystem::new(&root);
    assert_eq!(fs.join_path(&[], None).await.unwrap(), ".");
    assert_eq!(
        PathBuf::from(fs.join_path(&["a", "/b"], None).await.unwrap()),
        PathBuf::from("a").join("b")
    );
    assert_eq!(
        fs.read_text_file("invalid.txt", None).await.unwrap(),
        "a\u{fffd}\n"
    );
    assert_eq!(
        fs.read_text_lines("invalid.txt", None, None).await.unwrap(),
        ["a\u{fffd}"]
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn rename_replaces_an_existing_destination() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("source.txt"), "new").unwrap();
    std::fs::write(root.join("destination.txt"), "old").unwrap();
    let fs = LocalFileSystem::new(&root);
    fs.rename_file("source.txt", "destination.txt", None)
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(root.join("destination.txt")).unwrap(),
        "new"
    );
    assert!(!root.join("source.txt").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn cancellation_matches_the_operation_specific_contract() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("file.txt"), "content").unwrap();
    let fs = LocalFileSystem::new(&root);
    let controller = CancellationController::default();
    let signal = controller.signal();
    controller.abort();

    assert_eq!(
        fs.read_text_file("file.txt", Some(&signal))
            .await
            .unwrap_err()
            .code,
        FsErrorCode::Aborted
    );
    assert_eq!(
        fs.rename_file("file.txt", "renamed.txt", Some(&signal))
            .await
            .unwrap_err()
            .code,
        FsErrorCode::Aborted
    );
    // Pi accepts but does not inspect the signal on these operations.
    assert!(fs.absolute_path("file.txt", Some(&signal)).await.is_ok());
    assert!(fs.file_info("file.txt", Some(&signal)).await.is_ok());
    fs.append_file("file.txt", b"!", Some(&signal))
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(root.join("file.txt")).unwrap(),
        "content!"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn fs_target_is_location_based_provider_scoped_and_live() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.txt"), "same").unwrap();
    std::fs::write(root.join("b.txt"), "same").unwrap();
    let fs = LocalFileSystem::new(&root);
    let other = LocalFileSystem::new(&root);

    let first = fs.resolve("a.txt", None).await.unwrap();
    fs.write_file("a.txt", b"changed", None).await.unwrap();
    let second = fs.resolve("a.txt", None).await.unwrap();
    let distinct = fs.resolve("b.txt", None).await.unwrap();
    assert_eq!(first.target_key(), second.target_key());
    assert_ne!(first.target_key(), distinct.target_key());
    assert_eq!(
        fs.process_path(&first).await.unwrap(),
        std::fs::canonicalize(root.join("a.txt"))
            .unwrap()
            .to_string_lossy()
    );
    assert_eq!(
        other.process_path(&first).await.unwrap_err().code,
        FsErrorCode::Invalid
    );

    let missing = fs.resolve("future.txt", None).await.unwrap();
    let missing_again = fs.resolve("future.txt", None).await.unwrap();
    assert_eq!(missing.target_key(), missing_again.target_key());
    assert_eq!(
        fs.process_path(&missing).await.unwrap(),
        root.join("future.txt").to_string_lossy()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn temp_file_gets_a_private_directory_and_cleanup_is_a_noop() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    let fs = LocalFileSystem::new(&root);
    let file = PathBuf::from(fs.create_temp_file("pre-", ".tmp", None).await.unwrap());
    assert!(file.is_file());
    assert!(file.parent().unwrap().is_dir());
    fs.cleanup().await;
    assert!(file.is_file());
    std::fs::remove_dir_all(file.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn metadata_does_not_follow_symlinks_but_content_io_does() {
    use std::os::unix::fs::symlink;

    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("target.txt"), "target").unwrap();
    symlink("target.txt", root.join("link.txt")).unwrap();
    let fs = LocalFileSystem::new(&root);
    assert_eq!(
        fs.file_info("link.txt", None).await.unwrap().kind,
        FileKind::Symlink
    );
    assert_eq!(fs.read_text_file("link.txt", None).await.unwrap(), "target");
    let link = fs.resolve("link.txt", None).await.unwrap();
    let target = fs.resolve("target.txt", None).await.unwrap();
    assert_eq!(link.target_key(), target.target_key());
    assert_eq!(
        fs.process_path(&link).await.unwrap(),
        fs.process_path(&target).await.unwrap()
    );
    fs.rename_file("link.txt", "moved.txt", None).await.unwrap();
    assert!(
        std::fs::symlink_metadata(root.join("moved.txt"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        std::fs::read_to_string(root.join("target.txt")).unwrap(),
        "target"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn raw_directory_listing_preserves_provider_order_and_does_not_follow_entries() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("z_first.txt"), "z").unwrap();
    std::fs::write(root.join("a_second.txt"), "a").unwrap();
    symlink_file("missing-target", root.join("broken-link"));

    let expected = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let fs = LocalFileSystem::new(&root);
    let actual = fs.list_dir_raw(".", None).await.unwrap();

    assert_eq!(actual, expected);
    assert!(actual.iter().any(|name| name == "broken-link"));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn layer13_consumption_shape_checks_the_cap_before_the_next_probe() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a_ok"), "a").unwrap();
    std::fs::write(root.join("z_slow"), "z").unwrap();
    let fs = LocalFileSystem::new(&root);

    let mut names = fs.list_dir_raw(".", None).await.unwrap();
    names.sort();
    let limit = 1;
    let mut probed = Vec::new();
    let mut results = Vec::new();
    for name in names {
        if results.len() >= limit {
            break;
        }
        probed.push(name.clone());
        if let Ok(probe) = fs.probe_dir_entry(&name, None).await {
            results.push(probe);
        }
    }

    assert_eq!(probed, ["a_ok"]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "a_ok");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn entry_probe_classifies_plain_and_symlinked_files_and_directories() {
    let root = temp_root();
    std::fs::create_dir_all(root.join("real-directory")).unwrap();
    std::fs::write(root.join("real-file"), "content").unwrap();
    symlink_file("real-file", root.join("file-link"));
    symlink_directory("real-directory", root.join("directory-link"));
    let fs = LocalFileSystem::new(&root);

    assert_eq!(
        fs.probe_dir_entry("real-file", None).await.unwrap().kind,
        DirEntryProbeKind::File
    );
    assert_eq!(
        fs.probe_dir_entry("real-directory", None)
            .await
            .unwrap()
            .kind,
        DirEntryProbeKind::Directory
    );
    let file_link: DirEntryProbe = fs.probe_dir_entry("file-link", None).await.unwrap();
    assert_eq!(file_link.kind, DirEntryProbeKind::SymlinkToFile);
    assert_eq!(file_link.name, "file-link");
    assert_eq!(PathBuf::from(&file_link.path), root.join("file-link"));
    let directory_link = fs.probe_dir_entry("directory-link", None).await.unwrap();
    assert_eq!(directory_link.kind, DirEntryProbeKind::SymlinkToDirectory);
    assert_eq!(directory_link.name, "directory-link");
    assert_eq!(
        PathBuf::from(&directory_link.path),
        root.join("directory-link")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn entry_probe_collapses_plain_and_symlinked_special_kinds_to_other() {
    use std::os::unix::net::UnixListener;

    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    let socket_path = root.join("socket");
    let listener = UnixListener::bind(&socket_path).unwrap();
    symlink_file("socket", root.join("socket-link"));
    let fs = LocalFileSystem::new(&root);

    assert_eq!(
        fs.probe_dir_entry("socket", None).await.unwrap().kind,
        DirEntryProbeKind::Other
    );
    assert_eq!(
        fs.probe_dir_entry("socket-link", None).await.unwrap().kind,
        DirEntryProbeKind::Other
    );

    drop(listener);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn broken_symlink_is_a_per_probe_error_and_a_caller_can_continue() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("e1"), "one").unwrap();
    symlink_file("missing-target", root.join("e2-broken"));
    std::fs::write(root.join("e3"), "three").unwrap();
    let fs = LocalFileSystem::new(&root);

    assert_eq!(
        fs.probe_dir_entry("e2-broken", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::NotFound
    );
    let mut names = fs.list_dir_raw(".", None).await.unwrap();
    names.sort();
    let mut successes = Vec::new();
    for name in names {
        if let Ok(probe) = fs.probe_dir_entry(&name, None).await {
            successes.push(probe.name);
        }
    }
    assert_eq!(successes, ["e1", "e3"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn probe_identity_uses_the_resolved_addressed_path_not_raw_input_or_target() {
    let root = temp_root();
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("sub/item"), "content").unwrap();
    let fs = LocalFileSystem::new(&root);

    let probe = fs.probe_dir_entry("sub/item", None).await.unwrap();
    let info = fs.file_info("sub/item", None).await.unwrap();
    assert_eq!(probe.name, "item");
    assert_eq!(PathBuf::from(&probe.path), root.join("sub").join("item"));
    assert_ne!(probe.path, "sub/item");
    assert_eq!(probe.path, info.path);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn extension_cancellation_rules_are_deliberately_asymmetric() {
    let root = temp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("existing"), "content").unwrap();
    let fs = LocalFileSystem::new(&root);
    let controller = CancellationController::default();
    let signal = controller.signal();
    controller.abort();

    assert_eq!(
        fs.list_dir_raw("missing", Some(&signal))
            .await
            .unwrap_err()
            .code,
        FsErrorCode::Aborted
    );
    assert_eq!(
        fs.probe_dir_entry("existing", Some(&signal))
            .await
            .unwrap()
            .kind,
        DirEntryProbeKind::File
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn providers_without_the_extension_report_not_supported() {
    let provider = UnsupportedExtensionProvider {
        cwd: PathBuf::from("."),
        world: ExecutionWorldIdentity::local(),
    };

    assert_eq!(
        provider.list_dir_raw(".", None).await.unwrap_err().code,
        FsErrorCode::NotSupported
    );
    assert_eq!(
        provider
            .probe_dir_entry("entry", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::NotSupported
    );
}

#[test]
fn entry_probe_kind_serializes_with_the_exact_shared_vocabulary() {
    let cases = [
        (DirEntryProbeKind::File, "file"),
        (DirEntryProbeKind::Directory, "directory"),
        (DirEntryProbeKind::SymlinkToFile, "symlink_to_file"),
        (
            DirEntryProbeKind::SymlinkToDirectory,
            "symlink_to_directory",
        ),
        (DirEntryProbeKind::Other, "other"),
    ];

    for (kind, expected) in cases {
        assert_eq!(serde_json::to_value(kind).unwrap(), expected);
    }
}

#[tokio::test]
async fn existing_listing_and_metadata_behavior_remains_unchanged() {
    let root = temp_root();
    std::fs::create_dir_all(root.join("directory")).unwrap();
    std::fs::write(root.join("file"), "content").unwrap();
    symlink_file("file", root.join("link"));
    let fs = LocalFileSystem::new(&root);

    let _ = fs.list_dir_raw(".", None).await.unwrap();
    let _ = fs.probe_dir_entry("link", None).await.unwrap();
    assert_eq!(
        fs.file_info("file", None).await.unwrap().kind,
        FileKind::File
    );
    assert_eq!(
        fs.file_info("directory", None).await.unwrap().kind,
        FileKind::Directory
    );
    assert_eq!(
        fs.file_info("link", None).await.unwrap().kind,
        FileKind::Symlink
    );
    let entries = fs.list_dir(".", None).await.unwrap();
    assert!(
        entries
            .iter()
            .any(|entry| entry.name == "link" && entry.kind == FileKind::Symlink)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert_type<T: Send + Sync>() {}
    assert_type::<LocalFileSystem>();
    assert_type::<Arc<LocalFileSystem>>();
}
