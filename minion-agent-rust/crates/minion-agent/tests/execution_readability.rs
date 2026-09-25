use std::path::PathBuf;

use minion_agent::execution::{CancellationController, FileSystem, FsErrorCode, LocalFileSystem};
use uuid::Uuid;

struct Fixture {
    root: PathBuf,
    fs: LocalFileSystem,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("minion-readable-{}", Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let fs = LocalFileSystem::new(&root);
        Self { root, fs }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[cfg(unix)]
fn symlink(target: &str, path: &std::path::Path) {
    std::os::unix::fs::symlink(target, path).unwrap();
}

#[cfg(unix)]
struct RestoreMode {
    path: PathBuf,
    original: u32,
}

#[cfg(unix)]
impl RestoreMode {
    fn new(path: PathBuf) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let original = std::fs::metadata(&path).unwrap().permissions().mode();
        Self { path, original }
    }
}

#[cfg(unix)]
impl Drop for RestoreMode {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(self.original))
            .unwrap();
    }
}

#[cfg(windows)]
fn symlink(target: &str, path: &std::path::Path) {
    std::os::windows::fs::symlink_file(target, path).unwrap();
}

#[tokio::test]
async fn readable_file_directory_and_relative_path_do_not_consume_content() {
    let fixture = Fixture::new();
    std::fs::create_dir(fixture.root.join("sub")).unwrap();
    std::fs::write(fixture.root.join("sub/f"), b"content").unwrap();
    assert_eq!(fixture.fs.check_readable("sub/f", None).await, Ok(()));
    assert_eq!(fixture.fs.check_readable("sub", None).await, Ok(()));
    assert_eq!(
        std::fs::read(fixture.root.join("sub/f")).unwrap(),
        b"content"
    );
    assert_eq!(
        fixture.fs.file_info("sub/f", None).await.unwrap().kind,
        minion_agent::execution::FileKind::File
    );
}

#[tokio::test]
async fn missing_and_embedded_nul_are_distinct_host_error_classes() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture
            .fs
            .check_readable("missing", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::NotFound
    );
    std::fs::write(fixture.root.join("prefix"), b"x").unwrap();
    assert_eq!(
        fixture
            .fs
            .check_readable("prefix\0missing", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::Unknown
    );
    #[cfg(unix)]
    assert_eq!(
        fixture
            .fs
            .check_readable("prefix/child", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::NotDirectory
    );
    #[cfg(windows)]
    assert_eq!(
        fixture
            .fs
            .check_readable("prefix/child", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::NotFound
    );
}

#[tokio::test]
async fn pre_aborted_signal_is_not_inspected() {
    let fixture = Fixture::new();
    std::fs::write(fixture.root.join("readable"), b"x").unwrap();
    let controller = CancellationController::default();
    controller.abort();
    assert_eq!(
        fixture
            .fs
            .check_readable("readable", Some(&controller.signal()))
            .await,
        Ok(())
    );
}

#[tokio::test]
async fn symlink_is_followed_and_dangling_symlink_is_not_found() {
    let fixture = Fixture::new();
    std::fs::write(fixture.root.join("target"), b"x").unwrap();
    symlink("target", &fixture.root.join("link"));
    symlink("absent", &fixture.root.join("dangling"));
    assert_eq!(fixture.fs.check_readable("link", None).await, Ok(()));
    assert_eq!(
        fixture
            .fs
            .check_readable("dangling", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::NotFound
    );
}

#[cfg(windows)]
struct DenyRead {
    path: PathBuf,
}

#[cfg(windows)]
impl DenyRead {
    fn on(path: PathBuf) -> Self {
        use std::process::Command;
        let user = std::env::var("USERNAME").unwrap();
        let output = Command::new("icacls")
            .arg(&path)
            .arg("/deny")
            .arg(format!("{user}:(RD)"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { path }
    }
}

#[cfg(windows)]
impl Drop for DenyRead {
    fn drop(&mut self) {
        let output = std::process::Command::new("icacls")
            .arg(&self.path)
            .arg("/reset")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(windows)]
#[tokio::test]
async fn windows_deny_read_acl_is_not_mistaken_for_existence() {
    let fixture = Fixture::new();
    std::fs::write(fixture.root.join("denied"), b"x").unwrap();
    symlink("denied", &fixture.root.join("alias"));
    let _deny = DenyRead::on(fixture.root.join("denied"));
    for path in ["denied", "alias"] {
        assert_eq!(
            fixture
                .fs
                .check_readable(path, None)
                .await
                .unwrap_err()
                .code,
            FsErrorCode::PermissionDenied,
            "{path}"
        );
    }
}

#[cfg(windows)]
#[tokio::test]
async fn windows_deny_list_directory_acl_is_permission_denied() {
    let fixture = Fixture::new();
    std::fs::create_dir(fixture.root.join("denied_dir")).unwrap();
    let _deny = DenyRead::on(fixture.root.join("denied_dir"));
    assert_eq!(
        fixture
            .fs
            .check_readable("denied_dir", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::PermissionDenied
    );
}

#[cfg(unix)]
#[tokio::test]
async fn posix_mode_bits_and_parent_search_are_checked_by_access() {
    use std::os::unix::fs::PermissionsExt;
    if nix::unistd::geteuid().is_root() {
        return; // Root bypasses permission-bit checks; run this witness unprivileged.
    }
    let fixture = Fixture::new();
    let file = fixture.root.join("denied");
    let directory = fixture.root.join("directory");
    std::fs::write(&file, b"x").unwrap();
    symlink("denied", &fixture.root.join("denied_alias"));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("child"), b"x").unwrap();
    let _file_restore = RestoreMode::new(file.clone());
    let _directory_restore = RestoreMode::new(directory.clone());
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o000)).unwrap();
    assert_eq!(
        fixture
            .fs
            .check_readable("denied", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::PermissionDenied
    );
    assert_eq!(
        fixture
            .fs
            .check_readable("denied_alias", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::PermissionDenied
    );
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();

    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o444)).unwrap();
    assert_eq!(fixture.fs.check_readable("directory", None).await, Ok(()));
    assert_eq!(
        fixture
            .fs
            .check_readable("directory/child", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::PermissionDenied
    );
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o111)).unwrap();
    assert_eq!(
        fixture
            .fs
            .check_readable("directory", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::PermissionDenied
    );
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o000)).unwrap();
    assert_eq!(
        fixture
            .fs
            .check_readable("directory", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::PermissionDenied
    );
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn fifo_without_writer_does_not_block() {
    use nix::{sys::stat::Mode, unistd::mkfifo};
    let fixture = Fixture::new();
    mkfifo(&fixture.root.join("pipe"), Mode::S_IRUSR | Mode::S_IWUSR).unwrap();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        fixture.fs.check_readable("pipe", None),
    )
    .await
    .expect("readability must not open a FIFO for content");
    assert_eq!(result, Ok(()));
}

#[cfg(unix)]
#[tokio::test]
async fn symlink_loop_preserves_host_error_mapping() {
    let fixture = Fixture::new();
    symlink("b", &fixture.root.join("a"));
    symlink("a", &fixture.root.join("b"));
    assert_eq!(
        fixture.fs.check_readable("a", None).await.unwrap_err().code,
        FsErrorCode::Unknown
    );
}
