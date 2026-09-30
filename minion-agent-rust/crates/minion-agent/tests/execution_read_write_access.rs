//! EXEC-009 witnesses: one combined read+write access decision, not two layered probes.

use std::path::Path;

use minion_agent::execution::{CancellationController, FileSystem, FsErrorCode, LocalFileSystem};

fn fixture() -> (tempfile::TempDir, LocalFileSystem) {
    let root = tempfile::tempdir().unwrap();
    let fs = LocalFileSystem::new(root.path());
    (root, fs)
}

#[cfg(unix)]
fn symlink_file(target: &str, link: &Path) {
    std::os::unix::fs::symlink(target, link).unwrap();
}

#[cfg(windows)]
fn symlink_file(target: &str, link: &Path) {
    std::os::windows::fs::symlink_file(target, link).unwrap();
}

#[tokio::test]
async fn accessible_file_and_directory_are_nonmutating_access_answers() {
    let (root, fs) = fixture();
    std::fs::create_dir(root.path().join("sub")).unwrap();
    let file = root.path().join("sub/f");
    std::fs::write(&file, b"unchanged").unwrap();
    let before = std::fs::metadata(&file).unwrap();
    symlink_file("f", &root.path().join("sub/link"));

    assert_eq!(fs.check_read_write("sub/f", None).await, Ok(()));
    assert_eq!(fs.check_read_write("sub/link", None).await, Ok(()));
    assert_eq!(fs.check_read_write("sub", None).await, Ok(()));
    assert_eq!(std::fs::read(&file).unwrap(), b"unchanged");
    let after = std::fs::metadata(&file).unwrap();
    assert_eq!(after.len(), before.len());
    assert_eq!(after.modified().unwrap(), before.modified().unwrap());
    // An access answer does not consume content or alter the existing metadata operation.
    assert_eq!(fs.check_readable("sub/f", None).await, Ok(()));
    assert_eq!(fs.file_info("sub/f", None).await.unwrap().size, 9);
}

#[tokio::test]
async fn missing_dangling_non_directory_and_nul_have_host_classes() {
    let (root, fs) = fixture();
    std::fs::write(root.path().join("file"), b"x").unwrap();
    symlink_file("absent", &root.path().join("dangling"));
    for path in ["missing", "dangling"] {
        assert_eq!(
            fs.check_read_write(path, None).await.unwrap_err().code,
            FsErrorCode::NotFound,
            "{path}"
        );
    }
    assert_eq!(
        fs.check_read_write("file/child", None)
            .await
            .unwrap_err()
            .code,
        if cfg!(windows) {
            FsErrorCode::NotFound
        } else {
            FsErrorCode::NotDirectory
        }
    );
    assert_eq!(
        fs.check_read_write("file\0child", None)
            .await
            .unwrap_err()
            .code,
        FsErrorCode::Unknown
    );
}

#[tokio::test]
async fn signal_is_accepted_but_not_inspected() {
    let (root, fs) = fixture();
    std::fs::write(root.path().join("file"), b"x").unwrap();
    let controller = CancellationController::default();
    controller.abort();
    assert_eq!(
        fs.check_read_write("file", Some(&controller.signal()))
            .await,
        Ok(())
    );
}

#[cfg(unix)]
mod posix {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct RestoreMode {
        path: std::path::PathBuf,
        original: u32,
    }

    impl RestoreMode {
        fn new(path: &Path) -> Self {
            Self {
                path: path.to_owned(),
                original: std::fs::metadata(path).unwrap().permissions().mode(),
            }
        }
    }

    impl Drop for RestoreMode {
        fn drop(&mut self) {
            std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(self.original))
                .unwrap();
        }
    }

    fn mode(path: &Path, value: u32) {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(value)).unwrap();
    }

    #[tokio::test]
    async fn read_and_write_are_both_required_for_files_and_directories() {
        if nix::unistd::geteuid().is_root() {
            eprintln!("SKIP: POSIX permission witness requires non-root user");
            return;
        }
        let (root, fs) = fixture();
        let file = root.path().join("file");
        let dir = root.path().join("dir");
        std::fs::write(&file, b"x").unwrap();
        std::fs::create_dir(&dir).unwrap();
        symlink_file("file", &root.path().join("link"));
        let _file_restore = RestoreMode::new(&file);
        let _dir_restore = RestoreMode::new(&dir);

        mode(&file, 0o444);
        assert_eq!(fs.check_readable("file", None).await, Ok(()));
        for path in ["file", "link"] {
            assert_eq!(
                fs.check_read_write(path, None).await.unwrap_err().code,
                FsErrorCode::PermissionDenied,
                "{path}"
            );
        }
        mode(&file, 0o222);
        assert_eq!(
            fs.check_read_write("file", None).await.unwrap_err().code,
            FsErrorCode::PermissionDenied
        );

        mode(&dir, 0o666);
        assert_eq!(fs.check_read_write("dir", None).await, Ok(()));
        assert_eq!(
            std::fs::write(dir.join("child"), b"x").unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        mode(&dir, 0o555);
        assert_eq!(fs.check_readable("dir", None).await, Ok(()));
        assert_eq!(
            fs.check_read_write("dir", None).await.unwrap_err().code,
            FsErrorCode::PermissionDenied
        );
        mode(&dir, 0o444);
        assert_eq!(
            fs.check_read_write("dir/child", None)
                .await
                .unwrap_err()
                .code,
            FsErrorCode::PermissionDenied
        );
    }

    #[tokio::test]
    async fn fifo_does_not_block_and_symlink_loop_maps_unknown() {
        use nix::{sys::stat::Mode, unistd::mkfifo};

        let (root, fs) = fixture();
        mkfifo(&root.path().join("pipe"), Mode::S_IRUSR | Mode::S_IWUSR).unwrap();
        assert_eq!(
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                fs.check_read_write("pipe", None)
            )
            .await
            .expect("access must not open FIFO content"),
            Ok(())
        );
        symlink_file("b", &root.path().join("a"));
        symlink_file("a", &root.path().join("b"));
        assert_eq!(
            fs.check_read_write("a", None).await.unwrap_err().code,
            FsErrorCode::Unknown
        );
    }
}

#[cfg(windows)]
mod windows {
    use super::*;
    use std::os::windows::fs::OpenOptionsExt;

    struct ResetAcl(std::path::PathBuf);

    impl ResetAcl {
        fn deny(path: &Path, rights: &str) -> Self {
            let user = std::env::var("USERNAME").unwrap();
            let output = std::process::Command::new("icacls")
                .arg(path)
                .arg("/deny")
                .arg(format!("{user}:({rights})"))
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            Self(path.to_owned())
        }
    }

    impl Drop for ResetAcl {
        fn drop(&mut self) {
            let output = std::process::Command::new("icacls")
                .arg(&self.0)
                .arg("/reset")
                .output()
                .unwrap();
            assert!(output.status.success());
        }
    }

    #[tokio::test]
    async fn deny_write_and_deny_read_acls_both_fail_combined_access() {
        let (root, fs) = fixture();
        let file = root.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        symlink_file("file", &root.path().join("alias"));
        {
            let _deny = ResetAcl::deny(&file, "WD");
            assert_eq!(fs.check_readable("file", None).await, Ok(()));
            for path in ["file", "alias"] {
                assert_eq!(
                    fs.check_read_write(path, None).await.unwrap_err().code,
                    FsErrorCode::PermissionDenied
                );
            }
        }
        {
            let _deny = ResetAcl::deny(&file, "RD");
            assert_eq!(
                fs.check_read_write("file", None).await.unwrap_err().code,
                FsErrorCode::PermissionDenied
            );
        }
    }

    #[tokio::test]
    async fn directory_probe_requires_add_file_but_not_delete_child() {
        let (root, fs) = fixture();
        let denied_add = root.path().join("denied_add");
        let denied_delete = root.path().join("denied_delete");
        std::fs::create_dir(&denied_add).unwrap();
        std::fs::create_dir(&denied_delete).unwrap();
        {
            let _deny = ResetAcl::deny(&denied_add, "WD");
            assert_eq!(fs.check_readable("denied_add", None).await, Ok(()));
            assert_eq!(
                fs.check_read_write("denied_add", None)
                    .await
                    .unwrap_err()
                    .code,
                FsErrorCode::PermissionDenied
            );
        }
        {
            let _deny = ResetAcl::deny(&denied_delete, "DC");
            assert_eq!(fs.check_read_write("denied_delete", None).await, Ok(()));
        }
    }

    #[tokio::test]
    async fn read_only_file_denied_but_read_only_directory_is_allowed() {
        let (root, fs) = fixture();
        let file = root.path().join("file");
        let dir = root.path().join("dir");
        std::fs::write(&file, b"x").unwrap();
        std::fs::create_dir(&dir).unwrap();
        let original = [&file, &dir].map(|path| std::fs::metadata(path).unwrap().permissions());
        for path in [&file, &dir] {
            let mut perms = std::fs::metadata(path).unwrap().permissions();
            perms.set_readonly(true);
            std::fs::set_permissions(path, perms).unwrap();
        }
        assert_eq!(
            fs.check_read_write("file", None).await.unwrap_err().code,
            FsErrorCode::PermissionDenied
        );
        assert_eq!(fs.check_read_write("dir", None).await, Ok(()));
        for (path, permissions) in [&file, &dir].into_iter().zip(original) {
            std::fs::set_permissions(path, permissions).unwrap();
        }
    }

    #[tokio::test]
    async fn sharing_violation_uses_the_existing_filesystem_error_mapper() {
        let (root, fs) = fixture();
        let file = root.path().join("held.txt");
        std::fs::write(&file, b"x").unwrap();
        let held = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .share_mode(0)
            .open(&file)
            .unwrap();

        let combined = fs.check_read_write("held.txt", None).await.unwrap_err();
        let readable = fs.check_readable("held.txt", None).await.unwrap_err();
        let read = fs.read_binary_file("held.txt", None).await.unwrap_err();
        assert_eq!(combined.code, readable.code);
        assert_eq!(combined.code, read.code);

        drop(held);
        assert_eq!(fs.check_read_write("held.txt", None).await, Ok(()));
        assert_eq!(fs.check_readable("held.txt", None).await, Ok(()));
        assert_eq!(
            fs.read_binary_file("held.txt", None).await,
            Ok(b"x".to_vec())
        );
    }
}
