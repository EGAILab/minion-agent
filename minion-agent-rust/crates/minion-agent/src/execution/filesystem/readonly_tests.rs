use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ScriptedDirectory {
    entry: PathBuf,
    attempts: AtomicUsize,
    corrections: AtomicUsize,
    correction: &'static str,
    retry: &'static str,
}

#[async_trait]
impl RemoveDirectoryOperations for ScriptedDirectory {
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        if path != self.entry {
            return tokio::fs::remove_dir(path).await;
        }
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        if attempt == 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "first delete",
            ));
        }
        match self.retry {
            "distinct" => Err(io::Error::new(io::ErrorKind::NotADirectory, "retry delete")),
            "denied" => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "retry delete",
            )),
            "vanished" => {
                std::fs::remove_dir(path).unwrap();
                Err(io::Error::from(io::ErrorKind::NotFound))
            }
            _ => tokio::fs::remove_dir(path).await,
        }
    }

    async fn clear_readonly_entry(&self, path: &Path) -> io::Result<bool> {
        assert_eq!(path, self.entry);
        self.corrections.fetch_add(1, Ordering::SeqCst);
        match self.correction {
            "denied" => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "correction",
            )),
            "absent" => Ok(false),
            "vanished" => {
                std::fs::remove_dir(path).unwrap();
                Err(io::Error::from(io::ErrorKind::NotFound))
            }
            _ => {
                #[cfg(windows)]
                assert!(minion_agent_native_fs::clear_readonly_entry(path)?);
                Ok(true)
            }
        }
    }
}

fn fixture(
    correction: &'static str,
    retry: &'static str,
) -> (tempfile::TempDir, ScriptedDirectory) {
    let root = tempfile::tempdir().unwrap();
    let entry = root.path().join("tree").join("entry");
    std::fs::create_dir_all(&entry).unwrap();
    #[cfg(windows)]
    {
        let mut p = std::fs::metadata(&entry).unwrap().permissions();
        p.set_readonly(true);
        std::fs::set_permissions(&entry, p).unwrap();
    }
    (
        root,
        ScriptedDirectory {
            entry,
            attempts: AtomicUsize::new(0),
            corrections: AtomicUsize::new(0),
            correction,
            retry,
        },
    )
}

#[tokio::test]
async fn readonly_directory_retry_preserves_the_distinct_retry_error_and_origin() {
    let (root, operations) = fixture("clear", "distinct");
    let error = remove_addressed_with(&root.path().join("tree"), true, false, &operations)
        .await
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::NotDirectory);
    assert_eq!(error.path, Some(from_native(&operations.entry)));
    assert!(error.message.contains("retry delete"));
    assert_eq!(operations.attempts.load(Ordering::SeqCst), 2);
    assert!(operations.entry.exists());
    #[cfg(windows)]
    assert!(
        !std::fs::metadata(&operations.entry)
            .unwrap()
            .permissions()
            .readonly()
    );
}

#[tokio::test]
async fn readonly_directory_correction_failure_preserves_original_without_retry() {
    for correction in ["denied", "absent"] {
        let (root, operations) = fixture(correction, "distinct");
        let error = remove_addressed_with(&root.path().join("tree"), true, false, &operations)
            .await
            .unwrap_err();
        assert_eq!(error.code, FsErrorCode::PermissionDenied);
        assert!(error.message.contains("first delete"));
        assert_eq!(error.path, Some(from_native(&operations.entry)));
        assert_eq!(operations.attempts.load(Ordering::SeqCst), 1);
        #[cfg(windows)]
        minion_agent_native_fs::clear_readonly_entry(&operations.entry).unwrap();
    }
}

#[tokio::test]
async fn readonly_directory_retries_only_once() {
    let (root, operations) = fixture("clear", "denied");
    let error = remove_addressed_with(&root.path().join("tree"), true, false, &operations)
        .await
        .unwrap_err();
    assert!(error.message.contains("retry delete"));
    assert_eq!(operations.attempts.load(Ordering::SeqCst), 2);
    assert_eq!(operations.corrections.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn readonly_directory_concurrent_vanish_during_correction_or_retry_is_success() {
    for (correction, retry) in [("vanished", "ok"), ("clear", "vanished"), ("clear", "ok")] {
        let (root, operations) = fixture(correction, retry);
        #[cfg(windows)]
        if correction == "vanished" {
            minion_agent_native_fs::clear_readonly_entry(&operations.entry).unwrap();
        }
        remove_addressed_with(&root.path().join("tree"), true, false, &operations)
            .await
            .unwrap();
        assert!(!root.path().join("tree").exists());
        assert_eq!(
            operations.attempts.load(Ordering::SeqCst),
            if correction == "vanished" { 1 } else { 2 }
        );
    }
}

#[cfg(windows)]
#[tokio::test]
async fn readonly_directory_real_remove_handles_target_and_tree() {
    for target in ["tree", "tree/entry"] {
        let (root, operations) = fixture("unused", "unused");
        LocalFileSystem::new(root.path())
            .remove(target, true, false, None)
            .await
            .unwrap();
        assert!(!operations.entry.exists());
    }
}

#[cfg(windows)]
#[test]
fn readonly_correction_addresses_the_link_not_its_external_target() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("external");
    let link = root.path().join("link");
    std::fs::create_dir(&target).unwrap();
    std::os::windows::fs::symlink_dir(&target, &link).unwrap();
    let mut p = std::fs::metadata(&target).unwrap().permissions();
    p.set_readonly(true);
    std::fs::set_permissions(&target, p).unwrap();
    assert!(!minion_agent_native_fs::clear_readonly_entry(&link).unwrap());
    assert!(std::fs::metadata(&target).unwrap().permissions().readonly());
    std::fs::remove_dir(&link).unwrap();
    minion_agent_native_fs::clear_readonly_entry(&target).unwrap();
}

#[cfg(windows)]
#[tokio::test]
async fn readonly_directory_correction_keeps_the_existing_long_path_boundary() {
    let root = tempfile::tempdir().unwrap();
    let mut target = root.path().to_owned();
    for _ in 0..10 {
        target.push("directory-segment-123456789");
    }
    assert!(target.as_os_str().len() > 260);
    std::fs::create_dir_all(&target).unwrap();
    let mut permissions = std::fs::metadata(&target).unwrap().permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&target, permissions).unwrap();
    LocalFileSystem::new(root.path())
        .remove(from_native(&target), true, false, None)
        .await
        .unwrap();
    assert!(!target.exists());
}

#[cfg(windows)]
#[tokio::test]
async fn readonly_directory_nonrecursive_refusal_does_not_clear_the_attribute() {
    let (root, operations) = fixture("unused", "unused");
    let error = LocalFileSystem::new(root.path())
        .remove("tree/entry", false, false, None)
        .await
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::PermissionDenied);
    assert!(
        std::fs::metadata(&operations.entry)
            .unwrap()
            .permissions()
            .readonly()
    );
    minion_agent_native_fs::clear_readonly_entry(&operations.entry).unwrap();
}
