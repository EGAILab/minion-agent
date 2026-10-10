use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct RecursiveVanish {
    root: PathBuf,
    stage: &'static str,
    removes: AtomicUsize,
    fired: AtomicUsize,
}

#[async_trait]
impl RemoveDirectoryOperations for RecursiveVanish {
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        if path == self.root {
            let attempt = self.removes.fetch_add(1, Ordering::SeqCst);
            if self.stage == "initial" && attempt == 0 {
                std::fs::remove_dir_all(path)?;
                self.fired.fetch_add(1, Ordering::SeqCst);
                return Err(io::Error::from(io::ErrorKind::NotFound));
            }
            if self.stage == "final" && attempt == 1 {
                std::fs::remove_dir(path)?;
                self.fired.fetch_add(1, Ordering::SeqCst);
                return Err(io::Error::from(io::ErrorKind::NotFound));
            }
        }
        tokio::fs::remove_dir(path).await
    }

    async fn clear_readonly_entry(&self, _path: &Path) -> io::Result<bool> {
        panic!("disappearance must not invoke permission correction")
    }

    async fn read_dir(&self, path: &Path) -> io::Result<tokio::fs::ReadDir> {
        if self.stage == "readdir" && path == self.root {
            std::fs::remove_dir_all(path)?;
            self.fired.fetch_add(1, Ordering::SeqCst);
        }
        tokio::fs::read_dir(path).await
    }

    async fn before_child_metadata(&self, path: &Path) {
        if self.stage == "child" {
            // This is reached only after the real read_dir/next_entry returned
            // the child; its next call is the production symlink_metadata.
            std::fs::remove_file(path).unwrap();
            self.fired.fetch_add(1, Ordering::SeqCst);
        }
    }
}

async fn recursive_vanish_at(stage: &'static str) {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("tree");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("child"), "entry").unwrap();
    let operations = RecursiveVanish {
        root: root.clone(),
        stage,
        removes: AtomicUsize::new(0),
        fired: AtomicUsize::new(0),
    };
    remove_addressed_with(&root, true, false, &operations)
        .await
        .unwrap();
    assert_eq!(operations.fired.load(Ordering::SeqCst), 1);
    assert!(!root.exists());
}

#[tokio::test]
async fn readonly_recursive_child_vanishing_before_lstat_counts_as_removed() {
    recursive_vanish_at("child").await;
}

#[tokio::test]
async fn readonly_recursive_directory_vanishing_before_readdir_counts_as_removed() {
    recursive_vanish_at("readdir").await;
}

#[tokio::test]
async fn readonly_recursive_directory_vanishing_at_first_remove_counts_as_removed() {
    recursive_vanish_at("initial").await;
}

#[tokio::test]
async fn readonly_recursive_directory_vanishing_at_final_remove_counts_as_removed() {
    recursive_vanish_at("final").await;
}

#[tokio::test]
async fn readonly_recursive_missing_target_still_requires_force() {
    let root = tempfile::tempdir().unwrap();
    let filesystem = LocalFileSystem::new(root.path());
    for recursive in [false, true] {
        let error = filesystem
            .remove("missing", recursive, false, None)
            .await
            .unwrap_err();
        assert_eq!(error.code, FsErrorCode::NotFound);
        filesystem
            .remove("missing", recursive, true, None)
            .await
            .unwrap();
    }
}

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
