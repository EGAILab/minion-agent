//! WP12E1-OBS-001: exercise the real classification branch on every host.
//!
//! Windows std::fs::FileType partitions non-symlinks into file/directory, so a
//! Unix socket fixture cannot be replaced by a Windows disk fixture. The existing
//! private operations seam supplies a real FileType with both predicates false
//! at the followed-target boundary. No public API or production classifier changes.
use super::*;

#[derive(Debug)]
struct UnclassifiedTarget {
    entry: PathBuf,
    metadata: std::fs::Metadata,
    calls: std::sync::Mutex<Vec<&'static str>>,
}

#[async_trait]
impl DirectoryProbeOperations for UnclassifiedTarget {
    async fn read_dir_names(&self, _: &Path) -> io::Result<Vec<String>> {
        panic!("a single-entry probe must not enumerate")
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        assert_eq!(path, self.entry);
        self.calls.lock().unwrap().push("lstat");
        Ok(self.metadata.clone())
    }

    async fn metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        assert_eq!(path, self.entry);
        self.calls.lock().unwrap().push("stat");
        // Symlink metadata is an available, safe carrier with !is_file &&
        // !is_dir on Windows too. This is intentionally a synthetic stat result,
        // not a claim that native stat of this link returns symlink metadata.
        Ok(self.metadata.clone())
    }
}

#[tokio::test]
async fn entry_probe_unclassified_target_falls_through_to_other_on_every_host() {
    let root = tempfile::tempdir().unwrap();
    let entry = root.path().join("addressed-link");
    std::fs::write(root.path().join("target"), b"content").unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file("target", &entry).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("target", &entry).unwrap();
    let metadata = std::fs::symlink_metadata(&entry).unwrap();
    assert!(metadata.file_type().is_symlink());
    assert!(!metadata.is_file());
    assert!(!metadata.is_dir());

    let operations = Arc::new(UnclassifiedTarget {
        entry: entry.clone(),
        metadata,
        calls: std::sync::Mutex::new(Vec::new()),
    });
    let mut fs = LocalFileSystem::new(root.path());
    fs.directory_probe_operations = operations.clone();
    let probe = fs.probe_dir_entry("addressed-link", None).await.unwrap();
    assert_eq!(
        probe.kind,
        DirEntryProbeKind::Other,
        "a followed target that is neither file nor directory must be Other"
    );
    assert_eq!(probe.name, "addressed-link");
    assert_eq!(probe.path, from_native(&entry));
    assert_eq!(*operations.calls.lock().unwrap(), ["lstat", "stat"]);
}
