use super::*;

#[cfg(windows)]
fn fixture(_correction: &'static str, _retry: &'static str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let entry = root.path().join("tree/entry");
    std::fs::create_dir_all(&entry).unwrap();
    let mut p = std::fs::metadata(&entry).unwrap().permissions();
    p.set_readonly(true);
    std::fs::set_permissions(&entry, p).unwrap();
    (root, entry)
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
        assert!(!operations.exists());
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
    assert_eq!(error.code, FsErrorCode::Unknown);
    assert!(
        std::fs::metadata(&operations)
            .unwrap()
            .permissions()
            .readonly()
    );
    minion_agent_native_fs::clear_readonly_entry(&operations).unwrap();
}
