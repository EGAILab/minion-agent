use minion_agent::execution::{CancellationController, FsErrorCode, FsPath, LocalFileSystem};
use std::path::PathBuf;
#[cfg(unix)]
use std::sync::Arc;

fn path(units: &[u16]) -> FsPath {
    FsPath::from_code_units(units.to_vec())
}
fn root() -> PathBuf {
    let p = std::env::temp_dir().join(format!("minion-path-unit-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&p).unwrap();
    p
}

#[tokio::test]
async fn native_names_logical_metadata_and_location_keys_are_distinct_carriers() {
    let root = root();
    let fs = LocalFileSystem::new(&root);
    let a = path(&[97, 0xd800]);
    let b = path(&[97, 0xdc00]);
    let projected: FsPath = "a\u{fffd}".into();
    let missing_a = fs.resolve(&a, None).await.unwrap();
    let missing_b = fs.resolve(&b, None).await.unwrap();
    let missing_p = fs.resolve(&projected, None).await.unwrap();
    assert_ne!(missing_a.target_key(), missing_b.target_key());
    assert_ne!(missing_a.target_key(), missing_p.target_key());
    fs.write_file(&a, b"content", None).await.unwrap();
    assert_eq!(fs.read_binary_file(&b, None).await.unwrap(), b"content");
    assert_eq!(
        fs.read_binary_file(&projected, None).await.unwrap(),
        b"content"
    );
    assert!(root.join("a\u{fffd}").is_file());
    assert_eq!(fs.file_info(&a, None).await.unwrap().name, a);
    assert_eq!(fs.file_info(&b, None).await.unwrap().name, b);
    let existing_a = fs.resolve(&a, None).await.unwrap();
    let existing_b = fs.resolve(&b, None).await.unwrap();
    let existing_p = fs.resolve(&projected, None).await.unwrap();
    assert_eq!(existing_a.target_key(), existing_b.target_key());
    assert_eq!(existing_a.target_key(), existing_p.target_key());
    assert_ne!(missing_a.target_key(), existing_a.target_key());
    assert_eq!(fs.list_dir_raw(".", None).await.unwrap(), ["a\u{fffd}"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn listing_joins_native_entry_names_to_the_logical_directory() {
    let root = root();
    let fs = LocalFileSystem::new(&root);
    let directory = path(&[100, 0xd800]);
    let child = path(&[100, 0xd800, 47, 102, 0xdc00]);
    fs.write_file(&child, b"x", None).await.unwrap();
    let list = fs.list_dir(&directory, None).await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "f\u{fffd}");
    assert!(list[0].path.code_units().contains(&0xd800));
    assert!(!list[0].path.code_units().contains(&0xdc00));
    let probe = fs.probe_dir_entry(&child, None).await.unwrap();
    assert_eq!(probe.name, path(&[102, 0xdc00]));
    assert_eq!(probe.path, fs.absolute_path(&child, None).await.unwrap());
    let refusal = fs.remove(&directory, false, false, None).await.unwrap_err();
    assert_eq!(
        refusal.path,
        Some(fs.absolute_path(&directory, None).await.unwrap())
    );
    // No assertion about the separately excluded #125 code/outcome gap.
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn valid_pairs_survive_and_every_lone_unit_projects_independently() {
    let root = root();
    let fs = LocalFileSystem::new(&root);
    for (input, name) in [
        (vec![0xd83d, 0xde00], "\u{1f600}"),
        (vec![0xdc00, 0xd800], "\u{fffd}\u{fffd}"),
        (vec![0xd800, 0xd800], "\u{fffd}\u{fffd}"),
    ] {
        fs.write_file(path(&input), b"pair", None).await.unwrap();
        assert_eq!(std::fs::read(root.join(name)).unwrap(), b"pair");
        assert_eq!(
            fs.file_info(path(&input), None)
                .await
                .unwrap()
                .name
                .code_units(),
            input
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn abort_has_a_logical_path_and_does_not_touch_the_native_file() {
    let root = root();
    let fs = LocalFileSystem::new(&root);
    let p = path(&[0xd800]);
    let controller = CancellationController::default();
    controller.abort();
    let signal = controller.signal();
    for error in [
        fs.read_binary_file(&p, Some(&signal)).await.unwrap_err(),
        fs.write_file(&p, b"x", Some(&signal)).await.unwrap_err(),
        fs.list_dir(&p, Some(&signal)).await.unwrap_err(),
    ] {
        assert_eq!(error.code, FsErrorCode::Aborted);
        assert_eq!(error.path, Some(fs.absolute_path(&p, None).await.unwrap()));
    }
    assert!(!root.join("\u{fffd}").exists());
    let destination = path(&[100, 0xdc00]);
    let rename = fs
        .rename_file(&p, &destination, Some(&signal))
        .await
        .unwrap_err();
    assert_eq!(rename.code, FsErrorCode::Aborted);
    assert_eq!(
        rename.path,
        Some(fs.absolute_path(&destination, None).await.unwrap())
    );
    fs.append_file(&p, b"x", Some(&signal)).await.unwrap();
    assert!(root.join("\u{fffd}").is_file());
    assert!(fs.check_read_write(&p, Some(&signal)).await.is_ok());
    assert!(fs.probe_dir_entry(&p, Some(&signal)).await.is_ok());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn rename_error_always_names_the_native_source_not_the_destination() {
    let root = root();
    let fs = LocalFileSystem::new(&root);
    let source = path(&[115, 0xd800]);
    let dest = path(&[100, 0xdc00, 47, 102]);
    fs.write_file(&source, b"x", None).await.unwrap();
    let error = fs.rename_file(&source, &dest, None).await.unwrap_err();
    assert_eq!(error.code, FsErrorCode::NotFound);
    assert_eq!(
        error.path,
        Some(fs.absolute_path("s\u{fffd}", None).await.unwrap())
    );
    assert!(fs.read_binary_file(&source, None).await.is_ok());
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[tokio::test]
#[ignore = "real POSIX permission witness: run explicitly as a non-root user"]
async fn recursive_remove_single_failure_reports_the_inner_native_call() {
    use std::os::unix::fs::PermissionsExt;
    assert!(
        !nix::unistd::geteuid().is_root(),
        "run this binding witness as a non-root user"
    );
    let root = root();
    let fs = Arc::new(LocalFileSystem::new(&root));
    let tree = path(&[116, 0xd800]);
    let child = path(&[116, 0xd800, 47, 120]);
    fs.write_file(&child, b"x", None).await.unwrap();
    let native_dir = root.join("t\u{fffd}");
    std::fs::set_permissions(&native_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    let error = fs.remove(&tree, true, false, None).await.unwrap_err();
    std::fs::set_permissions(&native_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(error.code, FsErrorCode::PermissionDenied);
    assert_eq!(
        error.path,
        Some(fs.absolute_path("t\u{fffd}/x", None).await.unwrap())
    );
    std::fs::remove_dir_all(root).unwrap();
}
