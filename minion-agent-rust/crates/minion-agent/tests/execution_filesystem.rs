use std::{path::PathBuf, sync::Arc};

use minion_agent::execution::{
    CancellationController, FileKind, FileSystem, FsErrorCode, LocalFileSystem,
};
use uuid::Uuid;

fn temp_root() -> PathBuf {
    std::env::temp_dir().join(format!("minion-execution-test-{}", Uuid::new_v4()))
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

#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert_type<T: Send + Sync>() {}
    assert_type::<LocalFileSystem>();
    assert_type::<Arc<LocalFileSystem>>();
}
