//! EXEC-005: a rooted cwd reaches a real child, not only the lexical port.
use minion_agent::execution::{LocalSubprocess, SpawnOptions, Subprocess};
use std::path::PathBuf;
#[path = "support/fs_containment.rs"]
mod containment;

#[test]
#[ignore = "child observer, explicitly launched by rooted_spawn_uses_process_device"]
fn rooted_cwd_child() {
    let cwd = std::env::current_dir().unwrap();
    println!(
        "ROOTED_CHILD_CWD {}",
        serde_json::to_string(&cwd.to_string_lossy()).unwrap()
    );
}

#[tokio::test]
async fn rooted_spawn_uses_process_device() {
    let root = containment::sandbox("d013-rooted-spawn");
    let cwd = root.join("child");
    containment::check(&root, &cwd);
    std::fs::create_dir(&cwd).unwrap();
    containment::check(&root, &cwd);
    let spelling = cwd.to_str().unwrap();
    let requested = if cfg!(windows) {
        assert_eq!(&spelling[..2], "E:");
        PathBuf::from(&spelling[2..])
    } else {
        cwd.clone()
    };
    let provider = LocalSubprocess::new(if cfg!(windows) {
        r"C:\unrelated"
    } else {
        "/unrelated"
    });
    let exe = std::env::current_exe().unwrap();
    let child = provider
        .spawn(
            &[
                exe.to_str().unwrap().into(),
                "--exact".into(),
                "rooted_cwd_child".into(),
                "--ignored".into(),
                "--nocapture".into(),
            ],
            SpawnOptions {
                cwd: Some(requested),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let stream = child.stdout().unwrap();
    let mut output = Vec::new();
    while let Some(chunk) = stream.read_chunk().await.unwrap() {
        output.extend(chunk);
    }
    assert_eq!(child.wait().await.unwrap().exit_code, Some(0));
    let output = String::from_utf8(output).unwrap();
    let line = output
        .lines()
        .find_map(|s| s.strip_prefix("ROOTED_CHILD_CWD "))
        .expect("child cwd record");
    let observed: String = serde_json::from_str(line).unwrap();
    let expected = if cfg!(windows) {
        spelling.replace('/', "\\")
    } else {
        spelling.to_owned()
    };
    assert_eq!(observed, expected);
    // Retain the contained empty directories; never clean through computed paths.
}
