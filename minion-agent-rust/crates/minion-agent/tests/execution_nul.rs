//! L12-D006 binding witnesses. All operations reach the real filesystem provider.
use minion_agent::{
    execution::{FileSystem, FsErrorCode, FsPath, LocalFileSystem},
    llm::{RawValue, StopReason, ToolCall, ToolResultContentBlock},
    runtime::Runtime,
    tools::{
        ToolExecutionOptions,
        builtin::{
            ReadToolOptions, create_edit_tool, create_ls_tool, create_read_tool, create_write_tool,
        },
        execute_tool_calls,
    },
};
use std::{path::PathBuf, sync::Arc};
#[path = "support/fs_containment.rs"]
mod containment;

fn root() -> PathBuf {
    containment::sandbox("minion-nul")
}

#[tokio::test]
async fn nul_is_unknown_and_the_fallback_is_lossless() {
    let root = root();
    let fs = containment::GuardedFs::new(&root);
    let path = FsPath::from_code_units(vec![102, 0xd800, 0, 120]);
    let error = fs.read_binary_file(&path, None).await.unwrap_err();
    assert_eq!(error.code, FsErrorCode::Unknown, "L12-D006 NUL is unknown");
    assert_eq!(
        error.path,
        Some(fs.absolute_path(&path, None).await.unwrap()),
        "L12-D006 fallback stays logical"
    );
    containment::cleanup(&root);
}

#[tokio::test]
async fn the_resolved_file_url_is_the_nul_argument() {
    let root = root();
    let fs = containment::GuardedFs::new(&root);
    let url = format!("{}f%00x", url::Url::from_directory_path(&root).unwrap());
    assert!(!url.contains('\0'));
    let error = fs.read_binary_file(&url, None).await.unwrap_err();
    assert_eq!(
        error.code,
        FsErrorCode::Unknown,
        "L12-D006 decoded URL is unknown"
    );
    assert_eq!(
        error.path,
        Some(fs.absolute_path("f\0x", None).await.unwrap())
    );
    let rename = fs.rename_file("source", &url, None).await.unwrap_err();
    assert_eq!(rename.code, FsErrorCode::Unknown);
    assert_eq!(
        rename.path,
        Some(fs.absolute_path("source", None).await.unwrap())
    );
    containment::cleanup(&root);
}

#[tokio::test]
async fn a_final_nul_keeps_the_parent_creation_effect() {
    let root = root();
    let fs = containment::GuardedFs::new(&root);
    for (name, append) in [("write", false), ("append", true)] {
        let path = format!("{name}/f\0x");
        let error = if append {
            fs.append_file(&path, b"x", None).await
        } else {
            fs.write_file(&path, b"x", None).await
        }
        .unwrap_err();
        assert_eq!(error.code, FsErrorCode::Unknown);
        assert!(
            root.join(name).is_dir(),
            "L12-D006 parent exists before NUL leaf rejection"
        );
    }
    containment::cleanup(&root);
}

#[tokio::test]
async fn canonical_path_checks_the_whole_argument_before_a_missing_prefix() {
    let root = root();
    let fs = containment::GuardedFs::new(&root);
    let path = "missing/f\0x";
    let error = fs.canonical_path(path, None).await.unwrap_err();
    assert_eq!(
        error.code,
        FsErrorCode::Unknown,
        "L12-D006 whole argument precedes walk"
    );
    assert_eq!(
        error.path,
        Some(fs.absolute_path(path, None).await.unwrap())
    );
    let error = fs.resolve(path, None).await.unwrap_err();
    assert_eq!(error.code, FsErrorCode::Unknown);
    containment::cleanup(&root);
}

#[tokio::test]
async fn temporary_creation_keeps_each_operation_fallback() {
    let temp_base = containment::base();
    assert_eq!(
        std::env::temp_dir(),
        temp_base,
        "explicit contained temp API configuration"
    );
    containment::check(&temp_base, &temp_base.join("p\0x"));
    let fs = LocalFileSystem::new(&temp_base);
    let error = fs.create_temp_dir("p\0x", None).await.unwrap_err();
    assert_eq!(
        error.code,
        FsErrorCode::Unknown,
        "L12-D006 temp dir unknown"
    );
    assert_eq!(error.path, None, "L12-D006 temp dir has no fallback");
    for (prefix, suffix) in [("p\0x", ""), ("p", "s\0x")] {
        containment::check(
            &temp_base,
            &temp_base.join(format!("tmp-probe/{prefix}probe{suffix}")),
        );
        let error = fs.create_temp_file(prefix, suffix, None).await.unwrap_err();
        assert_eq!(error.code, FsErrorCode::Unknown);
        let path = error.path.expect("L12-D006 temp file names would-be file");
        assert!(path.code_units().contains(&0));
        let parent = PathBuf::from(path.as_str().unwrap())
            .parent()
            .unwrap()
            .to_path_buf();
        assert!(
            parent.is_dir(),
            "L12-D006 temp directory created before file rejection"
        );
        containment::cleanup(&parent);
    }
}

#[tokio::test]
async fn the_real_tools_render_the_certified_nul_failure() {
    let root = root();
    let fs = Arc::new(LocalFileSystem::new(&root));
    let runtime = Runtime::new();
    for tool in [
        create_read_tool(fs.clone(), ReadToolOptions::default()),
        create_write_tool(fs.clone()),
        create_edit_tool(fs.clone()),
        create_ls_tool(fs.clone()),
    ] {
        runtime.tools().register_for_scope(None, tool).unwrap();
    }
    let logical = fs.absolute_path("f\0x", None).await.unwrap();
    for (name, arguments, wanted) in [
        (
            "read",
            serde_json::json!({"path":"f\0x"}),
            format!(
                "Cannot access {}: unknown filesystem error",
                logical.as_str().unwrap()
            ),
        ),
        (
            "write",
            serde_json::json!({"path":"new/f\0x","content":"x"}),
            "Cannot resolve new/f\0x: unknown filesystem error".to_owned(),
        ),
        (
            "edit",
            serde_json::json!({"path":"f\0x","edits":[{"oldText":"a","newText":"b"}]}),
            "Cannot resolve f\0x: unknown filesystem error".to_owned(),
        ),
        (
            "ls",
            serde_json::json!({"path":"f\0x"}),
            format!("Path not found: {}", logical.as_str().unwrap()),
        ),
    ] {
        if matches!(name, "write" | "edit") {
            let path: FsPath = arguments["path"].as_str().unwrap().into();
            containment::argument(&fs, &root, &path).await;
        }
        let call = ToolCall::new_raw("nul", name, RawValue::from(arguments));
        let batch = execute_tool_calls(
            &runtime.context(),
            &[call],
            ToolExecutionOptions::new(StopReason::ToolUse, 0.0),
        )
        .await
        .unwrap();
        let message = &batch.messages[0];
        assert!(message.is_error);
        let ToolResultContentBlock::Text(text) = &message.content[0] else {
            panic!("expected text")
        };
        assert_eq!(
            text.text.as_str(),
            Some(wanted.as_str()),
            "L12-D006 real tool boundary {name}"
        );
    }
    assert!(
        !root.join("new").exists(),
        "mutation queue fails before parent creation"
    );
    containment::cleanup(&root);
}
