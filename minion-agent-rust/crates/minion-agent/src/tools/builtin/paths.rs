//! Shared `read`/`ls` path argument pipeline and R010-B cause phrases.

use crate::{
    execution::{FsErrorCode, file_url_to_path},
    tools::ToolCapabilityError,
};

pub(super) const OPERATION_ABORTED: &str = "Operation aborted";

pub(super) fn cause(code: FsErrorCode) -> &'static str {
    match code {
        FsErrorCode::NotFound => "no such file or directory",
        FsErrorCode::PermissionDenied => "permission denied",
        FsErrorCode::NotDirectory => "not a directory",
        FsErrorCode::IsDirectory => "is a directory",
        FsErrorCode::Invalid => "invalid path",
        FsErrorCode::NotSupported => "not supported by this provider",
        FsErrorCode::Unknown => "unknown filesystem error",
        // The pinned Pi template handles cancellation separately, not through the cause table.
        FsErrorCode::Aborted => OPERATION_ABORTED,
    }
}

fn normalized_space(ch: char) -> char {
    match ch {
        '\u{00a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
        other => other,
    }
}

fn windows_shell_path(path: &str) -> Option<String> {
    if !path.starts_with('/') || path.starts_with("//") || path.contains('\\') {
        return None;
    }
    let rest = path
        .strip_prefix("/mnt/")
        .or_else(|| path.strip_prefix("/cygdrive/"))
        .unwrap_or_else(|| path.strip_prefix('/').expect("starts with slash"));
    let bytes = rest.as_bytes();
    let drive = *bytes.first()?;
    if !drive.is_ascii_alphabetic() || (bytes.len() > 1 && bytes[1] != b'/') {
        return None;
    }
    let suffix = if bytes.len() > 1 { &rest[2..] } else { "" };
    if suffix
        .chars()
        .any(|ch| matches!(ch, '\n' | '\r' | '\u{2028}' | '\u{2029}'))
    {
        return None;
    }
    Some(format!(
        "{}:\\{}",
        (drive as char).to_ascii_uppercase(),
        suffix.replace('/', "\\")
    ))
}

/// TOOL-026 steps 1-4. A malformed `file://` fails here, before any `ctx.fs` call.
pub(super) fn preprocess_path(path: &str) -> Result<String, ToolCapabilityError> {
    let mut working: String = path.chars().map(normalized_space).collect();
    if let Some(stripped) = working.strip_prefix('@') {
        working = stripped.to_owned();
    }
    if cfg!(windows)
        && let Some(native) = windows_shell_path(&working)
    {
        working = native;
    }
    if working.starts_with("file://") {
        return file_url_to_path(&working, cfg!(windows)).ok_or_else(|| {
            ToolCapabilityError::new(format!(
                "Cannot access {working}: {}",
                cause(FsErrorCode::Invalid)
            ))
        });
    }
    Ok(working)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_space_and_one_at_only() {
        assert_eq!(preprocess_path("@ a\u{00a0}b ").unwrap(), " a b ");
        assert_eq!(preprocess_path("@@file").unwrap(), "@file");
    }

    #[test]
    fn malformed_file_url_rejected_before_any_provider() {
        let error = preprocess_path("file:///%ZZ").unwrap_err();
        assert_eq!(error.message(), "Cannot access file:///%ZZ: invalid path");
    }

    #[test]
    fn windows_drive_shape_is_ascii_only_and_preserves_shell_boundaries() {
        assert_eq!(
            windows_shell_path("/mnt/c/a/b").as_deref(),
            Some("C:\\a\\b")
        );
        assert_eq!(windows_shell_path("/c").as_deref(), Some("C:\\"));
        assert!(windows_shell_path("/\u{212a}/x").is_none());
        assert!(windows_shell_path("/c/x\n").is_none());
    }
}
