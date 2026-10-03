//! Shared `read`/`ls` path argument pipeline and R010-B cause phrases.

use crate::{
    execution::{FsErrorCode, FsPath, file_url_to_js_path},
    tools::{PreparedValue, ToolCapabilityError},
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

#[cfg(test)]
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
pub(super) fn preprocess_path(path: impl Into<FsPath>) -> Result<FsPath, ToolCapabilityError> {
    let path = path.into();
    let mut units: Vec<u16> = path
        .code_units()
        .iter()
        .map(|unit| {
            char::from_u32(u32::from(*unit)).map_or(*unit, |ch| normalized_space(ch) as u16)
        })
        .collect();
    if units.first() == Some(&64) {
        units.remove(0);
    }
    if cfg!(windows)
        && units.first() == Some(&47)
        && !units.starts_with(&[47, 47])
        && !units.contains(&92)
    {
        let prefix = if units.starts_with(&"/mnt/".encode_utf16().collect::<Vec<_>>()) {
            5
        } else if units.starts_with(&"/cygdrive/".encode_utf16().collect::<Vec<_>>()) {
            10
        } else {
            1
        };
        let rest = &units[prefix..];
        if let Some(drive @ (65..=90 | 97..=122)) = rest.first().copied()
            && (rest.len() == 1 || rest[1] == 47)
            && !rest.iter().any(|u| matches!(u, 10 | 13 | 0x2028 | 0x2029))
        {
            let mut native = vec![if drive >= 97 { drive - 32 } else { drive }, 58, 92];
            native.extend(
                rest.get(2..)
                    .unwrap_or_default()
                    .iter()
                    .map(|u| if *u == 47 { 92 } else { *u }),
            );
            units = native;
        }
    }
    let working = FsPath::from_code_units(units);
    if working
        .code_units()
        .starts_with(&"file://".encode_utf16().collect::<Vec<_>>())
    {
        return file_url_to_js_path(&working, cfg!(windows)).ok_or_else(|| {
            ToolCapabilityError::new(path_message("Cannot access {path}: invalid path", &working))
        });
    }
    Ok(working)
}

pub(super) fn argument_path(value: PreparedValue) -> Option<FsPath> {
    match value {
        PreparedValue::String(path) => Some(FsPath::from_code_units(path.code_units().to_vec())),
        _ => None,
    }
}

/// Interpolate a logical path without going through Unicode Display/JSON.
pub(super) fn path_message(template: &str, path: &FsPath) -> FsPath {
    let mut units = Vec::new();
    for (i, part) in template.split("{path}").enumerate() {
        if i > 0 {
            units.extend_from_slice(path.code_units());
        }
        units.extend(part.encode_utf16());
    }
    FsPath::from_code_units(units)
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
