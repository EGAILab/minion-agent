//! Pure Node-platform path projections used by search output; no filesystem calls.
use crate::execution::FsPath;
use crate::execution::Platform;

/// Pure path operations keep the logical UTF-16 domain. USV conversion belongs
/// to the subprocess/OS boundary, not to comparison with the engine's paths.
pub(super) fn logical_dirname(path: &FsPath, platform: Platform) -> FsPath {
    let mut units = path.code_units().to_vec();
    if platform == Platform::Windows {
        for u in &mut units {
            if *u == 47 {
                *u = 92;
            }
        }
    }
    let s = sep(platform) as u16;
    let root = if platform == Platform::Windows && units.get(1) == Some(&58) {
        3
    } else if platform == Platform::Windows && units.starts_with(&[92, 92]) {
        let server_end = units[2..].iter().position(|u| *u == 92).map(|i| i + 2);
        server_end
            .and_then(|i| {
                units[i + 1..]
                    .iter()
                    .position(|u| *u == 92)
                    .map(|j| i + 1 + j + 1)
            })
            .unwrap_or(units.len())
    } else {
        1
    };
    let end = units.iter().rposition(|u| *u != s).map_or(0, |i| i + 1);
    if end < root {
        return FsPath::from_code_units(units[..root.min(units.len())].to_vec());
    }
    let boundary = units[..end]
        .iter()
        .rposition(|u| *u == s)
        .map_or(0, |i| if i < root { root } else { i });
    if boundary == 0 {
        return ".".into();
    }
    FsPath::from_code_units(units[..boundary.min(units.len())].to_vec())
}
pub(super) fn logical_join(base: &FsPath, name: &str, platform: Platform) -> FsPath {
    let mut units = base.code_units().to_vec();
    let s = sep(platform) as u16;
    while units.last() == Some(&s) {
        units.pop();
    }
    units.push(s);
    units.extend(name.encode_utf16());
    FsPath::from_code_units(units)
}
fn components(units: &[u16], platform: Platform) -> Vec<Vec<u16>> {
    let mut units = units.to_vec();
    if platform == Platform::Windows {
        for u in &mut units {
            if *u == 47 {
                *u = 92;
            }
        }
    }
    let mut parts: Vec<Vec<u16>> = Vec::new();
    if units.first() == Some(&(sep(platform) as u16)) {
        parts.push(Vec::new());
    }
    for part in units.split(|u| *u == sep(platform) as u16) {
        if part.is_empty() || part == [46] {
            continue;
        }
        if part == [46, 46]
            && parts
                .last()
                .is_some_and(|p| !p.is_empty() && p != &[46, 46] && p.get(1) != Some(&58))
        {
            parts.pop();
        } else {
            parts.push(part.to_vec());
        }
    }
    parts
}
fn same_component(left: &[u16], right: &[u16], platform: Platform) -> bool {
    if platform == Platform::Posix {
        return left == right;
    }
    super::search_node_lower::lower(left) == super::search_node_lower::lower(right)
}
pub(super) fn relative_logical(base: &FsPath, path: &str, platform: Platform) -> String {
    let left = components(base.code_units(), platform);
    let right = components(&path.encode_utf16().collect::<Vec<_>>(), platform);
    let mut common = 0;
    while common < left.len().min(right.len())
        && same_component(&left[common], &right[common], platform)
    {
        common += 1;
    }
    if platform == Platform::Windows
        && (common == 0 || (left.first().is_some_and(Vec::is_empty) && common < 3))
    {
        return normalize(path, platform);
    }
    let mut result = vec!["..".to_owned(); left.len() - common];
    result.extend(
        right[common..]
            .iter()
            .map(|v| String::from_utf16(v).expect("engine paths are scalar")),
    );
    result.join(&sep(platform).to_string())
}
pub(super) fn sep(platform: Platform) -> char {
    if platform == Platform::Windows {
        '\\'
    } else {
        '/'
    }
}
fn normalize(path: &str, platform: Platform) -> String {
    let text = if platform == Platform::Windows {
        path.replace('/', "\\")
    } else {
        path.into()
    };
    let separator = sep(platform);
    let mut parts: Vec<&str> = Vec::new();
    for part in text.split(separator) {
        match part {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|p| *p != "..") {
                    parts.pop();
                } else {
                    parts.push(part);
                }
            }
            _ => parts.push(part),
        }
    }
    let prefix = if platform == Platform::Windows && text.starts_with("\\\\") {
        "\\\\".into()
    } else if text.starts_with(separator) {
        separator.to_string()
    } else {
        String::new()
    };
    format!("{prefix}{}", parts.join(&separator.to_string()))
}
#[cfg(test)]
pub(super) fn relative(base: &str, path: &str, platform: Platform) -> String {
    relative_logical(&base.into(), path, platform)
}
pub(super) fn basename(path: &str, platform: Platform) -> String {
    let path = if platform == Platform::Windows {
        path.replace('/', "\\")
    } else {
        path.into()
    };
    path.trim_end_matches(sep(platform))
        .rsplit(sep(platform))
        .next()
        .unwrap_or("")
        .into()
}
pub(super) fn absolute(path: &str, platform: Platform) -> bool {
    if platform == Platform::Windows {
        path.starts_with(['\\', '/'])
            || (path.as_bytes().get(1) == Some(&b':')
                && path
                    .as_bytes()
                    .get(2)
                    .is_some_and(|b| matches!(b, b'/' | b'\\')))
    } else {
        path.starts_with('/')
    }
}
#[cfg(test)]
pub(super) fn find_path(base: &str, line: &str, platform: Platform) -> String {
    find_path_logical(&base.into(), line, platform)
}
pub(super) fn find_path_logical(base: &FsPath, line: &str, platform: Platform) -> String {
    let trailing =
        line.ends_with(sep(platform)) || (platform == Platform::Windows && line.ends_with('/'));
    let rel = if absolute(line, platform) {
        relative_logical(base, line, platform)
    } else {
        line.into()
    };
    let mut result = rel.replace(sep(platform), "/");
    if trailing && !result.ends_with('/') {
        result.push('/');
    }
    result
}
pub(super) fn grep_path(base: &FsPath, file: &str, platform: Platform, directory: bool) -> String {
    if directory {
        let r = relative_logical(base, file, platform);
        if !r.is_empty() && !r.starts_with("..") {
            return r.replace('\\', "/");
        }
    }
    basename(file, platform)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_surrogate_root_is_not_its_native_replacement() {
        let root = FsPath::from_code_units("/root/".encode_utf16().chain([0xd800]).collect());
        assert_eq!(
            relative_logical(&root, "/root/\u{fffd}/child/a.ts", Platform::Posix),
            "../\u{fffd}/child/a.ts"
        );
        assert_eq!(
            grep_path(&root, "/root/\u{fffd}/child/a.ts", Platform::Posix, true),
            "a.ts"
        );
        assert_eq!(
            logical_dirname(
                &logical_join(&root, ".git", Platform::Posix),
                Platform::Posix
            ),
            root
        );
        let unc: FsPath = "\\\\server\\share".into();
        assert_eq!(logical_dirname(&unc, Platform::Windows), unc);
        assert_eq!(
            relative(
                "\\\\server\\share",
                "\\\\other\\share\\x",
                Platform::Windows
            ),
            "\\\\other\\share\\x"
        );
    }
    #[test]
    fn windows_relative_is_case_insensitive_not_prefix_matching() {
        assert_eq!(
            relative("C:\\Root", "c:\\rOOT\\A.ts", Platform::Windows),
            "A.ts"
        );
        assert_eq!(
            relative("C:\\Root", "C:\\Rooted\\A.ts", Platform::Windows),
            "..\\Rooted\\A.ts"
        );
        assert_eq!(find_path("/root", "/root/d/", Platform::Posix), "d/");
    }
}
