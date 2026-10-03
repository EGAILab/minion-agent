//! Logical filesystem paths retain the certified UTF-16 string primitive.
//! Native projection is explicit and belongs immediately before an OS call.
use std::{
    env,
    path::{Path, PathBuf},
};

/// A JavaScript-string filesystem path, not an OS path or a JSON string.
pub type FsPath = crate::llm::ResultString;

pub(crate) fn native(path: &FsPath) -> PathBuf {
    PathBuf::from(String::from_utf16_lossy(path.code_units()))
}

pub(crate) fn from_native(path: &Path) -> FsPath {
    let text = path.to_string_lossy();
    if cfg!(windows) {
        if let Some(unc) = text.strip_prefix("\\\\?\\UNC\\") {
            return format!("\\\\{unc}").into();
        }
        if let Some(drive) = text.strip_prefix("\\\\?\\") {
            return drive.into();
        }
    }
    text.as_ref().into()
}

fn starts(path: &FsPath, prefix: &str) -> bool {
    path.code_units()
        .starts_with(&prefix.encode_utf16().collect::<Vec<_>>())
}

fn separator(unit: u16) -> bool {
    unit == 47 || (cfg!(windows) && unit == 92)
}

/// Strict file URL conversion has a WHATWG USVString boundary of its own.
pub fn file_url_to_js_path(raw: &FsPath, windows: bool) -> Option<FsPath> {
    super::filesystem::file_url_to_path(&String::from_utf16_lossy(raw.code_units()), windows)
        .map(FsPath::from)
}

pub(crate) fn resolve(cwd: &Path, raw: &FsPath) -> FsPath {
    let mut expanded = raw.clone();
    if starts(raw, "file://") {
        if let Some(converted) = file_url_to_js_path(raw, cfg!(windows)) {
            expanded = converted;
        }
    } else if (raw == "~" || starts(raw, "~/") || starts(raw, "~\\"))
        && let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))
    {
        let home = from_native(Path::new(&home));
        let tail = FsPath::from_code_units(raw.code_units().get(2..).unwrap_or_default().to_vec());
        expanded = join(&[&home, &tail]);
    }
    let units = expanded.code_units();
    let rooted = units.first().is_some_and(|unit| separator(*unit));
    let drive = cfg!(windows) && units.get(1) == Some(&58);
    if drive || (rooted && !cfg!(windows)) || (cfg!(windows) && units.starts_with(&[92, 92])) {
        normalize(units)
    } else {
        let base = from_native(cwd);
        if rooted && cfg!(windows) {
            let mut value = base.code_units()[..2].to_vec();
            value.extend_from_slice(units);
            normalize(&value)
        } else {
            join(&[&base, &expanded])
        }
    }
}

pub(crate) fn join(parts: &[&FsPath]) -> FsPath {
    let mut joined = Vec::new();
    for part in parts {
        if part.code_units().is_empty() {
            continue;
        }
        if !joined.is_empty() {
            joined.push(if cfg!(windows) { 92 } else { 47 });
        }
        joined.extend_from_slice(part.code_units());
    }
    normalize(&joined)
}

pub(crate) fn parent(path: &FsPath) -> FsPath {
    let units = path.code_units();
    match units.iter().rposition(|unit| separator(*unit)) {
        Some(0) => FsPath::from_code_units(units[..1].to_vec()),
        Some(2) if cfg!(windows) && units.get(1) == Some(&58) => {
            FsPath::from_code_units(units[..3].to_vec())
        }
        Some(index) => FsPath::from_code_units(units[..index].to_vec()),
        None => ".".into(),
    }
}

pub(crate) fn basename(path: &FsPath) -> FsPath {
    let units = path.code_units();
    let index = units
        .iter()
        .rposition(|unit| separator(*unit))
        .map_or(0, |i| i + 1);
    FsPath::from_code_units(units[index..].to_vec())
}

fn normalize(units: &[u16]) -> FsPath {
    let slash = if cfg!(windows) { 92 } else { 47 };
    let mut prefix = Vec::new();
    let mut rest = units;
    let mut root = false;
    if cfg!(windows) && units.get(1) == Some(&58) {
        prefix.extend_from_slice(&units[..2]);
        rest = &units[2..];
    } else if cfg!(windows) && units.len() > 2 && separator(units[0]) && separator(units[1]) {
        // A UNC server/share is an indivisible path prefix.
        let end = units[2..]
            .iter()
            .enumerate()
            .filter(|(_, u)| separator(**u))
            .nth(1)
            .map_or(units.len(), |(index, _)| index + 2);
        prefix.extend(
            units[..end]
                .iter()
                .map(|u| if separator(*u) { slash } else { *u }),
        );
        rest = &units[end..];
        root = true;
    }
    root |= rest.first().is_some_and(|u| separator(*u));
    let mut components: Vec<&[u16]> = Vec::new();
    for component in rest.split(|u| separator(*u)) {
        if component.is_empty() || component == [46] {
            continue;
        }
        if component == [46, 46] {
            if components.last().is_some_and(|c| *c != [46, 46]) {
                components.pop();
            } else if !root {
                components.push(component);
            }
        } else {
            components.push(component);
        }
    }
    let mut output = prefix;
    if root {
        output.push(slash);
    }
    for component in components {
        if !output.is_empty() && output.last() != Some(&slash) && output.last() != Some(&58) {
            output.push(slash);
        }
        output.extend_from_slice(component);
    }
    if output.is_empty() {
        output.push(46);
    }
    FsPath::from_code_units(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_resolution_preserves_the_certified_existing_binding() {
        let cwd = if cfg!(windows) {
            Path::new("C:\\root")
        } else {
            Path::new("/root")
        };
        for raw in ["", ".", "a/./b", "a/../b", "../b", "~", "~/file", "~\\file"] {
            assert_eq!(
                resolve(cwd, &raw.into()),
                from_native(&super::super::filesystem::resolve_local_path(cwd, raw)),
                "{raw:?}"
            );
        }
    }
    #[test]
    fn lexical_paths_do_not_project_surrogates() {
        let cwd = if cfg!(windows) {
            Path::new("C:\\root")
        } else {
            Path::new("/root")
        };
        let raw = FsPath::from_code_units(vec![97, 47, 0xd800, 47, 46, 46, 47, 0xdc00]);
        let path = resolve(cwd, &raw);
        assert_eq!(basename(&path).code_units(), &[0xdc00]);
        assert!(path.code_units().contains(&0xdc00));
        assert!(!from_native(&native(&path)).code_units().contains(&0xdc00));
        assert_eq!(basename(&from_native(&native(&path))), "\u{fffd}");
    }
    #[test]
    fn file_url_has_its_own_scalar_boundary() {
        let mut units: Vec<u16> = if cfg!(windows) {
            "file:///C:/a"
        } else {
            "file:///a"
        }
        .encode_utf16()
        .collect();
        units.push(0xd800);
        let path = file_url_to_js_path(&FsPath::from_code_units(units), cfg!(windows)).unwrap();
        assert!(path.code_units().contains(&0xfffd));
        assert!(!path.code_units().contains(&0xd800));
        assert!(file_url_to_js_path(&"file:///%ED%A0%80".into(), cfg!(windows)).is_none());
    }
}
