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

fn native_units(path: &Path) -> FsPath {
    #[cfg(windows)]
    let units = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().collect::<Vec<_>>()
    };
    #[cfg(not(windows))]
    let units = path.to_string_lossy().encode_utf16().collect::<Vec<_>>();
    FsPath::from_code_units(units)
}

pub(crate) fn from_native(path: &Path) -> FsPath {
    // Preserve the certified native filesystem-result/display mapping. The raw
    // codec below is scoped to logical cwd transport and selected native reads.
    without_extended_prefix(path.to_string_lossy().as_ref().into())
}

fn logical_provider_path(path: &Path) -> FsPath {
    without_extended_prefix(native_units(path))
}

fn without_extended_prefix(text: FsPath) -> FsPath {
    if cfg!(windows) {
        if starts(&text, "\\\\?\\UNC\\") {
            let mut unc = vec![92, 92];
            unc.extend_from_slice(&text.code_units()[8..]);
            return FsPath::from_code_units(unc);
        }
        if starts(&text, "\\\\?\\") {
            return FsPath::from_code_units(text.code_units()[4..].to_vec());
        }
    }
    text
}

/// Node's native UTF-16 -> generalized UTF-8 -> replacement-decode boundary.
/// A lone surrogate yields THREE replacements, unlike the separate OS-call
/// projection in `native`. A valid pair remains a pair and a BOM is retained.
pub(super) fn native_view(raw: &FsPath) -> FsPath {
    let mut output = Vec::new();
    for value in char::decode_utf16(raw.code_units().iter().copied()) {
        match value {
            Ok(value) => output.extend_from_slice(value.encode_utf16(&mut [0; 2])),
            Err(_) => output.extend_from_slice(&[0xfffd; 3]),
        }
    }
    FsPath::from_code_units(output)
}

fn inherited_drive_view(device: &[u16]) -> Option<FsPath> {
    let mut name = vec![61];
    name.extend_from_slice(device);
    env::var_os(String::from_utf16_lossy(&name))
        .map(|value| inherited_native_view(&native_units(Path::new(&value))))
}

fn inherited_native_view(raw: &FsPath) -> FsPath {
    native_view(raw)
}

fn native_cwd_view(raw: &FsPath, windows: bool) -> FsPath {
    // libuv uv_cwd trims exactly one native separator, except the literal
    // drive-root predicate on Windows and the single slash root on POSIX.
    // This is NOT the provider/display namespace mapping.
    let units = raw.code_units();
    let separator = if windows { 92 } else { 47 };
    let keep_root = if windows {
        units.len() == 3 && units.get(1) == Some(&58)
    } else {
        units.len() == 1
    };
    let end = if units.last() == Some(&separator) && !keep_root {
        units.len() - 1
    } else {
        units.len()
    };
    native_view(&FsPath::from_code_units(units[..end].to_vec()))
}

fn process_cwd_view() -> std::io::Result<FsPath> {
    view_native_cwd_read(env::current_dir())
}

/// The native-read boundary is instance-owned, not process-global. Tests can
/// script a failed read without deleting a real cwd or changing the host env.
pub(super) trait NativeInputs: std::fmt::Debug + Send + Sync {
    fn cwd_view(&self) -> std::io::Result<FsPath>;
    fn drive_view(&self, device: &[u16]) -> Option<FsPath>;
}

#[derive(Debug)]
pub(super) struct ProcessInputs;

impl NativeInputs for ProcessInputs {
    fn cwd_view(&self) -> std::io::Result<FsPath> {
        process_cwd_view()
    }

    fn drive_view(&self, device: &[u16]) -> Option<FsPath> {
        inherited_drive_view(device)
    }
}

fn view_native_cwd_read(read: std::io::Result<PathBuf>) -> std::io::Result<FsPath> {
    read.map(|cwd| native_cwd_view(&native_units(&cwd), cfg!(windows)))
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

#[cfg(test)]
pub(crate) fn resolve(cwd: &Path, raw: &FsPath) -> std::io::Result<FsPath> {
    resolve_with_inputs(cwd, raw, &ProcessInputs)
}

pub(super) fn resolve_with_inputs(
    cwd: &Path,
    raw: &FsPath,
    inputs: &dyn NativeInputs,
) -> std::io::Result<FsPath> {
    resolve_logical_with_inputs(&logical_provider_path(cwd), raw, inputs)
}

/// Logical cwd and input are both JavaScript strings. The OS-path convenience
/// adapter above does not make this entry point project either string.
#[cfg(test)]
pub(crate) fn resolve_logical(cwd: &FsPath, raw: &FsPath) -> std::io::Result<FsPath> {
    resolve_logical_with_inputs(cwd, raw, &ProcessInputs)
}

pub(super) fn resolve_logical_with_inputs(
    cwd: &FsPath,
    raw: &FsPath,
    inputs: &dyn NativeInputs,
) -> std::io::Result<FsPath> {
    let mut expanded = raw.clone();
    if starts(raw, "file://") {
        if let Some(converted) = file_url_to_js_path(raw, cfg!(windows)) {
            expanded = converted;
        }
    } else if (raw == "~" || starts(raw, "~/") || (cfg!(windows) && starts(raw, "~\\")))
        && let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))
    {
        let home = from_native(Path::new(&home));
        let tail = FsPath::from_code_units(raw.code_units().get(2..).unwrap_or_default().to_vec());
        expanded = join(&[&home, &tail]);
    }
    let base = cwd.clone();
    let parts = if super::path_resolve::is_absolute(&expanded, cfg!(windows)) {
        vec![&expanded]
    } else {
        vec![&base, &expanded]
    };
    super::path_resolve::resolve(
        &parts,
        cfg!(windows),
        |device| inputs.drive_view(device),
        || inputs.cwd_view(),
    )
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
    for (index, component) in components.into_iter().enumerate() {
        if !output.is_empty()
            && output.last() != Some(&slash)
            && !(index == 0 && !root && cfg!(windows) && output.len() == 2 && output[1] == 58)
        {
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
    fn native_cwd_windows_frozen() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/node_native_operand.json"))
                .unwrap();
        for row in fixture["cwd_rows"].as_array().unwrap() {
            let raw = FsPath::from_code_units(serde_json::from_value(row["raw"].clone()).unwrap());
            let expected =
                FsPath::from_code_units(serde_json::from_value(row["node"].clone()).unwrap());
            assert_eq!(
                native_cwd_view(&raw, true),
                expected,
                "native-cwd/{}",
                row["label"]
            );
        }
        for row in fixture["predicate_rows"].as_array().unwrap() {
            let raw = FsPath::from_code_units(serde_json::from_value(row["raw"].clone()).unwrap());
            let expected =
                FsPath::from_code_units(serde_json::from_value(row["expected"].clone()).unwrap());
            assert_eq!(
                native_cwd_view(&raw, true),
                expected,
                "predicate/{}",
                row["label"]
            );
        }
    }

    #[test]
    fn inherited_native_boundary_frozen() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/node_native_operand.json"))
                .unwrap();
        for row in fixture["entry_rows"].as_array().unwrap() {
            let raw = FsPath::from_code_units(serde_json::from_value(row["raw"].clone()).unwrap());
            let expected =
                FsPath::from_code_units(serde_json::from_value(row["node"].clone()).unwrap());
            assert_eq!(
                inherited_native_view(&raw),
                expected,
                "native-entry/{}",
                row["label"]
            );
        }
    }

    #[test]
    fn native_cwd_posix_boundary() {
        for (label, raw, expected) in [
            ("root", "/", "/"),
            ("trailing", "/tmp/native/", "/tmp/native"),
            ("one-only", "/tmp/native//", "/tmp/native/"),
            ("not-backslash", "/tmp/native\\", "/tmp/native\\"),
            ("bom", "/tmp/\u{feff}/", "/tmp/\u{feff}"),
        ] {
            assert_eq!(
                native_cwd_view(&raw.into(), false),
                expected,
                "posix-cwd/{label}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let raw = std::ffi::OsString::from_vec(vec![47, 120, 47, 0xed, 0xa0, 0x80]);
            assert_eq!(
                view_native_cwd_read(Ok(PathBuf::from(raw))).unwrap(),
                "/x/\u{fffd}\u{fffd}\u{fffd}"
            );
        }
    }

    #[test]
    fn native_cwd_read_failure_is_an_error_value() {
        let error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "native cwd denied");
        let error = view_native_cwd_read(Err(error)).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(error.to_string(), "native cwd denied");
        let error = view_native_cwd_read(Err(std::io::Error::from_raw_os_error(2))).unwrap_err();
        assert_eq!(
            error.raw_os_error(),
            Some(2),
            "retain the native code for the existing error mapper"
        );
        assert_eq!(
            view_native_cwd_read(Ok(PathBuf::from("native-cwd"))).unwrap(),
            "native-cwd"
        );
    }

    #[tokio::test]
    async fn logical_provider_cwd_stays_raw() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/node_logical_cwd.json")).unwrap();
        let platform = if cfg!(windows) { "win32" } else { "linux" };
        let mut count = 0;
        let mut providers = 0;
        for row in fixture["rows"].as_array().unwrap() {
            if row["platform"] != platform {
                continue;
            }
            let decode = |field: &str| {
                FsPath::from_code_units(serde_json::from_value(row[field].clone()).unwrap())
            };
            let cwd = decode("cwd");
            let input = decode("input");
            let expected = decode("node");
            #[cfg(windows)]
            let provider = {
                use std::os::windows::ffi::OsStringExt;
                Some(PathBuf::from(std::ffi::OsString::from_wide(
                    cwd.code_units(),
                )))
            };
            #[cfg(not(windows))]
            let provider = String::from_utf16(cwd.code_units()).ok().map(PathBuf::from);
            // A PathBuf can transport raw UTF-16 on Windows. On POSIX, the
            // lossless logical entry point covers the non-scalar cwd domain;
            // the public lossless carrier is the separate #224 / #228 work.
            if let Some(provider) = provider {
                let fs = super::super::LocalFileSystem::new(provider);
                assert_eq!(
                    fs.absolute_path(&input, None).await.unwrap(),
                    expected,
                    "real provider {}",
                    row["label"]
                );
                providers += 1;
            }
            assert_eq!(
                resolve_logical(&cwd, &input).unwrap(),
                expected,
                "logical cwd {}",
                row["label"]
            );
            count += 1;
        }
        assert_eq!(count, 5);
        assert_eq!(providers, if cfg!(windows) { 5 } else { 3 });
    }

    #[tokio::test]
    async fn canonical_baseline_through_real_provider() {
        let corpus: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/agent/path-resolution/path-resolution.json"
        ))
        .unwrap();
        let platform = if cfg!(windows) { "win32" } else { "linux" };
        let mut count = 0;
        for row in corpus["path_resolution"]["cases"].as_array().unwrap() {
            if row["platform"] != platform || !row["host"].is_null() {
                continue;
            }
            if cfg!(windows) {
                for drive in row["preconditions"]["absent_entries"].as_array().unwrap() {
                    let key = format!("={drive}:", drive = drive.as_str().unwrap());
                    assert!(env::var_os(key).is_none(), "PRECONDITION REFUSED");
                }
                let cwd = env::current_dir().unwrap();
                let drive = native_units(&cwd).code_units()[0] as u8;
                for forbidden in row["preconditions"]["process_cwd_drive_not"]
                    .as_array()
                    .unwrap()
                {
                    assert_ne!(
                        drive.to_ascii_uppercase(),
                        forbidden.as_str().unwrap().as_bytes()[0],
                        "PRECONDITION REFUSED"
                    );
                }
            }
            let decode = |value: &serde_json::Value| {
                FsPath::from_code_units(serde_json::from_value(value["utf16"].clone()).unwrap())
            };
            let base = decode(&row["provider_cwd"]);
            let provider = native(&base); // canonical provider cwds are scalar
            let fs = super::super::LocalFileSystem::new(&provider);
            let input = decode(&row["input"]);
            assert_eq!(
                fs.absolute_path(&input, None).await.unwrap(),
                decode(&row["expect"]),
                "{}",
                row["id"]
            );
            count += 1;
        }
        assert_eq!(count, if cfg!(windows) { 27 } else { 29 });
    }

    /// External launcher supplies only raw inputs and preconditions, never an
    /// expected result. Every case path is a VALUE: no I/O uses it as a target.
    #[tokio::test]
    #[ignore = "explicit crafted-environment child, launched by path-resolution-probe.py"]
    async fn native_observer() {
        use serde_json::{Value, json};
        let payload: Value =
            serde_json::from_str(&env::var("MINION_PATH_PROBE").expect("probe payload")).unwrap();
        let process = env::current_dir().unwrap();
        let cwd = native_units(&process);
        if cfg!(windows) {
            for name in payload["absent_entries"].as_array().unwrap() {
                assert!(
                    env::var_os(format!("={}:", name.as_str().unwrap())).is_none(),
                    "PRECONDITION REFUSED: inherited entry"
                );
            }
            for drive in payload["process_cwd_drive_not"].as_array().unwrap() {
                assert_ne!(
                    (logical_provider_path(&process).code_units()[0] as u8).to_ascii_uppercase(),
                    drive.as_str().unwrap().as_bytes()[0],
                    "PRECONDITION REFUSED: process cwd drive"
                );
            }
        }
        let base: Vec<u16> = serde_json::from_value(payload["provider_cwd"].clone()).unwrap();
        #[cfg(windows)]
        let provider = {
            use std::os::windows::ffi::OsStringExt;
            PathBuf::from(std::ffi::OsString::from_wide(&base))
        };
        #[cfg(not(windows))]
        let provider = PathBuf::from(String::from_utf16(&base).unwrap());
        let fs = super::super::LocalFileSystem::new(&provider);
        let inputs: Vec<Vec<u16>> = serde_json::from_value(payload["inputs"].clone()).unwrap();
        let mut rows = Vec::new();
        for units in inputs {
            let input = FsPath::from_code_units(units);
            let actual = fs.absolute_path(&input, None).await.unwrap();
            let logical = resolve(&provider, &input).unwrap();
            let scalar = String::from_utf16(input.code_units()).ok().map(|raw| {
                native_units(
                    &super::super::filesystem::resolve_local_path(&provider, &raw).unwrap(),
                )
            });
            rows.push(json!({"absolute_path":actual.code_units(),"logical":logical.code_units(),"scalar":scalar.map(|v|v.code_units().to_vec())}));
        }
        let entries = ["A", "E", "X"].map(|drive| {
            (
                drive,
                env::var_os(format!("={drive}:"))
                    .map(|value| native_units(Path::new(&value)).code_units().to_vec()),
            )
        });
        let cwd_view = process_cwd_view().unwrap();
        let entry_views = ["A", "E", "X"].map(|drive| {
            (
                drive,
                inherited_drive_view(&format!("{drive}:").encode_utf16().collect::<Vec<_>>())
                    .map(|value| value.code_units().to_vec()),
            )
        });
        println!(
            "PATH_PROBE {}",
            json!({"cwd":cwd.code_units(),"entries":entries,"rows":rows,
                "cwd_view":cwd_view.code_units(), "entry_views":entry_views})
        );
    }
    #[test]
    fn scalar_and_lossless_adapters_share_the_resolver() {
        let cwd = if cfg!(windows) {
            Path::new("C:\\root")
        } else {
            Path::new("/root")
        };
        for raw in ["", ".", "a/./b", "a/../b", "../b", "~", "~/file", "~\\file"] {
            assert_eq!(
                resolve(cwd, &raw.into()).unwrap(),
                from_native(&super::super::filesystem::resolve_local_path(cwd, raw).unwrap()),
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
        let path = resolve(cwd, &raw).unwrap();
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
