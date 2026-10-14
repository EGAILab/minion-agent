//! Lexical Node v22.15.1 path.resolve on JavaScript UTF-16 strings.
//! No PathBuf component interpretation and no filesystem access belong here.
use super::{FsPath, node_lower::lower};

fn sep(unit: u16, windows: bool) -> bool {
    unit == 47 || (windows && unit == 92)
}

fn drive(unit: u16) -> bool {
    (65..=90).contains(&unit) || (97..=122).contains(&unit)
}

#[derive(Default)]
struct Root {
    device: Vec<u16>,
    end: usize,
    absolute: bool,
}

// Node's root scan deliberately recognizes a drive only at code-unit zero.
fn root(path: &[u16]) -> Root {
    let mut result = Root::default();
    if path.first().is_some_and(|&u| sep(u, true)) {
        result.absolute = true;
        if path.len() == 1 || !sep(path[1], true) {
            result.end = 1;
        } else {
            let mut index = 2;
            let server = index;
            while index < path.len() && !sep(path[index], true) {
                index += 1;
            }
            if index < path.len() && index != server {
                let server_end = index;
                while index < path.len() && sep(path[index], true) {
                    index += 1;
                }
                let share = index;
                while index < path.len() && !sep(path[index], true) {
                    index += 1;
                }
                if index != share {
                    result.device.extend_from_slice(&[92, 92]);
                    result.device.extend_from_slice(&path[server..server_end]);
                    result.device.push(92);
                    result.device.extend_from_slice(&path[share..index]);
                    result.end = index;
                }
            }
        }
    } else if path.len() >= 2 && drive(path[0]) && path[1] == 58 {
        result.device.extend_from_slice(&path[..2]);
        result.end = 2;
        if path.get(2).is_some_and(|&u| sep(u, true)) {
            result.absolute = true;
            result.end = 3;
        }
    }
    result
}

fn tail(path: &[u16], above_root: bool, windows: bool) -> Vec<u16> {
    let mut components: Vec<&[u16]> = Vec::new();
    for component in path.split(|&u| sep(u, windows)) {
        if component.is_empty() || component == [46] {
            continue;
        }
        if component == [46, 46] {
            if components.last().is_some_and(|&part| part != [46, 46]) {
                components.pop();
            } else if above_root {
                components.push(component);
            }
        } else {
            components.push(component);
        }
    }
    let mut result = Vec::new();
    for component in components {
        if !result.is_empty() {
            result.push(if windows { 92 } else { 47 });
        }
        result.extend_from_slice(component);
    }
    result
}

/// The two callbacks supply ALREADY viewed native values, lazily. Logical
/// arguments never cross that boundary. This is the Node select-then-check
/// algorithm, including its intentionally unusual conjunction at unit 2.
pub(super) fn resolve(
    parts: &[&FsPath],
    windows: bool,
    mut inherited: impl FnMut(&[u16]) -> Option<FsPath>,
    mut process_cwd: impl FnMut() -> FsPath,
) -> FsPath {
    let mut device = Vec::new();
    let mut combined = Vec::new();
    let mut absolute = false;
    for index in (0..=parts.len()).rev() {
        let path = if index > 0 {
            if parts[index - 1].code_units().is_empty() {
                continue;
            }
            (*parts[index - 1]).clone()
        } else if windows && !device.is_empty() {
            let selected = inherited(&device)
                .filter(|value| !value.code_units().is_empty())
                .unwrap_or_else(&mut process_cwd);
            let units = selected.code_units();
            if lower(&units[..units.len().min(2)]) != lower(&device) && units.get(2) == Some(&92) {
                let mut rooted = device.clone();
                rooted.push(92);
                FsPath::from_code_units(rooted)
            } else {
                selected
            }
        } else {
            process_cwd()
        };
        let units = path.code_units();
        let parsed = if windows {
            root(units)
        } else {
            Root {
                absolute: units.first() == Some(&47),
                ..Root::default()
            }
        };
        if !parsed.device.is_empty() {
            if !device.is_empty() && lower(&device) != lower(&parsed.device) {
                continue;
            }
            if device.is_empty() {
                device = parsed.device;
            }
        }
        if !absolute {
            let mut next = units[parsed.end..].to_vec();
            next.push(if windows { 92 } else { 47 });
            next.extend_from_slice(&combined);
            combined = next;
            absolute = parsed.absolute;
        }
        if absolute && (!windows || !device.is_empty()) {
            break;
        }
    }
    let mut result = device;
    if absolute {
        result.push(if windows { 92 } else { 47 });
    }
    result.extend(tail(&combined, !absolute, windows));
    if result.is_empty() {
        result.push(46);
    }
    FsPath::from_code_units(result)
}

pub(super) fn is_absolute(path: &FsPath, windows: bool) -> bool {
    if windows {
        root(path.code_units()).absolute
    } else {
        path.code_units().first() == Some(&47)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn string(value: &Value) -> FsPath {
        FsPath::from_code_units(serde_json::from_value(value.clone()).unwrap())
    }

    #[test]
    fn node_frozen_win32_port_rows() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/data/node_win32_resolve.json")).unwrap();
        for row in fixture["rows"].as_array().unwrap() {
            if let Some(input) = row["normalize"].as_str() {
                assert_eq!(
                    FsPath::from_code_units(tail(
                        &input.encode_utf16().collect::<Vec<_>>(),
                        true,
                        true
                    )),
                    row["out"].as_str().unwrap()
                );
                continue;
            }
            let base: FsPath = row["base"].as_str().unwrap_or("C:\\x").into();
            let input: FsPath = row["input"].as_str().unwrap().into();
            let actual = resolve(
                &[&base, &input],
                true,
                |_| {
                    Some(
                        row["entry"]
                            .as_str()
                            .expect("unexpected native environment read")
                            .into(),
                    )
                },
                || panic!("unexpected native cwd read: {row}"),
            );
            assert_eq!(actual, row["out"].as_str().unwrap(), "{row}");
        }
    }

    #[test]
    fn node_frozen_selected_entry_and_logical_rows() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/data/node_selected_value.json"))
                .unwrap();
        for row in fixture["entry_rows"].as_array().unwrap() {
            let base: FsPath = "C:\\x".into();
            let input: FsPath = "a:leaf".into();
            let actual = resolve(
                &[&base, &input],
                true,
                |_| {
                    (!row["entry"].is_null())
                        .then(|| super::super::path::native_view(&string(&row["entry"])))
                },
                || "E:\\native".into(),
            );
            assert_eq!(actual, string(&row["node"]), "{}", row["label"]);
        }
        for row in fixture["logical_rows"].as_array().unwrap() {
            let base: FsPath = "C:\\x".into();
            let input = string(&row["logical"]);
            let actual = resolve(
                &[&base, &input],
                true,
                |_| Some(super::super::path::native_view(&string(&row["entry"]))),
                || panic!("truthy inherited entry must keep cwd lazy"),
            );
            assert_eq!(actual, string(&row["node"]), "{}", row["label"]);
        }
    }

    #[test]
    fn posix_colons_remain_ordinary_components() {
        let base: FsPath = "/x".into();
        for (raw, expected) in [
            ("file:///C:/a%2Fb", "/x/file:/C:/a%2Fb"),
            ("a::/b", "/x/a::/b"),
            ("x:y:z/w", "/x/x:y:z/w"),
        ] {
            assert_eq!(
                resolve(
                    &[&base, &raw.into()],
                    false,
                    |_| panic!("POSIX has no per-drive lookup"),
                    || panic!("absolute provider")
                ),
                expected
            );
        }
    }
}
