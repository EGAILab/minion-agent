//! EXEC-013: Node v22.15.1's string diagnostic, before native projection.
use super::{Platform, SubprocessError};
use crate::llm::ResultString;
use std::collections::BTreeMap;

pub(super) fn effective_environment(
    baseline: &[(ResultString, ResultString)],
    overlay: &[(ResultString, ResultString)],
    inherit: bool,
    platform: Platform,
) -> Vec<(ResultString, ResultString)> {
    let mut values = BTreeMap::new();
    if inherit {
        values.extend(baseline.iter().cloned());
    }
    values.extend(overlay.iter().cloned());
    if platform == Platform::Windows {
        return values.into_iter().collect();
    }
    let mut indexed: Vec<_> = values
        .keys()
        .filter_map(|name| index(name).map(|i| (i, name.clone())))
        .collect();
    indexed.sort_by_key(|(i, _)| *i);
    let mut names: Vec<_> = indexed.into_iter().map(|(_, n)| n).collect();
    for (name, _) in overlay {
        if index(name).is_none() && !names.contains(name) {
            names.push(name.clone());
        }
    }
    names.extend(
        values
            .keys()
            .filter(|n| !names.contains(n))
            .cloned()
            .collect::<Vec<_>>(),
    );
    names
        .into_iter()
        .map(|name| {
            let value = values.remove(&name).expect("effective name");
            (name, value)
        })
        .collect()
}

fn index(value: &ResultString) -> Option<u32> {
    let units = value.code_units();
    if units.is_empty() || units.len() > 10 || (units.len() > 1 && units[0] == 48) {
        return None;
    }
    let mut number = 0u64;
    for &unit in units {
        if !(48..=57).contains(&unit) {
            return None;
        }
        number = number * 10 + u64::from(unit - 48);
    }
    (number < u64::from(u32::MAX)).then_some(number as u32)
}

pub(super) fn validate(
    argv: &[ResultString],
    cwd: &ResultString,
    env: &[(ResultString, ResultString)],
) -> Result<(), SubprocessError> {
    for (i, arg) in argv.iter().enumerate() {
        let name: ResultString = if i == 0 {
            "file".into()
        } else {
            format!("args[{}]", i - 1).into()
        };
        check(arg, &name, false)?;
    }
    check(cwd, &"options.cwd".into(), true)?;
    for (name, value) in env {
        let mut key: Vec<u16> = "options.env['".encode_utf16().collect();
        key.extend_from_slice(name.code_units());
        key.extend("']".encode_utf16());
        let key = ResultString::from_code_units(key);
        check(name, &key, false)?;
        check(value, &key, false)?;
    }
    Ok(())
}

fn check(value: &ResultString, name: &ResultString, cwd: bool) -> Result<(), SubprocessError> {
    if !value.code_units().contains(&0) {
        return Ok(());
    }
    let kind = if name.code_units().contains(&46) {
        "property"
    } else {
        "argument"
    };
    let reason = if cwd {
        "must be a string, Uint8Array, or URL without null bytes"
    } else {
        "must be a string without null bytes"
    };
    let mut message: Vec<u16> = format!("The {kind} '").encode_utf16().collect();
    message.extend_from_slice(name.code_units());
    message.extend(format!("' {reason}. Received ").encode_utf16());
    let mut received = inspect(value.code_units());
    if received.len() > 128 {
        received.truncate(128);
        received.extend("...".encode_utf16());
    }
    message.extend(received);
    Err(SubprocessError::nul_input(ResultString::from_code_units(
        message,
    )))
}

fn inspect(value: &[u16]) -> Vec<u16> {
    let kept = &value[..value.len().min(10000)];
    let mut output = Vec::new();
    if kept.len() > 76 {
        for (i, piece) in kept.split_inclusive(|u| *u == 10).enumerate() {
            if i != 0 {
                output.extend(" +\n  ".encode_utf16());
            }
            output.extend(quote(piece));
        }
    } else {
        output.extend(quote(kept));
    }
    if value.len() > kept.len() {
        let count = value.len() - kept.len();
        output.extend(
            format!(
                "... {count} more character{}",
                if count > 1 { "s" } else { "" }
            )
            .encode_utf16(),
        );
    }
    output
}

fn quote(value: &[u16]) -> Vec<u16> {
    let delimiter = if !value.contains(&39) {
        39
    } else if !value.contains(&34) {
        34
    } else if !value.contains(&96) && !value.windows(2).any(|p| p == [36, 123]) {
        96
    } else {
        39
    };
    let mut output = vec![delimiter];
    let mut i = 0;
    while i < value.len() {
        let unit = value[i];
        let escape = match unit {
            39 if delimiter == 39 => Some("\\'".to_owned()),
            92 => Some("\\\\".to_owned()),
            8 => Some("\\b".to_owned()),
            9 => Some("\\t".to_owned()),
            10 => Some("\\n".to_owned()),
            12 => Some("\\f".to_owned()),
            13 => Some("\\r".to_owned()),
            0..=31 | 127..=159 => Some(format!("\\x{unit:02X}")),
            0xd800..=0xdbff
                if value
                    .get(i + 1)
                    .is_some_and(|u| (0xdc00..=0xdfff).contains(u)) =>
            {
                output.extend_from_slice(&value[i..i + 2]);
                i += 2;
                continue;
            }
            0xd800..=0xdfff => Some(format!("\\u{unit:04x}")),
            _ => None,
        };
        if let Some(escape) = escape {
            output.extend(escape.encode_utf16());
        } else {
            output.push(unit);
        }
        i += 1;
    }
    output.push(delimiter);
    output
}
