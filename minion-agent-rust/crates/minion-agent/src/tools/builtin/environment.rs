//! Section 15.5: Node's view and explicit-spawn arbitration, not provider capture policy.
use crate::{
    execution::{EnvEntries, EnvSnapshot, Platform},
    javascript::JsString,
    tools::ToolCapabilityError,
};
use std::collections::{BTreeMap, BTreeSet};

fn generalized_utf8(value: &JsString) -> Vec<u8> {
    let mut output = Vec::new();
    for decoded in char::decode_utf16(value.code_units().iter().copied()) {
        match decoded {
            Ok(c) => {
                let mut bytes = [0; 4];
                output.extend_from_slice(c.encode_utf8(&mut bytes).as_bytes());
            }
            Err(error) => {
                let unit = error.unpaired_surrogate();
                output.extend([
                    0xe0 | (unit >> 12) as u8,
                    0x80 | ((unit >> 6) & 63) as u8,
                    0x80 | (unit & 63) as u8,
                ]);
            }
        }
    }
    output
}

/// Invalid names are dropped; values use maximal-subpart replacement, retaining a BOM.
pub fn node_environment_view(snapshot: &EnvSnapshot) -> BTreeMap<String, String> {
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = match snapshot.entries() {
        EnvEntries::Posix(entries) => entries.clone(),
        EnvEntries::Windows(entries) => entries
            .iter()
            .map(|(n, v)| (generalized_utf8(n), generalized_utf8(v)))
            .collect(),
    };
    pairs
        .into_iter()
        .filter_map(|(n, v)| {
            String::from_utf8(n)
                .ok()
                .map(|n| (n, String::from_utf8_lossy(&v).into_owned()))
        })
        .collect()
}

/// Node's UTF-16-first / ECMAScript-uppercase duplicate arbitration for explicit env.
pub fn windows_spawn_environment(
    env: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, ToolCapabilityError> {
    let mut names: Vec<&String> = env.keys().collect();
    names.sort_by_cached_key(|n| n.encode_utf16().collect::<Vec<_>>());
    let mut chosen = BTreeSet::new();
    let mut output = BTreeMap::new();
    let inputs: Vec<&str> = names.iter().map(|name| name.as_str()).collect();
    let uppers = minion_agent_pinned_icu::upper_unicode16_batch(&inputs)
        .map_err(ToolCapabilityError::new)?;
    for (name, upper) in names.into_iter().zip(uppers) {
        if chosen.insert(upper) {
            output.insert(name.clone(), env[name].clone());
        }
    }
    Ok(output)
}

pub fn compose_spawn_environment(
    snapshot: &EnvSnapshot,
    remove: &[&str],
    inject: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, ToolCapabilityError> {
    let mut env = node_environment_view(snapshot);
    for name in remove {
        env.remove(*name);
    }
    env.extend(inject.clone());
    if snapshot.platform() == Platform::Windows {
        windows_spawn_environment(&env)
    } else {
        Ok(env)
    }
}
