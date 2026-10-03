//! EXEC-010: owned, lossless native environment snapshots; no consumer deduplication.
use std::{collections::BTreeMap, sync::OnceLock};

use crate::javascript::JsString;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Platform {
    Windows,
    Posix,
}

impl Platform {
    pub const fn local() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else {
            Self::Posix
        }
    }
}

/// Owned native entries. A consumer can edit a COPY without editing the snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnvEntries {
    Posix(Vec<(Vec<u8>, Vec<u8>)>),
    Windows(Vec<(JsString, JsString)>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnvValue<'a> {
    Bytes(&'a [u8]),
    Windows(&'a JsString),
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// The public API exposes only borrowed entries and owned copies, never mutable entries.
/// ```compile_fail
/// use minion_agent::execution::{EnvEntries, EnvSnapshot};
/// let mut snapshot = EnvSnapshot::new(EnvEntries::Posix(vec![]));
/// snapshot.entries = EnvEntries::Posix(vec![]);
/// assert_eq!(snapshot.entries(), &EnvEntries::Posix(vec![]));
/// ```
/// This positive compilation witness also prevents a broken link environment making the
/// compile-fail protection above pass vacuously.
/// ```
/// use minion_agent::execution::{EnvEntries, EnvSnapshot};
/// let snapshot = EnvSnapshot::new(EnvEntries::Posix(vec![]));
/// assert_eq!(snapshot.entries(), &EnvEntries::Posix(vec![]));
/// ```
pub struct EnvSnapshot {
    entries: EnvEntries,
}

impl EnvSnapshot {
    pub fn new(entries: EnvEntries) -> Self {
        Self { entries }
    }

    pub fn platform(&self) -> Platform {
        match self.entries {
            EnvEntries::Posix(_) => Platform::Posix,
            EnvEntries::Windows(_) => Platform::Windows,
        }
    }

    pub fn entries(&self) -> &EnvEntries {
        &self.entries
    }

    pub fn copy(&self) -> EnvEntries {
        self.entries.clone()
    }

    pub fn get(&self, name: &str) -> Option<EnvValue<'_>> {
        match &self.entries {
            EnvEntries::Posix(entries) => entries.iter().find_map(|(key, value)| {
                (key == name.as_bytes()).then_some(EnvValue::Bytes(value))
            }),
            EnvEntries::Windows(_) => self
                .get_windows(&JsString::from(name))
                .map(EnvValue::Windows),
        }
    }

    pub fn get_windows(&self, name: &JsString) -> Option<&JsString> {
        let EnvEntries::Windows(entries) = &self.entries else {
            return None;
        };
        let wanted = windows_key(name);
        entries
            .iter()
            .find_map(|(key, value)| (windows_key(key) == wanted).then_some(value))
    }

    pub(super) fn configured(platform: Platform, map: &BTreeMap<String, String>) -> Self {
        Self::new(match platform {
            Platform::Windows => EnvEntries::Windows(
                map.iter()
                    .map(|(k, v)| (k.clone().into(), v.clone().into()))
                    .collect(),
            ),
            Platform::Posix => EnvEntries::Posix(
                map.iter()
                    .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
                    .collect(),
            ),
        })
    }
}

/// The exact shared build-26200 table, including identity for unmapped units.
pub fn pinned_windows_uppercase(unit: u16) -> u16 {
    static TABLE: OnceLock<BTreeMap<u16, u16>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let data: serde_json::Value = serde_json::from_str(include_str!("windows_upcase.json"))
            .expect("verified Windows table JSON");
        data["map"]
            .as_object()
            .expect("Windows table map")
            .iter()
            .map(|(k, v)| {
                (
                    u16::from_str_radix(k, 16).expect("table unit"),
                    u16::from_str_radix(v.as_str().expect("table uppercase"), 16)
                        .expect("table uppercase unit"),
                )
            })
            .collect()
    });
    table.get(&unit).copied().unwrap_or(unit)
}

fn windows_key(name: &JsString) -> Vec<u16> {
    name.code_units()
        .iter()
        .copied()
        .map(native_uppercase)
        .collect()
}

#[cfg(windows)]
fn native_uppercase(unit: u16) -> u16 {
    minion_agent_native_env::uppercase_unit(unit)
}

#[cfg(not(windows))]
fn native_uppercase(unit: u16) -> u16 {
    pinned_windows_uppercase(unit)
}
