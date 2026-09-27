//! R006-C: fail-closed ICU4C 78.3 identity and the pinned `ls` comparator.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    fs,
    path::Path,
};

use rust_icu_sys as sys;
use rust_icu_ucol::UCollator;
use sha2::{Digest, Sha256};

const SOURCE_SHA512: &str = "04a49455e1489030c520a4bfd2664fa2171e7938d08f2acdbbcb1fda976639fd8b1f0704f2eec89ba59a7b6d118ceaab6ec5a096e40d9085a0895d91ce225245";

fn failed(detail: impl AsRef<str>) -> String {
    format!("Pinned ICU4C 78.3 is unavailable: {}", detail.as_ref())
}

fn library_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

fn is_icu_library(path: &Path) -> bool {
    let name = library_name(path);
    if cfg!(windows) {
        name.starts_with("icu") && name.ends_with(".dll")
    } else {
        name.starts_with("libicu") && name.contains(".so")
    }
}

#[cfg(unix)]
fn loaded_modules() -> Result<Vec<std::path::PathBuf>, String> {
    let maps = fs::read_to_string("/proc/self/maps").map_err(|error| failed(error.to_string()))?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let path = fields.nth(5);
        if let Some(path) = path
            && path.starts_with('/')
        {
            paths.insert(std::path::PathBuf::from(path));
        }
    }
    Ok(paths.into_iter().collect())
}

#[cfg(windows)]
fn loaded_modules() -> Result<Vec<std::path::PathBuf>, String> {
    use std::{ffi::c_void, os::windows::ffi::OsStringExt};
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn EnumProcessModules(
            process: *mut c_void,
            handles: *mut *mut c_void,
            bytes: u32,
            needed: *mut u32,
        ) -> i32;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetModuleFileNameW(module: *mut c_void, path: *mut u16, capacity: u32) -> u32;
    }
    // Every module is inventoried, not merely the first module of a given basename.
    let process = unsafe { GetCurrentProcess() };
    let mut count = 1024usize;
    let handles = loop {
        let mut handles = vec![std::ptr::null_mut(); count];
        let mut needed = 0u32;
        let ok = unsafe {
            EnumProcessModules(
                process,
                handles.as_mut_ptr(),
                (handles.len() * std::mem::size_of::<*mut c_void>()) as u32,
                &mut needed,
            )
        };
        if ok == 0 {
            return Err(failed("cannot enumerate loaded modules"));
        }
        if needed as usize <= handles.len() * std::mem::size_of::<*mut c_void>() {
            handles.truncate(needed as usize / std::mem::size_of::<*mut c_void>());
            break handles;
        }
        count = needed as usize / std::mem::size_of::<*mut c_void>() + 64;
    };
    let mut paths = Vec::with_capacity(handles.len());
    for handle in handles {
        let mut buffer = vec![0u16; 32768];
        let length = unsafe { GetModuleFileNameW(handle, buffer.as_mut_ptr(), buffer.len() as u32) }
            as usize;
        if length == 0 || length >= buffer.len() {
            return Err(failed("cannot read a loaded module's complete path"));
        }
        paths.push(std::path::PathBuf::from(std::ffi::OsString::from_wide(
            &buffer[..length],
        )));
    }
    Ok(paths)
}

fn verify_identity(modules: &[std::path::PathBuf], identity_path: &Path) -> Result<(), String> {
    let identity = fs::read_to_string(identity_path)
        .map_err(|error| failed(format!("identity file: {error}")))?;
    let mut source = None;
    let mut libraries = BTreeMap::new();
    for line in identity.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["source-sha512", hash] => source = Some(*hash),
            ["library", role, name, hash] => {
                libraries.insert(name.to_ascii_lowercase(), (*role, *hash));
            }
            _ => {}
        }
    }
    if source != Some(SOURCE_SHA512) {
        return Err(failed("identity does not name the pinned source tarball"));
    }
    let mut roles = BTreeSet::new();
    for path in modules.iter().filter(|path| is_icu_library(path)) {
        let name = library_name(path);
        let (role, expected) = libraries
            .get(&name)
            .ok_or_else(|| failed(format!("unlisted ICU library {}", path.display())))?;
        let data = fs::read(path).map_err(|error| {
            failed(format!(
                "unreadable ICU library {}: {error}",
                path.display()
            ))
        })?;
        let actual = Sha256::digest(data);
        let actual: String = actual.iter().map(|byte| format!("{byte:02x}")).collect();
        if actual != *expected {
            return Err(failed(format!(
                "ICU library {} does not match the verified build",
                path.display()
            )));
        }
        roles.insert(*role);
    }
    for required in ["icuuc", "icui18n", "icudata"] {
        if !roles.contains(required) {
            return Err(failed(format!("required library {required} is not loaded")));
        }
    }
    Ok(())
}

fn root_lower(input: &str) -> Result<String, String> {
    let source: Vec<u16> = input.encode_utf16().collect();
    let locale = CString::new("").expect("empty locale has no NUL byte");
    let mut output = vec![0u16; source.len().saturating_mul(4).saturating_add(16)];
    let mut status = sys::UErrorCode::U_ZERO_ERROR;
    let len = unsafe {
        sys::versioned_function!(u_strToLower)(
            output.as_mut_ptr(),
            output.len() as i32,
            source.as_ptr(),
            source.len() as i32,
            locale.as_ptr(),
            &mut status,
        )
    };
    if status != sys::UErrorCode::U_ZERO_ERROR || len < 0 || len as usize > output.len() {
        return Err(failed(format!("ICU root lowercase failed: {status:?}")));
    }
    String::from_utf16(&output[..len as usize]).map_err(|error| failed(error.to_string()))
}

pub fn sort_names(names: Vec<String>) -> Result<Vec<String>, String> {
    let collator = UCollator::try_from("en-001").map_err(|error| failed(error.to_string()))?;
    collator
        .set_attribute(
            sys::UColAttribute::UCOL_STRENGTH,
            sys::UColAttributeValue::UCOL_TERTIARY,
        )
        .map_err(|error| failed(error.to_string()))?;
    collator
        .set_attribute(
            sys::UColAttribute::UCOL_NUMERIC_COLLATION,
            sys::UColAttributeValue::UCOL_OFF,
        )
        .map_err(|error| failed(error.to_string()))?;
    collator
        .set_attribute(
            sys::UColAttribute::UCOL_CASE_FIRST,
            sys::UColAttributeValue::UCOL_OFF,
        )
        .map_err(|error| failed(error.to_string()))?;
    collator
        .set_attribute(
            sys::UColAttribute::UCOL_NORMALIZATION_MODE,
            sys::UColAttributeValue::UCOL_ON,
        )
        .map_err(|error| failed(error.to_string()))?;
    let mut version: sys::UVersionInfo = [0; 4];
    unsafe { sys::versioned_function!(u_getVersion)(version.as_mut_ptr()) };
    if version != [78, 3, 0, 0] {
        return Err(failed(format!(
            "runtime version is {}.{}",
            version[0], version[1]
        )));
    }
    let identity = std::env::var_os("MINION_AGENT_ICU_IDENTITY")
        .ok_or_else(|| failed("MINION_AGENT_ICU_IDENTITY is not set"))?;
    verify_identity(&loaded_modules()?, Path::new(&identity))?;
    let mut keyed = names
        .into_iter()
        .map(|name| root_lower(&name).map(|key| (key, name)))
        .collect::<Result<Vec<_>, _>>()?;
    // `sort_by` is stable. Comparator errors cannot be hidden behind `Ordering::Equal`.
    let mut compare_error = None;
    keyed.sort_by(|a, b| match collator.strcoll_utf8(&a.0, &b.0) {
        Ok(ordering) => ordering,
        Err(error) => {
            compare_error = Some(failed(error.to_string()));
            Ordering::Equal
        }
    });
    if let Some(error) = compare_error {
        return Err(error);
    }
    Ok(keyed.into_iter().map(|(_, name)| name).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_checks_every_loaded_instance_and_required_role() {
        let temp = tempfile::tempdir().unwrap();
        let roles = ["icuuc", "icui18n", "icudata"];
        let mut lines = vec![format!("source-sha512 {SOURCE_SHA512}")];
        let mut modules = Vec::new();
        for role in roles {
            let name = if cfg!(windows) {
                format!("{role}78.dll")
            } else {
                format!("lib{role}.so.78")
            };
            let path = temp.path().join(&name);
            let bytes = role.as_bytes();
            fs::write(&path, bytes).unwrap();
            let hash: String = Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            lines.push(format!("library {role} {name} {hash}"));
            modules.push(path);
        }
        let identity = temp.path().join("identity.txt");
        fs::write(&identity, lines.join("\n")).unwrap();
        verify_identity(&modules, &identity).unwrap();

        let missing = &modules[..2];
        assert!(
            verify_identity(missing, &identity)
                .unwrap_err()
                .contains("icudata")
        );

        let foreign = temp.path().join(if cfg!(windows) {
            "icuuc72.dll"
        } else {
            "libicuuc.so.72"
        });
        fs::write(&foreign, b"foreign").unwrap();
        let mut with_foreign = modules.clone();
        with_foreign.push(foreign);
        assert!(
            verify_identity(&with_foreign, &identity)
                .unwrap_err()
                .contains("unlisted")
        );

        fs::write(&modules[0], b"changed").unwrap();
        assert!(
            verify_identity(&modules, &identity)
                .unwrap_err()
                .contains("does not match")
        );
    }

    #[test]
    fn identity_rejects_an_unlisted_duplicate_and_missing_required_role() {
        assert!(verify_identity(&[], Path::new("missing-identity-file")).is_err());
        assert!(is_icu_library(Path::new("icuuc72.dll")) == cfg!(windows));
    }

    #[test]
    fn pinned_sort_uses_stable_root_lowercase_and_normalization() {
        let sorted = sort_names(vec![
            "z".into(),
            "a".into(),
            "A".into(),
            "e\u{301}".into(),
            "é".into(),
        ])
        .unwrap();
        assert_eq!(&sorted[..2], ["a", "A"]);
        assert_eq!(&sorted[2..4], ["e\u{301}", "é"]);
        assert_eq!(sorted[4], "z");
    }
}
