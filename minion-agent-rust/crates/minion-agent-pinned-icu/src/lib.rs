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

// rust_icu_sys 5.8 exposes filtered normalization but omits the two UnicodeSet
// constructors it needs. These signatures come from the pinned ICU4C 78.3 uset.h.
// The versioned symbols intentionally cannot link to an arbitrary system ICU.
#[link(name = "icuuc")]
unsafe extern "C" {
    fn uset_openPattern_78(
        pattern: *const u16,
        length: i32,
        status: *mut sys::UErrorCode,
    ) -> *mut sys::USet;
    fn uset_close_78(set: *mut sys::USet);
}

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
fn module_path_with(
    getter: impl FnOnce(&mut [u16]) -> usize,
) -> Result<std::path::PathBuf, String> {
    use std::os::windows::ffi::OsStringExt;

    let mut buffer = vec![0u16; 32768];
    let length = getter(&mut buffer);
    if length == 0 || length >= buffer.len() {
        return Err(failed("cannot read a loaded module's complete path"));
    }
    Ok(std::path::PathBuf::from(std::ffi::OsString::from_wide(
        &buffer[..length],
    )))
}

#[cfg(windows)]
fn loaded_module_paths_with<H>(
    handles: impl IntoIterator<Item = H>,
    mut getter: impl FnMut(H, &mut [u16]) -> usize,
) -> Result<Vec<std::path::PathBuf>, String> {
    let mut paths = Vec::new();
    for handle in handles {
        paths.push(module_path_with(|buffer| getter(handle, buffer))?);
    }
    Ok(paths)
}

#[cfg(windows)]
fn loaded_modules() -> Result<Vec<std::path::PathBuf>, String> {
    use std::ffi::c_void;
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
    loaded_module_paths_with(handles, |handle, buffer| unsafe {
        GetModuleFileNameW(handle, buffer.as_mut_ptr(), buffer.len() as u32) as usize
    })
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

/// TOOL-031's Node-22.15.1 Unicode-16 NFKC view over the certified ICU4C build.
/// Filtering by assignment age also preserves multi-code-point composition, unlike
/// normalizing each scalar independently or post-hoc undoing Unicode-17 mappings.
pub fn nfkc_unicode16(input: &str) -> Result<String, String> {
    nfkc_unicode16_batch(&[input]).map(|mut values| values.remove(0))
}

/// Normalize one transaction's strings under a single verified artifact identity.
/// There is no persistent pin cache: every batch re-verifies the loaded build.
pub fn nfkc_unicode16_batch(inputs: &[&str]) -> Result<Vec<String>, String> {
    // Materialize the same complete ICU library set as the existing pin verifier.
    let _runtime = UCollator::try_from("en-001").map_err(|e| failed(e.to_string()))?;
    let identity = std::env::var_os("MINION_AGENT_ICU_IDENTITY")
        .ok_or_else(|| failed("MINION_AGENT_ICU_IDENTITY is not set"))?;
    verify_identity(&loaded_modules()?, Path::new(&identity))?;
    let mut version = [0; 4];
    unsafe { sys::versioned_function!(u_getVersion)(version.as_mut_ptr()) };
    if version != [78, 3, 0, 0] {
        return Err(failed("normalization runtime is not 78.3"));
    }
    let pattern: Vec<u16> = "[:age=16.0:]".encode_utf16().collect();
    let mut status = sys::UErrorCode::U_ZERO_ERROR;
    // The filter and filtered normalizer are local, immutable for the duration of
    // normalization, and closed in reverse order on every path. ICU owns the base.
    unsafe {
        let base = sys::versioned_function!(unorm2_getNFKCInstance)(&mut status);
        if status as i32 > 0 || base.is_null() {
            return Err(failed("cannot obtain NFKC normalizer"));
        }
        let filter = uset_openPattern_78(pattern.as_ptr(), pattern.len() as i32, &mut status);
        if status as i32 > 0 || filter.is_null() {
            if !filter.is_null() {
                uset_close_78(filter);
            }
            return Err(failed("cannot obtain Unicode-16 filter"));
        }
        let normalizer = sys::versioned_function!(unorm2_openFiltered)(base, filter, &mut status);
        if status as i32 > 0 || normalizer.is_null() {
            uset_close_78(filter);
            return Err(failed("cannot obtain filtered NFKC normalizer"));
        }
        let result = inputs
            .iter()
            .map(|input| {
                let source: Vec<u16> = input.encode_utf16().collect();
                let length = i32::try_from(source.len())
                    .map_err(|_| failed("normalization input is too long"))?;
                status = sys::UErrorCode::U_ZERO_ERROR;
                let needed = sys::versioned_function!(unorm2_normalize)(
                    normalizer,
                    source.as_ptr(),
                    length,
                    std::ptr::null_mut(),
                    0,
                    &mut status,
                );
                if needed < 0
                    || needed == i32::MAX
                    || (status as i32 > 0 && status != sys::UErrorCode::U_BUFFER_OVERFLOW_ERROR)
                {
                    return Err(failed("NFKC sizing failed"));
                }
                status = sys::UErrorCode::U_ZERO_ERROR;
                let mut output = vec![0u16; needed as usize + 1];
                let written = sys::versioned_function!(unorm2_normalize)(
                    normalizer,
                    source.as_ptr(),
                    length,
                    output.as_mut_ptr(),
                    output.len() as i32,
                    &mut status,
                );
                if status as i32 > 0 || written < 0 || written as usize >= output.len() {
                    return Err(failed("NFKC normalization failed"));
                }
                String::from_utf16(&output[..written as usize]).map_err(|e| failed(e.to_string()))
            })
            .collect();
        sys::versioned_function!(unorm2_close)(normalizer);
        uset_close_78(filter);
        result
    }
}

pub fn sort_names(names: Vec<String>) -> Result<Vec<String>, String> {
    sort_names_with_inventory(names, || {
        let identity = std::env::var_os("MINION_AGENT_ICU_IDENTITY")
            .ok_or_else(|| failed("MINION_AGENT_ICU_IDENTITY is not set"))?;
        Ok((loaded_modules()?, std::path::PathBuf::from(identity)))
    })
}

fn sort_names_with_inventory(
    names: Vec<String>,
    inventory: impl FnOnce() -> Result<(Vec<std::path::PathBuf>, std::path::PathBuf), String>,
) -> Result<Vec<String>, String> {
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
    let (modules, identity) = inventory()?;
    verify_identity(&modules, &identity)?;
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

    fn fixture() -> (
        tempfile::TempDir,
        Vec<std::path::PathBuf>,
        std::path::PathBuf,
    ) {
        let temp = tempfile::tempdir().unwrap();
        let mut lines = vec![format!("source-sha512 {SOURCE_SHA512}")];
        let mut modules = Vec::new();
        for role in ["icuuc", "icui18n", "icudata"] {
            let name = if cfg!(windows) {
                format!("{role}78.dll")
            } else {
                format!("lib{role}.so.78")
            };
            let path = temp.path().join(&name);
            fs::write(&path, role.as_bytes()).unwrap();
            let hash: String = Sha256::digest(role.as_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            lines.push(format!("library {role} {name} {hash}"));
            modules.push(path);
        }
        let identity = temp.path().join("identity.txt");
        fs::write(&identity, lines.join("\n")).unwrap();
        (temp, modules, identity)
    }

    #[test]
    fn sort_fails_closed_on_mismatching_identity() {
        let (_temp, modules, identity) = fixture();
        fs::write(&identity, "source-sha512 wrong").unwrap();
        let error =
            sort_names_with_inventory(vec!["b".into(), "a".into()], || Ok((modules, identity)))
                .unwrap_err();
        assert!(error.contains("pinned source tarball"), "{error}");
    }

    #[test]
    fn sort_checks_second_same_name_instance_and_accepts_identical_twin() {
        let (temp, mut modules, identity) = fixture();
        let twin_dir = temp.path().join("twin");
        fs::create_dir(&twin_dir).unwrap();
        let twin = twin_dir.join(modules[0].file_name().unwrap());
        fs::write(&twin, b"foreign").unwrap();
        modules.push(twin.clone());
        let error = sort_names_with_inventory(vec!["b".into(), "a".into()], || {
            Ok((modules.clone(), identity.clone()))
        })
        .unwrap_err();
        assert!(error.contains("does not match"), "{error}");

        fs::write(&twin, fs::read(&modules[0]).unwrap()).unwrap();
        assert_eq!(
            sort_names_with_inventory(vec!["b".into(), "a".into()], || { Ok((modules, identity)) })
                .unwrap(),
            ["a", "b"]
        );
    }

    #[cfg(windows)]
    #[test]
    fn failed_or_truncated_module_path_lookup_is_never_skipped() {
        assert!(
            module_path_with(|_| 0)
                .unwrap_err()
                .contains("complete path")
        );
        assert!(
            module_path_with(|buffer| buffer.len())
                .unwrap_err()
                .contains("complete path")
        );
        assert_eq!(
            module_path_with(|buffer| {
                buffer[0] = b'X' as u16;
                1
            })
            .unwrap(),
            std::path::PathBuf::from("X")
        );
    }

    #[cfg(windows)]
    fn synthetic_module_lookup(
        modules: &[std::path::PathBuf],
        handle: usize,
        buffer: &mut [u16],
        failed_length: usize,
    ) -> usize {
        use std::os::windows::ffi::OsStrExt;

        if handle == modules.len() {
            return failed_length;
        }
        let path: Vec<u16> = modules[handle].as_os_str().encode_wide().collect();
        buffer[..path.len()].copy_from_slice(&path);
        path.len()
    }

    #[cfg(windows)]
    #[test]
    fn failed_or_truncated_lookup_fails_the_whole_inventory() {
        let (_temp, modules, _identity) = fixture();
        for failed_length in [0, 32768] {
            let error = loaded_module_paths_with(0..=modules.len(), |handle, buffer| {
                synthetic_module_lookup(&modules, handle, buffer, failed_length)
            })
            .unwrap_err();
            assert!(error.contains("complete path"), "{error}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn failed_or_truncated_lookup_prevents_sort() {
        let (_temp, modules, identity) = fixture();
        for failed_length in [0, 32768] {
            let error = sort_names_with_inventory(vec!["b".into(), "a".into()], || {
                let paths = loaded_module_paths_with(0..=modules.len(), |handle, buffer| {
                    synthetic_module_lookup(&modules, handle, buffer, failed_length)
                })?;
                Ok((paths, identity.clone()))
            })
            .unwrap_err();
            assert!(error.contains("complete path"), "{error}");
        }
    }

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
