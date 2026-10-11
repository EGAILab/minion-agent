//! Own-entry Windows attribute binding; deletion and retry policy live in the caller.

/// Native fixture handle matching the pinned authority's sharing/range-lock setup.
/// The caller must prove containment before opening; dropping releases the lock.
#[cfg(windows)]
pub fn fixture_hold(path: &std::path::Path, range: bool) -> std::io::Result<std::fs::File> {
    use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};
    use windows_sys::Win32::Storage::FileSystem::*;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .share_mode(if range {
            FILE_SHARE_READ | FILE_SHARE_WRITE
        } else {
            0
        })
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    if range {
        // SAFETY: a live owned handle; LockFile retains no pointer and the lock
        // is released by CloseHandle when the owned File is dropped.
        if unsafe { LockFile(file.as_raw_handle(), 0, 0, 64, 0) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(file)
}

#[cfg(windows)]
pub fn open_like_libuv(
    path: &std::path::Path,
    write: bool,
    append: bool,
) -> std::io::Result<std::fs::File> {
    use std::os::windows::{ffi::OsStrExt, io::FromRawHandle};
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Storage::FileSystem::*;
    let mut name = path.as_os_str().encode_wide().collect::<Vec<_>>();
    if name.contains(&0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "embedded null",
        ));
    }
    // Node ToNamespacedPath / Rust's normal Windows filesystem boundary. Keep
    // the call valid above MAX_PATH without changing its addressed spelling.
    if path.is_absolute() && !name.starts_with(&[92, 92, 63, 92]) {
        if name.starts_with(&[92, 92]) {
            name = [vec![92, 92, 63, 92, 85, 78, 67, 92], name[2..].to_vec()].concat();
        } else {
            name = [vec![92, 92, 63, 92], name].concat();
        }
    }
    name.push(0);
    let access = if append {
        (FILE_GENERIC_WRITE & !FILE_WRITE_DATA) | FILE_APPEND_DATA
    } else if write {
        FILE_GENERIC_WRITE
    } else {
        FILE_GENERIC_READ
    };
    let disposition = if append {
        OPEN_ALWAYS
    } else if write {
        CREATE_ALWAYS
    } else {
        OPEN_EXISTING
    };
    // SAFETY: terminated live UTF-16 buffer, null optional pointers; success
    // transfers the unique native handle to File, which closes it exactly once.
    let handle = unsafe {
        CreateFileW(
            name.as_ptr(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null(),
            disposition,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_BACKUP_SEMANTICS,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        let e = std::io::Error::last_os_error();
        if write && e.raw_os_error() == Some(80) {
            return Err(std::io::Error::from_raw_os_error(1));
        }
        return Err(e);
    }
    Ok(unsafe { std::fs::File::from_raw_handle(handle) })
}

#[cfg(windows)]
pub fn scandir(path: &std::path::Path) -> std::io::Result<Vec<std::ffi::OsString>> {
    use std::os::windows::{ffi::OsStringExt, fs::OpenOptionsExt, io::AsRawHandle};
    use windows_sys::Win32::Storage::FileSystem::*;
    let file = std::fs::OpenOptions::new()
        .access_mode(FILE_LIST_DIRECTORY | SYNCHRONIZE)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    #[repr(C)]
    struct IoStatusBlock {
        status_or_pointer: usize,
        information: usize,
    }
    #[repr(C, align(8))]
    struct Buffer([u8; 8192]);
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQueryDirectoryFile(
            handle: *mut std::ffi::c_void,
            event: *mut std::ffi::c_void,
            apc_routine: *mut std::ffi::c_void,
            apc_context: *mut std::ffi::c_void,
            iosb: *mut IoStatusBlock,
            information: *mut std::ffi::c_void,
            length: u32,
            class: u32,
            single: u8,
            pattern: *mut std::ffi::c_void,
            restart: u8,
        ) -> i32;
        fn RtlNtStatusToDosError(status: i32) -> u32;
    }
    let mut names = Vec::new();
    let mut restart = 1;
    loop {
        let mut buffer = Buffer([0; 8192]);
        let mut iosb = IoStatusBlock {
            status_or_pointer: 0,
            information: 0,
        };
        // SAFETY: synchronous owned directory handle, live aligned buffer and
        // IO_STATUS_BLOCK, no callbacks/pattern retained. Class 1 is the same
        // FILE_DIRECTORY_INFORMATION used by pinned libuv fs__scandir.
        let status = unsafe {
            NtQueryDirectoryFile(
                file.as_raw_handle(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut iosb,
                buffer.0.as_mut_ptr().cast(),
                buffer.0.len() as u32,
                1,
                0,
                std::ptr::null_mut(),
                restart,
            )
        };
        if restart == 1 && status as u32 == 0xc000000d {
            // libuv overrides the query's STATUS_INVALID_PARAMETER to ENOTDIR;
            // do not infer a type by stat or rewrite unrelated Win32 267.
            return Err(std::io::Error::from(std::io::ErrorKind::NotADirectory));
        }
        if status as u32 == 0x80000006 {
            return Ok(names);
        } // STATUS_NO_MORE_FILES
        if status < 0 || iosb.information == 0 {
            let status = if status == 0 {
                0x80000005u32 as i32
            } else {
                status
            };
            // SAFETY: pure native status conversion, no pointers/ownership.
            return Err(std::io::Error::from_raw_os_error(
                unsafe { RtlNtStatusToDosError(status) } as i32,
            ));
        }
        if iosb.information > buffer.0.len() {
            return Err(std::io::Error::other("invalid directory buffer size"));
        }
        let bytes = &buffer.0[..iosb.information];
        let mut offset = 0;
        loop {
            // FILE_DIRECTORY_INFORMATION: next offset at 0; name byte length
            // at 60; UTF-16 name at 64. Bounds are validated before each read.
            let header = bytes
                .get(offset..offset + 64)
                .ok_or_else(|| std::io::Error::other("invalid directory entry header"))?;
            let next = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let length = u32::from_le_bytes(header[60..64].try_into().unwrap()) as usize;
            if !length.is_multiple_of(2) {
                return Err(std::io::Error::other("invalid directory name length"));
            }
            let name = bytes
                .get(offset + 64..offset + 64 + length)
                .ok_or_else(|| std::io::Error::other("invalid directory name buffer"))?;
            let mut units = name
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect::<Vec<_>>();
            while units.last() == Some(&0) {
                units.pop();
            }
            if !units.is_empty() && units != [46] && units != [46, 46] {
                names.push(std::ffi::OsString::from_wide(&units));
            }
            if next == 0 {
                break;
            }
            if next < 64 || next > bytes.len() - offset {
                return Err(std::io::Error::other("invalid directory next offset"));
            }
            offset += next;
        }
        restart = 0;
    }
}

/// Equivalent to libuv 1.49.2 fs__unlink_rmdir. Own-entry open/access rights
/// precede type checks, and every handle is owned by std through all exits.
#[cfg(windows)]
pub fn delete_entry(path: &std::path::Path, directory: bool) -> std::io::Result<()> {
    use std::{
        io, mem,
        os::windows::{fs::OpenOptionsExt, io::AsRawHandle},
    };
    use windows_sys::Win32::Storage::FileSystem::*;
    let file = std::fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES | DELETE)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    let metadata = file.metadata()?;
    if directory && !metadata.is_dir() {
        return Err(io::Error::from_raw_os_error(267));
    }
    if !directory && metadata.is_dir() && !metadata.file_type().is_symlink() {
        return Err(io::Error::from_raw_os_error(5));
    }
    let disposition = FILE_DISPOSITION_INFO_EX {
        Flags: FILE_DISPOSITION_FLAG_DELETE
            | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS
            | FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE,
    };
    // Valid owned handle and correctly sized initialized buffer, no retained pointer.
    if unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FileDispositionInfoEx,
            (&disposition as *const FILE_DISPOSITION_INFO_EX).cast(),
            mem::size_of_val(&disposition) as u32,
        )
    } != 0
    {
        return Ok(());
    }
    let e = io::Error::last_os_error();
    if !matches!(e.raw_os_error(), Some(1 | 50 | 87)) {
        return Err(e);
    }
    // Older filesystem fallback uses the same addressed handle, never a followed target.
    use std::os::windows::fs::MetadataExt;
    if metadata.file_attributes() & FILE_ATTRIBUTE_READONLY != 0 {
        let mut info: FILE_BASIC_INFO = unsafe { mem::zeroed() };
        info.FileAttributes =
            (metadata.file_attributes() & !FILE_ATTRIBUTE_READONLY) | FILE_ATTRIBUTE_ARCHIVE;
        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileBasicInfo,
                (&info as *const FILE_BASIC_INFO).cast(),
                mem::size_of_val(&info) as u32,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    let disposition = FILE_DISPOSITION_INFO { DeleteFile: true };
    if unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FileDispositionInfo,
            (&disposition as *const FILE_DISPOSITION_INFO).cast(),
            mem::size_of_val(&disposition) as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
pub fn clear_readonly_entry(path: &std::path::Path) -> std::io::Result<bool> {
    use std::{
        io, mem,
        os::windows::{fs::OpenOptionsExt, io::AsRawHandle},
    };
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_READONLY, FILE_BASIC_INFO,
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_WRITE_ATTRIBUTES, FileBasicInfo,
        GetFileInformationByHandleEx, SetFileInformationByHandle,
    };
    // OPEN_REPARSE_POINT addresses the link itself; BACKUP_SEMANTICS permits directories.
    // std owns the handle (including error/unwind paths), validates NUL, and applies
    // its normal Windows long-path conversion before CreateFileW(OPEN_EXISTING).
    let handle = std::fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    let mut info: FILE_BASIC_INFO = unsafe { mem::zeroed() };
    // The buffer is initialized, correctly sized and valid for the duration of the call.
    if unsafe {
        GetFileInformationByHandleEx(
            handle.as_raw_handle(),
            FileBasicInfo,
            (&mut info as *mut FILE_BASIC_INFO).cast(),
            mem::size_of::<FILE_BASIC_INFO>() as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    if info.FileAttributes & FILE_ATTRIBUTE_READONLY == 0 {
        return Ok(false);
    }
    // Zero times mean unchanged; preserve every other attribute.
    info.CreationTime = 0;
    info.LastAccessTime = 0;
    info.LastWriteTime = 0;
    info.ChangeTime = 0;
    info.FileAttributes &= !FILE_ATTRIBUTE_READONLY;
    if info.FileAttributes == 0 {
        info.FileAttributes = FILE_ATTRIBUTE_NORMAL;
    }
    if unsafe {
        SetFileInformationByHandle(
            handle.as_raw_handle(),
            FileBasicInfo,
            (&info as *const FILE_BASIC_INFO).cast(),
            mem::size_of::<FILE_BASIC_INFO>() as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(true)
}
