//! Own-entry Windows attribute binding; deletion and retry policy live in the caller.

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
