//! Narrow native Windows comparison binding. No environment capture or policy lives here.

/// Windows environment lookup uppercases each UTF-16 unit through the live OS table.
#[cfg(windows)]
pub fn uppercase_unit(unit: u16) -> u16 {
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn RtlUpcaseUnicodeChar(unit: u16) -> u16;
    }
    // A value-only OS call: no pointers, handles or ownership cross this boundary.
    unsafe { RtlUpcaseUnicodeChar(unit) }
}
