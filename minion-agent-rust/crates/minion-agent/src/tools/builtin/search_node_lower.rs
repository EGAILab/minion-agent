//! Node 22.15.1 / Unicode 16 default lowercase, for win32.relative comparison.
//! The table is generated exhaustively from the pinned runtime, not the host
//! Rust Unicode version. Final_Sigma is the sole contextual default mapping.
use std::sync::OnceLock;

#[derive(serde::Deserialize)]
struct Table {
    lower: Vec<(u32, Vec<u32>)>,
    cased: Vec<(u32, u32)>,
    ignorable: Vec<(u32, u32)>,
}
fn in_ranges(value: u32, ranges: &[(u32, u32)]) -> bool {
    let index = ranges.partition_point(|&(_, end)| end < value);
    ranges
        .get(index)
        .is_some_and(|&(start, end)| start <= value && value <= end)
}
pub(super) fn lower(units: &[u16]) -> Vec<u16> {
    static TABLE: OnceLock<Table> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        serde_json::from_str(include_str!("search_node_lower.json"))
            .expect("pinned Node lowercase table")
    });
    let points: Vec<u32> = char::decode_utf16(units.iter().copied())
        .map(|value| value.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from))
        .collect();
    let mut result = Vec::with_capacity(units.len());
    for (index, &point) in points.iter().enumerate() {
        let final_sigma = point == 0x3a3
            && points[..index]
                .iter()
                .rev()
                .find(|&&c| !in_ranges(c, &table.ignorable))
                .is_some_and(|&c| in_ranges(c, &table.cased))
            && !points[index + 1..]
                .iter()
                .find(|&&c| !in_ranges(c, &table.ignorable))
                .is_some_and(|&c| in_ranges(c, &table.cased));
        let mapping = table
            .lower
            .binary_search_by_key(&point, |&(c, _)| c)
            .ok()
            .map(|i| table.lower[i].1.as_slice());
        let sigma = [0x3c2];
        let original = [point];
        for &mapped in if final_sigma {
            &sigma[..]
        } else {
            mapping.unwrap_or(&original)
        } {
            if let Some(c) = char::from_u32(mapped) {
                let mut buffer = [0; 2];
                result.extend_from_slice(c.encode_utf16(&mut buffer));
            } else {
                result.push(mapped as u16);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn node_unicode16_lowercase_is_not_host_unicode17() {
        let input = "\u{a7ce}\u{a7d2}\u{16ea0}ΟΣ ΟΣΑ ΟΣ\u{301} İ";
        let expected = "\u{a7ce}\u{a7d2}\u{16ea0}ος οσα ος\u{301} i\u{307}";
        assert_eq!(
            lower(&input.encode_utf16().collect::<Vec<_>>()),
            expected.encode_utf16().collect::<Vec<_>>()
        );
        assert_eq!(lower(&[65, 0xd800, 0x3a3]), [97, 0xd800, 0x3c3]);
    }
}
