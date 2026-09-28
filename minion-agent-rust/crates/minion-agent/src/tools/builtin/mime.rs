//! The closed image sniff set from pinned Pi `utils/mime.ts` (first 4,100 bytes only).

fn byte(input: &[u8], index: usize) -> u32 {
    input.get(index).copied().unwrap_or(0).into()
}

fn be32(input: &[u8], index: usize) -> u32 {
    (byte(input, index) << 24)
        | (byte(input, index + 1) << 16)
        | (byte(input, index + 2) << 8)
        | byte(input, index + 3)
}

fn le32(input: &[u8], index: usize) -> u32 {
    byte(input, index)
        | (byte(input, index + 1) << 8)
        | (byte(input, index + 2) << 16)
        | (byte(input, index + 3) << 24)
}

fn le16(input: &[u8], index: usize) -> u32 {
    byte(input, index) | (byte(input, index + 1) << 8)
}

fn animated_png(input: &[u8]) -> bool {
    let mut offset = 8usize;
    while offset + 8 <= input.len() {
        let len = be32(input, offset) as usize;
        let kind = &input[offset + 4..offset + 8];
        if kind == b"acTL" {
            return true;
        }
        if kind == b"IDAT" {
            return false;
        }
        let Some(next) = offset
            .checked_add(8)
            .and_then(|n| n.checked_add(len))
            .and_then(|n| n.checked_add(4))
        else {
            return false;
        };
        if next <= offset || next > input.len() {
            return false;
        }
        offset = next;
    }
    false
}

fn valid_bmp(input: &[u8]) -> bool {
    if input.len() < 26 {
        return false;
    }
    let size = le32(input, 2);
    let pixel_offset = le32(input, 10);
    let dib = le32(input, 14);
    if (size != 0 && size < 26)
        || u64::from(pixel_offset) < 14 + u64::from(dib)
        || (size != 0 && pixel_offset >= size)
    {
        return false;
    }
    let (planes, bpp) = match dib {
        12 => (le16(input, 22), le16(input, 24)),
        40..=124 if input.len() >= 30 => (le16(input, 26), le16(input, 28)),
        _ => return false,
    };
    planes == 1 && [1, 4, 8, 16, 24, 32].contains(&bpp)
}

pub(super) fn detect_supported_image_mime_type(data: &[u8]) -> Option<&'static str> {
    let input = &data[..data.len().min(4100)];
    if input.starts_with(&[0xff, 0xd8, 0xff]) {
        return (byte(input, 3) != 0xf7).then_some("image/jpeg");
    }
    if input.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        return (input.len() >= 16
            && be32(input, 8) == 13
            && &input[12..16] == b"IHDR"
            && !animated_png(input))
        .then_some("image/png");
    }
    if input.starts_with(b"GIF") {
        return Some("image/gif");
    }
    if input.starts_with(b"RIFF") && input.get(8..12) == Some(&b"WEBP"[..]) {
        return Some("image/webp");
    }
    if input.starts_with(b"BM") && valid_bmp(input) {
        return Some("image/bmp");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_is_content_not_extension() {
        assert_eq!(
            detect_supported_image_mime_type(b"GIF89a"),
            Some("image/gif")
        );
        assert_eq!(detect_supported_image_mime_type(b"\xff\xd8\xff\xf7"), None);
        assert_eq!(
            detect_supported_image_mime_type(b"RIFFxxxxWEBP"),
            Some("image/webp")
        );
        assert_eq!(detect_supported_image_mime_type(b"BM"), None);
    }

    #[test]
    fn apng_animation_control_before_idat_is_not_a_supported_still_image() {
        let png = include_bytes!(
            "../../../../../../conformance/agent/fixtures/r005a-photon/png_small_rgb.png"
        );
        assert_eq!(detect_supported_image_mime_type(png), Some("image/png"));
        let mut apng = png[..33].to_vec(); // signature + complete IHDR chunk
        apng.extend_from_slice(&8u32.to_be_bytes());
        apng.extend_from_slice(b"acTL");
        apng.extend_from_slice(&1u32.to_be_bytes()); // frame count
        apng.extend_from_slice(&0u32.to_be_bytes()); // play count
        apng.extend_from_slice(&0u32.to_be_bytes()); // CRC is irrelevant to MIME sniff
        apng.extend_from_slice(&png[33..]);
        assert_eq!(detect_supported_image_mime_type(&apng), None);
    }
}
