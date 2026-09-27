//! R005-A image result path, using the exact pinned Photon WASM artifact.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;

use crate::{
    llm::{ImageBlock, TextBlock, ToolResultContentBlock},
    tools::{AgentToolResult, ToolCapabilityError},
};

use super::{photon::Photon, read::ReadToolOptions};

const MAX_BASE64_BYTES: usize = 4_718_592;
const NON_VISION_NOTE: &str =
    "[Current model does not support images. The image will be omitted from this request.]";

enum Processed {
    Ready {
        bytes: Vec<u8>,
        mime: &'static str,
        hints: Vec<String>,
    },
    Omitted(&'static str),
}

struct Resized {
    bytes: Vec<u8>,
    mime: &'static str,
    hint: Option<String>,
}

fn tiff_orientation(bytes: &[u8], start: usize) -> u8 {
    if start.checked_add(8).is_none_or(|end| end > bytes.len()) {
        return 1;
    }
    let little = bytes[start] == b'I' && bytes[start + 1] == b'I';
    let read16 = |pos: usize| -> Option<u16> {
        let two = bytes.get(pos..pos.checked_add(2)?)?;
        Some(if little {
            u16::from_le_bytes([two[0], two[1]])
        } else {
            u16::from_be_bytes([two[0], two[1]])
        })
    };
    let four = match bytes.get(start + 4..start + 8) {
        Some(four) => four,
        None => return 1,
    };
    let offset = if little {
        i32::from_le_bytes([four[0], four[1], four[2], four[3]]) as i64
    } else {
        u32::from_be_bytes([four[0], four[1], four[2], four[3]]) as i64
    };
    let ifd = start as i64 + offset;
    if ifd < 0 {
        return 1;
    }
    let ifd = ifd as usize;
    let Some(count) = read16(ifd) else {
        return 1;
    };
    for index in 0..count as usize {
        let Some(pos) = ifd
            .checked_add(2)
            .and_then(|value| value.checked_add(index * 12))
        else {
            return 1;
        };
        if pos.checked_add(12).is_none_or(|end| end > bytes.len()) {
            return 1;
        }
        if read16(pos) == Some(0x0112) {
            let value = read16(pos + 8).unwrap_or(1);
            return if (1..=8).contains(&value) {
                value as u8
            } else {
                1
            };
        }
    }
    1
}

fn orientation(bytes: &[u8]) -> u8 {
    if bytes.starts_with(&[0xff, 0xd8]) {
        let mut offset = 2usize;
        while offset < bytes.len().saturating_sub(1) {
            if bytes[offset] != 0xff {
                return 1;
            }
            if bytes[offset + 1] == 0xff {
                offset += 1;
                continue;
            }
            if bytes[offset + 1] == 0xe1 {
                if offset + 4 >= bytes.len() {
                    return 1;
                }
                let start = offset + 4;
                if bytes.get(start..start + 6) != Some(b"Exif\0\0") {
                    return 1;
                }
                return tiff_orientation(bytes, start + 6);
            }
            let Some(length) = bytes.get(offset + 2..offset + 4) else {
                return 1;
            };
            let length = u16::from_be_bytes([length[0], length[1]]) as usize;
            let Some(next) = offset.checked_add(2 + length) else {
                return 1;
            };
            offset = next;
        }
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        let mut offset = 12usize;
        while offset + 8 <= bytes.len() {
            let size = i32::from_le_bytes(
                bytes[offset + 4..offset + 8]
                    .try_into()
                    .expect("four bytes"),
            );
            let start = offset + 8;
            if &bytes[offset..offset + 4] == b"EXIF" {
                if size < 0
                    || start
                        .checked_add(size as usize)
                        .is_none_or(|end| end > bytes.len())
                {
                    return 1;
                }
                let tiff = if size >= 6 && bytes.get(start..start + 6) == Some(b"Exif\0\0") {
                    start + 6
                } else {
                    start
                };
                return tiff_orientation(bytes, tiff);
            }
            if size < 0 {
                return 1;
            }
            let Some(next) = start.checked_add(size as usize + size as usize % 2) else {
                return 1;
            };
            offset = next;
        }
    }
    1
}

fn orient(photon: &mut Photon, original: i32, bytes: &[u8]) -> Result<i32, String> {
    let which = orientation(bytes);
    match which {
        2 => photon.flip_horizontal(original)?,
        3 => {
            photon.flip_horizontal(original)?;
            photon.flip_vertical(original)?;
        }
        4 => photon.flip_vertical(original)?,
        5..=8 => {
            let width = photon.width(original)? as usize;
            let height = photon.height(original)? as usize;
            let src = photon.raw_pixels(original)?;
            let mut dst = vec![0u8; src.len()];
            for y in 0..height {
                for x in 0..width {
                    let source = (y * width + x) * 4;
                    let index = match which {
                        5 | 6 => x * height + (height - 1 - y),
                        7 | 8 => (width - 1 - x) * height + y,
                        _ => unreachable!(),
                    } * 4;
                    if source + 4 <= src.len() && index + 4 <= dst.len() {
                        dst[index..index + 4].copy_from_slice(&src[source..source + 4]);
                    }
                }
            }
            let rotated = photon.new_image(&dst, height as i32, width as i32)?;
            if matches!(which, 5 | 7) {
                photon.flip_horizontal(rotated)?;
            }
            photon.free(original);
            return Ok(rotated);
        }
        _ => {}
    }
    Ok(original)
}

fn resize(
    photon: &mut Photon,
    bytes: &[u8],
    mime: &'static str,
) -> Result<Option<Resized>, String> {
    let raw = photon.decode(bytes)?;
    let image = orient(photon, raw, bytes)?;
    let width = photon.width(image)?;
    let height = photon.height(image)?;
    if width <= 2000 && height <= 2000 && bytes.len().div_ceil(3) * 4 < MAX_BASE64_BYTES {
        photon.free(image);
        return Ok(Some(Resized {
            bytes: bytes.to_vec(),
            mime,
            hint: None,
        }));
    }
    let mut target_width = width;
    let mut target_height = height;
    if target_width > 2000 {
        target_height =
            ((target_height as f64 * 2000.0 / target_width as f64) + 0.5).floor() as i32;
        target_width = 2000;
    }
    if target_height > 2000 {
        target_width = ((target_width as f64 * 2000.0 / target_height as f64) + 0.5).floor() as i32;
        target_height = 2000;
    }
    loop {
        let resized = photon.resize(image, target_width, target_height)?;
        let candidates = (|| -> Result<Vec<(Vec<u8>, &'static str)>, String> {
            let mut candidates = vec![(photon.png(resized)?, "image/png")];
            for quality in [80, 85, 70, 55, 40] {
                candidates.push((photon.jpeg(resized, quality)?, "image/jpeg"));
            }
            Ok(candidates)
        })();
        photon.free(resized);
        for (candidate, candidate_mime) in candidates? {
            if candidate.len().div_ceil(3) * 4 < MAX_BASE64_BYTES {
                let scale = width as f64 / target_width as f64;
                // Pi uses Number::toFixed(2): exact positive half-cent ties round upward,
                // whereas Rust's formatter uses ties-to-even (1.125 -> 1.12).
                let scale_text = format!("{:.2}", ((scale * 100.0 + 0.5).floor()) / 100.0);
                let hint = format!(
                    "[Image: original {width}x{height}, displayed at {target_width}x{target_height}. Multiply coordinates by {scale_text} to map to original image.]"
                );
                photon.free(image);
                return Ok(Some(Resized {
                    bytes: candidate,
                    mime: candidate_mime,
                    hint: Some(hint),
                }));
            }
        }
        if target_width == 1 && target_height == 1 {
            break;
        }
        let next_width = if target_width == 1 {
            1
        } else {
            (target_width as f64 * 0.75).floor().max(1.0) as i32
        };
        let next_height = if target_height == 1 {
            1
        } else {
            (target_height as f64 * 0.75).floor().max(1.0) as i32
        };
        if next_width == target_width && next_height == target_height {
            break;
        }
        target_width = next_width;
        target_height = next_height;
    }
    photon.free(image);
    Ok(None)
}

fn process(bytes: &[u8], mime: &'static str, resize_enabled: bool) -> Processed {
    let mut normalized = bytes.to_vec();
    let mut final_mime = mime;
    let mut hints = Vec::new();
    let Ok(mut photon) = Photon::new() else {
        return Processed::Omitted(
            "[Image omitted: could not be resized below the inline image size limit.]",
        );
    };
    if mime == "image/bmp" {
        let converted = (|| -> Result<Vec<u8>, String> {
            let raw = photon.decode(bytes)?;
            let image = orient(&mut photon, raw, bytes)?;
            let png = photon.png(image)?;
            photon.free(image);
            Ok(png)
        })();
        let Ok(png) = converted else {
            return Processed::Omitted(
                "[Image omitted: could not be converted to a supported inline image format.]",
            );
        };
        normalized = png;
        final_mime = "image/png";
        hints.push("[Image converted from image/bmp to image/png.]".to_owned());
    }
    if resize_enabled {
        match resize(&mut photon, &normalized, final_mime) {
            Ok(Some(Resized {
                bytes: data,
                mime,
                hint,
            })) => {
                if let Some(hint) = hint {
                    hints.push(hint);
                }
                Processed::Ready {
                    bytes: data,
                    mime,
                    hints,
                }
            }
            _ => Processed::Omitted(
                "[Image omitted: could not be resized below the inline image size limit.]",
            ),
        }
    } else {
        Processed::Ready {
            bytes: normalized,
            mime: final_mime,
            hints,
        }
    }
}

pub(super) fn read_image(
    bytes: &[u8],
    mime: &'static str,
    options: &ReadToolOptions,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let processed = process(bytes, mime, options.auto_resize_images);
    let note = options
        .model_supports_images
        .as_ref()
        .and_then(|provider| provider())
        .is_some_and(|supported| !supported);
    let (mut text, image) = match processed {
        Processed::Ready { bytes, mime, hints } => {
            let mut text = format!("Read image file [{mime}]");
            for hint in hints {
                text.push('\n');
                text.push_str(&hint);
            }
            let encoded = STANDARD.encode(bytes);
            (text, Some(ImageBlock::data(mime, encoded)))
        }
        Processed::Omitted(message) => (format!("Read image file [{mime}]\n{message}"), None),
    };
    if note {
        text.push('\n');
        text.push_str(NON_VISION_NOTE);
    }
    let mut content = vec![ToolResultContentBlock::Text(TextBlock::new(text))];
    if let Some(image) = image {
        content.push(ToolResultContentBlock::Image(image));
    }
    Ok(AgentToolResult {
        content,
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn bmp_conversion_matches_pinned_photon_bytes() {
        let data =
            include_bytes!("../../../../../../conformance/agent/fixtures/r005a-photon/bmp_24.bmp");
        let Processed::Ready { bytes, mime, hints } = process(data, "image/bmp", false) else {
            panic!("BMP must convert");
        };
        assert_eq!(mime, "image/png");
        assert_eq!(hints, ["[Image converted from image/bmp to image/png.]"]);
        assert_eq!(bytes.len(), 926);
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            "54063b9f5c6cf330e48d6bb33d20986bf6257c0ef4a9187ebd16b8a56db8f15a"
        );
    }

    #[test]
    fn no_resize_fast_path_keeps_exact_input_bytes() {
        let data = include_bytes!(
            "../../../../../../conformance/agent/fixtures/r005a-photon/png_2001x40.png"
        );
        let Processed::Ready { bytes, mime, hints } = process(data, "image/png", false) else {
            panic!("PNG must pass through");
        };
        assert_eq!(mime, "image/png");
        assert!(hints.is_empty());
        assert_eq!(bytes, data);
    }

    #[test]
    fn real_read_image_serializes_standard_canonical_base64() {
        let data = include_bytes!(
            "../../../../../../conformance/agent/fixtures/r005a-photon/png_2001x40.png"
        );
        let outcome = read_image(
            data,
            "image/png",
            &ReadToolOptions {
                auto_resize_images: false,
                model_supports_images: None,
            },
        )
        .unwrap();
        let ToolResultContentBlock::Image(image) = &outcome.content[1] else {
            panic!("image block expected");
        };
        let serialized = serde_json::to_value(image).unwrap();
        let encoded = serialized["data"]
            .as_str()
            .expect("Layer-02 serializer exposes data");
        assert_eq!(STANDARD.decode(encoded).unwrap(), data);
        assert_eq!(STANDARD.encode(data), encoded);
    }

    #[test]
    fn bmp_conversion_then_resize_matches_pinned_photon_bytes() {
        let data = include_bytes!(
            "../../../../../../conformance/agent/fixtures/r005a-photon/bmp_2100x20.bmp"
        );
        let Processed::Ready { bytes, mime, hints } = process(data, "image/bmp", true) else {
            panic!("BMP must resize");
        };
        assert_eq!(mime, "image/png");
        assert_eq!(
            hints,
            [
                "[Image converted from image/bmp to image/png.]",
                "[Image: original 2100x20, displayed at 2000x19. Multiply coordinates by 1.05 to map to original image.]"
            ]
        );
        assert_eq!(bytes.len(), 54868);
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            "26d0520f76e95bd495ae5eed3c4fb33c7d4987dbd6cdc5f9f904807a523f3997"
        );
    }
}
