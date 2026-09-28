//! The text branch of pinned Pi `read.ts`, including JS-number slice behavior.

use serde_json::{Value, json};

use crate::tools::ToolCapabilityError;

use super::{
    numeric::{js_max, js_min, js_slice, number_to_string},
    truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head},
};

const UNDEFINED_LINE_ERROR: &str = "The \"string\" argument must be of type string or an instance of Buffer or ArrayBuffer. Received undefined";

pub(super) fn read_text(
    data: &[u8],
    path: &str,
    offset: Option<f64>,
    limit: Option<f64>,
) -> Result<(String, Value), ToolCapabilityError> {
    let decoded = String::from_utf8_lossy(data);
    let all_lines: Vec<&str> = decoded.split('\n').collect();
    let total_file_lines = all_lines.len();
    let start = match offset {
        Some(n) if n != 0.0 && !n.is_nan() => js_max(0.0, n - 1.0),
        _ => 0.0,
    };
    let start_display = start + 1.0;
    if start >= total_file_lines as f64 {
        return Err(ToolCapabilityError::new(format!(
            "Offset {} is beyond end of file ({} lines total)",
            number_to_string(offset.unwrap_or(0.0)),
            total_file_lines
        )));
    }
    let (selected, user_limited) = if let Some(user_limit) = limit {
        let end = js_min(start + user_limit, total_file_lines as f64);
        (
            js_slice(&all_lines, start, Some(end)).join("\n"),
            Some(end - start),
        )
    } else {
        (js_slice(&all_lines, start, None).join("\n"), None)
    };
    let truncation = truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let details = if truncation.truncated {
        json!({"truncation": truncation.details()})
    } else {
        json!({})
    };
    if truncation.first_line_exceeds_limit {
        if start.fract() != 0.0 {
            return Err(ToolCapabilityError::new(UNDEFINED_LINE_ERROR));
        }
        let first = all_lines
            .get(start as usize)
            .ok_or_else(|| ToolCapabilityError::new(UNDEFINED_LINE_ERROR))?;
        let display = number_to_string(start_display);
        let text = format!(
            "[Line {display} is {}, exceeds {} limit. Use bash: sed -n '{display}p' {path} | head -c {DEFAULT_MAX_BYTES}]",
            format_size(first.len()),
            format_size(DEFAULT_MAX_BYTES)
        );
        return Ok((text, details));
    }
    if truncation.truncated {
        let end_display = start_display + truncation.output_lines as f64 - 1.0;
        let next_offset = end_display + 1.0;
        let shown = format!(
            "lines {}-{} of {total_file_lines}",
            number_to_string(start_display),
            number_to_string(end_display)
        );
        let notice = if truncation.details()["truncated_by"] == "lines" {
            format!(
                "[Showing {shown}. Use offset={} to continue.]",
                number_to_string(next_offset)
            )
        } else {
            format!(
                "[Showing {shown} ({} limit). Use offset={} to continue.]",
                format_size(DEFAULT_MAX_BYTES),
                number_to_string(next_offset)
            )
        };
        return Ok((format!("{}\n\n{notice}", truncation.content), details));
    }
    if let Some(selected_count) = user_limited
        && start + selected_count < total_file_lines as f64
    {
        let remaining = total_file_lines as f64 - (start + selected_count);
        let next_offset = start + selected_count + 1.0;
        return Ok((
            format!(
                "{}\n\n[{} more lines in file. Use offset={} to continue.]",
                truncation.content,
                number_to_string(remaining),
                number_to_string(next_offset)
            ),
            details,
        ));
    }
    Ok((truncation.content, details))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_limit_notice_does_not_fabricate_truncation_details() {
        let (text, details) = read_text(b"one\ntwo\nthree", "x", None, Some(2.0)).unwrap();
        assert_eq!(
            text,
            "one\ntwo\n\n[1 more lines in file. Use offset=3 to continue.]"
        );
        assert_eq!(details, json!({}));
    }

    #[test]
    fn fractional_and_negative_js_slice_domain() {
        let (text, _) = read_text(b"1\n2\n3\n4", "x", None, Some(-1.0)).unwrap();
        assert!(text.starts_with("1\n2\n3"));
        let (text, _) = read_text(b"1\n2\n3\n4", "x", Some(2.5), Some(-1.0)).unwrap();
        assert!(text.contains("more lines in file"));
    }
}
