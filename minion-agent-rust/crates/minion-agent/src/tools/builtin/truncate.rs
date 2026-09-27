//! Pinned Pi's shared head-truncation limits and result projection.

use serde_json::{Value, json};

pub(super) const DEFAULT_MAX_LINES: usize = 2000;
pub(super) const DEFAULT_MAX_BYTES: usize = 50 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Truncation {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub first_line_exceeds_limit: bool,
}

impl Truncation {
    pub fn details(&self) -> Value {
        json!({
            "truncated": self.truncated,
            "truncated_by": self.truncated_by.map(|kind| match kind {
                TruncatedBy::Lines => "lines",
                TruncatedBy::Bytes => "bytes",
            }),
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "first_line_exceeds_limit": self.first_line_exceeds_limit,
        })
    }
}

fn counted_lines(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<_> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

pub(super) fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let total_bytes = content.len();
    let lines = counted_lines(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && total_bytes <= max_bytes {
        return Truncation {
            content: content.to_owned(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            first_line_exceeds_limit: false,
        };
    }
    if lines.first().is_some_and(|line| line.len() > max_bytes) {
        return Truncation {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            first_line_exceeds_limit: true,
        };
    }
    let mut kept = Vec::new();
    let mut kept_bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    for (index, line) in lines.iter().enumerate() {
        if index >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(index > 0);
        if kept_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        kept.push(*line);
        kept_bytes += line_bytes;
    }
    Truncation {
        content: kept.join("\n"),
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: kept.len(),
        first_line_exceeds_limit: false,
    }
}

pub(super) fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        return format!("{bytes}B");
    }
    if bytes < 1024 * 1024 {
        return format!("{:.1}KB", bytes as f64 / 1024.0);
    }
    format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_truncation_keeps_only_complete_lines_and_selected_counts() {
        let result = truncate_head("aa\nbb\ncc", 2, DEFAULT_MAX_BYTES);
        assert_eq!(result.content, "aa\nbb");
        assert_eq!(result.total_lines, 3);
        assert_eq!(result.total_bytes, 8);
        assert_eq!(result.details()["truncated_by"], "lines");

        let result = truncate_head("ab\ncd", DEFAULT_MAX_LINES, 3);
        assert_eq!(result.content, "ab");
        assert_eq!(result.details()["truncated_by"], "bytes");
    }

    #[test]
    fn first_line_overflow_and_empty_shape() {
        let result = truncate_head("abcdef\nx", DEFAULT_MAX_LINES, 3);
        assert!(result.first_line_exceeds_limit);
        assert_eq!(result.content, "");
        assert_eq!(
            truncate_head("", DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES).total_lines,
            0
        );
        assert_eq!(format_size(DEFAULT_MAX_BYTES), "50.0KB");
    }
}
