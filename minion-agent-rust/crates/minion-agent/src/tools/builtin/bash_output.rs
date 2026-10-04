//! TOOL-035: one merged streaming UTF-8 decoder and pinned Pi's rolling tail.
use serde_json::{Value, json};

pub(super) const MAX_BYTES: usize = 51200;
pub(super) const MAX_LINES: usize = 2000;

#[derive(Default)]
pub(super) struct Output {
    pending: Vec<u8>,
    decoded_started: bool,
    tail: String,
    starts_at_boundary: bool,
    pub raw_bytes: usize,
    decoded_bytes: usize,
    completed_lines: usize,
    last_line_bytes: usize,
    open_line: bool,
    pub persisting: bool,
    raw_chunks: Vec<Vec<u8>>,
}

pub(super) struct Snapshot {
    pub text: String,
    pub details: Value,
}

impl Output {
    pub fn new() -> Self {
        Self {
            starts_at_boundary: true,
            ..Self::default()
        }
    }

    fn decode(&mut self, bytes: &[u8], finish: bool) {
        self.pending.extend_from_slice(bytes);
        let mut text = String::new();
        let mut start = 0;
        while start < self.pending.len() {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(valid) => {
                    text.push_str(valid);
                    start = self.pending.len();
                }
                Err(error) => {
                    let end = start + error.valid_up_to();
                    text.push_str(
                        std::str::from_utf8(&self.pending[start..end]).expect("valid prefix"),
                    );
                    start = end;
                    if let Some(length) = error.error_len() {
                        text.push('\u{fffd}');
                        start += length;
                    } else if finish {
                        text.push('\u{fffd}');
                        start = self.pending.len();
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..start);
        if !text.is_empty() && !self.decoded_started {
            self.decoded_started = true;
            if text.starts_with('\u{feff}') {
                text.remove(0);
            }
        }
        self.decoded_bytes += text.len();
        for piece in text.split_inclusive('\n') {
            if piece.ends_with('\n') {
                self.completed_lines += 1;
                self.last_line_bytes = 0;
                self.open_line = false;
            } else {
                self.last_line_bytes += piece.len();
                self.open_line = true;
            }
        }
        self.tail.push_str(&text);
        if self.tail.len() > 4 * MAX_BYTES {
            let mut start = self.tail.len() - 2 * MAX_BYTES;
            while !self.tail.is_char_boundary(start) {
                start += 1;
            }
            self.starts_at_boundary = self.tail.as_bytes()[start - 1] == b'\n';
            self.tail.drain(..start);
        }
    }

    fn lines(&self) -> usize {
        self.completed_lines + usize::from(self.open_line)
    }

    fn needs_file(&self) -> bool {
        self.raw_bytes > MAX_BYTES || self.decoded_bytes > MAX_BYTES || self.lines() > MAX_LINES
    }

    /// Return only newly accepted raw bytes destined for persistence, in order.
    pub fn append(&mut self, bytes: Vec<u8>) -> Vec<Vec<u8>> {
        self.raw_bytes += bytes.len();
        self.decode(&bytes, false);
        self.raw_chunks.push(bytes);
        if self.persisting || self.needs_file() {
            self.persisting = true;
            std::mem::take(&mut self.raw_chunks)
        } else {
            Vec::new()
        }
    }

    pub fn finish(&mut self) -> Vec<Vec<u8>> {
        self.decode(&[], true);
        if self.persisting || self.needs_file() {
            self.persisting = true;
            std::mem::take(&mut self.raw_chunks)
        } else {
            Vec::new()
        }
    }

    pub fn snapshot(&self, path: Option<&str>, empty: &str) -> Snapshot {
        let tail = if self.starts_at_boundary {
            self.tail.as_str()
        } else {
            self.tail
                .split_once('\n')
                .map_or(self.tail.as_str(), |(_, rest)| rest)
        };
        let (content, tail_by, partial, output_lines) = truncate_tail(tail);
        let truncated = self.lines() > MAX_LINES || self.decoded_bytes > MAX_BYTES;
        let by = tail_by.or(if truncated {
            Some(if self.decoded_bytes > MAX_BYTES {
                "bytes"
            } else {
                "lines"
            })
        } else {
            None
        });
        let mut text = if content.is_empty() {
            empty.to_owned()
        } else {
            content.clone()
        };
        let details = if truncated {
            let path = path.expect("truncated output persisted before formatting");
            let end = self.lines();
            let start = end as i128 - output_lines as i128 + 1;
            let notice = if partial {
                format!(
                    "Showing last {} of line {end} (line is {}). Full output: {path}",
                    format_size(content.len()),
                    format_size(self.last_line_bytes)
                )
            } else if by == Some("lines") {
                format!("Showing lines {start}-{end} of {end}. Full output: {path}")
            } else {
                format!("Showing lines {start}-{end} of {end} (50.0KB limit). Full output: {path}")
            };
            text.push_str(&format!("\n\n[{notice}]"));
            json!({"truncation": {
                "content":content, "truncated":true,"truncatedBy":by,
                "totalLines":end,"totalBytes":self.decoded_bytes,
                "outputLines":output_lines,"outputBytes":content.len(),
                "lastLinePartial":partial,"firstLineExceedsLimit":false,
                "maxLines":MAX_LINES,"maxBytes":MAX_BYTES
            },"fullOutputPath":path})
        } else {
            json!({})
        };
        Snapshot { text, details }
    }
}

fn truncate_tail(text: &str) -> (String, Option<&'static str>, bool, usize) {
    let mut lines: Vec<_> = if text.is_empty() {
        vec![]
    } else {
        text.split('\n').collect()
    };
    if text.ends_with('\n') {
        lines.pop();
    }
    if text.len() <= MAX_BYTES && lines.len() <= MAX_LINES {
        return (text.into(), None, false, lines.len());
    }
    let mut output = Vec::new();
    let mut size = 0;
    let mut by = "lines";
    let mut partial = false;
    for line in lines.into_iter().rev().take(MAX_LINES) {
        let amount = line.len() + usize::from(!output.is_empty());
        if size + amount > MAX_BYTES {
            by = "bytes";
            if output.is_empty() {
                let mut start = line.len().saturating_sub(MAX_BYTES);
                while !line.is_char_boundary(start) {
                    start += 1;
                }
                output.push(&line[start..]);
                partial = true;
            }
            break;
        }
        output.push(line);
        size += amount;
    }
    if output.len() >= MAX_LINES && size <= MAX_BYTES {
        by = "lines";
    }
    output.reverse();
    (output.join("\n"), Some(by), partial, output.len())
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        return format!("{bytes}B");
    }
    let (denominator, suffix) = if bytes < 1048576 {
        (1024, "KB")
    } else {
        (1048576, "MB")
    };
    // The division is by a power of two; exact halfway values use JS toFixed's
    // positive tie direction, not Rust formatting's ties-to-even.
    let tenths = (bytes as u128 * 10 + denominator / 2) / denominator;
    format!(
        "{}.{suffix_digit}{suffix}",
        tenths / 10,
        suffix_digit = tenths % 10
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_boundary_probe_all_ten_rolling_rows() {
        use sha2::{Digest, Sha256};
        let authority: Value =
            serde_json::from_str(include_str!("../../../tests/data/bash-boundary.json")).unwrap();
        let cases = [
            ("singleLineOneChunk", vec!["a".repeat(250000)]),
            ("singleLineManyChunks", vec!["a".repeat(50000); 5]),
            ("singleLineMultibyte", vec!["\u{20ac}".repeat(70000)]),
            (
                "cutMidLineNewlineLater",
                vec![format!("{}\n{}", "a".repeat(150000), "b".repeat(60000))],
            ),
            (
                "cutAfterNewline",
                vec![format!("{}\n{}", "a".repeat(110000), "b".repeat(102400))],
            ),
            (
                "cutMidLineNewlineInLaterChunk",
                vec!["a".repeat(250000), format!("\n{}", "b".repeat(1000))],
            ),
            ("atTriggerNoTrim", vec!["a".repeat(204800)]),
            (
                "longLineThenNewline",
                vec![format!("{}\n", "a".repeat(250000))],
            ),
            (
                "longLineThenTenLines",
                vec![format!("{}{}", "a".repeat(250000), "x\n".repeat(10))],
            ),
            (
                "longLineThen2001Lines",
                vec![format!("{}{}", "a".repeat(250000), "x\n".repeat(2001))],
            ),
        ];
        assert_eq!(authority["node"], "v22.15.1");
        for (name, chunks) in cases {
            let mut output = Output::new();
            for chunk in chunks {
                output.append(chunk.into_bytes());
            }
            output.finish();
            let mut truncation =
                output.snapshot(Some("file"), "(no output)").details["truncation"].clone();
            let content = truncation
                .as_object_mut()
                .unwrap()
                .remove("content")
                .unwrap();
            let content = content.as_str().unwrap();
            let expected = &authority["rolling"][name];
            assert_eq!(
                format!("{:x}", Sha256::digest(content.as_bytes())),
                expected["content"]["sha256"].as_str().unwrap(),
                "{name}"
            );
            assert_eq!(
                content.len(),
                expected["content"]["utf8Bytes"].as_u64().unwrap() as usize,
                "{name}"
            );
            assert_eq!(truncation, expected["truncation"], "{name}");
            assert_eq!(
                output.last_line_bytes,
                expected["lastLineBytes"].as_u64().unwrap() as usize,
                "{name}"
            );
        }
    }
    #[test]
    fn merged_decoder_bom_eof_and_raw_bytes() {
        for prefix in [vec![0xef], vec![0xef, 0xbb]] {
            let mut output = Output::new();
            output.append(prefix);
            output.finish();
            assert_eq!(output.snapshot(None, "").text, "\u{fffd}");
        }
        let mut output = Output::new();
        for bytes in [
            vec![0xef],
            vec![0xbb],
            vec![0xbf, 0xe2],
            vec![0x82],
            vec![0xac, 0xef, 0xbb, 0xbf],
        ] {
            output.append(bytes);
        }
        output.append(vec![0xef, 0xbb, 0xbf]);
        output.finish();
        assert_eq!(output.snapshot(None, "").text, "\u{20ac}\u{feff}\u{feff}");
        assert_eq!(output.raw_bytes, 12);
    }
    #[test]
    fn rolling_tail_and_total_projection() {
        let mut output = Output::new();
        let writes = output.append(vec![b'x'; 250000]);
        assert_eq!(writes.concat().len(), 250000);
        let snapshot = output.snapshot(Some("file"), "(no output)");
        assert_eq!(snapshot.details["truncation"]["outputBytes"], 51200);
        assert_eq!(snapshot.details["truncation"]["totalBytes"], 250000);
        assert_eq!(snapshot.details["truncation"]["lastLinePartial"], true);
        output.append(b"\n".to_vec());
        assert!(
            output
                .snapshot(Some("file"), "(no output)")
                .text
                .starts_with("(no output)\n\n[Showing lines 2-1")
        );
        assert_eq!(format_size(1280), "1.3KB");
    }
    #[test]
    fn raw_threshold_opens_a_log_even_when_bom_stripping_avoids_truncation() {
        let mut output = Output::new();
        let bytes = [vec![0xef, 0xbb, 0xbf], vec![b'a'; 51198]].concat();
        assert_eq!(output.append(bytes.clone()).concat(), bytes);
        output.finish();
        assert!(output.persisting);
        assert_eq!(output.snapshot(Some("file"), "").details, json!({}));
        assert_eq!(output.raw_bytes, 51201);
        assert_eq!(output.decoded_bytes, 51198);
    }
}
