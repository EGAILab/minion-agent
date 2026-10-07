//! Search result text retains UTF-16 through line cutting and head truncation.
use super::{numeric::number_to_string, truncate::DEFAULT_MAX_BYTES, write::text_result};
use crate::llm::{ResultString, ResultValue};
use crate::tools::AgentToolResult;
use std::collections::BTreeMap;
pub(super) fn append(out: &mut Vec<u16>, text: &str) {
    out.extend(text.encode_utf16());
}
pub(super) fn cut_line(text: &str) -> (Vec<u16>, bool) {
    let mut units: Vec<_> = text.encode_utf16().collect();
    let truncated = units.len() > 500;
    if truncated {
        units.truncate(500);
        append(&mut units, "... [truncated]");
    }
    (units, truncated)
}
pub(super) fn finish(
    body: Vec<u16>,
    limit: Option<(&str, f64)>,
    lines_truncated: bool,
) -> AgentToolResult {
    let bytes = String::from_utf16_lossy(&body).len();
    let mut text = body.clone();
    let mut details = BTreeMap::new();
    let mut notices = Vec::new();
    if let Some((kind, n)) = limit {
        notices.push(format!(
            "{} {kind} limit reached. Use limit={} for more, or refine pattern",
            number_to_string(n),
            number_to_string(n * 2.0)
        ));
        details.insert(
            if kind == "results" {
                "resultLimitReached"
            } else {
                "matchLimitReached"
            }
            .into(),
            ResultValue::number(n),
        );
    }
    if bytes > DEFAULT_MAX_BYTES {
        let mut lines: Vec<&[u16]> = body.split(|u| *u == 10).collect();
        if body.last() == Some(&10) {
            lines.pop();
        }
        let mut kept = Vec::new();
        let mut used = 0;
        for line in &lines {
            let size = String::from_utf16_lossy(line).len() + usize::from(!kept.is_empty());
            if used + size > DEFAULT_MAX_BYTES {
                break;
            }
            kept.push(*line);
            used += size;
        }
        text.clear();
        for (i, line) in kept.iter().enumerate() {
            if i > 0 {
                text.push(10);
            }
            text.extend_from_slice(line);
        }
        let mut truncation:BTreeMap<ResultString,ResultValue>=serde_json::json!({"truncated":true,"truncatedBy":"bytes","totalLines":lines.len(),"totalBytes":bytes,"outputLines":kept.len(),"outputBytes":used,"lastLinePartial":false,"firstLineExceedsLimit":lines.first().is_some_and(|l|String::from_utf16_lossy(l).len()>DEFAULT_MAX_BYTES),"maxLines":9007199254740991u64,"maxBytes":DEFAULT_MAX_BYTES}).as_object().unwrap().iter().map(|(k,v)|(k.clone().into(),v.clone().into())).collect();
        truncation.insert(
            "content".into(),
            ResultValue::String(ResultString::from_code_units(text.clone())),
        );
        details.insert("truncation".into(), ResultValue::Object(truncation));
        notices.push("50.0KB limit reached".into());
    }
    if lines_truncated {
        notices.push("Some lines truncated to 500 chars. Use read tool to see full lines".into());
        details.insert("linesTruncated".into(), ResultValue::Bool(true));
    }
    if !notices.is_empty() {
        append(&mut text, &format!("\n\n[{}]", notices.join(". ")));
    }
    text_result(
        ResultString::from_code_units(text),
        ResultValue::Object(details),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn surrogate_cut_and_empty_body_do_not_lossily_serialize() {
        let (cut, truncated) = cut_line(&format!("{}😀", "x".repeat(499)));
        assert!(truncated);
        assert_eq!(cut[499], 0xd83d);
        assert_eq!(cut.len(), 500 + "... [truncated]".encode_utf16().count());
        assert_eq!(cut[500], u16::from(b'.'));
        let result = finish(cut, None, true);
        match &result.content[0] {
            crate::llm::ToolResultContentBlock::Text(t) => {
                assert_eq!(t.text.code_units()[499], 0xd83d)
            }
            _ => panic!(),
        }
    }
}
