//! Simultaneous matching and touched-line preservation from pinned edit-diff.ts.

use super::edit_text::fuzzy_normalize_batch;
use crate::tools::ToolCapabilityError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}
impl Edit {
    pub fn new(old_text: impl Into<String>, new_text: impl Into<String>) -> Self {
        Self {
            old_text: old_text.into(),
            new_text: new_text.into(),
        }
    }
}

pub(super) fn normalize_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[derive(Clone)]
struct Replacement {
    edit: usize,
    index: usize,
    length: usize,
    text: String,
}

fn find(
    content: &str,
    old: &str,
    fuzzy_content: &str,
    fuzzy_old: &str,
) -> Option<(usize, usize, bool)> {
    if let Some(index) = content.find(old) {
        return Some((index, old.len(), false));
    }
    fuzzy_content
        .find(fuzzy_old)
        .map(|index| (index, fuzzy_old.len(), true))
}

fn replace(content: &str, replacements: &[Replacement], offset: usize) -> String {
    let mut value = content.to_owned();
    for r in replacements.iter().rev() {
        let index = r.index - offset;
        value.replace_range(index..index + r.length, &r.text);
    }
    value
}

fn preserve(
    original: &str,
    base: &str,
    replacements: &[Replacement],
) -> Result<String, ToolCapabilityError> {
    let original_lines: Vec<_> = original.split_inclusive('\n').collect();
    let mut offset = 0;
    let spans: Vec<_> = base
        .split_inclusive('\n')
        .map(|line| {
            let start = offset;
            offset += line.len();
            (start, offset)
        })
        .collect();
    if original_lines.len() != spans.len() {
        return Err(ToolCapabilityError::new(
            "Cannot preserve unchanged lines because the base content has a different line count.",
        ));
    }
    let mut groups: Vec<(usize, usize, Vec<Replacement>)> = Vec::new();
    for r in replacements {
        let start = spans
            .iter()
            .position(|(s, e)| r.index >= *s && r.index < *e)
            .ok_or_else(|| {
                ToolCapabilityError::new("Replacement range is outside the base content.")
            })?;
        let end = spans
            .iter()
            .enumerate()
            .skip(start)
            .find(|(_, (_, e))| *e >= r.index + r.length)
            .map(|(i, _)| i + 1)
            .ok_or_else(|| {
                ToolCapabilityError::new("Replacement range is outside the base content.")
            })?;
        if let Some(group) = groups.last_mut()
            && start < group.1
        {
            group.1 = group.1.max(end);
            group.2.push(r.clone());
        } else {
            groups.push((start, end, vec![r.clone()]));
        }
    }
    let mut output = String::new();
    let mut original_line = 0;
    for (start, end, replacements) in groups {
        output.push_str(&original_lines[original_line..start].concat());
        let offset = spans[start].0;
        output.push_str(&replace(
            &base[offset..spans[end - 1].1],
            &replacements,
            offset,
        ));
        original_line = end;
    }
    output.push_str(&original_lines[original_line..].concat());
    Ok(output)
}

/// Match every edit against the original LF-normalized file, then replace disjoint regions.
/// Rust scalar-boundary offsets are a monotone mapping of the certified UTF-16 string
/// domain here; only the observable empty-separator count needs explicit code-unit lengths.
pub fn apply_edits(
    normalized: &str,
    edits: &[Edit],
    path: &str,
) -> Result<String, ToolCapabilityError> {
    apply_edits_with_base(normalized, edits, path).map(|(_, output)| output)
}

pub(super) fn apply_edits_with_base(
    normalized: &str,
    edits: &[Edit],
    path: &str,
) -> Result<(String, String), ToolCapabilityError> {
    let edits: Vec<_> = edits
        .iter()
        .map(|e| Edit::new(normalize_lf(&e.old_text), normalize_lf(&e.new_text)))
        .collect();
    let singular = edits.len() == 1;
    for (i, e) in edits.iter().enumerate() {
        if e.old_text.is_empty() {
            return Err(ToolCapabilityError::new(if singular {
                format!("oldText must not be empty in {path}.")
            } else {
                format!("edits[{i}].oldText must not be empty in {path}.")
            }));
        }
    }
    let mut inputs = vec![normalized];
    inputs.extend(edits.iter().map(|e| e.old_text.as_str()));
    let fuzzy = fuzzy_normalize_batch(&inputs)?;
    let mut used_fuzzy = false;
    for (i, e) in edits.iter().enumerate() {
        if find(normalized, &e.old_text, &fuzzy[0], &fuzzy[i + 1]).is_some_and(|m| m.2) {
            used_fuzzy = true;
        }
    }
    let base = if used_fuzzy {
        fuzzy[0].clone()
    } else {
        normalized.to_owned()
    };
    // NFKC and the subsequent quote/dash/space/line-end transforms are
    // idempotent: the fuzzy search base has the same fuzzy view as the original.
    let counted_base = &fuzzy[0];
    let mut replacements = Vec::new();
    for (i, e) in edits.iter().enumerate() {
        let (index,length,_) = find(&base, &e.old_text, counted_base, &fuzzy[i+1]).ok_or_else(|| ToolCapabilityError::new(if singular {
            format!("Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines.")
        } else { format!("Could not find edits[{i}] in {path}. The oldText must match exactly including all whitespace and newlines.") }))?;
        let old = &fuzzy[i + 1];
        let occurrences = if old.is_empty() {
            counted_base.encode_utf16().count() as isize - 1
        } else {
            counted_base.matches(old.as_str()).count() as isize
        };
        if occurrences > 1 {
            return Err(ToolCapabilityError::new(if singular {
                format!(
                    "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of edits[{i}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
                )
            }));
        }
        replacements.push(Replacement {
            edit: i,
            index,
            length,
            text: e.new_text.clone(),
        });
    }
    replacements.sort_by_key(|r| r.index);
    for pair in replacements.windows(2) {
        if pair[0].index + pair[0].length > pair[1].index {
            return Err(ToolCapabilityError::new(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                pair[0].edit, pair[1].edit
            )));
        }
    }
    let output = if used_fuzzy {
        preserve(normalized, &base, &replacements)?
    } else {
        replace(&base, &replacements, 0)
    };
    if output == normalized {
        return Err(ToolCapabilityError::new(if singular {
            format!(
                "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
            )
        } else {
            format!("No changes made to {path}. The replacements produced identical content.")
        }));
    }
    // Pi's display/patch base is the original normalized file, not its fuzzy
    // replacement-search view. Untouched whitespace must not appear as changes.
    Ok((normalized.to_owned(), output))
}
