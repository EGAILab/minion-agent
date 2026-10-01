//! Simultaneous matching and touched-line preservation from pinned edit-diff.ts.

use super::edit_text::fuzzy_normalize_units;
use crate::{
    llm::ResultString,
    tools::{PreparedString, ToolCapabilityError},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Edit {
    pub old_text: PreparedString,
    pub new_text: PreparedString,
}
impl Edit {
    pub fn new(old_text: impl Into<PreparedString>, new_text: impl Into<PreparedString>) -> Self {
        Self {
            old_text: old_text.into(),
            new_text: new_text.into(),
        }
    }
}

pub(super) fn normalize_lf_units(text: &[u16]) -> Vec<u16> {
    let mut output = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if text[i] == 13 {
            output.push(10);
            i += 1 + usize::from(text.get(i + 1) == Some(&10));
        } else {
            output.push(text[i]);
            i += 1;
        }
    }
    output
}

fn index_of(text: &[u16], needle: &[u16]) -> Option<usize> {
    if needle.is_empty() {
        Some(0)
    } else {
        text.windows(needle.len()).position(|w| w == needle)
    }
}

#[derive(Clone)]
struct Replacement {
    edit: usize,
    index: usize,
    length: usize,
    text: Vec<u16>,
}

fn find(
    content: &[u16],
    old: &[u16],
    fuzzy_content: &[u16],
    fuzzy_old: &[u16],
) -> Option<(usize, usize, bool)> {
    if let Some(index) = index_of(content, old) {
        return Some((index, old.len(), false));
    }
    index_of(fuzzy_content, fuzzy_old).map(|index| (index, fuzzy_old.len(), true))
}

fn replace(content: &[u16], replacements: &[Replacement], offset: usize) -> Vec<u16> {
    let mut value = content.to_owned();
    for r in replacements.iter().rev() {
        let index = r.index - offset;
        value.splice(index..index + r.length, r.text.iter().copied());
    }
    value
}

fn preserve(
    original: &[u16],
    base: &[u16],
    replacements: &[Replacement],
) -> Result<Vec<u16>, ToolCapabilityError> {
    let original_lines: Vec<_> = original.split_inclusive(|u| *u == 10).collect();
    let mut offset = 0;
    let spans: Vec<_> = base
        .split_inclusive(|u| *u == 10)
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
    let mut output = Vec::new();
    let mut original_line = 0;
    for (start, end, replacements) in groups {
        output.extend_from_slice(&original_lines[original_line..start].concat());
        let offset = spans[start].0;
        output.extend_from_slice(&replace(
            &base[offset..spans[end - 1].1],
            &replacements,
            offset,
        ));
        original_line = end;
    }
    output.extend_from_slice(&original_lines[original_line..].concat());
    Ok(output)
}

/// Match every edit against the original LF-normalized file, then replace disjoint regions.
/// Scalar file-input convenience API with a lossless runtime result.
/// Matching itself always runs over UTF-16 code units.
pub fn apply_edits(
    normalized: &str,
    edits: &[Edit],
    path: &str,
) -> Result<ResultString, ToolCapabilityError> {
    apply_edits_with_base(&normalized.encode_utf16().collect::<Vec<_>>(), edits, path)
        .map(|(_, output)| ResultString::from_code_units(output))
}

pub(super) fn apply_edits_with_base(
    normalized: &[u16],
    edits: &[Edit],
    path: &str,
) -> Result<(Vec<u16>, Vec<u16>), ToolCapabilityError> {
    let edits: Vec<_> = edits
        .iter()
        .map(|e| {
            (
                normalize_lf_units(e.old_text.code_units()),
                normalize_lf_units(e.new_text.code_units()),
            )
        })
        .collect();
    let singular = edits.len() == 1;
    for (i, e) in edits.iter().enumerate() {
        if e.0.is_empty() {
            return Err(ToolCapabilityError::new(if singular {
                format!("oldText must not be empty in {path}.")
            } else {
                format!("edits[{i}].oldText must not be empty in {path}.")
            }));
        }
    }
    let mut inputs = vec![normalized];
    inputs.extend(edits.iter().map(|e| e.0.as_slice()));
    let fuzzy = fuzzy_normalize_units(&inputs)?;
    let mut used_fuzzy = false;
    for (i, e) in edits.iter().enumerate() {
        if find(normalized, &e.0, &fuzzy[0], &fuzzy[i + 1]).is_some_and(|m| m.2) {
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
        let (index,length,_) = find(&base, &e.0, counted_base, &fuzzy[i+1]).ok_or_else(|| ToolCapabilityError::new(if singular {
            format!("Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines.")
        } else { format!("Could not find edits[{i}] in {path}. The oldText must match exactly including all whitespace and newlines.") }))?;
        let old = &fuzzy[i + 1];
        let occurrences = if old.is_empty() {
            counted_base.len() as isize - 1
        } else {
            let mut count = 0;
            let mut remainder = counted_base.as_slice();
            while let Some(index) = index_of(remainder, old) {
                count += 1;
                remainder = &remainder[index + old.len()..];
            }
            count
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
            text: e.1.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_matching_can_target_one_half_of_an_astral_pair() {
        let output = apply_edits(
            "\u{1f600}",
            &[Edit::new(
                PreparedString::from_code_units(vec![0xd83d]),
                "X",
            )],
            "f",
        )
        .unwrap();
        assert_eq!(output.code_units(), &[88, 0xde00]);
        assert_eq!(String::from_utf16_lossy(output.code_units()), "X\u{fffd}");
    }
    #[test]
    fn unicode16_nfkc_preserves_lone_units_and_composes_valid_runs() {
        // Independently replayed on Node 22.15.1, String.normalize('NFKC').
        let inputs = [
            &[0xd800, 97, 0x301][..],
            &[0xdc00, 0x212b][..],
            &[0xd83d, 0xde00, 0xfb01][..],
        ];
        assert_eq!(
            fuzzy_normalize_units(&inputs).unwrap(),
            vec![
                vec![0xd800, 225],
                vec![0xdc00, 197],
                vec![0xd83d, 0xde00, 102, 105]
            ]
        );
    }
}
