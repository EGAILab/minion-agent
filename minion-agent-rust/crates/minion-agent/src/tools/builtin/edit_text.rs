//! TOOL-031 text normalization; native trim/normalization is deliberately not used.
use crate::tools::ToolCapabilityError;

pub fn fuzzy_normalize(text: &str) -> Result<String, ToolCapabilityError> {
    let normalized =
        minion_agent_pinned_icu::nfkc_unicode16(text).map_err(ToolCapabilityError::new)?;
    Ok(finish(&normalized))
}

pub(super) fn fuzzy_normalize_batch(inputs: &[&str]) -> Result<Vec<String>, ToolCapabilityError> {
    minion_agent_pinned_icu::nfkc_unicode16_batch(inputs)
        .map(|values| values.iter().map(|value| finish(value)).collect())
        .map_err(ToolCapabilityError::new)
}

fn finish(normalized: &str) -> String {
    normalized
        .split('\n')
        .map(|line| line.trim_end_matches(js_whitespace))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .map(|c| match c {
            '\u{2018}'..='\u{201b}' => '\'',
            '\u{201c}'..='\u{201f}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            c => c,
        })
        .collect()
}

fn js_whitespace(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | '\u{20}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')
}
