//! TOOL-031 text normalization; native trim/normalization is deliberately not used.
use crate::tools::ToolCapabilityError;

pub fn fuzzy_normalize(text: &str) -> Result<String, ToolCapabilityError> {
    let normalized =
        minion_agent_pinned_icu::nfkc_unicode16(text).map_err(ToolCapabilityError::new)?;
    Ok(finish(&normalized))
}

pub(super) fn fuzzy_normalize_units(
    inputs: &[&[u16]],
) -> Result<Vec<Vec<u16>>, ToolCapabilityError> {
    minion_agent_pinned_icu::nfkc_unicode16_units_batch(inputs)
        .map(|values| values.iter().map(|value| finish_units(value)).collect())
        .map_err(ToolCapabilityError::new)
}

fn finish_units(normalized: &[u16]) -> Vec<u16> {
    let mut output = Vec::new();
    for (i, line) in normalized.split(|unit| *unit == 10).enumerate() {
        if i > 0 {
            output.push(10);
        }
        let end = line
            .iter()
            .rposition(|unit| char::from_u32(u32::from(*unit)).is_none_or(|c| !js_whitespace(c)))
            .map_or(0, |i| i + 1);
        output.extend(line[..end].iter().map(|unit| match *unit {
            0x2018..=0x201b => 39,
            0x201c..=0x201f => 34,
            0x2010..=0x2015 | 0x2212 => 45,
            0xa0 | 0x2002..=0x200a | 0x202f | 0x205f | 0x3000 => 32,
            unit => unit,
        }));
    }
    output
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
