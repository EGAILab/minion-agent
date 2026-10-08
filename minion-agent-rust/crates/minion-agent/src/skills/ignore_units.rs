//! The same pinned rewrite over UTF-16, including filesystem-origin surrogates.
//! Derived from ignore@7.0.5; its MIT notice is in IGNORE-LICENSE-MIT.txt.
use regress::{Match, Regex};

fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}
pub(super) fn points(units: &[u16]) -> impl Iterator<Item = u32> + Clone + '_ {
    char::decode_utf16(units.iter().copied())
        .map(|value| value.map_or_else(|error| u32::from(error.unpaired_surrogate()), u32::from))
}
fn group<'a>(s: &'a [u16], m: &Match, n: usize) -> &'a [u16] {
    m.group(n).map_or(&[], |r| &s[r])
}
fn concat(parts: &[&[u16]]) -> Vec<u16> {
    parts.concat()
}
fn replace(
    s: &[u16],
    pattern: &str,
    global: bool,
    f: impl Fn(&[u16], &Match) -> Vec<u16>,
) -> Vec<u16> {
    let re = Regex::new(pattern).expect("constant pinned rewrite");
    let mut out = Vec::new();
    let mut cursor = 0;
    for m in re.find_from_utf16(s, 0) {
        out.extend_from_slice(&s[cursor..m.range.start]);
        out.extend(f(s, &m));
        cursor = m.range.end;
        if !global {
            break;
        }
    }
    out.extend_from_slice(&s[cursor..]);
    out
}
pub(super) fn expression(body: &[u16]) -> Vec<u16> {
    let mut s = body.strip_prefix(&[0xfeff]).unwrap_or(body).to_vec();
    s = replace(&s, r"((?:\\\\)*?)(\\?\s+)$", false, |s, m| {
        concat(&[
            group(s, m, 1),
            if group(s, m, 2).starts_with(&[92]) {
                &[32]
            } else {
                &[]
            },
        ])
    });
    s = replace(&s, r"(\\+?)\s", true, |s, m| {
        let g = group(s, m, 1);
        concat(&[&g[..g.len() - g.len() % 2], &[32]])
    });
    s = replace(&s, r"[\\$.|*+(){^]", true, |s, m| {
        concat(&[&[92], group(s, m, 0)])
    });
    s = replace(&s, r"(?!\\)\?", true, |_, _| units("[^/]"));
    s = replace(&s, r"^/", false, |_, _| units("^"));
    s = replace(&s, r"/", true, |_, _| units(r"\/"));
    s = replace(&s, r"^\^*\\\*\\\*\\/", false, |_, _| units(r"^(?:.*\/)?"));
    s = replace(&s, r"^(?=[^^])", false, |_, _| {
        if Regex::new(r"/(?!$)")
            .expect("constant rewrite")
            .find_from_utf16(body, 0)
            .next()
            .is_some()
        {
            units("^")
        } else {
            units(r"(?:^|\/)")
        }
    });
    s = replace(&s, r"\\/\\\*\\\*(?=\\/|$)", true, |s, m| {
        if m.range.end < s.len() {
            units(r"(?:\/[^\/]+)*")
        } else {
            units(r"\/.+")
        }
    });
    s = replace(&s, r"(^|[^\\]+)(\\\*)+(?=.+)", true, |s, m| {
        concat(&[
            group(s, m, 1),
            &replace(group(s, m, 2), r"\\\*", true, |_, _| units(r"[^\/]*")),
        ])
    });
    s = replace(&s, r"\\\\\\(?=[$.|*+(){^])", true, |_, _| vec![92]);
    s = replace(&s, r"\\\\", true, |_, _| vec![92]);
    s = replace(&s, r"(\\)?\[([^\]/]*?)(\\*)($|\])", true, |s, m| {
        let lead = group(s, m, 1);
        let range = group(s, m, 2);
        let end = group(s, m, 3);
        let close = group(s, m, 4);
        if lead == [92] {
            concat(&[&[92, 91], range, &end[..end.len() - end.len() % 2], close])
        } else if close == [93] && end.len().is_multiple_of(2) {
            let range = replace(range, r"([0-z])-([0-z])", true, |s, m| {
                if group(s, m, 1) <= group(s, m, 2) {
                    group(s, m, 0).to_vec()
                } else {
                    Vec::new()
                }
            });
            concat(&[&[91], &range, end, &[93]])
        } else {
            units("[]")
        }
    });
    s = replace(&s, r"(?:[^*])$", false, |s, m| {
        let g = group(s, m, 0);
        concat(&[
            g,
            &units(if g.ends_with(&[47]) {
                "$"
            } else {
                r"(?=$|\/$)"
            }),
        ])
    });
    replace(&s, r"(^|\\/)?\\\*$", false, |s, m| {
        concat(&[
            group(s, m, 1),
            &units(if m.group(1).is_some() {
                "[^/]+"
            } else {
                "[^/]*"
            }),
            &units(r"(?=$|\/$)"),
        ])
    })
}

/// Annex B permits incomplete hex escapes as identity escapes. Normalize those
/// before asking regress to validate the otherwise unchanged pattern.
pub(super) fn legacy_hex(input: &[u16]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let c = input[i];
        i += 1;
        if c != 92 || i == input.len() {
            out.push(c);
            continue;
        }
        let next = input[i];
        i += 1;
        let width = if next == 120 {
            2
        } else if next == 117 {
            4
        } else {
            0
        };
        if width > 0
            && !(input.get(i..i + width).is_some_and(|digits| {
                digits
                    .iter()
                    .all(|&u| u <= 127 && (u as u8).is_ascii_hexdigit())
            }))
        {
            out.push(next);
            continue;
        }
        out.extend([c, next]);
    }
    out
}
fn hex(u: u16) -> Option<u16> {
    char::from_u32(u32::from(u))
        .and_then(|c| c.to_digit(16))
        .map(|d| d as u16)
}
pub(super) fn canonical(input: &[u16], table: &[u16]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let c = input[i];
        i += 1;
        if c == 91 {
            let negative = input.get(i) == Some(&94);
            if negative {
                i += 1;
            }
            let mut body = Vec::new();
            while i < input.len() {
                let c = input[i];
                i += 1;
                if c == 93 {
                    break;
                }
                body.push(c);
                if c == 92 && i < input.len() {
                    body.push(input[i]);
                    i += 1;
                }
            }
            let pattern = concat(&[&units("^["), &body, &units("]$")]);
            let class = Regex::from_unicode(points(&pattern), regress::Flags::default())
                .expect("original class validated");
            let mut extra = vec![false; 65536];
            for unit in 0..=u16::MAX {
                let canonical = table[usize::from(unit)];
                if canonical != unit && class.find_from_utf16(&[unit], 0).next().is_some() {
                    extra[usize::from(canonical)] = true;
                }
            }
            out.push(91);
            if negative {
                out.push(94);
            }
            out.extend(body);
            let mut index = 0;
            while index < extra.len() {
                if !extra[index] {
                    index += 1;
                    continue;
                }
                let start = index;
                while index + 1 < extra.len() && extra[index + 1] {
                    index += 1;
                }
                out.extend(units(&format!("\\u{start:04x}")));
                if index > start {
                    out.extend(units(&format!("-\\u{index:04x}")));
                }
                index += 1;
            }
            out.push(93);
            continue;
        }
        if c == 92 && i < input.len() {
            let next = input[i];
            i += 1;
            let width = if next == 117 {
                4
            } else if next == 120 {
                2
            } else {
                0
            };
            if width > 0
                && input
                    .get(i..i + width)
                    .is_some_and(|digits| digits.iter().all(|&u| hex(u).is_some()))
            {
                let unit = input[i..i + width]
                    .iter()
                    .fold(0, |v, &u| v * 16 + hex(u).expect("validated hex"));
                i += width;
                out.extend(units(&format!("\\u{:04x}", table[usize::from(unit)])));
                continue;
            }
            if (48..=55).contains(&next) {
                let mut unit = next - 48;
                for _ in 0..if next <= 51 { 2 } else { 1 } {
                    if input.get(i).is_some_and(|u| (48..=55).contains(u)) {
                        unit = unit * 8 + input[i] - 48;
                        i += 1;
                    } else {
                        break;
                    }
                }
                out.extend(units(&format!("\\u{:04x}", table[usize::from(unit)])));
                continue;
            }
            out.push(92);
            if "dDsSwWbBtvrnfuxc".encode_utf16().any(|u| u == next) {
                out.push(next);
            } else {
                out.push(table[usize::from(next)]);
            }
        } else {
            out.push(table[usize::from(c)]);
        }
    }
    out
}
