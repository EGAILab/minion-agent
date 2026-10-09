//! Private DIV-004 subset reader, not a general/public YAML authority.
//! Rules follow spec/harness.md HAR-010, including DIV-007's collection bound.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Mapping(Vec<(String, Value)>),
    Sequence(Vec<Value>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Invalid;
type ReadResult<T> = Result<T, Invalid>;
const MAX_DEPTH: usize = 64;

fn forbidden(c: char) -> bool {
    matches!(c as u32, 0..=8 | 11..=31 | 127..=159 | 0x2028 | 0x2029 | 0xfeff)
}

fn indent(line: &str) -> usize {
    line.bytes().take_while(|&c| c == b' ').count()
}
fn trim(line: &str) -> &str {
    line.trim_matches([' ', '\t'])
}
fn blank(line: &str) -> bool {
    trim(line).is_empty()
}
fn comment(line: &str) -> bool {
    trim(line).starts_with('#')
}
fn entry(line: &str) -> Option<(&str, &str)> {
    let (key, rest) = line.split_once(':')?;
    let mut bytes = key.bytes();
    if !bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        || !bytes.all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
        || matches!(
            key,
            "null" | "Null" | "NULL" | "true" | "True" | "TRUE" | "false" | "False" | "FALSE"
        )
        || !(rest.is_empty() || rest.starts_with(' '))
    {
        return None;
    }
    Some((key, rest.trim_start_matches([' ', '\t'])))
}

pub(super) fn read(input: &str) -> ReadResult<Value> {
    if input.chars().any(forbidden) {
        return Err(Invalid);
    }
    let mut lines: Vec<_> = input.split('\n').collect();
    if input.ends_with('\n') && lines.len() > 1 {
        lines.pop();
    }
    if lines
        .iter()
        .any(|line| line.as_bytes().get(indent(line)) == Some(&b'\t'))
    {
        return Err(Invalid);
    }
    let mut reader = Reader {
        lines,
        pos: 0,
        terminated: input.ends_with('\n'),
    };
    reader.skip();
    if reader.pos == reader.lines.len() {
        return Ok(Value::Mapping(Vec::new()));
    }
    if indent(reader.lines[reader.pos]) != 0 {
        return Err(Invalid);
    }
    let value = reader.mapping(0, 1)?;
    reader.skip();
    if reader.pos != reader.lines.len() {
        return Err(Invalid);
    }
    Ok(value)
}

struct Reader<'a> {
    lines: Vec<&'a str>,
    pos: usize,
    terminated: bool,
}
impl Reader<'_> {
    fn skip(&mut self) {
        while self.pos < self.lines.len()
            && (blank(self.lines[self.pos]) || comment(self.lines[self.pos]))
        {
            self.pos += 1;
        }
    }
    fn mapping(&mut self, n: usize, depth: usize) -> ReadResult<Value> {
        if depth > MAX_DEPTH {
            return Err(Invalid);
        }
        let mut values = Vec::new();
        let mut keys = BTreeMap::new();
        loop {
            self.skip();
            if self.pos == self.lines.len() || indent(self.lines[self.pos]) < n {
                break;
            }
            let line = self.lines[self.pos];
            if indent(line) != n {
                return Err(Invalid);
            }
            let (key, text) = entry(&line[n..]).ok_or(Invalid)?;
            if keys.insert(key, ()).is_some() {
                return Err(Invalid);
            }
            self.pos += 1;
            let value = if text.is_empty() || text.starts_with('#') {
                self.skip();
                if let Some(next) = self.lines.get(self.pos).copied() {
                    let m = indent(next);
                    if m >= n && next[m..].starts_with("- ") {
                        self.sequence(m, depth + 1)?
                    } else if m > n {
                        if entry(&next[m..]).is_none() {
                            return Err(Invalid);
                        }
                        self.mapping(m, depth + 1)?
                    } else {
                        Value::Null
                    }
                } else {
                    Value::Null
                }
            } else if text.starts_with(['|', '>']) {
                Value::String(self.block(text, n)?)
            } else if text.starts_with(['\'', '"']) {
                Value::String(quoted(text)?)
            } else {
                let (first, has_comment) = plain(text, true)?;
                let mut parts = vec![first];
                let mut pending = 0;
                if !has_comment {
                    while self.pos < self.lines.len() {
                        let next = self.lines[self.pos];
                        if blank(next) {
                            pending += 1;
                            self.pos += 1;
                            continue;
                        }
                        let m = indent(next);
                        if m <= n {
                            break;
                        }
                        if comment(next) {
                            return Err(Invalid);
                        }
                        let (part, has_comment) = plain(&next[m..], false)?;
                        if has_comment {
                            return Err(Invalid);
                        }
                        if pending > 0 {
                            parts.push("\n".repeat(pending));
                        } else {
                            parts.push(" ".into());
                        }
                        parts.push(part);
                        pending = 0;
                        self.pos += 1;
                    }
                }
                resolve(parts.concat())
            };
            values.push((key.to_owned(), value));
        }
        Ok(Value::Mapping(values))
    }
    fn sequence(&mut self, n: usize, depth: usize) -> ReadResult<Value> {
        if depth > MAX_DEPTH {
            return Err(Invalid);
        }
        let mut values = Vec::new();
        loop {
            self.skip();
            let Some(line) = self.lines.get(self.pos).copied() else {
                break;
            };
            let m = indent(line);
            if m < n || (m == n && !line[m..].starts_with("- ")) {
                break;
            }
            if m > n {
                return Err(Invalid);
            }
            let text = line[m + 2..].trim_start_matches([' ', '\t']);
            if text.is_empty() || text.starts_with('#') {
                return Err(Invalid);
            }
            let value = if text.starts_with(['\'', '"']) {
                Value::String(quoted(text)?)
            } else {
                resolve(plain(text, true)?.0)
            };
            values.push(value);
            self.pos += 1;
        }
        Ok(Value::Sequence(values))
    }
    fn block(&mut self, header: &str, n: usize) -> ReadResult<String> {
        let style = header.as_bytes()[0];
        let (chomp, tail) = match header.as_bytes().get(1) {
            Some(b'+' | b'-') => (header.as_bytes()[1], &header[2..]),
            _ => (b' ', &header[1..]),
        };
        if !tail.is_empty() && !tail.starts_with([' ', '\t']) {
            return Err(Invalid);
        }
        if !trim(tail).is_empty() && !trim(tail).starts_with('#') {
            return Err(Invalid);
        }
        let start = self.pos;
        while self.pos < self.lines.len()
            && (blank(self.lines[self.pos]) || indent(self.lines[self.pos]) > n)
        {
            self.pos += 1;
        }
        let region = &self.lines[start..self.pos];
        let content_indent = region
            .iter()
            .find(|line| !blank(line))
            .map(|line| indent(line));
        let mut content = Vec::new();
        for line in region {
            if blank(line) {
                if line.contains('\t')
                    || content_indent.map_or(!line.is_empty(), |i| line.len() > i)
                {
                    return Err(Invalid);
                }
                content.push(String::new());
            } else {
                let i = content_indent.ok_or(Invalid)?;
                if indent(line) < i {
                    return Err(Invalid);
                }
                let text = &line[i..];
                if style == b'>' && text.starts_with([' ', '\t']) {
                    return Err(Invalid);
                }
                content.push(text.to_owned());
            }
        }
        let trailing = content
            .iter()
            .rev()
            .take_while(|line| line.is_empty())
            .count();
        let e = trailing.saturating_sub(usize::from(
            self.pos == self.lines.len() && !self.terminated && trailing > 0,
        ));
        content.truncate(content.len() - trailing);
        if content.is_empty() {
            return Ok(if chomp == b'+' {
                "\n".repeat(e)
            } else {
                String::new()
            });
        }
        let mut out = if style == b'|' {
            content.join("\n")
        } else {
            fold(&content)
        };
        if chomp != b'-' {
            out.push('\n');
        }
        if chomp == b'+' {
            out.push_str(&"\n".repeat(e));
        }
        Ok(out)
    }
}

fn fold(lines: &[String]) -> String {
    let mut out = String::new();
    let mut blanks = 0;
    for line in lines {
        if line.is_empty() {
            blanks += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blanks == 0 { " " } else { "" });
        }
        out.push_str(&"\n".repeat(blanks));
        out.push_str(line);
        blanks = 0;
    }
    out
}

fn plain(text: &str, exceptions: bool) -> ReadResult<(String, bool)> {
    let split = text
        .char_indices()
        .find(|&(i, c)| {
            c == '#'
                && i > 0
                && text
                    .as_bytes()
                    .get(i - 1)
                    .is_some_and(|b| b" \t".contains(b))
        })
        .map(|(i, _)| i);
    let value = trim(&text[..split.unwrap_or(text.len())]);
    if value.is_empty()
        || value.contains('\t')
        || value.ends_with(':')
        || value
            .as_bytes()
            .windows(2)
            .any(|p| p[0] == b':' && b" \t".contains(&p[1]))
    {
        return Err(Invalid);
    }
    let first = value.as_bytes()[0];
    if b"-?:,[]{}#&*!|>'\"%@`".contains(&first)
        && !(exceptions
            && b"-?:".contains(&first)
            && value.as_bytes().get(1).is_some_and(|b| !b" \t".contains(b)))
    {
        return Err(Invalid);
    }
    Ok((value.to_owned(), split.is_some()))
}

fn quoted(text: &str) -> ReadResult<String> {
    let quote = text.chars().next().ok_or(Invalid)?;
    let mut chars = text[1..].char_indices().peekable();
    let mut out = String::new();
    while let Some((offset, c)) = chars.next() {
        if c == quote {
            if quote == '\'' && chars.peek().is_some_and(|&(_, c)| c == '\'') {
                chars.next();
                out.push('\'');
                continue;
            }
            let tail = &text[offset + 2..];
            if !tail.is_empty() && !tail.starts_with([' ', '\t']) {
                return Err(Invalid);
            }
            if !trim(tail).is_empty() && !trim(tail).starts_with('#') {
                return Err(Invalid);
            }
            return Ok(out);
        }
        if quote == '"' && c == '\\' {
            let (_, escaped) = chars.next().ok_or(Invalid)?;
            let c = match escaped {
                '\\' | '"' | '/' => escaped,
                't' => '\t',
                'n' => '\n',
                'x' | 'u' | 'U' => {
                    let count = match escaped {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let mut code = 0u32;
                    for _ in 0..count {
                        code = code
                            .checked_mul(16)
                            .and_then(|n| {
                                chars
                                    .next()
                                    .and_then(|(_, c)| c.to_digit(16))
                                    .map(|digit| n + digit)
                            })
                            .ok_or(Invalid)?;
                    }
                    char::from_u32(code).ok_or(Invalid)?
                }
                _ => return Err(Invalid),
            };
            if forbidden(c) {
                return Err(Invalid);
            }
            out.push(c);
        } else {
            out.push(c);
        }
    }
    Err(Invalid)
}

fn resolve(text: String) -> Value {
    match text.as_str() {
        "~" | "null" | "Null" | "NULL" => Value::Null,
        "true" | "True" | "TRUE" => Value::Bool(true),
        "false" | "False" | "FALSE" => Value::Bool(false),
        ".inf" | ".Inf" | ".INF" | "+.inf" | "+.Inf" | "+.INF" => Value::Number(f64::INFINITY),
        "-.inf" | "-.Inf" | "-.INF" => Value::Number(f64::NEG_INFINITY),
        ".nan" | ".NaN" | ".NAN" => Value::Number(f64::NAN),
        _ => {
            if let Some(digits) = text
                .strip_prefix("0o")
                .filter(|s| !s.is_empty() && s.bytes().all(|b| matches!(b, b'0'..=b'7')))
            {
                return Value::Number(
                    digits
                        .bytes()
                        .fold(0.0, |n, b| n * 8.0 + f64::from(b - b'0')),
                );
            }
            if let Some(digits) = text
                .strip_prefix("0x")
                .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                return Value::Number(digits.chars().fold(0.0, |n, c| {
                    n * 16.0 + f64::from(c.to_digit(16).expect("checked hex"))
                }));
            }
            // Rust's parser alone admits spellings outside the YAML core schema.
            let pattern = regress::Regex::new(r"^[+-]?(?:[0-9]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$").expect("constant regex");
            if pattern.find(&text).is_some()
                && let Ok(number) = text.parse()
            {
                return Value::Number(number);
            }
            Value::String(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as Json;
    fn equivalent(value: &Value, expected: &Json) -> bool {
        match value {
            Value::Null => expected.is_null(),
            Value::Bool(v) => expected.as_bool() == Some(*v),
            Value::String(v) => expected.as_str() == Some(v),
            Value::Number(v) => expected.get("$number").and_then(Json::as_str).is_some_and(|s| matches!(resolve(s.to_owned()), Value::Number(w) if v == &w || v.is_nan() && w.is_nan())),
            Value::Mapping(entries) => expected.get("$map").and_then(Json::as_array).is_some_and(|pairs| pairs.len() == entries.len() && entries.iter().zip(pairs).all(|((k, v), p)| p[0].as_str() == Some(k) && equivalent(v, &p[1]))),
            Value::Sequence(entries) => expected.get("$seq").and_then(Json::as_array).is_some_and(|items| items.len() == entries.len() && entries.iter().zip(items).all(|(v, e)| equivalent(v, e))),
        }
    }
    #[test]
    fn shared_subset_corpus() {
        use crate::llm::RawValue;
        let corpus = RawValue::decode(include_str!(
            "../../../../../minion-agent-python/tests/skills/data/frontmatter-corpus.json"
        ))
        .unwrap();
        let RawValue::Array(cases) = corpus.get("cases").unwrap() else {
            panic!("corpus cases");
        };
        for (index, case) in cases.iter().enumerate() {
            let source = case.get("src").unwrap();
            let source = source.as_string().unwrap().to_string();
            let result = source.as_deref().map_or(Err(Invalid), read);
            assert_eq!(
                result.is_ok(),
                case.get("ok").unwrap() == RawValue::Bool(true),
                "case {index}: {source:?}"
            );
            if let Ok(value) = result {
                let expected = case.get("value").unwrap().try_to_json().unwrap();
                assert!(
                    equivalent(&value, &expected),
                    "case {index}: {:?} != {}",
                    value,
                    expected
                );
            }
        }
    }

    #[test]
    fn collection_depth_is_exactly_64_and_root_counts() {
        for depth in [63, 64, 65, 100, 500, 1200] {
            let mut document = String::new();
            for index in 0..depth - 1 {
                document.push_str(&format!("{}a:\n", " ".repeat(index)));
            }
            document.push_str(&format!("{}description: value", " ".repeat(depth - 1)));
            assert_eq!(
                read(&document).is_ok(),
                depth <= 64,
                "mapping depth {depth}"
            );
        }
        for depth in [64, 65] {
            let mut document = String::new();
            for index in 0..depth - 1 {
                document.push_str(&format!("{}a:\n", " ".repeat(index)));
            }
            document.push_str(&format!("{}- value", " ".repeat(depth - 2)));
            assert_eq!(
                read(&document).is_ok(),
                depth <= 64,
                "sequence depth {depth}"
            );
        }
    }
}
