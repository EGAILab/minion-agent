use std::ops::Index;

use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum JsJsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(JsString),
    Array(Vec<JsJsonValue>),
    Object(Vec<(JsString, JsJsonValue)>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JsString(Vec<u16>);

impl JsString {
    pub fn to_string(&self) -> Option<String> {
        String::from_utf16(&self.0).ok()
    }

    fn equals(&self, value: &str) -> bool {
        self.0.iter().copied().eq(value.encode_utf16())
    }
}

impl JsJsonValue {
    pub fn get(&self, key: &str) -> Option<&Self> {
        let Self::Object(values) = self else {
            return None;
        };
        values
            .iter()
            .find_map(|(candidate, value)| candidate.equals(key).then_some(value))
    }

    pub fn as_f64(&self) -> Option<f64> {
        let Self::Number(value) = self else {
            return None;
        };
        Some(*value)
    }

    pub fn as_string(&self) -> Option<String> {
        let Self::String(value) = self else {
            return None;
        };
        value.to_string()
    }
}

impl Index<&str> for JsJsonValue {
    type Output = JsJsonValue;
    fn index(&self, key: &str) -> &Self::Output {
        self.get(key).expect("JSON object key is present")
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("invalid JSON at byte {offset}: {message}")]
pub struct JsJsonError {
    offset: usize,
    message: &'static str,
}

impl JsJsonError {
    fn new(offset: usize, message: &'static str) -> Self {
        Self { offset, message }
    }
}

pub fn js_trim(value: &str) -> &str {
    value.trim_matches(is_ecmascript_whitespace)
}

fn is_ecmascript_whitespace(value: char) -> bool {
    matches!(
        value,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            | '\u{2001}'
            | '\u{2002}'
            | '\u{2003}'
            | '\u{2004}'
            | '\u{2005}'
            | '\u{2006}'
            | '\u{2007}'
            | '\u{2008}'
            | '\u{2009}'
            | '\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
            | '\u{FEFF}'
    )
}

/// Parse JSON into JavaScript's IEEE-754 number and UTF-16 string domains.
pub fn js_json_loads(text: &str) -> Result<JsJsonValue, JsJsonError> {
    let mut parser = Parser {
        source: text,
        offset: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.offset != text.len() {
        return Err(JsJsonError::new(parser.offset, "trailing input"));
    }
    Ok(value)
}

struct Parser<'a> {
    source: &'a str,
    offset: usize,
}

impl Parser<'_> {
    fn value(&mut self) -> Result<JsJsonValue, JsJsonError> {
        match self.peek() {
            Some(b'n') => self.literal(b"null", JsJsonValue::Null),
            Some(b't') => self.literal(b"true", JsJsonValue::Bool(true)),
            Some(b'f') => self.literal(b"false", JsJsonValue::Bool(false)),
            Some(b'"') => self.string().map(JsJsonValue::String),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(JsJsonError::new(self.offset, "expected a JSON value")),
        }
    }

    fn literal(&mut self, literal: &[u8], value: JsJsonValue) -> Result<JsJsonValue, JsJsonError> {
        if self
            .source
            .as_bytes()
            .get(self.offset..self.offset + literal.len())
            == Some(literal)
        {
            self.offset += literal.len();
            Ok(value)
        } else {
            Err(JsJsonError::new(self.offset, "invalid literal"))
        }
    }

    fn string(&mut self) -> Result<JsString, JsJsonError> {
        self.offset += 1;
        let mut output = Vec::new();
        loop {
            let Some(character) = self.source[self.offset..].chars().next() else {
                return Err(JsJsonError::new(self.offset, "unterminated string"));
            };
            self.offset += character.len_utf8();
            match character {
                '"' => return Ok(JsString(output)),
                '\\' => self.escape(&mut output)?,
                value if value <= '\u{001f}' => {
                    return Err(JsJsonError::new(self.offset, "unescaped control character"));
                }
                value => {
                    let mut units = [0_u16; 2];
                    output.extend_from_slice(value.encode_utf16(&mut units));
                }
            }
        }
    }

    fn escape(&mut self, output: &mut Vec<u16>) -> Result<(), JsJsonError> {
        let Some(escape) = self.peek() else {
            return Err(JsJsonError::new(self.offset, "incomplete escape"));
        };
        self.offset += 1;
        match escape {
            b'"' | b'\\' | b'/' => output.push(u16::from(escape)),
            b'b' => output.push(8),
            b'f' => output.push(12),
            b'n' => output.push(10),
            b'r' => output.push(13),
            b't' => output.push(9),
            b'u' => {
                let end = self.offset + 4;
                let digits = self
                    .source
                    .as_bytes()
                    .get(self.offset..end)
                    .ok_or_else(|| JsJsonError::new(self.offset, "incomplete unicode escape"))?;
                let digits = std::str::from_utf8(digits)
                    .map_err(|_| JsJsonError::new(self.offset, "invalid unicode escape"))?;
                output.push(
                    u16::from_str_radix(digits, 16)
                        .map_err(|_| JsJsonError::new(self.offset, "invalid unicode escape"))?,
                );
                self.offset = end;
            }
            _ => return Err(JsJsonError::new(self.offset, "invalid escape")),
        }
        Ok(())
    }

    fn array(&mut self) -> Result<JsJsonValue, JsJsonError> {
        self.offset += 1;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume(b']') {
            return Ok(JsJsonValue::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.skip_whitespace();
            if self.consume(b']') {
                return Ok(JsJsonValue::Array(values));
            }
            self.expect(b',')?;
            self.skip_whitespace();
        }
    }

    fn object(&mut self) -> Result<JsJsonValue, JsJsonError> {
        self.offset += 1;
        self.skip_whitespace();
        let mut values: Vec<(JsString, JsJsonValue)> = Vec::new();
        if self.consume(b'}') {
            return Ok(JsJsonValue::Object(values));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(JsJsonError::new(self.offset, "expected an object key"));
            }
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value()?;
            if let Some(existing) = values.iter_mut().find(|(candidate, _)| candidate == &key) {
                existing.1 = value;
            } else {
                values.push((key, value));
            }
            self.skip_whitespace();
            if self.consume(b'}') {
                return Ok(JsJsonValue::Object(values));
            }
            self.expect(b',')?;
            self.skip_whitespace();
        }
    }

    fn number(&mut self) -> Result<JsJsonValue, JsJsonError> {
        let start = self.offset;
        self.consume(b'-');
        if self.consume(b'0') {
            if self.peek().is_some_and(|value| value.is_ascii_digit()) {
                return Err(JsJsonError::new(self.offset, "leading zero"));
            }
        } else {
            self.digits()?;
        }
        if self.consume(b'.') {
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.offset += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.offset += 1;
            }
            self.digits()?;
        }
        let value = self.source[start..self.offset]
            .parse::<f64>()
            .map_err(|_| JsJsonError::new(start, "invalid number"))?;
        Ok(JsJsonValue::Number(value))
    }

    fn digits(&mut self) -> Result<(), JsJsonError> {
        let start = self.offset;
        while self.peek().is_some_and(|value| value.is_ascii_digit()) {
            self.offset += 1;
        }
        if self.offset == start {
            Err(JsJsonError::new(self.offset, "expected a digit"))
        } else {
            Ok(())
        }
    }
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.offset += 1;
        }
    }
    fn expect(&mut self, byte: u8) -> Result<(), JsJsonError> {
        if self.consume(byte) {
            Ok(())
        } else {
            Err(JsJsonError::new(self.offset, "unexpected token"))
        }
    }
    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn peek(&self) -> Option<u8> {
        self.source.as_bytes().get(self.offset).copied()
    }
}

pub fn js_json_stringify(value: &JsJsonValue) -> String {
    let mut output = String::new();
    write_value(value, &mut output);
    output
}

fn write_value(value: &JsJsonValue, output: &mut String) {
    match value {
        JsJsonValue::Null => output.push_str("null"),
        JsJsonValue::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        JsJsonValue::Number(value) if value.is_finite() => {
            output.push_str(ryu_js::Buffer::new().format(*value))
        }
        JsJsonValue::Number(_) => output.push_str("null"),
        JsJsonValue::String(value) => write_string(value, output),
        JsJsonValue::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_value(value, output);
            }
            output.push(']');
        }
        JsJsonValue::Object(values) => {
            output.push('{');
            let mut indices = values
                .iter()
                .enumerate()
                .filter_map(|(position, (key, _))| array_index(key).map(|index| (index, position)))
                .collect::<Vec<_>>();
            indices.sort_by_key(|(index, _)| *index);
            let ordinary = values
                .iter()
                .enumerate()
                .filter(|(_, (key, _))| array_index(key).is_none())
                .map(|(position, _)| position);
            let mut first = true;
            for position in indices
                .into_iter()
                .map(|(_, position)| position)
                .chain(ordinary)
            {
                if !first {
                    output.push(',');
                }
                first = false;
                write_string(&values[position].0, output);
                output.push(':');
                write_value(&values[position].1, output);
            }
            output.push('}');
        }
    }
}

fn write_string(value: &JsString, output: &mut String) {
    output.push('"');
    let mut index = 0;
    while index < value.0.len() {
        let unit = value.0[index];
        let paired = (0xd800..=0xdbff).contains(&unit)
            && value
                .0
                .get(index + 1)
                .is_some_and(|low| (0xdc00..=0xdfff).contains(low));
        if paired {
            let scalar = 0x1_0000
                + ((u32::from(unit) - 0xd800) << 10)
                + (u32::from(value.0[index + 1]) - 0xdc00);
            output.push(char::from_u32(scalar).expect("paired surrogates form a scalar"));
            index += 2;
            continue;
        }
        match unit {
            0x22 => output.push_str("\\\""),
            0x5c => output.push_str("\\\\"),
            0x08 => output.push_str("\\b"),
            0x09 => output.push_str("\\t"),
            0x0a => output.push_str("\\n"),
            0x0c => output.push_str("\\f"),
            0x0d => output.push_str("\\r"),
            value @ (0x00..=0x1f | 0xd800..=0xdfff) => {
                use std::fmt::Write as _;
                write!(output, "\\u{value:04x}").expect("String write");
            }
            value => output.push(char::from_u32(u32::from(value)).expect("BMP scalar")),
        }
        index += 1;
    }
    output.push('"');
}

fn array_index(key: &JsString) -> Option<u32> {
    let key = key.to_string()?;
    if key.is_empty()
        || !key.bytes().all(|value| value.is_ascii_digit())
        || (key.len() > 1 && key.starts_with('0'))
    {
        return None;
    }
    let value = key.parse::<u64>().ok()?;
    (value <= u64::from(u32::MAX - 1)).then_some(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stringify_matches_discriminating_ecmascript_boundaries() {
        let value = js_json_loads(r#"{"plain":"é\u0001","2":"two","1":"one","small_fixed":0.000001,"small_exp":0.0000001,"large_fixed":100000000000000000000,"large_exp":1e21,"negative_zero":-0}"#).unwrap();
        assert_eq!(
            js_json_stringify(&value),
            "{\"1\":\"one\",\"2\":\"two\",\"plain\":\"é\\u0001\",\"small_fixed\":0.000001,\"small_exp\":1e-7,\"large_fixed\":100000000000000000000,\"large_exp\":1e+21,\"negative_zero\":0}"
        );
    }

    #[test]
    fn strings_preserve_unpaired_surrogates_and_combine_valid_pairs() {
        let value = js_json_loads(r#"{"lone":"\ud800","paired":"\ud83d\ude00"}"#).unwrap();
        assert_eq!(
            js_json_stringify(&value),
            r#"{"lone":"\ud800","paired":"😀"}"#
        );
    }

    #[test]
    fn loads_uses_ieee_754_and_rejects_non_json_constants() {
        let value =
            js_json_loads("{\"large\":9007199254740993,\"minus\":-0,\"huge\":1e400}").unwrap();
        assert_eq!(value["large"].as_f64(), Some(9_007_199_254_740_992.0));
        assert!(value["minus"].as_f64().unwrap().is_sign_negative());
        assert_eq!(js_json_stringify(&value["huge"]), "null");
        assert!(js_json_loads("NaN").is_err());
        assert!(js_json_loads("Infinity").is_err());
    }

    #[test]
    fn trim_uses_the_ecmascript_set_including_bom() {
        assert_eq!(js_trim("\u{feff}\u{2007} 5 \u{2029}"), "5");
    }
}
