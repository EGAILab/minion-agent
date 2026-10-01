//! Fixture-only lossless decoding of the corpus's JSON-style quoted YAML scalars.
//! Quoted scalars are decoded by the certified raw JSON parser, then substituted
//! back after YAML structural decoding. No tool behavior or expectation is computed.
use minion_agent::llm::{RawString, RawValue, ResultString, ResultValue};
use serde_json::Value;

pub fn decode(source: &str) -> RawValue {
    let prefix = "__WP132_YAML_SCALAR_";
    assert!(
        !source.contains(prefix),
        "fixture substitution namespace collision"
    );
    let mut quoted = Vec::new();
    let mut rewritten = String::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch != '"' {
            rewritten.push(ch);
            continue;
        }
        let mut escaped = false;
        let mut end = None;
        for (index, c) in chars.by_ref() {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                end = Some(index + 1);
                break;
            }
        }
        let token = &source[start..end.expect("closed fixture string")];
        // YAML's eight-hex-digit escape denotes one Unicode scalar; transcribe
        // just that spelling to a JSON scalar before the lossless JSON decoder.
        let mut json_token = String::new();
        let mut spelling = token.chars();
        while let Some(c) = spelling.next() {
            json_token.push(c);
            if c == '\\' {
                let next = spelling.next().unwrap();
                if next == 'U' {
                    json_token.pop();
                    let hex = spelling.by_ref().take(8).collect::<String>();
                    json_token
                        .push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
                } else {
                    json_token.push(next);
                }
            }
        }
        let value = RawValue::decode(&json_token)
            .expect("corpus quoted scalars have JSON/Unicode-scalar escape syntax");
        let RawValue::String(value) = value else {
            panic!("quoted scalar")
        };
        let reference = format!("{prefix}{}__", quoted.len());
        assert!(
            !value
                .code_units()
                .windows(prefix.len())
                .any(|w| w.iter().copied().eq(prefix.encode_utf16())),
            "decoded namespace collision"
        );
        quoted.push(value);
        rewritten.push_str(&format!("\"{reference}\""));
    }
    fn scalar(value: &str, quoted: &[RawString], prefix: &str) -> RawString {
        if let Some(index) = value.strip_prefix(prefix) {
            quoted[index.strip_suffix("__").unwrap().parse::<usize>().unwrap()].clone()
        } else {
            value.into()
        }
    }
    fn restore(value: Value, quoted: &[RawString], prefix: &str) -> RawValue {
        match value {
            Value::String(s) if s.starts_with(prefix) => {
                let index = s
                    .strip_prefix(prefix)
                    .unwrap()
                    .strip_suffix("__")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                RawValue::String(quoted[index].clone())
            }
            Value::Array(a) => {
                RawValue::Array(a.into_iter().map(|v| restore(v, quoted, prefix)).collect())
            }
            Value::Object(o) => RawValue::Object(
                o.into_iter()
                    .map(|(k, v)| (scalar(&k, quoted, prefix), restore(v, quoted, prefix)))
                    .collect(),
            ),
            v => RawValue::from(v),
        }
    }
    restore(serde_yaml::from_str(&rewritten).unwrap(), &quoted, prefix)
}

pub fn result(value: &RawValue) -> ResultValue {
    match value {
        RawValue::String(s) => {
            ResultValue::String(ResultString::from_code_units(s.code_units().to_vec()))
        }
        RawValue::Number(n) => ResultValue::number(n.as_f64()),
        RawValue::Array(a) => ResultValue::Array(a.iter().map(result).collect()),
        RawValue::Object(o) => ResultValue::Object(
            o.iter()
                .map(|(k, v)| {
                    (
                        ResultString::from_code_units(k.code_units().to_vec()),
                        result(v),
                    )
                })
                .collect(),
        ),
        RawValue::Bool(b) => ResultValue::Bool(*b),
        RawValue::Null => ResultValue::Null,
    }
}
