use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
use base64::{Engine as _, engine::general_purpose::STANDARD};
use parking_lot::RwLock;
use serde_json::Value;
use thiserror::Error;

use super::{JsJsonValue, ModelAuth, OAuthCredential, js_json_loads};

pub const ACCOUNT_NAMESPACE: &str = "https://api.openai.com/auth";
pub const ACCOUNT_ID_KEY: &str = "chatgpt_account_id";

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CodexProjectionError {
    #[error("Failed to extract accountId from token")]
    MissingAccountId,
}

pub fn decode_jwt(token: &str) -> Option<JsJsonValue> {
    let mut parts = token.split('.');
    let _header = parts.next()?;
    let payload = parts.next()?;
    let _signature = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let bytes = forgiving_base64_decode(payload)?;
    let latin1 = bytes.into_iter().map(char::from).collect::<String>();
    js_json_loads(&latin1).ok()
}

pub fn get_account_id(access_token: &str) -> Option<String> {
    decode_jwt(access_token)?
        .get(ACCOUNT_NAMESPACE)?
        .get(ACCOUNT_ID_KEY)?
        .as_string()
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

pub fn credentials_from_token(
    access: impl Into<String>,
    refresh: impl Into<String>,
    expires: f64,
) -> Result<OAuthCredential, CodexProjectionError> {
    let access = access.into();
    let account_id = get_account_id(&access).ok_or(CodexProjectionError::MissingAccountId)?;
    let extra = Arc::new(RwLock::new(BTreeMap::from([(
        "account_id".to_owned(),
        Value::String(account_id),
    )])));
    Ok(OAuthCredential::new(access, refresh.into(), expires, extra))
}

pub fn codex_to_auth(credential: &OAuthCredential) -> ModelAuth {
    ModelAuth {
        api_key: Some(credential.access()),
        ..ModelAuth::default()
    }
}

fn forgiving_base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut normalized = input
        .chars()
        .filter(|value| {
            !matches!(
                value,
                '\u{0009}' | '\u{000a}' | '\u{000c}' | '\u{000d}' | ' '
            )
        })
        .collect::<String>();

    if normalized.ends_with('=') {
        if normalized.len() % 4 != 0 {
            return None;
        }
        let padding = normalized
            .chars()
            .rev()
            .take_while(|value| *value == '=')
            .count();
        if !(1..=2).contains(&padding) {
            return None;
        }
        normalized.truncate(normalized.len() - padding);
    }
    if normalized.contains('=') || normalized.len() % 4 == 1 {
        return None;
    }
    if !normalized
        .bytes()
        .all(|value| value.is_ascii_alphanumeric() || matches!(value, b'+' | b'/'))
    {
        return None;
    }
    let mut output = Vec::with_capacity(normalized.len() * 3 / 4);
    let mut accumulator = 0_u32;
    let mut bits = 0_u8;
    for byte in normalized.bytes() {
        let sextet = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        accumulator = (accumulator << 6) | u32::from(sextet);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((accumulator >> bits) as u8);
            accumulator &= (1_u32 << bits) - 1;
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(payload: &[u8]) -> String {
        format!("x.{}.y", STANDARD.encode(payload).trim_end_matches('='))
    }

    #[test]
    fn jwt_decode_uses_forgiving_base64_latin1_and_javascript_numbers() {
        let encoded = STANDARD.encode(br#"{"n":9007199254740993,"z":-0}"#);
        let spaced = format!("{} \n{}", &encoded[..4], &encoded[4..]);
        let parsed = decode_jwt(&format!("x.{spaced}.y")).unwrap();
        assert_eq!(parsed["n"].as_f64(), Some(9_007_199_254_740_992.0));
        assert!(parsed["z"].as_f64().unwrap().is_sign_negative());

        let mojibake = decode_jwt(&token(b"\"\xc3\xa9\"")).unwrap();
        assert_eq!(mojibake.as_string().as_deref(), Some("Ã©"));
    }

    #[test]
    fn jwt_decode_rejects_url_alphabet_bad_padding_and_json_constants() {
        assert!(decode_jwt("x.e30").is_none());
        assert!(decode_jwt("x.e30.y.extra").is_none());
        assert_eq!(
            decode_jwt("x.e31=.y"),
            Some(JsJsonValue::Object(Vec::new()))
        );
        assert!(decode_jwt("x.-w.y").is_none());
        assert!(decode_jwt("x.ew=.y").is_none());
        assert!(decode_jwt(&token(b"NaN")).is_none());
        assert_eq!(decode_jwt(&token(b"null")), Some(JsJsonValue::Null));
    }

    #[test]
    fn account_projection_uses_open_extra_and_bearer_auth() {
        let access = token(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct"}}"#);
        let credential = credentials_from_token(access.clone(), "refresh", 42.0).unwrap();
        assert_eq!(credential.extra().read()["account_id"], "acct");
        assert_eq!(codex_to_auth(&credential).api_key, Some(access));
        assert_eq!(
            credentials_from_token(token(b"{}"), "refresh", 0.0).unwrap_err(),
            CodexProjectionError::MissingAccountId
        );
    }
}
