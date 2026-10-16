//! DIV-002 reads fd's Windows tokens, not a guessed Linux glob grammar.
const SEP: &str = r"[/\\]";
#[derive(Clone, Debug, PartialEq)]
enum Token {
    Sep,
    Stars(usize),
    Open,
    Close,
    Comma,
    Text(Vec<u16>),
}
fn lex(text: &[u16]) -> Vec<Token> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let tail = &text[offset..];
        if tail.starts_with(&SEP.encode_utf16().collect::<Vec<_>>()) {
            out.push(Token::Sep);
            offset += SEP.len();
            continue;
        }
        let ch = tail[0];
        if ch == 91 {
            let mut end = offset + 1;
            if matches!(text.get(end), Some(33 | 94)) {
                end += 1;
            }
            if text.get(end) == Some(&93) {
                end += 1;
            }
            if let Some(n) = text[end..].iter().position(|u| *u == 93) {
                end += n + 1;
            } else {
                end = text.len();
            }
            out.push(Token::Text(text[offset..end].to_vec()));
            offset = end;
            continue;
        }
        if ch == 42 {
            let n = tail.iter().take_while(|u| **u == 42).count();
            out.push(Token::Stars(n));
            offset += n;
            continue;
        }
        out.push(match ch {
            123 => Token::Open,
            125 => Token::Close,
            44 => Token::Comma,
            _ => Token::Text(vec![ch]),
        });
        offset += 1;
    }
    out
}
fn render(token: &Token) -> Vec<u16> {
    match token {
        Token::Sep => SEP.encode_utf16().collect(),
        Token::Stars(n) => vec![42; *n],
        Token::Open => vec![123],
        Token::Close => vec![125],
        Token::Comma => vec![44],
        Token::Text(s) => s.clone(),
    }
}
fn rewrite(tokens: &[Token], alternative: bool) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == Token::Open {
            let mut depth = 1;
            let mut end = i + 1;
            while end < tokens.len() && depth > 0 {
                match tokens[end] {
                    Token::Open => depth += 1,
                    Token::Close => depth -= 1,
                    _ => {}
                };
                if depth > 0 {
                    end += 1;
                }
            }
            if depth != 0 {
                out.extend(tokens[i..].iter().flat_map(render));
                return out;
            }
            let mut begin = i + 1;
            let mut nested = 0;
            let mut parts = Vec::new();
            for index in i + 1..end {
                match tokens[index] {
                    Token::Open => nested += 1,
                    Token::Close => nested -= 1,
                    Token::Comma if nested == 0 => {
                        parts.push(rewrite(&tokens[begin..index], true));
                        begin = index + 1;
                    }
                    _ => {}
                }
            }
            parts.push(rewrite(&tokens[begin..end], true));
            out.push(123);
            for (n, part) in parts.into_iter().enumerate() {
                if n != 0 {
                    out.push(44);
                }
                out.extend(part);
            }
            out.push(125);
            i = end + 1;
            continue;
        }
        // Separator + exactly ** + separator: the two non-empty alternatives
        // retain that separator alone, or it plus the recursive component.
        if tokens[i] == Token::Sep
            && tokens.get(i + 1) == Some(&Token::Stars(2))
            && tokens.get(i + 2) == Some(&Token::Sep)
        {
            out.extend(format!("{{{SEP},{SEP}**{SEP}}}").encode_utf16());
            i += 3;
            while tokens.get(i) == Some(&Token::Stars(2)) && tokens.get(i + 1) == Some(&Token::Sep)
            {
                i += 2;
            }
            continue;
        }
        // An alternative starts with **/. Empty alternatives do not match fd;
        // duplicate the continuation instead of generating {,**/}.
        if i == 0
            && alternative
            && tokens[i] == Token::Stars(2)
            && tokens.get(i + 1) == Some(&Token::Sep)
        {
            let rest = rewrite(&tokens[2..], true);
            out.push(123);
            out.extend_from_slice(&rest);
            out.extend(format!(",**{SEP}").encode_utf16());
            out.extend(rest);
            out.push(125);
            return out;
        }
        out.extend(render(&tokens[i]));
        i += 1;
    }
    out
}
#[cfg(test)]
pub(super) fn windows_full_path(pattern: &str) -> String {
    windows_full_path_lossless(&pattern.into())
        .as_str()
        .expect("scalar input stays scalar")
        .to_owned()
}
#[cfg(test)]
pub(super) fn pi_windows(pattern: &str) -> String {
    pattern.replace('/', SEP)
}

pub(super) fn pi_windows_lossless(pattern: &crate::llm::ResultString) -> crate::llm::ResultString {
    let mut units = Vec::new();
    for &unit in pattern.code_units() {
        if unit == 47 {
            units.extend(SEP.encode_utf16());
        } else {
            units.push(unit);
        }
    }
    crate::llm::ResultString::from_code_units(units)
}

pub(super) fn windows_full_path_lossless(
    pattern: &crate::llm::ResultString,
) -> crate::llm::ResultString {
    let pi = pi_windows_lossless(pattern);
    crate::llm::ResultString::from_code_units(rewrite(&lex(pi.code_units()), false))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separators_classes_and_alternation_context() {
        assert_eq!(windows_full_path("**/x,**/b"), pi_windows("**/x,**/b"));
        assert_ne!(windows_full_path("**/x,{**/b}"), pi_windows("**/x,{**/b}"));
        assert_eq!(
            windows_full_path("**/src/[!]/**/*.ts"),
            pi_windows("**/src/[!]/**/*.ts")
        );
        assert_eq!(
            windows_full_path("**/src/**b.ts"),
            pi_windows("**/src/**b.ts")
        );
    }
}
