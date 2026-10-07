//! DIV-002 reads fd's Windows tokens, not a guessed Linux glob grammar.
const SEP: &str = r"[/\\]";
#[derive(Clone, Debug, PartialEq)]
enum Token {
    Sep,
    Stars(usize),
    Open,
    Close,
    Comma,
    Text(String),
}
fn lex(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let tail = &text[offset..];
        if tail.starts_with(SEP) {
            out.push(Token::Sep);
            offset += SEP.len();
            continue;
        }
        let ch = tail.chars().next().unwrap();
        if ch == '[' {
            let mut end = offset + 1;
            if matches!(text.as_bytes().get(end), Some(b'!' | b'^')) {
                end += 1;
            }
            if text.as_bytes().get(end) == Some(&b']') {
                end += 1;
            }
            if let Some(n) = text[end..].find(']') {
                end += n + 1;
            } else {
                end = text.len();
            }
            out.push(Token::Text(text[offset..end].into()));
            offset = end;
            continue;
        }
        if ch == '*' {
            let n = tail.bytes().take_while(|b| *b == b'*').count();
            out.push(Token::Stars(n));
            offset += n;
            continue;
        }
        out.push(match ch {
            '{' => Token::Open,
            '}' => Token::Close,
            ',' => Token::Comma,
            _ => Token::Text(ch.to_string()),
        });
        offset += ch.len_utf8();
    }
    out
}
fn render(token: &Token) -> String {
    match token {
        Token::Sep => SEP.into(),
        Token::Stars(n) => "*".repeat(*n),
        Token::Open => "{".into(),
        Token::Close => "}".into(),
        Token::Comma => ",".into(),
        Token::Text(s) => s.clone(),
    }
}
fn rewrite(tokens: &[Token], alternative: bool) -> String {
    let mut out = String::new();
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
                out.extend(tokens[i..].iter().map(render));
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
            out.push_str(&format!("{{{}}}", parts.join(",")));
            i = end + 1;
            continue;
        }
        // Separator + exactly ** + separator: the two non-empty alternatives
        // retain that separator alone, or it plus the recursive component.
        if tokens[i] == Token::Sep
            && tokens.get(i + 1) == Some(&Token::Stars(2))
            && tokens.get(i + 2) == Some(&Token::Sep)
        {
            out.push_str(&format!("{{{SEP},{SEP}**{SEP}}}"));
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
            out.push_str(&format!("{{{rest},**{SEP}{rest}}}"));
            return out;
        }
        out.push_str(&render(&tokens[i]));
        i += 1;
    }
    out
}
pub(super) fn windows_full_path(pattern: &str) -> String {
    let pi = pattern.replace('/', SEP);
    rewrite(&lex(&pi), false)
}
pub(super) fn pi_windows(pattern: &str) -> String {
    pattern.replace('/', SEP)
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
