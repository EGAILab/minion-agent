//! Pinned jsdiff 8.0.4 line-token Myers search and context-4 patch projection.
//! Source: kpdecker/jsdiff tag 8.0.4 src/diff/{base,line}.ts and patch/create.ts;
//! display projection: pinned Pi edit-diff.ts. No third-party approximation is used.

use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Eq, PartialEq)]
enum Kind {
    Common,
    Add,
    Remove,
}
#[derive(Clone)]
struct Component {
    kind: Kind,
    count: usize,
}
#[derive(Clone)]
struct SearchPath {
    old_pos: isize,
    components: Vec<Component>,
}
struct Part {
    kind: Kind,
    value: String,
}

fn add(path: &SearchPath, kind: Kind) -> SearchPath {
    let mut next = path.clone();
    if kind == Kind::Remove {
        next.old_pos += 1;
    }
    if let Some(last) = next.components.last_mut()
        && last.kind == kind
    {
        last.count += 1;
    } else {
        next.components.push(Component { kind, count: 1 });
    }
    next
}

fn common(path: &mut SearchPath, old: &[&str], new: &[&str], diagonal: isize) -> isize {
    let mut new_pos = path.old_pos - diagonal;
    let mut count = 0;
    while new_pos + 1 < new.len() as isize
        && path.old_pos + 1 < old.len() as isize
        && old[(path.old_pos + 1) as usize] == new[(new_pos + 1) as usize]
    {
        path.old_pos += 1;
        new_pos += 1;
        count += 1;
    }
    if count > 0 {
        path.components.push(Component {
            kind: Kind::Common,
            count,
        });
    }
    new_pos
}

fn values(path: SearchPath, old: &[&str], new: &[&str]) -> Vec<Part> {
    let mut old_pos = 0;
    let mut new_pos = 0;
    path.components
        .into_iter()
        .map(|c| {
            let value = if c.kind == Kind::Remove {
                let text = old[old_pos..old_pos + c.count].concat();
                old_pos += c.count;
                text
            } else {
                let text = new[new_pos..new_pos + c.count].concat();
                new_pos += c.count;
                if c.kind == Kind::Common {
                    old_pos += c.count;
                }
                text
            };
            Part {
                kind: c.kind,
                value,
            }
        })
        .collect()
}

fn diff_lines(old: &str, new: &str) -> Vec<Part> {
    let old: Vec<_> = old.split_inclusive('\n').collect();
    let new: Vec<_> = new.split_inclusive('\n').collect();
    let mut seed = SearchPath {
        old_pos: -1,
        components: Vec::new(),
    };
    let new_pos = common(&mut seed, &old, &new, 0);
    if seed.old_pos + 1 >= old.len() as isize && new_pos + 1 >= new.len() as isize {
        return values(seed, &old, &new);
    }
    let mut best = BTreeMap::from([(0, seed)]);
    let mut minimum = isize::MIN;
    let mut maximum = isize::MAX;
    for length in 1..=old.len() + new.len() {
        let mut diagonal = minimum.max(-(length as isize));
        let end = maximum.min(length as isize);
        while diagonal <= end {
            let remove = best.remove(&(diagonal - 1));
            let insert = best.get(&(diagonal + 1)).cloned();
            let can_add = insert.as_ref().is_some_and(|p| {
                let n = p.old_pos - diagonal;
                n >= 0 && n < new.len() as isize
            });
            let can_remove = remove
                .as_ref()
                .is_some_and(|p| p.old_pos + 1 < old.len() as isize);
            if !can_add && !can_remove {
                best.remove(&diagonal);
                diagonal += 2;
                continue;
            }
            let mut path = if !can_remove
                || (can_add && remove.as_ref().unwrap().old_pos < insert.as_ref().unwrap().old_pos)
            {
                add(insert.as_ref().unwrap(), Kind::Add)
            } else {
                add(remove.as_ref().unwrap(), Kind::Remove)
            };
            let new_pos = common(&mut path, &old, &new, diagonal);
            if path.old_pos + 1 >= old.len() as isize && new_pos + 1 >= new.len() as isize {
                return values(path, &old, &new);
            }
            if path.old_pos + 1 >= old.len() as isize {
                maximum = maximum.min(diagonal - 1);
            }
            if new_pos + 1 >= new.len() as isize {
                minimum = minimum.max(diagonal + 1);
            }
            best.insert(diagonal, path);
            diagonal += 2;
        }
    }
    unreachable!("unbounded Myers search reaches the end of a finite graph")
}

fn display(parts: &[Part], old: &str, new: &str) -> (String, Option<usize>) {
    let width = old
        .split('\n')
        .count()
        .max(new.split('\n').count())
        .to_string()
        .len();
    let mut old_line = 1;
    let mut new_line = 1;
    let mut first = None;
    let mut output = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let mut lines: Vec<_> = part.value.split('\n').collect();
        if lines.last() == Some(&"") {
            lines.pop();
        }
        if part.kind != Kind::Common {
            first.get_or_insert(new_line);
            for line in lines {
                if part.kind == Kind::Add {
                    output.push(format!("+{new_line:>width$} {line}"));
                    new_line += 1;
                } else {
                    output.push(format!("-{old_line:>width$} {line}"));
                    old_line += 1;
                }
            }
            continue;
        }
        let leading = i > 0 && parts[i - 1].kind != Kind::Common;
        let trailing = i + 1 < parts.len() && parts[i + 1].kind != Kind::Common;
        let mut skipped = false;
        for (j, line) in lines.iter().enumerate() {
            let show = (leading && j < 4) || (trailing && j >= lines.len().saturating_sub(4));
            if show {
                output.push(format!(" {old_line:>width$} {line}"));
                skipped = false;
            } else if (leading || trailing) && !skipped {
                output.push(format!(" {:>width$} ...", ""));
                skipped = true;
            }
            old_line += 1;
            new_line += 1;
        }
    }
    (output.join("\n"), first)
}

fn patch(parts: &[Part], path: &str) -> String {
    let mut old_line = 1;
    let mut new_line = 1;
    let mut start = None;
    let mut range: Vec<String> = Vec::new();
    let mut previous: Vec<&str> = Vec::new();
    let mut output = vec![format!("--- {path}"), format!("+++ {path}")];
    // The empty final common component closes a last change exactly like structuredPatch.
    for i in 0..=parts.len() {
        let (kind, lines) = if i < parts.len() {
            (
                parts[i].kind,
                parts[i].value.split_inclusive('\n').collect::<Vec<_>>(),
            )
        } else {
            (Kind::Common, Vec::new())
        };
        if kind != Kind::Common {
            if start.is_none() {
                let context = &previous[previous.len().saturating_sub(4)..];
                start = Some((old_line - context.len(), new_line - context.len()));
                range.extend(context.iter().map(|line| format!(" {line}")));
            }
            let prefix = if kind == Kind::Add { '+' } else { '-' };
            range.extend(lines.iter().map(|line| format!("{prefix}{line}")));
            if kind == Kind::Add {
                new_line += lines.len();
            } else {
                old_line += lines.len();
            }
        } else {
            if let Some((old_start, new_start)) = start {
                if lines.len() <= 8 && i + 1 < parts.len() {
                    range.extend(lines.iter().map(|line| format!(" {line}")));
                } else {
                    let context = lines.len().min(4);
                    range.extend(lines[..context].iter().map(|line| format!(" {line}")));
                    let old_count = old_line - old_start + context;
                    let new_count = new_line - new_start + context;
                    let old_start = if old_count == 0 {
                        old_start - 1
                    } else {
                        old_start
                    };
                    let new_start = if new_count == 0 {
                        new_start - 1
                    } else {
                        new_start
                    };
                    output.push(format!(
                        "@@ -{old_start},{old_count} +{new_start},{new_count} @@"
                    ));
                    for line in range.drain(..) {
                        if let Some(line) = line.strip_suffix('\n') {
                            output.push(line.to_owned());
                        } else {
                            output.push(line);
                            output.push("\\ No newline at end of file".into());
                        }
                    }
                    start = None;
                }
            }
            old_line += lines.len();
            new_line += lines.len();
        }
        previous = lines;
    }
    output.join("\n") + "\n"
}

pub fn generate_edit_details(path: &str, base: &str, new: &str) -> Value {
    let parts = diff_lines(base, new);
    let (diff, first) = display(&parts, base, new);
    let mut details = json!({"diff":diff, "patch":patch(&parts, path)});
    if let Some(first) = first {
        details["firstChangedLine"] = json!(first);
    }
    details
}
