use super::{
    frontmatter::{self, Value},
    ignore::Ignore,
    *,
};
use crate::{
    execution::{FileInfo, FileKind, FileSystem, FsError, FsErrorCode, FsPath},
    javascript::js_trim,
};
use std::sync::Arc;

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[error("{0}")]
pub struct SkillDiscoveryError(pub String);

#[derive(Debug, thiserror::Error)]
pub enum SourcedSkillError<E> {
    #[error(transparent)]
    Discovery(#[from] SkillDiscoveryError),
    #[error("skill mapping failed")]
    Mapping(E),
}

fn diag(
    output: &mut LoadedSkills,
    code: SkillDiagnosticCode,
    message: impl Into<FsPath>,
    path: &FsPath,
) {
    output.diagnostics.push(SkillDiagnostic {
        diagnostic_type: SkillDiagnosticType::Warning,
        code,
        message: message.into(),
        path: path.clone(),
    });
}
fn info_error(output: &mut LoadedSkills, error: FsError, path: &FsPath) {
    if error.code != FsErrorCode::NotFound {
        diag(
            output,
            SkillDiagnosticCode::FileInfoFailed,
            error.message,
            path,
        );
    }
}
async fn kind(fs: &dyn FileSystem, info: &FileInfo, output: &mut LoadedSkills) -> Option<FileKind> {
    if info.kind != FileKind::Symlink {
        return Some(info.kind);
    }
    let canonical = match fs.canonical_path(&info.path, None).await {
        Ok(path) => path,
        Err(error) => {
            info_error(output, error, &info.path);
            return None;
        }
    };
    match fs.file_info(&canonical, None).await {
        Ok(target) if target.kind != FileKind::Symlink => Some(target.kind),
        Ok(_) => None,
        Err(error) => {
            info_error(output, error, &info.path);
            None
        }
    }
}
fn relative(root: &FsPath, path: &FsPath) -> Vec<u16> {
    let normalize = |value: &FsPath| {
        let mut units: Vec<_> = value
            .code_units()
            .iter()
            .map(|&u| if u == 92 { 47 } else { u })
            .collect();
        while units.last() == Some(&47) {
            units.pop();
        }
        units
    };
    let root = normalize(root);
    let path = normalize(path);
    if path == root {
        return Vec::new();
    }
    if path.starts_with(&root) && path.get(root.len()) == Some(&47) {
        return path[root.len() + 1..].to_vec();
    }
    path.into_iter().skip_while(|&u| u == 47).collect()
}
fn ignored(
    matcher: &mut Ignore,
    root: &FsPath,
    path: &FsPath,
    directory: bool,
    output: &mut LoadedSkills,
) -> bool {
    let mut relative = relative(root, path);
    if directory {
        relative.push(47);
    }
    match matcher.ignores(&relative) {
        Ok(ignored) => ignored,
        Err(()) => {
            diag(
                output,
                SkillDiagnosticCode::InvalidPath,
                "entry path cannot be matched against ignore rules",
                path,
            );
            true
        }
    }
}
fn prefix_pattern(line: &str, prefix: &[u16]) -> Option<Vec<u16>> {
    let trimmed = js_trim(line);
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let mut pattern = line;
    let negative = pattern.starts_with('!');
    if negative || pattern.starts_with(r"\!") {
        pattern = &pattern[1..];
    }
    if let Some(rest) = pattern.strip_prefix('/') {
        pattern = rest;
    }
    let mut result = Vec::new();
    if negative {
        result.push(33);
    }
    result.extend_from_slice(prefix);
    result.extend(pattern.encode_utf16());
    Some(result)
}
async fn ignore_files(
    fs: &dyn FileSystem,
    matcher: &mut Ignore,
    dir: &FsPath,
    root: &FsPath,
    output: &mut LoadedSkills,
) -> Result<(), SkillDiscoveryError> {
    let mut prefix = relative(root, dir);
    if !prefix.is_empty() {
        prefix.push(47);
    }
    for name in [".gitignore", ".ignore", ".fdignore"] {
        let name = FsPath::from(name);
        let path = match fs.join_path(&[dir, &name], None).await {
            Ok(path) => path,
            Err(error) => {
                diag(
                    output,
                    SkillDiagnosticCode::FileInfoFailed,
                    error.message,
                    dir,
                );
                continue;
            }
        };
        match fs.file_info(&path, None).await {
            Ok(info) if info.kind == FileKind::File => {}
            Ok(_) => continue,
            Err(error) => {
                info_error(output, error, &path);
                continue;
            }
        }
        let content = match fs.read_text_file(&path, None).await {
            Ok(content) => content,
            Err(error) => {
                diag(
                    output,
                    SkillDiagnosticCode::ReadFailed,
                    error.message,
                    &path,
                );
                continue;
            }
        };
        for line in content.split('\n') {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let Some(pattern) = prefix_pattern(line, &prefix) else {
                continue;
            };
            if matcher.add_units(&pattern).is_err() {
                diag(
                    output,
                    SkillDiagnosticCode::InvalidIgnorePattern,
                    "ignore pattern is not valid and was dropped",
                    &path,
                );
            }
        }
    }
    Ok(())
}

fn parse(content: &str) -> Result<(Value, String), frontmatter::Invalid> {
    let content = content.replace("\r\n", "\n").replace('\r', "\n");
    if !content.starts_with("---") {
        return Ok((Value::Mapping(Vec::new()), content));
    }
    let Some(end) = content[3..].find("\n---").map(|offset| offset + 3) else {
        return Ok((Value::Mapping(Vec::new()), content));
    };
    // JS's slice(4, end) can split a pair on a malformed opener. Such a
    // frontmatter is outside DIV-004; never silently replace its unpaired unit.
    let units: Vec<_> = content[..end].encode_utf16().collect();
    let text =
        String::from_utf16(units.get(4..).unwrap_or_default()).map_err(|_| frontmatter::Invalid)?;
    Ok((
        frontmatter::read(&text)?,
        js_trim(&content[end + 4..]).to_owned(),
    ))
}
fn field<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    match value {
        Value::Mapping(fields) => fields
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value),
        _ => None,
    }
}
fn string(value: Option<&Value>) -> Option<&str> {
    match value {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}
async fn load_file(fs: &dyn FileSystem, path: &FsPath, parent: &FsPath, output: &mut LoadedSkills) {
    let mut name = path.code_units();
    while name.last().is_some_and(|u| *u == 47 || *u == 92) {
        name = &name[..name.len() - 1];
    }
    let declared = name
        .rsplit(|&u| u == 47 || u == 92)
        .next()
        .unwrap_or_default()
        == "SKILL.md".encode_utf16().collect::<Vec<_>>();
    let content = match fs.read_text_file(path, None).await {
        Ok(content) => content,
        Err(error) => {
            diag(output, SkillDiagnosticCode::ReadFailed, error.message, path);
            return;
        }
    };
    let (frontmatter, content) = match parse(&content) {
        Ok(parsed) => parsed,
        Err(_) => {
            if declared {
                diag(
                    output,
                    SkillDiagnosticCode::ParseFailed,
                    "frontmatter is not valid in the supported YAML subset",
                    path,
                );
            }
            return;
        }
    };
    let description = string(field(&frontmatter, "description"));
    if !declared && description.is_none_or(|d| js_trim(d).is_empty()) {
        return;
    }
    if description.is_none_or(|d| js_trim(d).is_empty()) {
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            "description is required",
            path,
        );
    } else if let Some(description) = description
        && description.encode_utf16().count() > 1024
    {
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            format!(
                "description exceeds 1024 characters ({})",
                description.encode_utf16().count()
            ),
            path,
        );
    }
    let name = string(field(&frontmatter, "name"))
        .filter(|name| !name.is_empty())
        .map_or_else(|| parent.clone(), FsPath::from);
    if name != *parent {
        let mut message: Vec<_> = "name \"".encode_utf16().collect();
        message.extend_from_slice(name.code_units());
        message.extend("\" does not match parent directory \"".encode_utf16());
        message.extend_from_slice(parent.code_units());
        message.push(34);
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            FsPath::from_code_units(message),
            path,
        );
    }
    if name.code_units().len() > 64 {
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            format!("name exceeds 64 characters ({})", name.code_units().len()),
            path,
        );
    }
    if name.code_units().is_empty()
        || !name
            .code_units()
            .iter()
            .all(|&u| matches!(u,97..=122|48..=57|45))
    {
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)",
            path,
        );
    }
    if name.code_units().first() == Some(&45) || name.code_units().last() == Some(&45) {
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            "name must not start or end with a hyphen",
            path,
        );
    }
    if name.code_units().windows(2).any(|pair| pair == [45, 45]) {
        diag(
            output,
            SkillDiagnosticCode::InvalidMetadata,
            "name must not contain consecutive hyphens",
            path,
        );
    }
    if let Some(description) = description
        && !js_trim(description).is_empty()
    {
        output.skills.push(Skill {
            name,
            description: description.to_owned(),
            content,
            file_path: path.clone(),
            disable_model_invocation: field(&frontmatter, "disable-model-invocation")
                == Some(&Value::Bool(true)),
        });
    }
}

struct Frame {
    entries: std::vec::IntoIter<FileInfo>,
    parent_name: FsPath,
    root_files: bool,
}
async fn enter(
    fs: &dyn FileSystem,
    dir: &FsPath,
    root: &FsPath,
    root_files: bool,
    matcher: &mut Ignore,
    collation: &minion_agent_pinned_icu::SkillNameOrder,
    output: &mut LoadedSkills,
) -> Result<Option<Frame>, SkillDiscoveryError> {
    let info = match fs.file_info(dir, None).await {
        Ok(info) => info,
        Err(error) => {
            info_error(output, error, dir);
            return Ok(None);
        }
    };
    if kind(fs, &info, output).await != Some(FileKind::Directory) {
        return Ok(None);
    }
    ignore_files(fs, matcher, dir, root, output).await?;
    let entries = match fs.list_dir(dir, None).await {
        Ok(entries) => entries,
        Err(error) => {
            diag(output, SkillDiagnosticCode::ListFailed, error.message, dir);
            return Ok(None);
        }
    };
    for entry in &entries {
        if entry.name != "SKILL.md" || kind(fs, entry, output).await != Some(FileKind::File) {
            continue;
        }
        if ignored(matcher, root, &entry.path, false, output) {
            continue;
        }
        load_file(fs, &entry.path, &info.name, output).await;
        return Ok(None);
    }
    let order = collation
        .order(
            &entries
                .iter()
                .map(|entry| entry.name.code_units())
                .collect::<Vec<_>>(),
        )
        .map_err(SkillDiscoveryError)?;
    let mut entries: Vec<_> = entries.into_iter().map(Some).collect();
    let entries = order
        .into_iter()
        .map(|index| {
            entries[index]
                .take()
                .expect("permutation has no duplicates")
        })
        .collect::<Vec<_>>()
        .into_iter();
    Ok(Some(Frame {
        entries,
        parent_name: info.name,
        root_files,
    }))
}

pub async fn load_skills(
    fs: &dyn FileSystem,
    roots: &[FsPath],
) -> Result<LoadedSkills, SkillDiscoveryError> {
    let mut output = LoadedSkills::default();
    for root in roots {
        let info = match fs.file_info(root, None).await {
            Ok(info) => info,
            Err(error) => {
                info_error(&mut output, error, root);
                continue;
            }
        };
        if kind(fs, &info, &mut output).await != Some(FileKind::Directory) {
            continue;
        }
        let root = info.path;
        let mut matcher = Ignore::new().map_err(SkillDiscoveryError)?;
        let collation =
            minion_agent_pinned_icu::SkillNameOrder::begin().map_err(SkillDiscoveryError)?;
        let mut frames = Vec::new();
        if let Some(frame) = enter(
            fs,
            &root,
            &root,
            true,
            &mut matcher,
            &collation,
            &mut output,
        )
        .await?
        {
            frames.push(frame);
        }
        while let Some(frame) = frames.last_mut() {
            let Some(entry) = frame.entries.next() else {
                frames.pop();
                continue;
            };
            if entry.name.code_units().first() == Some(&46) || entry.name == "node_modules" {
                continue;
            }
            let root_files = frame.root_files;
            let parent_name = frame.parent_name.clone();
            let Some(kind) = kind(fs, &entry, &mut output).await else {
                continue;
            };
            if ignored(
                &mut matcher,
                &root,
                &entry.path,
                kind == FileKind::Directory,
                &mut output,
            ) {
                continue;
            }
            if kind == FileKind::Directory {
                if let Some(child) = enter(
                    fs,
                    &entry.path,
                    &root,
                    false,
                    &mut matcher,
                    &collation,
                    &mut output,
                )
                .await?
                {
                    frames.push(child);
                }
            } else if kind == FileKind::File
                && root_files
                && entry.name.code_units().ends_with(&[46, 109, 100])
            {
                load_file(fs, &entry.path, &parent_name, &mut output).await;
            }
        }
    }
    Ok(output)
}

pub async fn load_sourced_skills<S>(
    fs: &dyn FileSystem,
    inputs: &[SkillSource<Arc<S>>],
) -> Result<LoadedSourcedSkills<Arc<S>>, SkillDiscoveryError> {
    let result = load_sourced_skills_with(fs, inputs, |skill, _| {
        Ok::<_, std::convert::Infallible>(skill)
    })
    .await;
    match result {
        Ok(result) => Ok(result),
        Err(SourcedSkillError::Discovery(error)) => Err(error),
        Err(SourcedSkillError::Mapping(never)) => match never {},
    }
}
pub async fn load_sourced_skills_with<S, T, E>(
    fs: &dyn FileSystem,
    inputs: &[SkillSource<Arc<S>>],
    mut map: impl FnMut(Skill, &Arc<S>) -> Result<T, E>,
) -> Result<LoadedSourcedSkills<Arc<S>, T>, SourcedSkillError<E>> {
    let mut output = LoadedSourcedSkills {
        skills: Vec::new(),
        diagnostics: Vec::new(),
    };
    for input in inputs {
        let loaded = load_skills(fs, std::slice::from_ref(&input.path)).await?;
        for skill in loaded.skills {
            output.skills.push(SourcedSkill {
                skill: map(skill, &input.source).map_err(SourcedSkillError::Mapping)?,
                source: Arc::clone(&input.source),
            });
        }
        output
            .diagnostics
            .extend(
                loaded
                    .diagnostics
                    .into_iter()
                    .map(|diagnostic| SourcedSkillDiagnostic {
                        diagnostic,
                        source: Arc::clone(&input.source),
                    }),
            );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_ignore_path_emits_one_diagnostic_and_skips_only_that_entry() {
        let mut matcher = Ignore::new().unwrap();
        let mut output = LoadedSkills::default();
        let root = FsPath::from("root");
        assert!(ignored(&mut matcher, &root, &root, false, &mut output));
        assert_eq!(output.diagnostics.len(), 1);
        assert_eq!(output.diagnostics[0].code, SkillDiagnosticCode::InvalidPath);
        assert_eq!(
            output.diagnostics[0].message,
            "entry path cannot be matched against ignore rules"
        );
        assert!(!ignored(
            &mut matcher,
            &root,
            &FsPath::from("root/good"),
            false,
            &mut output
        ));
        assert_eq!(output.diagnostics.len(), 1);
    }
}
