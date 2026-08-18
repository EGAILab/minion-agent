use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MANIFEST_FILE: &str = "SOURCE.json";
const SOURCE_REPOSITORY: &str = "minion-agent-python/conformance";
const RUST_REPOSITORY: &str = "E:/AI/Projects/OpenMinds/Minions/Minion-Agent/minion-agent-rust";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotManifest {
    pub source_repository: String,
    pub source_commit: String,
    pub files: Vec<SnapshotFile>,
}

pub fn build_manifest(source: &Path, commit: &str) -> anyhow::Result<SnapshotManifest> {
    let files = snapshot_files(&source.join("conformance"))?;
    Ok(SnapshotManifest {
        source_repository: SOURCE_REPOSITORY.to_owned(),
        source_commit: commit.to_owned(),
        files,
    })
}

pub fn sync_snapshot(source: &Path, destination: &Path) -> anyhow::Result<()> {
    ensure_clean_source(source)?;
    let commit = source_commit(source)?;
    let manifest = build_manifest(source, &commit)?;
    let previous = read_manifest_if_present(destination)?;

    fs::create_dir_all(destination)
        .with_context(|| format!("create snapshot directory {}", destination.display()))?;

    let previous_paths: BTreeSet<_> = previous
        .as_ref()
        .map(|manifest| {
            manifest
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect()
        })
        .unwrap_or_default();
    let next_paths: BTreeSet<_> = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();

    for path in previous_paths.difference(&next_paths) {
        let destination_file = snapshot_path(destination, path)?;
        if destination_file.exists() {
            fs::remove_file(&destination_file).with_context(|| {
                format!("remove stale snapshot file {}", destination_file.display())
            })?;
        }
    }

    for file in &manifest.files {
        let relative_path = snapshot_path(Path::new(""), &file.path)?;
        let source_file = source.join("conformance").join(&relative_path);
        let destination_file = snapshot_path(destination, &file.path)?;
        let contents = fs::read(&source_file)
            .with_context(|| format!("read source snapshot file {}", source_file.display()))?;
        if let Some(parent) = destination_file.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create snapshot directory {}", parent.display()))?;
        }
        fs::write(&destination_file, contents)
            .with_context(|| format!("write snapshot file {}", destination_file.display()))?;
    }

    write_manifest(destination, &manifest)
}

pub fn verify_snapshot(destination: &Path) -> anyhow::Result<()> {
    let manifest = read_manifest(destination)?;
    let actual = snapshot_files(destination)?;
    let expected_by_path: BTreeMap<_, _> = manifest
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.sha256.as_str()))
        .collect();
    let actual_by_path: BTreeMap<_, _> = actual
        .iter()
        .map(|file| (file.path.as_str(), file.sha256.as_str()))
        .collect();
    let mut mismatches = Vec::new();

    for (path, expected_hash) in &expected_by_path {
        match actual_by_path.get(path) {
            None => mismatches.push(format!("missing file: {path}")),
            Some(actual_hash) if actual_hash != expected_hash => {
                mismatches.push(format!("changed file: {path}"));
            }
            Some(_) => {}
        }
    }
    for path in actual_by_path.keys() {
        if !expected_by_path.contains_key(path) {
            mismatches.push(format!("unrecorded file: {path}"));
        }
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        bail!("snapshot verification failed:\n{}", mismatches.join("\n"));
    }
}

pub fn run_cli(args: &[OsString]) -> anyhow::Result<()> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest directory has a workspace parent");
    let destination = workspace.join("conformance");

    match args {
        [command, flag, source] if command == "sync" && flag == "--source" => {
            sync_snapshot(Path::new(source), &destination)
        }
        [command] if command == "verify" => verify_snapshot(&destination),
        _ => bail!("usage: xtask conformance sync --source <path> | verify"),
    }
}

fn snapshot_files(root: &Path) -> anyhow::Result<Vec<SnapshotFile>> {
    let mut paths = Vec::new();
    collect_snapshot_paths(root, root, &mut paths)?;
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let source_file = root.join(&path);
            let bytes = fs::read(&source_file)
                .with_context(|| format!("read snapshot file {}", source_file.display()))?;
            Ok(SnapshotFile {
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            })
        })
        .collect()
}

fn collect_snapshot_paths(
    root: &Path,
    directory: &Path,
    paths: &mut Vec<String>,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("read snapshot directory {}", directory.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", directory.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_snapshot_paths(root, &path, paths)?;
            continue;
        }
        if !path.is_file()
            || !matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("json" | "yaml")
            )
        {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .expect("walked path remains below snapshot root");
        let relative = slash_normalized(relative)?;
        if relative != MANIFEST_FILE {
            paths.push(relative);
        }
    }
    Ok(())
}

fn slash_normalized(path: &Path) -> anyhow::Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(
                part.to_str()
                    .context("snapshot path must be valid UTF-8")?
                    .to_owned(),
            ),
            Component::CurDir => {}
            _ => bail!("snapshot path must be relative: {}", path.display()),
        }
    }
    Ok(parts.join("/"))
}

fn snapshot_path(root: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!("manifest path must be relative: {relative}");
    }
    Ok(root.join(path))
}

fn read_manifest_if_present(destination: &Path) -> anyhow::Result<Option<SnapshotManifest>> {
    let path = destination.join(MANIFEST_FILE);
    if !path.exists() {
        return Ok(None);
    }
    read_manifest(destination).map(Some)
}

fn read_manifest(destination: &Path) -> anyhow::Result<SnapshotManifest> {
    let path = destination.join(MANIFEST_FILE);
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("read snapshot manifest {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("parse snapshot manifest {}", path.display()))
}

fn write_manifest(destination: &Path, manifest: &SnapshotManifest) -> anyhow::Result<()> {
    let path = destination.join(MANIFEST_FILE);
    let mut contents =
        serde_json::to_string_pretty(manifest).context("serialize snapshot manifest")?;
    contents.push('\n');
    fs::write(&path, contents)
        .with_context(|| format!("write snapshot manifest {}", path.display()))
}

fn ensure_clean_source(source: &Path) -> anyhow::Result<()> {
    let output = git(source, ["status", "--porcelain"])?;
    if output.trim().is_empty() {
        Ok(())
    } else {
        bail!("source checkout is dirty");
    }
}

fn source_commit(source: &Path) -> anyhow::Result<String> {
    Ok(git(source, ["rev-parse", "HEAD"])?.trim().to_owned())
}

fn git<const N: usize>(source: &Path, arguments: [&str; N]) -> anyhow::Result<String> {
    let output = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={RUST_REPOSITORY}"))
        .arg("-C")
        .arg(source)
        .args(arguments)
        .output()
        .with_context(|| format!("run git in {}", source.display()))?;
    if !output.status.success() {
        bail!(
            "git command failed in {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("git returned non-UTF-8 output")
}
