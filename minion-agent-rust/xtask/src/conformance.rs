use std::ffi::OsString;
use std::path::Path;

use anyhow::{Context, bail};

const FAMILIES: [&str; 4] = ["schema", "runtime", "session", "agent"];

pub fn verify_layout(workspace: &Path) -> anyhow::Result<()> {
    let root = workspace
        .parent()
        .context("Rust workspace must be nested below the monorepo root")?;
    let private = workspace.join("conformance");
    let canonical = root.join("conformance");
    let manifest = root.join("pi-parity-manifest.yaml");

    if private.exists() {
        bail!(
            "private conformance snapshot is forbidden: {}",
            private.display()
        );
    }
    if !manifest.is_file() {
        bail!("missing root parity manifest: {}", manifest.display());
    }
    for family in FAMILIES {
        let path = canonical.join(family);
        if !path.is_dir() {
            bail!(
                "missing canonical conformance directory: {}",
                path.display()
            );
        }
    }
    Ok(())
}

pub fn run_cli(args: &[OsString]) -> anyhow::Result<()> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest directory has a workspace parent");

    match args {
        [command] if command == "verify" => verify_layout(workspace),
        _ => bail!("usage: xtask conformance verify"),
    }
}
