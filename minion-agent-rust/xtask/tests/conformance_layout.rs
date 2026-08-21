use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use xtask::conformance::verify_layout;

struct Layout {
    root: PathBuf,
    workspace: PathBuf,
}

impl Layout {
    fn canonical() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("xtask-conformance-layout-{nonce}"));
        let workspace = root.join("minion-agent-rust");

        fs::create_dir_all(&workspace).expect("Rust workspace fixture");
        fs::write(root.join("pi-parity-manifest.yaml"), b"schema_version: 1\n")
            .expect("parity manifest fixture");
        for family in ["schema", "runtime", "session", "agent"] {
            fs::create_dir_all(root.join("conformance").join(family))
                .expect("canonical conformance family fixture");
        }

        Self { root, workspace }
    }

    fn workspace(&self) -> &Path {
        &self.workspace
    }
}

impl Drop for Layout {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("layout fixture cleanup");
    }
}

#[test]
fn accepts_the_single_root_contract_layout() {
    let layout = Layout::canonical();

    verify_layout(layout.workspace()).expect("canonical monorepo layout should verify");
}

#[test]
fn rejects_a_missing_canonical_family() {
    let layout = Layout::canonical();
    fs::remove_dir(layout.root.join("conformance/agent")).expect("remove agent family fixture");

    let error = verify_layout(layout.workspace()).expect_err("missing family should fail");

    assert!(error.to_string().contains("agent"));
}

#[test]
fn rejects_a_private_rust_conformance_snapshot() {
    let layout = Layout::canonical();
    fs::create_dir(layout.workspace.join("conformance")).expect("private snapshot fixture");

    let error = verify_layout(layout.workspace()).expect_err("private snapshot should fail");

    assert!(error.to_string().contains("private conformance"));
}

#[test]
fn rejects_a_missing_root_parity_manifest() {
    let layout = Layout::canonical();
    fs::remove_file(layout.root.join("pi-parity-manifest.yaml"))
        .expect("remove parity manifest fixture");

    let error = verify_layout(layout.workspace()).expect_err("missing manifest should fail");

    assert!(error.to_string().contains("pi-parity-manifest.yaml"));
}
