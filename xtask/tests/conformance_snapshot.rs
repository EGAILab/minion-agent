use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use xtask::conformance::{build_manifest, sync_snapshot, verify_snapshot};

struct Fixture {
    root: PathBuf,
    snapshot: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("xtask-conformance-source-{nonce}"));
        let snapshot = std::env::temp_dir().join(format!("xtask-conformance-snapshot-{nonce}"));
        fs::create_dir_all(root.join("conformance/schema")).expect("fixture schema directory");
        fs::create_dir_all(root.join("conformance/runtime")).expect("fixture runtime directory");
        fs::write(
            root.join("conformance/schema/runtime-scenario.schema.json"),
            b"{\"type\":\"object\"}\n",
        )
        .expect("fixture schema file");
        fs::write(
            root.join("conformance/runtime/example.yaml"),
            b"name: example\n",
        )
        .expect("fixture runtime file");
        Self { root, snapshot }
    }

    fn source(&self) -> &Path {
        &self.root
    }

    fn snapshot(&self) -> PathBuf {
        self.snapshot.clone()
    }

    fn initialize_git(&self) {
        for arguments in [
            vec!["init"],
            vec!["config", "core.autocrlf", "false"],
            vec!["add", "conformance"],
            vec![
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-m",
                "fixture",
            ],
        ] {
            let status = Command::new("git")
                .arg("-c")
                .arg("safe.directory=E:/AI/Projects/OpenMinds/Minions/Minion-Agent/minion-agent-rust")
                .arg("-C")
                .arg(&self.root)
                .args(arguments)
                .status()
                .expect("fixture git command should run");
            assert!(status.success(), "fixture git command should succeed");
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("fixture cleanup");
        if self.snapshot.exists() {
            fs::remove_dir_all(&self.snapshot).expect("snapshot cleanup");
        }
    }
}

fn prepare_snapshot(fixture: &Fixture) -> PathBuf {
    let snapshot = fixture.snapshot();
    fs::create_dir_all(snapshot.join("schema")).expect("snapshot schema directory");
    fs::create_dir_all(snapshot.join("runtime")).expect("snapshot runtime directory");
    fs::copy(
        fixture
            .source()
            .join("conformance/schema/runtime-scenario.schema.json"),
        snapshot.join("schema/runtime-scenario.schema.json"),
    )
    .expect("copy schema file");
    fs::copy(
        fixture.source().join("conformance/runtime/example.yaml"),
        snapshot.join("runtime/example.yaml"),
    )
    .expect("copy runtime file");

    fs::write(
        snapshot.join("SOURCE.json"),
        concat!(
            "{\n",
            "  \"source_repository\": \"minion-agent-python/conformance\",\n",
            "  \"source_commit\": \"abc123\",\n",
            "  \"files\": [\n",
            "    {\"path\": \"runtime/example.yaml\", \"sha256\": \"15fcc3870625980bf58f15ba904736b4ffa1a84495a8f4f51d781e211016e743\"},\n",
            "    {\"path\": \"schema/runtime-scenario.schema.json\", \"sha256\": \"9091a8164f97eaca182b3d06d0e5a59e923c880ebc0148056c453c651f5b46cb\"}\n",
            "  ]\n",
            "}\n"
        ),
    )
    .expect("write fixture manifest");
    snapshot
}

#[test]
fn build_manifest_sorts_and_hashes_conformance_files() {
    let fixture = Fixture::new();

    let manifest = build_manifest(fixture.source(), "abc123").expect("build fixture manifest");

    assert_eq!(manifest.source_commit, "abc123");
    assert_eq!(manifest.files.len(), 2);
    assert!(manifest.files[0].path < manifest.files[1].path);
    assert!(manifest.files.iter().all(|file| file.sha256.len() == 64));
}

#[test]
fn verify_snapshot_reports_changed_missing_and_unrecorded_files() {
    let fixture = Fixture::new();
    let snapshot = prepare_snapshot(&fixture);
    fs::write(
        snapshot.join("schema/runtime-scenario.schema.json"),
        b"{\"type\":\"array\"}\n",
    )
    .expect("change snapshot schema");
    fs::remove_file(snapshot.join("runtime/example.yaml")).expect("remove snapshot runtime file");
    fs::write(snapshot.join("unrecorded.yaml"), b"name: extra\n").expect("add unrecorded file");

    let error = verify_snapshot(&snapshot).expect_err("snapshot should not verify");
    let message = format!("{error:#}");

    assert!(message.contains("changed file: schema/runtime-scenario.schema.json"));
    assert!(message.contains("missing file: runtime/example.yaml"));
    assert!(message.contains("unrecorded file: unrecorded.yaml"));
}

#[test]
fn sync_snapshot_copies_the_canonical_files_without_deleting_unrelated_files() {
    let fixture = Fixture::new();
    fixture.initialize_git();
    let snapshot = fixture.snapshot();
    fs::create_dir_all(&snapshot).expect("snapshot directory");
    fs::write(snapshot.join("unrelated.json"), b"{}\n").expect("unrelated snapshot file");

    sync_snapshot(fixture.source(), &snapshot).expect("sync fixture snapshot");

    assert!(
        snapshot
            .join("schema/runtime-scenario.schema.json")
            .is_file()
    );
    assert!(snapshot.join("runtime/example.yaml").is_file());
    assert!(snapshot.join("SOURCE.json").is_file());
    assert!(snapshot.join("unrelated.json").is_file());
}
