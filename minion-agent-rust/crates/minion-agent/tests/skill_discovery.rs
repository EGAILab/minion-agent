use minion_agent::{
    execution::{FsPath, LocalFileSystem},
    skills::{SkillDiagnosticCode, SkillSource, load_skills, load_sourced_skills_with},
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

fn code(code: SkillDiagnosticCode) -> &'static str {
    match code {
        SkillDiagnosticCode::FileInfoFailed => "file_info_failed",
        SkillDiagnosticCode::ListFailed => "list_failed",
        SkillDiagnosticCode::ReadFailed => "read_failed",
        SkillDiagnosticCode::ParseFailed => "parse_failed",
        SkillDiagnosticCode::InvalidMetadata => "invalid_metadata",
        SkillDiagnosticCode::InvalidPath => "invalid_path",
        SkillDiagnosticCode::InvalidIgnorePattern => "invalid_ignore_pattern",
    }
}
fn addressed(root: &Path, path: &FsPath) -> String {
    let path = path.as_str().expect("scalar canonical fixture path");
    let relative = Path::new(path)
        .strip_prefix(root)
        .expect("addressed fixture subtree")
        .to_str()
        .unwrap();
    if cfg!(windows) {
        relative.replace('\\', "/")
    } else {
        relative.to_owned()
    }
}
fn fixtures(root: &Path, entries: &[Value]) {
    for entry in entries {
        let path = root.join(entry["path"].as_str().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        if entry.get("dir").is_some() {
            std::fs::create_dir_all(path).unwrap();
        } else if let Some(text) = entry["text"].as_str() {
            std::fs::write(path, text.as_bytes()).unwrap();
        } else {
            let target = entry["symlink"].as_str().unwrap();
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, path).unwrap();
            #[cfg(windows)]
            {
                let target = target.replace('/', "\\");
                if entry["symlink_kind"] == "dir" {
                    std::os::windows::fs::symlink_dir(target, path).unwrap();
                } else {
                    std::os::windows::fs::symlink_file(target, path).unwrap();
                }
            }
        }
    }
}
#[tokio::test]
async fn canonical_skill_discovery() {
    let corpus =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/agent/skill-discovery");
    let mut files: Vec<_> = std::fs::read_dir(corpus)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 96);
    let mut passed = 0;
    let mut skipped = 0;
    let selected = std::env::var("WP141_SCENARIO").ok();
    for file in files {
        if selected
            .as_ref()
            .is_some_and(|name| file.file_stem().unwrap() != name.as_str())
        {
            continue;
        }
        let document: Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        println!("WP-14.1 scenario: {}", document["name"]);
        let scenario = &document["skill_discovery"];
        if cfg!(windows) && scenario["posix_only"] == true {
            skipped += 1;
            continue;
        }
        let temp = tempfile::tempdir().unwrap();
        fixtures(temp.path(), scenario["fixture"].as_array().unwrap());
        let fs = LocalFileSystem::new(temp.path());
        let roots: Vec<_> = scenario["roots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|root| FsPath::from(temp.path().join(root.as_str().unwrap()).to_str().unwrap()))
            .collect();
        let actual = load_skills(&fs, &roots).await.unwrap();
        let skills:Vec<_>=actual.skills.iter().map(|skill|json!({"name":skill.name,"description":skill.description,"content":skill.content,"path":addressed(temp.path(),&skill.file_path),"disable_model_invocation":skill.disable_model_invocation})).collect();
        assert_eq!(
            json!(skills),
            scenario["expect"]["skills"],
            "{} skills",
            document["name"]
        );
        let expected = scenario["expect"]["diagnostics"].as_array().unwrap();
        assert_eq!(
            actual.diagnostics.len(),
            expected.len(),
            "{} diagnostics {:?}",
            document["name"],
            actual.diagnostics
        );
        for (actual, expected) in actual.diagnostics.iter().zip(expected) {
            let path = addressed(temp.path(), &actual.path);
            if let Some(prefix) = expected["path_within"].as_str() {
                assert!(
                    path.starts_with(&format!("{prefix}/")),
                    "{}: {path}",
                    document["name"]
                );
                assert!(
                    expected["code_one_of"]
                        .as_array()
                        .unwrap()
                        .contains(&json!(code(actual.code)))
                );
            } else {
                assert_eq!(
                    code(actual.code),
                    expected["code"].as_str().unwrap(),
                    "{}",
                    document["name"]
                );
                assert_eq!(
                    path,
                    expected["path"].as_str().unwrap(),
                    "{}",
                    document["name"]
                );
                if let Some(message) = expected["message"].as_str() {
                    assert_eq!(actual.message, message, "{}", document["name"]);
                }
            }
        }
        passed += 1;
    }
    println!("WP-14.1 canonical: {passed} passed, {skipped} explicit platform skips");
    assert!(
        passed + skipped > 0,
        "no selected canonical document exists"
    );
}

#[test]
fn raw_name_collation_is_not_lowercased_or_lexical() {
    let names: Vec<Vec<u16>> = ["b", "A", "a", "a"]
        .iter()
        .map(|name| name.encode_utf16().collect())
        .collect();
    assert_eq!(
        minion_agent_pinned_icu::skill_name_order(
            &names.iter().map(Vec::as_slice).collect::<Vec<_>>()
        )
        .unwrap(),
        [2, 3, 1, 0]
    );
}

#[tokio::test]
async fn sourced_records_are_writable_and_preserve_opaque_source_identity() {
    let temp = tempfile::tempdir().unwrap();
    fixtures(
        temp.path(),
        &[
            json!({"path":"skills/a/SKILL.md","text":"---\nname: a\ndescription: Description.\n---\nBody."}),
        ],
    );
    let source = Arc::new(vec![1, 2]);
    let input = SkillSource {
        path: FsPath::from(temp.path().join("skills").to_str().unwrap()),
        source: Arc::clone(&source),
    };
    let fs = LocalFileSystem::new(temp.path());
    let mut mapping_calls = 0;
    let mut loaded = load_sourced_skills_with(&fs, &[input], |mut skill, observed| {
        mapping_calls += 1;
        assert!(Arc::ptr_eq(observed, &source));
        skill.description = "edited".into();
        Ok::<_, std::convert::Infallible>(skill)
    })
    .await
    .unwrap();
    assert_eq!(mapping_calls, 1);
    assert!(Arc::ptr_eq(&loaded.skills[0].source, &source));
    assert_eq!(loaded.skills[0].skill.description, "edited");
    loaded.skills[0].skill.content = "writable".into();
    assert_eq!(loaded.skills[0].skill.content, "writable");
}

#[tokio::test]
async fn walk_has_no_binding_depth_bound() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("skills");
    let mut leaf = root.clone();
    for _ in 0..80 {
        leaf.push("a");
        std::fs::create_dir_all(&leaf).unwrap();
    }
    std::fs::write(
        leaf.join("SKILL.md"),
        "---\nname: a\ndescription: Deep.\n---\nBody.",
    )
    .unwrap();
    let result = load_skills(
        &LocalFileSystem::new(temp.path()),
        &[FsPath::from(root.to_str().unwrap())],
    )
    .await
    .unwrap();
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.skills.len(), 1);
}

#[tokio::test]
async fn filesystem_origin_names_and_diagnostic_interpolation_preserve_units() {
    let temp = tempfile::tempdir().unwrap();
    fixtures(
        temp.path(),
        &[
            json!({"path":"�/SKILL.md","text":"---\nname: a\ndescription: Description.\n---\nBody."}),
        ],
    );
    let mut root: Vec<_> = temp.path().to_str().unwrap().encode_utf16().collect();
    root.extend([47, 0xd800]);
    let result = load_skills(
        &LocalFileSystem::new(temp.path()),
        &[FsPath::from_code_units(root)],
    )
    .await
    .unwrap();
    assert_eq!(result.skills.len(), 1);
    assert!(result.skills[0].file_path.code_units().contains(&0xd800));
    assert_eq!(result.diagnostics.len(), 1);
    assert!(result.diagnostics[0].message.code_units().contains(&0xd800));
    assert!(!result.diagnostics[0].message.code_units().contains(&0xfffd));
}

#[tokio::test]
async fn application_mapping_failure_is_not_a_diagnostic() {
    let temp = tempfile::tempdir().unwrap();
    fixtures(
        temp.path(),
        &[json!({"path":"skills/SKILL.md","text":"---\ndescription: Description.\n---\nBody."})],
    );
    let inputs = [SkillSource {
        path: FsPath::from(temp.path().join("skills").to_str().unwrap()),
        source: Arc::new(()),
    }];
    let result = load_sourced_skills_with(&LocalFileSystem::new(temp.path()), &inputs, |_, _| {
        Err::<(), _>("mapping failure")
    })
    .await;
    assert!(matches!(
        result,
        Err(minion_agent::skills::SourcedSkillError::Mapping(
            "mapping failure"
        ))
    ));
}

#[cfg(unix)]
#[tokio::test]
async fn deep_acyclic_walk_and_ignore_parents_do_not_have_a_depth_limit() {
    for (depth, leaf_ignore) in [100, 950, 1050]
        .into_iter()
        .flat_map(|depth| [false, true].map(|leaf_ignore| (depth, leaf_ignore)))
    {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("skills");
        std::fs::create_dir(&root).unwrap();
        let mut leaf = root.clone();
        for _ in 0..depth {
            leaf.push("a");
            std::fs::create_dir(&leaf).unwrap();
        }
        if leaf_ignore {
            std::fs::write(leaf.join(".gitignore"), "ignored\n").unwrap();
        }
        std::fs::write(
            leaf.join("SKILL.md"),
            "---\nname: a\ndescription: Deep.\n---\nBody.",
        )
        .unwrap();
        fixtures(
            temp.path(),
            &[
                json!({"path":"good/SKILL.md","text":"---\nname: good\ndescription: Later root.\n---\nBody."}),
            ],
        );
        let roots = [
            FsPath::from(root.to_str().unwrap()),
            FsPath::from(temp.path().join("good").to_str().unwrap()),
        ];
        let actual = load_skills(&LocalFileSystem::new(temp.path()), &roots)
            .await
            .unwrap();
        assert!(
            actual.diagnostics.is_empty(),
            "{depth}: {:?}",
            actual.diagnostics
        );
        assert_eq!(
            actual
                .skills
                .iter()
                .map(|skill| skill.name.as_str().unwrap())
                .collect::<Vec<_>>(),
            ["a", "good"]
        );
    }
}
