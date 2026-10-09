#![cfg(feature = "conformance")]

use minion_agent::{
    skills::Skill,
    system_prompt::{
        PromptConfiguration, SystemPromptComposer, format_skill_invocation, format_skills_block,
        format_tools_section,
    },
    tools::ToolDefinition,
};
use parking_lot::RwLock;
use serde::Deserialize;
use std::{fs, path::PathBuf, sync::Arc};

#[derive(Deserialize)]
struct Document {
    prompt_assembly: Case,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Case {
    SkillsBlock { input: Skills, expected: String },
    Invocation { input: Invocation, expected: String },
    ToolsSection { input: Tools, expected: String },
    Compose { input: Compose, expected: String },
}
#[derive(Deserialize)]
struct Skills {
    skills: Vec<Skill>,
}
#[derive(Deserialize)]
struct Invocation {
    skill: Skill,
    additional_instructions: Option<String>,
}
#[derive(Deserialize)]
struct Tools {
    tools: Vec<ToolInput>,
}
#[derive(Deserialize)]
struct Compose {
    base: String,
    tools: Vec<ToolInput>,
    tools_section: bool,
    sections: Vec<String>,
    skills: Vec<Skill>,
}
#[derive(Deserialize)]
struct ToolInput {
    name: String,
    snippet: Option<String>,
    guidelines: Option<Vec<String>>,
}

fn tools(inputs: Vec<ToolInput>) -> Vec<Arc<ToolDefinition>> {
    inputs
        .into_iter()
        .map(|input| {
            let mut tool = ToolDefinition::new(
                input.name,
                "",
                serde_json::from_str("{}").unwrap(),
                "",
                |_| Box::pin(async { unreachable!("formatting never executes tools") }),
            );
            if let Some(snippet) = input.snippet {
                tool = tool.with_prompt_snippet(snippet);
            }
            if let Some(guidelines) = input.guidelines {
                tool = tool.with_prompt_guidelines(guidelines);
            }
            Arc::new(tool)
        })
        .collect()
}

// serde_json rejects lone surrogate escapes anywhere, including ignored
// pi_reference values and object keys, while combining valid escaped pairs.
fn preflight(text: &str) -> serde_json::Result<serde_json::Value> {
    serde_json::from_str(text)
}

fn run_kind(kind: &str) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../conformance");
    let schema: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("schema/prompt-assembly-scenario.schema.json")).unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mut paths: Vec<_> = fs::read_dir(root.join("agent/prompt-assembly"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    let mut selected = 0;
    for path in paths {
        let text = fs::read_to_string(&path).unwrap();
        let value = preflight(&text).unwrap();
        assert!(validator.is_valid(&value), "schema: {}", path.display());
        if value["prompt_assembly"]["kind"] != kind {
            continue;
        }
        selected += 1;
        let document: Document = serde_json::from_value(value).unwrap();
        let (actual, expected) = match document.prompt_assembly {
            Case::SkillsBlock { input, expected } => (format_skills_block(&input.skills), expected),
            Case::Invocation { input, expected } => (
                format_skill_invocation(&input.skill, input.additional_instructions.as_deref()),
                expected,
            ),
            Case::ToolsSection { input, expected } => {
                (format_tools_section(&tools(input.tools)), expected)
            }
            Case::Compose { input, expected } => {
                let composer = SystemPromptComposer::new(&PromptConfiguration {
                    skills: input
                        .skills
                        .into_iter()
                        .map(|skill| Arc::new(RwLock::new(skill)))
                        .collect(),
                    tools_section: input.tools_section,
                    sections: input.sections,
                });
                (composer.compose(&input.base, &tools(input.tools)), expected)
            }
        };
        assert_eq!(
            actual,
            expected,
            "canonical byte mismatch: {}",
            path.display()
        );
    }
    assert!(selected > 0, "empty canonical selection");
    println!("{kind}: {selected} canonical documents PASS");
}

#[test]
fn canonical_skills_block() {
    run_kind("skills_block");
}
#[test]
fn canonical_invocation() {
    run_kind("invocation");
}
#[test]
fn canonical_tools_section() {
    run_kind("tools_section");
}
#[test]
fn canonical_compose() {
    run_kind("compose");
}

#[test]
fn scalar_domain_preflight() {
    for text in [
        r#"{"input":"\ud800"}"#,
        r#"{"expected":"\udfff"}"#,
        r#"{"\udc00":"value"}"#,
        r#"{"pi_reference":"\ud800"}"#,
    ] {
        assert!(
            preflight(text).is_err(),
            "unpaired surrogate must fail the document"
        );
    }
    assert_eq!(
        preflight(r#"{"text":"\ud83d\ude00"}"#).unwrap()["text"],
        "😀"
    );
    assert!(preflight(r#"{"text":"😀"}"#).is_ok());
}
