//! WP-14.2: exact scalar-value prompt text, composed over the request's tool
//! snapshot. No registry, filesystem, tool execution or prompt DSL lives here.
use std::{collections::HashSet, sync::Arc};

use parking_lot::RwLock;

use crate::{
    agent_loop::{PromptAssembler, PromptAssemblyError},
    javascript::js_trim,
    skills::Skill,
    tools::ToolDefinition,
};

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// HAR-002. Input fields are Unicode scalar-value strings (WP-14.2 domain).
/// FsPath's existing display is exact in that domain; no native-path projection
/// or path normalization is performed.
pub fn format_skills_block(skills: &[Skill]) -> String {
    let visible: Vec<_> = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect();
    if visible.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "The following skills provide specialized instructions for specific tasks.".to_owned(),
        "Read the full skill file when the task matches its description.".to_owned(),
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_owned(),
        String::new(),
        "<available_skills>".to_owned(),
    ];
    for skill in visible {
        lines.push("  <skill>".to_owned());
        lines.push(format!(
            "    <name>{}</name>",
            escape_xml(&skill.name.to_string())
        ));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path.to_string())
        ));
        lines.push("  </skill>".to_owned());
    }
    lines.push("</available_skills>".to_owned());
    lines.join("\n")
}

fn dirname(path: &crate::execution::FsPath) -> String {
    let units = path.code_units();
    let end = units
        .iter()
        .rposition(|unit| !matches!(unit, 47 | 92))
        .map_or(0, |i| i + 1);
    let units = &units[..end];
    let index = units.iter().rposition(|unit| matches!(unit, 47 | 92));
    let end = match index {
        Some(2) if units.get(1) == Some(&58) => 3,
        Some(i) if i > 0 => i,
        _ => return "/".to_owned(),
    };
    crate::execution::FsPath::from_code_units(units[..end].to_vec()).to_string()
}

/// HAR-014. No escaping; application code owns sending the invocation message.
pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let mut text = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        dirname(&skill.file_path),
        skill.content
    );
    if let Some(additional) = additional_instructions.filter(|text| !text.is_empty()) {
        text.push_str("\n\n");
        text.push_str(additional);
    }
    text
}

fn snippet(text: &str) -> String {
    // js_trim is the certified ECMAScript whitespace predicate, not Rust's
    // broader Unicode White_Space. A singleton trimmed to empty is whitespace.
    let mut output = String::new();
    let mut pending_space = false;
    for c in text.chars() {
        let mut buffer = [0; 4];
        if js_trim(c.encode_utf8(&mut buffer)).is_empty() {
            pending_space = !output.is_empty();
        } else {
            if pending_space {
                output.push(' ');
            }
            output.push(c);
            pending_space = false;
        }
    }
    output
}

/// HAR-018: opt-in metadata projection, input tool order and first guideline
/// occurrence preserved. This does not inspect or alter any tool schema.
pub fn format_tools_section(tools: &[Arc<ToolDefinition>]) -> String {
    let mut available = Vec::new();
    let mut guidelines = Vec::new();
    let mut seen = HashSet::new();
    for tool in tools {
        if let Some(value) = tool.prompt_snippet() {
            let value = snippet(value);
            if !value.is_empty() {
                available.push(format!("- {}: {value}", tool.name()));
            }
        }
        for value in tool.prompt_guidelines().unwrap_or_default() {
            let value = js_trim(value);
            if !value.is_empty() && seen.insert(value.to_owned()) {
                guidelines.push(format!("- {value}"));
            }
        }
    }
    let mut blocks = Vec::new();
    if !available.is_empty() {
        blocks.push(format!("Available tools:\n{}", available.join("\n")));
    }
    if !guidelines.is_empty() {
        blocks.push(format!("Guidelines:\n{}", guidelines.join("\n")));
    }
    blocks.join("\n\n")
}

/// Membership is copied on supply. Skill handles remain shared and writable;
/// mutate records/configuration only between request builds (HAR-016).
#[derive(Clone, Debug, Default)]
pub struct PromptConfiguration {
    pub skills: Vec<Arc<RwLock<Skill>>>,
    pub tools_section: bool,
    pub sections: Vec<String>,
}

/// One whole configuration value is captured at the start of each assembly.
/// It implements the certified Layer-08 seam, never re-reading ctx.tools.
#[derive(Debug, Default)]
pub struct SystemPromptComposer {
    configuration: RwLock<Arc<PromptConfiguration>>,
}

impl SystemPromptComposer {
    pub fn new(configuration: &PromptConfiguration) -> Self {
        Self {
            configuration: RwLock::new(Arc::new(configuration.clone())),
        }
    }

    pub fn replace_configuration(&self, configuration: &PromptConfiguration) {
        *self.configuration.write() = Arc::new(configuration.clone());
    }

    pub fn compose(&self, base: &str, tools: &[Arc<ToolDefinition>]) -> String {
        let configuration = self.configuration.read().clone();
        let mut sections = vec![base.to_owned()];
        if configuration.tools_section {
            sections.push(format_tools_section(tools));
        }
        sections.extend(configuration.sections.iter().cloned());
        if tools.iter().any(|tool| tool.name() == "read") {
            // Clone only this synchronous field snapshot, not the retained
            // records at configuration supply. No lower-layer calls under locks.
            let skills: Vec<_> = configuration
                .skills
                .iter()
                .map(|skill| skill.read().clone())
                .collect();
            sections.push(format_skills_block(&skills));
        }
        sections
            .into_iter()
            .filter(|section| !section.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl PromptAssembler for SystemPromptComposer {
    fn assemble(
        &self,
        base: &str,
        tools: &[Arc<ToolDefinition>],
    ) -> Result<String, PromptAssemblyError> {
        Ok(self.compose(base, tools))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirname_uses_utf16_not_scalar_indices() {
        assert_eq!(dirname(&"😀:\\b.md".into()), "😀:");
        assert_eq!(dirname(&"C:\\b.md".into()), "C:\\");
        assert_eq!(dirname(&"😀/b.md".into()), "😀");
    }

    #[test]
    fn metadata_is_additive_and_uses_js_not_rust_whitespace() {
        let plain = ToolDefinition::new(
            "read",
            "description",
            serde_json::from_str("{}").unwrap(),
            "read",
            |_| Box::pin(async { unreachable!() }),
        );
        let enriched = plain
            .clone()
            .with_prompt_snippet("\u{feff}a\u{85}b\u{feff}")
            .with_prompt_guidelines(vec!["\u{feff}g\u{feff}".into()]);
        assert_eq!(plain.schema().unwrap(), enriched.schema().unwrap());
        assert_eq!(plain.name(), enriched.name());
        assert_eq!(
            format_tools_section(&[Arc::new(enriched)]),
            "Available tools:\n- read: a\u{85}b\n\nGuidelines:\n- g"
        );
    }
}
