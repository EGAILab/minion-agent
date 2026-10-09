"""WP-14.2 controls, only in a disposable full code copy (--tree Rust root).

Baseline proves each exact intended test ran and passed. A kill requires the
test's own canonical/assertion failure, not compile/collection/fixture errors.
Sources are restored in finally. Use a separate Cargo target from positive gates.
"""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True, type=Path)
parser.add_argument("--logs", required=True, type=Path)
args = parser.parse_args()
tree = args.tree.resolve()
args.logs.mkdir(parents=True, exist_ok=True)
module = "crates/minion-agent/src/system_prompt.rs"
driver = "crates/minion-agent/src/agent_loop/driver.rs"

def control(name, witness, old, new, source=module, target="prompt_assembly_conformance"):
    return name, witness, old, new, source, target

controls = [
    control("metadata-alters-schema", "metadata_is_additive_and_uses_js_not_rust_whitespace", "description: self.description.clone(),", "description: self.prompt_snippet.clone().unwrap_or_else(|| self.description.clone()),", source="crates/minion-agent/src/tools/definition.rs", target="lib"),
    control("preflight-sanitizes-surrogates", "scalar_domain_preflight", "serde_json::from_str(text)", r'serde_json::from_str(&text.replace(r"\ud800", "x"))', source="crates/minion-agent/tests/prompt_assembly_conformance.rs"),
    control("disabled-visible", "canonical_skills_block", ".filter(|skill| !skill.disable_model_invocation)", ".filter(|_skill| true)"),
    control("ampersand-unescaped", "canonical_skills_block", "text.replace('&', \"&amp;\")", "text.to_owned()"),
    control("drive-root-separator-lost", "canonical_invocation", "Some(2) if units.get(1) == Some(&58) => 3,", "Some(2) if units.get(1) == Some(&58) => 2,"),
    control("scalar-drive-root-index", "dirname_uses_utf16_not_scalar_indices", "Some(2) if units.get(1) == Some(&58) => 3,", "Some(3) if units.get(2) == Some(&58) => 4,", target="lib"),
    control("empty-additional-appended", "canonical_invocation", "additional_instructions.filter(|text| !text.is_empty())", "additional_instructions"),
    control("invocation-escaped", "canonical_invocation", "        skill.content\n", "        escape_xml(&skill.content)\n"),
    control("rust-whitespace", "metadata_is_additive_and_uses_js_not_rust_whitespace", "js_trim(c.encode_utf8(&mut buffer)).is_empty()", "c.is_whitespace()", target="lib"),
    control("rust-guideline-trim", "canonical_tools_section", "let value = js_trim(value);", "let value = value.trim();"),
    control("duplicate-guidelines", "canonical_tools_section", "seen.insert(value.to_owned())", "{ seen.insert(value.to_owned()); true }"),
    control("tools-section-forced", "canonical_compose", "if configuration.tools_section {", "if true {"),
    control("read-gate-removed", "canonical_compose", 'tools.iter().any(|tool| tool.name() == "read")', "!tools.is_empty()"),
    control("contributed-lost", "canonical_compose", "sections.extend(configuration.sections.iter().cloned());", "sections.extend(std::iter::empty::<String>());"),
    control("empty-sections-kept", "canonical_compose", '.filter(|section| !section.is_empty())', '.filter(|_section| true)'),
    control("retained-record-copied", "wp142_composer_publishes_exact_prompt_and_retained_configuration", "configuration: RwLock::new(Arc::new(configuration.clone())),", "configuration: RwLock::new(Arc::new(PromptConfiguration { skills: configuration.skills.iter().map(|s| Arc::new(RwLock::new(s.read().clone()))).collect(), ..configuration.clone() })),", target="lib"),
    control("replacement-ignored", "wp142_composer_publishes_exact_prompt_and_retained_configuration", "*self.configuration.write() = Arc::new(configuration.clone());", "let _ = configuration;", target="lib"),
    control("live-registry-snapshot", "wp142_composer_follows_growth_and_replacement_not_live_registry", "assembler.assemble(&prepared.context.system_prompt, &prepared.context.tools)?", "assembler.assemble(&prepared.context.system_prompt, &self.agent.tools())?", source=driver, target="lib"),
]

originals = {source: (tree / source).read_text(encoding="utf-8") for *_, source, _ in controls}

def command(target, witness):
    if target == "lib":
        prefix = "system_prompt::tests::" if witness.startswith(("metadata_", "dirname_")) else "agent_loop::driver::tests::"
        return ["cargo", "test", "-p", "minion-agent", "--all-features", "--lib", prefix + witness, "--", "--exact", "--nocapture"], prefix + witness
    return ["cargo", "test", "-p", "minion-agent", "--all-features", "--test", target, witness, "--", "--exact", "--nocapture"], witness

def run(name, target, witness, mutant=False):
    cmd, qualified = command(target, witness)
    result = subprocess.run(cmd, cwd=tree, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / f"{name}-{'mutant' if mutant else 'baseline'}.log").write_text(result.stdout, encoding="utf-8")
    passed = result.returncode == 0 and f"test {qualified} ... ok" in result.stdout and "1 passed" in result.stdout
    signature = "unpaired surrogate must fail the document" if name == "preflight-sanitizes-surrogates" else "assertion"
    failed = result.returncode == 101 and f"test {qualified} ... FAILED" in result.stdout and "1 failed" in result.stdout and "panicked at" in result.stdout and signature in result.stdout
    if not (failed if mutant else passed):
        raise RuntimeError(f"INVALID {name}: exit {result.returncode}, intended witness {qualified}; see log")

try:
    for target, witness in sorted({(target, witness) for _, witness, *_, target in controls}):
        run(witness, target, witness)
    print("BASELINE: every intended witness selected and PASS", flush=True)
    for name, witness, old, new, source, target in controls:
        original = originals[source]
        if original.count(old) != 1 or old == new:
            raise RuntimeError(f"INVALID {name}: anchor count {original.count(old)}")
        try:
            (tree / source).write_text(original.replace(old, new, 1), encoding="utf-8")
            run(name, target, witness, True)
            print(f"KILLED {name} by {witness} (assertion)", flush=True)
        finally:
            (tree / source).write_text(original, encoding="utf-8")
finally:
    for source, original in originals.items():
        (tree / source).write_text(original, encoding="utf-8")
