"""L08-D001 source controls; --tree MUST be a disposable Rust tree.

Every control first proves its exact intended witness passes. Selection errors,
compiler errors, and unrelated failures are INVALID, not kills. Restore source
in finally; use the positive gate's Cargo and pinned ICU environment.
"""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
args = parser.parse_args()
tree = args.tree.resolve()
source = tree / "crates/minion-agent/src/agent_loop/driver.rs"
original = source.read_text(encoding="utf-8")
args.logs.mkdir(parents=True, exist_ok=True)
call = "assembler.assemble(&prepared.context.system_prompt, &prepared.context.tools)?"
start = original.index("        let system_prompt = match &decision.system_override {")
end = original.index("        // ToolSchema owns", start)
assembly = original[start:end]
transform = "        let request_messages = transform_context("
controls = [
    ("assembler-not-called", "prompt_assembler_uses_snapshot_order_without_mutating_base", call, "prepared.context.system_prompt.clone()"),
    ("live-registry", "prompt_assembler_uses_snapshot_order_without_mutating_base", call, "assembler.assemble(&prepared.context.system_prompt, &self.agent.tools())?"),
    ("reversed-tools", "prompt_assembler_uses_snapshot_order_without_mutating_base", call, "assembler.assemble(&prepared.context.system_prompt, &prepared.context.tools.iter().rev().cloned().collect::<Vec<_>>())?"),
    ("wrong-base", "prompt_assembler_seven_requests_follow_context_snapshots_not_registry_churn", call, 'assembler.assemble("wrong-base", &prepared.context.tools)?'),
    ("stale-first-result", "prompt_assembler_seven_requests_follow_context_snapshots_not_registry_churn", call, '{ static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new(); CACHED.get_or_init(|| assembler.assemble(&prepared.context.system_prompt, &prepared.context.tools).unwrap()).clone() }'),
    ("override-ignored", "prompt_assembler_empty_override_bypasses_callback", "match &decision.system_override {", "match &None::<String> {"),
    ("empty-override-treated-absent", "prompt_assembler_empty_override_bypasses_callback", "Some(text) => text.clone(),", "Some(text) if !text.is_empty() => text.clone(),\n            Some(_) => self.prompt_assembler.as_ref().unwrap().assemble(&prepared.context.system_prompt, &prepared.context.tools)?,"),
    ("growth-ignored", "added_tool_names_extend_only_the_run_local_snapshot_in_order", call, "assembler.assemble(&prepared.context.system_prompt, &prepared.context.tools[..1])?"),
]
# Keep the provider prompt assembled but publish the unassembled header first.
late = original.replace(assembly, "        let system_prompt = prepared.context.system_prompt.clone();\n", 1)
late = late.replace(transform, assembly + transform, 1)
controls.append(("assembly-after-header", "prompt_assembler_later_failure_keeps_only_the_prior_header", original, late))


def run(name, witness, mutant):
    qualified = "agent_loop::driver::tests::" + witness
    command = ["cargo", "test", "-p", "minion-agent", "--all-features", "--lib", qualified, "--", "--exact", "--nocapture"]
    result = subprocess.run(command, cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / f"{name}-{'mutant' if mutant else 'baseline'}.log").write_text(result.stdout, encoding="utf-8")
    passed = result.returncode == 0 and f"test {qualified} ... ok" in result.stdout and "1 passed" in result.stdout
    signature = "must not run" if name in {"override-ignored", "empty-override-treated-absent"} else "assertion"
    failed = result.returncode == 101 and f"test {qualified} ... FAILED" in result.stdout and "1 failed" in result.stdout and "panicked at" in result.stdout and signature in result.stdout
    if not (failed if mutant else passed):
        raise RuntimeError(f"INVALID {name}: exit {result.returncode}; see log")


try:
    # One unmutated batch proves every intended witness was selected and passed.
    # Rebuilding the same baseline between mutants adds cost, not discrimination.
    baseline = subprocess.run(["cargo", "test", "-p", "minion-agent", "--all-features", "--lib", "--", "--nocapture"], cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / "baseline-all.log").write_text(baseline.stdout, encoding="utf-8")
    if baseline.returncode != 0 or any(f"test agent_loop::driver::tests::{witness} ... ok" not in baseline.stdout for _, witness, _, _ in controls):
        raise RuntimeError(f"INVALID baseline: exit {baseline.returncode}; intended witnesses must all pass")
    for name, witness, old, new in controls:
        if original.count(old) != 1 or old == new:
            raise RuntimeError(f"INVALID {name}: anchor count {original.count(old)}")
        source.write_text(original, encoding="utf-8")
        source.write_text(original.replace(old, new, 1), encoding="utf-8")
        run(name, witness, True)
        print(f"KILLED {name} by {witness}", flush=True)
finally:
    source.write_text(original, encoding="utf-8")
