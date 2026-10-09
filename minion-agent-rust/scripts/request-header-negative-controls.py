"""L08-D002 controls, in an explicitly supplied disposable code worktree.

Use the normal pinned Rust/ICU environment. No pytest markers apply to Rust.
Each baseline must select and pass the intended test; compiler/infrastructure
errors and unrelated test failures are INVALID, never kills.
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
start = original.index("        prepared.agent.session().record_header(")
end = original.index("        let request_messages = transform_context(", start)
header = original[start:end]
transform_end = original.index("        let request = LlmRequest {", end)
controls = [
    ("missing-header", "request_header_single_request", header, ""),
    ("duplicate-header", "request_header_single_request", header, header + header),
    ("header-after-transform", "request_header_transform_first", original[start:transform_end], original[end:transform_end] + header),
    ("first-request-only", "request_header_request_order", header, "        if prepared.new_messages.len() <= 1 {\n" + header + "        }\n"),
    ("wrong-model", "request_header_single_request", "            prepared.config.model.model_id(),", '            "wrong-model",'),
    ("wrong-component", "request_header_single_request", '("system_base".to_owned(), system_prompt.clone())', '("wrong_component".to_owned(), system_prompt.clone())'),
    ("override-ignored", "request_header_literal_override", 'match &decision.system_override {', 'match &None::<String> {'),
    ("empty-header-tools", "request_header_full_schema_identity", "            schemas.clone(),", "            Vec::new(),"),
    ("reversed-header-tools", "request_header_full_schema_identity", "            schemas.clone(),", "            schemas.iter().rev().cloned().collect(),"),
    ("empty-provider-tools", "request_header_full_schema_identity", "                tools: Some(schemas),", "                tools: Some(Vec::new()),"),
    ("schema-failure-ignored", "agent_loop::driver::tests::request_header_schema_failure_precedes_publication_and_transform", ".collect::<Result<Vec<_>, _>>()?;", ".collect::<Result<Vec<_>, _>>().unwrap_or_default();"),
]


def run(name, witness, mutant):
    target = ["--lib"] if witness.startswith("agent_loop::") else ["--test", "agent_loop_conformance"]
    command = ["cargo", "test", "-p", "minion-agent", "--all-features", *target, witness, "--", "--exact", "--nocapture"]
    result = subprocess.run(command, cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / f"{name}-{'mutant' if mutant else 'baseline'}.log").write_text(result.stdout, encoding="utf-8")
    passed = f"test {witness} ... ok" in result.stdout
    signature = "assertion `left == right` failed" if name == "schema-failure-ignored" else "expect_"
    failed = f"test {witness} ... FAILED" in result.stdout and "panicked at" in result.stdout and signature in result.stdout
    valid = (result.returncode == 101 and failed and "1 failed" in result.stdout) if mutant else (result.returncode == 0 and passed and "1 passed" in result.stdout)
    if not valid:
        raise RuntimeError(f"INVALID {name}: exit {result.returncode}; see log")


try:
    # Validate the complete intended-witness selection before any mutation.
    for label, target in [("canonical", ["--test", "agent_loop_conformance"]), ("schema", ["--lib", "agent_loop::driver::tests::request_header_schema_failure_precedes_publication_and_transform", "--", "--exact"])]:
        baseline = subprocess.run(["cargo", "test", "-p", "minion-agent", "--all-features", *target], cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.logs / f"baseline-{label}.log").write_text(baseline.stdout, encoding="utf-8")
        witnesses = [witness for _, witness, _, _ in controls if witness.startswith("agent_loop::") == (label == "schema")]
        if baseline.returncode != 0 or any(f"test {witness} ... ok" not in baseline.stdout for witness in witnesses):
            raise RuntimeError(f"INVALID {label} baseline: exit {baseline.returncode}; every intended witness must pass")
    for name, witness, old, new in controls:
        if original.count(old) != 1:
            raise RuntimeError(f"INVALID {name}: anchor count {original.count(old)}")
        source.write_text(original, encoding="utf-8")
        source.write_text(original.replace(old, new, 1), encoding="utf-8")
        run(name, witness, True)
        print(f"KILLED {name} by {witness}", flush=True)
finally:
    source.write_text(original, encoding="utf-8")
