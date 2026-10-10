"""L0506-D005 source controls, only in a disposable scratch tree.

The baseline selects each exact intended witness and must pass. Compile/setup
failures and unrelated assertions are INVALID, never kills. No cycle is run.
"""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--tree", type=Path, required=True)
p.add_argument("--logs", type=Path, required=True)
a = p.parse_args()
tree = a.tree.resolve()
if "control" not in str(tree):
    raise RuntimeError("refusing non-control scratch tree")
execution = tree / "crates/minion-agent/src/tools/execution.rs"
prepared = tree / "crates/minion-agent/src/tools/prepared.rs"
original = {f: f.read_text(encoding="utf-8") for f in (execution, prepared)}

# Each edit is at the actual production seam, not in the adapter/expectations.
controls = [
    ("shallow-copy-restored", execution,
     "let params = params.structured_clone();",
     "let params = match params { PreparedValue::Object(o) => PreparedValue::Object(o.iter().collect()), other => other };",
     "reused_child_is_isolated_from_the_prepared_source", "clone isolates retained shim child"),
    ("clone-forgets-aliases", prepared,
     "if let Some(value) = memo.get(&id) {\n                        return value.clone();\n                    }",
     "if let Some(value) = memo.get(&id) {\n                        let _ = value;\n                    }",
     "alias_identity_survives_validation", "canonical isolation observation"),
    ("clone-per-listener", execution,
     "let future = listener(current.clone());",
     "let mut current = current;\n        current.call.arguments = current.call.arguments.structured_clone();\n        let future = listener(current.clone());",
     "one_graph_reaches_every_listener_and_execute", "one validated graph across listeners and execute"),
    ("json-round-trip-clone", execution,
     "let params = params.structured_clone();",
     "let params = params.try_to_json().map(PreparedValue::from).unwrap_or(PreparedValue::Null);",
     "runtime_values_survive_validation", "canonical isolation observation"),
    ("clone-reverses-ordinary-keys", prepared,
     "for (key, child) in object.iter() {\n                        out.insert(key, copy(&child, memo));\n                    }",
     "for (key, child) in object.iter().rev() {\n                        out.insert(key, copy(&child, memo));\n                    }",
     "enumeration_order_survives_validation", "canonical isolation observation"),
    ("updates-carry-validated-instead-of-raw", execution,
     "let original_arguments = arguments_value(&call.arguments);",
     "let original_arguments = crate::llm::RawValue::from(arguments.try_to_json().unwrap());",
     "raw_mutation_isolation_and_update_delivery", "canonical isolation observation"),
]
a.logs.mkdir(parents=True, exist_ok=True)


def run(label, witness, signature=None):
    r = subprocess.run(["cargo", "test", "--offline", "-p", "minion-agent", "--all-features",
                        "--test", "arg_isolation_conformance", witness, "--", "--exact", "--nocapture"],
                       cwd=tree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (a.logs / (label + ".log")).write_text(r.stdout, encoding="utf-8")
    if signature is None:
        good = r.returncode == 0 and f"test {witness} ... ok" in r.stdout and "1 passed" in r.stdout
    else:
        good = (r.returncode == 101 and f"test {witness} ... FAILED" in r.stdout
                and "1 failed" in r.stdout and "panicked at" in r.stdout and signature in r.stdout)
    if not good:
        raise RuntimeError(f"INVALID {label}: exit {r.returncode}; see log")


def baseline(label):
    r = subprocess.run(["cargo", "test", "--offline", "-p", "minion-agent", "--all-features",
                        "--test", "arg_isolation_conformance", "--", "--nocapture"], cwd=tree,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (a.logs / (label + ".log")).write_text(r.stdout, encoding="utf-8")
    if r.returncode or "8 passed" not in r.stdout or any(
        f"test {witness} ... ok" not in r.stdout for _, _, _, _, witness, _ in controls
    ):
        raise RuntimeError(f"INVALID {label}: every intended witness must be selected and pass")


try:
    baseline("baseline")
    print(f"BASELINE: all {len(controls)} intended witnesses selected and PASS", flush=True)
    for name, path, old, new, witness, signature in controls:
        for f, text in original.items():
            f.write_text(text, encoding="utf-8")
        # The memo guard occurs once per container kind, only within the clone.
        if name == "clone-forgets-aliases":
            text = original[path]
            head, clone = text.split("    pub fn structured_clone(&self) -> Self {", 1)
            clone, tail = clone.split("    /// Set a property", 1)
            if clone.count(old) != 2:
                raise RuntimeError("INVALID clone memo anchors")
            changed = head + "    pub fn structured_clone(&self) -> Self {" + clone.replace(old, new) + "    /// Set a property" + tail
        else:
            if original[path].count(old) != 1:
                raise RuntimeError(f"INVALID anchor {name}")
            changed = original[path].replace(old, new, 1)
        path.write_text(changed, encoding="utf-8")
        run(name, witness, signature)
        print(f"KILLED {name} by {witness}", flush=True)
finally:
    for f, text in original.items():
        f.write_text(text, encoding="utf-8")
    baseline("restored-baseline")
    print("RESTORED: sources and all intended baselines PASS", flush=True)
