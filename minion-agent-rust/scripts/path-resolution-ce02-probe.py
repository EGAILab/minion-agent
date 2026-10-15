"""Frozen CE02 Node operands/results through the real Rust native observer.

Only the NEW sandbox and five plain empty directories are created, each after
the lexical/no-follow/real containment proof. They are retained. Extended and
root cwd spellings are read-only child cwd arguments, never mutation targets.
Expected results are frozen arrays with only a fixture-prefix substitution;
this script contains no path resolver or native-view implementation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

TREE = Path(__file__).resolve().parents[1]
ROOT = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent")
spec = importlib.util.spec_from_file_location("probe", TREE / "scripts/path-resolution-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--focus", choices=("prefix", "roots", "entry-prefix", "entry-trim"))
    args = parser.parse_args()
    assert os.name == "nt", "Windows native partition only; POSIX has separate witnesses"
    binary = args.binary.resolve(strict=True)
    sandbox = probe.safety.checked(args.sandbox, ROOT)
    assert not sandbox.exists(), "NEW sandbox required; no cleanup"
    assert str(sandbox)[0].upper() == "E", "frozen drive authority is E"
    names = json.loads((TREE / "crates/minion-agent/tests/data/node_selected_value.json").read_text())["cwd_names"]
    # Prove every name's form before the first mutation.
    for row in names:
        name = probe.text(row["name"])
        assert name and name not in (".", "..") and not any(c in name for c in "/\\:\0")
    probe.safety.checked(sandbox, ROOT).mkdir()
    for row in names:
        child = sandbox / probe.text(row["name"])
        probe.safety.checked(child, ROOT).mkdir()
        assert child.resolve(strict=True) == child
    fixture = json.loads((TREE / "crates/minion-agent/tests/data/node_native_operand.json").read_text())

    def value(index):
        if index is None:
            return None
        out = []
        for part in fixture["values"][index]:
            if isinstance(part, list):
                out += part
            else:
                assert part in ({"fixture": "upper"}, {"fixture": "lower"})
                prefix = str(sandbox)
                prefix = (prefix[0].lower() if part["fixture"] == "lower" else prefix[0].upper()) + prefix[1:]
                out += probe.units(prefix)
        return out

    failures = []
    fired = set()
    report = []
    for row in fixture["native_rows"]:
        if args.focus == "prefix" and (row["cwd_class"] not in ("plain", "extended") or row["entry_class"] != "absent"):
            continue
        if args.focus == "roots" and (row["cwd_class"] not in ("drive-root", "extended-root") or row["entry_class"] != "absent"):
            continue
        if args.focus == "entry-prefix" and (row["cwd_class"] != "plain" or row["entry_class"] not in ("matching", "extended")):
            continue
        if args.focus == "entry-trim" and (row["cwd_class"] != "plain" or row["entry_class"] not in ("matching", "matching-trailing")):
            continue
        label = row["cwd_class"] + "/" + row["entry_class"]
        environment = {k: v for k, v in os.environ.items() if not k.startswith("=")}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        entry = value(row["entry"])
        if entry is not None:
            environment["=E:"] = probe.text(entry)
        environment["MINION_PATH_PROBE"] = json.dumps({"provider_cwd": probe.units("C:\\x"),
            "inputs": fixture["inputs"], "absent_entries": [], "process_cwd_drive_not": []})
        result = subprocess.run([str(binary), probe.TEST, "--exact", "--ignored", "--nocapture"],
            cwd=probe.text(value(row["cwd_arg"])), env=environment, capture_output=True)
        output = result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace")
        records = [json.loads(line[len("PATH_PROBE "):]) for line in output.splitlines() if line.startswith("PATH_PROBE ")]
        assert result.returncode == 0 and len(records) == 1, (label, result.returncode, output)
        actual = records[0]
        fired.update(line for line in output.splitlines() if line.startswith("CONTROL FIRED "))

        def check(seam, got, expected):
            if got != expected:
                failures.append({"id": label + "/" + seam, "actual": got, "expected": expected})

        check("raw", actual["cwd"], value(row["raw"]))
        check("cwd_view", actual["cwd_view"], value(row["cwd_view"]))
        check("entry_raw", dict(actual["entries"]), {"A": None, "E": entry, "X": None})
        check("entry_view", dict(actual["entry_views"]), {"A": None, "E": value(row["entry_view"]), "X": None})
        assert len(actual["rows"]) == len(row["expected"]) == 5
        for i, (got, expected) in enumerate(zip(actual["rows"], row["expected"], strict=True)):
            for seam in ("absolute_path", "logical"):
                check(seam + "/" + str(i), got[seam], value(expected))
        report.append({"id": label, "observed": actual})
    assert len(report) == (2 if args.focus else 196)
    print(json.dumps({"status": "FAIL" if failures else "MATCH", "states": len(report),
        "resolver_observations": len(report)*5, "failures": failures, "fired": sorted(fired),
        "rows": report}, ensure_ascii=True))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "INVALID", "type": type(error).__name__, "message": str(error)}, ensure_ascii=True))
        raise SystemExit(2) from None
