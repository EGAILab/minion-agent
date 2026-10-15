"""L12-D012: raw child inputs -> real Rust seams -> frozen Node observations.

No canonical case path is ever accessed or used as an OS cwd. The only created
objects are a NEW, contained sandbox and five plain, empty cwd directories. They
are retained, never cleaned. stdout is the evidence; the caller guards its log.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

TREE = Path(__file__).resolve().parents[1]
ROOT = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent") if os.name == "nt" else Path("/tmp")
spec = importlib.util.spec_from_file_location("cache_safety", TREE / "scripts/cargo-cache-safety.py")
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)
TEST = "execution::path::tests::native_observer"


class WitnessFailure(Exception):
    def __init__(self, label, seam, actual, expected, fired=None):
        self.record = {"status": "FAIL", "id": label, "reason": seam,
                       "actual": actual, "expected": expected, "fired": fired or []}
        super().__init__(label)


def units(value):
    encoded = value.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(encoded[i:i + 2], "little") for i in range(0, len(encoded), 2)]


def text(value):
    return b"".join(u.to_bytes(2, "little") for u in value).decode("utf-16-le", "surrogatepass")


def fill(value, cwd):
    if "utf16" in value:
        return value["utf16"]
    output = []
    for part in value["parts"]:
        if "utf16" in part:
            output += part["utf16"]
        elif "cwd_drive" in part:
            drive = cwd[0].lower() if part["cwd_drive"] == "lower" else cwd[0].upper()
            output += units(drive)
        elif part == {"cwd_tail": True}:
            output += units(cwd[2:])
        else:
            raise ValueError(f"unknown template part {part!r}")
    return output


def invoke(binary, cwd, base, inputs, entries=None, preconditions=None, refuse=False, label="child"):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("=")}
    for key, value in (entries or {}).items():
        if value is not None:
            environment["=" + key + ":"] = text(value)
    pre = preconditions or {}
    environment["MINION_PATH_PROBE"] = json.dumps({
        "provider_cwd": base, "inputs": inputs,
        "absent_entries": pre.get("absent_entries", []),
        "process_cwd_drive_not": pre.get("process_cwd_drive_not", []),
    })
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([str(binary), TEST, "--exact", "--ignored", "--nocapture"],
                            cwd=cwd, env=environment, capture_output=True, timeout=90)
    output = result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace")
    records = [json.loads(line[len("PATH_PROBE "):]) for line in output.splitlines() if line.startswith("PATH_PROBE ")]
    fired = [line for line in output.splitlines() if line.startswith("CONTROL FIRED ")]
    if refuse:
        if result.returncode == 0 and records:
            raise WitnessFailure(label, "precondition", "SEAM REACHED", "REFUSED WITH ZERO OBSERVATIONS", fired)
        assert result.returncode == 101 and "PRECONDITION REFUSED" in output and not records, output
        return {"exit": result.returncode, "reason": "PRECONDITION REFUSED", "observations": 0}
    assert result.returncode == 0 and len(records) == 1, output
    record = records[0]
    record["control_fired"] = fired
    assert record["cwd"] == units(str(cwd)), (record["cwd"], units(str(cwd)))
    assert len(record["rows"]) == len(inputs)
    if os.name == "nt":
        seen = dict(record["entries"])
        for drive in ["A", "E", "X"]:
            assert seen[drive] == (entries or {}).get(drive), (drive, seen, entries)
    return record


def compare(label, record, expected):
    assert len(record["rows"]) == len(expected)
    for row, wanted in zip(record["rows"], expected, strict=True):
        for seam in ("absolute_path", "logical"):
            if row[seam] != wanted:
                raise WitnessFailure(label, seam, row[seam], wanted, record["control_fired"])
        if row["scalar"] is not None:
            # The scalar return crosses the certified OS projection boundary.
            projected = units(text(wanted).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace"))
            if row["scalar"] != projected:
                raise WitnessFailure(label, "scalar", row["scalar"], projected, record["control_fired"])
    return {"id": label, "status": "MATCH", "observed": record}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sandbox", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    sandbox = safety.checked(args.sandbox, ROOT)
    assert not sandbox.exists(), "a NEW sandbox is required; never overwrite or clean"
    safety.checked(sandbox, ROOT).mkdir()
    cwd = sandbox
    report = []
    canonical = json.loads((TREE.parent / "conformance/agent/path-resolution/path-resolution.json").read_text())
    platform = "win32" if os.name == "nt" else "linux"
    cases = [row for row in canonical["path_resolution"]["cases"] if row["platform"] == platform]
    assert len(cases) == (55 if os.name == "nt" else 25)
    for row in cases:
        host = row["host"]
        entries = {}
        if host:
            drive = str(cwd)[0].upper() if host["drive"] == "cwd" else host["drive"]
            entries[drive] = None if host["entry"] is None else fill(host["entry"], str(cwd))
        record = invoke(binary, cwd, fill(row["provider_cwd"], str(cwd)),
                        [fill(row["input"], str(cwd))], entries, row["preconditions"])
        report.append(compare(row["id"], record, [fill(row["expect"], str(cwd))]))
    if os.name == "nt":
        assert str(cwd)[0].upper() not in ("A", "C"), "fixture drive must be neither A nor C"
        fixture = json.loads((TREE / "crates/minion-agent/tests/data/node_selected_value.json").read_text())
        for row in fixture["entry_rows"]:
            record = invoke(binary, cwd, units("C:\\x"), [units("a:leaf")], {"A": row["entry"]})
            report.append(compare("entry/" + row["label"], record, [row["node"]]))
        for row in fixture["logical_rows"]:
            record = invoke(binary, cwd, units("C:\\x"), [row["logical"]], {"E": row["entry"]})
            report.append(compare("logical/" + row["label"], record, [row["node"]]))
        for row in fixture["cwd_names"]:
            name = text(row["name"])
            assert name and name not in (".", "..") and not any(c in name for c in "/\\:\0")
            child_cwd = sandbox / name
            safety.checked(child_cwd, ROOT).mkdir()
            assert child_cwd.resolve(strict=True) == child_cwd
            viewed = units(str(sandbox) + "\\") + row["node_view"]
            for variant, entry in [("absent", None), ("empty", []),
                                   ("entry", units(str(cwd)[0] + ":\\deep")), ("other-drive", None)]:
                drive = "A" if variant == "other-drive" else str(cwd)[0].upper()
                raw = drive.lower() + ":leaf"
                expected = units("a:\\leaf") if variant == "other-drive" else (
                    [ord(drive.lower())] + (entry[1:] if entry else viewed[1:]) + units("\\leaf"))
                record = invoke(binary, child_cwd, units("C:\\x"), [units(raw)], {drive: entry})
                report.append(compare("cwd/" + row["label"] + "/" + variant, record, [expected]))
        # Actual entry-point refusal observations, not a helper-only truth table.
        for label, entries, pre in [
            ("present-empty", {"A": []}, {"absent_entries": ["A"]}),
            ("present-value", {"A": units("A:\\deep")}, {"absent_entries": ["A"]}),
            ("cwd-drive", {}, {"process_cwd_drive_not": [str(cwd)[0].upper()]}),
        ]:
            report.append({"id": "refusal/" + label, "status": "REFUSED",
                           "observed": invoke(binary, cwd, units("C:\\x"), [units("a:leaf")], entries, pre, refuse=True, label="refusal/" + label)})
    print(json.dumps({"platform": platform, "canonical": len(cases), "rows": report}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except WitnessFailure as error:
        print(json.dumps(error.record, ensure_ascii=True))
        raise SystemExit(1) from None
    except Exception as error:
        print(json.dumps({"status": "INVALID", "type": type(error).__name__, "message": str(error)}, ensure_ascii=True))
        raise SystemExit(2) from None
