"""Generate the WP-13.3 `bash` canonical scenarios (`conformance/agent/builtin-bash/*.yaml`) from
the pinned-Pi authority outputs -- no expectation is hand-written.

    python scripts/wp133_bash_scenarios.py <minion-agent-docs>/assurance/layers/data/13-wp133/out

Inputs are `pi-win32.json` (Windows 11, Git Bash) and `pi-linux.json` (`node:22.15.1-bookworm-slim`,
/bin/bash) from `harness/bash_probe.mjs`, which runs pinned Pi's own `bash.ts` `execute`. Each case
becomes one scenario with one expectation per platform.

Two deliberate mappings, both stated in each scenario's notes:
- `abort/before-spawn`: the probe called `execute` with an aborted signal (`Command aborted`);
  through the real Layer 06 pipeline a pre-aborted call never reaches `execute` and is answered
  `Operation aborted` (spec WP-13.3 step 3, `L13-WP132-R004`).
- `updates/two-chunks`: only the final result is certified (Owner Q2: no partial updates).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "conformance" / "agent" / "builtin-bash"
AUTHORITY = (
    "minion-agent-docs spec/tools.md WP-13.3 (master a805f6ed); pinned-Pi bash_probe outputs"
)
PI_REVISION = "b7bb00b936dbe21b8e160b3e89efdec361846699"
TOOL_034_PREFIXES = ("exit/", "timeout/", "abort/", "cwd/", "kill/")

CASE_OPTIONS: dict[str, dict[str, Any]] = {
    "abort/before-spawn": {"signal": "pre_aborted"},
    "abort/during": {"abort_after_ms": 400},
    "cwd/missing": {"missing_cwd": True},
}
NOTES: dict[str, str] = {
    "abort/before-spawn": (
        "The pinned probe called execute directly with an aborted signal (Command aborted). Through"
        " the real Layer 06 pipeline a pre-aborted call never reaches execute: preflight answers"
        " 'Operation aborted' (spec WP-13.3 step 3, L13-WP132-R004)."
    ),
    "updates/two-chunks": "Final result only: no partial updates are certified (Owner Q2).",
    "timeout/invalid-infinity": (
        "The pinned probe called execute directly (Invalid timeout: must be a finite number of"
        " seconds). Through the real Layer 06 pipeline the declared number rejects Infinity at"
        " validation first (L0506-D001); the validator's text is binding-specific (TOOL-003). The"
        " tool's own message is a binding witness of resolve_timeout_ms."
    ),
    "kill/external-sigkill": (
        "Platform fact (characterization section 11): Linux reports no exit code (success, no"
        " output); Windows Git Bash reports exit 2304."
    ),
}


def _units(value: list[int]) -> str:
    return b"".join(struct.pack("<H", unit) for unit in value).decode("utf-16-le", "surrogatepass")


def _normalize_cwd(text: str) -> str:
    marker = "Working directory does not exist: "
    if text.startswith(marker):
        rest = text[len(marker) :]
        return marker + "<CWD>" + rest[rest.index("\n") :]
    return text


def _expectation(case_id: str, result: dict[str, Any]) -> dict[str, Any]:
    text = result["text"]
    expect: dict[str, Any] = {"is_error": result["is_error"]}
    if case_id == "abort/before-spawn":
        expect["text"] = "Operation aborted"
    elif case_id == "timeout/invalid-infinity":
        expect["validation_rejected"] = True
    elif isinstance(text, dict):
        expect["text_length"] = text["length"]
        expect["text_head"] = _units(text["head"])
        expect["text_tail"] = _units(text["tail"])
    else:
        expect["text"] = _normalize_cwd(_units(text))
    details = result.get("details")
    if details:
        expect["details"] = {"truncation": details["truncation"], "fullOutputPath": "<FULL_OUTPUT>"}
    else:
        expect["details"] = {}
    full = result.get("full_output")
    expect["full_output"] = {"size": full["size"], "sha256": full["sha256"]} if full else None
    return expect


def main(out_dir: Path) -> int:
    platforms = {
        name: {
            r["id"]: r
            for r in json.loads((out_dir / f"pi-{name}.json").read_text("utf-8"))["results"]
        }
        for name in ("win32", "linux")
    }
    written = 0
    for case_id, win in platforms["win32"].items():
        linux = platforms["linux"][case_id]
        assert win["command"] == linux["command"] and win["timeout"] == linux["timeout"], case_id
        slug = case_id.replace("/", "-")
        spec: dict[str, Any] = {"command": win["command"]}
        if win["timeout"] is not None:
            spec["timeout"] = win["timeout"]
        spec.update(CASE_OPTIONS.get(case_id, {}))
        spec["expect"] = {
            "win32": _expectation(case_id, win),
            "linux": _expectation(case_id, linux),
        }
        requirement = "TOOL-034" if case_id.startswith(TOOL_034_PREFIXES) else "TOOL-035"
        document: dict[str, Any] = {
            "name": f"builtin-bash-{slug}",
            "family": "agent",
            "authority": AUTHORITY,
            "pi_revision": PI_REVISION,
            "requirements": [requirement],
            "witnesses": [f"bash_{slug.replace('-', '_')}"],
        }
        if case_id in NOTES:
            document["notes"] = NOTES[case_id]
        document["builtin_bash"] = spec
        path = OUT_DIR / f"builtin-bash-{slug}.yaml"
        path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=False, width=100),
            encoding="utf-8",
        )
        written += 1
    print(f"wrote {written} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
