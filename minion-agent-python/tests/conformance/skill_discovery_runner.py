"""Runner for `skill_discovery` scenarios
(`conformance/schema/skill-discovery-scenario.schema.json`).

Thin by design: it writes the fixture under a fresh directory, calls the REAL `load_skills` over
the REAL `LocalFileSystem`, and maps the addressed paths in the result back to '/'-separated paths
relative to that directory for comparison. It never discovers, parses or validates anything
itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem, Ok
from minion_agent.skills import LoadedSkills, load_skills


def build_fixture(base: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        target = base.joinpath(*entry["path"].split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if "dir" in entry:
            target.mkdir(exist_ok=True)
        elif "symlink" in entry:
            # the target is relative to the link's own directory, as the scenario states
            os.symlink(
                entry["symlink"].replace("/", os.sep),
                target,
                target_is_directory=entry["symlink_kind"] == "dir",
            )
        else:
            target.write_bytes(entry["text"].encode("utf-8"))


async def run(base: Path, scenario: dict[str, Any]) -> tuple[LoadedSkills, str]:
    """Build the fixture, run discovery, and return the result with the addressed base path."""
    spec = scenario["skill_discovery"]
    build_fixture(base, spec["fixture"])
    fs = LocalFileSystem(str(base))
    addressed = await fs.absolute_path(str(base))
    assert isinstance(addressed, Ok)
    roots = [str(base.joinpath(*root.split("/"))) for root in spec["roots"]]
    return await load_skills(fs, roots), addressed.value


def relative(addressed_base: str, path: str) -> str:
    assert path.startswith(addressed_base), (path, addressed_base)
    # only the host separator is mapped: a POSIX name may itself contain a backslash (DIV-005 rows)
    return path[len(addressed_base) :].lstrip(os.sep).replace(os.sep, "/")


def observed(result: LoadedSkills, addressed_base: str) -> dict[str, Any]:
    """The result in the scenario's expectation shape (messages kept; filtered by the caller)."""
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "content": s.content,
                "path": relative(addressed_base, s.file_path),
                "disable_model_invocation": s.disable_model_invocation,
            }
            for s in result.skills
        ],
        "diagnostics": [
            {"code": d.code, "path": relative(addressed_base, d.path), "message": d.message}
            for d in result.diagnostics
        ],
    }


def diagnostic_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if "code_one_of" in expected:
        return actual["code"] in expected["code_one_of"] and actual["path"].startswith(
            expected["path_within"] + "/"
        )
    if (expected["code"], expected["path"]) != (actual["code"], actual["path"]):
        return False
    return "message" not in expected or expected["message"] == actual["message"]
