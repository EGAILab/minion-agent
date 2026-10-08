"""Focused WP-14.1 discovery tests: the filesystem-failure branches of pinned Pi's harness loader
(spec/harness.md WP-14.1 HAR-001/HAR-011/HAR-012; PP-14-8), sourced loading (HAR-013), and the
reader/matcher edges that the canonical scenarios cannot reach through a real filesystem."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Err, FsError, FsErrorCode, LocalFileSystem, Ok
from minion_agent.execution.filesystem import FileInfo, FileKind
from minion_agent.skills import (
    PARSE_FAILED_MESSAGE,
    Skill,
    SkillDiagnostic,
    load_skills,
    load_sourced_skills,
)
from minion_agent.skills._frontmatter import extract
from minion_agent.skills._ignore import Ignore, InvalidIgnorePattern, to_python

SKILL = "---\nname: {name}\ndescription: Use {name}.\n---\nBody."


class ScriptedFs:
    """The real local `ctx.fs`; `fail[(method, path)]` makes that call return the given error code,
    and `kind[path]` reports a different `FileInfo.kind` for that path."""

    def __init__(self, base: Path) -> None:
        self._local = LocalFileSystem(str(base))
        self.fail: dict[tuple[str, str], FsErrorCode] = {}
        self.kind: dict[str, FileKind] = {}

    def _err(self, method: str, path: str) -> Err[FsError] | None:
        code = self.fail.get((method, path))
        return Err(FsError(code, f"scripted {code.value}", path)) if code is not None else None

    def _patch(self, info: FileInfo) -> FileInfo:
        return replace(info, kind=self.kind[info.path]) if info.path in self.kind else info

    async def file_info(self, path: str, signal: Any = None) -> Any:
        if err := self._err("file_info", path):
            return err
        result = await self._local.file_info(path)
        return Ok(self._patch(result.value)) if isinstance(result, Ok) else result

    async def list_dir(self, path: str, signal: Any = None) -> Any:
        if err := self._err("list_dir", path):
            return err
        result = await self._local.list_dir(path)
        return Ok([self._patch(i) for i in result.value]) if isinstance(result, Ok) else result

    async def canonical_path(self, path: str, signal: Any = None) -> Any:
        return self._err("canonical_path", path) or await self._local.canonical_path(path)

    async def join_path(self, parts: Any, signal: Any = None) -> Any:
        return self._err("join_path", parts[0]) or await self._local.join_path(parts)

    async def read_text_file(self, path: str, signal: Any = None) -> Any:
        return self._err("read_text_file", path) or await self._local.read_text_file(path)


def _write(base: Path, rel: str, text: str) -> str:
    path = base.joinpath(*rel.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _codes(diagnostics: list[SkillDiagnostic]) -> list[tuple[str, str]]:
    return [(d.code, d.message) for d in diagnostics]


async def test_a_single_string_root_is_one_root(tmp_path: Path) -> None:
    _write(tmp_path, "skills/a/SKILL.md", SKILL.format(name="a"))
    result = await load_skills(LocalFileSystem(str(tmp_path)), str(tmp_path / "skills"))
    assert [s.name for s in result.skills] == ["a"]


async def test_root_file_info_failure_other_than_not_found_is_a_diagnostic(tmp_path: Path) -> None:
    fs = ScriptedFs(tmp_path)
    root = str(tmp_path / "skills")
    fs.fail[("file_info", root)] = FsErrorCode.PERMISSION_DENIED
    result = await load_skills(fs, [root])  # type: ignore[arg-type]
    assert result.diagnostics == [
        SkillDiagnostic("file_info_failed", "scripted permission_denied", root)
    ]


async def test_symlink_resolution_failures(tmp_path: Path) -> None:
    """`resolveKind`: canonical_path or the target's file_info failing (not_found is silent), and a
    target that is neither a file nor a directory."""
    fs = ScriptedFs(tmp_path)
    paths = {
        name: _write(tmp_path, f"skills/{name}/SKILL.md", SKILL.format(name=name))
        for name in "abcde"
    }
    for name in "abcde":
        fs.kind[str(Path(paths[name]).parent)] = FileKind.SYMLINK
    dirs = {name: str(Path(paths[name]).parent) for name in "abcde"}
    fs.fail[("canonical_path", dirs["a"])] = FsErrorCode.PERMISSION_DENIED
    fs.fail[("canonical_path", dirs["b"])] = FsErrorCode.NOT_FOUND
    canonical_c = (await fs.canonical_path(dirs["c"])).value
    canonical_d = (await fs.canonical_path(dirs["d"])).value
    canonical_e = (await fs.canonical_path(dirs["e"])).value
    fs.fail[("file_info", canonical_c)] = FsErrorCode.PERMISSION_DENIED
    fs.fail[("file_info", canonical_d)] = FsErrorCode.NOT_FOUND
    fs.kind[canonical_e] = FileKind.SYMLINK  # the resolved target is itself not a file or directory
    result = await load_skills(fs, [str(tmp_path / "skills")])  # type: ignore[arg-type]
    assert result.skills == []
    assert [(d.code, d.path) for d in result.diagnostics] == [
        ("file_info_failed", dirs["a"]),
        ("file_info_failed", dirs["c"]),
    ]


async def test_walk_failures(tmp_path: Path) -> None:
    """A walked directory whose file_info or listing fails; a vanished directory is silent."""
    fs = ScriptedFs(tmp_path)
    _write(tmp_path, "skills/a/SKILL.md", SKILL.format(name="a"))
    _write(tmp_path, "skills/b/SKILL.md", SKILL.format(name="b"))
    _write(tmp_path, "skills/c/SKILL.md", SKILL.format(name="c"))
    skills = tmp_path / "skills"
    root = str(skills)
    # the root's own walk re-reads file_info: fail it only on the second call via the child dirs
    fs.fail[("file_info", str(skills / "a"))] = FsErrorCode.PERMISSION_DENIED
    fs.fail[("list_dir", str(skills / "b"))] = FsErrorCode.PERMISSION_DENIED
    result = await load_skills(fs, [root])  # type: ignore[arg-type]
    assert [s.name for s in result.skills] == ["c"]
    # `a` fails its listing-time kind resolution (the entry's own FileInfo is from list_dir, so the
    # failure surfaces in the walk's re-read); `b` fails its listing
    assert ("list_failed", str(skills / "b")) in [(d.code, d.path) for d in result.diagnostics]


async def test_walked_directory_file_info_failures(tmp_path: Path) -> None:
    fs = ScriptedFs(tmp_path)
    _write(tmp_path, "one/x/SKILL.md", SKILL.format(name="x"))
    _write(tmp_path, "two/y/SKILL.md", SKILL.format(name="y"))
    one, two = tmp_path / "one", tmp_path / "two"
    calls = {"n": 0}
    original = fs.file_info

    async def file_info(path: str, signal: Any = None) -> Any:
        # the root passes the first (load_skills) check, then fails in the walk's own re-read
        if path in (str(one), str(two)):
            calls["n"] += 1
            if calls["n"] in (2, 4):
                code = FsErrorCode.PERMISSION_DENIED if path == str(one) else FsErrorCode.NOT_FOUND
                return Err(FsError(code, f"scripted {code.value}", path))
        return await original(path)

    fs.file_info = file_info  # type: ignore[method-assign]
    result = await load_skills(fs, [str(one), str(two)])  # type: ignore[arg-type]
    assert result.skills == []
    assert [(d.code, d.path) for d in result.diagnostics] == [("file_info_failed", str(one))]


async def test_walked_path_that_is_no_longer_a_directory_contributes_nothing(
    tmp_path: Path,
) -> None:
    fs = ScriptedFs(tmp_path)
    _write(tmp_path, "skills/a/SKILL.md", SKILL.format(name="a"))
    root = str(tmp_path / "skills")
    calls = {"n": 0}
    original = fs.file_info

    async def file_info(path: str, signal: Any = None) -> Any:
        result = await original(path)
        if path == root:
            calls["n"] += 1
            if calls["n"] == 2:
                return Ok(replace(result.value, kind=FileKind.FILE))
        return result

    fs.file_info = file_info  # type: ignore[method-assign]
    result = await load_skills(fs, [root])  # type: ignore[arg-type]
    assert result == type(result)([], [])


async def test_ignore_file_failures(tmp_path: Path) -> None:
    """join_path, file_info and read_text_file failures on the three ignore files."""
    fs = ScriptedFs(tmp_path)
    _write(tmp_path, "skills/a/SKILL.md", SKILL.format(name="a"))
    _write(tmp_path, "skills/.ignore", "zzz\n")
    skills = tmp_path / "skills"
    sub = skills / "a"
    fs.fail[("join_path", str(sub))] = FsErrorCode.INVALID
    fs.fail[("file_info", str(skills / ".gitignore"))] = FsErrorCode.PERMISSION_DENIED
    fs.fail[("read_text_file", str(skills / ".ignore"))] = FsErrorCode.PERMISSION_DENIED
    result = await load_skills(fs, [str(skills)])  # type: ignore[arg-type]
    assert [s.name for s in result.skills] == ["a"]
    assert [(d.code, d.path) for d in result.diagnostics] == [
        ("file_info_failed", str(skills / ".gitignore")),
        ("read_failed", str(skills / ".ignore")),
        ("file_info_failed", str(sub)),
        ("file_info_failed", str(sub)),
        ("file_info_failed", str(sub)),
    ]


async def test_skill_file_read_failure_is_a_diagnostic(tmp_path: Path) -> None:
    fs = ScriptedFs(tmp_path)
    path = _write(tmp_path, "skills/a/SKILL.md", SKILL.format(name="a"))
    fs.fail[("read_text_file", path)] = FsErrorCode.PERMISSION_DENIED
    result = await load_skills(fs, [str(tmp_path / "skills")])  # type: ignore[arg-type]
    assert result.skills == []
    assert result.diagnostics == [
        SkillDiagnostic("read_failed", "scripted permission_denied", path)
    ]


async def test_sourced_loading_attaches_the_source_and_maps_skills(tmp_path: Path) -> None:
    _write(tmp_path, "one/a/SKILL.md", SKILL.format(name="a"))
    _write(tmp_path, "two/Bad/SKILL.md", "---\ndescription: d\n---\n")
    fs = LocalFileSystem(str(tmp_path))
    inputs = [
        (str(tmp_path / "one"), {"scope": "user"}),
        (str(tmp_path / "two"), {"scope": "project"}),
    ]
    plain = await load_sourced_skills(fs, inputs)
    assert [(s.skill.name, s.source) for s in plain.skills] == [
        ("a", {"scope": "user"}),
        ("Bad", {"scope": "project"}),
    ]
    assert {d.source["scope"] for d in plain.diagnostics} == {"project"}
    mapped = await load_sourced_skills(
        fs, inputs, lambda skill, source: (skill.name, source["scope"])
    )
    assert [s.skill for s in mapped.skills] == [("a", "user"), ("Bad", "project")]

    def boom(skill: Skill, source: object) -> object:
        raise RuntimeError("application failure")

    with pytest.raises(RuntimeError, match="application failure"):
        await load_sourced_skills(fs, inputs, boom)


def test_extraction_slices_code_units_like_pi() -> None:
    """`---` followed by an astral character: Pi's slice(4) starts mid-pair, so T begins with a
    lone low surrogate -- which the subset then refuses (rule 1)."""
    extracted = extract("---\U0001f600\nname: x\n---\nBody")
    assert extracted.yaml is not None and extracted.yaml.startswith("\ude00")
    assert extract("no frontmatter").yaml is None
    assert extract("---\nunclosed").yaml is None


def test_a_regex_source_ending_in_a_backslash_is_invalid_as_in_javascript() -> None:
    with pytest.raises(InvalidIgnorePattern):
        to_python("a\\")


def test_an_invalid_pattern_raises_when_first_evaluated() -> None:
    matcher = Ignore()
    matcher.add(["[~-a]"])
    with pytest.raises(InvalidIgnorePattern):
        matcher.ignores("x")


def test_a_three_digit_octal_escape_above_0o377_is_two_digits_then_a_literal() -> None:
    """Annex B (verified in Node v22.15.1): `\\477` is `\\47` (an apostrophe) followed by `7`."""
    from minion_agent.skills._ignore import canonicalize

    pattern = to_python("\\477")
    assert pattern.search(canonicalize("'7"))
    assert not pattern.search(canonicalize(chr(0o477)))


async def test_a_skill_md_directory_does_not_short_circuit(tmp_path: Path) -> None:
    """A `SKILL.md` that is a directory is not a declared skill file; it is walked like any
    other child."""
    _write(tmp_path, "skills/a/SKILL.md/x/SKILL.md", SKILL.format(name="x"))
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(tmp_path / "skills")])
    assert [s.name for s in result.skills] == ["x"]


class _ListingFs(ScriptedFs):
    """Adds one synthetic listing entry to a directory: a name the host may not be able to
    create."""

    def __init__(self, base: Path, directory: str, extra: FileInfo) -> None:
        super().__init__(base)
        self._directory = directory
        self._extra = extra

    async def list_dir(self, path: str, signal: Any = None) -> Any:
        result = await super().list_dir(path)
        if path == self._directory and isinstance(result, Ok):
            return Ok([*result.value, self._extra])
        return result


async def test_div005_invalid_entry_is_reported_and_skipped_while_discovery_continues(
    tmp_path: Path,
) -> None:
    """DIV-005, independent of the host: an entry whose root-relative path starts with `/` (a POSIX
    `\\x.md` after Pi's backslash-to-slash conversion) gets one invalid_path diagnostic and is
    skipped; the skills before and after it are kept."""
    _write(tmp_path, "skills/-a/SKILL.md", "---\ndescription: Before.\n---\n")
    _write(tmp_path, "skills/z/SKILL.md", SKILL.format(name="z"))
    root = str(tmp_path / "skills")
    sep = "\\" if "\\" in root else "/"
    odd = FileInfo(name="\\x.md", path=f"{root}{sep}\\x.md", kind=FileKind.FILE, size=0, mtime_ms=0)
    result = await load_skills(_ListingFs(tmp_path, root, odd), [root])  # type: ignore[arg-type]
    assert [s.name for s in result.skills] == ["-a", "z"]
    invalid = [d for d in result.diagnostics if d.code == "invalid_path"]
    assert invalid == [
        SkillDiagnostic(
            "invalid_path", "entry path cannot be matched against ignore rules", odd.path
        )
    ]


# ---- DIV-006: invalid ignore patterns are dropped with one diagnostic each ----


def test_add_valid_keeps_valid_patterns_in_order_and_returns_the_invalid_ones() -> None:
    matcher = Ignore()
    rejected = matcher.add_valid(["a*", "[~-a]", "", "# comment", "!ab", "x[ab/c", "[~-a]", "b"])
    assert rejected == ["[~-a]", "x[ab/c", "[~-a]"]  # line order, duplicates kept
    # the valid rules behave exactly as the same valid patterns added with Pi's `add`
    reference = Ignore()
    reference.add(["a*", "!ab", "b"])
    for path in ["a", "ab", "abc", "b", "c", "x/a", "x/ab"]:
        assert matcher.ignores(path) == reference.ignores(path), path


def test_add_valid_with_only_valid_patterns_rejects_nothing() -> None:
    matcher = Ignore()
    assert matcher.add_valid(["build/", "*.md", "!keep.md"]) == []
    assert matcher.ignores("build/") and matcher.ignores("x.md") and not matcher.ignores("keep.md")


def test_add_valid_with_only_invalid_patterns_adds_no_rule() -> None:
    matcher = Ignore()
    assert matcher.add_valid(["[~-a]", "![z-!]"]) == ["[~-a]", "![z-!]"]
    assert not matcher.ignores("anything")


async def test_invalid_patterns_are_reported_per_pattern_in_file_order(tmp_path: Path) -> None:
    """Mixed: diagnostics follow the ignore-file order (.gitignore before .ignore) and line order;
    a negated invalid pattern is reported too; the valid patterns from both files still apply."""
    _write(tmp_path, "skills/.gitignore", "drop\n[~-a]\n!keep[~-!]\n")
    _write(tmp_path, "skills/.ignore", "x[ab/c\nother*\n")
    for name in ["drop", "other1", "keep", "z"]:
        _write(tmp_path, f"skills/{name}/SKILL.md", SKILL.format(name=name))
    skills = tmp_path / "skills"
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(skills)])
    assert [s.name for s in result.skills] == ["keep", "z"]
    assert [(d.code, d.path) for d in result.diagnostics] == [
        ("invalid_ignore_pattern", str(skills / ".gitignore")),
        ("invalid_ignore_pattern", str(skills / ".gitignore")),
        ("invalid_ignore_pattern", str(skills / ".ignore")),
    ]
    assert {d.message for d in result.diagnostics} == {
        "ignore pattern is not valid and was dropped"
    }


async def test_an_all_valid_ignore_file_emits_no_diagnostic(tmp_path: Path) -> None:
    _write(tmp_path, "skills/.gitignore", "drop\n# c\n\nother*\n")
    for name in ["drop", "other1", "keep"]:
        _write(tmp_path, f"skills/{name}/SKILL.md", SKILL.format(name=name))
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(tmp_path / "skills")])
    assert [s.name for s in result.skills] == ["keep"]
    assert result.diagnostics == []


# ---- WP141-R001: the public records are writable, as Pi's are ----


async def test_map_skill_may_edit_the_loaded_skill_and_return_it(tmp_path: Path) -> None:
    """Pi passes the loaded `Skill` itself to `mapSkill`; an ordinary mapper edits it in place and
    returns it. The very object comes back, edited, with the opaque source untouched."""
    _write(tmp_path, "one/a/SKILL.md", SKILL.format(name="a"))
    source = object()
    seen: list[Skill] = []

    def mapping(skill: Skill, src: object) -> Skill:
        seen.append(skill)
        skill.name = "mapped"
        skill.description = "edited"
        return skill

    result = await load_sourced_skills(
        LocalFileSystem(str(tmp_path)), [(str(tmp_path / "one"), source)], mapping
    )
    assert len(result.skills) == 1
    assert result.skills[0].skill is seen[0]
    assert (result.skills[0].skill.name, result.skills[0].skill.description) == ("mapped", "edited")
    assert result.skills[0].source is source


async def test_loaded_records_and_their_lists_are_writable(tmp_path: Path) -> None:
    _write(tmp_path, "s/a/SKILL.md", SKILL.format(name="a"))
    _write(tmp_path, "s/B/SKILL.md", "---\ndescription: d\n---\n")
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(tmp_path / "s")])
    result.skills[0].disable_model_invocation = True
    result.diagnostics[0].message = "replaced"
    result.skills.append(result.skills[0])
    assert result.skills[0].disable_model_invocation is True
    assert result.diagnostics[0].message == "replaced"
    assert len(result.skills) == 3


# ---- DIV-007 (WP141-R003) and WP141-R002: nesting depth ----


def _nested(name: str, depth: int, *, seq: bool = False) -> str:
    """The root mapping is depth 1; depth-1 chained `k:` entries reach a mapping (or, with `seq`, a
    sequence) at exactly `depth` (DIV-007's counting)."""
    chain = "".join("  " * i + "k:\n" for i in range(depth - 1))
    leaf = "  " * (depth - 1) + ("- item\n" if seq else "leaf: value\n")
    return f"---\nname: {name}\ndescription: Example.\n{chain}{leaf}---\nBody."


@pytest.mark.parametrize(("depth", "seq"), [(63, False), (64, False), (64, True)])
async def test_nesting_up_to_64_loads(tmp_path: Path, depth: int, seq: bool) -> None:
    _write(tmp_path, "s/n/SKILL.md", _nested("n", depth, seq=seq))
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(tmp_path / "s")])
    assert [s.name for s in result.skills] == ["n"]
    assert result.diagnostics == []


@pytest.mark.parametrize(("depth", "seq"), [(65, False), (65, True), (100, False), (500, False)])
async def test_nesting_deeper_than_64_is_parse_failed(
    tmp_path: Path, depth: int, seq: bool
) -> None:
    """DIV-007: beyond 64 levels the file is outside the subset. Pinned Pi accepts 65, 100 and 500
    (canonical n02-n04 record that as divergence evidence); Minion's bound is normative."""
    path = _write(tmp_path, "s/n/SKILL.md", _nested("n", depth, seq=seq))
    _write(tmp_path, "s/ok/SKILL.md", SKILL.format(name="ok"))
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(tmp_path / "s")])
    assert [s.name for s in result.skills] == ["ok"]
    assert result.diagnostics == [SkillDiagnostic("parse_failed", PARSE_FAILED_MESSAGE, path)]


async def test_stack_exhaustion_is_still_contained_without_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP141-R002, kept as defence in depth: with the DIV-007 bound lifted, a nesting deep enough
    to exhaust the reader's stack is still one `parse_failed`, never an escaping RecursionError."""
    import minion_agent.skills._frontmatter as frontmatter

    monkeypatch.setattr(frontmatter, "MAX_DEPTH", 10**9)
    bad = _write(tmp_path, "bad/bad/SKILL.md", _nested("bad", 1200))
    _write(tmp_path, "good/good/SKILL.md", SKILL.format(name="good"))
    result = await load_skills(
        LocalFileSystem(str(tmp_path)), [str(tmp_path / "bad"), str(tmp_path / "good")]
    )
    assert [s.name for s in result.skills] == ["good"]
    assert result.diagnostics == [SkillDiagnostic("parse_failed", PARSE_FAILED_MESSAGE, bad)]


async def test_deep_nesting_is_one_parse_failed_and_later_roots_still_load(tmp_path: Path) -> None:
    """Pinned Pi (yaml@2.9.0) contains its parser's failure at depth 1200 as one `parse_failed` and
    keeps loading; here DIV-007's bound rejects it first, with the same outcome."""
    bad = _write(tmp_path, "bad/bad/SKILL.md", _nested("bad", 1200))
    _write(tmp_path, "good/good/SKILL.md", SKILL.format(name="good"))
    result = await load_skills(
        LocalFileSystem(str(tmp_path)), [str(tmp_path / "bad"), str(tmp_path / "good")]
    )
    assert [s.name for s in result.skills] == ["good"]
    assert result.diagnostics == [SkillDiagnostic("parse_failed", PARSE_FAILED_MESSAGE, bad)]


async def test_deep_nesting_in_an_undeclared_root_file_is_skipped_silently(tmp_path: Path) -> None:
    _write(tmp_path, "root/deep.md", _nested("deep", 1200))
    _write(tmp_path, "root/ok/SKILL.md", SKILL.format(name="ok"))
    result = await load_skills(LocalFileSystem(str(tmp_path)), [str(tmp_path / "root")])
    assert [s.name for s in result.skills] == ["ok"]
    assert result.diagnostics == []
