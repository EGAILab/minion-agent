"""Skill discovery and diagnostics (spec/harness.md WP-14.1: `HAR-001`, `HAR-010`..`HAR-013`).

A port of pinned Pi's harness loader (`packages/agent/src/harness/skills.ts`: `loadSkills`,
`loadSourcedSkills`, `loadSkillsFromDirInternal`, `loadSkillFromFile`, `resolveKind`), running over
the certified `ctx.fs` seam. The Owner-approved departures are:

- DIV-004 / PP-14-1: frontmatter YAML is the Minion subset (`_frontmatter`), and a declared skill
  whose frontmatter is outside it gets the Minion-defined `parse_failed` text;
- DIV-005: an entry whose ignore-check path is empty or starts with `/` gets one `invalid_path`
  diagnostic and is skipped, instead of rejecting the whole discovery;
- DIV-006: an ignore pattern whose RegExp pinned `ignore` rejects is dropped with one
  `invalid_ignore_pattern` diagnostic, instead of rejecting the whole discovery;
- PP-14-8: filesystem-origin diagnostics carry `FsError.message` (non-normative);
- skill order uses the `TOOL-040` pinned collator, compared raw (Owner-approved mapping).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from ..auth.js_json import js_trim
from ..execution import Err, FileSystem, FsErrorCode
from ..execution.filesystem import FileInfo, FileKind
from ..tools.builtin._utf16 import utf16_length
from ..tools.builtin.collation import pinned_collation
from ._frontmatter import FrontmatterError, extract, read_subset
from ._ignore import Ignore

MAX_NAME_LENGTH: Final = 64
MAX_DESCRIPTION_LENGTH: Final = 1024
IGNORE_FILE_NAMES: Final = (".gitignore", ".ignore", ".fdignore")
PARSE_FAILED_MESSAGE: Final = "frontmatter is not valid in the supported YAML subset"
INVALID_PATH_MESSAGE: Final = "entry path cannot be matched against ignore rules"
INVALID_IGNORE_PATTERN_MESSAGE: Final = "ignore pattern is not valid and was dropped"

type DiagnosticCode = Literal[
    "file_info_failed",
    "list_failed",
    "read_failed",
    "parse_failed",
    "invalid_metadata",
    "invalid_path",
    "invalid_ignore_pattern",
]


@dataclass(slots=True)
class Skill:
    """`HAR-013`: pinned Pi `Skill` (`types.ts`). Like every public record here it is writable, as
    Pi's are: `map_skill` receives the loaded record itself and may edit and return it
    (`WP141-R001`)."""

    name: str
    description: str
    content: str
    file_path: str
    """The addressed path `ctx.fs` listed; never canonicalized."""
    disable_model_invocation: bool


@dataclass(slots=True)
class SkillDiagnostic:
    """`HAR-013`: pinned Pi `SkillDiagnostic`."""

    code: DiagnosticCode
    message: str
    path: str
    type: Literal["warning"] = "warning"


@dataclass(slots=True)
class LoadedSkills:
    skills: list[Skill]
    diagnostics: list[SkillDiagnostic]


@dataclass(slots=True)
class SourcedSkill[TSkill, TSource]:
    skill: TSkill
    source: TSource


@dataclass(slots=True)
class SourcedSkillDiagnostic[TSource]:
    """A diagnostic with the input's opaque `source` attached (Pi `{...diagnostic, source}`)."""

    code: DiagnosticCode
    message: str
    path: str
    source: TSource
    type: Literal["warning"] = "warning"


@dataclass(slots=True)
class LoadedSourcedSkills[TSkill, TSource]:
    skills: list[SourcedSkill[TSkill, TSource]]
    diagnostics: list[SourcedSkillDiagnostic[TSource]]


_NAME: Final = re.compile(r"[a-z0-9-]+\Z")


async def load_skills(fs: FileSystem, roots: str | Sequence[str]) -> LoadedSkills:
    """Pi `loadSkills(env, dirs)`: roots in input order, results concatenated, never
    deduplicated."""
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    for root in [roots] if isinstance(roots, str) else list(roots):
        info = await fs.file_info(root)
        if isinstance(info, Err):
            if info.error.code != FsErrorCode.NOT_FOUND:
                diagnostics.append(SkillDiagnostic("file_info_failed", info.error.message, root))
            continue
        if await _resolve_kind(fs, info.value, diagnostics) != FileKind.DIRECTORY:
            continue
        await _walk(fs, info.value.path, True, Ignore(), info.value.path, skills, diagnostics)
    return LoadedSkills(skills, diagnostics)


async def load_sourced_skills[TSkill, TSource](
    fs: FileSystem,
    inputs: Sequence[tuple[str, TSource]],
    map_skill: Callable[[Skill, TSource], TSkill] | None = None,
) -> LoadedSourcedSkills[TSkill, TSource]:
    """Pi `loadSourcedSkills`: per `(path, source)` input, the source attached unchanged to every
    skill and diagnostic. A failure raised by `map_skill` propagates; it is not a diagnostic."""
    skills: list[SourcedSkill[TSkill, TSource]] = []
    diagnostics: list[SourcedSkillDiagnostic[TSource]] = []
    for path, source in inputs:
        result = await load_skills(fs, path)
        for skill in result.skills:
            mapped = map_skill(skill, source) if map_skill is not None else skill
            skills.append(SourcedSkill(mapped, source))  # type: ignore[arg-type]
        for d in result.diagnostics:
            diagnostics.append(SourcedSkillDiagnostic(d.code, d.message, d.path, source))
    return LoadedSourcedSkills(skills, diagnostics)


async def _resolve_kind(
    fs: FileSystem, info: FileInfo, diagnostics: list[SkillDiagnostic]
) -> FileKind | None:
    """Pi `resolveKind`: files and directories as they are; a symlink through `canonical_path`."""
    if info.kind in (FileKind.FILE, FileKind.DIRECTORY):
        return info.kind
    canonical = await fs.canonical_path(info.path)
    if isinstance(canonical, Err):
        if canonical.error.code != FsErrorCode.NOT_FOUND:
            diagnostics.append(
                SkillDiagnostic("file_info_failed", canonical.error.message, info.path)
            )
        return None
    target = await fs.file_info(canonical.value)
    if isinstance(target, Err):
        if target.error.code != FsErrorCode.NOT_FOUND:
            diagnostics.append(SkillDiagnostic("file_info_failed", target.error.message, info.path))
        return None
    return target.value.kind if target.value.kind in (FileKind.FILE, FileKind.DIRECTORY) else None


def _relative(root: str, path: str) -> str:
    """Pi `relativeEnvPath`: string-based, `\\` -> `/`, trailing `/` stripped."""
    r = root.replace("\\", "/").rstrip("/")
    p = path.replace("\\", "/").rstrip("/")
    if p == r:
        return ""
    return p[len(r) + 1 :] if p.startswith(f"{r}/") else p.lstrip("/")


def _invalid_ignore_path(path: str) -> bool:
    """DIV-005: the root-relative paths `ignore@7.0.5` refuses that the loader can produce."""
    return path == "" or path.startswith("/")


def _ignored(
    matcher: Ignore, check: str, entry_path: str, diagnostics: list[SkillDiagnostic]
) -> bool:
    """The ignore check, with the DIV-005 guard: an invalid path is reported and skipped."""
    if _invalid_ignore_path(check):
        diagnostics.append(SkillDiagnostic("invalid_path", INVALID_PATH_MESSAGE, entry_path))
        return True
    return matcher.ignores(check)


def _by_name(a: FileInfo, b: FileInfo) -> int:
    """The `TOOL-040` pinned collator, compared raw (no `ls` lowercase key). `sorted` is stable,
    so ties keep `list_dir` order."""
    return pinned_collation().compare(a.name, b.name)


@dataclass(slots=True)
class _Frame:
    """One directory being walked: its info, whether root `.md` files load, and its sorted
    children still to visit."""

    info: FileInfo
    include_root_files: bool
    children: Iterator[FileInfo]


async def _enter(
    fs: FileSystem,
    directory: str,
    include_root_files: bool,
    matcher: Ignore,
    root: str,
    skills: list[Skill],
    diagnostics: list[SkillDiagnostic],
) -> _Frame | None:
    """The start of Pi `loadSkillsFromDirInternal` for one directory: its own info and kind, its
    ignore files, its listing and the `SKILL.md` short-circuit. Returns the frame whose children
    are still to walk, or `None` when the directory is finished here."""
    info = await fs.file_info(directory)
    if isinstance(info, Err):
        if info.error.code != FsErrorCode.NOT_FOUND:
            diagnostics.append(SkillDiagnostic("file_info_failed", info.error.message, directory))
        return None
    dir_info = info.value
    if await _resolve_kind(fs, dir_info, diagnostics) != FileKind.DIRECTORY:
        return None
    await _add_ignore_rules(fs, matcher, directory, root, diagnostics)
    listing = await fs.list_dir(directory)
    if isinstance(listing, Err):
        diagnostics.append(SkillDiagnostic("list_failed", listing.error.message, directory))
        return None
    entries = listing.value
    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        if await _resolve_kind(fs, entry, diagnostics) != FileKind.FILE:
            continue
        if _ignored(matcher, _relative(root, entry.path), entry.path, diagnostics):
            continue
        await _load_file(fs, entry.path, dir_info.name, skills, diagnostics)
        return None
    children = iter(sorted(entries, key=functools.cmp_to_key(_by_name)))
    return _Frame(dir_info, include_root_files, children)


async def _walk(
    fs: FileSystem,
    directory: str,
    include_root_files: bool,
    matcher: Ignore,
    root: str,
    skills: list[Skill],
    diagnostics: list[SkillDiagnostic],
) -> None:
    """Pi `loadSkillsFromDirInternal`, appending in its depth-first emission order.

    Pi recurses into each child directory before its next sibling. Here an explicit stack of
    frames does the same, so the depth of a directory tree is bounded by the filesystem, never by
    the interpreter's call stack (`WP141-R004`): a child's frame is pushed and fully drained before
    its parent's iterator advances."""
    first = await _enter(fs, directory, include_root_files, matcher, root, skills, diagnostics)
    stack = [first] if first is not None else []
    while stack:
        frame = stack[-1]
        entry = next(frame.children, None)
        if entry is None:
            stack.pop()
            continue
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        kind = await _resolve_kind(fs, entry, diagnostics)
        if kind is None:
            continue
        rel = _relative(root, entry.path)
        if _ignored(
            matcher, f"{rel}/" if kind == FileKind.DIRECTORY else rel, entry.path, diagnostics
        ):
            continue
        if kind == FileKind.DIRECTORY:
            child = await _enter(fs, entry.path, False, matcher, root, skills, diagnostics)
            if child is not None:
                stack.append(child)
            continue
        if not frame.include_root_files or not entry.name.endswith(".md"):
            continue
        await _load_file(fs, entry.path, frame.info.name, skills, diagnostics)


async def _add_ignore_rules(
    fs: FileSystem, matcher: Ignore, directory: str, root: str, diagnostics: list[SkillDiagnostic]
) -> None:
    """Pi `addIgnoreRules` + `prefixIgnorePattern`."""
    relative_dir = _relative(root, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""
    for filename in IGNORE_FILE_NAMES:
        joined = await fs.join_path([directory, filename])
        if isinstance(joined, Err):
            diagnostics.append(SkillDiagnostic("file_info_failed", joined.error.message, directory))
            continue
        ignore_path = joined.value
        info = await fs.file_info(ignore_path)
        if isinstance(info, Err):
            if info.error.code != FsErrorCode.NOT_FOUND:
                diagnostics.append(
                    SkillDiagnostic("file_info_failed", info.error.message, ignore_path)
                )
            continue
        if info.value.kind != FileKind.FILE:
            continue
        content = await fs.read_text_file(ignore_path)
        if isinstance(content, Err):
            diagnostics.append(SkillDiagnostic("read_failed", content.error.message, ignore_path))
            continue
        patterns = [
            p for line in re.split(r"\r?\n", content.value) if (p := _prefix_pattern(line, prefix))
        ]
        # DIV-006: a pattern whose RegExp pinned `ignore` rejects is dropped with one diagnostic,
        # in line order, right after its ignore file is read; the valid ones keep their order
        for _ in matcher.add_valid(patterns):
            diagnostics.append(
                SkillDiagnostic(
                    "invalid_ignore_pattern", INVALID_IGNORE_PATTERN_MESSAGE, ignore_path
                )
            )


def _prefix_pattern(line: str, prefix: str) -> str | None:
    trimmed = js_trim(line)
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


async def _load_file(
    fs: FileSystem,
    file_path: str,
    parent_dir_name: str,
    skills: list[Skill],
    diagnostics: list[SkillDiagnostic],
) -> None:
    """Pi `loadSkillFromFile`, with the DIV-004 subset reader and the PP-14-1 message."""
    declared = re.split(r"[\\/]", re.sub(r"[\\/]+\Z", "", file_path))[-1] == "SKILL.md"
    raw = await fs.read_text_file(file_path)
    if isinstance(raw, Err):
        diagnostics.append(SkillDiagnostic("read_failed", raw.error.message, file_path))
        return
    extracted = extract(raw.value)
    try:
        frontmatter = {} if extracted.yaml is None else read_subset(extracted.yaml)
    except FrontmatterError:
        if declared:
            diagnostics.append(SkillDiagnostic("parse_failed", PARSE_FAILED_MESSAGE, file_path))
        return
    description = frontmatter.get("description")
    description = description if isinstance(description, str) else None
    if not declared and (description is None or js_trim(description) == ""):
        return
    for message in _validate_description(description):
        diagnostics.append(SkillDiagnostic("invalid_metadata", message, file_path))
    frontmatter_name = frontmatter.get("name")
    name = (
        frontmatter_name
        if isinstance(frontmatter_name, str) and frontmatter_name
        else parent_dir_name
    )
    for message in _validate_name(name, parent_dir_name):
        diagnostics.append(SkillDiagnostic("invalid_metadata", message, file_path))
    if description is None or js_trim(description) == "":
        return
    skills.append(
        Skill(
            name=name,
            description=description,
            content=extracted.body,
            file_path=file_path,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        )
    )


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    length = utf16_length(name)
    if length > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({length})")
    if not _NAME.match(name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    if description is None or js_trim(description) == "":
        return ["description is required"]
    length = utf16_length(description)
    if length > MAX_DESCRIPTION_LENGTH:
        return [f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({length})"]
    return []
